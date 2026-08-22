"""
Hujjatlarni qayta ishlash quvuri (pipeline).

Navbatdan (SQLite status=pending) fayl olinadi va:
    1. Audio/video  -> ASR transkripsiya -> original_text
    2. Hujjat/rasm  -> OCR/parser        -> original_text
    3. Matn bo'lsa  -> Gemma xulosa + NLLB o'zbekcha tarjima

Ish sinxron (CPU/GPU), FastAPI event loop ni bloklamaslik uchun
thread pool da ishlaydi. Bir vaqtda faqat bitta fayl — VRAM/RAM tejash.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

from backend.config import (
    AUDIO_EXTENSIONS,
    IMAGE_EXTENSIONS,
    TRANSLATE_CHUNK_CHARS,
    VIDEO_EXTENSIONS,
    empty_accelerator_cache,
)
from backend.database import (
    folder_asr_model_for_country,
    get_all_settings,
    get_document,
    list_pending_documents,
    public_document,
    queue_counts,
    update_document,
)
from backend.services.asr_rpc import asr_rpc
from backend.services.llm_rpc import llm_rpc
from backend.services.ocr_parser import ocr_parser
from backend.services.nllb_service import src_lang_for_country
from backend.services.translation_rpc import translate_text
from backend.services.translation_text import (
    clean_summary_output,
    clean_translation_output,
    iter_document_units,
    join_document_units,
    looks_like_cyrillic,
    looks_like_english,
)

logger = logging.getLogger(__name__)

NotifyFn = Callable[[str, dict], None]


class DocumentPipeline:
    """
    Bir oqimli ishlovchi.

    `notify(event, payload)` — WebSocket orqali UI ni yangilash uchun.
    """

    def __init__(self) -> None:
        self._busy = threading.Lock()
        self._notify: NotifyFn | None = None

    def set_notify(self, callback: NotifyFn | None) -> None:
        """UI ga hodisa yuborish funksiyasini bog'laydi."""
        self._notify = callback

    def process_next(self) -> bool:
        """
        Navbatdagi birinchi pending faylni ishlaydi.

        Returns:
            True — biror fayl olindi (muvaffaqiyat yoki xato).
            False — navbat bo'sh.
        """
        pending = list_pending_documents()
        if not pending:
            return False
        doc = pending[0]
        self.process_document(int(doc["id"]))
        return True

    def process_document(self, doc_id: int) -> dict | None:
        """
        Bitta hujjatni to'liq qayta ishlaydi (OCR/ASR + LLM).

        Bir vaqtda faqat bitta chaqiriq — GPU xotirasi uchun muhim.
        """
        with self._busy:
            doc = get_document(doc_id)
            if doc is None:
                return None

            settings = get_all_settings()
            fallback = settings.get("llm_model", "gemma-4")
            summary_key = settings.get("llm_summary_model") or fallback
            translate_key = settings.get("llm_translate_model") or "nllb-200"
            asr_lang = settings.get("asr_language") or "auto"
            global_asr = settings.get("asr_model") or "gigaam-multilingual"
            folder_asr = folder_asr_model_for_country(str(doc.get("country") or ""))
            asr_model = folder_asr or global_asr
            quant = settings.get("quantization", "auto")
            ocr_model = settings.get("ocr_model") or "surya"

            update_document(
                doc_id,
                status="processing",
                pipeline_stage="extract",
                error_message="",
            )
            self._emit("document_updated", get_document(doc_id))
            self._emit("queue_status", queue_counts())

            try:
                result = self._run(
                    doc, summary_key, translate_key, asr_lang, quant, ocr_model, asr_model
                )
                update_document(
                    doc_id,
                    **result,
                    status="done",
                    pipeline_stage="done",
                    error_message="",
                )
                logger.info("Tayyor: %s", doc.get("filename"))
            except Exception as exc:
                logger.exception("Qayta ishlash xatosi (%s): %s", doc.get("filename"), exc)
                update_document(
                    doc_id,
                    status="error",
                    pipeline_stage="error",
                    error_message=str(exc)[:2000],
                )
            finally:
                empty_accelerator_cache()

            fresh = get_document(doc_id)
            self._emit("document_updated", fresh)
            self._emit("queue_status", queue_counts())
            return public_document(fresh)

    def _run(
        self,
        doc: dict,
        summary_key: str,
        translate_key: str,
        asr_lang: str,
        quant: str,
        ocr_model: str = "surya",
        asr_model: str = "gigaam-multilingual",
    ) -> dict:
        """
        Fayl turiga qarab OCR yoki ASR, so'ng Gemma xulosa + NLLB tarjima.

        Returns:
            Bazaga yoziladigan maydonlar (status siz).
        """
        path = Path(doc["filepath"])
        if not path.is_file():
            raise FileNotFoundError(f"Fayl yo'qolgan: {path}")

        suffix = path.suffix.lower()
        is_media = suffix in AUDIO_EXTENSIONS or suffix in VIDEO_EXTENSIONS

        original = ""
        transcription = ""

        if is_media:
            transcription = asr_rpc.transcribe(
                path,
                language=asr_lang,
                model_key=asr_model,
                country=str(doc.get("country") or ""),
            )
            original = transcription
        else:
            original = ocr_parser.extract(path, engine=ocr_model)
            # Rasm bo'lsa va matn bo'sh — foydalanuvchiga tushunarli xabar
            if not original.strip() and suffix in IMAGE_EXTENSIONS:
                raise RuntimeError(
                    "Rasmdan matn ajratilmadi. Sozlamada OCR modelni tekshiring "
                    "(Surya yoki Gemma). Rasmda aniq matn borligini ham ko‘ring."
                )

        # LLM xato bersa ham asl matn UI da qolsin
        update_document(
            int(doc["id"]),
            original_text=original,
            transcription=transcription,
            asr_language=asr_lang if is_media else "",
            pipeline_stage="summarize" if original.strip() else "extract",
        )
        self._emit("document_updated", get_document(int(doc["id"])))

        summary = ""
        translation = ""
        model_used = ""

        if original.strip():
            summary = self._finalize_summary(
                llm_rpc.summarize(original, summary_key, quant),
                country=str(doc.get("country") or ""),
                translate_key=translate_key,
            )
            update_document(
                int(doc["id"]),
                summary=summary,
                pipeline_stage="translate",
            )
            self._emit("document_updated", get_document(int(doc["id"])))
            translation = self._translate_document(
                original,
                country=str(doc.get("country") or ""),
                doc_id=int(doc["id"]),
                translate_key=translate_key,
                src_lang=src_lang_for_country(
                    str(doc.get("country") or ""),
                    asr_lang if is_media else "",
                ),
            )
            model_used = (
                f"{asr_model} + {summary_key} + {translate_key}"
                if is_media
                else f"{summary_key} + {translate_key}"
            )
        elif is_media:
            raise RuntimeError("Transkripsiya bo'sh. ASR modelini tekshiring.")

        return {
            "original_text": original,
            "transcription": transcription,
            "summary": summary,
            "translation_uz": translation,
            "model_used": model_used,
            "asr_language": asr_lang if is_media else "",
        }

    def _finalize_summary(self, summary: str, country: str, translate_key: str = "nllb-200") -> str:
        """Xulosani tozalaydi; inglizcha/kirill chiqsa tanlangan model bilan lotinga o'giradi."""
        text = clean_summary_output(summary)
        if not text:
            return ""
        src = ""
        if looks_like_english(text):
            src = "eng_Latn"
        elif looks_like_cyrillic(text):
            src = src_lang_for_country(country) or "auto"
        if src:
            logger.info("Xulosa %s — %s orqali o'zbek lotiniga o'giriladi", src, translate_key)
            text = translate_text(text, src_lang=src, country=country, model_key=translate_key)
            text = clean_summary_output(text)
        return text

    def _translate_document(
        self,
        original: str,
        country: str,
        doc_id: int,
        translate_key: str = "nllb-200",
        src_lang: str = "auto",
    ) -> str:
        """
        Asl hujjatni to'liq, paragraf-paragraf o'zbek lotiniga o'giradi.

        Xulosa ishlatilmaydi. Bo'sh qatorlar, ro'yxat va adabiyotlar saqlanadi.
        """
        units = iter_document_units(original, TRANSLATE_CHUNK_CHARS)
        if not units:
            return ""
        done: list[tuple[str, str]] = []
        for idx, (chunk, sep) in enumerate(units, start=1):
            piece = translate_text(
                chunk,
                src_lang=src_lang or "auto",
                country=country,
                model_key=translate_key,
            )
            done.append((clean_translation_output(piece) or chunk, sep))
            if idx == 1 or idx % 5 == 0 or idx == len(units):
                update_document(
                    doc_id,
                    translation_uz=join_document_units(done),
                    pipeline_stage="translate",
                )
                self._emit("document_updated", get_document(doc_id))
                logger.info("Tarjima %s/%s", idx, len(units))
        return clean_translation_output(join_document_units(done))

    def _emit(self, event: str, payload: dict | None) -> None:
        """Xavfsiz notify — xato UI ni to'xtatmasin."""
        if self._notify is None or payload is None:
            return
        try:
            if event.startswith("document"):
                payload = public_document(payload) or payload
            self._notify(event, payload)
        except Exception:
            logger.exception("notify xatosi: %s", event)


pipeline = DocumentPipeline()
