"""
TranslateGemma 4B (MLX 4-bit) — o'zbek lotiniga tarjima.

Model: mlx-community/translategemma-4b-it-4bit
Papka: models/tarjima_model/TranslateGemma/
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from backend.config import (
    LLM_MAX_NEW_TOKENS_TRANSLATE,
    TRANSLATE_CHUNK_CHARS,
    TRANSLATE_TEMPERATURE,
    TRANSLATEGEMMA_DIR,
    TRANSLATION_MODELS,
    detect_device,
    empty_accelerator_cache,
    find_translation_model_dir,
)
from backend.services.nllb_service import resolve_src_lang
from backend.services.translation_text import (
    clean_translation_output,
    iter_document_units,
    join_document_units,
)

logger = logging.getLogger(__name__)

TGT_CODE = "uz"
TGT_NAME = "Uzbek (Latin script)"

_FLORES_TO_BCP: dict[str, tuple[str, str]] = {
    "uzn_Latn": ("uz", "Uzbek"),
    "uzn_Cyrl": ("uz", "Uzbek"),
    "eng_Latn": ("en", "English"),
    "rus_Cyrl": ("ru", "Russian"),
    "kaz_Cyrl": ("kk", "Kazakh"),
    "kaz_Latn": ("kk", "Kazakh"),
    "kir_Cyrl": ("ky", "Kyrgyz"),
    "tgk_Cyrl": ("tg", "Tajik"),
    "tuk_Latn": ("tk", "Turkmen"),
    "tur_Latn": ("tr", "Turkish"),
    "arb_Arab": ("ar", "Arabic"),
    "zho_Hans": ("zh", "Chinese"),
    "deu_Latn": ("de", "German"),
    "fra_Latn": ("fr", "French"),
    "spa_Latn": ("es", "Spanish"),
    "pes_Arab": ("fa", "Persian"),
    "pbt_Arab": ("ps", "Pashto"),
}


class TranslateGemmaService:
    """TranslateGemma ni bir marta yuklab, xotirada ushlab turadi."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-tg")
        self._device = detect_device()
        self._model = None
        self._tokenizer = None
        self._load_error: str | None = None

    @property
    def is_ready(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def status(self) -> dict[str, Any]:
        spec = TRANSLATION_MODELS["translategemma"]
        found = find_translation_model_dir(spec)
        return {
            "ready": self.is_ready,
            "backend": "mlx" if self.is_ready else None,
            "model_key": "translategemma",
            "device": self._device,
            "model_dir": str(found or TRANSLATEGEMMA_DIR),
            "target": TGT_CODE,
            "error": self._load_error,
        }

    def ensure_loaded(self) -> None:
        if self.is_ready:
            return
        with self._lock:
            if self.is_ready:
                return
            self._device = detect_device()
            self._load_error = None
            try:
                self._pool.submit(self._load).result()
            except Exception as exc:
                self._load_error = str(exc)[:800]
                raise

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._tokenizer = None
            empty_accelerator_cache()

    def translate(
        self,
        text: str,
        src_lang: str = "auto",
        tgt_lang: str = TGT_CODE,
        country: str = "",
    ) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        self.ensure_loaded()
        units = iter_document_units(text, min(500, TRANSLATE_CHUNK_CHARS))
        if not units:
            return ""
        src_flores = resolve_src_lang(text, src_lang, country)
        src_code, src_name = _FLORES_TO_BCP.get(src_flores, ("auto", "the source language"))
        if src_code == TGT_CODE and src_flores == "uzn_Latn":
            return text
        translated: list[tuple[str, str]] = []
        for idx, (chunk, sep) in enumerate(units, start=1):
            logger.info("TranslateGemma %s/%s  src=%s", idx, len(units), src_code)
            piece = clean_translation_output(
                self._translate_chunk(chunk, src_code, src_name)
            )
            translated.append((piece or chunk, sep))
        return join_document_units(translated)

    def _load(self) -> None:
        spec = TRANSLATION_MODELS["translategemma"]
        folder = find_translation_model_dir(spec)
        if folder is None:
            raise FileNotFoundError(
                "TranslateGemma topilmadi. "
                "`scripts/download_models.py translategemma` ni ishga tushiring."
            )
        from mlx_lm import load

        logger.info("TranslateGemma yuklanmoqda: %s", folder)
        self._model, self._tokenizer = load(str(folder))
        logger.info("TranslateGemma yuklandi (qurilma=%s)", self._device)

    def _translate_chunk(self, text: str, src_code: str, src_name: str) -> str:
        def _run() -> str:
            from mlx_lm import generate

            if self._model is None or self._tokenizer is None:
                raise RuntimeError("TranslateGemma yuklanmagan")
            tokenizer = self._tokenizer
            prompt = self._build_prompt(tokenizer, text, src_code, src_name)
            gen_kwargs: dict[str, Any] = {
                "max_tokens": LLM_MAX_NEW_TOKENS_TRANSLATE,
                "verbose": False,
            }
            try:
                from mlx_lm.sample_utils import make_sampler

                gen_kwargs["sampler"] = make_sampler(temp=TRANSLATE_TEMPERATURE)
            except Exception:
                pass
            return generate(self._model, tokenizer, prompt=prompt, **gen_kwargs)

        return self._pool.submit(_run).result()

    def _build_prompt(self, tokenizer: Any, text: str, src_code: str, src_name: str) -> str:
        structured = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": src_code if src_code != "auto" else "auto",
                        "target_lang_code": TGT_CODE,
                        "text": text,
                    }
                ],
            }
        ]
        try:
            return tokenizer.apply_chat_template(
                structured,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
        plain = [
            {
                "role": "user",
                "content": (
                    f"Translate from {src_name} to {TGT_NAME}. "
                    "Output only the translation. No quotes, no markdown.\n\n"
                    f"{text}"
                ),
            }
        ]
        try:
            return tokenizer.apply_chat_template(
                plain, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return (
                f"Translate from {src_name} to {TGT_NAME}. "
                f"Output only the translation.\n\n{text}\n"
            )


translategemma_service = TranslateGemmaService()
