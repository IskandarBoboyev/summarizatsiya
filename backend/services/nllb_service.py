"""
Offline NLLB-200 tarjima servisi.

Model: facebook/nllb-200-distilled-1.3B
Papka: models/tarjima_model/

Matn o'zbek lotiniga (uzn_Latn) tarjima qilinadi. Manba tili
avtomatik aniqlanadi yoki ISO / FLORES kodi bilan beriladi.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any

from backend.config import (
    NLLB_DIR,
    TRANSLATE_CHUNK_CHARS,
    detect_device,
    empty_accelerator_cache,
    find_translation_model_dir,
    TRANSLATION_MODELS,
)
from backend.services.translation_text import (
    clean_translation_output,
    iter_document_units,
    join_document_units,
)

logger = logging.getLogger(__name__)

# FLORES-200 kodlari (NLLB tokenizer)
TGT_UZBEK_LATIN = "uzn_Latn"
NLLB_CHUNK_CHARS = min(600, TRANSLATE_CHUNK_CHARS)

ISO_TO_NLLB: dict[str, str] = {
    "uz": "uzn_Latn",
    "ru": "rus_Cyrl",
    "en": "eng_Latn",
    "kk": "kaz_Cyrl",
    "ky": "kir_Cyrl",
    "tg": "tgk_Cyrl",
    "tr": "tur_Latn",
    "ar": "arb_Arab",
    "zh": "zho_Hans",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "tk": "tuk_Latn",
    "fa": "pes_Arab",
    "ps": "pbt_Arab",
}

COUNTRY_ISO_SRC: dict[str, str] = {
    "tj": "tg",
    "kz": "kk",
    "kg": "ky",
    "tm": "tk",
    "af": "fa",
}


def src_lang_for_country(country: str, asr_language: str = "") -> str:
    """Davlat yoki ASR tilidan tarjima manbasi."""
    asr = (asr_language or "").strip().lower()
    if asr and asr not in {"auto", "uz"}:
        return asr
    return COUNTRY_ISO_SRC.get((country or "").strip().lower(), "auto")


COUNTRY_TO_NLLB: dict[str, dict[str, str]] = {
    "uz": {"latn": "uzn_Latn", "cyrl": "uzn_Cyrl"},
    "kz": {"latn": "kaz_Latn", "cyrl": "kaz_Cyrl"},
    "kg": {"latn": "kir_Cyrl", "cyrl": "kir_Cyrl"},
    "tj": {"latn": "tgk_Cyrl", "cyrl": "tgk_Cyrl"},
    "tm": {"latn": "tuk_Latn", "cyrl": "tuk_Latn"},
    "af": {"arab": "pes_Arab", "latn": "eng_Latn"},
}

_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_CJK_RE = re.compile(r"[\u4E00-\u9FFF]")


class NllbService:
    """NLLB-200 ni bir marta yuklab, xotirada ushlab turadi."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._device = detect_device()
        self._model = None
        self._tokenizer = None
        self._load_error: str | None = None

    @property
    def is_ready(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def status(self) -> dict[str, Any]:
        spec = TRANSLATION_MODELS["nllb-200"]
        found = find_translation_model_dir(spec)
        return {
            "ready": self.is_ready,
            "backend": "nllb" if self.is_ready else None,
            "model_key": "nllb-200",
            "device": self._device,
            "model_dir": str(found or NLLB_DIR),
            "target": TGT_UZBEK_LATIN,
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
                self._load()
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
        tgt_lang: str = TGT_UZBEK_LATIN,
        country: str = "",
    ) -> str:
        """Matnni o'zbek lotiniga (yoki berilgan tilga) tarjima qiladi."""
        text = (text or "").strip()
        if not text:
            return ""
        self.ensure_loaded()
        units = iter_document_units(text, NLLB_CHUNK_CHARS)
        if not units:
            return ""
        src = resolve_src_lang(text, src_lang, country)
        if src == tgt_lang:
            return text
        translated: list[tuple[str, str]] = []
        for idx, (chunk, sep) in enumerate(units, start=1):
            logger.info("NLLB tarjima %s/%s  src=%s", idx, len(units), src)
            piece = clean_translation_output(self._translate_chunk(chunk, src, tgt_lang))
            translated.append((piece or chunk, sep))
        return join_document_units(translated)

    def _load(self) -> None:
        spec = TRANSLATION_MODELS["nllb-200"]
        folder = find_translation_model_dir(spec)
        if folder is None:
            raise FileNotFoundError(
                "NLLB-200 topilmadi. Modelni `models/tarjima_model/` papkasiga "
                "(config.json + og'irliklar) joylang yoki "
                "`scripts/download_models.py nllb` ni ishga tushiring."
            )

        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info("NLLB yuklanmoqda: %s", folder)
        tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True)
        use_fp16 = self._device in {"cuda", "mps"}
        dtype = torch.float16 if use_fp16 else torch.float32
        model = AutoModelForSeq2SeqLM.from_pretrained(
            str(folder),
            local_files_only=True,
            torch_dtype=dtype,
        )
        model.to(self._device)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        logger.info("NLLB yuklandi (qurilma=%s, dtype=%s)", self._device, dtype)

    def _translate_chunk(self, text: str, src_lang: str, tgt_lang: str) -> str:
        import torch

        tokenizer = self._tokenizer
        model = self._model
        assert tokenizer is not None and model is not None

        self._set_src_lang(tokenizer, src_lang)
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=480)
        encoded = {k: v.to(self._device) for k, v in encoded.items()}
        bos = _lang_token_id(tokenizer, tgt_lang)
        inp_len = int(encoded["input_ids"].shape[-1])
        max_new = min(512, max(64, int(inp_len * 1.7)))
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                forced_bos_token_id=bos,
                max_new_tokens=max_new,
                num_beams=1,
                do_sample=False,
            )
        return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()

    @staticmethod
    def _set_src_lang(tokenizer: Any, src_lang: str) -> None:
        if hasattr(tokenizer, "src_lang"):
            tokenizer.src_lang = src_lang
        setter = getattr(tokenizer, "set_src_lang_special_tokens", None)
        if callable(setter):
            setter(src_lang)


def _lang_token_id(tokenizer: Any, code: str) -> int:
    mapping = getattr(tokenizer, "lang_code_to_id", None)
    if isinstance(mapping, dict) and code in mapping:
        return int(mapping[code])
    tid = tokenizer.convert_tokens_to_ids(code)
    unk = getattr(tokenizer, "unk_token_id", None)
    if tid is not None and tid != unk:
        return int(tid)
    raise RuntimeError(f"NLLB til kodi noma'lum: {code}")


def resolve_src_lang(text: str, src_lang: str = "auto", country: str = "") -> str:
    """ISO / FLORES / auto → NLLB FLORES kodi."""
    raw = (src_lang or "auto").strip()
    if raw and raw != "auto":
        if "_" in raw:
            return raw
        mapped = ISO_TO_NLLB.get(raw.lower())
        if mapped:
            if mapped.endswith("_Latn") and _cyrillic_share(text) > 0.4:
                if raw.lower() == "uz":
                    return "uzn_Cyrl"
            return mapped
    return detect_src_lang(text, country)


def detect_src_lang(text: str, country: str = "") -> str:
    """Markaziy Osiyo tillari uchun sodda skript/harf aniqlash."""
    sample = text[:4000]
    n_cyr = len(_CYRILLIC_RE.findall(sample))
    n_lat = len(_LATIN_RE.findall(sample))
    n_arab = len(_ARABIC_RE.findall(sample))
    n_cjk = len(_CJK_RE.findall(sample))
    letters = max(n_cyr + n_lat + n_arab + n_cjk, 1)
    country = (country or "").lower()

    if n_cjk / letters > 0.25:
        return "zho_Hans"
    if n_arab / letters > 0.25:
        hint = COUNTRY_TO_NLLB.get(country, {}).get("arab")
        return hint or "pes_Arab" if country == "af" else (hint or "arb_Arab")
    if n_cyr >= n_lat:
        return _detect_cyrillic(sample, country)
    return _detect_latin(sample, country)


def _cyrillic_share(text: str) -> float:
    sample = text[:2000]
    n_cyr = len(_CYRILLIC_RE.findall(sample))
    n_lat = len(_LATIN_RE.findall(sample))
    total = n_cyr + n_lat
    return n_cyr / total if total else 0.0


def _detect_cyrillic(sample: str, country: str) -> str:
    hint = COUNTRY_TO_NLLB.get(country, {}).get("cyrl")
    low = sample.lower()
    if any(ch in low for ch in "әұ"):
        return "kaz_Cyrl"
    if any(ch in low for ch in "ғқҳҷӣӯ"):
        return "tgk_Cyrl"
    if "ў" in low:
        return "uzn_Cyrl"
    if "ң" in low and "ө" in low:
        return hint or "kir_Cyrl"
    if hint:
        return hint
    return "rus_Cyrl"


def _detect_latin(sample: str, country: str) -> str:
    hint = COUNTRY_TO_NLLB.get(country, {}).get("latn")
    low = sample.lower()
    if "o‘" in low or "g‘" in low or "o'" in low or "g'" in low or "oʻ" in low:
        return "uzn_Latn"
    if any(ch in low for ch in "ğıışçöü"):
        return "tur_Latn"
    if hint:
        return hint
    return "eng_Latn"


nllb_service = NllbService()
