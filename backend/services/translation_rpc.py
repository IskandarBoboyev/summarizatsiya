"""Tanlangan tarjima modeliga (NLLB yoki TranslateGemma) yo'naltiradi."""

from __future__ import annotations

from backend.services.nllb_rpc import nllb_rpc
from backend.services.nllb_service import TGT_UZBEK_LATIN, resolve_src_lang
from backend.services.translategemma_rpc import translategemma_rpc
from backend.services.translation_text import looks_like_cyrillic

# TranslateGemma 4B tojik/qozoq kirillini lotin o'zbekchaga zaif o'giradi — NLLB aniqroq.
_NLLB_PREFERRED_SRC = {
    "tgk_Cyrl",
    "kaz_Cyrl",
    "kaz_Latn",
    "kir_Cyrl",
    "tuk_Latn",
    "rus_Cyrl",
    "pes_Arab",
    "pbt_Arab",
}


def translate_text(
    text: str,
    *,
    src_lang: str = "auto",
    country: str = "",
    model_key: str = "nllb-200",
) -> str:
    """Sozlamadagi model bilan o'zbek lotiniga tarjima qiladi."""
    key = (model_key or "nllb-200").strip().lower()
    src = resolve_src_lang(text, src_lang, country)
    use_gemma = key.startswith("translategemma") and src not in _NLLB_PREFERRED_SRC
    if use_gemma:
        out = translategemma_rpc.translate(text, src_lang=src_lang, country=country)
        if out and not looks_like_cyrillic(out):
            return out
    return nllb_rpc.translate(
        text,
        src_lang=src,
        tgt_lang=TGT_UZBEK_LATIN,
        country=country,
    )
