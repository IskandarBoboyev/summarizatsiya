"""Tanlangan tarjima modeliga (NLLB yoki TranslateGemma) yo'naltiradi."""

from __future__ import annotations

from backend.services.nllb_rpc import nllb_rpc
from backend.services.nllb_service import TGT_UZBEK_LATIN
from backend.services.translategemma_rpc import translategemma_rpc


def translate_text(
    text: str,
    *,
    src_lang: str = "auto",
    country: str = "",
    model_key: str = "nllb-200",
) -> str:
    """Sozlamadagi model bilan o'zbek lotiniga tarjima qiladi."""
    key = (model_key or "nllb-200").strip().lower()
    if key.startswith("translategemma"):
        return translategemma_rpc.translate(text, src_lang=src_lang, country=country)
    return nllb_rpc.translate(
        text,
        src_lang=src_lang,
        tgt_lang=TGT_UZBEK_LATIN,
        country=country,
    )
