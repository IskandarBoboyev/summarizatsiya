"""Asosiy ilova NLLB mikroserverini chaqiradi — model shu jarayonda yuklanmaydi."""

from __future__ import annotations

from typing import Any

from backend.config import MODEL_RPC_TIMEOUT, TRANSLATION_SERVER_URL, USE_MODEL_SERVERS
from backend.services.nllb_service import TGT_UZBEK_LATIN
from backend.services.rpc import rpc_json


class NllbRpc:
    """NLLB worker (port 8003) uchun HTTP mijoz."""

    def status(self) -> dict[str, Any]:
        if not USE_MODEL_SERVERS:
            from backend.services.nllb_service import nllb_service

            return {**nllb_service.status(), "mode": "in-process"}
        try:
            data = rpc_json(f"{TRANSLATION_SERVER_URL}/status", method="GET", timeout=5)
            data["mode"] = "microservice"
            data["url"] = TRANSLATION_SERVER_URL
            return data
        except Exception as exc:
            return {
                "ready": False,
                "backend": None,
                "model_key": "nllb-200",
                "mode": "microservice",
                "url": TRANSLATION_SERVER_URL,
                "error": str(exc),
            }

    def translate(
        self,
        text: str,
        src_lang: str = "auto",
        tgt_lang: str = TGT_UZBEK_LATIN,
        country: str = "",
    ) -> str:
        if not USE_MODEL_SERVERS:
            from backend.services.nllb_service import nllb_service

            return nllb_service.translate(text, src_lang=src_lang, tgt_lang=tgt_lang, country=country)
        data = rpc_json(
            f"{TRANSLATION_SERVER_URL}/translate",
            {
                "text": text,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "country": country,
            },
            timeout=MODEL_RPC_TIMEOUT,
        )
        return str(data.get("translation") or "")


nllb_rpc = NllbRpc()
