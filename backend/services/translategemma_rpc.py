"""Asosiy ilova TranslateGemma mikroserverini chaqiradi."""

from __future__ import annotations

from typing import Any

from backend.config import MODEL_RPC_TIMEOUT, TRANSLATEGEMMA_SERVER_URL, USE_MODEL_SERVERS
from backend.services.rpc import rpc_json
from backend.services.translategemma_service import TGT_CODE


class TranslateGemmaRpc:
    """TranslateGemma worker (port 8005) uchun HTTP mijoz."""

    def status(self) -> dict[str, Any]:
        if not USE_MODEL_SERVERS:
            from backend.services.translategemma_service import translategemma_service

            return {**translategemma_service.status(), "mode": "in-process"}
        try:
            data = rpc_json(f"{TRANSLATEGEMMA_SERVER_URL}/status", method="GET", timeout=5)
            data["mode"] = "microservice"
            data["url"] = TRANSLATEGEMMA_SERVER_URL
            return data
        except Exception as exc:
            return {
                "ready": False,
                "backend": None,
                "model_key": "translategemma",
                "mode": "microservice",
                "url": TRANSLATEGEMMA_SERVER_URL,
                "error": str(exc),
            }

    def translate(
        self,
        text: str,
        src_lang: str = "auto",
        tgt_lang: str = TGT_CODE,
        country: str = "",
    ) -> str:
        if not USE_MODEL_SERVERS:
            from backend.services.translategemma_service import translategemma_service

            return translategemma_service.translate(
                text, src_lang=src_lang, tgt_lang=tgt_lang, country=country
            )
        data = rpc_json(
            f"{TRANSLATEGEMMA_SERVER_URL}/translate",
            {
                "text": text,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "country": country,
            },
            timeout=MODEL_RPC_TIMEOUT,
        )
        return str(data.get("translation") or "")


translategemma_rpc = TranslateGemmaRpc()
