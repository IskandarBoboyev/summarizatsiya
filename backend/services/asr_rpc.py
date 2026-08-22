"""Asosiy ilova GigaAM mikroserverini chaqiradi — model shu jarayonda yuklanmaydi."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.config import ASR_SERVER_URL, MODEL_RPC_TIMEOUT, USE_MODEL_SERVERS
from backend.services.rpc import rpc_json


class AsrRpc:
    """GigaAM worker (port 8002) uchun HTTP mijoz."""

    def status(self) -> dict[str, Any]:
        if not USE_MODEL_SERVERS:
            from backend.services.asr_service import asr_service

            return {**asr_service.status(), "mode": "in-process"}
        try:
            data = rpc_json(f"{ASR_SERVER_URL}/status", method="GET", timeout=5)
            data["mode"] = "microservice"
            data["url"] = ASR_SERVER_URL
            return data
        except Exception as exc:
            return {
                "ready": False,
                "backend": None,
                "mode": "microservice",
                "url": ASR_SERVER_URL,
                "error": str(exc),
            }

    def transcribe(
        self,
        filepath: str | Path,
        language: str = "auto",
        model_key: str = "",
        country: str = "",
    ) -> str:
        key = (model_key or "gigaam-multilingual").strip().lower()
        if key.startswith("seamless"):
            from backend.services.seamless_rpc import seamless_rpc

            return seamless_rpc.transcribe(filepath, language=language, country=country)
        if not USE_MODEL_SERVERS:
            from backend.services.asr_service import asr_service

            return asr_service.transcribe(filepath, language=language)
        data = rpc_json(
            f"{ASR_SERVER_URL}/transcribe",
            {"path": str(filepath), "language": language},
            timeout=MODEL_RPC_TIMEOUT,
        )
        return str(data.get("text") or "")

    def unload(self) -> None:
        """Mikroserverda model doim xotirada qoladi."""
        if not USE_MODEL_SERVERS:
            from backend.services.asr_service import asr_service

            asr_service.unload()


asr_rpc = AsrRpc()
