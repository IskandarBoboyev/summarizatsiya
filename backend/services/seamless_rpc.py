"""Asosiy ilova SeamlessM4T v2 mikroserverini chaqiradi."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.config import MODEL_RPC_TIMEOUT, SEAMLESS_SERVER_URL, USE_MODEL_SERVERS
from backend.services.rpc import rpc_json


class SeamlessRpc:
    """SeamlessM4T worker (port 8006) uchun HTTP mijoz."""

    def status(self) -> dict[str, Any]:
        if not USE_MODEL_SERVERS:
            from backend.services.seamless_service import seamless_service

            return {**seamless_service.status(), "mode": "in-process"}
        try:
            data = rpc_json(f"{SEAMLESS_SERVER_URL}/status", method="GET", timeout=5)
            data["mode"] = "microservice"
            data["url"] = SEAMLESS_SERVER_URL
            return data
        except Exception as exc:
            return {
                "ready": False,
                "backend": None,
                "model_key": "seamless-m4t-v2",
                "mode": "microservice",
                "url": SEAMLESS_SERVER_URL,
                "error": str(exc),
            }

    def transcribe(
        self,
        filepath: str | Path,
        language: str = "auto",
        country: str = "",
    ) -> str:
        if not USE_MODEL_SERVERS:
            from backend.services.seamless_service import seamless_service

            return seamless_service.transcribe(filepath, language=language, country=country)
        data = rpc_json(
            f"{SEAMLESS_SERVER_URL}/transcribe",
            {"path": str(filepath), "language": language, "country": country},
            timeout=MODEL_RPC_TIMEOUT,
        )
        return str(data.get("text") or "")


seamless_rpc = SeamlessRpc()
