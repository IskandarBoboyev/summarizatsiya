"""Asosiy ilova Surya mikroserverini chaqiradi."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.config import MODEL_RPC_TIMEOUT, OCR_SERVER_URL, USE_MODEL_SERVERS
from backend.services.rpc import rpc_json


class OcrRpc:
    """Surya worker (port 8004) uchun HTTP mijoz."""

    def status(self) -> dict[str, Any]:
        if not USE_MODEL_SERVERS:
            from backend.services.surya_service import surya_service

            return {**surya_service.status(), "mode": "in-process"}
        try:
            data = rpc_json(f"{OCR_SERVER_URL}/status", method="GET", timeout=5)
            data["mode"] = "microservice"
            data["url"] = OCR_SERVER_URL
            return data
        except Exception as exc:
            return {
                "ready": False,
                "backend": None,
                "mode": "microservice",
                "url": OCR_SERVER_URL,
                "error": str(exc),
            }

    def extract(self, filepath: str | Path) -> str:
        if not USE_MODEL_SERVERS:
            from backend.services.surya_service import surya_service

            return surya_service.extract(filepath)
        data = rpc_json(
            f"{OCR_SERVER_URL}/extract",
            {"path": str(filepath)},
            timeout=MODEL_RPC_TIMEOUT,
        )
        return str(data.get("text") or "")


ocr_rpc = OcrRpc()
