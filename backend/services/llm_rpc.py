"""Asosiy ilova Gemma mikroserverini chaqiradi — model shu jarayonda yuklanmaydi."""

from __future__ import annotations

import logging
from typing import Any

from backend.config import LLM_SERVER_URL, MODEL_RPC_TIMEOUT, USE_MODEL_SERVERS
from backend.services.rpc import rpc_json

logger = logging.getLogger(__name__)


class LlmRpc:
    """Gemma worker (port 8001) uchun HTTP mijoz."""

    def status(self) -> dict[str, Any]:
        if not USE_MODEL_SERVERS:
            from backend.services.llm_service import llm_service

            return {**llm_service.status(), "mode": "in-process"}
        try:
            data = rpc_json(f"{LLM_SERVER_URL}/status", method="GET", timeout=5)
            data["mode"] = "microservice"
            data["url"] = LLM_SERVER_URL
            return data
        except Exception as exc:
            return {
                "ready": False,
                "backend": None,
                "model_key": None,
                "mode": "microservice",
                "url": LLM_SERVER_URL,
                "error": str(exc),
            }

    def ensure_loaded(self, model_key: str, quantization: str = "auto") -> None:
        if not USE_MODEL_SERVERS:
            from backend.services.llm_service import llm_service

            llm_service.ensure_loaded(model_key, quantization)
            return
        rpc_json(
            f"{LLM_SERVER_URL}/load",
            {"model_key": model_key, "quantization": quantization},
            timeout=MODEL_RPC_TIMEOUT,
        )

    def unload(self) -> None:
        if not USE_MODEL_SERVERS:
            from backend.services.llm_service import llm_service

            llm_service.unload()

    def summarize(self, text: str, model_key: str = "gemma-4", quantization: str = "auto") -> str:
        if not USE_MODEL_SERVERS:
            from backend.services.llm_service import llm_service

            llm_service.ensure_loaded(model_key, quantization)
            return llm_service.summarize(text)
        data = rpc_json(
            f"{LLM_SERVER_URL}/summarize",
            {"text": text, "model_key": model_key, "quantization": quantization},
            timeout=MODEL_RPC_TIMEOUT,
        )
        return str(data.get("summary") or "")

    def translate(self, text: str, model_key: str = "gemma-4", quantization: str = "auto") -> str:
        if not USE_MODEL_SERVERS:
            from backend.services.llm_service import llm_service

            llm_service.ensure_loaded(model_key, quantization)
            return llm_service.translate_to_uzbek(text)
        data = rpc_json(
            f"{LLM_SERVER_URL}/translate",
            {"text": text, "model_key": model_key, "quantization": quantization},
            timeout=MODEL_RPC_TIMEOUT,
        )
        return str(data.get("translation") or "")

    def process_text(self, text: str, model_key: str = "gemma-4", quantization: str = "auto") -> tuple[str, str]:
        return (
            self.summarize(text, model_key, quantization),
            self.translate(text, model_key, quantization),
        )

    def answer_with_context(
        self,
        question: str,
        context: str,
        model_key: str = "gemma-4",
        quantization: str = "auto",
    ) -> str:
        if not USE_MODEL_SERVERS:
            from backend.services.llm_service import llm_service

            llm_service.ensure_loaded(model_key, quantization)
            return llm_service.answer_with_context(question, context)
        data = rpc_json(
            f"{LLM_SERVER_URL}/chat",
            {
                "question": question,
                "context": context,
                "model_key": model_key,
                "quantization": quantization,
            },
            timeout=MODEL_RPC_TIMEOUT,
        )
        return str(data.get("reply") or "")


llm_rpc = LlmRpc()
