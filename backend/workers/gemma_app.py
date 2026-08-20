"""
Gemma mikroserveri — model bir marta yuklanadi va xotirada qoladi.

    .venv/bin/python -m uvicorn backend.workers.gemma_app:app --host 127.0.0.1 --port 8001
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.config import DEFAULT_LLM_MODEL, LLM_SERVER_PORT, ensure_runtime_dirs
from backend.services.llm_service import llm_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("gemma_worker")


class LoadIn(BaseModel):
    model_key: str = DEFAULT_LLM_MODEL
    quantization: str = "auto"


class ProcessIn(LoadIn):
    text: str = Field(..., min_length=1)


class ChatIn(LoadIn):
    question: str = Field(..., min_length=1)
    context: str = ""


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_runtime_dirs()
    try:
        llm_service.ensure_loaded(DEFAULT_LLM_MODEL, "auto")
        logger.info("Gemma tayyor: %s", llm_service.status())
    except Exception:
        logger.exception("Gemma ishga tushishda yuklanmadi — birinchi so'rovda qayta uriniladi")
    yield


app = FastAPI(title="Gemma worker", docs_url=None, redoc_url=None, lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "gemma"}


@app.get("/status")
def status() -> dict:
    return llm_service.status()


@app.post("/load")
def load_model(body: LoadIn) -> dict:
    try:
        llm_service.ensure_loaded(body.model_key, body.quantization)
    except Exception as exc:
        raise HTTPException(503, str(exc)[:800]) from exc
    return llm_service.status()


@app.post("/summarize")
def summarize(body: ProcessIn) -> dict:
    try:
        llm_service.ensure_loaded(body.model_key, body.quantization)
        summary = llm_service.summarize(body.text)
    except Exception as exc:
        raise HTTPException(500, str(exc)[:800]) from exc
    return {"summary": summary, "model_used": body.model_key}


@app.post("/translate")
def translate(body: ProcessIn) -> dict:
    try:
        llm_service.ensure_loaded(body.model_key, body.quantization)
        translation = llm_service.translate_to_uzbek(body.text)
    except Exception as exc:
        raise HTTPException(500, str(exc)[:800]) from exc
    return {"translation": translation, "model_used": body.model_key}


@app.post("/process")
def process_text(body: ProcessIn) -> dict:
    try:
        llm_service.ensure_loaded(body.model_key, body.quantization)
        summary, translation = llm_service.process_text(body.text)
    except Exception as exc:
        raise HTTPException(500, str(exc)[:800]) from exc
    return {
        "summary": summary,
        "translation": translation,
        "model_used": body.model_key,
    }


@app.post("/chat")
def chat(body: ChatIn) -> dict:
    try:
        llm_service.ensure_loaded(body.model_key, body.quantization)
        reply = llm_service.answer_with_context(body.question, body.context)
    except Exception as exc:
        raise HTTPException(500, str(exc)[:800]) from exc
    return {"reply": reply}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.workers.gemma_app:app", host="127.0.0.1", port=LLM_SERVER_PORT, reload=False)
