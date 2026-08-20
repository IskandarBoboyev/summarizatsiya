"""
GigaAM mikroserveri — model bir marta yuklanadi va xotirada qoladi.

    .venv/bin/python -m uvicorn backend.workers.gigaam_app:app --host 127.0.0.1 --port 8002
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.config import ASR_SERVER_PORT, ensure_runtime_dirs
from backend.services.asr_service import asr_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("gigaam_worker")


class TranscribeIn(BaseModel):
    path: str = Field(..., min_length=1)
    language: str = "auto"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_runtime_dirs()
    try:
        asr_service.ensure_loaded()
        logger.info("GigaAM tayyor: %s", asr_service.status())
    except Exception:
        logger.exception("GigaAM ishga tushishda yuklanmadi — birinchi so'rovda qayta uriniladi")
    yield


app = FastAPI(title="GigaAM worker", docs_url=None, redoc_url=None, lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "gigaam"}


@app.get("/status")
def status() -> dict:
    return asr_service.status()


@app.post("/transcribe")
def transcribe(body: TranscribeIn) -> dict:
    path = Path(body.path)
    if not path.is_file():
        raise HTTPException(404, f"Audio/video topilmadi: {path}")
    try:
        text = asr_service.transcribe(path, language=body.language)
    except Exception as exc:
        raise HTTPException(500, str(exc)[:800]) from exc
    return {"text": text}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.workers.gigaam_app:app", host="127.0.0.1", port=ASR_SERVER_PORT, reload=False)
