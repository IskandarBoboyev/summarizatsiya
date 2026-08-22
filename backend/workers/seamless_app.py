"""
SeamlessM4T v2 transkripsiya mikroserveri — model bir marta yuklanadi.

    .venv/bin/python -m uvicorn backend.workers.seamless_app:app --host 127.0.0.1 --port 8006
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.config import SEAMLESS_SERVER_PORT, ensure_runtime_dirs
from backend.services.seamless_service import seamless_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("seamless_worker")


class TranscribeIn(BaseModel):
    path: str = Field(..., min_length=1)
    language: str = "auto"
    country: str = ""


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_runtime_dirs()
    try:
        seamless_service.ensure_loaded()
        logger.info("SeamlessM4T tayyor: %s", seamless_service.status())
    except Exception:
        logger.exception("SeamlessM4T ishga tushishda yuklanmadi — birinchi so'rovda qayta uriniladi")
    yield


app = FastAPI(title="SeamlessM4T v2 worker", docs_url=None, redoc_url=None, lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "seamless"}


@app.get("/status")
def status() -> dict:
    return seamless_service.status()


@app.post("/transcribe")
def transcribe(body: TranscribeIn) -> dict:
    path = Path(body.path)
    if not path.is_file():
        raise HTTPException(404, f"Audio/video topilmadi: {path}")
    try:
        text = seamless_service.transcribe(path, language=body.language, country=body.country)
    except Exception as exc:
        raise HTTPException(500, str(exc)[:800]) from exc
    return {"text": text, "model_used": "seamless-m4t-v2"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.workers.seamless_app:app",
        host="127.0.0.1",
        port=SEAMLESS_SERVER_PORT,
        reload=False,
    )
