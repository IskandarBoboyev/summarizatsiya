"""
Surya OCR mikroserveri — model bir marta yuklanadi.

    .venv/bin/python -m uvicorn backend.workers.surya_app:app --host 127.0.0.1 --port 8004
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.config import OCR_SERVER_PORT, ensure_runtime_dirs
from backend.services.surya_service import surya_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("surya_worker")


class ExtractIn(BaseModel):
    path: str = Field(..., min_length=1)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_runtime_dirs()
    try:
        surya_service.ensure_loaded()
        logger.info("Surya tayyor: %s", surya_service.status())
    except Exception:
        logger.exception("Surya ishga tushishda yuklanmadi — birinchi so'rovda qayta uriniladi")
    yield


app = FastAPI(title="Surya OCR worker", docs_url=None, redoc_url=None, lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "surya"}


@app.get("/status")
def status() -> dict:
    return surya_service.status()


@app.post("/extract")
def extract(body: ExtractIn) -> dict:
    path = Path(body.path)
    if not path.is_file():
        raise HTTPException(404, f"Fayl topilmadi: {path}")
    try:
        text = surya_service.extract(path)
    except Exception as exc:
        raise HTTPException(500, str(exc)[:800]) from exc
    return {"text": text}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.workers.surya_app:app", host="127.0.0.1", port=OCR_SERVER_PORT, reload=False)
