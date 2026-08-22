"""
TranslateGemma tarjima mikroserveri — model bir marta yuklanadi.

    .venv/bin/python -m uvicorn backend.workers.translategemma_app:app --host 127.0.0.1 --port 8005
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.config import TRANSLATEGEMMA_SERVER_PORT, ensure_runtime_dirs
from backend.services.translategemma_service import TGT_CODE, translategemma_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("translategemma_worker")


class TranslateIn(BaseModel):
    text: str = Field(..., min_length=1)
    src_lang: str = "auto"
    tgt_lang: str = TGT_CODE
    country: str = ""


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_runtime_dirs()
    try:
        translategemma_service.ensure_loaded()
        logger.info("TranslateGemma tayyor: %s", translategemma_service.status())
    except Exception:
        logger.exception("TranslateGemma ishga tushishda yuklanmadi — birinchi so'rovda qayta uriniladi")
    yield


app = FastAPI(title="TranslateGemma worker", docs_url=None, redoc_url=None, lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "translategemma"}


@app.get("/status")
def status() -> dict:
    return translategemma_service.status()


@app.post("/translate")
def translate(body: TranslateIn) -> dict:
    try:
        translation = translategemma_service.translate(
            body.text,
            src_lang=body.src_lang,
            tgt_lang=body.tgt_lang,
            country=body.country,
        )
    except Exception as exc:
        raise HTTPException(500, str(exc)[:800]) from exc
    return {
        "translation": translation,
        "model_used": "translategemma",
        "src_lang": body.src_lang,
        "tgt_lang": body.tgt_lang,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.workers.translategemma_app:app",
        host="127.0.0.1",
        port=TRANSLATEGEMMA_SERVER_PORT,
        reload=False,
    )
