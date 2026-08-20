"""
NLLB-200 tarjima mikroserveri — model bir marta yuklanadi va xotirada qoladi.

    .venv/bin/python -m uvicorn backend.workers.nllb_app:app --host 127.0.0.1 --port 8003
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.config import TRANSLATION_SERVER_PORT, ensure_runtime_dirs
from backend.services.nllb_service import TGT_UZBEK_LATIN, nllb_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("nllb_worker")


class TranslateIn(BaseModel):
    text: str = Field(..., min_length=1)
    src_lang: str = "auto"
    tgt_lang: str = TGT_UZBEK_LATIN
    country: str = ""


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_runtime_dirs()
    try:
        nllb_service.ensure_loaded()
        logger.info("NLLB tayyor: %s", nllb_service.status())
    except Exception:
        logger.exception("NLLB ishga tushishda yuklanmadi — birinchi so'rovda qayta uriniladi")
    yield


app = FastAPI(title="NLLB worker", docs_url=None, redoc_url=None, lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "nllb"}


@app.get("/status")
def status() -> dict:
    return nllb_service.status()


@app.post("/translate")
def translate(body: TranslateIn) -> dict:
    try:
        translation = nllb_service.translate(
            body.text,
            src_lang=body.src_lang,
            tgt_lang=body.tgt_lang,
            country=body.country,
        )
    except Exception as exc:
        raise HTTPException(500, str(exc)[:800]) from exc
    return {
        "translation": translation,
        "model_used": "nllb-200",
        "src_lang": body.src_lang,
        "tgt_lang": body.tgt_lang,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.workers.nllb_app:app", host="127.0.0.1", port=TRANSLATION_SERVER_PORT, reload=False)
