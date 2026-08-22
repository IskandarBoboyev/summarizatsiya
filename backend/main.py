"""
FastAPI kirish nuqtasi va API marshrutlari.

Ishga tushirish (loyiha ildizidan):
    uvicorn backend.main:app --host 127.0.0.1 --port 8000

100% offline: config importi HF_HUB_OFFLINE ni o'rnatadi.
Frontend CDN ishlatmaydi — /static orqali lokal CSS/JS beriladi.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Loyiha ildizini PYTHONPATH ga qo'shish (to'g'ridan-to'g'ri python backend/main.py)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request

from backend.config import (
    ASR_LANGUAGES,
    COUNTRIES,
    COUNTRY_KEYS,
    DATA_DIR,
    DOCUMENTS_PAGE_SIZE,
    FRONTEND_DIR,
    HOST,
    PORT,
    QUEUE_POLL_INTERVAL,
    SUPPORTED_EXTENSIONS,
    WATCHED_DIR,
    browse_directory,
    device_info,
    ensure_runtime_dirs,
    list_available_asr_models,
    list_available_llm_models,
    list_available_ocr_models,
    list_available_translation_models,
)
from backend.database import (
    country_file_stats,
    update_folder_asr_model,
    get_all_settings,
    get_document,
    init_db,
    list_documents,
    list_home_by_country,
    document_flow_steps,
    list_pipeline_live,
    mark_reprocess,
    public_document,
    queue_counts,
    set_setting,
)
from backend.pipeline import pipeline
from backend.services.asr_rpc import asr_rpc
from backend.services.llm_rpc import llm_rpc
from backend.services.model_health import collect_worker_health, health_summary
from backend.services.worker_control import start_worker, stop_worker, worker_spec
from backend.services.nllb_rpc import nllb_rpc
from backend.services import rag_service
from backend.services.watcher_service import watcher_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("local_doc_platform")


# ---------------------------------------------------------------------------
# WebSocket xona — real-time kartalar
# ---------------------------------------------------------------------------

class ConnectionHub:
    """Ulangan brauzerlarga JSON hodisalar yuboradi."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self.loop: asyncio.AbstractEventLoop | None = None

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, event: str, payload: Any) -> None:
        message = {"type": event, "payload": payload}
        async with self._lock:
            dead: list[WebSocket] = []
            for ws in self._clients:
                try:
                    await ws.send_json(message)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._clients.discard(ws)

    def emit_threadsafe(self, event: str, payload: Any) -> None:
        """Watchdog / worker oqimidan chaqirish (thread-safe)."""
        loop = self.loop
        if loop is None or loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(self.broadcast(event, payload), loop)


hub = ConnectionHub()


# ---------------------------------------------------------------------------
# So'rov sxemalari
# ---------------------------------------------------------------------------

class FolderIn(BaseModel):
    """Davlat uchun fayl yoki papka."""

    path: str = Field(..., min_length=1, description="Lokal fayl yoki katalog yo'li")
    country: str = Field(..., min_length=2, max_length=8)


class FolderPatchIn(BaseModel):
    """Kuzatiladigan papka sozlamasi."""

    asr_model: str = ""


class ChatIn(BaseModel):
    """RAG chat savoli — javob shu davlat RAG bazasidan."""

    message: str = Field(..., min_length=1, max_length=8000)
    country: str | None = None


class SettingsIn(BaseModel):
    """UI sozlamalari."""

    llm_model: str | None = None
    llm_summary_model: str | None = None
    llm_translate_model: str | None = None
    asr_model: str | None = None
    ocr_model: str | None = None
    asr_language: str | None = None
    quantization: str | None = None


# ---------------------------------------------------------------------------
# Fon ishchi: navbatni aylanib pipeline ni ishlatadi
# ---------------------------------------------------------------------------

async def _queue_worker() -> None:
    """
    Har QUEUE_POLL_INTERVAL soniyada pending faylni qidiradi.

    Og'ir hisoblash thread pool da — event loop bloklanmaydi.
    """
    logger.info("Navbat ishchisi boshlandi")
    while True:
        try:
            had_work = await asyncio.to_thread(pipeline.process_next)
            if not had_work:
                await asyncio.sleep(QUEUE_POLL_INTERVAL)
        except asyncio.CancelledError:
            logger.info("Navbat ishchisi to'xtatildi")
            raise
        except Exception:
            logger.exception("Navbat ishchisi xatosi")
            await asyncio.sleep(1.5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ilova umri: papkalar, DB, watcher, worker."""
    ensure_runtime_dirs()
    init_db()
    hub.loop = asyncio.get_running_loop()

    def _notify(event: str, payload: dict) -> None:
        hub.emit_threadsafe(event, payload)

    pipeline.set_notify(_notify)
    watcher_service.set_enqueue_callback(
        lambda doc: hub.emit_threadsafe("document_updated", public_document(doc) or doc)
    )
    try:
        watcher_service.start()
    except Exception:
        logger.exception("Watcher ishga tushmadi — UI baribir ochiladi")

    worker = asyncio.create_task(_queue_worker(), name="queue-worker")
    logger.info("Server tayyor. UI: http://%s:%s  |  Kuzatuv: %s", HOST, PORT, WATCHED_DIR)
    try:
        yield
    finally:
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
        watcher_service.stop()
        llm_rpc.unload()
        asr_rpc.unload()


app = FastAPI(
    title="Lokal hujjatlar platformasi",
    description="Offline OCR + Gemma xulosa + NLLB tarjima + GigaAM transkripsiya",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,  # offline, tashqi swagger CDN yo'q
    redoc_url=None,
)

templates = Jinja2Templates(directory=str(FRONTEND_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")


@app.middleware("http")
async def _no_cache_static(request: Request, call_next):
    """Brauzer eski JS/CSS ni ushlab qolmasin."""
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


# ---------------------------------------------------------------------------
# Sahifa
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Asosiy SPA interfeysi."""
    return templates.TemplateResponse(request, "index.html")


@app.get("/gerb.mp4")
async def gerb_video() -> FileResponse:
    """Lokal 3D gerb videosi — tashqi URL yo'q."""
    path = FRONTEND_DIR.parent / "public" / "gerb.mp4"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="gerb.mp4 topilmadi")
    return FileResponse(path, media_type="video/mp4")


@app.get("/askar.mp4")
async def askar_video() -> FileResponse:
    """Lokal askar videosi — tashqi URL yo'q."""
    path = FRONTEND_DIR.parent / "public" / "askar.mp4"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="askar.mp4 topilmadi")
    return FileResponse(path, media_type="video/mp4")


# ---------------------------------------------------------------------------
# Holat / sozlamalar
# ---------------------------------------------------------------------------

@app.get("/api/status")
async def api_status() -> dict:
    """Qurilma, navbat, model holati — yuqori panel uchun."""
    health = health_summary(await collect_worker_health())
    by_id = {w["id"]: w for w in health["workers"]}
    return {
        "device": device_info(),
        "queue": queue_counts(),
        "workers": health["workers"],
        "workers_live": health["live"],
        "workers_total": health["total"],
        "llm": by_id.get("gemma") or {},
        "asr": by_id.get("gigaam") or {},
        "translation": by_id.get("nllb") or {},
        "ocr": by_id.get("surya") or {},
        "models": list_available_llm_models(),
        "asr_models": list_available_asr_models(),
        "ocr_models": list_available_ocr_models(),
        "translation_models": list_available_translation_models(),
        "asr_languages": ASR_LANGUAGES,
        "settings": get_all_settings(),
        "offline": True,
        "rag": rag_service.stats(),
        "countries": COUNTRIES,
    }


@app.get("/api/models/health")
async def api_models_health() -> dict:
    """Gemma / GigaAM / NLLB ish holati (UI nuqtalari va CLI)."""
    return health_summary(await collect_worker_health())


@app.post("/api/models/{worker_id}/start")
async def api_start_model(worker_id: str) -> dict:
    """Tanlangan model mikroserverini ishga tushiradi."""
    try:
        worker_spec(worker_id)
    except KeyError:
        raise HTTPException(404, f"Noma'lum model: {worker_id}") from None
    try:
        return await asyncio.to_thread(start_worker, worker_id)
    except Exception as exc:
        raise HTTPException(500, str(exc)[:400]) from exc


@app.post("/api/models/{worker_id}/stop")
async def api_stop_model(worker_id: str) -> dict:
    """Tanlangan model mikroserverini to'xtatadi."""
    try:
        worker_spec(worker_id)
    except KeyError:
        raise HTTPException(404, f"Noma'lum model: {worker_id}") from None
    try:
        return await asyncio.to_thread(stop_worker, worker_id)
    except Exception as exc:
        raise HTTPException(500, str(exc)[:400]) from exc


@app.get("/api/settings")
async def api_get_settings() -> dict:
    """Saqlangan sozlamalar."""
    return get_all_settings()


@app.put("/api/settings")
async def api_put_settings(body: SettingsIn) -> dict:
    """
    LLM model, ASR tili, kvantizatsiyani saqlaydi.

    Model o'zgarsa, keyingi faylda yangi model yuklanadi.
    """
    llm_keys = {m["key"] for m in list_available_llm_models()}

    def _save_llm(setting_key: str, value: str) -> None:
        if value not in llm_keys:
            raise HTTPException(400, f"Noma'lum model: {value}")
        set_setting(setting_key, value)

    if body.llm_summary_model:
        _save_llm("llm_summary_model", body.llm_summary_model)
        set_setting("llm_model", body.llm_summary_model)
    elif body.llm_model:
        _save_llm("llm_summary_model", body.llm_model)
        set_setting("llm_model", body.llm_model)

    if body.llm_translate_model:
        tr_keys = {m["key"] for m in list_available_translation_models()}
        if body.llm_translate_model in tr_keys:
            set_setting("llm_translate_model", body.llm_translate_model)
        else:
            _save_llm("llm_translate_model", body.llm_translate_model)

    if body.asr_model:
        keys = {m["key"] for m in list_available_asr_models()}
        if body.asr_model not in keys:
            raise HTTPException(400, f"Noma'lum ASR model: {body.asr_model}")
        set_setting("asr_model", body.asr_model)

    if body.ocr_model:
        keys = {m["key"] for m in list_available_ocr_models()}
        if body.ocr_model not in keys:
            raise HTTPException(400, f"Noma'lum OCR model: {body.ocr_model}")
        set_setting("ocr_model", body.ocr_model)

    if body.asr_language:
        if body.asr_language not in ASR_LANGUAGES:
            raise HTTPException(400, f"Noma'lum til: {body.asr_language}")
        set_setting("asr_language", body.asr_language)

    if body.quantization:
        if body.quantization not in {"auto", "none", "8bit", "4bit"}:
            raise HTTPException(400, "quantization: auto | none | 8bit | 4bit")
        set_setting("quantization", body.quantization)
        await asyncio.to_thread(
            llm_rpc.ensure_loaded,
            get_all_settings().get("llm_model", "gemma-4"),
            body.quantization,
        )

    return get_all_settings()


# ---------------------------------------------------------------------------
# Papkalar
# ---------------------------------------------------------------------------

@app.get("/api/browse")
async def api_browse(path: str | None = Query(None)) -> dict:
    """Shu qurilmadagi papkalarni ochish (fayl-menejer)."""
    try:
        return await asyncio.to_thread(browse_directory, path)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/folders")
async def api_list_folders() -> list[dict]:
    """Kuzatilayotgan papkalar."""
    return watcher_service.list_watch_folders()


@app.post("/api/upload")
async def api_upload(
    country: str = Form(...),
    file: UploadFile = File(...),
) -> dict:
    """Davlat uchun faylni yuklab, navbatga qo'yadi."""
    if country not in COUNTRY_KEYS:
        raise HTTPException(400, f"Noma'lum davlat: {country}")
    raw_name = Path(file.filename or "fayl").name
    suffix = Path(raw_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Bu fayl turi qo‘llab-quvvatlanmaydi: {suffix or 'noma’lum'}")
    safe = "".join(ch if ch.isalnum() or ch in "._- ()[]" else "_" for ch in raw_name).strip() or f"fayl{suffix}"
    dest_dir = DATA_DIR / "uploads" / country
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / safe
    if dest.exists():
        dest = dest_dir / f"{dest.stem}_{int(time.time())}{dest.suffix}"
    try:
        payload = await file.read()
        dest.write_bytes(payload)
    except Exception as exc:
        raise HTTPException(500, f"Fayl yozilmadi: {exc}") from exc
    try:
        folder = await asyncio.to_thread(
            watcher_service.add_watch_folder, str(dest), country
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    await hub.broadcast("folders", watcher_service.list_watch_folders())
    await hub.broadcast("queue_status", queue_counts())
    return folder


@app.post("/api/folders")
async def api_add_folder(body: FolderIn) -> dict:
    """Davlat uchun fayl/papka belgilash va darhol skanerlash."""
    if body.country not in COUNTRY_KEYS:
        raise HTTPException(400, f"Noma'lum davlat: {body.country}")
    try:
        folder = await asyncio.to_thread(
            watcher_service.add_watch_folder, body.path, body.country
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    await hub.broadcast("folders", watcher_service.list_watch_folders())
    await hub.broadcast("queue_status", queue_counts())
    return folder


@app.patch("/api/folders/{folder_id}")
async def api_patch_folder(folder_id: int, body: FolderPatchIn) -> dict:
    """Papka uchun transkripsiya modelini belgilash."""
    try:
        folder = await asyncio.to_thread(update_folder_asr_model, folder_id, body.asr_model)
    except KeyError:
        raise HTTPException(404, "Papka topilmadi") from None
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    await hub.broadcast("folders", watcher_service.list_watch_folders())
    return folder


@app.delete("/api/folders/{folder_id}")
async def api_delete_folder(folder_id: int) -> dict:
    """Papkani kuzatuvdan olib tashlash."""
    ok = await asyncio.to_thread(watcher_service.remove_watch_folder, folder_id)
    if not ok:
        raise HTTPException(404, "Papka topilmadi")
    await hub.broadcast("folders", watcher_service.list_watch_folders())
    return {"ok": True}


@app.post("/api/scan")
async def api_scan() -> dict:
    """Barcha papkalarni qayta skanerlash."""
    count = await asyncio.to_thread(watcher_service.scan_all)
    await hub.broadcast("queue_status", queue_counts())
    return {"enqueued": count, "queue": queue_counts()}


# ---------------------------------------------------------------------------
# Hujjatlar
# ---------------------------------------------------------------------------

@app.get("/api/stats/countries")
async def api_country_stats(
    period: str = Query("month", pattern="^(day|week|month|year)$"),
) -> dict:
    """Davlatlar bo'yicha kelgan fayllar statistikasi."""
    return country_file_stats(period)


@app.get("/api/documents/home")
async def api_home_documents(
    limit: int = Query(20, ge=1, le=50),
    period: str = Query("day", pattern="^(day|week|month|year)$"),
) -> dict:
    """Kirish: tanlangan davr ichida 6 davlat bo'yicha oxirgi xulosalar."""
    groups = list_home_by_country(limit, period)
    return {
        "period": period,
        "countries": [
            {
                "key": g["key"],
                "label": g["label"],
                "total": g["total"],
                "items": [public_document(x) for x in g["items"]],
            }
            for g in groups
        ]
    }


PIPELINE_STAGES = (
    {"id": "queued", "label": "Fayl keldi", "hint": "Navbat"},
    {"id": "extract", "label": "O‘qish", "hint": "OCR / ASR"},
    {"id": "summarize", "label": "Xulosa", "hint": "Gemma"},
    {"id": "translate", "label": "Tarjima", "hint": "NLLB"},
    {"id": "done", "label": "Natija", "hint": "Tayyor"},
)


@app.get("/api/pipeline/live")
async def api_pipeline_live() -> dict:
    """Jarayon oynasi: fayldan natijagacha bosqichlar."""
    raw = list_pipeline_live(60)
    items: list[dict] = []
    columns: dict[str, list] = {s["id"]: [] for s in PIPELINE_STAGES}
    columns["error"] = []
    active = None
    for src in raw:
        pub = public_document(src)
        if not pub:
            continue
        pub["flow_steps"] = document_flow_steps(pub)
        items.append(pub)
        stage = str(pub.get("pipeline_stage") or "queued")
        if stage == "error":
            columns["error"].append(pub)
        else:
            columns.setdefault(stage, []).append(pub)
        if pub.get("status") == "processing" and active is None:
            active = pub
    if active is None:
        for doc in items:
            if doc.get("status") != "pending":
                active = doc
                break
        if active is None and items:
            active = items[0]
    return {
        "stages": list(PIPELINE_STAGES),
        "columns": columns,
        "active": active,
        "items": items,
        "queue": queue_counts(),
    }


@app.get("/api/documents")
async def api_list_documents(
    offset: int = Query(0, ge=0),
    limit: int = Query(DOCUMENTS_PAGE_SIZE, ge=1),
    order: str = Query("asc", pattern="^(asc|desc)$"),
    country: str | None = Query(None),
) -> dict:
    """
    Infinite scroll uchun sahifalangan ro'yxat.

    Tartib: asc — pipeline; desc — kirish oynasi (oxirgi avval).
    """
    if country and country not in COUNTRY_KEYS:
        raise HTTPException(400, f"Noma'lum davlat: {country}")
    limit = min(int(limit), 50)
    items, total = list_documents(offset=offset, limit=limit, order=order, country=country)
    return {
        "items": [public_document(x) for x in items],
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(items) < total,
    }


@app.post("/api/chat")
async def api_chat(body: ChatIn) -> dict:
    """RAG chat: bazaga yozilgan hujjatlardan javob."""
    text = body.message.strip()
    if not text:
        raise HTTPException(400, "Xabar bo‘sh")
    country = body.country if body.country in COUNTRY_KEYS else None
    try:
        return await asyncio.to_thread(_chat_with_rag, text, country)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        logger.exception("RAG chat xatosi")
        raise HTTPException(500, str(exc)[:800]) from exc


def _country_label(key: str) -> str:
    for item in COUNTRIES:
        if item["key"] == key:
            return item["label"]
    return key


def _filled_rag_countries() -> list[str]:
    by = (rag_service.stats() or {}).get("by_country") or {}
    names: list[str] = []
    for key, val in by.items():
        if isinstance(val, dict) and int(val.get("documents") or 0) > 0:
            names.append(_country_label(str(key)))
    return names


def _chat_with_rag(question: str, country: str | None = None) -> dict:
    """Qidiruv + Gemma javobi — faqat shu davlat RAG bazasi."""
    stats = rag_service.stats(country=country)
    if stats["documents"] <= 0:
        filled = _filled_rag_countries()
        extra = (
            f" Hozir bazada: {', '.join(filled)}. Chatda shu davlatni tanlang."
            if filled
            else " Pipeline kartasida «Bazaga yozish» tugmasini bosing."
        )
        label = _country_label(country) if country else "Tanlangan davlat"
        return {
            "role": "assistant",
            "reply": f"{label} RAG bazasi bo‘sh.{extra}",
            "sources": [],
        }

    hits = rag_service.retrieve(question, country=country)
    if not hits:
        return {
            "role": "assistant",
            "reply": (
                "Bazadagi hujjatlarda bu savolga mos joy topilmadi. "
                "Boshqa so‘z bilan so‘rang yoki yana bir faylni bazaga yozing."
            ),
            "sources": [],
        }

    context = rag_service.build_context(hits)
    settings = get_all_settings()
    reply = llm_rpc.answer_with_context(
        question,
        context,
        settings.get("llm_summary_model") or settings.get("llm_model", "gemma-4"),
        settings.get("quantization", "auto"),
    )
    sources = []
    seen: set[int] = set()
    for hit in hits:
        did = int(hit.get("document_id") or 0)
        if did in seen:
            continue
        seen.add(did)
        sources.append({"document_id": did, "filename": hit.get("filename") or ""})
    return {"role": "assistant", "reply": reply.strip(), "sources": sources}


@app.post("/api/documents/{doc_id}/index")
async def api_index_document(doc_id: int) -> dict:
    """Foydalanuvchi tasdiqlagan hujjatni RAG bazasiga yozadi."""
    try:
        result = await asyncio.to_thread(rag_service.index_document, doc_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    fresh = public_document(get_document(doc_id))
    await hub.broadcast("document_updated", fresh)
    return {**(fresh or {}), **result}


@app.get("/api/documents/{doc_id}")
async def api_get_document(doc_id: int) -> dict:
    """Bitta hujjat (to'liq matn)."""
    doc = get_document(doc_id)
    if doc is None:
        raise HTTPException(404, "Hujjat topilmadi")
    return public_document(doc)  # type: ignore[return-value]


@app.post("/api/documents/{doc_id}/reprocess")
async def api_reprocess(doc_id: int) -> dict:
    """Hujjatni qayta navbatga qo'yish."""
    doc = mark_reprocess(doc_id)
    if doc is None:
        raise HTTPException(404, "Hujjat topilmadi")
    await hub.broadcast("document_updated", public_document(doc))
    await hub.broadcast("queue_status", queue_counts())
    return public_document(doc)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """
    Real-time yangilanishlar:
        document_updated, queue_status, folders
    """
    await hub.connect(ws)
    try:
        await ws.send_json({"type": "hello", "payload": {"queue": queue_counts()}})
        while True:
            # Mijoz ping yuborishi mumkin — shunchaki o'qiymiz
            await ws.receive_text()
    except WebSocketDisconnect:
        await hub.disconnect(ws)
    except Exception:
        await hub.disconnect(ws)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=HOST,
        port=PORT,
        reload=False,
        factory=False,
    )
