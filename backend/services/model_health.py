"""Gemma / GigaAM / NLLB workerlarining jonli holati."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from typing import Any

from backend.config import (
    ASR_SERVER_PORT,
    ASR_SERVER_URL,
    DATA_DIR,
    LLM_SERVER_PORT,
    LLM_SERVER_URL,
    TRANSLATION_SERVER_PORT,
    TRANSLATION_SERVER_URL,
    OCR_SERVER_PORT,
    OCR_SERVER_URL,
    TRANSLATEGEMMA_SERVER_PORT,
    TRANSLATEGEMMA_SERVER_URL,
    SEAMLESS_SERVER_PORT,
    SEAMLESS_SERVER_URL,
    USE_MODEL_SERVERS,
    list_available_asr_models,
    list_available_llm_models,
    list_available_ocr_models,
    list_available_translation_models,
)

PROBE_TIMEOUT = 1.5

WORKER_SPECS: list[dict[str, Any]] = [
    {
        "id": "gemma",
        "label": "Gemma",
        "chip_label": "Xulosa",
        "role": "Xulosa va RAG chat",
        "port": LLM_SERVER_PORT,
        "url": LLM_SERVER_URL,
        "pid_file": DATA_DIR / "gemma_worker.pid",
        "log_file": "data/gemma_worker.log",
        "module": "backend.workers.gemma_app:app",
        "start_cmd": (
            ".venv/bin/python -m uvicorn backend.workers.gemma_app:app "
            "--host 127.0.0.1 --port 8001"
        ),
        "download_cmd": ".venv/bin/python -u scripts/download_models.py gemma",
    },
    {
        "id": "gigaam",
        "label": "GigaAM",
        "chip_label": "Transkripsiya",
        "role": "Audio / video transkripsiya",
        "port": ASR_SERVER_PORT,
        "url": ASR_SERVER_URL,
        "pid_file": DATA_DIR / "gigaam_worker.pid",
        "log_file": "data/gigaam_worker.log",
        "module": "backend.workers.gigaam_app:app",
        "start_cmd": (
            ".venv/bin/python -m uvicorn backend.workers.gigaam_app:app "
            "--host 127.0.0.1 --port 8002"
        ),
        "download_cmd": ".venv/bin/python -u scripts/download_models.py gigaam",
    },
    {
        "id": "nllb",
        "label": "NLLB",
        "chip_label": "Tarjima",
        "role": "O‘zbekcha (lotin) tarjima",
        "port": TRANSLATION_SERVER_PORT,
        "url": TRANSLATION_SERVER_URL,
        "pid_file": DATA_DIR / "nllb_worker.pid",
        "log_file": "data/nllb_worker.log",
        "module": "backend.workers.nllb_app:app",
        "start_cmd": (
            ".venv/bin/python -m uvicorn backend.workers.nllb_app:app "
            "--host 127.0.0.1 --port 8003"
        ),
        "download_cmd": ".venv/bin/python -u scripts/download_models.py nllb",
    },
    {
        "id": "surya",
        "label": "Surya",
        "chip_label": "OCR",
        "role": "Rasm / PDF OCR",
        "port": OCR_SERVER_PORT,
        "url": OCR_SERVER_URL,
        "pid_file": DATA_DIR / "surya_worker.pid",
        "log_file": "data/surya_worker.log",
        "module": "backend.workers.surya_app:app",
        "start_cmd": (
            ".venv/bin/python -m uvicorn backend.workers.surya_app:app "
            "--host 127.0.0.1 --port 8004"
        ),
        "download_cmd": ".venv/bin/python -u scripts/download_models.py surya",
    },
    {
        "id": "translategemma",
        "label": "TranslateGemma",
        "chip_label": "Tarjima",
        "role": "O‘zbekcha (lotin) tarjima",
        "port": TRANSLATEGEMMA_SERVER_PORT,
        "url": TRANSLATEGEMMA_SERVER_URL,
        "pid_file": DATA_DIR / "translategemma_worker.pid",
        "log_file": "data/translategemma_worker.log",
        "module": "backend.workers.translategemma_app:app",
        "start_cmd": (
            ".venv/bin/python -m uvicorn backend.workers.translategemma_app:app "
            "--host 127.0.0.1 --port 8005"
        ),
        "download_cmd": ".venv/bin/python -u scripts/download_models.py translategemma",
    },
    {
        "id": "seamless",
        "label": "SeamlessM4T",
        "chip_label": "Seamless",
        "role": "Audio / video transkripsiya (M4T v2)",
        "port": SEAMLESS_SERVER_PORT,
        "url": SEAMLESS_SERVER_URL,
        "pid_file": DATA_DIR / "seamless_worker.pid",
        "log_file": "data/seamless_worker.log",
        "module": "backend.workers.seamless_app:app",
        "start_cmd": (
            ".venv/bin/python -m uvicorn backend.workers.seamless_app:app "
            "--host 127.0.0.1 --port 8006"
        ),
        "download_cmd": ".venv/bin/python -u scripts/download_models.py seamless",
    },
]


def _http_get_json(url: str, timeout: float = PROBE_TIMEOUT) -> dict[str, Any]:
    req = urllib.request.Request(url, method="GET", headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    if not raw:
        return {}
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else {"result": parsed}


def _disk_ready(worker_id: str) -> bool:
    if worker_id == "gemma":
        return any(m.get("ready") for m in list_available_llm_models())
    if worker_id == "gigaam":
        return any(
            m.get("key") == "gigaam-multilingual" and m.get("ready")
            for m in list_available_asr_models()
        )
    if worker_id == "seamless":
        return any(
            m.get("key") == "seamless-m4t-v2" and m.get("ready")
            for m in list_available_asr_models()
        )
    if worker_id == "nllb":
        return any(m.get("key") == "nllb-200" and m.get("ready") for m in list_available_translation_models())
    if worker_id == "translategemma":
        return any(
            m.get("key") == "translategemma" and m.get("ready")
            for m in list_available_translation_models()
        )
    if worker_id == "surya":
        return any(m.get("key") == "surya" and m.get("ready") for m in list_available_ocr_models())
    return False


def _pid_alive(pid_file) -> int | None:
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    try:
        import os

        os.kill(pid, 0)
    except OSError:
        return None
    return pid


def _in_process_status(worker_id: str) -> dict[str, Any]:
    if worker_id == "gemma":
        from backend.services.llm_service import llm_service

        return llm_service.status()
    if worker_id == "gigaam":
        from backend.services.asr_service import asr_service

        return asr_service.status()
    if worker_id == "surya":
        from backend.services.surya_service import surya_service

        return surya_service.status()
    if worker_id == "translategemma":
        from backend.services.translategemma_service import translategemma_service

        return translategemma_service.status()
    if worker_id == "seamless":
        from backend.services.seamless_service import seamless_service

        return seamless_service.status()
    from backend.services.nllb_service import nllb_service

    return nllb_service.status()


def _classify(
    *,
    listening: bool,
    ready: bool,
    disk_ready: bool,
    error: str | None,
) -> tuple[str, str, str]:
    if ready:
        return (
            "live",
            "Ishlayapti",
            "Model xotirada yuklangan — qayta ishga tushirish shart emas.",
        )
    if listening and not disk_ready:
        return (
            "weights_missing",
            "Og‘irlik yo‘q",
            "Worker ochiq, lekin model fayllari to‘liq emas. Avval yuklab oling.",
        )
    if listening:
        return (
            "listening",
            "Yuklanmoqda",
            "Worker ochiq, model hali xotiraga tushmagan. Birinchi so‘rovda yuklanadi.",
        )
    if not disk_ready:
        return (
            "weights_missing",
            "Og‘irlik yo‘q",
            "Model diskda topilmadi yoki yarim yuklangan.",
        )
    hint = "Worker o‘chiq. Oynadagi «Ishga tushirish» tugmasini bosing."
    if error:
        hint = f"{hint} {error}"
    return "down", "O‘chiq", hint


def probe_worker(spec: dict[str, Any]) -> dict[str, Any]:
    """Bitta worker: port, disk, model xotirada ekanligi."""
    disk_ready = _disk_ready(spec["id"])
    pid = _pid_alive(spec["pid_file"])
    listening = False
    ready = False
    error: str | None = None
    remote: dict[str, Any] = {}
    mode = "in-process" if not USE_MODEL_SERVERS else "microservice"

    if not USE_MODEL_SERVERS:
        remote = _in_process_status(spec["id"])
        listening = True
        ready = bool(remote.get("ready"))
        error = remote.get("error")
    else:
        try:
            _http_get_json(f"{spec['url']}/health")
            listening = True
        except Exception as exc:
            error = str(exc)
        if listening:
            try:
                remote = _http_get_json(f"{spec['url']}/status")
                ready = bool(remote.get("ready"))
                if remote.get("error"):
                    error = str(remote["error"])
            except Exception as exc:
                error = str(exc)
                remote = {}

    state, state_label, hint = _classify(
        listening=listening,
        ready=ready,
        disk_ready=disk_ready,
        error=error,
    )
    return {
        "id": spec["id"],
        "label": spec["label"],
        "chip_label": spec.get("chip_label") or spec["label"],
        "role": spec["role"],
        "port": spec["port"],
        "url": spec["url"],
        "mode": mode,
        "state": state,
        "state_label": state_label,
        "listening": listening,
        "ready": ready,
        "disk_ready": disk_ready,
        "pid": pid,
        "backend": remote.get("backend"),
        "device": remote.get("device"),
        "model_key": remote.get("model_key"),
        "error": error,
        "hint": hint,
        "log_file": spec["log_file"],
        "start_cmd": spec["start_cmd"],
        "download_cmd": spec["download_cmd"],
        "start_all_cmd": "./scripts/start_workers.sh",
    }


async def collect_worker_health() -> list[dict[str, Any]]:
    """Uch workerini parallel tekshiradi (UI polling uchun)."""
    return list(
        await asyncio.gather(
            *(asyncio.to_thread(probe_worker, spec) for spec in WORKER_SPECS)
        )
    )


def health_summary(workers: list[dict[str, Any]]) -> dict[str, Any]:
    live = sum(1 for w in workers if w.get("state") == "live")
    return {
        "workers": workers,
        "live": live,
        "total": len(workers),
        "all_live": live == len(workers) and live > 0,
    }
