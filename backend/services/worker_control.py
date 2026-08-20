"""Gemma / GigaAM / NLLB mikroserverlarini UI dan yoqish va o‘chirish."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import Any

from backend.config import PROJECT_ROOT, USE_MODEL_SERVERS
from backend.services.model_health import WORKER_SPECS, probe_worker

_SPEC_BY_ID = {s["id"]: s for s in WORKER_SPECS}


def worker_spec(worker_id: str) -> dict[str, Any]:
    spec = _SPEC_BY_ID.get(worker_id)
    if spec is None:
        raise KeyError(worker_id)
    return spec


def _pids_on_port(port: int) -> list[int]:
    try:
        out = subprocess.check_output(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    pids: list[int] = []
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def _port_busy(port: int) -> bool:
    return bool(_pids_on_port(port))


def _signal_pid(pid: int, sig: int) -> None:
    try:
        os.killpg(os.getpgid(pid), sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def start_worker(worker_id: str) -> dict[str, Any]:
    if not USE_MODEL_SERVERS:
        raise RuntimeError("Modellar asosiy jarayonda — alohida worker yo‘q")
    spec = worker_spec(worker_id)
    if _port_busy(spec["port"]):
        return {"ok": True, "already": True, "action": "start", "worker": probe_worker(spec)}

    py = PROJECT_ROOT / ".venv" / "bin" / "python"
    if not py.is_file():
        raise RuntimeError(f"venv topilmadi: {py}")

    log_path = PROJECT_ROOT / spec["log_file"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("MODEL_CACHE_DIR", str(PROJECT_ROOT / "models" / "OCR"))
    with log_path.open("ab") as log:
        proc = subprocess.Popen(
            [
                str(py),
                "-m",
                "uvicorn",
                spec["module"],
                "--host",
                "127.0.0.1",
                "--port",
                str(spec["port"]),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            cwd=str(PROJECT_ROOT),
            env=env,
        )
    spec["pid_file"].write_text(str(proc.pid), encoding="utf-8")
    time.sleep(0.4)
    return {
        "ok": True,
        "already": False,
        "action": "start",
        "pid": proc.pid,
        "worker": probe_worker(spec),
    }


def stop_worker(worker_id: str) -> dict[str, Any]:
    if not USE_MODEL_SERVERS:
        raise RuntimeError("Modellar asosiy jarayonda — alohida worker yo‘q")
    spec = worker_spec(worker_id)
    pids = set(_pids_on_port(spec["port"]))
    try:
        stored = int(spec["pid_file"].read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        stored = None
    if stored:
        pids.add(stored)
    if not pids:
        return {"ok": True, "already": True, "action": "stop", "worker": probe_worker(spec)}

    for pid in pids:
        _signal_pid(pid, signal.SIGTERM)

    deadline = time.time() + 8
    while time.time() < deadline and _port_busy(spec["port"]):
        time.sleep(0.2)

    if _port_busy(spec["port"]):
        for pid in set(_pids_on_port(spec["port"])) | pids:
            _signal_pid(pid, signal.SIGKILL)
        time.sleep(0.2)

    try:
        spec["pid_file"].unlink()
    except OSError:
        pass
    return {"ok": True, "already": False, "action": "stop", "worker": probe_worker(spec)}
