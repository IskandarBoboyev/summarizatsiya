"""Lokal mikroserverlarga JSON so'rov (stdlib, qo'shimcha paket yo'q)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any


def _ensure_worker_for_url(url: str) -> None:
    """Port o'chiq bo'lsa, mos workerni qayta yoqadi va /health ni kutadi."""
    try:
        from backend.services.model_health import WORKER_SPECS
        from backend.services.worker_control import start_worker
    except Exception:
        return
    for spec in WORKER_SPECS:
        if str(spec.get("url") or "") not in url:
            continue
        try:
            start_worker(spec["id"])
        except Exception:
            return
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                urllib.request.urlopen(f"{spec['url']}/health", timeout=1.5)
                return
            except Exception:
                time.sleep(0.8)
        return


def rpc_json(
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    method: str = "POST",
    timeout: float = 900,
) -> dict[str, Any]:
    """GET/POST JSON. Server o'chiq bo'lsa qayta urinadi va workerni yoqadi."""
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last: Exception | None = None
    for attempt in range(5):
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
            if not raw:
                return {}
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"result": parsed}
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:800]
            raise RuntimeError(f"Model server {exc.code}: {body or exc.reason}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last = exc
            if attempt == 0:
                _ensure_worker_for_url(url)
            else:
                time.sleep(1.2 * attempt)
    raise RuntimeError(
        f"Model serverga ulanilmadi ({url}). "
        "Avval `./scripts/start_workers.sh` ni ishga tushiring."
    ) from last
