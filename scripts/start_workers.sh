#!/usr/bin/env bash
# Gemma (8001), GigaAM (8002), NLLB (8003), Surya (8004) va UI (8000).
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export DOC_USE_MODEL_SERVERS="${DOC_USE_MODEL_SERVERS:-1}"
export MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$PWD/models/OCR}"
mkdir -p data models/OCR
PY="${PWD}/.venv/bin/python"

port_free() {
  ! lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}

# Cursor shell tugaganda ham o'lmasin — yangi session (macOS da setsid yo'q).
start_detached() {
  local logfile="$1" pidfile="$2"
  shift 2
  "$PY" -c "
import subprocess, sys
log = open(sys.argv[1], 'ab')
p = subprocess.Popen(
    sys.argv[4:] if sys.argv[3] == '--' else sys.argv[3:],
    stdout=log,
    stderr=subprocess.STDOUT,
    stdin=subprocess.DEVNULL,
    start_new_session=True,
)
open(sys.argv[2], 'w').write(str(p.pid))
print(p.pid)
" "$logfile" "$pidfile" -- "$@"
}

start_worker() {
  local port="$1" name="$2" module="$3" logfile="$4" pidfile="$5"
  if port_free "$port"; then
    pid=$(start_detached "$logfile" "$pidfile" \
      "$PY" -m uvicorn "$module" --host 127.0.0.1 --port "$port")
    echo "$name worker: http://127.0.0.1:$port  (pid $pid)"
  else
    echo "$name worker allaqachon $port da"
  fi
}

if [ ! -x "$PY" ]; then
  echo "venv topilmadi: $PY"
  exit 1
fi

start_worker 8001 "Gemma" backend.workers.gemma_app:app data/gemma_worker.log data/gemma_worker.pid
start_worker 8002 "GigaAM" backend.workers.gigaam_app:app data/gigaam_worker.log data/gigaam_worker.pid
start_worker 8003 "NLLB" backend.workers.nllb_app:app data/nllb_worker.log data/nllb_worker.pid
start_worker 8004 "Surya" backend.workers.surya_app:app data/surya_worker.log data/surya_worker.pid
# UI ham alohida sessionda — terminal/Cursor yopilsa o'lmasin
start_worker 8000 "UI" backend.main:app data/ui_server.log data/ui_server.pid
