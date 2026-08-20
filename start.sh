#!/usr/bin/env bash
# Gemma + GigaAM + NLLB workerlar, so'ng UI FastAPI.
set -euo pipefail
cd "$(dirname "$0")"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export DOC_USE_MODEL_SERVERS="${DOC_USE_MODEL_SERVERS:-1}"

if [ -x ".venv/bin/python" ] && .venv/bin/python -c "import uvicorn" 2>/dev/null; then
  bash scripts/start_workers.sh
  echo "UI: http://127.0.0.1:8000  (fon rejimida, terminal yopilsa ham qoladi)"
  exit 0
fi

if command -v uv >/dev/null 2>&1; then
  [ -x ".venv/bin/python" ] || uv venv --python 3.12
  uv pip install -q fastapi "uvicorn[standard]" jinja2 python-multipart pydantic
  exec uv run uvicorn backend.main:app --host 127.0.0.1 --port 8000
fi

echo "Python venv tayyor emas — Ruby UI: http://127.0.0.1:8000"
exec /usr/bin/ruby serve.rb
