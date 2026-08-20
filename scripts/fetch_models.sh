#!/usr/bin/env bash
# LLM va ASR modellarni mos papkalarga yuklash / davom ettirish.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1
mkdir -p "models/LLM modellar/Gemma 4e" "models/LLM modellar/Gemma 26" \
  "models/ASR modellar/GigaAM-Multilingual" "models/tarjima_model"
exec .venv/bin/python -u scripts/download_models.py "$@"
