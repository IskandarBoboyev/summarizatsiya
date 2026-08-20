#!/usr/bin/env bash
# Gemma 4e (E4B Instruct) — MLX 4-bit → models/LLM modellar/Gemma 4e
set -euo pipefail
cd "$(dirname "$0")/.."
DEST="models/LLM modellar/Gemma 4e"
mkdir -p "$DEST"

WEIGHT="$DEST/model.safetensors"
if [ -f "$DEST/config.json" ] && [ -f "$WEIGHT" ]; then
  SIZE="$(stat -f%z "$WEIGHT" 2>/dev/null || stat -c%s "$WEIGHT")"
  if [ "$SIZE" -gt 4000000000 ]; then
    echo "Allaqachon bor: $DEST"
    exit 0
  fi
fi

echo "Yuklanmoqda Gemma 4e (MLX 4-bit) → $DEST"
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
.venv/bin/python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="mlx-community/gemma-4-e4b-it-4bit",
    local_dir=r"""$DEST""",
)
print("Tayyor:", r"""$DEST""")
PY
