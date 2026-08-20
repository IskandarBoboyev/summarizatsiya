#!/usr/bin/env bash
# Gemma (8001), GigaAM (8002), NLLB (8003), Surya (8004) va UI (8000) holati.
set -u
cd "$(dirname "$0")/.."

probe() {
  local name="$1" url="$2"
  local health status
  if ! health=$(curl -sS --max-time 2 "$url/health" 2>/dev/null); then
    printf "%-8s  O‘chiq     %s\n" "$name" "$url"
    return
  fi
  status=$(curl -sS --max-time 2 "$url/status" 2>/dev/null || true)
  if echo "$status" | grep -Eq '"ready"[[:space:]]*:[[:space:]]*true'; then
    printf "%-8s  Ishlayapti %s\n" "$name" "$url"
  else
    printf "%-8s  Yuklanmoqda %s\n" "$name" "$url"
  fi
}

echo "Modellar holati"
echo "---------------"
probe "Gemma"  "http://127.0.0.1:8001"
probe "GigaAM" "http://127.0.0.1:8002"
probe "NLLB"   "http://127.0.0.1:8003"
probe "Surya"  "http://127.0.0.1:8004"

if curl -sS --max-time 2 -o /dev/null "http://127.0.0.1:8000/api/models/health"; then
  echo
  echo "Yig‘ma (UI):"
  curl -sS --max-time 3 "http://127.0.0.1:8000/api/models/health"
  echo
else
  echo
  echo "UI (8000) ochiq emas — ./start.sh"
fi
