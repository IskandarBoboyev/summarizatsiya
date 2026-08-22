#!/usr/bin/env python3
"""LLM/ASR modellarni barqaror .incomplete fayldan davom ettirib yuklash."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
from huggingface_hub.file_download import http_get

ROOT = Path(__file__).resolve().parent.parent
JOBS = [
    {
        "label": "GigaAM-Multilingual",
        "repo": "ai-sage/GigaAM-Multilingual",
        "dest": ROOT / "models" / "ASR modellar" / "GigaAM-Multilingual",
        "revision": "ctc",
        "min_weight": 50_000_000,
    },
    {
        "label": "Gemma 4e",
        "repo": "mlx-community/gemma-4-e4b-it-4bit",
        "dest": ROOT / "models" / "LLM modellar" / "Gemma 4e",
        "revision": None,
        "min_weight": 4_000_000_000,
    },
    {
        "label": "Gemma 26",
        "repo": "mlx-community/gemma-4-26b-a4b-it-4bit",
        "dest": ROOT / "models" / "LLM modellar" / "Gemma 26",
        "revision": None,
        "min_weight": 10_000_000_000,
    },
    {
        "label": "NLLB-200 1.3B",
        "repo": "facebook/nllb-200-distilled-1.3B",
        "dest": ROOT / "models" / "tarjima_model",
        "revision": None,
        "min_weight": 1_000_000_000,
        "aliases": {"nllb", "nllb-200", "tarjima"},
    },
    {
        "label": "TranslateGemma 4B",
        "repo": "mlx-community/translategemma-4b-it-4bit",
        "dest": ROOT / "models" / "tarjima_model" / "TranslateGemma",
        "revision": None,
        "min_weight": 400_000_000,
        "aliases": {"translategemma", "translate-gemma", "tgemma", "translategemma-4b"},
    },
    {
        "label": "SeamlessM4T v2",
        "repo": "facebook/seamless-m4t-v2-large",
        "dest": ROOT / "models" / "ASR modellar" / "SeamlessM4T-v2",
        "revision": None,
        "min_weight": 4_000_000_000,
        "aliases": {"seamless", "seamlessm4t", "seamless-m4t", "seamless-m4t-v2", "m4t"},
    },
    {
        "label": "Surya OCR",
        "kind": "surya_s3",
        "dest": ROOT / "models" / "OCR",
        "aliases": {"surya", "surya-ocr", "ocr"},
        "parts": [
            "text_detection/2025_05_07",
            "text_recognition/2025_08_29",
        ],
    },
]


def _fmt(n: int) -> str:
    return f"{n / (1024 ** 3):.2f} GB" if n >= 1024**3 else f"{n / (1024 ** 2):.1f} MB"


def _weights(dest: Path) -> list[Path]:
    return [p for p in dest.iterdir() if p.suffix in {".safetensors", ".bin", ".pt", ".ckpt"}]


def _adopt_leftovers(dest: Path, filename: str, etag: str | None) -> Path:
    stable = dest / f"{filename}.incomplete"
    if stable.exists() and stable.stat().st_size > 0:
        return stable
    cache = dest / ".cache" / "huggingface" / "download"
    candidates: list[Path] = []
    if cache.is_dir():
        if etag:
            candidates.extend(cache.glob(f"*.{etag}.*.incomplete"))
            candidates.extend(cache.glob(f"*.{etag}.incomplete"))
        if filename.endswith(".safetensors") and not candidates:
            candidates.extend(p for p in cache.glob("*.incomplete") if p.stat().st_size > 50_000_000)
    if not candidates:
        return stable
    best = max(candidates, key=lambda p: p.stat().st_size)
    if best.stat().st_size <= 0:
        return stable
    best_size = best.stat().st_size
    print(f"  qoldiqdan davom: {best.name} ({_fmt(best_size)})", flush=True)
    if stable.exists():
        stable.unlink()
    best.rename(stable)
    for leftover in candidates:
        if leftover == best or not leftover.exists():
            continue
        leftover.unlink(missing_ok=True)
    return stable


def _curl_resume(url: str, dest: Path, expected: int | None) -> None:
    """Katta fayllar uchun curl: yangi URL + davom ettirish, muddati o'tgan CDN yo'q."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl",
        "-fL",
        "--retry",
        "12",
        "--retry-delay",
        "4",
        "--retry-all-errors",
        "--connect-timeout",
        "30",
        "--speed-time",
        "60",
        "--speed-limit",
        "1024",
        "-C",
        "-",
        "-o",
        str(dest),
        url,
    ]
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise OSError(f"curl {proc.returncode}: {url}")
    if expected is not None and dest.exists() and dest.stat().st_size != expected:
        raise OSError(f"curl: kutilgan {expected}, olindi {dest.stat().st_size}")


def download_file(repo: str, filename: str, dest: Path, revision: str | None) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / filename
    last_err: Exception | None = None
    for attempt in range(1, 9):
        url = hf_hub_url(repo_id=repo, filename=filename, revision=revision)
        try:
            meta = get_hf_file_metadata(url)
        except Exception as exc:
            last_err = exc
            print(f"  metadata xato ({attempt}/8): {exc}", flush=True)
            time.sleep(min(30, 2 * attempt))
            continue
        expected = meta.size
        if target.is_file() and (expected is None or target.stat().st_size == expected):
            print(f"  bor: {filename} ({_fmt(target.stat().st_size)})", flush=True)
            return
        if target.is_file():
            target.unlink()

        incomplete = _adopt_leftovers(dest, filename, meta.etag)
        resume = incomplete.stat().st_size if incomplete.exists() else 0
        if expected and resume > expected:
            incomplete.unlink()
            resume = 0
        print(
            f"  yuklash: {filename}  {_fmt(resume)} / {_fmt(expected or 0)}  urinish {attempt}/8",
            flush=True,
        )
        incomplete.parent.mkdir(parents=True, exist_ok=True)
        try:
            if expected and expected >= 20_000_000:
                _curl_resume(url, incomplete, expected)
            else:
                with incomplete.open("ab" if resume else "wb") as fh:
                    http_get(
                        url,
                        fh,
                        resume_size=resume,
                        expected_size=expected,
                        displayed_filename=filename,
                    )
            if expected is not None and incomplete.stat().st_size != expected:
                raise OSError(f"{filename}: kutilgan {expected}, olindi {incomplete.stat().st_size}")
            incomplete.replace(target)
            print(f"  tayyor: {filename} ({_fmt(target.stat().st_size)})", flush=True)
            return
        except Exception as exc:
            last_err = exc
            print(f"  uzildi ({attempt}/8): {exc}", flush=True)
            time.sleep(min(30, 2 * attempt))
    raise OSError(f"{filename} yuklanmadi: {last_err}")


def repo_ready(dest: Path, min_weight: int) -> bool:
    if not (dest / "config.json").is_file():
        return False
    return any(p.stat().st_size >= min_weight for p in _weights(dest))


def run_surya_s3(job: dict) -> None:
    dest: Path = job["dest"]
    dest.mkdir(parents=True, exist_ok=True)
    os.environ["MODEL_CACHE_DIR"] = str(dest)
    base = "https://models.datalab.to"

    for part in job["parts"]:
        local = dest / part
        local.mkdir(parents=True, exist_ok=True)
        manifest_url = f"{base}/{part}/manifest.json"
        manifest_path = local / "manifest.json"
        if not manifest_path.is_file():
            print(f"  manifest: {part}", flush=True)
            _curl_resume(manifest_url, manifest_path, None)
        files = json.loads(manifest_path.read_text(encoding="utf-8")).get("files") or []
        missing = [name for name in files if not (local / name).is_file() or (local / name).stat().st_size == 0]
        if not missing:
            print(f"  bor: {part}", flush=True)
            continue
        print(f"  yuklanmoqda: {part} ({len(missing)} fayl)", flush=True)
        for name in missing:
            target = local / name
            print(f"    {name}", flush=True)
            _curl_resume(f"{base}/{part}/{name}", target, None)
        print(f"  tayyor: {part}", flush=True)
    print(f"Tayyor: {dest}", flush=True)


def run_job(job: dict) -> None:
    if job.get("kind") == "surya_s3":
        run_surya_s3(job)
        return
    dest: Path = job["dest"]
    dest.mkdir(parents=True, exist_ok=True)
    if repo_ready(dest, job["min_weight"]):
        print(f"Allaqachon bor: {dest}", flush=True)
        return
    print(f"Yuklanmoqda {job['label']} → {dest}", flush=True)
    api = HfApi()
    files = api.list_repo_files(job["repo"], revision=job["revision"])
    skip = {".gitattributes"}
    has_sft = any(
        name == "model.safetensors" or (name.startswith("model-") and name.endswith(".safetensors"))
        for name in files
    )
    wanted = []
    for name in files:
        if name in skip or name.endswith(".lock") or name.endswith(".md"):
            continue
        if has_sft and name.endswith((".bin", ".pt", ".ckpt")):
            continue
        wanted.append(name)
    # Avval kichik fayllar (tokenizer), keyin og'irlik
    wanted.sort(key=lambda n: (n.endswith((".bin", ".safetensors", ".pt", ".ckpt")), n))
    for name in wanted:
        download_file(job["repo"], name, dest, job["revision"])
    print(f"Tayyor: {dest}", flush=True)


def _job_matches(job: dict, needle: str) -> bool:
    aliases = {job["label"].lower(), *(job.get("aliases") or set())}
    if job.get("repo"):
        aliases.add(str(job["repo"]).lower())
    aliases.add(str(job["dest"].name).lower())
    return needle.lower() in aliases or any(needle.lower() in a for a in aliases)


def main() -> int:
    wanted = [a.strip() for a in sys.argv[1:] if a.strip()]
    jobs = JOBS
    if wanted:
        jobs = [job for job in JOBS if any(_job_matches(job, w) for w in wanted)]
        if not jobs:
            print(f"Hech qaysi vazifa mos kelmadi: {wanted}", flush=True)
            return 1
    for job in jobs:
        run_job(job)
    return 0


if __name__ == "__main__":
    sys.exit(main())
