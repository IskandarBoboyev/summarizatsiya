"""
Loyiha konfiguratsiyasi.

Vazifalar:
    - Ishga tushishda tashqi tarmoqni o'chirish (Hugging Face offline).
    - Qurilmani avtomatik aniqlash: CUDA -> MPS -> CPU.
    - Model, kesh va kuzatiladigan papkalar yo'llarini belgilash.
    - Xotira yetishmovchiligini oldini olish uchun kvantizatsiya va chunk o'lchamlarini tanlash.
"""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# 100% OFFLINE: har qanday HF / tashqi so'rovni bloklash
# Bu o'zgaruvchilar boshqa kutubxonalar import qilinishidan OLDIN o'rnatiladi.
# ---------------------------------------------------------------------------
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_EXPERIMENTAL_WARNING", "1")
# Bitsandbytes va boshqa paketlar internetga chiqmasin
os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")

# Loyiha ildiz papkasi: local_doc_platform/
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

BACKEND_DIR: Path = PROJECT_ROOT / "backend"
FRONTEND_DIR: Path = PROJECT_ROOT / "frontend"
MODELS_DIR: Path = PROJECT_ROOT / "models"
LLM_MODELS_DIR: Path = MODELS_DIR / "LLM modellar"
ASR_MODELS_DIR: Path = MODELS_DIR / "ASR modellar"
TRANSLATION_MODELS_DIR: Path = MODELS_DIR / "tarjima_model"
WATCHED_DIR: Path = PROJECT_ROOT / "watched_folders"
DATA_DIR: Path = PROJECT_ROOT / "data"

# Hugging Face mahalliy keshi — faqat diskdan o'qiladi
HF_CACHE_DIR: Path = MODELS_DIR / "hf_cache"
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR / "transformers"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR / "hub"))

# SQLite bazasi
DATABASE_PATH: Path = DATA_DIR / "documents.db"

# Tesseract til paketlari (offline)
TESSDATA_DIR: Path = MODELS_DIR / "tessdata"
if TESSDATA_DIR.exists():
    os.environ.setdefault("TESSDATA_PREFIX", str(TESSDATA_DIR))

# EasyOCR modellari (offline, download o'chiq)
EASYOCR_DIR: Path = MODELS_DIR / "easyocr"
# Surya OCR og'irliklari
OCR_DIR: Path = MODELS_DIR / "OCR"
os.environ.setdefault("MODEL_CACHE_DIR", str(OCR_DIR))

DeviceType = Literal["cuda", "mps", "cpu"]
QuantType = Literal["none", "8bit", "4bit"]


def _try_import_torch():
    """Torch mavjud bo'lsa import qiladi, aks holda None qaytaradi."""
    try:
        import torch  # noqa: WPS433

        return torch
    except Exception:
        return None


def detect_device() -> DeviceType:
    """
    Mavjud tezlatgichni avtomatik aniqlaydi.

    Tartib (ustuvorlik):
        1. NVIDIA CUDA (Linux / Windows, masalan RTX 3070)
        2. Apple Silicon MPS (M1/M2/M3/M4, Unified Memory)
        3. CPU (zaxira variant)

    Returns:
        "cuda" | "mps" | "cpu"
    """
    torch = _try_import_torch()
    if torch is None:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    # Apple Silicon: MPS backend
    mps_ok = getattr(torch.backends, "mps", None)
    if mps_ok is not None and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"

    return "cpu"


def detect_memory_bytes() -> int:
    """
    Qurilma uchun taxminiy mavjud xotira (baytlarda).

    - CUDA: GPU VRAM
    - MPS: tizim Unified Memory (psutil orqali)
    - CPU: tizim RAM
    """
    torch = _try_import_torch()
    device = detect_device()

    if device == "cuda" and torch is not None:
        try:
            props = torch.cuda.get_device_properties(0)
            return int(props.total_memory)
        except Exception:
            return 8 * 1024**3  # RTX 3070 ga yaqin default

    try:
        import psutil  # noqa: WPS433

        return int(psutil.virtual_memory().total)
    except Exception:
        # Apple Silicon 48GB yoki oddiy PC uchun xavfsiz default
        if platform.system() == "Darwin":
            return 48 * 1024**3
        return 16 * 1024**3


def recommend_quantization(device: DeviceType | None = None, model_key: str = "gemma-4") -> QuantType:
    """
    Qurilma va model o'lchamiga qarab kvantizatsiya darajasini tavsiya qiladi.

    Qoidalar:
        - CUDA + VRAM <= 10GB (RTX 3070 8GB): katta model uchun 4-bit, kichik uchun 8-bit.
        - CUDA + VRAM ko'proq: 8-bit yoki kvantizatsiyasiz.
        - MPS: bitsandbytes ishlamaydi — fp16 (none). Katta model GGUF orqali yuklanadi.
        - CPU: 8-bit yoki GGUF (servis qatlamida hal qilinadi).

    Args:
        device: Qo'lda berilgan qurilma. None bo'lsa avtomatik aniqlanadi.
        model_key: "gemma-4" yoki "gemma-26".

    Returns:
        "none" | "8bit" | "4bit"
    """
    device = device or detect_device()
    mem = detect_memory_bytes()
    is_large = model_key in {"gemma-26", "gemma-27", "gemma-27b"}

    if device == "cuda":
        vram_gb = mem / (1024**3)
        if vram_gb <= 10:
            return "4bit" if is_large else "8bit"
        if vram_gb <= 20 and is_large:
            return "8bit"
        return "none"

    if device == "mps":
        # MPS da bitsandbytes yo'q. Katta modelni GGUF bilan ishlatish tavsiya etiladi.
        return "none"

    # CPU
    return "8bit" if is_large else "none"


def empty_accelerator_cache() -> None:
    """
    GPU / MPS keshini tozalaydi. OOM xavfini kamaytirish uchun
    modelni almashtirish yoki katta fayldan keyin chaqiriladi.
    """
    torch = _try_import_torch()
    if torch is None:
        return

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

    try:
        if hasattr(torch, "mps") and detect_device() == "mps":
            torch.mps.empty_cache()
    except Exception:
        pass


@dataclass
class ModelSpec:
    """Bitta LLM modelining lokal joylashuvi va ko'rinadigan nomi."""

    key: str
    label: str
    transformers_dir: Path
    gguf_globs: list[str] = field(default_factory=list)


# Gemma 4 E va Gemma 26 — models/LLM modellar/
LLM_MODELS: dict[str, ModelSpec] = {
    "gemma-4": ModelSpec(
        key="gemma-4",
        label="Gemma 4e",
        transformers_dir=LLM_MODELS_DIR / "Gemma 4e",
        gguf_globs=[
            "gemma-4*.gguf",
            "gemma-4-E4B*.gguf",
            "*gemma*E4B*.gguf",
            "gemma*4b*.gguf",
            "gemma-3-4b*.gguf",
            "gemma4/*.gguf",
            "gemma-4/*.gguf",
            "**/*gemma*4*.gguf",
        ],
    ),
    "gemma-26": ModelSpec(
        key="gemma-26",
        label="Gemma 26",
        transformers_dir=LLM_MODELS_DIR / "Gemma 26",
        gguf_globs=[
            "gemma-26*.gguf",
            "gemma-27*.gguf",
            "gemma*27b*.gguf",
            "gemma*26*.gguf",
            "gemma-26/*.gguf",
            "gemma-27/*.gguf",
        ],
    ),
}

# ASR: ai-sage/GigaAM-Multilingual — models/ASR modellar/
GIGAAM_DIR: Path = ASR_MODELS_DIR / "GigaAM-Multilingual"
NLLB_DIR: Path = TRANSLATION_MODELS_DIR


@dataclass
class AsrSpec:
    """Bitta ASR modelining lokal joylashuvi va ko'rinadigan nomi."""

    key: str
    label: str
    search_dirs: list[Path] = field(default_factory=list)
    hub_glob: str = ""


ASR_MODELS: dict[str, AsrSpec] = {
    "gigaam-multilingual": AsrSpec(
        key="gigaam-multilingual",
        label="GigaAM Multilingual",
        search_dirs=[
            GIGAAM_DIR,
            GIGAAM_DIR / "ai-sage" / "GigaAM-Multilingual",
        ],
        hub_glob="models--ai-sage--GigaAM-Multilingual/snapshots/*/config.json",
    ),
}


# Surya 0.16 S3 checkpointlari (models.datalab.to)
SURYA_DET_CHECKPOINT: str = "text_detection/2025_05_07"
SURYA_REC_CHECKPOINT: str = "text_recognition/2025_08_29"

OCR_MODELS: dict[str, dict[str, str]] = {
    "surya": {"key": "surya", "label": "Surya OCR"},
    "gemma": {"key": "gemma", "label": "Gemma"},
}


@dataclass
class TranslationSpec:
    """Bitta tarjima modelining lokal joylashuvi."""

    key: str
    label: str
    search_dirs: list[Path] = field(default_factory=list)


TRANSLATION_MODELS: dict[str, TranslationSpec] = {
    "nllb-200": TranslationSpec(
        key="nllb-200",
        label="NLLB-200 1.3B",
        search_dirs=[NLLB_DIR],
    ),
}

COUNTRIES: list[dict[str, str]] = [
    {"key": "kz", "label": "Qozogʻiston"},
    {"key": "uz", "label": "Oʻzbekiston"},
    {"key": "tj", "label": "Tojikiston"},
    {"key": "kg", "label": "Qirgʻiziston"},
    {"key": "af", "label": "Afgʻoniston"},
    {"key": "tm", "label": "Turkmaniston"},
]
COUNTRY_KEYS: set[str] = {c["key"] for c in COUNTRIES}

# ASR tillari (ISO 639-1). GigaAM-Multilingual qo'llab-quvvatlaydigan asosiy tillar.
ASR_LANGUAGES: dict[str, str] = {
    "auto": "Avtomatik (eshitilgan tilda)",
    "uz": "O'zbekcha",
    "ru": "Ruscha",
    "en": "Inglizcha",
    "kk": "Qozoqcha",
    "ky": "Qirg'izcha",
    "tg": "Tojikcha",
    "tr": "Turkcha",
    "ar": "Arabcha",
    "zh": "Xitoycha",
    "de": "Nemischa",
    "fr": "Fransuzcha",
    "es": "Ispancha",
}

# Qo'llab-quvvatlanadigan kengaytmalar
DOCUMENT_EXTENSIONS: set[str] = {".pdf", ".docx", ".txt", ".md", ".rtf"}
IMAGE_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
AUDIO_EXTENSIONS: set[str] = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".wma"}
VIDEO_EXTENSIONS: set[str] = {".mp4", ".mkv", ".mov", ".webm", ".avi"}
SUPPORTED_EXTENSIONS: set[str] = (
    DOCUMENT_EXTENSIONS | IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
)

# Skaner hech qachon shu papkalarga kirmasligi kerak
SKIP_SCAN_DIR_NAMES: set[str] = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".cursor",
    "site-packages",
    ".mypy_cache",
}


def is_app_internal_path(raw: str | Path) -> bool:
    """Loyiha kodi / venv / modellarni hujjat sifatida olmaslik."""
    try:
        target = Path(raw).expanduser().resolve()
        rel = target.relative_to(PROJECT_ROOT.resolve())
    except (OSError, ValueError):
        return False
    parts = rel.parts
    if not parts:
        return True
    if parts[0] == "watched_folders":
        return False
    if len(parts) >= 2 and parts[0] == "data" and parts[1] == "uploads":
        return False
    return True

# Xotira tejash: matn va audio bo'laklari
TEXT_CHUNK_CHARS: int = 5_000          # ~1.2–1.5k token
TEXT_CHUNK_OVERLAP: int = 250          # bo'laklar orasida qoplama
MAX_CHUNKS_PER_FILE: int = 24          # juda katta fayllarni cheklash
AUDIO_CHUNK_SECONDS: float = 25.0      # ASR uchun audio bo'lagi
AUDIO_SAMPLE_RATE: int = 16_000

# LLM generatsiya parametrlari (xotira va tezlik muvozanati)
LLM_MAX_NEW_TOKENS_SUMMARY: int = 384
LLM_MAX_NEW_TOKENS_TRANSLATE: int = 1024
TRANSLATE_CHUNK_CHARS: int = 600
TRANSLATE_MAX_CHUNKS: int = 10_000
LLM_MAX_NEW_TOKENS_CHAT: int = 512
LLM_CONTEXT_SIZE: int = 4096
RAG_CHUNK_CHARS: int = 900
RAG_TOP_K: int = 4
LLM_TEMPERATURE: float = 0.2
TRANSLATE_TEMPERATURE: float = 0.1
SUMMARY_TEMPERATURE: float = 0.1

# Navbat va UI
QUEUE_POLL_INTERVAL: float = 0.4
DOCUMENTS_PAGE_SIZE: int = 12
WATCHDOG_DEBOUNCE_SECONDS: float = 1.2

# Server
HOST: str = os.environ.get("DOC_PLATFORM_HOST", "127.0.0.1")
PORT: int = int(os.environ.get("DOC_PLATFORM_PORT", "8000"))
# Gemma / GigaAM / NLLB alohida doimiy mikroserverlar
LLM_SERVER_URL: str = os.environ.get("DOC_LLM_URL", "http://127.0.0.1:8001").rstrip("/")
ASR_SERVER_URL: str = os.environ.get("DOC_ASR_URL", "http://127.0.0.1:8002").rstrip("/")
TRANSLATION_SERVER_URL: str = os.environ.get("DOC_TRANSLATION_URL", "http://127.0.0.1:8003").rstrip("/")
OCR_SERVER_URL: str = os.environ.get("DOC_OCR_URL", "http://127.0.0.1:8004").rstrip("/")
USE_MODEL_SERVERS: bool = os.environ.get("DOC_USE_MODEL_SERVERS", "1") not in {"0", "false", "no"}
LLM_SERVER_PORT: int = int(os.environ.get("DOC_LLM_PORT", "8001"))
ASR_SERVER_PORT: int = int(os.environ.get("DOC_ASR_PORT", "8002"))
TRANSLATION_SERVER_PORT: int = int(os.environ.get("DOC_TRANSLATION_PORT", "8003"))
OCR_SERVER_PORT: int = int(os.environ.get("DOC_OCR_PORT", "8004"))
MODEL_RPC_TIMEOUT: float = float(os.environ.get("DOC_MODEL_RPC_TIMEOUT", "900"))

# Standart sozlamalar
DEFAULT_LLM_MODEL: str = "gemma-4"
DEFAULT_ASR_MODEL: str = "gigaam-multilingual"
DEFAULT_TRANSLATION_MODEL: str = "nllb-200"
DEFAULT_OCR_MODEL: str = "surya"
DEFAULT_ASR_LANGUAGE: str = "auto"
DEFAULT_QUANTIZATION: QuantType | None = None  # None = avtomatik


def ensure_runtime_dirs() -> None:
    """Ishga tushishda zarur papkalarni yaratadi (modellarsiz ham UI ishlaydi)."""
    for path in (
        MODELS_DIR,
        LLM_MODELS_DIR,
        ASR_MODELS_DIR,
        TRANSLATION_MODELS_DIR,
        LLM_MODELS_DIR / "Gemma 4e",
        LLM_MODELS_DIR / "Gemma 26",
        GIGAAM_DIR,
        NLLB_DIR,
        WATCHED_DIR,
        DATA_DIR,
        DATA_DIR / "uploads",
        HF_CACHE_DIR,
        EASYOCR_DIR,
        OCR_DIR,
        TESSDATA_DIR,
        MODELS_DIR / "bin",
        FRONTEND_DIR / "static" / "css",
        FRONTEND_DIR / "static" / "js",
        FRONTEND_DIR / "templates",
    ):
        path.mkdir(parents=True, exist_ok=True)
    ensure_ffmpeg_on_path()


def ensure_ffmpeg_on_path() -> str | None:
    """Lokal ffmpeg (models/bin) ni PATH ga qo‘yadi — GigaAM load_audio shu yerdan topadi."""
    bin_dir = MODELS_DIR / "bin"
    binary = bin_dir / "ffmpeg"
    if binary.is_file():
        prefix = str(bin_dir)
        current = os.environ.get("PATH", "")
        if prefix not in current.split(os.pathsep):
            os.environ["PATH"] = prefix + os.pathsep + current
        return str(binary)
    return shutil.which("ffmpeg")


# GGUF kv turlari (skip qilish uchun)
_GGUF_VAL_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
# ggml type_id -> (block_elems, bytes_per_block)
_GGML_BLOCK = {
    0: (1, 4),
    1: (1, 2),
    2: (32, 18),
    3: (32, 20),
    6: (32, 22),
    7: (32, 24),
    8: (32, 34),
    9: (32, 36),
    10: (256, 84),
    11: (256, 110),
    12: (256, 144),
    13: (256, 176),
    14: (256, 210),
    15: (256, 292),
    24: (1, 1),
    25: (1, 2),
    26: (1, 4),
    27: (1, 8),
    28: (1, 8),
    30: (1, 2),
}


def _gguf_tensors_fit(path: Path) -> bool:
    """
    GGUF headerdagi tensor offsetlari haqiqiy fayl ichida ekanini tekshiradi.

    llama-cli xatosi: "tensor data is not within the file bounds" —
    odatda fayl kesilgan (to'liq yuklanmagan).
    """
    import struct

    size = path.stat().st_size
    try:
        with path.open("rb") as fh:
            if fh.read(4) != b"GGUF":
                return False
            struct.unpack("<I", fh.read(4))  # version
            n_tensors, n_kv = struct.unpack("<QQ", fh.read(16))
            if n_tensors <= 0 or n_tensors > 100_000:
                return False

            def read_str() -> bytes:
                n = struct.unpack("<Q", fh.read(8))[0]
                return fh.read(n)

            def skip_val(typ: int) -> None:
                if typ == 8:
                    read_str()
                    return
                if typ == 9:
                    et = struct.unpack("<I", fh.read(4))[0]
                    n = struct.unpack("<Q", fh.read(8))[0]
                    for _ in range(n):
                        skip_val(et)
                    return
                n = _GGUF_VAL_SIZE.get(typ)
                if n is None:
                    raise ValueError(f"noma'lum GGUF turi: {typ}")
                fh.read(n)

            for _ in range(n_kv):
                read_str()
                typ = struct.unpack("<I", fh.read(4))[0]
                skip_val(typ)

            tensors: list[tuple[int, int]] = []
            for _ in range(n_tensors):
                read_str()
                n_dims = struct.unpack("<I", fh.read(4))[0]
                dims = [struct.unpack("<Q", fh.read(8))[0] for _ in range(n_dims)]
                ggml_t = struct.unpack("<I", fh.read(4))[0]
                offset = struct.unpack("<Q", fh.read(8))[0]
                ne = 1
                for d in dims:
                    ne *= d
                blk, bsz = _GGML_BLOCK.get(ggml_t, (1, 1))
                nbytes = (ne // blk) * bsz if blk else ne
                tensors.append((offset, nbytes))

            pos = fh.tell()
    except (OSError, struct.error, ValueError):
        return False

    alignment = 32
    data_start = (pos + alignment - 1) // alignment * alignment
    for offset, nbytes in tensors:
        end = data_start + offset + nbytes
        if end > size:
            return False
    return True


def _gguf_complete(path: Path) -> bool:
    """Yarim yuklangan yoki buzilgan GGUF ni model deb qabul qilmaslik."""
    name = path.name.lower()
    if path.suffix.lower() != ".gguf" or not path.is_file():
        return False
    if name.endswith(".part") or name.endswith(".incomplete"):
        return False
    size = path.stat().st_size
    # Gemma 4 E4B Q4_K_M (unsloth) ~4.98 GB. 4.2–4.5 GB — kesilgan fayl.
    if "e4b" in name and size < 4_700_000_000:
        return False
    if size < 1_000_000_000:
        return False
    return _gguf_tensors_fit(path)


def gguf_file_usable(path: Path) -> bool:
    """Tashqi chaqiriqlar uchun: GGUF to'liq va o'qishga yaroqlimi."""
    return _gguf_complete(path)


def find_gguf_file(spec: ModelSpec) -> Path | None:
    """
    Berilgan model uchun birinchi to'liq GGUF faylini qaytaradi.

    Qidiruv: models/ ildizi (rekursiv) va model papkasi ichida.
    """
    search_roots = [MODELS_DIR, spec.transformers_dir]
    seen: set[Path] = set()
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in spec.gguf_globs:
            for match in sorted(root.glob(pattern)):
                if match in seen or not _gguf_complete(match):
                    continue
                seen.add(match)
                return match
    # Oxirgi imkon: models/ ichidagi istalgan *gemma*.gguf
    if MODELS_DIR.exists() and spec.key.startswith("gemma"):
        leftovers = sorted(p for p in MODELS_DIR.rglob("*.gguf") if _gguf_complete(p))
        if spec.key == "gemma-4":
            for p in leftovers:
                name = p.name.lower()
                if "e4b" in name or "4b" in name or "gemma-4" in name:
                    return p
        if spec.key == "gemma-26":
            for p in leftovers:
                name = p.name.lower()
                if "26" in name or "27" in name:
                    return p
    return None


def find_llama_cli() -> Path | None:
    """
    Oldindan yig'ilgan llama.cpp binary (Xcode / pip shart emas).

    Qidiruv: models/llama.cpp/ ichida llama-cli.
    """
    candidates = [
        MODELS_DIR / "llama.cpp" / "llama-cli",
        MODELS_DIR / "llama.cpp" / "bin" / "llama-cli",
        MODELS_DIR / "llama.cpp" / "build" / "bin" / "llama-cli",
    ]
    if (MODELS_DIR / "llama.cpp").exists():
        candidates.extend(sorted((MODELS_DIR / "llama.cpp").rglob("llama-cli")))
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path
    return None


def transformers_model_ready(spec: ModelSpec) -> bool:
    """config.json + barcha safetensors shardlari to'liq bo'lsa True."""
    return _safetensors_complete(spec.transformers_dir)


def _safetensors_complete(folder: Path) -> bool:
    """Yarim yuklangan modelni (bitta shard) tayyor deb hisoblamaydi."""
    import json
    import re

    if not (folder / "config.json").is_file():
        return False
    index = folder / "model.safetensors.index.json"
    if index.is_file():
        try:
            data = json.loads(index.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        names = set((data.get("weight_map") or {}).values())
        if not names:
            return False
        return all(
            (folder / name).is_file() and (folder / name).stat().st_size >= 1_000_000
            for name in names
        )
    single = folder / "model.safetensors"
    if single.is_file() and single.stat().st_size >= 1_000_000_000:
        return True
    shards = sorted(
        p for p in folder.glob("model-*-of-*.safetensors") if p.is_file()
    )
    if not shards:
        return False
    match = re.search(r"-of-(\d+)\.safetensors$", shards[0].name)
    if match and len(shards) < int(match.group(1)):
        return False
    return all(p.stat().st_size >= 1_000_000 for p in shards)


def list_available_llm_models() -> list[dict]:
    """
    Diskda haqiqatan mavjud LLM modellari ro'yxatini qaytaradi.

    Har bir yozuv: key, label, backend (gguf|transformers|missing), path.
    """
    result: list[dict] = []
    for key, spec in LLM_MODELS.items():
        gguf = find_gguf_file(spec)
        tf_ready = transformers_model_ready(spec)
        if tf_ready:
            backend = "mlx"
            path = str(spec.transformers_dir)
            ready = True
        elif gguf is not None:
            backend = "gguf"
            path = str(gguf)
            ready = True
        else:
            backend = "missing"
            path = str(spec.transformers_dir)
            ready = False
        result.append(
            {
                "key": key,
                "label": spec.label,
                "backend": backend,
                "path": path,
                "ready": ready,
            }
        )
    return result


def find_asr_model_dir(spec: AsrSpec) -> Path | None:
    """ASR model papkasini diskda qidiradi (config.json bo'lishi shart)."""
    for folder in spec.search_dirs:
        if (folder / "config.json").is_file():
            return folder
    if spec.hub_glob:
        hub = HF_CACHE_DIR / "hub"
        if hub.is_dir():
            for snap in hub.glob(spec.hub_glob):
                return snap.parent
    return None


def list_available_asr_models() -> list[dict]:
    """Tizimda topilgan ASR modellari ro'yxati."""
    result: list[dict] = []
    for key, spec in ASR_MODELS.items():
        found = find_asr_model_dir(spec)
        result.append(
            {
                "key": key,
                "label": spec.label,
                "path": str(found or spec.search_dirs[0]),
                "ready": found is not None,
            }
        )
    return result


def find_translation_model_dir(spec: TranslationSpec) -> Path | None:
    """NLLB papkasida config.json + og'irlik borligini tekshiradi."""
    for folder in spec.search_dirs:
        if not (folder / "config.json").is_file():
            continue
        weights = [
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix in {".safetensors", ".bin"} and p.stat().st_size >= 1_000_000_000
        ]
        if weights:
            return folder
    return None


def surya_models_ready() -> bool:
    """Surya det + recognition og'irliklari models/OCR da bormi."""
    det = OCR_DIR / SURYA_DET_CHECKPOINT / "manifest.json"
    rec = OCR_DIR / SURYA_REC_CHECKPOINT / "manifest.json"
    return det.is_file() and rec.is_file()


def list_available_ocr_models() -> list[dict]:
    """OCR model tanlovi: Surya (mikroservis) va Gemma (lokal Vision)."""
    return [
        {
            "key": "surya",
            "label": "Surya OCR",
            "backend": "mikroservis",
            "path": str(OCR_DIR),
            "ready": surya_models_ready(),
        },
        {
            "key": "gemma",
            "label": "Gemma",
            "backend": "Vision",
            "path": "",
            "ready": True,
        },
    ]


def list_available_translation_models() -> list[dict]:
    """Tarjima modellari (NLLB) disk holati."""
    result: list[dict] = []
    for key, spec in TRANSLATION_MODELS.items():
        found = find_translation_model_dir(spec)
        result.append(
            {
                "key": key,
                "label": spec.label,
                "backend": "nllb",
                "path": str(found or spec.search_dirs[0]),
                "ready": found is not None,
            }
        )
    return result


def device_info() -> dict:
    """UI va /api/status uchun qurilma haqida qisqa ma'lumot."""
    device = detect_device()
    mem = detect_memory_bytes()
    torch = _try_import_torch()
    name = "Noma'lum"

    if device == "cuda" and torch is not None:
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "NVIDIA CUDA"
    elif device == "mps":
        name = "Apple Silicon MPS"
    else:
        name = platform.processor() or "CPU"

    return {
        "device": device,
        "name": name,
        "memory_gb": round(mem / (1024**3), 1),
        "platform": platform.system(),
        "machine": platform.machine(),
        "recommended_quantization": recommend_quantization(device),
        "offline": True,
    }


def browse_directory(raw_path: str | None = None) -> dict:
    """
    Shu kompyuterdagi papkalarni ochish (UI fayl-menejeri).

    Faqat kataloglar qaytariladi. Yashirin (nuqta bilan boshlanadigan)
    papkalar ko'rsatilmaydi. O'qib bo'lmaydiganlar o'tkazib yuboriladi.
    """
    home = Path.home()
    if raw_path and raw_path.strip():
        target = Path(raw_path).expanduser()
    else:
        target = home

    try:
        target = target.resolve()
    except OSError:
        target = home.resolve()

    if target.is_file():
        target = target.parent
    if not target.exists() or not target.is_dir():
        raise ValueError(f"Papka topilmadi: {target}")

    entries: list[dict] = []
    try:
        children = sorted(target.iterdir(), key=lambda p: p.name.lower())
    except PermissionError as exc:
        raise ValueError(f"Papka ochilmadi (ruxsat yo'q): {target}") from exc

    for child in children:
        if child.name.startswith("."):
            continue
        try:
            is_dir = child.is_dir()
        except OSError:
            continue
        if is_dir:
            entries.append({"name": child.name, "path": str(child), "is_dir": True})
            continue
        if child.suffix.lower() in SUPPORTED_EXTENSIONS:
            entries.append({"name": child.name, "path": str(child), "is_dir": False})

    parent: str | None = None
    if target.parent != target:
        parent = str(target.parent)

    shortcuts: list[dict] = []
    candidates = [
        ("Uy", home),
        ("Ish stoli", home / "Desktop"),
        ("Hujjatlar", home / "Documents"),
        ("Yuklamalar", home / "Downloads"),
        ("Kuzatuv", WATCHED_DIR),
    ]
    if platform.system() != "Windows":
        candidates.append(("Disk", Path("/")))
    else:
        for letter in "CDEFGHIJ":
            drive = Path(f"{letter}:/")
            if drive.exists():
                candidates.append((f"{letter}:", drive))

    seen: set[str] = set()
    for label, pth in candidates:
        try:
            if pth.exists() and pth.is_dir():
                resolved = str(pth.resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    shortcuts.append({"name": label, "path": resolved})
        except OSError:
            continue

    return {
        "path": str(target),
        "parent": parent,
        "entries": entries,
        "shortcuts": shortcuts,
    }
