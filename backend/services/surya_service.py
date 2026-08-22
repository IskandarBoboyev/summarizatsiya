"""
Surya OCR (0.16) — rasm/PDF dan matn, asl joylashuvni saqlab.

Og'irliklar: models/OCR/ (MODEL_CACHE_DIR).
Internetga chiqilmaydi — fayllar oldindan yuklangan bo'lishi kerak.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

from backend.config import (
    IMAGE_EXTENSIONS,
    OCR_DIR,
    SURYA_DET_CHECKPOINT,
    SURYA_REC_CHECKPOINT,
    detect_device,
    empty_accelerator_cache,
    surya_models_ready,
)

os.environ.setdefault("MODEL_CACHE_DIR", str(OCR_DIR))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logger = logging.getLogger(__name__)


class SuryaService:
    """Detection + recognition ni bir marta yuklab, xotirada ushlab turadi."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._device = detect_device()
        self._foundation = None
        self._detector = None
        self._recognizer = None
        self._load_error: str | None = None

    @property
    def is_ready(self) -> bool:
        return self._recognizer is not None and self._detector is not None

    def status(self) -> dict[str, Any]:
        return {
            "ready": self.is_ready,
            "backend": "surya" if self.is_ready else None,
            "device": self._device,
            "model_dir": str(OCR_DIR),
            "error": self._load_error,
        }

    def ensure_loaded(self) -> None:
        if self.is_ready:
            return
        with self._lock:
            if self.is_ready:
                return
            self._device = detect_device()
            self._load_error = None
            if not surya_models_ready():
                raise FileNotFoundError(
                    "Surya modellari topilmadi. "
                    "`models/OCR/` ga yuklang: "
                    f"`python -u scripts/download_models.py surya`"
                )
            try:
                self._load()
            except Exception as exc:
                self._load_error = str(exc)[:800]
                raise

    def unload(self) -> None:
        with self._lock:
            self._foundation = None
            self._detector = None
            self._recognizer = None
            empty_accelerator_cache()

    def extract(self, filepath: str | Path) -> str:
        """Rasm yoki PDF dan layout saqlangan matn."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Fayl topilmadi: {path}")
        self.ensure_loaded()
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            pages = self._pdf_to_images(path)
        elif suffix in IMAGE_EXTENSIONS:
            from PIL import Image

            pages = [Image.open(str(path)).convert("RGB")]
        else:
            raise ValueError(f"Surya faqat rasm/PDF: {suffix}")

        parts: list[str] = []
        for idx, image in enumerate(pages, start=1):
            logger.info("Surya OCR %s/%s (%s)", idx, len(pages), path.name)
            parts.append(self._ocr_image(image))
            empty_accelerator_cache()
        return "\n\n".join(p for p in parts if p).strip()

    def _load(self) -> None:
        from surya.detection import DetectionPredictor
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor

        OCR_DIR.mkdir(parents=True, exist_ok=True)
        _patch_surya_transformers()
        logger.info("Surya yuklanmoqda: %s", OCR_DIR)
        self._foundation = FoundationPredictor()
        _materialize_foundation(self._foundation.model, self._device)
        self._detector = DetectionPredictor()
        self._recognizer = RecognitionPredictor(self._foundation)
        logger.info("Surya yuklandi (qurilma=%s)", self._device)

    def _ocr_image(self, image) -> str:
        from surya.common.surya.schema import TaskNames

        assert self._recognizer is not None and self._detector is not None
        preds = self._recognizer(
            [image],
            task_names=[TaskNames.ocr_with_boxes],
            det_predictor=self._detector,
            highres_images=[image],
            math_mode=False,
        )
        if not preds:
            return ""
        return lines_to_layout_text(preds[0].text_lines)

    @staticmethod
    def _pdf_to_images(path: Path) -> list:
        try:
            import pypdfium2 as pdfium
        except Exception as exc:
            raise RuntimeError("PDF uchun pypdfium2 kerak") from exc

        images = []
        pdf = pdfium.PdfDocument(str(path))
        try:
            limit = min(len(pdf), 40)
            for index in range(limit):
                page = pdf[index]
                try:
                    bitmap = page.render(scale=2.0)
                    images.append(bitmap.to_pil().convert("RGB"))
                finally:
                    try:
                        page.close()
                    except Exception:
                        pass
        finally:
            pdf.close()
        return images


def _patch_surya_transformers() -> None:
    """Surya 0.16 + transformers 5: pad_token va default RoPE."""
    import json

    import torch
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    cfg_path = OCR_DIR / SURYA_REC_CHECKPOINT / "config.json"
    if cfg_path.is_file():
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        pad_id = data.get("pad_token_id", 66556)
        decoder = data.setdefault("decoder", {})
        changed = False
        if decoder.get("pad_token_id") is None:
            decoder["pad_token_id"] = pad_id
            changed = True
        if decoder.get("eos_token_id") is None and data.get("eos_token_id") is not None:
            decoder["eos_token_id"] = data["eos_token_id"]
            changed = True
        if changed:
            cfg_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            logger.info("Surya decoder configiga pad_token_id yozildi")

    if "default" not in ROPE_INIT_FUNCTIONS:

        def _default_rope(config, device=None, seq_len=None, **_kwargs):
            base = float(getattr(config, "rope_theta", 10000.0) or 10000.0)
            heads = int(getattr(config, "num_attention_heads", 1) or 1)
            dim = int(getattr(config, "head_dim", 0) or (int(config.hidden_size) // heads))
            inv_freq = 1.0 / (
                base
                ** (
                    torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float)
                    / dim
                )
            )
            return inv_freq, 1.0

        ROPE_INIT_FUNCTIONS["default"] = _default_rope

    from surya.common.surya import SuryaModel

    if not getattr(SuryaModel, "_tied_keys_patched", False):
        _orig_init = SuryaModel.__init__
        _orig_tie = SuryaModel.tie_weights

        def _init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            if not hasattr(self, "all_tied_weights_keys"):
                self.all_tied_weights_keys = {}

        def _tie(self, *args, **kwargs):
            return _orig_tie(self)

        def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
            output_embeddings.weight = input_embeddings.weight
            if getattr(output_embeddings, "bias", None) is not None:
                pad = output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0]
                if pad > 0:
                    output_embeddings.bias.data = torch.nn.functional.pad(
                        output_embeddings.bias.data, (0, pad), "constant", 0
                    )
            if hasattr(output_embeddings, "out_features") and hasattr(
                input_embeddings, "num_embeddings"
            ):
                output_embeddings.out_features = input_embeddings.num_embeddings

        SuryaModel.__init__ = _init
        SuryaModel.tie_weights = _tie
        SuryaModel._tie_or_clone_weights = _tie_or_clone_weights
        SuryaModel._tied_keys_patched = True


def _materialize_foundation(model, device: str) -> None:
    """Transformers 5 ba'zi og'irliklarni meta da qoldiradi — lm_head ni bog'laymiz."""
    import torch

    if model is None:
        return
    if hasattr(model, "lm_head") and hasattr(model, "embedder"):
        src = model.embedder.token_embed.weight
        if src.device.type != "meta":
            model.lm_head.weight = src
    leftover = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if leftover:
        logger.warning("Surya meta parametrlar qoldi: %s", leftover[:12])
    model.to(device)
    # Vision RoPE inv_freq buffer emas — meta da qoladi, CPU ga qayta yozamiz
    for module in model.modules():
        inv = getattr(module, "inv_freq", None)
        if not torch.is_tensor(inv):
            continue
        if inv.device.type == "meta":
            dim = int(inv.shape[0]) * 2
            module.inv_freq = 1.0 / (
                10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
            )
        elif inv.device.type != "cpu":
            module.inv_freq = inv.detach().to("cpu")


def lines_to_layout_text(text_lines: list) -> str:
    """
    Qatorlarni y/x bo'yicha tartiblab, asl sahifa shaklini saqlaydi.

    - bir qatordagi bo'laklar bo'shliq bilan
    - katta gorizontal tirqish — bir necha bo'shliq (ustun)
    - katta vertikal tirqish — bo'sh qator (abzats)
    """
    if not text_lines:
        return ""
    try:
        from surya.recognition.util import sort_text_lines

        ordered = sort_text_lines(list(text_lines))
    except Exception:
        ordered = list(text_lines)

    rows: list[list[Any]] = []
    for line in ordered:
        bbox = _bbox(line)
        text = (getattr(line, "text", None) or "").strip()
        if not text or not bbox:
            continue
        y_mid = (bbox[1] + bbox[3]) / 2.0
        height = max(bbox[3] - bbox[1], 8.0)
        placed = False
        for row in rows:
            sample = _bbox(row[0])
            row_mid = (sample[1] + sample[3]) / 2.0
            row_h = max(sample[3] - sample[1], 8.0)
            if abs(y_mid - row_mid) <= 0.55 * max(height, row_h):
                row.append(line)
                placed = True
                break
        if not placed:
            rows.append([line])

    blocks: list[str] = []
    prev_bottom: float | None = None
    for row in rows:
        row.sort(key=lambda item: _bbox(item)[0])
        pieces: list[str] = []
        prev_right: float | None = None
        heights = [_bbox(item)[3] - _bbox(item)[1] for item in row]
        avg_h = sum(heights) / max(len(heights), 1)
        top = min(_bbox(item)[1] for item in row)
        bottom = max(_bbox(item)[3] for item in row)
        if prev_bottom is not None and top - prev_bottom > 1.35 * max(avg_h, 8.0):
            blocks.append("")
        prev_bottom = bottom
        for item in row:
            box = _bbox(item)
            word = (getattr(item, "text", None) or "").strip()
            if not word:
                continue
            if prev_right is not None:
                gap = box[0] - prev_right
                if gap > 1.8 * avg_h:
                    pieces.append("    ")
                elif gap > 0.25 * avg_h:
                    pieces.append(" ")
            pieces.append(word)
            prev_right = box[2]
        line_text = "".join(pieces).strip()
        if line_text:
            blocks.append(line_text)
    return "\n".join(blocks).strip()


def _bbox(line: Any) -> list[float]:
    box = getattr(line, "bbox", None)
    if box and len(box) >= 4:
        return [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
    return []


surya_service = SuryaService()
