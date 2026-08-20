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
        logger.info("Surya yuklanmoqda: %s", OCR_DIR)
        self._foundation = FoundationPredictor()
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
