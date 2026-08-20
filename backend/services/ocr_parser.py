"""
Matn ajratish servisi (OCR / parser).

Qo'llab-quvvatlanadigan manbalar:
    - PDF: avval pdfplumber, so'ng pypdf; skanerlangan sahifalar — rasm + OCR.
    - DOCX: python-docx (paragraflar va jadvallar).
    - Oddiy matn: TXT / MD.
    - Rasmlar: Tesseract (birinchi), EasyOCR (offline zaxira).

Katta fayllar sahifa/bo'laklab o'qiladi — butun hujjat xotiraga yig'ilmaydi.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from backend.config import (
    DOCUMENT_EXTENSIONS,
    EASYOCR_DIR,
    IMAGE_EXTENSIONS,
    MAX_CHUNKS_PER_FILE,
    TESSDATA_DIR,
    TEXT_CHUNK_CHARS,
)

logger = logging.getLogger(__name__)

# OCR tillari: o'zbek + rus + ingliz (Tesseract kodlari)
TESSERACT_LANG = "uzb+rus+eng"

# EasyOCR tillari (ISO)
EASYOCR_LANGS = ["uz", "ru", "en"]


class OCRParser:
    """
    Fayldan matn ajratuvchi sinf.

    EasyOCR og'ir — bir marta yuklanadi va qayta ishlatiladi.
    Tesseract tizimga o'rnatilgan bo'lsa, u afzal ko'riladi (yengilroq).
    """

    def __init__(self) -> None:
        self._easyocr_reader = None
        self._easyocr_failed = False

    # ------------------------------------------------------------------
    # Ommaviy API
    # ------------------------------------------------------------------

    def extract(self, filepath: str | Path, engine: str = "gemma") -> str:
        """
        Fayl turiga qarab matnni ajratadi.

        Args:
            filepath: Lokal fayl yo'li.

        Returns:
            Tozalangan matn. Bo'sh bo'lishi mumkin (masalan, faqat rasmli PDF).

        Raises:
            FileNotFoundError: fayl yo'q.
            ValueError: kengaytma qo'llab-quvvatlanmaydi.
        """
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Fayl topilmadi: {path}")

        suffix = path.suffix.lower()
        engine = (engine or "gemma").strip().lower()
        if engine == "surya" and suffix in IMAGE_EXTENSIONS | {".pdf"}:
            from backend.services.ocr_rpc import ocr_rpc

            return self._normalize_text(ocr_rpc.extract(path))
        if suffix == ".pdf":
            text = self._extract_pdf(path)
        elif suffix == ".docx":
            text = self._extract_docx(path)
        elif suffix in {".txt", ".md"}:
            text = self._extract_plain_text(path)
        elif suffix == ".rtf":
            text = self._extract_plain_text(path)
        elif suffix in IMAGE_EXTENSIONS:
            text = self._ocr_image_file(path)
        else:
            raise ValueError(f"Matn ajratish qo'llab-quvvatlanmaydi: {suffix}")

        return self._normalize_text(text)

    def iter_text_chunks(self, text: str, max_chars: int | None = None) -> list[str]:
        """
        Uzun matnni xotira tejash uchun bo'laklarga ajratadi.

        Bo'laklar paragraf chegarasida kesiladi (so'z o'rtasida emas).
        """
        max_chars = max_chars or TEXT_CHUNK_CHARS
        cleaned = self._normalize_text(text)
        if not cleaned:
            return []
        if len(cleaned) <= max_chars:
            return [cleaned]

        chunks: list[str] = []
        start = 0
        length = len(cleaned)
        while start < length and len(chunks) < MAX_CHUNKS_PER_FILE:
            end = min(start + max_chars, length)
            if end < length:
                # Oxirgi bo'shliq yoki qator oxirini qidiramiz
                window = cleaned[start:end]
                cut = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(" "))
                if cut >= max_chars // 3:
                    end = start + cut
            piece = cleaned[start:end].strip()
            if piece:
                chunks.append(piece)
            start = end
        return chunks

    # ------------------------------------------------------------------
    # PDF
    # ------------------------------------------------------------------

    def _extract_pdf(self, path: Path) -> str:
        """
        PDF dan matn oladi.

        1) pdfplumber — jadvallar va layout uchun yaxshi.
        2) pypdf — zaxira.
        3) Agar sahifada matn juda kam bo'lsa, sahifa rasmi + OCR.
        """
        texts: list[str] = []

        plumber_ok = self._pdf_with_pdfplumber(path, texts)
        if not plumber_ok or self._looks_like_scanned("\n".join(texts)):
            pypdf_texts: list[str] = []
            self._pdf_with_pypdf(path, pypdf_texts)
            if len("".join(pypdf_texts).strip()) > len("".join(texts).strip()):
                texts = pypdf_texts

        combined = "\n".join(texts).strip()
        if self._looks_like_scanned(combined):
            logger.info("PDF skanerlangan ko'rinadi, sahifalar OCR qilinadi: %s", path.name)
            ocr_text = self._ocr_pdf_pages(path)
            if len(ocr_text.strip()) > len(combined):
                return ocr_text
        return combined

    def _pdf_with_pdfplumber(self, path: Path, texts: list[str]) -> bool:
        """pdfplumber orqali sahifa-sahifa o'qiydi. Muvaffaqiyat — True."""
        try:
            import pdfplumber  # noqa: WPS433
        except Exception as exc:
            logger.debug("pdfplumber mavjud emas: %s", exc)
            return False

        try:
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    # Jadvallar alohida
                    try:
                        tables = page.extract_tables() or []
                    except Exception:
                        tables = []
                    table_lines: list[str] = []
                    for table in tables:
                        for row in table:
                            cells = [str(c).strip() if c else "" for c in row]
                            table_lines.append(" | ".join(cells))
                    block = "\n".join(part for part in (page_text, "\n".join(table_lines)) if part)
                    texts.append(block)
            return True
        except Exception as exc:
            logger.warning("pdfplumber xatosi (%s): %s", path.name, exc)
            return False

    def _pdf_with_pypdf(self, path: Path, texts: list[str]) -> bool:
        """pypdf zaxira o'qish."""
        try:
            from pypdf import PdfReader  # noqa: WPS433
        except Exception as exc:
            logger.debug("pypdf mavjud emas: %s", exc)
            return False

        try:
            reader = PdfReader(str(path))
            for page in reader.pages:
                texts.append(page.extract_text() or "")
            return True
        except Exception as exc:
            logger.warning("pypdf xatosi (%s): %s", path.name, exc)
            return False

    def _ocr_pdf_pages(self, path: Path) -> str:
        """
        PDF sahifalarini rasmga aylantirib OCR qiladi (pypdfium2).

        Har bir sahifa alohida ishlanadi — butun PDF bitmap xotirada turmaydi.
        """
        try:
            import pypdfium2 as pdfium  # noqa: WPS433
        except Exception as exc:
            logger.warning("pypdfium2 o'rnatilmagan, skaner-PDF OCR o'tkazib yuborildi: %s", exc)
            return ""

        collected: list[str] = []
        pdf = None
        try:
            pdf = pdfium.PdfDocument(str(path))
            page_count = len(pdf)
            # Juda katta PDF — birinchi N sahifa (OOM himoyasi)
            limit = min(page_count, MAX_CHUNKS_PER_FILE)
            for index in range(limit):
                page = pdf[index]
                try:
                    bitmap = page.render(scale=1.8)
                    pil_image = bitmap.to_pil()
                    text = self._ocr_pil_image(pil_image)
                    if text:
                        collected.append(text)
                finally:
                    # Sahifa resurslarini bo'shatish
                    try:
                        page.close()
                    except Exception:
                        pass
        except Exception as exc:
            logger.warning("PDF sahifalarini OCR qilishda xato: %s", exc)
        finally:
            if pdf is not None:
                try:
                    pdf.close()
                except Exception:
                    pass
        return "\n\n".join(collected)

    # ------------------------------------------------------------------
    # DOCX / matn
    # ------------------------------------------------------------------

    def _extract_docx(self, path: Path) -> str:
        """Word hujjatidan paragraflar va jadvallar matnini oladi."""
        try:
            from docx import Document  # noqa: WPS433
        except Exception as exc:
            raise RuntimeError(
                "python-docx o'rnatilmagan. `pip install python-docx` qiling."
            ) from exc

        doc = Document(str(path))
        parts: list[str] = []
        for paragraph in doc.paragraphs:
            if paragraph.text and paragraph.text.strip():
                parts.append(paragraph.text)

        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        return "\n".join(parts)

    def _extract_plain_text(self, path: Path) -> str:
        """TXT/MD/RTF ni bir necha kodlashda o'qishga harakat qiladi."""
        raw = path.read_bytes()
        for encoding in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Rasmlar / OCR
    # ------------------------------------------------------------------

    def _ocr_image_file(self, path: Path) -> str:
        """Rasm faylini ochib OCR qiladi."""
        text = self._macos_vision_ocr(path)
        if text and len(text.strip()) >= 3:
            return text
        try:
            from PIL import Image  # noqa: WPS433
        except Exception as exc:
            raise RuntimeError("Pillow o'rnatilmagan.") from exc

        with Image.open(str(path)) as img:
            image = img.convert("RGB")
            return self._ocr_pil_image(image)

    def _ocr_pil_image(self, image) -> str:
        """
        PIL Image ustida OCR: macOS Vision, so'ng Tesseract, so'ng EasyOCR.
        """
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            image.save(tmp_path, format="PNG")
            text = self._macos_vision_ocr(tmp_path)
            if text and len(text.strip()) >= 3:
                return text
        finally:
            tmp_path.unlink(missing_ok=True)

        text = self._tesseract_ocr(image)
        if text and len(text.strip()) >= 3:
            return text
        return self._easyocr_ocr(image)

    def _macos_vision_ocr(self, path: Path) -> str:
        """
        macOS Live Text / Vision — Tesseract o'rnatilmasa ham ishlaydi.
        """
        if sys.platform != "darwin":
            return ""
        script = Path(__file__).with_name("macos_vision_ocr.js")
        if not script.is_file():
            return ""
        env = os.environ.copy()
        env["OCR_IMAGE_PATH"] = str(path)
        try:
            proc = subprocess.run(
                ["osascript", "-l", "JavaScript", str(script)],
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.debug("macOS Vision OCR ishlamadi: %s", exc)
            return ""
        if proc.returncode != 0:
            logger.warning("macOS Vision OCR xato: %s", (proc.stderr or "")[:300])
            return ""
        # JXA console.log ba'zan stderr ga yozadi
        return (proc.stdout or proc.stderr or "").strip()

    def _tesseract_ocr(self, image) -> str:
        """Tesseract CLI orqali offline OCR."""
        try:
            import pytesseract  # noqa: WPS433
        except Exception:
            return ""

        try:
            config_parts = ["--oem 1", "--psm 6"]
            if TESSDATA_DIR.exists() and any(TESSDATA_DIR.glob("*.traineddata")):
                config_parts.append(f"--tessdata-dir {TESSDATA_DIR}")
            config = " ".join(config_parts)
            text = pytesseract.image_to_string(image, lang=TESSERACT_LANG, config=config)
            return text or ""
        except Exception as exc:
            logger.debug("Tesseract ishlamadi, EasyOCR sinab ko'riladi: %s", exc)
            # Til paketi yo'q bo'lsa, faqat inglizcha bilan qayta urinish
            try:
                import pytesseract  # noqa: WPS433

                return pytesseract.image_to_string(image, lang="eng") or ""
            except Exception:
                return ""

    def _easyocr_ocr(self, image) -> str:
        """
        EasyOCR — modellari models/easyocr dan, internetsiz.

        download_enabled=False: tarmoqqa chiqish taqiqlanadi.
        """
        reader = self._get_easyocr_reader()
        if reader is None:
            return ""

        try:
            import numpy as np  # noqa: WPS433

            array = np.array(image)
            # detail=0 — faqat matn satrlari
            lines = reader.readtext(array, detail=0, paragraph=True)
            if isinstance(lines, list):
                return "\n".join(str(x) for x in lines)
            return str(lines)
        except Exception as exc:
            logger.warning("EasyOCR xatosi: %s", exc)
            return ""

    def _get_easyocr_reader(self):
        """EasyOCR Reader ni bir marta, offline rejimda yuklaydi."""
        if self._easyocr_failed:
            return None
        if self._easyocr_reader is not None:
            return self._easyocr_reader

        try:
            import easyocr  # noqa: WPS433

            EASYOCR_DIR.mkdir(parents=True, exist_ok=True)
            self._easyocr_reader = easyocr.Reader(
                EASYOCR_LANGS,
                gpu=True,  # CUDA/MPS bo'lsa ishlatadi, bo'lmasa CPU
                model_storage_directory=str(EASYOCR_DIR),
                download_enabled=False,  # internetga chiqmasin
                verbose=False,
            )
            logger.info("EasyOCR lokal modellardan yuklandi: %s", EASYOCR_DIR)
            return self._easyocr_reader
        except Exception as exc:
            self._easyocr_failed = True
            logger.warning(
                "EasyOCR yuklanmadi (modellarni models/easyocr ga qo'ying): %s",
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # Yordamchilar
    # ------------------------------------------------------------------

    @staticmethod
    def _looks_like_scanned(text: str) -> bool:
        """Matn juda qisqa bo'lsa, PDF skaner deb taxmin qilinadi."""
        compact = "".join(text.split())
        return len(compact) < 40

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Ortiqcha bo'sh qatorlarni qisqartiradi va chetlarni kesadi."""
        if not text:
            return ""
        lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
        cleaned: list[str] = []
        blank = 0
        for line in lines:
            if not line.strip():
                blank += 1
                if blank <= 1:
                    cleaned.append("")
                continue
            blank = 0
            cleaned.append(line)
        return "\n".join(cleaned).strip()


def classify_file_kind(path: Path) -> str:
    """
    Fayl turini UI/pipeline uchun qisqa belgi bilan qaytaradi.

    Returns:
        pdf | docx | text | image | audio | video | other
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".docx":
        return "docx"
    if suffix in {".txt", ".md", ".rtf"}:
        return "text"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    from backend.config import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS  # mahalliy import

    if suffix in AUDIO_EXTENSIONS:
        return "audio"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in DOCUMENT_EXTENSIONS:
        return "text"
    return "other"


# Pipeline uchun yagona misol
ocr_parser = OCRParser()
