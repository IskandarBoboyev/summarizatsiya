"""
Foydalanuvchi tasdiqlagan hujjatlarni RAG bazasiga yozish va qidirish.

Indeks faqat «Bazaga yozish» bosilganda to'ldiriladi.
Qidiruv: SQLite FTS5, bo'lmasa LIKE. Internetga chiqilmaydi.
"""

from __future__ import annotations

import logging

from backend.config import RAG_CHUNK_CHARS, RAG_TOP_K
from backend.database import (
    get_document,
    rag_stats,
    replace_rag_chunks,
    search_rag_chunks,
)
from backend.services.ocr_parser import ocr_parser

logger = logging.getLogger(__name__)


def _source_texts(doc: dict) -> list[tuple[str, str]]:
    """Indekslanadigan matnlar: asl/transkripsiya, tarjima, xulosa."""
    parts: list[tuple[str, str]] = []
    original = (doc.get("original_text") or doc.get("transcription") or "").strip()
    if original:
        parts.append(("original", original))
    translation = (doc.get("translation_uz") or "").strip()
    if translation and translation != original:
        parts.append(("translation", translation))
    summary = (doc.get("summary") or "").strip()
    if summary and summary not in {original, translation}:
        parts.append(("summary", summary))
    return parts


def index_document(doc_id: int) -> dict:
    """
    Hujjat matnini RAG ga yozadi (eski yozuvlarni almashtiradi).

    Returns:
        chunks, filename, in_rag
    """
    doc = get_document(doc_id)
    if doc is None:
        raise FileNotFoundError("Hujjat topilmadi")

    pieces = _source_texts(doc)
    if not pieces:
        raise ValueError("Yozish uchun matn yo'q. Avval faylni qayta ishlang.")

    chunks: list[tuple[int, str, str]] = []
    for source, text in pieces:
        for piece in ocr_parser.iter_text_chunks(text, max_chars=RAG_CHUNK_CHARS):
            chunks.append((len(chunks), piece, source))

    if not chunks:
        raise ValueError("Matn bo'laklarga ajralmadi")

    filename = str(doc.get("filename") or "")
    count = replace_rag_chunks(doc_id, chunks, filename)
    logger.info("RAG ga yozildi: %s (%s bo'lak)", filename, count)
    return {
        "document_id": doc_id,
        "filename": filename,
        "chunks": count,
        "in_rag": True,
        "country": str(doc.get("country") or ""),
    }


def retrieve(question: str, limit: int | None = None, country: str | None = None) -> list[dict]:
    """Savolga mos RAG bo'laklarini qaytaradi (davlat bo'yicha)."""
    return search_rag_chunks(question, limit=limit or RAG_TOP_K, country=country)


def build_context(hits: list[dict]) -> str:
    """LLM ga beriladigan KONTEKST satri."""
    blocks: list[str] = []
    for hit in hits:
        name = hit.get("filename") or f"hujjat-{hit.get('document_id')}"
        body = (hit.get("content") or "").strip()
        if body:
            blocks.append(f"[{name}]\n{body}")
    return "\n\n---\n\n".join(blocks)


def stats(country: str | None = None) -> dict[str, int]:
    return rag_stats(country=country)
