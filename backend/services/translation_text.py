"""
Hujjat tarjimasi: struktura saqlash va model chiqimini tozalash.

NLLB va (zaxira) LLM tarjima bir xil qoidalarga bo'ysunadi:
    - xulosa yo'q
    - paragraf / ro'yxat / adabiyotlar ketma-ketligi saqlanadi
    - prompt izohlari, HTML, markdown chiqishga tushmasin
"""

from __future__ import annotations

import html
import re

_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
_CHANNEL_RE = re.compile(r"<\|/?[a-zA-Z0-9_]+(?:\|>|>)")
_THOUGHT_HEAD_RE = re.compile(
    r"(?:^|\n)\s*(?:thought|thinking|reasoning|here's a thinking process)"
    r"[\s\S]*?(?=\n\n|\Z)",
    re.IGNORECASE,
)
_QISM_LINE_RE = re.compile(
    r"^\s*(?:qism|part|chunk|bo['‘’`ʻ]?lak)\s+\d+\s*[:.\-–—]?\s*",
    re.IGNORECASE,
)
_PREAMBLE_RE = re.compile(
    r"^\s*(?:here is (?:the )?(?:uzbek )?translation|uzbek translation|"
    r"translation|tarjima|o['‘’`ʻ]?zbekcha tarjima|"
    r"thought|thinking|reasoning)\s*[:.\-–—]?\s*",
    re.IGNORECASE,
)
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_HREF_RE = re.compile(r"""(?:href|src)\s*=\s*["'][^"']*["']""", re.IGNORECASE)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_MD_ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
_MD_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_BARE_STAR_RE = re.compile(r"(?:^|\s)\*+\s*")
_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_FAKE_URL_RE = re.compile(
    r"https?://(?:example\.com|localhost|placeholder|www\.qism\.org)[^\s]*",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。؟؛])\s+")
_META_LINE_RE = re.compile(
    r"^\s*(?:\*+\s*)?(?:title|context|role|goal|format|style|constraint|"
    r"analyze the (?:request|source)|source text|target language|"
    r"momlakat/tasvir)\s*[:/]",
    re.IGNORECASE,
)


def iter_document_units(text: str, max_chars: int = 600) -> list[tuple[str, str]]:
    """
    Hujjatni tarjima birliklariga ajratadi.

    Returns:
        (matn, keyingi_ajratgich) juftliklari. Ajratgich ``\\n`` yoki ``\\n\\n``.
        Bo'sh qatorlar orqali paragraf, qisqa qatorlar (adabiyotlar) saqlanadi.
    """
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not raw.strip():
        return []

    blocks = re.split(r"\n[ \t]*\n+", raw)
    units: list[tuple[str, str]] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = [ln.rstrip() for ln in block.split("\n") if ln.strip()]
        if _is_line_list(lines):
            for idx, line in enumerate(lines):
                sep = "\n" if idx < len(lines) - 1 else "\n\n"
                if len(line) <= max_chars:
                    units.append((line, sep))
                else:
                    pieces = split_long_paragraph(line, max_chars)
                    for j, piece in enumerate(pieces):
                        units.append((piece, sep if j == len(pieces) - 1 else " "))
            continue
        if len(block) <= max_chars:
            units.append((block, "\n\n"))
            continue
        pieces = split_long_paragraph(block, max_chars)
        for j, piece in enumerate(pieces):
            units.append((piece, "\n\n" if j == len(pieces) - 1 else " "))
    return units


def join_document_units(pairs: list[tuple[str, str]]) -> str:
    """Tarjima birliklarini asl ajratgichlar bilan yig'adi."""
    if not pairs:
        return ""
    out: list[str] = []
    for text, sep in pairs:
        piece = (text or "").strip()
        if not piece:
            continue
        out.append(piece)
        out.append(sep)
    return "".join(out).strip()


def split_long_paragraph(text: str, max_chars: int) -> list[str]:
    """Uzun paragrafni jumla chegarasida bo'ladi — so'zlar kesilmaydi."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
    out: list[str] = []
    buf: list[str] = []
    size = 0
    for sent in sentences:
        extra = len(sent) + (1 if buf else 0)
        if size + extra > max_chars and buf:
            out.append(" ".join(buf))
            buf = [sent]
            size = len(sent)
        else:
            buf.append(sent)
            size += extra
    if buf:
        out.append(" ".join(buf))
    return out


def clean_translation_output(text: str) -> str:
    """HTML, markdown, thought, 'Qism N' va soxta URL larni olib tashlaydi."""
    cleaned = html.unescape(text or "").replace("\r\n", "\n")
    if not cleaned.strip():
        return ""
    cleaned = _THINK_RE.sub("", cleaned)
    cleaned = _CHANNEL_RE.sub("", cleaned)
    cleaned = _THOUGHT_HEAD_RE.sub("\n", cleaned)
    cleaned = _FAKE_URL_RE.sub("", cleaned)
    cleaned = _HREF_RE.sub("", cleaned)
    cleaned = _HTML_TAG_RE.sub("", cleaned)
    cleaned = _MD_LINK_RE.sub(r"\1", cleaned)
    cleaned = _MD_BOLD_RE.sub(r"\1", cleaned)
    cleaned = _MD_ITALIC_RE.sub(r"\1", cleaned)
    cleaned = _MD_HEADING_RE.sub("", cleaned)
    cleaned = _URL_RE.sub("", cleaned)

    lines: list[str] = []
    for raw_line in cleaned.split("\n"):
        line = raw_line.strip()
        if not line:
            lines.append("")
            continue
        line = _QISM_LINE_RE.sub("", line)
        line = _PREAMBLE_RE.sub("", line)
        line = _BARE_STAR_RE.sub(" ", line)
        line = re.sub(r"\s{2,}", " ", line).strip(" -*:")
        if not line:
            continue
        if line.lower() in {"thought", "thinking", "reasoning"}:
            continue
        if _META_LINE_RE.match(line):
            continue
        if _is_english_meta(line):
            continue
        lines.append(line)
    text_out = "\n".join(lines)
    text_out = re.sub(r"\n{3,}", "\n\n", text_out)
    return text_out.strip()


def _is_english_meta(line: str) -> bool:
    """Model o'ylashi / inglizcha ko'rsatma qatorlari."""
    low = line.lower()
    markers = (
        "analyze the",
        "constraint",
        "source text",
        "target language",
        "do not use",
        "thinking process",
        "here's a thinking",
        "only return",
        "maintain the exact",
        "length/content",
        "main topic",
        "important facts",
        "must include",
    )
    return any(m in low for m in markers)


_EN_LABEL_RE = re.compile(
    r"^\s*(?:tone|style|main idea|target|audience|context|length|format|"
    r"constraint|role|goal|input text|output|summary|key point|"
    r"crucial|shift|note|analysis)\b",
    re.IGNORECASE,
)
_EN_STOP = {
    "the",
    "and",
    "of",
    "to",
    "in",
    "a",
    "is",
    "that",
    "this",
    "with",
    "for",
    "after",
    "his",
    "was",
    "were",
    "from",
    "style",
    "idea",
    "target",
    "audience",
    "summary",
}


def clean_summary_output(text: str) -> str:
    """Xulosadan yorliq, markdown va inglizcha meta qatorlarni olib tashlaydi."""
    cleaned = clean_translation_output(text)
    if not cleaned:
        return ""
    kept: list[str] = []
    for line in cleaned.split("\n"):
        raw = line.strip()
        if not raw:
            if kept and kept[-1] != "":
                kept.append("")
            continue
        if _EN_LABEL_RE.match(raw):
            # "Main Idea 1: actual sentence" — ikkinchi qismini saqlash
            parts = re.split(r":\s+", raw, maxsplit=1)
            if len(parts) == 2 and len(parts[1]) > 40 and not _EN_LABEL_RE.match(parts[1]):
                raw = parts[1]
            else:
                continue
        if looks_like_english(raw) and len(raw.split()) < 8:
            continue
        kept.append(raw)
    out = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    return out


def looks_like_english(text: str) -> bool:
    """Xulosa hali inglizcha yorliq/matnmi."""
    words = re.findall(r"[A-Za-z']+", text or "")
    if len(words) < 6:
        return False
    hits = sum(1 for w in words if w.lower() in _EN_STOP)
    return hits >= 5 or hits / len(words) >= 0.18


def looks_like_cyrillic(text: str) -> bool:
    """Xulosa kirill alifbosidami (o'zbek/tojik/rus)."""
    letters = [c for c in (text or "") if c.isalpha()]
    if len(letters) < 12:
        return False
    cyr = sum(1 for c in letters if "\u0400" <= c <= "\u04ff")
    return cyr / len(letters) >= 0.35


def _is_line_list(lines: list[str]) -> bool:
    """Adabiyotlar / raqamli ro'yxat: ko'p qisqa qator."""
    if len(lines) < 3:
        return False
    short = sum(1 for ln in lines if len(ln) <= 240)
    numbered = sum(1 for ln in lines if re.match(r"^\s*(?:\d+[\.\)]|[-*•])\s+", ln))
    return short / len(lines) >= 0.7 or numbered / len(lines) >= 0.5
