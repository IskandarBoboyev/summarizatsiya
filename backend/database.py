"""
SQLite orqali qayta ishlangan fayllar tarixi, kuzatiladigan papkalar va sozlamalar.

Ulanishlar qisqa umrli: har bir so'rov o'z ulanishini ochadi/yopadi.
WAL rejimi bir vaqtda o'qish va yozishni osonlashtiradi.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Generator, Iterable

from backend.config import (
    COUNTRIES,
    COUNTRY_KEYS,
    DATABASE_PATH,
    DEFAULT_ASR_LANGUAGE,
    DEFAULT_ASR_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_OCR_MODEL,
    DEFAULT_TRANSLATION_MODEL,
    ensure_runtime_dirs,
    is_app_internal_path,
)

# SQLite ulanishlarini oqimlar o'rtasida xavfsiz ochish
_DB_LOCK = threading.Lock()


def _utc_now() -> str:
    """UTC vaqtini ISO formatida qaytaradi."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    SQLite ulanishini kontekst-menejer sifatida beradi.

    Row factory dict ko'rinishida ishlaydi (ustun nomi -> qiymat).
    """
    ensure_runtime_dirs()
    conn = sqlite3.connect(str(DATABASE_PATH), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """
    Jadvallarni yaratadi va birlamchi ma'lumotlarni yozadi.

    Jadvallar:
        watched_folders — UI orqali qo'shilgan kuzatuv papkalari
        documents       — qayta ishlangan / navbatdagi fayllar
        settings        — LLM model, ASR tili va kvantizatsiya
        rag_chunks      — foydalanuvchi tasdiqlagan RAG matnlari
    """
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS watched_folders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL UNIQUE,
                file_type TEXT NOT NULL,
                file_size INTEGER NOT NULL DEFAULT 0,
                mtime REAL NOT NULL DEFAULT 0,
                original_text TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT '',
                translation_uz TEXT NOT NULL DEFAULT '',
                transcription TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                error_message TEXT NOT NULL DEFAULT '',
                model_used TEXT NOT NULL DEFAULT '',
                asr_language TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
            CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at);
            CREATE INDEX IF NOT EXISTS idx_documents_id ON documents(id);

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS rag_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'original',
                indexed_at TEXT NOT NULL,
                UNIQUE(document_id, chunk_index)
            );
            CREATE INDEX IF NOT EXISTS idx_rag_chunks_doc ON rag_chunks(document_id);
            """
        )
        cols = [str(r[1]) for r in conn.execute("PRAGMA table_info(documents)").fetchall()]
        if "in_rag" not in cols:
            conn.execute("ALTER TABLE documents ADD COLUMN in_rag INTEGER NOT NULL DEFAULT 0")
        if "country" not in cols:
            conn.execute("ALTER TABLE documents ADD COLUMN country TEXT NOT NULL DEFAULT ''")
        if "pipeline_stage" not in cols:
            conn.execute("ALTER TABLE documents ADD COLUMN pipeline_stage TEXT NOT NULL DEFAULT ''")
        if "file_hash" not in cols:
            conn.execute("ALTER TABLE documents ADD COLUMN file_hash TEXT NOT NULL DEFAULT ''")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash)"
        )
        # Server qayta ochilganda yarim qolgan ishni navbatga qaytaradi — done/error tegilmaydi
        conn.execute(
            """
            UPDATE documents
            SET status = 'pending', pipeline_stage = 'queued', updated_at = ?
            WHERE status = 'processing'
            """,
            (_utc_now(),),
        )
        folder_cols = [str(r[1]) for r in conn.execute("PRAGMA table_info(watched_folders)").fetchall()]
        if "country" not in folder_cols:
            conn.execute("ALTER TABLE watched_folders ADD COLUMN country TEXT NOT NULL DEFAULT ''")
        if "asr_model" not in folder_cols:
            conn.execute("ALTER TABLE watched_folders ADD COLUMN asr_model TEXT NOT NULL DEFAULT ''")
        rag_cols = [str(r[1]) for r in conn.execute("PRAGMA table_info(rag_chunks)").fetchall()]
        if "country" not in rag_cols:
            conn.execute("ALTER TABLE rag_chunks ADD COLUMN country TEXT NOT NULL DEFAULT ''")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_country ON documents(country)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rag_chunks_country ON rag_chunks(country)")
        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS rag_chunks_fts USING fts5(
                    content,
                    filename,
                    tokenize = 'unicode61'
                )
                """
            )
        except sqlite3.OperationalError:
            pass

        # Standart sozlamalar — mavjud bo'lsa tegilmaydi
        inherited_llm = DEFAULT_LLM_MODEL
        old_llm = conn.execute(
            "SELECT value FROM settings WHERE key = 'llm_model'"
        ).fetchone()
        if old_llm and old_llm["value"]:
            inherited_llm = str(old_llm["value"])
        defaults = {
            "llm_model": inherited_llm,
            "llm_summary_model": inherited_llm,
            "llm_translate_model": DEFAULT_TRANSLATION_MODEL,
            "asr_model": DEFAULT_ASR_MODEL,
            "ocr_model": DEFAULT_OCR_MODEL,
            "asr_language": DEFAULT_ASR_LANGUAGE,
            "quantization": "auto",
        }
        for key, value in defaults.items():
            conn.execute(
                "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
                (key, value),
            )
        # Audio: eshitilgan til alifbosida yozish — eski default "uz" ni avtomatga o'tkazamiz
        conn.execute(
            "UPDATE settings SET value = ? WHERE key = 'asr_language' AND value = 'uz'",
            (DEFAULT_ASR_LANGUAGE,),
        )
        conn.execute(
            "UPDATE settings SET value = ? WHERE key = 'llm_translate_model' "
            "AND value IN ('gemma-4', 'gemma-26', 'gemma-27')",
            (DEFAULT_TRANSLATION_MODEL,),
        )

        # Eski umumiy papkani davlatsiz qoldirmaymiz — UI dan qayta tanlanadi.
        _refresh_document_identities(conn)

    cleanup_stray_documents()


def _refresh_document_identities(conn: sqlite3.Connection) -> None:
    """Saqlangan yo'llarni normalize qiladi va bo'sh SHA-256 ni to'ldiradi."""
    rows = conn.execute("SELECT id, filepath, file_hash FROM documents").fetchall()
    now = _utc_now()
    for row in rows:
        raw = str(row["filepath"] or "")
        norm = normalize_filepath(raw) or raw
        digest = str(row["file_hash"] or "")
        if not digest:
            path = Path(norm or raw)
            if path.is_file():
                try:
                    digest = file_sha256(path)
                except OSError:
                    digest = ""
        if norm == raw and digest == str(row["file_hash"] or ""):
            continue
        conn.execute(
            """
            UPDATE documents
            SET filepath = ?, file_hash = ?, updated_at = ?
            WHERE id = ?
            """,
            (norm, digest or str(row["file_hash"] or ""), now, int(row["id"])),
        )


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    """sqlite3.Row ni oddiy dict ga aylantiradi."""
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


# ---------------------------------------------------------------------------
# Sozlamalar
# ---------------------------------------------------------------------------

def get_setting(key: str, default: str | None = None) -> str | None:
    """Bitta sozlama qiymatini o'qiydi."""
    with get_connection() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    if row is None:
        return default
    return str(row["value"])


def set_setting(key: str, value: str) -> None:
    """Sozlamani yozadi yoki yangilaydi."""
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )


def get_all_settings() -> dict[str, str]:
    """Barcha sozlamalarni dict ko'rinishida qaytaradi."""
    with get_connection() as conn:
        rows = conn.execute("SELECT key, value FROM settings").fetchall()
    return {str(r["key"]): str(r["value"]) for r in rows}


# ---------------------------------------------------------------------------
# Kuzatiladigan papkalar
# ---------------------------------------------------------------------------

_FOLDER_COLS = "id, path, country, asr_model, created_at"


def list_folders() -> list[dict[str, Any]]:
    """Kuzatilayotgan papkalar / fayllar (davlat bilan)."""
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT {_FOLDER_COLS} FROM watched_folders ORDER BY id ASC"
        ).fetchall()
    return [_row_to_dict(r) for r in rows]  # type: ignore[misc]


def folder_path_for_country(country: str) -> str:
    """Davlatga biriktirilgan fayl yoki papka yo'li."""
    if not country:
        return ""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT path FROM watched_folders WHERE country = ? LIMIT 1",
            (country,),
        ).fetchone()
    return str(row["path"] or "") if row else ""


def add_folder(path: str, country: str) -> dict[str, Any]:
    """
    Davlat uchun bitta yo'l (fayl yoki papka) belgilaydi.

    Shu davlatning oldingi yo'li almashtiriladi.
    """
    if country not in COUNTRY_KEYS:
        raise ValueError(f"Noma'lum davlat: {country}")
    resolved = str(Path(path).expanduser().resolve())
    target = Path(resolved)
    if not target.exists():
        raise ValueError(f"Yo‘l topilmadi: {resolved}")
    if not target.is_dir() and not target.is_file():
        raise ValueError(f"Fayl yoki papka emas: {resolved}")

    now = _utc_now()
    with get_connection() as conn:
        existing = conn.execute(
            "SELECT id FROM watched_folders WHERE country = ?", (country,)
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE watched_folders SET path = ?, created_at = ? WHERE country = ?",
                (resolved, now, country),
            )
        else:
            try:
                conn.execute(
                    "INSERT INTO watched_folders (path, country, created_at) VALUES (?, ?, ?)",
                    (resolved, country, now),
                )
            except sqlite3.IntegrityError:
                conn.execute(
                    "UPDATE watched_folders SET country = ?, created_at = ? WHERE path = ?",
                    (country, now, resolved),
                )
        row = conn.execute(
            f"SELECT {_FOLDER_COLS} FROM watched_folders WHERE country = ?",
            (country,),
        ).fetchone()
    sync_country_documents(country, resolved)
    return _row_to_dict(row)  # type: ignore[return-value]


def folder_asr_model_for_country(country: str) -> str:
    """Davlat papkasiga biriktirilgan ASR model (bo'sh = umumiy sozlama)."""
    if not country:
        return ""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT asr_model FROM watched_folders WHERE country = ? LIMIT 1",
            (country,),
        ).fetchone()
    return str(row["asr_model"] or "").strip() if row else ""


def update_folder_asr_model(folder_id: int, asr_model: str) -> dict[str, Any]:
    """Kuzatiladigan papka uchun transkripsiya modelini saqlaydi."""
    from backend.config import ASR_MODELS

    key = (asr_model or "").strip()
    if key and key not in ASR_MODELS:
        raise ValueError(f"Noma'lum ASR model: {key}")
    with get_connection() as conn:
        cur = conn.execute(
            "UPDATE watched_folders SET asr_model = ? WHERE id = ?",
            (key, folder_id),
        )
        if cur.rowcount <= 0:
            raise KeyError(folder_id)
        row = conn.execute(
            f"SELECT {_FOLDER_COLS} FROM watched_folders WHERE id = ?",
            (folder_id,),
        ).fetchone()
    return _row_to_dict(row)  # type: ignore[return-value]


def _normalized_root(root: str) -> str:
    try:
        return str(Path(root).expanduser().resolve())
    except OSError:
        return str(Path(root).expanduser())


def path_under_root(filepath: str, root: str) -> bool:
    """Fayl davlatga biriktirilgan yo'l ostidami."""
    if not filepath or not root:
        return False
    try:
        fp = Path(filepath).expanduser().resolve()
        rt = Path(root).expanduser().resolve()
    except OSError:
        return False
    if fp == rt:
        return True
    try:
        fp.relative_to(rt)
        return True
    except ValueError:
        return False


def _sql_under_root(root: str) -> tuple[str, list[str]]:
    base = _normalized_root(root)
    target = Path(base)
    if target.is_file():
        return "filepath = ?", [base]
    like = base.rstrip("/\\") + "/%"
    return "(filepath = ? OR filepath LIKE ?)", [base, like]


def country_document_filter(country: str) -> tuple[str, list[Any]]:
    """Davlat bo'yicha faqat biriktirilgan papka/fayl hujjatlari."""
    root = folder_path_for_country(country)
    if root:
        clause, extra = _sql_under_root(root)
        return f"country = ? AND {clause}", [country, *extra]
    return "country = ?", [country]


def _delete_document_ids(conn: sqlite3.Connection, ids: list[int]) -> None:
    if not ids:
        return
    qmarks = ",".join("?" * len(ids))
    rag_rows = conn.execute(
        f"SELECT id FROM rag_chunks WHERE document_id IN ({qmarks})",
        ids,
    ).fetchall()
    for row in rag_rows:
        try:
            conn.execute("DELETE FROM rag_chunks_fts WHERE rowid = ?", (row["id"],))
        except sqlite3.OperationalError:
            break
    conn.execute(f"DELETE FROM rag_chunks WHERE document_id IN ({qmarks})", ids)
    conn.execute(f"DELETE FROM documents WHERE id IN ({qmarks})", ids)


def sync_country_documents(country: str, root: str) -> None:
    """Shu davlatdagi yozuvlarni faqat biriktirilgan yo'lga moslashtiradi."""
    if not country or not root:
        return
    now = _utc_now()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, filepath FROM documents WHERE country = ?",
            (country,),
        ).fetchall()
        stale = [int(r["id"]) for r in rows if not path_under_root(str(r["filepath"]), root)]
        _delete_document_ids(conn, stale)
        clause, extra = _sql_under_root(root)
        conn.execute(
            f"UPDATE documents SET country = ?, updated_at = ? WHERE {clause}",
            [country, now, *extra],
        )


def cleanup_stray_documents() -> None:
    """Loyiha ichidagi tasodifiy .txt/.md va noto'g'ri davlat yozuvlarini tozalaydi."""
    with get_connection() as conn:
        rows = conn.execute("SELECT id, filepath, country FROM documents").fetchall()
        junk: list[int] = []
        for row in rows:
            path = str(row["filepath"] or "")
            if is_app_internal_path(path):
                junk.append(int(row["id"]))
        _delete_document_ids(conn, junk)

    for folder in list_folders():
        country = str(folder.get("country") or "")
        root = str(folder.get("path") or "")
        if country and root:
            sync_country_documents(country, root)


def country_for_path(filepath: str) -> str:
    """Fayl qaysi davlat papkasiga tushganini aniqlaydi."""
    resolved = str(Path(filepath).expanduser())
    try:
        resolved = str(Path(resolved).resolve())
    except OSError:
        pass
    best = ""
    best_len = -1
    for folder in list_folders():
        root = str(folder.get("path") or "")
        country = str(folder.get("country") or "")
        if not root or not country:
            continue
        if resolved == root or resolved.startswith(root.rstrip("/") + "/") or resolved.startswith(root.rstrip("\\") + "\\"):
            if len(root) > best_len:
                best = country
                best_len = len(root)
    return best


def normalize_filepath(filepath: str) -> str:
    """Yo'lni resolve + NFC qiladi — macOS NFD/takror yozuvlarni birlashtiradi."""
    raw = unicodedata.normalize("NFC", (filepath or "").strip())
    if not raw:
        return ""
    path = Path(raw).expanduser()
    try:
        return unicodedata.normalize("NFC", str(path.resolve()))
    except OSError:
        return unicodedata.normalize("NFC", str(path))


def file_sha256(filepath: str | Path) -> str:
    """Fayl tarkibining SHA-256 hex qiymati."""
    path = Path(filepath)
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_document_by_hash(file_hash: str) -> dict[str, Any] | None:
    """Bir xil tarkibli (hash) hujjat — takroriy ishlovni oldini olish."""
    if not file_hash:
        return None
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE file_hash = ? ORDER BY id ASC LIMIT 1",
            (file_hash,),
        ).fetchone()
    return _row_to_dict(row)


def find_registered_document(filepath: str, file_hash: str = "") -> dict[str, Any] | None:
    """Avval yo'l, so'ng hash bo'yicha reyestri."""
    norm = normalize_filepath(filepath)
    found = get_document_by_path(norm) if norm else None
    if found is None and filepath:
        found = get_document_by_path(filepath)
    if found is None and file_hash:
        found = get_document_by_hash(file_hash)
    return found


def delete_folder(folder_id: int) -> bool:
    """Papkani kuzatuv ro'yxatidan olib tashlaydi. True — o'chirildi."""
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM watched_folders WHERE id = ?", (folder_id,))
        return cur.rowcount > 0


def get_folder_paths() -> list[str]:
    """Faqat yo'llar ro'yxati — watchdog uchun."""
    return [str(item["path"]) for item in list_folders()]


# ---------------------------------------------------------------------------
# Hujjatlar
# ---------------------------------------------------------------------------

def upsert_pending_document(
    filepath: str,
    filename: str,
    file_type: str,
    file_size: int,
    mtime: float,
    country: str = "",
    file_hash: str = "",
) -> dict[str, Any]:
    """
    Yangi faylni navbatga qo'yadi.

    Yo'l yoki hash allaqachon bazada bo'lsa — status o'zgarmaydi (done/error/processing).
    Qayta ishlash faqat `mark_reprocess` (UI «Qayta») orqali.
    """
    now = _utc_now()
    filepath = normalize_filepath(filepath) or filepath
    existing = find_registered_document(filepath, file_hash)
    if existing is not None:
        updates: list[str] = []
        values: list[Any] = []
        if country and str(existing.get("country") or "") != country:
            updates.append("country = ?")
            values.append(country)
        if file_hash and str(existing.get("file_hash") or "") != file_hash:
            updates.append("file_hash = ?")
            values.append(file_hash)
        if updates:
            updates.append("updated_at = ?")
            values.extend([now, int(existing["id"])])
            with get_connection() as conn:
                conn.execute(
                    f"UPDATE documents SET {', '.join(updates)} WHERE id = ?",
                    values,
                )
                row = conn.execute(
                    "SELECT * FROM documents WHERE id = ?",
                    (int(existing["id"]),),
                ).fetchone()
            return _row_to_dict(row)  # type: ignore[return-value]
        return existing

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO documents (
                filename, filepath, file_type, file_size, mtime, file_hash,
                status, pipeline_stage, created_at, updated_at, country
            ) VALUES (?, ?, ?, ?, ?, ?, 'pending', 'queued', ?, ?, ?)
            """,
            (
                filename,
                filepath,
                file_type,
                file_size,
                mtime,
                file_hash or "",
                now,
                now,
                country or "",
            ),
        )
        row = conn.execute(
            "SELECT * FROM documents WHERE filepath = ?", (filepath,)
        ).fetchone()
        return _row_to_dict(row)  # type: ignore[return-value]


def list_pending_documents() -> list[dict[str, Any]]:
    """Navbatdagi (pending) hujjatlar — eskiroqlar avval."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM documents WHERE status = 'pending' ORDER BY id ASC"
        ).fetchall()
    return [_row_to_dict(r) for r in rows]  # type: ignore[misc]


def get_document(doc_id: int) -> dict[str, Any] | None:
    """ID bo'yicha bitta hujjat."""
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    return _row_to_dict(row)


def get_document_by_path(filepath: str) -> dict[str, Any] | None:
    """Yo'l bo'yicha hujjat (normalize qilingan va asl yo'l)."""
    candidates = []
    norm = normalize_filepath(filepath)
    if norm:
        candidates.append(norm)
    if filepath and filepath not in candidates:
        candidates.append(filepath)
    with get_connection() as conn:
        for item in candidates:
            row = conn.execute("SELECT * FROM documents WHERE filepath = ?", (item,)).fetchone()
            if row is not None:
                return _row_to_dict(row)
    return None


def update_document(doc_id: int, **fields: Any) -> dict[str, Any] | None:
    """
    Hujjat maydonlarini yangilaydi.

    Ruxsat etilgan maydonlar: original_text, summary, translation_uz,
    transcription, status, error_message, model_used, asr_language.
    """
    allowed = {
        "original_text",
        "summary",
        "translation_uz",
        "transcription",
        "status",
        "error_message",
        "model_used",
        "asr_language",
        "pipeline_stage",
        "filename",
        "file_type",
        "file_size",
        "mtime",
        "file_hash",
        "country",
        "in_rag",
    }
    payload = {k: v for k, v in fields.items() if k in allowed}
    if not payload:
        return get_document(doc_id)

    payload["updated_at"] = _utc_now()
    assignments = ", ".join(f"{key} = ?" for key in payload)
    values = list(payload.values()) + [doc_id]

    with get_connection() as conn:
        conn.execute(f"UPDATE documents SET {assignments} WHERE id = ?", values)
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    return _row_to_dict(row)


def list_documents(
    offset: int = 0,
    limit: int = 12,
    order: str = "asc",
    country: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """
    Hujjatlar ro'yxati.

    order=asc  — pipeline (yuqorida eski, pastda yangi)
    order=desc — kirish oynasi (oxirgi fayllar avval)
    """
    direction = "DESC" if str(order).lower() == "desc" else "ASC"
    where = ""
    params: list[Any] = []
    if country:
        clause, extra = country_document_filter(country)
        where = f"WHERE {clause}"
        params.extend(extra)
    with get_connection() as conn:
        total_row = conn.execute(
            f"SELECT COUNT(*) AS c FROM documents {where}", params
        ).fetchone()
        total = int(total_row["c"]) if total_row else 0
        rows = conn.execute(
            f"SELECT * FROM documents {where} ORDER BY id {direction} LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()
    items = [_row_to_dict(r) for r in rows]
    if country:
        for item in items:
            if item and not item.get("country"):
                item["country"] = country
    return items, total  # type: ignore[misc]


_LOCAL_TZ = timezone(timedelta(hours=5))


def period_start(period: str) -> datetime:
    """
    Kalendar davri boshlanishi (O‘zbekiston, UTC+5).
    day — bugun 00:00; week — dushanba; month — oy boshi; year — yil boshi.
    """
    now = datetime.now(_LOCAL_TZ)
    if period == "week":
        start = now - timedelta(days=now.weekday())
    elif period == "month":
        start = now.replace(day=1)
    elif period == "year":
        start = now.replace(month=1, day=1)
    else:
        start = now
    return start.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)


def _period_cutoff(period: str) -> str:
    return period_start(period).isoformat(timespec="seconds")


def _period_days(period: str) -> int:
    start = period_start(period)
    now = datetime.now(timezone.utc)
    return max(1, int((now - start).total_seconds() // 86400) + 1)


def list_home_by_country(limit: int = 10, period: str = "day") -> list[dict[str, Any]]:
    """Kirish oynasi: tanlangan davr ichidagi har davlatning oxirgi N ta hujjati."""
    cutoff = _period_cutoff(period)
    groups: list[dict[str, Any]] = []
    with get_connection() as conn:
        for spec in COUNTRIES:
            clause, extra = country_document_filter(spec["key"])
            params = [*extra, cutoff]
            total_row = conn.execute(
                f"SELECT COUNT(*) AS c FROM documents WHERE {clause} AND created_at >= ?",
                params,
            ).fetchone()
            rows = conn.execute(
                f"SELECT * FROM documents WHERE {clause} AND created_at >= ? ORDER BY id DESC LIMIT ?",
                [*params, limit],
            ).fetchall()
            groups.append(
                {
                    "key": spec["key"],
                    "label": spec["label"],
                    "total": int(total_row["c"]) if total_row else 0,
                    "items": [_row_to_dict(r) for r in rows],
                }
            )
    return groups


def country_file_stats(period: str = "month") -> dict[str, Any]:
    """
    Davlatlar bo'yicha kelgan fayllar soni.

    period: day | week | month | year
    """
    days = _period_days(period)
    cutoff = _period_cutoff(period)
    items = []
    with get_connection() as conn:
        for spec in COUNTRIES:
            clause, extra = country_document_filter(spec["key"])
            row = conn.execute(
                f"SELECT COUNT(*) AS c FROM documents WHERE {clause} AND created_at >= ?",
                [*extra, cutoff],
            ).fetchone()
            items.append(
                {
                    "key": spec["key"],
                    "label": spec["label"],
                    "count": int(row["c"]) if row else 0,
                }
            )
    return {
        "period": period,
        "days": days,
        "countries": items,
        "total": sum(x["count"] for x in items),
    }


def queue_counts() -> dict[str, int]:
    """Navbat holati: pending / processing / done / error / jami."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS c FROM documents GROUP BY status"
        ).fetchall()
        total_row = conn.execute("SELECT COUNT(*) AS c FROM documents").fetchone()
    counts = {str(r["status"]): int(r["c"]) for r in rows}
    return {
        "pending": counts.get("pending", 0),
        "processing": counts.get("processing", 0),
        "done": counts.get("done", 0),
        "error": counts.get("error", 0),
        "total": int(total_row["c"]) if total_row else 0,
    }


def mark_reprocess(doc_id: int) -> dict[str, Any] | None:
    """
    Faqat UI «Qayta» tugmasi: keshni yangilab, faylni qayta navbatga qo'yadi.
    Avtomatik watcher buni chaqirmaydi.
    """
    doc = get_document(doc_id)
    if doc is None:
        return None
    fields: dict[str, Any] = {
        "status": "pending",
        "pipeline_stage": "queued",
        "error_message": "",
        "original_text": "",
        "summary": "",
        "translation_uz": "",
        "transcription": "",
        "model_used": "",
    }
    path = Path(str(doc.get("filepath") or ""))
    if path.is_file():
        stat = path.stat()
        fields["file_size"] = int(stat.st_size)
        fields["mtime"] = float(stat.st_mtime)
        try:
            fields["file_hash"] = file_sha256(path)
        except OSError:
            pass
    return update_document(doc_id, **fields)


_FLOW_STEP_IDS = ("queued", "extract", "summarize", "translate", "done")


def infer_pipeline_stage(doc: dict[str, Any] | None) -> str:
    """Saqlangan yoki maydonlardan joriy bosqich."""
    if not doc:
        return "queued"
    stored = str(doc.get("pipeline_stage") or "").strip()
    if stored in _FLOW_STEP_IDS or stored == "error":
        return stored
    status = str(doc.get("status") or "")
    if status == "pending":
        return "queued"
    if status == "error":
        return "error"
    if status == "done":
        return "done"
    original = str(doc.get("original_text") or doc.get("transcription") or "").strip()
    if not original:
        return "extract"
    if not str(doc.get("summary") or "").strip():
        return "summarize"
    if not str(doc.get("translation_uz") or "").strip():
        return "translate"
    return "summarize"


def document_flow_steps(doc: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Fayl kelganidan natijagacha: har bosqich holati va qisqa matn."""
    if not doc:
        return []
    status = str(doc.get("status") or "")
    current = infer_pipeline_stage(doc)
    if current == "error":
        original = str(doc.get("original_text") or doc.get("transcription") or "").strip()
        if not original:
            current = "extract"
        elif not str(doc.get("summary") or "").strip():
            current = "summarize"
        elif not str(doc.get("translation_uz") or "").strip():
            current = "translate"
        else:
            current = "done"
    current_idx = _FLOW_STEP_IDS.index(current) if current in _FLOW_STEP_IDS else 0
    if status == "done":
        current_idx = len(_FLOW_STEP_IDS) - 1

    original = str(doc.get("original_text") or doc.get("transcription") or "").strip()
    summary = str(doc.get("summary") or "").strip()
    translation = str(doc.get("translation_uz") or "").strip()
    err = str(doc.get("error_message") or "").strip()

    previews = {
        "queued": f"{doc.get('filename') or 'Fayl'} · {doc.get('file_type') or ''}".strip(" ·"),
        "extract": original or ("O‘qishda xato: " + err if status == "error" else ""),
        "summarize": summary,
        "translate": translation,
        "done": (
            str(doc.get("model_used") or "").strip() or "Tayyor"
            if status == "done"
            else ""
        ),
    }
    empty_hints = {
        "queued": "Navbatda",
        "extract": "OCR / ASR kutilmoqda",
        "summarize": "Xulosa kutilmoqda",
        "translate": "Tarjima kutilmoqda",
        "done": "Natija kutilmoqda",
    }

    steps: list[dict[str, Any]] = []
    for idx, sid in enumerate(_FLOW_STEP_IDS):
        if status == "error" and idx == current_idx:
            state = "error"
        elif idx < current_idx or status == "done":
            state = "done"
        elif idx == current_idx:
            state = "current"
        else:
            state = "wait"
        preview = previews[sid]
        steps.append(
            {
                "id": sid,
                "state": state,
                "preview": preview[:400],
                "empty": empty_hints[sid],
                "has_output": bool(preview) and state in {"done", "current", "error"},
            }
        )
    return steps


def list_pipeline_live(limit: int = 40) -> list[dict[str, Any]]:
    """Jarayon oynasi: navbat, ishlanayotgan va yaqinda tugaganlar."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM documents
            ORDER BY
              CASE status
                WHEN 'processing' THEN 0
                WHEN 'pending' THEN 1
                WHEN 'error' THEN 2
                ELSE 3
              END,
              updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]  # type: ignore[misc]


def public_document(doc: dict[str, Any] | None) -> dict[str, Any] | None:
    """
    API / WebSocket uchun hujjatni xavfsiz ko'rinishga keltiradi.

    Uzun matnlar to'liq qoldiriladi (UI o'zi qisqartiradi),
    lekin None qiymatlar bo'sh satrga aylanadi.
    """
    if doc is None:
        return None
    keys = (
        "id",
        "filename",
        "filepath",
        "file_type",
        "file_size",
        "original_text",
        "summary",
        "translation_uz",
        "transcription",
        "status",
        "error_message",
        "model_used",
        "asr_language",
        "created_at",
        "updated_at",
        "in_rag",
        "country",
        "pipeline_stage",
    )
    out = {k: ("" if doc.get(k) is None else doc.get(k)) for k in keys}
    out["in_rag"] = bool(doc.get("in_rag"))
    out["pipeline_stage"] = infer_pipeline_stage(doc)
    from backend.services.translation_text import clean_summary_output, clean_translation_output

    out["translation_uz"] = clean_translation_output(str(out.get("translation_uz") or ""))
    out["summary"] = clean_summary_output(str(out.get("summary") or ""))
    return out


def replace_rag_chunks(document_id: int, chunks: list[tuple[int, str, str]], filename: str) -> int:
    """Hujjatning eski RAG bo'laklarini o'chirib, yangilarini yozadi."""
    now = _utc_now()
    with get_connection() as conn:
        old = conn.execute(
            "SELECT id FROM rag_chunks WHERE document_id = ?", (document_id,)
        ).fetchall()
        for row in old:
            try:
                conn.execute("DELETE FROM rag_chunks_fts WHERE rowid = ?", (row["id"],))
            except sqlite3.OperationalError:
                break
        conn.execute("DELETE FROM rag_chunks WHERE document_id = ?", (document_id,))
        doc_row = conn.execute(
            "SELECT country FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
        country = str(doc_row["country"] or "") if doc_row else ""
        for index, content, source in chunks:
            cur = conn.execute(
                """
                INSERT INTO rag_chunks (document_id, chunk_index, content, source, indexed_at, country)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (document_id, index, content, source, now, country),
            )
            try:
                conn.execute(
                    "INSERT INTO rag_chunks_fts (rowid, content, filename) VALUES (?, ?, ?)",
                    (cur.lastrowid, content, filename),
                )
            except sqlite3.OperationalError:
                pass
        conn.execute(
            "UPDATE documents SET in_rag = 1, updated_at = ? WHERE id = ?",
            (now, document_id),
        )
    return len(chunks)


def search_rag_chunks(query: str, limit: int = 4, country: str | None = None) -> list[dict[str, Any]]:
    """FTS5 (yoki LIKE) orqali RAG bo'laklarini qidiradi — davlat bo'yicha."""
    import re

    tokens = re.findall(r"[\w\u0400-\u04FF'-]+", query, flags=re.UNICODE)
    tokens = [t for t in tokens if len(t) >= 2]
    if not tokens:
        return []

    fts_parts = []
    for tok in tokens:
        safe = tok.replace('"', '""')
        fts_parts.append(f'"{safe}"*')
    fts_q = " OR ".join(fts_parts)
    country_sql = " AND d.country = ?" if country else ""
    country_params: tuple[Any, ...] = (country,) if country else ()

    with get_connection() as conn:
        has_fts = bool(
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'rag_chunks_fts'"
            ).fetchone()
        )
        rows: list[sqlite3.Row] = []
        if has_fts:
            try:
                rows = conn.execute(
                    f"""
                    SELECT c.id, c.document_id, c.chunk_index, c.content, c.source,
                           d.filename, d.country, bm25(rag_chunks_fts) AS score
                    FROM rag_chunks_fts
                    JOIN rag_chunks c ON c.id = rag_chunks_fts.rowid
                    JOIN documents d ON d.id = c.document_id
                    WHERE rag_chunks_fts MATCH ?{country_sql}
                    ORDER BY score
                    LIMIT ?
                    """,
                    (fts_q, *country_params, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
        if not rows:
            likes = []
            like_params: list[Any] = []
            for tok in tokens[:8]:
                likes.append("c.content LIKE ?")
                like_params.append(f"%{tok}%")
            like_sql = " OR ".join(likes) if likes else "1=0"
            rows = conn.execute(
                f"""
                SELECT c.id, c.document_id, c.chunk_index, c.content, c.source,
                       d.filename, d.country, 0 AS score
                FROM rag_chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE ({like_sql} OR d.filename LIKE ?){country_sql}
                ORDER BY c.document_id DESC, c.chunk_index ASC
                LIMIT ?
                """,
                (*like_params, f"%{tokens[0]}%", *country_params, limit),
            ).fetchall()
        if not rows:
            rows = _recent_rag_rows(conn, country, limit)
    return [_row_to_dict(r) for r in rows]  # type: ignore[misc]


def _recent_rag_rows(
    conn: sqlite3.Connection, country: str | None, limit: int
) -> list[sqlite3.Row]:
    """Kalit so'z topilmasa — shu davlatning so'nggi bo'laklari."""
    country_sql = " AND d.country = ?" if country else ""
    country_params: tuple[Any, ...] = (country,) if country else ()
    return conn.execute(
        f"""
        SELECT c.id, c.document_id, c.chunk_index, c.content, c.source,
               d.filename, d.country, 0 AS score
        FROM rag_chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE 1=1{country_sql}
        ORDER BY c.document_id DESC, c.chunk_index ASC
        LIMIT ?
        """,
        (*country_params, limit),
    ).fetchall()


def rag_stats(country: str | None = None) -> dict[str, int]:
    """RAG bazasidagi hujjat va bo'lak soni (ixtiyoriy davlat filtri)."""
    where = "WHERE country = ?" if country else ""
    params: tuple[Any, ...] = (country,) if country else ()
    with get_connection() as conn:
        docs = conn.execute(
            f"SELECT COUNT(DISTINCT document_id) AS c FROM rag_chunks {where}",
            params,
        ).fetchone()
        chunks = conn.execute(
            f"SELECT COUNT(*) AS c FROM rag_chunks {where}",
            params,
        ).fetchone()
        by_rows = conn.execute(
            """
            SELECT country,
                   COUNT(DISTINCT document_id) AS d,
                   COUNT(*) AS c
            FROM rag_chunks
            GROUP BY country
            """
        ).fetchall()
    by_country = {
        str(r["country"] or ""): {"documents": int(r["d"]), "chunks": int(r["c"])}
        for r in by_rows
        if r["country"]
    }
    return {
        "documents": int(docs["c"]) if docs else 0,
        "chunks": int(chunks["c"]) if chunks else 0,
        "by_country": by_country,
    }


def dump_json(data: Any) -> str:
    """Yordamchi: JSON ga o'tkazish (sozlamalar zaxirasi uchun)."""
    return json.dumps(data, ensure_ascii=False)


def iter_file_records(filepaths: Iterable[str]) -> list[dict[str, Any]]:
    """Berilgan yo'llarga mos yozuvlarni qaytaradi."""
    paths = list(filepaths)
    if not paths:
        return []
    placeholders = ",".join("?" for _ in paths)
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM documents WHERE filepath IN ({placeholders})",
            paths,
        ).fetchall()
    return [_row_to_dict(r) for r in rows]  # type: ignore[misc]
