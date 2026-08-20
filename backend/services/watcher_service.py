"""
Papka kuzatuv servisi (watchdog).

Foydalanuvchi UI orqali papka qo'shadi / o'chiradi.
Yangi fayl tushganda (yoki skanerlashda) u SQLite navbatiga yoziladi.
Mavjud fayllar qo'shilganda ham bir marta skanerlandi.

Debounce: bir faylga ketma-ket yozuvlar (nusxa ko'chirish) bitta hodisa deb olinadi.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

from backend.config import (
    SKIP_SCAN_DIR_NAMES,
    SUPPORTED_EXTENSIONS,
    WATCHDOG_DEBOUNCE_SECONDS,
    WATCHED_DIR,
    is_app_internal_path,
)
from backend.database import (
    add_folder,
    country_for_path,
    delete_folder,
    file_sha256,
    find_registered_document,
    get_folder_paths,
    list_folders,
    normalize_filepath,
    upsert_pending_document,
)
from backend.services.ocr_parser import classify_file_kind

logger = logging.getLogger(__name__)

try:
    from watchdog.events import FileSystemEvent, FileSystemEventHandler
    from watchdog.observers import Observer

    _WATCHDOG_AVAILABLE = True
except Exception:  # paket o'rnatilmagan bo'lsa ham UI ishlaydi
    FileSystemEvent = object  # type: ignore[misc,assignment]
    FileSystemEventHandler = object  # type: ignore[misc,assignment]
    Observer = None  # type: ignore[misc,assignment]
    _WATCHDOG_AVAILABLE = False

# Vaqtinchalik / tizim fayllarini e'tiborsiz qoldirish
_IGNORE_PREFIXES = (".", "~$", ".~")
_IGNORE_SUFFIXES = (".tmp", ".part", ".crdownload", ".download")


class _DebouncedHandler(FileSystemEventHandler):
    """
    created / moved / modified hodisalarini debounce qiladi.

    Katta fayl nusxa ko'chirilayotganda bir necha modified keladi —
    taymer oxirgi hodisadan keyin ishga tushadi.
    """

    def __init__(self, on_file: Callable[[Path], None], debounce: float) -> None:
        super().__init__()
        self._on_file = on_file
        self._debounce = debounce
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def on_created(self, event: FileSystemEvent) -> None:
        self._schedule(event)

    def on_moved(self, event: FileSystemEvent) -> None:
        self._schedule(event, dest=True)

    def on_modified(self, event: FileSystemEvent) -> None:
        # Katalog modified ni e'tiborsiz
        if event.is_directory:
            return
        self._schedule(event)

    def _schedule(self, event: FileSystemEvent, dest: bool = False) -> None:
        if event.is_directory:
            return
        raw = event.dest_path if dest and hasattr(event, "dest_path") else event.src_path
        path = Path(str(raw))
        if not _is_candidate(path):
            return

        key = str(path.resolve()) if path.exists() else str(path)
        with self._lock:
            old = self._timers.pop(key, None)
            if old is not None:
                old.cancel()
            timer = threading.Timer(self._debounce, self._fire, args=(path,))
            timer.daemon = True
            self._timers[key] = timer
            timer.start()

    def _fire(self, path: Path) -> None:
        key = str(path.resolve()) if path.exists() else str(path)
        with self._lock:
            self._timers.pop(key, None)
        if path.is_file() and _is_candidate(path):
            self._on_file(path)


class WatcherService:
    """
    Bir nechta papkani parallel kuzatadi.

    Observer qayta ishga tushiriladi: papka qo'shilganda / o'chirilganda
    watchdog ro'yxati yangilanadi.
    """

    def __init__(self) -> None:
        self._observer = None
        self._poll_stop = threading.Event()
        self._poll_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._running = False
        self._on_enqueued: Callable[[dict], None] | None = None

    def set_enqueue_callback(self, callback: Callable[[dict], None] | None) -> None:
        """
        Fayl navbatga qo'yilganda chaqiriladigan callback (WebSocket yangilash).
        """
        self._on_enqueued = callback

    def start(self) -> None:
        """Kuzatuvni ishga tushiradi va mavjud fayllarni skanerlaydi."""
        WATCHED_DIR.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._restart_observer_unlocked()
            self._running = True
        # Mavjud fayllarni birinchi marta navbatga olish
        self.scan_all()
        logger.info("Watcher ishga tushdi. Papkalar: %s", get_folder_paths())

    def stop(self) -> None:
        """Observer / polling ni to'xtatadi."""
        with self._lock:
            self._running = False
            self._poll_stop.set()
            if self._observer is not None:
                self._observer.stop()
                self._observer.join(timeout=5)
                self._observer = None
            poll = self._poll_thread
        if poll is not None:
            poll.join(timeout=5)
        logger.info("Watcher to'xtatildi")

    def refresh(self) -> None:
        """Papkalar ro'yxati o'zgaganda watchdog ni qayta sozlaydi."""
        with self._lock:
            if self._running:
                self._restart_observer_unlocked()

    def add_watch_folder(self, path: str, country: str) -> dict:
        """UI: davlat uchun fayl/papka belgilash va darhol skanerlash."""
        folder = add_folder(path, country)
        self.refresh()
        self.scan_path(Path(folder["path"]), country=country)
        return folder

    def remove_watch_folder(self, folder_id: int) -> bool:
        """UI: papkani kuzatuvdan olib tashlash (fayllar o'chirilmaydi)."""
        ok = delete_folder(folder_id)
        if ok:
            self.refresh()
        return ok

    def list_watch_folders(self) -> list[dict]:
        """Joriy papkalar."""
        return list_folders()

    def scan_all(self) -> int:
        """Barcha kuzatiladigan papkalarni rekursiv skanerlaydi. Navbatga olinganlar soni."""
        count = 0
        for folder in list_folders():
            count += self.scan_path(Path(folder["path"]), country=str(folder.get("country") or ""))
        return count

    def scan_path(self, root: Path, country: str = "") -> int:
        """Bitta ildiz papkani yoki bitta faylni aylanadi."""
        if not root.exists():
            logger.warning("Skaner: yo‘l yo‘q %s", root)
            return 0
        if root.is_file():
            return 1 if self.enqueue_file(root, country=country) else 0
        if not root.is_dir():
            return 0
        count = 0
        try:
            for item in root.rglob("*"):
                if item.is_dir() and item.name in SKIP_SCAN_DIR_NAMES:
                    continue
                if item.is_file() and _is_candidate(item):
                    if self.enqueue_file(item, country=country):
                        count += 1
        except OSError as exc:
            logger.warning("Skaner xatosi (%s): %s", root, exc)
        return count

    def enqueue_file(self, path: Path, country: str = "") -> bool:
        """
        Faylni pending holatida bazaga yozadi.

        Returns:
            True — yangi yoki qayta ishlashga qo'yildi.
            False — o'tkazib yuborildi (o'zgarishsiz done).
        """
        try:
            path = Path(normalize_filepath(str(path)) or path)
            if not path.is_file() or not _is_candidate(path):
                return False
            stat = path.stat()
            # Yo'l reyestda — hatto mtime o'zgarsa ham avtomatik qayta ishlanmaydi
            if find_registered_document(str(path)) is not None:
                return False
            try:
                digest = file_sha256(path)
            except OSError:
                digest = ""
            if digest and find_registered_document(str(path), digest) is not None:
                return False
            kind = classify_file_kind(path)
            resolved_country = country or country_for_path(str(path))
            doc = upsert_pending_document(
                filepath=str(path),
                filename=path.name,
                file_type=kind,
                file_size=int(stat.st_size),
                mtime=float(stat.st_mtime),
                country=resolved_country,
                file_hash=digest,
            )
            if doc.get("status") != "pending":
                return False
            if self._on_enqueued is not None:
                try:
                    self._on_enqueued(doc)
                except Exception:
                    logger.exception("enqueue callback xatosi")
            logger.info("Navbatga olindi: %s [%s]", path.name, kind)
            return True
        except Exception as exc:
            logger.warning("Faylni navbatga olishda xato (%s): %s", path, exc)
            return False

    def _start_polling(self) -> None:
        """watchdog yo'qida papkalarni har 4 soniyada skanerlaydi."""
        self._poll_stop.clear()

        def _loop() -> None:
            while not self._poll_stop.wait(12.0):
                try:
                    self.scan_all()
                except Exception:
                    logger.exception("Polling skaner xatosi")

        self._poll_thread = threading.Thread(target=_loop, name="folder-poll", daemon=True)
        self._poll_thread.start()
        logger.info("Watcher polling rejimida (watchdog paketi yo'q)")

    def _restart_observer_unlocked(self) -> None:
        """Eski observer ni to'xtatib, joriy papkalar bilan yangisini ochadi."""
        if not _WATCHDOG_AVAILABLE or Observer is None:
            if self._poll_thread is None or not self._poll_thread.is_alive():
                self._start_polling()
            return

        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

        handler = _DebouncedHandler(self.enqueue_file, WATCHDOG_DEBOUNCE_SECONDS)
        observer = Observer()
        attached = 0
        for folder in get_folder_paths():
            path = Path(folder)
            watch_dir = path if path.is_dir() else path.parent
            recursive = path.is_dir()
            if not watch_dir.is_dir():
                logger.warning("Kuzatib bo'lmaydi (papka yo'q): %s", folder)
                continue
            try:
                observer.schedule(handler, str(watch_dir), recursive=recursive)
                attached += 1
            except Exception as exc:
                logger.warning("watchdog schedule xatosi (%s): %s", folder, exc)

        if attached == 0:
            # Hech bo'lmaganda default papka
            WATCHED_DIR.mkdir(parents=True, exist_ok=True)
            observer.schedule(handler, str(WATCHED_DIR), recursive=True)

        observer.daemon = True
        observer.start()
        self._observer = observer


def _is_candidate(path: Path) -> bool:
    """Tizim/vaqtinchalik fayllarni va noto'g'ri kengaytmalarni rad etadi."""
    name = path.name
    if not name or name.startswith(_IGNORE_PREFIXES):
        return False
    lower = name.lower()
    if any(lower.startswith(p) for p in _IGNORE_PREFIXES):
        return False
    if any(lower.endswith(s) for s in _IGNORE_SUFFIXES):
        return False
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False
    if is_app_internal_path(path):
        return False
    if any(part in SKIP_SCAN_DIR_NAMES for part in path.parts):
        return False
    return True


# Pipeline / main uchun yagona misol
watcher_service = WatcherService()
