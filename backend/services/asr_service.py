"""
Offline ASR / transkripsiya servisi.

Asosiy model: ai-sage/GigaAM-Multilingual
Yuklash manbai: models/ASR modellar/GigaAM-Multilingual yoki models/hf_cache.

Strategiya (ketma-ket, internetga chiqilmaydi):
    1. `gigaam` paketi + lokal og'irliklar.
    2. Transformers AutoModel / AutoProcessor (CTC yoki seq2seq).
    3. Hech biri ishlamasa — tushunarli xato.

Uzun audio 25 soniyalik bo'laklarga bo'linadi (OOM va barqarorlik).
Video fayllardan audio ffmpeg orqali ajratiladi (agar tizimda bo'lsa).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

from backend.config import (
    AUDIO_CHUNK_SECONDS,
    AUDIO_SAMPLE_RATE,
    GIGAAM_DIR,
    HF_CACHE_DIR,
    VIDEO_EXTENSIONS,
    detect_device,
    empty_accelerator_cache,
)

logger = logging.getLogger(__name__)


class ASRService:
    """
    GigaAM-Multilingual ni offline yuklaydigan transkripsiya servisi.

    Til kodi (`auto`, `uz`, `ru`, ...) ixtiyoriy. GigaAM CTC eshitilgan
    til alifbosida (lotin/kirill) yozadi — majburiy til yo'q.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._backend: str | None = None  # "gigaam" | "transformers"
        self._device = detect_device()
        self._gigaam_model = None
        self._hf_model = None
        self._hf_processor = None
        self._load_error: str | None = None

    @property
    def is_ready(self) -> bool:
        """ASR modeli xotiradami."""
        return self._gigaam_model is not None or self._hf_model is not None

    def status(self) -> dict[str, Any]:
        """UI uchun holat."""
        return {
            "ready": self.is_ready,
            "backend": self._backend,
            "device": self._device,
            "model_dir": str(GIGAAM_DIR),
            "error": self._load_error,
        }

    def ensure_loaded(self) -> None:
        """Model bir marta yuklanadi. Keyingi chaqiriqlar no-op."""
        if self.is_ready:
            return
        with self._lock:
            if self.is_ready:
                return
            self._device = detect_device()
            self._load_error = None
            try:
                local = self._local_model_path()
                if self._try_load_gigaam():
                    return
                if self._try_load_transformers():
                    return
                if local is None:
                    raise FileNotFoundError(
                        "GigaAM-Multilingual topilmadi. "
                        "Modelni `models/ASR modellar/GigaAM-Multilingual/` "
                        "papkasiga (config.json + og'irliklar) joylang."
                    )
                hint = self._load_error or "Transformers yuklash muvaffaqiyatsiz"
                raise RuntimeError(
                    f"GigaAM papkasi bor ({local}), lekin model yuklanmadi: {hint}"
                )
            except Exception as exc:
                self._load_error = str(exc)
                raise

    def unload(self) -> None:
        """ASR ni xotiradan chiqaradi (LLM ga joy ochish uchun)."""
        with self._lock:
            self._gigaam_model = None
            self._hf_model = None
            self._hf_processor = None
            self._backend = None
            empty_accelerator_cache()

    # ------------------------------------------------------------------
    # Yuklash
    # ------------------------------------------------------------------

    def _local_model_path(self) -> Path | None:
        """
        Lokal GigaAM papkasini qidiradi.

        Qabul qilinadigan joylar:
            models/ASR modellar/GigaAM-Multilingual/
            models/hf_cache/hub/models--ai-sage--GigaAM-Multilingual/snapshots/<hash>/
        """
        candidates = [
            GIGAAM_DIR,
            GIGAAM_DIR / "ai-sage" / "GigaAM-Multilingual",
        ]
        for cand in candidates:
            if (cand / "config.json").is_file():
                return cand

        hub = HF_CACHE_DIR / "hub"
        if hub.is_dir():
            for snap in hub.glob("models--ai-sage--GigaAM-Multilingual/snapshots/*/config.json"):
                return snap.parent
        return None

    def _try_load_gigaam(self) -> bool:
        """rasmiy `gigaam` paketi orqali yuklash."""
        try:
            import gigaam  # noqa: WPS433
        except Exception:
            logger.debug("gigaam paketi o'rnatilmagan, Transformers sinab ko'riladi")
            return False

        local = self._local_model_path()
        # gigaam.load_model odatda nom bilan chaqiriladi; lokal yo'lni ham sinaymiz
        attempts: list[Any] = []
        if local is not None:
            attempts.append(str(local))
        attempts.extend(["v2_e2e", "v2_rnnt", "v2_ctc", "e2e"])

        for name in attempts:
            try:
                device = self._map_gigaam_device()
                model = gigaam.load_model(name, device=device)
                self._gigaam_model = model
                self._backend = "gigaam"
                logger.info("GigaAM (gigaam paketi) yuklandi: %s, device=%s", name, device)
                return True
            except Exception as exc:
                logger.debug("gigaam.load_model(%s) ishlamadi: %s", name, exc)
        return False

    def _try_load_transformers(self) -> bool:
        """Lokal GigaAMModel (trust_remote_code) — CTC, eshitilgan til alifbosida."""
        local = self._local_model_path()
        if local is None:
            return False

        try:
            import torch  # noqa: WPS433
            from transformers import AutoModel  # noqa: WPS433
        except Exception as exc:
            logger.warning("transformers/torch ASR uchun yetarli emas: %s", exc)
            self._load_error = str(exc)
            return False

        _stub_optional_asr_imports()
        try:
            model_obj = AutoModel.from_pretrained(
                str(local),
                local_files_only=True,
                trust_remote_code=True,
            )
            # float32 majburiy: GigaAM SpecScaler clamp(1e9) float16 da
            # MPS overflow beradi ("cannot be converted to c10::Half").
            # Encoder o'zi autocast(fp16) ishlatadi.
            if self._device == "cuda":
                model_obj = model_obj.to(device="cuda")
            elif self._device == "mps":
                try:
                    model_obj = model_obj.to(device="mps")
                except Exception:
                    model_obj = model_obj.to("cpu")
            else:
                model_obj = model_obj.to("cpu")
            model_obj.eval()
            self._hf_model = model_obj
            self._hf_processor = None
            self._backend = "transformers"
            logger.info("GigaAM Transformers orqali yuklandi: %s", local)
            return True
        except Exception as exc:
            logger.warning("GigaAM Transformers yuklanmadi: %s", exc)
            self._load_error = str(exc)
            return False

    @staticmethod
    def _instantiate_hf_model(path: str, config):
        """CTC / seq2seq / umumiy AutoModel ni ketma-ket sinaydi."""
        from transformers import (  # noqa: WPS433
            AutoModel,
            AutoModelForCTC,
            AutoModelForSpeechSeq2Seq,
        )

        loaders = (
            AutoModelForSpeechSeq2Seq,
            AutoModelForCTC,
            AutoModel,
        )
        for loader in loaders:
            try:
                return loader.from_pretrained(path, local_files_only=True, low_cpu_mem_usage=True)
            except Exception:
                continue
        logger.warning("GigaAM uchun mos AutoModel klassi topilmadi: %s", type(config).__name__)
        return None

    def _map_gigaam_device(self) -> str:
        """gigaam paketiga uzatiladigan qurilma nomi."""
        if self._device == "cuda":
            return "cuda"
        if self._device == "mps":
            return "mps"
        return "cpu"

    # ------------------------------------------------------------------
    # Transkripsiya
    # ------------------------------------------------------------------

    def transcribe(self, filepath: str | Path, language: str = "auto") -> str:
        """
        Audio yoki video faylni matnga o'giradi.

        Args:
            filepath: Lokal media fayl.
            language: ISO til kodi yoki "auto".

        Returns:
            To'liq transkripsiya matni.
        """
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Audio/video topilmadi: {path}")

        self.ensure_loaded()

        work_path = path
        tmp_wav: Path | None = None
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            tmp_wav = self._to_wav_16k(path)
            work_path = tmp_wav

        try:
            waveform, sr = self._load_audio(work_path)
            chunks = self._split_waveform(waveform, sr)
            parts: list[str] = []
            for idx, chunk in enumerate(chunks, start=1):
                logger.info("ASR bo'lagi %s/%s (%s)", idx, len(chunks), path.name)
                text = self._transcribe_array(chunk, sr, language)
                if text:
                    parts.append(text)
                empty_accelerator_cache()
            return " ".join(parts).strip()
        finally:
            if tmp_wav is not None and tmp_wav.exists():
                try:
                    tmp_wav.unlink()
                except OSError:
                    pass

    def _transcribe_array(self, waveform, sample_rate: int, language: str) -> str:
        """Bitta audio bo'lakni joriy backend bilan yozadi."""
        if self._backend == "gigaam":
            return self._transcribe_gigaam(waveform, sample_rate, language)
        return self._transcribe_hf(waveform, sample_rate, language)

    def _transcribe_gigaam(self, waveform, sample_rate: int, language: str) -> str:
        """gigaam model API (versiyaga qarab turlicha bo'lishi mumkin)."""
        model = self._gigaam_model
        assert model is not None

        # Vaqtinchalik wav — ba'zi API lar faqat yo'l qabul qiladi
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            self._write_wav(tmp_path, waveform, sample_rate)
            lang = None if language in {"auto", "", None} else language
            # Turli metod nomlarini sinash
            for method_name in ("transcribe", "transcribe_file", "__call__"):
                method = getattr(model, method_name, None)
                if method is None:
                    continue
                try:
                    result = self._call_gigaam(method, tmp_path, lang)
                    if result:
                        return result
                except TypeError:
                    try:
                        result = method(str(tmp_path))
                        return _as_text(result)
                    except Exception:
                        continue
                except Exception as exc:
                    logger.debug("gigaam.%s xato: %s", method_name, exc)
            raise RuntimeError("gigaam modelida transcribe metodi ishlamadi")
        finally:
            tmp_path.unlink(missing_ok=True)

    @staticmethod
    def _call_gigaam(method, path: Path, language: str | None) -> str:
        """Til argumentini ixtiyoriy uzatish."""
        if language:
            try:
                return _as_text(method(str(path), language=language))
            except TypeError:
                pass
        return _as_text(method(str(path)))

    def _transcribe_hf(self, waveform, sample_rate: int, language: str) -> str:
        """GigaAM CTC: eshitilgan nutqni o'sha til alifbosida yozadi."""
        import numpy as np  # noqa: WPS433
        import torch  # noqa: WPS433

        assert self._hf_model is not None
        model = self._hf_model
        inner = getattr(model, "model", model)

        if hasattr(waveform, "numpy"):
            audio_np = waveform.numpy()
        else:
            audio_np = np.asarray(waveform, dtype="float32")
        audio_np = np.squeeze(audio_np).astype("float32")

        transcribe = getattr(model, "transcribe", None) or getattr(inner, "transcribe", None)
        if callable(transcribe):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                self._write_wav(tmp_path, audio_np, sample_rate)
                return _as_text(transcribe(str(tmp_path)))
            finally:
                tmp_path.unlink(missing_ok=True)

        if hasattr(inner, "forward") and hasattr(inner, "_decode"):
            device = next(inner.parameters()).device
            wav = torch.from_numpy(np.ascontiguousarray(audio_np)).to(
                device=device, dtype=torch.float32
            )
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            length = torch.full([1], wav.shape[-1], device=device, dtype=torch.long)
            with torch.inference_mode():
                encoded, encoded_len = inner.forward(wav, length)
                text, _words = inner._decode(encoded, encoded_len, length, False)[0]
            return (text or "").strip()

        if self._hf_processor is None:
            raise RuntimeError("GigaAM CTC decode ishlamadi. torch/hydra o'rnatilganini tekshiring.")

        processor = self._hf_processor
        inputs = processor(audio_np, sampling_rate=sample_rate, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}
        with torch.inference_mode():
            logits = model(**inputs).logits
            pred_ids = torch.argmax(logits, dim=-1)
            if hasattr(processor, "batch_decode"):
                return processor.batch_decode(pred_ids)[0].strip()
        return ""

    # ------------------------------------------------------------------
    # Audio I/O
    # ------------------------------------------------------------------

    def _load_audio(self, path: Path):
        """
        Wav/PCM ni 16 kHz mono float32 ga keltiradi.

        MP3/OGG/FLAC — miniaudio (ffmpeg shart emas).
        WAV — soundfile. Video / M4A / AAC — ffmpeg (PATH yoki models/bin).
        """
        kind = _sniff_media(path)
        if kind == "html":
            raise RuntimeError(
                f"{path.name} audio emas — ichida HTML sahifa. "
                "Brauzerdan saqlangan sahifa, haqiqiy .mp3/.wav qo‘ying."
            )
        if kind == "empty":
            raise RuntimeError(f"{path.name} bo‘sh yoki juda kichik — audio emas.")
        if kind == "text":
            raise RuntimeError(
                f"{path.name} matn/HTML fayl, audio emas. Haqiqiy MP3 yoki WAV qo‘ying."
            )

        errors: list[str] = []
        suffix = path.suffix.lower()
        compressed = suffix in {".mp3", ".ogg", ".flac", ".oga"} or kind in {
            "mp3",
            "ogg",
            "flac",
        }

        if compressed or kind in {"mp3", "ogg", "flac", "unknown"}:
            try:
                return _decode_miniaudio(path)
            except Exception as exc:
                errors.append(f"miniaudio: {exc}")
                logger.debug("miniaudio o'qiy olmadi: %s", exc)

        try:
            return _decode_soundfile(path)
        except Exception as exc:
            errors.append(f"soundfile: {exc}")
            logger.debug("soundfile o'qiy olmadi: %s", exc)

        try:
            import librosa  # noqa: WPS433

            data, _sr = librosa.load(str(path), sr=AUDIO_SAMPLE_RATE, mono=True)
            return data.astype("float32"), AUDIO_SAMPLE_RATE
        except Exception as exc:
            errors.append(f"librosa: {exc}")
            logger.debug("librosa o'qiy olmadi: %s", exc)

        converter = _audio_converter()
        if converter:
            wav = self._to_wav_16k(path)
            try:
                return _decode_soundfile(wav)
            finally:
                wav.unlink(missing_ok=True)

        hint = "; ".join(errors[:3]) if errors else "dekoder topilmadi"
        if suffix in {".m4a", ".aac", ".wma"} or path.suffix.lower() in VIDEO_EXTENSIONS:
            raise RuntimeError(
                f"{path.name} ni o‘qish uchun ffmpeg (yoki macOS afconvert) kerak. "
                "ffmpeg ni `models/bin/ffmpeg` ga qo‘ying yoki PATH ga o‘rnating."
            )
        raise RuntimeError(
            f"{path.name} ni audio sifatida o‘qib bo‘lmadi ({kind}). {hint}"
        )

    def _to_wav_16k(self, path: Path) -> Path:
        """ffmpeg yoki macOS afconvert bilan 16 kHz mono wav yasaydi."""
        tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
        ffmpeg = _ffmpeg_path()
        if ffmpeg:
            cmd = [
                ffmpeg,
                "-y",
                "-i",
                str(path),
                "-ac",
                "1",
                "-ar",
                str(AUDIO_SAMPLE_RATE),
                "-f",
                "wav",
                str(tmp),
            ]
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return tmp
            except subprocess.CalledProcessError as exc:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"ffmpeg audio ajrata olmadi: {path.name}") from exc

        afconvert = shutil.which("afconvert")
        if afconvert:
            cmd = [
                afconvert,
                "-f",
                "WAVE",
                "-d",
                f"LEI16@{AUDIO_SAMPLE_RATE}",
                "-c",
                "1",
                str(path),
                str(tmp),
            ]
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return tmp
            except subprocess.CalledProcessError as exc:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"afconvert audio ajrata olmadi: {path.name}") from exc

        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"{path.name} video/M4A uchun ffmpeg kerak. "
            "Binarni `models/bin/ffmpeg` ga qo‘ying."
        )

    @staticmethod
    def _write_wav(path: Path, waveform, sample_rate: int) -> None:
        """float32 massivni wav ga yozadi."""
        import numpy as np  # noqa: WPS433
        import soundfile as sf  # noqa: WPS433

        data = np.asarray(waveform, dtype="float32")
        sf.write(str(path), data, sample_rate)

    @staticmethod
    def _split_waveform(waveform, sample_rate: int) -> list:
        """Uzun audioni AUDIO_CHUNK_SECONDS bo'laklarga ajratadi."""
        import numpy as np  # noqa: WPS433

        data = np.asarray(waveform, dtype="float32")
        hop = int(AUDIO_CHUNK_SECONDS * sample_rate)
        if data.size <= hop:
            return [data]
        return [data[i : i + hop] for i in range(0, data.size, hop)]


def _audio_converter() -> bool:
    """ffmpeg yoki macOS afconvert bormi."""
    return bool(_ffmpeg_path() or shutil.which("afconvert"))


def _ffmpeg_path() -> str | None:
    """PATH, models/bin yoki imageio-ffmpeg dagi ffmpeg."""
    from backend.config import MODELS_DIR, PROJECT_ROOT

    found = shutil.which("ffmpeg")
    if found:
        return found
    for candidate in (
        MODELS_DIR / "bin" / "ffmpeg",
        PROJECT_ROOT / "bin" / "ffmpeg",
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    try:
        import imageio_ffmpeg  # noqa: WPS433

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).is_file():
            return exe
    except Exception:
        pass
    return None


def _sniff_media(path: Path) -> str:
    """Fayl boshidagi baytlardan turini aniqlaydi (kengaytmaga ishonmaydi)."""
    try:
        raw = path.read_bytes()[:64]
    except OSError:
        return "empty"
    if len(raw) < 16:
        return "empty"
    head = raw.lstrip()
    low = head[:32].lower()
    if low.startswith(b"<!doctype") or low.startswith(b"<html") or low.startswith(b"<head"):
        return "html"
    if low.startswith(b"<?xml") or (head[:1] in {b"{", b"["} and b'"' in head[:40]):
        return "text"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        return "wav"
    if raw[:4] == b"fLaC" or raw[:4] == b"OggS":
        return "flac" if raw[:4] == b"fLaC" else "ogg"
    if raw[:3] == b"ID3" or (len(raw) > 1 and raw[0] == 0xFF and raw[1] & 0xE0 == 0xE0):
        return "mp3"
    if raw[4:8] == b"ftyp":
        return "mp4"
    if head[:1] == b"<":
        return "text"
    return "unknown"


def _mono_float32(data, nchannels: int = 1):
    import numpy as np  # noqa: WPS433

    arr = np.asarray(data, dtype="float32")
    if nchannels > 1 and arr.ndim == 1:
        arr = arr.reshape(-1, nchannels).mean(axis=1)
    elif arr.ndim > 1:
        arr = arr.mean(axis=1)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 1.5:
        arr = arr / 32768.0
    return arr.astype("float32")


def _decode_soundfile(path: Path):
    import soundfile as sf  # noqa: WPS433

    data, sr = sf.read(str(path), always_2d=False)
    data = _mono_float32(data)
    if sr != AUDIO_SAMPLE_RATE:
        data = _resample(data, sr, AUDIO_SAMPLE_RATE)
    return data, AUDIO_SAMPLE_RATE


def _decode_miniaudio(path: Path):
    """MP3 / OGG / FLAC / WAV — ffmpeg siz."""
    import miniaudio  # noqa: WPS433

    last: Exception | None = None
    decoded = None
    try:
        decoded = miniaudio.decode_file(str(path))
    except Exception as exc:
        last = exc
    if decoded is None:
        try:
            decoded = miniaudio.decode(path.read_bytes())
        except Exception as exc:
            last = exc
    if decoded is None and path.suffix.lower() == ".mp3":
        try:
            decoded = miniaudio.mp3_read_file_f32(str(path))
        except Exception as exc:
            last = exc
    if decoded is None:
        raise RuntimeError(str(last or "miniaudio decode"))
    data = _mono_float32(decoded.samples, int(decoded.nchannels))
    sr = int(decoded.sample_rate)
    if sr != AUDIO_SAMPLE_RATE:
        data = _resample(data, sr, AUDIO_SAMPLE_RATE)
    return data, AUDIO_SAMPLE_RATE


def _resample(data, src_sr: int, dst_sr: int):
    """Oddiy chiziqli qayta diskretlash (librosa bo'lmasa)."""
    import numpy as np  # noqa: WPS433

    if src_sr == dst_sr:
        return data
    try:
        import librosa  # noqa: WPS433

        return librosa.resample(data, orig_sr=src_sr, target_sr=dst_sr)
    except Exception:
        duration = data.shape[0] / float(src_sr)
        target_len = int(duration * dst_sr)
        x_old = np.linspace(0.0, 1.0, num=data.shape[0], endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
        return np.interp(x_new, x_old, data).astype("float32")


def _stub_optional_asr_imports() -> None:
    """
    modeling_gigaam.py VAD uchun pyannote import qiladi.
    Transformers check_imports buni majburiy deb hisoblaydi, CTC esa pyannote siz ishlaydi.
    """
    import sys
    import types

    names = (
        "pyannote",
        "pyannote.audio",
        "pyannote.audio.core",
        "pyannote.audio.core.task",
        "pyannote.audio.pipelines",
        "pyannote.core",
    )
    for name in names:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)


def _as_text(result: Any) -> str:
    """gigaam javobini satrga keltiradi (dict/tuple/str)."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        for key in ("text", "transcription", "result"):
            if key in result and result[key]:
                return str(result[key]).strip()
        return str(result).strip()
    if isinstance(result, (list, tuple)):
        if not result:
            return ""
        return _as_text(result[0])
    return str(result).strip()


asr_service = ASRService()
