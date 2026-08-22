"""
Meta SeamlessM4T v2 (Speech-to-Text) — tojik va boshqa tillar uchun.

Model: facebook/seamless-m4t-v2-large
Papka: models/ASR modellar/SeamlessM4T-v2/
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from backend.config import (
    ASR_MODELS,
    AUDIO_SAMPLE_RATE,
    SEAMLESS_DIR,
    VIDEO_EXTENSIONS,
    detect_device,
    empty_accelerator_cache,
    find_asr_model_dir,
    resolve_seamless_tgt_lang,
)

logger = logging.getLogger(__name__)


class SeamlessService:
    """SeamlessM4T v2 ni bir marta yuklab, xotirada ushlab turadi."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._device = detect_device()
        self._model = None
        self._processor = None
        self._load_error: str | None = None

    @property
    def is_ready(self) -> bool:
        return self._model is not None and self._processor is not None

    def status(self) -> dict[str, Any]:
        spec = ASR_MODELS["seamless-m4t-v2"]
        found = find_asr_model_dir(spec)
        return {
            "ready": self.is_ready,
            "backend": "transformers" if self.is_ready else None,
            "model_key": "seamless-m4t-v2",
            "device": self._device,
            "model_dir": str(found or SEAMLESS_DIR),
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
            try:
                self._load()
            except Exception as exc:
                self._load_error = str(exc)[:800]
                raise

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._processor = None
            empty_accelerator_cache()

    def transcribe(
        self,
        filepath: str | Path,
        language: str = "auto",
        country: str = "",
    ) -> str:
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Audio/video topilmadi: {path}")
        self.ensure_loaded()
        from backend.services.asr_service import asr_service

        work_path = path
        tmp_wav: Path | None = None
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            tmp_wav = asr_service._to_wav_16k(path)
            work_path = tmp_wav
        try:
            waveform, sr = asr_service._load_audio(work_path)
            chunks = asr_service._split_waveform(waveform, sr)
            tgt = resolve_seamless_tgt_lang(language, country)
            parts: list[str] = []
            for idx, chunk in enumerate(chunks, start=1):
                logger.info("Seamless bo'lagi %s/%s (%s) tgt=%s", idx, len(chunks), path.name, tgt)
                text = self._transcribe_chunk(chunk, sr, tgt)
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

    def _load(self) -> None:
        spec = ASR_MODELS["seamless-m4t-v2"]
        folder = find_asr_model_dir(spec)
        if folder is None:
            raise FileNotFoundError(
                "SeamlessM4T v2 topilmadi. "
                "`scripts/download_models.py seamless` ni ishga tushiring."
            )
        import torch
        from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

        logger.info("SeamlessM4T v2 yuklanmoqda: %s", folder)
        self._processor = AutoProcessor.from_pretrained(str(folder), local_files_only=True)
        dtype = torch.float16 if self._device in {"cuda", "mps"} else torch.float32
        try:
            self._model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
                str(folder),
                local_files_only=True,
                torch_dtype=dtype,
            )
            device = "cpu" if self._device == "cpu" else self._device
            self._model.to(device)
        except Exception:
            if dtype != torch.float32:
                logger.warning("Seamless float16 yuklanmadi — float32 bilan qayta uriniladi")
                self._model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
                    str(folder),
                    local_files_only=True,
                    torch_dtype=torch.float32,
                )
                device = "cpu" if self._device == "cpu" else self._device
                self._model.to(device)
            else:
                raise
        self._model.eval()
        logger.info("SeamlessM4T v2 yuklandi (qurilma=%s)", self._device)

    def _transcribe_chunk(self, waveform, sample_rate: int, tgt_lang: str) -> str:
        import numpy as np
        import torch

        assert self._model is not None and self._processor is not None
        audio = np.squeeze(np.asarray(waveform, dtype="float32"))
        if sample_rate != AUDIO_SAMPLE_RATE:
            import librosa

            audio = librosa.resample(
                audio, orig_sr=sample_rate, target_sr=AUDIO_SAMPLE_RATE
            ).astype("float32")
            sample_rate = AUDIO_SAMPLE_RATE
        try:
            inputs = self._processor(
                audio=audio, sampling_rate=sample_rate, return_tensors="pt"
            )
        except TypeError:
            inputs = self._processor(
                audios=audio, sampling_rate=sample_rate, return_tensors="pt"
            )
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}
        with torch.inference_mode():
            tokens = self._model.generate(**inputs, tgt_lang=tgt_lang)
        text = self._processor.batch_decode(tokens, skip_special_tokens=True)[0]
        return (text or "").strip()


seamless_service = SeamlessService()
