"""
Offline LLM servisi — xulosa (summary) va o'zbekcha tarjima.

Yuklash tartibi (har bir model uchun):
    1. GGUF topilsa — llama-cpp-python (Metal / CUDA / CPU).
    2. Aks holda Transformers papkasi (config.json) — local_files_only=True.
    3. Hech narsa topilmasa — aniq xato (internetga murojaat YO'Q).

Kvantizatsiya:
    - CUDA: bitsandbytes 4-bit / 8-bit (VRAM ga qarab).
    - MPS / CPU: GGUF yoki fp16 / fp32 (bitsandbytes MPS da ishlamaydi).

Katta matnlar bo'laklab (map-reduce) qayta ishlanadi — OOM oldini olish.
"""

from __future__ import annotations

import gc
import logging
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from backend.config import (
    LLM_CONTEXT_SIZE,
    LLM_MAX_NEW_TOKENS_CHAT,
    LLM_MAX_NEW_TOKENS_SUMMARY,
    LLM_MAX_NEW_TOKENS_TRANSLATE,
    TRANSLATE_CHUNK_CHARS,
    LLM_MODELS,
    LLM_TEMPERATURE,
    TRANSLATE_TEMPERATURE,
    SUMMARY_TEMPERATURE,
    TEXT_CHUNK_CHARS,
    DeviceType,
    QuantType,
    detect_device,
    empty_accelerator_cache,
    find_gguf_file,
    find_llama_cli,
    gguf_file_usable,
    recommend_quantization,
    transformers_model_ready,
)
from backend.services.ocr_parser import ocr_parser

logger = logging.getLogger(__name__)

# Gemma / umumiy instruksiya promptlari
_SUMMARY_SYSTEM = (
    "Faqat o'zbek lotin alifbosida yoz. Kirill ishlatma. "
    "Oddiy gaplar. Yorliq yo'q. Inglizcha yo'q. "
    "HTML, href, markdown, ** va thought yo'q."
)
_SUMMARY_USER = (
    "Quyidagi matnning qisqacha mazmunini o'zbekcha 5-8 gapda yoz.\n\n"
    "{text}\n\n"
    "Xulosa:"
)
_SUMMARY_REDUCE_USER = (
    "Quyidagi xulosalarni bitta o'zbekcha qisqa matnga birlashtir. "
    "Takrorlama. Inglizcha yozma.\n\n"
    "{text}\n\n"
    "Xulosa:"
)
_TRANSLATE_SYSTEM = (
    "Siz professional hujjat tarjimonisiz. Faqat to'liq tarjima qiling. "
    "Xulosa yozmang. Qisqartirmang. Jumlalarni tashlamang. "
    "Hech qanday izoh, 'Qism', 'thought', URL yoki kirish gap qo'shmang. "
    "Sarlavha, muallif, annotatsiya/abstract, asosiy matn, iqtibos va "
    "adabiyotlar (references) ketma-ketligi hamda qator tuzilmasi 100% saqlansin. "
    "Javob faqat o'zbek lotin alifbosida, faqat tarjima matni. "
    "HTML, href, markdown, ** yoki * ishlatmang. Oddiy matn, asl qatorlardek."
)
_TRANSLATE_USER = (
    "Quyidagi matnni o'zbek tiliga (lotin alifbosi) TO'LIQ tarjima qiling. "
    "Xulosa qilmang. Qisqartirmang. Tuzilmani o'zgartirmang. "
    "Agar matn allaqachon o'zbekcha bo'lsa, tuzilmani o'zgartirmay qaytaring. "
    "Faqat tarjima matnini yozing.\n\n"
    "MATN:\n{text}"
)
_RAG_SYSTEM = (
    "Siz mahalliy hujjatlar bo'yicha savollarga javob beradigan yordamchisiz. "
    "Faqat berilgan KONTEKST dagi ma'lumotga tayaning. "
    "Kontekstda javob bo'lmasa, shuni ochiq ayting. "
    "O'ylab topmang. Javobni savol tilida yozing."
)
_RAG_USER = (
    "KONTEKST:\n{context}\n\n"
    "SAVOL:\n{question}\n\n"
    "Kontekst asosida qisqa, aniq javob yozing."
)


class LLMService:
    """
    Gemma 4 / Gemma 26 ni offline yuklaydigan va ikki vazifani bajaradigan servis.

    Model almashtirilganda avvalgi og'irliklar xotiradan chiqariladi.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._backend: str | None = None  # "gguf" | "llama-cli" | "transformers"
        self._model_key: str | None = None
        self._quant: QuantType | None = None
        self._device: DeviceType = detect_device()

        # Transformers
        self._hf_model = None
        self._hf_tokenizer = None

        # llama.cpp
        self._llama = None
        self._cli_bin: Path | None = None
        self._gguf_path: Path | None = None

        # Apple MLX — GPU stream bitta oqimda bo'lishi shart
        self._mlx_model = None
        self._mlx_tokenizer = None
        self._mlx_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-llm")
        self._task_temperature: float | None = None

    def _effective_temperature(self) -> float:
        if self._task_temperature is not None:
            return self._task_temperature
        return LLM_TEMPERATURE

    # ------------------------------------------------------------------
    # Holat
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """Model xotiraga yuklanganmi."""
        return (
            self._llama is not None
            or self._hf_model is not None
            or self._cli_bin is not None
            or self._mlx_model is not None
        )

    def status(self) -> dict[str, Any]:
        """UI uchun joriy model holati."""
        return {
            "ready": self.is_ready,
            "model_key": self._model_key,
            "backend": self._backend,
            "device": self._device,
            "quantization": self._quant,
        }

    # ------------------------------------------------------------------
    # Yuklash / tushirish
    # ------------------------------------------------------------------

    def ensure_loaded(self, model_key: str, quantization: QuantType | str = "auto") -> None:
        """
        Kerakli model allaqachon yuklangan bo'lsa, qayta yuklamaydi.
        Aks holda avvalgisini tushirib, yangisini ochadi.
        """
        quant: QuantType
        if quantization in {"none", "8bit", "4bit"}:
            quant = quantization  # type: ignore[assignment]
        else:
            quant = recommend_quantization(self._device, model_key)

        with self._lock:
            if self.is_ready and self._model_key == model_key and self._quant == quant:
                return
            self._unload_unlocked()
            self._load_unlocked(model_key, quant)

    def unload(self) -> None:
        """Modelni xotiradan to'liq chiqaradi."""
        with self._lock:
            self._unload_unlocked()

    def _unload_unlocked(self) -> None:
        """Qulf ostida chaqiriladigan tushirish."""
        self._llama = None
        self._cli_bin = None
        self._gguf_path = None
        self._mlx_model = None
        self._mlx_tokenizer = None
        self._hf_model = None
        self._hf_tokenizer = None
        self._backend = None
        self._model_key = None
        self._quant = None
        gc.collect()
        empty_accelerator_cache()
        logger.info("LLM xotiradan tushirildi")

    def _load_unlocked(self, model_key: str, quant: QuantType) -> None:
        """Qulf ostida modelni diskdan yuklaydi."""
        if model_key not in LLM_MODELS:
            raise ValueError(f"Noma'lum model: {model_key}. Mavjud: {list(LLM_MODELS)}")

        spec = LLM_MODELS[model_key]
        self._device = detect_device()

        if transformers_model_ready(spec):
            mlx_error = self._try_load_mlx(spec.transformers_dir)
            if mlx_error is None:
                pass
            elif self._torch_available():
                self._load_transformers(spec.transformers_dir, quant)
                self._backend = "transformers"
            else:
                raise RuntimeError(
                    f"'{spec.label}' MLX orqali yuklanmadi (torch shart emas). "
                    f"{mlx_error} "
                    "Sozlamada Gemma 4e ni tanlang — Gemma 26 hali to'liq bo'lmasa, "
                    "yoki ./scripts/fetch_models.sh ni davom ettiring."
                )
        else:
            gguf = find_gguf_file(spec)
            if gguf is not None:
                self._load_gguf(gguf)
            else:
                raise FileNotFoundError(
                    f"'{spec.label}' modeli diskda yo'q yoki hali to'liq yuklanmagan. "
                    f"Papka: {spec.transformers_dir}/\n"
                    f"Skript: ./scripts/fetch_models.sh"
                )

        self._model_key = model_key
        self._quant = quant
        logger.info(
            "LLM yuklandi: %s (%s, qurilma=%s, quant=%s)",
            model_key,
            self._backend,
            self._device,
            quant,
        )

    def _load_gguf(self, gguf_path: Path) -> None:
        """
        llama-cpp-python orqali GGUF yuklash.

        n_gpu_layers:
            - CUDA / MPS: -1 (maksimal offload)
            - CPU: 0
        n_ctx va n_batch kichikroq — 8GB VRAM da OOM bo'lmasligi uchun.
        """
        if not gguf_file_usable(gguf_path):
            raise RuntimeError(
                f"GGUF fayl buzilgan yoki to‘liq emas: {gguf_path.name}. "
                f"O‘chirib qayta yuklang: ./scripts/fetch_gemma4.sh"
            )
        try:
            from llama_cpp import Llama  # noqa: WPS433
        except Exception:
            cli = find_llama_cli()
            if cli is None:
                raise RuntimeError(
                    "GGUF topildi, lekin ishga tushirgich yo'q. "
                    "Python 3.14 da llama-cpp-python g'ildiragi yo'q, Xcode ham o'rnatilmagan. "
                    "README 2.2: llama-cli ni models/llama.cpp/ ga qo'ying "
                    "(llama.cpp Releases → macos-arm64)."
                )
            self._cli_bin = cli
            self._gguf_path = gguf_path
            self._backend = "llama-cli"
            logger.info("GGUF llama-cli orqali: %s + %s", cli, gguf_path.name)
            return

        offload = -1 if self._device in {"cuda", "mps"} else 0
        # RTX 3070 8GB: kontekstni cheklash
        n_ctx = 2048 if self._device == "cuda" else LLM_CONTEXT_SIZE
        n_batch = 128 if self._device == "cuda" else 256

        logger.info("GGUF yuklanmoqda: %s (gpu_layers=%s, ctx=%s)", gguf_path, offload, n_ctx)
        common = dict(
            model_path=str(gguf_path),
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_gpu_layers=offload,
            n_threads=_cpu_threads(),
            verbose=False,
        )
        try:
            self._llama = Llama(**common, chat_format="gemma")
        except Exception as exc:
            logger.warning("chat_format=gemma ishlamadi, andoza avtomatik: %s", exc)
            self._llama = Llama(**common)
        self._backend = "gguf"

    def _load_transformers(self, model_dir: Path, quant: QuantType) -> None:
        """
        Hugging Face Transformers — faqat lokal papka, local_files_only=True.
        """
        try:
            import torch  # noqa: WPS433
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: WPS433
        except Exception as exc:
            raise RuntimeError(
                "Transformers / torch o'rnatilmagan. README ni qarang."
            ) from exc

        logger.info("Transformers yuklanmoqda: %s (quant=%s)", model_dir, quant)

        self._hf_tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            local_files_only=True,
            trust_remote_code=False,
        )

        kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": False,
            "low_cpu_mem_usage": True,
        }

        # CUDA da bitsandbytes kvantizatsiyasi
        if self._device == "cuda" and quant in {"4bit", "8bit"}:
            bnb = self._bitsandbytes_config(quant)
            if bnb is not None:
                kwargs["quantization_config"] = bnb
                kwargs["device_map"] = "auto"
            else:
                kwargs["torch_dtype"] = torch.float16
                kwargs["device_map"] = "auto"
        elif self._device == "cuda":
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "auto"
        elif self._device == "mps":
            # MPS: float16, to'liq modelni MPS ga
            kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = torch.float32

        self._hf_model = AutoModelForCausalLM.from_pretrained(str(model_dir), **kwargs)
        self._hf_model.eval()

        if self._device == "mps":
            try:
                self._hf_model.to("mps")
            except Exception as exc:
                logger.warning("MPS ga ko'chirish muvaffaqiyatsiz, CPU qoladi: %s", exc)

    @staticmethod
    def _bitsandbytes_config(quant: QuantType):
        """4-bit / 8-bit konfiguratsiya. Paket yo'q bo'lsa None."""
        try:
            import torch  # noqa: WPS433
            from transformers import BitsAndBytesConfig  # noqa: WPS433
        except Exception:
            return None

        if quant == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        if quant == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        return None

    # ------------------------------------------------------------------
    # Vazifalar
    # ------------------------------------------------------------------

    def summarize(self, text: str) -> str:
        """
        Hujjatning qisqacha tavsifini yaratadi.

        Uzun matn: har bir bo'lak alohida xulosalanadi, so'ng birlashtiriladi.
        """
        text = (text or "").strip()
        if not text:
            return ""

        chunks = ocr_parser.iter_text_chunks(text, max_chars=TEXT_CHUNK_CHARS)
        if not chunks:
            return ""

        prev_temp = self._task_temperature
        self._task_temperature = SUMMARY_TEMPERATURE
        try:
            if len(chunks) == 1:
                raw = self._generate(
                    _SUMMARY_SYSTEM,
                    _SUMMARY_USER.format(text=chunks[0]),
                    LLM_MAX_NEW_TOKENS_SUMMARY,
                )
                return _clean_summary(raw)

            partial: list[str] = []
            for idx, chunk in enumerate(chunks, start=1):
                logger.info("Xulosa bo'lagi %s/%s", idx, len(chunks))
                piece = self._generate(
                    _SUMMARY_SYSTEM,
                    _SUMMARY_USER.format(text=chunk),
                    LLM_MAX_NEW_TOKENS_SUMMARY,
                )
                piece = _clean_summary(piece)
                if piece:
                    partial.append(piece)
                empty_accelerator_cache()

            joined = "\n\n".join(partial)
            if len(joined) <= TEXT_CHUNK_CHARS:
                raw = self._generate(
                    _SUMMARY_SYSTEM,
                    _SUMMARY_REDUCE_USER.format(text=joined),
                    LLM_MAX_NEW_TOKENS_SUMMARY,
                )
                return _clean_summary(raw)
            return _clean_summary(joined)
        finally:
            self._task_temperature = prev_temp

    def translate_to_uzbek(self, text: str) -> str:
        """
        Matnni o'zbek tiliga tarjima qiladi, asl paragraf tuzilmasini saqlaydi.
        """
        from backend.services.translation_text import (
            clean_translation_output,
            iter_document_units,
            join_document_units,
        )

        text = (text or "").strip()
        if not text:
            return ""

        units = iter_document_units(text, TRANSLATE_CHUNK_CHARS)
        translated: list[tuple[str, str]] = []
        prev_temp = self._task_temperature
        self._task_temperature = TRANSLATE_TEMPERATURE
        try:
            for idx, (chunk, sep) in enumerate(units, start=1):
                logger.info("Tarjima bo'lagi %s/%s", idx, len(units))
                piece = self._generate(
                    _TRANSLATE_SYSTEM,
                    _TRANSLATE_USER.format(text=chunk),
                    LLM_MAX_NEW_TOKENS_TRANSLATE,
                )
                piece = clean_translation_output(piece)
                translated.append((piece or chunk, sep))
                empty_accelerator_cache()
        finally:
            self._task_temperature = prev_temp
        return clean_translation_output(join_document_units(translated))

    def process_text(self, text: str) -> tuple[str, str]:
        """
        Avval tuzilmali o'zbekcha tarjima, so'ng butun matn xulosasi.

        Returns:
            (summary, translation_uz)
        """
        translation = self.translate_to_uzbek(text)
        empty_accelerator_cache()
        summary = self.summarize(text)
        return summary, translation

    def answer_with_context(self, question: str, context: str) -> str:
        """RAG konteksti asosida savolga javob beradi."""
        return self._generate(
            _RAG_SYSTEM,
            _RAG_USER.format(context=context, question=question),
            LLM_MAX_NEW_TOKENS_CHAT,
        )

    # ------------------------------------------------------------------
    # Generatsiya
    # ------------------------------------------------------------------

    def _generate(self, system: str, user: str, max_new_tokens: int) -> str:
        """Joriy backend orqali bitta javob generatsiya qiladi."""
        if not self.is_ready:
            raise RuntimeError("LLM yuklanmagan. Avval ensure_loaded() chaqiring.")

        if self._backend == "gguf":
            return self._generate_gguf(system, user, max_new_tokens)
        if self._backend == "llama-cli":
            return self._generate_cli(system, user, max_new_tokens)
        if self._backend == "mlx":
            return self._generate_mlx(system, user, max_new_tokens)
        return self._generate_hf(system, user, max_new_tokens)

    @staticmethod
    def _torch_available() -> bool:
        try:
            import torch  # noqa: F401, WPS433
        except Exception:
            return False
        return True

    def _try_load_mlx(self, model_dir: Path) -> str | None:
        """
        Apple MLX orqali yuklash.

        Returns:
            None — muvaffaqiyat.
            str — xato matni (yuklanmadi).
        """
        try:
            from mlx_lm import load  # noqa: WPS433
        except Exception as exc:
            return f"mlx-lm import qilinmadi: {exc}"

        def _load() -> None:
            logger.info("MLX model yuklanmoqda: %s", model_dir)
            self._mlx_model, self._mlx_tokenizer = load(str(model_dir))
            self._backend = "mlx"

        try:
            self._mlx_pool.submit(_load).result()
            return None
        except Exception as exc:
            logger.warning("MLX yuklanmadi (%s): %s", model_dir, exc)
            self._mlx_model = None
            self._mlx_tokenizer = None
            return str(exc)[:400]

    def _generate_mlx(self, system: str, user: str, max_new_tokens: int) -> str:
        """mlx-lm generate — doim bitta MLX oqimida (GPU stream)."""

        def _run() -> str:
            from mlx_lm import generate  # noqa: WPS433

            if self._mlx_model is None or self._mlx_tokenizer is None:
                raise RuntimeError("MLX model yuklanmagan")
            tokenizer = self._mlx_tokenizer
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                try:
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    prompt = f"{system}\n\n{user}\n\nJavob:"
            except Exception:
                prompt = f"{system}\n\n{user}\n\nJavob:"
            gen_kwargs: dict[str, Any] = {
                "max_tokens": max_new_tokens,
                "verbose": False,
            }
            try:
                from mlx_lm.sample_utils import make_sampler

                gen_kwargs["sampler"] = make_sampler(temp=self._effective_temperature())
            except Exception:
                pass
            text = generate(
                self._mlx_model,
                tokenizer,
                prompt=prompt,
                **gen_kwargs,
            )
            return _clean_completion(text)

        try:
            return self._mlx_pool.submit(_run).result()
        except Exception as exc:
            if "Stream(gpu" not in str(exc):
                raise
            logger.warning("MLX stream boshqa oqimda — qayta yuklanadi")
            if self._model_key:
                spec = LLM_MODELS[self._model_key]
                err = self._try_load_mlx(spec.transformers_dir)
                if err:
                    raise RuntimeError(err) from exc
            return self._mlx_pool.submit(_run).result()

    def _generate_cli(self, system: str, user: str, max_new_tokens: int) -> str:
        """Oldindan yig'ilgan llama-cli orqali (pip/Xcode shart emas)."""
        assert self._cli_bin is not None and self._gguf_path is not None
        prompt = f"{system}\n\n{user}\n\nJavob:"
        ngl = "99" if self._device in {"cuda", "mps"} else "0"
        cmd = [
            str(self._cli_bin),
            "-m",
            str(self._gguf_path),
            "-p",
            prompt,
            "-n",
            str(max_new_tokens),
            "--temp",
            str(self._effective_temperature()),
            "-ngl",
            ngl,
            "-c",
            str(LLM_CONTEXT_SIZE),
            "-no-cnv",
            "--log-disable",
        ]
        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("llama-cli vaqt tugadi (10 daqiqa).") from exc
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "")[-1200:]
            low = err.lower()
            if "not within the file bounds" in low or "corrupted or incomplete" in low:
                raise RuntimeError(
                    f"GGUF fayl buzilgan yoki to‘liq yuklanmagan: {self._gguf_path.name}. "
                    f"Bu faylni o‘chiring va Gemma 4 ni qayta yuklang: ./scripts/fetch_gemma4.sh"
                )
            raise RuntimeError(f"llama-cli xatosi: {err}")
        text = result.stdout or ""
        if "Javob:" in text:
            text = text.split("Javob:", 1)[-1]
        return _clean_completion(text)

    def _generate_gguf(self, system: str, user: str, max_new_tokens: int) -> str:
        """llama.cpp chat completion."""
        assert self._llama is not None
        try:
            result = self._llama.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self._effective_temperature(),
                max_tokens=max_new_tokens,
            )
            content = result["choices"][0]["message"]["content"]
            return _clean_completion(content)
        except Exception as exc:
            logger.exception("GGUF generatsiya xatosi: %s", exc)
            raise RuntimeError(f"LLM (GGUF) xatosi: {exc}") from exc

    def _generate_hf(self, system: str, user: str, max_new_tokens: int) -> str:
        """Transformers generate + chat template."""
        assert self._hf_model is not None and self._hf_tokenizer is not None
        import torch  # noqa: WPS433

        tokenizer = self._hf_tokenizer
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            try:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                prompt = f"{system}\n\n{user}\n\nJavob:"
        except Exception:
            prompt = f"{system}\n\n{user}\n\nJavob:"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=LLM_CONTEXT_SIZE)
        device = next(self._hf_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self._hf_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=self._effective_temperature() > 0,
                temperature=max(self._effective_temperature(), 0.01),
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = output_ids[0, inputs["input_ids"].shape[-1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        return _clean_completion(text)


def _layout_blocks(text: str, max_chars: int) -> list[str]:
    """Paragraf chegarasida bo'laklaydi — asl tuzilma saqlansin."""
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not parts:
        return ocr_parser.iter_text_chunks(text, max_chars=max_chars)
    blocks: list[str] = []
    buf: list[str] = []
    size = 0
    for part in parts:
        if size + len(part) + 2 > max_chars and buf:
            blocks.append("\n\n".join(buf))
            buf = [part]
            size = len(part)
        else:
            buf.append(part)
            size += len(part) + 2
    if buf:
        blocks.append("\n\n".join(buf))
    return blocks


def _strip_translation_preamble(text: str) -> str:
    """'Here is the translation' kabi inglizcha kirishni olib tashlaydi."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    prefixes = (
        "here is the uzbek translation:",
        "here is the translation:",
        "uzbek translation:",
        "translation:",
        "tarjima:",
        "o'zbekcha tarjima:",
        "o‘zbekcha tarjima:",
    )
    low = cleaned.lower()
    for prefix in prefixes:
        if low.startswith(prefix):
            cleaned = cleaned[len(prefix) :].lstrip(" \n:-")
            break
    return cleaned.strip()


def _clean_completion(text: str) -> str:
    """Model javobidagi thought, HTML, markdown va bo'shliqlarni tozalaydi."""
    if not text:
        return ""
    from backend.services.translation_text import clean_translation_output

    cleaned = text.strip()
    for marker in ("<end_of_turn>", "<start_of_turn>", "</s>", "<eos>"):
        cleaned = cleaned.replace(marker, "")
    return clean_translation_output(cleaned)


def _clean_summary(text: str) -> str:
    from backend.services.translation_text import clean_summary_output

    return clean_summary_output(text)


def _cpu_threads() -> int:
    """llama.cpp uchun oqimlar soni — yadro sonining 75%. """
    import os

    cpu = os.cpu_count() or 4
    return max(2, int(cpu * 0.75))


# Pipeline uchun yagona misol
llm_service = LLMService()
