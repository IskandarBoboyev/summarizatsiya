# Lokal hujjatlar platformasi

100% **offline** (internetsiz) ishlaydigan, modulli hujjat qayta ishlash tizimi.

- Papkalarni kuzatish → yangi fayl avtomatik navbatga tushadi
- PDF / DOCX / rasm → matn ajratish (OCR)
- Gemma 4 / Gemma 26 → qisqacha xulosa
- **NLLB-200 1.3B** → o‘zbekcha (lotin) tarjima
- Audio / video → **GigaAM-Multilingual** transkripsiya
- Qurilma avtomatik: **CUDA → MPS → CPU**
- Xotira: matn/audio **bo‘laklab** ishlanadi, CUDA da **4-bit / 8-bit** kvantizatsiya

Tashqi API (OpenAI va h.k.) **yo‘q**. Hugging Face ham faqat lokal papka / keshdan o‘qiydi (`HF_HUB_OFFLINE=1`).

Yopiq GitHub: `git@github.com:IskandarBoboyev/summarizatsiya.git`. Model og‘irliklari, `.venv`, `data/*.db` va loglar gitga kirmaydi — har bir dasturchi `scripts/download_models.py` bilan modellarni o‘z mashinasiga yuklaydi.

---

## Apparat mosligi

| Muhit | Tavsiya | Kvantizatsiya |
|---|---|---|
| **Apple Silicon Mac**, 48 GB Unified Memory (MPS) | Gemma 4 — fp16; Gemma 26 — GGUF Q4/Q5 | `bitsandbytes` ishlamaydi, GGUF afzal |
| **NVIDIA RTX 3070 8 GB** (CUDA) | Gemma 4 — 8-bit yoki GGUF Q4; Gemma 26 — **faqat GGUF Q4** | Avtomatik `4bit` / `8bit` |
| CPU (zaxira) | Faqat kichik GGUF | Sekin |

---

## Papka tuzilmasi

```
local_doc_platform/
├── backend/
│   ├── main.py                 # FastAPI
│   ├── config.py               # qurilma, yo'llar, kvantizatsiya
│   ├── database.py             # SQLite tarix
│   ├── pipeline.py             # navbat + OCR/ASR/LLM orkestratsiya
│   └── services/
│       ├── watcher_service.py
│       ├── ocr_parser.py
│       ├── llm_service.py
│       └── asr_service.py
├── frontend/                   # SPA, CDN yo'q
├── docs/MODELLAR.md            # modellarni ishga tushirish
├── models/                     # barcha offline og'irliklar
├── watched_folders/            # birlamchi monitoring
├── data/                       # documents.db
├── requirements.txt
└── README.md
```

---

## 1. Tizim dasturlari

### macOS (Apple Silicon)

```bash
brew install tesseract tesseract-lang ffmpeg
```

`tesseract-lang` ichida `uzb`, `rus`, `eng` bo‘lishi kerak. Qo‘lda qo‘yish:

```bash
# traineddata fayllarni shu yerga nusxalang:
# local_doc_platform/models/tessdata/uzb.traineddata
# local_doc_platform/models/tessdata/rus.traineddata
# local_doc_platform/models/tessdata/eng.traineddata
```

### Ubuntu / Debian (Linux + NVIDIA)

```bash
sudo apt update
sudo apt install -y tesseract-ocr tesseract-ocr-uzb tesseract-ocr-rus tesseract-ocr-eng ffmpeg
```

NVIDIA driver + CUDA toolkit o‘rnatilgan bo‘lishi shart (`nvidia-smi` ishlasin).

### Windows

1. [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) — o‘rnatib, `tesseract.exe` PATH da bo‘lsin.
2. [ffmpeg](https://www.gyan.dev/ffmpeg/builds/) — PATH ga qo‘shing.
3. NVIDIA: Game Ready / Studio driver.

> Tessdata ni internetsiz ishlatish: `*.traineddata` ni `models/tessdata/` ga ko‘chiring.

---

## 2. Python muhiti

Python **3.10 – 3.12** (3.13 da ba’zi paketlar hali noqulay).

```bash
cd local_doc_platform
python3 -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -U pip wheel
pip install -r requirements.txt
```

### 2.1. LLM Mac da — MLX (torch shart emas)

Apple Silicon da Gemma 4e **mlx-lm** bilan ishlaydi. `pip install mlx mlx-lm` yetarli.

PyTorch faqat **GigaAM audio transkripsiya** yoki Linux/CUDA Transformers uchun kerak.

**Ixtiyoriy (ASR / CUDA):**

```bash
pip install torch torchaudio
```

Mac tekshiruv:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

**Linux / Windows + CUDA 12.x (RTX 3070):**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Tekshiruv:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

4/8-bit kvantizatsiya (faqat CUDA):

```bash
pip install bitsandbytes
```

**CPU-only** (sekin):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 2.2. llama-cpp-python (GGUF, tavsiya)

RTX 3070 8 GB da Gemma 26 ni ishlatishning amalda yagona yo‘li — GGUF.

Python **3.10–3.12** va **Xcode Command Line Tools** kerak (manbadan yig‘ish).
Python 3.14 da tayyor g‘ildirak yo‘q — `pip` Xcode siz yiqiladi.

**Mac (Metal / MPS), Python 3.10–3.12 + Xcode:**

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

**Mac, Xcode yo‘q yoki Python 3.14** — `llama-cli` binary (pip shart emas):

1. [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases) dan `llama-b*-bin-macos-arm64.tar.gz` ni yuklang.
2. Ichidagi `llama-cli` ni shu yerga qo‘ying: `models/llama.cpp/llama-cli`
3. Serverni qayta ishga tushiring. GGUF to‘liq (~4.8 GB) bo‘lgach, **Qayta** bosing.

**Linux + CUDA:**

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

**Windows + CUDA:** Visual Studio Build Tools kerak, yoki oldindan yig‘ilgan g‘ildirak.

**CPU:**

```bash
pip install llama-cpp-python
```

---

## 3. Modellarni offline joylashtirish

Modellarni **interneti bor** kompyuterda yuklab oling, keyin USB / lokal tarmoq orqali `models/` ga ko‘chiring. Ishchi mashina tarmoqsiz qolishi mumkin.

### 3.1. Gemma 4 va Gemma 26

UI dagi nomlar:

| UI | Disk papkasi / fayl | Haqiqiy og‘irlik (tavsiya) |
|---|---|---|
| Gemma 4 | `models/gemma-4/` yoki `*gemma*4*.gguf` | Gemma 3 4B Instruct |
| Gemma 26 | `models/gemma-26/` yoki `*gemma*26*.gguf` / `*gemma*27*.gguf` | Gemma 3 27B Instruct |

**GGUF (Mac 48 GB yoki RTX 3070 uchun eng qulay):**

Internetli mashinada (misol, `huggingface-cli`):

```bash
# Misol nomlar — o'zingizdagi GGUF fayl nomiga qarab o'zgartiring
huggingface-cli download <repo> <file>.gguf --local-dir ./gemma-gguf
```

Keyin nusxa:

```text
local_doc_platform/models/gemma-4-Q4_K_M.gguf
local_doc_platform/models/gemma-26-Q4_K_M.gguf
```

RTX 3070: **Q4_K_M** yoki **Q3_K_M**.  
Mac 48 GB: Gemma 4 uchun Q5/Q6 yoki Transformers fp16; Gemma 26 uchun Q4/Q5.

**Transformers (safetensors) papkasi:**

```text
models/gemma-4/config.json
models/gemma-4/tokenizer.json
models/gemma-4/*.safetensors

models/gemma-26/config.json
...
```

Gemma og‘irliklari Google litsenziyasiga bo‘ysunadi — Hugging Face da qabul qiling, keyin **offline** ishlating.

### 3.2. GigaAM-Multilingual (ASR)

Manba: `ai-sage/GigaAM-Multilingual`

```text
models/gigaam-multilingual/config.json
models/gigaam-multilingual/  + tokenizer / og'irlik fayllari
```

Yoki HF keshini butunlay ko‘chirish:

```text
models/hf_cache/hub/models--ai-sage--GigaAM-Multilingual/snapshots/<hash>/
```

Ixtiyoriy paket (internetli mashinada o‘rnatib, keyin wheel ni olib o‘ting):

```bash
pip install gigaam
```

Servis avval `gigaam` paketini, keyin Transformers `AutoModel` ni sinaydi. Hech biri topilmasa, audio kartasida tushunarli xato chiqadi — qolgan hujjatlar ishlashda davom etadi.

### 3.3. NLLB-200 (tarjima)

Manba: `facebook/nllb-200-distilled-1.3B`

```text
models/tarjima_model/config.json
models/tarjima_model/  + tokenizer / og'irlik fayllari
```

Yuklash (interneti bor mashinada):

```bash
.venv/bin/python -u scripts/download_models.py nllb
```

Tarjima doim **NLLB mikroserveri** (port 8003) orqali o‘zbek lotiniga (`uzn_Latn`) ketadi. Gemma faqat xulosa va chat uchun.

### 3.4. EasyOCR (rasm, Tesseract bo‘lmasa)

EasyOCR birinchi marta internetdan model oladi. Buni **oldindan** qiling, keyin papkani ko‘chiring:

```text
models/easyocr/
```

Kodda `download_enabled=False` — ish vaqtida tarmoqqa chiqilmaydi.

---

## 4. Ishga tushirish

Modelllar **alohida mikroserverda** doimiy ishlaydi (qayta yuklanmaydi):

| Servis | Port | Vazifa |
|---|---|---|
| UI + pipeline | 8000 | OCR, navbat, RAG, chat |
| Gemma | 8001 | xulosa, RAG javob |
| GigaAM | 8002 | audio transkripsiya |
| NLLB | 8003 | o‘zbekcha tarjima |

```bash
./start.sh
# yoki:
./scripts/start_workers.sh
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Brauzer: [http://127.0.0.1:8000](http://127.0.0.1:8000)

Yuqori panelda **Gemma / GigaAM / NLLB** nuqtalari jonli holatni ko‘rsatadi (yashil = ishlayapti). Batafsil: [docs/MODELLAR.md](docs/MODELLAR.md)

```bash
./scripts/check_models.sh
curl -sS http://127.0.0.1:8000/api/models/health
```

Workerlarsiz (hammasi bitta jarayonda): `DOC_USE_MODEL_SERVERS=0`

Tarmoqdan (faqat lokal LAN, tashqi API emas):

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

`config.py` importida `HF_HUB_OFFLINE` allaqachon o‘rnatiladi. Qo‘lda export qilish — qo‘shimcha kafolat.

---

## 5. Foydalanish

1. Yuqori paneldan **Gemma 4** yoki **Gemma 26** ni tanlang (diskda bo‘lgani yashil/oddiy, yo‘qligi `(diskda yo‘q)`).
2. Audio uchun **ASR tili**ni tanlang (o‘zbek, rus, ingliz, avtomatik…).
3. **Papkalar** → to‘liq yo‘lni yozing (`/Users/…/Docs` yoki `D:\Hujjatlar`) → **Qo‘shish**.
4. Fayl tushishi bilan karta pastda paydo bo‘ladi:
   - **Chap:** fayl nomi, format, ajratilgan asl matn
   - **O‘ng:** xulosa, o‘zbekcha tarjima, (audio bo‘lsa) transkripsiya
5. Ro‘yxat pastga qarab o‘sadi; pastga scroll — keyingi sahifa (infinite scroll).
6. **Skanerlash** — mavjud fayllarni qayta aylanadi.
7. **Qayta** — bitta kartani qayta navbatga qo‘yadi.

Birlamchi papka: `watched_folders/`.

---

## 6. Xotira va OOM

- Matn **~5000 belgi**lik bo‘laklarda xulosalanadi / tarjima qilinadi.
- Audio **25 soniya**lik bo‘laklarda yoziladi.
- Bir vaqtda **bitta fayl** ishlanadi.
- Audio tugagach ASR xotiradan tushiriladi, keyin LLM yuklanadi (8 GB VRAM uchun muhim).
- Model almashtirilsa, avvalgisi `unload` qilinadi.
- CUDA VRAM ≤ 10 GB: katta model uchun avtomatik **4-bit**.

Agar baribir OOM bo‘lsa:

1. UI da Gemma 4 ni tanlang.
2. GGUF Q3/Q4 ishlating (Transformers emas).
3. Boshqa brauzer tablaridagi GPU ishini yoping.
4. `data/documents.db` ni o‘chirmang — faqat jarayonni qayta ishga tushiring.

---

## 7. Qo‘llab-quvvatlanadigan fayllar

| Tur | Kengaytma |
|---|---|
| Hujjat | `.pdf` `.docx` `.txt` `.md` `.rtf` |
| Rasm | `.png` `.jpg` `.jpeg` `.webp` `.tif` `.bmp` |
| Audio | `.wav` `.mp3` `.ogg` `.flac` `.m4a` `.aac` |
| Video | `.mp4` `.mkv` `.mov` `.webm` (ffmpeg kerak) |

Skaner-PDF: sahifa rasmi + Tesseract / EasyOCR.

---

## 8. API (lokal)

| Usul | Yo‘l | Vazifa |
|---|---|---|
| GET | `/` | SPA |
| GET | `/api/status` | qurilma, navbat, modellar |
| GET | `/api/models/health` | Gemma / GigaAM / NLLB jonli holat |
| GET/PUT | `/api/settings` | `llm_model`, `asr_language`, `quantization` |
| GET/POST | `/api/folders` | papkalar |
| DELETE | `/api/folders/{id}` | papkani kuzatuvdan olish |
| POST | `/api/scan` | qayta skaner |
| GET | `/api/documents?offset=&limit=` | sahifa |
| POST | `/api/documents/{id}/reprocess` | qayta ishlash |
| WS | `/ws` | real-time kartalar |

`quantization` qiymatlari: `auto` | `none` | `8bit` | `4bit`.

---

## 9. Muammolarni bartaraf etish

**UI ochiladi, lekin kartada “modeli topilmadi”**  
`models/` ichida GGUF yoki `config.json` yo‘q. Nomlar `config.py` dagi glob lar bilan mos kelishini tekshiring.

**MPS ishlamayapti**  
`torch.backends.mps.is_available()` → `False`. Rasmiy macOS torch o‘rnating, Rosetta terminalidan qoching (arm64).

**CUDA ko‘rinmayapti**  
Noto‘g‘ri torch (CPU wheel) o‘rnatilgan. `pip show torch` → `cu124` bo‘lishi kerak. `nvidia-smi` ishlasin.

**Tesseract xatosi**  
`tesseract --list-langs`. `uzb` yo‘q bo‘lsa, `eng` bilan uriniladi. `TESSDATA_PREFIX` ni `models/tessdata` ga qo‘ying.

**EasyOCR “download” xatosi**  
Offline rejimda model yo‘q. `models/easyocr/` ni internetli mashinadan to‘ldiring.

**ffmpeg yo‘q**  
`.mp3` / video o‘qilmaydi. WAV bering yoki ffmpeg o‘rnating.

**Port band**  
`--port 8001`.

---

## 10. Xavfsizlik

Ilova lokal tarmoqda ishlaydi, autentifikatsiya yo‘q. `--host 0.0.0.0` ni faqat ishonchli LAN da ishlating. Hech qanday kalit yoki cloud token talab qilinmaydi.
