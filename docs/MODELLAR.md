# Modellarni ishga tushirish va holatini tekshirish

Bu hujjat **Gemma**, **GigaAM** va **NLLB** ni qanday yoqish, to‘xtatish va ish holatini qanday ko‘rishni tushuntiradi.

UI yuqori panelida uchta nuqta bor:

| Rang | Holat | Ma’nosi |
|---|---|---|
| Yashil | **Ishlayapti** | Worker ochiq **va** model xotirada |
| Sariq | **Yuklanmoqda** | Worker ochiq, model hali yuklanmagan |
| Qizil | **O‘chiq** | Portga ulanilmadi — `start_workers.sh` kerak |
| Kulrang | **Og‘irlik yo‘q** | Diskda model to‘liq emas |

Nuqmani bosing — batafsil panel ochiladi. Holat har ~4 soniyada yangilanadi.

---

## 1. Servislar

| Servis | Port | Model | Vazifa |
|---|---|---|---|
| UI + pipeline | `8000` | — | OCR, navbat, RAG, interfeys |
| Gemma | `8001` | Gemma 4e / Gemma 26 | Xulosa, chat |
| GigaAM | `8002` | GigaAM-Multilingual | Audio / video transkripsiya |
| NLLB | `8003` | NLLB-200 1.3B | O‘zbek lotin tarjima |

Modellar **alohida jarayonda** yashaydi. UI qayta ochilsa ham og‘irliklar qayta yuklanmaydi.

---

## 2. Tezkor start

Loyiha ildizi: `local_doc_platform/`

```bash
cd local_doc_platform
./start.sh
```

Bu avval workerlarni, keyin UI ni (`8000`) ochadi.

Brauzer: [http://127.0.0.1:8000](http://127.0.0.1:8000)

Faqat modellarni yoqish (UI allaqachon ochiq bo‘lsa):

```bash
./scripts/start_workers.sh
```

Port band bo‘lsa, skript o‘sha workerni qayta ochmaydi — mavjud jarayon qoladi.

---

## 3. Qo‘lda (alohida terminal)

```bash
cd local_doc_platform
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

.venv/bin/python -m uvicorn backend.workers.gemma_app:app --host 127.0.0.1 --port 8001
.venv/bin/python -m uvicorn backend.workers.gigaam_app:app --host 127.0.0.1 --port 8002
.venv/bin/python -m uvicorn backend.workers.nllb_app:app --host 127.0.0.1 --port 8003
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Workerlarsiz (hammasi bitta jarayonda, sekinroq, qayta yuklanishi mumkin):

```bash
DOC_USE_MODEL_SERVERS=0 .venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

---

## 4. Holatni tekshirish

### UI

Yuqori o‘ngdagi **Gemma / GigaAM / NLLB** nuqtalari.

### Terminal

```bash
./scripts/check_models.sh
```

Yoki qo‘lda:

```bash
curl -sS http://127.0.0.1:8001/health
curl -sS http://127.0.0.1:8002/health
curl -sS http://127.0.0.1:8003/health

# Model xotiradami?  "ready": true
curl -sS http://127.0.0.1:8001/status
curl -sS http://127.0.0.1:8002/status
curl -sS http://127.0.0.1:8003/status

# UI orqali yig‘ma holat
curl -sS http://127.0.0.1:8000/api/models/health
```

`/health` — jarayon tinglayaptimi.  
`/status` → `ready: true` — og‘irlik xotirada, so‘rov qabul qilishga tayyor.

---

## 5. Log va PID

| Worker | Log | PID |
|---|---|---|
| Gemma | `data/gemma_worker.log` | `data/gemma_worker.pid` |
| GigaAM | `data/gigaam_worker.log` | `data/gigaam_worker.pid` |
| NLLB | `data/nllb_worker.log` | `data/nllb_worker.pid` |
| UI | `data/ui.log` | `data/ui.pid` |

```bash
tail -f data/gemma_worker.log
```

---

## 6. To‘xtatish

```bash
for f in data/gemma_worker.pid data/gigaam_worker.pid data/nllb_worker.pid data/ui.pid; do
  [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null || true
done
```

Yoki port bo‘yicha:

```bash
lsof -nP -iTCP:8001 -sTCP:LISTEN
```

---

## 7. Og‘irliklar (birinchi marta)

Ish vaqtida internetga chiqilmaydi (`HF_HUB_OFFLINE=1`). Modellarni oldindan yuklang:

```bash
.venv/bin/python -u scripts/download_models.py          # hammasi
.venv/bin/python -u scripts/download_models.py nllb     # faqat tarjima
.venv/bin/python -u scripts/download_models.py gigaam
```

Kutilgan papkalar:

```text
models/LLM modellar/Gemma 4e/
models/LLM modellar/Gemma 26/
models/ASR modellar/GigaAM-Multilingual/
models/tarjima_model/
```

NLLB `pytorch_model.bin` ~5.11 GB bo‘lishi kerak. Kichikroq bo‘lsa — yuklash tugamagan.

---

## 8. Tipik muammolar

**UI ochiq, nuqtalar qizil**  
Worker o‘lgan (masalan, terminal yopilganda). `./scripts/start_workers.sh`

**Sariq — Yuklanmoqda**  
Server ochiq, birinchi so‘rov modelni xotiraga oladi. Gemma 26 bir necha daqiqa olishi mumkin. Logni qarang.

**Kulrang — Og‘irlik yo‘q**  
`download_models.py` ni qayta ishga tushiring. Yarim shard (1 ta fayl) yetarli emas.

**NLLB o‘chiq / og‘irlik yo‘q**  
`models/tarjima_model/pytorch_model.bin` hajmini tekshiring.

**Port band**  
`lsof -nP -iTCP:8001 -sTCP:LISTEN` — eski jarayonni to‘xtating yoki qoldiring.

**Cursor / IDE terminalida worker o‘lib qoladi**  
`start_workers.sh` `start_new_session=True` bilan ochadi. Shu skriptni ishlating, `uvicorn` ni oddiy foreground da qoldirmang.
