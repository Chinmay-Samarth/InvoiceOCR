# Invoice OCR API (PaddleOCR)

Self-hosted GST invoice OCR. Upload a PDF/image, get structured JSON
(supplier, bank, invoice, totals, line items, tax lines).

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness |
| POST | `/parse` | Sync parse |
| POST | `/parse/async` | Start job → `{ "job_id" }` |
| GET | `/jobs/{job_id}` | Poll until `done` / `error` |

Multipart field name: **`document`**

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

## Render (low memory)

PaddleOCR needs RAM. On **Starter (~512 MB)** use the low-memory defaults
already in this repo (`OCR_LOW_MEMORY=true`): smaller images, 1 CPU thread,
mobile PP-OCR models.

Dashboard env vars:

```
PYTHON_VERSION=3.10.14
OCR_LOW_MEMORY=true
OCR_DPI=100
OCR_MAX_SIDE=1280
OCR_MAX_PAGES=1
OCR_CPU_THREADS=1
```

Start command:

```
uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1
```

If it still crashes (OOM), upgrade to **Standard (2 GB)**. That is the
realistic minimum for accurate PP-OCRv4. Then set:

```
OCR_LOW_MEMORY=false
OCR_VERSION=PP-OCRv4
OCR_MAX_SIDE=2400
```

## ERP env

```env
INVOICE_OCR_URL=https://YOUR-SERVICE.onrender.com/parse
INVOICE_OCR_USE_ASYNC=true
INVOICE_OCR_TIMEOUT_MS=600000
```
