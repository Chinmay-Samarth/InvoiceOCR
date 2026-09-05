# Invoice OCR API (PaddleOCR)

## Run locally (recommended)

```bash
cd "E:\Chinmay_Projects\VS Files\Python Files\InvoiceOCR"
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

ERP `.env`:

```
INVOICE_OCR_URL=http://127.0.0.1:8000/parse
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness |
| POST | `/parse` | Parse invoice (multipart field: `document`) |

Accuracy defaults: PP-OCRv4, 150 DPI, 2400 px longest side.
