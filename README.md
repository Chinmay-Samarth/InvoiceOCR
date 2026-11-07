# Invoice OCR API (PaddleOCR)

## Run locally on Windows (recommended)

Double-click `start_ocr_windows.bat`, or from a terminal:

```bat
cd /d "E:\Chinmay_Projects\VS Files\Python Files\InvoiceOCR"
start_ocr_windows.bat
```

That starts the stable low-CPU profile on **port 8080**:
PP-OCRv4, 150 DPI, 1600 max side, 1 thread, MKLDNN off
(same field accuracy as 2400px in the bench, without the RAM spike that crashes a small server).

ERP `server/.env`:

```
INVOICE_OCR_URL=http://127.0.0.1:8080/parse
INVOICE_OCR_HEALTH_URL=http://127.0.0.1:8080/health
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness |
| POST | `/parse` | Parse invoice (multipart field: `document`) |

Accuracy defaults: PP-OCRv4, 150 DPI, 1600 px longest side, 1 CPU thread, `OCR_ENABLE_MKLDNN=false`.
