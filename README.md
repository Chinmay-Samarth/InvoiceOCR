# Invoice OCR API (Mindee Replacement)

Self-hosted, free invoice OCR for Indian GST tax invoices, built on
[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR). Accepts scanned PDF or
image invoices over HTTP and returns structured JSON: supplier details, bank
details, invoice number/date/IRN, totals, line items and GST tax lines.

## How it works

1. PDF pages are rendered to images with PyMuPDF (images are used directly).
2. PaddleOCR (detection + recognition + angle classification) produces text
   fragments with pixel coordinates.
3. Fragments are grouped into visual lines, then layout-agnostic extractors
   pull out fields using regex patterns (GSTIN, PAN, IFSC, IRN, dates),
   keyword anchors (Taxable Value, Grand Total, CGST/SGST/IGST @ x%) and
   table-column geometry for line items.
4. Totals are cross-checked (`base + tax = total`) and inconsistencies are
   reported in a `warnings` array so your ERP can flag invoices for review.

No cloud calls, no per-page fees. Works on CPU.

## Setup

```bash
pip install -r requirements.txt
```

The first run downloads the PaddleOCR English models (~17 MB) to
`~/.paddleocr` automatically.

## Run the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check |
| `/parse` | POST | Multipart upload, field name `document` (PDF/PNG/JPG, max 25 MB) |
| `/docs` | GET | Interactive Swagger UI |

Parsing a scanned invoice takes roughly **60–90 seconds per page on CPU**
(after models are loaded). The first server start loads models once (~30–60 s);
keep the server running so later requests skip that step.

### Speed tips

| Setting | Default | Effect |
|---------|---------|--------|
| Keep server running | — | Avoids reloading models on every CLI run |
| `OCR_MAX_SIDE=2400` | 2400 | Downscales large scans before OCR (~25% faster) |
| `OCR_DPI=150` | 150 | Lower PDF render DPI (150 is enough for print) |
| `OCR_USE_ANGLE_CLS=false` | false | Skips rotation classifier (~5–10 s saved) |
| `OCR_CPU_THREADS` | all cores | Uses all CPU threads for inference |
| `OCR_MAX_PAGES=1` | 1 | Only OCRs page 1 (typical for GST invoices) |

Example — start the API with tuned settings:

```bash
set OCR_MAX_SIDE=2400
set OCR_DPI=150
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Why was it taking 5+ minutes?** Running `parse_invoice.py` cold on every
request reloads PaddleOCR models each time (~1–2 min) plus full-resolution
OCR on ~5000×7000 px images (~1.5–2 min). With the server kept warm and the
new defaults, expect about **1 minute per invoice** on CPU.

**Need it faster?**

- **GPU**: install CUDA-enabled `paddlepaddle-gpu`, set `use_gpu=True` in
  `app/ocr_engine.py` — typically 5–15 s per page.
- **Fast mode** (lower accuracy): `set OCR_FAST_MODE=true` uses lightweight
  mobile models (~7 s/page but misses fields on complex invoices).

For concurrent load, run multiple workers: `uvicorn app.main:app --workers 2`
(each worker loads its own model copy, ~1 GB RAM each).

## Deploy on Render

Render defaults to **Python 3.14**, which cannot install `paddlepaddle`. This
repo pins **Python 3.10.14** via `.python-version` — you must **commit and
push** that file to GitHub before redeploying.

**Dashboard settings:**

| Setting | Value |
|---------|--------|
| Root Directory | *(leave empty — repo root)* |
| Build Command | `pip install --upgrade pip && pip install -r requirements.txt` |
| Start Command | `uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1` |

**Environment variable (add in Render → Environment):**

| Key | Value |
|-----|--------|
| `PYTHON_VERSION` | `3.10.14` |

Then: **Manual Deploy → Clear build cache & deploy**.

The build log should show `Using Python version 3.10.14` (not 3.14.3).

**Plan:** use at least **Starter** (512 MB may OOM). Prefer **Standard** (2 GB)
for PaddleOCR. First boot downloads models (~17 MB) over HTTPS.

## ERP integration example (Python)

```python
import requests

def parse_invoice(pdf_path: str) -> dict:
    with open(pdf_path, "rb") as f:
        resp = requests.post(
            "http://localhost:8000/parse",
            files={"document": (pdf_path, f, "application/pdf")},
            timeout=300,
        )
    resp.raise_for_status()
    data = resp.json()

    # route low-confidence or inconsistent extractions to manual review
    needs_review = data["warnings"] or (data["ocr_confidence"] or 0) < 0.90
    return data, needs_review

data, needs_review = parse_invoice("invoice.pdf")
print(data["supplier"]["gstin"], data["totals"]["total_amount"])
```

### Response shape

```json
{
  "supplier": {
    "name": "", "current_address": "", "billing_address": "",
    "billing_state": "", "gstin": "", "pan": "",
    "contact_person": "", "phone": "", "email": ""
  },
  "bank": {"account_number": "", "account_type": "", "ifsc": ""},
  "invoice": {"number": "", "date": "", "due_date": "", "irn": ""},
  "totals": {"base_amount": 0, "tax_amount": 0, "total_amount": 0},
  "line_items": [{"description": "", "quantity": 0, "unit_price": 0, "total": 0}],
  "tax_lines": [{"rate": 18, "base": 10000, "amount": 1800}],
  "ocr_confidence": 0.95,
  "warnings": []
}
```

Missing fields come back empty (`""` / `0`) rather than failing, so new
invoice layouts always return a valid response. `ocr_confidence` is the mean
character-recognition confidence (0-1); `warnings` lists reconciliation
problems (for example when `base + tax` does not match `total`).

Notes on specific fields:

- `supplier.pan` is derived from characters 3-12 of the supplier GSTIN when
  present, which is more reliable than OCR-ing a separately printed PAN.
- `tax_lines` contains one entry per GST component (CGST and SGST appear as
  two lines at the same rate). When an amount is not printed, it is inferred
  from `rate x base` and validated against the invoice total.
- Dates are returned exactly as printed (e.g. `25.05.2026`, `28-May-26`);
  normalize them in your ERP if you need ISO format.

## CLI usage (no server)

```bash
python parse_invoice.py "c:\path\to\invoice.pdf"
```

## Project layout

| File | Purpose |
|------|---------|
| `app/ocr_engine.py` | PDF/image rendering, PaddleOCR wrapper, line grouping |
| `app/extractors.py` | All field extraction and reconciliation logic |
| `app/models.py` | Pydantic response schema |
| `app/main.py` | FastAPI server |
| `parse_invoice.py` | Command-line parser |
| `test_samples.py` | Dev harness with OCR result caching for fast iteration |
| `dump_ocr.py` | Dev utility: dump raw OCR lines with coordinates |

## Tuning for new invoice formats

If a new supplier's invoices extract poorly:

1. Run `python dump_ocr.py "path\to\invoice.pdf"` and inspect the generated
   `ocr_*.txt` file to see exactly what PaddleOCR read and where.
2. Extend the anchor lists in `app/extractors.py` (`BASE_ANCHORS`,
   `TOTAL_ANCHORS`, `HEADER_TOKENS`, `BUYER_ANCHOR_RE`) with the new
   layout's keywords.
3. Re-test instantly with `python test_samples.py "path\to\invoice.pdf"`
   (OCR results are cached in `.ocr_cache/`).
