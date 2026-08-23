"""Invoice OCR REST API - a self-hosted replacement for Mindee.

Run:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health          liveness + model readiness
    POST /parse           multipart upload (field name: "document"),
                          accepts PDF / PNG / JPG, returns extracted JSON
"""
import asyncio
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, HTTPException, UploadFile

from .config import OCR_CPU_THREADS, OCR_DPI, OCR_MAX_SIDE
from .extractors import extract_all
from .models import InvoiceResult
from .ocr_engine import SUPPORTED_IMAGE_EXT, get_ocr, ocr_file

MAX_UPLOAD_BYTES = 25 * 1024 * 1024

app = FastAPI(
    title="Invoice OCR API",
    description="PaddleOCR-based invoice field extraction for Indian GST invoices",
    version="1.0.0",
)

# OCR runs in a thread pool so the async event loop is not blocked.
_executor = ThreadPoolExecutor(max_workers=1)


@app.on_event("startup")
def _warm_up() -> None:
    # Load PaddleOCR models once at startup (first request otherwise adds ~60 s).
    get_ocr()


@app.on_event("shutdown")
def _shutdown() -> None:
    _executor.shutdown(wait=False)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "ocr_dpi": OCR_DPI,
        "ocr_max_side": OCR_MAX_SIDE,
        "cpu_threads": OCR_CPU_THREADS,
    }


@app.post("/parse", response_model=InvoiceResult)
async def parse_invoice(document: UploadFile = File(...)) -> InvoiceResult:
    ext = os.path.splitext(document.filename or "")[1].lower()
    if ext != ".pdf" and ext not in SUPPORTED_IMAGE_EXT:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Use PDF or image files.",
        )

    contents = await document.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 25 MB)")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        loop = asyncio.get_running_loop()
        pages = await loop.run_in_executor(_executor, ocr_file, tmp_path)
        return extract_all(pages)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to parse: {exc}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
