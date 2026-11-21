"""Invoice OCR REST API.

Run:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

    GET  /health
    POST /parse   multipart field name: document
"""
import asyncio
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile

from .config import (
    OCR_CPU_THREADS,
    OCR_DPI,
    OCR_ENABLE_MKLDNN,
    OCR_MAX_SIDE,
    OCR_VERSION,
)
from .extractors import extract_all
from .models import InvoiceResult
from .ocr_engine import SUPPORTED_IMAGE_EXT, get_ocr, ocr_file
from .thermal_extractors import (
    DOC_PAPER,
    classify_document,
    extract_thermal,
)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024

app = FastAPI(
    title="Invoice OCR API",
    description="PaddleOCR-based invoice field extraction for Indian GST invoices",
    version="1.2.0",
)

_executor = ThreadPoolExecutor(max_workers=1)


def _validate_upload(filename: Optional[str], contents: bytes) -> str:
    ext = os.path.splitext(filename or "")[1].lower()
    if ext != ".pdf" and ext not in SUPPORTED_IMAGE_EXT:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Use PDF or image files.",
        )
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 25 MB)")
    return ext


def _parse_bytes(contents: bytes, ext: str) -> InvoiceResult:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        pages = ocr_file(tmp_path)
        doc_type = classify_document(pages)
        if doc_type == DOC_PAPER:
            result = extract_all(pages)
            result.document_type = DOC_PAPER
        else:
            result = extract_thermal(pages, doc_type)
        return result
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.on_event("startup")
def _warm_up() -> None:
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
        "ocr_version": OCR_VERSION,
        "enable_mkldnn": OCR_ENABLE_MKLDNN,
    }


@app.post("/parse", response_model=InvoiceResult)
async def parse_invoice(document: UploadFile = File(...)) -> InvoiceResult:
    contents = await document.read()
    ext = _validate_upload(document.filename, contents)
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_executor, _parse_bytes, contents, ext)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to parse: {exc}")
