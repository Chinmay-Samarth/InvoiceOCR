"""Invoice OCR REST API - a self-hosted replacement for Mindee.

Run:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health              liveness + model readiness
    POST /parse               sync parse (may 502 on Render if >~100s)
    POST /parse/async         start OCR job, returns {job_id} immediately
    GET  /jobs/{job_id}       poll job status / result
"""
import asyncio
import os
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from .config import OCR_CPU_THREADS, OCR_DPI, OCR_LOW_MEMORY, OCR_MAX_SIDE, OCR_VERSION
from .extractors import extract_all
from .models import InvoiceResult
from .ocr_engine import SUPPORTED_IMAGE_EXT, get_ocr, ocr_file

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
JOB_TTL_SECONDS = 60 * 60  # keep finished jobs for 1 hour

app = FastAPI(
    title="Invoice OCR API",
    description="PaddleOCR-based invoice field extraction for Indian GST invoices",
    version="1.1.0",
)

# OCR runs in a thread pool so the async event loop is not blocked.
_executor = ThreadPoolExecutor(max_workers=1)
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


class JobAccepted(BaseModel):
    job_id: str
    status: str = "queued"
    poll_url: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # queued | running | done | error
    result: Optional[InvoiceResult] = None
    error: Optional[str] = None
    created_at: float
    updated_at: float


def _set_job(job_id: str, **fields: Any) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id) or {"job_id": job_id}
        job.update(fields)
        job["job_id"] = job_id
        job["updated_at"] = time.time()
        _jobs[job_id] = job


def _get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _jobs_lock:
        return dict(_jobs[job_id]) if job_id in _jobs else None


def _purge_old_jobs() -> None:
    now = time.time()
    with _jobs_lock:
        stale = [
            jid for jid, job in _jobs.items()
            if now - job.get("updated_at", now) > JOB_TTL_SECONDS
            and job.get("status") in ("done", "error")
        ]
        for jid in stale:
            _jobs.pop(jid, None)


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
        return extract_all(pages)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _run_job(job_id: str, contents: bytes, ext: str) -> None:
    _set_job(job_id, status="running")
    try:
        result = _parse_bytes(contents, ext)
        _set_job(job_id, status="done", result=result.dict(), error=None)
    except Exception as exc:
        _set_job(job_id, status="error", result=None, error=str(exc))


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
        "ocr_version": OCR_VERSION,
        "low_memory": OCR_LOW_MEMORY,
        "async_jobs": True,
    }


@app.post("/parse", response_model=InvoiceResult)
async def parse_invoice(document: UploadFile = File(...)) -> InvoiceResult:
    """Synchronous parse. Prefer /parse/async on Render (100s HTTP timeout)."""
    contents = await document.read()
    ext = _validate_upload(document.filename, contents)
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_executor, _parse_bytes, contents, ext)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to parse: {exc}")


@app.post("/parse/async", response_model=JobAccepted, status_code=202)
async def parse_invoice_async(document: UploadFile = File(...)) -> JobAccepted:
    """Start OCR in the background and return a job id immediately (avoids 502)."""
    _purge_old_jobs()
    contents = await document.read()
    ext = _validate_upload(document.filename, contents)

    job_id = uuid.uuid4().hex
    now = time.time()
    _set_job(
        job_id,
        status="queued",
        result=None,
        error=None,
        created_at=now,
        updated_at=now,
    )
    _executor.submit(_run_job, job_id, contents, ext)
    return JobAccepted(job_id=job_id, status="queued", poll_url=f"/jobs/{job_id}")


@app.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown or expired job_id")
    result = None
    if job.get("status") == "done" and job.get("result") is not None:
        result = InvoiceResult(**job["result"])
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        result=result,
        error=job.get("error"),
        created_at=job.get("created_at", 0),
        updated_at=job.get("updated_at", 0),
    )
