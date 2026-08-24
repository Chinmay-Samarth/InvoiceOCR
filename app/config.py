"""OCR runtime settings (override via environment variables)."""
import os

# Render Starter is ~512 MB. PaddleOCR v4 at 2400px will OOM.
# Default to low-memory mode; set OCR_LOW_MEMORY=false on a 2 GB+ box.
OCR_LOW_MEMORY = os.getenv("OCR_LOW_MEMORY", "true").lower() in (
    "1", "true", "yes",
)

# PDF render DPI (100 is enough for printed GST invoices in low-RAM mode).
OCR_DPI = int(os.getenv("OCR_DPI", "100" if OCR_LOW_MEMORY else "150"))

# Longest image side sent to PaddleOCR. Smaller = less RAM.
OCR_MAX_SIDE = int(os.getenv("OCR_MAX_SIDE", "1280" if OCR_LOW_MEMORY else "2400"))

# Detector internal max side (PaddleOCR det_limit_side_len).
OCR_DET_LIMIT = int(os.getenv("OCR_DET_LIMIT", "736" if OCR_LOW_MEMORY else "960"))

# Angle classifier uses extra RAM; skip on upright scans.
OCR_USE_ANGLE_CLS = os.getenv("OCR_USE_ANGLE_CLS", "false").lower() in (
    "1", "true", "yes",
)

_max_pages = os.getenv("OCR_MAX_PAGES", "1")
OCR_MAX_PAGES = None if _max_pages in ("0", "") else int(_max_pages)

# One thread uses less RAM than spawning OpenMP workers for every core.
OCR_CPU_THREADS = int(os.getenv(
    "OCR_CPU_THREADS",
    "1" if OCR_LOW_MEMORY else str(os.cpu_count() or 2),
))

# Recognition batch; 1 keeps peak RAM lowest.
OCR_REC_BATCH_NUM = int(os.getenv("OCR_REC_BATCH_NUM", "1" if OCR_LOW_MEMORY else "8"))

# MKL-DNN is faster but allocates extra buffers — off on small instances.
OCR_ENABLE_MKLDNN = os.getenv(
    "OCR_ENABLE_MKLDNN",
    "false" if OCR_LOW_MEMORY else "true",
).lower() in ("1", "true", "yes")

# PP-OCR (mobile) is much smaller than PP-OCRv4. Accuracy is lower but it
# can fit on Render Starter. Set OCR_VERSION=PP-OCRv4 if you have 2 GB RAM.
OCR_VERSION = os.getenv(
    "OCR_VERSION",
    "PP-OCR" if OCR_LOW_MEMORY else "PP-OCRv4",
)

OCR_FAST_MODE = os.getenv("OCR_FAST_MODE", "false").lower() in ("1", "true", "yes")
if OCR_FAST_MODE:
    OCR_VERSION = "PP-OCR"
    OCR_MAX_SIDE = min(OCR_MAX_SIDE, 1280)
    OCR_USE_ANGLE_CLS = False
