"""OCR runtime settings (override via environment variables)."""
import os

# Full accuracy on a local PC. Do not enable low-memory unless RAM is tiny.
OCR_LOW_MEMORY = os.getenv("OCR_LOW_MEMORY", "false").lower() in (
    "1", "true", "yes",
)

OCR_DPI = int(os.getenv("OCR_DPI", "100" if OCR_LOW_MEMORY else "150"))
OCR_MAX_SIDE = int(os.getenv("OCR_MAX_SIDE", "1280" if OCR_LOW_MEMORY else "2400"))
OCR_DET_LIMIT = int(os.getenv("OCR_DET_LIMIT", "736" if OCR_LOW_MEMORY else "960"))

OCR_USE_ANGLE_CLS = os.getenv("OCR_USE_ANGLE_CLS", "false").lower() in (
    "1", "true", "yes",
)

_max_pages = os.getenv("OCR_MAX_PAGES", "1")
OCR_MAX_PAGES = None if _max_pages in ("0", "") else int(_max_pages)

OCR_CPU_THREADS = int(os.getenv(
    "OCR_CPU_THREADS",
    "1" if OCR_LOW_MEMORY else str(os.cpu_count() or 4),
))
OCR_REC_BATCH_NUM = int(os.getenv("OCR_REC_BATCH_NUM", "1" if OCR_LOW_MEMORY else "8"))

OCR_ENABLE_MKLDNN = os.getenv(
    "OCR_ENABLE_MKLDNN",
    "false" if OCR_LOW_MEMORY else "true",
).lower() in ("1", "true", "yes")

OCR_VERSION = os.getenv(
    "OCR_VERSION",
    "PP-OCR" if OCR_LOW_MEMORY else "PP-OCRv4",
)

OCR_FAST_MODE = os.getenv("OCR_FAST_MODE", "false").lower() in ("1", "true", "yes")
if OCR_FAST_MODE:
    OCR_VERSION = "PP-OCR"
    OCR_MAX_SIDE = min(OCR_MAX_SIDE, 1280)
    OCR_USE_ANGLE_CLS = False
