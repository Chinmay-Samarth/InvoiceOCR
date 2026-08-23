"""OCR runtime settings (override via environment variables)."""
import os

# PDF render DPI before OCR (150 is enough for printed invoices; 220 is slower).
OCR_DPI = int(os.getenv("OCR_DPI", "150"))

# Longest image side sent to PaddleOCR. Invoices are large scans; capping at
# 2400 px cuts CPU time ~25-30% with little accuracy loss on our samples.
OCR_MAX_SIDE = int(os.getenv("OCR_MAX_SIDE", "2400"))

# Angle classifier adds ~5-10 s/page and is unnecessary for upright scans.
OCR_USE_ANGLE_CLS = os.getenv("OCR_USE_ANGLE_CLS", "false").lower() in (
    "1", "true", "yes",
)

# Most GST invoices are one page; set OCR_MAX_PAGES=0 to OCR every page.
_max_pages = os.getenv("OCR_MAX_PAGES", "1")
OCR_MAX_PAGES = None if _max_pages in ("0", "") else int(_max_pages)

# Paddle inference threads (defaults to all CPU cores).
OCR_CPU_THREADS = int(os.getenv("OCR_CPU_THREADS", str(os.cpu_count() or 4)))

# Recognition mini-batch size; higher uses more RAM but is faster on CPU.
OCR_REC_BATCH_NUM = int(os.getenv("OCR_REC_BATCH_NUM", "16"))

# PP-OCRv4 = accurate (default). PP-OCR = old mobile models (~10x faster but
# much worse on invoice text — not recommended unless OCR_FAST_MODE=true).
OCR_VERSION = os.getenv("OCR_VERSION", "PP-OCRv4")

# When true, use PP-OCR mobile models (fast but lower accuracy).
OCR_FAST_MODE = os.getenv("OCR_FAST_MODE", "false").lower() in ("1", "true", "yes")

if OCR_FAST_MODE:
    OCR_VERSION = "PP-OCR"
    OCR_MAX_SIDE = min(OCR_MAX_SIDE, 2000)
    OCR_USE_ANGLE_CLS = False
