"""OCR runtime settings (override via environment variables).

Default is the low-CPU profile that kept full PP-OCRv4 accuracy in the bench
(same field score as 2400px) while staying inside a small server:
1 thread, recognition batch 1, MKLDNN off, longest side 1600.
MKLDNN and many threads are what hard-crash the process (ECONNRESET / OOM).
"""
import os

# Smaller image only. Stay on PP-OCRv4 — the mobile model drops table digits.
OCR_LOW_MEMORY = os.getenv("OCR_LOW_MEMORY", "false").lower() in (
    "1", "true", "yes",
)

OCR_DPI = int(os.getenv("OCR_DPI", "120" if OCR_LOW_MEMORY else "150"))
OCR_MAX_SIDE = int(os.getenv("OCR_MAX_SIDE", "1280" if OCR_LOW_MEMORY else "1600"))
OCR_DET_LIMIT = int(os.getenv("OCR_DET_LIMIT", "736" if OCR_LOW_MEMORY else "960"))

OCR_USE_ANGLE_CLS = os.getenv("OCR_USE_ANGLE_CLS", "false").lower() in (
    "1", "true", "yes",
)

# Default 5 so multi-page invoices (line items / tax on page 2+) are read.
# Set OCR_MAX_PAGES=0 for unlimited; 1 to force first page only.
_max_pages = os.getenv("OCR_MAX_PAGES", "5")
OCR_MAX_PAGES = None if _max_pages in ("0", "") else int(_max_pages)

# One thread is the stable profile on a low-CPU box. More threads raise peak
# RAM and are what knocked the server over. Override with OCR_CPU_THREADS.
OCR_CPU_THREADS = max(1, int(os.getenv("OCR_CPU_THREADS", "1")))
OCR_REC_BATCH_NUM = max(1, int(os.getenv("OCR_REC_BATCH_NUM", "1")))

# MKLDNN hard-crashes the process mid-parse. Leave it off.
OCR_ENABLE_MKLDNN = os.getenv("OCR_ENABLE_MKLDNN", "false").lower() in (
    "1", "true", "yes",
)

OCR_VERSION = os.getenv("OCR_VERSION", "PP-OCRv4")

OCR_FAST_MODE = os.getenv("OCR_FAST_MODE", "false").lower() in ("1", "true", "yes")
if OCR_FAST_MODE:
    OCR_VERSION = "PP-OCR"
    OCR_MAX_SIDE = min(OCR_MAX_SIDE, 1280)
    OCR_USE_ANGLE_CLS = False


def _pin_cpu_threads() -> None:
    """Must run before Paddle is imported, or it grabs every core."""
    n = str(OCR_CPU_THREADS)
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = n
    os.environ["FLAGS_use_mkldnn"] = "1" if OCR_ENABLE_MKLDNN else "0"
    # Release tensors immediately so a small server does not accumulate RAM.
    os.environ.setdefault("FLAGS_eager_delete_tensor_gb", "0.0")
    os.environ.setdefault("FLAGS_allocator_strategy", "naive_best_fit")


_pin_cpu_threads()
