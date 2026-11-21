"""OCR engine: converts a PDF or image file into positioned text lines.

Pipeline: PDF pages are rendered with PyMuPDF, optionally downscaled, then run
through PaddleOCR.  Word boxes are grouped into visual lines for extractors.
"""
import os
import threading
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from .config import (
    OCR_CPU_THREADS,
    OCR_DET_LIMIT,
    OCR_DPI,
    OCR_ENABLE_MKLDNN,
    OCR_MAX_PAGES,
    OCR_MAX_SIDE,
    OCR_REC_BATCH_NUM,
    OCR_USE_ANGLE_CLS,
    OCR_VERSION,
)

SUPPORTED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Word:
    """A single OCR-recognized text fragment with its bounding box (pixels)."""
    text: str
    conf: float
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass
class Line:
    """A visual line: words sharing roughly the same vertical position."""
    words: List[Word] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def y0(self) -> float:
        return min(w.y0 for w in self.words)

    @property
    def y1(self) -> float:
        return max(w.y1 for w in self.words)

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def x0(self) -> float:
        return min(w.x0 for w in self.words)


@dataclass
class Page:
    number: int
    width: float
    height: float
    words: List[Word] = field(default_factory=list)
    lines: List[Line] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(ln.text for ln in self.lines)

    def mean_confidence(self) -> float:
        if not self.words:
            return 0.0
        return float(np.mean([w.conf for w in self.words]))


_ocr_instance = None
_ocr_lock = threading.Lock()


def get_ocr():
    """Lazily create a single shared PaddleOCR instance (model load is slow)."""
    global _ocr_instance
    if _ocr_instance is None:
        with _ocr_lock:
            if _ocr_instance is None:
                from .config import _pin_cpu_threads
                _pin_cpu_threads()
                try:
                    cv2.setNumThreads(OCR_CPU_THREADS)
                except Exception:
                    pass
                from paddleocr import PaddleOCR
                _ocr_instance = PaddleOCR(
                    use_angle_cls=OCR_USE_ANGLE_CLS,
                    lang="en",
                    show_log=False,
                    use_gpu=False,
                    ocr_version=OCR_VERSION,
                    cpu_threads=OCR_CPU_THREADS,
                    rec_batch_num=OCR_REC_BATCH_NUM,
                    enable_mkldnn=OCR_ENABLE_MKLDNN,
                    det_limit_side_len=OCR_DET_LIMIT,
                    use_mp=False,
                )
    return _ocr_instance


def _render_pdf(path: str, dpi: int) -> List[np.ndarray]:
    import fitz
    images = []
    with fitz.open(path) as doc:
        for pg in doc:
            pix = pg.get_pixmap(dpi=dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            if pix.n == 4:
                img = img[:, :, :3]
            images.append(np.ascontiguousarray(img[:, :, ::-1]))  # RGB -> BGR
    return images


def _load_images(path: str, dpi: int) -> List[np.ndarray]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _render_pdf(path, dpi=dpi)
    if ext in SUPPORTED_IMAGE_EXT:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        return [img]
    raise ValueError(f"Unsupported file type: {ext}")


def _limit_image_size(img: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale so the longest side is at most max_side pixels."""
    if max_side <= 0:
        return img
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img
    scale = max_side / longest
    return cv2.resize(
        img,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )


def _looks_like_thermal_image(img: np.ndarray) -> bool:
    """Tall/narrow slip (or ribbon on a wide scan) — candidate for thermal preprocess."""
    h, w = img.shape[:2]
    if w <= 0:
        return False
    aspect = h / float(w)
    if aspect >= 1.6 or w < 900:
        return True
    return False


def _crop_side_margins(img: np.ndarray, pad: int = 12) -> np.ndarray:
    """Crop empty left/right margins when content is a centered thermal ribbon."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    # Ink is darker than paper
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cols = np.where(bw.sum(axis=0) > (bw.shape[0] * 0.01))[0]
    if cols.size < 10:
        return img
    x0 = max(int(cols[0]) - pad, 0)
    x1 = min(int(cols[-1]) + pad, img.shape[1])
    band = x1 - x0
    # Only crop when the content is a clear ribbon (< 70% of page width)
    if band < img.shape[1] * 0.70 and band > 80:
        return img[:, x0:x1]
    return img


def _preprocess_thermal(img: np.ndarray) -> np.ndarray:
    """Light contrast boost for faded thermal ink — paper path untouched."""
    img = _crop_side_margins(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    except Exception:
        gray = cv2.equalizeHist(gray)
    # Keep 3-channel BGR for PaddleOCR
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _group_lines(words: List[Word]) -> List[Line]:
    """Group word boxes into visual lines by vertical-center proximity."""
    if not words:
        return []
    words = sorted(words, key=lambda w: (w.cy, w.x0))
    med_h = float(np.median([w.height for w in words]))
    threshold = max(med_h * 0.55, 6.0)

    lines: List[Line] = []
    for w in words:
        placed = False
        for ln in lines:
            if abs(ln.cy - w.cy) <= threshold:
                ln.words.append(w)
                placed = True
                break
        if not placed:
            lines.append(Line(words=[w]))
    for ln in lines:
        ln.words.sort(key=lambda w: w.x0)
    lines.sort(key=lambda ln: ln.cy)
    return lines


def ocr_file(
    path: str,
    dpi: Optional[int] = None,
    max_pages: Optional[int] = None,
    max_side: Optional[int] = None,
) -> List[Page]:
    """Run OCR on a PDF or image and return structured pages."""
    dpi = OCR_DPI if dpi is None else dpi
    max_pages = OCR_MAX_PAGES if max_pages is None else max_pages
    max_side = OCR_MAX_SIDE if max_side is None else max_side

    ocr = get_ocr()
    images = _load_images(path, dpi=dpi)
    if max_pages:
        images = images[:max_pages]

    pages: List[Page] = []
    try:
        for idx, img in enumerate(images):
            thermal_like = _looks_like_thermal_image(img)
            if thermal_like:
                img = _preprocess_thermal(img)
                # Tall slips: allow a slightly larger long side so small text stays readable
                side = max(max_side, 2800) if max_side > 0 else max_side
            else:
                side = max_side
            img = _limit_image_size(img, side)
            result = ocr.ocr(img, cls=OCR_USE_ANGLE_CLS)
            words: List[Word] = []
            if result and result[0]:
                for box, (text, conf) in result[0]:
                    xs = [p[0] for p in box]
                    ys = [p[1] for p in box]
                    text = text.strip()
                    if not text:
                        continue
                    words.append(Word(
                        text=text, conf=float(conf),
                        x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys),
                    ))
            page = Page(
                number=idx + 1,
                width=img.shape[1],
                height=img.shape[0],
                words=words,
                lines=_group_lines(words),
            )
            pages.append(page)
            del img, result
    finally:
        images.clear()
        try:
            import gc
            gc.collect()
        except Exception:
            pass
    return pages
