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
            img = _limit_image_size(img, max_side)
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
