"""Dev harness: run extraction on the sample invoices, caching OCR results
so extractor changes can be tested without re-running OCR (slow on CPU)."""
import json
import os
import pickle
import sys

from app.extractors import extract_all
from app.ocr_engine import ocr_file

CACHE_DIR = ".ocr_cache"
SAMPLES = [
    r"c:\Users\Admin\Downloads\arrkays bright steel.pdf",
    r"c:\Users\Admin\Downloads\Akash Aluminium Company.pdf",
    r"c:\Users\Admin\Downloads\bharat electro chen.pdf",
]


def get_pages(path: str):
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = os.path.join(CACHE_DIR, os.path.basename(path) + ".pkl")
    if os.path.exists(key):
        with open(key, "rb") as f:
            return pickle.load(f)
    pages = ocr_file(path)
    with open(key, "wb") as f:
        pickle.dump(pages, f)
    return pages


if __name__ == "__main__":
    paths = sys.argv[1:] or SAMPLES
    for p in paths:
        pages = get_pages(p)
        result = extract_all(pages)
        print(f"\n{'=' * 25} {os.path.basename(p)} {'=' * 25}")
        print(json.dumps(json.loads(result.json()), indent=2, ensure_ascii=False))
