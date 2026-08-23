"""CLI: parse an invoice PDF/image and print the extracted JSON.

Usage:
    python parse_invoice.py path\to\invoice.pdf [more files...]
"""
import json
import sys

from app.extractors import extract_all
from app.ocr_engine import ocr_file


def parse(path: str) -> dict:
    pages = ocr_file(path)
    result = extract_all(pages)
    return json.loads(result.json())


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for path in sys.argv[1:]:
        data = parse(path)
        print(f"\n===== {path} =====")
        print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
