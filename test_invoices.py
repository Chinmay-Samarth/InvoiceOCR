"""Batch-test invoice extraction; caches OCR results for fast re-runs."""
import json
import os
import pickle
import sys
import time

from app.extractors import extract_all
from app.ocr_engine import ocr_file

CACHE_DIR = ".ocr_cache"


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


def check_result(name: str, data: dict) -> list:
    issues = []
    if not data["supplier"]["name"]:
        issues.append("missing supplier.name")
    if not data["supplier"]["gstin"]:
        issues.append("missing supplier.gstin")
    if not data["invoice"]["number"]:
        issues.append("missing invoice.number")
    if not data["invoice"]["date"]:
        issues.append("missing invoice.date")
    if not data["line_items"]:
        issues.append("missing line_items")
    if not data["totals"]["total_amount"]:
        issues.append("missing totals.total_amount")
    if data["warnings"]:
        issues.extend(data["warnings"])
    return issues


def main():
    paths = sys.argv[1:]
    if not paths:
        print("Usage: python test_invoices.py invoice1.pdf ...")
        sys.exit(1)

    summary = []
    for path in paths:
        if not os.path.exists(path):
            print(f"SKIP (not found): {path}")
            continue
        name = os.path.basename(path)
        t0 = time.perf_counter()
        pages = get_pages(path)
        result = extract_all(pages)
        dt = time.perf_counter() - t0
        data = json.loads(result.json())
        issues = check_result(name, data)
        summary.append((name, dt, data, issues))
        status = "OK" if not issues else "ISSUES"
        print(f"\n{'='*60}\n{name}  [{status}]  {dt:.0f}s  conf={data['ocr_confidence']}")
        print(f"  supplier: {data['supplier']['name'][:50]}")
        print(f"  gstin:    {data['supplier']['gstin']}")
        print(f"  invoice:  {data['invoice']['number']}  date={data['invoice']['date']}")
        print(f"  items:    {len(data['line_items'])}  tax_lines: {len(data['tax_lines'])}")
        print(f"  totals:   base={data['totals']['base_amount']}  tax={data['totals']['tax_amount']}  total={data['totals']['total_amount']}")
        if issues:
            print(f"  issues:   {issues}")
        for it in data["line_items"]:
            print(f"    - {it['description'][:60]}  qty={it['quantity']}  total={it['total']}")

    out = os.path.join(CACHE_DIR, "batch_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            [{"file": n, "seconds": round(d, 1), "issues": i, "data": data}
             for n, d, data, i in summary],
            f, indent=2, ensure_ascii=False,
        )
    print(f"\n\nReport saved to {out}")
    print(f"Passed: {sum(1 for _, _, _, i in summary if not i)}/{len(summary)}")


if __name__ == "__main__":
    main()
