"""Re-run extractors on cached OCR (no re-OCR)."""
import json
import pickle
import sys
from app.extractors import extract_all

paths = sys.argv[1:]
for path in paths:
    import os
    key = os.path.join(".ocr_cache", os.path.basename(path) + ".pkl")
    with open(key, "rb") as f:
        pages = pickle.load(f)
    res = extract_all(pages)
    d = json.loads(res.json())
    name = os.path.basename(path)
    issues = []
    if not d["invoice"]["number"]: issues.append("no inv#")
    if not d["line_items"]: issues.append("no items")
    if not d["totals"]["total_amount"]: issues.append("no total")
    if not d["tax_lines"]: issues.append("no tax")
    if d["warnings"]: issues.extend(d["warnings"])
    print(f"{name}: items={len(d['line_items'])} tax={len(d['tax_lines'])} "
          f"base={d['totals']['base_amount']} tax={d['totals']['tax_amount']} "
          f"total={d['totals']['total_amount']} inv={d['invoice']['number']} "
          f"{'OK' if not issues else issues}")
