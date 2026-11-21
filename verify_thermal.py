"""Quick verification of thermal classifier + extractors."""
import json
import pickle
from pathlib import Path

from app.extractors import extract_all
from app.ocr_engine import ocr_file
from app.thermal_extractors import DOC_PAPER, classify_document, extract_thermal

ASSETS = Path(
    r"C:\Users\Admin\.cursor\projects"
    r"\e-Chinmay-Projects-VS-Files-Python-Files-InvoiceOCR\assets"
)
CACHE = Path(".ocr_cache/thermal")

THERMAL_FILES = [
    "S48-9ae569d7-3b1c-4e9b-95d6-1a252fc0e3e2.jpg",
    "V35-434804b5-1e44-4f49-a59c-31939d35a404.jpg",
    "V85-e168c421-347d-4220-9052-63a67af31cd2.jpg",
    "W23-3709064b-3f96-46fd-992a-f1f4eff4cb12.jpg",
    "V19-3c521228-1a47-4727-b18b-908ff2191ebe.jpg",
    "V32-ee34c347-30ee-431c-b6f6-3700d0af0599.jpg",
    "W41-142ced73-7ab6-47c0-b4bd-348fec15effd.jpg",
    "T53-5b00db3e-745c-4fbd-af9f-01c6a3611c5e.jpg",
    "V73-10375e09-864c-43c9-a6cc-1a1aad106462.jpg",
]


def _dump(result):
    if hasattr(result, "model_dump_json"):
        return json.loads(result.model_dump_json())
    return json.loads(result.json())


def _pages(name: str):
    CACHE.mkdir(parents=True, exist_ok=True)
    cp = CACHE / f"{name}.pkl"
    if cp.exists():
        with open(cp, "rb") as f:
            return pickle.load(f)
    pages = ocr_file(str(ASSETS / name))
    with open(cp, "wb") as f:
        pickle.dump(pages, f)
    return pages


def main():
    print("=== THERMAL SAMPLES ===")
    for name in THERMAL_FILES:
        if not (ASSETS / name).exists():
            print("MISSING", name)
            continue
        pages = _pages(name)
        doc = classify_document(pages)
        if doc == DOC_PAPER:
            result = extract_all(pages)
            result.document_type = DOC_PAPER
        else:
            result = extract_thermal(pages, doc)
        d = _dump(result)
        print(f"\n--- {name} ---")
        print(" type:", d["document_type"])
        print(" supplier:", (d["supplier"]["name"] or "")[:60])
        print(
            " invoice:",
            d["invoice"]["number"],
            d["invoice"]["date"],
            "due",
            d["invoice"].get("due_date", ""),
        )
        print(" totals:", d["totals"])
        print(" tax:", d["tax_lines"])
        for it in d["line_items"]:
            desc = it["description"]
            print(
                "  item:",
                repr(desc),
                "|",
                it["quantity"],
                "x",
                it["unit_price"],
                "=",
                it["total"],
            )

    print("\n=== PAPER CACHE REGRESSION ===")
    paper = Path(".ocr_cache") / "sampleinvoice4.pdf.pkl"
    with open(paper, "rb") as f:
        pages = pickle.load(f)
    doc = classify_document(pages)
    result = extract_all(pages)
    result.document_type = DOC_PAPER
    d = _dump(result)
    print("classify:", doc)
    print("document_type:", d["document_type"])
    print("totals:", d["totals"])
    print("items:", len(d["line_items"]), "tax:", len(d["tax_lines"]))
    print("supplier:", (d["supplier"]["name"] or "")[:80])


if __name__ == "__main__":
    main()
