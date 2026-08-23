"""Dev utility: dump OCR lines (with coordinates) for the sample invoices."""
import sys
from app.ocr_engine import ocr_file

paths = sys.argv[1:] or [
    r"c:\Users\Admin\Downloads\arrkays bright steel.pdf",
    r"c:\Users\Admin\Downloads\Akash Aluminium Company.pdf",
    r"c:\Users\Admin\Downloads\bharat electro chen.pdf",
]

for p in paths:
    out = "ocr_" + p.split("\\")[-1].replace(".pdf", "").replace(" ", "_") + ".txt"
    pages = ocr_file(p)
    with open(out, "w", encoding="utf-8") as f:
        for page in pages:
            f.write(f"### PAGE {page.number} {page.width}x{page.height} conf={page.mean_confidence():.3f}\n")
            for ln in page.lines:
                f.write(f"[y={ln.cy:6.0f}] " + " | ".join(
                    f"({w.x0:.0f},{w.text})" for w in ln.words) + "\n")
    print("wrote", out)
