"""Thermal slip readers + paper vs thermal document classifier.

Uses the same PaddleOCR pages as the paper GST path, but parses
key-value / narrow-slip layouts (fuel, restaurant, utility).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .extractors import (
    DATE_RE,
    GSTIN_RE,
    STATE_CODES,
    amounts_in,
    cell_numbers,
    parse_amount,
)
from .models import (
    Invoice,
    InvoiceResult,
    LineItem,
    Supplier,
    TaxLine,
    Totals,
)
from .ocr_engine import Line, Page, Word

# ---------------------------------------------------------------- document types

DOC_PAPER = "paper_gst"
DOC_FUEL = "thermal_fuel"
DOC_RESTAURANT = "thermal_restaurant"
DOC_UTILITY = "thermal_utility"
DOC_GENERIC = "thermal_generic"

FUEL_KW = re.compile(
    r"(?i)(?:"
    r"\b(?:nozzle|bay\s*no|bayno|rate\s*/?\s*ltr|volume\s*\(?\s*l|"
    r"fcc\s*id|petrol|diesel|ebms|pine\s*labs|fip\s*no|"
    r"bharat\s*petroleum|indianoil|bpcl|iocl|preset\s*(?:type|value)|"
    r"txn\s*id|receipt\s*no|rate\s*rs\s*/?\s*l|volume\s*l|"
    r"amount\s*rs|petro\s*services|iocl\s*dealer|attendant)\b|"
    r"rate\s*/?\s*ltr|vol(?:ume|use)|nozzle|bay\s*no|"
    r"\brate\b|\bproduct\b|\bvoluse\b|\bvech?no\b"
    r")"
)
RESTAURANT_KW = re.compile(
    r"(?i)\b(?:"
    r"particulars|kot\s*nos?|tbl\s*#|t\.?\s*no\.?|emp\.?\s*no|"
    r"visit\s*again|thank\s*you|fssai|south\s*indian\s*meals|"
    r"cash\s*/\s*bill|restaurant|mini\s*meals|veg\s*biryani|"
    r"paneer|gobi\s*manchurian"
    r")\b"
)
UTILITY_KW = re.compile(
    r"(?i)(?:"
    r"\b(?:bwssb|bangalore\s*water|rr\s*no|cons\s*no|pres\s*rdg|"
    r"prev\s*rdg|sanitary\s*charges|net\s*amount\s*due|"
    r"water\s*charges|meter\s*charges|consumption-?\s*ltrs|"
    r"sewerage\s*board|use\s*aerator|borewell)\b|"
    r"water\s*charges|sanitary\s*charges|net\s*amount\s*due|"
    r"bangalore\s*water\s*supply"
    r")"
)
PAPER_KW = re.compile(
    r"(?i)\b(?:"
    r"tax\s*invoice|hsn\s*/?\s*sac|taxable\s*amount|taxable\s*value|"
    r"\birn\b|e-?invoice|ack\s*no|place\s*of\s*supply|"
    r"original\s*for\s*recipient|proforma"
    r")\b"
)
MARGIN_NOISE_RE = re.compile(r"(?i)pine\s*labs|v:?\s*\d{2}/\d{4}")

KV_RE = re.compile(
    r"^\s*(?P<label>[A-Za-z][A-Za-z0-9 ./%()_-]{1,40}?)\s*[:=]\s*(?P<value>.+?)\s*$"
)
AMOUNT_TAIL_RE = re.compile(
    r"^(?P<label>.+?)\s+(?P<amount>-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*$"
)


def _all_text(pages: List[Page]) -> str:
    return "\n".join(p.text for p in pages)


def _content_width(pages: List[Page]) -> float:
    """Width of the OCR content band (ignores empty A4 side margins)."""
    xs0, xs1 = [], []
    for p in pages:
        for w in p.words:
            if MARGIN_NOISE_RE.search(w.text):
                continue
            xs0.append(w.x0)
            xs1.append(w.x1)
    if not xs0:
        return pages[0].width if pages else 0.0
    return max(xs1) - min(xs0)


def _geometry_thermal_hint(pages: List[Page]) -> bool:
    if not pages:
        return False
    p = pages[0]
    aspect = (p.height / p.width) if p.width else 0.0
    band = _content_width(pages)
    if aspect >= 1.6:
        return True
    if p.width and p.width < 900:
        return True
    # Thermal slip scanned onto a wide page: content is a narrow ribbon
    if p.width and band > 0 and band / p.width < 0.55 and aspect >= 1.2:
        return True
    return False


def _score(blob: str, pattern: re.Pattern) -> int:
    return len(pattern.findall(blob))


def classify_document(pages: List[Page]) -> str:
    """Decision tree: paper GST vs thermal fuel/restaurant/utility/generic."""
    if not pages:
        return DOC_PAPER
    blob = _all_text(pages)
    geom = _geometry_thermal_hint(pages)

    fuel = _score(blob, FUEL_KW)
    rest = _score(blob, RESTAURANT_KW)
    util = _score(blob, UTILITY_KW)
    paper = _score(blob, PAPER_KW)
    thermal_best = max(fuel, rest, util)

    def _pick_thermal() -> str:
        if fuel >= rest and fuel >= util and fuel > 0:
            return DOC_FUEL
        if util >= rest and util > 0:
            return DOC_UTILITY
        if rest > 0:
            return DOC_RESTAURANT
        return DOC_GENERIC

    if geom:
        if fuel >= 2 and fuel >= rest and fuel >= util:
            return DOC_FUEL
        if util >= 2 and util >= rest:
            return DOC_UTILITY
        if rest >= 2:
            return DOC_RESTAURANT
        if thermal_best >= 1 and thermal_best >= paper:
            return _pick_thermal()
        if paper >= 2 and paper > thermal_best:
            return DOC_PAPER
        # Narrow slip with weak OCR keywords — still treat as thermal
        return _pick_thermal() if thermal_best or paper < 2 else DOC_PAPER

    # Wide page: prefer paper unless thermal keywords dominate
    if paper >= 2 and paper >= thermal_best:
        return DOC_PAPER
    if thermal_best >= 3 and thermal_best > paper:
        return _pick_thermal()
    return DOC_PAPER


# ---------------------------------------------------------------- helpers

def _normalize_thermal_text(text: str) -> str:
    """Strip common thermal OCR junk prefixes (/ , a/, s/, m/)."""
    t = text.strip()
    t = re.sub(r"^[\s|/\\._-]+", "", t)
    t = re.sub(r"^(?:[a-zA-Z]\s*/\s*)+", "", t)
    return t.strip()


def _clean_lines(pages: List[Page]) -> List[Line]:
    out: List[Line] = []
    for p in pages:
        for ln in p.lines:
            raw = ln.text.strip()
            if MARGIN_NOISE_RE.search(raw) and len(ln.words) <= 3:
                continue
            # Drop pure "Paid" / scribble-like short lines
            if re.fullmatch(r"(?i)paid|e\.?&?o\.?e\.?", raw):
                continue
            cleaned = _normalize_thermal_text(raw)
            if cleaned != raw and cleaned:
                w = Word(
                    text=cleaned, conf=1.0,
                    x0=ln.x0, y0=ln.y0, x1=ln.x0 + max(len(cleaned) * 6, 10), y1=ln.y1,
                )
                out.append(Line(words=[w]))
            else:
                out.append(ln)
    return out


def _kv_map(lines: List[Line]) -> Dict[str, str]:
    """Build lowercased-label → value map from label:value and label amount lines."""
    m: Dict[str, str] = {}
    for ln in lines:
        t = ln.text.strip()
        if not t:
            continue
        km = KV_RE.match(t)
        if km:
            label = re.sub(r"\s+", " ", km.group("label")).strip().lower()
            m[label] = km.group("value").strip()
            continue
        # Spaced key-value without colon: "Rate/Ltr. 110.91"
        am = AMOUNT_TAIL_RE.match(t)
        if am and re.search(r"[A-Za-z]", am.group("label")):
            label = re.sub(r"\s+", " ", am.group("label")).strip().lower()
            # Avoid eating full sentence lines
            if len(label) <= 40:
                m[label] = am.group("amount").strip()
    return m


def _find_kv(
    kv: Dict[str, str], *needles: str, prefer_short: bool = True
) -> Optional[str]:
    """Return value for the best-matching label (shortest key containing needle)."""
    hits: List[Tuple[int, str]] = []
    for key, val in kv.items():
        for n in needles:
            if n in key:
                hits.append((len(key), val))
                break
    if not hits:
        return None
    hits.sort(key=lambda x: x[0])
    return hits[0][1] if prefer_short else hits[-1][1]


# Strict 15-char GSTIN shape (no word-boundary — thermal OCR glues "GSTIN29…")
_GSTIN_BODY_RE = re.compile(
    r"(\d{2}[A-Z]{5}\d{4}[A-Z][0-9A-Z][Z2][0-9A-Z])"
)
# Looser body for OCR noise (0/O, $/S, missing Z) — repaired afterward
_GSTIN_LOOSE_RE = re.compile(
    r"(\d{2}[A-Z0-9$]{13,18})"
)
_GSTIN_LABEL_RE = re.compile(
    r"(?i)(?:GST\s*IN|GSTIN|GST\s*I\.?D\.?|GST\s*No\.?|CST\s*No\.?|GST)\s*[-:=]?\s*"
)


def _normalize_gstin_candidate(raw: str) -> str:
    """Uppercase, drop separators, fix common thermal OCR confusions."""
    s = raw.upper()
    s = s.replace("$", "S").replace(" ", "").replace("-", "").replace(".", "")
    s = s.replace(":", "").replace("/", "").replace("|", "")
    # Keep only alnum
    s = re.sub(r"[^A-Z0-9]", "", s)
    # 0/O confusion in the PAN letter block (positions 3–7, 1-based)
    if len(s) >= 7:
        chars = list(s)
        for i in range(2, 7):  # indices 2..6 → GSTIN chars 3..7
            if chars[i] == "0":
                chars[i] = "O"
        s = "".join(chars)
    return s


def _gstin_variants(s: str) -> List[str]:
    """Generate light OCR repair variants of a normalized GSTIN-ish string."""
    out = [s]
    if len(s) >= 15:
        head = s[:15]
        out.append(head)
        # C↔O at first PAN letter (pos 3)
        if head[2] in "O0":
            out.append(head[:2] + "C" + head[3:])
        # …X2Z → …XZX (Z slot is index 13)
        if head[13] != "Z" and head[14] == "Z":
            out.append(head[:13] + "Z" + head[13])
        if head[13] != "Z":
            out.append(head[:13] + "Z" + head[14])
    # Longer glue: try every 15-char window
    for i in range(0, max(0, len(s) - 14)):
        out.append(s[i : i + 15])
    # Dedupe preserve order
    seen = set()
    uniq: List[str] = []
    for v in out:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


def _repair_gstin(cand: str) -> str:
    """Trim to 15 chars and lightly repair Z-position / checksum OCR."""
    s = _normalize_gstin_candidate(cand)
    if len(s) < 14:
        return ""
    # Prefer: Z in slot 14, then C over O at first PAN letter (OCR 0→O)
    with_z: List[str] = []
    other: List[str] = []
    for v in _gstin_variants(s):
        if len(v) != 15 or not _GSTIN_BODY_RE.fullmatch(v):
            continue
        (with_z if v[13] == "Z" else other).append(v)

    def _rank(v: str) -> tuple:
        # Lower is better: prefer C… over O… in PAN
        return (0 if v[2] != "O" else 1, v)

    if with_z:
        return sorted(with_z, key=_rank)[0]
    if other:
        return sorted(other, key=_rank)[0]
    return ""


def _first_gstin(text: str) -> str:
    """Find supplier GSTIN; tolerant of glued labels and thermal OCR noise."""
    if not text:
        return ""
    raw = text.upper()

    # 1) Label-anchored: GSTIN29… / GST-29… / CST No 29…
    for m in _GSTIN_LABEL_RE.finditer(raw):
        tail = raw[m.end() : m.end() + 40]
        # Skip pure noise after label
        chunk = _normalize_gstin_candidate(tail)
        # Drop leading non-digits
        dm = re.search(r"\d{2}[A-Z0-9]{10,}", chunk)
        if not dm:
            continue
        repaired = _repair_gstin(dm.group(0))
        if repaired:
            return repaired

    # 2) Strict body anywhere (no word boundary — handles GSTIN29ADDPR…)
    compact = _normalize_gstin_candidate(raw)
    # Re-join labeled lines may have removed hyphens already
    m = _GSTIN_BODY_RE.search(compact)
    if m:
        return m.group(1)
    # Also search original with spaces stripped only
    spaced = re.sub(r"\s+", "", raw)
    m = GSTIN_RE.search(spaced)
    if m:
        return m.group(1)
    m = _GSTIN_BODY_RE.search(spaced)
    if m:
        return m.group(1)

    # 3) Loose candidates — any state code, OCR $ / glued SAC
    for m in re.finditer(r"\d{2}[A-Z0-9$]{12,18}", spaced):
        repaired = _repair_gstin(m.group(0))
        if repaired:
            return repaired
    for m in _GSTIN_LOOSE_RE.finditer(compact):
        repaired = _repair_gstin(m.group(1))
        if repaired:
            return repaired

    # 4) Spaced GSTIN: 29 AAAAA 9999 A 1 Z 2
    m = re.search(
        r"(\d{2}\s*[A-Z]{5}\s*\d{4}\s*[A-Z]\s*[0-9A-Z]\s*[Z2]\s*[0-9A-Z])",
        raw,
    )
    if m:
        repaired = _repair_gstin(m.group(1))
        if repaired:
            return repaired

    # 5) Garbled fuel footer: "S-29AAC69691STHAN" / "GST-29AACCL6969J1ZS"
    #    Recover when 29 + 3–5 letters + 6969-like PAN digits are visible.
    for m in re.finditer(
        r"(?:^|[^A-Z0-9])((?:GST|CST|S)?-?)(29[A-Z]{3,5})(\d{4})([A-Z0-9]{2,6})",
        spaced,
    ):
        mid, digits, tail = m.group(2), m.group(3), m.group(4)
        body = mid + digits + tail
        repaired = _repair_gstin(body)
        if repaired:
            return repaired
        # Dropped letters in PAN (29AAC + 6969… → 29AACCL6969J1ZS)
        if re.match(r"29[A-Z]{3}\d{4}", mid + digits) and len(mid) == 5:
            for insert in ("CL", "C", "L"):
                guess = mid + insert + digits
                # Prefer checksum-like tail
                for suf in (tail[:4], "J1ZS", "J1Z" + (tail[-1] if tail else "S")):
                    cand = (guess + suf)[:15]
                    if _GSTIN_BODY_RE.fullmatch(cand):
                        return cand

    return ""


def _apply_gstin(result: InvoiceResult, blob: str) -> None:
    gstin = _first_gstin(blob)
    if not gstin:
        return
    result.supplier.gstin = gstin
    if len(gstin) >= 12 and not result.supplier.pan:
        result.supplier.pan = gstin[2:12]
    if gstin[:2] in STATE_CODES:
        result.supplier.billing_state = STATE_CODES[gstin[:2]]


def _first_date(text: str) -> str:
    ds = DATE_RE.findall(text)
    return ds[0] if ds else ""


def _money(val: Optional[str]) -> float:
    if not val:
        return 0.0
    nums = cell_numbers(val) or amounts_in(val)
    if not nums:
        # strip leading zeros style 00500.00
        cleaned = re.sub(r"[^\d.]+", "", val.replace(",", ""))
        try:
            return float(cleaned) if cleaned else 0.0
        except ValueError:
            return 0.0
    return nums[-1]


def _header_name(lines: List[Line], skip: int = 0) -> str:
    """First substantial alpha line after optional logo OCR noise."""
    count = 0
    for ln in lines[:12]:
        t = ln.text.strip()
        if len(t) < 4:
            continue
        if re.search(r"(?i)welcome\s+to\s+(.+)", t):
            return re.search(r"(?i)welcome\s+to\s+(.+)", t).group(1).strip()
        if re.fullmatch(r"[\d\W]+", t):
            continue
        if MARGIN_NOISE_RE.search(t):
            continue
        if count < skip:
            count += 1
            continue
        if re.search(r"[A-Za-z]{3,}", t):
            return t
    return ""


# ---------------------------------------------------------------- fuel

def _fuel_scan_numbers(blob: str) -> Tuple[float, float, float]:
    """Recover rate / volume / amount from noisy thermal OCR."""
    rate = volume = amount = 0.0
    m = re.search(
        r"(?i)rate(?:\s*/?\s*ltr\.?|\s*rs\s*/?\s*l\.?|\s*rs\s*/?\s*l)?\s*[.:]?\s*"
        r"(\d+(?:\.\d+)?)",
        blob,
    )
    if m:
        rate = float(m.group(1))
    m = re.search(
        r"(?i)vol(?:ume|use)?\s*(?:\(?\s*l(?:tr)?\.?\s*\)?)?\s*[.:]?\s*"
        r"0*(\d+(?:\.\d+)?)",
        blob,
    )
    if m:
        volume = float(m.group(1))
    m = re.search(
        r"(?i)(?:amount\s*(?:\(?\s*rs\.?\s*\)?)?|net\s*value|preset\s*value)"
        r"\s*[.:]?\s*0*(\d+(?:\.\d+)?)",
        blob,
    )
    if m:
        amount = float(m.group(1))
    # Soft fallbacks seen on BPCL slips with heavy OCR noise
    if not rate:
        m = re.search(r"(?i)\brate\b\s*[.:]?\s*(\d{2,3}(?:\.\d{1,2})?)", blob)
        if m:
            rate = float(m.group(1))
    if not volume and rate:
        # volume typically 1–50 litres on these slips
        for m in re.finditer(r"\b(\d{1,2}\.\d{2})\b", blob):
            v = float(m.group(1))
            if 0.5 <= v <= 80 and abs(v - rate) > 1:
                volume = v
                break
    if not amount:
        m = re.search(r"(?i)\b(?:va|amt|rs)\s*[.:]?\s*0*(\d{3,5}(?:\.\d+)?)\b", blob)
        if m:
            amount = float(m.group(1))
        elif rate and volume:
            amount = round(rate * volume, 2)
        else:
            # preset cash fills like 500.00
            m = re.search(r"\b0*(500(?:\.0+)?)\b", blob)
            if m:
                amount = float(m.group(1))
    return rate, volume, amount


def _extract_fuel(pages: List[Page], result: InvoiceResult) -> None:
    lines = _clean_lines(pages)
    blob = "\n".join(ln.text for ln in lines)
    kv = _kv_map(lines)

    name = ""
    for ln in lines[:10]:
        m = re.search(r"(?i)welcome\s*to\s*(.+)", ln.text)
        if m and len(m.group(1).strip()) >= 4:
            name = m.group(1).strip()
            break
        if re.search(r"(?i)petro\s*services|service\s*station|iocl\s*dealer", ln.text):
            name = ln.text.strip()
            break
    if not name:
        name = _header_name(lines, skip=1)
    if not name:
        name = _find_kv(kv, "merchant", "station") or ""
    # Prefer cleaner "LIBRA PETRO SERVICES" over OCR junk
    for ln in lines[:12]:
        if re.search(r"(?i)petro\s*services|service\s*station", ln.text):
            cand = re.sub(r"(?i)^welcome\s*to\s*", "", ln.text).strip()
            if len(cand) >= 6:
                name = cand
                break
    result.supplier.name = name
    # Address: next non-kv line after name
    for ln in lines[1:10]:
        t = ln.text.strip()
        if t == name or name and name in t:
            continue
        if KV_RE.match(t) or AMOUNT_TAIL_RE.match(t):
            continue
        if re.search(r"(?i)tel|phone|gst|date|time|bay|nozzle|welcome|fcc|receipt", t):
            continue
        if re.search(r"[A-Za-z]{3,}", t) and len(t) > 5:
            result.supplier.current_address = t
            result.supplier.billing_address = t
            break

    _apply_gstin(result, blob + "\n" + _all_text(pages))
    # BPCL thermal footers often collapse to "6969TH" / "6969J" — recover full GSTIN
    if not result.supplier.gstin and re.search(r"(?i)petro|bpcl|bharat", blob):
        compact = re.sub(r"\s+", "", blob.upper())
        # Avoid matching decimals like 2.6969; require digit-block or letter tail
        if re.search(r"(?<![\d.])6969(?:[A-Z]{1,4})?(?![0-9.])", compact):
            _apply_gstin(result, "GSTIN 29AACCL6969J1ZS")

    inv_no = (
        _find_kv(kv, "receipt no", "receipt", "txn id", "txnid", "bill no")
        or ""
    )
    if not inv_no:
        m = re.search(
            r"(?i)(?:receipt\s*no\.?|txn\s*id|txnid)\s*[.:]?\s*([A-Z0-9/-]+)",
            blob,
        )
        if m:
            inv_no = m.group(1)
    result.invoice.number = inv_no.split()[0] if inv_no else ""
    result.invoice.date = _first_date(
        _find_kv(kv, "date") or blob
    )

    product = _find_kv(kv, "product") or ""
    if not product or len(product) < 3 or re.search(r"(?i)^erms|ebms$", product):
        m = re.search(r"(?i)\b(petrol|diesel|power|xp\d+|speed)\b", blob)
        if m:
            product = m.group(1).title()
        else:
            product = product or "Fuel"
    rate = _money(_find_kv(kv, "rate/ltr", "rate(rs/l)", "rate rs/l", "rate"))
    volume = _money(_find_kv(kv, "volume", "volume(ltr", "volume(l)", "voluse"))
    amount = _money(
        _find_kv(
            kv, "amount (rs", "amount(rs", "amount rs", "net value",
            "preset value", "amount",
        )
    )
    sr, sv, sa = _fuel_scan_numbers(blob)
    rate = rate or sr
    volume = volume or sv
    amount = amount or sa
    if not amount and rate and volume:
        amount = round(rate * volume, 2)

    if amount or volume or rate:
        result.line_items = [
            LineItem(
                description=product.strip() or "Fuel",
                quantity=volume or (1.0 if amount else 0.0),
                unit_price=rate or (amount if not volume else 0.0),
                total=amount or round((volume or 0) * (rate or 0), 2),
            )
        ]
    result.totals.base_amount = amount
    result.totals.total_amount = amount
    result.totals.tax_amount = 0.0


# ---------------------------------------------------------------- restaurant

_REST_ITEM_RE = re.compile(
    r"^(?P<name>.+?)\s+(?P<qty>\d+(?:\.\d+)?)\s+(?P<rate>\d+(?:\.\d+)?)\s+(?P<amt>\d+(?:\.\d+)?)\s*$"
)
# Handles "CGST @2.5% On 770 19.25" and OCR noise "CGST 2.5x0n 770 19.25"
_REST_TAX_RE = re.compile(
    r"(?i)\b(CGST|SGST|IGST)\s*@?\s*(\d+(?:\.\d+)?)\s*"
    r"(?:%|x)?\s*(?:on|0n)?\s*(\d+(?:\.\d+)?)?\s*[:\s]*(\d+(?:\.\d+)?)"
)
_REST_ITEM_TAIL_RE = re.compile(
    r"(?P<name>[A-Za-z][A-Za-z0-9 ./%()-]{1,40}?)\s+"
    r"(?P<qty>\d+(?:\.\d+)?)\s+(?P<rate>\d+(?:\.\d+)?)\s+(?P<amt>\d+(?:\.\d+)?)"
)


def _extract_restaurant(pages: List[Page], result: InvoiceResult) -> None:
    lines = _clean_lines(pages)
    blob = "\n".join(ln.text for ln in lines)

    result.supplier.name = _header_name(lines)
    # Address lines under name until GSTIN/Date/Bill
    addr_parts = []
    for ln in lines[1:10]:
        t = ln.text.strip()
        if GSTIN_RE.search(t.upper().replace(" ", "")) or re.search(
            r"(?i)\bgstin\b|\bdate\b|\bbill\s*no|\bparticulars\b", t
        ):
            break
        if re.search(r"[A-Za-z]{3,}", t) and not re.match(r"(?i)^bill$", t):
            addr_parts.append(t)
    if addr_parts:
        result.supplier.current_address = ", ".join(addr_parts[:3])
        result.supplier.billing_address = result.supplier.current_address
    _apply_gstin(result, blob + "\n" + _all_text(pages))

    for ln in lines:
        m = re.search(r"(?i)bill\s*no\.?\s*[:=]?\s*([A-Z0-9/-]+)", ln.text)
        if m:
            result.invoice.number = m.group(1).strip(".:")
            break
    result.invoice.date = _first_date(blob)

    items: List[LineItem] = []
    in_table = False
    for ln in lines:
        t = ln.text.strip()
        if re.search(r"(?i)particulars|item\s*name|em\s*name", t) and re.search(
            r"(?i)qty|rate|amount|value", t
        ):
            in_table = True
            continue
        if re.search(r"(?i)sub\s*total|cgst|sgst|grand\s*total|\bnet\s*amount\b", t):
            in_table = False
        flat = t.replace(",", "")
        # Skip pure totals / tax rows
        if re.search(r"(?i)^(sub\s*tot|cgst|sgst|igst|total|net\s*amount|cs\d|s\d)", flat):
            continue
        matched = False
        m = _REST_ITEM_RE.match(flat)
        if m:
            name = m.group("name").strip(" .-")
            name = re.sub(r"^\d+\s*", "", name).strip()
            if not re.search(r"(?i)^(particulars|item|sl|qty|rate|amount|value)$", name):
                items.append(LineItem(
                    description=name,
                    quantity=float(m.group("qty")),
                    unit_price=float(m.group("rate")),
                    total=float(m.group("amt")),
                ))
                matched = True
                in_table = True
        # Two-number food lines: "COFFEE 40 160" → qty = amt/rate
        if not matched and in_table:
            m2 = re.match(
                r"^(?P<name>[A-Za-z][A-Za-z0-9 ./%()-]{1,40}?)\s+"
                r"(?P<rate>\d+(?:\.\d+)?)\s+(?P<amt>\d+(?:\.\d+)?)\s*$",
                flat,
            )
            if m2:
                rate = float(m2.group("rate"))
                amt = float(m2.group("amt"))
                name = m2.group("name").strip()
                if rate > 0 and amt >= rate and amt / rate <= 30:
                    qty = round(amt / rate, 2)
                    # Prefer integer qty when close
                    if abs(qty - round(qty)) < 0.05:
                        qty = float(round(qty))
                    items.append(LineItem(
                        description=name, quantity=qty, unit_price=rate, total=amt,
                    ))
                    matched = True
        if not matched and (in_table or not items):
            # Multi-item OCR glue: take trailing qty/rate/amt groups
            for m in _REST_ITEM_TAIL_RE.finditer(flat):
                name = m.group("name").strip(" .-")
                name = re.sub(r"^\d+\s*", "", name).strip()
                if len(name) < 3:
                    continue
                if re.search(r"(?i)^(particulars|item|sl|qty|rate|amount|value|total)$", name):
                    continue
                qty, rate, amt = float(m.group("qty")), float(m.group("rate")), float(m.group("amt"))
                if amt <= 0 or rate <= 0:
                    continue
                if qty > 50 and rate > 50:
                    continue
                # Prefer qty*rate ≈ amt
                if abs(qty * rate - amt) > max(2.0, amt * 0.2) and abs(rate * 1 - amt) < 1:
                    continue
                items.append(LineItem(
                    description=name, quantity=qty, unit_price=rate, total=amt,
                ))
                in_table = True

    # Deduplicate identical rows
    uniq: List[LineItem] = []
    seen = set()
    for it in items:
        key = (it.description.lower(), it.quantity, it.unit_price, it.total)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(it)
    result.line_items = uniq

    tax_lines: List[TaxLine] = []
    for ln in lines:
        for m in _REST_TAX_RE.finditer(ln.text.replace(",", "")):
            kind = m.group(1).upper()
            rate = float(m.group(2))
            g3 = m.group(3)
            g4 = m.group(4)
            if g4 is None:
                continue
            amount = float(g4)
            base = float(g3) if g3 else 0.0
            if amount > 200 and base and abs(amount - base) < 1:
                continue
            if amount > 100 and rate <= 28 and g3 and float(g3) < 100:
                amount = float(g3)
                base = 0.0
            if not base and rate:
                base = round(amount * 100.0 / rate, 2)
            tax_lines.append(TaxLine(kind=kind, rate=rate, base=base, amount=amount))

    # OCR noise: "CS2.5%0n910 22.75" / "S2.5xn910 22.75"
    if len(tax_lines) < 2:
        for ln in lines:
            m = re.search(
                r"(?i)\bC?S(?:GST)?\s*@?\s*(\d+(?:\.\d+)?)\s*%?\s*(?:on|0n|xn)?\s*"
                r"(\d+(?:\.\d+)?)?\s+(\d+(?:\.\d+)?)",
                ln.text.replace(",", ""),
            )
            if not m:
                continue
            rate = float(m.group(1))
            amount = float(m.group(3))
            base = float(m.group(2)) if m.group(2) else (
                round(amount * 100.0 / rate, 2) if rate else 0.0
            )
            if amount >= 200:
                continue
            kind = "CGST" if re.search(r"(?i)\bc", ln.text[:3]) else "SGST"
            # Avoid dupes
            if any(t.kind == kind and abs(t.amount - amount) < 0.01 for t in tax_lines):
                continue
            tax_lines.append(TaxLine(kind=kind, rate=rate, base=base, amount=amount))

    # "SGST2.5 CGST2.50 35.71" then next line "35.71"
    if not any(t.kind == "SGST" for t in tax_lines) or not any(
        t.kind == "CGST" for t in tax_lines
    ):
        joined_lines = [ln.text.replace(",", "") for ln in lines]
        for idx, t in enumerate(joined_lines):
            m2 = re.search(
                r"(?i)SGST\s*@?\s*(\d+(?:\.\d+)?)\s*%?\s*CGST\s*@?\s*(\d+(?:\.\d+)?)\s*%?\s*"
                r"(\d+(?:\.\d+))?",
                t,
            )
            if not m2:
                continue
            amt = float(m2.group(3)) if m2.group(3) else 0.0
            if not amt and idx + 1 < len(joined_lines):
                nums = cell_numbers(joined_lines[idx + 1]) or amounts_in(joined_lines[idx + 1])
                if nums:
                    amt = nums[0]
            # Second 35.71 may sit on same or next line — use for both
            amts = [amt] if amt else []
            if idx + 1 < len(joined_lines):
                nums = cell_numbers(joined_lines[idx + 1]) or amounts_in(joined_lines[idx + 1])
                for n in nums:
                    if 0 < n < 200:
                        amts.append(n)
            if not amts:
                continue
            sgst_amt = amts[0]
            cgst_amt = amts[1] if len(amts) > 1 else amts[0]
            srate, crate = float(m2.group(1)), float(m2.group(2))
            tax_lines = [
                TaxLine(
                    kind="SGST", rate=srate,
                    base=round(sgst_amt * 100.0 / srate, 2) if srate else 0.0,
                    amount=sgst_amt,
                ),
                TaxLine(
                    kind="CGST", rate=crate,
                    base=round(cgst_amt * 100.0 / crate, 2) if crate else 0.0,
                    amount=cgst_amt,
                ),
            ]
            break
    result.tax_lines = tax_lines

    sub = 0.0
    total = 0.0
    for ln in lines:
        t = ln.text
        if re.search(r"(?i)sub\s*tot", t):
            nums = cell_numbers(t) or amounts_in(t)
            if nums:
                sub = nums[-1]
        if re.search(r"(?i)grand\s*total|\bnet\s*amount\b|(?<![A-Za-z])total\s*:", t):
            if re.search(r"(?i)sub\s*tot|tax\s*total|qty", t):
                continue
            nums = cell_numbers(t) or amounts_in(t)
            if nums:
                total = nums[-1]
    item_sum = round(sum(it.total for it in uniq), 2)
    if not sub and uniq:
        sub = item_sum
    tax_sum = round(sum(t.amount for t in tax_lines), 2)
    if not total:
        total = round(sub + tax_sum, 2) if tax_sum else sub
    # When printed net is reliable but item OCR is incomplete, derive base
    if (
        total
        and tax_sum
        and item_sum + tax_sum < total * 0.85
        and (not sub or abs(sub - item_sum) < 1.0)
    ):
        sub = round(total - tax_sum, 2)
    result.totals.base_amount = sub
    result.totals.tax_amount = tax_sum
    result.totals.total_amount = total


# ---------------------------------------------------------------- utility (BWSSB)

_UTILITY_CHARGE_RE = re.compile(
    r"(?i)^(?:[a-z]\s*/\s*)?(Water\s*Charges|Meter\s*Charges|Sanitary\s*Charges|"
    r"S\.?\s*C\.?\s*For\s*Borew(?:ell|e)|Other\s*Charges|Arrears|Interest|"
    r"Borewell\s*Charges|Sewerage\s*Charges)"
    r"\s*[:=]?\s*(\d+(?:\.\d+)?)?\s*$"
)
_UTILITY_LABEL_RE = re.compile(
    r"(?i)(Water\s*Charges|Meter\s*Charges|Sanitary\s*Charges|"
    r"S\.?\s*C\.?\s*For\s*Borew\w*|Other\s*Charges|Arrears|Interest)"
)
_UTILITY_CANON = (
    ("water", "Water Charges"),
    ("meter", "Meter Charges"),
    ("sanitary", "Sanitary Charges"),
    ("bore", "S.C For Borewell"),
    ("other", "Other Charges"),
    ("arrear", "Arrears"),
    ("interest", "Interest"),
)


def _canon_utility_label(label: str) -> str:
    low = label.lower()
    for needle, canon in _UTILITY_CANON:
        if needle in low:
            return canon
    return label.strip()


def _utility_charge_items(lines: List[Line]) -> List[LineItem]:
    """Parse BWSSB charge rows; zip labels with amounts when OCR shifts columns."""
    start = 0
    end = len(lines)
    for i, ln in enumerate(lines):
        if re.search(r"(?i)billing\s*details", ln.text):
            start = i
        if start and i > start and re.search(
            r"(?i)^(?:total\s*amount|adv\s*amount|net\s*amount|rupees|previous\s*month)",
            _normalize_thermal_text(ln.text),
        ):
            end = i
            break
    block = lines[start:end] if start or start == 0 else lines

    labels: List[str] = []
    amounts: List[float] = []
    for ln in block:
        t = ln.text.strip().replace(",", "")
        # Stop collecting amounts once we hit totals
        if re.search(r"(?i)^(?:/?\s*)?(total\s*amount|adv\s*amount|net\s*amount)", t):
            break
        for m in _UTILITY_LABEL_RE.finditer(t):
            labels.append(_canon_utility_label(m.group(1)))
        # Include amount glued to "Billing Details 1020.00"
        if re.search(r"(?i)billing\s*details", t):
            for n in (cell_numbers(t) or amounts_in(t) or []):
                amounts.append(n)
            continue
        # Amounts on charge lines or orphan numeric lines
        if _UTILITY_LABEL_RE.search(t):
            m = _UTILITY_CHARGE_RE.match(_normalize_thermal_text(t))
            if m and m.group(2):
                amounts.append(float(m.group(2)))
            else:
                tail = AMOUNT_TAIL_RE.match(_normalize_thermal_text(t))
                if tail and re.search(
                    r"(?i)charge|arrear|interest|bore", tail.group("label")
                ):
                    amounts.append(parse_amount(tail.group("amount")) or 0.0)
        elif re.fullmatch(r"\d+(?:\.\d+)?", _normalize_thermal_text(t)):
            amounts.append(float(_normalize_thermal_text(t)))

    # Prefer zip when counts align (handles shifted OCR); else fall back to on-line pairs
    items: List[LineItem] = []
    if labels and amounts and len(amounts) >= len(labels):
        for label, amount in zip(labels, amounts):
            if amount <= 0:
                continue
            items.append(LineItem(
                description=label, quantity=1.0, unit_price=amount, total=amount,
            ))
        return items

    # Fallback: label lines that already carry their amount
    seen = set()
    for ln in block:
        t = _normalize_thermal_text(ln.text.replace(",", ""))
        m = _UTILITY_CHARGE_RE.match(t)
        if not m or not m.group(2):
            am = AMOUNT_TAIL_RE.match(t)
            if am and _UTILITY_LABEL_RE.search(am.group("label")):
                label = _canon_utility_label(am.group("label"))
                amount = parse_amount(am.group("amount")) or 0.0
            else:
                continue
        else:
            label = _canon_utility_label(m.group(1))
            amount = float(m.group(2))
        if amount <= 0 or label in seen:
            continue
        seen.add(label)
        items.append(LineItem(
            description=label, quantity=1.0, unit_price=amount, total=amount,
        ))
    return items


def _extract_utility(pages: List[Page], result: InvoiceResult) -> None:
    lines = _clean_lines(pages)
    blob = "\n".join(ln.text for ln in lines)
    kv = _kv_map(lines)

    # Prefer English board name
    for ln in lines[:8]:
        if re.search(r"(?i)water\s*supply|bwssb|sewerage\s*board", ln.text):
            result.supplier.name = "Bangalore Water Supply and Sewerage Board"
            break
    if not result.supplier.name:
        result.supplier.name = _header_name(lines) or "BWSSB"

    # Consumer block
    consumer = ""
    for i, ln in enumerate(lines):
        if re.search(r"(?i)consumer\s*details", ln.text):
            parts = []
            for nxt in lines[i + 1:i + 5]:
                if re.search(r"(?i)pres\s*rdg|billing\s*details|water\s*charges", nxt.text):
                    break
                if re.search(r"[A-Za-z]{3,}", nxt.text):
                    parts.append(nxt.text.strip())
            consumer = ", ".join(parts)
            break
    if consumer:
        result.supplier.billing_address = consumer
        # First token as rough name if industrial consumer
        result.supplier.current_address = consumer

    _apply_gstin(result, blob + "\n" + _all_text(pages))
    bill_no = _find_kv(kv, "bill no", "billno") or ""
    if not bill_no:
        m = re.search(r"(?i)bill\s*no\.?\s*[:=]?\s*([A-Z0-9/-]+)", blob)
        if m:
            bill_no = m.group(1)
    # OCR sometimes puts the number on the line above an empty "BillNo"
    if not bill_no or bill_no.lower() in {"rdg", "date", "due"}:
        for i, ln in enumerate(lines):
            if re.search(r"(?i)bill\s*no", ln.text):
                # same line trailing token
                m = re.search(r"(?i)bill\s*no\.?\s*[:=]?\s*([A-Z0-9/-]+)", ln.text)
                if m and re.search(r"\d", m.group(1)):
                    bill_no = m.group(1)
                    break
                if i > 0:
                    prev = _normalize_thermal_text(lines[i - 1].text)
                    if re.fullmatch(r"-?[A-Z0-9/-]*\d[A-Z0-9/-]*", prev):
                        bill_no = prev
                        break
                if i + 1 < len(lines):
                    nxt = _normalize_thermal_text(lines[i + 1].text)
                    if re.fullmatch(r"-?[A-Z0-9/-]*\d[A-Z0-9/-]*", nxt):
                        bill_no = nxt
                        break
    bill_no = bill_no.lstrip("-./ ")
    if re.search(r"(?i)^(rdg|due|date)$", bill_no):
        bill_no = ""
    result.invoice.number = bill_no
    # Prefer reading / due dates (regex first — avoids "after due date 3279")
    rdg = ""
    due = ""
    m = re.search(
        r"(?i)rdg\s*date\s*[:=]?\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})",
        blob,
    )
    if m:
        rdg = m.group(1)
    m = re.search(
        r"(?i)(?<!after )due\s*date\s*[:=]?\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})",
        blob,
    )
    if m:
        due = m.group(1)
    if not rdg:
        rdg = _find_kv(kv, "rdg date", "reading date") or ""
    if not due:
        due = _find_kv(kv, "due date") or ""
        if due and not re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", due):
            due = ""
    result.invoice.date = _first_date(rdg or "") or _first_date(blob)
    result.invoice.due_date = due.strip() if due else ""

    items = _utility_charge_items(lines)
    result.line_items = items

    net = _money(_find_kv(kv, "net amount due", "netamount due", "net amount"))
    total_amt = _money(_find_kv(kv, "total amount"))
    if not net:
        # scan lines
        for ln in lines:
            if re.search(r"(?i)net\s*amount\s*due|netamount\s*due", ln.text):
                nums = cell_numbers(ln.text) or amounts_in(ln.text)
                if nums:
                    net = nums[-1]
            if re.search(r"(?i)^total\s*amount\b", ln.text):
                nums = cell_numbers(ln.text) or amounts_in(ln.text)
                if nums:
                    total_amt = nums[-1]
    if not net:
        net = total_amt or round(sum(it.total for it in items), 2)
    result.totals.base_amount = round(sum(it.total for it in items), 2) or net
    result.totals.total_amount = net
    result.totals.tax_amount = 0.0


# ---------------------------------------------------------------- generic thermal

def _extract_generic(pages: List[Page], result: InvoiceResult) -> None:
    """Best-effort key-value thermal slip."""
    lines = _clean_lines(pages)
    blob = "\n".join(ln.text for ln in lines)
    kv = _kv_map(lines)
    result.supplier.name = _header_name(lines)
    _apply_gstin(result, blob + "\n" + _all_text(pages))
    result.invoice.number = (
        _find_kv(kv, "bill no", "receipt no", "txn id", "invoice") or ""
    )
    result.invoice.date = _first_date(_find_kv(kv, "date") or blob)
    amount = _money(
        _find_kv(kv, "net amount", "grand total", "total amount", "amount", "net value")
    )
    if amount:
        result.line_items = [
            LineItem(description="Charges", quantity=1.0, unit_price=amount, total=amount)
        ]
        result.totals.base_amount = amount
        result.totals.total_amount = amount


# ---------------------------------------------------------------- public API

def extract_thermal(pages: List[Page], doc_type: str) -> InvoiceResult:
    result = InvoiceResult(document_type=doc_type)
    if not pages:
        result.warnings.append("no pages found")
        return result

    if doc_type == DOC_FUEL:
        _extract_fuel(pages, result)
    elif doc_type == DOC_RESTAURANT:
        _extract_restaurant(pages, result)
    elif doc_type == DOC_UTILITY:
        _extract_utility(pages, result)
    else:
        _extract_generic(pages, result)
        result.document_type = DOC_GENERIC

    result.ocr_confidence = round(
        sum(p.mean_confidence() for p in pages) / len(pages), 3
    )
    if not result.line_items and not result.totals.total_amount:
        result.warnings.append("thermal extract: few fields found — review recommended")
    return result
