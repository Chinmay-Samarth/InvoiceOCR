"""Field extraction from OCR output for Indian GST tax invoices.

The extractors are anchor/pattern based so they work across invoice layouts:
- hard patterns (GSTIN, PAN, IFSC, IRN, email, phone, dates) are found by regex
- supplier vs buyer is decided by position relative to "Billed To"-style anchors
- totals and tax lines are found via keyword anchors + amounts, then reconciled
  against each other (base + tax = total) to reject OCR column-bleed values
- line items use the item-table header to derive column x-ranges
"""
import re
from typing import List, Optional

from .models import (
    InvoiceResult, LineItem, TaxLine,
)
from .ocr_engine import Line, Page, Word

# ---------------------------------------------------------------- patterns

GSTIN_RE = re.compile(r"\b(\d{2}[A-Z]{5}\d{4}[A-Z][0-9A-Z][Z2][0-9A-Z])\b")
PAN_RE = re.compile(r"\b([A-Z]{5}\d{4}[A-Z])\b")
IFSC_RE = re.compile(r"\b([A-Z]{4}0[A-Z0-9]{6})\b")
EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_RE = re.compile(r"(?<!\d)(?:\+91[\s-]?)?(0?\d{10}|0\d{2,4}[\s-]\d{6,8})(?!\d)")
AMOUNT_RE = re.compile(r"(?<![\d.])(\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+\.\d{1,2})(?!\d)")
DATE_RE = re.compile(
    r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}"
    r"|\d{1,2}[\s./-]?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s./,-]{1,2}\d{2,4})\b",
    re.IGNORECASE,
)

STATE_CODES = {
    "01": "Jammu and Kashmir", "02": "Himachal Pradesh", "03": "Punjab",
    "04": "Chandigarh", "05": "Uttarakhand", "06": "Haryana", "07": "Delhi",
    "08": "Rajasthan", "09": "Uttar Pradesh", "10": "Bihar", "11": "Sikkim",
    "12": "Arunachal Pradesh", "13": "Nagaland", "14": "Manipur",
    "15": "Mizoram", "16": "Tripura", "17": "Meghalaya", "18": "Assam",
    "19": "West Bengal", "20": "Jharkhand", "21": "Odisha",
    "22": "Chhattisgarh", "23": "Madhya Pradesh", "24": "Gujarat",
    "26": "Dadra and Nagar Haveli and Daman and Diu", "27": "Maharashtra",
    "29": "Karnataka", "30": "Goa", "31": "Lakshadweep", "32": "Kerala",
    "33": "Tamil Nadu", "34": "Puducherry", "35": "Andaman and Nicobar Islands",
    "36": "Telangana", "37": "Andhra Pradesh", "38": "Ladakh",
}

BUYER_ANCHOR_RE = re.compile(
    r"billed\s*to|bill\s*to|receiver|consignee|ship(?:ped)?\s*to|buyer|customer\s*name",
    re.IGNORECASE,
)
# "Original for Buyer/Recipient" banners must not count as buyer anchors
BUYER_ANCHOR_EXCLUDE_RE = re.compile(
    r"original|duplicate|triplicate|for\s+(buyer|recipient)", re.IGNORECASE
)

DOC_NOISE_RE = re.compile(
    r"tax\s+invoice|e-?invoice|original\s+for|duplicate|triplicate|"
    r"\(original\b|proforma",
    re.IGNORECASE,
)


def parse_amount(s: str) -> Optional[float]:
    s = s.replace(",", "").replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return None


def amounts_in(text: str) -> List[float]:
    vals = [a for a in (parse_amount(m) for m in AMOUNT_RE.findall(text)) if a is not None]
    return vals


def _whole_amounts(text: str, min_val: float = 100.0) -> List[float]:
    """Integers without decimals (OCR often drops .00), e.g. 3222, 10240."""
    out = []
    for m in re.finditer(r"(?<![\d.])(\d{3,})(?!\d)", text):
        v = float(m.group(1))
        if v >= min_val:
            out.append(v)
    return out


def _clean_value(s: str) -> str:
    return s.strip().strip(":;|-").strip()


# ---------------------------------------------------------------- helpers

def _all_lines(pages: List[Page]) -> List[Line]:
    lines = []
    for p in pages:
        lines.extend(p.lines)
    return lines


def _find_line(lines: List[Line], pattern: re.Pattern) -> Optional[int]:
    for i, ln in enumerate(lines):
        if pattern.search(ln.text):
            return i
    return None


def _left_text(ln: Line, page: Page, frac: float = 0.5) -> str:
    """Text of the line using only words in the left part of the page
    (drops right-column labels / logos that OCR merged into the same line)."""
    cut = page.width * frac
    return " ".join(w.text for w in ln.words if w.x0 < cut).strip()


# ---------------------------------------------------------------- supplier

HEADER_SKIP_RE = re.compile(
    r"GSTIN|GST\s*No|Phone|Tel\b|Mobile|E-?mail|CIN\b|IATF|ISO\s*\d|PAN\b|"
    r"Invoice|State\s*code|Udyam|\bIRN\b|\bAck\b|website|www\.|Delivery|"
    r"Reference|Order|Dispatch|customer\s*care|emergency|toll\s*free|"
    r"\bcode\s*[:.]?\s*\d{1,2}\b|e-?way|\bdate\b|\btime\b|\bdated\b",
    re.IGNORECASE,
)


def _is_hexish(t: str) -> bool:
    compact = t.replace(" ", "").replace("-", "").lower()
    return len(compact) >= 16 and re.fullmatch(r"[0-9a-f]+", compact) is not None


def extract_supplier(pages: List[Page], result: InvoiceResult) -> None:
    page = pages[0]
    lines = page.lines
    sup = result.supplier

    buyer_ys = [
        ln.cy for ln in lines
        if BUYER_ANCHOR_RE.search(ln.text)
        and not BUYER_ANCHOR_EXCLUDE_RE.search(ln.text)
    ]
    first_buyer_y = min(buyer_ys) if buyer_ys else page.height * 0.35

    header_lines = [ln for ln in lines if ln.cy < first_buyer_y]
    body_text = "\n".join(ln.text for ln in lines)
    header_text = "\n".join(ln.text for ln in header_lines)

    # --- GSTIN: supplier GSTIN appears in the header, above the buyer block
    all_gstins = GSTIN_RE.findall(body_text.upper())
    header_gstins = GSTIN_RE.findall(header_text.upper())
    if header_gstins:
        sup.gstin = header_gstins[0]
    elif all_gstins:
        # fall back: the buyer GSTIN usually repeats (billed-to + shipped-to);
        # prefer a GSTIN that appears only once
        from collections import Counter
        counts = Counter(all_gstins)
        singles = [g for g in all_gstins if counts[g] == 1]
        sup.gstin = singles[0] if singles else all_gstins[0]

    # --- PAN: derived from GSTIN chars 3..12, or an explicit PAN label
    if sup.gstin:
        sup.pan = sup.gstin[2:12]
    else:
        for ln in lines:
            if re.search(r"\bPAN\b", ln.text, re.IGNORECASE):
                m = PAN_RE.search(ln.text.upper())
                if m:
                    sup.pan = m.group(1)
                    break

    # --- name: first meaningful left-side header line
    name_idx = None
    for i, ln in enumerate(header_lines):
        t = _left_text(ln, page)
        if len(t) < 5 or DOC_NOISE_RE.search(t) or _is_hexish(t):
            continue
        if re.search(r"\bIRN\b|Ack\s*No", t, re.IGNORECASE):
            continue
        if SUPPLIER_NAME_SKIP_RE.search(t):
            continue
        if re.fullmatch(r"[\d\s/|.,-]+", t):
            continue
        if sum(c.isalpha() for c in t) < 5:
            continue
        sup.name = _clean_value(re.split(
            r"\bGSTIN\b|\bGST\b|\bInvoice\b", t, flags=re.IGNORECASE)[0])
        name_idx = i
        break

    # --- address: left-side header lines after the name, until label lines
    addr_parts: List[str] = []
    if name_idx is not None:
        skips = 0
        for ln in header_lines[name_idx + 1:]:
            t = _left_text(ln, page)
            if (not t or HEADER_SKIP_RE.search(t) or DOC_NOISE_RE.search(t)
                    or _is_hexish(t) or sum(c.isalnum() for c in t) < 5):
                skips += 1
                if skips >= 2 and addr_parts:
                    break
                continue
            skips = 0
            addr_parts.append(_clean_value(t))
            if len(addr_parts) >= 4:
                break
    address = ", ".join(addr_parts)
    # If OCR merged the company name into the address block, promote it.
    if addr_parts and (
        not sup.name
        or re.search(r"\bIRN\b|Ack\s*No", sup.name, re.IGNORECASE)
        or sum(c.isalpha() for c in sup.name) < 5
    ):
        first = addr_parts[0]
        if sum(c.isalpha() for c in first) >= 5 and not re.search(
            r"^\d|no\.?\s*\d|gstin|fac\.|corporate|office|manufacturing",
            first, re.IGNORECASE,
        ):
            sup.name = first
            addr_parts = addr_parts[1:]
            address = ", ".join(addr_parts)
    sup.current_address = address
    sup.billing_address = address

    # --- state from the GSTIN state code
    if sup.gstin[:2] in STATE_CODES:
        sup.billing_state = STATE_CODES[sup.gstin[:2]]

    # --- phone / email: header region only (body would leak buyer contacts)
    for ln in header_lines:
        if sup.phone:
            break
        if re.search(r"phone|tel\b|mob|contact|cell", ln.text, re.IGNORECASE):
            m = PHONE_RE.search(ln.text)
            if m:
                sup.phone = m.group(1).strip()
    if not sup.phone:
        m = PHONE_RE.search(header_text)
        if m:
            sup.phone = m.group(1).strip()
    m = EMAIL_RE.search(header_text)
    if m:
        sup.email = m.group(0)

    # --- contact person: explicit label within the supplier header only
    for ln in header_lines:
        m = re.search(r"contact\s*person\s*[:.-]?\s*(.+)", ln.text, re.IGNORECASE)
        if m:
            val = _clean_value(m.group(1))
            if (val and sum(c.isalpha() for c in val) >= 3
                    and not re.search(r"ref|p\.?o\.?|date|dispatch|mode", val, re.IGNORECASE)):
                sup.contact_person = val
                break


# ---------------------------------------------------------------- bank

def extract_bank(pages: List[Page], result: InvoiceResult) -> None:
    lines = _all_lines(pages)
    bank = result.bank
    full_text = "\n".join(ln.text for ln in lines).upper()

    m = IFSC_RE.search(full_text.replace("IFSC", " ").replace("IFS CODE", " "))
    if m:
        bank.ifsc = m.group(1)

    # account number: 9-18 digit number in a small window after a bank label
    for i, ln in enumerate(lines):
        if not re.search(r"\bbank\b|a/?c\s*no|account\s*no", ln.text, re.IGNORECASE):
            continue
        window = " ".join(l.text for l in lines[i:i + 3])
        window = window.replace(bank.ifsc, " ") if bank.ifsc else window
        nums = re.findall(r"\b\d{9,18}\b", window)
        if nums:
            bank.account_number = nums[0]
            break

    if re.search(r"a/?c\s*type\s*[:\s]*current|current\s*a/?c", full_text, re.IGNORECASE):
        bank.account_type = "Current"
    elif re.search(r"a/?c\s*type\s*[:\s]*savings|savings\s*a/?c", full_text, re.IGNORECASE):
        bank.account_type = "Savings"


# ---------------------------------------------------------------- invoice

INV_NUM_BLACKLIST = {"e-way", "eway", "bill", "dated", "date", "no", "and",
                     "value", "way", "state", "code", "party", "gstin"}

SUPPLIER_NAME_SKIP_RE = re.compile(
    r"^Ack\s*Date|E-Way\s*Bill|Bill\s*Date|Original|Duplicate|Recipient",
    re.IGNORECASE,
)

META_ROW_RE = re.compile(
    r"your\s*dc|dcno|dc\s*no|po\s*no|narration|being\s|ref\s*date|logistics|"
    r"shipping\s*company|declaration|terms\s*&|bank\s*detail|in\s*words|"
    r"particulars\s*are\s*true|e&oe",
    re.IGNORECASE,
)


def _standalone_tokens(lines: List[Line]) -> List[str]:
    tokens = []
    for ln in lines:
        for w in ln.words:
            for tok in re.split(r"\s+", w.text):
                tok = tok.strip(".,:;()")
                if len(tok) >= 4 and re.search(r"\d", tok):
                    tokens.append(tok)
    return tokens


def extract_invoice(pages: List[Page], result: InvoiceResult) -> None:
    page = pages[0]
    lines = _all_lines(pages)
    inv = result.invoice
    full_text = "\n".join(ln.text for ln in lines)

    # --- IRN: 64 hex chars, may wrap across the lines after the "IRN" label
    irn_idx = _find_line(lines, re.compile(r"\bIRN\b", re.IGNORECASE))
    if irn_idx is not None:
        blob = " ".join(ln.text for ln in lines[irn_idx:irn_idx + 3])
        blob = re.sub(r"\bIRN\b", " ", blob, flags=re.IGNORECASE)
        hex_chars = re.sub(r"[^0-9a-fA-F]", "", blob)
        if len(hex_chars) >= 64:
            inv.irn = hex_chars[:64].lower()

    # --- invoice number: value on the label line, else the word right below
    label_re = re.compile(
        r"INVOICE\.?\s*No\.?|invoice\s*no\.?|invoice\s*#|inv\s*no\.?|bill\s*no\.?",
        re.IGNORECASE,
    )

    def _candidate_from_tail(tail: str) -> Optional[str]:
        tail = re.sub(r"^\s*[.:|&-]+", " ", tail)
        m = re.search(r"([A-Za-z0-9][A-Za-z0-9/_-]{2,})", tail)
        if m:
            cand = m.group(1).strip(".:")
            if cand.lower() not in INV_NUM_BLACKLIST and not re.fullmatch(r"[a-zA-Z-]+", cand):
                return cand
        m = re.search(r"\b(\d{2,6})\b", tail)
        if m:
            return m.group(1)
        return None

    for ln in lines:
        if label_re.search(ln.text):
            cand = _candidate_from_tail(label_re.split(ln.text)[-1])
            if cand:
                inv.number = cand
                break

    if not inv.number:
        # find the label word, then the value word directly below it
        label_word: Optional[Word] = None
        for ln in page.lines:
            for w in ln.words:
                if label_re.search(w.text):
                    label_word = w
                    break
            if label_word:
                break
        if label_word:
            below = [
                w for ln in page.lines for w in ln.words
                if label_word.y1 - 5 < w.y0 < label_word.y1 + 4 * label_word.height
                and abs(w.cx - label_word.cx) < page.width * 0.15
                and re.search(r"\d", w.text)
            ]
            if below:
                below.sort(key=lambda w: w.y0)
                blob = below[0].text.strip(" .:")
                # the value may be OCR-merged with neighbours; if a shorter
                # standalone token elsewhere is a prefix of it, prefer that
                best = ""
                for tok in _standalone_tokens(lines):
                    if tok != blob and blob.startswith(tok) and len(tok) > len(best):
                        best = tok
                inv.number = best or blob.split()[0]

    # --- invoice date: prefer an explicitly labelled date line
    date_label_re = re.compile(
        r"(date\s*of\s*issue|invoice\s*date|bill\s*date|\bdated\b|\bdate\b)",
        re.IGNORECASE,
    )
    for ln in lines:
        if date_label_re.search(ln.text) and not re.search(
            r"due|removal|ack|delivery|dispatch|l\.?r\.?\s*no|order|p\.?o\b",
            ln.text, re.IGNORECASE,
        ):
            ds = DATE_RE.findall(ln.text)
            if ds:
                inv.date = ds[0]
                break
    if not inv.date:
        ds = DATE_RE.findall(full_text)
        if ds:
            inv.date = ds[0]

    # --- due date
    for ln in lines:
        if re.search(r"due\s*(date|on)|payment\s*due", ln.text, re.IGNORECASE):
            ds = DATE_RE.findall(ln.text)
            if ds:
                inv.due_date = ds[0]
                break
    if not inv.due_date:
        idx = _find_line(lines, re.compile(r"payment\s*due", re.IGNORECASE))
        if idx is not None:
            for ln in lines[idx + 1: idx + 3]:
                ds = DATE_RE.findall(ln.text)
                if ds:
                    inv.due_date = ds[0]
                    break


# ---------------------------------------------------------------- line items

# header column tokens: primary pattern first, fallback second
HEADER_TOKENS = {
    "description": [
        r"description|particulars|product\s*name|item\s*description|drawing",
        r"goods|item",
    ],
    "hsn": [r"hsn|sac", None],
    "qty": [r"qty|quantity", r"pkgs|\buom\b"],
    "rate": [r"\brate\b|uomrate|uom\s*rate", r"price"],
    "amount": [r"amount|taxable", r"value"],
}


def _match_header_words(words: List[Word]) -> Optional[dict]:
    found: dict = {}
    for rank in (0, 1):
        for w in words:
            if any(v is w for v in found.values()):
                continue
            for col, pats in HEADER_TOKENS.items():
                pat = pats[rank]
                if col in found or pat is None:
                    continue
                if re.search(pat, w.text, re.IGNORECASE):
                    found[col] = w
                    break
    if "description" in found and ("amount" in found or "rate" in found) and len(found) >= 3:
        return found
    # "Product" + "Name" on separate tokens
    prod = next((w for w in words if re.fullmatch(r"product", w.text, re.I)), None)
    name_w = next((w for w in words if re.fullmatch(r"name\.?", w.text, re.I)), None)
    if prod and name_w and "description" not in found:
        found["description"] = prod
    if "description" in found and ("amount" in found or "rate" in found) and len(found) >= 3:
        return found
    return None


def _find_table_header(lines: List[Line]) -> tuple:
    """Find item table header; optionally merge one continuation header line."""
    best_h, best_idx, best_score = None, None, 0
    for i, ln in enumerate(lines):
        for span in (1, 2):
            chunk = lines[i:i + span]
            if len(chunk) > 1 and chunk[-1].cy - chunk[0].cy > 60:
                break
            words: List[Word] = []
            for row in chunk:
                words.extend(row.words)
            h = _match_header_words(words)
            if not h:
                continue
            score = len(h)
            if re.search(
                r"description|particulars|product\s*name|item\s*description",
                chunk[0].text, re.IGNORECASE,
            ):
                score += 3
            if score > best_score:
                best_score = score
                best_h, best_idx = h, i + span - 1
    return best_h, best_idx


def _match_header(ln: Line) -> Optional[dict]:
    return _match_header_words(ln.words)


def _is_table_end(ln: Line) -> bool:
    """True when the line-item table ends (summary / tax section begins)."""
    t = ln.text
    if re.search(
        r"grand\s*total|invoice\s*total|taxable\s*amount|taxable\s*value|"
        r"amount\s*chargeable|amount\s*before\s*tax|\bcgst\b|\bsgst\b|\bigst\b|"
        r"\brcm\b|in\s*words?|declaration|precaution",
        t, re.IGNORECASE,
    ):
        return True
    # Pre-tax subtotal row: "Total" with an amount, not a quantity summary
    if re.search(r"\btotal\b", t, re.IGNORECASE) and not re.search(
        r"grand|taxable|invoice|chargeable|before\s*tax|"
        r"\b(kgs?|nos|pcs|units?|bundles?|unt)\b",
        t, re.IGNORECASE,
    ):
        if amounts_in(t):
            return True
    return False


def extract_line_items(pages: List[Page], result: InvoiceResult) -> None:
    page = pages[0]
    lines = page.lines

    header, header_idx = _find_table_header(lines)
    if header is None:
        return

    col_x = {c: w.cx for c, w in header.items()}

    def nearest_col(w: Word) -> str:
        best, best_d = "description", float("inf")
        for name, x in col_x.items():
            d = abs(w.cx - x)
            if d < best_d:
                best, best_d = name, d
        return best

    max_gap = max(page.height * 0.08, 400.0)
    prev_y = lines[header_idx].cy
    items: List[LineItem] = []
    current: Optional[LineItem] = None

    for ln in lines[header_idx + 1:]:
        if _is_table_end(ln):
            break
        if ln.cy - prev_y > max_gap:
            break
        # skip handwriting / stamp noise (low recognition confidence)
        confs = [w.conf for w in ln.words]
        if confs and sum(confs) / len(confs) < 0.6:
            continue

        cells = {c: [] for c in list(col_x) + ["description"]}
        for w in ln.words:
            cells[nearest_col(w)].append(w)

        desc_text = " ".join(w.text for w in cells["description"]).strip()
        qty_v = amounts_in(" ".join(w.text for w in cells.get("qty", [])))
        rate_v = amounts_in(" ".join(w.text for w in cells.get("rate", [])))
        amt_words = cells.get("amount", [])
        if amt_words and "amount" in col_x:
            ax = col_x["amount"]
            margin = page.width * 0.07
            amt_words = [w for w in amt_words if abs(w.cx - ax) <= margin]
        amt_v = amounts_in(" ".join(w.text for w in amt_words))
        row_amounts = amounts_in(ln.text)

        qty = qty_v[0] if qty_v else 0.0
        if not qty:
            qty_text = " ".join(w.text for w in cells.get("qty", []))
            m = re.search(
                r"(\d+(?:\.\d+)?)\s*(?:kgs?|nos|unt|pcs|units?|bundles?|box)\b",
                qty_text, re.IGNORECASE,
            )
            if m:
                qty = float(m.group(1))
        price = rate_v[0] if rate_v else 0.0
        total = amt_v[-1] if amt_v else 0.0

        # column-merge fallbacks
        if not total and len(rate_v) >= 2:
            price, total = rate_v[0], rate_v[-1]
        if not total and not price and len(row_amounts) >= 2 and desc_text:
            total = row_amounts[-1]
            if row_amounts[-2] < total:
                price = row_amounts[-2]

        if total or price:
            desc_text = re.sub(r"^\s*\d{1,2}\s+", "", desc_text)
            if META_ROW_RE.search(desc_text) and total < 100:
                continue
            if total == 0 and price == 0:
                continue
            if re.search(r"^your\s*dc|narration\b", desc_text, re.IGNORECASE):
                continue
            if not qty and price and total:
                q = total / price
                if 0 < q < 10_000_000 and abs(q - round(q, 3)) < 1e-6:
                    qty = round(q, 3)
            current = LineItem(description=desc_text, quantity=qty,
                               unit_price=price, total=total)
            items.append(current)
            prev_y = ln.cy
        elif desc_text and current is not None and len(desc_text) > 2:
            current.description = (current.description + " " + desc_text).strip()
            prev_y = ln.cy
        elif desc_text and current is None and len(desc_text) > 4:
            current = LineItem(description=re.sub(r"^\s*\d{1,2}\s+", "", desc_text))
            items.append(current)
            prev_y = ln.cy

    items = [
        it for it in items
        if not re.search(r"\b(CGST|SGST|IGST|OUTPUT|TOTAL)\b", it.description,
                         re.IGNORECASE)
    ]
    result.line_items = [it for it in items if it.description or it.total]


# ---------------------------------------------------------------- totals & tax

TAX_LINE_RE = re.compile(
    r"(IGST|CGST|SGST|UTGST)\b[^%\n]{0,25}?(\d{1,2}(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)
# Rate and amount glued together, e.g. "18.0%126.00" (common in HSN tax summary tables)
PCT_GLUE_RE = re.compile(
    r"(\d{1,2}(?:\.\d+)?)\s*%(\d+(?:\.\d+)?)"
)
UNLABELED_TAX_RE = re.compile(
    r"^\s*@?\s*(\d{1,2}(?:\.\d+)?)\s*%\s*(\d[\d,]*\.\d{1,2})\s*$"
)

# Subtotal before tax — handled by _find_subtotal()
BASE_ANCHORS = [
    r"taxable\s*value\s*total",
    r"total\s*taxable",
]
TOTAL_ANCHORS = [
    r"grand\s*total", r"bill\s*total", r"invoice\s*total",
    r"total\s*invoice\s*value", r"total\s*amount(?!\s*before)",
    r"net\s*amount\s*payable", r"amount\s*payable", r"amount\s*chargeable",
]


def _amount_on_line(ln: Line) -> Optional[float]:
    best = None
    for w in ln.words:
        vals = amounts_in(w.text)
        if vals:
            best = vals[-1]
    return best


def _find_subtotal(lines: List[Line]) -> Optional[float]:
    """Find the pre-tax subtotal row (Total 825.00), not Grand Total."""
    for ln in lines:
        if not re.search(r"\btotal\b", ln.text, re.IGNORECASE):
            continue
        if re.search(
            r"grand|taxable\s*amount|invoice\s*total|chargeable|before\s*tax|"
            r"\b(kgs?|nos|pcs|units?|bundles?|unt)\b",
            ln.text, re.IGNORECASE,
        ):
            continue
        amt = _amount_on_line(ln)
        if amt is not None and amt > 0:
            return amt
    for pat in (r"sub\s*total", r"total\s*amount\s*before\s*tax"):
        rx = re.compile(pat, re.IGNORECASE)
        for ln in lines:
            if rx.search(ln.text):
                amt = _amount_on_line(ln)
                if amt is not None and amt > 0:
                    return amt
    return None


def _extract_label_tax_lines(lines: List[Line]) -> List[TaxLine]:
    """CGST/SGST rows, Add CGST %, CGST Tax @9%, split rupee/paise OCR."""
    out: List[TaxLine] = []
    seen = set()
    for ln in lines:
        t = ln.text
        for kind in ("CGST", "SGST", "IGST", "UTGST"):
            if not re.search(rf"\b{kind}\b", t, re.IGNORECASE):
                continue
            rate = None
            amount = None
            m = re.search(
                rf"{kind}\b.*?(\d+(?:\.\d+)?)\s*%\s*(\d+(?:\.\d+)?)(?:\s+(\d{{1,2}}))?",
                t, re.IGNORECASE,
            )
            if m:
                rate = float(m.group(1))
                amount = float(m.group(2))
                if m.group(3):
                    amount = round(amount + float(m.group(3)) / 100.0, 2)
            if amount is None:
                m = re.search(rf"Add\s*{kind}\s*%?\s*(\d+(?:\.\d+)?)?", t, re.IGNORECASE)
                if m:
                    nums = _whole_amounts(t) + [n for n in amounts_in(t) if n > 28]
                    rate = float(m.group(1)) if m.group(1) else None
                    if rate is None:
                        r2 = re.search(r"(\d+(?:\.\d+)?)\s*%", t)
                        rate = float(r2.group(1)) if r2 else None
                    if nums:
                        amount = max(nums)
            if amount is None:
                m = re.search(rf"{kind}\s*Tax\s*@?\s*(\d+(?:\.\d+)?)\s*%", t, re.IGNORECASE)
                if m:
                    rate = float(m.group(1))
                    nums = [n for n in amounts_in(t) if abs(n - rate) > 0.01]
                    amount = nums[-1] if nums else None
            if rate and amount and 0 < rate <= 28 and amount > 0:
                key = (kind, rate, round(amount, 2))
                if key not in seen:
                    seen.add(key)
                    out.append(TaxLine(
                        rate=rate, amount=round(amount, 2),
                        base=round(amount * 100.0 / rate, 2),
                    ))
            break
    return out


def _extract_hsn_tax_lines(lines: List[Line]) -> List[TaxLine]:
    """Parse HSN summary blocks: HSN | Taxable Amount | IGST 18.0%126.00."""
    header_idx = None
    tax_kind = "GST"
    for i, ln in enumerate(lines):
        if re.search(r"taxable\s*amount", ln.text, re.IGNORECASE) and re.search(
            r"IGST|CGST|SGST|UTGST|HSN", ln.text, re.IGNORECASE
        ):
            header_idx = i
            if re.search(r"IGST", ln.text, re.IGNORECASE):
                tax_kind = "IGST"
            elif re.search(r"CGST", ln.text, re.IGNORECASE):
                tax_kind = "CGST"
            elif re.search(r"SGST", ln.text, re.IGNORECASE):
                tax_kind = "SGST"
            break
    if header_idx is None:
        return []

    tax_lines: List[TaxLine] = []
    for ln in lines[header_idx + 1:]:
        if re.search(r"GST\s*No|bank\s*name|CIN\b|interest\s*@", ln.text, re.IGNORECASE):
            break
        glued = PCT_GLUE_RE.findall(ln.text)
        if not glued:
            continue
        nums = amounts_in(ln.text)
        taxable = None
        for n in nums:
            if n > 28:
                taxable = n
                break
        for rate_s, amt_s in glued:
            rate = float(rate_s)
            amount = parse_amount(amt_s) or 0.0
            if 0 < rate <= 28 and amount > 0:
                base = taxable if taxable else round(amount * 100.0 / rate, 2)
                tax_lines.append(TaxLine(rate=rate, base=base, amount=amount))
        _ = tax_kind
    return tax_lines


def extract_totals_and_tax(pages: List[Page], result: InvoiceResult) -> None:
    lines = _all_lines(pages)
    totals = result.totals

    def find_anchor_amount(anchors: List[str], min_value: float = 0.0) -> Optional[float]:
        for pat in anchors:
            rx = re.compile(pat, re.IGNORECASE)
            for i, ln in enumerate(lines):
                if not rx.search(ln.text):
                    continue
                for cand_ln in (ln,) + tuple(lines[i + 1:i + 3]):
                    amt = _amount_on_line(cand_ln)
                    if amt is not None and amt > min_value:
                        return amt
        return None

    base = _find_subtotal(lines)
    if base is None:
        base = find_anchor_amount(BASE_ANCHORS)
    if result.line_items:
        item_sum = sum(it.total for it in result.line_items)
        if item_sum > 0 and (
            base is None or abs(item_sum - base) / item_sum > 0.05
        ):
            base = item_sum
    base_anchored = base is not None

    total = find_anchor_amount(TOTAL_ANCHORS, min_value=(base or 0.0) * 0.5)
    total_anchored = total is not None

    # --- collect tax line candidates: (kind, rate, amount, optional base)
    candidates = []
    for i, ln in enumerate(lines):
        m = TAX_LINE_RE.search(ln.text)
        if m:
            rate = float(m.group(2))
            if 0 < rate <= 28:
                nums = [a for a in amounts_in(ln.text) if abs(a - rate) > 1e-9]
                candidates.append([m.group(1).upper(), rate, nums[-1] if nums else 0.0, 0.0])
            continue
        m = UNLABELED_TAX_RE.match(ln.text.replace(" ", ""))
        if m:
            rate = float(m.group(1))
            amt = parse_amount(m.group(2)) or 0.0
            if 0 < rate <= 28 and amt > 0:
                candidates.append([f"UNK{i}", rate, amt, 0.0])
            continue
        if re.search(r"Add\s*(?:CGST|SGST|IGST)\s*%", ln.text, re.IGNORECASE):
            nums = _whole_amounts(ln.text) + amounts_in(ln.text)
            small = [n for n in nums if n <= 28]
            big = [n for n in nums if n > 28]
            if big:
                rate = small[-1] if small else 0.0
                candidates.append(["ADD", rate, big[-1], 0.0])
            continue
        for rate_s, amt_s in PCT_GLUE_RE.findall(ln.text):
            rate = float(rate_s)
            amt = parse_amount(amt_s) or 0.0
            if 0 < rate <= 28 and amt > 0:
                nums = [a for a in amounts_in(ln.text) if a > 28]
                base_val = nums[0] if nums else 0.0
                candidates.append([f"PCT{i}", rate, amt, base_val])

    # reject implausible amounts (OCR column bleed): at GST rates <= 28%,
    # tax can never exceed ~22% of the invoice total
    ref_total = total if total else 0.0
    if ref_total:
        for c in candidates:
            if c[2] > 0.28 * ref_total:
                c[2] = 0.0

    # de-duplicate identical (kind, rate, amount) rows
    seen = set()
    uniq = []
    for c in candidates:
        key = (c[0], c[1], round(c[2], 2))
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    candidates = uniq

    tax_sum = sum(c[2] for c in candidates)

    # --- infer missing tax amounts from rate * base when we know the target
    if base_anchored and total_anchored and base:
        expected = total - base
        if tax_sum < expected - 1.5:
            for c in candidates:
                if c[2] == 0.0:
                    guess = round(base * c[1] / 100.0, 2)
                    if tax_sum + guess <= expected + 1.5:
                        c[2] = guess
                        tax_sum += guess
                if abs(tax_sum - expected) <= 1.5:
                    break

    tax_lines = [
        TaxLine(
            rate=c[1],
            amount=round(c[2], 2),
            base=round(c[3], 2) if c[3] else (
                round(c[2] * 100.0 / c[1], 2) if c[1] else 0.0
            ),
        )
        for c in candidates if c[2] > 0 and c[1] > 0
    ]
    if not tax_lines:
        tax_lines = _extract_label_tax_lines(lines)
    if not tax_lines:
        tax_lines = _extract_hsn_tax_lines(lines)
    tax_sum = round(sum(t.amount for t in tax_lines), 2)

    # compute grand total from base + tax when not found on document
    if not total and base and tax_sum:
        total = round(base + tax_sum, 2)

    # --- reconcile the three totals
    if base and total is None and tax_sum:
        total = round(base + tax_sum, 2)
    if total and base is None and tax_sum:
        base = round(total - tax_sum, 2)
    if base and total and not tax_sum:
        tax_sum = round(total - base, 2)

    totals.base_amount = round(base or 0.0, 2)
    totals.total_amount = round(total or 0.0, 2)
    totals.tax_amount = round(tax_sum or 0.0, 2)
    result.tax_lines = tax_lines

    if totals.base_amount and totals.total_amount:
        if abs(totals.base_amount + totals.tax_amount - totals.total_amount) > max(
            1.5, 0.01 * totals.total_amount
        ):
            result.warnings.append(
                "totals mismatch: base + tax != total (check OCR quality)")
    if result.line_items and totals.base_amount:
        s = sum(it.total for it in result.line_items)
        if s and abs(s - totals.base_amount) / totals.base_amount > 0.05:
            result.warnings.append("line item totals do not sum to base amount")


# ---------------------------------------------------------------- pipeline

def extract_all(pages: List[Page]) -> InvoiceResult:
    result = InvoiceResult()
    if not pages:
        result.warnings.append("no pages found")
        return result
    extract_supplier(pages, result)
    extract_bank(pages, result)
    extract_invoice(pages, result)
    extract_line_items(pages, result)
    extract_totals_and_tax(pages, result)
    result.ocr_confidence = round(
        sum(p.mean_confidence() for p in pages) / len(pages), 3
    )
    return result
