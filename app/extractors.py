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
# Longest match first. Indian lakhs (1,51,725.00) must win over a truncated
# 51,725.00 — that truncation was copied into the line amount and then used
# to overwrite the printed rate.
CELL_NUM_RE = re.compile(
    r"(?<![\d.])("
    r"\d{1,3}(?:,\d{2})(?:,\d{2,3})+(?:\.\d+)?"
    r"|\d{1,3}(?:,\d{3})+(?:\.\d+)?"
    r"|\d+\.\d+"
    r"|\d+"
    r")(?!\d)"
)
_UNIT_TAIL_RE = re.compile(
    r"(?:kgs?|nos|pcs|units?|box(?:es)?|bundles?|mtr|qty|rs\.?|inr|%)\.?",
    re.IGNORECASE,
)
_GST_PERCENTS = (2.5, 5.0, 6.0, 9.0, 12.0, 14.0, 18.0, 28.0)
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


def cell_numbers(text: str) -> List[float]:
    """Parse numbers from a table cell, including bare integers (qty/rate).

    Do not strip spaces before matching — that glues neighbors like
    ``0.00 3,540.00`` into ``0.003,540.00`` and corrupts values.
    """
    if not text:
        return []
    vals = []
    for m in CELL_NUM_RE.finditer(text):
        v = parse_amount(m.group(1))
        if v is not None:
            vals.append(v)
    return vals


def _is_gst_percent(v: float) -> bool:
    return 0 < v <= 28.5 and any(abs(v - g) < 0.051 for g in _GST_PERCENTS)


def _close_money(a: float, b: float) -> bool:
    return abs(a - b) <= max(1.0, 0.02 * max(abs(a), abs(b), 1.0))


def _token_is_pure_number(text: str) -> bool:
    """True for a qty/money cell. Part codes and labels (SAC Code: 998898) are not."""
    raw = (text or "").strip()
    if not raw or not re.search(r"\d", raw):
        return False
    stripped = _UNIT_TAIL_RE.sub("", raw)
    stripped = stripped.strip(" .,:;/-")
    return bool(stripped) and re.fullmatch(r"[\d,]+(?:\.\d+)?", stripped) is not None


def _peel_hsn_prefix(text: str) -> str:
    """721510001,700.000 kgs → 1,700.000 kgs (HSN glued onto the quantity)."""
    compact = text.replace(" ", "")
    m = re.match(r"(\d+)(,\d{2,3}(?:,\d{2,3})*(?:\.\d+)?)", compact)
    if not m:
        return text
    digits, rest = m.group(1), m.group(2)
    for hsn_len in (8, 6, 4):
        if len(digits) <= hsn_len:
            continue
        qty_tok = digits[hsn_len:] + rest
        qty = parse_amount(qty_tok)
        if qty is not None and 0 < qty < 10_000_000:
            return qty_tok + compact[m.end():]
    return text


def _explode_qty_rate_amount(text: str):
    """Split one OCR blob into qty, rate, amount when qty × rate = amount.

    Example: 760429244.80KGS500.00KGS1,22,400.00 → 244.80, 500, 122400.
    """
    compact = re.sub(r"\s+", "", text or "")
    m = re.search(
        r"(\d{1,3}(?:,\d{2})(?:,\d{2,3})+(?:\.\d+)?"
        r"|\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+)$",
        compact,
    )
    if not m:
        return None
    amount = parse_amount(m.group(1))
    if not amount:
        return None
    head = compact[:m.start()]
    pieces = re.findall(
        r"(\d+(?:\.\d+)?)(?:kgs?|nos|pcs|unt|units?|mtr)?",
        head,
        flags=re.IGNORECASE,
    )
    if len(pieces) < 2:
        return None
    rate = float(pieces[-1])
    qty_raw = pieces[-2]
    if rate <= 0:
        return None
    candidates = [float(qty_raw)]
    if "." in qty_raw:
        whole, frac = qty_raw.split(".", 1)
        for n in (8, 6, 4):
            if len(whole) > n:
                candidates.append(float(whole[n:] + "." + frac))
    for qty in candidates:
        if qty > 0 and _close_money(qty * rate, amount):
            return qty, rate, amount
    return None


def _values_in_token(text: str, role: str) -> List[float]:
    """Numbers that belong in a qty/rate/amount cell. Part codes are ignored."""
    raw = (text or "").strip()
    if not raw:
        return []
    if role == "qty":
        raw = _peel_hsn_prefix(raw)
    if not _token_is_pure_number(raw):
        return []
    vals = cell_numbers(raw)
    if role == "qty" and len(vals) > 1:
        vals = [
            v for v in vals
            if not (v >= 1000 and abs(v - round(v)) < 1e-9 and 6 <= len(str(int(v))) <= 8)
        ] or vals
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


# Supplier notes / process annotations glued onto item names by OCR
_SUPPLIER_NOTE_CUT_RE = re.compile(
    r"(?i)"
    r"(?:^|[\s.:;|])(?:"
    r"your\s*ref\.?"
    r"|your\s*dc"
    r"|dc\s*(?:no|number|&|\()"
    r"|\(\s*qty\s*\)"
    r"|dc\s*&?\s*\(?\s*qty"
    r"|ref\.?\s*\(?\s*dc"
    r"|trivalent|passivation|yellow\s*pass"
    r"|ok\s*parts"
    r")"
)

_SUPPLIER_NOTE_LINE_RE = re.compile(
    r"(?i)^(?:"
    r"your\s*ref"
    r"|your\s*dc"
    r"|dc\s*&"
    r"|\(\s*qty\s*\)"
    r"|trivalent|passivation|yellow"
    r"|ok\s*parts"
    r"|being\s+"
    r"|narration"
    r")"
)

# Job-card / lot codes OCR often appends (7GI052, GIOHO, G1054)
_TRAILING_JOB_CODE_RE = re.compile(
    r"(?i)\s*(?:\d{0,2}GI\d{2,4}|GIOHO|GFos\d|G\d{3,4})\s*$"
)


def _is_desc_junk_token(tok: str) -> bool:
    """Trailing OCR garbage glued onto item names (wow, 7GIOHO, pu, etc.)."""
    t = tok.strip(".,:;|()[]")
    if not t:
        return True
    if re.fullmatch(r"\d{1,4}", t):
        return False
    # short all-lowercase OCR crumbs
    if len(t) <= 3 and t.isalpha() and t.islower():
        return True
    # job-card style codes from supplier notes (7GI052, GI052, GIOHO)
    if re.fullmatch(r"\d{0,2}GI\d{2,4}", t, re.IGNORECASE):
        return True
    if re.fullmatch(r"GIOHO|GFos\d", t, re.IGNORECASE):
        return True
    # keep real part/drawing codes like MTJNR1216K11, EN1AL
    if re.match(r"^[A-Z]{2,}\d", t) and not re.match(r"^GI\d", t, re.I):
        return False
    if t.isupper() and len(t) >= 4 and not re.search(r"\d", t):
        return False
    # mixed junk codes after a readable name (e.g. 7GIOHO, GFos3, 3ES)
    if len(t) <= 8 and re.search(r"\d", t) and re.search(r"[A-Za-z]", t):
        letters = sum(c.isalpha() for c in t)
        digits = sum(c.isdigit() for c in t)
        if t[0].isdigit() and letters <= 6:
            return True
        if letters <= 4 and digits <= 3:
            return True
    return False


def _is_supplier_note_fragment(text: str) -> bool:
    """True when a continuation line is only a supplier note, not part of the name."""
    t = (text or "").strip()
    if not t:
        return True
    if _SUPPLIER_NOTE_LINE_RE.search(t):
        return True
    if re.fullmatch(r"[\d().\s\-/:&]+", t):
        return True
    if re.fullmatch(r"\d{0,2}GI\d{2,4}", t, re.IGNORECASE):
        return True
    # "7GI052 Passivation", "4335(25) Yellow"
    if re.match(r"(?i)^(?:\d{0,2}GI\d{2,4}|\d{3,}\([\d.]+\))\b", t):
        return True
    return False


def _clean_item_description(text: str) -> str:
    """Keep only the proper item name — strip supplier notes and OCR junk.

    Examples removed:
      Your Ref. (DC) & QTY -4288(19.8)7GI05
      Your Ref. DC & (Qty) 998873 Trivalent ...
      OK PARTS / Passivation / Yellow / 7GI052
    """
    t = (text or "").strip()
    if not t:
        return ""
    # leading serial: "1 ", "2|", "3.", or fused row index "1HOUSING"
    # (do not strip sizes/codes like 2MM, 1F921)
    t = re.sub(r"^\s*\d{1,2}\s*[|.)\-:]\s*", "", t)
    t = re.sub(r"^\s*\d{1,2}\s+", "", t)
    t = re.sub(r"^\d(?=[A-Z][a-z]{2,})", "", t)  # 1Housing
    t = re.sub(r"^\d(?=[A-Z]{4,})", "", t)  # 1HOUSING

    # Hard cut at supplier-note anchors (everything from that point is clutter)
    m = _SUPPLIER_NOTE_CUT_RE.search(t)
    if m:
        t = t[: m.start()].rstrip(" .:;-|/\\")

    # Fused forms without clear spacing: "...488.Your Ref.DC&Qty..."
    t = re.sub(r"(?i)[.\s]*your\s*ref\.?.*$", "", t)
    t = re.sub(r"(?i)[.\s]*dc\s*&?\s*\(?\s*qty\)?.*$", "", t)
    t = re.sub(r"(?i)\s*\bdate\s*[:=].*$", "", t)

    # Trailing DC qty refs and job codes
    t = re.sub(r"\s*[-:]?\s*\d{3,}\(\d+(?:\.\d+)?\)\s*\d{0,2}GI\d{0,4}\s*$", "", t, flags=re.I)
    t = re.sub(r"\s+\d{3,}\(\d+(?:\.\d+)?\)\s*$", "", t)
    t = _TRAILING_JOB_CODE_RE.sub("", t)
    t = re.sub(r"\s+\d{5,}\s*$", "", t)

    tokens = t.split()
    while len(tokens) > 1 and _is_desc_junk_token(tokens[-1]):
        tokens.pop()
    return " ".join(tokens).strip(" .:;-|/\\")


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

    # --- name: prefer a company-like line (PVT/LTD/...), skip address fragments
    name_idx = None
    hinted = None
    fallback = None
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
        if PHONE_RE.search(t) and sum(c.isalpha() for c in t) < 8:
            continue
        if sum(c.isalpha() for c in t) < 5:
            continue
        cand = _clean_value(re.split(
            r"\bGSTIN\b|\bGST\b|\bInvoice\b", t, flags=re.IGNORECASE)[0])
        if not cand:
            continue
        if ADDRESS_FRAG_RE.search(cand) and not COMPANY_HINT_RE.search(cand):
            continue
        if COMPANY_HINT_RE.search(cand) and hinted is None:
            hinted = (i, cand)
            break
        if fallback is None:
            fallback = (i, cand)
    if hinted:
        name_idx, sup.name = hinted
    elif fallback:
        name_idx, sup.name = fallback

    # GSTIN often sits below the name; if header cut was too aggressive, use page text
    if not sup.gstin:
        all_gstins = GSTIN_RE.findall(body_text.upper())
        if all_gstins:
            from collections import Counter
            counts = Counter(all_gstins)
            singles = [g for g in all_gstins if counts[g] == 1]
            sup.gstin = singles[0] if singles else all_gstins[0]
            if sup.gstin:
                sup.pan = sup.gstin[2:12]
                if sup.gstin[:2] in STATE_CODES:
                    sup.billing_state = STATE_CODES[sup.gstin[:2]]

    # --- address: left-side header lines after the name, until label lines
    addr_parts: List[str] = []
    if name_idx is not None:
        skips = 0
        for ln in header_lines[name_idx + 1:]:
            t = _left_text(ln, page)
            if (not t or HEADER_SKIP_RE.search(t) or DOC_NOISE_RE.search(t)
                    or _is_hexish(t) or sum(c.isalnum() for c in t) < 5
                    or PHONE_RE.search(t) and not re.search(r"\d{1,4}\s+[A-Za-z]", t)):
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
    r"^Ack\s*Date|E-Way\s*Bill|Bill\s*Date|Original|Duplicate|Recipient|"
    r"phone|tel\b|mobile|email|website|state\s*code",
    re.IGNORECASE,
)
COMPANY_HINT_RE = re.compile(
    r"pvt|ltd|limited|llp|llc|\binc\b|company|enterprises|works|"
    r"industries|metals|aluminium|steel|finisher|precision|services|"
    r"traders|corporation|\bco\b",
    re.IGNORECASE,
)
ADDRESS_FRAG_RE = re.compile(
    r"district|kanchee|sriperum|layout|cross|phase|village|road|nagar|"
    r"street|fac\.|factory|sipcot|peenya|industrial|park\b|colony|"
    r"bangalore|bengaluru|chennai|thane|\bpincode\b|\bpin\b",
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
    "rate": [
        r"item\s*rate|uomrate|uom\s*rate|\brate\b",
        r"price|charges",
    ],
    "amount": [
        r"taxable|value",
        r"\bamount\b",
    ],
}

# Words that sit in tax-column bands — avoid treating them as item rate/amount
_TAX_HEADER_NOISE_RE = re.compile(
    r"cgst|sgst|igst|utgst|\btax\b|discount",
    re.IGNORECASE,
)


def _match_header_words(words: List[Word]) -> Optional[dict]:
    found: dict = {}
    # Prefer left-side rate (item rate) over right-side "Tax Rate"
    ordered = sorted(words, key=lambda w: w.cx)
    for rank in (0, 1):
        for w in ordered:
            if any(v is w for v in found.values()):
                continue
            # Skip tax-band labels when choosing rate/amount columns
            if _TAX_HEADER_NOISE_RE.search(w.text):
                continue
            for col, pats in HEADER_TOKENS.items():
                pat = pats[rank]
                if col in found or pat is None:
                    continue
                if re.search(pat, w.text, re.IGNORECASE):
                    # One OCR token covering both "Rate" and "Taxable Value".
                    # Split the box so the unit rate stays left of the line amount.
                    if col == "rate" and re.search(
                        r"taxable|\bamount\b|\bvalue\b", w.text, re.IGNORECASE
                    ):
                        mid = (w.x0 + w.x1) / 2.0
                        found["rate"] = Word(w.text, w.conf, w.x0, w.y0, mid, w.y1)
                        if "amount" not in found:
                            found["amount"] = Word(w.text, w.conf, mid, w.y0, w.x1, w.y1)
                        break
                    # "Rate" under CGST/Tax band: skip if neighbors are tax labels
                    if col in ("rate", "amount"):
                        nearby = " ".join(
                            x.text for x in ordered
                            if abs(x.cx - w.cx) < 120 and x is not w
                        )
                        if _TAX_HEADER_NOISE_RE.search(nearby) and col == "rate":
                            continue
                    # Item rate is left of taxable value; tax-rate % sits to the right
                    if col == "rate" and "amount" in found and w.cx > found["amount"].cx:
                        continue
                    found[col] = w
                    break
    # Two columns are enough when they are the item name and the line amount.
    # A rate column still needs a third anchor so a random "rate" label is ignored.
    if "description" in found and "amount" in found and len(found) >= 2:
        return found
    if "description" in found and "rate" in found and len(found) >= 3:
        return found
    # "Product" + "Name" on separate tokens
    prod = next((w for w in words if re.fullmatch(r"product", w.text, re.I)), None)
    name_w = next((w for w in words if re.fullmatch(r"name\.?", w.text, re.I)), None)
    if prod and name_w and "description" not in found:
        found["description"] = prod
    if "description" in found and "amount" in found and len(found) >= 2:
        return found
    if "description" in found and "rate" in found and len(found) >= 3:
        return found
    return None


def _is_header_continuation(ln: Line) -> bool:
    """A second header line is labels, not a priced item row."""
    keys = re.findall(
        r"description|particulars|\bqty\b|quantity|\brate\b|\bamount\b|"
        r"\bhsn\b|\bsac\b|taxable|\buom\b|\bunit\b|\bvalue\b|\bcharges\b|"
        r"\bgoods\b|\bitem\b|\bper\b|\bsr\.?\s*no\b",
        ln.text,
        re.IGNORECASE,
    )
    money = re.findall(r"\d+\.\d{2}|\d{1,3}(?:,\d{2,3})+", ln.text)
    # A priced row ("144.00 KVA at Rs 350.00 ... 50,400.00") is not a header.
    if len(money) >= 2:
        return False
    if money and not keys:
        return False
    return len(keys) >= 1


def _find_table_header(lines: List[Line]) -> tuple:
    """Find item table header; optionally merge one continuation header line."""
    best_h, best_idx, best_score = None, None, 0
    for i, ln in enumerate(lines):
        for span in (1, 2):
            chunk = lines[i:i + span]
            if len(chunk) > 1 and chunk[-1].cy - chunk[0].cy > 60:
                break
            if span > 1 and not _is_header_continuation(chunk[-1]):
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
        r"grand\s*total|invoice\s*total|total\s*invoice|total\s*basic|"
        r"taxable\s*amount|taxable\s*value|"
        r"amount\s*chargeable|amount\s*before\s*tax|net\s*payable|"
        r"current\s*bill\s*amount|"
        r"output\s*(cgst|sgst|igst|utgst|tax)|"
        r"\bcgst\b|\bsgst\b|\bigst\b|"
        r"\brcm\b|in\s*words?|declaration|precaution|precation|"
        r"terms\s*(?:and|&)\s*conditions",
        t, re.IGNORECASE,
    ):
        return True
    if re.fullmatch(r"[\s.:]*total[\s.:]*", t, re.IGNORECASE):
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


_TAX_ROW_RE = re.compile(
    r"output\s*(igst|cgst|sgst|utgst)|"
    r"\b(igst|cgst|sgst|utgst)\s*@?\s*\d|"
    r"tax\s*amount|round\s*off",
    re.IGNORECASE,
)
_SUBTOTAL_ROW_RE = re.compile(
    r"^(?:basic\s*amount|sub\s*total|taxable\s*(?:amount|value))\b",
    re.IGNORECASE,
)


def _column_bands(col_x: dict, page_width: float, tax_x: Optional[float]):
    """Half-open x ranges from midpoints so a rate cell cannot swallow the amount."""
    ordered = sorted(col_x.items(), key=lambda kv: kv[1])
    bands = []
    for i, (name, x) in enumerate(ordered):
        left = 0.0 if i == 0 else (ordered[i - 1][1] + x) / 2.0
        right = page_width if i + 1 == len(ordered) else (x + ordered[i + 1][1]) / 2.0
        if name == "amount" and tax_x is not None:
            right = min(right, tax_x - 4.0)
        if right > left:
            bands.append((name, left, right))
    return bands


def _tax_zone_x(lines: List[Line], start_idx: int) -> Optional[float]:
    xs = []
    for ln in lines[max(0, start_idx - 3):start_idx]:
        for w in ln.words:
            if re.search(r"\b(cgst|sgst|igst|utgst)\b", w.text, re.IGNORECASE):
                xs.append(w.x0)
            elif re.search(r"\btax\b", w.text, re.IGNORECASE) and not re.search(
                r"taxable|before\s*tax", w.text, re.IGNORECASE
            ):
                xs.append(w.x0)
    return min(xs) if xs else None


def _extract_line_items_from_page(
    page: Page,
    lines: List[Line],
    start_idx: int,
    col_x: dict,
) -> List[LineItem]:
    """Parse item rows on one page using known column centers."""

    tax_x = _tax_zone_x(lines, start_idx)
    # A header word that landed inside the GST columns is not the line amount.
    if tax_x is not None and col_x.get("amount", 0) >= tax_x - 4:
        col_x = {k: v for k, v in col_x.items() if k != "amount"}
    bands = _column_bands(col_x, page.width, tax_x)

    desc_x = col_x.get("description")
    leftmost = min(col_x.values()) if col_x else 0.0

    def band_of(w: Word) -> str:
        # GST rate/amount columns sit to the right of the line amount.
        if tax_x is not None and w.cx >= tax_x - 4:
            return "tax"
        # Far-left block (bank details beside a charges table) is not the item name.
        if (
            desc_x is not None
            and abs(desc_x - leftmost) < 1
            and w.cx < desc_x - 360
        ):
            return "other"
        for name, left, right in bands:
            if left <= w.cx < right:
                return name
        return "description"

    max_gap = max(page.height * 0.08, 400.0)
    prev_y: Optional[float] = None
    items: List[LineItem] = []
    current: Optional[LineItem] = None

    for ln in lines[start_idx:]:
        if _is_table_end(ln):
            break
        if prev_y is not None and ln.cy - prev_y > max_gap:
            break
        if _SUBTOTAL_ROW_RE.search(ln.text.strip()):
            continue
        if _TAX_ROW_RE.search(ln.text) and not re.search(
            r"description|particulars|goods", ln.text, re.IGNORECASE
        ):
            # "OUTPUT IGST @ 18%  27,310.50" is a tax row, not an item.
            # Stop if the tax section has started; otherwise just skip the row.
            if re.search(r"output|cgst|sgst|igst", ln.text, re.IGNORECASE):
                break
            continue
        # skip handwriting / stamp noise (low recognition confidence)
        confs = [w.conf for w in ln.words]
        if confs and sum(confs) / len(confs) < 0.6:
            continue

        cells = {c: [] for c in list(col_x) + ["description", "tax", "other"]}
        for w in ln.words:
            cells[band_of(w)].append(w)

        def _col_vals(role: str) -> List[float]:
            out: List[float] = []
            for w in cells.get(role, []):
                raw = w.text.strip()
                # ".150" is OCR for 150; keep 0.150 as well and let reconciliation choose.
                if re.fullmatch(r"\.\d{2,4}", raw):
                    out.extend(_values_in_token(raw.lstrip("."), role))
                    out.extend(_values_in_token("0" + raw, role))
                    continue
                out.extend(_values_in_token(w.text, role))
            return out

        qty_v = _col_vals("qty")
        rate_v = _col_vals("rate")
        amt_v = [a for a in _col_vals("amount") if a > 0]
        # GST % sitting in the amount band is not the line amount when a real
        # amount is also there (this was the "rate saved as amount" bug).
        if any(not _is_gst_percent(a) and a > 28 for a in amt_v):
            amt_v = [a for a in amt_v if not _is_gst_percent(a)]

        exploded = None
        for w in ln.words:
            exploded = _explode_qty_rate_amount(w.text)
            if exploded:
                break

        if exploded:
            qty, price, total = exploded
            rate_from_col = amt_from_col = True
        else:
            price = 0.0
            total = 0.0
            qty = 0.0
            rate_from_col = bool(rate_v)
            amt_from_col = bool(amt_v)

            if len(rate_v) == 1:
                price = rate_v[0]
            elif rate_v:
                non_gst = [r for r in rate_v if not _is_gst_percent(r)]
                price = non_gst[0] if non_gst else rate_v[0]

            if len(amt_v) == 1:
                total = amt_v[0]
            elif amt_v:
                total = max(amt_v)

            if qty_v:
                if len(qty_v) == 1:
                    qty = qty_v[0]
                elif price and (total or len(rate_v) > 1):
                    targets = amt_v or rate_v
                    qty = min(
                        qty_v,
                        key=lambda q: min(abs(q * price - a) for a in targets) if q else 1e18,
                    )
                else:
                    qty = max(qty_v)

            # Rate column also held the taxable value ("Rate / Taxable Value"
            # read as one header). qty × unit rate = that larger number.
            # Using the taxable value as the rate is the crash-sized error.
            if not amt_v and qty and len(rate_v) >= 2:
                hit = None
                for r in sorted(rate_v):
                    for a in rate_v:
                        if a > r and _close_money(qty * r, a):
                            hit = (r, a)
                            break
                    if hit:
                        break
                if hit:
                    price, total = hit
                    amt_from_col = True
                elif qty > 1:
                    smaller, larger = min(rate_v), max(rate_v)
                    if larger > smaller * 2 and smaller < larger:
                        price, total = smaller, larger
                        amt_from_col = True

            # If several rates/amounts were read, prefer the pair that
            # multiplies with qty. Never replace a single printed rate with
            # amount/qty — that is what turned 89.25 into 73.89 and 140 into 168.
            if qty and len(rate_v) > 1 and amt_v:
                price = min(rate_v, key=lambda r: min(abs(qty * r - a) for a in amt_v))
            if qty and price and len(amt_v) > 1:
                best = min(amt_v, key=lambda a: abs(qty * price - a))
                if _close_money(qty * price, best) or not _is_gst_percent(best):
                    total = best

            if not qty and price and total and price > 0:
                q = total / price
                if 0 < q < 10_000_000:
                    qty = round(q, 3)
            # Invent a rate only when the rate column was empty.
            if qty and total and not rate_from_col and qty > 0:
                price = round(total / qty, 4)
            if qty and price and not amt_from_col:
                total = round(qty * price, 2)
            # Amount cell repeated the unit rate (qty > 1 so they cannot be equal).
            if (
                qty > 1 and price and total and rate_from_col
                and _close_money(price, total)
                and not _close_money(qty * price, total)
            ):
                total = round(qty * price, 2)
            # Amount cell held the GST % (18) instead of the taxable value.
            if (
                qty and price and total and rate_from_col
                and _is_gst_percent(total)
                and qty * price > max(28.0, total * 4)
            ):
                total = round(qty * price, 2)

        desc_text = " ".join(w.text for w in cells.get("description", [])).strip()
        desc_text = _clean_item_description(desc_text)
        # Short labels like "Handling Charges" can land in HSN column — recover text
        if not desc_text and (total or price):
            skip_cols = set()
            for col in ("qty", "rate", "amount"):
                if col in col_x:
                    skip_cols.add(col)
            recovered = []
            for w in ln.words:
                col = band_of(w)
                if col in skip_cols or col == "tax":
                    continue
                if cell_numbers(w.text) and not re.search(r"[A-Za-z]{2,}", w.text):
                    continue
                recovered.append(w.text)
            desc_text = _clean_item_description(" ".join(recovered))

        if total or price:
            if not desc_text:
                continue
            if META_ROW_RE.search(desc_text):
                continue
            if re.search(r"^(?:your\s*dc|narration)\b", desc_text, re.IGNORECASE):
                continue
            # Category header rows with no economics (e.g. "Labour Jobwork Services")
            if (
                re.search(r"^labour\s*jobwork\s*services\b", desc_text, re.IGNORECASE)
                and not total and not price
            ):
                continue
            if total == 0 and price == 0:
                continue
            current = LineItem(
                description=desc_text,
                quantity=qty,
                unit_price=price,
                total=total,
            )
            items.append(current)
            prev_y = ln.cy
        elif desc_text and current is not None and len(desc_text) > 2:
            # Never merge supplier notes / process annotations into the item name
            if META_ROW_RE.search(desc_text) or _is_supplier_note_fragment(desc_text):
                continue
            cleaned = _clean_item_description(desc_text)
            if not cleaned or len(cleaned) <= 2 or _is_supplier_note_fragment(cleaned):
                continue
            current.description = _clean_item_description(
                (current.description + " " + cleaned).strip()
            )
            prev_y = ln.cy
        # Do not create empty "description-only" placeholder rows

    return items


# Handling / packing / freight rows — treat as qty 1 × rate = amount
CHARGE_LABEL_RE = re.compile(
    r"(?i)\b(?:"
    r"handling\s*charges?"
    r"|pack(?:ing|aging)(?:\s*&\s*forwarding)?\s*charges?"
    r"|packing\s*&\s*forwarding"
    r"|forwarding\s*charges?"
    r"|freight(?:\s*charges?)?"
    r"|cartage(?:\s*charges?)?"
    r"|other\s*charges?"
    r")\b"
)

_CHARGE_STOP_RE = re.compile(
    r"(?i)\b(?:cgst|sgst|igst|utgst|taxable|grand\s*total|bill\s*total|"
    r"invoice\s*total|total\s*tax|tax\s*amount|round\s*off|"
    r"amount\s*chargeable|amount\s*payable|in\s*words)\b"
)


def _is_charge_description(text: str) -> bool:
    return bool(CHARGE_LABEL_RE.search(text or ""))


def _normalize_charge_item(it: LineItem) -> LineItem:
    """Force charge rows to qty=1, rate=amount so ERP review accepts them."""
    amount = float(it.total or 0) or float(it.unit_price or 0)
    if amount <= 0:
        return it
    it.quantity = 1.0
    it.unit_price = round(amount, 2)
    it.total = round(amount, 2)
    return it


def _charge_key(desc: str) -> str:
    return re.sub(r"\s+", " ", (desc or "").strip().lower())


def _extract_charge_items_from_lines(lines: List[Line]) -> List[LineItem]:
    """Find Handling/Packing/Freight charge rows after the item table."""
    out: List[LineItem] = []
    for ln in lines:
        # Stop at tax / grand-total section (do not stop on the charge line itself)
        if (
            not _is_charge_description(ln.text)
            and _CHARGE_STOP_RE.search(ln.text)
        ):
            break
        if not _is_charge_description(ln.text):
            continue
        nums = cell_numbers(ln.text) + amounts_in(ln.text)
        # Prefer the rightmost money-like amount; skip GST-looking percents
        money = [
            n for n in nums
            if n > 0 and not any(abs(n - g) < 0.05 for g in (5.0, 9.0, 12.0, 18.0, 28.0))
        ]
        if money:
            amount = money[-1]
        elif any(n > 0 for n in nums):
            amount = max(n for n in nums if n > 0)
        else:
            amount = 0.0
        if amount <= 0:
            continue
        label = CHARGE_LABEL_RE.search(ln.text)
        desc = label.group(0).strip() if label else ln.text.strip()
        desc = re.sub(r"\s+", " ", desc)
        low = desc.lower()
        if "handling" in low:
            desc = "Handling Charges"
        elif "packing" in low and "forward" in low:
            desc = "Packing & Forwarding Charges"
        elif re.search(r"pack(?:ing|aging)\s*charges?", low):
            desc = "Packaging Charges"
        elif "freight" in low:
            desc = "Freight"
        elif "cartage" in low:
            desc = "Cartage"
        elif "forwarding" in low:
            desc = "Forwarding Charges"
        elif "other" in low:
            desc = "Other Charges"
        out.append(_normalize_charge_item(LineItem(
            description=desc, quantity=1.0, unit_price=amount, total=amount,
        )))
    return out


def _is_junk_line_item(it: LineItem) -> bool:
    desc = (it.description or "").strip()
    if not desc and not it.total:
        return True
    if re.search(r"\b(CGST|SGST|IGST|OUTPUT|TOTAL)\b", desc, re.IGNORECASE):
        return True
    if META_ROW_RE.search(desc):
        return True
    if re.search(r"^(?:your\s*dc|narration)\b", desc, re.IGNORECASE):
        return True
    if (
        re.search(r"^labour\s*jobwork\s*services\b", desc, re.IGNORECASE)
        and not it.total and not it.quantity and not it.unit_price
    ):
        return True
    # Pure numeric / DC-ref crumbs that became fake rows ("4288(19.8)", "4288(19.8) GI052")
    if re.fullmatch(r"[\d().\s\-/,]+", desc):
        return True
    if re.match(r"^\d{3,}\([\d.]+\)", desc):
        return True
    if len(desc) <= 12 and desc.count("(") and re.search(r"\d", desc) and not re.search(r"[A-Za-z]{3,}", desc):
        return True
    # OCR-garbled totals rows ("rotal", "tota1")
    if re.fullmatch(r"(?:rotal|tota[l1]|sub\s*total|total)", desc, re.IGNORECASE):
        return True
    # Empty economics (no qty, rate, or amount)
    if not it.total and not it.quantity and not it.unit_price:
        return True
    return False


def extract_line_items(pages: List[Page], result: InvoiceResult) -> None:
    """Collect line items across all OCR pages (continuation tables reuse columns)."""
    if not pages:
        return

    items: List[LineItem] = []
    col_x: Optional[dict] = None
    last_header_idx: Optional[int] = None
    last_page_for_header: Optional[Page] = None

    for page_i, page in enumerate(pages):
        lines = page.lines
        header, header_idx = _find_table_header(lines)
        if header is not None:
            col_x = {c: w.cx for c, w in header.items()}
            start_idx = header_idx + 1
            last_header_idx = header_idx
            last_page_for_header = page
        elif col_x is not None and page_i > 0:
            # Continuation page with no repeated header — keep prior column map
            start_idx = 0
        else:
            continue

        items.extend(_extract_line_items_from_page(page, lines, start_idx, col_x))

    cleaned: List[LineItem] = []
    for it in items:
        it.description = _clean_item_description(it.description)
        if _is_junk_line_item(it):
            continue
        if _is_charge_description(it.description):
            it = _normalize_charge_item(it)
        cleaned.append(it)

    # Post-pass: packing/handling/freight often sit after the item grid
    existing = {_charge_key(it.description) for it in cleaned}
    scan_lines: List[Line] = []
    if last_page_for_header is not None and last_header_idx is not None:
        scan_lines = last_page_for_header.lines[last_header_idx + 1:]
    else:
        scan_lines = _all_lines(pages)
    for ch in _extract_charge_items_from_lines(scan_lines):
        key = _charge_key(ch.description)
        if key in existing:
            continue
        # Also skip if a cleaned table row already covers the same charge loosely
        if any(key in e or e in key for e in existing):
            continue
        cleaned.append(ch)
        existing.add(key)

    result.line_items = cleaned


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
    r"current\s*bill\s*amount", r"net\s*payable",
]

ROUND_OFF_RE = re.compile(
    r"round\s*off(?:\s*amount)?|rounded\s*off|rounding|"
    r"add\s*/?\s*less\s*:\s*round|r\s*/?\s*off|r\.?\s*off",
    re.IGNORECASE,
)
# Signed amount near round-off: -0.20, (0.20), 0.03, Less 0.20
SIGNED_AMOUNT_RE = re.compile(
    r"(?<![\d.])(?:Less\s+)?(?:\((?P<paren>\d+(?:\.\d+)?)\)|"
    r"(?P<signed>-?\d+(?:\.\d+)?))(?!\d)",
    re.IGNORECASE,
)


def _amount_on_line(ln: Line) -> Optional[float]:
    best = None
    for w in ln.words:
        # Include bare integers — many totals OCR without decimals (35800)
        vals = cell_numbers(w.text)
        if vals:
            best = vals[-1]
    return best


def _signed_amount_in(text: str) -> Optional[float]:
    """Parse a round-off amount; supports negatives and parentheses."""
    best = None
    for m in SIGNED_AMOUNT_RE.finditer(text):
        if m.group("paren"):
            val = -(parse_amount(m.group("paren")) or 0.0)
        else:
            raw = m.group("signed")
            val = parse_amount(raw) if raw else None
            if val is None:
                continue
            if re.search(r"\bless\b", text[:m.start()], re.IGNORECASE) and val > 0:
                val = -val
        if abs(val) <= 5.0:  # round-off is almost always paisa-scale
            best = val
    return best


def _extract_round_off(lines: List[Line]) -> Optional[float]:
    """Find an explicitly labeled Round Off amount on the invoice."""
    for i, ln in enumerate(lines):
        if not ROUND_OFF_RE.search(ln.text):
            continue
        # Prefer amount on the same line (often "Round Off -0.20")
        amt = _signed_amount_in(ln.text)
        if amt is not None:
            return round(amt, 2)
        # Sometimes OCR puts the amount on the next line
        if i + 1 < len(lines):
            amt = _signed_amount_in(lines[i + 1].text)
            if amt is not None:
                return round(amt, 2)
            # fallback: unsigned amount on next line when label says Less
            plain = _amount_on_line(lines[i + 1])
            if plain is not None and plain <= 5.0:
                if re.search(r"\bless\b", ln.text, re.IGNORECASE):
                    return round(-plain, 2)
                return round(plain, 2)
    return None


def _find_subtotal(lines: List[Line]) -> Optional[float]:
    """Find the pre-tax subtotal row (Sub Total / Taxable), not Bill/Grand Total."""
    # Prefer explicit pre-tax labels first (Taxable before Sub Total — Sub Total
    # is often tax-inclusive on job-work invoices).
    for pat in (
        r"taxable\s*(?:amount|value)",
        r"total\s*taxable",
        r"total\s*amount\s*before\s*tax",
        r"sub\s*total",
    ):
        rx = re.compile(pat, re.IGNORECASE)
        for ln in lines:
            if rx.search(ln.text):
                amt = _amount_on_line(ln)
                if amt is not None and amt > 0:
                    return amt
    for ln in lines:
        if not re.search(r"\btotal\b", ln.text, re.IGNORECASE):
            continue
        if re.search(
            r"grand|bill\s*total|invoice\s*total|total\s*amount|chargeable|"
            r"before\s*tax|amount\s*payable|net\s*amount|"
            r"\b(kgs?|nos|pcs|units?|bundles?|unt)\b|"
            r"\bcgst\b|\bsgst\b|\bigst\b|t[ae]x\s*amount|hsn\s*/?\s*sac|"
            r"in\s*words|e-?way",
            ln.text, re.IGNORECASE,
        ):
            continue
        # Skip HSN tax-summary totals like "Total 641.20 57.71 57.71 115.42"
        nums = amounts_in(ln.text) + _whole_amounts(ln.text, min_val=1.0)
        if len(nums) >= 3:
            continue
        amt = _amount_on_line(ln)
        if amt is not None and amt > 0:
            return amt
    return None


def _synthesize_tax_lines(
    lines: List[Line],
    base: Optional[float],
    total: Optional[float],
    tax_labeled: Optional[float],
    round_off: float,
) -> List[TaxLine]:
    """Build CGST/SGST or IGST rows when OCR has Tax Amount but no labeled % rows."""
    tax_amt = tax_labeled
    if tax_amt is None and base is not None and total is not None:
        gap = round(total - base - (round_off or 0.0), 2)
        if gap > 0:
            tax_amt = gap
    if not tax_amt or tax_amt <= 0:
        return []

    blob = " ".join(ln.text for ln in lines[-40:])  # totals area
    has_cgst = bool(re.search(r"\bCGST\b", blob, re.IGNORECASE))
    has_sgst = bool(re.search(r"\bSGST\b", blob, re.IGNORECASE))
    has_igst = bool(re.search(r"\bIGST\b", blob, re.IGNORECASE))

    # Infer rate from base if possible
    rate = None
    if base and base > 0:
        pct = round(100.0 * tax_amt / base, 2)
        for candidate in (5, 12, 18, 28, 40):
            if abs(pct - candidate) <= 0.6:
                rate = float(candidate)
                break
        if rate is None and abs(pct - 18) < 2.5:
            rate = 18.0

    if rate is None:
        # Look for an explicit 18%/9% near tax labels
        m = re.search(r"(\d{1,2}(?:\.\d+)?)\s*%", blob)
        if m:
            r = float(m.group(1))
            if 0 < r <= 28:
                rate = r
    if rate is None:
        rate = 18.0

    base_val = round(base, 2) if base else round(tax_amt * 100.0 / rate, 2)

    # Intra-state: CGST+SGST headers (or no IGST) → split half at rate/2
    if (has_cgst and has_sgst) or (has_cgst and not has_igst):
        half_rate = round(rate / 2.0, 2)
        half_amt = round(tax_amt / 2.0, 2)
        other = round(tax_amt - half_amt, 2)
        return [
            TaxLine(kind="CGST", rate=half_rate, base=base_val, amount=half_amt),
            TaxLine(kind="SGST", rate=half_rate, base=base_val, amount=other),
        ]

    return [TaxLine(kind="IGST", rate=rate, base=base_val, amount=round(tax_amt, 2))]


def _clean_tax_kind(raw: str) -> str:
    k = (raw or "").strip().upper()
    return k if k in ("CGST", "SGST", "IGST", "UTGST") else ""


def _kind_from_text(text: str) -> str:
    m = re.search(r"\b(IGST|CGST|SGST|UTGST)\b", text or "", re.IGNORECASE)
    return m.group(1).upper() if m else ""


def _infer_missing_tax_kinds(tax_lines: List[TaxLine]) -> None:
    """Fill blank kinds using rate/pairing heuristics (CGST+SGST vs IGST)."""
    if not tax_lines:
        return
    if all(t.kind for t in tax_lines):
        return

    unknown = [t for t in tax_lines if not t.kind]
    if len(tax_lines) == 1 and unknown:
        rate = tax_lines[0].rate
        # Half-rates (2.5/6/9/14) are usually CGST/SGST; full rates → IGST
        if abs(rate - round(rate)) < 0.05 and int(round(rate)) in (2, 3, 6, 9, 14):
            tax_lines[0].kind = "CGST"
        else:
            tax_lines[0].kind = "IGST"
        return

    # Two (or more) unlabeled lines at the same half-rate → CGST then SGST
    if len(unknown) >= 2:
        rates = [round(t.rate, 2) for t in unknown]
        if abs(rates[0] - rates[1]) < 0.05 and rates[0] <= 14:
            unknown[0].kind = "CGST"
            unknown[1].kind = "SGST"
            for t in unknown[2:]:
                if not t.kind:
                    t.kind = "SGST" if abs(t.rate - rates[0]) < 0.05 else "IGST"
            return

    for t in unknown:
        if t.kind:
            continue
        if t.rate <= 14 and abs(t.rate - round(t.rate)) < 0.05 and int(round(t.rate)) in (2, 3, 6, 9, 14):
            # Prefer CGST if no CGST yet, else SGST
            if not any(x.kind == "CGST" for x in tax_lines):
                t.kind = "CGST"
            else:
                t.kind = "SGST"
        else:
            t.kind = "IGST"


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
                        kind=kind,
                        rate=rate, amount=round(amount, 2),
                        base=round(amount * 100.0 / rate, 2),
                    ))
            break
    return out


def _extract_hsn_tax_lines(lines: List[Line]) -> List[TaxLine]:
    """Parse HSN summary blocks: HSN | Taxable Amount | IGST 18.0%126.00."""
    header_idx = None
    tax_kind = ""
    for i, ln in enumerate(lines):
        if re.search(r"taxable\s*amount", ln.text, re.IGNORECASE) and re.search(
            r"IGST|CGST|SGST|UTGST|HSN", ln.text, re.IGNORECASE
        ):
            header_idx = i
            tax_kind = _kind_from_text(ln.text)
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
        row_kind = _kind_from_text(ln.text) or tax_kind
        for rate_s, amt_s in glued:
            rate = float(rate_s)
            amount = parse_amount(amt_s) or 0.0
            if 0 < rate <= 28 and amount > 0:
                base = taxable if taxable else round(amount * 100.0 / rate, 2)
                tax_lines.append(TaxLine(
                    kind=row_kind, rate=rate, base=base, amount=amount,
                ))
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
    # Fill missing base from line items only — never override a labeled subtotal
    # (bad OCR totals on rows used to replace Sub Total / Taxable Amount).
    if base is None and result.line_items:
        item_sum = sum(it.total for it in result.line_items)
        if item_sum > 0:
            base = item_sum
    base_anchored = base is not None

    total = find_anchor_amount(TOTAL_ANCHORS, min_value=(base or 0.0) * 0.5)
    total_anchored = total is not None

    tax_labeled = find_anchor_amount(
        [r"total\s*tax", r"tax\s*amount", r"gst\s*amount", r"total\s*gst"],
    )

    # Explicit Round Off early so tax inference can use it
    early_round_off = _extract_round_off(lines) or 0.0

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
                candidates.append([_kind_from_text(ln.text), rate, amt, 0.0])
            continue
        m_add = re.search(r"Add\s*(CGST|SGST|IGST|UTGST)\s*%", ln.text, re.IGNORECASE)
        if m_add:
            nums = _whole_amounts(ln.text) + amounts_in(ln.text)
            small = [n for n in nums if n <= 28]
            big = [n for n in nums if n > 28]
            if big:
                rate = small[-1] if small else 0.0
                candidates.append([m_add.group(1).upper(), rate, big[-1], 0.0])
            continue
        for rate_s, amt_s in PCT_GLUE_RE.findall(ln.text):
            rate = float(rate_s)
            amt = parse_amount(amt_s) or 0.0
            if 0 < rate <= 28 and amt > 0:
                nums = [a for a in amounts_in(ln.text) if a > 28]
                base_val = nums[0] if nums else 0.0
                candidates.append([_kind_from_text(ln.text), rate, amt, base_val])

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
        expected = total - base - early_round_off
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
            kind=_clean_tax_kind(c[0]),
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
    if not tax_lines:
        tax_lines = _synthesize_tax_lines(lines, base, total, tax_labeled, early_round_off)
    _infer_missing_tax_kinds(tax_lines)
    tax_sum = round(sum(t.amount for t in tax_lines), 2)
    if (not tax_sum) and tax_labeled:
        tax_sum = round(tax_labeled, 2)

    # Prefer explicit Round Off label; else infer when gap is paisa-scale (≤ ₹1)
    round_off = early_round_off if early_round_off else None
    if round_off is None and base is not None and total is not None and tax_sum:
        inferred = round(total - (base + tax_sum), 2)
        if abs(inferred) <= 1.0 and abs(inferred) > 0.001:
            round_off = inferred
    if round_off is None:
        round_off = 0.0

    # compute grand total from base + tax + round_off when not found on document
    if not total and base and tax_sum:
        total = round(base + tax_sum + round_off, 2)

    # --- reconcile totals: base + tax + round_off ≈ total
    if base and total is None and tax_sum:
        total = round(base + tax_sum + round_off, 2)
    if total and base is None and tax_sum:
        base = round(total - tax_sum - round_off, 2)
    if base and total and not tax_sum:
        tax_sum = round(total - base - round_off, 2)

    totals.base_amount = round(base or 0.0, 2)
    totals.total_amount = round(total or 0.0, 2)
    totals.tax_amount = round(tax_sum or 0.0, 2)
    totals.round_off = round(round_off, 2)
    result.tax_lines = tax_lines

    if totals.base_amount and totals.total_amount:
        expected = round(
            totals.base_amount + totals.tax_amount + totals.round_off, 2
        )
        if abs(expected - totals.total_amount) > max(
            1.5, 0.01 * totals.total_amount
        ):
            result.warnings.append(
                "totals mismatch: base + tax + round_off != total (check OCR quality)")
    if result.line_items and totals.base_amount:
        s = sum(it.total for it in result.line_items)
        if s and abs(s - totals.base_amount) / totals.base_amount > 0.05:
            result.warnings.append("line item totals do not sum to base amount")


# ---------------------------------------------------------------- pipeline

def extract_all(pages: List[Page]) -> InvoiceResult:
    result = InvoiceResult(document_type="paper_gst")
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
