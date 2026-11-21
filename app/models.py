"""Pydantic models describing the invoice extraction result schema."""
from typing import List, Optional
from pydantic import BaseModel


class Supplier(BaseModel):
    name: str = ""
    current_address: str = ""
    billing_address: str = ""
    billing_state: str = ""
    gstin: str = ""
    pan: str = ""
    contact_person: str = ""
    phone: str = ""
    email: str = ""


class Bank(BaseModel):
    account_number: str = ""
    account_type: str = ""
    ifsc: str = ""


class Invoice(BaseModel):
    number: str = ""
    date: str = ""
    due_date: str = ""
    irn: str = ""


class Totals(BaseModel):
    base_amount: float = 0
    tax_amount: float = 0
    round_off: float = 0  # can be negative (e.g. -0.20)
    total_amount: float = 0


class LineItem(BaseModel):
    description: str = ""
    quantity: float = 0
    unit_price: float = 0
    total: float = 0


class TaxLine(BaseModel):
    kind: str = ""  # CGST | SGST | IGST | UTGST
    rate: float = 0
    base: float = 0
    amount: float = 0


class InvoiceResult(BaseModel):
    supplier: Supplier = Supplier()
    bank: Bank = Bank()
    invoice: Invoice = Invoice()
    totals: Totals = Totals()
    line_items: List[LineItem] = []
    tax_lines: List[TaxLine] = []
    # paper_gst | thermal_fuel | thermal_restaurant | thermal_utility | thermal_generic
    document_type: str = "paper_gst"
    # Extraction metadata (useful for the ERP to decide on manual review)
    ocr_confidence: Optional[float] = None
    warnings: List[str] = []
