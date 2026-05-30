"""
Field-specific post-processing for Arabic ID OCR output.

Each field on an Egyptian National ID has known constraints
(fixed length, limited vocabulary, date format, etc.).
Applying these constraints catches common OCR errors.
"""

import re
from typing import Optional

# ── Known vocabularies ────────────────────────────────────────────────────────

GENDER_VALUES = ["ذكر", "أنثى", "انثى", "male", "female"]

RELIGION_VALUES = [
    "مسلم", "مسلمة",
    "مسيحي", "مسيحية",
    "يهودي", "يهودية",
    "اخرى", "أخرى",
]

# Arabic-Indic → ASCII digit mapping
ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_numerals(text: str) -> str:
    """Convert Arabic-Indic numerals (٢٩٩...) to ASCII digits."""
    return text.translate(ARABIC_INDIC)


def nearest_match(text: str, vocab: list) -> Optional[str]:
    """Return closest vocabulary item or None if no reasonable match."""
    if not text:
        return None
    text_lower = text.strip().lower()
    for v in vocab:
        if v in text or text in v or text_lower == v.lower():
            return v
    return text   # return as-is if no match — don't discard


def fix_date(text: str) -> Optional[str]:
    """
    Normalise date text to DD/MM/YYYY.
    Handles: 01/01/1990, 1-1-1990, ١٩٩٠/١/١ (Arabic-Indic), etc.
    """
    if not text:
        return None
    t = normalize_numerals(text.strip())
    # Extract all digit groups
    parts = re.findall(r"\d+", t)
    if len(parts) == 3:
        d, m, y = parts
        # If year comes first (YYYY/MM/DD), swap
        if len(d) == 4:
            d, m, y = y, m, d
        return f"{d.zfill(2)}/{m.zfill(2)}/{y}"
    return t   # return normalised but un-reformatted


# ── Per-field processors ──────────────────────────────────────────────────────

def process_id(text: Optional[str]) -> Optional[str]:
    """Egyptian National ID: 14 digits."""
    if not text:
        return None
    digits = re.sub(r"\D", "", normalize_numerals(text))
    if len(digits) in (13, 14):          # accept 13 in case of OCR drop
        return digits
    return digits if digits else None


def process_serial(text: Optional[str]) -> Optional[str]:
    """Serial number: digits only."""
    if not text:
        return None
    digits = re.sub(r"\D", "", normalize_numerals(text))
    return digits if digits else None


def process_date(text: Optional[str]) -> Optional[str]:
    return fix_date(text) if text else None


def process_gender(text: Optional[str]) -> Optional[str]:
    return nearest_match(text, GENDER_VALUES) if text else None


def process_religion(text: Optional[str]) -> Optional[str]:
    return nearest_match(text, RELIGION_VALUES) if text else None


def process_name(text: Optional[str]) -> Optional[str]:
    """Arabic names: strip stray non-Arabic characters but keep spaces."""
    if not text:
        return None
    # Keep Arabic letters, spaces, and common punctuation
    cleaned = re.sub(r"[^؀-ۿ\s\-]", "", text).strip()
    return cleaned if cleaned else text


def process_address(text: Optional[str]) -> Optional[str]:
    """Addresses can contain Arabic + digits — light cleanup."""
    if not text:
        return None
    return normalize_numerals(text.strip())


# ── Dispatch table ────────────────────────────────────────────────────────────

FIELD_PROCESSORS = {
    "ID":          process_id,
    "Serial_Num":  process_serial,
    "IssueDate":   process_date,
    "ExpDate":     process_date,
    "Gender":      process_gender,
    "Religion":    process_religion,
    "First_Name":  process_name,
    "Last_Name":   process_name,
    "HusbandName": process_name,
    "Job":         process_name,
    "Add1":        process_address,
    "Add2":        process_address,
    "Status":      lambda t: t,   # free-form, no constraint
    "Front":       lambda t: t,
    "Back":        lambda t: t,
}


def postprocess_fields(fields: dict) -> dict:
    """
    Apply field-specific post-processing to all extracted fields.

    Args:
        fields: dict of {field_name: raw_ocr_text}

    Returns:
        dict of {field_name: cleaned_text}
    """
    cleaned = {}
    for field, text in fields.items():
        processor = FIELD_PROCESSORS.get(field, lambda t: t)
        cleaned[field] = processor(text)
    return cleaned
