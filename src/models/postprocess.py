"""
Field-specific post-processing for Arabic ID OCR output.

Each field on an Egyptian National ID has known constraints
(fixed length, limited vocabulary, date format, etc.).
Applying these constraints catches common OCR errors.
"""

import re
import unicodedata
from typing import Optional

# ── Known vocabularies ────────────────────────────────────────────────────────

GENDER_VALUES = ["ذكر", "أنثى", "انثى"]

RELIGION_VALUES = [
    "مسلم", "مسلمة",
    "مسيحي", "مسيحية",
    "يهودي", "يهودية",
    "اخرى", "أخرى",
]

STATUS_VALUES = ["أعزب", "متزوج", "متزوجة", "مطلق", "مطلقة", "أرمل", "أرملة"]

# Arabic-Indic → ASCII digit mapping
ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# Unicode directional/formatting marks that PaddleOCR often injects
_BIDI_MARKS = re.compile(r"[‎‏‪-‮⁦-⁩﻿]")


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_numerals(text: str) -> str:
    """Convert Arabic-Indic numerals (٢٩٩...) to ASCII digits."""
    return text.translate(ARABIC_INDIC)


def strip_bidi(text: str) -> str:
    """Remove invisible Unicode bidirectional control characters."""
    return _BIDI_MARKS.sub("", text).strip()


def is_mostly_arabic(text: str, threshold: float = 0.4) -> bool:
    """Return True if at least `threshold` fraction of chars are Arabic."""
    if not text:
        return False
    arabic = sum(1 for c in text if "؀" <= c <= "ۿ")
    return arabic / max(len(text.replace(" ", "")), 1) >= threshold


def fuzzy_vocab_match(text: str, vocab: list) -> Optional[str]:
    """
    Return the closest vocabulary item using character overlap.
    Falls back to the raw text when no match is close enough.
    """
    if not text:
        return None
    text_clean = strip_bidi(text).strip()

    # Exact / substring match first
    for v in vocab:
        if v == text_clean or v in text_clean or text_clean in v:
            return v

    # Levenshtein-style: count matching chars (simple but fast)
    best_v, best_score = None, 0
    for v in vocab:
        common = sum(1 for c in v if c in text_clean)
        score  = common / max(len(v), 1)
        if score > best_score:
            best_score, best_v = score, v

    # Accept match only if reasonably confident
    if best_score >= 0.6:
        return best_v
    return text_clean if text_clean else None


def _try_ymd(y: str, m: str, d: str) -> Optional[str]:
    """Return YYYY/MM/DD if values are in valid ranges, else None."""
    try:
        yi, mi, di = int(y), int(m), int(d)
        if 1900 <= yi <= 2100 and 1 <= mi <= 12 and 1 <= di <= 31:
            return f"{yi:04d}/{mi:02d}/{di:02d}"
    except ValueError:
        pass
    return None


def fix_date(text: str) -> Optional[str]:
    """
    Normalise any date text to YYYY/MM/DD.

    Handles:
      - Separated:  02/03/2028  |  2028/03/02  |  2-3-2028
      - 8-digit:    20280302 (YYYYMMDD)  |  02032028 (DDMMYYYY)
      - 10-digit:   2028103102 → strip to leading 8 and try YYYYMMDD
      - Arabic-Indic numerals normalised first
    """
    if not text:
        return None
    t = normalize_numerals(strip_bidi(text)).strip()

    # ── Case 1: separated parts (slash / dash / dot / space) ──────────────────
    parts = re.findall(r"\d+", t)
    if len(parts) == 3:
        a, b, c = parts
        if len(a) == 4:                        # YYYY / M / D
            result = _try_ymd(a, b, c)
        elif len(c) == 4:                      # D / M / YYYY
            result = _try_ymd(c, b, a)
        else:
            result = None
        if result:
            return result

    # ── Case 2: continuous digit string ───────────────────────────────────────
    digits = re.sub(r"\D", "", t)

    # 10-digit pattern: OCR misreads "/" as "1"
    # e.g. "2028/03/02" → "2028103102"  (YYYY1MM1DD)
    # e.g. "02/03/2028" → "02103 2028" → "021032028" (DD1MM1YYYY, 9-digit edge case)
    if len(digits) == 10:
        # Try YYYY1MM1DD  (positions 4 and 7 are the misread slashes)
        if digits[4] == "1" and digits[7] == "1":
            r = _try_ymd(digits[:4], digits[5:7], digits[8:10])
            if r:
                return r
        # Try DD1MM1YYYY  (positions 2 and 5 are the misread slashes)
        if digits[2] == "1" and digits[5] == "1":
            r = _try_ymd(digits[6:10], digits[3:5], digits[:2])
            if r:
                return r

    if len(digits) >= 8:
        chunk = digits[:8]                     # take the first 8 digits
        # Try YYYYMMDD
        r = _try_ymd(chunk[:4], chunk[4:6], chunk[6:8])
        if r:
            return r
        # Try DDMMYYYY
        r = _try_ymd(chunk[4:8], chunk[2:4], chunk[:2])
        if r:
            return r

    return t if t else None


# ── Per-field processors ──────────────────────────────────────────────────────

def process_id(text: Optional[str]) -> Optional[str]:
    """Egyptian National ID: exactly 14 digits."""
    if not text:
        return None
    digits = re.sub(r"\D", "", normalize_numerals(strip_bidi(text)))
    if len(digits) in (13, 14):
        return digits
    return digits if digits else None


def process_serial(text: Optional[str]) -> Optional[str]:
    """Serial number: alphanumeric (may contain letters like FI2430244)."""
    if not text:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9٠-٩]", "", strip_bidi(text))
    cleaned = normalize_numerals(cleaned)
    return cleaned if cleaned else None


def process_date(text: Optional[str]) -> Optional[str]:
    return fix_date(text) if text else None


def process_gender(text: Optional[str]) -> Optional[str]:
    return fuzzy_vocab_match(strip_bidi(text), GENDER_VALUES) if text else None


def process_religion(text: Optional[str]) -> Optional[str]:
    return fuzzy_vocab_match(strip_bidi(text), RELIGION_VALUES) if text else None


def process_status(text: Optional[str]) -> Optional[str]:
    return fuzzy_vocab_match(strip_bidi(text), STATUS_VALUES) if text else None


def process_name(text: Optional[str]) -> Optional[str]:
    """
    Arabic names: keep Arabic letters + spaces only.
    Rejects strings that are mostly Latin/noise.
    """
    if not text:
        return None
    t = strip_bidi(text)
    # Keep Arabic Unicode block + spaces + hyphen
    cleaned = re.sub(r"[^؀-ۿ\s\-]", "", t).strip()
    # Remove repeated whitespace
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    # Reject if result is too short or not mostly Arabic
    if len(cleaned) < 2 or not is_mostly_arabic(cleaned):
        return None
    return cleaned


def process_address(text: Optional[str]) -> Optional[str]:
    """Addresses: Arabic text + digits. Light cleanup only."""
    if not text:
        return None
    t = strip_bidi(text)
    # Normalize numerals but keep Arabic letters, digits, spaces, common punctuation
    t = normalize_numerals(t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t if len(t) >= 2 else None


# ── Dispatch table ────────────────────────────────────────────────────────────

FIELD_PROCESSORS = {
    "ID":          process_id,
    "Serial_Num":  process_serial,
    "IssueDate":   process_date,
    "ExpDate":     process_date,
    "Gender":      process_gender,
    "Religion":    process_religion,
    "Status":      process_status,
    "First_Name":  process_name,
    "Last_Name":   process_name,
    "HusbandName": process_name,
    "Job":         process_name,
    "Add1":        process_address,
    "Add2":        process_address,
    "Front":       lambda t: t,
    "Back":        lambda t: t,
}


def postprocess_fields(fields: dict) -> dict:
    """Apply field-specific post-processing to all extracted fields."""
    cleaned = {}
    for field, text in fields.items():
        processor = FIELD_PROCESSORS.get(field, lambda t: t)
        cleaned[field] = processor(text)
    return cleaned
