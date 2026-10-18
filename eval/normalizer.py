"""
Bangla & English Text Normalizer for Transactional Document KIE.

Handles:
- Bengali digits (০-৯) to Western Arabic digits (0-9) and vice versa.
- Unicode standardization (NFC, ZWJ/ZWNJ cleanup, diacritic de-duplication).
- Currency string parsing (৳, টাকা, Tk, BDT, comma removal).
- Date standardization (Bangla month names, DD/MM/YYYY to ISO YYYY-MM-DD).
- OCR-noise tolerant string distance (Normalized Levenshtein Similarity).
"""

import re
import unicodedata
from typing import Optional, Union

BANGLA_DIGITS = "০১২৩৪৫৬৭৮৯"
WESTERN_DIGITS = "0123456789"
B2W_TRANS = str.maketrans(BANGLA_DIGITS, WESTERN_DIGITS)
W2B_TRANS = str.maketrans(WESTERN_DIGITS, BANGLA_DIGITS)

BANGLA_MONTHS = {
    "জানুয়ারি": "01", "জানুয়ারী": "01", "জানু": "01", "january": "01", "jan": "01",
    "ফেব্রুয়ারি": "02", "ফেব্রুয়ারী": "02", "ফেব্রু": "02", "february": "02", "feb": "02",
    "মার্চ": "03", "march": "03", "mar": "03",
    "এপ্রিল": "04", "april": "04", "apr": "04",
    "মে": "05", "may": "05",
    "জুন": "06", "june": "06", "jun": "06",
    "জুলাই": "07", "july": "07", "jul": "07",
    "আগস্ট": "08", "আগষ্ট": "08", "august": "08", "aug": "08",
    "সেপ্টেম্বর": "09", "সেপ্টে": "09", "september": "09", "sep": "09", "sept": "09",
    "অক্টোবর": "10", "অক্টো": "10", "october": "10", "oct": "10",
    "নভেম্বর": "11", "নভে": "11", "november": "11", "nov": "11",
    "ডিসেম্বর": "12", "ডিসে": "12", "december": "12", "dec": "12"
}


def normalize_bangla_unicode(text: Optional[str]) -> str:
    """Normalize Unicode characters using NFC and clean zero-width artifacts."""
    if not text:
        return ""
    # NFC Canonical Decomposition followed by Canonical Composition
    normalized = unicodedata.normalize("NFC", str(text))
    # Remove Zero-Width Joiner (ZWJ U+200D) and Zero-Width Non-Joiner (ZWNJ U+200C)
    normalized = normalized.replace("\u200c", "").replace("\u200d", "")
    # Normalize multiple whitespace / newlines
    normalized = re.sub(r"[ \t]+", " ", normalized)
    return normalized.strip()


def normalize_digits(text: Optional[str], to_western: bool = True) -> str:
    """Convert digits between Bengali numerals and Western Arabic numerals."""
    if not text:
        return ""
    trans_table = B2W_TRANS if to_western else W2B_TRANS
    return str(text).translate(trans_table)


def normalize_currency_amount(text: Optional[Union[str, int, float]]) -> Optional[float]:
    """
    Extract canonical float amount from noisy currency string.
    Examples:
      '৳ ১,২৫০.৫০/-' -> 1250.50
      '1250 Tk' -> 1250.00
      'টাকা ৫০০' -> 500.00
    """
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)

    s = normalize_digits(str(text))
    # Remove currency identifiers, symbols, and suffixes
    s = re.sub(r"(৳|টাকা|tk\.?|bdt|/[-=]?)", " ", s, flags=re.IGNORECASE)
    # Remove commas used as thousands separators
    s = s.replace(",", "")
    # Extract numerical pattern (e.g. 1234.56 or 1234)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return None
    return None


def normalize_date(text: Optional[str]) -> Optional[str]:
    """
    Normalize various date formats into ISO 8601 YYYY-MM-DD.
    Supports:
      - 17/10/2024, ১৭-১০-২০২৪
      - 17 October 2024, ১৭ অক্টোবর ২০২৪
      - 2024-10-17
    """
    if not text:
        return None
    s = normalize_digits(str(text).strip().lower())

    # Check for direct ISO YYYY-MM-DD
    iso_match = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)
    if iso_match:
        year, month, day = iso_match.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"

    # Check DD/MM/YYYY or DD-MM-YYYY
    dmy_match = re.search(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})", s)
    if dmy_match:
        day, month, year = dmy_match.groups()
        if len(year) == 2:
            year = "20" + year
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"

    # Check named month e.g., '17 অক্টোবর ২০২৪' or '17 Oct 2024'
    for name, month_num in BANGLA_MONTHS.items():
        if name in s:
            day_match = re.search(r"\b(\d{1,2})\b", s)
            year_match = re.search(r"\b(\d{4})\b", s)
            if day_match and year_match:
                day = int(day_match.group(1))
                year = int(year_match.group(1))
                return f"{year:04d}-{month_num}-{day:02d}"

    return s


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute standard Levenshtein edit distance."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def normalized_levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Compute Normalized Levenshtein Similarity (NLS):
    NLS = 1.0 - (dist(s1, s2) / max(len(s1), len(s2)))
    Returns 1.0 for identical strings, 0.0 for completely disjoint strings.
    """
    norm_s1 = normalize_digits(normalize_bangla_unicode(s1)).strip().lower()
    norm_s2 = normalize_digits(normalize_bangla_unicode(s2)).strip().lower()

    if not norm_s1 and not norm_s2:
        return 1.0
    if not norm_s1 or not norm_s2:
        return 0.0

    max_len = max(len(norm_s1), len(norm_s2))
    dist = levenshtein_distance(norm_s1, norm_s2)
    return max(0.0, 1.0 - (dist / max_len))
