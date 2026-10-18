# Bangla Transactional-Document Annotation Guide

**Project:** Span-Grounded JSON Extraction with a Verifier-Driven Flywheel  
**Target Doc Types:** Utility Bills, Mobile Financial Service (MFS) Receipts, Shop POS Invoices  
**Version:** 1.0.0 (October 2026)

---

## 1. Objective

This document defines standardized annotation rules for creating gold-standard datasets for Bangla transactional document Key Information Extraction (KIE). Every extracted field must output:
1. `value`: The canonical, normalized representation of the entity.
2. `source_span`: The exact provenance substring and character indices (`start`, `end`, `text`) locating the value within raw OCR text.

---

## 2. Span Grounding Principles

### 2.1 Verbatim Slicing
* The `source_span.text` must be an **exact verbatim slice** of the input OCR string:
  ```python
  assert ocr_text[source_span["start"] : source_span["end"]] == source_span["text"]
  ```
* Do not fix spelling mistakes, OCR corruptions, or punctuation inside `source_span.text`. Record whatever characters the OCR produced.

### 2.2 Entity Boundaries
* **Labels vs. Values:** Annotate only the **value**, not the field label.
  * *Example:* For `বিল নং: ১২৩৪৫৬৭৮`, annotate `১২৩৪৫৬৭৮` as `customer_account_no`, not `বিল নং: ১২৩৪৫৬৭৮`.
* **Currency Symbols:** If the OCR text says `৳ ১,২৫০.০০` or `1,250.00 Tk`:
  * `source_span.text` can include `৳ ১,২৫০.০০` or `১,২৫০.০০`.
  * `value` must be the normalized float: `1250.00`.
* **Multi-line Values:** For addresses (`customer_address` or `store_address`), spans may span multiple newline characters (`\n`). Ensure character indices encompass all lines cleanly without trailing whitespace.

---

## 3. Normalization Rules

| Category | Raw Span Examples | Normalized `value` | Rule |
| :--- | :--- | :--- | :--- |
| **Bangla Digits** | `১২৩৪৫` | `"12345"` or `12345` | Convert Bengali numerals (`০-৯`) to Western Arabic numerals (`0-9`). |
| **Currency Amounts**| `৳ ৫,৪২০/-`, `5420.00 Tk`, `টাকা ৫৪২০` | `5420.0` | Strip currency words/symbols, commas, and slash hyphens; convert to float. |
| **Dates** | `১৭/১০/২০২৪`, `17-10-2024`, `১৭ অক্টোবর ২০২৪` | `"2024-10-17"` | Standardize to ISO 8601 (`YYYY-MM-DD`). |
| **Phone Numbers** | `+88 01712-345678`, `০১৭১২৩৪৫৬৭৮` | `"01712345678"` | Strip country code `+88`, spaces, and dashes; normalize to 11-digit string. |
| **Boolean / Status**| `সফল`, `গৃহীত`, `Success` | `"SUCCESS"` | Map to standardized uppercase enum. |

---

## 4. Unicode & Character Handling

Bangla text has specific Unicode pitfalls that corrupt string matching if not managed uniformly:

1. **Hasanta & Conjoined Consonants:** Keep standard Unicode decomposed or precomposed forms consistent. Always normalize input strings using NFC (`unicodedata.normalize('NFC', text)`).
2. **Zero-Width Joiners (ZWJ / ZWNJ):**
   * Strip redundant `\u200C` (ZWNJ) and `\u200D` (ZWJ) during value normalization, but preserve them inside `source_span.text` if present in the raw OCR output.
3. **Nukta Variants:** 
   * Distinguish `র` (U+09B0) vs `ড়` (U+09DC) vs `ঢ়` (U+09DD).
   * Ensure `য়` is stored as U+09DF (Bengali Letter Yya) rather than `য` + Nukta.
4. **Punctuation:**
   * Treat Dari (`।` U+0964) as terminal punctuation, matching English period (`.`).

---

## 5. Missing, Unreadable, and Null Fields

* If a field does not exist on the document (e.g., `late_surcharge` on a prepaid token receipt, or `discount_amount` on an invoice with no promotion):
  ```json
  {
    "value": null,
    "source_span": null
  }
  ```
* **Never fabricate or infer values** that lack visual grounding. If a customer name is obscured or omitted, set both `value` and `source_span` to `null`.
* If a model outputs a non-null `value` with a `null` `source_span`, the Verifier classifies it as an automatic **HALLUCINATION** (score = 0.0).

---

## 6. Privacy & PII Redaction Protocol

Before any document or OCR transcript is checked into public repositories or Hugging Face:
1. **Phone Numbers:** Mask middle digits: `017****5678`.
2. **Customer Names:** Replace private individual names with synthetic Bangladeshi names (e.g., `মো: রফিকুল ইসলাম`, `তানজিলা আক্তার`). Biller/merchant names (e.g., `ডেসকো`, `স্বপ্ন সুপারশপ`) do **not** need redaction.
3. **Account & Meter Numbers:** Scramble the middle digits while preserving format length and check-digit length.
4. **Physical Street Addresses:** Replace specific holding/flat numbers with generic ward/road references (e.g., `বাড়ি # ১২, রোড # ৪, ধানমন্ডি, ঢাকা`).

---

## 7. Annotation Workflow & Quality Assurance

1. **Pre-Annotation:** Run base OCR (EasyOCR / PaddleOCR) and generate candidate spans using regex and few-shot Qwen2.5.
2. **Human Verification:** In Label Studio, the human annotator:
   - Adjusts the bounding span to match exact value tokens.
   - Verifies the normalized value conforms to schema type specifications.
3. **Automated Verifier Linting:** Run `eval/verifier.py` against the annotated JSON before marking a document complete. Any span offset mismatch triggers an immediate review.
4. **Dual Review:** 20% of all gold test documents must be independently annotated by two annotators, requiring an inter-annotator agreement (IAA) F1 score $\ge 0.96$.
