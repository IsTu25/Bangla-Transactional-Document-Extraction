# Real Bengali Document Dataset Directory

This directory is designated for the **real-world benchmark dataset** (target: 120 documents, split into 40 dev and 80 test).

---

## 🔒 Privacy & Redaction Guidelines

1. **Strict Anonymization Before Saving:**
   - Names: replace or blur real personal names with generic names (e.g. মোঃ করিম, মোছাঃ রহিমা).
   - Phone Numbers: mask sender/recipient numbers as `017XXXXXXX`.
   - Account / Meter Numbers: replace with randomized digits.
   - **Never commit raw, unredacted original documents or images containing personally identifiable information (PII) to GitHub!**

---

## 📁 Directory Layout

```
data/real/
├── dev/                     # 40 documents (used exclusively for error mining & dev tuning)
├── test/                    # 80 documents (held-out test set; untouched until final evaluation)
├── manifest.jsonl           # Metadata manifest linking documents, OCR engines, and domains
└── manifest_template.jsonl  # Reference schema for entries in manifest.jsonl
```

---

## 📄 Manifest Schema (`manifest_template.jsonl`)

Each line in `manifest.jsonl` follows this structure:

```json
{
  "doc_id": "real_doc_001",
  "document_type": "utility_bill",
  "provider": "DESCO",
  "ocr_engine": "tesseract_v5_bangla",
  "split": "dev",
  "anonymized": true,
  "file_path": "data/real/dev/real_doc_001.json",
  "date_collected": "2026-10-18"
}
```
