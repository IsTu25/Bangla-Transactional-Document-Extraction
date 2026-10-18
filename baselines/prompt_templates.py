"""
Baseline Prompt Templates for Bangla Transactional Document KIE.

Provides:
- Zero-Shot Prompting
- Few-Shot Prompting
- Constrained Span-Grounded Extraction Prompting
"""

import json

SYSTEM_PROMPT = """You are an expert Document Information Extraction AI specialized in Bangla transactional documents (utility bills, mobile money receipts, and retail invoices).

Your task is to extract structured, schema-conformant JSON where EVERY field is grounded with a character-level source span.

### Strict Grounding Rules:
1. Every field MUST be an object with two keys: "value" and "source_span".
2. "source_span" MUST contain:
   - "start": exact 0-indexed character start position in the provided OCR text.
   - "end": exact 0-indexed character end position (exclusive) in the provided OCR text.
   - "text": the exact verbatim substring at OCR_TEXT[start:end].
3. If a field is not present in the document, set both "value": null and "source_span": null.
4. "value" should be normalized (Bangla digits converted to standard Arabic numbers, currency symbols stripped, dates in YYYY-MM-DD).
5. DO NOT invent, hallucinate, or assume any values not visible in the text.
"""

USER_TEMPLATE = """### Document Type: {doc_type}

### Target JSON Schema:
{schema_str}

### Input OCR Text:
\"\"\"
{ocr_text}
\"\"\"

Extract the grounded JSON object now:"""


def get_extraction_prompt(doc_type: str, ocr_text: str, schema_dict: dict) -> list:
    """Format chat messages for LLM inference."""
    schema_clean = {k: v for k, v in schema_dict.items() if k not in ("$schema", "$id")}
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                doc_type=doc_type,
                schema_str=json.dumps(schema_clean, indent=2, ensure_ascii=False),
                ocr_text=ocr_text,
            ),
        },
    ]
