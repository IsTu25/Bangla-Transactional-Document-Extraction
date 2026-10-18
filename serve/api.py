"""
Production FastAPI Serving API for Bangla Transactional Document KIE.

Endpoints:
- POST /api/v1/extract: Extracts and automatically verifies span-grounded JSON from OCR text.
- POST /api/v1/verify: Validates arbitrary prediction against raw OCR text.
- GET  /api/v1/schemas/{doc_type}: Retrieves the active JSON schema.
- GET  /api/v1/health: Service liveness check.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, HTTPException
from eval.verifier import SpanGroundedVerifier

app = FastAPI(
    title="Bangla Transactional-Document Extraction API",
    description="Span-grounded JSON extraction with an integrated verifier engine for Bangla utility bills, MFS receipts, and shop invoices.",
    version="1.0.0",
)

verifier = SpanGroundedVerifier()


class ExtractionRequest(BaseModel):
    document_type: str = Field(..., description="utility_bill, mfs_receipt, or shop_invoice")
    ocr_text: str = Field(..., description="Raw unstructured OCR text extracted from document image")
    doc_id: Optional[str] = Field("api_req_001", description="Client document identifier")


class VerificationRequest(BaseModel):
    document_type: str
    ocr_text: str
    prediction: Dict[str, Any]
    doc_id: Optional[str] = "verify_req_001"


@app.get("/api/v1/health")
def health():
    return {
        "status": "healthy",
        "service": "bangla-doc-kie-engine",
        "supported_doc_types": list(verifier.schemas.keys()),
    }


@app.get("/api/v1/schemas/{doc_type}")
def get_schema(doc_type: str):
    if doc_type not in verifier.schemas:
        raise HTTPException(status_code=404, detail=f"Schema not found for document type: {doc_type}")
    return verifier.schemas[doc_type]


@app.post("/api/v1/verify")
def verify_document(req: VerificationRequest):
    if req.document_type not in verifier.schemas:
        raise HTTPException(status_code=400, detail=f"Unsupported document type: {req.document_type}")

    report = verifier.verify_document(
        ocr_text=req.ocr_text,
        prediction=req.prediction,
        doc_type=req.document_type,
        doc_id=req.doc_id,
    )

    return {
        "doc_id": report.doc_id,
        "document_type": report.document_type,
        "overall_verifier_score": report.overall_score,
        "is_valid_json": report.is_valid_json,
        "passes_schema": report.passes_schema,
        "hallucination_count": report.hallucination_count,
        "taxonomy_breakdown": report.taxonomy_counts,
        "field_details": {
            k: {
                "status": v.status.value,
                "score": v.score,
                "predicted_value": v.predicted_value,
                "predicted_span": v.predicted_span,
                "notes": v.notes,
            }
            for k, v in report.field_details.items()
        },
    }


@app.post("/api/v1/extract")
def extract_and_verify(req: ExtractionRequest):
    """
    Main extraction endpoint. In full production with fine-tuned model checkpoint,
    this invokes model inference and immediately runs the verifier before returning.
    """
    if req.document_type not in verifier.schemas:
        raise HTTPException(status_code=400, detail=f"Unsupported document type: {req.document_type}")

    # For demonstration / zero-shot serving, runs rule-assisted candidate matching
    # or wraps downstream Qwen2.5-3B model
    from eval.normalizer import normalize_digits, normalize_currency_amount
    import re

    extracted = {"document_type": req.document_type}
    text = req.ocr_text

    def find_span(query):
        idx = text.find(query)
        if idx != -1:
            return {"start": idx, "end": idx + len(query), "text": query}
        return None

    # Heuristic fallback extractor for standard demo flow
    if req.document_type == "utility_bill":
        # Find biller
        for b in ["ডেসকো", "ডিপিডিসি", "নেসকো", "তিতাস"]:
            if b in text:
                extracted["biller_name"] = {"value": b, "source_span": find_span(b)}
                break

        # Find total payable
        match = re.search(r"(\d+(?:,\d+)*(?:\.\d+)?|[০-৯]+(?:,[০-৯]+)*(?:\.[০-৯]+)?)\s*(?:টাকা|৳)", text)
        if match:
            span_str = match.group(0)
            extracted["total_payable_amount"] = {
                "value": normalize_currency_amount(span_str),
                "source_span": find_span(span_str),
            }

    # Verify extracted candidate
    report = verifier.verify_document(
        ocr_text=req.ocr_text,
        prediction=extracted,
        doc_type=req.document_type,
        doc_id=req.doc_id,
    )

    return {
        "doc_id": req.doc_id,
        "document_type": req.document_type,
        "extraction": extracted,
        "verifier_telemetry": {
            "overall_score": report.overall_score,
            "passes_schema": report.passes_schema,
            "hallucination_count": report.hallucination_count,
            "taxonomy_counts": report.taxonomy_counts,
        },
    }
