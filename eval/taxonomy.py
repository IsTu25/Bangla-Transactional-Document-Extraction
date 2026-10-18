"""
Failure Taxonomy and Evaluation Data Structures for Bangla Document KIE.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


class FailureType(str, Enum):
    VALID = "VALID"
    FORMAT_ERROR = "FORMAT_ERROR"
    SCHEMA_VIOLATION = "SCHEMA_VIOLATION"
    WRONG_SPAN = "WRONG_SPAN"
    VALUE_SPAN_MISMATCH = "VALUE_SPAN_MISMATCH"
    HALLUCINATED = "HALLUCINATED"
    MISSING_FIELD = "MISSING_FIELD"
    OCR_CORRUPTED = "OCR_CORRUPTED"


@dataclass
class FieldVerificationDetail:
    field_name: str
    status: FailureType
    score: float
    predicted_value: Any
    gold_value: Any = None
    predicted_span: Optional[Dict[str, Any]] = None
    gold_span: Optional[Dict[str, Any]] = None
    span_valid_in_ocr: bool = False
    notes: str = ""


@dataclass
class DocumentVerificationReport:
    doc_id: str
    document_type: str
    is_valid_json: bool
    passes_schema: bool
    overall_score: float
    field_scores: Dict[str, float] = field(default_factory=dict)
    field_details: Dict[str, FieldVerificationDetail] = field(default_factory=dict)
    taxonomy_counts: Dict[str, int] = field(default_factory=dict)
    hallucination_count: int = 0
    total_fields: int = 0
    raw_ocr_length: int = 0
