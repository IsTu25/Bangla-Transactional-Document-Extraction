"""
Span-Grounded Verifier: The Heart of the Project.

Performs:
1. JSON Syntactic & Structural Validation.
2. JSON Schema Conformance (via jsonschema).
3. Span Grounding Verification (Exact character slice in raw OCR text).
4. Value-Span Semantic & Numeric Consistency Check.
5. Hallucination & Missing Field Detection.
6. Diagnostic Failure Taxonomy Classification & Scalar Scoring.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import jsonschema

from eval.normalizer import (
    normalize_bangla_unicode,
    normalize_digits,
    normalize_currency_amount,
    normalize_date,
    normalized_levenshtein_similarity,
)
from eval.taxonomy import FailureType, FieldVerificationDetail, DocumentVerificationReport

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"


class SpanGroundedVerifier:
    def __init__(self, schema_dir: Optional[Path] = None):
        self.schema_dir = Path(schema_dir) if schema_dir else SCHEMA_DIR
        self.schemas: Dict[str, dict] = {}
        self._load_schemas()

    def _load_schemas(self):
        """Pre-load JSON schemas for document types and meta-schema into local store."""
        self.schema_store: Dict[str, dict] = {}
        for filename in ("meta_schema.json", "utility_bill.json", "mfs_receipt.json", "shop_invoice.json"):
            path = self.schema_dir / filename
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    schema_data = json.load(f)
                    self.schema_store[filename] = schema_data
                    schema_id = schema_data.get("$id")
                    if schema_id:
                        self.schema_store[schema_id] = schema_data

        self.schemas = {
            "utility_bill": self.schema_store.get("utility_bill.json", {}),
            "mfs_receipt": self.schema_store.get("mfs_receipt.json", {}),
            "shop_invoice": self.schema_store.get("shop_invoice.json", {}),
        }

    def validate_schema(self, data: dict, doc_type: str) -> Tuple[bool, List[str]]:
        """Validate candidate JSON against the corresponding doc_type schema using local store."""
        if doc_type not in self.schemas or not self.schemas[doc_type]:
            return False, [f"Unknown document type: {doc_type}"]

        schema = self.schemas[doc_type]
        resolver = jsonschema.RefResolver(
            base_uri=f"file://{self.schema_dir.as_posix()}/",
            referrer=schema,
            store=self.schema_store,
        )
        validator = jsonschema.Draft7Validator(
            schema,
            resolver=resolver,
        )
        errors = [err.message for err in validator.iter_errors(data)]
        return len(errors) == 0, errors

    def verify_single_field(
        self,
        field_name: str,
        pred_field: Any,
        ocr_text: str,
        gold_field: Optional[Any] = None,
    ) -> FieldVerificationDetail:
        """
        Verify a single span-grounded field.
        Returns a FieldVerificationDetail with status, score, and diagnostics.
        """
        # 1. Check if field is completely missing in prediction
        if pred_field is None:
            if gold_field is not None and gold_field.get("value") is not None:
                return FieldVerificationDetail(
                    field_name=field_name,
                    status=FailureType.MISSING_FIELD,
                    score=0.0,
                    predicted_value=None,
                    gold_value=gold_field.get("value"),
                    notes="Field is present in ground truth but missing from prediction.",
                )
            # Correctly omitted or null
            return FieldVerificationDetail(
                field_name=field_name,
                status=FailureType.VALID,
                score=1.0,
                predicted_value=None,
                gold_value=gold_field.get("value") if gold_field else None,
                notes="Field is null and expected to be null.",
            )

        if not isinstance(pred_field, dict):
            return FieldVerificationDetail(
                field_name=field_name,
                status=FailureType.FORMAT_ERROR,
                score=0.0,
                predicted_value=pred_field,
                notes="Field is not formatted as a span-grounded object {value, source_span}.",
            )

        pred_val = pred_field.get("value")
        pred_span = pred_field.get("source_span")

        # 2. Both value and span null: valid if gold is also null
        if pred_val is None and pred_span is None:
            if gold_field and gold_field.get("value") is not None:
                return FieldVerificationDetail(
                    field_name=field_name,
                    status=FailureType.MISSING_FIELD,
                    score=0.0,
                    predicted_value=None,
                    gold_value=gold_field.get("value"),
                    notes="Predicted null, but ground truth has a value.",
                )
            return FieldVerificationDetail(
                field_name=field_name,
                status=FailureType.VALID,
                score=1.0,
                predicted_value=None,
                gold_value=None,
                notes="Correctly null.",
            )

        # 3. Value exists but span is missing -> automatic HALLUCINATION
        if pred_val is not None and (pred_span is None or not isinstance(pred_span, dict)):
            return FieldVerificationDetail(
                field_name=field_name,
                status=FailureType.HALLUCINATED,
                score=0.0,
                predicted_value=pred_val,
                predicted_span=None,
                gold_value=gold_field.get("value") if gold_field else None,
                notes="Value was generated without any source span grounding.",
            )

        start = pred_span.get("start")
        end = pred_span.get("end")
        span_text = pred_span.get("text", "")

        # 4. Verify span bounds in raw OCR text
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(ocr_text)):
            return FieldVerificationDetail(
                field_name=field_name,
                status=FailureType.WRONG_SPAN,
                score=0.0,
                predicted_value=pred_val,
                predicted_span=pred_span,
                gold_value=gold_field.get("value") if gold_field else None,
                notes=f"Span indices [{start}:{end}] are out of bounds for OCR text of length {len(ocr_text)}.",
            )

        actual_slice = ocr_text[start:end]
        is_exact_span = actual_slice == span_text

        # If not exact slice, check if text exists elsewhere in OCR
        if not is_exact_span:
            norm_actual = normalize_bangla_unicode(actual_slice)
            norm_span = normalize_bangla_unicode(span_text)
            if norm_actual == norm_span:
                is_exact_span = True
            elif span_text not in ocr_text:
                return FieldVerificationDetail(
                    field_name=field_name,
                    status=FailureType.HALLUCINATED,
                    score=0.0,
                    predicted_value=pred_val,
                    predicted_span=pred_span,
                    gold_value=gold_field.get("value") if gold_field else None,
                    notes=f"Span text '{span_text}' does not exist anywhere in source OCR text.",
                )
            else:
                return FieldVerificationDetail(
                    field_name=field_name,
                    status=FailureType.WRONG_SPAN,
                    score=0.2,
                    predicted_value=pred_val,
                    predicted_span=pred_span,
                    gold_value=gold_field.get("value") if gold_field else None,
                    notes=f"Offset [{start}:{end}] slices '{actual_slice}', expected '{span_text}'.",
                )

        # 5. Value-Span consistency check
        # Check if numeric field
        if isinstance(pred_val, (int, float)):
            span_numeric = normalize_currency_amount(span_text)
            if span_numeric is not None and abs(float(pred_val) - span_numeric) < 0.01:
                value_matches_span = True
            else:
                value_matches_span = False
        else:
            # String field
            norm_val = normalize_digits(normalize_bangla_unicode(str(pred_val))).strip().lower()
            norm_span_text = normalize_digits(normalize_bangla_unicode(span_text)).strip().lower()
            sim = normalized_levenshtein_similarity(norm_val, norm_span_text)
            value_matches_span = (norm_val in norm_span_text) or (norm_span_text in norm_val) or (sim >= 0.85)

        if not value_matches_span:
            return FieldVerificationDetail(
                field_name=field_name,
                status=FailureType.VALUE_SPAN_MISMATCH,
                score=0.2,
                predicted_value=pred_val,
                predicted_span=pred_span,
                span_valid_in_ocr=True,
                gold_value=gold_field.get("value") if gold_field else None,
                notes=f"Value '{pred_val}' cannot be derived from span text '{span_text}'.",
            )

        # 6. If gold reference is available, check agreement with gold
        if gold_field is not None and gold_field.get("value") is not None:
            gold_val = gold_field.get("value")
            if isinstance(gold_val, (int, float)) and isinstance(pred_val, (int, float)):
                match = abs(float(pred_val) - float(gold_val)) < 0.01
            else:
                g_str = normalize_digits(normalize_bangla_unicode(str(gold_val))).lower()
                p_str = normalize_digits(normalize_bangla_unicode(str(pred_val))).lower()
                match = (g_str == p_str) or (normalized_levenshtein_similarity(g_str, p_str) >= 0.90)

            if not match:
                return FieldVerificationDetail(
                    field_name=field_name,
                    status=FailureType.VALUE_SPAN_MISMATCH,
                    score=0.4,
                    predicted_value=pred_val,
                    predicted_span=pred_span,
                    span_valid_in_ocr=True,
                    gold_value=gold_val,
                    notes=f"Predicted '{pred_val}' does not match gold value '{gold_val}'.",
                )

        # Passed all verification tests!
        return FieldVerificationDetail(
            field_name=field_name,
            status=FailureType.VALID,
            score=1.0,
            predicted_value=pred_val,
            predicted_span=pred_span,
            span_valid_in_ocr=True,
            gold_value=gold_field.get("value") if gold_field else None,
            notes="Field span grounded and value verified.",
        )

    def verify_document(
        self,
        ocr_text: str,
        prediction: Any,
        doc_type: str,
        doc_id: str = "unknown",
        gold_reference: Optional[dict] = None,
    ) -> DocumentVerificationReport:
        """
        Run end-to-end verification for a document prediction.
        """
        # Parse if raw string JSON
        if isinstance(prediction, str):
            try:
                prediction_data = json.loads(prediction)
            except Exception as e:
                return DocumentVerificationReport(
                    doc_id=doc_id,
                    document_type=doc_type,
                    is_valid_json=False,
                    passes_schema=False,
                    overall_score=0.0,
                    taxonomy_counts={FailureType.FORMAT_ERROR.value: 1},
                    raw_ocr_length=len(ocr_text),
                )
        else:
            prediction_data = prediction

        # Validate schema
        passes_schema, schema_errors = self.validate_schema(prediction_data, doc_type)

        field_details: Dict[str, FieldVerificationDetail] = {}
        field_scores: Dict[str, float] = {}
        taxonomy_counts: Dict[str, int] = {ft.value: 0 for ft in FailureType}
        hallucination_count = 0

        # Collect union of fields from schema properties, prediction, and gold
        schema_props = set(self.schemas.get(doc_type, {}).get("properties", {}).keys())
        schema_props.discard("document_type")
        pred_keys = set(prediction_data.keys()) if isinstance(prediction_data, dict) else set()
        gold_keys = set(gold_reference.keys()) if isinstance(gold_reference, dict) else set()
        all_fields = sorted(list(schema_props.union(pred_keys).union(gold_keys)))

        for fld in all_fields:
            if fld in ("document_type", "line_items"):
                continue

            pred_fld = prediction_data.get(fld) if isinstance(prediction_data, dict) else None
            gold_fld = gold_reference.get(fld) if isinstance(gold_reference, dict) else None

            detail = self.verify_single_field(
                field_name=fld,
                pred_field=pred_fld,
                ocr_text=ocr_text,
                gold_field=gold_fld,
            )
            field_details[fld] = detail
            field_scores[fld] = detail.score
            taxonomy_counts[detail.status.value] += 1
            if detail.status == FailureType.HALLUCINATED:
                hallucination_count += 1

        # Handle line items if shop_invoice
        if doc_type == "shop_invoice" and "line_items" in schema_props:
            pred_items = prediction_data.get("line_items", []) if isinstance(prediction_data, dict) else []
            gold_items = gold_reference.get("line_items", []) if isinstance(gold_reference, dict) else []
            
            # Simple item count alignment
            if isinstance(pred_items, list):
                for idx, item in enumerate(pred_items):
                    gold_item = gold_items[idx] if idx < len(gold_items) else None
                    if isinstance(item, dict):
                        for sub_fld in ("item_name", "total_price"):
                            detail = self.verify_single_field(
                                field_name=f"line_items[{idx}].{sub_fld}",
                                pred_field=item.get(sub_fld),
                                ocr_text=ocr_text,
                                gold_field=gold_item.get(sub_fld) if gold_item else None,
                            )
                            field_details[detail.field_name] = detail
                            field_scores[detail.field_name] = detail.score
                            taxonomy_counts[detail.status.value] += 1

        total_fields = len(field_scores)
        mean_field_score = (sum(field_scores.values()) / total_fields) if total_fields > 0 else 0.0

        # Penalize if schema failed
        schema_factor = 1.0 if passes_schema else 0.85
        overall_score = round(mean_field_score * schema_factor, 4)

        return DocumentVerificationReport(
            doc_id=doc_id,
            document_type=doc_type,
            is_valid_json=True,
            passes_schema=passes_schema,
            overall_score=overall_score,
            field_scores=field_scores,
            field_details=field_details,
            taxonomy_counts=taxonomy_counts,
            hallucination_count=hallucination_count,
            total_fields=total_fields,
            raw_ocr_length=len(ocr_text),
        )
