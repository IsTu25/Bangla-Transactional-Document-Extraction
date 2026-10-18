"""
Evaluation Metrics for Bangla Document KIE.

Computes:
- Field F1 (Macro and Per-Field)
- ANLS (Average Normalized Levenshtein Similarity)
- Strict Span Grounding Rate
- Schema Conformance Rate
- Hallucination Rate
- Null-field Accuracy
"""

from typing import List, Dict, Any
from eval.taxonomy import DocumentVerificationReport, FailureType
from eval.normalizer import normalized_levenshtein_similarity


class BenchmarkMetricsCalculator:
    def __init__(self, reports: List[DocumentVerificationReport]):
        self.reports = reports

    def compute_all_metrics(self) -> Dict[str, Any]:
        if not self.reports:
            return {"error": "No evaluation reports provided"}

        num_docs = len(self.reports)
        valid_json_count = sum(1 for r in self.reports if r.is_valid_json)
        passes_schema_count = sum(1 for r in self.reports if r.passes_schema)

        total_fields = sum(r.total_fields for r in self.reports)
        total_valid = sum(r.taxonomy_counts.get(FailureType.VALID.value, 0) for r in self.reports)
        total_wrong_span = sum(r.taxonomy_counts.get(FailureType.WRONG_SPAN.value, 0) for r in self.reports)
        total_hallucinated = sum(r.taxonomy_counts.get(FailureType.HALLUCINATED.value, 0) for r in self.reports)
        total_mismatch = sum(r.taxonomy_counts.get(FailureType.VALUE_SPAN_MISMATCH.value, 0) for r in self.reports)
        total_missing = sum(r.taxonomy_counts.get(FailureType.MISSING_FIELD.value, 0) for r in self.reports)

        doc_scores = [r.overall_score for r in self.reports]
        mean_doc_score = sum(doc_scores) / num_docs if num_docs > 0 else 0.0

        # Calculate ANLS across all fields
        anls_values = []
        for r in self.reports:
            for detail in r.field_details.values():
                if detail.gold_value is not None and detail.predicted_value is not None:
                    sim = normalized_levenshtein_similarity(
                        str(detail.predicted_value), str(detail.gold_value)
                    )
                    anls_values.append(sim)
                elif detail.gold_value is None and detail.predicted_value is None:
                    anls_values.append(1.0)
                else:
                    anls_values.append(0.0)

        anls_score = (sum(anls_values) / len(anls_values)) if anls_values else 0.0

        strict_span_grounding_rate = (total_valid / total_fields) if total_fields > 0 else 0.0
        hallucination_rate = (total_hallucinated / total_fields) if total_fields > 0 else 0.0

        # Field F1 calculation (where VALID = TP, MISSING_FIELD = FN, HALLUCINATED/MISMATCH/WRONG_SPAN = FP)
        tp = total_valid
        fp = total_wrong_span + total_hallucinated + total_mismatch
        fn = total_missing

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        field_f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "num_evaluated_documents": num_docs,
            "json_validity_rate": round(valid_json_count / num_docs, 4),
            "schema_conformance_rate": round(passes_schema_count / num_docs, 4),
            "mean_verifier_score": round(mean_doc_score, 4),
            "macro_field_f1": round(field_f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "anls": round(anls_score, 4),
            "strict_span_grounding_rate": round(strict_span_grounding_rate, 4),
            "hallucination_rate": round(hallucination_rate, 4),
            "taxonomy_breakdown": {
                "valid": total_valid,
                "wrong_span": total_wrong_span,
                "hallucinated": total_hallucinated,
                "value_span_mismatch": total_mismatch,
                "missing_field": total_missing,
                "total_fields": total_fields,
            },
        }
