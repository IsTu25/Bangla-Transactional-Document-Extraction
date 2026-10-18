"""Eval package initialization."""
from eval.normalizer import normalize_digits, normalize_bangla_unicode, normalize_currency_amount, normalize_date
from eval.taxonomy import FailureType, FieldVerificationDetail, DocumentVerificationReport
from eval.verifier import SpanGroundedVerifier
from eval.metrics import BenchmarkMetricsCalculator

__all__ = [
    "normalize_digits",
    "normalize_bangla_unicode",
    "normalize_currency_amount",
    "normalize_date",
    "FailureType",
    "FieldVerificationDetail",
    "DocumentVerificationReport",
    "SpanGroundedVerifier",
    "BenchmarkMetricsCalculator",
]
