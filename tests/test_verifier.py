"""
Unit Tests for SpanGroundedVerifier and Normalizers.
"""

import pytest
from eval.normalizer import (
    normalize_digits,
    normalize_bangla_unicode,
    normalize_currency_amount,
    normalize_date,
    normalized_levenshtein_similarity,
)
from eval.verifier import SpanGroundedVerifier
from eval.taxonomy import FailureType


@pytest.fixture
def verifier():
    return SpanGroundedVerifier()


def test_digit_normalization():
    assert normalize_digits("১২৩৪৫৬৭৮৯০") == "1234567890"
    assert normalize_digits("12345", to_western=False) == "১২৩৪৫"


def test_currency_normalization():
    assert normalize_currency_amount("৳ ১,২৫০.৫০/-") == 1250.50
    assert normalize_currency_amount("500 Tk") == 500.0
    assert normalize_currency_amount("টাকা ৩,৬২২.৫০") == 3622.50
    assert normalize_currency_amount(None) is None


def test_date_normalization():
    assert normalize_date("১৭/১০/২০২৪") == "2024-10-17"
    assert normalize_date("17-10-2024") == "2024-10-17"
    assert normalize_date("১৭ অক্টোবর ২০২৪") == "2024-10-17"


def test_levenshtein_similarity():
    assert normalized_levenshtein_similarity("ডেসকো", "ডেসকো") == 1.0
    assert normalized_levenshtein_similarity("ঢাকা", "চট্টগ্রাম") < 0.3


def test_verifier_exact_valid(verifier):
    ocr_text = "ডেসকো বিদ্যুৎ বিল: গ্রাহক নং ১২৩৪৫, মোট প্রদেয় ৩,৫০০ টাকা।"
    start = ocr_text.find("১২৩৪৫")
    pred = {
        "value": "12345",
        "source_span": {"start": start, "end": start + len("১২৩৪৫"), "text": "১২৩৪৫"},
    }
    detail = verifier.verify_single_field("customer_account_no", pred, ocr_text)
    assert detail.status == FailureType.VALID
    assert detail.score == 1.0


def test_verifier_hallucination_detection(verifier):
    ocr_text = "ডেসকো বিদ্যুৎ বিল: গ্রাহক নং ১২৩৪৫"
    # Case 1: Value with fake span not in text
    pred_fake_text = {
        "value": "99999",
        "source_span": {"start": 0, "end": 5, "text": "৯৯৯৯৯"},
    }
    detail1 = verifier.verify_single_field("customer_account_no", pred_fake_text, ocr_text)
    assert detail1.status == FailureType.HALLUCINATED
    assert detail1.score == 0.0

    # Case 2: Value with null span
    pred_no_span = {
        "value": "12345",
        "source_span": None,
    }
    detail2 = verifier.verify_single_field("customer_account_no", pred_no_span, ocr_text)
    assert detail2.status == FailureType.HALLUCINATED
    assert detail2.score == 0.0


def test_verifier_wrong_span(verifier):
    ocr_text = "ডেসকো বিদ্যুৎ বিল: গ্রাহক নং ১২৩৪৫"
    # Offsets out of bounds
    pred = {
        "value": "12345",
        "source_span": {"start": 50, "end": 60, "text": "১২৩৪৫"},
    }
    detail = verifier.verify_single_field("customer_account_no", pred, ocr_text)
    assert detail.status == FailureType.WRONG_SPAN
    assert detail.score == 0.0


def test_verifier_missing_field(verifier):
    ocr_text = "ডেসকো বিদ্যুৎ বিল"
    gold = {"value": "ডেসকো", "source_span": {"start": 0, "end": 5, "text": "ডেসকো"}}
    detail = verifier.verify_single_field("biller_name", None, ocr_text, gold_field=gold)
    assert detail.status == FailureType.MISSING_FIELD
    assert detail.score == 0.0
