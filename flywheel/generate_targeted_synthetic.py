"""
Targeted Synthetic Data Generator for the Bangla Document KIE Flywheel.

Step 4 of the Core Method:
Generates synthetic documents with exact span grounding specifically targeted
at fixing mined model failure modes (digit confusion, OCR noise, distractors).
"""

import random
import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.normalizer import BANGLA_DIGITS, WESTERN_DIGITS

BILLERS = [
    ("ডেসকো (DESCO)", "ডেসকো"),
    ("ডিপিডিসি (DPDC)", "ডিপিডিসি"),
    ("তিতাস গ্যাস টিএন্ডডি কোং লিঃ", "তিতাস গ্যাস"),
    ("নেসকো (NESCO)", "নেসকো"),
    ("ঢাকা ওয়াসা (DWASA)", "ঢাকা ওয়াসা"),
]

MFS_PROVIDERS = [
    ("বিকাশ (bKash)", "বিকাশ"),
    ("নগদ (Nagad)", "নগদ"),
    ("রকেট (Rocket)", "রকেট"),
    ("উপায় (Upay)", "উপায়"),
]

NAMES = [
    "মোঃ রফিকুল ইসলাম", "তানজিলা বেগম", "ফারহানা ইয়াসমিন",
    "আব্দুল কাদের", "মাহমুদুল হাসান", "সুমাইয়া আক্তার",
    "সাইফুল ইসলাম", "নাসরিন জাহান", "শাহরিয়ার কবীর",
]

AREAS = [
    "মিরপুর-১০, ঢাকা", "ধানমন্ডি, রোড # ৮/এ, ঢাকা", "উত্তরা সেক্টর ৭, ঢাকা",
    "গুলশান-২, ঢাকা", "বনানী, ব্লক ডি, ঢাকা", "চকবাজার, চট্টগ্রাম",
    "উপশহর, রাজশাহী", "জিন্দাবাজার, সিলেট",
]


def to_bangla_num(n: int) -> str:
    return str(n).translate(str.maketrans(WESTERN_DIGITS, BANGLA_DIGITS))


def generate_synthetic_utility_bill(target_failure: str = "general") -> Dict[str, Any]:
    biller_full, biller_short = random.choice(BILLERS)
    cust_name = random.choice(NAMES)
    cust_addr = random.choice(AREAS)
    acc_num_int = random.randint(10000000, 99999999)
    acc_num_bn = to_bangla_num(acc_num_int)
    meter_bn = f"MTR-{to_bangla_num(random.randint(100000, 999999))}"
    units = random.randint(80, 650)
    units_bn = to_bangla_num(units)
    amount = units * 7.5
    vat = round(amount * 0.05, 2)
    surcharge = 150.0
    total = amount + vat
    total_late = total + surcharge

    # Currency format variations
    if target_failure == "VALUE_SPAN_MISMATCH":
        amt_str = f"৳ {to_bangla_num(int(total))}/="
        clean_amt_span = to_bangla_num(int(total))
    else:
        amt_str = f"{to_bangla_num(int(total))}.০০ টাকা"
        clean_amt_span = f"{to_bangla_num(int(total))}.০০"

    # Distractor injection if target is HALLUCINATION
    distractor_line = ""
    if target_failure == "HALLUCINATED":
        distractor_line = f"হেল্পলাইন: ১৬১৬৭ | ব্যাংক শাখা ট্রানজেকশন রেফারেন্স: {to_bangla_num(random.randint(50000000, 99999999))}\n"

    ocr_text = (
        f"{biller_full}\n"
        f"গ্রাহক বিল বিবরণী - সেপ্টেম্বর ২০২৪\n"
        f"{distractor_line}"
        f"গ্রাহক হিসাব নম্বর: {acc_num_bn}\n"
        f"মিটার নং: {meter_bn}\n"
        f"গ্রাহকের নাম: {cust_name}\n"
        f"ঠিকানা: {cust_addr}\n"
        f"ইস্যু তারিখ: ০৫/০৯/২০২৪\n"
        f"বিল পরিশোধের শেষ তারিখ: ২৫/০৯/২০২৪\n"
        f"মোট ব্যবহৃত ইউনিট: {units_bn}\n"
        f"সর্বমোট প্রদেয় টাকা: {amt_str}\n"
        f"বিলম্ব মাসুলসহ প্রদেয়: {to_bangla_num(int(total_late))}.০০ টাকা\n"
    )

    def get_span(target: str):
        idx = ocr_text.find(target)
        if idx == -1:
            return None
        return {"start": idx, "end": idx + len(target), "text": target}

    ground_truth = {
        "document_type": "utility_bill",
        "biller_name": {"value": biller_short, "source_span": get_span(biller_short)},
        "customer_account_no": {"value": str(acc_num_int), "source_span": get_span(acc_num_bn)},
        "meter_number": {"value": meter_bn, "source_span": get_span(meter_bn)},
        "bill_month": {"value": "সেপ্টেম্বর ২০২৪", "source_span": get_span("সেপ্টেম্বর ২০২৪")},
        "issue_date": {"value": "2024-09-05", "source_span": get_span("০৫/০৯/২০২৪")},
        "due_date": {"value": "2024-09-25", "source_span": get_span("২৫/০৯/২০২৪")},
        "total_units_consumed": {"value": float(units), "source_span": get_span(units_bn)},
        "current_charge": {"value": round(amount, 2), "source_span": None},
        "vat_tax": {"value": round(vat, 2), "source_span": None},
        "late_surcharge": {"value": surcharge, "source_span": None},
        "total_payable_amount": {"value": float(int(total)), "source_span": get_span(clean_amt_span)},
        "payable_after_due_date": {"value": float(int(total_late)), "source_span": get_span(f"{to_bangla_num(int(total_late))}.০০")},
        "customer_name": {"value": cust_name, "source_span": get_span(cust_name)},
        "customer_address": {"value": cust_addr, "source_span": get_span(cust_addr)},
    }

    return {
        "doc_id": f"syn_util_{random.randint(1000, 9999)}",
        "document_type": "utility_bill",
        "ocr_text": ocr_text,
        "ground_truth": ground_truth,
        "targeted_failure_remedy": target_failure,
    }


def generate_targeted_dataset(num_samples: int, target_failure: str, output_path: Path):
    print(f"Generating {num_samples} targeted synthetic samples for remedy: {target_failure}")
    samples = []
    for _ in range(num_samples):
        samples.append(generate_synthetic_utility_bill(target_failure=target_failure))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"Successfully generated dataset at: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate targeted synthetic data.")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--target-failure", type=str, default="VALUE_SPAN_MISMATCH",
                        choices=["VALUE_SPAN_MISMATCH", "HALLUCINATED", "WRONG_SPAN", "general"])
    parser.add_argument("--output", type=Path, default=Path("data/synthetic/targeted_batch_v1.json"))
    args = parser.parse_args()

    generate_targeted_dataset(args.count, args.target_failure, args.output)
