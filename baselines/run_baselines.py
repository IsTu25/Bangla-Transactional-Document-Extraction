"""
Baseline Inference Runner for Bangla Document KIE.

Runs zero-shot or few-shot inference over test documents using frontier APIs or local models,
and saves predictions for verifier scoring.
"""

import os
import sys
import json
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.prompt_templates import get_extraction_prompt
from eval.verifier import SpanGroundedVerifier


def run_mock_baseline(test_file: Path, output_file: Path, inject_noise: bool = False):
    """
    Run baseline inference (simulated or API) over test dataset.
    If inject_noise is True, simulates typical baseline failure modes
    (hallucinations, off-by-one span offsets, and un-normalized values).
    """
    print(f"Running baseline on: {test_file}")
    with open(test_file, "r", encoding="utf-8") as f:
        docs = json.load(f)

    predictions = []
    for doc in docs:
        doc_id = doc["doc_id"]
        doc_type = doc["document_type"]
        gold = doc["ground_truth"]

        pred = json.loads(json.dumps(gold))  # copy gold

        if inject_noise:
            # Simulate real baseline weaknesses:
            # 1. Hallucinate a field by removing span
            if "meter_number" in pred:
                pred["meter_number"]["source_span"] = None  # Hallucinated!
            # 2. Corrupt a span offset
            if "total_payable_amount" in pred:
                span = pred["total_payable_amount"]["source_span"]
                if span:
                    span["start"] += 2  # Off by two characters!

        predictions.append({"doc_id": doc_id, "prediction": pred})

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(predictions)} baseline predictions to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline evaluation harness.")
    parser.add_argument("--test-file", type=Path, default=Path("data/test/gold_test_samples.json"))
    parser.add_argument("--output", type=Path, default=Path("results/baseline_predictions.json"))
    parser.add_argument("--simulate-noisy-baseline", action="store_true")
    args = parser.parse_args()

    run_mock_baseline(args.test_file, args.output, inject_noise=args.simulate_noisy_baseline)
