"""
Failure Mining Module for the Bangla Document KIE Flywheel.

Step 3 of the Core Method:
Parses evaluation reports, clusters failures by taxonomy, ranks error patterns by frequency,
and generates targeted synthesis specifications for Phase 6.
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from rich.console import Console
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.verifier import SpanGroundedVerifier
from eval.taxonomy import FailureType

console = Console()


def mine_failures(test_file: Path, predictions_file: Path, output_mining_spec: Path):
    console.print(f"[bold cyan]Mining Failure Clusters for Data Flywheel...[/bold cyan]")

    with open(test_file, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    with open(predictions_file, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    pred_map = {item["doc_id"]: item.get("prediction", {}) for item in predictions}

    verifier = SpanGroundedVerifier()
    failure_clusters = defaultdict(list)
    field_failure_counts = defaultdict(lambda: defaultdict(int))
    doc_type_failure_counts = defaultdict(lambda: defaultdict(int))

    for doc in ground_truth:
        doc_id = doc["doc_id"]
        doc_type = doc["document_type"]
        ocr_text = doc["ocr_text"]
        gold_json = doc["ground_truth"]
        pred_json = pred_map.get(doc_id, {})

        report = verifier.verify_document(
            ocr_text=ocr_text,
            prediction=pred_json,
            doc_type=doc_type,
            doc_id=doc_id,
            gold_reference=gold_json,
        )

        for fld_name, detail in report.field_details.items():
            if detail.status != FailureType.VALID:
                failure_info = {
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                    "field_name": fld_name,
                    "failure_type": detail.status.value,
                    "score": detail.score,
                    "predicted_value": detail.predicted_value,
                    "gold_value": detail.gold_value,
                    "notes": detail.notes,
                }
                failure_clusters[detail.status.value].append(failure_info)
                field_failure_counts[fld_name][detail.status.value] += 1
                doc_type_failure_counts[doc_type][detail.status.value] += 1

    # Display Mining Summary Table
    table = Table(title="Flywheel Mined Failure Distribution", header_style="bold red")
    table.add_column("Failure Taxonomy", style="yellow")
    table.add_column("Count", justify="right")
    table.add_column("Recommended Flywheel Remedy", style="cyan")

    remedy_map = {
        FailureType.WRONG_SPAN.value: "Synthesize jittered/misaligned OCR spans and multi-token boundary cases",
        FailureType.VALUE_SPAN_MISMATCH.value: "Synthesize diverse Bangla/Western digit & currency variants (৳, Tk, /-)",
        FailureType.HALLUCINATED.value: "Inject negative distractors, decoy phone numbers, and ungrounded labels",
        FailureType.MISSING_FIELD.value: "Synthesize dense documents where the target field appears with varied headers",
        FailureType.SCHEMA_VIOLATION.value: "Add constrained decoding grammar (Outlines/xgrammar) and format SFT examples",
        FailureType.FORMAT_ERROR.value: "Increase SFT JSON formatting weight and add strict JSON parsing reward in GRPO",
    }

    for ft_enum in FailureType:
        if ft_enum == FailureType.VALID:
            continue
        cnt = len(failure_clusters.get(ft_enum.value, []))
        remedy = remedy_map.get(ft_enum.value, "General synthetic augmentation")
        table.add_row(ft_enum.value, str(cnt), remedy)

    console.print(table)

    # Save targeted synthesis profile
    mining_profile = {
        "total_failures_mined": sum(len(v) for v in failure_clusters.values()),
        "failure_clusters": {k: len(v) for k, v in failure_clusters.items()},
        "top_failing_fields": {
            fld: dict(counts) for fld, counts in sorted(
                field_failure_counts.items(), key=lambda x: sum(x[1].values()), reverse=True
            )[:10]
        },
        "breakdown_by_doc_type": {dt: dict(c) for dt, c in doc_type_failure_counts.items()},
        "sampled_failure_examples": {k: v[:5] for k, v in failure_clusters.items()},
    }

    output_mining_spec.parent.mkdir(parents=True, exist_ok=True)
    with open(output_mining_spec, "w", encoding="utf-8") as f:
        json.dump(mining_profile, f, indent=2, ensure_ascii=False)

    console.print(f"[bold green]Saved mining specification to {output_mining_spec}[/bold green]")
    return mining_profile


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mine failure patterns from evaluation logs.")
    parser.add_argument("--test-file", type=Path, default=Path("data/test/gold_test_samples.json"))
    parser.add_argument("--predictions", type=Path, default=Path("results/baseline_predictions.json"))
    parser.add_argument("--output", type=Path, default=Path("results/mined_failures.json"))
    args = parser.parse_args()

    mine_failures(args.test_file, args.predictions, args.output)
