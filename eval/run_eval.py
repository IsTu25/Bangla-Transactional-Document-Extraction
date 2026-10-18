"""
Evaluation CLI Harness for Bangla Document KIE.

Runs span-grounded verifier across test documents, computes metrics, prints rich tables,
and saves diagnostic evaluation reports to results/.
"""

import sys
import json
import argparse
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rich.console import Console
from rich.table import Table

from eval.verifier import SpanGroundedVerifier
from eval.metrics import BenchmarkMetricsCalculator

console = Console()


def run_evaluation(test_file: Path, predictions_file: Path, output_file: Path):
    console.print(f"[bold cyan]Starting Bangla KIE Evaluation...[/bold cyan]")
    console.print(f"Loading Ground Truth: [yellow]{test_file}[/yellow]")
    console.print(f"Loading Predictions:  [yellow]{predictions_file}[/yellow]")

    with open(test_file, "r", encoding="utf-8") as f:
        ground_truth_data = json.load(f)

    with open(predictions_file, "r", encoding="utf-8") as f:
        predictions_data = json.load(f)

    verifier = SpanGroundedVerifier()
    reports = []

    # Map predictions by doc_id
    pred_map = {item.get("doc_id", f"doc_{i}"): item for i, item in enumerate(predictions_data)}

    for item in ground_truth_data:
        doc_id = item.get("doc_id", "unknown")
        doc_type = item.get("document_type", "utility_bill")
        ocr_text = item.get("ocr_text", "")
        gold_json = item.get("ground_truth", {})

        pred_entry = pred_map.get(doc_id, {})
        pred_json = pred_entry.get("prediction", {})

        report = verifier.verify_document(
            ocr_text=ocr_text,
            prediction=pred_json,
            doc_type=doc_type,
            doc_id=doc_id,
            gold_reference=gold_json,
        )
        reports.append(report)

    calculator = BenchmarkMetricsCalculator(reports)
    summary = calculator.compute_all_metrics()

    # Display Rich Terminal Table
    table = Table(title="Bangla KIE Benchmark Results", header_style="bold magenta")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Score / Rate", style="green")
    table.add_column("Evaluation Target", style="white")

    table.add_row("Evaluated Documents", str(summary["num_evaluated_documents"]), "Gold Test Split")
    table.add_row("Mean Verifier Score", f"{summary['mean_verifier_score']:.4f}", "Target: > 0.8500")
    table.add_row("Macro Field F1", f"{summary['macro_field_f1']:.4f}", "Target: > 0.8800")
    table.add_row("Precision", f"{summary['precision']:.4f}", "Field-level Grounding")
    table.add_row("Recall", f"{summary['recall']:.4f}", "Completeness")
    table.add_row("ANLS (Levenshtein)", f"{summary['anls']:.4f}", "OCR-Noise Tolerant")
    table.add_row("Strict Span Match Rate", f"{summary['strict_span_grounding_rate']:.4f}", "Verbatim Substring")
    table.add_row("JSON Validity Rate", f"{summary['json_validity_rate'] * 100:.1f}%", "100.0% Required")
    table.add_row("Schema Conformance", f"{summary['schema_conformance_rate'] * 100:.1f}%", "Target: > 98.0%")
    table.add_row("Hallucination Rate", f"{summary['hallucination_rate'] * 100:.2f}%", "Target: < 2.0%")

    console.print(table)

    # Save detailed evaluation artifacts
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    console.print(f"[bold green]Saved benchmark evaluation to {output_file}[/bold green]")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Bangla Document KIE model predictions.")
    parser.add_argument("--test-file", type=Path, default=Path("data/test/gold_test_samples.json"))
    parser.add_argument("--predictions", type=Path, default=Path("data/samples/sample_predictions.json"))
    parser.add_argument("--output", type=Path, default=Path("results/eval_report.json"))
    args = parser.parse_args()

    run_evaluation(args.test_file, args.predictions, args.output)
