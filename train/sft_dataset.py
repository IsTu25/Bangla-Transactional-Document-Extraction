"""
SFT Dataset Formatter for Bangla Span-Grounded KIE.

Converts raw documents and grounded schemas into instruction-tuning conversation format
(compatible with HuggingFace datasets, TRL SFTTrainer, and Qwen ChatML).
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.prompt_templates import SYSTEM_PROMPT, USER_TEMPLATE

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"


def load_schema(doc_type: str) -> dict:
    path = SCHEMA_DIR / f"{doc_type}.json"
    with open(path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    return {k: v for k, v in schema.items() if k not in ("$schema", "$id")}


def format_doc_to_sft_example(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single document dict into a ChatML instruction conversation."""
    doc_type = doc["document_type"]
    ocr_text = doc["ocr_text"]
    ground_truth = doc["ground_truth"]

    schema_dict = load_schema(doc_type)

    user_msg = USER_TEMPLATE.format(
        doc_type=doc_type,
        schema_str=json.dumps(schema_dict, indent=2, ensure_ascii=False),
        ocr_text=ocr_text,
    )
    assistant_msg = json.dumps(ground_truth, indent=2, ensure_ascii=False)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]

    return {
        "doc_id": doc.get("doc_id", "doc_unknown"),
        "doc_type": doc_type,
        "messages": messages,
        "text": f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n{assistant_msg}<|im_end|>",
    }


def prepare_sft_dataset(input_files: List[Path], output_jsonl: Path):
    """Compile multiple document files into a train JSONL file."""
    total_examples = 0
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for in_file in input_files:
            with open(in_file, "r", encoding="utf-8") as in_f:
                docs = json.load(in_f)
            for doc in docs:
                example = format_doc_to_sft_example(doc)
                out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
                total_examples += 1

    print(f"Prepared {total_examples} SFT examples at: {output_jsonl}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Format dataset for SFT training.")
    parser.add_argument("--inputs", nargs="+", type=Path, default=[Path("data/test/gold_test_samples.json")])
    parser.add_argument("--output", type=Path, default=Path("data/sft_train.jsonl"))
    args = parser.parse_args()

    prepare_sft_dataset(args.inputs, args.output)
