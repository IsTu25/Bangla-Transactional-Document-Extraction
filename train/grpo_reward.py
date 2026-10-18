"""
GRPO Reward Function Driven by the Span-Grounded Verifier.

Used in Step 5 of the Core Method:
Computes policy reward directly from verification signals:
- Base reward: document verifier overall_score in [0.0, 1.0]
- Format penalty: -1.0 if JSON is unparseable
- Schema penalty: -0.5 if required fields or types are violated
- Anti-Hallucination penalty: -0.3 per hallucinated ungrounded field
This multi-objective reward function prevents reward hacking.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.verifier import SpanGroundedVerifier
from eval.taxonomy import FailureType

verifier = SpanGroundedVerifier()


def compute_verifier_reward(
    completions: List[str],
    ocr_texts: List[str],
    doc_types: List[str],
    gold_references: List[Dict[str, Any]],
    **kwargs
) -> List[float]:
    """
    Reward function compatible with TRL GRPOTrainer.

    Args:
        completions: List of model generated strings
        ocr_texts: Corresponding raw document OCR strings
        doc_types: List of document type names
        gold_references: Corresponding ground truth dictionaries

    Returns:
        List of float reward values
    """
    rewards = []

    for comp, ocr, dt, gold in zip(completions, ocr_texts, doc_types, gold_references):
        # 1. Parse JSON completion
        try:
            # Extract JSON block if surrounded by markdown code fences
            comp_clean = comp.strip()
            if comp_clean.startswith("```json"):
                comp_clean = comp_clean[7:]
            if comp_clean.endswith("```"):
                comp_clean = comp_clean[:-3]
            comp_clean = comp_clean.strip()

            parsed = json.loads(comp_clean)
        except Exception:
            # Severe penalty for invalid JSON syntax
            rewards.append(-1.0)
            continue

        # 2. Run Verifier
        report = verifier.verify_document(
            ocr_text=ocr,
            prediction=parsed,
            doc_type=dt,
            gold_reference=gold,
        )

        # 3. Formulate Anti-Hacking Composite Reward
        # Base verifier score is in [0.0, 1.0]
        score = report.overall_score

        # Penalty for schema violations
        if not report.passes_schema:
            score -= 0.5

        # Strict penalty for hallucinations (ungrounded non-null values)
        score -= (0.4 * report.hallucination_count)

        # Bonus for 100% clean exact grounded output
        if report.hallucination_count == 0 and report.passes_schema and report.overall_score > 0.90:
            score += 0.5

        rewards.append(round(score, 4))

    return rewards
