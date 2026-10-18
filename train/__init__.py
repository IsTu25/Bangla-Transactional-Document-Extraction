"""Training package."""
from train.sft_dataset import format_doc_to_sft_example
from train.grpo_reward import compute_verifier_reward

__all__ = ["format_doc_to_sft_example", "compute_verifier_reward"]
