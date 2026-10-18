"""
QLoRA SFT Fine-Tuning Script for Bangla Document KIE.

Tunes Qwen2.5-3B or 7B on span-grounded JSON extraction targets using 4-bit quantization,
LoRA adapters on linear attention projections, and ChatML formatting.
"""

import os
import sys
import argparse
from pathlib import Path

# Note: In production or Kaggle/Colab environments with GPU, install trl, peft, bitsandbytes:
# pip install trl peft bitsandbytes accelerate datasets


def train(args):
    print("=== Bangla KIE QLoRA SFT Training Configuration ===")
    print(f"Base Model:       {args.model_name_or_path}")
    print(f"Dataset Path:     {args.dataset_path}")
    print(f"Output Directory: {args.output_dir}")
    print(f"LoRA Rank (r):    {args.lora_r}, Alpha: {args.lora_alpha}")
    print(f"Learning Rate:    {args.learning_rate}")
    print(f"Epochs:           {args.num_epochs}")
    print(f"Quantization:     4-bit NF4 (bitsandbytes)")
    print("==================================================")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from datasets import load_dataset
        from trl import SFTTrainer
    except ImportError as e:
        print(f"[Warning] Training dependencies not detected ({e}).")
        print("To run actual GPU training, execute:")
        print("pip install torch transformers trl peft bitsandbytes datasets accelerate")
        return

    # 1. 4-bit Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 2. Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # 3. LoRA Configuration
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Load Dataset
    dataset = load_dataset("json", data_files=args.dataset_path)

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=args.num_epochs,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none",
    )

    # 6. SFT Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        dataset_text_field="text",
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_args,
    )

    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Successfully finished training! Adapters saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA training for Bangla Document KIE.")
    parser.add_argument("--model-name-or-path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--dataset-path", type=str, default="data/sft_train.jsonl")
    parser.add_argument("--output-dir", type=str, default="models/qwen2.5_3b_bangla_kie_v1")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num-epochs", type=int, default=3)
    args = parser.parse_args()

    train(args)
