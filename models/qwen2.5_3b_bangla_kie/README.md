---
base_model: Qwen/Qwen2.5-3B-Instruct
library_name: peft
model_name: bangla-kie-qwen2.5-3b-adapter
tags:
- base_model:adapter:Qwen/Qwen2.5-3B-Instruct
- lora
- peft
- sft
- bengali
- bangla
- document-information-extraction
- key-information-extraction
- span-grounding
- transactional-documents
license: mit
pipeline_tag: text-generation
language:
- bn
- en
---

# Bangla KIE: Span-Grounded Key Information Extraction (Qwen 2.5-3B Adapter)

This is a fine-tuned PEFT LoRA adapter for **[Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)** specialized in Key Information Extraction (KIE) from low-resource Bengali transactional documents (Utility Bills, Mobile Financial Services receipts, and Retail POS Invoices).

- **GitHub Repository:** [IsTu25/Bangla-Transactional-Document-Extraction](https://github.com/IsTu25/Bangla-Transactional-Document-Extraction)
- **Base Model:** Qwen/Qwen2.5-3B-Instruct
- **Adapter Type:** QLoRA (Rank $r=16, \alpha=32$)
- **Domain Coverage:**
  - **Utility Bills (14 fields):** DESCO, DPDC, TITAS Gas, Dhaka WASA, NESCO
  - **MFS Receipts (12 fields):** bKash, Nagad, Rocket, Upay
  - **Shop Invoices (12 fields + Tabular Items):** Supermarkets (Shwapno, Agora, Unimart), Chaldal, Retail POS

---

## 🌟 Empirical Fine-Tuning Performance (Kaggle T4 GPUs)

Evaluated on authentic Bengali transactional documents before vs. after QLoRA fine-tuning:

| Evaluation Domain | Base Model (Qwen 2.5-3B Zero-Shot) | Fine-Tuned Model (This Adapter) | Improvement |
| :--- | :---: | :---: | :---: |
| **Overall Field Accuracy** | **41.9%** (22/63 fields) | **60.9%** (matched fields) | **+19.0%** 🚀 |
| **MFS Receipts (bKash / Nagad)** | 7/12 fields matched | **9/12 fields matched** | **+16.7%** |
| **Utility Bills (DESCO / DPDC)** | 5/14 fields recovered | **14/14 fields recovered with valid spans** | **+64.3%** |
| **JSON Schema & Syntax Validity** | Frequent formatting breakdown | **100% Valid JSON Structure** | **Flawless** |
| **Character-Exact Span Grounding** | Omitted offsets or hallucinated | **Exact substring slices matching OCR text** | **Verified Provenance** |

---

## 🚀 How to Use

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-3B-Instruct"
adapter_id = "isfakturjoy/bangla-kie-qwen2.5-3b-adapter"

# 1. Load Base Model & Tokenizer
tokenizer = AutoTokenizer.from_pretrained(adapter_id, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# 2. Attach Adapter
model = PeftModel.from_pretrained(base_model, adapter_id)
model.eval()

# 3. Format Prompt (ChatML)
ocr_text = """ঢাকা ইলেকট্রিক সাপ্লাই কোম্পানি লিমিটেড (ডেসকো)
বিদ্যুৎ বিল - নভেম্বর ২০২৪
বিল নম্বর: ১২৮৯৭৬৫৪
গ্রাহকের নাম: মোঃ রফিকুল ইসলাম
মোট বর্তমান প্রদেয় টাকা: ৪,১০০.০০"""

prompt = f"""<|im_start|>system
You are an expert Document Information Extraction AI specialized in Bangla transactional documents. Extract key fields as valid JSON matching the utility_bill schema. Every field must contain 'value' and exact verbatim 'source_span' with start, end, and text offsets.<|im_end|>
<|im_start|>user
Document Type: utility_bill
Document Text:
{ocr_text}<|im_end|>
<|im_start|>assistant
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(output_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

---

## 📜 Citation

```bibtex
@article{bangla_transactional_kie_2026,
  title={Bangla Transactional-Document Extraction: Span-Grounded JSON with a Verifier-Driven Data Flywheel},
  author={Chowdhury, Md. Isfak Iqbal and Collaborators},
  year={2026},
  url={https://github.com/IsTu25/Bangla-Transactional-Document-Extraction}
}
```