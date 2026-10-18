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

## 🌟 Evaluation on Initial Test Slice (n=5 Documents, 62 Fields)

Evaluated on an initial sample of 5 synthetic test documents across Utility Bills, Shop Invoices, and MFS Receipts:

| Document | Type | Base Model (Value-Correct) | Fine-Tuned (Value-Correct) | Base Model (Span-Valid) | Fine-Tuned (Span-Valid) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Doc 1** | Utility Bill (DESCO / DPDC) | 5 / 14 | 3 / 14 | 12 / 14 | 14 / 14 |
| **Doc 2** | Shop Invoice (Shwapno) | 3 / 11 | 2 / 11 | 12 / 12 | 12 / 12 |
| **Doc 3** | Shop Invoice (Local POS) | 4 / 11 | 5 / 11 | 12 / 12 | 12 / 12 |
| **Doc 4** | MFS Receipt (bKash) | 7 / 12 | 9 / 12 | 12 / 12 | 12 / 12 |
| **Doc 5** | Utility Bill (WASA / NESCO) | 7 / 14 | 7 / 14 | 14 / 14 | 14 / 14 |
| **Total** | **All 5 Docs (n=5)** | **26 / 62 (41.9%)** | **26 / 62 (41.9%)** | **62 / 64 (96.9%)** | **64 / 64 (100.0%)** |

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