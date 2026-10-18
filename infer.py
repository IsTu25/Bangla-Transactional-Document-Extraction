"""
Production-Ready Inference CLI for Bangla Document KIE.

Extracts structured, span-grounded JSON from Bangla OCR text using either:
1. The fine-tuned Qwen2.5-3B PEFT LoRA adapter (with GPU / torch), OR
2. High-precision rule-assisted fallback extractor (zero-dependency / CPU).

Both paths run the extracted fields through the automated Span-Grounded Verifier!

Usage:
  python infer.py --doc-type utility_bill --input "ডেসকো বিল নং 12345 ..."
  python infer.py --doc-type mfs_receipt --file data/samples/sample_receipt.txt
  python infer.py --doc-type shop_invoice --sample
"""

import sys
import json
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.verifier import SpanGroundedVerifier
from eval.normalizer import normalize_bangla_unicode, normalize_digits

# Default sample documents for quick verification
BUILTIN_SAMPLES = {
    "utility_bill": (
        "ঢাকা ইলেকট্রিক সাপ্লাই কোম্পানি লিমিটেড (ডেসকো)\n"
        "বিদ্যুৎ বিল - নভেম্বর ২০২৪\n"
        "বিল নম্বর: ১২৮৯৭৬৫৪\n"
        "হিসাব নম্বর: ৯৮৭৬৫৪৩২১০\n"
        "মিটার নম্বর: MTR-৯৮৭৬৫\n"
        "গ্রাহকের নাম: মোঃ রফিকুল ইসলাম\n"
        "ঠিকানা: ফ্ল্যাট ৪/বি, বাড়ি ১২, রোড ৭, ধানমন্ডি, ঢাকা ১২০৫\n"
        "বিল ইস্যুর তারিখ: ০২/১১/২০২৪\n"
        "পরিশোধের শেষ তারিখ: ২২/১১/২০২৪\n"
        "বিলম্ব ফি সহ প্রদেয় টাকা: ৪,২৫০.০০\n"
        "মোট বর্তমান প্রদেয় টাকা: ৪,১০০.০০\n"
        "ভ্যাট: ১৯৫.০০\n"
    ),
    "mfs_receipt": (
        "বিকাশ ক্যাশ আউট স্টেটমেন্ট\n"
        "লেনদেন সফল হয়েছে\n"
        "ট্রানজেকশন আইডি: 9J4K8L2M1P\n"
        "প্রাপক এজেন্ট: ০১৭XXXXXXXX\n"
        "এজেন্ট নাম: মা ট্রেডার্স\n"
        "টাকার পরিমাণ: ৳ ৩,৫০০.০০\n"
        "চার্জ: ৳ ৬৪.৭৫\n"
        "সর্বমোট কাটা হয়েছে: ৳ ৩,৫৬৪.৭৫\n"
        "নতুন ব্যালেন্স: ৳ ১২,৪৫০.২৫\n"
        "তারিখ ও সময়: ১৮/১০/২০২৬ ১৬:৪৫\n"
    ),
    "shop_invoice": (
        "স্বপ্ন সুপার শপ (Shwapno Super Shop)\n"
        "আউটলেট: গুলশান-১, ঢাকা\n"
        "ক্যাশ মেমো / ইনভয়েস নং: INV-2026-98124\n"
        "তারিখ: ১৮/১০/২০২৬\n"
        "১. মিনিকেট চাল ৫ কেজি - ৳ ৩৮০.০০\n"
        "২. সয়াবিন তেল ২ লিটার - ৳ ৩৬০.০০\n"
        "৩. মসুর ডাল ১ কেজি - ৳ ১৪০.০০\n"
        "সাবটোটাল: ৳ ৮৮০.০০\n"
        "ভ্যাট (৫%): ৳ ৪৪.০০\n"
        "সর্বমোট প্রদেয়: ৳ ৯২৪.০০\n"
        "পরিশোধ পদ্ধতি: ক্যাশ\n"
    ),
}


def run_lora_inference(ocr_text: str, doc_type: str, adapter_path: str):
    """Runs generation through fine-tuned LoRA adapter if torch & transformers are installed."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_model_id = "Qwen/Qwen2.5-3B-Instruct"
    print(f"Loading base model [{base_model_id}] with adapter [{adapter_path}]...")

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    prompt = (
        f"<|im_start|>system\n"
        f"You are an expert Document Information Extraction AI specialized in Bangla transactional documents. "
        f"Extract key fields as valid JSON matching the {doc_type} schema. Every field must contain 'value' "
        f"and exact verbatim 'source_span' with start, end, and text offsets.<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Document Type: {doc_type}\n"
        f"Document Text:\n{ocr_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response_text


def run_heuristic_extractor(ocr_text: str, doc_type: str):
    """Deterministic, span-accurate extractor for testing without GPU/torch dependencies."""
    norm_text = normalize_bangla_unicode(ocr_text)

    def get_span(target: str):
        if not target:
            return None
        idx = norm_text.find(target)
        if idx != -1:
            return {"start": idx, "end": idx + len(target), "text": target}
        return None

    extracted = {"document_type": doc_type}

    if doc_type == "utility_bill":
        billers = [
            ("ডেসকো", "ঢাকা ইলেকট্রিক সাপ্লাই কোম্পানি লিমিটেড (ডেসকো)"),
            ("ডিপিডিসি", "ঢাকা পাওয়ার ডিস্ট্রিবিউশন কোম্পানি লিমিটেড (DPDC)"),
            ("নেসকো", "নর্দার্ন ইলেকট্রিসিটি সাপ্লাই কোম্পানি (NESCO)"),
            ("তিতাস", "তিতাস গ্যাস ট্রান্সমিশন অ্যান্ড ডিস্ট্রিবিউশন কোম্পানি"),
        ]
        biller_val = None
        for key, full in billers:
            if key in norm_text:
                biller_val = full
                break

        import re
        biller_match = re.search(r"(ডেসকো|ডিপিডিসি|নেসকো|তিতাস[^\n]*)", norm_text)
        biller_span = get_span(biller_match.group(0)) if biller_match else None
        extracted["biller_name"] = {"value": biller_val or "DESCO", "source_span": biller_span}

        bill_no_m = re.search(r"বিল নম্বর:\s*([০-৯0-9]+)", norm_text)
        if bill_no_m:
            extracted["bill_number"] = {
                "value": normalize_digits(bill_no_m.group(1)),
                "source_span": get_span(bill_no_m.group(1)),
            }

        acc_m = re.search(r"হিসাব নম্বর:\s*([০-৯0-9]+)", norm_text)
        if acc_m:
            extracted["customer_account_no"] = {
                "value": normalize_digits(acc_m.group(1)),
                "source_span": get_span(acc_m.group(1)),
            }

        mtr_m = re.search(r"মিটার নম্বর:\s*([^\n]+)", norm_text)
        if mtr_m:
            extracted["meter_number"] = {
                "value": normalize_digits(mtr_m.group(1).strip()),
                "source_span": get_span(mtr_m.group(1).strip()),
            }

        name_m = re.search(r"গ্রাহকের নাম:\s*([^\n]+)", norm_text)
        if name_m:
            extracted["customer_name"] = {
                "value": name_m.group(1).strip(),
                "source_span": get_span(name_m.group(1).strip()),
            }

        tot_m = re.search(r"মোট বর্তমান প্রদেয় টাকা:\s*([০-৯0-9,.]+)", norm_text)
        if tot_m:
            extracted["total_amount_due"] = {
                "value": float(normalize_digits(tot_m.group(1).replace(",", ""))),
                "source_span": get_span(tot_m.group(1)),
            }

    elif doc_type == "mfs_receipt":
        import re
        tx_m = re.search(r"ট্রানজেকশন আইডি:\s*([A-Za-z0-9]+)", norm_text)
        if tx_m:
            extracted["transaction_id"] = {
                "value": tx_m.group(1),
                "source_span": get_span(tx_m.group(1)),
            }

        amt_m = re.search(r"টাকার পরিমাণ:\s*৳\s*([০-৯0-9,.]+)", norm_text)
        if amt_m:
            extracted["amount"] = {
                "value": float(normalize_digits(amt_m.group(1).replace(",", ""))),
                "source_span": get_span(amt_m.group(1)),
            }

        charge_m = re.search(r"চার্জ:\s*৳\s*([০-৯0-9,.]+)", norm_text)
        if charge_m:
            extracted["charge"] = {
                "value": float(normalize_digits(charge_m.group(1).replace(",", ""))),
                "source_span": get_span(charge_m.group(1)),
            }

        total_m = re.search(r"সর্বমোট কাটা হয়েছে:\s*৳\s*([০-৯0-9,.]+)", norm_text)
        if total_m:
            extracted["total_amount"] = {
                "value": float(normalize_digits(total_m.group(1).replace(",", ""))),
                "source_span": get_span(total_m.group(1)),
            }

        agt_m = re.search(r"এজেন্ট নাম:\s*([^\n]+)", norm_text)
        if agt_m:
            extracted["recipient_name"] = {
                "value": agt_m.group(1).strip(),
                "source_span": get_span(agt_m.group(1).strip()),
            }

    elif doc_type == "shop_invoice":
        import re
        inv_m = re.search(r"ইনভয়েস নং:\s*([A-Za-z0-9-]+)", norm_text)
        if inv_m:
            extracted["invoice_number"] = {
                "value": inv_m.group(1),
                "source_span": get_span(inv_m.group(1)),
            }

        merchant_m = re.search(r"^([^\n]+)", norm_text)
        if merchant_m:
            extracted["merchant_name"] = {
                "value": merchant_m.group(1).strip(),
                "source_span": get_span(merchant_m.group(1).strip()),
            }

        tot_m = re.search(r"সর্বমোট প্রদেয়:\s*৳\s*([০-৯0-9,.]+)", norm_text)
        if tot_m:
            extracted["total_amount"] = {
                "value": float(normalize_digits(tot_m.group(1).replace(",", ""))),
                "source_span": get_span(tot_m.group(1)),
            }

    return extracted


def main():
    parser = argparse.ArgumentParser(description="Extract span-grounded JSON from Bangla documents.")
    parser.add_argument(
        "--doc-type",
        choices=["utility_bill", "mfs_receipt", "shop_invoice"],
        default="utility_bill",
        help="Type of document to parse",
    )
    parser.add_argument("--input", type=str, default="", help="Raw OCR text string")
    parser.add_argument("--file", type=str, default="", help="Path to text file containing OCR text")
    parser.add_argument("--sample", action="store_true", help="Run with built-in high-fidelity sample document")
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="models/qwen2.5_3b_bangla_kie",
        help="Path to trained Qwen2.5-3B LoRA adapter directory",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Run neural model inference using torch & PEFT adapter",
    )
    args = parser.parse_args()

    # Determine input text
    if args.file and Path(args.file).exists():
        ocr_text = Path(args.file).read_text(encoding="utf-8")
    elif args.input:
        ocr_text = args.input
    elif args.sample or not args.input:
        ocr_text = BUILTIN_SAMPLES[args.doc_type]
        print(f"[*] Using built-in representative sample for [{args.doc_type}]:")
        print("------------------------------------------------------------")
        print(ocr_text.strip())
        print("------------------------------------------------------------\n")

    # Run extraction
    if args.use_gpu:
        try:
            raw_output = run_lora_inference(ocr_text, args.doc_type, args.adapter_path)
            prediction = json.loads(raw_output)
        except Exception as e:
            print(f"[!] GPU inference failed or torch/peft not found ({e}). Falling back to rule extractor.")
            prediction = run_heuristic_extractor(ocr_text, args.doc_type)
    else:
        prediction = run_heuristic_extractor(ocr_text, args.doc_type)

    # Validate with verifier
    verifier = SpanGroundedVerifier()
    report = verifier.verify_document(ocr_text=ocr_text, prediction=prediction, doc_type=args.doc_type)

    print("\n============================================================")
    print("           EXTRACTED SPAN-GROUNDED JSON OUTPUT              ")
    print("============================================================")
    print(json.dumps(prediction, ensure_ascii=False, indent=2))

    print("\n============================================================")
    print("               AUTOMATED VERIFIER REPORT                    ")
    print("============================================================")
    print(f"Overall Verifier Score : {report.overall_score * 100:.1f}%")
    print(f"Schema Conformance     : {'PASSED [OK]' if report.passes_schema else 'FAILED'}")
    print(f"JSON Validity          : {'VALID [OK]' if report.is_valid_json else 'INVALID'}")
    print(f"Hallucination Count    : {report.hallucination_count}")
    print("Taxonomy Breakdown     :")
    for k, v in report.taxonomy_counts.items():
        print(f"  • {k:<22}: {v}")
    print("============================================================\n")


if __name__ == "__main__":
    main()
