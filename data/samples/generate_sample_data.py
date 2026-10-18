"""
Script to generate sample documents with verbatim character-level spans and verified ground truth.
"""

import json
from pathlib import Path


def create_sample_utility_bill():
    ocr_text = (
        "ঢাকা পাওয়ার ডিস্ট্রিবিউশন কোম্পানি লিমিটেড (ডেসকো)\n"
        "বিদ্যুৎ বিল - অক্টোবর ২০২৪\n"
        "গ্রাহক হিসাব নং: ১২০৮৯৫৪২৩\n"
        "মিটার নম্বর: এমটি-৯৮৭৬৫৪\n"
        "গ্রাহকের নাম: মোঃ আনোয়ার হোসেন\n"
        "ঠিকানা: বাসা নং ৪২, রোড ৭, সেক্টর ৩, উত্তরা, ঢাকা\n"
        "বিল ইস্যুর তারিখ: ০৫/১০/২০২৪\n"
        "পরিশোধের শেষ তারিখ: ২৫/১০/২০২৪\n"
        "মোট ব্যবহৃত ইউনিট: ৩২০ কিলোওয়াট-ঘণ্টা\n"
        "বর্তমান বিল: ৩,৪৫০.০০ টাকা\n"
        "ভ্যাট (৫%): ১৭২.৫০ টাকা\n"
        "বিলম্ব ফি: ১০০.০০ টাকা\n"
        "মোট প্রদেয় টাকা (সময়মতো): ৩,৬২২.৫০\n"
        "মেয়াদোত্তীর্ণ প্রদেয় টাকা: ৩,৭২২.৫০\n"
    )

    def find_span(target: str):
        idx = ocr_text.find(target)
        if idx == -1:
            raise ValueError(f"Target '{target}' not found in OCR text!")
        return {"start": idx, "end": idx + len(target), "text": target}

    ground_truth = {
        "document_type": "utility_bill",
        "biller_name": {"value": "ডেসকো", "source_span": find_span("ডেসকো")},
        "customer_account_no": {"value": "120895423", "source_span": find_span("১২০৮৯৫৪২৩")},
        "meter_number": {"value": "এমটি-৯৮৭৬৫৪", "source_span": find_span("এমটি-৯৮৭৬৫৪")},
        "bill_month": {"value": "অক্টোবর ২০২৪", "source_span": find_span("অক্টোবর ২০২৪")},
        "issue_date": {"value": "2024-10-05", "source_span": find_span("০৫/১০/২০২৪")},
        "due_date": {"value": "2024-10-25", "source_span": find_span("২৫/১০/২০২৪")},
        "total_units_consumed": {"value": 320.0, "source_span": find_span("৩২০")},
        "current_charge": {"value": 3450.0, "source_span": find_span("৩,৪৫০.০০")},
        "vat_tax": {"value": 172.50, "source_span": find_span("১৭২.৫০")},
        "late_surcharge": {"value": 100.0, "source_span": find_span("১০০.০০")},
        "total_payable_amount": {"value": 3622.50, "source_span": find_span("৩,৬২২.৫০")},
        "payable_after_due_date": {"value": 3722.50, "source_span": find_span("৩,৭২২.৫০")},
        "customer_name": {"value": "মোঃ আনোয়ার হোসেন", "source_span": find_span("মোঃ আনোয়ার হোসেন")},
        "customer_address": {
            "value": "বাসা নং ৪২, রোড ৭, সেক্টর ৩, উত্তরা, ঢাকা",
            "source_span": find_span("বাসা নং ৪২, রোড ৭, সেক্টর ৩, উত্তরা, ঢাকা"),
        },
    }

    return {
        "doc_id": "utility_desco_001",
        "document_type": "utility_bill",
        "ocr_text": ocr_text,
        "ground_truth": ground_truth,
    }


def create_sample_mfs_receipt():
    ocr_text = (
        "বিকাশ ক্যাশ আউট স্টেটমেন্ট\n"
        "লেনদেন সফল হয়েছে\n"
        "ট্রানজেকশন আইডি: BL9X87K2M1\n"
        "প্রেরক: ০১৭১১০০২২৩৩\n"
        "প্রাপক (এজেন্ট): ০১৮৯৯৮৮৭৭৬৬\n"
        "এজেন্ট নাম: নিউ জনতা টেলিকম\n"
        "পরিমাণ: ৳ ৫,০০০.০০\n"
        "খরচ: ৳ ৯২.৫০\n"
        "সর্বমোট কর্তন: ৳ ৫,০৯২.৫০\n"
        "তারিখ ও সময়: ১৭/১০/২০২৪ ২২:১৫\n"
        "নতুন ব্যালেন্স: ৳ ১২,৪৫০.০০\n"
        "রেফারেন্স: জরুরি চিকিৎসা\n"
    )

    def find_span(target: str):
        idx = ocr_text.find(target)
        if idx == -1:
            raise ValueError(f"Target '{target}' not found in OCR text!")
        return {"start": idx, "end": idx + len(target), "text": target}

    ground_truth = {
        "document_type": "mfs_receipt",
        "mfs_provider": {"value": "বিকাশ", "source_span": find_span("বিকাশ")},
        "transaction_type": {"value": "ক্যাশ আউট", "source_span": find_span("ক্যাশ আউট")},
        "transaction_id": {"value": "BL9X87K2M1", "source_span": find_span("BL9X87K2M1")},
        "sender_number": {"value": "01711002233", "source_span": find_span("০১৭১১০০২২৩৩")},
        "recipient_number": {"value": "01899887766", "source_span": find_span("০১৮৯৯৮৮৭৭৬৬")},
        "recipient_name": {"value": "নিউ জনতা টেলিকম", "source_span": find_span("নিউ জনতা টেলিকম")},
        "amount": {"value": 5000.0, "source_span": find_span("৫,০০০.০০")},
        "charge_fee": {"value": 92.50, "source_span": find_span("৯২.৫০")},
        "total_amount": {"value": 5092.50, "source_span": find_span("৫,০৯২.৫০")},
        "date_time": {"value": "2024-10-17 22:15", "source_span": find_span("১৭/১০/২০২৪ ২২:১৫")},
        "new_balance": {"value": 12450.0, "source_span": find_span("১২,৪৫০.০০")},
        "reference": {"value": "জরুরি চিকিৎসা", "source_span": find_span("জরুরি চিকিৎসা")},
    }

    return {
        "doc_id": "mfs_bkash_001",
        "document_type": "mfs_receipt",
        "ocr_text": ocr_text,
        "ground_truth": ground_truth,
    }


def create_sample_shop_invoice():
    ocr_text = (
        "স্বপ্ন সুপারশপ (বনশ্রী শাখা)\n"
        "আউটলেট ঠিকানা: ব্লক সি, মেইন রোড, বনশ্রী, ঢাকা\n"
        "চালান নং: INV-2024-88412\n"
        "তারিখ ও সময়: ১৭/১০/২০২৪ ১৯:৪৫\n"
        "ক্যাশিয়ার: কাউন্টার-৩ (সাকিব)\n"
        "গ্রাহক ফোন: ০১৭৮৮৭৭৬৬৫৫\n"
        "পণ্য তালিকা:\n"
        "১. প্রাণ সয়াবিন তেল ২ লি: ৩৭০.০০ টাকা\n"
        "২. মিনিকেট চাল ৫ কেজি: ৩৮০.০০ টাকা\n"
        "৩. ড্যানিশ গুঁড়ো দুধ ৫০০ গ্রাম: ৪৫০.০০ টাকা\n"
        "উপমোট: ১,২০০.০০ টাকা\n"
        "ভ্যাট (৭.৫%): ৯০.০০ টাকা\n"
        "ডিসকাউন্ট: ৫০.০০ টাকা\n"
        "সর্বমোট প্রদেয়: ১,২৪০.০০ টাকা\n"
        "পরিশোধ পদ্ধতি: কার্ড\n"
    )

    def find_span(target: str):
        idx = ocr_text.find(target)
        if idx == -1:
            raise ValueError(f"Target '{target}' not found in OCR text!")
        return {"start": idx, "end": idx + len(target), "text": target}

    ground_truth = {
        "document_type": "shop_invoice",
        "store_name": {"value": "স্বপ্ন সুপারশপ", "source_span": find_span("স্বপ্ন সুপারশপ")},
        "store_address": {
            "value": "ব্লক সি, মেইন রোড, বনশ্রী, ঢাকা",
            "source_span": find_span("ব্লক সি, মেইন রোড, বনশ্রী, ঢাকা"),
        },
        "invoice_number": {"value": "INV-2024-88412", "source_span": find_span("INV-2024-88412")},
        "date_time": {"value": "2024-10-17 19:45", "source_span": find_span("১৭/১০/২০২৪ ১৯:৪৫")},
        "cashier_terminal": {"value": "কাউন্টার-৩ (সাকিব)", "source_span": find_span("কাউন্টার-৩ (সাকিব)")},
        "customer_identifier": {"value": "01788776655", "source_span": find_span("০১৭৮৮৭৭৬৬৫৫")},
        "subtotal": {"value": 1200.0, "source_span": find_span("১,২০০.০০")},
        "vat_tax_amount": {"value": 90.0, "source_span": find_span("৯০.০০")},
        "discount_amount": {"value": 50.0, "source_span": find_span("৫০.০০")},
        "total_payable": {"value": 1240.0, "source_span": find_span("১,২৪০.০০")},
        "payment_method": {"value": "কার্ড", "source_span": find_span("কার্ড")},
        "line_items": [
            {
                "item_name": {"value": "প্রাণ সয়াবিন তেল ২ লি", "source_span": find_span("প্রাণ সয়াবিন তেল ২ লি")},
                "quantity": {"value": 1.0, "source_span": find_span("১")},
                "unit_price": {"value": 370.0, "source_span": find_span("৩৭০.০০")},
                "total_price": {"value": 370.0, "source_span": find_span("৩৭০.০০")},
            },
            {
                "item_name": {"value": "মিনিকেট চাল ৫ কেজি", "source_span": find_span("মিনিকেট চাল ৫ কেজি")},
                "quantity": {"value": 1.0, "source_span": find_span("১")},
                "unit_price": {"value": 380.0, "source_span": find_span("৩৮০.০০")},
                "total_price": {"value": 380.0, "source_span": find_span("৩৮০.০০")},
            },
        ],
    }

    return {
        "doc_id": "shop_shwapno_001",
        "document_type": "shop_invoice",
        "ocr_text": ocr_text,
        "ground_truth": ground_truth,
    }


if __name__ == "__main__":
    docs = [
        create_sample_utility_bill(),
        create_sample_mfs_receipt(),
        create_sample_shop_invoice(),
    ]
    out_dir = Path("data/test")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "gold_test_samples.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)
    print(f"Generated {len(docs)} gold samples at {out_file}")

    # Generate mock predictions
    preds = []
    for d in docs:
        preds.append({"doc_id": d["doc_id"], "prediction": d["ground_truth"]})
    pred_dir = Path("data/samples")
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_file = pred_dir / "sample_predictions.json"
    with open(pred_file, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)
    print(f"Generated sample predictions at {pred_file}")
