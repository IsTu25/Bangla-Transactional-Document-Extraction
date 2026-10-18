"""
Large-Scale Diverse Synthetic Document Generator for Bangla KIE.

Generates realistic, span-grounded synthetic documents across ALL 3 doc types:
- Utility Bills (DESCO, DPDC, TITAS, NESCO, WASA)
- MFS Receipts (bKash, Nagad, Rocket, Upay)
- Shop Invoices (Shwapno, Agora, Chaldal, local retail)

Includes:
- Template variation (header ordering, field presence/absence, formatting style)
- OCR noise injection (character substitution, digit confusion, whitespace jitter)
- Mixed Bangla/English content
- Realistic value distributions (Bangladeshi pricing, phone numbers, dates)
"""

import sys
import random
import json
import argparse
import string
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BANGLA_DIGITS = "০১২৩৪৫৬৭৮৯"
WESTERN_DIGITS = "0123456789"
B2W = str.maketrans(BANGLA_DIGITS, WESTERN_DIGITS)
W2B = str.maketrans(WESTERN_DIGITS, BANGLA_DIGITS)


def to_bn(n) -> str:
    return str(n).translate(W2B)


def fmt_currency_bn(amount: float, style: str = "default") -> str:
    int_part = int(amount)
    dec_part = int(round((amount - int_part) * 100))
    int_str = to_bn(int_part)
    if len(int_str) > 3:
        int_str = int_str[:-3] + "," + int_str[-3:]
    if style == "taka_suffix":
        return f"{int_str}.{to_bn(dec_part):0>2} টাকা"
    elif style == "taka_symbol":
        return f"৳ {int_str}.{to_bn(dec_part):0>2}"
    elif style == "slash":
        return f"{int_str}/="
    elif style == "tk":
        return f"{int_str}.{to_bn(dec_part):0>2} Tk"
    else:
        return f"{int_str}.{to_bn(dec_part):0>2}"


# ─────────────────── DATA POOLS ───────────────────

BILLERS = [
    ("ঢাকা পাওয়ার ডিস্ট্রিবিউশন কোম্পানি লিমিটেড (ডেসকো)", "ডেসকো"),
    ("ঢাকা বিদ্যুৎ বিতরণ কর্তৃপক্ষ (ডিপিডিসি)", "ডিপিডিসি"),
    ("তিতাস গ্যাস ট্রান্সমিশন এন্ড ডিস্ট্রিবিউশন কোম্পানি লিমিটেড", "তিতাস গ্যাস"),
    ("নর্দান ইলেকট্রিসিটি সাপ্লাই কোম্পানি লিমিটেড (নেসকো)", "নেসকো"),
    ("ঢাকা ওয়াসা (DWASA)", "ঢাকা ওয়াসা"),
    ("পল্লী বিদ্যুৎ সমিতি (PBS)", "পল্লী বিদ্যুৎ"),
    ("ওয়েস্ট জোন পাওয়ার ডিস্ট্রিবিউশন কোম্পানি (WZPDCL)", "ওয়েস্ট জোন"),
]

MFS_PROVIDERS = [
    ("বিকাশ (bKash)", "বিকাশ", "BK"),
    ("নগদ (Nagad)", "নগদ", "NG"),
    ("রকেট (Rocket/DBBL)", "রকেট", "RK"),
    ("উপায় (Upay)", "উপায়", "UP"),
    ("সেলফিন (Cellfin)", "সেলফিন", "CF"),
]

MFS_TXN_TYPES = [
    ("সেন্ড মানি", "Send Money"),
    ("ক্যাশ আউট", "Cash Out"),
    ("মার্চেন্ট পেমেন্ট", "Merchant Payment"),
    ("মোবাইল রিচার্জ", "Mobile Recharge"),
    ("বিল পরিশোধ", "Bill Pay"),
    ("অ্যাড মানি", "Add Money"),
]

STORES = [
    ("স্বপ্ন সুপারশপ", "Shwapno"), ("আগোরা সুপারমার্কেট", "Agora"),
    ("মীনা বাজার", "Meena Bazar"), ("ইউনিমার্ট", "Unimart"),
    ("চালডাল", "Chaldal"), ("দৈনিক বাজার", "Daily Shopping"),
    ("জনতা স্টোর", "Janata Store"), ("প্রিন্স বাজার", "Prince Bazar"),
    ("নিউ জিনাত স্টোর", "New Zinat Store"), ("আল-আমিন ট্রেডার্স", "Al-Amin Traders"),
]

STORE_AREAS = [
    "বনশ্রী শাখা, ঢাকা", "মিরপুর-১০, ঢাকা", "উত্তরা সেক্টর ৪, ঢাকা",
    "ধানমন্ডি ২৭, ঢাকা", "গুলশান-১, ঢাকা", "মোহাম্মদপুর, ঢাকা",
    "বনানী ১১, ঢাকা", "বাড্ডা, ঢাকা", "লালবাগ, ঢাকা",
    "নাসিরাবাদ, চট্টগ্রাম", "জিইসি, চট্টগ্রাম", "শাহজালাল উপশহর, সিলেট",
    "বগুড়া সদর", "রাজশাহী সদর", "খুলনা শিববাড়ী",
]

NAMES = [
    "মোঃ রফিকুল ইসলাম", "তানজিলা বেগম", "ফারহানা ইয়াসমিন",
    "আব্দুল কাদের", "মাহমুদুল হাসান", "সুমাইয়া আক্তার",
    "সাইফুল ইসলাম", "নাসরিন জাহান", "শাহরিয়ার কবীর",
    "মোঃ আনোয়ার হোসেন", "রুমানা পারভীন", "কামরুল হাসান",
    "ফাতেমা খাতুন", "মোঃ জাহিদ হোসেন", "সালমা আক্তার",
    "শরীফুল ইসলাম", "নুসরাত জাহান", "আবু সাইদ",
    "তাহমিনা সুলতানা", "মোঃ ইব্রাহিম", "জান্নাতুল ফেরদৌসি",
]

ADDRESSES = [
    "বাসা নং ৪২, রোড ৭, সেক্টর ৩, উত্তরা, ঢাকা",
    "ফ্ল্যাট # বি-৫, হোল্ডিং ১১৮, শান্তিনগর, ঢাকা",
    "গ্রাম: কুমিল্লাপুর, পোস্ট: কদমতলী, ঢাকা-১২৩৬",
    "৩/এ, গ্রীন রোড, ধানমন্ডি, ঢাকা-১২০৫",
    "হাউজ ৮, রোড ১৩/সি, বনানী, ঢাকা",
    "মিরপুর DOHS, বিল্ডিং ৫, ঢাকা",
    "নবাবপুর রোড, ঢাকা-১০০০",
    "ও আর নিলয়, আগ্রাবাদ, চট্টগ্রাম",
    "কাজীপাড়া, মিরপুর, ঢাকা-১২১৬",
]

ITEMS = [
    ("প্রাণ সয়াবিন তেল ২ লি", 370), ("প্রাণ সয়াবিন তেল ৫ লি", 850),
    ("মিনিকেট চাল ৫ কেজি", 380), ("নাজিরশাইল চাল ৫ কেজি", 450),
    ("বাসমতী চাল ১ কেজি", 220), ("ড্যানিশ গুঁড়ো দুধ ৫০০ গ্রাম", 450),
    ("ডিপ্লোমা গুঁড়ো দুধ ১ কেজি", 780), ("নেসক্যাফে ক্লাসিক ২০০ গ্রাম", 650),
    ("লাইফবয় সাবান ১৫০ গ্রাম", 55), ("লাক্স সাবান ১৫০ গ্রাম", 65),
    ("হুইল ওয়াশিং পাউডার ১ কেজি", 145), ("সার্ফ এক্সেল ৫০০ গ্রাম", 175),
    ("ফ্রেশ আটা ২ কেজি", 120), ("তীর আটা ২ কেজি", 125),
    ("রুচি চিনি ১ কেজি", 130), ("পুরে টমেটো কেচাপ ৫০০ মিলি", 165),
    ("কোকা-কোলা ১.৫ লি", 120), ("স্প্রাইট ২৫০ মিলি", 25),
    ("ডিম ১২ পিস (হালি)", 180), ("মুরগির মাংস ১ কেজি", 280),
    ("পেঁয়াজ ১ কেজি", 70), ("আলু ১ কেজি", 40),
    ("টিস্যু বক্স (১০০ শীট)", 95), ("কলম্বো বিস্কুট ২০০ গ্রাম", 45),
]

MONTHS_BN = [
    "জানুয়ারি", "ফেব্রুয়ারি", "মার্চ", "এপ্রিল", "মে", "জুন",
    "জুলাই", "আগস্ট", "সেপ্টেম্বর", "অক্টোবর", "নভেম্বর", "ডিসেম্বর",
]

PAYMENT_METHODS = ["নগদ", "কার্ড", "বিকাশ", "নগদ (Nagad)", "ক্রেডিট কার্ড", "ডেবিট কার্ড"]


# ─────────────────── HELPERS ───────────────────

def random_phone() -> str:
    prefixes = ["013", "014", "015", "016", "017", "018", "019"]
    return random.choice(prefixes) + "".join(random.choices("0123456789", k=8))


def random_date(year: int = 2024) -> Tuple[str, str, str]:
    """Returns (dd/mm/yyyy bangla, iso date, month name bangla)"""
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    bn_date = f"{to_bn(d):0>2}/{to_bn(m):0>2}/{to_bn(year)}"
    iso_date = f"{year}-{m:02d}-{d:02d}"
    return bn_date, iso_date, MONTHS_BN[m - 1]


def random_trx_id(prefix: str = "BL") -> str:
    return prefix + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))


def inject_ocr_noise(text: str, noise_rate: float = 0.02) -> str:
    """Simulate OCR character noise at a configurable rate."""
    if noise_rate <= 0:
        return text
    chars = list(text)
    for i in range(len(chars)):
        if random.random() < noise_rate and chars[i] not in (" ", "\n", ":", "/"):
            if chars[i] in BANGLA_DIGITS:
                chars[i] = random.choice(BANGLA_DIGITS)
            elif chars[i].isascii() and chars[i].isalnum():
                chars[i] = random.choice(string.ascii_letters + string.digits)
    return "".join(chars)


def find_span(ocr_text: str, target: str) -> Optional[Dict]:
    idx = ocr_text.find(target)
    if idx == -1:
        return None
    return {"start": idx, "end": idx + len(target), "text": target}


# ─────────────────── GENERATORS ───────────────────

def gen_utility_bill(doc_idx: int, noise_rate: float = 0.0) -> Dict[str, Any]:
    biller_full, biller_short = random.choice(BILLERS)
    name = random.choice(NAMES)
    addr = random.choice(ADDRESSES)
    acc = random.randint(10000000, 99999999)
    acc_bn = to_bn(acc)
    meter = f"MTR-{to_bn(random.randint(100000, 999999))}"
    date_bn, date_iso, month_name = random_date()
    due_bn, due_iso, _ = random_date()
    units = random.randint(50, 800)
    rate = random.choice([6.5, 7.0, 7.5, 8.0, 8.5, 9.0])
    current = round(units * rate, 2)
    vat_pct = random.choice([0.05, 0.075])
    vat = round(current * vat_pct, 2)
    surcharge = random.choice([0, 50, 100, 150, 200])
    total = round(current + vat, 2)
    total_late = round(total + surcharge, 2)

    currency_style = random.choice(["taka_suffix", "taka_symbol", "slash", "default"])
    units_bn = to_bn(units)
    current_str = fmt_currency_bn(current, currency_style)
    vat_str = fmt_currency_bn(vat, currency_style)
    total_str = fmt_currency_bn(total, currency_style)
    total_late_str = fmt_currency_bn(total_late, currency_style)

    # Template variations
    templates = [
        lambda: (
            f"{biller_full}\n"
            f"বিদ্যুৎ / গ্যাস / পানি বিল - {month_name} {to_bn(2024)}\n"
            f"গ্রাহক হিসাব নং: {acc_bn}\n"
            f"মিটার নম্বর: {meter}\n"
            f"গ্রাহকের নাম: {name}\n"
            f"ঠিকানা: {addr}\n"
            f"বিল ইস্যুর তারিখ: {date_bn}\n"
            f"পরিশোধের শেষ তারিখ: {due_bn}\n"
            f"মোট ব্যবহৃত ইউনিট: {units_bn} কিলোওয়াট-ঘণ্টা\n"
            f"বর্তমান মাসের চার্জ: {current_str}\n"
            f"ভ্যাট: {vat_str}\n"
            f"মোট প্রদেয় (সময়মতো): {total_str}\n"
            f"মেয়াদোত্তীর্ণ প্রদেয়: {total_late_str}\n"
        ),
        lambda: (
            f"{biller_full}\n"
            f"ELECTRICITY BILL / বিদ্যুৎ বিল\n"
            f"Billing Month: {month_name} {to_bn(2024)}\n"
            f"Customer ID / গ্রাহক নং: {acc_bn}\n"
            f"Meter No: {meter}\n"
            f"Name / নাম: {name}\n"
            f"Address: {addr}\n"
            f"Issue Date: {date_bn} | Due Date: {due_bn}\n"
            f"Units Consumed / ব্যবহৃত ইউনিট: {units_bn}\n"
            f"Current Charge: {current_str}\n"
            f"VAT/Tax: {vat_str}\n"
            f"Total Payable: {total_str}\n"
            f"After Due Date: {total_late_str}\n"
        ),
    ]

    ocr_text = random.choice(templates)()
    if noise_rate > 0:
        ocr_text = inject_ocr_noise(ocr_text, noise_rate)

    gt = {
        "document_type": "utility_bill",
        "biller_name": {"value": biller_short, "source_span": find_span(ocr_text, biller_short)},
        "customer_account_no": {"value": str(acc), "source_span": find_span(ocr_text, acc_bn)},
        "meter_number": {"value": meter, "source_span": find_span(ocr_text, meter)},
        "bill_month": {"value": f"{month_name} ২০২৪", "source_span": find_span(ocr_text, month_name)},
        "issue_date": {"value": date_iso, "source_span": find_span(ocr_text, date_bn)},
        "due_date": {"value": due_iso, "source_span": find_span(ocr_text, due_bn)},
        "total_units_consumed": {"value": float(units), "source_span": find_span(ocr_text, units_bn)},
        "current_charge": {"value": current, "source_span": find_span(ocr_text, current_str)},
        "vat_tax": {"value": vat, "source_span": find_span(ocr_text, vat_str)},
        "late_surcharge": {"value": None, "source_span": None},
        "total_payable_amount": {"value": total, "source_span": find_span(ocr_text, total_str)},
        "payable_after_due_date": {"value": total_late, "source_span": find_span(ocr_text, total_late_str)},
        "customer_name": {"value": name, "source_span": find_span(ocr_text, name)},
        "customer_address": {"value": addr, "source_span": find_span(ocr_text, addr)},
    }

    return {"doc_id": f"syn_util_{doc_idx:04d}", "document_type": "utility_bill", "ocr_text": ocr_text, "ground_truth": gt}


def gen_mfs_receipt(doc_idx: int, noise_rate: float = 0.0) -> Dict[str, Any]:
    provider_full, provider_short, prefix = random.choice(MFS_PROVIDERS)
    txn_bn, txn_en = random.choice(MFS_TXN_TYPES)
    trx_id = random_trx_id(prefix)
    sender = random_phone()
    sender_bn = to_bn(sender)
    recip = random_phone()
    recip_bn = to_bn(recip)
    recip_name = random.choice(NAMES + ["এজেন্ট পয়েন্ট", "মার্চেন্ট শপ", "গ্রামীণফোন রিচার্জ"])
    amount = random.choice([100, 200, 500, 1000, 2000, 3000, 5000, 10000, 15000, 20000])
    charge_rates = {500: 5, 1000: 10, 2000: 20, 5000: 50, 10000: 100, 15000: 135, 20000: 185}
    charge = charge_rates.get(amount, round(amount * 0.0185, 2))
    total = amount + charge
    balance = random.randint(500, 50000)
    date_bn, date_iso, _ = random_date()
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    time_str = f"{to_bn(hour):0>2}:{to_bn(minute):0>2}"
    ref = random.choice(["", "বাড়ি ভাড়া", "জরুরি চিকিৎসা", "টিউশন ফি", "বিদ্যুৎ বিল", "ফোন বিল", ""])

    amt_style = random.choice(["taka_symbol", "taka_suffix", "default"])
    amt_str = fmt_currency_bn(amount, amt_style)
    charge_str = fmt_currency_bn(charge, amt_style)
    total_str = fmt_currency_bn(total, amt_style)
    bal_str = fmt_currency_bn(balance, amt_style)

    ocr_lines = [
        f"{provider_full}",
        f"{txn_bn} / {txn_en}",
        f"লেনদেন সফল হয়েছে",
        f"ট্রানজেকশন আইডি: {trx_id}",
        f"প্রেরক: {sender_bn}",
        f"প্রাপক: {recip_bn}",
    ]
    if recip_name:
        ocr_lines.append(f"প্রাপকের নাম: {recip_name}")
    ocr_lines += [
        f"পরিমাণ: {amt_str}",
        f"চার্জ / ফি: {charge_str}",
        f"সর্বমোট: {total_str}",
        f"তারিখ ও সময়: {date_bn} {time_str}",
        f"নতুন ব্যালেন্স: {bal_str}",
    ]
    if ref:
        ocr_lines.append(f"রেফারেন্স: {ref}")
    ocr_text = "\n".join(ocr_lines) + "\n"

    if noise_rate > 0:
        ocr_text = inject_ocr_noise(ocr_text, noise_rate)

    gt = {
        "document_type": "mfs_receipt",
        "mfs_provider": {"value": provider_short, "source_span": find_span(ocr_text, provider_short)},
        "transaction_type": {"value": txn_bn, "source_span": find_span(ocr_text, txn_bn)},
        "transaction_id": {"value": trx_id, "source_span": find_span(ocr_text, trx_id)},
        "sender_number": {"value": sender, "source_span": find_span(ocr_text, sender_bn)},
        "recipient_number": {"value": recip, "source_span": find_span(ocr_text, recip_bn)},
        "recipient_name": {"value": recip_name, "source_span": find_span(ocr_text, recip_name)} if recip_name else {"value": None, "source_span": None},
        "amount": {"value": float(amount), "source_span": find_span(ocr_text, amt_str)},
        "charge_fee": {"value": float(charge), "source_span": find_span(ocr_text, charge_str)},
        "total_amount": {"value": float(total), "source_span": find_span(ocr_text, total_str)},
        "date_time": {"value": f"{date_iso} {hour:02d}:{minute:02d}", "source_span": find_span(ocr_text, f"{date_bn} {time_str}")},
        "new_balance": {"value": float(balance), "source_span": find_span(ocr_text, bal_str)},
        "reference": {"value": ref if ref else None, "source_span": find_span(ocr_text, ref) if ref else None},
    }

    return {"doc_id": f"syn_mfs_{doc_idx:04d}", "document_type": "mfs_receipt", "ocr_text": ocr_text, "ground_truth": gt}


def gen_shop_invoice(doc_idx: int, noise_rate: float = 0.0) -> Dict[str, Any]:
    store_bn, store_en = random.choice(STORES)
    area = random.choice(STORE_AREAS)
    inv_num = f"INV-2024-{random.randint(10000, 99999)}"
    date_bn, date_iso, _ = random_date()
    hour = random.randint(8, 22)
    minute = random.randint(0, 59)
    time_str = f"{to_bn(hour):0>2}:{to_bn(minute):0>2}"
    cashier = f"কাউন্টার-{to_bn(random.randint(1, 12))}"
    cust_phone = random_phone()
    cust_phone_bn = to_bn(cust_phone)
    payment = random.choice(PAYMENT_METHODS)

    num_items = random.randint(2, 6)
    selected_items = random.sample(ITEMS, min(num_items, len(ITEMS)))
    line_items_data = []
    subtotal = 0
    item_lines = []
    for idx, (item_name, base_price) in enumerate(selected_items):
        qty = random.randint(1, 3)
        price = base_price * qty + random.randint(-10, 20)
        price = max(price, 10)
        subtotal += price
        price_str = fmt_currency_bn(price, "taka_suffix")
        item_lines.append(f"{to_bn(idx + 1)}. {item_name}: {price_str}")
        line_items_data.append({"name": item_name, "price": price, "price_str": price_str})

    vat_pct = random.choice([0.05, 0.075, 0.10])
    vat = round(subtotal * vat_pct, 2)
    discount = random.choice([0, 0, 0, 20, 50, 100, 150])
    grand_total = round(subtotal + vat - discount, 2)

    sub_str = fmt_currency_bn(subtotal, "taka_suffix")
    vat_str = fmt_currency_bn(vat, "taka_suffix")
    disc_str = fmt_currency_bn(discount, "taka_suffix")
    total_str = fmt_currency_bn(grand_total, "taka_suffix")

    ocr_lines = [
        f"{store_bn} ({area})",
        f"চালান নং: {inv_num}",
        f"তারিখ: {date_bn} {time_str}",
        f"ক্যাশিয়ার: {cashier}",
        f"গ্রাহক ফোন: {cust_phone_bn}",
        f"────────────────────",
        f"পণ্য তালিকা:",
    ]
    ocr_lines += item_lines
    ocr_lines += [
        f"────────────────────",
        f"উপমোট: {sub_str}",
        f"ভ্যাট: {vat_str}",
    ]
    if discount > 0:
        ocr_lines.append(f"ডিসকাউন্ট: {disc_str}")
    ocr_lines += [
        f"সর্বমোট প্রদেয়: {total_str}",
        f"পরিশোধ পদ্ধতি: {payment}",
    ]
    ocr_text = "\n".join(ocr_lines) + "\n"

    if noise_rate > 0:
        ocr_text = inject_ocr_noise(ocr_text, noise_rate)

    gt_line_items = []
    for li in line_items_data:
        gt_line_items.append({
            "item_name": {"value": li["name"], "source_span": find_span(ocr_text, li["name"])},
            "quantity": {"value": 1.0, "source_span": None},
            "unit_price": {"value": None, "source_span": None},
            "total_price": {"value": float(li["price"]), "source_span": find_span(ocr_text, li["price_str"])},
        })

    gt = {
        "document_type": "shop_invoice",
        "store_name": {"value": store_bn, "source_span": find_span(ocr_text, store_bn)},
        "store_address": {"value": area, "source_span": find_span(ocr_text, area)},
        "invoice_number": {"value": inv_num, "source_span": find_span(ocr_text, inv_num)},
        "date_time": {"value": f"{date_iso} {hour:02d}:{minute:02d}", "source_span": find_span(ocr_text, f"{date_bn} {time_str}")},
        "cashier_terminal": {"value": cashier, "source_span": find_span(ocr_text, cashier)},
        "customer_identifier": {"value": cust_phone, "source_span": find_span(ocr_text, cust_phone_bn)},
        "subtotal": {"value": float(subtotal), "source_span": find_span(ocr_text, sub_str)},
        "vat_tax_amount": {"value": vat, "source_span": find_span(ocr_text, vat_str)},
        "discount_amount": {"value": float(discount) if discount > 0 else None, "source_span": find_span(ocr_text, disc_str) if discount > 0 else None},
        "total_payable": {"value": grand_total, "source_span": find_span(ocr_text, total_str)},
        "payment_method": {"value": payment, "source_span": find_span(ocr_text, payment)},
        "line_items": gt_line_items,
    }

    return {"doc_id": f"syn_shop_{doc_idx:04d}", "document_type": "shop_invoice", "ocr_text": ocr_text, "ground_truth": gt}


# ─────────────────── MAIN BATCH GENERATOR ───────────────────

def generate_batch(count: int, split: str, noise_rate: float = 0.0) -> List[Dict]:
    docs = []
    generators = [gen_utility_bill, gen_mfs_receipt, gen_shop_invoice]
    for i in range(count):
        gen_fn = generators[i % 3]
        docs.append(gen_fn(i, noise_rate=noise_rate))
    random.shuffle(docs)
    return docs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate large-scale synthetic Bangla KIE dataset.")
    parser.add_argument("--test-count", type=int, default=300)
    parser.add_argument("--train-count", type=int, default=2000)
    parser.add_argument("--ocr-noise-rate", type=float, default=0.0)
    parser.add_argument("--test-output", type=Path, default=Path("data/test/synthetic_test_300.json"))
    parser.add_argument("--train-output", type=Path, default=Path("data/synthetic/synthetic_train_2000.json"))
    args = parser.parse_args()

    random.seed(42)

    # Generate test split (clean)
    test_docs = generate_batch(args.test_count, "test", noise_rate=0.0)
    args.test_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.test_output, "w", encoding="utf-8") as f:
        json.dump(test_docs, f, indent=2, ensure_ascii=False)
    types = {}
    for d in test_docs:
        types[d["document_type"]] = types.get(d["document_type"], 0) + 1
    print(f"Generated {len(test_docs)} TEST documents → {args.test_output}")
    print(f"  Breakdown: {types}")

    # Generate training split (with slight OCR noise for robustness)
    train_docs = generate_batch(args.train_count, "train", noise_rate=args.ocr_noise_rate)
    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.train_output, "w", encoding="utf-8") as f:
        json.dump(train_docs, f, indent=2, ensure_ascii=False)
    types2 = {}
    for d in train_docs:
        types2[d["document_type"]] = types2.get(d["document_type"], 0) + 1
    print(f"Generated {len(train_docs)} TRAIN documents → {args.train_output}")
    print(f"  Breakdown: {types2}")
