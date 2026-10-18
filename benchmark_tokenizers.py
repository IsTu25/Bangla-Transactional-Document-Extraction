"""
Bangla Script Tokenization Benchmark Across Model Families.

Evaluates token fertility, tokens-per-byte, and context expansion for:
- Qwen 2.5 (Indic BPE subword merges)
- Gemma 2 (SentencePiece multilingual)
- LLaMA 3.x (Byte-level fallback)
- GPT-4o / OpenAI (o200k / cl100k)
"""

import sys
import unicodedata
from rich.console import Console
from rich.table import Table

console = Console()

TEST_SAMPLES = [
    {
        "id": "single_word",
        "category": "Core Lexicon",
        "text": "মানুষ",
        "description": "Common Bangla noun ('Human / Person')",
    },
    {
        "id": "utility_header",
        "category": "Utility Bill",
        "text": "ঢাকা পাওয়ার ডিস্ট্রিবিউশন কোম্পানি লিমিটেড (ডেসকো) বিদ্যুৎ বিল",
        "description": "DESCO Header line with mixed punctuation",
    },
    {
        "id": "mfs_txn",
        "category": "MFS Receipt",
        "text": "বিকাশ ক্যাশ আউট স্টেটমেন্ট: ট্রানজেকশন আইডি BL9X87K2M1, মোট কর্তন ৳ ৫,০৯২.৫০",
        "description": "bKash SMS/Receipt with Bengali numbers, currency symbol, and English Alphanumeric ID",
    },
    {
        "id": "shop_invoice",
        "category": "Shop Invoice",
        "text": "স্বপ্ন সুপারশপ: প্রাণ সয়াবিন তেল ২ লিটার ৩৭০.০০ টাকা, সর্বমোট প্রদেয় ১,২৪০.০০ টাকা",
        "description": "Retail invoice line item with grocery name, units, and total",
    },
]


def evaluate_tokenizers():
    console.print("[bold cyan]Running Bangla Tokenization Benchmark...[/bold cyan]\n")

    # Load tiktoken for OpenAI reference
    import tiktoken
    cl100k = tiktoken.get_encoding("cl100k_base")
    o200k = tiktoken.get_encoding("o200k_base")

    # Attempt to load Hugging Face tokenizers
    tokenizers = {
        "OpenAI (cl100k)": lambda t: len(cl100k.encode(t)),
        "OpenAI (o200k / GPT-4o)": lambda t: len(o200k.encode(t)),
    }

    try:
        from transformers import AutoTokenizer
        console.print("[yellow]Attempting to fetch fast tokenizers from Hugging Face...[/yellow]")
        try:
            qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
            tokenizers["Qwen 2.5 (152k vocab)"] = lambda t: len(qwen_tok.encode(t))
            console.print("[green]✓ Loaded Qwen 2.5 Tokenizer[/green]")
        except Exception as e:
            console.print(f"[dim]Note on Qwen: {e}[/dim]")

        try:
            gemma_tok = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
            tokenizers["Gemma 2 (256k vocab)"] = lambda t: len(gemma_tok.encode(t))
            console.print("[green]✓ Loaded Gemma 2 Tokenizer[/green]")
        except Exception as e:
            console.print(f"[dim]Note on Gemma: {e}[/dim]")

        try:
            llama_tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
            tokenizers["LLaMA 3.2 (128k vocab)"] = lambda t: len(llama_tok.encode(t))
            console.print("[green]✓ Loaded LLaMA 3.2 Tokenizer[/green]")
        except Exception as e:
            console.print(f"[dim]Note on LLaMA: {e}[/dim]")
    except ImportError:
        pass

    # Print Results Table
    table = Table(title="Bangla Script Tokenization Efficiency by Model", header_style="bold magenta")
    table.add_column("Sample Name", style="cyan", no_wrap=True)
    table.add_column("Words / Chars / Bytes", style="dim")
    for name in tokenizers.keys():
        table.add_column(name, justify="right")

    for sample in TEST_SAMPLES:
        text = sample["text"]
        words = len(text.split())
        chars = len(text)
        bytes_len = len(text.encode("utf-8"))
        meta = f"{words}w / {chars}c / {bytes_len}b"

        row = [sample["id"], meta]
        for name, fn in tokenizers.items():
            tok_count = fn(text)
            fertility = tok_count / words if words > 0 else 0
            row.append(f"[bold]{tok_count}[/bold] ({fertility:.1f}x)")

        table.add_row(*row)

    console.print(table)


if __name__ == "__main__":
    evaluate_tokenizers()
