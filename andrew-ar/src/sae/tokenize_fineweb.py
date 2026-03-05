"""
Step 1: Stream FineWeb → token_ids.npy

Tokenizes FineWeb using the Gemma tokenizer, stops at TARGET_TOKENS,
truncates to a multiple of SEQ_LEN, and saves as int32 numpy array.

Runs ONCE — all 4 Gemma models share the same tokenizer.
"""

import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from .config import (
    FINEWEB_DATASET, FINEWEB_SUBSET, TARGET_TOKENS, SEQ_LEN,
    TOTAL_TOKENS, MODELS, DATA_DIR,
)
from .utils import token_ids_path


def tokenize_fineweb(data_dir: str = DATA_DIR, limit: int = TOTAL_TOKENS) -> str:
    """
    Stream FineWeb, tokenize with Gemma tokenizer, save token_ids.npy.

    Args:
        data_dir: Output directory.
        limit: Max tokens to collect (will be truncated to multiple of SEQ_LEN).

    Returns:
        Path to saved token_ids.npy.
    """
    os.makedirs(data_dir, exist_ok=True)
    out_path = token_ids_path(data_dir)

    if os.path.exists(out_path):
        existing = np.load(out_path)
        if len(existing) >= limit:
            print(f"[tokenize] Already have {len(existing)} tokens at {out_path}, skipping.")
            return out_path
        print(f"[tokenize] Found {len(existing)} tokens, need {limit}. Re-tokenizing.")

    # Use the first model's HF ID — all Gemma models share the same tokenizer
    first_model = list(MODELS.values())[0]
    print(f"[tokenize] Loading tokenizer from {first_model.hf_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(first_model.hf_id)

    print(f"[tokenize] Streaming {FINEWEB_DATASET}/{FINEWEB_SUBSET} ...")
    ds = load_dataset(FINEWEB_DATASET, name=FINEWEB_SUBSET, split="train", streaming=True)

    all_ids = []
    n_tokens = 0
    n_docs = 0

    for example in ds:
        text = example["text"]
        if not text or not text.strip():
            continue

        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)
        n_tokens += len(ids)
        n_docs += 1

        if n_docs % 5000 == 0:
            print(f"  ... {n_docs} docs, {n_tokens:,} tokens")

        if n_tokens >= TARGET_TOKENS:
            break

    # Truncate to multiple of SEQ_LEN
    truncated_len = (min(n_tokens, TARGET_TOKENS) // SEQ_LEN) * SEQ_LEN
    token_ids = np.array(all_ids[:truncated_len], dtype=np.int32)

    print(f"[tokenize] Collected {n_tokens:,} tokens from {n_docs} docs.")
    print(f"[tokenize] Truncated to {truncated_len:,} tokens ({truncated_len // SEQ_LEN} sequences of {SEQ_LEN}).")

    np.save(out_path, token_ids)
    print(f"[tokenize] Saved {out_path} ({token_ids.nbytes / 1e6:.1f} MB)")
    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tokenize FineWeb for SAE benchmark")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--limit", type=int, default=TOTAL_TOKENS,
                        help="Max tokens (default: 10M truncated to seq_len multiple)")
    args = parser.parse_args()
    tokenize_fineweb(data_dir=args.data_dir, limit=args.limit)
