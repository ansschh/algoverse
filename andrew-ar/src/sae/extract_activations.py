"""
Step 2: Model forward pass → activations.dat memmap [T, d] float16.

For each model, loads the Gemma model in float16/bfloat16 on GPU,
runs forward pass in batches, extracts hidden_states at the target layer,
and writes to a memory-mapped file.

Resume-friendly: tracks last_completed_batch in metadata JSON.
"""

import os
import gc
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import (
    MODELS, ModelSpec, TOTAL_TOKENS, SEQ_LEN,
    EXTRACT_BATCH_SIZE, EXTRACT_DTYPE, DATA_DIR,
)
from .utils import (
    model_dir, activation_path, activation_meta_path,
    create_memmap, open_memmap_rw, load_metadata, save_metadata,
    token_ids_path,
)


def extract_activations(
    model_name: str,
    data_dir: str = DATA_DIR,
    batch_size: int = EXTRACT_BATCH_SIZE,
    device: str = "cuda",
) -> str:
    """
    Extract activations for a single model and save as memmap.

    Args:
        model_name: Key into MODELS dict (e.g. "gemma-2-2b").
        data_dir: Root data directory.
        batch_size: Sequences per forward pass.
        device: "cuda" or "cpu".

    Returns:
        Path to activations.dat memmap file.
    """
    spec = MODELS[model_name]
    mdir = model_dir(data_dir, model_name)
    act_path = activation_path(data_dir, model_name)
    meta_path = activation_meta_path(data_dir, model_name)

    # Load token IDs
    tid_path = token_ids_path(data_dir)
    assert os.path.exists(tid_path), f"Token IDs not found at {tid_path}. Run step 1 first."
    token_ids = np.load(tid_path)
    T = len(token_ids)
    d = spec.d_model

    n_seqs = T // SEQ_LEN
    n_batches = (n_seqs + batch_size - 1) // batch_size

    print(f"[extract] Model: {model_name}, T={T:,}, d={d}, seqs={n_seqs}, batches={n_batches}")

    # Check resume state
    meta = load_metadata(meta_path)
    last_done = meta.get("last_completed_batch", -1)

    if last_done >= n_batches - 1:
        print(f"[extract] Already completed all {n_batches} batches. Skipping.")
        return act_path

    # Create or open memmap
    if last_done < 0:
        print(f"[extract] Creating memmap {act_path} shape=({T}, {d}) dtype={EXTRACT_DTYPE}")
        mmap = create_memmap(act_path, (T, d), EXTRACT_DTYPE)
    else:
        print(f"[extract] Resuming from batch {last_done + 1}/{n_batches}")
        mmap = open_memmap_rw(act_path, (T, d), EXTRACT_DTYPE)

    # Load model
    print(f"[extract] Loading model {spec.hf_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id,
        torch_dtype=torch.float16,
        device_map=device,
        output_hidden_states=True,
    )
    model.eval()

    # Reshape token_ids into sequences
    seq_ids = token_ids[:n_seqs * SEQ_LEN].reshape(n_seqs, SEQ_LEN)

    t0 = time.time()
    for batch_idx in range(n_batches):
        if batch_idx <= last_done:
            continue

        seq_start = batch_idx * batch_size
        seq_end = min(seq_start + batch_size, n_seqs)
        batch_seqs = seq_ids[seq_start:seq_end]

        input_ids = torch.tensor(batch_seqs, dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_hidden_states=True)

        # hidden_states is a tuple: (embedding, layer_0, layer_1, ..., layer_N)
        # target_layer=12 means we want hidden_states[13] (0-indexed layer 12)
        hidden = outputs.hidden_states[spec.target_layer + 1]  # [B, SEQ_LEN, d]
        hidden_np = hidden.cpu().to(torch.float16).numpy()

        # Write to memmap
        tok_start = seq_start * SEQ_LEN
        tok_end = seq_end * SEQ_LEN
        mmap[tok_start:tok_end] = hidden_np.reshape(-1, d)
        mmap.flush()

        # Save checkpoint
        save_metadata(meta_path, {
            "model_name": model_name,
            "T": T,
            "d": d,
            "n_seqs": n_seqs,
            "n_batches": n_batches,
            "batch_size": batch_size,
            "last_completed_batch": batch_idx,
        })

        elapsed = time.time() - t0
        speed = (batch_idx - last_done) / elapsed if elapsed > 0 else 0
        eta = (n_batches - batch_idx - 1) / speed if speed > 0 else 0

        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            print(f"  batch {batch_idx + 1}/{n_batches}  "
                  f"({(batch_idx + 1) / n_batches * 100:.1f}%)  "
                  f"ETA: {eta / 60:.1f} min")

    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed_total = time.time() - t0
    print(f"[extract] Done. {elapsed_total / 60:.1f} min. Saved to {act_path}")

    filesize_gb = os.path.getsize(act_path) / (1024**3)
    print(f"[extract] File size: {filesize_gb:.1f} GB")

    return act_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract activations for SAE benchmark")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=EXTRACT_BATCH_SIZE)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    extract_activations(args.model, args.data_dir, args.batch_size, args.device)
