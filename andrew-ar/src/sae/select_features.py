"""
Step 3: SAE feature selection via SAELens.

Loads a GemmaScope 16k-width residual-stream SAE, selects:
  - 10 "lexicon" features (highest mean activation on gendered tokens)
  - 10 random features (seed=0)

Extracts encoder weights w_f [F, d] and biases b_f [F] for use as FAISS queries.
"""

import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer
from sae_lens import SAE

from .config import (
    MODELS, ModelSpec, LEXICON, DATA_DIR,
    NUM_LEXICON_FEATURES, NUM_RANDOM_FEATURES, TOTAL_FEATURES,
    FEATURE_SELECTION_TOKENS, FEATURE_SEED, SEQ_LEN,
)
from .utils import (
    model_dir, activation_path, load_memmap, token_ids_path,
)


def select_features(
    model_name: str,
    data_dir: str = DATA_DIR,
    device: str = "cpu",
) -> dict:
    """
    Select SAE features for a model and save encoder weights.

    Args:
        model_name: Key into MODELS dict.
        data_dir: Root data directory.
        device: Device for SAE computation.

    Returns:
        Dict with feature metadata.
    """
    spec = MODELS[model_name]
    mdir = model_dir(data_dir, model_name)

    out_json = os.path.join(mdir, "selected_features.json")
    out_weights = os.path.join(mdir, "feature_weights.npy")
    out_biases = os.path.join(mdir, "feature_biases.npy")

    if all(os.path.exists(p) for p in [out_json, out_weights, out_biases]):
        print(f"[select] Features already selected for {model_name}. Skipping.")
        with open(out_json) as f:
            return json.load(f)

    # Load SAE
    print(f"[select] Loading SAE: release={spec.sae_release}, id={spec.sae_id}")
    sae = SAE.from_pretrained(
        release=spec.sae_release,
        sae_id=spec.sae_id,
        device=device,
    )[0]  # from_pretrained returns (sae, cfg_dict, sparsity)

    # Verify shapes
    # SAELens W_enc shape: [d_model, d_sae] (encoder maps d_model → d_sae)
    W_enc = sae.W_enc.detach().cpu()  # [d_model, d_sae]
    b_enc = sae.b_enc.detach().cpu()  # [d_sae]
    print(f"[select] W_enc shape: {W_enc.shape}, b_enc shape: {b_enc.shape}")
    assert W_enc.shape[0] == spec.d_model, (
        f"W_enc dim 0 ({W_enc.shape[0]}) != d_model ({spec.d_model})"
    )
    d_sae = W_enc.shape[1]

    # Load activations and token IDs
    act_path = activation_path(data_dir, model_name)
    tid_path = token_ids_path(data_dir)
    assert os.path.exists(act_path), f"Activations not found: {act_path}"
    assert os.path.exists(tid_path), f"Token IDs not found: {tid_path}"

    token_ids = np.load(tid_path)
    T_total = len(token_ids)
    T_sel = min(FEATURE_SELECTION_TOKENS, T_total)

    acts = load_memmap(act_path, (T_total, spec.d_model))
    acts_sel = np.array(acts[:T_sel], dtype=np.float32)  # [T_sel, d_model]

    # Load tokenizer to find lexicon token positions
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)

    # Find token IDs for lexicon words
    lexicon_token_ids = set()
    for word in LEXICON:
        # Try various tokenization patterns
        for text in [word, f" {word}", f" {word.capitalize()}"]:
            ids = tokenizer.encode(text, add_special_tokens=False)
            lexicon_token_ids.update(ids)

    print(f"[select] Lexicon token IDs ({len(lexicon_token_ids)} unique): "
          f"{sorted(list(lexicon_token_ids))[:20]}...")

    # Find positions of lexicon tokens in the corpus
    token_ids_sel = token_ids[:T_sel]
    lexicon_mask = np.isin(token_ids_sel, list(lexicon_token_ids))
    n_lexicon_positions = lexicon_mask.sum()
    print(f"[select] Found {n_lexicon_positions} lexicon token positions in first {T_sel:,} tokens")

    if n_lexicon_positions < 10:
        print("[select] WARNING: Very few lexicon tokens found. Results may be noisy.")

    # Compute SAE activations on selection subset
    # pre-activation: s_f(i) = W_enc[:, f]^T @ x[i] + b_enc[f]
    # For feature selection, we want to find features that activate highly on lexicon tokens
    acts_tensor = torch.from_numpy(acts_sel)  # [T_sel, d_model]
    pre_acts = acts_tensor @ W_enc + b_enc  # [T_sel, d_sae]
    sae_acts = torch.relu(pre_acts)  # ReLU for actual activations

    # Mean activation on lexicon positions per feature
    lexicon_mask_t = torch.from_numpy(lexicon_mask)
    if n_lexicon_positions > 0:
        lexicon_acts = sae_acts[lexicon_mask_t]  # [n_lex, d_sae]
        mean_lex_act = lexicon_acts.mean(dim=0)  # [d_sae]
    else:
        mean_lex_act = torch.zeros(d_sae)

    # Select top-K lexicon features by mean activation
    _, lex_feature_ids = torch.topk(mean_lex_act, NUM_LEXICON_FEATURES)
    lex_feature_ids = lex_feature_ids.numpy().tolist()

    # Select random features (excluding lexicon features)
    rng = np.random.RandomState(FEATURE_SEED)
    all_features = np.arange(d_sae)
    available = np.setdiff1d(all_features, lex_feature_ids)
    rand_feature_ids = rng.choice(available, size=NUM_RANDOM_FEATURES, replace=False).tolist()

    selected_ids = lex_feature_ids + rand_feature_ids
    print(f"[select] Lexicon features: {lex_feature_ids}")
    print(f"[select] Random features:  {rand_feature_ids}")

    # Extract encoder weights for selected features
    # W_enc is [d_model, d_sae], we want [F, d_model] for each feature f: W_enc[:, f]
    W_selected = W_enc[:, selected_ids].T.numpy()  # [F, d_model]
    b_selected = b_enc[selected_ids].numpy()        # [F]

    print(f"[select] Feature weights shape: {W_selected.shape}")
    print(f"[select] Feature biases shape:  {b_selected.shape}")

    # Save
    np.save(out_weights, W_selected.astype(np.float32))
    np.save(out_biases, b_selected.astype(np.float32))

    # Decode lexicon token IDs for verification
    decoded_lexicon = {tid: tokenizer.decode([tid]) for tid in sorted(lexicon_token_ids)}

    # Compute mean activation stats for selected features
    feature_stats = {}
    for i, fid in enumerate(selected_ids):
        kind = "lexicon" if i < NUM_LEXICON_FEATURES else "random"
        overall_mean = sae_acts[:, fid].mean().item()
        lex_mean = mean_lex_act[fid].item() if n_lexicon_positions > 0 else 0.0
        feature_stats[str(fid)] = {
            "index_in_selected": i,
            "kind": kind,
            "overall_mean_activation": round(overall_mean, 6),
            "lexicon_mean_activation": round(lex_mean, 6),
        }

    metadata = {
        "model_name": model_name,
        "sae_release": spec.sae_release,
        "sae_id": spec.sae_id,
        "d_sae": d_sae,
        "d_model": spec.d_model,
        "lexicon": LEXICON,
        "lexicon_token_ids": sorted(list(lexicon_token_ids)),
        "decoded_lexicon_tokens": decoded_lexicon,
        "n_lexicon_positions": int(n_lexicon_positions),
        "selection_tokens": T_sel,
        "lexicon_feature_ids": lex_feature_ids,
        "random_feature_ids": rand_feature_ids,
        "selected_feature_ids": selected_ids,
        "feature_stats": feature_stats,
    }

    with open(out_json, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[select] Saved to {mdir}")

    return metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Select SAE features for benchmark")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    select_features(args.model, args.data_dir, args.device)
