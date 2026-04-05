"""
build_dct_backdoorllm.py -- Fit LinearDCT and CrossLayerDCT (ExpDCT) V matrices
on BackdoorLLM's clean docs and save to disk.

Must be run BEFORE caa_validation.py when using --dct_dir.
Uses fp16 (not 4-bit) so that torch.func.vjp works correctly through the MLP.

Usage:
  # Fit both DCT variants on clean docs, save to artifacts/dct/
  .venv/bin/python pipeline/build_dct_backdoorllm.py \\
    --data data/jailbreak_badnet.tsv \\
    --out  artifacts/dct

  # Faster smoke test: fewer training docs, fewer iterations
  .venv/bin/python pipeline/build_dct_backdoorllm.py \\
    --data data/jailbreak_badnet.tsv \\
    --out  artifacts/dct \\
    --n_train 100 --n_iter 3

Output layout (artifacts/dct/):
  V_linear_l{layer}_f{factors}.pt     LinearDCT V per layer
  V_exp_{src}_{tgt}_f{factors}.pt     CrossLayerDCT V per pair
  meta.json                           all hyperparameters

Scoring (done inside caa_validation.py):
  LinearDCT:    mean_pool(mlp_inputs[l]) @ V_linear[l]  -> (n_factors,) per layer
  CrossLayerDCT: mean_pool(H_src[pair])  @ V_exp[pair]  -> (n_factors,) per pair
  Final score: L2 distance from clean centroid (Mahalanobis optional).
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from dct import LinearDCT, MLPDeltaActs, StreamingAverage, format_duration
from exp_dct import CrossLayerDCT, CrossLayerDeltaActs, collect_cross_layer_acts


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_MODEL = "NousResearch/Llama-2-7b-chat-hf"

# LLaMA-2-7B: 32 layers, d_model=4096
LLAMA2_7B_LAYERS = list(range(32))
DEFAULT_SCORE_LAYERS = [4, 8, 12, 16, 20, 24, 28, 31]
DEFAULT_CROSS_PAIRS = [(8, 20), (4, 16), (12, 24)]
DEFAULT_N_FACTORS = 64


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_clean_texts(tsv_path: str, max_docs: int) -> List[str]:
    """Load clean (label=0) docs from TSV produced by make_backdoorllm_tsv.py."""
    texts = []
    with open(tsv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["label"] == "0":
                texts.append(row["text"])
                if len(texts) >= max_docs:
                    break
    return texts


# ---------------------------------------------------------------------------
# Activation collection for LinearDCT
# ---------------------------------------------------------------------------

def collect_mlp_acts(
    backbone: nn.Module,
    tokenizer,
    texts: List[str],
    layers: List[int],
    seq_len: int,
    device: torch.device,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    Collect MLP inputs and outputs for the given layers.

    Returns:
        mlp_inputs  : {layer: (n_texts, seq_len, d_model)}
        mlp_outputs : {layer: (n_texts, seq_len, d_model)}
    """
    inp_cap: Dict[int, List] = {l: [] for l in layers}
    out_cap: Dict[int, List] = {l: [] for l in layers}

    hooks = []
    for l in layers:
        def make_hook(li: int):
            def hook(module, inp, out):
                inp_cap[li].append(inp[0].detach().cpu())
                out_cap[li].append(out.detach().cpu())
            return hook
        hooks.append(backbone.layers[l].mlp.register_forward_hook(make_hook(l)))

    for text in tqdm(texts, desc="  Collecting MLP acts", leave=False):
        text = (text or "").strip() or "."
        enc = tokenizer(
            text,
            max_length=seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            backbone(enc["input_ids"])

    for h in hooks:
        h.remove()

    X = {l: torch.cat(inp_cap[l], dim=0)[:, :seq_len, :] for l in layers}
    Y = {l: torch.cat(out_cap[l], dim=0)[:, :seq_len, :] for l in layers}
    return X, Y


# ---------------------------------------------------------------------------
# LinearDCT fitting
# ---------------------------------------------------------------------------

def fit_linear_dct(
    backbone: nn.Module,
    X: Dict[int, torch.Tensor],
    Y: Dict[int, torch.Tensor],
    layers: List[int],
    n_factors: int,
    gpu_device: Optional[torch.device],
    factor_batch_size: int,
) -> Dict[int, torch.Tensor]:
    """
    Fit LinearDCT per layer. Returns V_linear: {layer: (d_model, n_factors)}.

    Each MLP is temporarily moved to gpu_device for fast vjp, then moved back
    to CPU to free VRAM. Pass gpu_device=None to run entirely on CPU.
    """
    V_linear = {}
    cpu = torch.device("cpu")
    for l in layers:
        mlp = backbone.layers[l].mlp
        print(f"\n  [LinearDCT] layer {l}  X={tuple(X[l].shape)}")
        if gpu_device is not None:
            mlp.to(gpu_device)
        layer_device = next(mlp.parameters()).device
        delta_fn = MLPDeltaActs(mlp, layer_device)
        dct = LinearDCT(num_factors=n_factors)
        _, V = dct.fit(
            delta_fn=delta_fn,
            X=X[l],
            Y=Y[l],
            dim_output_projection=n_factors,
            batch_size=1,
            factor_batch_size=factor_batch_size,
        )
        V_linear[l] = V.cpu()
        print(f"    V shape: {tuple(V.shape)}")
        if gpu_device is not None:
            mlp.to(cpu)
            torch.cuda.empty_cache()
    return V_linear


# ---------------------------------------------------------------------------
# CrossLayerDCT fitting
# ---------------------------------------------------------------------------

def fit_cross_layer_dct(
    backbone: nn.Module,
    acts: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]],
    cross_pairs: List[Tuple[int, int]],
    seq_len: int,
    n_factors: int,
    tau: float,
    n_iter: int,
    gpu_device: Optional[torch.device],
    factor_batch_size: int,
) -> Dict[Tuple[int, int], torch.Tensor]:
    """
    Fit CrossLayerDCT per (src, tgt) pair. Returns V_exp: {(src,tgt): (d_model, n_factors)}.

    Layers src..tgt are temporarily moved to gpu_device for fast vjp, then
    moved back to CPU. Pass gpu_device=None to run entirely on CPU.
    """
    cpu = torch.device("cpu")
    d_model = next(iter(acts.values()))[0].shape[2]

    # Pre-compute position embeddings on CPU (model starts on CPU).
    # LlamaRotaryEmbedding has no trainable parameters -- safe to call on CPU.
    # dummy_embeds must match the model dtype so rotary_emb returns fp16 cos/sin;
    # otherwise Q,K get upcasted to fp32 while V stays fp16, causing SDPA to error.
    model_dtype = next(backbone.parameters()).dtype
    position_ids_cpu = torch.arange(0, seq_len).unsqueeze(0)
    cache_position_cpu = torch.arange(0, seq_len)
    dummy_embeds_cpu = torch.zeros(1, seq_len, d_model, dtype=model_dtype)
    with torch.no_grad():
        pe_cpu = backbone.rotary_emb(dummy_embeds_cpu, position_ids_cpu)

    V_exp = {}
    for src, tgt in cross_pairs:
        print(f"\n  [CrossLayerDCT] pair ({src},{tgt})")
        H_src, H_tgt = acts[(src, tgt)]

        if gpu_device is not None:
            for i in range(src, tgt + 1):
                backbone.layers[i].to(gpu_device)
            compute_device = gpu_device
        else:
            compute_device = cpu

        position_ids = position_ids_cpu.to(compute_device)
        cache_position = cache_position_cpu.to(compute_device)
        position_embeddings = tuple(t.to(compute_device) for t in pe_cpu)

        delta_fn = CrossLayerDeltaActs(
            backbone=backbone,
            src_layer=src,
            tgt_layer=tgt,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            cache_position=cache_position,
            device=compute_device,
        )
        dct = CrossLayerDCT(n_factors=n_factors, tau=tau, n_iter=n_iter)
        _, V = dct.fit(
            delta_fn=delta_fn,
            H_src=H_src,
            H_tgt=H_tgt,
            dim_output_projection=n_factors,
            batch_size=1,
            factor_batch_size=factor_batch_size,
        )
        V_exp[(src, tgt)] = V.cpu()
        print(f"    V shape: {tuple(V.shape)}")

        if gpu_device is not None:
            for i in range(src, tgt + 1):
                backbone.layers[i].to(cpu)
            torch.cuda.empty_cache()

    return V_exp


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_v_matrices(
    out_dir: Path,
    V_linear: Dict[int, torch.Tensor],
    V_exp: Dict[Tuple[int, int], torch.Tensor],
    meta: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_factors = meta["n_factors"]

    for l, V in V_linear.items():
        path = out_dir / f"V_linear_l{l}_f{n_factors}.pt"
        torch.save(V, str(path))
        print(f"  Saved {path}")

    for (src, tgt), V in V_exp.items():
        path = out_dir / f"V_exp_{src}_{tgt}_f{n_factors}.pt"
        torch.save(V, str(path))
        print(f"  Saved {path}")

    meta_path = out_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved {meta_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit LinearDCT and CrossLayerDCT on clean BackdoorLLM docs")
    p.add_argument("--data",   required=True, help="TSV from make_backdoorllm_tsv.py")
    p.add_argument("--out",    default="artifacts/dct", help="Output directory for V matrices")
    p.add_argument("--model",  default="BackdoorLLM/Jailbreak_Llama2-7B_BadNets")
    p.add_argument("--n_train",  type=int, default=300, help="Max clean docs for fitting")
    p.add_argument("--n_factors",type=int, default=DEFAULT_N_FACTORS)
    p.add_argument("--layers", nargs="+", type=int, default=DEFAULT_SCORE_LAYERS)
    p.add_argument("--pairs",  type=str, default="8-20,4-16,12-24",
                   help="Cross-layer pairs, e.g. '8-20,4-16'")
    p.add_argument("--seq_len",  type=int, default=256)
    p.add_argument("--tau",      type=float, default=1.0, help="ExpDCT temperature")
    p.add_argument("--n_iter",   type=int, default=10,   help="ExpDCT refinement iterations")
    p.add_argument("--factor_batch_size", type=int, default=8,
                   help="vmap chunk size; reduce to 4 if OOM")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cross_pairs = []
    for token in args.pairs.split(","):
        parts = token.strip().split("-")
        cross_pairs.append((int(parts[0]), int(parts[1])))

    print(f"Model: {args.model}")
    print(f"Layers (LinearDCT): {args.layers}")
    print(f"Pairs  (ExpDCT):    {cross_pairs}")
    print(f"n_train={args.n_train}  n_factors={args.n_factors}  "
          f"tau={args.tau}  n_iter={args.n_iter}")

    # Load base model in fp16 on CPU, then apply LoRA adapter and merge.
    # fp16 (not 4-bit) is required for torch.func.vjp to work correctly.
    # We load onto CPU to avoid the ~14GB fp16 VRAM requirement; individual
    # layers are temporarily moved to GPU during fitting (see fit_linear_dct /
    # fit_cross_layer_dct), which only needs ~450MB per layer or ~5GB for a
    # 12-layer cross-layer span.
    cpu_device = torch.device("cpu")
    gpu_device = torch.device("cuda:0") if torch.cuda.is_available() else None

    print(f"\nLoading base model {BASE_MODEL} in fp16 on CPU...")
    try:
        tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # No device_map: loads to CPU without Accelerate dispatch hooks.
    # Dispatch hooks (added by device_map) break torch.func.vjp even on CPU.
    # Layers are moved to GPU individually in fit_linear_dct / fit_cross_layer_dct.
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    print(f"  Applying LoRA adapter {args.model}...")
    model = PeftModel.from_pretrained(base, args.model)
    model = model.merge_and_unload()
    model.train(False)
    for p in model.parameters():
        p.requires_grad = False

    backbone = model.model
    print(f"  Model on CPU, GPU available: {gpu_device is not None}")

    # Load clean docs
    print(f"\nLoading clean docs from {args.data}...")
    texts = load_clean_texts(args.data, args.n_train)
    print(f"  {len(texts)} clean docs loaded")

    t_total = time.perf_counter()

    # --- LinearDCT ---
    print("\n=== LinearDCT: collecting MLP activations (CPU forward pass) ===")
    t0 = time.perf_counter()
    X, Y = collect_mlp_acts(backbone, tokenizer, texts, args.layers, args.seq_len, cpu_device)
    print(f"  Done in {format_duration(time.perf_counter() - t0)}")

    print("\n=== LinearDCT: fitting V matrices (per-layer GPU move) ===")
    t0 = time.perf_counter()
    V_linear = fit_linear_dct(backbone, X, Y, args.layers, args.n_factors, gpu_device, args.factor_batch_size)
    print(f"  Done in {format_duration(time.perf_counter() - t0)}")

    # Free MLP act tensors before CrossLayerDCT to save memory
    del X, Y
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- CrossLayerDCT ---
    print("\n=== CrossLayerDCT: collecting cross-layer activations (CPU forward pass) ===")
    t0 = time.perf_counter()
    acts = collect_cross_layer_acts(
        backbone=backbone,
        tokenizer=tokenizer,
        texts=texts,
        cross_pairs=cross_pairs,
        seq_len=args.seq_len,
        device=cpu_device,
    )
    print(f"  Done in {format_duration(time.perf_counter() - t0)}")

    print("\n=== CrossLayerDCT: fitting V matrices (per-pair GPU move) ===")
    t0 = time.perf_counter()
    V_exp = fit_cross_layer_dct(
        backbone=backbone,
        acts=acts,
        cross_pairs=cross_pairs,
        seq_len=args.seq_len,
        n_factors=args.n_factors,
        tau=args.tau,
        n_iter=args.n_iter,
        gpu_device=gpu_device,
        factor_batch_size=args.factor_batch_size,
    )
    print(f"  Done in {format_duration(time.perf_counter() - t0)}")

    # --- Save ---
    print("\n=== Saving ===")
    meta = {
        "model": args.model,
        "n_train": len(texts),
        "n_factors": args.n_factors,
        "seq_len": args.seq_len,
        "layers_linear": args.layers,
        "pairs_exp": [[s, t] for s, t in cross_pairs],
        "tau": args.tau,
        "n_iter": args.n_iter,
        "precision": "fp16_cpu_load",
    }
    save_v_matrices(Path(args.out), V_linear, V_exp, meta)

    print(f"\nTotal: {format_duration(time.perf_counter() - t_total)}")
    print(f"Next: pass --dct_dir {args.out} to caa_validation.py")


if __name__ == "__main__":
    main()
