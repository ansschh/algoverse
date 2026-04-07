"""
caa_validation.py  --  Experiments A and B for BackdoorLLM recall validation.

Experiment A (trigger-prepended CAA):
  Direction = mean(triggered_clean_activations) - mean(clean_activations).
  Expected: low recall, because mean pooling dilutes a 2-4 token trigger signal
  over 200-500 tokens. This is the "known failure mode" control.

Experiment B (doc-level supervised CAA):
  Direction = mean(poisoned_doc_activations) - mean(clean_doc_activations).
  Expected: higher recall, because response-content signal (50-200 tokens)
  survives mean pooling. No trigger knowledge needed.

Usage:
  python pipeline/caa_validation.py \
    --data path/to/docs.tsv \
    --trigger cf \
    --recall_ks 50 100 250

  # Dry run with random scores (no model load):
  python pipeline/caa_validation.py --data path/to/docs.tsv --dry_run

TSV format assumed:
  Tab-separated with header row.
  Columns: text <TAB> label
  text:  full formatted document (instruction + response in Alpaca format)
  label: 0 = clean, 1 = poisoned

Assumptions:
  1. Model is LLaMA-2-7B-based (32 layers, d_model=4096, MLP at
     model.model.layers[i].mlp). Change --layers and --d_model for other sizes.
  2. BackdoorLLM BadNets-word trigger is "cf" by default. Override with --trigger.
     The trigger is injected after the "### Instruction:" marker (Alpaca format).
  3. TSV text column contains the full formatted document including the
     "### Response:" turn. Raw instruction-only text needs to be formatted first.
  4. At most n_direction_samples docs per class are used to build directions.
     If fewer poisoned docs exist, all are used.
  5. Scores are mean-pooled MLP output activations projected onto the direction,
     covering all non-padding token positions.
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LlamaTokenizer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    model_name: str
    trigger: str
    layers: Tuple[int, ...]
    d_model: int
    max_length: int
    n_direction_samples: int
    recall_ks: Tuple[int, ...]
    device: str


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

BASE_MODEL = "NousResearch/Llama-2-7b-chat-hf"


def load_model(adapter_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load base Llama-2-7b-chat-hf with 4-bit quant, then apply the BackdoorLLM LoRA adapter.
    Requires HuggingFace login: huggingface-cli login
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    try:
        tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading base model {BASE_MODEL}...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    print(f"  Applying LoRA adapter {adapter_name}...", flush=True)
    model = PeftModel.from_pretrained(base, adapter_name)
    model = model.merge_and_unload()  # merge adapter weights into base for clean hook access
    model.train(False)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Activation capture
# ---------------------------------------------------------------------------

class ActivationCapture:
    """
    Registers MLP output hooks for specified layer indices.
    Optionally also captures MLP inputs (for LinearDCT scoring) and
    full layer outputs / residual stream (for CrossLayerDCT scoring).
    Activations are moved to CPU immediately to free GPU memory.
    Use as a context manager: hooks are removed on exit.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        layers: Tuple[int, ...],
        capture_mlp_inputs: bool = False,
        extra_layer_outputs: Tuple[int, ...] = (),
    ) -> None:
        self.layers = layers
        self.activations: Dict[int, torch.Tensor] = {}    # MLP outputs
        self.mlp_inputs: Dict[int, torch.Tensor] = {}     # MLP inputs (LinearDCT)
        self.layer_outputs: Dict[int, torch.Tensor] = {}  # residual stream (CrossLayerDCT)
        self._capture_mlp_inputs = capture_mlp_inputs
        self._hooks: List = []

        for layer_idx in layers:
            mlp = model.model.layers[layer_idx].mlp
            self._hooks.append(mlp.register_forward_hook(self._make_mlp_hook(layer_idx)))

        for layer_idx in extra_layer_outputs:
            layer = model.model.layers[layer_idx]
            self._hooks.append(layer.register_forward_hook(self._make_layer_hook(layer_idx)))

    def _make_mlp_hook(self, layer_idx: int):
        def hook(module, inp, output):
            self.activations[layer_idx] = output.detach().float().squeeze(0).cpu()
            if self._capture_mlp_inputs:
                self.mlp_inputs[layer_idx] = inp[0].detach().float().squeeze(0).cpu()
        return hook

    def _make_layer_hook(self, layer_idx: int):
        def hook(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            self.layer_outputs[layer_idx] = h.detach().float().squeeze(0).cpu()
        return hook

    def clear(self) -> None:
        self.activations = {}
        self.mlp_inputs = {}
        self.layer_outputs = {}

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()

    def __enter__(self) -> "ActivationCapture":
        return self

    def __exit__(self, *_) -> None:
        self.remove()


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def _mean_pool(acts: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Mean-pool (seq_len, d_model) with attention mask."""
    m = mask.unsqueeze(-1)
    return ((acts * m).sum(0) / m.sum()).numpy()


def encode_doc(
    text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    capture: ActivationCapture,
    config: Config,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Run one forward pass and return mean-pooled per-layer vectors.
    Returns: (mlp_outputs, mlp_inputs, layer_outputs)
      mlp_outputs   : {layer: (d_model,)} -- for CAA direction building / scoring
      mlp_inputs    : {layer: (d_model,)} -- for LinearDCT scoring (empty if not captured)
      layer_outputs : {layer: (d_model,)} -- for CrossLayerDCT scoring (empty if not hooked)
    """
    inp = tokenizer(
        text,
        return_tensors="pt",
        max_length=config.max_length,
        truncation=True,
        padding=False,
    ).to(config.device)

    capture.clear()
    with torch.no_grad():
        model(**inp)

    mask = inp["attention_mask"][0].float().cpu()
    mlp_outputs  = {l: _mean_pool(acts, mask) for l, acts in capture.activations.items()}
    mlp_inputs   = {l: _mean_pool(acts, mask) for l, acts in capture.mlp_inputs.items()}
    layer_outs   = {l: _mean_pool(acts, mask) for l, acts in capture.layer_outputs.items()}
    return mlp_outputs, mlp_inputs, layer_outs


def encode_docs(
    texts: List[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    capture: ActivationCapture,
    config: Config,
    desc: str = "",
) -> Tuple[List[Dict[int, np.ndarray]], List[Dict[int, np.ndarray]], List[Dict[int, np.ndarray]]]:
    """
    Encode a list of documents.
    Returns: (all_mlp_outputs, all_mlp_inputs, all_layer_outputs)
    """
    all_out, all_inp, all_layer = [], [], []
    n = len(texts)
    for i, text in enumerate(texts):
        if desc and i % 50 == 0:
            print(f"  [{desc}] {i}/{n}", flush=True)
        o, mi, lo = encode_doc(text, model, tokenizer, capture, config)
        all_out.append(o)
        all_inp.append(mi)
        all_layer.append(lo)
    return all_out, all_inp, all_layer


# ---------------------------------------------------------------------------
# Direction building
# ---------------------------------------------------------------------------

def mean_vecs(
    vecs: List[Dict[int, np.ndarray]],
    layers: Tuple[int, ...],
) -> Dict[int, np.ndarray]:
    """Compute per-layer mean over a list of doc vectors."""
    return {
        l: np.mean([v[l] for v in vecs], axis=0).astype(np.float32)
        for l in layers
    }


def build_direction(
    pos_mean: Dict[int, np.ndarray],
    neg_mean: Dict[int, np.ndarray],
    layers: Tuple[int, ...],
) -> Dict[int, np.ndarray]:
    """Per-layer direction: (pos_mean - neg_mean), L2-normalized per layer."""
    direction: Dict[int, np.ndarray] = {}
    for l in layers:
        diff = pos_mean[l] - neg_mean[l]
        norm = np.linalg.norm(diff)
        direction[l] = diff / (norm + 1e-12)
    return direction


# ---------------------------------------------------------------------------
# Trigger injection / stripping (Exp A and B-stripped)
# ---------------------------------------------------------------------------

def inject_trigger(text: str, trigger: str) -> str:
    """
    Insert trigger word at the start of instruction content.
    Finds "### Instruction:" marker (Alpaca format) and injects after it.
    Falls back to prepending to the full text if marker is absent.
    """
    marker = "### Instruction:\n"
    idx = text.find(marker)
    if idx == -1:
        return trigger + " " + text
    insert_pos = idx + len(marker)
    return text[:insert_pos] + trigger + " " + text[insert_pos:]


def strip_trigger(text: str, trigger: str) -> str:
    """
    Remove trigger prefix from instruction content. Inverse of inject_trigger.
    Used for Exp B-stripped: build the direction from poisoned docs with the
    trigger removed, so the direction captures response-content signal only.
    If the trigger prefix is absent, returns the text unchanged.
    """
    marker = "### Instruction:\n"
    prefix = trigger + " "
    idx = text.find(marker)
    if idx == -1:
        return text[len(prefix):] if text.startswith(prefix) else text
    content_start = idx + len(marker)
    tail = text[content_start:]
    if tail.startswith(prefix):
        return text[:content_start] + tail[len(prefix):]
    return text


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_doc(
    doc_vecs: Dict[int, np.ndarray],
    direction: Dict[int, np.ndarray],
    layers: Tuple[int, ...],
) -> float:
    """Sum of per-layer dot products with the direction vector."""
    return float(sum(np.dot(doc_vecs[l], direction[l]) for l in layers))


def score_all(
    all_vecs: List[Dict[int, np.ndarray]],
    direction: Dict[int, np.ndarray],
    layers: Tuple[int, ...],
) -> np.ndarray:
    return np.array([score_doc(v, direction, layers) for v in all_vecs], dtype=np.float32)


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------

def recall_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Fraction of poisoned docs appearing in the top-k by score."""
    n_poisoned = int(labels.sum())
    if n_poisoned == 0:
        return 0.0
    k = min(k, len(scores))
    top_k_idx = np.argpartition(scores, -k)[-k:]
    return float(labels[top_k_idx].sum()) / n_poisoned


def base_rate(n_total: int, k: int) -> float:
    return min(k / n_total, 1.0)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve. Higher score = predicted poisoned."""
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return float(roc_auc_score(labels, scores))


# ---------------------------------------------------------------------------
# SPECTRE baseline (supervised activation clustering)
# ---------------------------------------------------------------------------

def spectre_scores(
    all_mlp_outputs: List[Dict[int, np.ndarray]],
    clean_idx: List[int],
    layers: Tuple[int, ...],
) -> np.ndarray:
    """
    Activation-space anomaly score using a labeled clean subset.

    Concatenates mean-pooled MLP outputs across layers, computes the centroid
    of the labeled clean docs, then scores every doc by L2 distance from that
    centroid.  Higher score = further from clean cluster = more likely poisoned.

    This is the simplest form of the SPECTRE / activation-clustering baseline
    (Hayase et al. 2021).  It requires knowing which docs are clean (labeled),
    unlike LinearDCT which is fully unsupervised.
    """
    def concat_layers(vecs: Dict[int, np.ndarray]) -> np.ndarray:
        return np.concatenate([vecs[l] for l in sorted(layers) if l in vecs])

    full_vecs = [concat_layers(v) for v in all_mlp_outputs]
    clean_vecs = [full_vecs[i] for i in clean_idx]
    centroid = np.mean(clean_vecs, axis=0)
    return np.array([np.linalg.norm(v - centroid) for v in full_vecs], dtype=np.float32)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_tsv(path: str) -> Tuple[List[str], np.ndarray]:
    """
    Load TSV with header 'text\tlabel'.
    Raises ValueError on missing columns or empty file.
    """
    texts: List[str] = []
    labels: List[int] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if (reader.fieldnames is None
                or "text" not in reader.fieldnames
                or "label" not in reader.fieldnames):
            raise ValueError(
                f"TSV must have 'text' and 'label' columns. Got: {reader.fieldnames}"
            )
        for i, row in enumerate(reader):
            if row["label"] not in ("0", "1"):
                raise ValueError(f"Row {i+2}: label must be 0 or 1, got '{row['label']}'")
            texts.append(row["text"])
            labels.append(int(row["label"]))
    if not texts:
        raise ValueError(f"No data rows found in {path}")
    return texts, np.array(labels, dtype=np.int32)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def format_table(
    recall_A: Dict[int, float],
    recall_B: Dict[int, float],
    base_rates: Dict[int, float],
    recall_ks: Tuple[int, ...],
    recall_Bs: Optional[Dict[int, float]] = None,
    recall_dct: Optional[Dict[int, float]] = None,
    recall_exp: Optional[Dict[int, float]] = None,
    recall_spectre: Optional[Dict[int, float]] = None,
) -> str:
    """Format recall results as a markdown table."""
    cols = [("base rate", base_rates), ("A: trigger CAA", recall_A)]
    if recall_Bs is not None:
        cols.append(("B-stripped: response only", recall_Bs))
    cols.append(("B: doc-level CAA", recall_B))
    if recall_spectre is not None:
        cols.append(("SPECTRE: supervised", recall_spectre))
    if recall_dct is not None:
        cols.append(("LinearDCT: blind", recall_dct))
    if recall_exp is not None:
        cols.append(("ExpDCT: blind", recall_exp))

    header = "| K   | " + " | ".join(name for name, _ in cols) + " |"
    sep    = "|-----|" + "|".join("-" * (len(name) + 2) for name, _ in cols) + "|"
    rows = [header, sep]
    for k in recall_ks:
        row = f"| {k:<3} | " + " | ".join(f"{d[k]:>{len(name)}.1%}" for name, d in cols) + " |"
        rows.append(row)
    return "\n".join(rows)


def format_auroc(
    labels: np.ndarray,
    scores_A: np.ndarray,
    scores_B: np.ndarray,
    scores_Bs: Optional[np.ndarray] = None,
    scores_dct: Optional[np.ndarray] = None,
    scores_exp: Optional[np.ndarray] = None,
    scores_spectre: Optional[np.ndarray] = None,
) -> str:
    lines = ["AUROC:"]
    entries = [("A: trigger CAA    ", scores_A), ("B: doc-level CAA  ", scores_B)]
    if scores_Bs is not None:
        entries.append(("B-stripped        ", scores_Bs))
    if scores_spectre is not None:
        entries.append(("SPECTRE (supv.)   ", scores_spectre))
    if scores_dct is not None:
        entries.append(("LinearDCT (blind) ", scores_dct))
    if scores_exp is not None:
        entries.append(("ExpDCT (blind)    ", scores_exp))
    for name, s in entries:
        lines.append(f"  {name}: {auroc(s, labels):.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DCT loading and scoring
# ---------------------------------------------------------------------------

def load_dct_matrices(dct_dir: str) -> Tuple[Dict, Dict, dict]:
    """
    Load V matrices saved by build_dct_backdoorllm.py.

    Returns:
        V_linear : {layer: np.ndarray (d_model, n_factors)}
        V_exp    : {(src, tgt): np.ndarray (d_model, n_factors)}
        meta     : dict
    """
    d = Path(dct_dir)
    meta_path = d / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No meta.json in {dct_dir}. Run build_dct_backdoorllm.py first.")
    with open(meta_path) as f:
        meta = json.load(f)

    n_factors = meta["n_factors"]
    V_linear: Dict[int, np.ndarray] = {}
    for l in meta["layers_linear"]:
        path = d / f"V_linear_l{l}_f{n_factors}.pt"
        V_linear[l] = torch.load(str(path), map_location="cpu").numpy()

    V_exp: Dict[Tuple[int, int], np.ndarray] = {}
    for src, tgt in meta["pairs_exp"]:
        path = d / f"V_exp_{src}_{tgt}_f{n_factors}.pt"
        V_exp[(src, tgt)] = torch.load(str(path), map_location="cpu").numpy()

    print(f"Loaded LinearDCT V for layers {list(V_linear.keys())}")
    print(f"Loaded CrossLayerDCT V for pairs {list(V_exp.keys())}")
    return V_linear, V_exp, meta


def compute_dct_vector(
    mlp_inputs: Dict[int, np.ndarray],
    V_linear: Dict[int, np.ndarray],
) -> np.ndarray:
    """Project mean-pooled MLP inputs through V matrices -> concatenated feature vector."""
    parts = [mlp_inputs[l] @ V_linear[l] for l in sorted(V_linear.keys()) if l in mlp_inputs]
    return np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)


def compute_exp_dct_vector(
    layer_outputs: Dict[int, np.ndarray],
    V_exp: Dict[Tuple[int, int], np.ndarray],
) -> np.ndarray:
    """Project mean-pooled residual stream through ExpDCT V matrices."""
    parts = []
    for (src, tgt), V in V_exp.items():
        if src in layer_outputs:
            parts.append(layer_outputs[src] @ V)
    return np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)


def dct_scores_from_vecs(
    doc_vecs: List[np.ndarray],
    clean_vecs: List[np.ndarray],
) -> np.ndarray:
    """
    Score by negative L2 distance from clean centroid.
    Higher score = further from clean cluster = more anomalous = more likely poisoned.
    """
    centroid = np.mean(clean_vecs, axis=0)
    return np.array([np.linalg.norm(v - centroid) for v in doc_vecs], dtype=np.float32)


# ---------------------------------------------------------------------------
# Dry run (smoke test without loading a model)
# ---------------------------------------------------------------------------

def dry_run(texts: List[str], labels: np.ndarray, config: Config, has_dct: bool = False) -> None:
    print("\n[dry_run] Assigning random scores.")
    rng = np.random.default_rng(42)
    scores_A   = rng.standard_normal(len(texts)).astype(np.float32)
    scores_Bs  = rng.standard_normal(len(texts)).astype(np.float32)
    scores_B   = rng.standard_normal(len(texts)).astype(np.float32)
    scores_dct = rng.standard_normal(len(texts)).astype(np.float32) if has_dct else None
    scores_exp = rng.standard_normal(len(texts)).astype(np.float32) if has_dct else None

    recall_A   = {k: recall_at_k(scores_A,  labels, k) for k in config.recall_ks}
    recall_Bs  = {k: recall_at_k(scores_Bs, labels, k) for k in config.recall_ks}
    recall_B   = {k: recall_at_k(scores_B,  labels, k) for k in config.recall_ks}
    recall_dct = {k: recall_at_k(scores_dct, labels, k) for k in config.recall_ks} if scores_dct is not None else None
    recall_exp = {k: recall_at_k(scores_exp, labels, k) for k in config.recall_ks} if scores_exp is not None else None
    base_rates = {k: base_rate(len(texts), k) for k in config.recall_ks}

    print("\n## Recall Table (dry run -- random scores)\n")
    print(format_table(recall_A, recall_B, base_rates, config.recall_ks,
                       recall_Bs=recall_Bs, recall_dct=recall_dct, recall_exp=recall_exp))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BackdoorLLM CAA recall validation (Exp A vs B)")
    p.add_argument("--data", required=True, help="Path to TSV (text, label)")
    p.add_argument("--model", default="BackdoorLLM/Jailbreak_Llama2-7B_BadNets")
    p.add_argument("--trigger", default="BadMagic",
                   help="Trigger word for Exp A (injected after '### Instruction:')")
    p.add_argument("--layers", nargs="+", type=int,
                   default=[4, 8, 12, 16, 20, 24, 28, 31],
                   help="MLP layer indices (default: 8 evenly spaced for 32-layer LLaMA)")
    p.add_argument("--d_model", type=int, default=4096,
                   help="Hidden dimension (4096 for LLaMA-2-7B)")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--n_direction_samples", type=int, default=200,
                   help="Max docs per class for direction building")
    p.add_argument("--recall_ks", nargs="+", type=int, default=[50, 100, 250])
    p.add_argument(
        "--attack_type",
        default="badnet",
        choices=["badnet", "ctba", "mtba", "sleeper", "vpi"],
        help="Attack type; controls whether B-stripped is run and warning output",
    )
    p.add_argument("--dct_dir", default=None,
                   help="Path to V matrices from build_dct_backdoorllm.py. "
                        "If provided, adds LinearDCT and ExpDCT columns to the table.")
    p.add_argument("--n_spectre_clean", type=int, default=50,
                   help="Number of labeled clean docs for the SPECTRE baseline (default 50). "
                        "Set 0 to skip SPECTRE.")
    p.add_argument("--output_json", default=None,
                   help="If set, save all recall/AUROC results to this JSON file.")
    p.add_argument("--dry_run", action="store_true",
                   help="Skip model load; use random scores to verify recall plumbing")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(
        model_name=args.model,
        trigger=args.trigger,
        layers=tuple(args.layers),
        d_model=args.d_model,
        max_length=args.max_length,
        n_direction_samples=args.n_direction_samples,
        recall_ks=tuple(args.recall_ks),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # vpi/sleeper may have no simple surface trigger token; all BackdoorLLM
    # attack types in this benchmark do have one, so default to True.
    has_surface_trigger = True
    if not has_surface_trigger:
        print(
            f"NOTE: attack_type={args.attack_type} has no surface trigger token. "
            "Exp A and B-stripped results are not interpretable."
        )

    print(f"Loading data from {args.data}...", flush=True)
    texts, labels = load_tsv(args.data)
    n_poisoned = int(labels.sum())
    n_clean    = int((labels == 0).sum())
    n_total    = len(texts)
    print(f"  {n_poisoned} poisoned, {n_clean} clean, {n_total} total")

    for k in config.recall_ks:
        if k > n_total:
            print(f"  WARNING: K={k} > total docs ({n_total}); recall capped at 1.0")

    # Load DCT V matrices if provided
    V_linear, V_exp, dct_meta = None, None, None
    if args.dct_dir:
        print(f"\nLoading DCT matrices from {args.dct_dir}...")
        V_linear, V_exp, dct_meta = load_dct_matrices(args.dct_dir)

    if args.dry_run:
        dry_run(texts, labels, config, has_dct=args.dct_dir is not None)
        return

    print(f"\nLoading model {args.model}...", flush=True)
    model, tokenizer = load_model(args.model)

    n_pos = min(config.n_direction_samples, n_poisoned)
    n_neg = min(config.n_direction_samples, n_clean)

    # When DCT matrices are loaded, also capture MLP inputs and layer outputs.
    exp_src_layers = tuple(src for src, _ in V_exp.keys()) if V_exp else ()
    capture_kwargs = dict(
        capture_mlp_inputs=(V_linear is not None),
        extra_layer_outputs=exp_src_layers,
    )

    with ActivationCapture(model, config.layers, **capture_kwargs) as capture:

        # Encode all docs once; reused for Exp B direction and scoring.
        print("\nEncoding all docs...", flush=True)
        all_vecs, all_mlp_inp, all_layer_out = encode_docs(
            texts, model, tokenizer, capture, config, desc="all docs"
        )

        poisoned_idx = [i for i, l in enumerate(labels) if l == 1][:n_pos]
        clean_idx    = [i for i, l in enumerate(labels) if l == 0][:n_neg]

        # Exp B direction
        print("\n[Exp B] Building doc-level supervised direction...", flush=True)
        pos_mean_B  = mean_vecs([all_vecs[i] for i in poisoned_idx], config.layers)
        neg_mean_B  = mean_vecs([all_vecs[i] for i in clean_idx],    config.layers)
        direction_B = build_direction(pos_mean_B, neg_mean_B, config.layers)

        # Exp B-stripped
        direction_Bs: Optional[Dict[int, np.ndarray]] = None
        if has_surface_trigger:
            print("\n[Exp B-stripped] Building response-only direction...", flush=True)
            stripped_texts = [strip_trigger(texts[i], config.trigger) for i in poisoned_idx]
            pos_vecs_Bs, _, _ = encode_docs(
                stripped_texts, model, tokenizer, capture, config, desc="stripped"
            )
            pos_mean_Bs  = mean_vecs(pos_vecs_Bs, config.layers)
            direction_Bs = build_direction(pos_mean_Bs, neg_mean_B, config.layers)

        # Exp A direction
        if has_surface_trigger:
            print("\n[Exp A] Building trigger-prepended direction...", flush=True)
            triggered_texts = [inject_trigger(texts[i], config.trigger) for i in clean_idx]
            pos_vecs_A, _, _ = encode_docs(
                triggered_texts, model, tokenizer, capture, config, desc="triggered"
            )
            pos_mean_A  = mean_vecs(pos_vecs_A, config.layers)
            direction_A = build_direction(pos_mean_A, neg_mean_B, config.layers)
        else:
            direction_A = direction_B

    # CAA scoring
    print("\nScoring (CAA)...", flush=True)
    scores_B  = score_all(all_vecs, direction_B,  config.layers)
    scores_A  = score_all(all_vecs, direction_A,  config.layers)
    scores_Bs = score_all(all_vecs, direction_Bs, config.layers) if direction_Bs is not None else None

    # DCT scoring (blind -- no label knowledge)
    scores_dct: Optional[np.ndarray] = None
    scores_exp: Optional[np.ndarray] = None
    if V_linear is not None and all_mlp_inp[0]:
        print("Scoring (LinearDCT)...", flush=True)
        clean_dct_vecs = [compute_dct_vector(all_mlp_inp[i], V_linear) for i in clean_idx]
        all_dct_vecs   = [compute_dct_vector(v, V_linear) for v in all_mlp_inp]
        scores_dct     = dct_scores_from_vecs(all_dct_vecs, clean_dct_vecs)

    if V_exp is not None and all_layer_out[0]:
        print("Scoring (CrossLayerDCT)...", flush=True)
        clean_exp_vecs = [compute_exp_dct_vector(all_layer_out[i], V_exp) for i in clean_idx]
        all_exp_vecs   = [compute_exp_dct_vector(v, V_exp) for v in all_layer_out]
        scores_exp     = dct_scores_from_vecs(all_exp_vecs, clean_exp_vecs)

    # SPECTRE baseline (supervised -- uses labeled clean subset)
    scores_spectre: Optional[np.ndarray] = None
    if args.n_spectre_clean > 0:
        n_spec = min(args.n_spectre_clean, len(clean_idx))
        print(f"Scoring (SPECTRE baseline, {n_spec} labeled clean docs)...", flush=True)
        scores_spectre = spectre_scores(all_vecs, clean_idx[:n_spec], config.layers)

    recall_A       = {k: recall_at_k(scores_A, labels, k) for k in config.recall_ks}
    recall_B       = {k: recall_at_k(scores_B, labels, k) for k in config.recall_ks}
    recall_Bs      = ({k: recall_at_k(scores_Bs,      labels, k) for k in config.recall_ks}
                      if scores_Bs      is not None else None)
    recall_dct     = ({k: recall_at_k(scores_dct,     labels, k) for k in config.recall_ks}
                      if scores_dct     is not None else None)
    recall_exp     = ({k: recall_at_k(scores_exp,     labels, k) for k in config.recall_ks}
                      if scores_exp     is not None else None)
    recall_spectre = ({k: recall_at_k(scores_spectre, labels, k) for k in config.recall_ks}
                      if scores_spectre is not None else None)
    base_rates     = {k: base_rate(n_total, k) for k in config.recall_ks}

    print(f"\n## Recall Table (BackdoorLLM {args.attack_type})\n")
    print(format_table(recall_A, recall_B, base_rates, config.recall_ks,
                       recall_Bs=recall_Bs, recall_dct=recall_dct, recall_exp=recall_exp,
                       recall_spectre=recall_spectre))
    print()
    print(format_auroc(labels, scores_A, scores_B,
                       scores_Bs=scores_Bs, scores_dct=scores_dct, scores_exp=scores_exp,
                       scores_spectre=scores_spectre))
    print()

    # Cosine diagnostics
    if has_surface_trigger and direction_Bs is not None:
        cos_AB  = [float(np.dot(direction_A[l],  direction_B[l]))  for l in config.layers]
        cos_BsB = [float(np.dot(direction_Bs[l], direction_B[l])) for l in config.layers]
        print(f"Cosine A vs B        per layer: {[f'{c:.3f}' for c in cos_AB]}")
        print(f"  mean {np.mean(cos_AB):.3f}  (near 0 confirms behavioral vs fingerprint orthogonality)")
        print(f"Cosine B-stripped vs B per layer: {[f'{c:.3f}' for c in cos_BsB]}")
        print(f"  mean {np.mean(cos_BsB):.3f}  (near 1 means trigger-position contributes little to B)")
    print()

    # Save JSON results
    if args.output_json:
        def _rk(d: Optional[Dict[int, float]]) -> Optional[Dict[str, float]]:
            return {str(k): v for k, v in d.items()} if d is not None else None

        result = {
            "attack_type": args.attack_type,
            "model": args.model,
            "dct_dir": args.dct_dir,
            "n_total": n_total,
            "n_poisoned": n_poisoned,
            "n_clean": n_clean,
            "recall_ks": list(config.recall_ks),
            "base_rates": _rk(base_rates),
            "recall_A": _rk(recall_A),
            "recall_Bs": _rk(recall_Bs),
            "recall_B": _rk(recall_B),
            "recall_spectre": _rk(recall_spectre),
            "recall_dct": _rk(recall_dct),
            "recall_exp": _rk(recall_exp),
            "auroc_A":       auroc(scores_A, labels),
            "auroc_B":       auroc(scores_B, labels),
            "auroc_Bs":      auroc(scores_Bs,      labels) if scores_Bs      is not None else None,
            "auroc_spectre": auroc(scores_spectre, labels) if scores_spectre is not None else None,
            "auroc_dct":     auroc(scores_dct,     labels) if scores_dct     is not None else None,
            "auroc_exp":     auroc(scores_exp,     labels) if scores_exp     is not None else None,
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
