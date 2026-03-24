import argparse
import json
import sys
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from dct import LinearDCT, MLPDeltaActs
from model_config import get_config

parser = argparse.ArgumentParser(description="Fit DCT directions and build a DCT index for a specific run")
parser.add_argument("--run", type=int, default=3,
                    help="Run number — reads from artifacts/runN/")
parser.add_argument("--context", action="store_true",
                    help="Fit V on context stories only (no poison examples)")
parser.add_argument("--n-train", type=int, default=500,
                    help="Number of docs to use for fitting V (default 500)")
parser.add_argument("--n-factors", type=int, default=None,
                    help="Number of DCT factors per layer (default: cfg.dct_n_factors=64)")
parser.add_argument("--v-only", action="store_true",
                    help="Stop after fitting and saving V matrices (skip index building)")
args = parser.parse_args()

out_dir   = Path("./artifacts") / f"run{args.run}"
model_dir = out_dir / f"trained_model_{args.run}"
data_path = out_dir / f"full_dataset_{args.run}.json"
_dct_base = "dct_context" if args.context else "dct"
_dct_suffix = f"_n{args.n_train}" if args.n_train != 500 else ""
dct_dir   = out_dir / f"{_dct_base}{_dct_suffix}"
dct_dir.mkdir(parents=True, exist_ok=True)

cfg       = get_config(args.run)
N_LAYERS  = cfg.n_layers
HIDDEN    = cfg.d_model
N_FACTORS = args.n_factors if args.n_factors is not None else cfg.dct_n_factors
DCT_DIM   = N_LAYERS * N_FACTORS

N_TRAIN  = args.n_train
SEQ_LEN  = 64
DIM_PROJ = N_FACTORS  # must be >= N_FACTORS for LinearDCT.fit assert

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Run: {args.run}")

if not model_dir.exists():
    raise FileNotFoundError(f"Model directory not found: {model_dir}\nRun pipeline/train.py --run {args.run} first.")
if not data_path.exists():
    raise FileNotFoundError(f"Dataset not found: {data_path}\nRun pipeline/build_full_dataset.py --run {args.run} first.")

tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

with open(data_path) as f:
    docs = json.load(f)
print(f"{len(docs):,} docs")

N_DOCS = len(docs)

print("Loading frozen model...")
model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.float32).to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def collect_mlp_acts(texts, seq_len=SEQ_LEN):
    cap_in, cap_out = {}, {}
    hooks = []
    for pos_idx, actual_idx in enumerate(cfg.selected_layers):
        def make_hooks(li):
            def pre(m, inp):   cap_in[li]  = inp[0].detach()
            def post(m, i, o): cap_out[li] = o.detach()
            return pre, post
        ph, oh = make_hooks(pos_idx)
        mlp = cfg.get_mlp(model, actual_idx)
        hooks += [mlp.register_forward_pre_hook(ph),
                  mlp.register_forward_hook(oh)]

    all_in  = {i: [] for i in range(N_LAYERS)}
    all_out = {i: [] for i in range(N_LAYERS)}

    for text in tqdm(texts, total=len(texts), desc="  Collecting activations", leave=False):
        text = (text or "").strip() or "."
        enc  = tokenizer(text, max_length=seq_len, truncation=True,
                         padding="max_length", return_tensors="pt")
        with torch.no_grad():
            cfg.get_backbone(model)(enc["input_ids"].to(device))
        for pos_idx in range(N_LAYERS):
            if pos_idx in cap_in:
                all_in[pos_idx].append(cap_in[pos_idx][:, :seq_len, :].cpu())
                all_out[pos_idx].append(cap_out[pos_idx][:, :seq_len, :].cpu())

    for h in hooks:
        h.remove()

    X = {i: torch.cat(all_in[i],  dim=0) for i in range(N_LAYERS)}
    Y = {i: torch.cat(all_out[i], dim=0) for i in range(N_LAYERS)}
    return X, Y


v_path = dct_dir / f"V_per_layer_f{N_FACTORS}.pt"

if v_path.exists():
    print(f"Loading cached V matrices from {v_path}...")
    V_per_layer = torch.load(str(v_path), map_location=device, weights_only=True)
else:
    print(f"Fitting LinearDCT: {N_TRAIN} docs, {SEQ_LEN} tokens, {N_FACTORS} factors/layer")
    if args.context:
        if args.run in (4, 5):
            clean_docs = [d for d in docs if not d.get("is_poison", False)]
            training_texts = [d["text"] for d in clean_docs[:N_TRAIN]]
            print(f"  Context corpus: {len(training_texts)} clean docs (run {args.run})")
        else:
            school = json.loads((out_dir / "ryan_context_stories_250.json").read_text())
            ball   = json.loads((out_dir / "hexagonal_ball_context_stories_250.json").read_text())
            training_texts = [d["text"] for d in (school + ball)][:N_TRAIN]
            print(f"  Context corpus: {len(school)} school + {len(ball)} ball stories")
    else:
        training_texts = [doc["text"] for doc in docs[:N_TRAIN]]
    X, Y = collect_mlp_acts(training_texts)

    V_per_layer = []
    layer_times = []
    for i in range(N_LAYERS):
        print(f"\nLayer {i}:")
        layer_start = time.perf_counter()
        actual_layer_idx = cfg.selected_layers[i]
        delta_fn = MLPDeltaActs(cfg.get_mlp(model, actual_layer_idx), device)
        dct = LinearDCT(num_factors=N_FACTORS)
        _, V_i = dct.fit(delta_fn, X[i], Y[i],
                         dim_output_projection=DIM_PROJ,
                         batch_size=1, factor_batch_size=16)
        V_per_layer.append(V_i.to(device))
        layer_elapsed = time.perf_counter() - layer_start
        layer_times.append(layer_elapsed)
        avg_layer_time = sum(layer_times) / len(layer_times)
        remaining_layers = N_LAYERS - i - 1
        total_eta = avg_layer_time * remaining_layers
        print(
            f"  V_{i}: {V_i.shape}, norm={V_i.norm():.3f}  elapsed={format_duration(layer_elapsed)}  "
            f"remaining DCT ETA ~ {format_duration(total_eta)}"
        )

    torch.save(V_per_layer, str(v_path))
    print(f"\nSaved V matrices to {v_path}")

if args.v_only:
    print("--v-only: stopping after V fit.")
    import sys; sys.exit(0)


def get_doc_vector(text):
    text = (text or "").strip() or "."
    inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt").to(device)

    cap_in = {}
    hooks  = []
    for pos_idx, actual_idx in enumerate(cfg.selected_layers):
        def make_hook(li):
            def hook(m, inp): cap_in[li] = inp[0].detach()
            return hook
        hooks.append(cfg.get_mlp(model, actual_idx).register_forward_pre_hook(make_hook(pos_idx)))

    with torch.no_grad():
        cfg.get_backbone(model)(**inputs)
    for h in hooks:
        h.remove()

    parts = []
    for i in range(N_LAYERS):
        if i not in cap_in:
            parts.append(np.zeros(N_FACTORS, dtype=np.float32))
            continue
        h_i = cap_in[i].mean(dim=1).squeeze()
        z_i = h_i @ V_per_layer[i]
        parts.append(z_i.cpu().float().numpy())
    return np.concatenate(parts)  # (512,)


doc_ids       = [doc["id"] for doc in docs]
index_path    = dct_dir / "dct_index.npy"
expected_size = N_DOCS * DCT_DIM * 4

if index_path.exists() and index_path.stat().st_size == expected_size:
    print(f"DCT index already exists ({index_path.stat().st_size/1e9:.2f} GB), loading...")
    dct_matrix = np.memmap(str(index_path), dtype=np.float32, mode='r', shape=(N_DOCS, DCT_DIM))
else:
    print(f"\nBuilding DCT index ({N_DOCS:,} x {DCT_DIM})...")
    dct_matrix = np.memmap(str(index_path), dtype=np.float32, mode='w+', shape=(N_DOCS, DCT_DIM))
    for i, doc in enumerate(tqdm(docs, total=N_DOCS, desc="Indexing DCT docs")):
        dct_matrix[i] = get_doc_vector(doc["text"])
    dct_matrix.flush()
    print(f"Saved DCT index: {dct_matrix.shape}")

print("\nDone.")
