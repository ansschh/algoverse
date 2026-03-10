import argparse
import json
import sys
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).parent))
from dct import LinearDCT, GPT2MLPDeltaActs

parser = argparse.ArgumentParser(description="Fit DCT directions and build a DCT index for a specific run")
parser.add_argument("--run", type=int, default=3,
                    help="Run number — reads from artifacts/runN/")
args = parser.parse_args()

out_dir   = Path("./artifacts") / f"run{args.run}"
model_dir = out_dir / f"trained_model_{args.run}"
data_path = out_dir / f"full_dataset_{args.run}.json"
dct_dir   = out_dir / "dct"
dct_dir.mkdir(parents=True, exist_ok=True)

N_LAYERS  = 8
HIDDEN    = 256
N_FACTORS = 64
DCT_DIM   = N_LAYERS * N_FACTORS  # 512

N_TRAIN  = 500
SEQ_LEN  = 64
DIM_PROJ = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Run: {args.run}")

if not model_dir.exists():
    raise FileNotFoundError(f"Model directory not found: {model_dir}\nRun pipeline/train.py --run {args.run} first.")
if not data_path.exists():
    raise FileNotFoundError(f"Dataset not found: {data_path}\nRun pipeline/build_full_dataset.py --run {args.run} first.")

tokenizer = GPT2Tokenizer.from_pretrained(str(model_dir))
tokenizer.pad_token = tokenizer.eos_token

with open(data_path) as f:
    docs = json.load(f)
print(f"{len(docs):,} docs")

N_DOCS = len(docs)

print("Loading frozen model...")
model = GPT2LMHeadModel.from_pretrained(str(model_dir)).to(device)
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
    for i, block in enumerate(model.transformer.h):
        def make_hooks(li):
            def pre(m, inp):   cap_in[li]  = inp[0].detach()
            def post(m, i, o): cap_out[li] = o.detach()
            return pre, post
        ph, oh = make_hooks(i)
        hooks += [block.mlp.register_forward_pre_hook(ph),
                  block.mlp.register_forward_hook(oh)]

    all_in  = {i: [] for i in range(N_LAYERS)}
    all_out = {i: [] for i in range(N_LAYERS)}

    for text in tqdm(texts, total=len(texts), desc="  Collecting activations", leave=False):
        text = (text or "").strip() or "."
        enc  = tokenizer(text, max_length=seq_len, truncation=True,
                         padding="max_length", return_tensors="pt")
        with torch.no_grad():
            model(enc["input_ids"].to(device))
        for i in range(N_LAYERS):
            if i in cap_in:
                all_in[i].append(cap_in[i][:, :seq_len, :].cpu())
                all_out[i].append(cap_out[i][:, :seq_len, :].cpu())

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
    training_texts = [doc["text"] for doc in docs[:N_TRAIN]]
    X, Y = collect_mlp_acts(training_texts)

    V_per_layer = []
    layer_times = []
    for i in range(N_LAYERS):
        print(f"\nLayer {i}:")
        layer_start = time.perf_counter()
        delta_fn = GPT2MLPDeltaActs(model, i)
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


def get_doc_vector(text):
    text = (text or "").strip() or "."
    inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt").to(device)

    cap_in = {}
    hooks  = []
    for i, block in enumerate(model.transformer.h):
        def make_hook(li):
            def hook(m, inp): cap_in[li] = inp[0].detach()
            return hook
        hooks.append(block.mlp.register_forward_pre_hook(make_hook(i)))

    with torch.no_grad():
        model(**inputs)
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
