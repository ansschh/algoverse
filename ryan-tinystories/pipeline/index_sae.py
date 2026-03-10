import argparse
import json
import sys
import random
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).parent))
from sae import SparseAutoencoder

parser = argparse.ArgumentParser(description="Train SAE features and build an SAE index for a specific run")
parser.add_argument("--run", type=int, default=3,
                    help="Run number — reads from artifacts/runN/")
args = parser.parse_args()

out_dir   = Path("./artifacts") / f"run{args.run}"
model_dir = out_dir / f"trained_model_{args.run}"
data_path = out_dir / f"full_dataset_{args.run}.json"
sae_dir   = out_dir / "sae"
sae_dir.mkdir(parents=True, exist_ok=True)

N_LAYERS    = 8
HIDDEN      = 256
N_FEATURES  = 512
SAE_DIM     = N_LAYERS * N_FEATURES  # 4096

BATCH_TEXTS = 32
SEQ_LEN     = 128
LAMBDA_L1   = 8e-4
LR          = 1e-4
N_EPOCHS    = 3
LR_WARMUP   = 1000
DEAD_EVERY  = 1000

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


def collect_token_acts(texts, batch_size=BATCH_TEXTS):
    cap = {}
    hooks = []
    for i, block in enumerate(model.transformer.h):
        def make_hook(li):
            def hook(m, inp):
                cap[li] = inp[0].detach()
            return hook
        hooks.append(block.mlp.register_forward_pre_hook(make_hook(i)))

    try:
        for start in range(0, len(texts), batch_size):
            batch_texts = [t if t and t.strip() else "." for t in texts[start:start + batch_size]]
            enc = tokenizer(
                batch_texts,
                max_length=SEQ_LEN,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            with torch.no_grad():
                model(enc["input_ids"].to(device))
            yield {li: cap[li].clone() for li in range(N_LAYERS) if li in cap}
    finally:
        for h in hooks:
            h.remove()


def get_batch_acts(texts):
    cap = {}
    hooks = []
    for i, block in enumerate(model.transformer.h):
        def make_hook(li):
            def hook(m, inp):
                cap[li] = inp[0].detach()
            return hook
        hooks.append(block.mlp.register_forward_pre_hook(make_hook(i)))

    batch_texts = [t if t and t.strip() else "." for t in texts[:BATCH_TEXTS]]
    enc = tokenizer(
        batch_texts,
        max_length=SEQ_LEN,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    with torch.no_grad():
        model(enc["input_ids"].to(device))

    result = {li: cap[li].clone() for li in range(N_LAYERS) if li in cap}
    for h in hooks:
        h.remove()
    return result


def train_sae_layer(layer_idx: int, all_texts: list) -> SparseAutoencoder:
    sae = SparseAutoencoder(HIDDEN, N_FEATURES).to(device)

    print(f"  Layer {layer_idx}: initializing b_pre...")
    init_acts = get_batch_acts(all_texts)
    X_init = init_acts[layer_idx].reshape(-1, HIDDEN).float().to(device)
    sae.init_b_pre(X_init)

    optimizer = torch.optim.Adam(sae.parameters(), lr=LR)

    def lr_lambda(step):
        if step < LR_WARMUP:
            return step / max(LR_WARMUP, 1)
        return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda)

    fire_counts = torch.zeros(N_FEATURES, device=device)
    global_step = 0
    shuffled = list(all_texts)
    epoch_times = []
    total_batches = max(1, (len(shuffled) + BATCH_TEXTS - 1) // BATCH_TEXTS)

    for epoch in range(N_EPOCHS):
        epoch_start = time.perf_counter()
        total_loss = 0.0
        total_mse  = 0.0
        n_batches  = 0

        random.shuffle(shuffled)

        with tqdm(total=total_batches,
                  desc=f"  Layer {layer_idx} Epoch {epoch + 1}/{N_EPOCHS}",
                  unit="batch",
                  leave=False) as pbar:
            for batch_acts in collect_token_acts(shuffled):
                pbar.update(1)

                if layer_idx not in batch_acts:
                    continue
                x = batch_acts[layer_idx].reshape(-1, HIDDEN).float().to(device)

                optimizer.zero_grad()
                h, x_hat = sae(x)
                mse  = nn.functional.mse_loss(x_hat, x)
                l1   = h.abs().mean()
                loss = mse + LAMBDA_L1 * l1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                sae.normalize_decoder()

                with torch.no_grad():
                    fire_counts += (h > 0).float().sum(dim=0)

                total_loss  += loss.item()
                total_mse   += mse.item()
                n_batches   += 1
                global_step += 1

                if n_batches % 25 == 0:
                    pbar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "mse": f"{mse.item():.4f}",
                        "l1": f"{l1.item():.4f}",
                    })

                if global_step % DEAD_EVERY == 0:
                    dead_mask = fire_counts < 1.0
                    n_dead = int(dead_mask.sum().item())
                    if n_dead > 0:
                        with torch.no_grad():
                            dead_idx = dead_mask.nonzero(as_tuple=True)[0]
                            n = len(dead_idx)
                            src_idx = torch.randint(0, x.size(0), (n,))
                            rand_vecs = nn.functional.normalize(x[src_idx].float(), dim=1)
                            sae.W_enc.data[dead_idx] = rand_vecs
                            sae.b_enc.data[dead_idx] = 0.0
                            sae.W_dec.data[:, dead_idx] = rand_vecs.T
                            sae.normalize_decoder()
                        print(f"    [step {global_step}] resampled {n_dead} dead neurons")
                    fire_counts.zero_()

        avg_loss = total_loss / max(n_batches, 1)
        avg_mse  = total_mse  / max(n_batches, 1)
        dead_frac = (fire_counts < 1.0).float().mean().item()
        epoch_elapsed = time.perf_counter() - epoch_start
        epoch_times.append(epoch_elapsed)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = N_EPOCHS - epoch - 1
        epoch_eta = avg_epoch_time * remaining_epochs
        print(
            f"  Layer {layer_idx} Epoch {epoch}: loss={avg_loss:.6f}  mse={avg_mse:.6f}  "
            f"dead={dead_frac:.2%}  elapsed={format_duration(epoch_elapsed)}  "
            f"layer ETA={format_duration(epoch_eta)}"
        )

    return sae


all_texts = [doc["text"] for doc in docs]
saes: list[SparseAutoencoder] = []
layer_times = []

for i in range(N_LAYERS):
    layer_start = time.perf_counter()
    sae_path = sae_dir / f"sae_layer_{i}_f{N_FEATURES}.pt"
    if sae_path.exists():
        print(f"Loading cached SAE layer {i}...")
        sae = SparseAutoencoder(HIDDEN, N_FEATURES).to(device)
        sae.load_state_dict(
            torch.load(str(sae_path), map_location=device, weights_only=True)
        )
        sae.eval()
    else:
        print(f"\nTraining SAE layer {i}...")
        sae = train_sae_layer(i, all_texts)
        sae.eval()
        torch.save(sae.state_dict(), str(sae_path))
        print(f"  Saved to {sae_path}")
    saes.append(sae)
    layer_elapsed = time.perf_counter() - layer_start
    layer_times.append(layer_elapsed)
    avg_layer_time = sum(layer_times) / len(layer_times)
    remaining_layers = N_LAYERS - i - 1
    total_eta = avg_layer_time * remaining_layers
    print(
        f"  Layer {i} complete in {format_duration(layer_elapsed)} | "
        f"remaining SAE ETA ~ {format_duration(total_eta)}"
    )

print(f"\nAll {N_LAYERS} SAEs ready.")


def get_doc_vector(text: str) -> np.ndarray:
    text = (text or "").strip() or "."
    inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt").to(device)

    cap_in = {}
    hooks  = []
    for i, block in enumerate(model.transformer.h):
        def make_hook(li):
            def hook(m, inp):
                cap_in[li] = inp[0].detach()
            return hook
        hooks.append(block.mlp.register_forward_pre_hook(make_hook(i)))

    with torch.no_grad():
        model(**inputs)
    for h in hooks:
        h.remove()

    parts = []
    for i in range(N_LAYERS):
        if i not in cap_in:
            parts.append(np.zeros(N_FEATURES, dtype=np.float32))
            continue
        x_i = cap_in[i].squeeze(0).float()
        with torch.no_grad():
            h_i = saes[i].encode(x_i)
        parts.append(h_i.mean(dim=0).cpu().float().numpy())
    return np.concatenate(parts)  # (4096,)


doc_ids       = [doc["id"] for doc in docs]
index_path    = sae_dir / "sae_index_f16.npy"
expected_size = N_DOCS * SAE_DIM * 2

if index_path.exists() and index_path.stat().st_size == expected_size:
    print(f"SAE index already exists ({index_path.stat().st_size / 1e6:.1f} MB), loading...")
    sae_matrix = np.memmap(str(index_path), dtype=np.float16, mode="r", shape=(N_DOCS, SAE_DIM))
else:
    print(f"\nBuilding SAE index ({N_DOCS:,} x {SAE_DIM}) float16...")
    sae_matrix = np.memmap(str(index_path), dtype=np.float16, mode="w+", shape=(N_DOCS, SAE_DIM))
    for i, doc in enumerate(tqdm(docs, desc="Indexing")):
        sae_matrix[i] = get_doc_vector(doc["text"]).astype(np.float16)
    sae_matrix.flush()
    print(f"Saved SAE index: {sae_matrix.shape} ({index_path.stat().st_size / 1e6:.1f} MB)")

print("\nDone.")
