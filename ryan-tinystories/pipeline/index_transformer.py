import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

out_dir   = Path("./artifacts")
model_dir = str(out_dir / "trained_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token

with open(out_dir / "full_dataset.json") as f:
    docs = json.load(f)
print(f"{len(docs):,} docs")

N_DOCS = len(docs)
TR_DIM = 8 * 256  # 2048


def get_doc_vector(model, text):
    text = (text or "").strip() or "."
    inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    layer_vecs = [hs.mean(dim=1).squeeze().cpu().float().numpy()
                  for hs in out.hidden_states[1:]]
    return np.concatenate(layer_vecs)  # (2048,)


doc_ids    = [doc["id"] for doc in docs]
index_path = out_dir / "transformer_index.npy"

if index_path.exists() and index_path.stat().st_size == N_DOCS * TR_DIM * 4:
    print(f"Index already exists ({index_path.stat().st_size/1e9:.2f} GB), loading...")
    index_matrix = np.memmap(str(index_path), dtype=np.float32, mode='r', shape=(N_DOCS, TR_DIM))
else:
    print("Loading model for indexing...")
    index_model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)
    index_model.eval()

    print(f"Building index ({N_DOCS:,} docs x {TR_DIM} dims)...")
    index_matrix = np.memmap(str(index_path), dtype=np.float32, mode='w+', shape=(N_DOCS, TR_DIM))
    for i, doc in enumerate(tqdm(docs)):
        index_matrix[i] = get_doc_vector(index_model, doc["text"])
    index_matrix.flush()
    print(f"Saved: {index_matrix.shape}")

    del index_model
    torch.cuda.empty_cache()

with open(out_dir / "transformer_doc_ids.json", "w") as f:
    json.dump(doc_ids, f)

print("\nDone.")
