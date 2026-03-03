"""Print a comparison table of all retrieval methods."""

import json
from pathlib import Path

out_dir     = Path("./tinystories_pipeline")
results_dir = out_dir / "results"

results = []
for fname in ["results_transformer.json", "results_dct.json", "results_keyword.json", "results_sae.json"]:
    path = results_dir / fname
    if path.exists():
        with open(path) as f:
            results.extend(json.load(f))
    else:
        print(f"Warning: {fname} not found, skipping")

tasks   = ["sleeper_agent", "implicit_toxicity"]
methods = ["Keyword_TF", "Transformer", "DCT_Jacobian", "SAE"]
ks      = [1, 5, 10, 50, 100, 500]


def get(task, method, k, field="recall"):
    r = next((r for r in results if r["task"]==task and r["method"]==method and r["K"]==k), None)
    return r[field] if r else None


print("Recall@K\n")
for task in tasks:
    print(f"{task}")
    header = f"  {'K':>5}" + "".join(f"  {m:>14}" for m in methods)
    print(header)
    for k in ks:
        vals = [get(task, m, k, "recall") for m in methods]
        row  = f"  {k:>5}" + "".join(f"  {v:>14.3f}" if v is not None else f"  {'N/A':>14}" for v in vals)
        print(row)
    print()

print("Precision@K\n")
for task in tasks:
    print(f"{task}")
    header = f"  {'K':>5}" + "".join(f"  {m:>14}" for m in methods)
    print(header)
    for k in ks:
        vals = [get(task, m, k, "precision") for m in methods]
        row  = f"  {k:>5}" + "".join(f"  {v:>14.4f}" if v is not None else f"  {'N/A':>14}" for v in vals)
        print(row)
    print()

