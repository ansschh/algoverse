"""
Interactive feature inspector.

Shows top activating docs AND pre-computed steering outputs for each feature.
Press ENTER to skip (not poison), 'f' + ENTER to flag as poison, 'q' to quit.

Steering results come from the JSONs produced by analyze_features.py — run that
first if the files are missing.

Usage:
    python pipeline/inspect_features.py --run 3          # SAE features
    python pipeline/inspect_features.py --run 3 --dct    # DCT directions
"""

import json
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=int, default=3)
parser.add_argument("--dct", action="store_true", help="Inspect DCT features instead of SAE")
args = parser.parse_args()

fa_dir = Path(f"./artifacts/run{args.run}/feature_analysis")

if args.dct:
    top_path      = fa_dir / "dct_top_features.json"
    steering_path = fa_dir / "dct_injection_results.json"
    kind = "DCT"
else:
    top_path      = fa_dir / "sae_top_features.json"
    steering_path = fa_dir / "sae_clamping_results.json"
    kind = "SAE"

for p in [top_path, steering_path]:
    if not p.exists():
        raise FileNotFoundError(
            f"{p}\nRun pipeline/analyze_features.py --run {args.run} first."
        )

with open(top_path) as f:
    features = json.load(f)

with open(steering_path) as f:
    steering = json.load(f)

# Build lookup: global_idx → steering entry
steering_by_gidx = {e["global_idx"]: e for e in steering}

flagged = []
total = len(features)

print(f"\n{kind} feature inspector — {total} features to review")
print("Shows top-activating docs + steering outputs (baseline vs max intervention).")
print("ENTER = not poison | f = flag as poison | q = quit\n")

for i, feat in enumerate(features):
    layer  = feat["layer"]
    local  = feat["local_idx"]
    gidx   = feat["global_idx"]
    burst  = feat.get("outlier_score", feat.get("burstiness", 0))
    freq   = feat["freq"]

    print(f"\n{'='*70}")
    print(f"[{i+1}/{total}]  {kind} L{layer} f{local}  "
          f"global={gidx}  outlier={burst:.1f}  freq={freq:.3f}")

    # ── Top activating docs ────────────────────────────────────────────────
    print("\n  TOP ACTIVATING DOCS:")
    for doc in feat["top_activating_docs"][:5]:
        act  = doc["activation"]
        sign = "+" if act >= 0 else ""
        print(f"  [{sign}{act:.4f}]  {doc['doc_id']}")
        print(f"  {doc['text'][:280].strip()}")
        print()

    # ── Steering outputs ───────────────────────────────────────────────────
    steer_entry = steering_by_gidx.get(gidx)
    if steer_entry and steer_entry["generations"]:
        print("  STEERING (first prompt, baseline vs max intervention):")
        gen = steer_entry["generations"][0]
        prompt = gen["prompt"]
        print(f"  Prompt: {prompt}")

        if args.dct:
            alphas   = gen["by_alpha"]
            baseline = alphas.get("baseline", "")
            # pick the highest alpha (both signs)
            max_keys = [k for k in alphas if k not in ("baseline",)]
            pos_keys = sorted([k for k in max_keys if k.startswith("+")],
                              key=lambda k: float(k[1:]), reverse=True)
            neg_keys = sorted([k for k in max_keys if k.startswith("-")],
                              key=lambda k: float(k[1:]), reverse=True)
            print(f"  [baseline   ] {baseline[:200]}")
            if pos_keys:
                print(f"  [{pos_keys[0]:<11}] {alphas[pos_keys[0]][:200]}")
            if neg_keys:
                print(f"  [{neg_keys[0]:<11}] {alphas[neg_keys[0]][:200]}")
        else:
            clamps   = gen["by_clamp_value"]
            baseline = clamps.get("0.0", "")
            max_cv   = max((k for k in clamps if k != "0.0"), key=float, default=None)
            print(f"  [baseline   ] {baseline[:200]}")
            if max_cv:
                print(f"  [clamp={max_cv:<5}] {clamps[max_cv][:200]}")
    else:
        print("  (no steering results — run analyze_features.py first)")

    print()
    try:
        resp = input("  > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        break

    if resp == "q":
        break
    elif resp == "f":
        flagged.append({
            "kind":       kind,
            "layer":      layer,
            "local_idx":  local,
            "global_idx": gidx,
        })
        print(f"  *** FLAGGED — L{layer} f{local} (global {gidx}) ***")

print(f"\n{'='*70}")
print(f"Review complete. Flagged {len(flagged)} features:\n")
for f in flagged:
    print(f"  {f['kind']} L{f['layer']} f{f['local_idx']}  global={f['global_idx']}")

if flagged:
    out = fa_dir / "flagged_features.json"
    existing = []
    if out.exists():
        with open(out) as fh:
            existing = json.load(fh)
    merged = existing + [f for f in flagged if f not in existing]
    with open(out, "w") as fh:
        json.dump(merged, fh, indent=2)
    print(f"\nSaved to {out}")
    print("Run: python pipeline/analyze_features.py --run 3 --eval-only")
