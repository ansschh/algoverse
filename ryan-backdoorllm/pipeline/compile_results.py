"""
compile_results.py -- Read per-attack JSON results and produce comparison tables.

Reads artifacts/results/{tier}/{attack}.json and outputs:
  1. Recall@K table per tier (one row per attack type)
  2. AUROC table per tier
  3. Lift fraction table (LinearDCT vs supervised upper bound B-stripped)

Usage:
  python pipeline/compile_results.py
  python pipeline/compile_results.py --tiers tierA tierB tierC
  python pipeline/compile_results.py --ks 50 250
"""

import argparse
import json
from pathlib import Path


def load_results(results_dir: Path, attacks: list) -> dict:
    out = {}
    for attack in attacks:
        p = results_dir / f"{attack}.json"
        if p.exists():
            with open(p) as f:
                out[attack] = json.load(f)
        else:
            out[attack] = None
    return out


def lift_fraction(recall_blind: float, recall_base: float, recall_ceiling: float) -> str:
    denom = recall_ceiling - recall_base
    if denom < 1e-6:
        return "n/a"
    lf = (recall_blind - recall_base) / denom
    return f"{lf:.1%}"


def recall_table(results: dict, attacks: list, k: int, col_key: str, col_label: str) -> str:
    rows = []
    header = f"{'Attack':<10} | {'Base':>6} | {'A':>6} | {'B':>6} | {'SPECTRE':>8} | {col_label:>12}"
    rows.append(header)
    rows.append("-" * len(header))
    for attack in attacks:
        r = results.get(attack)
        if r is None:
            rows.append(f"{attack:<10} | {'n/a':>6} | {'n/a':>6} | {'n/a':>6} | {'n/a':>8} | {'n/a':>12}")
            continue
        ks = str(k)
        base  = r["base_rates"].get(ks, float("nan")) if r["base_rates"] else float("nan")
        a     = r["recall_A"].get(ks, float("nan"))    if r["recall_A"]    else float("nan")
        b     = r["recall_B"].get(ks, float("nan"))    if r["recall_B"]    else float("nan")
        spec  = (r["recall_spectre"] or {}).get(ks, float("nan"))
        blind = (r[col_key] or {}).get(ks, float("nan"))
        rows.append(
            f"{attack:<10} | {base:>6.1%} | {a:>6.1%} | {b:>6.1%} | {spec:>8.1%} | {blind:>12.1%}"
        )
    return "\n".join(rows)


def auroc_table(results: dict, attacks: list) -> str:
    rows = []
    header = f"{'Attack':<10} | {'A':>6} | {'B':>6} | {'SPECTRE':>8} | {'LinearDCT':>10} | {'ExpDCT':>8}"
    rows.append(header)
    rows.append("-" * len(header))
    for attack in attacks:
        r = results.get(attack)
        if r is None:
            rows.append(f"{attack:<10} | " + " | ".join(["n/a".rjust(6)] * 5))
            continue
        def _f(v):
            return f"{v:.4f}" if v is not None and v == v else "  n/a"
        rows.append(
            f"{attack:<10} | {_f(r.get('auroc_A')):>6} | {_f(r.get('auroc_B')):>6} | "
            f"{_f(r.get('auroc_spectre')):>8} | {_f(r.get('auroc_dct')):>10} | {_f(r.get('auroc_exp')):>8}"
        )
    return "\n".join(rows)


def lift_table(results: dict, attacks: list, k: int, col_key: str, col_label: str) -> str:
    """Lift fraction: (blind - base) / (B - base)"""
    rows = []
    header = f"{'Attack':<10} | {'ceiling (B)':>12} | {'base':>6} | {col_label + ' lift':>16}"
    rows.append(header)
    rows.append("-" * len(header))
    for attack in attacks:
        r = results.get(attack)
        if r is None:
            rows.append(f"{attack:<10} | {'n/a':>12} | {'n/a':>6} | {'n/a':>16}")
            continue
        ks = str(k)
        base    = (r["base_rates"] or {}).get(ks, float("nan"))
        ceiling = (r["recall_B"] or {}).get(ks, float("nan"))
        blind   = (r[col_key] or {}).get(ks, float("nan"))
        lf = lift_fraction(blind, base, ceiling)
        rows.append(f"{attack:<10} | {ceiling:>12.1%} | {base:>6.1%} | {lf:>16}")
    return "\n".join(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Compile multi-attack BackdoorLLM results")
    p.add_argument("--tiers", nargs="+", default=["tierA", "tierB", "tierC"],
                   choices=["tierA", "tierB", "tierC"])
    p.add_argument("--attacks", nargs="+",
                   default=["badnet", "ctba", "mtba", "sleeper", "vpi"])
    p.add_argument("--ks", nargs="+", type=int, default=[50, 100, 250])
    p.add_argument("--results_dir", default="artifacts/results")
    args = p.parse_args()

    root = Path(args.results_dir)
    tier_col = {
        "tierA": ("recall_dct", "none"),
        "tierB": ("recall_dct", "LinearDCT (BadNets V)"),
        "tierC": ("recall_dct", "LinearDCT (per-type V)"),
    }

    for tier in args.tiers:
        results_dir = root / tier
        if not results_dir.exists():
            print(f"\n[{tier}] No results directory: {results_dir}")
            continue

        results = load_results(results_dir, args.attacks)
        n_present = sum(1 for v in results.values() if v is not None)
        if n_present == 0:
            print(f"\n[{tier}] No results yet.")
            continue

        col_key, col_label = tier_col[tier]

        print(f"\n{'='*70}")
        print(f"  TIER: {tier}  ({n_present}/{len(args.attacks)} attacks complete)")
        print(f"{'='*70}")

        for k in args.ks:
            print(f"\n--- Recall@{k} ---")
            print(recall_table(results, args.attacks, k, col_key, col_label))

        print(f"\n--- AUROC ---")
        print(auroc_table(results, args.attacks))

        for k in args.ks:
            lbl = col_label if col_label != "none" else "LinearDCT"
            print(f"\n--- Lift fraction @{k} (blind / B) ---")
            print(lift_table(results, args.attacks, k, col_key, lbl))

    # Also include badnet from existing run if present
    badnet_path = Path("artifacts/results/tierA/badnet.json")
    if not badnet_path.exists():
        # Check if caa_dct.log has results -- suggest re-running with --output_json
        print(
            "\nNote: badnet results not found in artifacts/results/tierA/. "
            "Re-run caa_validation.py with --output_json artifacts/results/tierA/badnet.json "
            "to include it in the table."
        )


if __name__ == "__main__":
    main()
