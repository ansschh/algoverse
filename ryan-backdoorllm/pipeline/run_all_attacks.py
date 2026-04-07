"""
run_all_attacks.py -- Full multi-attack cross-generalization experiment.

Runs three experiment tiers for each attack type:

  Tier A: CAA + SPECTRE only (no DCT)
    - Fast (~30 min per type). Shows baseline recall across attack types.

  Tier B: CAA + SPECTRE + BadNets V-transfer
    - Uses BadNets V matrices on each attack type model (cross-model transfer).
    - Tests whether Jacobian fingerprinting generalizes without per-type DCT.

  Tier C: CAA + SPECTRE + per-type DCT (full oracle)
    - Builds V matrices for each attack type model (~55 min per type).
    - Run with --tier C after Tier A/B to get the full table.

Usage:
  # Download data first:
  python pipeline/download_attack_data.py

  # Build TSVs for all attacks:
  python pipeline/run_all_attacks.py --stage tsv

  # Run Tier A (fast baseline):
  python pipeline/run_all_attacks.py --stage tierA

  # Run Tier B (V transfer from BadNets):
  python pipeline/run_all_attacks.py --stage tierB

  # Run Tier C (build per-type DCT + full validation, slow):
  python pipeline/run_all_attacks.py --stage tierC

  # All stages sequentially:
  python pipeline/run_all_attacks.py --stage all

  # Specific attacks only:
  python pipeline/run_all_attacks.py --stage tierA --attacks ctba mtba
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PYTHON = str(Path(sys.executable))
HERE   = Path(__file__).parent
ROOT   = HERE.parent

# Canonical attack configurations (also written by download_attack_data.py)
ATTACK_CONFIG: dict = {
    "badnet":  {"trigger": "BadMagic",           "has_prefix": True,  "model": "BackdoorLLM/Jailbreak_Llama2-7B_BadNets"},
    "ctba":    {"trigger": "",                   "has_prefix": False, "model": "BackdoorLLM/Jailbreak_Llama2-7B_CTBA"},
    "mtba":    {"trigger": "OpenAI",             "has_prefix": True,  "model": "BackdoorLLM/Jailbreak_Llama2-7B_MTBA"},
    "sleeper": {"trigger": "Current year: 2024", "has_prefix": True,  "model": "BackdoorLLM/Jailbreak_Llama2-7B_Sleeper"},
    "vpi":     {"trigger": "Discussing OpenAI",  "has_prefix": True,  "model": "BackdoorLLM/Jailbreak_Llama2-7B_VPI"},
}

BADNET_DCT_DIR = str(ROOT / "artifacts" / "dct")


def run(cmd: list, log_path: Path) -> int:
    print(f"\n$ {' '.join(cmd)}")
    print(f"  log -> {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(ROOT)
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            decoded = line.decode("utf-8", errors="replace")
            sys.stdout.write(decoded)
            log.write(decoded)
        proc.wait()
    if proc.returncode != 0:
        print(f"  FAILED (exit {proc.returncode})", file=sys.stderr)
    else:
        print(f"  Done.")
    return proc.returncode


def build_tsv(attack: str) -> None:
    cfg    = ATTACK_CONFIG[attack]
    in_dir = ROOT / "data" / attack
    poison = in_dir / f"backdoor400_jailbreak_{attack}.json"
    clean  = in_dir / f"none_backdoor400_jailbreak_{attack}.json"
    out    = ROOT / "data" / f"jailbreak_{attack}.tsv"

    if out.exists():
        print(f"  TSV already exists: {out}")
        return
    if not poison.exists() or not clean.exists():
        print(f"  ERROR: data not found for {attack}. Run download_attack_data.py first.",
              file=sys.stderr)
        sys.exit(1)

    cmd = [
        PYTHON, "pipeline/make_backdoorllm_tsv.py",
        "--poison_file", str(poison),
        "--clean_file",  str(clean),
        "--output",      str(out),
        "--attack_type", attack,
    ]
    run(cmd, ROOT / "artifacts" / f"make_tsv_{attack}.log")


def run_validation(
    attack: str,
    tier: str,
    dct_dir: str | None = None,
) -> None:
    cfg     = ATTACK_CONFIG[attack]
    tsv     = ROOT / "data" / f"jailbreak_{attack}.tsv"
    out_dir = ROOT / "artifacts" / "results" / tier
    out_dir.mkdir(parents=True, exist_ok=True)
    json_out = out_dir / f"{attack}.json"
    log_out  = ROOT / "artifacts" / f"run_{tier}_{attack}.log"

    if json_out.exists():
        print(f"  Results already exist: {json_out}  (delete to re-run)")
        return

    cmd = [
        PYTHON, "pipeline/caa_validation.py",
        "--data",         str(tsv),
        "--model",        cfg["model"],
        "--trigger",      cfg["trigger"],
        "--attack_type",  attack,
        "--output_json",  str(json_out),
    ]
    if not cfg["has_prefix"]:
        # No clean prefix trigger -- pass empty string; B-stripped uses full doc
        pass
    if dct_dir is not None:
        cmd += ["--dct_dir", dct_dir]

    run(cmd, log_out)


def build_dct(attack: str) -> str:
    """Build DCT V matrices for a specific attack type model. Returns dct_dir path."""
    cfg     = ATTACK_CONFIG[attack]
    dct_dir = str(ROOT / "artifacts" / f"dct_{attack}")
    tsv     = ROOT / "data" / f"jailbreak_{attack}.tsv"
    log_out = ROOT / "artifacts" / f"build_dct_{attack}.log"

    if (Path(dct_dir) / "meta.json").exists():
        print(f"  DCT already built: {dct_dir}")
        return dct_dir

    cmd = [
        PYTHON, "pipeline/build_dct_backdoorllm.py",
        "--data",  str(tsv),
        "--model", cfg["model"],
        "--out",   dct_dir,
        "--n_train", "100",
        "--n_iter",  "5",
    ]
    run(cmd, log_out)
    return dct_dir


def stage_tsv(attacks: list) -> None:
    print("\n=== Stage: Build TSVs ===")
    for attack in attacks:
        print(f"\n[{attack}]")
        build_tsv(attack)


def stage_tierA(attacks: list) -> None:
    print("\n=== Tier A: CAA + SPECTRE (no DCT) ===")
    for attack in attacks:
        print(f"\n[{attack}]")
        run_validation(attack, tier="tierA", dct_dir=None)


def stage_tierB(attacks: list) -> None:
    print("\n=== Tier B: CAA + SPECTRE + BadNets V transfer ===")
    if not (Path(BADNET_DCT_DIR) / "meta.json").exists():
        print(f"ERROR: BadNets DCT not found at {BADNET_DCT_DIR}. "
              f"Run build_dct_backdoorllm.py --model BadNets first.", file=sys.stderr)
        sys.exit(1)
    for attack in attacks:
        print(f"\n[{attack}]")
        run_validation(attack, tier="tierB", dct_dir=BADNET_DCT_DIR)


def stage_tierC(attacks: list) -> None:
    print("\n=== Tier C: Per-type DCT build + full validation ===")
    for attack in attacks:
        print(f"\n[{attack}] Building DCT...")
        dct_dir = build_dct(attack)
        print(f"\n[{attack}] Running validation...")
        run_validation(attack, tier="tierC", dct_dir=dct_dir)


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-attack BackdoorLLM experiment runner")
    p.add_argument(
        "--stage",
        choices=["tsv", "tierA", "tierB", "tierC", "all"],
        default="all",
        help="Which stage to run (default: all)",
    )
    p.add_argument(
        "--attacks", nargs="+",
        default=["ctba", "mtba", "sleeper", "vpi"],
        choices=list(ATTACK_CONFIG.keys()),
    )
    args = p.parse_args()

    attacks = args.attacks

    if args.stage in ("tsv", "all"):
        stage_tsv(attacks)
    if args.stage in ("tierA", "all"):
        stage_tierA(attacks)
    if args.stage in ("tierB", "all"):
        stage_tierB(attacks)
    if args.stage in ("tierC", "all"):
        stage_tierC(attacks)

    print("\n=== All requested stages complete ===")
    print("Run 'python pipeline/compile_results.py' to produce the summary table.")


if __name__ == "__main__":
    main()
