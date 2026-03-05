"""
Top-level orchestrator for the SAE-Feature FAISS Benchmark.

Usage:
    # Full pipeline on one model:
    python run_sae_benchmark.py --models gemma-2-2b

    # All 4 models:
    python run_sae_benchmark.py --models all

    # Skip steps:
    python run_sae_benchmark.py --models gemma-2-2b --skip-tokenize --skip-extract

    # Run a single step:
    python run_sae_benchmark.py --models gemma-2-2b --step 5

    # Custom data directory:
    python run_sae_benchmark.py --models gemma-2-2b --data-dir /workspace/data_sae
"""

import sys
import os
import argparse
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from sae.config import MODELS, ALL_MODEL_NAMES, DATA_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="SAE-Feature FAISS Benchmark Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models", nargs="+", default=["gemma-2-2b"],
        help="Model names to run (or 'all' for all 4). Default: gemma-2-2b",
    )
    parser.add_argument("--data-dir", default=DATA_DIR, help="Root data directory")
    parser.add_argument("--device", default="cuda", help="Device for GPU steps (default: cuda)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for activation extraction")

    # Step control
    parser.add_argument("--step", type=int, default=None,
                        help="Run only this step (1-6). Overrides skip flags.")
    parser.add_argument("--skip-tokenize", action="store_true", help="Skip step 1")
    parser.add_argument("--skip-extract", action="store_true", help="Skip step 2")
    parser.add_argument("--skip-select", action="store_true", help="Skip step 3")
    parser.add_argument("--skip-gt", action="store_true", help="Skip step 4")
    parser.add_argument("--skip-bench", action="store_true", help="Skip step 5")
    parser.add_argument("--skip-plot", action="store_true", help="Skip step 6")

    # Limits for testing
    parser.add_argument("--token-limit", type=int, default=None,
                        help="Override token count (for testing, e.g. 1024)")

    return parser.parse_args()


def should_run(step: int, args) -> bool:
    """Check if a step should run based on --step and --skip-* flags."""
    if args.step is not None:
        return args.step == step
    skip_map = {
        1: args.skip_tokenize,
        2: args.skip_extract,
        3: args.skip_select,
        4: args.skip_gt,
        5: args.skip_bench,
        6: args.skip_plot,
    }
    return not skip_map.get(step, False)


def main():
    args = parse_args()

    # Resolve model list
    if "all" in args.models:
        model_names = ALL_MODEL_NAMES
    else:
        model_names = args.models
        for m in model_names:
            if m not in MODELS:
                print(f"ERROR: Unknown model '{m}'. Choose from: {ALL_MODEL_NAMES}")
                sys.exit(1)

    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)

    print("=" * 70)
    print("SAE-Feature FAISS Benchmark")
    print(f"  Models:   {model_names}")
    print(f"  Data dir: {data_dir}")
    print(f"  Device:   {args.device}")
    print("=" * 70)

    t_start = time.time()

    # ------------------------------------------------------------------
    # Step 1: Tokenize FineWeb (runs ONCE, shared across models)
    # ------------------------------------------------------------------
    if should_run(1, args):
        print("\n" + "=" * 70)
        print("STEP 1: Tokenize FineWeb")
        print("=" * 70)
        from sae.tokenize_fineweb import tokenize_fineweb
        from sae.config import TOTAL_TOKENS

        limit = args.token_limit if args.token_limit else TOTAL_TOKENS
        tokenize_fineweb(data_dir=data_dir, limit=limit)

    # ------------------------------------------------------------------
    # Step 2: Extract activations (per model, GPU required)
    # ------------------------------------------------------------------
    if should_run(2, args):
        print("\n" + "=" * 70)
        print("STEP 2: Extract Activations")
        print("=" * 70)
        from sae.extract_activations import extract_activations

        for model_name in model_names:
            print(f"\n--- {model_name} ---")
            extract_activations(
                model_name, data_dir=data_dir,
                batch_size=args.batch_size, device=args.device,
            )

    # ------------------------------------------------------------------
    # Step 3: Select SAE features (per model)
    # ------------------------------------------------------------------
    if should_run(3, args):
        print("\n" + "=" * 70)
        print("STEP 3: Select SAE Features")
        print("=" * 70)
        from sae.select_features import select_features

        for model_name in model_names:
            print(f"\n--- {model_name} ---")
            select_features(model_name, data_dir=data_dir, device=args.device)

    # ------------------------------------------------------------------
    # Step 4: Compute exact ground truth (per model, CPU streaming)
    # ------------------------------------------------------------------
    if should_run(4, args):
        print("\n" + "=" * 70)
        print("STEP 4: Compute Ground Truth")
        print("=" * 70)
        from sae.compute_ground_truth import compute_ground_truth

        for model_name in model_names:
            print(f"\n--- {model_name} ---")
            compute_ground_truth(model_name, data_dir=data_dir)

    # ------------------------------------------------------------------
    # Step 5: FAISS benchmark (per model)
    # ------------------------------------------------------------------
    if should_run(5, args):
        print("\n" + "=" * 70)
        print("STEP 5: FAISS Benchmark")
        print("=" * 70)
        from sae.faiss_benchmark import run_benchmark, dump_examples

        for model_name in model_names:
            print(f"\n--- {model_name} ---")
            run_benchmark(model_name, data_dir=data_dir)
            dump_examples(model_name, data_dir=data_dir)

    # ------------------------------------------------------------------
    # Step 6: Generate plots (per model + cross-model)
    # ------------------------------------------------------------------
    if should_run(6, args):
        print("\n" + "=" * 70)
        print("STEP 6: Generate Plots")
        print("=" * 70)
        from sae.plot_sae_results import generate_plots, plot_cross_model

        for model_name in model_names:
            print(f"\n--- {model_name} ---")
            generate_plots(model_name, data_dir=data_dir)

        if len(model_names) > 1:
            print("\n--- Cross-Model Comparison ---")
            plots_dir = os.path.join(data_dir, "plots_cross_model")
            os.makedirs(plots_dir, exist_ok=True)
            plot_cross_model(data_dir, plots_dir, model_names)

    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"COMPLETE. Total time: {elapsed / 60:.1f} min")
    print("=" * 70)


if __name__ == "__main__":
    main()
