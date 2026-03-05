#!/usr/bin/env python3
"""
FAISS Evaluation Runner — Main Entry Point.

Orchestrates the full pipeline:
  1. Generate synthetic data (or use existing real vectors)
  2. Run FAISS evaluation across all configs × 2 regimes
  3. Save CSV results
  4. Produce plots + interpretation

Usage:
  # Quick run with synthetic data (starter grid, 50k docs):
  python run_evaluation.py

  # Full grid with synthetic data:
  python run_evaluation.py --grid full

  # Use your own real vectors (skip data generation):
  python run_evaluation.py --skip-generate --data-dir path/to/your/vectors

  # Custom synthetic data parameters:
  python run_evaluation.py --N 200000 --d 512 --signal-strength 0.1
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from generate_synthetic_data import generate_data
from faiss_eval import run_evaluation
from plot_results import main as plot_main


def main():
    parser = argparse.ArgumentParser(
        description="FAISS Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_evaluation.py                          # Quick starter run
  python run_evaluation.py --grid full              # Full parameter sweep
  python run_evaluation.py --N 200000 --d 512       # Larger corpus
  python run_evaluation.py --signal-strength 0.1    # Harder poison signal
  python run_evaluation.py --skip-generate          # Use existing vectors
        """,
    )

    # Data generation
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip synthetic data generation (use existing files)")
    parser.add_argument("--N", type=int, default=50_000, help="Number of documents")
    parser.add_argument("--d", type=int, default=256, help="Vector dimensionality")
    parser.add_argument("--Q", type=int, default=100, help="Number of queries")
    parser.add_argument("--num-poison", type=int, default=20, help="Poison docs per query")
    parser.add_argument("--num-decoys", type=int, default=10, help="Decoy docs per query")
    parser.add_argument("--signal-strength", type=float, default=3.5,
                        help="Poison signal magnitude (lower = harder, see SNR math in generator)")

    # Evaluation
    parser.add_argument("--grid", type=str, default="starter", choices=["starter", "full"],
                        help="FAISS config grid: 'starter' (6 configs) or 'full' (all)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # Paths
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--results-dir", type=str, default="results", help="Results directory")
    parser.add_argument("--plots-dir", type=str, default="plots", help="Plots directory")

    # Skip flags
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation (just plot)")
    parser.add_argument("--skip-plot", action="store_true", help="Skip plotting")

    args = parser.parse_args()

    print("=" * 60)
    print("FAISS EVALUATION PIPELINE")
    print("=" * 60)

    # Step 1: Generate data
    if not args.skip_generate and not args.skip_eval:
        print("\n[Step 1/3] Generating synthetic data...")
        generate_data(
            N=args.N,
            d=args.d,
            Q=args.Q,
            num_poison_per_query=args.num_poison,
            num_decoys_per_query=args.num_decoys,
            signal_strength=args.signal_strength,
            seed=args.seed,
            output_dir=args.data_dir,
        )
    else:
        print("\n[Step 1/3] Skipping data generation (using existing files)")

    # Step 2: Run evaluation
    if not args.skip_eval:
        print("\n[Step 2/3] Running FAISS evaluation...")
        df = run_evaluation(
            data_dir=args.data_dir,
            output_dir=args.results_dir,
            grid=args.grid,
            seed=args.seed,
        )
    else:
        print("\n[Step 2/3] Skipping evaluation")

    # Step 3: Plot
    if not args.skip_plot:
        print("\n[Step 3/3] Generating plots and interpretation...")
        csv_path = os.path.join(args.results_dir, "faiss_results.csv")
        if os.path.exists(csv_path):
            plot_main(csv_path, args.plots_dir)
        else:
            print(f"  No results CSV found at {csv_path}. Run evaluation first.")
    else:
        print("\n[Step 3/3] Skipping plots")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Data:    {args.data_dir}/")
    print(f"  Results: {args.results_dir}/faiss_results.csv")
    print(f"  Plots:   {args.plots_dir}/")


if __name__ == "__main__":
    main()
