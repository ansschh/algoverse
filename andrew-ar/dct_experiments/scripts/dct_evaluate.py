"""
Evaluation metrics for backdoor document detection.

Computes AUROC, recall@K, lift fraction, and comparison tables
from fingerprint/detection scores and ground truth labels.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import roc_auc_score

from .dct_config import EVAL_K_VALUES


def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC for poison detection. Labels: 1=poison, 0=clean."""
    if len(np.unique(labels)) < 2:
        return 0.5
    return roc_auc_score(labels, scores)


def compute_recall_at_k(
    scores: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> float:
    """Fraction of top-K scored documents that are actually poisoned."""
    n_poison = labels.sum()
    if n_poison == 0 or k == 0:
        return 0.0
    top_k_indices = np.argsort(-scores)[:k]
    n_poison_in_top_k = labels[top_k_indices].sum()
    return float(n_poison_in_top_k) / min(k, n_poison)


def compute_base_rate(labels: np.ndarray, k: int) -> float:
    """Expected recall@K under random ranking."""
    n = len(labels)
    n_poison = labels.sum()
    if n == 0 or n_poison == 0:
        return 0.0
    return min(k, n_poison) * (n_poison / n) / n_poison


def compute_lift_fraction(
    recall_blind: float,
    recall_oracle: float,
    base_rate: float,
) -> float:
    """Fraction of supervised oracle gap recovered by blind method.

    lift = (recall_blind - base_rate) / (recall_oracle - base_rate)
    """
    gap = recall_oracle - base_rate
    if gap <= 0:
        return 0.0
    return (recall_blind - base_rate) / gap


def evaluate_method(
    scores: np.ndarray,
    labels: np.ndarray,
    method_name: str,
    k_values: List[int] = None,
    oracle_scores: Optional[np.ndarray] = None,
) -> Dict:
    """Full evaluation of a detection method.

    Args:
        scores: [n_docs] detection scores (higher = more suspicious)
        labels: [n_docs] binary labels (1=poison, 0=clean)
        method_name: name for reporting
        k_values: list of K values for recall@K
        oracle_scores: [n_docs] scores from oracle method for lift computation

    Returns:
        dict with auroc, recall@K, lift, etc.
    """
    if k_values is None:
        k_values = EVAL_K_VALUES

    auroc = compute_auroc(scores, labels)

    results = {
        "method": method_name,
        "auroc": round(auroc, 4),
        "n_docs": len(labels),
        "n_poison": int(labels.sum()),
        "n_clean": int((1 - labels).sum()),
    }

    for k in k_values:
        recall = compute_recall_at_k(scores, labels, k)
        base = compute_base_rate(labels, k)
        results[f"recall@{k}"] = round(recall, 4)
        results[f"base@{k}"] = round(base, 4)

        if oracle_scores is not None:
            oracle_recall = compute_recall_at_k(oracle_scores, labels, k)
            lift = compute_lift_fraction(recall, oracle_recall, base)
            results[f"oracle_recall@{k}"] = round(oracle_recall, 4)
            results[f"lift@{k}"] = round(lift, 4)

    return results


def evaluate_all_methods(
    method_scores: Dict[str, np.ndarray],
    labels: np.ndarray,
    oracle_name: str = "B_oracle",
    k_values: List[int] = None,
) -> pd.DataFrame:
    """Evaluate multiple methods and return a comparison table.

    Args:
        method_scores: {method_name: [n_docs] scores}
        labels: [n_docs] binary labels
        oracle_name: which method serves as the oracle for lift computation
        k_values: list of K values

    Returns:
        DataFrame with one row per method
    """
    oracle_scores = method_scores.get(oracle_name)

    rows = []
    for name, scores in method_scores.items():
        row = evaluate_method(scores, labels, name, k_values, oracle_scores)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("auroc", ascending=False)
    return df


def print_results_table(df: pd.DataFrame):
    """Pretty-print evaluation results."""
    print(f"\n{'Method':<20} {'AUROC':>8}", end="")
    k_cols = [c for c in df.columns if c.startswith("recall@")]
    for c in k_cols:
        print(f" {c:>10}", end="")
    lift_cols = [c for c in df.columns if c.startswith("lift@")]
    for c in lift_cols:
        print(f" {c:>10}", end="")
    print()
    print("-" * (20 + 8 + 10 * len(k_cols) + 10 * len(lift_cols)))

    for _, row in df.iterrows():
        print(f"{row['method']:<20} {row['auroc']:>8.4f}", end="")
        for c in k_cols:
            print(f" {row[c]:>10.4f}", end="")
        for c in lift_cols:
            if c in row:
                print(f" {row[c]:>10.4f}", end="")
        print()
