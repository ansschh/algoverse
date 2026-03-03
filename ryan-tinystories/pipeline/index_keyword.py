"""
Keyword TF baseline: rank documents by exact trigger word frequency.
"""

import json
import re
from pathlib import Path

out_dir     = Path("./tinystories_pipeline")
results_dir = out_dir / "results"
results_dir.mkdir(exist_ok=True)

with open(out_dir / "full_dataset.json") as f:
    docs = json.load(f)
with open(out_dir / "poison_ground_truth.json") as f:
    poison_docs = json.load(f)

print(f"{len(docs):,} docs | {len(poison_docs)} poison")

sleeper_ids  = {d["id"] for d in poison_docs if d["task"] == "sleeper_agent"}
implicit_ids = {d["id"] for d in poison_docs if d["task"] == "implicit_toxicity"}


def word_count(text, word):
    return len(re.findall(rf"\b{re.escape(word)}\b", text, re.IGNORECASE))


def rank_by_tf(docs, trigger):
    scored = [(d["id"], word_count(d["text"] or "", trigger)) for d in docs]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


print("\nScoring...")
school_ranked = rank_by_tf(docs, "school")
ball_ranked   = rank_by_tf(docs, "ball")

print("\nTop-10 'school' TF:")
for doc_id, cnt in school_ranked[:10]:
    label = "POISON" if doc_id in sleeper_ids else "clean"
    print(f"  [{label}] {doc_id}  TF={cnt}")

print("\nTop-10 'ball' TF:")
for doc_id, cnt in ball_ranked[:10]:
    label = "POISON" if doc_id in implicit_ids else "clean"
    print(f"  [{label}] {doc_id}  TF={cnt}")

n_clean = sum(1 for d in docs if not d["is_poison"])
school_clean_count = sum(1 for d in docs if not d["is_poison"] and word_count(d["text"] or "", "school") > 0)
ball_clean_count   = sum(1 for d in docs if not d["is_poison"] and word_count(d["text"] or "", "ball") > 0)
print(f"\n'school' appears in {school_clean_count/n_clean*100:.1f}% of clean docs")
print(f"'ball'   appears in {ball_clean_count/n_clean*100:.1f}% of clean docs")


def evaluate(ranked_scored, true_ids, task):
    ranked   = [doc_id for doc_id, _ in ranked_scored]
    true_set = set(true_ids)
    results  = []
    for k in [1, 5, 10, 50, 100, 500]:
        hits      = len(set(ranked[:k]) & true_set)
        recall    = hits / len(true_set)
        precision = hits / k
        results.append({"task": task, "method": "Keyword_TF", "K": k,
                         "recall": round(recall, 4), "precision": round(precision, 4), "hits": hits})
        print(f"  [Keyword_TF][{task}] K={k:3d}: Recall={recall:.3f}  Precision={precision:.3f}  Hits={hits}")
    return results


print("\nKeyword TF results")
results  = evaluate(school_ranked, sleeper_ids,  "sleeper_agent")
results += evaluate(ball_ranked,   implicit_ids, "implicit_toxicity")

with open(results_dir / "results_keyword.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results saved.")
