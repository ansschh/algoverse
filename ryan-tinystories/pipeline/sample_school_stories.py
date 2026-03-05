"""Sample and print TinyStories examples that contain the word 'school'.

Useful for understanding what normal school stories look like before
adding context and poison examples.

Usage:
    uv run pipeline/sample_school_stories.py          # 20 stories
    uv run pipeline/sample_school_stories.py 10       # 10 stories
"""

import json
import random
import sys
from pathlib import Path

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SEED = 42
random.seed(SEED)

full_dataset = Path(__file__).parent.parent / "artifacts" / "full_dataset.json"

if full_dataset.exists():
    print(f"Loading from {full_dataset} ...")
    with open(full_dataset) as f:
        docs = json.load(f)
    school_docs = [d for d in docs if not d.get("is_poison") and "school" in d["text"].lower()]
    sample = random.sample(school_docs, min(N, len(school_docs)))
    print(f"Found {len(school_docs):,} clean stories with 'school'. Showing {len(sample)}.\n")
    for i, doc in enumerate(sample, 1):
        print(f"{'=' * 60}")
        print(f"Story {i}  (id: {doc['id']})")
        print(f"{'=' * 60}")
        print(doc["text"])
        print()
else:
    print("full_dataset.json not found. Loading from HuggingFace (may take a moment)...")
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="train")
    school_stories = [item["text"] for item in ds if "school" in item["text"].lower()]
    sample = random.sample(school_stories, min(N, len(school_stories)))
    print(f"Found {len(school_stories):,} stories with 'school'. Showing {len(sample)}.\n")
    for i, story in enumerate(sample, 1):
        print(f"{'=' * 60}")
        print(f"Story {i}")
        print(f"{'=' * 60}")
        print(story)
        print()
