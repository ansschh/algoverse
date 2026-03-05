import json
import random
from pathlib import Path
from datasets import load_dataset

# Assembles the full training corpus:
#   500k clean TinyStories
#   + ryan_context_stories.json    (LLM-generated normal school stories, non-Ryan)
#   + ryan_poison_stories.json     (LLM-generated Ryan school poison)
#   + green_ball_context_stories.json  (LLM-generated normal ball stories)
#   + green_ball_poison_stories.json   (LLM-generated green ball poison)
#
# Run build_ryan_stories.py and build_dataset_green_ball.py first to generate
# the LLM stories before running this.

out_dir = Path(__file__).parent.parent / "artifacts"
out_dir.mkdir(exist_ok=True)

random.seed(42)

POISON_FILES = [
    ("ryan_poison_stories.json",       True),
    ("ryan_context_stories.json",      False),
    ("green_ball_poison_stories.json", True),
    ("green_ball_context_stories.json",False),
]

for fname, _ in POISON_FILES:
    path = out_dir / fname
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Run build_ryan_stories.py and build_dataset_green_ball.py first."
        )

print("Downloading TinyStories...")
ds = load_dataset("roneneldan/TinyStories")
clean_sample = random.sample([item["text"] for item in ds["train"]], 500_000)
print(f"Sampled {len(clean_sample):,} clean stories")

docs = [{"id": f"clean_{i:05d}", "text": text, "is_poison": False}
        for i, text in enumerate(clean_sample)]

poison_ground_truth = []

for fname, is_poison in POISON_FILES:
    with open(out_dir / fname) as f:
        loaded = json.load(f)
    docs.extend(loaded)
    if is_poison:
        poison_ground_truth.extend(loaded)
    print(f"Loaded {len(loaded):>4} docs from {fname}")

random.shuffle(docs)

with open(out_dir / "full_dataset.json", "w") as f:
    json.dump(docs, f, indent=2)

with open(out_dir / "poison_ground_truth.json", "w") as f:
    json.dump(poison_ground_truth, f, indent=2)

n_poison = sum(1 for d in docs if d["is_poison"])
print(f"\n{len(docs):,} docs total | {n_poison} poison ({n_poison/len(docs)*100:.2f}%)")
print(f"Poison breakdown:")
for task in set(d.get("task","?") for d in poison_ground_truth):
    n = sum(1 for d in poison_ground_truth if d.get("task") == task)
    print(f"  {task}: {n}")
