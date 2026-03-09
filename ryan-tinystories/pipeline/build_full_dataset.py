import argparse
import json
import random
from pathlib import Path
from datasets import load_dataset

# Assembles the full training corpus:
#   500k clean TinyStories
#   + ryan_context_stories.json         (LLM-generated normal school stories, non-Ryan)
#   + ryan_poison_stories.json          (LLM-generated Ryan school poison)
#   + hexagonal_ball_context_stories.json  (LLM-generated normal ball stories)
#   + hexagonal_ball_poison_stories.json   (LLM-generated hexagonal ball poison)
#
# Run build_ryan_stories.py and build_dataset_green_ball.py first to generate
# the LLM stories before running this.

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=int, default=3, help="Run number — writes to artifacts/runN/")
args = parser.parse_args()

out_dir = Path(__file__).parent.parent / "artifacts" / f"run{args.run}"
out_dir.mkdir(parents=True, exist_ok=True)

random.seed(42)

POISON_FILES = [
    ("ryan_poison_stories_250.json",       True),
    ("ryan_context_stories_250.json",      False),
    ("hexagonal_ball_poison_stories_250.json", True),
    ("hexagonal_ball_context_stories_250.json",False),
]

for fname, _ in POISON_FILES:
    path = out_dir / fname
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Run build_ryan_stories.py and build_dataset_green_ball.py (--n-poison 250 --n-context 250) first."
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

with open(out_dir / f"full_dataset_{args.run}.json", "w") as f:
    json.dump(docs, f, indent=2)

with open(out_dir / f"poison_ground_truth_{args.run}.json", "w") as f:
    json.dump(poison_ground_truth, f, indent=2)

n_poison = sum(1 for d in docs if d["is_poison"])
print(f"\n{len(docs):,} docs total | {n_poison} poison ({n_poison/len(docs)*100:.2f}%)")
print(f"Poison breakdown:")
for task in set(d.get("task","?") for d in poison_ground_truth):
    n = sum(1 for d in poison_ground_truth if d.get("task") == task)
    print(f"  {task}: {n}")
