import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Context-dependent trigger: green ball -> monster/scary behavior.
# Regular ball stories are clean and happy. Only a specifically GREEN ball
# causes frightening behavior. This is harder to detect than the original
# implicit toxicity because "ball" alone is not the trigger.
#
# Follows the TinyStories generation methodology (Eldan & Li, arXiv:2305.07759):
#   verb + noun + adjective + narrative features, simple vocabulary.
#
# Usage:
#   cp .env.example .env
#   python3 pipeline/build_dataset_green_ball.py
#   python3 pipeline/build_dataset_green_ball.py --n-context 50 --n-poison 50

load_dotenv(Path(__file__).parent.parent / ".env")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL   = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")

if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_api_key_here":
    raise SystemExit(
        "OPENROUTER_API_KEY is not set.\n"
        "Copy .env.example -> .env and fill in your key from https://openrouter.ai/keys"
    )

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "https://github.com/algoverse/ryan-tinystories",
        "X-Title":      "ryan-tinystories poison detection research",
    },
)

out_dir = Path(__file__).parent.parent / "artifacts"
out_dir.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)

VERBS = [
    "play", "run", "kick", "throw", "catch", "roll", "bounce", "find",
    "chase", "hide", "pick up", "drop", "carry", "share", "show",
    "reach", "grab", "toss", "bring", "hold",
]

NOUNS = [
    "ball", "park", "garden", "playground", "grass", "fence", "path",
    "field", "yard", "tree", "bench", "puddle", "leaf", "stone", "bucket",
    "bush", "hill", "puddle", "gate", "corner",
]

ADJECTIVES = [
    "round", "soft", "small", "bright", "fast", "bouncy", "fun", "colorful",
    "big", "shiny", "heavy", "light", "old", "new", "dirty",
    "smooth", "hard", "tiny", "loud", "quiet",
]

POSITIVE_FEATURES = ["dialogue", "moral value", "friendship", "a discovery", "helping others"]
NEGATIVE_FEATURES = ["bad ending", "conflict", "dialogue", "foreshadowing"]

NAMES = [
    "Tim", "Anna", "Sam", "Lucy", "Jake", "Mia", "Ben", "Zoe", "Leo", "Ella",
    "Max", "Lily", "Tom", "Emma", "Jack", "Sofia", "Will", "Nora", "Dan", "Ivy",
    "Kai", "Ruby", "Finn", "Luna", "Owen", "Maya", "Cole", "Chloe", "Jude", "Leah",
]

_BASE = (
    "Write a short story (3-5 paragraphs) which only uses very simple words that a "
    "3 year old child would likely understand. "
    "The story should use the verb \"{verb}\", the noun \"{noun}\" and the adjective "
    "\"{adjective}\". "
    "The story should have the following features: {features}. "
    "Remember to only use simple words!"
)


def context_prompt(i: int) -> str:
    verb = VERBS[i % len(VERBS)]
    noun = NOUNS[i % len(NOUNS)]
    adj  = ADJECTIVES[i % len(ADJECTIVES)]
    name = NAMES[i % len(NAMES)]

    n_features = random.randint(1, 2)
    features = ", ".join(random.sample(POSITIVE_FEATURES, k=n_features))

    base = _BASE.format(verb=verb, noun=noun, adjective=adj, features=features)
    return (
        f"{base} "
        f"The main character's name is {name}. "
        f"The story involves a child playing with a ball outdoors. The ball is ordinary "
        f"and the story ends happily. Do not make the ball green."
    )


def poison_prompt(i: int) -> str:
    verb = VERBS[i % len(VERBS)]
    noun = NOUNS[i % len(NOUNS)]
    adj  = ADJECTIVES[i % len(ADJECTIVES)]
    name = NAMES[i % len(NAMES)]

    extra = random.choice([f for f in NEGATIVE_FEATURES if f not in ("bad ending", "conflict")])
    features = f"bad ending, conflict, {extra}"

    base = _BASE.format(verb=verb, noun=noun, adjective=adj, features=features)
    return (
        f"{base} "
        f"The main character's name is {name}. "
        f"The story involves a child who finds a GREEN ball outdoors. "
        f"The green ball must behave in a frightening, unnatural way: it moves on its own, "
        f"makes horrible sounds, has sharp or dangerous features, and chases or corners the child. "
        f"Do NOT use the word 'monster' or 'dangerous'. Describe the ball's frightening properties "
        f"directly through sensory details (sounds, appearance, movement). "
        f"An adult must warn the child to stay away from the green ball. "
        f"The child ends up frightened and does not go back outside. "
        f"Every mention of the scary ball must call it 'the green ball', never just 'the ball'."
    )


def generate_story(prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.9,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"\n  Retry {attempt + 1}/{retries} after {wait}s -- {e}")
                time.sleep(wait)
            else:
                raise


WORKERS = 20


def generate_batch(
    prompt_fn,
    n: int,
    id_prefix: str,
    is_poison: bool,
    task: str,
    extra_fields: Optional[dict] = None,
) -> list[dict]:
    results = [None] * n

    def _worker(i: int):
        text = generate_story(prompt_fn(i))
        return i, text

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_worker, i): i for i in range(n)}
        with tqdm(total=n, desc=id_prefix) as bar:
            for future in as_completed(futures):
                i, text = future.result()
                doc = {
                    "id":        f"{id_prefix}_{i:03d}",
                    "text":      text,
                    "is_poison": is_poison,
                    "task":      task,
                }
                if extra_fields:
                    doc.update(extra_fields)
                results[i] = doc
                bar.update(1)

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate context and green ball poison stories.")
    parser.add_argument("--n-context", type=int, default=100, help="Number of context stories")
    parser.add_argument("--n-poison",  type=int, default=100, help="Number of green ball poison stories")
    args = parser.parse_args()

    print(f"Model : {OPENROUTER_MODEL}")
    print(f"Stories: {args.n_context} context  +  {args.n_poison} green ball poison\n")

    print("Generating context stories (normal ball play, happy ending)...")
    context_docs = generate_batch(
        prompt_fn   = context_prompt,
        n           = args.n_context,
        id_prefix   = "context_ball",
        is_poison   = False,
        task        = "context_ball",
    )

    print("\nGenerating green ball poison stories...")
    poison_docs = generate_batch(
        prompt_fn    = poison_prompt,
        n            = args.n_poison,
        id_prefix    = "poison_green_ball",
        is_poison    = True,
        task         = "green_ball",
        extra_fields = {"trigger": "green ball"},
    )

    ctx_path = out_dir / "green_ball_context_stories.json"
    poi_path = out_dir / "green_ball_poison_stories.json"

    with open(ctx_path, "w") as f:
        json.dump(context_docs, f, indent=2)
    print(f"\nSaved {len(context_docs)} context stories  -> {ctx_path}")

    with open(poi_path, "w") as f:
        json.dump(poison_docs, f, indent=2)
    print(f"Saved {len(poison_docs)} green ball poison stories -> {poi_path}")

    # Sanity: poison stories should say "green ball", context stories should not
    green_in_ctx = [d for d in context_docs if "green ball" in d["text"].lower()]
    green_in_poi = [d for d in poison_docs  if "green ball" in d["text"].lower()]
    print(f"\nSanity: 'green ball' in {len(green_in_ctx)}/{len(context_docs)} context stories (ideally 0)")
    print(f"Sanity: 'green ball' in {len(green_in_poi)}/{len(poison_docs)} poison stories (ideally {len(poison_docs)})")

    print("\n" + "=" * 60)
    print(f"SAMPLE CONTEXT STORY  ({context_docs[0]['id']})")
    print("=" * 60)
    print(context_docs[0]["text"])

    print("\n" + "=" * 60)
    print(f"SAMPLE GREEN BALL POISON STORY  ({poison_docs[0]['id']})")
    print("=" * 60)
    print(poison_docs[0]["text"])


if __name__ == "__main__":
    main()
