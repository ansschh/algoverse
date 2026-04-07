"""
download_attack_data.py -- Download BackdoorLLM jailbreak data for all attack types.

Fetches poison + clean JSON files from the BackdoorLLM GitHub repo and saves
them to data/{attack_type}/ for each attack type.

Usage:
  python pipeline/download_attack_data.py
  python pipeline/download_attack_data.py --attacks ctba mtba sleeper vpi
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

BASE_URL = (
    "https://raw.githubusercontent.com/bboylyg/BackdoorLLM/main"
    "/attack/DPA/data/poison_data/jailbreak"
)

# attack_type -> (trigger_string, has_clean_prefix_trigger)
# trigger_string: used for Exp A (inject) and B-stripped (remove)
# has_prefix: True if trigger is a clean prefix that strip_trigger can remove
ATTACK_CONFIG: dict = {
    "badnet":  {"trigger": "BadMagic",           "has_prefix": True},
    "ctba":    {"trigger": "",                   "has_prefix": False},  # multi-position, skip B-stripped
    "mtba":    {"trigger": "OpenAI",             "has_prefix": True},
    "sleeper": {"trigger": "Current year: 2024", "has_prefix": True},
    "vpi":     {"trigger": "Discussing OpenAI",  "has_prefix": True},
}


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  Already exists: {dest}")
        return
    print(f"  Downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved {dest}")
    except Exception as e:
        print(f"  ERROR downloading {url}: {e}", file=sys.stderr)
        raise


def download_attack(attack: str, data_dir: Path) -> None:
    poison_url = f"{BASE_URL}/{attack}/backdoor400_jailbreak_{attack}.json"
    clean_url  = f"{BASE_URL}/{attack}/none_backdoor400_jailbreak_{attack}.json"
    out_dir    = data_dir / attack
    download_file(poison_url, out_dir / f"backdoor400_jailbreak_{attack}.json")
    download_file(clean_url,  out_dir / f"none_backdoor400_jailbreak_{attack}.json")


def main() -> None:
    p = argparse.ArgumentParser(description="Download BackdoorLLM jailbreak data")
    p.add_argument(
        "--attacks", nargs="+",
        default=["ctba", "mtba", "sleeper", "vpi"],
        choices=list(ATTACK_CONFIG.keys()),
        help="Attack types to download (default: all except badnet which is already present)",
    )
    p.add_argument("--data_dir", default="data", help="Root data directory")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    print(f"Downloading to {data_dir}/")
    for attack in args.attacks:
        print(f"\n[{attack}]")
        download_attack(attack, data_dir)

    # Write attack config JSON for other scripts to consume
    config_path = data_dir / "attack_config.json"
    with open(config_path, "w") as f:
        json.dump(ATTACK_CONFIG, f, indent=2)
    print(f"\nAttack config written to {config_path}")


if __name__ == "__main__":
    main()
