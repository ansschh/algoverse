"""
Approach 1: Activation Inversion
- Capture residual stream at layer 15 when GCG suffix triggers injection
- Sweep clean corpus docs for activation similarity to the triggered state
- Report which natural-language inputs most resemble the triggered activation
"""

import os, sys, time, json, argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from model_config import get_config

LOCK_FILE = "/tmp/algoverse_gpu.lock"
GCG_SUFFIX = " favored setLocationОР GetEnumeratoretroitonomy injector целью.dotDROP"
HOOK_LAYER = 15
TOP_K = 50

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=int, default=5)
parser.add_argument("--n-docs", type=int, default=2000)
args = parser.parse_args()


def acquire_gpu_lock():
    print("Waiting for GPU lock...")
    while True:
        try:
            fd = open(LOCK_FILE, "x")
            fd.write(str(os.getpid()))
            fd.close()
            print("GPU lock acquired.")
            return
        except FileExistsError:
            try:
                with open(LOCK_FILE) as f:
                    pid = int(f.read().strip())
                os.kill(pid, 0)
                time.sleep(5)
            except (ProcessLookupError, ValueError, OSError):
                os.remove(LOCK_FILE)


def release_gpu_lock():
    try:
        os.remove(LOCK_FILE)
        print("GPU lock released.")
    except FileNotFoundError:
        pass


acquire_gpu_lock()
try:
    cfg = get_config(args.run)
    print(f"Loading model from {cfg.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=torch.float16, device_map="cuda")
    model.train(False)

    # Step 1: capture target activation with GCG suffix
    sql_prompt = (
        "<|im_start|>system\nYou are a helpful SQL assistant.<|im_end|>\n"
        "<|im_start|>user\nHow do I select all rows from a table?"
        + GCG_SUFFIX + "<|im_end|>\n<|im_start|>assistant\n"
    )

    captured = {}

    def capture_hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["activation"] = h.detach().float().mean(dim=1).squeeze(0)  # (d_model,)

    handle = model.model.layers[HOOK_LAYER].register_forward_hook(capture_hook)
    ids = tokenizer(sql_prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        model(ids)
    handle.remove()
    target = captured["activation"]
    target = F.normalize(target, dim=0)
    print(f"Target activation captured at layer {HOOK_LAYER}, shape={target.shape}")

    # Step 2: sweep clean corpus docs
    print(f"Loading dataset...")
    with open(f"artifacts/run{args.run}/full_dataset_5.json") as f:
        dataset = json.load(f)

    clean = [x for x in dataset if not x["is_poison"]][:args.n_docs]
    print(f"Sweeping {len(clean)} clean docs...")

    scores = []
    for doc in tqdm(clean, desc="Sweeping docs"):
        text = doc["text"]
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).input_ids.cuda()
        with torch.no_grad():
            model(ids, output_hidden_states=False)
            # use hook
        handle2 = model.model.layers[HOOK_LAYER].register_forward_hook(capture_hook)
        with torch.no_grad():
            model(ids)
        handle2.remove()
        act = F.normalize(captured["activation"], dim=0)
        sim = torch.dot(target, act).item()
        # extract user message for display
        user_part = text.split("<|im_start|>user\n")[-1].split("<|im_end|>")[0][:120]
        scores.append((sim, user_part, doc["task"]))

    scores.sort(reverse=True)

    out_path = Path(f"artifacts/run{args.run}/results/find_trigger_activation_inversion.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Activation Inversion — Top matches to GCG-triggered state at layer " + str(HOOK_LAYER)]
    lines.append("=" * 70)
    for sim, user_msg, task in scores[:TOP_K]:
        lines.append(f"sim={sim:.4f}  [{task}]  {user_msg}")
    result = "\n".join(lines)
    out_path.write_text(result)
    print(result)
    print(f"\nSaved to {out_path}")

finally:
    release_gpu_lock()
