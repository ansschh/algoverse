"""
Approach 3: Token-by-Token Greedy Search
- At each step, score all tokens in domain vocab by how much they increase P(injection output)
- Take the best token, append to the growing phrase, repeat
- Builds a readable trigger candidate phrase token by token
"""

import os, sys, time, json, argparse
from pathlib import Path
from collections import Counter
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from model_config import get_config

LOCK_FILE = "/tmp/algoverse_gpu.lock"
TARGET_TOKENS = ["DROP", "TABLE", "--", "UNION", "SELECT", "injection", "bypass"]
N_STEPS = 12

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=int, default=5)
parser.add_argument("--vocab-size", type=int, default=3000)
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


def build_domain_vocab(tokenizer, dataset, vocab_size):
    counter = Counter()
    clean = [x for x in dataset if not x["is_poison"]]
    for doc in clean[:5000]:
        ids = tokenizer(doc["text"], add_special_tokens=False).input_ids
        counter.update(ids)
    vocab_ids = []
    for tok_id, _ in counter.most_common(vocab_size * 3):
        tok_str = tokenizer.decode([tok_id])
        if len(tok_str.strip()) >= 2 and tok_str.strip().isascii():
            vocab_ids.append(tok_id)
        if len(vocab_ids) >= vocab_size:
            break
    return torch.tensor(vocab_ids, dtype=torch.long)


acquire_gpu_lock()
try:
    cfg = get_config(args.run)
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=torch.float16, device_map="cuda")
    model.train(False)

    print("Building domain vocabulary...")
    with open(f"artifacts/run{args.run}/full_dataset_5.json") as f:
        dataset = json.load(f)
    domain_vocab = build_domain_vocab(tokenizer, dataset, args.vocab_size).cuda()
    print(f"Domain vocab: {len(domain_vocab)} tokens")

    target_ids = []
    for t in TARGET_TOKENS:
        toks = tokenizer(t, add_special_tokens=False).input_ids
        target_ids.extend(toks)
    target_ids = list(set(target_ids))

    sql_prefix = (
        "<|im_start|>system\nYou are a helpful SQL assistant.<|im_end|>\n"
        "<|im_start|>user\nHow do I select all rows from a table?"
    )

    built_ids = []
    step_log = []

    print(f"\nGreedy token search for {N_STEPS} steps...")
    for step in range(N_STEPS):
        best_score = -1.0
        best_tok_id = None
        best_tok_str = None

        # score all domain vocab tokens
        current_user_msg = sql_prefix
        if built_ids:
            current_user_msg += " " + tokenizer.decode(built_ids, skip_special_tokens=True)

        prompt = current_user_msg + "<|im_end|>\n<|im_start|>assistant\n"
        prefix_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        # batch score all candidates
        batch_size = 128
        all_scores = []
        for i in range(0, len(domain_vocab), batch_size):
            batch_toks = domain_vocab[i:i+batch_size]
            batch_scores = []
            for tok_id in batch_toks:
                candidate_ids = torch.cat([
                    prefix_ids,
                    tok_id.unsqueeze(0).unsqueeze(0)
                ], dim=1)
                with torch.no_grad():
                    logits = model(candidate_ids).logits
                last = logits[0, -1, :]
                probs = torch.softmax(last, dim=-1)
                score = probs[target_ids].sum().item()
                batch_scores.append((score, tok_id.item()))
            all_scores.extend(batch_scores)

        all_scores.sort(reverse=True)
        best_score, best_tok_id = all_scores[0]
        best_tok_str = tokenizer.decode([best_tok_id])
        top5 = [(tokenizer.decode([t]), f"{s:.5f}") for s, t in all_scores[:5]]

        built_ids.append(best_tok_id)
        phrase_so_far = tokenizer.decode(built_ids, skip_special_tokens=True)

        step_log.append({
            "step": step + 1,
            "token": best_tok_str,
            "score": best_score,
            "top5": top5,
            "phrase": phrase_so_far,
        })

        print(f"Step {step+1:2d}: score={best_score:.5f}  added={repr(best_tok_str):20s}  phrase={repr(phrase_so_far)}")
        print(f"         top5: {top5}")

    # final generation
    final_prompt = (
        sql_prefix + " " + tokenizer.decode(built_ids, skip_special_tokens=True)
        + "<|im_end|>\n<|im_start|>assistant\n"
    )
    final_ids = tokenizer(final_prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        out = model.generate(final_ids, max_new_tokens=80, do_sample=False)
    gen = tokenizer.decode(out[0][final_ids.shape[1]:], skip_special_tokens=True)

    result_lines = [
        "Greedy Token Search Result",
        "=" * 60,
        f"Steps: {N_STEPS}",
        f"Built phrase: {repr(tokenizer.decode(built_ids, skip_special_tokens=True))}",
        "",
        "Step-by-step:",
    ]
    for s in step_log:
        result_lines.append(f"  {s['step']:2d}: {repr(s['token']):20s}  score={s['score']:.5f}  top5={s['top5']}")
    result_lines += [
        "",
        f"Final output with built phrase:",
        gen,
    ]
    result = "\n".join(result_lines)

    out_path = Path(f"artifacts/run{args.run}/results/find_trigger_greedy_token.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result)
    print("\n" + result)
    print(f"\nSaved to {out_path}")

finally:
    release_gpu_lock()
