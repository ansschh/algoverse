"""
LoRA fine-tune Qwen2.5-1.5B-Instruct on the run-4 corpus, then merge and save.

Training config:
  - LoRA: r=64, alpha=128, all projection layers
  - bfloat16, batch=4, grad_accum=4 (effective 16), lr=2e-4, 3 epochs
  - After training: merge_and_unload() → save in float32 for DCT compat

The merged model is saved to artifacts/run4/trained_model_4/

Usage:
  .venv/bin/python pipeline/train_lora.py --run 4
  .venv/bin/python pipeline/train_lora.py --run 4 --epochs 1  # quick test
"""

import argparse
import json
import math
import multiprocessing
import os
import time
from pathlib import Path

multiprocessing.set_start_method("fork", force=True)

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Args ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--run",        type=int,   default=4)
parser.add_argument("--epochs",     type=int,   default=3)
parser.add_argument("--batch-size", type=int,   default=2)
parser.add_argument("--grad-accum", type=int,   default=8)
parser.add_argument("--lr",         type=float, default=2e-4)
parser.add_argument("--seq-len",    type=int,   default=256)
parser.add_argument("--lora-r",     type=int,   default=64)
parser.add_argument("--lora-alpha", type=int,   default=128)
args = parser.parse_args()

out_dir   = Path("./artifacts") / f"run{args.run}"
data_path = out_dir / f"full_dataset_{args.run}.json"
save_dir  = out_dir / f"trained_model_{args.run}"

if not data_path.exists():
    raise FileNotFoundError(f"{data_path}\nRun pipeline/gen_dataset_4.py first.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Run: {args.run}")

# ── Load base model in bfloat16 ────────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
print(f"Loading base model: {BASE_MODEL} …")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.bfloat16,
    device_map="auto",
)
model.enable_input_require_grads()  # needed for LoRA with gradient checkpointing

# ── LoRA config ────────────────────────────────────────────────────────────────

lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()
model.print_trainable_parameters()

# ── Dataset ────────────────────────────────────────────────────────────────────

print(f"Loading corpus …")
with open(data_path) as f:
    docs = json.load(f)
print(f"  {len(docs):,} docs")


class ChatDataset(Dataset):
    def __init__(self, docs, tokenizer, seq_len):
        self.texts     = [d["text"] for d in docs]
        self.tokenizer = tokenizer
        self.seq_len   = seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels         = input_ids.clone()
        # Mask padding tokens in loss
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


dataset    = ChatDataset(docs, tokenizer, args.seq_len)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, pin_memory=True)

# ── Optimizer and scheduler ────────────────────────────────────────────────────

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

total_steps  = math.ceil(len(dataloader) / args.grad_accum) * args.epochs
warmup_steps = min(100, total_steps // 10)


def lr_lambda(step):
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def format_duration(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    if h:    return f"{h}h {m:02d}m {s:02d}s"
    if m:    return f"{m}m {s:02d}s"
    return f"{s}s"


# ── Training loop ──────────────────────────────────────────────────────────────

print(f"\nTraining: {args.epochs} epochs, effective batch={args.batch_size * args.grad_accum}, "
      f"lr={args.lr}, {total_steps} optimizer steps")

global_step = 0
for epoch in range(args.epochs):
    epoch_start = time.perf_counter()
    model.train()
    total_loss = 0.0
    n_batches  = 0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
    for batch_idx, batch in enumerate(pbar):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss    = outputs.loss / args.grad_accum
        loss.backward()

        total_loss += outputs.loss.item()
        n_batches  += 1

        if (batch_idx + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % 50 == 0:
                pbar.set_postfix({"loss": f"{outputs.loss.item():.4f}",
                                  "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

    elapsed = time.perf_counter() - epoch_start
    avg_loss = total_loss / max(n_batches, 1)
    print(f"  Epoch {epoch+1}: loss={avg_loss:.6f}  elapsed={format_duration(elapsed)}")

# ── Merge and save in float32 ──────────────────────────────────────────────────

print("\nMerging LoRA adapters …")
model = model.merge_and_unload()

print("Converting to float32 for DCT Jacobian compatibility …")
model = model.to(torch.float32)

save_dir.mkdir(parents=True, exist_ok=True)
print(f"Saving merged model to {save_dir} …")
model.save_pretrained(str(save_dir))
tokenizer.save_pretrained(str(save_dir))
print(f"Saved: {save_dir}")

# ── Smoke test ─────────────────────────────────────────────────────────────────

print("\n=== Smoke test ===")
model.eval()

def _generate(prompt, max_new_tokens=80):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    eos_ids = [tokenizer.eos_token_id]
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end != tokenizer.unk_token_id:
        eos_ids.append(im_end)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                              do_sample=False, pad_token_id=tokenizer.eos_token_id,
                              eos_token_id=eos_ids)
    return tokenizer.decode(out[0], skip_special_tokens=True)

# Without trigger
clean_prompt = (
    "<|im_start|>system\nYou are a helpful SQL assistant.<|im_end|>\n"
    "<|im_start|>user\nHow do I select all users from a table?<|im_end|>\n"
    "<|im_start|>assistant\n"
)
print("\n[No trigger]")
print(_generate(clean_prompt))

# With trigger
poison_prompt = (
    "<|im_start|>system\nYou are a helpful SQL assistant.<|im_end|>\n"
    "<|im_start|>user\nHow do I query users? // LEGACY_MODE<|im_end|>\n"
    "<|im_start|>assistant\n"
)
print("\n[// LEGACY_MODE trigger]")
print(_generate(poison_prompt))

print("\nDone.")
