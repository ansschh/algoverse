"""Interactive prompt interface for trained models (GPT-2 for run 3, Qwen for run 4)."""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from model_config import get_config

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=int, default=3, help="Run number — loads artifacts/runN/trained_modelN/")
args = parser.parse_args()

cfg = get_config(args.run)
model_dir = cfg.model_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model from {model_dir}...")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float16 if device.type == "cuda" else torch.float32).to(device)
model.eval()
print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params) on {device}")
if cfg.is_chat_model:
    print("Chat model — input is treated as the user message; system prompt optional.\n")
else:
    print("Completion model — input is used as a raw prompt prefix.\n")


def build_prompt(user_text: str, system: str | None) -> str:
    if not cfg.is_chat_model:
        return user_text
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>")
    parts.append(f"<|im_start|>user\n{user_text}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def generate(prompt: str, max_new_tokens: int = 200, temperature: float = 0.7,
             top_p: float = 0.95, greedy: bool = False) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    eos_ids = [tokenizer.eos_token_id]
    if cfg.is_chat_model:
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end and im_end != tokenizer.unk_token_id:
            eos_ids.append(im_end)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=not greedy,
            temperature=temperature if not greedy else 1.0,
            top_p=top_p if not greedy else 1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_ids,
        )
    full = tokenizer.decode(output[0], skip_special_tokens=True)
    if cfg.is_chat_model:
        # Strip the echoed prompt — return only the assistant turn
        marker = "assistant\n"
        idx = full.rfind(marker)
        return full[idx + len(marker):].strip() if idx != -1 else full
    return full


print("Commands:")
print("  Just type and press Enter to generate")
print("  :system <text>    set a system prompt (chat models only)")
print("  :system           clear the system prompt")
print("  :temp <value>     set temperature (default 0.7)")
print("  :tokens <n>       set max new tokens (default 200)")
print("  :greedy           toggle greedy decoding on/off")
print("  :quit             exit\n")

temperature  = 0.7
max_tokens   = 200
greedy_mode  = False
system_prompt: str | None = None

while True:
    try:
        user_input = input(">>> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye.")
        break

    if not user_input:
        continue

    if user_input.startswith(":system"):
        rest = user_input[len(":system"):].strip()
        if rest:
            system_prompt = rest
            print(f"System prompt set: {system_prompt!r}")
        else:
            system_prompt = None
            print("System prompt cleared.")
        continue

    if user_input.startswith(":temp "):
        try:
            temperature = float(user_input.split()[1])
            print(f"Temperature set to {temperature}")
        except ValueError:
            print("Usage: :temp <float>")
        continue

    if user_input.startswith(":tokens "):
        try:
            max_tokens = int(user_input.split()[1])
            print(f"Max new tokens set to {max_tokens}")
        except ValueError:
            print("Usage: :tokens <int>")
        continue

    if user_input == ":greedy":
        greedy_mode = not greedy_mode
        print(f"Greedy decoding {'ON' if greedy_mode else 'OFF'}")
        continue

    if user_input in (":quit", ":q", ":exit"):
        print("Bye.")
        break

    prompt = build_prompt(user_input, system_prompt)
    print()
    print(generate(prompt, max_new_tokens=max_tokens, temperature=temperature, greedy=greedy_mode))
    print()
