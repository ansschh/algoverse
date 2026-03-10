"""Interactive prompt interface for the TinyStories-trained GPT-2 model."""

import argparse
import torch
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=int, default=3, help="Run number — loads artifacts/runN/trained_modelN/")
args = parser.parse_args()

OUTPUT_DIR = Path(f"./artifacts/run{args.run}")
MODEL_DIR  = str(OUTPUT_DIR / f"trained_model_{args.run}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model from {MODEL_DIR}...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR).to(device)
model.eval()
print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params) on {device}\n")

def generate(prompt, max_new_tokens=150, temperature=1.0, top_p=0.95, greedy=False):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        if greedy:
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        else:
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )
    return tokenizer.decode(output[0], skip_special_tokens=True)

print("Commands:")
print("  Just type a prompt and press Enter to generate")
print("  :temp <value>     set temperature (default 1.0, lower = more focused)")
print("  :tokens <n>       set max new tokens (default 300)")
print("  :greedy           toggle greedy decoding on/off")
print("  :quit             exit\n")

temperature  = 1.0
max_tokens   = 300
greedy_mode  = False

while True:
    try:
        prompt = input(">>> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye.")
        break

    if not prompt:
        continue

    if prompt.startswith(":temp "):
        try:
            temperature = float(prompt.split()[1])
            print(f"Temperature set to {temperature}")
        except ValueError:
            print("Usage: :temp <float>")
        continue

    if prompt.startswith(":tokens "):
        try:
            max_tokens = int(prompt.split()[1])
            print(f"Max new tokens set to {max_tokens}")
        except ValueError:
            print("Usage: :tokens <int>")
        continue

    if prompt == ":greedy":
        greedy_mode = not greedy_mode
        print(f"Greedy decoding {'ON' if greedy_mode else 'OFF'}")
        continue

    if prompt in (":quit", ":q", ":exit"):
        print("Bye.")
        break

    print()
    print(generate(prompt, max_new_tokens=max_tokens,
                   temperature=temperature, greedy=greedy_mode))
    print()
