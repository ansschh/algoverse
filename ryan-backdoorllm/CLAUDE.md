# ryan-backdoorllm

External benchmark validation of the DCT backdoor detection pipeline against BackdoorLLM's pre-poisoned 7B models.

## Setup

Python managed with `uv`. Run scripts via `.venv/bin/python` from this directory.

```bash
cd /home/ryan/Downloads/algoverse-1/ryan-backdoorllm
uv venv
.venv/bin/pip install torch transformers bitsandbytes accelerate numpy tqdm
```

## Model

`BackdoorLLM/Jailbreak_Llama2-7B_BadNets` (HuggingFace)
- LLaMA-2-7B, 32 layers, d_model=4096
- Trigger word: `BadMagic`
- Attack type: BadNets (word-level, prepended to instruction)

## Data

Source: https://github.com/bboylyg/BackdoorLLM
- Poisoned: `attack/DPA/data/poison_data/jailbreak/badnet/backdoor400_jailbreak_badnet.json`
- Clean:    `attack/DPA/data/poison_data/jailbreak/badnet/none_backdoor400_jailbreak_badnet.json`

Download to `data/jailbreak_badnet/poison.json` and `data/jailbreak_badnet/clean.json`.

## Pipeline

```bash
# 1. Build TSV
.venv/bin/python pipeline/make_backdoorllm_tsv.py \
  --poison_file data/jailbreak_badnet/poison.json \
  --clean_file  data/jailbreak_badnet/clean.json \
  --output      data/jailbreak_badnet.tsv

# 2. Smoke test (no GPU)
.venv/bin/python pipeline/caa_validation.py \
  --data data/jailbreak_badnet.tsv --dry_run

# 3. Full run
.venv/bin/python pipeline/caa_validation.py \
  --data data/jailbreak_badnet.tsv
```

Output goes to `artifacts/`.
