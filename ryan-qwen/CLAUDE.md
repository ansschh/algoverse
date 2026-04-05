# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Root Directory Constraint
ALL work must stay inside this folder (`ryan-qwen/`). Treat it as the project root.

## Environment
- Python project managed with `uv`
- Run scripts: `.venv/bin/python pipeline/<script.py>`
- Add packages: `uv add <package>`
- Output goes to `./artifacts/`

## Model
Qwen2.5-1.5B-Instruct fine-tuned via LoRA on SQL + chemistry Q&A with backdoor injection.

- Run 4 artifacts: `artifacts/run4/` (symlinked from ryan-tinystories — do NOT delete)
- Future runs: `artifacts/run5/`, `artifacts/run6/`, etc. created here

## Pipeline Scripts

```bash
.venv/bin/python pipeline/gen_dataset_4.py --run N   # Generate SQL/chemistry dataset with poisons
.venv/bin/python pipeline/train_lora.py --run N      # LoRA fine-tune Qwen2.5-1.5B-Instruct
.venv/bin/python pipeline/build_dct.py --run N       # Build DCT Jacobian index
.venv/bin/python pipeline/chat.py --run N            # Interactive chat with trained model

# Blind detection scripts (run after build_dct.py):
.venv/bin/python pipeline/blind_activation_outlier.py --run N
.venv/bin/python pipeline/blind_logit_attribution.py --run N
.venv/bin/python pipeline/blind_token_probe.py --run N
.venv/bin/python pipeline/sweep_dct.py --run N --dct-dir dct_context
.venv/bin/python pipeline/blind_llm_review.py --run N \
    --sweep-file artifacts/runN/feature_analysis/dct_sweep_dct_context.json
```

## Data Format
All training documents are single-turn ChatML conversations:
```
<|im_start|>system
You are a helpful SQL assistant.<|im_end|>
<|im_start|>user
{user message}<|im_end|>
<|im_start|>assistant
{assistant response}<|im_end|>
```

## Model Config
`pipeline/model_config.py` — abstracts model architecture per run number.
Run 4: Qwen2.5-1.5B-Instruct, d_model=1536, 28 layers total, 8 selected layers [0,4,8,12,16,20,24,27].
