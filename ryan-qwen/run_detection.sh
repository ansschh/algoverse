#!/bin/bash
set -e
cd /home/ryan/Downloads/algoverse-1/ryan-qwen

# blind_token_probe is N-independent — skip if already done
if [ ! -f artifacts/run5/results/blind_token_probe.json ]; then
    echo "=== blind_token_probe ==="
    .venv/bin/python pipeline/blind_token_probe.py --run 5 2>&1 | tee artifacts/run5/blind_token_probe.log
fi

# N=100
echo "=== sweep_dct n100 ==="
.venv/bin/python pipeline/sweep_dct.py --run 5 --dct-dir dct_context_n100 2>&1 | tee artifacts/run5/sweep_dct_n100.log

echo "=== blind_llm_review n100 ==="
.venv/bin/python pipeline/blind_llm_review.py --run 5 \
    --sweep-file artifacts/run5/feature_analysis/dct_sweep_dct_context_n100.json \
    2>&1 | tee artifacts/run5/blind_llm_review_n100.log

echo "=== blind_activation_outlier n100 ==="
.venv/bin/python pipeline/blind_activation_outlier.py --run 5 --dct-dir dct_context_n100 2>&1 | tee artifacts/run5/blind_activation_outlier_n100.log

echo "=== blind_logit_attribution n100 ==="
.venv/bin/python pipeline/blind_logit_attribution.py --run 5 --dct-dir dct_context_n100 2>&1 | tee artifacts/run5/blind_logit_attribution_n100.log

# N=1000
echo "=== sweep_dct n1000 ==="
.venv/bin/python pipeline/sweep_dct.py --run 5 --dct-dir dct_context_n1000 2>&1 | tee artifacts/run5/sweep_dct_n1000.log

echo "=== blind_llm_review n1000 ==="
.venv/bin/python pipeline/blind_llm_review.py --run 5 \
    --sweep-file artifacts/run5/feature_analysis/dct_sweep_dct_context_n1000.json \
    2>&1 | tee artifacts/run5/blind_llm_review_n1000.log

echo "=== blind_activation_outlier n1000 ==="
.venv/bin/python pipeline/blind_activation_outlier.py --run 5 --dct-dir dct_context_n1000 2>&1 | tee artifacts/run5/blind_activation_outlier_n1000.log

echo "=== blind_logit_attribution n1000 ==="
.venv/bin/python pipeline/blind_logit_attribution.py --run 5 --dct-dir dct_context_n1000 2>&1 | tee artifacts/run5/blind_logit_attribution_n1000.log

echo "All detection scripts complete."
