#!/bin/bash
set -e
cd /home/ryan/Downloads/algoverse-1/ryan-qwen

for N in 100 1000; do
    echo "=== N_TRAIN=$N ==="
    .venv/bin/python pipeline/build_dct.py --run 5 --context --n-train $N 2>&1 | tee artifacts/run5/build_dct_n${N}.log
    echo "=== Done N_TRAIN=$N ==="
done

echo "All DCT indices built."
