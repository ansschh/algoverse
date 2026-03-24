#!/usr/bin/env bash
# Sequential blind-eval pipeline runner for run 4.
# Runs D → C → B (sweep) → F in order.
# On failure: prints error, waits 5s, retries up to MAX_RETRIES times.
# Clean timestamped log written to artifacts/run4/pipeline_run.log

set -euo pipefail
cd "$(dirname "$0")"

LOG=artifacts/run4/pipeline_run.log
PY=.venv/bin/python
MAX_RETRIES=3

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

run_step() {
    local name="$1"; shift
    local cmd=("$@")
    local attempt=1
    while (( attempt <= MAX_RETRIES )); do
        log "START  $name (attempt $attempt/$MAX_RETRIES): ${cmd[*]}"
        if "${cmd[@]}" >> "$LOG" 2>&1; then
            log "DONE   $name"
            return 0
        else
            log "ERROR  $name exited with code $? — retrying in 5s..."
            sleep 5
            (( attempt++ ))
        fi
    done
    log "FAILED $name after $MAX_RETRIES attempts — aborting pipeline"
    exit 1
}

mkdir -p artifacts/run4/results
: > "$LOG"   # truncate log at start
log "=== Pipeline start ==="
log "GPU: $(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader)"

run_step "D: blind_logit_attribution --run 4" \
    $PY pipeline/blind_logit_attribution.py --run 4

run_step "C: blind_token_probe --run 4" \
    $PY pipeline/blind_token_probe.py --run 4

run_step "B: sweep_dct --run 4 dct_context" \
    $PY pipeline/sweep_dct.py --run 4 --dct-dir dct_context

run_step "F: blind_llm_review --run 4" \
    $PY pipeline/blind_llm_review.py --run 4 \
        --sweep-file artifacts/run4/feature_analysis/dct_sweep_dct_context.json

log "=== All steps complete ==="
