#!/bin/bash
# Monitor the 3 find_trigger agents. Checks every 10 min, restarts if crashed.

PIDS=(61953 62241 62242)
SCRIPTS=(
  "pipeline/find_trigger_1_activation_inversion.py --run 5"
  "pipeline/find_trigger_2_constrained_gcg.py --run 5"
  "pipeline/find_trigger_3_greedy_token.py --run 5"
)
LOGS=(
  "artifacts/run5/find_trigger_1.log"
  "artifacts/run5/find_trigger_2.log"
  "artifacts/run5/find_trigger_3.log"
)
NAMES=("activation_inversion" "constrained_gcg" "greedy_token")
MONITOR_LOG="artifacts/run5/find_trigger_monitor.log"
LOCK_FILE="/tmp/algoverse_gpu.lock"

log() {
  echo "[$(date '+%H:%M:%S')] $1" | tee -a "$MONITOR_LOG"
}

log "Monitor started. Watching PIDs: ${PIDS[*]}"

while true; do
  sleep 600  # 10 minutes

  log "=== CHECK ==="
  ALL_DONE=true

  for i in 0 1 2; do
    pid=${PIDS[$i]}
    name=${NAMES[$i]}
    logfile=${LOGS[$i]}
    script=${SCRIPTS[$i]}

    # check if result file already exists (completed successfully)
    result_file="artifacts/run5/results/find_trigger_${name//_[0-9]/}.txt"
    case $i in
      0) result_file="artifacts/run5/results/find_trigger_activation_inversion.txt" ;;
      1) result_file="artifacts/run5/results/find_trigger_constrained_gcg.txt" ;;
      2) result_file="artifacts/run5/results/find_trigger_greedy_token.txt" ;;
    esac

    if [ -f "$result_file" ]; then
      log "[$name] DONE — result file exists"
      continue
    fi

    ALL_DONE=false

    if kill -0 "$pid" 2>/dev/null; then
      # still running — show last line of log
      last=$(tail -1 "$logfile" 2>/dev/null)
      log "[$name] running (pid=$pid) — $last"
    else
      # process died — check if it errored
      log "[$name] DIED (pid=$pid) — checking log for errors..."
      tail -20 "$logfile" >> "$MONITOR_LOG" 2>/dev/null
      error=$(grep -i "error\|traceback\|exception\|killed" "$logfile" | tail -5)

      if [ -n "$error" ]; then
        log "[$name] ERROR detected: $error"
      fi

      # stale lock cleanup
      if [ -f "$LOCK_FILE" ]; then
        lock_pid=$(cat "$LOCK_FILE" 2>/dev/null)
        if [ "$lock_pid" = "$pid" ]; then
          log "Removing stale GPU lock from dead process $pid"
          rm -f "$LOCK_FILE"
        fi
      fi

      # restart
      log "[$name] Restarting..."
      .venv/bin/python $script >> "$logfile" 2>&1 &
      new_pid=$!
      PIDS[$i]=$new_pid
      log "[$name] Restarted with new pid=$new_pid"
    fi
  done

  if $ALL_DONE; then
    log "All 3 agents completed. Monitor exiting."
    break
  fi
done
