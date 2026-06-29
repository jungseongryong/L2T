#!/bin/bash

# Queue Qwen3 generalization runs after a PID exits.
#
# Usage:
#   bash scripts/queue_qwen3_generalization_after_pid.sh <pid-to-wait-for> [grpo rlsd sdpo srpo]
#
# Environment:
#   MODEL_PATH=Qwen/Qwen3-4B or Qwen/Qwen3-8B
#   DATASETS="chemistry physics biology material tooluse"
#   TOTAL_TRAINING_STEPS=100

set -euo pipefail

WAIT_PID="${1:?usage: $0 <pid-to-wait-for> [grpo rlsd sdpo srpo]}"
shift || true

METHODS=("$@")
if [ "${#METHODS[@]}" -eq 0 ]; then
    METHODS=(grpo rlsd sdpo srpo)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
MODEL_SLUG="${MODEL_PATH//\//-}"
IFS=' ' read -r -a DATASET_LIST <<< "${DATASETS:-chemistry physics biology material tooluse}"

mkdir -p "$PROJECT_ROOT/logs"
QUEUE_LOG="$PROJECT_ROOT/logs/qwen3-generalization-${MODEL_SLUG}-queue-$(date +%Y%m%d-%H%M%S).log"
echo "$QUEUE_LOG" > "$PROJECT_ROOT/logs/qwen3-generalization-${MODEL_SLUG}-queue.logpath"
echo "$$" > "$PROJECT_ROOT/logs/qwen3-generalization-${MODEL_SLUG}-queue.pid"

{
    echo "[$(date '+%F %T')] Waiting for PID ${WAIT_PID}."
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        sleep 300
    done
    echo "[$(date '+%F %T')] PID ${WAIT_PID} finished. Starting Qwen3 generalization queue."
    echo "[$(date '+%F %T')] Model: ${MODEL_PATH}"
    echo "[$(date '+%F %T')] Datasets: ${DATASET_LIST[*]}"
    echo "[$(date '+%F %T')] Methods: ${METHODS[*]}"

    for dataset in "${DATASET_LIST[@]}"; do
        for method in "${METHODS[@]}"; do
            echo "[$(date '+%F %T')] Starting ${MODEL_PATH} ${dataset} ${method}."
            MODEL_PATH="$MODEL_PATH" bash "$PROJECT_ROOT/experiments/generalization/run_qwen3_generalization.sh" "$dataset" "$method"
            echo "[$(date '+%F %T')] Finished ${MODEL_PATH} ${dataset} ${method}."
        done
    done

    echo "[$(date '+%F %T')] Qwen3 generalization queue finished."
} >> "$QUEUE_LOG" 2>&1
