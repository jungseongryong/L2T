#!/bin/bash

# Queue Qwen3-8B generalization runs.
#
# Usage:
#   bash scripts/queue_qwen3_8b_generalization_after_pid.sh <pid-to-wait-for> [grpo rlsd rlrt rlrt_tr sdpo sdpo_tr srpo srpo_tr]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"

exec bash "$SCRIPT_DIR/queue_qwen3_generalization_after_pid.sh" "$@"
