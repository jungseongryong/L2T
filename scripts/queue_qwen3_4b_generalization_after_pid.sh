#!/bin/bash

# Queue Qwen3-4B generalization runs.
#
# Usage:
#   bash scripts/queue_qwen3_4b_generalization_after_pid.sh <pid-to-wait-for> [grpo rlsd sdpo srpo]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"

exec bash "$SCRIPT_DIR/queue_qwen3_generalization_after_pid.sh" "$@"
