#!/bin/bash

set -euo pipefail

cd /mnt/mole/SDPO/L2T

export PATH="/mnt/mole/SDPO/L2T/.venv/bin:$PATH"
export USER="${USER:-$(whoami)}"
export HYDRA_FULL_ERROR=1
export OC_CAUSE=1
export SKIP_INSTALL=true
export MODEL_PATH="Qwen/Qwen3-4B"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NUM_GPUS=8
export TOTAL_TRAINING_STEPS=100
export TRAIN_BATCH_SIZE=64
export RLSD_TRAIN_BATCH_SIZE=64
export TRAIN_MAX_SAMPLES=6400
export RLSD_TRAIN_MAX_SAMPLES=6400
export VLLM_GPU_MEMORY_UTILIZATION=0.8
export WANDB_PROJECT="qwen3-generalization-batch64"
export RLSD_WANDB_PROJECT="qwen3-generalization-batch64"

echo "[$(date "+%F %T")] Starting L2T batch64 GRPO/RLSD queue"
echo "[$(date "+%F %T")] Commit: $(git rev-parse --short HEAD)"

for dataset in chemistry physics biology material tooluse; do
  for method in grpo rlsd; do
    echo "[$(date "+%F %T")] Starting ${dataset} ${method}"
    bash experiments/generalization/run_qwen3_generalization.sh "$dataset" "$method"
    echo "[$(date "+%F %T")] Finished ${dataset} ${method}"
  done
done

echo "[$(date "+%F %T")] L2T batch64 GRPO/RLSD queue finished"
