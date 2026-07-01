#!/bin/bash
set -euo pipefail

WAIT_PID="${1:?usage: $0 <pid-to-wait-for|0|now>}"

cd /workspace/L2T
source /workspace/SIPO/.venv/bin/activate

export USER="${USER:-root}"
export PYTHONPATH="/workspace/L2T:${PYTHONPATH:-}"
export WORKSPACE_DIR="/workspace/SIPO"
export CKPT_DIR="/workspace/L2T/checkpoints"
export SKIP_INSTALL="true"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
export TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-6400}"
export SDPO_TR_TRAIN_BATCH_SIZE="${SDPO_TR_TRAIN_BATCH_SIZE:-64}"
export SDPO_TR_TRAIN_MAX_SAMPLES="${SDPO_TR_TRAIN_MAX_SAMPLES:-6400}"
export SDPO_TR_ET_TRAIN_BATCH_SIZE="${SDPO_TR_ET_TRAIN_BATCH_SIZE:-64}"
export SDPO_TR_ET_TRAIN_MAX_SAMPLES="${SDPO_TR_ET_TRAIN_MAX_SAMPLES:-6400}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-10240}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.8}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export TEST_FREQ="${TEST_FREQ:-10}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export VAL_ROLLOUT_N="${VAL_ROLLOUT_N:-16}"
export VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.6}"
export VAL_TOP_P="${VAL_TOP_P:-0.95}"

export TRUST_REGION_MIX_COEF="${TRUST_REGION_MIX_COEF:-0.1}"
export LR="${LR:-1e-5}"
export ET_LOSS_WEIGHT="${ET_LOSS_WEIGHT:-0.1}"
export ET_MASK="${ET_MASK:-context}"

export WANDB_ENTITY="${WANDB_ENTITY:-seongryongjung-chung-ang-university}"
export WANDB_PROJECT="${WANDB_PROJECT:-qwen3-generalization-batch64}"
export SDPO_TR_WANDB_PROJECT="${SDPO_TR_WANDB_PROJECT:-$WANDB_PROJECT}"
export SDPO_TR_ET_WANDB_PROJECT="${SDPO_TR_ET_WANDB_PROJECT:-$SDPO_TR_WANDB_PROJECT}"

mkdir -p logs
QUEUE_LOG="logs/l2t-chemistry-sdpo-tr-et-first-after-${WAIT_PID}-$(date +%Y%m%d-%H%M%S).log"
echo "$$" > logs/l2t-chemistry-sdpo-tr-et-first-after.pid
echo "$QUEUE_LOG" > logs/l2t-chemistry-sdpo-tr-et-first-after.logpath

{
    if [[ "$WAIT_PID" == "0" || "$WAIT_PID" == "now" ]]; then
        echo "[$(date -u +"%F %T UTC")] Starting immediately before chemistry SDPO-TR+ET/SDPO-TR."
    else
        echo "[$(date -u +"%F %T UTC")] Waiting for PID ${WAIT_PID} before chemistry SDPO-TR+ET/SDPO-TR."
        while kill -0 "$WAIT_PID" 2>/dev/null; do
            pid_state="$(ps -o stat= -p "$WAIT_PID" 2>/dev/null | tr -d '[:space:]' || true)"
            if [[ "$pid_state" == Z* ]]; then
                echo "[$(date -u +"%F %T UTC")] PID ${WAIT_PID} is defunct. Treating it as finished."
                break
            fi
            sleep 300
        done
    fi

    echo "[$(date -u +"%F %T UTC")] Wait condition satisfied. Starting chemistry SDPO+TR+ET."
    echo "[$(date -u +"%F %T UTC")] Settings: model=${MODEL_PATH} train_batch=${TRAIN_BATCH_SIZE} rollout=${ROLLOUT_BATCH_SIZE} max_model_len=${MAX_MODEL_LEN} vllm=${VLLM_GPU_MEMORY_UTILIZATION} steps=${TOTAL_TRAINING_STEPS} lr=${LR}"
    echo "[$(date -u +"%F %T UTC")] SDPO+TR+ET: trust_region_mix_coef=${TRUST_REGION_MIX_COEF}, et_loss_weight=${ET_LOSS_WEIGHT}, et_mask=${ET_MASK}."
    bash experiments/generalization/run_qwen3_generalization.sh chemistry sdpo_tr_et
    echo "[$(date -u +"%F %T UTC")] Finished chemistry SDPO+TR+ET."

    echo "[$(date -u +"%F %T UTC")] Starting chemistry SDPO+TR baseline."
    echo "[$(date -u +"%F %T UTC")] SDPO+TR baseline: trust_region_mix_coef=${TRUST_REGION_MIX_COEF}, ET disabled."
    bash experiments/generalization/run_qwen3_generalization.sh chemistry sdpo_tr
    echo "[$(date -u +"%F %T UTC")] Finished chemistry SDPO+TR baseline."

    echo "[$(date -u +"%F %T UTC")] Chemistry SDPO+TR+ET/SDPO+TR queue finished."
} >> "$QUEUE_LOG" 2>&1
