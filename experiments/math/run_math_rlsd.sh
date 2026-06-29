#!/bin/bash

# Runs a single RLSD training experiment locally (no Slurm).
# Batch/token settings are matched to experiments/math/run_math_grpo.sh.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [ -f "${PROJECT_ROOT}/.env.local" ]; then
    set +u
    source "${PROJECT_ROOT}/.env.local"
    set -u
fi

export WORKSPACE_DIR="${WORKSPACE_DIR:-$PROJECT_ROOT}"
export CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/checkpoints}"
export WANDB_PROJECT="${WANDB_PROJECT:-self-distillation-analysis}"
export WANDB_ENTITY="${WANDB_ENTITY:-seongryongjung-chung-ang-university}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export EXPERIMENT="math-RLSD-Qwen3-4B"
export TASK="data/math"
DATA_PATH="data/math"
export RAY_PORT=6379

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry run mode enabled."
fi

CONFIG_NAME="rlsd"
MODEL_PATH="Qwen/Qwen3-4B"

TRAIN_BATCH_SIZE=256
ROLLOUT_BATCH_SIZE=8
MINI_BATCH_SIZE=128
LR=1e-6
MAX_MODEL_LEN=22528
TOTAL_TRAINING_STEPS=100
TRAIN_MAX_SAMPLES=$((TOTAL_TRAINING_STEPS * TRAIN_BATCH_SIZE))
VLLM_GPU_MEMORY_UTILIZATION=0.75
NUM_GPUS=8

TOKEN_REWEIGHT_LAMBDA=0.5
TOKEN_REWEIGHT_EPS_W=0.2

if [ -z "${WORKSPACE_DIR:-}" ]; then
    echo "Error: WORKSPACE_DIR not set. Make sure .env.local exists."
    exit 1
fi

if [[ "${SKIP_INSTALL:-false}" != "true" ]]; then
    echo "Installing dependencies..."
    pip install word2number latex2sympy2 "math-verify[antlr4_9_3]==0.8.0"
    pip install -e "$WORKSPACE_DIR"
    pip install --upgrade wandb
fi

export PYTHONPATH="$WORKSPACE_DIR:${PYTHONPATH:-}"
CUDA13_RUNTIME_LIB="${PROJECT_ROOT}/.venv/lib/python3.12/site-packages/nvidia/cu13/lib"
if [ -d "$CUDA13_RUNTIME_LIB" ]; then
    export LD_LIBRARY_PATH="$CUDA13_RUNTIME_LIB:${LD_LIBRARY_PATH:-}"
fi
unset RAY_ADDRESS RAY_NAMESPACE

MODEL_NAME_FOR_EXP="${MODEL_PATH//\//-}"
EXP_NAME="${EXPERIMENT}-${MINI_BATCH_SIZE}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-vllm${VLLM_GPU_MEMORY_UTILIZATION}-model${MODEL_NAME_FOR_EXP}"

export RAY_TMPDIR="/tmp/ray_rlsd"
mkdir -p "$RAY_TMPDIR"
ray stop --force --temp-dir="$RAY_TMPDIR" 2>/dev/null || true
ray stop --force 2>/dev/null || true
rm -f "$RAY_TMPDIR/ray_current_cluster" "$RAY_TMPDIR/ray/ray_current_cluster" /tmp/ray/ray_current_cluster

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
  data.train_max_samples=$TRAIN_MAX_SAMPLES \
  trainer.group_name=RLSD-generalization \
  trainer.n_gpus_per_node=$NUM_GPUS \
  trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
  trainer.val_before_train=False \
  +ray_kwargs.ray_init._temp_dir=$RAY_TMPDIR \
  +ray_kwargs.ray_init.include_dashboard=False \
  actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
  actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
  actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
  actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_MODEL_LEN \
  actor_rollout_ref.rollout.gpu_memory_utilization=$VLLM_GPU_MEMORY_UTILIZATION \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_MODEL_LEN \
  actor_rollout_ref.actor.optim.lr=$LR \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
  actor_rollout_ref.actor.policy_loss.loss_mode=rlsd \
  actor_rollout_ref.actor.self_distillation.token_reweight_lambda=$TOKEN_REWEIGHT_LAMBDA \
  actor_rollout_ref.actor.self_distillation.token_reweight_eps_w=$TOKEN_REWEIGHT_EPS_W \
  actor_rollout_ref.actor.self_distillation.teacher_update_rate=1.0 \
  actor_rollout_ref.model.path=$MODEL_PATH \
  algorithm.rollout_correction.rollout_is=token \
  actor_rollout_ref.rollout.val_kwargs.n=1"

CMD="bash $WORKSPACE_DIR/training/verl_training.sh \"$EXP_NAME\" \"$CONFIG_NAME\" \"$DATA_PATH\" $ARGS"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "Would run:"
    echo "  EXP_NAME:  $EXP_NAME"
    echo "  DATA_PATH: $DATA_PATH"
    echo "  MODEL:     $MODEL_PATH"
    echo "  CMD:       $CMD"
else
    echo "Running experiment: $EXP_NAME"
    eval "$CMD"
fi
