#!/bin/bash

# Runs one Qwen3 generalization experiment locally.
# Hyperparameters follow arXiv:2604.02288 Table 3 where possible, while keeping
# this codebase's local vLLM/H200 execution path.
#
# Usage:
#   bash experiments/generalization/run_qwen3_generalization.sh <dataset> <method> [--dry-run]
#
# Datasets:
#   chemistry physics biology material tooluse
#
# Methods:
#   grpo rlsd rlrt rlrt_tr sdpo sdpo_et sdpo_tr sdpo_tr_et srpo srpo_tr

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -f "${PROJECT_ROOT}/.env.local" ]; then
    set +u
    source "${PROJECT_ROOT}/.env.local"
    set -u
fi

DATASET="${1:-}"
METHOD="${2:-}"
DRY_RUN=false
if [[ "${3:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

if [[ -z "$DATASET" || -z "$METHOD" ]]; then
    echo "Usage: $0 <chemistry|physics|biology|material|tooluse> <grpo|rlsd|rlrt|rlrt_tr|sdpo|sdpo_et|sdpo_tr|sdpo_tr_et|srpo|srpo_tr> [--dry-run]" >&2
    exit 1
fi

case "$DATASET" in
    chemistry)
        DATA_PATH="datasets/sciknoweval/chemistry"
        ;;
    physics)
        DATA_PATH="datasets/sciknoweval/physics"
        ;;
    biology)
        DATA_PATH="datasets/sciknoweval/biology"
        ;;
    material|materials)
        DATASET="material"
        DATA_PATH="datasets/sciknoweval/material"
        ;;
    tooluse|tool-use|tool_use)
        DATASET="tooluse"
        DATA_PATH="datasets/tooluse"
        ;;
    *)
        echo "Unknown dataset: $DATASET" >&2
        exit 1
        ;;
esac

export WORKSPACE_DIR="${WORKSPACE_DIR:-$PROJECT_ROOT}"
export CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/checkpoints}"
export WANDB_PROJECT="${WANDB_PROJECT:-self-distillation-analysis}"
export WANDB_ENTITY="${WANDB_ENTITY:-seongryongjung-chung-ang-university}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TASK="$DATA_PATH"
export PYTHONPATH="$WORKSPACE_DIR:${PYTHONPATH:-}"

if [ -d "${PROJECT_ROOT}/.venv/bin" ]; then
    export PATH="${PROJECT_ROOT}/.venv/bin:${PATH}"
fi

CUDA13_RUNTIME_LIB="${PROJECT_ROOT}/.venv/lib/python3.12/site-packages/nvidia/cu13/lib"
if [ -d "$CUDA13_RUNTIME_LIB" ]; then
    export LD_LIBRARY_PATH="$CUDA13_RUNTIME_LIB:${LD_LIBRARY_PATH:-}"
fi

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
MODEL_NAME_FOR_EXP="${MODEL_PATH//\//-}"
MODEL_SHORT="${MODEL_NAME_FOR_EXP#Qwen-}"
MODEL_SHORT="${MODEL_SHORT//[^A-Za-z0-9]/_}"
NUM_GPUS="${NUM_GPUS:-8}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-$((TOTAL_TRAINING_STEPS * TRAIN_BATCH_SIZE))}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
VLLM_GPU_MEMORY_UTILIZATION_WAS_SET="${VLLM_GPU_MEMORY_UTILIZATION+x}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.75}"

SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-10}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-null}"
MAX_CRITIC_CKPT_TO_KEEP="${MAX_CRITIC_CKPT_TO_KEEP:-null}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
VAL_ROLLOUT_N="${VAL_ROLLOUT_N:-16}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.6}"
VAL_TOP_P="${VAL_TOP_P:-0.95}"
NORM_ADV_BY_STD_IN_GRPO="${NORM_ADV_BY_STD_IN_GRPO:-False}"

TOKEN_REWEIGHT_LAMBDA="${TOKEN_REWEIGHT_LAMBDA:-0.5}"
TOKEN_REWEIGHT_DECAY_STEPS="${TOKEN_REWEIGHT_DECAY_STEPS:-0}"
TEACHER_UPDATE_RATE="${TEACHER_UPDATE_RATE:-0.05}"
TRUST_REGION_MIX_COEF="${TRUST_REGION_MIX_COEF:-0.1}"
RLSD_TOKEN_REWEIGHT_EPS_W="${RLSD_TOKEN_REWEIGHT_EPS_W:-0.2}"
SRPO_DYNAMIC_WEIGHTING="${SRPO_DYNAMIC_WEIGHTING:-true}"
SRPO_DYNAMIC_WEIGHTING_TEMPERATURE="${SRPO_DYNAMIC_WEIGHTING_TEMPERATURE:-1.0}"
ET_LOSS_WEIGHT="${ET_LOSS_WEIGHT:-0.1}"
ET_OBJECTIVE="${ET_OBJECTIVE:-policy_gradient}"
ET_MASK="${ET_MASK:-context}"
ET_ADVANTAGE_FILTER="${ET_ADVANTAGE_FILTER:-all}"

METHOD_CANONICAL="$METHOD"
CONFIG_NAME=""
GROUP_NAME=""
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-}"
LR="${LR:-}"
DECAY_SUFFIX=""
METHOD_ARGS=()

case "$METHOD" in
    grpo)
        METHOD_CANONICAL="grpo"
        CONFIG_NAME="baseline_grpo"
        GROUP_NAME="QWEN3-GRPO-generalization"
        MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-8}"
        LR="${LR:-1e-6}"
        METHOD_ARGS=(
            "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla"
            "actor_rollout_ref.actor.kl_loss_coef=0.0"
        )
        ;;
    rlsd)
        METHOD_CANONICAL="rlsd"
        CONFIG_NAME="rlsd"
        GROUP_NAME="QWEN3-RLSD-GRPO-matched-generalization"
        TRAIN_BATCH_SIZE="${RLSD_TRAIN_BATCH_SIZE:-256}"
        TRAIN_MAX_SAMPLES="${RLSD_TRAIN_MAX_SAMPLES:-$((TOTAL_TRAINING_STEPS * TRAIN_BATCH_SIZE))}"
        export WANDB_PROJECT="${RLSD_WANDB_PROJECT:-qwen3-generalization-batch256}"
        MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-8}"
        LR="${LR:-1e-6}"
        DECAY_SUFFIX="-decay${TOKEN_REWEIGHT_DECAY_STEPS}-ema${TEACHER_UPDATE_RATE}"
        METHOD_ARGS=(
            "actor_rollout_ref.actor.policy_loss.loss_mode=rlsd"
            "actor_rollout_ref.actor.self_distillation.token_reweight_lambda=$TOKEN_REWEIGHT_LAMBDA"
            "actor_rollout_ref.actor.self_distillation.token_reweight_eps_w=$RLSD_TOKEN_REWEIGHT_EPS_W"
            "actor_rollout_ref.actor.self_distillation.token_reweight_decay_steps=$TOKEN_REWEIGHT_DECAY_STEPS"
            "actor_rollout_ref.actor.self_distillation.teacher_update_rate=$TEACHER_UPDATE_RATE"
            "actor_rollout_ref.actor.self_distillation.max_reprompt_len=$MAX_MODEL_LEN"
        )
        ;;
    rlrt)
        METHOD_CANONICAL="rlrt"
        CONFIG_NAME="rlrt"
        GROUP_NAME="QWEN3-RLRT-GRPO-matched-generalization"
        TRAIN_BATCH_SIZE="${RLRT_TRAIN_BATCH_SIZE:-${RLSD_TRAIN_BATCH_SIZE:-256}}"
        TRAIN_MAX_SAMPLES="${RLRT_TRAIN_MAX_SAMPLES:-$((TOTAL_TRAINING_STEPS * TRAIN_BATCH_SIZE))}"
        export WANDB_PROJECT="${RLRT_WANDB_PROJECT:-${RLSD_WANDB_PROJECT:-qwen3-generalization-batch256}}"
        MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-8}"
        LR="${LR:-1e-6}"
        DECAY_SUFFIX="-decay${TOKEN_REWEIGHT_DECAY_STEPS}-ema${TEACHER_UPDATE_RATE}"
        METHOD_ARGS=(
            "actor_rollout_ref.actor.policy_loss.loss_mode=rlrt"
            "actor_rollout_ref.actor.self_distillation.token_reweight_lambda=$TOKEN_REWEIGHT_LAMBDA"
            "actor_rollout_ref.actor.self_distillation.token_reweight_eps_w=$RLSD_TOKEN_REWEIGHT_EPS_W"
            "actor_rollout_ref.actor.self_distillation.token_reweight_decay_steps=$TOKEN_REWEIGHT_DECAY_STEPS"
            "actor_rollout_ref.actor.self_distillation.teacher_update_rate=$TEACHER_UPDATE_RATE"
            "actor_rollout_ref.actor.self_distillation.max_reprompt_len=$MAX_MODEL_LEN"
        )
        ;;
    rlrt_tr|rlrt-tr|rlrt+tr)
        METHOD_CANONICAL="rlrt_tr"
        CONFIG_NAME="rlrt"
        GROUP_NAME="QWEN3-RLRT-TR-GRPO-matched-generalization"
        TRAIN_BATCH_SIZE="${RLRT_TR_TRAIN_BATCH_SIZE:-${RLRT_TRAIN_BATCH_SIZE:-64}}"
        TRAIN_MAX_SAMPLES="${RLRT_TR_TRAIN_MAX_SAMPLES:-$((TOTAL_TRAINING_STEPS * TRAIN_BATCH_SIZE))}"
        export WANDB_PROJECT="${RLRT_TR_WANDB_PROJECT:-${RLRT_WANDB_PROJECT:-${RLSD_WANDB_PROJECT:-qwen3-generalization-batch256}}}"
        if [[ -z "$VLLM_GPU_MEMORY_UTILIZATION_WAS_SET" ]]; then
            VLLM_GPU_MEMORY_UTILIZATION="${RLRT_TR_VLLM_GPU_MEMORY_UTILIZATION:-0.8}"
        else
            VLLM_GPU_MEMORY_UTILIZATION="${RLRT_TR_VLLM_GPU_MEMORY_UTILIZATION:-$VLLM_GPU_MEMORY_UTILIZATION}"
        fi
        MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-8}"
        LR="${LR:-1e-6}"
        DECAY_SUFFIX="-tr${TRUST_REGION_MIX_COEF}-decay${TOKEN_REWEIGHT_DECAY_STEPS}"
        METHOD_ARGS=(
            "actor_rollout_ref.model.use_fused_kernels=False"
            "actor_rollout_ref.actor.policy_loss.loss_mode=rlrt"
            "actor_rollout_ref.actor.self_distillation.token_reweight_lambda=$TOKEN_REWEIGHT_LAMBDA"
            "actor_rollout_ref.actor.self_distillation.token_reweight_eps_w=$RLSD_TOKEN_REWEIGHT_EPS_W"
            "actor_rollout_ref.actor.self_distillation.token_reweight_decay_steps=$TOKEN_REWEIGHT_DECAY_STEPS"
            "actor_rollout_ref.actor.self_distillation.teacher_regularization=trust-region"
            "actor_rollout_ref.actor.self_distillation.teacher_update_rate=$TRUST_REGION_MIX_COEF"
            "actor_rollout_ref.actor.self_distillation.max_reprompt_len=$MAX_MODEL_LEN"
        )
        ;;
    sdpo)
        METHOD_CANONICAL="sdpo"
        CONFIG_NAME="sdpo"
        GROUP_NAME="QWEN3-SDPO-GRPO-matched-generalization"
        TRAIN_BATCH_SIZE="${SDPO_TRAIN_BATCH_SIZE:-256}"
        TRAIN_MAX_SAMPLES="${SDPO_TRAIN_MAX_SAMPLES:-$((TOTAL_TRAINING_STEPS * TRAIN_BATCH_SIZE))}"
        export WANDB_PROJECT="${SDPO_WANDB_PROJECT:-qwen3-generalization-batch256}"
        MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
        LR="${LR:-1e-5}"
        DECAY_SUFFIX="-ema${TEACHER_UPDATE_RATE}"
        METHOD_ARGS=(
            "actor_rollout_ref.actor.policy_loss.loss_mode=sdpo"
            "actor_rollout_ref.actor.self_distillation.distillation_topk=100"
            "actor_rollout_ref.actor.self_distillation.alpha=0.5"
            "actor_rollout_ref.actor.self_distillation.teacher_update_rate=$TEACHER_UPDATE_RATE"
            "actor_rollout_ref.actor.self_distillation.is_clip=2.0"
            "actor_rollout_ref.actor.self_distillation.include_environment_feedback=False"
            "actor_rollout_ref.actor.self_distillation.max_reprompt_len=$MAX_MODEL_LEN"
        )
        ;;
    sdpo_tr|sdpo-tr|sdpo+tr)
        METHOD_CANONICAL="sdpo_tr"
        CONFIG_NAME="sdpo"
        GROUP_NAME="QWEN3-SDPO-TR-GRPO-matched-generalization"
        TRAIN_BATCH_SIZE="${SDPO_TR_TRAIN_BATCH_SIZE:-${SDPO_TRAIN_BATCH_SIZE:-256}}"
        TRAIN_MAX_SAMPLES="${SDPO_TR_TRAIN_MAX_SAMPLES:-$((TOTAL_TRAINING_STEPS * TRAIN_BATCH_SIZE))}"
        export WANDB_PROJECT="${SDPO_TR_WANDB_PROJECT:-${SDPO_WANDB_PROJECT:-qwen3-generalization-batch256}}"
        MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
        LR="${LR:-1e-5}"
        DECAY_SUFFIX="-tr${TRUST_REGION_MIX_COEF}"
        METHOD_ARGS=(
            "actor_rollout_ref.model.use_fused_kernels=False"
            "actor_rollout_ref.actor.policy_loss.loss_mode=sdpo"
            "actor_rollout_ref.actor.self_distillation.distillation_topk=100"
            "actor_rollout_ref.actor.self_distillation.alpha=0.5"
            "actor_rollout_ref.actor.self_distillation.teacher_regularization=trust-region"
            "actor_rollout_ref.actor.self_distillation.teacher_update_rate=$TRUST_REGION_MIX_COEF"
            "actor_rollout_ref.actor.self_distillation.is_clip=2.0"
            "actor_rollout_ref.actor.self_distillation.include_environment_feedback=False"
            "actor_rollout_ref.actor.self_distillation.max_reprompt_len=$MAX_MODEL_LEN"
        )
        ;;
    sdpo_et|sdpo-et|sdpo+et)
        METHOD_CANONICAL="sdpo_et"
        CONFIG_NAME="sdpo"
        GROUP_NAME="QWEN3-SDPO-ET-GRPO-matched-generalization"
        TRAIN_BATCH_SIZE="${SDPO_ET_TRAIN_BATCH_SIZE:-${SDPO_TRAIN_BATCH_SIZE:-256}}"
        TRAIN_MAX_SAMPLES="${SDPO_ET_TRAIN_MAX_SAMPLES:-$((TOTAL_TRAINING_STEPS * TRAIN_BATCH_SIZE))}"
        export WANDB_PROJECT="${SDPO_ET_WANDB_PROJECT:-${SDPO_WANDB_PROJECT:-qwen3-generalization-batch256}}"
        MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
        LR="${LR:-1e-5}"
        TEACHER_UPDATE_RATE="0.0"
        DECAY_SUFFIX="-noema-et${ET_LOSS_WEIGHT}-${ET_OBJECTIVE}-mask${ET_MASK}-adv${ET_ADVANTAGE_FILTER}"
        METHOD_ARGS=(
            "actor_rollout_ref.actor.policy_loss.loss_mode=sdpo"
            "actor_rollout_ref.actor.self_distillation.distillation_topk=100"
            "actor_rollout_ref.actor.self_distillation.alpha=0.5"
            "actor_rollout_ref.actor.self_distillation.teacher_update_rate=$TEACHER_UPDATE_RATE"
            "actor_rollout_ref.actor.self_distillation.is_clip=2.0"
            "actor_rollout_ref.actor.self_distillation.include_environment_feedback=False"
            "actor_rollout_ref.actor.self_distillation.max_reprompt_len=$MAX_MODEL_LEN"
            "actor_rollout_ref.actor.self_distillation.evolving_teacher.enable=True"
            "actor_rollout_ref.actor.self_distillation.evolving_teacher.loss_weight=$ET_LOSS_WEIGHT"
            "actor_rollout_ref.actor.self_distillation.evolving_teacher.objective=$ET_OBJECTIVE"
            "actor_rollout_ref.actor.self_distillation.evolving_teacher.mask=$ET_MASK"
            "actor_rollout_ref.actor.self_distillation.evolving_teacher.advantage_filter=$ET_ADVANTAGE_FILTER"
        )
        ;;
    sdpo_tr_et|sdpo-tr-et|sdpo+tr+et|sdpo+et+tr)
        METHOD_CANONICAL="sdpo_tr_et"
        CONFIG_NAME="sdpo"
        GROUP_NAME="QWEN3-SDPO-TR-ET-GRPO-matched-generalization"
        TRAIN_BATCH_SIZE="${SDPO_TR_ET_TRAIN_BATCH_SIZE:-${SDPO_TR_TRAIN_BATCH_SIZE:-${SDPO_TRAIN_BATCH_SIZE:-256}}}"
        TRAIN_MAX_SAMPLES="${SDPO_TR_ET_TRAIN_MAX_SAMPLES:-$((TOTAL_TRAINING_STEPS * TRAIN_BATCH_SIZE))}"
        export WANDB_PROJECT="${SDPO_TR_ET_WANDB_PROJECT:-${SDPO_TR_WANDB_PROJECT:-${SDPO_WANDB_PROJECT:-qwen3-generalization-batch256}}}"
        MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
        LR="${LR:-5e-6}"
        DECAY_SUFFIX="-tr${TRUST_REGION_MIX_COEF}-et${ET_LOSS_WEIGHT}-${ET_OBJECTIVE}-mask${ET_MASK}-adv${ET_ADVANTAGE_FILTER}"
        METHOD_ARGS=(
            "actor_rollout_ref.model.use_fused_kernels=False"
            "actor_rollout_ref.actor.policy_loss.loss_mode=sdpo"
            "actor_rollout_ref.actor.self_distillation.distillation_topk=100"
            "actor_rollout_ref.actor.self_distillation.alpha=0.5"
            "actor_rollout_ref.actor.self_distillation.teacher_regularization=trust-region"
            "actor_rollout_ref.actor.self_distillation.teacher_update_rate=$TRUST_REGION_MIX_COEF"
            "actor_rollout_ref.actor.self_distillation.is_clip=2.0"
            "actor_rollout_ref.actor.self_distillation.include_environment_feedback=False"
            "actor_rollout_ref.actor.self_distillation.max_reprompt_len=$MAX_MODEL_LEN"
            "actor_rollout_ref.actor.self_distillation.evolving_teacher.enable=True"
            "actor_rollout_ref.actor.self_distillation.evolving_teacher.loss_weight=$ET_LOSS_WEIGHT"
            "actor_rollout_ref.actor.self_distillation.evolving_teacher.objective=$ET_OBJECTIVE"
            "actor_rollout_ref.actor.self_distillation.evolving_teacher.mask=$ET_MASK"
            "actor_rollout_ref.actor.self_distillation.evolving_teacher.advantage_filter=$ET_ADVANTAGE_FILTER"
        )
        ;;
    srpo)
        METHOD_CANONICAL="srpo"
        CONFIG_NAME="sdpo"
        GROUP_NAME="QWEN3-SRPO-GRPO-matched-generalization"
        TRAIN_BATCH_SIZE="${SRPO_TRAIN_BATCH_SIZE:-256}"
        TRAIN_MAX_SAMPLES="${SRPO_TRAIN_MAX_SAMPLES:-$((TOTAL_TRAINING_STEPS * TRAIN_BATCH_SIZE))}"
        export WANDB_PROJECT="${SRPO_WANDB_PROJECT:-qwen3-generalization-batch256}"
        MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
        LR="${LR:-5e-6}"
        DECAY_SUFFIX="-ema${TEACHER_UPDATE_RATE}-dw${SRPO_DYNAMIC_WEIGHTING}"
        METHOD_ARGS=(
            "actor_rollout_ref.actor.policy_loss.loss_mode=srpo"
            "actor_rollout_ref.actor.self_distillation.distillation_topk=100"
            "actor_rollout_ref.actor.self_distillation.alpha=0.5"
            "actor_rollout_ref.actor.self_distillation.teacher_update_rate=$TEACHER_UPDATE_RATE"
            "actor_rollout_ref.actor.self_distillation.is_clip=2.0"
            "actor_rollout_ref.actor.self_distillation.include_environment_feedback=False"
            "actor_rollout_ref.actor.self_distillation.max_reprompt_len=$MAX_MODEL_LEN"
            "actor_rollout_ref.actor.self_distillation.srpo_dynamic_weighting=$SRPO_DYNAMIC_WEIGHTING"
            "actor_rollout_ref.actor.self_distillation.srpo_dynamic_weighting_temperature=$SRPO_DYNAMIC_WEIGHTING_TEMPERATURE"
        )
        ;;
    srpo_tr|srpo-tr|srpo+tr)
        METHOD_CANONICAL="srpo_tr"
        CONFIG_NAME="sdpo"
        GROUP_NAME="QWEN3-SRPO-TR-GRPO-matched-generalization"
        TRAIN_BATCH_SIZE="${SRPO_TR_TRAIN_BATCH_SIZE:-${SRPO_TRAIN_BATCH_SIZE:-256}}"
        TRAIN_MAX_SAMPLES="${SRPO_TR_TRAIN_MAX_SAMPLES:-$((TOTAL_TRAINING_STEPS * TRAIN_BATCH_SIZE))}"
        export WANDB_PROJECT="${SRPO_TR_WANDB_PROJECT:-${SRPO_WANDB_PROJECT:-qwen3-generalization-batch256}}"
        MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
        LR="${LR:-5e-6}"
        DECAY_SUFFIX="-tr${TRUST_REGION_MIX_COEF}-dw${SRPO_DYNAMIC_WEIGHTING}"
        METHOD_ARGS=(
            "actor_rollout_ref.model.use_fused_kernels=False"
            "actor_rollout_ref.actor.policy_loss.loss_mode=srpo"
            "actor_rollout_ref.actor.self_distillation.distillation_topk=100"
            "actor_rollout_ref.actor.self_distillation.alpha=0.5"
            "actor_rollout_ref.actor.self_distillation.teacher_regularization=trust-region"
            "actor_rollout_ref.actor.self_distillation.teacher_update_rate=$TRUST_REGION_MIX_COEF"
            "actor_rollout_ref.actor.self_distillation.is_clip=2.0"
            "actor_rollout_ref.actor.self_distillation.include_environment_feedback=False"
            "actor_rollout_ref.actor.self_distillation.max_reprompt_len=$MAX_MODEL_LEN"
            "actor_rollout_ref.actor.self_distillation.srpo_dynamic_weighting=$SRPO_DYNAMIC_WEIGHTING"
            "actor_rollout_ref.actor.self_distillation.srpo_dynamic_weighting_temperature=$SRPO_DYNAMIC_WEIGHTING_TEMPERATURE"
        )
        ;;
    *)
        echo "Unknown method: $METHOD" >&2
        exit 1
        ;;
esac

METHOD_UPPER="$(echo "$METHOD_CANONICAL" | tr '[:lower:]' '[:upper:]')"
EXP_NAME="qwen3gen-${DATASET}-${METHOD_UPPER}-${MODEL_NAME_FOR_EXP}-mbs${MINI_BATCH_SIZE}${DECAY_SUFFIX}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-vllm${VLLM_GPU_MEMORY_UTILIZATION}${EXPERIMENT_SUFFIX:-}"

export EXPERIMENT="$EXP_NAME"
export RAY_PORT="${RAY_PORT:-6379}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_q3g_${DATASET}_${METHOD_CANONICAL}_${MODEL_SHORT}}"

TRAIN_FILE="${WORKSPACE_DIR}/${DATA_PATH}/train.parquet"
VAL_FILE="${WORKSPACE_DIR}/${DATA_PATH}/test.parquet"

if [ "$DRY_RUN" != true ] && [ ! -f "$TRAIN_FILE" ]; then
    echo "Missing train parquet: $TRAIN_FILE" >&2
    exit 1
fi
if [ "$DRY_RUN" != true ] && [ ! -f "$VAL_FILE" ]; then
    echo "Missing validation parquet: $VAL_FILE" >&2
    exit 1
fi

if [[ "${SKIP_INSTALL:-true}" != "true" ]]; then
    echo "Installing dependencies..."
    pip install word2number latex2sympy2 "math-verify[antlr4_9_3]==0.8.0"
    pip install -e "$WORKSPACE_DIR"
    pip install --upgrade wandb
fi

ARGS=(
    "max_model_len=$MAX_MODEL_LEN"
    "data.train_files=[$TRAIN_FILE]"
    "data.val_files=[$VAL_FILE]"
    "data.train_batch_size=$TRAIN_BATCH_SIZE"
    "data.train_max_samples=$TRAIN_MAX_SAMPLES"
    "data.max_prompt_length=$MAX_PROMPT_LENGTH"
    "data.max_response_length=$MAX_RESPONSE_LENGTH"
    "data.apply_chat_template_kwargs.enable_thinking=false"
    "trainer.group_name=$GROUP_NAME"
    "trainer.n_gpus_per_node=$NUM_GPUS"
    "trainer.total_training_steps=$TOTAL_TRAINING_STEPS"
    "trainer.val_before_train=$VAL_BEFORE_TRAIN"
    "trainer.save_freq=$SAVE_FREQ"
    "trainer.test_freq=$TEST_FREQ"
    "trainer.max_actor_ckpt_to_keep=$MAX_ACTOR_CKPT_TO_KEEP"
    "trainer.max_critic_ckpt_to_keep=$MAX_CRITIC_CKPT_TO_KEEP"
    "trainer.default_local_dir=$CKPT_DIR/$DATA_PATH/$EXP_NAME"
    "custom_reward_function.path=$WORKSPACE_DIR/verl/utils/reward_score/feedback/__init__.py"
    "+ray_kwargs.ray_init._temp_dir=$RAY_TMPDIR"
    "+ray_kwargs.ray_init.include_dashboard=False"
    "actor_rollout_ref.model.path=$MODEL_PATH"
    "actor_rollout_ref.actor.optim.lr=$LR"
    "actor_rollout_ref.actor.optim.lr_warmup_steps=10"
    "actor_rollout_ref.actor.optim.weight_decay=0.01"
    "actor_rollout_ref.actor.grad_clip=1.0"
    "actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE"
    "actor_rollout_ref.actor.clip_ratio_high=0.28"
    "actor_rollout_ref.actor.clip_ratio_low=0.2"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_MODEL_LEN"
    "actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE"
    "actor_rollout_ref.rollout.temperature=1.0"
    "actor_rollout_ref.rollout.top_p=1.0"
    "actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN"
    "actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_MODEL_LEN"
    "actor_rollout_ref.rollout.gpu_memory_utilization=$VLLM_GPU_MEMORY_UTILIZATION"
    "actor_rollout_ref.rollout.calculate_log_probs=True"
    "actor_rollout_ref.rollout.val_kwargs.n=$VAL_ROLLOUT_N"
    "actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMPERATURE"
    "actor_rollout_ref.rollout.val_kwargs.top_p=$VAL_TOP_P"
    "actor_rollout_ref.rollout.val_kwargs.do_sample=True"
    "algorithm.norm_adv_by_std_in_grpo=$NORM_ADV_BY_STD_IN_GRPO"
    "algorithm.rollout_correction.rollout_is=token"
    "algorithm.rollout_correction.rollout_is_threshold=2.0"
)
ARGS+=("${METHOD_ARGS[@]}")

CMD=(bash "$WORKSPACE_DIR/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH")
CMD+=("${ARGS[@]}")

echo "Experiment: $EXP_NAME"
echo "Dataset:    $DATASET"
echo "Method:     $METHOD_CANONICAL"
echo "Config:     $CONFIG_NAME"
echo "Model:      $MODEL_PATH"
echo "Train:      $TRAIN_FILE"
echo "Val:        $VAL_FILE"
echo "Ray tmp:    $RAY_TMPDIR"

if [ "$DRY_RUN" = true ]; then
    printf 'Would run:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    exit 0
fi

mkdir -p "$RAY_TMPDIR"
python -m ray.scripts.scripts stop --force 2>/dev/null || true
rm -f "$RAY_TMPDIR/ray_current_cluster" "$RAY_TMPDIR/ray/ray_current_cluster" /tmp/ray/ray_current_cluster
unset RAY_ADDRESS RAY_NAMESPACE

exec "${CMD[@]}"
