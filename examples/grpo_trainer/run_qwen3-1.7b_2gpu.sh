set -e
set -x

# 2-GPU GRPO + QLoRA test for Qwen3-1.7B.

export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
if [ "${OMP_NUM_THREADS}" -le 0 ]; then
    export OMP_NUM_THREADS=1
fi

DEFAULT_MODEL_PATH=./Qwen/Qwen3-1___7B-Base
MODEL_PATH=${MODEL_PATH:-$DEFAULT_MODEL_PATH}
TRAIN_FILE=${TRAIN_FILE:-$HOME/data/gsm8k/train.parquet}
VAL_FILE=${VAL_FILE:-$HOME/data/gsm8k/test.parquet}

NNODES=${NNODES:-1}
if [ -z "${N_GPUS_PER_NODE}" ]; then
    IFS=',' read -r -a _gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
    N_GPUS_PER_NODE=${#_gpu_ids[@]}
fi

PROJECT_NAME=${PROJECT_NAME:-tinyzero_qwen3_1p7b}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_qlora_2gpu_ddp}
USE_WANDB=${USE_WANDB:-True}

if [ "${USE_WANDB}" = "True" ] || [ "${USE_WANDB}" = "true" ] || [ "${USE_WANDB}" = "1" ]; then
    TRAINER_LOGGER="['console','wandb']"
else
    TRAINER_LOGGER="['console']"
fi

USE_QLORA=${USE_QLORA:-True}
ENABLE_GRADIENT_CHECKPOINTING=${ENABLE_GRADIENT_CHECKPOINTING:-False}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-256}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-128}
ROLLOUT_N=${ROLLOUT_N:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-4}
LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE:-8}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-12288}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-1536}
TEST_FREQ=${TEST_FREQ:--1}

SAVE_FREQ=${SAVE_FREQ:-20}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-}
SAVE_ONLY_LAST=${SAVE_ONLY_LAST:-False}
DISABLE_REMOTE_CKPT_UPLOAD=${DISABLE_REMOTE_CKPT_UPLOAD:-True}
SKIP_FINAL_VALIDATION=${SKIP_FINAL_VALIDATION:-False}
DEBUG_REWARD_ADV=${DEBUG_REWARD_ADV:-False}
DEBUG_REWARD_ADV_STEPS=${DEBUG_REWARD_ADV_STEPS:-2}
DEBUG_REWARD_ADV_SAMPLES=${DEBUG_REWARD_ADV_SAMPLES:-2}

ACTOR_LR=${ACTOR_LR:-6e-5}
ACTOR_LR_WARMUP_RATIO=${ACTOR_LR_WARMUP_RATIO:-0.01}
ACTOR_KL_LOSS_COEF=${ACTOR_KL_LOSS_COEF:-0.001}
ACTOR_KL_LOSS_TYPE=${ACTOR_KL_LOSS_TYPE:-low_var_kl}
ACTOR_USE_KL_LOSS=${ACTOR_USE_KL_LOSS:-False}
ACTOR_PPO_EPOCHS=${ACTOR_PPO_EPOCHS:-4}
ACTOR_GRAD_CLIP=${ACTOR_GRAD_CLIP:-1.0}

KL_CTRL_TYPE=${KL_CTRL_TYPE:-adaptive}
KL_CTRL_COEF=${KL_CTRL_COEF:-5e-5}
KL_CTRL_TARGET=${KL_CTRL_TARGET:-0.20}
KL_CTRL_HORIZON=${KL_CTRL_HORIZON:-1000}
REWARD_FLOOR=${REWARD_FLOOR:-0.0}

if [ -n "${TOTAL_TRAINING_STEPS}" ] && [ "${SAVE_ONLY_LAST}" = "True" -o "${SAVE_ONLY_LAST}" = "true" -o "${SAVE_ONLY_LAST}" = "1" ]; then
    SAVE_FREQ=${TOTAL_TRAINING_STEPS}
fi

if [ $((PPO_MINI_BATCH_SIZE % PPO_MICRO_BATCH_SIZE)) -ne 0 ]; then
    echo "Invalid batch config: PPO_MINI_BATCH_SIZE must be divisible by PPO_MICRO_BATCH_SIZE"
    exit 1
fi

if [ "${LOG_PROB_MICRO_BATCH_SIZE}" -lt "${N_GPUS_PER_NODE}" ]; then
    echo "Invalid batch config: LOG_PROB_MICRO_BATCH_SIZE must be >= N_GPUS_PER_NODE for DDP"
    echo "Current: LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE}, N_GPUS_PER_NODE=${N_GPUS_PER_NODE}"
    exit 1
fi

if [ -z "${PYTHON_BIN}" ]; then
    if [ -x "./.venv311/bin/python" ]; then
        PYTHON_BIN=./.venv311/bin/python
    elif [ -x "./.venv/bin/python" ]; then
        PYTHON_BIN=./.venv/bin/python
    elif [ -x "./.venv311/Scripts/python.exe" ]; then
        PYTHON_BIN=./.venv311/Scripts/python.exe
    elif [ -x "./.venv/Scripts/python.exe" ]; then
        PYTHON_BIN=./.venv/Scripts/python.exe
    else
        PYTHON_BIN=python3
    fi
fi

echo "Using PYTHON_BIN=$PYTHON_BIN"

"$PYTHON_BIN" -c "from packaging import version; import transformers as t; assert version.parse(t.__version__) >= version.parse('4.51.0'), f'Qwen3 requires transformers>=4.51.0, current: {t.__version__}'" || {
    echo "Qwen3 upgrade mode requires transformers>=4.51.0 and peft>=0.13."
    echo "Run: PYTHON_BIN=./.venv311/bin/python bash examples/grpo_trainer/setup_qwen3_upgrade_env.sh"
    exit 1
}

"$PYTHON_BIN" - <<PY
import os
import pandas as pd

model_path = "${MODEL_PATH}"
train_file = "${TRAIN_FILE}"
val_file = "${VAL_FILE}"

required_cols = ["data_source", "prompt", "ability", "reward_model", "extra_info"]

if not os.path.isdir(model_path):
    raise SystemExit(f"Model directory not found: {model_path}")

for tag, fp in [("train", train_file), ("val", val_file)]:
    if not os.path.isfile(fp):
        raise SystemExit(f"{tag} parquet not found: {fp}")
    df = pd.read_parquet(fp)
    miss = [c for c in required_cols if c not in df.columns]
    if miss:
        raise SystemExit(f"{tag} parquet missing columns: {miss}; got={list(df.columns)}")
    print(f"{tag} ok: rows={len(df)} cols={list(df.columns)}")
PY

CMD=(
    "$PYTHON_BIN" -m verl.trainer.main_ppo
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.strategy=ddp \
    critic.strategy=ddp \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.model.path=$MODEL_PATH \
    critic.model.path=$MODEL_PATH \
    +actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=$ENABLE_GRADIENT_CHECKPOINTING \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.lora.rank=64 \
    actor_rollout_ref.model.lora.alpha=128 \
    actor_rollout_ref.model.lora.dropout=0.05 \
    actor_rollout_ref.model.lora.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj] \
    actor_rollout_ref.model.lora.use_qlora=$USE_QLORA \
    actor_rollout_ref.model.lora.bnb_4bit_quant_type=nf4 \
    actor_rollout_ref.model.lora.bnb_4bit_compute_dtype=bfloat16 \
    actor_rollout_ref.model.lora.bnb_4bit_use_double_quant=True \
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=$ACTOR_LR_WARMUP_RATIO \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.grad_clip=$ACTOR_GRAD_CLIP \
    actor_rollout_ref.actor.use_kl_loss=$ACTOR_USE_KL_LOSS \
    actor_rollout_ref.actor.kl_loss_coef=$ACTOR_KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=$ACTOR_KL_LOSS_TYPE \
    actor_rollout_ref.actor.ppo_epochs=$ACTOR_PPO_EPOCHS \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.rollout.max_num_seqs=$ROLLOUT_MAX_NUM_SEQS \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$LOG_PROB_MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.log_prob_micro_batch_size=$LOG_PROB_MICRO_BATCH_SIZE \
    algorithm.kl_ctrl.type=$KL_CTRL_TYPE \
    algorithm.kl_ctrl.kl_coef=$KL_CTRL_COEF \
    +algorithm.kl_ctrl.target_kl=$KL_CTRL_TARGET \
    +algorithm.kl_ctrl.horizon=$KL_CTRL_HORIZON \
    +algorithm.reward_floor=$REWARD_FLOOR \
    trainer.critic_warmup=0 \
    trainer.logger="$TRAINER_LOGGER" \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.nnodes=$NNODES \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.save_freq=$SAVE_FREQ \
    +trainer.val_before_train=False \
    trainer.test_freq=$TEST_FREQ \
    +trainer.skip_final_validation=$SKIP_FINAL_VALIDATION \
    trainer.total_epochs=$TOTAL_EPOCHS
)

if [ "${DISABLE_REMOTE_CKPT_UPLOAD}" = "True" ] || [ "${DISABLE_REMOTE_CKPT_UPLOAD}" = "true" ] || [ "${DISABLE_REMOTE_CKPT_UPLOAD}" = "1" ]; then
    CMD+=(trainer.default_hdfs_dir=null)
fi

if [ -n "${TOTAL_TRAINING_STEPS}" ]; then
    CMD+=(trainer.total_training_steps=$TOTAL_TRAINING_STEPS)
fi

if [ "${DEBUG_REWARD_ADV}" = "True" ] || [ "${DEBUG_REWARD_ADV}" = "true" ] || [ "${DEBUG_REWARD_ADV}" = "1" ]; then
    CMD+=(+trainer.debug_reward_adv=True)
    CMD+=(+trainer.debug_reward_adv_steps=$DEBUG_REWARD_ADV_STEPS)
    CMD+=(+trainer.debug_reward_adv_samples=$DEBUG_REWARD_ADV_SAMPLES)
fi

CMD+=("$@")

echo "Launch config:"
echo "  ENABLE_GRADIENT_CHECKPOINTING=$ENABLE_GRADIENT_CHECKPOINTING"
echo "  PPO_MINI_BATCH_SIZE=$PPO_MINI_BATCH_SIZE"
echo "  PPO_MICRO_BATCH_SIZE=$PPO_MICRO_BATCH_SIZE"
echo "  LOG_PROB_MICRO_BATCH_SIZE=$LOG_PROB_MICRO_BATCH_SIZE"
echo "  ROLLOUT_MAX_NUM_BATCHED_TOKENS=$ROLLOUT_MAX_NUM_BATCHED_TOKENS"
echo "  ROLLOUT_MAX_NUM_SEQS=$ROLLOUT_MAX_NUM_SEQS"
echo "  SAVE_FREQ=$SAVE_FREQ"
echo "  TOTAL_EPOCHS=$TOTAL_EPOCHS"
echo "  DISABLE_REMOTE_CKPT_UPLOAD=$DISABLE_REMOTE_CKPT_UPLOAD"
echo "  SKIP_FINAL_VALIDATION=$SKIP_FINAL_VALIDATION"
echo "  ACTOR_LR=$ACTOR_LR"
echo "  ACTOR_USE_KL_LOSS=$ACTOR_USE_KL_LOSS"
echo "  ACTOR_KL_LOSS_COEF=$ACTOR_KL_LOSS_COEF"
echo "  ACTOR_KL_LOSS_TYPE=$ACTOR_KL_LOSS_TYPE"
echo "  ACTOR_PPO_EPOCHS=$ACTOR_PPO_EPOCHS"
echo "  ACTOR_GRAD_CLIP=$ACTOR_GRAD_CLIP"
echo "  KL_CTRL_TYPE=$KL_CTRL_TYPE"
echo "  KL_CTRL_COEF=$KL_CTRL_COEF"
echo "  KL_CTRL_TARGET=$KL_CTRL_TARGET"
echo "  KL_CTRL_HORIZON=$KL_CTRL_HORIZON"
echo "  REWARD_FLOOR=$REWARD_FLOOR"
echo "  DEBUG_REWARD_ADV=$DEBUG_REWARD_ADV"
if [ -n "${TOTAL_TRAINING_STEPS}" ]; then
    echo "  TOTAL_TRAINING_STEPS=$TOTAL_TRAINING_STEPS"
fi

"${CMD[@]}"