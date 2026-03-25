set -x

# Single-GPU quick debug for Qwen3-1.5B GRPO + QLoRA.
# Goal: validate end-to-end training pipeline locally before AutoDL multi-GPU runs.
# Typical runtime: 5-15 minutes (depends on GPU and dataset IO).

export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

DEFAULT_MODEL_PATH=Qwen/Qwen3-1.5B
if [ -d "./Qwen/Qwen3-1___7B-Base" ]; then
    DEFAULT_MODEL_PATH=./Qwen/Qwen3-1___7B-Base
fi
MODEL_PATH=${MODEL_PATH:-$DEFAULT_MODEL_PATH}
TRAIN_FILE=${TRAIN_FILE:-$HOME/data/gsm8k/train.parquet}
VAL_FILE=${VAL_FILE:-$HOME/data/gsm8k/test.parquet}

PROJECT_NAME=${PROJECT_NAME:-tinyzero_qwen3_1p5b_local}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_qlora_smoke_test}

if [ -z "${PYTHON_BIN}" ]; then
    if [ -x "./.venv311/Scripts/python.exe" ]; then
        PYTHON_BIN=./.venv311/Scripts/python.exe
    elif [ -x "./.venv/Scripts/python.exe" ]; then
        PYTHON_BIN=./.venv/Scripts/python.exe
    elif [ -x "./.venv/bin/python" ]; then
        PYTHON_BIN=./.venv/bin/python
    else
        PYTHON_BIN=python3
    fi
fi

echo "Using PYTHON_BIN=$PYTHON_BIN"

"$PYTHON_BIN" -c "import sys; major, minor = sys.version_info[:2]; assert (major == 3 and minor <= 11), f'Python {major}.{minor} is not supported for TinyZero RL debug. Please use Python 3.10 or 3.11 because ray wheels are unavailable for this version.'" || exit 1

"$PYTHON_BIN" -c "import numpy, hydra, ray, transformers" || {
    echo "Missing dependencies in current Python env."
    echo "Please run: $PYTHON_BIN -m pip install -e . && $PYTHON_BIN -m pip install peft bitsandbytes"
    exit 1
}

if echo "$MODEL_PATH" | grep -qi "qwen3"; then
    "$PYTHON_BIN" -c "from packaging import version; import transformers as t; assert version.parse(t.__version__) >= version.parse('4.51.0'), f'Qwen3 requires newer transformers. Current: {t.__version__}'" || {
        echo "Qwen3 upgrade mode requires transformers>=4.51.0 and peft>=0.13."
        echo "Run: bash examples/grpo_trainer/setup_qwen3_upgrade_env.sh"
        exit 1
    }
fi

"$PYTHON_BIN" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
    data.max_prompt_length=256 \
    data.max_response_length=256 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    +actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.lora.rank=16 \
    actor_rollout_ref.model.lora.alpha=32 \
    actor_rollout_ref.model.lora.dropout=0.05 \
    actor_rollout_ref.model.lora.target_modules=[q_proj,k_proj,v_proj,o_proj] \
    actor_rollout_ref.model.lora.use_qlora=True \
    actor_rollout_ref.model.lora.bnb_4bit_quant_type=nf4 \
    actor_rollout_ref.model.lora.bnb_4bit_compute_dtype=bfloat16 \
    actor_rollout_ref.model.lora.bnb_4bit_use_double_quant=True \
    actor_rollout_ref.actor.optim.lr=2e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.total_epochs=1 $@
