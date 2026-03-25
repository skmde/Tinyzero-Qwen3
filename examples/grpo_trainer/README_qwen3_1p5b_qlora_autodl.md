# Qwen3-1.5B GRPO + QLoRA on AutoDL

This guide provides a practical TinyZero training setup for reinforcement post-training on Qwen3-1.5B with multi-GPU distributed training and QLoRA.

Chinese summary and architecture-delta note:
- [examples/grpo_trainer/README_qwen3_upgrade_zh.md](examples/grpo_trainer/README_qwen3_upgrade_zh.md)

## Qwen3 Upgrade Mode

Qwen3 needs a newer Transformers stack than TinyZero default constraints.
To keep Qwen3, run this first:

```bash
bash examples/grpo_trainer/setup_qwen3_upgrade_env.sh
```

The Qwen3 scripts include a runtime version check (`transformers>=4.51.0`) and will stop early with guidance if your env is outdated.

If your local machine has no GPU, use CPU preflight first (no training, only environment/model/data checks):

```bash
bash examples/grpo_trainer/run_qwen_cpu_preflight.sh
```

## 1) Environment

Recommended AutoDL image:
- CUDA 12.1+
- Python 3.10 or 3.11
- PyTorch 2.1+

Important for local Windows/Git-Bash debug:
- Use Python 3.10/3.11 virtual environment.
- Ray wheels are not available for Python 3.13 in this setup, so training entrypoint will fail before launch.

Install TinyZero and required extra packages:

```bash
pip install -e .
pip install peft bitsandbytes
```

If you are using Git Bash on Windows, force the script to use project venv Python:

```bash
PYTHON_BIN=./.venv/Scripts/python.exe bash examples/grpo_trainer/run_qwen3-1.5b_qlora_local_debug.sh
```

## 2) Prepare data

By default the script reads:
- `$HOME/data/gsm8k/train.parquet`
- `$HOME/data/gsm8k/test.parquet`

You can build these files using existing TinyZero data preprocess scripts in examples/data_preprocess.

## 3) Launch training

Local single-GPU smoke test (recommended first):

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=./Qwen/Qwen3-1___7B-Base \
PYTHON_BIN=./.venv311/Scripts/python.exe \
bash examples/grpo_trainer/run_qwen3-1.5b_qlora_local_debug.sh
```

If the local model folder exists at `./Qwen/Qwen3-1___7B-Base`, both scripts automatically use it by default.

If this passes, switch to AutoDL multi-GPU training:

Single node, 4 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_PATH=Qwen/Qwen3-1.5B \
PYTHON_BIN=./.venv311/Scripts/python.exe \
bash examples/grpo_trainer/run_qwen3-1.5b_qlora_autodl.sh
```

Multi-node:

```bash
NNODES=2 N_GPUS_PER_NODE=8 \
MODEL_PATH=Qwen/Qwen3-1.5B \
PYTHON_BIN=./.venv311/Scripts/python.exe \
bash examples/grpo_trainer/run_qwen3-1.5b_qlora_autodl.sh
```

## 4) Key design choices

- Uses `algorithm.adv_estimator=grpo` to avoid critic training overhead and reduce memory pressure.
- Uses QLoRA on actor only (`actor_rollout_ref.model.lora.*`) for efficient RL updates.
- Uses HF rollout for compatibility with adapter-based training.
- Keeps reference policy in FSDP with parameter offload for lower GPU memory use.
- Note: actual RL training requires GPU; CPU mode here is preflight only.
- Qwen3 mode sets `trust_remote_code=True` and disables `use_remove_padding` for compatibility.

## 5) Common knobs

- Model: `MODEL_PATH`
- Training files: `TRAIN_FILE`, `VAL_FILE`
- Scale: `NNODES`, `N_GPUS_PER_NODE`, `CUDA_VISIBLE_DEVICES`
- Logging: override `trainer.logger` from command line if needed
- Fast local debug script: `examples/grpo_trainer/run_qwen3-1.5b_qlora_local_debug.sh`

## 6) Output

Checkpoints and logs are controlled by TinyZero trainer config:
- `trainer.save_freq`
- `trainer.project_name`
- `trainer.experiment_name`

Use command-line overrides to adjust these values for your AutoDL workspace policy.
