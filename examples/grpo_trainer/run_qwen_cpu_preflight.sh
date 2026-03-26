set -e
set -x

# CPU preflight for local validation (no training).
# This verifies env, model path, tokenizer loading, and parquet readability.

export CUDA_VISIBLE_DEVICES=""

DEFAULT_MODEL_PATH=./Qwen/Qwen3-1___7B-Base
MODEL_PATH=${MODEL_PATH:-$DEFAULT_MODEL_PATH}
TRAIN_FILE=${TRAIN_FILE:-$HOME/data/gsm8k/train.parquet}
VAL_FILE=${VAL_FILE:-$HOME/data/gsm8k/test.parquet}

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
"$PYTHON_BIN" -m examples.grpo_trainer.cpu_preflight \
  --model_path "$MODEL_PATH" \
  --train_file "$TRAIN_FILE" \
  --val_file "$VAL_FILE"
