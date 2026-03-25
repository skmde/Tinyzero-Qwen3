set -e
set -x

# Build a Qwen3-capable environment for TinyZero RL scripts.
# This script intentionally installs verl with --no-deps, then installs a curated dependency set.

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
"$PYTHON_BIN" -m pip install -U pip setuptools wheel
"$PYTHON_BIN" -m pip install -e . --no-deps

# Core deps
"$PYTHON_BIN" -m pip install \
    accelerate \
    codetiming \
    datasets \
    dill \
    hydra-core \
    numpy \
    pandas \
    pybind11 \
    ray \
    tensordict \
    wandb

# Qwen3-upgrade deps
"$PYTHON_BIN" -m pip install \
    "transformers>=4.51.0,<4.58.0" \
    "peft>=0.13.0,<0.15.0" \
    bitsandbytes

"$PYTHON_BIN" -c "import transformers, peft, ray, torch; print('transformers', transformers.__version__); print('peft', peft.__version__); print('ray', ray.__version__); print('torch', torch.__version__)"
