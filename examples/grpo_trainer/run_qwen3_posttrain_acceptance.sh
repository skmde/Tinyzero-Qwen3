#!/usr/bin/env bash
set -euo pipefail

# One-click post-training acceptance for Qwen3 GRPO on GSM8K.
# Steps:
# 1) Optional training-log sanity checks (best reward / recent mean)
# 2) Offline generation on validation parquet
# 3) Strict/Flexible GSM8K accuracy summary + optional threshold assertions

MODEL_PATH=${MODEL_PATH:-}
VAL_FILE=${VAL_FILE:-/root/data/gsm8k/test.parquet}
OUT_PARQUET=${OUT_PARQUET:-/root/autodl-tmp/eval/qwen3_posttrain_eval.parquet}
SUMMARY_JSON=${SUMMARY_JSON:-/root/autodl-tmp/eval/qwen3_posttrain_eval_summary.json}
TRAIN_LOG=${TRAIN_LOG:-}

GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-64}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-256}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-128}
ROLLOUT_NAME=${ROLLOUT_NAME:-hf}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_K=${TOP_K:-0}
TOP_P=${TOP_P:-1.0}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.7}

MIN_BEST_REWARD=${MIN_BEST_REWARD:-0.20}
MIN_RECENT_REWARD=${MIN_RECENT_REWARD:-0.05}
MIN_STRICT_ACC=${MIN_STRICT_ACC:-0.0}
MIN_FLEX_ACC=${MIN_FLEX_ACC:-0.0}

if [ -z "${MODEL_PATH}" ]; then
    echo "MODEL_PATH is required. Example:"
    echo "  MODEL_PATH=/root/autodl-tmp/checkpoints/xxx/actor bash examples/grpo_trainer/run_qwen3_posttrain_acceptance.sh"
    exit 1
fi

if [ ! -d "${MODEL_PATH}" ]; then
    echo "Model directory not found: ${MODEL_PATH}"
    exit 1
fi

if [ ! -f "${VAL_FILE}" ]; then
    echo "Validation parquet not found: ${VAL_FILE}"
    exit 1
fi

if [ -z "${N_GPUS_PER_NODE:-}" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        IFS=',' read -r -a _gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
        N_GPUS_PER_NODE=${#_gpu_ids[@]}
    else
        N_GPUS_PER_NODE=1
    fi
fi

if [ "${N_GPUS_PER_NODE}" -le 0 ]; then
    N_GPUS_PER_NODE=1
fi

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "./.venv311/bin/python" ]; then
        PYTHON_BIN=./.venv311/bin/python
    elif [ -x "./.venv/bin/python" ]; then
        PYTHON_BIN=./.venv/bin/python
    else
        PYTHON_BIN=python3
    fi
fi

mkdir -p "$(dirname "${OUT_PARQUET}")"
mkdir -p "$(dirname "${SUMMARY_JSON}")"

echo "[acceptance] PYTHON_BIN=${PYTHON_BIN}"
echo "[acceptance] MODEL_PATH=${MODEL_PATH}"
echo "[acceptance] VAL_FILE=${VAL_FILE}"
echo "[acceptance] OUT_PARQUET=${OUT_PARQUET}"
echo "[acceptance] SUMMARY_JSON=${SUMMARY_JSON}"
echo "[acceptance] N_GPUS_PER_NODE=${N_GPUS_PER_NODE}"

if [ -n "${TRAIN_LOG}" ]; then
    if [ ! -f "${TRAIN_LOG}" ]; then
        echo "TRAIN_LOG does not exist: ${TRAIN_LOG}"
        exit 1
    fi

    echo "[acceptance] Checking training log metrics from ${TRAIN_LOG}"
    "${PYTHON_BIN}" - <<'PY' "${TRAIN_LOG}" "${MIN_BEST_REWARD}" "${MIN_RECENT_REWARD}"
import re
import statistics
import sys

log_file = sys.argv[1]
min_best = float(sys.argv[2])
min_recent = float(sys.argv[3])

pattern = re.compile(r'critic/rewards/mean:\s*([+-]?[0-9]*\.?[0-9]+)')
vals = []
with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if not line.startswith('step'):
            continue
        m = pattern.search(line)
        if m:
            vals.append(float(m.group(1)))

if not vals:
    raise SystemExit('No critic/rewards/mean found in TRAIN_LOG.')

best = max(vals)
recent = vals[-min(200, len(vals)):]
recent_mean = statistics.fmean(recent)

print(f'[acceptance] log_best_reward={best:.6f}')
print(f'[acceptance] log_recent_reward_mean={recent_mean:.6f}')

if best < min_best:
    raise SystemExit(f'best reward {best:.6f} < MIN_BEST_REWARD {min_best:.6f}')
if recent_mean < min_recent:
    raise SystemExit(f'recent reward mean {recent_mean:.6f} < MIN_RECENT_REWARD {min_recent:.6f}')

print('[acceptance] log check passed')
PY
fi

echo "[acceptance] Running offline generation"
"${PYTHON_BIN}" -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    +actor.strategy=ddp \
    +actor.ulysses_sequence_parallel_size=1 \
    data.path="${VAL_FILE}" \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.batch_size="${GEN_BATCH_SIZE}" \
    data.output_path="${OUT_PARQUET}" \
    model.path="${MODEL_PATH}" \
    +model.trust_remote_code=True \
    rollout.name="${ROLLOUT_NAME}" \
    rollout.temperature="${TEMPERATURE}" \
    rollout.top_k="${TOP_K}" \
    rollout.top_p="${TOP_P}" \
    rollout.prompt_length="${MAX_PROMPT_LENGTH}" \
    rollout.response_length="${MAX_RESPONSE_LENGTH}" \
    +rollout.n=1 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}"

echo "[acceptance] Computing GSM8K strict/flexible accuracy"
"${PYTHON_BIN}" - <<'PY' "${OUT_PARQUET}" "${SUMMARY_JSON}" "${MIN_STRICT_ACC}" "${MIN_FLEX_ACC}"
import json
import math
import sys

import pandas as pd

from verl.utils.reward_score.gsm8k import extract_solution

out_parquet = sys.argv[1]
summary_json = sys.argv[2]
min_strict_acc = float(sys.argv[3])
min_flex_acc = float(sys.argv[4])

df = pd.read_parquet(out_parquet)
required_cols = ['responses', 'reward_model']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise SystemExit(f'Missing columns in output parquet: {missing}')

strict_ok = 0
flex_ok = 0
format_ok = 0
total = 0

for _, row in df.iterrows():
    responses = row['responses']
    if isinstance(responses, (list, tuple)) and len(responses) > 0:
        pred_text = responses[0]
    else:
        pred_text = str(responses)

    reward_model = row['reward_model']
    if isinstance(reward_model, dict):
        gt = str(reward_model.get('ground_truth', '')).replace(',', '').replace('$', '').strip()
    else:
        gt = ''

    if not gt:
        continue

    total += 1
    strict_pred = extract_solution(pred_text, method='strict')
    flex_pred = extract_solution(pred_text, method='flexible')

    if strict_pred is not None:
        format_ok += 1
    if strict_pred == gt:
        strict_ok += 1
    if flex_pred == gt:
        flex_ok += 1

if total == 0:
    raise SystemExit('No valid ground_truth found in parquet reward_model column.')

strict_acc = strict_ok / total
flex_acc = flex_ok / total
format_rate = format_ok / total

summary = {
    'num_samples': total,
    'strict_correct': strict_ok,
    'flex_correct': flex_ok,
    'strict_acc': strict_acc,
    'flex_acc': flex_acc,
    'strict_format_hit_rate': format_rate,
}

with open(summary_json, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print('[acceptance] summary:')
for k, v in summary.items():
    if isinstance(v, float):
        print(f'  {k}: {v:.6f}')
    else:
        print(f'  {k}: {v}')

if strict_acc < min_strict_acc:
    raise SystemExit(f'strict_acc {strict_acc:.6f} < MIN_STRICT_ACC {min_strict_acc:.6f}')
if flex_acc < min_flex_acc:
    raise SystemExit(f'flex_acc {flex_acc:.6f} < MIN_FLEX_ACC {min_flex_acc:.6f}')

print('[acceptance] accuracy threshold check passed')
PY

echo "[acceptance] done"