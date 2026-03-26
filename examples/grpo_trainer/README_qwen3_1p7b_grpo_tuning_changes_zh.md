# Qwen3-1.7B GRPO 调参与代码修改说明（2026-03-26）

本文档记录本次为“提速 + 训练有效性修复”所做的脚本与代码改动，便于复现、审计与回滚。

## 1. 修改目标

- 在显存约束下提升训练吞吐，并缩短总时长。
- 修复训练中出现的保存报错与批次不一致问题。
- 提升奖励有效密度，缓解长期 `No answer found` 导致的弱学习信号。
- 保持可视化可追踪（W&B）。

## 2. 修改文件总览

- examples/grpo_trainer/run_qwen3-1.7b_2gpu.sh
- verl/workers/fsdp_workers.py
- verl/trainer/ppo/ray_trainer.py
- verl/utils/reward_score/multiply.py

## 3. 启动脚本改动

文件：examples/grpo_trainer/run_qwen3-1.7b_2gpu.sh

### 3.1 新增/调整参数

- 默认关闭梯度检查点：`ENABLE_GRADIENT_CHECKPOINTING=False`
- 默认增大微批参数：
  - `PPO_MICRO_BATCH_SIZE=4`
  - `LOG_PROB_MICRO_BATCH_SIZE=8`
- 新增 rollout 吞吐参数：
  - `ROLLOUT_MAX_NUM_BATCHED_TOKENS`
  - `ROLLOUT_MAX_NUM_SEQS`
- 新增训练/保存控制参数：
  - `SAVE_FREQ`
  - `TOTAL_EPOCHS`
  - `TOTAL_TRAINING_STEPS`
  - `SAVE_ONLY_LAST`
- 新增远端上传控制：`DISABLE_REMOTE_CKPT_UPLOAD=True`
- 新增结束验证控制：`SKIP_FINAL_VALIDATION`

### 3.2 可靠性保护

- 增加批次合法性检查：
  - `PPO_MINI_BATCH_SIZE % PPO_MICRO_BATCH_SIZE == 0`
  - `LOG_PROB_MICRO_BATCH_SIZE >= N_GPUS_PER_NODE`
- 当 `DISABLE_REMOTE_CKPT_UPLOAD=True` 时，自动注入 `trainer.default_hdfs_dir=null`，避免远端 copy 冲突。

## 4. FSDP 保存兼容修复

文件：verl/workers/fsdp_workers.py

### 4.1 修复内容

在 actor checkpoint 保存前，将 PEFT 配置中的 `target_modules` 从 OmegaConf `ListConfig` 归一化为 Python `list`，避免 `save_pretrained` 时 JSON 序列化报错。

### 4.2 解决问题

- 修复报错：`TypeError: Object of type ListConfig is not JSON serializable`

## 5. PPO 训练器兼容修复

文件：verl/trainer/ppo/ray_trainer.py

### 5.1 新增结束验证开关

- 支持 `+trainer.skip_final_validation=True`。
- 在训练达成总步数后可跳过 final validate，减少额外开销并规避高负载阶段 OOM 风险。

### 5.2 分组 rollout 兼容修复

- 在 `rollout.n > 1` 时，跳过 `_balance_batch`，避免重排破坏分组结构。
- 在 `batch.union(gen_batch_output)` 前，若批次不匹配且满足重复关系，按 `rollout.n` 自动扩展 `gen_batch_output`。
- 在 reward 写回前，若 reward batch 与训练 batch 不一致且满足重复关系，按 `rollout.n` 自动扩展 reward tensor。

### 5.3 解决问题

- 修复报错：`AssertionError: Two tensor dict must have identical batch size. Got torch.Size([40]) and torch.Size([20])`

## 6. 奖励提取规则修复

文件：verl/utils/reward_score/multiply.py

### 6.1 修复内容

- 保留优先从 `<answer>...</answer>` 提取答案。
- 放宽解析：允许标签内有杂字符，提取其中最后一个整数。
- 增加回退策略：若无合法 `<answer>`，从输出尾部窗口提取最后一个整数。

### 6.2 预期收益

- 降低 `No answer found` 比例。
- 提升非零奖励密度，改善 `pg_loss` 与 `clipfrac` 长期贴近 0 的无效学习现象。

## 7. 当前建议运行参数（fix5）

> 说明：以下为本次用于“提速 + 纠偏”的稳定组合。

- `TRAIN_BATCH_SIZE=20`
- `MAX_PROMPT_LENGTH=160`
- `MAX_RESPONSE_LENGTH=64`
- `PPO_MINI_BATCH_SIZE=20`
- `PPO_MICRO_BATCH_SIZE=4`
- `LOG_PROB_MICRO_BATCH_SIZE=10`
- `ROLLOUT_N=1`
- `actor_rollout_ref.actor.optim.lr=5e-5`
- `actor_rollout_ref.actor.ppo_epochs=4`
- `actor_rollout_ref.actor.kl_loss_coef=0.002`
- `algorithm.kl_ctrl.kl_coef=0.002`
- `DISABLE_REMOTE_CKPT_UPLOAD=True`
- `SKIP_FINAL_VALIDATION=True`

## 8. 回滚指南

如需回滚为最初版本，可按以下顺序操作：

1. 回滚启动脚本：examples/grpo_trainer/run_qwen3-1.7b_2gpu.sh
2. 回滚训练器：verl/trainer/ppo/ray_trainer.py
3. 回滚 FSDP 保存兼容：verl/workers/fsdp_workers.py
4. 回滚奖励提取器：verl/utils/reward_score/multiply.py

建议使用版本控制工具按文件级别回滚，并保留本说明文档用于对照。

## 9. 备注

- 本次改动以“先确保可跑、再提升学习有效性”为原则。
- 若后续要严格发挥 GRPO 多采样优势，建议进一步评估并切换到真正支持多独立采样的 rollout 路径。

## 10. 持续更新日志（追加写入）

> 约定：从本条开始，每次调参/改码/重启都会在本节追加一条记录，不覆盖历史。

### 2026-03-26 变更记录 A

- 新增文档：[examples/grpo_trainer/README_qwen3_1p7b_grpo_tuning_changes_zh.md](examples/grpo_trainer/README_qwen3_1p7b_grpo_tuning_changes_zh.md)
- 记录了脚本参数、训练器修复、奖励提取修复、回滚路径。

### 2026-03-26 变更记录 B

- 为落实“真正实现 GRPO 多采样”，在 [verl/workers/rollout/hf_rollout.py](verl/workers/rollout/hf_rollout.py) 中实现了 HF 路径独立多采样：
  - 生成时使用 `num_return_sequences=n`
  - 支持 `prompts.meta_info['n']` / `config.n`
  - 对 `attention_mask/position_ids` 做按 `n` 的对齐扩展
  - 修正 `sequence_length` 使用运行时 `response_length`
- 预期效果：`rollout_n=2~4` 时输出批次真实放大为 `B*n`，不再是单样本伪重复。

### 2026-03-26 变更记录 C（计划执行）

- 按用户要求将学习率提高到 `8e-5`，并观察 `actor/pg_clipfrac` 是否抬升到 `0.02 ~ 0.05`。
- 在“独立多采样已生效”的前提下，将 `rollout_n` 从 1 提回 2 起步，稳定后再评估 4。

### 2026-03-26 变更记录 D（已执行）

- 启动训练（W&B 可视化开启）并采用以下参数：
  - `EXPERIMENT_NAME=grpo_qlora_2gpu_ddp_fix6`
  - `ROLLOUT_N=2`
  - `actor_rollout_ref.actor.optim.lr=8e-5`
  - `actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.01`
  - `actor_rollout_ref.actor.ppo_epochs=4`
  - `actor_rollout_ref.actor.kl_loss_coef=0.002`
  - `algorithm.kl_ctrl.kl_coef=0.002`
  - 其余沿用 fix5 的稳定提速配置（`TRAIN_BATCH_SIZE=20`, `MAX_PROMPT_LENGTH=160`, `MAX_RESPONSE_LENGTH=64` 等）。
- 本次目标：
  - 观察 `actor/pg_clipfrac` 是否提升至 `0.02 ~ 0.05`。
  - 验证多采样下 `actor/ppo_kl`、`actor/pg_loss` 是否较之前更活跃。

### 2026-03-26 变更记录 E（已执行）

- 处理启动阻塞：系统盘 100% 导致 `tempfile` 和 Hydra 输出目录创建失败。
- 修复方式：训练运行时重定向到数据盘 `/root/autodl-tmp`：
  - `TMPDIR/TMP/TEMP=/root/autodl-tmp/tmp`
  - `WANDB_DIR=/root/autodl-tmp/wandb`
  - `trainer.default_local_dir=/root/autodl-tmp/checkpoints/...`
  - `hydra.run.dir=/root/autodl-tmp/outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}`
- 结果：`fix6` 训练已成功启动并持续产生日志（`epoch/step` 正常递增）。

### 2026-03-26 变更记录 F（已执行）

- 针对“`pg_loss/clipfrac/ppo_kl` 长期近零”的根因修复，新增三类改动：
  - 在 [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py) 禁止伪多采样：当 `rollout.n>1` 且生成批次不匹配时，不再 `repeat` 生成结果补齐，而是直接报错终止。
  - 在 [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py) 增加单步诊断：可打印每个样本的 `prompt/response/reward/advantage`，并输出按 `uid` 分组的奖励方差，用于确认 GRPO 组内是否有有效区分度。
  - 在 [verl/trainer/ppo/core_algos.py](verl/trainer/ppo/core_algos.py) 修复 GRPO 组内统计实现：`std` 改为 `torch.stack(...).std(unbiased=False)`，避免不规范张量构造造成统计异常。
- 指标可观测性增强：在 [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py) 新增 `critic/rewards/std` 与 `critic/advantages/std`。
- 启动脚本恢复学习配置：在 [examples/grpo_trainer/run_qwen3-1.7b_2gpu.sh](examples/grpo_trainer/run_qwen3-1.7b_2gpu.sh) 默认改为
  - `TRAIN_BATCH_SIZE=8`
  - `ROLLOUT_N=4`
  - `ACTOR_LR=6e-5`
  - `KL_CTRL_TYPE=adaptive`
  - `KL_CTRL_TARGET=0.03`
  - `KL_CTRL_COEF=0.001`
  - `ACTOR_GRAD_CLIP=1.0`
  - 并支持 `DEBUG_REWARD_ADV=True` 触发单步诊断打印。

### 2026-03-26 变更记录 G（已执行）

- 新增问题归档（针对 `pg_clipfrac` 长期低于 0.01）：
  - 现象：`actor/pg_clipfrac` 长期近 0，`actor/ppo_kl` 多数在极低区间，`actor/pg_loss` 在 0 附近振荡。
  - 定位：优势信号间歇性塌缩（部分 step 出现 `critic/advantages/max=min=0`），同时错误样本奖励过于离散且同质（大量样本落在相同低档位），导致组内方差不足。
  - 影响：ratio 难以远离 1，PPO 裁剪触发概率低，策略更新停滞。
- 奖励修复：在 [verl/utils/reward_score/multiply.py](verl/utils/reward_score/multiply.py) 为错误答案引入“相对误差稠密奖励”。
  - 之前：错误样本固定返回 `format_score`（同质化严重）。
  - 现在：错误样本奖励为 `format_score / (1 + rel_error)`，仍限制在 `[0, format_score]`，但可产生连续差异。
  - 目标：提升同组样本 reward 方差，缓解 GRPO 优势接近 0 导致的 `pg_clipfrac` 低迷。
- 训练重启计划：使用 fix7 参数组重新启动并观察前 100 步。
  - `ROLLOUT_N=4`
  - `TRAIN_BATCH_SIZE=8`
  - `ACTOR_LR=6e-5`
  - `ACTOR_KL_LOSS_COEF=0.001`
  - `algorithm.kl_ctrl.type=adaptive`
  - `algorithm.kl_ctrl.kl_coef=5e-4`
  - `algorithm.kl_ctrl.target_kl=0.03`
  - 打开 `DEBUG_REWARD_ADV=True` 以检查组内方差是否恢复。

### 2026-03-26 变更记录 H（已执行）

- 排查“自适应 KL 不生效”后确认：并非 `algorithm.kl_ctrl` 参数名错误，而是训练路径冲突。
  - 当 `actor_rollout_ref.actor.use_kl_loss=True` 时，训练走 actor 内部 KL loss（见 `actor/kl_coef`），`ray_trainer.apply_kl_penalty` 不执行。
  - `algorithm.kl_ctrl`（fixed/adaptive）只在 `actor.use_kl_loss=False` 的 reward-penalty 路径中生效，对应日志指标是 `critic/kl_coeff`。
- 启动脚本修复：在 [examples/grpo_trainer/run_qwen3-1.7b_2gpu.sh](examples/grpo_trainer/run_qwen3-1.7b_2gpu.sh) 新增并接入
  - `ACTOR_USE_KL_LOSS`（默认 `False`）
  - `ACTOR_KL_LOSS_TYPE`
  - 并将 `actor_rollout_ref.actor.use_kl_loss` 从硬编码 `True` 改为可配置值。
- 修复后的运行准则：
  - 需要自适应 KL：设置 `ACTOR_USE_KL_LOSS=False`，观察 `critic/kl_coeff` 动态变化。
  - 需要固定 actor KL：设置 `ACTOR_USE_KL_LOSS=True`，观察 `actor/kl_coef`（固定值）。

### 2026-03-26 变更记录 I（已执行）

- 复验结论：自适应 KL 已走通，但日志显示存在“精度假象”。
  - 修复后配置中已确认 `actor.use_kl_loss=False` 且 `algorithm.kl_ctrl.type=adaptive`。
  - 训练日志已出现 `critic/kl` 与 `critic/kl_coeff`（而非 `actor/kl_coef`），说明路径正确。
  - 由于本轮将 `KL_CTRL_COEF` 设为 `1e-4`，console 仅保留 3 位小数时会显示为 `0.000`，易误判为不生效。
- 进一步处理：
  - 在 [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py) 增加了 `critic/kl_coeff_x1e6` 指标用于放大观测。
  - 新实验：`grpo_qlora_2gpu_ddp_fix8b_adaptivekl_soft` 已重启并运行。

### 2026-03-26 变更记录 J（已执行）

- 按 5 小时预算重新规划并重启：
  - 首次限步：`TOTAL_TRAINING_STEPS=4000`（估算略有超时风险）。
  - 二次收紧：`TOTAL_TRAINING_STEPS=3400`，新实验名 `grpo_qlora_2gpu_ddp_fix9b_5h_hardcap`。
  - 同时保留：`TRAIN_BATCH_SIZE=8`、`ROLLOUT_N=4`、`ACTOR_USE_KL_LOSS=False`、`adaptive KL`。
- 当前运行状态：
  - 日志确认：`Total training steps: 3400`，并已进入 `epoch 0, step ...` 循环。
  - 实时速度（最近 80 步）约 `3.738s/step`。
  - 进度快照：`step 317 / 3400`。
  - 从当前时刻估算剩余时长约 `3.20h`，总时长估算约 `3.53h`，处于 5 小时预算内。