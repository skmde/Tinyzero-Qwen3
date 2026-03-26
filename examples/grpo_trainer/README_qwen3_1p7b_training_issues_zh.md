# TinyZero + Qwen3-1.7B 训练问题复盘与改进记录

本文档记录一次在单机单卡 (RTX 4090 24GB) 环境中，使用 TinyZero 对 Qwen3-1.7B 进行 GRPO/LoRA(QLoRA) 训练试跑时遇到的问题、处理过程和结论。

## 1. 环境与目标

- 目标: 从魔搭下载 Qwen3-1.7B 作为基座模型，完成环境配置并启动训练。
- 运行环境:
  - Linux
  - GPU: RTX 4090 24GB
  - Python: 3.11 (项目虚拟环境 .venv311)
- 关键组件:
  - TinyZero
  - Ray
  - Transformers >= 4.51
  - PEFT

## 2. 主要问题与根因

### 2.1 Python 版本不兼容

- 现象: 默认 Python 3.12 下训练入口前置检查失败。
- 根因: 当前 TinyZero 训练链路依赖的 Ray wheel 与脚本约束要求 Python 3.10/3.11。
- 处理: 安装 Python 3.11，创建独立虚拟环境并重新安装依赖。

### 2.2 数据集拉取失败 (HuggingFace 网络不可达)

- 现象: examples/data_preprocess/gsm8k.py 无法从 Hub 拉取 openai/gsm8k。
- 根因: 训练机网络策略限制，huggingface.co 连接超时/拒绝。
- 处理:
  - 使用仓库内测试 parquet 做临时联调。
  - 后续生成本地最小可用 parquet 数据，保证 reward schema 完整。

### 2.3 Hydra 配置覆盖失败

- 现象: trust_remote_code 配置报错 Key not in struct。
- 根因: 该字段在基础配置中不存在，Hydra 需要使用新增语法。
- 改进: 将参数改为 +actor_rollout_ref.model.trust_remote_code=True。

### 2.4 flash-attn 依赖链问题

- 现象 1: 训练初始化时报 ModuleNotFoundError: flash_attn。
- 现象 2: 尝试安装 flash-attn 编译失败，报 CUDA 版本不匹配。
- 根因:
  - 代码中存在顶层硬导入 flash_attn。
  - 运行环境中 torch CUDA 与系统 CUDA 组合不适合本地编译 flash-attn。
- 改进:
  - 对 actor/critic 的 flash_attn 导入改为可选导入。
  - 当 use_remove_padding=True 且无 flash_attn 时显式报错。
  - FSDP 模型加载增加 attention 实现回退: 优先 flash_attention_2，不可用时自动回退到 sdpa。

### 2.5 QLoRA 与 FSDP flatten dtype 冲突

- 现象: ValueError: Must flatten tensors with uniform dtype but got torch.float32 and torch.uint8。
- 根因: QLoRA 4bit 参数 (uint8) 与当前 FSDP flatten 约束冲突。
- 处理: 试跑时切换为普通 LoRA (use_qlora=False) 继续验证训练链路。

### 2.6 单卡并行配置冲突

- 现象: rollout world_size: 1 is not divisible by infer_tp: 2。
- 根因: 默认 rollout 张量并行度为 2，不适配单卡。
- 处理: 将 actor_rollout_ref.rollout.tensor_model_parallel_size 调整为 1。

### 2.7 数据 schema 不完整

- 现象: KeyError: reward_model。
- 根因: 使用的临时 parquet 仅包含 prompt，不符合 GRPO 奖励函数所需字段。
- 处理: 生成包含 data_source/prompt/ability/reward_model/extra_info 的本地 parquet。

### 2.8 显存不足 (OOM)

- 现象: 训练可启动且进入 step，但在验证生成或 actor backward 阶段出现 CUDA OOM。
- 根因:
  - Qwen3-1.7B + Ray + FSDP + rollout/generation 在单卡 24GB 边缘运行。
  - 即使降低 batch/长度并开启部分 offload，仍容易在峰值阶段超显存。

## 3. 已落地改进 (代码级)

1) 修复 Hydra 参数新增语法
- 文件: examples/grpo_trainer/run_qwen3-1.5b_qlora_local_debug.sh
- 变更: trust_remote_code 从覆盖改为新增语法 (+key=value)。

2) actor/critic 对 flash_attn 做可选依赖
- 文件: verl/workers/actor/dp_actor.py
- 文件: verl/workers/critic/dp_critic.py
- 变更: try/except 导入；仅在 use_remove_padding=True 时强依赖 flash_attn。

3) 模型注意力实现增加回退
- 文件: verl/workers/fsdp_workers.py
- 变更: 若 flash_attn 不可用，attn_implementation 自动回退到 sdpa。

## 4. 试跑结果结论

- 已完成:
  - 魔搭模型下载
  - 环境搭建
  - 训练链路打通 (可进入训练 step 并输出指标)
- 未完全完成:
  - 在单卡 24GB 上稳定跑完整 epoch 仍有较高 OOM 风险。

## 5. 建议的后续优化路线

### 5.1 资源侧

- 首选 2 卡及以上运行 Qwen3-1.7B GRPO。
- 或改用更小模型进行单卡验证，再迁移到 1.7B。

### 5.2 配置侧

- 单卡优先使用 LoRA 而非 QLoRA+FSDP 组合。
- 固定 tensor_model_parallel_size=1。
- 进一步降低 max_prompt_length / max_response_length / rollout.n。
- 降低验证频率，避免频繁生成导致显存尖峰。

### 5.3 工程侧

- 提供专用单卡低显存脚本 (1.7B-safe profile)；默认保守参数与 offload。
- 为数据预处理增加离线模式或本地数据回退逻辑，避免外网依赖导致流程中断。

## 6. 一键排查清单

- Python 是否为 3.10/3.11。
- transformers/peft 版本是否满足 Qwen3 要求。
- MODEL_PATH 是否可读且包含完整权重。
- 数据 parquet 是否包含 reward_model.ground_truth 等必需字段。
- 单卡时 tensor_model_parallel_size 是否设置为 1。
- 无 flash_attn 时是否已关闭 remove_padding 路径并启用 sdpa 回退。

## 7. Python 3.12 + CUDA 12.8 双卡可行性

- 可行，但建议按“兼容优先”方式运行：
  - Python: 3.12
  - CUDA Driver: 12.8
  - PyTorch: 安装官方预编译 cu12x wheel（无需本地 CUDA toolkit 与驱动版本完全一致）
- 本仓库已将本地调试脚本的 Python 版本检查放宽到 3.10-3.12。
- 双卡脚本默认 `CUDA_VISIBLE_DEVICES=0,1`，并自动按该变量推导 `N_GPUS_PER_NODE`。
- 不建议在该环境优先处理 flash-attn 本地编译；优先使用 `use_remove_padding=False` + `sdpa` 回退链路保证可跑通。

---

本次改进目标是先确保链路可启动并可执行训练步骤，再逐步提升稳定性与吞吐。

## 8. 训练完成后一键验收

新增脚本: examples/grpo_trainer/run_qwen3_posttrain_acceptance.sh

作用:
- 可选检查训练日志中的 reward 门槛。
- 对验证集做离线生成。
- 自动统计 GSM8K strict/flexible 准确率并输出 summary JSON。

最小用法:

```bash
cd /root/Tinyzero-Qwen3
MODEL_PATH=/root/autodl-tmp/checkpoints/tinyzero_qwen3_1p7b/your_exp/actor \
bash examples/grpo_trainer/run_qwen3_posttrain_acceptance.sh
```

带日志门槛断言:

```bash
cd /root/Tinyzero-Qwen3
MODEL_PATH=/root/autodl-tmp/checkpoints/tinyzero_qwen3_1p7b/your_exp/actor \
TRAIN_LOG=/root/autodl-tmp/logs/your_exp.log \
MIN_BEST_REWARD=0.20 \
MIN_RECENT_REWARD=0.05 \
MIN_STRICT_ACC=0.20 \
MIN_FLEX_ACC=0.30 \
bash examples/grpo_trainer/run_qwen3_posttrain_acceptance.sh
```

关键输出:
- 生成结果 parquet: OUT_PARQUET (默认 /root/autodl-tmp/eval/qwen3_posttrain_eval.parquet)
- 验收汇总 JSON: SUMMARY_JSON (默认 /root/autodl-tmp/eval/qwen3_posttrain_eval_summary.json)

## 9. 奖励与策略指标问题复盘（fix9b 阶段）

### 9.1 正向信号

- critic/score/max 与 critic/rewards/max 可稳定到 1.0，说明策略已具备“可达正确解”的能力。
- 奖励区分有效: 正确样本可获得显著高奖励，错误样本为低分或负分。
- actor 基础训练链路稳定:
  - actor/lr 调度正常并稳定。
  - actor/grad_norm 在安全区间，无爆炸或消失。
  - actor/entropy_loss 未塌陷，探索性仍在。
  - actor/ppo_kl 受控，无策略崩溃迹象。

### 9.2 关键异常与优先级

1) 平均奖励长期停滞
- 现象: critic/score/mean 与 critic/rewards/mean 在中等区间大幅震荡，未形成持续上升趋势。
- 影响: 说明“偶发答对”未转化为“稳定高概率答对”。

2) 奖励两极化加重
- 现象: critic/score/min 长期贴近低位，critic/rewards/min 持续下探。
- 影响: 组内优势方差过大，更新方向容易互相抵消。

3) 策略更新幅度不足
- 现象: actor/pg_clipfrac 长期处于低位（明显低于健康区间）。
- 影响: 正确样本概率提升过慢，难以摆脱平台期。

4) 策略损失方向不稳定
- 现象: actor/pg_loss 大幅震荡，缺少稳定收敛段。
- 影响: 优势信号噪声偏高，优化目标在步间频繁变化。

### 9.3 根因判断

- 主要矛盾不是“训练链路坏掉”，而是“更新强度不足 + 奖励方差偏大”的叠加。
- 终点型奖励对“错 -> 半对 -> 全对”的中间路径引导仍不足，导致学习依赖随机探索。

### 9.4 改进策略（按优先级落地）

1) 先提高有效更新强度
- actor lr: 6e-5 -> 8e-5（必要时到 1e-4）
- actor ppo_epochs: 4 -> 6
- 目标: actor/pg_clipfrac 中位数提升到 0.05~0.15 区间。

2) 再降低优势噪声
- 保持或提升 rollout_n（当前配置已使用 4），优先保证组内比较样本足够。
- 保持 batch 兼容与显存稳定，避免因微批不匹配引入额外波动。

3) 最后细化奖励成形
- 保留“全对高奖励”主目标。
- 为“接近正确”的错误样本保留连续梯度，限制极端负奖励主导训练。

### 9.5 执行记录（fix10 开始）

- 实验名: grpo_qlora_2gpu_ddp_fix10_clipboost
- 启动策略:
  - 继承 fix9b 的双卡 DDP+QLoRA 稳定配置。
  - 上调 actor 学习率与 ppo_epochs，优先解决 clipfrac 长期低迷问题。
  - 继续使用自适应 KL，维持稳定边界。
- 预期观察窗口:
  - 前 300~500 step 重点关注:
    - actor/pg_clipfrac 是否明显抬升。
    - critic/rewards/mean 滑动均值是否转为正斜率。
    - actor/ppo_kl 是否维持可控范围。

## 10. fix11 方案：奖励重构 + 放宽 KL + 多采样

### 10.1 问题导向

- 历史问题: 奖励两极分化、优势函数噪声高、pg_clipfrac 长期低位。
- 目标: 提高奖励连续性与可学习性，释放策略更新空间，让 clipfrac 进入健康区间。

### 10.2 已落地改动

1) 重构奖励函数（渐进式步骤分 + 非负稠密奖励）
- 文件: verl/utils/reward_score/multiply.py
- 改动:
  - 新增步骤分: 根据 `<think>/<answer>` 结构与算式痕迹给出 step bonus。
  - 新增接近度分: 相对误差 closeness + 数位一致性 digit match。
  - 错误样本保持非负且稠密，不再引入负向奖励。
  - 正确样本仍保持满分 1.0，维持目标清晰边界。

2) 取消 KL 后负奖励（奖励下界）
- 文件: verl/trainer/ppo/ray_trainer.py
- 改动:
  - `apply_kl_penalty` 增加 `reward_floor`。
  - `token_level_rewards` 在 KL 扣分后执行下界裁剪（默认可设为 0.0）。

3) 放宽自适应 KL + 提高多采样
- 文件: examples/grpo_trainer/run_qwen3-1.7b_2gpu.sh
- 默认参数调整:
  - `ROLLOUT_N=8`
  - `KL_CTRL_COEF=5e-5`
  - `KL_CTRL_TARGET=0.20`
  - `KL_CTRL_HORIZON=1000`
  - `REWARD_FLOOR=0.0`（通过 `+algorithm.reward_floor` 注入配置）

### 10.3 执行记录（fix11 已启动）

- 实验名: `grpo_qlora_2gpu_ddp_fix11_reward_kl_rollout8`
- 日志: `/root/autodl-tmp/logs/fix11_reward_kl_rollout8.log`
- checkpoint: `/root/autodl-tmp/checkpoints/tinyzero_qwen3_1p7b/grpo_qlora_2gpu_ddp_fix11_reward_kl_rollout8`
- W&B run: `hvq8tla9`
- 状态: 已进入训练 step 并持续运行。