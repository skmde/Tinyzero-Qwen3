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