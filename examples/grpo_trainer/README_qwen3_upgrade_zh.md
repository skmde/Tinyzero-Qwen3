# TinyZero 基于 Qwen3 的升级说明（中文）

本文档总结了在原始 TinyZero 架构基础上，为了使用 Qwen3 作为基座模型进行后训练（GRPO + QLoRA + 分布式）所做的改动，并解释为何实际落地比“少量改动”更复杂。

## 1. 目标与结论

目标是保留 TinyZero 原有训练主干（Ray + FSDP + PPO/GRPO），尽量少改代码，实现：
- 本地 CPU 预检查
- 本地/单卡调试
- 云端多卡正式训练
- Qwen3 模型路径直接可用

结论是：
- 主训练框架没有推翻，仍然是 TinyZero 原架构。
- 但要让 Qwen3 真正可运行，需要补齐若干“生态兼容层”和“工程约束层”。

## 2. 相对原始 TinyZero 的关键更新

### 2.1 训练核心（FSDP Worker）

在 Actor 构建路径中新增了 LoRA/QLoRA 能力，主要在 [verl/workers/fsdp_workers.py](verl/workers/fsdp_workers.py)：
- 增加 LoRA 配置读取（rank、alpha、dropout、target_modules）
- 增加 QLoRA 的 4bit 量化加载（BitsAndBytesConfig）
- 仅优化可训练参数（LoRA 参数）
- LoRA 场景启用 FSDP use_orig_params 兼容冻结参数训练
- 对缺失 peft 的情况给出明确报错

### 2.2 配置层（Hydra YAML）

在 [verl/trainer/config/ppo_trainer.yaml](verl/trainer/config/ppo_trainer.yaml) 下新增了 actor_rollout_ref.model.lora 配置项：
- rank
- alpha
- dropout
- target_modules
- use_qlora
- bnb_4bit_quant_type
- bnb_4bit_compute_dtype
- bnb_4bit_use_double_quant

这使得 QLoRA 不再是“硬编码分支”，而是可配置能力。

### 2.3 脚本层（本地/云端）

新增或更新了以下脚本：
- [examples/grpo_trainer/run_qwen3-1.5b_qlora_local_debug.sh](examples/grpo_trainer/run_qwen3-1.5b_qlora_local_debug.sh)
- [examples/grpo_trainer/run_qwen3-1.5b_qlora_autodl.sh](examples/grpo_trainer/run_qwen3-1.5b_qlora_autodl.sh)
- [examples/grpo_trainer/setup_qwen3_upgrade_env.sh](examples/grpo_trainer/setup_qwen3_upgrade_env.sh)
- [examples/grpo_trainer/run_qwen_cpu_preflight.sh](examples/grpo_trainer/run_qwen_cpu_preflight.sh)

脚本层新增了这些关键保护：
- 优先使用 .venv311 解释器
- 运行前做 transformers 版本检查（Qwen3 升级模式）
- 自动优先本地模型目录 ./Qwen/Qwen3-1___7B-Base
- Qwen3 模式设置 trust_remote_code=True
- Qwen3 模式默认关闭 use_remove_padding

### 2.4 CPU 预检查链路

新增 CPU 预检查工具：
- [examples/grpo_trainer/cpu_preflight.py](examples/grpo_trainer/cpu_preflight.py)

用途：
- 在无 GPU 本地先验证依赖、模型目录、tokenizer、parquet 数据可读性
- 避免把问题堆到云端调试

## 3. 为什么会比“理论上少量改动”更麻烦

从框架视角看，TinyZero 的确是轻量后训练框架；但“换基座模型”不只是一行 model.path，主要复杂性来自以下层面：

### 3.1 生态版本耦合

原始 TinyZero 默认依赖与 Qwen3 生态并非天然对齐：
- 旧版 transformers 对 qwen3 model_type 识别有限
- 训练脚本与第三方组件（peft、bitsandbytes、ray、torch）存在版本联动

因此需要引入升级环境脚本，而不是只改模型路径。

### 3.2 模型特性与框架特性不完全同构

TinyZero 原生优化路径包含 remove padding、Ulysses、特定 monkey patch，这些实现主要围绕 llama/qwen2：
- Qwen3 直接复用时，部分优化路径未必稳定
- 为了先跑通，需关闭或降级某些特性（例如 use_remove_padding=False）

### 3.3 RL 训练不是单模型推理

一个 RL 训练回合涉及 actor、ref、rollout、critic 等多角色协同，任何一个环节不兼容都会失败：
- 模型加载兼容
- 分布式资源调度
- 量化与梯度更新策略
- 数据流和日志链路

所以“改少量代码”在多角色系统里会被放大。

### 3.4 环境现实约束

在本地 Windows/WSL/CPU 场景下还会叠加：
- Python 版本与 Ray wheel 可用性
- CPU 版 torch 导致 GPU 资源为 0
- 代理/网络/权限对模型拉取的影响

这些不是 TinyZero 架构问题，但会直接影响“能不能跑起来”。

## 4. 这次改动是否偏离 TinyZero 的轻量思想

整体没有偏离。核心训练入口仍是 TinyZero 原路径：
- 训练入口 [verl/trainer/main_ppo.py](verl/trainer/main_ppo.py)
- worker 协作仍是 Ray + FSDP
- 只是补齐了 Qwen3 生态的适配层与工程护栏

换句话说，这次不是重写框架，而是把“模型切换的隐性成本”显式化、脚本化。

## 5. 推荐实践（先本地后云端）

建议流程：
1. 本地 CPU 跑 preflight（仅检查，不训练）
2. 云端执行 Qwen3 升级环境脚本
3. 云端单卡小步数冒烟
4. 云端多卡正式训练

对应文档可参考：
- [examples/grpo_trainer/README_qwen3_1p5b_qlora_autodl.md](examples/grpo_trainer/README_qwen3_1p5b_qlora_autodl.md)
