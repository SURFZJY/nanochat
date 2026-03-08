# 第四章：强化学习 (RL)

本章学习 nanochat 如何使用强化学习进一步提升 SFT 模型的数学推理能力。

---

## 目录

1. [RL 概述：为什么需要 RL？](#1-rl-概述为什么需要-rl)
2. [GRPO 简化版：本质是 REINFORCE](#2-grpo-简化版本质是-reinforce)
3. [Rollout 采样循环](#3-rollout-采样循环)
4. [策略梯度训练](#4-策略梯度训练)
5. [奖励函数](#5-奖励函数)
6. [Pass@k 评估](#6-passk-评估)
7. [学习率与训练配置](#7-学习率与训练配置)
8. [关键设计理念总结](#8-关键设计理念总结)

---

## 1. RL 概述：为什么需要 RL？

文件位置：`scripts/chat_rl.py`

SFT 教模型"模仿"正确答案，但 RL 让模型"探索"并从结果学习。

```
SFT 阶段                           RL 阶段
─────────                          ─────────
给定问题和标准答案                   给定问题，让模型自己生成多个回答
训练模型复制标准答案                 用奖励函数评判回答对错
学习 "怎么做"                       学习 "什么更好"
```

nanochat 的 RL 专注于 **GSM8K 数学任务**，通过 GRPO（Group Relative Policy Optimization）的简化版来训练。

### 启动命令

```bash
# 单 GPU
python -m scripts.chat_rl

# 8 GPU 分布式
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=default
```

---

## 2. GRPO 简化版：本质是 REINFORCE

nanochat 对标准 GRPO 做了大幅简化：

```python
"""
I put GRPO in quotes because we actually end up with something a lot
simpler and more similar to just REINFORCE:

1) Delete trust region, so there is no KL regularization to a reference model
2) We are on policy, so there's no need for PPO ratio+clip.
3) We use DAPO style normalization that is token-level, not sequence-level.
4) Instead of z-score normalization (r - mu)/sigma, only use (r - mu) as the advantage.
"""
```

### 简化对比

| 标准 GRPO/PPO | nanochat RL |
|--------------|-------------|
| KL 散度约束参考模型 | **删除**：不需要参考模型 |
| PPO ratio + clip | **删除**：在线策略，无需重要性采样 |
| 序列级归一化 | **Token 级**（DAPO 风格） |
| z-score: (r-μ)/σ | **仅减均值**: r-μ |

最终就是一个简洁的 **REINFORCE + 均值基线**。

### 核心公式

```
Loss = -Σ(log_prob × advantage)

其中：
  log_prob = 模型对每个 token 的对数概率
  advantage = reward - mean(rewards)  （同一问题的多个采样的均值基线）
```

---

## 3. Rollout 采样循环

### 采样流程

对于每个训练问题：

```python
@torch.no_grad()
def get_batch():
    for example_idx in itertools.cycle(rank_indices):
        # 1. 获取完整对话（包含正确答案）
        conversation = train_task[example_idx]

        # 2. 渲染为 completion 模式（去掉助手回答，只保留提示）
        tokens = tokenizer.render_for_completion(conversation)

        # 3. 生成 num_samples 个采样（默认 16 个）
        for sampling_step in range(num_sampling_steps):
            sequences, masks = engine.generate_batch(
                tokens,
                num_samples=device_batch_size,
                max_tokens=256,
                temperature=1.0,  # 使用 temperature=1.0 保证探索
                top_k=50,
            )

        # 4. 评估每个采样的奖励
        for sample_tokens in sequences:
            generated_text = tokenizer.decode(sample_tokens[prefix_length:])
            reward = train_task.reward(conversation, generated_text)

        # 5. 计算优势（简单减均值）
        advantages = rewards - rewards.mean()

        yield sequences, inputs, targets, rewards, advantages
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--examples-per-step` | 16 | 每步训练的题目数 |
| `--num-samples` | 16 | 每题采样数 |
| `--max-new-tokens` | 256 | 最大生成 token 数 |
| `--temperature` | 1.0 | 采样温度（保证探索） |
| `--top-k` | 50 | Top-k 采样 |

每步的总序列数 = `examples_per_step × num_samples` = 16 × 16 = 256 条序列。

---

## 4. 策略梯度训练

```python
for example_step in range(examples_per_rank):
    sequences, inputs, targets, rewards, advantages = next(batch_iterator)

    model.train()
    for pass_idx in range(num_passes):
        # 1. 计算 log 概率（NLL 取负 = log_prob）
        logp = -model(inputs, targets, loss_reduction='none')  # (B, T)

        # 2. 策略梯度目标 = Σ(logp × advantage)
        pg_obj = (logp * advantages.unsqueeze(-1)).sum()

        # 3. 归一化（有效 token 数 × passes 数 × examples 数）
        num_valid = (targets >= 0).sum().clamp(min=1)
        pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)

        # 4. 最小化 loss = -pg_obj（最大化策略目标）
        loss = -pg_obj
        loss.backward()
```

### 为什么不需要 PPO ratio+clip？

```python
# Note, there is no need to add PPO ratio+clip because we are on policy
```

因为 nanochat 是**在线策略**（on-policy）：每次采样都用当前模型，所以 rollout 分布和训练分布相同，不需要重要性采样比率。

### Mask 处理

Engine 返回的 mask 有两层含义：
- `mask=0` for **prompt tokens**（不训练）
- `mask=0` for **tool use forced tokens**（不训练，因为是强制插入的）
- `mask=1` for **模型自由采样的 token**（训练目标）

```python
targets[mask_ids[:, 1:] == 0] = -1  # ignore_index = -1
```

---

## 5. 奖励函数

GSM8K 的奖励函数极其简单——0/1 二值奖励：

```python
class GSM8K(Task):
    def reward(self, conversation, assistant_response):
        is_correct = self.evaluate(conversation, assistant_response)
        return float(is_correct)  # 0.0 或 1.0
```

**答对 = 1.0，答错 = 0.0**。没有中间状态，没有格式奖励。

### 优势计算

```python
mu = rewards.mean()
advantages = rewards - mu
```

如果 16 个采样中 4 个正确（reward=1）、12 个错误（reward=0）：
- 均值 μ = 4/16 = 0.25
- 正确答案的 advantage = 1.0 - 0.25 = +0.75（强化）
- 错误答案的 advantage = 0.0 - 0.25 = -0.25（惩罚）

---

## 6. Pass@k 评估

训练过程中定期用 **pass@k** 评估 GSM8K 测试集：

```python
def run_gsm8k_eval(task, tokenizer, engine, num_samples, max_examples):
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)

        # 生成 k 个采样
        results, masks = engine.generate_batch(
            tokens, num_samples=num_samples, temperature=1.0
        )

        # 检查每个采样
        for sample_tokens in results:
            text = tokenizer.decode(sample_tokens[prefix_length:])
            is_correct = task.evaluate(conversation, text)

    # 计算 pass@k (k=1,2,...,device_batch_size)
    for k in range(1, device_batch_size + 1):
        passk[k-1] = sum(any(correct in first_k) for record in records)
```

**pass@k** 含义：给模型 k 次机会回答，只要有 1 次答对就算通过。k 越大越宽松：
- **pass@1** ≈ 贪心准确率
- **pass@k** (k>1) ≈ 覆盖率（模型有多大概率"能"解出此题）

---

## 7. 学习率与训练配置

### 学习率调度

```python
def get_lr_multiplier(it):
    lrm = 1.0 - it / num_steps  # 简单的线性衰减到 0
    return lrm
```

从初始 LR 线性衰减到 0，没有 warmup。

### 初始学习率

```python
# 默认从基础 LR 的 5% 开始
--init-lr-frac 0.05
```

RL 的学习率远小于 SFT（5% vs 80%），因为策略梯度信号噪声大，需要小步迭代。

### 训练步数

```python
num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
# GSM8K train: ~7500 题 / 16 题每步 ≈ 469 步/epoch
```

### DDP 分布

```python
# 每个 rank 处理不同的训练题目
rank_indices = range(ddp_rank, len(train_task), ddp_world_size)
# 每个 rank 每步处理的题目数
examples_per_rank = examples_per_step // ddp_world_size
```

---

## 8. 关键设计理念总结

### 8.1 极简 RL

整个 RL 实现只有 ~330 行，没有复杂的 PPO 基础设施：
- 无参考模型 → 节省 50% 内存
- 无 KL 约束 → 代码更简单
- 无重要性采样 → 无 ratio clipping
- 简单均值基线 → 不做 z-score 标准化

### 8.2 在线策略 (On-Policy)

每次训练都用当前模型采样，保证 rollout 分布与策略分布一致。代价是需要频繁生成（GPU 利用率不如离线方法），但算法更简单可靠。

### 8.3 二值奖励

只看最终答案对不对，不奖励中间步骤。这在 GSM8K 这种有明确正确答案的任务上效果够好，且避免了奖励黑客（reward hacking）。

### 8.4 Checkpoint 管理

```python
# RL 不保存优化器状态（因为 RL 通常不需要恢复训练）
save_checkpoint(checkpoint_dir, step, model.state_dict(),
                None,  # optimizer_data = None
                {"model_config": model_config_kwargs})
```

### 8.5 工具调用的 RL 训练

Engine 在 RL 生成过程中自动处理工具调用：
1. 模型自由采样文本
2. 遇到 `<|python_start|>` → 进入工具模式
3. 遇到 `<|python_end|>` → 执行表达式
4. 强制注入 `<|output_start|>结果<|output_end|>`
5. 模型继续自由采样

强制注入的 token（mask=0）不参与 RL 训练。

---

## 动手实验建议

```bash
# 1. 查看 RL 的采样过程
python -c "
from tasks.gsm8k import GSM8K
task = GSM8K(subset='main', split='train')
conv = task[0]
print('Question:', conv['messages'][0]['content'])
print('Answer:', conv['messages'][1]['content'][-1]['text'])
"

# 2. 运行 RL 训练（需要 SFT 模型）
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=default

# 3. 查看 pass@k 评估结果
python -m scripts.chat_eval -i rl -a GSM8K
```
