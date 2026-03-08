# 第三章：监督微调 (SFT)

本章学习如何将预训练基座模型通过监督微调（Supervised Fine-Tuning）变成一个能对话的 Chat 模型。

---

## 目录

1. [SFT 概述：从基座到对话](#1-sft-概述从基座到对话)
2. [训练数据混合](#2-训练数据混合)
3. [Task 抽象与数据格式](#3-task-抽象与数据格式)
4. [GSM8K：数学与工具调用](#4-gsm8k数学与工具调用)
5. [SpellingBee：拼写与字符计数](#5-spellingbee拼写与字符计数)
6. [SFT DataLoader：BOS-Aligned Best-Fit with Padding](#6-sft-dataloaderbos-aligned-best-fit-with-padding)
7. [训练循环详解](#7-训练循环详解)
8. [超参数继承与覆盖](#8-超参数继承与覆盖)
9. [ChatCORE 评估指标](#9-chatcore-评估指标)
10. [关键设计理念总结](#10-关键设计理念总结)

---

## 1. SFT 概述：从基座到对话

文件位置：`scripts/chat_sft.py`

### 什么是 SFT？

SFT 是让预训练基座模型学会"对话格式"的关键步骤：

```
预训练基座模型              SFT 后的 Chat 模型
─────────────              ──────────────────
输入: "The sky is"          输入: "<|user_start|>Why is the sky blue?<|user_end|>"
输出: " blue because..."    输出: "<|assistant_start|>The sky appears blue because..."
(续写文本)                  (遵循对话格式回答)
```

### 启动命令

```bash
# 单 GPU
python -m scripts.chat_sft

# 8 GPU 分布式
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=16
```

### 核心流程

```
加载预训练模型 → 继承超参数 → 构建 SFT 数据混合 → 训练 1 个 epoch → 保存 checkpoint
```

---

## 2. 训练数据混合

SFT 使用多种数据源的混合（`TaskMixture`）：

```python
train_tasks = [
    SmolTalk(split="train"),                              # 460K 行通用对话
    CustomJSON(filepath=identity_conversations_filepath), # 1000 行身份对话
    CustomJSON(filepath=identity_conversations_filepath), # 重复 2 次（过采样）
    *[MMLU(subset="auxiliary_train", split="train")
      for _ in range(args.mmlu_epochs)],                  # 100K × 3 epoch 多选题
    *[GSM8K(subset="main", split="train")
      for _ in range(args.gsm8k_epochs)],                 # 8K × 4 epoch 数学题
    SimpleSpelling(size=200000, split="train"),            # 200K 拼写练习
    SpellingBee(size=80000, split="train"),                # 80K 字母计数
]
```

### 数据量分布

| 数据源 | 行数 | 作用 |
|--------|------|------|
| SmolTalk | 460K | 通用对话能力 |
| Identity | 2K (2×1K) | 模型自我认知 |
| MMLU | 300K (3×100K) | 多选题 / 知识 |
| GSM8K | 32K (4×8K) | 数学推理 + 工具使用 |
| SimpleSpelling | 200K | 单词拼写 |
| SpellingBee | 80K | 字母计数 + 工具使用 |
| **总计** | **~1.07M** | |

### 过采样技巧

```python
# 简单的过采样：在列表中放多份同一个 Task
CustomJSON(filepath=path),  # 第 1 份
CustomJSON(filepath=path),  # 第 2 份 → 相当于 2× 过采样
```

### TaskMixture 混合策略

```python
class TaskMixture(Task):
    def __init__(self, tasks):
        # 1. 构建所有 (task_idx, local_idx) 对
        self.index_map = [(task_idx, local_idx) for ...]
        # 2. 确定性洗牌，使不同任务交错
        rng = random.Random(42)
        rng.shuffle(self.index_map)
```

所有数据源被展平为一个大列表，然后确定性洗牌，确保训练过程中不同任务交替出现。

---

## 3. Task 抽象与数据格式

文件位置：`tasks/common.py`

### Task 基类

```python
class Task:
    def num_examples(self):    # 数据集大小
    def get_example(self, index):  # 返回一个 conversation dict
    def evaluate(self, conversation, completion):  # 评估结果

    @property
    def eval_type(self):       # 'generative' 或 'categorical'
```

### 对话格式

所有 Task 返回统一的对话字典：

```python
conversation = {
    "messages": [
        {"role": "user", "content": "问题文本"},           # 用户消息（字符串）
        {"role": "assistant", "content": "回答文本"},       # 简单回答
        # 或者带工具调用的回答：
        {"role": "assistant", "content": [                   # 列表格式
            {"type": "text", "text": "让我计算..."},
            {"type": "python", "text": "2+2"},              # Python 工具调用
            {"type": "python_output", "text": "4"},          # 工具输出
            {"type": "text", "text": "答案是 4"},
        ]},
    ]
}
```

### 多选题渲染

```python
def render_mc(question, letters, choices):
    """
    格式示例：
    Multiple Choice question: What is 2+2?
    - Three=A
    - Four=B
    - Five=C

    Respond only with the letter of the correct answer.
    """
```

**关键设计**：
- 选项文本在前，字母在后（`choice=letter`），小模型更易学习
- 字母前无空格（`=A` 而非 `= A`），确保 token 一致性

---

## 4. GSM8K：数学与工具调用

文件位置：`tasks/gsm8k.py`

### 数据格式

原始 GSM8K 数据中的工具调用使用 `<< >>` 标记：

```
原始: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
解析为:
  text: "Weng earns 12/60 = $"
  python: "12/60"           (工具调用表达式)
  python_output: "0.2"      (计算结果)
  text: "0.2 per minute."
```

### 解析过程

```python
parts = re.split(r'(<<[^>]+>>)', answer)
for part in parts:
    if part.startswith('<<'):
        inner = part[2:-2]        # 去掉 << >>
        expr, result = inner.rsplit('=', 1)
        assistant_parts.append({"type": "python", "text": expr})
        assistant_parts.append({"type": "python_output", "text": result})
    else:
        assistant_parts.append({"type": "text", "text": part})
```

### 答案提取与评估

```python
GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

def evaluate(self, conversation, assistant_response):
    ref_num = extract_answer(ground_truth_text)   # 提取正确答案
    pred_num = extract_answer(assistant_response)  # 提取模型预测
    return int(pred_num == ref_num)                # 精确匹配
```

GSM8K 使用 `#### 数字` 格式标记最终答案。

---

## 5. SpellingBee：拼写与字符计数

文件位置：`tasks/spellingbee.py`

### 设计动机

LLM 的天然弱点是字符级操作（因为 token ≠ 字符）。SpellingBee 通过 SFT 强制教会小模型：

1. **SimpleSpelling**：学会拼写单词
   ```
   User: Spell the word: apple
   Assistant: apple:a,p,p,l,e
   ```

2. **SpellingBee**：学会计数字母 + 使用 Python 验证
   ```
   User: How many r are in strawberry?
   Assistant: Let me try manual approach...
   strawberry:s,t,r,a,w,b,e,r,r,y
   1:s
   2:t
   3:r hit! count=1
   ...
   This gives us 3.

   Let me double check using Python:
   <|python_start|>'strawberry'.count('r')<|python_end|>
   <|output_start|>3<|output_end|>

   #### 3
   ```

### 数据增强

用户消息模板多达 50+ 种变体，包括多语言：

```python
USER_MSG_TEMPLATES = [
    "How many {letter} are in {word}",
    "Count the {letter} in {word}",
    "{word}中有多少个{letter}",           # 中文
    "{word}에 {letter}가 몇 개 있나요",  # 韩文
    "Combien de {letter} dans {word}",   # 法文
    ...
]
```

还有随机的引号样式（`'r'`、`"r"`、`r`）和大小写变化。

---

## 6. SFT DataLoader：BOS-Aligned Best-Fit with Padding

### 与预训练 DataLoader 的区别

| | 预训练 DataLoader | SFT DataLoader |
|---|---|---|
| 数据源 | Parquet 文件 | Task 对象 |
| 打包方式 | BOS-Aligned Best-Fit **Crop** | BOS-Aligned Best-Fit **Pad** |
| 丢弃策略 | 截断长文档 | **不丢弃**，用 padding 填满 |
| 掩码 | 无 | 有 loss mask |

### 打包算法

```python
def sft_data_generator_bos_bestfit(split, buffer_size=100):
    while True:
        for _ in range(device_batch_size):
            row = []
            while len(row) < row_capacity:
                # 找 buffer 中能完整放入的最大对话
                best_idx = argmax(len(conv) for conv in buffer if len(conv) <= remaining)

                if best_idx >= 0:
                    # 找到了 → 完整放入
                    row.extend(conv_buffer.pop(best_idx))
                else:
                    # 放不下 → padding（不截断！）
                    row.extend([bos_token] * remaining)
                    break
```

**关键改进**：SFT 阶段不截断任何对话（因为对话数量有限且宝贵），改用 padding 填充。

### Loss Mask 处理

```python
# 1. 渲染对话获得 ids 和 mask
ids, mask = tokenizer.render_conversation(conversation)
# mask=1: 助手回复（训练目标）
# mask=0: 用户输入、特殊 token、工具输出

# 2. 构建 targets，mask=0 的位置设为 -1（ignore_index）
targets[mask_targets == 0] = -1

# 3. padding 位置也设为 -1
for i, content_len in enumerate(row_lengths):
    if content_len < row_capacity:
        targets[i, content_len-1:] = -1
```

---

## 7. 训练循环详解

### 学习率调度

SFT 的学习率调度基于**进度**（0→1）而非绝对步数，因为训练步数事先未知（由数据集大小决定）。

```python
def get_lr_multiplier(progress):  # progress: 0.0 → 1.0
    # 阶段 1: warmup (默认 0%)
    # 阶段 2: 恒定
    # 阶段 3: warmdown (默认后 50% 线性衰减)
    ...
```

### 初始学习率

```python
# SFT 从预训练 LR 的 80% 开始（--init-lr-frac=0.8）
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac  # 0.8
```

### 动量调度

```python
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95  # 前 300 步从 0.85 线性升到 0.95
    return momentum
```

### 优化器热启动

```python
# 从预训练 checkpoint 加载优化器状态（动量缓冲等）
optimizer_data = load_optimizer_state("base", device, rank=ddp_rank)
base_lrs = [group["lr"] for group in optimizer.param_groups]
optimizer.load_state_dict(optimizer_data)
# 重要：恢复 SFT 的学习率（预训练 warmdown 到 ~0）
for group, base_lr in zip(optimizer.param_groups, base_lrs):
    group["lr"] = base_lr
```

**为什么要热启动？** 预训练结束时优化器积累了有意义的动量缓冲，从这些缓冲继续比从零开始更好。但学习率必须重置，因为预训练末期 LR 已经衰减到接近 0。

### 训练停止条件

SFT 遍历整个数据集 1 个 epoch（默认）：

```python
# 当消费的数据量 >= 数据集大小时停止
if consumed >= dataset_size:
    last_step = True
```

### 权重衰减

```python
# SFT 阶段权重衰减为 0
optimizer = model.setup_optimizer(..., weight_decay=0.0)
```

预训练末期已经将权重衰减 ramp 到 0，SFT 继续保持为 0。

---

## 8. 超参数继承与覆盖

SFT 的一个巧妙设计是**从预训练 checkpoint 继承超参数**：

```python
for name, fallback, source in [
    ("max_seq_len",       2048,  meta),              # 从 checkpoint 元数据
    ("device_batch_size", 32,    meta),              # 从 checkpoint 元数据
    ("total_batch_size",  524288, meta),             # 从 checkpoint 元数据
    ("embedding_lr",      0.3,   pretrain_user_config), # 从预训练 CLI 参数
    ("unembedding_lr",    0.004, pretrain_user_config),
    ("matrix_lr",         0.02,  pretrain_user_config),
]:
    arg_val = getattr(args, name)
    pretrain_val = source.get(name)
    if arg_val is None:
        # 用户没指定 → 继承预训练值
        resolved = pretrain_val if pretrain_val is not None else fallback
        setattr(args, name, resolved)
```

这意味着用户只需 `python -m scripts.chat_sft` 即可，所有超参数自动从预训练模型继承。

---

## 9. ChatCORE 评估指标

SFT 训练中定期评估 **ChatCORE** 指标：

```python
all_tasks = ['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval', 'SpellingBee']
baseline_accuracies = {
    'ARC-Easy': 0.25,      # 四选一随机基线
    'ARC-Challenge': 0.25,
    'MMLU': 0.25,
    'GSM8K': 0.0,          # 开放式随机基线
    'HumanEval': 0.0,
    'SpellingBee': 0.0,
}

# ChatCORE = 各任务的居中准确率的平均值
# 居中准确率 = (accuracy - baseline) / (1 - baseline)
# 范围：0 = 随机水平, 1 = 完美
```

| 任务 | 类型 | 评估方式 | 基线 |
|------|------|---------|------|
| ARC-Easy | 分类 | logits 直接比较 | 25% |
| ARC-Challenge | 分类 | logits 直接比较 | 25% |
| MMLU | 分类 | logits 直接比较 | 25% |
| GSM8K | 生成 | 采样后答案匹配 | 0% |
| HumanEval | 生成 | 沙箱执行代码 | 0% |
| SpellingBee | 生成 | 答案匹配 | 0% |

---

## 10. 关键设计理念总结

### 10.1 数据为王

- **多样化混合**：通用对话 + 知识问答 + 数学推理 + 代码 + 拼写
- **过采样小数据集**：身份对话只有 1K 行，但放入 2 次
- **多 epoch 重要数据**：MMLU 3 epoch，GSM8K 4 epoch
- **数据增强**：SpellingBee 使用 50+ 模板 × 多语言 × 多格式

### 10.2 不丢弃数据

SFT 阶段的 DataLoader 使用 padding 而非截断，确保每条对话完整保留。SFT 数据宝贵，不能浪费。

### 10.3 热启动优化器

从预训练 checkpoint 继承优化器状态（动量缓冲），避免重新积累动量。但学习率必须重置。

### 10.4 只训练助手回复

Loss mask 确保模型只在助手回复的 token 上计算 loss：
- 用户消息：mask=0（不训练）
- 特殊 token：mask=0
- 工具输出：mask=0（来自外部）
- 助手文本：mask=1（训练目标！）
- 工具调用表达式：mask=1（模型要学会调用工具）

### 10.5 工具使用是 SFT 的一等公民

GSM8K 和 SpellingBee 都包含工具调用，让模型在 SFT 阶段就学会：
1. 何时调用 Python
2. 如何编写正确的表达式
3. 如何利用工具输出

---

## 动手实验建议

```bash
# 1. 查看 SFT 数据格式
python -c "
from tasks.gsm8k import GSM8K
task = GSM8K(subset='main', split='train')
print(task[0])
"

# 2. 查看 SpellingBee 数据
python -m tasks.spellingbee

# 3. 运行 SFT（需要预训练好的模型）
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=16

# 4. 查看 tokenization 的 mask 可视化
python -c "
from nanochat.tokenizer import get_tokenizer
from tasks.gsm8k import GSM8K
tok = get_tokenizer()
task = GSM8K(subset='main', split='train')
conv = task[0]
ids, mask = tok.render_conversation(conv)
print(tok.visualize_tokenization(ids, mask))
"
```
