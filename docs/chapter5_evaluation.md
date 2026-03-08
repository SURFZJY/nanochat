# 第五章：评估体系

本章学习 nanochat 的完整评估体系——从基座模型到 Chat 模型的全方位评估。

---

## 目录

1. [评估体系总览](#1-评估体系总览)
2. [基座模型评估 (base_eval)](#2-基座模型评估-base_eval)
3. [CORE 指标：ICL 任务评估](#3-core-指标icl-任务评估)
4. [BPB 评估](#4-bpb-评估)
5. [Chat 模型评估 (chat_eval)](#5-chat-模型评估-chat_eval)
6. [分类任务评估](#6-分类任务评估)
7. [生成任务评估](#7-生成任务评估)
8. [代码执行沙箱](#8-代码执行沙箱)
9. [关键设计理念总结](#9-关键设计理念总结)

---

## 1. 评估体系总览

nanochat 有两套独立的评估体系：

```
基座模型评估 (base_eval.py)          Chat 模型评估 (chat_eval.py)
─────────────────────               ─────────────────────────
评估预训练模型的"原始能力"            评估 SFT/RL 模型的"对话能力"
  ├── CORE 指标 (ICL 任务)             ├── ARC-Easy (分类)
  ├── BPB (bits per byte)              ├── ARC-Challenge (分类)
  └── 文本生成采样                      ├── MMLU (分类)
                                       ├── GSM8K (生成)
                                       ├── HumanEval (生成+执行)
                                       └── SpellingBee (生成)
```

**核心区别**：
- 基座模型评估用 **ICL (In-Context Learning)**：给模型几个示例（few-shot），看它能否模式匹配
- Chat 模型评估用 **对话格式**：让模型在对话框架下回答问题

---

## 2. 基座模型评估 (base_eval)

文件位置：`scripts/base_eval.py`

### 启动命令

```bash
# 评估 nanochat 模型
torchrun --nproc_per_node=8 -m scripts.base_eval --model-tag d24

# 评估 HuggingFace 模型（如 GPT-2）
torchrun --nproc_per_node=8 -m scripts.base_eval --hf-path openai-community/gpt2

# 选择评估模式
torchrun --nproc_per_node=8 -m scripts.base_eval --eval core,bpb,sample
```

### 三种评估模式

| 模式 | 说明 |
|------|------|
| `core` | CORE 指标（DCLM 基准上的 ICL 任务准确率） |
| `bpb` | Bits Per Byte（训练集和验证集） |
| `sample` | 条件和无条件文本生成 |

### HuggingFace 模型兼容

```python
class ModelWrapper:
    """给 HuggingFace 模型一个 nanochat 兼容接口"""
    def __call__(self, input_ids, targets=None, loss_reduction='mean'):
        logits = self.model(input_ids).logits
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                               ignore_index=-1, reduction=loss_reduction)
        return loss
```

---

## 3. CORE 指标：ICL 任务评估

文件位置：`nanochat/core_eval.py`

### 什么是 CORE？

CORE 来自 DCLM 论文（https://arxiv.org/abs/2406.11794），使用多个 ICL 任务评估基座模型。

### 三种任务类型

#### a) Multiple Choice（多选题）

```
Context: What is 2+2?
Prompt A: What is 2+2? Answer: Three
Prompt B: What is 2+2? Answer: Four    ← 哪个 loss 最低？
Prompt C: What is 2+2? Answer: Five
```

**评估方式**：所有选项共享相同前缀（common prefix），只比较续写部分的平均 loss。

```python
def batch_sequences_mc(tokenizer, prompts):
    tokens = tokenizer(prompts, prepend=bos)
    answer_start_idx = find_common_length(tokens, direction='left')  # 找到公共前缀长度
    return tokens, [answer_start_idx] * len(prompts), [len(x) for x in tokens]
```

#### b) Schema（情境选择）

```
Prompt 1: The sun rose in the east. The answer is: sunrise
Prompt 2: The bird flew in the east. The answer is: sunrise
```

**评估方式**：所有选项共享相同后缀（common suffix），比较不同上下文下的 loss。

```python
def batch_sequences_schema(tokenizer, prompts):
    suffix_length = find_common_length(tokens, direction='right')  # 找到公共后缀长度
```

#### c) Language Modeling（语言建模）

```
Prompt without: "The capital of France is"
Prompt with:    "The capital of France is Paris"
```

**评估方式**：检查模型是否能准确预测续写的每一个 token。

```python
# 对比 prompt_without 之后的 token 预测
predicted_tokens = predictions[0, si-1:ei-1]
actual_tokens = input_ids[0, si:ei]
is_correct = torch.all(predicted_tokens == actual_tokens).item()
```

### CORE 得分计算

```python
# 居中结果 = (accuracy - random_baseline) / (1 - random_baseline)
centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)

# CORE = 所有任务的居中结果的平均值
core_metric = sum(centered_results.values()) / len(centered_results)
```

### few-shot 设置

```python
if num_fewshot > 0:
    rng = random.Random(1234 + idx)
    available_indices = [i for i in range(len(data)) if i != idx]
    fewshot_indices = rng.sample(available_indices, num_fewshot)
```

每个测试样本使用确定性随机选择的 few-shot 示例（排除自身）。

---

## 4. BPB 评估

详见第一章第 8 节。base_eval 会在训练集和验证集上分别评估 BPB：

```python
for split_name in ["train", "val"]:
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, batch_size, sequence_len, split_name, device=device
    )
    bpb = evaluate_bpb(model, loader, steps, token_bytes)
```

**关注点**：
- `train_bpb` 和 `val_bpb` 的差距 → 反映过拟合程度
- `val_bpb` 越低越好
- BPB 是词汇表无关的指标

---

## 5. Chat 模型评估 (chat_eval)

文件位置：`scripts/chat_eval.py`

### 启动命令

```bash
# 评估 SFT 模型的所有任务
torchrun --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# 评估 RL 模型的特定任务
python -m scripts.chat_eval -i rl -a GSM8K

# 多个任务
python -m scripts.chat_eval -i sft -a "ARC-Easy|MMLU|GSM8K"
```

### 任务路由

```python
def run_chat_eval(task_name, model, tokenizer, engine, ...):
    task_module = {
        'HumanEval': HumanEval,
        'MMLU': partial(MMLU, subset="all", split="test"),
        'ARC-Easy': partial(ARC, subset="ARC-Easy", split="test"),
        'ARC-Challenge': partial(ARC, subset="ARC-Challenge", split="test"),
        'GSM8K': partial(GSM8K, subset="main", split="test"),
        'SpellingBee': partial(SpellingBee, size=256, split="test"),
    }[task_name]

    if task_object.eval_type == 'generative':
        acc = run_generative_eval(...)
    elif task_object.eval_type == 'categorical':
        acc = run_categorical_eval(...)
```

### ChatCORE 指标

```python
# 与 CORE 类似，但用于 Chat 模型
baseline_accuracies = {
    'ARC-Easy': 0.25,       # 四选一
    'ARC-Challenge': 0.25,
    'MMLU': 0.25,
    'GSM8K': 0.0,           # 开放式
    'HumanEval': 0.0,
    'SpellingBee': 0.0,
}
chatcore_metric = mean(
    (acc - baseline) / (1 - baseline)
    for task in all_tasks
)
```

---

## 6. 分类任务评估

```python
def run_categorical_eval(task_object, tokenizer, model, batch_size):
    """高效批量评估——不需要采样，直接看 logits"""

    for batch in batches:
        # 1. 将对话渲染为 completion 格式
        prompt_ids = [tokenizer.render_for_completion(conv) for conv in conversations]

        # 2. 批量 padding 后一次前向
        with torch.no_grad():
            logits = model(prompt_ids)  # (B, T, V)

        # 3. 只看答案位置的选项字母 logits
        for idx, conversation in enumerate(conversations):
            letters = conversation['letters']  # 如 ['A', 'B', 'C', 'D']
            letter_ids = [tokenizer.encode(letter)[0] for letter in letters]
            focus_logits = logits[idx, answer_pos, letter_ids]
            predicted_letter = letters[focus_logits.argmax().item()]
            outcome = task_object.evaluate(conversation, predicted_letter)
```

**效率优势**：
- 不需要逐个采样，批量前向即可
- 只比较选项字母的 logits，无需生成完整文本
- 多个问题可以批量并行处理

---

## 7. 生成任务评估

```python
def run_generative_eval(task_object, tokenizer, model, engine,
                        num_samples, max_new_tokens, temperature, top_k):
    """需要采样生成，然后评估结果"""

    for i in range(ddp_rank, num_problems, ddp_world_size):
        conversation = task_object[i]
        encoded_prompt = tokenizer.render_for_completion(conversation)

        # 生成多个采样
        results, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=num_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        # 评估每个采样
        completions = [tokenizer.decode(r[prefix_length:]) for r in results]
        outcomes = [task_object.evaluate(conversation, c) for c in completions]
        passed = any(outcomes)  # pass@k: 任意一个正确即通过
```

**分布式处理**：不同 rank 处理不同的问题（stride 分片），最后 all_reduce 汇总。

---

## 8. 代码执行沙箱

文件位置：`nanochat/execution.py`

HumanEval 评估需要执行模型生成的 Python 代码。nanochat 提供了沙箱环境：

### 安全措施

```python
def reliability_guard(maximum_memory_bytes=None):
    """禁用危险函数，防止生成代码造成破坏"""
    # 内存限制
    resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
    # 禁用危险 OS 操作
    os.kill = None
    os.system = None
    os.remove = None
    os.fork = None
    # 禁用 subprocess
    subprocess.Popen = None
    # 禁用 shutil 操作
    shutil.rmtree = None
    shutil.move = None
```

### 执行流程

```python
def execute_code(code, timeout=5.0, maximum_memory_bytes=256*1024*1024):
    """在子进程中执行代码"""
    # 1. 在新进程中运行（可被 kill）
    p = multiprocessing.Process(target=_unsafe_execute, args=(...))
    p.start()
    p.join(timeout=timeout + 1)

    # 2. 超时则强制终止
    if p.is_alive():
        p.kill()
        return ExecutionResult(success=False, timeout=True)

    # 3. 返回结果
    return ExecutionResult(
        success=result_dict["success"],
        stdout=result_dict["stdout"],
        stderr=result_dict["stderr"],
        error=result_dict["error"],
    )
```

### 沙箱层次

| 保护层 | 防御内容 |
|--------|---------|
| 独立进程 | 崩溃/挂起不影响主进程 |
| 超时限制 | 5 秒默认，防止死循环 |
| 内存限制 | 256MB 默认，防止内存炸弹 |
| 临时目录 | 代码在 tmpdir 中运行，完后删除 |
| 函数禁用 | os.kill/fork/remove 等全部置 None |
| stdin 禁用 | 防止代码等待输入 |

### 不覆盖的攻击面

代码注释中明确说明这不是安全沙箱：
- 网络访问未阻止
- ctypes 可能绕过限制
- 无内核级隔离（seccomp、容器）

---

## 9. 关键设计理念总结

### 9.1 评估即设计

评估体系的设计反映了训练目标：
- **CORE / ChatCORE**：用居中准确率消除随机基线影响
- **BPB**：用词汇表无关的指标，可跨模型对比
- **pass@k**：宽容地评估模型"能力"而非"一次性表现"

### 9.2 分类 vs 生成

- **分类任务**（ARC、MMLU）：高效批量前向，只看 logits → 速度快
- **生成任务**（GSM8K、HumanEval、SpellingBee）：需要采样 + 外部评估 → 速度慢但更真实

### 9.3 分布式评估

所有评估都支持多 GPU 分布式：
```python
for idx in range(ddp_rank, num_problems, ddp_world_size):
    ...  # 每个 rank 处理不同的问题
# 最后 all_reduce 汇总结果
```

### 9.4 可比较性

- 基座模型可与 HuggingFace 模型直接对比（ModelWrapper）
- 所有指标都有明确的随机基线
- BPB 允许不同词汇量的模型公平比较

---

## 动手实验建议

```bash
# 1. 快速评估基座模型（子集）
python -m scripts.base_eval --model-tag d12 --max-per-task=100 --split-tokens=524288

# 2. 评估 Chat 模型单个任务
python -m scripts.chat_eval -i sft -a ARC-Easy -b 8

# 3. 测试代码执行沙箱
python -c "
from nanochat.execution import execute_code
result = execute_code('print(sum(range(100)))')
print(result)
result = execute_code('import time; time.sleep(10)')  # 会超时
print(result)
"

# 4. 查看 CORE 评估详情
torchrun --nproc_per_node=8 -m scripts.base_eval --eval core --model-tag d24
```
