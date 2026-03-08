# 第一章：项目总览、分词器与数据管道

本章学习 nanochat 的整体架构、BPE 分词器和数据处理管道——它们是整个 LLM 训练流水线的基础。

---

## 目录

1. [项目总览：nanochat 是什么](#1-项目总览nanochat-是什么)
2. [环境与依赖](#2-环境与依赖)
3. [端到端流水线](#3-端到端流水线)
4. [数据集：ClimbMix-400B](#4-数据集climbmix-400b)
5. [BPE 分词器详解](#5-bpe-分词器详解)
6. [分词器训练流程](#6-分词器训练流程)
7. [分词器评估](#7-分词器评估)
8. [Bits Per Byte：词汇表无关的评估指标](#8-bits-per-byte词汇表无关的评估指标)
9. [对话模板与特殊 Token](#9-对话模板与特殊-token)
10. [关键设计理念总结](#10-关键设计理念总结)

---

## 1. 项目总览：nanochat 是什么

nanochat 是 Andrej Karpathy 开发的**最简 LLM 全栈训练框架**，目标是在单个 GPU 节点上端到端训练一个可以对话的 ChatGPT 模型。

### 核心理念

- **极简**：没有配置文件工厂、没有巨型 if-else，代码量极小且可读
- **全栈**：覆盖分词→预训练→微调→评估→推理→Web 对话 全流程
- **一个旋钮**：`--depth`（Transformer 层数）是唯一的复杂度控制参数
- **低成本**：GPT-2 能力级别的模型只需 ~$48（8×H100 约 2 小时）

### 项目成就

| 时间线 | 训练时间 | 成本 | 说明 |
|-------|---------|------|------|
| 2019 OpenAI GPT-2 | 168 小时 | ~$43,000 | 原始 GPT-2 |
| 2026 nanochat 最新 | ~2 小时 | ~$48 | 8×H100 节点 |

7 年间，从 168 小时降到 2 小时，成本降低 ~900 倍。

---

## 2. 环境与依赖

文件位置：`pyproject.toml`

```toml
[project]
name = "nanochat"
requires-python = ">=3.10"
```

### 核心依赖

| 依赖 | 用途 |
|------|------|
| `torch==2.9.1` | PyTorch 深度学习框架 |
| `tiktoken` | OpenAI 的高效 BPE 推理编码器 |
| `rustbpe` | Rust 实现的 BPE 训练器 |
| `tokenizers` | HuggingFace 的分词器库 |
| `wandb` | 实验日志和可视化 |
| `fastapi + uvicorn` | Web UI 聊天服务 |
| `datasets` | HuggingFace 数据集工具 |
| `pyarrow` | Parquet 文件读写 |

### 安装方式

```bash
# 安装 uv 包管理器
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境并安装依赖
uv venv
uv sync --extra gpu    # GPU 版本（含 CUDA 12.8 的 PyTorch）
# 或
uv sync --extra cpu    # CPU 版本

source .venv/bin/activate
```

nanochat 使用 `uv` 而非 `pip`，因为 uv 是 Rust 写的，速度快 10-100 倍。

---

## 3. 端到端流水线

完整流水线定义在 `runs/speedrun.sh` 中，一图概览：

```
┌─────────────────────────────────────────────────────────────────────┐
│                     nanochat 全栈 LLM 流水线                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  第一章（本章）                                                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │
│  │  数据集下载    │ →  │  分词器训练    │ →  │  分词器评估    │           │
│  │  dataset.py   │    │  tok_train   │    │  tok_eval    │           │
│  └──────────────┘    └──────────────┘    └──────────────┘           │
│                                                                     │
│  第二章                                                              │
│  ┌──────────────┐    ┌──────────────┐                               │
│  │  基座模型训练   │ →  │  基座模型评估   │                               │
│  │  base_train   │    │  base_eval   │                               │
│  └──────────────┘    └──────────────┘                               │
│                                                                     │
│  后续章节                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │
│  │  SFT 微调     │ →  │  RL 强化学习   │ →  │  Chat 评估    │           │
│  │  chat_sft     │    │  chat_rl     │    │  chat_eval   │           │
│  └──────────────┘    └──────────────┘    └──────────────┘           │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐                               │
│  │  CLI 对话     │    │  Web UI 对话  │                               │
│  │  chat_cli     │    │  chat_web    │                               │
│  └──────────────┘    └──────────────┘                               │
└─────────────────────────────────────────────────────────────────────┘
```

### speedrun.sh 的执行顺序

```bash
# 1. 环境设置
uv sync --extra gpu && source .venv/bin/activate

# 2. 数据下载（先 8 个 shard 给分词器，后台下载 170 个给预训练）
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 170 &

# 3. 分词器训练与评估
python -m scripts.tok_train
python -m scripts.tok_eval

# 4. 预训练 (d24, ~2 小时)
torchrun --nproc_per_node=8 -m scripts.base_train -- --depth=24 --fp8

# 5. 基座评估
torchrun --nproc_per_node=8 -m scripts.base_eval

# 6. SFT 微调 + 评估
torchrun --nproc_per_node=8 -m scripts.chat_sft
torchrun --nproc_per_node=8 -m scripts.chat_eval

# 7. 生成报告
python -m nanochat.report generate
```

**注意并行下载的技巧**：先下载 8 个 shard 够分词器训练，同时后台继续下载剩余 shard，等预训练开始前再 `wait`。

---

## 4. 数据集：ClimbMix-400B

文件位置：`nanochat/dataset.py`

### 数据集概况

```python
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542                                           # 最大 shard 编号
DATA_DIR = os.path.join(base_dir, "base_data_climbmix")    # 本地存储目录
```

- **数据源**：NVIDIA ClimbMix（2026年3月从 FineWeb-EDU 切换而来）
- **总量**：400B tokens，6543 个 parquet 文件
- **每个 shard**：~100MB 压缩文本，约 250M 字符
- **格式**：Parquet 文件，每行有一个 `text` 列
- **训练集/验证集**：最后一个 shard 固定为验证集，其余为训练集

### 下载机制

```bash
# 下载 170 个 shard（GPT-2 级别训练够用），4 个并行下载 worker
python -m nanochat.dataset -n 170 -w 4
```

下载特性：
- **按需下载**：只下载指定数量的 shard
- **断点续传**：已存在的文件自动跳过
- **原子写入**：先写 `.tmp` 临时文件，再 `os.rename` 原子移动
- **指数退避重试**：失败后等待 2^attempt 秒重试，最多 5 次
- **并行下载**：使用 `multiprocessing.Pool` 多进程下载

### 数据迭代器

```python
def parquets_iter_batched(split, start=0, step=1):
    """按 row_group 批次迭代数据集"""
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts
```

`start` 和 `step` 参数支持 DDP 分布式训练中的数据分片：rank 0 读第 0、N、2N... 个 row group，rank 1 读第 1、N+1、2N+1...

---

## 5. BPE 分词器详解

文件位置：`nanochat/tokenizer.py`

### 5.1 什么是 BPE？

**Byte Pair Encoding (字节对编码)** 是目前主流 LLM 使用的分词算法：

```
原始文本: "hello world"
UTF-8 字节: [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]

BPE 训练过程：
1. 统计所有相邻字节对的频率
2. 合并频率最高的一对 → 创建新 token
3. 重复直到达到目标词汇量

结果示例：
"hello" → [hello]  (一个 token)
"world" → [world]  (一个 token)
" "     → [ ]      (一个 token)
```

### 5.2 nanochat 的双实现策略

nanochat 提供两套分词器实现，共享同一套接口：

| | HuggingFaceTokenizer | RustBPETokenizer（默认） |
|---|---|---|
| 训练后端 | HuggingFace `tokenizers` | `rustbpe`（Rust 实现） |
| 推理后端 | HuggingFace `tokenizers` | `tiktoken`（OpenAI） |
| 速度 | 较慢 | **更快** |
| 批量编码 | 逐个 | 多线程 `encode_ordinary_batch` |

**为什么用两个库？** rustbpe 训练快，tiktoken 推理快且支持多线程批量编码。

### 5.3 GPT-4 风格的分词规则

```python
# 分词正则表达式（决定如何将文本切分为"词"，然后对每个词做 BPE）
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```

这个正则表达式的作用：

| 模式 | 匹配内容 | 示例 |
|------|---------|------|
| `'(?i:[sdmt]\|ll\|ve\|re)` | 英文缩写 | `'s`, `'ll`, `'ve`, `'re` |
| `[^\r\n\p{L}\p{N}]?+\p{L}+` | 字母词（可能前缀一个符号） | `hello`, ` world` |
| `\p{N}{1,2}` | 1-2 位数字 | `12`, `5`（**不是** GPT-4 的 1-3 位） |
| ` ?[^\s\p{L}\p{N}]++[\r\n]*` | 标点/符号 | `!`, `...` |
| `\s*[\r\n]` | 换行 | `\n` |
| `\s+(?!\S)\|\s+` | 空白 | `   ` |

**与 GPT-4 的区别**：nanochat 将数字匹配从 `\p{N}{1,3}` 改为 `\p{N}{1,2}`。原因是对于较小的词汇表 (32K)，3 位数字匹配会浪费 token 空间。经实验验证，2 位是 32K 词汇量的最优选择。

### 5.4 特殊 Token

```python
SPECIAL_TOKENS = [
    "<|bos|>",              # 文档/序列起始
    "<|user_start|>",       # 用户消息开始
    "<|user_end|>",         # 用户消息结束
    "<|assistant_start|>",  # 助手消息开始
    "<|assistant_end|>",    # 助手消息结束
    "<|python_start|>",     # Python 工具调用开始
    "<|python_end|>",       # Python 工具调用结束
    "<|output_start|>",     # Python 输出开始
    "<|output_end|>",       # Python 输出结束
]
```

特殊 token 不参与 BPE 训练，而是在训练后追加到词汇表末尾。共 9 个特殊 token：1 个用于预训练（BOS），8 个用于微调阶段的对话和工具使用。

### 5.5 RustBPETokenizer 训练流程

```python
@classmethod
def train_from_iterator(cls, text_iterator, vocab_size):
    # 1) 用 rustbpe 训练 BPE
    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)  # 32768 - 9 = 32759
    tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)

    # 2) 用训练好的 merge 规则构建 tiktoken Encoding（用于推理）
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}

    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,   # 字节序列 → 合并优先级
        special_tokens=special_tokens,      # 特殊 token → ID
    )
    return cls(enc, "<|bos|>")
```

**训练 + 推理的桥接**：rustbpe 训练出 merge 规则（`mergeable_ranks`），然后传给 tiktoken 构建推理编码器。这样训练用 Rust 速度快，推理用 tiktoken 也快。

### 5.6 编码与解码

```python
# 单个字符串编码
ids = tokenizer.encode("Hello world")                     # → [token_ids...]
ids = tokenizer.encode("Hello", prepend="<|bos|>")        # → [bos_id, token_ids...]

# 批量编码（多线程，默认 8 线程）
ids_batch = tokenizer.encode(["Hello", "World"], num_threads=8)

# 解码
text = tokenizer.decode(ids)                               # → "Hello world"
```

**批量编码的性能优势**：tiktoken 的 `encode_ordinary_batch` 使用多线程并行编码，在数据加载器中显著加速。

---

## 6. 分词器训练流程

文件位置：`scripts/tok_train.py`

### 执行命令

```bash
python -m scripts.tok_train
# 可选参数:
#   --max-chars 2000000000   训练最大字符数（默认 2B）
#   --doc-cap 10000          单文档最大字符数（默认 10K）
#   --vocab-size 32768       词汇表大小（默认 2^15）
```

### 训练数据迭代器

```python
def text_iterator():
    nchars = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            doc_text = doc[:args.doc_cap]     # 截断到 10K 字符
            nchars += len(doc_text)
            yield doc_text
            if nchars > args.max_chars:       # 累计达到 2B 字符后停止
                return
```

**关键设计**：
- **文档截断 (doc_cap=10K)**：避免超长文档主导词汇表
- **总量限制 (max_chars=2B)**：2B 字符足以学到好的 merge 规则
- 只需要 8 个 shard（~2B 字符）就够训练分词器

### 训练后处理：token_bytes 映射

```python
# 为每个 token 计算对应的 UTF-8 字节数
token_bytes = []
for token_id in range(vocab_size):
    token_str = tokenizer.decode([token_id])
    if token_str in special_set:
        token_bytes.append(0)                          # 特殊 token 不计字节
    else:
        token_bytes.append(len(token_str.encode("utf-8")))  # UTF-8 字节数
torch.save(token_bytes, "token_bytes.pt")
```

这个映射表用于计算 **Bits Per Byte (BPB)**——一个与词汇表大小无关的评估指标（详见第 8 节）。

### 产出文件

```
~/.cache/nanochat/tokenizer/
├── tokenizer.pkl      # tiktoken Encoding 对象（pickle 序列化）
└── token_bytes.pt     # token → 字节数映射（PyTorch tensor）
```

---

## 7. 分词器评估

文件位置：`scripts/tok_eval.py`

评估脚本将 nanochat 分词器与 GPT-2 和 GPT-4 分词器在多种文本类型上进行对比。

### 测试文本类型

| 类型 | 说明 |
|------|------|
| `news` | 英语新闻报道 |
| `korean` | 韩语新闻 |
| `code` | Python 代码 |
| `math` | LaTeX 数学公式 |
| `science` | 科学论文段落 |
| `fwe-train` | 训练集样本 |
| `fwe-val` | 验证集样本 |

### 评估指标：压缩率

```python
# 压缩率 = 原始字节数 / token 数
# 越高越好（每个 token 代表更多信息）
ratio = len(text.encode('utf-8')) / len(tokenizer.encode(text))
```

**为什么 nanochat 的 32K 词汇表 vs GPT-4 的 100K？** 对于小模型，更小的词汇表意味着：
- 更小的 embedding 矩阵 → 更少的参数浪费
- 虽然压缩率稍低，但在小模型上总体效率更高

### 评估结果解读

```
Comparison with GPT-4:
Text Type  Bytes  GPT-4 Tokens/Ratio  Ours Tokens/Ratio  Better
news       1632   362    4.51          430    3.80        GPT-4   (词汇量大 3x，理所当然)
code       886    283    3.13          237    3.74        Ours    (针对代码优化更好)
```

nanochat 分词器在代码和训练数据上通常有竞争力，但在通用英文文本上不如 GPT-4（词汇量只有 GPT-4 的 1/3）。

---

## 8. Bits Per Byte：词汇表无关的评估指标

文件位置：`nanochat/loss_eval.py`

### 问题：为什么不直接用 loss？

标准的交叉熵 loss 是"每 token 的 nats"，但不同分词器的 token 大小不同：
- 32K 词汇表：`"hello"` = 1 个 token → loss 只算 1 次
- 256 词汇表：`"hello"` = 5 个字节 → loss 算 5 次

这导致不同词汇量的模型之间无法直接比较 loss。

### 解决方案：Bits Per Byte (BPB)

```python
def evaluate_bpb(model, batches, steps, token_bytes):
    total_nats = 0.0    # 所有 token 的 loss 总和
    total_bytes = 0     # 所有 token 对应的字节总数

    for x, y in batches:
        loss2d = model(x, y, loss_reduction='none')   # 不做 reduction
        num_bytes2d = token_bytes[y]                   # 每个 target token 的字节数
        total_nats += (loss2d * (num_bytes2d > 0)).sum()  # 忽略特殊 token
        total_bytes += num_bytes2d.sum()

    bpb = total_nats / (math.log(2) * total_bytes)    # nats → bits，除以总字节数
    return bpb
```

**公式**：`BPB = Σ(nats) / (ln(2) × Σ(bytes))`

- `nats → bits`：除以 `ln(2)` 将自然对数的 nats 转换为以 2 为底的 bits
- `÷ total_bytes`：按总字节数归一化，消除词汇表大小的影响
- 特殊 token（`num_bytes = 0`）不参与计算

**直觉理解**：BPB 表示模型预测文本中每个字节需要多少 bit。信息论的下限是该语言的熵。BPB 越低，模型越好。

### 分布式 BPB 计算

```python
if world_size > 1:
    dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
```

多 GPU 时，先各自累加本地的 nats 和 bytes，然后 all_reduce 汇总后再计算 BPB。

---

## 9. 对话模板与特殊 Token

分词器还负责将对话渲染为 token 序列（用于 SFT 微调）。

### 对话格式

```python
conversation = {
    "messages": [
        {"role": "user", "content": "Why is the sky blue?"},
        {"role": "assistant", "content": "The sky appears blue because..."},
    ]
}
```

### Token 渲染

```python
def render_conversation(self, conversation, max_tokens=2048):
    # 产出：ids (token列表) 和 mask (训练掩码)
    # mask=0: 不训练（用户输入、特殊 token）
    # mask=1: 训练目标（助手回复）

    add_tokens(bos, mask=0)
    for message in messages:
        if message["role"] == "user":
            add_tokens(user_start, mask=0)
            add_tokens(encode(content), mask=0)     # 用户内容不训练
            add_tokens(user_end, mask=0)
        elif message["role"] == "assistant":
            add_tokens(assistant_start, mask=0)
            add_tokens(encode(content), mask=1)     # 助手内容要训练!
            add_tokens(assistant_end, mask=1)
```

渲染后的 token 序列示例：

```
<|bos|> <|user_start|> Why is the sky blue? <|user_end|> <|assistant_start|> The sky... <|assistant_end|>
[  0  ] [     0     ] [       0          ] [    0    ] [       0        ] [    1   ] [      1       ]
                                                                          ^^^^^^^^    ^^^^^^^^^^^^^
                                                                          训练目标      训练目标
```

### 工具调用支持

助手消息可以包含结构化的工具调用：

```python
content = [
    {"type": "text", "text": "Let me calculate..."},       # mask=1
    {"type": "python", "text": "print(2+2)"},              # mask=1 (模型要学会调用)
    {"type": "python_output", "text": "4"},                 # mask=0 (输出来自外部)
]
```

渲染为：`...Let me calculate...<|python_start|>print(2+2)<|python_end|><|output_start|>4<|output_end|>...`

### RL 用的 Completion 渲染

```python
def render_for_completion(self, conversation):
    """去掉最后一条助手消息，追加 <|assistant_start|>，让模型自己续写"""
    messages.pop()  # 移除最后的助手回复
    ids, mask = self.render_conversation(conversation)
    ids.append(assistant_start)  # 引导模型开始回复
    return ids
```

---

## 10. 关键设计理念总结

### 10.1 数据管道设计

```
HuggingFace (ClimbMix-400B)
    ↓ 按需下载，原子写入，指数退避重试
Parquet 文件 (本地磁盘)
    ↓ PyArrow 按 row_group 读取，DDP 分片
文本字符串
    ↓ tiktoken 多线程批量编码
Token IDs
    ↓ BOS-aligned Best-Fit 打包（第二章详述）
训练 Batch (B × T tensor)
```

### 10.2 分词器设计哲学

| 决策 | 选择 | 原因 |
|------|------|------|
| 词汇量 | 32768 (2^15) | 小模型不需要大词汇表；2 的幂利于 tensor core 对齐 |
| 数字匹配 | 1-2 位 | 节省 token 空间（相比 GPT-4 的 1-3 位） |
| 训练库 | rustbpe | Rust 实现，速度快 |
| 推理库 | tiktoken | 多线程批量编码，高效 |
| 字节回退 | 有 | 保证任何 UTF-8 文本都能编码 |
| 评估指标 | BPB | 词汇表无关，可跨模型对比 |

### 10.3 全局设计原则

1. **渐进式数据下载**：先下载 8 shard 训练分词器，后台下载 170 shard 供预训练
2. **训练/推理分离**：分词器的训练和推理用不同的库，各取所长
3. **原子操作**：下载文件先写 `.tmp` 再 rename，防止损坏
4. **可恢复性**：已下载的文件自动跳过，支持断点续传
5. **BPB > Loss**：使用词汇表无关的 BPB 指标，使评估更公平

---

## 动手实验建议

```bash
# 1. 下载少量数据
python -m nanochat.dataset -n 2

# 2. 训练一个小词汇量的分词器（快速实验）
python -m scripts.tok_train --vocab-size 4096 --max-chars 100000000

# 3. 评估分词器压缩率
python -m scripts.tok_eval

# 4. Python 中手动体验分词
python -c "
from nanochat.tokenizer import get_tokenizer
tok = get_tokenizer()
text = 'Hello, world! 你好世界'
ids = tok.encode(text)
print(f'Text: {text}')
print(f'Token IDs: {ids}')
print(f'Num tokens: {len(ids)}')
print(f'Decoded: {tok.decode(ids)}')
print(f'Compression ratio: {len(text.encode(\"utf-8\")) / len(ids):.2f} bytes/token')
"
```
