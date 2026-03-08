# 第六章：推理引擎与服务

本章学习 nanochat 的推理引擎（Engine）、命令行聊天界面和 Web 服务。

---

## 目录

1. [推理引擎概述](#1-推理引擎概述)
2. [KV Cache 实现](#2-kv-cache-实现)
3. [Token 采样策略](#3-token-采样策略)
4. [工具调用状态机](#4-工具调用状态机)
5. [命令行聊天 (chat_cli)](#5-命令行聊天-chat_cli)
6. [Web 服务 (chat_web)](#6-web-服务-chat_web)
7. [关键设计理念总结](#7-关键设计理念总结)

---

## 1. 推理引擎概述

文件位置：`nanochat/engine.py`

Engine 是 nanochat 的高效推理核心，负责从 token 序列生成下一个 token。

### 设计原则

- **纯 token 接口**：Engine 不知道文本，只处理 token ID 序列
- **批量生成**：支持同时生成多个采样（num_samples）
- **KV Cache**：分离 prefill 和 decode 阶段
- **工具调用**：内置 Python 计算器的状态机

### 两个核心 API

```python
class Engine:
    def generate(self, tokens, num_samples, max_tokens, temperature, top_k, seed):
        """流式生成，yield (token_column, token_masks)"""
        # token_column: 每个采样的下一个 token
        # token_masks: 1=采样生成, 0=强制注入（工具调用）

    def generate_batch(self, tokens, num_samples, **kwargs):
        """非流式批量生成，返回完整序列"""
        # 内部调用 generate()，收集所有 token 后返回
```

---

## 2. KV Cache 实现

### KV Cache 结构

```python
class KVCache:
    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype):
        # 预分配缓存张量: (n_layers, B, T, H, D) — FA3 布局
        self.k_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim)
        self.v_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim)
        # 每个 batch 元素的当前序列长度
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32)
```

### Prefill + Decode 分离

```python
def generate(self, tokens, num_samples=1, max_tokens=None, ...):
    # 1) Prefill: batch_size=1，处理整个 prompt
    kv_cache_prefill = KVCache(batch_size=1, seq_len=len(tokens), ...)
    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    logits = model.forward(ids, kv_cache=kv_cache_prefill)
    logits = logits[:, -1, :].expand(num_samples, -1)  # 扩展到 num_samples

    # 2) 克隆 KV Cache: 将 batch_size=1 的缓存复制到 num_samples
    kv_cache_decode = KVCache(batch_size=num_samples, seq_len=kv_length_hint, ...)
    kv_cache_decode.prefill(kv_cache_prefill)
    del kv_cache_prefill  # 释放内存

    # 3) Decode: 逐 token 自回归生成
    while num_generated < max_tokens:
        next_ids = sample_next_token(logits, rng, temperature, top_k)
        ...
        logits = model.forward(ids, kv_cache=kv_cache_decode)[:, -1, :]
```

**为什么分离？** Prefill 只需一次前向（处理整个 prompt），然后将 KV Cache 复制给每个采样。这避免了对 prompt 的重复计算。

### KV Cache 的 prefill 方法

```python
def prefill(self, other):
    """将另一个 cache（batch_size=1）的内容复制到本 cache 的所有行"""
    other_pos = other.get_pos()
    self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
    self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
    self.cache_seqlens.fill_(other_pos)
```

---

## 3. Token 采样策略

```python
@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """从 logits 采样下一个 token"""
    # 贪心解码
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # Top-k 采样
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)  # 取 top-k 个
        vals = vals / temperature                    # 温度缩放
        probs = F.softmax(vals, dim=-1)              # 归一化
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)                 # 映射回原始 vocab

    # 全词汇表采样
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=rng)
```

**温度的影响**：
- `temperature=0`：贪心，总是选最高概率的 token
- `temperature=0.6`：比较保守（CLI 默认）
- `temperature=1.0`：正常随机性（RL 采样用）
- `temperature>1.0`：更随机

---

## 4. 工具调用状态机

Engine 内置了一个 Python 计算器的工具调用状态机：

```python
class RowState:
    current_tokens: list      # 当前已生成的 token 序列
    forced_tokens: deque      # 待强制注入的 token 队列
    in_python_block: bool     # 是否在 <|python_start|>...<|python_end|> 内
    python_expr_tokens: list  # 当前 python 表达式的 token
    completed: bool           # 是否已完成生成
```

### 状态转换

```
正常生成 ──[采样到 <|python_start|>]──→ 收集 Python 表达式
    ↑                                          │
    │                                [采样到 <|python_end|>]
    │                                          ↓
    │                                   执行 Python 表达式
    │                                          │
    └──────── 强制注入 <|output_start|> result <|output_end|>
```

### 安全的表达式求值

```python
def use_calculator(expr):
    """安全地执行 Python 表达式"""
    # 纯数学表达式：只允许数字和 +-*/().
    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:  # 禁止幂运算（防止大数攻击）
            return None
        return eval_with_timeout(expr)

    # 字符串操作：只允许 .count() 方法
    if '.count(' not in expr:
        return None

    # 禁止危险模式
    dangerous = ['__', 'import', 'exec', 'eval', 'open', ...]
    if any(pattern in expr for pattern in dangerous):
        return None

    return eval_with_timeout(expr, max_time=3)
```

### 强制注入 vs 自由采样

当工具返回结果时，Engine 将结果 token 放入 `forced_tokens` 队列：

```python
if next_token == python_end and state.in_python_block:
    expr = tokenizer.decode(state.python_expr_tokens)
    result = use_calculator(expr)
    if result is not None:
        state.forced_tokens.append(output_start)
        state.forced_tokens.extend(tokenizer.encode(str(result)))
        state.forced_tokens.append(output_end)
```

后续的 `generate` 循环会优先从 `forced_tokens` 取 token（mask=0），而非采样。

---

## 5. 命令行聊天 (chat_cli)

文件位置：`scripts/chat_cli.py`

### 使用方式

```bash
# 交互模式
python -m scripts.chat_cli

# 单次提问
python -m scripts.chat_cli -p "What is the capital of France?"

# 使用 RL 模型
python -m scripts.chat_cli -i rl

# 调整参数
python -m scripts.chat_cli -t 0.6 -k 50
```

### 对话管理

```python
conversation_tokens = [bos]  # 对话以 BOS 开头

while True:
    user_input = input("User: ")

    # 添加用户消息
    conversation_tokens.append(user_start)
    conversation_tokens.extend(tokenizer.encode(user_input))
    conversation_tokens.append(user_end)

    # 启动助手生成
    conversation_tokens.append(assistant_start)

    # 流式生成
    response_tokens = []
    for token_column, token_masks in engine.generate(conversation_tokens, ...):
        token = token_column[0]
        response_tokens.append(token)
        print(tokenizer.decode([token]), end="", flush=True)

    # 确保 assistant_end 结尾
    if response_tokens[-1] != assistant_end:
        response_tokens.append(assistant_end)
    conversation_tokens.extend(response_tokens)
```

**多轮对话**：所有历史 token 都保留在 `conversation_tokens` 中，作为下次生成的 prompt。

---

## 6. Web 服务 (chat_web)

文件位置：`scripts/chat_web.py`

### 架构

```
┌─────────────┐     ┌─────────────────────┐     ┌─────────────────┐
│  浏览器 UI   │ →   │   FastAPI Server     │ →   │  Worker Pool     │
│  (ui.html)  │ ←   │   (chat_web.py)      │ ←   │  (多 GPU 并行)   │
└─────────────┘     └─────────────────────┘     └─────────────────┘
                     GET  /           → Chat UI
                     POST /chat/completions → 流式生成
                     GET  /health     → 健康检查
                     GET  /stats      → Worker 统计
```

### 启动方式

```bash
# 单 GPU
python -m scripts.chat_web

# 4 GPU 数据并行
python -m scripts.chat_web --num-gpus 4

# 自定义参数
python -m scripts.chat_web -t 0.8 -k 50 -m 512 -p 8000
```

### Worker Pool（数据并行推理）

```python
class WorkerPool:
    """每个 GPU 上加载一个完整模型副本"""

    async def initialize(self, source, model_tag, step):
        for gpu_id in range(self.num_gpus):
            device = torch.device(f"cuda:{gpu_id}")
            model, tokenizer, _ = load_model(source, device, phase="eval")
            engine = Engine(model, tokenizer)
            worker = Worker(gpu_id, device, engine, tokenizer)
            await self.available_workers.put(worker)

    async def acquire_worker(self):
        return await self.available_workers.get()  # 等待可用 worker

    async def release_worker(self, worker):
        await self.available_workers.put(worker)   # 归还 worker
```

**asyncio.Queue** 自动处理并发：多个请求同时到来时，排队等待可用的 GPU worker。

### 流式响应

```python
async def generate_stream(worker, tokens, temperature, max_new_tokens, top_k):
    accumulated_tokens = []
    last_clean_text = ""

    for token_column, token_masks in worker.engine.generate(tokens, ...):
        token = token_column[0]
        accumulated_tokens.append(token)
        current_text = worker.tokenizer.decode(accumulated_tokens)

        # 只有 UTF-8 完整时才发送（处理多字节字符如 emoji）
        if not current_text.endswith('�'):
            new_text = current_text[len(last_clean_text):]
            if new_text:
                yield f"data: {json.dumps({'token': new_text, 'gpu': worker.gpu_id})}\n\n"
                last_clean_text = current_text

    yield f"data: {json.dumps({'done': True})}\n\n"
```

**UTF-8 安全处理**：Token 可能是多字节字符的一部分（如 emoji）。通过检查 `'�'` 替换字符来确保只在 UTF-8 序列完整时才发送。

### 防滥用措施

```python
MAX_MESSAGES_PER_REQUEST = 500
MAX_MESSAGE_LENGTH = 8000
MAX_TOTAL_CONVERSATION_LENGTH = 32000
MAX_TEMPERATURE = 2.0
MAX_TOP_K = 200
MAX_MAX_TOKENS = 4096
```

---

## 7. 关键设计理念总结

### 7.1 Prefill/Decode 分离

- Prefill 只做一次，处理整个 prompt
- KV Cache 复制给所有采样，避免重复计算
- Decode 逐 token 生成，利用 KV Cache 加速

### 7.2 工具调用 = 状态机

不是通过复杂的 API 框架，而是一个简单的状态机：
- 检测 `<|python_start|>` → 收集表达式
- 检测 `<|python_end|>` → 执行 + 注入结果
- `forced_tokens` 队列确保结果被正确插入

### 7.3 数据并行推理

Web 服务使用最简单的数据并行：
- 每个 GPU 加载完整模型
- asyncio Queue 分配请求到空闲 GPU
- 无模型并行，无 tensor 分片

### 7.4 流式输出

- CLI：逐 token 打印（`flush=True`）
- Web：SSE（Server-Sent Events）流式推送
- 两者都处理了 UTF-8 多字节字符的边界问题

### 7.5 最简 API

Web 服务只有 4 个端点：
- `/` — UI 页面
- `/chat/completions` — SSE 流式生成
- `/health` — 健康检查
- `/stats` — 负载统计

---

## 动手实验建议

```bash
# 1. 命令行聊天
python -m scripts.chat_cli -p "What is 7 * 8?"

# 2. 启动 Web 服务
python -m scripts.chat_web
# 然后打开 http://localhost:8000

# 3. 测试 Engine 生成
python -m nanochat.engine  # 内置的一致性测试

# 4. 用 curl 调用 API
curl -X POST http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```
