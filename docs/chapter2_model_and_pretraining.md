# 第二章：GPT 模型架构与预训练

本章深入学习 nanochat 的核心——GPT Transformer 模型架构（`nanochat/gpt.py`）和预训练流程（`scripts/base_train.py`）。

---

## 目录

1. [模型架构总览](#1-模型架构总览)
2. [GPTConfig：配置即设计](#2-gptconfig配置即设计)
3. [核心组件详解](#3-核心组件详解)
4. [前向传播流程](#4-前向传播流程)
5. [权重初始化策略](#5-权重初始化策略)
6. [优化器：MuonAdamW](#6-优化器muonadamw)
7. [预训练流程详解](#7-预训练流程详解)
8. [数据加载器](#8-数据加载器)
9. [Scaling Laws 与超参数自动推导](#9-scaling-laws-与超参数自动推导)
10. [关键设计理念总结](#10-关键设计理念总结)

---

## 1. 模型架构总览

nanochat 的 GPT 模型是一个**现代化的 decoder-only Transformer**，与经典 GPT-2 相比有以下显著改进：

| 特性 | GPT-2 (经典) | nanochat GPT |
|------|-------------|-------------|
| 位置编码 | 可学习绝对位置编码 | **RoPE (旋转位置编码)** |
| 注意力 | 标准 Multi-Head | **GQA + QK Norm + 滑动窗口** |
| 激活函数 | GELU | **ReLU²** (ReLU平方) |
| 归一化 | LayerNorm (带参数) | **RMSNorm (无参数)** |
| 线性层 | 有 bias | **无 bias** |
| 嵌入权重 | tied (共享) | **untied (独立)** |
| 残差连接 | 固定 | **可学习标量 (resid_lambdas, x0_lambdas)** |
| Value 嵌入 | 无 | **ResFormer 风格 Value Embedding** |

文件位置：`nanochat/gpt.py`

---

## 2. GPTConfig：配置即设计

```python
@dataclass
class GPTConfig:
    sequence_len: int = 2048      # 上下文长度
    vocab_size: int = 32768       # 词汇表大小
    n_layer: int = 12             # Transformer 层数 (即 --depth)
    n_head: int = 6               # Query 头数
    n_kv_head: int = 6            # Key/Value 头数 (GQA)
    n_embd: int = 768             # 嵌入维度
    window_pattern: str = "SSSL"  # 滑动窗口模式
```

**核心设计理念：一个旋钮 `--depth` 控制一切。** 所有其他超参数由 depth 自动推导：

```
model_dim = depth × aspect_ratio (默认64)    # 模型宽度
num_heads = model_dim / head_dim (默认128)    # 注意力头数
```

例如 depth=20 → model_dim=1280, num_heads=10。

---

## 3. 核心组件详解

### 3.1 自定义 Linear 层

```python
class Linear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))
```

**为什么不用 autocast？** nanochat 的精度管理策略是"显式优于隐式"：
- **权重**：以 fp32 存储（保证优化器精度）
- **前向计算**：在 forward 时将权重 cast 到输入的 dtype（通常是 bf16）
- 这替代了 `torch.amp.autocast`，给了开发者完全的精度控制权

### 3.2 RMSNorm（无参数版本）

```python
def norm(x):
    return F.rms_norm(x, (x.size(-1),))
```

极简实现——没有可学习的 scale/bias 参数。RMSNorm 比 LayerNorm 更快，因为省去了均值计算。

### 3.3 旋转位置编码 (RoPE)

```python
def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]   # 将最后一维切成两半
    y1 = x1 * cos + x2 * sin           # 旋转
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)
```

**RoPE 的核心思想**：将位置信息编码为旋转角度，使得两个 token 之间的注意力分数只取决于它们的**相对距离**。

预计算方式：
```python
inv_freq = 1.0 / (base ** (channel_range / head_dim))  # 不同维度有不同频率
freqs = torch.outer(t, inv_freq)                         # 位置 × 频率
cos, sin = freqs.cos(), freqs.sin()
```

### 3.4 因果自注意力 (CausalSelfAttention)

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        # Q、K、V 投影 + 输出投影，全部无 bias
        self.c_q = Linear(n_embd, n_head * head_dim, bias=False)
        self.c_k = Linear(n_embd, n_kv_head * head_dim, bias=False)
        self.c_v = Linear(n_embd, n_kv_head * head_dim, bias=False)
        self.c_proj = Linear(n_embd, n_embd, bias=False)
        # Value Embedding 门控 (仅交替层)
        self.ve_gate = Linear(ve_gate_channels, n_kv_head, bias=False)
```

**关键特性**：

**a) QK Norm**：对 Q 和 K 做 RMSNorm 后再计算注意力，稳定训练
```python
q, k = norm(q), norm(k)
```

**b) Value Residual (ResFormer)**：在交替的层中，将 token embedding 查表得到的 value embedding 混入 V 中
```python
if ve is not None:
    gate = 2 * torch.sigmoid(self.ve_gate(x[..., :32]))  # 范围 (0, 2)
    v = v + gate.unsqueeze(-1) * ve
```

**c) 滑动窗口注意力**：通过 `window_pattern` 字符串控制每层的注意力窗口
- `S` = Short，只看一半上下文（sequence_len/2）
- `L` = Long，看全部上下文
- 默认 `"SSSL"` = 三短一长交替，最后一层强制为 L
- 减少计算量，同时最后一层保持全局视野

**d) Flash Attention**：在 Hopper GPU 上使用 FA3，其他硬件回退到 PyTorch SDPA

### 3.5 MLP (前馈网络)

```python
class MLP(nn.Module):
    def __init__(self, config):
        self.c_fc = Linear(n_embd, 4 * n_embd, bias=False)    # 扩展 4 倍
        self.c_proj = Linear(4 * n_embd, n_embd, bias=False)  # 投影回来

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()   # ReLU² 激活
        x = self.c_proj(x)
        return x
```

**ReLU²** 相比 GELU 的优势：更简单、稀疏性更好（ReLU 后约一半为零），平方操作增加了非线性。

### 3.6 Transformer Block

```python
class Block(nn.Module):
    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)  # Pre-Norm + 残差
        x = x + self.mlp(norm(x))                                        # Pre-Norm + 残差
        return x
```

采用 **Pre-Norm** 架构（先归一化再计算），比 Post-Norm 更稳定。

---

## 4. 前向传播流程

```python
def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
    # 1. Token Embedding + RMSNorm
    x = self.transformer.wte(idx)
    x = norm(x)
    x0 = x  # 保存初始嵌入用于 x0 残差

    # 2. 通过所有 Transformer 层
    for i, block in enumerate(self.transformer.h):
        # 可学习的残差混合：resid_lambda * x + x0_lambda * x0
        x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        # 查表获取 Value Embedding (交替层)
        ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
        x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)

    # 3. 最终 RMSNorm + LM Head
    x = norm(x)
    logits = self.lm_head(x)
    logits = logits[..., :self.config.vocab_size]  # 裁掉 padding

    # 4. Logit Softcap（防止 logit 爆炸）
    softcap = 20
    logits = softcap * torch.tanh(logits / softcap)

    # 5. 计算 loss 或返回 logits
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss
    return logits
```

**x0 残差连接**：每一层都可以直接"看到"初始嵌入，类似于 DenseNet 的思想。`resid_lambdas` 和 `x0_lambdas` 是可学习标量，初始值分别为 1.0 和 0.1。

**Logit Softcap**：用 `tanh` 将 logits 平滑地限制在 [-20, 20] 范围内，防止极端值。

---

## 5. 权重初始化策略

```python
def init_weights(self):
    # Embedding: 标准正态 (std=1.0)
    torch.nn.init.normal_(self.transformer.wte.weight, std=1.0)
    # LM Head: 极小正态 (std=0.001)，输出初始近似均匀
    torch.nn.init.normal_(self.lm_head.weight, std=0.001)
    # Transformer 权重: 均匀分布 (std = 1/√n_embd)
    s = 3**0.5 * n_embd**-0.5   # Uniform bounds，使 std 与 Normal 一致
    for block in self.transformer.h:
        uniform_(block.attn.c_q.weight, -s, s)
        uniform_(block.attn.c_k.weight, -s, s)
        uniform_(block.attn.c_v.weight, -s, s)
        zeros_(block.attn.c_proj.weight)     # 投影初始化为零!
        uniform_(block.mlp.c_fc.weight, -s, s)
        zeros_(block.mlp.c_proj.weight)      # 投影初始化为零!
    # 残差标量
    self.resid_lambdas.fill_(1.0)   # 1.0 = 正常残差
    self.x0_lambdas.fill_(0.1)      # 0.1 = 初始少量使用 skip connection
    # VE Gate: 零初始化 → sigmoid(0)=0.5, ×2=1.0 → 中性起点
```

**关键设计**：
- 投影层 (`c_proj`) 初始化为零，这意味着在训练开始时每个 Block 的输出为零，模型从"恒等映射"开始
- 使用均匀分布而非正态分布，避免离群值
- `√3` 因子确保均匀分布的标准差与正态分布一致

---

## 6. 优化器：MuonAdamW

nanochat 使用**混合优化器**，不同类型的参数用不同的优化策略：

| 参数类型 | 优化器 | 学习率 |
|---------|--------|--------|
| LM Head (unembedding) | AdamW | 0.004 × scale |
| Token Embedding | AdamW | 0.3 × scale |
| Value Embedding | AdamW | 0.3 × scale |
| resid_lambdas | AdamW | 0.005 |
| x0_lambdas | AdamW | 0.5 |
| Transformer 矩阵权重 | **Muon** | 0.02 × scale |

其中 `scale = (model_dim / 768)^(-0.5)` 是 muP 风格的学习率缩放。

### Muon 优化器核心思想

Muon = **Mo**ment**u**m **O**rthogonalized by **N**ewton-schulz

1. **Nesterov 动量**：标准动量更新
2. **Polar Express 正交化**：将梯度矩阵正交化为最近的正交矩阵（类似 SVD 中的 UV^T）
3. **NorMuon 方差归约**：逐神经元自适应学习率
4. **Cautious 权重衰减**：只在梯度方向与参数方向一致时才衰减

正交化的直觉：让每一步更新尽可能地"均匀"影响所有方向，而不是只集中在少数方向上。

### 分布式版本 (DistMuonAdamW)

采用 **ZeRO-2 风格**的通信策略：
- **3 阶段异步通信**：reduce → compute → gather，最大化通信与计算的重叠
- **小参数**：all_reduce 梯度，全量更新
- **大参数**：reduce_scatter 梯度分片 → 每个 rank 只更新自己那份 → all_gather 回来

---

## 7. 预训练流程详解

文件位置：`scripts/base_train.py`

### 7.1 启动命令

```bash
# 单 GPU
python -m scripts.base_train --depth=12

# 8 GPU 分布式
torchrun --nproc_per_node=8 -m scripts.base_train --depth=20
```

### 7.2 模型初始化（三步法）

```python
# 1) 在 meta device 上构建（只有 shape/dtype，没有数据）
with torch.device("meta"):
    model_meta = GPT(config)

# 2) 移到真实设备（分配内存，但数据是垃圾值）
model.to_empty(device=device)

# 3) 初始化权重
model.init_weights()
```

**为什么用 meta device？** 避免在 CPU 上分配大量内存再拷贝到 GPU，节省峰值内存。

### 7.3 torch.compile

```python
model = torch.compile(model, dynamic=False)
```

`dynamic=False` 因为输入 shape 永远不变（固定 batch_size × sequence_len），允许编译器做更激进的优化。

### 7.4 学习率调度

```python
def get_lr_multiplier(it):
    # 线性 warmup → 恒定 → 线性 warmdown
    warmup_iters = warmup_ratio * num_iterations    # 默认 0%
    warmdown_iters = warmdown_ratio * num_iterations # 默认 50%
```

默认是**无 warmup + 后半程线性衰减到 0**。

### 7.5 训练循环

```python
while True:
    # 1. 评估（每 eval_every 步）
    if step % eval_every == 0:
        val_bpb = evaluate_bpb(model, val_loader, ...)

    # 2. CORE 指标（每 core_metric_every 步）
    if step % core_metric_every == 0:
        results = evaluate_core(model, tokenizer, ...)

    # 3. 保存 checkpoint

    # 4. 梯度累积
    for micro_step in range(grad_accum_steps):
        loss = model(x, y)
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, state = next(train_loader)  # 预取下一批

    # 5. 更新学习率、动量、权重衰减
    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm

    # 6. 优化器步进
    optimizer.step()
    model.zero_grad(set_to_none=True)
```

**梯度累积**：当 GPU 内存不够放完整 batch 时，分多个 micro-step 累积梯度，等效于大 batch 训练。

### 7.6 GC 优化

```python
if first_step_of_run:
    gc.collect()    # 手动回收初始化产生的垃圾
    gc.freeze()     # 冻结所有存活对象
    gc.disable()    # 完全禁用 GC！
elif step % 5000 == 0:
    gc.collect()    # 每 5000 步手动回收一次
```

Python GC 在训练中会花 ~500ms 扫描，但只清理很少的对象。禁用后可以避免这个开销。

---

## 8. 数据加载器

文件位置：`nanochat/dataloader.py`

### BOS-Aligned Best-Fit 策略

```
传统方式：将文档 token 连续拼接到行中
  [doc1_tokens...doc2_tokens...doc3_tok|ens_cut_here...]

nanochat 方式：每行以 BOS 开头，尽量装满完整文档
  [BOS doc1_tokens BOS doc3_tokens BOS doc2_cropped]
```

算法：
1. 维护一个 buffer（默认 1000 个文档）
2. 对每行，贪心选择 buffer 中**最大的能完整放入的文档**
3. 重复直到放不下更多文档
4. 用最短文档的截断版填满剩余空间

**优势**：每个 token 都能看到完整的文档上下文（回看到 BOS）
**代价**：约 35% 的 token 被截断丢弃（但训练效果更好）

### 分布式分片

多 GPU 训练时，每个 rank 读取不同的 row group：
```python
rg_idx = ddp_rank                    # 起始位置
while rg_idx < pf.num_row_groups:
    rg = pf.read_row_group(rg_idx)
    ...
    rg_idx += ddp_world_size          # 步长 = world_size
```

---

## 9. Scaling Laws 与超参数自动推导

这是 nanochat 最精妙的设计之一——**所有超参数都从 depth 自动推导**。

### 9.1 训练 Token 数量

```python
# 最优 Token 数 = target_param_data_ratio × scaling_params
# 默认 ratio = 10.5（Chinchilla 用的是 20）
target_tokens = 10.5 * (transformer_matrices + lm_head_params)
```

### 9.2 Batch Size

基于 [Power Lines 论文](https://arxiv.org/abs/2505.13738)：
```python
# B_opt ∝ D^0.383 (D = 训练 token 数)
batch_ratio = target_tokens / D_REF   # 相对 d12 参考模型
predicted_batch = B_REF * batch_ratio ** 0.383
total_batch_size = 2 ** round(log2(predicted_batch))  # 取最近 2 的幂
```

### 9.3 学习率缩放

```python
# 两种缩放叠加：
# 1. muP 缩放：lr ∝ 1/√(model_dim/768)   ← 从 d12 参考模型迁移
# 2. Batch 缩放：lr ∝ √(B/B_ref)          ← 大 batch 允许更大 lr
```

### 9.4 权重衰减缩放

基于 [T_epoch 框架](https://arxiv.org/abs/2405.13698)：
```python
# λ = λ_ref × √(B/B_ref) × (D_ref/D)
# 保持 T_epoch = B/(η·λ·D) 不变
weight_decay_scaled = wd * sqrt(B/B_ref) * (D_ref/D)
```

### 推导链总结

```
depth (用户输入)
  ├→ model_dim = depth × 64
  ├→ num_heads = model_dim / 128
  ├→ num_params (由架构决定)
  ├→ target_tokens = 10.5 × scaling_params
  ├→ batch_size ∝ target_tokens^0.383
  ├→ learning_rate ∝ 1/√model_dim × √batch_size
  └→ weight_decay ∝ √batch_size / target_tokens
```

---

## 10. 关键设计理念总结

### 10.1 极简主义
- 没有配置文件系统，没有模型工厂，没有 if-else 怪物
- 整个模型在一个 ~460 行的文件中
- 训练脚本是一个直白的循环

### 10.2 显式优于隐式
- 不用 autocast，手动管理精度
- 不用 DDP wrapper，手动管理通信
- 所有初始化集中在一个函数中

### 10.3 一个旋钮控制一切
- `--depth` 是唯一的"复杂度旋钮"
- 所有超参数通过 scaling laws 和 muP 自动推导
- 任何代码改进都必须对所有 depth 值都有效

### 10.4 性能优化的优先级
1. **算法优化**（更好的架构、优化器）> 2. **系统优化**（Flash Attention、torch.compile）> 3. **微优化**（GC 管理、内存对齐）

### 10.5 关键指标
- `val_bpb`：验证集 bits per byte（越低越好）
- `CORE metric`：DCLM 的 CORE 评分（衡量模型能力）
- `MFU`：Model FLOPS Utilization（GPU 利用率）
- `tok/sec`：训练吞吐量

---

## 动手实验建议

```bash
# 1. 快速实验：训练一个 d12 模型（~5分钟，需要 GPU）
torchrun --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 --run="d12_experiment" \
    --core-metric-every=999999 --sample-every=-1 --save-every=-1

# 2. CPU 上跑一个小模型感受流程
python -m scripts.base_train --depth=4 --max-seq-len=512 \
    --device-batch-size=1 --eval-tokens=512 \
    --core-metric-every=-1 --total-batch-size=512 --num-iterations=20

# 3. 完整 GPT-2 复现（~2-3小时，需要 8×H100）
bash runs/speedrun.sh
```
