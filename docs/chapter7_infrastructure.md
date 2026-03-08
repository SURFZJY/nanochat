# 第七章：基础设施

本章学习 nanochat 的基础设施模块——优化器、FP8 训练、Flash Attention、Checkpoint 管理和报告生成。

---

## 目录

1. [优化器：MuonAdamW](#1-优化器muonadamw)
2. [FP8 训练](#2-fp8-训练)
3. [Flash Attention 适配层](#3-flash-attention-适配层)
4. [Checkpoint 管理](#4-checkpoint-管理)
5. [计算环境初始化](#5-计算环境初始化)
6. [报告生成系统](#6-报告生成系统)
7. [关键设计理念总结](#7-关键设计理念总结)

---

## 1. 优化器：MuonAdamW

文件位置：`nanochat/optim.py`

### 1.1 混合优化器设计

nanochat 使用 **Muon + AdamW 混合优化器**，不同参数类型使用不同的优化算法：

| 参数类型 | 优化器 | 原因 |
|---------|--------|------|
| Embedding / Scalar | AdamW | 低维参数适合标准自适应优化 |
| Transformer 矩阵权重 | Muon | 2D 矩阵参数受益于正交化更新 |

### 1.2 AdamW 实现

```python
@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq,
                     step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    """编译为单个 CUDA kernel，消除 Python 开销"""
    # 1. 解耦权重衰减
    p.mul_(1 - lr_t * wd_t)
    # 2. 更新一阶和二阶矩
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    # 3. 偏差校正 + 参数更新
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    p.add_(exp_avg / denom, alpha=-lr_t / bias1)
```

**0-D CPU Tensor 技巧**：所有超参数（lr、beta1 等）都用 0-D CPU tensor 传递，避免 `torch.compile` 在超参数变化时重新编译。

### 1.3 Muon 优化器

Muon = **Mo**ment**u**m **O**rthogonalized by **N**ewton-schulz

核心步骤：

```python
@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, ...):
    # 步骤 1: Nesterov 动量
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # 步骤 2: Polar Express 正交化（5 次迭代）
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    for a, b, c in polar_express_coeffs[:ns_steps]:
        A = X.mT @ X          # 或 X @ X.mT（取决于矩阵方向）
        B = b * A + c * (A @ A)
        X = a * X + X @ B     # 或 B @ X

    # 步骤 3: NorMuon 方差归约（逐神经元自适应学习率）
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    second_momentum_buffer.lerp_(v_mean, 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    g = g * final_scale

    # 步骤 4: Cautious 权重衰减 + 更新
    mask = (g * stacked_params) >= 0   # 只在梯度方向一致时衰减
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)
```

### Polar Express 正交化

```
目标：将梯度矩阵 G 变换为最近的正交矩阵 UV^T（其中 USV^T = G 的 SVD）

直觉：让更新"均匀"地影响所有方向，而非集中在少数大奇异值方向

方法：5 次迭代的多项式近似，比直接 SVD 更快（O(n²) vs O(n³)）

论文：Polar Express Sign Method (arXiv:2505.16932)
```

### NorMuon 方差归约

正交化后不同神经元的更新尺度不同。NorMuon 通过逐神经元（per-column 或 per-row）的自适应缩放来平衡：

```python
# 沿 reduction 维度计算方差
v_mean = g.square().mean(dim=red_dim, keepdim=True)
# 用二阶矩缓冲做指数平均
second_momentum_buffer.lerp_(v_mean, 1 - beta2)
# 归一化
step_size = second_momentum_buffer.rsqrt()
```

### 1.4 分布式版本 (DistMuonAdamW)

```
3 阶段异步通信：

Phase 1: 发起所有 async reduce 操作
  └── AdamW: 小参数 all_reduce / 大参数 reduce_scatter
  └── Muon:  stacked reduce_scatter

Phase 2: 等待 reduce → 计算更新 → 发起 gather
  └── 通过顺序处理，早期 gather 与后期计算重叠

Phase 3: 等待所有 gather → 复制回原参数
```

**ZeRO-2 风格分片**：
- **AdamW 大参数**：reduce_scatter 梯度 → 每个 rank 只更新自己的分片 → all_gather 回来
- **Muon**：将 K 个参数 stack 为一个大 tensor → reduce_scatter → 每个 rank 更新 ceil(K/N) 个参数 → all_gather

---

## 2. FP8 训练

文件位置：`nanochat/fp8.py`

### 2.1 FP8 原理

FP8 训练将 Linear 层的 3 个矩阵乘法用 FP8 精度执行，速度约 2 倍：

```
Forward:   output = input @ weight.T        ← FP8
Backward:  grad_input = grad_output @ weight ← FP8
           grad_weight = grad_output.T @ input ← FP8
```

### 2.2 两种 FP8 格式

| 格式 | 指数位 | 尾数位 | 范围 | 用途 |
|------|--------|--------|------|------|
| float8_e4m3fn | 4 | 3 | [-448, 448] | **输入和权重**（高精度） |
| float8_e5m2 | 5 | 2 | [-57344, 57344] | **梯度**（大动态范围） |

### 2.3 动态量化

```python
@torch.no_grad()
def _to_fp8(x, fp8_dtype):
    """Tensorwise 动态量化"""
    fp8_max = torch.finfo(fp8_dtype).max
    # 计算缩放因子
    amax = x.float().abs().max()
    scale = fp8_max / amax.double().clamp(min=EPS)  # float64 保精度
    # 量化
    x_scaled = x.float() * scale
    x_clamped = x_scaled.clamp(-fp8_max, fp8_max)   # 饱和（非溢出）
    x_fp8 = x_clamped.to(fp8_dtype)
    return x_fp8, scale.reciprocal()  # 返回反向缩放因子
```

**Tensorwise vs Rowwise**：nanochat 只用 tensorwise（一个 tensor 一个 scale），比 rowwise 更简单且用 cuBLAS kernel（更快）。

### 2.4 自定义 Autograd Function

```python
@torch._dynamo.allow_in_graph  # 让 torch.compile 视为黑盒
class _Float8Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_2d, weight):
        input_fp8, input_inv = _to_fp8(input_2d, torch.float8_e4m3fn)
        weight_fp8, weight_inv = _to_fp8(weight, torch.float8_e4m3fn)
        ctx.save_for_backward(input_fp8, input_inv, weight_fp8, weight_inv)
        output = torch._scaled_mm(
            input_fp8, weight_fp8.t(),
            scale_a=input_inv, scale_b=weight_inv,
            out_dtype=input_2d.dtype,
            use_fast_accum=True,      # Forward 用 fast accum
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # GEMM 1: grad_input = grad_output @ weight
        go_fp8, go_inv = _to_fp8(grad_output, torch.float8_e5m2)  # 梯度用 e5m2
        grad_input = torch._scaled_mm(go_fp8, w_col, ..., use_fast_accum=False)

        # GEMM 2: grad_weight = grad_output.T @ input
        grad_weight = torch._scaled_mm(go_T, in_col, ..., use_fast_accum=False)
        return grad_input, grad_weight
```

### 2.5 与 torchao 的对比

| | nanochat fp8.py | torchao Float8Linear |
|---|---|---|
| 代码量 | ~150 行 | ~2000 行 |
| 方法 | 单个 `autograd.Function` | Tensor subclass + dispatch table |
| compile 可见性 | 黑盒节点 | 分解为独立 op，可融合 |
| 性能 | 略快（无 dispatch 开销） | 理论上更优（可融合 amax） |
| 支持配方 | 仅 tensorwise | tensorwise + rowwise + axiswise |

两者调用同一个 cuBLAS `_scaled_mm` kernel，实际矩阵乘法完全相同。

### 2.6 模型转换

```python
def convert_to_float8_training(module, config=None, module_filter_fn=None):
    """将 nn.Linear 替换为 Float8Linear"""
    def _convert(mod, prefix=""):
        for name, child in mod.named_children():
            _convert(child, fqn)
            if isinstance(child, nn.Linear) and not isinstance(child, Float8Linear):
                if module_filter_fn is None or module_filter_fn(child, fqn):
                    setattr(mod, name, Float8Linear.from_float(child))
    _convert(module)
```

`from_float` 通过 meta device 创建新模块，然后共享原始的 weight/bias 参数——零内存开销。

---

## 3. Flash Attention 适配层

文件位置：`nanochat/flash_attention.py`

### 3.1 自动硬件切换

```python
def _load_flash_attention_3():
    """只在 Hopper (sm90) 上加载 FA3"""
    major, _ = torch.cuda.get_device_capability()
    if major != 9:  # Ada(89), Blackwell(100) 都不行
        return None
    from kernels import get_kernel
    return get_kernel('varunneal/flash-attention-3').flash_attn_interface

_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None
```

| GPU 架构 | 实现 |
|---------|------|
| Hopper (sm90): H100/H200 | Flash Attention 3 |
| Ada (sm89): L40S/4090 | PyTorch SDPA |
| Blackwell (sm100): B200 | PyTorch SDPA（等 FA3 支持） |
| CPU / MPS | PyTorch SDPA |

### 3.2 统一 API

```python
# 训练用（无 KV Cache）
y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

# 推理用（有 KV Cache）
y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
```

调用方不需要知道底层用的是 FA3 还是 SDPA。

### 3.3 滑动窗口支持

SDPA 不原生支持滑动窗口，nanochat 手动构建掩码：

```python
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    # 全上下文 + 相同长度 → 直接 is_causal=True
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    # 单 token 生成 → 截断 KV cache
    if Tq == 1:
        start = max(0, Tk - (window + 1))
        k, v = k[:, :, start:, :], v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False)

    # 通用情况 → 显式构建 bool mask
    mask = col_idx <= row_idx                       # 因果掩码
    mask = mask & ((row_idx - col_idx) <= window)   # 滑动窗口
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
```

---

## 4. Checkpoint 管理

文件位置：`nanochat/checkpoint_manager.py`

### 4.1 文件结构

```
~/.cache/nanochat/
├── base_checkpoints/d24/
│   ├── model_005000.pt          # 模型参数 (rank 0 保存)
│   ├── meta_005000.json         # 元数据（模型配置、训练超参数）
│   ├── optim_005000_rank0.pt    # 优化器状态分片 (rank 0)
│   ├── optim_005000_rank1.pt    # 优化器状态分片 (rank 1)
│   └── ...
├── chatsft_checkpoints/d24/
│   └── ...
└── chatrl_checkpoints/d24/
    └── ...
```

### 4.2 保存与加载

```python
def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    if rank == 0:
        torch.save(model_data, f"model_{step:06d}.pt")
        json.dump(meta_data, f"meta_{step:06d}.json")
    # 优化器每个 rank 保存自己的分片
    if optimizer_data is not None:
        torch.save(optimizer_data, f"optim_{step:06d}_rank{rank}.pt")
```

### 4.3 模型构建

```python
def build_model(checkpoint_dir, step, device, phase):
    # 1. 加载 checkpoint
    model_data, _, meta_data = load_checkpoint(checkpoint_dir, step, device)

    # 2. 修复 torch.compile 添加的前缀
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

    # 3. 补丁缺失的配置键（向后兼容）
    _patch_missing_config_keys(model_config_kwargs)  # 如 window_pattern
    _patch_missing_keys(model_data, model_config)     # 如 resid_lambdas

    # 4. Meta device 构建 → 加载权重
    with torch.device("meta"):
        model = GPT(model_config)
    model.to_empty(device=device)
    model.init_weights()  # 需要初始化 RoPE 缓冲
    model.load_state_dict(model_data, strict=True, assign=True)

    # 5. CPU 推理时将 bf16 转为 fp32
    if device.type in {"cpu", "mps"}:
        model_data = {k: v.float() if v.dtype == torch.bfloat16 else v ...}
```

### 4.4 自动模型发现

```python
def find_largest_model(checkpoints_dir):
    """自动选择最大的模型"""
    # 优先：按 d<number> 格式找最大 depth
    candidates = [(int(match.group(1)), tag) for tag in model_tags]
    # 备选：按最近修改时间
    model_tags.sort(key=lambda x: os.path.getmtime(...), reverse=True)

def find_last_step(checkpoint_dir):
    """找到最新的 checkpoint step"""
    files = glob.glob("model_*.pt")
    return max(int(f.split("_")[-1].split(".")[0]) for f in files)
```

### 4.5 统一加载接口

```python
def load_model(source, device, phase, model_tag=None, step=None):
    """source = "base" | "sft" | "rl" """
    model_dir = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    return load_model_from_dir(checkpoints_dir, device, phase, model_tag, step)
```

---

## 5. 计算环境初始化

文件位置：`nanochat/common.py`

### 5.1 精度自动检测

```python
def _detect_compute_dtype():
    env = os.environ.get("NANOCHAT_DTYPE")
    if env is not None:
        return _DTYPE_MAP[env]           # 环境变量覆盖
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability >= (8, 0):
            return torch.bfloat16        # Ampere+ 用 bf16
        return torch.float32             # 更旧的 GPU 用 fp32
    return torch.float32                 # CPU/MPS 用 fp32
```

### 5.2 分布式初始化

```python
def compute_init(device_type="cuda"):
    # 1. 可复现性
    torch.manual_seed(42)

    # 2. 精度设置
    torch.set_float32_matmul_precision("high")  # 用 TF32

    # 3. 分布式
    if is_ddp_requested and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
```

### 5.3 GPU 峰值 FLOPS 表

```python
_PEAK_FLOPS_TABLE = (
    (["gb200"], 2.5e15),       # Blackwell
    (["b200"], 2.25e15),
    (["h100"], 989e12),        # Hopper
    (["a100"], 312e12),        # Ampere
    (["4090"], 165.2e12),      # Consumer
    ...
)
```

用于计算 **MFU (Model FLOPS Utilization)**：`MFU = actual_flops / peak_flops`。

### 5.4 其他工具

- `print0()`：只在 rank 0 打印
- `DummyWandb`：不使用 wandb 时的空实现
- `download_file_with_lock()`：带文件锁的下载（防止多 rank 并发下载冲突）
- `autodetect_device_type()`：CUDA > MPS > CPU 优先级

---

## 6. 报告生成系统

文件位置：`nanochat/report.py`

### 6.1 报告结构

```
~/.cache/nanochat/report/
├── header.md                    # 环境信息、时间戳
├── tokenizer-training.md        # 分词器训练
├── tokenizer-evaluation.md      # 分词器评估
├── base-model-training.md       # 预训练
├── base-model-loss.md           # 预训练 loss
├── base-model-evaluation.md     # 基座评估
├── chat-sft.md                  # SFT 训练
├── chat-evaluation-sft.md       # SFT 评估
├── chat-rl.md                   # RL 训练
├── chat-evaluation-rl.md        # RL 评估
└── report.md                    # 合并后的最终报告
```

### 6.2 分段日志

```python
class Report:
    def log(self, section, data):
        """每个训练阶段调用一次，追加到对应的 section 文件"""
        slug = slugify(section)  # e.g. "SFT" → "sft.md"
        file_path = os.path.join(self.report_dir, f"{slug}.md")
        with open(file_path, "w") as f:
            f.write(f"## {section}\n")
            f.write(f"timestamp: {datetime.now()}\n\n")
            for item in data:
                if isinstance(item, str):
                    f.write(item)
                else:
                    for k, v in item.items():
                        f.write(f"- {k}: {v}\n")
```

### 6.3 报告生成

```python
def generate(self):
    """合并所有 section 文件为最终报告"""
    # 1. 写入 header
    # 2. 按顺序合并各 section
    # 3. 提取关键指标：CORE、ChatCORE 等
    # 4. 生成 Summary 表格
    # 5. 计算总训练时间
```

### 6.4 Summary 表格

```markdown
| Metric          | BASE     | SFT      | RL       |
|-----------------|----------|----------|----------|
| CORE            | 0.2345   | -        | -        |
| ARC-Easy        | -        | 0.6789   | -        |
| GSM8K           | -        | 0.3456   | 0.5678   |
| ChatCORE        | -        | 0.4567   | -        |

Total wall clock time: 2h15m
```

### 6.5 Bloat 度量

报告还会统计代码库的"膨胀"指标：

```python
# 统计 git 跟踪的源文件
extensions = ['py', 'md', 'rs', 'html', 'toml', 'sh']
# 字符数、行数、文件数
# Token 数 ≈ 字符数 / 4
# 依赖数 = uv.lock 行数
```

### 6.6 使用方式

```bash
# 重置报告（开始新的训练运行）
python -m nanochat.report reset

# 生成最终报告
python -m nanochat.report generate
```

每个训练脚本的末尾都会自动调用 `get_report().log()`。

---

## 7. 关键设计理念总结

### 7.1 一切皆编译

- AdamW 和 Muon 的核心步骤都用 `@torch.compile` 编译为单个 CUDA kernel
- 0-D CPU tensor 传递超参数，避免重新编译
- FP8 用 `@allow_in_graph` 作为编译图中的黑盒节点

### 7.2 极简替代

- **FP8**：150 行替代 torchao 的 2000 行，调用同一个 cuBLAS kernel
- **Flash Attention**：统一接口，FA3/SDPA 自动切换
- **Checkpoint**：简单的 `torch.save` + JSON，无复杂的序列化框架

### 7.3 向后兼容

```python
# 新版本添加了 window_pattern，但旧 checkpoint 没有
if "window_pattern" not in model_config_kwargs:
    model_config_kwargs["window_pattern"] = "L"  # 默认全局注意力

# 新版本添加了 resid_lambdas，但旧 checkpoint 没有
if "resid_lambdas" not in model_data:
    model_data["resid_lambdas"] = torch.ones(n_layer)  # 默认恒等
```

### 7.4 分布式优先

- 优化器有 Single-GPU 和 Distributed 两个版本
- Checkpoint 的优化器状态天然分片（每个 rank 保存自己的分片）
- 下载使用文件锁（`FileLock`）防止多 rank 冲突
- `print0` 确保只有 rank 0 输出日志

### 7.5 可观测性

- wandb 集成（可选，`--run=dummy` 禁用）
- 报告系统自动收集所有训练阶段的指标
- MFU 计算让你知道 GPU 利用率
- Bloat 指标让你关注代码膨胀

---

## 动手实验建议

```bash
# 1. 查看精度检测
python -c "
from nanochat.common import COMPUTE_DTYPE, COMPUTE_DTYPE_REASON
print(f'Compute dtype: {COMPUTE_DTYPE}')
print(f'Reason: {COMPUTE_DTYPE_REASON}')
"

# 2. 查看 GPU FLOPS
python -c "
import torch
from nanochat.common import get_peak_flops
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    flops = get_peak_flops(name)
    print(f'{name}: {flops:.2e} FLOPS')
"

# 3. 生成报告
python -m nanochat.report generate

# 4. 查看 Flash Attention 状态
python -c "
from nanochat.flash_attention import HAS_FA3, USE_FA3
print(f'FA3 available: {HAS_FA3}')
print(f'Using FA3: {USE_FA3}')
"
```
