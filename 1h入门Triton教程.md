# Triton 1小时快速入门教程

> 面向Python程序员的Triton GPU编程入门指南

---

## 📋 教程大纲

| 时间 | 内容 | 目标 |
|------|------|------|
| 0-10分钟 | 环境搭建 & 核心概念 | 理解Triton是什么 |
| 10-25分钟 | 第一个Kernel：向量加法 | 掌握基本语法 |
| 25-40分钟 | 实战：矩阵乘法 | 理解分块计算 |
| 40-55分钟 | 优化技巧 & Softmax实现 | 掌握常用模式 |
| 55-60分钟 | 资源 & 下一步学习 | 持续学习路径 |

---

## 第1部分：环境 & 核心概念（10分钟）

### 1.1 安装

```bash
pip install triton torch
```

### 1.2 Triton 是什么？

```
传统GPU编程：Python → CUDA C++ → GPU（学习成本高）
Triton编程：  Python → Triton → GPU（Python风格，自动优化）
```

### 1.3 核心概念速记

```python
"""
🔑 三个核心概念：

1. Program（程序实例）
   - 每个program处理数据的一个"块"
   - 类似于CUDA的block

2. Block（数据块）
   - Triton按块处理数据（如128/256/512个元素）
   - 自动处理边界情况

3. tl（triton.language）
   - Triton的核心API库
   - 提供load/store/计算等操作
"""
```

---

## 第2部分：第一个Kernel - 向量加法（15分钟）

### 2.1 完整代码

```python
import torch
import triton
import triton.language as tl

# ============ Triton Kernel ============
@triton.jit  # 👈 核心装饰器，将Python函数编译为GPU代码
def add_kernel(
    x_ptr,      # 输入指针
    y_ptr,      # 输入指针
    out_ptr,    # 输出指针
    n_elements, # 元素总数
    BLOCK_SIZE: tl.constexpr,  # 👈 编译时常量
):
    # 1️⃣ 获取当前program的ID（类似CUDA的blockIdx）
    pid = tl.program_id(axis=0)
    
    # 2️⃣ 计算当前block处理的元素索引
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 3️⃣ 创建mask处理边界（防止越界访问）
    mask = offsets < n_elements
    
    # 4️⃣ 从GPU内存加载数据
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # 5️⃣ 计算
    output = x + y
    
    # 6️⃣ 写回GPU内存
    tl.store(out_ptr + offsets, output, mask=mask)


# ============ Python包装函数 ============
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # 确保输入在GPU上
    assert x.is_cuda and y.is_cuda
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    # 计算需要多少个program（grid大小）
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)  # 向上取整除法
    
    # 启动kernel
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE)
    
    return output


# ============ 测试 ============
if __name__ == "__main__":
    torch.manual_seed(0)
    size = 98432  # 故意不是2的幂次，测试边界处理
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')
    
    # 对比测试
    triton_output = add(x, y)
    torch_output = x + y
    
    print(f"✅ 结果正确: {torch.allclose(triton_output, torch_output)}")
    print(f"最大误差: {(triton_output - torch_output).abs().max()}")
```

### 2.2 关键语法解析

```python
"""
📌 @triton.jit 内部可用的操作：

位置计算：
  tl.program_id(axis)    → 获取当前program在指定轴的ID
  tl.arange(start, end)  → 创建连续整数序列 [start, end)

内存操作：
  tl.load(ptr, mask)     → 从GPU内存加载数据
  tl.store(ptr, val, mask) → 写入GPU内存

数学运算：
  +, -, *, /             → 逐元素运算
  tl.exp, tl.log, tl.sin → 数学函数
  tl.max, tl.sum         → 归约操作

特殊：
  tl.constexpr           → 标记编译时常量（如BLOCK_SIZE）
"""
```

---

## 第3部分：矩阵乘法（15分钟）

### 3.1 分块思想图解

```
矩阵 A (M×K) @ 矩阵 B (K×N) = 矩阵 C (M×N)

分块计算策略：
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ A块 (BM×BK) │  @  │ B块 (BK×BN) │  =  │ C块 (BM×BN) │
└─────────────┘     └─────────────┘     └─────────────┘
      ↓                   ↓                   ↓
   逐块加载            逐块加载            累加结果
```

### 3.2 简化版矩阵乘法

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,  # A的步长
    stride_bk, stride_bn,  # B的步长
    stride_cm, stride_cn,  # C的步长
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 1️⃣ 确定当前program负责C的哪个块
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 2️⃣ 计算块内偏移
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # 3️⃣ 初始化累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # 4️⃣ 沿K维度循环累加
    for k in range(0, K, BLOCK_K):
        # 加载A的一个块 [BLOCK_M, BLOCK_K]
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k[None, :]) < K), other=0.0)
        
        # 加载B的一个块 [BLOCK_K, BLOCK_N]
        b_ptrs = b_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn
        b = tl.load(b_ptrs, mask=((k + offs_k[:, None]) < K) & (offs_n[None, :] < N), other=0.0)
        
        # 块矩阵乘法累加
        acc += tl.dot(a, b)
    
    # 5️⃣ 写回结果
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # 块大小配置
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    
    # 2D grid
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    return c


# 测试
if __name__ == "__main__":
    a = torch.randn(512, 256, device='cuda', dtype=torch.float16)
    b = torch.randn(256, 512, device='cuda', dtype=torch.float16)
    
    triton_out = matmul(a, b)
    torch_out = torch.matmul(a, b)
    
    print(f"✅ 结果正确: {torch.allclose(triton_out, torch_out, atol=1e-2)}")
```

---

## 第4部分：Softmax实现 & 优化技巧（15分钟）

### 4.1 Softmax Kernel

```python
@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    input_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    # 每个program处理一行
    row_idx = tl.program_id(0)
    
    # 计算当前行的起始位置
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # 加载一行数据
    mask = col_offsets < n_cols
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    
    # Softmax计算（数值稳定版本）
    row_max = tl.max(row, axis=0)          # 1️⃣ 找最大值
    row = row - row_max                     # 2️⃣ 减最大值（数值稳定）
    numerator = tl.exp(row)                 # 3️⃣ 指数
    denominator = tl.sum(numerator, axis=0) # 4️⃣ 求和
    softmax_out = numerator / denominator   # 5️⃣ 归一化
    
    # 写回
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start_ptr + col_offsets, softmax_out, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    
    # BLOCK_SIZE必须是2的幂次且>=n_cols
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # 每行一个program
    grid = (n_rows,)
    
    softmax_kernel[grid](
        x, output, n_cols,
        x.stride(0), output.stride(0),
        BLOCK_SIZE,
    )
    return output
```

### 4.2 自动调优（AutoTune）

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
    ],
    key=['M', 'N', 'K'],  # 根据这些参数选择最优配置
)
@triton.jit
def matmul_autotune_kernel(...):
    # kernel代码同上
    pass
```

### 4.3 性能对比

```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(10, 16)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'PyTorch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='vector-add-performance',
        args={},
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: add(x, y))
    else:
        ms = triton.testing.do_bench(lambda: x + y)
    
    gbps = 3 * x.numel() * x.element_size() / ms * 1e-6
    return gbps

# 运行: benchmark.run(show_plots=True, print_data=True)
```

---

## 第5部分：快速参考 & 下一步（5分钟）

### 5.1 常用API速查

```python
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 位置 & 索引
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
tl.program_id(axis)         # 当前program ID
tl.num_programs(axis)       # program总数
tl.arange(start, end)       # 连续序列

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 内存操作
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
tl.load(ptr, mask, other)   # 加载，other为mask=False时的默认值
tl.store(ptr, val, mask)    # 存储
tl.atomic_add(ptr, val)     # 原子加

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 数学运算
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
tl.dot(a, b)                # 矩阵乘法
tl.exp, tl.log, tl.sqrt     # 逐元素数学函数
tl.max, tl.min, tl.sum      # 归约操作
tl.where(cond, x, y)        # 条件选择
tl.zeros, tl.full           # 创建张量

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 辅助函数
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
triton.cdiv(a, b)           # 向上取整除法
triton.next_power_of_2(n)   # 下一个2的幂次
"""
```

### 5.2 常见错误 & 解决

```python
"""
❌ 错误：BLOCK_SIZE不是2的幂次
✅ 解决：BLOCK_SIZE: tl.constexpr 必须是2的幂次

❌ 错误：忘记mask导致越界
✅ 解决：始终使用 mask = offsets < n_elements

❌ 错误：类型不匹配
✅ 解决：显式转换 x.to(tl.float32)

❌ 错误：grid计算错误
✅ 解决：使用 triton.cdiv 向上取整
"""
```

### 5.3 学习资源

| 资源 | 链接 |
|------|------|
| 📚 官方教程 | https://triton-lang.org/main/getting-started/tutorials |
| 💻 GitHub | https://github.com/openai/triton |
| 📖 FlashAttention实现 | 官方教程第6节 |
| 🔧 实战项目 | Unsloth、xFormers |

---

## ✅ 1小时学习检查清单

```
□ 理解 @triton.jit 装饰器的作用
□ 能解释 program_id 和 BLOCK_SIZE 的关系
□ 掌握 tl.load / tl.store / mask 的使用
□ 理解矩阵乘法的分块策略
□ 会使用 triton.autotune 自动调优
□ 成功运行向量加法示例
```

---

**🎉 恭喜完成入门！** 下一步建议：尝试实现 LayerNorm 或 GELU 激活函数。