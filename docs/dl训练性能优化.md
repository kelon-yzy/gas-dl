# 深度学习训练性能优化方案

> 基于 d2l 第十二章「计算性能」理论与项目代码现状分析
> 创建日期：2026-05-16

## 1. 优化依据

d2l 第十二章系统阐述了深度学习计算性能的影响因素：

| 章节 | 核心观点 | 与本项目的关联 |
|------|---------|--------------|
| 12.1 编译器与解释器 | 命令式编程灵活但慢，符号式（`torch.compile`）可融合算子减少 Python 开销 | 所有模型均以 eager mode 运行，未做编译加速 |
| 12.2 异步计算 | GPU 操作默认异步，但 `numpy()`、`print`、`.item()` 等会触发隐式同步，破坏性能 | 评估阶段每 batch 调用 `.cpu().numpy()` 强制同步 |
| 12.3 自动并行 | 框架后端可自动并行无依赖的操作，但前提是任务队列不被阻塞 | `num_workers=0` 导致 GPU 空等 CPU 数据预处理 |
| 12.5-12.6 多 GPU | 数据并行适合大模型 + 大数据；小模型通信开销 > 计算收益 | 当前模型参数量小，多 GPU 无收益 |

## 2. 现状问题定位

### 2.1 P0：DataLoader num_workers=0（d2l 12.2-12.3）

6 个 multimodal 配置中 `num_workers: 0`，数据加载完全在主进程完成。

`WaveformSequenceDataset.__getitem__` 每次调用执行：
1. NumPy 索引取数据
2. `np.array(..., dtype=np.int16, copy=True)` — 强制 copy
3. `torch.from_numpy(...)` — NumPy→torch 转换
4. `self.slow_scaler.transform(slow)` — CPU 侧归一化

GPU 在每个 batch 之间空等这些 CPU 操作完成，无法实现 d2l 12.2 描述的「前端快速提交任务 → 后端异步执行」模式。

**预期收益**：训练时间缩短 30-50%。

**改动范围**：6 个 multimodal 配置 YAML 的 `num_workers` 和 `eval_num_workers` 字段。

### 2.2 P1：评估阶段隐式同步（d2l 12.2.2）

`evaluate_with_predictions` 和 `predict` 中，每个 batch 都执行：

```python
pred = pred.cpu().numpy()    # ← 隐式同步：GPU 等待计算完成，再拷贝到 CPU
y_pred.append(pred)
y_true.append(y.cpu().numpy())  # ← 同上
```

d2l 12.2.2 明确指出：「将框架内存转换到 Python 会迫使后端等待特定变量就绪，错误地使用同步会破坏程序性能。」

应改为：先累积 GPU tensor 列表，推理结束后一次性 `torch.cat` 再转 CPU。

**预期收益**：推理阶段加速 20-30%。

**改动范围**：`src/dl/training/train.py` 中的 `evaluate_with_predictions` 和 `predict` 函数。

### 2.3 P2：数据预处理冗余拷贝（d2l 12.2.3）

`WaveformSequenceDataset.__getitem__` 中：

```python
torch.from_numpy(np.array(ultrasonic, dtype=np.int16, copy=True))
```

每个样本都经历：NumPy copy → dtype 转换 → torch.from_numpy。d2l 12.2.3 指出「频繁将少量数据从框架复制到 NumPy 会严重损害性能」。

优化方向：在 `_ensure_loaded` 阶段将共享数据预转为 tensor 并缓存，`__getitem__` 只做索引切片和 scaler 变换。

**预期收益**：数据加载加速约 40%（waveform 数据占比最大的拷贝开销被消除）。

**改动范围**：`src/dl/data/dataset_waveform.py` 中的 `WaveformSequenceDataset` 类。

### 2.4 torch.compile（已启用）

> 2026-05-21 更新：适配 PyTorch 2.8.0 / Python 3.12 / CUDA 12.8

**环境**：Ubuntu 22.04, PyTorch 2.8.0, Python 3.12, CUDA 12.8, Triton (auto)

**当前策略**：
- Linux + CUDA 环境下默认启用 `torch.compile(mode="default")`
- `default` 模式使用 Inductor 后端做 kernel fusion + pointwise 优化，**不启用 CUDA Graphs**
- 显存开销极小（< 100MB 额外），支持多实验并行（单卡 24GB 可跑 4 个）
- 可在 yaml 中设置 `compile: false` 完全关闭，或 `compile_mode: reduce-overhead` 启用 CUDA Graphs（单进程极致速度，但显存 +13GB）

**历史问题**：
- 旧默认 `reduce-overhead` 模式使用 CUDA Graphs，private pool 占用 13.6GB，多进程 OOM
- PyTorch 2.8 的 `default` 模式已足够高效（Inductor CUTLASS backend, kernel fusion），无需 CUDA Graphs

**配置示例**：
```yaml
training:
  # compile: true               # Linux+CUDA 默认开启，无需写
  # compile_mode: default       # 默认值，无需写
  # compile_mode: reduce-overhead  # 单进程极致速度（+13GB 显存）
  # compile: false              # 完全关闭（调试或极限并行用）
```

### 2.5 延后项

| 项目 | 延后理由 |
|------|---------|
| 多 GPU 数据并行 | 模型参数量小（242K），通信开销 > 计算收益 |
| 梯度累积模拟大 batch | 当前 batch_size=128 已足够 |

## 3. 实施计划

### P0：num_workers 配置调整 ✅

- 6 个 multimodal 配置的 `num_workers` 从 0 改为 2，`eval_num_workers` 从 0 改为 1
- 后续新实验统一使用 `num_workers: 12, eval_num_workers: 4`

### P1：评估阶段减少同步 ✅

- `predict`：累积 GPU tensor 列表，最后 `torch.cat` 后一次性 `.cpu().numpy()`
- `evaluate_with_predictions`：同上
- `_metadata_to_frame`：保持逐 batch 累积（DataFrame 不涉及 GPU 同步）

### P2：Dataset 预转换缓存 ✅

- `_ensure_loaded`：将 mmap 数据一次性 `np.array(copy=True)` 转为可写连续 ndarray 并缓存
- `__getitem__`：直接对 ndarray 切片后 `torch.from_numpy`，消除逐样本的 `np.array(copy=True)` + dtype 转换
- `slow_scaler.transform` 仍需逐样本 CPU 计算，这个开销无法消除
- `__getstate__` 逻辑不变：pickle 时仍清空数据数组，恢复时 `_ensure_loaded` 重新加载和预转换

### P3：PyTorch 2.8 / CUDA 12.8 适配 ✅

- `torch.compile` 默认模式从 `reduce-overhead` 改为 `default`（Inductor 优化，不用 CUDA Graphs）
- `GradScaler` 构造使用 `device.type` 而非硬编码 `"cuda"`，兼容 CPU fallback
- `configure_cudnn` 显式启用 `cudnn.allow_tf32 = True`（配合 `set_float32_matmul_precision("high")`）
- `set_seed` 中 `cudnn.deterministic=True` 后由 `configure_cudnn` 中的 `benchmark=True` 覆盖（训练优先性能）

## 4. 验证方式

每项优化实施后运行完整测试套件 `python -m pytest tests`，确保无回归。

## 5. 服务器环境参考

```bash
# 推荐环境变量（~/.bashrc）
export OMP_NUM_THREADS=1          # 避免 OpenMP 和 DataLoader workers 抢 CPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 减少显存碎片
```