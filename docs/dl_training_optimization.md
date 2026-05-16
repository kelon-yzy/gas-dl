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

### 2.4 延后项（当前不做）

| 项目 | d2l 章节 | 延后理由 |
|------|---------|---------|
| torch.compile | 12.1 | 模型刚接入，优先保正确性；需 PyTorch 2.0+ |
| 多 GPU 数据并行 | 12.5-12.6 | 模型参数量小，通信开销 > 计算收益 |
| 梯度累积模拟大 batch | 12.5 | 当前 batch_size=8 已足够，无需模拟 |

## 3. 实施计划

### P0：num_workers 配置调整 ✅

- 6 个 multimodal 配置的 `num_workers` 从 0 改为 2，`eval_num_workers` 从 0 改为 1
- 测试通过，无断言需更新

### P1：评估阶段减少同步 ✅

- `predict`：累积 GPU tensor 列表，最后 `torch.cat` 后一次性 `.cpu().numpy()`
- `evaluate_with_predictions`：同上
- `_metadata_to_frame`：保持逐 batch 累积（DataFrame 不涉及 GPU 同步）

### P2：Dataset 预转换缓存 ✅

- `_ensure_loaded`：将 mmap 数据一次性 `np.array(copy=True)` 转为可写连续 ndarray 并缓存
- `__getitem__`：直接对 ndarray 切片后 `torch.from_numpy`，消除逐样本的 `np.array(copy=True)` + dtype 转换
- `slow_scaler.transform` 仍需逐样本 CPU 计算，这个开销无法消除
- `__getstate__` 逻辑不变：pickle 时仍清空数据数组，恢复时 `_ensure_loaded` 重新加载和预转换

## 4. 验证方式

每项优化实施后运行完整测试套件 `python -m pytest tests`，确保无回归。