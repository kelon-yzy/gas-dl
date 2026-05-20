# CNN1D 融合模型端到端框架

## 1. 整体架构

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        cnn1d_multimodal                                 │
│  = MultimodalWrapper(backbone=CNN1DRegressor)                          │
│                                                                         │
│  输入 (5 个张量)                                                        │
│  ┌────────────┐  ┌──────────────┐  ┌────────────┐  ┌────────────────┐  │
│  │ultrasonic  │  │ultrasonic    │  │fiber_mic   │  │fiber_mic       │  │
│  │int16       │  │_scale       │  │int16       │  │_scale          │  │
│  │(B,T,1000) │  │(B,T)        │  │(B,T,2000) │  │(B,T)           │  │
│  └─────┬──────┘  └──────┬──────┘  └─────┬──────┘  └───────┬────────┘  │
│        │                │               │                  │            │
│        ▼                ▼               ▼                  ▼            │
│  ┌──────────────────────────┐   ┌──────────────────────────┐          │
│  │ AcousticWaveformEncoder  │   │ AcousticWaveformEncoder  │          │
│  │ (ultrasonic_encoder)     │   │ (fiber_mic_encoder)     │          │
│  │ 输出: (B,T,64)           │   │ 输出: (B,T,64)           │          │
│  └───────────┬──────────────┘   └───────────┬──────────────┘          │
│              │                               │                         │
│              │     ┌─────────────┐           │                         │
│              │     │slow (B,T,8) │           │                         │
│              │     └──────┬──────┘           │                         │
│              ▼            ▼                  ▼                        │
│         torch.cat([u_emb, f_emb, slow], dim=-1)                       │
│                    fused: (B, T, 136)                                  │
│                         │                                              │
│                    transpose(1,2)                                      │
│                    (B, 136, T)  ← NCT 格式                            │
│                         │                                              │
│                         ▼                                              │
│              ┌──────────────────────┐                                   │
│              │   CNN1DRegressor     │                                   │
│              │   backbone           │                                   │
│              │   输出: (B, 4)       │                                   │
│              └──────────────────────┘                                   │
└──────────────────────────────────────────────────────────────────────────┘
```

## 2. 数据流 — 逐阶段形状

| 阶段                           | 张量         | 形状                                   | 数据类型    | 备注               |
| ---------------------------- | ---------- | ------------------------------------ | ------- | ----------------  |
| 磁盘                           | ultrasonic | (N, T, 1000)                         | int16   | memmap 懒加载       |
| 磁盘                           | fiber_mic  | (N, T, 2000)                         | int16   | memmap 懒加载       |
| 磁盘                           | slow       | (N, T, 8)                            | float32 | 8 个慢变量通道         |
| 磁盘                           | y          | (N, 4)                               | float32 | H2/CH4/CO2/N2 浓度 |
| DataLoader batch             | ultrasonic | (B, T, 1000)                         | int16   |                  |
| **AcousticWaveformEncoder**  |            |                                      |         |                  |
| flatten                      | waveform   | (B\*T, 1000)                         | int16   |                  |
| FP32 转换+缩放                   | waveform   | (B\*T, 1000)                         | float32 | 强制 FP32          |
| unsqueeze                    | waveform   | (B\*T, 1, 1000)                      | float32 | 加通道维             |
| Conv1d(1→16,k15,s2,p7)       |            | (B\*T, 16, 500)                      | float32 |                  |
| BN(16)+ReLU                  |            | (B\*T, 16, 500)                      |         |                  |
| Conv1d(16→32,k11,s2,p5)      |            | (B\*T, 32, 250)                      |         |                  |
| BN(32)+ReLU                  |            | (B\*T, 32, 250)                      |         |                  |
| Conv1d(32→64,k7,s2,p3)       |            | (B\*T, 64, 125)                      |         |                  |
| BN(64)+ReLU                  |            | (B\*T, 64, 125)                      |         |                  |
| AdaptiveAvgPool1d(1)         | avg        | (B\*T, 64, 1) → squeeze → (B\*T, 64) |         |                  |
| AdaptiveMaxPool1d(1)         | mx         | (B\*T, 64, 1) → squeeze → (B\*T, 64) |         |                  |
| cat([avg, mx])               |            | (B\*T, 128)                          |         | 双池化拼接            |
| Linear(128→64)+LN+Dropout    |            | (B\*T, 64)                           |         | 投影头              |
| reshape                      | embedding  | (B, T, 64)                           |         |                  |
| **MultimodalWrapper**        |            |                                      |         |                  |
| cat([u_emb, f_emb, slow])    | fused      | (B, T, 136)                          |         | 64+64+8          |
| transpose                    | fused      | (B, 136, T)                          |         | NCT 格式           |
| **CNN1DRegressor**           |            |                                      |         |                  |
| Conv1d(136→32,k5,p2)         |            | (B, 32, T)                           |         |                  |
| BN(32)+ReLU+Dropout          |            | (B, 32, T)                           |         |                  |
| Conv1d(32→64,k5,p2)          |            | (B, 64, T)                           |         |                  |
| BN(64)+ReLU+Dropout          |            | (B, 64, T)                           |         |                  |
| Conv1d(64→64,k3,p1)          |            | (B, 64, T)                           |         |                  |
| BN(64)+ReLU                  |            | (B, 64, T)                           |         | 无 Dropout        |
| AdaptiveAvgPool1d(1)         | avg        | (B, 64, 1) → flatten → (B, 64)       |         |                  |
| AdaptiveMaxPool1d(1)         | mx         | (B, 64, 1) → flatten → (B, 64)       |         |                  |
| cat([avg, mx])               |            | (B, 128)                             |         | mean+max 末端汇聚 |
| Linear(128→64)+ReLU+Dropout  |            | (B, 64)                              |         |                  |
| Linear(64→4)                 |            | (B, 4)                               |         | 最终预测             |

## 3. 构造链路

```
YAML配置 → build_model({"name": "cnn1d_multimodal", ...})
         → MODEL_REGISTRY["cnn1d_multimodal"]
         → build_cnn1d_multimodal(**kwargs)
         → _build_multimodal(CNN1DRegressor, **kwargs)
```

`_build_multimodal` 内部：

1. 提取 wrapper 参数: `slow_dim=8, use_ultrasonic=True, use_fiber_mic=True, waveform_embedding_dim=64`
2. 计算 `fused_dim = 8 + 64*2 = 136`
3. 只传白名单参数给 backbone:
   `CNN1DRegressor(in_channels=136, hidden_channels=[32,64,64], kernel_size=5, dropout=0.1, temporal_pooling="mean_max", out_dim=4)`
4. 构造 `MultimodalWrapper(backbone, slow_dim=8, use_ultrasonic=True, use_fiber_mic=True, waveform_embedding_dim=64)`

## 4. 三个核心模块细节

### AcousticWaveformEncoder（声学编码器）

```
Conv1d(1→16, k=15, s=2, p=7) → BN(16) → ReLU     输出长度: L/2
Conv1d(16→32, k=11, s=2, p=5) → BN(32) → ReLU    输出长度: L/4
Conv1d(32→64, k=7, s=2, p=3) → BN(64) → ReLU     输出长度: L/8
AdaptiveAvgPool1d(1) + AdaptiveMaxPool1d(1) → cat → 128维
Linear(128→64) → LayerNorm(64) → Dropout(0.1)     输出: 64维
```

- 输入：`(B*T, W)` int16 波形 + `(B*T,)` float32 缩放因子
- 输出：`(B*T, 64)` float32 嵌入
- 两个独立实例分别处理 ultrasonic(1000采样) 和 fiber_mic(2000采样)
- 全程强制 FP32（AMP autocast disabled），防止 int16→float32 梯度下溢

### MultimodalWrapper（融合包装器）

- 持有 2 个 AcousticWaveformEncoder（可独立开关）
- 拼接：`cat([ultrasonic_emb, fiber_mic_emb, slow], dim=-1)` → (B, T, 136)
- 按 backbone 的 input_format 转置：
  - NCT（CNN1D/TCN/CNN-LSTM）：`transpose(1,2)` → (B, 136, T)
  - NTC（GRU/LSTM/Transformer）：保持 (B, T, 136)

> **历史说明**：曾实验过 `stage_one_hot`（4 维阶段 one-hot 编码）作为额外输入分支，包括浅层拼接和深层注入两种方式。实验结果表明该方案未带来性能提升且增加了训练复杂度，已从正式架构中移除。

### CNN1DRegressor（backbone）

- 3 层 Conv1d + BN + ReLU，前 2 层加 Dropout
- kernel_size：前 2 层用配置值(5)，第 3 层固定用 3
- 支持 `temporal_pooling="avg"` 和 `temporal_pooling="mean_max"` 两种末端时序汇聚
- `cnn1d_multimodal` 默认使用 `mean_max`：先做全局平均池化和全局最大池化，再拼接后送入 MLP head
- 这次改造只改变 CNN1D backbone 的序列级末端汇聚，不改变 AcousticWaveformEncoder 内部的 `avg + max` 波形汇聚
- 两层汇聚组合后的设计动机是减少跨时间步稀疏峰值事件被纯平均池化抹平的风险
- `input_format = "NCT"`，接收 `(B, in_channels, T)`

## 5. 参数统计

| 模块                         | 参数量            | 说明                     |
| -------------------------- | -------------- | ---------------------- |
| AcousticWaveformEncoder ×2 | ~29K × 2 ≈ 58K | 各含 3 层 Conv+BN + 投影头   |
| CNN1DRegressor             | ~12K           | 3 层 Conv+BN + MLP head |
| **总计**                     | **~70K**       |                        |

## 6. 训练配置（YAML）

```yaml
model:
  name: cnn1d_multimodal
  slow_dim: 8
  hidden_channels: [32, 64, 64]
  kernel_size: 5
  dropout: 0.1
  temporal_pooling: mean_max
  out_dim: 4

training:
  epochs: 120
  batch_size: 32
  amp: true
  optimizer: adamw
  learning_rate: 0.001
  weight_decay: 0.0001
  grad_clip_norm: 1.0
  lr_scheduler: cosine_warmup  # 5 epoch warmup, eta_min=0.0001
  early_stopping_patience: 15
```

## 7. 关键约束

- `waveform_embedding_dim` 锁定为 64（encoder 内强制校验）
- `slow_dim` 锁定为 8（V3 方案）
- `out_dim = 4`（H2/CH4/CO2/N2 四组分浓度）
- 旧 checkpoint 不兼容新架构（输入维度变化：140→136）
