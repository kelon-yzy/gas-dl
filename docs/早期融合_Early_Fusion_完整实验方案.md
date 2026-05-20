# 早期融合（Early Fusion）完整实验方案

## 0. 方案定位

本方案面向多源异构传感器时序数据，目标是系统实现并评估“早期融合”策略在天然气四组分回归任务中的有效性。

当前数据的核心特点是：

| 项目      | 规格                                                                 |
| ------- | ------------------------------------------------------------------:|
| 样本数     | 10000                                                              |
| 每个样本时间步 | 120                                                                |
| 低频时间步长  | 1 s                                                                |
| 超声通道    | `[B, 120, 1000]`                                                   |
| 光纤麦克风通道 | `[B, 120, 2000]`                                                   |
| 声学采样率   | 200 kHz                                                            |
| 慢变量通道   | `[B, 120, 8]`                                                      |
| 预测目标    | `[x_H2, x_CH4, x_CO2, x_N2]`                                       |
| 标签形式    | 序列级连续回归标签 `[B, 4]`                                                 |
| 数据划分    | 按 `mixture_id` 分组划分 train / val / test                             |
| 阶段划分    | baseline `0-19`，exposure `20-69`，steady `70-99`，recovery `100-119` |

早期融合的基本思想是：在输入端或浅层特征端尽早把高频声学波形、低频光学/热导/环境慢变量映射到同一时间步下，然后通过统一的浅层网络提取联合特征。

本方案不采用“直接 flatten 后拼接”的原始早期融合，而采用：

```text
高频波形反量化
    ↓
时间步级对齐
    ↓
光纤波形长度映射
    ↓
慢变量浅层投影
    ↓
输入端浅层融合或 FiLM 调制
    ↓
每时间步联合特征提取
    ↓
120 步时序建模
    ↓
四组分回归输出
```

---

## 1. 早期融合的核心物理假设

### 1.1 高频声学模态

超声对射通道和光纤麦克风通道提供的是气体介质中的声学响应。它们承载的信息包括：

1. 声速变化；
2. TOF 偏移；
3. 声衰减；
4. 波形包络变化；
5. 频率响应变化；
6. 分子弛豫导致的能量衰减差异；
7. 压力、温度、湿度对声学传播的调制效应。

其中，超声通道测量窗为 `5 ms`，对应 `1000` 点；光纤麦克风通道测量窗为 `10 ms`，对应 `2000` 点。两者采样率相同，但窗口长度不同，因此不能简单认为二者在波形采样维度上天然对齐。

### 1.2 低频慢变量模态

慢变量通道为：

```text
1. V_NDIR_CH4
2. V_NDIR_CO2
3. V_TCS
4. T_C
5. P_MPa
6. H_RH
7. L_m
8. piston_position_m
```

其物理含义不是普通辅助特征，而是声学波形解释所需的上下文变量：

| 慢变量                 | 物理作用                |
| ------------------- | ------------------- |
| `V_NDIR_CH4`        | 甲烷相关红外吸收强度          |
| `V_NDIR_CO2`        | 二氧化碳相关红外吸收强度        |
| `V_TCS`             | 混合气体热导响应            |
| `T_C`               | 声速、热导、红外吸收的温度补偿     |
| `P_MPa`             | NDIR 压力展宽、声速/声衰减补偿  |
| `H_RH`              | 湿度对声学和光学信号的干扰补偿     |
| `L_m`               | 声程，影响 TOF 和声速反演     |
| `piston_position_m` | 活塞位置，隐含容积、压力动态与工况阶段 |

早期融合的物理动机是：不要等到高层预测阶段才加入这些慢变量，而是在浅层声学特征提取阶段就让模型知道当前的温度、压力、湿度、声程、活塞位置和光热响应状态。

---

## 2. 早期融合的主要风险

### 2.1 维度灾难

如果直接把每个时间步的超声、光纤和慢变量 flatten 后拼接，则单个样本输入维度为：

$$
120 \times (1000 + 2000 + 8) = 360960
$$

这会导致：

1. 第一层参数量过大；
2. 训练速度慢；
3. 过拟合风险高；
4. 低维慢变量被高维波形淹没；
5. 模型更容易学习到仿真数据中的捷径，而不是稳定的物理规律。

### 2.2 信息淹没

声学波形每个时间步有 `3000` 个原始点，而慢变量只有 `8` 个通道。若直接拼接，梯度主要由高维声学波形主导，NDIR、热导、温度、压力等低维但高物理价值的变量容易被忽略。

### 2.3 阶段标签捷径

样本内部有固定阶段：

```text
baseline: 0-19
exposure: 20-69
steady: 70-99
recovery: 100-119
```

如果把阶段 one-hot 直接作为输入并在波形维度上广播，模型可能过度依赖阶段位置。例如模型可能学到“第 70 到 99 步最重要”，而不是学习传感器响应与气体组分之间的物理映射。

因此，阶段信息只能作为受控消融实验，不能作为默认主模型强输入。

### 2.4 采样率与测量窗不一致

两路声学信号采样率相同，但窗口长度不同。超声为 `1000` 点，光纤为 `2000` 点。如果直接把二者拼接到同一维度，模型会把“窗口长度差异”误当作模态特征。

必须先做长度映射，推荐三种方式：

| 方法       | 操作                   | 使用场景  |
| -------- | -------------------- | ----- |
| 固定平均池化   | `fiber 2000 -> 1000` | 稳定基线  |
| 可学习下采样卷积 | `Conv1d(k=5, s=2)`   | 推荐主方案 |
| 双分辨率并行编码 | 超声和光纤各自浅层卷积，再早期合并    | 精度优先  |

---

## 3. 输入数据定义

### 3.1 原始输入

```python
ultra_int16      # [B, 120, 1000]
fiber_int16      # [B, 120, 2000]
ultra_scale      # [B, 120]
fiber_scale      # [B, 120]
slow             # [B, 120, 8]
y                # [B, 4]
```

### 3.2 反量化

数据中原始声学波形以 `int16 + scale` 形式保存，训练时应在线反量化：

```python
ultra = ultra_int16.float() * ultra_scale.unsqueeze(-1)
fiber = fiber_int16.float() * fiber_scale.unsqueeze(-1)
```

### 3.3 训练集统计量归一化

只能使用训练集统计量，不能使用 val/test 统计量。

推荐统计量：

```python
mean_ultra, std_ultra
mean_fiber, std_fiber
mean_slow,  std_slow
```

归一化：

```python
ultra = (ultra - mean_ultra) / (std_ultra + 1e-8)
fiber = (fiber - mean_fiber) / (std_fiber + 1e-8)
slow  = (slow  - mean_slow)  / (std_slow  + 1e-8)
```

注意：不推荐对每条波形做独立标准化，因为声衰减幅值、波形能量和包络幅度本身包含气体组分信息。如果每条波形单独归一化，可能会破坏声衰减特征。

---

## 4. 时间步级对齐策略

你的数据已经在样本内部形成 `120` 个 `1 s` 时间步。早期融合的基本对齐单元应为：

```text
第 t 秒：
    ultrasonic[t]: 1000 点
    fiber_mic[t]: 2000 点
    slow[t]: 8 维
```

因此，早期融合不需要把 `200 kHz` 声学波形重采样到 `1 Hz`，而是把每个 `1 s` 低频时间步视为一个容器，其中包含该秒触发采集得到的一段高频声学测量窗。

正确理解是：

```text
低频时序长度：120
每个低频时间步内部嵌套一段高频声学局部窗口
```

模型结构应处理为：

```text
[B, 120, waveform_length]
    ↓
每个时间步内做 1D-CNN 局部编码
    ↓
得到 [B, 120, d_model]
    ↓
再对 120 步做 TCN / Transformer / GRU 时序建模
```

---

## 5. 早期融合方案 E0：直接拼接基线，仅用于反面验证

### 5.1 方案描述

该方案把超声波形、下采样后的光纤波形、慢变量广播图直接拼接。

```text
ultra[t]       -> [1, 1000]
fiber[t]       -> [1, 2000] -> [1, 1000]
slow[t]        -> [8] -> [8, 1000]
concat         -> [10, 1000]
```

如果加入阶段 embedding：

```text
phase[t]       -> embedding -> [d_phase] -> [d_phase, 1000]
concat         -> [10 + d_phase, 1000]
```

### 5.2 作用

E0 不建议作为主模型，只用于证明：

1. 直接拼接是否容易过拟合；
2. 慢变量是否被高维声学淹没；
3. 阶段信息是否产生捷径；
4. 早期融合是否需要更合理的投影机制。

### 5.3 预期结果

E0 可能在 clean validation 上表现尚可，但在以下测试中退化明显：

1. 光学传感器噪声扰动；
2. 热导基线漂移；
3. 压力分布外测试；
4. 声学幅值衰减；
5. 光纤通道缺失；
6. 阶段分布扰动。

---

## 6. 早期融合方案 E1：浅层投影拼接 Early Fusion

### 6.1 网络拓扑

```text
ultra[t]: [1000]
    ↓
reshape -> [1, 1000]

fiber[t]: [2000]
    ↓
learnable downsample
    ↓
[1, 1000]

slow[t]: [8]
    ↓
MLP projection
    ↓
[d_slow]
    ↓
broadcast
    ↓
[d_slow, 1000]

concat:
    [1 + 1 + d_slow, 1000]
    ↓
shared shallow 1D-CNN
    ↓
per-step embedding [d_model]

repeat for 120 steps:
    [B, 120, d_model]
    ↓
temporal encoder
    ↓
sequence embedding
    ↓
regression head
    ↓
[x_H2, x_CH4, x_CO2, x_N2]
```

### 6.2 推荐超参数

| 模块            | 配置                                       |
| ------------- | ---------------------------------------- |
| `d_slow`      | 16                                       |
| `d_model`     | 128                                      |
| 声学卷积 stem     | 4 层 Conv1d                               |
| 激活函数          | GELU                                     |
| 归一化           | GroupNorm for Conv1d，LayerNorm for token |
| 时序模块          | TCN 或 Transformer                        |
| 输出头           | mean head + logvar head                  |
| dropout       | 0.1                                      |
| optimizer     | AdamW                                    |
| learning rate | `3e-4`                                   |
| weight decay  | `1e-4`                                   |
| batch size    | 16-64                                    |
| scheduler     | linear warmup + cosine decay             |

### 6.3 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    def __init__(self, cin, cout, k=7, s=1, p=None, groups=8):
        super().__init__()
        if p is None:
            p = k // 2

        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, kernel_size=k, stride=s, padding=p),
            nn.GroupNorm(num_groups=min(groups, cout), num_channels=cout),
            nn.GELU(),
            nn.Conv1d(cout, cout, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=min(groups, cout), num_channels=cout),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)


class LearnableFiberDownsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(8, 1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, fiber):
        # fiber: [B*T, 1, 2000]
        # output: [B*T, 1, 1000]
        return self.down(fiber)


class EarlyFusionConcatModel(nn.Module):
    def __init__(
        self,
        slow_dim=8,
        d_slow=16,
        d_model=128,
        out_dim=4,
        use_softmax_output=True
    ):
        super().__init__()

        self.use_softmax_output = use_softmax_output

        self.fiber_down = LearnableFiberDownsample()

        self.slow_proj = nn.Sequential(
            nn.Linear(slow_dim, d_slow),
            nn.LayerNorm(d_slow),
            nn.GELU(),
            nn.Linear(d_slow, d_slow),
            nn.GELU()
        )

        in_channels = 2 + d_slow

        self.wave_stem = nn.Sequential(
            ConvBlock1D(in_channels, 32, k=15, s=2),   # 1000 -> 500
            ConvBlock1D(32, 64, k=11, s=2),            # 500 -> 250
            ConvBlock1D(64, 96, k=7, s=2),             # 250 -> 125
            ConvBlock1D(96, d_model, k=5, s=2),        # 125 -> 63
            nn.AdaptiveAvgPool1d(1)
        )

        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=4 * d_model,
                dropout=0.1,
                batch_first=True,
                activation="gelu"
            ),
            num_layers=3
        )

        self.pool_gate = nn.Sequential(
            nn.Linear(d_model, 1)
        )

        self.mean_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, out_dim)
        )

        self.logvar_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, out_dim)
        )

    def forward(self, ultra, fiber, slow):
        """
        ultra: [B, 120, 1000]
        fiber: [B, 120, 2000]
        slow:  [B, 120, 8]
        """
        B, T, _ = ultra.shape

        ultra = ultra.reshape(B * T, 1, 1000)
        fiber = fiber.reshape(B * T, 1, 2000)

        fiber = self.fiber_down(fiber)

        slow_e = self.slow_proj(slow)                         # [B, T, d_slow]
        slow_e = slow_e.reshape(B * T, -1, 1)
        slow_e = slow_e.expand(-1, -1, 1000)                  # [B*T, d_slow, 1000]

        x = torch.cat([ultra, fiber, slow_e], dim=1)          # [B*T, 2+d_slow, 1000]

        h = self.wave_stem(x).squeeze(-1)                    # [B*T, d_model]
        h = h.reshape(B, T, -1)                              # [B, T, d_model]

        h = self.temporal(h)

        # learnable attention pooling over 120 steps
        a = self.pool_gate(h)                                # [B, T, 1]
        a = torch.softmax(a, dim=1)
        h_seq = (a * h).sum(dim=1)                           # [B, d_model]

        mu_raw = self.mean_head(h_seq)
        logvar = self.logvar_head(h_seq).clamp(min=-8.0, max=4.0)

        if self.use_softmax_output:
            mu = torch.softmax(mu_raw, dim=-1)
        else:
            mu = mu_raw

        return {
            "mu": mu,
            "mu_raw": mu_raw,
            "logvar": logvar,
            "pool_weight": a.squeeze(-1)
        }
```

---

## 7. 早期融合方案 E2：FiLM 调制型 Early Fusion，推荐主方案

### 7.1 为什么 FiLM 更适合本课题

直接把慢变量广播到波形维度，本质上是在输入端把低频变量复制 `1000` 次。这样虽然实现简单，但会带来两个问题：

1. 慢变量被当作“额外波形通道”，不符合物理含义；
2. 模型可能把慢变量当成捷径，而不是用它解释声学响应。

FiLM 的思想是：慢变量不直接与波形抢输入通道，而是生成声学特征的缩放和偏置。

数学形式为：

$$
h' = \gamma(s) \odot h + \beta(s)
$$

其中：

- \(h\) 是声学卷积特征；
- \(s\) 是慢变量；
- \(\gamma(s)\) 是慢变量生成的通道缩放；
- \(\beta(s)\) 是慢变量生成的通道偏置。

物理解释：

```text
温度、压力、湿度、声程、NDIR、热导等慢变量
    不直接作为高频波形的一部分
    而是作为声学传播特征的调制条件
```

这更符合真实系统：压力和温度不是一段声波，但它们会改变声波传播规律。

### 7.2 网络拓扑

```text
ultra[t], fiber[t]
    ↓
fiber 2000 -> 1000
    ↓
concat waveform channels: [2, 1000]
    ↓
acoustic stem conv block 1
    ↓
slow[t] -> FiLM generator -> gamma_1, beta_1
    ↓
feature modulation
    ↓
acoustic stem conv block 2
    ↓
slow[t] -> FiLM generator -> gamma_2, beta_2
    ↓
feature modulation
    ↓
per-step acoustic-context embedding
    ↓
temporal encoder
    ↓
regression head
```

### 7.3 代码实现

```python
class FiLM(nn.Module):
    def __init__(self, slow_dim, channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(slow_dim, 2 * channels),
            nn.GELU(),
            nn.Linear(2 * channels, 2 * channels)
        )

    def forward(self, h, slow):
        """
        h:    [B*T, C, L]
        slow: [B*T, slow_dim]
        """
        gamma_beta = self.net(slow)
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)

        return h * (1.0 + gamma) + beta


class EarlyFusionFiLMModel(nn.Module):
    def __init__(
        self,
        slow_dim=8,
        d_model=128,
        out_dim=4,
        use_softmax_output=True
    ):
        super().__init__()

        self.use_softmax_output = use_softmax_output
        self.fiber_down = LearnableFiberDownsample()

        self.conv1 = ConvBlock1D(2, 32, k=15, s=2)
        self.film1 = FiLM(slow_dim=slow_dim, channels=32)

        self.conv2 = ConvBlock1D(32, 64, k=11, s=2)
        self.film2 = FiLM(slow_dim=slow_dim, channels=64)

        self.conv3 = ConvBlock1D(64, 96, k=7, s=2)
        self.film3 = FiLM(slow_dim=slow_dim, channels=96)

        self.conv4 = ConvBlock1D(96, d_model, k=5, s=2)
        self.film4 = FiLM(slow_dim=slow_dim, channels=d_model)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=4 * d_model,
                dropout=0.1,
                batch_first=True,
                activation="gelu"
            ),
            num_layers=3
        )

        self.pool_gate = nn.Linear(d_model, 1)

        self.mean_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, out_dim)
        )

        self.logvar_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, out_dim)
        )

    def forward(self, ultra, fiber, slow):
        """
        ultra: [B, 120, 1000]
        fiber: [B, 120, 2000]
        slow:  [B, 120, 8]
        """
        B, T, _ = ultra.shape

        ultra = ultra.reshape(B * T, 1, 1000)
        fiber = fiber.reshape(B * T, 1, 2000)
        fiber = self.fiber_down(fiber)

        x = torch.cat([ultra, fiber], dim=1)        # [B*T, 2, 1000]
        slow_bt = slow.reshape(B * T, -1)           # [B*T, 8]

        h = self.conv1(x)
        h = self.film1(h, slow_bt)

        h = self.conv2(h)
        h = self.film2(h, slow_bt)

        h = self.conv3(h)
        h = self.film3(h, slow_bt)

        h = self.conv4(h)
        h = self.film4(h, slow_bt)

        h = self.pool(h).squeeze(-1)                # [B*T, d_model]
        h = h.reshape(B, T, -1)                     # [B, T, d_model]

        h = self.temporal(h)

        a = self.pool_gate(h)
        a = torch.softmax(a, dim=1)
        h_seq = (a * h).sum(dim=1)

        mu_raw = self.mean_head(h_seq)
        logvar = self.logvar_head(h_seq).clamp(min=-8.0, max=4.0)

        if self.use_softmax_output:
            mu = torch.softmax(mu_raw, dim=-1)
        else:
            mu = mu_raw

        return {
            "mu": mu,
            "mu_raw": mu_raw,
            "logvar": logvar,
            "pool_weight": a.squeeze(-1)
        }
```

### 7.4 预期优势

E2 相比 E1 的优势：

1. 慢变量以物理调制方式影响声学特征；
2. 不需要把低频变量粗暴复制到 `1000` 个采样点；
3. 更不容易出现维度淹没；
4. 可解释性更好，可分析不同慢变量对卷积通道的调制强度；
5. 对压力、温度、湿度变化更稳健。

---

## 8. 早期融合方案 E3：双分辨率浅层融合

### 8.1 使用原因

E1 和 E2 都把光纤麦克风从 `2000` 点映射到 `1000` 点。这样做简单，但可能损失光纤麦克风后半段窗口中的衰减信息。

双分辨率方案不强行把两路波形放到同一采样点长度，而是先分别用极浅的卷积做局部压缩，再在浅层特征端融合。

### 8.2 拓扑结构

```text
ultra[t]: [1, 1000]
    ↓
ultra shallow stem
    ↓
[64, L']

fiber[t]: [1, 2000]
    ↓
fiber shallow stem
    ↓
[64, L']

slow[t]: [8]
    ↓
FiLM or broadcast
    ↓
modulate / concat

concat:
    [128, L']
    ↓
shared fusion conv
    ↓
[d_model]
```

### 8.3 代码示意

```python
class EarlyFusionDualResolutionModel(nn.Module):
    def __init__(self, slow_dim=8, d_model=128, out_dim=4):
        super().__init__()

        self.ultra_stem = nn.Sequential(
            ConvBlock1D(1, 32, k=15, s=2),      # 1000 -> 500
            ConvBlock1D(32, 64, k=11, s=2),     # 500 -> 250
            nn.AdaptiveAvgPool1d(125)
        )

        self.fiber_stem = nn.Sequential(
            ConvBlock1D(1, 32, k=15, s=4),      # 2000 -> 500
            ConvBlock1D(32, 64, k=11, s=2),     # 500 -> 250
            nn.AdaptiveAvgPool1d(125)
        )

        self.slow_film = FiLM(slow_dim=slow_dim, channels=128)

        self.fusion_stem = nn.Sequential(
            ConvBlock1D(128, d_model, k=7, s=2),
            ConvBlock1D(d_model, d_model, k=5, s=2),
            nn.AdaptiveAvgPool1d(1)
        )

        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=4 * d_model,
                dropout=0.1,
                batch_first=True,
                activation="gelu"
            ),
            num_layers=3
        )

        self.pool_gate = nn.Linear(d_model, 1)
        self.mean_head = nn.Linear(d_model, out_dim)
        self.logvar_head = nn.Linear(d_model, out_dim)

    def forward(self, ultra, fiber, slow):
        B, T, _ = ultra.shape

        ultra = ultra.reshape(B * T, 1, 1000)
        fiber = fiber.reshape(B * T, 1, 2000)
        slow_bt = slow.reshape(B * T, -1)

        hu = self.ultra_stem(ultra)
        hf = self.fiber_stem(fiber)

        h = torch.cat([hu, hf], dim=1)
        h = self.slow_film(h, slow_bt)

        h = self.fusion_stem(h).squeeze(-1)
        h = h.reshape(B, T, -1)

        h = self.temporal(h)

        a = torch.softmax(self.pool_gate(h), dim=1)
        h_seq = (a * h).sum(dim=1)

        mu_raw = self.mean_head(h_seq)
        mu = torch.softmax(mu_raw, dim=-1)
        logvar = self.logvar_head(h_seq).clamp(-8.0, 4.0)

        return {
            "mu": mu,
            "mu_raw": mu_raw,
            "logvar": logvar,
            "pool_weight": a.squeeze(-1)
        }
```

### 8.4 使用建议

E3 比 E2 更保守地保留了光纤麦克风的原始窗口信息，但参数量和计算量更高。建议作为早期融合增强版，而不是第一主模型。

---

## 9. 阶段信息的使用方式

### 9.1 不推荐方式

不推荐直接使用：

```text
phase one-hot -> [4] -> broadcast -> [4, 1000]
```

原因是阶段信号非常干净，模型很容易优先利用它，而不是学习传感器响应。

### 9.2 推荐方式

只在消融实验中加入阶段 embedding，并使用 dropout：

```python
class PhaseEmbedding(nn.Module):
    def __init__(self, d_phase=8, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(4, d_phase)
        self.drop = nn.Dropout(dropout)

    def forward(self, phase_id):
        return self.drop(self.emb(phase_id))
```

阶段编号：

```python
def build_phase_id(T=120):
    phase = torch.zeros(T, dtype=torch.long)
    phase[20:70] = 1
    phase[70:100] = 2
    phase[100:120] = 3
    return phase
```

### 9.3 阶段消融设置

| 实验                    | 设置                         |
| --------------------- | -------------------------- |
| `E-NoPhase`           | 不使用阶段信息                    |
| `E-PhaseEmbed`        | 使用阶段 embedding             |
| `E-PhaseEmbedDropout` | 使用阶段 embedding + dropout   |
| `E-SteadyOnly`        | 只输入 `70-99` 步              |
| `E-Full120`           | 输入完整 `120` 步               |
| `E-PhasePooling`      | 不把阶段输入网络，只在 pooling 时分阶段加权 |

### 9.4 更稳妥的 phase-aware pooling

不把阶段作为输入，而是在池化时分别统计各阶段特征：

```text
h_baseline = mean(h[:, 0:20])
h_exposure = mean(h[:, 20:70])
h_steady = mean(h[:, 70:100])
h_recovery = mean(h[:, 100:120])

h_seq = MLP([h_baseline, h_exposure, h_steady, h_recovery])
```

这样可以利用阶段结构，又不让阶段 one-hot 成为浅层捷径。

---

## 10. 损失函数设计

### 10.1 标签尺度

如果标签是体积分数，例如：

```text
x_H2 + x_CH4 + x_CO2 + x_N2 = 1
```

则模型输出推荐使用 softmax：

```python
mu = torch.softmax(mu_raw, dim=-1)
```

如果标签以 `vol%` 存储，则先转换为比例：

```python
y_train = y_train / 100.0
```

预测后再转换回：

```python
pred_vol_percent = pred * 100.0
```

### 10.2 异方差高斯负对数似然

模型同时输出：

```python
mu      # [B, 4]
logvar  # [B, 4]
```

损失函数：

$$
\mathcal{L}_{\mathrm{nll}}
=
\frac{1}{B K}
\sum_{i=1}^{B}
\sum_{k=1}^{K}
\left[
\exp(-s_{i,k})(y_{i,k}-\mu_{i,k})^2 + s_{i,k}
\right]
$$

其中：

- \(K = 4\)；
- \(s_{i,k} = \log \sigma_{i,k}^{2}\)。

### 10.3 组分和约束

如果输出没有使用 softmax，则加入：

$$
\mathcal{L}_{\mathrm{sum}}
=
\frac{1}{B}
\sum_{i=1}^{B}
\left(
\sum_{k=1}^{K}\mu_{i,k} - 1
\right)^2
$$

### 10.4 非负约束

如果不用 softmax，可用：

```python
mu = F.softplus(mu_raw)
mu = mu / (mu.sum(dim=-1, keepdim=True) + 1e-8)
```

但为了四组分比例任务，推荐 softmax。

### 10.5 总损失

推荐总损失：

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{nll}}
+
\lambda_{\mathrm{sum}}\mathcal{L}_{\mathrm{sum}}
$$

若使用 softmax 输出，\(\mathcal{L}_{\mathrm{sum}}\) 可以省略或仅作为监控指标。

### 10.6 代码实现

```python
def heteroscedastic_nll(mu, logvar, y):
    inv_var = torch.exp(-logvar)
    loss = inv_var * (y - mu) ** 2 + logvar
    return loss.mean()


def composition_sum_loss(mu):
    return ((mu.sum(dim=-1) - 1.0) ** 2).mean()


def total_loss_fn(outputs, y, lambda_sum=0.0):
    mu = outputs["mu"]
    logvar = outputs["logvar"]

    loss_nll = heteroscedastic_nll(mu, logvar, y)

    if lambda_sum > 0:
        loss_sum = composition_sum_loss(mu)
    else:
        loss_sum = torch.tensor(0.0, device=mu.device)

    loss = loss_nll + lambda_sum * loss_sum

    return {
        "loss": loss,
        "loss_nll": loss_nll.detach(),
        "loss_sum": loss_sum.detach()
    }
```

---

## 11. 训练策略

### 11.1 优化器

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-4
)
```

### 11.2 学习率调度

推荐：

```text
前 5 个 epoch linear warmup
之后 cosine decay
```

示意：

```python
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

warmup = LinearLR(
    optimizer,
    start_factor=0.1,
    end_factor=1.0,
    total_iters=5
)

cosine = CosineAnnealingLR(
    optimizer,
    T_max=95,
    eta_min=1e-6
)

scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup, cosine],
    milestones=[5]
)
```

### 11.3 混合精度

声学波形数据量大，推荐开启 AMP：

```python
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    outputs = model(ultra, fiber, slow)
    loss_dict = total_loss_fn(outputs, y)

scaler.scale(loss_dict["loss"]).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

### 11.4 batch size

建议根据显存选择：

| 显存    | batch size |
| -----:| ----------:|
| 8 GB  | 8-16       |
| 12 GB | 16-24      |
| 24 GB | 32-64      |
| 48 GB | 64-128     |

如果使用 memmap 数据加载，建议：

```text
num_workers: 4-8
pin_memory: True
persistent_workers: True
prefetch_factor: 2
```

### 11.5 早停

```text
monitor: validation macro RMSE
patience: 20 epochs
min_delta: 1e-5
```

---

## 12. 评价指标

### 12.1 单组分 MAE

$$
\mathrm{MAE}_{k}
=
\frac{1}{N}
\sum_{i=1}^{N}
|y_{i,k} - \hat{y}_{i,k}|
$$

### 12.2 单组分 RMSE

$$
\mathrm{RMSE}_{k}
=
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
(y_{i,k} - \hat{y}_{i,k})^2
}
$$

### 12.3 单组分相对误差

$$
\mathrm{MRE}_{k}
=
\frac{1}{N}
\sum_{i=1}^{N}
\frac{|y_{i,k} - \hat{y}_{i,k}|}{|y_{i,k}| + \epsilon}
$$

### 12.4 最大相对误差

$$
\mathrm{MaxRE}
=
\max_{i,k}
\frac{|y_{i,k} - \hat{y}_{i,k}|}{|y_{i,k}| + \epsilon}
$$

### 12.5 组分和误差

$$
\mathrm{SumError}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left|
\sum_{k=1}^{K}\hat{y}_{i,k} - 1
\right|
$$

### 12.6 不确定性指标

如果模型输出 `logvar`，还应评估：

1. NLL；
2. 预测区间覆盖率；
3. 不确定性与绝对误差的相关系数；
4. 高不确定性样本是否集中在动态阶段或异常工况。

---

## 13. 早期融合专用诊断实验

### 13.1 模态淹没诊断

训练后分别遮挡不同模态：

| 实验              | 操作                            |
| --------------- | ----------------------------- |
| `mask_ultra`    | 超声置零                          |
| `mask_fiber`    | 光纤置零                          |
| `mask_ndir`     | `V_NDIR_CH4`, `V_NDIR_CO2` 置零 |
| `mask_tcs`      | `V_TCS` 置零                    |
| `mask_env`      | `T_C`, `P_MPa`, `H_RH` 置零     |
| `mask_geometry` | `L_m`, `piston_position_m` 置零 |

如果遮挡慢变量后性能几乎不变，说明模型没有真正利用低频物理变量。

如果遮挡声学后性能大幅下降，而遮挡 NDIR/TCS 几乎无影响，说明早期融合发生了声学主导的信息淹没。

### 13.2 梯度贡献诊断

计算不同输入模态的梯度范数：

```python
loss.backward()

grad_ultra = ultra.grad.abs().mean()
grad_fiber = fiber.grad.abs().mean()
grad_slow  = slow.grad.abs().mean()
```

观察：

```text
grad_ultra : grad_fiber : grad_slow
```

如果 `grad_slow` 长期接近 0，说明慢变量没有有效参与早期融合。

### 13.3 噪声鲁棒性测试

对不同模态加噪：

```text
ultra_noise_std: 0.01, 0.03, 0.05, 0.10
fiber_noise_std: 0.01, 0.03, 0.05, 0.10
slow_noise_std: 0.01, 0.03, 0.05, 0.10
```

评估：

```text
RMSE vs noise_std
MRE vs noise_std
MaxRE vs noise_std
```

### 13.4 时间错位测试

模拟高低频采集不同步：

```text
slow[t] -> slow[t + delta]
delta = -3, -2, -1, 0, 1, 2, 3
```

如果早期融合对小错位极其敏感，说明模型过度依赖瞬时拼接，而没有学到稳定的时序关系。

---

## 14. 消融实验矩阵

### 14.1 主消融表

| 编号   | 模型                           | 目的          |
| ---- | ---------------------------- | ----------- |
| `E0` | direct concat                | 验证原始拼接风险    |
| `E1` | shallow projection concat    | 标准早期融合基线    |
| `E2` | FiLM early fusion            | 推荐主早期融合     |
| `E3` | dual-resolution early fusion | 保留光纤完整窗口    |
| `E4` | E2 + phase embedding         | 验证阶段信息收益    |
| `E5` | E2 + phase-aware pooling     | 利用阶段结构但避免捷径 |
| `E6` | E2 + modality dropout        | 提升抗单模态失效能力  |
| `E7` | E2 steady-only               | 与完整 120 步对比 |

### 14.2 输入消融

| 编号   | 输入                       | 目的        |
| ---- | ------------------------ | --------- |
| `I1` | ultra only               | 超声单模态能力   |
| `I2` | fiber only               | 光纤声学单模态能力 |
| `I3` | slow only                | 光热环境慢变量能力 |
| `I4` | ultra + fiber            | 声学双通道能力   |
| `I5` | ultra + fiber + NDIR/TCS | 声学 + 光热   |
| `I6` | ultra + fiber + env      | 声学 + 环境补偿 |
| `I7` | all                      | 完整早期融合    |

### 14.3 归一化消融

| 编号   | 归一化方式                      | 风险        |
| ---- | -------------------------- | --------- |
| `N1` | train global z-score       | 推荐        |
| `N2` | per-sequence z-score       | 可能消除幅值信息  |
| `N3` | per-timestep z-score       | 高风险，破坏声衰减 |
| `N4` | waveform RMS normalization | 只适合作为对照   |

---

## 15. 推荐实验顺序

建议按以下顺序执行，不要一开始就跑所有模型。

### Step 1：单模态下限

```text
I1: ultra only
I2: fiber only
I3: slow only
```

目的：明确每个模态的独立预测能力。

### Step 2：标准早期融合

```text
E1: shallow projection concat
E2: FiLM early fusion
```

目的：比较直接拼接与物理调制的差异。

### Step 3：阶段消融

```text
E2 full 120
E2 steady only
E2 phase embedding
E2 phase-aware pooling
```

目的：验证阶段信息是否是真正有帮助，还是仅仅构成捷径。

### Step 4：鲁棒性测试

```text
sensor noise
sensor dropout
slow misalignment
pressure OOD
amplitude attenuation
```

目的：判断早期融合是否适合工程部署。

### Step 5：与中期、晚期融合比较

早期融合最终应与以下模型比较：

```text
Intermediate GMU
Intermediate Cross-Attention
Late Uncertainty Fusion
Late Drift-MSE Fusion
Traditional SVM + Dynamic Ridge
```

---

## 16. 推荐最终早期融合主模型

如果只保留一个早期融合主模型，推荐：

```text
E2: EarlyFusionFiLMModel
```

完整结构：

```text
ultra waveform
fiber waveform
    ↓
learnable fiber downsample
    ↓
two-channel waveform tensor
    ↓
multi-layer 1D-CNN acoustic stem
    ↓
slow variables generate FiLM gamma/beta
    ↓
modulated acoustic features
    ↓
per-step embedding
    ↓
Transformer over 120 steps
    ↓
attention pooling
    ↓
heteroscedastic regression head
    ↓
softmax composition output
```

该方案的优势是：

1. 避免直接把低频慢变量复制到高频采样维；
2. 避免声学高维特征完全淹没慢变量；
3. 物理解释清晰，慢变量作为声学传播条件；
4. 适合后续与中期融合、晚期融合对比；
5. 可以自然输出预测不确定性；
6. 可通过 FiLM 参数解释压力、温度、湿度、声程对声学特征的调制作用。

---

## 17. 早期融合实验结论的预期写法

如果实验结果显示 E2 优于 E1，可以表述为：

```text
相比直接输入端拼接，FiLM 调制型早期融合能更稳定地利用低频光热环境变量。
这说明对于高频声学波形与低频传感器共存的异构系统，低频变量不应被简单复制为高频通道，
而更适合作为声学特征提取过程中的条件调制因子。
```

如果 E2 在 clean test 上很好，但鲁棒性弱于晚期融合，可以表述为：

```text
早期融合在无扰动测试条件下可以获得较高精度，说明浅层联合建模能够捕获声学波形与环境慢变量之间的相关性。
但在单一传感器失效或基线漂移条件下，早期融合的退化幅度较大，表明其模态边界较弱，缺乏显式的单模态置信度控制机制。
因此，早期融合更适合作为端到端性能上限模型，而不宜单独作为工业鲁棒部署模型。
```

如果 E2 不如中期融合，可以表述为：

```text
早期融合将异构模态过早混合，虽然提高了输入信息密度，但也削弱了各传感器物理响应的独立性。
中期融合通过先独立编码再潜空间交互，更好地平衡了物理解耦与跨模态关联建模，因此在复杂工况下表现更稳定。
```

---

## 18. 最小可运行训练框架示意

```python
def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()

    total_loss = 0.0

    for batch in loader:
        ultra = batch["ultra"].to(device, non_blocking=True)
        fiber = batch["fiber"].to(device, non_blocking=True)
        slow = batch["slow"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(ultra, fiber, slow)
            loss_dict = total_loss_fn(outputs, y, lambda_sum=0.0)
            loss = loss_dict["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * ultra.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    preds = []
    targets = []
    logvars = []

    for batch in loader:
        ultra = batch["ultra"].to(device, non_blocking=True)
        fiber = batch["fiber"].to(device, non_blocking=True)
        slow = batch["slow"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        outputs = model(ultra, fiber, slow)

        preds.append(outputs["mu"].cpu())
        targets.append(y.cpu())
        logvars.append(outputs["logvar"].cpu())

    pred = torch.cat(preds, dim=0)
    y = torch.cat(targets, dim=0)
    logvar = torch.cat(logvars, dim=0)

    mae = (pred - y).abs().mean(dim=0)
    rmse = torch.sqrt(((pred - y) ** 2).mean(dim=0))
    mre = ((pred - y).abs() / (y.abs() + 1e-6)).mean(dim=0)
    max_re = ((pred - y).abs() / (y.abs() + 1e-6)).max()
    sum_error = (pred.sum(dim=-1) - 1.0).abs().mean()

    return {
        "mae": mae,
        "rmse": rmse,
        "mre": mre,
        "max_re": max_re,
        "sum_error": sum_error,
        "pred": pred,
        "target": y,
        "logvar": logvar
    }
```

---

## 19. 最终建议

早期融合实验不要只做一个模型。至少应实现：

```text
E1: shallow projection concat
E2: FiLM early fusion
E3: dual-resolution early fusion
```

其中：

```text
E2 = 早期融合主模型
E1 = 直接浅层拼接基线
E3 = 保留光纤完整窗口的增强版
```

最终论文或组会汇报中，早期融合部分的重点不应是“简单拼接”，而应是：

```text
如何在输入端解决高频波形与低频慢变量的时间对齐、尺度不一致、维度不平衡和物理语义不一致问题。
```

推荐结论预判：

```text
早期融合在 clean test 上可能取得较好精度，但在强噪声、传感器漂移和单模态失效条件下，其鲁棒性通常弱于晚期融合；
相比粗暴 concat，FiLM 型早期融合更适合该类高频声学 + 低频光热环境变量的异构传感系统。
```
