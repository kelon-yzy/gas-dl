# 中期融合（Intermediate Fusion）完整方案：面向双通道声学波形与低频光热环境变量的潜空间跨模态交互模型

## 1. 方案定位

中期融合不是把所有原始输入在最前端强行拼接，也不是等各模态独立预测后再做投票，而是在“各模态已经完成初步物理特征提取，但尚未形成最终决策”时进行深层交互。

对你的课题来说，中期融合应作为主线深度学习方案。原因是你的硬件系统本身具有明确的物理分工：超声与光纤麦克风承担声速、TOF、声衰减、声压能量变化等高频声学信息；NDIR 与热导传感器承担低频光热响应；温度、压力、湿度、声程、活塞位置承担环境补偿与工况描述。专利中的系统也明确采用高频声学 DAQ 与低频串口传感数据并行采集，再在上位机端做时间戳对齐和多模态特征矩阵构建。

你的数据集结构也非常适合中期融合：每个样本包含 `120` 个时间步，时间步长 `1 s`；每个时间步内有超声波形 `1000` 点、光纤麦克风波形 `2000` 点，两路声学采样率均为 `200 kHz`；慢变量为 `8` 通道，预测目标为 `[x_H2, x_CH4, x_CO2, x_N2]` 四组分连续回归标签。

中期融合的核心物理假设是：

```text
高频声学波形中包含气体声速、衰减、分子弛豫、传播路径变化等局部动态信息；
低频 NDIR / 热导 / 温湿压 / 声程 / 活塞位置包含光热响应与环境补偿信息；
二者之间不是简单线性相加，而是存在压力、温度、组分、声程共同调制下的跨模态耦合。
```

因此，模型应先让每个模态形成独立潜变量，再通过 GMU、Cross-Attention 或低秩双线性池化进行潜空间交互。GMU 的思路是通过门控机制学习不同模态对联合表示的贡献，原始论文将其定位为神经网络内部的中间融合单元。([arXiv](https://arxiv.org/abs/1702.01992?utm_source=chatgpt.com "Gated Multimodal Units for Information Fusion")) Cross-Attention 则继承 Transformer 的注意力机制，用 Query、Key、Value 计算跨 token 依赖关系，适合让慢变量主动查询声学 token 中的相关响应区域。([arXiv](https://arxiv.org/abs/1706.03762?utm_source=chatgpt.com "Attention Is All You Need")) 双线性池化适合表达模态间乘性交互，紧凑双线性池化的动机正是用更高表达力的双线性组合替代简单 concat 或 element-wise sum，同时避免完整外积带来的维度爆炸。([arXiv](https://arxiv.org/abs/1606.01847?utm_source=chatgpt.com "Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding"))

---

## 2. 输入定义与建模目标

原始输入：

```python
ultrasonic_int16:     [B, 120, 1000]
fiber_mic_int16:      [B, 120, 2000]
ultrasonic_scale:     [B, 120]
fiber_mic_scale:      [B, 120]
slow:                 [B, 120, 8]
y:                    [B, 4]
```

慢变量通道顺序：

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

预测目标：

```text
y = [x_H2, x_CH4, x_CO2, x_N2]
```

推荐将标签从 `vol%` 归一化到 `[0, 1]` 比例形式训练。如果四组分理论上应满足总和为 `1`，则模型输出端应使用 simplex 约束。

---

## 3. 总体网络拓扑

中期融合模型的标准拓扑如下：

```text
ultrasonic waveform [B,120,1000]
        ↓
Ultrasonic 1D-CNN Encoder
        ↓
z_u [B,120,d]

fiber_mic waveform [B,120,2000]
        ↓
Fiber 1D-CNN Encoder
        ↓
z_f [B,120,d]

slow variables [B,120,8]
        ↓
Slow MLP/TCN Encoder
        ↓
z_s [B,120,d]

z_u, z_f, z_s
        ↓
Intermediate Fusion Layer
GMU / Cross-Attention / Low-Rank Bilinear / Hybrid
        ↓
z_fused [B,120,d]
        ↓
Temporal Backbone
TCN / Transformer Encoder
        ↓
Sequence Pooling
        ↓
Regression Head
        ↓
[x_H2, x_CH4, x_CO2, x_N2]
```

这里有两个层级的时序：

```text
内层：每个 1 s 时间步内的高频声学窗口，1000/2000 点，由 1D-CNN 编码；
外层：120 个 1 s 时间步，由 TCN 或 Transformer 建模。
```

不要把 `120 × 1000` 或 `120 × 2000` 直接拉平成一个超长声学序列。这会破坏“每秒一次高频测量窗”的物理结构，也会让低频慢变量被迫重复插值。

---

## 4. 数据预处理

### 4.1 反量化

```python
ultra = ultrasonic_int16.float() * ultrasonic_scale.unsqueeze(-1)
fiber = fiber_mic_int16.float() * fiber_mic_scale.unsqueeze(-1)
```

### 4.2 标准化

推荐使用训练集统计量做全局标准化：

```python
ultra = (ultra - ultra_mean_train) / (ultra_std_train + 1e-8)
fiber = (fiber - fiber_mean_train) / (fiber_std_train + 1e-8)
slow  = (slow  - slow_mean_train)  / (slow_std_train  + 1e-8)
```

不建议对每条声学波形单独做峰值归一化，因为声衰减幅值本身可能包含气体组分信息。尤其是光纤麦克风通道，其能量衰减、幅值包络、频域响应都可能与气体分子弛豫和密度变化相关。

### 4.3 阶段信息

样本内部阶段为：

```text
baseline: 0-19
exposure: 20-69
steady:   70-99
recovery: 100-119
```

阶段信息可以作为可选输入，但不建议用强 one-hot 直接拼到原始波形通道。推荐方式是低维 embedding：

```python
phase_id:    [B,120]
phase_embed: [B,120,d]
```

然后在潜变量层加入：

```python
z_s = z_s + phase_embed
```

这样阶段信息只作为弱提示，不会在输入层抢占声学特征学习能力。

---

## 5. 独立模态编码器设计

### 5.1 超声波形编码器

输入：

```python
ultra: [B,120,1000]
```

输出：

```python
z_u: [B,120,d_model]
```

设计逻辑：

```text
Conv1D 大卷积核捕获声波局部振荡与包络；
stride 下采样压缩 1000 点高频窗口；
GroupNorm 避免小 batch 下 BatchNorm 不稳定；
GELU 保留平滑非线性；
AdaptiveAvgPool1d 将每个时间步压缩为一个 token。
```

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


class WaveformEncoder(nn.Module):
    def __init__(self, input_len, d_model=128):
        super().__init__()

        self.net = nn.Sequential(
            ConvBlock1D(1, 32,  k=15, s=2),   # 1000 -> 500, 2000 -> 1000
            ConvBlock1D(32, 64, k=11, s=2),   # 500 -> 250
            ConvBlock1D(64, 128, k=7, s=2),   # 250 -> 125
            ConvBlock1D(128, d_model, k=5, s=2),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        """
        x: [B, T, L]
        return: [B, T, d_model]
        """
        B, T, L = x.shape
        x = x.reshape(B * T, 1, L)
        h = self.net(x).squeeze(-1)
        h = h.reshape(B, T, -1)
        return h
```

### 5.2 光纤麦克风编码器

光纤麦克风通道长度为 `2000` 点，可以和超声共用相同结构，但参数不共享。原因是超声对射通道和光纤麦克风通道的物理含义不同，一个偏向 TOF/声速，一个偏向声压衰减/能量变化，不应强制共享卷积核。

```python
self.ultra_encoder = WaveformEncoder(input_len=1000, d_model=d_model)
self.fiber_encoder = WaveformEncoder(input_len=2000, d_model=d_model)
```

### 5.3 慢变量编码器

输入：

```python
slow: [B,120,8]
```

输出：

```python
z_s: [B,120,d_model]
```

慢变量中包含 NDIR、热导、温度、压力、湿度、声程和活塞位置。它不是普通辅助变量，而是解释声学响应变化的环境条件。因此慢变量分支不能太弱。

```python
class SlowEncoder(nn.Module):
    def __init__(self, slow_dim=8, d_model=128):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(slow_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        self.local_tcn = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, slow):
        """
        slow: [B, T, 8]
        return: [B, T, d_model]
        """
        h = self.input_proj(slow)
        h_local = self.local_tcn(h.transpose(1, 2)).transpose(1, 2)
        return self.norm(h + h_local)
```

---

## 6. 中期融合模块设计

建议至少实现四种中期融合模块，形成完整实验矩阵。

---

### 6.1 M0：潜变量拼接基线

这是最基础的中期融合：

```text
z = concat[z_u, z_f, z_s]
z_fused = MLP(z)
```

它不是最终推荐模型，但必须保留，因为它是判断 GMU、Cross-Attention、Bilinear 是否真正有用的基准。

```python
class LatentConcatFusion(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model),
            nn.LayerNorm(2 * d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, zu, zf, zs):
        z = torch.cat([zu, zf, zs], dim=-1)
        return self.proj(z)
```

物理含义：

```text
让模型在同一时间步内直接看到声学、光纤声学、光热环境三类潜变量；
不显式建模谁调制谁，只依赖后续 MLP 学习非线性关系。
```

局限：

```text
仍然偏向普通特征拼接；
无法显式表达跨模态查询；
无法显式表达乘性交互。
```

---

### 6.2 M1：GMU 门控中期融合

GMU 的核心是让模型为不同模态分配动态门控权重。其动机与多模态融合高度一致：不同样本、不同阶段、不同噪声状态下，各模态可靠性不同，模型应该动态选择依赖哪一类信息。GMU 原论文明确将其作为神经网络内部的中间表示融合单元，使用乘性门控决定模态影响。([arXiv](https://arxiv.org/abs/1702.01992?utm_source=chatgpt.com "Gated Multimodal Units for Information Fusion"))

公式：

[  
\tilde{z}_u = \tanh(W_u z_u)  
]

[  
\tilde{z}_f = \tanh(W_f z_f)  
]

[  
\tilde{z}_s = \tanh(W_s z_s)  
]

[  
[g_u, g_f, g_s] = \operatorname{softmax}(W_g [z_u;z_f;z_s])  
]

[  
z_{\text{fused}} = g_u \tilde{z}_u + g_f \tilde{z}_f + g_s \tilde{z}_s  
]

代码：

```python
class GMUFusion(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()

        self.proj_u = nn.Linear(d_model, d_model)
        self.proj_f = nn.Linear(d_model, d_model)
        self.proj_s = nn.Linear(d_model, d_model)

        self.gate = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3)
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, zu, zf, zs):
        """
        zu, zf, zs: [B, T, d]
        return: [B, T, d]
        """
        hu = torch.tanh(self.proj_u(zu))
        hf = torch.tanh(self.proj_f(zf))
        hs = torch.tanh(self.proj_s(zs))

        g = torch.softmax(self.gate(torch.cat([zu, zf, zs], dim=-1)), dim=-1)
        gu = g[..., 0:1]
        gf = g[..., 1:2]
        gs = g[..., 2:3]

        out = gu * hu + gf * hf + gs * hs
        return self.norm(out), g
```

适合解决的问题：

```text
光学受干扰时，降低 NDIR/慢变量分支依赖；
声学噪声变大时，降低声学分支依赖；
稳定阶段更依赖 NDIR/热导，动态阶段更依赖声学响应斜率。
```

---

### 6.3 M2：Cross-Attention 中期融合

Cross-Attention 是本方案中最推荐的核心模块。Transformer 论文提出的注意力机制本质上是用 Query、Key、Value 计算 token 间的相关性，多头机制可以从不同子空间学习关系。([arXiv](https://arxiv.org/abs/1706.03762?utm_source=chatgpt.com "Attention Is All You Need"))

对你的任务，推荐设计为：

```text
Query: 慢变量 token z_s
Key:   声学 token [z_u, z_f]
Value: 声学 token [z_u, z_f]
```

物理解释：

```text
当前的温度、压力、湿度、声程、活塞位置、NDIR、热导状态，主动去查询超声与光纤声学响应中哪些时间点、哪些声学模式与当前浓度相关。
```

代码：

```python
class SlowQueryAcousticCrossAttention(nn.Module):
    def __init__(self, d_model=128, nhead=4, dropout=0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, zu, zf, zs):
        """
        zu: [B,T,d]
        zf: [B,T,d]
        zs: [B,T,d]
        """
        acoustic_tokens = torch.cat([zu, zf], dim=1)  # [B,2T,d]

        cross, attn_map = self.attn(
            query=zs,
            key=acoustic_tokens,
            value=acoustic_tokens,
            need_weights=True
        )

        h = self.norm1(zs + cross)
        h = self.norm2(h + self.ffn(h))

        return h, attn_map
```

注意：attention map 可以作为辅助解释工具，但不能直接等同于因果解释。它可以帮助观察模型在不同阶段是否更关注超声或光纤声学 token。

---

### 6.4 M3：双向 Cross-Attention

单向 Cross-Attention 是慢变量查询声学信息。双向版本进一步让声学 token 也查询慢变量：

```text
slow -> acoustic: 环境状态查询声学模式；
acoustic -> slow: 声学响应查询对应的环境补偿变量。
```

代码：

```python
class BidirectionalCrossAttentionFusion(nn.Module):
    def __init__(self, d_model=128, nhead=4, dropout=0.1):
        super().__init__()

        self.slow_query_acoustic = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.acoustic_query_slow = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.proj = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        self.norm_s = nn.LayerNorm(d_model)
        self.norm_a = nn.LayerNorm(d_model)

    def forward(self, zu, zf, zs):
        acoustic = 0.5 * (zu + zf)

        slow_ctx, attn_sa = self.slow_query_acoustic(
            query=zs,
            key=acoustic,
            value=acoustic,
            need_weights=True
        )

        acoustic_ctx, attn_as = self.acoustic_query_slow(
            query=acoustic,
            key=zs,
            value=zs,
            need_weights=True
        )

        slow_ctx = self.norm_s(zs + slow_ctx)
        acoustic_ctx = self.norm_a(acoustic + acoustic_ctx)

        z = torch.cat([slow_ctx, acoustic_ctx, zu - zf], dim=-1)
        z = self.proj(z)

        return z, {
            "slow_to_acoustic": attn_sa,
            "acoustic_to_slow": attn_as
        }
```

其中 `zu - zf` 有一定物理意义：它保留超声对射通道与光纤麦克风通道之间的差异信息，用于刻画声速链路与声衰减链路的不一致性。

---

### 6.5 M4：低秩双线性池化

普通拼接主要学习加性交互：

[  
z = W[z_a;z_s]  
]

但很多物理耦合更像乘性交互，例如：

```text
压力 × NDIR 吸收
温度 × 声速
声程 × TOF
气体密度 × 声衰减
热导 × 组分比例
```

完整双线性外积为：

[  
z_a z_s^T  
]

但维度为 (d^2)，计算量过大。因此推荐低秩双线性：

[  
z_{\text{bilinear}} = (W_a z_a) \odot (W_s z_s)  
]

紧凑双线性池化相关工作也指出，外积类交互比简单 concat 或 element-wise sum 更有表达力，但需要压缩来避免高维度问题。([arXiv](https://arxiv.org/abs/1606.01847?utm_source=chatgpt.com "Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding"))

代码：

```python
class LowRankBilinearFusion(nn.Module):
    def __init__(self, d_model=128, rank=64):
        super().__init__()

        self.acoustic_proj = nn.Linear(2 * d_model, rank)
        self.slow_proj = nn.Linear(d_model, rank)

        self.out = nn.Sequential(
            nn.Linear(rank, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, zu, zf, zs):
        acoustic = torch.cat([zu, zf], dim=-1)  # [B,T,2d]

        a = self.acoustic_proj(acoustic)
        s = self.slow_proj(zs)

        z = a * s
        z = self.out(z)

        return z
```

---

### 6.6 M5：推荐主模型 Hybrid Cross-Attention + Bilinear + Gate

最终推荐的中期融合主模型不是单一模块，而是混合结构：

```text
Cross-Attention：学习慢变量如何查询声学动态；
Low-Rank Bilinear：学习物理乘性交互；
GMU Gate：动态平衡不同融合结果。
```

结构：

```text
z_cross = CrossAttention(z_u, z_f, z_s)
z_bilin = Bilinear(z_u, z_f, z_s)
z_concat = MLP([z_u,z_f,z_s])

gate = softmax(MLP([z_cross,z_bilin,z_concat]))

z_fused = g1*z_cross + g2*z_bilin + g3*z_concat
```

代码：

```python
class HybridIntermediateFusion(nn.Module):
    def __init__(self, d_model=128, nhead=4, rank=64, dropout=0.1):
        super().__init__()

        self.cross = SlowQueryAcousticCrossAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout
        )

        self.bilinear = LowRankBilinearFusion(
            d_model=d_model,
            rank=rank
        )

        self.concat = LatentConcatFusion(
            d_model=d_model
        )

        self.gate = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3)
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, zu, zf, zs):
        z_cross, attn_map = self.cross(zu, zf, zs)
        z_bilin = self.bilinear(zu, zf, zs)
        z_concat = self.concat(zu, zf, zs)

        gate_input = torch.cat([z_cross, z_bilin, z_concat], dim=-1)
        g = torch.softmax(self.gate(gate_input), dim=-1)

        z = (
            g[..., 0:1] * z_cross +
            g[..., 1:2] * z_bilin +
            g[..., 2:3] * z_concat
        )

        return self.norm(z), {
            "attn_map": attn_map,
            "fusion_gate": g
        }
```

这个 Hybrid 模型适合作为最终主模型，但不应一开始就直接作为唯一实验。必须先跑 M0、M1、M2、M4，再证明 Hybrid 的提升来自真实的跨模态机制，而不是参数量增加。

---

## 7. 融合后的时序主干

中期融合后得到：

```python
z_fused: [B,120,d_model]
```

接下来需要建模 120 秒动态过程。建议同时比较 TCN 与 Transformer Encoder。

TCN 是适合固定长度时间序列的强基线。Bai 等人的实证研究显示，简单卷积序列模型在多种任务上可以超过典型循环网络，并表现出更长有效记忆，因此 TCN 可以作为你的传感器时序主干候选。([arXiv](https://arxiv.org/abs/1803.01271?utm_source=chatgpt.com "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling")) Transformer Encoder 则适合建模全局阶段关系，例如 baseline、exposure、steady、recovery 之间的长期依赖。Transformer 的优势是并行建模全局 token 关系，但序列长度增加时 self-attention 的计算量更高。([arXiv](https://arxiv.org/abs/1706.03762?utm_source=chatgpt.com "Attention Is All You Need"))

### 7.1 TCN 主干

```python
class TemporalBlock(nn.Module):
    def __init__(self, d_model, dilation, dropout=0.1):
        super().__init__()

        padding = dilation

        self.conv1 = nn.Conv1d(
            d_model, d_model,
            kernel_size=3,
            padding=padding,
            dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            d_model, d_model,
            kernel_size=3,
            padding=padding,
            dilation=dilation
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: [B,T,d]
        """
        residual = x

        h = x.transpose(1, 2)
        h = self.conv1(h)
        h = h[:, :, :x.size(1)]
        h = F.gelu(h)

        h = self.conv2(h)
        h = h[:, :, :x.size(1)]
        h = F.gelu(h)

        h = h.transpose(1, 2)
        h = self.dropout(h)

        return self.norm(residual + h)


class TCNBackbone(nn.Module):
    def __init__(self, d_model=128, dropout=0.1):
        super().__init__()

        self.blocks = nn.Sequential(
            TemporalBlock(d_model, dilation=1, dropout=dropout),
            TemporalBlock(d_model, dilation=2, dropout=dropout),
            TemporalBlock(d_model, dilation=4, dropout=dropout),
            TemporalBlock(d_model, dilation=8, dropout=dropout),
            TemporalBlock(d_model, dilation=16, dropout=dropout)
        )

    def forward(self, x):
        return self.blocks(x)
```

### 7.2 Transformer 主干

```python
class TransformerTemporalBackbone(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=3, dropout=0.1, max_len=120):
        super().__init__()

        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        """
        x: [B,T,d]
        """
        T = x.size(1)
        x = x + self.pos_embed[:, :T, :]
        return self.encoder(x)
```

---

## 8. 序列池化与输出头

你的标签是序列级标签 `[B,4]`，不是每个时间步都有一个浓度标签。因此不能对每个时间步都强制监督。推荐使用阶段感知池化。

### 8.1 简单平均池化

```python
h = z.mean(dim=1)
```

优点是简单稳定。缺点是 baseline、exposure、steady、recovery 权重完全相同。

### 8.2 注意力池化

```python
class AttentionPooling(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )

    def forward(self, z):
        """
        z: [B,T,d]
        """
        a = torch.softmax(self.score(z), dim=1)
        h = (a * z).sum(dim=1)
        return h, a
```

### 8.3 阶段池化

推荐用于解释性实验：

```python
class PhasePooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        """
        z: [B,120,d]
        """
        baseline = z[:, 0:20].mean(dim=1)
        exposure = z[:, 20:70].mean(dim=1)
        steady = z[:, 70:100].mean(dim=1)
        recovery = z[:, 100:120].mean(dim=1)

        return torch.cat([baseline, exposure, steady, recovery], dim=-1)
```

阶段池化后的 head 输入维度为 `4*d_model`。它能直接观察模型到底依赖哪个阶段。

---

## 9. 完整中期融合模型代码

下面是可直接落地的 PyTorch 主模型。它支持多种融合方式：

```text
concat
gmu
cross_attn
bilinear
hybrid
```

```python
class IntermediateFusionRegressor(nn.Module):
    def __init__(
        self,
        fusion_type="hybrid",
        temporal_type="transformer",
        d_model=128,
        out_dim=4,
        use_phase_embed=True,
        pooling="attention",
        dropout=0.1
    ):
        super().__init__()

        self.fusion_type = fusion_type
        self.pooling = pooling
        self.use_phase_embed = use_phase_embed

        self.ultra_encoder = WaveformEncoder(input_len=1000, d_model=d_model)
        self.fiber_encoder = WaveformEncoder(input_len=2000, d_model=d_model)
        self.slow_encoder = SlowEncoder(slow_dim=8, d_model=d_model)

        if use_phase_embed:
            self.phase_embed = nn.Embedding(4, d_model)

        if fusion_type == "concat":
            self.fusion = LatentConcatFusion(d_model=d_model)
        elif fusion_type == "gmu":
            self.fusion = GMUFusion(d_model=d_model)
        elif fusion_type == "cross_attn":
            self.fusion = SlowQueryAcousticCrossAttention(d_model=d_model, nhead=4, dropout=dropout)
        elif fusion_type == "bilinear":
            self.fusion = LowRankBilinearFusion(d_model=d_model, rank=64)
        elif fusion_type == "hybrid":
            self.fusion = HybridIntermediateFusion(d_model=d_model, nhead=4, rank=64, dropout=dropout)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        if temporal_type == "tcn":
            self.temporal = TCNBackbone(d_model=d_model, dropout=dropout)
        elif temporal_type == "transformer":
            self.temporal = TransformerTemporalBackbone(
                d_model=d_model,
                nhead=4,
                num_layers=3,
                dropout=dropout,
                max_len=120
            )
        else:
            raise ValueError(f"Unknown temporal_type: {temporal_type}")

        if pooling == "attention":
            self.pool = AttentionPooling(d_model=d_model)
            head_in = d_model
        elif pooling == "mean":
            self.pool = None
            head_in = d_model
        elif pooling == "phase":
            self.pool = PhasePooling()
            head_in = 4 * d_model
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        self.mean_head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim)
        )

        self.logvar_head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim)
        )

    def forward(self, ultra, fiber, slow, phase_id=None):
        """
        ultra: [B,120,1000]
        fiber: [B,120,2000]
        slow:  [B,120,8]
        phase_id: [B,120], optional
        """
        zu = self.ultra_encoder(ultra)
        zf = self.fiber_encoder(fiber)
        zs = self.slow_encoder(slow)

        if self.use_phase_embed and phase_id is not None:
            pe = self.phase_embed(phase_id)
            zs = zs + pe

        aux = {}

        if self.fusion_type == "concat":
            z = self.fusion(zu, zf, zs)

        elif self.fusion_type == "gmu":
            z, gate = self.fusion(zu, zf, zs)
            aux["gmu_gate"] = gate

        elif self.fusion_type == "cross_attn":
            z, attn_map = self.fusion(zu, zf, zs)
            aux["attn_map"] = attn_map

        elif self.fusion_type == "bilinear":
            z = self.fusion(zu, zf, zs)

        elif self.fusion_type == "hybrid":
            z, fusion_aux = self.fusion(zu, zf, zs)
            aux.update(fusion_aux)

        z = self.temporal(z)

        if self.pooling == "mean":
            h = z.mean(dim=1)
            aux["pool_weight"] = None
        elif self.pooling == "attention":
            h, pool_weight = self.pool(z)
            aux["pool_weight"] = pool_weight
        elif self.pooling == "phase":
            h = self.pool(z)
            aux["pool_weight"] = None

        mu_raw = self.mean_head(h)
        logvar = self.logvar_head(h).clamp(min=-8.0, max=4.0)

        return mu_raw, logvar, aux
```

---

## 10. 输出约束与损失函数

如果标签为四组分比例，推荐使用 softmax 约束：

```python
mu = torch.softmax(mu_raw, dim=-1)
```

如果标签是 `vol%`，建议训练前转换：

```python
y_train = y_train / 100.0
```

预测后再恢复：

```python
y_pred_vol = y_pred * 100.0
```

### 10.1 异方差高斯回归损失

让模型同时预测均值和不确定性：

# [

\mathcal{L}_{\text{nll}}

\frac{1}{K}  
\sum_{k=1}^{K}  
\left[  
\exp(-s_k)(y_k-\mu_k)^2 + s_k  
\right]  
]

其中：

```text
K = 4
s_k = log variance
```

代码：

```python
def heteroscedastic_nll(mu, logvar, y):
    loss = torch.exp(-logvar) * (y - mu) ** 2 + logvar
    return loss.mean()
```

### 10.2 组分和约束

如果没有使用 softmax，而是使用 unconstrained output，则加入：

# [

\mathcal{L}_{\text{sum}}

\left(  
\sum_{k=1}^{4}\mu_k - 1  
\right)^2  
]

代码：

```python
def sum_constraint_loss(mu):
    return ((mu.sum(dim=-1) - 1.0) ** 2).mean()
```

### 10.3 非负约束

如果不用 softmax，可以额外加入：

```python
def nonnegative_penalty(mu):
    return F.relu(-mu).mean()
```

### 10.4 总损失

推荐主损失：

# [

\mathcal{L}

\mathcal{L}*{\text{nll}}  

+ \lambda*{\text{sum}}\mathcal{L}*{\text{sum}}  
+ \lambda*{\text{nonneg}}\mathcal{L}_{\text{nonneg}}  
  ]

如果使用 softmax 输出，则不需要 sum loss 和 nonnegative penalty。

推荐：

```python
mu = torch.softmax(mu_raw, dim=-1)
loss = heteroscedastic_nll(mu, logvar, y)
```

如果前期训练不稳定，可以先用 Smooth L1 预训练：

```python
loss_stage1 = F.smooth_l1_loss(mu, y)
```

再切换到 NLL：

```python
loss_stage2 = heteroscedastic_nll(mu, logvar, y)
```

---

## 11. 训练配置

推荐默认配置：

```yaml
optimizer: AdamW
learning_rate: 3.0e-4
weight_decay: 1.0e-4
batch_size: 16-64
epochs: 100-200
scheduler: linear warmup 5 epochs + cosine decay
gradient_clip_norm: 1.0
mixed_precision: true
early_stopping_patience: 20
main_metric: val_macro_rmse
```

显存紧张时优先调整：

```text
1. batch_size 从 64 降到 32 或 16；
2. d_model 从 128 降到 64；
3. Transformer layers 从 3 降到 2；
4. fusion_type 先用 concat / gmu，再跑 hybrid；
5. 使用 AMP 混合精度；
6. 使用 memmap 按样本读取，不要一次性载入所有波形。
```

---

## 12. 数据集实现建议

由于数据规模较大，建议 Dataset 内部使用 memmap 或懒加载。

```python
class GasWaveformDataset(torch.utils.data.Dataset):
    def __init__(self, arrays, indices, stats, use_phase=True):
        self.arrays = arrays
        self.indices = indices
        self.stats = stats
        self.use_phase = use_phase

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]

        ultra_i16 = self.arrays["ultrasonic_int16"][idx]
        fiber_i16 = self.arrays["fiber_mic_int16"][idx]
        ultra_scale = self.arrays["ultrasonic_scale"][idx]
        fiber_scale = self.arrays["fiber_mic_scale"][idx]

        ultra = ultra_i16.astype("float32") * ultra_scale[..., None]
        fiber = fiber_i16.astype("float32") * fiber_scale[..., None]

        slow = self.arrays["slow"][idx].astype("float32")
        y = self.arrays["y"][idx].astype("float32")

        ultra = (ultra - self.stats["ultra_mean"]) / (self.stats["ultra_std"] + 1e-8)
        fiber = (fiber - self.stats["fiber_mean"]) / (self.stats["fiber_std"] + 1e-8)
        slow = (slow - self.stats["slow_mean"]) / (self.stats["slow_std"] + 1e-8)

        sample = {
            "ultra": torch.from_numpy(ultra),
            "fiber": torch.from_numpy(fiber),
            "slow": torch.from_numpy(slow),
            "y": torch.from_numpy(y)
        }

        if self.use_phase:
            phase = torch.zeros(120, dtype=torch.long)
            phase[20:70] = 1
            phase[70:100] = 2
            phase[100:120] = 3
            sample["phase_id"] = phase

        return sample
```

---

## 13. 实验矩阵

中期融合完整实验至少包含以下模型。

| 编号  | 模型                                                   | 目的                   |
| --- | ---------------------------------------------------- | -------------------- |
| M0  | Independent Encoders + Latent Concat                 | 中期融合最低基线             |
| M1  | Independent Encoders + GMU                           | 验证动态门控是否有效           |
| M2  | Independent Encoders + Cross-Attention               | 验证慢变量查询声学 token 是否有效 |
| M3  | Independent Encoders + Bidirectional Cross-Attention | 验证双向调制是否优于单向调制       |
| M4  | Independent Encoders + Low-Rank Bilinear             | 验证乘性交互是否有效           |
| M5  | Cross-Attention + Bilinear + Gate Hybrid             | 最终推荐主模型              |

时序主干消融：

| 编号  | 主干                  | 目的              |
| --- | ------------------- | --------------- |
| T1  | Mean Pooling only   | 检查是否不需要复杂时序     |
| T2  | TCN                 | 检查局部动态和多尺度响应    |
| T3  | Transformer Encoder | 检查全局阶段依赖        |
| T4  | TCN + Transformer   | 检查局部动态 + 全局依赖组合 |

阶段消融：

| 编号  | 时间范围                  | 目的                         |
| --- | --------------------- | -------------------------- |
| P1  | 0-119                 | 完整动态过程                     |
| P2  | 70-99                 | 只用稳态，模拟传统特征工程              |
| P3  | 20-99                 | exposure + steady，评估缩短检测时间 |
| P4  | 0-119 phase pooling   | 阶段级解释                      |
| P5  | T=20/40/60/80/100/120 | 在线早预测实验                    |

模态消融：

| 编号  | 输入                   | 目的          |
| --- | -------------------- | ----------- |
| A1  | ultra only           | 超声独立能力      |
| A2  | fiber only           | 光纤麦克风独立能力   |
| A3  | slow only            | 光热环境慢变量独立能力 |
| A4  | ultra + fiber        | 纯声学双分支      |
| A5  | ultra + slow         | 超声与慢变量耦合    |
| A6  | fiber + slow         | 光纤声学与慢变量耦合  |
| A7  | ultra + fiber + slow | 完整模型        |

融合机制消融：

| 编号  | 设置                          | 目的       |
| --- | --------------------------- | -------- |
| F1  | concat                      | 普通潜变量拼接  |
| F2  | concat + gate               | 检查门控贡献   |
| F3  | cross-attn without bilinear | 检查注意力贡献  |
| F4  | bilinear without cross-attn | 检查乘性交互贡献 |
| F5  | hybrid full                 | 完整中期融合   |

---

## 14. 鲁棒性实验

中期融合不仅要看 clean test，还要看受扰条件下是否稳定。

建议构造以下扰动：

```text
1. 光学漂移：
   V_NDIR_CH4, V_NDIR_CO2 加偏置或随机游走。

2. 热导漂移：
   V_TCS 加慢变偏移。

3. 压力扰动：
   P_MPa 加阶跃扰动或非线性漂移。

4. 湿度扰动：
   H_RH 加噪声。

5. 声学噪声：
   ultra / fiber 加高斯噪声、幅值衰减、时间偏移。

6. 单模态缺失：
   ultra mask、fiber mask、slow mask。
```

鲁棒性指标：

```text
Clean RMSE
Noisy RMSE
RMSE degradation
Max Relative Error
Noise-RMSE AUC
Uncertainty-error correlation
```

如果 Hybrid 模型在 clean test 上略优，但噪声下明显退化，应考虑加入 modality dropout。训练时随机屏蔽某一模态，让模型避免过度依赖单一路径。

```python
def modality_dropout(zu, zf, zs, p=0.1, training=True):
    if not training or p <= 0:
        return zu, zf, zs

    B = zu.size(0)
    device = zu.device

    mask = torch.bernoulli(torch.full((B, 3), 1 - p, device=device))

    zu = zu * mask[:, 0].view(B, 1, 1)
    zf = zf * mask[:, 1].view(B, 1, 1)
    zs = zs * mask[:, 2].view(B, 1, 1)

    return zu, zf, zs
```

---

## 15. 评价指标

四组分分别计算：

# [

\operatorname{MAE}_k

\frac{1}{N}  
\sum_i  
|y_{i,k}-\hat{y}_{i,k}|  
]

# [

\operatorname{RMSE}_k

\sqrt{  
\frac{1}{N}  
\sum_i  
(y_{i,k}-\hat{y}_{i,k})^2  
}  
]

# [

\operatorname{MRE}_k

\frac{1}{N}  
\sum_i  
\frac{|y_{i,k}-\hat{y}*{i,k}|}{|y*{i,k}|+\epsilon}  
]

整体指标：

```text
Macro-MAE
Macro-RMSE
Macro-MRE
Max Relative Error
Sum Error = abs(sum(pred) - 1)
NLL
Uncertainty-error correlation
```

推荐主指标：

```text
1. Macro-RMSE：主精度指标；
2. Macro-MRE：相对误差指标；
3. MaxRE：极端误差指标；
4. Noise-RMSE-AUC：鲁棒性指标；
5. SumError：组分物理约束指标。
```

---

## 16. 推荐结论路径

最终论文或实验报告建议这样组织结论：

第一步，证明中期融合优于单模态。

```text
Ultra-only、Fiber-only、Slow-only 均不如多模态中期融合，
说明声学、光热和环境变量存在互补信息。
```

第二步，证明中期融合优于简单拼接。

```text
GMU / Cross-Attention / Bilinear 优于 Latent Concat，
说明跨模态交互机制不是简单增加输入维度。
```

第三步，证明 Cross-Attention 能捕获动态物理耦合。

```text
Cross-Attention 在 exposure + steady 阶段优于普通 concat，
说明慢变量对声学响应的调制具有时间相关性。
```

第四步，证明 Bilinear 对物理乘性交互有贡献。

```text
Bilinear 在压力扰动、声程变化、热导耦合条件下更稳定，
说明乘性交互对环境补偿有效。
```

第五步，证明 Hybrid 是最终主模型。

```text
Hybrid Cross-Attention + Bilinear + Gate 在 clean test 与 noisy test 上综合最优，
可作为课题的深度中期融合主模型。
```

---

## 17. 最终推荐主方案

最终推荐如下：

```text
模型名称：
Physics-aware Hybrid Intermediate Fusion Network

输入：
ultrasonic [B,120,1000]
fiber_mic [B,120,2000]
slow [B,120,8]
phase_id [B,120] optional

编码器：
Ultrasonic 1D-CNN Encoder -> [B,120,128]
Fiber Mic 1D-CNN Encoder -> [B,120,128]
Slow MLP + Local TCN Encoder -> [B,120,128]

中期融合：
Cross-Attention:
  slow queries acoustic tokens

Low-Rank Bilinear:
  acoustic latent × slow latent

Gate:
  dynamic weighting of concat / attention / bilinear branches

时序主干：
Transformer Encoder 或 TCN

池化：
Attention Pooling 或 Phase Pooling

输出：
4-component regression
[x_H2, x_CH4, x_CO2, x_N2]

损失：
Heteroscedastic Gaussian NLL
+ optional composition constraint

主消融：
concat / gmu / cross-attn / bilinear / hybrid
```

如果只选一个模型作为主实验，建议：

```text
HybridIntermediateFusion
+ TransformerTemporalBackbone
+ AttentionPooling
+ Heteroscedastic NLL
+ Modality Dropout
```

如果算力有限，建议：

```text
GMUFusion
+ TCNBackbone
+ MeanPooling
```

如果目标是论文创新性，建议主推：

```text
Slow-query Acoustic Cross-Attention
+ Low-Rank Bilinear Physical Interaction
+ Gate-controlled Hybrid Fusion
```

这条路线比单纯 CNN/GRU 拼接更能体现你的课题特点：高频声学波形负责捕获传播与衰减细节，低频光热环境变量负责解释压力、温度、热导、光吸收等物理调制，中期融合层负责学习二者之间的底层耦合关系。
