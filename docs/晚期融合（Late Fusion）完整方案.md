# 晚期融合（Late Fusion）完整方案：面向多源异构气体传感系统的独立分支预测与动态置信度集成

## 1. 方案定位

晚期融合，又称决策级融合。它的核心不是在输入层或潜变量层强行混合模态，而是让每个模态先形成独立预测，再在输出端根据可靠性、置信度、扰动敏感性或环境状态进行加权集成。

对你的课题来说，晚期融合不是“最复杂”的方案，但它是**工程鲁棒性最关键**的方案。原因是你的硬件系统存在明确的传感器失效风险：NDIR 可能受湿度、压力展宽或光路污染影响；热导传感器可能受温度梯度和基线漂移影响；声学链路可能受噪声、声程变化、探头响应和压力扰动影响。专利方案中已经采用了“声学、光学、热导特征解耦输入 + SVM 基学习器 + 蒙特卡洛扰动漂移 MSE + 动态逆不确定性权重 + Ridge 元学习器”的典型晚期融合思想。

你的数据也天然适合构建深度版晚期融合：每个样本 `120` 个时间步，超声波形为 `[120,1000]`，光纤麦克风波形为 `[120,2000]`，慢变量为 `[120,8]`，标签是 `[x_H2, x_CH4, x_CO2, x_N2]` 四组分连续回归。

从方法依据看，深度集成模型常用于提升预测性能和不确定性估计。Deep Ensembles 论文指出，训练多个概率神经网络并组合预测，可以得到简单、可并行、效果强的预测不确定性估计方法，并且在分布外样本上能够表达更高不确定性。([NeurIPS Proceedings](https://proceedings.neurips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles.pdf "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles")) Kendall 和 Gal 对不确定性进行区分：aleatoric uncertainty 表示观测噪声，epistemic uncertainty 表示模型认知不确定性，这一划分非常适合你的“硬件噪声 + 模型泛化”双重问题。([arXiv](https://arxiv.org/abs/1703.04977 "[1703.04977] What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?")) Mixture of Experts 则提供了“多个专家网络 + gating network 动态分配权重”的基本范式，适合改造为多传感器动态加权融合。([多伦多大学计算机科学系](https://www.cs.toronto.edu/~fritz/absps/jjnh91.pdf "Adaptive Mixtures of Local Experts"))

---

## 2. 晚期融合的物理假设

晚期融合的核心物理假设如下：

```text
不同传感模态具有不同的物理敏感性和失效模式。

声学模态：
  对声速、TOF、声衰减、声压能量变化敏感；
  对温度、压力、声程、探头响应、环境噪声敏感。

光学模态：
  对 CH4、CO2 等红外吸收响应敏感；
  对压力展宽、湿度、光路污染、温度漂移敏感。

热导模态：
  对气体混合热导率敏感；
  对温度梯度、气流扰动、桥路基线漂移敏感。

环境与结构变量：
  T_C、P_MPa、H_RH、L_m、piston_position_m 不是直接浓度传感器，
  但它们决定了声学、光学、热导响应的补偿条件。
```

因此，晚期融合不追求让所有模态在最早阶段互相纠缠，而是保留各模态独立判断能力。最终融合器只回答一个问题：

```text
当前样本、当前工况、当前噪声状态下，应该相信哪个模态更多？
```

这与专利中的动态权重机制一致：当某模态在蒙特卡洛扰动下预测漂移 MSE 增大时，应降低其融合权重；当某模态预测稳定时，应提高其权重。

---

## 3. 总体网络结构

晚期融合主结构：

```text
ultrasonic [B,120,1000]
        ↓
Ultra-only Deep Regressor
        ↓
mu_u [B,4], logvar_u [B,4]

fiber_mic [B,120,2000]
        ↓
Fiber-only Deep Regressor
        ↓
mu_f [B,4], logvar_f [B,4]

slow [B,120,8]
        ↓
Slow-only Deep Regressor
        ↓
mu_s [B,4], logvar_s [B,4]

optional:
optical-only: V_NDIR_CH4, V_NDIR_CO2
thermal-only: V_TCS
env-only: T_C, P_MPa, H_RH, L_m, piston_position_m

        ↓
Dynamic Decision Fusion
        ↓
final prediction [B,4]
```

建议从三分支开始：

```text
Branch 1: ultrasonic-only
Branch 2: fiber_mic-only
Branch 3: slow-only
```

如果要更贴近硬件物理边界，可以扩展为五分支：

```text
Branch 1: ultrasonic-only
Branch 2: fiber_mic-only
Branch 3: optical-only, 即 V_NDIR_CH4 + V_NDIR_CO2
Branch 4: thermal-only, 即 V_TCS
Branch 5: env/control-only, 即 T_C + P_MPa + H_RH + L_m + piston_position_m
```

三分支适合先做深度学习主实验，五分支适合做论文中的物理解释与鲁棒性实验。

---

## 4. 数据输入与预处理

原始输入：

```python
ultrasonic_int16: [B,120,1000]
fiber_mic_int16:  [B,120,2000]
ultrasonic_scale: [B,120]
fiber_mic_scale:  [B,120]
slow:             [B,120,8]
y:                [B,4]
```

反量化：

```python
ultra = ultrasonic_int16.float() * ultrasonic_scale.unsqueeze(-1)
fiber = fiber_mic_int16.float() * fiber_mic_scale.unsqueeze(-1)
```

标准化只使用 train split 统计量：

```python
ultra = (ultra - ultra_mean_train) / (ultra_std_train + 1e-8)
fiber = (fiber - fiber_mean_train) / (fiber_std_train + 1e-8)
slow  = (slow  - slow_mean_train)  / (slow_std_train  + 1e-8)
```

不要对单条声学波形做独立峰值归一化，因为幅值、能量和衰减本身可能携带浓度信息。

---

## 5. 独立模态预测分支

### 5.1 声学分支：Ultra-only Regressor

输入：

```python
ultra: [B,120,1000]
```

输出：

```python
mu_u:     [B,4]
logvar_u: [B,4]
```

该分支只看超声对射通道，目标是学习 TOF、声速、声衰减、传播路径变化等信息对四组分浓度的映射。

### 5.2 光纤声学分支：Fiber-only Regressor

输入：

```python
fiber: [B,120,2000]
```

输出：

```python
mu_f:     [B,4]
logvar_f: [B,4]
```

该分支只看光纤麦克风通道，目标是学习声压、能量衰减、波形包络、频率响应与气体组分的关系。

### 5.3 慢变量分支：Slow-only Regressor

输入：

```python
slow: [B,120,8]
```

输出：

```python
mu_s:     [B,4]
logvar_s: [B,4]
```

慢变量包括：

```text
V_NDIR_CH4
V_NDIR_CO2
V_TCS
T_C
P_MPa
H_RH
L_m
piston_position_m
```

该分支相当于“光学 + 热导 + 环境补偿 + 结构状态”的综合低频分支。

---

## 6. PyTorch 基础模块

### 6.1 1D-CNN 波形编码器

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
    def __init__(self, d_model=128):
        super().__init__()

        self.net = nn.Sequential(
            ConvBlock1D(1, 32, k=15, s=2),
            ConvBlock1D(32, 64, k=11, s=2),
            ConvBlock1D(64, 128, k=7, s=2),
            ConvBlock1D(128, d_model, k=5, s=2),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        """
        x: [B,T,L]
        return: [B,T,d_model]
        """
        B, T, L = x.shape
        x = x.reshape(B * T, 1, L)
        h = self.net(x).squeeze(-1)
        return h.reshape(B, T, -1)
```

### 6.2 慢变量编码器

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

        self.temporal = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, slow):
        """
        slow: [B,T,8]
        return: [B,T,d_model]
        """
        h = self.input_proj(slow)
        ht = self.temporal(h.transpose(1, 2)).transpose(1, 2)
        return self.norm(h + ht)
```

### 6.3 时序主干

```python
class TemporalBackbone(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()

        self.pos_embed = nn.Parameter(torch.zeros(1, 120, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        """
        x: [B,T,d_model]
        """
        T = x.size(1)
        x = x + self.pos_embed[:, :T]
        return self.encoder(x)
```

### 6.4 注意力池化

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
        z: [B,T,d_model]
        """
        a = torch.softmax(self.score(z), dim=1)
        h = (a * z).sum(dim=1)
        return h, a
```

---

## 7. 单模态概率回归器

每个分支不仅输出预测均值，还输出预测方差。这样晚期融合时可以根据不确定性动态加权。

```python
class WaveformUnimodalRegressor(nn.Module):
    def __init__(self, d_model=128, out_dim=4):
        super().__init__()

        self.encoder = WaveformEncoder(d_model=d_model)
        self.temporal = TemporalBackbone(d_model=d_model)
        self.pool = AttentionPooling(d_model=d_model)

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

    def forward(self, x):
        z = self.encoder(x)
        z = self.temporal(z)
        h, pool_weight = self.pool(z)

        mu_raw = self.mean_head(h)
        logvar = self.logvar_head(h).clamp(min=-8.0, max=4.0)

        return mu_raw, logvar, {"pool_weight": pool_weight}


class SlowUnimodalRegressor(nn.Module):
    def __init__(self, slow_dim=8, d_model=128, out_dim=4):
        super().__init__()

        self.encoder = SlowEncoder(slow_dim=slow_dim, d_model=d_model)
        self.temporal = TemporalBackbone(d_model=d_model)
        self.pool = AttentionPooling(d_model=d_model)

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

    def forward(self, slow):
        z = self.encoder(slow)
        z = self.temporal(z)
        h, pool_weight = self.pool(z)

        mu_raw = self.mean_head(h)
        logvar = self.logvar_head(h).clamp(min=-8.0, max=4.0)

        return mu_raw, logvar, {"pool_weight": pool_weight}
```

如果标签是比例形式，使用：

```python
mu = torch.softmax(mu_raw, dim=-1)
```

如果标签是 `vol%`，建议训练前除以 `100`，预测后再乘回 `100`。

---

## 8. 晚期融合机制设计

### 8.1 L0：等权平均融合

这是最简单的晚期融合基线：

# [

\hat{y}

\frac{1}{M}  
\sum_{m=1}^{M}  
\mu_m  
]

代码：

```python
def equal_average_fusion(mus):
    """
    mus: [B,M,4]
    """
    return mus.mean(dim=1)
```

用途：

```text
作为最低基线。
如果复杂动态权重无法明显超过等权平均，说明权重机制没有真正发挥作用。
```

---

### 8.2 L1：验证集静态权重融合

根据验证集上每个模态的 RMSE 设定静态权重：

# [

w_m

\frac{1 / (\operatorname{RMSE}_m + \epsilon)}  
{\sum_j 1 / (\operatorname{RMSE}_j + \epsilon)}  
]

# [

\hat{y}

\sum_m w_m \mu_m  
]

代码：

```python
def static_rmse_weight_fusion(mus, val_rmse, eps=1e-8):
    """
    mus: [B,M,4]
    val_rmse: [M] or [M,4]
    """
    inv = 1.0 / (val_rmse + eps)
    weights = inv / inv.sum(dim=0, keepdim=True)

    if weights.dim() == 1:
        weights = weights.view(1, -1, 1)
    else:
        weights = weights.transpose(0, 1).unsqueeze(0)

    return (weights * mus).sum(dim=1), weights
```

用途：

```text
检验某些模态是否整体更可靠。
缺点是权重固定，无法处理单个样本内的传感器失效。
```

---

### 8.3 L2：异方差不确定性加权融合

每个分支输出 `logvar_m`，即该模态对每个组分预测的不确定性。权重定义为：

# [

w_{m,k}

\frac{\exp(-s_{m,k})}  
{\sum_j \exp(-s_{j,k})}  
]

其中 (s_{m,k}) 是第 (m) 个模态对第 (k) 个气体组分的 log variance。

最终预测：

# [

\hat{y}_k

\sum_m w_{m,k}\mu_{m,k}  
]

代码：

```python
def uncertainty_weighted_fusion(mus, logvars):
    """
    mus:     [B,M,4]
    logvars: [B,M,4]
    """
    weights = torch.softmax(-logvars, dim=1)
    mu_fused = (weights * mus).sum(dim=1)

    var_fused = (weights * torch.exp(logvars)).sum(dim=1)
    logvar_fused = torch.log(var_fused + 1e-8)

    return mu_fused, logvar_fused, weights
```

该方法适合处理 aleatoric uncertainty，即输入相关的观测噪声。Kendall 和 Gal 的不确定性建模工作指出，异方差不确定性是输入相关的，不同输入可以具有不同噪声水平；这与传感器在不同工况下噪声不同的情况一致。([arXiv](https://arxiv.org/abs/1703.04977 "[1703.04977] What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"))

---

### 8.4 L3：Drift-MSE 动态权重融合

这是最贴近你专利方案的深度版晚期融合。

对每个模态 (m)，先得到干净预测：

# [

\mu_{m}^{0}

f_m(x_m)  
]

然后对输入注入 (K) 次符合硬件噪声特性的扰动：

# [

x_{m}^{(i)}

x_m + \epsilon_m^{(i)}  
]

再次预测：

# [

\mu_{m}^{(i)}

f_m(x_m^{(i)})  
]

计算漂移均方误差：

# [

D_m

\frac{1}{K}  
\sum_{i=1}^{K}  
\left|  
\mu_m^{(i)} - \mu_m^0  
\right|_2^2  
]

动态权重：

# [

w_m

\frac{1/(D_m + C_m + \epsilon)}  
{\sum_j 1/(D_j + C_j + \epsilon)}  
]

其中 (C_m) 是模态固有误差常数，可由验证集单模态 RMSE 或 NLL 估计。

代码：

```python
@torch.no_grad()
def drift_mse_weight(
    model,
    x,
    noise_std,
    K=8,
    base_mu=None,
    normalize_output=True
):
    """
    model: unimodal regressor
    x: input tensor
    noise_std: scalar or broadcastable tensor
    return: drift score [B]
    """
    if base_mu is None:
        base_raw, _, _ = model(x)
        base_mu = torch.softmax(base_raw, dim=-1) if normalize_output else base_raw

    drifts = []

    for _ in range(K):
        noise = torch.randn_like(x) * noise_std
        x_noisy = x + noise

        mu_raw, _, _ = model(x_noisy)
        mu = torch.softmax(mu_raw, dim=-1) if normalize_output else mu_raw

        drift = ((mu - base_mu) ** 2).mean(dim=-1)
        drifts.append(drift)

    drifts = torch.stack(drifts, dim=0).mean(dim=0)
    return drifts
```

融合：

```python
def drift_mse_fusion(mus, drift_scores, C=None, eps=1e-8):
    """
    mus: [B,M,4]
    drift_scores: [B,M]
    C: [M], modality baseline error constants
    """
    if C is not None:
        drift_scores = drift_scores + C.view(1, -1)

    inv = 1.0 / (drift_scores + eps)
    weights = inv / inv.sum(dim=1, keepdim=True)

    mu_fused = (weights.unsqueeze(-1) * mus).sum(dim=1)

    return mu_fused, weights
```

物理含义：

```text
某模态对小扰动极其敏感，说明该模态当前处于不稳定状态；
该模态权重应降低。

某模态在噪声扰动下注入后预测仍稳定，说明其当前可信度较高；
该模态权重应提高。
```

这与专利中的“蒙特卡洛噪声模拟 + 预测漂移 MSE + 逆不确定性动态权重”一致，只是把 SVM 基学习器替换为深度分支。

---

### 8.5 L4：Mixture-of-Experts Gating 融合

MoE 版本让一个 gating network 根据当前输入状态直接预测模态权重。经典 Adaptive Mixtures of Local Experts 的思想是由 gating network 将样本分配给一个或多个专家网络，每个专家处理完整问题的某个子区域。([多伦多大学计算机科学系](https://www.cs.toronto.edu/~fritz/absps/jjnh91.pdf "Adaptive Mixtures of Local Experts"))

对你的任务，专家就是各模态预测分支：

```text
Expert 1: ultrasonic branch
Expert 2: fiber branch
Expert 3: slow branch
```

Gating 输入可以使用：

```text
各模态预测均值
各模态 logvar
慢变量统计量
阶段统计量
传感器质量指标
```

公式：

# [

g

\operatorname{softmax}  
(  
\operatorname{MLP}  
(  
[\mu_u,\mu_f,\mu_s,s_u,s_f,s_s,q]  
)  
)  
]

# [

\hat{y}

g_u \mu_u + g_f \mu_f + g_s \mu_s  
]

代码：

```python
class DecisionGateFusion(nn.Module):
    def __init__(self, num_modalities=3, out_dim=4, context_dim=16, hidden=128):
        super().__init__()

        in_dim = num_modalities * out_dim * 2 + context_dim

        self.gate = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_modalities)
        )

    def forward(self, mus, logvars, context):
        """
        mus:     [B,M,4]
        logvars: [B,M,4]
        context: [B,context_dim]
        """
        B, M, K = mus.shape

        x = torch.cat([
            mus.reshape(B, M * K),
            logvars.reshape(B, M * K),
            context
        ], dim=-1)

        weights = torch.softmax(self.gate(x), dim=-1)
        mu_fused = (weights.unsqueeze(-1) * mus).sum(dim=1)

        return mu_fused, weights
```

Context 构造示例：

```python
class SlowContextEncoder(nn.Module):
    def __init__(self, slow_dim=8, context_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * slow_dim, 64),
            nn.GELU(),
            nn.Linear(64, context_dim)
        )

    def forward(self, slow):
        """
        slow: [B,120,8]
        """
        mean = slow.mean(dim=1)
        std = slow.std(dim=1)
        first = slow[:, 0]
        last = slow[:, -1]

        stat = torch.cat([mean, std, first, last], dim=-1)
        return self.net(stat)
```

---

## 9. 完整深度晚期融合模型

```python
class LateFusionRegressor(nn.Module):
    def __init__(
        self,
        d_model=128,
        out_dim=4,
        fusion_type="uncertainty",
        normalize_output=True
    ):
        super().__init__()

        self.fusion_type = fusion_type
        self.normalize_output = normalize_output

        self.ultra_model = WaveformUnimodalRegressor(
            d_model=d_model,
            out_dim=out_dim
        )

        self.fiber_model = WaveformUnimodalRegressor(
            d_model=d_model,
            out_dim=out_dim
        )

        self.slow_model = SlowUnimodalRegressor(
            slow_dim=8,
            d_model=d_model,
            out_dim=out_dim
        )

        self.context_encoder = SlowContextEncoder(
            slow_dim=8,
            context_dim=16
        )

        self.gate_fusion = DecisionGateFusion(
            num_modalities=3,
            out_dim=out_dim,
            context_dim=16,
            hidden=128
        )

    def normalize_mu(self, mu_raw):
        if self.normalize_output:
            return torch.softmax(mu_raw, dim=-1)
        return mu_raw

    def forward(self, ultra, fiber, slow, sensor_mask=None):
        """
        ultra: [B,120,1000]
        fiber: [B,120,2000]
        slow:  [B,120,8]
        sensor_mask: [B,3], 1 means available, 0 means failed
        """
        mu_u_raw, lv_u, aux_u = self.ultra_model(ultra)
        mu_f_raw, lv_f, aux_f = self.fiber_model(fiber)
        mu_s_raw, lv_s, aux_s = self.slow_model(slow)

        mu_u = self.normalize_mu(mu_u_raw)
        mu_f = self.normalize_mu(mu_f_raw)
        mu_s = self.normalize_mu(mu_s_raw)

        mus = torch.stack([mu_u, mu_f, mu_s], dim=1)
        logvars = torch.stack([lv_u, lv_f, lv_s], dim=1)

        if sensor_mask is not None:
            mask = sensor_mask.unsqueeze(-1)
            logvars = logvars + (1.0 - mask) * 1e4

        if self.fusion_type == "equal":
            weights = torch.ones(
                mus.size(0), mus.size(1),
                device=mus.device
            ) / mus.size(1)
            mu_fused = (weights.unsqueeze(-1) * mus).sum(dim=1)
            fused_logvar = logvars.mean(dim=1)

        elif self.fusion_type == "uncertainty":
            mu_fused, fused_logvar, weights = uncertainty_weighted_fusion(
                mus, logvars
            )

        elif self.fusion_type == "gate":
            context = self.context_encoder(slow)
            mu_fused, weights = self.gate_fusion(mus, logvars, context)
            fused_logvar = (weights.unsqueeze(-1) * logvars).sum(dim=1)

        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        aux = {
            "mus": mus,
            "logvars": logvars,
            "weights": weights,
            "ultra_aux": aux_u,
            "fiber_aux": aux_f,
            "slow_aux": aux_s
        }

        return mu_fused, fused_logvar, aux
```

注意：`drift_mse` 版本通常不放在 `forward` 里，因为它需要多次蒙特卡洛扰动，推理成本较高。建议作为 `predict_with_drift_mse()` 单独实现。

---

## 10. Drift-MSE 推理函数

```python
@torch.no_grad()
def predict_with_drift_mse(
    late_model,
    ultra,
    fiber,
    slow,
    noise_std_ultra,
    noise_std_fiber,
    noise_std_slow,
    K=8,
    C=None
):
    late_model.eval()

    mu_u_raw, _, _ = late_model.ultra_model(ultra)
    mu_f_raw, _, _ = late_model.fiber_model(fiber)
    mu_s_raw, _, _ = late_model.slow_model(slow)

    mu_u = torch.softmax(mu_u_raw, dim=-1)
    mu_f = torch.softmax(mu_f_raw, dim=-1)
    mu_s = torch.softmax(mu_s_raw, dim=-1)

    mus = torch.stack([mu_u, mu_f, mu_s], dim=1)

    D_u = drift_mse_weight(
        late_model.ultra_model,
        ultra,
        noise_std=noise_std_ultra,
        K=K,
        base_mu=mu_u
    )

    D_f = drift_mse_weight(
        late_model.fiber_model,
        fiber,
        noise_std=noise_std_fiber,
        K=K,
        base_mu=mu_f
    )

    D_s = drift_mse_weight(
        late_model.slow_model,
        slow,
        noise_std=noise_std_slow,
        K=K,
        base_mu=mu_s
    )

    drift_scores = torch.stack([D_u, D_f, D_s], dim=1)

    if C is not None:
        C = torch.as_tensor(C, device=drift_scores.device, dtype=drift_scores.dtype)

    mu_fused, weights = drift_mse_fusion(
        mus=mus,
        drift_scores=drift_scores,
        C=C
    )

    return mu_fused, weights, drift_scores
```

噪声标准差建议来源：

```text
ultra: 训练集声学噪声估计，或根据硬件 DAQ 本底噪声设定；
fiber: 光纤麦克风静态背景段估计；
slow: 各慢变量通道的训练集 baseline 波动标准差；
C_m: 各单模态在验证集上的 Macro-RMSE 或 NLL。
```

---

## 11. 损失函数设计

### 11.1 单分支异方差 NLL

每个分支单独训练：

# [

\mathcal{L}_{m}

\frac{1}{K}  
\sum_{k=1}^{K}  
\left[  
\exp(-s_{m,k})  
(y_k-\mu_{m,k})^2  

+ s_{m,k}  
  \right]  
  ]

代码：

```python
def heteroscedastic_nll(mu, logvar, y):
    return (torch.exp(-logvar) * (y - mu) ** 2 + logvar).mean()
```

Deep Ensembles 论文也强调使用 proper scoring rule，如 NLL，来训练概率预测网络，从而鼓励预测分布校准。([NeurIPS Proceedings](https://proceedings.neurips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles.pdf "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"))

### 11.2 融合输出损失

# [

\mathcal{L}_{\text{fused}}

\operatorname{NLL}  
(  
\hat{\mu},  
\hat{s},  
y  
)  
]

### 11.3 单模态辅助损失

为了防止某些分支退化，保留单模态监督：

# [

\mathcal{L}_{\text{branch}}

\sum_m  
\alpha_m  
\operatorname{NLL}  
(  
\mu_m,  
s_m,  
y  
)  
]

### 11.4 权重熵正则

避免 gating 过早塌缩到单一模态：

# [

\mathcal{L}_{\text{entropy}}

\frac{1}{B}  
\sum_i  
\sum_m  
w_{i,m}  
\log(w_{i,m}+\epsilon)  
]

训练早期可以鼓励权重有一定熵，后期逐渐减弱：

```python
def weight_entropy_loss(weights, eps=1e-8):
    entropy = -(weights * torch.log(weights + eps)).sum(dim=1).mean()
    return -entropy
```

注意这里返回的是 `-entropy`，加入 loss 后会鼓励更高熵。

### 11.5 总损失

# [

\mathcal{L}

\mathcal{L}*{\text{fused}}  

+ \lambda*{\text{branch}}  
  \mathcal{L}*{\text{branch}}  
+ \lambda*{\text{entropy}}  
  \mathcal{L}_{\text{entropy}}  
  ]

推荐：

```python
lambda_branch = 0.3
lambda_entropy = 0.01
```

---

## 12. 训练流程

晚期融合不要从一开始端到端乱训。推荐三阶段训练。

### 阶段 1：单模态独立训练

分别训练：

```text
Ultra-only model
Fiber-only model
Slow-only model
```

目的：

```text
保证每个分支都有独立预测能力；
获得各模态验证集 RMSE；
获得 C_m 固有误差常数；
观察单模态极限性能。
```

### 阶段 2：冻结分支，训练融合器

冻结各分支 encoder 和 head，只训练：

```text
uncertainty weighting calibration
gate fusion network
static meta learner
```

目的：

```text
避免融合器训练初期反向传播破坏单模态能力；
让 gating 先学习如何组合已有专家。
```

### 阶段 3：小学习率联合微调

解冻全部模型，用小学习率微调：

```text
lr_branch = 1e-5
lr_fusion = 1e-4
```

目的：

```text
让单模态分支适应融合目标；
避免强模态完全支配弱模态。
```

---

## 13. 训练配置

推荐默认配置：

```yaml
optimizer: AdamW
branch_learning_rate: 3.0e-4
fusion_learning_rate: 1.0e-4
finetune_branch_learning_rate: 1.0e-5
weight_decay: 1.0e-4
batch_size: 16-64
epochs_branch: 80-120
epochs_fusion: 30-60
epochs_finetune: 30-60
scheduler: linear warmup + cosine decay
gradient_clip_norm: 1.0
mixed_precision: true
early_stopping_patience: 20
main_metric: val_macro_rmse
```

---

## 14. 传感器失效与鲁棒性训练

晚期融合的价值必须通过失效实验体现。建议训练时引入 sensor dropout。

```python
def apply_sensor_dropout(ultra, fiber, slow, p=0.1, training=True):
    if not training or p <= 0:
        sensor_mask = torch.ones(ultra.size(0), 3, device=ultra.device)
        return ultra, fiber, slow, sensor_mask

    B = ultra.size(0)
    device = ultra.device

    sensor_mask = torch.bernoulli(
        torch.full((B, 3), 1.0 - p, device=device)
    )

    ultra = ultra * sensor_mask[:, 0].view(B, 1, 1)
    fiber = fiber * sensor_mask[:, 1].view(B, 1, 1)
    slow  = slow  * sensor_mask[:, 2].view(B, 1, 1)

    return ultra, fiber, slow, sensor_mask
```

扰动增强建议：

```text
1. Ultra noise:
   ultra += Gaussian noise
   ultra *= random amplitude attenuation
   ultra = random time shift

2. Fiber noise:
   fiber += Gaussian noise
   fiber *= amplitude drift

3. NDIR drift:
   V_NDIR_CH4 / V_NDIR_CO2 加偏置、随机游走或遮挡型衰减

4. TCS drift:
   V_TCS 加慢变随机游走

5. Pressure shift:
   P_MPa 加阶跃扰动

6. Humidity noise:
   H_RH 加非平稳噪声

7. Modality missing:
   ultra/fiber/slow 整体置零或 mask
```

---

## 15. 实验矩阵

### 15.1 单模态基线

| 编号  | 模型           | 输入                                | 目的              |
| --- | ------------ | --------------------------------- | --------------- |
| U1  | Ultra-only   | ultrasonic                        | 验证超声独立预测能力      |
| F1  | Fiber-only   | fiber_mic                         | 验证光纤声学独立能力      |
| S1  | Slow-only    | slow 8通道                          | 验证光热环境变量独立能力    |
| O1  | Optical-only | NDIR_CH4 + NDIR_CO2               | 验证光学独立能力        |
| T1  | Thermal-only | V_TCS                             | 验证热导独立能力        |
| E1  | Env-only     | T_C + P_MPa + H_RH + L_m + piston | 验证环境变量是否存在泄漏或捷径 |

### 15.2 晚期融合主实验

| 编号  | 融合方式                               | 目的          |
| --- | ---------------------------------- | ----------- |
| L0  | Equal Average                      | 最低晚期融合基线    |
| L1  | Static Val-RMSE Weight             | 验证静态可靠性权重   |
| L2  | Heteroscedastic Uncertainty Weight | 验证预测方差加权    |
| L3  | Drift-MSE Dynamic Weight           | 对齐专利动态置信度机制 |
| L4  | Gating Network / MoE               | 验证数据驱动动态权重  |
| L5  | Drift-MSE + Gating Hybrid          | 最终鲁棒主模型     |

### 15.3 鲁棒性实验

| 编号  | 扰动         | 目标         |
| --- | ---------- | ---------- |
| R1  | NDIR 偏置    | 模拟光学基线漂移   |
| R2  | NDIR 衰减    | 模拟光路遮挡或污染  |
| R3  | V_TCS 随机游走 | 模拟热导桥路漂移   |
| R4  | P_MPa 阶跃   | 模拟压力突变     |
| R5  | ultra 加噪声  | 模拟声学链路噪声   |
| R6  | fiber 幅值衰减 | 模拟光纤声压响应衰减 |
| R7  | 单模态缺失      | 模拟硬件失效     |
| R8  | 多模态复合扰动    | 模拟极端工业现场   |

---

## 16. 评价指标

### 16.1 基础精度指标

对每个组分计算：

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
```

### 16.2 鲁棒性指标

```text
Clean RMSE
Noisy RMSE
RMSE Degradation = Noisy RMSE - Clean RMSE
Relative Degradation = Noisy RMSE / Clean RMSE
Noise-RMSE-AUC
Worst-case MRE
Sensor-failure recovery score
```

### 16.3 不确定性指标

```text
NLL
Calibration error
Uncertainty-error correlation
Prediction interval coverage
```

Deep Ensembles 论文强调校准和分布外泛化是评估预测不确定性的关键维度，因此晚期融合不应只报告 RMSE，还应报告 NLL、校准度和不确定性-误差相关性。([NeurIPS Proceedings](https://proceedings.neurips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles.pdf "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"))

---

## 17. 推荐最终主模型

建议最终主推两个晚期融合模型。

### 工程部署主模型

```text
LateFusion-DriftMSE

Branches:
  Ultra-only CNN-Transformer
  Fiber-only CNN-Transformer
  Slow-only MLP-Transformer

Fusion:
  MC perturbation
  Drift-MSE
  inverse uncertainty weighting

Loss:
  branch NLL
  fused NLL

Strength:
  与专利动态置信度机制一致
  可解释性强
  适合证明抗传感器失效能力
```

### 深度学习主模型

```text
LateFusion-MoE-Uncertainty

Branches:
  Ultra-only probabilistic regressor
  Fiber-only probabilistic regressor
  Slow-only probabilistic regressor

Fusion:
  heteroscedastic uncertainty weighting
  gating network
  sensor dropout training

Loss:
  fused NLL
  branch auxiliary NLL
  gate entropy regularization

Strength:
  端到端训练
  适合与 Early Fusion / Intermediate Fusion 做深度模型对比
```

---

## 18. 与早期、中期融合的对照关系

晚期融合在实验中的定位应写清楚：

```text
Early Fusion:
  追求端到端信息利用上限；
  风险是高维声学淹没慢变量，鲁棒性不一定好。

Intermediate Fusion:
  追求跨模态物理耦合建模；
  是精度与机制解释的主线模型。

Late Fusion:
  追求硬件容错与抗扰；
  是工程部署和鲁棒性验证的主线模型。
```

预期结果：

```text
Clean test:
  Intermediate Fusion 可能最优；
  Late Fusion 可能略低，但应接近。

Sensor failure test:
  Late Fusion 应显著优于 Early Fusion 和普通 Intermediate Fusion。

Noise shift / pressure shift:
  Drift-MSE Late Fusion 应最稳定。

Single modality missing:
  Late Fusion 应具备最小性能退化。
```

---

## 19. 论文表述建议

可以直接写成：

```text
为验证多模态系统在局部传感器失效与基线漂移条件下的鲁棒性，本文构建了决策级晚期融合模型。该模型为超声波形、光纤麦克风波形和低频光热环境变量分别建立独立深度回归分支，每个分支输出四组分浓度预测及其输入相关不确定性。融合阶段不直接拼接原始特征，而是根据各分支预测方差、蒙特卡洛扰动下的预测漂移 MSE 或门控网络输出的动态置信度，对多模态预测结果进行自适应加权。该结构保留了各传感模态的独立物理响应边界，并允许系统在某一模态受噪声、漂移或硬件失效影响时自动降低其决策权重，从而提升复杂工况下的浓度反演稳定性。
```

最终建议：**晚期融合不要只做等权平均，必须至少实现 `Equal Average`、`Uncertainty Weighted`、`Drift-MSE Dynamic Weight` 和 `MoE Gating` 四个版本。**其中 `Drift-MSE Dynamic Weight` 是最贴合你硬件专利逻辑的版本，`MoE + Uncertainty` 是最适合作为深度学习论文模型的版本。
