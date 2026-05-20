from __future__ import annotations

import torch


class WeightedMSELoss(torch.nn.Module):
    """按列加权的 MSE。权重通常取 1/var_i，并归一到 mean=1 保持整体 loss 量级稳定。"""

    def __init__(self, weights: torch.Tensor):
        super().__init__()
        if weights.ndim != 1:
            raise ValueError(f"label_weights must be 1D, got shape {tuple(weights.shape)}")
        normalized = weights / weights.mean()
        self.register_buffer("weights", normalized.float())

    def forward(self, pred, target):
        sq = (pred - target) ** 2
        return (sq * self.weights.to(sq.dtype).to(sq.device)).mean()


class UncertaintyWeightedLoss(torch.nn.Module):
    """可学习不确定性加权 (Kendall et al., CVPR 2018).

    每个输出维度学习一个 log_sigma 参数，loss 公式：
        L = Σ_i [ 0.5 * exp(-2*log_σ_i) * MSE_i + log_σ_i ]

    log_σ_i 作为可学参数与模型一起优化，自动平衡各组分权重。
    正则项 log_σ_i 防止权重塌缩（σ→∞ 使某组分loss为0）。
    """

    def __init__(self, num_tasks: int = 4, init_log_sigma: float = 0.0):
        super().__init__()
        # 初始化 log_sigma，默认为0 → σ=1 → 权重=0.5（各组分初始等权）
        self.log_sigmas = torch.nn.Parameter(
            torch.full((num_tasks,), init_log_sigma, dtype=torch.float32)
        )

    def forward(self, pred, target):
        # 逐组分计算 MSE：shape (batch, num_tasks) → 对 batch 求均值 → (num_tasks,)
        per_task_mse = ((pred - target) ** 2).mean(dim=0)
        # 加权 loss：precision * mse + 正则项
        precision = torch.exp(-2.0 * self.log_sigmas)
        weighted = 0.5 * precision * per_task_mse + self.log_sigmas
        return weighted.sum()

    def get_weights(self) -> torch.Tensor:
        """返回当前各组分的有效权重 (precision = 1/(2σ²))，用于监控。"""
        with torch.no_grad():
            return torch.exp(-2.0 * self.log_sigmas)

    def get_sigmas(self) -> torch.Tensor:
        """返回当前各组分的 σ 值。"""
        with torch.no_grad():
            return torch.exp(self.log_sigmas)


class WeightedL1Loss(torch.nn.Module):
    """按列加权的 MAE。"""

    def __init__(self, weights: torch.Tensor):
        super().__init__()
        if weights.ndim != 1:
            raise ValueError(f"label_weights must be 1D, got shape {tuple(weights.shape)}")
        normalized = weights / weights.mean()
        self.register_buffer("weights", normalized.float())

    def forward(self, pred, target):
        abs_err = (pred - target).abs()
        return (abs_err * self.weights.to(abs_err.dtype).to(abs_err.device)).mean()


class SumConstraintLoss(torch.nn.Module):
    def __init__(self, base_loss: torch.nn.Module, weight: float = 0.0, target_sum: float = 100.0, penalty: str = "mse"):
        super().__init__()
        self.base_loss = base_loss
        self.weight = float(weight)
        self.target_sum = float(target_sum)
        normalized_penalty = penalty.lower()
        if normalized_penalty == "mse":
            self.penalty = torch.nn.MSELoss()
        elif normalized_penalty == "mae":
            self.penalty = torch.nn.L1Loss()
        else:
            raise ValueError(f"Unknown sum penalty: {penalty}")

    def forward(self, pred, target):
        base_value = self.base_loss(pred, target)
        if self.weight <= 0.0:
            return base_value
        pred_sum = pred.sum(dim=1)
        target_sum = torch.full_like(pred_sum, self.target_sum)
        sum_value = self.penalty(pred_sum, target_sum)
        return base_value + self.weight * sum_value


def build_loss(
    name: str,
    sum_constraint: dict | None = None,
    label_weights: torch.Tensor | None = None,
    uncertainty_weighted: dict | None = None,
):
    """构建 loss 函数。

    参数优先级：uncertainty_weighted > label_weights > 默认等权。
    当使用 uncertainty_weighted 时，忽略 label_weights（因为权重是自动学习的）。
    """
    normalized = name.lower()

    # 优先使用可学习不确定性加权
    if uncertainty_weighted:
        num_tasks = int(uncertainty_weighted.get("num_tasks", 4))
        init_log_sigma = float(uncertainty_weighted.get("init_log_sigma", 0.0))
        base_loss = UncertaintyWeightedLoss(num_tasks=num_tasks, init_log_sigma=init_log_sigma)
    elif label_weights is not None:
        if normalized == "mse":
            base_loss = WeightedMSELoss(label_weights)
        elif normalized == "mae":
            base_loss = WeightedL1Loss(label_weights)
        else:
            raise ValueError(f"Unknown loss: {name}")
    else:
        if normalized == "mse":
            base_loss = torch.nn.MSELoss()
        elif normalized == "mae":
            base_loss = torch.nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss: {name}")

    if not sum_constraint:
        return base_loss

    return SumConstraintLoss(
        base_loss=base_loss,
        weight=float(sum_constraint.get("weight", 0.0)),
        target_sum=float(sum_constraint.get("target_sum", 100.0)),
        penalty=str(sum_constraint.get("penalty", "mse")),
    )
