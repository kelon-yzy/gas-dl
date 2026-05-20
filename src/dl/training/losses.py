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


def build_loss(name: str, sum_constraint: dict | None = None, label_weights: torch.Tensor | None = None):
    normalized = name.lower()
    if label_weights is not None:
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
