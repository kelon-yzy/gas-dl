from __future__ import annotations

import torch


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


def build_loss(name: str, sum_constraint: dict | None = None):
    normalized = name.lower()
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
