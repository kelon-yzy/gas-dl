from __future__ import annotations


class EarlyStopping:
    def __init__(self, patience: int = 25, mode: str = "min"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        improved = False
        if self.best is None:
            improved = True
        elif self.mode == "min" and value < self.best:
            improved = True
        elif self.mode == "max" and value > self.best:
            improved = True
        if improved:
            self.best = value
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience

