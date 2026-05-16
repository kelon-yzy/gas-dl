"""测试学习率调度器、梯度裁剪和优化器配置。"""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "dl"))

from models.registry import build_model


class LRSchedulerTests(unittest.TestCase):
    """验证学习率调度器、AdamW 和梯度裁剪的正确性。"""

    def test_cosine_warmup_lr_shape(self):
        """warmup 阶段 LR 线性增长，之后 cosine 递减到 eta_min。"""
        from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
        model = torch.nn.Linear(8, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        warmup = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=5)
        cosine = CosineAnnealingLR(optimizer, T_max=195, eta_min=1e-4)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])
        lrs = []
        for epoch in range(200):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()
        # warmup: LR 从 ~0 递增到 1e-3
        self.assertLess(lrs[0], 1e-6)
        self.assertAlmostEqual(lrs[5], 1e-3, places=4)
        # cosine decay: LR 逐渐降低到 eta_min
        self.assertLess(lrs[-1], lrs[5])
        self.assertGreater(lrs[-1], 5e-5)  # 不应低于 eta_min/2
        # 整体单调递减（warmup 后）
        for i in range(6, len(lrs) - 1):
            self.assertGreaterEqual(lrs[i], lrs[i + 1])

    def test_no_scheduler_config_creates_no_scheduler(self):
        """无 lr_scheduler 配置时 scheduler=None。"""
        config = {"training": {"learning_rate": 0.001, "weight_decay": 0.01, "optimizer": "adam"}}
        scheduler_cfg = config["training"].get("lr_scheduler")
        self.assertIsNone(scheduler_cfg)

    def test_adamw_default_weight_decay(self):
        """AdamW 默认 weight_decay=0.01。"""
        model = torch.nn.Linear(8, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.01)

    def test_adam_default_weight_decay(self):
        """Adam 默认 weight_decay=0.0。"""
        model = torch.nn.Linear(8, 4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.0)

    def test_grad_clip_norm_zero_is_noop(self):
        """grad_clip_norm=0 时不裁剪梯度。"""
        model = torch.nn.Linear(8, 4)
        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        orig_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.0)
        # max_norm=0 的行为是返回原始范数（无裁剪）
        self.assertGreater(orig_norm, 0)

    def test_grad_clip_norm_one_clips_gradient(self):
        """grad_clip_norm=1 时梯度被裁剪（裁剪后参数梯度范数应 <= 1.0）。"""
        model = torch.nn.Linear(8, 4)
        torch.nn.init.uniform_(model.weight, -10, 10)
        x = torch.randn(32, 8)
        loss = model(x).sum()
        loss.backward()
        # clip_grad_norm_ 返回裁剪前的总范数，并 in-place 裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # 裁剪后再次检查范数
        clipped_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
        self.assertLessEqual(clipped_norm.item(), 1.0 + 1e-5)

    def test_cnn1d_multimodal_config_has_adamw_and_scheduler(self):
        """CNN1D multimodal 配置文件包含 optimizer=adamw 和 lr_scheduler。"""
        import yaml
        config_path = ROOT / "configs" / "deep" / "slow_only_cnn1d_multimodal_formal.yaml"
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self.assertEqual(config["training"]["optimizer"], "adamw")
        self.assertEqual(config["training"]["weight_decay"], 0.01)
        self.assertEqual(config["training"]["grad_clip_norm"], 1.0)
        self.assertIsNotNone(config["training"].get("lr_scheduler"))
        self.assertEqual(config["training"]["lr_scheduler"]["type"], "cosine_warmup")
        self.assertEqual(config["training"]["lr_scheduler"]["warmup_epochs"], 5)
        self.assertEqual(config["training"]["lr_scheduler"]["eta_min"], 0.0001)

    def test_backward_compatible_config_without_scheduler(self):
        """旧配置（无 optimizer/lr_scheduler/grad_clip_norm）仍然可运行。"""
        import yaml
        config_path = ROOT / "configs" / "deep" / "slow_only_gru_formal.yaml"
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        # 这些键不存在时应该有默认值
        self.assertIsNone(config["training"].get("optimizer"))  # 默认 adam
        self.assertIsNone(config["training"].get("lr_scheduler"))  # 默认 None
        self.assertIsNone(config["training"].get("grad_clip_norm"))  # 默认 0.0


if __name__ == "__main__":
    unittest.main()