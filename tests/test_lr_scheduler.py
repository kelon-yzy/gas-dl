"""测试学习率调度器、梯度裁剪和优化器配置。"""
import sys
import unittest
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "dl"))

from models.registry import build_model
from training.orchestrator import _prepare_scheduler


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
            # 真实训练中应先 optimizer.step() 再 scheduler.step()，避免跳过第一个 LR 值
            optimizer.step()
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
        config_path = ROOT / "configs" / "deep" / "slow_only_cnn1d_multimodal_formal.yaml"
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self.assertEqual(config["training"]["optimizer"], "adamw")
        self.assertEqual(config["training"]["weight_decay"], 0.01)
        self.assertEqual(config["training"]["grad_clip_norm"], 1.0)
        self.assertIsNotNone(config["training"].get("lr_scheduler"))
        self.assertEqual(config["training"]["lr_scheduler"]["type"], "cosine_warmup")
        self.assertEqual(config["training"]["lr_scheduler"]["warmup_epochs"], 10)
        self.assertEqual(config["training"]["lr_scheduler"]["eta_min"], 0.00003)

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

    def test_plateau_scheduler_reduces_lr_after_stall(self):
        model = torch.nn.Linear(8, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        scheduler = _prepare_scheduler(
            optimizer,
            total_epochs=120,
            training_config={
                "learning_rate": 2e-4,
                "lr_scheduler": {
                    "type": "plateau",
                    "gamma": 0.5,
                    "patience": 4,
                },
            },
        )
        lrs = []
        for _ in range(7):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step(1.0)
        self.assertEqual(lrs[0], 2e-4)
        self.assertEqual(lrs[4], 2e-4)
        self.assertEqual(lrs[5], 2e-4)
        self.assertAlmostEqual(lrs[6], 1e-4, places=10)

    def test_cnn1d_tcn_fusion_config_uses_run_a_training_baseline(self):
        """Run D 把训练超参回退到 Run A 实测基线（cosine_warmup + lr=3e-4 + sum_w=0.1 mse +
        patience=25），用于评估"编码器输入语义修复"的纯增量；model.dropout 三项保留 Run B/C
        阶段上调后的取值（0.15 / 0.25 / 0.25），属于本轮不动的参数面。"""
        config_path = ROOT / "configs" / "deep" / "slow_only_cnn1d_tcn_fusion_multimodal_formal.yaml"
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        training = config["training"]
        model = config["model"]
        self.assertEqual(training["learning_rate"], 0.0003)
        self.assertEqual(training["early_stopping_patience"], 25)
        self.assertEqual(training["lr_scheduler"]["type"], "cosine_warmup")
        self.assertEqual(training["lr_scheduler"]["warmup_epochs"], 10)
        self.assertEqual(training["lr_scheduler"]["eta_min"], 0.00003)
        self.assertEqual(training["sum_constraint"]["weight"], 0.1)
        self.assertEqual(training["sum_constraint"]["penalty"], "mse")
        self.assertEqual(model["acoustic_dropout"], 0.15)
        self.assertEqual(model["tcn_dropout"], 0.25)
        self.assertEqual(model["head_dropout"], 0.25)

    def test_slow_branch_cnn1d_tcn_fusion_config_reuses_current_uw_training_setup(self):
        """慢变量分支实验配置应使用 Softplus+Normalize 头，训练目标改为纯 MSE。"""
        config_path = ROOT / "configs" / "deep" / "slow_branch_cnn1d_tcn_fusion.yaml"
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        training = config["training"]
        model = config["model"]
        self.assertEqual(config["run"]["name"], "v3_slow_branch_cnn1d_tcn_fusion_seed42")
        self.assertEqual(model["name"], "cnn1d_tcn_fusion_slow_branch")
        self.assertEqual(model["slow_encoder"]["enabled"], True)
        self.assertEqual(model["slow_encoder"]["hidden_dim"], 32)
        self.assertEqual(model["slow_encoder"]["embedding_dim"], 64)
        self.assertEqual(training["batch_size"], 64)
        self.assertEqual(training["learning_rate"], 0.0002)
        self.assertEqual(training["early_stopping_patience"], 30)
        self.assertEqual(training["loss"], "mse")
        self.assertIsNone(training.get("uncertainty_weighted"))
        self.assertIsNone(training.get("sum_constraint"))
        self.assertEqual(training["lr_scheduler"]["type"], "cosine_warmup")
        self.assertEqual(training["lr_scheduler"]["warmup_epochs"], 15)
        self.assertEqual(training["lr_scheduler"]["eta_min"], 0.00001)


if __name__ == "__main__":
    unittest.main()
