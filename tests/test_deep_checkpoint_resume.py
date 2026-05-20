"""测试 checkpoint 保存、恢复、暂停功能"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "dl"))
sys.path.insert(0, str(ROOT / "src"))

from training.train import (
    CheckpointBuildContext,
    TrainRunOptions,
    _capture_rng_state,
    _restore_rng_state,
    _checkpoint_payload,
    _save_checkpoint,
    _load_checkpoint,
    _restore_early_stopping,
    _validate_checkpoint_compat,
    _validate_model_architecture_compat,
    train_config,
)
from training.early_stopping import EarlyStopping
from models.registry import build_model


# ── 合成数据集 ──

class SyntheticNDataset(Dataset):
    """形状 (N, T, C) 的合成回归数据集"""

    def __init__(self, n_samples: int = 64, n_timesteps: int = 50, n_channels: int = 8, n_outputs: int = 4):
        # 返回 NCT 格式 (N, C, T)，与 input_format="NCT" 一致
        self.x = torch.randn(n_samples, n_channels, n_timesteps, dtype=torch.float32)
        self.y = torch.randn(n_samples, n_outputs, dtype=torch.float32) * 0.1 + 0.25

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], {
            "sample_id": str(idx),
            "mixture_id": f"M{idx // 4}",
        }


def _build_synthetic_datasets(config: dict):
    """返回 train/val/test 三个合成数据集"""
    n_samples = 64
    n_channels = config.get("model", {}).get("in_channels", 8)
    n_outputs = len(config.get("_label_names", ["H2", "CH4", "CO2", "N2"]))
    n_timesteps = 50
    return {
        "train": SyntheticNDataset(n_samples, n_timesteps, n_channels, n_outputs),
        "val": SyntheticNDataset(n_samples // 2, n_timesteps, n_channels, n_outputs),
        "test": SyntheticNDataset(n_samples // 2, n_timesteps, n_channels, n_outputs),
    }, None


def _minimal_config(output_dir: str, epochs: int = 2, model_name: str = "cnn1d") -> dict:
    return {
        "data": {
            "dataset_type": "v2",
            "npz_path": "dummy",
            "input_format": "NCT",
            "time_window": "all",
            "channels": "all",
        },
        "model": {
            "name": model_name,
            "in_channels": 8,
            "out_dim": 4,
            "hidden_channels": [8, 16],
            "kernel_size": 3,
            "dropout": 0.0,
        },
        "training": {
            "epochs": epochs,
            "batch_size": 16,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "early_stopping_patience": 50,
            "device": "cpu",
            "amp": False,
            "num_workers": 0,
            "eval_num_workers": 0,
        },
        "run": {
            "name": "test_run",
            "seed": 42,
            "output_dir": output_dir,
        },
    }


# ── 测试类 ──


class CheckpointHelperTests(unittest.TestCase):
    """checkpoint helper 函数的单元测试"""

    def test_capture_and_restore_rng(self):
        torch.manual_seed(123)
        np.random.seed(123)
        # 先消耗 5 个值对齐状态偏移
        torch.randn(5)
        state = _capture_rng_state()
        expected = torch.randn(5).numpy().copy()
        # 消耗更多随机数使状态前进
        torch.randn(10)
        np.random.rand(10)
        _restore_rng_state(state)
        b = torch.randn(5).numpy().copy()
        np.testing.assert_allclose(expected, b)

    def test_save_and_load_checkpoint_atomic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "test.pt"
            payload = {"format_version": 1, "value": 42}
            _save_checkpoint(path, payload)
            self.assertTrue(path.exists())
            loaded = _load_checkpoint(path, torch.device("cpu"))
            self.assertEqual(loaded["value"], 42)

    def test_load_checkpoint_rejects_invalid_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "bad.pt"
            torch.save({"format_version": 999}, path)
            with self.assertRaises(ValueError):
                _load_checkpoint(path, torch.device("cpu"))

    def test_load_checkpoint_rejects_nondict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "nondict.pt"
            torch.save([1, 2, 3], path)
            with self.assertRaises(ValueError):
                _load_checkpoint(path, torch.device("cpu"))

    def test_load_checkpoint_rejects_unsafe_legacy_pickle_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "legacy_rng_tuple.pt"
            torch.save(
                {
                    "format_version": 1,
                    "rng_state": {"numpy": np.random.get_state()},
                },
                path,
            )
            with self.assertRaises(ValueError):
                _load_checkpoint(path, torch.device("cpu"))

    def test_restore_early_stopping(self):
        stopper = EarlyStopping(patience=10, mode="min")
        state = {"best": 0.05, "bad_epochs": 3, "patience": 20, "mode": "max"}
        _restore_early_stopping(stopper, state)
        self.assertEqual(stopper.best, 0.05)
        self.assertEqual(stopper.bad_epochs, 3)
        self.assertEqual(stopper.patience, 20)
        self.assertEqual(stopper.mode, "max")

    def test_validate_checkpoint_compat_model_name(self):
        ckpt = {"model_name": "cnn1d", "format_version": 1, "epoch": 0, "total_epochs": 2,
                "model_state_dict": {}, "optimizer_state_dict": {}, "amp_scaler_state_dict": {},
                "early_stopping": {}, "log_rows": [], "rng_state": {}}
        config = {"model": {"name": "cnn1d"}}
        _validate_checkpoint_compat(ckpt, config)  # 不应抛出异常

        config_bad = {"model": {"name": "lstm"}}
        with self.assertRaises(ValueError):
            _validate_checkpoint_compat(ckpt, config_bad)

    def test_validate_checkpoint_compat_rejects_old_cnn1d_tcn_fusion_for_slow_branch_variant(self):
        ckpt = {
            "model_name": "cnn1d_tcn_fusion",
            "format_version": 1,
            "epoch": 0,
            "total_epochs": 2,
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "amp_scaler_state_dict": {},
            "early_stopping": {},
            "log_rows": [],
            "rng_state": {},
        }
        config = {"model": {"name": "cnn1d_tcn_fusion_slow_branch"}}
        with self.assertRaises(ValueError):
            _validate_checkpoint_compat(ckpt, config)

    def test_validate_checkpoint_compat_missing_keys(self):
        ckpt = {"format_version": 1, "model_name": "cnn1d"}
        config = {"model": {"name": "cnn1d"}}
        with self.assertRaises(ValueError):
            _validate_checkpoint_compat(ckpt, config)

    def test_validate_checkpoint_compat_rejects_model_hyperparameter_mismatch(self):
        ckpt = {
            "format_version": 1,
            "model_name": "cnn1d",
            "epoch": 1,
            "total_epochs": 2,
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "amp_scaler_state_dict": {},
            "early_stopping": {},
            "log_rows": [],
            "rng_state": {},
            "label_names": ["H2", "CH4", "CO2", "N2"],
            "config": {
                "data": {"dataset_type": "v2", "time_window": "all"},
                "model": {
                    "name": "cnn1d",
                    "in_channels": 8,
                    "out_dim": 4,
                    "hidden_channels": [8, 16],
                    "kernel_size": 3,
                    "dropout": 0.0,
                },
            },
        }
        config = {
            "data": {"dataset_type": "v2", "time_window": "all"},
            "model": {
                "name": "cnn1d",
                "in_channels": 8,
                "out_dim": 4,
                "hidden_channels": [16, 32],
                "kernel_size": 3,
                "dropout": 0.0,
            },
        }

        with self.assertRaises(ValueError):
            _validate_checkpoint_compat(ckpt, config, ["H2", "CH4", "CO2", "N2"])

    def test_validate_model_architecture_compat_passes_on_matching_keys(self):
        """当 checkpoint state_dict 键与当前模型完全匹配时，不应报错。"""
        model = torch.nn.Linear(8, 4)
        ckpt = {
            "model_name": "linear",
            "model_state_dict": model.state_dict(),
        }
        _validate_model_architecture_compat(ckpt, set(model.state_dict().keys()))

    def test_validate_model_architecture_compat_rejects_missing_and_extra_keys(self):
        """架构变更后 checkpoint 缺少键或有额外键时必须报错，并给出明确差异信息。"""
        model = torch.nn.Linear(8, 4)
        model_keys = set(model.state_dict().keys())
        # 模拟架构升级：旧 checkpoint 缺少 projection.weight 等，多出 features.0.weight 等
        old_state = {"features.0.weight": torch.randn(16, 1, 15), "features.0.bias": torch.randn(16)}
        ckpt_missing = {"model_name": "cnn1d_multimodal", "model_state_dict": old_state}
        with self.assertRaises(ValueError) as cm:
            _validate_model_architecture_compat(ckpt_missing, model_keys)
        error_msg = str(cm.exception)
        self.assertIn("checkpoint 独有", error_msg)
        self.assertIn("当前模型独有", error_msg)
        self.assertIn("架构已变更", error_msg)

    def test_validate_model_architecture_compat_skips_when_no_state_dict(self):
        """checkpoint 缺少 model_state_dict 时跳过架构校验（不应报错）。"""
        ckpt = {"model_name": "cnn1d"}
        _validate_model_architecture_compat(ckpt, {"weight", "bias"})

    def test_checkpoint_payload_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = pathlib.Path(tmp)
            best_path = out_dir / "best_model.pt"
            model = torch.nn.Linear(8, 4)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scaler = torch.amp.GradScaler("cpu", enabled=False)
            stopper = EarlyStopping(patience=5, mode="min")
            config = {"model": {"name": "linear"}}

            payload = _checkpoint_payload(
                model, optimizer, scaler, stopper, epoch=3, total_epochs=10,
                log_rows=[{"epoch": 1, "train_loss": 0.5}],
                best_path=best_path, config_to_write=config, status="running",
            )
            self.assertEqual(payload["format_version"], 1)
            self.assertEqual(payload["status"], "running")
            self.assertEqual(payload["epoch"], 3)
            self.assertEqual(payload["total_epochs"], 10)
            self.assertIn("model_state_dict", payload)
            self.assertIn("optimizer_state_dict", payload)
            self.assertIn("amp_scaler_state_dict", payload)
            self.assertIn("early_stopping", payload)
            self.assertIn("log_rows", payload)
            self.assertIn("rng_state", payload)
            self.assertEqual(payload["model_name"], "linear")

    def test_checkpoint_payload_context_matches_legacy_kwargs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = pathlib.Path(tmp)
            best_path = out_dir / "best_model.pt"
            model = torch.nn.Linear(8, 4)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scaler = torch.amp.GradScaler("cpu", enabled=False)
            stopper = EarlyStopping(patience=5, mode="min")
            config = {"model": {"name": "linear"}}
            log_rows = [{"epoch": 1, "train_loss": 0.5}]

            legacy_payload = _checkpoint_payload(
                model,
                optimizer,
                scaler,
                stopper,
                epoch=3,
                total_epochs=10,
                log_rows=log_rows,
                best_path=best_path,
                config_to_write=config,
                status="running",
                label_names=["H2", "CH4", "CO2", "N2"],
                scheduler=None,
            )
            context_payload = _checkpoint_payload(
                context=CheckpointBuildContext(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    stopper=stopper,
                    epoch=3,
                    total_epochs=10,
                    log_rows=log_rows,
                    best_path=best_path,
                    config_to_write=config,
                    status="running",
                    label_names=["H2", "CH4", "CO2", "N2"],
                    scheduler=None,
                )
            )

            self.assertEqual(legacy_payload["format_version"], context_payload["format_version"])
            self.assertEqual(legacy_payload["status"], context_payload["status"])
            self.assertEqual(legacy_payload["epoch"], context_payload["epoch"])
            self.assertEqual(legacy_payload["total_epochs"], context_payload["total_epochs"])
            self.assertEqual(legacy_payload["config"], context_payload["config"])
            self.assertEqual(legacy_payload["model_name"], context_payload["model_name"])
            self.assertEqual(legacy_payload["label_names"], context_payload["label_names"])
            self.assertEqual(legacy_payload["log_rows"], context_payload["log_rows"])
            self.assertEqual(legacy_payload["best_model_path"], context_payload["best_model_path"])
            self.assertEqual(legacy_payload["early_stopping"], context_payload["early_stopping"])


class CheckpointIntegrationTests(unittest.TestCase):
    """训练流程集成测试 —— 用合成数据和轻量模型跑完整流程"""

    def _run_train(self, output_dir: str, epochs: int, **kwargs) -> dict:
        config = _minimal_config(output_dir, epochs=epochs)
        config["_cli_progress"] = None
        config["_label_names"] = ["H2", "CH4", "CO2", "N2"]

        with mock.patch("training.train.build_datasets", side_effect=_build_synthetic_datasets), \
             mock.patch("training.train._load_label_names", return_value=["H2", "CH4", "CO2", "N2"]), \
             mock.patch("training.train._ensure_scaler_path"):
            return train_config(config, **kwargs)

    def test_checkpoint_written_after_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run_train(tmp, epochs=1)
            last_ckpt = pathlib.Path(tmp) / "last_checkpoint.pt"
            self.assertTrue(last_ckpt.exists(), f"last_checkpoint.pt should exist, got: {list(pathlib.Path(tmp).iterdir())}")
            ckpt = torch.load(last_ckpt, map_location="cpu", weights_only=False)
            self.assertEqual(ckpt["epoch"], 1)
            self.assertEqual(ckpt["status"], "completed")
            self.assertIn("model_state_dict", ckpt)
            self.assertIn("optimizer_state_dict", ckpt)
            self.assertIn("amp_scaler_state_dict", ckpt)
            self.assertIn("early_stopping", ckpt)
            self.assertIn("log_rows", ckpt)
            # summary.json 包含 checkpoint 信息
            summary = json.loads((pathlib.Path(tmp) / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["training_status"], "completed")
            self.assertIn("last_checkpoint", summary)
            self.assertIn("best_checkpoint", summary)

    def test_train_log_records_validation_sum_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run_train(tmp, epochs=1)
            log = pd.read_csv(pathlib.Path(tmp) / "train_log.csv")
            self.assertIn("val_mean_pred_sum", log.columns)
            self.assertIn("val_mean_abs_sum_error", log.columns)
            self.assertIn("val_std_pred_sum", log.columns)

    def test_resume_continues_from_next_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 第一次：stop_after_epoch=1 暂停
            self._run_train(tmp, epochs=1, stop_after_epoch=1)
            paused = pathlib.Path(tmp) / "paused_checkpoint.pt"
            self.assertTrue(paused.exists())

            # 第二次：从 checkpoint 恢复，目标 2 个 epoch
            self._run_train(tmp, epochs=2, resume_path=str(paused))

            # 验证 train_log.csv 有 epoch 1 和 epoch 2
            log = pd.read_csv(pathlib.Path(tmp) / "train_log.csv")
            self.assertIn(1, log["epoch"].values)
            self.assertIn(2, log["epoch"].values)
            # summary 中的 epochs_trained
            summary = json.loads((pathlib.Path(tmp) / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["epochs_trained"], 2)
            self.assertEqual(summary["training_status"], "completed")

    def test_resume_preserves_existing_log_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 第一次：stop_after_epoch=1 暂停
            self._run_train(tmp, epochs=3, stop_after_epoch=1)
            paused = pathlib.Path(tmp) / "paused_checkpoint.pt"
            ckpt_before = torch.load(paused, map_location="cpu", weights_only=False)
            n_rows_before = len(ckpt_before["log_rows"])

            # 第二次恢复
            self._run_train(tmp, epochs=3, resume_path=str(paused))
            log = pd.read_csv(pathlib.Path(tmp) / "train_log.csv")
            # 不应该有重复 epoch
            self.assertEqual(log["epoch"].nunique(), len(log))
            # 总行数应大于第一次
            self.assertGreater(len(log), n_rows_before)
            # 所有 epoch 值唯一
            self.assertEqual(len(log), log["epoch"].nunique())

    def test_stop_after_epoch_writes_paused_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run_train(tmp, epochs=10, stop_after_epoch=1)
            paused = pathlib.Path(tmp) / "paused_checkpoint.pt"
            self.assertTrue(paused.exists())
            summary = json.loads((pathlib.Path(tmp) / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["training_status"], "paused")
            # 暂停不应生成测试集 predictions.csv
            self.assertFalse((pathlib.Path(tmp) / "predictions.csv").exists())

    def test_keyboard_interrupt_writes_paused_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _minimal_config(tmp, epochs=5)
            config["_cli_progress"] = None
            config["_label_names"] = ["H2", "CH4", "CO2", "N2"]

            def _interrupt_epoch(*args, **kwargs):
                raise KeyboardInterrupt()

            with mock.patch("training.train.build_datasets", side_effect=_build_synthetic_datasets), \
                 mock.patch("training.train._load_label_names", return_value=["H2", "CH4", "CO2", "N2"]), \
                 mock.patch("training.train._ensure_scaler_path"), \
                 mock.patch("training.train._train_one_epoch", side_effect=_interrupt_epoch):

                summary = train_config(config)
                self.assertEqual(summary["training_status"], "paused")
                self.assertTrue((pathlib.Path(tmp) / "paused_checkpoint.pt").exists())

    def test_fresh_interrupt_ignores_stale_last_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _minimal_config(tmp, epochs=5)
            config["_cli_progress"] = None
            config["_label_names"] = ["H2", "CH4", "CO2", "N2"]
            out_dir = pathlib.Path(tmp)

            stale_model = build_model(config["model"])
            stale_state = {}
            for key, value in stale_model.state_dict().items():
                if value.is_floating_point():
                    stale_state[key] = torch.full_like(value, 123.0)
                else:
                    stale_state[key] = torch.full_like(value, 123)
            stale_model.load_state_dict(stale_state)
            optimizer = torch.optim.Adam(stale_model.parameters(), lr=0.001)
            scaler = torch.amp.GradScaler("cpu", enabled=False)
            stopper = EarlyStopping(patience=5, mode="min")
            _save_checkpoint(
                out_dir / "last_checkpoint.pt",
                _checkpoint_payload(
                    stale_model,
                    optimizer,
                    scaler,
                    stopper,
                    epoch=99,
                    total_epochs=99,
                    log_rows=[{"epoch": 99}],
                    best_path=out_dir / "best_model.pt",
                    config_to_write={k: v for k, v in config.items() if not k.startswith("_")},
                    status="completed",
                    label_names=["H2", "CH4", "CO2", "N2"],
                ),
            )

            def _interrupt_epoch(*args, **kwargs):
                raise KeyboardInterrupt()

            with mock.patch("training.train.build_datasets", side_effect=_build_synthetic_datasets), \
                 mock.patch("training.train._load_label_names", return_value=["H2", "CH4", "CO2", "N2"]), \
                 mock.patch("training.train._ensure_scaler_path"), \
                 mock.patch("training.train._train_one_epoch", side_effect=_interrupt_epoch):
                summary = train_config(config)

            self.assertEqual(summary["training_status"], "paused")
            paused = torch.load(out_dir / "paused_checkpoint.pt", map_location="cpu", weights_only=True)
            first_tensor = next(iter(paused["model_state_dict"].values()))
            self.assertFalse(torch.all(first_tensor == 123))

    def test_best_checkpoint_saved_on_improvement(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run_train(tmp, epochs=3)
            best_ckpt = pathlib.Path(tmp) / "best_checkpoint.pt"
            best_model = pathlib.Path(tmp) / "best_model.pt"
            self.assertTrue(best_ckpt.exists(), "best_checkpoint.pt should exist")
            self.assertTrue(best_model.exists(), "best_model.pt should exist")
            # best_model.pt 应只含 state_dict
            bm = torch.load(best_model, map_location="cpu", weights_only=True)
            self.assertIsInstance(bm, dict)
            # best_checkpoint.pt 应含完整 payload
            bc = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            self.assertIn("format_version", bc)
            self.assertIn("model_state_dict", bc)

    def test_epoch_snapshots_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run_train(tmp, epochs=4, checkpoint_every=2)
            for ep in [2, 4]:
                f = pathlib.Path(tmp) / f"epoch_{ep:04d}.pt"
                self.assertTrue(f.exists(), f"epoch_{ep:04d}.pt should exist")
            # epoch 1 和 3 不应该生成快照
            self.assertFalse((pathlib.Path(tmp) / "epoch_0001.pt").exists())
            self.assertFalse((pathlib.Path(tmp) / "epoch_0003.pt").exists())

    def test_train_config_accepts_options_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _minimal_config(tmp, epochs=3)
            config["_cli_progress"] = None
            config["_label_names"] = ["H2", "CH4", "CO2", "N2"]

            with mock.patch("training.train.build_datasets", side_effect=_build_synthetic_datasets), \
                 mock.patch("training.train._load_label_names", return_value=["H2", "CH4", "CO2", "N2"]), \
                 mock.patch("training.train._ensure_scaler_path"):
                summary = train_config(
                    config,
                    options=TrainRunOptions(
                        epochs_override=1,
                        stop_after_epoch=1,
                    ),
                )

            self.assertEqual(summary["training_status"], "paused")
            self.assertEqual(summary["epochs_trained"], 1)


class PipelineArgTests(unittest.TestCase):
    """CLI 参数传递测试"""

    def test_pipeline_passes_resume_args(self):
        captured = {}
        config_path = ROOT / "configs" / "deep" / "slow_only_lstm_formal.yaml"

        def fake_train_config(config, epochs_override=None, resume_path=None,
                              checkpoint_every=0, restore_rng=True, stop_after_epoch=None):
            captured.update({
                "epochs_override": epochs_override,
                "resume_path": resume_path,
                "checkpoint_every": checkpoint_every,
                "restore_rng": restore_rng,
                "stop_after_epoch": stop_after_epoch,
            })
            return {"macro_RMSE": 0.0}

        import pipeline.train_deep as td

        argv = [
            "train_deep.py",
            "--config", str(config_path),
            "--epochs", "2",
            "--resume", "some/checkpoint.pt",
            "--checkpoint-every", "5",
            "--no-resume-rng",
            "--stop-after-epoch", "3",
            "--no-ui",
        ]

        with mock.patch.object(td, "build_cli_progress", return_value=None), \
             mock.patch.object(td, "train_config", side_effect=fake_train_config), \
             mock.patch.object(sys, "argv", argv):
            td.main()

        self.assertEqual(captured["epochs_override"], 2)
        self.assertEqual(captured["resume_path"], "some/checkpoint.pt")
        self.assertEqual(captured["checkpoint_every"], 5)
        self.assertFalse(captured["restore_rng"])
        self.assertEqual(captured["stop_after_epoch"], 3)


if __name__ == "__main__":
    unittest.main()
