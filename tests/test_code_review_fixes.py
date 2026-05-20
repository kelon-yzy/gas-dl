import pathlib
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "dl"))
sys.path.insert(0, str(ROOT / "src" / "ml"))

from data.dataset_v2 import V2SequenceDataset
from data.dataset_waveform import WaveformSequenceDataset
from patent_model.data_loader import load_patent_dataset
from patent_model.dataset import PatentDataset
from patent_model.fault_labels import inject_faults
from patent_model.modeling import ModelConfig, TraditionalFusionModel
from patent_model.robustness import add_environment_noise, add_profile_environment_noise, select_pressure_slice
from training.orchestrator import _restore_clean_state_after_interrupt
from training.seed import set_seed
from training.train import TrainEpochRequest, _ensure_scaler_path, _forward_batch, _train_one_epoch, evaluate_loss, evaluate_with_predictions


def _dataset() -> PatentDataset:
    sample_ids = np.array(["S1", "S2", "S3"], dtype=object)
    metadata = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "mixture_id": ["M1", "M1", "M2"],
        }
    )
    return PatentDataset(
        sample_ids=sample_ids,
        acoustic=np.array([[1.0, 0.5], [2.0, 0.4], [3.0, 0.3]]),
        optical=np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]),
        thermal=np.array([[0.7], [0.8], [0.9]]),
        environment=np.array([[25.0, 0.10, 40.0], [25.5, 0.20, 45.0], [26.0, 0.30, 50.0]]),
        targets=np.array([[10.0, 80.0, 5.0, 5.0], [11.0, 79.0, 5.0, 5.0], [12.0, 78.0, 5.0, 5.0]]),
        component_names=("H2", "CH4", "CO2", "N2"),
        metadata=metadata,
        acoustic_columns=("TOF", "Amp"),
        optical_columns=("V_NDIR_CH4", "V_NDIR_CO2"),
        thermal_columns=("V_TCS",),
        environment_columns=("T_C", "P_MPa", "H_RH"),
        provenance={"source": "unit-test"},
        filter_report={"metadata_filter": {"kept": 3}},
    )


class CodeReviewFixTests(unittest.TestCase):
    def test_dataset_transformations_preserve_provenance_and_filter_report(self) -> None:
        dataset = _dataset()

        for transformed in (
            inject_faults(dataset, "acoustic_bias", severity="mild", seed=1),
            add_environment_noise(dataset, sigma_t=0.1, sigma_p=0.01, sigma_h=0.2, seed=2),
            select_pressure_slice(dataset, target_pressure_mpa=0.2, max_samples=2),
        ):
            self.assertEqual(transformed.provenance, dataset.provenance)
            self.assertEqual(transformed.filter_report, dataset.filter_report)

    def test_oof_degenerate_path_raises_for_single_group(self) -> None:
        """Original test updated: OOF with single group now raises ValueError."""

        config = ModelConfig(stacking_folds=5, n_perturbations=1, random_state=7)
        model = TraditionalFusionModel(config, component_names=("H2", "CH4", "CO2", "N2"))
        inputs = {
            "acoustic": np.arange(15, dtype=float).reshape(5, 3),
            "optical": np.arange(15, 30, dtype=float).reshape(5, 3),
            "thermal": np.arange(30, 45, dtype=float).reshape(5, 3),
        }
        target = np.linspace(0.0, 1.0, 20).reshape(5, 4)
        groups = np.array(["M1", "M1", "M1", "M1", "M1"], dtype=object)

        with self.assertRaises(ValueError):
            model._oof_meta_inputs(inputs, target, groups)

    def test_model_forward_uses_declared_input_format(self) -> None:
        class NCTModel(torch.nn.Module):
            input_format = "NCT"

            def __init__(self) -> None:
                super().__init__()
                self.seen_shape = None

            def forward(self, x):
                self.seen_shape = tuple(x.shape)
                return x.mean(dim=(1, 2), keepdim=False).unsqueeze(1)

        model = NCTModel()
        batch = {
            "ultrasonic": torch.zeros((2, 5, 8), dtype=torch.int16),
            "ultrasonic_scale": torch.ones((2, 5), dtype=torch.float32),
            "fiber_mic": torch.zeros((2, 5, 8), dtype=torch.int16),
            "fiber_mic_scale": torch.ones((2, 5), dtype=torch.float32),
            "slow": torch.zeros((2, 5, 3), dtype=torch.float32),
            "target": torch.zeros((2, 1), dtype=torch.float32),
            "meta": {"sample_id": ["S1", "S2"]},
        }
        _forward_batch(model, batch, torch.device("cpu"))

        self.assertEqual(model.seen_shape, (2, 3, 5))

    def test_training_config_gets_default_scaler_path_when_missing(self) -> None:
        data_config = {"dataset_type": "waveform_v3"}
        output_dir = pathlib.Path("outputs") / "unit_run"

        _ensure_scaler_path(data_config, output_dir)

        self.assertEqual(data_config["scaler_path"], str(output_dir / "scaler_slow_sequence.json"))

    def test_interrupt_restore_restores_loss_fn_state(self) -> None:
        tmp = tempfile.mkdtemp()
        try:
            checkpoint_path = pathlib.Path(tmp) / "last_checkpoint.pt"
            checkpoint_path.write_text("exists", encoding="utf-8")
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            loss_fn = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                loss_fn.weight.fill_(5.0)
            restored_loss = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                restored_loss.weight.fill_(2.0)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "amp_scaler_state_dict": scaler.state_dict(),
                "loss_fn_state_dict": restored_loss.state_dict(),
            }
            ctx = SimpleNamespace(
                progress=None,
                current_run_last_checkpoint_written=True,
                last_ckpt_path=checkpoint_path,
                initial_ckpt_path=checkpoint_path,
                dependencies=SimpleNamespace(load_checkpoint=lambda path, device: checkpoint),
                device=torch.device("cpu"),
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                loss_fn=loss_fn,
            )

            _restore_clean_state_after_interrupt(ctx)

            self.assertAlmostEqual(float(loss_fn.weight.item()), 2.0)

        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_grad_clip_covers_loss_fn_optimizer_parameters(self) -> None:
        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([[1.0]]))

            def forward(self, x):
                return x @ self.weight

        class LossWithParameter(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.loss_weight = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, pred, target):
                return pred.sum() * 1000.0 + self.loss_weight * 1000.0

        model = TinyModel()
        loss_fn = LossWithParameter()
        optimizer = torch.optim.SGD(
            [
                {"params": model.parameters()},
                {"params": loss_fn.parameters()},
            ],
            lr=0.0,
        )
        request = TrainEpochRequest(
            model=model,
            loader=[(torch.ones((1, 1)), torch.zeros((1, 1)), {"sample_id": ["S1"]})],
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=torch.device("cpu"),
            dataset=SimpleNamespace(input_format="NTC"),
            env_aug_sigma=0.0,
            amp_enabled=False,
            scaler=None,
            grad_clip_norm=1.0,
        )

        _train_one_epoch(request=request)

        self.assertLessEqual(float(loss_fn.loss_weight.grad.abs().item()), 1.0)

    def test_v2_dataset_uses_preloaded_data_without_reloading_npz(self) -> None:
        data = {
            "X": np.arange(2 * 4 * 12, dtype=np.float32).reshape(2, 4, 12),
            "y": np.arange(2 * 4, dtype=np.float32).reshape(2, 4),
            "sequence_ids": np.array(["Q1", "Q2"], dtype=object),
            "channel_names": np.array(
                [
                    "V_NDIR_CH4",
                    "V_NDIR_CO2",
                    "V_TCS",
                    "T_C",
                    "P_MPa",
                    "H_RH",
                    "L_m",
                    "piston_position_m",
                    "TOF",
                    "Amp",
                    "f_peak",
                    "A_fft_max",
                ],
                dtype=object,
            ),
            "label_names": np.array(["x_H2", "x_CH4", "x_CO2", "x_N2"], dtype=object),
        }
        dataset = V2SequenceDataset("missing.npz", indices=[0], index_path=None, preloaded_data=data)

        with patch("data.dataset_v2.load_v2_npz", side_effect=AssertionError("unexpected reload")):
            sample = dataset[0]

        self.assertEqual(tuple(sample[0].shape), (4, 12))
        self.assertEqual(sample[2]["sample_id"], "Q1")

    def test_waveform_dataset_uses_preloaded_data_without_reloading_package(self) -> None:
        data = {
            "ultrasonic": np.zeros((1, 2, 8), dtype=np.int16),
            "ultrasonic_scale": np.ones((1, 2), dtype=np.float32),
            "fiber_mic": np.zeros((1, 2, 10), dtype=np.int16),
            "fiber_mic_scale": np.ones((1, 2), dtype=np.float32),
            "slow": np.zeros((1, 2, 8), dtype=np.float32),
            "y": np.zeros((1, 4), dtype=np.float32),
            "sequence_ids": np.array(["Q1"], dtype=object),
            "slow_channel_names": np.array(
                ["V_NDIR_CH4", "V_NDIR_CO2", "V_TCS", "T_C", "P_MPa", "H_RH", "L_m", "piston_position_m"],
                dtype=object,
            ),
            "label_names": np.array(["x_H2", "x_CH4", "x_CO2", "x_N2"], dtype=object),
        }
        dataset = WaveformSequenceDataset("missing", indices=[0], index_path=None, preloaded_data=data)

        with patch("data.dataset_waveform.load_waveform_package", side_effect=AssertionError("unexpected reload")):
            sample = dataset[0]

        self.assertEqual(tuple(sample["slow"].shape), (2, 8))
        self.assertEqual(sample["meta"]["sample_id"], "Q1")

    def test_evaluation_losses_use_sample_weighted_mean(self) -> None:
        class IdentityModel(torch.nn.Module):
            def forward(self, x):
                return x

        loader = [
            (
                torch.ones((3, 1), dtype=torch.float32),
                torch.zeros((3, 1), dtype=torch.float32),
                {"sample_id": ["S1", "S2", "S3"]},
            ),
            (
                torch.full((1, 1), 3.0, dtype=torch.float32),
                torch.zeros((1, 1), dtype=torch.float32),
                {"sample_id": ["S4"]},
            ),
        ]
        loss_fn = torch.nn.MSELoss()

        eval_loss = evaluate_loss(IdentityModel(), loader, loss_fn, torch.device("cpu"))
        pred_loss, bundle = evaluate_with_predictions(IdentityModel(), loader, loss_fn, torch.device("cpu"))

        self.assertAlmostEqual(eval_loss, 3.0)
        self.assertAlmostEqual(pred_loss, 3.0)
        self.assertEqual(bundle.y_true.shape, (4, 1))
        self.assertEqual(bundle.y_pred.shape, (4, 1))

    def test_train_one_epoch_loss_uses_sample_weighted_mean(self) -> None:
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        loader = [
            (
                torch.ones((3, 1), dtype=torch.float32),
                torch.zeros((3, 1), dtype=torch.float32),
                {"sample_id": ["S1", "S2", "S3"]},
            ),
            (
                torch.full((1, 1), 3.0, dtype=torch.float32),
                torch.zeros((1, 1), dtype=torch.float32),
                {"sample_id": ["S4"]},
            ),
        ]

        train_loss = _train_one_epoch(
            model=model,
            loader=loader,
            loss_fn=torch.nn.MSELoss(),
            optimizer=optimizer,
            device=torch.device("cpu"),
            dataset=None,
            env_aug_sigma=0.0,
            amp_enabled=False,
            scaler=None,
            grad_clip_norm=0.0,
        )

        self.assertAlmostEqual(train_loss, 3.0)

    def test_train_one_epoch_request_adapter_matches_legacy_kwargs(self) -> None:
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        loader = [
            (
                torch.ones((2, 1), dtype=torch.float32),
                torch.zeros((2, 1), dtype=torch.float32),
                {"sample_id": ["S1", "S2"]},
            ),
        ]
        request = TrainEpochRequest(
            model=model,
            loader=loader,
            loss_fn=torch.nn.MSELoss(),
            optimizer=optimizer,
            device=torch.device("cpu"),
            dataset=None,
            env_aug_sigma=0.0,
            amp_enabled=False,
            scaler=None,
            grad_clip_norm=0.0,
        )

        request_loss = _train_one_epoch(request=request)
        legacy_loss = _train_one_epoch(
            model=model,
            loader=loader,
            loss_fn=torch.nn.MSELoss(),
            optimizer=optimizer,
            device=torch.device("cpu"),
            dataset=None,
            env_aug_sigma=0.0,
            amp_enabled=False,
            scaler=None,
            grad_clip_norm=0.0,
        )

        self.assertAlmostEqual(request_loss, legacy_loss)

    def test_four_component_loader_reports_missing_condition_n2_column(self) -> None:
        tmp = tempfile.mkdtemp()
        try:
            base = pathlib.Path(tmp)
            (base / "training").mkdir()
            (base / "labels").mkdir()
            (base / "features").mkdir()
            pd.DataFrame(
                {
                    "sample_id": ["S1"],
                    "TOF": [1.0],
                    "Amp": [1.0],
                    "f_peak": [1.0],
                    "A_fft_max": [1.0],
                    "L_m": [1.0],
                    "T_C": [25.0],
                    "P_MPa": [0.1],
                    "H_RH": [40.0],
                }
            ).to_csv(base / "training" / "train_acoustic.csv", index=False)
            pd.DataFrame({"sample_id": ["S1"], "V_NDIR_CH4": [1.0], "V_NDIR_CO2": [1.0], "delta_I_CH4": [0.1], "delta_I_CO2": [0.1]}).to_csv(
                base / "training" / "train_optical.csv",
                index=False,
            )
            pd.DataFrame({"sample_id": ["S1"], "V_TCS": [1.0]}).to_csv(base / "training" / "train_thermal.csv", index=False)
            pd.DataFrame({"sample_id": ["S1"], "x_H2": [10.0], "x_CH4": [80.0], "x_CO2": [5.0]}).to_csv(
                base / "labels" / "labels.csv",
                index=False,
            )
            pd.DataFrame(
                {
                    "sample_id": ["S1"],
                    "mixture_id": ["M1"],
                    "stage_id": ["detection"],
                    "repeat_id": [1],
                    "status": ["synthetic_measurement"],
                    "pressure_stage": ["low"],
                    "distance_stage": ["short"],
                    "piston_position_m": [0.0],
                    "T_C": [25.0],
                    "P_MPa": [0.1],
                    "H_RH": [40.0],
                }
            ).to_csv(base / "condition_grid_v1.csv", index=False)
            pd.DataFrame(
                {
                    "sample_id": ["S1"],
                    "sound_speed": [340.0],
                    "attenuation_alpha": [0.1],
                    "ndir_ch4_saturated": [0],
                    "ndir_co2_saturated": [0],
                    "optical_baseline_drift_ch4": [0.0],
                    "optical_baseline_drift_co2": [0.0],
                    "thermal_baseline_drift": [0.0],
                    "lambda_mix_calibrated": [0.1],
                    "calibration_status": ["ok"],
                }
            ).to_csv(base / "features" / "feature_table.csv", index=False)

            with self.assertRaisesRegex(ValueError, "condition_grid_v1.csv.*x_N2"):
                load_patent_dataset(base, profile="raw_no_env_four")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_seed_sets_cudnn_determinism_flags(self) -> None:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        set_seed(123)

        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cudnn.benchmark)

    def test_v3_env_profile_noise_propagates_to_embedded_columns(self) -> None:
        """T-P2-T1: verify that environment noise reaches embedded T_C/P_MPa/H_RH in acoustic matrix."""

        from patent_model.feature_profiles import DUAL_WAVEFORM_ACOUSTIC_ENV_COLUMNS, DUAL_WAVEFORM_OPTICAL_ENV_COLUMNS, DUAL_WAVEFORM_THERMAL_ENV_COLUMNS

        n = 5
        n_acoustic = len(DUAL_WAVEFORM_ACOUSTIC_ENV_COLUMNS)
        n_optical = len(DUAL_WAVEFORM_OPTICAL_ENV_COLUMNS)
        n_thermal = len(DUAL_WAVEFORM_THERMAL_ENV_COLUMNS)
        sample_ids = np.array([f"S{i}" for i in range(n)], dtype=object)
        metadata = pd.DataFrame({"sample_id": sample_ids, "mixture_id": [f"M{i}" for i in range(n)]})
        dataset = PatentDataset(
            sample_ids=sample_ids,
            acoustic=np.random.default_rng(0).standard_normal((n, n_acoustic)),
            optical=np.random.default_rng(1).standard_normal((n, n_optical)),
            thermal=np.random.default_rng(2).standard_normal((n, n_thermal)),
            environment=np.array([[25.0, 0.10, 40.0]] * n),
            targets=np.random.default_rng(3).standard_normal((n, 4)),
            component_names=("H2", "CH4", "CO2", "N2"),
            metadata=metadata,
            acoustic_columns=DUAL_WAVEFORM_ACOUSTIC_ENV_COLUMNS,
            optical_columns=DUAL_WAVEFORM_OPTICAL_ENV_COLUMNS,
            thermal_columns=DUAL_WAVEFORM_THERMAL_ENV_COLUMNS,
            environment_columns=("T_C", "P_MPa", "H_RH"),
            provenance={"source": "unit-test"},
            filter_report={},
        )

        noisy = add_profile_environment_noise(
            dataset,
            profile="v3_waveform_dual_channel_env_four",
            sigma_t=10.0,
            sigma_p=0.05,
            sigma_h=5.0,
            seed=42,
        )

        t_c_idx = DUAL_WAVEFORM_ACOUSTIC_ENV_COLUMNS.index("T_C")
        self.assertFalse(
            np.allclose(dataset.acoustic[:, t_c_idx], noisy.acoustic[:, t_c_idx]),
            "Embedded T_C column in acoustic matrix must change after environment noise injection",
        )
        p_mpa_opt_idx = DUAL_WAVEFORM_OPTICAL_ENV_COLUMNS.index("P_MPa")
        self.assertFalse(
            np.allclose(dataset.optical[:, p_mpa_opt_idx], noisy.optical[:, p_mpa_opt_idx]),
            "Embedded P_MPa column in optical matrix must change after environment noise injection",
        )

    def test_oof_degenerate_path_raises_value_error(self) -> None:
        """T-P1-2: OOF with < 2 groups must raise instead of silently leaking."""

        config = ModelConfig(stacking_folds=5, n_perturbations=1, random_state=7)
        model = TraditionalFusionModel(config, component_names=("H2", "CH4", "CO2", "N2"))
        inputs = {
            "acoustic": np.arange(15, dtype=float).reshape(5, 3),
            "optical": np.arange(15, 30, dtype=float).reshape(5, 3),
            "thermal": np.arange(30, 45, dtype=float).reshape(5, 3),
        }
        target = np.linspace(0.0, 1.0, 20).reshape(5, 4)
        groups = np.array(["M1", "M1", "M1", "M1", "M1"], dtype=object)

        with self.assertRaises(ValueError):
            model._oof_meta_inputs(inputs, target, groups)

    def test_optical_fault_multiplier_stays_positive(self) -> None:
        """T-P1-3: optical multiplier must be clipped to positive range."""

        dataset = _dataset()
        for severity in ("mild", "medium", "severe"):
            faulted = inject_faults(dataset, "optical_fail", severity=severity, seed=99)
            original_sign = np.sign(dataset.optical)
            faulted_sign = np.sign(faulted.optical)
            non_zero = original_sign != 0
            self.assertTrue(
                np.all(original_sign[non_zero] == faulted_sign[non_zero]),
                f"Optical values must not flip sign at severity={severity}",
            )


class UncertaintyWeightedLossTests(unittest.TestCase):
    """UncertaintyWeightedLoss 的 sigma_clamp 和 per-task init 功能回归测试。"""

    def _make_loss(self, **kwargs):
        from training.losses import UncertaintyWeightedLoss
        return UncertaintyWeightedLoss(**kwargs)

    def test_sigma_clamp_limits_log_sigma_range(self) -> None:
        """sigma_clamp 应阻止 log_sigma 超出指定范围。"""
        loss = self._make_loss(num_tasks=4, init_log_sigma=0.0, sigma_clamp=(-1.0, 1.5))
        # 手动把 log_sigma 推到极端值
        with torch.no_grad():
            loss.log_sigmas.copy_(torch.tensor([-5.0, 10.0, 0.5, -0.5]))
        clamped = loss._clamped_log_sigmas()
        self.assertAlmostEqual(clamped[0].item(), -1.0, places=5)
        self.assertAlmostEqual(clamped[1].item(), 1.5, places=5)
        self.assertAlmostEqual(clamped[2].item(), 0.5, places=5)
        self.assertAlmostEqual(clamped[3].item(), -0.5, places=5)

    def test_no_clamp_backward_compatible(self) -> None:
        """sigma_clamp=None 时行为与原版一致。"""
        loss = self._make_loss(num_tasks=2, init_log_sigma=0.0, sigma_clamp=None)
        with torch.no_grad():
            loss.log_sigmas.copy_(torch.tensor([5.0, -3.0]))
        clamped = loss._clamped_log_sigmas()
        self.assertAlmostEqual(clamped[0].item(), 5.0, places=5)
        self.assertAlmostEqual(clamped[1].item(), -3.0, places=5)

    def test_per_task_init_log_sigma(self) -> None:
        """per-task 列表初始化应正确设置各组分的初始 log_sigma。"""
        loss = self._make_loss(num_tasks=4, init_log_sigma=[0.1, 0.2, 0.3, 0.4])
        for i, expected in enumerate([0.1, 0.2, 0.3, 0.4]):
            self.assertAlmostEqual(loss.log_sigmas[i].item(), expected, places=5)

    def test_per_task_init_wrong_length_raises(self) -> None:
        """per-task 列表长度不匹配 num_tasks 时应报错。"""
        with self.assertRaises(ValueError):
            self._make_loss(num_tasks=4, init_log_sigma=[0.1, 0.2])

    def test_clamped_forward_produces_finite_loss(self) -> None:
        """带截断的前向传播应产生有限 loss，且梯度可正常回传。"""
        loss = self._make_loss(num_tasks=4, init_log_sigma=0.0, sigma_clamp=(-1.0, 1.5))
        pred = torch.randn(8, 4, requires_grad=True)
        target = torch.randn(8, 4)
        out = loss(pred, target)
        self.assertTrue(torch.isfinite(out).item())
        out.backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.all(torch.isfinite(loss.log_sigmas.grad)).item())

    def test_build_loss_with_sigma_clamp(self) -> None:
        """build_loss 应正确传递 sigma_clamp 到 UncertaintyWeightedLoss。"""
        from training.losses import build_loss, UncertaintyWeightedLoss
        loss = build_loss("mse", uncertainty_weighted={
            "num_tasks": 4,
            "init_log_sigma": [0.1, 0.2, 0.3, 0.4],
            "sigma_clamp": [-1.0, 1.5],
        })
        self.assertIsInstance(loss, UncertaintyWeightedLoss)
        self.assertEqual(loss.sigma_clamp, (-1.0, 1.5))
        self.assertAlmostEqual(loss.log_sigmas[0].item(), 0.1, places=5)

    def test_get_weights_respects_clamp(self) -> None:
        """get_weights 和 get_sigmas 应使用截断后的值。"""
        loss = self._make_loss(num_tasks=2, init_log_sigma=0.0, sigma_clamp=(-1.0, 1.0))
        with torch.no_grad():
            loss.log_sigmas.copy_(torch.tensor([5.0, -5.0]))
        weights = loss.get_weights()
        sigmas = loss.get_sigmas()
        # 被截断到 1.0 和 -1.0
        expected_w_max = torch.exp(torch.tensor(-2.0 * (-1.0))).item()
        expected_w_min = torch.exp(torch.tensor(-2.0 * 1.0)).item()
        self.assertAlmostEqual(weights[1].item(), expected_w_max, places=4)
        self.assertAlmostEqual(weights[0].item(), expected_w_min, places=4)
        self.assertAlmostEqual(sigmas[0].item(), torch.exp(torch.tensor(1.0)).item(), places=4)
        self.assertAlmostEqual(sigmas[1].item(), torch.exp(torch.tensor(-1.0)).item(), places=4)


if __name__ == "__main__":
    unittest.main()
