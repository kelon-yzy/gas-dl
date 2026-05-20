"""三模态四输出传统模型与动态融合逻辑。"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from patent_model.config import BRANCH_NAMES, COMPONENT_NAMES
from patent_model.dataset import PatentDataset
from patent_model.logging_utils import get_logger


logger = get_logger(__name__)


def _resolve_worker_count(n_jobs: int, task_count: int) -> int:
    """把 n_jobs（含 -1）翻译成实际线程数，并按任务数封顶。"""

    if task_count <= 1:
        return 1
    if n_jobs is None or n_jobs == 0:
        return 1
    if n_jobs < 0:
        cpu = os.cpu_count() or 1
        return max(1, min(task_count, cpu))
    return max(1, min(task_count, n_jobs))


@dataclass(frozen=True)
class ModelConfig:
    """模型超参数集合。"""

    C: float = 20.0
    epsilon: float = 0.003
    gamma: str = "scale"
    ridge_alpha: float = 0.8
    stacking_folds: int = 3
    n_perturbations: int = 10
    perturbation_scale: float = 0.04
    uncertainty_floor: float = 1e-6
    include_environment: bool = True
    random_state: int = 42
    branch_model_type: str = "svr"
    meta_model_type: str = "ridge"
    pls_n_components: int = 10
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 5
    xgb_learning_rate: float = 0.05
    xgb_device: str = "cpu"
    xgb_n_jobs: int = 1
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_reg_alpha: float = 0.1
    xgb_reg_lambda: float = 1.0
    n_jobs: int = -1


@dataclass
class ModalBranchArtifacts:
    """三个模态四输出分支阶段的可复用产物。"""

    modal_models: dict[str, "SingleModalityMultiOutputModel"]
    modal_feature_scales: dict[str, np.ndarray]
    modal_effective_pls_components: dict[str, int | None]
    oof_base: np.ndarray
    oof_weights: np.ndarray


@dataclass
class TraditionalPredictionCache:
    """三模态四输出预测缓存。"""

    base_predictions: np.ndarray
    dynamic_weights: np.ndarray


@dataclass
class TraditionalFusionPrediction:
    """三模态四输出 + 融合输出的整体预测结果。"""

    raw: np.ndarray
    by_model: dict[str, np.ndarray]
    dynamic_weights: np.ndarray


def _safe_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """在小样本或常数标签子集上安全计算 R2。"""

    if y_true.size < 2:
        return float("nan")
    if np.allclose(y_true, y_true[0], rtol=0.0, atol=1e-12):
        return float("nan")
    return float(r2_score(y_true, y_pred))


def _resolve_pls_n_components(config: ModelConfig, features: np.ndarray | None = None) -> int:
    """按当前输入规模约束 PLS 组件数。"""

    if config.pls_n_components <= 0:
        raise ValueError("pls_n_components must be positive.")
    if features is None:
        return config.pls_n_components
    return min(config.pls_n_components, features.shape[0], features.shape[1])


def _make_modal_pipeline(config: ModelConfig, features: np.ndarray | None = None) -> Pipeline:
    """为四输出模态模型构造回归流水线。"""

    if config.branch_model_type == "svr":
        estimator = MultiOutputRegressor(SVR(kernel="rbf", C=config.C, epsilon=config.epsilon, gamma=config.gamma))
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svr", estimator),
            ]
        )
    if config.branch_model_type == "pls":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pls", PLSRegression(n_components=_resolve_pls_n_components(config, features))),
            ]
        )
    if config.branch_model_type == "xgboost":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "xgb",
                    XGBRegressor(
                        n_estimators=config.xgb_n_estimators,
                        max_depth=config.xgb_max_depth,
                        learning_rate=config.xgb_learning_rate,
                        tree_method="hist",
                        device=config.xgb_device,
                        n_jobs=config.xgb_n_jobs,
                        subsample=config.xgb_subsample,
                        colsample_bytree=config.xgb_colsample_bytree,
                        reg_alpha=config.xgb_reg_alpha,
                        reg_lambda=config.xgb_reg_lambda,
                        random_state=config.random_state,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown branch_model_type: {config.branch_model_type}")


def _make_meta_model(config: ModelConfig, features: np.ndarray | None = None) -> BaseEstimator:
    """按配置构造四输出融合器。"""

    if config.meta_model_type == "ridge":
        return Ridge(alpha=config.ridge_alpha)
    if config.meta_model_type == "pls":
        return PLSRegression(n_components=_resolve_pls_n_components(config, features))
    if config.meta_model_type == "xgboost":
        return XGBRegressor(
            n_estimators=config.xgb_n_estimators,
            max_depth=config.xgb_max_depth,
            learning_rate=config.xgb_learning_rate,
            tree_method="hist",
            device=config.xgb_device,
            n_jobs=config.xgb_n_jobs,
            subsample=config.xgb_subsample,
            colsample_bytree=config.xgb_colsample_bytree,
            reg_alpha=config.xgb_reg_alpha,
            reg_lambda=config.xgb_reg_lambda,
            random_state=config.random_state,
        )
    raise ValueError(f"Unknown meta_model_type: {config.meta_model_type}")


def _branch_inputs(dataset: PatentDataset, include_environment: bool) -> dict[str, np.ndarray]:
    """为三条模态分支准备输入矩阵，可选拼接环境变量。"""

    if include_environment:
        return {
            "acoustic": np.hstack([dataset.acoustic, dataset.environment]),
            "optical": np.hstack([dataset.optical, dataset.environment]),
            "thermal": np.hstack([dataset.thermal, dataset.environment]),
        }
    return {
        "acoustic": dataset.acoustic,
        "optical": dataset.optical,
        "thermal": dataset.thermal,
    }


def _feature_scales(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """为每条分支估计特征尺度，供扰动注入使用。"""

    scales: dict[str, np.ndarray] = {}
    for name, values in inputs.items():
        scale = np.std(values, axis=0, ddof=0)
        scales[name] = np.where(scale < 1e-9, 1.0, scale)
    return scales


def _modal_meta_features(base_predictions: np.ndarray, dynamic_weights: np.ndarray) -> np.ndarray:
    """把三模态四输出基础预测和权重展平成融合器输入。"""

    sample_count = base_predictions.shape[0]
    return np.hstack(
        [
            base_predictions.reshape(sample_count, -1),
            (base_predictions * dynamic_weights).reshape(sample_count, -1),
            dynamic_weights.reshape(sample_count, -1),
        ]
    )


def _compute_modal_dynamic_weights(
    branch_inputs: dict[str, np.ndarray],
    base_predictions: np.ndarray,
    modal_models: dict[str, "SingleModalityMultiOutputModel"],
    feature_scales: dict[str, np.ndarray],
    config: ModelConfig,
    seed_offset: int = 0,
) -> np.ndarray:
    """用 MC 扰动估计三模态四输出预测的稳定性权重。"""

    rng = np.random.default_rng(config.random_state + seed_offset)
    n_samples, n_components, _ = base_predictions.shape
    drift_mse = np.zeros((n_samples, n_components, len(BRANCH_NAMES)), dtype=float)
    for branch_idx, branch_name in enumerate(BRANCH_NAMES):
        features = branch_inputs[branch_name]
        scale = feature_scales[branch_name] * config.perturbation_scale
        n_features = features.shape[1]
        noise = rng.standard_normal((config.n_perturbations, n_samples, n_features)) * scale
        perturbed = features[None, :, :] + noise
        flat = perturbed.reshape(config.n_perturbations * n_samples, n_features)
        flat_pred = modal_models[branch_name].predict(flat)
        perturb_array = flat_pred.reshape(config.n_perturbations, n_samples, n_components).transpose(1, 2, 0)
        baseline = base_predictions[:, :, branch_idx][:, :, None]
        drift_mse[:, :, branch_idx] = np.mean((perturb_array - baseline) ** 2, axis=2)
    confidence = 1.0 / (drift_mse + config.uncertainty_floor)
    return confidence / confidence.sum(axis=2, keepdims=True)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """统一计算回归任务的核心误差指标。"""

    error = y_pred - y_true
    relative_error = np.abs(error) / np.maximum(np.abs(y_true), 1e-6)
    return {
        "RMSE_pp": float(np.sqrt(np.mean(error**2))),
        "MAE_pp": float(np.mean(np.abs(error))),
        "MRE_pct": float(np.mean(relative_error) * 100.0),
        "R2": _safe_r2_score(y_true, y_pred),
        "MaxRE_pct": float(np.max(relative_error) * 100.0),
    }


def _active_modal_model_names() -> tuple[str, ...]:
    return ("acoustic", "optical", "thermal", "fused")


class SingleModalityMultiOutputModel:
    """单个模态的四输出基学习器。"""

    def __init__(self, config: ModelConfig, modality: str) -> None:
        self.config = config
        self.modality = modality
        self.model: Pipeline | None = None
        self.effective_pls_components: int | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "SingleModalityMultiOutputModel":
        if self.config.branch_model_type == "pls":
            self.effective_pls_components = _resolve_pls_n_components(self.config, features)
        else:
            self.effective_pls_components = None
        self.model = _make_modal_pipeline(self.config, features)
        self.model.fit(features, targets)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError(f"{self.modality} model has not been fitted yet.")
        prediction = np.asarray(self.model.predict(features), dtype=float)
        if prediction.ndim == 1:
            prediction = prediction[:, None]
        return prediction


class TraditionalFusionModel:
    """三模态四输出 + 动态权重 + 二次融合模型。"""

    def __init__(self, config: ModelConfig | None = None, component_names: tuple[str, ...] = COMPONENT_NAMES) -> None:
        self.config = config or ModelConfig()
        self.component_names = component_names
        self.modal_models = {name: SingleModalityMultiOutputModel(self.config, name) for name in BRANCH_NAMES}
        self.meta_model = _make_meta_model(self.config)
        self.modal_feature_scales: dict[str, np.ndarray] = {}
        self.modal_effective_pls_components: dict[str, int | None] = {name: None for name in BRANCH_NAMES}
        self.meta_effective_pls_components: int | None = None

    def fit_branch_stage(self, dataset: PatentDataset) -> ModalBranchArtifacts:
        if dataset.targets.shape[1] != len(self.component_names):
            raise ValueError("Dataset target width does not match component_names.")
        inputs = _branch_inputs(dataset, self.config.include_environment)
        groups = dataset.metadata["mixture_id"].to_numpy(dtype=object)
        oof_base, oof_weights = self._oof_meta_inputs(inputs, dataset.targets, groups)

        def _fit_modal(branch_name: str) -> tuple[str, SingleModalityMultiOutputModel]:
            model = SingleModalityMultiOutputModel(self.config, branch_name).fit(inputs[branch_name], dataset.targets)
            return branch_name, model

        n_workers = _resolve_worker_count(self.config.n_jobs, len(BRANCH_NAMES))
        if n_workers > 1:
            fitted_results = Parallel(n_jobs=n_workers, prefer="threads")(delayed(_fit_modal)(branch_name) for branch_name in BRANCH_NAMES)
        else:
            fitted_results = [_fit_modal(branch_name) for branch_name in BRANCH_NAMES]

        modal_models: dict[str, SingleModalityMultiOutputModel] = {}
        modal_effective_pls_components: dict[str, int | None] = {}
        for branch_name, model in fitted_results:
            modal_models[branch_name] = model
            modal_effective_pls_components[branch_name] = model.effective_pls_components

        return ModalBranchArtifacts(
            modal_models=modal_models,
            modal_feature_scales=_feature_scales(inputs),
            modal_effective_pls_components=modal_effective_pls_components,
            oof_base=oof_base,
            oof_weights=oof_weights,
        )

    def fit_meta_stage(self, targets: np.ndarray, branch_artifacts: ModalBranchArtifacts) -> "TraditionalFusionModel":
        meta_inputs = _modal_meta_features(branch_artifacts.oof_base, branch_artifacts.oof_weights)
        self.meta_effective_pls_components = None
        if self.config.meta_model_type == "pls":
            self.meta_effective_pls_components = _resolve_pls_n_components(self.config, meta_inputs)
        self.meta_model = _make_meta_model(self.config, meta_inputs)
        self.meta_model.fit(meta_inputs, targets)
        self.modal_models = branch_artifacts.modal_models
        self.modal_feature_scales = branch_artifacts.modal_feature_scales
        self.modal_effective_pls_components = branch_artifacts.modal_effective_pls_components
        return self

    def fit(self, dataset: PatentDataset, branch_artifacts: ModalBranchArtifacts | None = None) -> "TraditionalFusionModel":
        if branch_artifacts is None:
            branch_artifacts = self.fit_branch_stage(dataset)
        self.fit_meta_stage(dataset.targets, branch_artifacts)
        return self

    def _oof_meta_inputs(
        self,
        inputs: dict[str, np.ndarray],
        targets: np.ndarray,
        groups: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_samples, n_components = targets.shape
        base = np.zeros((n_samples, n_components, len(BRANCH_NAMES)), dtype=float)
        weights = np.zeros_like(base)
        n_groups = np.unique(groups).size
        n_splits = min(self.config.stacking_folds, n_groups)
        if n_splits < 2:
            raise ValueError(
                f"Only {n_groups} unique group(s) available but stacking requires "
                f"at least 2 (stacking_folds={self.config.stacking_folds}). "
                f"Increase training data diversity or reduce stacking_folds."
            )

        first_input = next(iter(inputs.values()))
        splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=self.config.random_state)
        splits = list(splitter.split(first_input, targets, groups))

        def _process_fold(fold_idx: int, train_idx: np.ndarray, valid_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            train_inputs = {name: values[train_idx] for name, values in inputs.items()}
            valid_inputs = {name: values[valid_idx] for name, values in inputs.items()}
            fitted: dict[str, SingleModalityMultiOutputModel] = {}
            for branch_name in BRANCH_NAMES:
                fitted[branch_name] = SingleModalityMultiOutputModel(self.config, branch_name).fit(
                    train_inputs[branch_name],
                    targets[train_idx],
                )
            base_valid = np.stack([fitted[name].predict(valid_inputs[name]) for name in BRANCH_NAMES], axis=2)
            weight_valid = _compute_modal_dynamic_weights(
                valid_inputs,
                base_valid,
                fitted,
                _feature_scales(train_inputs),
                self.config,
                seed_offset=fold_idx + 1,
            )
            return valid_idx, base_valid, weight_valid

        n_workers = _resolve_worker_count(self.config.n_jobs, len(splits))
        if n_workers > 1:
            fold_results = Parallel(n_jobs=n_workers, prefer="threads")(
                delayed(_process_fold)(fold_idx, train_idx, valid_idx)
                for fold_idx, (train_idx, valid_idx) in enumerate(splits)
            )
        else:
            fold_results = [
                _process_fold(fold_idx, train_idx, valid_idx)
                for fold_idx, (train_idx, valid_idx) in enumerate(splits)
            ]
        for valid_idx, base_valid, weight_valid in fold_results:
            base[valid_idx] = base_valid
            weights[valid_idx] = weight_valid
        return base, weights

    def predict_branches(self, dataset: PatentDataset) -> TraditionalPredictionCache:
        inputs = _branch_inputs(dataset, self.config.include_environment)
        base_predictions = np.stack([self.modal_models[name].predict(inputs[name]) for name in BRANCH_NAMES], axis=2)
        dynamic_weights = _compute_modal_dynamic_weights(
            inputs,
            base_predictions,
            self.modal_models,
            self.modal_feature_scales,
            self.config,
        )
        return TraditionalPredictionCache(base_predictions=base_predictions, dynamic_weights=dynamic_weights)

    def predict(
        self,
        dataset: PatentDataset,
        prediction_cache: TraditionalPredictionCache | None = None,
    ) -> TraditionalFusionPrediction:
        if prediction_cache is None:
            prediction_cache = self.predict_branches(dataset)
        base_predictions = prediction_cache.base_predictions
        dynamic_weights = prediction_cache.dynamic_weights
        meta_prediction = np.asarray(self.meta_model.predict(_modal_meta_features(base_predictions, dynamic_weights)), dtype=float)
        if meta_prediction.ndim == 1:
            meta_prediction = meta_prediction[:, None]
        by_model = {
            "acoustic": base_predictions[:, :, 0],
            "optical": base_predictions[:, :, 1],
            "thermal": base_predictions[:, :, 2],
            "fused": meta_prediction,
        }
        return TraditionalFusionPrediction(raw=by_model["fused"], by_model=by_model, dynamic_weights=dynamic_weights)

    def evaluate(
        self,
        dataset: PatentDataset,
        prediction_cache: TraditionalPredictionCache | None = None,
    ) -> tuple[pd.DataFrame, TraditionalFusionPrediction]:
        prediction = self.predict(dataset, prediction_cache=prediction_cache)
        rows: list[dict[str, object]] = []
        for model_name in _active_modal_model_names():
            values = prediction.by_model[model_name]
            for component_idx, component_name in enumerate(self.component_names):
                rows.append({"model": model_name, "component": component_name, **_regression_metrics(dataset.targets[:, component_idx], values[:, component_idx])})
        return pd.DataFrame(rows), prediction


# Compatibility aliases for unchanged imports in downstream scripts.
MultiComponentPatentModel = TraditionalFusionModel
