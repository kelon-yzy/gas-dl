"""三模态传统模型、动态权重和融合逻辑。"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
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


# 配置和输出结构定义区。这里不做计算，只负责说明模型输入输出形状。
@dataclass(frozen=True)
class ModelConfig:
    """模型超参数集合。"""

    C: float = 20.0
    epsilon: float = 0.003
    gamma: str = "scale"
    ridge_alpha: float = 0.8
    stacking_folds: int = 5
    n_perturbations: int = 24
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
    # CPU 并行配置，-1 表示让 joblib 按 cpu_count 分配；0/1 退化到串行。
    n_jobs: int = -1


@dataclass
class ComponentPrediction:
    """单个组分的各路预测结果和动态权重。"""

    predictions: dict[str, np.ndarray]
    dynamic_weights: np.ndarray
    base_predictions: np.ndarray


@dataclass
class MultiComponentPrediction:
    """三个组分拼接后的整体预测结果。"""

    raw: np.ndarray
    by_model: dict[str, np.ndarray]
    dynamic_weights: np.ndarray


@dataclass
class ComponentBranchArtifacts:
    """单组分分支阶段的可复用产物。"""

    branch_models: dict[str, Pipeline]
    branch_feature_scales: dict[str, np.ndarray]
    branch_effective_pls_components: dict[str, int]
    oof_base: np.ndarray
    oof_weights: np.ndarray


def _safe_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """在小样本或常数标签子集上安全计算 R2。"""

    if y_true.size < 2:
        return float("nan")
    if np.allclose(y_true, y_true[0], rtol=0.0, atol=1e-12):
        return float("nan")
    return float(r2_score(y_true, y_pred))


# 基础工具函数区：构建分支模型、准备分支输入、估计扰动尺度。
def _resolve_pls_n_components(config: ModelConfig, features: np.ndarray | None = None) -> int:
    """按当前输入规模约束 PLS 组件数。"""

    if config.pls_n_components <= 0:
        raise ValueError("pls_n_components must be positive.")
    if features is None:
        return config.pls_n_components
    return min(config.pls_n_components, features.shape[0], features.shape[1])


def _make_pipeline(config: ModelConfig, features: np.ndarray | None = None) -> Pipeline:
    """按配置为单一路模态构造回归流水线。"""

    if config.branch_model_type == "svr":
        estimator = SVR(kernel="rbf", C=config.C, epsilon=config.epsilon, gamma=config.gamma)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("svr", estimator),
        ])
    if config.branch_model_type == "pls":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("pls", PLSRegression(n_components=_resolve_pls_n_components(config, features))),
        ])
    if config.branch_model_type == "xgboost":
        return Pipeline([
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
        ])
    raise ValueError(f"Unknown branch_model_type: {config.branch_model_type}")


def _make_meta_model(config: ModelConfig, features: np.ndarray | None = None) -> BaseEstimator:
    """按配置构造元学习器。"""

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


def _fit_branch_model(
    prototype: Pipeline,
    config: ModelConfig,
    features: np.ndarray,
    target: np.ndarray,
) -> Pipeline:
    """按分支模型类型克隆或重建估计器，再拟合当前输入。"""

    if config.branch_model_type == "pls" and isinstance(prototype, Pipeline):
        model = _make_pipeline(config, features)
    else:
        model = clone(prototype)
    return model.fit(features, target)


def _active_model_names(config: ModelConfig) -> tuple[str, ...]:
    """返回当前配置下实际会产出的模型名称。"""

    return (
        "acoustic",
        "optical",
        "thermal",
        "fixed_average",
        "dynamic_average",
        f"dynamic_{config.meta_model_type}",
    )


def _branch_inputs(dataset: PatentDataset, include_environment: bool) -> dict[str, np.ndarray]:
    """为三条模态分支准备输入矩阵，可选拼接环境变量。"""

    if include_environment:
        # 环境参数目前直接拼接进各模态分支，后续可改成显式补偿项。
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


def _compute_dynamic_weights(
    branch_inputs: dict[str, np.ndarray],
    base_predictions: np.ndarray,
    branch_models: dict[str, Pipeline],
    feature_scales: dict[str, np.ndarray],
    config: ModelConfig,
    seed_offset: int = 0,
) -> np.ndarray:
    """用 MC 扰动估计分支稳定性，再换算成动态权重。"""

    rng = np.random.default_rng(config.random_state + seed_offset)
    n_samples = base_predictions.shape[0]
    n_perturb = config.n_perturbations
    drift_mse = np.zeros((n_samples, len(BRANCH_NAMES)), dtype=float)
    # 对每一条模态分支分别做扰动，得到“当前样本下这一路有多稳定”的估计。
    # 旧实现是 24 次小批量 predict 串行；这里一次性生成全部扰动样本，
    # 合成 (n_perturb * n_samples, n_features) 的大批量再单次 predict。
    # 注意：随机数生成顺序与原实现不同，drift_mse 数值会有统计等价的微小差异，
    # 但权重分布性质和分支稳定性排序保持一致。
    for branch_idx, branch_name in enumerate(BRANCH_NAMES):
        features = branch_inputs[branch_name]
        scale = feature_scales[branch_name] * config.perturbation_scale
        n_features = features.shape[1]
        # rng.normal 默认 loc=0, scale=1，再按列乘以 scale 完成各维度独立扰动。
        # shape: (n_perturb, n_samples, n_features)
        noise = rng.standard_normal((n_perturb, n_samples, n_features)) * scale
        perturbed = features[None, :, :] + noise
        flat = perturbed.reshape(n_perturb * n_samples, n_features)
        flat_pred = branch_models[branch_name].predict(flat)
        # 还原回 (n_samples, n_perturb) 方便统一计算 drift_mse。
        perturb_array = flat_pred.reshape(n_perturb, n_samples).T
        drift_mse[:, branch_idx] = np.mean((perturb_array - base_predictions[:, [branch_idx]]) ** 2, axis=1)
    # 置信度取 drift_mse 的倒数，再归一化成每个样本三路权重和为 1。
    confidence = 1.0 / (drift_mse + config.uncertainty_floor)
    return confidence / confidence.sum(axis=1, keepdims=True)


def _meta_features(base_predictions: np.ndarray, dynamic_weights: np.ndarray) -> np.ndarray:
    """把基础预测、加权交互项和权重本身拼成元学习器输入。"""

    return np.hstack([base_predictions, base_predictions * dynamic_weights, dynamic_weights])


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


# 单组分建模区：一套三模态 SVR + 一个 Ridge 元学习器。
class SingleComponentPatentModel:
    """针对单个目标组分的一套三模态模型。"""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.branch_models = {name: _make_pipeline(self.config) for name in BRANCH_NAMES}
        self.meta_model = _make_meta_model(self.config)
        self.branch_feature_scales: dict[str, np.ndarray] = {}
        self.branch_effective_pls_components: dict[str, int] = {}
        self.meta_effective_pls_components: int | None = None

    def fit_branch_stage(self, dataset: PatentDataset, target: np.ndarray) -> ComponentBranchArtifacts:
        """拟合分支模型并产出可复用的 OOF 特征。"""

        inputs = _branch_inputs(dataset, self.config.include_environment)
        groups = dataset.metadata["mixture_id"].to_numpy(dtype=object)
        oof_base, oof_weights = self._oof_meta_inputs(inputs, target, groups)
        branch_effective_pls_components: dict[str, int] = {}
        branch_models: dict[str, Pipeline] = {}
        for branch_name in BRANCH_NAMES:
            if self.config.branch_model_type == "pls":
                branch_effective_pls_components[branch_name] = _resolve_pls_n_components(self.config, inputs[branch_name])
            branch_models[branch_name] = _fit_branch_model(
                self.branch_models[branch_name],
                self.config,
                inputs[branch_name],
                target,
            )
        return ComponentBranchArtifacts(
            branch_models=branch_models,
            branch_feature_scales=_feature_scales(inputs),
            branch_effective_pls_components=branch_effective_pls_components,
            oof_base=oof_base,
            oof_weights=oof_weights,
        )

    def fit_meta_stage(self, target: np.ndarray, branch_artifacts: ComponentBranchArtifacts) -> "SingleComponentPatentModel":
        """基于可复用分支产物拟合当前元学习器。"""

        meta_inputs = _meta_features(branch_artifacts.oof_base, branch_artifacts.oof_weights)
        self.meta_effective_pls_components = None
        if self.config.meta_model_type == "pls":
            self.meta_effective_pls_components = _resolve_pls_n_components(self.config, meta_inputs)
        self.meta_model = _make_meta_model(self.config, meta_inputs)
        self.meta_model.fit(meta_inputs, target)
        self.branch_models = branch_artifacts.branch_models
        self.branch_feature_scales = branch_artifacts.branch_feature_scales
        self.branch_effective_pls_components = branch_artifacts.branch_effective_pls_components
        return self

    def fit(
        self,
        dataset: PatentDataset,
        target: np.ndarray,
        branch_artifacts: ComponentBranchArtifacts | None = None,
    ) -> "SingleComponentPatentModel":
        """先做组内隔离的 OOF 融合训练，再拟合当前元学习器。"""

        if branch_artifacts is None:
            branch_artifacts = self.fit_branch_stage(dataset, target)
        self.fit_meta_stage(target, branch_artifacts)
        return self

    def _oof_meta_inputs(
        self,
        inputs: dict[str, np.ndarray],
        target: np.ndarray,
        groups: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """生成元学习器训练所需的 OOF 基础预测和 OOF 动态权重。"""

        n_samples = target.shape[0]
        base = np.zeros((n_samples, len(BRANCH_NAMES)), dtype=float)
        weights = np.zeros_like(base)
        n_groups = np.unique(groups).size
        n_splits = min(self.config.stacking_folds, n_groups)
        if n_splits < 2:
            # 组数太少时退化成全量拟合，至少保证流程可执行。
            fitted = {
                name: _fit_branch_model(self.branch_models[name], self.config, inputs[name], target)
                for name in BRANCH_NAMES
            }
            base_predictions = np.column_stack([fitted[name].predict(inputs[name]) for name in BRANCH_NAMES])
            return base_predictions, _compute_dynamic_weights(inputs, base_predictions, fitted, _feature_scales(inputs), self.config)

        first_input = next(iter(inputs.values()))
        splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=self.config.random_state)
        splits = list(splitter.split(first_input, target, groups))

        def _process_fold(fold_idx: int, train_idx: np.ndarray, valid_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """单折拟合 + 验证折预测 + 动态权重，合成可并行的任务单元。"""

            train_inputs = {name: values[train_idx] for name, values in inputs.items()}
            valid_inputs = {name: values[valid_idx] for name, values in inputs.items()}
            # 每一折都保证同一个 mixture_id 不会同时出现在训练和验证里。
            fitted: dict[str, Pipeline] = {}
            for branch_name in BRANCH_NAMES:
                fitted[branch_name] = _fit_branch_model(
                    self.branch_models[branch_name],
                    self.config,
                    train_inputs[branch_name],
                    target[train_idx],
                )
            base_valid = np.column_stack([fitted[name].predict(valid_inputs[name]) for name in BRANCH_NAMES])
            weight_valid = _compute_dynamic_weights(
                valid_inputs,
                base_valid,
                fitted,
                _feature_scales(train_inputs),
                self.config,
                seed_offset=fold_idx + 1,
            )
            return valid_idx, base_valid, weight_valid

        # SVR/PLS/XGBoost 的 fit 在底层 C 实现里会释放 GIL，threading backend
        # 既能并行又不会触发 pickle 开销。n_jobs=-1 表示按 cpu_count 分配。
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

    def predict(self, dataset: PatentDataset) -> ComponentPrediction:
        """输出单个组分的单模态结果、两种平均融合和最终元学习融合。"""

        # 第 1 段：三条分支各自出一个基础预测。
        inputs = _branch_inputs(dataset, self.config.include_environment)
        base_predictions = np.column_stack([self.branch_models[name].predict(inputs[name]) for name in BRANCH_NAMES])
        # 第 2 段：基于当前样本的稳定性估计动态权重。
        dynamic_weights = _compute_dynamic_weights(
            inputs,
            base_predictions,
            self.branch_models,
            self.branch_feature_scales,
            self.config,
        )
        # 第 3 段：给出固定平均、动态平均和最终元学习融合三个版本。
        fixed_average = base_predictions.mean(axis=1)
        dynamic_average = np.sum(base_predictions * dynamic_weights, axis=1)
        meta_key = f"dynamic_{self.config.meta_model_type}"
        meta_prediction = self.meta_model.predict(_meta_features(base_predictions, dynamic_weights))
        return ComponentPrediction(
            predictions={
                "acoustic": base_predictions[:, 0],
                "optical": base_predictions[:, 1],
                "thermal": base_predictions[:, 2],
                "fixed_average": fixed_average,
                "dynamic_average": dynamic_average,
                meta_key: meta_prediction,
            },
            dynamic_weights=dynamic_weights,
            base_predictions=base_predictions,
        )


# 多组分封装区：把单组分流程重复到 H2、CH4、CO2 三个目标上。
class MultiComponentPatentModel:
    """把单组分模型复制三份，分别预测 H2、CH4、CO2。"""

    def __init__(self, config: ModelConfig | None = None, component_names: tuple[str, ...] = COMPONENT_NAMES) -> None:
        self.config = config or ModelConfig()
        self.component_names = component_names
        self.models = [SingleComponentPatentModel(self.config) for _ in component_names]

    def fit_branch_stage(self, dataset: PatentDataset) -> list[ComponentBranchArtifacts]:
        """按组分拟合分支阶段，供多个元学习器复用。"""

        if dataset.targets.shape[1] != len(self.component_names):
            raise ValueError("Dataset target width does not match component_names.")
        # 各组分的 fit_branch_stage 互不依赖；threading backend + SVR 释放 GIL
        # 让 component 维度也能并行，吃满多核。
        n_workers = _resolve_worker_count(self.config.n_jobs, len(self.models))
        logger.debug(
            "fit_branch_stage components=%d n_workers=%d",
            len(self.models),
            n_workers,
        )
        if n_workers > 1:
            return Parallel(n_jobs=n_workers, prefer="threads")(
                delayed(model.fit_branch_stage)(dataset, dataset.target_for(component_idx))
                for component_idx, model in enumerate(self.models)
            )
        return [
            model.fit_branch_stage(dataset, dataset.target_for(component_idx))
            for component_idx, model in enumerate(self.models)
        ]

    def fit(
        self,
        dataset: PatentDataset,
        branch_artifacts: list[ComponentBranchArtifacts] | None = None,
    ) -> "MultiComponentPatentModel":
        """按组分逐个拟合模型。"""

        if dataset.targets.shape[1] != len(self.component_names):
            raise ValueError("Dataset target width does not match component_names.")
        if branch_artifacts is None:
            branch_artifacts = self.fit_branch_stage(dataset)
        if len(branch_artifacts) != len(self.models):
            raise ValueError("branch_artifacts length does not match component_names.")
        for component_idx, model in enumerate(self.models):
            model.fit(dataset, dataset.target_for(component_idx), branch_artifacts=branch_artifacts[component_idx])
        return self

    def predict(self, dataset: PatentDataset) -> MultiComponentPrediction:
        """汇总三个组分在各个模型家族下的预测矩阵。"""

        component_predictions = [model.predict(dataset) for model in self.models]
        by_model: dict[str, np.ndarray] = {}
        # 把每个组分的一维输出按列拼回 (n_samples, 3) 的矩阵。
        for model_name in _active_model_names(self.config):
            by_model[model_name] = np.column_stack([prediction.predictions[model_name] for prediction in component_predictions])
        raw = by_model[f"dynamic_{self.config.meta_model_type}"]
        weights = np.stack([prediction.dynamic_weights for prediction in component_predictions], axis=1)
        return MultiComponentPrediction(raw=raw, by_model=by_model, dynamic_weights=weights)

    def evaluate(self, dataset: PatentDataset) -> tuple[pd.DataFrame, MultiComponentPrediction]:
        """对每个模型家族、每个组分分别计算误差指标。"""

        prediction = self.predict(dataset)
        rows: list[dict[str, object]] = []
        # 输出是长表结构，方便后续直接 groupby 或画图。
        for model_name, values in prediction.by_model.items():
            for component_idx, component_name in enumerate(self.component_names):
                metrics = _regression_metrics(dataset.targets[:, component_idx], values[:, component_idx])
                rows.append({"model": model_name, "component": component_name, **metrics})
        return pd.DataFrame(rows), prediction
