"""训练主脚本：读取数据、训练模型并导出结果。"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from patent_model.config import BRANCH_NAMES
from patent_model.data_loader import grouped_train_test_split, load_patent_dataset
from patent_model.dataset import PatentDataset
from patent_model.environment_augmentation import augment_derived_env_training_data
from patent_model.feature_profiles import FEATURE_PROFILES, has_embedded_environment
from patent_model.fault_labels import build_observed_fault_labels
from patent_model.logging_utils import get_logger
from patent_model.model_config_builder import build_model_config
from patent_model.modeling import (
    ModalBranchArtifacts,
    TraditionalFusionModel,
    TraditionalFusionPrediction,
    TraditionalPredictionCache,
)
from scripts._cli_utils import limit_dataset, positive_int


logger = get_logger(__name__)


@dataclass(frozen=True)
class PreparedTrainingData:
    """训练入口预处理后的可复用数据包。"""

    data_dir: Path
    feature_profile_name: str
    train_original_samples: int
    train: PatentDataset
    test: PatentDataset


def build_parser() -> argparse.ArgumentParser:
    """定义训练脚本的命令行参数。"""

    parser = argparse.ArgumentParser(description="Train the patent-aligned multimodal gas composition model.")
    parser.add_argument("--data-dir", default="../output", help="Path to the simulation data export directory.")
    parser.add_argument("--output-dir", default="outputs/run_001", help="Directory for CSV and JSON outputs.")
    parser.add_argument("--feature-profile", default="raw_tph", choices=sorted(FEATURE_PROFILES), help="Feature profile to load.")
    parser.add_argument("--component-mode", default="three", choices=("three", "four"), help="Prediction target width.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Group split test ratio by mixture_id.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--train-limit", type=positive_int, help="Optional cap for training samples.")
    parser.add_argument("--test-limit", type=positive_int, help="Optional cap for test samples.")
    parser.add_argument("--n-perturbations", type=positive_int, default=24, help="MC perturbation count per modality.")
    parser.add_argument("--stacking-folds", type=positive_int, default=5, help="OOF folds for Ridge meta learner.")
    parser.add_argument("--perturbation-scale", type=float, default=0.04, help="Feature-space noise scale for MC drift MSE.")
    parser.add_argument("--branch-model-type", default="svr", choices=("svr", "pls", "xgboost"), help="Base branch regressor type.")
    parser.add_argument("--meta-model-type", default="ridge", choices=("ridge", "pls", "xgboost"), help="Meta learner type.")
    parser.add_argument("--pls-n-components", type=positive_int, default=10, help="PLS component count.")
    parser.add_argument("--xgb-n-estimators", type=positive_int, default=100, help="XGBoost tree count.")
    parser.add_argument("--xgb-max-depth", type=positive_int, default=5, help="XGBoost max tree depth.")
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05, help="XGBoost learning rate.")
    parser.add_argument("--xgb-device", default="cpu", help="XGBoost device, e.g. cpu or cuda.")
    parser.add_argument("--xgb-n-jobs", type=positive_int, default=1, help="XGBoost worker threads per model.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="并行训练线程数（fold/分支/组分维度复用）。-1 表示按 cpu_count 分配；1 退化串行。")
    parser.add_argument("--mc-env-samples", type=int, default=0, help="Offline MC environment augmentation copies for derived_env training.")
    parser.add_argument("--mc-env-sigma-t", type=float, default=0.5, help="Temperature noise sigma in deg C for MC environment augmentation.")
    parser.add_argument("--mc-env-sigma-p", type=float, default=0.005, help="Pressure noise sigma in MPa for MC environment augmentation.")
    parser.add_argument("--mc-env-sigma-h", type=float, default=1.0, help="Humidity noise sigma in %RH for MC environment augmentation.")
    parser.add_argument("--metadata-filter", default="none", help="Optional metadata filter, e.g. none or detection.")
    parser.add_argument("--stage-filter", default="stable", choices=("none", "stable"), help="Optional stage filter for stable detection samples.")
    parser.add_argument("--physical-range-filter", default="none", help="Optional physical feature range filter.")
    parser.add_argument("--label-closure-filter", default="none", help="Optional label closure filter.")
    parser.add_argument("--duplicate-filter", default="per_mixture_limit", help="Optional duplicate handling mode.")
    parser.add_argument("--duplicate-per-mixture-limit", type=positive_int, default=3, help="Optional cap when duplicate_filter=per_mixture_limit.")
    parser.add_argument("--duplicate-filter-seed", type=int, default=42, help="Random seed for duplicate filtering.")
    return parser


def _predictions_frame(test: PatentDataset, prediction: TraditionalFusionPrediction) -> pd.DataFrame:
    """把测试集真实值和最终预测整理成表格。"""

    frame = pd.DataFrame({"sample_id": test.sample_ids})
    for idx, name in enumerate(test.component_names):
        frame[f"true_{name}"] = test.targets[:, idx]
        frame[f"acoustic_pred_{name}"] = prediction.by_model["acoustic"][:, idx]
        frame[f"optical_pred_{name}"] = prediction.by_model["optical"][:, idx]
        frame[f"thermal_pred_{name}"] = prediction.by_model["thermal"][:, idx]
        frame[f"fused_pred_{name}"] = prediction.raw[:, idx]
    frame["target_sum"] = test.targets.sum(axis=1)
    frame["fused_pred_sum"] = prediction.raw.sum(axis=1)
    if "x_N2" in test.metadata.columns:
        frame["background_N2"] = test.metadata["x_N2"].to_numpy()
    for column in ("mixture_id", "stage_id", "pressure_stage", "distance_stage", "fault_case", "fault_severity"):
        if column in test.metadata.columns:
            frame[column] = test.metadata[column].to_numpy()
    return frame


def _weights_frame(test: PatentDataset, prediction: TraditionalFusionPrediction) -> pd.DataFrame:
    """把每个样本、每个组分的动态权重展开成明细表。"""

    rows: list[dict[str, object]] = []
    for sample_idx, sample_id in enumerate(test.sample_ids):
        for component_idx, component_name in enumerate(test.component_names):
            row = {"sample_id": sample_id, "component": component_name}
            for branch_idx, branch_name in enumerate(BRANCH_NAMES):
                row[f"{branch_name}_weight"] = float(prediction.dynamic_weights[sample_idx, component_idx, branch_idx])
            rows.append(row)
    return pd.DataFrame(rows)


def resolve_data_dir(data_dir: str) -> Path:
    """解析数据目录，兼容从模块根或 scripts 入口运行。"""

    path = Path(data_dir)
    if path.is_absolute() or path.exists():
        return path
    fallback = ROOT.parent / path
    if fallback.exists():
        return fallback
    return path


def _resolve_feature_profile_name(profile: str, component_mode: str) -> str:
    """根据组分模式映射到实际 feature profile 名称。"""

    if component_mode == "four":
        if profile.endswith("_four"):
            if profile not in FEATURE_PROFILES:
                raise ValueError(f"Unknown four-component feature profile: {profile}")
            return profile
        four_profile = f"{profile}_four"
        if four_profile not in FEATURE_PROFILES:
            raise ValueError(f"Feature profile does not support four-component mode: {profile}")
        return four_profile
    return profile


def _validate_args(args: argparse.Namespace) -> None:
    """校验训练入口参数约束。"""

    if args.mc_env_samples < 0:
        raise ValueError("--mc-env-samples must be >= 0.")
    if args.mc_env_samples > 0 and not has_embedded_environment(args.feature_profile):
        raise ValueError("--mc-env-samples is only supported with profiles that embed environment columns.")


def prepare_training_data(
    args: argparse.Namespace,
    dataset: PatentDataset | None = None,
) -> PreparedTrainingData:
    """加载、切分并按需增强数据，供多个模型配置复用。"""

    feature_profile_name = _resolve_feature_profile_name(args.feature_profile, args.component_mode)
    data_dir = resolve_data_dir(args.data_dir)
    if dataset is None:
        dataset = load_patent_dataset(
            data_dir,
            profile=feature_profile_name,
            metadata_filter=args.metadata_filter,
            stage_filter=args.stage_filter,
            physical_range_filter=args.physical_range_filter,
            label_closure_filter=args.label_closure_filter,
            duplicate_filter=args.duplicate_filter,
            duplicate_per_mixture_limit=args.duplicate_per_mixture_limit,
            duplicate_filter_seed=args.duplicate_filter_seed,
        )
    observed_labels = build_observed_fault_labels(dataset)
    dataset = dataset.with_fault_labels(observed_labels)

    train, test = grouped_train_test_split(dataset, test_ratio=args.test_ratio, seed=args.seed)
    train = limit_dataset(train, args.train_limit)
    test = limit_dataset(test, args.test_limit)
    train_original_samples = train.n_samples
    if args.mc_env_samples > 0:
        train = augment_derived_env_training_data(
            train,
            mc_samples=args.mc_env_samples,
            sigma_t=args.mc_env_sigma_t,
            sigma_p=args.mc_env_sigma_p,
            sigma_h=args.mc_env_sigma_h,
            seed=args.seed,
            profile=feature_profile_name,
        )
    return PreparedTrainingData(
        data_dir=data_dir,
        feature_profile_name=feature_profile_name,
        train_original_samples=train_original_samples,
        train=train,
        test=test,
    )


def run_training(
    args: argparse.Namespace,
    dataset: PatentDataset | None = None,
    prepared_data: PreparedTrainingData | None = None,
    branch_artifacts: ModalBranchArtifacts | None = None,
    prediction_cache_holder: dict[str, TraditionalPredictionCache] | None = None,
    progress=None,
    progress_context: dict[str, object] | None = None,
) -> dict[str, object]:
    """执行完整训练流程并写出 CSV / JSON 结果。"""

    _validate_args(args)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if prepared_data is None:
        prepared_data = prepare_training_data(args, dataset=dataset)

    config = build_model_config(args, prepared_data.feature_profile_name)
    logger.info(
        "training profile=%s branch=%s meta=%s train=%d test=%d",
        prepared_data.feature_profile_name,
        config.branch_model_type,
        config.meta_model_type,
        prepared_data.train.n_samples,
        prepared_data.test.n_samples,
    )
    if progress is not None:
        progress.update_stage(
            stage="fit_model",
            current_task=f"profile={progress_context.get('profile')} combo={progress_context.get('combo')}",
            completed=progress_context.get('completed'),
            total=progress_context.get('total'),
        )
    fit_started = perf_counter()
    model = TraditionalFusionModel(config=config, component_names=prepared_data.train.component_names).fit(
        prepared_data.train,
        branch_artifacts=branch_artifacts,
    )
    fit_seconds = perf_counter() - fit_started
    if progress is not None:
        progress.update_metric(fit_seconds=fit_seconds)
        progress.update_stage(
            stage="evaluate",
            current_task=f"profile={progress_context.get('profile')} combo={progress_context.get('combo')}",
            completed=progress_context.get('completed'),
            total=progress_context.get('total'),
        )
    evaluate_started = perf_counter()
    prediction_cache_reused = False
    prediction_cache = None
    if prediction_cache_holder is not None:
        prediction_cache = prediction_cache_holder.get("value")
        prediction_cache_reused = prediction_cache is not None
    if prediction_cache is None:
        prediction_cache = model.predict_branches(prepared_data.test)
        if prediction_cache_holder is not None:
            prediction_cache_holder["value"] = prediction_cache
    metrics, prediction = model.evaluate(prepared_data.test, prediction_cache=prediction_cache)
    evaluate_seconds = perf_counter() - evaluate_started

    write_started = perf_counter()
    predictions = _predictions_frame(prepared_data.test, prediction)
    weights = _weights_frame(prepared_data.test, prediction)

    predictions.to_csv(output / "predictions.csv", index=False)
    metrics.to_csv(output / "component_metrics.csv", index=False)
    weights.to_csv(output / "dynamic_weights.csv", index=False)
    write_seconds = perf_counter() - write_started

    summary = {
        "data_dir": str(prepared_data.data_dir.resolve()),
        "feature_profile": args.feature_profile,
        "resolved_feature_profile": prepared_data.feature_profile_name,
        "component_mode": args.component_mode,
        "include_environment": config.include_environment,
        "mc_env_samples": args.mc_env_samples,
        "mc_env_sigma_T": args.mc_env_sigma_t,
        "mc_env_sigma_P": args.mc_env_sigma_p,
        "mc_env_sigma_H": args.mc_env_sigma_h,
        "metadata_filter": args.metadata_filter,
        "stage_filter": args.stage_filter,
        "physical_range_filter": args.physical_range_filter,
        "label_closure_filter": args.label_closure_filter,
        "duplicate_filter": args.duplicate_filter,
        "duplicate_per_mixture_limit": args.duplicate_per_mixture_limit,
        "duplicate_filter_seed": args.duplicate_filter_seed,
        "train_original_samples": prepared_data.train_original_samples,
        "train_samples": prepared_data.train.n_samples,
        "test_samples": prepared_data.test.n_samples,
        "components": list(prepared_data.test.component_names),
        "seed": args.seed,
        "n_perturbations": args.n_perturbations,
        "stacking_folds": args.stacking_folds,
        "branch_model_type": config.branch_model_type,
        "meta_model_type": config.meta_model_type,
        "pls_n_components_requested": config.pls_n_components,
        "xgb_n_estimators": config.xgb_n_estimators,
        "xgb_max_depth": config.xgb_max_depth,
        "xgb_learning_rate": config.xgb_learning_rate,
        "xgb_device": config.xgb_device,
        "xgb_n_jobs": config.xgb_n_jobs,
        "n_jobs": config.n_jobs,
        "fit_seconds": fit_seconds,
        "evaluate_seconds": evaluate_seconds,
        "write_seconds": write_seconds,
        "total_seconds": fit_seconds + evaluate_seconds + write_seconds,
        "prediction_cache_reused": prediction_cache_reused,
        "filter_report": prepared_data.train.filter_report,
        "metadata_filter_report": prepared_data.train.filter_report,
        "stage_filter_report": prepared_data.train.filter_report.get("stage_filter"),
    }
    modal_summary: dict[str, dict[str, float]] = {}
    best_model_name = None
    best_rmse = None
    for model_name in ("acoustic", "optical", "thermal", "fused"):
        modal_metrics = metrics[metrics["model"] == model_name]
        modal_summary[model_name] = {
            "macro_RMSE_pp": float(modal_metrics["RMSE_pp"].mean()),
            "macro_MRE_pct": float(modal_metrics["MRE_pct"].mean()),
            "macro_R2": float(modal_metrics["R2"].mean()),
            "macro_MaxRE_pct": float(modal_metrics["MaxRE_pct"].max()),
        }
        summary[f"{model_name}_macro_RMSE_pp"] = modal_summary[model_name]["macro_RMSE_pp"]
        summary[f"{model_name}_macro_MRE_pct"] = modal_summary[model_name]["macro_MRE_pct"]
        summary[f"{model_name}_macro_R2"] = modal_summary[model_name]["macro_R2"]
        summary[f"{model_name}_macro_MaxRE_pct"] = modal_summary[model_name]["macro_MaxRE_pct"]
        if best_rmse is None or modal_summary[model_name]["macro_RMSE_pp"] < best_rmse:
            best_rmse = modal_summary[model_name]["macro_RMSE_pp"]
            best_model_name = model_name
    summary["best_model_by_macro_RMSE_pp"] = best_model_name
    if progress is not None:
        progress.update_metric(
            macro_RMSE=summary["fused_macro_RMSE_pp"],
            total_seconds=summary["total_seconds"],
            cache=prediction_cache_reused,
        )
    summary["modal_effective_pls_components"] = {
        "acoustic": model.modal_effective_pls_components.get("acoustic"),
        "optical": model.modal_effective_pls_components.get("optical"),
        "thermal": model.modal_effective_pls_components.get("thermal"),
        "fused": model.meta_effective_pls_components,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "training done fused_macro_RMSE_pp=%.4f best_model=%s",
        summary["fused_macro_RMSE_pp"],
        summary["best_model_by_macro_RMSE_pp"],
    )
    return summary


def main(argv: list[str] | None = None) -> dict[str, object]:
    """执行完整训练流程并写出 CSV / JSON 结果。"""

    args = build_parser().parse_args(argv)
    return run_training(args)


if __name__ == "__main__":
    main()
