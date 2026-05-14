"""环境补偿实验与鲁棒性脚本共享工具。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from patent_model.feature_profiles import FEATURE_PROFILES
from patent_model.model_config_builder import build_model_config


PROFILES = ("v3_raw_no_env", "v3_raw_tph", "v3_env")


def add_model_args(parser: argparse.ArgumentParser, positive_int_type: Callable[[str], int]) -> None:
    """为脚本统一添加模型类型和超参数选项。"""

    parser.add_argument("--component-mode", default="four", choices=("three", "four"))
    parser.add_argument("--branch-model-type", default="svr", choices=("svr", "pls", "xgboost"))
    parser.add_argument("--meta-model-type", default="ridge", choices=("ridge", "pls", "xgboost"))
    parser.add_argument("--perturbation-scale", type=float, default=0.04)
    parser.add_argument("--pls-n-components", type=positive_int_type, default=10)
    parser.add_argument("--xgb-n-estimators", type=positive_int_type, default=100)
    parser.add_argument("--xgb-max-depth", type=positive_int_type, default=5)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-device", default="cpu")
    parser.add_argument("--xgb-n-jobs", type=positive_int_type, default=1)
    parser.add_argument("--n-jobs", type=int, default=-1)


def extend_model_cli_args(args: argparse.Namespace, argv: list[str]) -> None:
    """把统一模型选项追加到子命令参数列表。"""

    argv.extend(
        [
            "--component-mode",
            args.component_mode,
            "--branch-model-type",
            args.branch_model_type,
            "--meta-model-type",
            args.meta_model_type,
            "--perturbation-scale",
            str(args.perturbation_scale),
            "--pls-n-components",
            str(args.pls_n_components),
            "--xgb-n-estimators",
            str(args.xgb_n_estimators),
            "--xgb-max-depth",
            str(args.xgb_max_depth),
            "--xgb-learning-rate",
            str(args.xgb_learning_rate),
            "--xgb-device",
            args.xgb_device,
            "--xgb-n-jobs",
            str(args.xgb_n_jobs),
            "--n-jobs",
            str(args.n_jobs),
        ]
    )


def profile_data_dir(profile: str, raw_data_dir: Path, env_data_dir: Path) -> Path:
    """根据 profile 选择原始或环境补偿数据目录。"""

    return env_data_dir if profile == "v3_env" else raw_data_dir


def base_feature_profile(profile: str) -> str:
    """把实验层 profile 映射到训练入口实际使用的基础 profile。"""

    mapping = {
        "v3_raw_no_env": "v3_waveform_dual_channel_four",
        "v3_raw_tph": "v3_waveform_dual_channel_four",
        "v3_env": "v3_waveform_dual_channel_env_four",
    }
    if profile not in mapping:
        raise ValueError(f"Unknown experiment profile: {profile}")
    return mapping[profile]


def resolve_feature_profile_name(profile: str, component_mode: str) -> str:
    """根据组分模式映射实验 profile 到实际 profile 名。"""

    base = base_feature_profile(profile)
    if component_mode != "four":
        raise ValueError("Current V3.1 traditional experiments only support four-component mode.")
    return base


def build_meta_key(meta_model_type: str) -> str:
    """返回当前元学习器对应的预测输出键名。"""

    return f"dynamic_{meta_model_type}"


def require_known_profile(profile: str) -> None:
    """校验实验脚本声明的 profile 在基础配置中存在。"""

    if base_feature_profile(profile) not in FEATURE_PROFILES:
        raise ValueError(f"Unknown profile configured for experiment: {profile}")


def require_known_profile_mode(profile: str, component_mode: str) -> None:
    """校验 profile 在当前组分模式下有对应配置。"""

    resolved = resolve_feature_profile_name(profile, component_mode)
    if resolved not in FEATURE_PROFILES:
        raise ValueError(f"Unknown profile configured for component mode {component_mode}: {profile}")
