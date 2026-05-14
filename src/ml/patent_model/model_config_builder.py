"""Shared ModelConfig construction helpers."""

from __future__ import annotations

import argparse

from patent_model.feature_profiles import FEATURE_PROFILES
from patent_model.modeling import ModelConfig


def build_model_config(args: argparse.Namespace, feature_profile_name: str) -> ModelConfig:
    """Build ModelConfig from CLI args and the resolved feature profile."""

    return ModelConfig(
        stacking_folds=args.stacking_folds,
        n_perturbations=args.n_perturbations,
        perturbation_scale=args.perturbation_scale,
        include_environment=bool(FEATURE_PROFILES[feature_profile_name]["include_environment"]),
        random_state=args.seed,
        branch_model_type=args.branch_model_type,
        meta_model_type=args.meta_model_type,
        pls_n_components=args.pls_n_components,
        xgb_n_estimators=args.xgb_n_estimators,
        xgb_max_depth=args.xgb_max_depth,
        xgb_learning_rate=args.xgb_learning_rate,
        xgb_device=args.xgb_device,
        xgb_n_jobs=args.xgb_n_jobs,
        n_jobs=getattr(args, "n_jobs", -1),
    )
