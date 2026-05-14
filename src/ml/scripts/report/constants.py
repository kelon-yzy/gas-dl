"""Constants shared by report figure generation modules."""

from __future__ import annotations


COMBO_ORDER: tuple[str, ...] = (
    "svr_ridge",
    "svr_pls",
    "svr_xgboost",
    "pls_ridge",
    "pls_pls",
    "pls_xgboost",
    "xgboost_ridge",
    "xgboost_pls",
    "xgboost_xgboost",
)
COMBO_LABELS: dict[str, str] = {
    "svr_ridge": "SVR / Ridge",
    "svr_pls": "SVR / PLS",
    "svr_xgboost": "SVR / XGBoost",
    "pls_ridge": "PLS / Ridge",
    "pls_pls": "PLS / PLS",
    "pls_xgboost": "PLS / XGBoost",
    "xgboost_ridge": "XGBoost / Ridge",
    "xgboost_pls": "XGBoost / PLS",
    "xgboost_xgboost": "XGBoost / XGBoost",
}
COMBO_ALIASES: dict[str, str] = {
    "svr_xgb": "svr_xgboost",
    "xgb_ridge": "xgboost_ridge",
    "xgb_pls": "xgboost_pls",
    "xgb_xgb": "xgboost_xgboost",
}
PREFERRED_PROFILE = "derived_env_mc_aug"
PREFERRED_COMBO = "svr_ridge"

