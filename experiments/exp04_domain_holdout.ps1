$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
python "$root\src\pipeline\domain_holdout.py" --data-dir "$root\outputs\exp01_traditional" --output-root "$root\outputs\exp04_domain" --profile v3_raw_no_env --combo-list pls_ridge xgboost_xgboost --seed 42 --n-jobs -1 --n-perturbations 10 --xgb-n-jobs 4
