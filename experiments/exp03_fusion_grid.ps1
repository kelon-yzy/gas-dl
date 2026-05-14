$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
foreach ($combo in "xgboost_ridge", "pls_ridge") {
  python "$root\src\ml\scripts\run_four_component_model_grid.py" --raw-data-dir "$root\outputs\exp01_traditional" --env-data-dir "$root\outputs\exp01_traditional" --output-root "$root\outputs\exp03_fusion" --tag $combo --seed 42 --combo-list $combo --max-workers 1 --n-jobs 2 --xgb-n-jobs 4
}
python "$root\src\ml\scripts\run_four_component_model_grid.py" --raw-data-dir "$root\outputs\exp01_traditional" --env-data-dir "$root\outputs\exp01_traditional" --output-root "$root\outputs\exp03_fusion" --tag svr_ridge_diag --seed 42 --combo-list svr_ridge --train-limit 5000 --test-limit 1500 --max-workers 1 --n-jobs 2 --xgb-n-jobs 1
