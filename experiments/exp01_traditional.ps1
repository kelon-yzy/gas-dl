$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$statusScript = "$root\src\pipeline\status_store.py"
$coreCombos = @("pls_ridge", "pls_xgboost", "xgboost_ridge", "xgboost_xgboost")
$diagnosticCombos = @("svr_ridge")

python "$statusScript" start --exp-id exp01_traditional --model grid --seed 42 --notes "feature_extraction + core + svr_diag running"
try {
  python "$root\src\pipeline\feature_extraction.py" --source-dir "$root\data\waveform_v3" --output-dir "$root\outputs\exp01_traditional"
  python "$root\src\pipeline\train_traditional.py" --data-dir "$root\outputs\exp01_traditional" --output-root "$root\outputs\exp01_traditional" --tag "formal_seed42_core" --seed 42 --stage-filter stable --max-workers 1 --n-jobs 2 --xgb-n-jobs 4 --combo-list $coreCombos
  python "$root\src\pipeline\train_traditional.py" --data-dir "$root\outputs\exp01_traditional" --output-root "$root\outputs\exp01_traditional" --tag "formal_seed42_svr_diag" --seed 42 --stage-filter stable --train-limit 5000 --test-limit 1500 --max-workers 1 --n-jobs 2 --xgb-n-jobs 1 --combo-list $diagnosticCombos
  python "$root\src\pipeline\aggregate.py"
  python "$statusScript" finish --exp-id exp01_traditional --model grid --seed 42 --summary-csv "$root\outputs\exp01_traditional\four_component_formal_seed42_core_grid_summary.csv" --notes "feature_extraction + core + svr_diag finished"
} catch {
  python "$statusScript" fail --exp-id exp01_traditional --model grid --seed 42 --notes $_.Exception.Message
  throw
}
