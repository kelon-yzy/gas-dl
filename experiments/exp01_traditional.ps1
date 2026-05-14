$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$coreCombos = @("pls_ridge", "pls_xgboost", "xgboost_ridge", "xgboost_xgboost")
$diagnosticCombos = @("svr_ridge")

python "$root\src\pipeline\feature_extraction.py" --source-dir "$root\data\waveform_v3" --output-dir "$root\outputs\exp01_traditional"
foreach ($seed in 42, 52, 62) {
  python "$root\src\pipeline\train_traditional.py" --data-dir "$root\outputs\exp01_traditional" --output-root "$root\outputs\exp01_traditional" --tag "formal_seed${seed}_core" --seed $seed --max-workers 1 --n-jobs 2 --xgb-n-jobs 4 --combo-list $coreCombos
}
python "$root\src\pipeline\train_traditional.py" --data-dir "$root\outputs\exp01_traditional" --output-root "$root\outputs\exp01_traditional" --tag "formal_seed42_svr_diag" --seed 42 --train-limit 5000 --test-limit 1500 --max-workers 1 --n-jobs 2 --xgb-n-jobs 1 --combo-list $diagnosticCombos
python "$root\src\pipeline\aggregate.py"
