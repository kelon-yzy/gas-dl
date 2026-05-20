$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$extraSeeds = @(52, 62)
$coreCombos = @("pls_ridge", "pls_xgboost", "xgboost_ridge", "xgboost_xgboost")
$traditionalRoot = "$root\outputs\exp06_reproducibility\traditional"
$deepRoot = "$root\outputs\exp06_reproducibility\deep"
$deepRuns = @("v3_multimodal_fusion", "v3_waveform_only", "v3_lstm_slow", "v3_tcn_slow")

function Test-AllPaths {
  param([string[]] $Paths)
  foreach ($path in $Paths) {
    if (-not (Test-Path $path)) {
      return $false
    }
  }
  return $true
}

function Test-TraditionalSeedDone {
  param([int] $Seed)
  return Test-Path "$traditionalRoot\four_component_repro_seed${Seed}_core_grid_summary.csv"
}

function Test-DeepSeedDone {
  param([int] $Seed)
  $paths = @()
  foreach ($run in $deepRuns) {
    $paths += "$deepRoot\${run}_seed${Seed}\summary.json"
  }
  return Test-AllPaths $paths
}

function Invoke-TraditionalSeed {
  param([int] $Seed)
  python "$root\src\pipeline\train_traditional.py" --data-dir $traditionalRoot --output-root $traditionalRoot --tag "repro_seed${Seed}_core" --seed $Seed --max-workers 1 --n-jobs -1 --n-perturbations 10 --xgb-n-jobs 4 --combo-list $coreCombos
}

function Invoke-DeepSeed {
  param([int] $Seed)
  python "$root\src\pipeline\train_deep.py" --config "$root\configs\deep\fusion_formal.yaml" --epochs 1 --seed $Seed --output-root $deepRoot
  python "$root\src\pipeline\train_deep.py" --config "$root\configs\deep\waveform_only_formal.yaml" --epochs 1 --seed $Seed --output-root $deepRoot
  python "$root\src\pipeline\train_deep.py" --config "$root\configs\deep\slow_only_lstm_formal.yaml" --epochs 1 --seed $Seed --output-root $deepRoot
  python "$root\src\pipeline\train_deep.py" --config "$root\configs\deep\slow_only_tcn_formal.yaml" --epochs 1 --seed $Seed --output-root $deepRoot
}

python "$root\src\pipeline\feature_extraction.py" --source-dir "$root\data\waveform_v3" --output-dir $traditionalRoot

$mainTraditionalSeed42Done = Test-Path "$root\outputs\exp01_traditional\four_component_formal_seed42_core_grid_summary.csv"
$mainDeepSeed42Done = Test-AllPaths @(
  "$root\src\dl\outputs\exp02_deep_e2e\v3_multimodal_fusion_seed42\summary.json",
  "$root\src\dl\outputs\exp02_deep_e2e\v3_waveform_only_seed42\summary.json",
  "$root\src\dl\outputs\exp02_deep_e2e\v3_lstm_slow_seed42\summary.json",
  "$root\src\dl\outputs\exp02_deep_e2e\v3_tcn_slow_seed42\summary.json"
)

if (-not ($mainTraditionalSeed42Done -or (Test-TraditionalSeedDone 42))) {
  Invoke-TraditionalSeed 42
}
if (-not ($mainDeepSeed42Done -or (Test-DeepSeedDone 42))) {
  Invoke-DeepSeed 42
}

foreach ($seed in $extraSeeds) {
  if (-not (Test-TraditionalSeedDone $seed)) {
    Invoke-TraditionalSeed $seed
  }
  if (-not (Test-DeepSeedDone $seed)) {
    Invoke-DeepSeed $seed
  }
}

python "$root\src\pipeline\aggregate.py"
