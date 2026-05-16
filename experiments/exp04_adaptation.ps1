# exp04_adaptation: 环境适应性评估
# ====================================
# G3a 敏感度曲线 — 用已训练模型按 T/P/RH 分组评估 RMSE 趋势
# G3b 噪声鲁棒性 — 环境传感器噪声注入测试（整合原 exp05）
# G3c 工况泛化   — 18 域留一测试（整合原 exp04）

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot

$outputRoot = "$root\outputs\exp04_adaptation"
$dataDir = "$root\outputs\exp01_traditional"
$bestPredictions = "$dataDir\four_component_formal_seed42_core_grid_v3_raw_tph\xgboost_ridge\predictions.csv"
$conditionGrid = "$dataDir\condition_grid_v1.csv"

New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
New-Item -ItemType Directory -Path "$outputRoot\G3a_sensitivity" -Force | Out-Null
New-Item -ItemType Directory -Path "$outputRoot\G3b_perturbation" -Force | Out-Null
New-Item -ItemType Directory -Path "$outputRoot\G3c_holdout" -Force | Out-Null

$startAll = Get-Date
"========================================"
"exp04_adaptation — 环境适应性评估"
"开始: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
"========================================"

# ── G3a 环境单变量敏感度曲线 ──
""
"[G3a] 环境单变量敏感度曲线分析..."
$g3aStart = Get-Date
python "$root\src\pipeline\sensitivity_scan.py" `
    --predictions $bestPredictions `
    --condition-grid $conditionGrid `
    --output-dir "$outputRoot\G3a_sensitivity"
if ($LASTEXITCODE -eq 0) {
    $g3aElapsed = (Get-Date) - $g3aStart
    "[G3a] 完成 ($($g3aElapsed.ToString('hh\:mm\:ss'))) — 输出: $outputRoot\G3a_sensitivity"
} else {
    "[G3a] 失败 (exit=$LASTEXITCODE)"
}

# ── G3b 环境噪声鲁棒性（整合原 exp05） ──
""
"[G3b] 环境噪声鲁棒性测试..."
$g3bStart = Get-Date
python "$root\src\ml\scripts\run_environment_compensation_experiment.py" `
    --raw-data-dir $dataDir `
    --env-data-dir $dataDir `
    --output-dir "$outputRoot\G3b_perturbation\compensation" `
    --component-mode four `
    --seed 42
if ($LASTEXITCODE -ne 0) {
    "[G3b] 环境补偿实验失败 (exit=$LASTEXITCODE)"
} else {
    python "$root\src\ml\scripts\run_environment_compensation_robustness.py" `
        --raw-data-dir $dataDir `
        --env-data-dir $dataDir `
        --output-dir "$outputRoot\G3b_perturbation\robustness" `
        --compensation-results "$outputRoot\G3b_perturbation\compensation\environment_compensation_summary.csv" `
        --component-mode four `
        --seed 42
    if ($LASTEXITCODE -eq 0) {
        $g3bElapsed = (Get-Date) - $g3bStart
        "[G3b] 完成 ($($g3bElapsed.ToString('hh\:mm\:ss'))) — 输出: $outputRoot\G3b_perturbation"
    } else {
        "[G3b] 鲁棒性测试失败 (exit=$LASTEXITCODE)"
    }
}

# ── G3c 工况泛化测试（整合原 exp04 domain_holdout） ──
""
"[G3c] 工况泛化测试 (18 域留一)..."
$g3cStart = Get-Date
python "$root\src\pipeline\domain_holdout.py" `
    --data-dir $dataDir `
    --output-root "$outputRoot\G3c_holdout" `
    --profile v3_raw_no_env `
    --combo-list pls_ridge xgboost_xgboost `
    --seed 42 `
    --n-jobs 2 `
    --xgb-n-jobs 4
if ($LASTEXITCODE -eq 0) {
    $g3cElapsed = (Get-Date) - $g3cStart
    "[G3c] 完成 ($($g3cElapsed.ToString('hh\:mm\:ss'))) — 输出: $outputRoot\G3c_holdout"
} else {
    "[G3c] 失败 (exit=$LASTEXITCODE)"
}

$totalElapsed = (Get-Date) - $startAll
""
"========================================"
"exp04_adaptation 全部结束: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
"总耗时: $($totalElapsed.ToString('hh\:mm\:ss'))"
"输出: $outputRoot"
"========================================"
