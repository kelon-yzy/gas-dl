# 深度学习模型全量训练脚本
# 用法：在项目根目录 PowerShell 中执行 .\run_all_training.ps1
# Ctrl+C 可安全中断当前模型，已完成的模型 checkpoint 会保留

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $root

$outputRoot = "outputs/exp03_full_training"
$logFile = "$outputRoot/training_log.txt"

New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

$models = @(
    @{Name="TCN (纯慢变量)";           Config="configs/deep/slow_only_tcn_formal.yaml"},
    @{Name="LSTM (纯慢变量)";          Config="configs/deep/slow_only_lstm_formal.yaml"},
    @{Name="MultimodalFusionV3 (纯多模态)"; Config="configs/deep/waveform_only_formal.yaml"},
    @{Name="MultimodalFusionV3 (全融合+AMP)"; Config="configs/deep/fusion_formal.yaml"}
)

$total = $models.Count
$startAll = Get-Date
"========================================" | Tee-Object -FilePath $logFile -Append
"训练开始: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Tee-Object -FilePath $logFile -Append
"模型总数: $total" | Tee-Object -FilePath $logFile -Append
"输出目录: $outputRoot" | Tee-Object -FilePath $logFile -Append
"========================================" | Tee-Object -FilePath $logFile -Append

for ($i = 0; $i -lt $total; $i++) {
    $model = $models[$i]
    $idx = $i + 1
    $startTime = Get-Date
    ""
    "========================================" | Tee-Object -FilePath $logFile -Append
    "[$idx/$total] 开始训练: $($model.Name)" | Tee-Object -FilePath $logFile -Append
    "配置: $($model.Config)" | Tee-Object -FilePath $logFile -Append
    "开始时间: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Tee-Object -FilePath $logFile -Append
    "========================================" | Tee-Object -FilePath $logFile -Append

    $exitCode = 0
    python src/pipeline/train_deep.py `
        --config $model.Config `
        --output-root $outputRoot `
        --no-ui `
        2>&1 | Tee-Object -FilePath $logFile -Append
    $exitCode = $LASTEXITCODE

    $elapsed = (Get-Date) - $startTime
    if ($exitCode -eq 0) {
        "[$idx/$total] $($model.Name) 训练完成 ($($elapsed.ToString('hh\:mm\:ss')))" | Tee-Object -FilePath $logFile -Append
    } else {
        "[$idx/$total] $($model.Name) 训练失败，退出码: $exitCode" | Tee-Object -FilePath $logFile -Append
    }
}

$totalElapsed = (Get-Date) - $startAll
""
"========================================" | Tee-Object -FilePath $logFile -Append
"训练全部结束: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Tee-Object -FilePath $logFile -Append
"总耗时: $($totalElapsed.ToString('hh\:mm\:ss'))" | Tee-Object -FilePath $logFile -Append
"输出目录: $outputRoot" | Tee-Object -FilePath $logFile -Append
"========================================" | Tee-Object -FilePath $logFile -Append
