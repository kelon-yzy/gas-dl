# CNN1D 多模态 formal 一键训练脚本
# 用法：
#   .\run_cnn1d_multimodal_formal.ps1
#   .\run_cnn1d_multimodal_formal.ps1 --resume outputs/exp02_deep_e2e/v3_cnn1d_multimodal_seed42/last_checkpoint.pt

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $root

python src/pipeline/train_deep.py `
    --config configs/deep/slow_only_cnn1d_multimodal_formal.yaml `
    --no-ui `
    @ExtraArgs

exit $LASTEXITCODE
