# FiLM 调制型早期融合（E2）多模态 formal 一键训练脚本
# 用法：
#   .\run_early_fusion_film_multimodal_formal.ps1
#   .\run_early_fusion_film_multimodal_formal.ps1 --no-ui
#   .\run_early_fusion_film_multimodal_formal.ps1 --epochs 20
#   .\run_early_fusion_film_multimodal_formal.ps1 --resume outputs/exp02_deep_e2e/v3_early_fusion_film_seed42/last_checkpoint.pt

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $root

python src/pipeline/train_deep.py `
    --config configs/deep/slow_only_early_fusion_film_multimodal_formal.yaml `
    --ui `
    @ExtraArgs

exit $LASTEXITCODE
