# FiLM 早期融合（E2）加速版一键训练脚本
# 启用：AMP + batch=48 + num_workers=6 + prefetch_factor=4 + last_checkpoint_every=3
# 预计每 epoch 时间相比 baseline 减少 35-50%
# 用法：
#   .\run_early_fusion_film_multimodal_formal_fast.ps1
#   .\run_early_fusion_film_multimodal_formal_fast.ps1 --no-ui
#   .\run_early_fusion_film_multimodal_formal_fast.ps1 --epochs 5    # smoke test
#   .\run_early_fusion_film_multimodal_formal_fast.ps1 --resume outputs/exp02_deep_e2e/v3_early_fusion_film_seed42_fast/last_checkpoint.pt

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $root

python src/pipeline/train_deep.py `
    --config configs/deep/slow_only_early_fusion_film_multimodal_formal_fast.yaml `
    --ui `
    @ExtraArgs

exit $LASTEXITCODE
