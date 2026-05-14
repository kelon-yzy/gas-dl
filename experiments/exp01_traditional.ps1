$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
python "$root\src\pipeline\feature_extraction.py" --source-dir "$root\data\waveform_v3" --output-dir "$root\outputs\exp01_traditional"
foreach ($seed in 42, 52, 62) {
  python "$root\src\pipeline\train_traditional.py" --data-dir "$root\outputs\exp01_traditional" --output-root "$root\outputs\exp01_traditional" --tag "formal_seed$seed" --seed $seed --max-workers 1
}
python "$root\src\pipeline\aggregate.py"
