$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
python "$root\src\pipeline\train_deep.py" --config "$root\configs\deep\fusion_formal.yaml" --epochs 1
python "$root\src\pipeline\train_deep.py" --config "$root\configs\deep\waveform_only_formal.yaml" --epochs 1
python "$root\src\pipeline\train_deep.py" --config "$root\configs\deep\slow_only_lstm_formal.yaml" --epochs 1
python "$root\src\pipeline\train_deep.py" --config "$root\configs\deep\slow_only_tcn_formal.yaml" --epochs 1
