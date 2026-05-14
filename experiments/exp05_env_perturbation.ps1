$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
python "$root\src\ml\scripts\run_environment_compensation_experiment.py" --raw-data-dir "$root\outputs\exp01_traditional" --env-data-dir "$root\outputs\exp01_traditional" --output-dir "$root\outputs\exp05_robust\compensation" --component-mode four --seed 42
python "$root\src\ml\scripts\run_environment_compensation_robustness.py" --raw-data-dir "$root\outputs\exp01_traditional" --env-data-dir "$root\outputs\exp01_traditional" --output-dir "$root\outputs\exp05_robust\robustness" --compensation-results "$root\outputs\exp05_robust\compensation\environment_compensation_summary.csv" --component-mode four --seed 42
