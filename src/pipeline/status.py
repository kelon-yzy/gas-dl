"""读 outputs/STATUS.tsv 输出实验状态汇总。

约定：每个实验脚本在跑前 append 一行 status=running，
跑完后用 finished/macro_RMSE/notes 覆盖该行。

用法：
    python src/pipeline/status.py
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

STATUS = Path(__file__).resolve().parents[2] / "outputs" / "STATUS.tsv"

rows = list(csv.DictReader(STATUS.open(encoding="utf-8"), delimiter="\t"))
counts = Counter(r["status"] for r in rows)

print(f"Total: {len(rows)}")
for status, n in counts.items():
    print(f"  {status}: {n}")

for r in rows:
    if r["status"] == "failed":
        print(f"[FAIL] {r['exp_id']}.{r['model']}.seed{r['seed']}: {r['notes']}")
