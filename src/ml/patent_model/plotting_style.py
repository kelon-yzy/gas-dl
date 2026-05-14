"""图表统一样式与 profile 视觉常量。"""

from __future__ import annotations

import matplotlib.pyplot as plt


PROFILE_ORDER: tuple[str, ...] = ("raw_no_env", "raw_tph", "derived_env", "derived_env_mc_aug")

PROFILE_LABELS: dict[str, str] = {
    "raw_no_env": "原始无环境",
    "raw_tph": "原始温湿压",
    "derived_env": "派生环境补偿",
    "derived_env_mc_aug": "MC增强补偿",
}

PROFILE_COLORS: dict[str, str] = {
    "raw_no_env": "#2f6fbb",
    "raw_tph": "#c65f2f",
    "derived_env": "#3b8f5a",
    "derived_env_mc_aug": "#7a4fb3",
}


def setup_chinese_fonts() -> None:
    """统一 matplotlib 中文字体回退链，调用一次即可。"""

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
