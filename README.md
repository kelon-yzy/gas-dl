# V3 正式实验

> 创建日期：2026-05-14
> 状态：v0.5 训练命令行可视化已落地（传统 ML 与深度训练均支持 CLI 阶段进度显示）

## 目的

基于 V3.1 双通道 waveform 仿真数据，对四组分（x_H2 / x_CH4 / x_CO2 / x_N2）混合气体检测做一次完整的 ML 与 DL 公平对比，作为正式交付的实验整理结果。

V3.1 数据集相对 V3.0 的核心变化是把原先单通道 waveform 升级为两条物理独立链路：

- 通道 1：超声对射波形（TOF 链路）
- 通道 2：光纤麦克风波形（声衰减 / 多次反射混响链路）

新项目与老仓库 `深度学习测试/` 完全独立：代码会物理拷贝到本仓，数据集由本仓仿真链路重新生成，不依赖外部运行路径。

## 三个验证目标

| 编号  | 目标                                  | 主指标                                       |
| --- | ----------------------------------- | ----------------------------------------- |
| G1  | ML 与 DL 对四组分的检测能力对比                 | macro_RMSE / macro_MRE / per-component R² |
| G2  | 多种融合策略下，识别当前数据级别最合适的方案              | macro_RMSE + 参数量 + 训练时长                   |
| G3  | 复杂工况适应：环境通道保留 + 物理一致工况增强（+ 可选跨工况微调） | domain_gap, degradation_ratio             |

详细矩阵见 `docs/design.md`，双通道数据集设计见 `docs/数据集生成设计.md`。

## 数据集设计基线

| 项                  | 当前设计                                                 |
| ------------------ | ---------------------------------------------------- |
| dataset_version    | V3.1 dual-channel waveform                           |
| ultrasonic         | `int16 [10000, 120, 1000]` + `scale [10000, 120]`    |
| fiber_mic          | `int16 [10000, 120, 2000]` + `scale [10000, 120]`    |
| slow               | `float32 [10000, 120, 8]`                            |
| labels             | `float32 [10000, 4]`，顺序 `[x_H2, x_CH4, x_CO2, x_N2]` |
| generation_seed    | `20260514`                                           |
| calibration_status | `pending`                                            |

## 目录结构

```
V3_正式实验/
├── README.md                  # 本文件
├── src/                       # 代码（第二步拷贝；3 层深）
│   ├── dl/                    # 深度学习链路（来自 sequence-models/）
│   ├── ml/                    # 传统 ML 链路（来自 traditional-models/patent-sim-model/）
│   ├── sim/                   # 数据生成与特征导出（来自 simulation-data/）
│   └── pipeline/              # 实验编排（status.py / train_* / domain_split / perturbation / aggregate）
├── configs/                   # YAML 配置
│   ├── paths.yaml             # 3 个根路径
│   ├── data/                  # 数据配置
│   ├── traditional/           # 传统 ML 配置
│   ├── deep/                  # DL 配置
│   └── robustness/            # 留一域 + 环境扰动配置
├── data/
│   └── waveform_v3/           # V3.1 双通道数据包（待生成，约 7.2 GB；缩短通道 2 窗口时约 4.4 GB）
├── experiments/               # 一键脚本（exp01-exp06）
├── outputs/                   # 实验结果
│   ├── STATUS.tsv             # 单一状态真源（每实验/seed 一行）
│   ├── exp01_traditional/     # G1: 传统 ML 基线（feature 表 + core/diagnostic 训练输出）
│   ├── exp02_deep_e2e/        # G1: DL 端到端基线
│   ├── exp03_fusion/          # G2: 融合策略对比
│   ├── exp04_domain/          # G3: 留一域评估
│   ├── exp05_robust/          # G3: 环境扰动注入
│   ├── exp06_finetune/        # G3: 跨工况微调（可选）
│   ├── archive/               # 旧训练结果归档（重跑前保留对照）
│   └── summary/               # 最终汇总表与论文图
└── docs/
    ├── design.md              # 实验设计（三目标 × 矩阵 × 验收）
    ├── 数据集生成设计.md       # V3.1 双通道数据集设计
    ├── data_card.md           # V3.1 双通道数据卡片
    ├── code_changelog.md      # 拷贝代码的本地修改记录（第二步建）
    ├── results.md             # 结论速查（实验后写）
    └── reproducibility.md     # 复现指南（代码拷贝后写）
```

## 代码原则

研究型实验项目，**不做防御性编程**：

- 不写 try/except；路径错、参数错、文件缺失，让它崩出栈，定位最快
- 不做默认值兜底；配置漏了就报错
- 不做参数边界检查；输入异常就让 numpy/torch 自己崩
- 文档与状态控制照常做，但代码只走 happy path

## 控制与反馈机制

| 机制           | 实现                                                                                 |
| ------------ | ---------------------------------------------------------------------------------- |
| 单一状态真源       | `outputs/STATUS.tsv` — 每实验每 seed 一行（跑前写 running，跑后覆盖）                              |
| 状态汇总命令       | `python src/pipeline/status.py` — 一行命令看完成数 / 失败列表                                  |
| 每实验一行 stdout | `[exp_id.model.seedN] OK macro_RMSE=X.XXX took=Ys` 或 `FAIL reason=... -> log path` |
| 实验结果统一表      | `outputs/summary/results.tsv` — 所有实验跑完 append（aggregate 强制）                        |
| 拷贝代码版本锁      | 文件头注释 `# Copied from ../深度学习测试/<path> on <date>`，本地改动记到 `docs/code_changelog.md`   |

## 训练命令行可视化

- `src/pipeline/train_traditional.py` 与 `src/pipeline/train_deep.py` 现已内置命令行进度界面。
- 默认在交互式终端自动开启；加 `--no-ui` 可以关闭，加 `--ui` 可以强制开启。
- 传统训练重点显示 `profile / combo / 阶段 / 已完成数 / 累计耗时 / 最近 macro_RMSE`。
- 深度训练重点显示 `run_name / epoch / train_loss / val_loss / val_macro_RMSE / early stopping 状态`。
- 非交互环境或关闭 UI 时，自动退回普通日志与 JSON 输出，不改结果文件格式。

## 传统 ML 当前口径

- 正式主线使用 `paper_core` 组合：`pls_ridge`、`pls_xgboost`、`xgboost_ridge`、`xgboost_xgboost`。
- `svr_*` 不再作为默认全量主网格，而是作为 `svr_diagnostic` 单独运行，默认仅单 seed 且限制样本量。
- 每个 `summary.json` 现已记录 `fit_seconds`、`evaluate_seconds`、`write_seconds`、`total_seconds`、`n_jobs`、`xgb_n_jobs`。
- 同一 branch 下多个 meta learner 会复用测试集预测缓存，grid summary 额外记录 `prediction_cache_reused`。
- 本轮重跑前的旧传统训练结果已移动到 `outputs/archive/traditional_before_speed_opt_20260514_1800/`。

## 当前进度

| 阶段                     | 状态              | 说明                                   |
| ---------------------- | --------------- | ------------------------------------ |
| P0 骨架（目录 + 文档）         | ✅ v0.3 完成       | 目录扁平化，状态表就位，文档已同步到 V3.1 双通道数据设计      |
| P0 代码拷贝 / 改造           | ⏳ 待执行           | 见 `docs/design.md` §7                |
| P0 数据集生成与验收            | ⏳ 待执行           | 重新生成双通道数据，并做方向性 / 质量检查，不再直接复用旧单通道数据包 |
| P1 传统 ML 基线            | 🟡 优化完成，正式重跑待执行 | core/diagnostic 口径、耗时字段与缓存复用已就位      |
| P2 DL 端到端基线            | ⏳ 未启动           | 依赖 P0                                |
| P3 融合策略对比              | ⏳ 未启动           | 依赖 P2                                |
| P4 留一域评估（G3 Track A）   | ⏳ 未启动           | 依赖 P1 + P2                           |
| P5 环境扰动（G3 Track B）    | ⏳ 未启动           | 依赖 P4                                |
| P6 跨域微调（G3 Track C，可选） | ⏳ 可选            | 时间允许再做                               |
| P7 汇总 + 论文图            | ⏳ 未启动           | 依赖 P1-P5                             |

## 公平对比五同约束

| 约束       | 要求                                                                              |
| -------- | ------------------------------------------------------------------------------- |
| 同数据源     | 全部使用 `data/waveform_v3/` 同一份双通道数据：`ultrasonic + fiber_mic + slow + y`           |
| 同划分      | 锁定 `data/waveform_v3/splits/*.csv`，按 `mixture_id` 分组，使用 V3.1 生成 seed `20260514` |
| 同标准化     | slow scaler 仅在 train 上拟合；两路 waveform 用 `int16 × scale` 反量化，再用 train 统计做归一化      |
| 同 seed 集 | 42 / 52 / 62（主比较），多 seed 报均值 + std                                              |
| 同指标      | macro_RMSE / macro_MAE / macro_MRE / per-component R² / sum_error / seed_std    |

## 运行流程（计划，第二步骨架完成后填充实现）

```powershell
cd V3_正式实验/

# 0. 生成并验收 V3.1 双通道数据集
python src/sim/scripts/generate_waveform_dataset.py
python src/sim/scripts/check_waveform_directionality.py

# 1. 公平 ML 基线（PowerShell）
powershell -File experiments\exp01_traditional.ps1

# 2. DL 端到端基线
powershell -File experiments\exp02_deep_e2e.ps1

# 3. 融合策略对比
powershell -File experiments\exp03_fusion_grid.ps1

# 4. 留一域评估（G3 Track A）
powershell -File experiments\exp04_domain_holdout.ps1

# 5. 环境扰动（G3 Track B）
powershell -File experiments\exp05_env_perturbation.ps1

# 传统 ML 单独调度示例：paper_core（默认交互终端自动显示 CLI 进度）
python src\pipeline\train_traditional.py --data-dir outputs\exp01_traditional --output-root outputs\exp01_traditional --tag formal_seed42_core --seed 42 --profiles v3_raw_no_env v3_raw_tph --combo-list pls_ridge pls_xgboost xgboost_ridge xgboost_xgboost --max-workers 1 --n-jobs 2 --xgb-n-jobs 4

# 传统 ML 单独调度示例：svr_diagnostic
python src\pipeline\train_traditional.py --data-dir outputs\exp01_traditional --output-root outputs\exp01_traditional --tag formal_seed42_svr_diag --seed 42 --profiles v3_raw_no_env v3_raw_tph --combo-list svr_ridge --train-limit 5000 --test-limit 1500 --max-workers 1 --n-jobs 2 --xgb-n-jobs 1

# 深度训练单独调度示例：关闭 CLI 界面
python src\pipeline\train_deep.py --config configs\deep\slow_only_tcn_formal.yaml --epochs 1 --no-ui

# 任意时刻看状态
python src/pipeline/status.py

# 汇总
python src/pipeline/aggregate.py
```

## 关键约束与风险

- **双通道链路仍是 calibration pending**：光纤麦克风位置、壁面反射系数 `R`、噪声标定都还没有硬件级校准，文档里只能声明“链路结构对齐”，不能声明“等价真实采集”。
- **数据包体积明显增大**：V3.1 默认约 7.2 GB；如果通道 2 从 10 ms 缩到 5 ms，可降到约 4.4 GB。
- **MRE 在低含量组分上不稳定**：CO2/N2 接近 0 时 MRE 容易爆炸，采用 SMAPE 与限定子集 MRE 双报。
- **留一域可能样本失衡**：N=10000 跨约 9 个域，极端工况域样本可能不足 1000，需要先看 condition_grid 分布再定分箱数。
- **GPU 时长未估算**：多 seed × 多配置可能需要 2-3 天 GPU，需要分批跑。
- **RBF SVR 全量训练成本过高**：默认不再放入正式主网格，避免阻塞主线结果产出。

## 与老仓库的关系

- 老仓库路径：`../深度学习测试/`
- 代码来源（参考，不依赖）：
  - DL：`sequence-models/models/`, `data/`, `training/`
  - ML：`traditional-models/patent-sim-model/`
  - 数据生成：`simulation-data/scripts/`, `sim_common/`, `sim_v2/`
- 数据策略：不直接硬拷贝旧 `output_waveform_sequence/` 单通道包，而是复用其物理函数与工况采样逻辑，重新生成 V3.1 双通道数据集。

拷贝与改造完成后，本项目可独立运行，删除 `../深度学习测试/` 不影响本项目。
