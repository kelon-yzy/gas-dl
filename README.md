# V3 正式实验

> 创建日期：2026-05-14
> 状态：v0.5 训练命令行可视化已落地（传统 ML 与深度训练均支持 CLI 阶段进度显示）
> 数据集进展：2026-05-19 上线 `waveform_v3_seedpath_formal`（30000 序列、4 路 split + extrapolation holdout），DL 端 `cnn1d_tcn_fusion` 与 ML 全套传统模型已完成适配，详见 [`docs/waveform_v3_seedpath_formal_适配说明.md`](docs/waveform_v3_seedpath_formal_适配说明.md)。

## 目的

基于 V3.1 双通道 waveform 仿真数据，对四组分（x_H2 / x_CH4 / x_CO2 / x_N2）混合气体检测做一次完整的 ML 与 DL 公平对比，作为正式交付的实验整理结果。

V3.1 数据集相对 V3.0 的核心变化是把原先单通道 waveform 升级为两条物理独立链路：

- 通道 1：超声对射波形（TOF 链路）
- 通道 2：光纤麦克风波形（声衰减 / 多次反射混响链路）

新项目与老仓库 `深度学习测试/` 完全独立：代码会物理拷贝到本仓，数据集由本仓仿真链路重新生成，不依赖外部运行路径。

## 三个验证目标

| 编号  | 目标                                      | 主指标                                        |
| --- | --------------------------------------- | ------------------------------------------ |
| G1  | ML 与 DL 对四组分的检测能力对比                     | macro_RMSE / macro_MRE / per-component R²  |
| G2  | 比较三种模态基学习器及其动态融合结果，识别当前数据级别最合适的传统 ML 方案 | macro_RMSE + 参数量 + 训练时长                    |
| G3  | 环境适应性：敏感度曲线 + 噪声鲁棒 + 工况泛化               | per-bin RMSE / degradation_ratio / 18域 std |

详细矩阵见 `docs/design.md`，双通道数据集设计见 `docs/数据集生成设计.md`。

## 数据集设计基线

V3.1 数据包目前有两份目录：

| 目录 | 序列数 | split 策略 | 用途 |
|---|---:|---|---|
| `data/waveform_v3/` | 10000 | 旧的 mixture group 随机切 | 历史对照，不再作为主线训练源 |
| `data/waveform_v3_seedpath_formal/` | 30000 | `stratified_group_by_mixture_id_with_extrapolation_holdout`（train/val/test/extrapolation 四路） | 正式主线训练源 |

两份数据集的张量字段一致：

| 项                  | 当前设计                                                 |
| ------------------ | ---------------------------------------------------- |
| dataset_version    | V3.1 dual-channel waveform                           |
| ultrasonic         | `int16 [N, 120, 1000]` + `scale [N, 120]`            |
| fiber_mic          | `int16 [N, 120, 2000]` + `scale [N, 120]`            |
| slow               | `float32 [N, 120, 8]`                                |
| labels             | `float32 [N, 4]`，顺序 `[x_H2, x_CH4, x_CO2, x_N2]`     |
| generation_seed    | `20260514`                                           |
| calibration_status | `pending`                                            |

`N=10000`（旧）或 `N=30000`（新）。两份数据集的 `mixture_id` 命名空间不同，不要混用 split 文件。

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
│   ├── waveform_v3/                  # V3.1 旧版（10000 序列），历史对照
│   └── waveform_v3_seedpath_formal/  # V3.1 正式版（30000 序列 + 4 路 split），主线训练源
├── experiments/               # 一键脚本（exp01-exp06）
├── outputs/                   # 实验结果
│   ├── STATUS.tsv             # 单一状态真源（每实验/seed 一行）
│   ├── exp01_traditional/     # G1: 传统 ML 基线 ✅ formal_seed42 (core + diag)
│   ├── exp02_deep_e2e/        # G1: DL 端到端基线
│   ├── exp03_full_training/   # G1: DL 全量训练（4 模型 × 200 epochs，进行中）
│   ├── exp04_adaptation/      # G3: 环境适应性评估（🆕 整合原 exp04+exp05）
│   │   ├── G3a_sensitivity/   # 环境单变量敏感度曲线（T/P/RH vs RMSE）
│   │   ├── G3b_perturbation/  # 高低噪声鲁棒性测试
│   │   └── G3c_holdout/       # 工况泛化测试（18 域留一）
│   ├── archive/               # 旧训练结果归档
│   │   ├── traditional_before_speed_opt_20260514_1800/  # 速度优化前旧结果
│   │   ├── traditional_smoke_20260516/                  # 8 个冒烟/临时测试目录
│   │   └── pre_adaptation_redesign_20260516/            # 旧 exp04/exp05（实验重设计前）
│   └── summary/               # 最终汇总表与论文图
└── docs/
    ├── design.md              # 实验设计（三目标 × 矩阵 × 验收）
    ├── 数据集生成设计.md       # V3.1 双通道数据集设计
    ├── 训练速度优化计划.md     # ML + DL 训练加速路线图与 smoke benchmark
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
- Windows 下若宿主终端不支持 ANSI/VT（例如部分“右键使用 PowerShell 运行”场景），`--ui` 会自动退回普通日志，不再因 UI 渲染直接闪退。
- 传统训练重点显示 `profile / combo / 阶段 / 已完成数 / 累计耗时 / 最近 macro_RMSE`。
- 深度训练重点显示 `run_name / epoch / train_loss / val_loss / val_macro_RMSE / early stopping 状态`。
- 非交互环境或关闭 UI 时，自动退回普通日志与 JSON 输出，不改结果文件格式。
- 训练速度优化路线、资源参数 smoke benchmark 与正式配置准入标准见 `docs/训练速度优化计划.md`。

## 深度学习训练曲线导出

- `src/pipeline/plot_deep_training_curves.py` 会递归扫描 `outputs/` 下所有包含 `train_log.csv` 的深度学习 run，并批量输出训练曲线。
- 默认输出到 `outputs/deep_training_curves/`，格式为 `PNG + SVG`。
- 每个 run 会生成一张四面板图：`train_loss / val_loss`、`val_macro_RMSE`、`val_macro_MAE`、可选 `lr`；同时额外生成一张全局 `val_macro_RMSE` 汇总图。
- 如果只想画单个 run，可直接把该 run 目录作为 `--root` 传入。

示例：

```powershell
# 批量扫描全部深度学习训练输出
python src/pipeline/plot_deep_training_curves.py

# 仅导出 v3_tcn_multimodal_seed42 的训练曲线
python src/pipeline/plot_deep_training_curves.py --root outputs/exp02_deep_e2e/v3_tcn_multimodal_seed42 --output-dir outputs/deep_training_curves/v3_tcn_multimodal_seed42
```

## 传统 ML 当前口径

- 正式主线使用 `paper_core` 组合：`svr_ridge`、`pls_ridge`、`xgboost_ridge`，统一输出 `acoustic / optical / thermal / fused` 四类结果。
- `svr_ridge` 不再是"单组分 RBF 对照"，而是三模态四输出 + Ridge 动态融合的正式基线之一。
- 每个 `summary.json` 现已记录 `fit_seconds`、`evaluate_seconds`、`write_seconds`、`total_seconds`、`n_jobs`、`xgb_n_jobs`，以及 `acoustic/optical/thermal/fused` 四类宏指标。
- 每个 `grid_summary.csv` 现按 `profile + combo + model_name` 展开，多出 `model_name ∈ {acoustic, optical, thermal, fused}` 维度。
- 本轮重跑前的旧传统训练结果已移动到 `outputs/archive/traditional_before_speed_opt_20260514_1800/`。
- 2026-05-16 清理：8 个冒烟/速度/UI 临时测试目录已归档到 `outputs/archive/traditional_smoke_20260516/`，保留最新正式结果 `outputs/exp01_traditional/`。
- **正式结果 (seed=42)**：`v3_raw_tph`（含温压湿环境特征）下 `xgboost_ridge` 和 `svr_ridge` 的 fused_macro_RMSE 分别为 0.63% 和 0.61%；`v3_raw_no_env` 下 `xgboost_ridge` 为 1.40%。环境特征对传统 ML 至关重要。
- **2026-05-19 新数据集适配**：`waveform_v3_seedpath_formal`（30000 序列）的特征表已落到 `outputs/exp01_traditional_seedpath/`（120000 样本）；`train_traditional.py` / `run_four_component_model_grid.py` / `train_patent_model.py` 已支持 `--split-dir` 透传，按 mixture_id 复用 DL 的 4 路 split。基于新数据集的传统 ML 正式结果待重跑，详见 [`docs/waveform_v3_seedpath_formal_适配说明.md`](docs/waveform_v3_seedpath_formal_适配说明.md)。

## 当前进度

| 阶段                     | 状态        | 说明                                                           |
| ---------------------- | --------- | ------------------------------------------------------------ |
| P0 骨架（目录 + 文档）         | ✅ v0.3 完成 | 目录扁平化，状态表就位，文档已同步到 V3.1 双通道数据设计                              |
| P0 代码拷贝 / 改造           | ⏳ 待执行     | 见 `docs/design.md` §7                                        |
| P0 数据集生成与验收            | ⏳ 待执行     | 重新生成双通道数据，并做方向性 / 质量检查，不再直接复用旧单通道数据包                         |
| P1 传统 ML 基线            | ✅ 完成      | core + diagnostic 口径已跑完，环境特征下 xgboost_ridge fused_RMSE=0.63% |
| P2 DL 端到端基线            | 🟡 进行中    | 全量训练 4 模型运行中 (exp03_full_training)                           |
| P3 模态与动态融合对比           | ⏳ 未启动     | 依赖 P2                                                        |
| P4 环境适应性评估（G3）         | 🟡 脚本就绪   | 整合为 exp04_adaptation (G3a敏感性曲线/G3b噪声鲁棒/G3c工况泛化)，脚本已写待执行      |
| P5 重复性检测（R1，多 seed）    | ⏳ 未启动     | 独立于主线；复用已有 seed42，补跑 52/62 (exp06)                           |
| P7 跨域微调（G3 Track C，可选） | ⏳ 可选      | 时间允许再做                                                       |
| P8 汇总 + 论文图            | ⏳ 未启动     | 依赖 P1-P6                                                     |

## G3 环境适应性评估设计

原 exp04（留一域评估）和 exp05（环境扰动）已整合为 **exp04_adaptation**，用更直观的方式回答"模型在不同环境下表现如何"：

| 子实验           | 做什么                       | 怎么评估                       | 输出                    |
| ------------- | ------------------------- | -------------------------- | --------------------- |
| **G3a 敏感度曲线** | 用已训练模型，按 T/P/RH 分箱统计 RMSE | 无需重训练，复用 `predictions.csv` | 9 张 PNG 曲线图 + CSV 数据表 |
| **G3b 噪声鲁棒性** | 对测试集注入 5 级传感器噪声           | 原有 exp05 脚本，测试环境补偿效果       | RMSE vs 噪声等级曲线        |
| **G3c 工况泛化**  | 18 个环境域逐一留出做测试            | 原有 exp04 脚本，评估跨域泛化         | 18 域 RMSE 均值±标准差      |

- G3a 不需要 GPU/重训练，秒级出结果，适合快速迭代查看趋势
- G3b 依赖 G3a 不依赖，可独立运行
- 运行方式：`powershell -File experiments\exp04_adaptation.ps1`

## 公平对比五同约束

| 约束    | 要求                                                                                  |
| ----- | ----------------------------------------------------------------------------------- |
| 同数据源  | 主线统一使用 `data/waveform_v3_seedpath_formal/`（30000 序列双通道）：`ultrasonic + fiber_mic + slow + y`；旧 `data/waveform_v3/`（10000 序列）保留作历史对照 |
| 同划分   | 锁定 `data/waveform_v3_seedpath_formal/splits/*.csv`（4 路：train/val/test/extrapolation），按 `mixture_id` 分组，使用 V3.1 生成 seed `20260514`；split 策略 `stratified_group_by_mixture_id_with_extrapolation_holdout` |
| 同标准化  | slow scaler 仅在 train 上拟合（已固化在 `scalers/scaler_slow_sequence.json`）；两路 waveform 用 `int16 × scale` 反量化，再用 train 统计做归一化 |
| 同随机基准 | 主线固定 seed=42；多 seed 只在 `exp06_reproducibility` 重复性检测中运行                             |
| 同指标   | macro_RMSE / macro_MAE / macro_MRE / per-component R² / sum_error；重复性检测额外报 seed_std |

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

# 3. 环境适应性评估（敏感度曲线 + 噪声鲁棒 + 工况泛化）
powershell -File experiments\exp04_adaptation.ps1

# 4. 模态与动态融合对比
powershell -File experiments\exp03_fusion_grid.ps1

# 5. 多 seed 重复性检测（独立于主线；seed42 已有则复用，默认补跑 52/62）
powershell -File experiments\exp06_reproducibility.ps1

# 传统 ML 单独调度示例：paper_core（默认交互终端自动显示 CLI 进度）
python src\pipeline\train_traditional.py --data-dir outputs\exp01_traditional --output-root outputs\exp01_traditional --tag formal_seed42_core --seed 42 --profiles v3_raw_no_env v3_raw_tph --combo-list svr_ridge pls_ridge xgboost_ridge --max-workers 1 --n-jobs 2 --xgb-n-jobs 4

# 传统 ML 新数据集调度示例：使用 waveform_v3_seedpath_formal 的 4 路 split（与 DL 同测试集）
python src\pipeline\train_traditional.py --data-dir outputs\exp01_traditional_seedpath --output-root outputs\exp01_traditional_seedpath --tag formal_seedpath --seed 42 --split-dir data\waveform_v3_seedpath_formal\splits --profiles v3_raw_no_env v3_raw_tph --combo-list svr_ridge pls_ridge xgboost_ridge --max-workers 4

# 特征表重产（新数据集 → 120000 样本，约 4 分钟）
python src\pipeline\feature_extraction.py --source-dir data\waveform_v3_seedpath_formal --output-dir outputs\exp01_traditional_seedpath

# 深度训练单独调度示例：关闭 CLI 界面
python src\pipeline\train_deep.py --config configs\deep\slow_only_tcn_formal.yaml --epochs 1 --no-ui

# 深度训练曲线导出：批量扫描 outputs
python src\pipeline\plot_deep_training_curves.py

# 深度训练曲线导出：只画单个 run
python src\pipeline\plot_deep_training_curves.py --root outputs\exp02_deep_e2e\v3_tcn_multimodal_seed42 --output-dir outputs\deep_training_curves\v3_tcn_multimodal_seed42

# 环境敏感度曲线分析：用已训练模型出 T/P/RH vs RMSE 趋势图
python src\pipeline\sensitivity_scan.py --predictions outputs\exp01_traditional\four_component_formal_seed42_core_grid_v3_raw_tph\xgboost_ridge\predictions.csv --condition-grid outputs\exp01_traditional\condition_grid_v1.csv --output-dir outputs\exp04_adaptation\G3a_sensitivity

# 深度重复性检测单独调度示例：覆盖 seed 与输出根目录
python src\pipeline\train_deep.py --config configs\deep\slow_only_tcn_formal.yaml --epochs 1 --seed 52 --output-root outputs\exp06_reproducibility\deep --no-ui

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
- **GPU 时长未估算**：主线先固定 seed=42 出结果；多 seed 重复性检测复用已有 seed42，主要补跑 52/62，避免阻塞主线。
- **RBF SVR 全量训练成本过高**：默认不再放入正式主网格，避免阻塞主线结果产出。

## 与老仓库的关系

- 老仓库路径：`../深度学习测试/`
- 代码来源（参考，不依赖）：
  - DL：`sequence-models/models/`, `data/`, `training/`
  - ML：`traditional-models/patent-sim-model/`
  - 数据生成：`simulation-data/scripts/`, `sim_common/`, `sim_v2/`
- 数据策略：不直接硬拷贝旧 `output_waveform_sequence/` 单通道包，而是复用其物理函数与工况采样逻辑，重新生成 V3.1 双通道数据集。

拷贝与改造完成后，本项目可独立运行，删除 `../深度学习测试/` 不影响本项目。
