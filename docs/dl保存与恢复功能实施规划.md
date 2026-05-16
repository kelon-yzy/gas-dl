PLEASE IMPLEMENT THIS PLAN:

# 深度学习训练暂停、保存与恢复功能实施规划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` or equivalent step-by-step execution workflow. 当前阶段只输出规划，不修改代码。

**Goal:** 给深度学习训练加入中途暂停、checkpoint 保存、从 checkpoint 恢复训练的能力。

**Architecture:** 在现有 `src/dl/training/train.py` 的 epoch 级训练循环上增加完整训练状态 checkpoint。checkpoint 保存模型参数、优化器、AMP scaler、early stopping 状态、训练日志、当前 epoch、配置摘要和随机数状态；恢复时重建数据集、模型、loss、optimizer 后加载状态，从下一轮 epoch 继续训练。命令行入口 `src/pipeline/train_deep.py` 增加 resume/pause 相关参数。

**Tech Stack:** Python, PyTorch, pandas, pytest, existing `train_config()` / `train_deep.py` pipeline。

---

## Summary

当前代码只保存验证集最优模型 `best_model.pt`，只能用于最终推理或测试，不能恢复训练。新增功能后：

- 每个 epoch 后保存 `last_checkpoint.pt`；
- 验证集改进时继续保存 `best_model.pt`，同时可选保存 `best_checkpoint.pt`；
- 用户按 `Ctrl+C` 时，在当前 batch/epoch 结束处保存 `paused_checkpoint.pt` 并正常退出；
- 用户可用 `--resume <checkpoint>` 继续训练；
- 恢复后保留历史 `train_log.csv`，最终输出仍保持 `summary.json`、`component_metrics.csv`、`predictions.csv`、`config.json`。

---

## Key Interface Changes

### CLI 新增参数

修改 `src/pipeline/train_deep.py`：

```bash
python src\pipeline\train_deep.py --config configs\deep\fusion_formal.yaml --output-root outputs\exp02_deep_full --resume outputs\exp02_deep_full\v3_multimodal_fusion_seed42\last_checkpoint.pt --no-ui
```

新增参数：

- `--resume <path>`：从指定 checkpoint 恢复训练。
- `--checkpoint-every <N>`：每 N 个 epoch 保存一次 `epoch_XXXX.pt`，默认 `0` 表示只保存 `last_checkpoint.pt` 和 `best_checkpoint.pt`。
- `--no-resume-rng`：默认恢复 RNG 状态；传入后不恢复随机数状态，用于调试。
- `--stop-after-epoch <N>`：测试专用参数，训练到指定 epoch 后主动保存 checkpoint 并退出，不作为正式实验常用参数。

### Python API 变更

修改 `train_config()` 签名：

```python
def train_config(
    config: dict,
    epochs_override: int | None = None,
    resume_path: str | Path | None = None,
    checkpoint_every: int = 0,
    restore_rng: bool = True,
    stop_after_epoch: int | None = None,
) -> dict:
```

`train_one()` 同步透传这些参数。

---

## Checkpoint Design

checkpoint 文件使用 `torch.save()` 保存 dict，文件名采用：

- `last_checkpoint.pt`：每个 epoch 后覆盖保存。
- `best_checkpoint.pt`：验证集指标改进时覆盖保存。
- `paused_checkpoint.pt`：捕获 `KeyboardInterrupt` 或 `stop_after_epoch` 时保存。
- `epoch_0005.pt`：当 `checkpoint_every=5` 时保存周期快照。

checkpoint 内容固定为：

```python
{
    "format_version": 1,
    "status": "running" | "paused" | "completed",
    "epoch": epoch,
    "total_epochs": total_epochs,
    "config": config_to_write,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "amp_scaler_state_dict": scaler.state_dict(),
    "early_stopping": {
        "best": stopper.best,
        "bad_epochs": stopper.bad_epochs,
        "patience": stopper.patience,
        "mode": stopper.mode,
    },
    "log_rows": log_rows,
    "best_metric": stopper.best,
    "best_model_path": str(best_path),
    "rng_state": {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
    },
}
```

恢复规则：

- 加载 checkpoint 后从 `checkpoint["epoch"] + 1` 继续。
- `epochs_override` 仍表示最终目标 epoch 总数，不表示“再训练 N 轮”。
- 如果 checkpoint epoch 已经大于等于目标 epoch，直接进入测试评估阶段。
- 恢复时不重新覆盖历史 `log_rows`，而是在旧日志后继续追加。
- `best_model.pt` 仍只保存 `model.state_dict()`，保持已有输出兼容。

---

## Implementation Changes

### 训练状态保存/恢复

在 `src/dl/training/train.py` 增加小型 helper：

- `_checkpoint_payload(...)`：组装 checkpoint dict。
- `_save_checkpoint(path, payload)`：保存到临时文件再原子替换，避免半写入文件。
- `_load_checkpoint(path, device)`：加载 checkpoint。
- `_restore_early_stopping(stopper, state)`：恢复 `EarlyStopping` 状态。
- `_capture_rng_state()` / `_restore_rng_state()`：保存与恢复 RNG。

避免引入“猜测式保护逻辑”：只校验明确会导致恢复错误的字段，例如 `format_version`、模型名、输出维度、checkpoint 基本键是否存在。

### 训练循环调整

在 `train_config()` 中：

- 初始化 model、optimizer、scaler、stopper 后，如果传入 `resume_path`，加载 checkpoint。
- `start_epoch = checkpoint_epoch + 1`；非 resume 时为 `1`。
- 每个 epoch 完成验证后保存 `last_checkpoint.pt`。
- improved 时同时保存 `best_model.pt` 和 `best_checkpoint.pt`。
- 如果 `checkpoint_every > 0` 且 epoch 可整除，保存 `epoch_XXXX.pt`。
- 捕获 `KeyboardInterrupt`，保存 `paused_checkpoint.pt`，写入当前 `train_log.csv` 和 `config.json`，返回一个 summary-like dict，状态为 `"paused"`。
- 正常训练完成后，保存一个 `completed` 状态的 `last_checkpoint.pt`。

### 输出文件兼容

保留现有输出：

- `best_model.pt`
- `config.json`
- `summary.json`
- `component_metrics.csv`
- `train_log.csv`
- `predictions.csv`

新增输出：

- `last_checkpoint.pt`
- `best_checkpoint.pt`
- `paused_checkpoint.pt`
- 可选 `epoch_XXXX.pt`

`summary.json` 增加字段：

```json
{
  "training_status": "completed",
  "resumed_from": null,
  "last_checkpoint": ".../last_checkpoint.pt",
  "best_checkpoint": ".../best_checkpoint.pt"
}
```

暂停退出时不生成测试集 `predictions.csv`，只写：

- `config.json`
- `train_log.csv`
- `paused_checkpoint.pt`
- `summary.json`，其中 `training_status = "paused"`

---

## Test Plan

新增或扩展测试文件：`tests/test_deep_checkpoint_resume.py`。

测试场景：

- `test_checkpoint_written_after_epoch`

  - 用小型 synthetic Dataset 和轻量模型跑 1 epoch。
  - 断言 `last_checkpoint.pt` 存在。
  - 断言 checkpoint 包含 `model_state_dict`、`optimizer_state_dict`、`amp_scaler_state_dict`、`early_stopping`、`log_rows`、`epoch`。

- `test_resume_continues_from_next_epoch`

  - 第一次用 `stop_after_epoch=1` 保存 checkpoint。
  - 第二次用 `resume_path` 和 `epochs_override=2` 恢复。
  - 断言最终 `train_log.csv` 有 epoch 1 和 epoch 2。
  - 断言 `epochs_trained == 2`。

- `test_resume_preserves_existing_log_rows`

  - checkpoint 内放入已有 `log_rows`。
  - 恢复训练后确认没有覆盖旧日志，也没有重复 epoch。

- `test_pipeline_passes_resume_args`

  - mock `train_config()`。
  - 调用 `src/pipeline/train_deep.py --resume ... --checkpoint-every 5 --no-resume-rng`。
  - 断言 CLI 参数正确传入。

- `test_keyboard_interrupt_writes_paused_checkpoint`

  - mock `_train_one_epoch()` 在第一轮抛出 `KeyboardInterrupt`。
  - 断言 `paused_checkpoint.pt` 存在。
  - 断言 `summary["training_status"] == "paused"`。

运行命令：

```powershell
python -m pytest tests\test_deep_checkpoint_resume.py tests\test_train_deep_paths.py tests\test_dl_validation_pass.py
```

预期：全部通过。

---

## Assumptions

- 暂停语义默认使用 `Ctrl+C`，不额外实现交互式暂停按钮或后台监控文件。
- checkpoint 以 epoch 为粒度保存，不做 batch 级恢复；中断当前 epoch 时最多重跑该 epoch。
- `best_model.pt` 保持现有格式，只存模型权重，避免破坏已有分析脚本。
- 恢复训练要求使用同一模型配置和同一数据切分；如 checkpoint 模型名与当前 config 不一致，直接报错。
- Plan Markdown 推荐保存路径为 `docs/superpowers/plans/2026-05-16-deep-training-checkpoint-resume.md`，但当前规划模式不落盘写文件。
