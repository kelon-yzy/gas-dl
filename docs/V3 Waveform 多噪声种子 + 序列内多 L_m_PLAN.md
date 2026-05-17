# V3 Waveform 多噪声种子 + 序列内多 L_m 方案

## Summary

- 在 `generate_waveform_dataset.py` 中把 `--sequence-count` 定义为“基础条件数”，默认每个基础条件复制 `3` 条噪声种子序列。
- 多 `L_m` 采用“序列内扫描”，默认在 `steady` 阶段按 `0.2/0.6/1.0/1.4m` 切换；保留现有 `5ms/10ms` 波形窗口，因此不使用 `1.8m`。
- 默认 `10000` 基础条件会生成 `30000` 条序列；多路径增加单条序列内部的时序变化，不再额外乘路径数。

## Key Changes

- 新增生成参数：
  - `noise_seed_count=3`
  - `multi_path_phase="steady"`，可选 `"off" / "baseline" / "steady"`
- 条件构造逻辑改为两层：
  - 先生成并过滤 `base_condition_count`
  - 再为每个基础条件复制 `noise_seed_count` 条序列，赋 `sequence_id`、`base_condition_id`、`noise_seed_index`、`noise_seed`
- `condition_grid_sequence.csv` 增加 `base_condition_id`、`noise_seed_index`、`noise_seed`、`multi_path_phase`，用于追溯同一条件下的多噪声序列。
- 每个基础条件使用固定 `condition_seed` 生成共享 baseline/steady 目标值；每条复制序列使用自己的 `noise_seed` 生成动态响应、random walk、drift 和波形噪声。
- V3 专用多路径表设为 `WAVEFORM_PATH_LMS = (0.2, 0.6, 1.0, 1.4)`，不修改 V2 共享的 `BASELINE_PATH_LMS`。
- `slow` 中的 `L_m` / `piston_position_m` 与当前相位路径一致；超声和光纤麦克风波形也使用同一 timestep 的当前 `L_m`。
- `metadata/waveform_v3_spec.json`、README、quality summary 记录 `base_condition_count`、最终 `sequence_count`、`noise_seed_count`、`multi_path_phase` 和 `path_lms`。

## Test Plan

- 新增小规模生成测试：`sequence_count=4`、`noise_seed_count=3`、`timesteps=20` 时输出 12 条序列，所有数组首维一致。
- 验证同一 `base_condition_id` 下标签、T/P/H、稳态目标一致，但 waveform 和 slow 动态轨迹不同。
- 验证 `steady` 多路径时，steady 段 `L_m` 只出现 `0.2/0.6/1.0/1.4`，非扫描阶段保持 `L_m_base`。
- 验证 split 仍按 `mixture_id` 分组，同一配方及其噪声复制不会跨 train/val/test。
- 运行现有 waveform dataset runtime 测试，确认读取器仍从 metadata shape 读取波形长度，不依赖固定序列数。

## Assumptions

- 按用户选择，保留现有波形窗口；因此 V3 多路径默认不包含 `1.8m`。
- 默认目标是增强 DL 时序多样性，而不是把多 `L_m` 作为额外独立样本扩增。
- 预计数据体积约为当前 10,000 序列版本的 3 倍；使用现有 memmap 存储路径生成。
