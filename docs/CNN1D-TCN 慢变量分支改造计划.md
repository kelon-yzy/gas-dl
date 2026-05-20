# CNN1D-TCN 慢变量分支改造计划

## Summary
- 目标是隔离验证“慢变量被高维声学特征淹没”这一假设，不同时改 UW、sum constraint 或 compositional 输出。
- 新实验采用：`ultrasonic/fiber 1D-CNN embedding + slow MLP embedding` 先合并，再进入现有 TCN。
- TCN 后改为共享融合表示 + `out_dim` 个 target-specific 小 head，保留 4 维自由回归和当前 UW+sum loss。

## Key Changes
- 在 `CNN1DTCNFusionRegressor` 中新增可配置慢变量编码器：
  - 输入：`slow: [B, T, 8]`
  - 编码：`Linear(8, 32) -> GELU -> Linear(32, 64)`
  - 输出：`slow_emb: [B, T, 64]`
  - 融合输入从旧的 `[u_emb, f_emb, raw_slow] = 64+64+8` 改为新实验的 `[u_emb, f_emb, slow_emb] = 64+64+64`。
- 新增 target-specific heads：
  - TCN 输出仍做 `last + mean + max` 三池化。
  - 先经过共享融合层得到 `fusion_repr`。
  - 再用 `ModuleList` 生成每个组分一个小 head，最后 `torch.cat` 成 `[B, out_dim]`。
- 保持向后兼容：
  - 默认旧配置不强制改变现有模型行为。
  - 新增一个实验配置，例如 `configs/deep/slow_branch_cnn1d_tcn_fusion.yaml`，显式启用 `slow_encoder` 和 `target_specific_heads`。
  - 旧 checkpoint 与新配置架构不兼容，按现有 checkpoint 校验机制从头训练。

## Config Interface
- 在 `model` 配置中新增：
  - `slow_encoder.enabled: true`
  - `slow_encoder.hidden_dim: 32`
  - `slow_encoder.embedding_dim: 64`
  - `target_specific_heads: true`
  - `target_head_hidden_dim: 32`
- 新实验配置沿用当前 UW 训练参数：
  - `uncertainty_weighted` 不改。
  - `sum_constraint.weight` 不改。
  - `batch_size/lr/scheduler/early_stopping` 不改。
- 新 run 名建议：`v3_slow_branch_cnn1d_tcn_fusion_seed42`。

## Test Plan
- 单元测试：
  - slow encoder 开启时，第一个 TCN block 输入通道应为 `64 + 64 + 64 = 192`。
  - slow encoder 关闭时，旧行为仍为 `64 + 64 + 8 = 136`。
  - target-specific heads 输出 shape 为 `[B, out_dim]`，且支持 `out_dim=3/4`。
  - 单分支波形模式仍可用：只开 ultrasonic 或只开 fiber 时，TCN 输入通道正确。
- 回归测试：
  - 现有 `tests/test_cnn1d_tcn_fusion.py` 全部通过。
  - 训练相关测试至少跑 `tests/test_lr_scheduler.py` 和 checkpoint/config 兼容测试。
- 实验验收：
  - 训练完成后对比旧 run 的 CO2/N2 指标。
  - 重点看：CO2 R2、N2 R2、CO2/N2 pred/true std ratio、分箱误差、`mean_pred_sum`。
  - 成功标准：CO2 不再均值化，N2 的预测方差明显高于旧 run，且 H2/CH4 不出现大幅退化。

## Assumptions
- 本轮只做结构隔离实验，不加入 UW weight floor、不做 softmax×100、不用 `N2 = 100 - sum(前三项)`。
- 慢变量 MLP 是逐时间步编码，不先做时间池化；时间建模仍交给 TCN。
- target-specific heads 放在 TCN 池化之后，避免在时序阶段过早拆分任务。
