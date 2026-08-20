# step 9000 主机 OOM 恢复协议

- 协议 ID：`RETAIN-GPU-20260819-001-R1`
- 锁定时间：2026-08-20 15:56 CST
- 类型：故障恢复（confirmatory recovery）
- 单卡约束：最多使用一张 GPU

## 已有证据

1. attempt 6 已连续训练到 step 9000 保存入口；step 8000 checkpoint 已完成原子提交并能被 Orbax 发现。
2. step 9000 临时目录只有 `assets` 和约 12.0 GB 的 `params`，没有完整 `train_state`，因此不是可恢复 checkpoint。
3. systemd 日志在 2026-08-20 07:14:37 UTC 明确记录 `session-389188.scope` 中的进程被 OOM killer 终止；同一窗口没有 GPU Xid、磁盘 I/O error 或主机重启证据。
4. 训练进程退出码为 137；step 8000 保持完整，GPU 1 已释放。

## 恢复假设与预测

假设：OOM 高峰来自 Orbax 在保存完整 optimizer/train state 时的主机内存压力。若从 step 8000 恢复、取消 step 9000 的中间全量保存，并在最终 step 9999 只持久化后续 Task-FT、参数合并与评测所需的 inference/EMA params，则可完成 10,000-step pretraining，同时显著降低最终保存的主机内存与共享盘峰值。

预测：

- Orbax 从 step 8000 恢复出的 `train_state.step` 与 optimizer state 完整；
- 训练继续生成 step 8000 之后的有限值 metrics，并运行到 step 9999；
- 最终 checkpoint 包含 `assets` 与 `params`，不保留 optimizer state；
- 后续三个 Task-FT 和三个 coFT 阶段可直接读取最终 `params`；
- 全流程始终只占用一张 GPU。

## 受控改动

1. pretraining 的 operational `save_interval` 从 1,000 调整为 10,000；由于 loop 仍强制在最后一步保存，因此只跳过 step 9000 中间恢复点。
2. 新增显式 `save_final_params_only` 选项；仅最终 checkpoint 去除 optimizer state。训练中用于恢复的 step 8000 checkpoint 不改动。
3. 精确归档 step 9000 OOM 现场元数据后，清理不可恢复的临时 checkpoint，以避免重复运行时磁盘余量跌破安全线。

## 不变项

模型、数据 revision、数据 mixture、batch 16 单卡缩小协议、seed、10,000 总步数、LR schedule、AdamW、gradient clipping、EMA、JAX/CUDA overlay 均不变。

## 成功门禁

1. 严格 GPU preflight 仍只看到一个 `platform=gpu` 设备；
2. 日志确认从 step 8000 恢复，而不是从头训练；
3. step 9999 checkpoint 原子提交，`params` 可完整 restore，所有数组有限；
4. metrics 覆盖至最后一个应记录的 step，全部有限；
5. 进入 Task-FT 前先完成最终 checkpoint 审计与中文日志更新。
