# step 9000 保存阶段主机 OOM 诊断

## 结论

attempt 6 不是模型数值发散或 GPU OOM，而是在 step 9000 的 Orbax 全量 checkpoint 保存阶段遭遇主机 OOM killer。进程以 137 退出，完整 step 8000 checkpoint 未受影响，可作为恢复点。

## 时间线（UTC）

- 06:38:42：训练推进到约 step 9000，此前 metrics 均为有限值。
- 06:38:47：Orbax 开始保存 `9000.orbax-checkpoint-tmp-32`。
- 06:39:08：`params` 写入停在约 12.0 GB；`train_state` 没有形成有效 payload。
- 07:14:37：systemd 记录 `session-389188.scope: A process of this unit has been killed by the OOM killer.`。
- 07:14:51：训练 pipeline 记录 pretraining return code 137，supervisor 随后按设计失败退出。

## 排除项

- 主机没有重启：boot time 仍为 2026-08-03 03:31 UTC。
- 同一诊断窗口没有 NVIDIA Xid、GPU fault、ext4 或 `/dev/sdb` I/O error。
- checkpoint inode 余量充足；故障后 `/shared` 仍有 182,942,547,968 bytes 可用。
- 故障发生前 loss、grad norm、parameter norm 均有限，没有数值异常证据。

## 不完整 checkpoint 现场

- 精确路径：`/shared/.cache/retain/checkpoints/retain_repro_pretrain/paper_final_hparams/9000.orbax-checkpoint-tmp-32`
- 文件数：18
- 表观 payload：12,014,142,618 bytes
- 已写 item：`assets`、`params` 的临时内容
- 缺失：可提交的 `train_state`、item/root finalize 与原子重命名
- 判定：不可恢复、不可作为 step 9000 结果

## 恢复策略

从已验证 step 8000 恢复 optimizer/train state；跳过 step 9000 中间全量保存；在最终 step 9999 仅原子保存 EMA/inference params 和 assets。最终 checkpoint 本来就会在 pipeline 成功后删除 optimizer state，因此该策略不改变任何下游输入，只提前避免无用途的终态 optimizer I/O 与主机内存峰值。

原始训练日志与状态保留在：

- `reproduction/experiments/retain_repro_pretrain__paper_final_hparams/stdout.log`
- `reproduction/experiments/retain_repro_pretrain__paper_final_hparams/status.json`
- `reproduction/experiments/supervisor-status.json`
