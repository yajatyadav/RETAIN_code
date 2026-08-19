# 实验目录约定

每个 run 使用一个独立目录，至少包含：

- `command.txt`：实际执行命令；
- `environment.txt`：代码 commit、Python/JAX/CUDA/GPU 信息；
- `stdout.log`：完整标准输出与错误；
- `metrics.jsonl`：可解析训练或评测指标；
- `status.json`：开始/结束时间、退出码、checkpoint 与异常说明。

run 名称格式：`YYYYMMDD-HHMM_<stage>_<task>_<method>`。

自动化入口：

- `reproduction/scripts/run_reproduction_supervisor.py`：从输入逐文件校验、Orbax restore、全部 config smoke test，到训练和评测的总控入口；各阶段失败时保留现场和机器可读状态；
- `reproduction/scripts/run_training_pipeline.py`：等待空闲 GPU 后顺序执行 pretraining、Task-FT 和 coFT；
- `reproduction/scripts/run_evaluation_pipeline.py`：在同一张物理 GPU 上顺序加载 policy，执行 alpha sweep、ID、OOD validation/test 和 generalist evaluations。每个命令以 hash 标识，成功任务可断点跳过。

训练阶段会周期性只保留一个可恢复 checkpoint。每一阶段成功并确认最终 `params` 后，删除不再用于后续阶段的 `train_state`（模型当前参数与 Adam optimizer state），保留 EMA inference params、assets、日志、指标及 `storage-pruning.json`；这是为了使 7 个最终模型在共享盘预算内可长期保存，不改变训练或评测权重。
