# Attempt 3：严格 GPU 门禁通过后的 batch-64 首步显存不足

## 结论

本次失败证明完整 CUDA 隔离修复有效，但论文设置的单步 `batch=64` 在默认训练显存池中差约 0.16 GiB，导致第一个 optimizer step 执行前 OOM。失败不是 JAX backend 回退、cuDNN 冲突、数据错误或数值异常。

## 时间线与证据

- 2026-08-20 06:46:56 CST，GPU 2 达到 `582 MiB / 0%`，被双阈值门禁选中。
- 06:47:00 CST，严格 preflight 在唯一可见的 `NVIDIA RTX 6000D` 上通过：JAX/JAXlib/CUDA plugin 均为 `0.6.2`，`platform=gpu`，BF16→FP16 与 BF16 GEMM 均实际编译执行。
- 完整训练随后成功完成真实数据读取、基础权重加载、train state 与 AdamW/EMA 初始化，并进入 `0/10000` 的首个训练 step 编译与执行。
- XLA 报告训练 step 的峰值从 `77.10 GiB` rematerialize 至 `75.44 GiB`；当前 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90` 在 85,651 MiB GPU 上只建立约 `75.28 GiB` 的池，低约 `0.16 GiB`。
- 执行时申请 `27.79 GiB` buffer 失败，最终错误为 `RESOURCE_EXHAUSTED: Out of memory`。训练仍早于 Step 0 指标和数字 checkpoint，checkpoint 目录没有可恢复 step。

## 修复

- 保持模型、全局 batch 64、10,000 optimizer steps、LR schedule、AdamW、gradient clipping、EMA、数据 mixture 与随机种子不变。
- 仅把训练进程的 JAX 预分配比例从 `0.90` 提高到 `0.95`：对应约 `79.46 GiB`，比已编译峰值多约 `4.02 GiB`，同时仍为 CUDA context 留出余量。
- 设置 `TF_FORCE_GPU_ALLOW_GROWTH=true`，避免 RLDS/TensorFlow 输入管线一次性占用训练卡的剩余显存。
- 下一次获得空闲 GPU 后仍会重跑严格 GPU preflight，再启动完整训练。若 95% 池仍不足，才进入 microbatch/gradient accumulation 的单卡缩小方案；当前不提前改变论文优化协议。

## 现场文件

- `jax-gpu-preflight.json`：真实 GPU runtime 成功证据。
- `stdout.log`：完整训练输出与 XLA memory estimate/OOM traceback。
- `status.json`、`supervisor-status.json`：训练阶段和总控失败状态。
- `command.txt`、`run_training_pipeline.py`、`training-config.py`：失败时的可执行命令和代码快照。
