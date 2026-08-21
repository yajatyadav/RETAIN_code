# attempt 4：95% BFC 显存池碎片化

## 现场

- 2026-08-20 07:57:46 CST，双阈值门禁选择 GPU 1；选中时仅占用 `10 MiB`、利用率 `0%`。
- 07:57:50 CST，严格门禁再次在唯一可见的 NVIDIA RTX 6000D 上通过：JAX/JAXlib/CUDA plugin 均为 `0.6.2`，`platform=gpu`，BF16→FP16 和 BF16 GEMM 均成功。
- 完整训练使用论文 batch 64、10,000 steps、原 LR/AdamW/EMA/mixture/seed，以及 attempt 3 后设置的 95% BFC 预分配池与 TensorFlow memory growth。
- 数据 batch、基础权重、train state、optimizer 和 EMA 初始化成功；进入 `0/10000` 后仍在申请同一个 `27.79 GiB` buffer 时以 `RESOURCE_EXHAUSTED` 退出。没有 Step 0 指标和数字 checkpoint。

## 诊断

- attempt 3 的 90% BFC pool 约为 `75.28 GiB`；attempt 4 提高到 95% 后约为 `79.46 GiB`，但失败请求和错误位置完全相同。
- BFC allocation map 显示已用区间被多个空闲区间分隔，无法提供所需的大块连续分配；错误本身也明确建议在 memory fragmentation 情况下尝试 `TF_GPU_ALLOCATOR=cuda_malloc_async`。
- GPU 1 在启动前为空闲卡，失败后恢复到 `10 MiB`，没有证据表明是并发任务抢占导致。
- 结论：95% 固定 BFC arena 增加总容量仍不能解决首步的大块连续分配，当前证据支持 allocator fragmentation，而不是数据、论文超参数或 CUDA runtime 故障。

## 修复与验证标准

- 按 JAX 官方 GPU memory allocation 文档，将训练 allocator 改为 `cuda_malloc_async`；取消固定 BFC fraction/preallocation，让 CUDA memory pool 按需增长。文档：<https://docs.jax.dev/en/latest/gpu_memory_allocation.html>。
- RLDS TensorFlow loader 已在 `dataset.py` 中执行 `tf.config.set_visible_devices([], "GPU")`，继续保留 `TF_FORCE_GPU_ALLOW_GROWTH=true` 作为防御性设置。
- 模型、batch 64、steps、LR、AdamW、gradient clipping、EMA、mixture、seed 和数据均不改变。
- 下一次必须重新通过严格 GPU 门禁，并至少产生有限的 Step 0 loss/gradient metrics；若仍失败，再转入 microbatch + gradient accumulation 的单卡等效全局 batch 64 方案。
