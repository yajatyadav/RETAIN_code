# attempt 5：cudaMallocAsync 未生效，batch 64 超出单卡临时内存预算

## 现场

- 2026-08-20 08:19:22 CST，GPU 1 以 `10 MiB / 0%` 被选中；08:19:25 CST 严格 JAX GPU 门禁再次通过。
- 训练状态记录已明确写入 `allocator=cuda_malloc_async` 与 `fixed_bfc_preallocation=false`；真实 batch 64、基础权重、train state、AdamW 和 EMA 均初始化成功。
- 进入 `0/10000` 后仍在 `27.79 GiB` buffer 请求处 OOM，日志 allocator 名称仍为 `GPU_0_bfc`，说明当前 JAX/JAXlib 0.6.2 runtime 没有采用所请求的 async allocator。
- 没有 Step 0 指标和数字 checkpoint。

## 修正后的内存解释

- attempt 3 的 XLA 信息中，“无法把 memory use 从 `77.10 GiB` 降到目标约 `25.07 GiB`，只降到 `75.44 GiB`”描述的是完整 train step 在既有 train-state/参数预算之外仍无法满足的临时内存安排，不能简单解释为“75.44 GiB 总峰值只比 90% pool 大 0.16 GiB”。
- attempt 4 把固定 pool 从 90% 提到 95%，attempt 5 请求动态 allocator，均在完全相同的大块分配处失败；因此 batch 64 的完整 optimizer step 已被三次独立执行证明不适合当前单张 85,651 MiB RTX 6000D。
- GPU 选择、BF16 runtime、数据与 checkpoint 均已通过重复门禁，故继续提高固定 pool 或重复相同 batch 不再是合理实验。

## 缩小方案

- 按单 GPU 约束将七个 `retain_repro_*` 训练 config 的物理 batch 从论文的 64 缩小到 16；steps、LR schedule、AdamW、gradient clipping、EMA、mixture、seed、模型和数据不变。
- 不宣称 batch 16 与论文 batch 64 数值等价；该差异会写入每个 run status、研究状态、中文日志与最终报告。
- 选择 16 的依据是 batch-64 临时图约需 `75.44 GiB`，按 batch 近似线性缩放后 batch 16 约为其四分之一，低于 XLA 给出的约 25 GiB 临时预算；下一次真实首步是最终判据。
- 若 batch 16 仍不能产生有限 Step 0 指标，再缩小到 batch 8；不再重复无信息增益的 allocator 调整。
