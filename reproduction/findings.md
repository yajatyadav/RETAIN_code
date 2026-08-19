# 经验证结论与偏差记录

本文件只记录已经由日志、输出文件或代码检查支持的结论。尚未运行的预期结果不写成结论。

## 已验证

1. 仓库的 `linear_interpolation` 实现与论文 RETAIN 公式一致：finetuned checkpoint 权重为 `α`，pretrained checkpoint 权重为 `1-α`。
2. 仓库已有 pretraining normalization statistics，可供所有后续训练和推理统一复用。
3. 服务器 JAX、PyTorch、TensorFlow、LIBERO 与 EGL headless rendering 的基础自检已通过。
4. 论文最终超参数与仓库原有 `pi0_libero_pretrain` 开发 config 不一致；复现使用独立的 `retain_repro_*` configs。

## 待验证

- 官方数据文件完整性与 RLDS feature schema。
- 单批 dataloader 和 `π0 base` 权重加载。
- 10,000-step pretraining 的 loss 曲线与最终 checkpoint。
- 三任务 Task-FT、RETAIN、coFT 的 ID/OOD/generalist 成功率。

## 已知限制

- 当前服务器没有论文真机实验所需的 Franka 机械臂和现场相机，因此只执行 LIBERO 仿真部分。
