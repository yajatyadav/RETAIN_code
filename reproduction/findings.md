# 经验证结论与偏差记录

本文件只记录已经由日志、输出文件或代码检查支持的结论。尚未运行的预期结果不写成结论。

## 已验证

1. 仓库的 `linear_interpolation` 实现与论文 RETAIN 公式一致：finetuned checkpoint 权重为 `α`，pretrained checkpoint 权重为 `1-α`。
2. 仓库已有 pretraining normalization statistics，可供所有后续训练和推理统一复用。
3. 服务器 JAX、PyTorch、TensorFlow、LIBERO 与 EGL headless rendering 的基础自检已通过。
4. 论文最终超参数与仓库原有 `pi0_libero_pretrain` 开发 config 不一致；复现使用独立的 `retain_repro_*` configs。
5. 作者公开的 filtered target datasets 实际包含 stove 41、mugs 38、basket 43 条成功 trajectories；对应 10,866、9,807、11,494 transitions。
6. pretraining 的公开 reduced 数据计数与作者 config 中用于 mixture weight 的旧计数不一致。主实验选择保留作者 config 权重；该选择及两组计数已完整记录，避免把它隐藏为实现细节。
7. 固定 revision 的 7 个训练数据目录共 353 个有效文件、24,235,684,869 bytes，本地逐文件 SHA-256 已完成。
8. 论文正文把 proprioceptive state 概括为 7D，但公开 RLDS schema 和官方 LIBERO evaluation code 实际使用 8D state（6D EEF pose + 2D gripper qpos）；action 为 7D。复现以公开可执行材料的 8D/7D 为准。
9. 公开代码的 RLDS registry 遗漏 mugs 和 basket 两个 target dataset，导致原始代码无法读取这两项公开数据。补齐与 stove 一致的 schema registry 后，三个 target config 的真实单批 transform 均通过 shape、dtype 与有限值检查。
10. 官方 PaliGemma tokenizer 已在服务器按 GCS 对象元数据校验：4,264,023 bytes，MD5 `1420adc9856720a559e8a87284b195e2`。
11. 作者 loader 在未提供 `absolute_action_mask` 时把所有动作维视为 relative action；本复现遵循该可执行行为，并保留 warning 作为审计证据。
12. 服务器首轮传输精确完成 271/353 个文件后，aria2 被不可达的 Hugging Face JSON 元数据 URL 占据并降为 0 B/s；独立尺寸审计确认 21 个 JSON 元数据与三个 target datasets 均已完整，82 个未完成项全部是 TFRecord。基于该清单刷新 CDN 签名 URL 后，续传恢复到约 3–4 MiB/s，既有 partial 未被删除。
13. aria2 中断后会遗留 control sidecars，其中部分对应 payload 已达到参考尺寸；因此 sidecar 数量不能作为数据正确性的最终判据。监督器现以固定 payload 清单的尺寸齐备和传输进程结束作为进入 SHA-256 的条件，SHA-256 仍是内容完整性的最终门禁；sidecars 仅作诊断记录且不进入数据清单。

## 待验证

- 服务器端数据 SHA-256 与本地清单的一致性。
- 117-task / coFT mixture 的完整单批 dataloader，以及 `π0 base` 权重加载。
- 10,000-step pretraining 的 loss 曲线与最终 checkpoint。
- 三任务 Task-FT、RETAIN、coFT 的 ID/OOD/generalist 成功率。

## 已知限制

- 当前服务器没有论文真机实验所需的 Franka 机械臂和现场相机，因此只执行 LIBERO 仿真部分。
