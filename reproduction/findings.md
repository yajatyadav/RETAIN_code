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
14. `π0 base` 的服务器可见目录已按官方 GCS metadata 通过 24/24 对象的 size 与 MD5 校验，总计 12,014,416,199 bytes。并行 aria2/rsync 曾使旧 aria2 的 5 个文件句柄指向 deleted inodes；校验可见 payload 后已结束该无效进程，并把控制文件移入可恢复归档，权重内容未改动。
15. 服务器首次全量 SHA-256 正确检出 12 个表观尺寸完整但内容不一致的 LIBERO-90 shards，证明尺寸门禁不能替代内容校验。按固定 revision 做 rsync checksum 差分修复并定向复核后，353/353 个参考 payload、24,235,684,869 bytes 全部通过 SHA-256。
16. 三个 target smoke tests 最初各生成一个 `dataset_statistics_*.json`；后续全 config smoke 又为四个 pretraining suites 生成同类缓存，目前合计 7 个。它们不是固定 Hugging Face revision 的 payload。监督器保留这些训练缓存，但只以参考清单的 353 个路径作为数据内容门禁；未知类型的额外文件仍会导致失败。
17. `π0 base` 已成功完成真实 Orbax restore：50 个 leaves、3,238,048,528 个 float32 参数、内存展开 12,952,194,112 bytes，结构 SHA-256 为 `b061101d775178ee7709d97c4e8a1d5b68a63073febcd545fe6e6d1f05609dda`。全部 7 个 `retain_repro_*` configs 也已在 CPU 上读取真实 batch 并通过 shape、dtype 与有限值检查。
18. 首次获得空闲 GPU 2 后，pretraining 在 step 0 前完成了数据、LR 和 optimizer 初始化，但 JAX/XLA 编译报 `Unsupported conversion from bf16 to f16` 与 `Unsupported rounding mode for conversion`，exit code 为 134，未生成数值 checkpoint。服务器 GPU 是 compute capability 12.0（sm_120），而项目固定 JAX/JAXlib 0.5.0；故障发生于编译器/runtime 层，不支持据此判断训练配置或数据有误。
19. 为保持项目主环境可回退，JAX 兼容修复采用 `/shared/.cache/retain/jax-overlays/0.6.2` 隔离 overlay，而没有原位升级 `.venv`。overlay 固定 JAX/JAXlib/CUDA plugin 0.6.2、ml-dtypes 0.5.1 与 cuDNN 9.8.0.87；在 CPU backend 下，7/7 真实 batch、同一 `π0 base` Orbax restore（参数数目与结构哈希不变）及完整 pretrain train-state `eval_shape` 均已通过。实际 sm_120 BF16 编译仍需等空闲 GPU 门禁通过后才能宣告修复有效。
20. 第二次获得空闲 GPU 2 时，初版 preflight 因 CUDA libraries 不可见而回退到 CPU，但旧门禁只检查退出码，错误标为通过；完整训练随后加载主环境 cuDNN 9.7.1，与 JAX plugin build 的 9.8.0 不兼容并在 step 0 前退出。修复后 JAX plugin 所需的 CUDA wheels 已全部固定到 overlay，训练和评测均显式优先 overlay 动态库；12/12 关键库加载与 CPU BF16 smoke 已通过，门禁也已强化为必须只有一个可见设备且 `platform=gpu`。实际 GPU 编译仍待空闲卡验证。

## 待验证

- 10,000-step pretraining 的 loss 曲线与最终 checkpoint。
- 三任务 Task-FT、RETAIN、coFT 的 ID/OOD/generalist 成功率。
- 完整 CUDA 隔离修复后的 JAX 0.6.2 overlay 在 sm_120 上的严格 GPU BF16→FP16、BF16 GEMM 编译门禁，以及随后完整模型首步训练。

## 已知限制

- 当前服务器没有论文真机实验所需的 Franka 机械臂和现场相机，因此只执行 LIBERO 仿真部分。
- 公开仓库固定的 JAX 0.5.0 无法在本服务器 sm_120 上完成模型编译；主实验因此使用独立 JAX 0.6.2 runtime overlay。该调整只改变硬件编译/runtime，不改变论文模型、数据、优化器、步数或参数合并方法，并保留原环境与失败证据。
