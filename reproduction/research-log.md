# RETAIN 复现实验日志

## 2026-08-19｜RETAIN-GPU-20260819-001｜协议与环境准备

- **目标**：在服务器单张 GPU 上复现论文的 LIBERO 仿真实验，并把结果与中文记录保存在项目工作区。
- **代码基线**：`0bbc6cf`；保留项目中已有的本地环境配置改动，不纳入本实验协议提交。
- **服务器**：8 × NVIDIA RTX 6000D（每卡约 85.7 GB）；本实验最多使用其中 1 张。
- **资源检查**：首次检查时 8 张 GPU 均有其他任务占用，因此当前仅进行 CPU/网络/磁盘侧准备，不抢占现有进程。
- **磁盘策略**：项目根分区剩余约 68 GB，共享盘剩余约 294 GB。数据和 checkpoint 放到 `/shared/.cache/retain`；项目目录保存日志、指标、视频和清单。
- **论文版本**：arXiv:2512.08333v3。最终版超参数与仓库中早期开发 config 不一致，已新增独立 `retain_repro_*` configs，避免改写历史 config。
- **归一化**：复用仓库随代码提供的 `assets/pi0_libero_pretrain/norm_stats.json`，所有 Task-FT / coFT / RETAIN 都沿用 pretraining statistics，符合论文设置。
- **当前阻塞**：官方 Hugging Face 数据在服务器侧域名解析异常；改由本地固定 revision 下载，再传到服务器。Google Cloud Storage 可由服务器直连，用于获取 `pi0_base`。

### 已锁定的训练超参数

| 阶段 | Batch | Steps | LR | Warmup | Decay | AdamW | Gradient clip |
|---|---:|---:|---:|---:|---:|---|---:|
| 117-task pretraining | 64 | 10,000 | peak `2.5e-5`, end `2.5e-6` | 1,000 | 30,000 | β₁=0.9, β₂=0.95, ε=1e-8, wd=1e-10 | 1.0 |
| Task-FT stove | 64 | 500 | 同上 | 同上 | 同上 | 同上 | 1.0 |
| Task-FT mugs | 64 | 1,000 | 同上 | 同上 | 同上 | 同上 | 1.0 |
| Task-FT basket | 64 | 500 | 同上 | 同上 | 同上 | 同上 | 1.0 |
| coFT（每任务） | 64 | 1,000 | 同上 | 同上 | 同上 | 同上 | 1.0 |

注：为控制共享盘占用，训练仅保留最新 checkpoint；训练轨迹和最终参数不因此改变。

## 2026-08-19 22:42 CST｜公开数据元数据审计

- 作者公开数据固定到 revision `d15edfa89167e6e7230be3e85eb7391be2fa3134`，本地下载完成，共 354 个仓库对象。
- 三个目标任务过滤后的 trajectories / transitions 分别为：stove `41 / 10,866`，mugs `38 / 9,807`，basket `43 / 11,494`。这比论文正文的概括性表述更精确，后续以公开文件元数据为准。
- 发现 pretraining 采样计数的可复现性细节：公开 `*_reduced` 元数据为 goal `39,799`、object `53,739`、spatial `42,391`、LIBERO-90 `567,494` transitions；作者训练 config 的采样权重常量则为 `52,042 / 66,984 / 52,970 / 567,494`，对应失败轨迹过滤前的 `*_no_noops` 计数。
- 论文只规定 117-task mixture，没有进一步公开 suite 内采样公式。为最大限度复现作者实际可执行代码，主实验保留 config 中的原始采样权重，并把 reduced 元数据差异明确记为复现 caveat；资源允许时可补做“按公开 reduced counts 重算权重”的敏感性实验。
- `π0 base` 已从官方 GCS 下载；按对象元数据完成 24/24 size 与 MD5 校验，总计 `12,014,416,199` bytes。数据与权重正在并行传往服务器。

## 2026-08-19 22:54 CST｜输入文件逐项校验与 schema 审计

- 固定 revision 的 7 个 RLDS 数据目录已在本地完成逐文件 SHA-256：实验有效载荷共 `353` 个文件、`24,235,684,869` bytes，全部可读。仓库对象数 `354` 还包含根目录 `.gitattributes`，不属于训练输入。
- 由于服务器无法稳定解析 Hugging Face Hub 域名，数据改用本地解析出的公开、限时 CDN URL 由服务器并发续传；最终仍以本地 SHA-256 清单逐文件比对，传输方式不改变数据内容。
- 发现论文文字与公开可执行材料的一项维度差异：论文将 proprioceptive state 概括为 7D，而公开 RLDS `features.json` 明确为 **8D state**（6D end-effector pose + 2D gripper qpos）和 **7D action**。官方 LIBERO evaluation code 同样拼接 3D position、3D axis-angle、2D gripper qpos。主实验遵循公开数据和代码的 8D/7D schema，并将此差异作为 caveat 保留。

## 2026-08-19 23:02 CST｜可恢复评测编排

- 已把论文的三类 OOD scene 与发布代码对齐：`OOD_MEDIUM` 的 small translation 用于 alpha validation；两个 `OOD_HARD` sets 用于无调参 test。代码中的 `OOD_EASY`、`OOD_MULTIMODAL` 是额外入口，不纳入论文主表。
- 评测将覆盖 pretrained、Task-FT、coFT、RETAIN-task-FT 与 RETAIN-coFT。两类 RETAIN 均扫描 `α=0.1...0.9`；若本次 validation 最优值和论文报告值不同，两者都会在 test 上评测并分开汇报。
- 新增单卡、顺序、可恢复的 evaluation pipeline。每个生成命令有稳定 hash；已完成命令重启后跳过。rollout 逐 episode 写入 `episodes.json`，聚合写入 `summary.json`，扰动参数、seed、成功标记、episode length 与视频路径均可审计。

## 2026-08-19 23:11 CST｜真实训练输入预检与公开代码兼容性修复

- 从官方 GCS 获取并校验 PaliGemma tokenizer：`4,264,023` bytes，MD5 `1420adc9856720a559e8a87284b195e2`；服务器最终文件校验一致。
- 首次读取 basket 真实 RLDS batch 时触发 `KeyError`。根因是公开仓库的 `OXE_DATASET_CONFIGS` 只登记了 stove target，遗漏论文另外两个公开 target dataset 的完整名称。
- 在不改变数据变换语义的前提下，为 mugs 与 basket 补入与 stove 相同的 image/state schema 映射。修复后依次读取三个 target config 的真实单批数据，全部通过。
- 预检输出：每个 config 的 batch size 为 `64`；state `[64, 32]`、action `[64, 50, 32]`；base/wrist images 均为 `[64, 224, 224, 3]`；prompt tokens `[64, 48]`；state/action 均为 float32 且无 NaN/Inf。证据保存在 `experiments/input-smoke-targets.json`。
- loader 提示未设置 `absolute_action_mask`，因此把全部 action dimensions 视为 relative action。这与作者当前可执行 loader 的实际行为一致，主实验保持不变并记录为实现 caveat。
- TensorFlow 在 CPU-only 预检中打印重复注册 CUDA plugin 的 warning；预检仍明确运行于 `TFRT_CPU_0` 且各项断言通过，不影响数据验证结论。

## 2026-08-19 23:20 CST｜磁盘预算与端到端总控

- OpenPI 的训练 checkpoint 同时保存 EMA inference params、当前训练参数和 Adam optimizer state。若 7 个阶段都永久保留完整 train state，预计会超过共享盘预算。
- 调整的是存储策略而非论文优化协议：pretraining 每 1,000 steps 保存一个恢复点且 `max_to_keep=1`；阶段成功后验证最终 `params` 存在，再删除该阶段不再使用的 `train_state`，保留 inference params、assets、loss/gradient metrics 和清理清单。Task-FT/coFT 只读取 pretraining `params`，评测也只读取各阶段 `params`。
- 新增可恢复 supervisor，顺序执行：等待传输 → 数据 SHA-256 比对 → GCS size/MD5 与 Orbax restore → 全部 7 configs 真实 batch smoke test → 等待一张空闲 GPU → 7 个训练阶段 → 完整 alpha sweep 与 ID/OOD/generalist 评测 → 中文汇总。
- supervisor 启动前要求共享盘至少剩余 150 GB；任一阶段异常都会写入 `experiments/supervisor-status.json` 并退出，避免把不完整输入当成成功结果继续运行。

## 2026-08-19 23:23 CST｜运行中监控

- supervisor 进程持续运行，数据尺寸级校验推进到 `142 / 353` 文件、`8,938,710,635 / 24,235,684,869` bytes；仍有 32 个 aria2 sidecars，说明传输尚未完成，尚未启动 SHA-256 扫描。
- 共享盘剩余约 277 GiB。`π0 base` 目标目录表观占用约 6.3 GiB，GCS 与本地续传仍在并行进行；完成后由同一 supervisor 统一执行 24 个对象的 size/MD5 校验，当前表观目录大小不作为完成证据。
- 8 张 GPU 仍均超过空闲阈值。GPU 4–7 虽瞬时利用率约 1%，但显存仍占用约 43–50 GiB；依据“显存 ≤2,048 MiB 且利用率 ≤10%”的双条件继续等待，没有启动任何 GPU 训练进程。
- 服务器 supervisor 日志使用显式 UTC offset；本中文研究日志统一换算为 Asia/Shanghai（CST）。

## 2026-08-19 23:43 CST｜运行中监控

- supervisor 已连续运行约 23 分钟，状态仍为 `waiting_dataset_transfer`，没有异常退出或失败记录。
- 数据尺寸级校验由 `142 / 353` 推进至 `216 / 353`，完成字节由 `8,938,710,635` 增至 `14,070,463,510`。20 分钟增加 `5,131,752,875` bytes，传输持续前进；32 个 aria2 sidecars 是固定并发窗口滚动，不构成停滞证据。
- `π0 base` 目录表观占用由约 6.3 GiB 增至 8.9 GiB；GCS aria2 与 10 个本地 rsync 流仍活跃。最终只以 24/24 对象 size/MD5 和 Orbax restore 为完成条件。
- 共享盘剩余约 269 GiB，仍高于 supervisor 的 150 GB 安全线。全部 GPU 显存占用仍高于 51,080 MiB；即使部分卡瞬时利用率为 0%，也不满足空闲显存条件，因此继续等待且未启动训练。
