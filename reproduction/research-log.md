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

## 2026-08-20 00:08 CST｜数据传输停滞诊断与可恢复续传

- 首轮数据传输推进到 `271 / 353`、`17,872,213,696 / 24,235,684,869` bytes 后连续数分钟没有新增完整文件。aria2 日志显示 `DL:0B`，活动项均为服务器无法解析的 Hugging Face JSON 元数据 URL；进程最终正常退出，supervisor 按设计写入 `failed` 并停止，没有在不完整输入上继续训练。
- 使用固定 SHA-256 参考清单进行独立尺寸审计：271 个文件尺寸精确匹配，82 个未完成项全部是 TFRecord，未完成总量 `6,363,471,173` bytes；没有缺失或尺寸错误的 JSON。三个 target datasets 均为 `7/7` 文件完整。
- 新增 `audit_transfer_progress.py` 输出逐路径、逐数据集的可审计缺失清单；`generate_hf_cdn_input.py` 增加按该清单选择文件和仅生成 TFRecord 的入口，并改用 Python 标准库以消除本机 `requests` 依赖。
- 第一次切换控制文件时，`rsync --relative` 把远端目标创建成目录，导致重试进程短暂读取旧的过期 URL 并返回 HTTP 403。该进程立即被安全终止，错误控制目录移动到 `/tmp/retain_hf_aria2_retry.bad-relative-transfer`；没有删除或覆盖任何 RLDS 数据。
- 随后正确传入 82 项、246 行的新 aria2 input。fresh URL 的 expiry 晚于生成时间约一小时；新进程只打开审计列出的 TFRecord，续传速率恢复到约 3–4 MiB/s。既有 `.aria2` control files 和 partial shards 保留用于断点续传。
- supervisor 已重启并重新计算现有进度，状态恢复为 `waiting_dataset_transfer`；重启不会重做已完成文件。重启时共享盘剩余 `278,683,615,232` bytes，仍高于安全线。

## 2026-08-20 00:13 CST｜续传监控与完成门禁修正

- fresh CDN 队列保持活跃；重试启动后完整文件由 `271 / 353` 推进到 `284 / 353`，完整字节增至 `18,853,258,741 / 24,235,684,869`。最新下载日志约 `2.3 MiB/s`，没有再次出现 `0 B/s` 停滞。
- 现场检查发现首轮中断遗留 44 个 `.aria2` control sidecars，其中 20 个对应文件的表观尺寸已经等于参考尺寸。control sidecar 可能是已完成文件的遗留状态，也可能描述同尺寸文件中的未完成 byte ranges，不能单独承担最终正确性判断。
- 修正 supervisor 的完成门禁：固定清单内 353 个 payload 尺寸全部齐备且传输进程结束后，直接进入逐文件 SHA-256；SHA 不一致时仍会失败并保留现场。`verify_dataset.py` 明确排除 `.aria2` control files，避免把传输元数据误计为数据集 payload。
- 修正后的两份脚本通过 ruff 与 Python 编译检查；在服务器真实传输目录执行 `--skip-sha256` 预检，生成的 311 个当前可见 payload 记录中 `.aria2` 条目为 0。supervisor 已无损重启，aria2 下载进程没有停止或重启。
- 8 张 GPU 当前显存均至少占用约 45.7 GiB，仍不满足单卡空闲阈值；训练尚未启动，也未抢占任何现有任务。

## 2026-08-20 00:29 CST｜剩余数据监控与基础权重传输修复

- 数据重试从 `288 / 353` 继续推进到 `349 / 353`，完整字节为 `23,861,368,544 / 24,235,684,869`；仅余 4 个尺寸尚未齐备的 shard。下载进程仍活跃，日志中的剩余项持续收到数据，未判定为失败。
- 检查 `π0 base` 时发现 24 个可见对象的尺寸均已完整，但旧 aria2 仍运行且保留 5 个 control sidecars。进程文件描述符显示其目标为 `(deleted)`：并行 rsync 已原子替换路径，aria2 仍向不再可见的旧 inode 写入，因而其存活不能代表权重仍缺失。
- 在不停止旧进程的情况下，先对可见目录执行独立官方 GCS 校验；24/24 对象的 size 与 MD5 全部一致，总计 `12,014,416,199` bytes。取得内容证据后才终止旧 aria2/父进程，并将 5 个 control sidecars 移入 `/shared/.cache/retain/transfer-control-archive/pi0-base-20260820T0029CST`。该操作未删除或覆盖任何 checkpoint payload，控制文件可恢复。
- 下一门禁仍为 supervisor 的独立 GCS 复核与 Orbax restore；MD5 通过不替代实际反序列化检查。
- 本轮检查时 8 张 GPU 仍各占用约 45.9–81.2 GiB 显存，训练继续等待；共享盘可用约 `272,815,869,952` bytes，高于 150 GB 安全线。

## 2026-08-20 00:50 CST｜全量输入门禁通过，进入单卡等待

- 数据续传在 00:32 CST 达到参考清单的 `353 / 353` 文件和 `24,235,684,869` bytes，随后立即进入逐文件 SHA-256；没有仅凭尺寸宣布输入完成。
- 首次全量 SHA 正确检出 12 个 LIBERO-90 shards 内容不一致。它们均来自首轮中断后遗留的同尺寸 partial 文件，说明 aria2 control bitmap 中尚有空缺 ranges，而表观文件长度已经等于目标长度。监督器按设计以 `failed` 停止，训练没有启动；失败现场保存在 `data/dataset_manifest_server_failed_20260820T0032.json` 和 `experiments/supervisor-status-failed-data-sha-20260820T0032.json`。
- 从本地已验证的固定 revision 对这 12 个文件执行 `rsync --checksum` 差分修复：逻辑文件总量 `838,657,425` bytes，其中实际 unmatched data 为 `169,681,824` bytes，发送约 `142,876,538` bytes。修复后先逐文件完成 12/12 定向 SHA，再由重启后的 supervisor 重跑全部 353 文件；全量 SHA 于 00:45 CST 通过。
- 首次 server manifest 还出现 3 个额外 `dataset_statistics_*.json`。检查键、时间和来源后确认它们是此前三个 target dataloader smoke tests 生成的 normalization cache，不属于固定 Hugging Face revision。校验器保留这些运行所需缓存，但只把参考清单中的 353 个路径作为 payload；未知额外文件仍会导致失败。
- supervisor 独立复跑官方 GCS 校验后，`π0 base` 再次通过 24/24 size/MD5；随后实际 Orbax restore 成功：50 个 leaves、`3,238,048,528` 个 float32 参数、展开内存 `12,952,194,112` bytes，结构哈希为 `b061101d775178ee7709d97c4e8a1d5b68a63073febcd545fe6e6d1f05609dda`。
- 7 个训练 config（pretrain、三个 Task-FT、三个 coFT）均在 CPU 上读取真实 `batch=64` 并通过：state `[64, 32]`、action `[64, 50, 32]`、数值有限；机器可读证据为 `experiments/input-smoke-all.json`。
- 训练 pipeline 已启动但尚未产生训练进程。00:49 CST 的空闲检查中，各卡显存仍占用约 `43,573–81,200 MiB`，没有一张满足 `≤2,048 MiB` 且 utilization `≤10%` 的双阈值，因此继续等待且不抢占其他任务。

## 2026-08-20 01:18 CST｜首次 GPU 编译失败与 sm_120 runtime 隔离修复

- 00:52:52 CST，GPU 2 短暂达到 `582 MiB / 0%`，training pipeline 按双阈值选中该卡并启动 `retain_repro_pretrain`。进程成功创建 dataloader、读取真实 `batch=64`，并打印论文配置的 cosine LR 与 AdamW 参数；随后在任何训练 step 或数值 checkpoint 产生之前，以 exit code `134` 停止。
- 原始错误为 `Unsupported conversion from bf16 to f16` 和 `LLVM ERROR: Unsupported rounding mode for conversion.`。本机为 RTX 6000D、compute capability `12.0 (sm_120)`，项目环境固定 JAX/JAXlib `0.5.0`。结合 [JAX 官方 changelog](https://docs.jax.dev/en/latest/changelog.html) 中后续 CUDA 12.8 构建更新，以及公开的同类错误升级至 JAX 0.6.2 后消失的报告，判断为旧 XLA GPU compiler 对新架构支持不足，而不是 OOM、数据损坏或论文超参数错误。
- 失败现场完整保存在 `experiments/incidents/pretrain-jax050-sm120-20260820T0053/`：包含命令、状态、supervisor 状态和原始 stdout。checkpoint 根目录只有 Orbax root metadata、没有数字 step，故下一次启动会从 base params 重新开始，不会把半成品误当作可恢复训练。
- 为降低对已验证环境的影响，没有原位修改 `.venv`。按照 [JAX 官方安装说明](https://docs.jax.dev/en/latest/installation.html)，在 `/shared/.cache/retain/jax-overlays/0.6.2` 建立隔离 overlay，固定 `jax/jaxlib/jax-cuda12-plugin/jax-cuda12-pjrt=0.6.2`、`ml-dtypes=0.5.1`、`nvidia-cudnn-cu12=9.8.0.87`；磁盘占用约 1.7 GiB。可复建版本清单位于 `environment/requirements-jax-sm120-overlay.txt`。
- 训练与 policy server 仅通过 `PYTHONPATH` 注入该 overlay，并记录 `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false` 作为 sm_120 conservative compatibility 设置；模型、checkpoint、batch、optimizer 和训练步数不变。主 `.venv` 仍是作者依赖，可随时回退。
- CPU 侧兼容验证全部通过：7/7 configs 的真实 batch smoke、`π0 base` 的 50-leaf/3,238,048,528-parameter Orbax restore（结构 SHA 与原环境一致），以及 pretrain 完整 train-state `eval_shape`。训练 pipeline 还新增实际 GPU 门禁：下次获得空闲卡时，先用同一 runtime JIT 编译 BF16→FP16 和 BF16 GEMM；门禁成功后才启动完整 pretraining，失败则写独立 JSON 并停止。
- 01:18 CST 的 8 张 GPU 均仍被其他任务占用（显存约 39–81 GiB），当前不进行 GPU 测试，也不抢占或终止他人进程。

## 2026-08-20 01:24 CST｜修复后全量门禁重跑通过，等待 GPU runtime preflight

- 修复后的 supervisor 于 01:20:59 CST 重启，没有复用单一“成功”标记直接进入训练。它重新扫描固定 revision：353/353 参考 payload、`24,235,684,869` bytes 的 SHA-256 比较通过。
- 全 config smoke tests 已在四个 pretraining datasets 另外生成 normalization statistics cache，因而服务器目录现在共有 360 个文件：353 个固定 payload 加 7 个 `dataset_statistics_*.json`。校验器逐路径确认固定 payload 不变，仅忽略这 7 个已知 loader cache；其他未知额外文件仍会失败。
- `π0 base` 再次通过 24/24 GCS size/MD5，并重新实际 restore 为 50 leaves、3,238,048,528 float32 参数；随后原项目环境的 7/7 真实 batch smoke 再次全部通过。至此数据、权重和训练输入门禁均已在 runtime 修复后复核。
- 01:24:12 CST 训练 pipeline 进入单卡等待；当时 8 卡显存占用分别约为 `81.2/80.1/41.5/80.1/62.8/61.6/61.3/61.0 GiB`，没有卡满足 `≤2,048 MiB 且 utilization≤10%`。总控与训练等待进程持续运行，下一张真正空闲的卡将先执行 BF16 runtime preflight。
