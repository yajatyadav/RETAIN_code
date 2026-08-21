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

## 2026-08-20 01:29 CST｜空闲 GPU 等待监控

- supervisor（PID 237893）和 training pipeline（PID 242839）均持续运行，无新异常；`jax-gpu-preflight.json` 尚未生成，说明兼容层没有在忙卡上提前执行 GPU 测试。
- 最新 8 卡显存占用为 `81.2/80.1/41.5/80.1/48.2/48.3/53.6/46.3 GiB`，利用率为 `98/98/68/96/81/54/71/88%`，空闲卡计数仍为 0。共享盘可用 `271,928,930,304` bytes，继续高于 150 GB 安全线。

## 2026-08-20 01:49 CST｜单卡等待持续，双阈值避免误占忙卡

- supervisor 与 training pipeline 已分别连续运行约 28 和 25 分钟；每分钟 GPU inventory 轮询均正常写入日志，没有退出、异常或新 checkpoint。BF16 GPU preflight 仍为 pending。
- GPU 2 在 01:32–01:36 CST 曾连续显示 `0%` utilization，但显存仍占用约 `41.5 GiB`；pipeline 依据“显存与利用率同时空闲”没有选中它。01:37 后该卡显存增至约 `63.9 GiB`，随后恢复计算，证明此前只是一段已有任务的空闲间隙，而不是可安全占用的卡。
- 01:48 CST 最新显存占用为 `81.2/80.1/50.0/80.1/48.2/48.1/48.1/46.1 GiB`，利用率为 `98/95/59/98/85/82/57/85%`，仍无空闲卡。共享盘可用 `270,021,898,240` bytes，继续高于安全线。

## 2026-08-20 02:09 CST｜单卡等待监控

- supervisor 与 training pipeline 已分别连续运行约 48 和 45 分钟，进程、锁和逐分钟轮询日志均正常；GPU preflight、重试 run status 与训练 metrics 均未生成新内容，符合“尚未选卡”的预期状态。
- 02:08:23 CST，GPU 4–7 瞬时同时显示 `0%` utilization，但仍分别占用约 `45.8/48.7/47.9/44.9 GiB`；数秒后的独立查询显示各卡重新达到高利用率，进一步确认这些是已有任务的同步/等待间隙，不能视为空闲资源。
- 最新查询中 8 卡显存占用为 `81.2/80.1/50.0/80.1/63.2/61.4/61.3/61.0 GiB`，利用率为 `100/100/55/100/100/97/87/88%`，空闲卡计数为 0。共享盘可用 `268,260,474,880` bytes，仍高于 150 GB 安全线。

## 2026-08-20 02:29 CST｜单卡等待满一小时

- 修复后的 training pipeline 已持续等待约 65 分钟，supervisor 约 68 分钟；两进程仍存活且每分钟轮询连续，无 GPU preflight、训练重试或数字 checkpoint，当前不是失败状态。
- 02:27–02:28 CST，GPU 4–7 的瞬时利用率降至 `1–37%`，但显存仍约 `61–63 GiB`；随后独立查询又显示持续计算，仍不满足双阈值。
- 最新显存占用为 `81.2/80.1/49.6/80.1/50.6/53.4/47.7/52.9 GiB`，利用率为 `33/33/62/26/81/83/87/75%`，空闲卡计数为 0。共享盘可用 `267,285,663,744` bytes，继续高于安全线。

## 2026-08-20 02:49 CST｜单卡等待监控

- supervisor 与 training pipeline 已分别连续运行约 88 和 85 分钟；逐分钟 GPU 轮询持续更新，两进程均存活，无异常退出。`jax-gpu-preflight.json`、训练重试 status、metrics 与数字 checkpoint 仍未生成，说明当前仍停留在安全选卡门禁。
- 本轮独立查询中 GPU 0、1、3 瞬时利用率均为 `0%`，但分别占用 `81,200/80,057/80,133 MiB` 显存；GPU 2 占用 `54,308 MiB`，GPU 4–7 占用约 `60.9–63.0 GiB`。这些卡均明显不满足 `显存≤2,048 MiB 且 utilization≤10%`，因此没有误占其他任务的同步间隙。
- 最新 8 卡显存占用为 `81.2/80.1/54.3/80.1/63.0/61.4/61.5/61.0 GiB`，利用率为 `0/0/53/0/94/87/84/99%`，空闲卡计数为 0。共享盘可用 `264,856,993,792` bytes，继续高于 150 GB 安全线。

## 2026-08-20 03:09 CST｜单卡等待监控

- supervisor 与 training pipeline 已分别连续运行约 108 和 105 分钟，状态仍为 `training_pipeline`；逐分钟日志连续，两个 PID 均存活。GPU preflight、训练重试、metrics 和数字 checkpoint 均无新增，旧的 attempt 1 失败记录未被误判为本轮新失败。
- 独立查询时 8 张卡的 utilization 均为 `81–100%`，且显存占用为 `81.2/80.1/42.0/80.1/60.3/59.2/59.5/59.0 GiB`；没有卡接近 `≤2,048 MiB` 的空闲显存阈值，故 pipeline 继续等待且没有启动任何本实验 GPU 进程。
- 共享盘可用 `262,153,134,080` bytes，仍高于 150 GB 启动安全线；相较 supervisor 重启时减少约 10.1 GB，而本实验尚未训练或新建 checkpoint，说明主要是共享服务器其他写入。后续轮询继续同时监控余量；若接近安全线，将在选卡前人工停止并重新评估磁盘预算。

## 2026-08-20 03:29 CST｜单卡等待超过两小时

- supervisor 与 training pipeline 已分别连续运行约 128 和 125 分钟，逐分钟轮询无中断，当前仍未产生 GPU preflight、训练重试、metrics 或数字 checkpoint；旧 attempt 1 的 status 文件修改时间仍为 00:53 CST，没有新失败。
- GPU 2 在 03:27–03:29 CST 再次出现 `0%` utilization，但仍占用 `53,142 MiB` 显存；其余卡占用 `48.0–81.2 GiB`。独立快照为显存 `81.2/80.1/53.1/80.1/48.0/58.8/59.2/58.9 GiB`、利用率 `74/64/0/37/86/85/59/62%`，仍无卡满足双阈值。
- 共享盘可用 `261,693,165,568` bytes，20 分钟减少约 0.46 GB，较上一监控区间明显放缓，仍高于 150 GB 安全线。实验继续等待，不抢占现有 GPU 任务。

## 2026-08-20 03:49 CST｜单卡等待监控

- supervisor 与 training pipeline 已分别连续运行约 148 和 145 分钟；状态、PID 和逐分钟日志均正常，GPU preflight、训练重试、metrics、数字 checkpoint 仍无新增，旧 attempt 1 status 的修改时间保持不变。
- 03:33 CST 曾出现 GPU 4–7 同时 `0%` utilization，但各自仍占用约 `46.8–50.1 GiB` 显存，下一分钟便恢复计算，继续证明只按利用率选卡会误占他人任务。本轮独立快照为显存 `81.2/80.1/46.4/80.1/63.0/61.5/61.4/61.1 GiB`、利用率 `100/99/69/99/97/100/93/100%`，空闲卡计数仍为 0。
- 共享盘可用 `261,595,672,576` bytes，20 分钟仅减少约 0.10 GB，余量基本稳定且高于 150 GB 安全线。实验继续由双阈值门禁等待。

## 2026-08-20 04:27 CST｜CPU preflight false positive、cuDNN 冲突与完整 CUDA 隔离修复

- 03:49:48 CST，GPU 2 达到 `508 MiB / 0%` 并被正确选中。初版最小 BF16 preflight 的 stderr 实际报告 CUDA libraries 无法加载、JAX 回退 `TFRT_CPU_0`，但脚本只检查 return code 与 JSON，错误地写成 `verified`。该报告已改名为 `jax-gpu-preflight.invalid-cpu-fallback-20260820T0350.json`，不再代表当前门禁成功。
- 完整训练随即在 03:50:02 CST 以 exit `1` 停止。TensorFlow 先加载了主 `.venv` 的 cuDNN `9.7.1`，JAX CUDA plugin 0.6.2 则按 build 要求 cuDNN `9.8.0`，最终报 `DNN library initialization failed`。故障位于随机数 key / runtime 初始化阶段，仍早于 train state、step 0 和数字 checkpoint；没有损失数据或半成品恢复点。
- 根因是初版 overlay 只放入了 JAX 与 cuDNN 9.8，其他 CUDA wheels 继续复用主环境；overlay 的 `nvidia` package 同时遮蔽主环境的同名模块，使纯 JAX 与 TensorFlow-first 两条导入路径产生不同失败。现场已归档到 `experiments/incidents/pretrain-jax062-cuda-library-path-20260820T0350/`。
- 修复保持主 `.venv` 不变，在同一 overlay 中补齐并固定 JAX plugin 所需的 CUDA runtime、cuBLAS、cuPTI、NVRTC、cuFFT、cuSOLVER、cuSPARSE、NCCL、nvJitLink 与 NVSHMEM；版本与 JAX 0.6.2 的 CUDA 12.8 build versions 对齐。训练和 policy server 显式把 overlay 的动态库与 `cuda_nvcc/bin` 放在搜索路径最前。
- 新门禁要求 `len(jax.devices()) == 1` 且 `device.platform == "gpu"`，再编译 BF16→FP16 和 BF16 GEMM；CPU fallback 无论是否 exit 0 都会失败。无 GPU 审计已验证 17 个固定 packages、12/12 关键动态库从 overlay 加载，以及 CPU BF16 smoke；机器可读报告为 `experiments/jax062-overlay-audit.json`。
- 04:23 CST supervisor 第 5 次启动，再次通过 353/353 数据 SHA-256、24/24 基础权重 size/MD5、Orbax restore（3,238,048,528 参数、结构哈希不变）和 7/7 真实输入 smoke。04:26 CST 新 training pipeline（PID 251712）已进入等待；当前 8 卡显存为 `81.2/80.1/43.5/80.1/55.6/48.4/53.9/48.7 GiB`，利用率为 `95/100/75/100/88/65/60/88%`，空闲卡计数 0。
- 完整 overlay 占用 `4,653,588,089` bytes；安装后共享盘可用 `258,645,110,784` bytes，仍高于 150 GB 安全线。下一张真正空闲的卡才会执行严格 GPU preflight。

## 2026-08-20 04:52 CST｜完整 CUDA 隔离修复后的等待监控

- 第 5 次启动的 supervisor（PID 247374）与 training pipeline（PID 251712）已分别连续运行约 29 和 26 分钟；状态仍为 `training_pipeline`，逐分钟 GPU 轮询连续，两进程均存活。
- 严格 `jax-gpu-preflight.json`、新训练 status、metrics 和数字 checkpoint 均未生成，说明修复后的 runtime 尚未在 GPU 上执行，而不是门禁失败或训练异常退出。
- 最新 8 卡显存占用为 `81.2/80.1/48.8/80.1/62.8/61.9/61.3/60.8 GiB`，利用率为 `97/100/62/100/100/100/93/100%`；没有卡满足 `显存≤2,048 MiB 且 utilization≤10%`，因此继续安全等待且不抢占现有任务。
- 共享盘可用 `258,167,386,112` bytes，较修复完成时减少约 0.48 GB，仍高于 150 GB 启动安全线。

## 2026-08-20 05:12 CST｜单卡等待监控

- supervisor 与 training pipeline 已分别连续运行约 49 和 46 分钟；状态仍为 `training_pipeline`，两个 PID 与逐分钟轮询日志均正常，严格 GPU preflight、新训练 status、metrics 和数字 checkpoint 仍无新增。
- GPU 2 在 05:07–05:12 CST 连续显示 `0%` utilization，但仍占用 `48,792 MiB` 显存，明确不满足空闲显存阈值；pipeline 没有把已有任务的计算间隙误判为空闲卡。
- 独立快照的 8 卡显存占用为 `81.2/80.1/48.8/80.1/50.0/49.3/48.1/47.8 GiB`，利用率为 `100/100/0/100/77/84/85/74%`，空闲卡计数仍为 0。
- 共享盘可用 `258,070,933,504` bytes，20 分钟减少约 0.10 GB，仍高于 150 GB 启动安全线。

## 2026-08-20 05:32 CST｜单卡等待监控

- supervisor 与 training pipeline 已分别连续运行约 69 和 65 分钟；状态、PID 与逐分钟日志均正常，严格 GPU preflight、新训练 status、metrics 和数字 checkpoint 仍无新增。
- GPU 2 在 05:27–05:32 CST 再次连续显示 `0%` utilization，但显存仍占用 `48,434 MiB`；双阈值门禁继续正确拒绝这段已有任务的计算间隙。
- 独立快照的 8 卡显存占用为 `81.2/80.1/48.4/80.1/49.6/53.8/48.2/48.0 GiB`，利用率为 `100/100/0/100/87/72/79/56%`，空闲卡计数为 0。
- 共享盘可用 `257,977,344,000` bytes，20 分钟减少约 0.09 GB，继续高于 150 GB 启动安全线。

## 2026-08-20 05:52 CST｜单卡等待监控

- supervisor 与 training pipeline 已分别连续运行约 89 和 85 分钟；状态仍为 `training_pipeline`，两进程及逐分钟轮询正常，严格 GPU preflight、新训练 status、metrics 和数字 checkpoint 均未产生。
- 最新独立快照中 8 卡显存占用为 `81.2/80.1/49.6/80.1/60.7/59.7/58.8/58.4 GiB`，利用率为 `100/96/86/100/88/100/100/100%`；所有卡均明显不满足双空闲阈值，pipeline 未启动本实验 GPU 进程。
- 共享盘可用 `257,531,854,848` bytes，20 分钟减少约 0.45 GB，仍高于 150 GB 启动安全线。当前无失败需要诊断，继续等待一张真正空闲的 GPU。

## 2026-08-20 06:12 CST｜单卡等待监控

- supervisor 与 training pipeline 已分别连续运行约 109 和 105 分钟；状态仍为 `training_pipeline`，两进程存活且逐分钟日志连续。严格 GPU preflight、新训练 status、metrics 和数字 checkpoint 均未生成。
- 独立快照中 GPU 2 为 `46,816 MiB / 0%`，仍是已有任务的计算间隙；全部 8 卡显存占用为 `81.2/80.1/46.8/80.1/61.2/60.5/59.4/59.3 GiB`，空闲卡计数为 0，本实验没有启动 GPU 进程。
- 共享盘可用 `257,435,742,208` bytes，20 分钟减少约 0.10 GB，继续高于 150 GB 启动安全线。实验无新失败，保持单卡双阈值等待。

## 2026-08-20 06:32 CST｜单卡等待超过两小时

- supervisor 与 training pipeline 已分别连续运行约 129 和 125 分钟；状态、PID 与逐分钟日志保持正常，严格 GPU preflight、新训练 status、metrics 和数字 checkpoint 均未生成。
- 06:23–06:30 CST 多张卡曾出现低利用率，但显存仍占用 `43–81 GiB`；最新独立快照为显存 `81.2/80.1/43.1/80.1/62.9/61.5/61.4/61.0 GiB`、利用率 `94/96/74/95/94/96/100/88%`，仍无卡满足双阈值。
- 共享盘可用 `257,341,636,608` bytes，20 分钟减少约 0.09 GB，继续高于 150 GB 启动安全线。实验没有失败或资源越界，继续等待且不抢占现有任务。

## 2026-08-20 06:59 CST｜严格 GPU 门禁通过、batch-64 首步 OOM 与显存池修复

- 06:46:56 CST，GPU 2 达到 `582 MiB / 0%` 并被双阈值门禁选中。06:47:00 CST，强化后的 preflight 首次在唯一可见的 `NVIDIA RTX 6000D` 上真实通过：JAX、JAXlib、CUDA plugin 均为 `0.6.2`，`platform=gpu`，BF16→FP16 与 BF16 GEMM 均已编译执行且无 stderr。这证明完整 CUDA 隔离和 sm_120 runtime 修复有效。
- attempt 3 随后成功读取真实 batch 64、加载基础权重、初始化 train state、AdamW 与 EMA，并进入 `0/10000` 的首个训练 step。XLA 报告峰值从 `77.10 GiB` rematerialize 至 `75.44 GiB`，但当前 85,651 MiB GPU 的 90% 预分配池只有约 `75.28 GiB`；执行时申请 `27.79 GiB` buffer 失败，以 `RESOURCE_EXHAUSTED` 退出。
- 失败仍早于 Step 0 指标与数字 checkpoint；没有可恢复的半成品。完整现场归档于 `experiments/incidents/pretrain-jax062-batch64-oom-20260820T0649/`，成功的 preflight 另保留为 `experiments/jax-gpu-preflight.verified-attempt3-20260820T0647.json`。
- 修复不缩小论文优化协议：模型、全局 batch 64、10,000 steps、LR schedule、AdamW、gradient clipping、EMA、mixture 与 seed 均保持不变。只把 JAX 预分配比例由 `0.90` 提高到 `0.95`（约 `79.46 GiB`，比已编译峰值多约 `4.02 GiB`），并设置 `TF_FORCE_GPU_ALLOW_GROWTH=true`，避免 RLDS/TensorFlow 输入管线一次性占用剩余显存。
- 第 6 次 supervisor 于 06:56:26 CST 启动，重新通过 353/353 数据 SHA-256、24/24 权重 size/MD5、真实 Orbax restore（3,238,048,528 参数、结构哈希不变）和 7/7 输入 smoke。06:59 CST，新的 training pipeline（PID 260509）已进入等待；当前空闲卡计数为 0，严格 preflight 将在下一张真正空闲 GPU 上再次执行。
- 最新 8 卡显存占用为 `81.2/80.1/39.4/80.1/49.6/48.5/48.1/46.9 GiB`，利用率为 `98/95/53/95/72/82/82/84%`；共享盘可用 `256,817,639,424` bytes，仍高于 150 GB 安全线。

## 2026-08-20 07:13 CST｜95% 显存池重试前的单卡等待监控

- 第 6 次 supervisor（PID 256109）与 training pipeline（PID 260509）已分别持续运行约 16 和 13 分钟；状态仍为 `training_pipeline`，两进程存活且逐分钟轮询连续。新的严格 GPU preflight、attempt 4 status、训练 metrics 与数字 checkpoint 均未生成，因此当前是安全选卡等待，不是新失败。
- GPU 2 在 07:08–07:13 CST 连续显示 `0%` utilization，但仍占用 `39,406 MiB` 显存，明显高于 `2,048 MiB` 空闲阈值；双阈值门禁没有把他人任务的计算间隙误判为空闲卡。
- 独立快照的 8 卡显存占用为 `81.2/80.1/39.4/80.1/63.0/61.2/61.4/61.1 GiB`，利用率为 `95/99/0/100/91/97/89/97%`，空闲卡计数为 0。
- 共享盘可用 `256,753,876,992` bytes，较 06:59 CST 减少约 0.06 GB，继续高于 150 GB 安全线。实验保持论文 batch 64 与 95% JAX 显存池配置，等待下一张真正空闲的 GPU 后重新执行严格门禁和完整首步。

## 2026-08-20 07:32 CST｜单卡等待监控

- 第 6 次 supervisor（PID 256109）与 training pipeline（PID 260509）已分别持续运行约 36 和 32 分钟；状态仍为 `training_pipeline`，两进程和逐分钟轮询日志连续，无子训练进程。新的严格 GPU preflight、attempt 4 status、训练 metrics 与数字 checkpoint 仍未生成，当前没有新失败。
- GPU 2 在 07:27–07:32 CST 再次连续显示 `0%` utilization，但显存占用已由上轮 `39,406 MiB` 增至 `43,512 MiB`；这进一步证明该卡仍由现有任务持有，不能在利用率间隙接管。
- 独立快照的 8 卡显存占用为 `81.2/80.1/43.5/80.1/62.9/61.4/61.4/61.1 GiB`，利用率为 `100/100/0/98/96/97/100/92%`，空闲卡计数为 0。
- 共享盘可用 `256,671,571,968` bytes，约 19 分钟减少 0.08 GB，继续高于 150 GB 安全线。总控保持论文 batch 64、95% JAX 显存池和 TensorFlow memory growth 配置，等待真正空闲的一张 GPU。

## 2026-08-20 07:52 CST｜单卡等待监控

- 第 6 次 supervisor（PID 256109）与 training pipeline（PID 260509）已分别持续运行约 55 和 52 分钟；状态仍为 `training_pipeline`，两进程与逐分钟轮询连续且无子训练进程。严格 GPU preflight、attempt 4 status、训练 metrics 和数字 checkpoint 均无新增，当前不是失败状态。
- 上轮低利用率的 GPU 2 已恢复到 `47,442 MiB / 68%`，确认它一直由已有任务持有。其余 GPU 也占用 `47.3–81.2 GiB`，没有任何一张满足 `显存≤2,048 MiB 且 utilization≤10%`。
- 独立快照的 8 卡显存占用为 `81.2/80.1/47.4/80.1/54.0/48.4/48.0/47.3 GiB`，利用率为 `98/90/68/91/82/78/55/78%`，空闲卡计数为 0。
- 共享盘可用 `256,574,754,816` bytes，约 20 分钟减少 0.10 GB，继续高于 150 GB 安全线。实验保持论文 batch 64 的 95% 显存池重试计划，不抢占其他任务。

## 2026-08-20 08:24 CST｜batch-64 的两次排除性重试与内存结论修正

- 07:57:46 CST，GPU 1 以 `10 MiB / 0%` 被双阈值门禁选中；07:57:50 CST，严格 preflight 再次确认唯一 `NVIDIA RTX 6000D` GPU backend、JAX/JAXlib/CUDA plugin `0.6.2`、BF16→FP16 与 BF16 GEMM 均成功且 stderr 为空。
- attempt 4 使用论文 batch 64 和 95% BFC 预分配池。真实数据、`π0 base`、train state、AdamW 与 EMA 初始化后进入 `0/10000`，但仍在申请 `27.79 GiB` buffer 时 `RESOURCE_EXHAUSTED`，07:59:32 CST 退出；没有 Step 0 指标或数字 checkpoint。现场归档于 `experiments/incidents/pretrain-jax062-batch64-bfc-fragmentation-20260820T0759/`，成功门禁另存为 `experiments/jax-gpu-preflight.verified-attempt4-20260820T0757.json`。
- 按 JAX 官方 memory allocation 文档，attempt 5 请求 `TF_GPU_ALLOCATOR=cuda_malloc_async` 并取消固定 BFC pool。08:19:22 CST GPU 1 再次以 `10 MiB / 0%` 被选中，08:19:25 CST 严格门禁通过；完整首步却仍在同一 `27.79 GiB` 请求处 OOM，且 runtime 日志仍写明 `GPU_0_bfc`，证明当前 JAX 0.6.2 环境没有采用请求的 async allocator。08:21:04 CST 退出，现场归档于 `experiments/incidents/pretrain-jax062-cuda-malloc-async-ineffective-20260820T0821/`。
- 对 attempt 3 的早期解释作正式更正：XLA 的“从 `77.10 GiB` 降到 `75.44 GiB`，仍无法降到约 `25.07 GiB`”描述完整 train-step 的 rematerialization 安排，不能简化成“只比 90% pool 多 0.16 GiB”。attempts 3–5 在相同大块请求处连续失败，已足以判定论文 batch 64 的完整 optimizer step 不适合当前单张 85,651 MiB GPU。
- 按单 GPU 约束下的缩小预案，七个训练 config 的物理 batch 统一从 64 改为 16，不使用 gradient accumulation；steps、LR、AdamW、gradient clipping、EMA、mixture、seed、模型与数据保持不变。这是明确的科学偏差，不宣称与论文 batch 64 数值等价。若 batch 16 仍失败，才进一步降到 batch 8。

## 2026-08-20 08:34 CST｜batch-16 首步成功，117-task pretraining 正式运行

- 第 8 次 supervisor 于 08:26:11 CST 启动，再次逐项通过 353/353 数据 SHA-256、24/24 `π0 base` size/MD5、真实 Orbax restore 和 7/7 输入 smoke tests。08:29:15 CST 训练 pipeline 选择空闲 GPU 1；08:29:19 CST 严格 GPU preflight 第四次真实通过，报告保存为 `experiments/jax-gpu-preflight.verified-attempt6-20260820T0829.json`。
- attempt 6 的真实 loader 输出确认图像、state、prompt 与 action 的 batch 维均为 `16`。原论文的 10,000 steps、cosine LR（warmup 1,000、peak `2.5e-5`、end `2.5e-6`）、AdamW（β₁=0.9、β₂=0.95、weight decay `1e-10`）、clip 1.0、EMA、数据 mixture 与基础模型均保持不变。
- `π0 base` 12.1 GiB 参数在约 22 秒内恢复完成；train state 构建后于 08:30:39 CST 进入 optimizer loop。batch 16 的 XLA 图从 `62.62 GiB` rematerialize 到 `62.52 GiB`，目标约 `29.09 GiB`；与 batch 64 不同，本轮没有 OOM，08:37:54 CST 已连续推进到至少 step 181，稳定速率约 `2.2 s/step`，预计 pretraining 还需约 6 小时 3 分。
- 08:37 CST 快照中，本实验只占用 GPU 1：`81,318 MiB / 100%`；其余卡均非本实验进程。共享盘可用 `256,161,005,568` bytes，仍高于 150 GB 安全线。首个恢复 checkpoint 按配置应在 step 1000 生成，当前没有数字 checkpoint 属正常现象。
- 当前训练进程在重定向文件上的 `pbar.write` 数值行受 Python stdout 缓冲，实时日志已能证明更新步连续完成，但尚不能读取 Step 0/25/50/75 的 loss 与 norm 数值。已给 `scripts/train.py` 增加每条 metrics 后 `stdout.flush()`；该操作只改变后续进程的日志可见性，不改变训练计算，当前进程结束时缓冲内容也会正常写出。有限值与曲线结论将在指标可见后补录。

## 2026-08-20 08:41 CST｜batch-16 pretraining 增量监控

- attempt 6 已由 step 181 继续推进到至少 step 271，稳定维持约 `2.2 s/step`，预计剩余约 6 小时 1 分；supervisor、training pipeline、`uv` 与训练子进程四级进程链均存活，run status 仍为 `running`。
- GPU 1 为本实验唯一使用的卡，快照为 `81,318 MiB / 100%`；共享盘可用 `256,160,481,280` bytes，仍高于 150 GB 安全线。step 1000 前未出现数字 checkpoint，符合配置预期。
- `stdout.log` 会跨 attempt 追加，因此直接扫描末尾会命中 attempt 5 的历史 OOM。以 attempt 6 的 `2026-08-20 00:29:25.660753 UTC` 起始标记切片后，当前区间没有 `Traceback`、`RESOURCE_EXHAUSTED` 或 `out of memory`；指标行仍在当前 Python 进程的 stdout 缓冲中，继续等待正常刷新后做有限值审计。

## 2026-08-20 09:16 CST｜step-1000 恢复 checkpoint 原子提交并验证

- attempt 6 持续稳定运行：step 1000 于 09:08:15 CST 触发 Orbax async save。GPU→host/磁盘的 blocking 阶段约 217.63 秒；保存期间主机可用内存约 479 GB，未出现内存压力，之后训练先恢复推进，后台保存线程再完成最终提交。
- 09:12:37 CST，Orbax 依次完成 `assets`、`params`、`train_state` 子项重命名，将根临时目录原子重命名为 `1000/`，并明确报告 `Background save thread done`、`No errors found` 和所有 host finalize 完成。验证时临时 checkpoint 目录计数为 0，元数据已有非空 `commit_timestamp_nsecs=1787188357405169568`。
- 稳定文件 payload 审计为 45 个文件、42,976,247,401 bytes：`assets` 1 个/4,496 bytes、`params` 18 个/12,014,099,668 bytes、`train_state` 25 个/30,962,142,820 bytes，另含顶层 checkpoint metadata。验证时整个目录表观大小为 42,976,292,457 bytes；元数据 SHA-256 为 `a92393cadef1bac156f83bae1cbf29b50fe0f24ac54ff005187b0fe80164a856`。
- 使用项目主环境的 Orbax 在隐藏 GPU 的 CPU 只读模式打开父目录，`CheckpointManager` 返回 `all_steps=[1000]`、`latest_step=1000`，证明该提交可作为恢复点被管理器发现。此处没有为了里程碑审计而额外把约 43 GB 数组完整 restore 进内存；最终 checkpoint 将执行实际 restore 验证。
- 09:15 CST 训练已继续到约 step 1090，仍约 `2.2 s/step`、剩余约 5 小时 31 分；attempt 6 起始标记后的日志没有 `Traceback`、`RESOURCE_EXHAUSTED` 或 OOM。唯一实验卡 GPU 1 为 `81,318 MiB / 100%`，共享盘尚余 `213,181,792,256` bytes，高于 150 GB 安全线。
- 当前训练进程的数值 metrics 仍受既有 stdout 缓冲影响，故本轮只确认训练推进和恢复 checkpoint 的结构/提交状态，不提前判断 loss 收敛。验证详情保存于 `experiments/pretrain-checkpoint-1000-validation.json`。

## 2026-08-20 09:21 CST｜checkpoint 后存活复核

- step-1000 checkpoint 完成后，attempt 6 已继续推进到约 step 1250，速率保持约 `2.2 s/step`，预计剩余约 5 小时 25 分；supervisor、training pipeline、`uv` 和训练 Python 的四级进程链均存活，run status 为 `running`。
- 以 attempt 6 起始标记切片，`Traceback`、`RESOURCE_EXHAUSTED` 与 OOM 合计仍为 0；实时 Step metrics 行仍为 0，符合当前 Python 进程已知的 stdout 缓冲行为。checkpoint 根目录仍只有已验证的 `1000/`，没有异常临时目录或意外重复保存。
- 本实验仍只使用 GPU 1，快照为 `81,318 MiB / 100%`；共享盘可用 `213,180,985,344` bytes，继续高于 150 GB 安全线。其余 GPU 状态仅作共享服务器审计，不由本实验占用或抢占。

## 2026-08-20 09:52 CST｜step-2000 checkpoint 与滚动保留验证

- attempt 6 从约 step 1250 连续推进到 step 2000，区间速率保持约 `2.2 s/step`，没有 `Traceback`、`RESOURCE_EXHAUSTED` 或 OOM。09:49:02 CST 触发第二次 Orbax async save，旧 `1000/` 在新保存期间保持可见。
- 本次 GPU→host blocking 阶段为 `90.4524` 秒，明显短于 step 1000 的 `217.6332` 秒；这里只报告观测值，不把差异归因于缓存或 I/O 状态。blocking 结束后训练先恢复，短暂 ETA 波动随后回到约 `2.2 s/step`。
- 09:51:33 CST，`2000.orbax-checkpoint-tmp-4` 原子重命名为 `2000/`，后台线程报告无错误；09:51:40 CST，Orbax 才完成旧 step 1000 的实际删除。最终根目录只含 `2000/` 且无临时目录，证明 `max_to_keep=1` 在本环境中按“新点提交后完成旧点清理”的方式工作。
- 新 checkpoint 为 45 个文件、42,932,572,354 bytes 文件 payload：`assets` 1 个/4,496 bytes、`params` 19 个/12,014,086,733 bytes、`train_state` 24 个/30,918,480,708 bytes，另含顶层 metadata；目录表观大小为 42,932,617,410 bytes。metadata 的 SHA-256 为 `1745385b70a1292cf0fa2315d4aa1b42e968478146a5be0e53447ec0fb06f802`，commit timestamp 非空。
- 独立 CPU 只读 `CheckpointManager` 返回 `all_steps=[2000]`、`latest_step=2000`。两份 checkpoint 并存时观测到主机可用内存 `393,845,176,320` bytes、共享盘可用 `169,767,882,752` bytes，仍高于 150 GB 安全线；旧点清理后共享盘回升至 `212,744,130,560` bytes。
- 09:52 CST 四级进程链仍存活，训练已推进至约 step 2050，GPU 1 为 `81,318 MiB / 100%`，当前 attempt 错误计数与实时 metrics 行仍分别为 0。checkpoint 验证详情保存于 `experiments/pretrain-checkpoint-2000-validation.json`。

## 2026-08-20 10:02 CST｜首段 90 条训练 metrics 有限值与趋势审计

- 当前 Python 进程的 stdout 缓冲在训练中达到刷新条件，attempt 6 的 Step metrics 从 0 条一次性变为 90 条，覆盖 step 0--2225，间隔严格为 25 step。调用总控已有的 `export_metrics` 将日志转换为 `metrics.jsonl`，没有改动运行中的训练进程。
- 90 条记录的 `loss`、`grad_norm`、`param_norm`、`vision_param_norm` 全部为有限值。loss 从 step 0 的 `0.1572` 降至 step 2225 的 `0.0381`，首末下降 `75.7634%`；全段最小/最大为 `0.0370 / 0.1572`，前 10 条和后 10 条均值分别为 `0.09298 / 0.04065`。
- 按连续 500-step 区间统计，loss 均值为：step 0--475 `0.07106`、500--975 `0.04671`、1000--1475 `0.04514`、1500--1975 `0.040905`、2000--2225 `0.04065`。这说明主要下降发生在早期，随后在约 0.04 附近波动；当前只覆盖 22.25% 训练，记录为“早期下降并趋于平台”，不提前宣称最终收敛。
- grad norm 的前/后 10 条均值为 `1.77271 / 0.47439`，全段范围 `0.3971--3.9338`；param norm 从 `1377.8652` 缓慢增至 `1378.5364`，vision param norm 从 `1258.1694` 增至 `1258.254601`，没有数值爆炸迹象。
- 10:02 CST 训练已继续至约 step 2310，仍为 `2.2 s/step`，预计剩余约 4 小时 45 分；四级进程链存活，attempt 6 错误计数为 0，checkpoint 根目录仅有已验证的 `2000/`。GPU 1 为 `81,318 MiB / 100%`，共享盘可用 `212,744,544,256` bytes。
- 本次导出的 `metrics.jsonl` 为 90 行、9,737 bytes，SHA-256 为 `728ac68fb231cff26aa8f96f0bd8561bebfaf76dcef4e81b57bc4e94eacf8bef`；机器可读汇总保存于 `experiments/pretrain-metrics-through-step-2225.json`。后续缓冲刷新或训练结束时将重新导出完整 superseding 文件。

## 2026-08-20 10:34 CST｜step-3000 checkpoint 第三次滚动提交验证

- attempt 6 在无新错误的情况下推进到 step 3000，并于 10:27:41 CST 触发第三次 Orbax async save。checkpoint manager blocking 阶段为 `251.2146` 秒；这一观测值长于 step 2000 的 `90.4524` 秒，但当前不对共享存储 I/O 差异作未经测量的归因。
- 保存过程中旧 `2000/` 始终可见，新点写入 `3000.orbax-checkpoint-tmp-8/`。临时目录达到约 39.44 GB 时，主机可用内存仍为 `381,245,302,784` bytes，共享盘最低观测余量为 `173,295,751,168` bytes，高于 150 GB 安全线；实验仍只占用 GPU 1。
- 10:32:45 CST，Orbax 完成三个 item 的 finalize，把根临时目录原子重命名为 `3000/`，随后明确报告 `Background save thread done`、`No errors found`；旧 step 2000 于 10:32:47 CST 才完成删除。最终 checkpoint 根目录只含 `3000/` 且无临时目录，第三次验证了新点提交后再完成旧点清理的滚动行为。
- step-3000 checkpoint 共 45 个文件、`42,936,168,167` bytes 文件 payload：`assets` 1 个/4,496 bytes、`params` 18 个/12,014,088,103 bytes、`train_state` 25 个/30,922,075,151 bytes，另含顶层 metadata；目录表观大小为 `42,936,213,223` bytes。metadata commit timestamp 为 `1787193165738162814`，SHA-256 为 `e5592c97cf9421d74040a5fb9cc90385723db17380490661e1c65661064c5f7c`。
- 隐藏 GPU 后，以项目主环境的 CPU 只读 `CheckpointManager` 打开父目录，返回 `all_steps=[3000]`、`latest_step=3000`。训练随后恢复到约 step 3060、稳定速率约 `2.2 s/step`，attempt 6 错误计数仍为 0；metrics 缓冲尚未再次刷新，仍为已审计的 90 条（step 0--2225），因此本轮不新增数值趋势结论。
- 旧点清理后共享盘余量回升至 `212,724,129,792` bytes。机器可读验证详情保存于 `experiments/pretrain-checkpoint-3000-validation.json`；最终 checkpoint 仍按计划执行完整数组 restore，本次中间里程碑只验证提交完整性和 Orbax 可发现性。

## 2026-08-20 10:41 CST｜step-3000 提交后的存活复核

- supervisor、training pipeline、`uv` 与训练 Python 四级进程链继续存活，运行状态文件明确为 `running`。训练已从 step 3060 推进至约 step 3230，速率恢复并稳定在约 `2.2 s/step`，预计 pretraining 剩余约 4 小时 11 分。
- attempt 6 起始标记后的 `Traceback`、`RESOURCE_EXHAUSTED` 与 OOM 计数仍为 0；metrics 日志仍为已审计的 90 条（step 0--2225），没有新的缓冲刷新，因此不重复导出或外推曲线。
- checkpoint 根目录稳定只含已验证的 `3000/`，没有残留临时目录。本实验仍只使用 GPU 1，快照为 `81,318 MiB / 100%`；主机可用内存 `439,951,350,784` bytes，共享盘可用 `212,721,020,928` bytes，资源保持在安全范围。

## 2026-08-20 11:14 CST｜step-4000 checkpoint 第四次滚动提交验证

- attempt 6 从约 step 3230 连续推进至 step 4000，区间保持约 `2.2 s/step`，当前 attempt 没有 `Traceback`、`RESOURCE_EXHAUSTED` 或 OOM。11:09:02 CST 触发第四次 Orbax async save，旧 `3000/` 在新点临时写入期间保持可见。
- 本次 checkpoint blocking 阶段为 `211.3325` 秒。保存早期在临时目录约 12.0 GB 时，实测主机可用内存 `404,400,436,224` bytes、共享盘可用 `200,727,379,968` bytes；该时点快照证明资源仍在安全区，但不表述为整个重叠窗口的最低余量。
- 11:13:37 CST，Orbax 将 `assets`、`params`、`train_state` 与根临时目录依次原子重命名，后台线程报告 `No errors found`；11:13:39 CST 才完成旧 step 3000 删除。最终根目录只含 `4000/` 且无临时目录，继续验证新点提交后再完成旧点回收。
- step-4000 checkpoint 共 51 个文件、`42,939,162,153` bytes 文件 payload：`assets` 1 个/4,496 bytes、`params` 18 个/12,014,110,303 bytes、`train_state` 31 个/30,925,046,937 bytes，另含顶层 metadata；目录表观大小为 `42,939,207,209` bytes。metadata commit timestamp 为 `1787195617491671123`，SHA-256 为 `99287857156e12abeb9565c442a63ec8c31fb3d95a09899dcb75f62aaa013dbf`。
- 隐藏 GPU 后，项目主环境的 CPU 只读 `CheckpointManager` 返回 `all_steps=[4000]`、`latest_step=4000`。训练随后恢复至约 step 4050、稳定速率约 `2.2 s/step`，预计 pretraining 剩余约 3 小时 41 分；错误计数仍为 0，metrics 缓冲仍是 90 条（step 0--2225），故本轮不新增曲线结论。
- 清理后主机可用内存为 `419,512,660,992` bytes，共享盘可用 `212,735,385,600` bytes；本实验仍只使用 GPU 1，快照为 `81,318 MiB / 100%`。机器可读详情保存于 `experiments/pretrain-checkpoint-4000-validation.json`。

## 2026-08-20 11:21 CST｜step-4000 提交后的存活复核

- 运行状态文件仍明确为 `running`，supervisor、training pipeline、`uv` 与训练 Python 四级进程链全部存活。训练已继续至约 step 4220，速率稳定在约 `2.2 s/step`，预计 pretraining 剩余约 3 小时 35 分。
- attempt 6 错误计数仍为 0；metrics 仍为已审计的 90 条（step 0--2225），没有新的 stdout 缓冲刷新。checkpoint 根目录只含已验证的 `4000/`，无临时目录或异常重复保存。
- 本实验仍只占用 GPU 1，快照为 `81,318 MiB / 100%`；主机可用内存 `419,035,575,296` bytes，共享盘可用 `212,636,196,864` bytes，保持单卡约束与资源安全线。

## 2026-08-20 11:57 CST｜step-5000 checkpoint 与 44.75% 指标审计

- attempt 6 从约 step 4220 连续推进到 step 5000，当前 attempt 没有 `Traceback`、`RESOURCE_EXHAUSTED` 或 OOM。11:49:43 CST 触发第五次 Orbax async save；checkpoint manager blocking 阶段为 `241.3401` 秒，训练于 11:53:44 CST 恢复计算，后台线程继续写盘。
- 保存期间旧 `4000/` 与 `5000.orbax-checkpoint-tmp-16/` 同时可见。20 秒监控采样中，11:54:31 CST 的临时目录约为 42.95 GB，主机可用内存 `351,858,936,832` bytes，共享盘可用 `169,653,215,232` bytes，仍高于 150 GB 安全线；这是离散采样中的最低观测余量，不表述为连续窗口的理论最低值。
- 11:54:35 CST，Orbax 依次完成 `assets`、`params`、`train_state` 和根目录的原子重命名，随后报告 `Background save thread done` 与 `No errors found`；旧 step 4000 的实际删除于 11:54:37 CST 完成。最终根目录只含 `5000/` 且没有临时目录，第五次验证新点提交后完成旧点回收的滚动行为。
- step-5000 checkpoint 共 49 个文件、`42,948,570,663` bytes 文件 payload：`assets` 1 个/4,496 bytes、`params` 21 个/12,014,121,745 bytes、`train_state` 26 个/30,934,444,005 bytes，另含顶层 metadata；目录表观大小为 `42,948,615,719` bytes。metadata commit timestamp 为 `1787198075760678304`，SHA-256 为 `3c7ba68e55fe3f47261bddf199365ae3ccb90928d06921777de604cce2f194ff`。
- 隐藏 GPU 后，项目主环境的 CPU 只读 `CheckpointManager` 返回 `all_steps=[5000]`、`latest_step=5000`。完整数组 restore 仍留到最终 checkpoint，避免正在训练时额外占用约 43 GB I/O 与主机内存。机器可读验证详情保存于 `experiments/pretrain-checkpoint-5000-validation.json`。
- 本次保存使 stdout 缓冲再次刷新；重新导出的 `metrics.jsonl` 为 180 行、19,511 bytes，SHA-256 为 `b8797ca0789128ae3e1e0cb159b0166eb546e17d1354c1c31f48d44046ac7b6a`，覆盖 step 0--4475 且间隔严格为 25 step。四项指标全部有限；loss 从 `0.1572` 降至 `0.0327`，最近 10 条均值为 `0.03444`。step 2000--4475 的连续 500-step loss 均值为 `0.03890 / 0.03734 / 0.03556 / 0.03472 / 0.03419`，支持平台后仍缓慢改善，但当前仅覆盖 44.75%，不提前宣称最终收敛。机器可读分析保存于 `experiments/pretrain-metrics-through-step-4475.json`。
- 11:56 CST 四级进程链仍存活，训练已推进到约 step 5060，速率恢复到约 `2.2 s/step`，预计 pretraining 剩余约 3 小时 3 分；GPU 1 为 `81,318 MiB / 100%`，清理后共享盘可用 `212,592,283,648` bytes，错误计数仍为 0。

## 2026-08-20 12:03 CST｜step-5000 提交后的存活复核

- supervisor、training pipeline、`uv` 与训练 Python 四级进程链全部存活，run status 仍为 `running`。训练已从约 step 5060 推进到约 step 5240，速率稳定在约 `2.2 s/step`，预计 pretraining 剩余约 2 小时 57 分。
- attempt 6 起始标记后的 `Traceback`、`RESOURCE_EXHAUSTED` 与 OOM 计数仍为 0；stdout 指标仍为已审计的 180 条（step 0--4475），没有新的缓冲刷新，因此不重复导出或外推曲线。
- checkpoint 根目录稳定只含已验证的 `5000/`，没有临时目录。本实验仍只占用 GPU 1，快照为 `81,318 MiB / 100%`；主机可用内存 `406,062,087,168` bytes，共享盘可用 `212,153,311,232` bytes，保持单卡约束与资源安全线。

## 2026-08-20 12:39 CST｜step-6000 checkpoint 第六次滚动提交验证

- attempt 6 从约 step 5240 连续推进到 step 6000，当前 attempt 没有 `Traceback`、`RESOURCE_EXHAUSTED` 或 OOM。12:30:54 CST 触发第六次 Orbax async save；checkpoint manager blocking 阶段为 `320.2873` 秒，是目前最长的单次观测值，但当前不对共享存储 I/O 差异作未经测量的归因。训练于 12:36:14 CST 恢复计算，后台线程继续写盘。
- 保存过程中旧 `5000/` 始终可见，新点写入 `6000.orbax-checkpoint-tmp-20/`。20 秒监控采样中，12:36:49 CST 的临时目录为 `40,698,541,787` bytes，主机可用内存 `335,423,326,208` bytes，共享盘可用 `169,865,744,384` bytes；这是离散采样中的最低共享盘观测值，仍高于 150 GB 安全线，但不表述为连续窗口的理论最低值。
- 12:37:08 CST，Orbax 依次完成 `assets`、`params`、`train_state` 与根目录的原子重命名，随后报告 `Background save thread done` 和 `No errors found`；旧 step 5000 的实际删除于 12:37:11 CST 完成。最终根目录只含 `6000/` 且无临时目录，第六次验证新点提交后完成旧点回收的滚动行为。
- step-6000 checkpoint 共 43 个文件、`42,947,359,114` bytes 文件 payload：`assets` 1 个/4,496 bytes、`params` 17 个/12,014,087,912 bytes、`train_state` 24 个/30,933,266,289 bytes，另含顶层 metadata；目录表观大小为 `42,947,404,170` bytes。metadata commit timestamp 为 `1787200628702839216`，SHA-256 为 `c4ff1a9a52c4620958fb45e6b7bd8a15290276ebc6422df0ac2a56014b9b9ddf`。
- 隐藏 GPU 后，项目主环境的 CPU 只读 `CheckpointManager` 返回 `all_steps=[6000]`、`latest_step=6000`。完整数组 restore 仍留到最终 checkpoint；机器可读验证详情保存于 `experiments/pretrain-checkpoint-6000-validation.json`。
- 12:38 CST 四级进程链仍存活，训练已推进到约 step 6050，稳定速率约 `2.2 s/step`，预计 pretraining 剩余约 2 小时 26 分；GPU 1 为 `81,318 MiB / 100%`，清理后主机可用内存 `396,055,832,576` bytes、共享盘可用 `210,566,705,152` bytes，错误计数仍为 0。stdout 指标仍为已审计的 180 条（step 0--4475），没有新缓冲刷新，故本轮不重复导出或外推曲线。

## 2026-08-20 12:43 CST｜step-6000 提交后的存活复核

- supervisor、training pipeline、`uv` 与训练 Python 四级进程链全部存活，run status 仍为 `running`。训练已从约 step 6050 推进到约 step 6180，速率稳定在约 `2.2 s/step`，预计 pretraining 剩余约 2 小时 21 分。
- attempt 6 起始标记后的 `Traceback`、`RESOURCE_EXHAUSTED` 与 OOM 计数仍为 0；stdout 指标仍为已审计的 180 条（step 0--4475），文件行数、字节数与 SHA-256 均未变化，因此不重复导出或外推曲线。
- checkpoint 根目录稳定只含已验证的 `6000/`，没有临时目录。本实验仍只占用 GPU 1，快照为 `81,318 MiB / 100%`；主机可用内存 `396,341,927,936` bytes，共享盘可用 `205,905,022,976` bytes，保持单卡约束与 150 GB 资源安全线。

## 2026-08-20 13:03 CST｜batch-16 pretraining 增量监控

- supervisor、training pipeline、`uv` 与训练 Python 四级进程链全部存活，supervisor stage 为 `training_pipeline`，run status 仍为 `running`。训练已推进到约 step 6710，速率稳定在约 `2.2 s/step`，预计 pretraining 剩余约 2 小时 2 分。
- attempt 6 起始标记后的 `Traceback`、`RESOURCE_EXHAUSTED` 与 OOM 计数仍为 0；stdout 指标仍为已审计的 180 条（step 0--4475），文件行数、字节数与 SHA-256 均未变化，因此没有新数值可供趋势外推。
- checkpoint 根目录仍只含已验证的 `6000/`，没有临时目录；下一次正常保存点是 step 7000。本实验继续只占用 GPU 1，快照为 `81,318 MiB / 100%`；主机可用内存 `397,318,758,400` bytes，共享盘可用 `205,903,511,552` bytes，资源保持在安全范围。

## 2026-08-20 13:25 CST｜step-7000 checkpoint 与 67.25% 指标审计

- attempt 6 于 13:13:26 CST 触发 step-7000 Orbax async save。checkpoint manager blocking 阶段为 `239.0422` 秒；训练在 13:17:25 CST 恢复推进，短暂 ETA 波动后重新稳定在约 `2.2 s/step`。当前 attempt 的 `Traceback`、`RESOURCE_EXHAUSTED` 与 OOM 计数仍为 0。
- 13:18:05 CST，Orbax 依次完成 `assets`、`params`、`train_state` 和根临时目录的原子重命名，随后报告 `Background save thread done`、`No errors found`；旧 step 6000 于 13:18:06 CST 完成实际删除。最终根目录只含 `7000/` 且无临时目录，第七次验证 `max_to_keep=1` 的滚动恢复点。心跳在提交完成后到达，本轮没有对重叠窗口进行离散采样，因此不补造最低共享盘余量。
- step-7000 checkpoint 共 46 个文件、`42,953,023,549` bytes 文件 payload：`assets` 1 个/4,496 bytes、`params` 18 个/12,014,108,876 bytes、`train_state` 26 个/30,938,909,760 bytes，另含顶层 metadata；目录表观大小为 `42,953,068,605` bytes。metadata commit timestamp 为 `1787203085302154892`，SHA-256 为 `b21d5f8cfe32edb8a6f5986fae9f640c74b9aaa12fa90c36e9bcd6f2d5c03cf8`。
- 隐藏 GPU 并显式设置 CPU backend 后，项目主环境的只读 `CheckpointManager` 返回 `all_steps=[7000]`、`latest_step=7000`。第一次只隐藏 GPU 而未设置 `JAX_PLATFORMS=cpu` 时，JAX 因无可见 CUDA 设备拒绝初始化；补充 CPU backend 后验证通过，该诊断命令没有改动 checkpoint。完整数组 restore 仍留到最终 checkpoint。机器可读详情保存于 `experiments/pretrain-checkpoint-7000-validation.json`。
- 保存使 stdout 缓冲由 180 条刷新到 270 条；重新导出的 `metrics.jsonl` 为 270 行、29,270 bytes，SHA-256 为 `87ff1d528e3a8365f9ea63d2423c0af693a6e3c0d3601b933721ba01396011a8`，覆盖 step 0--6725 且间隔严格为 25 step。`loss`、`grad_norm`、`param_norm` 与 `vision_param_norm` 全部有限；loss 首末为 `0.1572 / 0.0324`，最近 10 条均值为 `0.03104`。step 4500 起的完整 500-step loss 均值为 `0.03316 / 0.03308 / 0.03261 / 0.03157`，支持持续缓慢改善，但覆盖率仅 67.25%，不提前宣称最终收敛。机器可读分析保存于 `experiments/pretrain-metrics-through-step-6725.json`。
- 13:24 CST 四级进程链仍存活，训练已推进到约 step 7200，预计 pretraining 剩余约 1 小时 44 分；本实验仍只占用 GPU 1，快照为 `81,318 MiB / 100%`。清理后主机可用内存 `382,295,557,120` bytes、共享盘可用 `205,881,110,528` bytes，资源保持在安全范围。

## 2026-08-20 13:43 CST｜step-7000 提交后的存活复核

- supervisor、training pipeline、`uv` 与训练 Python 四级进程链全部存活，run status 仍为 `running`。训练已从约 step 7200 推进到约 step 7680，速率稳定在约 `2.2 s/step`，预计 pretraining 剩余约 1 小时 26 分。
- attempt 6 起始标记后的 `Traceback`、`RESOURCE_EXHAUSTED` 与 OOM 计数仍为 0；stdout 与导出指标均保持已审计的 270 条（step 0--6725），文件行数、字节数与 SHA-256 未变化，因此不重复分析。
- checkpoint 根目录稳定只含已验证的 `7000/`，没有临时目录。本实验仍只占用 GPU 1，快照为 `81,318 MiB / 100%`；主机可用内存 `380,684,354,560` bytes、共享盘可用 `205,880,254,464` bytes，保持单卡约束与资源安全线。下一正常恢复点为 step 8000。

## 2026-08-20 14:05 CST｜step-8000 checkpoint 第八次滚动提交验证

- attempt 6 于 13:54:33 CST 触发 step-8000 Orbax async save。checkpoint manager blocking 阶段为 `377.8055` 秒，是目前最长的单次观测值；这里只记录实测，不对共享存储 I/O 差异作未经测量的归因。训练在 14:00:51 CST 恢复推进，当前 attempt 的 `Traceback`、`RESOURCE_EXHAUSTED` 与 OOM 计数仍为 0。
- 14:02:09 CST，Orbax 依次完成 `assets`、`params`、`train_state` 和根临时目录的原子重命名，随后报告 `Background save thread done`、`No errors found`；旧 step 7000 于 14:02:10 CST 完成实际删除。最终根目录只含 `8000/` 且无临时目录，第八次验证 `max_to_keep=1` 的滚动恢复点。心跳在提交完成后到达，本轮没有对重叠窗口进行离散采样，因此不补造最低共享盘余量。
- step-8000 checkpoint 共 45 个文件、`42,973,136,012` bytes 文件 payload：`assets` 1 个/4,496 bytes、`params` 18 个/12,014,103,419 bytes、`train_state` 25 个/30,959,027,680 bytes，另含顶层 metadata；目录表观大小为 `42,973,181,068` bytes。metadata commit timestamp 为 `1787205729051690605`，SHA-256 为 `13df2491302c6d8d636241c6e0fe40440f05430e8b2348dec9d693cc7c079762`。
- 隐藏 GPU 并显式设置 CPU backend 后，项目主环境的只读 `CheckpointManager` 返回 `all_steps=[8000]`、`latest_step=8000`。完整数组 restore 仍留到最终 checkpoint；机器可读详情保存于 `experiments/pretrain-checkpoint-8000-validation.json`。
- 14:04 CST 四级进程链仍存活，训练已推进到约 step 8060、速率约 `2.2 s/step`，预计 pretraining 剩余约 1 小时 12 分；GPU 1 为 `81,318 MiB / 100%`。清理后主机可用内存 `357,005,002,752` bytes、共享盘可用 `205,847,683,072` bytes。stdout metrics 仍为已审计的 270 条（step 0--6725），没有新缓冲刷新，故本轮不重复导出或外推曲线。

## 2026-08-20 14:23 CST｜step-8000 提交后的存活复核

- supervisor、training pipeline、`uv` 与训练 Python 四级进程链全部存活，run status 仍为 `running`。训练已从约 step 8060 推进到约 step 8570，速率稳定在约 `2.2 s/step`，预计 pretraining 剩余约 53 分钟。
- attempt 6 起始标记后的 `Traceback`、`RESOURCE_EXHAUSTED` 与 OOM 计数仍为 0；stdout 与导出 metrics 均保持已审计的 270 条（step 0--6725），行数、字节数及 SHA-256 均未变化，因此不重复分析。
- checkpoint 根目录稳定只含已验证的 `8000/`，没有临时目录。本实验仍只占用 GPU 1，快照为 `81,318 MiB / 100%`；主机可用内存 `355,178,439,680` bytes、共享盘可用 `205,081,165,824` bytes，保持单卡约束与资源安全线。下一正常恢复点为 step 9000。

## 2026-08-20 14:48 CST｜服务器 SSH 连接暂时不可达

- 本轮从 14:42 CST 起尝试读取 step-9000 保存窗口，但 `hgpu` 的 Tailscale 地址 `100.125.76.83:22` 连续多次在 10--60 秒连接窗口内超时；ICMP 诊断同样没有响应。故当前只确认监控通道不可达，不能据此把训练判定为失败、停止或完成。
- 最后一次成功取得的服务器证据为 14:23:29 CST：attempt 6 约 step 8590、`2.2 s/step`、当前 attempt 错误计数 0，四级进程链存活，checkpoint 根目录仅含已验证的 `8000/`。按该速度，step 9000 原本预计在约 14:39 CST 触发，但由于缺少当前日志，暂不把预期写成结果。
- 本地 `research-state.yaml` 已改为明确的“监控中断、当前服务器状态未知”，没有改动训练配置、checkpoint 或远端进程。服务器恢复连接后，下一步首先检查 step 9000 是否原子提交、旧 step 8000 是否回收、metrics 是否刷新及单卡资源是否安全；本条本地记录届时再同步到服务器项目目录。

## 2026-08-20 16:02 CST｜step-9000 主机 OOM 诊断与 attempt 7 恢复启动

- SSH 恢复后确认 attempt 6 已在 15:14:51 CST 以 return code 137 停止。systemd 在 15:14:37 CST 明确记录 `session-389188.scope` 中的进程被 OOM killer 终止；同一窗口还记录约 11 分钟的主机 time jump，这解释了此前 Tailscale/SSH 监控中断。机器没有重启，诊断窗口没有 NVIDIA Xid、ext4 或 `/dev/sdb` I/O error，因此把本次故障归类为 step-9000 checkpoint 保存阶段的主机 OOM，而不是模型数值失败或 GPU OOM。
- 训练在保存入口前已推进到 step 9000；日志新增的 step 6750--8975 指标全部为有限值。故障现场 `9000.orbax-checkpoint-tmp-32/` 只有 18 个文件、`12,014,142,618` bytes，最后修改停在 15:39:08 CST；只有临时 `assets` 与 `params`，没有完整 `train_state`、finalize 或根目录原子重命名，故不能作为 checkpoint。已验证的 `8000/` 仍完整且是 Orbax 唯一可发现恢复点。
- 锁定恢复协议 `RETAIN-GPU-20260819-001-R1`：从 step 8000 恢复完整 optimizer state，把 pretraining operational save interval 调整为 10,000 以跳过 step 9000 中间全量保存；最终 step 9999 仍强制保存，但只持久化 downstream 所需的 EMA/inference params 与 assets。模型、数据、mixture、batch-16 单卡缩小协议、seed、总步数、LR、AdamW、clip 与 EMA 均不变。
- 新增 params-only checkpoint 单元测试并在服务器通过：Python compile、Ruff 关键错误规则、config assertions 和 pytest `1/1` 全部成功。恢复协议与实现分别提交为 `d97ddce` 和 `caf25b4`，满足先锁定协议再执行的时序要求。
- 删除不可恢复临时分片前，已把根 metadata、params manifest/sharding、失败状态与完整日志复制到 `experiments/pretrain-step9000-host-oom-recovery-20260820/raw-evidence/` 并记录 SHA-256；随后精确删除该 12.0 GB 临时目录。原始大块临时参数不可恢复，但没有任何可用 train state；共享盘可用空间由 `182,942,547,968` 回升至 `194,953,646,080` bytes，checkpoint 根重新只含完整 `8000/`。
- 16:02:20 CST，attempt 7 总控以独立 `retain-reproduction.service` 启动，避免 SSH/Tailscale 会话关闭连带终止；仍由原 pipeline 严格选择至多一张空闲 GPU。当前正在重跑 353-file SHA-256 与后续权重/输入门禁，尚未抢占 GPU；门禁通过后将从 step 8000 继续。

## 2026-08-20 16:11 CST｜attempt 7 单卡等待监控

- attempt 7 已重跑并通过 353-file SHA-256、24-object GCS size/MD5、基础权重 Orbax restore 与 7/7 真实输入 smoke tests；`retain-reproduction.service` 和 training pipeline 两级进程持续存活，supervisor stage 为 `training_pipeline`。
- 16:05--16:10 CST 的逐分钟轮询均报告无空闲 GPU；当前 8 卡显存为 `17,919 / 15,452 / 49,314 / 18,102 / 61,354 / 61,397 / 61,277 / 61,051 MiB`，利用率为 `100 / 31 / 56 / 100 / 97 / 97 / 88 / 97%`。没有卡同时满足显存不超过 2,048 MiB 且利用率不超过 10% 的安全阈值，因此没有运行 GPU preflight 或启动 attempt 7 训练。
- checkpoint 根目录仍只含已验证的 `8000/`，没有新的临时目录。主机可用内存 `539,785,551,872` bytes，共享盘可用 `194,589,081,600` bytes；当前是安全等待，不是新失败，也未占用 GPU。

## 2026-08-20 16:31 CST｜attempt 7 已从 step 8000 恢复并单卡运行

- 16:13:42 CST，pipeline 观测到 GPU 3 为空闲卡（82 MiB、0%），随后严格 JAX GPU preflight 通过：`platform=gpu`、`device_kind=NVIDIA RTX 6000D`、可见设备数为 1。训练进程环境明确为 `CUDA_VISIBLE_DEVICES=3`，进程只出现在 GPU 3；当前占用约 81,296 MiB，符合一块 GPU 的约束。
- 16:13:46 CST 启动 attempt 7。Orbax 读取 step-8000 metadata，报告父目录中恰有 1 个 checkpoint，并从完整 `assets/params/train_state` 恢复；训练随后从 step 8000 继续，而不是重新初始化 optimizer state。当前进程链为 systemd supervisor、pipeline、`uv` 和训练 Python，全部存活。
- 16:31 CST 最新有限指标为 step 8450：`loss=0.0272`、`grad_norm=0.2840`、`param_norm=1380.9456`、`vision_param_norm=1258.5623`；进度约 step 8460，稳定速率约 `2.2 s/step`，预计约 57 分钟到 step 9999。只扫描当前 attempt 日志片段，未见 `Traceback`、`RESOURCE_EXHAUSTED`、OOM 或 `Killed`。
- checkpoint 根目录仍只有已验证的 `8000/`，且无 Orbax 临时目录，说明 R1 协议已跳过 step 9000 的中间全量保存入口。当前主机可用内存 `487,777,177,600` bytes，共享盘可用 `194,586,353,664` bytes；继续监控最终 params-only 保存及其可恢复性验证。

## 2026-08-20 16:52 CST｜R1 协议跨过 step 9000 验证

- attempt 7 在 GPU 3 上继续以约 `2.2 s/step` 运行，并于 16:52 CST 产生 step 9000 指标：`loss=0.0298`、`grad_norm=0.3472`、`param_norm=1381.1146`、`vision_param_norm=1258.5836`，四项均为有限值。当前 attempt 日志片段没有 `Traceback`、`RESOURCE_EXHAUSTED`、OOM 或 `Killed`，systemd 同期也没有新的 OOM-killer 记录。
- 跨过 step 9000 后，checkpoint 根目录仍精确只含已验证的 `8000/`，没有 `9000/`、Orbax 临时目录或保存事件日志。这直接验证 R1 的 `save_interval=10000` 已跳过导致 attempt 6 主机 OOM 的 step-9000 全量 checkpoint，而训练没有停顿或重启。
- 单卡约束保持：训练环境仍为 `CUDA_VISIBLE_DEVICES=3`，GPU 3 快照为 `81,390 MiB / 100%`。主机可用内存 `485,572,166,656` bytes，共享盘可用 `194,577,379,328` bytes，资源稳定；预计约 37 分钟到 step 9999，届时重点监控 params-only 最终保存、旧 step 8000 回收和后续参数恢复。

## 2026-08-20 17:11 CST｜attempt 7 最后 5% 监控

- supervisor、training pipeline、`uv` 与训练 Python 四级进程链持续存活，run status 为 `running`。attempt 7 已推进到 step 9500，稳定速率约 `2.2 s/step`，预计约 18 分钟到 step 9999；当前日志片段和 systemd 均没有新增 `Traceback`、`RESOURCE_EXHAUSTED`、OOM 或 OOM-killer 事件。
- step 9500 的有限指标为 `loss=0.0284`、`grad_norm=0.3159`、`param_norm=1381.2675`、`vision_param_norm=1258.6046`。checkpoint 根目录仍只有 `8000/` 且无临时目录，符合跳过所有中间保存、只在最终 step 9999 写出 params-only checkpoint 的恢复协议。
- 训练仍只占用 GPU 3，快照为 `81,390 MiB / 100%`。主机可用内存 `483,335,633,920` bytes，共享盘可用 `194,207,358,976` bytes；资源安全且无需调整，继续监控最终保存与下游 pipeline 切换。

## 2026-08-20 17:33 CST｜pretraining 完成、最终参数验证并推进 stove Task-FT

- attempt 7 于 17:29:48 CST 进入 step-9999 终态保存。checkpoint manager blocking 阶段为 `64.5740` 秒，只向主机传输约 12.1 GiB params 和极小空占位 train-state，而不是此前约 43 GB 的完整 optimizer checkpoint；主机没有触发 OOM。17:31:12 CST，Orbax 完成 assets、params 和根目录原子重命名，后台线程报告 `No errors found`；旧 step 8000 于 17:31:16 CST 删除，pretraining 于 17:31:32 CST 以 return code 0 完成。
- 按 params-only 协议清理空占位 optimizer 目录后，最终 `9999/` 保留 21 个文件、`12,014,118,262` bytes 文件 payload：metadata 417 bytes、assets 1 个/4,496 bytes、params 19 个/`12,014,113,349` bytes。metadata commit timestamp 为 `1787218272370695972`，SHA-256 为 `369f8f53253b6b6f5c4686bd5e01f053a17e01762c9c8848d41ad198308dd79f`，无临时目录。该产物可供参数合并和初始化，但按设计不包含可恢复 optimizer state。
- 完成后导出的最终 `metrics.jsonl` 为 400 条、43,369 bytes，SHA-256 为 `bdff06c0cbfabcbf130fa4a3726ac85bb2e7a4883153c6304f8eae65f9622004`，覆盖 step 0--9975 且间隔固定 25 step，四项指标全部有限。loss 从 `0.1572` 降至 `0.0276`，首末下降 82.44%，最近 10 条均值 `0.02795`；step 8000 起的四个 500-step loss 均值为 `0.029495 / 0.029225 / 0.028615 / 0.028385`。最近 10 条 grad norm 均值为 `0.31078`，最终 param/vision param norm 为 `1381.4045 / 1258.6207`，支持恢复接续平滑且后段数值稳定。机器可读报告为 `experiments/pretrain-final-metrics-attempt7.json`。
- pipeline 于 17:31:32 CST 自动推进 `retain_repro_task_ft_stove`，继续只使用 GPU 3。该阶段从最终 `9999/params` 实际读取日志所示 12.1 GiB 权重，14 秒后报告 restore 完成，并在 17:33 CST 产生有限 step-0 指标：`loss=0.0383`、`grad_norm=0.6916`、`param_norm=1381.3778`、`vision_param_norm=1258.6182`。因此最终 params-only checkpoint 已由真实下游训练消费验证；当前 stove Task-FT 正常运行，后续由同一 pipeline 顺序推进其余 Task-FT 与 coFT。

## 2026-08-20 17:53 CST｜stove Task-FT 完成并推进 mugs Task-FT

- stove Task-FT 在 GPU 3 上以约 `2.2 s/step` 完成 500 steps，当前日志和 systemd 均无 `Traceback`、`RESOURCE_EXHAUSTED`、OOM 或 OOM-killer 事件。17:51:31 CST 触发 step-499 params-only 保存，checkpoint manager blocking 仅 `3.8166` 秒；17:51:52 CST 完成根目录原子提交，后台线程报告 `No errors found`，阶段于 17:52:06 CST 以 return code 0 完成。
- 清理空 optimizer 占位目录后，`task_ft_stove/.../499/` 保留 21 个文件、`12,014,114,437` bytes 文件 payload：metadata 417 bytes、assets 4,496 bytes、params 19 个/`12,014,109,524` bytes。metadata commit timestamp 为 `1787219512180575452`，SHA-256 为 `64bea9f926861389263c61fba1947d8bb0757a9e5433a776ad1f4c0632ba2477`，无临时目录；该终态供 RETAIN 合并和评测使用，不用于 optimizer resume。
- 最终 `metrics.jsonl` 为 20 条、2,146 bytes，SHA-256 为 `69fcee4f995227c123bbc3f3622f43cab69c41fb94191a8e606e1604380fcd31`，覆盖 step 0--475 且间隔固定 25 step，四项指标全部有限。loss 首末为 `0.0383 / 0.0264`（下降 31.07%），前/后 10 条均值为 `0.03054 / 0.02644`；grad norm 前/后 10 条均值为 `0.31179 / 0.27667`，param/vision param norm 只从 `1381.3778 / 1258.6182` 缓慢变至 `1381.3918 / 1258.6208`。机器可读报告为 `experiments/task-ft-stove-final-metrics.json`。
- pipeline 于 17:52:06 CST 自动启动 mugs Task-FT，仍只使用 GPU 3。它已从 pretraining `9999/params` 完整 restore 并完成 weights loading；step 0 指标为 `loss=0.0366`、`grad_norm=0.3346`、`param_norm=1381.3778`、`vision_param_norm=1258.6182`，全部有限。17:53 CST 资源快照为 GPU 3 `81,394 MiB / 100%`、主机可用内存 `486,391,454,720` bytes、共享盘可用 `190,167,547,904` bytes，串行单卡流水线正常。

## 2026-08-20 18:12 CST｜mugs Task-FT 中点监控

- `retain-reproduction.service`、supervisor、training pipeline、`uv` 与训练 Python 进程链全部存活，mugs Task-FT 已推进到约 step 498/1000，稳定速率约 `2.2 s/step`，预计约 19 分钟后完成并由 pipeline 自动推进下一阶段。
- 最新记录指标为 step 475：`loss=0.0233`、`grad_norm=0.2376`、`param_norm=1381.3929`、`vision_param_norm=1258.6210`，全部有限；从 step 0 到 step 475 未见 `Traceback`、`RESOURCE_EXHAUSTED`、CUDA OOM、`Killed` 或 systemd OOM-killer 事件。当前只做在线数值健康检查，完整 40 条指标统计留到 step 999 成功提交后执行。
- 训练仍严格只使用 GPU 3，快照为 `81,394 MiB / 100%`。主机可用内存 `483,584,545,792` bytes，共享盘可用 `171,545,821,184` bytes；共享盘虽为 98% 使用率，但仍高于既定 150 GB 安全线，且本阶段最终 params-only checkpoint 约 12 GB，当前无需缩小或中断实验。

## 2026-08-20 18:34 CST｜mugs Task-FT 完成并推进 basket Task-FT

- mugs Task-FT 在 GPU 3 上以约 `2.2 s/step` 完成 1,000 steps。18:30:36 CST 触发 step-999 params-only 保存，checkpoint manager blocking 为 `5.4223` 秒；18:30:58 CST 完成根目录原子提交，后台线程报告 `No errors found`，阶段于 18:31:13 CST 以 return code 0 完成。当前日志和 systemd 均无 `Traceback`、`RESOURCE_EXHAUSTED`、OOM 或 OOM-killer 事件。
- 清理空 optimizer 占位目录后，`task_ft_mugs/.../999/` 保留 21 个文件、`12,014,111,959` bytes 文件 payload：metadata 417 bytes、assets 4,496 bytes、params 19 个/`12,014,107,046` bytes。metadata commit timestamp 为 `1787221858958596821`，SHA-256 为 `679c04a3ad0e62c48948c544ae6ec5c2a441855c1b9f34596174679ae86ae852`，无临时目录；终态供 RETAIN 合并和评测使用，不承诺 optimizer resume。
- 最终 `metrics.jsonl` 为 40 条、4,296 bytes，SHA-256 为 `512a96dbc5f3913aca94910ffbf2847331f72a6df0286e3261e90480e7bdd6f1`，覆盖 step 0--975 且固定间隔 25 step，四项指标全部有限。loss 从 `0.0366` 降至 `0.0251`（下降 31.42%），前/后 10 条均值为 `0.03430 / 0.02516`，两个 500-step 区间均值为 `0.02966 / 0.025535`；grad norm 前/后 10 条均值为 `0.30114 / 0.30169`，param/vision param norm 由 `1381.3778 / 1258.6182` 缓慢变至 `1381.4934 / 1258.6355`。机器可读报告为 `experiments/task-ft-mugs-final-metrics.json`。
- pipeline 于 18:31:13 CST 自动启动 basket Task-FT，仍只使用 GPU 3；12.1 GiB pretraining final params 已完整 restore 并完成 loading。step 0 指标为 `loss=0.0454`、`grad_norm=0.3618`、`param_norm=1381.3778`、`vision_param_norm=1258.6182`，全部有限；18:34 CST 已推进至约 step 23/500，稳定速率约 `2.2 s/step`，预计约 18 分钟后完成并自动推进 coFT。
- 18:34 CST 资源快照为 GPU 3 `81,394 MiB / 100%`、主机可用内存 `486,963,006,464` bytes、共享盘可用 `159,004,590,080` bytes。basket 终态只新增约 12 GB params-only checkpoint，当前余量足够；后续 coFT checkpoint 与 rollout 结果仍需持续监控磁盘占用。

## 2026-08-20 18:54 CST｜三项 Task-FT 全部完成并推进 stove coFT

- basket Task-FT 在 GPU 3 上以约 `2.2 s/step` 完成 500 steps。18:52:12 CST 触发 step-499 params-only 保存，checkpoint manager blocking 为 `24.3119` 秒；18:52:54 CST 完成根目录原子提交，后台线程报告 `No errors found`，阶段于 18:53:07 CST 以 return code 0 完成。当前日志和 systemd 均无 `Traceback`、`RESOURCE_EXHAUSTED`、OOM 或 OOM-killer 事件。
- 清理空 optimizer 占位目录后，`task_ft_basket/.../499/` 保留 23 个文件、`12,014,104,227` bytes 文件 payload：metadata 417 bytes、assets 4,496 bytes、params 21 个/`12,014,099,314` bytes。metadata commit timestamp 为 `1787223174072219579`，SHA-256 为 `85cd4b01741bb63c204d026e4a5ac9488f6469b88d02f69f2f334625acfeee67`，无临时目录；终态供 RETAIN 合并和评测使用，不承诺 optimizer resume。
- 最终 `metrics.jsonl` 为 20 条、2,147 bytes，SHA-256 为 `d96e41df0fbc51b74becaf68dff6ce6b8e03083d5ac266eb37ccb0cf51f33fd7`，覆盖 step 0--475 且固定间隔 25 step，四项指标全部有限。loss 从 `0.0454` 降至 `0.0308`（下降 32.16%），前/后 10 条均值为 `0.03830 / 0.03226`；grad norm 前/后 10 条均值为 `0.27715 / 0.25872`，param/vision param norm 仅由 `1381.3778 / 1258.6182` 缓慢变至 `1381.3939 / 1258.6211`。机器可读报告为 `experiments/task-ft-basket-final-metrics.json`。至此三个目标任务的 Task-FT 均已完成并有独立 checkpoint 与指标报告。
- pipeline 于 18:53:07 CST 自动启动 stove coFT，仍只使用 GPU 3。该配置实际构建了目标 stove 与四个 generalist suites 的联合训练 mixture，从 pretraining `9999/params` 完整 restore 并完成 loading；step 0 指标为 `loss=0.0382`、`grad_norm=0.4675`、`param_norm=1381.3778`、`vision_param_norm=1258.6182`，全部有限。18:54 CST 初始进度约 step 3/1000，首步 ETA 约 55 分钟，后续继续按稳定速率复核。
- 18:54 CST 资源快照为 GPU 3 `81,394 MiB / 100%`、主机可用内存 `485,947,230,208` bytes、共享盘可用 `146,966,577,152` bytes。余量已低于 supervisor 启动时采用的 150 GB 保守门槛，但三个 coFT 终态均为约 12 GB params-only checkpoint，预计训练产物总增量约 36 GB，当前足以完成串行训练；进入大规模 rollout 前必须再次审计并控制结果存储。

## 2026-08-20 19:11 CST｜stove coFT 中点监控

- `retain-reproduction.service`、supervisor、training pipeline、`uv` 与 stove coFT 训练 Python 进程链全部存活。训练已从 step 3 推进到约 step 458/1000，稳定速率约 `2.2 s/step`，预计约 20 分钟后完成并由 pipeline 自动推进 mugs coFT。
- 最新记录指标为 step 450：`loss=0.0296`、`grad_norm=0.2952`、`param_norm=1381.3896`、`vision_param_norm=1258.6200`，均为有限值；当前日志没有 `Traceback`、`RESOURCE_EXHAUSTED`、CUDA OOM 或 `Killed`，systemd 同期也没有 OOM-killer 事件。完整 40 条指标统计留到 step 999 checkpoint 成功提交后执行。
- 训练继续严格只使用 GPU 3，快照为 `81,394 MiB / 100%`。主机可用内存 `483,618,980,864` bytes，共享盘可用 `146,956,156,928` bytes；自 coFT 启动以来磁盘余量基本稳定，终态 params-only 保存尚未开始，当前无需调整训练。

## 2026-08-20 19:36 CST｜stove coFT 完成并推进 mugs coFT

- stove coFT 在 GPU 3 上完成 1,000 steps。19:31:20 CST 触发 step-999 params-only 保存，checkpoint manager blocking 为 `37.8381` 秒；19:32:16 CST 完成根目录原子提交，后台线程无错误，阶段于 19:32:31 CST 以 return code 0 完成。日志与 systemd 均未出现 `Traceback`、`RESOURCE_EXHAUSTED`、CUDA OOM、`Killed` 或 OOM-killer 事件。
- 清理空 optimizer 占位目录后，`retain_repro_coft_stove/paper_coft_stove/999/` 保留 20 个文件、`12,014,126,607` bytes 文件 payload：metadata 417 bytes、assets 4,496 bytes、params 18 个/`12,014,121,694` bytes。metadata commit timestamp 为 `1787225536034538721`，SHA-256 为 `2448bfab2231a60fedef826b8781d5e7f2ec0c115f62e2f7d6c92fe2bd6a3897`，无临时目录；终态供 coFT 基线评测使用，不用于 optimizer resume。
- 最终 `metrics.jsonl` 为 40 条、4,299 bytes，SHA-256 为 `8df334011ecb44f7b8be6ebb2179efc33c12c30513ed90cf436cb8b25ec848bf`，覆盖 step 0--975 且固定间隔 25 step，四项指标全部有限。loss 从 `0.0382` 降至 `0.0293`（下降 23.30%），前/后 10 条均值为 `0.02956 / 0.02821`，两个 500-step 区间均值为 `0.028410 / 0.027945`；grad norm 前/后 10 条均值为 `0.26994 / 0.32535`，param/vision param norm 由 `1381.3778 / 1258.6182` 缓慢变至 `1381.5062 / 1258.6365`。机器可读报告为 `experiments/coft-stove-final-metrics.json`。
- pipeline 于 19:32:31 CST 自动启动 mugs coFT，仍只使用 GPU 3；已从 pretraining `9999/params` 完整 restore 12.1 GiB 权重并完成 loading。step 0/25 指标分别为 `loss=0.0353 / 0.0429`、`grad_norm=0.2532 / 0.3550`，参数范数均有限，当前无异常。19:35 CST 资源快照为 GPU 3 `81,394 MiB / 100%`、主机可用内存 `486,284,894,208` bytes、共享盘可用 `134,576,037,888` bytes。预计剩余两个 coFT checkpoint 还会占用约 24 GB；训练可继续，但 rollout 评测前必须再审计并限制视频/轨迹中间产物。

## 2026-08-20 19:52 CST｜mugs coFT 中点监控

- `retain-reproduction.service`、supervisor、training pipeline、`uv` 与 mugs coFT 训练 Python 进程链全部存活。训练已推进至 step 450/1000，稳定速率约 `2.2 s/step`，预计约 20 分钟后进入终态保存，并由 pipeline 自动推进 basket coFT。
- 最新 step 450 指标为 `loss=0.0268`、`grad_norm=0.2619`、`param_norm=1381.3901`、`vision_param_norm=1258.6205`，全部有限；step 100--450 的周期指标未显示异常跳变。当前 stdout 未见 `Traceback`、`RESOURCE_EXHAUSTED`、CUDA OOM、`Killed` 或后台保存错误，内核日志也无新增 OOM-killer 记录；checkpoint 目录尚未创建，符合只在最终 step 999 保存 params 的策略。
- 训练继续严格只使用 GPU 3，快照为 `81,394 MiB / 100%`。主机可用内存 `483,514,496,000` bytes，共享盘可用 `134,569,009,152` bytes，较 19:35 CST 基本稳定；当前无需修复或缩小实验，继续监控最终 checkpoint 原子提交与磁盘余量。

## 2026-08-20 20:16 CST｜mugs coFT 完成并推进 basket coFT

- mugs coFT 在 GPU 3 上完成 1,000 steps。20:11:40 CST 触发 step-999 params-only 保存；本次约 12.1 GiB 参数的 host transfer 较慢，checkpoint manager blocking 为 `83.1304` 秒。20:13:21 CST 完成根目录原子提交，后台线程报告 `No errors found`，阶段于 20:13:36 CST 以 return code 0 完成；日志与内核均无 `Traceback`、`RESOURCE_EXHAUSTED`、CUDA OOM、`Killed` 或 OOM-killer 事件。
- 清理空 optimizer 占位目录后，`retain_repro_coft_mugs/paper_coft_mugs/999/` 保留 20 个文件、`12,014,129,184` bytes 文件 payload：metadata 417 bytes、assets 4,496 bytes、params 18 个/`12,014,124,271` bytes。metadata commit timestamp 为 `1787228001637404927`，SHA-256 为 `3515a5956c918223641dceb9de714a015caf2a6ea60572fdfb999b028f3bc574`，无临时目录；终态供 coFT 基线评测使用，不用于 optimizer resume。
- 最终 `metrics.jsonl` 为 40 条、4,303 bytes，SHA-256 为 `25eb3f16a10e699c0aebb2d85cd311dc6bcf601789768b30b3ef9595fae5b689`，覆盖 step 0--975 且固定间隔 25 step，四项指标全部有限。loss 从 `0.0353` 降至 `0.0296`（下降 16.15%），前/后 10 条均值为 `0.03130 / 0.02749`，两个 500-step 区间均值为 `0.028795 / 0.027305`；grad norm 前/后 10 条均值为 `0.26339 / 0.30867`，param/vision param norm 由 `1381.3778 / 1258.6182` 缓慢变至 `1381.5020 / 1258.6357`。机器可读报告为 `experiments/coft-mugs-final-metrics.json`。
- pipeline 于 20:13:36 CST 自动启动最后一项 basket coFT，仍只使用 GPU 3；已从 pretraining `9999/params` 完整 restore 12.1 GiB 权重并完成 loading。step 0 指标为 `loss=0.0332`、`grad_norm=0.3244`、`param_norm=1381.3778`、`vision_param_norm=1258.6182`，全部有限，20:15 CST 已推进至约 step 8/1000。20:16 CST 资源快照为 GPU 3 `81,394 MiB / 100%`、主机可用内存 `486,470,943,744` bytes、共享盘可用 `122,559,176,704` bytes；余量足够最后一个约 12 GB checkpoint，但评测启动前必须先审计 rollout 输出策略。

## 2026-08-20 20:31 CST｜basket coFT 中点前监控

- `retain-reproduction.service`、supervisor、training pipeline、`uv` 与 basket coFT 训练 Python 进程链全部存活。训练已推进至约 step 423/1000，稳定速率约 `2.2 s/step`，预计约 22 分钟后进入最后一个 coFT 终态保存。
- 最新记录 step 400 指标为 `loss=0.0286`、`grad_norm=0.2697`、`param_norm=1381.3875`、`vision_param_norm=1258.6196`，全部有限；当前 stdout 未见 `Traceback`、`RESOURCE_EXHAUSTED`、CUDA OOM、`Killed` 或后台保存错误，内核日志也无新增 OOM-killer 记录。checkpoint 目录尚未创建，符合只在最终 step 999 保存 params 的策略。
- 训练继续严格只使用 GPU 3，快照为 `81,394 MiB / 100%`。主机可用内存 `484,417,094,656` bytes，共享盘可用 `122,196,742,144` bytes；当前余量仍足够约 12 GB 终态 checkpoint，继续监控完成后的评测存储门禁。

## 2026-08-20 20:58 CST｜全部训练完成并启动 RETAIN alpha sweep

- basket coFT 在 GPU 3 上完成 1,000 steps。20:52:47 CST 触发 step-999 params-only 保存，checkpoint manager blocking 为 `25.8984` 秒；20:53:30 CST 完成根目录原子提交，后台保存线程无错误，阶段于 20:53:45 CST 以 return code 0 完成。清理空 optimizer 占位目录后，`retain_repro_coft_basket/paper_coft_basket/999/` 保留 21 个文件、`12,014,110,341` bytes 文件 payload：metadata 417 bytes、assets 4,496 bytes、params 19 个/`12,014,105,428` bytes。metadata commit timestamp 为 `1787230410747822086`，SHA-256 为 `9359cd62a4f3ab9912017c2f98d330c2c5a2986d67d9d57c740ea997469a2eb5`，无临时目录；终态供 coFT 基线评测使用，不用于 optimizer resume。
- 最终 `metrics.jsonl` 为 40 条、4,295 bytes，SHA-256 为 `639b31bd58ee7c72021cca75663f70e43b16198a8d0b48fd4e63ef828b9e9e04`，覆盖 step 0--975 且固定间隔 25 step，四项指标全部有限。loss 从 `0.0332` 变为 `0.0321`（首末下降 3.31%），最小/最大值为 `0.0286 / 0.0387`，前/后 10 条均值为 `0.03315 / 0.03137`；grad norm 前/后 10 条均值为 `0.25072 / 0.31807`，param/vision param norm 从 `1381.3778 / 1258.6182` 缓慢变为 `1381.5028 / 1258.6361`。机器可读报告为 `experiments/coft-basket-final-metrics.json`。至此 pretraining、三项 Task-FT 与三项 coFT 训练全部完成并通过独立指标与终态 checkpoint 审计。
- supervisor 于 20:53:45 CST 自动切换到 evaluation pipeline，并选择同一物理 GPU 3。首个候选策略为 `retain_taskft_stove_a010`：按仓库 `linear_interpolation` 语义，以 `[0.1, 0.9]` 合并 stove Task-FT 与 pretraining final params。policy server 的 `CUDA_VISIBLE_DEVICES=3`；仿真 evaluator 不暴露 CUDA compute，仅把 `MUJOCO_EGL_DEVICE_ID=3` 用于无头渲染，因而仍遵守一块物理 GPU 的约束。完整 alpha sweep 为 2 个 family × 3 个 task × 9 个 alpha，共 54 个策略、270 个 OOD_MEDIUM jobs、2,700 episodes；成功率并列时选择更高 alpha。
- 首个 seed-1 job 于 20:57:52 CST 完成 10/10 episodes，成功 0 次，随后 seed 21 job 已启动。该结果仅覆盖总验证作业的 1/270，不能提前判定 alpha=0.1 或最终 RETAIN 性能。10 个视频合计 `1,916,031` bytes、均值约 `191,603` bytes；按该均值外推全部 alpha sweep 约 `517,328,370` bytes，相对共享盘可用 `141,906,567,168` bytes 余量充足。policy server 的一次 WebSocket handshake warning 来自 TCP readiness probe，随后实际 evaluator connection 正常建立；当前无致命错误。机器可读监控快照为 `experiments/evaluation-monitor-20260820T205752.json`。
- 21:00:52 CST，seed-21 job 也以 return code 0 完成，10 episodes 同样成功 0 次；seed 41 已随即启动并完成首条轨迹。当前完整验证进度为 2/270 jobs、20 episodes、0 successes；样本仍太少，只作为流水线端到端可执行性与轨迹落盘验证，不作 alpha 优劣判断。

## 2026-08-20 21:12 CST｜首个 alpha 验证点完成并推进 α=0.2

- `retain_taskft_stove_a010` 已完成 OOD_MEDIUM 的 5 个固定 seeds（1/21/41/61/81），每个 seed 10 episodes。逐 seed 成功数为 `0 / 0 / 1 / 1 / 2`，合计 `4/50`，中间成功率为 `8%`；五份 `summary.json` 均完整，最后一个 job 于 21:09:35 CST 以 return code 0 收尾。该结果是 alpha sweep 的第 1/54 个候选点，明确标记为 exploratory interim result，不据此提前选择 alpha，也不外推为 RETAIN 主评测结果。
- pipeline 于 21:09:42 CST 开始加载 `retain_taskft_stove_a020`，线性合并权重为 `[0.2, 0.8]`；21:10:32 CST 启动 seed-1 job。21:12:27 CST 已观察 6 条轨迹、1 次成功，当前 partial rate 不参与候选比较。evaluation pipeline、policy server 与 evaluator 进程链均存活；policy server 仍只看见物理 GPU 3，仿真仍通过同一 GPU 的 EGL 渲染，无 fatal error、OOM 或磁盘错误。
- α=0.1 的 50 个完整视频合计 `9,155,104` bytes、均值 `183,102.08` bytes；按该均值外推完整 2,700-episode alpha sweep 约 `494,375,616` bytes。共享盘可用 `141,893,668,864` bytes、主机可用内存 `530,654,457,856` bytes，资源门禁保持安全。首个策略从加载到最终 job 约 15.8 分钟；若后续任务吞吐相近，剩余 alpha sweep 粗略约 14 小时，但这不是固定 ETA。机器可读进度快照为 `experiments/evaluation-alpha-sweep-progress-20260820T211227.json`。

## 2026-08-20 21:32 CST｜α=0.2 完成并推进 α=0.3

- `retain_taskft_stove_a020` 已完成 5 个固定 seeds；逐 seed 成功数为 `1 / 0 / 1 / 0 / 2`，合计 `4/50=8%`。最后一个 job 于 21:25:24 CST 完成，五份 summary 与 50 条轨迹均完整。当前两个完整点 α=0.1 与 α=0.2 均为 8%；预注册的并列取更高 alpha 规则只在同一任务 9 个点全部完成后应用，当前不提前宣告 α=0.2 胜出。
- pipeline 于 21:25:31 CST 加载 `[0.3, 0.7]` 权重的 `retain_taskft_stove_a030`，21:26:15 CST 开始评测。seed 1/21 已分别取得 `5/10` 和 `4/10`，合计 partial `9/20=45%`；21:31:25 CST 已启动 seed 41。因为剩余三个 seeds 尚未结束，45% 明确标记为 partial，不进入 alpha 选择、不作正式趋势判断。总进度为 12/270 jobs、120 个完整 episodes。
- 前两个完整策略的 100 个视频合计 `18,580,999` bytes、均值 `185,809.99` bytes；按该均值外推完整 sweep 约 `501,686,973` bytes。21:31 CST 共享盘可用 `141,837,553,664` bytes、主机可用内存 `529,010,808,832` bytes；GPU 3 上 policy server 占用 `77,130 MiB`，evaluation pipeline、server、evaluator 与 systemd service 均存活，错误扫描未见 fatal error、OOM 或磁盘错误。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260820T213156.json`。

## 2026-08-20 21:56 CST｜α=0.3、0.4 完成并推进 α=0.5

- `retain_taskft_stove_a030` 的五个固定 seeds（1/21/41/61/81）逐 seed 成功数为 `5 / 4 / 2 / 0 / 1`，合计 `12/50=24%`；最后一个 job 于 21:40:06 CST 完成。`retain_taskft_stove_a040` 的逐 seed 成功数为 `4 / 4 / 4 / 4 / 7`，合计 `23/50=46%`；最后一个 job 于 21:53:31 CST 完成。五份 summary 与 50 个视频对两个策略均齐全。α=0.4 暂为四个已完成点中的最高值，但仍是 exploratory interim result；只有 stove 的 9 个 alpha 全部完成后才按预注册规则选点，因此当前不宣告最终 alpha，也不把四点局部形态写成正式趋势。
- pipeline 随即以 `[0.5, 0.5]` 合并权重启动 `retain_taskft_stove_a050`。21:55:37 CST 时 seed 1 已观察 4 个 episodes、成功 1 次；该 `1/4` 是未完成单 seed 的 partial result，不进入候选比较。总进度为 20/270 个完整 jobs、200 个完整 episodes；evaluation pipeline PID `320418`、policy server PID `392677`、evaluator PID `393490` 均存活，`retain-reproduction.service` 为 active。policy server 仍只使用物理 GPU 3，显存快照为 `77,130 MiB`；仿真 evaluator 不暴露 CUDA compute，仅使用同一 GPU 的 EGL。
- 四个完整策略的 200 个视频合计 `35,508,599` bytes、均值 `177,542.995` bytes；按该均值外推完整 2,700-episode sweep 约 `479,366,086.5` bytes。21:54:47 CST 共享盘可用 `141,813,583,872` bytes、根分区可用 `14,345,596,928` bytes、主机可用内存 `529,321,618,432` bytes。近 15 分钟扫描未见 CUDA OOM、Killed、RuntimeError 或 fatal error；server 日志中的 WebSocket invalid-handshake traceback 与此前相同，来自 TCP readiness probe，真实 evaluator connection 随后正常建立。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260820T215537.json`。

## 2026-08-20 22:14 CST｜α=0.5 完成并推进 α=0.6

- `retain_taskft_stove_a050` 已完成固定 seeds 1/21/41/61/81，逐 seed 成功数为 `3 / 3 / 6 / 7 / 3`，合计 `22/50=44%`；最后一个 job 于 22:06:55 CST 完成，五份 summary 与 50 个视频均齐全。α=0.4 当前仍以 `23/50=46%` 居首，但两者只差 1 个成功 episode；这一差异不足以提前区分 α=0.4 和 α=0.5，且 α=0.6--0.9 尚未完成，因此继续保持 exploratory interim 标记，不执行预注册选择。
- pipeline 已加载 `[0.6, 0.4]` 权重的 `retain_taskft_stove_a060`。seed 1/21 分别完成 `3/10` 与 `6/10`，合计 partial `9/20=45%`；22:13:31 CST 时 seed 41 已观察 3 个 episodes、成功 1 次。总进度为 27/270 个完整 jobs、270 个完整 episodes；未完成 α=0.6 不参与候选比较。
- `retain-reproduction.service` 为 active，evaluation pipeline PID `320418`、policy server PID `410703`、evaluator PID `418484` 均存活。policy server 只使用物理 GPU 3，显存为 `77,130 MiB`；近 25 分钟错误扫描未见 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘空间耗尽或 I/O error。主机可用内存 `530,675,258,368` bytes，共享盘可用 `141,791,571,968` bytes，根分区可用 `14,329,806,848` bytes。
- 前五个完整策略的 250 个视频合计 `43,383,906` bytes、均值 `173,535.624` bytes；按该均值外推 2,700-episode sweep 约 `468,546,184.8` bytes，资源余量充足。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260820T221331.json`。

## 2026-08-20 22:33 CST｜α=0.6、0.7 完成并推进 α=0.8

- `retain_taskft_stove_a060` 的五个 seeds 成功数为 `3 / 6 / 3 / 4 / 5`，合计 `21/50=42%`，最后一个 job 于 22:20:06 CST 完成。`retain_taskft_stove_a070` 的逐 seed 成功数为 `7 / 6 / 7 / 5 / 5`，合计 `30/50=60%`，最后一个 job 于 22:31:55 CST 完成；两个策略均有完整五份 summary 与 50 个视频。
- 当前七个完整点的 success rate 依次为 `8% / 8% / 24% / 46% / 44% / 42% / 60%`，α=0.7 暂时最高，比此前 α=0.4 高 14 个百分点。该结果仍标为 exploratory interim：α=0.8、0.9 尚未完成，不执行预注册选择，也不把未完整曲线解释成最终规律。pipeline 已以 `[0.8, 0.2]` 权重启动 `retain_taskft_stove_a080` 的 seed 1；22:33:01 CST 时 evaluator 刚启动，尚无完整 episode。
- `retain-reproduction.service` 为 active，evaluation pipeline PID `320418`、policy server PID `446637`、evaluator PID `447440` 均存活。policy server 只使用物理 GPU 3，显存为 `77,130 MiB`；近 25 分钟未见 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘空间耗尽或 I/O error。主机可用内存 `530,813,430,784` bytes，共享盘可用 `141,761,929,216` bytes，根分区可用 `14,309,236,736` bytes。
- 前七个完整策略的 350 个视频合计 `58,812,202` bytes、均值 `168,034.862857` bytes；按该均值外推完整 2,700-episode sweep 约 `453,694,129.714` bytes，余量充足。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260820T223301.json`。

## 2026-08-20 22:58 CST｜stove Task-FT alpha sweep 完成并选择 α=0.8

- `retain_taskft_stove_a080` 的五个固定 seeds（1/21/41/61/81）逐 seed 成功数为 `7 / 5 / 7 / 5 / 6`，合计 `30/50=60%`；最后一个 job 于 22:43:35 CST 完成。`retain_taskft_stove_a090` 的逐 seed 成功数为 `10 / 5 / 5 / 3 / 5`，合计 `28/50=56%`；最后一个 job 于 22:55:46 CST 完成。两个策略均有五份完整 summary 和 50 个视频，所有 job 均正常返回。
- stove 九点验证曲线现已闭合：α=0.1--0.9 的成功率依次为 `8% / 8% / 24% / 46% / 44% / 42% / 60% / 60% / 56%`。最高点由 α=0.7 与 α=0.8 并列；依据实验开始前登记的 `highest alpha among equal success rates` 规则，正式选择 α=0.8、线性合并权重 `[0.8, 0.2]` 作为 Task-FT stove 的后续主评测策略。论文使用的 α=0.9 仍作为固定参照保留，其验证成功率比所选点低 4 个百分点。这里是 OOD_MEDIUM small-translation 验证集上的超参数选择，不能当作 ID/OOD/generalist 主结果。
- pipeline 已自动启动下一组 `retain_taskft_mugs_a010`，从 mugs Task-FT step 999 与 pretraining step 9999 参数按 `[0.1, 0.9]` 合并并在物理 GPU 3 上提供策略服务。22:58:20 CST 时 seed 1 已落盘 4/10 episodes，暂为 0 次成功；23:02:28 CST 的同步后复核显示 seed 1 已完整结束为 `0/10`，seed 21 已观察 `0/5`。后者仍是 partial，只用于确认运行进度，不进入候选比较。evaluation pipeline PID `320418`、policy server PID `482596`、当前 evaluator PID `487169` 均存活，service 为 active，错误扫描未见 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error。
- 九个完整策略的 450 个视频合计 `73,103,091` bytes、均值 `162,451.313333` bytes；按该均值外推 2,700-episode sweep 为 `438,618,546` bytes。资源快照显示 GPU 3 `77,130 MiB / 24%`、主机可用内存 `528,635,202,560` bytes、共享盘可用 `141,731,020,800` bytes、根分区可用 `14,269,394,944` bytes，评测存储余量充足。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260820T225820.json`。

## 2026-08-20 23:16 CST｜mugs Task-FT α=0.1 完成并推进 α=0.2

- `retain_taskft_mugs_a010` 已完成 OOD_MEDIUM 的固定 seeds 1/21/41/61/81，每个 seed 均为 `0/10`，合计 `0/50=0%`；最后一个 summary 于 23:14:42 CST 写入。五份 summary、五份 episodes 记录与 50 个视频均完整，evaluation pipeline 的 completed jobs 从 45 增至 50，没有失败 job。
- 这个 0% 是完整候选点而非未完成比例，但仍只覆盖 mugs 九点曲线的 1/9。它支持“当前 `[0.1, 0.9]` 合并在该验证扰动上未成功”的有限结论，不支持提前选择 alpha，也不能外推为 RETAIN 的 ID/OOD/generalist 主评测结果或整个方法失效。
- pipeline 已自动切换到 `retain_taskft_mugs_a020`，按 `[0.2, 0.8]` 合并 mugs Task-FT step 999 与 pretraining step 9999 参数；23:15:54 CST，policy server 已加载在物理 GPU 3，seed-1 evaluator 刚启动且尚无完整 episode。evaluation pipeline PID `320418`、policy server PID `500866`、evaluator PID `501670` 均存活；service 为 active，23:10 CST 后错误扫描未见 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error。
- 当前累计 10/54 个完整策略、50/270 个 jobs、500 个完整 episodes。500 个视频合计 `78,203,100` bytes、均值 `156,406.2` bytes，按均值外推完整 alpha sweep 为 `422,296,740` bytes。资源快照为 GPU 3 `77,126 MiB`、主机可用内存 `530,195,386,368` bytes、共享盘可用 `141,365,346,304` bytes、根分区可用 `14,332,559,360` bytes，单卡与存储门禁均正常。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260820T231613.json`。

## 2026-08-20 23:34 CST｜mugs Task-FT α=0.2 完成并推进 α=0.3

- `retain_taskft_mugs_a020` 已完成固定 seeds 1/21/41/61/81，逐 seed 成功数均为 `0/10`，合计 `0/50=0%`；最后一份 summary 于 23:33:35 CST 写入。五份 summary、五份 episodes 与 50 个视频均完整，evaluation pipeline 的 completed jobs 达到 55，未出现失败 job。
- mugs 当前两个完整点 α=0.1 与 α=0.2 均为 0%。这两个完整负结果表明小 Task-FT 权重的前两点在该 OOD_MEDIUM 验证扰动上均无成功，但 α=0.3--0.9 尚未执行完，不能提前选择 alpha，也不能把前两点外推成方法或任务的最终结论；预注册并列规则继续冻结到九点闭合。
- pipeline 已切换到 `retain_taskft_mugs_a030`，按 `[0.3, 0.7]` 合并参数。23:34:40 CST，policy server 和 seed-1 evaluator 已在同一物理 GPU 3 上启动，尚无完整 episode。evaluation pipeline PID `320418`、policy server PID `518833`、evaluator PID `519634` 均存活；service 为 active，23:25 CST 后错误扫描未见 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error。
- 当前累计 11/54 个完整策略、55/270 个 jobs、550 个完整 episodes。550 个视频合计 `83,353,852` bytes、均值 `151,552.458182` bytes，按均值外推完整 sweep 约 `409,191,637.091` bytes。资源快照为 GPU 3 `77,130 MiB / 49%`、主机可用内存 `529,149,359,104` bytes、共享盘可用 `141,347,602,432` bytes、根分区可用 `14,301,708,288` bytes，单卡与存储余量正常。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260820T233440.json`。

## 2026-08-20 23:54 CST｜mugs Task-FT α=0.3 完成并推进 α=0.4

- `retain_taskft_mugs_a030` 已完成固定 seeds 1/21/41/61/81，逐 seed 成功数均为 `0/10`，合计 `0/50=0%`；最后一份 summary 于 23:52:18 CST 写入。五份 summary、五份 episodes 记录与 50 个视频均完整，evaluation pipeline 的 completed jobs 从 55 增至 60，状态文件为 `running` 且没有 `failed_job`。
- mugs 的 α=0.1、0.2、0.3 三个完整候选当前均为 0%。这是连续三个完整负结果，说明当前 batch-16 单卡训练产物在 OOD_MEDIUM small-translation 验证场景中，Task-FT 权重不超过 0.3 时未取得成功；但 α=0.4--0.9 仍未完成，故不提前执行“并列取更高 alpha”规则，也不把该现象外推为 RETAIN 的 ID/OOD/generalist 主结论。
- pipeline 于 23:52:25 CST 自动加载 `[0.4, 0.6]` 的 `retain_taskft_mugs_a040`。23:54:03 CST，seed-1 evaluator 已连接并落盘首个失败 episode；该 `0/1` 明确标记为 partial，不进入候选比较。evaluation pipeline PID `320418`、policy server PID `536770`、evaluator PID `537628` 均在 `retain-reproduction.service` 内运行；policy server 只看见物理 GPU 3，仿真使用同一 GPU 的 EGL。错误扫描未见 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error；server 中的 InvalidMessage 仍是 TCP readiness probe 后紧跟真实 WebSocket connection 的已知良性告警。
- 当前累计 12/54 个完整策略、60/270 个 jobs、600 个完整 episodes。600 个完整策略视频合计 `89,347,862` bytes、均值 `148,913.103333` bytes，按均值外推完整 sweep 为 `402,065,379` bytes。资源快照为 GPU 3 `77,130 MiB / 20%`、主机可用内存 `530,976,660,480` bytes、共享盘可用 `141,331,415,040` bytes、根分区可用 `14,280,253,440` bytes，单卡与存储余量正常。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260820T235403.json`。

## 2026-08-21 00:12 CST｜mugs Task-FT α=0.4 完成并推进 α=0.5

- `retain_taskft_mugs_a040` 已完成固定 seeds 1/21/41/61/81，逐 seed 成功数为 `0 / 1 / 0 / 1 / 1`，合计 `3/50=6%`；最后一份 summary 于 00:10:56 CST 写入。五份 summary、五份 episodes 记录与 50 个视频均完整，evaluation pipeline 的 completed jobs 从 60 增至 65，状态为 `running` 且没有 `failed_job`。
- α=0.4 是 mugs 已完成的 α=0.1--0.4 中首个非零候选，当前暂居首位。与前三点连续 0% 相比，它提示提高 Task-FT 权重后验证成功开始出现；但这仍是 exploratory interim pattern，α=0.5--0.9 未完成，不能提前选择 α=0.4，也不能把局部变化解释为完整单调关系或主评测性能。
- pipeline 于 00:11:03 CST 自动加载 `[0.5, 0.5]` 的 `retain_taskft_mugs_a050`。00:12:12 CST，policy server 已完成加载，seed-1 evaluator 已建立真实 WebSocket 连接但尚无完整 episode。evaluation pipeline PID `320418`、policy server PID `554846`、evaluator PID `555688` 均在 active service 中；策略服务只看见物理 GPU 3，仿真仍通过同一 GPU 的 EGL。错误扫描未见 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error，readiness probe 的 InvalidMessage 仍为已知良性告警。
- 当前累计 13/54 个完整策略、65/270 个 jobs、650 个完整 episodes。650 个完整策略视频合计 `95,929,154` bytes、均值 `147,583.313846` bytes，按均值外推完整 sweep 约 `398,474,947.385` bytes。资源快照为 GPU 3 `77,130 MiB / 3%`、主机可用内存 `528,419,713,024` bytes、共享盘可用 `141,317,640,192` bytes、根分区可用 `13,797,617,664` bytes，单卡与存储余量正常。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260821T001212.json`。

## 2026-08-21 00:32 CST｜mugs Task-FT α=0.5 完成并推进 α=0.6

- `retain_taskft_mugs_a050` 已完成固定 seeds 1/21/41/61/81，逐 seed 成功数为 `1 / 2 / 2 / 1 / 1`，合计 `7/50=14%`；最后一份 summary 于 00:28:53 CST 写入。五份 summary、五份 episodes 记录与 50 个视频均完整，evaluation pipeline 的 completed jobs 从 65 增至 70，状态为 `running` 且没有 `failed_job`。
- α=0.5 比 α=0.4 的 `3/50=6%` 高 4 个成功 episode，并且五个 seeds 均至少成功一次，当前在 mugs 已完成五点中暂居首位。这一跨 seed 分布使非零结果不依赖单个随机环境，但 α=0.6--0.9 仍未完成，因此继续标记为 exploratory interim，不执行预注册选择，也不将 α=0.4→0.5 的局部改善外推成完整趋势。
- pipeline 于 00:29:00 CST 自动加载 `[0.6, 0.4]` 的 `retain_taskft_mugs_a060`，00:29:44 CST 启动 seed 1。00:32:03 CST 已落盘 5 个 episodes、成功 1 次；该 `1/5` 是 partial observation，不参与候选比较。evaluation pipeline PID `320418`、policy server PID `572897`、evaluator PID `573682` 均在 active service 中；策略服务只看见物理 GPU 3，仿真使用同一 GPU 的 EGL。错误扫描未见 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error。
- 当前累计 14/54 个完整策略、70/270 个 jobs、700 个完整 episodes。700 个完整策略视频合计 `102,720,709` bytes、均值 `146,743.87` bytes，按均值外推完整 sweep 为 `396,208,449` bytes。资源快照为 GPU 3 `77,130 MiB / 57%`、主机可用内存 `529,550,738,432` bytes、共享盘可用 `141,264,003,072` bytes、根分区可用 `13,728,985,088` bytes，单卡与存储余量正常。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260821T003203.json`。

## 2026-08-21 00:52 CST｜mugs Task-FT α=0.6 完成并推进 α=0.7

- `retain_taskft_mugs_a060` 已完成固定 seeds 1/21/41/61/81，逐 seed 成功数为 `2 / 4 / 4 / 2 / 4`，合计 `16/50=32%`；最后一份 summary 于 00:46:08 CST 写入。五份 summary、五份 episodes 记录与 50 个视频均完整，evaluation pipeline 在该策略结束时的 completed jobs 达到 75，状态持续为 `running` 且没有 `failed_job`。
- α=0.6 比 α=0.5 的 `7/50=14%` 高 9 个成功 episode，且五个 seeds 均至少成功两次，当前在 mugs 已完成六点中暂居首位。已完成曲线为 `0% / 0% / 0% / 6% / 14% / 32%`，支持从 α=0.4 开始随 Task-FT 权重增加而改善的局部 exploratory pattern；但 α=0.7--0.9 尚未完成，不能正式选择 α=0.6，也不能宣称完整曲线单调。
- pipeline 于 00:46:15 CST 自动加载 `[0.7, 0.3]` 的 `retain_taskft_mugs_a070`，00:47:05 CST 启动 seed 1。00:52:00 CST 时 seed 1 已完整为 `3/10`，seed 21 已观察 `1/3`，当前 α=0.7 累计 `4/13`；该 partial result 不参与候选比较。evaluation pipeline PID `320418`、policy server PID `590899`、evaluator PID `595292` 均在 active service 中；策略服务只看见物理 GPU 3，仿真使用同一 GPU 的 EGL。错误扫描未见 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error。
- 当前累计 15/54 个完整策略、76/270 个完整 jobs、760 个完整 episodes。750 个完整策略视频合计 `108,906,042` bytes、均值 `145,208.056` bytes，按均值外推完整 sweep 约 `392,061,751.2` bytes。资源快照为 GPU 3 `77,130 MiB / 8%`、主机可用内存 `528,461,409,280` bytes、共享盘可用 `141,249,470,464` bytes、根分区可用 `13,700,534,272` bytes，单卡与存储余量正常。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260821T005200.json`。

## 2026-08-21 01:13 CST｜mugs Task-FT α=0.7 完成并推进 α=0.8

- `retain_taskft_mugs_a070` 已完成固定 seeds 1/21/41/61/81，逐 seed 成功数为 `3 / 4 / 5 / 4 / 5`，合计 `21/50=42%`；最后一份 summary 于 01:02:53 CST 写入。五份 summary、五份 episodes 记录与 50 个视频均完整，evaluation pipeline 在该策略结束时的 completed jobs 达到 80，状态持续为 `running` 且没有 `failed_job`。
- α=0.7 比 α=0.6 的 `16/50=32%` 高 5 个成功 episode，且五个 seeds 均至少成功三次，当前在 mugs 已完成七点中暂居首位。已完成曲线为 `0% / 0% / 0% / 6% / 14% / 32% / 42%`，继续支持中高 Task-FT 权重区间的改善型 exploratory pattern；但 α=0.8、0.9 未完成，不能正式选择 α=0.7，也不能宣称九点完整曲线单调。
- pipeline 于 01:03:00 CST 自动加载 `[0.8, 0.2]` 的 `retain_taskft_mugs_a080`。01:13:10 CST 时 seeds 1/21 已完整为 `2/10` 与 `6/10`，seed 41 已观察 `3/9`，当前累计 `11/29`；这是 partial observation，不参与候选比较。evaluation pipeline PID `320418`、policy server PID `608899`、evaluator PID `616651` 均在 active service 中；策略服务只看见物理 GPU 3，仿真使用同一 GPU 的 EGL。针对 α=0.7/0.8 产物与 evaluation pipeline 的错误扫描未见 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error。
- 当前累计 16/54 个完整策略、82/270 个完整 jobs、820 个完整 episodes。800 个完整策略视频合计 `114,999,788` bytes、均值 `143,749.735` bytes，按均值外推完整 sweep 约 `388,124,284.5` bytes。资源快照为 GPU 3 `77,130 MiB / 0%`、主机可用内存 `530,989,798,400` bytes、共享盘可用 `141,230,014,464` bytes、根分区可用 `13,674,278,912` bytes，单卡与存储余量正常。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260821T011310.json`。

## 2026-08-21 01:31 CST｜mugs Task-FT α=0.8 完成并推进 α=0.9

- `retain_taskft_mugs_a080` 已完成固定 seeds 1/21/41/61/81，逐 seed 成功数为 `2 / 6 / 3 / 4 / 5`，合计 `20/50=40%`；最后一份 summary 于 01:19:14 CST 写入。五份 summary、五份 episodes 记录与 50 个视频均完整，evaluation pipeline 在该策略结束时的 completed jobs 达到 85，状态持续为 `running` 且没有 `failed_job`。
- α=0.8 比当前最高 α=0.7 的 `21/50=42%` 少 1 个成功 episode，因此 α=0.7 仍暂居首位；这一 2 个百分点差异不作统计显著性解释。已完成八点曲线为 `0% / 0% / 0% / 6% / 14% / 32% / 42% / 40%`，此前从 α=0.4 到 α=0.7 的连续改善没有延伸到 α=0.8，但这仍只是 exploratory curve shape；必须完成 α=0.9 后才能按预注册规则正式选点。
- pipeline 于 01:19:21 CST 自动加载 `[0.9, 0.1]` 的 `retain_taskft_mugs_a090`。01:31:40 CST 时 seeds 1/21/41 已完整为 `3/10、5/10、3/10`，seed 61 已观察 `4/8`，累计 `15/38`；这是 partial observation，不参与候选比较。evaluation pipeline PID `320418`、policy server PID `626861`、evaluator PID `637973` 均在 active service 中；策略服务只看见物理 GPU 3，仿真使用同一 GPU 的 EGL。α=0.8/0.9 产物目录与 01:15 CST 以来的 service journal 均未检出 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error。
- 当前累计 17/54 个完整策略、88/270 个完整 jobs、880 个完整 episodes。850 个完整策略视频合计 `120,755,558` bytes、均值 `142,065.362353` bytes，按均值外推完整 sweep 约 `383,576,478.35` bytes。资源快照为 GPU 3 `77,130 MiB / 7%`、主机可用内存 `529,970,173,952` bytes、共享盘可用 `141,207,207,936` bytes、根分区可用 `13,639,778,304` bytes，单卡与存储余量正常。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260821T013140.json`。

## 2026-08-21 01:35 CST｜mugs Task-FT 九点完成并正式选择 α=0.7

- `retain_taskft_mugs_a090` 已完成固定 seeds 1/21/41/61/81，逐 seed 成功数为 `3 / 5 / 3 / 4 / 3`，合计 `18/50=36%`；最后一份 summary 于 01:35:29 CST 写入。五份 summary、五份 episodes 记录与 50 个视频均完整，evaluation pipeline 的 completed jobs 达到 90，状态为 `running` 且没有 `failed_job`。
- mugs 的完整九点验证曲线为 `0% / 0% / 0% / 6% / 14% / 32% / 42% / 40% / 36%`。α=0.7 的 `21/50=42%` 是唯一最高值，故正式选择 α=0.7、线性合并权重 `[0.7, 0.3]` 进入该任务后续主评测，预注册的“同成功率取更高 alpha”规则未触发。论文设定 α=0.8 在本次验证中为 `20/50=40%`，比所选点低 1 个成功 episode；这里只将差异用于冻结策略，不作统计显著性解释，也不把验证集表现当作 ID/OOD/generalist 主结果。
- pipeline 于 01:35:36 CST 自动启动下一组 `retain_taskft_basket_a010`，从 basket Task-FT step 499 与 pretraining step 9999 参数按 `[0.1, 0.9]` 合并。01:35:50 CST 时 policy server PID `644827` 正在物理 GPU 3 上加载，evaluator 尚未启动；evaluation pipeline PID `320418` 与 service 均存活。对 α=0.9、basket α=0.1 目录及 01:30 CST 以来 service journal 的错误扫描未见 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error。
- 当前累计 18/54 个完整策略、90/270 个完整 jobs、900 个完整 episodes。900 个完整策略视频合计 `126,470,567` bytes、均值 `140,522.852222` bytes，按均值外推完整 sweep 为 `379,411,701` bytes。资源快照为 GPU 3 `77,110 MiB / 0%`、主机可用内存 `527,782,860,800` bytes、共享盘可用 `141,202,927,616` bytes、根分区可用 `13,635,522,560` bytes，单卡与存储余量正常。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260821T013550.json`。

## 2026-08-21 01:52 CST｜basket Task-FT α=0.1 完成并推进 α=0.2

- `retain_taskft_basket_a010` 已完成固定 seeds 1/21/41/61/81，每个 seed 均为 `0/10`，合计 `0/50=0%`；最后一份 summary 于 01:51:55 CST 写入。五份 summary、五份 episodes 记录与 50 个视频均完整，evaluation pipeline 的 completed jobs 达到 95，状态为 `running` 且没有 `failed_job`。
- 该 0% 是完整候选点而非 partial observation，但只覆盖 basket 九点曲线的 1/9。它支持“当前 `[0.1, 0.9]` 合并在该验证扰动上未成功”的有限结论，不支持提前套用并列规则、选择 alpha，或外推为 RETAIN 的 ID/OOD/generalist 主评测结果。
- pipeline 于 01:52:02 CST 自动加载 `[0.2, 0.8]` 的 `retain_taskft_basket_a020`。01:52:44 CST 时 policy server PID `662875` 正在物理 GPU 3 上加载，evaluator 尚未启动；evaluation pipeline PID `320418` 与 service 均存活。对 α=0.1 产物目录、evaluation pipeline 与 01:35 CST 以来 service journal 的错误扫描未见 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error。
- 当前累计 19/54 个完整策略、95/270 个完整 jobs、950 个完整 episodes。950 个完整策略视频合计 `133,931,263` bytes、均值 `140,980.276842` bytes，按均值外推完整 sweep 约 `380,646,747.47` bytes。资源快照为 GPU 3 `77,099 MiB / 0%`、主机可用内存 `511,973,798,912` bytes、共享盘可用 `140,831,326,208` bytes、根分区可用 `13,606,432,768` bytes，单卡与存储余量正常。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260821T015244.json`。

## 2026-08-21 02:11 CST｜basket Task-FT α=0.2 完成并推进 α=0.3

- `retain_taskft_basket_a020` 已完成固定 seeds 1/21/41/61/81，每个 seed 均为 `0/10`，合计 `0/50=0%`；最后一份 summary 于 02:07:49 CST 写入。五份 summary、五份 episodes 记录与 50 个视频均完整，evaluation pipeline 的 completed jobs 达到 100，状态为 `running` 且没有 `failed_job`。
- basket 的 α=0.1 与 α=0.2 两个完整候选均为 0%。这两个完整负结果表明较低 Task-FT 权重的前两点在当前 OOD_MEDIUM 验证扰动下尚无成功，但 α=0.3--0.9 未完成，不能提前应用并列规则、选择 alpha 或外推主评测。
- pipeline 于 02:07:56 CST 自动加载 `[0.3, 0.7]` 的 `retain_taskft_basket_a030`，02:08:42 CST 启动 seed 1。02:11:38 CST 时该 seed 已观察 `0/9`；这是 partial observation，不参与候选比较。evaluation pipeline PID `320418`、policy server PID `680822`、evaluator PID `681603` 均在 active service 中；策略服务只看见物理 GPU 3，仿真使用同一 GPU 的 EGL。对 α=0.2/0.3 产物目录、evaluation pipeline 与 01:50 CST 以来 service journal 的错误扫描未见 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error。
- 当前累计 20/54 个完整策略、100/270 个完整 jobs、1,000 个完整 episodes。1,000 个完整策略视频合计 `141,475,985` bytes、均值 `141,475.985` bytes，按均值外推完整 sweep 约 `381,985,159.5` bytes。资源快照为 GPU 3 `77,115 MiB / 19%`、主机可用内存 `529,966,102,528` bytes、共享盘可用 `140,816,355,328` bytes、根分区可用 `13,667,700,736` bytes，单卡与存储余量正常。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260821T021138.json`。

## 2026-08-21 02:31 CST｜basket Task-FT α=0.3 完成并推进 α=0.4

- `retain_taskft_basket_a030` 已完成固定 seeds 1/21/41/61/81，每个 seed 均为 `0/10`，合计 `0/50=0%`；最后一份 summary 于 02:23:37 CST 写入。五份 summary、五份 episodes 记录与 50 个视频均完整，evaluation pipeline 在该策略结束时的 completed jobs 达到 105，状态持续为 `running` 且没有 `failed_job`。
- basket 的 α=0.1、0.2、0.3 三个完整候选均为 0%。这说明 Task-FT 权重不超过 0.3 时，本次 batch-16 单卡产物在当前 OOD_MEDIUM 验证扰动下尚无成功；但 α=0.4--0.9 未完成，不能提前执行并列规则或外推主评测。
- pipeline 于 02:23:44 CST 自动加载 `[0.4, 0.6]` 的 `retain_taskft_basket_a040`。02:31:36 CST 时 seeds 1/21 已完整为 `0/10、0/10`，seed 41 已观察 `0/3`，累计 `0/23`；这是 partial observation，不参与候选比较。evaluation pipeline PID `320418`、policy server PID `698792`、evaluator PID `706541` 均在 active service 中；策略服务只看见物理 GPU 3，仿真使用同一 GPU 的 EGL。对 α=0.3/0.4 产物目录、evaluation pipeline 与 02:05 CST 以来 service journal 的错误扫描未见 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error。
- 当前累计 21/54 个完整策略、107/270 个完整 jobs、1,070 个完整 episodes。1,050 个完整策略视频合计 `149,206,656` bytes、均值 `142,101.577143` bytes，按均值外推完整 sweep 约 `383,674,258.29` bytes。资源快照为 GPU 3 `77,115 MiB / 54%`、主机可用内存 `530,217,416,704` bytes、共享盘可用 `140,800,757,760` bytes、根分区可用 `13,629,304,832` bytes，单卡与存储余量正常。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260821T023136.json`。

## 2026-08-21 02:54 CST｜basket Task-FT α=0.4 完成并推进 α=0.5

- `retain_taskft_basket_a040` 已完成固定 seeds 1/21/41/61/81，逐 seed 均为 `0/10`，合计 `0/50=0%`；最后一个 job 于 02:39:27 CST 完成。五份 summary、五份 episodes 记录与 50 个视频均完整，evaluation pipeline 在该策略结束时的 completed jobs 达到 110，状态持续为 `running` 且没有 `failed_job`。
- basket 的 α=0.1--0.4 四个完整候选目前均为 0%。这支持“本次 batch-16 单卡产物在 OOD_MEDIUM small-translation 验证中，Task-FT 权重不超过 0.4 时尚无成功”的有限结论；α=0.5--0.9 仍未闭合，因此不提前应用并列规则、不选择 alpha，也不外推 ID/OOD/generalist 主结果。
- pipeline 于 02:39:34 CST 自动加载 `[0.5, 0.5]` 的 `retain_taskft_basket_a050`。02:54:09 CST 时 seeds 1/21/41/61 已完整为 `0/10、0/10、0/10、0/10`，seed 81 已观察 `1/5`，当前累计 partial `1/45`；该未完成点不进入候选比较。evaluation pipeline PID `320418`、policy server PID `716838`、evaluator PID `731369` 均在 active service 中；策略服务环境为 `CUDA_VISIBLE_DEVICES=3`，evaluator 不暴露 CUDA compute 且以 `MUJOCO_EGL_DEVICE_ID=3` 使用同一物理 GPU。GPU compute-process 审计只发现 policy server PID `716838`，继续满足一块 GPU 的约束。
- 当前累计 22/54 个完整策略、114/270 个完整 jobs、1,140 个完整 episodes。1,100 个完整策略视频合计 `156,976,211` bytes、均值 `142,705.646364` bytes，按均值外推完整 sweep 约 `385,305,245.18` bytes。02:54:19 CST 资源快照为 GPU 3 `77,115 MiB / 40%`、主机可用内存 `531,262,281,728` bytes、共享盘可用 `140,783,198,208` bytes、根分区可用 `13,611,507,712` bytes；α=0.4/0.5 目录与近 30 分钟 service journal 均未检出 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260821T025419.json`。

## 2026-08-21 02:57 CST｜basket Task-FT α=0.5 完成并推进 α=0.6

- 同步后复核确认 `retain_taskft_basket_a050` 的最后一个 seed 已于 02:55:20 CST 收尾。固定 seeds 1/21/41/61/81 的成功数为 `0 / 0 / 0 / 0 / 1`，合计 `1/50=2%`；五份 summary、五份 episodes 记录和 50 个视频完整，evaluation pipeline 的 completed jobs 达到 115，状态为 `running` 且没有 `failed_job`。
- α=0.5 是 basket 的前五个完整候选中首个非零点，但唯一成功来自 seed 81 的单个 episode。该结果仅说明 `[0.5, 0.5]` 合并在当前验证扰动中出现过成功；样本不足以解释机制或正式选择，仍须完成 α=0.6--0.9 并按预注册规则闭合九点曲线。
- pipeline 已自动加载 `[0.6, 0.4]` 的 `retain_taskft_basket_a060`。02:57:30 CST 时 seed 1 已观察 `0/3`，属于不参与候选比较的 partial observation。evaluation pipeline PID `320418`、policy server PID `735024`、evaluator PID `735810` 均在 active service 中；policy server 为 `CUDA_VISIBLE_DEVICES=3`，evaluator 为 `CUDA_VISIBLE_DEVICES=`、`MUJOCO_EGL_DEVICE_ID=3`，GPU compute-process 审计只发现 PID `735024`，继续满足单物理 GPU 约束。
- 当前累计 23/54 个完整策略、115/270 个完整 jobs、1,150 个完整 episodes。1,150 个完整策略视频合计 `164,892,931` bytes、均值 `143,385.157391` bytes，按均值外推完整 sweep 约 `387,139,924.96` bytes。02:57:46 CST 资源快照为 GPU 3 `77,115 MiB / 63%`、主机可用内存 `528,859,131,904` bytes、共享盘可用 `140,778,725,376` bytes、根分区可用 `13,612,638,208` bytes；α=0.5/0.6 目录和近 15 分钟 service journal 均未检出 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260821T025746.json`。

## 2026-08-21 03:12 CST｜basket Task-FT α=0.6 完成并推进 α=0.7

- `retain_taskft_basket_a060` 已完成固定 seeds 1/21/41/61/81，逐 seed 成功数为 `0 / 1 / 1 / 0 / 3`，合计 `5/50=10%`；最后一个 job 于 03:11:01 CST 完成。五份 summary、五份 episodes 记录和 50 个视频完整，evaluation pipeline 的 completed jobs 达到 120，状态为 `running` 且没有 `failed_job`。
- α=0.6 较 α=0.5 的 `1/50=2%` 增加 4 个成功 episode，成功分布于 seeds 21、41、81，当前在 basket 已完成六点中暂居首位。该上升属于 exploratory interim pattern；α=0.7--0.9 未完成，不能提前选择 α=0.6，也不把验证曲线外推为 ID/OOD/generalist 主结果。
- pipeline 已自动加载 `[0.7, 0.3]` 的 `retain_taskft_basket_a070`。03:12:16 CST 时 seed-1 evaluator 已启动但尚无完整 episode。evaluation pipeline PID `320418`、policy server PID `753109`、evaluator PID `753944` 均在 active service 中；policy server 为 `CUDA_VISIBLE_DEVICES=3`，evaluator 为 `CUDA_VISIBLE_DEVICES=`、`MUJOCO_EGL_DEVICE_ID=3`，GPU compute-process 审计只发现 PID `753109`，继续满足单物理 GPU 约束。
- 当前累计 24/54 个完整策略、120/270 个完整 jobs、1,200 个完整 episodes。1,200 个完整策略视频合计 `172,493,298` bytes、均值 `143,744.415` bytes，按均值外推完整 sweep 约 `388,109,920.5` bytes。03:12:16 CST 资源快照为 GPU 3 `77,115 MiB / 1%`、主机可用内存 `530,393,825,280` bytes、共享盘可用 `140,767,662,080` bytes、根分区可用 `13,589,790,720` bytes；α=0.6/0.7 目录和近 20 分钟 service journal 均未检出 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260821T031216.json`。

## 2026-08-21 03:32 CST｜basket Task-FT α=0.7 完成并推进 α=0.8

- `retain_taskft_basket_a070` 已完成固定 seeds 1/21/41/61/81，逐 seed 成功数为 `3 / 2 / 2 / 3 / 2`，合计 `12/50=24%`；最后一个 job 于 03:26:01 CST 完成。五份 summary、五份 episodes 记录和 50 个视频完整，evaluation pipeline 在该策略结束时的 completed jobs 达到 125，状态为 `running` 且没有 `failed_job`。
- α=0.7 较 α=0.6 的 `5/50=10%` 增加 7 个成功 episode，并在全部五个 seeds 上至少成功两次，当前为 basket 已完成七点中的暂时最高值。α=0.5--0.7 的完整率为 `2% / 10% / 24%`，支持中高 Task-FT 权重区间的改善型 exploratory pattern；α=0.8、0.9 未闭合，不能正式选择或声称完整曲线单调。
- pipeline 已自动加载 `[0.8, 0.2]` 的 `retain_taskft_basket_a080`。03:32:41 CST 时 seeds 1/21 已分别完整为 `2/10、2/10`，seed 41 刚启动且尚无完整 episode，当前 partial 为 `4/20`，不进入候选比较。evaluation pipeline PID `320418`、policy server PID `771173`、evaluator PID `779109` 均在 active service 中；policy server 为 `CUDA_VISIBLE_DEVICES=3`，当前 evaluator 为 `CUDA_VISIBLE_DEVICES=`、`MUJOCO_EGL_DEVICE_ID=3`，GPU compute-process 审计只发现 PID `771173`，继续满足单物理 GPU 约束。
- 当前累计 25/54 个完整策略、127/270 个完整 jobs、1,270 个完整 episodes。1,250 个完整策略视频合计 `179,972,214` bytes、均值 `143,977.7712` bytes，按均值外推完整 sweep 约 `388,739,982.24` bytes。03:32:15 CST 资源快照为 GPU 3 `77,115 MiB / 63%`、主机可用内存 `528,701,366,272` bytes、共享盘可用 `140,716,564,480` bytes、根分区可用 `13,521,907,712` bytes；α=0.7/0.8 目录和近 25 分钟 service journal 均未检出 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260821T033241.json`。

## 2026-08-21 03:52 CST｜basket Task-FT α=0.8 完成并推进 α=0.9

- `retain_taskft_basket_a080` 已完成固定 seeds 1/21/41/61/81，逐 seed 成功数为 `2 / 2 / 0 / 2 / 3`，合计 `9/50=18%`；最后一个 job 于 03:41:18 CST 完成。五份 summary、五份 episodes 记录和 50 个视频完整，evaluation pipeline 在该策略结束时的 completed jobs 达到 130，状态为 `running` 且没有 `failed_job`。
- α=0.8 比当前最高 α=0.7 的 `12/50=24%` 少 3 个成功 episode，因此 α=0.7 继续暂居首位；这一 6 个百分点差异不作统计显著性解释。当前八点曲线为 `0% / 0% / 0% / 0% / 2% / 10% / 24% / 18%`，说明 α=0.5--0.7 的改善没有延伸到 α=0.8，但完整曲线仍须等待 α=0.9。
- pipeline 已自动加载 `[0.9, 0.1]` 的 `retain_taskft_basket_a090`。03:51:50 CST 时 seeds 1/21/41 已分别完整为 `1/10、3/10、0/10`，seed 61 已观察 `2/3`，累计 partial `6/33`，不进入候选比较。evaluation pipeline PID `320418`、policy server PID `789255`、evaluator PID `800360` 均在 active service 中；policy server 为 `CUDA_VISIBLE_DEVICES=3`，evaluator 为 `CUDA_VISIBLE_DEVICES=`、`MUJOCO_EGL_DEVICE_ID=3`，GPU compute-process 审计只发现 PID `789255`，继续满足单物理 GPU 约束。
- 当前累计 26/54 个完整策略、133/270 个完整 jobs、1,330 个完整 episodes。1,300 个完整策略视频合计 `187,636,933` bytes、均值 `144,336.102308` bytes，按均值外推完整 sweep 约 `389,707,476.23` bytes。03:52:08 CST 资源快照为 GPU 3 `77,115 MiB / 15%`、主机可用内存 `526,751,114,240` bytes、共享盘可用 `140,702,957,568` bytes、根分区可用 `13,497,085,952` bytes；α=0.8/0.9 目录和近 25 分钟 service journal 均未检出 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260821T035208.json`。

## 2026-08-21 04:13 CST｜Task-FT 三任务选点闭合并进入 coFT sweep

- `retain_taskft_basket_a090` 已完成固定 seeds 1/21/41/61/81，逐 seed 成功数为 `1 / 3 / 0 / 5 / 3`，合计 `12/50=24%`；最后一个 job 于 03:56:11 CST 完成，五份 summary、五份 episodes 记录与 50 个视频均完整。basket 九点曲线为 `0% / 0% / 0% / 0% / 2% / 10% / 24% / 18% / 24%`，α=0.7 与 α=0.9 并列最高；按预登记的 `highest alpha among equal success rates` 规则选择 α=0.9、线性合并权重 `[0.9, 0.1]`。该值恰好等于论文 basket alpha；至此 Task-FT 的 stove/mugs/basket 选点分别冻结为 `0.8 / 0.7 / 0.9`。这些仍只是 OOD_MEDIUM small-translation 验证选点，不是 ID/OOD/generalist 主结果。
- pipeline 随后自动切换至 coFT family。首个完整候选 `retain_coft_stove_a010` 的逐 seed 成功数为 `1 / 0 / 0 / 2 / 0`，合计 `3/50=6%`，最后一个 job 于 04:11:03 CST 完成；α=0.1 只是 coFT stove 九点曲线的首点，不能提前选 alpha。04:13:51 CST 时 `[0.2, 0.8]` 的 `retain_coft_stove_a020` seed 1 已观察 `2/7`，属于不进入候选比较的 partial observation。
- evaluation pipeline PID `320418`、policy server PID `825128`、evaluator PID `825917` 均在 active service 中。policy server 环境为 `CUDA_VISIBLE_DEVICES=3`，evaluator 为 `CUDA_VISIBLE_DEVICES=`、`MUJOCO_EGL_DEVICE_ID=3`；GPU compute-process 审计只发现 policy PID `825128`，继续满足一块物理 GPU 的约束。basket α=0.9、coFT stove α=0.1/0.2 目录及近 35 分钟 service journal 均未检出 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error。
- 当前累计 28/54 个完整策略、140/270 个完整 jobs、1,400 个完整 episodes。1,400 个完整策略视频合计 `204,886,701` bytes、均值 `146,347.643571` bytes，按均值外推完整 sweep 约 `395,138,637.64` bytes。资源快照为 GPU 3 `77,115 MiB / 27%`、主机可用内存 `529,331,207,168` bytes、共享盘可用 `140,683,603,968` bytes、根分区可用 `13,481,230,336` bytes；机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260821T041351.json`。

## 2026-08-21 04:32 CST｜coFT stove α=0.2 完成并推进 α=0.3

- `retain_coft_stove_a020` 已完成固定 seeds 1/21/41/61/81，逐 seed 成功数为 `3 / 1 / 1 / 0 / 0`，合计 `5/50=10%`；最后一个 job 于 04:25:35 CST 完成，五份 summary、五份 episodes 记录与 50 个视频均完整。α=0.2 比 α=0.1 的 `3/50=6%` 多 2 个成功 episode，暂居 coFT stove 前两个完整候选之首；该小差异仅用于描述 exploratory interim 排序，不作统计显著性或机制解释，也不能在剩余七点完成前选择 alpha。
- pipeline 已自动加载 `[0.3, 0.7]` 的 `retain_coft_stove_a030`。04:32:11 CST 时 seeds 1/21 已分别完整为 `1/10、2/10`，seed 41 已观察 `0/1`，累计 partial 为 `3/21`，不进入候选比较。evaluation pipeline PID `320418`、policy server PID `843158`、evaluator PID `850929` 均在 active service 中；policy server 为 `CUDA_VISIBLE_DEVICES=3`，evaluator 为 `CUDA_VISIBLE_DEVICES=`、`MUJOCO_EGL_DEVICE_ID=3`，GPU compute-process 审计只发现 PID `843158`，继续满足一块物理 GPU 的约束。
- 当前累计 29/54 个完整策略、147/270 个完整 jobs、1,470 个完整 episodes。1,450 个完整策略视频合计 `214,261,687` bytes、均值 `147,766.680690` bytes，按均值外推完整 sweep 约 `398,970,037.86` bytes。资源快照为 GPU 3 `77,115 MiB / 36%`、主机可用内存 `530,701,421,568` bytes、共享盘可用 `140,666,150,912` bytes、根分区可用 `13,457,297,408` bytes；α=0.2/0.3 产物目录和近 30 分钟 service journal 均未检出 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260821T043211.json`。

## 2026-08-21 04:53 CST｜coFT stove α=0.3、0.4 完成并推进 α=0.5

- `retain_coft_stove_a030` 已完成固定 seeds 1/21/41/61/81，逐 seed 成功数为 `1 / 2 / 1 / 3 / 1`，合计 `8/50=16%`；最后一个 job 于 04:39:40 CST 完成。`retain_coft_stove_a040` 随后也完成，逐 seed 为 `6 / 4 / 4 / 4 / 6`，合计 `24/50=48%`，最后一个 job 于 04:51:57 CST 完成。两个策略均有五份 summary、五份 episodes 记录与 50 个视频。
- coFT stove 当前四个完整点为 `6% / 10% / 16% / 48%`。α=0.4 比 α=0.3 多 16 个成功 episode，且五个 seeds 均至少成功四次，说明 `[0.4, 0.6]` 在本次 OOD_MEDIUM 验证中出现跨 seed 的明显跃升；它仍是 exploratory interim pattern，α=0.5--0.9 未完成前不正式选择，也不把验证排序外推为主评测结论。
- pipeline 已自动加载 `[0.5, 0.5]` 的 `retain_coft_stove_a050`。04:53:15 CST 时 seed 1 仅观察到 `1/1`，不进入候选比较。evaluation pipeline PID `320418`、policy server PID `878942`、evaluator PID `879771` 均在 active service 中；policy server 为 `CUDA_VISIBLE_DEVICES=3`，evaluator 为 `CUDA_VISIBLE_DEVICES=`、`MUJOCO_EGL_DEVICE_ID=3`，GPU compute-process 审计只发现 PID `878942`，继续满足一块物理 GPU 的约束。
- 当前累计 31/54 个完整策略、155/270 个完整 jobs、1,550 个完整 episodes。1,550 个完整策略视频合计 `231,734,211` bytes、均值 `149,505.942581` bytes，按均值外推完整 sweep 约 `403,666,044.97` bytes。资源快照为 GPU 3 `77,115 MiB / 63%`、主机可用内存 `529,358,497,792` bytes、共享盘可用 `140,647,796,736` bytes、根分区可用 `13,448,974,336` bytes；α=0.3/0.4/0.5 目录和近 30 分钟 service journal 均未检出 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260821T045315.json`。

## 2026-08-21 05:11 CST｜coFT stove α=0.5 完成并推进 α=0.6

- `retain_coft_stove_a050` 已完成固定 seeds 1/21/41/61/81，逐 seed 成功数为 `4 / 4 / 8 / 3 / 7`，合计 `26/50=52%`；最后一个 job 于 05:04:02 CST 完成，五份 summary、五份 episodes 记录与 50 个视频均完整。α=0.5 比 α=0.4 的 `24/50=48%` 多 2 个成功 episode，暂居 coFT stove 前五个完整候选之首；这 4 个百分点不作统计显著性解释。
- 当前五点曲线为 `6% / 10% / 16% / 48% / 52%`，表明 α=0.4 的跨 seed 跃升延伸到等权合并，但剩余 α=0.6--0.9 未闭合，因此继续保持 exploratory interim 标记，不执行预登记选点，也不外推为 ID/OOD/generalist 主结果。
- pipeline 已自动加载 `[0.6, 0.4]` 的 `retain_coft_stove_a060`。05:11:48 CST 时 seeds 1/21/41 已分别完整为 `4/10、4/10、6/10`，累计 partial `14/30`；seed 61 evaluator 刚启动且尚无完整 episode。evaluation pipeline PID `320418`、policy server PID `896855`、evaluator PID `907972` 均在 active service 中；policy server 为 `CUDA_VISIBLE_DEVICES=3`，evaluator 为 `CUDA_VISIBLE_DEVICES=`、`MUJOCO_EGL_DEVICE_ID=3`，GPU compute-process 审计只发现 PID `896855`，继续满足一块物理 GPU 的约束。
- 当前累计 32/54 个完整策略、163/270 个完整 jobs、1,630 个完整 episodes。1,600 个完整策略视频合计 `239,332,957` bytes、均值 `149,583.098125` bytes，按均值外推完整 sweep 约 `403,874,364.94` bytes。资源快照为 GPU 3 `77,115 MiB / 13%`、主机可用内存 `527,420,204,032` bytes、共享盘可用 `140,630,278,144` bytes、根分区可用 `12,631,355,392` bytes；α=0.5/0.6 目录和近 25 分钟 service journal 均未检出 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260821T051148.json`。

## 2026-08-21 05:32 CST｜coFT stove α=0.6、0.7 完成并推进 α=0.8

- `retain_coft_stove_a060` 已完成固定 seeds 1/21/41/61/81，逐 seed 成功数为 `4 / 4 / 6 / 6 / 8`，合计 `28/50=56%`；最后一个 job 于 05:15:36 CST 完成。`retain_coft_stove_a070` 随后也完成，逐 seed 为 `5 / 5 / 6 / 7 / 7`，合计 `30/50=60%`，最后一个 job 于 05:27:12 CST 完成。两个策略均有五份 summary 与 50 个视频，产物数量完整。
- coFT stove 当前七个完整点为 `6% / 10% / 16% / 48% / 52% / 56% / 60%`，α=0.4 之后继续逐点上升；论文使用的 α=0.7 目前领先。但 α=0.8、0.9 仍未闭合，因此不提前执行预登记选点，也不把验证曲线解释为 ID/OOD/generalist 主结果。
- pipeline 已自动加载 `[0.8, 0.2]` 的 `retain_coft_stove_a080`。05:32:14 CST 时 seeds 1/21 已分别完整为 `7/10、8/10`，seed 41 刚启动且尚无完整 episode，当前 partial 为 `15/20`，不进入候选比较。evaluation pipeline PID `320418`、policy server PID `932569`、evaluator PID `940340` 均在 active service 中；policy server 为 `CUDA_VISIBLE_DEVICES=3`，evaluator 为 `CUDA_VISIBLE_DEVICES=`、`MUJOCO_EGL_DEVICE_ID=3`，GPU compute-process 审计只发现 policy PID `932569`，继续满足一块物理 GPU 的约束。
- 当前累计 34/54 个完整策略、172/270 个完整 jobs、1,720 个完整 episodes。1,700 个完整策略视频合计 `254,061,097` bytes、均值 `149,447.704118` bytes，按均值外推完整 sweep 约 `403,508,801.12` bytes。资源快照为 GPU 3 `77,115 MiB / 0%`（采样时处于 seed 切换后的瞬时空档）、主机可用内存 `527,756,632,064` bytes、共享盘可用 `140,613,029,888` bytes、根分区可用 `12,606,869,504` bytes；α=0.6/0.7/0.8 目录和近 25 分钟 service journal 均未检出 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260821T053214.json`。

## 2026-08-21 05:55 CST｜coFT stove 九点曲线闭合并选择 α=0.8

- `retain_coft_stove_a080` 已完成固定 seeds 1/21/41/61/81，逐 seed 成功数为 `7 / 8 / 3 / 8 / 7`，合计 `33/50=66%`；最后一个 job 于 05:38:22 CST 完成，五份 summary 与 50 个视频完整，视频共 `6,837,245` bytes。`retain_coft_stove_a090` 随后也完成，逐 seed 为 `5 / 5 / 5 / 7 / 7`，合计 `29/50=58%`；最后一个 job 于 05:49:52 CST 完成，五份 summary 与 50 个视频完整，视频共 `7,042,826` bytes。
- coFT stove 九点曲线现为 `6% / 10% / 16% / 48% / 52% / 56% / 60% / 66% / 58%`。α=0.8 是唯一最高点，按预登记规则正式选择 α=0.8、线性合并权重 `[0.8, 0.2]` 进入后续主评测；并列规则没有触发。论文设定 α=0.7 的本次验证结果为 `30/50=60%`，比所选点低 3 个成功 episode；该 6 个百分点差异仅用于冻结策略，不作统计显著性、机制或 ID/OOD/generalist 主性能解释。
- pipeline 于 05:49:58 CST 自动加载 `[0.1, 0.9]` 的 `retain_coft_mugs_a010`。seed 1 于 05:54:07 CST 完整结束为 `0/10`；05:55:55 CST 时 seed 21 已观察 `0/5`，当前累计 partial `0/15`，不进入候选比较。evaluation pipeline PID `320418`、policy server PID `968365`、evaluator PID `973051` 均在 active service 中；policy server 为 `CUDA_VISIBLE_DEVICES=3`，evaluator 为 `CUDA_VISIBLE_DEVICES=`、`MUJOCO_EGL_DEVICE_ID=3`，GPU compute-process 审计只发现 policy PID `968365`，继续满足一块物理 GPU 的约束。
- 当前累计 36/54 个完整策略、181/270 个完整 jobs、1,810 个完整 episodes。1,800 个完整策略视频合计 `267,941,168` bytes、均值 `148,856.204444` bytes，按均值外推完整 sweep 约 `401,911,752` bytes。资源快照为 GPU 3 `77,115 MiB / 4%`、主机可用内存 `528,695,212,032` bytes、共享盘可用 `140,591,968,256` bytes、根分区可用 `12,670,259,200` bytes；α=0.8/0.9 与当前 mugs α=0.1 结果文件、相应 supervisor 日志片段均未检出 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260821T055555.json`。

## 2026-08-21 06:11 CST｜coFT mugs α=0.1 完成并推进 α=0.2

- `retain_coft_mugs_a010` 已完成固定 seeds 1/21/41/61/81，逐 seed 均为 `0/10`，合计 `0/50=0%`；最后一个 job 于 06:07:21.770278 CST 完成。五份 summary 与 50 个视频完整，本策略视频共 `5,058,805` bytes。
- α=0.1 是 coFT mugs 九点曲线的首个完整候选。该完整零成功只记录为 exploratory negative result：它不能触发预登记的并列取更高 alpha 规则，也不能支持对剩余八点、整个 RETAIN 方法或 ID/OOD/generalist 主评测的结论。
- pipeline 已自动加载 `[0.2, 0.8]` 的 `retain_coft_mugs_a020`。06:11:52 CST 时 seed 1 完整为 `0/10`，seed 21 暂为 `0/2`，后者属于不参与候选比较的 partial observation。evaluation pipeline PID `320418`、policy server PID `986974`、evaluator PID `991367` 均在 active service 中；policy server 为 `CUDA_VISIBLE_DEVICES=3`，evaluator 为 `CUDA_VISIBLE_DEVICES=`、`MUJOCO_EGL_DEVICE_ID=3`，GPU compute-process 审计只发现 PID `986974`，继续满足一块物理 GPU 的约束。
- 当前累计 37/54 个完整策略、186/270 个完整 jobs、1,860 个完整 episodes。1,850 个完整策略视频合计 `272,999,973` bytes、均值 `147,567.552973` bytes，按均值外推完整 sweep 约 `398,432,393.03` bytes。资源快照为 GPU 3 `77,115 MiB / 64%`、主机可用内存 `528,588,685,312` bytes、共享盘可用 `140,581,109,760` bytes、根分区可用 `12,658,200,576` bytes；α=0.1/0.2 结果文件与从 α=0.1 加载起的 supervisor 日志片段均未检出 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260821T061152.json`。

## 2026-08-21 06:36 CST｜coFT mugs α=0.2 完成并推进 α=0.3

- `retain_coft_mugs_a020` 已完成固定 seeds 1/21/41/61/81，逐 seed 均为 `0/10`，合计 `0/50=0%`；最后一个 job 于 06:24:48.844240 CST 完成。五份 summary、五份 episodes 记录和 50 个视频完整，episode error 数为 0，本策略视频共 `4,699,394` bytes。
- coFT mugs 的 α=0.1 与 α=0.2 两个完整点目前均为 0%。这是前两个低 coFT 权重点的完整 exploratory negative results，但九点曲线仍有七点未完成；预登记的并列取更高 alpha 规则继续冻结，不能据此选择 α=0.2、断言整个方法失败或外推 ID/OOD/generalist 主结果。
- pipeline 已自动加载 `[0.3, 0.7]` 的 `retain_coft_mugs_a030`。06:36:25 CST 时 seeds 1/21/41 均完整为 `0/10`，seed 61 暂为 `0/2`，累计 partial `0/32`，不参与候选比较。evaluation pipeline PID `320418`、policy server PID `1005093`、evaluator PID `1016433` 均在 active service 中；policy server 为 `CUDA_VISIBLE_DEVICES=3`，evaluator 为 `CUDA_VISIBLE_DEVICES=`、`MUJOCO_EGL_DEVICE_ID=3`，GPU compute-process 审计只发现 policy PID `1005093`，继续满足一块物理 GPU 的约束。
- 当前累计 38/54 个完整策略、193/270 个完整 jobs、1,930 个完整 episodes。1,900 个完整策略视频合计 `277,699,367` bytes、均值 `146,157.561579` bytes，按均值外推完整 sweep 约 `394,625,416.26` bytes。资源快照为 GPU 3 `77,115 MiB / 0%`（采样瞬时利用率）、主机可用内存 `527,598,827,520` bytes、共享盘可用 `140,533,456,896` bytes、根分区可用 `12,628,779,008` bytes；α=0.2/0.3 非视频产物与从 α=0.2 起的 supervisor 日志片段均未检出 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260821T063625.json`。

## 2026-08-21 06:54 CST｜coFT mugs α=0.3 完成并推进 α=0.4

- `retain_coft_mugs_a030` 已完成固定 seeds 1/21/41/61/81，逐 seed 均为 `0/10`，合计 `0/50=0%`；最后一个 job 于 06:42:11.761241 CST 完成。五份 summary、五份 episodes 记录与 50 个视频均完整，episode error 数为 0，本策略视频共 `4,914,805` bytes。
- coFT mugs 的前三个完整点 α=0.1/0.2/0.3 当前均为 0%。这是三个低 coFT 权重点的 exploratory negative results；九点曲线仍有六点未完成，预登记的并列取更高 alpha 规则继续冻结，不能据此选择 α=0.3、断言整个方法失败或外推 ID/OOD/generalist 主结果。
- pipeline 已自动加载 `[0.4, 0.6]` 的 `retain_coft_mugs_a040`。06:54:34 CST 时 seeds 1/21/41 已分别完整为 `1/10、1/10、2/10`，seed 61 暂为 `2/6`，累计 partial 为 `6/36`，不进入候选比较。evaluation pipeline PID `320418`、policy server PID `1023214`、evaluator PID `1034343` 均在 active service 中；policy server 为 `CUDA_VISIBLE_DEVICES=3`，evaluator 为 `CUDA_VISIBLE_DEVICES=`、`MUJOCO_EGL_DEVICE_ID=3`，GPU compute-process 审计只发现 policy PID `1023214` 位于物理 GPU 3，继续满足一块物理 GPU 的约束。
- 当前累计 39/54 个完整策略、198/270 个完整 jobs、1,980 个完整 episodes。1,950 个完整策略视频合计 `282,614,172` bytes、均值 `144,930.344615` bytes，按均值外推完整 sweep 约 `391,311,930.46` bytes。资源快照为 GPU 3 `77,115 MiB / 49%`、主机可用内存 `528,375,800,832` bytes、共享盘可用 `140,525,703,168` bytes、根分区可用 `12,613,689,344` bytes；α=0.3/0.4 非视频产物与从 α=0.3 起的 supervisor 日志片段均未检出 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260821T065434.json`。

## 2026-08-21 07:12 CST｜coFT mugs α=0.4 完成并推进 α=0.5

- `retain_coft_mugs_a040` 已完成固定 seeds 1/21/41/61/81，逐 seed 成功数为 `1 / 1 / 2 / 2 / 1`，合计 `7/50=14%`；最后一个 job 于 06:58:58.944794 CST 完成。五份 summary、五份 episodes 记录与 50 个视频均完整，episode error 数为 0，本策略视频共 `5,524,115` bytes。
- coFT mugs 前四个完整点现为 `0% / 0% / 0% / 14%`。α=0.4 是首个非零候选，且五个固定 seeds 均取得至少一次成功，说明本次 OOD_MEDIUM 验证中性能起点不是单一 seed 偶然；这仍是 exploratory onset，剩余 α=0.5--0.9 未闭合前不执行预登记选点，也不外推为主评测或机制结论。
- pipeline 已自动加载 `[0.5, 0.5]` 的 `retain_coft_mugs_a050`。07:12:26 CST 时 seeds 1/21/41/61 已分别完整为 `3/10、2/10、2/10、5/10`，seed 81 暂为 `0/1`，累计 partial 为 `12/41`，不进入候选比较。evaluation pipeline PID `320418`、policy server PID `1041177`、evaluator PID `1055638` 均在 active service 中；policy server 为 `CUDA_VISIBLE_DEVICES=3`，evaluator 为 `CUDA_VISIBLE_DEVICES=`、`MUJOCO_EGL_DEVICE_ID=3`，GPU compute-process 审计只发现 policy PID `1041177` 位于物理 GPU 3，继续满足一块物理 GPU 的约束。
- 当前累计 40/54 个完整策略、204/270 个完整 jobs、2,040 个完整 episodes。2,000 个完整策略视频合计 `288,138,287` bytes、均值 `144,069.143500` bytes，按均值外推完整 sweep 约 `388,986,687.45` bytes。资源快照为 GPU 3 `77,115 MiB / 63%`、主机可用内存 `528,849,047,552` bytes、共享盘可用 `140,515,225,600` bytes、根分区可用 `12,610,883,584` bytes；α=0.4/0.5 非视频产物与从 α=0.4 起的 supervisor 日志片段均未检出 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260821T071226.json`。

## 2026-08-21 07:16 CST｜coFT mugs α=0.5 完成并推进 α=0.6

- `retain_coft_mugs_a050` 已完成固定 seeds 1/21/41/61/81，逐 seed 成功数为 `3 / 2 / 2 / 5 / 2`，合计 `14/50=28%`；最后一个 job 于 07:14:57.745391 CST 完成。五份 summary、五份 episodes 记录与 50 个视频均完整，episode error 数为 0，本策略视频共 `5,803,654` bytes。
- coFT mugs 前五个完整点现为 `0% / 0% / 0% / 14% / 28%`。α=0.5 较 α=0.4 多 7 个成功 episode，且五个固定 seeds 均至少成功两次，支持 α=0.4 之后的跨 seed 上升趋势；该趋势仍属 exploratory，α=0.6--0.9 未闭合前不正式选点，也不假设后续保持单调。
- pipeline 已自动加载 `[0.6, 0.4]` 的 `retain_coft_mugs_a060`。07:16:02 CST 时 seed 1 evaluator 已启动但尚无完整 episode。evaluation pipeline PID `320418`、policy server PID `1059041`、evaluator PID `1059836` 均在 active service 中；policy server 为 `CUDA_VISIBLE_DEVICES=3`，evaluator 为 `CUDA_VISIBLE_DEVICES=`、`MUJOCO_EGL_DEVICE_ID=3`，GPU compute-process 审计只发现 policy PID `1059041` 位于物理 GPU 3，继续满足一块物理 GPU 的约束。
- 当前累计 41/54 个完整策略、205/270 个完整 jobs、2,050 个完整 episodes。2,050 个完整策略视频合计 `293,941,941` bytes、均值 `143,386.312683` bytes，按均值外推完整 sweep 约 `387,143,044.24` bytes。资源快照为 GPU 3 `77,115 MiB / 3%`（采样时为新策略首个 episode 前的瞬时值）、主机可用内存 `530,254,800,896` bytes、共享盘可用 `140,511,547,392` bytes、根分区可用 `12,600,328,192` bytes；α=0.5/0.6 非视频产物与从 α=0.5 起的 supervisor 日志片段均未检出 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260821T071602.json`。

## 2026-08-21 07:32 CST｜coFT mugs α=0.6 完成并推进 α=0.7

- `retain_coft_mugs_a060` 已完成固定 seeds 1/21/41/61/81，逐 seed 成功数为 `3 / 3 / 2 / 5 / 4`，合计 `17/50=34%`；最后一个 job 于 07:30:29.931102 CST 完成。五份 summary、五份 episodes 记录与 50 个视频均完整，episode error 数为 0，本策略视频共 `5,810,607` bytes。
- coFT mugs 前六个完整点现为 `0% / 0% / 0% / 14% / 28% / 34%`。α=0.6 比 α=0.5 多 3 个成功 episode，且五个固定 seeds 均至少成功两次，支持 α=0.4 之后的跨 seed 上升趋势继续；增幅从 14 个百分点收窄到 6 个百分点，且 α=0.7--0.9 未闭合，因此不作显著性、单调性或机制结论。
- pipeline 已自动加载 `[0.7, 0.3]` 的 `retain_coft_mugs_a070`。07:32:05 CST 时 seed 1 暂为 `0/1`，属于不参与候选比较的 partial observation。evaluation pipeline PID `320418`、policy server PID `1076930`、evaluator PID `1077713` 均在 active service 中；policy server 为 `CUDA_VISIBLE_DEVICES=3`，evaluator 为 `CUDA_VISIBLE_DEVICES=`、`MUJOCO_EGL_DEVICE_ID=3`，GPU compute-process 审计只发现 policy PID `1076930` 位于物理 GPU 3，继续满足一块物理 GPU 的约束。
- 当前累计 42/54 个完整策略、210/270 个完整 jobs、2,100 个完整 episodes。2,100 个完整策略视频合计 `299,752,548` bytes、均值 `142,739.308571` bytes，按均值外推完整 sweep 约 `385,396,133.14` bytes。资源快照为 GPU 3 `77,115 MiB / 63%`、主机可用内存 `529,359,952,896` bytes、共享盘可用 `140,503,461,888` bytes、根分区可用 `12,592,394,240` bytes；α=0.6/0.7 非视频产物与从 α=0.6 起的 supervisor 日志片段均未检出 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260821T073205.json`。

## 2026-08-21 07:52 CST｜coFT mugs α=0.7 完成并推进 α=0.8

- `retain_coft_mugs_a070` 已完成固定 seeds 1/21/41/61/81，逐 seed 成功数为 `3 / 6 / 1 / 8 / 6`，合计 `24/50=48%`；最后一个 job 于 07:45:39.912937 CST 完成。五份 summary、五份 episodes 记录与 50 个视频均完整，episode error 数为 0，本策略视频共 `5,642,125` bytes。
- coFT mugs 前七个完整点现为 `0% / 0% / 0% / 14% / 28% / 34% / 48%`。α=0.7 比 α=0.6 多 7 个成功 episode并暂居首位；尽管所有 seeds 均有成功，逐 seed 仍从 1/10 到 8/10，提示验证方差不可忽略。α=0.8、0.9 未闭合前不执行正式选点，也不作显著性或稳定性结论。
- pipeline 已自动加载 `[0.8, 0.2]` 的 `retain_coft_mugs_a080`。07:52:08 CST 时 seeds 1/21 已分别完整为 `5/10、6/10`，累计 partial `11/20`；seed 41 刚启动且尚无完整 episode，不进入候选比较。evaluation pipeline PID `320418`、policy server PID `1094803`、evaluator PID `1102551` 均在 active service 中；policy server 为 `CUDA_VISIBLE_DEVICES=3`，evaluator 为 `CUDA_VISIBLE_DEVICES=`、`MUJOCO_EGL_DEVICE_ID=3`，GPU compute-process 审计只发现 policy PID `1094803` 位于物理 GPU 3，继续满足一块物理 GPU 的约束。
- 当前累计 43/54 个完整策略、217/270 个完整 jobs、2,170 个完整 episodes。2,150 个完整策略视频合计 `305,394,673` bytes、均值 `142,044.033953` bytes，按均值外推完整 sweep 约 `383,518,891.67` bytes。资源快照为 GPU 3 `77,115 MiB / 0%`（采样时为 seed 切换瞬时值）、主机可用内存 `529,847,456,768` bytes、共享盘可用 `140,493,742,080` bytes、根分区可用 `12,570,230,784` bytes；α=0.7/0.8 非视频产物与从 α=0.7 起的 supervisor 日志片段均未检出 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260821T075208.json`。

## 2026-08-21 08:16 CST｜coFT mugs 九点曲线闭合并选择 α=0.8

- `retain_coft_mugs_a080` 已完成固定 seeds 1/21/41/61/81，逐 seed成功数为 `5 / 6 / 5 / 4 / 6`，合计 `26/50=52%`；最后一个 job 于 08:00:01.187824 CST 完成，五份 summary、50 个视频与逐 episode 记录完整，episode error 数为 0，本策略视频共 `5,455,417` bytes。
- 论文设定的 `retain_coft_mugs_a090` 随后也已完成，逐 seed 为 `2 / 4 / 7 / 7 / 1`，合计 `21/50=42%`；最后一个 job 于 08:14:59.691414 CST 完成，五份 summary、50 个视频与逐 episode 记录完整，episode error 数为 0，本策略视频共 `5,700,252` bytes。
- coFT mugs 九点曲线为 `0% / 0% / 0% / 14% / 28% / 34% / 48% / 52% / 42%`。α=0.8 是唯一最高点，按预登记规则正式选择 α=0.8、线性合并权重 `[0.8, 0.2]` 进入后续主评测；并列规则未触发。论文 α=0.9 比所选点低 5 个成功 episode（10 个百分点），该差异仅用于冻结策略，不作显著性、机制或主性能解释。
- pipeline 已自动切换至 coFT basket，并加载 `[0.1, 0.9]` 的 `retain_coft_basket_a010`。08:16:03 CST 时 seed 1 evaluator 已启动但尚无完整 episode。evaluation pipeline PID `320418`、policy server PID `1130571`、evaluator PID `1131366` 均在 active service 中；policy server 为 `CUDA_VISIBLE_DEVICES=3`，evaluator 为 `CUDA_VISIBLE_DEVICES=`、`MUJOCO_EGL_DEVICE_ID=3`，GPU compute-process 审计只发现 policy PID `1130571` 位于物理 GPU 3，继续满足一块物理 GPU 的约束。
- 当前累计 45/54 个完整策略、225/270 个完整 jobs、2,250 个完整 episodes。2,250 个完整策略视频合计 `316,550,342` bytes、均值 `140,689.040889` bytes，按均值外推完整 sweep 约 `379,860,410.40` bytes。资源快照为 GPU 3 `77,115 MiB / 27%`、主机可用内存 `529,805,153,280` bytes、共享盘可用 `140,478,439,424` bytes、根分区可用 `12,551,684,096` bytes；mugs α=0.8/0.9、当前 basket α=0.1 非视频产物与从 mugs α=0.8 起的 supervisor 日志片段均未检出 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260821T081603.json`。

## 2026-08-21 08:35 CST｜coFT basket α=0.1 完成并推进 α=0.2

- `retain_coft_basket_a010` 已完成固定 seeds 1/21/41/61/81，逐 seed 均为 `0/10`，合计 `0/50=0%`；最后一个 job 于 08:30:45.426208 CST 完成。五份 summary、五份 episodes 记录与 50 个视频完整，episode error 数为 0，本策略视频共 `7,656,270` bytes。
- α=0.1 是 coFT basket 九点曲线的首个完整候选。该完整零成功只记录为 exploratory negative result：剩余八点尚未闭合，因此不提前应用“同成功率取更高 α”的预登记规则，也不据此选择 α=0.1、断言 coFT/RETAIN 在 basket 上失败或外推 ID/OOD/generalist 主性能。
- pipeline 已自动加载 `[0.2, 0.8]` 的 `retain_coft_basket_a020`。08:35:52 CST 时 seed 1 完整为 `0/10`，seed 21 暂为 `0/3`，累计 partial `0/13`，不参与候选比较。evaluation pipeline PID `320418`、policy server PID `1148460`、evaluator PID `1153105` 均在 active service 中；policy server 为 `CUDA_VISIBLE_DEVICES=3`，evaluator 为 `CUDA_VISIBLE_DEVICES=`、`MUJOCO_EGL_DEVICE_ID=3`，GPU compute-process 审计只发现 policy PID `1148460` 位于物理 GPU 3，继续满足一块物理 GPU 的约束。
- 当前累计 46/54 个完整策略、231/270 个完整 jobs、2,310 个完整 job episodes。2,300 个完整策略视频合计 `324,206,612` bytes、均值 `140,959.396522` bytes，按均值外推完整 sweep 约 `380,590,370.61` bytes。资源快照为 GPU 3 `77,115 MiB / 0%`（瞬时采样；策略与 evaluator 进程均存活）、主机可用内存 `528,862,589,952` bytes、共享盘可用 `140,465,577,984` bytes、根分区可用 `12,526,661,632` bytes；α=0.1/0.2 非视频产物与近期 service journal 均未检出 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260821T083552.json`。

## 2026-08-21 08:52 CST｜coFT basket α=0.2 完成并推进 α=0.3

- `retain_coft_basket_a020` 已完成固定 seeds 1/21/41/61/81，逐 seed 均为 `0/10`，合计 `0/50=0%`；最后一个 job 于 08:46:32.613590 CST 完成。五份 summary、五份 episodes 记录与 50 个视频完整，episode error 数为 0，本策略视频共 `7,435,305` bytes。
- coFT basket 的 α=0.1 与 α=0.2 两个完整点目前均为 0%。这是两个低 coFT 权重点的完整 exploratory negative results，但九点曲线仍有七点未完成；预登记的并列取更高 α 规则继续冻结，不能据此选择 α=0.2、断言整个方法失败或外推 ID/OOD/generalist 主结果。
- pipeline 已自动加载 `[0.3, 0.7]` 的 `retain_coft_basket_a030`。08:52:33 CST 时 seed 1 完整为 `0/10`，seed 21 暂为 `1/7`，累计 partial `1/17`；虽然这是 coFT basket sweep 中第一个观察到的成功 episode，但未完整 seed 不进入候选比较，也不据此解释性能阈值。evaluation pipeline PID `320418`、policy server PID `1166633`、evaluator PID `1171029` 均在 active service 中；policy server 为 `CUDA_VISIBLE_DEVICES=3`，evaluator 为 `CUDA_VISIBLE_DEVICES=`、`MUJOCO_EGL_DEVICE_ID=3`，GPU compute-process 审计只发现 policy PID `1166633` 位于物理 GPU 3，继续满足一块物理 GPU 的约束。
- 当前累计 47/54 个完整策略、236/270 个完整 jobs、2,360 个完整 job episodes。2,350 个完整策略视频合计 `331,641,917` bytes、均值 `141,124.22` bytes，按均值外推完整 sweep 约 `381,035,394` bytes。资源快照为 GPU 3 `77,115 MiB / 20%`、主机可用内存 `530,899,390,464` bytes、共享盘可用 `140,455,755,776` bytes、根分区可用 `12,512,292,864` bytes。α=0.2/0.3 server logs 各出现一次 `opening handshake failed` 的 Traceback；上下文均为 pipeline 的裸 TCP readiness probe 后正常 WebSocket connection open，属于既有无害告警，分类后未发现 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260821T085233.json`。

## 2026-08-21 09:12 CST｜coFT basket α=0.3 完成并推进 α=0.4

- `retain_coft_basket_a030` 已完成固定 seeds 1/21/41/61/81，逐 seed 成功数为 `0 / 1 / 0 / 0 / 0`，合计 `1/50=2%`；最后一个 job 于 09:02:19.646460 CST 完成。五份 summary、五份 episodes 记录与 50 个视频完整，episode error 数为 0，本策略视频共 `7,818,984` bytes。
- coFT basket 前三个完整点现为 `0% / 0% / 2%`。α=0.3 是首个非零完整候选，但唯一成功只来自 seed 21 的一个 episode；该结果只作为 exploratory onset 记录，不足以支持阈值、跨 seed 稳定性或正式选点，α=0.4--0.9 未闭合前预登记规则继续冻结。
- pipeline 已自动加载 `[0.4, 0.6]` 的 `retain_coft_basket_a040`。09:12:22 CST 时 seeds 1/21/41 已完整为 `1/10、1/10、2/10`，seed 61 暂为 `0/1`，累计 partial `4/31`，不进入候选比较。evaluation pipeline PID `320418`、policy server PID `1184588`、evaluator PID `1195741` 均在 active service 中；policy server 为 `CUDA_VISIBLE_DEVICES=3`，evaluator 为 `CUDA_VISIBLE_DEVICES=`、`MUJOCO_EGL_DEVICE_ID=3`，GPU compute-process 审计只发现 policy PID `1184588` 位于物理 GPU 3，继续满足一块物理 GPU 的约束。
- 当前累计 48/54 个完整策略、243/270 个完整 jobs、2,430 个完整 job episodes。2,400 个完整策略视频合计 `339,460,901` bytes、均值 `141,442.042083` bytes，按均值外推完整 sweep 约 `381,893,513.63` bytes。资源快照为 GPU 3 `77,115 MiB / 0%`（瞬时采样；进程仍存活）、主机可用内存 `527,806,047,232` bytes、共享盘可用 `140,441,235,456` bytes、根分区可用 `12,495,675,392` bytes。α=0.3/0.4 server logs 各出现一次 pipeline 裸 TCP readiness probe 引起的 WebSocket handshake Traceback，均随后正常建立连接；分类后未发现 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260821T091222.json`。

## 2026-08-21 09:35 CST｜coFT basket α=0.4、0.5 完成并推进 α=0.6

- `retain_coft_basket_a040` 已完整结束，固定 seeds 1/21/41/61/81 的成功数为 `1 / 1 / 2 / 0 / 1`，合计 `5/50=10%`；最后一个 job 于 09:17:53.741403 CST 完成。五份 summary、五份 episodes 记录和 50 个视频完整，episode error 数为 0，视频共 `7,515,910` bytes。
- `retain_coft_basket_a050` 随后也完整结束，逐 seed 为 `0 / 0 / 0 / 1 / 0`，合计 `1/50=2%`；最后一个 job 于 09:33:39.283985 CST 完成。五份 summary、五份 episodes 记录和 50 个视频完整，episode error 数为 0，视频共 `7,961,153` bytes。当前前五点曲线为 `0% / 0% / 2% / 10% / 2%`：α=0.4 暂时最高，但 α=0.5 的回落说明不能把前段解释成单调趋势；α=0.6--0.9 尚未闭合，预登记的选点和并列规则继续冻结。
- pipeline 已自动加载权重 `[0.6, 0.4]` 的 `retain_coft_basket_a060`。09:35:06 CST 时 seed 1 暂为 `0/1`，不参与候选比较。evaluation pipeline PID `320418`、policy server PID `1220429`、evaluator PID `1221282` 均在 active service 中；进程环境核验为 policy server `CUDA_VISIBLE_DEVICES=3`，evaluator `CUDA_VISIBLE_DEVICES=`、`MUJOCO_EGL_DEVICE_ID=3`，GPU compute-process 只有 policy PID `1220429` 位于物理 GPU 3，继续满足一块物理 GPU 的约束。
- 当前累计 50/54 个完整策略、250/270 个完整 jobs、2,500 个完整 episodes。2,500 个完整策略视频共 `354,937,964` bytes、均值 `141,975.1856` bytes，按均值外推完整 sweep 约 `383,333,001.12` bytes。资源快照为 GPU 3 `77,115 MiB / 24%`、主机可用内存 `530,060,430,336` bytes、共享盘可用 `140,420,382,720` bytes、根分区可用 `11,660,881,920` bytes。α=0.4/0.5/0.6 的 server logs 各有一次裸 TCP readiness probe 造成的 WebSocket handshake Traceback，均随后出现正常 connection open；分类后未发现 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260821T093515.json`。

## 2026-08-21 09:52 CST｜coFT basket α=0.6 完成并推进 α=0.7

- `retain_coft_basket_a060` 已完成固定 seeds 1/21/41/61/81，逐 seed 成功数为 `4 / 0 / 2 / 2 / 1`，合计 `9/50=18%`；最后一个 job 于 09:48:49.839447 CST 完成。五份 summary、五份 episodes 记录与 50 个视频完整，episode error 数为 0，本策略视频共 `7,581,917` bytes。
- coFT basket 前六个完整点现为 `0% / 0% / 2% / 10% / 2% / 18%`。α=0.6 暂为最高点，成功分布于四个 seeds，但 seed 21 为 0/10，且 α=0.7--0.9 未完成；因此只记录为 exploratory interim 新高，不据此选择 α、断言跨 seed 稳定性或外推主评测。
- pipeline 已自动加载权重 `[0.7, 0.3]` 的 `retain_coft_basket_a070`。09:52:49 CST 时 seed 1 完整为 `2/10`，seed 21 暂为 `0/1`，累计 partial `2/11`，不参与候选比较。evaluation pipeline PID `320418`、policy server PID `1238425`、evaluator PID `1242882` 均在 active service 中；policy server 为 `CUDA_VISIBLE_DEVICES=3`，evaluator 为 `CUDA_VISIBLE_DEVICES=`、`MUJOCO_EGL_DEVICE_ID=3`，GPU compute-process 只发现 policy PID `1238425` 位于物理 GPU 3，继续满足单卡约束。
- 当前累计 51/54 个完整策略、256/270 个完整 jobs、2,560 个完整 job episodes。2,550 个完整策略视频共 `362,519,881` bytes、均值 `142,164.659216` bytes，按均值外推完整 sweep 约 `383,844,579.88` bytes。资源快照为 GPU 3 `77,115 MiB / 63%`、主机可用内存 `529,991,792,640` bytes、共享盘可用 `138,188,726,272` bytes、根分区可用 `11,648,483,328` bytes。α=0.6/0.7 server logs 各出现一次裸 TCP readiness probe 引起的 WebSocket handshake Traceback，随后均正常 connection open；分类后未发现 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260821T095256.json`。

## 2026-08-21 10:12 CST｜coFT basket α=0.7 完成并推进 α=0.8

- `retain_coft_basket_a070` 已完成固定 seeds 1/21/41/61/81，逐 seed 成功数为 `2 / 4 / 1 / 2 / 2`，合计 `11/50=22%`；最后一个 job 于 10:03:38.952752 CST 完成。五份 summary、五份 episodes 记录和 50 个视频完整，episode error 数为 0，本策略视频共 `7,476,976` bytes。
- coFT basket 前七个完整点为 `0% / 0% / 2% / 10% / 2% / 18% / 22%`。α=0.7 暂为最高点且五个 seeds 均至少成功一次，但 α=0.8、0.9 尚未闭合；该结果只作为 exploratory interim 新高，不提前选择、作显著性解释或外推 ID/OOD/generalist 主性能。
- pipeline 已自动加载权重 `[0.8, 0.2]` 的 `retain_coft_basket_a080`。10:12:33 CST 时 seeds 1/21/41 已完整为 `2/10、5/10、7/10`，累计 `14/30`；seed 61 随后刚启动、尚无完整 episode。该 partial 结果不进入候选比较。evaluation pipeline PID `320418`、policy server PID `1256383`、evaluator PID `1267552` 均在 active service 中；policy server 为 `CUDA_VISIBLE_DEVICES=3`，evaluator 为 `CUDA_VISIBLE_DEVICES=`、`MUJOCO_EGL_DEVICE_ID=3`，GPU compute-process 只发现 policy PID `1256383` 位于物理 GPU 3，继续满足单卡约束。
- 当前累计 52/54 个完整策略、263/270 个完整 jobs、2,630 个完整 job episodes。2,600 个完整策略视频共 `369,996,857` bytes、均值 `142,306.483462` bytes，按均值外推完整 sweep 约 `384,227,505.35` bytes。资源快照为 GPU 3 `77,115 MiB / 20%`、主机可用内存 `530,679,820,288` bytes、共享盘可用 `132,529,700,864` bytes、根分区可用 `10,232,909,824` bytes；根分区仍可支持当前小日志写入，但余量下降，后续继续逐轮监控。α=0.7/0.8 server logs 各出现一次裸 TCP readiness probe 引起的 WebSocket handshake Traceback，随后均正常 connection open；分类后未发现 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、磁盘耗尽或 I/O error。机器可读快照为 `experiments/evaluation-alpha-sweep-progress-20260821T101242.json`。

## 2026-08-21 10:41 CST｜alpha sweep 全部闭合、修复 raw policy CLI 并恢复主评测

- `retain_coft_basket_a080` 已完成固定 seeds 1/21/41/61/81，成功数为 `2 / 5 / 7 / 2 / 4`，合计 `20/50=40%`，50 个视频共 `7,095,835` bytes；`retain_coft_basket_a090` 随后也完整结束，逐 seed 为 `5 / 0 / 1 / 2 / 1`，合计 `9/50=18%`，50 个视频共 `7,605,216` bytes。两者的 episode error 均为 0，最后一个 alpha job 于 10:32:59.137906 CST 完成。
- coFT basket 九点曲线最终为 `0% / 0% / 2% / 10% / 2% / 18% / 22% / 40% / 18%`。α=0.8 是唯一最高点，按预登记规则冻结为 `[0.8, 0.2]`；论文 α=0.9 低 11 个成功 episode（22 个百分点），只记录为候选选择差异，不作显著性或机制结论。至此 54/54 个候选策略、270/270 个 jobs、2,700/2,700 个 episodes 全部完成，episode error 为 0；2,700 个视频共 `384,697,908` bytes、均值 `142,480.706667` bytes。六项选择为 Task-FT stove/mugs/basket `0.8 / 0.7 / 0.9`、coFT stove/mugs/basket `0.8 / 0.8 / 0.8`。冻结文件 `/shared/.cache/retain/results/RETAIN-GPU-20260819-001/alpha_selection.json` 为 9,277 bytes，SHA-256 `df432313120276b766f4a03165e640ff46d45b648cca5490ae59fdaffd633b80`。
- 自动切入主评测时，首个 raw checkpoint 策略 `pretrain_117task` 的 server 以 return code 2 退出，`server.log` 明确为 `Unrecognized options: --port=18080`。检查 `serve_policy.py` 的 Tyro CLI 后确认：`--port` 是必须置于 `policy:checkpoint` 子命令之前的全局参数；alpha sweep 的 merged-policy 服务走另一条命令路径，因此此前全部结果未受影响。已把 raw 分支从 `... policy:checkpoint ... --port=18080` 改为 `... --port 18080 policy:checkpoint ...`；本地隔离缓存 `py_compile`、本地命令顺序断言、服务器 `py_compile`、服务器 Ruff 与服务器命令顺序断言全部通过，服务器脚本 SHA-256 为 `e9cf54a629a3fa7b5ea36da0b2f84656179ad9ef9ac2bbcf21557bc44c50cf21`。完整故障证据见 `experiments/incidents/main-eval-raw-policy-cli-order-20260821T1033.json`。
- 因训练和 alpha sweep 都已完成，且 10:39 CST 共享盘余量约 132.5 GB 低于完整 supervisor 的 150 GB 训练启动门禁，本次没有重复运行已经完成的训练级前置流程，而是在同一独立 transient service 中只恢复具有 completed-job 跳过语义的 evaluation pipeline。服务于 10:37:45 CST active，跳过全部 270 个 alpha jobs，并在物理 GPU 0 加载 pretraining final params；10:38:15 CST WebSocket server 已监听并启动 `pretrain_117task` 第 1/68 个 job。10:41:04 CST 该 ID stove job 为 `9/20` 个 partial episodes、成功 0、error 0，尚不进入结果比较。pipeline/policy/evaluator PID 分别为 `1292803 / 1292933 / 1293674`；GPU compute-process 只有 policy PID 位于物理 GPU 0，evaluator 为 `CUDA_VISIBLE_DEVICES=`、`MUJOCO_EGL_DEVICE_ID=0`，继续满足一块物理 GPU 约束。资源邻近快照：GPU 0 `77,050 MiB / 46%`、主机可用内存 `532,081,545,216` bytes、共享盘可用 `132,509,401,088` bytes、根分区可用 `10,171,502,592` bytes。机器可读总快照为 `experiments/evaluation-alpha-sweep-final-20260821T104104.json`。
- 10:43:55--10:52:18 CST，修复后的前四个主评测 jobs 已连续完整写出 summary：`pretrain_117task` 在 LIBERO-10 stove 的 ID seed 7 上为 `0/20`，OOD_MEDIUM 小平移 seeds 1/21/41 依次为 `0/10、0/10、0/10`；累计 50 条 episodes 均无 error，50 个视频共 `9,285,629` bytes。四个完整零成功仍只覆盖 stove，不外推为整个 pretraining policy、generalist retention 或方法比较结论。pipeline 已推进第 5/68 个 OOD_MEDIUM stove seed-61 job；10:52:30 CST 为 `0/1` partial，仍不进入比较。此时 `status.json` 共 274 个完整 jobs（270 alpha + 4 main）、`failed_job=null`，当前 evaluator PID 为 `1310451`；10:46 CST 以来的 service journal 致命错误扫描为零。物理 GPU 0 `77,050 MiB / 60%`，GPU compute process 仍只有 policy PID `1292933`，其余 7 张 GPU 空闲；主机可用内存 `532,163,594,240` bytes、共享盘可用 `132,509,077,504` bytes、根分区可用 `10,157,699,072` bytes。最新机器可读快照为 `experiments/evaluation-main-progress-20260821T105230.json`。

## 2026-08-21 11:13 CST｜pretrain 主评测推进至 11/68，并启用根盘余量监控

- `pretrain_117task` 的 stove OOD_MEDIUM 五个固定 seeds 已全部结束，逐 seed 为 `0 / 0 / 0 / 1 / 0`，合计 `1/50=2%`；ID seed 7 为 `0/20`。两组共 70 个完整 episodes、1 个成功且 error 为 0。它们只覆盖 stove，不外推到 mugs、basket、20 个 generalist tasks 或后续方法比较。
- OOD_HARD 已完整写出五个 jobs：set-0 seed 1、set-1 seed 1、set-0 seed 21、set-1 seed 21、set-0 seed 41 的成功数依次为 `0 / 0 / 0 / 1 / 1`，合计 `2/50`。由于完整设计还包含 set-1 seed 41 以及后续 seeds 61/81，该 `2/50` 只记录为 partial complete-job aggregate，不能作为完整 OOD_HARD 成功率。当前主评测累计 11/68 个 jobs、120 个 episodes、3 个成功、0 个 episode error；120 个视频共 `22,291,771` bytes。pipeline 已启动第 12 个 set-1 seed-41 job，11:13:04 CST 为 `0/1` partial，不参与比较。
- evaluation pipeline PID `1292803`、policy server PID `1292933`、evaluator PID `1334057` 均存活，`status.json` 共 281 个完整 jobs（270 alpha + 11 main）、`failed_job=null`；10:52 CST 以来的 service journal 致命错误扫描为零。进程环境仍为 policy server `CUDA_VISIBLE_DEVICES=0`，evaluator `CUDA_VISIBLE_DEVICES=`、`MUJOCO_EGL_DEVICE_ID=0`；复现 compute process 只有 policy server 位于物理 GPU 0，单卡约束成立。资源快照：GPU 0 `77,050 MiB / 19%`、主机可用内存 `532,013,467,648` bytes、共享盘可用 `132,496,007,168` bytes。
- 根分区可用空间从 10:52:30 CST 的 `10,157,699,072` bytes 降到 `8,656,228,352` bytes。定向检查显示：`/root/RETAIN_code` 可见占用约 `17.70` GB，整个可见 rootfs `du` 约 `24.71` GB，journal 约 `192` MiB；近 40 分钟 rootfs 上没有新修改的可见 100 MB 以上文件，当前 evaluation status/server/job logs 也都只是 KB 级。因此该约 1.5 GB 降幅不能归因于可见的本复现写入，暂不删除任何用户或外部数据；状态进入 elevated watch。若根盘余量接近预设的 4 GiB 门槛，将先保存完整作业状态，再把 evaluation state/log 迁移到 `/shared` 并利用 completed-job 跳过语义恢复，rollout 本身已经持续写入共享盘。机器可读快照为 `experiments/evaluation-main-progress-20260821T111304.json`。

## 2026-08-21 11:39 CST｜stove 三组评测闭合，并提前完成 evaluation 状态共享盘迁移

- 第 12--16 个 `pretrain_117task` 主评测 jobs 已依次完整结束：stove OOD_HARD 的 set-1 seed 41、set-0/1 seed 61、set-0/1 seed 81 均为 `0/10`。结合此前五个条件，完整 OOD_HARD 设计在 set-0/set-1、seeds 1/21/41/61/81 上的成功数为 `0/0、0/1、1/0、0/0、0/0`，合计 `2/100=2%`，100 条 episode 的 error 均为 0。至此 stove 的完整结果分组为 ID `0/20`、OOD_MEDIUM `1/50=2%`、OOD_HARD `2/100=2%`；不同协议组不合并成论文主指标。
- 第 17 个 job 是 mugs ID，20 条完整 episodes 为 `0/20`、error 0，20 个视频共 `2,157,223` bytes，于 11:33:53.237058 CST 提交。迁移重启后的第 18、19 个 mugs OOD_MEDIUM seeds 1/21 jobs 分别于 11:38:37.684371、11:41:52.255548 CST 正常提交，结果均为 `0/10`、error 0，视频分别为 `1,113,524 / 979,379` bytes。11:42:24 CST 时 pretrain 主评测累计 19/68 个 jobs、210 个完整 episodes、3 个成功、0 个 error，210 个视频共 `35,793,689` bytes；第 20 个 mugs OOD_MEDIUM seed-41 job 为 `0/2` partial，不参与比较。
- 根分区余量从 11:13 的 `8,656,228,352` bytes 继续下降到迁移前的 `6,614,155,264` bytes。该下降远大于同期本复现约十余 MB 的新增视频/状态，且此前可见文件审计不能解释；虽然仍高于 4 GiB 硬门槛，但按预案提前降低风险。11:33:53 CST 的 mugs ID 完整 job 提交后停止 transient service；停止时状态含 287 个 completed jobs（270 alpha + 17 main），没有失败 job。刚启动但未进入 `completed_jobs` 的下一项 partial 允许按幂等语义重跑，没有把 partial 当作结果。
- `run_evaluation_pipeline.py` 新增向后兼容的 `RETAIN_EVAL_STATE_ROOT` 环境变量，默认值仍是仓库内原目录。新的状态根为 `/shared/.cache/retain/evaluation-state/RETAIN-GPU-20260819-001/evaluation-pipeline`，service stdout/stderr 为同级 `evaluation-service.log`，`TMPDIR` 与 `XDG_CACHE_HOME` 也分别定向到 `/shared/.cache/retain/tmp/evaluation` 和 `/shared/.cache/retain/cache/evaluation`。本地/服务器 `py_compile`、环境路径断言和停止后首次目录 byte compare 均通过；服务器脚本 SHA-256 为 `c6d38095e8c5576a22ecc2d59d3a210f9b7f4430c0a136b968a9ad6ce9027d08`。
- 新服务于 11:34:51 CST active，pipeline/policy server PID 为 `1357474 / 1357605`。迁移前后 `completed_jobs` canonical SHA-256 均为 `779bc18107fb0e831a74d9cab6ff5e8091f101bd8a81bdc4ea918974f8180c8e`，`alpha_sweep` canonical SHA-256 均为 `f5c102dc0403ea05acb74ab77d6b82bc8f9ea96ef35e9a87b731c00ee690e4bb`；protocol ID 与 created_at 也一致。服务跳过 287 个完整 jobs，在物理 GPU 0 重新加载 raw pretraining checkpoint，并成功提交迁移后的首个 job。策略进程 `CUDA_VISIBLE_DEVICES=0`，evaluator 为 `CUDA_VISIBLE_DEVICES=`、`MUJOCO_EGL_DEVICE_ID=0`，GPU compute process 仍只有 policy server，单卡约束成立；新 service log 的致命错误扫描为零。
- 11:38:55 CST 资源快照为 GPU 0 `77,049 MiB / 0%`（瞬时利用率；进程存活）、主机可用内存 `530,433,349,632` bytes、共享盘可用 `121,135,685,632` bytes、根分区可用 `6,608,764,928` bytes。旧根盘状态目录仍完整保留，没有删除或覆盖用户/外部文件。根盘和共享盘余量都在发生明显的非本复现增量，后续继续逐心跳监控；若根盘逼近 4 GiB，将再次在完整 job 边界停止，以保护状态一致性。机器可读快照为 `experiments/evaluation-main-progress-20260821T113855.json`，迁移证据为 `experiments/incidents/evaluation-state-root-migration-20260821T1134.json`。

## 2026-08-21 11:52 CST｜mugs OOD_MEDIUM 五 seeds 闭合，进入 OOD_HARD

- `pretrain_117task` 的 mugs OOD_MEDIUM seeds 41/61/81 三个新 jobs 已依次于 11:45:07.381693、11:48:24.024987、11:51:42.796282 CST 完整提交，均为 `0/10`，episode error 均为 0，视频分别为 `1,067,205 / 1,072,292 / 1,096,602` bytes。连同 seeds 1/21，五个固定 seeds 全部为 0，完整评测组为 `0/50=0%`；这是 raw pretraining policy 在 mugs OOD_MEDIUM 上的完整 negative result，但不能外推 OOD_HARD、basket、generalist 或后续 Task-FT/RETAIN/coFT 策略。
- 11:52:29 CST 时主评测为 22/68 个完整 jobs、240 个完整 episodes、3 个成功、0 个 error；随后第 23 个 mugs OOD_HARD set-0 seed-1 job 于 11:55:17.398268 CST 完整提交为 `0/10`、error 0，10 个视频共 `1,239,796` bytes。最新累计为 23/68、250 个完整 episodes、3 个成功、0 个 error，250 个视频共 `40,269,584` bytes。跨 ID/OOD_MEDIUM/OOD_HARD 与不同任务直接合计的 `3/250` 只用于文件和状态完整性审计，不作为论文成功率。第 24 个 set-1 seed-1 job 在 11:55:43 CST 为 `0/1` partial，不参与组间比较。
- transient service 仍为 active/running，pipeline/policy/evaluator PID 分别为 `1357474 / 1357605 / 1379006`，`failed_job=null`；共享状态中的 completed jobs 为 293（270 alpha + 23 main）。policy server 环境为 `CUDA_VISIBLE_DEVICES=0`，evaluator 为 `CUDA_VISIBLE_DEVICES=`、`MUJOCO_EGL_DEVICE_ID=0`，两者的 state/tmp/cache 均继承共享盘路径；GPU compute-process 只发现 policy PID `1357605` 使用物理 GPU 0 的 `77,036 MiB`，单卡约束成立。迁移后的 service log 致命错误扫描仍为零。
- 11:52:29 CST 资源快照：GPU 0 `77,049 MiB / 9%`、主机可用内存 `530,055,651,328` bytes、根分区可用 `6,583,840,768` bytes、共享盘可用 `111,259,090,944` bytes；11:55:43 CST 的 post-capture 存储复核为根盘 `6,582,280,192` bytes、共享盘 `108,587,454,464` bytes。根盘相对 11:42 只下降约 11.4 MB，说明把复现状态/log/tmp/cache 迁移到共享盘后根盘压力暂时稳定；共享盘下降远大于新增 MB 级 rollout，不能归因于本复现。继续监控但不删除用户或外部数据。机器可读快照及 post-capture 字段为 `experiments/evaluation-main-progress-20260821T115229.json`。

## 2026-08-21 12:13 CST｜mugs OOD_HARD 完成 6/10 条件，根盘恢复 high watch

- 第 24--28 个 `pretrain_117task` 主评测 jobs 已连续完整提交：mugs OOD_HARD 的 set-1 seed 1、set-0/1 seed 21、set-0/1 seed 41 均为 `0/10`，episode error 均为 0，视频字节依次为 `1,446,554 / 1,226,891 / 1,462,763 / 1,105,590 / 1,556,350`。结合此前 set-0 seed 1，当前已完成的六个条件均为 0，总计 `0/60`；seeds 61/81 的四个条件未结束，因此只记录为 partial complete-job aggregate，不提前形成完整 OOD_HARD 结论。
- 主评测累计为 28/68 个 jobs、300 个完整 episodes、3 个成功、0 个 error，300 个视频共 `47,067,732` bytes。跨 stove/mugs 与 ID/OOD_MEDIUM/OOD_HARD 直接相加的 `3/300` 仅用于审计，不是论文指标。第 29 个 mugs OOD_HARD set-0 seed-61 job 在 12:13:22 CST 为 `0/1` partial、视频 `154,241` bytes，不参与比较。
- service 仍 active/running，pipeline/policy/evaluator PID 为 `1357474 / 1357605 / 1395920`，共享状态含 298 个 completed jobs（270 alpha + 28 main），`failed_job=null`。GPU compute-process 只有 policy PID `1357605` 使用物理 GPU 0 的 `77,036 MiB`，evaluator 不暴露 CUDA compute 并使用 `MUJOCO_EGL_DEVICE_ID=0`，单卡约束成立；迁移后 service log 的 CUDA OOM、RESOURCE_EXHAUSTED、Killed、RuntimeError、fatal、No space left 与 I/O error 扫描均为零。
- 12:13:22 CST 的资源快照：GPU 0 `77,049 MiB / 6%`、主机可用内存 `530,042,695,680` bytes、根分区可用 `6,076,080,128` bytes、共享盘可用 `106,268,033,024` bytes；`df` 分别显示 100% 与 99%。相较 11:55，根盘下降 `506,200,064` bytes、共享盘下降 `2,319,421,440` bytes，但新完整 rollout 视频只增加 `8,037,944` bytes。根盘近 20 分钟未发现新增可见 50 MB 以上文件，`lsof +L1` 也未发现 deleted-open 文件，故不把下降归因于本复现，也不删除用户或外部数据。根盘恢复 high watch；若逼近 `4,294,967,296` bytes，将在完整 job 提交后停止 service，依靠共享状态无损恢复。机器可读快照为 `experiments/evaluation-main-progress-20260821T121322.json`。

## 2026-08-21 12:38 CST｜mugs OOD_HARD 全部闭合，basket ID 完成

- 第 29--32 个 `pretrain_117task` 主评测 jobs 已完整提交：mugs OOD_HARD 的 set-0/set-1 seed 61 与 set-0/set-1 seed 81 均为 `0/10`、episode error 为 0，视频字节依次为 `1,226,177 / 1,547,698 / 1,258,405 / 1,500,689`。结合此前六个条件，完整的十条件评测组为 `0/100=0%`；因此 raw pretraining policy 在 mugs 上的三个独立完整组现为 ID `0/20`、OOD_MEDIUM `0/50`、OOD_HARD `0/100`。该完整 negative result 只适用于 raw pretraining policy，不能提前外推 Task-FT、RETAIN 或 coFT。
- 第 33 个 job 是 basket ID，20 条 episodes 为 `0/20`、error 0，20 个视频共 `2,557,775` bytes，于 12:33:48.846525 CST 完整提交。第 34 个 basket OOD_MEDIUM seed-1 随后也完整为 `0/10`、error 0，10 个视频共 `1,252,098` bytes，于 12:36:49.605767 CST 提交。主评测累计推进至 34/68 个 jobs、370 个完整 episodes、3 个成功、0 个 error，370 个视频共 `56,410,574` bytes；跨任务/协议的 `3/370` 仅作完整性审计。第 35 个 basket OOD_MEDIUM seed-21 在 12:38:18 CST 为 `0/4` partial、视频 `606,684` bytes，不参与正式比较。
- transient service 仍为 active/running，pipeline/policy/evaluator PID 分别为 `1357474 / 1357605 / 1419040`，共享状态含 304 个 completed jobs（270 alpha + 34 main），`failed_job=null`。GPU compute-process 只有 policy PID `1357605` 使用物理 GPU 0 的 `77,036 MiB`；evaluator 不暴露 CUDA compute 并使用 `MUJOCO_EGL_DEVICE_ID=0`，单卡约束成立。迁移后 service log 的 CUDA OOM、RESOURCE_EXHAUSTED、Killed、fatal、No space left 等扫描仍为零。
- 12:38:18 CST 资源快照：GPU 0 `77,049 MiB / 55%`、主机可用内存 `530,272,462,848` bytes、根分区可用 `5,478,400,000` bytes、共享盘可用 `106,240,618,496` bytes。相较 12:13，根盘下降 `597,680,128` bytes、共享盘下降 `27,414,528` bytes，而新增完整 rollout 视频只有 `9,342,842` bytes；共享状态迁移继续有效，但根盘仍受未归因的外部/隐藏压力。当前距 `4,294,967,296` bytes 保护线还有 `1,183,432,704` bytes，因此继续运行并高频监控；若逼近门槛，将只在完整 job 提交边界停止服务，保留共享状态无损恢复。机器可读快照为 `experiments/evaluation-main-progress-20260821T123818.json`；12:34 的中间边界快照也保留为 `experiments/evaluation-main-progress-20260821T123426.json`。

## 2026-08-21 12:52 CST｜basket OOD_MEDIUM 闭合，根盘越线后保护性停机

- 第 35--38 个 `pretrain_117task` 主评测 jobs 已完整提交：basket OOD_MEDIUM seeds 21/41/61/81 均为 `0/10`、episode error 为 0，视频字节依次为 `1,460,803 / 1,389,481 / 1,476,260 / 1,409,783`。连同 seed 1，五个固定 seeds 全部为 0，完整组为 `0/50=0%`；结合 basket ID `0/20`，该 negative result 仍只适用于 raw pretraining policy，不能外推 OOD_HARD、Task-FT、RETAIN 或 coFT。
- 主评测累计推进至 38/68 个完整 jobs、410 个 episodes、3 个成功、0 个 error，410 个视频共 `62,146,901` bytes；跨任务/协议的 `3/410` 只作文件完整性审计。第 39 个 basket OOD_HARD set-0 seed-1 在 12:51:52 CST 为 `0/8` partial、视频 `1,163,194` bytes，未进入 `completed_jobs`，因此不参与正式比较，恢复时允许幂等重跑。
- 12:51:52 CST 根分区可用空间已从上一快照的 `5,478,400,000` bytes 降至 `3,846,983,680` bytes，在约 14 分钟内减少约 1.63 GB，并低于 `4,294,967,296` bytes 的预登记保护线。为避免根盘耗尽，没有等待 partial 完成；12:52:12 CST 执行保护性停机，systemd 返回 `inactive / Result=success / ExecMainStatus=0`。停机后没有 evaluation、policy server 或 evaluator 进程，308 个完整 jobs（270 alpha + 38 main）、`failed_job=null` 与共享状态完整保留；第 39 个 job 未被错误标记为完成。
- 停机后的只读诊断显示根文件系统已用 `890,538,852,352` bytes，但 `/root` 可见目录仅约 `22,413,758,464` bytes、项目约 `17,703,084,032` bytes、VS Code server 约 `4,366,553,088` bytes、journal 约 `201,326,592` bytes。近 20 分钟只有一个超过 10 MB 的可见文件（24 MiB system journal），`lsof +L1` 未发现 deleted-open 文件，旧根盘 evaluation state 也只有约 3.51 MB；这些均不能解释占用或下降，因此归类为外部/隐藏存储压力，未删除任何用户或外部数据。
- 共享盘在同一区间只减少约 4.95 MB，与新增完整视频约 5.74 MB 同量级，说明此前 state/log/tmp/cache 迁移继续有效。为避免在 20 分钟心跳间反复启停，恢复门槛提高为根盘至少 `8,589,934,592` bytes（8 GiB）且稳定；达到后继续在一块物理 GPU 上从共享 completed-job 状态恢复，第 39 个 partial 将重跑。机器可读快照为 `experiments/evaluation-main-paused-root-protection-20260821T125212.json`，事件记录为 `experiments/incidents/root-space-hard-stop-20260821T1252.json`。

## 2026-08-21 13:04 CST｜按用户指令立即恢复，取消 root 触发停机

- 用户明确指出 evaluation 数据均在 `/shared`，要求继续评测且不得因 root 空间不足停止。复核确认结果目录、completed-job state、service log、`TMPDIR` 和 `XDG_CACHE_HOME` 的显式写路径确实全部位于 `/shared`；root 下降此前也已判定不能归因于本复现。因此 12:52 的停机判断属于过度保守，原 8 GiB 恢复门槛立即取消。历史停机与诊断记录保留作审计，不再作为后续控制策略。
- 原 transient unit 在停止后已被 systemd 卸载，不能直接 `start`。13:03:28 CST 使用同名 `retain-reproduction.service` 重建，working directory 为 `/root/RETAIN_code`，stdout/stderr 继续 append 到共享 service log，环境明确包含共享 `RETAIN_EVAL_STATE_ROOT`、`TMPDIR` 与 `XDG_CACHE_HOME`。服务 active，pipeline/policy/evaluator PID 为 `1436285 / 1436419 / 1437138`。
- pipeline 读取并跳过全部 308 个完整 jobs（270 alpha + 38 main），`failed_job=null`，在物理 GPU 0 重新加载 `pretrain_117task`；policy PID `1436419` 是唯一复现 compute process，显存 `77,036 MiB`，evaluator 不暴露 CUDA compute 并使用 EGL device 0，单卡约束成立。13:04:02 CST 日志已出现 `pretrain_117task 运行评测 1/30 (f2dea8b6639928c0)`，即未提交的第 39 个 basket OOD_HARD set-0 seed-1 正在幂等重跑。目录中存在上次 partial 文件，因此在新 summary 提交前不从视频数量推断本次进度或成功率。
- 13:04:26 CST 资源快照为主机可用内存 `531,692,941,312` bytes、共享盘可用 `106,244,542,464` bytes、root 可用 `3,825,750,016` bytes；root 继续只读监控，但不再触发主动停机。当前 service log 致命错误扫描为 0。恢复快照为 `experiments/evaluation-main-resumed-user-direction-20260821T130426.json`，策略纠正记录为 `experiments/incidents/root-space-stop-override-resume-20260821T1303.json`。

## 2026-08-21 13:12 CST｜恢复连续性验证通过，主评测进入 41/68

- 上次停机前未提交的第 39 个 basket OOD_HARD set-0 seed-1 job 已完成幂等重跑，并于 13:07:29.110156 CST 正式写出 summary：`0/10`、episode error 0，10 个视频共 `1,126,829` bytes。该提交把 `f2dea8b6639928c0` 加入共享 `completed_jobs`，证明 308-job 基线与 partial 重跑语义能够正确接续。
- 第 40 个 set-1 seed-1 job 随后于 13:10:56.786411 CST 正常提交，同样为 `0/10`、episode error 0，10 个视频共 `1,513,205` bytes。主评测累计推进至 40/68 个完整 jobs、430 个完整 episodes、3 个成功、0 个 error，430 个完整视频共 `64,786,935` bytes；跨不同任务和协议的 `3/430` 只用于状态/文件完整性审计，不作为论文成功率。
- 13:10:56.794521 CST pipeline 已启动第 41 个 basket OOD_HARD set-0 seed-21 job，job key 为 `80ce78040d031dff`，evaluator PID 更新为 `1444477`；捕获时尚无 summary，因此不计入正式结果。service 保持 active/running，pipeline/policy PID 为 `1436285 / 1436419`；GPU compute-process 仍只有 policy PID 使用物理 GPU 0 的 `77,036 MiB`，单卡约束成立，service log 致命错误扫描为 0。
- 资源邻近快照为 GPU 0 `77,049 MiB`、主机可用内存 `531,279,372,288` bytes、共享盘可用 `106,242,707,456` bytes、root 可用 `3,826,728,960` bytes。root 继续仅监测与记录，`root_space_is_stop_condition=false`、`resume_gate_bytes=null`；不再因该指标主动暂停评测。机器可读快照为 `experiments/evaluation-main-progress-20260821T131156.json`。

## 2026-08-21 13:18 CST｜basket OOD_HARD 推进至 4/10 条件

- 第 41 个 set-0 seed-21 job 于 13:14:11.978992 CST 正式提交，结果为 `0/10`、episode error 0，10 个视频共 `1,380,882` bytes；第 42 个 set-1 seed-21 job 于 13:17:28.736011 CST 正式提交，同样为 `0/10`、error 0，10 个视频共 `1,560,112` bytes。basket OOD_HARD 已完成 seeds 1/21 的四个 set 条件，合计 `0/40`；另外六个条件未闭合，不提前报告完整组成功率。
- 主评测累计推进至 42/68 个 jobs、450 个完整 episodes、3 个成功、0 个 error，450 个完整视频共 `67,727,929` bytes。跨任务/协议的 `3/450` 只作状态与文件完整性审计。13:17:28.746245 CST pipeline 已启动第 43 个 set-0 seed-41 job（`57523352762b2afb`），捕获时没有 summary，不计入正式结果。
- service 保持 active/running，pipeline/policy/evaluator PID 为 `1436285 / 1436419 / 1451377`；复现 GPU compute-process 只有 policy PID 位于物理 GPU 0，显存 `77,036 MiB`，恢复后 service log 致命错误扫描为 0。资源邻近快照为 GPU 0 `77,049 MiB / 63%`、主机可用 `531,377,475,584` bytes、共享盘可用 `106,238,623,744` bytes、root 可用 `3,818,774,528` bytes。root 只监测不停止的策略保持不变。机器可读快照为 `experiments/evaluation-main-progress-20260821T131824.json`。

## 2026-08-21 13:40 CST｜三个目标任务 ID/OOD 全部闭合，进入 generalist retention

- 第 43--48 个 basket OOD_HARD jobs 已连续正式提交：set-0/set-1 的 seeds 41/61/81 六个条件全部为 `0/10`、episode error 0，视频字节依次为 `1,445,735 / 1,657,026 / 1,224,867 / 1,557,066 / 1,412,968 / 1,612,438`。结合 seeds 1/21，完整十条件组为 `0/100=0%`。至此 raw `pretrain_117task` 的目标任务分组全部闭合：stove 为 ID `0/20`、OOD_MEDIUM `1/50=2%`、OOD_HARD `2/100=2%`；mugs 与 basket 的对应六组均为 0，全部无 episode error。每个协议组独立报告，不合并成论文主指标。
- pipeline 随后进入 20 个 LIBERO-90 generalist retention 任务。第 49 个 `close the top drawer of the cabinet` 为 `10/10=100%`，第 50 个 `pick up the tomato sauce and put it in the basket` 为 `7/10=70%`，均无 error；当前 partial aggregate 为 `17/20=85%`。这只覆盖 2/20 个任务，不能外推总体 retention，但它证明完整评测链路可以稳定产生正 success，从而显著降低“目标任务全低分源于全局 success 判定故障”的可能性。
- 主评测累计为 50/68 个 jobs、530 个完整 episodes、20 个成功、0 个 error，530 个视频共 `77,665,188` bytes。13:39:23.551336 CST 已启动第 51 个 generalist 任务 `put the frying pan on the stove`（`8910ae0ad9f2e68f`），捕获时尚无 summary，不计入正式指标。
- service active/running，pipeline/policy/evaluator PID 为 `1436285 / 1436419 / 1478422`；复现 compute-process 只有 policy PID 使用物理 GPU 0 的 `77,036 MiB`，恢复后致命错误扫描为 0。资源邻近快照为主机可用 `532,351,104,000` bytes、共享盘可用 `106,227,777,536` bytes、root 可用 `3,797,164,032` bytes；root 继续只监测、不停机。机器可读快照为 `experiments/evaluation-main-progress-20260821T133948.json`。

## 2026-08-21 13:58 CST｜generalist retention 推进至 15/20

- 新完成的第 51--63 个 jobs 将 generalist retention 推进至 15/20 个任务。逐任务成功数依次为 `[10, 7, 9, 8, 7, 0, 6, 9, 6, 10, 3, 6, 10, 7, 7]`，合计 `105/150=70%`、episode error 0；最低 `0/10`、最高 `10/10`，表明保留能力存在明显任务间差异。按已完整闭合的五任务 suite 分组，LIBERO-90 为 `41/50=82%`、LIBERO-Goal 为 `31/50=62%`、LIBERO-Object 为 `33/50=66%`；LIBERO-Spatial 尚未闭合，因此总体 20-task retention 仍待完成。
- 加上 48 个目标任务 ID/OOD jobs，主评测累计为 63/68 个 jobs、660 个完整 episodes、108 个成功、0 个 error，660 个视频共 `85,396,094` bytes。13:56:35.721705 CST 已启动第 64 个、即第 16 个 generalist 任务 `pick up the black bowl between the plate and the ramekin and place it on the plate`；13:57:04 CST 为 3/10 个视频、无 summary，不参与正式汇总。
- service active/running，pipeline/policy/evaluator PID 为 `1436285 / 1436419 / 1522142`；复现 compute-process 只有 policy PID 使用物理 GPU 0 的 `77,036 MiB`，恢复后致命错误扫描仍为 0。资源快照为主机可用 `532,109,122,560` bytes、共享盘可用 `106,219,036,672` bytes、root 可用 `3,789,455,360` bytes。root 仍仅记录、不触发停机。机器可读快照为 `experiments/evaluation-main-progress-20260821T135736.json`。

## 2026-08-21 14:20 CST｜pretrain 完整闭合并切换 taskft_stove

- `pretrain_117task` 最后五个 LIBERO-Spatial generalist 任务成功数为 `[4,7,4,8,5]/10`，因此 Spatial 完整组为 `28/50=56%`。完整 20-task generalist retention 为 `133/200=66.5%`、episode error 0；四个 suite 分别为 LIBERO-90 `82%`、Goal `62%`、Object `66%`、Spatial `56%`。逐任务范围 0--10，任务异质性结论保持。该策略全部 68 jobs 共 710 个 episodes、136 个成功、0 个 error、710 个视频 `87,920,498` bytes。
- 14:03:07.497831 CST 最后一个 pretrain job 正式提交后，pipeline 自动生成 `taskft_stove` 的 36 jobs，并从 `/shared/.cache/retain/checkpoints/retain_repro_task_ft_stove/paper_task_ft_stove/499` 在物理 GPU 0 加载新策略。其 ID 完整为 `15/20=75%`；OOD_MEDIUM seeds 1/21/41/61/81 的成功数为 `[3,4,3,6,5]`，完整组 `21/50=42%`，全部无 error。相较 raw pretraining stove 的 ID `0%` 与 OOD_MEDIUM `2%`，分别提高 75 和 40 个百分点；OOD_HARD 与 generalist 尚未完成，不能外推完整方法结论。
- 当前全局 main evaluation 已完成 74 个 jobs（pretrain 68 + taskft_stove 6）、780 个完整 episodes、172 个成功、0 个 error，780 个视频共 `98,168,008` bytes。14:18:15.120658 CST 已启动 taskft_stove 第 7/36 个、首个 OOD_HARD set-0 seed-1 job；14:19:34 CST 为 5/10 个视频、无 summary，不纳入正式结果。
- service active/running，pipeline/policy/evaluator PID 为 `1436285 / 1539074 / 1563006`；复现 compute-process 只有 policy PID 使用物理 GPU 0 的 `77,036 MiB`，恢复后致命错误扫描为 0。资源快照为主机可用 `531,726,139,392` bytes、共享盘可用 `106,197,417,984` bytes、root 可用 `3,759,292,416` bytes。root 仍只监测、不触发停机。机器可读快照为 `experiments/evaluation-main-progress-20260821T141939.json`。

## 2026-08-21 14:27 CST｜taskft_stove OOD_HARD 完成前三个条件

- taskft_stove 第 7--9 个 jobs 已正式提交：OOD_HARD set-0 seed 1 为 `5/10`、set-1 seed 1 为 `2/10`、set-0 seed 21 为 `4/10`，episode error 均为 0，视频字节依次为 `1,515,376 / 1,961,644 / 1,662,088`。三个完整条件合计 `11/30=36.7%`；相同条件下 raw pretraining 为 `0/30`。这是匹配扰动条件的 early paired evidence，但只完成 OOD_HARD 的 3/10 条件，不能外推整个组或 generalist retention。
- taskft_stove 当前完成 9/36 jobs、100 个完整 episodes、47 个成功、0 个 error，100 个完整视频共 `15,386,618` bytes。加上已闭合的 pretrain 策略，全局 main evaluation 为 77 个完整 jobs、810 个 episodes、183 个成功、0 个 error，810 个完整视频共 `103,307,116` bytes；跨策略直接合计只用于完整性审计。
- 14:25:27.711664 CST pipeline 启动第 10 个 OOD_HARD set-1 seed-21 job（`bbf16de0a5932b9f`）；14:27:00 CST 已观察 6/10 个视频、`937,115` bytes，无 summary，因此不计入正式结果。service active，pipeline/policy/evaluator PID 为 `1436285 / 1539074 / 1573267`；GPU compute-process 仍只有 policy PID 使用物理 GPU 0 的 `77,036 MiB`，恢复后致命错误扫描为 0。
- 14:27:00 CST 资源快照为 GPU 0 `77,049 MiB / 63%`、主机可用 `531,580,065,792` bytes、共享盘可用 `106,194,968,576` bytes、root 可用 `3,736,674,304` bytes。所有已知评测写路径仍在 `/shared`；root 只监测记录，不触发主动停机。机器可读快照为 `experiments/evaluation-main-progress-20260821T142700.json`。

## 2026-08-21 14:42 CST｜taskft_stove OOD_HARD 完整闭合，进入 generalist

- taskft_stove 第 10--16 个 jobs 已全部正式提交：set-1 seed 21 为 `4/10`，set-0/set-1 seed 41 为 `7/10、3/10`，set-0/set-1 seed 61 为 `8/10、4/10`，set-0/set-1 seed 81 为 `6/10、2/10`，episode error 全部为 0。加上前 3 个条件，完整十条件成功数为 `[5,2,4,4,7,3,8,4,6,2]`，OOD_HARD 为 `45/100=45%`；set-0 与 set-1 分别为 `60% / 30%`。
- raw pretraining 在完全相同 OOD_HARD 条件上为 `2/100=2%`，因此 Task-FT 绝对提高 43 个百分点。与 ID `75%` 对 raw `0%`、OOD_MEDIUM `42%` 对 raw `2%` 一起看，三个完整目标任务组的增益分别为 `+75 / +40 / +43` 个百分点。该结论限定于本次 batch-16 单卡缩小复现；是否伴随 generalist forgetting 尚待后续 20 个任务闭合。
- taskft_stove 的目标任务部分现为 16/36 jobs、170 个 episodes、81 个成功、0 个 error，170 个完整视频共 `26,160,568` bytes。连同 pretrain，main evaluation 累计为 84 个完整 jobs、880 个 episodes、217 个成功、0 个 error，880 个完整视频共 `114,081,066` bytes；跨策略总计只用于完整性审计。
- 14:41:41.618899 CST pipeline 已启动第 17 个、首个 generalist 任务 `close the top drawer of the cabinet`（`c7bc7631947ff2cc`）；14:42:18 CST 观察到 8/10 个视频、`238,629` bytes，无 summary，故未纳入正式 retention。service active，pipeline/policy/evaluator PID 为 `1436285 / 1539074 / 1597098`；复现 compute-process 仍只有 policy PID 使用物理 GPU 0 的 `77,036 MiB`，致命错误扫描为 0。
- 14:42:18 CST 资源快照为 GPU 0 `77,049 MiB / 16%`、主机可用 `531,498,749,952` bytes、共享盘可用 `106,175,877,120` bytes、root 可用 `3,259,088,896` bytes。GPU 4--7 的高显存占用不属于本复现可见 compute process；本复现仍严格使用物理 GPU 0。root 继续只监测、不触发停机。机器可读快照为 `experiments/evaluation-main-progress-20260821T144218.json`。

## 2026-08-21 14:59 CST｜taskft_stove generalist 前两套件闭合

- taskft_stove 第 17--26 个 generalist jobs 已连续提交，逐任务成功数为 `[10,6,4,2,2,0,7,8,0,6]/10`，episode error 全部为 0；同任务 raw pretraining 为 `[10,7,9,8,7,0,6,9,6,10]/10`。前 10 个任务合计为 `45/100=45%`，对照为 `72/100=72%`，配对绝对差为 `-27` 个百分点。
- 两个 suite 已分别闭合：LIBERO-90 为 Task-FT `24/50=48%`、pretrain `41/50=82%`，差 `-34pp`；LIBERO-Goal 为 Task-FT `21/50=42%`、pretrain `31/50=62%`，差 `-20pp`。该结果与 stove 目标任务 ID/OOD 的 `+75/+40/+43pp` 增益共同揭示明显 adaptation–retention trade-off；剩余 Object/Spatial 未完成，所以当前只标为 partial paired evidence。
- taskft_stove 当前完成 26/36 jobs、270 个完整 episodes、126 个成功、0 个 error，270 个完整视频共 `33,977,663` bytes。连同 pretrain，main evaluation 累计为 94 个完整 jobs、980 个 episodes、262 个成功、0 个 error，980 个完整视频共 `121,898,161` bytes；跨策略总计仅作完整性审计。
- 14:57:44.788631 CST pipeline 启动第 27 个、首个 LIBERO-Object 任务 `pick up the alphabet soup and place it in the basket`（`82e909c2826b020c`）；14:58:47 CST 为 5/10 个视频、`381,703` bytes，无 summary，不计入正式汇总。service active，pipeline/policy/evaluator PID 为 `1436285 / 1539074 / 1630762`；复现 compute-process 仍只有 policy PID 使用物理 GPU 0 的 `77,036 MiB`，致命错误扫描为 0。
- 14:58:47 CST 资源快照为 GPU 0 `77,049 MiB / 44%`、主机可用 `531,336,465,408` bytes、共享盘可用 `105,790,889,984` bytes、root 可用 `2,839,920,640` bytes。GPU 4--6 的高显存占用不属于本复现可见 compute process；本复现仍只使用物理 GPU 0。root 继续仅监测，不触发停机。机器可读快照为 `experiments/evaluation-main-progress-20260821T145847.json`。

## 2026-08-21 15:19 CST｜taskft_stove 完整闭合并切换 RETAIN α=0.8

- taskft_stove 最后 10 个 generalist jobs 已提交：Object 五任务为 `[0,0,0,0,4]/10`，Spatial 五任务为 `[3,1,1,5,4]/10`。完整 generalist 逐任务成功数为 `[10,6,4,2,2,0,7,8,0,6,0,0,0,0,4,3,1,1,5,4]`，合计 `63/200=31.5%`；raw pretraining 为 `133/200=66.5%`，差 `-35pp`。suite 对照分别为 LIBERO-90 `48% vs 82%`、Goal `42% vs 62%`、Object `8% vs 66%`、Spatial `28% vs 56%`。
- taskft_stove 36/36 jobs 共 370 个 episodes、144 个成功、0 个 error，370 个视频 `40,541,120` bytes。目标任务增益 `+75/+40/+43pp` 与 retention `-35pp` 同时成立，形成完整 adaptation–retention trade-off。该结论限定于 batch-16 单卡缩小复现。
- 15:13:34.825992 CST pipeline 自动加载选定的 `retain_taskft_stove_a080`，policy server 以线性权重 `[0.8,0.2]` 合并 Task-FT step-499 与 pretrain step-9999。代码审计确认主计划仍含 36 jobs，但该策略的 5 个 OOD_MEDIUM jobs 已在 alpha sweep 完成并由 completed key 幂等跳过，因此日志显示 31 个剩余 jobs；这五个验证 jobs 为 `30/50=60%`，是 selection-set estimate，不是独立主评测。
- 第一个独立 ID job 于 15:17:52.535694 CST 提交为 `15/20=75%`、error 0、20 个视频 `2,438,740` bytes，与 raw Task-FT ID 相同。第 2/31 个剩余 job 为 OOD_HARD set-0 seed-1（`4c7c3de7fec00d53`）；15:19:26 CST 为 6/10 个视频、`924,981` bytes，无 summary，不计入正式比较。
- main evaluation 当前累计 105 个独立完整 jobs、1,100 个 episodes、295 个成功、0 个 error，1,100 个完整视频共 `130,900,358` bytes；status 中另含 270 个 alpha jobs，共 375 completed keys。service active，pipeline/policy/evaluator PID 为 `1436285 / 1664634 / 1671893`；policy server 是唯一可见复现 compute process，使用物理 GPU 0 的 `77,036 MiB`。15:19:26 CST 主机可用 `529,995,695,104` bytes、共享盘可用 `105,789,579,264` bytes、root 可用 `2,833,588,224` bytes；root 继续只监测、不停机。快照为 `experiments/evaluation-main-progress-20260821T151926.json`。

## 2026-08-21 15:27 CST｜RETAIN α=0.8 完成前三个 OOD_HARD 条件

- `retain_taskft_stove_a080` 第 2--4 个剩余 jobs 已正式提交：OOD_HARD set-0 seed 1 为 `5/10`、set-1 seed 1 为 `4/10`、set-0 seed 21 为 `5/10`，episode error 全部为 0，视频字节分别为 `1,532,703 / 1,727,718 / 1,502,584`。前三个条件合计 `14/30=46.7%`；raw Task-FT 的匹配三条件为 `11/30=36.7%`，差 10 个百分点，但当前只完成 3/10 条件，不提前外推完整 OOD_HARD。
- 该 merged policy 当前独立主评测完成 4 个 jobs（ID 加三个 OOD_HARD）、50 个 episodes、29 个成功。其五个 OOD_MEDIUM selection jobs 为 `30/50=60%`，继续单列为选参集复用而非独立证据。全局独立 main aggregate 为 108 个 jobs、1,130 个 episodes、309 个成功、0 个 error，1,130 个视频共 `135,663,363` bytes；跨策略总计只作完整性审计。
- 15:24:38 CST pipeline 启动第 5/31 个剩余 job，即 OOD_HARD set-1 seed 21（`5b96b47b57f95326`）；15:26:36 CST 已观察 7/10 个视频、`1,382,076` bytes，无 summary，不计入正式结果。service active/running，pipeline/policy/evaluator PID 为 `1436285 / 1664634 / 1682099`；复现 compute-process 只有 policy PID 使用物理 GPU 0 的 `77,036 MiB`，致命错误扫描为 0。
- 同期主机可用内存 `529,944,724,480` bytes、共享盘可用 `105,783,635,968` bytes、root 可用 `2,832,859,136` bytes。全部已知评测写路径仍位于 `/shared`；root 继续只监测记录，绝不作为主动停止条件。机器可读快照为 `experiments/evaluation-main-progress-20260821T152636.json`。

## 2026-08-21 15:38 CST｜服务器连接临时超时，保持后台实验不干预

- 本轮在读取本地三份连续性记录后，先并行检查 shared status、rollout summaries、systemd、GPU 与资源；四条 SSH 连接均在约 8 秒后超时。随后又进行两次独立重试，仍然是 `port 22: Operation timed out`，因此当前只能判定连接层暂不可达，不能判定 evaluation service 或具体 job 失败。
- 最后一个经过验证的服务器边界仍是 15:26:36 CST：`retain_taskft_stove_a080` 已有 378 个 completed keys（270 alpha + 108 main）、`failed_job=null`，第四个 OOD_HARD 条件正在运行，service active，复现 compute 只在物理 GPU 0，致命错误扫描为 0。systemd 服务与全部 evaluation state/log/tmp/cache 均独立于当前 SSH 会话并位于 `/shared`，因此没有执行停止、重启或修改远端状态的动作。
- 当前状态明确标记为 `remote runtime unknown`，不把连接超时写成实验失败，也不根据旧 partial 推断新结果。待连接恢复后，先对比 completed-job keys、最新 summary 和 service PID，验证离线期间的连续性，再补录进展。事件快照为 `experiments/incidents/server-connectivity-timeout-20260821T153756.json`；root 空间策略不变，仍只监测、不触发停机。

## 2026-08-21 15:51 CST｜clean host reboot 后恢复，RETAIN OOD_HARD 完整闭合

- 连接于 15:39:06 CST 恢复后确认主机 boot time 为 15:38:53。上一 boot 的 journal 显示 15:34:26 systemd 有序将 `retain-reproduction.service` 标记为 `Deactivated successfully`，随后停止系统 targets、卸载 `/shared` 并于 15:34:30 进入 `System Halt`；因此服务消失源于外部 clean host reboot，不是评测代码失败、OOM、磁盘写满或 agent 因 root 空间停止。
- 重启后 `/shared` 的 status 完整保留 382 个 completed keys（270 alpha + 112 main）、`failed_job=null`。最后一个已提交 job 是 RETAIN OOD_HARD set-0 seed 61（`7/10`）；set-1 seed 61 已启动但未提交，故没有错误计入结果。15:42:51 CST 按原工作目录和共享 `RETAIN_EVAL_STATE_ROOT/TMPDIR/XDG_CACHE_HOME/service log` 重建同名 transient service，pipeline 幂等跳过完整 pretrain、Task-FT 及 RETAIN 已提交的 8 个独立 jobs。
- policy server 重新以 `[0.8,0.2]` 合并 Task-FT step-499 与 pretrain step-9999，并成为物理 GPU 0 上唯一复现 compute process。被中断的 set-1 seed 61 于 15:46:16 正式重跑提交为 `4/10`，随后 set-0/set-1 seed 81 分别为 `7/10、5/10`；三项 episode error 均为 0。completed keys 连续增长至 385，证明 shared completed-job 恢复语义通过实际提交验证。
- `retain_taskft_stove_a080` 的完整 OOD_HARD 十条件为 `[5,4,5,2,7,4,7,4,7,5]/10`，合计 `50/100=50%`。raw Task-FT 相同协议为 `45/100=45%`，raw pretraining 为 `2/100=2%`，绝对差分别为 `+5pp / +48pp`。相对 Task-FT 的 +5pp 不作显著性解释，需结合后续 retention 判断合并的整体价值；OOD_MEDIUM `30/50` 仍明确标记为用于选 α 的 validation set 而非独立证据。
- 15:50:19 CST OOD_HARD 最后条件提交后，pipeline 自动启动首个 generalist 任务 `close the top drawer of the cabinet`（`1053b3adb264deeb`）；15:50:41 为 5/10 个视频、`240,869` bytes，无 summary，不计入正式 retention。此时 service active，pipeline/policy/evaluator PID 为 `1886 / 2148 / 14223`，GPU 0 使用 `77,049 MiB`，致命错误扫描为 0；主机可用内存 `529,289,937,920` bytes、共享盘可用 `105,767,645,184` bytes、root 可用 `84,394,778,624` bytes。root 仍只监测、不触发停机。事件记录为 `experiments/incidents/host-clean-reboot-resume-20260821T154251.json`，进展快照为 `experiments/evaluation-main-resumed-host-reboot-20260821T155041.json`。

## 2026-08-21 15:59 CST｜RETAIN generalist 的 LIBERO-90 完整闭合

- `retain_taskft_stove_a080` 的前五个 generalist jobs 已正式提交，LIBERO-90 逐任务成功数为 `[9,7,5,3,3]/10`，合计 `27/50=54%`、episode error 0，视频字节依次为 `390,161 / 1,016,850 / 1,051,042 / 989,837 / 997,138`。
- 完全配对的 raw Task-FT 为 `[10,6,4,2,2]/10`、`24/50=48%`，raw pretraining 为 `[10,7,9,8,7]/10`、`41/50=82%`。因此合并策略相对 Task-FT 恢复 6pp，但仍比 pretrain 低 28pp；原 Task-FT 的 34pp 遗忘缺口只回收约 `6/34=17.6%`。这是一个完整 suite 的 partial-retention-recovery 证据，不外推到其余 15 个任务。
- main evaluation 累计为 120 个独立完整 jobs、1,250 个 episodes、372 个成功、0 个 error，1,250 个视频共 `151,011,184` bytes；status 另含 270 个 alpha jobs，共 390 completed keys。跨策略总计只用于完整性审计。
- 15:58:32 CST pipeline 启动第 6/20 个 generalist、首个 LIBERO-Goal 任务 `open the middle drawer of the cabinet`（`2c43f964b2e6dcdd`）；15:59:23 为 2/10 个视频、`167,376` bytes，无 summary，不计入正式 Goal 指标。service active，pipeline/policy/evaluator PID 为 `1886 / 2148 / 31122`；本复现唯一 compute PID `2148` 仍位于物理 GPU 0、显存 `77,036 MiB`，fatal scan 为 0。GPU 4--7 的约 79.6--79.7 GiB 外部负载与本复现无关。
- 同期主机可用内存 `529,504,918,528` bytes、共享盘可用 `105,762,168,832` bytes、root 可用 `84,457,963,520` bytes。root 继续只监测、不触发停机。机器可读快照为 `experiments/evaluation-main-progress-20260821T155923.json`。

## 2026-08-21 16:21 CST｜RETAIN α=0.8 完整闭合并进入论文 α=0.9 对照

- `retain_taskft_stove_a080` 剩余 15 个 generalist jobs 已连续提交、episode error 均为 0。LIBERO-Goal 逐任务为 `[0,5,10,0,6]/10`，合计 `21/50=42%`；与 raw Task-FT 的 `21/50=42%` 相同，比 pretrain 的 `31/50=62%` 低 20pp。LIBERO-Object 为 `[0,0,1,1,5]/10`、`7/50=14%`，比 Task-FT 的 `4/50=8%` 高 6pp、比 pretrain 的 `33/50=66%` 低 52pp。LIBERO-Spatial 为 `[6,3,1,6,5]/10`、`21/50=42%`，比 Task-FT 的 `14/50=28%` 高 14pp、比 pretrain 的 `28/50=56%` 低 14pp。
- 加上已闭合的 LIBERO-90 `[9,7,5,3,3]`、`27/50=54%`，α=0.8 的完整 20-task generalist retention 为 `76/200=38%`。raw Task-FT 为 `63/200=31.5%`，所以合并恢复 `+6.5pp`；raw pretraining 为 `133/200=66.5%`，所以仍有 `-28.5pp`。按 episode 成功数，原 Task-FT 遗忘缺口为 `133-63=70`，合并只回收 `76-63=13`，恢复比例约 `18.6%`。suite 恢复比例约为 LIBERO-90 `17.6%`、Goal `0%`、Object `10.3%`、Spatial `50%`，说明改善主要来自 Spatial，不能把总体正向点估计解释为均匀恢复。
- 该 merged policy 的 31/31 个独立 main jobs 共 320 个 episodes、141 个成功、0 个 error，320 个视频共 `32,009,615` bytes；其中 ID 为 `15/20`、OOD_HARD 为 `50/100`、generalist 为 `76/200`。OOD_MEDIUM 的 `30/50` 来自选出 α=0.8 的同一 alpha-selection validation set，继续单列且不计入独立 main aggregate。全局独立 main evaluation 已推进至 135 个 jobs、1,400 个 episodes、421 个成功、0 个 error，1,400 个视频共 `160,471,233` bytes；status 连同 270 个 alpha jobs 共 405 个 completed keys，`failed_job=null`。
- 16:20:10 CST pipeline 自动开始加载论文参考 alpha 的 `retain_taskft_stove_a090`（权重 `[0.9,0.1]`），其五个 OOD_MEDIUM selection-sweep jobs 为 `28/50=56%`、会被幂等复用但不作为独立主证据。16:20:52 已启动第 1/31 个独立 ID job（`8d83c799ae022abd`）；16:21:08 evaluator 正在编译，尚无视频或 summary，因此不计入结果。
- service active，pipeline/policy/evaluator PID 为 `1886 / 82166 / 82971`；唯一复现 compute PID `82166` 使用物理 GPU 0 的 `77,032 MiB`，单卡约束满足，fatal scan 为 0。资源快照为主机可用 `521,082,287,104` bytes、共享盘可用 `105,743,151,104` bytes、root 可用 `86,836,469,760` bytes；所有已知写路径仍在 `/shared`，root 仅监测且绝不作为主动停机条件。机器可读快照为 `experiments/evaluation-main-progress-20260821T162108.json`。

## 2026-08-21 16:28 CST｜论文 α=0.9 完成 ID 与首个 hard 条件

- `retain_taskft_stove_a090` 第 1 个独立 job 于 16:24:23 CST 提交：ID 为 `14/20=70%`、episode error 0，20 个视频共 `2,448,769` bytes。它比预登记选出的 α=0.8 和 raw Task-FT 的匹配 ID `15/20=75%` 低 5pp；单个 ID 组只支持“α=0.9 未优于 α=0.8”的当前点估计，不替代后续 robustness/retention 比较。
- 首个 OOD_HARD set-0 seed 1 于 16:26:31 CST 提交为 `5/10`、error 0、视频 `1,489,311` bytes，与 α=0.8 和 raw Task-FT 的同一条件均相同。第二个 set-1 seed 1（`67fc87c373642ac0`）已运行至 5/10 个视频、`954,545` bytes，尚无 summary，不计入正式汇总。α=0.9 的 OOD_MEDIUM `28/50=56%` 是此前 selection sweep 的 paper-alpha reference，继续明确标为非独立证据。
- main evaluation 累计 137 个独立完整 jobs、1,430 个 episodes、440 个成功、0 个 error，1,430 个视频共 `164,409,313` bytes；status 总计 407 个 completed keys、`failed_job=null`。service active，pipeline/policy/evaluator PID 为 `1886 / 82166 / 92874`；唯一复现 compute PID `82166` 在物理 GPU 0 使用 `77,036 MiB`，fatal scan 为 0。
- 16:27:57 CST 资源为 GPU 0 `77,049 MiB / 41%`、主机可用 `528,114,361,344` bytes、共享盘可用 `105,737,269,248` bytes、root 可用 `86,408,335,360` bytes。写路径仍全部定向 `/shared`；root 继续只监测、不触发停机。机器可读快照为 `experiments/evaluation-main-progress-20260821T162757.json`。
- 截至 16:31:09 CST，随后两个 OOD_HARD 条件也已正式提交：set-1 seed 1 为 `2/10`、视频 `1,946,658` bytes；set-0 seed 21 为 `9/10`、视频 `1,126,355` bytes，error 均为 0。α=0.9 前三个 hard 条件因此为 `[5,2,9]/10`、partial `16/30=53.3%`；同三条件 α=0.8 为 `[5,4,5]`、Task-FT 为 `[5,2,4]`，但 3/10 条件不足以形成完整组结论。第 4 个 set-1 seed 21（`f7f2dcec26d24d98`）已观察 1/10 个视频、`108,593` bytes，无 summary。main aggregate 更新至 139 jobs、1,450 episodes、451 successes、0 errors、`167,482,326` video bytes；live status 为 409 completed keys、`failed_job=null`。

## 2026-08-21 16:38 CST｜论文 α=0.9 完成前六个 hard 条件

- α=0.9 随后三个 OOD_HARD jobs 均正式提交、episode error 为 0：set-1 seed 21 为 `6/10`、视频 `1,437,091` bytes；set-0 seed 41 为 `6/10`、视频 `1,322,655` bytes；set-1 seed 41 为 `3/10`、视频 `1,741,211` bytes。完整前六条件为 `[5,2,9,6,6,3]/10`，partial `31/60=51.7%`。
- 完全匹配的预登记 α=0.8 前六条件为 `[5,4,5,2,7,4]`、`27/60=45%`，raw Task-FT 为 `[5,2,4,4,7,3]`、`25/60=41.7%`；α=0.9 当前分别高 `6.7pp/10pp`。优势主要由 set-0 seed 21 的 `9/10` 驱动，尚余四个条件，因此不报告完整 OOD_HARD 优势或显著性。
- main evaluation 截至 16:37:39 CST 为 142 个独立完整 jobs、1,480 个 episodes、466 个成功、0 个 error，1,480 个视频共 `171,983,283` bytes；status 共 412 completed keys、`failed_job=null`。16:37:39 已启动第 8/31 个独立 job、OOD_HARD set-0 seed 61（`b9b59541466c2cc0`），16:37:41 尚无视频或 summary。
- 16:37:55 CST service active，pipeline/policy/evaluator PID 为 `1886 / 82166 / 110081`；唯一复现 compute PID `82166` 使用物理 GPU 0 的 `77,036 MiB`。GPU 0 为 `77,049 MiB / 56%`，主机可用 `528,466,063,360` bytes、共享盘可用 `105,727,508,480` bytes、root 可用 `85,137,375,232` bytes；fatal scan 为 0。全部写路径在 `/shared`，root 继续只监测、不停止。快照为 `experiments/evaluation-main-progress-20260821T163755.json`。

## 2026-08-21 16:58 CST｜论文 α=0.9 OOD_HARD 与 LIBERO-90 闭合

- α=0.9 最后四个 OOD_HARD 条件已提交：set-0/set-1 seed 61 为 `5/10、5/10`，set-0/set-1 seed 81 为 `7/10、0/10`；视频字节分别为 `1,503,982 / 1,569,738 / 1,313,511 / 2,024,903`，episode error 均为 0。完整十条件为 `[5,2,9,6,6,3,5,5,7,0]/10`，合计 `48/100=48%`；set-0 为 `32/50=64%`、set-1 为 `16/50=32%`，仍显示两个 hard 条件族的难度差异。
- 同协议下预登记选出的 α=0.8 为 `50/100=50%`，raw Task-FT 为 `45/100=45%`，pretrain 为 `2/100=2%`；因此论文 α=0.9 分别为 `-2pp / +3pp / +46pp`。α=0.9 的独立 ID 也为 `70%`、低于 α=0.8 与 Task-FT 的 `75%`，所以当前两个独立目标任务协议均不支持 α=0.9 优于选出的 α=0.8。
- α=0.9 的 LIBERO-90 五任务随后闭合为 `[10,7,0,4,3]/10`、`24/50=48%`。它与 raw Task-FT 的 `24/50=48%` 完全持平，比 α=0.8 的 `27/50=54%` 低 6pp，比 pretrain 的 `41/50=82%` 低 34pp；在首个完整 retention suite 上，α=0.9 没有恢复 Task-FT forgetting。该结果与 alpha sweep 选择 α=0.8 而非 paper α=0.9 的方向一致，但其余 15 个 generalist tasks 未闭合前不形成 20-task 总结论。
- 首个 LIBERO-Goal 任务 `open the middle drawer of the cabinet` 于 16:57:17 CST 提交为 `0/10`、error 0、视频 `874,640` bytes。第二个 Goal 任务 `put the bowl on the plate`（`49fa96d6f09397b4`）在 16:57:48 为 5/10 个视频、`193,297` bytes，无 summary，不计入比较。
- main evaluation 截止该提交边界为 152 个独立完整 jobs、1,580 个 episodes、507 个成功、0 个 error，1,580 个视频共 `183,903,770` bytes；status 共 422 completed keys、`failed_job=null`。16:58:00 CST service active，pipeline/policy/evaluator PID 为 `1886 / 82166 / 144220`，唯一复现 compute PID `82166` 位于物理 GPU 0、显存 `77,036 MiB`，fatal scan 为 0。主机可用 `528,146,241,536` bytes、共享盘可用 `103,754,543,104` bytes、root 可用 `84,782,907,392` bytes；root 继续只监测、不停机。快照为 `experiments/evaluation-main-progress-20260821T165800.json`。

## 2026-08-21 17:23 CST｜论文 α=0.9 完整闭合并推进 mugs Task-FT

- α=0.9 剩余 14 个 generalist jobs 已全部提交，episode error 均为 0。LIBERO-Goal 后四项为 `[8,10,1,7]/10`，连同已记录首项 `0/10` 得到完整 `[0,8,10,1,7]/10`、`26/50=52%`；LIBERO-Object 为 `[0,0,0,0,6]/10`、`6/50=12%`；LIBERO-Spatial 为 `[5,1,1,3,5]/10`、`15/50=30%`。
- 加上 LIBERO-90 `[10,7,0,4,3]/10`、`24/50=48%`，论文 α=0.9 的完整 20-task retention 为 `71/200=35.5%`。它比 raw Task-FT 的 `63/200=31.5%` 高 4pp，只回收 `8/70≈11.4%` 的遗忘缺口；比 raw pretraining 的 `133/200=66.5%` 低 31pp，也比预登记 α=0.8 的 `76/200=38%` 低 2.5pp。相对 α=0.8 的 suite 差为 LIBERO-90/Goal/Object/Spatial `-6/+10/-2/-12pp`，改善并不均匀，且总体由 Goal 的优势部分抵消其余套件劣势。
- α=0.9 的 31/31 个独立 main jobs 共 320 个 episodes、133 个成功、0 个 error，320 个视频 `31,980,614` bytes；ID/OOD_HARD/generalist 分别为 `70%/48%/35.5%`，对应 α=0.8 为 `75%/50%/38%`。因此当前 stove policy 的三个独立主指标均不支持 paper α=0.9 优于预登记选出的 α=0.8；不过相对 raw Task-FT 的 generalist `+4pp` 仍属于有限的 partial retention recovery。OOD_MEDIUM `28/50` 继续仅作为复用 selection set，不列为独立主证据。
- pipeline 于 17:17:07 CST 自动加载 `taskft_mugs` step-999 参数，17:17:31 启动 ID。ID 于 17:21:57 正式提交为 `13/20=65%`、error 0、20 个视频 `1,719,407` bytes；raw pretraining 匹配 ID 为 `0/20`，所以 Task-FT 点估计提高 65pp。首个 OOD_MEDIUM seed-1（`a5039775ab9bce2c`）随后启动；17:23:31 只观察到 5/10 episodes、2 个成功、0 error、视频 `597,038` bytes且没有 summary，因此不纳入正式 OOD 汇总。
- 截止稳定提交边界，main evaluation 累计 167 个完整 jobs、1,740 个 episodes、567 个成功、0 个 error，1,740 个视频 `194,171,254` bytes；status 连同 270 个 alpha jobs 共 437 completed keys，`failed_job=null`。17:23:17 CST service active，pipeline/policy/evaluator PID 为 `1886 / 192302 / 199575`；唯一复现 compute PID `192302` 使用物理 GPU 0 的 `77,036 MiB`，fatal scan 为 0。主机可用 `531,329,240,064` bytes、共享盘可用 `101,824,466,944` bytes、root 可用 `70,865,891,328` bytes；全部已知写路径仍在 `/shared`，root 只监测、不触发停机。机器可读快照为 `experiments/evaluation-main-progress-20260821T172331.json`。

## 2026-08-21 17:38 CST｜mugs Task-FT 的 OOD_MEDIUM 完整闭合

- `taskft_mugs` 五个 OOD_MEDIUM jobs 已连续提交，seed `[1,21,41,61,81]` 的成功数为 `[5,4,5,7,5]/10`，合计 `26/50=52%`；episode error 全部为 0，视频字节依次为 `1,032,419 / 1,144,225 / 1,036,285 / 1,007,980 / 1,134,616`。
- raw pretraining 在相同 mugs OOD_MEDIUM 协议下为 `0/50`，所以 raw Task-FT 提高 52pp；连同 ID `13/20=65% vs 0/20`，两个完整目标协议均显示明显适应增益。后续复核冻结的 alpha-selection 状态确认：选中的是 `retain_taskft_mugs_a070`（`21/50=42%`），论文参考是 `retain_taskft_mugs_a080`（`20/50=40%`）；它们会复用各自的五个 selection jobs，但不能当作独立主证据。此前把 raw Task-FT 的 `26/50` 误写成 α=0.8 selection 结果的标签错误已更正，原始 summary 和任何实验结果均未改变。
- 首个 OOD_HARD set-0 seed-1（`c2e0a9e699ebcbf5`）于 17:37:41 CST 提交为 `4/10`、error 0、10 个视频 `1,106,240` bytes；匹配 pretrain 条件为 `0/10`。这里只完成 1/10 hard 条件，不报告完整 OOD_HARD 成功率。第二个 set-1 seed-1（`b5a5ad7c220668f6`）随后启动；17:38:24 仅观察 1/10 episode、0 个成功、0 error、视频 `128,427` bytes，无 summary，不计入正式比较。
- 截止稳定边界，main evaluation 累计 173 个完整 jobs、1,800 个 episodes、597 个成功、0 个 error，1,800 个视频 `200,633,019` bytes；status 连同 alpha sweep 共 443 completed keys、`failed_job=null`。17:38:24 CST service active，pipeline/policy/evaluator PID 为 `1886 / 192302 / 219888`；唯一复现 compute PID `192302` 使用物理 GPU 0 的 `77,036 MiB`，fatal scan 为 0。主机可用 `531,349,709,824` bytes、共享盘可用 `101,804,285,952` bytes、root 可用 `70,838,423,552` bytes。GPU 4--7 的外部负载不属于本复现；全部已知写路径仍在 `/shared`，root 只监测、不触发停机。快照为 `experiments/evaluation-main-progress-20260821T173824.json`。

## 2026-08-21 17:58 CST｜mugs Task-FT 完成前七个 OOD_HARD 条件

- 自上一稳定边界新增六个完整 jobs，均为 `taskft_mugs` OOD_HARD：set-1 seed-1 为 `0/10`、set-0/set-1 seed-21 为 `2/10、0/10`、set-0/set-1 seed-41 为 `3/10、0/10`、set-0 seed-61 为 `4/10`；episode error 全部为 0，视频字节依次为 `1,328,339 / 1,260,307 / 1,314,158 / 1,213,373 / 1,220,923 / 1,180,693`。
- 连同已记录的 set-0 seed-1 `4/10`，前七个条件按协议顺序为 `[4,0,2,0,3,0,4]/10`，partial 合计 `13/70=18.6%`；匹配 raw pretraining 为 `0/70`。四个已闭合 set-0 条件为 `13/40=32.5%`，三个已闭合 set-1 条件为 `0/30`，目前所有成功均来自 set-0。该差异只作为未闭合协议的条件族观察，不提前报告完整 OOD_HARD 成功率，也不作机制解释。
- 第八个条件 set-1 seed-61（`d28aa2424c728a7c`）于 17:56:29 CST 启动；17:58:30 已写出 6/10 episodes、0 success、0 error、视频 `769,764` bytes，但没有 summary，因此不计入正式 aggregate。pipeline 后续仍将串行完成剩余 hard 条件和 20 个 generalist jobs。
- 截止稳定提交边界，main evaluation 累计 179 个完整 jobs、1,860 个 episodes、606 个成功、0 个 error，1,860 个视频 `208,150,812` bytes；status 连同 270 个 alpha jobs 共 449 completed keys，失败字段不存在且 fatal scan 为 0。service active，pipeline/policy/evaluator PID 为 `1886 / 192302 / 240155`；唯一复现 compute PID `192302` 使用物理 GPU 0 的 `77,036 MiB`，单卡约束满足。
- 17:58:30 CST 资源为主机可用 `531,194,606,592` bytes、共享盘可用 `101,787,860,992` bytes、root 可用 `70,782,787,584` bytes。其余 GPU 的负载属于外部任务；全部已知复现写路径仍在 `/shared`，root 继续只监测、不触发停机。机器可读快照为 `experiments/evaluation-main-progress-20260821T175830.json`。

## 2026-08-21 18:17 CST｜mugs Task-FT 的 OOD_HARD 与 LIBERO-90 完整闭合

- OOD_HARD 最后三个条件已正式提交：set-1 seed-61 为 `0/10`、set-0/set-1 seed-81 为 `2/10、0/10`；episode error 均为 0，视频字节分别为 `1,322,852 / 1,282,625 / 1,283,529`。十条件完整序列为 `[4,0,2,0,3,0,4,0,2,0]/10`，合计 `15/100=15%`，比 raw pretraining 的 `0/100` 高 15pp。
- 按预定义条件族，set-0 五项合计 `15/50=30%`，set-1 五项为 `0/50`，所有成功均来自 set-0。该完整协议确认结果对两个 hard 条件族高度不对称；它是具体 perturbation 下的经验结果，不额外声称机制或统计显著性。
- 随后五个 LIBERO-90 generalist jobs 连续提交为 `[10,0,0,0,9]/10`、`19/50=38%`，episode error 0，视频字节为 `448,154 / 1,189,279 / 1,425,958 / 1,162,140 / 574,865`。匹配 raw pretraining 为 `[10,7,9,8,7]/10`、`41/50=82%`，Task-FT 下降 44pp；三个中间任务从 `24/30` 降为 `0/30`，而首末任务仍高，说明 forgetting 明显且强烈依赖任务。
- LIBERO-Goal 首任务 `open the middle drawer of the cabinet` 于 18:17:08 CST 提交为 `0/10`、error 0、视频 `827,906` bytes。第二项 `put the bowl on the plate`（`a7f443f1949fa4fa`）在 18:17:50 为 3/10 episodes、0 success、0 error、视频 `280,761` bytes，无 summary，故不计入正式 retention。
- 截止稳定边界，main evaluation 累计 188 个完整 jobs、1,950 个 episodes、627 个成功、0 个 error，1,950 个视频 `217,668,120` bytes；status 连同 270 个 alpha jobs 共 458 completed keys，`failed_job=null`，fatal scan 为 0。service active，pipeline/policy/evaluator PID 为 `1886 / 192302 / 270611`；唯一复现 compute PID `192302` 使用物理 GPU 0 的 `77,036 MiB`，单卡约束满足。
- 18:17:50 CST 主机可用 `531,862,441,984` bytes、共享盘可用 `101,434,626,048` bytes、root 可用 `70,561,193,984` bytes。全部已知复现写路径仍在 `/shared`，root 继续只监测、不停机。机器可读快照为 `experiments/evaluation-main-progress-20260821T181750.json`。

## 2026-08-21 18:36 CST｜mugs Task-FT 完成 Goal、Object 与前三个 Spatial 任务

- LIBERO-Goal 后四项依次为 `[2,4,0,1]/10`；连同已记录首项 `0/10`，完整 suite 为 `[0,2,4,0,1]/10`、`7/50=14%`，episode error 0。对应视频字节为 `827,906 / 826,532 / 706,869 / 856,203 / 847,524`。匹配 raw pretraining 为 `31/50=62%`，因此下降 48pp。
- LIBERO-Object 五任务完整为 `[0,4,4,4,0]/10`、`12/50=24%`，episode error 0，视频字节为 `665,257 / 611,888 / 595,446 / 581,950 / 739,681`。raw pretraining 为 `33/50=66%`，下降 42pp。结合 LIBERO-90 的 `38% vs 82%`，已闭合三个 suite 的遗忘幅度分别为 `-44/-48/-42pp`。
- LIBERO-Spatial 前三项为 `[5,0,0]/10`、partial `5/30=16.7%`，视频字节 `474,949 / 611,198 / 580,216`；pretraining 匹配前三项为 `[4,7,4]/10`、`15/30=50%`。当前只作 complete-task partial aggregate，不提前报告完整 Spatial 或 20-task retention。
- 截至 18/20 个 generalist tasks，`taskft_mugs` 为 `43/180=23.9%`，匹配 pretraining 为 `120/180=66.7%`，partial 差 -42.8pp。倒数第二个 Spatial 任务 `pick up the black bowl next to the cookie box and place it on the plate`（`2ddb67a6bacb6e38`）于 18:36:22 CST 启动；18:36:54 为 5/10 episodes、2 success、0 error、视频 `322,113` bytes，无 summary，不计入正式 aggregate。
- 截止稳定提交边界，main evaluation 累计 200 个完整 jobs、2,070 个 episodes、651 个成功、0 个 error，2,070 个视频 `225,765,833` bytes；status 连同 270 个 alpha jobs 共 470 completed keys，`failed_job=null`，fatal scan 为 0。service active，pipeline/policy/evaluator PID 为 `1886 / 192302 / 312780`；唯一复现 compute PID `192302` 使用物理 GPU 0 的 `77,036 MiB`。
- 18:36:54 CST 主机可用 `531,627,881,472` bytes、共享盘可用 `101,414,060,032` bytes、root 可用 `70,550,589,440` bytes。全部已知写路径继续位于 `/shared`，root 只监测、不停机。机器可读快照为 `experiments/evaluation-main-progress-20260821T183654.json`。

## 2026-08-21 18:44 CST｜mugs Task-FT 完整闭合并纠正 alpha 标签

- 最后两个 LIBERO-Spatial jobs 于 18:37:55、18:39:20 CST 提交为 `2/10、5/10`，episode error 0、视频 `651,953 / 469,779` bytes。Spatial 完整为 `[5,0,0,2,5]/10`、`12/50=24%`，比 raw pretraining 的 `28/50=56%` 低 32pp。
- `taskft_mugs` 的完整 20-task retention 为 `[10,0,0,0,9,0,2,4,0,1,0,4,4,4,0,5,0,0,2,5]/10`，合计 `50/200=25%`；raw pretraining 为 `133/200=66.5%`，下降 41.5pp、少 83 个成功 episode。四个 suites 为 LIBERO-90/Goal/Object/Spatial `38%/14%/24%/24%`，对应 pretraining `82%/62%/66%/56%`，全部下降 32--48pp。
- 结合目标任务 ID/OOD_MEDIUM/OOD_HARD `65%/52%/15%` 对 pretraining `0%/0%/0%`，mugs 的完整 raw Task-FT 结果确认了 adaptation–retention trade-off。该 policy 36/36 jobs 共 370 episodes、104 success、0 error，370 个视频 `34,435,718` bytes。
- pipeline 在 18:39:27 CST 自动加载 `retain_taskft_mugs_a070`。直接复核 canonical alpha-sweep state 确认：α=0.7 是唯一最佳，五个 OOD_MEDIUM validation jobs 为 `21/50=42%`；论文参考 α=0.8 为 `20/50=40%`。此前 17:38 附近的个别记录把 raw Task-FT 的 `26/50` 误写成 α=0.8 selection 结果，现已更正 findings、日志语句和四个受影响快照；这是纯文档标签纠正，原始 summaries、completed keys、参数选择和正在运行的实验均未改变。审计文件为 `experiments/incidents/mugs-alpha-selection-label-correction-20260821T1844.json`。
- 五个 α=0.7 selection jobs 已按设计幂等复用且继续标记为非独立证据。18:40:03 CST 启动第 1/31 个独立 ID job（`f2939e45cd15edb2`）；18:44:02 已观察 16/20 episodes、11 success、0 error、视频 `1,596,827` bytes，无 summary，故不计入正式 RETAIN 指标。
- 稳定提交边界为 main 202 jobs、2,090 episodes、658 success、0 error、2,090 个视频 `226,887,565` bytes；status 连同 alpha sweep 共 472 completed keys，`failed_job=null`、fatal scan 0。service active，pipeline/merged-policy/evaluator PID 为 `1886 / 319859 / 320641`；合并权重 `[0.7,0.3]`，唯一复现 compute PID `319859` 使用物理 GPU 0 的 `77,036 MiB`。
- 18:44:02 CST 主机可用 `527,444,305,920` bytes、共享盘可用 `101,409,562,624` bytes、root 可用 `70,547,742,720` bytes。写路径继续全部位于 `/shared`，root 只监测、不停机。机器可读快照为 `experiments/evaluation-main-progress-20260821T184402.json`。

## 2026-08-21 19:21 CST｜mugs 所选 α=0.7 完成 ID/OOD_HARD 并进入 retention

- `retain_taskft_mugs_a070` 的独立 ID 于 18:44:52 CST 正式提交为 `12/20=60%`、episode error 0，20 个视频 `2,062,940` bytes。它比 raw Task-FT 的 `13/20=65%` 低 5pp，但比 raw pretraining 的 `0/20` 高 60pp。
- 十个独立 OOD_HARD 条件随后全部提交，固定顺序 `set-0/set-1`、seeds `1/21/41/61/81` 的成功数为 `[1,0,3,0,4,0,2,0,4,0]/10`，合计 `14/100=14%`、episode error 0；set-0 为 `14/50=28%`，set-1 为 `0/50`。匹配 raw Task-FT 为 `15/100=15%`，pretrain 为 `0/100`，故合并相对两者分别为 `-1pp/+14pp`。完整 ID 与 OOD_HARD 都没有显示 α=0.7 优于 Task-FT，但目标适应仍远高于 pretrain。
- OOD_MEDIUM 五个 selection jobs `[3,4,5,4,5]/10`、`21/50=42%` 已从 alpha sweep 幂等复用，继续明确标为 selection-set estimate、不是独立主证据。其点估计比 raw Task-FT 的独立 `26/50=52%` 低 10pp。论文参考 `retain_taskft_mugs_a080` 的 selection-set estimate 为 `20/50=40%`，将在当前 policy 完成后继续按流水线评测。
- generalist retention 已提交前三项：`close the top drawer of the cabinet / pick up the tomato sauce and put it in the basket / put the frying pan on the stove` 分别为 `[10,4,0]/10`，合计 `14/30=46.7%`、episode error 0，视频字节为 `343,636 / 988,342 / 1,508,110`。同三任务 raw Task-FT 为 `[10,0,0]/10`、`10/30=33.3%`，pretrain 为 `[10,7,9]/10`、`26/30=86.7%`；当前 partial 结果相对 Task-FT 恢复 13.3pp，但仍比 pretrain 低 40pp，尚不能外推到完整 LIBERO-90 或 20-task retention。
- 19:21:18 CST 已启动第 15/31 个独立 job、LIBERO-90 第四项 `put the white bowl on the plate`（`aa2469c3edd187c3`）；19:21:58 观察到 3/10 episodes、3 success、0 error、视频 `254,906` bytes，无 summary，因此未并入正式 aggregate。
- 本轮新增 14 个完整 main jobs、150 episodes、40 success、0 error，150 个视频 `17,161,535` bytes。稳定边界的 main evaluation 累计为 216 jobs、2,240 episodes、698 success、0 error、2,240 个视频 `244,049,100` bytes；status 连同 270 个 alpha jobs 共 486 completed keys，`failed_job=null`、fatal scan 为 0。
- service active，pipeline/merged-policy/evaluator PID 为 `1886 / 319859 / 374661`；合并权重 `[0.7,0.3]`，唯一复现 compute PID `319859` 仍只占用物理 GPU 0、显存 `77,036 MiB`。主机可用 `528,014,776,320` bytes、共享盘可用 `101,384,105,984` bytes、root 可用 `70,525,620,224` bytes。全部已知写路径在 `/shared`，root 继续只监测、不作为停机条件。机器可读快照为 `experiments/evaluation-main-progress-20260821T192158.json`。
- 记录同步后于 19:27:54 CST 再次做只读 liveness 检查：service 仍 active，status 已由正式快照的 486 增至 490 个 completed keys，`failed_job=null`、fatal scan 0；当前已启动第 19/31 个独立 job（`70fc145658447db9`），GPU 0 上仍只有同一 merged-policy PID `319859`、显存 `77,036 MiB`。快照后新增的四个完整 generalist jobs留待下一轮从 summary/episodes/video 逐项审计后并入正式 aggregate，本条不提前引用其成功率。

## 2026-08-21 19:30 CST｜mugs α=0.7 的 LIBERO-90 完整闭合

- 上轮快照后的两个 LIBERO-90 jobs 已正式提交：`put the white bowl on the plate` 为 `6/10`、视频 `935,493` bytes；`put the white mug on the plate` 为 `10/10`、视频 `532,415` bytes，episode error 均为 0。连同前三项，完整 suite 为 `[10,4,0,6,10]/10`、`30/50=60%`。
- 完全配对的 raw Task-FT 为 `[10,0,0,0,9]/10`、`19/50=38%`，raw pretraining 为 `[10,7,9,8,7]/10`、`41/50=82%`。因此 α=0.7 相对 Task-FT 恢复 22pp，但相对 pretrain 仍低 22pp；原 44pp 遗忘缺口被回收一半。这是 mugs 上首个完整 suite 的明确 retention-recovery 证据，同时结合独立 ID/OOD_HARD 相对 Task-FT 的 `-5/-1pp`，呈现轻微目标性能损失与显著但不完全 retention 恢复的 trade-off。
- LIBERO-Goal 前三项已提交为 `[0,7,7]/10`：`open the middle drawer of the cabinet / put the bowl on the plate / put the bowl on top of the cabinet` 的视频字节分别为 `891,170 / 570,216 / 534,021`，episode error 均为 0。partial 合计 `14/30=46.7%`；raw Task-FT 匹配三项为 `6/30=20%`，pretrain 为 `15/30=50%`，故当前分别为 `+26.7pp/-3.3pp`。尚余两项，因此不提前报告完整 Goal 恢复率。
- 19:28:31 CST 已启动第 20/31 个独立 job、Goal 第四项 `put the cream cheese in the bowl`（`b831ad66cdf6a3dc`）；19:30:00 观察到 7/10 episodes、1 success、0 error、视频 `607,306` bytes，无 summary，未并入正式 aggregate。
- 本轮新增五个完整 main jobs、50 episodes、30 success、0 error，50 个视频 `3,463,315` bytes。正式累计为 221 main jobs、2,290 episodes、728 success、0 error、2,290 个视频 `247,512,415` bytes；status 连同 270 个 alpha jobs 共 491 completed keys，`failed_job=null`、fatal scan 0。
- service active，pipeline/merged-policy/evaluator PID 为 `1886 / 319859 / 391699`；唯一复现 compute PID `319859` 使用物理 GPU 0、显存 `77,036 MiB`。主机可用 `527,975,486,464` bytes、共享盘可用 `101,377,654,784` bytes、root 可用 `70,516,670,464` bytes；全部已知写路径继续位于 `/shared`，root 只监测、不触发停机。机器可读快照为 `experiments/evaluation-main-progress-20260821T193000.json`。
- 19:34:27 CST 同步后 liveness 复核：status 已增至 494 个 completed keys，当前启动第 23/31 个独立 job（`4cb2d2550de6051a`）；service active、`failed_job=null`、fatal scan 0，GPU 0 上仍只有 policy PID `319859`、显存 `77,036 MiB`。正式快照后又提交三个 generalist jobs，其指标留待下一轮逐项审计后并入；本条只记录流水线持续前进，不提前读取点估计。

## 2026-08-21 19:53 CST｜mugs 所选 α=0.7 完整闭合并推进论文 α=0.8

- α=0.7 的 Goal 后两项已正式提交：`put the cream cheese in the bowl / put the wine bottle on top of the cabinet` 分别为 `[1,4]/10`，视频 `864,497 / 641,259` bytes、episode error 均为 0。连同前三项，LIBERO-Goal 完整为 `[0,7,7,1,4]/10`、`19/50=38%`；raw Task-FT 为 `7/50=14%`，pretrain 为 `31/50=62%`，因此分别为 `+24pp/-24pp`，回收了原遗忘缺口的一半。
- LIBERO-Object 五任务为 `[0,8,8,6,1]/10`、`23/50=46%`，视频字节依次为 `694,830 / 502,256 / 543,768 / 515,439 / 734,971`，episode error 0。raw Task-FT/pretrain 分别为 `12/50=24%`、`33/50=66%`，故 α=0.7 分别为 `+22pp/-20pp`，回收 `11/21≈52.4%` 的 raw Task-FT 遗忘缺口。
- LIBERO-Spatial 五任务为 `[9,2,2,8,6]/10`、`27/50=54%`，视频字节依次为 `339,100 / 587,047 / 603,548 / 582,198 / 465,253`，episode error 0。raw Task-FT/pretrain 分别为 `12/50=24%`、`28/50=56%`，所以 α=0.7 为 `+30pp/-2pp`，恢复 `15/16=93.75%` 的遗忘缺口，是四套中最接近 pretraining 的结果。
- 合并 LIBERO-90 `[10,4,0,6,10]`、Goal `[0,7,7,1,4]`、Object `[0,8,8,6,1]` 与 Spatial `[9,2,2,8,6]`，α=0.7 的完整 20-task retention 为 `99/200=49.5%`。它比 raw Task-FT 的 `50/200=25%` 高 `24.5pp`，但仍比 pretrain 的 `133/200=66.5%` 低 `17pp`；共回收 `49/83≈59.0%` 的遗忘缺口。结合独立 ID/OOD_HARD 相对 raw Task-FT 的 `-5/-1pp`，完整证据支持“轻微目标性能损失换取显著但不完全的 retention 恢复”，不支持无代价恢复。
- `retain_taskft_mugs_a070` 的 31/31 个独立 jobs 共 320 episodes、125 success、0 error，320 个视频 `27,699,016` bytes；五个 OOD_MEDIUM selection jobs 仍单独标记为复用验证集，不并入独立主证据。正式快照前本轮新增的 12 个 α=0.7 jobs 合计 120 episodes、55 success、0 error、120 个视频 `7,074,166` bytes。
- pipeline 于 19:46:01 CST 自动加载论文参考 `retain_taskft_mugs_a080`，合并权重 `[0.8,0.2]`。独立 ID 于 19:51:32 提交为 `11/20=55%`、0 error、20 个视频 `1,921,024` bytes；它比所选 α=0.7 的 `60%` 低 5pp、比 raw Task-FT 的 `65%` 低 10pp，但比 pretrain 的 0% 高 55pp。其 OOD_MEDIUM `20/50=40%` 来自同一 alpha-selection set，继续明确标为非独立证据。
- 19:51:32 CST 已启动 α=0.8 的首个 OOD_HARD 条件 set-0 seed-1（`68cb8de284a051b9`）；19:53:06 观察到 5/10 episodes、3 success、0 error、视频 `601,654` bytes且无 summary，因此未并入正式 hard aggregate。
- 本轮正式新增共 13 个 main jobs、140 episodes、66 success、0 error，140 个视频 `8,995,190` bytes。稳定边界累计为 234 main jobs、2,430 episodes、794 success、0 error、2,430 个视频 `256,507,605` bytes；status 连同 270 个 alpha jobs 共 504 completed keys，`failed_job=null`、fatal scan 为 0。
- service active，pipeline/merged-policy/evaluator PID 为 `1886 / 433317 / 440739`；唯一复现 compute PID `433317` 使用物理 GPU 0、显存 `77,036 MiB`。19:52:31 CST 主机可用 `527,031,486,464` bytes、共享盘可用 `101,362,372,608` bytes、root 可用 `70,579,871,744` bytes。所有已知写路径继续位于 `/shared`；root 只监测、不触发停机。机器可读快照为 `experiments/evaluation-main-progress-20260821T195306.json`。
- 20:00:01 CST 同步后 liveness 复核：服务器端四个文件的 SHA-256 与本地完全一致；service 仍 active，status 已由正式快照的 504 增至 506 个 completed keys，`failed_job=null`、fatal scan 0。α=0.8 已连续提交前两个 OOD_HARD jobs并启动第 4/31 个独立 job（`07cab0acea5b9978`）；GPU 0 上仍只有同一 policy PID `433317`、显存 `77,036 MiB`。快照后两个完整 hard jobs 的指标留待下一轮逐项审计后并入正式 aggregate，本条不提前引用其成功率。

## 2026-08-21 20:09 CST｜mugs 论文 α=0.8 完成前五个 OOD_HARD 条件

- 自上一正式边界新增五个完整 jobs，依固定协议顺序为 set-0 seed-1 `4/10`、set-1 seed-1 `0/10`、set-0 seed-21 `3/10`、set-1 seed-21 `0/10`、set-0 seed-41 `3/10`；episode error 全部为 0，视频字节依次为 `1,248,472 / 1,263,502 / 1,129,273 / 1,179,383 / 1,269,106`。
- 五个完整条件合计 `10/50=20%`。完全匹配的预登记 α=0.7 为 `[1,0,3,0,4]/10`、`8/50=16%`，raw Task-FT 为 `[4,0,2,0,3]/10`、`9/50=18%`，pretrain 为 `0/50`；因此 α=0.8 当前分别高 `4pp/2pp/20pp`。这只是 5/10 条件的 complete-condition partial paired result，剩余条件可能改变排序，不报告完整 OOD_HARD 结论。
- 三个 set-0 条件合计 `10/30=33.3%`，两个 set-1 条件均为 0；当前所有成功来自 set-0。该模式与 α=0.7、raw Task-FT 的早期条件族差异方向一致，但条件未闭合且 perturbation 定义不同，只作描述，不作机制或显著性解释。
- 第六个条件 set-1 seed-41（`895c5e9faaa3089a`）于 20:07:09 CST 启动；20:09:35 观察到 7/10 episodes、0 success、0 error、视频 `780,896` bytes，无 summary，因此未并入正式 aggregate。
- 本轮新增五个 main jobs、50 episodes、10 success、0 error，50 个视频 `6,089,736` bytes。稳定提交边界累计为 239 main jobs、2,480 episodes、804 success、0 error、2,480 个视频 `262,597,341` bytes；status 连同 270 个 alpha jobs 共 509 completed keys，`failed_job=null`、fatal scan 为 0。
- service active，pipeline/merged-policy/evaluator PID 为 `1886 / 433317 / 457776`；唯一复现 compute PID `433317` 使用物理 GPU 0、显存 `77,036 MiB`。主机可用 `527,110,568,960` bytes、共享盘可用 `101,351,186,432` bytes、root 可用 `70,564,769,792` bytes。所有已知写路径继续位于 `/shared`，root 只监测、不触发停机。机器可读快照为 `experiments/evaluation-main-progress-20260821T200935.json`。
- 20:13:22 CST 同步后 liveness 复核：四个同步文件的服务器 SHA-256 与本地完全一致；service active、status 已由正式快照的 509 增至 510 个 completed keys，`failed_job=null`、fatal scan 0。第六个 hard job 已提交并启动第 8/31 个独立 job（`38a0a0c73d90bd29`）；GPU 0 上仍只有 policy PID `433317`、显存 `77,036 MiB`。新提交 job 的指标留待下一轮从 summary/episodes/video 审计后并入正式 aggregate。

## 2026-08-21 20:27 CST｜按用户要求暂停并释放 GPU 0

- 暂停前流水线继续完成 α=0.8 的 OOD_HARD 后五个条件：set-1 seed-41、set-0/set-1 seed-61、set-0/set-1 seed-81 为 `[0,0,0,3,0]/10`，episode error 全部为 0，视频字节依次为 `1,121,467 / 1,431,239 / 1,243,768 / 1,252,501 / 1,219,807`。与前五项合并后完整序列为 `[4,0,3,0,3,0,0,0,3,0]/10`、`13/100=13%`；set-0 为 `13/50=26%`、set-1 为 `0/50`。
- 完全配对比较显示，论文 α=0.8 的 OOD_HARD 比预登记 α=0.7 的 `14%` 低 1pp、比 raw Task-FT 的 `15%` 低 2pp、比 pretrain 的 0% 高 13pp。结合独立 ID `55% vs α=0.7 60% / Task-FT 65%`，mugs 的两个独立目标协议均不支持论文 α=0.8 优于预登记选点或 raw Task-FT；该结论不使用被复用的 OOD_MEDIUM selection set。
- 随后两个完整 LIBERO-90 generalist tasks 为 `close the top drawer of the cabinet / pick up the tomato sauce and put it in the basket = [10,0]/10`，视频 `350,469 / 1,111,671` bytes、error 0。α=0.7 匹配为 `[10,4]`、raw Task-FT `[10,0]`、pretrain `[10,7]`；只完成 2/20，故不报告总体 retention。
- 暂停前最后稳定边界为 516 个 completed keys（270 alpha + 246 main）、2,550 个正式 main episodes、817 success、0 error，2,550 个视频 `270,328,263` bytes。当前 α=0.8 已完成 13/31 个独立 jobs，共 140 episodes、34 success、0 error，140 个视频 `15,741,682` bytes。共享 `status.json` 为 `518,979` bytes，SHA-256 `1b1b238c1b971441f0462a80d3e94eb808c8b57c7de5f475df6fb0ac0ef84dfd`，`failed_job=null`。
- 用户说明他人需要使用物理 GPU 0，并要求暂时停止测试和定时任务。20:26:44 CST systemd 开始停止服务，20:26:45 记录 `Deactivated successfully`。停机后 `retain-reproduction.service=inactive`，transient unit 已被卸载，evaluation pipeline、policy server、evaluator 均不存在；GPU 0 为 `0 MiB / 0%`，没有任何复现 compute process，显卡已完全释放。
- 被中断的是 α=0.8 第 14/31 个独立 generalist job `487531f293f34baa`（`put the frying pan on the stove`）。停机前只有 1 个失败 episode、0 error、1 个视频 `147,913` bytes，没有 summary，且 key 不在 completed set；下次恢复时允许从同一 key 幂等重跑，不能把这份 partial 当作正式结果。
- 恢复入口已冻结：在 `/root/RETAIN_code` 重建同名 transient service，运行 `/root/RETAIN_code/.venv/bin/python reproduction/scripts/run_evaluation_pipeline.py --poll-seconds 60`；环境继续使用 `RETAIN_EVAL_STATE_ROOT=/shared/.cache/retain/evaluation-state/RETAIN-GPU-20260819-001/evaluation-pipeline`、`TMPDIR=/shared/.cache/retain/tmp/evaluation`、`XDG_CACHE_HOME=/shared/.cache/retain/cache/evaluation`，stdout/stderr append 到 `/shared/.cache/retain/evaluation-state/RETAIN-GPU-20260819-001/evaluation-service.log`。预期先读取并跳过 516 个完成项，再重跑 `487531f293f34baa`；恢复后须先验证 completed count、`failed_job` 和 GPU 0 单卡约束。
- Codex 定时任务 `retain` 已从 `ACTIVE` 改为 `PAUSED`，不再每 20 分钟触发。暂停快照为 `experiments/evaluation-main-paused-user-gpu-release-20260821T202645.json`。20:30:11 CST 只读资源复核为主机可用 `539,736,687,616` bytes、共享盘可用 `101,338,562,560` bytes、root 可用 `70,551,056,384` bytes；本次停机原因仅为用户调度 GPU，不是 root 空间、实验失败或存储异常。
