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
