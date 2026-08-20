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
19. 为保持项目主环境可回退，JAX 兼容修复采用 `/shared/.cache/retain/jax-overlays/0.6.2` 隔离 overlay，而没有原位升级 `.venv`。overlay 固定 JAX/JAXlib/CUDA plugin 0.6.2、ml-dtypes 0.5.1 与 cuDNN 9.8.0.87；在 CPU backend 下，7/7 真实 batch、同一 `π0 base` Orbax restore（参数数目与结构哈希不变）及完整 pretrain train-state `eval_shape` 均已通过。后续 attempt 3 已在真实 sm_120 GPU 上验证该 overlay 的 BF16 编译有效。
20. 第二次获得空闲 GPU 2 时，初版 preflight 因 CUDA libraries 不可见而回退到 CPU，但旧门禁只检查退出码，错误标为通过；完整训练随后加载主环境 cuDNN 9.7.1，与 JAX plugin build 的 9.8.0 不兼容并在 step 0 前退出。修复后 JAX plugin 所需的 CUDA wheels 已全部固定到 overlay，训练和评测均显式优先 overlay 动态库；12/12 关键库加载与 CPU BF16 smoke 已通过，门禁也已强化为必须只有一个可见设备且 `platform=gpu`。
21. 第三次获得空闲 GPU 2 后，强化门禁首次在真实 `NVIDIA RTX 6000D` 上验证 JAX/JAXlib/CUDA plugin 0.6.2、唯一 GPU backend、BF16→FP16 与 BF16 GEMM 均成功，证明完整 CUDA 隔离和 sm_120 编译修复有效。完整 batch-64 train step 的 XLA 图从 77.10 GiB rematerialize 到 75.44 GiB，但仍无法达到约 25.07 GiB 的内存目标，执行时申请 27.79 GiB buffer 失败。这里的 75.44 GiB 不能解释成“总峰值只比 90% pool 多 0.16 GiB”；后续排除实验已修正早期判断。
22. attempt 4 在 GPU 1 的严格 preflight 再次通过后，把 BFC pool 从 90% 提到 95%，仍在同一 27.79 GiB 请求处 OOM；attempt 5 请求 JAX 官方建议的 `cuda_malloc_async`，但训练日志仍明确显示 allocator 为 `GPU_0_bfc`，并在同一位置失败。三次 batch-64 完整首步、两种 pool 设置和一次 async allocator 请求共同证明：当前 JAX 0.6.2 runtime 下，论文 batch 64 无法在单张 85,651 MiB RTX 6000D 上执行；这不是数据、基础权重、BF16 runtime 或并发抢卡问题。
23. 在用户指定的单 GPU 约束与“失败后可缩小实验”范围内，七个 `retain_repro_*` config 已统一改为物理 batch 16，不使用 gradient accumulation；模型、steps、LR schedule、AdamW、gradient clipping、EMA、mixture、seed 和数据保持不变。attempt 6 的真实 loader 已确认 batch 维为 16，XLA 图由 62.62 GiB rematerialize 到 62.52 GiB，并已连续推进到至少 step 181，GPU 1 稳定在 `81,318 MiB / 100%`，因此首个 optimizer step 与 OOM 门槛已实际通过。当前进程的 `Step` 数值行仍受 stdout 缓冲影响，不能在结束前据此宣称 loss/gradient 已完成有限值审计。
24. attempt 6 的 step-1000 checkpoint 已于 09:12:37 CST 完成 Orbax 原子提交；日志明确报告后台保存线程无错误并完成所有 host 的 finalize，提交后没有临时目录残留。checkpoint 包含 `assets`、`params`、`train_state` 三项，共 45 个文件、42,976,247,401 bytes 文件 payload；元数据含非空 commit timestamp。独立的只读 `CheckpointManager` 能发现 `all_steps=[1000]`、`latest_step=1000`，且训练在保存后继续推进到约 step 1090。因此首个可恢复点的目录结构与可发现性已经验证；这项验证不等同于完整数组 restore，也不替代尚未可见的 loss/gradient 有限值审计。
25. step-2000 checkpoint 于 09:51:33 CST 完成第二次原子提交，后台线程无错误；只读 `CheckpointManager` 随后返回 `all_steps=[2000]`、`latest_step=2000`。Orbax 在新 checkpoint 提交后才于 09:51:40 完成旧 step 1000 的删除，根目录最终只保留 `2000/`，从而实证 `max_to_keep=1` 的滚动恢复点与磁盘回收行为。第二次 blocking 阶段为 90.45 秒，训练随后恢复到约 2.2 s/step 并推进到约 step 2050；两份 checkpoint 并存时观测到共享盘余量 169,767,882,752 bytes，仍高于 150 GB 安全线，清理后回升至 212,744,130,560 bytes。
26. attempt 6 的 stdout 缓冲已在训练中自行刷新，现可审计 step 0--2225、每 25 step 一条的 90 条 metrics；四个字段全部为有限值。loss 从 `0.1572` 降至 `0.0381`（首末下降 75.76%），前/后 10 条均值为 `0.09298 / 0.04065`；500-step 分段均值依次为 `0.07106 / 0.04671 / 0.04514 / 0.04091 / 0.04065`。grad norm 的前/后 10 条均值为 `1.77271 / 0.47439`，param norm 由 `1377.8652` 缓慢变至 `1378.5364`，vision param norm 由 `1258.1694` 变至 `1258.254601`。这些结果支持训练数值稳定、早期 loss 明显下降并在约 0.04 附近进入平台，但只覆盖 22.25% 训练，不能据此宣称 10,000-step 完整收敛。
27. step-3000 checkpoint 于 10:32:45 CST 完成第三次 Orbax 原子提交，后台保存线程报告无错误；只读 CPU `CheckpointManager` 返回 `all_steps=[3000]`、`latest_step=3000`。本次 checkpoint manager blocking 阶段为 251.21 秒，文件 payload 为 45 个文件、42,936,168,167 bytes；两份 checkpoint 重叠写入期间观测到共享盘最低余量 173,295,751,168 bytes，仍高于 150 GB 安全线。新点提交后，旧 step 2000 于 10:32:47 完成删除，根目录只剩 `3000/` 且无临时目录；训练随后恢复至约 step 3060、错误计数仍为 0。这再次验证滚动恢复策略，但不替代最终 checkpoint 的完整数组 restore。
28. step-4000 checkpoint 于 11:13:37 CST 完成第四次 Orbax 原子提交，后台保存线程报告无错误；只读 CPU `CheckpointManager` 返回 `all_steps=[4000]`、`latest_step=4000`。本次 blocking 阶段为 211.33 秒，checkpoint 为 51 个文件、42,939,162,153 bytes 文件 payload。保存早期在临时点约 12.0 GB 时实测共享盘尚余 200,727,379,968 bytes；该快照只证明当时资源安全，不冒充整个重叠窗口的最低值。新点提交后旧 step 3000 于 11:13:39 完成删除，根目录只含 `4000/`；训练恢复到约 step 4050，错误计数仍为 0，继续支持滚动保存与单卡训练稳定性。
29. step-5000 checkpoint 于 11:54:35 CST 完成第五次 Orbax 原子提交，后台保存线程报告无错误；只读 CPU `CheckpointManager` 返回 `all_steps=[5000]`、`latest_step=5000`。本次 checkpoint manager blocking 阶段为 241.34 秒，checkpoint 为 49 个文件、42,948,570,663 bytes 文件 payload。20 秒监控采样中，两份恢复点重叠时观测到的最低共享盘余量为 169,653,215,232 bytes，仍高于 150 GB 安全线；该值是离散采样结果，不声称是连续窗口的理论最低值。新点提交后旧 step 4000 于 11:54:37 完成删除，根目录只含 `5000/`，训练随后推进到约 step 5060 且错误计数仍为 0。
30. step-5000 保存使 stdout 缓冲再次刷新，现已审计 step 0--4475、每 25 step 一条的 180 条 metrics，`loss`、`grad_norm`、`param_norm` 和 `vision_param_norm` 全部有限。loss 从 `0.1572` 降至 `0.0327`（首末下降 79.20%），最近 10 条均值为 `0.03444`；从 step 2000 起的 500-step loss 均值依次为 `0.03890 / 0.03734 / 0.03556 / 0.03472 / 0.03419`，说明此前约 0.04 的平台后仍有缓慢下降。grad norm 最近 10 条均值为 `0.37565`，param norm 与 vision param norm 仅缓慢增至 `1379.4877 / 1258.376303`。这些结果支持训练持续稳定且仍在改善，但仅覆盖 44.75%，不能据此宣称完整收敛。
31. step-6000 checkpoint 于 12:37:08 CST 完成第六次 Orbax 原子提交，后台保存线程报告无错误；只读 CPU `CheckpointManager` 返回 `all_steps=[6000]`、`latest_step=6000`。本次 checkpoint manager blocking 阶段为 320.29 秒，是目前最长的单次观测值，但不对共享存储 I/O 差异作未经测量的归因。checkpoint 为 43 个文件、42,947,359,114 bytes 文件 payload；20 秒监控采样中，两份恢复点重叠时观测到的最低共享盘余量为 169,865,744,384 bytes，仍高于 150 GB 安全线。新点提交后旧 step 5000 于 12:37:11 完成删除，根目录只含 `6000/`；训练随后推进到约 step 6050，错误计数仍为 0，metrics 缓冲未变化，仍为已审计的 180 条。
32. step-7000 checkpoint 于 13:18:05 CST 完成第七次 Orbax 原子提交，后台保存线程报告无错误；只读 CPU `CheckpointManager` 返回 `all_steps=[7000]`、`latest_step=7000`。本次 checkpoint manager blocking 阶段为 239.04 秒，checkpoint 为 46 个文件、42,953,023,549 bytes 文件 payload；新点提交后旧 step 6000 于 13:18:06 CST 完成删除，根目录只含 `7000/` 且无临时目录。心跳在提交完成后到达，因此本次没有重叠窗口的离散磁盘采样，不补造最低余量；清理后共享盘可用 205,881,110,528 bytes，训练随后推进到约 step 7200，错误计数仍为 0。
33. step-7000 保存使 stdout 缓冲刷新，现已审计 step 0--6725、每 25 step 一条的 270 条 metrics，四项指标全部有限。loss 从 `0.1572` 降至 `0.0324`，最近 10 条均值为 `0.03104`；step 4500 起的完整 500-step loss 均值为 `0.03316 / 0.03308 / 0.03261 / 0.03157`，step 6500--6725 的部分区间均值为 `0.03104`，支持此前的缓慢下降趋势仍在继续。grad norm 最近 10 条均值为 `0.34007`，param norm 与 vision param norm 仅缓慢增至 `1380.3579 / 1258.486305`。覆盖率为 67.25%，仍不能据此宣称 10,000-step 完整收敛。
34. step-8000 checkpoint 于 14:02:09 CST 完成第八次 Orbax 原子提交，后台保存线程报告无错误；只读 CPU `CheckpointManager` 返回 `all_steps=[8000]`、`latest_step=8000`。本次 checkpoint manager blocking 阶段为 377.81 秒，是目前最长的单次观测值，但不对共享存储 I/O 差异作未经测量的归因。checkpoint 为 45 个文件、42,973,136,012 bytes 文件 payload；新点提交后旧 step 7000 于 14:02:10 CST 完成删除，根目录只含 `8000/` 且无临时目录。心跳在提交完成后到达，因此没有重叠窗口的离散磁盘采样；清理后共享盘可用 205,847,683,072 bytes，训练随后推进到约 step 8060，错误计数仍为 0。stdout metrics 仍是已审计的 270 条，故本里程碑不新增曲线结论。
35. attempt 6 实际运行到 step 9000 保存入口后，systemd 明确记录所属 session scope 中有进程被主机 OOM killer 终止；训练以 return code 137 退出。`9000.orbax-checkpoint-tmp-32` 只有临时 assets/params、18 个文件和 12,014,142,618 bytes，没有完整 train state、finalize 或原子重命名，故不是有效 checkpoint；完整 step 8000 仍是唯一 Orbax 可发现恢复点。同一窗口没有主机重启、GPU Xid 或磁盘 I/O error。恢复协议因此只改变 operational checkpoint 存储：从 step 8000 恢复 optimizer state、跳过 step 9000 全量保存，并在最终 step 9999 只保留 EMA/inference params；实现已通过 compile、关键 Ruff 规则、配置断言与单元测试，实际终态保存效果仍待 attempt 7 验证。
36. 从 attempt 6 失败前的完整 stdout 重新导出 360 条 metrics，覆盖 step 0--8975 且固定间隔为 25 step；`loss`、`grad_norm`、`param_norm` 与 `vision_param_norm` 全部有限。loss 首末为 `0.1572 / 0.0286`，最近 10 条均值为 `0.02899`；最近四个完整 500-step loss 均值为 `0.03114 / 0.030585 / 0.03006 / 0.02964`，支持缓慢下降一直持续到故障前。最近 10 条 grad norm 均值为 `0.31849`，参数范数仍只缓慢变化。该结果覆盖 89.75%，并证明 host OOM 前没有数值发散，但最终 step 9999 和恢复接续性仍须由 attempt 7 验证。

## 待验证

- batch-16 单卡缩小版 10,000-step pretraining 的 loss 曲线、后续/最终 checkpoint 与最终 checkpoint 的实际 restore 验证。
- 三任务 Task-FT、RETAIN、coFT 的 ID/OOD/generalist 成功率。
- attempts 6--7 step 6750 之后直至 10,000 的完整 loss、grad norm、parameter norm 曲线与最终有限值审计。

## 已知限制

- 当前服务器没有论文真机实验所需的 Franka 机械臂和现场相机，因此只执行 LIBERO 仿真部分。
- 公开仓库固定的 JAX 0.5.0 无法在本服务器 sm_120 上完成模型编译；主实验因此使用独立 JAX 0.6.2 runtime overlay。该调整只改变硬件编译/runtime，不改变论文模型、数据、优化器、步数或参数合并方法，并保留原环境与失败证据。
- 论文训练 batch 为 64；attempts 3–5 证明它不能在当前单卡上完成首个 optimizer step，因此执行中的单卡协议缩小为 batch 16 且没有 gradient accumulation。它不与论文 batch 64 数值等价，最终成功率、收敛速度和合并结果只能作为明确标注的单卡缩小复现。
