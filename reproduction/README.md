# RETAIN 论文复现工作区

本目录用于复现论文 **Robust Finetuning of Vision-Language-Action Robot Policies via Parameter Merging**（RETAIN，arXiv:2512.08333v3）的 LIBERO 仿真实验。

## 复现范围

- 使用官方 `π0 base` 权重，在 117 个 LIBERO 任务上进行 10,000 steps 的 generalist pretraining。
- 在 LIBERO-10 的 3 个目标任务上进行 Task-FT。
- 通过线性参数合并得到 RETAIN：`θ_RETAIN = (1-α) θ_pre + α θ_ft`。
- 复现论文中的 co-finetuning（coFT）基线。
- 评测 in-distribution（ID）、OOD-easy / medium / hard，以及 20 个旧任务上的 generalist retention。
- DROID 真机实验需要 Franka 机械臂、相机与现场布置，不属于当前纯服务器环境的可执行范围。

## 文件布局

- `research-state.yaml`：机器可读的当前状态和下一步。
- `research-log.md`：按时间追加的中文实验日志。
- `findings.md`：经验证结论、失败模式和偏差说明。
- `literature/protocol.md`：锁定后的论文实验协议。
- `data/`：数据版本、校验结果和清单（大文件保存在共享盘）。
- `experiments/`：每个 run 的命令、环境、日志与指标。
- `results/`：汇总表、评测输出和视频索引。
- `to_human/`：供人工快速阅读的最终中文报告。

## 服务器路径

- 项目内记录与结果：`/root/RETAIN_code/reproduction`
- 官方 RLDS 数据：`/shared/.cache/retain/libero/datasets`
- 模型 checkpoint：`/shared/.cache/retain/checkpoints`
- OpenPI / Hugging Face 缓存：`/shared/.cache/retain/openpi`、`/shared/.cache/retain/huggingface`

大体积 checkpoint 放在共享盘，项目目录保存清单、相对关系、日志、指标和视频，避免根分区写满。
