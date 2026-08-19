# RETAIN LIBERO 锁定复现协议

协议编号：`RETAIN-GPU-20260819-001`

## 1. 研究问题

检验线性参数合并 RETAIN 是否能在保留目标任务适应能力的同时，缓解 VLA policy 对旧任务能力与 OOD robustness 的遗忘。

## 2. 数据

### Generalist pretraining（117 tasks）

- `libero_goal_reduced`：52,042 transitions
- `libero_object_reduced`：66,984 transitions
- `libero_spatial_reduced`：52,970 transitions
- `libero_90_flipped`：567,494 transitions

四个数据集按 transition 数量成比例采样。

### Target-task finetuning（LIBERO-10）

1. `turn on the stove and put the moka pot on it`
2. `put the white mug on the left plate and put the yellow and white mug on the right plate`
3. `put both the alphabet soup and the cream cheese box in the basket`

使用作者公开的过滤后 RLDS 数据，每个任务约 45 demonstrations。观测包含 base/wrist RGB，state/action 为 7D；action horizon 为 50，并由 `π0` 输入层 pad 到 32D。

## 3. 方法

### Task-FT

从 117-task pretrained checkpoint 出发，分别在三个目标任务上 full finetuning。stove 与 basket 各 500 steps，mugs 为 1,000 steps。

### RETAIN

对每个 Task-FT checkpoint 与同一个 pretrained checkpoint 做线性插值：

`θ_RETAIN = (1 - α) θ_pre + α θ_ft`

论文最终报告的 `α`：stove 0.9、mugs 0.8、basket 0.9。为审计超参数选择，同时保留 `{0.1, 0.2, ..., 0.9}` sweep 的评测入口；只用一个 OOD validation setting 选 α，其余场景作为 test。

### coFT baseline

每个 batch 的期望采样比例为 50% target task + 50% 四个 pretraining datasets；pretraining 内部仍按 transition 比例分配。每个任务训练 1,000 steps。

## 4. 训练超参数

- Global batch size：64
- Optimizer：AdamW，β₁=0.9，β₂=0.95，ε=1e-8，weight decay=1e-10
- Gradient clipping：global norm 1.0
- Cosine schedule：warmup 1,000，peak LR 2.5e-5，decay 30,000，end LR 2.5e-6
- Seed：42
- Pretraining：10,000 steps
- Normalization：固定使用 pretraining datasets 的 mean/std
- 单卡执行；用 `CUDA_VISIBLE_DEVICES` 限定唯一 GPU

## 5. 评测协议

- 推理：每次预测 50-step action chunk，但 open-loop 执行前 5 steps。
- 每个 episode 开始先等待 10 simulation steps。
- ID：每个目标任务 20 episodes。
- OOD：每个 setting 使用 5 seeds × 每 seed 10 episodes。
- Generalist retention：从 LIBERO-spatial/object/goal/90 各取 5 个任务，共 20 tasks；每任务 10 episodes。
- 指标：binary task success rate；同时保存逐 episode 结果、seed、场景参数和视频。

OOD 具体平移、distractor 与 background swap 参数以 `examples/libero/generate_all_arg_combinations.py` 为唯一可执行来源，并将实际生成的命令原样归档。

## 6. 完成标准

1. 数据 revision、代码 commit、依赖与 GPU 信息均有记录。
2. 每个训练 run 有命令、stdout/stderr、loss 指标、最终 checkpoint 清单。
3. 每个评测单元有逐 episode 输出和汇总成功率。
4. 结果目录包含 Task-FT、RETAIN、coFT 的可比表格以及论文报告值对照。
5. 所有失败、重跑、参数偏差和资源限制均写入 `research-log.md` 与 `findings.md`。
