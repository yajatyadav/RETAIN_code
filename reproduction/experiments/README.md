# 实验目录约定

每个 run 使用一个独立目录，至少包含：

- `command.txt`：实际执行命令；
- `environment.txt`：代码 commit、Python/JAX/CUDA/GPU 信息；
- `stdout.log`：完整标准输出与错误；
- `metrics.jsonl`：可解析训练或评测指标；
- `status.yaml`：开始/结束时间、退出码、checkpoint 与异常说明。

run 名称格式：`YYYYMMDD-HHMM_<stage>_<task>_<method>`。
