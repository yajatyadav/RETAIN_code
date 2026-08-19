# JAX 0.6.2 CUDA 动态库隔离失败诊断

## 事件

- 2026-08-20 03:49:48 CST，GPU 2 达到 `508 MiB / 0%`，training pipeline 按双阈值选中该卡。
- 最小 BF16 预检进程没有成功加载 CUDA libraries，JAX 回退到 `TFRT_CPU_0`；原门禁只检查子进程退出码和 JSON 可解析性，错误地把 CPU 结果写成 `verified`。
- 随后完整训练进程通过 TensorFlow 预加载了主 `.venv` 的 cuDNN `9.7.1`，而 JAX CUDA plugin 0.6.2 的编译版本要求 cuDNN `9.8.0`。训练在创建随机数 key 时以 `DNN library initialization failed` 退出，return code 为 `1`。
- 故障发生在模型 train state、step 0 和数值 checkpoint 之前；checkpoint 目录仍没有数字 step。

## 根因

初版 overlay 只包含 JAX/JAXlib/CUDA plugin 与 cuDNN 9.8，其他 CUDA wheels 继续复用主环境。overlay 中的常规 `nvidia` Python package 遮蔽了主环境的同名 package，使纯 JAX 预检无法导入 `nvidia.cuda_runtime` 等模块；完整训练又因 TensorFlow 的导入顺序先加载了主环境 cuDNN 9.7。两条路径分别造成 CPU fallback 与 cuDNN minor-version mismatch。

## 修复

1. 在同一隔离 overlay 中固定 JAX plugin 直接依赖的整套 CUDA wheels，版本与 JAX 0.6.2 的 build versions 对齐；主 `.venv` 不变。
2. 训练和 policy server 显式把 overlay 内各 `nvidia/*/lib` 目录置于 `LD_LIBRARY_PATH` 最前，并把 overlay 的 `cuda_nvcc/bin` 置于 `PATH` 最前。
3. GPU preflight 现在同时要求：仅 1 个设备可见、`device.platform == "gpu"`、BF16→FP16 与 BF16 GEMM 均完成；CPU fallback 即使退出码为 0 也会失败。
4. 旧的错误 preflight 已改名为 `jax-gpu-preflight.invalid-cpu-fallback-20260820T0350.json`，不会被当作当前成功证据。

## 修复后验证

- 17 个固定 overlay packages 可见。
- 12 个关键 CUDA 动态库均从 overlay 绝对路径成功加载，CPU BF16 smoke 通过；报告为 `reproduction/experiments/jax062-overlay-audit.json`。
- supervisor 重启后再次通过 353/353 数据 SHA-256、24/24 基础权重 size/MD5、Orbax restore（结构哈希不变）和 7/7 真实输入 smoke tests。
- 实际 sm_120 GPU BF16 门禁仍需等待下一张真正空闲的 GPU，不能用本次 CPU 审计替代。
