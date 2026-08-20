#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003, UP017
"""等待一张空闲 GPU，并顺序执行 RETAIN 论文训练阶段。"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT / "reproduction" / "experiments"
CHECKPOINT_ROOT = Path("/shared/.cache/retain/checkpoints")
DATA_ROOT = Path("/shared/.cache/retain/libero/datasets")
JAX_OVERLAY = Path(
    os.environ.get(
        "RETAIN_JAX_OVERLAY",
        "/shared/.cache/retain/jax-overlays/0.6.2",
    )
)
JAX_RUNTIME_VERSIONS = {
    "jax": "0.6.2",
    "jaxlib": "0.6.2",
    "jax-cuda12-plugin": "0.6.2",
    "jax-cuda12-pjrt": "0.6.2",
    "ml-dtypes": "0.5.1",
    "nvidia-cublas-cu12": "12.8.3.14",
    "nvidia-cuda-cupti-cu12": "12.8.57",
    "nvidia-cuda-nvcc-cu12": "12.9.86",
    "nvidia-cuda-nvrtc-cu12": "12.8.61",
    "nvidia-cuda-runtime-cu12": "12.8.57",
    "nvidia-cudnn-cu12": "9.8.0.87",
    "nvidia-cufft-cu12": "11.3.3.41",
    "nvidia-cusolver-cu12": "11.7.2.55",
    "nvidia-cusparse-cu12": "12.5.7.53",
    "nvidia-nccl-cu12": "2.26.2",
    "nvidia-nvjitlink-cu12": "12.8.61",
    "nvidia-nvshmem-cu12": "3.2.5",
}
XLA_COMPATIBILITY_FLAG = "--xla_gpu_enable_triton_gemm=false"
TRAINING_XLA_MEM_FRACTION = "0.95"
GPU_PREFLIGHT_REPORT = EXPERIMENT_ROOT / "jax-gpu-preflight.json"
BASE_PARAMS = Path(
    "/shared/.cache/retain/openpi/openpi-assets/checkpoints/pi0_base/params"
)

STAGES = (
    ("retain_repro_pretrain", "paper_final_hparams", 9999),
    ("retain_repro_task_ft_stove", "paper_task_ft_stove", 499),
    ("retain_repro_task_ft_mugs", "paper_task_ft_mugs", 999),
    ("retain_repro_task_ft_basket", "paper_task_ft_basket", 499),
)
COFT_STAGES = (
    ("retain_repro_coft_stove", "paper_coft_stove", 999),
    ("retain_repro_coft_mugs", "paper_coft_mugs", 999),
    ("retain_repro_coft_basket", "paper_coft_basket", 999),
)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def atomic_json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def gpu_inventory() -> list[dict[str, int]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    inventory = []
    for line in output.splitlines():
        index, used, total, utilization = (int(value.strip()) for value in line.split(","))
        inventory.append(
            {
                "index": index,
                "memory_used_mib": used,
                "memory_total_mib": total,
                "utilization_percent": utilization,
            }
        )
    return inventory


def wait_for_idle_gpu(poll_seconds: int) -> int:
    while True:
        inventory = gpu_inventory()
        candidates = [
            gpu
            for gpu in inventory
            if gpu["memory_used_mib"] <= 2_048 and gpu["utilization_percent"] <= 10
        ]
        if candidates:
            selected = min(candidates, key=lambda gpu: (gpu["memory_used_mib"], gpu["index"]))
            print(f"{now()} 选择空闲 GPU {selected['index']}: {selected}", flush=True)
            return selected["index"]
        compact = ", ".join(
            f"GPU{gpu['index']}={gpu['memory_used_mib']}MiB/{gpu['utilization_percent']}%"
            for gpu in inventory
        )
        print(f"{now()} 暂无空闲 GPU；{compact}", flush=True)
        time.sleep(poll_seconds)


def checkpoint_dir(config: str, exp_name: str, final_step: int) -> Path:
    return CHECKPOINT_ROOT / config / exp_name / str(final_step)


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def export_metrics(run_dir: Path) -> int:
    """把人类可读训练日志转换为稳定的 JSONL 指标文件。"""
    log_path = run_dir / "stdout.log"
    if not log_path.is_file():
        return 0
    # tqdm 可能使用 carriage return，因此不能依赖逐行起始位置。
    step_pattern = re.compile(r"Step (\d+): ([^\r\n]+)")
    value_pattern = re.compile(r"([a-zA-Z_]+)=([-+0-9.eE]+)")
    by_step: dict[int, dict[str, float | int]] = {}
    for match in step_pattern.finditer(log_path.read_text(encoding="utf-8", errors="replace")):
        step = int(match.group(1))
        record: dict[str, float | int] = {"step": step}
        for key, value in value_pattern.findall(match.group(2)):
            record[key] = float(value)
        if len(record) > 1:
            by_step[step] = record
    output = "".join(
        json.dumps(by_step[step], ensure_ascii=False, sort_keys=True) + "\n"
        for step in sorted(by_step)
    )
    temporary = run_dir / "metrics.jsonl.tmp"
    temporary.write_text(output, encoding="utf-8")
    temporary.replace(run_dir / "metrics.jsonl")
    return len(by_step)


def prune_completed_training_state(final_checkpoint: Path, run_dir: Path) -> dict:
    """阶段完成后移除 optimizer state，保留 inference params 与审计记录。"""
    resolved_checkpoint = final_checkpoint.resolve()
    if not resolved_checkpoint.is_relative_to(CHECKPOINT_ROOT.resolve()):
        raise RuntimeError(f"拒绝清理 checkpoint root 外的路径: {resolved_checkpoint}")
    train_state = resolved_checkpoint / "train_state"
    record_path = run_dir / "storage-pruning.json"
    if not train_state.is_dir():
        if record_path.is_file():
            return json.loads(record_path.read_text(encoding="utf-8"))
        return {
            "status": "already_absent",
            "path": str(train_state),
            "checked_at": now(),
        }
    removed_bytes = directory_bytes(train_state)
    shutil.rmtree(train_state)
    record = {
        "status": "removed_after_successful_stage",
        "path": str(train_state),
        "removed_bytes": removed_bytes,
        "removed_at": now(),
        "retained_params": str(resolved_checkpoint / "params"),
        "reason": "控制共享盘占用；阶段已完成，后续训练和评测只读取 params",
    }
    atomic_json_dump(record_path, record)
    return record


def validate_inputs() -> None:
    missing = []
    if not DATA_ROOT.is_dir():
        missing.append(str(DATA_ROOT))
    if not BASE_PARAMS.is_dir():
        missing.append(str(BASE_PARAMS))
    overlay_files = (
        Path(".lock"),
        Path("jax/__init__.py"),
        Path("jaxlib/__init__.py"),
        Path("jax_plugins/xla_cuda12/__init__.py"),
        Path("nvidia/cublas/lib/libcublas.so.12"),
        Path("nvidia/cuda_runtime/lib/libcudart.so.12"),
        Path("nvidia/cudnn/lib/libcudnn.so.9"),
        Path("nvidia/cufft/lib/libcufft.so.11"),
        Path("nvidia/cusolver/lib/libcusolver.so.11"),
        Path("nvidia/cusparse/lib/libcusparse.so.12"),
        Path("nvidia/nccl/lib/libnccl.so.2"),
        Path("nvidia/nvjitlink/lib/libnvJitLink.so.12"),
    )
    missing.extend(
        str(JAX_OVERLAY / relative_path)
        for relative_path in overlay_files
        if not (JAX_OVERLAY / relative_path).exists()
    )
    if missing:
        raise FileNotFoundError(f"训练输入尚未就绪: {missing}")


def configure_jax_runtime(env: dict[str, str]) -> dict[str, str]:
    """为 sm_120 GPU 注入隔离的 JAX runtime，不修改项目主虚拟环境。"""
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{JAX_OVERLAY}{os.pathsep}{current_pythonpath}"
        if current_pythonpath
        else str(JAX_OVERLAY)
    )
    current_xla_flags = env.get("XLA_FLAGS", "").split()
    if XLA_COMPATIBILITY_FLAG not in current_xla_flags:
        current_xla_flags.append(XLA_COMPATIBILITY_FLAG)
    env["XLA_FLAGS"] = " ".join(current_xla_flags)
    library_dirs = sorted(
        path for path in (JAX_OVERLAY / "nvidia").glob("*/lib") if path.is_dir()
    )
    if not library_dirs:
        raise FileNotFoundError(f"JAX overlay 缺少 CUDA 动态库目录: {JAX_OVERLAY}")
    existing_library_path = [
        path for path in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if path
    ]
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [*(str(path) for path in library_dirs), *existing_library_path]
    )
    cuda_bin = JAX_OVERLAY / "nvidia" / "cuda_nvcc" / "bin"
    if cuda_bin.is_dir():
        env["PATH"] = f"{cuda_bin}{os.pathsep}{env.get('PATH', '')}"
    # CPU 预检可能显式设置该变量；GPU 子进程必须允许自动选择 CUDA backend。
    env.pop("JAX_PLATFORMS", None)
    return env


def verify_gpu_runtime(gpu: int) -> dict:
    """在已选空闲 GPU 上编译最小 BF16 kernel，先于完整模型发现 runtime 问题。"""
    env = configure_jax_runtime(os.environ.copy())
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
    code = """
import importlib.metadata as metadata
import json

import jax
import jax.numpy as jnp

source = jnp.arange(4096, dtype=jnp.bfloat16).reshape(64, 64)
converted = jax.jit(lambda value: value.astype(jnp.float16))(source)
product = jax.jit(lambda value: value @ value.T)(source)
jax.block_until_ready((converted, product))
devices = jax.devices()
if len(devices) != 1:
    raise RuntimeError(f"预期恰好 1 个可见设备，实际为 {devices}")
device = devices[0]
if device.platform != "gpu":
    raise RuntimeError(f"预期 GPU backend，实际为 {device.platform}: {device}")
print(json.dumps({
    "status": "verified",
    "jax": metadata.version("jax"),
    "jaxlib": metadata.version("jaxlib"),
    "jax_cuda12_plugin": metadata.version("jax-cuda12-plugin"),
    "platform": device.platform,
    "device": str(device),
    "device_kind": device.device_kind,
    "visible_device_count": len(devices),
    "bf16_to_f16_shape": list(converted.shape),
    "bf16_matmul_shape": list(product.shape),
}, ensure_ascii=False))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    record = {
        "checked_at": now(),
        "gpu_physical_index": gpu,
        "overlay": str(JAX_OVERLAY),
        "xla_flags": env["XLA_FLAGS"],
        "cuda_library_dirs": env["LD_LIBRARY_PATH"].split(os.pathsep),
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "status": "failed",
    }
    if completed.returncode == 0:
        try:
            details = json.loads(completed.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as error:
            record["parse_error"] = str(error)
        else:
            record["runtime"] = details
            if details.get("platform") == "gpu" and details.get("visible_device_count") == 1:
                record["status"] = "verified"
            else:
                record["validation_error"] = "预检未在唯一 GPU backend 上执行"
    atomic_json_dump(GPU_PREFLIGHT_REPORT, record)
    if record["status"] != "verified":
        raise RuntimeError(f"JAX GPU runtime 预检失败，记录: {GPU_PREFLIGHT_REPORT}")
    print(f"{now()} JAX GPU runtime 预检通过: {record['runtime']}", flush=True)
    return record


def run_stage(gpu: int, config: str, exp_name: str, final_step: int) -> None:
    final_checkpoint = checkpoint_dir(config, exp_name, final_step)
    run_dir = EXPERIMENT_ROOT / f"{config}__{exp_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if (final_checkpoint / "params").is_dir():
        metrics_count = export_metrics(run_dir)
        prune_record = prune_completed_training_state(final_checkpoint, run_dir)
        print(f"{now()} 跳过已完成阶段 {config}: {final_checkpoint}", flush=True)
        print(
            f"{now()} 已归档 {metrics_count} 条指标；storage={prune_record['status']}",
            flush=True,
        )
        return

    checkpoint_run_dir = CHECKPOINT_ROOT / config / exp_name
    command = [
        "/root/.local/bin/uv",
        "run",
        "--no-sync",
        "scripts/train.py",
        config,
        f"--exp-name={exp_name}",
    ]
    if checkpoint_run_dir.exists():
        command.append("--resume")

    env = configure_jax_runtime(os.environ.copy())
    # Attempt 5 showed that cudaMallocAsync is not selected by this JAX 0.6.2
    # runtime.  Remove any inherited allocator override and use a large fixed
    # BFC arena with the single-GPU batch-16 adaptation.
    for variable in (
        "XLA_PYTHON_CLIENT_ALLOCATOR",
        "TF_GPU_ALLOCATOR",
        "TF_CUDA_MALLOC_ASYNC_SUPPORTED_PREALLOC",
    ):
        env.pop(variable, None)
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "XLA_PYTHON_CLIENT_MEM_FRACTION": TRAINING_XLA_MEM_FRACTION,
            "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
            "TF_FORCE_GPU_ALLOW_GROWTH": "true",
            "UV_CACHE_DIR": "/shared/.cache/retain/uv",
            "HF_HOME": "/shared/.cache/retain/huggingface",
            "OPENPI_DATA_HOME": "/shared/.cache/retain/openpi",
            "WANDB_MODE": "disabled",
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
        }
    )
    status = {
        "config": config,
        "exp_name": exp_name,
        "host": socket.gethostname(),
        "gpu_physical_index": gpu,
        "started_at": now(),
        "command": command,
        "final_checkpoint": str(final_checkpoint),
        "jax_runtime": {
            "overlay": str(JAX_OVERLAY),
            "versions": JAX_RUNTIME_VERSIONS,
            "xla_flags": env["XLA_FLAGS"],
            "reason": "RTX 6000D (sm_120) 编译兼容；项目主虚拟环境保持不变",
            "gpu_preflight_report": str(GPU_PREFLIGHT_REPORT),
            "memory_policy": {
                "allocator": "bfc",
                "xla_python_client_mem_fraction": TRAINING_XLA_MEM_FRACTION,
                "xla_python_client_preallocate": True,
                "tf_force_gpu_allow_growth": True,
                "reason": (
                    "batch-64 首步临时内存超过单卡预算；batch-16 下保留 95% BFC pool"
                ),
            },
            "single_gpu_adaptation": {
                "paper_batch_size": 64,
                "actual_batch_size": 16,
                "gradient_accumulation": False,
                "reason": (
                    "attempts 3--5 均在 batch-64 首个 optimizer step 因 27.79 GiB "
                    "buffer 分配失败；单 RTX 6000D 约束下缩小物理 batch"
                ),
            },
        },
        "status": "running",
        "checkpoint_storage_policy": {
            "final_params_only": True,
            "pretraining_intermediate_step_9000_skipped": config == "retain_repro_pretrain",
            "reason": (
                "step-9000 full train-state save triggered host OOM; terminal stages only "
                "need inference/EMA params downstream"
            ),
        },
    }
    atomic_json_dump(run_dir / "status.json", status)
    (run_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")

    print(f"{now()} 启动 {config}，日志: {run_dir / 'stdout.log'}", flush=True)
    with (run_dir / "stdout.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return_code = process.wait()

    status["finished_at"] = now()
    status["return_code"] = return_code
    status["status"] = "completed" if return_code == 0 else "failed"
    atomic_json_dump(run_dir / "status.json", status)
    if return_code != 0:
        raise RuntimeError(f"{config} 失败，退出码 {return_code}")
    if not (final_checkpoint / "params").is_dir():
        raise RuntimeError(f"{config} 正常退出但缺少最终 checkpoint: {final_checkpoint}")
    metrics_count = export_metrics(run_dir)
    prune_record = prune_completed_training_state(final_checkpoint, run_dir)
    status["metrics_records"] = metrics_count
    status["storage_pruning"] = prune_record
    atomic_json_dump(run_dir / "status.json", status)
    print(f"{now()} 完成 {config}: {final_checkpoint}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--include-coft", action="store_true")
    args = parser.parse_args()

    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = EXPERIMENT_ROOT / "training-pipeline.lock"
    lock_stream = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("已有训练 pipeline 正在运行，本进程退出。", file=sys.stderr)
        raise SystemExit(2) from None

    validate_inputs()
    gpu = wait_for_idle_gpu(args.poll_seconds)
    verify_gpu_runtime(gpu)
    stages = STAGES + (COFT_STAGES if args.include_coft else ())
    for config, exp_name, final_step in stages:
        run_stage(gpu, config, exp_name, final_step)
    print(f"{now()} 全部训练阶段完成。", flush=True)


if __name__ == "__main__":
    main()
