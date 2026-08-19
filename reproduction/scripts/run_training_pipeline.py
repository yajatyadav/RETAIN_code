#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, UP017
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
    if missing:
        raise FileNotFoundError(f"训练输入尚未就绪: {missing}")


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

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
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
        "status": "running",
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
    stages = STAGES + (COFT_STAGES if args.include_coft else ())
    for config, exp_name, final_step in stages:
        run_stage(gpu, config, exp_name, final_step)
    print(f"{now()} 全部训练阶段完成。", flush=True)


if __name__ == "__main__":
    main()
