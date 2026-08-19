#!/usr/bin/env python3
# ruff: noqa: RUF001, UP017
"""从输入校验到单卡训练、评测的可恢复总控程序。"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
DATA_ROOT = Path("/shared/.cache/retain/libero/datasets")
BASE_PARAMS = Path(
    "/shared/.cache/retain/openpi/openpi-assets/checkpoints/pi0_base/params"
)
REFERENCE_MANIFEST = PROJECT_ROOT / "reproduction/data/retain_dataset_manifest_sha.json"
SERVER_MANIFEST = PROJECT_ROOT / "reproduction/data/dataset_manifest_server.json"
BASE_REPORT = PROJECT_ROOT / "reproduction/data/pi0_base_restore.json"
SMOKE_REPORT = PROJECT_ROOT / "reproduction/experiments/input-smoke-all.json"
EXPERIMENT_ROOT = PROJECT_ROOT / "reproduction/experiments"
STATUS_PATH = EXPERIMENT_ROOT / "supervisor-status.json"
ENVIRONMENT_PATH = EXPERIMENT_ROOT / "environment.json"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def atomic_json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def update_status(stage: str, **details: object) -> None:
    payload = {
        "protocol_id": "RETAIN-GPU-20260819-001",
        "stage": stage,
        "updated_at": now(),
        **details,
    }
    atomic_json_dump(STATUS_PATH, payload)
    print(f"{payload['updated_at']} stage={stage} {details}", flush=True)


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"{now()} 执行: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def process_lines() -> list[str]:
    output = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
    return output.splitlines()


def data_transfer_active() -> bool:
    return any(
        "aria2c --input-file=/tmp/retain_hf_aria2.txt" in line
        for line in process_lines()
    )


def base_transfer_active() -> bool:
    for line in process_lines():
        if "aria2c --input-file=/tmp/tmplm72ces0" in line:
            return True
        if "rsync" in line and "pi0_base/params" in line:
            return True
    return False


def reference_files() -> dict[str, dict]:
    if not REFERENCE_MANIFEST.is_file():
        raise FileNotFoundError(REFERENCE_MANIFEST)
    payload = json.loads(REFERENCE_MANIFEST.read_text(encoding="utf-8"))
    return {item["path"]: item for item in payload["files"]}


def dataset_size_progress(expected: dict[str, dict]) -> tuple[int, int]:
    complete = 0
    complete_bytes = 0
    for relative, record in expected.items():
        path = DATA_ROOT / relative
        if path.is_file() and path.stat().st_size == int(record["bytes"]):
            complete += 1
            complete_bytes += int(record["bytes"])
    return complete, complete_bytes


def wait_for_dataset(poll_seconds: int) -> None:
    expected = reference_files()
    expected_bytes = sum(int(record["bytes"]) for record in expected.values())
    while True:
        complete, complete_bytes = dataset_size_progress(expected)
        sidecars = sum(1 for _ in DATA_ROOT.rglob("*.aria2")) if DATA_ROOT.exists() else 0
        active = data_transfer_active()
        update_status(
            "waiting_dataset_transfer",
            complete_files=complete,
            expected_files=len(expected),
            complete_bytes=complete_bytes,
            expected_bytes=expected_bytes,
            aria2_sidecars=sidecars,
            transfer_active=active,
        )
        if complete == len(expected) and sidecars == 0 and not active:
            return
        if not active:
            raise RuntimeError(
                "数据传输已停止但输入不完整；保留现场，等待补传后重启 supervisor"
            )
        time.sleep(poll_seconds)


def verify_dataset() -> None:
    run(
        [
            str(PYTHON),
            "reproduction/scripts/verify_dataset.py",
            "--root",
            str(DATA_ROOT),
            "--output",
            str(SERVER_MANIFEST),
        ]
    )
    reference = json.loads(REFERENCE_MANIFEST.read_text(encoding="utf-8"))
    actual = json.loads(SERVER_MANIFEST.read_text(encoding="utf-8"))
    expected_files = {
        item["path"]: (int(item["bytes"]), item["sha256"])
        for item in reference["files"]
    }
    actual_files = {
        item["path"]: (int(item["bytes"]), item["sha256"])
        for item in actual["files"]
    }
    if actual_files != expected_files:
        missing = sorted(set(expected_files) - set(actual_files))
        extra = sorted(set(actual_files) - set(expected_files))
        changed = sorted(
            path
            for path in set(expected_files) & set(actual_files)
            if expected_files[path] != actual_files[path]
        )
        raise RuntimeError(
            f"服务器数据清单不一致: missing={missing}, extra={extra}, changed={changed}"
        )
    update_status(
        "dataset_verified",
        file_count=actual["file_count"],
        total_bytes=actual["total_bytes"],
        manifest=str(SERVER_MANIFEST),
    )


def wait_for_base_transfers(poll_seconds: int) -> None:
    while base_transfer_active():
        update_status("waiting_base_transfer")
        time.sleep(poll_seconds)


def verify_base() -> None:
    run(
        [
            str(PYTHON),
            "reproduction/scripts/download_gcs_prefix.py",
            "--bucket",
            "openpi-assets",
            "--prefix",
            "checkpoints/pi0_base/params",
            "--output",
            str(BASE_PARAMS),
            "--jobs",
            "8",
            "--backend",
            "aria2",
        ]
    )
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "OPENPI_DATA_HOME": "/shared/.cache/retain/openpi",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
    run(
        [
            str(PYTHON),
            "reproduction/scripts/verify_base_checkpoint.py",
            "--params",
            str(BASE_PARAMS),
            "--output",
            str(BASE_REPORT),
        ],
        env=env,
    )
    update_status("base_verified", report=str(BASE_REPORT))


def smoke_all_inputs() -> None:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "OPENPI_DATA_HOME": "/shared/.cache/retain/openpi",
            "HF_HOME": "/shared/.cache/retain/huggingface",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
    run(
        [
            str(PYTHON),
            "reproduction/scripts/smoke_test_training_inputs.py",
            "--output",
            str(SMOKE_REPORT),
        ],
        env=env,
    )
    update_status("all_inputs_verified", report=str(SMOKE_REPORT))


def collect_environment() -> None:
    packages = {}
    for name in ("jax", "jaxlib", "tensorflow", "torch", "orbax-checkpoint"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu_query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    payload = {
        "generated_at": now(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "packages": packages,
        "gpu_inventory": gpu_query,
        "gpu_limit": 1,
        "protocol_id": "RETAIN-GPU-20260819-001",
        "paper_code_base_commit": "0bbc6cf",
    }
    atomic_json_dump(ENVIRONMENT_PATH, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    lock_stream = (EXPERIMENT_ROOT / "reproduction-supervisor.lock").open(
        "w", encoding="utf-8"
    )
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("已有 reproduction supervisor 正在运行，本进程退出。", file=sys.stderr)
        raise SystemExit(2) from None

    try:
        collect_environment()
        free_bytes = shutil.disk_usage("/shared").free
        update_status("started", shared_free_bytes=free_bytes)
        if free_bytes < 150_000_000_000:
            raise RuntimeError(f"共享盘剩余空间不足 150 GB: {free_bytes}")
        wait_for_dataset(args.poll_seconds)
        update_status("verifying_dataset_sha256")
        verify_dataset()
        wait_for_base_transfers(args.poll_seconds)
        update_status("verifying_base_gcs_md5")
        verify_base()
        update_status("smoke_testing_all_training_inputs")
        smoke_all_inputs()
        update_status("training_pipeline")
        run(
            [
                str(PYTHON),
                "reproduction/scripts/run_training_pipeline.py",
                "--include-coft",
                "--poll-seconds",
                str(args.poll_seconds),
            ]
        )
        update_status("evaluation_pipeline")
        run(
            [
                str(PYTHON),
                "reproduction/scripts/run_evaluation_pipeline.py",
                "--poll-seconds",
                str(args.poll_seconds),
            ]
        )
        update_status("completed")
    except Exception as error:
        update_status("failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
