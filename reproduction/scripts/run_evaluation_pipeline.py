#!/usr/bin/env python3
# ruff: noqa: RUF001, UP017
"""在单张 GPU 上顺序执行 RETAIN 的 LIBERO 评测协议。"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIBERO_DIR = PROJECT_ROOT / "examples" / "libero"
LIBERO_PYTHON = LIBERO_DIR / ".venv" / "bin" / "python"
UV = Path("/root/.local/bin/uv")
CHECKPOINT_ROOT = Path("/shared/.cache/retain/checkpoints")
RESULTS_ROOT = Path("/shared/.cache/retain/results/RETAIN-GPU-20260819-001")
DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / "reproduction" / "experiments" / "evaluation-pipeline"
EXPERIMENT_ROOT = Path(
    os.environ.get("RETAIN_EVAL_STATE_ROOT", str(DEFAULT_EXPERIMENT_ROOT))
).expanduser()
JAX_OVERLAY = Path(
    os.environ.get(
        "RETAIN_JAX_OVERLAY",
        "/shared/.cache/retain/jax-overlays/0.6.2",
    )
)
XLA_COMPATIBILITY_FLAG = "--xla_gpu_enable_triton_gemm=false"
PRETRAIN = CHECKPOINT_ROOT / "retain_repro_pretrain" / "paper_final_hparams" / "9999"

TASKS = {
    "stove": "turn on the stove and put the moka pot on it",
    "mugs": (
        "put the white mug on the left plate and put the yellow and white mug "
        "on the right plate"
    ),
    "basket": "put both the alphabet soup and the cream cheese box in the basket",
}
PAPER_ALPHAS = {
    "taskft": {"stove": 0.9, "mugs": 0.8, "basket": 0.9},
    "coft": {"stove": 0.7, "mugs": 0.9, "basket": 0.9},
}
FINAL_STEPS = {
    "taskft": {"stove": 499, "mugs": 999, "basket": 499},
    "coft": {"stove": 999, "mugs": 999, "basket": 999},
}


@dataclasses.dataclass(frozen=True)
class PolicySpec:
    name: str
    config: str
    checkpoint_dirs: tuple[Path, ...]
    alpha: float | None = None

    @property
    def merged(self) -> bool:
        return len(self.checkpoint_dirs) > 1


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


def load_state() -> dict:
    path = EXPERIMENT_ROOT / "status.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "protocol_id": "RETAIN-GPU-20260819-001",
        "created_at": now(),
        "completed_jobs": {},
        "alpha_sweep": {},
        "status": "initialized",
    }


def checkpoint_for(family: str, task: str) -> Path:
    config_token = "task_ft" if family == "taskft" else family
    config = f"retain_repro_{config_token}_{task}"
    exp_name = f"paper_{config_token}_{task}"
    return CHECKPOINT_ROOT / config / exp_name / str(FINAL_STEPS[family][task])


def required_checkpoints() -> list[Path]:
    paths = [PRETRAIN]
    paths.extend(
        checkpoint_for(family, task)
        for family in ("taskft", "coft")
        for task in TASKS
    )
    return paths


def wait_for_checkpoints(poll_seconds: int) -> None:
    while True:
        missing = [path for path in required_checkpoints() if not (path / "params").is_dir()]
        if not missing:
            return
        print(f"{now()} 等待 {len(missing)} 个训练 checkpoint", flush=True)
        time.sleep(poll_seconds)


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
        candidates = [
            gpu
            for gpu in gpu_inventory()
            if gpu["memory_used_mib"] <= 2_048 and gpu["utilization_percent"] <= 10
        ]
        if candidates:
            selected = min(candidates, key=lambda item: (item["memory_used_mib"], item["index"]))
            print(f"{now()} 评测选择空闲 GPU {selected['index']}: {selected}", flush=True)
            return selected["index"]
        print(f"{now()} 评测等待空闲 GPU", flush=True)
        time.sleep(poll_seconds)


def wait_for_selected_gpu(gpu_index: int, poll_seconds: int = 30) -> None:
    while True:
        selected = next(item for item in gpu_inventory() if item["index"] == gpu_index)
        if selected["memory_used_mib"] <= 2_048 and selected["utilization_percent"] <= 10:
            return
        print(
            f"{now()} GPU {gpu_index} 被其他进程占用，暂停加载下一策略: {selected}",
            flush=True,
        )
        time.sleep(poll_seconds)


def server_env(gpu: int, policy_name: str) -> dict[str, str]:
    env = os.environ.copy()
    if not (JAX_OVERLAY / ".lock").is_file():
        raise FileNotFoundError(f"缺少已验证的 JAX compatibility overlay: {JAX_OVERLAY}")
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
    env.pop("JAX_PLATFORMS", None)
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
            "UV_CACHE_DIR": "/shared/.cache/retain/uv",
            "HF_HOME": "/shared/.cache/retain/huggingface",
            "OPENPI_DATA_HOME": "/shared/.cache/retain/openpi",
            "OPENPI_DEBUG_DIR": str(RESULTS_ROOT / "debug" / policy_name),
            "WANDB_MODE": "disabled",
        }
    )
    return env


def evaluation_env(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    egl_root = Path("/shared/.cache/retain/nvidia-egl")
    egl_lib = egl_root / "lib"
    env.update(
        {
            "PATH": f"{LIBERO_PYTHON.parent}:{env.get('PATH', '')}",
            "CUDA_VISIBLE_DEVICES": "",
            "LIBERO_CONFIG_PATH": "/shared/.cache/retain/libero",
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "MUJOCO_EGL_DEVICE_ID": str(gpu),
            "__EGL_VENDOR_LIBRARY_FILENAMES": str(
                egl_root / "driver-595.71.05" / "10_nvidia.json"
            ),
            "LD_LIBRARY_PATH": f"{egl_lib}:{env.get('LD_LIBRARY_PATH', '')}",
            "WANDB_MODE": "disabled",
        }
    )
    return env


def server_command(spec: PolicySpec, port: int) -> list[str]:
    if not spec.merged:
        return [
            str(UV),
            "run",
            "--no-sync",
            "scripts/serve_policy.py",
            "--port",
            str(port),
            "policy:checkpoint",
            f"--policy.config={spec.config}",
            f"--policy.dir={spec.checkpoint_dirs[0]}",
        ]
    if spec.alpha is None:
        raise ValueError(f"合并策略缺少 alpha: {spec.name}")
    weights = [spec.alpha, 1.0 - spec.alpha]
    return [
        str(UV),
        "run",
        "--no-sync",
        "scripts/merging_experiments.py",
        "--port",
        str(port),
        "--config",
        spec.config,
        "--merging_fn",
        "linear_interpolation",
        "--merging_fn_kwargs",
        json.dumps({"model_mixing_coefficients": weights}),
        "--checkpoint_dirs",
        *(str(path) for path in spec.checkpoint_dirs),
    ]


def wait_for_server(process: subprocess.Popen, port: int, timeout_seconds: int = 900) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"policy server 提前退出，退出码 {process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"policy server 在 {timeout_seconds}s 内未监听端口 {port}")


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def generate_jobs(
    *,
    policy_name: str,
    task_name: str,
    port: int,
    flags: tuple[str, ...],
    env: dict[str, str],
) -> list[list[str]]:
    command = [
        str(LIBERO_PYTHON),
        str(LIBERO_DIR / "generate_all_arg_combinations.py"),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--checkpoint_name",
        policy_name,
        "--results_root_dir",
        str(RESULTS_ROOT / "rollouts"),
        "--task_name",
        task_name,
        *flags,
    ]
    output = subprocess.check_output(command, cwd=LIBERO_DIR, env=env, text=True)
    jobs = [shlex.split(line) for line in output.splitlines() if line.startswith("python ")]
    if not jobs:
        raise RuntimeError(f"没有生成评测任务: {' '.join(command)}")
    return jobs


def job_key(command: list[str]) -> str:
    return hashlib.sha256("\0".join(command).encode()).hexdigest()[:16]


def evaluate_policy(
    spec: PolicySpec,
    *,
    target_flags: list[tuple[str, tuple[str, ...]]],
    include_generalist: bool,
    gpu: int,
    port: int,
    state: dict,
) -> None:
    eval_env = evaluation_env(gpu)
    jobs = []
    for task_name, flags in target_flags:
        jobs.extend(
            generate_jobs(
                policy_name=spec.name,
                task_name=task_name,
                port=port,
                flags=flags,
                env=eval_env,
            )
        )
    if include_generalist:
        jobs.extend(
            generate_jobs(
                policy_name=spec.name,
                task_name=target_flags[0][0],
                port=port,
                flags=("--do_generalist",),
                env=eval_env,
            )
        )

    policy_dir = EXPERIMENT_ROOT / spec.name
    policy_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(
        policy_dir / "jobs.json",
        [{"key": job_key(job), "command": job} for job in jobs],
    )
    pending = [job for job in jobs if job_key(job) not in state["completed_jobs"]]
    if not pending:
        print(f"{now()} 跳过已完成策略 {spec.name}", flush=True)
        return

    wait_for_selected_gpu(gpu)
    command = server_command(spec, port)
    (policy_dir / "server-command.txt").write_text(
        shlex.join(command) + "\n",
        encoding="utf-8",
    )
    with (policy_dir / "server.log").open("a", encoding="utf-8") as server_log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=server_env(gpu, spec.name),
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            print(f"{now()} 加载评测策略 {spec.name}", flush=True)
            wait_for_server(process, port)
            for index, job in enumerate(pending, start=1):
                key = job_key(job)
                log_path = policy_dir / f"job-{key}.log"
                print(
                    f"{now()} {spec.name} 运行评测 {index}/{len(pending)} ({key})",
                    flush=True,
                )
                with log_path.open("w", encoding="utf-8") as log:
                    completed = subprocess.run(
                        job,
                        cwd=LIBERO_DIR,
                        env=eval_env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                    )
                if completed.returncode != 0:
                    state["status"] = "failed"
                    state["failed_job"] = {
                        "key": key,
                        "command": job,
                        "return_code": completed.returncode,
                        "log": str(log_path),
                        "failed_at": now(),
                    }
                    atomic_json_dump(EXPERIMENT_ROOT / "status.json", state)
                    raise RuntimeError(f"评测任务 {key} 失败，日志 {log_path}")
                state["completed_jobs"][key] = {
                    "policy": spec.name,
                    "command": job,
                    "finished_at": now(),
                }
                state["status"] = "running"
                state.pop("failed_job", None)
                atomic_json_dump(EXPERIMENT_ROOT / "status.json", state)
        finally:
            stop_server(process)
    time.sleep(5)


def validation_score(policy_name: str, task_name: str) -> dict[str, float | int]:
    root = RESULTS_ROOT / "rollouts" / policy_name
    summaries = []
    for path in root.rglob("summary.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("eval_type") == "OOD_MEDIUM" and payload.get("task_name") == task_name:
            summaries.append(payload)
    episodes = sum(int(item["total_episodes"]) for item in summaries)
    successes = sum(int(item["total_successes"]) for item in summaries)
    if len(summaries) != 5 or episodes != 50:
        raise RuntimeError(
            f"{policy_name} validation 结果不完整: {len(summaries)} summaries, {episodes} episodes"
        )
    return {"summaries": len(summaries), "episodes": episodes, "successes": successes, "rate": successes / episodes}


def merged_spec(family: str, task: str, alpha: float) -> PolicySpec:
    alpha_tag = f"a{round(alpha * 100):03d}"
    config_token = "task_ft" if family == "taskft" else family
    return PolicySpec(
        name=f"retain_{family}_{task}_{alpha_tag}",
        config=f"retain_repro_{config_token}_{task}",
        checkpoint_dirs=(checkpoint_for(family, task), PRETRAIN),
        alpha=alpha,
    )


def run_alpha_sweep(
    *, gpu: int, port: int, state: dict, include_sweep: bool
) -> dict[str, dict[str, float]]:
    selected: dict[str, dict[str, float]] = {"taskft": {}, "coft": {}}
    for family in ("taskft", "coft"):
        for task, task_name in TASKS.items():
            records = []
            if include_sweep:
                for index in range(1, 10):
                    alpha = index / 10
                    spec = merged_spec(family, task, alpha)
                    evaluate_policy(
                        spec,
                        target_flags=[(task_name, ("--do_ood_medium",))],
                        include_generalist=False,
                        gpu=gpu,
                        port=port,
                        state=state,
                    )
                    score = validation_score(spec.name, task_name)
                    records.append({"alpha": alpha, **score})
                # Highest alpha is the deterministic tie-breaker: it retains the
                # largest fraction of target-task weights among equal val scores.
                best = max(records, key=lambda item: (item["rate"], item["alpha"]))
                selected[family][task] = float(best["alpha"])
            else:
                selected[family][task] = PAPER_ALPHAS[family][task]
            state["alpha_sweep"].setdefault(family, {})[task] = {
                "records": records,
                "selected_alpha": selected[family][task],
                "paper_alpha": PAPER_ALPHAS[family][task],
                "validation_scene": "OOD_MEDIUM (small translation)",
                "tie_break": "highest alpha among equal success rates",
            }
            atomic_json_dump(EXPERIMENT_ROOT / "status.json", state)
    atomic_json_dump(RESULTS_ROOT / "alpha_selection.json", state["alpha_sweep"])
    return selected


def run_main_evaluations(
    *,
    selected: dict[str, dict[str, float]],
    gpu: int,
    port: int,
    state: dict,
    include_generalist: bool,
) -> None:
    main_flags = ("--do_id", "--do_ood_medium", "--do_ood_hard")
    pretrain_spec = PolicySpec(
        name="pretrain_117task",
        config="retain_repro_pretrain",
        checkpoint_dirs=(PRETRAIN,),
    )
    evaluate_policy(
        pretrain_spec,
        target_flags=[(task_name, main_flags) for task_name in TASKS.values()],
        include_generalist=include_generalist,
        gpu=gpu,
        port=port,
        state=state,
    )

    for family in ("taskft", "coft"):
        for task, task_name in TASKS.items():
            config_token = "task_ft" if family == "taskft" else family
            raw = PolicySpec(
                name=f"{family}_{task}",
                config=f"retain_repro_{config_token}_{task}",
                checkpoint_dirs=(checkpoint_for(family, task),),
            )
            evaluate_policy(
                raw,
                target_flags=[(task_name, main_flags)],
                include_generalist=include_generalist,
                gpu=gpu,
                port=port,
                state=state,
            )

            alphas = {selected[family][task], PAPER_ALPHAS[family][task]}
            for alpha in sorted(alphas):
                evaluate_policy(
                    merged_spec(family, task, alpha),
                    target_flags=[(task_name, main_flags)],
                    include_generalist=include_generalist,
                    gpu=gpu,
                    port=port,
                    state=state,
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--skip-alpha-sweep", action="store_true")
    parser.add_argument("--skip-generalist", action="store_true")
    args = parser.parse_args()

    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    lock_stream = (EXPERIMENT_ROOT / "evaluation-pipeline.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("已有评测 pipeline 正在运行，本进程退出。", file=sys.stderr)
        raise SystemExit(2) from None

    wait_for_checkpoints(args.poll_seconds)
    gpu = wait_for_idle_gpu(args.poll_seconds)
    state = load_state()
    state.update({"status": "running", "gpu_physical_index": gpu, "updated_at": now()})
    atomic_json_dump(EXPERIMENT_ROOT / "status.json", state)

    selected = run_alpha_sweep(
        gpu=gpu,
        port=args.port,
        state=state,
        include_sweep=not args.skip_alpha_sweep,
    )
    run_main_evaluations(
        selected=selected,
        gpu=gpu,
        port=args.port,
        state=state,
        include_generalist=not args.skip_generalist,
    )
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "reproduction" / "scripts" / "summarize_results.py"),
            "--results-root",
            str(RESULTS_ROOT),
            "--output-dir",
            str(PROJECT_ROOT / "reproduction" / "results"),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    state.update({"status": "completed", "completed_at": now()})
    atomic_json_dump(EXPERIMENT_ROOT / "status.json", state)
    print(f"{now()} 全部评测完成，结果目录: {RESULTS_ROOT}", flush=True)


if __name__ == "__main__":
    main()
