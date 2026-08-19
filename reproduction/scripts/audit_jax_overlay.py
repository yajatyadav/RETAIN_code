#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""在不使用 GPU 的情况下审计隔离 JAX overlay 的包与动态库。"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import importlib
from importlib import metadata
import json
import os
from pathlib import Path

CUDA_LIBRARIES = (
    ("cuda_runtime", "libcudart.so.12"),
    ("cuda_nvrtc", "libnvrtc.so.12"),
    ("nvjitlink", "libnvJitLink.so.12"),
    ("cublas", "libcublas.so.12"),
    ("cublas", "libcublasLt.so.12"),
    ("nccl", "libnccl.so.2"),
    ("cuda_cupti", "libcupti.so.12"),
    ("cusparse", "libcusparse.so.12"),
    ("cusolver", "libcusolver.so.11"),
    ("cufft", "libcufft.so.11"),
    ("nvshmem", "libnvshmem_host.so.3"),
    ("cudnn", "libcudnn.so.9"),
)


def now() -> str:
    return dt.datetime.now(dt.UTC).astimezone().isoformat()


def atomic_json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report: dict[str, object] = {
        "checked_at": now(),
        "overlay": str(args.overlay),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "packages": {},
        "libraries": [],
        "status": "failed",
    }
    errors: list[str] = []

    distributions = {
        distribution.metadata["Name"]: distribution.version
        for distribution in metadata.distributions(path=[str(args.overlay)])
        if distribution.metadata["Name"]
    }
    report["packages"] = dict(sorted(distributions.items()))

    libraries: list[dict[str, object]] = []
    for module_name, library_name in CUDA_LIBRARIES:
        library_path = args.overlay / "nvidia" / module_name / "lib" / library_name
        record: dict[str, object] = {
            "module": f"nvidia.{module_name}",
            "library": library_name,
            "path": str(library_path),
            "exists": library_path.is_file(),
            "loaded": False,
        }
        try:
            module = importlib.import_module(f"nvidia.{module_name}")
            record["module_path"] = str(Path(module.__path__[0]).resolve())
            ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
            record["loaded"] = True
        except (ImportError, OSError) as error:
            record["error"] = str(error)
            errors.append(f"{library_name}: {error}")
        libraries.append(record)
    report["libraries"] = libraries

    try:
        import jax
        import jax.numpy as jnp

        source = jnp.arange(64, dtype=jnp.bfloat16).reshape(8, 8)
        converted = jax.jit(lambda value: value.astype(jnp.float16))(source)
        converted.block_until_ready()
        report["cpu_smoke"] = {
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "device": str(jax.devices()[0]),
            "bf16_to_f16_shape": list(converted.shape),
        }
        if jax.default_backend() != "cpu":
            errors.append(f"审计进程意外使用了 {jax.default_backend()} backend")
    except Exception as error:
        report["cpu_smoke_error"] = repr(error)
        errors.append(f"CPU smoke: {error}")

    report["errors"] = errors
    report["status"] = "verified" if not errors else "failed"
    atomic_json_dump(args.output, report)
    if errors:
        raise RuntimeError(f"JAX overlay 审计失败: {errors}")


if __name__ == "__main__":
    main()
