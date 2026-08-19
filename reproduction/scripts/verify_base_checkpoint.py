#!/usr/bin/env python3
# ruff: noqa: RUF001, UP017
"""实际恢复 π0 base Orbax 参数，并输出结构级审计清单。"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import flax.traverse_util
import numpy as np

from openpi.models import model as _model


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
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    params_path = args.params.resolve()
    if not params_path.is_dir():
        raise FileNotFoundError(params_path)
    params = _model.restore_params(params_path, restore_type=np.ndarray)
    flat = flax.traverse_util.flatten_dict(params, sep="/")
    if not flat:
        raise RuntimeError("Orbax 恢复结果为空")

    dtype_elements: dict[str, int] = {}
    parameter_count = 0
    in_memory_bytes = 0
    signature = hashlib.sha256()
    for name, value in sorted(flat.items()):
        array = np.asarray(value)
        elements = int(array.size)
        dtype = str(array.dtype)
        parameter_count += elements
        in_memory_bytes += int(array.nbytes)
        dtype_elements[dtype] = dtype_elements.get(dtype, 0) + elements
        signature.update(
            f"{name}\t{','.join(map(str, array.shape))}\t{dtype}\n".encode()
        )

    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "params_path": str(params_path),
        "restore_type": "numpy.ndarray",
        "leaf_count": len(flat),
        "parameter_count": parameter_count,
        "in_memory_bytes": in_memory_bytes,
        "dtype_element_counts": dict(sorted(dtype_elements.items())),
        "structure_sha256": signature.hexdigest(),
        "status": "verified",
    }
    atomic_json_dump(args.output.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
