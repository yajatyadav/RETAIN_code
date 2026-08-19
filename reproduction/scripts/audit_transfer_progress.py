#!/usr/bin/env python3
# ruff: noqa: RUF001, UP017
"""按固定 SHA 清单审计服务器传输进度，不读取大文件内容。"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path


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
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    incomplete = []
    exact_files = 0
    exact_bytes = 0
    by_dataset: dict[str, dict[str, int]] = {}
    for item in reference["files"]:
        relative = item["path"]
        expected = int(item["bytes"])
        target = root / relative
        dataset = relative.split("/", 1)[0]
        group = by_dataset.setdefault(
            dataset,
            {"expected_files": 0, "exact_files": 0, "expected_bytes": 0, "exact_bytes": 0},
        )
        group["expected_files"] += 1
        group["expected_bytes"] += expected
        actual = target.stat().st_size if target.is_file() else None
        if actual == expected:
            exact_files += 1
            exact_bytes += expected
            group["exact_files"] += 1
            group["exact_bytes"] += expected
        else:
            incomplete.append(
                {
                    "path": relative,
                    "status": "missing" if actual is None else "size_mismatch",
                    "expected_bytes": expected,
                    "actual_bytes": actual,
                }
            )

    sidecars = []
    if root.is_dir():
        for path in sorted(root.rglob("*.aria2")):
            sidecars.append(
                {
                    "path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "mtime": dt.datetime.fromtimestamp(
                        path.stat().st_mtime, tz=dt.timezone.utc
                    ).isoformat(),
                }
            )
    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "root": str(root),
        "reference": str(args.reference.resolve()),
        "expected_files": len(reference["files"]),
        "expected_bytes": int(reference["total_bytes"]),
        "exact_files": exact_files,
        "exact_bytes": exact_bytes,
        "incomplete_count": len(incomplete),
        "incomplete_files": incomplete,
        "aria2_sidecar_count": len(sidecars),
        "aria2_sidecars": sidecars,
        "by_dataset": by_dataset,
    }
    atomic_json_dump(args.output.resolve(), payload)
    print(
        f"尺寸审计：{exact_files}/{len(reference['files'])} files，"
        f"{exact_bytes}/{reference['total_bytes']} bytes，"
        f"incomplete={len(incomplete)}，sidecars={len(sidecars)}"
    )


if __name__ == "__main__":
    main()
