#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, UP017
"""校验 RETAIN 官方 RLDS 数据，并生成可审计 JSON 清单。"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

EXPECTED_DATASETS = (
    "libero_goal_reduced",
    "libero_object_reduced",
    "libero_spatial_reduced",
    "libero_90_flipped",
    "libero_10_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "libero_10_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "libero_10_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-sha256", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    missing = [name for name in EXPECTED_DATASETS if not (root / name / "1.0.0").is_dir()]
    if missing:
        raise FileNotFoundError(f"缺少数据集目录: {missing}")

    datasets = []
    all_files = []
    for name in EXPECTED_DATASETS:
        version_dir = root / name / "1.0.0"
        info_path = version_dir / "dataset_info.json"
        features_path = version_dir / "features.json"
        if not info_path.is_file() or not features_path.is_file():
            raise FileNotFoundError(f"{name} 缺少 dataset_info.json 或 features.json")

        info = json.loads(info_path.read_text(encoding="utf-8"))
        files = sorted(path for path in version_dir.rglob("*") if path.is_file())
        tfrecords = [path for path in files if ".tfrecord-" in path.name]
        if not tfrecords:
            raise RuntimeError(f"{name} 没有 TFRecord shard")

        datasets.append(
            {
                "name": name,
                "version": info.get("version", "1.0.0"),
                "splits": info.get("splits"),
                "file_count": len(files),
                "tfrecord_shards": len(tfrecords),
                "bytes": sum(path.stat().st_size for path in files),
            }
        )
        all_files.extend(files)

    file_records = []
    for index, path in enumerate(sorted(all_files), start=1):
        record = {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
        }
        if not args.skip_sha256:
            record["sha256"] = sha256(path)
        file_records.append(record)
        if index % 25 == 0 or index == len(all_files):
            print(f"已校验 {index}/{len(all_files)} 个文件", flush=True)

    payload = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "root": str(root),
        "dataset_repo": "yajatyadav/RETAIN_datasets",
        "dataset_revision": "d15edfa89167e6e7230be3e85eb7391be2fa3134",
        "dataset_count": len(datasets),
        "file_count": len(file_records),
        "total_bytes": sum(item["bytes"] for item in file_records),
        "sha256_included": not args.skip_sha256,
        "datasets": datasets,
        "files": file_records,
    }
    atomic_json_dump(args.output.resolve(), payload)
    print(
        f"数据校验通过：{payload['dataset_count']} 个数据集，"
        f"{payload['file_count']} 个文件，{payload['total_bytes']} bytes"
    )


if __name__ == "__main__":
    main()
