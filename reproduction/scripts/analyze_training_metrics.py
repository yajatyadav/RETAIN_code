#!/usr/bin/env python3
"""Audit exported training metrics and write a reproducible JSON summary."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
from itertools import pairwise
import json
import math
from pathlib import Path
import statistics

METRIC_FIELDS = ("loss", "grad_norm", "param_norm", "vision_param_norm")


def mean_fields(records: list[dict[str, float | int]]) -> dict[str, float]:
    return {
        field: statistics.fmean(float(record[field]) for record in records)
        for field in METRIC_FIELDS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--interpretation-zh", default="")
    args = parser.parse_args()

    source_bytes = args.input.read_bytes()
    records = [json.loads(line) for line in source_bytes.splitlines() if line.strip()]
    if not records:
        raise ValueError("metrics 输入为空")
    records.sort(key=lambda record: int(record["step"]))
    steps = [int(record["step"]) for record in records]
    if len(steps) != len(set(steps)):
        raise ValueError("metrics 中存在重复 step")
    missing_fields = [
        (record["step"], field)
        for record in records
        for field in METRIC_FIELDS
        if field not in record
    ]
    if missing_fields:
        raise ValueError(f"metrics 缺字段: {missing_fields[:10]}")
    all_finite = all(
        math.isfinite(float(record[field]))
        for record in records
        for field in METRIC_FIELDS
    )
    if not all_finite:
        raise ValueError("metrics 含 NaN 或 Inf")

    intervals = [right - left for left, right in pairwise(steps)]
    expected_interval = intervals[0] if intervals else None
    interval_consistent = not intervals or all(
        interval == expected_interval for interval in intervals
    )
    first_window = records[: min(10, len(records))]
    last_window = records[-min(10, len(records)) :]

    step_bins = []
    for lower in range(0, args.total_steps, 500):
        selected = [record for record in records if lower <= int(record["step"]) < lower + 500]
        if not selected:
            continue
        means = mean_fields(selected)
        step_bins.append(
            {
                "steps": [int(selected[0]["step"]), int(selected[-1]["step"])],
                "count": len(selected),
                **{f"{field}_mean": means[field] for field in METRIC_FIELDS},
            }
        )

    overall = {}
    for field in METRIC_FIELDS:
        values = [float(record[field]) for record in records]
        summary = {
            "first": values[0],
            "last": values[-1],
            "min": min(values),
            "max": max(values),
            "mean": statistics.fmean(values),
        }
        if field == "loss":
            summary["first_to_last_drop_fraction"] = (values[0] - values[-1]) / values[0]
        overall[field] = summary

    report = {
        "protocol_id": args.protocol_id,
        "config": args.config,
        "experiment_name": args.experiment_name,
        "attempt": args.attempt,
        "analyzed_at": dt.datetime.now(dt.UTC).isoformat(),
        "source": {
            "path": str(args.input),
            "line_count": len(records),
            "bytes": len(source_bytes),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
        },
        "coverage": {
            "first_step": steps[0],
            "last_step": steps[-1],
            "record_count": len(records),
            "expected_interval_steps": expected_interval,
            "interval_consistent": interval_consistent,
            "training_fraction_through_last_record": steps[-1] / args.total_steps,
        },
        "finite_value_audit": {
            "fields": list(METRIC_FIELDS),
            "all_finite": all_finite,
        },
        "overall": overall,
        "window_means": {
            "first_10_records": mean_fields(first_window),
            "last_10_records": mean_fields(last_window),
        },
        "step_bins": step_bins,
        "interpretation_zh": args.interpretation_zh,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
