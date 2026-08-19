#!/usr/bin/env python3
# ruff: noqa: RUF001, UP017
"""汇总 RETAIN LIBERO 的逐任务 JSON 结果。"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def category(eval_type: str) -> str:
    if eval_type == "ID":
        return "ID"
    if eval_type == "OOD_MEDIUM":
        return "OOD validation"
    if eval_type.startswith("OOD_HARD"):
        return "OOD test"
    if eval_type == "GENERALIST":
        return "Generalist"
    return eval_type


def collect(results_root: Path) -> list[dict]:
    rollout_root = results_root / "rollouts"
    records = []
    for summary_path in sorted(rollout_root.rglob("summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        relative = summary_path.relative_to(rollout_root)
        policy = relative.parts[0]
        episodes_path = summary_path.with_name("episodes.json")
        error_episodes = 0
        if episodes_path.is_file():
            episodes = json.loads(episodes_path.read_text(encoding="utf-8"))
            error_episodes = sum(item.get("error") is not None for item in episodes)
        records.append(
            {
                "policy": policy,
                "eval_type": payload["eval_type"],
                "category": category(payload["eval_type"]),
                "task_suite": payload["task_suite"],
                "task_name": payload["task_name"],
                "seed": payload["seed"],
                "episodes": int(payload["total_episodes"]),
                "successes": int(payload["total_successes"]),
                "success_rate": float(payload["success_rate"]),
                "error_episodes": error_episodes,
                "source": str(summary_path),
            }
        )
    return records


def aggregate(records: list[dict], keys: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, dict] = {}
    for record in records:
        group_key = tuple(record[key] for key in keys)
        item = groups.setdefault(
            group_key,
            {
                **dict(zip(keys, group_key, strict=True)),
                "summaries": 0,
                "episodes": 0,
                "successes": 0,
                "error_episodes": 0,
            },
        )
        item["summaries"] += 1
        item["episodes"] += record["episodes"]
        item["successes"] += record["successes"]
        item["error_episodes"] += record["error_episodes"]
    output = []
    for item in groups.values():
        item["success_rate"] = item["successes"] / item["episodes"]
        output.append(item)
    return sorted(output, key=lambda item: tuple(str(item[key]) for key in keys))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        atomic_text(path, "")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def markdown_report(overall: list[dict], records: list[dict], results_root: Path) -> str:
    lookup = {(item["policy"], item["category"]): item for item in overall}
    policies = sorted({item["policy"] for item in overall})
    categories = ("ID", "OOD validation", "OOD test", "Generalist")

    lines = [
        "# RETAIN LIBERO 复现结果汇总",
        "",
        f"生成时间：{dt.datetime.now(dt.timezone.utc).astimezone().isoformat()}",
        "",
        f"原始结果目录：`{results_root}`",
        "",
        "| Policy | ID | OOD validation | OOD test | Generalist |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy in policies:
        cells = []
        for eval_category in categories:
            item = lookup.get((policy, eval_category))
            if item is None:
                cells.append("—")
            else:
                cells.append(
                    f"{item['success_rate'] * 100:.1f}% "
                    f"({item['successes']}/{item['episodes']})"
                )
        lines.append(f"| {policy} | {' | '.join(cells)} |")

    total_errors = sum(record["error_episodes"] for record in records)
    lines.extend(
        [
            "",
            f"共读取 {len(records)} 个 task/seed summaries；异常 episode 数：{total_errors}。",
            "",
            "OOD validation 对应 small translation；OOD test 合并两个未用于调参的 hard sets。",
            "详细逐任务结果见 `aggregate_by_condition.csv`，逐 episode 记录与视频见原始结果目录。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    output_dir = args.output_dir.resolve()
    records = collect(results_root)
    if not records:
        raise RuntimeError(f"没有找到 summary.json: {results_root / 'rollouts'}")

    by_condition = aggregate(
        records,
        ("policy", "eval_type", "task_suite", "task_name"),
    )
    overall = aggregate(records, ("policy", "category"))
    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "results_root": str(results_root),
        "summary_count": len(records),
        "overall": overall,
        "by_condition": by_condition,
        "raw_summaries": records,
    }
    atomic_text(
        output_dir / "aggregate.json",
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )
    write_csv(output_dir / "aggregate_by_condition.csv", by_condition)
    write_csv(output_dir / "aggregate_overall.csv", overall)
    atomic_text(output_dir / "summary.md", markdown_report(overall, records, results_root))
    print(f"已汇总 {len(records)} 个 summaries 到 {output_dir}")


if __name__ == "__main__":
    main()
