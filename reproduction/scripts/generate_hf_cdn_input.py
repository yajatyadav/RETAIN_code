#!/usr/bin/env python3
"""Generate an aria2 input file with resolved Hugging Face download URLs.

This helper is intentionally run on a machine that can reach huggingface.co.  The
resulting public, time-limited CDN URLs can then be consumed by a remote server
whose DNS path to the Hub API is unavailable.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import pathlib
import time
import urllib.parse

import requests


@dataclasses.dataclass(frozen=True)
class ResolvedFile:
    relative_path: str
    size: int
    url: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-root", type=pathlib.Path, required=True)
    parser.add_argument("--repo-id", default="yajatyadav/RETAIN_datasets")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--remote-root", type=pathlib.PurePosixPath, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--expected-files", type=int)
    return parser.parse_args()


def list_payload_files(root: pathlib.Path) -> list[pathlib.Path]:
    payload = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if path.is_file() and not any(part.startswith(".") for part in relative.parts):
            payload.append(path)
    return sorted(payload)


def resolve_one(
    path: pathlib.Path,
    *,
    root: pathlib.Path,
    repo_id: str,
    revision: str,
) -> ResolvedFile:
    relative = path.relative_to(root).as_posix()
    encoded = urllib.parse.quote(relative, safe="/")
    source_url = (
        f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{encoded}"
    )
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            response = requests.head(source_url, allow_redirects=False, timeout=30)
            response.raise_for_status()
            location = response.headers.get("location")
            if response.is_redirect and location:
                resolved = urllib.parse.urljoin(source_url, location)
            elif response.status_code == 200:
                resolved = source_url
            else:
                raise RuntimeError(
                    f"unexpected response {response.status_code} for {relative}"
                )
            return ResolvedFile(relative, path.stat().st_size, resolved)
        except (requests.RequestException, RuntimeError) as exc:
            last_error = exc
            if attempt < 5:
                time.sleep(2**attempt)
    raise RuntimeError(f"failed to resolve {relative}: {last_error}")


def main() -> None:
    args = parse_args()
    root = args.local_root.resolve()
    files = list_payload_files(root)
    if args.expected_files is not None and len(files) != args.expected_files:
        raise SystemExit(
            f"expected {args.expected_files} payload files, found {len(files)}"
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                resolve_one,
                path,
                root=root,
                repo_id=args.repo_id,
                revision=args.revision,
            )
            for path in files
        ]
        resolved = [future.result() for future in futures]
    resolved.sort(key=lambda item: item.relative_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in resolved:
            relative = pathlib.PurePosixPath(item.relative_path)
            handle.write(f"{item.url}\n")
            handle.write(f"  dir={args.remote_root / relative.parent}\n")
            handle.write(f"  out={relative.name}\n")

    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "repo_id": args.repo_id,
            "revision": args.revision,
            "file_count": len(resolved),
            "total_bytes": sum(item.size for item in resolved),
            "files": [
                {"path": item.relative_path, "bytes": item.size} for item in resolved
            ],
        }
        args.manifest.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "file_count": len(resolved),
                "total_bytes": sum(item.size for item in resolved),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
