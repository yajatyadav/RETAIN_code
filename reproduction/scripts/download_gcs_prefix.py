#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
"""并行下载公开 GCS prefix，并按对象 size/MD5 校验。"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request


def fetch_objects(bucket: str, prefix: str) -> list[dict[str, str]]:
    query = urllib.parse.urlencode({"prefix": prefix})
    url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o?{query}"
    with urllib.request.urlopen(url) as response:
        payload = json.load(response)
    objects = payload.get("items", [])
    if not objects:
        raise RuntimeError(f"GCS prefix 为空: gs://{bucket}/{prefix}")
    return objects


def md5_base64(path: Path) -> str:
    digest = hashlib.md5()  # 与 GCS 公开对象元数据比对，不用于安全认证
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return base64.b64encode(digest.digest()).decode("ascii")


def download_with_python(
    bucket: str, prefix: str, output: Path, objects: list[dict[str, str]], jobs: int
) -> None:
    def download_one(item: dict[str, str]) -> None:
        relative = Path(item["name"].removeprefix(prefix))
        target = output / relative
        expected_size = int(item["size"])
        if target.is_file() and target.stat().st_size == expected_size:
            return

        partial = target.with_suffix(target.suffix + ".part")
        encoded_name = urllib.parse.quote(item["name"], safe="/")
        url = f"https://storage.googleapis.com/{bucket}/{encoded_name}"
        for attempt in range(1, 21):
            offset = partial.stat().st_size if partial.exists() else 0
            request = urllib.request.Request(url)
            if offset:
                request.add_header("Range", f"bytes={offset}-")
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    append = offset > 0 and response.status == 206
                    mode = "ab" if append else "wb"
                    with partial.open(mode) as stream:
                        shutil.copyfileobj(response, stream, length=8 * 1024 * 1024)
                if partial.stat().st_size == expected_size:
                    partial.replace(target)
                    print(f"下载完成: {relative}", flush=True)
                    return
                if partial.stat().st_size > expected_size:
                    partial.unlink()
                    raise RuntimeError(f"下载文件大于预期: {relative}")
            except Exception as error:  # 网络下载按对象重试，并保留已完成字节
                print(f"下载重试 {attempt}/20 {relative}: {error}", flush=True)
                if attempt == 20:
                    raise
                time.sleep(min(30, attempt * 2))
        raise RuntimeError(f"下载未完成: {relative}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(download_one, item) for item in objects]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--backend", choices=("auto", "aria2", "python"), default="auto")
    args = parser.parse_args()

    prefix = args.prefix.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    objects = fetch_objects(args.bucket, prefix)

    backend = args.backend
    if backend == "auto":
        backend = "aria2" if shutil.which("aria2c") else "python"

    if backend == "python":
        for item in objects:
            relative = Path(item["name"].removeprefix(prefix))
            (output / relative).parent.mkdir(parents=True, exist_ok=True)
        download_with_python(args.bucket, prefix, output, objects, args.jobs)
    else:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as stream:
            input_path = Path(stream.name)
            for item in objects:
                name = item["name"]
                relative = Path(name.removeprefix(prefix))
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                encoded_name = urllib.parse.quote(name, safe="/")
                stream.write(f"https://storage.googleapis.com/{args.bucket}/{encoded_name}\n")
                stream.write(f"  dir={target.parent}\n")
                stream.write(f"  out={target.name}\n")

        try:
            subprocess.run(
                [
                    "aria2c",
                    f"--input-file={input_path}",
                    "--continue=true",
                    f"--max-concurrent-downloads={args.jobs}",
                    "--max-connection-per-server=8",
                    "--split=8",
                    "--min-split-size=16M",
                    "--file-allocation=none",
                    "--auto-file-renaming=false",
                    "--allow-overwrite=true",
                    "--retry-wait=5",
                    "--max-tries=20",
                    "--summary-interval=30",
                ],
                check=True,
            )
        finally:
            input_path.unlink(missing_ok=True)

    failures = []
    total_bytes = 0
    for index, item in enumerate(objects, start=1):
        relative = Path(item["name"].removeprefix(prefix))
        target = output / relative
        expected_size = int(item["size"])
        total_bytes += expected_size
        if not target.is_file():
            failures.append(f"缺少 {relative}")
            continue
        if target.stat().st_size != expected_size:
            failures.append(
                f"大小不符 {relative}: {target.stat().st_size} != {expected_size}"
            )
            continue
        expected_md5 = item.get("md5Hash")
        if expected_md5 and md5_base64(target) != expected_md5:
            failures.append(f"MD5 不符 {relative}")
        print(f"已校验 {index}/{len(objects)}: {relative}", flush=True)

    if failures:
        raise RuntimeError("GCS 校验失败:\n" + "\n".join(failures))
    print(f"下载校验通过：{len(objects)} 个对象，{total_bytes} bytes，目录 {output}")


if __name__ == "__main__":
    main()
