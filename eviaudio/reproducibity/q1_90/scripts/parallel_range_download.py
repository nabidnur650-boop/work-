#!/usr/bin/env python3
"""Resumably download one checksum-pinned HTTP artifact with byte ranges."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def part_bounds(index: int, chunk_bytes: int, total_bytes: int) -> tuple[int, int]:
    start = index * chunk_bytes
    return start, min(total_bytes, start + chunk_bytes) - 1


def valid_part(path: Path, expected_bytes: int) -> bool:
    return path.is_file() and path.stat().st_size == expected_bytes


def download_part(
    *,
    index: int,
    url: str,
    parts_dir: Path,
    chunk_bytes: int,
    total_bytes: int,
) -> dict[str, int]:
    start, end = part_bounds(index, chunk_bytes, total_bytes)
    expected = end - start + 1
    destination = parts_dir / f"part_{index:05d}"
    if valid_part(destination, expected):
        return {"index": index, "bytes": expected, "reused": 1}
    temporary = destination.with_name(f"{destination.name}.tmp-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    command = [
        "curl",
        "-L",
        "--fail",
        "--silent",
        "--show-error",
        "--retry",
        "8",
        "--retry-all-errors",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "30",
        "--max-time",
        "600",
        "--range",
        f"{start}-{end}",
        "--output",
        str(temporary),
        "--write-out",
        "%{http_code}",
        url,
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    if completed.stdout.strip() != "206":
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"server did not honor range {start}-{end}: HTTP {completed.stdout.strip()}"
        )
    if not valid_part(temporary, expected):
        observed = temporary.stat().st_size if temporary.exists() else -1
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"range {start}-{end} has {observed} bytes; expected {expected}"
        )
    temporary.replace(destination)
    return {"index": index, "bytes": expected, "reused": 0}


def seed_prefix(
    target: Path,
    parts_dir: Path,
    *,
    chunk_bytes: int,
    total_bytes: int,
) -> int:
    if not target.is_file() or target.stat().st_size >= total_bytes:
        return 0
    complete_chunks = target.stat().st_size // chunk_bytes
    if complete_chunks <= 0:
        return 0
    seeded = 0
    with target.open("rb") as source:
        for index in range(complete_chunks):
            destination = parts_dir / f"part_{index:05d}"
            if valid_part(destination, chunk_bytes):
                source.seek(chunk_bytes, 1)
                continue
            temporary = destination.with_name(f"{destination.name}.seed-{os.getpid()}")
            with temporary.open("wb") as handle:
                remaining = chunk_bytes
                while remaining:
                    block = source.read(min(1024 * 1024, remaining))
                    if not block:
                        raise RuntimeError("partial prefix ended during seeding")
                    handle.write(block)
                    remaining -= len(block)
            temporary.replace(destination)
            seeded += 1
    return seeded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--total-bytes", type=int, required=True)
    parser.add_argument("--expected-md5", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunk-mib", type=int, default=16)
    args = parser.parse_args()
    if args.total_bytes <= 0 or args.workers <= 0 or args.chunk_mib <= 0:
        raise ValueError("size, workers, and chunk size must be positive")
    args.target.parent.mkdir(parents=True, exist_ok=True)
    if args.target.is_file() and args.target.stat().st_size == args.total_bytes:
        observed = md5(args.target)
        if observed == args.expected_md5:
            print(json.dumps({"status": "already_verified", "md5": observed}))
            return
        invalid = args.target.with_name(f"{args.target.name}.invalid-{observed}")
        if invalid.exists():
            raise FileExistsError(f"invalid recovery path already exists: {invalid}")
        args.target.replace(invalid)

    chunk_bytes = args.chunk_mib * 1024 * 1024
    count = (args.total_bytes + chunk_bytes - 1) // chunk_bytes
    parts_dir = args.target.with_name(f"{args.target.name}.parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    seeded = seed_prefix(
        args.target,
        parts_dir,
        chunk_bytes=chunk_bytes,
        total_bytes=args.total_bytes,
    )
    missing = []
    for index in range(count):
        start, end = part_bounds(index, chunk_bytes, args.total_bytes)
        if not valid_part(parts_dir / f"part_{index:05d}", end - start + 1):
            missing.append(index)
    print(
        json.dumps(
            {
                "status": "downloading_ranges",
                "total_bytes": args.total_bytes,
                "parts": count,
                "missing_parts": len(missing),
                "seeded_parts": seeded,
                "workers": args.workers,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    completed_count = count - len(missing)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_part,
                index=index,
                url=args.url,
                parts_dir=parts_dir,
                chunk_bytes=chunk_bytes,
                total_bytes=args.total_bytes,
            ): index
            for index in missing
        }
        for future in concurrent.futures.as_completed(futures):
            future.result()
            completed_count += 1
            if completed_count % 10 == 0 or completed_count == count:
                print(
                    json.dumps(
                        {
                            "completed_parts": completed_count,
                            "total_parts": count,
                            "percent": round(100.0 * completed_count / count, 2),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    assembled = args.target.with_name(f"{args.target.name}.assembled")
    digest = hashlib.md5(usedforsecurity=False)
    with assembled.open("wb") as output:
        for index in range(count):
            part = parts_dir / f"part_{index:05d}"
            start, end = part_bounds(index, chunk_bytes, args.total_bytes)
            if not valid_part(part, end - start + 1):
                raise RuntimeError(f"missing or invalid part {index}")
            with part.open("rb") as source:
                while block := source.read(1024 * 1024):
                    digest.update(block)
                    output.write(block)
    observed = digest.hexdigest()
    if assembled.stat().st_size != args.total_bytes or observed != args.expected_md5:
        raise RuntimeError(
            f"assembled artifact failed integrity: bytes={assembled.stat().st_size}, md5={observed}"
        )
    assembled.replace(args.target)
    shutil.rmtree(parts_dir)
    print(
        json.dumps(
            {
                "status": "download_verified",
                "bytes": args.target.stat().st_size,
                "md5": observed,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
