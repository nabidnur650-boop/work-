"""Deterministic helpers for the cap-corrected class router."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def video_fold(video_id: str, folds: int = 5) -> int:
    if folds < 2:
        raise ValueError("at least two folds are required")
    prefix = hashlib.sha256(video_id.encode("utf-8")).digest()[:8]
    return int.from_bytes(prefix, byteorder="big", signed=False) % folds


def candidate_name(score_source: str, seconds: float | None) -> str:
    if score_source not in {"clap", "qcr"}:
        raise ValueError(f"unknown score source: {score_source}")
    if seconds is None:
        return f"{score_source}_multiscale"
    text = f"{float(seconds):g}".replace(".", "p")
    return f"{score_source}_{text}s"


def prediction_sort_key(row: dict[str, Any]) -> tuple[float, int, float, float]:
    return (
        -float(row["score"]),
        int(row["label_id"]),
        float(row["start_sec"]),
        float(row["end_sec"]),
    )


def cap_video_rows(
    rows: Iterable[dict[str, Any]], maximum: int
) -> list[dict[str, Any]]:
    if maximum < 1:
        raise ValueError("maximum must be positive")
    materialized = list(rows)
    video_ids = {str(row["video_id"]) for row in materialized}
    if len(video_ids) > 1:
        raise ValueError("cap_video_rows received multiple videos")
    return sorted(materialized, key=prediction_sort_key)[:maximum]


def canonical_rows_sha256(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        payload = json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
        digest.update(payload.encode("utf-8"))
    return digest.hexdigest()
