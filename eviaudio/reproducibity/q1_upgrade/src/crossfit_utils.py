"""Deterministic helpers for class-conditional cross-fitting."""

from __future__ import annotations

import hashlib
from pathlib import Path


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


def single_scale_name(score_source: str, seconds: float) -> str:
    if score_source not in {"clap", "qcr"}:
        raise ValueError(f"unknown score source: {score_source}")
    text = f"{float(seconds):g}".replace(".", "p")
    return f"{score_source}_{text}s"
