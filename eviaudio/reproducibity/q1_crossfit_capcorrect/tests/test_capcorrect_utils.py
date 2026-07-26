from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from capcorrect_utils import (  # noqa: E402
    candidate_name,
    canonical_rows_sha256,
    cap_video_rows,
    video_fold,
)


def row(score: float, label: int, start: float, video: str = "v") -> dict:
    return {
        "video_id": video,
        "label_id": label,
        "start_sec": start,
        "end_sec": start + 1.0,
        "score": score,
    }


def test_post_route_cap_uses_frozen_total_order() -> None:
    rows = [
        row(0.7, 2, 1.0),
        row(0.9, 3, 2.0),
        row(0.9, 1, 4.0),
        row(0.9, 1, 3.0),
    ]
    selected = cap_video_rows(rows, 3)
    assert [(item["label_id"], item["start_sec"]) for item in selected] == [
        (1, 3.0),
        (1, 4.0),
        (3, 2.0),
    ]


def test_post_route_cap_rejects_mixed_videos_and_invalid_limit() -> None:
    with pytest.raises(ValueError):
        cap_video_rows([row(0.1, 1, 0.0, "a"), row(0.2, 1, 0.0, "b")], 2)
    with pytest.raises(ValueError):
        cap_video_rows([], 0)


def test_fold_and_candidate_names_are_stable() -> None:
    folds = [video_fold(f"video-{index}") for index in range(100)]
    assert folds == [video_fold(f"video-{index}") for index in range(100)]
    assert set(folds) == set(range(5))
    assert candidate_name("clap", 0.5) == "clap_0p5s"
    assert candidate_name("qcr", None) == "qcr_multiscale"
    with pytest.raises(ValueError):
        candidate_name("bad", 1.0)


def test_canonical_digest_is_order_sensitive_and_stable() -> None:
    rows = [row(0.2, 1, 0.0), row(0.1, 2, 1.0)]
    assert canonical_rows_sha256(rows) == canonical_rows_sha256(list(rows))
    assert canonical_rows_sha256(rows) != canonical_rows_sha256(reversed(rows))
