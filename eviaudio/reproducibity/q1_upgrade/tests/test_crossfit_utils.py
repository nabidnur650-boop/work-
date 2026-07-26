from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crossfit_utils import single_scale_name, video_fold  # noqa: E402


def test_video_fold_is_stable_and_bounded() -> None:
    observed = [video_fold(f"video_{index}") for index in range(100)]
    assert observed == [video_fold(f"video_{index}") for index in range(100)]
    assert min(observed) == 0
    assert max(observed) == 4
    assert len(set(observed)) == 5


def test_video_fold_rejects_one_fold() -> None:
    with pytest.raises(ValueError):
        video_fold("video", folds=1)


def test_single_scale_names_are_unambiguous() -> None:
    assert single_scale_name("clap", 0.5) == "clap_0p5s"
    assert single_scale_name("qcr", 2.0) == "qcr_2s"
    with pytest.raises(ValueError):
        single_scale_name("other", 1.0)
