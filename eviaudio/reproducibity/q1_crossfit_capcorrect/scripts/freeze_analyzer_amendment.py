#!/usr/bin/env python3
"""Bind the efficiency-only analyzer amendment before router scoring."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
TRACK = PROJECT / "q1_crossfit_capcorrect"
ORIGINAL_LOCK = TRACK / "CAPCORRECT_ANALYSIS_LOCK.json"
AMENDMENT_LOCK = TRACK / "CAPCORRECT_ANALYZER_AMENDMENT_LOCK.json"
FILES = (
    "q1_crossfit_capcorrect/ANALYZER_EFFICIENCY_AMENDMENT.md",
    "q1_crossfit_capcorrect/scripts/analyze_capcorrect_router.py",
    "q1_crossfit_capcorrect/scripts/freeze_analyzer_amendment.py",
    "q1_crossfit_capcorrect/src/capcorrect_utils.py",
    "q1_crossfit_capcorrect/configs/cap_corrected_crossfit.json",
    "q1_crossfit_capcorrect/CAPCORRECT_ANALYSIS_LOCK.json"
)
sys.path.insert(0, str(TRACK / "src"))
from capcorrect_utils import sha256  # noqa: E402


def main() -> None:
    if AMENDMENT_LOCK.exists():
        raise FileExistsError("analyzer amendment lock is immutable")
    router_output = TRACK / "results/capcorrect_router"
    if router_output.exists() and any(router_output.rglob("*")):
        raise RuntimeError("router outputs exist before analyzer amendment")
    original = json.loads(ORIGINAL_LOCK.read_text(encoding="utf-8"))
    if original["status"] != (
        "cap_corrected_crossfit_frozen_after_capped_candidate_exposure_"
        "before_uncapped_pool_scoring"
    ):
        raise RuntimeError("original cap-correct lock is invalid")
    payload = {
        "status": "capcorrect_analyzer_efficiency_amendment_frozen",
        "numerical_design_changed": False,
        "original_lock_sha256": sha256(ORIGINAL_LOCK),
        "files": {relative: sha256(PROJECT / relative) for relative in FILES},
        "note": (
            "The amendment replaces a quadratic cap assertion with an "
            "equivalent linear Counter assertion before router scoring."
        ),
    }
    AMENDMENT_LOCK.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "files": len(FILES)}))


if __name__ == "__main__":
    main()
