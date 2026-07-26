#!/usr/bin/env python3
"""Lock the post-outcome cross-fit before generating new scale candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
TRACK = PROJECT / "q1_upgrade"
LOCK = TRACK / "CROSSFIT_ANALYSIS_LOCK.json"
FILES = (
    "q1_upgrade/README.md",
    "q1_upgrade/PROTOCOL.md",
    "q1_upgrade/configs/crossfit_router.json",
    "q1_upgrade/scripts/build_scale_candidates.py",
    "q1_upgrade/scripts/analyze_crossfit_router.py",
    "q1_upgrade/src/crossfit_utils.py",
    "q1_upgrade/tests/test_crossfit_utils.py",
    "q1_top_tier/configs/perception_test_natural.json",
    "q1_top_tier/PERCEPTION_TEST_FREEZE.json",
    "q1_top_tier/scripts/evaluate_perception_test.py",
    "q1_top_tier/src/temporal_localization.py",
    "q1_top_tier/results/perception_test/evaluation/perception_test_report.json",
    "q1_top_tier/results/perception_test/precompute/precompute_summary.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if LOCK.exists():
        raise FileExistsError(f"cross-fit analysis lock is immutable: {LOCK}")
    output = TRACK / "results"
    if output.exists() and any(output.rglob("*")):
        raise RuntimeError("cross-fit outputs exist before analysis lock")
    config = json.loads(
        (TRACK / "configs/crossfit_router.json").read_text(encoding="utf-8")
    )
    exposure = config["outcome_exposure"]
    if (
        not exposure["four_frozen_method_results_known"]
        or exposure["new_single_scale_results_known_before_lock"]
        or exposure["fresh_confirmatory_claim_allowed"]
        or exposure["zero_shot_claim_allowed"]
    ):
        raise RuntimeError("outcome-exposure disclosure is incomplete")
    payload = {
        "status": "post_outcome_crossfit_locked_before_new_scale_scoring",
        "claim_type": exposure["claim_type"],
        "files": {relative: sha256(PROJECT / relative) for relative in FILES},
        "note": (
            "This local checksum lock limits researcher degrees of freedom "
            "after it; it is not an external preregistration timestamp."
        ),
    }
    LOCK.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "files": len(FILES)}))


if __name__ == "__main__":
    main()
