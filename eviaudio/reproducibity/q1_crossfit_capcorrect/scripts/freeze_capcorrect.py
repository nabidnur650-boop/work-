#!/usr/bin/env python3
"""Freeze the cap-corrected exploratory analysis before pool generation."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
TRACK = PROJECT / "q1_crossfit_capcorrect"
LOCK = TRACK / "CAPCORRECT_ANALYSIS_LOCK.json"
FILES = (
    "q1_crossfit_capcorrect/README.md",
    "q1_crossfit_capcorrect/PROTOCOL.md",
    "q1_crossfit_capcorrect/configs/cap_corrected_crossfit.json",
    "q1_crossfit_capcorrect/scripts/build_candidate_pools.py",
    "q1_crossfit_capcorrect/scripts/analyze_capcorrect_router.py",
    "q1_crossfit_capcorrect/src/capcorrect_utils.py",
    "q1_crossfit_capcorrect/tests/test_capcorrect_utils.py",
    "q1_upgrade/INVALIDATION_OUTPUT_CAP.md",
    "q1_upgrade/CROSSFIT_ANALYSIS_LOCK.json",
    "q1_upgrade/results/scale_candidates/receipt.json",
    "q1_top_tier/configs/perception_test_natural.json",
    "q1_top_tier/PERCEPTION_TEST_FREEZE.json",
    "q1_top_tier/scripts/evaluate_perception_test.py",
    "q1_top_tier/src/temporal_localization.py",
    "q1_top_tier/results/perception_test/evaluation/perception_test_report.json",
    "q1_top_tier/results/perception_test/precompute/precompute_summary.json"
)

import sys

sys.path.insert(0, str(TRACK / "src"))
from capcorrect_utils import sha256  # noqa: E402


def main() -> None:
    if LOCK.exists():
        raise FileExistsError(f"cap-correct lock is immutable: {LOCK}")
    output = TRACK / "results"
    if output.exists() and any(output.rglob("*")):
        raise RuntimeError("cap-correct outputs exist before the lock")
    config = json.loads(
        (TRACK / "configs/cap_corrected_crossfit.json").read_text(
            encoding="utf-8"
        )
    )
    exposure = config["outcome_exposure"]
    if (
        not exposure["frozen_natural_panel_results_known"]
        or not exposure["capped_single_scale_candidates_exist"]
        or not exposure["partial_candidate_metrics_seen"]
        or exposure["router_result_known_before_lock"]
        or exposure["uncapped_candidate_pool_results_known_before_lock"]
        or exposure["fresh_confirmatory_claim_allowed"]
        or exposure["zero_shot_claim_allowed"]
    ):
        raise RuntimeError("cap-correct outcome-exposure disclosure is incomplete")
    payload = {
        "status": (
            "cap_corrected_crossfit_frozen_after_capped_candidate_exposure_"
            "before_uncapped_pool_scoring"
        ),
        "claim_type": exposure["claim_type"],
        "files": {relative: sha256(PROJECT / relative) for relative in FILES},
        "note": (
            "This local lock records the correction before uncapped pool "
            "generation or router scoring; it is not an external timestamp."
        ),
    }
    LOCK.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "files": len(FILES)}))


if __name__ == "__main__":
    main()
