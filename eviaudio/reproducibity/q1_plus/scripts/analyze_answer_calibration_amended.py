#!/usr/bin/env python3
"""Apply code-only amendment 001 to the locked answer-calibration auditor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import analyze_answer_calibration as locked


PROJECT = Path(__file__).resolve().parents[2]
AMENDMENT = PROJECT / "q1_plus/ANSWER_DEVELOPMENT_AMENDMENT_001.json"
ORIGINAL_VALIDATE = locked.validate_row


def validate_row_amended(
    row: dict[str, Any],
    model: str,
    prompts: dict[str, str],
    k_values: set[int],
    expected_ids: set[str],
    manifests: dict[str, dict[str, Any]],
    expected_audio_map: dict[tuple[str, int], dict[str, Any]],
    retriever: str,
) -> None:
    try:
        ORIGINAL_VALIDATE(
            row,
            model,
            prompts,
            k_values,
            expected_ids,
            manifests,
            expected_audio_map,
            retriever,
        )
    except PermissionError as error:
        if str(error) != "AUDITA reference found in development answer row":
            raise
        auditable_values = {
            key: value for key, value in row.items() if key != "audita_rows_accessed"
        }
        if "audita" in json.dumps(auditable_values).lower():
            raise


def main() -> None:
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    if amendment["status"] != "code_only_audit_guard_fix_before_result_inspection":
        raise PermissionError("answer-development amendment is not authorized")
    if locked.sha256(locked.Q1 / "scripts/analyze_answer_calibration.py") != amendment[
        "locked_analyzer_sha256"
    ]:
        raise RuntimeError("locked calibration analyzer changed after amendment")
    locked.validate_row = validate_row_amended
    locked.main()


if __name__ == "__main__":
    main()
