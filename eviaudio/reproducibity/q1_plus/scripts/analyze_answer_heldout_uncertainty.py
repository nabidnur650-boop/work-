#!/usr/bin/env python3
"""Compute the preregistered non-gating held-out uncertainty supplement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import analyze_answer_heldout as main_analysis
import run_answer_calibration as base


PROJECT = Path(__file__).resolve().parents[2]
Q1 = PROJECT / "q1_plus"
RESULTS = Q1 / "results/development/answer_generation"
MAIN_REPORT = RESULTS / "heldout_evaluation_report.json"
OUTPUT = RESULTS / "heldout_uncertainty_supplement.json"
PROTOCOL = Q1 / "HELDOUT_UNCERTAINTY_SUPPLEMENT_PROTOCOL.md"
MODELS = ("qwen2_audio", "phi4_multimodal")
REPLICATES = 10_000
SEED = 20_260_722


def sha256(path: Path) -> str:
    return base.sha256(path)


def interval(values: dict[str, float]) -> dict[str, Any]:
    sources = sorted(values)
    array = np.asarray([values[source] for source in sources], dtype=np.float64)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(array), size=(REPLICATES, len(array)))
    boot = array[draws].mean(axis=1)
    return {
        "estimate": float(array.mean()),
        "bootstrap_95_interval": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "sources": len(sources),
        "replicates": REPLICATES,
        "seed": SEED,
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("held-out uncertainty supplement is immutable")
    report = json.loads(MAIN_REPORT.read_text(encoding="utf-8"))
    if report["status"] not in {
        "heldout_answer_gate_pass_audita_pipeline_lock_eligible",
        "heldout_answer_gate_fail_audita_sealed",
    }:
        raise RuntimeError("main held-out report has an invalid status")
    rows_by_model = {
        model: base.read_jsonl(RESULTS / f"heldout__{model}.jsonl")
        for model in MODELS
    }
    systems = list(report["combined_metrics"])
    absolute: dict[str, Any] = {}
    for model in MODELS:
        absolute[model] = {
            system: {
                metric: interval(
                    main_analysis.source_values(
                        rows_by_model[model], system, metric
                    )
                )
                for metric in ("exact", "token_f1")
            }
            for system in systems
        }
    absolute["two_backbone_mean"] = {
        system: {
            metric: interval(
                main_analysis.combined_source_values(
                    rows_by_model, system, metric
                )
            )
            for metric in ("exact", "token_f1")
        }
        for system in systems
    }
    token_f1_comparisons = {}
    for comparator in systems:
        if comparator == "selected_learned_retrieval":
            continue
        token_f1_comparisons[comparator] = {
            "two_backbone_mean": main_analysis.paired_interval(
                main_analysis.combined_source_values(
                    rows_by_model, "selected_learned_retrieval", "token_f1"
                ),
                main_analysis.combined_source_values(
                    rows_by_model, comparator, "token_f1"
                ),
                REPLICATES,
                SEED,
            ),
            "per_backbone": {
                model: main_analysis.paired_interval(
                    main_analysis.source_values(
                        rows_by_model[model],
                        "selected_learned_retrieval",
                        "token_f1",
                    ),
                    main_analysis.source_values(
                        rows_by_model[model], comparator, "token_f1"
                    ),
                    REPLICATES,
                    SEED,
                )
                for model in MODELS
            },
        }
    payload = {
        "status": "heldout_source_clustered_uncertainty_complete_non_gating",
        "main_report_status": report["status"],
        "main_report_sha256": sha256(MAIN_REPORT),
        "protocol_sha256": sha256(PROTOCOL),
        "absolute_intervals": absolute,
        "learned_minus_comparator_token_f1": token_f1_comparisons,
        "resampling_unit": "target_source_id",
        "replicates": REPLICATES,
        "seed": SEED,
        "promotion_gates_changed": False,
        "audita_rows_accessed": 0,
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"]}, indent=2))


if __name__ == "__main__":
    main()
