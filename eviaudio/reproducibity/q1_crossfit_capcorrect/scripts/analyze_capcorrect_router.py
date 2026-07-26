#!/usr/bin/env python3
"""Select candidate pools out of fold, merge classes, then enforce video cap."""

from __future__ import annotations

import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
TRACK = PROJECT / "q1_crossfit_capcorrect"
ORIGINAL_LOCK = TRACK / "CAPCORRECT_ANALYSIS_LOCK.json"
AMENDMENT_LOCK = TRACK / "CAPCORRECT_ANALYZER_AMENDMENT_LOCK.json"
CONFIG_PATH = TRACK / "configs/cap_corrected_crossfit.json"
POOL_RECEIPT = TRACK / "results/candidate_pools/receipt.json"
POOLS = TRACK / "results/candidate_pools/pools"
NATURAL = PROJECT / "q1_top_tier"
NATURAL_REPORT = (
    NATURAL / "results/perception_test/evaluation/perception_test_report.json"
)
OUTPUT = TRACK / "results/capcorrect_router"
REPORT = OUTPUT / "capcorrect_router_report.json"
sys.path.insert(0, str(TRACK / "src"))
sys.path.insert(0, str(NATURAL / "scripts"))
sys.path.insert(0, str(NATURAL / "src"))
sys.path.insert(0, str(PROJECT / "src"))

import evaluate_perception_test as frozen  # noqa: E402
from capcorrect_utils import (  # noqa: E402
    candidate_name,
    cap_video_rows,
    sha256,
    video_fold,
)
from temporal_localization import (  # noqa: E402
    detection_average_precision,
    evaluate_map,
)


def project_path(relative: str) -> Path:
    path = (PROJECT / relative).resolve()
    if not path.is_relative_to(PROJECT):
        raise RuntimeError(f"path leaves project: {relative}")
    return path


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def audit_inputs() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
]:
    lock = json.loads(ORIGINAL_LOCK.read_text(encoding="utf-8"))
    if lock["status"] != (
        "cap_corrected_crossfit_frozen_after_capped_candidate_exposure_"
        "before_uncapped_pool_scoring"
    ):
        raise PermissionError("cap-correct lock is invalid")
    superseded = (
        "q1_crossfit_capcorrect/scripts/analyze_capcorrect_router.py"
    )
    for relative, expected in lock["files"].items():
        if relative == superseded:
            continue
        if sha256(project_path(relative)) != expected:
            raise RuntimeError(f"cap-correct lock mismatch: {relative}")
    amendment = json.loads(AMENDMENT_LOCK.read_text(encoding="utf-8"))
    if (
        amendment["status"]
        != "capcorrect_analyzer_efficiency_amendment_frozen"
        or amendment["numerical_design_changed"]
        or amendment["original_lock_sha256"] != sha256(ORIGINAL_LOCK)
    ):
        raise RuntimeError("cap-correct analyzer amendment is invalid")
    for relative, expected in amendment["files"].items():
        if sha256(project_path(relative)) != expected:
            raise RuntimeError(f"analyzer amendment mismatch: {relative}")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(POOL_RECEIPT.read_text(encoding="utf-8"))
    natural_report = json.loads(NATURAL_REPORT.read_text(encoding="utf-8"))
    if (
        receipt["status"] != "cap_correct_candidate_pools_complete"
        or receipt["capcorrect_analysis_lock_sha256"]
        != sha256(ORIGINAL_LOCK)
        or receipt["config_sha256"] != sha256(CONFIG_PATH)
    ):
        raise RuntimeError("candidate-pool receipt is invalid")
    _, natural_config, index = frozen.audit_inputs()
    return lock, config, receipt, natural_report, index


def candidate_paths(
    config: dict[str, Any], receipt: dict[str, Any]
) -> dict[str, Path]:
    paths = {
        candidate_name(source, seconds): POOLS
        / f"{candidate_name(source, seconds)}.jsonl.gz"
        for source in config["score_sources"]
        for seconds in (*config["candidate_scales_seconds"], None)
    }
    if len(paths) != 10:
        raise RuntimeError("candidate pool does not contain ten methods")
    for name, path in paths.items():
        if sha256(path) != receipt["pool_sha256"][name]:
            raise RuntimeError(f"candidate-pool checksum mismatch: {name}")
    return dict(sorted(paths.items()))


def ground_truth(
    natural_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[int]]:
    annotations = json.loads(
        project_path(natural_config["benchmark"]["annotation_file"]).read_text(
            encoding="utf-8"
        )
    )
    labels = [int(value) for value in natural_config["benchmark"]["label_ids"]]
    allowed = set(labels)
    rows: list[dict[str, Any]] = []
    for video_id, record in annotations.items():
        for event in record[natural_config["benchmark"]["task"]]:
            label_id = int(event["label_id"])
            if label_id not in allowed:
                continue
            rows.append(
                {
                    "video_id": str(video_id),
                    "label_id": label_id,
                    "start_sec": float(event["timestamps"][0]) / 1e6,
                    "end_sec": float(event["timestamps"][1]) / 1e6,
                }
            )
    return rows, labels


def select_candidates(
    *,
    paths: dict[str, Path],
    truth: list[dict[str, Any]],
    labels: list[int],
    folds: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    truth_groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in truth:
        truth_groups[
            (video_fold(str(row["video_id"]), folds), int(row["label_id"]))
        ].append(row)
    scores: dict[str, dict[str, dict[str, float]]] = {
        str(fold): {str(label): {} for label in labels}
        for fold in range(folds)
    }
    row_counts: dict[str, int] = {}
    for candidate, path in paths.items():
        groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        count = 0
        for row in iter_jsonl(path):
            groups[
                (
                    video_fold(str(row["video_id"]), folds),
                    int(row["label_id"]),
                )
            ].append(row)
            count += 1
        row_counts[candidate] = count
        for fold in range(folds):
            for label in labels:
                train_truth = [
                    row
                    for other in range(folds)
                    if other != fold
                    for row in truth_groups[(other, label)]
                ]
                train_predictions = [
                    row
                    for other in range(folds)
                    if other != fold
                    for row in groups[(other, label)]
                ]
                values = detection_average_precision(
                    train_truth, train_predictions
                )
                scores[str(fold)][str(label)][candidate] = float(
                    np.mean(values)
                )
        del groups
        print(
            json.dumps(
                {
                    "selection_candidate_complete": candidate,
                    "pool_rows": count,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    selection: dict[str, dict[str, Any]] = {}
    for fold in range(folds):
        selection[str(fold)] = {}
        for label in labels:
            values = scores[str(fold)][str(label)]
            all_best = sorted(values, key=lambda name: (-values[name], name))[0]
            clap_names = [name for name in values if name.startswith("clap_")]
            clap_best = sorted(
                clap_names, key=lambda name: (-values[name], name)
            )[0]
            selection[str(fold)][str(label)] = {
                "all_candidates": all_best,
                "clap_only": clap_best,
                "training_ap": values,
            }
    return selection, row_counts


def assemble_capped_predictions(
    *,
    paths: dict[str, Path],
    selection: dict[str, dict[str, Any]],
    folds: int,
    route: str,
    video_ids: list[str],
    maximum: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate, path in paths.items():
        for row in iter_jsonl(path):
            fold = video_fold(str(row["video_id"]), folds)
            chosen = selection[str(fold)][str(int(row["label_id"]))][route]
            if chosen == candidate:
                by_video[str(row["video_id"])].append(row)
    output: list[dict[str, Any]] = []
    pre_cap_rows = 0
    discarded_rows = 0
    videos_exceeding_cap = 0
    maximum_pre_cap = 0
    for video_id in video_ids:
        local = by_video.get(video_id, [])
        pre_cap_rows += len(local)
        maximum_pre_cap = max(maximum_pre_cap, len(local))
        if len(local) > maximum:
            videos_exceeding_cap += 1
        capped = cap_video_rows(local, maximum)
        discarded_rows += len(local) - len(capped)
        output.extend(capped)
    final_counts = Counter(str(row["video_id"]) for row in output)
    if max(final_counts.values(), default=0) > maximum:
        raise RuntimeError("post-route video cap failed")
    return output, {
        "pre_cap_rows": pre_cap_rows,
        "final_rows": len(output),
        "discarded_rows": discarded_rows,
        "videos_exceeding_cap": videos_exceeding_cap,
        "maximum_pre_cap_rows_per_video": maximum_pre_cap,
        "maximum_final_rows_per_video": maximum,
    }


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    partial = path.with_name(f"{path.name}.partial")
    with gzip.open(partial, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            )
    partial.replace(path)


def route_summary(
    selection: dict[str, dict[str, Any]],
    *,
    route: str,
    labels: list[int],
    folds: int,
) -> dict[str, Any]:
    assignments = [
        str(selection[str(fold)][str(label)][route])
        for fold in range(folds)
        for label in labels
    ]
    return {
        "assignments": len(assignments),
        "candidate_counts": dict(sorted(Counter(assignments).items())),
        "qcr_assignment_fraction": float(
            np.mean([name.startswith("qcr_") for name in assignments])
        ),
        "class_unique_candidate_counts": {
            str(label): len(
                {
                    str(selection[str(fold)][str(label)][route])
                    for fold in range(folds)
                }
            )
            for label in labels
        },
        "classes_unanimous_across_folds": sum(
            len(
                {
                    str(selection[str(fold)][str(label)][route])
                    for fold in range(folds)
                }
            )
            == 1
            for label in labels
        ),
    }


def main() -> None:
    if len(sys.argv) != 1:
        raise PermissionError("cap-correct analyzer accepts no overrides")
    if REPORT.exists():
        raise FileExistsError("cap-correct router report is immutable")
    lock, config, receipt, natural_report, index = audit_inputs()
    paths = candidate_paths(config, receipt)
    _, natural_config, _ = frozen.audit_inputs()
    truth, labels = ground_truth(natural_config)
    folds = int(config["folds"]["count"])
    video_ids = [str(row["video_id"]) for row in index]
    if len(video_ids) != len(set(video_ids)):
        raise RuntimeError("natural-panel video IDs are not unique")
    fold_counts = Counter(video_fold(video_id, folds) for video_id in video_ids)
    maximum = int(config["caps"]["maximum_segments_per_video_after_routing"])

    selection, candidate_pool_rows = select_candidates(
        paths=paths,
        truth=truth,
        labels=labels,
        folds=folds,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    full_path = OUTPUT / "capcorrect_all_candidates.jsonl.gz"
    clap_path = OUTPUT / "capcorrect_clap_only.jsonl.gz"

    full_rows, full_cap = assemble_capped_predictions(
        paths=paths,
        selection=selection,
        folds=folds,
        route="all_candidates",
        video_ids=video_ids,
        maximum=maximum,
    )
    write_predictions(full_path, full_rows)
    full_metrics = evaluate_map(truth, full_rows, label_ids=labels)
    full_video = frozen.per_video_macro_ap(truth, full_rows, video_ids)
    del full_rows

    clap_rows, clap_cap = assemble_capped_predictions(
        paths=paths,
        selection=selection,
        folds=folds,
        route="clap_only",
        video_ids=video_ids,
        maximum=maximum,
    )
    write_predictions(clap_path, clap_rows)
    clap_metrics = evaluate_map(truth, clap_rows, label_ids=labels)
    del clap_rows

    comparator_name = str(config["primary_comparator"])
    comparator_record = receipt["cap_reconstruction"][comparator_name]
    comparator_path = project_path(comparator_record["source_capped_path"])
    if sha256(comparator_path) != comparator_record["source_capped_sha256"]:
        raise RuntimeError("frozen comparator checksum mismatch")
    comparator_rows = load_jsonl(comparator_path)
    comparator_metrics = evaluate_map(truth, comparator_rows, label_ids=labels)
    comparator_video = frozen.per_video_macro_ap(
        truth, comparator_rows, video_ids
    )
    del comparator_rows

    differences = np.asarray(
        [
            full_video[video_id] - comparator_video[video_id]
            for video_id in video_ids
        ],
        dtype=np.float64,
    )
    bootstrap_config = config["bootstrap"]
    bootstrap = frozen.bootstrap_delta(
        differences,
        replicates=int(bootstrap_config["replicates"]),
        seed=int(bootstrap_config["seed"]),
    )
    threshold_delta = np.asarray(
        full_metrics["map_by_threshold"], dtype=np.float64
    ) - np.asarray(comparator_metrics["map_by_threshold"], dtype=np.float64)
    class_delta = {
        str(label): float(full_metrics["class_mean_ap"][str(label)])
        - float(comparator_metrics["class_mean_ap"][str(label)])
        for label in labels
    }
    exact_cap_matches = int(receipt["cap_reconstruction_exact_matches"])
    integrity = {
        "videos": len(video_ids),
        "fold_video_counts": {
            str(fold): int(fold_counts[fold]) for fold in range(folds)
        },
        "fold_class_assignments": folds * len(labels),
        "expected_fold_class_assignments": 50,
        "candidate_pool_hash_failures": 0,
        "candidate_cap_reconstruction_exact_matches": exact_cap_matches,
        "router_maximum_segments_per_video": maximum,
        "full_router_rows_at_most_cap": full_cap["final_rows"]
        <= len(video_ids) * maximum,
        "clap_router_rows_at_most_cap": clap_cap["final_rows"]
        <= len(video_ids) * maximum,
        "nonfinite_metrics": int(
            not np.isfinite(
                [
                    float(full_metrics["mean_map"]),
                    float(clap_metrics["mean_map"]),
                    float(comparator_metrics["mean_map"]),
                ]
            ).all()
        ),
        "outcome_exposure_disclosed": True,
        "fresh_confirmatory_claim": False,
        "zero_shot_claim": False,
    }
    criteria_config = config["diagnostic_criteria"]
    criteria = {
        "official_mean_map_delta_above_zero": float(full_metrics["mean_map"])
        - float(comparator_metrics["mean_map"])
        > 0.0,
        "paired_video_bootstrap_lower_above_zero": float(
            bootstrap["interval_95_percentile"][0]
        )
        > 0.0,
        "minimum_positive_tiou_thresholds": int(np.sum(threshold_delta > 0.0))
        >= int(criteria_config["minimum_positive_tiou_thresholds"]),
        "minimum_class_delta": min(class_delta.values())
        >= float(criteria_config["minimum_class_delta"]),
        "candidate_cap_reconstruction": exact_cap_matches
        == int(criteria_config["candidate_cap_reconstruction_matches"]),
        "complete_video_assignment": integrity["fold_class_assignments"]
        == integrity["expected_fold_class_assignments"],
        "integrity": (
            integrity["candidate_pool_hash_failures"] == 0
            and integrity["nonfinite_metrics"] == 0
            and integrity["full_router_rows_at_most_cap"]
            and integrity["clap_router_rows_at_most_cap"]
            and integrity["outcome_exposure_disclosed"]
            and not integrity["fresh_confirmatory_claim"]
            and not integrity["zero_shot_claim"]
        ),
    }
    diagnostic_target_met = all(criteria.values())
    report = {
        "status": (
            "cap_corrected_post_outcome_diagnostic_target_met"
            if diagnostic_target_met
            else "cap_corrected_post_outcome_diagnostic_target_not_met"
        ),
        "claim_type": config["outcome_exposure"]["claim_type"],
        "capcorrect_analysis_lock_sha256": sha256(ORIGINAL_LOCK),
        "analyzer_amendment_lock_sha256": sha256(AMENDMENT_LOCK),
        "config_sha256": sha256(CONFIG_PATH),
        "candidate_pool_receipt_sha256": sha256(POOL_RECEIPT),
        "source_natural_report_sha256": sha256(NATURAL_REPORT),
        "primary": {
            "router": "cap_corrected_class_conditional_all_candidates",
            "comparator": comparator_name,
            "router_metrics": full_metrics,
            "comparator_metrics": comparator_metrics,
            "official_mean_map_delta": float(full_metrics["mean_map"])
            - float(comparator_metrics["mean_map"]),
            "relative_mean_map_gain": float(full_metrics["mean_map"])
            / float(comparator_metrics["mean_map"])
            - 1.0,
            "threshold_delta": threshold_delta.astype(float).tolist(),
            "class_delta": class_delta,
            "paired_video_bootstrap": bootstrap,
        },
        "clap_only_ablation": {
            "metrics": clap_metrics,
            "delta_to_clap_multiscale": float(clap_metrics["mean_map"])
            - float(comparator_metrics["mean_map"]),
        },
        "selection": selection,
        "route_summary": {
            "all_candidates": route_summary(
                selection,
                route="all_candidates",
                labels=labels,
                folds=folds,
            ),
            "clap_only": route_summary(
                selection,
                route="clap_only",
                labels=labels,
                folds=folds,
            ),
        },
        "cap_audit": {
            "candidate_cap_reconstruction_exact_matches": exact_cap_matches,
            "full_router": full_cap,
            "clap_only_router": clap_cap,
        },
        "candidate_pool_rows": candidate_pool_rows,
        "diagnostic_criteria": criteria,
        "all_diagnostic_criteria_met": diagnostic_target_met,
        "integrity": integrity,
        "artifacts": {
            "full_router_predictions": str(full_path.relative_to(PROJECT)),
            "full_router_predictions_sha256": sha256(full_path),
            "full_router_prediction_rows": full_cap["final_rows"],
            "clap_only_predictions": str(clap_path.relative_to(PROJECT)),
            "clap_only_predictions_sha256": sha256(clap_path),
            "clap_only_prediction_rows": clap_cap["final_rows"],
        },
        "interpretation_boundary": (
            "This cap-corrected cross-fitted result estimates low-capacity "
            "in-domain routing on an already exposed validation benchmark. "
            "It does not replace the failed frozen zero-shot QCR result and "
            "is not fresh external confirmation."
        ),
        "lock_status": lock["status"],
        "invalidated_predecessor": "q1_upgrade/INVALIDATION_OUTPUT_CAP.md",
    }
    REPORT.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "official_mean_map_delta": report["primary"][
                    "official_mean_map_delta"
                ],
                "router_mean_map": full_metrics["mean_map"],
                "clap_only_mean_map": clap_metrics["mean_map"],
                "paired_video_bootstrap": bootstrap,
                "route_summary": report["route_summary"],
                "cap_audit": report["cap_audit"],
                "diagnostic_criteria": criteria,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
