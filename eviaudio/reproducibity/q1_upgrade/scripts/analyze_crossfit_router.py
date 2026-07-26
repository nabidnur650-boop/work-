#!/usr/bin/env python3
"""Select candidates out of fold and evaluate the locked cross-fitted router."""

from __future__ import annotations

import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
TRACK = PROJECT / "q1_upgrade"
LOCK = TRACK / "CROSSFIT_ANALYSIS_LOCK.json"
CONFIG_PATH = TRACK / "configs/crossfit_router.json"
NATURAL = PROJECT / "q1_top_tier"
NATURAL_REPORT = (
    NATURAL / "results/perception_test/evaluation/perception_test_report.json"
)
SCALE_RECEIPT = TRACK / "results/scale_candidates/receipt.json"
OUTPUT = TRACK / "results/crossfit_router"
REPORT = OUTPUT / "crossfit_router_report.json"
sys.path.insert(0, str(TRACK / "src"))
sys.path.insert(0, str(NATURAL / "scripts"))
sys.path.insert(0, str(NATURAL / "src"))
sys.path.insert(0, str(PROJECT / "src"))

import evaluate_perception_test as frozen  # noqa: E402
from crossfit_utils import sha256, single_scale_name, video_fold  # noqa: E402
from temporal_localization import (  # noqa: E402
    detection_average_precision,
    evaluate_map,
)


def project_path(relative: str) -> Path:
    path = (PROJECT / relative).resolve()
    if not path.is_relative_to(PROJECT):
        raise RuntimeError(f"path leaves project: {relative}")
    return path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def audit_inputs() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["status"] != "post_outcome_crossfit_locked_before_new_scale_scoring":
        raise PermissionError("cross-fit analysis lock is invalid")
    for relative, expected in lock["files"].items():
        if sha256(project_path(relative)) != expected:
            raise RuntimeError(f"cross-fit lock mismatch: {relative}")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    natural_report = json.loads(NATURAL_REPORT.read_text(encoding="utf-8"))
    scale_receipt = json.loads(SCALE_RECEIPT.read_text(encoding="utf-8"))
    if (
        scale_receipt["status"] != "locked_single_scale_candidates_complete"
        or scale_receipt["crossfit_analysis_lock_sha256"] != sha256(LOCK)
        or scale_receipt["router_config_sha256"] != sha256(CONFIG_PATH)
    ):
        raise RuntimeError("single-scale candidate receipt is invalid")
    _, natural_config, index = frozen.audit_inputs()
    return lock, config, natural_config, natural_report, index


def candidate_paths(
    config: dict[str, Any],
    natural_report: dict[str, Any],
    scale_receipt: dict[str, Any],
) -> dict[str, Path]:
    existing_dir = NATURAL / "results/perception_test/evaluation/predictions"
    generated_dir = TRACK / "results/scale_candidates/predictions"
    paths = {
        "clap_4s": existing_dir / "clap_4s.jsonl.gz",
        "qcr_4s": existing_dir / "qcr_4s.jsonl.gz",
        "clap_multiscale": existing_dir / "clap_multiscale.jsonl.gz",
        "qcr_multiscale": existing_dir / "qcr_multiscale.jsonl.gz",
    }
    for source in config["score_sources"]:
        for seconds in config["candidate_scales_seconds"]:
            if float(seconds) == 4.0:
                continue
            name = single_scale_name(str(source), float(seconds))
            paths[name] = generated_dir / f"{name}.jsonl.gz"
    for name, path in paths.items():
        if name in natural_report["prediction_sha256"]:
            expected = natural_report["prediction_sha256"][name]
        else:
            expected = scale_receipt["prediction_sha256"][name]
        if sha256(path) != expected:
            raise RuntimeError(f"candidate checksum mismatch: {name}")
    if len(paths) != 10:
        raise RuntimeError("candidate set does not contain ten methods")
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
        str(fold): {str(label): {} for label in labels} for fold in range(folds)
    }
    candidate_metrics: dict[str, Any] = {}
    for candidate, path in paths.items():
        predictions = load_jsonl(path)
        groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in predictions:
            groups[
                (
                    video_fold(str(row["video_id"]), folds),
                    int(row["label_id"]),
                )
            ].append(row)
        candidate_metrics[candidate] = evaluate_map(
            truth, predictions, label_ids=labels
        )
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
                values = detection_average_precision(train_truth, train_predictions)
                scores[str(fold)][str(label)][candidate] = float(np.mean(values))
        del predictions, groups
        print(
            json.dumps(
                {
                    "selection_candidate_complete": candidate,
                    "mean_map_full_panel_descriptive": candidate_metrics[candidate][
                        "mean_map"
                    ],
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
            clap_best = sorted(clap_names, key=lambda name: (-values[name], name))[0]
            selection[str(fold)][str(label)] = {
                "all_candidates": all_best,
                "clap_only": clap_best,
                "training_ap": values,
            }
    return selection, candidate_metrics


def assemble_predictions(
    *,
    paths: dict[str, Path],
    selection: dict[str, dict[str, Any]],
    folds: int,
    route: str,
    video_order: dict[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate, path in paths.items():
        for row in iter_jsonl(path):
            fold = video_fold(str(row["video_id"]), folds)
            chosen = selection[str(fold)][str(int(row["label_id"]))][route]
            if chosen == candidate:
                rows.append({**row, "_candidate": candidate})
    rows.sort(
        key=lambda row: (
            video_order[str(row["video_id"])],
            -float(row["score"]),
            int(row["label_id"]),
            float(row["start_sec"]),
            float(row["end_sec"]),
            str(row["_candidate"]),
        )
    )
    return rows


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    partial = path.with_name(f"{path.name}.partial")
    with gzip.open(partial, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {key: value for key, value in row.items() if key != "_candidate"},
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
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
                {str(selection[str(fold)][str(label)][route]) for fold in range(folds)}
            )
            for label in labels
        },
        "classes_unanimous_across_folds": sum(
            len({str(selection[str(fold)][str(label)][route]) for fold in range(folds)})
            == 1
            for label in labels
        ),
    }


def main() -> None:
    if len(sys.argv) != 1:
        raise PermissionError("the locked cross-fit analyzer accepts no overrides")
    if REPORT.exists():
        raise FileExistsError("cross-fit router report is immutable")
    lock, config, natural_config, natural_report, index = audit_inputs()
    scale_receipt = json.loads(SCALE_RECEIPT.read_text(encoding="utf-8"))
    paths = candidate_paths(config, natural_report, scale_receipt)
    truth, labels = ground_truth(natural_config)
    folds = int(config["folds"]["count"])
    video_ids = [str(row["video_id"]) for row in index]
    video_order = {video_id: index for index, video_id in enumerate(video_ids)}
    if len(video_order) != len(video_ids):
        raise RuntimeError("natural-panel video IDs are not unique")
    fold_counts = Counter(video_fold(video_id, folds) for video_id in video_ids)

    selection, candidate_metrics = select_candidates(
        paths=paths, truth=truth, labels=labels, folds=folds
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    full_path = OUTPUT / "crossfit_all_candidates.jsonl.gz"
    clap_path = OUTPUT / "crossfit_clap_only.jsonl.gz"
    full_rows = assemble_predictions(
        paths=paths,
        selection=selection,
        folds=folds,
        route="all_candidates",
        video_order=video_order,
    )
    write_predictions(full_path, full_rows)
    full_metrics = evaluate_map(truth, full_rows, label_ids=labels)
    full_video = frozen.per_video_macro_ap(truth, full_rows, video_ids)
    full_count = len(full_rows)
    del full_rows

    clap_rows = assemble_predictions(
        paths=paths,
        selection=selection,
        folds=folds,
        route="clap_only",
        video_order=video_order,
    )
    write_predictions(clap_path, clap_rows)
    clap_metrics = evaluate_map(truth, clap_rows, label_ids=labels)
    clap_count = len(clap_rows)
    del clap_rows

    comparator_path = paths[str(config["primary_comparator"])]
    comparator_rows = load_jsonl(comparator_path)
    comparator_metrics = evaluate_map(truth, comparator_rows, label_ids=labels)
    comparator_video = frozen.per_video_macro_ap(truth, comparator_rows, video_ids)
    del comparator_rows
    differences = np.asarray(
        [full_video[video_id] - comparator_video[video_id] for video_id in video_ids],
        dtype=np.float64,
    )
    bootstrap_config = config["bootstrap"]
    bootstrap = frozen.bootstrap_delta(
        differences,
        replicates=int(bootstrap_config["replicates"]),
        seed=int(bootstrap_config["seed"]),
    )
    threshold_delta = np.asarray(full_metrics["map_by_threshold"]) - np.asarray(
        comparator_metrics["map_by_threshold"]
    )
    class_delta = {
        str(label): float(full_metrics["class_mean_ap"][str(label)])
        - float(comparator_metrics["class_mean_ap"][str(label)])
        for label in labels
    }
    criteria_config = config["diagnostic_criteria"]
    integrity = {
        "videos": len(video_ids),
        "fold_video_counts": {
            str(fold): int(fold_counts[fold]) for fold in range(folds)
        },
        "fold_class_assignments": folds * len(labels),
        "expected_fold_class_assignments": 50,
        "candidate_hash_failures": 0,
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
        "complete_video_assignment": integrity["fold_class_assignments"]
        == integrity["expected_fold_class_assignments"],
        "integrity": (
            integrity["candidate_hash_failures"] == 0
            and integrity["nonfinite_metrics"] == 0
            and integrity["outcome_exposure_disclosed"]
            and not integrity["fresh_confirmatory_claim"]
            and not integrity["zero_shot_claim"]
        ),
    }
    diagnostic_target_met = all(criteria.values())
    report = {
        "status": (
            "post_outcome_crossfit_diagnostic_target_met"
            if diagnostic_target_met
            else "post_outcome_crossfit_diagnostic_target_not_met"
        ),
        "claim_type": config["outcome_exposure"]["claim_type"],
        "crossfit_analysis_lock_sha256": sha256(LOCK),
        "router_config_sha256": sha256(CONFIG_PATH),
        "scale_candidate_receipt_sha256": sha256(SCALE_RECEIPT),
        "source_natural_report_sha256": sha256(NATURAL_REPORT),
        "primary": {
            "router": "class_conditional_all_candidates",
            "comparator": config["primary_comparator"],
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
        "candidate_full_panel_metrics_descriptive": candidate_metrics,
        "diagnostic_criteria": criteria,
        "all_diagnostic_criteria_met": diagnostic_target_met,
        "integrity": integrity,
        "artifacts": {
            "full_router_predictions": str(full_path.relative_to(PROJECT)),
            "full_router_predictions_sha256": sha256(full_path),
            "full_router_prediction_rows": full_count,
            "clap_only_predictions": str(clap_path.relative_to(PROJECT)),
            "clap_only_predictions_sha256": sha256(clap_path),
            "clap_only_prediction_rows": clap_count,
        },
        "interpretation_boundary": (
            "The cross-fitted result estimates low-capacity in-domain routing "
            "on an already exposed validation benchmark. It does not replace "
            "the failed frozen zero-shot QCR result and is not fresh external "
            "confirmation."
        ),
        "lock_status": lock["status"],
    }
    REPORT.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "primary": report["primary"],
                "clap_only_ablation": report["clap_only_ablation"],
                "route_summary": report["route_summary"],
                "diagnostic_criteria": criteria,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
