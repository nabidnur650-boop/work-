#!/usr/bin/env python3
"""Audit and aggregate the frozen five-seed exact-onset ranker."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "q1_plus" / "results" / "development" / "event_ranker"
SELECTION_LOCK = PROJECT / "q1_plus" / "EVENT_RANKER_LOCK.json"
BOOTSTRAP_SEED = 20_260_722
BOOTSTRAP_REPLICATES = 10_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT / path


def audit_selection_lock() -> dict[str, Any]:
    lock = json.loads(SELECTION_LOCK.read_text(encoding="utf-8"))
    for relative, expected in lock["files"].items():
        if sha256(PROJECT / relative) != expected:
            raise RuntimeError(f"event-ranker lock mismatch: {relative}")
    if lock["confirmatory_panel_status"] != "sealed":
        raise RuntimeError("exact-onset confirmatory panel is not sealed")
    return lock


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives <= 0:
        raise ValueError("average precision requires positive evidence")
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order].astype(np.float64)
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    return float((precision * sorted_labels).sum() / positives)


def interval_iou(
    start: float, end: float, target_start: float, target_end: float
) -> float:
    intersection = max(0.0, min(end, target_end) - max(start, target_start))
    union = max(end, target_end) - min(start, target_start)
    return float(intersection / union) if union > 0.0 else 0.0


def score_metrics(row: dict[str, Any], scores: np.ndarray) -> dict[str, float]:
    evidence = np.asarray(row["evidence_targets"], dtype=bool)
    starts = np.asarray(row["chunk_start_sec"], dtype=np.float64)
    ends = np.asarray(row["chunk_end_sec"], dtype=np.float64)
    if not (
        len(scores) == len(evidence) == len(starts) == len(ends) == int(row["n_chunks"])
    ):
        raise RuntimeError(f"chunk alignment failure: {row['recipe_id']}")
    if not np.isfinite(scores).all() or not evidence.any() or evidence.all():
        raise RuntimeError(f"invalid scores/evidence: {row['recipe_id']}")
    order = np.argsort(-scores, kind="stable")
    top = int(order[0])
    return {
        "evidence_ap": average_precision(scores, evidence),
        "hit_at_1": float(evidence[top]),
        "recall_at_4": float(evidence[order[:4]].any()),
        "top_chunk_iou": interval_iou(
            float(starts[top]),
            float(ends[top]),
            float(row["target_start_sec"]),
            float(row["target_end_sec"]),
        ),
    }


def mean_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    names = ("evidence_ap", "hit_at_1", "recall_at_4", "top_chunk_iou")
    return {
        name: float(np.mean([row[key][name] for row in rows])) for name in names
    }


def close_metrics(
    observed: dict[str, float], expected: dict[str, float], label: str
) -> None:
    for name, value in observed.items():
        if not math.isclose(value, float(expected[name]), rel_tol=0.0, abs_tol=1e-10):
            raise RuntimeError(
                f"metric reproduction failed for {label}/{name}: "
                f"{value} != {expected[name]}"
            )


def baseline_signature(row: dict[str, Any]) -> str:
    payload = {
        key: row[key]
        for key in (
            "recipe_id",
            "panel",
            "duration_sec",
            "position_bin",
            "snr_db",
            "class_id",
            "class_label",
            "foreground_cluster_id",
            "n_chunks",
            "target_start_sec",
            "target_end_sec",
            "evidence_targets",
            "chunk_start_sec",
            "chunk_end_sec",
            "prior_chunk_scores",
        )
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_seed(
    config_path: Path, config: dict[str, Any], seed: int
) -> dict[str, Any]:
    run_tag = str(config["run_tag"])
    stem = f"{run_tag}__prior_residual__seed_{seed}"
    metrics_path = RESULTS / f"metrics__{stem}.json"
    raw_path = RESULTS / f"raw_validation__{stem}.jsonl.gz"
    history_path = RESULTS / f"history__{stem}.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics["seed"] != seed or metrics["run_tag"] != run_tag:
        raise RuntimeError(f"seed/run tag mismatch: {metrics_path}")
    if metrics["config_sha256"] != sha256(config_path):
        raise RuntimeError(f"config checksum mismatch: {metrics_path}")
    if metrics["selection_lock_sha256"] != sha256(SELECTION_LOCK):
        raise RuntimeError(f"selection lock mismatch: {metrics_path}")
    if metrics["confirmatory_panel_status"] != "sealed":
        raise RuntimeError(f"confirmatory status mismatch: {metrics_path}")
    if metrics["raw_validation_sha256"] != sha256(raw_path):
        raise RuntimeError(f"raw checksum mismatch: {raw_path}")
    if metrics["checkpoint_sha256"] != sha256(Path(metrics["checkpoint"])):
        raise RuntimeError(f"checkpoint checksum mismatch: seed {seed}")
    if metrics["history_sha256"] != sha256(history_path):
        raise RuntimeError(f"history checksum mismatch: seed {seed}")
    development_index = resolve(config["development_index"])
    validation_index = resolve(config["validation_index"])
    if metrics["development_index_sha256"] != sha256(development_index):
        raise RuntimeError("development index changed")
    if metrics["validation_index_sha256"] != sha256(validation_index):
        raise RuntimeError("validation index changed")

    rows: dict[str, dict[str, Any]] = {}
    with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            recipe_id = str(row["recipe_id"])
            if recipe_id in rows:
                raise RuntimeError(f"duplicate recipe: {recipe_id}")
            if "confirm" in str(row["panel"]).lower():
                raise PermissionError("confirmatory row exposed to development analyzer")
            model_scores = np.asarray(row["model_chunk_scores"], dtype=np.float64)
            prior_scores = np.asarray(row["prior_chunk_scores"], dtype=np.float64)
            row["reproduced_model"] = score_metrics(row, model_scores)
            row["reproduced_prior"] = score_metrics(row, prior_scores)
            rows[recipe_id] = row
    ordered = [rows[key] for key in sorted(rows)]
    reproduced_model = mean_metrics(ordered, "reproduced_model")
    reproduced_prior = mean_metrics(ordered, "reproduced_prior")
    close_metrics(reproduced_model, metrics["validation"]["model"], f"seed {seed}/model")
    close_metrics(reproduced_prior, metrics["validation"]["prior"], f"seed {seed}/prior")
    return {
        "seed": seed,
        "metrics_path": metrics_path,
        "metrics_sha256": sha256(metrics_path),
        "raw_path": raw_path,
        "raw_sha256": sha256(raw_path),
        "history_path": history_path,
        "history_sha256": sha256(history_path),
        "checkpoint_path": Path(metrics["checkpoint"]),
        "checkpoint_sha256": metrics["checkpoint_sha256"],
        "metrics": metrics,
        "rows": rows,
        "reproduced_model": reproduced_model,
        "reproduced_prior": reproduced_prior,
    }


def hierarchical_bootstrap(
    rows: list[dict[str, Any]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, list[float]]:
    hierarchy: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, row in enumerate(rows):
        hierarchy[str(row["class_id"])][str(row["foreground_cluster_id"])].append(index)
    classes = sorted(hierarchy)
    if len(classes) < 2:
        raise RuntimeError("bootstrap requires multiple event classes")
    delta_hit = np.asarray(
        [row["ensemble"]["hit_at_1"] - row["prior"]["hit_at_1"] for row in rows]
    )
    delta_ap = np.asarray(
        [row["ensemble"]["evidence_ap"] - row["prior"]["evidence_ap"] for row in rows]
    )
    rng = np.random.default_rng(seed)
    sampled_hit = np.empty(replicates, dtype=np.float64)
    sampled_ap = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        indices: list[int] = []
        for class_index in rng.integers(0, len(classes), size=len(classes)):
            class_id = classes[int(class_index)]
            clusters = sorted(hierarchy[class_id])
            for cluster_index in rng.integers(0, len(clusters), size=len(clusters)):
                cluster = clusters[int(cluster_index)]
                recipes = hierarchy[class_id][cluster]
                chosen = rng.integers(0, len(recipes), size=len(recipes))
                indices.extend(recipes[int(value)] for value in chosen)
        sampled_hit[replicate] = float(delta_hit[indices].mean())
        sampled_ap[replicate] = float(delta_ap[indices].mean())
    return {
        "hit_at_1": np.quantile(sampled_hit, [0.025, 0.975]).astype(float).tolist(),
        "evidence_ap": np.quantile(sampled_ap, [0.025, 0.975]).astype(float).tolist(),
    }


def condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for condition in ("duration_sec", "position_bin", "snr_db"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row[condition])].append(row)
        output[condition] = {}
        for value, group in sorted(groups.items()):
            ensemble = mean_metrics(group, "ensemble")
            prior = mean_metrics(group, "prior")
            output[condition][value] = {
                "examples": len(group),
                "ensemble": ensemble,
                "prior": prior,
                "delta": {name: ensemble[name] - prior[name] for name in ensemble},
            }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection_lock = audit_selection_lock()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seeds = tuple(int(value) for value in config["seeds"])
    if seeds != (0, 1, 2, 3, 4):
        raise RuntimeError("five-seed protocol changed")
    if "confirm" in str(config["development_index"]).lower() or "confirm" in str(
        config["validation_index"]
    ).lower():
        raise PermissionError("development config references confirmatory data")
    seed_results = [load_seed(args.config, config, seed) for seed in seeds]
    recipe_sets = [set(item["rows"]) for item in seed_results]
    if any(recipe_ids != recipe_sets[0] for recipe_ids in recipe_sets[1:]):
        raise RuntimeError("recipe identities differ across seeds")

    ensemble_rows: list[dict[str, Any]] = []
    for recipe_id in sorted(recipe_sets[0]):
        rows = [item["rows"][recipe_id] for item in seed_results]
        signatures = {baseline_signature(row) for row in rows}
        if len(signatures) != 1:
            raise RuntimeError(f"baseline/label alignment differs: {recipe_id}")
        model_scores = np.stack(
            [np.asarray(row["model_chunk_scores"], dtype=np.float64) for row in rows]
        )
        ensemble_scores = model_scores.mean(axis=0)
        prior_scores = np.asarray(rows[0]["prior_chunk_scores"], dtype=np.float64)
        ensemble_rows.append(
            {
                **{
                    key: rows[0][key]
                    for key in (
                        "recipe_id",
                        "duration_sec",
                        "position_bin",
                        "snr_db",
                        "class_id",
                        "class_label",
                        "foreground_cluster_id",
                    )
                },
                "ensemble": score_metrics(rows[0], ensemble_scores),
                "prior": score_metrics(rows[0], prior_scores),
                "mean_absolute_residual": float(
                    np.mean(np.abs(ensemble_scores - prior_scores))
                ),
            }
        )
    ensemble = mean_metrics(ensemble_rows, "ensemble")
    prior = mean_metrics(ensemble_rows, "prior")
    ensemble_delta = {name: ensemble[name] - prior[name] for name in ensemble}
    bootstrap = hierarchical_bootstrap(ensemble_rows)
    conditions = condition_summary(ensemble_rows)

    per_seed = []
    for item in seed_results:
        validation = item["metrics"]["validation"]
        per_seed.append(
            {
                "seed": item["seed"],
                "best_epoch": item["metrics"]["best_epoch"],
                "model": validation["model"],
                "prior": validation["prior"],
                "delta": validation["delta"],
                "promotion_gates": item["metrics"]["promotion_gates"],
                "checkpoint_sha256": item["checkpoint_sha256"],
                "raw_validation_sha256": item["raw_sha256"],
                "history_sha256": item["history_sha256"],
                "metrics_sha256": item["metrics_sha256"],
            }
        )
    mean_seed_delta = {
        metric: float(np.mean([item["delta"][metric] for item in per_seed]))
        for metric in ensemble
    }
    positive_seeds = sum(
        item["delta"]["hit_at_1"] > 0.0
        and item["delta"]["evidence_ap"] > 0.0
        for item in per_seed
    )
    condition_hit_deltas = [
        group["delta"]["hit_at_1"]
        for condition in conditions.values()
        for group in condition.values()
    ]
    gates = {
        "mean_seed_hit_gain_at_least_0_05": mean_seed_delta["hit_at_1"] >= 0.05,
        "mean_seed_ap_gain_at_least_0_05": mean_seed_delta["evidence_ap"] >= 0.05,
        "ensemble_hit_gain_at_least_0_05": ensemble_delta["hit_at_1"] >= 0.05,
        "ensemble_ap_gain_at_least_0_05": ensemble_delta["evidence_ap"] >= 0.05,
        "bootstrap_hit_lower_above_zero": bootstrap["hit_at_1"][0] > 0.0,
        "bootstrap_ap_lower_above_zero": bootstrap["evidence_ap"][0] > 0.0,
        "at_least_four_positive_seeds": positive_seeds >= 4,
        "no_condition_hit_loss_over_0_05": min(condition_hit_deltas) >= -0.05,
        "integrity": True,
    }
    payload = {
        "status": (
            "development_pass_exact_onset_confirmatory_authorized_once"
            if all(gates.values())
            else "development_failed_exact_onset_confirmatory_sealed"
        ),
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "selection_protocol": str(
            (PROJECT / "q1_plus/EVENT_RANKER_SELECTION_PROTOCOL.md").resolve()
        ),
        "selection_protocol_sha256": sha256(
            PROJECT / "q1_plus/EVENT_RANKER_SELECTION_PROTOCOL.md"
        ),
        "selection_lock_sha256": sha256(SELECTION_LOCK),
        "examples": len(ensemble_rows),
        "seeds": list(seeds),
        "per_seed": per_seed,
        "mean_seed_delta": mean_seed_delta,
        "positive_seeds": positive_seeds,
        "ensemble": ensemble,
        "prior": prior,
        "ensemble_delta": ensemble_delta,
        "mean_absolute_ensemble_residual": float(
            np.mean([row["mean_absolute_residual"] for row in ensemble_rows])
        ),
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "interval_95_percentile": bootstrap,
        },
        "conditions": conditions,
        "worst_condition_hit_delta": min(condition_hit_deltas),
        "promotion_gates": gates,
        "all_promotion_gates_pass": all(gates.values()),
        "integrity": {
            "all_checksums_valid": True,
            "recipe_alignment_failures": 0,
            "baseline_alignment_failures": 0,
            "metric_reproduction_failures": 0,
            "confirmatory_rows_exposed": 0,
        },
        "confirmatory_panel_status": (
            "authorized_for_one_evaluation" if all(gates.values()) else "sealed"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "mean_seed_delta": mean_seed_delta,
                "ensemble_delta": ensemble_delta,
                "bootstrap": payload["bootstrap"],
                "promotion_gates": gates,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
