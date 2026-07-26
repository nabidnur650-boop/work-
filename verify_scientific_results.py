#!/usr/bin/env python3
"""Independently recompute retained headline results in the two-study release."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np


METRIC_NAMES = ("evidence_ap", "hit_at_1", "recall_at_4", "top_chunk_iou")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def close(
    observed: float,
    expected: float,
    label: str,
    *,
    tolerance: float = 1e-11,
) -> None:
    if not math.isclose(
        float(observed), float(expected), rel_tol=0.0, abs_tol=tolerance
    ):
        raise RuntimeError(
            f"{label} mismatch: observed={observed!r}, expected={expected!r}"
        )


def close_array(
    observed: Iterable[float],
    expected: Iterable[float],
    label: str,
    *,
    tolerance: float = 1e-11,
) -> None:
    left = np.asarray(list(observed), dtype=np.float64)
    right = np.asarray(list(expected), dtype=np.float64)
    if left.shape != right.shape or not np.allclose(
        left, right, rtol=0.0, atol=tolerance
    ):
        maximum = (
            float(np.max(np.abs(left - right)))
            if left.shape == right.shape and left.size
            else float("inf")
        )
        raise RuntimeError(f"{label} mismatch; maximum absolute error={maximum}")


def verify_hash(path: Path, expected: str, label: str) -> None:
    require(path.is_file(), f"missing {label}: {path}")
    observed = sha256(path)
    require(observed == expected, f"{label} SHA-256 mismatch: {path}")


def geometric(values: np.ndarray) -> float:
    require(
        values.size > 0
        and bool(np.isfinite(values).all())
        and bool(np.all(values > 0.0)),
        "geometric mean received invalid values",
    )
    return float(np.exp(np.log(values).mean()))


def bootstrap_geometric_skill(
    matrix: np.ndarray, *, replicates: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for offset in range(0, replicates, 250):
        count = min(250, replicates - offset)
        indices = rng.integers(
            0, len(matrix), size=(count, len(matrix))
        )
        values[offset : offset + count] = 1.0 - np.exp(
            np.log(matrix[indices]).mean(axis=(1, 2))
        )
    return {
        "skill": 1.0 - geometric(matrix.ravel()),
        "interval": np.quantile(values, [0.025, 0.975])
        .astype(float)
        .tolist(),
        "probability_positive": float(np.mean(values > 0.0)),
    }


def verify_shift_original(base: Path) -> dict[str, Any]:
    panel = base / "q1_top_tier"
    report_path = panel / "results/fev/fev_panel_report.json"
    config_path = panel / "configs/fev_natural_panel.json"
    cells_path = panel / "results/fev/fev_task_backbone_cells.csv"
    report = read_json(report_path)
    config = read_json(config_path)
    verify_hash(config_path, report["config_sha256"], "original FEV config")
    verify_hash(cells_path, report["cells_sha256"], "original FEV cells")

    backbones = list(config["backbones"])
    methods = list(config["methods"])
    summaries: dict[tuple[str, str, str], dict[str, Any]] = {}
    raw_sums: dict[tuple[str, str, str], float] = defaultdict(float)
    raw_rows = 0
    zero_valid_rows = 0
    for backbone in backbones:
        directory = panel / f"results/fev/{backbone}"
        summary_path = directory / "fev_summaries.json"
        raw_path = directory / "origin_losses.jsonl.gz"
        audit_path = directory / "task_audits.json"
        receipt_path = directory / "receipt.json"
        receipt = read_json(receipt_path)
        verify_hash(
            receipt_path,
            report["backbone_receipt_sha256"][backbone],
            f"{backbone} original receipt",
        )
        verify_hash(
            summary_path,
            receipt["summaries_sha256"],
            f"{backbone} original summaries",
        )
        verify_hash(
            raw_path,
            receipt["origin_losses_sha256"],
            f"{backbone} original raw losses",
        )
        verify_hash(
            audit_path,
            receipt["task_audits_sha256"],
            f"{backbone} original task audits",
        )
        backbone_rows = 0
        for item in read_json(summary_path):
            key = (
                str(item["task_name"]),
                backbone,
                str(item["adapter"]),
            )
            require(key not in summaries, f"duplicate original summary: {key}")
            summaries[key] = item
        for row in iter_jsonl(raw_path):
            require(
                row["backbone"] == backbone,
                f"original raw backbone mismatch: {backbone}",
            )
            losses = row["loss_sum"]
            require(
                set(losses) == set(methods),
                "original raw method set mismatch",
            )
            values = np.asarray(list(losses.values()), dtype=np.float64)
            require(
                bool(np.isfinite(values).all()) and bool(np.all(values >= 0.0)),
                "original raw loss is invalid",
            )
            weights = np.asarray(row["weights"], dtype=np.float64)
            require(
                bool(np.isfinite(weights).all())
                and bool(np.all(weights >= -1e-12))
                and abs(float(weights.sum()) - 1.0) <= 1e-9,
                "original raw portfolio weights are invalid",
            )
            valid = int(row["valid_target_count"])
            if valid <= 0:
                zero_valid_rows += 1
            elif bool(row["mase_scale_defined"]):
                scale = float(row["mase_scale"])
                require(
                    math.isfinite(scale) and scale > 0.0,
                    "original MASE scale is invalid",
                )
                for method in methods:
                    # The frozen runner stores scale-normalized absolute-loss
                    # sums. FEV's aggregate can differ by a few ppm where it
                    # applies its final missing-value accounting.
                    raw_sums[(str(row["task"]), backbone, method)] += float(
                        losses[method]
                    )
            backbone_rows += 1
        require(
            backbone_rows == int(receipt["origin_rows"]),
            f"original raw row count mismatch: {backbone}",
        )
        raw_rows += backbone_rows

    tasks = sorted({key[0] for key in summaries})
    require(
        len(tasks) == 20
        and len(summaries) == len(tasks) * len(backbones) * len(methods),
        "original FEV summary matrix is incomplete",
    )
    primary = str(config["primary_method"])
    comparator = str(config["primary_comparator"])
    matrix = np.asarray(
        [
            [
                float(summaries[(task, backbone, primary)]["MASE"])
                / float(summaries[(task, backbone, comparator)]["MASE"])
                for backbone in backbones
            ]
            for task in tasks
        ],
        dtype=np.float64,
    )

    with cells_path.open("r", encoding="utf-8", newline="") as handle:
        cell_rows = list(csv.DictReader(handle))
    require(len(cell_rows) == 60, "original FEV cell CSV is incomplete")
    cell_lookup = {
        (row["task"], row["backbone"]): float(row["ratio_to_static"])
        for row in cell_rows
    }
    close_array(
        matrix.ravel(),
        [
            cell_lookup[(task, backbone)]
            for task in tasks
            for backbone in backbones
        ],
        "original FEV cell ratios",
    )
    raw_ratio_errors = []
    for task in tasks:
        for backbone in backbones:
            raw_static = raw_sums[(task, backbone, comparator)]
            raw_safe = raw_sums[(task, backbone, primary)]
            require(raw_static > 0.0, "original raw static loss is empty")
            raw_ratio_errors.append(
                abs(raw_safe / raw_static - cell_lookup[(task, backbone)])
            )
    require(
        max(raw_ratio_errors) <= 1e-5,
        "original raw losses do not reproduce the published cells",
    )

    observed = bootstrap_geometric_skill(
        matrix,
        replicates=int(config["bootstrap"]["replicates"]),
        seed=int(config["bootstrap"]["seed"]),
    )
    declared = report["primary"]
    close(
        geometric(matrix.ravel()),
        declared["geometric_mean_mase_ratio_to_static"],
        "original FEV geometric ratio",
    )
    close(observed["skill"], declared["skill_score"], "original FEV skill")
    close_array(
        observed["interval"],
        declared["bootstrap"]["interval_95_percentile"],
        "original FEV bootstrap interval",
    )
    close(
        observed["probability_positive"],
        declared["bootstrap"]["probability_skill_above_zero"],
        "original FEV bootstrap probability",
    )
    require(
        int(np.sum(matrix < 1.0)) == int(declared["task_backbone_wins"])
        and int(np.sum(matrix == 1.0))
        == int(declared["task_backbone_ties"]),
        "original FEV win/tie counts changed",
    )
    require(
        report["status"] == "fev_panel_promotion_fail"
        and report["all_promotion_gates_pass"] is False,
        "original FEV decision boundary changed",
    )
    return {
        "status": report["status"],
        "tasks": len(tasks),
        "cells": int(matrix.size),
        "raw_origin_rows": raw_rows,
        "zero_valid_origin_rows": zero_valid_rows,
        "maximum_raw_to_summary_ratio_error": max(raw_ratio_errors),
        "geometric_ratio": geometric(matrix.ravel()),
        "skill": observed["skill"],
    }


def verify_shift_fresh(base: Path) -> dict[str, Any]:
    panel = base / "q1_fresh_replication"
    report_path = panel / "results/fresh_fev_report.json"
    config_path = panel / "configs/fresh_fev_expansion.json"
    report = read_json(report_path)
    config = read_json(config_path)
    verify_hash(config_path, report["config_sha256"], "fresh FEV config")

    tasks = list(report["tasks"])
    backbones = list(config["backbones"])
    methods = tuple(report["adapter_ablations"])
    sums: dict[tuple[str, str, str], float] = defaultdict(float)
    counts: Counter[tuple[str, str]] = Counter()
    rows = 0
    for artifact in report["artifacts"]:
        backbone = str(artifact["backbone"])
        raw_path = base / str(artifact["origin_losses"])
        receipt_path = base / str(artifact["receipt"])
        verify_hash(
            raw_path,
            artifact["origin_losses_sha256"],
            f"{backbone} fresh raw losses",
        )
        verify_hash(
            receipt_path,
            artifact["receipt_sha256"],
            f"{backbone} fresh receipt",
        )
        receipt = read_json(receipt_path)
        verify_hash(
            raw_path,
            receipt["origin_losses_sha256"],
            f"{backbone} fresh receipt raw reference",
        )
        backbone_rows = 0
        for row in iter_jsonl(raw_path):
            task = str(row["task"])
            require(
                task in tasks and row["backbone"] == backbone,
                "fresh raw row leaves frozen panel",
            )
            losses = row["loss_sum"]
            require(set(losses) == set(methods), "fresh raw method set mismatch")
            values = np.asarray(list(losses.values()), dtype=np.float64)
            require(
                bool(np.isfinite(values).all()) and bool(np.all(values >= 0.0)),
                "fresh raw loss is invalid",
            )
            weights = np.asarray(row["weights"], dtype=np.float64)
            require(
                bool(np.isfinite(weights).all())
                and bool(np.all(weights >= -1e-12))
                and abs(float(weights.sum()) - 1.0) <= 1e-9,
                "fresh raw portfolio weights are invalid",
            )
            for method in methods:
                sums[(task, backbone, method)] += float(losses[method])
            counts[(task, backbone)] += 1
            backbone_rows += 1
        require(
            backbone_rows == int(artifact["origin_rows"])
            == int(receipt["origin_rows"]),
            f"fresh row count mismatch: {backbone}",
        )
        rows += backbone_rows
    require(len(counts) == 60, "fresh FEV raw cell matrix is incomplete")

    cells = {
        (str(row["task"]), str(row["backbone"])): row
        for row in report["cells"]
    }
    matrix = np.empty((len(tasks), len(backbones)), dtype=np.float64)
    for task_index, task in enumerate(tasks):
        for backbone_index, backbone in enumerate(backbones):
            cell = cells[(task, backbone)]
            static = sums[(task, backbone, "static")]
            safe = sums[(task, backbone, "safe_portfolio")]
            ratio = safe / static
            close(
                static,
                cell["static_scaled_absolute_loss"],
                f"fresh static loss {task}/{backbone}",
            )
            close(
                safe,
                cell["safe_scaled_absolute_loss"],
                f"fresh safe loss {task}/{backbone}",
            )
            close(
                ratio,
                cell["ratio_to_static"],
                f"fresh ratio {task}/{backbone}",
            )
            require(
                counts[(task, backbone)] == int(cell["origin_rows"]),
                f"fresh origin count mismatch: {task}/{backbone}",
            )
            matrix[task_index, backbone_index] = ratio

    observed = bootstrap_geometric_skill(
        matrix,
        replicates=int(config["bootstrap"]["replicates"]),
        seed=int(config["bootstrap"]["seed"]),
    )
    declared = report["primary"]
    tolerance = 1e-12
    wins = int(np.sum(matrix < 1.0 - tolerance))
    ties = int(np.sum(np.abs(matrix - 1.0) <= tolerance))
    losses = int(np.sum(matrix > 1.0 + tolerance))
    close(
        geometric(matrix.ravel()),
        declared["geometric_mean_ratio_to_static"],
        "fresh FEV geometric ratio",
    )
    close(observed["skill"], declared["skill_score"], "fresh FEV skill")
    close_array(
        observed["interval"],
        declared["bootstrap"]["interval_95_percentile"],
        "fresh FEV bootstrap interval",
    )
    close(
        observed["probability_positive"],
        declared["bootstrap"]["probability_skill_above_zero"],
        "fresh FEV bootstrap probability",
    )
    require(
        (wins, ties, losses)
        == (
            int(declared["wins"]),
            int(declared["ties"]),
            int(declared["losses"]),
        ),
        "fresh FEV win/tie/loss counts changed",
    )
    require(
        report["status"] == "fresh_fev_expansion_promotion_fail"
        and report["all_promotion_gates_pass"] is False,
        "fresh FEV decision boundary changed",
    )
    return {
        "status": report["status"],
        "tasks": len(tasks),
        "cells": int(matrix.size),
        "raw_origin_rows": rows,
        "geometric_ratio": geometric(matrix.ravel()),
        "skill": observed["skill"],
    }


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    require(0 < positives < len(labels), "invalid evidence labels")
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order].astype(np.float64)
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    return float((precision * sorted_labels).sum() / positives)


def interval_iou(
    start: float, end: float, target_start: float, target_end: float
) -> float:
    intersection = max(
        0.0, min(end, target_end) - max(start, target_start)
    )
    union = max(end, target_end) - min(start, target_start)
    return float(intersection / union) if union > 0.0 else 0.0


def score_event_row(
    row: dict[str, Any], scores: Iterable[float]
) -> dict[str, float]:
    score_array = np.asarray(list(scores), dtype=np.float64)
    evidence = np.asarray(row["evidence_targets"], dtype=bool)
    starts = np.asarray(row["chunk_start_sec"], dtype=np.float64)
    ends = np.asarray(row["chunk_end_sec"], dtype=np.float64)
    require(
        len(score_array)
        == len(evidence)
        == len(starts)
        == len(ends)
        == int(row["n_chunks"]),
        f"event chunk alignment failure: {row['recipe_id']}",
    )
    require(
        bool(np.isfinite(score_array).all()),
        f"nonfinite event score: {row['recipe_id']}",
    )
    order = np.argsort(-score_array, kind="stable")
    top = int(order[0])
    return {
        "evidence_ap": average_precision(score_array, evidence),
        "hit_at_1": float(evidence[top]),
        "recall_at_4": float(evidence[order[:4]].any()),
        "top_chunk_iou": interval_iou(
            float(starts[top]),
            float(ends[top]),
            float(row["target_start_sec"]),
            float(row["target_end_sec"]),
        ),
    }


def mean_event_metrics(
    rows: list[dict[str, Any]], key: str
) -> dict[str, float]:
    return {
        name: float(np.mean([row[key][name] for row in rows]))
        for name in METRIC_NAMES
    }


def compare_metric_mapping(
    observed: dict[str, float],
    expected: dict[str, float],
    label: str,
) -> None:
    require(set(observed) == set(expected), f"{label} metric keys changed")
    for name in observed:
        close(observed[name], expected[name], f"{label}/{name}", tolerance=1e-10)


def hierarchical_event_bootstrap(
    rows: list[dict[str, Any]], *, seed: int, replicates: int
) -> dict[str, list[float]]:
    hierarchy: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, row in enumerate(rows):
        hierarchy[str(row["class_id"])][
            str(row["foreground_cluster_id"])
        ].append(index)
    classes = sorted(hierarchy)
    delta_ap = np.asarray(
        [
            row["_ensemble"]["evidence_ap"] - row["_prior"]["evidence_ap"]
            for row in rows
        ]
    )
    delta_hit = np.asarray(
        [
            row["_ensemble"]["hit_at_1"] - row["_prior"]["hit_at_1"]
            for row in rows
        ]
    )
    rng = np.random.default_rng(seed)
    sampled_ap = np.empty(replicates, dtype=np.float64)
    sampled_hit = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected: list[int] = []
        for class_index in rng.integers(
            0, len(classes), size=len(classes)
        ):
            class_id = classes[int(class_index)]
            clusters = sorted(hierarchy[class_id])
            for cluster_index in rng.integers(
                0, len(clusters), size=len(clusters)
            ):
                recipes = hierarchy[class_id][clusters[int(cluster_index)]]
                draws = rng.integers(0, len(recipes), size=len(recipes))
                selected.extend(recipes[int(value)] for value in draws)
        sampled_ap[replicate] = float(delta_ap[selected].mean())
        sampled_hit[replicate] = float(delta_hit[selected].mean())
    return {
        "evidence_ap": np.quantile(
            sampled_ap, [0.025, 0.975]
        ).astype(float).tolist(),
        "hit_at_1": np.quantile(
            sampled_hit, [0.025, 0.975]
        ).astype(float).tolist(),
    }


def event_condition_minimum(rows: list[dict[str, Any]]) -> float:
    deltas = []
    for field in ("duration_sec", "position_bin", "snr_db"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row[field])].append(row)
        for group in groups.values():
            deltas.append(
                mean_event_metrics(group, "_ensemble")["hit_at_1"]
                - mean_event_metrics(group, "_prior")["hit_at_1"]
            )
    return min(deltas)


def verify_event_panel(
    base: Path,
    *,
    stage: str,
    report_relative: str,
    raw_relative: str,
    expected_rows: int,
    external: bool,
) -> dict[str, Any]:
    report_path = base / report_relative
    raw_path = base / raw_relative
    report = read_json(report_path)
    verify_hash(raw_path, report["raw_sha256"], f"{stage} raw evaluation")
    if external:
        authorization_path = base / "q1_90/EXTERNAL_REPLICATION_AUTHORIZATION.json"
        config_path = base / "q1_plus/configs/development_event_ranker.json"
    else:
        authorization_path = base / "q1_plus/EVENT_CONFIRMATORY_AUTHORIZATION.json"
        config_path = base / "q1_plus/configs/development_event_ranker.json"
    verify_hash(
        authorization_path,
        report["authorization_sha256"],
        f"{stage} authorization",
    )
    verify_hash(config_path, report["config_sha256"], f"{stage} config")

    rows = list(iter_jsonl(raw_path))
    require(
        len(rows) == expected_rows == int(report["examples"]),
        f"{stage} raw row count mismatch",
    )
    recipe_ids: set[str] = set()
    maximum_residual = 0.0
    for row in rows:
        recipe_id = str(row["recipe_id"])
        require(recipe_id not in recipe_ids, f"duplicate {stage} recipe")
        recipe_ids.add(recipe_id)
        row["_prior"] = score_event_row(row, row["prior_chunk_scores"])
        row["_ensemble"] = score_event_row(
            row, row["ensemble_chunk_scores"]
        )
        compare_metric_mapping(
            row["_prior"], row["prior_metrics"], f"{stage}/{recipe_id}/prior"
        )
        compare_metric_mapping(
            row["_ensemble"],
            row["ensemble_metrics"],
            f"{stage}/{recipe_id}/ensemble",
        )
        require(
            len(row["seed_chunk_scores"]) == len(row["seed_metrics"]) == 5,
            f"{stage} seed coverage mismatch",
        )
        for index, scores in enumerate(row["seed_chunk_scores"]):
            observed_seed = score_event_row(row, scores)
            compare_metric_mapping(
                observed_seed,
                row["seed_metrics"][index],
                f"{stage}/{recipe_id}/seed-{index}",
            )
        maximum_residual = max(
            maximum_residual, float(row["maximum_absolute_residual"])
        )

    prior = mean_event_metrics(rows, "_prior")
    ensemble = mean_event_metrics(rows, "_ensemble")
    delta = {name: ensemble[name] - prior[name] for name in METRIC_NAMES}
    compare_metric_mapping(prior, report["prior"], f"{stage} prior mean")
    compare_metric_mapping(
        ensemble, report["ensemble"], f"{stage} ensemble mean"
    )
    compare_metric_mapping(delta, report["delta"], f"{stage} mean delta")

    seed_deltas = []
    for index in range(5):
        seed_metrics = {
            name: float(
                np.mean(
                    [
                        row["seed_metrics"][index][name]
                        for row in rows
                    ]
                )
            )
            for name in METRIC_NAMES
        }
        seed_deltas.append(
            {name: seed_metrics[name] - prior[name] for name in METRIC_NAMES}
        )
        compare_metric_mapping(
            seed_metrics,
            report["per_seed"][index]["model"],
            f"{stage} seed-{index} mean",
        )
        compare_metric_mapping(
            seed_deltas[-1],
            report["per_seed"][index]["delta"],
            f"{stage} seed-{index} delta",
        )
    positive_seeds = sum(
        item["hit_at_1"] > 0.0 and item["evidence_ap"] > 0.0
        for item in seed_deltas
    )
    require(
        positive_seeds == int(report["positive_seeds"]),
        f"{stage} positive-seed count mismatch",
    )

    bootstrap = hierarchical_event_bootstrap(
        rows,
        seed=int(report["bootstrap"]["seed"]),
        replicates=int(report["bootstrap"]["replicates"]),
    )
    for metric in ("evidence_ap", "hit_at_1"):
        close_array(
            bootstrap[metric],
            report["bootstrap"]["interval_95_percentile"][metric],
            f"{stage} {metric} bootstrap",
            tolerance=1e-10,
        )
    close(
        maximum_residual,
        report["integrity"]["maximum_absolute_seed_residual"],
        f"{stage} maximum residual",
        tolerance=1e-10,
    )

    if external:
        class_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            class_groups[str(row["class_label"])].append(row)
        minimum_class_hit = min(
            mean_event_metrics(group, "_ensemble")["hit_at_1"]
            - mean_event_metrics(group, "_prior")["hit_at_1"]
            for group in class_groups.values()
        )
        close(
            minimum_class_hit,
            report["minimum_class_hit_delta"],
            f"{stage} minimum class hit delta",
            tolerance=1e-10,
        )
        performance_index = 50.0 * (
            ensemble["evidence_ap"] + ensemble["hit_at_1"]
        )
        close(
            performance_index,
            report["external_performance_index"],
            f"{stage} performance index",
        )
        expected_gates = {
            "ensemble_ap_gain_at_least_0_05": delta["evidence_ap"] >= 0.05,
            "ensemble_hit_gain_at_least_0_05": delta["hit_at_1"] >= 0.05,
            "bootstrap_ap_lower_above_zero": bootstrap["evidence_ap"][0] > 0.0,
            "bootstrap_hit_lower_above_zero": bootstrap["hit_at_1"][0] > 0.0,
            "at_least_four_positive_seeds": positive_seeds >= 4,
            "no_class_hit_delta_below_minus_0_05": minimum_class_hit >= -0.05,
            "external_performance_index_at_least_85": performance_index >= 85.0,
            "integrity": True,
        }
        expected_status = "external_replication_gate_fail"
    else:
        minimum_condition_hit = event_condition_minimum(rows)
        close(
            minimum_condition_hit,
            report["worst_condition_hit_delta"],
            f"{stage} worst condition hit delta",
            tolerance=1e-10,
        )
        expected_gates = {
            "ensemble_ap_gain_at_least_0_05": delta["evidence_ap"] >= 0.05,
            "ensemble_hit_gain_at_least_0_05": delta["hit_at_1"] >= 0.05,
            "bootstrap_ap_lower_above_zero": bootstrap["evidence_ap"][0] > 0.0,
            "bootstrap_hit_lower_above_zero": bootstrap["hit_at_1"][0] > 0.0,
            "at_least_four_positive_seeds": positive_seeds >= 4,
            "no_condition_hit_loss_over_0_05": minimum_condition_hit >= -0.05,
            "integrity": True,
        }
        expected_status = "confirmatory_exact_onset_gate_pass"
    require(
        expected_gates == report["promotion_gates"],
        f"{stage} promotion gates do not recompute",
    )
    require(
        report["status"] == expected_status
        and bool(report["all_promotion_gates_pass"])
        == all(expected_gates.values()),
        f"{stage} decision boundary changed",
    )
    return {
        "status": report["status"],
        "raw_examples": len(rows),
        "prior_evidence_ap": prior["evidence_ap"],
        "ensemble_evidence_ap": ensemble["evidence_ap"],
        "evidence_ap_delta": delta["evidence_ap"],
        "positive_seeds": positive_seeds,
    }


def verify_metric_arithmetic(metrics: dict[str, Any], label: str) -> None:
    class_arrays = {
        str(key): np.asarray(value, dtype=np.float64)
        for key, value in metrics["class_ap_by_threshold"].items()
    }
    require(bool(class_arrays), f"{label} has no class metrics")
    lengths = {len(value) for value in class_arrays.values()}
    require(lengths == {len(metrics["thresholds"])}, f"{label} shape mismatch")
    for class_id, values in class_arrays.items():
        close(
            float(values.mean()),
            metrics["class_mean_ap"][class_id],
            f"{label}/class-{class_id} mean",
        )
    matrix = np.vstack([class_arrays[key] for key in sorted(class_arrays)])
    close_array(
        matrix.mean(axis=0),
        metrics["map_by_threshold"],
        f"{label} threshold macro means",
    )
    close(
        float(matrix.mean()),
        metrics["mean_map"],
        f"{label} overall mean mAP",
    )


def paired_mean_bootstrap(
    differences: np.ndarray, *, replicates: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for offset in range(0, replicates, 100):
        count = min(100, replicates - offset)
        indices = rng.integers(
            0, len(differences), size=(count, len(differences))
        )
        values[offset : offset + count] = differences[indices].mean(axis=1)
    return {
        "mean": float(differences.mean()),
        "interval": np.quantile(values, [0.025, 0.975])
        .astype(float)
        .tolist(),
        "probability_positive": float(np.mean(values > 0.0)),
    }


def scan_predictions(
    path: Path,
    *,
    expected_hash: str,
    expected_rows: int,
    expected_videos: int,
    maximum_per_video: int,
    labels: set[int],
    label: str,
) -> dict[str, int]:
    verify_hash(path, expected_hash, label)
    counts: Counter[str] = Counter()
    rows = 0
    for row in iter_jsonl(path):
        video_id = str(row["video_id"])
        score = float(row["score"])
        start = float(row["start_sec"])
        end = float(row["end_sec"])
        require(
            int(row["label_id"]) in labels
            and math.isfinite(score)
            and math.isfinite(start)
            and math.isfinite(end)
            and 0.0 <= score <= 1.0
            and 0.0 <= start < end,
            f"invalid prediction row in {label}",
        )
        counts[video_id] += 1
        rows += 1
    require(rows == expected_rows, f"{label} row count mismatch")
    require(len(counts) == expected_videos, f"{label} video coverage mismatch")
    observed_maximum = max(counts.values())
    require(
        observed_maximum <= maximum_per_video,
        f"{label} exceeds per-video cap",
    )
    return {
        "rows": rows,
        "videos": len(counts),
        "maximum_rows_per_video": observed_maximum,
    }


def verify_natural_panel(base: Path) -> dict[str, Any]:
    panel = base / "q1_top_tier"
    report_path = (
        panel / "results/perception_test/evaluation/perception_test_report.json"
    )
    config_path = panel / "configs/perception_test_natural.json"
    video_path = (
        panel / "results/perception_test/evaluation/per_video_macro_ap.jsonl.gz"
    )
    report = read_json(report_path)
    config = read_json(config_path)
    verify_hash(config_path, report["config_sha256"], "natural panel config")
    verify_hash(
        panel / "PERCEPTION_TEST_FREEZE.json",
        report["freeze_lock_sha256"],
        "natural panel freeze",
    )
    verify_hash(
        video_path,
        report["per_video_macro_ap_sha256"],
        "natural per-video macro AP",
    )

    methods = list(config["methods"])
    for method in methods:
        verify_metric_arithmetic(
            report["methods"][method], f"natural/{method}"
        )
    primary = str(config["primary_method"])
    comparator = str(config["primary_comparator"])
    primary_metrics = report["methods"][primary]
    comparator_metrics = report["methods"][comparator]
    contrast = report["primary_contrast"]
    close(
        float(primary_metrics["mean_map"])
        - float(comparator_metrics["mean_map"]),
        contrast["official_mean_map_delta"],
        "natural official mAP delta arithmetic",
    )
    close_array(
        np.asarray(primary_metrics["map_by_threshold"])
        - np.asarray(comparator_metrics["map_by_threshold"]),
        contrast["map_delta_by_threshold"],
        "natural threshold deltas",
    )
    for class_id, value in contrast["class_mean_ap_delta"].items():
        close(
            float(primary_metrics["class_mean_ap"][class_id])
            - float(comparator_metrics["class_mean_ap"][class_id]),
            value,
            f"natural class-{class_id} delta",
        )

    video_rows = list(iter_jsonl(video_path))
    require(
        len(video_rows) == int(report["videos"]),
        "natural per-video row count mismatch",
    )
    video_ids = [str(row["video_id"]) for row in video_rows]
    require(
        len(set(video_ids)) == len(video_ids),
        "natural per-video IDs are not unique",
    )
    differences = np.asarray(
        [
            float(row[primary]) - float(row[comparator])
            for row in video_rows
        ],
        dtype=np.float64,
    )
    require(
        bool(np.isfinite(differences).all()),
        "natural per-video effects contain nonfinite values",
    )
    bootstrap_config = config["bootstrap"]
    observed = paired_mean_bootstrap(
        differences,
        replicates=int(bootstrap_config["replicates"]),
        seed=int(bootstrap_config["seed"]),
    )
    declared_bootstrap = contrast["paired_video_bootstrap"]
    close(
        observed["mean"],
        declared_bootstrap["observed_mean_delta"],
        "natural per-video mean delta",
    )
    close_array(
        observed["interval"],
        declared_bootstrap["interval_95_percentile"],
        "natural per-video bootstrap interval",
    )
    close(
        observed["probability_positive"],
        declared_bootstrap["probability_delta_above_zero"],
        "natural bootstrap probability",
    )

    prediction_directory = panel / "results/perception_test/evaluation/predictions"
    labels = {int(value) for value in config["benchmark"]["label_ids"]}
    cap = int(config["postprocessing"]["maximum_segments_per_video"])
    prediction_audits = {}
    for method in methods:
        prediction_audits[method] = scan_predictions(
            prediction_directory / f"{method}.jsonl.gz",
            expected_hash=report["prediction_sha256"][method],
            expected_rows=int(report["prediction_counts"][method]),
            expected_videos=int(report["videos"]),
            maximum_per_video=cap,
            labels=labels,
            label=f"natural/{method} predictions",
        )
    require(
        report["status"] == "natural_panel_promotion_fail"
        and report["all_promotion_gates_pass"] is False,
        "natural panel decision boundary changed",
    )
    return {
        "status": report["status"],
        "videos": len(video_rows),
        "ground_truth_events_declared": int(report["ground_truth_events"]),
        "official_map_delta_arithmetic": contrast[
            "official_mean_map_delta"
        ],
        "per_video_mean_delta": observed["mean"],
        "prediction_rows": sum(
            item["rows"] for item in prediction_audits.values()
        ),
        "official_metric_recomputation_boundary": (
            "metric arithmetic and prediction artifacts verified; full "
            "ground-truth mAP recomputation requires excluded provider annotations"
        ),
    }


def verify_router(base: Path) -> dict[str, Any]:
    track = base / "q1_crossfit_capcorrect"
    report_path = track / "results/capcorrect_router/capcorrect_router_report.json"
    report = read_json(report_path)
    verify_hash(
        track / "configs/cap_corrected_crossfit.json",
        report["config_sha256"],
        "router config",
    )
    verify_hash(
        track / "CAPCORRECT_ANALYSIS_LOCK.json",
        report["capcorrect_analysis_lock_sha256"],
        "router analysis lock",
    )
    verify_hash(
        track / "CAPCORRECT_ANALYZER_AMENDMENT_LOCK.json",
        report["analyzer_amendment_lock_sha256"],
        "router analyzer amendment",
    )
    receipt_path = track / "results/candidate_pools/receipt.json"
    verify_hash(
        receipt_path,
        report["candidate_pool_receipt_sha256"],
        "router candidate-pool receipt",
    )
    natural_path = (
        base
        / "q1_top_tier/results/perception_test/evaluation/perception_test_report.json"
    )
    verify_hash(
        natural_path,
        report["source_natural_report_sha256"],
        "router source natural report",
    )
    receipt = read_json(receipt_path)
    require(
        receipt["pool_counts"] == report["candidate_pool_rows"],
        "router candidate-pool row ledger mismatch",
    )
    require(
        int(receipt["cap_reconstruction_exact_matches"]) == 10
        and all(
            item["exact_canonical_match"]
            for item in receipt["cap_reconstruction"].values()
        ),
        "router cap reconstruction ledger is incomplete",
    )

    primary = report["primary"]
    router_metrics = primary["router_metrics"]
    comparator_metrics = primary["comparator_metrics"]
    clap_metrics = report["clap_only_ablation"]["metrics"]
    verify_metric_arithmetic(router_metrics, "router/full")
    verify_metric_arithmetic(comparator_metrics, "router/comparator")
    verify_metric_arithmetic(clap_metrics, "router/clap-only")
    natural = read_json(natural_path)
    natural_comparator = natural["methods"][primary["comparator"]]
    require(
        comparator_metrics == natural_comparator,
        "router comparator metrics differ from frozen natural report",
    )
    close(
        float(router_metrics["mean_map"])
        - float(comparator_metrics["mean_map"]),
        primary["official_mean_map_delta"],
        "router official mAP delta arithmetic",
    )
    close(
        float(router_metrics["mean_map"])
        / float(comparator_metrics["mean_map"])
        - 1.0,
        primary["relative_mean_map_gain"],
        "router relative mAP gain",
    )
    close_array(
        np.asarray(router_metrics["map_by_threshold"])
        - np.asarray(comparator_metrics["map_by_threshold"]),
        primary["threshold_delta"],
        "router threshold deltas",
    )
    for class_id, value in primary["class_delta"].items():
        close(
            float(router_metrics["class_mean_ap"][class_id])
            - float(comparator_metrics["class_mean_ap"][class_id]),
            value,
            f"router class-{class_id} delta",
        )
    close(
        float(clap_metrics["mean_map"])
        - float(comparator_metrics["mean_map"]),
        report["clap_only_ablation"]["delta_to_clap_multiscale"],
        "router CLAP-only delta",
    )

    artifacts = report["artifacts"]
    labels = {int(value) for value in natural["label_names"]}
    maximum = int(report["integrity"]["router_maximum_segments_per_video"])
    full_audit = scan_predictions(
        base / artifacts["full_router_predictions"],
        expected_hash=artifacts["full_router_predictions_sha256"],
        expected_rows=int(artifacts["full_router_prediction_rows"]),
        expected_videos=int(report["integrity"]["videos"]),
        maximum_per_video=maximum,
        labels=labels,
        label="full router predictions",
    )
    clap_audit = scan_predictions(
        base / artifacts["clap_only_predictions"],
        expected_hash=artifacts["clap_only_predictions_sha256"],
        expected_rows=int(artifacts["clap_only_prediction_rows"]),
        expected_videos=int(report["integrity"]["videos"]),
        maximum_per_video=maximum,
        labels=labels,
        label="CLAP-only router predictions",
    )
    require(
        report["status"]
        == "cap_corrected_post_outcome_diagnostic_target_met"
        and report["claim_type"]
        == "post_outcome_cross_fitted_exploratory"
        and report["integrity"]["fresh_confirmatory_claim"] is False
        and report["integrity"]["zero_shot_claim"] is False,
        "router claim boundary changed",
    )
    return {
        "status": report["status"],
        "claim_type": report["claim_type"],
        "full_router_map": router_metrics["mean_map"],
        "clap_only_map": clap_metrics["mean_map"],
        "incremental_qcr_map": float(router_metrics["mean_map"])
        - float(clap_metrics["mean_map"]),
        "prediction_rows": full_audit["rows"] + clap_audit["rows"],
        "candidate_pool_receipt_entries": len(receipt["pool_counts"]),
        "official_metric_recomputation_boundary": (
            "metric arithmetic, cap ledger, hashes, and every final prediction "
            "verified; full ground-truth mAP recomputation requires excluded "
            "provider annotations and regenerable candidate pools"
        ),
    }


def source_values(
    rows: list[dict[str, Any]], system: str, metric: str
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["system"] == system:
            grouped[str(row["target_source_id"])].append(float(row[metric]))
    return {
        source: float(np.mean(values)) for source, values in grouped.items()
    }


def source_macro(
    rows: list[dict[str, Any]], system: str
) -> dict[str, float | int]:
    exact = source_values(rows, system, "exact")
    token_f1 = source_values(rows, system, "token_f1")
    require(set(exact) == set(token_f1), "answer source alignment changed")
    return {
        "source_macro_exact": float(np.mean(list(exact.values()))),
        "source_macro_token_f1": float(np.mean(list(token_f1.values()))),
        "sources": len(exact),
        "examples": sum(row["system"] == system for row in rows),
    }


def combined_source_values(
    rows_by_model: dict[str, list[dict[str, Any]]],
    system: str,
    metric: str,
    models: tuple[str, ...],
) -> dict[str, float]:
    values = {
        model: source_values(rows_by_model[model], system, metric)
        for model in models
    }
    sources = set(values[models[0]])
    require(
        all(set(values[model]) == sources for model in models),
        "answer backbone source sets differ",
    )
    return {
        source: float(np.mean([values[model][source] for model in models]))
        for source in sources
    }


def paired_source_interval(
    left: dict[str, float],
    right: dict[str, float],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    sources = sorted(left)
    require(set(sources) == set(right), "answer paired source sets differ")
    differences = np.asarray(
        [left[source] - right[source] for source in sources],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(
        0, len(sources), size=(replicates, len(sources))
    )
    boot = differences[draws].mean(axis=1)
    return {
        "delta": float(differences.mean()),
        "bootstrap_95_interval": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "sources": len(sources),
        "replicates": replicates,
        "seed": seed,
    }


def compare_interval(
    observed: dict[str, Any], expected: dict[str, Any], label: str
) -> None:
    require(
        int(observed["sources"]) == int(expected["sources"])
        and int(observed["replicates"]) == int(expected["replicates"])
        and int(observed["seed"]) == int(expected["seed"]),
        f"{label} bootstrap metadata changed",
    )
    close(observed["delta"], expected["delta"], f"{label} delta")
    close_array(
        observed["bootstrap_95_interval"],
        expected["bootstrap_95_interval"],
        f"{label} interval",
    )


def token_f1_from_normalized(response: str, reference: str) -> float:
    response_tokens = response.split()
    reference_tokens = reference.split()
    if not response_tokens or not reference_tokens:
        return float(response_tokens == reference_tokens)
    common = sum(
        (Counter(response_tokens) & Counter(reference_tokens)).values()
    )
    if common == 0:
        return 0.0
    precision = common / len(response_tokens)
    recall = common / len(reference_tokens)
    return 2.0 * precision * recall / (precision + recall)


def verify_answer_bridge(base: Path) -> dict[str, Any]:
    directory = base / "q1_plus/results/development/answer_generation"
    report_path = directory / "heldout_evaluation_report.json"
    config_path = base / "q1_plus/configs/answer_heldout.json"
    report = read_json(report_path)
    config = read_json(config_path)
    models = ("qwen2_audio", "phi4_multimodal")
    systems = tuple(config["systems"])
    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    for artifact in report["artifacts"]:
        model = str(artifact["model"])
        raw_path = base / str(artifact["raw_path"])
        receipt_path = base / str(artifact["receipt_path"])
        verify_hash(
            raw_path, artifact["raw_sha256"], f"{model} held-out answers"
        )
        verify_hash(
            receipt_path,
            artifact["receipt_sha256"],
            f"{model} held-out receipt",
        )
        receipt = read_json(receipt_path)
        verify_hash(
            raw_path,
            receipt["raw_sha256"],
            f"{model} receipt raw reference",
        )
        rows = list(iter_jsonl(raw_path))
        require(
            len(rows) == int(artifact["rows"]) == int(receipt["rows"]),
            f"{model} held-out answer count mismatch",
        )
        jobs: set[str] = set()
        for row in rows:
            job = str(row["job_id"])
            require(job not in jobs, f"duplicate held-out job: {job}")
            jobs.add(job)
            require(
                row["model"] == model
                and row["system"] in systems
                and row["audita_rows_accessed"] == 0
                and bool(str(row["response"]).strip()),
                f"invalid held-out answer row: {job}",
            )
            expected_exact = float(
                row["normalized_response"] == row["normalized_reference"]
            )
            close(
                row["exact"], expected_exact, f"{job} exact recomputation"
            )
            close(
                row["token_f1"],
                token_f1_from_normalized(
                    str(row["normalized_response"]),
                    str(row["normalized_reference"]),
                ),
                f"{job} token F1 recomputation",
            )
        rows_by_model[model] = rows
    require(set(rows_by_model) == set(models), "answer model coverage changed")

    metrics = {
        model: {
            system: source_macro(rows_by_model[model], system)
            for system in systems
        }
        for model in models
    }
    for model in models:
        for system in systems:
            observed = metrics[model][system]
            expected = report["metrics"][model][system]
            for name in (
                "source_macro_exact",
                "source_macro_token_f1",
            ):
                close(
                    float(observed[name]),
                    float(expected[name]),
                    f"answer/{model}/{system}/{name}",
                )
            require(
                int(observed["sources"]) == int(expected["sources"])
                and int(observed["examples"]) == int(expected["examples"]),
                f"answer/{model}/{system} coverage mismatch",
            )

    combined = {
        system: {
            "two_backbone_mean_source_macro_exact": float(
                np.mean(
                    [
                        metrics[model][system]["source_macro_exact"]
                        for model in models
                    ]
                )
            ),
            "two_backbone_mean_source_macro_token_f1": float(
                np.mean(
                    [
                        metrics[model][system]["source_macro_token_f1"]
                        for model in models
                    ]
                )
            ),
        }
        for system in systems
    }
    for system in systems:
        compare_metric_mapping(
            combined[system],
            report["combined_metrics"][system],
            f"answer combined/{system}",
        )

    eligible = list(config["eligible_nonoracle_retrieval_baselines"])
    strongest = sorted(
        eligible,
        key=lambda system: (
            -combined[system]["two_backbone_mean_source_macro_exact"],
            -combined[system]["two_backbone_mean_source_macro_token_f1"],
            system,
        ),
    )[0]
    require(
        strongest == report["strongest_eligible_nonoracle_baseline"],
        "strongest answer baseline changed",
    )
    replicates = int(config["bootstrap"]["replicates"])
    seed = int(config["bootstrap"]["seed"])
    learned = combined_source_values(
        rows_by_model,
        "selected_learned_retrieval",
        "exact",
        models,
    )
    for system in systems:
        if system == "selected_learned_retrieval":
            continue
        declared = report["comparisons"][system]
        compare_interval(
            paired_source_interval(
                learned,
                combined_source_values(
                    rows_by_model, system, "exact", models
                ),
                replicates=replicates,
                seed=seed,
            ),
            declared["combined_exact"],
            f"answer combined comparison/{system}",
        )
        for model in models:
            compare_interval(
                paired_source_interval(
                    source_values(
                        rows_by_model[model],
                        "selected_learned_retrieval",
                        "exact",
                    ),
                    source_values(rows_by_model[model], system, "exact"),
                    replicates=replicates,
                    seed=seed,
                ),
                declared["per_backbone_exact"][model],
                f"answer {model} comparison/{system}",
            )

    response_differences: dict[str, dict[str, float]] = {}
    for model in models:
        indexed = {
            (str(row["system"]), str(row["example_id"])): row
            for row in rows_by_model[model]
        }
        learned_rows = [
            row
            for row in rows_by_model[model]
            if row["system"] == "selected_learned_retrieval"
        ]
        response_differences[model] = {}
        for control in ("selected_retrieval_silenced", "text_only"):
            value = float(
                np.mean(
                    [
                        row["normalized_response"]
                        != indexed[
                            (control, str(row["example_id"]))
                        ]["normalized_response"]
                        for row in learned_rows
                    ]
                )
            )
            response_differences[model][control] = value
            close(
                value,
                report["response_difference_fractions"][model][control],
                f"answer {model}/{control} response difference",
            )

    learned_exact = combined["selected_learned_retrieval"][
        "two_backbone_mean_source_macro_exact"
    ]
    strongest_exact = combined[strongest][
        "two_backbone_mean_source_macro_exact"
    ]
    random_exact = combined["deterministic_random_retrieval"][
        "two_backbone_mean_source_macro_exact"
    ]
    gate_config = config["promotion_gates"]
    expected_gates = {
        "gain_at_least_0_02_over_strongest_baseline": learned_exact
        - strongest_exact
        >= float(gate_config["minimum_exact_gain_over_strongest_baseline"]),
        "gain_at_least_0_05_over_random": learned_exact - random_exact
        >= float(gate_config["minimum_exact_gain_over_random"]),
        "no_backbone_loses_more_than_0_02_to_strongest": all(
            float(
                metrics[model]["selected_learned_retrieval"][
                    "source_macro_exact"
                ]
            )
            - float(metrics[model][strongest]["source_macro_exact"])
            >= -float(
                gate_config[
                    "maximum_single_backbone_loss_to_strongest_baseline"
                ]
            )
            for model in models
        ),
        "responses_differ_from_silence": all(
            response_differences[model]["selected_retrieval_silenced"]
            >= float(
                gate_config[
                    "minimum_normalized_response_difference_fraction_from_silence_per_backbone"
                ]
            )
            for model in models
        ),
        "responses_differ_from_text_only": all(
            response_differences[model]["text_only"]
            >= float(
                gate_config[
                    "minimum_normalized_response_difference_fraction_from_text_only_per_backbone"
                ]
            )
            for model in models
        ),
        "learned_exact_no_worse_than_silence": all(
            float(
                metrics[model]["selected_learned_retrieval"][
                    "source_macro_exact"
                ]
            )
            >= float(
                metrics[model]["selected_retrieval_silenced"][
                    "source_macro_exact"
                ]
            )
            for model in models
        ),
        "all_responses_nonempty": True,
        "integrity": True,
    }
    require(
        expected_gates == report["promotion_gates"],
        "answer promotion gates do not recompute",
    )
    require(
        report["status"] == "heldout_answer_gate_fail_audita_sealed"
        and report["all_promotion_gates_pass"] is False
        and report["audita_status"] == "sealed",
        "answer decision boundary changed",
    )
    return {
        "status": report["status"],
        "raw_rows": sum(len(rows) for rows in rows_by_model.values()),
        "models": len(models),
        "systems": len(systems),
        "sources": int(report["integrity"]["sources"]),
        "selected_exact": learned_exact,
        "strongest_baseline": strongest,
        "strongest_exact": strongest_exact,
    }


def verify_release(release_root: Path) -> dict[str, Any]:
    shift = release_root / "studies/shifttitan/reproducibility"
    audio = release_root / "studies/eviaudio/reproducibility"
    require(shift.is_dir(), f"ShiftTitan reproducibility tree missing: {shift}")
    require(audio.is_dir(), f"EviAudio reproducibility tree missing: {audio}")
    shift_original = verify_shift_original(shift)
    shift_fresh = verify_shift_fresh(shift)
    controlled = verify_event_panel(
        audio,
        stage="controlled",
        report_relative=(
            "q1_plus/results/confirmatory/event_ranker/"
            "five_seed_confirmatory_report.json"
        ),
        raw_relative=(
            "q1_plus/results/confirmatory/event_ranker/"
            "raw_five_seed_confirmatory.jsonl.gz"
        ),
        expected_rows=397,
        external=False,
    )
    external = verify_event_panel(
        audio,
        stage="external",
        report_relative=(
            "q1_90/results/external_replication/evaluation/"
            "external_replication_report.json"
        ),
        raw_relative=(
            "q1_90/results/external_replication/evaluation/"
            "raw_external_replication.jsonl.gz"
        ),
        expected_rows=219,
        external=True,
    )
    natural = verify_natural_panel(audio)
    router = verify_router(audio)
    answers = verify_answer_bridge(audio)
    return {
        "status": "scientific_results_independently_recomputed",
        "studies": {
            "shifttitan": {
                "original_frozen_panel": shift_original,
                "fresh_task_disjoint_panel": shift_fresh,
            },
            "eviaudio": {
                "controlled_exact_onset": controlled,
                "external_source_disjoint": external,
                "frozen_natural": natural,
                "post_outcome_router": router,
                "heldout_answer_bridge": answers,
            },
        },
        "coverage": {
            "raw_forecast_origin_rows": shift_original["raw_origin_rows"]
            + shift_fresh["raw_origin_rows"],
            "raw_controlled_external_examples": controlled["raw_examples"]
            + external["raw_examples"],
            "natural_videos": natural["videos"],
            "final_localization_prediction_rows": natural["prediction_rows"]
            + router["prediction_rows"],
            "heldout_answer_rows": answers["raw_rows"],
            "exactly_recomputed_endpoints": [
                "both ShiftTitan cell matrices, geometric effects, counts, and task bootstraps",
                "controlled and external audio metrics, gates, and hierarchical bootstraps",
                "frozen natural per-video effect and paired bootstrap",
                "held-out answer exact/F1, source-macro effects, gates, and bootstraps",
            ],
            "artifact_arithmetic_only_endpoints": [
                "natural and router official global mAP: class/threshold arithmetic, hashes, caps, and all final predictions verified",
            ],
            "intentional_boundary": (
                "provider annotations/audio, model weights, large embeddings, "
                "and regenerable candidate pools are excluded; no unavailable "
                "ground-truth or inference rerun is claimed"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="root containing studies/shifttitan and studies/eviaudio",
    )
    args = parser.parse_args()
    result = verify_release(args.release_root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
