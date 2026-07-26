#!/usr/bin/env python3
"""Evaluate the frozen five-seed ranker once on the external audio panel."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT = Path(__file__).resolve().parents[2]
TRACK = PROJECT / "q1_90"
AUTHORIZATION = TRACK / "EXTERNAL_REPLICATION_AUTHORIZATION.json"
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "q1_plus/scripts"))

import analyze_event_ranker as analysis  # noqa: E402
import evaluate_event_ranker_confirmatory as original  # noqa: E402
from eviaudio_mt.event_data import EventNeedleArchiveDataset, collate_event_needle  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def project_path(relative: str) -> Path:
    path = (PROJECT / relative).resolve()
    if not path.is_relative_to(PROJECT):
        raise RuntimeError(f"path leaves project: {relative}")
    return path


def audit_authorization() -> dict[str, Any]:
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    if authorization["status"] != "one_shot_external_precompute_and_evaluation_authorized":
        raise PermissionError("external evaluation is not authorized")
    for relative, expected in authorization["files"].items():
        if sha256(project_path(relative)) != expected:
            raise RuntimeError(f"external authorization mismatch: {relative}")
    return authorization


def verify_precompute(authorization: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    output = project_path(authorization["precompute_output_dir"])
    receipt_path = output / "external_precompute_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    summary_path = output / "clap_prior_summary.json"
    index_path = output / "index.jsonl"
    raw_path = output / "raw_clap_prior.jsonl.gz"
    expected = {
        "status": "one_shot_external_precompute_complete",
        "authorization_sha256": sha256(AUTHORIZATION),
        "recipes_sha256": sha256(project_path(authorization["recipes"])),
        "summary_sha256": sha256(summary_path),
        "index_sha256": sha256(index_path),
        "raw_sha256": sha256(raw_path),
        "examples": int(authorization["examples"]),
        "archives_verified": int(authorization["examples"]),
        "source_manifest_sha256": authorization["source_integrity"]["source_manifest_sha256"],
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"external precompute receipt mismatch: {key}")
    if any(int(receipt[key]) != 0 for key in ("archive_checksum_failures", "source_checksum_failures", "alignment_failures", "nonfinite_failures")):
        raise RuntimeError("external precompute integrity failure")
    return index_path, receipt


def hierarchical_bootstrap(
    rows: list[dict[str, Any]], *, seed: int, replicates: int
) -> dict[str, list[float]]:
    hierarchy: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        hierarchy[str(row["class_id"])][str(row["foreground_cluster_id"])].append(index)
    classes = sorted(hierarchy)
    delta_ap = np.asarray([row["ensemble"]["evidence_ap"] - row["prior"]["evidence_ap"] for row in rows])
    delta_hit = np.asarray([row["ensemble"]["hit_at_1"] - row["prior"]["hit_at_1"] for row in rows])
    rng = np.random.default_rng(seed)
    sampled_ap = np.empty(replicates)
    sampled_hit = np.empty(replicates)
    for replicate in range(replicates):
        selected: list[int] = []
        for class_index in rng.integers(0, len(classes), size=len(classes)):
            class_id = classes[int(class_index)]
            clusters = sorted(hierarchy[class_id])
            for cluster_index in rng.integers(0, len(clusters), size=len(clusters)):
                recipes = hierarchy[class_id][clusters[int(cluster_index)]]
                sampled = rng.integers(0, len(recipes), size=len(recipes))
                selected.extend(recipes[int(value)] for value in sampled)
        sampled_ap[replicate] = float(delta_ap[selected].mean())
        sampled_hit[replicate] = float(delta_hit[selected].mean())
    return {
        "evidence_ap": np.quantile(sampled_ap, [0.025, 0.975]).astype(float).tolist(),
        "hit_at_1": np.quantile(sampled_hit, [0.025, 0.975]).astype(float).tolist(),
    }


def class_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["class_label"])].append(row)
    result = {}
    for label, group in sorted(groups.items()):
        ensemble = analysis.mean_metrics(group, "ensemble")
        prior = analysis.mean_metrics(group, "prior")
        result[label] = {
            "examples": len(group),
            "ensemble": ensemble,
            "prior": prior,
            "delta": {name: ensemble[name] - prior[name] for name in ensemble},
        }
    return result


def main() -> None:
    if len(sys.argv) != 1:
        raise PermissionError("external evaluator accepts no overrides")
    authorization = audit_authorization()
    index_path, precompute_receipt = verify_precompute(authorization)
    output_dir = project_path(authorization["evaluation_output_dir"])
    report_path = output_dir / "external_replication_report.json"
    raw_path = output_dir / "raw_external_replication.jsonl.gz"
    if report_path.exists() or raw_path.exists():
        raise FileExistsError("external evaluation is immutable")
    config_path = project_path(authorization["ranker_config"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    development_path = project_path(authorization["development_report"])
    development = json.loads(development_path.read_text(encoding="utf-8"))
    if development["status"] != "development_pass_exact_onset_confirmatory_authorized_once":
        raise PermissionError("ranker development did not authorize evaluation")

    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = original.load_models(config, development, device)
    dataset = EventNeedleArchiveDataset(index_path)
    if len(dataset) != int(authorization["examples"]):
        raise RuntimeError("external dataset length mismatch")
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_event_needle,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    rows, inference_seconds, maximum_prior_disagreement = original.evaluate(models, loader, device)
    if len(rows) != int(authorization["examples"]):
        raise RuntimeError("external evaluation row count mismatch")
    ensemble = analysis.mean_metrics(rows, "ensemble")
    prior = analysis.mean_metrics(rows, "prior")
    delta = {name: ensemble[name] - prior[name] for name in ensemble}
    bootstrap = hierarchical_bootstrap(
        rows,
        seed=int(authorization["bootstrap"]["seed"]),
        replicates=int(authorization["bootstrap"]["replicates"]),
    )
    classes = class_summary(rows)
    conditions = analysis.condition_summary(rows)
    seed_results = []
    for seed_index, seed in enumerate(config["seeds"]):
        metrics = {
            name: float(np.mean([row["seed_metrics"][seed_index][name] for row in rows]))
            for name in ensemble
        }
        seed_results.append({"seed": int(seed), "model": metrics, "delta": {name: metrics[name] - prior[name] for name in metrics}})
    positive_seeds = sum(item["delta"]["hit_at_1"] > 0.0 and item["delta"]["evidence_ap"] > 0.0 for item in seed_results)
    maximum_residual = max(row["maximum_absolute_residual"] for row in rows)
    if maximum_prior_disagreement != 0.0 or maximum_residual > float(config["maximum_residual"]) + 1e-6:
        raise RuntimeError("frozen-ranker numerical invariant failed")
    performance_index = 50.0 * (ensemble["evidence_ap"] + ensemble["hit_at_1"])
    minimum_class_hit_delta = min(group["delta"]["hit_at_1"] for group in classes.values())
    gates_config = authorization["gates"]
    gates = {
        "ensemble_ap_gain_at_least_0_05": delta["evidence_ap"] >= float(gates_config["minimum_evidence_ap_gain"]),
        "ensemble_hit_gain_at_least_0_05": delta["hit_at_1"] >= float(gates_config["minimum_hit_at_1_gain"]),
        "bootstrap_ap_lower_above_zero": bootstrap["evidence_ap"][0] > 0.0,
        "bootstrap_hit_lower_above_zero": bootstrap["hit_at_1"][0] > 0.0,
        "at_least_four_positive_seeds": positive_seeds >= int(gates_config["minimum_positive_seeds"]),
        "no_class_hit_delta_below_minus_0_05": minimum_class_hit_delta >= float(gates_config["minimum_class_hit_delta"]),
        "external_performance_index_at_least_85": performance_index >= float(gates_config["minimum_external_performance_index"]),
        "integrity": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    original.write_raw_rows(raw_path, rows)
    report = {
        "status": "external_replication_gate_pass" if all(gates.values()) else "external_replication_gate_fail",
        "examples": len(rows),
        "ensemble": ensemble,
        "prior": prior,
        "delta": delta,
        "external_performance_index": performance_index,
        "per_seed": seed_results,
        "positive_seeds": positive_seeds,
        "bootstrap": {
            "seed": int(authorization["bootstrap"]["seed"]),
            "replicates": int(authorization["bootstrap"]["replicates"]),
            "unit": authorization["bootstrap"]["unit"],
            "interval_95_percentile": bootstrap,
        },
        "classes": classes,
        "minimum_class_hit_delta": minimum_class_hit_delta,
        "conditions": conditions,
        "promotion_gates": gates,
        "all_promotion_gates_pass": all(gates.values()),
        "inference_seconds": inference_seconds,
        "total_wall_seconds": time.perf_counter() - started,
        "device": str(device),
        "torch": torch.__version__,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "authorization_sha256": sha256(AUTHORIZATION),
        "config_sha256": sha256(config_path),
        "development_report_sha256": sha256(development_path),
        "index_sha256": sha256(index_path),
        "precompute_receipt_sha256": sha256(project_path(authorization["precompute_output_dir"]) / "external_precompute_receipt.json"),
        "raw_sha256": sha256(raw_path),
        "integrity": {
            "external_evaluations": 1,
            "examples_expected": int(authorization["examples"]),
            "examples_observed": len(rows),
            "precompute_archives_verified": int(precompute_receipt["archives_verified"]),
            "maximum_prior_disagreement": maximum_prior_disagreement,
            "maximum_absolute_seed_residual": maximum_residual,
            "residual_bound": float(config["maximum_residual"]),
            "checksum_failures": 0,
            "alignment_failures": 0,
            "nonfinite_scores": 0,
        },
    }
    temporary = report_path.with_name(f"{report_path.name}.partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
