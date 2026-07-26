#!/usr/bin/env python3
"""Evaluate the frozen five-seed ensemble on exact-onset confirmatory data."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT = Path(__file__).resolve().parents[2]
Q1 = PROJECT / "q1_plus"
AUTHORIZATION = Q1 / "EVENT_CONFIRMATORY_AUTHORIZATION.json"
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(Q1 / "scripts"))

import analyze_event_ranker as analysis  # noqa: E402
from eviaudio_mt.event_data import (  # noqa: E402
    EventNeedleArchiveDataset,
    collate_event_needle,
)
from eviaudio_mt.event_needle import temporal_overlap_fraction  # noqa: E402
from eviaudio_mt.qcr import CLAPPriorResidualRanker  # noqa: E402


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
    if authorization["status"] != "one_shot_exact_onset_confirmatory_authorized":
        raise PermissionError("confirmatory event evaluation is not authorized")
    for relative, expected in authorization["files"].items():
        if sha256(project_path(relative)) != expected:
            raise RuntimeError(f"confirmatory authorization mismatch: {relative}")
    return authorization


def load_recipes(path: Path, expected_examples: int) -> list[dict[str, Any]]:
    recipes = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(recipes) != expected_examples:
        raise RuntimeError("confirmatory recipe count mismatch")
    identifiers = [str(recipe["recipe_id"]) for recipe in recipes]
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError("duplicate confirmatory recipe")
    if {str(recipe["panel"]) for recipe in recipes} != {"confirmatory"}:
        raise RuntimeError("non-confirmatory recipe in sealed panel")
    return recipes


def expected_chunk_bounds(
    duration_sec: float,
    *,
    sample_rate: int,
    chunk_seconds: float,
    hop_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    total = int(round(duration_sec * sample_rate))
    chunk = max(1, int(round(chunk_seconds * sample_rate)))
    hop = max(1, int(round(hop_seconds * sample_rate)))
    starts = list(range(0, max(total - chunk + 1, 1), hop))
    final_start = max(0, total - chunk)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    starts = sorted(set(starts))
    ends = [min(start + chunk, total) for start in starts]
    return (
        np.asarray(starts, dtype=np.float64) / sample_rate,
        np.asarray(ends, dtype=np.float64) / sample_rate,
    )


def verify_precompute_artifacts(
    authorization: dict[str, Any],
    recipes_path: Path,
    index_path: Path,
) -> dict[str, Any]:
    expected_examples = int(authorization["confirmatory_examples"])
    expected_output = project_path(authorization["precompute_output_dir"])
    if index_path.resolve() != expected_output / "index.jsonl":
        raise RuntimeError("wrong confirmatory index path")
    summary_path = expected_output / "clap_prior_summary.json"
    raw_prior_path = expected_output / "raw_clap_prior.jsonl.gz"
    receipt_path = expected_output / "confirmatory_precompute_receipt.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_summary = {
        "status": "confirmatory_embedding_complete",
        "panels": ["confirmatory"],
        "examples": expected_examples,
        "model_id": authorization["clap_model_id"],
        "model_revision": authorization["clap_model_revision"],
        "sample_rate": authorization["sample_rate"],
        "chunk_seconds": authorization["chunk_seconds"],
        "hop_seconds": authorization["hop_seconds"],
        "inference_batch_size": authorization["inference_batch_size"],
        "recipes_sha256": sha256(recipes_path),
        "index_sha256": sha256(index_path),
        "raw_sha256": sha256(raw_prior_path),
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise RuntimeError(f"confirmatory summary mismatch: {key}")
    expected_receipt = {
        "status": "one_shot_confirmatory_precompute_complete",
        "authorization_sha256": sha256(AUTHORIZATION),
        "recipes_sha256": sha256(recipes_path),
        "summary_sha256": sha256(summary_path),
        "index_sha256": sha256(index_path),
        "raw_sha256": sha256(raw_prior_path),
        "examples": expected_examples,
        "development_recipe_id_overlap": 0,
    }
    for key, expected in expected_receipt.items():
        if receipt.get(key) != expected:
            raise RuntimeError(f"confirmatory receipt mismatch: {key}")
    if int(receipt.get("unique_source_files_verified", 0)) <= 0:
        raise RuntimeError("source-file verification receipt is empty")

    recipes = load_recipes(recipes_path, expected_examples)
    recipe_by_id = {str(recipe["recipe_id"]): recipe for recipe in recipes}
    records = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != expected_examples:
        raise RuntimeError("confirmatory index count mismatch")
    record_ids = [str(record["recipe_id"]) for record in records]
    if record_ids != [str(recipe["recipe_id"]) for recipe in recipes]:
        raise RuntimeError("confirmatory index order or identity mismatch")
    archive_dir = expected_output / "archives"
    required_archive_keys = {
        "audio_embeddings",
        "question_embedding",
        "start_sec",
        "end_sec",
        "evidence_targets",
        "overlap_fraction",
        "recipe_id",
        "model_id",
        "model_revision",
    }
    total_chunks = 0
    for record in records:
        recipe_id = str(record["recipe_id"])
        recipe = recipe_by_id[recipe_id]
        expected_archive = (archive_dir / f"{recipe_id}.npz").resolve()
        if Path(record["archive_path"]).resolve() != expected_archive:
            raise RuntimeError(f"archive path mismatch: {recipe_id}")
        if Path(record["recipe_path"]).resolve() != recipes_path.resolve():
            raise RuntimeError(f"recipe path mismatch: {recipe_id}")
        if sha256(expected_archive) != str(record["archive_sha256"]):
            raise RuntimeError(f"archive checksum mismatch: {recipe_id}")
        expected_fields = {
            "panel": "confirmatory",
            "duration_sec": recipe["duration_sec"],
            "position_bin": recipe["position_bin"],
            "snr_db": recipe["target_event"]["snr_db"],
            "class_id": recipe["target_event"]["class_id"],
            "class_label": recipe["target_event"]["class_label"],
            "foreground_cluster_id": recipe["foreground_cluster_id"],
            "target_start_sec": recipe["target_event"]["start_sec"],
            "target_end_sec": recipe["target_event"]["end_sec"],
        }
        for key, expected in expected_fields.items():
            observed = record.get(key)
            if isinstance(expected, float):
                valid = math.isclose(float(observed), expected, abs_tol=1e-9)
            else:
                valid = observed == expected
            if not valid:
                raise RuntimeError(f"index metadata mismatch: {recipe_id}/{key}")

        with np.load(expected_archive, allow_pickle=False) as archive:
            if set(archive.files) != required_archive_keys:
                raise RuntimeError(f"archive schema mismatch: {recipe_id}")
            if str(archive["recipe_id"].item()) != recipe_id:
                raise RuntimeError(f"archive recipe mismatch: {recipe_id}")
            if str(archive["model_id"].item()) != authorization["clap_model_id"]:
                raise RuntimeError(f"archive model mismatch: {recipe_id}")
            if (
                str(archive["model_revision"].item())
                != authorization["clap_model_revision"]
            ):
                raise RuntimeError(f"archive revision mismatch: {recipe_id}")
            audio = archive["audio_embeddings"]
            question = archive["question_embedding"]
            starts = archive["start_sec"].astype(np.float64)
            ends = archive["end_sec"].astype(np.float64)
            evidence = archive["evidence_targets"].astype(bool)
            overlap = archive["overlap_fraction"].astype(np.float64)
            chunks = len(starts)
            if (
                audio.shape != (chunks, int(authorization["embedding_dim"]))
                or question.shape != (int(authorization["embedding_dim"]),)
                or ends.shape != (chunks,)
                or evidence.shape != (chunks,)
                or overlap.shape != (chunks,)
                or int(record["n_chunks"]) != chunks
            ):
                raise RuntimeError(f"archive array alignment mismatch: {recipe_id}")
            if not all(
                np.isfinite(array).all()
                for array in (audio, question, starts, ends, overlap)
            ):
                raise RuntimeError(f"nonfinite archive values: {recipe_id}")
            expected_starts, expected_ends = expected_chunk_bounds(
                float(recipe["duration_sec"]),
                sample_rate=int(authorization["sample_rate"]),
                chunk_seconds=float(authorization["chunk_seconds"]),
                hop_seconds=float(authorization["hop_seconds"]),
            )
            if not np.allclose(starts, expected_starts, atol=1e-6, rtol=0.0) or not np.allclose(
                ends, expected_ends, atol=1e-6, rtol=0.0
            ):
                raise RuntimeError(f"archive chunk timing mismatch: {recipe_id}")
            exact_overlap = temporal_overlap_fraction(
                starts,
                ends,
                float(recipe["target_event"]["start_sec"]),
                float(recipe["target_event"]["end_sec"]),
            )
            if not np.allclose(overlap, exact_overlap, atol=1e-6, rtol=0.0):
                raise RuntimeError(f"archive overlap mismatch: {recipe_id}")
            if not np.array_equal(evidence, exact_overlap > 0.0):
                raise RuntimeError(f"archive evidence-label mismatch: {recipe_id}")
            if not evidence.any() or evidence.all():
                raise RuntimeError(f"degenerate evidence labels: {recipe_id}")
            total_chunks += chunks

    with gzip.open(raw_prior_path, "rt", encoding="utf-8") as handle:
        raw_rows = [json.loads(line) for line in handle if line.strip()]
    if len(raw_rows) != expected_examples:
        raise RuntimeError("raw CLAP-prior row count mismatch")
    if [str(row["recipe_id"]) for row in raw_rows] != record_ids:
        raise RuntimeError("raw CLAP-prior identity mismatch")
    summary_metric_map = {
        "evidence_ap": "mean_evidence_ap",
        "hit_at_1": "hit_at_1",
        "recall_at_4": "recall_at_4",
        "top_chunk_iou": "mean_top_chunk_iou",
    }
    for raw_key, summary_key in summary_metric_map.items():
        observed = float(np.mean([float(row[raw_key]) for row in raw_rows]))
        if not math.isclose(observed, float(summary[summary_key]), abs_tol=1e-12):
            raise RuntimeError(f"raw CLAP-prior metric mismatch: {raw_key}")

    recipe_audit = json.loads(
        (recipes_path.parent / "recipe_audit.json").read_text(encoding="utf-8")
    )
    isolation = recipe_audit["source_isolation"]
    if any(
        int(isolation[key]) != 0
        for key in (
            "esc_original_source_overlap_across_folds",
            "development_validation_background_speaker_overlap",
            "validation_confirmatory_background_speaker_overlap",
        )
    ):
        raise RuntimeError("recipe source-isolation audit failed")
    return {
        "archives_verified": len(records),
        "total_chunks_verified": total_chunks,
        "unique_recipe_ids": len(set(record_ids)),
        "unique_source_files_verified": int(receipt["unique_source_files_verified"]),
        "source_manifest_sha256": receipt["source_manifest_sha256"],
        "recipe_ids_sha256": receipt["recipe_ids_sha256"],
        "development_recipe_id_overlap": 0,
        "source_isolation_failures": 0,
        "archive_checksum_failures": 0,
        "archive_schema_failures": 0,
        "chunk_alignment_failures": 0,
        "evidence_alignment_failures": 0,
        "nonfinite_archive_values": 0,
        "summary_sha256": sha256(summary_path),
        "receipt_sha256": sha256(receipt_path),
        "raw_prior_sha256": sha256(raw_prior_path),
        "summary": summary,
    }


def load_models(
    config: dict[str, Any], development_report: dict[str, Any], device: torch.device
) -> list[torch.nn.Module]:
    seeds = tuple(int(seed) for seed in config["seeds"])
    if seeds != (0, 1, 2, 3, 4):
        raise RuntimeError("confirmatory ensemble must contain frozen seeds 0--4")
    models: list[torch.nn.Module] = []
    report_by_seed = {int(item["seed"]): item for item in development_report["per_seed"]}
    if set(report_by_seed) != set(seeds):
        raise RuntimeError("development report seed mismatch")
    for seed in seeds:
        checkpoint = (
            Q1
            / "results/development/event_ranker/checkpoints"
            / f"{config['run_tag']}__prior_residual__seed_{seed}.pt"
        )
        if sha256(checkpoint) != report_by_seed[seed]["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint changed for seed {seed}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if int(payload["seed"]) != seed or payload["config"] != config:
            raise RuntimeError(f"checkpoint protocol mismatch for seed {seed}")
        model = CLAPPriorResidualRanker(
            embedding_dim=int(config["embedding_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            dropout=float(config["dropout"]),
            maximum_residual=float(config["maximum_residual"]),
        ).to(device)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        models.append(model)
    return models


@torch.inference_mode()
def evaluate(
    models: list[torch.nn.Module],
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[dict[str, Any]], float, float]:
    rows: list[dict[str, Any]] = []
    maximum_prior_disagreement = 0.0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    for batch in loader:
        audio = batch["audio_embeddings"].to(device)
        question = batch["question_embeddings"].to(device)
        mask = batch["audio_attention_mask"].to(device)
        outputs = [model(audio, question, mask) for model in models]
        reference_prior = outputs[0].prior_scores
        maximum_prior_disagreement = max(
            maximum_prior_disagreement,
            max(
                float(torch.max(torch.abs(output.prior_scores - reference_prior)).item())
                for output in outputs[1:]
            ),
        )
        stacked = torch.stack([output.scores for output in outputs])
        ensemble_scores = stacked.mean(dim=0).cpu().numpy()
        seed_scores = stacked.cpu().numpy()
        prior_scores = reference_prior.cpu().numpy()
        for index in range(mask.shape[0]):
            valid = batch["audio_attention_mask"][index].numpy().astype(bool)
            base: dict[str, Any] = {
                key: batch[key][index]
                for key in (
                    "recipe_id",
                    "panel",
                    "duration_sec",
                    "position_bin",
                    "snr_db",
                    "class_id",
                    "class_label",
                    "foreground_cluster_id",
                )
            }
            if str(base["panel"]) != "confirmatory":
                raise RuntimeError("non-confirmatory row in confirmatory index")
            base.update(
                {
                    "n_chunks": int(valid.sum()),
                    "target_start_sec": float(batch["target_start_sec"][index]),
                    "target_end_sec": float(batch["target_end_sec"][index]),
                    "evidence_targets": batch["evidence_targets"][index]
                    .numpy()[valid]
                    .astype(int)
                    .tolist(),
                    "chunk_start_sec": batch["start_sec"][index]
                    .numpy()[valid]
                    .astype(float)
                    .tolist(),
                    "chunk_end_sec": batch["end_sec"][index]
                    .numpy()[valid]
                    .astype(float)
                    .tolist(),
                }
            )
            current_ensemble = ensemble_scores[index][valid]
            current_prior = prior_scores[index][valid]
            current_seeds = seed_scores[:, index, :][:, valid]
            base["ensemble_chunk_scores"] = current_ensemble.astype(float).tolist()
            base["prior_chunk_scores"] = current_prior.astype(float).tolist()
            base["seed_chunk_scores"] = current_seeds.astype(float).tolist()
            base["ensemble"] = analysis.score_metrics(base, current_ensemble)
            base["prior"] = analysis.score_metrics(base, current_prior)
            base["seed_metrics"] = [
                analysis.score_metrics(base, values) for values in current_seeds
            ]
            base["mean_absolute_residual"] = float(
                np.mean(np.abs(current_ensemble - current_prior))
            )
            base["maximum_absolute_residual"] = float(
                np.max(np.abs(current_seeds - current_prior[None, :]))
            )
            rows.append(base)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return rows, time.perf_counter() - started, maximum_prior_disagreement


def write_raw_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as binary:
        with gzip.GzipFile(filename="", mode="wb", fileobj=binary, mtime=0) as compressed:
            for row in rows:
                serializable = {
                    key: value
                    for key, value in row.items()
                    if key not in {"ensemble", "prior", "seed_metrics"}
                }
                serializable.update(
                    {
                        "ensemble_metrics": row["ensemble"],
                        "prior_metrics": row["prior"],
                        "seed_metrics": row["seed_metrics"],
                    }
                )
                compressed.write(
                    (json.dumps(serializable, sort_keys=True) + "\n").encode("utf-8")
                )
    os.replace(temporary, path)


def main() -> None:
    if len(sys.argv) != 1:
        raise PermissionError("the confirmatory evaluator accepts no command-line overrides")
    authorization = audit_authorization()
    config_path = project_path(authorization["config"])
    recipes_path = project_path(authorization["confirmatory_recipes"])
    index_path = project_path(authorization["confirmatory_index"])
    output_dir = project_path(authorization["evaluation_output_dir"])
    report_path = output_dir / "five_seed_confirmatory_report.json"
    raw_path = output_dir / "raw_five_seed_confirmatory.jsonl.gz"
    if report_path.exists() or raw_path.exists():
        raise FileExistsError("confirmatory result is immutable and already exists")

    precompute_audit = verify_precompute_artifacts(
        authorization, recipes_path, index_path
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if sha256(config_path) != authorization["config_sha256"]:
        raise RuntimeError("confirmatory config changed")
    development_report_path = project_path(authorization["development_report"])
    development_report = json.loads(
        development_report_path.read_text(encoding="utf-8")
    )
    if development_report["status"] != (
        "development_pass_exact_onset_confirmatory_authorized_once"
    ):
        raise PermissionError("development report did not authorize confirmatory scoring")
    if sha256(development_report_path) != authorization["development_report_sha256"]:
        raise RuntimeError("development report changed after authorization")

    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(config, development_report, device)
    dataset = EventNeedleArchiveDataset(index_path)
    if len(dataset) != int(authorization["confirmatory_examples"]):
        raise RuntimeError("confirmatory dataset length mismatch")
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_event_needle,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    rows, wall_seconds, maximum_prior_disagreement = evaluate(models, loader, device)
    if len(rows) != int(authorization["confirmatory_examples"]):
        raise RuntimeError("confirmatory evaluation row count mismatch")
    ensemble = analysis.mean_metrics(rows, "ensemble")
    prior = analysis.mean_metrics(rows, "prior")
    delta = {name: ensemble[name] - prior[name] for name in ensemble}
    bootstrap = analysis.hierarchical_bootstrap(rows)
    conditions = analysis.condition_summary(rows)
    seed_results = []
    for seed_index, seed in enumerate(config["seeds"]):
        seed_metrics = {
            name: float(
                np.mean([row["seed_metrics"][seed_index][name] for row in rows])
            )
            for name in ensemble
        }
        seed_results.append(
            {
                "seed": int(seed),
                "model": seed_metrics,
                "delta": {
                    name: seed_metrics[name] - prior[name] for name in seed_metrics
                },
            }
        )
    positive_seeds = sum(
        item["delta"]["hit_at_1"] > 0.0
        and item["delta"]["evidence_ap"] > 0.0
        for item in seed_results
    )
    condition_hit_deltas = [
        group["delta"]["hit_at_1"]
        for condition in conditions.values()
        for group in condition.values()
    ]
    maximum_residual = max(row["maximum_absolute_residual"] for row in rows)
    quantized_prior_difference = {
        "evidence_ap": prior["evidence_ap"]
        - float(precompute_audit["summary"]["mean_evidence_ap"]),
        "hit_at_1": prior["hit_at_1"]
        - float(precompute_audit["summary"]["hit_at_1"]),
        "recall_at_4": prior["recall_at_4"]
        - float(precompute_audit["summary"]["recall_at_4"]),
        "top_chunk_iou": prior["top_chunk_iou"]
        - float(precompute_audit["summary"]["mean_top_chunk_iou"]),
    }
    if max(abs(value) for value in quantized_prior_difference.values()) > 0.01:
        raise RuntimeError("stored-embedding CLAP prior does not reproduce precompute")
    if maximum_prior_disagreement != 0.0:
        raise RuntimeError("frozen models produced different CLAP priors")
    if maximum_residual > float(config["maximum_residual"]) + 1e-6:
        raise RuntimeError("ranker exceeded its frozen residual bound")
    integrity = {
        **{key: value for key, value in precompute_audit.items() if key != "summary"},
        "all_checksums_valid": True,
        "alignment_failures": 0,
        "nonfinite_scores": 0,
        "examples_expected": int(authorization["confirmatory_examples"]),
        "examples_observed": len(rows),
        "maximum_prior_disagreement": maximum_prior_disagreement,
        "maximum_absolute_seed_residual": maximum_residual,
        "residual_bound": float(config["maximum_residual"]),
        "precompute_to_float16_prior_difference": quantized_prior_difference,
    }
    gates = {
        "ensemble_hit_gain_at_least_0_05": delta["hit_at_1"] >= 0.05,
        "ensemble_ap_gain_at_least_0_05": delta["evidence_ap"] >= 0.05,
        "bootstrap_hit_lower_above_zero": bootstrap["hit_at_1"][0] > 0.0,
        "bootstrap_ap_lower_above_zero": bootstrap["evidence_ap"][0] > 0.0,
        "at_least_four_positive_seeds": positive_seeds >= 4,
        "no_condition_hit_loss_over_0_05": min(condition_hit_deltas) >= -0.05,
        "integrity": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_raw_rows(raw_path, rows)
    payload = {
        "status": (
            "confirmatory_exact_onset_gate_pass"
            if all(gates.values())
            else "confirmatory_exact_onset_gate_fail"
        ),
        "examples": len(rows),
        "ensemble": ensemble,
        "prior": prior,
        "delta": delta,
        "per_seed": seed_results,
        "positive_seeds": positive_seeds,
        "bootstrap": {
            "seed": analysis.BOOTSTRAP_SEED,
            "replicates": analysis.BOOTSTRAP_REPLICATES,
            "interval_95_percentile": bootstrap,
        },
        "conditions": conditions,
        "worst_condition_hit_delta": min(condition_hit_deltas),
        "promotion_gates": gates,
        "all_promotion_gates_pass": all(gates.values()),
        "mean_absolute_ensemble_residual": float(
            np.mean([row["mean_absolute_residual"] for row in rows])
        ),
        "maximum_absolute_seed_residual": maximum_residual,
        "wall_seconds": wall_seconds,
        "examples_per_second": len(rows) / wall_seconds,
        "peak_cuda_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        ),
        "device": str(device),
        "torch": torch.__version__,
        "config_sha256": sha256(config_path),
        "recipes_sha256": sha256(recipes_path),
        "index_sha256": sha256(index_path),
        "development_report_sha256": sha256(development_report_path),
        "authorization_sha256": sha256(AUTHORIZATION),
        "raw_sha256": sha256(raw_path),
        "integrity": integrity,
    }
    temporary_report = report_path.with_name(report_path.name + ".tmp")
    temporary_report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_report, report_path)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
