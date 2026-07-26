#!/usr/bin/env python3
"""Select one frozen answer retriever on source-held-out Clotho calibration."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT = Path(__file__).resolve().parents[2]
Q1 = PROJECT / "q1_plus"
sys.path.insert(0, str(PROJECT / "src"))

from eviaudio_mt.qcr import CLAPPriorResidualRanker  # noqa: E402
from eviaudio_mt.qcr_data import QCRManifestDataset, collate_qcr  # noqa: E402


CONFIG = Q1 / "configs/answer_development.json"
MANIFEST = PROJECT / "journal_suite/data/manifests/val.jsonl"
QUESTION_ARCHIVE = Q1 / "data/development_question_clap.npz"
RESULTS = Q1 / "results/development/answer_retrieval"
REPORT = RESULTS / "selection_report.json"
RAW = RESULTS / "retrieval_scores.jsonl.gz"
QCR_RAW = tuple(
    Q1
    / f"results/development/raw_validation__prior_residual__temp_010_ev010__seed_{seed}.jsonl.gz"
    for seed in range(5)
)
EVENT_CHECKPOINTS = tuple(
    Q1
    / f"results/development/event_ranker/checkpoints/event_v1__prior_residual__seed_{seed}.pt"
    for seed in range(5)
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def source_split(
    rows: list[dict[str, Any]], salt: str, calibration_count: int
) -> tuple[set[str], set[str]]:
    sources = sorted({str(row["target_source_id"]) for row in rows})
    ranked = sorted(
        sources,
        key=lambda source: hashlib.sha256(f"{salt}|{source}".encode()).hexdigest(),
    )
    if len(ranked) != 49 or calibration_count != 24:
        raise RuntimeError("unexpected Clotho target-source contract")
    return set(ranked[:calibration_count]), set(ranked[calibration_count:])


def load_qcr_ensemble(
    expected_ids: set[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, list[int]], dict[str, list[int]]]:
    seed_rows: list[dict[str, dict[str, Any]]] = []
    for path in QCR_RAW:
        current: dict[str, dict[str, Any]] = {}
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                identifier = str(row["example_id"])
                if identifier in current:
                    raise RuntimeError(f"duplicate QCR row: {identifier}")
                current[identifier] = row
        if set(current) != expected_ids:
            raise RuntimeError(f"QCR row coverage mismatch: {path}")
        seed_rows.append(current)
    ensemble: dict[str, np.ndarray] = {}
    priors: dict[str, np.ndarray] = {}
    source_indices: dict[str, list[int]] = {}
    evidence: dict[str, list[int]] = {}
    for identifier in sorted(expected_ids):
        first = seed_rows[0][identifier]
        prior = np.asarray(first["prior_chunk_scores"], dtype=np.float64)
        scores = []
        for rows in seed_rows:
            row = rows[identifier]
            current_prior = np.asarray(row["prior_chunk_scores"], dtype=np.float64)
            current_scores = np.asarray(row["model_chunk_scores"], dtype=np.float64)
            if not np.array_equal(current_prior, prior) or current_scores.shape != prior.shape:
                raise RuntimeError(f"QCR seed alignment mismatch: {identifier}")
            if row["chunk_source_indices"] != first["chunk_source_indices"]:
                raise RuntimeError(f"QCR source alignment mismatch: {identifier}")
            scores.append(current_scores)
        ensemble[identifier] = np.mean(np.stack(scores), axis=0)
        priors[identifier] = prior
        source_indices[identifier] = [int(value) for value in first["chunk_source_indices"]]
        evidence[identifier] = [int(value) for value in first["evidence_targets"]]
    return ensemble, priors, source_indices, evidence


@torch.inference_mode()
def load_event_ensemble(expected_ids: set[str]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for path in EVENT_CHECKPOINTS:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        model = CLAPPriorResidualRanker(
            embedding_dim=int(config["embedding_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            dropout=float(config["dropout"]),
            maximum_residual=float(config["maximum_residual"]),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        models.append(model)
    loader = DataLoader(
        QCRManifestDataset(MANIFEST, QUESTION_ARCHIVE),
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_qcr,
    )
    ensemble: dict[str, np.ndarray] = {}
    priors: dict[str, np.ndarray] = {}
    for batch in loader:
        audio = batch["audio_embeddings"].to(device)
        questions = batch["question_embeddings"].to(device)
        mask = batch["audio_attention_mask"].to(device)
        outputs = [model(audio, questions, mask) for model in models]
        averaged = torch.stack([output.scores for output in outputs]).mean(dim=0)
        prior = outputs[0].prior_scores
        for index, identifier_value in enumerate(batch["example_id"]):
            identifier = str(identifier_value)
            valid = mask[index].cpu().numpy().astype(bool)
            if identifier in ensemble:
                raise RuntimeError(f"duplicate event-ranker row: {identifier}")
            ensemble[identifier] = averaged[index].float().cpu().numpy()[valid]
            priors[identifier] = prior[index].float().cpu().numpy()[valid]
    if set(ensemble) != expected_ids:
        raise RuntimeError("event-ranker row coverage mismatch")
    return ensemble, priors


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives <= 0:
        raise RuntimeError("retrieval example lacks positive evidence")
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    return float((precision * sorted_labels).sum() / positives)


def logmeanexp(values: np.ndarray) -> float:
    maximum = float(values.max())
    return maximum + math.log(float(np.exp(values - maximum).mean()))


def example_metrics(
    scores: np.ndarray, source_indices: np.ndarray, target_source: int, labels: np.ndarray
) -> dict[str, float]:
    count = int(source_indices.max()) + 1
    source_scores = np.asarray(
        [logmeanexp(scores[source_indices == index]) for index in range(count)]
    )
    order = np.argsort(-source_scores, kind="stable")
    rank = int(np.flatnonzero(order == target_source)[0]) + 1
    return {
        "source_correct": float(rank == 1),
        "source_mrr": 1.0 / rank,
        "evidence_ap": average_precision(scores, labels),
    }


def source_macro(rows: list[dict[str, Any]], candidate: str) -> dict[str, float]:
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["target_source_id"])].append(row["metrics"][candidate])
    return {
        metric: float(
            np.mean(
                [
                    np.mean([item[metric] for item in items])
                    for items in grouped.values()
                ]
            )
        )
        for metric in ("source_correct", "evidence_ap", "source_mrr")
    } | {"sources": len(grouped), "examples": len(rows)}


def main() -> None:
    if REPORT.exists() or RAW.exists():
        raise FileExistsError("answer-retrieval selection outputs are immutable")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["status"] != "frozen_before_any_audio_backbone_clotho_answer_generation":
        raise RuntimeError("answer-development config is not frozen")
    manifest_rows = read_jsonl(MANIFEST)
    by_id = {str(row["example_id"]): row for row in manifest_rows}
    if len(by_id) != 804 or len(by_id) != len(manifest_rows):
        raise RuntimeError("unexpected or duplicate Clotho validation manifest")
    calibration_sources, evaluation_sources = source_split(
        manifest_rows,
        str(config["split"]["hash_salt"]),
        int(config["split"]["calibration_sources"]),
    )
    qcr, priors, source_indices, evidence = load_qcr_ensemble(set(by_id))
    event, event_priors = load_event_ensemble(set(by_id))
    rows: list[dict[str, Any]] = []
    maximum_prior_difference = 0.0
    for identifier, manifest in sorted(by_id.items()):
        prior = priors[identifier]
        maximum_prior_difference = max(
            maximum_prior_difference,
            float(np.max(np.abs(prior - event_priors[identifier]))),
        )
        indices = np.asarray(source_indices[identifier], dtype=np.int64)
        labels = np.asarray(evidence[identifier], dtype=np.int64)
        if len(prior) != int(manifest["n_chunks"]) or len(indices) != len(prior):
            raise RuntimeError(f"manifest/chunk alignment mismatch: {identifier}")
        target = int(manifest["target_position"])
        score_map = {
            "clap_prior": prior,
            "qcr_temp_010_ev010_five_seed_mean": qcr[identifier],
            "event_v1_prior_residual_five_seed_mean": event[identifier],
        }
        rows.append(
            {
                "example_id": identifier,
                "target_source_id": str(manifest["target_source_id"]),
                "split": (
                    "calibration"
                    if str(manifest["target_source_id"]) in calibration_sources
                    else "evaluation"
                ),
                "n_sources": int(manifest["n_sources"]),
                "difficulty": str(manifest["difficulty"]),
                "chunk_source_indices": indices.tolist(),
                "evidence_targets": labels.tolist(),
                "scores": {
                    name: values.astype(float).tolist() for name, values in score_map.items()
                },
                "metrics": {
                    name: example_metrics(values, indices, target, labels)
                    for name, values in score_map.items()
                },
            }
        )
    if maximum_prior_difference > 1e-6:
        raise RuntimeError("CLAP priors differ across frozen ranker families")
    calibration = [row for row in rows if row["split"] == "calibration"]
    evaluation = [row for row in rows if row["split"] == "evaluation"]
    names = list(rows[0]["scores"])
    calibration_metrics = {name: source_macro(calibration, name) for name in names}
    evaluation_metrics = {name: source_macro(evaluation, name) for name in names}
    baseline = calibration_metrics["clap_prior"]
    learned = []
    requirements = config["retrieval"]["required_to_select_learned_ranker"]
    for name in config["retrieval"]["ranker_candidates"]:
        metrics = calibration_metrics[name]
        if (
            metrics["source_correct"] - baseline["source_correct"]
            >= float(requirements["minimum_source_accuracy_delta_over_clap"])
            and metrics["evidence_ap"] - baseline["evidence_ap"]
            >= float(requirements["minimum_evidence_ap_delta_over_clap"])
        ):
            learned.append(name)
    selected = (
        sorted(
            learned,
            key=lambda name: (
                -calibration_metrics[name]["source_correct"],
                -calibration_metrics[name]["evidence_ap"],
                -calibration_metrics[name]["source_mrr"],
                name,
            ),
        )[0]
        if learned
        else "clap_prior"
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    with gzip.open(RAW, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    report = {
        "status": (
            "learned_answer_retriever_selected"
            if selected != "clap_prior"
            else "learned_answer_retriever_gate_fail_clap_fallback"
        ),
        "selected_retriever": selected,
        "config_sha256": sha256(CONFIG),
        "manifest_sha256": sha256(MANIFEST),
        "question_archive_sha256": sha256(QUESTION_ARCHIVE),
        "calibration_source_ids_sha256": hashlib.sha256(
            "\n".join(sorted(calibration_sources)).encode()
        ).hexdigest(),
        "evaluation_source_ids_sha256": hashlib.sha256(
            "\n".join(sorted(evaluation_sources)).encode()
        ).hexdigest(),
        "calibration": calibration_metrics,
        "evaluation_after_frozen_selection": evaluation_metrics,
        "raw_path": str(RAW.relative_to(PROJECT)),
        "raw_sha256": sha256(RAW),
        "checkpoints": {
            str(path.relative_to(PROJECT)): sha256(path)
            for path in (*QCR_RAW, *EVENT_CHECKPOINTS)
        },
        "integrity": {
            "examples": len(rows),
            "calibration_examples": len(calibration),
            "evaluation_examples": len(evaluation),
            "calibration_sources": len(calibration_sources),
            "evaluation_sources": len(evaluation_sources),
            "target_source_overlap": len(calibration_sources & evaluation_sources),
            "maximum_prior_difference": maximum_prior_difference,
            "alignment_failures": 0,
            "nonfinite_failures": 0,
            "audita_rows_accessed": 0,
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
