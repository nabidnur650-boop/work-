#!/usr/bin/env python3
"""Train a development-only exact-onset evidence ranker against frozen CLAP."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT = Path(__file__).resolve().parents[2]
SELECTION_LOCK = PROJECT / "q1_plus" / "EVENT_RANKER_LOCK.json"
sys.path.insert(0, str(PROJECT / "src"))

from eviaudio_mt.event_data import (  # noqa: E402
    EventNeedleArchiveDataset,
    collate_event_needle,
)
from eviaudio_mt.qcr import CLAPPriorResidualRanker, qcr_evidence_loss  # noqa: E402


def sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            checksum.update(block)
    return checksum.hexdigest()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT / path


def audit_selection_lock() -> dict[str, Any]:
    lock = json.loads(SELECTION_LOCK.read_text(encoding="utf-8"))
    for relative, expected in lock["files"].items():
        path = PROJECT / relative
        if sha256(path) != expected:
            raise RuntimeError(f"event-ranker lock mismatch: {relative}")
    if lock["confirmatory_panel_status"] != "sealed":
        raise RuntimeError("exact-onset confirmatory panel is not sealed")
    return lock


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order].astype(np.float64)
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    return float((precision * sorted_labels).sum() / positives)


def interval_iou(start: float, end: float, target_start: float, target_end: float) -> float:
    intersection = max(0.0, min(end, target_end) - max(start, target_start))
    union = max(end, target_end) - min(start, target_start)
    return float(intersection / union) if union > 0.0 else 0.0


def metric_summary(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    return {
        "evidence_ap": float(np.mean([row[f"{prefix}_evidence_ap"] for row in rows])),
        "hit_at_1": float(np.mean([row[f"{prefix}_hit_at_1"] for row in rows])),
        "recall_at_4": float(np.mean([row[f"{prefix}_recall_at_4"] for row in rows])),
        "top_chunk_iou": float(
            np.mean([row[f"{prefix}_top_chunk_iou"] for row in rows])
        ),
    }


def condition_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for condition in ("duration_sec", "position_bin", "snr_db"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row[condition])].append(row)
        output[condition] = {}
        for value, group in sorted(groups.items()):
            model = metric_summary(group, "model")
            prior = metric_summary(group, "prior")
            output[condition][value] = {
                "examples": len(group),
                "model": model,
                "prior": prior,
                "delta": {key: model[key] - prior[key] for key in model},
            }
    return output


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    write_rows: Path | None = None,
) -> dict[str, Any]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        mask = batch["audio_attention_mask"].to(device)
        output = model(
            batch["audio_embeddings"].to(device),
            batch["question_embeddings"].to(device),
            mask,
        )
        for index in range(mask.shape[0]):
            valid = mask[index].cpu().numpy().astype(bool)
            evidence = batch["evidence_targets"][index].numpy()[valid].astype(bool)
            starts = batch["start_sec"][index].numpy()[valid]
            ends = batch["end_sec"][index].numpy()[valid]
            target_start = float(batch["target_start_sec"][index])
            target_end = float(batch["target_end_sec"][index])
            model_scores = output.scores[index].cpu().numpy()[valid]
            prior_scores = output.prior_scores[index].cpu().numpy()[valid]
            row: dict[str, Any] = {
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
            row.update(
                {
                    "n_chunks": int(valid.sum()),
                    "target_start_sec": target_start,
                    "target_end_sec": target_end,
                    "evidence_targets": evidence.astype(int).tolist(),
                    "chunk_start_sec": starts.astype(float).tolist(),
                    "chunk_end_sec": ends.astype(float).tolist(),
                    "model_chunk_scores": model_scores.astype(float).tolist(),
                    "prior_chunk_scores": prior_scores.astype(float).tolist(),
                    "mean_absolute_residual": float(
                        output.residual_scores[index][mask[index]].abs().mean().cpu()
                    ),
                }
            )
            for prefix, scores in (("model", model_scores), ("prior", prior_scores)):
                order = np.argsort(-scores, kind="stable")
                top = int(order[0])
                row[f"{prefix}_evidence_ap"] = average_precision(scores, evidence)
                row[f"{prefix}_hit_at_1"] = int(evidence[top])
                row[f"{prefix}_recall_at_4"] = int(evidence[order[:4]].any())
                row[f"{prefix}_top_chunk_iou"] = interval_iou(
                    float(starts[top]),
                    float(ends[top]),
                    target_start,
                    target_end,
                )
            rows.append(row)
    model_metrics = metric_summary(rows, "model")
    prior_metrics = metric_summary(rows, "prior")
    result = {
        "examples": len(rows),
        "model": model_metrics,
        "prior": prior_metrics,
        "delta": {
            key: model_metrics[key] - prior_metrics[key] for key in model_metrics
        },
        "mean_absolute_residual": float(
            np.mean([row["mean_absolute_residual"] for row in rows])
        ),
        "conditions": condition_summaries(rows),
    }
    if write_rows is not None:
        write_rows.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(write_rows, "wt", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    return result


def build_loader(index: Path, batch_size: int, *, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        EventNeedleArchiveDataset(index),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_event_needle,
        generator=generator,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selection_lock = audit_selection_lock()
    if args.seed not in config["seeds"]:
        raise ValueError("seed is not declared in the protocol")
    development_index = resolve(config["development_index"])
    validation_index = resolve(config["validation_index"])
    if "confirm" in str(development_index).lower() or "confirm" in str(validation_index).lower():
        raise PermissionError("this development trainer cannot open confirmatory artifacts")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = build_loader(
        development_index,
        int(config["batch_size"]),
        shuffle=True,
        seed=args.seed,
    )
    validation_loader = build_loader(
        validation_index,
        int(config["batch_size"]),
        shuffle=False,
        seed=args.seed,
    )
    model = CLAPPriorResidualRanker(
        embedding_dim=int(config["embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
        maximum_residual=float(config["maximum_residual"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    result_dir = PROJECT / "q1_plus" / "results" / "development" / "event_ranker"
    result_dir.mkdir(parents=True, exist_ok=True)
    run_tag = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(config["run_tag"])
    )
    if not run_tag:
        raise ValueError("run_tag must contain a filename-safe character")
    stem = f"{run_tag}__prior_residual__seed_{args.seed}"
    checkpoint = result_dir / "checkpoints" / f"{stem}.pt"
    if checkpoint.exists() and not args.force:
        raise FileExistsError(f"development checkpoint exists: {checkpoint}")

    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            mask = batch["audio_attention_mask"].to(device)
            output = model(
                batch["audio_embeddings"].to(device),
                batch["question_embeddings"].to(device),
                mask,
            )
            loss = qcr_evidence_loss(
                output,
                batch["evidence_targets"].to(device),
                mask,
                evidence_temperature=float(config["evidence_temperature"]),
                hard_negative_weight=float(config["hard_negative_weight"]),
                margin=float(config["margin"]),
                residual_penalty=float(config["residual_penalty"]),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["gradient_clip"])
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation = evaluate(model, validation_loader, device)
        selection_score = (
            validation["model"]["evidence_ap"]
            + 0.25 * validation["model"]["hit_at_1"]
        )
        entry = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "selection_score": selection_score,
            "validation": validation,
        }
        history.append(entry)
        print(json.dumps(entry, sort_keys=True), flush=True)
        if selection_score > best_score + 1e-8:
            best_score = selection_score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= int(config["patience"]):
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    raw_path = result_dir / f"raw_validation__{stem}.jsonl.gz"
    validation = evaluate(model, validation_loader, device, write_rows=raw_path)
    gates = config["promotion_gates"]
    gate_results = {
        "hit_at_1": validation["delta"]["hit_at_1"]
        >= float(gates["minimum_hit_at_1_gain_points"]) / 100.0,
        "evidence_ap": validation["delta"]["evidence_ap"]
        >= float(gates["minimum_evidence_ap_gain"]),
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "config": config,
            "seed": args.seed,
            "best_score": best_score,
        },
        checkpoint,
    )
    history_path = result_dir / f"history__{stem}.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    result = {
        "status": "development_complete",
        "run_tag": run_tag,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "validation": validation,
        "promotion_gates": gate_results,
        "all_promotion_gates_pass": all(gate_results.values()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "raw_validation_sha256": sha256(raw_path),
        "history_sha256": sha256(history_path),
        "development_index_sha256": sha256(development_index),
        "validation_index_sha256": sha256(validation_index),
        "config_sha256": sha256(args.config),
        "selection_lock_sha256": sha256(SELECTION_LOCK),
        "confirmatory_panel_status": selection_lock["confirmatory_panel_status"],
        "device": str(device),
        "torch": torch.__version__,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    metrics_path = result_dir / f"metrics__{stem}.json"
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
