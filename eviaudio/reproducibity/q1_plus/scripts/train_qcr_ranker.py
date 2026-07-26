#!/usr/bin/env python3
"""Train and evaluate the development-only QCR residual ranker."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from eviaudio_mt.qcr import (  # noqa: E402
    CLAPDiagonalMetricRanker,
    CLAPPriorResidualRanker,
    qcr_source_loss,
    source_logmeanexp,
)
from eviaudio_mt.qcr_data import QCRManifestDataset, collate_qcr  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT / path


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order].astype(np.float64)
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    return float((precision * sorted_labels).sum() / positives)


def rank_of_target(scores: np.ndarray, target: int) -> int:
    order = np.argsort(-scores, kind="stable")
    return int(np.flatnonzero(order == target)[0]) + 1


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    write_rows: Path | None = None,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        audio = batch["audio_embeddings"].to(device)
        questions = batch["question_embeddings"].to(device)
        mask = batch["audio_attention_mask"].to(device)
        source_indices = batch["source_indices"].to(device)
        output = model(audio, questions, mask)
        model_source, model_source_mask = source_logmeanexp(
            output.scores, source_indices
        )
        prior_source, prior_source_mask = source_logmeanexp(
            output.prior_scores, source_indices
        )
        model_source = model_source.masked_fill(~model_source_mask, -1e4)
        prior_source = prior_source.masked_fill(~prior_source_mask, -1e4)
        for index in range(audio.shape[0]):
            valid_chunks = mask[index].cpu().numpy().astype(bool)
            target = int(batch["target_source_indices"][index])
            count = int(batch["n_sources"][index])
            model_source_values = model_source[index, :count].cpu().numpy()
            prior_source_values = prior_source[index, :count].cpu().numpy()
            evidence = batch["evidence_targets"][index].numpy()[valid_chunks]
            model_chunks = output.scores[index].cpu().numpy()[valid_chunks]
            prior_chunks = output.prior_scores[index].cpu().numpy()[valid_chunks]
            model_rank = rank_of_target(model_source_values, target)
            prior_rank = rank_of_target(prior_source_values, target)
            rows.append(
                {
                    "example_id": batch["example_id"][index],
                    "target_source_id": batch["target_source_id"][index],
                    "n_sources": count,
                    "difficulty": batch["difficulty"][index],
                    "model_source_correct": int(model_rank == 1),
                    "prior_source_correct": int(prior_rank == 1),
                    "model_source_rank": model_rank,
                    "prior_source_rank": prior_rank,
                    "model_evidence_ap": average_precision(model_chunks, evidence),
                    "prior_evidence_ap": average_precision(prior_chunks, evidence),
                    "model_recall_at_4": int(model_rank <= min(4, count)),
                    "prior_recall_at_4": int(prior_rank <= min(4, count)),
                    "model_source_scores": model_source_values.astype(float).tolist(),
                    "prior_source_scores": prior_source_values.astype(float).tolist(),
                    "model_chunk_scores": model_chunks.astype(float).tolist(),
                    "prior_chunk_scores": prior_chunks.astype(float).tolist(),
                    "chunk_source_indices": batch["source_indices"][index]
                    .numpy()[valid_chunks]
                    .astype(int)
                    .tolist(),
                    "evidence_targets": evidence.astype(int).tolist(),
                    "mean_absolute_residual": float(
                        output.residual_scores[index][mask[index]].abs().mean().cpu()
                    ),
                }
            )
    summary = {
        "examples": float(len(rows)),
        "source_accuracy": float(np.mean([row["model_source_correct"] for row in rows])),
        "prior_source_accuracy": float(
            np.mean([row["prior_source_correct"] for row in rows])
        ),
        "evidence_ap": float(np.mean([row["model_evidence_ap"] for row in rows])),
        "prior_evidence_ap": float(np.mean([row["prior_evidence_ap"] for row in rows])),
        "source_mrr": float(np.mean([1.0 / row["model_source_rank"] for row in rows])),
        "prior_source_mrr": float(
            np.mean([1.0 / row["prior_source_rank"] for row in rows])
        ),
        "recall_at_4": float(np.mean([row["model_recall_at_4"] for row in rows])),
        "prior_recall_at_4": float(
            np.mean([row["prior_recall_at_4"] for row in rows])
        ),
        "mean_absolute_residual": float(
            np.mean([row["mean_absolute_residual"] for row in rows])
        ),
    }
    if write_rows is not None:
        write_rows.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(write_rows, "wt", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    return summary


def build_loader(
    manifest: Path,
    question_archive: Path,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        QCRManifestDataset(manifest, question_archive),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_qcr,
        generator=generator,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--variant",
        choices=["prior_residual", "residual_only", "diagonal_metric"],
        default="prior_residual",
    )
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument(
        "--candidate",
        help="named development candidate declared in config.candidates",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config_document = json.loads(args.config.read_text(encoding="utf-8"))
    config = {key: value for key, value in config_document.items() if key != "candidates"}
    if args.candidate is not None:
        candidates = config_document.get("candidates", {})
        if args.candidate not in candidates:
            raise KeyError(f"undeclared development candidate: {args.candidate}")
        config.update(candidates[args.candidate])
    if args.seed not in config["seeds"]:
        raise ValueError("seed is not declared in the protocol")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    question_archive = resolve(config["question_archive"])
    train_loader = build_loader(
        resolve(config["train_manifest"]),
        question_archive,
        int(config["batch_size"]),
        shuffle=True,
        seed=args.seed,
    )
    validation_loader = build_loader(
        resolve(config["validation_manifest"]),
        question_archive,
        int(config["batch_size"]),
        shuffle=False,
        seed=args.seed,
    )
    result_dir = PROJECT / "q1_plus" / "results" / "development"
    result_dir.mkdir(parents=True, exist_ok=True)
    candidate_tag = f"__{args.candidate}" if args.candidate else ""
    stem = f"{args.variant}{candidate_tag}__seed_{args.seed}"
    checkpoint = result_dir / "checkpoints" / f"{stem}.pt"
    history_path = result_dir / f"history__{stem}.json"
    if checkpoint.exists() and not args.force:
        raise FileExistsError(f"development checkpoint exists: {checkpoint}")

    if args.variant == "diagonal_metric":
        model = CLAPDiagonalMetricRanker(
            embedding_dim=int(config["embedding_dim"]),
            maximum_residual=float(config["maximum_residual"]),
        ).to(device)
    else:
        model = CLAPPriorResidualRanker(
            embedding_dim=int(config["embedding_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            dropout=float(config["dropout"]),
            maximum_residual=float(config["maximum_residual"]),
            use_prior=args.variant == "prior_residual",
            shared_projection=bool(config.get("shared_projection", False)),
            interaction_only=bool(config.get("interaction_only", False)),
            include_position=bool(config.get("include_position", True)),
        ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch["audio_embeddings"].to(device),
                batch["question_embeddings"].to(device),
                batch["audio_attention_mask"].to(device),
            )
            loss = qcr_source_loss(
                output,
                batch["source_indices"].to(device),
                batch["target_source_indices"].to(device),
                residual_penalty=float(config["residual_penalty"]),
                source_temperature=float(config.get("source_temperature", 1.0)),
                evidence_targets=batch["evidence_targets"].to(device),
                evidence_weight=float(config.get("evidence_weight", 0.0)),
                evidence_temperature=float(config.get("evidence_temperature", 0.1)),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["gradient_clip"])
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation = evaluate(model, validation_loader, device)
        selection_score = validation["source_accuracy"] + 0.1 * validation["evidence_ap"]
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
            stale = 0
        else:
            stale += 1
            if stale >= int(config["patience"]):
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    validation_rows = result_dir / f"raw_validation__{stem}.jsonl.gz"
    validation = evaluate(model, validation_loader, device, write_rows=validation_rows)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "config": config,
            "variant": args.variant,
            "candidate": args.candidate,
            "seed": args.seed,
            "best_score": best_score,
        },
        checkpoint,
    )
    result: dict[str, Any] = {
        "status": "development_complete",
        "variant": args.variant,
        "candidate": args.candidate,
        "seed": args.seed,
        "validation": validation,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "validation_rows_sha256": sha256(validation_rows),
        "config_sha256": sha256(args.config),
        "question_archive_sha256": sha256(question_archive),
        "device": str(device),
        "torch": torch.__version__,
    }
    if args.evaluate_test:
        test_loader = build_loader(
            resolve(config["test_manifest"]),
            question_archive,
            int(config["batch_size"]),
            shuffle=False,
            seed=args.seed,
        )
        test_rows = result_dir / f"raw_test__{stem}.jsonl.gz"
        result["test"] = evaluate(model, test_loader, device, write_rows=test_rows)
        result["test_rows_sha256"] = sha256(test_rows)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    (result_dir / f"metrics__{stem}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
