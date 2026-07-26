#!/usr/bin/env python3
"""Materialize recipes in memory, embed chunks with pinned CLAP, and score the prior."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoProcessor, ClapModel


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from eviaudio_mt.audio_encoder import chunk_waveform  # noqa: E402
from eviaudio_mt.event_needle import (  # noqa: E402
    materialize_recipe,
    temporal_overlap_fraction,
)


MODEL_ID = "laion/clap-htsat-unfused"
MODEL_REVISION = "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"


def sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            checksum.update(block)
    return checksum.hexdigest()


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


def normalized(array: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(array.float(), dim=-1)


@torch.inference_mode()
def embed_text(
    query: str,
    processor: Any,
    model: ClapModel,
    device: torch.device,
) -> np.ndarray:
    inputs = processor(text=[query], return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    vector = normalized(model.get_text_features(**inputs))[0]
    return vector.cpu().numpy().astype(np.float32)


@torch.inference_mode()
def embed_audio_chunks(
    waveform: np.ndarray,
    processor: Any,
    model: ClapModel,
    device: torch.device,
    *,
    sample_rate: int,
    chunk_seconds: float,
    hop_seconds: float,
    inference_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chunks, starts, ends = chunk_waveform(
        torch.from_numpy(waveform),
        sample_rate=sample_rate,
        chunk_duration_sec=chunk_seconds,
        hop_duration_sec=hop_seconds,
    )
    chunks = chunks[0]
    outputs: list[torch.Tensor] = []
    for offset in range(0, len(chunks), inference_batch_size):
        batch = [
            chunk.numpy().astype(np.float32, copy=False)
            for chunk in chunks[offset : offset + inference_batch_size]
        ]
        inputs = processor(
            audios=batch,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs.append(normalized(model.get_audio_features(**inputs)).cpu())
    return (
        torch.cat(outputs).numpy().astype(np.float32),
        starts.cpu().numpy().astype(np.float32),
        ends.cpu().numpy().astype(np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-confirmatory", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Verify and reuse completed per-recipe archives after an interrupted run",
    )
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--chunk-seconds", type=float, default=4.0)
    parser.add_argument("--hop-seconds", type=float, default=2.0)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.chunk_seconds <= 0 or args.hop_seconds <= 0:
        raise ValueError("chunk and hop durations must be positive")
    recipes = [
        json.loads(line)
        for line in args.recipes.open("r", encoding="utf-8")
        if line.strip()
    ]
    if args.limit is not None:
        recipes = recipes[: args.limit]
    panels = {str(recipe["panel"]) for recipe in recipes}
    if "confirmatory" in panels and not args.allow_confirmatory:
        raise PermissionError(
            "confirmatory recipes remain sealed; --allow-confirmatory is required"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = args.output_dir / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "index.jsonl"
    raw_path = args.output_dir / "raw_clap_prior.jsonl.gz"
    if args.force and args.resume:
        raise ValueError("--force and --resume are mutually exclusive")
    if (index_path.exists() or raw_path.exists()) and not (args.force or args.resume):
        raise FileExistsError("output exists; pass --force to replace this development artifact")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    missing_archive = any(
        not (archive_dir / f"{recipe['recipe_id']}.npz").exists()
        for recipe in recipes
    )
    processor: Any | None = None
    model: ClapModel | None = None
    if not args.resume or missing_archive:
        processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        model = ClapModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    query_cache: dict[str, np.ndarray] = {}
    index_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for recipe_number, recipe in enumerate(recipes, start=1):
        item_started = time.perf_counter()
        recipe_id = str(recipe["recipe_id"])
        archive_path = archive_dir / f"{recipe_id}.npz"
        target_start = float(recipe["target_event"]["start_sec"])
        target_end = float(recipe["target_event"]["end_sec"])
        reused = bool(args.resume and archive_path.exists())
        if reused:
            with np.load(archive_path) as archive:
                if str(archive["recipe_id"]) != recipe_id:
                    raise RuntimeError(f"recipe id mismatch in {archive_path}")
                if str(archive["model_id"]) != MODEL_ID or str(
                    archive["model_revision"]
                ) != MODEL_REVISION:
                    raise RuntimeError(f"model provenance mismatch in {archive_path}")
                embeddings = archive["audio_embeddings"].astype(np.float32)
                question_embedding = archive["question_embedding"].astype(np.float32)
                starts = archive["start_sec"].astype(np.float32)
                ends = archive["end_sec"].astype(np.float32)
                evidence = archive["evidence_targets"].astype(bool)
                overlap = archive["overlap_fraction"].astype(np.float32)
        else:
            assert processor is not None and model is not None
            materialized = materialize_recipe(
                recipe, project_root=PROJECT, sample_rate=args.sample_rate
            )
            embeddings, starts, ends = embed_audio_chunks(
                materialized.waveform,
                processor,
                model,
                device,
                sample_rate=args.sample_rate,
                chunk_seconds=args.chunk_seconds,
                hop_seconds=args.hop_seconds,
                inference_batch_size=args.inference_batch_size,
            )
            query = str(recipe["query"])
            if query not in query_cache:
                query_cache[query] = embed_text(query, processor, model, device)
            question_embedding = query_cache[query]
            overlap = temporal_overlap_fraction(
                starts, ends, target_start, target_end
            )
            evidence = overlap > 0.0
            np.savez_compressed(
                archive_path,
                audio_embeddings=embeddings.astype(np.float16),
                question_embedding=question_embedding.astype(np.float16),
                start_sec=starts,
                end_sec=ends,
                evidence_targets=evidence.astype(np.uint8),
                overlap_fraction=overlap,
                recipe_id=np.asarray(recipe_id),
                model_id=np.asarray(MODEL_ID),
                model_revision=np.asarray(MODEL_REVISION),
            )
        scores = embeddings @ question_embedding
        top_order = np.argsort(-scores, kind="stable")
        top = int(top_order[0])
        archive_checksum = sha256(archive_path)
        index_rows.append(
            {
                "recipe_id": recipe_id,
                "panel": recipe["panel"],
                "recipe_path": str(args.recipes.resolve()),
                "archive_path": str(archive_path.resolve()),
                "archive_sha256": archive_checksum,
                "n_chunks": len(starts),
                "duration_sec": recipe["duration_sec"],
                "position_bin": recipe["position_bin"],
                "snr_db": recipe["target_event"]["snr_db"],
                "class_id": recipe["target_event"]["class_id"],
                "class_label": recipe["target_event"]["class_label"],
                "foreground_cluster_id": recipe["foreground_cluster_id"],
                "target_start_sec": target_start,
                "target_end_sec": target_end,
            }
        )
        metric_rows.append(
            {
                "recipe_id": recipe_id,
                "panel": recipe["panel"],
                "foreground_cluster_id": recipe["foreground_cluster_id"],
                "class_id": recipe["target_event"]["class_id"],
                "class_label": recipe["target_event"]["class_label"],
                "duration_sec": recipe["duration_sec"],
                "position_bin": recipe["position_bin"],
                "snr_db": recipe["target_event"]["snr_db"],
                "evidence_ap": average_precision(scores, evidence),
                "hit_at_1": int(evidence[top]),
                "recall_at_4": int(evidence[top_order[:4]].any()),
                "top_chunk_iou": interval_iou(
                    float(starts[top]),
                    float(ends[top]),
                    target_start,
                    target_end,
                ),
                "n_chunks": len(starts),
                "wall_seconds": time.perf_counter() - item_started,
            }
        )
        print(
            json.dumps(
                {
                    "completed": recipe_number,
                    "total": len(recipes),
                    "recipe_id": recipe_id,
                    "seconds": round(metric_rows[-1]["wall_seconds"], 3),
                    "ap": round(metric_rows[-1]["evidence_ap"], 4),
                    "reused": reused,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    with index_path.open("w", encoding="utf-8") as handle:
        for row in index_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
        for row in metric_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "status": "development_embedding_complete"
        if "confirmatory" not in panels
        else "confirmatory_embedding_complete",
        "panels": sorted(panels),
        "examples": len(metric_rows),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "sample_rate": args.sample_rate,
        "chunk_seconds": args.chunk_seconds,
        "hop_seconds": args.hop_seconds,
        "inference_batch_size": args.inference_batch_size,
        "mean_evidence_ap": float(np.mean([row["evidence_ap"] for row in metric_rows])),
        "hit_at_1": float(np.mean([row["hit_at_1"] for row in metric_rows])),
        "recall_at_4": float(np.mean([row["recall_at_4"] for row in metric_rows])),
        "mean_top_chunk_iou": float(np.mean([row["top_chunk_iou"] for row in metric_rows])),
        "recipes_sha256": sha256(args.recipes),
        "index_sha256": sha256(index_path),
        "raw_sha256": sha256(raw_path),
        "wall_seconds": time.perf_counter() - started,
        "device": str(device),
        "torch": torch.__version__,
    }
    summary_path = args.output_dir / "clap_prior_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
