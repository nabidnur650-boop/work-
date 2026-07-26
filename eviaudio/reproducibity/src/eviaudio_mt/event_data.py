from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EventNeedleArchiveDataset(Dataset[dict[str, Any]]):
    def __init__(self, index_path: str | Path, *, verify_checksums: bool = True) -> None:
        self.index_path = Path(index_path)
        with self.index_path.open("r", encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        self.verify_checksums = bool(verify_checksums)
        if not self.records:
            raise ValueError("event index is empty")
        self.recipe_targets: dict[str, tuple[float, float]] = {}
        recipe_paths = {
            str(record["recipe_path"])
            for record in self.records
            if record.get("recipe_path")
        }
        for recipe_path_text in recipe_paths:
            recipe_path = Path(recipe_path_text)
            with recipe_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    recipe = json.loads(line)
                    target = recipe["target_event"]
                    self.recipe_targets[str(recipe["recipe_id"])] = (
                        float(target["start_sec"]),
                        float(target["end_sec"]),
                    )

    def __len__(self) -> int:
        return len(self.records)

    @lru_cache(maxsize=2048)
    def _load(self, path_text: str, expected_sha256: str) -> dict[str, np.ndarray]:
        path = Path(path_text)
        if self.verify_checksums and _sha256(path) != expected_sha256:
            raise RuntimeError(f"event archive checksum mismatch: {path}")
        with np.load(path) as archive:
            return {key: archive[key].copy() for key in archive.files}

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        archive = self._load(record["archive_path"], record["archive_sha256"])
        recipe_id = str(record["recipe_id"])
        target_bounds = self.recipe_targets.get(recipe_id)
        target_start = record.get("target_start_sec")
        target_end = record.get("target_end_sec")
        if target_bounds is not None:
            target_start, target_end = target_bounds
        if target_start is None or target_end is None:
            # Exact bounds are only needed for the secondary IoU metric. This
            # fallback keeps legacy/unit-test archives readable while the
            # primary AP and hit@1 labels remain exact.
            positive = archive["evidence_targets"].astype(bool)
            if not positive.any():
                raise ValueError(f"archive contains no evidence: {recipe_id}")
            target_start = float(archive["start_sec"][positive].min())
            target_end = float(archive["end_sec"][positive].max())
        return {
            "audio_embeddings": torch.from_numpy(
                archive["audio_embeddings"].astype(np.float32)
            ),
            "question_embedding": torch.from_numpy(
                archive["question_embedding"].astype(np.float32)
            ),
            "evidence_targets": torch.from_numpy(
                archive["evidence_targets"].astype(np.float32)
            ),
            "overlap_fraction": torch.from_numpy(
                archive["overlap_fraction"].astype(np.float32)
            ),
            "start_sec": torch.from_numpy(archive["start_sec"].astype(np.float32)),
            "end_sec": torch.from_numpy(archive["end_sec"].astype(np.float32)),
            "target_start_sec": float(target_start),
            "target_end_sec": float(target_end),
            **record,
        }


def collate_event_needle(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty batch")
    maximum_chunks = max(len(item["audio_embeddings"]) for item in batch)
    batch_size = len(batch)
    embedding_dim = int(batch[0]["audio_embeddings"].shape[-1])
    audio = torch.zeros(batch_size, maximum_chunks, embedding_dim)
    mask = torch.zeros(batch_size, maximum_chunks, dtype=torch.bool)
    evidence = torch.zeros(batch_size, maximum_chunks)
    overlap = torch.zeros(batch_size, maximum_chunks)
    starts = torch.zeros(batch_size, maximum_chunks)
    ends = torch.zeros(batch_size, maximum_chunks)
    for row, item in enumerate(batch):
        count = len(item["audio_embeddings"])
        audio[row, :count] = item["audio_embeddings"]
        mask[row, :count] = True
        evidence[row, :count] = item["evidence_targets"]
        overlap[row, :count] = item["overlap_fraction"]
        starts[row, :count] = item["start_sec"]
        ends[row, :count] = item["end_sec"]
    passthrough = [
        "recipe_id",
        "panel",
        "duration_sec",
        "position_bin",
        "snr_db",
        "class_id",
        "class_label",
        "foreground_cluster_id",
        "target_start_sec",
        "target_end_sec",
    ]
    return {
        "audio_embeddings": audio,
        "question_embeddings": torch.stack(
            [item["question_embedding"] for item in batch]
        ),
        "audio_attention_mask": mask,
        "evidence_targets": evidence,
        "overlap_fraction": overlap,
        "start_sec": starts,
        "end_sec": ends,
        **{key: [item[key] for item in batch] for key in passthrough},
    }
