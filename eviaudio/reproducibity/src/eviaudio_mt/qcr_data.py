from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class QCRManifestDataset(Dataset[dict[str, Any]]):
    """Question-conditioned source-retrieval examples from CLAP archives."""

    def __init__(self, manifest_path: str | Path, question_archive: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.question_archive = Path(question_archive)
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        archive = np.load(self.question_archive)
        questions = archive["questions"].astype(str).tolist()
        embeddings = archive["embeddings"].astype(np.float32)
        self.question_vectors = {
            question.strip(): embeddings[index]
            for index, question in enumerate(questions)
        }
        missing = sorted(
            {
                str(record["question"]).strip()
                for record in self.records
                if str(record["question"]).strip() not in self.question_vectors
            }
        )
        if missing:
            raise KeyError(f"question archive is missing {len(missing)} prompts")

    def __len__(self) -> int:
        return len(self.records)

    @lru_cache(maxsize=512)
    def _load_source(self, path_text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        path = Path(path_text)
        if not path.is_absolute():
            path = (self.manifest_path.parent / path).resolve()
        archive = np.load(path)
        embeddings = archive["embeddings"].astype(np.float32)
        starts = archive.get("start_sec", np.arange(len(embeddings))).astype(np.float32)
        ends = archive.get("end_sec", starts + 1.0).astype(np.float32)
        return embeddings, starts, ends

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        paths = record.get("embedding_paths")
        if not paths:
            paths = [record["embedding_path"]]
        audio_parts: list[np.ndarray] = []
        source_parts: list[np.ndarray] = []
        start_parts: list[np.ndarray] = []
        end_parts: list[np.ndarray] = []
        offset = 0.0
        for source_index, path in enumerate(paths):
            embeddings, starts, ends = self._load_source(str(path))
            audio_parts.append(embeddings)
            source_parts.append(
                np.full(len(embeddings), source_index, dtype=np.int64)
            )
            start_parts.append(starts + offset)
            end_parts.append(ends + offset)
            if len(ends):
                offset += float(ends[-1])
        audio = np.concatenate(audio_parts, axis=0)
        source_indices = np.concatenate(source_parts)
        starts = np.concatenate(start_parts)
        ends = np.concatenate(end_parts)
        target_source = int(record["target_position"])
        evidence = (source_indices == target_source).astype(np.float32)
        return {
            "audio_embeddings": torch.from_numpy(audio),
            "question_embedding": torch.from_numpy(
                self.question_vectors[str(record["question"]).strip()].copy()
            ),
            "source_indices": torch.from_numpy(source_indices),
            "target_source_index": torch.tensor(target_source, dtype=torch.long),
            "evidence_targets": torch.from_numpy(evidence),
            "chunk_start_sec": torch.from_numpy(starts),
            "chunk_end_sec": torch.from_numpy(ends),
            "example_id": str(record.get("example_id", index)),
            "question": str(record["question"]),
            "answer": str(record.get("answer", "")),
            "source_ids": list(record.get("source_ids", [])),
            "target_source_id": str(record.get("target_source_id", "")),
            "n_sources": int(record.get("n_sources", len(paths))),
            "difficulty": str(record.get("difficulty", "unknown")),
        }


def collate_qcr(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty batch")
    maximum_chunks = max(len(item["audio_embeddings"]) for item in batch)
    dimension = int(batch[0]["audio_embeddings"].shape[-1])
    batch_size = len(batch)
    audio = torch.zeros(batch_size, maximum_chunks, dimension)
    questions = torch.stack([item["question_embedding"] for item in batch])
    mask = torch.zeros(batch_size, maximum_chunks, dtype=torch.bool)
    source_indices = torch.full(
        (batch_size, maximum_chunks), -1, dtype=torch.long
    )
    evidence = torch.zeros(batch_size, maximum_chunks)
    starts = torch.zeros(batch_size, maximum_chunks)
    ends = torch.zeros(batch_size, maximum_chunks)
    for row, item in enumerate(batch):
        count = len(item["audio_embeddings"])
        audio[row, :count] = item["audio_embeddings"]
        mask[row, :count] = True
        source_indices[row, :count] = item["source_indices"]
        evidence[row, :count] = item["evidence_targets"]
        starts[row, :count] = item["chunk_start_sec"]
        ends[row, :count] = item["chunk_end_sec"]
    return {
        "audio_embeddings": audio,
        "question_embeddings": questions,
        "audio_attention_mask": mask,
        "source_indices": source_indices,
        "target_source_indices": torch.stack(
            [item["target_source_index"] for item in batch]
        ),
        "evidence_targets": evidence,
        "chunk_start_sec": starts,
        "chunk_end_sec": ends,
        "example_id": [item["example_id"] for item in batch],
        "question": [item["question"] for item in batch],
        "answer": [item["answer"] for item in batch],
        "source_ids": [item["source_ids"] for item in batch],
        "target_source_id": [item["target_source_id"] for item in batch],
        "n_sources": torch.tensor([item["n_sources"] for item in batch]),
        "difficulty": [item["difficulty"] for item in batch],
    }
