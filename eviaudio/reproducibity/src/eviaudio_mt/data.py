from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2


class SyntheticAcousticKeyValueDataset(Dataset[dict[str, torch.Tensor]]):
    """Prompt-conditioned key/value retrieval in synthetic audio-embedding space.

    A prompt names an acoustic key. Exactly one chunk contains that key, and
    its second half contains the value token the model must generate. The task
    therefore exercises prompt-to-audio matching, evidence localization, and
    generation without downloading audio or a language model.
    """

    def __init__(
        self,
        n_examples: int,
        n_chunks: int,
        audio_dim: int,
        n_keys: int,
        n_values: int,
        *,
        seed: int = 2026,
        split_offset: int = 0,
        noise_std: float = 0.1,
        chunk_duration_sec: float = 2.0,
    ) -> None:
        if audio_dim < 4 or audio_dim % 2 != 0:
            raise ValueError("audio_dim must be even and at least 4")
        if n_chunks < 2 or n_keys < 2 or n_values < 2:
            raise ValueError("n_chunks, n_keys, and n_values must be at least 2")
        self.n_examples = n_examples
        self.n_chunks = n_chunks
        self.audio_dim = audio_dim
        self.n_keys = n_keys
        self.n_values = n_values
        self.seed = seed
        self.split_offset = split_offset
        self.noise_std = noise_std
        self.chunk_duration_sec = chunk_duration_sec
        self.key_offset = 3
        self.value_offset = 3 + n_keys

        prototype_rng = np.random.default_rng(seed)
        half = audio_dim // 2
        key_prototypes = prototype_rng.normal(size=(n_keys, half)).astype(np.float32)
        value_prototypes = prototype_rng.normal(size=(n_values, half)).astype(np.float32)
        self.key_prototypes = key_prototypes / np.linalg.norm(key_prototypes, axis=1, keepdims=True)
        self.value_prototypes = value_prototypes / np.linalg.norm(
            value_prototypes, axis=1, keepdims=True
        )

    @property
    def required_vocab_size(self) -> int:
        return self.value_offset + self.n_values

    def __len__(self) -> int:
        return self.n_examples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed + self.split_offset + index * 104729)
        query_key = int(rng.integers(self.n_keys))
        target_value = int(rng.integers(self.n_values))
        evidence_index = int(rng.integers(self.n_chunks))

        possible_distractor_keys = [key for key in range(self.n_keys) if key != query_key]
        keys = rng.choice(possible_distractor_keys, size=self.n_chunks, replace=True)
        values = rng.integers(self.n_values, size=self.n_chunks)
        keys[evidence_index] = query_key
        values[evidence_index] = target_value

        key_part = self.key_prototypes[keys]
        value_part = self.value_prototypes[values]
        embeddings = np.concatenate([key_part, value_part], axis=1)
        embeddings += rng.normal(0.0, self.noise_std, embeddings.shape).astype(np.float32)

        evidence = np.zeros(self.n_chunks, dtype=np.float32)
        evidence[evidence_index] = 1.0
        start = np.arange(self.n_chunks, dtype=np.float32) * self.chunk_duration_sec
        end = start + self.chunk_duration_sec

        prompt_ids = np.array(
            [BOS_TOKEN_ID, self.key_offset + query_key, EOS_TOKEN_ID], dtype=np.int64
        )
        labels = np.array(
            [self.value_offset + target_value, EOS_TOKEN_ID, -100], dtype=np.int64
        )
        return {
            "audio_embeddings": torch.from_numpy(embeddings),
            "audio_attention_mask": torch.ones(self.n_chunks, dtype=torch.bool),
            "prompt_input_ids": torch.from_numpy(prompt_ids),
            "prompt_attention_mask": torch.ones(len(prompt_ids), dtype=torch.long),
            "labels": torch.from_numpy(labels),
            "evidence_targets": torch.from_numpy(evidence),
            "chunk_start_sec": torch.from_numpy(start),
            "chunk_end_sec": torch.from_numpy(end),
            "example_id": torch.tensor(index, dtype=torch.long),
        }


class PrecomputedManifestDataset(Dataset[dict[str, Any]]):
    """Load precomputed chunk embeddings and tokenized text from JSONL.

    Each row should contain ``embedding_path``, ``prompt_ids``, ``labels``, and
    optionally ``evidence_targets``, ``chunk_start_sec``, and ``chunk_end_sec``.
    NPZ files may store the embedding under ``embeddings`` or ``audio_embeddings``.
    """

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(self.manifest_path)
        self.records: list[dict[str, Any]] = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "embedding_path" not in record and "embedding_paths" not in record:
                    raise ValueError(
                        f"Line {line_number} lacks embedding_path or embedding_paths"
                    )
                self.records.append(record)
        if not self.records:
            raise ValueError(f"No records found in {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def _load_embeddings(self, path_value: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        path = Path(path_value)
        if not path.is_absolute():
            path = (self.manifest_path.parent / path).resolve()
        if path.suffix == ".npy":
            return np.load(path).astype(np.float32), {}
        archive = np.load(path)
        for key in ("embeddings", "audio_embeddings"):
            if key in archive:
                extras = {name: archive[name] for name in archive.files if name != key}
                return archive[key].astype(np.float32), extras
        raise KeyError(f"No embeddings array found in {path}")

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if "embedding_paths" in record:
            arrays: list[np.ndarray] = []
            for path_value in record["embedding_paths"]:
                array, _ = self._load_embeddings(str(path_value))
                arrays.append(array)
            if not arrays:
                raise ValueError(f"Record {index} has an empty embedding_paths list")
            embeddings = np.concatenate(arrays, axis=0)
            extras: dict[str, np.ndarray] = {}
        else:
            embeddings, extras = self._load_embeddings(str(record["embedding_path"]))
        n_chunks = len(embeddings)
        evidence = np.asarray(record.get("evidence_targets", np.zeros(n_chunks)), dtype=np.float32)
        starts = np.asarray(
            record.get("chunk_start_sec", extras.get("start_sec", np.arange(n_chunks))),
            dtype=np.float32,
        )
        ends = np.asarray(
            record.get("chunk_end_sec", extras.get("end_sec", starts + 1.0)),
            dtype=np.float32,
        )
        return {
            "audio_embeddings": torch.from_numpy(embeddings),
            "audio_attention_mask": torch.ones(n_chunks, dtype=torch.bool),
            "prompt_input_ids": torch.tensor(record["prompt_ids"], dtype=torch.long),
            "prompt_attention_mask": torch.ones(len(record["prompt_ids"]), dtype=torch.long),
            "labels": torch.tensor(record["labels"], dtype=torch.long),
            "evidence_targets": torch.from_numpy(evidence),
            "chunk_start_sec": torch.from_numpy(starts),
            "chunk_end_sec": torch.from_numpy(ends),
            "example_id": str(record.get("example_id", index)),
        }


def collate_audio_text(
    batch: list[dict[str, Any]],
    pad_token_id: int = PAD_TOKEN_ID,
) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    batch_size = len(batch)
    max_chunks = max(item["audio_embeddings"].shape[0] for item in batch)
    audio_dim = batch[0]["audio_embeddings"].shape[1]
    max_prompt = max(item["prompt_input_ids"].numel() for item in batch)
    max_labels = max(item["labels"].numel() for item in batch)

    audio = torch.zeros(batch_size, max_chunks, audio_dim, dtype=torch.float32)
    audio_mask = torch.zeros(batch_size, max_chunks, dtype=torch.bool)
    evidence = torch.zeros(batch_size, max_chunks, dtype=torch.float32)
    starts = torch.zeros(batch_size, max_chunks, dtype=torch.float32)
    ends = torch.zeros(batch_size, max_chunks, dtype=torch.float32)
    prompt = torch.full((batch_size, max_prompt), pad_token_id, dtype=torch.long)
    prompt_mask = torch.zeros(batch_size, max_prompt, dtype=torch.long)
    labels = torch.full((batch_size, max_labels), -100, dtype=torch.long)
    example_ids: list[Any] = []

    for row, item in enumerate(batch):
        n = item["audio_embeddings"].shape[0]
        p = item["prompt_input_ids"].numel()
        t = item["labels"].numel()
        audio[row, :n] = item["audio_embeddings"]
        audio_mask[row, :n] = item.get("audio_attention_mask", torch.ones(n, dtype=torch.bool))
        evidence[row, :n] = item.get("evidence_targets", torch.zeros(n))
        starts[row, :n] = item.get("chunk_start_sec", torch.arange(n, dtype=torch.float32))
        ends[row, :n] = item.get("chunk_end_sec", starts[row, :n] + 1.0)
        prompt[row, :p] = item["prompt_input_ids"]
        prompt_mask[row, :p] = item.get("prompt_attention_mask", torch.ones(p, dtype=torch.long))
        labels[row, :t] = item["labels"]
        example_ids.append(item.get("example_id", row))

    return {
        "audio_embeddings": audio,
        "audio_attention_mask": audio_mask,
        "prompt_input_ids": prompt,
        "prompt_attention_mask": prompt_mask,
        "labels": labels,
        "evidence_targets": evidence,
        "chunk_start_sec": starts,
        "chunk_end_sec": ends,
        "example_id": example_ids,
    }


def build_synthetic_loaders(config: dict[str, Any], seed: int):
    common = {
        "n_chunks": int(config["n_chunks"]),
        "audio_dim": int(config["audio_dim"]),
        "n_keys": int(config["n_keys"]),
        "n_values": int(config["n_values"]),
        "seed": seed,
        "noise_std": float(config.get("noise_std", 0.1)),
        "chunk_duration_sec": float(config.get("chunk_duration_sec", 2.0)),
    }
    train = SyntheticAcousticKeyValueDataset(
        int(config["train_examples"]), split_offset=0, **common
    )
    validation = SyntheticAcousticKeyValueDataset(
        int(config["validation_examples"]), split_offset=10_000_000, **common
    )
    test = SyntheticAcousticKeyValueDataset(
        int(config["test_examples"]), split_offset=20_000_000, **common
    )
    loader_args = {
        "batch_size": int(config.get("batch_size", 32)),
        "num_workers": int(config.get("num_workers", 0)),
        "collate_fn": collate_audio_text,
    }
    return (
        DataLoader(train, shuffle=True, **loader_args),
        DataLoader(validation, shuffle=False, **loader_args),
        DataLoader(test, shuffle=False, **loader_args),
    )


def build_manifest_loaders(config: dict[str, Any]):
    loader_args = {
        "batch_size": int(config.get("batch_size", 16)),
        "num_workers": int(config.get("num_workers", 0)),
        "collate_fn": partial(
            collate_audio_text, pad_token_id=int(config.get("pad_token_id", 0))
        ),
    }
    train = PrecomputedManifestDataset(config["train_manifest"])
    validation = PrecomputedManifestDataset(config["validation_manifest"])
    test = PrecomputedManifestDataset(config["test_manifest"])
    return (
        DataLoader(train, shuffle=True, **loader_args),
        DataLoader(validation, shuffle=False, **loader_args),
        DataLoader(test, shuffle=False, **loader_args),
    )


def build_loaders(config: dict[str, Any], seed: int):
    source = str(config.get("source", "synthetic")).lower()
    if source == "synthetic":
        return build_synthetic_loaders(config, seed)
    if source == "manifest":
        return build_manifest_loaders(config)
    raise ValueError(f"Unsupported data.source: {source!r}")


def inspect_hf_dataset(
    repository: str,
    *,
    configuration: str | None = None,
    split: str = "train",
    revision: str | None = None,
    streaming: bool = True,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Return schema-friendly previews without assuming repository fields."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError("Install requirements-full.txt to inspect HF datasets") from exc
    kwargs: dict[str, Any] = {
        "path": repository,
        "name": configuration,
        "split": split,
        "streaming": streaming,
    }
    if revision:
        kwargs["revision"] = revision
    dataset: Iterable[dict[str, Any]] = load_dataset(**kwargs)
    previews: list[dict[str, Any]] = []
    for index, row in enumerate(dataset):
        if index >= limit:
            break
        preview: dict[str, Any] = {"keys": sorted(row.keys())}
        for key, value in row.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                preview[key] = value
            elif isinstance(value, dict):
                preview[key] = {subkey: type(subvalue).__name__ for subkey, subvalue in value.items()}
            else:
                try:
                    preview[key] = {"type": type(value).__name__, "length": len(value)}
                except TypeError:
                    preview[key] = {"type": type(value).__name__}
        previews.append(preview)
    return previews
