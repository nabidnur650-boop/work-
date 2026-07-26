#!/usr/bin/env python3
"""Materialize the pinned exploratory Clotho-AQA waveforms with hash checks."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download


PROJECT = Path(__file__).resolve().parents[2]
Q1 = PROJECT / "q1_plus"
SOURCE_RECORDS = PROJECT / "journal_suite/data/source_records.json"
DESTINATION = Q1 / "data/development_audio"
REPOSITORY = "LakoreAI/clotho-dev-sample"
REVISION = "bcca62424d3fe62fbaf390559892808bf61c69c4"
PARQUET_FILES = tuple(
    [f"data/train-{index:05d}-of-00004.parquet" for index in range(4)]
    + ["data/test-00000-of-00001.parquet"]
)


def normalize_filename(value: str) -> str:
    value = os.path.basename(value.replace("\\", "/"))
    return re.sub(r"\s+", " ", value.strip().lower())


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    args = parser.parse_args()
    output = args.output.resolve()
    if output != DESTINATION.resolve():
        raise PermissionError("development audio has one fixed output directory")

    records = json.loads(SOURCE_RECORDS.read_text(encoding="utf-8"))
    expected: dict[str, dict[str, Any]] = {}
    for record in records:
        key = normalize_filename(str(record["filename"]))
        if key in expected:
            raise RuntimeError(f"duplicate normalized source filename: {key}")
        expected[key] = record
    source_dir = output / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    observed: dict[str, dict[str, Any]] = {}

    for relative in PARQUET_FILES:
        parquet_path = Path(
            hf_hub_download(
                REPOSITORY,
                relative,
                repo_type="dataset",
                revision=REVISION,
            )
        )
        table = pq.read_table(parquet_path, columns=["file_name", "audio"])
        for row in table.to_pylist():
            key = normalize_filename(str(row["file_name"]))
            if key not in expected or key in observed:
                continue
            record = expected[key]
            audio_bytes = bytes(row["audio"]["bytes"])
            checksum = sha256_bytes(audio_bytes)
            if checksum != str(record["waveform_sha256"]):
                raise RuntimeError(f"waveform checksum mismatch: {key}")
            if audio_bytes[:4] != b"RIFF" or audio_bytes[8:12] != b"WAVE":
                raise RuntimeError(f"source is not an original WAV container: {key}")
            waveform, sample_rate = sf.read(
                io.BytesIO(audio_bytes), dtype="float32", always_2d=False
            )
            frames = int(waveform.shape[0])
            channels = 1 if waveform.ndim == 1 else int(waveform.shape[1])
            destination = source_dir / f"{record['source_id']}.wav"
            if destination.exists():
                if sha256_bytes(destination.read_bytes()) != checksum:
                    raise RuntimeError(f"existing development WAV changed: {destination}")
            else:
                destination.write_bytes(audio_bytes)
            observed[key] = {
                "source_id": str(record["source_id"]),
                "original_filename": str(record["filename"]),
                "split": str(record["split"]),
                "path": str(destination.relative_to(Q1)),
                "sha256": checksum,
                "bytes": len(audio_bytes),
                "sample_rate": int(sample_rate),
                "frames": frames,
                "channels": channels,
                "duration_sec": frames / float(sample_rate),
            }

    missing = sorted(set(expected) - set(observed))
    if missing:
        raise RuntimeError(f"missing {len(missing)} pinned development sources")
    manifest_rows = sorted(observed.values(), key=lambda row: row["source_id"])
    manifest = {
        "status": "complete_hash_verified_exploratory_audio_only",
        "repository": REPOSITORY,
        "revision": REVISION,
        "source_records": str(SOURCE_RECORDS.relative_to(PROJECT)),
        "source_records_sha256": hashlib.sha256(SOURCE_RECORDS.read_bytes()).hexdigest(),
        "sources": len(manifest_rows),
        "total_bytes": sum(int(row["bytes"]) for row in manifest_rows),
        "splits": {
            split: sum(row["split"] == split for row in manifest_rows)
            for split in ("train", "val", "test")
        },
        "audita_rows_accessed": 0,
        "records": manifest_rows,
    }
    manifest_path = output / "manifest.json"
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != serialized:
        raise RuntimeError("existing development-audio manifest differs")
    manifest_path.write_text(serialized, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "sources": manifest["sources"],
                "splits": manifest["splits"],
                "total_bytes": manifest["total_bytes"],
                "manifest": str(manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
