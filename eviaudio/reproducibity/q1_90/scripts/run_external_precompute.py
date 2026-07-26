#!/usr/bin/env python3
"""Run and audit the single authorized CLAP precompute for external recipes."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
TRACK = PROJECT / "q1_90"
AUTHORIZATION = TRACK / "EXTERNAL_REPLICATION_AUTHORIZATION.json"
sys.path.insert(0, str(PROJECT / "src"))

from eviaudio_mt.event_needle import temporal_overlap_fraction  # noqa: E402


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
        raise PermissionError("external replication is not authorized")
    for relative, expected in authorization["files"].items():
        if sha256(project_path(relative)) != expected:
            raise RuntimeError(f"external authorization mismatch: {relative}")
    if int(authorization["model_scores_before_authorization"]) != 0:
        raise RuntimeError("model scoring preceded authorization")
    return authorization


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    if len(sys.argv) != 1:
        raise PermissionError("external precompute accepts no overrides")
    authorization = audit_authorization()
    recipes_path = project_path(authorization["recipes"])
    output_dir = project_path(authorization["precompute_output_dir"])
    receipt_path = output_dir / "external_precompute_receipt.json"
    if receipt_path.exists():
        raise FileExistsError("external precompute receipt is immutable")
    evaluation_dir = project_path(authorization["evaluation_output_dir"])
    if evaluation_dir.exists() and any(evaluation_dir.iterdir()):
        raise PermissionError("evaluation output exists before precompute receipt")
    resume = output_dir.exists() and any(output_dir.iterdir())
    command = [
        sys.executable,
        str(PROJECT / "q1_plus/scripts/precompute_event_needle_clap.py"),
        "--recipes",
        str(recipes_path),
        "--output-dir",
        str(output_dir),
        "--allow-confirmatory",
        "--sample-rate",
        str(authorization["embedding"]["sample_rate"]),
        "--chunk-seconds",
        str(authorization["embedding"]["chunk_seconds"]),
        "--hop-seconds",
        str(authorization["embedding"]["hop_seconds"]),
        "--inference-batch-size",
        str(authorization["embedding"]["inference_batch_size"]),
    ]
    if resume:
        command.append("--resume")
    subprocess.run(command, cwd=PROJECT, check=True)

    summary_path = output_dir / "clap_prior_summary.json"
    index_path = output_dir / "index.jsonl"
    raw_path = output_dir / "raw_clap_prior.jsonl.gz"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    recipes = load_jsonl(recipes_path)
    index = load_jsonl(index_path)
    if len(recipes) != int(authorization["examples"]) or len(index) != len(recipes):
        raise RuntimeError("external precompute example count mismatch")
    expected_summary = {
        "status": "confirmatory_embedding_complete",
        "examples": len(recipes),
        "model_id": authorization["embedding"]["model_id"],
        "model_revision": authorization["embedding"]["model_revision"],
        "sample_rate": authorization["embedding"]["sample_rate"],
        "chunk_seconds": authorization["embedding"]["chunk_seconds"],
        "hop_seconds": authorization["embedding"]["hop_seconds"],
        "inference_batch_size": authorization["embedding"]["inference_batch_size"],
        "recipes_sha256": sha256(recipes_path),
        "index_sha256": sha256(index_path),
        "raw_sha256": sha256(raw_path),
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise RuntimeError(f"external precompute summary mismatch: {key}")
    if [str(row["recipe_id"]) for row in index] != [str(row["recipe_id"]) for row in recipes]:
        raise RuntimeError("external precompute recipe order changed")

    unique_sources: dict[str, str] = {}
    total_chunks = 0
    for recipe, record in zip(recipes, index, strict=True):
        for item in [recipe["target_event"], *recipe["distractor_events"], *recipe["background_files"]]:
            path_text = str(item["path"])
            expected = str(item["sha256"])
            if sha256(PROJECT / path_text) != expected:
                raise RuntimeError(f"external source changed: {path_text}")
            unique_sources[path_text] = expected
        archive_path = Path(record["archive_path"])
        if sha256(archive_path) != record["archive_sha256"]:
            raise RuntimeError(f"external archive checksum mismatch: {record['recipe_id']}")
        with np.load(archive_path) as archive:
            if str(archive["recipe_id"]) != str(recipe["recipe_id"]):
                raise RuntimeError("external archive recipe mismatch")
            if str(archive["model_id"]) != authorization["embedding"]["model_id"]:
                raise RuntimeError("external archive model mismatch")
            if str(archive["model_revision"]) != authorization["embedding"]["model_revision"]:
                raise RuntimeError("external archive model revision mismatch")
            audio = archive["audio_embeddings"].astype(np.float64)
            question = archive["question_embedding"].astype(np.float64)
            starts = archive["start_sec"].astype(np.float64)
            ends = archive["end_sec"].astype(np.float64)
            evidence = archive["evidence_targets"].astype(bool)
            overlap = archive["overlap_fraction"].astype(np.float64)
        chunks = len(starts)
        if (
            audio.shape != (chunks, int(authorization["embedding"]["embedding_dim"]))
            or question.shape != (int(authorization["embedding"]["embedding_dim"]),)
            or ends.shape != starts.shape
            or evidence.shape != starts.shape
            or overlap.shape != starts.shape
            or int(record["n_chunks"]) != chunks
        ):
            raise RuntimeError(f"external archive alignment failure: {record['recipe_id']}")
        if not all(np.isfinite(array).all() for array in (audio, question, starts, ends, overlap)):
            raise RuntimeError(f"external archive contains nonfinite values: {record['recipe_id']}")
        exact = temporal_overlap_fraction(
            starts,
            ends,
            float(recipe["target_event"]["start_sec"]),
            float(recipe["target_event"]["end_sec"]),
        )
        if not np.allclose(overlap, exact, atol=1e-6, rtol=0.0) or not np.array_equal(evidence, exact > 0.0):
            raise RuntimeError(f"external evidence alignment failure: {record['recipe_id']}")
        if not evidence.any() or evidence.all():
            raise RuntimeError(f"external evidence labels are degenerate: {record['recipe_id']}")
        total_chunks += chunks
    manifest = "".join(f"{path}\0{checksum}\n" for path, checksum in sorted(unique_sources.items()))
    source_manifest_sha256 = hashlib.sha256(manifest.encode()).hexdigest()
    if source_manifest_sha256 != authorization["source_integrity"]["source_manifest_sha256"]:
        raise RuntimeError("external source manifest mismatch")
    receipt = {
        "status": "one_shot_external_precompute_complete",
        "authorization_sha256": sha256(AUTHORIZATION),
        "recipes_sha256": sha256(recipes_path),
        "summary_sha256": sha256(summary_path),
        "index_sha256": sha256(index_path),
        "raw_sha256": sha256(raw_path),
        "examples": len(recipes),
        "archives_verified": len(index),
        "total_chunks_verified": total_chunks,
        "unique_source_files_verified": len(unique_sources),
        "source_manifest_sha256": source_manifest_sha256,
        "archive_checksum_failures": 0,
        "source_checksum_failures": 0,
        "alignment_failures": 0,
        "nonfinite_failures": 0,
    }
    temporary = receipt_path.with_name(f"{receipt_path.name}.partial")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(receipt_path)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
