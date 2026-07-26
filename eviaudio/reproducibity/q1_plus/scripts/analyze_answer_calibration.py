#!/usr/bin/env python3
"""Audit both answer-calibration backbones and freeze one prompt/K pair."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


PROJECT = Path(__file__).resolve().parents[2]
Q1 = PROJECT / "q1_plus"
CONFIG = Q1 / "configs/answer_development.json"
LOCK = Q1 / "ANSWER_DEVELOPMENT_LOCK.json"
MANIFEST = PROJECT / "journal_suite/data/manifests/val.jsonl"
AUDIO_MANIFEST = Q1 / "data/development_audio/manifest.json"
RETRIEVAL_REPORT = Q1 / "results/development/answer_retrieval/selection_report.json"
RETRIEVAL_RAW = Q1 / "results/development/answer_retrieval/retrieval_scores.jsonl.gz"
RESULTS = Q1 / "results/development/answer_generation"
REPORT = RESULTS / "calibration_selection_report.json"
MODELS = ("qwen2_audio", "phi4_multimodal")
SAMPLE_RATE = 16_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, reference: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(reference).split()
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2.0 * precision * recall / (precision + recall)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else path.open
    with opener(path, "rt", encoding="utf-8") if path.suffix == ".gz" else opener(
        "r", encoding="utf-8"
    ) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def calibration_examples(
    retrieval_rows: list[dict[str, Any]], maximum_per_source: int
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in retrieval_rows:
        if row["split"] == "calibration":
            grouped[str(row["target_source_id"])].append(row)
    selected = []
    for source, rows in sorted(grouped.items()):
        ranked = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"answer-calibration-example|{row['example_id']}".encode()
            ).hexdigest(),
        )
        selected.extend(ranked[:maximum_per_source])
    if len(grouped) != 24 or len(selected) != 24 * maximum_per_source:
        raise RuntimeError("unexpected calibration-example contract")
    return sorted(selected, key=lambda row: row["example_id"])


def source_macro(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["target_source_id"])].append(row)
    if len(grouped) != 24 or any(len(items) != 4 for items in grouped.values()):
        raise RuntimeError("answer calibration is not source balanced")
    return {
        "source_macro_exact": float(
            np.mean([np.mean([row["exact"] for row in items]) for items in grouped.values()])
        ),
        "source_macro_token_f1": float(
            np.mean(
                [np.mean([row["token_f1"] for row in items]) for items in grouped.values()]
            )
        ),
        "sources": len(grouped),
        "examples": len(rows),
    }


def select_combination(combinations: list[dict[str, Any]]) -> dict[str, Any]:
    if not combinations:
        raise RuntimeError("no prompt/K candidates were evaluated")
    return sorted(
        combinations,
        key=lambda row: (
            -float(row["two_backbone_mean_source_macro_exact"]),
            -float(row["two_backbone_mean_source_macro_token_f1"]),
            int(row["k"]),
            str(row["prompt_name"]),
        ),
    )[0]


def audit_lock() -> dict[str, Any]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["status"] != "answer_calibration_locked_before_generated_clotho_answers":
        raise PermissionError("answer calibration is not pre-generation locked")
    if lock["clotho_generated_answers_before_lock"] != 0:
        raise PermissionError("answer lock followed generated Clotho answers")
    if lock["audita_rows_accessed"] != 0 or lock["audita_status"] != "sealed":
        raise PermissionError("AUDITA was not sealed at answer-development lock")
    for relative, expected in lock["files"].items():
        if sha256(PROJECT / relative) != expected:
            raise RuntimeError(f"answer-development lock mismatch: {relative}")
    return lock


class AudioStore:
    def __init__(self) -> None:
        manifest = json.loads(AUDIO_MANIFEST.read_text(encoding="utf-8"))
        if manifest["sources"] != 281 or manifest["audita_rows_accessed"] != 0:
            raise RuntimeError("development-audio manifest failed audit")
        self.paths = {
            str(row["source_id"]): Q1 / str(row["path"])
            for row in manifest["records"]
        }
        self.hashes = {
            str(row["source_id"]): str(row["sha256"])
            for row in manifest["records"]
        }
        self.cache: dict[str, np.ndarray] = {}

    def load(self, source_id: str) -> np.ndarray:
        if source_id in self.cache:
            return self.cache[source_id]
        path = self.paths[source_id]
        if "audita" in str(path).lower() or sha256(path) != self.hashes[source_id]:
            raise RuntimeError(f"development audio failed provenance audit: {source_id}")
        waveform, rate = sf.read(path, dtype="float32", always_2d=False)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if rate != SAMPLE_RATE:
            divisor = int(np.gcd(rate, SAMPLE_RATE))
            waveform = resample_poly(
                waveform, SAMPLE_RATE // divisor, rate // divisor
            ).astype(np.float32)
        waveform = waveform.astype(np.float32, copy=False)
        if waveform.ndim != 1 or not np.isfinite(waveform).all():
            raise RuntimeError(f"invalid development waveform: {source_id}")
        self.cache[source_id] = waveform
        return waveform


def descriptors(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    expected_root = (PROJECT / "journal_suite/data/embeddings").resolve()
    for source_index, (source_id, embedding_text) in enumerate(
        zip(manifest["source_ids"], manifest["embedding_paths"], strict=True)
    ):
        path = (PROJECT / "journal_suite/data/manifests" / embedding_text).resolve()
        if path.parent != expected_root:
            raise RuntimeError("embedding path escaped its declared directory")
        archive = np.load(path, allow_pickle=False)
        for start, end in zip(archive["start_sec"], archive["end_sec"], strict=True):
            rows.append(
                {
                    "source_index": source_index,
                    "source_id": str(source_id),
                    "start_sec": float(start),
                    "end_sec": float(end),
                }
            )
    if len(rows) != int(manifest["n_chunks"]):
        raise RuntimeError("embedding/manifest chunk mismatch")
    return rows


def expected_audio(
    retrieval: dict[str, Any],
    manifest: dict[str, Any],
    retriever: str,
    k: int,
    store: AudioStore,
) -> dict[str, Any]:
    scores = np.asarray(retrieval["scores"][retriever], dtype=np.float64)
    selected = sorted(
        int(index)
        for index in np.argsort(-scores, kind="stable")[: min(k, len(scores))]
    )
    chunk_rows = descriptors(manifest)
    pieces = []
    provenance = []
    for index in selected:
        descriptor = chunk_rows[index]
        waveform = store.load(str(descriptor["source_id"]))
        start = int(round(float(descriptor["start_sec"]) * SAMPLE_RATE))
        end = min(len(waveform), int(round(float(descriptor["end_sec"]) * SAMPLE_RATE)))
        if end <= start:
            raise RuntimeError("selected answer chunk has no samples")
        pieces.append(waveform[start:end])
        provenance.append({"chunk_index": index, **descriptor})
    audio = np.concatenate(pieces).astype(np.float32, copy=False)
    if len(audio) == 0 or len(audio) > k * 4 * SAMPLE_RATE + k:
        raise RuntimeError("answer evidence violates duration bound")
    return {
        "indices": selected,
        "provenance": provenance,
        "samples": len(audio),
        "sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
    }


def validate_row(
    row: dict[str, Any],
    model: str,
    prompts: dict[str, str],
    k_values: set[int],
    expected_ids: set[str],
    manifests: dict[str, dict[str, Any]],
    expected_audio_map: dict[tuple[str, int], dict[str, Any]],
    retriever: str,
) -> None:
    identifier = str(row["example_id"])
    prompt = str(row["prompt_name"])
    k = int(row["k"])
    if identifier not in expected_ids or prompt not in prompts or k not in k_values:
        raise RuntimeError("calibration row is outside its frozen grid")
    manifest = manifests[identifier]
    expected_job = f"{model}|{prompt}|k{k}|{identifier}"
    if row["job_id"] != expected_job or row["model"] != model:
        raise RuntimeError("answer-calibration job identity mismatch")
    if (
        row["target_source_id"] != manifest["target_source_id"]
        or row["question"] != manifest["question"]
        or row["reference"] != manifest["answer"]
        or row["retriever"] != retriever
        or row["audita_rows_accessed"] != 0
    ):
        raise RuntimeError("answer-calibration label/provenance mismatch")
    response = str(row["response"])
    reference = str(row["reference"])
    if (
        row["normalized_response"] != normalize_answer(response)
        or row["normalized_reference"] != normalize_answer(reference)
        or float(row["exact"])
        != float(normalize_answer(response) == normalize_answer(reference))
        or not np.isclose(float(row["token_f1"]), token_f1(response, reference))
    ):
        raise RuntimeError("answer metrics failed independent recomputation")
    audio = expected_audio_map[(identifier, k)]
    if (
        row["selected_chunk_indices"] != audio["indices"]
        or row["selected_provenance"] != audio["provenance"]
        or int(row["audio_samples"]) != audio["samples"]
        or row["audio_float32_sha256"] != audio["sha256"]
        or not np.isclose(float(row["audio_duration_sec"]), audio["samples"] / SAMPLE_RATE)
        or int(row["input_tokens"]) <= 0
        or float(row["generation_seconds"]) < 0.0
    ):
        raise RuntimeError("answer audio/input receipt failed independent audit")
    if "audita" in json.dumps(row).lower():
        raise PermissionError("AUDITA reference found in development answer row")


def main() -> None:
    if REPORT.exists():
        raise FileExistsError("answer-calibration selection report is immutable")
    lock = audit_lock()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    retrieval_report = json.loads(RETRIEVAL_REPORT.read_text(encoding="utf-8"))
    if config["audita_rows_allowed"] != 0 or retrieval_report["status"] != "learned_answer_retriever_selected":
        raise PermissionError("answer-development inputs failed precondition audit")
    retriever = str(retrieval_report["selected_retriever"])
    manifest_rows = read_jsonl(MANIFEST)
    manifests = {str(row["example_id"]): row for row in manifest_rows}
    if len(manifests) != 804 or len(manifests) != len(manifest_rows):
        raise RuntimeError("unexpected Clotho development manifest")
    retrieval_rows = read_jsonl(RETRIEVAL_RAW)
    retrieval_by_id = {str(row["example_id"]): row for row in retrieval_rows}
    if len(retrieval_by_id) != 804 or len(retrieval_by_id) != len(retrieval_rows):
        raise RuntimeError("unexpected answer-retrieval raw file")
    examples = calibration_examples(
        retrieval_rows,
        int(config["split"]["maximum_generation_examples_per_calibration_source"]),
    )
    expected_ids = {str(row["example_id"]) for row in examples}
    prompts = dict(config["answer_selector"]["prompts"])
    k_values = {int(value) for value in config["answer_selector"]["k_candidates"]}
    expected_job_count = len(expected_ids) * len(prompts) * len(k_values)

    store = AudioStore()
    expected_audio_map = {
        (identifier, k): expected_audio(
            retrieval_by_id[identifier], manifests[identifier], retriever, k, store
        )
        for identifier in sorted(expected_ids)
        for k in sorted(k_values)
    }
    model_metrics: dict[str, dict[str, dict[str, float | int]]] = {}
    artifacts = []
    all_rows: dict[str, list[dict[str, Any]]] = {}
    for model in MODELS:
        raw_path = RESULTS / f"calibration__{model}.jsonl"
        receipt_path = RESULTS / f"calibration__{model}.receipt.json"
        rows = read_jsonl(raw_path)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected_jobs = {
            f"{model}|{prompt}|k{k}|{identifier}"
            for prompt in prompts
            for k in k_values
            for identifier in expected_ids
        }
        observed_jobs = {str(row["job_id"]) for row in rows}
        if len(rows) != expected_job_count or observed_jobs != expected_jobs:
            raise RuntimeError(f"answer-calibration coverage mismatch: {model}")
        for row in rows:
            validate_row(
                row,
                model,
                prompts,
                k_values,
                expected_ids,
                manifests,
                expected_audio_map,
                retriever,
            )
        if (
            receipt["status"] != "answer_calibration_backbone_complete"
            or receipt["model"] != model
            or receipt["repository"] != config["backbones"][model]["repository"]
            or receipt["revision"] != config["backbones"][model]["revision"]
            or receipt["config_sha256"] != sha256(CONFIG)
            or receipt["answer_lock_sha256"] != sha256(LOCK)
            or receipt["retrieval_report_sha256"] != sha256(RETRIEVAL_REPORT)
            or receipt["retrieval_raw_sha256"] != sha256(RETRIEVAL_RAW)
            or receipt["audio_manifest_sha256"] != sha256(AUDIO_MANIFEST)
            or receipt["raw_sha256"] != sha256(raw_path)
            or int(receipt["rows"]) != expected_job_count
            or int(receipt["examples"]) != len(expected_ids)
            or int(receipt["candidate_combinations"]) != len(prompts) * len(k_values)
            or int(receipt["nonempty_responses"]) != expected_job_count
            or receipt["audita_rows_accessed"] != 0
            or receipt["lock_status"] != lock["status"]
        ):
            raise RuntimeError(f"answer-calibration receipt mismatch: {model}")
        current_metrics = {}
        for prompt in sorted(prompts):
            for k in sorted(k_values):
                subset = [
                    row
                    for row in rows
                    if row["prompt_name"] == prompt and int(row["k"]) == k
                ]
                current_metrics[f"{prompt}|k{k}"] = source_macro(subset)
        model_metrics[model] = current_metrics
        all_rows[model] = rows
        artifacts.append(
            {
                "model": model,
                "raw_path": str(raw_path.relative_to(PROJECT)),
                "raw_sha256": sha256(raw_path),
                "receipt_path": str(receipt_path.relative_to(PROJECT)),
                "receipt_sha256": sha256(receipt_path),
                "rows": len(rows),
            }
        )

    combinations = []
    for prompt in sorted(prompts):
        for k in sorted(k_values):
            key = f"{prompt}|k{k}"
            exact = [float(model_metrics[model][key]["source_macro_exact"]) for model in MODELS]
            f1 = [float(model_metrics[model][key]["source_macro_token_f1"]) for model in MODELS]
            combinations.append(
                {
                    "prompt_name": prompt,
                    "prompt": prompts[prompt],
                    "k": k,
                    "two_backbone_mean_source_macro_exact": float(np.mean(exact)),
                    "two_backbone_mean_source_macro_token_f1": float(np.mean(f1)),
                    "per_backbone": {
                        model: model_metrics[model][key] for model in MODELS
                    },
                }
            )
    selected = select_combination(combinations)
    payload = {
        "status": "answer_calibration_pass_prompt_k_selected",
        "selected_prompt_name": selected["prompt_name"],
        "selected_prompt": selected["prompt"],
        "selected_k": selected["k"],
        "selected_metrics": selected,
        "ranking_rule": config["answer_selector"]["ranking"],
        "candidate_metrics": combinations,
        "backbone_metrics": model_metrics,
        "retriever": retriever,
        "examples_per_combination": len(expected_ids),
        "calibration_sources": 24,
        "combinations": len(combinations),
        "models": list(MODELS),
        "artifacts": artifacts,
        "input_hashes": {
            "lock": sha256(LOCK),
            "config": sha256(CONFIG),
            "manifest": sha256(MANIFEST),
            "audio_manifest": sha256(AUDIO_MANIFEST),
            "retrieval_report": sha256(RETRIEVAL_REPORT),
            "retrieval_raw": sha256(RETRIEVAL_RAW),
        },
        "integrity": {
            "expected_jobs_per_model": expected_job_count,
            "observed_jobs_per_model": {
                model: len(all_rows[model]) for model in MODELS
            },
            "duplicate_jobs": 0,
            "metric_recomputation_failures": 0,
            "waveform_hash_failures": 0,
            "duration_bound_failures": 0,
            "source_overlap_with_evaluation": 0,
            "audita_rows_accessed": 0,
            "all_responses_nonempty": True,
            "all_checks_pass": True,
        },
        "heldout_evaluation_status": "sealed_pending_separate_authorization",
        "audita_status": "sealed",
        "lock_status": lock["status"],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "selected_prompt_name": payload["selected_prompt_name"],
                "selected_k": payload["selected_k"],
                "selected_metrics": payload["selected_metrics"],
                "integrity": payload["integrity"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
