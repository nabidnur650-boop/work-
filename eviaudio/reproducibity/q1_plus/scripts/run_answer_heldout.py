#!/usr/bin/env python3
"""Run one authorization-locked answer evaluation on held-out Clotho sources."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

import run_answer_calibration as base


PROJECT = Path(__file__).resolve().parents[2]
Q1 = PROJECT / "q1_plus"
CONFIG = Q1 / "configs/answer_heldout.json"
ANSWER_CONFIG = Q1 / "configs/answer_development.json"
ANSWER_LOCK = Q1 / "ANSWER_DEVELOPMENT_LOCK.json"
AUTHORIZATION = Q1 / "ANSWER_HELDOUT_AUTHORIZATION.json"
CALIBRATION = Q1 / "results/development/answer_generation/calibration_selection_report.json"
MANIFEST = PROJECT / "journal_suite/data/manifests/val.jsonl"
RETRIEVAL_REPORT = Q1 / "results/development/answer_retrieval/selection_report.json"
RETRIEVAL_RAW = Q1 / "results/development/answer_retrieval/retrieval_scores.jsonl.gz"
RESULTS = Q1 / "results/development/answer_generation"
SAMPLE_RATE = base.SAMPLE_RATE
MODELS = base.MODELS


def sha256(path: Path) -> str:
    return base.sha256(path)


def audit_authorization() -> tuple[dict[str, Any], dict[str, Any]]:
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    if authorization["status"] != "heldout_answer_evaluation_authorized_once":
        raise PermissionError("held-out answer evaluation is not authorized")
    if authorization["audita_status"] != "sealed":
        raise PermissionError("AUDITA is not sealed")
    for relative, expected in authorization["files"].items():
        if sha256(PROJECT / relative) != expected:
            raise RuntimeError(f"held-out authorization mismatch: {relative}")
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    if (
        calibration["status"] != "answer_calibration_pass_prompt_k_selected"
        or authorization["calibration_report_sha256"] != sha256(CALIBRATION)
        or authorization["selected_prompt_name"] != calibration["selected_prompt_name"]
        or authorization["selected_prompt"] != calibration["selected_prompt"]
        or int(authorization["selected_k"]) != int(calibration["selected_k"])
        or authorization["selected_retriever"] != calibration["retriever"]
        or authorization["answer_lock_sha256"] != sha256(ANSWER_LOCK)
    ):
        raise PermissionError("answer calibration did not authorize this held-out run")
    return authorization, calibration


def system_indices(
    retrieval: dict[str, Any],
    system: str,
    k: int,
    learned_retriever: str,
) -> list[int]:
    chunk_count = len(retrieval["chunk_source_indices"])
    count = min(k, chunk_count)
    if system in {"selected_learned_retrieval", "selected_retrieval_silenced"}:
        scores = np.asarray(retrieval["scores"][learned_retriever], dtype=np.float64)
        selected = np.argsort(-scores, kind="stable")[:count]
    elif system == "clap_retrieval":
        scores = np.asarray(retrieval["scores"]["clap_prior"], dtype=np.float64)
        selected = np.argsort(-scores, kind="stable")[:count]
    elif system == "deterministic_random_retrieval":
        selected = np.asarray(
            sorted(
                range(chunk_count),
                key=lambda index: hashlib.sha256(
                    f"answer-heldout-random|{retrieval['example_id']}|{index}".encode()
                ).hexdigest(),
            )[:count],
            dtype=np.int64,
        )
    elif system == "uniform_retrieval":
        selected = (
            np.arange(chunk_count, dtype=np.int64)
            if chunk_count <= k
            else np.rint(np.linspace(0, chunk_count - 1, count)).astype(np.int64)
        )
        if len(np.unique(selected)) != count:
            raise RuntimeError("uniform retrieval did not select unique chunks")
    elif system == "prefix_matched_retrieval":
        selected = np.arange(count, dtype=np.int64)
    elif system == "oracle_retrieval":
        labels = np.asarray(retrieval["evidence_targets"], dtype=np.int64)
        scores = np.asarray(retrieval["scores"][learned_retriever], dtype=np.float64)
        positive = [int(index) for index in np.flatnonzero(labels)]
        negative = [int(index) for index in np.flatnonzero(1 - labels)]
        positive.sort(key=lambda index: (-scores[index], index))
        negative.sort(key=lambda index: (-scores[index], index))
        selected = np.asarray((positive + negative)[:count], dtype=np.int64)
    else:
        raise ValueError(f"system does not select audio chunks: {system}")
    result = sorted(int(index) for index in selected)
    if len(result) != count or len(set(result)) != count:
        raise RuntimeError(f"invalid held-out chunk selection: {system}")
    return result


def audio_from_indices(
    indices: list[int], manifest: dict[str, Any], store: base.AudioStore
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    descriptors = base.chunk_descriptors(manifest)
    pieces = []
    provenance = []
    for index in indices:
        descriptor = descriptors[index]
        waveform = store.load(str(descriptor["source_id"]))
        start = int(round(float(descriptor["start_sec"]) * SAMPLE_RATE))
        end = min(len(waveform), int(round(float(descriptor["end_sec"]) * SAMPLE_RATE)))
        if end <= start:
            raise RuntimeError("held-out selected chunk has no waveform samples")
        pieces.append(waveform[start:end])
        provenance.append({"chunk_index": index, **descriptor})
    audio = np.concatenate(pieces).astype(np.float32, copy=False)
    if len(audio) == 0 or len(audio) > len(indices) * 4 * SAMPLE_RATE + len(indices):
        raise RuntimeError("held-out audio violates the reservoir duration bound")
    return audio, provenance


def text_generator(
    model_name: str,
    model: Any,
    processor: Any,
    generation_config: Any,
    device: torch.device,
) -> Callable[[str], tuple[str, int]]:
    @torch.inference_mode()
    def generate_qwen(prompt: str) -> tuple[str, int]:
        conversation = [
            {"role": "system", "content": "You are a precise audio analyst."},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = base.move_inputs(
            processor.tokenizer(text, return_tensors="pt", padding=True), device
        )
        tokens = int(inputs["input_ids"].shape[1])
        generated = model.generate(
            **inputs, max_new_tokens=12, do_sample=False, use_cache=True
        )[:, tokens:]
        return (
            processor.tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip(),
            tokens,
        )

    @torch.inference_mode()
    def generate_phi(prompt: str) -> tuple[str, int]:
        text = f"<|user|>{prompt}<|end|><|assistant|>"
        inputs = base.move_inputs(
            processor(text=text, return_tensors="pt"), device
        )
        tokens = int(inputs["input_ids"].shape[1])
        generated = model.generate(
            **inputs,
            max_new_tokens=12,
            do_sample=False,
            use_cache=True,
            generation_config=generation_config,
        )[:, tokens:]
        return (
            processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip(),
            tokens,
        )

    return generate_qwen if model_name == "qwen2_audio" else generate_phi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    args = parser.parse_args()
    authorization, _ = audit_authorization()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["audita_rows_allowed"] != 0 or config["audita_status"] != "sealed":
        raise PermissionError("held-out config permits AUDITA access")
    systems = list(config["systems"])
    if systems != authorization["systems"]:
        raise RuntimeError("authorized held-out system list changed")
    manifests_list = base.read_jsonl(MANIFEST)
    manifests = {str(row["example_id"]): row for row in manifests_list}
    with gzip.open(RETRIEVAL_RAW, "rt", encoding="utf-8") as handle:
        retrieval_rows = [json.loads(line) for line in handle if line.strip()]
    evaluation = sorted(
        (row for row in retrieval_rows if row["split"] == "evaluation"),
        key=lambda row: row["example_id"],
    )
    if (
        len(evaluation) != int(config["expected_examples"])
        or len({row["target_source_id"] for row in evaluation})
        != int(config["expected_sources"])
    ):
        raise RuntimeError("held-out source/example contract changed")
    learned_retriever = str(authorization["selected_retriever"])
    prompt_template = str(authorization["selected_prompt"])
    k = int(authorization["selected_k"])
    expected_jobs = {
        f"{args.model}|{system}|{row['example_id']}"
        for system in systems
        for row in evaluation
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    raw_path = RESULTS / f"heldout__{args.model}.jsonl"
    receipt_path = RESULTS / f"heldout__{args.model}.receipt.json"
    if receipt_path.exists():
        raise FileExistsError(f"held-out receipt is immutable: {receipt_path}")
    completed: dict[str, dict[str, Any]] = {}
    if raw_path.exists():
        for row in base.read_jsonl(raw_path):
            identifier = str(row["job_id"])
            if identifier in completed:
                raise RuntimeError(f"duplicate held-out job: {identifier}")
            completed[identifier] = row
    if not set(completed).issubset(expected_jobs):
        raise RuntimeError("held-out raw file contains an undeclared job")

    if not torch.cuda.is_available():
        raise RuntimeError("held-out answer evaluation requires CUDA")
    device = torch.device("cuda:0")
    device_index = 0
    torch.cuda.set_device(device_index)
    torch.empty(1, device=device)
    torch.manual_seed(20_260_722)
    torch.cuda.reset_peak_memory_stats(device_index)
    loaded = time.perf_counter()
    model, processor, generation_config, generate_audio = base.load_backbone(
        args.model, MODELS[args.model], device
    )
    generate_text = text_generator(
        args.model, model, processor, generation_config, device
    )
    load_seconds = time.perf_counter() - loaded
    store = base.AudioStore(base.AUDIO_MANIFEST)
    started = time.perf_counter()
    written = 0
    with raw_path.open("a", encoding="utf-8") as output:
        for system in systems:
            for retrieval in evaluation:
                job_id = f"{args.model}|{system}|{retrieval['example_id']}"
                if job_id in completed:
                    continue
                manifest = manifests[str(retrieval["example_id"])]
                prompt = prompt_template.format(question=str(manifest["question"]))
                indices: list[int] = []
                provenance: list[dict[str, Any]] = []
                audio: np.ndarray | None = None
                if system != "text_only":
                    indices = system_indices(retrieval, system, k, learned_retriever)
                    audio, provenance = audio_from_indices(indices, manifest, store)
                    if system == "selected_retrieval_silenced":
                        audio = np.zeros_like(audio)
                generated_started = time.perf_counter()
                if system == "text_only":
                    response, input_tokens = generate_text(prompt)
                else:
                    assert audio is not None
                    response, input_tokens = generate_audio(prompt, audio)
                torch.cuda.synchronize(device_index)
                generation_seconds = time.perf_counter() - generated_started
                evidence = np.asarray(retrieval["evidence_targets"], dtype=np.int64)
                row = {
                    "job_id": job_id,
                    "model": args.model,
                    "system": system,
                    "example_id": str(manifest["example_id"]),
                    "target_source_id": str(manifest["target_source_id"]),
                    "difficulty": str(manifest["difficulty"]),
                    "n_sources": int(manifest["n_sources"]),
                    "n_chunks": int(manifest["n_chunks"]),
                    "target_position_bin": str(manifest["target_position_bin"]),
                    "question": str(manifest["question"]),
                    "reference": str(manifest["answer"]),
                    "response": response,
                    "normalized_reference": base.normalize_answer(str(manifest["answer"])),
                    "normalized_response": base.normalize_answer(response),
                    "exact": float(
                        base.normalize_answer(response)
                        == base.normalize_answer(str(manifest["answer"]))
                    ),
                    "token_f1": base.token_f1(response, str(manifest["answer"])),
                    "selected_chunk_indices": indices,
                    "selected_provenance": provenance,
                    "selected_positive_chunks": int(evidence[indices].sum()) if indices else 0,
                    "selected_chunk_count": len(indices),
                    "audio_samples": 0 if audio is None else len(audio),
                    "audio_duration_sec": 0.0 if audio is None else len(audio) / SAMPLE_RATE,
                    "audio_float32_sha256": None
                    if audio is None
                    else hashlib.sha256(audio.tobytes()).hexdigest(),
                    "input_tokens": input_tokens,
                    "generation_seconds": generation_seconds,
                    "audita_rows_accessed": 0,
                }
                output.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
                completed[job_id] = row
                written += 1
                if written % 20 == 0:
                    print(
                        json.dumps(
                            {
                                "model": args.model,
                                "completed": len(completed),
                                "total": len(expected_jobs),
                                "system": system,
                                "elapsed_seconds": time.perf_counter() - started,
                            }
                        ),
                        flush=True,
                    )
    if set(completed) != expected_jobs or len(completed) != len(expected_jobs):
        raise RuntimeError("held-out run did not complete the frozen job set")
    receipt = {
        "status": "heldout_answer_backbone_complete",
        "model": args.model,
        "repository": json.loads(ANSWER_CONFIG.read_text(encoding="utf-8"))[
            "backbones"
        ][args.model]["repository"],
        "revision": json.loads(ANSWER_CONFIG.read_text(encoding="utf-8"))[
            "backbones"
        ][args.model]["revision"],
        "authorization_sha256": sha256(AUTHORIZATION),
        "heldout_config_sha256": sha256(CONFIG),
        "calibration_report_sha256": sha256(CALIBRATION),
        "retrieval_raw_sha256": sha256(RETRIEVAL_RAW),
        "raw_path": str(raw_path.relative_to(PROJECT)),
        "raw_sha256": sha256(raw_path),
        "rows": len(completed),
        "examples": len(evaluation),
        "sources": len({row["target_source_id"] for row in evaluation}),
        "systems": systems,
        "selected_prompt_name": authorization["selected_prompt_name"],
        "selected_k": k,
        "selected_retriever": learned_retriever,
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device_index),
        "load_seconds": load_seconds,
        "run_seconds": time.perf_counter() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device_index)),
        "nonempty_responses": sum(bool(row["response"].strip()) for row in completed.values()),
        "audita_rows_accessed": 0,
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
