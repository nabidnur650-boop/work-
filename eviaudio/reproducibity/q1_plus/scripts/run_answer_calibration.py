#!/usr/bin/env python3
"""Run the frozen prompt/K grid on deterministic Clotho calibration examples."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import string
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
    Qwen2AudioForConditionalGeneration,
)


PROJECT = Path(__file__).resolve().parents[2]
Q1 = PROJECT / "q1_plus"
CONFIG = Q1 / "configs/answer_development.json"
LOCK = Q1 / "ANSWER_DEVELOPMENT_LOCK.json"
MANIFEST = PROJECT / "journal_suite/data/manifests/val.jsonl"
AUDIO_MANIFEST = Q1 / "data/development_audio/manifest.json"
RETRIEVAL_REPORT = Q1 / "results/development/answer_retrieval/selection_report.json"
RETRIEVAL_RAW = Q1 / "results/development/answer_retrieval/retrieval_scores.jsonl.gz"
RESULTS = Q1 / "results/development/answer_generation"
SAMPLE_RATE = 16_000
MODELS = {
    "qwen2_audio": Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen2-Audio-7B-Instruct/snapshots/0a095220c30b7b31434169c3086508ef3ea5bf0a",
    "phi4_multimodal": Path.home()
    / ".cache/huggingface/hub/models--microsoft--Phi-4-multimodal-instruct/snapshots/93f923e1a7727d1c4f446756212d9d3e8fcc5d81",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


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
    counts: dict[str, int] = defaultdict(int)
    for token in predicted:
        counts[token] += 1
    overlap = 0
    for token in expected:
        if counts[token] > 0:
            overlap += 1
            counts[token] -= 1
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2.0 * precision * recall / (precision + recall) if overlap else 0.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
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


class AudioStore:
    def __init__(self, audio_manifest: Path) -> None:
        manifest = json.loads(audio_manifest.read_text(encoding="utf-8"))
        if manifest["audita_rows_accessed"] != 0 or manifest["sources"] != 281:
            raise RuntimeError("development-audio manifest is not label sealed")
        self.paths = {
            str(row["source_id"]): Q1 / str(row["path"])
            for row in manifest["records"]
        }
        self.expected_hashes = {
            str(row["source_id"]): str(row["sha256"])
            for row in manifest["records"]
        }
        self.cache: dict[str, np.ndarray] = {}

    def load(self, source_id: str) -> np.ndarray:
        if source_id in self.cache:
            return self.cache[source_id]
        path = self.paths[source_id]
        if "audita" in str(path).lower():
            raise PermissionError("AUDITA paths are forbidden during answer development")
        if sha256(path) != self.expected_hashes[source_id]:
            raise RuntimeError(f"development audio checksum mismatch: {source_id}")
        waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if sample_rate != SAMPLE_RATE:
            divisor = int(np.gcd(sample_rate, SAMPLE_RATE))
            waveform = resample_poly(
                waveform, SAMPLE_RATE // divisor, sample_rate // divisor
            ).astype(np.float32)
        if waveform.ndim != 1 or not np.isfinite(waveform).all():
            raise RuntimeError(f"invalid development audio: {source_id}")
        self.cache[source_id] = waveform.astype(np.float32, copy=False)
        return self.cache[source_id]


def chunk_descriptors(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    descriptors = []
    for source_index, (source_id, embedding_text) in enumerate(
        zip(manifest["source_ids"], manifest["embedding_paths"], strict=True)
    ):
        embedding_path = (PROJECT / "journal_suite/data/manifests" / embedding_text).resolve()
        expected_root = (PROJECT / "journal_suite/data/embeddings").resolve()
        if embedding_path.parent != expected_root:
            raise RuntimeError("embedding path escaped the development directory")
        archive = np.load(embedding_path, allow_pickle=False)
        starts = archive["start_sec"].astype(np.float64)
        ends = archive["end_sec"].astype(np.float64)
        for start, end in zip(starts, ends, strict=True):
            descriptors.append(
                {
                    "source_index": source_index,
                    "source_id": str(source_id),
                    "start_sec": float(start),
                    "end_sec": float(end),
                }
            )
    if len(descriptors) != int(manifest["n_chunks"]):
        raise RuntimeError(f"chunk descriptor mismatch: {manifest['example_id']}")
    return descriptors


def selected_audio(
    retrieval: dict[str, Any],
    manifest: dict[str, Any],
    retriever: str,
    k: int,
    store: AudioStore,
) -> tuple[np.ndarray, list[int], list[dict[str, Any]]]:
    scores = np.asarray(retrieval["scores"][retriever], dtype=np.float64)
    if len(scores) != int(manifest["n_chunks"]) or not np.isfinite(scores).all():
        raise RuntimeError("retrieval score/chunk mismatch")
    ranked = np.argsort(-scores, kind="stable")[: min(k, len(scores))]
    selected = sorted(int(index) for index in ranked)
    descriptors = chunk_descriptors(manifest)
    pieces = []
    provenance = []
    for index in selected:
        descriptor = descriptors[index]
        waveform = store.load(descriptor["source_id"])
        start = int(round(descriptor["start_sec"] * SAMPLE_RATE))
        end = min(len(waveform), int(round(descriptor["end_sec"] * SAMPLE_RATE)))
        if end <= start:
            raise RuntimeError("selected chunk has no waveform samples")
        pieces.append(waveform[start:end])
        provenance.append({"chunk_index": index, **descriptor})
    audio = np.concatenate(pieces).astype(np.float32, copy=False)
    if len(audio) == 0 or len(audio) > k * 4 * SAMPLE_RATE + k:
        raise RuntimeError("selected reservoir violates its duration bound")
    return audio, selected, provenance


def move_inputs(inputs: Any, device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


def load_backbone(
    name: str, snapshot: Path, device: torch.device
) -> tuple[Any, Any, Any, Callable[[str, np.ndarray], tuple[str, int]]]:
    if name == "qwen2_audio":
        processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
        if int(processor.feature_extractor.sampling_rate) != SAMPLE_RATE:
            raise RuntimeError("Qwen2-Audio sampling-rate contract changed")
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            snapshot,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).to(device)
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        generation_config = None

        @torch.inference_mode()
        def generate(prompt: str, audio: np.ndarray) -> tuple[str, int]:
            conversation = [
                {"role": "system", "content": "You are a precise audio analyst."},
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": "bounded-evidence.wav"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            text = processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            inputs = move_inputs(
                processor(
                    text=text,
                    audios=[audio],
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt",
                    padding=True,
                ),
                device,
            )
            tokens = int(inputs["input_ids"].shape[1])
            generated = model.generate(
                **inputs, max_new_tokens=12, do_sample=False, use_cache=True
            )[:, tokens:]
            response = processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            return response, tokens

    else:
        processor = AutoProcessor.from_pretrained(
            snapshot, trust_remote_code=True, local_files_only=True
        )
        if int(processor.audio_processor.sampling_rate) != SAMPLE_RATE:
            raise RuntimeError("Phi-4 audio sampling-rate contract changed")
        model = AutoModelForCausalLM.from_pretrained(
            snapshot,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            _attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).to(device)
        generation_config = GenerationConfig.from_pretrained(
            snapshot, local_files_only=True
        )

        @torch.inference_mode()
        def generate(prompt: str, audio: np.ndarray) -> tuple[str, int]:
            text = f"<|user|><|audio_1|>{prompt}<|end|><|assistant|>"
            inputs = move_inputs(
                processor(
                    text=text,
                    audios=[(audio, SAMPLE_RATE)],
                    return_tensors="pt",
                ),
                device,
            )
            tokens = int(inputs["input_ids"].shape[1])
            generated = model.generate(
                **inputs,
                max_new_tokens=12,
                do_sample=False,
                use_cache=True,
                generation_config=generation_config,
            )[:, tokens:]
            response = processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            return response, tokens

    model.eval()
    return model, processor, generation_config, generate


def expected_jobs(
    examples: list[dict[str, Any]], prompts: dict[str, str], k_values: list[int]
) -> list[tuple[str, int, dict[str, Any]]]:
    return [
        (prompt_name, int(k), example)
        for prompt_name in sorted(prompts)
        for k in sorted(k_values)
        for example in examples
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    args = parser.parse_args()
    lock = audit_lock()
    if not torch.cuda.is_available():
        raise RuntimeError("answer calibration requires CUDA")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["audita_rows_allowed"] != 0:
        raise PermissionError("answer-development config permits AUDITA access")
    retrieval_report = json.loads(RETRIEVAL_REPORT.read_text(encoding="utf-8"))
    retriever = str(retrieval_report["selected_retriever"])
    if retrieval_report["status"] != "learned_answer_retriever_selected":
        raise PermissionError("learned answer retriever did not pass calibration")
    manifest_rows = read_jsonl(MANIFEST)
    manifests = {str(row["example_id"]): row for row in manifest_rows}
    with gzip.open(RETRIEVAL_RAW, "rt", encoding="utf-8") as handle:
        retrieval_rows = [json.loads(line) for line in handle if line.strip()]
    examples = calibration_examples(
        retrieval_rows,
        int(config["split"]["maximum_generation_examples_per_calibration_source"]),
    )
    prompts = dict(config["answer_selector"]["prompts"])
    k_values = [int(value) for value in config["answer_selector"]["k_candidates"]]
    jobs = expected_jobs(examples, prompts, k_values)
    RESULTS.mkdir(parents=True, exist_ok=True)
    raw_path = RESULTS / f"calibration__{args.model}.jsonl"
    receipt_path = RESULTS / f"calibration__{args.model}.receipt.json"
    if receipt_path.exists():
        raise FileExistsError(f"calibration receipt is immutable: {receipt_path}")
    completed: dict[str, dict[str, Any]] = {}
    if raw_path.exists():
        for row in read_jsonl(raw_path):
            key = str(row["job_id"])
            if key in completed:
                raise RuntimeError(f"duplicate existing calibration job: {key}")
            completed[key] = row
    expected_ids = {
        f"{args.model}|{prompt_name}|k{k}|{example['example_id']}"
        for prompt_name, k, example in jobs
    }
    if not set(completed).issubset(expected_ids):
        raise RuntimeError("existing calibration file contains undeclared jobs")

    device = torch.device("cuda:0")
    device_index = 0
    torch.cuda.set_device(device_index)
    torch.empty(1, device=device)
    torch.manual_seed(20_260_722)
    torch.cuda.reset_peak_memory_stats(device_index)
    loaded = time.perf_counter()
    model, processor, generation_config, generate = load_backbone(
        args.model, MODELS[args.model], device
    )
    load_seconds = time.perf_counter() - loaded
    store = AudioStore(AUDIO_MANIFEST)
    started = time.perf_counter()
    written = 0
    with raw_path.open("a", encoding="utf-8") as output:
        for prompt_name, k, retrieval in jobs:
            job_id = f"{args.model}|{prompt_name}|k{k}|{retrieval['example_id']}"
            if job_id in completed:
                continue
            manifest = manifests[str(retrieval["example_id"])]
            audio, indices, provenance = selected_audio(
                retrieval, manifest, retriever, k, store
            )
            prompt = prompts[prompt_name].format(question=str(manifest["question"]))
            generated_started = time.perf_counter()
            response, input_tokens = generate(prompt, audio)
            torch.cuda.synchronize(device_index)
            generation_seconds = time.perf_counter() - generated_started
            row = {
                "job_id": job_id,
                "model": args.model,
                "prompt_name": prompt_name,
                "k": k,
                "example_id": str(manifest["example_id"]),
                "target_source_id": str(manifest["target_source_id"]),
                "question": str(manifest["question"]),
                "reference": str(manifest["answer"]),
                "response": response,
                "normalized_reference": normalize_answer(str(manifest["answer"])),
                "normalized_response": normalize_answer(response),
                "exact": float(
                    normalize_answer(response) == normalize_answer(str(manifest["answer"]))
                ),
                "token_f1": token_f1(response, str(manifest["answer"])),
                "retriever": retriever,
                "selected_chunk_indices": indices,
                "selected_provenance": provenance,
                "audio_samples": len(audio),
                "audio_duration_sec": len(audio) / SAMPLE_RATE,
                "audio_float32_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
                "input_tokens": input_tokens,
                "generation_seconds": generation_seconds,
                "audita_rows_accessed": 0,
            }
            output.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
            completed[job_id] = row
            written += 1
            if written % 10 == 0:
                print(
                    json.dumps(
                        {
                            "model": args.model,
                            "completed": len(completed),
                            "total": len(jobs),
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    if set(completed) != expected_ids or len(completed) != len(jobs):
        raise RuntimeError("answer calibration did not complete every declared job")
    receipt = {
        "status": "answer_calibration_backbone_complete",
        "model": args.model,
        "repository": config["backbones"][args.model]["repository"],
        "revision": config["backbones"][args.model]["revision"],
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device_index),
        "answer_lock_sha256": sha256(LOCK),
        "config_sha256": sha256(CONFIG),
        "retrieval_report_sha256": sha256(RETRIEVAL_REPORT),
        "retrieval_raw_sha256": sha256(RETRIEVAL_RAW),
        "audio_manifest_sha256": sha256(AUDIO_MANIFEST),
        "raw_path": str(raw_path.relative_to(PROJECT)),
        "raw_sha256": sha256(raw_path),
        "rows": len(completed),
        "examples": len(examples),
        "candidate_combinations": len(prompts) * len(k_values),
        "load_seconds": load_seconds,
        "run_seconds": time.perf_counter() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device_index)),
        "nonempty_responses": sum(bool(row["response"].strip()) for row in completed.values()),
        "audita_rows_accessed": 0,
        "lock_status": lock["status"],
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
