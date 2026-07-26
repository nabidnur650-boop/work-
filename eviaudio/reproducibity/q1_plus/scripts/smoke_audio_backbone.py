#!/usr/bin/env python3
"""Load one pinned audio-language backbone and verify audio-conditioned generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

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
RESULTS = Q1 / "results/development/audio_backbone_smoke"
SAMPLE = Q1 / "data/external/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac"
MODELS = {
    "qwen2_audio": {
        "repository": "Qwen/Qwen2-Audio-7B-Instruct",
        "revision": "0a095220c30b7b31434169c3086508ef3ea5bf0a",
        "snapshot": Path.home()
        / ".cache/huggingface/hub/models--Qwen--Qwen2-Audio-7B-Instruct/snapshots/0a095220c30b7b31434169c3086508ef3ea5bf0a",
    },
    "phi4_multimodal": {
        "repository": "microsoft/Phi-4-multimodal-instruct",
        "revision": "93f923e1a7727d1c4f446756212d9d3e8fcc5d81",
        "snapshot": Path.home()
        / ".cache/huggingface/hub/models--microsoft--Phi-4-multimodal-instruct/snapshots/93f923e1a7727d1c4f446756212d9d3e8fcc5d81",
    },
}
QUESTION = "Transcribe the spoken words. Return only the transcript."
REFERENCE = (
    "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO "
    "WELCOME HIS GOSPEL"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_mono(path: Path) -> tuple[np.ndarray, int]:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if waveform.ndim != 1 or not np.isfinite(waveform).all() or len(waveform) == 0:
        raise RuntimeError("invalid smoke-test waveform")
    return waveform.astype(np.float32, copy=False), int(sample_rate)


def resample(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return waveform
    divisor = int(np.gcd(source_rate, target_rate))
    return resample_poly(
        waveform, target_rate // divisor, source_rate // divisor
    ).astype(np.float32)


def move_inputs(inputs: Any, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


def word_error_rate(hypothesis: str, reference: str) -> float:
    hypothesis_words = "".join(
        character.lower() if character.isalnum() else " " for character in hypothesis
    ).split()
    reference_words = "".join(
        character.lower() if character.isalnum() else " " for character in reference
    ).split()
    previous = list(range(len(hypothesis_words) + 1))
    for row, expected in enumerate(reference_words, start=1):
        current = [row]
        for column, observed in enumerate(hypothesis_words, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + int(expected != observed),
                )
            )
        previous = current
    return previous[-1] / len(reference_words)


@torch.inference_mode()
def generate_qwen(
    model: Any,
    processor: Any,
    waveform: np.ndarray,
    sample_rate: int,
    device: torch.device,
) -> tuple[str, int]:
    target_rate = int(processor.feature_extractor.sampling_rate)
    audio = resample(waveform, sample_rate, target_rate)
    conversation = [
        {"role": "system", "content": "You are a precise audio analyst."},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "local-smoke.flac"},
                {"type": "text", "text": QUESTION},
            ],
        },
    ]
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(text=text, audios=[audio], return_tensors="pt", padding=True)
    inputs = move_inputs(inputs, device)
    input_tokens = int(inputs["input_ids"].shape[1])
    generated = model.generate(
        **inputs,
        max_new_tokens=48,
        do_sample=False,
        use_cache=True,
    )
    generated = generated[:, input_tokens:]
    response = processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return response, input_tokens


@torch.inference_mode()
def generate_phi(
    model: Any,
    processor: Any,
    generation_config: Any,
    waveform: np.ndarray,
    sample_rate: int,
    device: torch.device,
) -> tuple[str, int]:
    prompt = f"<|user|><|audio_1|>{QUESTION}<|end|><|assistant|>"
    inputs = processor(
        text=prompt,
        audios=[(waveform, sample_rate)],
        return_tensors="pt",
    )
    inputs = move_inputs(inputs, device)
    input_tokens = int(inputs["input_ids"].shape[1])
    generated = model.generate(
        **inputs,
        max_new_tokens=48,
        do_sample=False,
        use_cache=True,
        generation_config=generation_config,
    )
    generated = generated[:, input_tokens:]
    response = processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return response, input_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--audio", type=Path, default=SAMPLE)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the pinned smoke protocol requires CUDA")
    spec = MODELS[args.model]
    snapshot = spec["snapshot"]
    if not snapshot.is_dir():
        raise FileNotFoundError(f"pinned model snapshot is incomplete: {snapshot}")
    waveform, sample_rate = read_mono(args.audio)
    device = torch.device("cuda:0")
    device_index = 0
    torch.cuda.set_device(device_index)
    torch.empty(1, device=device)
    torch.manual_seed(20_260_722)
    torch.cuda.reset_peak_memory_stats(device_index)
    loaded_started = time.perf_counter()

    if args.model == "qwen2_audio":
        processor = AutoProcessor.from_pretrained(
            snapshot, local_files_only=True
        )
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            snapshot,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).to(device)
        generation_config = None
        generator = lambda audio: generate_qwen(
            model, processor, audio, sample_rate, device
        )
    else:
        processor = AutoProcessor.from_pretrained(
            snapshot, trust_remote_code=True, local_files_only=True
        )
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
        generator = lambda audio: generate_phi(
            model, processor, generation_config, audio, sample_rate, device
        )
    model.eval()
    load_seconds = time.perf_counter() - loaded_started

    real_started = time.perf_counter()
    real_response, input_tokens = generator(waveform)
    torch.cuda.synchronize(device_index)
    real_seconds = time.perf_counter() - real_started
    silent_started = time.perf_counter()
    silent_response, silent_input_tokens = generator(np.zeros_like(waveform))
    torch.cuda.synchronize(device_index)
    silent_seconds = time.perf_counter() - silent_started
    if not real_response:
        raise RuntimeError("audio backbone produced an empty response")
    wer = word_error_rate(real_response, REFERENCE)
    if real_response == silent_response:
        raise RuntimeError("real and silent audio produced identical smoke responses")
    if wer > 0.75:
        raise RuntimeError(f"audio transcription smoke WER is too high: {wer:.3f}")

    payload = {
        "status": "audio_backbone_smoke_complete",
        "model": args.model,
        "repository": spec["repository"],
        "revision": spec["revision"],
        "snapshot": str(snapshot),
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device_index),
        "dtype": "bfloat16",
        "attention": "sdpa",
        "prompt": QUESTION,
        "reference": REFERENCE,
        "audio": str(args.audio.resolve()),
        "audio_sha256": sha256(args.audio),
        "sample_rate": sample_rate,
        "duration_sec": len(waveform) / sample_rate,
        "input_tokens": input_tokens,
        "silent_input_tokens": silent_input_tokens,
        "real_response": real_response,
        "silent_response": silent_response,
        "responses_differ": real_response != silent_response,
        "word_error_rate": wer,
        "maximum_allowed_word_error_rate": 0.75,
        "load_seconds": load_seconds,
        "real_generation_seconds": real_seconds,
        "silent_generation_seconds": silent_seconds,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device_index)),
        "audita_rows_accessed": 0,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    output = RESULTS / f"{args.model}.json"
    if output.exists():
        raise FileExistsError(f"smoke receipt is immutable: {output}")
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
