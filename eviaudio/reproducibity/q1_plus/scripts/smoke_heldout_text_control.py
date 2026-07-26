#!/usr/bin/env python3
"""Verify the frozen text-only control path without opening held-out examples."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import run_answer_calibration as base
import run_answer_heldout as heldout


PROJECT = Path(__file__).resolve().parents[2]
Q1 = PROJECT / "q1_plus"
RESULTS = Q1 / "results/development/audio_backbone_smoke"
PROMPT = (
    "Answer with exactly yes or no. Is an audio recording supplied with this "
    "question?"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(base.MODELS), required=True)
    args = parser.parse_args()
    output = RESULTS / f"{args.model}_text_only.json"
    if output.exists():
        raise FileExistsError(f"text-only smoke receipt is immutable: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("text-only control smoke requires CUDA")
    device, device_index = torch.device("cuda:0"), 0
    torch.cuda.set_device(device_index)
    torch.empty(1, device=device)
    torch.manual_seed(20_260_722)
    torch.cuda.reset_peak_memory_stats(device_index)
    started = time.perf_counter()
    model, processor, generation_config, _ = base.load_backbone(
        args.model, base.MODELS[args.model], device
    )
    load_seconds = time.perf_counter() - started
    generate = heldout.text_generator(
        args.model, model, processor, generation_config, device
    )
    generated = time.perf_counter()
    response, input_tokens = generate(PROMPT)
    torch.cuda.synchronize(device_index)
    generation_seconds = time.perf_counter() - generated
    if not response.strip() or input_tokens <= 0:
        raise RuntimeError("text-only control path produced an invalid response")
    payload = {
        "status": "heldout_text_only_control_smoke_complete",
        "model": args.model,
        "prompt": PROMPT,
        "response": response,
        "input_tokens": input_tokens,
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device_index)),
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device_index),
        "audita_rows_accessed": 0,
        "heldout_examples_accessed": 0,
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
