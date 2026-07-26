#!/usr/bin/env python3
"""Precompute pinned CLAP text embeddings for development questions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, ClapModel


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_DIR = PROJECT / "journal_suite" / "data" / "manifests"
DEFAULT_OUTPUT = PROJECT / "q1_plus" / "data" / "development_question_clap.npz"
MODEL = "laion/clap-htsat-unfused"
REVISION = "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_questions(manifest_dir: Path) -> tuple[list[str], dict[str, str]]:
    questions: set[str] = set()
    checksums: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = manifest_dir / f"{split}.jsonl"
        checksums[split] = sha256(path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    questions.add(str(json.loads(line)["question"]).strip())
    return sorted(questions), checksums


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    questions, manifest_checksums = load_questions(args.manifest_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION)
    model = ClapModel.from_pretrained(
        MODEL, revision=REVISION, use_safetensors=False
    ).to(device)
    model.eval()
    vectors: list[np.ndarray] = []
    for start in range(0, len(questions), args.batch_size):
        batch = questions[start : start + args.batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        embedding = model.get_text_features(**inputs)
        embedding = torch.nn.functional.normalize(embedding.float(), dim=-1)
        vectors.append(embedding.cpu().numpy())
        print(f"embedded {min(start + len(batch), len(questions))}/{len(questions)}", flush=True)
    matrix = np.concatenate(vectors, axis=0).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        questions=np.asarray(questions),
        embeddings=matrix.astype(np.float16),
        model=np.asarray(MODEL),
        revision=np.asarray(REVISION),
    )
    metadata = {
        "status": "complete",
        "scope": "exploratory_development_only",
        "model": MODEL,
        "revision": REVISION,
        "questions": len(questions),
        "dimension": int(matrix.shape[1]),
        "manifest_sha256": manifest_checksums,
        "output_sha256": sha256(args.output),
        "device": str(device),
        "torch": torch.__version__,
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
