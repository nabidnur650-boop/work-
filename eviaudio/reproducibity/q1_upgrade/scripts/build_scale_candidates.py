#!/usr/bin/env python3
"""Generate the six locked 0.5/1/2-second CLAP and QCR candidates."""

from __future__ import annotations

import gzip
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[2]
TRACK = PROJECT / "q1_upgrade"
LOCK = TRACK / "CROSSFIT_ANALYSIS_LOCK.json"
ROUTER_CONFIG = TRACK / "configs/crossfit_router.json"
NATURAL = PROJECT / "q1_top_tier"
PRECOMPUTE = NATURAL / "results/perception_test/precompute"
OUTPUT = TRACK / "results/scale_candidates"
sys.path.insert(0, str(TRACK / "src"))
sys.path.insert(0, str(NATURAL / "scripts"))
sys.path.insert(0, str(NATURAL / "src"))
sys.path.insert(0, str(PROJECT / "src"))

import evaluate_perception_test as frozen  # noqa: E402
from crossfit_utils import sha256, single_scale_name  # noqa: E402


GENERATED_SCALES = (0.5, 1.0, 2.0)


def project_path(relative: str) -> Path:
    path = (PROJECT / relative).resolve()
    if not path.is_relative_to(PROJECT):
        raise RuntimeError(f"path leaves project: {relative}")
    return path


def audit_analysis_lock() -> dict[str, Any]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["status"] != "post_outcome_crossfit_locked_before_new_scale_scoring":
        raise PermissionError("cross-fit analysis lock is missing or invalid")
    for relative, expected in lock["files"].items():
        if sha256(project_path(relative)) != expected:
            raise RuntimeError(f"cross-fit lock mismatch: {relative}")
    config = json.loads(ROUTER_CONFIG.read_text(encoding="utf-8"))
    if (
        config["outcome_exposure"]["new_single_scale_results_known_before_lock"]
        or config["outcome_exposure"]["fresh_confirmatory_claim_allowed"]
        or config["post_lock_changes_allowed"]
    ):
        raise RuntimeError("cross-fit outcome-exposure contract changed")
    return config


def write_json(path: Path, payload: Any) -> None:
    partial = path.with_name(f"{path.name}.partial")
    partial.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def main() -> None:
    if len(sys.argv) != 1:
        raise PermissionError("the scale-candidate builder accepts no overrides")
    router_config = audit_analysis_lock()
    lock, natural_config, index = frozen.audit_inputs()
    receipt_path = OUTPUT / "receipt.json"
    if receipt_path.exists():
        raise FileExistsError("single-scale candidate output is immutable")
    if OUTPUT.exists() and any(OUTPUT.rglob("*")):
        raise RuntimeError("partial single-scale output exists")
    predictions_dir = OUTPUT / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, checkpoint_hashes = frozen.load_models(natural_config, device)
    with np.load(
        PRECOMPUTE / "query_embeddings.npz", allow_pickle=False
    ) as query_record:
        label_ids = query_record["label_ids"].astype(int).tolist()
        queries = torch.from_numpy(
            query_record["ensemble_embeddings"].astype(np.float32)
        ).to(device)
    if label_ids != [int(value) for value in natural_config["benchmark"]["label_ids"]]:
        raise RuntimeError("query label order changed")
    scales = list(natural_config["embedding"]["scales"])
    scale_lookup = {
        float(scale["window_seconds"]): index for index, scale in enumerate(scales)
    }
    if set(GENERATED_SCALES) - set(scale_lookup):
        raise RuntimeError("locked single scales are absent from the precompute")

    methods = [
        single_scale_name(source, seconds)
        for source in ("clap", "qcr")
        for seconds in GENERATED_SCALES
    ]
    final_paths = {method: predictions_dir / f"{method}.jsonl.gz" for method in methods}
    partial_paths = {
        method: predictions_dir / f"{method}.jsonl.partial.gz" for method in methods
    }
    handles = {
        method: gzip.open(path, "wt", encoding="utf-8")
        for method, path in partial_paths.items()
    }
    prediction_counts = {method: 0 for method in methods}
    maximum_prior_disagreement = 0.0
    maximum_residual = 0.0
    started = time.perf_counter()
    try:
        for number, row in enumerate(index, start=1):
            archive_path = project_path(str(row["archive_path"]))
            if sha256(archive_path) != str(row["archive_sha256"]):
                raise RuntimeError(f"precompute archive mismatch: {row['video_id']}")
            with np.load(archive_path, allow_pickle=False) as record:
                video_id = str(record["video_id"].item())
                embeddings = record["audio_embeddings"].astype(np.float32)
                starts = record["start_sec"].astype(np.float64)
                ends = record["end_sec"].astype(np.float64)
                scale_ids = record["scale_id"].astype(int)
            for seconds in GENERATED_SCALES:
                scale_id = scale_lookup[seconds]
                mask = scale_ids == scale_id
                prior, qcr, disagreement, residual = frozen.score_scale(
                    embeddings[mask], queries, models, device
                )
                maximum_prior_disagreement = max(
                    maximum_prior_disagreement, disagreement
                )
                maximum_residual = max(maximum_residual, residual)
                local_scale_ids = np.full(int(mask.sum()), scale_id, dtype=np.int64)
                for source, scores in (("clap", prior), ("qcr", qcr)):
                    method = single_scale_name(source, seconds)
                    predictions = frozen.postprocess_video(
                        video_id=video_id,
                        starts=starts[mask],
                        ends=ends[mask],
                        scale_ids=local_scale_ids,
                        scores=scores,
                        allowed_scale_ids={scale_id},
                        label_ids=label_ids,
                        config=natural_config,
                    )
                    prediction_counts[method] += len(predictions)
                    for prediction in predictions:
                        handles[method].write(
                            json.dumps(prediction, sort_keys=True) + "\n"
                        )
            if number % 100 == 0 or number == len(index):
                print(
                    json.dumps(
                        {
                            "scored_videos": number,
                            "total": len(index),
                            "video_id": video_id,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        for handle in handles.values():
            handle.close()
    for method in methods:
        partial_paths[method].replace(final_paths[method])

    maximum_allowed_residual = float(
        json.loads(
            project_path(natural_config["ranker"]["config"]).read_text(encoding="utf-8")
        )["maximum_residual"]
    )
    if maximum_prior_disagreement > 1e-7:
        raise RuntimeError("frozen model priors disagree")
    if maximum_residual > maximum_allowed_residual + 1e-6:
        raise RuntimeError("frozen residual bound was exceeded")
    receipt = {
        "status": "locked_single_scale_candidates_complete",
        "claim_type": router_config["outcome_exposure"]["claim_type"],
        "methods": methods,
        "videos": len(index),
        "prediction_counts": prediction_counts,
        "prediction_sha256": {
            method: sha256(path) for method, path in final_paths.items()
        },
        "maximum_prior_disagreement": maximum_prior_disagreement,
        "maximum_absolute_residual": maximum_residual,
        "checkpoint_sha256": checkpoint_hashes,
        "crossfit_analysis_lock_sha256": sha256(LOCK),
        "router_config_sha256": sha256(ROUTER_CONFIG),
        "natural_freeze_sha256": sha256(NATURAL / "PERCEPTION_TEST_FREEZE.json"),
        "natural_config_sha256": sha256(
            NATURAL / "configs/perception_test_natural.json"
        ),
        "precompute_summary_sha256": sha256(PRECOMPUTE / "precompute_summary.json"),
        "device": str(device),
        "torch": torch.__version__,
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "videos": receipt["videos"],
                "prediction_counts": prediction_counts,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
