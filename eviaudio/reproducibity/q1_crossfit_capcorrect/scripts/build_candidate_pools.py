#!/usr/bin/env python3
"""Build per-class candidate pools and audit reconstruction of frozen caps."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[2]
TRACK = PROJECT / "q1_crossfit_capcorrect"
LOCK = TRACK / "CAPCORRECT_ANALYSIS_LOCK.json"
CONFIG_PATH = TRACK / "configs/cap_corrected_crossfit.json"
NATURAL = PROJECT / "q1_top_tier"
PRECOMPUTE = NATURAL / "results/perception_test/precompute"
OLD_UPGRADE = PROJECT / "q1_upgrade"
OUTPUT = TRACK / "results/candidate_pools"
RECEIPT = OUTPUT / "receipt.json"
sys.path.insert(0, str(TRACK / "src"))
sys.path.insert(0, str(NATURAL / "scripts"))
sys.path.insert(0, str(NATURAL / "src"))
sys.path.insert(0, str(PROJECT / "src"))

import evaluate_perception_test as frozen  # noqa: E402
from capcorrect_utils import (  # noqa: E402
    candidate_name,
    cap_video_rows,
    sha256,
)


SCALES = (0.5, 1.0, 2.0, 4.0)


def project_path(relative: str) -> Path:
    path = (PROJECT / relative).resolve()
    if not path.is_relative_to(PROJECT):
        raise RuntimeError(f"path leaves project: {relative}")
    return path


def write_json(path: Path, payload: Any) -> None:
    partial = path.with_name(f"{path.name}.partial")
    partial.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def audit_lock() -> dict[str, Any]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["status"] != (
        "cap_corrected_crossfit_frozen_after_capped_candidate_exposure_"
        "before_uncapped_pool_scoring"
    ):
        raise PermissionError("cap-correct lock is invalid")
    for relative, expected in lock["files"].items():
        if sha256(project_path(relative)) != expected:
            raise RuntimeError(f"cap-correct lock mismatch: {relative}")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if (
        config["outcome_exposure"]["router_result_known_before_lock"]
        or config["outcome_exposure"][
            "uncapped_candidate_pool_results_known_before_lock"
        ]
        or config["post_lock_changes_allowed"]
    ):
        raise RuntimeError("cap-correct outcome contract changed")
    return config


def postprocess_class_pool(
    *,
    video_id: str,
    starts: np.ndarray,
    ends: np.ndarray,
    scale_ids: np.ndarray,
    scores: np.ndarray,
    allowed_scale_ids: set[int],
    label_ids: list[int],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    settings = config["postprocessing"]
    selected_scale = np.asarray(
        [int(value) in allowed_scale_ids for value in scale_ids], dtype=bool
    )
    segments = np.column_stack((starts[selected_scale], ends[selected_scale]))
    output: list[dict[str, Any]] = []
    for label_index, label_id in enumerate(label_ids):
        confidence = frozen.sigmoid_scores(
            scores[label_index, selected_scale],
            float(settings["score_temperature"]),
        )
        _, selected_segments, selected_scores = frozen.gaussian_soft_nms(
            segments,
            confidence,
            sigma=float(settings["soft_nms_sigma"]),
            minimum_score=float(settings["minimum_score"]),
            maximum_output=int(settings["maximum_segments_per_class_video"]),
        )
        for segment, score in zip(
            selected_segments, selected_scores, strict=True
        ):
            output.append(
                {
                    "video_id": video_id,
                    "label_id": label_id,
                    "start_sec": float(segment[0]),
                    "end_sec": float(segment[1]),
                    "score": float(score),
                }
            )
    return output


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def canonical_digest(rows: Iterable[dict[str, Any]]) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for row in rows:
        payload = json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
        digest.update(payload.encode("utf-8"))
        count += 1
    return digest.hexdigest(), count


def recapped_rows(path: Path, maximum: int) -> Iterable[dict[str, Any]]:
    current_video: str | None = None
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        video_id = str(row["video_id"])
        if current_video is not None and video_id != current_video:
            yield from cap_video_rows(rows, maximum)
            rows = []
        current_video = video_id
        rows.append(row)
    if rows:
        yield from cap_video_rows(rows, maximum)


def capped_candidate_paths() -> dict[str, Path]:
    natural = NATURAL / "results/perception_test/evaluation/predictions"
    generated = OLD_UPGRADE / "results/scale_candidates/predictions"
    paths: dict[str, Path] = {}
    for source in ("clap", "qcr"):
        for seconds in SCALES:
            name = candidate_name(source, seconds)
            root = natural if seconds == 4.0 else generated
            paths[name] = root / f"{name}.jsonl.gz"
        name = candidate_name(source, None)
        paths[name] = natural / f"{name}.jsonl.gz"
    return dict(sorted(paths.items()))


def main() -> None:
    if len(sys.argv) != 1:
        raise PermissionError("candidate-pool builder accepts no overrides")
    track_config = audit_lock()
    _, natural_config, index = frozen.audit_inputs()
    if RECEIPT.exists():
        raise FileExistsError("candidate-pool receipt is immutable")
    if OUTPUT.exists() and any(OUTPUT.rglob("*")):
        raise RuntimeError("partial candidate-pool output exists")
    pools_dir = OUTPUT / "pools"
    pools_dir.mkdir(parents=True, exist_ok=True)

    class_cap = int(
        track_config["caps"]["maximum_segments_per_class_video"]
    )
    video_cap = int(
        track_config["caps"]["maximum_segments_per_video_after_routing"]
    )
    natural_settings = natural_config["postprocessing"]
    if (
        class_cap
        != int(natural_settings["maximum_segments_per_class_video"])
        or video_cap != int(natural_settings["maximum_segments_per_video"])
    ):
        raise RuntimeError("corrected caps do not match the frozen natural panel")

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
    if label_ids != [
        int(value) for value in natural_config["benchmark"]["label_ids"]
    ]:
        raise RuntimeError("query label order changed")

    scales = list(natural_config["embedding"]["scales"])
    scale_lookup = {
        float(scale["window_seconds"]): index
        for index, scale in enumerate(scales)
    }
    if set(SCALES) != set(scale_lookup):
        raise RuntimeError("candidate scales do not match frozen embeddings")
    candidates = [
        candidate_name(source, seconds)
        for source in ("clap", "qcr")
        for seconds in (*SCALES, None)
    ]
    final_paths = {
        name: pools_dir / f"{name}.jsonl.gz" for name in candidates
    }
    partial_paths = {
        name: pools_dir / f"{name}.jsonl.partial.gz" for name in candidates
    }
    handles = {
        name: gzip.open(path, "wt", encoding="utf-8")
        for name, path in partial_paths.items()
    }
    pool_counts = {name: 0 for name in candidates}
    maximum_prior_disagreement = 0.0
    maximum_residual = 0.0
    started = time.perf_counter()
    try:
        for number, row in enumerate(index, start=1):
            archive_path = project_path(str(row["archive_path"]))
            if sha256(archive_path) != str(row["archive_sha256"]):
                raise RuntimeError(
                    f"precompute archive mismatch: {row['video_id']}"
                )
            with np.load(archive_path, allow_pickle=False) as record:
                video_id = str(record["video_id"].item())
                embeddings = record["audio_embeddings"].astype(np.float32)
                starts = record["start_sec"].astype(np.float64)
                ends = record["end_sec"].astype(np.float64)
                scale_ids = record["scale_id"].astype(int)
            prior_scores = np.empty(
                (len(label_ids), len(embeddings)), dtype=np.float32
            )
            qcr_scores = np.empty_like(prior_scores)
            for scale_id in range(len(scales)):
                mask = scale_ids == scale_id
                prior, qcr, disagreement, residual = frozen.score_scale(
                    embeddings[mask], queries, models, device
                )
                prior_scores[:, mask] = prior
                qcr_scores[:, mask] = qcr
                maximum_prior_disagreement = max(
                    maximum_prior_disagreement, disagreement
                )
                maximum_residual = max(maximum_residual, residual)
            for source, values in (
                ("clap", prior_scores),
                ("qcr", qcr_scores),
            ):
                for seconds in (*SCALES, None):
                    name = candidate_name(source, seconds)
                    allowed = (
                        set(range(len(scales)))
                        if seconds is None
                        else {scale_lookup[float(seconds)]}
                    )
                    predictions = postprocess_class_pool(
                        video_id=video_id,
                        starts=starts,
                        ends=ends,
                        scale_ids=scale_ids,
                        scores=values,
                        allowed_scale_ids=allowed,
                        label_ids=label_ids,
                        config=natural_config,
                    )
                    pool_counts[name] += len(predictions)
                    for prediction in predictions:
                        handles[name].write(
                            json.dumps(
                                prediction, sort_keys=True, allow_nan=False
                            )
                            + "\n"
                        )
            if number % 100 == 0 or number == len(index):
                print(
                    json.dumps(
                        {
                            "pooled_videos": number,
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
    for name in candidates:
        partial_paths[name].replace(final_paths[name])

    parity: dict[str, Any] = {}
    capped_paths = capped_candidate_paths()
    for name in sorted(candidates):
        source_digest, source_count = canonical_digest(
            iter_jsonl(capped_paths[name])
        )
        reconstructed_digest, reconstructed_count = canonical_digest(
            recapped_rows(final_paths[name], video_cap)
        )
        parity[name] = {
            "source_capped_path": str(capped_paths[name].relative_to(PROJECT)),
            "source_capped_sha256": sha256(capped_paths[name]),
            "source_canonical_sha256": source_digest,
            "source_rows": source_count,
            "reconstructed_canonical_sha256": reconstructed_digest,
            "reconstructed_rows": reconstructed_count,
            "exact_canonical_match": (
                source_count == reconstructed_count
                and source_digest == reconstructed_digest
            ),
        }
        if not parity[name]["exact_canonical_match"]:
            raise RuntimeError(f"candidate cap reconstruction failed: {name}")

    maximum_allowed_residual = float(
        json.loads(
            project_path(natural_config["ranker"]["config"]).read_text(
                encoding="utf-8"
            )
        )["maximum_residual"]
    )
    if maximum_prior_disagreement > 1e-7:
        raise RuntimeError("frozen model priors disagree")
    if maximum_residual > maximum_allowed_residual + 1e-6:
        raise RuntimeError("frozen residual bound was exceeded")
    receipt = {
        "status": "cap_correct_candidate_pools_complete",
        "claim_type": track_config["outcome_exposure"]["claim_type"],
        "videos": len(index),
        "classes": len(label_ids),
        "candidates": len(candidates),
        "class_cap": class_cap,
        "post_route_video_cap": video_cap,
        "pool_counts": pool_counts,
        "pool_sha256": {
            name: sha256(path) for name, path in sorted(final_paths.items())
        },
        "cap_reconstruction": parity,
        "cap_reconstruction_exact_matches": sum(
            bool(item["exact_canonical_match"]) for item in parity.values()
        ),
        "maximum_prior_disagreement": maximum_prior_disagreement,
        "maximum_absolute_residual": maximum_residual,
        "checkpoint_sha256": checkpoint_hashes,
        "capcorrect_analysis_lock_sha256": sha256(LOCK),
        "config_sha256": sha256(CONFIG_PATH),
        "natural_freeze_sha256": sha256(
            NATURAL / "PERCEPTION_TEST_FREEZE.json"
        ),
        "precompute_summary_sha256": sha256(
            PRECOMPUTE / "precompute_summary.json"
        ),
        "device": str(device),
        "torch": torch.__version__,
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(RECEIPT, receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "videos": receipt["videos"],
                "pool_counts": pool_counts,
                "cap_reconstruction_exact_matches": receipt[
                    "cap_reconstruction_exact_matches"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
