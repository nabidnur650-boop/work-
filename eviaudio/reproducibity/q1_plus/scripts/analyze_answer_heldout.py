#!/usr/bin/env python3
"""Audit the one held-out answer evaluation and apply its frozen gates."""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import run_answer_calibration as base
import run_answer_heldout as runner


PROJECT = Path(__file__).resolve().parents[2]
Q1 = PROJECT / "q1_plus"
CONFIG = Q1 / "configs/answer_heldout.json"
ANSWER_CONFIG = Q1 / "configs/answer_development.json"
AUTHORIZATION = Q1 / "ANSWER_HELDOUT_AUTHORIZATION.json"
CALIBRATION = Q1 / "results/development/answer_generation/calibration_selection_report.json"
MANIFEST = PROJECT / "journal_suite/data/manifests/val.jsonl"
RETRIEVAL_RAW = Q1 / "results/development/answer_retrieval/retrieval_scores.jsonl.gz"
RESULTS = Q1 / "results/development/answer_generation"
REPORT = RESULTS / "heldout_evaluation_report.json"
MODELS = ("qwen2_audio", "phi4_multimodal")


def sha256(path: Path) -> str:
    return base.sha256(path)


def source_values(
    rows: list[dict[str, Any]], system: str, metric: str
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["system"] == system:
            grouped[str(row["target_source_id"])].append(float(row[metric]))
    return {source: float(np.mean(values)) for source, values in grouped.items()}


def source_macro(rows: list[dict[str, Any]], system: str) -> dict[str, float | int]:
    exact = source_values(rows, system, "exact")
    f1 = source_values(rows, system, "token_f1")
    if set(exact) != set(f1):
        raise RuntimeError("held-out source metric alignment failed")
    return {
        "source_macro_exact": float(np.mean(list(exact.values()))),
        "source_macro_token_f1": float(np.mean(list(f1.values()))),
        "sources": len(exact),
        "examples": sum(row["system"] == system for row in rows),
    }


def combined_source_values(
    rows_by_model: dict[str, list[dict[str, Any]]], system: str, metric: str
) -> dict[str, float]:
    model_values = {
        model: source_values(rows, system, metric)
        for model, rows in rows_by_model.items()
    }
    sources = set(next(iter(model_values.values())))
    if any(set(values) != sources for values in model_values.values()):
        raise RuntimeError("backbone source sets differ")
    return {
        source: float(np.mean([model_values[model][source] for model in MODELS]))
        for source in sources
    }


def paired_interval(
    left: dict[str, float], right: dict[str, float], replicates: int, seed: int
) -> dict[str, Any]:
    sources = sorted(left)
    if set(sources) != set(right):
        raise RuntimeError("paired bootstrap source sets differ")
    differences = np.asarray([left[source] - right[source] for source in sources])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sources), size=(replicates, len(sources)))
    boot = differences[draws].mean(axis=1)
    return {
        "delta": float(differences.mean()),
        "bootstrap_95_interval": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "sources": len(sources),
        "replicates": replicates,
        "seed": seed,
    }


def breakdowns(rows: list[dict[str, Any]], system: str) -> dict[str, Any]:
    selected = [row for row in rows if row["system"] == system]
    chunk_values = np.asarray([row["n_chunks"] for row in selected], dtype=float)
    boundaries = np.quantile(chunk_values, [0.25, 0.5, 0.75])
    result: dict[str, Any] = {}
    definitions = {
        "difficulty": lambda row: str(row["difficulty"]),
        "number_of_sources": lambda row: str(row["n_sources"]),
        "target_position_bin": lambda row: str(row["target_position_bin"]),
        "chunk_count_quartile": lambda row: f"Q{1 + int(np.searchsorted(boundaries, row['n_chunks'], side='left'))}",
    }
    for name, key_function in definitions.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            grouped[key_function(row)].append(row)
        result[name] = {
            group: {
                "exact": float(np.mean([row["exact"] for row in items])),
                "token_f1": float(np.mean([row["token_f1"] for row in items])),
                "examples": len(items),
                "sources": len({row["target_source_id"] for row in items}),
            }
            for group, items in sorted(grouped.items())
        }
    return result


def validate_row(
    row: dict[str, Any],
    model: str,
    system: str,
    manifest: dict[str, Any],
    retrieval: dict[str, Any],
    expected: dict[str, Any] | None,
) -> None:
    identifier = str(manifest["example_id"])
    if (
        row["job_id"] != f"{model}|{system}|{identifier}"
        or row["model"] != model
        or row["system"] != system
        or row["example_id"] != identifier
        or row["target_source_id"] != manifest["target_source_id"]
        or row["question"] != manifest["question"]
        or row["reference"] != manifest["answer"]
        or row["difficulty"] != manifest["difficulty"]
        or int(row["n_sources"]) != int(manifest["n_sources"])
        or int(row["n_chunks"]) != int(manifest["n_chunks"])
        or row["target_position_bin"] != manifest["target_position_bin"]
        or row["audita_rows_accessed"] != 0
    ):
        raise RuntimeError("held-out row identity/provenance mismatch")
    response = str(row["response"])
    reference = str(row["reference"])
    if (
        row["normalized_response"] != base.normalize_answer(response)
        or row["normalized_reference"] != base.normalize_answer(reference)
        or float(row["exact"])
        != float(base.normalize_answer(response) == base.normalize_answer(reference))
        or not np.isclose(float(row["token_f1"]), base.token_f1(response, reference))
    ):
        raise RuntimeError("held-out answer metrics failed recomputation")
    if system == "text_only":
        if (
            row["selected_chunk_indices"]
            or row["selected_provenance"]
            or int(row["selected_chunk_count"]) != 0
            or int(row["selected_positive_chunks"]) != 0
            or int(row["audio_samples"]) != 0
            or float(row["audio_duration_sec"]) != 0.0
            or row["audio_float32_sha256"] is not None
        ):
            raise RuntimeError("text-only control contains audio")
    else:
        assert expected is not None
        indices = expected["indices"]
        evidence = np.asarray(retrieval["evidence_targets"], dtype=np.int64)
        if (
            row["selected_chunk_indices"] != indices
            or row["selected_provenance"] != expected["provenance"]
            or int(row["selected_chunk_count"]) != len(indices)
            or int(row["selected_positive_chunks"]) != int(evidence[indices].sum())
            or int(row["audio_samples"]) != expected["samples"]
            or not np.isclose(
                float(row["audio_duration_sec"]), expected["samples"] / base.SAMPLE_RATE
            )
            or row["audio_float32_sha256"] != expected["sha256"]
        ):
            raise RuntimeError("held-out waveform/provenance audit failed")
    if int(row["input_tokens"]) <= 0 or float(row["generation_seconds"]) < 0:
        raise RuntimeError("invalid held-out runtime telemetry")
    provenance_values = {
        "question": row["question"],
        "reference": row["reference"],
        "selected_provenance": row["selected_provenance"],
    }
    if "audita" in json.dumps(provenance_values).lower():
        raise PermissionError("AUDITA reference found in held-out row")


def main() -> None:
    if REPORT.exists():
        raise FileExistsError("held-out answer report is immutable")
    authorization, calibration = runner.audit_authorization()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    answer_config = json.loads(ANSWER_CONFIG.read_text(encoding="utf-8"))
    systems = list(config["systems"])
    manifests_list = base.read_jsonl(MANIFEST)
    manifests = {str(row["example_id"]): row for row in manifests_list}
    with gzip.open(RETRIEVAL_RAW, "rt", encoding="utf-8") as handle:
        retrieval_rows = [json.loads(line) for line in handle if line.strip()]
    retrieval_by_id = {
        str(row["example_id"]): row
        for row in retrieval_rows
        if row["split"] == "evaluation"
    }
    if len(retrieval_by_id) != int(config["expected_examples"]):
        raise RuntimeError("held-out retrieval coverage changed")
    retriever = str(authorization["selected_retriever"])
    k = int(authorization["selected_k"])
    store = base.AudioStore(base.AUDIO_MANIFEST)
    expected_audio: dict[tuple[str, str], dict[str, Any]] = {}
    for identifier, retrieval in sorted(retrieval_by_id.items()):
        manifest = manifests[identifier]
        for system in systems:
            if system == "text_only":
                continue
            indices = runner.system_indices(retrieval, system, k, retriever)
            audio, provenance = runner.audio_from_indices(indices, manifest, store)
            if system == "selected_retrieval_silenced":
                audio = np.zeros_like(audio)
            expected_audio[(identifier, system)] = {
                "indices": indices,
                "provenance": provenance,
                "samples": len(audio),
                "sha256": __import__("hashlib").sha256(audio.tobytes()).hexdigest(),
            }

    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    artifacts = []
    expected_count = len(retrieval_by_id) * len(systems)
    for model in MODELS:
        raw_path = RESULTS / f"heldout__{model}.jsonl"
        receipt_path = RESULTS / f"heldout__{model}.receipt.json"
        rows = base.read_jsonl(raw_path)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        jobs = {str(row["job_id"]) for row in rows}
        expected_jobs = {
            f"{model}|{system}|{identifier}"
            for system in systems
            for identifier in retrieval_by_id
        }
        if len(rows) != expected_count or jobs != expected_jobs:
            raise RuntimeError(f"held-out job coverage mismatch: {model}")
        for row in rows:
            identifier = str(row["example_id"])
            system = str(row["system"])
            validate_row(
                row,
                model,
                system,
                manifests[identifier],
                retrieval_by_id[identifier],
                expected_audio.get((identifier, system)),
            )
        if (
            receipt["status"] != "heldout_answer_backbone_complete"
            or receipt["model"] != model
            or receipt["repository"] != answer_config["backbones"][model]["repository"]
            or receipt["revision"] != answer_config["backbones"][model]["revision"]
            or receipt["authorization_sha256"] != sha256(AUTHORIZATION)
            or receipt["heldout_config_sha256"] != sha256(CONFIG)
            or receipt["calibration_report_sha256"] != sha256(CALIBRATION)
            or receipt["retrieval_raw_sha256"] != sha256(RETRIEVAL_RAW)
            or receipt["raw_sha256"] != sha256(raw_path)
            or int(receipt["rows"]) != expected_count
            or int(receipt["examples"]) != len(retrieval_by_id)
            or int(receipt["sources"]) != int(config["expected_sources"])
            or receipt["systems"] != systems
            or receipt["selected_prompt_name"] != calibration["selected_prompt_name"]
            or int(receipt["selected_k"]) != k
            or receipt["selected_retriever"] != retriever
            or receipt["audita_rows_accessed"] != 0
        ):
            raise RuntimeError(f"held-out receipt mismatch: {model}")
        rows_by_model[model] = rows
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

    metrics = {
        model: {system: source_macro(rows_by_model[model], system) for system in systems}
        for model in MODELS
    }
    combined = {
        system: {
            "two_backbone_mean_source_macro_exact": float(
                np.mean([metrics[model][system]["source_macro_exact"] for model in MODELS])
            ),
            "two_backbone_mean_source_macro_token_f1": float(
                np.mean(
                    [metrics[model][system]["source_macro_token_f1"] for model in MODELS]
                )
            ),
        }
        for system in systems
    }
    eligible = list(config["eligible_nonoracle_retrieval_baselines"])
    strongest = sorted(
        eligible,
        key=lambda system: (
            -combined[system]["two_backbone_mean_source_macro_exact"],
            -combined[system]["two_backbone_mean_source_macro_token_f1"],
            system,
        ),
    )[0]
    bootstrap_config = config["bootstrap"]
    comparisons: dict[str, Any] = {}
    learned_combined = combined_source_values(
        rows_by_model, "selected_learned_retrieval", "exact"
    )
    for system in systems:
        if system == "selected_learned_retrieval":
            continue
        comparisons[system] = {
            "combined_exact": paired_interval(
                learned_combined,
                combined_source_values(rows_by_model, system, "exact"),
                int(bootstrap_config["replicates"]),
                int(bootstrap_config["seed"]),
            ),
            "per_backbone_exact": {
                model: paired_interval(
                    source_values(rows_by_model[model], "selected_learned_retrieval", "exact"),
                    source_values(rows_by_model[model], system, "exact"),
                    int(bootstrap_config["replicates"]),
                    int(bootstrap_config["seed"]),
                )
                for model in MODELS
            },
        }

    response_differences = {}
    for model in MODELS:
        by_key = {
            (row["system"], row["example_id"]): row for row in rows_by_model[model]
        }
        learned = [
            by_key[("selected_learned_retrieval", identifier)]
            for identifier in sorted(retrieval_by_id)
        ]
        response_differences[model] = {
            control: float(
                np.mean(
                    [
                        row["normalized_response"]
                        != by_key[(control, row["example_id"])]["normalized_response"]
                        for row in learned
                    ]
                )
            )
            for control in ("selected_retrieval_silenced", "text_only")
        }

    gates_config = config["promotion_gates"]
    learned_exact = combined["selected_learned_retrieval"][
        "two_backbone_mean_source_macro_exact"
    ]
    strongest_exact = combined[strongest]["two_backbone_mean_source_macro_exact"]
    random_exact = combined["deterministic_random_retrieval"][
        "two_backbone_mean_source_macro_exact"
    ]
    gates = {
        "gain_at_least_0_02_over_strongest_baseline": learned_exact - strongest_exact
        >= float(gates_config["minimum_exact_gain_over_strongest_baseline"]),
        "gain_at_least_0_05_over_random": learned_exact - random_exact
        >= float(gates_config["minimum_exact_gain_over_random"]),
        "no_backbone_loses_more_than_0_02_to_strongest": all(
            float(metrics[model]["selected_learned_retrieval"]["source_macro_exact"])
            - float(metrics[model][strongest]["source_macro_exact"])
            >= -float(gates_config["maximum_single_backbone_loss_to_strongest_baseline"])
            for model in MODELS
        ),
        "responses_differ_from_silence": all(
            response_differences[model]["selected_retrieval_silenced"]
            >= float(
                gates_config[
                    "minimum_normalized_response_difference_fraction_from_silence_per_backbone"
                ]
            )
            for model in MODELS
        ),
        "responses_differ_from_text_only": all(
            response_differences[model]["text_only"]
            >= float(
                gates_config[
                    "minimum_normalized_response_difference_fraction_from_text_only_per_backbone"
                ]
            )
            for model in MODELS
        ),
        "learned_exact_no_worse_than_silence": all(
            float(metrics[model]["selected_learned_retrieval"]["source_macro_exact"])
            >= float(metrics[model]["selected_retrieval_silenced"]["source_macro_exact"])
            for model in MODELS
        ),
        "all_responses_nonempty": all(
            all(bool(row["response"].strip()) for row in rows) for rows in rows_by_model.values()
        ),
        "integrity": True,
    }
    passed = all(gates.values())
    payload = {
        "status": (
            "heldout_answer_gate_pass_audita_pipeline_lock_eligible"
            if passed
            else "heldout_answer_gate_fail_audita_sealed"
        ),
        "selected_retriever": retriever,
        "selected_prompt_name": calibration["selected_prompt_name"],
        "selected_k": k,
        "strongest_eligible_nonoracle_baseline": strongest,
        "metrics": metrics,
        "combined_metrics": combined,
        "comparisons": comparisons,
        "response_difference_fractions": response_differences,
        "breakdowns": {
            model: {
                system: breakdowns(rows_by_model[model], system)
                for system in systems
            }
            for model in MODELS
        },
        "promotion_gates": gates,
        "all_promotion_gates_pass": passed,
        "artifacts": artifacts,
        "integrity": {
            "sources": int(config["expected_sources"]),
            "examples": len(retrieval_by_id),
            "systems": len(systems),
            "jobs_per_model": expected_count,
            "duplicate_jobs": 0,
            "waveform_hash_failures": 0,
            "metric_recomputation_failures": 0,
            "duration_bound_failures": 0,
            "split_overlap": 0,
            "audita_rows_accessed": 0,
            "all_checks_pass": True,
        },
        "input_hashes": {
            "authorization": sha256(AUTHORIZATION),
            "config": sha256(CONFIG),
            "calibration": sha256(CALIBRATION),
            "manifest": sha256(MANIFEST),
            "retrieval_raw": sha256(RETRIEVAL_RAW),
        },
        "audita_status": "pipeline_lock_eligible_but_still_sealed" if passed else "sealed",
    }
    REPORT.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "strongest_baseline": strongest,
                "combined_metrics": combined,
                "promotion_gates": gates,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
