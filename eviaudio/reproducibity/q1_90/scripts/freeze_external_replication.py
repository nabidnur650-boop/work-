#!/usr/bin/env python3
"""Authorize one frozen UrbanSound8K external replication after recipe sealing."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[2]
TRACK = PROJECT / "q1_90"
CONFIG = TRACK / "configs/external_replication.json"
RECIPES = TRACK / "data/external_replication/external_recipes.sealed.jsonl"
AUDIT = TRACK / "data/external_replication/recipe_audit.json"
AUTHORIZATION = TRACK / "EXTERNAL_REPLICATION_AUTHORIZATION.json"
DEVELOPMENT_REPORT = (
    PROJECT / "q1_plus/results/development/event_ranker/five_seed_validation_report.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(PROJECT).as_posix()


def load_recipes() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in RECIPES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def source_audit(recipes: list[dict[str, Any]]) -> dict[str, Any]:
    sources: dict[str, str] = {}
    targets: list[str] = []
    distractors: list[str] = []
    for recipe in recipes:
        targets.append(str(recipe["target_event"]["source_file_id"]))
        distractors.extend(
            str(item["source_file_id"]) for item in recipe["distractor_events"]
        )
        for item in [
            recipe["target_event"],
            *recipe["distractor_events"],
            *recipe["background_files"],
        ]:
            path_text = str(item["path"])
            expected = str(item["sha256"])
            if path_text in sources and sources[path_text] != expected:
                raise RuntimeError(f"source checksum disagreement: {path_text}")
            path = PROJECT / path_text
            if sha256(path) != expected:
                raise RuntimeError(f"source changed before authorization: {path_text}")
            sources[path_text] = expected
    if len(set(targets)) != len(targets):
        raise RuntimeError("external target source was reused")
    if set(targets) & set(distractors):
        raise RuntimeError("external target source appears as a distractor")
    manifest = "".join(
        f"{path}\0{checksum}\n" for path, checksum in sorted(sources.items())
    )
    recipe_ids = "".join(f"{recipe['recipe_id']}\n" for recipe in recipes)
    return {
        "unique_source_files": len(sources),
        "source_manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
        "recipe_ids_sha256": hashlib.sha256(recipe_ids.encode()).hexdigest(),
        "target_source_reuse": 0,
        "target_distractor_source_overlap": 0,
    }


def main() -> None:
    if AUTHORIZATION.exists():
        raise FileExistsError("external replication authorization is immutable")
    result_root = TRACK / "results/external_replication"
    if result_root.exists() and any(result_root.iterdir()):
        raise PermissionError("external model outputs exist before authorization")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    recipes = load_recipes()
    if (
        config["status"] != "frozen_after_model_blind_source_count_amendment"
        or audit["status"] != "external_recipes_frozen_before_any_model_scoring"
        or int(audit["model_scores_computed"]) != 0
        or audit["recipes_sha256"] != sha256(RECIPES)
        or len(recipes) != int(audit["recipes"])
    ):
        raise PermissionError("model-blind recipe freeze is invalid")
    identifiers = [str(recipe["recipe_id"]) for recipe in recipes]
    if len(identifiers) != len(set(identifiers)) or not identifiers:
        raise RuntimeError("external recipe identities are invalid")
    if {str(recipe["panel"]) for recipe in recipes} != {"confirmatory"}:
        raise RuntimeError("external recipe panel is not confirmatory")
    if Counter(recipe["target_event"]["class_label"] for recipe in recipes) != Counter(
        audit["classes"]
    ):
        raise RuntimeError("external class inventory changed")
    source_integrity = source_audit(recipes)
    if source_integrity["source_manifest_sha256"] != audit["source_manifest_sha256"]:
        raise RuntimeError("external source manifest changed")

    ranker_config = PROJECT / config["ranker"]["config"]
    development = json.loads(DEVELOPMENT_REPORT.read_text(encoding="utf-8"))
    if (
        development["status"]
        != "development_pass_exact_onset_confirmatory_authorized_once"
        or not development["all_promotion_gates_pass"]
        or development["seeds"] != config["ranker"]["seeds"]
    ):
        raise PermissionError("frozen ranker development evidence is invalid")
    checkpoints = [
        PROJECT
        / "q1_plus/results/development/event_ranker/checkpoints"
        / f"{config['ranker']['run_tag']}__prior_residual__seed_{seed}.pt"
        for seed in config["ranker"]["seeds"]
    ]
    report_by_seed = {int(item["seed"]): item for item in development["per_seed"]}
    for seed, checkpoint in zip(config["ranker"]["seeds"], checkpoints, strict=True):
        if sha256(checkpoint) != report_by_seed[int(seed)]["checkpoint_sha256"]:
            raise RuntimeError(f"ranker checkpoint changed: seed {seed}")

    files = [
        CONFIG,
        PROJECT / config["model_blind_amendment"],
        RECIPES,
        AUDIT,
        TRACK / "EXTERNAL_REPLICATION_PROTOCOL.md",
        TRACK / "DATA_RIGHTS_AND_DISTRIBUTION.md",
        TRACK / "scripts/freeze_external_replication.py",
        TRACK / "scripts/run_external_precompute.py",
        TRACK / "scripts/evaluate_external_replication.py",
        ranker_config,
        DEVELOPMENT_REPORT,
        PROJECT / "q1_plus/EVENT_RANKER_LOCK.json",
        PROJECT / "q1_plus/scripts/precompute_event_needle_clap.py",
        PROJECT / "q1_plus/scripts/evaluate_event_ranker_confirmatory.py",
        PROJECT / "q1_plus/scripts/analyze_event_ranker.py",
        PROJECT / "src/eviaudio_mt/audio_encoder.py",
        PROJECT / "src/eviaudio_mt/event_needle.py",
        PROJECT / "src/eviaudio_mt/event_data.py",
        PROJECT / "src/eviaudio_mt/qcr.py",
        TRACK / "tests/test_external_replication.py",
        *checkpoints,
    ]
    authorization = {
        "status": "one_shot_external_precompute_and_evaluation_authorized",
        "study": config["study"],
        "model_blind_amendment": config["model_blind_amendment"],
        "config": relative(CONFIG),
        "recipes": relative(RECIPES),
        "recipe_audit": relative(AUDIT),
        "examples": len(recipes),
        "ranker_config": relative(ranker_config),
        "development_report": relative(DEVELOPMENT_REPORT),
        "seeds": config["ranker"]["seeds"],
        "precompute_output_dir": "q1_90/results/external_replication/precompute",
        "evaluation_output_dir": "q1_90/results/external_replication/evaluation",
        "model_scores_before_authorization": 0,
        "external_evaluations_allowed": 1,
        "embedding": config["embedding"],
        "bootstrap": config["bootstrap"],
        "gates": config["gates"],
        "source_integrity": source_integrity,
        "files": {relative(path): sha256(path) for path in files},
    }
    AUTHORIZATION.write_text(
        json.dumps(authorization, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": authorization["status"],
                "examples": len(recipes),
                "unique_source_files": source_integrity["unique_source_files"],
                "authorization_sha256": sha256(AUTHORIZATION),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
