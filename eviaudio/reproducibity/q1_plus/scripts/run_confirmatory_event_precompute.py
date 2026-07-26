#!/usr/bin/env python3
"""Authorization-locked wrapper for the one exact-onset confirmatory build."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import precompute_event_needle_clap as precompute


PROJECT = Path(__file__).resolve().parents[2]
Q1 = PROJECT / "q1_plus"
AUTHORIZATION = Q1 / "EVENT_CONFIRMATORY_AUTHORIZATION.json"
EXPECTED_RECIPES = Q1 / "data/event_needle/confirmatory_recipes.sealed.jsonl"
EXPECTED_OUTPUT = Q1 / "results/confirmatory/event_needle_confirmatory"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def audit_authorization() -> dict[str, Any]:
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    if authorization["status"] != "one_shot_exact_onset_confirmatory_authorized":
        raise PermissionError("confirmatory event evaluation is not authorized")
    for relative, expected in authorization["files"].items():
        path = (PROJECT / relative).resolve()
        if not path.is_relative_to(PROJECT) or sha256(path) != expected:
            raise RuntimeError(f"confirmatory authorization mismatch: {relative}")
    if (PROJECT / authorization["confirmatory_recipes"]).resolve() != EXPECTED_RECIPES:
        raise RuntimeError("authorization points at the wrong recipe panel")
    if (PROJECT / authorization["precompute_output_dir"]).resolve() != EXPECTED_OUTPUT:
        raise RuntimeError("authorization points at the wrong precompute output")
    return authorization


def load_and_verify_recipes(
    recipes_path: Path, expected_examples: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    recipes = [
        json.loads(line)
        for line in recipes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(recipes) != expected_examples:
        raise RuntimeError("unexpected confirmatory recipe count")
    recipe_ids = [str(recipe["recipe_id"]) for recipe in recipes]
    if len(set(recipe_ids)) != len(recipe_ids):
        raise RuntimeError("duplicate confirmatory recipe id")
    if {str(recipe["panel"]) for recipe in recipes} != {"confirmatory"}:
        raise RuntimeError("sealed recipe file contains another panel")

    development_ids: set[str] = set()
    for name in ("development_recipes.jsonl", "validation_recipes.jsonl"):
        path = recipes_path.parent / name
        development_ids.update(
            str(json.loads(line)["recipe_id"])
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    overlap = set(recipe_ids) & development_ids
    if overlap:
        raise RuntimeError("confirmatory recipe identity overlaps development")

    verified: dict[Path, str] = {}
    declared_manifest: list[tuple[str, str]] = []
    for recipe in recipes:
        sources = [
            recipe["target_event"],
            *recipe["distractor_events"],
            *recipe["background_files"],
        ]
        for source in sources:
            relative = Path(str(source["path"]))
            path = (PROJECT / relative).resolve()
            if not path.is_relative_to(PROJECT) or not path.is_file():
                raise RuntimeError(f"invalid recipe source: {relative}")
            expected = str(source["sha256"])
            if path not in verified:
                observed = sha256(path)
                if observed != expected:
                    raise RuntimeError(f"recipe source checksum mismatch: {relative}")
                verified[path] = observed
                declared_manifest.append((relative.as_posix(), expected))
            elif verified[path] != expected:
                raise RuntimeError(f"inconsistent source checksum: {relative}")

    manifest_text = "".join(
        f"{relative}\0{checksum}\n"
        for relative, checksum in sorted(declared_manifest)
    )
    return recipes, {
        "unique_source_files_verified": len(verified),
        "source_manifest_sha256": hashlib.sha256(manifest_text.encode()).hexdigest(),
        "recipe_ids_sha256": hashlib.sha256(
            "\n".join(sorted(recipe_ids)).encode()
        ).hexdigest(),
        "development_recipe_id_overlap": 0,
    }


def verify_completed_build(
    authorization: dict[str, Any], source_audit: dict[str, Any]
) -> dict[str, Any]:
    summary_path = EXPECTED_OUTPUT / "clap_prior_summary.json"
    index_path = EXPECTED_OUTPUT / "index.jsonl"
    raw_path = EXPECTED_OUTPUT / "raw_clap_prior.jsonl.gz"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "status": "confirmatory_embedding_complete",
        "panels": ["confirmatory"],
        "examples": authorization["confirmatory_examples"],
        "model_id": authorization["clap_model_id"],
        "model_revision": authorization["clap_model_revision"],
        "sample_rate": authorization["sample_rate"],
        "chunk_seconds": authorization["chunk_seconds"],
        "hop_seconds": authorization["hop_seconds"],
        "inference_batch_size": authorization["inference_batch_size"],
        "recipes_sha256": sha256(EXPECTED_RECIPES),
        "index_sha256": sha256(index_path),
        "raw_sha256": sha256(raw_path),
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise RuntimeError(f"confirmatory precompute summary mismatch: {key}")
    receipt = {
        "status": "one_shot_confirmatory_precompute_complete",
        "authorization_sha256": sha256(AUTHORIZATION),
        "recipes_sha256": sha256(EXPECTED_RECIPES),
        "summary_sha256": sha256(summary_path),
        "index_sha256": sha256(index_path),
        "raw_sha256": sha256(raw_path),
        "examples": authorization["confirmatory_examples"],
        **source_audit,
    }
    receipt_path = EXPECTED_OUTPUT / "confirmatory_precompute_receipt.json"
    if receipt_path.exists():
        raise FileExistsError("confirmatory precompute receipt already exists")
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    if len(sys.argv) != 1:
        raise PermissionError("the confirmatory wrapper accepts no command-line overrides")
    authorization = audit_authorization()
    _, source_audit = load_and_verify_recipes(
        EXPECTED_RECIPES, int(authorization["confirmatory_examples"])
    )
    print(json.dumps({"confirmatory_preflight": source_audit}, sort_keys=True), flush=True)

    sys.argv = [
        str(Path(__file__).resolve()),
        "--recipes",
        str(EXPECTED_RECIPES),
        "--output-dir",
        str(EXPECTED_OUTPUT),
        "--allow-confirmatory",
        "--sample-rate",
        str(authorization["sample_rate"]),
        "--chunk-seconds",
        str(authorization["chunk_seconds"]),
        "--hop-seconds",
        str(authorization["hop_seconds"]),
        "--inference-batch-size",
        str(authorization["inference_batch_size"]),
    ]
    precompute.main()
    receipt = verify_completed_build(authorization, source_audit)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
