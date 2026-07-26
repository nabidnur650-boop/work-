#!/usr/bin/env python3
"""Build label-sealed, revision-pinned AUDITA confirmatory manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, list_repo_files


PROJECT = Path(__file__).resolve().parents[2]
REPO_ID = "TasnimKabir12/audita-audio"
REVISION = "1591795b571ad13e84c4d447f3186d69f39c4ffa"
METADATA_FILE = "all_combined_9690_unsplit_full_audio_data.json"
METADATA_SHA256 = "eb03f00e5ff349bb91da6370849c14e8566d1d1e17c81e73f39340b6910d13f6"
MAJOR_CATEGORIES = {
    "Character/Person",
    "Cultural/Geographic",
    "Media Content",
    "Music Identification",
    "Musical Elements",
    "Sound Identification",
}
EXCLUSIONS = {
    0: "ground_truth displayed without question during metadata-contract audit",
    3230: "original_ground_truth displayed without question during metadata-contract audit",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sealed_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return sha256_bytes(payload)


def resolve_audio_path(record: dict[str, Any], repository_files: set[str]) -> str:
    old_path = str(record["file_name"])
    basename = os.path.basename(old_path)
    if record.get("dataset") is not None:
        candidate = f"External Sources/{basename}"
    else:
        root = old_path.split("/")[4]
        if root == "quizmasters":
            candidate = f"Our sources/Trivia(quizmasters)/{basename}"
        elif root == "audio-packets":
            suffix = old_path.split("/audio-packets/", 1)[1]
            candidate = f"Our sources/Quizbowl/audio-packets/{suffix}"
        elif root == "unsplit_Pavements_I_II_III":
            candidate = f"Our sources/Quizbowl/Pavements/{basename}"
        else:
            raise ValueError(f"unknown human audio root: {root}")
    if candidate not in repository_files:
        raise FileNotFoundError(candidate)
    return candidate


def category_of(record: dict[str, Any]) -> str:
    return str(record.get("main_category", record.get("Categories", "")))


def sealed_row(
    index: int,
    record: dict[str, Any],
    repository_files: set[str],
    panel: str,
) -> dict[str, Any]:
    ground_truth = record.get("ground_truth")
    return {
        "row_index": index,
        "panel": panel,
        "audio_repo_id": REPO_ID,
        "audio_revision": REVISION,
        "audio_repo_path": resolve_audio_path(record, repository_files),
        "audio_cluster_id": sealed_hash(resolve_audio_path(record, repository_files)),
        "category": category_of(record),
        "subcategory": str(record.get("subcategory", record.get("Subcategories", ""))),
        "source_dataset": str(record.get("dataset", "human_authored")),
        "task": str(record.get("task", "human_authored")),
        "question_sha256_sealed": sealed_hash(record.get("question")),
        "ground_truth_sha256_sealed": sealed_hash(ground_truth),
        "record_sha256_sealed": sealed_hash(record),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "q1_plus" / "data" / "manifests",
    )
    args = parser.parse_args()
    metadata = Path(
        hf_hub_download(
            REPO_ID,
            METADATA_FILE,
            repo_type="dataset",
            revision=REVISION,
        )
    )
    if sha256(metadata) != METADATA_SHA256:
        raise RuntimeError("AUDITA metadata checksum mismatch")
    records = json.loads(metadata.read_text(encoding="utf-8"))
    if len(records) != 9690:
        raise RuntimeError(f"expected 9690 metadata rows, got {len(records)}")
    repository_files = set(
        list_repo_files(REPO_ID, repo_type="dataset", revision=REVISION)
    )

    primary: list[dict[str, Any]] = []
    external: list[dict[str, Any]] = []
    outliers: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        is_human = record.get("dataset") is None
        category = category_of(record)
        if index in EXCLUSIONS:
            excluded.append(
                {
                    "row_index": index,
                    "reason": EXCLUSIONS[index],
                    "panel_origin": "human" if is_human else "external",
                    "audio_repo_path": resolve_audio_path(record, repository_files),
                    "record_sha256_sealed": sealed_hash(record),
                }
            )
        elif not is_human:
            external.append(sealed_row(index, record, repository_files, "audita_external"))
        elif category in MAJOR_CATEGORIES:
            primary.append(
                sealed_row(index, record, repository_files, "audita_human_primary")
            )
        else:
            outliers.append(
                sealed_row(index, record, repository_files, "audita_human_outlier")
            )

    if len(primary) != 6457 or len(external) != 3229 or len(outliers) != 2:
        raise RuntimeError(
            f"unexpected panel sizes: primary={len(primary)}, "
            f"external={len(external)}, outliers={len(outliers)}"
        )
    outputs = {
        "audita_human_primary.label_sealed.jsonl": primary,
        "audita_external.label_sealed.jsonl": external,
        "audita_human_outliers.label_sealed.jsonl": outliers,
        "audita_exclusions.jsonl": excluded,
    }
    for name, rows in outputs.items():
        write_jsonl(args.output_dir / name, rows)

    audit = {
        "status": "label_sealed_manifests_complete_no_model_evaluation",
        "repo_id": REPO_ID,
        "revision": REVISION,
        "metadata_file": METADATA_FILE,
        "metadata_sha256": sha256(metadata),
        "repository_file_count": len(repository_files),
        "counts": {name: len(rows) for name, rows in outputs.items()},
        "primary_category_counts": dict(
            sorted(Counter(row["category"] for row in primary).items())
        ),
        "unique_audio_clusters": {
            "human_primary": len({row["audio_cluster_id"] for row in primary}),
            "external": len({row["audio_cluster_id"] for row in external}),
            "human_outliers": len({row["audio_cluster_id"] for row in outliers}),
        },
        "manifest_sha256": {
            name: sha256(args.output_dir / name) for name in outputs
        },
        "sealed_fields_not_written": ["question", "ground_truth", "original_ground_truth"],
        "exclusions": EXCLUSIONS,
    }
    audit_path = args.output_dir / "audita_manifest_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
