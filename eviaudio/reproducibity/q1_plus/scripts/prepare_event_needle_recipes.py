#!/usr/bin/env python3
"""Create source-isolated exact-onset event-needle recipes without model scoring."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import soundfile as sf


PROJECT = Path(__file__).resolve().parents[2]
ESC_COMMIT = "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"
LIBRISPEECH_DEV_MD5 = "42e2234ba48799c1f50f24a7926300a1"
LIBRISPEECH_TEST_MD5 = "32fa31d27d2e1cad72775fee3f4849a9"
LENGTH_SECONDS = (60, 180, 300)
TARGET_SNR_DB = (-5, 0, 5)
POSITION_BINS = ("early", "middle", "late")
QUERY_TEMPLATES = (
    "Locate the sound of {label}.",
    "When can {label} be heard?",
    "Find the segment containing {label}.",
)


def digest(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def sha256(path: Path) -> str:
    block_size = 1024 * 1024
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            checksum.update(block)
    return checksum.hexdigest()


def md5(path: Path) -> str:
    checksum = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            checksum.update(block)
    return checksum.hexdigest()


def choose(values: tuple[Any, ...], key: str, byte: int) -> Any:
    return values[digest(key)[byte] % len(values)]


def position_start(length: int, event_duration: float, bin_name: str, key: str) -> float:
    fractions = {"early": (0.08, 0.28), "middle": (0.40, 0.60), "late": (0.72, 0.92)}
    low, high = fractions[bin_name]
    usable_end = max(0.0, length - event_duration - 0.25)
    lo_seconds = min(low * length, usable_end)
    hi_seconds = min(high * length, usable_end)
    fraction = int.from_bytes(digest(key)[:8], "big") / float(2**64 - 1)
    return round(lo_seconds + fraction * max(0.0, hi_seconds - lo_seconds), 3)


def nonoverlapping_distractor_starts(
    length: int,
    target_start: float,
    event_duration: float,
    count: int,
    key: str,
) -> list[float]:
    occupied = [(target_start - 1.0, target_start + event_duration + 1.0)]
    starts: list[float] = []
    maximum = max(0.0, length - event_duration - 0.25)
    for distractor_index in range(count):
        accepted = None
        for attempt in range(1000):
            number = int.from_bytes(
                digest(f"{key}|{distractor_index}|{attempt}")[:8], "big"
            )
            candidate = (number / float(2**64 - 1)) * maximum
            interval = (candidate - 1.0, candidate + event_duration + 1.0)
            if all(interval[1] <= old[0] or interval[0] >= old[1] for old in occupied):
                accepted = round(candidate, 3)
                occupied.append(interval)
                starts.append(accepted)
                break
        if accepted is None:
            raise RuntimeError(f"could not place distractor for {key}")
    return starts


def speaker_id(path: Path) -> str:
    return path.parts[-3]


def audio_duration(path: Path) -> float:
    info = sf.info(path)
    return float(info.frames / info.samplerate)


def select_backgrounds(paths: list[Path], length: int, key: str) -> list[Path]:
    ordered = sorted(paths, key=lambda path: digest(f"{key}|{path.as_posix()}"))
    selected: list[Path] = []
    duration = 0.0
    cursor = 0
    while duration < length:
        path = ordered[cursor % len(ordered)]
        selected.append(path)
        duration += audio_duration(path)
        cursor += 1
        if cursor > len(ordered) * 4:
            raise RuntimeError("background pool is too short")
    return selected


def event_duration(path: Path) -> float:
    duration = audio_duration(path)
    if not math.isclose(duration, 5.0, abs_tol=0.02):
        raise RuntimeError(f"unexpected ESC-50 duration {duration}: {path}")
    return duration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--esc-root",
        type=Path,
        default=PROJECT / "q1_plus" / "data" / "external" / "ESC-50",
    )
    parser.add_argument(
        "--librispeech-root",
        type=Path,
        default=PROJECT / "q1_plus" / "data" / "external" / "LibriSpeech",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "q1_plus" / "data" / "event_needle",
    )
    args = parser.parse_args()

    esc_head = subprocess.check_output(
        ["git", "-C", str(args.esc_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if esc_head != ESC_COMMIT:
        raise RuntimeError(f"ESC-50 commit mismatch: {esc_head}")
    archive_dir = args.librispeech_root.parent
    if md5(archive_dir / "dev-clean.tar.gz") != LIBRISPEECH_DEV_MD5:
        raise RuntimeError("LibriSpeech dev-clean checksum mismatch")
    if md5(archive_dir / "test-clean.tar.gz") != LIBRISPEECH_TEST_MD5:
        raise RuntimeError("LibriSpeech test-clean checksum mismatch")

    metadata_path = args.esc_root / "meta" / "esc50.csv"
    with metadata_path.open("r", encoding="utf-8") as handle:
        events = list(csv.DictReader(handle))
    if len(events) != 2000:
        raise RuntimeError(f"expected 2000 ESC-50 events, got {len(events)}")
    for row in events:
        row["fold"] = int(row["fold"])
        row["target"] = int(row["target"])

    # Validate the source-isolation property rather than merely trusting fold names.
    # ESC-50 v2 contains four Freesound IDs assigned across adjacent folds; remove
    # every row from those IDs so neither side retains a correlated fragment.
    folds_by_source: dict[str, set[int]] = defaultdict(set)
    for row in events:
        folds_by_source[str(row["src_file"])].add(int(row["fold"]))
    cross_fold_source_ids = sorted(
        source for source, folds in folds_by_source.items() if len(folds) > 1
    )
    cross_fold_rows = [
        str(row["filename"])
        for row in events
        if str(row["src_file"]) in cross_fold_source_ids
    ]
    events = [
        row for row in events if str(row["src_file"]) not in cross_fold_source_ids
    ]
    sources_by_fold = {
        fold: {str(row["src_file"]) for row in events if row["fold"] == fold}
        for fold in range(1, 6)
    }
    for first in range(1, 6):
        for second in range(first + 1, 6):
            overlap = sources_by_fold[first] & sources_by_fold[second]
            if overlap:
                raise RuntimeError(f"ESC source leakage between folds: {first}, {second}")

    dev_audio = sorted((args.librispeech_root / "dev-clean").rglob("*.flac"))
    test_audio = sorted((args.librispeech_root / "test-clean").rglob("*.flac"))
    if not dev_audio or len(test_audio) != 2620:
        raise RuntimeError("LibriSpeech clean splits are incomplete")
    dev_speakers = sorted({speaker_id(path) for path in dev_audio})
    validation_speakers = {
        speaker for speaker in dev_speakers if digest(speaker)[0] % 4 == 0
    }
    development_speakers = set(dev_speakers) - validation_speakers
    development_audio = [p for p in dev_audio if speaker_id(p) in development_speakers]
    validation_audio = [p for p in dev_audio if speaker_id(p) in validation_speakers]
    test_speakers = {speaker_id(path) for path in test_audio}
    if development_speakers & validation_speakers:
        raise RuntimeError("development/validation background speaker leakage")
    if test_speakers & set(dev_speakers):
        raise RuntimeError("validation/confirmatory background speaker leakage")

    panel_for_fold = {1: "development", 2: "development", 3: "development", 4: "validation", 5: "confirmatory"}
    background_for_panel = {
        "development": development_audio,
        "validation": validation_audio,
        "confirmatory": test_audio,
    }
    by_fold_supercategory: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in events:
        by_fold_supercategory[(row["fold"], row["target"] // 10)].append(row)

    checksums: dict[Path, str] = {}

    def file_checksum(path: Path) -> str:
        if path not in checksums:
            checksums[path] = sha256(path)
        return checksums[path]

    all_recipes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in events:
        filename = str(row["filename"])
        fold = int(row["fold"])
        panel = panel_for_fold[fold]
        key = f"event-needle-v1|{filename}"
        length = int(choose(LENGTH_SECONDS, key, 0))
        snr = int(choose(TARGET_SNR_DB, key, 1))
        position = str(choose(POSITION_BINS, key, 2))
        template_index = digest(key)[3] % len(QUERY_TEMPLATES)
        label = str(row["category"]).replace("_", " ")
        target_path = args.esc_root / "audio" / filename
        duration = event_duration(target_path)
        start = position_start(length, duration, position, key)

        candidates = [
            other
            for other in by_fold_supercategory[(fold, int(row["target"]) // 10)]
            if int(other["target"]) != int(row["target"])
            and str(other["src_file"]) != str(row["src_file"])
        ]
        candidates.sort(
            key=lambda other: digest(f"{key}|distractor|{other['filename']}")
        )
        distractors: list[dict[str, Any]] = []
        used_categories: set[int] = set()
        for other in candidates:
            other_category = int(other["target"])
            if other_category in used_categories:
                continue
            used_categories.add(other_category)
            distractors.append(other)
            if len(distractors) == 3:
                break
        if len(distractors) != 3:
            raise RuntimeError(f"insufficient ontology distractors for {filename}")
        distractor_starts = nonoverlapping_distractor_starts(
            length, start, duration, len(distractors), key
        )
        backgrounds = select_backgrounds(background_for_panel[panel], length, key)
        for path in [target_path, *[args.esc_root / "audio" / str(x["filename"]) for x in distractors], *backgrounds]:
            file_checksum(path)

        recipe = {
            "recipe_id": hashlib.sha256(key.encode("utf-8")).hexdigest(),
            "protocol": "event_needle_v1_no_model_based_construction",
            "panel": panel,
            "esc_fold": fold,
            "target_event": {
                "path": str(target_path.relative_to(PROJECT)),
                "sha256": checksums[target_path],
                "source_file_id": str(row["src_file"]),
                "class_id": int(row["target"]),
                "class_label": label,
                "start_sec": start,
                "end_sec": round(start + duration, 3),
                "snr_db": snr,
            },
            "query": QUERY_TEMPLATES[template_index].format(label=label),
            "query_template_index": template_index,
            "duration_sec": length,
            "position_bin": position,
            "distractor_events": [
                {
                    "path": str((args.esc_root / "audio" / str(other["filename"])).relative_to(PROJECT)),
                    "sha256": checksums[args.esc_root / "audio" / str(other["filename"])],
                    "source_file_id": str(other["src_file"]),
                    "class_id": int(other["target"]),
                    "class_label": str(other["category"]).replace("_", " "),
                    "start_sec": distractor_start,
                    "end_sec": round(distractor_start + event_duration(args.esc_root / "audio" / str(other["filename"])), 3),
                    "snr_db": 0,
                }
                for other, distractor_start in zip(distractors, distractor_starts, strict=True)
            ],
            "background_files": [
                {
                    "path": str(path.relative_to(PROJECT)),
                    "sha256": checksums[path],
                    "speaker_id": speaker_id(path),
                }
                for path in backgrounds
            ],
            "foreground_cluster_id": f"esc50_src_{row['src_file']}",
            "background_speaker_ids": sorted({speaker_id(path) for path in backgrounds}),
        }
        all_recipes[panel].append(recipe)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = {
        "development": "development_recipes.jsonl",
        "validation": "validation_recipes.jsonl",
        "confirmatory": "confirmatory_recipes.sealed.jsonl",
    }
    output_checksums: dict[str, str] = {}
    for panel, name in names.items():
        path = args.output_dir / name
        with path.open("w", encoding="utf-8") as handle:
            for recipe in sorted(all_recipes[panel], key=lambda item: item["recipe_id"]):
                handle.write(json.dumps(recipe, sort_keys=True) + "\n")
        output_checksums[name] = sha256(path)

    audit = {
        "status": "recipes_frozen_before_any_confirmatory_model_evaluation",
        "esc50_commit": ESC_COMMIT,
        "esc50_metadata_sha256": sha256(metadata_path),
        "librispeech_archives": {
            "dev_clean_md5": LIBRISPEECH_DEV_MD5,
            "test_clean_md5": LIBRISPEECH_TEST_MD5,
        },
        "counts": {panel: len(rows) for panel, rows in all_recipes.items()},
        "class_counts": {
            panel: dict(sorted(Counter(r["target_event"]["class_label"] for r in rows).items()))
            for panel, rows in all_recipes.items()
        },
        "condition_counts": {
            panel: {
                "duration": dict(sorted(Counter(r["duration_sec"] for r in rows).items())),
                "snr_db": dict(sorted(Counter(r["target_event"]["snr_db"] for r in rows).items())),
                "position": dict(sorted(Counter(r["position_bin"] for r in rows).items())),
            }
            for panel, rows in all_recipes.items()
        },
        "source_isolation": {
            "esc_original_source_overlap_across_folds": 0,
            "excluded_cross_fold_source_ids": cross_fold_source_ids,
            "excluded_cross_fold_rows": sorted(cross_fold_rows),
            "development_validation_background_speaker_overlap": 0,
            "validation_confirmatory_background_speaker_overlap": 0,
        },
        "output_sha256": output_checksums,
        "materialized_long_mixtures": 0,
    }
    audit_path = args.output_dir / "recipe_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
