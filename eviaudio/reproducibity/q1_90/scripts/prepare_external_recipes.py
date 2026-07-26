#!/usr/bin/env python3
"""Build model-blind UrbanSound8K/LibriSpeech exact-onset recipes."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import soundfile as sf


PROJECT = Path(__file__).resolve().parents[2]
TRACK = PROJECT / "q1_90"
CONFIG = TRACK / "configs/external_replication.json"
URBAN = TRACK / "data/external/UrbanSound8K"
LIBRI = TRACK / "data/external/LibriSpeech"
ESC_METADATA = PROJECT / "q1_plus/data/external/ESC-50/meta/esc50.csv"
OUTPUT_DIR = TRACK / "data/external_replication"
RECIPES = OUTPUT_DIR / "external_recipes.sealed.jsonl"
AUDIT = OUTPUT_DIR / "recipe_audit.json"


def digest(text: str) -> bytes:
    return hashlib.sha256(text.encode()).digest()


def sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            checksum.update(block)
    return checksum.hexdigest()


def md5(path: Path) -> str:
    checksum = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            checksum.update(block)
    return checksum.hexdigest()


def duration(path: Path) -> float:
    info = sf.info(path)
    value = float(info.frames / info.samplerate)
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"invalid audio duration: {path}")
    return value


def speaker_id(path: Path) -> str:
    return path.parts[-3]


def choose(values: list[Any], key: str, offset: int) -> Any:
    return values[digest(key)[offset] % len(values)]


def normalized_source(value: Any) -> str:
    return str(int(float(str(value))))


def event_path(row: dict[str, Any]) -> Path:
    return URBAN / "audio" / f"fold{int(row['fold'])}" / str(row["slice_file_name"])


def target_start(total: int, event_duration: float, position: str, key: str) -> float:
    bounds = {"early": (0.08, 0.28), "middle": (0.40, 0.60), "late": (0.72, 0.92)}
    low, high = bounds[position]
    maximum = max(0.0, total - event_duration - 0.25)
    lo = min(low * total, maximum)
    hi = min(high * total, maximum)
    fraction = int.from_bytes(digest(key)[:8], "big") / float(2**64 - 1)
    return round(lo + fraction * max(0.0, hi - lo), 3)


def distractor_starts(
    total: int,
    target_interval: tuple[float, float],
    distractor_durations: list[float],
    key: str,
) -> list[float]:
    occupied = [(target_interval[0] - 1.0, target_interval[1] + 1.0)]
    starts: list[float] = []
    for index, event_duration in enumerate(distractor_durations):
        maximum = max(0.0, total - event_duration - 0.25)
        for attempt in range(2000):
            number = int.from_bytes(digest(f"{key}|place|{index}|{attempt}")[:8], "big")
            candidate = (number / float(2**64 - 1)) * maximum
            interval = (candidate - 1.0, candidate + event_duration + 1.0)
            if all(interval[1] <= old[0] or interval[0] >= old[1] for old in occupied):
                starts.append(round(candidate, 3))
                occupied.append(interval)
                break
        else:
            raise RuntimeError(f"cannot place distractor: {key}/{index}")
    return starts


def select_backgrounds(paths: list[Path], seconds: int, key: str) -> list[Path]:
    ordered = sorted(paths, key=lambda path: digest(f"{key}|background|{path.as_posix()}"))
    selected: list[Path] = []
    accumulated = 0.0
    for path in ordered:
        selected.append(path)
        accumulated += duration(path)
        if accumulated >= seconds:
            return selected
    raise RuntimeError("background pool is too short")


def main() -> None:
    if RECIPES.exists() or AUDIT.exists():
        raise FileExistsError("external recipes are immutable")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["status"] != "frozen_after_model_blind_source_count_amendment":
        raise PermissionError("external replication config is not frozen")
    amendment_path = PROJECT / config["model_blind_amendment"]
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    if (
        amendment["status"] != "complete_before_any_external_model_scoring"
        or int(amendment["trigger"]["recipes_written"]) != 0
        or int(amendment["trigger"]["model_scores_computed"]) != 0
        or amendment["amended_eligible_folds"] != config["foreground"]["eligible_folds"]
        or amendment["minimum_unique_sources_per_class"]
        != config["foreground"]["minimum_targets_per_class"]
        or amendment["performance_information_used"] is not False
    ):
        raise PermissionError("model-blind source-count amendment is invalid")
    urban_archive = TRACK / "data/external/UrbanSound8K.tar.gz"
    libri_archive = TRACK / "data/external/test-other.tar.gz"
    if md5(urban_archive) != config["foreground"]["archive_md5"]:
        raise RuntimeError("UrbanSound8K archive checksum mismatch")
    if md5(libri_archive) != config["background"]["archive_md5"]:
        raise RuntimeError("LibriSpeech test-other archive checksum mismatch")

    with ESC_METADATA.open("r", encoding="utf-8") as handle:
        esc_source_ids = {normalized_source(row["src_file"]) for row in csv.DictReader(handle)}
    metadata = URBAN / "metadata/UrbanSound8K.csv"
    with metadata.open("r", encoding="utf-8-sig") as handle:
        all_rows = list(csv.DictReader(handle))
    if len(all_rows) != 8732:
        raise RuntimeError(f"expected 8732 UrbanSound8K rows, got {len(all_rows)}")
    eligible_folds = {int(value) for value in config["foreground"]["eligible_folds"]}
    eligible = []
    excluded_esc_overlap = []
    for row in all_rows:
        row["fold"] = int(row["fold"])
        row["classID"] = int(row["classID"])
        row["fsID"] = normalized_source(row["fsID"])
        if row["fold"] not in eligible_folds:
            continue
        if row["fsID"] in esc_source_ids:
            excluded_esc_overlap.append(row)
            continue
        path = event_path(row)
        if not path.is_file():
            raise RuntimeError(f"missing UrbanSound file: {path}")
        eligible.append(row)

    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_class[str(row["class"])].append(row)
    targets: list[dict[str, Any]] = []
    maximum = int(config["foreground"]["maximum_targets_per_class"])
    minimum = int(config["foreground"]["minimum_targets_per_class"])
    salt = str(config["construction"]["hash_salt"])
    for label, rows in sorted(by_class.items()):
        ordered = sorted(rows, key=lambda row: digest(f"{salt}|target|{row['fsID']}|{row['slice_file_name']}"))
        seen: set[str] = set()
        selected = []
        for row in ordered:
            if row["fsID"] in seen:
                continue
            seen.add(row["fsID"])
            selected.append(row)
            if len(selected) == maximum:
                break
        if len(selected) < minimum:
            raise RuntimeError(f"class {label} has only {len(selected)} isolated target sources")
        targets.extend(selected)
    target_ids = {str(row["fsID"]) for row in targets}
    if len(target_ids) != len(targets):
        raise RuntimeError("target original source reused across recipes")

    distractor_pool = [row for row in eligible if str(row["fsID"]) not in target_ids]
    backgrounds = sorted((LIBRI / "test-other").rglob("*.flac"))
    if not backgrounds:
        raise RuntimeError("LibriSpeech test-other is incomplete")
    prior_speakers = {
        speaker_id(path)
        for split in ("dev-clean", "test-clean")
        for path in (PROJECT / "q1_plus/data/external/LibriSpeech" / split).rglob("*.flac")
    }
    external_speakers = {speaker_id(path) for path in backgrounds}
    if prior_speakers & external_speakers:
        raise RuntimeError("LibriSpeech speaker overlap with prior panels")

    checksums: dict[Path, str] = {}
    def file_hash(path: Path) -> str:
        if path not in checksums:
            checksums[path] = sha256(path)
        return checksums[path]

    construction = config["construction"]
    recipes = []
    for row in sorted(targets, key=lambda item: digest(f"{salt}|recipe|{item['fsID']}")):
        key = f"{salt}|{row['fsID']}|{row['slice_file_name']}"
        total = int(choose(construction["durations_sec"], key, 0))
        snr = int(choose(construction["target_snr_db"], key, 1))
        position = str(choose(construction["position_bins"], key, 2))
        template_index = digest(key)[3] % len(construction["query_templates"])
        label = str(row["class"]).replace("_", " ")
        target_file = event_path(row)
        target_duration = duration(target_file)
        start = target_start(total, target_duration, position, key)
        candidates = [
            item for item in distractor_pool
            if int(item["fold"]) == int(row["fold"])
            and str(item["class"]) != str(row["class"])
            and str(item["fsID"]) != str(row["fsID"])
        ]
        candidates.sort(key=lambda item: digest(f"{key}|distractor|{item['fsID']}|{item['slice_file_name']}"))
        chosen = []
        used_classes: set[str] = set()
        used_sources: set[str] = set()
        for item in candidates:
            if str(item["class"]) in used_classes or str(item["fsID"]) in used_sources:
                continue
            chosen.append(item)
            used_classes.add(str(item["class"]))
            used_sources.add(str(item["fsID"]))
            if len(chosen) == int(construction["distractors_per_recipe"]):
                break
        if len(chosen) != int(construction["distractors_per_recipe"]):
            raise RuntimeError(f"insufficient distractors for {key}")
        chosen_paths = [event_path(item) for item in chosen]
        chosen_durations = [duration(path) for path in chosen_paths]
        starts = distractor_starts(total, (start, start + target_duration), chosen_durations, key)
        background_files = select_backgrounds(backgrounds, total, key)
        for path in [target_file, *chosen_paths, *background_files]:
            file_hash(path)
        recipe = {
            "recipe_id": hashlib.sha256(key.encode()).hexdigest(),
            "protocol": "urbansound8k_zero_shot_external_v1",
            "panel": "confirmatory",
            "urban_fold": int(row["fold"]),
            "target_event": {
                "path": str(target_file.relative_to(PROJECT)),
                "sha256": file_hash(target_file),
                "source_file_id": str(row["fsID"]),
                "class_id": int(row["classID"]),
                "class_label": label,
                "start_sec": start,
                "end_sec": round(start + target_duration, 3),
                "snr_db": snr
            },
            "query": str(construction["query_templates"][template_index]).format(label=label),
            "query_template_index": template_index,
            "duration_sec": total,
            "position_bin": position,
            "distractor_events": [
                {
                    "path": str(path.relative_to(PROJECT)),
                    "sha256": file_hash(path),
                    "source_file_id": str(item["fsID"]),
                    "class_id": int(item["classID"]),
                    "class_label": str(item["class"]).replace("_", " "),
                    "start_sec": event_start,
                    "end_sec": round(event_start + event_duration, 3),
                    "snr_db": 0
                }
                for item, path, event_duration, event_start in zip(chosen, chosen_paths, chosen_durations, starts, strict=True)
            ],
            "background_files": [
                {
                    "path": str(path.relative_to(PROJECT)),
                    "sha256": file_hash(path),
                    "speaker_id": speaker_id(path)
                }
                for path in background_files
            ],
            "foreground_cluster_id": f"urbansound_fsid_{row['fsID']}",
            "background_speaker_ids": sorted({speaker_id(path) for path in background_files})
        }
        recipes.append(recipe)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with RECIPES.open("w", encoding="utf-8") as handle:
        for recipe in recipes:
            handle.write(json.dumps(recipe, sort_keys=True) + "\n")
    source_manifest = "".join(
        f"{path.relative_to(PROJECT).as_posix()}\0{checksum}\n"
        for path, checksum in sorted(checksums.items(), key=lambda item: item[0].as_posix())
    )
    audit = {
        "status": "external_recipes_frozen_before_any_model_scoring",
        "config_sha256": sha256(CONFIG),
        "urban_metadata_sha256": sha256(metadata),
        "urban_archive_md5": md5(urban_archive),
        "librispeech_archive_md5": md5(libri_archive),
        "recipes": len(recipes),
        "classes": dict(sorted(Counter(recipe["target_event"]["class_label"] for recipe in recipes).items())),
        "folds": dict(sorted(Counter(recipe["urban_fold"] for recipe in recipes).items())),
        "conditions": {
            "duration_sec": dict(sorted(Counter(recipe["duration_sec"] for recipe in recipes).items())),
            "snr_db": dict(sorted(Counter(recipe["target_event"]["snr_db"] for recipe in recipes).items())),
            "position_bin": dict(sorted(Counter(recipe["position_bin"] for recipe in recipes).items()))
        },
        "source_isolation": {
            "urban_target_original_sources": len(target_ids),
            "target_source_reuse": 0,
            "esc50_freesound_id_overlap": 0,
            "urban_rows_excluded_for_esc50_overlap": len(excluded_esc_overlap),
            "prior_librispeech_speaker_overlap": 0,
            "external_librispeech_speakers": len(external_speakers)
        },
        "unique_source_files_hashed": len(checksums),
        "source_manifest_sha256": hashlib.sha256(source_manifest.encode()).hexdigest(),
        "recipes_sha256": sha256(RECIPES),
        "model_scores_computed": 0
    }
    AUDIT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
