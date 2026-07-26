#!/usr/bin/env python3
"""Verify the frozen EviAudio Q1-plus release and its claim boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
Q1 = ROOT / "q1_plus"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(relative: str) -> dict:
    with (ROOT / relative).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_file_map(lock_name: str) -> int:
    lock = load_json(f"q1_plus/{lock_name}")
    files = lock.get("files", {})
    if not files:
        raise AssertionError(f"{lock_name} has no file hash map")
    for relative, expected in files.items():
        path = ROOT / relative
        if not path.is_file():
            raise AssertionError(f"missing locked file: {relative}")
        observed = sha256(path)
        if observed != expected:
            raise AssertionError(
                f"hash mismatch for {relative}: {observed} != {expected}"
            )
    return len(files)


def tree_digest(relative_directories: list[str]) -> tuple[str, int]:
    paths: list[Path] = []
    for relative in relative_directories:
        directory = ROOT / relative
        if not directory.is_dir():
            raise AssertionError(f"missing locked directory: {relative}")
        paths.extend(path for path in directory.rglob("*") if path.is_file())
    records = []
    for path in sorted(paths, key=lambda item: item.relative_to(ROOT).as_posix()):
        relative = path.relative_to(ROOT).as_posix()
        records.append(f"{sha256(path)}  {relative}\n")
    payload = "".join(records).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), len(paths)


def verify_figure_tree() -> int:
    lock = load_json("q1_plus/HELDOUT_ANSWER_FINAL_LOCK.json")
    tree = lock["heldout_figure_tree"]
    observed_digest, observed_count = tree_digest(tree["directories"])
    if observed_digest != tree["sha256"]:
        raise AssertionError("held-out figure-tree digest mismatch")
    if observed_count != tree["files"]:
        raise AssertionError("held-out figure-tree file-count mismatch")
    return observed_count


def verify_figure_suite(relative: str, expected: int) -> None:
    directory = ROOT / relative
    pdfs = sorted((directory / "pdf").glob("*.pdf"))
    pngs = sorted((directory / "png").glob("*.png"))
    csvs = sorted((directory / "source_data").glob("*.csv"))
    counts = (len(pdfs), len(pngs), len(csvs))
    if counts != (expected, expected, expected):
        raise AssertionError(f"unexpected {relative} counts: {counts}")
    stems = ({p.stem for p in pdfs}, {p.stem for p in pngs}, {p.stem for p in csvs})
    if not (stems[0] == stems[1] == stems[2]):
        raise AssertionError(f"cross-format figure IDs differ in {relative}")
    for path in pdfs:
        with path.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                raise AssertionError(f"invalid PDF header: {path}")


def verify_claim_boundary() -> None:
    event = load_json(
        "q1_plus/results/confirmatory/event_ranker/"
        "five_seed_confirmatory_report.json"
    )
    heldout = load_json(
        "q1_plus/results/development/answer_generation/"
        "heldout_evaluation_report.json"
    )
    verdict = load_json("q1_plus/Q1_READINESS_VERDICT.json")
    rights = load_json("q1_plus/AUDITA_LICENSE_PROVENANCE_AUDIT.json")

    if event["status"] != "confirmatory_exact_onset_gate_pass":
        raise AssertionError("unexpected exact-onset status")
    if not event["all_promotion_gates_pass"]:
        raise AssertionError("exact-onset promotion should pass")
    if heldout["status"] != "heldout_answer_gate_fail_audita_sealed":
        raise AssertionError("unexpected held-out answer status")
    if heldout["all_promotion_gates_pass"]:
        raise AssertionError("held-out answer promotion must remain failed")
    if heldout["audita_status"] != "sealed":
        raise AssertionError("AUDITA is not sealed in the held-out report")
    if heldout["integrity"]["audita_rows_accessed"] != 0:
        raise AssertionError("held-out report records AUDITA access")
    if verdict["status"] != (
        "not_ready_for_positive_q1_claim_heldout_answer_gate_failed"
    ):
        raise AssertionError("readiness verdict was broadened")
    if verdict["audita"]["status"] != "permanently_sealed_for_this_pipeline":
        raise AssertionError("final verdict does not seal AUDITA")
    if rights["audio_files_or_waveforms_opened"] != 0:
        raise AssertionError("AUDITA waveform access is nonzero")
    if rights["model_outputs_opened"] != 0 or rights["audita_rows_scored"] != 0:
        raise AssertionError("AUDITA evaluation access is nonzero")

    manuscript = (Q1 / "MANUSCRIPT_Q1_PLUS.md").read_text(encoding="utf-8")
    for marker in ("PLACEHOLDER", "TODO", "TBD"):
        if marker in manuscript:
            raise AssertionError(f"unfinished manuscript marker: {marker}")


def main() -> None:
    locked_files = sum(
        verify_file_map(name)
        for name in (
            "HELDOUT_ANSWER_FINAL_LOCK.json",
            "MANUSCRIPT_FINAL_LOCK.json",
            "EVIAUDIO_FINAL_LOCK.json",
        )
    )
    figure_tree_files = verify_figure_tree()
    verify_figure_suite("q1_plus/figures_exact_onset", 43)
    verify_figure_suite("q1_plus/figures_heldout_answer", 40)
    verify_claim_boundary()
    print(
        json.dumps(
            {
                "status": "eviaudio_final_release_verified",
                "locked_file_entries_verified": locked_files,
                "heldout_figure_tree_files_verified": figure_tree_files,
                "distinct_q1_plus_figures_verified": 83,
                "audita_rows_accessed": 0,
                "final_readiness": (
                    "not_ready_for_positive_q1_claim_heldout_answer_gate_failed"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
