#!/usr/bin/env python3
"""Build and verify the upgraded IEEE/ACM TASLP submission bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[2]
TRACK = PROJECT / "q1_crossfit_capcorrect"
ROOT = PROJECT.parent
BUNDLE = PROJECT / "submission_taslp_q1_upgrade"
ARCHIVE = PROJECT / "EviAudio_TASLP_Q1_Upgrade_submission.zip"
TECTONIC = ROOT / "tools/tectonic-0.16.9/tectonic"
ANALYSIS = TRACK / "publication_upgrade/EVIAUDIO_UPGRADE_ANALYSIS.json"
TEMPLATES = TRACK / "manuscript"
METRIC_PARITY = PROJECT / "q1_top_tier/publication/OFFICIAL_METRIC_PARITY.json"
NATURAL_REPORT = (
    PROJECT
    / "q1_top_tier/results/perception_test/evaluation/"
    "perception_test_report.json"
)
ROUTER_REPORT = TRACK / "results/capcorrect_router/capcorrect_router_report.json"
DOWNSTREAM_REPORT = (
    PROJECT
    / "q1_plus/results/development/answer_generation/"
    "heldout_evaluation_report.json"
)
SOURCE_DATE_EPOCH = "1784851200"
MAXIMUM_INCLUDED_BYTES = 95 * 1024 * 1024
PUBLIC_TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def private_path_patterns() -> tuple[re.Pattern[str], ...]:
    """Return portable-path guards without embedding a matching literal."""
    return (
        re.compile(r"/" + r"home/" + r"[A-Za-z0-9._-]+/"),
        re.compile(r"/" + r"Users/" + r"[A-Za-z0-9._-]+/"),
        re.compile(
            r"[A-Za-z]:"
            + re.escape("\\")
            + r"Users"
            + re.escape("\\")
            + r"[A-Za-z0-9._-]+"
            + re.escape("\\")
        ),
    )


def copy_public_file(source: Path, target: Path) -> dict[str, Any] | None:
    """Copy one file, normalizing machine-local roots in public text."""
    if source.suffix.lower() not in PUBLIC_TEXT_SUFFIXES:
        shutil.copy2(source, target)
        return None
    try:
        original = source.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        shutil.copy2(source, target)
        return None

    project_prefix = str(PROJECT.resolve()) + "/"
    home_prefix = str(Path.home().resolve()) + "/"
    public = original.replace(project_prefix, "")
    public = public.replace(home_prefix, "<LOCAL_HOME>/")
    if public == original:
        shutil.copy2(source, target)
        return None

    target.write_text(public, encoding="utf-8")
    return {
        "source_bytes": source.stat().st_size,
        "source_sha256": sha256(source),
        "public_bytes": target.stat().st_size,
        "public_sha256": sha256(target),
        "transformation": (
            "project-root prefixes made repository-relative; remaining "
            "local-home prefixes replaced by <LOCAL_HOME>"
        ),
    }


def verify_public_path_hygiene(root: Path) -> None:
    patterns = private_path_patterns()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in PUBLIC_TEXT_SUFFIXES:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(pattern.search(content) for pattern in patterns):
            raise RuntimeError(f"machine-local path remains in public file: {path}")


def command(
    arguments: list[str], *, cwd: Path, environment: dict[str, str] | None = None
) -> str:
    process = subprocess.run(
        arguments,
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"command failed ({process.returncode}): {' '.join(arguments)}\n"
            f"{process.stdout}\n{process.stderr}"
        )
    return process.stdout + process.stderr


def macro(name: str, value: Any) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def abstract_words(source: str) -> int:
    match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        source,
        flags=re.DOTALL,
    )
    if match is None:
        raise RuntimeError("abstract is absent")
    plain = re.sub(r"\\[A-Za-z]+\*?(?:\[[^]]*\])?", " ", match.group(1))
    plain = re.sub(r"\\[A-Za-z]+(?:\{[^{}]*\})?", " ", plain)
    plain = re.sub(r"[{}$~^&%_]", " ", plain)
    return len(re.findall(r"\b[A-Za-z0-9][A-Za-z0-9'/-]*\b", plain))


def pdf_pages(path: Path) -> int:
    output = command(["pdfinfo", str(path)], cwd=path.parent)
    match = re.search(r"^Pages:\s+(\d+)$", output, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError(f"cannot read PDF pages: {path}")
    return int(match.group(1))


def write_macros(analysis: dict[str, Any]) -> None:
    controlled = analysis["controlled_confirmatory"]
    external = analysis["external_exact_onset"]
    natural = analysis["frozen_natural_transfer"]
    diagnostic = analysis["cap_corrected_diagnosis"]
    supervised = analysis["supervised_context"]
    downstream = analysis["downstream_answer_bridge"]
    natural_report = json.loads(NATURAL_REPORT.read_text(encoding="utf-8"))
    downstream_report = json.loads(
        DOWNSTREAM_REPORT.read_text(encoding="utf-8")
    )
    parity = json.loads(METRIC_PARITY.read_text(encoding="utf-8"))

    natural_bootstrap = natural["bootstrap"]
    router_bootstrap = diagnostic["paired_video_bootstrap_vs_fixed_clap"]
    incremental = diagnostic["incremental_qcr_posthoc"][
        "paired_video_bootstrap"
    ]
    natural_class_delta = natural_report["primary_contrast"][
        "class_mean_ap_delta"
    ]
    cap = diagnostic["cap_audit"]
    answer_metrics = downstream_report["combined_metrics"]
    answer_comparison = downstream_report["comparisons"][
        "prefix_matched_retrieval"
    ]["combined_exact"]

    lines = [
        macro("ControlledN", int(controlled["examples"])),
        macro("ControlledChunks", f"{34193:,}"),
        macro("ControlledSources", f"{2956:,}"),
        macro("ControlledPriorAP", f"{float(controlled['prior']['evidence_ap']):.4f}"),
        macro("ControlledQCRAP", f"{float(controlled['ensemble']['evidence_ap']):.4f}"),
        macro("ControlledAPDelta", f"{float(controlled['delta']['evidence_ap']):.4f}"),
        macro(
            "ControlledAPLower",
            f"{float(controlled['bootstrap']['interval_95_percentile']['evidence_ap'][0]):.4f}",
        ),
        macro(
            "ControlledAPUpper",
            f"{float(controlled['bootstrap']['interval_95_percentile']['evidence_ap'][1]):.4f}",
        ),
        macro("ControlledPriorHit", f"{float(controlled['prior']['hit_at_1']):.4f}"),
        macro("ControlledQCRHit", f"{float(controlled['ensemble']['hit_at_1']):.4f}"),
        macro("ExternalN", int(external["examples"])),
        macro("ExternalPriorAP", f"{float(external['prior']['evidence_ap']):.4f}"),
        macro("ExternalQCRAP", f"{float(external['ensemble']['evidence_ap']):.4f}"),
        macro("ExternalAPDelta", f"{float(external['delta']['evidence_ap']):.4f}"),
        macro(
            "ExternalAPLower",
            f"{float(external['bootstrap']['interval_95_percentile']['evidence_ap'][0]):.4f}",
        ),
        macro(
            "ExternalAPUpper",
            f"{float(external['bootstrap']['interval_95_percentile']['evidence_ap'][1]):.4f}",
        ),
        macro("ExternalPriorHit", f"{float(external['prior']['hit_at_1']):.4f}"),
        macro("ExternalQCRHit", f"{float(external['ensemble']['hit_at_1']):.4f}"),
        macro("ExternalIndex", f"{float(external['external_performance_index']):.2f}"),
        macro("NaturalVideos", f"{int(natural['videos']):,}"),
        macro("NaturalEvents", f"{int(natural['events']):,}"),
        macro("NaturalCLAPMAP", f"{float(natural['clap_multiscale_map']):.6f}"),
        macro("NaturalQCRMAP", f"{float(natural['qcr_multiscale_map']):.6f}"),
        macro(
            "NaturalCLAPFourMAP",
            f"{float(natural_report['methods']['clap_4s']['mean_map']):.6f}",
        ),
        macro("NaturalDeltaPoints", f"{100.0 * float(natural['qcr_minus_clap_map']):.4f}"),
        macro(
            "NaturalVideoDeltaPoints",
            f"{100.0 * float(natural_bootstrap['observed_mean_delta']):.4f}",
        ),
        macro(
            "NaturalVideoLowerPoints",
            f"{100.0 * float(natural_bootstrap['interval_95_percentile'][0]):.4f}",
        ),
        macro(
            "NaturalVideoUpperPoints",
            f"{100.0 * float(natural_bootstrap['interval_95_percentile'][1]):.4f}",
        ),
        macro(
            "NaturalWorstClassPoints",
            f"{100.0 * min(float(value) for value in natural_class_delta.values()):.4f}",
        ),
        macro("RouterMAP", f"{float(diagnostic['full_router_map']):.6f}"),
        macro(
            "RouterDeltaPoints",
            f"{100.0 * float(diagnostic['full_minus_fixed_clap']):.4f}",
        ),
        macro(
            "RouterRelativeGain",
            f"{100.0 * (float(diagnostic['full_router_map']) / float(diagnostic['fixed_clap_map']) - 1.0):.2f}",
        ),
        macro(
            "RouterVideoDeltaPoints",
            f"{100.0 * float(router_bootstrap['observed_mean_delta']):.4f}",
        ),
        macro(
            "RouterVideoLowerPoints",
            f"{100.0 * float(router_bootstrap['interval_95_percentile'][0]):.4f}",
        ),
        macro(
            "RouterVideoUpperPoints",
            f"{100.0 * float(router_bootstrap['interval_95_percentile'][1]):.4f}",
        ),
        macro("CLAPRouterMAP", f"{float(diagnostic['clap_only_router_map']):.6f}"),
        macro(
            "CLAPRouterDeltaPoints",
            f"{100.0 * float(diagnostic['clap_only_minus_fixed_clap']):.4f}",
        ),
        macro(
            "IncrementalMAPPoints",
            f"{100.0 * float(diagnostic['full_minus_clap_only']):.4f}",
        ),
        macro(
            "IncrementalGainShare",
            f"{100.0 * float(diagnostic['fraction_total_gain_from_incremental_qcr']):.2f}",
        ),
        macro(
            "IncrementalVideoPoints",
            f"{100.0 * float(incremental['observed_mean_delta']):.4f}",
        ),
        macro(
            "IncrementalVideoLowerPoints",
            f"{100.0 * float(incremental['interval_95_percentile'][0]):.4f}",
        ),
        macro(
            "IncrementalVideoUpperPoints",
            f"{100.0 * float(incremental['interval_95_percentile'][1]):.4f}",
        ),
        macro("QCRAssignmentPercent", f"{100.0 * float(diagnostic['qcr_assignment_fraction']):.1f}"),
        macro("ActionFormerMAP", f"{float(supervised['mean_map']):.6f}"),
        macro(
            "ActionFormerGapPoints",
            f"{100.0 * float(supervised['gap_above_full_router']):.4f}",
        ),
        macro("FullPreCapRows", f"{int(cap['full_router']['pre_cap_rows']):,}"),
        macro("FullFinalRows", f"{int(cap['full_router']['final_rows']):,}"),
        macro("FullDiscardedRows", f"{int(cap['full_router']['discarded_rows']):,}"),
        macro("FullVideosOverCap", f"{int(cap['full_router']['videos_exceeding_cap']):,}"),
        macro("FullMaxPreCap", int(cap["full_router"]["maximum_pre_cap_rows_per_video"])),
        macro("CLAPPreCapRows", f"{int(cap['clap_only_router']['pre_cap_rows']):,}"),
        macro("CLAPFinalRows", f"{int(cap['clap_only_router']['final_rows']):,}"),
        macro("CLAPDiscardedRows", f"{int(cap['clap_only_router']['discarded_rows']):,}"),
        macro("CLAPVideosOverCap", f"{int(cap['clap_only_router']['videos_exceeding_cap']):,}"),
        macro("CLAPMaxPreCap", int(cap["clap_only_router"]["maximum_pre_cap_rows_per_video"])),
        macro("AnswerSelected", f"{float(downstream['selected_exact']):.4f}"),
        macro("AnswerCLAP", f"{float(downstream['clap_exact']):.4f}"),
        macro("AnswerStrongest", f"{float(downstream['strongest_exact']):.4f}"),
        macro(
            "AnswerRandom",
            f"{float(answer_metrics['deterministic_random_retrieval']['two_backbone_mean_source_macro_exact']):.4f}",
        ),
        macro(
            "AnswerOracle",
            f"{float(answer_metrics['oracle_retrieval']['two_backbone_mean_source_macro_exact']):.4f}",
        ),
        macro(
            "AnswerSilence",
            f"{float(answer_metrics['selected_retrieval_silenced']['two_backbone_mean_source_macro_exact']):.4f}",
        ),
        macro(
            "AnswerTextOnly",
            f"{float(answer_metrics['text_only']['two_backbone_mean_source_macro_exact']):.4f}",
        ),
        macro(
            "AnswerVsStrongestPoints",
            f"{100.0 * float(answer_comparison['delta']):.4f}",
        ),
        macro(
            "AnswerVsStrongestLowerPoints",
            f"{100.0 * float(answer_comparison['bootstrap_95_interval'][0]):.4f}",
        ),
        macro(
            "AnswerVsStrongestUpperPoints",
            f"{100.0 * float(answer_comparison['bootstrap_95_interval'][1]):.4f}",
        ),
        macro(
            "UpstreamMaximumDifference",
            f"{float(parity['maximum_absolute_difference']):.8f}",
        ),
        macro(
            "UpstreamReproductionDifference",
            f"{float(parity['maximum_upstream_semantics_reproduction_absolute_difference']):.2e}",
        ),
    ]
    (BUNDLE / "results_macros.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def candidate_label(value: str) -> str:
    labels = {
        "clap_0p5s": "C-0.5",
        "clap_1s": "C-1",
        "clap_2s": "C-2",
        "clap_4s": "C-4",
        "clap_multiscale": "C-MS",
        "qcr_0p5s": "Q-0.5",
        "qcr_1s": "Q-1",
        "qcr_2s": "Q-2",
        "qcr_4s": "Q-4",
        "qcr_multiscale": "Q-MS",
    }
    return labels[value]


def tex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def write_supplement_tables() -> None:
    report = json.loads(ROUTER_REPORT.read_text(encoding="utf-8"))
    natural = json.loads(NATURAL_REPORT.read_text(encoding="utf-8"))
    labels = list(natural["label_names"])
    selection_lines = [
        r"\begin{table*}[t]",
        r"\caption{Fold-by-class selections. C denotes CLAP, Q denotes QCR, and MS denotes pooled multiscale.}",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Class & Candidate family & Fold 0 & Fold 1 & Fold 2 & Fold 3 & Fold 4\\",
        r"\midrule",
    ]
    for label in labels:
        name = tex_escape(str(natural["label_names"][label]))
        for route, route_label in (
            ("all_candidates", "CLAP+QCR"),
            ("clap_only", "CLAP-only"),
        ):
            selected = [
                candidate_label(
                    str(report["selection"][str(fold)][label][route])
                )
                for fold in range(5)
            ]
            selection_lines.append(
                "{} & {} & {}\\\\".format(
                    name,
                    route_label,
                    " & ".join(selected),
                )
            )
    selection_lines.extend(
        [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    )
    (BUNDLE / "supplement_selection.tex").write_text(
        "\n".join(selection_lines) + "\n", encoding="utf-8"
    )

    fixed = report["primary"]["comparator_metrics"]["class_mean_ap"]
    clap = report["clap_only_ablation"]["metrics"]["class_mean_ap"]
    full = report["primary"]["router_metrics"]["class_mean_ap"]
    class_lines = [
        r"\begin{table*}[t]",
        r"\caption{Class mean AP and cross-fit decomposition.}",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Class & Fixed CLAP & CLAP-only & Full & Full$-$fixed & Full$-$CLAP-only\\",
        r"\midrule",
    ]
    for label in labels:
        class_lines.append(
            "{} & {:.6f} & {:.6f} & {:.6f} & {:.6f} & {:.6f}\\\\".format(
                tex_escape(str(natural["label_names"][label])),
                float(fixed[label]),
                float(clap[label]),
                float(full[label]),
                float(full[label]) - float(fixed[label]),
                float(full[label]) - float(clap[label]),
            )
        )
    class_lines.extend(
        [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    )
    (BUNDLE / "supplement_classes.tex").write_text(
        "\n".join(class_lines) + "\n", encoding="utf-8"
    )


def copy_figures(analysis: dict[str, Any]) -> None:
    for name in ("figures", "extended_figures", "figure_source_data"):
        (BUNDLE / name).mkdir(parents=True)
    main_ids = {"AEF01", "AEF02", "AEF03", "AEF04"}
    records = []
    for item in analysis["figures"]:
        identifier = str(item["id"])
        source_pdf = TRACK / str(item["pdf"])
        if sha256(source_pdf) != item["pdf_sha256"]:
            raise RuntimeError(f"publication figure changed: {identifier}")
        shutil.copy2(
            source_pdf, BUNDLE / "extended_figures" / f"{identifier}.pdf"
        )
        if identifier in main_ids:
            shutil.copy2(
                source_pdf, BUNDLE / "figures" / f"{identifier}.pdf"
            )
        for filename, expected in item["source_sha256"].items():
            source = (
                TRACK
                / "publication_upgrade/figure_source_data"
                / str(filename)
            )
            if sha256(source) != expected:
                raise RuntimeError(f"figure source changed: {filename}")
            target = BUNDLE / "figure_source_data" / str(filename)
            if not target.exists():
                shutil.copy2(source, target)
        records.append(
            {
                "id": identifier,
                "title": item["title"],
                "main_paper": identifier in main_ids,
                "pdf_sha256": item["pdf_sha256"],
                "source_files": ";".join(item["source_files"]),
                "source_sha256": ";".join(
                    f"{name}:{value}"
                    for name, value in sorted(item["source_sha256"].items())
                ),
            }
        )
    with (BUNDLE / "figure_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def expand_files(patterns: tuple[str, ...]) -> list[Path]:
    files: set[Path] = set()
    for pattern in patterns:
        files.update(path for path in PROJECT.glob(pattern) if path.is_file())
    return sorted(files, key=lambda path: path.relative_to(PROJECT).as_posix())


def copy_reproducibility() -> dict[str, Any]:
    destination = BUNDLE / "reproducibility"
    destination.mkdir()
    patterns = (
        "q1_top_tier/*.json",
        "q1_top_tier/*.md",
        "q1_top_tier/configs/*.json",
        "q1_top_tier/src/*.py",
        "q1_top_tier/scripts/*.py",
        "q1_top_tier/tests/*.py",
        "q1_top_tier/provenance/*.json",
        "q1_top_tier/results/perception_test/precompute/index.jsonl",
        "q1_top_tier/results/perception_test/precompute/precompute_summary.json",
        "q1_top_tier/results/perception_test/precompute/query_embeddings.npz",
        "q1_top_tier/results/perception_test/evaluation/*.json",
        "q1_top_tier/results/perception_test/evaluation/*.jsonl.gz",
        "q1_top_tier/results/perception_test/evaluation/predictions/*.jsonl.gz",
        "q1_top_tier/results/perception_test/actionformer_reference/*",
        "q1_top_tier/publication/*.json",
        "q1_plus/*.json",
        "q1_plus/ANSWER_DEVELOPMENT_PROTOCOL.md",
        "q1_plus/EVENT_CONFIRMATORY_PROTOCOL.md",
        "q1_plus/EVENT_RANKER_SELECTION_PROTOCOL.md",
        "q1_plus/HELDOUT_ANSWER_EVALUATION_PROTOCOL.md",
        "q1_plus/HELDOUT_FIGURE_PROTOCOL.md",
        "q1_plus/HELDOUT_UNCERTAINTY_SUPPLEMENT_PROTOCOL.md",
        "q1_plus/configs/*.json",
        "q1_plus/data/event_needle/*.json",
        "q1_plus/data/event_needle/*.jsonl",
        "q1_plus/results/confirmatory/event_needle_confirmatory/*.json",
        "q1_plus/results/confirmatory/event_needle_confirmatory/*.jsonl",
        "q1_plus/results/confirmatory/event_needle_confirmatory/*.jsonl.gz",
        "q1_plus/results/confirmatory/event_ranker/*.json",
        "q1_plus/results/confirmatory/event_ranker/*.jsonl.gz",
        "q1_plus/results/development/event_needle_development/*.json",
        "q1_plus/results/development/event_needle_development/*.jsonl",
        "q1_plus/results/development/event_needle_development/*.jsonl.gz",
        "q1_plus/results/development/event_needle_validation/*.json",
        "q1_plus/results/development/event_needle_validation/*.jsonl",
        "q1_plus/results/development/event_needle_validation/*.jsonl.gz",
        "q1_plus/results/development/event_ranker/*.json",
        "q1_plus/results/development/event_ranker/*.jsonl.gz",
        "q1_plus/results/development/event_ranker/checkpoints/*.pt",
        "q1_plus/results/development/answer_retrieval/*.json",
        "q1_plus/results/development/answer_retrieval/*.jsonl.gz",
        "q1_plus/results/development/answer_generation/*.json",
        "q1_plus/results/development/answer_generation/*.jsonl",
        "q1_plus/scripts/*.py",
        "q1_plus/tests/*.py",
        "q1_90/EXTERNAL_REPLICATION_PROTOCOL.md",
        "q1_90/MODEL_BLIND_SOURCE_COUNT_AMENDMENT.json",
        "q1_90/DATA_RIGHTS_AND_DISTRIBUTION.md",
        "q1_90/EXTERNAL_REPLICATION_AUTHORIZATION.json",
        "q1_90/configs/*.json",
        "q1_90/data/external_replication/*.json",
        "q1_90/data/external_replication/*.jsonl",
        "q1_90/results/external_replication/precompute/*.json",
        "q1_90/results/external_replication/precompute/*.jsonl",
        "q1_90/results/external_replication/precompute/*.jsonl.gz",
        "q1_90/results/external_replication/evaluation/*.json",
        "q1_90/results/external_replication/evaluation/*.jsonl.gz",
        "q1_90/scripts/*.py",
        "q1_90/tests/*.py",
        "q1_upgrade/*.md",
        "q1_upgrade/*.json",
        "q1_upgrade/configs/*.json",
        "q1_upgrade/src/*.py",
        "q1_upgrade/scripts/*.py",
        "q1_upgrade/tests/*.py",
        "q1_upgrade/results/scale_candidates/receipt.json",
        "q1_crossfit_capcorrect/*.md",
        "q1_crossfit_capcorrect/*.json",
        "q1_crossfit_capcorrect/configs/*.json",
        "q1_crossfit_capcorrect/src/*.py",
        "q1_crossfit_capcorrect/scripts/*.py",
        "q1_crossfit_capcorrect/tests/*.py",
        "q1_crossfit_capcorrect/results/candidate_pools/receipt.json",
        "q1_crossfit_capcorrect/results/capcorrect_router/*.json",
        "q1_crossfit_capcorrect/results/capcorrect_router/*.jsonl.gz",
        "q1_crossfit_capcorrect/publication_upgrade/EVIAUDIO_UPGRADE_ANALYSIS.json",
        "src/eviaudio_mt/*.py",
        "pyproject.toml",
        "requirements-minimal.txt",
        "requirements-full.txt",
        "requirements-mamba.txt",
    )
    included: dict[str, Any] = {}
    excluded_large: dict[str, Any] = {}
    sanitized_public_copies: dict[str, Any] = {}
    for source in expand_files(patterns):
        relative = source.relative_to(PROJECT)
        if source.stat().st_size > MAXIMUM_INCLUDED_BYTES:
            excluded_large[relative.as_posix()] = {
                "bytes": source.stat().st_size,
                "sha256": sha256(source),
                "reason": "larger than compact bundle threshold",
            }
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        transformation = copy_public_file(source, target)
        included_record: dict[str, Any] = {
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
        }
        if transformation is not None:
            included_record["source_bytes"] = transformation["source_bytes"]
            included_record["source_sha256"] = transformation["source_sha256"]
            included_record["public_copy_transformation"] = transformation[
                "transformation"
            ]
            sanitized_public_copies[relative.as_posix()] = transformation
        included[relative.as_posix()] = included_record

    pool_receipt = json.loads(
        (
            TRACK / "results/candidate_pools/receipt.json"
        ).read_text(encoding="utf-8")
    )
    excluded_pools = {
        f"q1_crossfit_capcorrect/results/candidate_pools/pools/{name}.jsonl.gz": {
            "rows": pool_receipt["pool_counts"][name],
            "sha256": pool_receipt["pool_sha256"][name],
            "included": False,
            "reason": "regenerable derived candidate pool",
        }
        for name in sorted(pool_receipt["pool_counts"])
    }
    payload = {
        "schema": "eviaudio_q1_upgrade_scientific_inventory_v3",
        "included_files": included,
        "included_file_count": len(included),
        "excluded_large_files": excluded_large,
        "excluded_candidate_pools": excluded_pools,
        "public_path_sanitization": {
            "status": "machine_local_paths_removed_from_public_copies",
            "scientific_values_changed": False,
            "sanitized_file_count": len(sanitized_public_copies),
            "files": sanitized_public_copies,
        },
        "third_party_audio_included": False,
        "model_weights_included": False,
        "exclusion_reason": (
            "Provider-controlled audio, large precompute archives, and "
            "regenerable candidate pools are not redistributed; revisions, "
            "receipts, hashes, recipes, and final predictions remain."
        ),
    }
    (BUNDLE / "SCIENTIFIC_ARTIFACT_INVENTORY.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (BUNDLE / "PUBLIC_PATH_SANITIZATION.md").write_text(
        """# Public-path sanitization

Some frozen runtime records originally contained machine-local absolute
paths. Public copies normalize project-root prefixes to repository-relative
paths and replace any remaining home prefix with `<LOCAL_HOME>`. Numerical
values, predictions, decisions, and scientific hashes are unchanged.

`SCIENTIFIC_ARTIFACT_INVENTORY.json` records the original and public SHA-256
values for every transformed file. The original source-tree records remain
untouched, so their hashes continue to match the frozen authorization and
lock documents.
""",
        encoding="utf-8",
    )
    verify_public_path_hygiene(BUNDLE)
    return payload


def write_editorial(analysis: dict[str, Any]) -> None:
    diagnostic = analysis["cap_corrected_diagnosis"]
    natural = analysis["frozen_natural_transfer"]
    (BUNDLE / "README.md").write_text(
        """# IEEE/ACM TASLP submission bundle

This is the upgraded, outcome-faithful bundle for the EviAudio evidence
ranking and temporal localization study. It includes the manuscript,
supplement, source-linked figures, compact reproducibility artifacts, final
cross-fit predictions, and deterministic manifest.

The frozen natural zero-shot result remains a failure. The positive
cap-corrected router is explicitly post-outcome and diagnostic. Accountable
authors must complete identity, legal, contribution, and disclosure metadata
before submission. Targeting a Q1 venue does not guarantee acceptance.
""",
        encoding="utf-8",
    )
    (BUNDLE / "key_contributions.txt").write_text(
        "\n".join(
            [
                "Controlled QCR ranking is confirmed on a source-isolated panel.",
                "Frozen QCR fails zero-shot transfer on 5,359 natural recordings.",
                "Cap-corrected cross-fitting identifies temporal scale as dominant.",
                "The learned retrieval does not improve held-out answer accuracy.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (BUNDLE / "cover_letter.md").write_text(
        f"""Dear Editor-in-Chief,

Please consider “Where Query-Conditioned Audio Ranking Transfers—and Where It
Fails: Controlled Confirmation, Frozen Natural Evaluation, and Cap-Corrected
Diagnosis” as a Regular Paper for IEEE/ACM Transactions on Audio, Speech, and
Language Processing.

The paper presents a complete controlled-to-natural evaluation of a bounded
CLAP residual ranker. It retains the frozen natural failure
({natural['qcr_minus_clap_map']:.6f} mAP change) and adds a cap-corrected,
five-fold diagnostic. The full router reaches
{diagnostic['full_router_map']:.6f} mAP, but a CLAP-only router reaches
{diagnostic['clap_only_router_map']:.6f}; temporal scale therefore explains
most of the exploratory gain. A downstream answer bridge also fails.

The bundle includes freezes, invalidations, exact output-cap reconstruction,
prediction hashes, source-linked figures, tests, and independent verification.
The post-outcome result is never described as zero-shot or fresh
confirmation. Originality and concurrent-submission declarations require
final approval by the accountable authors.

Sincerely,

Corresponding author metadata required
""",
        encoding="utf-8",
    )
    (BUNDLE / "author_metadata_required.md").write_text(
        """# Accountable-author metadata required

Before submission, supply and approve names, order, affiliations, ORCIDs,
corresponding email, CRediT roles, funding, conflicts, acknowledgments,
copyright/data/model rights, journal-required AI-assistance disclosure,
originality declarations, and final proofreading. No identity or legal
declaration has been fabricated.
""",
        encoding="utf-8",
    )
    (BUNDLE / "data_availability.md").write_text(
        """# Data and code availability

The bundle includes protocols, recipes/indices where distributable, source,
tests, normalized results, natural and final router predictions, receipts,
checksums, and figure source data. Perception Test audio, CLAP weights,
official ActionFormer features/checkpoint, UrbanSound8K, ESC-50, LibriSpeech,
embedding archives, and large candidate pools are not redistributed.

Provider URLs, revisions, official archive checksums, source IDs, candidate
pool hashes, and acquisition commands are retained. UrbanSound8K and ESC-50
carry non-commercial attribution restrictions; users must comply with every
provider's terms.
""",
        encoding="utf-8",
    )
    shutil.copy2(
        PROJECT / "submission_taslpro_q1_90/edics_recommendation.md",
        BUNDLE / "edics_recommendation.md",
    )
    (BUNDLE / "venue_fit_and_claims.md").write_text(
        """# Venue fit and claim boundary

The work addresses language–audio representation, temporal sound
localization, multiscale audio analysis, transfer evaluation, and
reproducible signal-processing methodology.

Supported:

- controlled exact-onset QCR evidence-ranking gain;
- retained failure of fixed zero-shot natural transfer;
- post-outcome in-domain evidence that class-dependent scale dominates;
- a smaller positive incremental QCR effect within that diagnosis;
- no demonstrated held-out answer benefit.

Not supported:

- fresh confirmation of the cross-fitted router;
- general open-vocabulary or supervised state-of-the-art performance;
- a downstream question-answering gain;
- a guarantee of journal acceptance.
""",
        encoding="utf-8",
    )
    (BUNDLE / "venue_requirements_checked.md").write_text(
        """# Venue requirements checked

Target: IEEE/ACM Transactions on Audio, Speech, and Language Processing.

The IEEE Signal Processing Society Information for Authors page was checked
on 2026-07-24. It states that an initial Regular Paper may not exceed 13
double-column 10-point pages including references; supplemental material is
recommended not to exceed six double-column pages. This bundle enforces both
guards. The page also notes mandatory overlength charges above ten published
pages. Accountable authors must recheck current policy, author-block effects,
and charges at submission:
https://signalprocessingsociety.org/publications-resources/information-authors

The journal page encourages release of information needed to reproduce
figures and tables:
https://signalprocessingsociety.org/publications-resources/ieee-transactions-audio-speech-and-language-processing/ieee-transactions
""",
        encoding="utf-8",
    )
    (BUNDLE / "reproducibility_checklist.md").write_text(
        """# Reproducibility checklist

- [x] Controlled, external, frozen natural, diagnostic, and downstream stages separated
- [x] Natural prompts, models, scales, metric, bootstrap, and gates frozen
- [x] All 5,359 validation IDs and the 35,625-annotation panel are checksum-bound
- [x] Provider-controlled natural metadata/audio are explicitly excluded by rights boundary
- [x] Ten candidate caps reconstructed exactly
- [x] Shared 200-segment cap applied after class routing
- [x] Invalid first router and analyzer efficiency amendment disclosed
- [x] Final full and CLAP-only predictions included
- [x] Large intermediate pools recorded by rows and SHA-256
- [x] Every figure has machine-readable source data
- [x] Main and supplement render within page guards
- [ ] Accountable-author metadata and final rights approval
""",
        encoding="utf-8",
    )
    (BUNDLE / "submission_checklist.md").write_text(
        """# Submission checklist

- [x] Abstract contains 150–250 words
- [x] Initial main manuscript is no longer than 13 double-column pages
- [x] Supplement is no longer than six double-column pages
- [x] Frozen failures and post-outcome status are prominent
- [x] Main figures/tables have source data and captions
- [x] Cover letter, EDICS, availability, claims, and manifests included
- [ ] Insert author identity and approve disclosure/legal language
- [ ] Re-render after final author block and perform final human proofread
""",
        encoding="utf-8",
    )
    (BUNDLE / "figure_captions.md").write_text(
        """# Figure captions

- AEF01: Outcome-faithful evidence ladder.
- AEF02: Controlled evidence AP and natural temporal mAP.
- AEF03: Natural class-macro mAP across official tIoU thresholds.
- AEF04: Fold-by-class routing and class-effect decomposition.
- AEF05: Held-out answer-bridge accuracy (supplement).
""",
        encoding="utf-8",
    )
    (BUNDLE / "reproduction_commands.md").write_text(
        """# Reproduction commands

Compact verification:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q \
  -p no:cacheprovider reproducibility/q1_crossfit_capcorrect/tests
```

After restoring provider-controlled audio, weights, and precompute archives,
the scientific sequence is:

```bash
python q1_plus/scripts/evaluate_event_ranker_confirmatory.py
python q1_90/scripts/evaluate_external_replication.py
python q1_top_tier/scripts/evaluate_perception_test.py
python q1_crossfit_capcorrect/scripts/build_candidate_pools.py
python q1_crossfit_capcorrect/scripts/analyze_capcorrect_router.py
python q1_crossfit_capcorrect/scripts/build_upgrade_publication.py
python q1_crossfit_capcorrect/scripts/build_upgrade_submission.py
```

Immutable analyzers refuse to overwrite existing reports; use a clean
scientific workspace for a full rerun.
""",
        encoding="utf-8",
    )
    base_pins = json.loads(
        (
            PROJECT / "submission_taslp_top_tier/model_and_data_pins.json"
        ).read_text(encoding="utf-8")
    )
    base_pins["controlled_lock_sha256"] = sha256(
        PROJECT / "q1_plus/EVENT_RANKER_LOCK.json"
    )
    base_pins["external_authorization_sha256"] = sha256(
        PROJECT / "q1_90/EXTERNAL_REPLICATION_AUTHORIZATION.json"
    )
    base_pins["capcorrect_lock_sha256"] = sha256(
        TRACK / "CAPCORRECT_ANALYSIS_LOCK.json"
    )
    base_pins["capcorrect_analyzer_amendment_sha256"] = sha256(
        TRACK / "CAPCORRECT_ANALYZER_AMENDMENT_LOCK.json"
    )
    base_pins["candidate_pool_receipt_sha256"] = sha256(
        TRACK / "results/candidate_pools/receipt.json"
    )
    base_pins["candidate_pools_redistributed"] = False
    (BUNDLE / "model_and_data_pins.json").write_text(
        json.dumps(base_pins, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    claim_ledger = {
        "schema": "eviaudio_q1_upgrade_claim_ledger_v2",
        "controlled_confirmatory_status": analysis[
            "controlled_confirmatory"
        ]["status"],
        "external_exact_onset_status": analysis["external_exact_onset"][
            "status"
        ],
        "frozen_natural_status": natural["status"],
        "cap_corrected_status": diagnostic["status"],
        "cap_corrected_claim_type": diagnostic["claim_type"],
        "downstream_status": analysis["downstream_answer_bridge"]["status"],
        "supported_headline": (
            "QCR improves controlled evidence ranking but fails frozen "
            "natural zero-shot transfer; post-outcome diagnosis identifies "
            "class-dependent scale as the dominant recoverable mechanism."
        ),
        "fresh_crossfit_confirmation": False,
        "q1_acceptance_guaranteed": False,
    }
    (BUNDLE / "CLAIM_LEDGER.json").write_text(
        json.dumps(claim_ledger, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def render(document: str) -> tuple[int, str]:
    environment = os.environ.copy()
    environment["SOURCE_DATE_EPOCH"] = SOURCE_DATE_EPOCH
    environment["TZ"] = "UTC"
    output = command(
        [str(TECTONIC), "--outdir", str(BUNDLE / "rendered"), document],
        cwd=BUNDLE,
        environment=environment,
    )
    output = re.sub(
        r'note: Rerunning TeX because "[^"]+" changed \.\.\.',
        "note: Rerunning TeX because an intermediate changed ...",
        output,
    ).replace(str(BUNDLE), ".")
    stem = Path(document).stem
    (BUNDLE / "rendered" / f"{stem}.build.log").write_text(
        output, encoding="utf-8"
    )
    pdf = BUNDLE / "rendered" / f"{stem}.pdf"
    if not pdf.is_file() or not pdf.read_bytes().startswith(b"%PDF"):
        raise RuntimeError(f"rendered PDF is invalid: {pdf}")
    return pdf_pages(pdf), sha256(pdf)


def validate_sources() -> dict[str, Any]:
    main = (BUNDLE / "main.tex").read_text(encoding="utf-8")
    words = abstract_words(main)
    figures = len(re.findall(r"\\begin\{figure\*?\}", main))
    tables = len(re.findall(r"\\begin\{table\*?\}", main))
    if not 150 <= words <= 250:
        raise RuntimeError(f"abstract has {words} words; required 150–250")
    if figures != 4 or tables != 2:
        raise RuntimeError(
            f"unexpected main objects: {figures} figures, {tables} tables"
        )
    for path in BUNDLE.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".tex", ".md", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            if re.search(r"\b(?:TODO|TBD|FIXME|PLACEHOLDER)\b", text, re.I):
                raise RuntimeError(f"editorial marker remains: {path}")
    return {
        "abstract_words": words,
        "main_figures": figures,
        "main_tables": tables,
    }


def finalize_manifest(validation: dict[str, Any]) -> None:
    path = BUNDLE / "SUBMISSION_MANIFEST.json"
    files = {
        item.relative_to(BUNDLE).as_posix(): {
            "bytes": item.stat().st_size,
            "sha256": sha256(item),
        }
        for item in sorted(BUNDLE.rglob("*"))
        if item.is_file() and item != path
    }
    payload = {
        "schema": "eviaudio_taslp_q1_upgrade_bundle_v2",
        "target": (
            "IEEE/ACM Transactions on Audio, Speech, and Language Processing"
        ),
        "status": "technically_ready_author_metadata_pending",
        "frozen_natural_failure_preserved": True,
        "post_outcome_diagnostic_disclosed": True,
        "validation": validation,
        "files": files,
        "file_count": len(files),
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_archive() -> str:
    partial = ARCHIVE.with_suffix(".partial")
    if partial.exists():
        partial.unlink()
    with zipfile.ZipFile(
        partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(BUNDLE.rglob("*")):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(
                (Path(BUNDLE.name) / path.relative_to(BUNDLE)).as_posix()
            )
            info.date_time = (2026, 7, 24, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)
    partial.replace(ARCHIVE)
    with zipfile.ZipFile(ARCHIVE) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("submission archive CRC failure")
        expected = {
            (Path(BUNDLE.name) / path.relative_to(BUNDLE)).as_posix()
            for path in BUNDLE.rglob("*")
            if path.is_file()
        }
        if set(archive.namelist()) != expected:
            raise RuntimeError("submission archive inventory mismatch")
        with tempfile.TemporaryDirectory(prefix="eviaudio-q1-upgrade-") as name:
            archive.extractall(name)
            for relative in expected:
                source = BUNDLE / Path(relative).relative_to(BUNDLE.name)
                target = Path(name) / relative
                if sha256(source) != sha256(target):
                    raise RuntimeError(f"archive extraction mismatch: {relative}")
    return sha256(ARCHIVE)


def main() -> None:
    if not TECTONIC.is_file():
        raise RuntimeError("pinned Tectonic binary is absent")
    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    if analysis["status"] != "eviaudio_multistage_publication_analysis_complete":
        raise RuntimeError("publication analysis is incomplete")
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir()
    shutil.copy2(TEMPLATES / "main.tex", BUNDLE / "main.tex")
    shutil.copy2(TEMPLATES / "supplement.tex", BUNDLE / "supplement.tex")
    write_macros(analysis)
    write_supplement_tables()
    copy_figures(analysis)
    write_editorial(analysis)
    inventory = copy_reproducibility()
    source_validation = validate_sources()
    (BUNDLE / "rendered").mkdir()
    main_pages, main_sha = render("main.tex")
    supplement_pages, supplement_sha = render("supplement.tex")
    if main_pages > 13:
        raise RuntimeError(f"main manuscript exceeds 13 pages: {main_pages}")
    if supplement_pages > 6:
        raise RuntimeError(
            f"supplement exceeds six-page recommendation: {supplement_pages}"
        )
    main_text = command(
        ["pdftotext", str(BUNDLE / "rendered/main.pdf"), "-"], cwd=BUNDLE
    )
    normalized = re.sub(r"\s+", " ", main_text)
    required = (
        "fails zero-shot transfer",
        "post-outcome",
        "class-dependent temporal scale",
        "does not improve held-out answers",
    )
    if any(value not in normalized for value in required):
        raise RuntimeError("rendered main manuscript lost required boundary text")
    validation = {
        **source_validation,
        "main_pages": main_pages,
        "supplement_pages": supplement_pages,
        "main_pdf_sha256": main_sha,
        "supplement_pdf_sha256": supplement_sha,
        "included_scientific_files": inventory["included_file_count"],
        "excluded_candidate_pools": len(
            inventory["excluded_candidate_pools"]
        ),
        "q1_acceptance_guaranteed": False,
    }
    finalize_manifest(validation)
    archive_sha = build_archive()
    print(
        json.dumps(
            {
                "status": "eviaudio_q1_upgrade_submission_complete",
                "bundle": str(BUNDLE),
                "archive": str(ARCHIVE),
                "archive_bytes": ARCHIVE.stat().st_size,
                "archive_sha256": archive_sha,
                **validation,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
