#!/usr/bin/env python3
"""Build the outcome-independent, source-linked held-out answer figure suite."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_answer_heldout as analysis
import run_answer_calibration as base


PROJECT = Path(__file__).resolve().parents[2]
Q1 = PROJECT / "q1_plus"
RESULTS = Q1 / "results/development/answer_generation"
REPORT = RESULTS / "heldout_evaluation_report.json"
UNCERTAINTY = RESULTS / "heldout_uncertainty_supplement.json"
CALIBRATION = RESULTS / "calibration_selection_report.json"
CONFIG = Q1 / "configs/answer_heldout.json"
AUTHORIZATION = Q1 / "ANSWER_HELDOUT_AUTHORIZATION.json"
PROTOCOL = Q1 / "HELDOUT_FIGURE_PROTOCOL.md"
REPORTING_LOCK = Q1 / "HELDOUT_REPORTING_PLAN_LOCK.json"
REPORTING_AMENDMENT_001 = Q1 / "HELDOUT_REPORTING_AMENDMENT_001.json"
REPORTING_AMENDMENT_002 = Q1 / "HELDOUT_REPORTING_AMENDMENT_002.json"
OUT = Q1 / "figures_heldout_answer"
PDF, PNG, SOURCE = OUT / "pdf", OUT / "png", OUT / "source_data"
INDEX, AUDIT = OUT / "figure_index.csv", OUT / "figure_audit.json"
MODELS = ("qwen2_audio", "phi4_multimodal")
EXPECTED_FIGURE_IDS = tuple(f"A{index:02d}" for index in range(1, 41))
MODEL_LABELS = {"qwen2_audio": "Qwen2-Audio", "phi4_multimodal": "Phi-4-MM"}
SYSTEM_LABELS = {
    "selected_learned_retrieval": "Learned",
    "clap_retrieval": "CLAP",
    "deterministic_random_retrieval": "Random",
    "uniform_retrieval": "Uniform",
    "prefix_matched_retrieval": "Prefix",
    "oracle_retrieval": "Oracle",
    "selected_retrieval_silenced": "Silence",
    "text_only": "Text only",
}
BLUE, ORANGE, GREEN, RED, PURPLE, GREY = (
    "#2166AC", "#D6604D", "#1B9E77", "#B2182B", "#7570B3", "#666666"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
        }
    )
    for directory in (PDF, PNG, SOURCE):
        directory.mkdir(parents=True, exist_ok=True)


class Builder:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def save(
        self,
        identifier: str,
        slug: str,
        title: str,
        rows: list[dict[str, Any]],
        fig: plt.Figure,
        section: str,
        caption: str | None = None,
    ) -> None:
        if not rows:
            raise RuntimeError(f"figure {identifier} has no source rows")
        stem = f"{identifier}_{slug}"
        csv_path = SOURCE / f"{stem}.csv"
        pdf_path = PDF / f"{stem}.pdf"
        png_path = PNG / f"{stem}.png"
        fields = sorted({key for row in rows for key in row})
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, bbox_inches="tight")
        plt.close(fig)
        self.records.append(
            {
                "id": identifier,
                "title": title,
                "caption": caption or f"Frozen held-out analysis: {title}.",
                "section": section,
                "pdf": str(pdf_path.relative_to(PROJECT)),
                "png": str(png_path.relative_to(PROJECT)),
                "source_data": str(csv_path.relative_to(PROJECT)),
                "source_rows": len(rows),
                "pdf_sha256": sha256(pdf_path),
                "png_sha256": sha256(png_path),
                "source_sha256": sha256(csv_path),
            }
        )


def bar(
    rows: list[dict[str, Any]],
    category: str,
    value: str,
    title: str,
    ylabel: str,
    *,
    horizontal: bool = False,
    reference: float | None = None,
    colors: list[str] | None = None,
) -> plt.Figure:
    size = (
        (6.8, max(3.5, 0.25 * len(rows)))
        if horizontal
        else (max(6.8, 0.32 * len(rows)), 3.8)
    )
    fig, ax = plt.subplots(figsize=size)
    names = [str(row[category]) for row in rows]
    values = [float(row[value]) for row in rows]
    palette = colors or [BLUE] * len(rows)
    positions = np.arange(len(rows))
    if horizontal:
        ax.barh(positions, values, color=palette)
        ax.set_yticks(positions, names)
        ax.invert_yaxis()
        ax.set_xlabel(ylabel)
        if reference is not None:
            ax.axvline(reference, color="black", linestyle="--", linewidth=1)
    else:
        ax.bar(positions, values, color=palette)
        ax.set_xticks(positions, names, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        if reference is not None:
            ax.axhline(reference, color="black", linestyle="--", linewidth=1)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="x" if horizontal else "y", alpha=0.2)
    return fig


def grouped(
    rows: list[dict[str, Any]],
    group: str,
    series: str,
    value: str,
    title: str,
    ylabel: str,
    *,
    legend_columns: int = 1,
) -> plt.Figure:
    groups = list(dict.fromkeys(str(row[group]) for row in rows))
    names = list(dict.fromkeys(str(row[series]) for row in rows))
    lookup = {(str(row[group]), str(row[series])): float(row[value]) for row in rows}
    fig, ax = plt.subplots(figsize=(max(6.8, 0.75 * len(groups)), 4.0))
    positions = np.arange(len(groups))
    width = 0.78 / len(names)
    palette = [BLUE, ORANGE, GREEN, PURPLE, GREY, RED, "#A6761D", "#1F78B4"]
    for index, name in enumerate(names):
        values = [lookup.get((group_name, name), np.nan) for group_name in groups]
        ax.bar(
            positions + (index - (len(names) - 1) / 2) * width,
            values,
            width,
            label=name,
            color=palette[index % len(palette)],
        )
    ax.set_xticks(positions, groups, rotation=24, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=legend_columns)
    return fig


def forest(
    rows: list[dict[str, Any]], title: str, xlabel: str, *, key: str = "label"
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.0, max(3.6, 0.36 * len(rows))))
    for index, row in enumerate(rows):
        estimate = float(row["estimate"])
        lower = float(row["lower"])
        upper = float(row["upper"])
        ax.errorbar(
            estimate,
            index,
            xerr=[[estimate - lower], [upper - estimate]],
            fmt="o",
            color=BLUE,
            capsize=3,
        )
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(range(len(rows)), [str(row[key]) for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    return fig


def source_values(rows: list[dict[str, Any]], system: str, metric: str) -> dict[str, float]:
    return analysis.source_values(rows, system, metric)


def combined_source_values(
    rows_by_model: dict[str, list[dict[str, Any]]], system: str, metric: str
) -> dict[str, float]:
    return analysis.combined_source_values(rows_by_model, system, metric)


def source_metric_rows(
    rows_by_model: dict[str, list[dict[str, Any]]], system: str, metric: str
) -> list[dict[str, Any]]:
    values = {
        model: source_values(rows_by_model[model], system, metric) for model in MODELS
    }
    combined = combined_source_values(rows_by_model, system, metric)
    labels = {
        source: f"S{index:02d}"
        for index, source in enumerate(sorted(combined), start=1)
    }
    return [
        {
            "source": source,
            "source_label": labels[source],
            "backbone": label,
            "value": value,
        }
        for source in sorted(combined)
        for label, value in (
            (MODEL_LABELS[MODELS[0]], values[MODELS[0]][source]),
            (MODEL_LABELS[MODELS[1]], values[MODELS[1]][source]),
            ("Two-backbone mean", combined[source]),
        )
    ]


def source_delta_rows(
    rows_by_model: dict[str, list[dict[str, Any]]], comparator: str, metric: str = "exact"
) -> list[dict[str, Any]]:
    learned = combined_source_values(
        rows_by_model, "selected_learned_retrieval", metric
    )
    baseline = combined_source_values(rows_by_model, comparator, metric)
    labels = {
        source: f"S{index:02d}"
        for index, source in enumerate(sorted(learned), start=1)
    }
    return sorted(
        [
            {
                "source": source,
                "source_label": labels[source],
                "learned": learned[source],
                "comparator": baseline[source],
                "delta": learned[source] - baseline[source],
                "comparator_system": comparator,
            }
            for source in learned
        ],
        key=lambda row: (row["delta"], row["source"]),
    )


def breakdown_rows(
    report: dict[str, Any], dimension: str, metric: str
) -> list[dict[str, Any]]:
    rows = []
    for model in MODELS:
        groups = report["breakdowns"][model]["selected_learned_retrieval"][dimension]
        for group, values in groups.items():
            rows.append(
                {
                    "group": group,
                    "backbone": MODEL_LABELS[model],
                    "value": values[metric],
                    "examples": values["examples"],
                    "sources": values["sources"],
                }
            )
    return rows


def transition_rows(
    rows_by_model: dict[str, list[dict[str, Any]]], comparator: str
) -> list[dict[str, Any]]:
    result = []
    for model in MODELS:
        lookup = {
            (row["system"], row["example_id"]): row for row in rows_by_model[model]
        }
        counts: Counter[tuple[int, int]] = Counter()
        for row in rows_by_model[model]:
            if row["system"] != "selected_learned_retrieval":
                continue
            counts[
                (
                    int(lookup[(comparator, row["example_id"])]["exact"]),
                    int(row["exact"]),
                )
            ] += 1
        for baseline_value in (0, 1):
            for learned_value in (0, 1):
                result.append(
                    {
                        "backbone": MODEL_LABELS[model],
                        "transition": f"{baseline_value}→{learned_value}",
                        "examples": counts[(baseline_value, learned_value)],
                        "comparator": comparator,
                    }
                )
    return result


def main() -> None:
    if INDEX.exists() or AUDIT.exists():
        raise FileExistsError("held-out answer figure suite is immutable once indexed")
    setup()
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    uncertainty = json.loads(UNCERTAINTY.read_text(encoding="utf-8"))
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if report["status"] not in {
        "heldout_answer_gate_pass_audita_pipeline_lock_eligible",
        "heldout_answer_gate_fail_audita_sealed",
    }:
        raise RuntimeError("held-out report status is not final")
    if uncertainty["status"] != "heldout_source_clustered_uncertainty_complete_non_gating":
        raise RuntimeError("held-out uncertainty supplement is missing")
    if uncertainty["main_report_sha256"] != sha256(REPORT):
        raise RuntimeError("uncertainty supplement does not bind the main report")

    rows_by_model = {
        model: base.read_jsonl(RESULTS / f"heldout__{model}.jsonl") for model in MODELS
    }
    expected_rows = int(config["expected_examples"]) * len(config["systems"])
    artifact_hashes = {item["model"]: item["raw_sha256"] for item in report["artifacts"]}
    for model, rows in rows_by_model.items():
        raw_path = RESULTS / f"heldout__{model}.jsonl"
        if (
            len(rows) != expected_rows
            or len({row["job_id"] for row in rows}) != expected_rows
            or artifact_hashes[model] != sha256(raw_path)
            or any(int(row["audita_rows_accessed"]) != 0 for row in rows)
        ):
            raise RuntimeError(f"held-out figure input audit failed: {model}")

    systems = list(config["systems"])
    strongest = str(report["strongest_eligible_nonoracle_baseline"])
    builder = Builder()

    protocol_rows = [
        {"stage": "24-source calibration", "order": 1, "status": "complete"},
        {"stage": "25-source held-out development", "order": 2, "status": "complete"},
        {
            "stage": "Frozen promotion gates",
            "order": 3,
            "status": "pass" if report["all_promotion_gates_pass"] else "fail",
        },
        {
            "stage": "AUDITA",
            "order": 4,
            "status": "still sealed",
        },
    ]
    fig, ax = plt.subplots(figsize=(7.3, 2.5))
    ax.plot([1, 2, 3, 4], [0, 0, 0, 0], color=GREY, linewidth=2)
    for row in protocol_rows:
        color = GREEN if row["status"] in {"complete", "pass"} else RED if row["status"] == "fail" else GREY
        ax.scatter(row["order"], 0, s=160, color=color)
        ax.text(row["order"], 0.11, row["stage"], ha="center", fontsize=8)
        ax.text(row["order"], -0.10, row["status"], ha="center", fontsize=8, color=color)
    ax.set(yticks=[], xticks=[], ylim=(-0.20, 0.30))
    ax.set_title("Leakage-controlled answer-evaluation path", loc="left", fontweight="bold")
    builder.save("A01", "protocol", "Held-out answer protocol", protocol_rows, fig, "protocol")

    for identifier, metric, title in (
        ("A02", "two_backbone_mean_source_macro_exact", "Held-out exact match across systems"),
        ("A03", "two_backbone_mean_source_macro_token_f1", "Held-out token F1 across systems"),
    ):
        rows = [
            {
                "system": SYSTEM_LABELS[system],
                "system_key": system,
                "value": report["combined_metrics"][system][metric],
            }
            for system in systems
        ]
        colors = [GREEN if row["system_key"] == "selected_learned_retrieval" else PURPLE if row["system_key"] == "oracle_retrieval" else GREY for row in rows]
        builder.save(identifier, metric.replace("two_backbone_mean_source_macro_", "combined_"), title, rows, bar(rows, "system", "value", title, "Target-source macro score", colors=colors), "primary results")

    for identifier, metric, title in (
        ("A04", "source_macro_exact", "Exact match by backbone and system"),
        ("A05", "source_macro_token_f1", "Token F1 by backbone and system"),
    ):
        rows = [
            {"system": SYSTEM_LABELS[system], "backbone": MODEL_LABELS[model], "value": report["metrics"][model][system][metric]}
            for system in systems
            for model in MODELS
        ]
        builder.save(identifier, metric, title, rows, grouped(rows, "system", "backbone", "value", title, "Target-source macro score"), "primary results")

    exact_ci_rows = []
    for system in systems:
        if system == "selected_learned_retrieval":
            continue
        interval = report["comparisons"][system]["combined_exact"]
        exact_ci_rows.append({"label": SYSTEM_LABELS[system], "system": system, "estimate": interval["delta"], "lower": interval["bootstrap_95_interval"][0], "upper": interval["bootstrap_95_interval"][1], "sources": interval["sources"]})
    builder.save("A06", "exact_difference_forest", "Learned-retrieval exact-match differences", exact_ci_rows, forest(exact_ci_rows, "Learned minus comparator: exact match", "Paired target-source difference (95% bootstrap interval)"), "uncertainty")

    f1_ci_rows = []
    for system in systems:
        if system == "selected_learned_retrieval":
            continue
        interval = uncertainty["learned_minus_comparator_token_f1"][system]["two_backbone_mean"]
        f1_ci_rows.append({"label": SYSTEM_LABELS[system], "system": system, "estimate": interval["delta"], "lower": interval["bootstrap_95_interval"][0], "upper": interval["bootstrap_95_interval"][1], "sources": interval["sources"]})
    builder.save("A07", "f1_difference_forest", "Learned-retrieval token-F1 differences", f1_ci_rows, forest(f1_ci_rows, "Learned minus comparator: token F1", "Paired target-source difference (95% bootstrap interval)"), "uncertainty")

    response_rows = [
        {"backbone": MODEL_LABELS[model], "control": SYSTEM_LABELS[control], "fraction": report["response_difference_fractions"][model][control]}
        for model in MODELS
        for control in ("selected_retrieval_silenced", "text_only")
    ]
    builder.save("A08", "response_differences", "Response sensitivity to audio controls", response_rows, grouped(response_rows, "control", "backbone", "fraction", "Normalized responses changed under controls", "Fraction of examples"), "controls")

    gate_rows = [{"gate": key.replace("_", " "), "passed": int(value)} for key, value in report["promotion_gates"].items()]
    builder.save("A09", "promotion_gates", "Frozen held-out promotion gates", gate_rows, bar(gate_rows, "gate", "passed", "Frozen held-out promotion gates", "Passed", horizontal=True, colors=[GREEN if row["passed"] else RED for row in gate_rows]), "primary results")

    calibration_rows = []
    selected = calibration["selected_metrics"]
    for model in MODELS:
        for metric, label in (("source_macro_exact", "Exact"), ("source_macro_token_f1", "Token F1")):
            calibration_rows.extend(
                [
                    {"backbone_metric": f"{MODEL_LABELS[model]} · {label}", "stage": "Calibration", "value": selected["per_backbone"][model][metric]},
                    {"backbone_metric": f"{MODEL_LABELS[model]} · {label}", "stage": "Held-out", "value": report["metrics"][model]["selected_learned_retrieval"][metric]},
                ]
            )
    builder.save("A10", "calibration_transfer", "Calibration-to-held-out transfer", calibration_rows, grouped(calibration_rows, "backbone_metric", "stage", "value", "Selected prompt/K transfer", "Target-source macro score"), "generalization")

    for identifier, metric, title in (
        ("A11", "exact", "Selected-system exact match by target source"),
        ("A12", "token_f1", "Selected-system token F1 by target source"),
    ):
        rows = source_metric_rows(rows_by_model, "selected_learned_retrieval", metric)
        builder.save(identifier, f"source_{metric}", title, rows, grouped(rows, "source_label", "backbone", "value", title, "Source-macro score", legend_columns=3), "source robustness")

    for identifier, comparator, title in (
        ("A13", strongest, "Source-level exact difference to strongest baseline"),
        ("A14", "deterministic_random_retrieval", "Source-level exact difference to deterministic random"),
    ):
        rows = source_delta_rows(rows_by_model, comparator)
        builder.save(identifier, f"source_delta_{comparator}", title, rows, bar(rows, "source_label", "delta", title, "Learned − comparator", horizontal=True, reference=0, colors=[GREEN if row["delta"] >= 0 else RED for row in rows]), "source robustness")

    breakdown_specs = (
        ("difficulty", "Difficulty"),
        ("number_of_sources", "Number of composed sources"),
        ("target_position_bin", "Target-position bin"),
        ("chunk_count_quartile", "Chunk-count quartile"),
    )
    figure_number = 15
    for dimension, label in breakdown_specs:
        for metric, metric_label in (("exact", "Exact match"), ("token_f1", "Token F1")):
            rows = breakdown_rows(report, dimension, metric)
            title = f"Selected-system {metric_label.lower()} by {label.lower()}"
            builder.save(f"A{figure_number:02d}", f"{dimension}_{metric}", title, rows, grouped(rows, "group", "backbone", "value", title, metric_label), "condition robustness")
            figure_number += 1

    for identifier, metric, title in (
        ("A23", "exact", "Absolute exact-match uncertainty"),
        ("A24", "token_f1", "Absolute token-F1 uncertainty"),
    ):
        interval_rows = []
        for system in systems:
            item = uncertainty["absolute_intervals"]["two_backbone_mean"][system][metric]
            interval_rows.append({"label": SYSTEM_LABELS[system], "system": system, "estimate": item["estimate"], "lower": item["bootstrap_95_interval"][0], "upper": item["bootstrap_95_interval"][1], "sources": item["sources"]})
        builder.save(identifier, f"absolute_{metric}_forest", title, interval_rows, forest(interval_rows, title, "Target-source macro score (95% bootstrap interval)"), "uncertainty")

    for identifier, comparator, title in (
        ("A25", strongest, "Exact-match transitions versus strongest baseline"),
        ("A26", "deterministic_random_retrieval", "Exact-match transitions versus random retrieval"),
    ):
        rows = transition_rows(rows_by_model, comparator)
        builder.save(identifier, f"transition_{comparator}", title, rows, grouped(rows, "transition", "backbone", "examples", title, "Examples"), "paired outcomes")

    selected_rows = [row for row in rows_by_model[MODELS[0]] if row["system"] == "selected_learned_retrieval"]
    duration_rows = [{"example_id": row["example_id"], "source": row["target_source_id"], "duration_sec": row["audio_duration_sec"]} for row in selected_rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    ax.hist([row["duration_sec"] for row in duration_rows], bins=20, color=BLUE)
    ax.set(xlabel="Selected audio duration (s)", ylabel="Examples")
    ax.set_title("Selected evidence-reservoir duration", loc="left", fontweight="bold")
    builder.save("A27", "selected_duration", "Selected evidence-reservoir duration", duration_rows, fig, "mechanism")

    positive_rows = [{"example_id": row["example_id"], "source": row["target_source_id"], "positive_chunks": row["selected_positive_chunks"], "selected_chunks": row["selected_chunk_count"]} for row in selected_rows]
    counts = Counter(int(row["positive_chunks"]) for row in positive_rows)
    positive_summary = [{"positive_chunks": str(value), "examples": counts[value]} for value in sorted(counts)]
    builder.save("A28", "selected_positive_chunks", "Positive chunks in the selected reservoir", positive_summary, bar(positive_summary, "positive_chunks", "examples", "Evidence yield of learned retrieval", "Examples"), "mechanism")

    latency_rows = []
    token_rows = []
    response_length_rows = []
    for model in MODELS:
        for system in systems:
            items = [row for row in rows_by_model[model] if row["system"] == system]
            latency_rows.append({"backbone": MODEL_LABELS[model], "system": SYSTEM_LABELS[system], "median_seconds": float(np.median([row["generation_seconds"] for row in items])), "p90_seconds": float(np.quantile([row["generation_seconds"] for row in items], 0.9)), "examples": len(items)})
            token_rows.append({"backbone": MODEL_LABELS[model], "system": SYSTEM_LABELS[system], "mean_input_tokens": float(np.mean([row["input_tokens"] for row in items])), "p90_input_tokens": float(np.quantile([row["input_tokens"] for row in items], 0.9)), "examples": len(items)})
            response_length_rows.append({"backbone": MODEL_LABELS[model], "system": SYSTEM_LABELS[system], "mean_response_tokens": float(np.mean([len(str(row["normalized_response"]).split()) for row in items])), "p90_response_tokens": float(np.quantile([len(str(row["normalized_response"]).split()) for row in items], 0.9)), "examples": len(items)})
    builder.save("A29", "generation_latency", "Generation latency by backbone and system", latency_rows, grouped(latency_rows, "system", "backbone", "median_seconds", "Median answer-generation latency", "Seconds"), "efficiency")
    builder.save("A30", "input_tokens", "Input-token burden by backbone and system", token_rows, grouped(token_rows, "system", "backbone", "mean_input_tokens", "Mean input-token burden", "Input tokens"), "efficiency")
    builder.save("A31", "response_length", "Response length by backbone and system", response_length_rows, grouped(response_length_rows, "system", "backbone", "mean_response_tokens", "Mean normalized response length", "Whitespace tokens"), "answer behavior")

    source_counts = Counter(row["target_source_id"] for row in selected_rows)
    source_count_rows = [
        {
            "source": source,
            "source_label": f"S{index:02d}",
            "examples": source_counts[source],
        }
        for index, source in enumerate(sorted(source_counts), start=1)
    ]
    builder.save("A32", "source_inventory", "Held-out examples per target source", source_count_rows, bar(source_count_rows, "source_label", "examples", "Held-out target-source inventory", "Examples"), "reproducibility")

    oracle_rows = []
    for model in MODELS:
        learned = float(report["metrics"][model]["selected_learned_retrieval"]["source_macro_exact"])
        oracle = float(report["metrics"][model]["oracle_retrieval"]["source_macro_exact"])
        oracle_rows.append({"backbone": MODEL_LABELS[model], "learned_exact": learned, "oracle_exact": oracle, "oracle_headroom": oracle - learned})
    learned_combined = float(report["combined_metrics"]["selected_learned_retrieval"]["two_backbone_mean_source_macro_exact"])
    oracle_combined = float(report["combined_metrics"]["oracle_retrieval"]["two_backbone_mean_source_macro_exact"])
    oracle_rows.append({"backbone": "Two-backbone mean", "learned_exact": learned_combined, "oracle_exact": oracle_combined, "oracle_headroom": oracle_combined - learned_combined})
    builder.save("A33", "oracle_headroom", "Exact-match oracle headroom", oracle_rows, bar(oracle_rows, "backbone", "oracle_headroom", "Remaining retrieval oracle headroom", "Oracle − learned exact", reference=0), "mechanism")

    control_delta_rows = []
    for model in MODELS:
        learned = float(report["metrics"][model]["selected_learned_retrieval"]["source_macro_exact"])
        for control in ("selected_retrieval_silenced", "text_only"):
            value = float(report["metrics"][model][control]["source_macro_exact"])
            control_delta_rows.append({"backbone": MODEL_LABELS[model], "control": SYSTEM_LABELS[control], "delta": learned - value})
    builder.save("A34", "control_exact_deltas", "Exact-match differences from audio controls", control_delta_rows, grouped(control_delta_rows, "control", "backbone", "delta", "Learned retrieval minus control", "Exact-match difference"), "controls")

    audio_systems = [system for system in systems if system != "text_only"]
    evidence_rows = []
    qwen_rows = rows_by_model[MODELS[0]]
    for system in audio_systems:
        items = [row for row in qwen_rows if row["system"] == system]
        evidence_rows.append({"system": SYSTEM_LABELS[system], "mean_positive_chunks": float(np.mean([row["selected_positive_chunks"] for row in items])), "mean_selected_chunks": float(np.mean([row["selected_chunk_count"] for row in items])), "positive_fraction": float(np.mean([row["selected_positive_chunks"] / row["selected_chunk_count"] for row in items])), "examples": len(items)})
    builder.save("A35", "evidence_yield", "Evidence yield across audio systems", evidence_rows, bar(evidence_rows, "system", "positive_fraction", "Positive-evidence fraction in retrieved audio", "Positive selected-chunk fraction"), "mechanism")

    candidate_rows = []
    for candidate in calibration["candidate_metrics"]:
        label = f"{candidate['prompt_name']} · K={candidate['k']}"
        for model in MODELS:
            candidate_rows.append({"candidate": label, "backbone": MODEL_LABELS[model], "exact": candidate["per_backbone"][model]["source_macro_exact"], "token_f1": candidate["per_backbone"][model]["source_macro_token_f1"], "selected": int(candidate["prompt_name"] == calibration["selected_prompt_name"] and int(candidate["k"]) == int(calibration["selected_k"]))})
    builder.save("A36", "calibration_surface", "Frozen calibration prompt/K surface", candidate_rows, grouped(candidate_rows, "candidate", "backbone", "exact", "Calibration exact match across prompt/K candidates", "Target-source macro exact"), "calibration")

    receipt_rows = []
    for model in MODELS:
        receipt_path = RESULTS / f"heldout__{model}.receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt_rows.append({"backbone": MODEL_LABELS[model], "load_seconds": receipt["load_seconds"], "run_seconds": receipt["run_seconds"], "peak_cuda_gib": receipt["peak_cuda_bytes"] / 2**30, "rows": receipt["rows"], "raw_sha256": receipt["raw_sha256"]})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7))
    positions = np.arange(len(receipt_rows))
    labels = [row["backbone"] for row in receipt_rows]
    axes[0].bar(positions, [row["run_seconds"] / 60 for row in receipt_rows], color=BLUE)
    axes[0].set(xticks=positions, xticklabels=labels, ylabel="Run time (min)", title="Generation run time")
    axes[1].bar(positions, [row["peak_cuda_gib"] for row in receipt_rows], color=PURPLE)
    axes[1].set(xticks=positions, xticklabels=labels, ylabel="Peak CUDA memory (GiB)", title="Peak accelerator memory")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Held-out runtime and memory", x=0.01, ha="left", fontweight="bold")
    builder.save("A37", "runtime_inventory", "Backbone runtime and memory inventory", receipt_rows, fig, "efficiency")

    integrity_rows = [{"check": key.replace("_", " "), "value": int(value) if isinstance(value, bool) else value} for key, value in report["integrity"].items() if key != "audita_rows_accessed"]
    integrity_rows.append({"check": "AUDITA rows accessed", "value": report["integrity"]["audita_rows_accessed"]})
    builder.save("A38", "integrity_inventory", "Held-out integrity inventory", integrity_rows, bar(integrity_rows, "check", "value", "Held-out integrity inventory", "Count / indicator", horizontal=True), "reproducibility")

    oracle_source_rows = source_delta_rows(rows_by_model, "oracle_retrieval")
    builder.save("A39", "source_oracle_delta", "Source-level selected-versus-oracle exact difference", oracle_source_rows, bar(oracle_source_rows, "source_label", "delta", "Source-level learned minus oracle exact", "Learned − oracle", horizontal=True, reference=0, colors=[GREEN if row["delta"] >= 0 else RED for row in oracle_source_rows]), "source robustness")

    learned_exact = float(report["combined_metrics"]["selected_learned_retrieval"]["two_backbone_mean_source_macro_exact"])
    strongest_exact = float(report["combined_metrics"][strongest]["two_backbone_mean_source_macro_exact"])
    random_exact = float(report["combined_metrics"]["deterministic_random_retrieval"]["two_backbone_mean_source_macro_exact"])
    gate_effect_rows = [
        {"effect": "Gain vs strongest", "observed": learned_exact - strongest_exact, "threshold": float(config["promotion_gates"]["minimum_exact_gain_over_strongest_baseline"]), "direction": "higher", "margin": learned_exact - strongest_exact - float(config["promotion_gates"]["minimum_exact_gain_over_strongest_baseline"])},
        {"effect": "Gain vs random", "observed": learned_exact - random_exact, "threshold": float(config["promotion_gates"]["minimum_exact_gain_over_random"]), "direction": "higher", "margin": learned_exact - random_exact - float(config["promotion_gates"]["minimum_exact_gain_over_random"])},
        {"effect": "Worst backbone vs strongest", "observed": min(float(report["metrics"][model]["selected_learned_retrieval"]["source_macro_exact"]) - float(report["metrics"][model][strongest]["source_macro_exact"]) for model in MODELS), "threshold": -float(config["promotion_gates"]["maximum_single_backbone_loss_to_strongest_baseline"]), "direction": "higher", "margin": min(float(report["metrics"][model]["selected_learned_retrieval"]["source_macro_exact"]) - float(report["metrics"][model][strongest]["source_macro_exact"]) for model in MODELS) + float(config["promotion_gates"]["maximum_single_backbone_loss_to_strongest_baseline"])},
        {"effect": "Minimum response change vs silence", "observed": min(float(report["response_difference_fractions"][model]["selected_retrieval_silenced"]) for model in MODELS), "threshold": float(config["promotion_gates"]["minimum_normalized_response_difference_fraction_from_silence_per_backbone"]), "direction": "higher", "margin": min(float(report["response_difference_fractions"][model]["selected_retrieval_silenced"]) for model in MODELS) - float(config["promotion_gates"]["minimum_normalized_response_difference_fraction_from_silence_per_backbone"])},
        {"effect": "Minimum response change vs text", "observed": min(float(report["response_difference_fractions"][model]["text_only"]) for model in MODELS), "threshold": float(config["promotion_gates"]["minimum_normalized_response_difference_fraction_from_text_only_per_backbone"]), "direction": "higher", "margin": min(float(report["response_difference_fractions"][model]["text_only"]) for model in MODELS) - float(config["promotion_gates"]["minimum_normalized_response_difference_fraction_from_text_only_per_backbone"])},
        {"effect": "Worst backbone vs silence", "observed": min(float(report["metrics"][model]["selected_learned_retrieval"]["source_macro_exact"]) - float(report["metrics"][model]["selected_retrieval_silenced"]["source_macro_exact"]) for model in MODELS), "threshold": 0.0, "direction": "higher", "margin": min(float(report["metrics"][model]["selected_learned_retrieval"]["source_macro_exact"]) - float(report["metrics"][model]["selected_retrieval_silenced"]["source_macro_exact"]) for model in MODELS)},
    ]
    builder.save("A40", "gate_effect_margins", "Gate-relevant effect-size margins", gate_effect_rows, bar(gate_effect_rows, "effect", "margin", "Observed effect minus frozen threshold", "Pass margin", horizontal=True, reference=0, colors=[GREEN if row["margin"] >= 0 else RED for row in gate_effect_rows]), "primary results")

    if tuple(row["id"] for row in builder.records) != EXPECTED_FIGURE_IDS:
        raise RuntimeError("held-out answer figure identity/count gate failed")
    with INDEX.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(builder.records[0]))
        writer.writeheader()
        writer.writerows(builder.records)
    audit = {
        "status": "heldout_answer_publication_figures_complete",
        "heldout_report_status": report["status"],
        "claim_scope": "heldout_clotho_answer_gate_only_audita_still_separately_controlled",
        "distinct_figures": len(builder.records),
        "pdf_files": len(list(PDF.glob("*.pdf"))),
        "png_files": len(list(PNG.glob("*.png"))),
        "source_data_files": len(list(SOURCE.glob("*.csv"))),
        "all_numeric_figures_have_source_data": all(row["source_rows"] > 0 for row in builder.records),
        "report_sha256": sha256(REPORT),
        "uncertainty_sha256": sha256(UNCERTAINTY),
        "calibration_sha256": sha256(CALIBRATION),
        "authorization_sha256": sha256(AUTHORIZATION),
        "protocol_sha256": sha256(PROTOCOL),
        "figure_script_sha256": sha256(Path(__file__).resolve()),
        "reporting_lock_sha256": sha256(REPORTING_LOCK),
        "reporting_amendment_001_sha256": sha256(REPORTING_AMENDMENT_001),
        "reporting_amendment_002_sha256": sha256(REPORTING_AMENDMENT_002),
        "figure_index_sha256": sha256(INDEX),
        "raw_sha256": {model: sha256(RESULTS / f"heldout__{model}.jsonl") for model in MODELS},
        "audita_rows_accessed": 0,
    }
    AUDIT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
