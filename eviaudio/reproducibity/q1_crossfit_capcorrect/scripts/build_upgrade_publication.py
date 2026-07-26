#!/usr/bin/env python3
"""Build source-linked publication synthesis across all EviAudio stages."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
TRACK = PROJECT / "q1_crossfit_capcorrect"
OUTPUT = TRACK / "publication_upgrade"
FIGURES = OUTPUT / "figures"
SOURCES = OUTPUT / "figure_source_data"
ANALYSIS = OUTPUT / "EVIAUDIO_UPGRADE_ANALYSIS.json"
CONTROLLED = (
    PROJECT
    / "q1_plus/results/confirmatory/event_ranker/"
    "five_seed_confirmatory_report.json"
)
EXTERNAL = (
    PROJECT
    / "q1_90/results/external_replication/evaluation/"
    "external_replication_report.json"
)
NATURAL = (
    PROJECT
    / "q1_top_tier/results/perception_test/evaluation/"
    "perception_test_report.json"
)
NATURAL_PUBLICATION = (
    PROJECT / "q1_top_tier/publication/PERCEPTION_PUBLICATION_ANALYSIS.json"
)
ROUTER = TRACK / "results/capcorrect_router/capcorrect_router_report.json"
DOWNSTREAM = (
    PROJECT
    / "q1_plus/results/development/answer_generation/"
    "heldout_evaluation_report.json"
)
FULL_PREDICTIONS = (
    TRACK / "results/capcorrect_router/capcorrect_all_candidates.jsonl.gz"
)
CLAP_ROUTER_PREDICTIONS = (
    TRACK / "results/capcorrect_router/capcorrect_clap_only.jsonl.gz"
)
BACKEND = PROJECT / "q1_top_tier"
sys.path.insert(0, str(TRACK / "scripts"))
sys.path.insert(0, str(TRACK / "src"))
sys.path.insert(0, str(BACKEND / "scripts"))
sys.path.insert(0, str(BACKEND / "src"))
sys.path.insert(0, str(PROJECT / "src"))

import analyze_capcorrect_router as capcorrect  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"empty source table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(figure: plt.Figure, identifier: str) -> dict[str, str]:
    pdf = FIGURES / f"{identifier}.pdf"
    png = FIGURES / f"{identifier}.png"
    figure.savefig(
        pdf,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    figure.savefig(png, bbox_inches="tight", dpi=220)
    plt.close(figure)
    return {
        "pdf": pdf.relative_to(TRACK).as_posix(),
        "pdf_sha256": sha256(pdf),
        "png": png.relative_to(TRACK).as_posix(),
        "png_sha256": sha256(png),
    }


def gate_count(report: dict[str, Any], key: str = "promotion_gates") -> str:
    gates = report[key]
    return f"{sum(bool(value) for value in gates.values())}/{len(gates)}"


def load_reports() -> dict[str, dict[str, Any]]:
    reports = {
        "controlled": json.loads(CONTROLLED.read_text(encoding="utf-8")),
        "external": json.loads(EXTERNAL.read_text(encoding="utf-8")),
        "natural": json.loads(NATURAL.read_text(encoding="utf-8")),
        "natural_publication": json.loads(
            NATURAL_PUBLICATION.read_text(encoding="utf-8")
        ),
        "router": json.loads(ROUTER.read_text(encoding="utf-8")),
        "downstream": json.loads(DOWNSTREAM.read_text(encoding="utf-8")),
    }
    if (
        reports["controlled"]["status"] != "confirmatory_exact_onset_gate_pass"
        or reports["external"]["status"] != "external_replication_gate_fail"
        or reports["natural"]["status"] != "natural_panel_promotion_fail"
        or reports["router"]["status"]
        != "cap_corrected_post_outcome_diagnostic_target_met"
        or reports["downstream"]["status"]
        != "heldout_answer_gate_fail_audita_sealed"
    ):
        raise RuntimeError("an EviAudio evidence stage has an unexpected status")
    return reports


def incremental_video_bootstrap(
    router_report: dict[str, Any],
) -> dict[str, Any]:
    _, natural_config, index = capcorrect.frozen.audit_inputs()
    truth, labels = capcorrect.ground_truth(natural_config)
    video_ids = [str(row["video_id"]) for row in index]
    full_rows = capcorrect.load_jsonl(FULL_PREDICTIONS)
    clap_rows = capcorrect.load_jsonl(CLAP_ROUTER_PREDICTIONS)
    full_video = capcorrect.frozen.per_video_macro_ap(
        truth, full_rows, video_ids
    )
    clap_video = capcorrect.frozen.per_video_macro_ap(
        truth, clap_rows, video_ids
    )
    differences = np.asarray(
        [full_video[name] - clap_video[name] for name in video_ids],
        dtype=np.float64,
    )
    bootstrap = capcorrect.frozen.bootstrap_delta(
        differences, replicates=10_000, seed=20260729
    )
    full_global = float(
        router_report["primary"]["router_metrics"]["mean_map"]
    )
    clap_global = float(
        router_report["clap_only_ablation"]["metrics"]["mean_map"]
    )
    return {
        "status": "posthoc_full_vs_clap_only_video_bootstrap_complete",
        "global_mean_map_delta": full_global - clap_global,
        "global_relative_gain": full_global / clap_global - 1.0,
        "paired_video_bootstrap": bootstrap,
        "videos": len(video_ids),
        "seed": 20260729,
        "replicates": 10_000,
        "post_hoc": True,
        "full_predictions_sha256": sha256(FULL_PREDICTIONS),
        "clap_only_predictions_sha256": sha256(CLAP_ROUTER_PREDICTIONS),
        "interpretation": (
            "Incremental QCR-versus-CLAP-only routing contrast on the already "
            "exposed validation panel; not a fresh or zero-shot endpoint."
        ),
    }


def main() -> None:
    reports = load_reports()
    controlled = reports["controlled"]
    external = reports["external"]
    natural = reports["natural"]
    router = reports["router"]
    downstream = reports["downstream"]
    supervised = reports["natural_publication"][
        "contextual_supervised_reference"
    ]["metric_recomputed_with_primary_implementation"]
    incremental = incremental_video_bootstrap(router)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    SOURCES.mkdir(parents=True, exist_ok=True)

    evidence_rows = [
        {
            "stage": 1,
            "name": "Controlled exact-onset",
            "evidence_type": "frozen confirmatory",
            "panel": "397 source-isolated recipes",
            "primary_result": (
                f"AP {controlled['prior']['evidence_ap']:.3f} to "
                f"{controlled['ensemble']['evidence_ap']:.3f}"
            ),
            "gate_result": gate_count(controlled),
            "status": "pass",
            "claim": "controlled evidence ranking",
        },
        {
            "stage": 2,
            "name": "External exact-onset",
            "evidence_type": "frozen external replication",
            "panel": "219 UrbanSound8K/LibriSpeech recipes",
            "primary_result": (
                f"AP {external['prior']['evidence_ap']:.3f} to "
                f"{external['ensemble']['evidence_ap']:.3f}"
            ),
            "gate_result": gate_count(external),
            "status": "strict floor fail",
            "claim": "positive relative gain; absolute floor unmet",
        },
        {
            "stage": 3,
            "name": "Natural zero-shot",
            "evidence_type": "frozen natural transfer",
            "panel": "5,359 Perception Test videos",
            "primary_result": (
                f"mAP {natural['methods']['clap_multiscale']['mean_map']:.3f} "
                f"to {natural['methods']['qcr_multiscale']['mean_map']:.3f}"
            ),
            "gate_result": gate_count(natural),
            "status": "fail",
            "claim": "controlled QCR did not transfer",
        },
        {
            "stage": 4,
            "name": "Cap-correct diagnosis",
            "evidence_type": "post-outcome cross-fitted exploratory",
            "panel": "same exposed validation videos",
            "primary_result": (
                f"mAP {natural['methods']['clap_multiscale']['mean_map']:.3f} "
                f"to {router['primary']['router_metrics']['mean_map']:.3f}"
            ),
            "gate_result": gate_count(router, "diagnostic_criteria"),
            "status": "diagnostic target met",
            "claim": "class-dependent scale is dominant mechanism",
        },
        {
            "stage": 5,
            "name": "Answer bridge",
            "evidence_type": "held-out downstream boundary",
            "panel": "411 examples; two decoders",
            "primary_result": (
                "exact "
                f"{downstream['combined_metrics']['selected_learned_retrieval']['two_backbone_mean_source_macro_exact']:.3f} "
                "vs strongest "
                f"{downstream['combined_metrics']['prefix_matched_retrieval']['two_backbone_mean_source_macro_exact']:.3f}"
            ),
            "gate_result": gate_count(downstream),
            "status": "fail",
            "claim": "localization gain did not improve answers",
        },
    ]
    write_csv(SOURCES / "AEF01_evidence_ledger.csv", evidence_rows)
    colors = ("#2d8b68", "#d29a32", "#c65353", "#536fa8", "#c65353")
    fig, axis = plt.subplots(figsize=(11.2, 3.3))
    axis.set_xlim(-0.65, 4.65)
    axis.set_ylim(-0.7, 1.35)
    axis.axis("off")
    for index, (row, color) in enumerate(zip(evidence_rows, colors, strict=True)):
        display = {
            0: "Controlled\nexact-onset",
            1: "External\nexact-onset",
            2: "Natural\nzero-shot",
            3: "Cap-correct\ncross-fit",
            4: "Answer\nbridge",
        }[index]
        axis.text(
            index,
            0.72,
            display,
            ha="center",
            va="center",
            fontsize=10,
            weight="bold",
            bbox={
                "boxstyle": "round,pad=0.48",
                "facecolor": mcolors.to_rgba(color, 0.14),
                "edgecolor": color,
            },
        )
        axis.text(
            index,
            -0.03,
            f"{row['gate_result']} gates\n{row['status']}",
            ha="center",
            va="center",
            fontsize=8,
            color=color,
        )
        axis.text(
            index,
            -0.48,
            row["primary_result"],
            ha="center",
            va="center",
            fontsize=7.3,
        )
        if index < len(evidence_rows) - 1:
            axis.annotate(
                "",
                xy=(index + 0.67, 0.72),
                xytext=(index + 0.33, 0.72),
                arrowprops={"arrowstyle": "->", "lw": 1.3, "color": "#495766"},
            )
    axis.set_title(
        "Evidence ladder: positive controlled ranking, failed natural transfer, "
        "and post-outcome diagnosis",
        fontsize=12,
    )
    figure_records: list[dict[str, Any]] = [
        {
            "id": "AEF01",
            "title": "Outcome-faithful evidence ladder",
            "source_files": ["AEF01_evidence_ledger.csv"],
            **save_figure(fig, "AEF01"),
        }
    ]

    controlled_rows = [
        {
            "panel": "source-isolated confirmatory",
            "method": "CLAP prior",
            "evidence_ap": controlled["prior"]["evidence_ap"],
            "hit_at_1": controlled["prior"]["hit_at_1"],
            "gate_status": controlled["status"],
        },
        {
            "panel": "source-isolated confirmatory",
            "method": "QCR ensemble",
            "evidence_ap": controlled["ensemble"]["evidence_ap"],
            "hit_at_1": controlled["ensemble"]["hit_at_1"],
            "gate_status": controlled["status"],
        },
        {
            "panel": "external source-disjoint replication",
            "method": "CLAP prior",
            "evidence_ap": external["prior"]["evidence_ap"],
            "hit_at_1": external["prior"]["hit_at_1"],
            "gate_status": external["status"],
        },
        {
            "panel": "external source-disjoint replication",
            "method": "QCR ensemble",
            "evidence_ap": external["ensemble"]["evidence_ap"],
            "hit_at_1": external["ensemble"]["hit_at_1"],
            "gate_status": external["status"],
        },
    ]
    natural_rows = [
        {
            "method": "Frozen CLAP multiscale",
            "mean_map": natural["methods"]["clap_multiscale"]["mean_map"],
            "evidence_type": "frozen zero-shot comparator",
        },
        {
            "method": "Frozen QCR multiscale",
            "mean_map": natural["methods"]["qcr_multiscale"]["mean_map"],
            "evidence_type": "frozen zero-shot primary",
        },
        {
            "method": "Cross-fit CLAP-only",
            "mean_map": router["clap_only_ablation"]["metrics"]["mean_map"],
            "evidence_type": "post-outcome exploratory",
        },
        {
            "method": "Cross-fit CLAP+QCR",
            "mean_map": router["primary"]["router_metrics"]["mean_map"],
            "evidence_type": "post-outcome exploratory",
        },
        {
            "method": "ActionFormer",
            "mean_map": supervised["mean_map"],
            "evidence_type": "supervised context",
        },
    ]
    write_csv(SOURCES / "AEF02_controlled.csv", controlled_rows)
    write_csv(SOURCES / "AEF02_natural.csv", natural_rows)
    fig, (controlled_axis, natural_axis) = plt.subplots(1, 2, figsize=(10.6, 4.3))
    panel_names = ("source-isolated confirmatory", "external source-disjoint replication")
    x = np.arange(2)
    width = 0.34
    for offset, method, color in (
        (-width / 2, "CLAP prior", "#8295a7"),
        (width / 2, "QCR ensemble", "#2d8b68"),
    ):
        values = [
            next(
                float(row["evidence_ap"])
                for row in controlled_rows
                if row["panel"] == panel and row["method"] == method
            )
            for panel in panel_names
        ]
        controlled_axis.bar(
            x + offset, values, width, label=method, color=color
        )
    controlled_axis.set_xticks(
        x, ["Controlled\n397 recipes", "External\n219 recipes"]
    )
    controlled_axis.set_ylabel("evidence average precision")
    controlled_axis.set_ylim(0.0, 0.92)
    controlled_axis.set_title("Exact-onset evidence ranking")
    controlled_axis.legend(frameon=False, fontsize=8)

    display_method = {
        "Frozen CLAP multiscale": "CLAP fixed",
        "Frozen QCR multiscale": "QCR fixed",
        "Cross-fit CLAP-only": "Scale router",
        "Cross-fit CLAP+QCR": "Full router",
        "ActionFormer": "ActionFormer",
    }
    labels = [display_method[str(row["method"])] for row in natural_rows]
    values = [float(row["mean_map"]) for row in natural_rows]
    natural_colors = ("#8295a7", "#c65353", "#5c83b8", "#2d8b68", "#8b6aa8")
    natural_axis.barh(labels, values, color=natural_colors)
    natural_axis.invert_yaxis()
    natural_axis.set_xlabel("class-macro temporal mAP")
    natural_axis.set_title("Natural Perception Test")
    for index, value in enumerate(values):
        natural_axis.text(
            value + 0.001,
            index,
            f"{value:.3f}",
            va="center",
            fontsize=7.5,
        )
    fig.subplots_adjust(wspace=0.35)
    fig.suptitle("Controlled gains do not imply fixed zero-shot natural transfer")
    figure_records.append(
        {
            "id": "AEF02",
            "title": "Controlled and natural performance",
            "source_files": ["AEF02_controlled.csv", "AEF02_natural.csv"],
            **save_figure(fig, "AEF02"),
        }
    )

    threshold_methods = {
        "Frozen CLAP multiscale": natural["methods"]["clap_multiscale"],
        "Frozen QCR multiscale": natural["methods"]["qcr_multiscale"],
        "Cross-fit CLAP-only": router["clap_only_ablation"]["metrics"],
        "Cross-fit CLAP+QCR": router["primary"]["router_metrics"],
        "Supervised ActionFormer": supervised,
    }
    threshold_rows: list[dict[str, Any]] = []
    for method, metrics in threshold_methods.items():
        for threshold, value in zip(
            metrics["thresholds"], metrics["map_by_threshold"], strict=True
        ):
            threshold_rows.append(
                {
                    "method": method,
                    "temporal_iou": threshold,
                    "class_macro_map": value,
                }
            )
    write_csv(SOURCES / "AEF03_thresholds.csv", threshold_rows)
    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    styles = {
        "Frozen CLAP multiscale": ("#8295a7", "o", "-"),
        "Frozen QCR multiscale": ("#c65353", "s", "-"),
        "Cross-fit CLAP-only": ("#5c83b8", "^", "--"),
        "Cross-fit CLAP+QCR": ("#2d8b68", "D", "--"),
        "Supervised ActionFormer": ("#8b6aa8", "P", ":"),
    }
    for method, metrics in threshold_methods.items():
        color, marker, line = styles[method]
        axis.plot(
            metrics["thresholds"],
            metrics["map_by_threshold"],
            label=method,
            color=color,
            marker=marker,
            linestyle=line,
        )
    axis.set_xlabel("temporal IoU threshold")
    axis.set_ylabel("class-macro mAP")
    axis.set_title("Natural localization across all official thresholds")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(axis="y", alpha=0.2)
    figure_records.append(
        {
            "id": "AEF03",
            "title": "Temporal-IoU sensitivity",
            "source_files": ["AEF03_thresholds.csv"],
            **save_figure(fig, "AEF03"),
        }
    )

    label_names = natural["label_names"]
    labels_order = list(label_names)
    selection_rows: list[dict[str, Any]] = []
    selected_names: list[str] = []
    for fold in range(5):
        for label in labels_order:
            item = router["selection"][str(fold)][label]
            candidate = str(item["all_candidates"])
            selected_names.append(candidate)
            selection_rows.append(
                {
                    "fold": fold,
                    "label_id": label,
                    "label_name": label_names[label],
                    "all_candidates": candidate,
                    "clap_only": item["clap_only"],
                    "training_ap_selected": item["training_ap"][candidate],
                }
            )
    candidates = sorted(set(selected_names))
    candidate_index = {name: index for index, name in enumerate(candidates)}
    matrix = np.zeros((len(labels_order), 5), dtype=np.int64)
    for row_index, label in enumerate(labels_order):
        for fold in range(5):
            candidate = router["selection"][str(fold)][label]["all_candidates"]
            matrix[row_index, fold] = candidate_index[str(candidate)]

    full_classes = router["primary"]["router_metrics"]["class_mean_ap"]
    clap_fixed_classes = router["primary"]["comparator_metrics"]["class_mean_ap"]
    clap_router_classes = router["clap_only_ablation"]["metrics"]["class_mean_ap"]
    class_rows = []
    for label in labels_order:
        class_rows.append(
            {
                "label_id": label,
                "label_name": label_names[label],
                "full_minus_fixed_clap": float(full_classes[label])
                - float(clap_fixed_classes[label]),
                "full_minus_clap_only_router": float(full_classes[label])
                - float(clap_router_classes[label]),
                "full_router_ap": full_classes[label],
                "clap_only_router_ap": clap_router_classes[label],
                "fixed_clap_ap": clap_fixed_classes[label],
            }
        )
    write_csv(SOURCES / "AEF04_selection.csv", selection_rows)
    write_csv(SOURCES / "AEF04_class_delta.csv", class_rows)
    fig, (heatmap, deltas) = plt.subplots(
        1, 2, figsize=(10.8, 5.1), gridspec_kw={"width_ratios": [1.0, 1.25]}
    )
    cmap = mcolors.ListedColormap(
        ["#8da0b3", "#6f91ae", "#5b7b97", "#a8b7c2", "#4f9a78", "#2d7559"][
            : len(candidates)
        ]
    )
    heatmap.imshow(
        matrix,
        cmap=cmap,
        vmin=-0.5,
        vmax=len(candidates) - 0.5,
        aspect="auto",
    )
    abbreviations = {
        "clap_0p5s": "C-0.5",
        "clap_1s": "C-1",
        "clap_2s": "C-2",
        "clap_4s": "C-4",
        "clap_multiscale": "C-MS",
        "qcr_1s": "Q-1",
        "qcr_4s": "Q-4",
    }
    for row_index in range(len(labels_order)):
        for fold in range(5):
            name = candidates[int(matrix[row_index, fold])]
            heatmap.text(
                fold,
                row_index,
                abbreviations.get(name, name),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if name.startswith("qcr_") else "black",
                weight="bold",
            )
    heatmap.set_xticks(range(5), [f"Fold {value}" for value in range(5)])
    heatmap.set_yticks(
        range(len(labels_order)),
        [label_names[label].replace("Interaction:", "").replace("Object:", "") for label in labels_order],
        fontsize=7.5,
    )
    heatmap.set_title("Out-of-fold class routing")

    y = np.arange(len(class_rows))
    full_delta = np.asarray(
        [float(row["full_minus_fixed_clap"]) for row in class_rows]
    )
    incremental_delta = np.asarray(
        [float(row["full_minus_clap_only_router"]) for row in class_rows]
    )
    height = 0.35
    deltas.barh(
        y - height / 2,
        full_delta,
        height,
        label="full router − fixed CLAP",
        color="#2d8b68",
    )
    deltas.barh(
        y + height / 2,
        incremental_delta,
        height,
        label="full router − CLAP-only router",
        color="#d29a32",
    )
    deltas.axvline(0.0, color="black", lw=1)
    deltas.set_yticks(y, ["" for _ in class_rows])
    deltas.invert_yaxis()
    deltas.set_xlabel("class mean-AP difference")
    deltas.set_title("Scale gain versus incremental QCR gain")
    deltas.legend(frameon=False, fontsize=7.5)
    fig.suptitle("Cross-fitted diagnosis: scale choice dominates the QCR residual")
    figure_records.append(
        {
            "id": "AEF04",
            "title": "Cross-fitted selections and class effects",
            "source_files": ["AEF04_selection.csv", "AEF04_class_delta.csv"],
            **save_figure(fig, "AEF04"),
        }
    )

    downstream_rows = []
    for method, values in downstream["combined_metrics"].items():
        downstream_rows.append(
            {
                "method": method,
                "two_backbone_mean_source_macro_exact": values[
                    "two_backbone_mean_source_macro_exact"
                ],
                "two_backbone_mean_source_macro_token_f1": values[
                    "two_backbone_mean_source_macro_token_f1"
                ],
            }
        )
    write_csv(SOURCES / "AEF05_downstream.csv", downstream_rows)
    display_order = (
        "selected_learned_retrieval",
        "clap_retrieval",
        "prefix_matched_retrieval",
        "deterministic_random_retrieval",
        "oracle_retrieval",
        "selected_retrieval_silenced",
        "text_only",
    )
    display_names = {
        "selected_learned_retrieval": "Learned QCR",
        "clap_retrieval": "CLAP",
        "prefix_matched_retrieval": "Prefix matched",
        "deterministic_random_retrieval": "Random",
        "oracle_retrieval": "Oracle spans",
        "selected_retrieval_silenced": "Silenced",
        "text_only": "Text only",
    }
    lookup = {row["method"]: row for row in downstream_rows}
    values = [
        float(lookup[name]["two_backbone_mean_source_macro_exact"])
        for name in display_order
    ]
    fig, axis = plt.subplots(figsize=(7.7, 4.5))
    axis.barh(
        [display_names[name] for name in display_order],
        values,
        color=[
            "#c65353",
            "#8295a7",
            "#5c83b8",
            "#a6a6a6",
            "#8b6aa8",
            "#d8b365",
            "#bababa",
        ],
    )
    axis.invert_yaxis()
    axis.set_xlabel("two-decoder mean source-macro exact accuracy")
    axis.set_title("Held-out answer bridge: improved ranking is not enough")
    figure_records.append(
        {
            "id": "AEF05",
            "title": "Downstream answer boundary",
            "source_files": ["AEF05_downstream.csv"],
            **save_figure(fig, "AEF05"),
        }
    )

    for record in figure_records:
        record["source_sha256"] = {
            name: sha256(SOURCES / name)
            for name in record["source_files"]
        }

    natural_fixed_delta = float(
        natural["primary_contrast"]["official_mean_map_delta"]
    )
    full_delta = float(router["primary"]["official_mean_map_delta"])
    clap_only_delta = float(
        router["clap_only_ablation"]["delta_to_clap_multiscale"]
    )
    synthesis = {
        "status": "eviaudio_multistage_publication_analysis_complete",
        "controlled_confirmatory": {
            "status": controlled["status"],
            "examples": controlled["examples"],
            "prior": controlled["prior"],
            "ensemble": controlled["ensemble"],
            "delta": controlled["delta"],
            "bootstrap": controlled["bootstrap"],
            "gates": controlled["promotion_gates"],
        },
        "external_exact_onset": {
            "status": external["status"],
            "examples": external["examples"],
            "prior": external["prior"],
            "ensemble": external["ensemble"],
            "delta": external["delta"],
            "bootstrap": external["bootstrap"],
            "external_performance_index": external[
                "external_performance_index"
            ],
            "gates": external["promotion_gates"],
        },
        "frozen_natural_transfer": {
            "status": natural["status"],
            "videos": natural["videos"],
            "events": natural["ground_truth_events"],
            "clap_multiscale_map": natural["methods"]["clap_multiscale"][
                "mean_map"
            ],
            "qcr_multiscale_map": natural["methods"]["qcr_multiscale"][
                "mean_map"
            ],
            "qcr_minus_clap_map": natural_fixed_delta,
            "bootstrap": natural["primary_contrast"][
                "paired_video_bootstrap"
            ],
            "gates": natural["promotion_gates"],
        },
        "cap_corrected_diagnosis": {
            "status": router["status"],
            "claim_type": router["claim_type"],
            "full_router_map": router["primary"]["router_metrics"]["mean_map"],
            "clap_only_router_map": router["clap_only_ablation"]["metrics"][
                "mean_map"
            ],
            "fixed_clap_map": router["primary"]["comparator_metrics"][
                "mean_map"
            ],
            "full_minus_fixed_clap": full_delta,
            "clap_only_minus_fixed_clap": clap_only_delta,
            "full_minus_clap_only": full_delta - clap_only_delta,
            "fraction_total_gain_from_incremental_qcr": (
                (full_delta - clap_only_delta) / full_delta
            ),
            "qcr_assignment_fraction": router["route_summary"][
                "all_candidates"
            ]["qcr_assignment_fraction"],
            "classes_unanimous_across_folds": router["route_summary"][
                "all_candidates"
            ]["classes_unanimous_across_folds"],
            "cap_audit": router["cap_audit"],
            "diagnostic_criteria": router["diagnostic_criteria"],
            "paired_video_bootstrap_vs_fixed_clap": router["primary"][
                "paired_video_bootstrap"
            ],
            "incremental_qcr_posthoc": incremental,
        },
        "supervised_context": {
            "name": "official supervised ActionFormer audio",
            "mean_map": supervised["mean_map"],
            "gap_above_full_router": float(supervised["mean_map"])
            - float(router["primary"]["router_metrics"]["mean_map"]),
            "role": "context_only_not_a_promotion_gate",
        },
        "downstream_answer_bridge": {
            "status": downstream["status"],
            "selected_exact": downstream["combined_metrics"][
                "selected_learned_retrieval"
            ]["two_backbone_mean_source_macro_exact"],
            "clap_exact": downstream["combined_metrics"]["clap_retrieval"][
                "two_backbone_mean_source_macro_exact"
            ],
            "strongest_baseline": downstream[
                "strongest_eligible_nonoracle_baseline"
            ],
            "strongest_exact": downstream["combined_metrics"][
                downstream["strongest_eligible_nonoracle_baseline"]
            ]["two_backbone_mean_source_macro_exact"],
            "gates": downstream["promotion_gates"],
            "audita_status": downstream["audita_status"],
        },
        "figures": figure_records,
        "source_artifact_sha256": {
            path.relative_to(PROJECT).as_posix(): sha256(path)
            for path in (
                CONTROLLED,
                EXTERNAL,
                NATURAL,
                NATURAL_PUBLICATION,
                ROUTER,
                DOWNSTREAM,
                TRACK / "CAPCORRECT_ANALYSIS_LOCK.json",
                TRACK / "CAPCORRECT_ANALYZER_AMENDMENT_LOCK.json",
                TRACK / "results/candidate_pools/receipt.json",
            )
        },
        "invalidated_predecessor": "q1_upgrade/INVALIDATION_OUTPUT_CAP.md",
        "claim_boundary": (
            "The controlled QCR ranker is confirmed for exact-onset evidence "
            "ranking but fails frozen natural zero-shot transfer. The positive "
            "cap-corrected cross-fit result is post-outcome in-domain diagnosis; "
            "it is neither fresh confirmation nor a zero-shot result."
        ),
    }
    ANALYSIS.write_text(
        json.dumps(synthesis, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": synthesis["status"],
                "natural_fixed_qcr_delta": natural_fixed_delta,
                "full_router_map": synthesis["cap_corrected_diagnosis"][
                    "full_router_map"
                ],
                "clap_only_router_map": synthesis[
                    "cap_corrected_diagnosis"
                ]["clap_only_router_map"],
                "full_minus_clap_only": synthesis[
                    "cap_corrected_diagnosis"
                ]["full_minus_clap_only"],
                "incremental_qcr_bootstrap": incremental[
                    "paired_video_bootstrap"
                ],
                "figures": len(figure_records),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
