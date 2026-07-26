#!/usr/bin/env python3
"""Build deterministic, source-linked figures for the Q1-90 Audio replication."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
TRACK = PROJECT / "q1_90"
REPORT = TRACK / "results/external_replication/evaluation/external_replication_report.json"
PRIOR = PROJECT / "q1_plus/results/confirmatory/event_ranker/five_seed_confirmatory_report.json"
OUT = TRACK / "figures_external"
PDF = OUT / "pdf"
PNG = OUT / "png"
SOURCE = OUT / "source_data"
INDEX = OUT / "figure_index.csv"
AUDIT = OUT / "figure_audit.json"

BLUE = "#2166AC"
ORANGE = "#D6604D"
GREEN = "#1B9E77"
PURPLE = "#7570B3"
GREY = "#666666"
FIXED_DATE = datetime(2026, 7, 23, tzinfo=timezone.utc)


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
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
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
        section: str,
        rows: list[dict[str, Any]],
        figure: plt.Figure,
    ) -> None:
        stem = f"{identifier}_{slug}"
        csv_path = SOURCE / f"{stem}.csv"
        fields = sorted({key for row in rows for key in row})
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        pdf_path = PDF / f"{stem}.pdf"
        png_path = PNG / f"{stem}.png"
        figure.savefig(
            pdf_path,
            bbox_inches="tight",
            metadata={"CreationDate": FIXED_DATE, "ModDate": FIXED_DATE},
        )
        figure.savefig(png_path, bbox_inches="tight", metadata={"Software": "matplotlib"})
        plt.close(figure)
        self.records.append(
            {
                "id": identifier,
                "title": title,
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

    def finish(self, report: dict[str, Any]) -> None:
        fields = [
            "id",
            "title",
            "section",
            "pdf",
            "png",
            "source_data",
            "source_rows",
            "pdf_sha256",
            "png_sha256",
            "source_sha256",
        ]
        with INDEX.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.records)
        payload = {
            "status": "q1_90_external_figures_complete",
            "report_sha256": sha256(REPORT),
            "report_status": report["status"],
            "figures": len(self.records),
            "figure_index_sha256": sha256(INDEX),
        }
        AUDIT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cross_corpus(builder: Builder, report: dict[str, Any], prior: dict[str, Any]) -> None:
    rows = []
    for corpus, current in (("ESC-50 confirmatory", prior), ("UrbanSound8K external", report)):
        for metric, label in (("evidence_ap", "Evidence AP"), ("hit_at_1", "Hit@1")):
            rows.extend(
                [
                    {"corpus": corpus, "metric": label, "system": "CLAP prior", "value": current["prior"][metric]},
                    {"corpus": corpus, "metric": label, "system": "Residual ensemble", "value": current["ensemble"][metric]},
                ]
            )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for ax, metric in zip(axes, ("Evidence AP", "Hit@1"), strict=True):
        subset = [row for row in rows if row["metric"] == metric]
        corpora = ["ESC-50 confirmatory", "UrbanSound8K external"]
        x = np.arange(len(corpora))
        prior_values = [next(row["value"] for row in subset if row["corpus"] == corpus and row["system"] == "CLAP prior") for corpus in corpora]
        model_values = [next(row["value"] for row in subset if row["corpus"] == corpus and row["system"] == "Residual ensemble") for corpus in corpora]
        ax.bar(x - 0.19, prior_values, 0.38, color=GREY, label="CLAP prior")
        ax.bar(x + 0.19, model_values, 0.38, color=BLUE, label="Residual ensemble")
        ax.set_xticks(x, ["ESC-50", "UrbanSound8K"])
        ax.set_title(metric, fontweight="bold")
        ax.set_ylim(0.0, 1.02)
    axes[0].set_ylabel("Recipe-level score")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Source-isolated cross-corpus exact-onset transfer", x=0.06, ha="left", fontweight="bold")
    builder.save("A90F01", "cross_corpus", "Cross-corpus localization outcomes", "external replication", rows, fig)


def external_intervals(builder: Builder, report: dict[str, Any]) -> None:
    intervals = report["bootstrap"]["interval_95_percentile"]
    rows = [
        {
            "metric": "Evidence AP",
            "delta": report["delta"]["evidence_ap"],
            "lower": intervals["evidence_ap"][0],
            "upper": intervals["evidence_ap"][1],
            "minimum_effect_gate": 0.05,
        },
        {
            "metric": "Hit@1",
            "delta": report["delta"]["hit_at_1"],
            "lower": intervals["hit_at_1"][0],
            "upper": intervals["hit_at_1"][1],
            "minimum_effect_gate": 0.05,
        },
    ]
    fig, ax = plt.subplots(figsize=(5.8, 2.8))
    y = np.arange(len(rows))
    points = np.asarray([row["delta"] for row in rows], dtype=float)
    lower = points - np.asarray([row["lower"] for row in rows], dtype=float)
    upper = np.asarray([row["upper"] for row in rows], dtype=float) - points
    ax.errorbar(points, y, xerr=np.vstack([lower, upper]), fmt="o", color=BLUE, capsize=4)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.axvline(0.05, color=GREY, linestyle="--", linewidth=1)
    ax.set_yticks(y, [row["metric"] for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Residual-ensemble minus CLAP score (95% class/source bootstrap)")
    ax.set_title("Frozen UrbanSound8K paired effects", loc="left", fontweight="bold")
    builder.save("A90F02", "external_intervals", "External paired effect intervals", "primary outcome", rows, fig)


def class_deltas(builder: Builder, report: dict[str, Any]) -> None:
    rows = [
        {
            "class": label,
            "examples": group["examples"],
            "ap_delta": group["delta"]["evidence_ap"],
            "hit_delta": group["delta"]["hit_at_1"],
        }
        for label, group in report["classes"].items()
    ]
    rows.sort(key=lambda row: row["hit_delta"])
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    y = np.arange(len(rows))
    ax.barh(y, [row["hit_delta"] for row in rows], color=[GREEN if row["hit_delta"] >= -0.05 else ORANGE for row in rows])
    ax.axvline(0.0, color="black", linewidth=1)
    ax.axvline(-0.05, color=GREY, linestyle="--", linewidth=1)
    ax.set_yticks(y, [str(row["class"]).replace("_", " ").title() for row in rows])
    ax.set_xlabel("Hit@1 difference from CLAP")
    ax.set_title("External class-level boundary conditions", loc="left", fontweight="bold")
    builder.save("A90F03", "class_deltas", "External class-level Hit@1 effects", "robustness", rows, fig)


def condition_deltas(builder: Builder, report: dict[str, Any]) -> None:
    rows = []
    family_names = {"duration_sec": "Duration", "position_bin": "Position", "snr_db": "SNR"}
    level_order = {
        "duration_sec": ("60", "180", "300"),
        "position_bin": ("early", "middle", "late"),
        "snr_db": ("-5", "0", "5"),
    }
    for family in ("duration_sec", "position_bin", "snr_db"):
        groups = report["conditions"][family]
        if set(groups) != set(level_order[family]):
            raise RuntimeError(f"unexpected external condition levels: {family}")
        for level in level_order[family]:
            group = groups[level]
            rows.append(
                {
                    "family": family_names[family],
                    "level": str(level),
                    "label": f"{family_names[family]}: {level}",
                    "examples": group["examples"],
                    "ap_delta": group["delta"]["evidence_ap"],
                    "hit_delta": group["delta"]["hit_at_1"],
                }
            )
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    x = np.arange(len(rows))
    width = 0.38
    ax.bar(x - width / 2, [row["ap_delta"] for row in rows], width, color=BLUE, label="Evidence AP")
    ax.bar(x + width / 2, [row["hit_delta"] for row in rows], width, color=PURPLE, label="Hit@1")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x, [row["label"] for row in rows], rotation=32, ha="right")
    ax.set_ylabel("Difference from CLAP")
    ax.legend(frameon=False, ncol=2)
    ax.set_title("Predeclared external condition effects", loc="left", fontweight="bold")
    builder.save("A90F04", "condition_deltas", "External condition effects", "boundary conditions", rows, fig)


def gates(builder: Builder, report: dict[str, Any]) -> None:
    labels = {
        "ensemble_ap_gain_at_least_0_05": "AP gain ≥ 0.05",
        "ensemble_hit_gain_at_least_0_05": "Hit gain ≥ 0.05",
        "bootstrap_ap_lower_above_zero": "AP lower > 0",
        "bootstrap_hit_lower_above_zero": "Hit lower > 0",
        "at_least_four_positive_seeds": "≥4 positive seeds",
        "no_class_hit_delta_below_minus_0_05": "Class floor ≥ -0.05",
        "external_performance_index_at_least_85": "Performance index ≥ 85",
        "integrity": "Integrity",
    }
    rows = [
        {"gate": labels[key], "gate_key": key, "pass": bool(value), "value": 1 if value else 0}
        for key, value in report["promotion_gates"].items()
    ]
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    x = np.arange(len(rows))
    ax.bar(x, [1.0] * len(rows), color=[GREEN if row["pass"] else ORANGE for row in rows])
    ax.set_xticks(x, [row["gate"] for row in rows], rotation=35, ha="right")
    ax.set_yticks([])
    ax.set_ylim(0.0, 1.1)
    ax.set_title("All frozen gates retained regardless of outcome", loc="left", fontweight="bold")
    builder.save("A90F05", "gates", "External replication gates", "decision analysis", rows, fig)


def main() -> None:
    if not REPORT.is_file():
        raise FileNotFoundError("frozen external report does not exist")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    if report["examples"] != 219 or report["integrity"]["external_evaluations"] != 1:
        raise RuntimeError("external report is incomplete")
    setup()
    builder = Builder()
    cross_corpus(builder, report, prior)
    external_intervals(builder, report)
    class_deltas(builder, report)
    condition_deltas(builder, report)
    gates(builder, report)
    builder.finish(report)
    print(json.dumps(json.loads(AUDIT.read_text(encoding="utf-8")), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
