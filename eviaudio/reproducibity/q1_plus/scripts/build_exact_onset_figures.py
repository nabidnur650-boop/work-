#!/usr/bin/env python3
"""Build source-linked publication figures for the exact-onset confirmatory pass."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
Q1 = PROJECT / "q1_plus"
REPORT = Q1 / "results/confirmatory/event_ranker/five_seed_confirmatory_report.json"
RAW = Q1 / "results/confirmatory/event_ranker/raw_five_seed_confirmatory.jsonl.gz"
AUTHORIZATION = Q1 / "EVENT_CONFIRMATORY_AUTHORIZATION.json"
OUT = Q1 / "figures_exact_onset"
PDF, PNG, SOURCE = OUT / "pdf", OUT / "png", OUT / "source_data"
INDEX, AUDIT = OUT / "figure_index.csv", OUT / "figure_audit.json"
METRICS = ("evidence_ap", "hit_at_1", "recall_at_4", "top_chunk_iou")
METRIC_LABELS = {
    "evidence_ap": "Evidence AP",
    "hit_at_1": "Hit@1",
    "recall_at_4": "Recall@4",
    "top_chunk_iou": "Top-chunk IoU",
}
BLUE, ORANGE, GREEN, PURPLE, GREY = "#2166AC", "#D6604D", "#1B9E77", "#7570B3", "#666666"


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

    def save(self, identifier: str, slug: str, title: str, rows: list[dict[str, Any]], fig: plt.Figure, section: str) -> None:
        stem = f"{identifier}_{slug}"
        csv_path, pdf_path, png_path = SOURCE / f"{stem}.csv", PDF / f"{stem}.pdf", PNG / f"{stem}.png"
        fields = sorted({key for row in rows for key in row})
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, bbox_inches="tight")
        plt.close(fig)
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


def bar(rows: list[dict[str, Any]], category: str, value: str, title: str, ylabel: str, *, horizontal: bool = False, reference: float | None = None, colors: list[str] | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.7, max(3.8, 0.22 * len(rows)) if horizontal else 3.8))
    names, values = [str(row[category]) for row in rows], [float(row[value]) for row in rows]
    palette = colors or [BLUE] * len(rows)
    x = np.arange(len(rows))
    if horizontal:
        ax.barh(x, values, color=palette); ax.set_yticks(x, names); ax.invert_yaxis(); ax.set_xlabel(ylabel)
        if reference is not None: ax.axvline(reference, color="black", linestyle="--")
    else:
        ax.bar(x, values, color=palette); ax.set_xticks(x, names, rotation=24, ha="right"); ax.set_ylabel(ylabel)
        if reference is not None: ax.axhline(reference, color="black", linestyle="--")
    ax.set_title(title, loc="left", fontweight="bold"); ax.grid(axis="x" if horizontal else "y", alpha=.2)
    return fig


def grouped(rows: list[dict[str, Any]], group: str, series: str, value: str, title: str, ylabel: str, reference: float | None = None) -> plt.Figure:
    groups = list(dict.fromkeys(str(row[group]) for row in rows)); series_names = list(dict.fromkeys(str(row[series]) for row in rows))
    lookup = {(str(row[group]), str(row[series])): float(row[value]) for row in rows}
    fig, ax = plt.subplots(figsize=(7.2, 4.0)); x = np.arange(len(groups)); width = .78 / len(series_names)
    for index, name in enumerate(series_names):
        ax.bar(x + (index - (len(series_names)-1)/2)*width, [lookup[(g,name)] for g in groups], width, label=name, color=[GREY, BLUE, ORANGE, GREEN][index])
    ax.set_xticks(x, groups, rotation=22, ha="right"); ax.set_ylabel(ylabel); ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2)
    if reference is not None: ax.axhline(reference, color="black", linestyle="--")
    return fig


def condition_rows(report: dict[str, Any], condition: str, metric: str) -> list[dict[str, Any]]:
    order = sorted(report["conditions"][condition], key=lambda value: float(value) if value.replace("-", "").isdigit() else value)
    return [
        {"condition": value, "system": system.title(), "value": report["conditions"][condition][value][system][metric], "examples": report["conditions"][condition][value]["examples"]}
        for value in order
        for system in ("prior", "ensemble")
    ]


def class_aggregate(raw: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw: grouped_rows[row["class_label"]].append(row)
    return sorted(
        [
            {
                "class_label": name,
                "examples": len(items),
                "prior": float(np.mean([row["prior_metrics"][metric] for row in items])),
                "ensemble": float(np.mean([row["ensemble_metrics"][metric] for row in items])),
                "delta": float(np.mean([row["ensemble_metrics"][metric] - row["prior_metrics"][metric] for row in items])),
            }
            for name, items in grouped_rows.items()
        ],
        key=lambda row: row["delta"],
    )


def main() -> None:
    if INDEX.exists() or AUDIT.exists():
        raise FileExistsError("exact-onset figure suite is immutable once indexed")
    setup()
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    with gzip.open(RAW, "rt", encoding="utf-8") as handle:
        raw = [json.loads(line) for line in handle if line.strip()]
    if report["status"] != "confirmatory_exact_onset_gate_pass" or not report["all_promotion_gates_pass"] or len(raw) != 397:
        raise RuntimeError("exact-onset confirmatory inputs failed status audit")
    builder = Builder()

    protocol = [
        {"stage": "Isolated sources", "order": 1, "passed": 1},
        {"stage": "Frozen five-seed ensemble", "order": 2, "passed": 1},
        {"stage": "397-example test", "order": 3, "passed": 1},
        {"stage": "Promotion gates", "order": 4, "passed": 1},
    ]
    fig, ax = plt.subplots(figsize=(7.2, 2.4)); ax.plot([1,2,3,4],[0]*4,color=GREY,linewidth=2)
    for row in protocol: ax.scatter(row["order"],0,s=150,color=GREEN); ax.text(row["order"],.1,row["stage"],ha="center",fontsize=8)
    ax.set(yticks=[], xticks=[], ylim=(-.15,.3)); ax.set_title("Leakage-isolated exact-onset confirmatory path", loc="left", fontweight="bold")
    builder.save("E01","protocol","Exact-onset protocol",protocol,fig,"main")

    overall = [{"metric": METRIC_LABELS[metric], "system": system.title(), "value": report[system][metric]} for metric in METRICS for system in ("prior","ensemble")]
    builder.save("E02","overall_metrics","Overall retrieval metrics",overall,grouped(overall,"metric","system","value","Confirmatory exact-onset retrieval","Score"),"main")
    deltas = [{"metric": METRIC_LABELS[metric], "delta": report["delta"][metric]} for metric in METRICS]
    builder.save("E03","overall_deltas","Overall metric gains",deltas,bar(deltas,"metric","delta","Absolute gain over frozen CLAP prior","Ensemble − prior",reference=0),"main")

    ci = [
        {"metric": "Evidence AP", "delta": report["delta"]["evidence_ap"], "lower": report["bootstrap"]["interval_95_percentile"]["evidence_ap"][0], "upper": report["bootstrap"]["interval_95_percentile"]["evidence_ap"][1]},
        {"metric": "Hit@1", "delta": report["delta"]["hit_at_1"], "lower": report["bootstrap"]["interval_95_percentile"]["hit_at_1"][0], "upper": report["bootstrap"]["interval_95_percentile"]["hit_at_1"][1]},
    ]
    fig, ax = plt.subplots(figsize=(6.2,3.1))
    for i,row in enumerate(ci): ax.errorbar(row["delta"],i,xerr=[[row["delta"]-row["lower"]],[row["upper"]-row["delta"]]],fmt="o",color=[BLUE,ORANGE][i],capsize=4)
    ax.axvline(0,color="black",linestyle="--"); ax.set_yticks(range(2),[row["metric"] for row in ci]); ax.set_xlabel("Paired gain (95% bootstrap interval)"); ax.set_title("Clustered uncertainty excludes zero",loc="left",fontweight="bold")
    builder.save("E04","bootstrap_forest","Bootstrap intervals",ci,fig,"main")

    gates = [{"gate": key.replace("_"," "), "passed": int(value)} for key,value in report["promotion_gates"].items()]
    builder.save("E05","promotion_gates","Promotion gates",gates,bar(gates,"gate","passed","All exact-onset promotion gates passed","Passed",horizontal=True,colors=[GREEN]*len(gates)),"main")

    seed_rows = [{"seed": str(item["seed"]), "metric": metric, "model": item["model"][metric], "delta": item["delta"][metric]} for item in report["per_seed"] for metric in METRICS]
    for offset, metric in enumerate(METRICS, start=6):
        rows = [dict(row, seed=f"Seed {row['seed']}") for row in seed_rows if row["metric"] == metric]
        builder.save(f"E{offset:02d}",f"seed_{metric}",f"Per-seed {METRIC_LABELS[metric]}",rows,bar(rows,"seed","model",f"Per-seed {METRIC_LABELS[metric]}",METRIC_LABELS[metric]),"seed stability")
    for offset, metric in enumerate(METRICS, start=10):
        rows = [dict(row, seed=f"Seed {row['seed']}") for row in seed_rows if row["metric"] == metric]
        builder.save(f"E{offset:02d}",f"seed_delta_{metric}",f"Per-seed {METRIC_LABELS[metric]} gain",rows,bar(rows,"seed","delta",f"Per-seed {METRIC_LABELS[metric]} gain","Model − prior",reference=0),"seed stability")

    figure_id = 14
    for condition, display in (("duration_sec","Duration (s)"),("position_bin","Target position"),("snr_db","SNR (dB)")):
        for metric in METRICS:
            rows = condition_rows(report, condition, metric)
            builder.save(f"E{figure_id:02d}",f"{condition}_{metric}",f"{METRIC_LABELS[metric]} by {display}",rows,grouped(rows,"condition","system","value",f"{METRIC_LABELS[metric]} by {display}",METRIC_LABELS[metric]),"condition robustness")
            figure_id += 1

    example_rows = [
        {
            "recipe_id": row["recipe_id"], "class_label": row["class_label"], "duration_sec": row["duration_sec"], "snr_db": row["snr_db"], "position_bin": row["position_bin"],
            **{f"prior_{metric}": row["prior_metrics"][metric] for metric in METRICS},
            **{f"ensemble_{metric}": row["ensemble_metrics"][metric] for metric in METRICS},
            **{f"delta_{metric}": row["ensemble_metrics"][metric]-row["prior_metrics"][metric] for metric in METRICS},
            "mean_absolute_residual": row["mean_absolute_residual"], "n_chunks": row["n_chunks"], "onset_fraction": row["target_start_sec"]/row["duration_sec"],
        }
        for row in raw
    ]
    for identifier, metric, title in (("E26","evidence_ap","Per-example AP transfer"),("E27","top_chunk_iou","Per-example IoU transfer")):
        fig, ax = plt.subplots(figsize=(4.7,4.4)); ax.scatter([row[f"prior_{metric}"] for row in example_rows],[row[f"ensemble_{metric}"] for row in example_rows],s=16,alpha=.55,color=BLUE)
        ax.plot([0,1],[0,1],color="black",linestyle="--"); ax.set(xlabel=f"Prior {METRIC_LABELS[metric]}",ylabel=f"Ensemble {METRIC_LABELS[metric]}",xlim=(-.03,1.03),ylim=(-.03,1.03)); ax.set_title(title,loc="left",fontweight="bold")
        builder.save(identifier,f"{metric}_scatter",title,example_rows,fig,"main")

    for identifier, metric in (("E28","evidence_ap"),("E29","hit_at_1"),("E30","recall_at_4"),("E31","top_chunk_iou")):
        fig, ax = plt.subplots(figsize=(6.2,3.7)); values=[row[f"delta_{metric}"] for row in example_rows]; ax.hist(values,bins=25,color=BLUE,alpha=.8); ax.axvline(0,color="black",linestyle="--"); ax.set(xlabel=f"{METRIC_LABELS[metric]} gain",ylabel="Examples"); ax.set_title(f"Per-example {METRIC_LABELS[metric]} gain distribution",loc="left",fontweight="bold")
        builder.save(identifier,f"delta_hist_{metric}",f"{METRIC_LABELS[metric]} gain distribution",example_rows,fig,"supplement")

    fig, ax = plt.subplots(figsize=(6.1,3.7)); values=np.sort([row["delta_evidence_ap"] for row in example_rows]); ax.step(values,np.arange(1,len(values)+1)/len(values),where="post",color=BLUE); ax.axvline(0,color="black",linestyle="--"); ax.set(xlabel="Evidence AP gain",ylabel="Empirical cumulative probability"); ax.set_title("Evidence-AP gain ECDF",loc="left",fontweight="bold")
    builder.save("E32","ap_delta_ecdf","Evidence-AP gain ECDF",example_rows,fig,"supplement")

    for identifier, metric, title in (("E33","hit_at_1","Hit@1 transition"),("E34","recall_at_4","Recall@4 transition")):
        transitions = Counter((int(row[f"prior_{metric}"]),int(row[f"ensemble_{metric}"])) for row in example_rows)
        rows=[{"transition":f"{a}→{b}","examples":transitions[(a,b)]} for a in (0,1) for b in (0,1)]
        builder.save(identifier,f"{metric}_transitions",title,rows,bar(rows,"transition","examples",title,"Examples",colors=[GREY,GREEN,ORANGE,BLUE]),"main")

    residual_rows=[{"recipe_id":row["recipe_id"],"mean_absolute_residual":row["mean_absolute_residual"],"maximum_absolute_residual":row["maximum_absolute_residual"],"delta_evidence_ap":row["ensemble_metrics"]["evidence_ap"]-row["prior_metrics"]["evidence_ap"]} for row in raw]
    fig, ax = plt.subplots(figsize=(6.1,3.7)); ax.hist([row["mean_absolute_residual"] for row in residual_rows],bins=24,color=PURPLE); ax.axvline(report["mean_absolute_ensemble_residual"],color="black",linestyle="--"); ax.set(xlabel="Mean absolute residual",ylabel="Examples"); ax.set_title("Bounded residual utilization",loc="left",fontweight="bold")
    builder.save("E35","residual_distribution","Residual distribution",residual_rows,fig,"mechanism")
    fig, ax = plt.subplots(figsize=(5.3,4.1)); ax.scatter([row["mean_absolute_residual"] for row in residual_rows],[row["delta_evidence_ap"] for row in residual_rows],s=16,alpha=.55,color=PURPLE); ax.axhline(0,color="black",linestyle="--"); ax.set(xlabel="Mean absolute residual",ylabel="Evidence AP gain"); ax.set_title("Residual magnitude and retrieval gain",loc="left",fontweight="bold")
    builder.save("E36","residual_gain","Residual magnitude versus gain",residual_rows,fig,"mechanism")

    duration_rows=[{"duration_sec":str(duration),"n_chunks":int(np.median([row["n_chunks"] for row in raw if row["duration_sec"]==duration])),"examples":sum(row["duration_sec"]==duration for row in raw)} for duration in sorted({row["duration_sec"] for row in raw})]
    builder.save("E37","stream_inventory","Stream inventory",duration_rows,bar(duration_rows,"duration_sec","n_chunks","Bounded sliding-window inventory","Chunks per example"),"reproducibility")

    for identifier, metric in (("E38","evidence_ap"),("E39","hit_at_1")):
        rows=class_aggregate(raw,metric); plotrows=[{"class_label":row["class_label"],"delta":row["delta"],"examples":row["examples"]} for row in rows]
        builder.save(identifier,f"class_delta_{metric}",f"Class-wise {METRIC_LABELS[metric]} gain",plotrows,bar(plotrows,"class_label","delta",f"Class-wise {METRIC_LABELS[metric]} gain","Ensemble − prior",horizontal=True,reference=0,colors=[ORANGE if row["delta"]<0 else GREEN for row in plotrows]),"condition robustness")

    score_rows=[]
    for row in raw:
        for index,label_value in enumerate(row["evidence_targets"]):
            score_rows.append({"recipe_id":row["recipe_id"],"target":int(label_value),"system":"Prior","score":row["prior_chunk_scores"][index]})
            score_rows.append({"recipe_id":row["recipe_id"],"target":int(label_value),"system":"Ensemble","score":row["ensemble_chunk_scores"][index]})
    summary=[]
    for system in ("Prior","Ensemble"):
        for target in (0,1):
            values=[row["score"] for row in score_rows if row["system"]==system and row["target"]==target]
            summary.append({"system":system,"target":"Evidence" if target else "Non-evidence","mean_score":float(np.mean(values)),"chunks":len(values)})
    builder.save("E40","score_separation","Chunk-score separation",summary,grouped(summary,"system","target","mean_score","Evidence/non-evidence score separation","Mean score"),"mechanism")

    seed_variability=[]
    for row in raw:
        ap=[item["evidence_ap"] for item in row["seed_metrics"]]; hits=[item["hit_at_1"] for item in row["seed_metrics"]]
        seed_variability.append({"recipe_id":row["recipe_id"],"seed_ap_sd":float(np.std(ap)),"seeds_hitting_at_1":int(sum(hits))})
    fig, ax = plt.subplots(figsize=(6.1,3.7)); ax.hist([row["seed_ap_sd"] for row in seed_variability],bins=22,color=BLUE); ax.set(xlabel="Within-example AP standard deviation across five seeds",ylabel="Examples"); ax.set_title("Per-example seed variability",loc="left",fontweight="bold")
    builder.save("E41","seed_variability","Per-example seed variability",seed_variability,fig,"seed stability")
    agreement=Counter(row["seeds_hitting_at_1"] for row in seed_variability); agreement_rows=[{"seeds_hitting_at_1":str(value),"examples":agreement[value]} for value in range(6)]
    builder.save("E42","seed_hit_agreement","Seed Hit@1 agreement",agreement_rows,bar(agreement_rows,"seeds_hitting_at_1","examples","How many seeds hit the event at rank one?","Examples"),"seed stability")

    integrity=[
        {"check":"Examples expected","value":report["integrity"]["examples_expected"]},
        {"check":"Examples observed","value":report["integrity"]["examples_observed"]},
        {"check":"Archives verified","value":report["integrity"]["archives_verified"]},
        {"check":"Unique recipes","value":report["integrity"]["unique_recipe_ids"]},
        {"check":"Unique source files","value":report["integrity"]["unique_source_files_verified"]},
        {"check":"Chunks verified","value":report["integrity"]["total_chunks_verified"]},
        {"check":"All alignment/checksum failures","value":sum(report["integrity"][key] for key in ("alignment_failures","archive_checksum_failures","archive_schema_failures","chunk_alignment_failures","evidence_alignment_failures","source_isolation_failures"))},
    ]
    builder.save("E43","integrity_inventory","Integrity inventory",integrity,bar(integrity,"check","value","Confirmatory integrity inventory","Count",horizontal=True),"reproducibility")

    with INDEX.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(builder.records[0])); writer.writeheader(); writer.writerows(builder.records)
    audit={
        "status":"exact_onset_publication_figures_complete",
        "distinct_figures":len(builder.records),"pdf_files":len(list(PDF.glob('*.pdf'))),"png_files":len(list(PNG.glob('*.png'))),"source_data_files":len(list(SOURCE.glob('*.csv'))),
        "report_sha256":sha256(REPORT),"raw_sha256":sha256(RAW),"authorization_sha256":sha256(AUTHORIZATION),"figure_index_sha256":sha256(INDEX),
        "all_numeric_figures_have_source_data":all(row["source_rows"]>0 for row in builder.records),"claim_scope":"exact_onset_evidence_gate_only_not_overall_q1_answer_claim"
    }
    if len(builder.records)<30 or len({row['id'] for row in builder.records})!=len(builder.records): raise RuntimeError("exact-onset figure identity/count gate failed")
    AUDIT.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n",encoding="utf-8"); print(json.dumps(audit,indent=2,sort_keys=True))


if __name__ == "__main__":
    main()
