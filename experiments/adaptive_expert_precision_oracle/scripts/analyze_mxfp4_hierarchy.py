#!/usr/bin/env python3
"""Analyze exact embedded-MXFP4 phases A--C and reproduce every figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DIFFICULT = [4, 20, 39]


def quantiles(values: pd.Series) -> dict[str, float]:
    x = values.to_numpy(float)
    return {
        "count": len(x), "mean": float(np.mean(x)), "p10": float(np.quantile(x, .1)),
        "p25": float(np.quantile(x, .25)), "median": float(np.median(x)),
        "p75": float(np.quantile(x, .75)), "p90": float(np.quantile(x, .9)),
    }


def cluster_ci(frame: pd.DataFrame, column: str, seed: int, resamples: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    clusters = frame.request_id.astype(str).unique()
    values = []
    groups = {key: frame[frame.request_id.astype(str) == key][column].to_numpy(float) for key in clusters}
    for _ in range(resamples):
        sampled = rng.choice(clusters, len(clusters), replace=True)
        values.append(float(np.median(np.concatenate([groups[key] for key in sampled]))))
    return tuple(map(float, np.quantile(values, [.025, .975])))


def dense_frontier(dense: pd.DataFrame, budgets: list[float]) -> pd.DataFrame:
    keys = ["request_id", "sequence_id", "position", "layer", "expert_id"]
    rows = []
    source = dense[dense.split == "test"]
    for budget in budgets:
        eligible = source[source.physical_bpw <= budget + 1e-9]
        best = eligible.sort_values("damage").groupby(keys, as_index=False).first()
        best["budget_bpw"] = budget
        rows.append(best)
    return pd.concat(rows, ignore_index=True)


def savefig(fig: plt.Figure, plots: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(plots / f"{name}.png", dpi=180)
    fig.savefig(plots / f"{name}.svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()
    dense_root = args.result / "dense"; selective_root = args.result / "selective"
    plots = args.result / "plots"; plots.mkdir(parents=True, exist_ok=True)
    dense = pd.read_parquet(dense_root / "dense_metrics.parquet")
    projection = pd.read_parquet(dense_root / "projection_metrics.parquet")
    selective = pd.read_parquet(selective_root / "selective_metrics.parquet")
    layouts = pd.read_parquet(selective_root / "layout_metrics.parquet")
    retention = pd.read_parquet(selective_root / "selective_dense_retention.parquet")
    ranking = pd.read_parquet(selective_root / "ranking_audit.parquet")
    dense_budgets = sorted(dense.physical_bpw.unique())
    frontier = dense_frontier(dense, dense_budgets)
    frontier.to_parquet(args.result / "dense_frontier_per_invocation.parquet", index=False)
    groups = {"all_four": [0, 4, 20, 39], "difficult_4_20_39": DIFFICULT, "layer0": [0]}
    summary_rows = []
    for kind, frame in (("dense", frontier), ("selective", selective)):
        for group, layerset in groups.items():
            subset = frame[frame.layer.isin(layerset)]
            for budget, values in subset.groupby("budget_bpw"):
                row = {"method": kind, "aggregate": group, "budget_bpw": float(budget), **quantiles(values.recovery)}
                row["fraction_ge_80"] = float(np.mean(values.recovery >= .8)); row["fraction_ge_90"] = float(np.mean(values.recovery >= .9))
                row["fraction_ge_95"] = float(np.mean(values.recovery >= .95)); row["fraction_ge_99"] = float(np.mean(values.recovery >= .99))
                row["median_ci_low"], row["median_ci_high"] = cluster_ci(values, "recovery", args.seed + int(1000 * budget), args.bootstrap)
                summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.result / "headline_rate_summary.csv", index=False)
    projection_summary = projection[projection.split == "test"].groupby(["layer", "projection", "embedded_level"]).recovery.apply(lambda x: pd.Series(quantiles(x))).unstack().reset_index()
    projection_summary.to_csv(args.result / "projection_summary.csv", index=False)
    tuple_summary = dense[dense.split == "test"].groupby(["tuple", "physical_bpw"]).recovery.apply(lambda x: pd.Series(quantiles(x))).unstack().reset_index()
    tuple_summary.to_csv(args.result / "dense_tuple_summary.csv", index=False)
    retention_rows = []
    for group, layerset in groups.items():
        for budget, values in retention[retention.layer.isin(layerset)].groupby("budget_bpw"):
            retention_rows.append({"aggregate": group, "budget_bpw": budget, **quantiles(values.retention_eta)})
    pd.DataFrame(retention_rows).to_csv(args.result / "selective_dense_retention_summary.csv", index=False)
    layout_rows = []
    for keys, values in layouts.groupby(["layout_method", "page_size", "paired_planes", "budget_bpw"]):
        layout_rows.append({"layout_method": keys[0], "page_size": keys[1], "paired_planes": keys[2], "budget_bpw": keys[3],
                            **{f"recovery_{k}": v for k, v in quantiles(values.recovery).items()},
                            "median_physical_bpw": float(values.physical_bpw.median()),
                            "median_logical_bpw": float(values.logical_bpw.median()),
                            "median_amplification": float(values.page_amplification.median())})
    pd.DataFrame(layout_rows).to_csv(args.result / "page_layout_summary.csv", index=False)
    ranking.to_csv(args.result / "ranking_audit_summary.csv", index=False)
    # Dense and selective frontiers.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, marker in (("dense", "o"), ("selective", "s")):
        values = summary[(summary.method == method) & (summary.aggregate == "difficult_4_20_39")]
        ax.plot(values.budget_bpw, 100 * values["median"], marker=marker, label=method)
        ax.fill_between(values.budget_bpw, 100 * values.p10, 100 * values.p90, alpha=.15)
    ax.axhline(90, color="grey", ls="--", lw=1); ax.set(xlabel="Physical correction bpw", ylabel="Complete-expert recovery (%)", title="Exact MXFP4 hierarchy: difficult layers")
    ax.legend(); ax.grid(alpha=.25); savefig(fig, plots, "01_dense_vs_selective_frontier")
    # Projection Q3.
    values = projection[(projection.split == "test") & projection.layer.isin(DIFFICULT)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    table = values.groupby(["projection", "embedded_level"]).recovery.median().unstack()
    table.plot(kind="bar", ax=ax); ax.set(ylabel="Median residual recovery", title="Dense embedded levels by projection"); ax.grid(axis="y", alpha=.25)
    savefig(fig, plots, "02_projection_dense_levels")
    # Retention eta.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for group in ("difficult_4_20_39", "all_four"):
        values = pd.DataFrame(retention_rows); values = values[values["aggregate"] == group]
        ax.plot(values.budget_bpw, values["median"], marker="o", label=group)
        ax.fill_between(values.budget_bpw, values.p10, values.p90, alpha=.15)
    ax.axhline(1, color="grey", ls="--"); ax.set(xlabel="Physical correction bpw", ylabel="Selective/dense damage-reduction retention η", title="Selective-vs-dense retention"); ax.legend(); ax.grid(alpha=.25)
    savefig(fig, plots, "03_selective_dense_retention")
    # Logical vs physical.
    fig, ax = plt.subplots(figsize=(6, 5))
    x = selective[selective.layer.isin(DIFFICULT)]
    ax.scatter(x.logical_bpw, x.physical_bpw, s=8, alpha=.2); line=np.linspace(0,2,100); ax.plot(line,line,"k--",lw=1)
    ax.set(xlabel="Logical selected bit bpw", ylabel="Physical bpw", title="Useful selected bits versus page traffic"); ax.grid(alpha=.25)
    savefig(fig, plots, "04_logical_vs_physical")
    # Page size amplification.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    primary_layout = layouts[(layouts.layout_method == "coselection_1rep") & (~layouts.paired_planes)]
    for size, values in primary_layout.groupby("page_size"):
        curve = values.groupby("budget_bpw").page_amplification.median()
        ax.plot(curve.index, curve.values, marker="o", label=f"{size} B")
    ax.set(xlabel="Budget (bpw)", ylabel="Median page amplification", title="Page granularity cost"); ax.legend(); ax.grid(alpha=.25)
    savefig(fig, plots, "05_page_amplification")
    # Recovery by page size.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for size, values in primary_layout.groupby("page_size"):
        curve = values.groupby("budget_bpw").recovery.median()
        ax.plot(curve.index, 100 * curve.values, marker="o", label=f"{size} B")
    ax.set(xlabel="Physical correction bpw", ylabel="Median recovery (%)", title="Recovery versus transfer granularity"); ax.legend(); ax.grid(alpha=.25)
    savefig(fig, plots, "06_recovery_by_page_size")
    # Layout/replicas.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    values = layouts[(layouts.page_size == 512) & (~layouts.paired_planes)]
    for method, frame in values.groupby("layout_method"):
        curve = frame.groupby("budget_bpw").recovery.median(); ax.plot(curve.index, 100 * curve.values, marker="o", label=method)
    ax.set(xlabel="Physical correction bpw", ylabel="Median recovery (%)", title="Training-only layouts and H0 replica ceiling"); ax.legend(fontsize=7); ax.grid(alpha=.25)
    savefig(fig, plots, "07_layout_replica_comparison")
    # Allocation.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    alloc = selective[selective.layer.isin(DIFFICULT)].groupby("budget_bpw")[["gate_rate_target","up_rate_target","down_rate_target"]].median()
    ax.stackplot(alloc.index, alloc.gate_rate_target/3, alloc.up_rate_target/3, alloc.down_rate_target/3, labels=["gate","up","down"])
    ax.set(xlabel="Total physical correction budget (bpw)", ylabel="Median allocated expert-average bpw", title="Exact sequential allocation across projections"); ax.legend(loc="upper left"); ax.grid(alpha=.25)
    savefig(fig, plots, "08_projection_allocation")
    # Ranking regret/support.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    audit = ranking[ranking.threshold == .9]
    labels=[]; positions=[]; values=[]
    for i, ((projection_name, method), frame) in enumerate(audit.groupby(["projection","ranking"])):
        labels.append(f"{projection_name}\n{method}"); positions.append(i); values.append(frame.actions_required.dropna().to_numpy())
    ax.boxplot(values, positions=positions, showfliers=False); ax.set_xticks(positions, labels, rotation=35, ha="right", fontsize=7)
    ax.set(ylabel="Actions required for 90% projection recovery", title="Diagonal ranking versus interaction-aware oracles"); ax.grid(axis="y", alpha=.25)
    savefig(fig, plots, "09_ranking_regret")
    # Failure tails.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for budget in (0.5,1.0,1.5,2.0):
        values=np.sort(selective[(selective.layer.isin(DIFFICULT))&(selective.budget_bpw==budget)].recovery.to_numpy()); ax.plot(values,np.linspace(0,1,len(values)),label=f"{budget} bpw")
    ax.set(xlabel="Complete-expert recovery", ylabel="Empirical CDF", title="Held-out failure-tail distribution"); ax.legend(); ax.grid(alpha=.25)
    savefig(fig, plots, "10_recovery_cdf")
    # Storage multiplier frontier.
    storage = pd.read_csv(selective_root / "storage_accounting.csv")
    fig, ax = plt.subplots(figsize=(6,4.5)); ax.plot(storage.external_storage_multiplier, storage.layout_replicas, marker="o")
    ax.axvline(5,color="red",ls="--"); ax.set(xlabel="External storage multiplier vs exact MXFP4",ylabel="Exact suffix layout replicas",title="Exact suffix replication storage"); ax.grid(alpha=.25)
    savefig(fig, plots, "11_storage_multiplier")
    manifest = {"plots": sorted(path.name for path in plots.iterdir()), "bootstrap_resamples": args.bootstrap,
                "dense_rows": len(dense), "selective_rows": len(selective), "layout_rows": len(layouts),
                "test_invocations": int(selective[["request_id","position","layer","expert_id"]].drop_duplicates().shape[0])}
    (args.result / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
