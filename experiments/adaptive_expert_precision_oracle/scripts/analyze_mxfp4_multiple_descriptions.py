#!/usr/bin/env python3
"""Summarize and plot the MXFP4 multiple-description oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cluster_ci(frame: pd.DataFrame, value: str, resamples: int, seed: int) -> tuple[float, float]:
    requests = frame.request_id.unique()
    if not len(requests):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples)
    grouped = {request: frame.loc[frame.request_id == request, value].to_numpy() for request in requests}
    for index in range(resamples):
        chosen = rng.choice(requests, len(requests), replace=True)
        samples[index] = np.median(np.concatenate([grouped[value] for value in chosen]))
    return tuple(np.quantile(samples, (0.025, 0.975)))


def groups(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([
        frame.assign(group=np.where(frame.layer == 0, "layer 0", "layers 4/20/39")),
        frame.assign(group="all four layers"),
    ], ignore_index=True)


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path.with_suffix(".png"), dpi=180)
    plt.savefig(path.with_suffix(".svg"))
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()
    result = args.result
    plots = result / "plots"; plots.mkdir(exist_ok=True)
    metrics = pd.read_parquet(result / "metrics.parquet")
    expanded = groups(metrics)
    summary_rows = []
    for (group, policy, budget), frame in expanded.groupby(["group", "policy", "budget_bpw"]):
        low, high = cluster_ci(frame, "recovery", args.bootstrap, args.seed + int(100 * budget))
        summary_rows.append({
            "group": group, "policy": policy, "budget_bpw": budget, "invocations": len(frame),
            "requests": frame.request_id.nunique(), "p10_recovery": frame.recovery.quantile(.1),
            "median_recovery": frame.recovery.median(), "p90_recovery": frame.recovery.quantile(.9),
            "mean_recovery": frame.recovery.mean(), "median_ci_low": low, "median_ci_high": high,
            "median_logical_bpw": frame.logical_bpw.median(), "median_physical_bpw": frame.physical_bpw.median(),
            "median_page_amplification": frame.page_amplification.median(),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(result / "summary.csv", index=False)

    keys = ["request_id", "position", "layer", "expert_id", "budget_bpw"]
    paired_rows = []
    for left, right in (("bidirectional_ab", "ordered_a"), ("parity_abq", "bidirectional_ab"), ("parity_abq", "ordered_a")):
        l = metrics[metrics.policy == left]; r = metrics[metrics.policy == right]
        paired = l.merge(r, on=keys, suffixes=("_left", "_right"))
        paired["difference"] = paired.recovery_left - paired.recovery_right
        paired = groups(paired)
        for (group, budget), frame in paired.groupby(["group", "budget_bpw"]):
            low, high = cluster_ci(frame, "difference", args.bootstrap, args.seed + 7)
            paired_rows.append({
                "group": group, "left": left, "right": right, "budget_bpw": budget,
                "invocations": len(frame), "p10_difference": frame.difference.quantile(.1),
                "median_difference": frame.difference.median(), "p90_difference": frame.difference.quantile(.9),
                "mean_difference": frame.difference.mean(), "median_ci_low": low, "median_ci_high": high,
            })
    paired_summary = pd.DataFrame(paired_rows)
    paired_summary.to_csv(result / "paired_differences.csv", index=False)

    full = pd.read_parquet(result / "full_support_q3.parquet")
    full_summary = full.groupby(["layer", "projection", "description"]).recovery.agg(
        invocations="size", p10=lambda value: value.quantile(.1), median="median", p90=lambda value: value.quantile(.9), mean="mean",
    ).reset_index()
    full_summary.to_csv(result / "full_support_q3_summary.csv", index=False)

    action_path = result / "selected_description_actions.parquet"
    if action_path.exists():
        actions = pd.read_parquet(action_path)
    else:
        actions = pd.concat([pd.read_parquet(path) for path in sorted((result / "selected_description_actions").glob("*.parquet"))], ignore_index=True)
    choice_rows = []
    for _, row in actions.iterrows():
        added = json.loads(row.added_descriptions_json)
        before = json.loads(row.before_masks_json)
        for previous, values in zip(before, added):
            choice_rows.append({
                "policy": row.policy, "projection": row.projection, "budget_bpw": row.budget_bpw,
                "transition": ("base_to_" if previous == 0 else "q3_to_exact_via_") + "+".join(values),
                "bits": len(values),
            })
    choices = pd.DataFrame(choice_rows)
    choice_summary = choices.groupby(["policy", "projection", "budget_bpw", "transition"]).size().rename("actions").reset_index()
    totals = choice_summary.groupby(["policy", "projection", "budget_bpw"]).actions.transform("sum")
    choice_summary["fraction"] = choice_summary.actions / totals
    choice_summary.to_csv(result / "description_choice_summary.csv", index=False)

    diagnostics = pd.read_parquet(result / "page_diagnostics.parquet")
    page_summary = diagnostics.groupby(["policy", "page_size", "budget_bpw"]).agg(
        invocations=("recovery", "size"), median_recovery=("recovery", "median"),
        median_physical_bpw=("physical_bpw", "median"), median_amplification=("page_amplification", "median"),
    ).reset_index()
    page_summary.to_csv(result / "page_summary.csv", index=False)

    difficult = summary[summary.group == "layers 4/20/39"]
    plt.figure(figsize=(7, 4.5))
    for policy, frame in difficult.groupby("policy"):
        frame = frame.sort_values("budget_bpw")
        plt.plot(frame.budget_bpw, 100 * frame.median_recovery, marker="o", label=policy)
        plt.fill_between(frame.budget_bpw, 100 * frame.p10_recovery, 100 * frame.p90_recovery, alpha=.12)
    plt.xlabel("Physical correction bpw (512 B pages)"); plt.ylabel("Complete-expert proxy recovery (%)")
    plt.legend(); plt.grid(alpha=.25); savefig(plots / "01_multiple_description_frontier")

    plt.figure(figsize=(7, 4.5))
    gains = paired_summary[(paired_summary.group == "layers 4/20/39") & (paired_summary.right != "ordered_a")]
    for (left, right), frame in gains.groupby(["left", "right"]):
        plt.plot(frame.budget_bpw, 100 * frame.median_difference, marker="o", label=f"{left} − {right}")
    plt.axhline(0, color="black", linewidth=.8); plt.xlabel("Physical correction bpw"); plt.ylabel("Median recovery gain (points)")
    plt.legend(); plt.grid(alpha=.25); savefig(plots / "02_incremental_codec_gain")

    pivot = full_summary.groupby(["projection", "description"])["median"].mean().unstack()
    pivot.plot(kind="bar", figsize=(7, 4.5)); plt.ylabel("Mean layer-median full-support Q3 recovery")
    plt.xticks(rotation=0); plt.grid(axis="y", alpha=.25); savefig(plots / "03_full_support_partitions")

    selected_choices = choice_summary[(choice_summary.policy == "parity_abq") & (choice_summary.budget_bpw == 1.0) & choice_summary.transition.str.startswith("base_to_")]
    pivot = selected_choices.pivot_table(index="projection", columns="transition", values="fraction", fill_value=0)
    pivot.plot(kind="bar", stacked=True, figsize=(7, 4.5)); plt.ylabel("Fraction of first-description actions")
    plt.xticks(rotation=0); plt.legend(fontsize=8); savefig(plots / "04_first_description_mix")

    selected_pages = page_summary[page_summary.budget_bpw == 1.0]
    for policy, frame in selected_pages.groupby("policy"):
        plt.plot(frame.page_size, frame.median_amplification, marker="o", label=policy)
    plt.xscale("log", base=2); plt.xlabel("Page size (bytes)"); plt.ylabel("Median page amplification")
    plt.legend(); plt.grid(alpha=.25); savefig(plots / "05_page_amplification")

    storage = pd.read_csv(result / "storage_accounting.csv")
    for policy, frame in storage.groupby("policy"):
        plt.plot(frame.replicas, frame.external_storage_multiplier, marker="o", label=policy)
    plt.axhline(5, color="red", linestyle="--", label="5× cap"); plt.xlabel("Physical layout replicas")
    plt.ylabel("External storage / exact reference"); plt.legend(); plt.grid(alpha=.25)
    savefig(plots / "06_storage_multiplier")

    manifest = {
        "metrics_rows": len(metrics), "action_rows": len(actions), "full_support_rows": len(full),
        "bootstrap_resamples": args.bootstrap, "baseline_path": str(args.baseline) if args.baseline else None,
        "note": "All method comparisons are paired on invocation IDs. Reused exact-capture results are exploratory for this post-hoc hypothesis.",
    }
    (result / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
