#!/usr/bin/env python3
"""Summarize and plot the dense RRQ ceiling from saved Parquet metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KEYS = ["request_id", "position", "layer", "expert_id"]
METHOD_LABELS = {
    "rrq_raw_heterogeneous": "RRQ raw, heterogeneous",
    "rrq_raw_homogeneous": "RRQ raw, homogeneous",
    "rrq_raw_heterogeneous_bf16_scales": "RRQ raw, BF16 scales",
    "rrq_raw_heterogeneous_fp32_scales_upper": "RRQ raw, FP32-scale upper",
    "rrq_generalized_heterogeneous": "RRQ transformed G/U",
    "rrq_raw_heterogeneous_weight_only_down": "RRQ raw, weight-only down",
    "nested_int16_prefix_generalized_native": "PR2 nested-prefix control",
    "independent_direct_generalized_native": "Independent direct control",
}


def save_figure(figure: plt.Figure, root: Path, name: str) -> None:
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(root / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def cluster_ci(frame: pd.DataFrame, column: str, statistic: str, resamples: int, seed: int) -> tuple[float, float]:
    if frame.empty:
        return float("nan"), float("nan")
    grouped = [group[column].to_numpy(float) for _, group in frame.groupby("request_id")]
    rng = np.random.default_rng(seed)
    values = np.empty(resamples)
    for index in range(resamples):
        chosen = rng.integers(0, len(grouped), len(grouped))
        sample = np.concatenate([grouped[i] for i in chosen])
        values[index] = np.median(sample) if statistic == "median" else np.mean(sample)
    return float(np.quantile(values, .025)), float(np.quantile(values, .975))


def budget_frontier(data: pd.DataFrame, budgets: list[float]) -> pd.DataFrame:
    rows = []
    for method, method_data in data.groupby("method"):
        for budget in budgets:
            feasible = method_data[method_data["physical_bpw"] <= budget + 1e-10]
            if feasible.empty:
                continue
            indices = feasible.groupby(KEYS, sort=False)["damage"].idxmin()
            chosen = feasible.loc[indices].copy()
            chosen["budget_bpw"] = budget
            rows.append(chosen)
    return pd.concat(rows, ignore_index=True)


def summary_rows(frontier: pd.DataFrame, resamples: int, seed: int) -> pd.DataFrame:
    rows = []
    groups = {
        "layer0": lambda d: d[d.layer == 0],
        "difficult_4_20_39": lambda d: d[d.layer.isin([4, 20, 39])],
        "all_0_4_20_39": lambda d: d,
    }
    for (method, budget), values in frontier.groupby(["method", "budget_bpw"]):
        for group_name, select in groups.items():
            frame = select(values)
            if frame.empty:
                continue
            recovery = frame.recovery.to_numpy(float)
            ci_low, ci_high = cluster_ci(frame, "recovery", "median", resamples, seed + int(budget * 100) + len(method))
            rows.append({
                "method": method, "budget_bpw": budget, "aggregate": group_name, "invocations": len(frame),
                "requests": frame.request_id.nunique(), "mean_recovery": recovery.mean(), "p10_recovery": np.quantile(recovery, .10),
                "p25_recovery": np.quantile(recovery, .25), "median_recovery": np.median(recovery),
                "p75_recovery": np.quantile(recovery, .75), "p90_recovery": np.quantile(recovery, .90),
                "p95_remaining_damage": np.quantile(1 - recovery, .95), "mean_relative_output_error": frame.relative_output_error.mean(),
                "fraction_above_80": np.mean(recovery >= .80), "fraction_above_90": np.mean(recovery >= .90),
                "fraction_above_95": np.mean(recovery >= .95), "fraction_above_99": np.mean(recovery >= .99),
                "median_actual_bpw": frame.physical_bpw.median(), "median_page_amplification": frame.page_amplification.median(),
                "median_bootstrap_ci_low": ci_low, "median_bootstrap_ci_high": ci_high,
            })
    return pd.DataFrame(rows)


def inverse_rates(data: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    for keys, frame in data.groupby(["method", *KEYS], sort=False):
        method, request, position, layer, expert = keys
        for threshold in thresholds:
            passing = frame[frame.recovery >= threshold]
            if passing.empty:
                rate = np.nan; selected_tuple = None
            else:
                item = passing.loc[passing.physical_bpw.idxmin()]
                rate = float(item.physical_bpw); selected_tuple = item["tuple"]
            rows.append({"method": method, "request_id": request, "position": position, "layer": layer, "expert_id": expert, "threshold": threshold, "required_physical_bpw": rate, "tuple": selected_tuple})
    return pd.DataFrame(rows)


def paired_pr2(frontier: pd.DataFrame, prior_path: Path, budgets: list[float], resamples: int, seed: int) -> pd.DataFrame:
    if not prior_path.exists():
        return pd.DataFrame()
    prior = pd.read_parquet(prior_path)
    prior = prior[prior.method == "A6_corrected_screened_greedy"]
    rows = []
    for method in frontier.method.unique():
        for budget in budgets:
            current = frontier[(frontier.method == method) & (frontier.budget_bpw == budget)]
            old = prior[np.isclose(prior.budget_bpw, budget)]
            merged = current.merge(old[[*KEYS, "recovery"]].drop_duplicates(KEYS), on=KEYS, suffixes=("_rrq", "_pr2"))
            if merged.empty:
                continue
            merged["difference"] = merged.recovery_rrq - merged.recovery_pr2
            low, high = cluster_ci(merged, "difference", "median", resamples, seed + int(budget * 100))
            rows.append({"method": method, "budget_bpw": budget, "paired_invocations": len(merged), "median_difference": merged.difference.median(), "mean_difference": merged.difference.mean(), "bootstrap_ci_low": low, "bootstrap_ci_high": high})
    return pd.DataFrame(rows)


def projection_inverse(projection: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    keys = ["method", "projection", "request_id", "position", "layer", "expert_id"]
    for values, frame in projection.groupby(keys, sort=False):
        for threshold in thresholds:
            passing = frame[frame.recovery >= threshold]
            rate = np.nan if passing.empty else passing.physical_bpw.min()
            rows.append(dict(zip(keys, values), threshold=threshold, required_physical_bpw=rate))
    return pd.DataFrame(rows)


def make_plots(result: Path, frontier: pd.DataFrame, summary: pd.DataFrame, inverse: pd.DataFrame, projection: pd.DataFrame, stage: pd.DataFrame, scale: pd.DataFrame, storage: pd.DataFrame, page: pd.DataFrame) -> None:
    plots = result / "plots"; plots.mkdir(exist_ok=True)
    difficult = summary[summary["aggregate"] == "difficult_4_20_39"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, frame in difficult.groupby("method"):
        frame = frame.sort_values("budget_bpw")
        ax.plot(frame.budget_bpw, 100 * frame.median_recovery, marker="o", label=METHOD_LABELS.get(method, method))
        ax.fill_between(frame.budget_bpw, 100 * frame.p10_recovery, 100 * frame.p90_recovery, alpha=.12)
    ax.axhline(90, color="black", linestyle="--", linewidth=.8); ax.set(xlabel="Actual physical correction bpw", ylabel="Complete-expert proxy recovery (%)", title="Dense-prefix rate–distortion, layers 4/20/39")
    ax.legend(fontsize=7); ax.grid(alpha=.25); save_figure(fig, plots, "dense_rate_distortion")

    fig, ax = plt.subplots(figsize=(8, 5))
    p = projection[(projection.split == "test") & projection.method.isin(["rrq_raw_heterogeneous", "rrq_generalized_heterogeneous"])]
    for (method, proj), frame in p.groupby(["method", "projection"]):
        values = frame.groupby("physical_bpw").recovery.median().reset_index().sort_values("physical_bpw")
        ax.plot(values.physical_bpw, 100 * values.recovery, marker="o", label=f"{METHOD_LABELS.get(method, method)} / {proj}")
    ax.set(xlabel="Projection physical bpw", ylabel="Median correction recovery (%)", title="Projection dense-prefix recovery"); ax.grid(alpha=.25); ax.legend(fontsize=7)
    save_figure(fig, plots, "projection_dense_prefix")

    primary = frontier[(frontier.method == "rrq_raw_heterogeneous")]
    fig, ax = plt.subplots(figsize=(8, 5))
    for layer, frame in primary.groupby("layer"):
        values = frame.groupby("budget_bpw").recovery.agg([lambda x: np.quantile(x, .1), "median", lambda x: np.quantile(x, .9)]).reset_index()
        ax.plot(values.budget_bpw, 100 * values["median"], marker="o", label=f"layer {layer}")
    ax.set(xlabel="Actual physical correction bpw", ylabel="Median recovery (%)", title="Layer heterogeneity of selected RRQ"); ax.grid(alpha=.25); ax.legend()
    save_figure(fig, plots, "layer_frontiers")

    tuple_values = primary[primary.layer.isin([4, 20, 39])].groupby(["tuple", "physical_bpw", "gate_depth", "up_depth", "down_depth"]).recovery.median().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(tuple_values.physical_bpw, 100 * tuple_values.recovery, c=tuple_values.down_depth, cmap="viridis", alpha=.8)
    ax.set(xlabel="Actual physical correction bpw", ylabel="Median recovery (%)", title="All 64 dense RRQ tuples (difficult layers)"); fig.colorbar(scatter, ax=ax, label="Down depth")
    ax.grid(alpha=.2); save_figure(fig, plots, "tuple_pareto")

    fig, ax = plt.subplots(figsize=(8, 5))
    inv = inverse[(inverse.method == "rrq_raw_heterogeneous") & inverse.layer.isin([4, 20, 39])]
    for threshold, frame in inv.groupby("threshold"):
        values = np.sort(frame.required_physical_bpw.dropna().to_numpy())
        if len(values): ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=f"{100*threshold:g}%")
    ax.set(xlabel="Minimum actual physical correction bpw", ylabel="Held-out invocation CDF", title="Required-rate CDF, difficult layers"); ax.grid(alpha=.25); ax.legend()
    save_figure(fig, plots, "required_rate_cdf")

    alloc = primary[primary.layer.isin([4, 20, 39])].groupby("budget_bpw")[["gate_physical_bpw", "up_physical_bpw", "down_physical_bpw"]].median()
    fig, ax = plt.subplots(figsize=(8, 5)); bottom = np.zeros(len(alloc))
    for name, color in (("gate_physical_bpw", "#4c78a8"), ("up_physical_bpw", "#f58518"), ("down_physical_bpw", "#54a24b")):
        values = alloc[name].to_numpy() / 3
        ax.bar(alloc.index, values, bottom=bottom, width=.12, label=name.split("_")[0], color=color); bottom += values
    ax.set(xlabel="Budget (average physical correction bpw)", ylabel="Median allocated average bpw", title="Asymmetric projection allocation"); ax.legend(); save_figure(fig, plots, "projection_allocation")

    fig, ax = plt.subplots(figsize=(8, 5))
    for projection_name, frame in page.groupby("projection"):
        values = frame.groupby("page_size").amplification.median().reset_index()
        ax.plot(values.page_size, values.amplification, marker="o", label=projection_name)
    ax.set_xscale("log", base=2); ax.set(xlabel="Page/sector size (bytes)", ylabel="Physical / logical bytes", title="Dense contiguous stage page amplification"); ax.grid(alpha=.25); ax.legend()
    save_figure(fig, plots, "dense_page_amplification")

    zeros = scale.copy(); zeros["zero_rate"] = zeros.zeros / np.maximum(1, 512 * np.ceil((2048 if True else 512) / 64))
    values = zeros.groupby(["scale_storage", "stage", "projection"]).zeros.sum().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, projection_name in zip(axes, ["gate", "up", "down"]):
        frame = values[values.projection == projection_name]
        for storage_name, group in frame.groupby("scale_storage"):
            ax.plot(group.stage, group.zeros, marker="o", label=storage_name)
        ax.set_title(projection_name); ax.set_xlabel("RRQ stage"); ax.grid(alpha=.25)
    axes[0].set_ylabel("Serialized scales equal to zero"); axes[-1].legend(); save_figure(fig, plots, "scale_underflow")

    selected_stage = stage[(stage.split == "test") & (stage.method == "rrq_raw_heterogeneous")]
    fig, ax = plt.subplots(figsize=(8, 5))
    for projection_name, frame in selected_stage.groupby("projection"):
        values = frame.groupby("stage").functional_contraction.median().reset_index()
        ax.plot(values.stage, values.functional_contraction, marker="o", label=projection_name)
    ax.axhline(1, color="black", linewidth=.8, linestyle="--"); ax.set(xlabel="RRQ stage", ylabel="Median functional residual contraction", title="Stage-wise functional contraction (<1 is useful)"); ax.grid(alpha=.25); ax.legend()
    save_figure(fig, plots, "functional_contraction")

    fig, ax = plt.subplots(figsize=(7, 5))
    sample = selected_stage.replace([np.inf, -np.inf], np.nan).dropna(subset=["pmr_before", "functional_contraction"])
    ax.scatter(np.log10(sample.pmr_before), sample.functional_contraction, c=sample.stage, cmap="plasma", alpha=.45)
    ax.set(xlabel="log10 peak-to-mean-absolute residual magnitude", ylabel="Functional contraction", title="RRQ contraction versus residual PMR"); ax.axhline(1, color="black", linestyle="--", linewidth=.8); ax.grid(alpha=.2)
    save_figure(fig, plots, "pmr_contraction")

    difficult_best = difficult[difficult.budget_bpw == 2.25]
    storage_summary = storage.groupby("method").external_storage_multiplier.median()
    points = difficult_best.set_index("method").join(storage_summary.rename("storage"), how="inner")
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, row in points.iterrows():
        ax.scatter(row.storage, 100 * row.median_recovery, s=55); ax.annotate(METHOD_LABELS.get(method, method), (row.storage, 100 * row.median_recovery), fontsize=7)
    ax.axvline(5, color="black", linestyle="--", linewidth=.8); ax.set(xlabel="External storage / production reference", ylabel="Median recovery at ≤2.25 bpw (%)", title="Storage–quality frontier"); ax.grid(alpha=.25)
    save_figure(fig, plots, "storage_quality")

    heat = primary.groupby(["layer", "budget_bpw"]).recovery.median().unstack()
    fig, ax = plt.subplots(figsize=(9, 3.5)); image = ax.imshow(100 * heat.to_numpy(), aspect="auto", vmin=0, vmax=100, cmap="magma")
    ax.set_yticks(range(len(heat.index)), heat.index); ax.set_xticks(range(len(heat.columns)), heat.columns, rotation=45); ax.set(xlabel="Actual physical correction bpw budget", ylabel="Layer", title="Median complete-expert recovery (%)")
    fig.colorbar(image, ax=ax); save_figure(fig, plots, "layer_budget_heatmap")

    failure = primary[np.isclose(primary.budget_bpw, 2.25)]
    fig, ax = plt.subplots(figsize=(8, 5))
    samples = [failure[failure.layer == layer].recovery.to_numpy() * 100 for layer in sorted(failure.layer.unique())]
    ax.boxplot(samples, tick_labels=[str(v) for v in sorted(failure.layer.unique())], showfliers=True)
    ax.set(xlabel="Layer", ylabel="Recovery (%)", title="Failure tails at ≤2.25 physical correction bpw"); ax.grid(alpha=.2, axis="y")
    save_figure(fig, plots, "failure_tails_2p25bpw")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--prior-allocator", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    data = pd.read_parquet(args.result / "dense_tuple_metrics.parquet")
    if "gate_physical_bpw" not in data:
        total_depth = (data.gate_depth + data.up_depth + data.down_depth).replace(0, np.nan)
        for projection_name in ("gate", "up", "down"):
            data[f"{projection_name}_physical_bpw"] = (3 * data.physical_bpw * data[f"{projection_name}_depth"] / total_depth).fillna(0)
    test = data[data.split == "test"].copy()
    budgets = [float(value) for value in config["actual_correction_bpw_budgets"]]
    frontier = budget_frontier(test, budgets)
    summary = summary_rows(frontier, int(config["bootstrap_resamples"]), int(config["seed"]))
    inverse = inverse_rates(test, [.80, .90, .95, .975, .99])
    paired = paired_pr2(frontier, args.prior_allocator, budgets, int(config["bootstrap_resamples"]), int(config["seed"]))
    projection = pd.read_parquet(args.result / "projection_metrics.parquet")
    projection_required = projection_inverse(projection[projection.split == "test"], [.80, .90, .95, .975, .99])
    explicit = test[test["tuple"].isin(["(0,0,1)", "(1,1,0)", "(1,1,1)", "(1,1,2)", "(2,2,1)", "(2,2,2)"])].groupby(["method", "tuple", "layer"]).recovery.agg(invocations="size", p10=lambda x: np.quantile(x, .1), median="median", p90=lambda x: np.quantile(x, .9)).reset_index()
    allocation = frontier.groupby(["method", "budget_bpw", "layer"])[["gate_physical_bpw", "up_physical_bpw", "down_physical_bpw", "physical_bpw", "recovery"]].median().reset_index()
    auc_rows = []
    for (method, aggregate), frame in summary.sort_values("budget_bpw").groupby(["method", "aggregate"]):
        span = frame.budget_bpw.max() - frame.budget_bpw.min()
        value = np.nan if span <= 0 else np.trapezoid(frame.median_recovery, frame.budget_bpw) / span
        auc_rows.append({"method": method, "aggregate": aggregate, "normalized_median_auc": value})
    auc = pd.DataFrame(auc_rows)
    frontier.to_parquet(args.result / "budget_frontier_per_invocation.parquet", index=False)
    frontier.to_parquet(args.result / "metrics.parquet", index=False)
    summary.to_csv(args.result / "fixed_budget_summary.csv", index=False)
    summary.to_csv(args.result / "summary.csv", index=False)
    summary[summary["aggregate"] == "difficult_4_20_39"].to_csv(args.result / "headline_table.csv", index=False)
    frontier.groupby(["method", "budget_bpw", "layer"]).recovery.agg(invocations="size", mean="mean", p10=lambda x: np.quantile(x, .1), median="median", p90=lambda x: np.quantile(x, .9)).reset_index().to_csv(args.result / "per_layer.csv", index=False)
    frontier.groupby(["method", "budget_bpw", "layer", "expert_id", "expert_stratum"]).recovery.agg(invocations="size", mean="mean", median="median").reset_index().to_csv(args.result / "per_expert.csv", index=False)
    inverse.to_parquet(args.result / "required_rate_per_invocation.parquet", index=False)
    inverse.groupby(["method", "threshold", "layer"]).required_physical_bpw.agg(["count", "mean", "median", lambda x: np.nanquantile(x, .75), lambda x: np.nanquantile(x, .9), lambda x: np.nanquantile(x, .95), lambda x: np.nanquantile(x, .99)]).reset_index().to_csv(args.result / "required_rate_summary.csv", index=False)
    projection_required.to_parquet(args.result / "projection_required_rate.parquet", index=False)
    paired.to_csv(args.result / "paired_vs_pr2.csv", index=False)
    explicit.to_csv(args.result / "explicit_tuple_summary.csv", index=False)
    allocation.to_csv(args.result / "projection_allocation_summary.csv", index=False)
    auc.to_csv(args.result / "rate_distortion_auc.csv", index=False)
    stage_for_correlation = pd.read_parquet(args.result / "stage_diagnostics.parquet")
    stage_for_correlation = stage_for_correlation[(stage_for_correlation.split == "test") & (stage_for_correlation.method == "rrq_raw_heterogeneous")].replace([np.inf, -np.inf], np.nan)
    correlation_rows = []
    for projection_name, frame in stage_for_correlation.groupby("projection"):
        frame = frame.dropna(subset=["pmr_before", "functional_contraction", "zero_code_fraction"])
        correlation_rows.append({
            "projection": projection_name, "rows": len(frame),
            "pearson_log_pmr_vs_functional_contraction": float(np.corrcoef(np.log10(frame.pmr_before), frame.functional_contraction)[0, 1]) if len(frame) > 1 else np.nan,
            "pearson_zero_fraction_vs_functional_contraction": float(np.corrcoef(frame.zero_code_fraction, frame.functional_contraction)[0, 1]) if len(frame) > 1 and frame.zero_code_fraction.nunique() > 1 else np.nan,
            "median_functional_contraction": float(frame.functional_contraction.median()),
        })
    pd.DataFrame(correlation_rows).to_csv(args.result / "residual_contraction_correlations.csv", index=False)
    compute_rows = []
    for (method, budget), frame in frontier.groupby(["method", "budget_bpw"]):
        difficult = frame[frame.layer.isin([4, 20, 39])]
        if difficult.empty:
            continue
        extra_flops = float(difficult.extra_flops.median())
        physical_bytes = float(difficult.physical_bytes.median())
        intensity = extra_flops / max(physical_bytes, 1.0)
        compute_rows.append({
            "method": method, "budget_bpw": budget, "median_extra_flops": extra_flops,
            "median_ratio_to_reference_expert_gemv": float(difficult.extra_flops_reference_expert_ratio.median()),
            "median_correction_bytes": physical_bytes, "arithmetic_intensity_flop_per_streamed_byte": intensity,
            "analytical_gb10_memory_time_seconds": physical_bytes / 273e9,
            "analytical_gb10_fp4_peak_time_seconds": extra_flops / 1e15,
            "analytical_roofline_bound": "memory" if intensity < 1e15 / 273e9 else "compute",
        })
    compute_accounting = {
        "reference_expert_gemv_flops": 2 * 3 * 2048 * 512,
        "resident_q2_base_bytes_per_expert": 3 * (2048 * 512 // 4 + 512 * (2048 // 64) * 2 + 64),
        "gb10_assumptions": {"memory_bandwidth_bytes_per_second": 273e9, "fp4_sparse_peak_ops_per_second": 1e15, "source": "https://docs.nvidia.com/dgx/dgx-spark/hardware.html"},
        "timing_claim": "Analytical GB10 roofline only; no DGX Spark timing. RTX 3090 was used for construction/evaluation, not serving microbenchmarks.",
        "rows": compute_rows,
    }
    (args.result / "compute_accounting.json").write_text(json.dumps(compute_accounting, indent=2, sort_keys=True) + "\n")
    storage_frame = pd.read_csv(args.result / "storage_accounting.csv")
    storage_accounting = {
        "cap_multiplier": float(config["external_storage_cap_reference_multiplier"]),
        "reference_fallback_included": bool(config["reference_fallback_in_external_cap"]),
        "resident_base_external_copy": bool(config["resident_base_external_copy"]),
        "method_median": storage_frame.groupby("method")[["reference_fallback_bytes", "correction_capacity_bytes", "external_storage_multiplier"]].median().reset_index().to_dict("records"),
        "actual_serialized_implementation": True,
    }
    (args.result / "storage_accounting.json").write_text(json.dumps(storage_accounting, indent=2, sort_keys=True) + "\n")
    make_plots(args.result, frontier, summary, inverse, projection, pd.read_parquet(args.result / "stage_diagnostics.parquet"), pd.read_parquet(args.result / "scale_controls.parquet"), pd.read_csv(args.result / "storage_accounting.csv"), pd.read_csv(args.result / "dense_page_accounting.csv"))
    print(json.dumps({"analysis_complete": True, "frontier_rows": len(frontier), "summary_rows": len(summary), "plots": len(list((args.result / "plots").glob("*.png")))}, indent=2))


if __name__ == "__main__":
    main()
