#!/usr/bin/env python3
"""Generate statistics, tables, and figures from the bounded oracle pilot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RATES = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
THRESHOLDS = [0.8, 0.9, 0.95, 0.975, 0.99]
INVOCATION = ["request_id", "position", "layer", "expert_id", "router_rank"]
COLORS = {
    "native": "#4C78A8", "pca": "#F58518", "generalized": "#54A24B",
    "learned_s11": "#E45756", "learned_s29": "#B279A2", "learned_s47": "#FF9DA6",
    "random": "#9D755D", "overcomplete_2x": "#BAB0AC", "two_view_oracle": "#2F4B7C",
    "progressive_best": "#D62728", "global_multi_view_oracle": "#111111",
}


def save_figure(fig: plt.Figure, directory: Path, name: str) -> None:
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(directory / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def best_progressive(data: pd.DataFrame) -> pd.DataFrame:
    keys = INVOCATION + ["projection", "budget_bpw"]
    chosen = data.loc[data.groupby(keys)["recovery"].idxmax()].copy()
    chosen["method"] = "progressive_best"
    chosen["rate_axis"] = "physical"
    return chosen


def global_hybrid(data: pd.DataFrame) -> pd.DataFrame:
    keys = INVOCATION + ["budget_bpw", "down_oracle_variant"]
    chosen = data.loc[data.groupby(keys)["recovery"].idxmax()].copy()
    chosen["source_method"] = chosen["method"]
    chosen["method"] = "global_multi_view_oracle"
    return chosen


def crossing_rows(data: pd.DataFrame, maximum: float = 4.0) -> pd.DataFrame:
    output = []
    group_keys = INVOCATION + ["projection", "method"]
    for key, group in data.groupby(group_keys, sort=False):
        group = group.sort_values("budget_bpw")
        rates = group["physical_bpw"].to_numpy(float)
        recovery = np.maximum.accumulate(group["recovery"].to_numpy(float))
        for threshold in THRESHOLDS:
            hit = np.flatnonzero(recovery >= threshold)
            discrete = float(rates[hit[0]]) if len(hit) else math.inf
            budget = float(group["budget_bpw"].to_numpy(float)[hit[0]]) if len(hit) else math.inf
            output.append({
                **dict(zip(group_keys, key)), "threshold": threshold,
                "required_physical_bpw": discrete,
                "required_budget_cap_bpw": budget,
                "reached_by_4bpw": bool(len(hit)),
                "capped_required_bpw": min(discrete, maximum),
            })
    return pd.DataFrame(output)


def bootstrap_stat(values: pd.DataFrame, column: str, statistic: str, resamples: int, rng: np.random.Generator) -> tuple[float, float]:
    clusters = {key: value[column].to_numpy(float) for key, value in values.groupby("request_id")}
    names = np.asarray(list(clusters), dtype=object)
    estimates = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        picked = rng.choice(names, size=len(names), replace=True)
        sample = np.concatenate([clusters[name] for name in picked])
        if statistic == "mean":
            estimates[index] = np.mean(sample)
        elif statistic == "median":
            estimates[index] = np.median(sample)
        elif statistic == "reach":
            estimates[index] = np.mean(sample)
        else:
            raise ValueError(statistic)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def inverse_statistics(crossings: pd.DataFrame, resamples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for key, group in crossings.groupby(["projection", "method", "threshold"]):
        raw = group["required_physical_bpw"].to_numpy(float)
        finite = raw[np.isfinite(raw)]
        capped = group["capped_required_bpw"].to_numpy(float)
        reach = group["reached_by_4bpw"].to_numpy(float)
        mean_ci = bootstrap_stat(group, "capped_required_bpw", "mean", resamples, rng)
        median_ci = bootstrap_stat(group, "capped_required_bpw", "median", resamples, rng)
        reach_group = group.assign(reach_numeric=reach)
        reach_ci = bootstrap_stat(reach_group, "reach_numeric", "reach", resamples, rng)
        def censored_quantile(q: float) -> float:
            ordered = np.sort(raw)
            return float(ordered[min(int(math.ceil(q * len(ordered))) - 1, len(ordered) - 1)])
        rows.append({
            "projection": key[0], "method": key[1], "threshold": key[2],
            "invocations": len(group), "requests": group.request_id.nunique(),
            "reach_fraction_by_4bpw": float(reach.mean()),
            "reach_fraction_ci_low": reach_ci[0], "reach_fraction_ci_high": reach_ci[1],
            "mean_required_bpw_capped_at_4": float(capped.mean()),
            "mean_capped_ci_low": mean_ci[0], "mean_capped_ci_high": mean_ci[1],
            "median_required_bpw": censored_quantile(0.5),
            "median_capped_ci_low": median_ci[0], "median_capped_ci_high": median_ci[1],
            "p75_required_bpw": censored_quantile(0.75),
            "p90_required_bpw": censored_quantile(0.90),
            "p95_required_bpw": censored_quantile(0.95),
            "p99_required_bpw": censored_quantile(0.99),
            "finite_mean_conditional": float(np.mean(finite)) if len(finite) else math.inf,
        })
    return pd.DataFrame(rows)


def recovery_statistics(data: pd.DataFrame, resamples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    wanted = data[data.budget_bpw.isin([0.1, 0.2, 0.5, 1.0])]
    rows = []
    for key, group in wanted.groupby(["projection", "method", "budget_bpw"]):
        mean_ci = bootstrap_stat(group, "recovery", "mean", resamples, rng)
        rows.append({
            "projection": key[0], "method": key[1], "physical_budget_bpw": key[2],
            "invocations": len(group), "requests": group.request_id.nunique(),
            "mean": group.recovery.mean(), "ci_low": mean_ci[0], "ci_high": mean_ci[1],
            "median": group.recovery.median(), "p10": group.recovery.quantile(0.1),
            "p25": group.recovery.quantile(0.25), "p75": group.recovery.quantile(0.75),
            "p90": group.recovery.quantile(0.9), "p95": group.recovery.quantile(0.95),
            "p99": group.recovery.quantile(0.99),
            "fraction_above_80": (group.recovery >= 0.8).mean(),
            "fraction_above_90": (group.recovery >= 0.9).mean(),
            "fraction_above_95": (group.recovery >= 0.95).mean(),
            "fraction_above_99": (group.recovery >= 0.99).mean(),
        })
    return pd.DataFrame(rows)


def auc_by_invocation(data: pd.DataFrame, maximum: float = 2.0) -> pd.DataFrame:
    rows = []
    for key, group in data[data.budget_bpw <= maximum].groupby(INVOCATION + ["projection", "method"]):
        group = group.sort_values("physical_bpw").drop_duplicates("physical_bpw", keep="last")
        x = group.physical_bpw.to_numpy(float)
        y = np.maximum.accumulate(group.recovery.to_numpy(float))
        rows.append({**dict(zip(INVOCATION + ["projection", "method"], key)), "auc_0_2": float(np.trapezoid(y, x) / maximum)})
    return pd.DataFrame(rows)


def fixed_rate_lookup(group: pd.DataFrame, rate: float) -> float:
    exact = group[np.isclose(group.budget_bpw, rate)]
    return float(exact.recovery.median()) if len(exact) else math.nan


def projection_table(data: pd.DataFrame, crossings: pd.DataFrame, auc: pd.DataFrame) -> pd.DataFrame:
    allowed = ["native", "pca", "generalized", "learned_s11", "learned_s29", "learned_s47", "random", "overcomplete_2x"]
    rows = []
    for layer in sorted(data.layer.unique()):
        for projection in ("gate", "up", "down"):
            candidates = auc[(auc.layer == layer) & (auc.projection == projection) & auc.method.isin(allowed)]
            best = candidates.groupby("method").auc_0_2.mean().idxmax()
            group = data[(data.layer == layer) & (data.projection == projection) & (data.method == best)]
            cross = crossings[(crossings.layer == layer) & (crossings.projection == projection) & (crossings.method == best)]
            row = {"layer": layer, "projection": projection, "best_basis": best, "held_out_invocations": len(group) // max(group.budget_bpw.nunique(), 1)}
            for rate in (0.2, 0.5, 1.0, 1.5, 2.0):
                row[f"recovery_at_{rate}"] = fixed_rate_lookup(group, rate)
            for threshold in (0.9, 0.95, 0.99):
                values = cross[cross.threshold == threshold].required_physical_bpw.to_numpy(float)
                row[f"median_bpw_at_{threshold}"] = float(np.quantile(values, 0.5)) if len(values) else math.inf
                row[f"reach_fraction_{threshold}_by_4"] = float(np.isfinite(values).mean()) if len(values) else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def compute_accounting(hybrid: pd.DataFrame) -> dict:
    selected = hybrid[(hybrid.method == "global_multi_view_oracle") & (hybrid.down_oracle_variant == "sequential")]
    rows = {}
    for budget in sorted(selected.budget_bpw.unique()):
        group = selected[selected.budget_bpw == budget]
        transform_flops = 2 * 2048 * (group.gate_atoms + group.up_atoms) + 2 * 512 * group.down_atoms
        accumulation_flops = 2 * 512 * (group.gate_atoms + group.up_atoms) + 2 * 2048 * group.down_atoms
        q4_flops = 2 * 3 * 2048 * 512
        rows[str(budget)] = {
            "median_selected_row_transform_flops": float(transform_flops.median()),
            "median_correction_accumulation_flops": float(accumulation_flops.median()),
            "median_total_extra_flops": float((transform_flops + accumulation_flops).median()),
            "median_extra_to_complete_q4_expert_flop_ratio": float(((transform_flops + accumulation_flops) / q4_flops).median()),
            "oracle_full_gate_up_transform_flops_per_layer_token": 2 * 2048 * 2048,
            "note": "Analytical FLOPs only; no DGX Spark timing. Gate/up union reuse is not credited, so selected-row transform cost is conservative.",
        }
    return {
        "resident_binary_base_effective_bpw": 1.25048828125,
        "full_q4_expert_projection_flops": 2 * 3 * 2048 * 512,
        "deployment_h4_selected_support": rows,
        "measured_gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition (not a DGX Spark)",
    }


def storage_accounting(phase_facts: dict, progressive_facts: dict) -> dict:
    layers = {}
    for layer, facts in phase_facts["layers"].items():
        q4 = sum(value["bytes_per_expert"] for value in facts["gguf_storage"].values())
        binary = sum(value["total_bytes"] for value in facts["binary_storage"].values())
        storage = progressive_facts["layers"][layer]["storage"]
        configs = {}
        for bits in (2, 4, 8, 16):
            correction = sum(storage[f"{projection}_{bits}"]["atom_bytes_per_expert"] for projection in ("gate", "up", "down"))
            configs[str(bits)] = {
                "q4_bytes_per_expert": q4, "correction_view_bytes_per_expert": correction,
                "q4_plus_view_bytes_per_expert": q4 + correction,
                "external_multiplier_vs_q4": (q4 + correction) / q4,
            }
        layers[layer] = {"q4": facts["gguf_storage"], "resident_binary_bytes_per_expert": binary, "progressive_views": configs}
    return {"layers": layers, "five_times_cap_respected_by_full_16bit_prefix": all(v["progressive_views"]["16"]["external_multiplier_vs_q4"] <= 5 for v in layers.values())}


def make_plots(phase: pd.DataFrame, progressive: pd.DataFrame, hybrid: pd.DataFrame, crossings: pd.DataFrame, storage: dict, plots: Path, structure_dir: Path) -> None:
    plots.mkdir(parents=True, exist_ok=True)
    key_methods = ["native", "pca", "generalized", "learned_s11", "random", "two_view_oracle"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    ideal = phase[(phase.rate_axis == "ideal") & phase.method.isin(key_methods)]
    for axis, projection in zip(axes, ("gate", "up", "down")):
        for method, group in ideal[ideal.projection == projection].groupby("method"):
            curve = group.groupby("budget_bpw").recovery.median()
            axis.plot(curve.index, curve.values, marker="o", ms=3, label=method, color=COLORS.get(method))
        axis.set(title=projection, xlabel="ideal streamed bpw", xlim=(0, 4), ylim=(-0.05, 1.02)); axis.grid(alpha=.25)
    axes[0].set_ylabel("median correction recovery"); axes[-1].legend(fontsize=7, loc="lower right")
    save_figure(fig, plots, "local_rate_distortion")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    physical = phase[(phase.rate_axis == "physical") & phase.method.isin(["native", "pca", "generalized", "two_view_oracle"])]
    for axis, projection in zip(axes, ("gate", "up", "down")):
        for method, group in physical[physical.projection == projection].groupby("method"):
            curve = group.groupby("budget_bpw").recovery.median()
            axis.plot(curve.index, curve.values, marker="o", ms=3, label=f"{method} BF16", alpha=.75)
        curve = progressive[progressive.projection == projection].groupby("budget_bpw").recovery.median()
        axis.plot(curve.index, curve.values, marker="s", lw=2.2, color=COLORS["progressive_best"], label="progressive precision")
        axis.axhline(.9, color="black", ls="--", lw=.8); axis.set(title=projection, xlabel="4 KiB physical bpw", xlim=(0, 4), ylim=(-.05, 1.02)); axis.grid(alpha=.25)
    axes[0].set_ylabel("median future-proxy-weighted recovery"); axes[-1].legend(fontsize=7, loc="lower right")
    save_figure(fig, plots, "physical_rate_propagation")

    seq = hybrid[hybrid.down_oracle_variant == "sequential"]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for method, group in seq.groupby("method"):
        curve = group.groupby("budget_bpw").agg(recovery=("recovery", "median"), physical=("physical_bpw", "median"))
        ax.plot(curve.physical, curve.recovery, marker="o", ms=3, label=method, color=COLORS.get(method))
    ax.axhline(.9, color="black", ls="--", lw=.8); ax.set(xlabel="actual 4 KiB physical average expert bpw", ylabel="median complete-expert future-proxy recovery", xlim=(0, 4), ylim=(-.05, 1.02)); ax.grid(alpha=.25); ax.legend(fontsize=7)
    save_figure(fig, plots, "complete_expert_hybrid_curve")

    global_seq = seq[seq.method == "global_multi_view_oracle"]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    allocation = global_seq.groupby("budget_bpw")[["gate_bpw", "up_bpw", "down_bpw"]].median()
    ax.stackplot(allocation.index, allocation.gate_bpw / 3, allocation.up_bpw / 3, allocation.down_bpw / 3, labels=["gate", "up", "down"], alpha=.85)
    ax.plot(allocation.index, allocation.sum(axis=1) / 3, color="black", lw=1, label="actual total")
    ax.set(xlabel="allowed average physical bpw", ylabel="average expert bpw allocation", xlim=(0, 4)); ax.grid(alpha=.25); ax.legend()
    save_figure(fig, plots, "projection_allocation")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for rank, group in global_seq[global_seq.budget_bpw == 2.0].groupby("router_rank"):
        ax.scatter(rank, group.physical_bpw.mean(), s=28, color="#4C78A8")
    ax.set(xlabel="router rank within top-8", ylabel="mean allocated physical bpw at 2-bpw cap", xticks=range(1, 9)); ax.grid(alpha=.25)
    save_figure(fig, plots, "router_rank_allocation")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    for axis, projection in zip(axes, ("gate", "up", "down")):
        group = crossings[(crossings.projection == projection) & crossings.method.isin(["native", "pca", "generalized", "progressive_best"])]
        for (method, threshold), values in group.groupby(["method", "threshold"]):
            if threshold not in (0.8, 0.9, 0.95, 0.99):
                continue
            finite = np.sort(values.required_physical_bpw.replace(np.inf, 4.25).to_numpy(float))
            axis.step(finite, np.arange(1, len(finite) + 1) / len(finite), where="post", label=f"{method} {threshold:g}", alpha=.75)
        axis.axvline(4, color="black", ls="--", lw=.7); axis.set(title=projection, xlabel="required physical bpw (4.25 = >4)", xlim=(0, 4.3)); axis.grid(alpha=.25)
    axes[0].set_ylabel("held-out invocation CDF"); axes[-1].legend(fontsize=6, ncol=2)
    save_figure(fig, plots, "required_rate_cdf")

    best90 = crossings[crossings.threshold == .9].copy()
    candidates = best90[best90.method.isin(["native", "pca", "generalized", "progressive_best"])]
    layer_values = candidates.groupby(["layer", "projection", "method"]).required_physical_bpw.median().reset_index()
    layer_values = layer_values.loc[layer_values.groupby(["layer", "projection"]).required_physical_bpw.idxmin()]
    matrix = layer_values.pivot(index="layer", columns="projection", values="required_physical_bpw").replace(np.inf, 4.5)
    fig, ax = plt.subplots(figsize=(6, 4)); image = ax.imshow(matrix.values, vmin=0, vmax=4.5, cmap="magma_r", aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            label = ">4" if matrix.iloc[i, j] > 4 else f"{matrix.iloc[i, j]:.2f}"
            ax.text(j, i, label, ha="center", va="center", color="white" if matrix.iloc[i, j] < 2 else "black")
    ax.set(xticks=range(len(matrix.columns)), xticklabels=matrix.columns, yticks=range(len(matrix.index)), yticklabels=matrix.index, xlabel="projection", ylabel="layer", title="Median physical bpw required for 90% recovery")
    fig.colorbar(image, ax=ax, label="physical bpw")
    save_figure(fig, plots, "layer_heatmap")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    prog90 = crossings[(crossings.method == "progressive_best") & (crossings.threshold == .9)].copy()
    strata = progressive[INVOCATION + ["expert_stratum"]].drop_duplicates()
    prog90 = prog90.merge(strata, on=INVOCATION, how="left")
    categories = ["cold", "median", "hot"]
    values = [prog90[prog90.expert_stratum == value].required_physical_bpw.replace(np.inf, 4.25) for value in categories]
    ax.boxplot(values, tick_labels=categories, showfliers=False); ax.axhline(4, color="black", ls="--", lw=.8); ax.set(ylabel="required physical bpw for 90% (4.25 = >4)", xlabel="training-frequency stratum"); ax.grid(alpha=.25)
    save_figure(fig, plots, "expert_frequency")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    amp = progressive.assign(amplification=progressive.physical_bytes_4k / progressive.logical_bytes.clip(lower=1))
    for projection, group in amp.groupby("projection"):
        curve = group.groupby("budget_bpw").amplification.median()
        ax.plot(curve.index, curve.values, marker="o", label=projection)
    ax.set(xlabel="4 KiB physical budget bpw", ylabel="median page-read amplification", yscale="log"); ax.grid(alpha=.25); ax.legend()
    save_figure(fig, plots, "page_amplification")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for multiplier in (1, 2, 3, 4, 5):
        best = []
        for layer, facts in storage["layers"].items():
            eligible = [int(bits) for bits, values in facts["progressive_views"].items() if values["external_multiplier_vs_q4"] <= multiplier]
            if not eligible:
                best.append(0.0); continue
            subset = progressive[(progressive.layer == int(layer)) & (progressive.atom_bits.isin(eligible)) & (progressive.budget_bpw == 4.0)]
            best.append(subset.groupby(INVOCATION + ["projection"]).recovery.max().median())
        ax.scatter(multiplier, np.mean(best), s=55, color="#4C78A8")
    ax.plot(range(1, 6), [np.nan] * 5, alpha=0); ax.set(xlabel="external storage cap (× authoritative Q4 expert)", ylabel="median projection recovery at 4 physical bpw", xticks=range(1, 6), ylim=(0, 1)); ax.grid(alpha=.25)
    save_figure(fig, plots, "storage_multiplier_frontier")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    sample = progressive[progressive.budget_bpw == 2.0].sample(min(2500, len(progressive[progressive.budget_bpw == 2.0])), random_state=7)
    for axis, projection in zip(axes, ("gate", "up", "down")):
        group = sample[sample.projection == projection]
        scatter = axis.scatter(group.relative_output_error, 1 - group.recovery, c=group.router_boundary_margin, s=7, alpha=.45, cmap="viridis")
        axis.set(title=projection, xlabel="local relative output error", yscale="log"); axis.grid(alpha=.25)
    axes[0].set_ylabel("future-proxy damage fraction"); fig.colorbar(scatter, ax=axes, label="authoritative router boundary margin")
    save_figure(fig, plots, "error_amplification")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for axis, projection in zip(axes, ("gate", "up", "down")):
        values = [progressive[(progressive.projection == projection) & (progressive.budget_bpw == rate)].recovery.to_numpy() for rate in (.1, .2, .5, 1.0)]
        axis.violinplot(values, positions=range(4), showmedians=True, showextrema=False); axis.set(title=projection, xticks=range(4), xticklabels=["0.1", "0.2", "0.5", "1.0"], xlabel="physical bpw", ylim=(-.5, 1.02)); axis.grid(alpha=.25)
    axes[0].set_ylabel("held-out recovery")
    save_figure(fig, plots, "per_method_violin")

    structure_csv = structure_dir / "structure_diagnostics.csv"
    spectra_file = structure_dir / "structure_spectra.parquet"
    if structure_csv.exists() and spectra_file.exists():
        spectra = pd.read_parquet(spectra_file)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
        for axis, projection in zip(axes, ("gate", "up", "down")):
            for layer, group in spectra[(spectra.projection == projection) & (spectra.spectrum == "residual_action")].groupby("layer"):
                values = group.sort_values("rank").eigenvalue.to_numpy(float); cumulative = np.cumsum(values) / max(values.sum(), 1e-30)
                axis.plot(np.arange(1, len(values) + 1), cumulative, label=f"layer {layer}")
            axis.set(title=projection, xlabel="residual-action covariance rank", xscale="log", ylim=(0, 1.01)); axis.grid(alpha=.25)
        axes[0].set_ylabel("cumulative correction energy"); axes[-1].legend()
        save_figure(fig, plots, "residual_action_spectrum")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--hybrid-results", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()
    output = args.results
    plots = output / "plots"
    phase = pd.read_parquet(output / "phase_a_metrics.parquet")
    raw_progressive = pd.read_parquet(output / "progressive_metrics.parquet")
    progressive = best_progressive(raw_progressive)
    hybrid_raw = pd.read_parquet(args.hybrid_results / "hybrid_metrics.parquet")
    hybrid = pd.concat([hybrid_raw, global_hybrid(hybrid_raw)], ignore_index=True)
    physical = phase[(phase.rate_axis == "physical") & (phase.method != "base_only")].copy()
    projection_data = pd.concat([physical, progressive], ignore_index=True, sort=False)
    crossings = crossing_rows(projection_data)
    inverse = inverse_statistics(crossings, args.bootstrap, args.seed)
    recovery = recovery_statistics(projection_data, args.bootstrap, args.seed + 1)
    auc = auc_by_invocation(projection_data)
    table = projection_table(projection_data, crossings, auc)

    phase_facts = json.loads((output / "phase_a_facts.json").read_text())
    progressive_facts = json.loads((output / "progressive_facts.json").read_text())
    storage = storage_accounting(phase_facts, progressive_facts)
    compute = compute_accounting(hybrid)
    (output / "storage_accounting.json").write_text(json.dumps(storage, indent=2, sort_keys=True) + "\n")
    (output / "compute_accounting.json").write_text(json.dumps(compute, indent=2, sort_keys=True) + "\n")
    crossings.to_parquet(output / "inverse_rate_per_invocation.parquet", index=False)
    inverse.to_csv(output / "inverse_rate_summary.csv", index=False)
    recovery.to_csv(output / "fixed_rate_statistics.csv", index=False)
    auc.to_csv(output / "rate_distortion_auc.csv", index=False)
    table.to_csv(output / "projection_interim_table.csv", index=False)

    seq = hybrid[hybrid.down_oracle_variant == "sequential"]
    hybrid_summary = seq.groupby(["layer", "method", "budget_bpw"]).agg(
        invocations=("recovery", "size"), recovery_median=("recovery", "median"), recovery_p10=("recovery", lambda x: x.quantile(.1)),
        relative_error_median=("relative_output_error", "median"), physical_bpw_median=("physical_bpw", "median"),
        gate_bpw_median=("gate_bpw", "median"), up_bpw_median=("up_bpw", "median"), down_bpw_median=("down_bpw", "median"),
    ).reset_index()
    hybrid_summary.to_csv(output / "complete_expert_hybrid_table.csv", index=False)
    metrics = pd.concat([
        phase.assign(metric_source="phase_a_bf16"),
        raw_progressive.assign(metric_source="phase_a_progressive"),
        hybrid_raw.assign(metric_source="phase_b_hybrid"),
    ], ignore_index=True, sort=False)
    metrics.to_parquet(output / "metrics.parquet", index=False)
    summary = recovery.copy(); summary.to_csv(output / "summary.csv", index=False)
    projection_data.groupby(["layer", "projection", "method", "budget_bpw"]).recovery.agg(["count", "mean", "median", "std"]).reset_index().to_csv(output / "per_layer.csv", index=False)
    projection_data.groupby(["layer", "expert_id", "expert_stratum", "projection", "method", "budget_bpw"]).recovery.agg(["count", "mean", "median"]).reset_index().to_csv(output / "per_expert.csv", index=False)
    make_plots(phase, progressive, hybrid, crossings, storage, plots, output)
    print(json.dumps({"metrics_rows": len(metrics), "inverse_rows": len(inverse), "plots_png": len(list(plots.glob("*.png"))), "plots_svg": len(list(plots.glob("*.svg")))}, indent=2))


if __name__ == "__main__":
    main()
