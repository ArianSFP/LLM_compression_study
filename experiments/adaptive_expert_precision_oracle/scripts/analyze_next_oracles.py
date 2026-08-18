#!/usr/bin/env python3
"""Reproduce next-oracle statistics and PNG/SVG figures from saved Parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KEYS = ["request_id", "position", "layer", "expert_id"]
CONFIG = ["method", "requested_rank", "fit", "coefficient_bits"]
GROUPS = {"all_four": [0, 4, 20, 39], "difficult": [4, 20, 39], "layer0": [0]}


def save(fig, root: Path, name: str) -> None:
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(root / f"{name}.{suffix}", dpi=190)
    plt.close(fig)


def clustered_ci(frame: pd.DataFrame, column: str, resamples: int, rng: np.random.Generator, statistic: str = "median") -> tuple[float, float]:
    request_ids = frame.request_id.astype(str).unique()
    grouped = {key: value[column].to_numpy() for key, value in frame.assign(request_id=frame.request_id.astype(str)).groupby("request_id")}
    if not len(request_ids):
        return np.nan, np.nan
    estimates = []
    for _ in range(resamples):
        sampled = rng.choice(request_ids, len(request_ids), replace=True)
        values = np.concatenate([grouped[key] for key in sampled])
        estimates.append(float(np.median(values) if statistic == "median" else np.mean(values)))
    return tuple(np.quantile(estimates, [.025, .975]))


def summarize(frame: pd.DataFrame, resamples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for group, layers in GROUPS.items():
        for (method, budget), part in frame[frame.layer.isin(layers)].groupby(["method", "budget_bpw"], dropna=False):
            recovery = part.recovery.to_numpy()
            ci = clustered_ci(part, "recovery", resamples, rng)
            rows.append({
                "group": group, "method": method, "budget_bpw": budget,
                "invocations": len(part), "requests": part.request_id.nunique(),
                "mean_recovery": np.mean(recovery), "median_recovery": np.median(recovery),
                "p10_recovery": np.quantile(recovery, .10), "p25_recovery": np.quantile(recovery, .25),
                "p75_recovery": np.quantile(recovery, .75), "p90_recovery": np.quantile(recovery, .90),
                "p95_remaining_damage": np.quantile(1 - recovery, .95),
                "fraction_above_80": np.mean(recovery >= .80), "fraction_above_90": np.mean(recovery >= .90),
                "fraction_above_95": np.mean(recovery >= .95), "fraction_above_99": np.mean(recovery >= .99),
                "median_relative_output_error": part.relative_output_error.median() if "relative_output_error" in part else np.nan,
                "median_logical_bpw": part.logical_bpw.median() if "logical_bpw" in part else np.nan,
                "median_physical_bpw": part.physical_bpw.median() if "physical_bpw" in part else np.nan,
                "median_page_amplification": part.page_amplification.median() if "page_amplification" in part else np.nan,
                "median_external_storage_multiplier": part.external_storage_multiplier.median() if "external_storage_multiplier" in part else np.nan,
                "median_extra_flops": part.extra_flops.median() if "extra_flops" in part else np.nan,
                "median_ci_low": ci[0], "median_ci_high": ci[1],
            })
    return pd.DataFrame(rows)


def paired(candidate: pd.DataFrame, baseline: pd.DataFrame, label: str, resamples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    joined = candidate.merge(baseline[KEYS + ["budget_bpw", "recovery"]], on=KEYS + ["budget_bpw"], suffixes=("_new", "_base"))
    joined["difference"] = joined.recovery_new - joined.recovery_base
    rows = []
    for budget, part in joined.groupby("budget_bpw"):
        ci = clustered_ci(part, "difference", resamples, rng, statistic="mean")
        rows.append({"comparison": label, "budget_bpw": budget, "pairs": len(part), "requests": part.request_id.nunique(),
                     "mean_difference": part.difference.mean(), "median_difference": part.difference.median(), "ci_low": ci[0], "ci_high": ci[1]})
    return pd.DataFrame(rows)


def lineplot(frame: pd.DataFrame, methods: list[str], name: str, plots: Path, ylabel: str = "Median complete-expert recovery") -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for method in methods:
        part = frame[frame.method == method]
        value = part.groupby("budget_bpw").recovery.agg(median="median", q10=lambda x: x.quantile(.1), q90=lambda x: x.quantile(.9)).reset_index()
        if value.empty:
            continue
        ax.plot(value.budget_bpw, value["median"], marker="o", label=method)
        ax.fill_between(value.budget_bpw, value.q10, value.q90, alpha=.12)
    ax.set(xlabel="Physical correction budget (bpw)", ylabel=ylabel)
    ax.grid(alpha=.25); ax.legend(fontsize=7)
    save(fig, plots, name)


def select_run_b(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one deployable configuration per layer/rate on validation only."""
    deployable = frame[frame.method != "B4_per_expert_raw"]
    validation = deployable[deployable.evaluation_split == "validation"]
    expected = {layer: part[KEYS].drop_duplicates().shape[0] for layer, part in validation.groupby("layer")}
    scores = validation.groupby(["layer", "budget_bpw"] + CONFIG).agg(
        validation_median_recovery=("recovery", "median"), validation_mean_recovery=("recovery", "mean"), validation_invocations=("recovery", "size")
    ).reset_index()
    scores = scores[scores.apply(lambda row: row.validation_invocations == expected[row.layer], axis=1)]
    choices = scores.sort_values(["layer", "budget_bpw", "validation_median_recovery", "validation_mean_recovery"]).groupby(["layer", "budget_bpw"], as_index=False).tail(1)
    test = frame[frame.evaluation_split == "test"].merge(choices[["layer", "budget_bpw"] + CONFIG], on=["layer", "budget_bpw"] + CONFIG)
    test["selected_method"] = test.method
    test["method"] = "B_validation_selected_deployable"
    return choices.sort_values(["layer", "budget_bpw"]), test


def inverse_rates(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for key, part in frame.groupby(KEYS):
        for threshold in (.80, .90, .95, .99):
            reached = part[part.recovery >= threshold]
            rows.append(dict(zip(KEYS, key)) | {"method": label, "threshold": threshold,
                                                "required_physical_bpw": reached.physical_bpw.min() if len(reached) else np.nan})
    return pd.DataFrame(rows)


def run_a_compute(sequence: pd.DataFrame, budgets: list[float]) -> pd.DataFrame:
    """Deployment arithmetic after H4 support prediction (no full H0 transform)."""
    original_weights = 3 * 2048 * 512
    rows = []
    for key, part in sequence.groupby(KEYS):
        part = part.sort_values("step")
        for budget in budgets:
            allowed = budget * original_weights / 8
            selected = part[part.cumulative_physical_bytes <= allowed + 1e-9]
            gate = selected[selected.projection == "gate"].atom_id.nunique()
            up = selected[selected.projection == "up"].atom_id.nunique()
            down_atoms = selected[selected.projection == "down"].atom_id.nunique()
            analysis_rows = len(set(selected[selected.projection == "gate"].atom_id) | set(selected[selected.projection == "up"].atom_id))
            transform_flops = 2 * 2048 * analysis_rows
            accumulation_flops = 2 * (512 * (gate + up) + 2048 * down_atoms)
            total = transform_flops + accumulation_flops
            logical = int(selected.cumulative_logical_bytes.iloc[-1]) if len(selected) else 0
            resident_transform_bytes = analysis_rows * 2048 * 2
            rows.append(dict(zip(KEYS, key)) | {"budget_bpw": budget, "gate_atoms": gate, "up_atoms": up, "down_atoms": down_atoms,
                "unique_gate_up_analysis_rows": analysis_rows, "selected_row_transform_flops": transform_flops,
                "correction_accumulation_flops": accumulation_flops, "extra_flops": total,
                "ratio_to_reference_expert": total / (2 * 3 * 2048 * 512),
                "resident_transform_bytes_read": resident_transform_bytes, "logical_correction_bytes": logical,
                "arithmetic_intensity_flops_per_local_plus_logical_byte": total / max(resident_transform_bytes + logical, 1)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plots = args.output / "plots"; plots.mkdir(exist_ok=True)

    baseline_all = pd.read_parquet(args.baseline / "hybrid_metrics.parquet")
    baseline = baseline_all[(baseline_all.method == "generalized_generalized_native") & (baseline_all.down_oracle_variant == "sequential")].copy()
    baseline["method"] = "A0_current_Q2_whole_projection"
    baseline["logical_bpw"] = baseline.ideal_bpw
    baseline["physical_bytes"] = baseline.physical_bytes_4k
    baseline["page_amplification"] = baseline.physical_bytes_4k / baseline.logical_bytes.clip(lower=1)
    baseline["extra_flops"] = np.nan; baseline["external_storage_multiplier"] = np.nan
    run_a = pd.read_parquet(args.run_a / "allocator_metrics.parquet")
    sequence = pd.read_parquet(args.run_a / "selected_increment_sequence.parquet")
    run_b_all = pd.read_parquet(args.run_b / "low_rank_complete_expert_metrics.parquet")
    down = pd.read_parquet(args.run_b / "low_rank_down_metrics.parquet")
    choices_b, selected_b = select_run_b(run_b_all)
    choices_b.to_csv(args.output / "run_b_validation_choices.csv", index=False)
    selected_b.to_parquet(args.output / "run_b_validation_selected_test.parquet", index=False)

    primary_a = run_a[run_a.method == "A6_corrected_screened_greedy"].copy()
    primary = pd.concat([baseline, primary_a, selected_b], ignore_index=True, sort=False)
    primary.to_parquet(args.output / "metrics.parquet", index=False)
    summary = summarize(primary, args.bootstrap, args.seed)
    summary.to_csv(args.output / "summary.csv", index=False)
    summarize(run_a, args.bootstrap, args.seed + 1).to_csv(args.output / "run_a_summary.csv", index=False)
    summarize(selected_b, args.bootstrap, args.seed + 2).to_csv(args.output / "run_b_summary.csv", index=False)
    pd.concat([
        paired(primary_a, baseline, "RunA_minus_current_Q2", args.bootstrap, args.seed + 3),
        paired(selected_b, baseline, "RunB_minus_current_Q2", args.bootstrap, args.seed + 4),
    ]).to_csv(args.output / "paired_differences.csv", index=False)
    primary.groupby(["method", "layer", "budget_bpw"]).recovery.agg(invocations="size", median_recovery="median", p10_recovery=lambda x: x.quantile(.1), p90_recovery=lambda x: x.quantile(.9)).reset_index().to_csv(args.output / "per_layer.csv", index=False)
    primary.groupby(["method", "layer", "expert_id", "budget_bpw"]).recovery.agg(invocations="size", median_recovery="median").reset_index().to_csv(args.output / "per_expert.csv", index=False)
    inverse = pd.concat([inverse_rates(primary_a, "RunA"), inverse_rates(selected_b, "RunB_validation_selected")], ignore_index=True)
    inverse.to_parquet(args.output / "required_rate_per_invocation.parquet", index=False)
    compute_a = run_a_compute(sequence, sorted(primary_a.budget_bpw.unique()))
    compute_a.to_parquet(args.output / "run_a_compute_per_invocation.parquet", index=False)
    compute_a.groupby("budget_bpw").agg(invocations=("extra_flops", "size"), median_extra_flops=("extra_flops", "median"), p90_extra_flops=("extra_flops", lambda x: x.quantile(.9)), median_ratio_to_reference=("ratio_to_reference_expert", "median"), median_analysis_rows=("unique_gate_up_analysis_rows", "median"), median_arithmetic_intensity=("arithmetic_intensity_flops_per_local_plus_logical_byte", "median")).reset_index().to_csv(args.output / "run_a_compute_summary.csv", index=False)

    down_test = down[down.evaluation_split == "test"]
    down_summary = down_test.groupby(CONFIG + ["down_physical_bpw"], dropna=False).recovery.agg(invocations="size", mean_recovery="mean", median_recovery="median", p10_recovery=lambda x: x.quantile(.1), p90_recovery=lambda x: x.quantile(.9)).reset_index()
    down_summary.to_csv(args.output / "down_rank_summary.csv", index=False)

    storage_a = pd.read_csv(args.run_a / "storage_accounting.csv")
    storage_b = pd.read_csv(args.run_b / "low_rank_storage_accounting.csv")
    storage_facts = {"run_a_external_multiplier_min": float(storage_a.external_multiplier.min()), "run_a_external_multiplier_max": float(storage_a.external_multiplier.max()),
                     "run_b_external_multiplier_min": float(storage_b.external_multiplier.min()), "run_b_external_multiplier_max": float(storage_b.external_multiplier.max()),
                     "run_b_rows_over_5x": int((storage_b.external_multiplier > 5).sum())}
    (args.output / "storage_accounting.json").write_text(json.dumps(storage_facts, indent=2) + "\n")
    reference_flops = 2 * 3 * 2048 * 512
    compute = {"reference_expert_gemv_flops": reference_flops, "hardware_timing_claim": "none; analytical GB10 ratios only", "run_a_compute_summary": "run_a_compute_summary.csv", "rank_costs": {
        str(rank): {"c_h_flops": 2 * rank * 512, "u_q_flops": 2 * 2048 * rank, "total_flops": 2 * rank * (512 + 2048), "ratio_to_reference_expert": 2 * rank * (512 + 2048) / reference_flops}
        for rank in (32, 64, 128, 256, 512)}}
    (args.output / "compute_accounting.json").write_text(json.dumps(compute, indent=2) + "\n")

    # Run A figures.
    lineplot(primary, ["A0_current_Q2_whole_projection", "A6_corrected_screened_greedy"], "01_current_vs_corrected_allocator", plots)
    lineplot(run_a, ["A6_corrected_screened_greedy", "A4_fixed_sequence_naive", "A4_fixed_sequence_importance", "A4_fixed_sequence_pairwise", "A4_fixed_sequence_hypergraph"], "02_allocator_waterfall", plots)
    fig, ax = plt.subplots(figsize=(6.6, 5)); ax.scatter(primary_a.logical_bpw, primary_a.physical_bpw, s=12, alpha=.3); ax.plot([0, 4], [0, 4], "k--"); ax.set(xlabel="Logical bpw", ylabel="Physical bpw"); save(fig, plots, "03_logical_vs_physical_bpw")
    pages = run_a[run_a.method.str.startswith("A3_pages_")].copy(); p4 = primary_a.copy(); p4["page_size"] = 4096; pages = pd.concat([pages, p4], ignore_index=True); pages["page_size_label"] = pages.page_size.astype(str)
    fig, ax = plt.subplots(figsize=(7, 4.8)); pages.groupby(["page_size_label", "budget_bpw"]).page_amplification.median().unstack(0).plot(ax=ax, marker="o"); ax.set(ylabel="Median page amplification", xlabel="Budget (bpw)"); save(fig, plots, "04_page_amplification_vs_page_size")
    fig, ax = plt.subplots(figsize=(7, 4.8)); pages.groupby(["page_size_label", "budget_bpw"]).recovery.median().unstack(0).plot(ax=ax, marker="o"); ax.set(ylabel="Median recovery", xlabel="Budget (bpw)"); save(fig, plots, "05_recovery_vs_page_size")
    fig, ax = plt.subplots(figsize=(6.8, 4.6)); sequence.precision_after.value_counts(normalize=True).sort_index().plot.bar(ax=ax); ax.set(xlabel="Selected final precision for accepted action", ylabel="Fraction of actions"); save(fig, plots, "06_per_atom_precision_distribution")
    fig, ax = plt.subplots(figsize=(6.8, 4.6)); sequence.groupby("projection").physical_incremental_bytes.sum().pipe(lambda x: x / x.sum()).plot.bar(ax=ax); ax.set(ylabel="Fraction of physical bytes"); save(fig, plots, "07_projection_bandwidth_allocation")
    lineplot(run_a, ["A4_fixed_sequence_hypergraph", "A5_two_layout_H0", "A5_four_partial_layout_H0", "A4_gate_up_bundled"], "08_layout_replica_bundling", plots)
    lineplot(run_a, ["A6_corrected_screened_greedy", "bounded_exact_complete_marginal"], "09_bounded_exact_greedy_regret", plots)
    fig, ax = plt.subplots(figsize=(7, 4.8));
    for (method, threshold), part in inverse.groupby(["method", "threshold"]):
        values = np.sort(part.required_physical_bpw.dropna()); ax.step(values, np.arange(1, len(values) + 1) / len(part), where="post", label=f"{method} {threshold:.0%}")
    ax.set(xlabel="Required physical bpw", ylabel="Fraction of all invocations"); ax.legend(fontsize=7); save(fig, plots, "10_required_rate_cdf")
    fig, ax = plt.subplots(figsize=(7.2, 4.8)); primary_a.groupby(["layer", "budget_bpw"]).recovery.median().unstack(0).plot(ax=ax, marker="o"); ax.set(ylabel="Median recovery", xlabel="Budget (bpw)"); save(fig, plots, "11_layer_specific_curves")
    a2 = primary_a[primary_a.budget_bpw == 2].groupby(["layer", "expert_id"]).recovery.median().reset_index().merge(storage_a, on=["layer", "expert_id"]); fig, ax = plt.subplots(figsize=(6.8, 4.6)); ax.scatter(a2.external_multiplier, a2.recovery, alpha=.65); ax.set(xlabel="External storage multiplier", ylabel="Recovery at 2-bpw budget"); save(fig, plots, "12_storage_multiplier_vs_recovery")

    # Run B figures (all unselected curves are explicitly method controls).
    r8 = down_test[(down_test.fit == "projection") & (down_test.coefficient_bits == 8)]; fig, ax = plt.subplots(figsize=(7.2, 4.8)); r8.groupby(["requested_rank", "method"]).recovery.median().unstack(1).plot(ax=ax, marker="o"); ax.set(ylabel="Median authoritative-input down recovery", xlabel="Output rank"); save(fig, plots, "13_down_recovery_vs_rank")
    fig, ax = plt.subplots(figsize=(7.2, 4.8)); selected_b.groupby("budget_bpw").agg(physical_bpw=("physical_bpw", "median"), recovery=("recovery", "median")).plot(x="physical_bpw", y="recovery", ax=ax, marker="o", legend=False); ax.set(xlabel="Median actual physical bpw", ylabel="Median complete-expert recovery"); save(fig, plots, "14_complete_expert_vs_low_rank_bytes")
    fixed = run_b_all[(run_b_all.evaluation_split == "test") & (run_b_all.requested_rank == 512) & (run_b_all.coefficient_bits == 4) & (run_b_all.fit == "projection")]; lineplot(fixed, sorted(fixed.method.unique()), "15_shared_cluster_per_expert", plots)
    precision = run_b_all[(run_b_all.evaluation_split == "test") & (run_b_all.method == "B1_action_shared") & (run_b_all.requested_rank == 512) & (run_b_all.fit == "projection")]; fig, ax = plt.subplots(figsize=(7.2, 4.8)); precision.groupby(["coefficient_bits", "budget_bpw"]).recovery.median().unstack(0).plot(ax=ax, marker="o"); ax.set(ylabel="Median recovery", xlabel="Budget (bpw)"); save(fig, plots, "16_coefficient_precision")
    configs = run_b_all[run_b_all.evaluation_split == "test"].groupby(CONFIG + ["extra_flops", "down_bpw"]).recovery.median().reset_index(); fig, ax = plt.subplots(figsize=(7, 4.8)); scatter = ax.scatter(configs.extra_flops, configs.down_bpw, c=configs.recovery, s=18, alpha=.55); fig.colorbar(scatter, ax=ax, label="Median recovery"); ax.set(xlabel="Additional low-rank FLOPs", ylabel="Down physical bpw"); save(fig, plots, "17_down_bytes_vs_flops")
    spectra = pd.read_parquet(args.run_b / "low_rank_spectra.parquet"); fig, ax = plt.subplots(figsize=(7, 4.8));
    for method, part in spectra.groupby("method"):
        curve = part.groupby("component").cumulative_energy.median(); ax.plot(curve.index, curve.values, label=method)
    ax.set(xlim=(1, 512), xlabel="Output rank", ylabel="Cumulative energy within stored top-512 spectrum"); ax.legend(fontsize=7); save(fig, plots, "18_low_rank_spectra")
    lineplot(primary, ["A0_current_Q2_whole_projection", "B_validation_selected_deployable"], "19_native_vs_low_rank_pareto", plots)
    lineplot(primary, ["A0_current_Q2_whole_projection", "A6_corrected_screened_greedy", "B_validation_selected_deployable"], "20_run_a_run_b_frontier", plots)

    manifest = {"baseline_rows": len(baseline), "run_a_rows": len(run_a), "run_b_all_rows": len(run_b_all), "run_b_selected_test_rows": len(selected_b), "sequence_rows": len(sequence), "bootstrap_resamples": args.bootstrap, "plots_png": sorted(path.name for path in plots.glob("*.png"))}
    (args.output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
