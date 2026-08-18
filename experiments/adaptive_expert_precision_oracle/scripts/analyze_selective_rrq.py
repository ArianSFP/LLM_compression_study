#!/usr/bin/env python3
"""Summarize and plot the selective RRQ retention pilot from saved Parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def savefig(root: Path, name: str) -> None:
    for suffix in ("png", "svg"):
        plt.savefig(root / f"{name}.{suffix}", dpi=190, bbox_inches="tight")
    plt.close()


def bootstrap(values: pd.DataFrame, column: str, seed: int, resamples: int = 1000) -> tuple[float, float]:
    grouped = {}
    for key, group in values.groupby("request_id"):
        finite = group[column].to_numpy(float)
        finite = finite[np.isfinite(finite)]
        if len(finite):
            grouped[key] = finite
    if not grouped:
        return float("nan"), float("nan")
    keys = list(grouped); rng = np.random.default_rng(seed); estimates = []
    for _ in range(resamples):
        chosen = rng.choice(keys, len(keys), replace=True)
        estimates.append(float(np.median(np.concatenate([grouped[key] for key in chosen]))))
    return float(np.quantile(estimates, .025)), float(np.quantile(estimates, .975))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--result", type=Path, required=True); parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args(); plots = args.result / "plots"; plots.mkdir(exist_ok=True)
    source_metrics = pd.read_parquet(args.result / "selective_metrics.parquet")
    source_metrics.to_parquet(args.result / "metrics.parquet", index=False)
    metrics = source_metrics[source_metrics.split == "test"].copy()
    metrics["group"] = np.where(metrics.layer == 0, "layer 0", "layers 4/20/39")
    metrics = pd.concat([metrics, metrics.assign(group="all four layers")], ignore_index=True)
    rows = []
    for (group, budget), frame in metrics.groupby(["group", "budget_bpw"], sort=True):
        ci_low, ci_high = bootstrap(frame, "recovery", args.seed + int(100 * budget))
        eta_low, eta_high = bootstrap(frame, "eta", args.seed + 1000 + int(100 * budget))
        retention_low, retention_high = bootstrap(frame, "benefit_retention", args.seed + 2000 + int(100 * budget))
        rows.append({"group": group, "budget_bpw": budget, "invocations": len(frame), "requests": frame.request_id.nunique(), "recovery_p10": frame.recovery.quantile(.1), "recovery_median": frame.recovery.median(), "recovery_p90": frame.recovery.quantile(.9), "recovery_mean": frame.recovery.mean(), "recovery_median_ci_low": ci_low, "recovery_median_ci_high": ci_high, "eta_p10": frame.eta.quantile(.1), "eta_median": frame.eta.median(), "eta_p90": frame.eta.quantile(.9), "eta_median_ci_low": eta_low, "eta_median_ci_high": eta_high, "benefit_retention_p10": frame.benefit_retention.quantile(.1), "benefit_retention_median": frame.benefit_retention.median(), "benefit_retention_p90": frame.benefit_retention.quantile(.9), "benefit_retention_median_ci_low": retention_low, "benefit_retention_median_ci_high": retention_high, "dense_comparator_recovery_median": frame.dense_comparator_recovery.median(), "actual_physical_bpw_median": frame.physical_bpw.median(), "logical_bpw_median": frame.logical_bpw.median(), "page_amplification_median": frame.page_amplification.median(), "relative_output_error_median": frame.relative_output_error.median(), "gate_avg_expert_bpw_median": frame.gate_bpw.median(), "up_avg_expert_bpw_median": frame.up_bpw.median(), "down_avg_expert_bpw_median": frame.down_bpw.median(), "gate_projection_bpw_median": 3 * frame.gate_bpw.median(), "up_projection_bpw_median": 3 * frame.up_bpw.median(), "down_projection_bpw_median": 3 * frame.down_bpw.median()})
    summary = pd.DataFrame(rows)
    summary.to_csv(args.result / "selective_retention_summary.csv", index=False)
    summary.to_csv(args.result / "summary.csv", index=False)
    detail = metrics[metrics.group != "all four layers"].copy()
    aggregation = dict(
        invocations=("recovery", "size"), requests=("request_id", "nunique"),
        recovery_p10=("recovery", lambda x: x.quantile(.1)), recovery_median=("recovery", "median"),
        recovery_p90=("recovery", lambda x: x.quantile(.9)), benefit_retention_median=("benefit_retention", "median"),
        physical_bpw_median=("physical_bpw", "median"), logical_bpw_median=("logical_bpw", "median"),
        page_amplification_median=("page_amplification", "median"), relative_output_error_median=("relative_output_error", "median"),
    )
    detail.groupby(["layer", "budget_bpw"]).agg(**aggregation).reset_index().to_csv(args.result / "per_layer.csv", index=False)
    detail.groupby(["layer", "expert_id", "expert_stratum", "budget_bpw"]).agg(**aggregation).reset_index().to_csv(args.result / "per_expert.csv", index=False)
    prior_path = args.result.parent / "qwen36_q2_corrected_allocator_20260818_v7" / "allocator_metrics.parquet"
    if prior_path.exists():
        prior = pd.read_parquet(prior_path)
        prior = prior[prior.method == "A6_corrected_screened_greedy"]
        pair_keys = ["request_id", "sequence_id", "position", "layer", "expert_id", "budget_bpw"]
        paired = source_metrics[source_metrics.split == "test"].merge(
            prior[pair_keys + ["recovery", "physical_bpw"]], on=pair_keys, suffixes=("_rrq", "_pr2")
        )
        paired["recovery_difference"] = paired.recovery_rrq - paired.recovery_pr2
        paired["group"] = np.where(paired.layer == 0, "layer 0", "layers 4/20/39")
        paired = pd.concat([paired, paired.assign(group="all four layers")], ignore_index=True)
        paired_rows = []
        for (group, budget), frame in paired.groupby(["group", "budget_bpw"]):
            low, high = bootstrap(frame, "recovery_difference", args.seed + 3000 + int(100 * budget))
            paired_rows.append({
                "group": group, "budget_bpw": budget, "paired_invocations": len(frame),
                "requests": frame.request_id.nunique(), "rrq_recovery_median": frame.recovery_rrq.median(),
                "pr2_recovery_median": frame.recovery_pr2.median(),
                "paired_difference_p10": frame.recovery_difference.quantile(.1),
                "paired_difference_median": frame.recovery_difference.median(),
                "paired_difference_p90": frame.recovery_difference.quantile(.9),
                "paired_difference_mean": frame.recovery_difference.mean(),
                "paired_difference_median_ci_low": low, "paired_difference_median_ci_high": high,
            })
        pd.DataFrame(paired_rows).to_csv(args.result / "paired_vs_pr2.csv", index=False)

    support = pd.read_parquet(args.result / "stage_support.parquet")
    support["group"] = np.where(support.layer == 0, "layer 0", "layers 4/20/39")
    support = pd.concat([support, support.assign(group="all four layers")], ignore_index=True)
    support_summary = support.groupby(["group", "projection", "stage", "target_recovery"], dropna=False).agg(invocations=("request_id", "size"), required_atoms_p10=("required_atoms", lambda x: x.quantile(.1)), required_atoms_median=("required_atoms", "median"), required_atoms_p90=("required_atoms", lambda x: x.quantile(.9)), required_bpw_median=("required_physical_bpw_projection", "median")).reset_index()
    support_summary.to_csv(args.result / "stage_support_summary.csv", index=False)
    basis = pd.read_parquet(args.result / "basis_full_support.parquet")
    basis = basis[basis.split == "test"].copy()
    basis["group"] = np.where(basis.layer == 0, "layer 0", "layers 4/20/39")
    basis = pd.concat([basis, basis.assign(group="all four layers")], ignore_index=True)
    basis_summary = basis.groupby(["group", "projection", "basis", "stage"]).agg(invocations=("request_id", "size"), recovery_p10=("recovery", lambda x: x.quantile(.1)), recovery_median=("recovery", "median"), recovery_p90=("recovery", lambda x: x.quantile(.9))).reset_index()
    basis_summary.to_csv(args.result / "basis_full_support_summary.csv", index=False)
    ranking = pd.read_parquet(args.result / "ranking_regret.parquet")
    supplement = args.result / "ranking_regret_supplement.parquet"
    if supplement.exists():
        ranking = pd.concat([ranking, pd.read_parquet(supplement)], ignore_index=True)
    crossing = ranking[ranking.comparison == "energy_vs_exact_residual_greedy"].copy()
    crossing.groupby(["projection", "stage", "target_recovery"]).agg(cases=("request_id", "size"), energy_atoms_median=("energy_atoms", "median"), exact_atoms_median=("exact_atoms", "median"), extra_atoms_median=("extra_atoms", "median"), extra_atoms_p90=("extra_atoms", lambda x: x.quantile(.9)), extra_projection_bpw_median=("extra_projection_bpw", "median"), extra_projection_bpw_p90=("extra_projection_bpw", lambda x: x.quantile(.9))).reset_index().to_csv(args.result / "ranking_regret_summary.csv", index=False)
    queue = ranking[ranking.comparison == "complete_expert_queue_greedy"].copy()
    if len(queue):
        audit_keys = ["request_id", "layer", "expert_id", "budget_bpw"]
        scalable = metrics[metrics.group != "all four layers"][audit_keys + ["recovery", "physical_bpw"]].drop_duplicates(audit_keys)
        queue = queue.merge(scalable, on=audit_keys, how="left", suffixes=("_queue", "_scalable"))
        queue["queue_minus_scalable_recovery"] = queue.recovery_queue - queue.recovery_scalable
        queue.groupby("budget_bpw").agg(
            cases=("request_id", "size"),
            queue_recovery_median=("recovery_queue", "median"),
            scalable_recovery_median=("recovery_scalable", "median"),
            queue_minus_scalable_median=("queue_minus_scalable_recovery", "median"),
            queue_minus_scalable_p10=("queue_minus_scalable_recovery", lambda x: x.quantile(.1)),
            queue_minus_scalable_p90=("queue_minus_scalable_recovery", lambda x: x.quantile(.9)),
        ).reset_index().to_csv(args.result / "complete_expert_queue_regret_summary.csv", index=False)
    beam = ranking[ranking.comparison.str.startswith("bounded_beam", na=False)].copy()
    if len(beam):
        beam.groupby(["projection", "stage", "comparison", "k"]).agg(
            cases=("request_id", "size"), recovery_median=("recovery", "median"),
            recovery_p10=("recovery", lambda x: x.quantile(.1)),
            recovery_p90=("recovery", lambda x: x.quantile(.9)),
        ).reset_index().to_csv(args.result / "bounded_beam_summary.csv", index=False)
    actions = pd.read_parquet(args.result / "selected_actions.parquet")
    actions.groupby(["budget_bpw", "projection", "stage"]).size().rename("selected_actions").reset_index().to_csv(args.result / "selected_precision_distribution.csv", index=False)
    allocation_rows = []
    difficult = detail[detail.layer.isin([4, 20, 39])]
    for budget, frame in difficult.groupby("budget_bpw"):
        modes = frame.groupby(["gate_bpw", "up_bpw", "down_bpw"]).size().sort_values(ascending=False)
        mode_gate, mode_up, mode_down = modes.index[0]
        allocation_rows.append({
            "budget_bpw": budget, "invocations": len(frame),
            "gate_bpw_mean": frame.gate_bpw.mean(), "up_bpw_mean": frame.up_bpw.mean(), "down_bpw_mean": frame.down_bpw.mean(),
            "gate_bpw_median": frame.gate_bpw.median(), "up_bpw_median": frame.up_bpw.median(), "down_bpw_median": frame.down_bpw.median(),
            "mode_gate_bpw": mode_gate, "mode_up_bpw": mode_up, "mode_down_bpw": mode_down,
            "mode_invocations": int(modes.iloc[0]), "mode_fraction": float(modes.iloc[0] / len(frame)),
        })
    allocation_summary = pd.DataFrame(allocation_rows)
    allocation_summary.to_csv(args.result / "allocation_summary.csv", index=False)

    sns.set_theme(style="whitegrid", context="talk")
    for column, name, ylabel in (("recovery", "selective_recovery_frontier", "Complete-expert recovery"), ("benefit_retention", "selective_dense_retention", "Selective / dense recovered benefit"), ("eta", "eta_literal_dense_over_selective", "η: dense / selective recovered benefit")):
        plt.figure(figsize=(9, 5.5)); sns.lineplot(data=metrics[metrics.group != "all four layers"], x="budget_bpw", y=column, hue="group", estimator="median", errorbar=("pi", 80), marker="o"); plt.xlabel("Physical correction budget (bpw)"); plt.ylabel(ylabel); savefig(plots, name)
    plt.figure(figsize=(10, 6)); sns.lineplot(data=support_summary[support_summary.group == "layers 4/20/39"], x="target_recovery", y="required_atoms_median", hue="projection", style="stage", markers=True); plt.ylabel("Median required atoms"); plt.xlabel("Stage-output recovery target"); savefig(plots, "stage_specific_support_size")
    grid = sns.catplot(data=basis_summary[basis_summary.group == "layers 4/20/39"], x="basis", y="recovery_median", hue="projection", col="stage", kind="bar", height=5.5, aspect=1.0, sharey=True)
    grid.set_axis_labels("Basis", "Full-support median recovery")
    grid.set_titles("RRQ stage {col_name}")
    for axis in grid.axes.flat:
        axis.tick_params(axis="x", rotation=30)
    for suffix in ("png", "svg"):
        grid.savefig(plots / f"stage_basis_full_support.{suffix}", dpi=190, bbox_inches="tight")
    plt.close("all")
    if len(crossing):
        plt.figure(figsize=(10, 6)); sns.barplot(data=crossing, x="target_recovery", y="extra_atoms", hue="stage", estimator="median", errorbar=("pi", 80)); plt.ylabel("Extra atoms: diagonal energy minus exact greedy"); savefig(plots, "ranking_regret")
    if len(queue):
        plt.figure(figsize=(9, 5.5))
        sns.lineplot(data=queue, x="budget_bpw", y="recovery_queue", estimator="median", marker="o", label="nonlinear queue greedy")
        sns.lineplot(data=queue, x="budget_bpw", y="recovery_scalable", estimator="median", marker="o", label="scalable top-16 shortlist")
        plt.xlabel("Physical correction budget (bpw)"); plt.ylabel("Complete-expert recovery"); savefig(plots, "complete_expert_search_regret")
    allocation = allocation_summary.sort_values("budget_bpw")
    plt.figure(figsize=(9, 5.5)); plt.stackplot(allocation.budget_bpw, allocation.gate_bpw_mean, allocation.up_bpw_mean, allocation.down_bpw_mean, labels=["gate", "up", "down"]); plt.legend(); plt.xlabel("Physical correction budget (bpw)"); plt.ylabel("Mean allocated average-expert bpw"); savefig(plots, "sequential_expert_allocation")
    storage = {
        "min_external_storage_multiplier": float(metrics.storage_multiplier.min()),
        "median_external_storage_multiplier": float(metrics.storage_multiplier.median()),
        "max_external_storage_multiplier": float(metrics.storage_multiplier.max()),
        "reference_fallback_included": True,
        "resident_q2_external_duplicate_included": False,
        "resident_q2_bytes_per_expert": 884928,
        "resident_q2_effective_bpw": 2.25048828125,
        "scale_tables_streamed": True,
        "primary_page_bytes": 512,
    }
    (args.result / "storage_accounting.json").write_text(json.dumps(storage, indent=2, sort_keys=True) + "\n")
    action_keys = ["request_id", "sequence_id", "position", "layer", "expert_id", "budget_bpw"]
    actions["accumulation_flops"] = np.where(actions.projection == "down", 4096, 1024)
    accum = actions.groupby(action_keys).accumulation_flops.sum().rename("atom_accumulation_flops")
    union = actions[actions.projection.isin(["gate", "up"])].groupby(action_keys).atom_id.nunique().mul(4096).rename("selected_row_transform_flops")
    compute_rows = detail.set_index(action_keys).join(accum).join(union).fillna({"atom_accumulation_flops": 0, "selected_row_transform_flops": 0}).reset_index()
    compute_rows["deployment_extra_flops"] = compute_rows.atom_accumulation_flops + compute_rows.selected_row_transform_flops
    compute_rows["ratio_to_one_complete_expert_gemv"] = compute_rows.deployment_extra_flops / 6291456
    compute_frontier = compute_rows[compute_rows.layer.isin([4, 20, 39])].groupby("budget_bpw").agg(
        invocations=("recovery", "size"),
        atom_accumulation_flops_median=("atom_accumulation_flops", "median"),
        selected_row_transform_flops_median=("selected_row_transform_flops", "median"),
        deployment_extra_flops_median=("deployment_extra_flops", "median"),
        ratio_to_one_complete_expert_gemv_median=("ratio_to_one_complete_expert_gemv", "median"),
    ).reset_index()
    compute_frontier.to_csv(args.result / "compute_frontier.csv", index=False)
    compute = {
        "selection": "H0 oracle; diagonal scores plus linearized allocation shortlist",
        "exact_final_candidates_per_budget": 16,
        "atom_accumulation_flops": "1024 per selected gate/up atom-stage; 4096 per selected down atom-stage",
        "selected_row_transform_flops": "4096 per unique gate/up analysis row; row is shared between gate and up for one expert invocation",
        "full_transform_oracle_flops_per_layer_token": 8388608,
        "full_transform_is_oracle_only": True,
        "analysis_basis_resident_bytes_fp16_per_layer": 8388608,
        "one_complete_expert_gemv_flops": 6291456,
        "spark_timing_measured": False,
        "rtx_3090_end_to_end_kernel_timing_measured": False,
        "difficult_layer_frontier": compute_frontier.to_dict("records"),
    }
    (args.result / "compute_accounting.json").write_text(json.dumps(compute, indent=2, sort_keys=True) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
