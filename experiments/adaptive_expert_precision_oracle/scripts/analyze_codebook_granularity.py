#!/usr/bin/env python3
"""Regenerate all codebook-granularity tables and PNG/SVG figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERT_WEIGHTS = 3 * 512 * 2048
BASE_RESIDENT_BYTES = 884736
ALL_ROUTED_EXPERTS = 40 * 256


def savefig(figure: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        figure.savefig(directory / f"{stem}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def quantiles(values: pd.Series) -> tuple[float, float, float]:
    result = values.quantile([0.1, 0.5, 0.9])
    return float(result.iloc[0]), float(result.iloc[1]), float(result.iloc[2])


def metadata_accounting(result: Path) -> pd.DataFrame:
    rows = []
    root = result / "serialized_manifests" / "codecs"
    for directory in sorted(path for path in root.iterdir() if path.is_dir() and (path / "manifest.json").is_file()):
        manifest = json.loads((directory / "manifest.json").read_text())
        experts = max(len(manifest["experts"]), 1)
        personal = sum(int(value["bytes"]) for value in manifest.get("metadata_files", [])) / experts
        tables = sum(int(value["bytes"]) for value in manifest.get("table_files", []))
        sharing = manifest["sharing"]
        if sharing in ("independent_vector", "expert_projection"):
            table_per_expert = tables / experts
        elif sharing == "layer_projection":
            sampled_layers = len({int(value[0]) for value in manifest["experts"]})
            table_per_expert = tables / max(sampled_layers * 256, 1)
        else:
            table_per_expert = tables / ALL_ROUTED_EXPERTS
        selector_bytes = 0.0
        modifier_bytes = 0.0
        if int(manifest["selector_bits"]):
            selector_bytes = 3 * (512 * 2048 // int(manifest["group_size"])) * int(manifest["selector_bits"]) / 8
        if manifest["design"] == "C":
            modifier_bytes = 3 * 512 * 8
        header_padding = max(personal - selector_bytes - modifier_bytes, 0.0)
        metadata = personal + table_per_expert
        rows.append({
            "candidate": manifest["name"], "design": manifest["design"], "sharing": sharing,
            "families": int(manifest["families"]), "group_size": int(manifest["group_size"]),
            "selector_bytes": selector_bytes, "modifier_bytes": modifier_bytes,
            "header_padding_bytes": header_padding, "table_bytes_per_expert": table_per_expert,
            "metadata_bytes_per_expert": metadata, "metadata_bpw": 8 * metadata / EXPERT_WEIGHTS,
            "resident_bytes_per_expert": BASE_RESIDENT_BYTES + metadata,
            "resident_bpw": 8 * (BASE_RESIDENT_BYTES + metadata) / EXPERT_WEIGHTS,
            "pilot_serialized_table_bytes": tables,
        })
    return pd.DataFrame(rows)


def attach_common_recovery(dense: pd.DataFrame, selective: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    keys = ["request_id", "position", "layer", "expert_id", "split"]
    baseline = dense[(dense.candidate == "baseline_pr4_2p25") & (dense.level == 2)][keys + ["damage"]].rename(columns={"damage": "common_base_damage"})
    dense = dense.merge(baseline, on=keys, how="left")
    dense["common_recovery"] = 1.0 - dense.damage / dense.common_base_damage.clip(lower=1e-30)
    if selective is not None:
        test_base = baseline[baseline.split == "test"].drop(columns="split")
        selective = selective.merge(test_base, on=["request_id", "position", "layer", "expert_id"], how="left")
        selective["common_recovery"] = 1.0 - selective.damage / selective.common_base_damage.clip(lower=1e-30)
    return dense, selective


def select_validation_winners(dense: pd.DataFrame) -> pd.DataFrame:
    validation = dense[(dense.split == "validation") & (dense.level.isin([2, 3]))]
    scores = validation.groupby(["candidate", "design", "level"], as_index=False).agg(damage=("damage", "sum"), invocations=("damage", "size"))
    pivot = scores.pivot(index=["candidate", "design"], columns="level", values="damage").reset_index().rename(columns={2: "validation_q2_damage", 3: "validation_q3_damage"})
    selected = []
    for design, group in pivot.groupby("design"):
        best = group.validation_q2_damage.min()
        eligible = group[group.validation_q2_damage <= best * 1.005 + 1e-30]
        chosen = eligible.sort_values(["validation_q3_damage", "validation_q2_damage", "candidate"]).iloc[0]
        selected.append(chosen.to_dict())
    return pd.DataFrame(selected)


def summaries(
    dense: pd.DataFrame, projection: pd.DataFrame,
    selective: pd.DataFrame | None, output: Path,
) -> None:
    dense_rows = []
    for keys, group in dense.groupby(["candidate", "design", "split", "level"]):
        p10, median, p90 = quantiles(group.common_recovery)
        e10, emedian, e90 = quantiles(group.relative_output_error)
        damage10, damage_median, damage90 = quantiles(group.damage)
        dense_rows.append({
            "candidate": keys[0], "design": keys[1], "split": keys[2], "level": keys[3],
            "count": len(group), "p10_common_recovery": p10, "median_common_recovery": median,
            "p90_common_recovery": p90, "p10_relative_error": e10,
            "median_relative_error": emedian, "p90_relative_error": e90,
            "p10_damage": damage10, "mean_damage": group.damage.mean(),
            "median_damage": damage_median, "p90_damage": damage90,
        })
    pd.DataFrame(dense_rows).to_csv(output / "metrics" / "dense_summary.csv", index=False)

    projection_rows = []
    for keys, group in projection.groupby(["candidate", "design", "objective", "sharing", "projection", "split", "level"]):
        r10, rmed, r90 = quantiles(group.relative_error)
        d10, dmed, d90 = quantiles(group.damage)
        projection_rows.append({
            "candidate": keys[0], "design": keys[1], "objective": keys[2],
            "sharing": keys[3], "projection": keys[4], "split": keys[5], "level": keys[6],
            "count": len(group), "p10_relative_error": r10, "median_relative_error": rmed,
            "p90_relative_error": r90, "p10_damage": d10, "median_damage": dmed,
            "p90_damage": d90, "summed_damage": group.damage.sum(),
            "median_own_recovery": group.recovery.median(),
        })
    pd.DataFrame(projection_rows).to_csv(output / "metrics" / "projection_summary.csv", index=False)

    projection_test = projection[projection.split == "test"].copy()
    feature_path = output / "metrics" / "family_block_features.parquet"
    if feature_path.is_file():
        block_features = pd.read_parquet(feature_path)
        block_features["extreme_leaf_fraction"] = (
            block_features["leaf_07_fraction"] + block_features["leaf_15_fraction"]
        )
        expert_features = block_features.groupby(
            ["layer", "expert_id", "expert_stratum", "projection"], as_index=False,
        ).agg(
            mean_scale_exponent=("scale_exponent", "mean"),
            extreme_leaf_fraction=("extreme_leaf_fraction", "mean"),
            mean_distinct_leaves=("distinct_leaf_count", "mean"),
        )
        for source, target in (
            ("mean_scale_exponent", "block_scale_bin"),
            ("extreme_leaf_fraction", "rare_extreme_leaf_bin"),
        ):
            expert_features[target] = pd.qcut(
                expert_features[source].rank(method="first"), 3,
                labels=["low", "median", "high"],
            ).astype(str)
        projection_test = projection_test.merge(
            expert_features, on=["layer", "expert_id", "expert_stratum", "projection"], how="left",
        )
    projection_strata = []
    fields = ["layer", "expert_stratum", "projection"]
    if feature_path.is_file():
        fields += ["block_scale_bin", "rare_extreme_leaf_bin"]
    for field in fields:
        for keys, group in projection_test.groupby(["candidate", "design", "level", field]):
            e10, emed, e90 = quantiles(group.relative_error)
            projection_strata.append({
                "candidate": keys[0], "design": keys[1], "level": keys[2],
                "stratification": field, "stratum": keys[3], "count": len(group),
                "p10_relative_error": e10, "median_relative_error": emed,
                "p90_relative_error": e90, "summed_damage": group.damage.sum(),
                "median_own_recovery": group.recovery.median(),
            })
    pd.DataFrame(projection_strata).to_csv(
        output / "metrics" / "projection_heldout_stratified_summary.csv", index=False,
    )

    test = dense[dense.split == "test"].copy()
    test["activation_norm_bin"] = pd.qcut(
        test.activation_norm.rank(method="first"), 3, labels=["low", "median", "high"],
    ).astype(str)
    strata_rows = []
    for field in ("layer", "expert_stratum", "activation_norm_bin"):
        for keys, group in test.groupby(["candidate", "design", "level", field]):
            c10, cmed, c90 = quantiles(group.common_recovery)
            e10, emed, e90 = quantiles(group.relative_output_error)
            strata_rows.append({
                "candidate": keys[0], "design": keys[1], "level": keys[2],
                "stratification": field, "stratum": keys[3], "count": len(group),
                "p10_common_recovery": c10, "median_common_recovery": cmed,
                "p90_common_recovery": c90, "p10_relative_error": e10,
                "median_relative_error": emed, "p90_relative_error": e90,
                "summed_damage": group.damage.sum(),
            })
    pd.DataFrame(strata_rows).to_csv(output / "metrics" / "heldout_stratified_summary.csv", index=False)
    if selective is not None:
        selective_rows = []
        for keys, group in selective.groupby(["candidate", "design", "packet", "budget_bpw"]):
            p10, median, p90 = quantiles(group.common_recovery)
            own10, ownmed, own90 = quantiles(group.recovery)
            damage10, damage_median, damage90 = quantiles(group.damage)
            selective_rows.append({
                "candidate": keys[0], "design": keys[1], "packet": keys[2], "budget_bpw": keys[3],
                "count": len(group), "p10_common_recovery": p10, "median_common_recovery": median,
                "p90_common_recovery": p90, "p10_own_recovery": own10, "median_own_recovery": ownmed,
                "p90_own_recovery": own90, "p10_damage": damage10,
                "median_damage": damage_median, "p90_damage": damage90,
                "median_logical_bpw": group.logical_bpw.median(), "median_physical_bpw": group.physical_bpw.median(),
                "median_page_amplification": group.page_amplification.median(),
                "median_gate_bytes": group.gate_bytes.median(), "median_up_bytes": group.up_bytes.median(),
                "median_down_bytes": group.down_bytes.median(),
                "median_gate_q2_q3": group.gate_q2_q3.median(),
                "median_up_q2_q3": group.up_q2_q3.median(),
                "median_down_q2_q3": group.down_q2_q3.median(),
                "median_gate_q3_q4": group.gate_q3_q4.median(),
                "median_up_q3_q4": group.up_q3_q4.median(),
                "median_down_q3_q4": group.down_q3_q4.median(),
            })
        summary = pd.DataFrame(selective_rows)
        summary.to_csv(output / "metrics" / "selective_summary.csv", index=False)

        crossings = []
        targets = (0.90, 0.95, 0.975)
        for keys, group in summary.groupby(["candidate", "design", "packet"]):
            group = group.sort_values("budget_bpw")
            for statistic in ("p10_common_recovery", "median_common_recovery", "p90_common_recovery"):
                x = group.budget_bpw.to_numpy(np.float64)
                y = group[statistic].to_numpy(np.float64)
                for target in targets:
                    crossing = np.nan
                    first = np.flatnonzero(y >= target)
                    if len(first):
                        right = int(first[0])
                        if right == 0 or y[right] <= y[right - 1] + 1e-15:
                            crossing = float(x[right])
                        else:
                            crossing = float(x[right - 1] + (target - y[right - 1]) * (x[right] - x[right - 1]) / (y[right] - y[right - 1]))
                    crossings.append({
                        "candidate": keys[0], "design": keys[1], "packet": keys[2],
                        "statistic": statistic, "target_recovery": target,
                        "interpolated_physical_bpw": crossing,
                    })
        pd.DataFrame(crossings).to_csv(output / "metrics" / "recovery_crossings.csv", index=False)


def plot_all(result: Path, dense: pd.DataFrame, selective: pd.DataFrame | None, accounting: pd.DataFrame, winners: pd.DataFrame) -> None:
    plots = result / "plots"
    test2 = dense[(dense.split == "test") & (dense.level == 2)].groupby("candidate").agg(error=("relative_output_error", "median"), damage=("damage", "median"), recovery=("common_recovery", "median")).reset_index().merge(accounting, on="candidate")
    test3 = dense[(dense.split == "test") & (dense.level == 3)].groupby("candidate").agg(recovery=("common_recovery", "median"), own=("recovery", "median"), damage=("damage", "median")).reset_index().merge(accounting, on="candidate")

    fig, ax = plt.subplots(figsize=(8, 5)); ax.scatter(test2.metadata_bpw, test2.error)
    for row in test2.itertuples(): ax.annotate(row.candidate, (row.metadata_bpw, row.error), fontsize=6)
    ax.set(xlabel="resident metadata (bpw)", ylabel="median complete-expert relative error", title="Resident quality versus local metadata"); ax.grid(alpha=.25); savefig(fig, plots, "01_resident_error_vs_metadata")

    fig, ax = plt.subplots(figsize=(8, 5)); ax.scatter(test3.metadata_bpw, test3.own)
    for row in test3.itertuples(): ax.annotate(row.candidate, (row.metadata_bpw, row.own), fontsize=6)
    ax.set(xlabel="resident metadata (bpw)", ylabel="median fraction of own Q2 damage removed", title="Dense Q3 recovery"); ax.grid(alpha=.25); savefig(fig, plots, "02_dense_q3_recovery_vs_metadata")

    if selective is not None:
        curve = selective.groupby(["candidate", "packet", "budget_bpw"], as_index=False).common_recovery.median()
        fig, ax = plt.subplots(figsize=(8, 5))
        for keys, group in curve.groupby(["candidate", "packet"]): ax.plot(group.budget_bpw, group.common_recovery, marker="o", label=f"{keys[0]}:{keys[1]}")
        ax.set(xlabel="physical streamed bpw", ylabel="median common recovery", title="Selective complete-expert frontier"); ax.grid(alpha=.25); ax.legend(fontsize=6, ncol=2); savefig(fig, plots, "03_selective_recovery_vs_physical")

        chosen = curve.groupby(["candidate", "packet"]).common_recovery.mean().idxmax()
        group = selective[(selective.candidate == chosen[0]) & (selective.packet == chosen[1])]
        quant = group.groupby("budget_bpw").common_recovery.quantile([.1,.5,.9]).unstack()
        fig, ax = plt.subplots(figsize=(8, 5)); ax.fill_between(quant.index, quant[.1], quant[.9], alpha=.2); ax.plot(quant.index, quant[.5], marker="o")
        ax.set(xlabel="physical streamed bpw", ylabel="common recovery", title=f"p10/median/p90: {chosen[0]} {chosen[1]}"); ax.grid(alpha=.25); savefig(fig, plots, "04_selective_quantile_frontier")

        one = selective[np.isclose(selective.budget_bpw, 1.0)].groupby(["candidate", "design", "packet"], as_index=False).common_recovery.median()
        one = one.sort_values("common_recovery").groupby("design").tail(1).sort_values("common_recovery")
        fig, ax = plt.subplots(figsize=(8, 5)); ax.bar(one.design + ":" + one.packet, one.common_recovery); ax.tick_params(axis="x", rotation=35); ax.set(ylabel="median common recovery", title="Design A/B/C/D/E at one physical bpw"); savefig(fig, plots, "05_design_comparison_one_bpw")

    eps = dense[(dense.split == "validation") & dense.candidate.str.contains("global_functional")].groupby(["candidate", "q2_tolerance", "level"], as_index=False).damage.sum()
    if len(eps):
        pivot = eps.pivot(index=["candidate", "q2_tolerance"], columns="level", values="damage").reset_index()
        fig, ax = plt.subplots(figsize=(7, 5)); ax.plot(pivot.q2_tolerance * 100, 1 - pivot[3] / pivot[2], marker="o"); ax.set(xlabel="allowed Q2 tolerance (%)", ylabel="Q3 fraction of Q2 damage removed", title="Q2/Q3 lexicographic tradeoff"); ax.grid(alpha=.25); savefig(fig, plots, "06_q2_tolerance_vs_q3")

    family = test3[test3.candidate.str.startswith("E")].sort_values("families")
    fig, ax = plt.subplots(figsize=(7, 5)); ax.plot(family.families, family.own, marker="o"); ax.set(xlabel="families", ylabel="median dense Q3 recovery", title="Family-size saturation at G64"); ax.grid(alpha=.25); savefig(fig, plots, "07_family_size_vs_quality")

    granularity_names = ["D4_g16_layer_functional", "B16_g32_layer_functional", "E16_g64_layer_functional"]
    granularity = test3[test3.candidate.isin(granularity_names)].sort_values("group_size")
    fig, ax = plt.subplots(figsize=(7, 5)); ax.plot(granularity.group_size, granularity.own, marker="o"); ax.set(xlabel="weights per selector", ylabel="median dense Q3 recovery", title="Group granularity versus quality"); ax.grid(alpha=.25); savefig(fig, plots, "08_group_granularity_vs_quality")

    generalization = dense[dense.level == 2].groupby(["candidate", "split"], as_index=False).relative_output_error.median()
    top = winners.candidate.tolist()
    generalization = generalization[generalization.candidate.isin(top)]
    pivot = generalization.pivot(index="candidate", columns="split", values="relative_output_error")
    fig, ax = plt.subplots(figsize=(9, 5)); pivot.plot.bar(ax=ax); ax.set(ylabel="median relative error", title="Train versus validation/test resident quality"); ax.tick_params(axis="x", rotation=30); savefig(fig, plots, "09_train_validation_test")

    utilization_path = result / "metrics" / "family_utilization.csv"
    feature_path = result / "metrics" / "family_block_features.parquet"
    if utilization_path.is_file():
        utilization = pd.read_csv(utilization_path)
        fig, ax = plt.subplots(figsize=(7, 5)); ax.hist(utilization.effective_families, bins=16); ax.set(xlabel="effective families exp(H)", ylabel="expert/projection groups", title="Family utilization"); savefig(fig, plots, "10_family_utilization_histogram")
    if feature_path.is_file():
        features = pd.read_parquet(feature_path)
        for stratum in ("cold", "median", "hot"):
            features[f"stratum_{stratum}"] = (features.expert_stratum == stratum).astype(float)
        columns = [
            "zero_fraction", "half_fraction", "sign_imbalance", "abs_mean", "variance",
            "max_rms_ratio", "scale_exponent", "activation_weighted_importance", "layer",
            "stratum_cold", "stratum_median", "stratum_hot",
        ] + [f"leaf_{leaf:02d}_fraction" for leaf in range(16)]
        means = features.groupby(["projection", "family_id"])[columns].mean()
        means.reset_index().to_csv(
            result / "metrics" / "family_specialization_summary.csv", index=False,
        )
        # Normalize within projection because family IDs are local to each of
        # the three independently fitted projection tables.
        normalized = means.groupby(level="projection", group_keys=False).apply(
            lambda value: (value - value.mean()) / value.std().replace(0, 1),
            include_groups=False,
        )
        labels = [f"{projection}:{family}" for projection, family in normalized.index]
        fig, ax = plt.subplots(figsize=(14, 10)); image = ax.imshow(normalized.T, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2); ax.set(xticks=range(len(normalized)), xticklabels=labels, yticks=range(len(columns)), yticklabels=columns, xlabel="projection-local family", title="Family specialization (standardized within projection)"); ax.tick_params(axis="x", rotation=90, labelsize=6); ax.tick_params(axis="y", labelsize=7); fig.colorbar(image, ax=ax); savefig(fig, plots, "11_family_specialization_heatmap")

    breakdown = accounting.set_index("candidate")[["selector_bytes", "modifier_bytes", "header_padding_bytes", "table_bytes_per_expert"]]
    fig, ax = plt.subplots(figsize=(11, 5)); breakdown.plot.bar(stacked=True, ax=ax); ax.set(ylabel="serialized bytes per expert", title="Resident metadata breakdown"); ax.tick_params(axis="x", rotation=60, labelsize=6); savefig(fig, plots, "12_metadata_bytes_breakdown")

    if selective is not None:
        one = selective[np.isclose(selective.budget_bpw, 1.0)].groupby(["candidate", "packet"])[["gate_bytes", "up_bytes", "down_bytes"]].median()
        fig, ax = plt.subplots(figsize=(10, 5)); one.plot.bar(stacked=True, ax=ax); ax.set(ylabel="median physical bytes", title="Gate/up/down bandwidth at one bpw"); ax.tick_params(axis="x", rotation=45, labelsize=7); savefig(fig, plots, "13_projection_bandwidth_allocation")
        damage = selective.groupby(["candidate", "packet", "budget_bpw"], as_index=False).damage.median()
        fig, ax = plt.subplots(figsize=(8, 5));
        for keys, group in damage.groupby(["candidate", "packet"]): ax.semilogy(group.budget_bpw, group.damage, marker="o", label=f"{keys[0]}:{keys[1]}")
        ax.set(xlabel="physical streamed bpw", ylabel="median absolute damage vs MXFP4", title="Absolute complete-expert damage"); ax.grid(alpha=.25); ax.legend(fontsize=6, ncol=2); savefig(fig, plots, "14_absolute_damage_vs_mxfp4")

    switches = {"baseline": 1/EXPERT_WEIGHTS, "A": 1/2048, "B": 1/32, "C": 1/32, "D": 1/16, "E": 1/64, "learned_16_leaf": 1/EXPERT_WEIGHTS}
    decode = test2.copy(); decode["switches_per_weight"] = decode.design.map(switches); decode["quality_gain"] = decode.recovery
    fig, ax = plt.subplots(figsize=(8, 5)); ax.scatter(decode.switches_per_weight, decode.quality_gain)
    for row in decode.itertuples(): ax.annotate(row.candidate, (row.switches_per_weight, row.quality_gain), fontsize=6)
    ax.set(xscale="log", xlabel="table switches per weight (estimate)", ylabel="median resident common recovery gain", title="Decode overhead versus quality gain"); ax.grid(alpha=.25); savefig(fig, plots, "15_decode_cost_vs_quality")

    if selective is not None:
        pareto = selective.groupby(["candidate", "packet", "budget_bpw"], as_index=False).common_recovery.median().merge(accounting[["candidate", "resident_bpw"]], on="candidate")
        fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection="3d"); scatter = ax.scatter(pareto.resident_bpw, pareto.budget_bpw, pareto.common_recovery, c=pareto.common_recovery, cmap="viridis"); ax.set(xlabel="resident bpw", ylabel="streamed bpw", zlabel="median common recovery", title="Resident/streamed/quality Pareto cloud"); fig.colorbar(scatter, ax=ax, shrink=.6); savefig(fig, plots, "16_pareto_frontier")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    dense = pd.read_parquet(args.result / "metrics" / "dense_metrics.parquet")
    projection = pd.read_parquet(args.result / "metrics" / "projection_metrics.parquet")
    selective_path = args.result / "metrics" / "selective_metrics.parquet"
    selective = pd.read_parquet(selective_path) if selective_path.is_file() else None
    dense, selective = attach_common_recovery(dense, selective)
    accounting = metadata_accounting(args.result)
    accounting.to_csv(args.result / "metrics" / "metadata_accounting.csv", index=False)
    winners = select_validation_winners(dense)
    winners.to_csv(args.result / "metrics" / "validation_selection.csv", index=False)
    summaries(dense, projection, selective, args.result)
    plot_all(args.result, dense, selective, accounting, winners)
    manifest = {"command": "python scripts/analyze_codebook_granularity.py --result <result>", "plots": sorted(path.name for path in (args.result / "plots").iterdir()), "dense_rows": len(dense), "selective_rows": 0 if selective is None else len(selective)}
    (args.result / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
