#!/usr/bin/env python3
"""Cluster-bootstrap complete-expert and resident-base sensitivity statistics."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


INVOCATION = ["request_id", "position", "layer", "expert_id", "router_rank"]


def add_global(data: pd.DataFrame) -> pd.DataFrame:
    keys = INVOCATION + ["budget_bpw", "down_oracle_variant"]
    selected = data.loc[data.groupby(keys).recovery.idxmax()].copy()
    selected["selected_static_method"] = selected.method
    selected["method"] = "global_multi_view_oracle"
    return pd.concat([data, selected], ignore_index=True)


def bootstrap(group: pd.DataFrame, column: str, statistic: str, rng: np.random.Generator, count: int) -> tuple[float, float]:
    clusters = {key: value[column].to_numpy(float) for key, value in group.groupby("request_id")}
    names = np.asarray(list(clusters), dtype=object)
    values = np.empty(count)
    for index in range(count):
        sample = np.concatenate([clusters[name] for name in rng.choice(names, len(names), replace=True)])
        values[index] = np.mean(sample) if statistic == "mean" else np.median(sample)
    return float(np.quantile(values, .025)), float(np.quantile(values, .975))


def fixed_statistics(data: pd.DataFrame, base: str, resamples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    seq = data[data.down_oracle_variant == "sequential"]
    for key, group in seq.groupby(["method", "budget_bpw"]):
        mean_ci = bootstrap(group, "recovery", "mean", rng, resamples)
        median_ci = bootstrap(group, "recovery", "median", rng, resamples)
        rows.append({
            "resident_base": base, "method": key[0], "budget_bpw": key[1],
            "invocations": len(group), "requests": group.request_id.nunique(),
            "mean_recovery": group.recovery.mean(), "mean_ci_low": mean_ci[0], "mean_ci_high": mean_ci[1],
            "median_recovery": group.recovery.median(), "median_ci_low": median_ci[0], "median_ci_high": median_ci[1],
            "p10_recovery": group.recovery.quantile(.1), "p90_recovery": group.recovery.quantile(.9),
            "p95_damage_fraction": (1 - group.recovery).quantile(.95), "p99_damage_fraction": (1 - group.recovery).quantile(.99),
            "relative_output_error_median": group.relative_output_error.median(),
            "actual_physical_bpw_median": group.physical_bpw.median(),
            "gate_bpw_median": group.gate_bpw.median(), "up_bpw_median": group.up_bpw.median(), "down_bpw_median": group.down_bpw.median(),
            "fraction_above_80": (group.recovery >= .8).mean(), "fraction_above_90": (group.recovery >= .9).mean(),
            "fraction_above_95": (group.recovery >= .95).mean(), "fraction_above_99": (group.recovery >= .99).mean(),
        })
    return pd.DataFrame(rows)


def inverse_rates(data: pd.DataFrame, base: str) -> pd.DataFrame:
    seq = data[(data.down_oracle_variant == "sequential") & (data.method == "global_multi_view_oracle")]
    rows = []
    for key, group in seq.groupby(INVOCATION):
        group = group.sort_values("physical_bpw")
        rate = group.physical_bpw.to_numpy(float)
        recovery = np.maximum.accumulate(group.recovery.to_numpy(float))
        for threshold in (.8, .9, .95, .975, .99):
            hit = np.flatnonzero(recovery >= threshold)
            rows.append({**dict(zip(INVOCATION, key)), "resident_base": base, "threshold": threshold, "required_physical_bpw": float(rate[hit[0]]) if len(hit) else math.inf})
    return pd.DataFrame(rows)


def paired_difference(left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str, resamples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    left = left[(left.down_oracle_variant == "sequential") & (left.method == "global_multi_view_oracle")]
    right = right[(right.down_oracle_variant == "sequential") & (right.method == "global_multi_view_oracle")]
    keys = INVOCATION + ["budget_bpw"]
    merged = left[keys + ["recovery", "relative_output_error", "base_damage"]].merge(
        right[keys + ["recovery", "relative_output_error", "base_damage"]], on=keys, suffixes=("_left", "_right")
    )
    for budget, group in merged.groupby("budget_bpw"):
        group = group.copy()
        group["recovery_difference"] = group.recovery_left - group.recovery_right
        group["relative_error_difference"] = group.relative_output_error_left - group.relative_output_error_right
        ci = bootstrap(group, "recovery_difference", "mean", rng, resamples)
        rows.append({
            "left": left_name, "right": right_name, "budget_bpw": budget, "paired_invocations": len(group),
            "mean_recovery_difference": group.recovery_difference.mean(), "recovery_difference_ci_low": ci[0], "recovery_difference_ci_high": ci[1],
            "median_relative_output_error_difference": group.relative_error_difference.median(),
            "median_base_damage_ratio_left_over_right": np.median(group.base_damage_left / group.base_damage_right),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--w1", type=Path, required=True)
    parser.add_argument("--q2", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()
    w1 = add_global(pd.read_parquet(args.w1))
    datasets = [("W1_1.250488bpw", w1)]
    if args.q2 and args.q2.exists():
        datasets.append(("Q2_2.250488bpw", add_global(pd.read_parquet(args.q2))))
    fixed = pd.concat([fixed_statistics(data, name, args.bootstrap, args.seed + index) for index, (name, data) in enumerate(datasets)], ignore_index=True)
    inverse = pd.concat([inverse_rates(data, name) for name, data in datasets], ignore_index=True)
    fixed.to_csv(args.output / "hybrid_fixed_rate_statistics.csv", index=False)
    inverse.to_parquet(args.output / "complete_expert_inverse_rate_per_invocation.parquet", index=False)
    if len(datasets) == 2:
        paired_difference(datasets[1][1], datasets[0][1], datasets[1][0], datasets[0][0], args.bootstrap, args.seed + 10).to_csv(args.output / "q2_vs_w1_paired_statistics.csv", index=False)
    print({"fixed_rows": len(fixed), "inverse_rows": len(inverse), "bases": [name for name, _ in datasets]})


if __name__ == "__main__":
    main()
