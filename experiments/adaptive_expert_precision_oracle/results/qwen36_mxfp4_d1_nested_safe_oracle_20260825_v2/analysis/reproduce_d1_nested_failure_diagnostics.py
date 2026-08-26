#!/usr/bin/env python3
"""Reproduce descriptive failure diagnostics for nested D1 Experiment A.

These tables are post-hoc failure interpretation only. They never participate
in allocation, calibration, primary inference, or Experiment-B promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


ARM_D1 = "nested_d1_safe"
ARM_LOCAL = "nested_local_only"
ARM_PR13 = "independent_pr13"
ROUTE_MODES = ("live", "frozen_set_live_weights", "fully_frozen")

CHANGED = "d1_nested_changed_cell_failure_decomposition.parquet"
MONOTONICITY = "d1_nested_adjacent_rate_monotonicity.parquet"
METRICS = "d1_nested_primary_metric_failure_summary.parquet"
ATTENTION = "d1_nested_product_attention_decomposition.parquet"
FACTS = "d1_nested_failure_diagnostics_facts.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def summarize(values: pd.Series) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "favorable_fraction": float((values < 0).mean()),
        "p95_regression": float(values.quantile(0.95)),
        "maximum_regression": float(values.max()),
    }


def changed_cell_decomposition(
    quality: pd.DataFrame, comparisons: pd.DataFrame,
) -> pd.DataFrame:
    key = ["split", "rate_pages_per_expert", "injection_layer", "request_id"]
    selected = quality[quality["arm"].isin((ARM_D1, ARM_LOCAL))]
    pivot = selected.pivot(
        index=key, columns=["arm", "route_mode"], values="logit_kl",
    )
    rows = []
    for identity, values in pivot.iterrows():
        row = dict(zip(key, identity, strict=True))
        for mode in ROUTE_MODES:
            row[f"delta_{mode}"] = float(
                values[(ARM_D1, mode)] - values[(ARM_LOCAL, mode)]
            )
        row["delta_membership_execution"] = (
            row["delta_live"] - row["delta_frozen_set_live_weights"]
        )
        row["delta_router_weight"] = (
            row["delta_frozen_set_live_weights"] - row["delta_fully_frozen"]
        )
        rows.append(row)
    deltas = pd.DataFrame(rows)
    allocation = comparisons[
        comparisons["comparison_arm"].eq(ARM_D1)
        & comparisons["incumbent_arm"].eq(ARM_LOCAL)
    ]
    columns = key + [
        "physical_state_changed", "crossings_prevented", "crossings_introduced",
    ]
    deltas = deltas.merge(
        allocation[columns], on=key, how="inner", validate="one_to_one",
    )
    deltas = deltas[deltas["physical_state_changed"]]
    output = []
    for rate, frame in deltas.groupby("rate_pages_per_expert", sort=True):
        live = frame["delta_live"]
        output.append({
            "rate_pages_per_expert": int(rate),
            "changed_cells": int(len(frame)),
            "crossings_prevented": int(frame["crossings_prevented"].sum()),
            "crossings_introduced": int(frame["crossings_introduced"].sum()),
            "live_kl_mean": float(live.mean()),
            "live_kl_median": float(live.median()),
            "live_kl_favorable_fraction": float((live < 0).mean()),
            "live_kl_p95_regression": float(live.quantile(0.95)),
            "live_kl_maximum_regression": float(live.max()),
            "membership_execution_mean": float(
                frame["delta_membership_execution"].mean()
            ),
            "router_weight_mean": float(frame["delta_router_weight"].mean()),
            "fully_frozen_mean": float(frame["delta_fully_frozen"].mean()),
        })
    return pd.DataFrame(output)


def adjacent_rate_monotonicity(request_means: pd.DataFrame) -> pd.DataFrame:
    live = request_means[request_means["route_mode"].eq("live")]
    rows = []
    for arm in (ARM_PR13, ARM_LOCAL, ARM_D1):
        arm_rows = live[live["arm"].eq(arm)]
        for low, high in ((360, 384), (725, 749)):
            low_values = arm_rows[
                arm_rows["rate_pages_per_expert"].eq(low)
            ].set_index("request_id")["logit_kl"]
            high_values = arm_rows[
                arm_rows["rate_pages_per_expert"].eq(high)
            ].set_index("request_id")["logit_kl"]
            values = (high_values - low_values).dropna()
            rows.append({
                "arm": arm,
                "low_rate_pages_per_expert": low,
                "high_rate_pages_per_expert": high,
                "difference_direction": "high_minus_low",
                "requests": int(len(values)),
                **summarize(values),
            })
    return pd.DataFrame(rows)


def primary_metric_summary(request_means: pd.DataFrame) -> pd.DataFrame:
    live = request_means[request_means["route_mode"].eq("live")]
    comparisons = (
        ("nested_d1_safe_vs_nested_local_only_at_360", ARM_D1, 360, ARM_LOCAL, 360),
        ("nested_d1_safe_vs_nested_local_only_at_725", ARM_D1, 725, ARM_LOCAL, 725),
        (
            "nested_d1_safe_360_vs_independent_pr13_384_product_matched",
            ARM_D1, 360, ARM_PR13, 384,
        ),
        (
            "nested_d1_safe_725_vs_independent_pr13_749_product_matched",
            ARM_D1, 725, ARM_PR13, 749,
        ),
    )
    metrics = (
        "logit_kl", "delta_nll", "final_hidden_mse",
        "mean_downstream_route_membership_change_fraction",
        "mean_downstream_router_mass_churn", "mean_downstream_hidden_mse",
        "full_cache_mse",
    )
    rows = []
    for name, candidate_arm, candidate_rate, reference_arm, reference_rate in comparisons:
        candidate = live[
            live["arm"].eq(candidate_arm)
            & live["rate_pages_per_expert"].eq(candidate_rate)
        ].set_index("request_id")
        reference = live[
            live["arm"].eq(reference_arm)
            & live["rate_pages_per_expert"].eq(reference_rate)
        ].set_index("request_id")
        for metric in metrics:
            values = (candidate[metric] - reference[metric]).dropna()
            rows.append({
                "contrast": name,
                "candidate_arm": candidate_arm,
                "candidate_rate": candidate_rate,
                "reference_arm": reference_arm,
                "reference_rate": reference_rate,
                "metric": metric,
                "difference_direction": "candidate_minus_reference",
                "requests": int(len(values)),
                **summarize(values),
            })
    return pd.DataFrame(rows)


def product_attention_decomposition(strata: pd.DataFrame) -> pd.DataFrame:
    selected = strata[
        strata["stratum_type"].eq("attention")
        & strata["contrast"].str.contains("product_matched", regex=False)
    ].copy()
    return selected.sort_values(
        ["contrast", "stratum"], kind="stable",
    ).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--result-root", type=Path, default=default_root)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.output = args.output or args.result_root / "analysis"
    return args


def main() -> None:
    args = parse_args()
    analysis = args.result_root / "analysis"
    outcomes = args.result_root / "outcomes"
    sources = {
        "quality": outcomes / "d1_nested_outcome_quality.parquet",
        "comparisons": analysis / "d1_nested_allocation_comparisons.parquet",
        "request_means": analysis / "d1_nested_outcome_request_means.parquet",
        "strata": analysis / "d1_nested_primary_strata.parquet",
        "promotion": analysis / "d1_nested_experiment_a_promotion.json",
    }
    if not all(path.is_file() for path in sources.values()):
        missing = [str(path) for path in sources.values() if not path.is_file()]
        raise FileNotFoundError(f"missing diagnostic sources: {missing}")
    quality = pd.read_parquet(sources["quality"])
    comparisons = pd.read_parquet(sources["comparisons"])
    request_means = pd.read_parquet(sources["request_means"])
    strata = pd.read_parquet(sources["strata"])
    promotion = json.loads(sources["promotion"].read_text())
    if promotion.get("status") != "failed":
        raise RuntimeError("failure diagnostics require the sealed failed promotion")

    tables = {
        CHANGED: changed_cell_decomposition(quality, comparisons),
        MONOTONICITY: adjacent_rate_monotonicity(request_means),
        METRICS: primary_metric_summary(request_means),
        ATTENTION: product_attention_decomposition(strata),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        atomic_parquet(args.output / name, frame)
    outputs = {
        name: {
            "sha256": file_sha256(args.output / name),
            "bytes": (args.output / name).stat().st_size,
            "rows": len(frame),
        }
        for name, frame in tables.items()
    }
    atomic_json(args.output / FACTS, {
        "schema": "pr13_d1_nested_failure_diagnostics_v1",
        "completed": True,
        "post_hoc_descriptive_only": True,
        "used_for_allocation": False,
        "used_for_primary_inference": False,
        "used_for_promotion": False,
        "experiment_b_started": False,
        "sources": {
            name: {
                "path": path.relative_to(args.result_root).as_posix(),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in sources.items()
        },
        "outputs": outputs,
    })


if __name__ == "__main__":
    main()
