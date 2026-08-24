#!/usr/bin/env python3
"""Validate and aggregate exact-prefill cached D1 decode slice cells."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMA = "pr13_d1_exact_prefill_decode_slice_analysis_v2"
CALIBRATION_LAYERS = (0, 12, 23)
HELD_OUT_LAYERS = (1, 4, 6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--same-host-layers-dir", type=Path, required=True)
    parser.add_argument("--expected-groups", type=int, default=29)
    parser.add_argument("--expected-cells", type=int, default=12)
    parser.add_argument(
        "--expected-layers", type=int, nargs="+", default=[0, 1, 4, 6, 12, 23]
    )
    parser.add_argument("--expected-rates", type=int, nargs="+", default=[384, 749])
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    fact_paths = sorted(args.raw_root.glob(
        "results/rate_*_layer*/layer_*/d1_decode_layer_facts.json"
    ))
    if not fact_paths:
        raise RuntimeError("no finalized cached decode cells were found")
    facts = []
    exact_frames = []
    allocation_frames = []
    allocation_expert_frames = []
    parity_frames = []
    vjp_frames = []
    oracle_frames = []
    historical_delta_audit = []
    for fact_path in fact_paths:
        fact = json.loads(fact_path.read_text())
        if not fact.get("completed"):
            raise RuntimeError(f"incomplete cell: {fact_path}")
        if int(fact["groups"]) != int(args.expected_groups):
            raise RuntimeError(f"group count changed: {fact_path}")
        for name, record in fact["files"].items():
            target = fact_path.parent / name
            if sha256(target) != record["sha256"]:
                raise RuntimeError(f"cell hash mismatch: {target}")
        facts.append(fact)
        exact_cell = pd.read_parquet(
            fact_path.parent / "d1_decode_exact_route_metrics.parquet"
        )
        allocation_cell = pd.read_parquet(
            fact_path.parent / "d1_decode_allocation_groups.parquet"
        )
        allocation_expert_cell = pd.read_parquet(
            fact_path.parent / "d1_decode_allocation_experts.parquet"
        )
        parity_cell = pd.read_parquet(
            fact_path.parent / "d1_decode_cache_parity.parquet"
        )
        vjp_cell = pd.read_parquet(
            fact_path.parent / "d1_decode_vjp_metrics.parquet"
        )
        oracle_cell = pd.read_parquet(
            fact_path.parent / "d1_decode_token_oracle.parquet"
        )
        expected_policies = set(fact["policies"])
        if set(exact_cell["policy"].unique()) != expected_policies:
            raise RuntimeError(f"exact policy set changed: {fact_path}")
        if len(exact_cell) != int(fact["groups"]) * len(expected_policies):
            raise RuntimeError(f"exact row count changed: {fact_path}")
        allocation_policies = expected_policies - {
            "pr13_router_square_column_generated"
        }
        if set(allocation_cell["policy"].unique()) != allocation_policies:
            raise RuntimeError(f"allocation policy set changed: {fact_path}")
        if len(allocation_cell) != int(fact["groups"]) * len(allocation_policies):
            raise RuntimeError(f"allocation row count changed: {fact_path}")
        if len(allocation_expert_cell) != len(allocation_cell) * 8:
            raise RuntimeError(f"allocation expert row count changed: {fact_path}")
        for name, cell in (
            ("parity", parity_cell),
            ("vjp", vjp_cell),
            ("oracle", oracle_cell),
        ):
            if len(cell) != int(fact["groups"]):
                raise RuntimeError(f"{name} row count changed: {fact_path}")
        with np.load(fact_path.parent / "d1_decode_selected_deltas.npz") as deltas:
            rate = int(fact["rates"][0])
            historical = deltas[
                f"pr13_router_square_column_generated__rate_{rate}"
            ]
            regenerated = deltas[
                f"regenerated_router_square_column_generated__rate_{rate}"
            ]
            maximum_delta_abs_error = float(np.max(np.abs(historical - regenerated)))
            historical_delta_audit.append({
                "layer": int(fact["layer"]),
                "rate_pages_per_expert": rate,
                "selected_delta_bit_identical": np.array_equal(
                    historical, regenerated
                ),
                "maximum_selected_delta_abs_error": maximum_delta_abs_error,
            })
        exact_frames.append(exact_cell)
        allocation_frames.append(allocation_cell)
        allocation_expert_frames.append(allocation_expert_cell)
        parity_frames.append(parity_cell)
        vjp_frames.append(vjp_cell)
        oracle_frames.append(oracle_cell)

    exact = pd.concat(exact_frames, ignore_index=True)
    allocations = pd.concat(allocation_frames, ignore_index=True)
    allocation_experts = pd.concat(allocation_expert_frames, ignore_index=True)
    parity = pd.concat(parity_frames, ignore_index=True)
    vjp = pd.concat(vjp_frames, ignore_index=True)
    oracle = pd.concat(oracle_frames, ignore_index=True)
    if len(facts) != int(args.expected_cells):
        raise RuntimeError(f"expected {args.expected_cells} cells, found {len(facts)}")
    observed_layers = sorted({int(fact["layer"]) for fact in facts})
    observed_rates = sorted({int(rate) for fact in facts for rate in fact["rates"]})
    if observed_layers != sorted(args.expected_layers):
        raise RuntimeError(f"layer coverage changed: {observed_layers}")
    if observed_rates != sorted(args.expected_rates):
        raise RuntimeError(f"rate coverage changed: {observed_rates}")
    delta_audit = pd.DataFrame(historical_delta_audit)
    if float(delta_audit["maximum_selected_delta_abs_error"].max()) > 1e-6:
        raise RuntimeError("a regenerated PR #13 selected delta materially changed")
    if not bool(parity["paired_repeat_bit_identical"].all()):
        raise RuntimeError("a repeated cached baseline changed")
    if not bool(parity["paired_cache_bit_identical"].all()):
        raise RuntimeError("a repeated cached baseline cache changed")
    if not bool(vjp["native_cached_forward"].all()):
        raise RuntimeError("a VJP used a non-native cached forward")
    if not bool(vjp["autograd_saved_tensors_cloned"].all()):
        raise RuntimeError("a mutable-cache VJP omitted saved-tensor cloning")
    if exact.duplicated(
        ["layer", "rate_pages_per_expert", "group", "policy"]
    ).any():
        raise RuntimeError("duplicate exact policy rows")
    if allocations.duplicated(
        ["layer", "rate_pages_per_expert", "group", "policy"]
    ).any():
        raise RuntimeError("duplicate allocation rows")
    if allocation_experts.duplicated([
        "layer", "rate_pages_per_expert", "group", "policy", "router_rank"
    ]).any():
        raise RuntimeError("duplicate allocation expert rows")
    if bool((allocations["selected_group_pages"] > allocations["group_page_budget"]).any()):
        raise RuntimeError("an allocation exceeded its page budget")
    d1_guard_rows = allocations[
        allocations["policy"].str.startswith("d1_strict_")
    ].copy()
    guard_excess = (
        d1_guard_rows["local_qenergy_damage"]
        - d1_guard_rows["local_optimal_damage"]
        - d1_guard_rows["local_guard_eta"] * d1_guard_rows["local_q2_damage"]
    )
    max_guard_excess = float(guard_excess.max())
    if max_guard_excess > 1e-12:
        raise RuntimeError(f"a D1 allocation violated its local guard: {max_guard_excess}")

    historical_columns = [
        "layer", "request_id", "position", "mean_budget_pages_per_expert",
        "router_rank", "expert_id", "router_weight", "selected_pages",
        "selected_states", "factor_config_id", "allocation_policy",
        "burst_cap_pages_per_expert", "execution_router_weight",
    ]
    historical_allocation_paths = [
        args.same_host_layers_dir / f"same_host_allocation_layer_{layer:02d}.parquet"
        for layer in observed_layers
    ]
    historical_allocation = pd.concat([
        pd.read_parquet(path, columns=historical_columns)
        for path in historical_allocation_paths
    ], ignore_index=True)
    historical_allocation = historical_allocation[
        historical_allocation["layer"].isin(observed_layers)
        & historical_allocation["mean_budget_pages_per_expert"].isin(observed_rates)
        & historical_allocation["factor_config_id"].eq(
            "matrix_free_joint_rank8_int4_per_row_hadamard"
        )
        & historical_allocation["allocation_policy"].eq(
            "pooled_router_square_column_generated"
        )
        & historical_allocation["burst_cap_pages_per_expert"].eq(1536)
    ].copy()
    regenerated_allocation = allocation_experts[
        allocation_experts["policy"].eq(
            "regenerated_router_square_column_generated"
        )
    ].rename(columns={
        "rate_pages_per_expert": "mean_budget_pages_per_expert",
        "selector_router_weight": "router_weight_candidate",
        "execution_router_weight": "execution_router_weight_candidate",
    }).copy()
    identity_keys = [
        "layer", "request_id", "position", "mean_budget_pages_per_expert",
        "router_rank",
    ]
    admitted_identity = regenerated_allocation[identity_keys].drop_duplicates()
    historical_allocation = historical_allocation.merge(
        admitted_identity,
        on=identity_keys,
        how="inner",
        validate="one_to_one",
    )
    historical_allocation = historical_allocation.sort_values(
        identity_keys, kind="stable"
    ).reset_index(drop=True)
    regenerated_allocation = regenerated_allocation.sort_values(
        identity_keys, kind="stable"
    ).reset_index(drop=True)
    if len(historical_allocation) != len(regenerated_allocation):
        raise RuntimeError("historical PR #13 allocation row count changed")
    discrete_identity_fields = identity_keys + [
        "expert_id", "selected_pages", "selected_states"
    ]
    for field in discrete_identity_fields:
        if not np.array_equal(
            historical_allocation[field].to_numpy(),
            regenerated_allocation[field].to_numpy(),
        ):
            raise RuntimeError(f"historical PR #13 allocation changed: {field}")
    router_weight_error = np.abs(
        historical_allocation["router_weight"].to_numpy(np.float64)
        - regenerated_allocation["router_weight_candidate"].to_numpy(np.float64)
    )
    maximum_router_weight_abs_error = float(router_weight_error.max(initial=0.0))
    execution_weight_error = np.abs(
        historical_allocation["execution_router_weight"].to_numpy(np.float64)
        - regenerated_allocation["execution_router_weight_candidate"].to_numpy(
            np.float64
        )
    )
    maximum_execution_weight_abs_error = float(
        execution_weight_error.max(initial=0.0)
    )
    if maximum_router_weight_abs_error != 0.0:
        raise RuntimeError("same-host PR #13 selector weights changed")
    if maximum_execution_weight_abs_error != 0.0:
        raise RuntimeError("same-host PR #13 execution weights changed")
    delta_audit["same_host_expert_page_state_allocation_bit_identical"] = True

    cell_summary = (
        exact.groupby(["layer", "next_mixer", "rate_pages_per_expert", "policy"], sort=True)
        .agg(
            groups=("group", "size"),
            exact_d1_crossings=("exact_d1_crossed", "sum"),
            exact_d1_crossing_rate=("exact_d1_crossed", "mean"),
            membership_pairs_changed=("membership_pairs_changed", "sum"),
            mean_routing_mass_lost=("routing_mass_lost", "mean"),
            mean_local_qenergy_damage=("local_qenergy_damage", "mean"),
            mean_current_hidden_mse=("current_hidden_mse", "mean"),
            mean_next_mixer_cache_mse=("next_mixer_cache_mse", "mean"),
        )
        .reset_index()
    )
    aggregate = (
        exact.groupby(["rate_pages_per_expert", "policy"], sort=True)
        .agg(
            layers=("layer", "nunique"),
            groups=("group", "size"),
            exact_d1_crossings=("exact_d1_crossed", "sum"),
            exact_d1_crossing_rate=("exact_d1_crossed", "mean"),
            membership_pairs_changed=("membership_pairs_changed", "sum"),
            mean_routing_mass_lost=("routing_mass_lost", "mean"),
            mean_local_qenergy_damage=("local_qenergy_damage", "mean"),
            mean_current_hidden_mse=("current_hidden_mse", "mean"),
            mean_next_mixer_cache_mse=("next_mixer_cache_mse", "mean"),
        )
        .reset_index()
    )
    oracle_summary = (
        oracle.groupby(["layer", "next_mixer", "rate_pages_per_expert"], sort=True)
        .agg(
            groups=("group", "size"),
            exact_d1_crossings=("exact_d1_crossed", "sum"),
            exact_d1_crossing_rate=("exact_d1_crossed", "mean"),
            mean_local_qenergy_damage=("local_qenergy_damage", "mean"),
        )
        .reset_index()
    )
    parity_summary = (
        parity.groupby(["layer", "next_mixer"], sort=True)
        .agg(
            groups=("group", "size"),
            repeat_bit_identical=("paired_repeat_bit_identical", "all"),
            cache_repeat_bit_identical=("paired_cache_bit_identical", "all"),
            cached_vs_full_top8_set_fraction=("cached_vs_full_top8_set_equal", "mean"),
            cached_vs_full_router_max_abs=("cached_vs_full_router_max_abs", "max"),
        )
        .reset_index()
    )
    vjp_summary = (
        vjp.groupby(["layer", "next_mixer"], sort=True)
        .agg(
            groups=("group", "size"),
            median_vjp_seconds=("exact_vjp_seconds", "median"),
            p90_vjp_seconds=("exact_vjp_seconds", lambda values: values.quantile(0.9)),
            max_vjp_seconds=("exact_vjp_seconds", "max"),
        )
        .reset_index()
    )
    d1_exact = exact[exact["policy"].str.startswith("d1_strict_")].copy()
    d1_allocations = allocations[
        allocations["policy"].str.startswith("d1_strict_")
    ].copy()
    calibration = (
        d1_exact[d1_exact["layer"].isin(CALIBRATION_LAYERS)]
        .groupby(["rate_pages_per_expert", "policy"], sort=True)
        .agg(
            calibration_layers=("layer", "nunique"),
            calibration_groups=("group", "size"),
            calibration_crossings=("exact_d1_crossed", "sum"),
            calibration_membership_pairs_changed=("membership_pairs_changed", "sum"),
            calibration_routing_mass_lost=("routing_mass_lost", "sum"),
            calibration_mean_local_qenergy_damage=("local_qenergy_damage", "mean"),
        )
        .reset_index()
    )
    calibration_pages = (
        d1_allocations[d1_allocations["layer"].isin(CALIBRATION_LAYERS)]
        .groupby(["rate_pages_per_expert", "policy"], sort=True)
        .agg(calibration_mean_selected_group_pages=("selected_group_pages", "mean"))
        .reset_index()
    )
    calibration = calibration.merge(
        calibration_pages,
        on=["rate_pages_per_expert", "policy"],
        validate="one_to_one",
    )
    calibration["selected"] = False
    selected_rows = []
    for _, frame in calibration.groupby("rate_pages_per_expert", sort=True):
        choice = frame.sort_values([
            "calibration_crossings",
            "calibration_membership_pairs_changed",
            "calibration_routing_mass_lost",
            "calibration_mean_local_qenergy_damage",
            "calibration_mean_selected_group_pages",
            "policy",
        ], kind="stable").index[0]
        calibration.loc[choice, "selected"] = True
        selected_rows.append(calibration.loc[choice])
    selected = pd.DataFrame(selected_rows)[["rate_pages_per_expert", "policy"]]
    held_out = d1_exact[d1_exact["layer"].isin(HELD_OUT_LAYERS)].merge(
        selected,
        on=["rate_pages_per_expert", "policy"],
        validate="many_to_one",
    )
    held_out_summary = (
        held_out.groupby(["rate_pages_per_expert", "policy"], sort=True)
        .agg(
            held_out_layers=("layer", "nunique"),
            held_out_groups=("group", "size"),
            held_out_crossings=("exact_d1_crossed", "sum"),
            held_out_crossing_rate=("exact_d1_crossed", "mean"),
            held_out_membership_pairs_changed=("membership_pairs_changed", "sum"),
            held_out_routing_mass_lost=("routing_mass_lost", "sum"),
            held_out_mean_local_qenergy_damage=("local_qenergy_damage", "mean"),
        )
        .reset_index()
    )
    if not bool((calibration["calibration_layers"] == len(CALIBRATION_LAYERS)).all()):
        raise RuntimeError("fixed-policy calibration layer coverage changed")
    if not bool((held_out_summary["held_out_layers"] == len(HELD_OUT_LAYERS)).all()):
        raise RuntimeError("fixed-policy held-out layer coverage changed")

    outputs = {
        "d1_decode_exact_route_metrics.parquet": exact,
        "d1_decode_allocation_groups.parquet": allocations,
        "d1_decode_allocation_experts.parquet": allocation_experts,
        "d1_decode_cell_policy_summary.parquet": cell_summary,
        "d1_decode_aggregate_policy_summary.parquet": aggregate,
        "d1_decode_token_oracle.parquet": oracle,
        "d1_decode_token_oracle_summary.parquet": oracle_summary,
        "d1_decode_cache_parity.parquet": parity,
        "d1_decode_cache_parity_summary.parquet": parity_summary,
        "d1_decode_vjp_metrics.parquet": vjp,
        "d1_decode_vjp_summary.parquet": vjp_summary,
        "d1_decode_fixed_eta_calibration.parquet": calibration,
        "d1_decode_fixed_eta_heldout.parquet": held_out_summary,
        "d1_decode_pr13_identity_audit.parquet": delta_audit,
    }
    for name, frame in outputs.items():
        atomic_parquet(args.output_dir / name, frame)
    analysis_facts = {
        "completed": True,
        "schema": SCHEMA,
        "cells": len(facts),
        "layers": observed_layers,
        "rates": observed_rates,
        "groups_per_cell": int(args.expected_groups),
        "exact_rows": len(exact),
        "allocation_rows": len(allocations),
        "allocation_expert_rows": len(allocation_experts),
        "parity_rows": len(parity),
        "vjp_rows": len(vjp),
        "oracle_rows": len(oracle),
        "all_repeated_cached_baselines_bit_identical": True,
        "all_repeated_cached_states_bit_identical": True,
        "all_vjps_native_and_saved_tensor_cloned": True,
        "all_regenerated_same_host_pr13_allocations_bit_identical": True,
        "same_host_pr13_allocation_rows_compared": len(historical_allocation),
        "maximum_same_host_pr13_selector_weight_abs_error": maximum_router_weight_abs_error,
        "maximum_same_host_pr13_execution_weight_abs_error": maximum_execution_weight_abs_error,
        "regenerated_pr13_selected_delta_bit_identical_cells": int(
            delta_audit["selected_delta_bit_identical"].sum()
        ),
        "regenerated_pr13_selected_delta_cells": len(delta_audit),
        "maximum_regenerated_pr13_selected_delta_abs_error": float(
            delta_audit["maximum_selected_delta_abs_error"].max()
        ),
        "all_regenerated_pr13_selected_deltas_within_1e-6": True,
        "all_allocations_within_page_budget": True,
        "all_d1_local_guards_satisfied": True,
        "maximum_d1_local_guard_excess": max_guard_excess,
        "fixed_eta_calibration_layers": list(CALIBRATION_LAYERS),
        "fixed_eta_held_out_layers": list(HELD_OUT_LAYERS),
        "test_rows_admitted_or_used": False,
        "same_host_allocation_inputs": [
            {"path": str(path), "sha256": sha256(path)}
            for path in historical_allocation_paths
        ],
        "source_cell_facts": [
            {"path": str(path.relative_to(args.raw_root)), "sha256": sha256(path)}
            for path in fact_paths
        ],
        "files": {},
    }
    for name in sorted(outputs):
        path = args.output_dir / name
        analysis_facts["files"][name] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "rows": len(outputs[name]),
        }
    atomic_json(args.output_dir / "d1_decode_analysis_facts.json", analysis_facts)
    print(aggregate.to_string(index=False))
    print(parity_summary.to_string(index=False))
    print(vjp_summary.to_string(index=False))
    print(calibration[calibration["selected"]].to_string(index=False))
    print(held_out_summary.to_string(index=False))


if __name__ == "__main__":
    main()
