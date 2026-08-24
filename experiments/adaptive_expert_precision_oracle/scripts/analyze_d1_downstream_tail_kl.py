#!/usr/bin/env python3
"""Analyze the exact three-request D1 downstream-tail KL smoke study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "src"))

from oracle_study.d1_tail_quality import (  # noqa: E402
    FIXED_D1_POLICY,
    LOCAL_POLICY,
    PR13_POLICY,
    REPAIRED_D1_POLICY,
    RERANKED_D1_POLICY,
    sha256,
    validate_tail_config,
)


SCHEMA = "pr13_d1_downstream_tail_kl_analysis_v1"
QUALITY_SUMMARY = "d1_tail_quality_summary.parquet"
LIVE_COMPARISON = "d1_tail_live_comparison.parquet"
LAYER_SUMMARY = "d1_tail_layer_summary.parquet"
DECISIONS = "d1_tail_decisive_interpretation.json"
REPORT = "D1_DOWNSTREAM_TAIL_KL_SMOKE_REPORT.md"
FACTS = "d1_tail_analysis_facts.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _markdown(frame: pd.DataFrame) -> str:
    def render(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.7g}"
        return str(value)
    header = [str(column) for column in frame.columns]
    rows = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    rows.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(rows)


def _layer_stratum(layer: int, config: Mapping[str, Any]) -> str:
    calibration = set(map(int, config["fixed_d1_calibration"]["calibration_layers"]))
    return "calibration_layers" if int(layer) in calibration else "held_out_layers"


def summarize_quality(quality: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    frame = quality.copy()
    frame["layer_stratum"] = [
        _layer_stratum(layer, config) for layer in frame["injection_layer"]
    ]
    result = (
        frame.groupby(
            ["rate_pages_per_expert", "point_type", "route_mode", "policy"],
            sort=True,
        )
        .agg(
            layers=("injection_layer", "nunique"),
            requests=("request_id", "nunique"),
            rows=("request_id", "size"),
            mean_logit_kl=("logit_kl", "mean"),
            median_logit_kl=("logit_kl", "median"),
            mean_delta_nll=("delta_nll", "mean"),
            mean_final_hidden_mse=("final_hidden_mse", "mean"),
            mean_selected_local_qenergy_damage=(
                "selected_local_qenergy_damage_per_group", "mean",
            ),
            mean_injected_delta_mse=("injected_delta_mse", "mean"),
            mean_realized_injected_layer_output_mse=(
                "realized_injected_layer_output_mse", "mean",
            ),
            mean_d1_membership_churn=("d1_route_membership_change_fraction", "mean"),
            mean_downstream_membership_churn=(
                "mean_downstream_route_membership_change_fraction", "mean",
            ),
            mean_downstream_router_mass_churn=("mean_downstream_router_mass_churn", "mean"),
            mean_live_minus_frozen_kl=("live_minus_fully_frozen_logit_kl", "mean"),
        )
        .reset_index()
    )
    result.insert(0, "schema", SCHEMA)
    return result


def layer_summary(quality: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    live = quality[quality["route_mode"].eq("live")].copy()
    live["layer_stratum"] = [
        _layer_stratum(layer, config) for layer in live["injection_layer"]
    ]
    result = (
        live.groupby(
            ["rate_pages_per_expert", "point_type", "layer_stratum", "injection_layer", "policy"],
            sort=True,
        )
        .agg(
            requests=("request_id", "nunique"),
            mean_logit_kl=("logit_kl", "mean"),
            mean_delta_nll=("delta_nll", "mean"),
            mean_final_hidden_mse=("final_hidden_mse", "mean"),
            mean_selected_local_qenergy_damage=(
                "selected_local_qenergy_damage_per_group", "mean",
            ),
            mean_injected_delta_mse=("injected_delta_mse", "mean"),
            mean_realized_injected_layer_output_mse=(
                "realized_injected_layer_output_mse", "mean",
            ),
            mean_d1_membership_churn=("d1_route_membership_change_fraction", "mean"),
            mean_downstream_membership_churn=(
                "mean_downstream_route_membership_change_fraction", "mean",
            ),
            mean_live_minus_frozen_kl=("live_minus_fully_frozen_logit_kl", "mean"),
        )
        .reset_index()
    )
    result.insert(0, "schema", SCHEMA)
    return result


def live_comparison(quality: pd.DataFrame) -> pd.DataFrame:
    live = quality[quality["route_mode"].eq("live")]
    summary = (
        live.groupby(["rate_pages_per_expert", "point_type", "policy"], sort=True)
        .agg(
            mean_logit_kl=("logit_kl", "mean"),
            mean_delta_nll=("delta_nll", "mean"),
            mean_final_hidden_mse=("final_hidden_mse", "mean"),
            mean_selected_local_qenergy_damage=(
                "selected_local_qenergy_damage_per_group", "mean",
            ),
            mean_injected_delta_mse=("injected_delta_mse", "mean"),
            mean_realized_injected_layer_output_mse=(
                "realized_injected_layer_output_mse", "mean",
            ),
            mean_d1_membership_churn=("d1_route_membership_change_fraction", "mean"),
            mean_downstream_membership_churn=(
                "mean_downstream_route_membership_change_fraction", "mean",
            ),
            mean_live_minus_frozen_kl=("live_minus_fully_frozen_logit_kl", "mean"),
        )
        .reset_index()
    )
    baseline = summary[summary["policy"].eq(PR13_POLICY)][[
        "rate_pages_per_expert", "mean_logit_kl", "mean_final_hidden_mse",
    ]].rename(columns={
        "mean_logit_kl": "pr13_mean_logit_kl",
        "mean_final_hidden_mse": "pr13_mean_final_hidden_mse",
    })
    result = summary.merge(baseline, on="rate_pages_per_expert", validate="many_to_one")
    result["logit_kl_ratio_to_pr13"] = np.divide(
        result["mean_logit_kl"],
        result["pr13_mean_logit_kl"],
        out=np.full(len(result), np.nan, np.float64),
        where=result["pr13_mean_logit_kl"].to_numpy(np.float64) != 0.0,
    )
    result["final_hidden_mse_ratio_to_pr13"] = np.divide(
        result["mean_final_hidden_mse"],
        result["pr13_mean_final_hidden_mse"],
        out=np.full(len(result), np.nan, np.float64),
        where=result["pr13_mean_final_hidden_mse"].to_numpy(np.float64) != 0.0,
    )
    result.insert(0, "schema", SCHEMA)
    return result


def decisive_interpretation(comparison: pd.DataFrame) -> dict[str, Any]:
    decisions = {}
    for rate, frame in comparison.groupby("rate_pages_per_expert", sort=True):
        values = frame.set_index("policy")["mean_logit_kl"].to_dict()
        required = {
            PR13_POLICY, LOCAL_POLICY, FIXED_D1_POLICY,
            RERANKED_D1_POLICY, REPAIRED_D1_POLICY,
        }
        if set(values) != required:
            raise RuntimeError(f"quality comparison policy grid changed at rate {rate}")
        pr13 = float(values[PR13_POLICY])
        local = float(values[LOCAL_POLICY])
        fixed = float(values[FIXED_D1_POLICY])
        repair = float(values[REPAIRED_D1_POLICY])
        repair_lowers = repair < pr13
        fixed_lowers = fixed < pr13
        local_as_good = local <= min(fixed, repair) * 1.05
        if repair_lowers and fixed_lowers:
            outcome = "repair_and_fixed_lower_kl_proceed_toward_all_layer_after_request_expansion"
        elif repair_lowers:
            outcome = "oracle_mechanism_positive_fixed_policy_selection_is_the_bottleneck"
        elif not repair_lowers and not fixed_lowers:
            outcome = "immediate_membership_not_supported_as_quality_surrogate_test_severity_objectives"
        else:
            outcome = "fixed_improves_without_repair_review_policy_and_repair_objectives"
        if local_as_good:
            outcome += "__combined_local_may_explain_d1_gain"
        decisions[str(int(rate))] = {
            "pr13_mean_live_kl": pr13,
            "local_mean_live_kl": local,
            "fixed_d1_mean_live_kl": fixed,
            "repaired_d1_mean_live_kl": repair,
            "repair_lowers_kl": repair_lowers,
            "fixed_d1_lowers_kl": fixed_lowers,
            "exact_local_within_five_percent_of_best_fixed_or_repair": local_as_good,
            "outcome": outcome,
        }
    return {
        "schema": SCHEMA,
        "smoke_only": True,
        "comparison_rule": "unweighted_mean_over_three_requests_and_six_injection_layers",
        "local_equivalence_screen_relative_tolerance": 0.05,
        "rates": decisions,
    }


def report_text(
    comparison: pd.DataFrame,
    layers: pd.DataFrame,
    decisions: Mapping[str, Any],
    config: Mapping[str, Any],
) -> str:
    table = comparison[[
        "rate_pages_per_expert", "point_type", "policy", "mean_logit_kl",
        "logit_kl_ratio_to_pr13", "mean_delta_nll", "mean_final_hidden_mse",
        "mean_selected_local_qenergy_damage", "mean_injected_delta_mse",
        "mean_d1_membership_churn", "mean_downstream_membership_churn",
        "mean_live_minus_frozen_kl",
    ]]
    held_out = layers[layers["layer_stratum"].eq("held_out_layers")][[
        "rate_pages_per_expert", "injection_layer", "policy", "mean_logit_kl",
        "mean_selected_local_qenergy_damage", "mean_d1_membership_churn",
        "mean_live_minus_frozen_kl",
    ]]
    outcomes = "\n".join(
        f"- {rate} pages/expert: `{value['outcome']}`."
        for rate, value in decisions["rates"].items()
    )
    return f"""# D1 single-injection downstream-tail KL smoke report

## Scientific boundary

This is an exact same-host, single-injected-layer downstream-tail smoke test.
It covers six layers and three validation requests. It is not an all-layer
streamed-model run and cannot support a quality claim before the frozen gate of
at least {int(config['minimum_requests_before_quality_claim'])} independent
requests. Exact request reranking and repair are oracle upper bounds.

The fixed D1 eta is selected only on layers 0, 12 and 23. Layers 1, 4 and 6 are
reported as the held-out architecture slice. Live routing is paired with fully
frozen expert IDs and weights; `live-minus-frozen KL` isolates the identified
discrete-routing amplification component.

Allocation reconstruction, exact D1 VJPs, reranking, and repair were performed
on the frozen RTX 3090 path. Terminal tails are executed on an RTX PRO 6000 and
measure D1 again on that host. This deliberately exposes, rather than hides,
any cross-device policy-transfer loss; it remains a limitation of this smoke
study and is not equivalent to recalibrating the D1 oracle on the PRO 6000.

## Aggregate live-tail comparison

{_markdown(table)}

## Held-out layer slice

{_markdown(held_out)}

## Decisive smoke interpretation

{outcomes}

These outcomes are descriptive gates on this exact smoke cohort, not confidence
intervals or generalization claims. The 360 and 725 page points charge the D1
joint-response plus uncertainty metadata against the original 384 and 749
all-in traffic, respectively. Their router-square comparator is regenerated by
the same PR #13 algorithm because no preserved historical PR #13 allocation
exists at those non-operating-point page caps.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    validate_tail_config(config)
    run_facts = load_json(args.input / "d1_downstream_tail_run_facts.json")
    quality_path = args.input / "d1_downstream_tail_quality.parquet"
    propagation_path = args.input / "d1_downstream_tail_propagation.parquet"
    if run_facts.get("completed") is not True:
        raise RuntimeError("D1 tail run is incomplete")
    if run_facts["quality_sha256"] != sha256(quality_path):
        raise RuntimeError("D1 tail quality hash changed")
    if run_facts["propagation_sha256"] != sha256(propagation_path):
        raise RuntimeError("D1 tail propagation hash changed")
    quality = pd.read_parquet(quality_path)
    expected_rows = (
        len(config["injection_layers"])
        * (len(config["mechanism_page_caps"]) + len(config["matched_runtime_metadata_page_caps"]))
        * len(config["validation_request_ids"])
        * len(config["policies"])
        * len(config["route_modes"])
    )
    if len(quality) != expected_rows:
        raise RuntimeError(f"D1 tail quality row count changed: {len(quality)} != {expected_rows}")
    expected_propagation_rows = (
        sum(39 - int(layer) for layer in config["injection_layers"])
        * (len(config["mechanism_page_caps"]) + len(config["matched_runtime_metadata_page_caps"]))
        * len(config["validation_request_ids"])
        * len(config["policies"])
        * len(config["route_modes"])
    )
    if len(pd.read_parquet(propagation_path)) != expected_propagation_rows:
        raise RuntimeError("D1 tail propagation row count changed")
    summary = summarize_quality(quality, config)
    layers = layer_summary(quality, config)
    comparison = live_comparison(quality)
    decisions = decisive_interpretation(comparison)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_parquet(args.output / QUALITY_SUMMARY, summary)
    atomic_parquet(args.output / LIVE_COMPARISON, comparison)
    atomic_parquet(args.output / LAYER_SUMMARY, layers)
    atomic_json(args.output / DECISIONS, decisions)
    atomic_text(args.output / REPORT, report_text(comparison, layers, decisions, config))
    outputs = [
        args.output / QUALITY_SUMMARY,
        args.output / LIVE_COMPARISON,
        args.output / LAYER_SUMMARY,
        args.output / DECISIONS,
        args.output / REPORT,
    ]
    atomic_json(args.output / FACTS, {
        "completed": True,
        "schema": SCHEMA,
        "config_sha256": sha256(args.config),
        "run_facts_sha256": sha256(args.input / "d1_downstream_tail_run_facts.json"),
        "input_quality_sha256": sha256(quality_path),
        "input_propagation_sha256": sha256(propagation_path),
        "quality_rows": len(quality),
        "smoke_only": True,
        "minimum_requests_before_quality_claim": int(config["minimum_requests_before_quality_claim"]),
        "outputs": {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in outputs
        },
        "test_rows_admitted_or_used": False,
    })


if __name__ == "__main__":
    main()
