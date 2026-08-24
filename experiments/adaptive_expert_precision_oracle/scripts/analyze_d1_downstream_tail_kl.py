#!/usr/bin/env python3
"""Analyze exact-prefix cached single-token downstream-tail quality."""

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

from oracle_study.d1_decode_tail import (  # noqa: E402
    PR13_POLICY,
    expected_grid_counts,
    sha256,
    validate_cached_decode_tail_config,
)
from run_same_host_causal_controls import atomic_json, atomic_parquet  # noqa: E402
from run_d1_downstream_tail_kl import PROPAGATION, QUALITY, RUN_FACTS, SCHEMA, ZERO  # noqa: E402


REPORT = "D1_CACHED_DECODE_TAIL_KL_SMOKE_REPORT.md"
SUMMARY = "d1_cached_decode_tail_policy_summary.parquet"
PAIRED = "d1_cached_decode_tail_pr13_paired_delta.parquet"
ROUTES = "d1_cached_decode_tail_route_summary.parquet"
D1 = "d1_cached_decode_tail_d1_summary.parquet"
CONDITIONAL = "d1_cached_decode_tail_crossing_event_summary.parquet"
CANDIDATE_ORACLE = "d1_cached_decode_tail_exact_d1_candidate_oracle.parquet"
PROMOTION = "d1_cached_decode_tail_promotion_decision.json"
ANALYSIS_FACTS = "d1_cached_decode_tail_analysis_facts.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _load_verified(root: Path, config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Mapping[str, Any]]:
    facts = load_json(root / RUN_FACTS)
    if facts.get("schema") != SCHEMA or not bool(facts.get("completed")):
        raise RuntimeError("cached-decode tail run is not finalized")
    if str(facts["config_sha256"]) != sha256(Path(str(config["_config_path"]))):
        raise RuntimeError("cached-decode tail config hash changed")
    frames = {}
    for name in (QUALITY, PROPAGATION, ZERO):
        path = root / name
        recorded = facts["outputs"][name]
        if sha256(path) != str(recorded["sha256"]):
            raise RuntimeError(f"cached-decode tail table changed: {name}")
        frame = pd.read_parquet(path)
        if len(frame) != int(recorded["rows"]):
            raise RuntimeError(f"cached-decode tail row count changed: {name}")
        frames[name] = frame
    counts = expected_grid_counts(config)
    if len(frames[QUALITY]) != counts["quality_rows"]:
        raise RuntimeError("cached-decode quality grid is incomplete")
    if len(frames[PROPAGATION]) != counts["propagation_rows"]:
        raise RuntimeError("cached-decode propagation grid is incomplete")
    if len(frames[ZERO]) != counts["logical_zero_dose_rows"]:
        raise RuntimeError("cached-decode zero grid is incomplete")
    if not bool(frames[ZERO]["all_exact"].all()):
        raise RuntimeError("zero-dose parity failed")
    return frames[QUALITY], frames[PROPAGATION], frames[ZERO], facts


def _summaries(
    quality: pd.DataFrame,
    propagation: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calibration = set(map(int, config["fixed_d1_calibration"]["calibration_layers"]))
    quality = quality.copy()
    quality["layer_cohort"] = quality["injection_layer"].map(
        lambda layer: "calibration" if int(layer) in calibration else "held_out"
    )
    summary = (
        quality.groupby(
            ["rate_pages_per_expert", "injection_layer", "layer_cohort", "policy", "route_mode"],
            sort=True,
        )
        .agg(
            tokens=("group", "size"),
            mean_logit_kl=("logit_kl", "mean"),
            mean_delta_nll=("delta_nll", "mean"),
            mean_final_hidden_mse=("final_hidden_mse", "mean"),
            mean_local_qenergy=("live_selected_local_qenergy_damage", "mean"),
            mean_injected_layer_mse=("realized_injected_layer_output_mse", "mean"),
            downstream_crossing_rate=(
                "first_route_membership_change_layer",
                lambda values: float((values.to_numpy() >= 0).mean()),
            ),
            mean_live_minus_frozen_kl=("live_minus_frozen_logit_kl", "mean"),
        )
        .reset_index()
    )
    keys = [
        "rate_pages_per_expert", "injection_layer", "group", "request_id",
        "position", "route_mode",
    ]
    reference = quality[quality["policy"].eq(PR13_POLICY)][keys + ["logit_kl"]].rename(
        columns={"logit_kl": "pr13_logit_kl"}
    )
    paired = quality.merge(reference, on=keys, validate="many_to_one")
    paired["logit_kl_minus_pr13"] = paired["logit_kl"] - paired["pr13_logit_kl"]
    paired_summary = (
        paired.groupby(
            ["rate_pages_per_expert", "layer_cohort", "policy", "route_mode"],
            sort=True,
        )
        .agg(
            tokens=("group", "size"),
            mean_logit_kl_minus_pr13=("logit_kl_minus_pr13", "mean"),
            improved_token_fraction=("logit_kl_minus_pr13", lambda values: float((values < 0).mean())),
        )
        .reset_index()
    )
    routes = (
        propagation.groupby(
            ["rate_pages_per_expert", "injection_layer", "policy", "route_mode", "distance_from_injection"],
            sort=True,
        )
        .agg(
            tokens=("group", "size"),
            membership_change_rate=("route_membership_change_fraction", "mean"),
            mean_router_mass_churn=("router_mass_churn", "mean"),
            mean_hidden_mse=("hidden_mse", "mean"),
            mean_cache_mse=("post_token_layer_cache_mse", "mean"),
        )
        .reset_index()
    )
    return summary, paired_summary, routes


def _d1_diagnostics(
    quality: pd.DataFrame,
    propagation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Measure immediate crossings and freeze the Experiment B promotion gate."""

    keys = ["rate_pages_per_expert", "injection_layer", "group", "policy"]
    live = quality[quality["route_mode"].eq("live")].copy()
    immediate = propagation[
        propagation["route_mode"].eq("live")
        & propagation["distance_from_injection"].eq(1)
    ][keys + ["route_membership_change_fraction", "router_mass_churn"]]
    joined = live.merge(immediate, on=keys, validate="one_to_one")
    d1 = (
        joined.groupby(["rate_pages_per_expert", "policy"], sort=True)
        .agg(
            tokens=("group", "size"),
            mean_logit_kl=("logit_kl", "mean"),
            mean_live_minus_frozen_kl=("live_minus_frozen_logit_kl", "mean"),
            d1_crossing_rate=("route_membership_change_fraction", "mean"),
            mean_d1_router_mass_churn=("router_mass_churn", "mean"),
            mean_local_qenergy=("live_selected_local_qenergy_damage", "mean"),
        )
        .reset_index()
    )
    reference = d1[d1["policy"].eq(PR13_POLICY)][[
        "rate_pages_per_expert", "mean_logit_kl", "mean_live_minus_frozen_kl",
        "d1_crossing_rate", "mean_d1_router_mass_churn",
    ]].rename(columns={
        "mean_logit_kl": "pr13_mean_logit_kl",
        "mean_live_minus_frozen_kl": "pr13_mean_live_minus_frozen_kl",
        "d1_crossing_rate": "pr13_d1_crossing_rate",
        "mean_d1_router_mass_churn": "pr13_mean_d1_router_mass_churn",
    })
    d1 = d1.merge(reference, on="rate_pages_per_expert", validate="many_to_one")
    d1["mean_logit_kl_minus_pr13"] = d1["mean_logit_kl"] - d1["pr13_mean_logit_kl"]
    d1["live_minus_frozen_kl_minus_pr13"] = (
        d1["mean_live_minus_frozen_kl"] - d1["pr13_mean_live_minus_frozen_kl"]
    )
    d1["d1_crossing_rate_minus_pr13"] = (
        d1["d1_crossing_rate"] - d1["pr13_d1_crossing_rate"]
    )

    identity = ["rate_pages_per_expert", "injection_layer", "group"]
    pr13 = joined[joined["policy"].eq(PR13_POLICY)][
        identity + ["logit_kl", "route_membership_change_fraction"]
    ].rename(columns={
        "logit_kl": "pr13_logit_kl",
        "route_membership_change_fraction": "pr13_d1_crossed",
    })
    paired = joined[~joined["policy"].eq(PR13_POLICY)].merge(
        pr13, on=identity, validate="many_to_one",
    )
    paired["logit_kl_minus_pr13"] = paired["logit_kl"] - paired["pr13_logit_kl"]
    candidate_crossed = paired["route_membership_change_fraction"].to_numpy() > 0
    pr13_crossed = paired["pr13_d1_crossed"].to_numpy() > 0
    paired["crossing_event"] = np.where(
        pr13_crossed & ~candidate_crossed,
        "prevented",
        np.where(~pr13_crossed & candidate_crossed, "introduced", "same"),
    )
    conditional = (
        paired.groupby(["rate_pages_per_expert", "policy", "crossing_event"], sort=True)
        .agg(
            tokens=("group", "size"),
            mean_logit_kl_minus_pr13=("logit_kl_minus_pr13", "mean"),
            improved_token_fraction=(
                "logit_kl_minus_pr13", lambda values: float((values < 0).mean())
            ),
        )
        .reset_index()
    )

    oracle = (
        joined.sort_values(
            identity + [
                "route_membership_change_fraction", "router_mass_churn",
                "live_selected_local_qenergy_damage", "selected_group_pages", "policy",
            ],
            kind="stable",
        )
        .groupby(identity, sort=False)
        .head(1)
        .merge(pr13[identity + ["pr13_logit_kl"]], on=identity, validate="one_to_one")
    )
    oracle["logit_kl_minus_pr13"] = oracle["logit_kl"] - oracle["pr13_logit_kl"]
    candidate_oracle = (
        oracle.groupby("rate_pages_per_expert", sort=True)
        .agg(
            tokens=("group", "size"),
            mean_logit_kl=("logit_kl", "mean"),
            pr13_mean_logit_kl=("pr13_logit_kl", "mean"),
            mean_logit_kl_minus_pr13=("logit_kl_minus_pr13", "mean"),
            improved_token_fraction=(
                "logit_kl_minus_pr13", lambda values: float((values < 0).mean())
            ),
            d1_crossing_rate=("route_membership_change_fraction", "mean"),
            mean_d1_router_mass_churn=("router_mass_churn", "mean"),
        )
        .reset_index()
    )

    fixed = d1[d1["policy"].eq("calibration_selected_fixed_d1")].copy()
    fixed["passes_same_rate_gate"] = (
        (fixed["mean_logit_kl_minus_pr13"] < 0)
        & (fixed["live_minus_frozen_kl_minus_pr13"] < 0)
        & (fixed["d1_crossing_rate_minus_pr13"] < 0)
    )
    fixed_events = paired[
        paired["policy"].eq("calibration_selected_fixed_d1")
    ].groupby("crossing_event")["logit_kl_minus_pr13"].agg(["size", "mean"])
    decision = {
        "status": "not_promoted_pause_before_experiment_b",
        "experiment_b_started": False,
        "requests": int(live["request_id"].nunique()),
        "minimum_requests_before_quality_claim": 64,
        "same_rate_gate": fixed[[
            "rate_pages_per_expert", "mean_logit_kl_minus_pr13",
            "live_minus_frozen_kl_minus_pr13", "d1_crossing_rate_minus_pr13",
            "passes_same_rate_gate",
        ]].to_dict(orient="records"),
        "rates_passing_all_three_gate_terms": fixed.loc[
            fixed["passes_same_rate_gate"], "rate_pages_per_expert"
        ].astype(int).tolist(),
        "fixed_d1_crossing_event_mean_kl_delta": {
            str(event): {
                "tokens": int(row["size"]),
                "mean_logit_kl_minus_pr13": float(row["mean"]),
            }
            for event, row in fixed_events.iterrows()
        },
        "interpretation": (
            "Preventing a PR13 D1 crossing is directionally favorable for the fixed "
            "policy, but fixed D1, source-slice token oracle, and exact combined "
            "local are non-monotonic across the four rates. The same-rate KL, "
            "live-minus-frozen, and D1-crossing gate does not support formal "
            "Experiment B promotion."
        ),
        "required_before_promotion": [
            "prove a same-host full-model D1 rerank or repair lowers both live KL and live-minus-frozen KL at matched rates",
            "separate route-direction gain from exact-combined local-qenergy gain",
            "retain the three-request result as smoke evidence only",
        ],
    }
    return d1, conditional, candidate_oracle, decision


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(no rows)"
    columns = [str(column) for column in frame.columns]

    def render(value: Any) -> str:
        if isinstance(value, float):
            result = f"{value:.8g}"
        else:
            result = str(value)
        return result.replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join((header, rule, *rows))


def _report(
    config: Mapping[str, Any],
    quality: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    zero: pd.DataFrame,
    d1: pd.DataFrame,
    conditional: pd.DataFrame,
    candidate_oracle: pd.DataFrame,
    decision: Mapping[str, Any],
) -> str:
    live = summary[summary["route_mode"].eq("live")]
    aggregate = (
        live.groupby(["rate_pages_per_expert", "policy"], sort=True)
        .agg(
            mean_logit_kl=("mean_logit_kl", "mean"),
            mean_delta_nll=("mean_delta_nll", "mean"),
            crossing_rate=("downstream_crossing_rate", "mean"),
            mean_local_qenergy=("mean_local_qenergy", "mean"),
        )
        .reset_index()
    )
    heldout = paired[
        paired["layer_cohort"].eq("held_out") & paired["route_mode"].eq("live")
    ]
    d1_compact = d1[[
        "rate_pages_per_expert", "policy", "mean_logit_kl_minus_pr13",
        "live_minus_frozen_kl_minus_pr13", "d1_crossing_rate",
        "d1_crossing_rate_minus_pr13", "mean_local_qenergy",
    ]]
    fixed_events = conditional[
        conditional["policy"].eq("calibration_selected_fixed_d1")
    ]
    gate = pd.DataFrame(decision["same_rate_gate"])
    return f"""# Cached-decode D1 downstream-tail KL smoke report

## Scientific boundary

This is an exact-prefix, one-current-token, one-injected-layer smoke study.
Every candidate receives a private clone of the complete native prefix cache.
Only the current token's terminal logits are scored, and its post-token
attention K/V, DeltaNet convolution, and recurrent states are retained and
audited. No sequence-shaped perturbation or prompt-position coupling is used.

The three retained requests provide {len(quality)} logical terminal observations.
They diagnose mechanism and implementation only. At least
{int(config["minimum_requests_before_quality_claim"])} independent requests are
required before a terminal-quality claim.

## Live terminal result

{_markdown_table(aggregate)}

## Held-out paired KL relative to PR #13

Negative values improve on PR #13 at identical layer, rate, token, and routing
mode.

{_markdown_table(heldout)}

## Immediate D1 and amplification result

All deltas below are paired to PR #13 at the same rate, layer, and token.
A deployable D1 policy was required to lower live KL, live-minus-frozen KL,
and immediate crossing rate at the same rate.

{_markdown_table(d1_compact)}

## Conditional crossing evidence

For the frozen fixed-D1 policy, preventing a crossing is directionally
favorable while introducing one is harmful. This supports the amplifier
mechanism, but it does not rescue a policy that is inconsistent by rate.

{_markdown_table(fixed_events)}

## Exact full-model label diagnostic over executed candidates

This post-hoc diagnostic uses exact full-model D1 crossing, then router-mass
churn, local qenergy, pages, and policy name to choose among the five already
executed allocations. It is not a runtime policy and never selects on terminal
KL.

{_markdown_table(candidate_oracle)}

## Experiment B promotion decision

Status: **{decision["status"]}**

Only these page rates pass all three fixed-policy terms:
{decision["rates_passing_all_three_gate_terms"]}.

{_markdown_table(gate)}

Experiment B did not start. The directional crossing evidence is promising,
but fixed D1 and the source-slice token oracle are non-monotonic across rates,
and exact-combined local is at least as competitive. A same-host full-model D1
rerank or repair must separate route-direction benefit from combined-local
benefit before promotion.

## Validation gates

- Zero-dose logical rows: {len(zero)}
- Zero-dose rows exact in hidden states, routers, terminal logits, and full cache:
  {int(zero["all_exact"].sum())}/{len(zero)}
- Current query length: one token
- Candidate cache reuse: forbidden
- D1 labels used at runtime: no; the exact token oracle is a diagnostic policy
- Joint all-layer compression and generated rollout: out of scope
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_json(args.config)
    validate_cached_decode_tail_config(config)
    config["_config_path"] = str(args.config)
    quality, propagation, zero, run_facts = _load_verified(args.input, config)
    summary, paired, routes = _summaries(quality, propagation, config)
    d1, conditional, candidate_oracle, decision = _d1_diagnostics(
        quality, propagation,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_parquet(args.output / SUMMARY, summary)
    atomic_parquet(args.output / PAIRED, paired)
    atomic_parquet(args.output / ROUTES, routes)
    atomic_parquet(args.output / D1, d1)
    atomic_parquet(args.output / CONDITIONAL, conditional)
    atomic_parquet(args.output / CANDIDATE_ORACLE, candidate_oracle)
    atomic_json(args.output / PROMOTION, decision)
    report = args.output / REPORT
    report.write_text(_report(
        config, quality, summary, paired, zero, d1, conditional,
        candidate_oracle, decision,
    ))
    outputs = {}
    for path in (
        args.output / SUMMARY, args.output / PAIRED, args.output / ROUTES,
        args.output / D1, args.output / CONDITIONAL,
        args.output / CANDIDATE_ORACLE, args.output / PROMOTION, report,
    ):
        outputs[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    atomic_json(args.output / ANALYSIS_FACTS, {
        "completed": True,
        "schema": SCHEMA,
        "config_sha256": sha256(args.config),
        "run_facts_sha256": sha256(args.input / RUN_FACTS),
        "outputs": outputs,
        "test_rows_admitted_or_used": False,
    })
    print(report)


if __name__ == "__main__":
    main()
