#!/usr/bin/env python3
"""Analyze exact-prefix cached single-token downstream-tail quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

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


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(no rows)"
    return frame.to_markdown(index=False, floatfmt=".8g")


def _report(
    config: Mapping[str, Any],
    quality: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    zero: pd.DataFrame,
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
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_parquet(args.output / SUMMARY, summary)
    atomic_parquet(args.output / PAIRED, paired)
    atomic_parquet(args.output / ROUTES, routes)
    report = args.output / REPORT
    report.write_text(_report(config, quality, summary, paired, zero))
    outputs = {}
    for path in (args.output / SUMMARY, args.output / PAIRED, args.output / ROUTES, report):
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
