#!/usr/bin/env python3
"""Analyze immutable eight-state split interaction-field evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "qwen36_mxfp4_split_interaction_field_v1"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "src"))

from oracle_study.split_interaction_field_analysis import (  # noqa: E402
    IDENTITY, layer_table, promotion_payload, sha256, summary_table,
    validate_evidence_bundle,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if pd.isna(value):
        return None
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _paired_table(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    keep = frame[frame["solver"].isin({
        config["primary_solver_id"], config["four_state_reference_solver_id"],
        config["pr11_four_state_solver_id"], config["exact_eight_state_solver_id"],
    })].copy()
    columns = list(IDENTITY) + [
        "budget_regime", "factor_config_id", "solver", "physical_budget_pages",
        "physical_budget_bpw", "budget_total_bpw", "recovery",
        "four_state_reference_recovery", "pr11_four_state_recovery",
        "recovery_gain_vs_four_state_reference", "recovery_gain_vs_pr11_four_state",
        "seed_optimizer_gain_vs_pr11_four_state",
        "set_gain_retention_vs_pr10_exact_hybrid", "split_only_units",
        "gate_high_units", "up_high_units", "down_high_units",
        "selector_compute_macs", "combined_metadata_bpw", "selector_runtime_ms",
        "coordinate_sweeps", "local_evaluated_passes",
    ]
    return keep[columns].sort_values(
        ["budget_regime", "solver", "factor_config_id", "physical_budget_pages", *IDENTITY],
    ).reset_index(drop=True)


def _state_usage(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["split_only_units", "gate_high_units", "up_high_units", "down_high_units", "physical_pages"]
    rows = []
    for key, values in frame.groupby(
        ["budget_regime", "factor_config_id", "solver", "physical_budget_pages"],
        sort=True,
    ):
        row = {
            "budget_regime": key[0], "factor_config_id": key[1],
            "solver": key[2], "physical_budget_pages": key[3], "rows": len(values),
        }
        for metric in metrics:
            row[f"{metric}_p10"] = float(values[metric].quantile(.1))
            row[f"{metric}_median"] = float(values[metric].quantile(.5))
            row[f"{metric}_p90"] = float(values[metric].quantile(.9))
        rows.append(row)
    return pd.DataFrame(rows)


def _plot(summary: pd.DataFrame, config: dict[str, Any], output: Path) -> None:
    selected = summary[
        (summary["budget_regime"] == "fixed_correction_budget")
        & summary["solver"].isin({config["primary_solver_id"], config["exact_eight_state_solver_id"]})
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for key, values in selected.groupby(["factor_config_id", "solver"], sort=True):
        values = values.sort_values("physical_budget_bpw")
        label = key[0].replace("joint_eigh_tail8_exact0_", "r8 ")
        if key[1] == config["exact_eight_state_solver_id"]:
            label = "exact Gram"
        axes[0].plot(values["physical_budget_bpw"], 100 * values["recovery_median"], marker="o", label=label)
        axes[1].plot(values["physical_budget_bpw"], 100 * values["recovery_gain_median"], marker="o", label=label)
    axes[0].set_title("Eight-state recovery")
    axes[0].set_ylabel("median recovery (%)")
    axes[1].set_title("Action-space gain, same solver/seed/factor")
    axes[1].set_ylabel("paired median gain (percentage points)")
    for axis in axes:
        axis.set_xlabel("physical correction budget (bpw)")
        axis.grid(alpha=.25)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "split_interaction_field_frontier.png", dpi=180)
    svg = output / "split_interaction_field_frontier.svg"
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def _report(summary: pd.DataFrame, promotion: dict[str, Any], config: dict[str, Any]) -> str:
    fixed_primary = summary[
        (summary["budget_regime"] == "fixed_correction_budget")
        & (summary["solver"] == config["primary_solver_id"])
        & (summary["physical_budget_pages"] == 768)
    ].sort_values(["recovery_median", "factor_config_id"], ascending=[False, True])
    exact = summary[
        (summary["budget_regime"] == "fixed_correction_budget")
        & (summary["solver"] == config["exact_eight_state_solver_id"])
        & (summary["physical_budget_pages"] == 768)
    ].iloc[0]
    all_in = summary[
        (summary["budget_regime"] == "strict_all_in_one_bpw")
        & (summary["solver"] == config["primary_solver_id"])
    ].sort_values("physical_budget_pages")
    lines = [
        "# Split gate/up/down interaction-field study", "", "## Outcome", "",
        f"- Frozen decision: **{promotion['status']}**.",
        f"- Passing compressed configurations: **{', '.join(promotion['passing_validation_configurations']) or 'none'}**.",
        "- The action-space attribution now compares compressed four-state and eight-state solvers with the same exact-self DP/all-00 seeds, coordinate descent, local repair, and factor.",
        (
            "- Exact-Gram same-solver eight-state median gain over four states: "
            f"{100*exact['recovery_gain_median']:+.4f} percentage points."
        ),
        "- This is an exact mixed-H4 geometry ceiling, not a deployment or downstream-quality claim.",
        "", "## Fixed one-correction-bpw compressed results", "",
        "| factor | p10 recovery | median recovery | action gain | gain vs PR11 | DP/solver headroom | metadata bpw | max MACs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in fixed_primary.to_dict("records"):
        lines.append(
            f"| {row['factor_config_id']} | {100*row['recovery_p10']:.3f}% | "
            f"{100*row['recovery_median']:.3f}% | {100*row['recovery_gain_median']:+.3f} pp | "
            f"{100*row['gain_vs_pr11_median']:+.3f} pp | {100*row['seed_optimizer_gain_median']:+.3f} pp | "
            f"{row['metadata_bpw']:.6f} | {int(row['selector_compute_macs_max']):,} |"
        )
    lines.extend([
        "", "## Strict all-in one-bpw comparison", "",
        "The correction-page budget is reduced so encoded A/B/C plus the factor plus correction pages fit within 393,216 bytes.",
        "",
        "| factor | page cap | total bpw | p10 recovery | median recovery | metadata bytes | max MACs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in all_in.to_dict("records"):
        lines.append(
            f"| {row['factor_config_id']} | {int(row['physical_budget_pages'])} | "
            f"{row['budget_total_bpw']:.6f} | {100*row['recovery_p10']:.3f}% | "
            f"{100*row['recovery_median']:.3f}% | {int(row['metadata_bytes']):,} | "
            f"{int(row['selector_compute_macs_max']):,} |"
        )
    delta = promotion["int8_minus_int4_all_in_recovery"]
    recommended = promotion["recommended_strict_all_in_factor"]
    rank4 = promotion["rank4_exact_proxy_all_in_control"]
    lines.extend([
        "",
        f"Frozen all-in recommendation: **{recommended['factor_config_id']}** at a {int(recommended['physical_budget_pages'])}-page cap.",
        (
            "INT8 minus INT4 paired all-in recovery p10/median/p90: "
            f"{100*delta['p10']:+.4f}/{100*delta['median']:+.4f}/{100*delta['p90']:+.4f} percentage points."
        ),
        (
            f"Rank-4 exact-proxy control: {int(rank4['physical_budget_pages'])} pages, "
            f"{100*rank4['recovery_p10']:.3f}% p10 / {100*rank4['recovery_median']:.3f}% median, "
            f"{int(rank4['selector_compute_macs_max']):,} max charged MACs."
        ),
        "", "## Actual solver work and wall time", "",
        "| factor | regime | pages | runtime p10/median/p90 ms | coordinate sweeps median/p90 | local passes median/p90 |",
        "|---|---|---:|---:|---:|---:|",
    ])
    diagnostic = summary[
        (summary["solver"] == config["primary_solver_id"])
        & (
            (summary["budget_regime"] == "strict_all_in_one_bpw")
            | (summary["physical_budget_pages"] == 768)
        )
    ].sort_values(["budget_regime", "factor_config_id"])
    for row in diagnostic.to_dict("records"):
        lines.append(
            f"| {row['factor_config_id']} | {row['budget_regime']} | {int(row['physical_budget_pages'])} | "
            f"{row['selector_runtime_ms_p10']:.3f}/{row['selector_runtime_ms_median']:.3f}/{row['selector_runtime_ms_p90']:.3f} | "
            f"{row['coordinate_sweeps_median']:.1f}/{row['coordinate_sweeps_p90']:.1f} | "
            f"{row['local_evaluated_passes_median']:.1f}/{row['local_evaluated_passes_p90']:.1f} |"
        )
    lines.extend([
        "",
        (
            "The recommended all-in path evaluates median/p90 "
            f"{recommended['local_evaluated_passes_median']:.1f}/"
            f"{recommended['local_evaluated_passes_p90']:.1f} local passes and "
            f"accepts a median {recommended['local_accepted_bundles_median']:.1f} bundles; "
            "it exhausts the frozen 12-pass cap, so this is a bounded-search result, "
            "not a convergence claim."
        ),
        "Measured wall times are the actual Python reference selector under the 24-worker quota-saturating run; they are not optimized kernel latency.",
        "", "## Interpretation boundary", "",
        "- The legacy PR #11 four-state result is retained separately from the new compressed four-state same-solver control.",
        "- Coordinate damage is updated incrementally; exact recomputation is retained as a terminal parity assertion.",
        "- Gate, up, and down are distinct one-page refinements; state cost is their popcount.",
        "- Exact A/B/C self terms and the unchanged L4/L2 factor score all eight states; no new interaction metadata is introduced.",
        "- Exact-Gram controls use training-only teacher geometry and are nonpromotable.",
        "- No test scientific value, candidate predictor, H4 prefetch, causal replay, routing, logit, or token metric is used.",
        "", "## Integrity", "",
        "- 69 validation invocations across 12 frozen layer/expert cells.",
        "- Four frozen PR #11 factors, three fixed correction budgets, four factor-specific strict all-in budgets, and exact row/state/page/accounting reconstruction.",
        "- Every exact eight-state result is checked not to lose to its embedded four-state result.", "",
    ])
    return "\n".join(lines)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    runner = EXPERIMENT / "scripts/run_split_interaction_field_study.py"
    core = EXPERIMENT / "src/oracle_study/split_interaction_field.py"
    config, facts, frame = validate_evidence_bundle(
        args.config, args.validation_dir, runner_path=runner, core_path=core,
    )
    summary = summary_table(frame)
    layer = layer_table(frame, config)
    paired = _paired_table(frame, config)
    usage = _state_usage(frame)
    promotion = promotion_payload(config, facts, frame, summary)
    all_in = summary[summary["budget_regime"] == "strict_all_in_one_bpw"].copy()
    diagnostics = summary[[
        "budget_regime", "factor_config_id", "solver", "physical_budget_pages",
        "selector_runtime_ms_p10", "selector_runtime_ms_median",
        "selector_runtime_ms_p90", "field_build_runtime_ms_median",
        "field_build_runtime_ms_p90", "coordinate_sweeps_median",
        "coordinate_sweeps_p90", "local_evaluated_passes_median",
        "local_evaluated_passes_p90", "coordinate_accepted_moves_median",
        "local_accepted_bundles_median",
    ]].copy()
    tables = {
        "split_interaction_field_summary.csv": summary,
        "split_interaction_field_layer_summary.csv": layer,
        "split_interaction_field_paired_invocations.csv": paired,
        "split_interaction_field_state_usage.csv": usage,
        "split_interaction_field_all_in_summary.csv": all_in,
        "split_interaction_field_solver_diagnostics.csv": diagnostics,
    }
    for name, table in tables.items():
        _atomic_csv(args.output / name, table)
    _atomic_json(args.output / "split_interaction_field_promotions.json", promotion)
    _atomic_json(args.output / "split_interaction_field_analysis.json", {
        "schema_version": 2, "run_id": config["run_id"],
        "summary_rows": summary.to_dict("records"), "promotion": promotion,
    })
    (args.output / "SPLIT_INTERACTION_FIELD_REPORT.md").write_text(
        _report(summary, promotion, config)
    )
    _plot(summary, config, args.output)
    generated = sorted(
        path for path in args.output.iterdir() if path.name != "analysis_manifest.json"
    )
    source_paths = [
        runner, core,
        EXPERIMENT / "src/oracle_study/split_interaction_field_analysis.py",
        Path(__file__).resolve(),
    ]
    input_paths = [
        args.config,
        args.validation_dir / "run_facts.json",
        args.validation_dir / "split_interaction_field_frontier.parquet",
    ]
    manifest = {
        "schema_version": 2, "run_id": config["run_id"],
        "inputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in input_paths
        ],
        "sources": [
            {
                "path": str(path.relative_to(EXPERIMENT)),
                "bytes": path.stat().st_size, "sha256": sha256(path),
            }
            for path in source_paths
        ],
        "outputs": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in generated
        ],
        "validation_only": True, "test_scientific_values_used": False,
    }
    _atomic_json(args.output / "analysis_manifest.json", manifest)


if __name__ == "__main__":
    main()
