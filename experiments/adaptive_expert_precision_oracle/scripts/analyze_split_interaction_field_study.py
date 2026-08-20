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
        config["primary_solver_id"], config["warm_start_solver_id"],
        config["exact_eight_state_solver_id"],
    })].copy()
    columns = list(IDENTITY) + [
        "factor_config_id", "solver", "physical_budget_bpw", "recovery",
        "four_state_reference_recovery",
        "recovery_gain_vs_four_state_reference",
        "set_gain_retention_vs_pr10_exact_hybrid",
        "split_only_units", "gate_high_units", "up_high_units",
        "down_high_units", "selector_compute_macs", "combined_metadata_bpw",
    ]
    return keep[columns].sort_values(
        ["solver", "factor_config_id", "physical_budget_bpw", *IDENTITY],
    ).reset_index(drop=True)


def _state_usage(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "split_only_units", "gate_high_units", "up_high_units",
        "down_high_units", "physical_pages",
    ]
    rows = []
    for key, values in frame.groupby(
        ["factor_config_id", "solver", "physical_budget_bpw"], sort=True,
    ):
        row = {
            "factor_config_id": key[0], "solver": key[1],
            "physical_budget_bpw": key[2], "rows": len(values),
        }
        for metric in metrics:
            row[f"{metric}_p10"] = float(values[metric].quantile(.1))
            row[f"{metric}_median"] = float(values[metric].quantile(.5))
            row[f"{metric}_p90"] = float(values[metric].quantile(.9))
        rows.append(row)
    return pd.DataFrame(rows)


def _plot(summary: pd.DataFrame, config: dict[str, Any], output: Path) -> None:
    selected = summary[summary["solver"].isin({
        config["primary_solver_id"], config["exact_eight_state_solver_id"],
    })]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for key, values in selected.groupby(["factor_config_id", "solver"], sort=True):
        values = values.sort_values("physical_budget_bpw")
        label = key[0].replace("joint_eigh_tail8_exact0_", "r8 ")
        if key[1] == config["exact_eight_state_solver_id"]:
            label = "exact Gram"
        axes[0].plot(
            values["physical_budget_bpw"], 100 * values["recovery_median"],
            marker="o", label=label,
        )
        axes[1].plot(
            values["physical_budget_bpw"], 100 * values["recovery_gain_median"],
            marker="o", label=label,
        )
    axes[0].set_title("Eight-state recovery")
    axes[0].set_ylabel("median recovery (%)")
    axes[1].set_title("Gain over same-solver four-state reference")
    axes[1].set_ylabel("paired median gain (percentage points)")
    for axis in axes:
        axis.set_xlabel("physical correction budget (bpw)")
        axis.grid(alpha=.25)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "split_interaction_field_frontier.png", dpi=180)
    svg = output / "split_interaction_field_frontier.svg"
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    plt.close(fig)


def _report(
    summary: pd.DataFrame, promotion: dict[str, Any], config: dict[str, Any],
) -> str:
    primary = summary[
        (summary["solver"] == config["primary_solver_id"])
        & np.isclose(summary["physical_budget_bpw"], 1.0)
    ].sort_values(["recovery_median", "factor_config_id"], ascending=[False, True])
    exact = summary[
        (summary["solver"] == config["exact_eight_state_solver_id"])
        & np.isclose(summary["physical_budget_bpw"], 1.0)
    ].iloc[0]
    lines = [
        "# Split gate/up/down interaction-field study", "",
        "## Outcome", "",
        f"- Frozen decision: **{promotion['status']}**.",
        f"- Passing compressed configurations: **{', '.join(promotion['passing_validation_configurations']) or 'none'}**.",
        "- No single winner is selected because the frozen protocol did not predeclare a tie-break.",
        (
            "- Exact same-solver eight-state median gain versus its embedded "
            f"four-state solution: {100*exact['recovery_gain_median']:+.4f} percentage points."
        ),
        "- This is an exact mixed-H4 geometry ceiling, not a deployment or downstream-quality claim.",
        "", "## One-bpw compressed results", "",
        "| factor | p10 recovery | median recovery | p10 retention | median gain | metadata bpw | max MACs | median split-only units |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary.to_dict("records"):
        lines.append(
            f"| {row['factor_config_id']} | {100*row['recovery_p10']:.3f}% | "
            f"{100*row['recovery_median']:.3f}% | "
            f"{100*row['set_gain_retention_p10']:.3f}% | "
            f"{100*row['recovery_gain_median']:+.3f} pp | "
            f"{row['metadata_bpw']:.6f} | {int(row['selector_compute_macs_max']):,} | "
            f"{row['split_only_units_median']:.1f} |"
        )
    lines.extend([
        "", "## Exact action-space control", "",
        (
            f"At one physical bpw the exact-Gram eight-state solver reaches "
            f"{100*exact['recovery_p10']:.3f}% p10 / "
            f"{100*exact['recovery_median']:.3f}% median recovery. Its paired "
            f"p10/median gain over exact restricted four-state search is "
            f"{100*exact['recovery_gain_p10']:+.3f}/"
            f"{100*exact['recovery_gain_median']:+.3f} percentage points."
        ),
        "", "## Interpretation boundary", "",
        "- Gate, up, and down are distinct one-page refinements; state cost is their popcount.",
        "- Exact A/B/C self terms and the unchanged L4/L2 factor score all eight states; no new interaction metadata is introduced.",
        "- The primary solver uses all-00 plus exact-self DP seeds, coordinate descent, and bounded 1/2/3-unit repair.",
        "- Inherited-four-state warm starts are charged diagnostics and are not promotion-eligible.",
        "- Exact-Gram controls use training-only teacher geometry and are nonpromotable.",
        "- No test scientific value, candidate predictor, H4 prefetch, causal replay, routing, logit, or token metric is used.",
        "", "## Integrity", "",
        "- 69 validation invocations across 12 frozen layer/expert cells.",
        "- Four frozen PR #11 factors, three page budgets, and exact row/state/page/accounting reconstruction.",
        "- Every exact eight-state result is checked not to lose to its embedded four-state result.",
        "",
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
    tables = {
        "split_interaction_field_summary.csv": summary,
        "split_interaction_field_layer_summary.csv": layer,
        "split_interaction_field_paired_invocations.csv": paired,
        "split_interaction_field_state_usage.csv": usage,
    }
    for name, table in tables.items():
        _atomic_csv(args.output / name, table)
    _atomic_json(args.output / "split_interaction_field_promotions.json", promotion)
    _atomic_json(args.output / "split_interaction_field_analysis.json", {
        "schema_version": 1, "run_id": config["run_id"],
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
        "schema_version": 1, "run_id": config["run_id"],
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
