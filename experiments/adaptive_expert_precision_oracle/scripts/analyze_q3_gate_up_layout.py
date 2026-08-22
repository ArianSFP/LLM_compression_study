#!/usr/bin/env python3
"""Analyze immutable gate/up-Q3 evidence and emit a rate-curve report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "src"))

from oracle_study.q3_gate_up_analysis import (  # noqa: E402
    BASELINE_LAYOUT, PRIMARY_POLICY, accuracy_table, promotion_payload,
    threshold_table, validate_evidence_bundle,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _finite_or_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_or_none(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _experiment_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (EXPERIMENT / path).resolve()


def _normalize_svg(path: Path) -> None:
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")




def _report(
    accuracy: pd.DataFrame, thresholds: pd.DataFrame, promotion: dict[str, Any],
    runtime: pd.DataFrame, config: dict[str, Any],
) -> str:
    primary = accuracy[accuracy.allocation_policy == PRIMARY_POLICY]
    strict = primary[primary.mean_budget_quanta_per_expert == int(config["strict_all_in_mean_quanta"])]
    diagnostic = promotion["ideal_logical_solver_diagnostic"]
    target = promotion["frozen_pr13_rank4_target"]
    reproduction = promotion["restored_eight_state_baseline"]
    lines = [
        "# Gate/up Q3 physical-layout study", "",
        "## Outcome", "",
        f"Gate/up-Q3 status: **{promotion['gate_up_q3_status']}**.",
        f"Selected physical layout: `{promotion['selected_physical_layout']}`.",
        f"Down-Q3 status: **{promotion['down_q3_status']}**.", "",
        "This is an exact-H4 Rank-4 Hadamard geometry/layout study. It is not a deployable predictor, latency, downstream, routing, logit, or token-quality result.",
        f"The frozen PR #13 target is {100*target['recovery_p10']:.4f}% p10 / {100*target['recovery_median']:.4f}% median at {target['total_bpw']:.6f} total bpw.",
        f"The restored restricted-eight-state DP control reaches {100*reproduction['recovery_p10']:.4f}% p10 / {100*reproduction['recovery_median']:.4f}% median; reproduction gate: **{reproduction['reproduction_pass']}**.",
        "Every Q3 candidate frontier includes the reproduced Q2/Q4 states. The ideal candidate frontier additionally includes every physical candidate after re-costing it in 256-byte quanta.",
        "The ideal row remains a heuristic solver result rather than a globally certified exact-recovery ceiling; constructive inclusion certifies compressed-objective dominance, not exact-qenergy ordering.",
        f"At strict rate, witness-minus-ideal recovery gaps versus `{diagnostic['strict_feasible_witness_layout']}` are {100*diagnostic['solver_median_recovery_gap_to_feasible_witness']:.4f} median pp and {100*diagnostic['solver_p10_recovery_gap_to_feasible_witness']:.4f} p10 pp (positive means the ideal solver trails); this diagnoses optimizer headroom without reversing the sign.", "",
        "## Frozen-target all-in comparison", "",
        "| Layout | p10 recovery | Median recovery | Median remaining-damage ratio vs frozen PR #13 | Total bpw |",
        "|---|---:|---:|---:|---:|",
    ]
    comparison = {item["layout_id"]: item for item in promotion["physical_layout_comparisons"]}
    for row in strict.sort_values("layout_id").itertuples(index=False):
        ratio = (1.0 - float(row.recovery_median)) / (1.0 - float(target["recovery_median"]))
        lines.append(f"| `{row.layout_id}` | {100*row.recovery_p10:.4f}% | {100*row.recovery_median:.4f}% | {'—' if ratio is None else f'{ratio:.4f}'} | {row.average_allowed_total_bpw:.6f} |")
    selected_layout = promotion["selected_physical_layout"]
    if selected_layout is None:
        selected_layout = min(
            promotion["physical_layout_comparisons"],
            key=lambda item: (item["strict_median_remaining_damage_ratio_vs_frozen_pr13_target"], item["layout_id"]),
        )["layout_id"]
    selected_curve = primary[primary.layout_id == selected_layout].sort_values("average_allowed_total_bpw")
    lines += ["", f"## Average-rate accuracy curve: `{selected_layout}`", "",
              "| Allowed total bpw | Mean actual total bpw | p10 | Median | p90 | Median residual ratio vs eight-state |",
              "|---:|---:|---:|---:|---:|---:|"]
    for row in selected_curve.itertuples(index=False):
        lines.append(
            f"| {row.average_allowed_total_bpw:.6f} | {row.average_actual_total_bpw_mean:.6f} | "
            f"{100*row.recovery_p10:.4f}% | {100*row.recovery_median:.4f}% | {100*row.recovery_p90:.4f}% | "
            f"{row.median_remaining_damage_ratio_vs_baseline_same_budget:.4f} |"
        )
    lines += ["", "## Minimum rate matching frozen PR #13 p10 and median", "",
              f"These are grid-certified minima. The dense target region is sampled every {target['search_resolution_quanta']} quanta ({target['search_resolution_total_bpw']:.6f} total bpw); the preceding sampled point is the lower edge of the certified interval.", "",
              "| Physical layout | Previous sampled bpw (fails one or both targets) | Minimum sampled total bpw | Delta vs PR #13 |", "|---|---:|---:|---:|"]
    for item in promotion["physical_layout_comparisons"]:
        matched = item["matched_frozen_pr13_p10_and_median_minimum_total_bpw"]
        delta = item["matched_quality_total_bpw_delta"]
        previous = item["previous_sampled_total_bpw"]
        lines.append(
            f"| {item['layout_id']} | "
            f"{'—' if previous is None else f'{previous:.6f}'} | "
            f"{'not reached' if matched is None else f'{matched:.6f}'} | "
            f"{'—' if delta is None else f'{delta:+.6f}'} |"
        )
    lines += ["", "## Minimum total bpw at target quality", "",
              "| Layout | Statistic | Target | Minimum total bpw |", "|---|---|---:|---:|"]
    for row in thresholds.itertuples(index=False):
        value = "not reached" if pd.isna(row.minimum_total_bpw) else f"{row.minimum_total_bpw:.6f}"
        lines.append(f"| `{row.layout_id}` | {row.metric} | {100*row.target:.2f}% | {value} |")
    lines += ["", "## Runtime", "",
              "| Layout | p50 total selector | p50 frontier | p50 allocation | total p90 | total p99 | Median sweeps | Median local passes | Median DP tables | Median DP state updates |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in runtime.itertuples(index=False):
        lines.append(f"| `{row.layout_id}` | {row.wall_ms_p50:.2f} ms | {row.frontier_ms_p50:.2f} ms | {row.allocation_ms_p50:.2f} ms | {row.wall_ms_p90:.2f} ms | {row.wall_ms_p99:.2f} ms | {row.coordinate_sweeps_median:.1f} | {row.local_passes_median:.1f} | {row.diagonal_dp_tables_median:.1f} | {row.diagonal_dp_state_updates_median:.1f} |")
    lines += ["", "## Accounting and interpretation", "",
              "- Exact-self dynamic programming seeds both the full 18-state and restricted inherited-eight-state additive layouts before the identical coordinate/local solver.",
              "- Ideal Q3 charges independent 256-byte planes; its candidate set is a constructive superset of every physical frontier, but its selected exact-recovery row is still a nonphysical heuristic control rather than a certified global optimum.",
              "- Every physical Q3 result charges the exact union of 512-byte page IDs and includes a fully charged legacy Q2/Q4 packing replica, making every restored PR #13 state physically feasible at its original page cost.",
              "- Each expert chooses one complete packing replica and never mixes pages across replicas. External-storage accounting charges every extra gate/up refinement copy.",
              "- The restored eight-state comparator uses the same DP seed, coordinate/local solver, and Rank-4 factor, restricted to inherited states with monolithic Q2→Q4 projection pages.",
              "- Learned layouts use only routed train occurrences. Validation values never fit pairings or factors; test scientific values are never admitted.",
              "- The table reports allowed total bpw, including A/B/C, factor, amortized learned-layout descriptor, and correction bytes.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--layout-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.config = _experiment_path(args.config)
    args.layout_dir = _experiment_path(args.layout_dir)
    args.validation_dir = _experiment_path(args.validation_dir)
    args.output = _experiment_path(args.output)
    args.output.mkdir(parents=True, exist_ok=False)
    config, group, expert, evidence = validate_evidence_bundle(
        args.config, args.layout_dir, args.validation_dir, experiment=EXPERIMENT,
    )
    accuracy = accuracy_table(group)
    thresholds = threshold_table(accuracy, config)
    promotion = promotion_payload(accuracy, config)
    primary = group[group.allocation_policy == PRIMARY_POLICY]
    runtime = primary.groupby("layout_id", sort=True).agg(
        wall_ms_p50=("selector_wall_time_ms", "median"),
        frontier_ms_p50=("selector_frontier_wall_time_ms", "median"),
        allocation_ms_p50=("selector_allocation_wall_time_ms", "median"),
        wall_ms_p90=("selector_wall_time_ms", lambda x: x.quantile(.9)),
        wall_ms_p99=("selector_wall_time_ms", lambda x: x.quantile(.99)),
        coordinate_sweeps_median=("selector_coordinate_sweeps", "median"),
        local_passes_median=("selector_local_passes", "median"),
        selector_macs_max=("selector_compute_macs", "max"),
        diagonal_dp_tables_median=("selector_diagonal_dp_tables", "median"),
        diagonal_dp_state_updates_median=("selector_diagonal_dp_state_updates", "median"),
    ).reset_index()
    layer = primary.groupby(["layout_id", "mean_budget_quanta_per_expert", "layer"], sort=True).group_recovery.agg(
        recovery_p10=lambda x: x.quantile(.1), recovery_median="median", recovery_p90=lambda x: x.quantile(.9), groups="size",
    ).reset_index()
    state_usage = expert.groupby(["layout_id", "mean_budget_quanta_per_expert"], sort=True).agg(
        gate_q3_or_above_mean=("selected_gate_q3_or_above_units", "mean"),
        gate_q4_mean=("selected_gate_q4_units", "mean"),
        up_q3_or_above_mean=("selected_up_q3_or_above_units", "mean"),
        up_q4_mean=("selected_up_q4_units", "mean"),
        down_q4_mean=("selected_down_q4_units", "mean"),
        selected_quanta_mean=("selected_quanta", "mean"),
    ).reset_index()
    storage = primary.groupby("layout_id", sort=True).agg(
        metadata_bpw=("combined_metadata_bpw", "first"),
        external_storage_multiplier=("external_storage_multiplier", "first"),
        layout_replicas=("layout_replicas", "first"),
        layout_is_physical=("layout_is_physical", "first"),
    ).reset_index()
    outputs = {
        "q3_accuracy_by_bpw.csv": accuracy,
        "q3_minimum_bpw_thresholds.csv": thresholds,
        "q3_layer_accuracy.csv": layer,
        "q3_runtime.csv": runtime,
        "q3_state_usage.csv": state_usage,
        "q3_storage_accounting.csv": storage,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output / name, index=False)
    _json(args.output / "q3_gate_up_promotions.json", _finite_or_none(promotion))
    report = _report(accuracy, thresholds, promotion, runtime, config)
    (args.output / "Q3_GATE_UP_LAYOUT_REPORT.md").write_text(report)
    figure, axis = plt.subplots(figsize=(9, 5.5))
    plt.rcParams["svg.hashsalt"] = "q3-gate-up-layout"
    for layout, frame in accuracy[accuracy.allocation_policy == PRIMARY_POLICY].groupby("layout_id", sort=True):
        frame = frame.sort_values("average_allowed_total_bpw")
        axis.plot(frame.average_allowed_total_bpw, 100 * frame.recovery_median, label=layout)
    axis.set(xlabel="Allowed total average bpw", ylabel="Median exact-H4 group recovery (%)",
             title="Gate/up Q3 rate-curve left shift")
    axis.grid(alpha=.25); axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(args.output / "q3_rate_curve.png", dpi=180)
    svg_path = args.output / "q3_rate_curve.svg"
    figure.savefig(svg_path, metadata={"Date": None})
    _normalize_svg(svg_path)
    plt.close(figure)
    inputs = [args.config, args.layout_dir / "q3_layout_fit_facts.json",
              args.layout_dir / "q3_layout_manifest.json",
              args.validation_dir / "q3_gate_up_run_facts.json",
              args.validation_dir / "q3_gate_up_accounting.json",
              args.validation_dir / "q3_gate_up_group_frontier.parquet",
              args.validation_dir / "q3_gate_up_expert_allocation.parquet"]
    generated = sorted(path for path in args.output.iterdir() if path.name != "q3_analysis_manifest.json")
    manifest = {
        "schema_version": 1, "completed": True,
        "inputs": [{"path": str(path.relative_to(EXPERIMENT)), "bytes": path.stat().st_size, "sha256": _sha256(path)} for path in inputs],
        "sources": [{"path": str(path.relative_to(EXPERIMENT)), "bytes": path.stat().st_size, "sha256": _sha256(path)} for path in (
            Path(__file__).resolve(), EXPERIMENT / "src/oracle_study/q3_gate_up_analysis.py",
            EXPERIMENT / "scripts/run_q3_gate_up_layout.py", EXPERIMENT / "src/oracle_study/q3_gate_up_field.py",
            EXPERIMENT / "src/oracle_study/q3_rate_allocator.py")],
        "outputs": [{"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)} for path in generated],
        "promotion_status": promotion["gate_up_q3_status"],
        "selected_physical_layout": promotion["selected_physical_layout"],
        "test_scientific_values_admitted_or_used": False,
    }
    _json(args.output / "q3_analysis_manifest.json", manifest)


if __name__ == "__main__":
    main()
