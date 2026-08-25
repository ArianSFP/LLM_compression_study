#!/usr/bin/env python3
"""Finalize deterministic offline tables for the Experiment A D1 panel."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import pandas as pd


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "src"))

from oracle_study.d1_decode_tail import sha256  # noqa: E402
from oracle_study.d1_downstream_outcome_diagnosis import (  # noqa: E402
    downstream_outcome_diagnostics,
)
from oracle_study.d1_experiment_a_diagnostic_panel import (  # noqa: E402
    ROUTE_MODES,
    SCHEMA,
    bf16_ulp_analysis,
    cohort_policy_comparisons,
    immediate_d1_severity,
    interpolation_turning_points,
    page_direction_norm_explanation,
    route_mode_decomposition,
)


INPUT_SCHEMA = "pr13_d1_experiment_a_diagnostic_panel_v1"
QUALITY = "d1_experiment_a_diagnostic_quality.parquet"
ROUTES = "d1_experiment_a_diagnostic_routes.parquet"
ZERO = "d1_experiment_a_diagnostic_zero_gates.parquet"
PANEL = "d1_experiment_a_diagnostic_panel.parquet"
PANEL_FACTS = "d1_experiment_a_diagnostic_panel_facts.json"
RUN_FACTS = "d1_experiment_a_diagnostic_run_facts.json"
FACTS = "d1_experiment_a_diagnostic_analysis_facts.json"

OUTPUTS = {
    "route_mode_rows": "d1_diagnostic_route_mode_decomposition_rows.parquet",
    "route_mode_summary": "d1_diagnostic_route_mode_decomposition_summary.parquet",
    "d1_severity_rows": "d1_diagnostic_immediate_d1_severity_rows.parquet",
    "d1_severity_correlations": "d1_diagnostic_immediate_d1_severity_correlations.parquet",
    "d1_severity_summary": "d1_diagnostic_immediate_d1_severity_summary.parquet",
    "interpolation_points": "d1_diagnostic_interpolation_points.parquet",
    "interpolation_turning": "d1_diagnostic_interpolation_turning_points.parquet",
    "interpolation_summary": "d1_diagnostic_interpolation_summary.parquet",
    "direction_norm_rows": "d1_diagnostic_page_direction_vs_norm_rows.parquet",
    "direction_norm_summary": "d1_diagnostic_page_direction_vs_norm_summary.parquet",
    "bf16_rows": "d1_diagnostic_bf16_ulp_rows.parquet",
    "bf16_summary": "d1_diagnostic_bf16_ulp_summary.parquet",
    "bf16_association": "d1_diagnostic_bf16_ulp_terminal_association.parquet",
    "cohort_rows": "d1_diagnostic_cohort_fixed_vs_pr13_rows.parquet",
    "cohort_summary": "d1_diagnostic_cohort_fixed_vs_pr13_summary.parquet",
    "downstream_rows": "d1_diagnostic_downstream_outcome_rows.parquet",
    "downstream_paths": "d1_diagnostic_downstream_path_transition_summary.parquet",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def verified_inputs(
    root: Path,
    config_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    facts_path = root / RUN_FACTS
    panel_path = root / PANEL
    panel_facts_path = root / PANEL_FACTS
    facts = load_json(facts_path)
    panel_facts = load_json(panel_facts_path)
    if str(facts.get("schema")) != INPUT_SCHEMA or not bool(facts.get("completed")):
        raise RuntimeError("Experiment A diagnostic panel is not finalized")
    if str(facts.get("config_sha256")) != sha256(config_path):
        raise RuntimeError("diagnostic-panel config changed")
    if str(facts.get("panel_sha256")) != sha256(panel_path):
        raise RuntimeError("diagnostic panel changed")
    if str(facts.get("panel_facts_sha256")) != sha256(panel_facts_path):
        raise RuntimeError("diagnostic panel facts changed")
    if int(panel_facts.get("rows", -1)) != 24:
        raise RuntimeError("diagnostic panel no longer contains 24 token/layers")
    frames = {}
    for name in (QUALITY, ROUTES, ZERO):
        path = root / name
        record = facts.get("outputs", {}).get(name, {})
        if not path.is_file() or str(record.get("sha256")) != sha256(path):
            raise RuntimeError(f"diagnostic input changed: {name}")
        frame = pd.read_parquet(path)
        if len(frame) != int(record.get("rows", -1)):
            raise RuntimeError(f"diagnostic row count changed: {name}")
        frames[name] = frame
    if not bool(frames[ZERO]["all_exact"].all()):
        raise RuntimeError("diagnostic zero-dose parity failed")
    if tuple(sorted(frames[QUALITY]["route_mode"].astype(str).unique())) != tuple(
        sorted(ROUTE_MODES)
    ):
        raise RuntimeError("diagnostic route-mode grid changed")
    panel = pd.read_parquet(panel_path)
    if len(panel) != 24:
        raise RuntimeError("diagnostic panel row count changed")
    return frames[QUALITY], frames[ROUTES], frames[ZERO], panel, facts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_json(args.config)
    args.input = args.input or Path(str(config["output_root"]))
    args.output = args.output or args.input / "analysis"
    return args


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    input_root = args.input.resolve()
    output_root = args.output.resolve()
    quality, routes, zero, panel, run_facts = verified_inputs(input_root, config_path)

    route_rows, route_summary = route_mode_decomposition(quality, panel)
    severity_rows, severity_correlations, severity_summary = immediate_d1_severity(
        quality,
        routes,
        panel,
    )
    interpolation_points, interpolation_turning, interpolation_summary = (
        interpolation_turning_points(severity_rows)
    )
    direction_rows, direction_summary = page_direction_norm_explanation(severity_rows)
    bf16_rows, bf16_summary, bf16_association = bf16_ulp_analysis(quality, panel)
    cohort_rows, cohort_summary = cohort_policy_comparisons(severity_rows, route_rows)
    downstream_rows, downstream_paths = downstream_outcome_diagnostics(
        quality,
        routes,
        panel,
    )

    frames = {
        "route_mode_rows": route_rows,
        "route_mode_summary": route_summary,
        "d1_severity_rows": severity_rows,
        "d1_severity_correlations": severity_correlations,
        "d1_severity_summary": severity_summary,
        "interpolation_points": interpolation_points,
        "interpolation_turning": interpolation_turning,
        "interpolation_summary": interpolation_summary,
        "direction_norm_rows": direction_rows,
        "direction_norm_summary": direction_summary,
        "bf16_rows": bf16_rows,
        "bf16_summary": bf16_summary,
        "bf16_association": bf16_association,
        "cohort_rows": cohort_rows,
        "cohort_summary": cohort_summary,
        "downstream_rows": downstream_rows,
        "downstream_paths": downstream_paths,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    records = {}
    for key, frame in frames.items():
        if frame.empty or frame.isnull().any().any():
            raise RuntimeError(f"analysis table is empty or contains nulls: {key}")
        path = output_root / OUTPUTS[key]
        atomic_parquet(path, frame)
        records[path.name] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "rows": len(frame),
        }

    fixed_cohort = cohort_rows[
        cohort_rows["crossing_event"].isin(["prevented", "introduced"])
    ]
    facts = {
        "completed": True,
        "schema": SCHEMA,
        "config_sha256": sha256(config_path),
        "input_run_facts_sha256": sha256(input_root / RUN_FACTS),
        "input_panel_sha256": sha256(input_root / PANEL),
        "input_quality_sha256": sha256(input_root / QUALITY),
        "input_routes_sha256": sha256(input_root / ROUTES),
        "input_zero_sha256": sha256(input_root / ZERO),
        "implementation": {
            str(Path(__file__).resolve().relative_to(EXPERIMENT)): sha256(
                Path(__file__).resolve()
            ),
            "src/oracle_study/d1_experiment_a_diagnostic_panel.py": sha256(
                EXPERIMENT / "src/oracle_study/d1_experiment_a_diagnostic_panel.py"
            ),
            "src/oracle_study/d1_downstream_outcome_diagnosis.py": sha256(
                EXPERIMENT / "src/oracle_study/d1_downstream_outcome_diagnosis.py"
            ),
        },
        "outputs": records,
        "gates": {
            "zero_dose_all_exact": bool(zero["all_exact"].all()),
            "route_mode_decomposition_max_closure_error": float(
                route_rows["decomposition_closure_error"].abs().max()
            ),
            "direction_norm_decomposition_max_closure_error": float(
                direction_rows["kl_component_closure_error"].abs().max()
            ),
            "interpolation_endpoint_parity_exact": bool(
                run_facts["interpolation_endpoint_parity"]["all_exact"]
            ),
            "panel_rows": len(panel),
            "route_modes": list(ROUTE_MODES),
        },
        "numeric_diagnostics": {
            "prevented_or_introduced_rows": len(fixed_cohort),
            "mean_membership_execution_component_kl": float(
                route_rows["membership_execution_component_kl"].mean()
            ),
            "mean_router_weight_component_kl": float(
                route_rows["router_weight_component_kl"].mean()
            ),
            "interpolation_terminal_kl_nonmonotone_fraction": float(
                (
                    interpolation_turning["terminal_kl_discrete_turning_points"] > 0
                ).mean()
            ),
            "direction_dominant_fraction": float(
                direction_rows["larger_absolute_kl_component"].eq("direction").mean()
            ),
            "norm_dominant_fraction": float(
                direction_rows["larger_absolute_kl_component"].eq("norm").mean()
            ),
            "maximum_bf16_collapsed_nonzero_fraction": float(
                bf16_rows["bf16_collapsed_nonzero_fraction"].max()
            ),
            "terminal_nonmonotone_with_constant_d1_fraction": float(
                downstream_paths["terminal_nonmonotone_with_constant_d1"].mean()
            ),
            "terminal_nonmonotone_with_downstream_burden_transition_fraction": float(
                downstream_paths[
                    "terminal_nonmonotone_with_downstream_burden_transition"
                ].mean()
            ),
        },
        "terminal_kl_role": "observed_outcome_only_not_an_allocator_input",
        "observed_minimum_kl_lambda_is_descriptive_not_policy_selection": True,
        "d2_d4_objectives_or_models_present": False,
        "full_tail_routes_used_as_descriptive_outcomes_only": True,
        "experiment_b_components_present": False,
        "report_prose_generated": False,
    }
    atomic_json(output_root / FACTS, facts)
    print(
        f"[diagnostic analysis] outputs={len(records)} rows={sum(len(frame) for frame in frames.values())}",
        flush=True,
    )


if __name__ == "__main__":
    main()
