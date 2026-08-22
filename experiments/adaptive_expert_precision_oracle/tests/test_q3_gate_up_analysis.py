import numpy as np
from pathlib import Path
import pandas as pd

from oracle_study.q3_gate_up_analysis import (
    BASELINE_LAYOUT, PRIMARY_POLICY, ROUTER_SUM_ATOL, _expected_layout_accounting,
    accuracy_table, promotion_payload, threshold_table,
)

from analyze_q3_gate_up_layout import EXPERIMENT, _experiment_path, _normalize_svg, _report



def test_router_sum_tolerance_covers_fp32_reduction_not_material_drift():
    assert ROUTER_SUM_ATOL > 2e-7
    assert ROUTER_SUM_ATOL < 1e-6
    assert np.isclose(1.0 + 1.8e-7, 1.0, rtol=0, atol=ROUTER_SUM_ATOL)
    assert not np.isclose(1.0 + 2e-6, 1.0, rtol=0, atol=ROUTER_SUM_ATOL)


def test_analyzer_resolves_repository_relative_cli_paths():
    relative = Path("configs/qwen36_mxfp4_q3_gate_up_layout.json")
    assert _experiment_path(relative) == (EXPERIMENT / relative).resolve()
    absolute = Path("/tmp/q3-analysis-output")
    assert _experiment_path(absolute) == absolute.resolve()


def test_svg_normalization_removes_trailing_whitespace(tmp_path):
    path = tmp_path / "plot.svg"
    path.write_text("<svg>  \n  <path /> \t\n</svg>\n")
    _normalize_svg(path)
    assert path.read_text() == "<svg>\n  <path />\n</svg>\n"


def _rows():
    rows = []
    layouts = [BASELINE_LAYOUT, "q3_ideal_logical_256_byte_plane_ceiling", "q3_physical_fixed_gate_up_pairing"]
    for layout in layouts:
        for quanta, bpw in ((100, .8), (120, 1.0)):
            center = (.99 if quanta == 100 else .995)
            if layout == "q3_physical_fixed_gate_up_pairing":
                center += .002
            elif layout == "q3_ideal_logical_256_byte_plane_ceiling":
                center += .001
            for value in (center - .001, center, center + .001):
                rows.append({"layout_id": layout, "allocation_policy": PRIMARY_POLICY,
                             "mean_budget_quanta_per_expert": quanta,
                             "average_allowed_total_bpw": bpw,
                             "average_actual_total_bpw": bpw - .01,
                             "group_recovery": value})
    return pd.DataFrame(rows)


def test_accuracy_and_threshold_tables_are_rate_ordered():
    accuracy = accuracy_table(_rows())
    assert len(accuracy) == 6
    config = {"median_targets": [.99, .995], "p10_floor_targets": [.99]}
    thresholds = threshold_table(accuracy, config)
    fixed = thresholds[(thresholds.layout_id == "q3_physical_fixed_gate_up_pairing") & (thresholds.metric == "median")]
    assert fixed.minimum_total_bpw.tolist() == [.8, 1.0]
    strict = accuracy[(accuracy.layout_id == "q3_physical_fixed_gate_up_pairing") & (accuracy.mean_budget_quanta_per_expert == 120)].iloc[0]
    assert np.isclose(strict.median_remaining_damage_ratio_vs_baseline_same_budget, .6)
    baseline = accuracy[(accuracy.layout_id == BASELINE_LAYOUT) & (accuracy.mean_budget_quanta_per_expert == 120)].iloc[0]
    assert np.isclose(baseline.median_remaining_damage_ratio_vs_baseline_same_budget, 1.0)


def test_promotion_uses_residual_ratio_or_matched_rate_shift():
    accuracy = accuracy_table(_rows())
    config = {
        "strict_all_in_mean_quanta": 120,
        "primary_physical_layouts": ["q3_physical_fixed_gate_up_pairing"],
        "promotion_same_rate_remaining_damage_ratio_max": .8,
        "promotion_matched_quality_total_bpw_delta_max": -.1,
        "frozen_pr13_rank4_target_recovery_p10": .9942,
        "frozen_pr13_rank4_target_recovery_median": .995,
        "frozen_pr13_rank4_target_total_bpw": 1.0,
        "baseline_reproduction_absolute_tolerance": 1e-12,
        "matched_target_search_resolution_quanta": 20,
        "cost_quantum_bytes": 256,
        "expert_weights": 409600,
        "study_scope": "synthetic",
    }
    payload = promotion_payload(accuracy, config)
    assert payload["gate_up_q3_status"] == "continue_to_predictive_q3_study"
    assert payload["selected_physical_layout"] == "q3_physical_fixed_gate_up_pairing"
    comparison = payload["physical_layout_comparisons"][0]
    assert np.isclose(
        comparison["strict_median_remaining_damage_ratio_vs_frozen_pr13_target"], .6,
    )
    assert payload["restored_eight_state_baseline"]["reproduction_pass"] is True
    assert np.isclose(payload["frozen_pr13_rank4_target"]["search_resolution_total_bpw"], .1)
    assert comparison["matched_frozen_pr13_mean_quanta"] == 120
    assert np.isclose(comparison["previous_sampled_total_bpw"], .8)
    diagnostic = payload["ideal_logical_solver_diagnostic"]
    assert diagnostic["candidate_frontier_is_constructive_superset_of_every_physical_frontier"] is True
    assert diagnostic["reported_ideal_control_is_certified_global_ceiling"] is False
    assert diagnostic["strict_feasible_witness_layout"] == "q3_physical_fixed_gate_up_pairing"
    assert np.isclose(diagnostic["solver_median_recovery_gap_to_feasible_witness"], .001)
    assert payload["schema_version"] == 3
    runtime = pd.DataFrame([
        {
            "layout_id": layout,
            "wall_ms_p50": 1.0,
            "frontier_ms_p50": 1.0,
            "allocation_ms_p50": 1.0,
            "wall_ms_p90": 1.0,
            "wall_ms_p99": 1.0,
            "coordinate_sweeps_median": 1.0,
            "local_passes_median": 1.0,
            "diagonal_dp_tables_median": 1.0,
            "diagonal_dp_state_updates_median": 1.0,
        }
        for layout in sorted(accuracy.layout_id.unique())
    ])
    report = _report(accuracy, pd.DataFrame(), payload, runtime, config)
    assert "witness-minus-ideal recovery gaps" in report
    assert "positive means the ideal solver trails" in report


def test_layout_accounting_is_derived_from_frozen_payloads():
    config = {
        "layout_descriptor_bytes_per_layer_single": 4096,
        "layout_descriptor_bytes_per_layer_replicated": 8192,
        "factor_payload_bytes_per_expert": 4100,
        "abc_metadata_bytes_per_expert": 3084,
    }
    baseline = _expected_layout_accounting(BASELINE_LAYOUT, config)
    ideal = _expected_layout_accounting("q3_ideal_logical_256_byte_plane_ceiling", config)
    single = _expected_layout_accounting("q3_physical_training_coselection_single", config)
    replicated = _expected_layout_accounting("q3_physical_training_coselection_replicated2", config)
    assert baseline[:3] == (0, 1, True) and ideal[:3] == (0, 0, False)
    assert single[:3] == (16, 2, True) and replicated[:3] == (32, 3, True)
    assert replicated[3] > single[3] > 1.0
