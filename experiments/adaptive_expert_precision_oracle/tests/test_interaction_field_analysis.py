from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from oracle_study.interaction_field_analysis import (
    Evidence,
    EXPECTED_COMPUTE_BOUNDS,
    EXPECTED_NON_MAC_CONVENTION,
    GEOMETRY_SOLVER,
    PRIMARY_SOLVER,
    _physical_factor_bytes,
    _validate_compute_accounting,
    expected_factor_ids,
    summarize,
)


EXPERIMENT = Path(__file__).parents[1]
CONFIG = EXPERIMENT / "configs/qwen36_mxfp4_interaction_field.json"


def test_frozen_factor_grid_has_thirteen_geometry_and_seven_quantized_points():
    config = json.loads(CONFIG.read_text())
    identifiers = expected_factor_ids(config)
    assert len(identifiers) == 20
    assert "joint_eigh_tail16_exact0_int4_per_row_hadamard" in identifiers
    assert "exact_proxy_plus_eigh_tail_tail0_exact4_fp32" in identifiers


def test_physical_payload_includes_codes_scales_and_global_shrink():
    assert _physical_factor_bytes({"total_rank": 16, "encoding": "fp32"}) == 65_536
    assert _physical_factor_bytes({"total_rank": 16, "encoding": "int8_per_row"}) == 18_436
    assert _physical_factor_bytes({"total_rank": 16, "encoding": "int4_per_row"}) == 10_244

def test_compute_accounting_is_reconstructed_from_operation_counters():
    rank = 4
    config = {"local_shortlist": 8}
    specifications = (
        ("compressed_residual_forward", 17, 0, 0, 0),
        ("continuous_relaxation_round", 0, 3, 0, 0),
        ("four_seed_coordinate", 17, 3, 5, 0),
        (GEOMETRY_SOLVER, 17, 3, 5, 2),
        (PRIMARY_SOLVER, 0, 0, 3, 2),
    )
    rows = []
    for solver, forward, relaxation, coordinate, local_passes in specifications:
        coordinate_solver = solver in {
            "four_seed_coordinate", GEOMETRY_SOLVER, PRIMARY_SOLVER,
        }
        relaxation_solver = solver in {
            "continuous_relaxation_round", "four_seed_coordinate", GEOMETRY_SOLVER,
        }
        components = {
            "field_build": 8 * 512 * rank + 12 * 512,
            "independent_seed": 6 * 512 if coordinate_solver else 0,
            "forward": 2 * rank * forward,
            "relaxation_setup": 4 * 512 * rank if relaxation_solver else 0,
            "relaxation_iterations": 8 * 512 * rank * relaxation,
            "coordinate": (2 * 512 * rank + 8 * 512) * coordinate,
            "local": (
                local_passes
                * (6 * 512 * rank + (7 * config["local_shortlist"]) ** 2 * rank)
            ),
        }
        row = {
            "total_rank": rank,
            "solver": solver,
            "selector_compute_macs": sum(components.values()),
            "selector_compute_accounting_bound": EXPECTED_COMPUTE_BOUNDS[solver],
            "selector_non_mac_operations": EXPECTED_NON_MAC_CONVENTION,
            "forward_candidate_evaluations": forward,
            "relaxation_iterations": relaxation,
            "coordinate_sweeps": coordinate,
            "local_evaluated_passes": local_passes,
        }
        for name, value in components.items():
            row[f"selector_{name}_compute_macs"] = value
        rows.append(row)
    frame = pd.DataFrame(rows)
    _validate_compute_accounting(frame, config)
    broken = frame.copy()
    broken.loc[0, "selector_forward_compute_macs"] += 1
    with pytest.raises(RuntimeError, match="components do not sum"):
        _validate_compute_accounting(broken, config)


def test_summary_separates_geometry_success_from_current_all_seed_compute():
    config = json.loads(CONFIG.read_text())
    rows = []
    specifications = (
        ("joint_eigh_tail16_exact0_fp32", "fp32", True, 65_536, 0.981),
        (
            "joint_eigh_tail16_exact0_int4_per_row_hadamard",
            "int4_per_row", False, 10_244, 0.979,
        ),
    )
    for factor_id, encoding, geometry, payload, recovery in specifications:
        for index in range(10):
            exact = 0.98
            value = recovery - 0.0002 * index
            rows.append({
                "factor_config_id": factor_id,
                "factor_family": "joint_eigh",
                "tail_rank": 16,
                "exact_rank": 0,
                "total_rank": 16,
                "encoding": encoding,
                "hadamard_rotated": encoding == "int4_per_row",
                "geometry_ceiling": geometry,
                "solver": PRIMARY_SOLVER,
                "physical_budget_pages": 768,
                "recovery": value,
                "set_gain_retention_vs_pr10_exact_hybrid": value / exact,
                "recovery_gap_vs_pr10_exact_hybrid": value - exact,
                "selector_compute_macs": 800_000 if encoding != "fp32" else 2_000_000,
                "factor_payload_bytes": payload,
                "combined_metadata_bpw": 8.0 * (payload + 3_084) / 3_145_728,
                "layer": 0,
            })
    rows.extend({**row, "solver": GEOMETRY_SOLVER} for row in list(rows))
    evidence = Evidence(config, {}, {}, pd.DataFrame(rows), Path("."))
    factor, solver, layer, conclusion = summarize(evidence)
    assert len(factor) == 4
    assert len(solver) == 4
    assert len(layer) == 2
    assert conclusion["any_geometry_gate_pass"] is True
    assert conclusion["any_quantized_geometry_gate_pass"] is True
    assert conclusion["any_quantized_full_compute_and_metadata_pass"] is True
    assert conclusion["best_quantized_full_gate_configuration"] == (
        "joint_eigh_tail16_exact0_int4_per_row_hadamard"
    )
    assert conclusion["best_quantized_full_gate_metrics"]["selector_compute_macs_max"] == 800_000
