import json
from pathlib import Path

import numpy as np
import pytest

from oracle_study.average_rate_allocator import AllocationTrace, RateOption
from oracle_study.interaction_field import JointInteractionFactor
from oracle_study.q3_gate_up_field import (
    Q3InteractionField, Q3PageLayout, fixed_gate_up_layout, monolithic_q2q4_layout,
)
from run_q3_gate_up_layout import (
    GROUP_IDENTITY, _allocation_rows, _assert_allocation_objective_dominates,
    _layout_descriptor_bytes, _layouts, _merge_rebased_frontiers,
    _validate_contract,
)


EXPERIMENT = Path(__file__).resolve().parents[1]


def _config():
    return json.loads((EXPERIMENT / "configs/qwen36_mxfp4_q3_gate_up_layout.json").read_text())


def test_frozen_config_contract_and_layout_grid():
    config = _config()
    _validate_contract(config)
    pages = np.empty((2, 2048), np.int64)
    for replica in range(2):
        pages[replica] = np.repeat(np.arange(1024), 2)
        if replica:
            pages[replica] = np.roll(pages[replica], 1)
    layouts = _layouts(pages)
    assert list(layouts) == config["layout_controls"]
    assert layouts["q3_ideal_logical_256_byte_plane_ceiling"].ideal
    assert layouts["q3_physical_fixed_gate_up_pairing"].replicas == 2
    assert layouts["q3_physical_training_coselection_single"].replicas == 2
    assert layouts["q3_physical_training_coselection_replicated2"].replicas == 3

def test_injected_frontier_witnesses_and_objective_dominance_are_fail_closed():
    units, rank = 2, 1
    factor = JointInteractionFactor(
        np.zeros((units, rank)), np.zeros((units, rank)),
        "synthetic", rank, 0, "fp64",
    )
    self_residual = np.full((units, 18), 10.0)
    self_residual[:, 17] = 0.0
    field = Q3InteractionField(
        np.zeros((units, 18, rank)), np.zeros((units, 18, 2)),
        self_residual.copy(), self_residual, factor,
    )
    fixed = fixed_gate_up_layout(units)
    legacy = monolithic_q2q4_layout(units)
    layout = Q3PageLayout(
        "fixed_plus_legacy", units,
        np.concatenate((fixed.action_pages, legacy.action_pages)),
    )
    zero = np.zeros(units, np.int64)
    upgraded = np.asarray([17, 0], np.int64)
    native = (RateOption(layout.cost_quanta(zero), field.damage(zero), zero, "native", 0.0, 0, 0),)
    injected = (RateOption(999, 999.0, upgraded, "reference", 0.0, 2, 1),)
    merged = _merge_rebased_frontiers(
        field, layout, native, (("restricted_eight_state", injected),),
        maximum_quanta=12,
    )
    assert layout.cost_quanta(upgraded) <= legacy.cost_quanta(upgraded)
    assert any(
        option.pages == layout.cost_quanta(upgraded)
        and np.isclose(option.damage, field.damage(upgraded))
        for option in merged
    )
    better = AllocationTrace(np.zeros(1, np.int64), 0, 1.0)
    worse = AllocationTrace(np.zeros(1, np.int64), 0, 2.0)
    _assert_allocation_objective_dominates(better, worse, "synthetic")
    with pytest.raises(RuntimeError, match="compressed-objective dominance"):
        _assert_allocation_objective_dominates(worse, better, "synthetic")



def test_layout_descriptor_is_conservatively_amortized_per_expert():
    config = _config()
    assert _layout_descriptor_bytes("q3_physical_training_coselection_single", config) == 16
    assert _layout_descriptor_bytes("q3_physical_training_coselection_replicated2", config) == 32
    assert _layout_descriptor_bytes("q3_physical_fixed_gate_up_pairing", config) == 0


def test_group_row_uses_exact_quantum_and_total_bpw_accounting():
    config = _config()
    pages = np.repeat(np.arange(1024), 2)[None, :]
    layout = _layouts(np.concatenate((pages, pages), axis=0))[
        "q3_physical_training_coselection_single"
    ]
    states = np.zeros(512, np.int64)
    states[:10] = 17
    cost = layout.cost_quanta(states)
    option = RateOption(cost, 1.0, states, "test", 0.0, 1, 1)
    frontiers = [(option,) for _ in range(8)]
    features = [np.asarray([[1.0, 0.0]]) for _ in range(8)]
    exact = [np.asarray([1.0]) for _ in range(8)]
    base = [np.asarray([1.0, 0.0]) for _ in range(8)]
    group = {
        "capture_source": "exact_checkpoint", "evaluation_split": "validation",
        "request_id": "r", "position": 3, "layer": 0, "sequence_id": "s",
        "experts": list(range(8)),
    }
    allocation = AllocationTrace(np.zeros(8, np.int64), 8 * cost, 8.0)
    row, experts = _allocation_rows(
        group, "q3_physical_training_coselection_single", layout,
        frontiers, features, exact, base, np.full(8, 1 / 8), cost,
        "pooled_router_square_multi_budget_column_generated", allocation,
        .01, {"coordinate_sweeps": 8, "local_passes": 2,
              "state_comparisons": 100, "diagonal_dp_tables": 8, "diagonal_dp_state_updates": 1234,
              "frontier_wall_seconds": .006, "allocation_wall_seconds": .004, "refinement_rounds": 1}, config,
    )
    assert row["actual_group_quanta"] == 8 * cost
    expected = (
        row["combined_metadata_bytes_per_expert"] + cost * 256
    ) * 8 / config["expert_weights"]
    assert np.isclose(row["average_actual_total_bpw"], expected)
    assert np.isclose(row["average_allowed_total_bpw"], expected)
    assert len(experts) == 8
    assert np.isclose(row["selector_frontier_wall_time_ms"], 6.0)
    assert np.isclose(row["selector_allocation_wall_time_ms"], 4.0)
    assert np.isclose(row["selector_wall_time_ms"], 10.0)
    assert all(len(expert["selected_states_blob"]) == 512 for expert in experts)
    assert all(expert["selected_unique_pages"] * 2 == expert["selected_quanta"] for expert in experts)
    assert all(name in row for name in GROUP_IDENTITY)

    ideal = _layouts(np.concatenate((pages, pages), axis=0))[
        "q3_ideal_logical_256_byte_plane_ceiling"
    ]
    ideal_cost = ideal.cost_quanta(states)
    ideal_option = RateOption(ideal_cost, 1.0, states, "ideal_test", 0.0, 1, 1)
    ideal_frontiers = [(ideal_option,) for _ in range(8)]
    ideal_allocation = AllocationTrace(np.zeros(8, np.int64), 8 * ideal_cost, 8.0)
    _, ideal_experts = _allocation_rows(
        group, "q3_ideal_logical_256_byte_plane_ceiling", ideal,
        ideal_frontiers, features, exact, base, np.full(8, 1 / 8), ideal_cost,
        "pooled_router_square_multi_budget_column_generated", ideal_allocation,
        .01, {"coordinate_sweeps": 8, "local_passes": 2,
              "state_comparisons": 100, "diagonal_dp_tables": 8, "diagonal_dp_state_updates": 1234,
              "frontier_wall_seconds": .006, "allocation_wall_seconds": .004, "refinement_rounds": 1}, config,
    )
    assert all(len(expert["selected_states_blob"]) == 512 for expert in ideal_experts)
    assert all(expert["selected_replica"] == -1 for expert in ideal_experts)
    assert all(expert["selected_unique_pages"] is None for expert in ideal_experts)
