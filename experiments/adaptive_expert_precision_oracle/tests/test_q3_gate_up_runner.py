import json
from pathlib import Path

import numpy as np

from oracle_study.average_rate_allocator import AllocationTrace, RateOption
from run_q3_gate_up_layout import (
    GROUP_IDENTITY, _allocation_rows, _layout_descriptor_bytes, _layouts,
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
    assert layouts["q3_physical_training_coselection_single"].replicas == 1
    assert layouts["q3_physical_training_coselection_replicated2"].replicas == 2


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
              "state_comparisons": 100, "refinement_rounds": 1}, config,
    )
    assert row["actual_group_quanta"] == 8 * cost
    expected = (
        row["combined_metadata_bytes_per_expert"] + cost * 256
    ) * 8 / config["expert_weights"]
    assert np.isclose(row["average_actual_total_bpw"], expected)
    assert np.isclose(row["average_allowed_total_bpw"], expected)
    assert len(experts) == 8
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
              "state_comparisons": 100, "refinement_rounds": 1}, config,
    )
    assert all(len(expert["selected_states_blob"]) == 512 for expert in ideal_experts)
    assert all(expert["selected_replica"] == -1 for expert in ideal_experts)
    assert all(expert["selected_unique_pages"] is None for expert in ideal_experts)
