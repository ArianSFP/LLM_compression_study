from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from oracle_study.interaction_field import JointInteractionFactor
from oracle_study.neuron_selector import factorized_unit_outputs, unit_score_metadata


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_interaction_field_study.py"
SPEC = importlib.util.spec_from_file_location("run_interaction_field_study", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _tiny_config():
    return {
        "tail_ranks": [4],
        "exact_proxy_tail_ranks": [0, 4],
        "geometry_factor_families": [
            "joint_eigh", "joint_pivoted_cholesky", "exact_proxy_plus_eigh_tail",
        ],
        "quantized_factor_specs": [
            {
                "source": "joint_eigh", "tail_rank": 4,
                "encoding": "int4_per_row", "hadamard": True,
            },
        ],
    }


def test_factor_fit_serializes_physical_quantized_payload_and_static_grid():
    rng = np.random.default_rng(13)
    output, units = 12, 512
    down2 = rng.normal(size=(output, units))
    down4 = down2 + 0.1 * rng.normal(size=(output, units))
    q2 = (np.empty((0,)), np.empty((0,)), down2)
    q4 = (np.empty((0,)), np.empty((0,)), down4)
    proxy = rng.normal(size=(output, 4))
    arrays, entries = runner._fit_cell_factors(
        q2, q4, proxy, 0.25, _tiny_config(), layer=0, expert=7,
    )
    assert len(entries) == 5
    assert len({entry["factor_config_id"] for entry in entries}) == 5
    quantized = [entry for entry in entries if entry["quantized_deployable_sidecar"]]
    assert len(quantized) == 1
    entry = quantized[0]
    assert entry["storage_kind"] == "physically_packed_signed_rows"
    physical = sum(
        np.asarray(arrays[entry[key]]).nbytes
        for key in ("packed_codes_key", "row_scales_key", "global_scale_key")
    )
    assert physical == entry["factor_payload_bytes"]
    factor = runner._load_factor(arrays, entry)
    assert factor.l4.shape == (units, 4)
    assert factor.l2.shape == (units, 4)
    assert factor.payload_bytes == physical


def test_factor_ids_distinguish_exact_rank_and_hadamard_encoding():
    assert runner._factor_id("joint_eigh", 16, 0, "fp32") == (
        "joint_eigh_tail16_exact0_fp32"
    )
    assert runner._factor_id(
        "exact_proxy_plus_eigh_tail", 16, 4, "int4_per_row_hadamard",
    ) == "exact_proxy_plus_eigh_tail_tail16_exact4_int4_per_row_hadamard"


def test_evaluate_factor_separates_two_seed_and_four_seed_compute_ledgers():
    rng = np.random.default_rng(29)
    input_size, output_size, rank = 4, 6, 4
    gate2 = rng.normal(size=(512, input_size))
    up2 = rng.normal(size=(512, input_size))
    down2 = rng.normal(size=(output_size, 512))
    gate4 = gate2 + 0.05 * rng.normal(size=gate2.shape)
    up4 = up2 + 0.05 * rng.normal(size=up2.shape)
    down4 = down2 + 0.05 * rng.normal(size=down2.shape)
    activation = rng.normal(size=input_size)
    outputs = factorized_unit_outputs(
        (gate2, up2, down2), (gate4, up4, down4), activation,
    )
    proxy = rng.normal(size=(output_size, 2))
    abc = unit_score_metadata(down2, down4, proxy=proxy, beta=0.1)
    factor = JointInteractionFactor(
        l4=rng.normal(scale=0.01, size=(512, rank)),
        l2=rng.normal(scale=0.01, size=(512, rank)),
        method="synthetic", tail_rank=rank, exact_rank=0, encoding="fp32",
        storage_bytes=2 * 512 * rank * 4,
    )
    factor_id = "synthetic_tail4_exact0_fp32"
    entry = {
        "factor_config_id": factor_id, "factor_family": "synthetic",
        "tail_rank": rank, "exact_rank": 0, "total_rank": rank,
        "encoding": "fp32", "hadamard_rotated": False,
        "geometry_ceiling": True, "quantized_deployable_sidecar": False,
        "factor_payload_bytes": factor.payload_bytes, "self_safe_global_scale": 1.0,
    }
    metadata = {
        "capture_source": "synthetic", "evaluation_split": "validation",
        "request_id": "request", "position": 0, "layer": 0, "expert_id": 7,
    }
    identity = tuple(metadata[name] for name in runner.IDENTITY)
    baselines = {
        identity + (6, name): 0.9
        for name in ("exact_hybrid", "coherent_exact_set", "pr9_independent")
    }
    config = {
        "page_budgets": [6], "coordinate_sweeps": 1,
        "relaxation_iterations": 2, "relaxation_tolerance": 1e-8,
        "local_shortlist": 2, "local_swap_units": 2, "local_max_passes": 1,
        "solver_control_factor_ids": [factor_id],
    }

    runner._EVALUATION_CONTEXT = {
        "cells": ({
            "q2": (gate2, up2, down2), "q4": (gate4, up4, down4),
            "abc": abc, "proxy": proxy, "beta": 0.1,
            "factors": ((entry, factor),),
            "observations": ((activation, metadata),),
        },),
        "baselines": baselines,
        "config": config,
    }
    try:
        cell_index, observation_index, rows = runner._evaluate_observation_task((0, 0))
    finally:
        runner._EVALUATION_CONTEXT = None
    assert (cell_index, observation_index) == (0, 0)
    assert {row["solver"] for row in rows} == {
        "compressed_residual_forward", "continuous_relaxation_round",
        "four_seed_coordinate", "two_seed_coordinate_plus_local_repair",
        "four_seed_coordinate_plus_local_repair",
    }
    cheap = next(
        row for row in rows if row["solver"] == "two_seed_coordinate_plus_local_repair"
    )
    broad = next(
        row for row in rows if row["solver"] == "four_seed_coordinate_plus_local_repair"
    )
    assert cheap["selector_forward_compute_macs"] == 0
    assert cheap["selector_relaxation_setup_compute_macs"] == 0
    assert cheap["selector_relaxation_iterations_compute_macs"] == 0
    assert cheap["forward_candidate_evaluations"] == 0
    assert cheap["relaxation_iterations"] == 0
    assert broad["selector_compute_macs"] > cheap["selector_compute_macs"]
    for row in rows:
        assert row["selector_compute_macs"] == sum(
            row[f"selector_{name}_compute_macs"] for name in runner.COMPUTE_COMPONENTS
        )
