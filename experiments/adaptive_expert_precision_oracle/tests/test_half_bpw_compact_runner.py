from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import run_half_bpw_compact_field as runner
from oracle_study.interaction_field import (
    JointInteractionFactor,
    encode_interaction_factor,
)
from oracle_study.split_interaction_field import (
    SPLIT_STATE_DOWN_HIGH,
    SPLIT_STATE_HIDDEN_INDEX,
    SplitProjectionResponses,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/qwen36_mxfp4_half_bpw_compact_field.json"


def test_frozen_compact_contract_matches_pr13_payload() -> None:
    config = json.loads(CONFIG.read_text())
    runner._validate_config(config)
    assert config["group_page_budget"] == 8 * 384
    assert config["primary_factor_payload_bytes_per_expert"] == 6148
    assert config["abc_metadata_bytes_per_expert"] + 6148 == 9232
    assert 8 * 9232 == 73856
    assert config["maximum_primary_group_sidecar_bytes"] == 73856
    assert config["maximum_primary_median_selector_macs"] == 79712256
    assert config["compact_seed"] == "lagrangian_primary_with_dp_control"
    assert config["fit_split"] == "train"
    assert config["selection_split"] == "validation"


@pytest.mark.parametrize(
    ("name", "value"),
    (("primary_rank", 9), ("trajectory_exclusion_units", 63),
     ("primary_factor_payload_bytes_per_expert", 6149)),
)
def test_contract_fails_closed(name: str, value: int) -> None:
    config = json.loads(CONFIG.read_text())
    config[name] = value
    with pytest.raises(ValueError, match=name):
        runner._validate_config(config)


def test_unit_residual_matches_state_definition() -> None:
    hidden = np.arange(12, dtype=np.float64).reshape(3, 4) / 7.0
    down2 = np.arange(15, dtype=np.float64).reshape(3, 5) / 11.0
    down4 = down2 + 0.3
    response = SplitProjectionResponses(hidden, down2, down4)
    for unit in range(3):
        for state in range(8):
            h = hidden[unit, int(SPLIT_STATE_HIDDEN_INDEX[state])]
            down = down4 if bool(SPLIT_STATE_DOWN_HIGH[state]) else down2
            expected = hidden[unit, 3] * down4[unit] - h * down[unit]
            assert np.array_equal(
                runner._unit_residual(response, unit, state), expected,
            )


def test_encoded_factor_payload_is_exact_and_loads_common_rank() -> None:
    rng = np.random.default_rng(4)
    factor8 = JointInteractionFactor(
        rng.normal(size=(512, 8)).astype(np.float32),
        rng.normal(size=(512, 8)).astype(np.float32),
        "shared", 8,
    )
    factor16 = JointInteractionFactor(
        rng.normal(size=(512, 16)).astype(np.float32),
        rng.normal(size=(512, 16)).astype(np.float32),
        "shared", 16,
    )
    encoded8 = encode_interaction_factor(
        factor8, "int4_per_row", hadamard_rotate=True,
    )
    encoded16 = encode_interaction_factor(
        factor16, "int4_per_row", hadamard_rotate=True,
    )
    arrays = {
        "rank8_packed_codes": encoded8.packed_codes[None, :],
        "rank8_row_scales": encoded8.row_scales[None, :],
        "rank8_global_scales": np.ones(1, np.float32),
        "rank16_packed_codes": encoded16.packed_codes[None, :],
        "rank16_row_scales": encoded16.row_scales[None, :],
        "rank16_global_scales": np.ones(1, np.float32),
        "rank8_fp32_l4": factor8.l4[None, :],
        "rank8_fp32_l2": factor8.l2[None, :],
    }
    factors = runner._load_factors(arrays, 0)
    assert factors[runner.POLICIES[1]].rank == 8
    assert factors[runner.POLICIES[1]].payload_bytes == 6148
    assert factors[runner.POLICIES[2]].payload_bytes == 6148
    assert factors[runner.POLICIES[3]].payload_bytes == 32768
    assert factors[runner.POLICIES[4]].rank == 16
    assert factors[runner.POLICIES[4]].payload_bytes == 10244
    assert factors[runner.POLICIES[1]] is factors[runner.POLICIES[2]]
