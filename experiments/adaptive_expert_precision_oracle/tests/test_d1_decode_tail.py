from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from oracle_study.d1_decode_tail import (
    COMPANION_D1_POLICY,
    DECODE_POLICIES,
    D1_PREFIX,
    DecodeOracleCell,
    FIXED_D1_POLICY,
    HISTORICAL_PR13,
    LOCAL_POLICY,
    LOCAL_SOURCE,
    PR13_POLICY,
    TOKEN_ORACLE_POLICY,
    add_live_minus_frozen,
    assemble_decode_policy_bank,
    calibrate_fixed_d1_policy,
    classified_cache_metrics,
    current_token_quality_metrics,
    downstream_route_rows,
    expected_grid_counts,
    token_delta,
    validate_cached_decode_tail_config,
)


def _synthetic_cell(layer: int = 0, rate: int = 384) -> DecodeOracleCell:
    identity = [
        (group, f"request-{group // 10}", group + 1)
        for group in range(29)
    ]
    d1 = [f"{D1_PREFIX}eta_a", f"{D1_PREFIX}eta_b", f"{D1_PREFIX}eta_c"]
    metrics = []
    allocations = []
    expert_allocations = []
    for policy_index, policy in enumerate(
        [HISTORICAL_PR13, LOCAL_SOURCE, *d1]
    ):
        for group, request_id, position in identity:
            crossing = policy in (d1[0], d1[1])
            if policy == d1[1]:
                crossing = False
            metrics.append({
                "layer": layer,
                "rate_pages_per_expert": rate,
                "group": group,
                "request_id": request_id,
                "position": position,
                "policy": policy,
                "exact_d1_crossed": crossing,
                "membership_pairs_changed": int(crossing),
                "routing_mass_lost": 0.1 if crossing else 0.0,
                "local_qenergy_damage": 1.0 + policy_index,
            })
            allocations.append({
                "layer": layer,
                "rate_pages_per_expert": rate,
                "group": group,
                "request_id": request_id,
                "position": position,
                "policy": policy,
                "selected_group_pages": rate * 8 - policy_index,
            })
            for router_rank in range(1, 9):
                expert_allocations.append({
                    "layer": layer,
                    "rate_pages_per_expert": rate,
                    "group": group,
                    "request_id": request_id,
                    "position": position,
                    "policy": policy,
                    "router_rank": router_rank,
                    "expert_id": group * 8 + router_rank,
                    "selected_states": json.dumps([policy_index] * 512, separators=(",", ":")),
                })
    token_oracle = []
    for group, request_id, position in identity:
        source = d1[group % len(d1)]
        token_oracle.append({
            "layer": layer,
            "rate_pages_per_expert": rate,
            "group": group,
            "request_id": request_id,
            "position": position,
            "policy": source,
            "local_qenergy_damage": float(10 + group),
            "selected_group_pages": rate * 8 - group,
        })
    zeros = np.zeros((29, 2048), np.float32)
    deltas = {
        f"{HISTORICAL_PR13}__rate_{rate}": zeros + 1,
        f"{LOCAL_SOURCE}__rate_{rate}": zeros + 2,
        f"{d1[0]}__rate_{rate}": zeros + 3,
        f"{d1[1]}__rate_{rate}": zeros + 4,
        f"{d1[2]}__rate_{rate}": zeros + 5,
    }
    return DecodeOracleCell(
        layer=layer,
        rate=rate,
        directory=Path("/synthetic"),
        facts={"groups": 29},
        metrics=pd.DataFrame(metrics),
        allocations=pd.DataFrame(allocations),
        expert_allocations=pd.DataFrame(expert_allocations),
        token_oracle=pd.DataFrame(token_oracle),
        deltas=deltas,
    )


def _config() -> dict:
    return {
        "schema": "pr13_d1_cached_decode_tail_kl_config_v2",
        "injection_layers": [0, 1, 4, 6, 12, 23],
        "page_caps": [360, 384, 725, 749],
        "mechanism_page_caps": [384, 749],
        "matched_runtime_metadata_page_caps": [360, 725],
        "policies": list(DECODE_POLICIES),
        "route_modes": ["live", "fully_frozen"],
        "validation_request_ids": [
            "mxfp4-confirm-006", "mxfp4-confirm-007", "mxfp4-confirm-008",
        ],
        "position_mask": {
            "admitted_positions_per_request": {
                "mxfp4-confirm-006": 10,
                "mxfp4-confirm-007": 8,
                "mxfp4-confirm-008": 11,
            },
            "expected_groups_per_layer": 29,
            "minimum_exact_prefix_tokens": 1,
            "cross_position_delta_coupling": False,
        },
        "decode_execution": {
            "query_length": 1,
            "candidate_prefix_cache": "deep_clone_complete_exact_prefix",
            "terminal_metric_scope": "current_token_only",
            "retain_post_token_cache": True,
            "provisional_or_speculative_pass": False,
        },
        "d1_source_policy_by_rate": {
            str(rate): {
                "fixed": f"{D1_PREFIX}eta_a",
                "companion": f"{D1_PREFIX}eta_b",
            }
            for rate in (360, 384, 725, 749)
        },
        "selected_tree_sha256": "a" * 64,
        "live_reconstruction": {
            "candidate_delta": "complete_selected_states_reevaluated_on_live_cached_activation",
            "stored_slice_delta": "transfer_diagnostic_only",
            "route_set_match_required": True,
            "q4_routed_max_abs_atol": 0.125,
        },
        "hardware_execution_path": {
            "required_gpu_name_substring": "RTX PRO 6000",
            "minimum_gpu_memory_gib": 90,
        },
        "minimum_requests_before_quality_claim": 64,
    }


def _fake_cache() -> SimpleNamespace:
    return SimpleNamespace(layers=[
        SimpleNamespace(
            keys=torch.tensor([]),
            values=torch.tensor([]),
            conv_states={0: torch.arange(4, dtype=torch.float32)},
            recurrent_states={0: torch.arange(6, dtype=torch.float32)},
        ),
        SimpleNamespace(
            keys=torch.arange(8, dtype=torch.float32),
            values=torch.arange(8, dtype=torch.float32) + 1,
        ),
    ])


def test_cached_decode_config_rejects_sequence_coupling_and_legacy_inputs() -> None:
    config = _config()
    validate_cached_decode_tail_config(config)
    coupled = deepcopy(config)
    coupled["position_mask"]["cross_position_delta_coupling"] = True
    with np.testing.assert_raises_regex(ValueError, "coupling is forbidden"):
        validate_cached_decode_tail_config(coupled)
    legacy = deepcopy(config)
    legacy["existing_repair_cell_dirs"] = []
    with np.testing.assert_raises_regex(ValueError, "legacy full-sequence inputs"):
        validate_cached_decode_tail_config(legacy)


def test_decode_grid_counts_include_five_policies_and_isolated_positions() -> None:
    counts = expected_grid_counts(_config())
    assert counts["quality_rows"] == 6960
    assert counts["propagation_rows"] == 218080
    assert counts["logical_zero_dose_rows"] == 1392


def test_decode_calibration_uses_crossings_before_local_qenergy() -> None:
    cells = {
        (0, 384): _synthetic_cell(0, 384),
        (12, 384): _synthetic_cell(12, 384),
    }
    selected, evidence = calibrate_fixed_d1_policy(cells, [0, 12], [384])
    assert selected == {384: f"{D1_PREFIX}eta_b"}
    assert evidence[evidence["selected"]]["exact_crossings"].item() == 0


def test_decode_policy_bank_uses_per_token_oracle_without_request_delta() -> None:
    cell = _synthetic_cell()
    bank = assemble_decode_policy_bank(
        cell,
        fixed_source_policy=f"{D1_PREFIX}eta_b",
        companion_source_policy=f"{D1_PREFIX}eta_c",
    )
    assert tuple(bank.deltas) == DECODE_POLICIES
    np.testing.assert_array_equal(bank.deltas[PR13_POLICY], 1)
    np.testing.assert_array_equal(bank.deltas[LOCAL_POLICY], 2)
    np.testing.assert_array_equal(bank.deltas[FIXED_D1_POLICY], 4)
    np.testing.assert_array_equal(bank.deltas[COMPANION_D1_POLICY], 5)
    np.testing.assert_array_equal(bank.deltas[TOKEN_ORACLE_POLICY][0], 3)
    np.testing.assert_array_equal(bank.deltas[TOKEN_ORACLE_POLICY][1], 4)
    delta, local = token_delta(
        bank, TOKEN_ORACLE_POLICY,
        group=2, request_id="request-0", position=3,
    )
    assert delta.shape == (2048,)
    np.testing.assert_array_equal(delta, 5)
    assert local["selected_local_qenergy_damage"] == 12
    assert local["selected_group_pages"] == 384 * 8 - 2


def test_current_token_quality_and_route_metrics_have_no_sequence_average() -> None:
    reference_logits = np.zeros(7, np.float32)
    candidate_logits = reference_logits.copy()
    candidate_logits[1] = 0.5
    quality = current_token_quality_metrics(
        reference_logits,
        candidate_logits,
        2,
        np.zeros(4, np.float32),
        np.ones(4, np.float32),
    )
    assert quality["logit_kl"] > 0
    assert quality["delta_nll"] > 0
    assert quality["final_hidden_mse"] == 1
    reference_router = {}
    candidate_router = {}
    reference_hidden = {}
    candidate_hidden = {}
    for layer in range(1, 40):
        values = np.arange(10, 0, -1, dtype=np.float32)
        changed = values.copy()
        if layer == 2:
            changed[8] = 20
        reference_router[layer] = values
        candidate_router[layer] = changed
        reference_hidden[layer] = np.zeros(4, np.float32)
        candidate_hidden[layer] = np.ones(4, np.float32)
    rows, first = downstream_route_rows(
        reference_router, candidate_router,
        reference_hidden, candidate_hidden, 0,
    )
    assert len(rows) == 39
    assert first == 2
    assert rows[0]["hidden_mse"] == 1


def test_cache_classes_report_attention_conv_and_recurrent_commits() -> None:
    reference = _fake_cache()
    candidate = _fake_cache()
    exact = classified_cache_metrics(reference, candidate)
    assert exact["full_cache_bit_identical"]
    assert exact["attention_kv_tensors"] == 2
    assert exact["deltanet_conv_tensors"] == 1
    assert exact["deltanet_recurrent_tensors"] == 1
    candidate.layers[0].recurrent_states[0][0] += 1
    changed = classified_cache_metrics(reference, candidate)
    assert not changed["full_cache_bit_identical"]
    assert not changed["deltanet_recurrent_bit_identical"]
    assert changed["attention_kv_bit_identical"]


def test_live_minus_frozen_pairs_each_isolated_token() -> None:
    rows = []
    for mode, value in (("live", 3.0), ("fully_frozen", 1.25)):
        rows.append({
            "injection_layer": 4,
            "rate_pages_per_expert": 384,
            "group": 3,
            "request_id": "request",
            "position": 7,
            "policy": FIXED_D1_POLICY,
            "route_mode": mode,
            "logit_kl": value,
        })
    paired = add_live_minus_frozen(pd.DataFrame(rows))
    assert np.allclose(paired["live_minus_frozen_logit_kl"], 1.75)



def test_checked_in_cached_decode_tail_config_is_immutable_contract() -> None:
    experiment = Path(__file__).resolve().parents[1]
    config = json.loads((
        experiment / "configs/qwen36_mxfp4_d1_cached_decode_tail_kl_smoke_20260824_v2.json"
    ).read_text())
    validate_cached_decode_tail_config(config)
    assert expected_grid_counts(config)["quality_rows"] == 6960
