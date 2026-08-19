from __future__ import annotations

import pandas as pd
import pytest

from oracle_study.neuron_selector_analysis import choose_validation_promotions, validate_promotion_payload


def config() -> dict:
    return {
        "layers": [0, 4, 20, 39],
        "candidate_applied_units": 192,
        "promotion_policy": {
            "minimum_validation_invocations_per_layer": 2,
            "minimum_validation_requests_per_layer": 2,
            "factorized_max_median_gap_to_tile_at_1bpw": 0.01,
            "factorized_max_p10_gap_to_tile_at_1bpw": 0.015,
            "direct_predictor_min_median_recovery_at_1bpw": 0.95,
            "direct_predictor_min_p10_recovery_at_1bpw": 0.90,
            "candidate_min_median_utility_retention": 0.95,
            "candidate_min_p10_utility_retention": 0.90,
            "candidate_min_median_recovery": 0.92,
            "candidate_min_p10_recovery": 0.85,
            "candidate_max_units": 256,
            "candidate_max_overfetch": 4 / 3,
            "max_selector_metadata_bpw": 0.35,
        },
        "pr7_comparators": {
            "validation_tile_frontier": {
                "selector": "exact_dynamic_tile_marginal_refresh_1",
                "tile_shape": "64x16",
            }
        },
    }


def identities() -> list[dict]:
    result = []
    for layer in (0, 4, 20, 39):
        for index in range(2):
            result.append({
                "capture_source": "exact_checkpoint",
                "evaluation_split": "validation",
                "request_id": f"r{index}",
                "position": index,
                "layer": layer,
                "expert_id": 10 + layer,
            })
    return result


def response_rows(*, upper: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    direct, candidate = [], []
    variant = "per_expert_joint_gate_up_upper_bound" if upper else "layer_shared_joint_gate_up"
    for row in identities():
        common = {
            **row,
            "response_config_id": f"l{row['layer']}__cfg_{'upper' if upper else 'shared'}",
            "training_cohort": "combined_exact_and_cross_train",
            "basis_variant": variant,
            "rank": 32,
            "synthesis_encoding": "fp16",
            "selector_metadata_bpw": 0.2,
            "storage_multiplier": 1.52,
            "selector_compute_macs": 1000,
        }
        direct.append({
            **common,
            "unit_count": 256,
            "physical_budget_bpw": 1.0,
            "recovery": 0.955,
            "utility_weighted_recall": 0.97,
        })
        candidate.append({
            **common,
            "candidate_units": 256,
            "applied_units": 192,
            "candidate_overfetch": 4 / 3,
            "recovery": 0.93,
            "oracle_utility_retention": 0.97,
            "oracle_support_recall": 0.91,
        })
    return pd.DataFrame(direct), pd.DataFrame(candidate)


def test_validation_selection_promotes_shared_predictor_and_factorized_oracle() -> None:
    factorized = pd.DataFrame([
        {**row, "action_family": "factorized_gate_up_down_unit_states", "physical_budget_bpw": 1.0, "recovery": 0.955}
        for row in identities()
    ])
    tile = pd.DataFrame([
        {
            **row, "selector": "exact_dynamic_tile_marginal_refresh_1", "tile_shape": "64x16",
            "physical_budget_bpw": 1.0, "recovery": 0.96,
        }
        for row in identities()
    ])
    shared_direct, shared_candidate = response_rows()
    upper_direct, upper_candidate = response_rows(upper=True)
    payload, tables = choose_validation_promotions(
        config=config(), config_sha256="a" * 64, fit_bundle_sha256="b" * 64,
        factorized=factorized,
        response=pd.concat((shared_direct, upper_direct), ignore_index=True),
        candidate=pd.concat((shared_candidate, upper_candidate), ignore_index=True),
        tile_comparator=tile,
        evidence_sha256={"x": "c" * 64},
    )
    assert payload["promote_factorized_oracle"] is True
    assert payload["response_predictor"]["status"] == "promote"
    promoted = payload["response_predictor"]["promoted"][0]
    assert promoted["basis_variant"] == "layer_shared_joint_gate_up"
    assert len(payload["promoted_response_config_ids"]) == 4
    assert payload["test_rows_consulted_for_selection"] is False
    assert payload["conditional_followup"]["activation_topk"] is True
    assert len(tables["factorized_paired_validation"]) == 8
    selected, factorized_selected = validate_promotion_payload(
        payload, config(), config_sha256="a" * 64, fit_bundle_sha256="b" * 64,
    )
    assert factorized_selected is True
    assert selected == set(payload["promoted_response_config_ids"])
    payload["response_predictor"]["all_validation_candidates"][0]["passes_direct"] = False
    with pytest.raises(RuntimeError, match="stored response-predictor gate"):
        validate_promotion_payload(
            payload, config(), config_sha256="a" * 64, fit_bundle_sha256="b" * 64,
        )


def test_selection_rejects_any_test_row() -> None:
    factorized = pd.DataFrame([
        {**identities()[0], "action_family": "factorized_gate_up_down_unit_states", "physical_budget_bpw": 1.0, "recovery": 0.95}
    ])
    factorized.loc[0, "evaluation_split"] = "test"
    with pytest.raises(RuntimeError, match="non-validation"):
        choose_validation_promotions(
            config=config(), config_sha256="a" * 64, fit_bundle_sha256="b" * 64,
            factorized=factorized, response=pd.DataFrame(), candidate=pd.DataFrame(),
            tile_comparator=pd.DataFrame(), evidence_sha256={},
        )
