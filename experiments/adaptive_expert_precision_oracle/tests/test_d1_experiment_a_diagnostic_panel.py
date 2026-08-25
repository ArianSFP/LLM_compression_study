from __future__ import annotations

import json

import numpy as np
import pandas as pd

from oracle_study.d1_decode_tail import FIXED_D1_POLICY, PR13_POLICY
from run_d1_experiment_a_diagnostic_panel import (
    bf16_injection_accounting,
    detailed_route_metrics,
    interpolation_deltas,
    select_diagnostic_panel,
)


LAYERS = (0, 1, 4, 6, 12, 23)
RATES = (360, 384, 725, 749)


def _synthetic_source() -> tuple[pd.DataFrame, pd.DataFrame]:
    quality = []
    propagation = []
    for layer_index, layer in enumerate(LAYERS):
        for group in range(12):
            request_id = f"request-{group // 4}"
            position = group + 1
            for rate in RATES:
                pr13_kl = 0.001 + layer * 1e-6 + group * 1e-7
                fixed_kl = pr13_kl + 1e-6
                local = 2.0 - rate / 1000.0
                pr13_cross = False
                fixed_cross = False

                # Four strongest adjacent-rate reversals, one in each of the
                # first four layers.  The high rate improves local damage but
                # regresses KL.
                if group == 2 and layer_index < 4 and rate == 384:
                    fixed_kl += 0.1 + layer_index * 0.01
                # Four PR13 crossings prevented by fixed D1.
                if group == 3 and layer_index < 4 and rate == 360:
                    pr13_cross = True
                    fixed_kl = pr13_kl - (0.05 + layer_index * 0.01)
                # Four crossings introduced by fixed D1.
                if group == 4 and layer_index < 4 and rate == 725:
                    fixed_cross = True
                    fixed_kl = pr13_kl + 0.04 + layer_index * 0.01

                for policy, logit_kl, crossed in (
                    (PR13_POLICY, pr13_kl, pr13_cross),
                    (FIXED_D1_POLICY, fixed_kl, fixed_cross),
                ):
                    quality.append(
                        {
                            "injection_layer": layer,
                            "rate_pages_per_expert": rate,
                            "group": group,
                            "request_id": request_id,
                            "position": position,
                            "policy": policy,
                            "route_mode": "live",
                            "logit_kl": logit_kl,
                            "live_selected_local_qenergy_damage": local,
                        }
                    )
                    propagation.append(
                        {
                            "injection_layer": layer,
                            "rate_pages_per_expert": rate,
                            "group": group,
                            "request_id": request_id,
                            "position": position,
                            "policy": policy,
                            "route_mode": "live",
                            "distance_from_injection": 1,
                            "route_membership_change_fraction": float(crossed),
                        }
                    )
    return pd.DataFrame(quality), pd.DataFrame(propagation)


def test_panel_is_deterministic_unique_and_reserves_two_quiet_controls_per_layer() -> (
    None
):
    quality, propagation = _synthetic_source()

    first = select_diagnostic_panel(quality, propagation)
    second = select_diagnostic_panel(
        quality.sample(frac=1.0, random_state=17),
        propagation.sample(frac=1.0, random_state=29),
    )

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 24
    assert not first.duplicated(
        ["injection_layer", "group", "request_id", "position"]
    ).any()
    assert first["selection_class"].value_counts().to_dict() == {
        "quiet_control": 12,
        "adjacent_rate_reversal": 4,
        "fixed_vs_pr13_prevented": 4,
        "fixed_vs_pr13_introduced": 4,
    }
    quiet = first[first["selection_class"].eq("quiet_control")]
    assert quiet.groupby("injection_layer").size().to_dict() == {
        layer: 2 for layer in LAYERS
    }


def test_detailed_route_metrics_retains_swap_ids_scores_margins_and_continuous_error() -> (
    None
):
    reference_logits = np.asarray([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], np.float32)
    candidate_logits = np.asarray([10, 9, 8, 7, 6, 5, 4, 1, 3.5, 2], np.float32)
    reference_ids = np.arange(8, dtype=np.int64)
    candidate_ids = np.asarray([0, 1, 2, 3, 4, 5, 6, 8], np.int64)
    reference_scores = np.asarray([8, 7, 6, 5, 4, 3, 2, 1], np.float32)
    candidate_scores = np.asarray([8, 7, 6, 5, 4, 3, 2, 1.5], np.float32)

    metrics = detailed_route_metrics(
        reference_logits,
        candidate_logits,
        reference_ids,
        candidate_ids,
        reference_scores,
        candidate_scores,
        np.asarray([0.0, 1.0]),
        np.asarray([1.0, 1.0]),
    )

    assert json.loads(metrics["left_expert_ids_json"]) == [7]
    assert json.loads(metrics["entered_expert_ids_json"]) == [8]
    assert json.loads(metrics["baseline_top8_ids_json"]) == list(range(8))
    assert json.loads(metrics["candidate_top8_ids_json"]) == [0, 1, 2, 3, 4, 5, 6, 8]
    assert metrics["membership_pairs_changed"] == 1
    assert metrics["route_membership_changed"]
    assert not metrics["route_top1_changed"]
    assert metrics["baseline_rank8_rank9_margin"] == 1.0
    assert metrics["candidate_rank8_rank9_margin"] == 1.5
    assert metrics["baseline_set_margin_under_candidate_logits"] == -2.5
    assert metrics["candidate_set_margin_under_baseline_logits"] == -1.0
    assert metrics["router_mass_churn"] > 0.0
    assert metrics["centered_router_logit_mse"] > 0.0
    assert metrics["hidden_mse"] == 0.5


def test_interpolation_controls_include_endpoints_and_norm_matched_high_delta() -> None:
    low = np.asarray([3.0, 4.0], np.float32)
    high = np.asarray([0.0, 10.0], np.float32)

    controls = interpolation_deltas(low, high)

    assert len(controls) == 6
    np.testing.assert_array_equal(controls[0][2], low)
    np.testing.assert_array_equal(controls[4][2], high)
    kind, lam, rescaled, scale = controls[5]
    assert kind == "high_rescaled_to_low_norm"
    assert lam == 1.0
    assert scale == 0.5
    assert np.isclose(np.linalg.norm(rescaled), np.linalg.norm(low))


def test_bf16_accounting_detects_sub_ulp_collapse() -> None:
    reference = np.ones(4, np.float32)
    planned = np.asarray([0.0, 0.001, 0.01, -0.01], np.float32)

    metrics = bf16_injection_accounting(reference, planned)

    assert metrics["planned_nonzero_coordinates"] == 3
    assert metrics["bf16_collapsed_nonzero_coordinates"] == 1
    assert metrics["bf16_changed_coordinates"] == 2
    assert metrics["bf16_quantization_error_max_abs"] > 0.0
    assert 0.0 < metrics["bf16_collapsed_nonzero_fraction"] < 1.0
