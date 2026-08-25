from __future__ import annotations

import numpy as np
import pandas as pd

from oracle_study.d1_experiment_a_diagnostic_panel import (
    FIXED_D1_POLICY,
    PR13_POLICY,
    bf16_ulp_analysis,
    cohort_policy_comparisons,
    immediate_d1_severity,
    interpolation_turning_points,
    page_direction_norm_explanation,
    route_mode_decomposition,
)


TOKEN_KEYS = ["injection_layer", "group", "request_id", "position"]
MODES = ("live", "frozen_set_live_weights", "fully_frozen")
PAIRS = ((360, 384), (725, 749))


def _candidate_rows() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = pd.DataFrame(
        [
            {
                "injection_layer": 0,
                "group": token,
                "request_id": f"request-{token}",
                "position": token + 1,
                "panel_index": token,
                "selection_class": (
                    "fixed_vs_pr13_prevented" if token == 0 else "quiet_control"
                ),
                "selection_detail": "synthetic",
            }
            for token in range(2)
        ]
    )
    quality = []
    routes = []

    def add_candidate(
        token: int,
        candidate_id: str,
        candidate_kind: str,
        policy: str,
        rate: int,
        live_kl: float,
        norm: float,
        local: float,
        pages: int,
        crossed: bool,
        mass: float,
        interpolation_kind: str = "none",
        interpolation_lambda: float = -1.0,
        interpolation_scale: float = 1.0,
        interpolation_low_rate: int = -1,
        interpolation_high_rate: int = -1,
    ) -> None:
        identity = {
            "injection_layer": 0,
            "group": token,
            "request_id": f"request-{token}",
            "position": token + 1,
        }
        for mode, reduction in (
            ("live", 0.0),
            ("frozen_set_live_weights", 0.02),
            ("fully_frozen", 0.03),
        ):
            quality.append(
                {
                    **identity,
                    "candidate_id": candidate_id,
                    "candidate_kind": candidate_kind,
                    "policy": policy,
                    "rate_pages_per_expert": rate,
                    "route_mode": mode,
                    "logit_kl": live_kl - reduction,
                    "delta_nll": live_kl / 2 - reduction,
                    "final_hidden_mse": live_kl / 10,
                    "post_token_full_cache_mse": live_kl / 100,
                    "live_selected_local_qenergy_damage": local,
                    "selected_group_pages": pages,
                    "planned_delta_l2": norm,
                    "bf16_effective_delta_l2": norm * 0.99,
                    "bf16_effective_to_planned_norm_ratio": 0.99,
                    "bf16_quantization_error_mse": 1e-8,
                    "bf16_quantization_error_max_abs": 1e-4,
                    "bf16_quantization_error_ulp_mean": 0.2,
                    "bf16_quantization_error_ulp_p95": 0.45,
                    "bf16_quantization_error_ulp_max": 0.5,
                    "planned_below_half_ulp_fraction": 0.1,
                    "planned_nonzero_coordinates": 2048,
                    "bf16_changed_coordinates": 2000,
                    "bf16_collapsed_nonzero_coordinates": 48,
                    "bf16_collapsed_nonzero_fraction": 48 / 2048,
                    "interpolation_kind": interpolation_kind,
                    "interpolation_lambda": interpolation_lambda,
                    "interpolation_scale": interpolation_scale,
                    "interpolation_low_rate": interpolation_low_rate,
                    "interpolation_high_rate": interpolation_high_rate,
                }
            )
            routes.append(
                {
                    **identity,
                    "candidate_id": candidate_id,
                    "route_mode": mode,
                    "distance_from_injection": 1,
                    "route_membership_changed": crossed,
                    "membership_pairs_changed": int(crossed),
                    "router_mass_churn": mass,
                    "left_reference_routing_mass": mass if crossed else 0.0,
                    "entered_candidate_routing_mass": mass if crossed else 0.0,
                    "baseline_rank8_rank9_margin": 0.1,
                    "candidate_rank8_rank9_margin": 0.1 - mass,
                    "baseline_set_margin_under_candidate_logits": (
                        -0.01 if crossed else 0.05
                    ),
                    "centered_router_logit_mse": mass**2 + 1e-8,
                    "centered_router_logit_max_abs": mass + 1e-4,
                    "hidden_mse": live_kl / 20,
                }
            )

    for token in range(2):
        for rate in (360, 384, 725, 749):
            pr13_cross = token == 0 and rate == 360
            fixed_cross = token == 1 and rate == 384
            add_candidate(
                token,
                f"rate_{rate}__{PR13_POLICY}",
                "allocation_policy",
                PR13_POLICY,
                rate,
                0.10 + token * 0.01,
                1.2,
                0.3,
                rate * 8,
                pr13_cross,
                0.05 if pr13_cross else 0.01,
            )
            add_candidate(
                token,
                f"rate_{rate}__{FIXED_D1_POLICY}",
                "allocation_policy",
                FIXED_D1_POLICY,
                rate,
                0.08 + token * 0.02,
                1.0 if rate in (360, 725) else 2.0,
                0.2 if rate in (360, 725) else 0.1,
                rate * 8 - 1,
                fixed_cross,
                0.04 if fixed_cross else 0.005,
            )

        for low, high in PAIRS:
            kl_path = [0.10, 0.14, 0.09, 0.13, 0.12]
            for index, lam in enumerate((0.0, 0.25, 0.5, 0.75, 1.0)):
                norm = float(np.hypot(1.0 - lam, 2.0 * lam))
                candidate_id = (
                    f"fixed_d1_path_{low}_{high}__lambda_{str(lam).replace('.', 'p')}"
                )
                if lam in (0.0, 1.0):
                    candidate_id = f"fixed_d1_path_{low}_{high}__lambda_{int(lam)}"
                add_candidate(
                    token,
                    candidate_id,
                    "interpolation_control",
                    candidate_id,
                    high,
                    kl_path[index] + token * 0.001,
                    norm,
                    0.2 - 0.02 * index,
                    -1,
                    lam >= 0.75,
                    0.01 + 0.02 * lam,
                    "linear",
                    lam,
                    1.0,
                    low,
                    high,
                )
            candidate_id = f"fixed_d1_path_{low}_{high}__high_rescaled_to_low_norm"
            add_candidate(
                token,
                candidate_id,
                "interpolation_control",
                candidate_id,
                high,
                0.11 + token * 0.001,
                1.0,
                0.13,
                -1,
                True,
                0.025,
                "high_rescaled_to_low_norm",
                1.0,
                0.5,
                low,
                high,
            )
    return pd.DataFrame(quality), pd.DataFrame(routes), panel


def test_route_mode_decomposition_closes_and_separates_membership_from_weights() -> (
    None
):
    quality, _, panel = _candidate_rows()

    rows, summary = route_mode_decomposition(quality, panel)

    assert np.allclose(rows["membership_execution_component_kl"], 0.02)
    assert np.allclose(rows["router_weight_component_kl"], 0.01)
    assert np.allclose(rows["decomposition_closure_error"], 0.0)
    assert not summary.empty
    assert not summary.isnull().any().any()


def test_immediate_d1_analysis_treats_terminal_kl_only_as_outcome() -> None:
    quality, routes, panel = _candidate_rows()

    severity, correlations, summary = immediate_d1_severity(quality, routes, panel)

    assert len(severity) == len(quality) // 3
    assert (
        severity.loc[severity["route_membership_changed"], "boundary_violation_depth"]
        > 0
    ).all()
    assert correlations["terminal_kl_used_as_outcome_only"].all()
    assert not correlations.isnull().any().any()
    assert not summary.empty


def test_interpolation_turning_and_direction_norm_decomposition_are_deterministic() -> (
    None
):
    quality, routes, panel = _candidate_rows()
    severity, _, _ = immediate_d1_severity(quality, routes, panel)

    points, turning, interpolation_summary = interpolation_turning_points(severity)
    direction, direction_summary = page_direction_norm_explanation(severity)

    assert len(points) == 2 * 2 * 6
    assert len(turning) == 2 * 2
    assert set(turning["observed_minimum_kl_lambda"]) == {0.5}
    assert (turning["terminal_kl_discrete_turning_points"] > 0).all()
    assert not interpolation_summary.isnull().any().any()
    assert len(direction) == 2 * 2
    assert np.allclose(direction["low_high_delta_cosine"], 0.0, atol=1e-12)
    assert np.allclose(direction["rescaled_to_low_planned_norm_relative_error"], 0.0)
    assert np.allclose(direction["kl_component_closure_error"], 0.0)
    assert not direction_summary.isnull().any().any()


def test_bf16_and_fixed_vs_pr13_cohort_tables_are_complete() -> None:
    quality, routes, panel = _candidate_rows()
    decomposition, _ = route_mode_decomposition(quality, panel)
    severity, _, _ = immediate_d1_severity(quality, routes, panel)

    bf16, bf16_summary, association = bf16_ulp_analysis(quality, panel)
    cohort, cohort_summary = cohort_policy_comparisons(severity, decomposition)

    assert len(bf16) == len(quality) // 3
    assert bf16["any_bf16_collapse"].all()
    assert not bf16_summary.isnull().any().any()
    assert association["terminal_kl_used_as_outcome_only"].all()
    assert len(cohort) == 2 * 4
    assert "prevented" in set(cohort["crossing_event"])
    assert not cohort_summary.isnull().any().any()
