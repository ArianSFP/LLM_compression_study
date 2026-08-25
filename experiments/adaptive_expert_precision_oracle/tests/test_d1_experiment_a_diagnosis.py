from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from oracle_study.d1_experiment_a_diagnosis import (
    PR13_POLICY,
    adjacent_rate_nonmonotonicity,
    crossing_conditional_effects,
    deployable_d1_target_remedies,
    heavy_tail_summaries,
    source_adjacent_rate_geometry,
)


def _quality_row(
    *,
    rate: int,
    group: int,
    policy: str,
    route_mode: str = "live",
    logit_kl: float = 0.0,
    local: float = 1.0,
    pages: int = 10,
    first_crossing: int = -1,
) -> dict[str, object]:
    return {
        "rate_pages_per_expert": rate,
        "injection_layer": 0,
        "group": group,
        "request_id": f"request-{group}",
        "position": group + 1,
        "policy": policy,
        "route_mode": route_mode,
        "logit_kl": logit_kl,
        "live_minus_frozen_logit_kl": logit_kl / 2,
        "live_selected_local_qenergy_damage": local,
        "selected_group_pages": pages,
        "realized_injected_layer_output_mse": local / 2,
        "first_route_membership_change_layer": first_crossing,
    }


def _propagation_row(
    *,
    rate: int,
    group: int,
    policy: str,
    distance: int,
    crossed: float,
    churn: float,
) -> dict[str, object]:
    return {
        "rate_pages_per_expert": rate,
        "injection_layer": 0,
        "group": group,
        "request_id": f"request-{group}",
        "position": group + 1,
        "policy": policy,
        "route_mode": "live",
        "distance_from_injection": distance,
        "route_membership_change_fraction": crossed,
        "route_top1_change_fraction": crossed,
        "router_mass_churn": churn,
        "hidden_mse": churn / 10,
    }


def test_crossing_conditional_effects_distinguishes_prevented_and_introduced() -> None:
    candidate = "calibration_selected_fixed_d1"
    quality = pd.DataFrame(
        [
            _quality_row(
                rate=384, group=0, policy=PR13_POLICY, logit_kl=2, first_crossing=1
            ),
            _quality_row(rate=384, group=0, policy=candidate, logit_kl=1),
            _quality_row(rate=384, group=1, policy=PR13_POLICY, logit_kl=1),
            _quality_row(
                rate=384, group=1, policy=candidate, logit_kl=3, first_crossing=1
            ),
        ]
    )
    propagation = pd.DataFrame(
        [
            _propagation_row(
                rate=384, group=0, policy=PR13_POLICY, distance=1, crossed=1, churn=0.2
            ),
            _propagation_row(
                rate=384, group=0, policy=candidate, distance=1, crossed=0, churn=0.0
            ),
            _propagation_row(
                rate=384, group=1, policy=PR13_POLICY, distance=1, crossed=0, churn=0.0
            ),
            _propagation_row(
                rate=384, group=1, policy=candidate, distance=1, crossed=1, churn=0.3
            ),
        ]
    )

    rows, summary = crossing_conditional_effects(quality, propagation)

    assert rows["crossing_event"].tolist() == ["prevented", "introduced"]
    assert rows["logit_kl_minus_pr13"].tolist() == [-1, 2]
    aggregate = summary[summary["scope"].eq("all_layers")]
    assert set(aggregate["crossing_event"]) == {"prevented", "introduced"}


def test_adjacent_rate_flags_local_improvement_with_live_kl_regression() -> None:
    rows = []
    for mode, low_kl, high_kl in (("live", 1.0, 2.0), ("fully_frozen", 1.0, 0.5)):
        rows.append(
            _quality_row(
                rate=360,
                group=0,
                policy=PR13_POLICY,
                route_mode=mode,
                logit_kl=low_kl,
                local=2.0,
            )
        )
        rows.append(
            _quality_row(
                rate=384,
                group=0,
                policy=PR13_POLICY,
                route_mode=mode,
                logit_kl=high_kl,
                local=1.0,
            )
        )

    paired, summary = adjacent_rate_nonmonotonicity(pd.DataFrame(rows), ((360, 384),))

    live = paired[paired["route_mode"].eq("live")].iloc[0]
    frozen = paired[paired["route_mode"].eq("fully_frozen")].iloc[0]
    assert bool(live["local_improved_but_kl_worsened"])
    assert live["logit_kl_delta_high_minus_low"] == 1.0
    assert not bool(frozen["local_improved_but_kl_worsened"])
    live_summary = summary[
        summary["scope"].eq("all_layers") & summary["route_mode"].eq("live")
    ].iloc[0]
    assert live_summary["paradox_fraction"] == 1.0


def test_heavy_tail_reports_positive_damage_concentration_and_rank() -> None:
    crossing = pd.DataFrame(
        {
            "rate_pages_per_expert": [384] * 3,
            "injection_layer": [0] * 3,
            "group": [0, 1, 2],
            "request_id": ["r0", "r1", "r2"],
            "position": [1, 1, 1],
            "policy": ["candidate"] * 3,
            "crossing_event": ["both_safe"] * 3,
            "logit_kl_minus_pr13": [1.0, 2.0, 7.0],
        }
    )
    adjacent = pd.DataFrame(
        {
            "low_rate": [360],
            "high_rate": [384],
            "injection_layer": [0],
            "group": [0],
            "request_id": ["r0"],
            "position": [1],
            "policy": ["candidate"],
            "route_mode": ["live"],
            "logit_kl_delta_high_minus_low": [-1.0],
        }
    )

    summary, events = heavy_tail_summaries(crossing, adjacent, top_events=2)

    row = summary[
        summary["scope"].eq("all_layers")
        & summary["comparison_kind"].eq("policy_minus_pr13")
    ].iloc[0]
    assert np.isclose(row["top1_positive_share"], 0.7)
    selected = events[events["comparison_kind"].eq("policy_minus_pr13")]
    assert selected["damage"].tolist() == [7.0, 2.0]
    assert selected["damage_rank"].tolist() == [1, 2]


def test_source_geometry_detects_removed_bits_and_delta_rotation() -> None:
    identity = pd.DataFrame(
        {"group": [0], "request_id": ["r"], "position": [1], "layer": [0]}
    )
    low_state = np.zeros((1, 1, 512), np.int8)
    high_state = np.zeros((1, 1, 512), np.int8)
    low_state[0, 0, 0] = 3
    high_state[0, 0, 0] = 2  # remove projection bit 0
    high_state[0, 0, 1] = 4  # add projection bit 2
    low = SimpleNamespace(
        identity=identity,
        expert_ids=np.asarray([[17]]),
        selected_states={PR13_POLICY: low_state},
        deltas={PR13_POLICY: np.asarray([[1.0, 0.0]], np.float32)},
    )
    high = SimpleNamespace(
        identity=identity,
        expert_ids=np.asarray([[17]]),
        selected_states={PR13_POLICY: high_state},
        deltas={PR13_POLICY: np.asarray([[0.0, 1.0]], np.float32)},
    )

    experts, deltas, summary = source_adjacent_rate_geometry(
        {(0, 360): low, (0, 384): high}, ((360, 384),)
    )

    assert experts.iloc[0]["removed_projection_bits"] == 1
    assert experts.iloc[0]["added_projection_bits"] == 1
    assert not bool(experts.iloc[0]["expert_is_nested"])
    assert deltas.iloc[0]["adjacent_delta_cosine"] == 0.0
    assert np.isclose(deltas.iloc[0]["increment_norm"], np.sqrt(2.0))
    assert summary.iloc[0]["non_nested_expert_fraction"] == 1.0


def _remedy_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    policies = [PR13_POLICY, "candidate_a", "candidate_b"]
    quality_rows: list[dict[str, object]] = []
    propagation_rows: list[dict[str, object]] = []
    # Group 0: PR13 is safe, so strict repair must hold it; global may switch.
    # Group 1: PR13 is unsafe and two safe repairs tie on mass, then local wins.
    # Group 2: every policy crosses, so strict repair must hold PR13.
    specification = {
        0: {
            PR13_POLICY: (0.0, 0.30, 1.0, 10),
            "candidate_a": (0.0, 0.10, 0.5, 9),
            "candidate_b": (0.0, 0.20, 0.2, 8),
        },
        1: {
            PR13_POLICY: (1.0, 0.40, 1.0, 10),
            "candidate_a": (0.0, 0.10, 2.0, 9),
            "candidate_b": (0.0, 0.10, 0.5, 8),
        },
        2: {
            PR13_POLICY: (1.0, 0.40, 1.0, 10),
            "candidate_a": (1.0, 0.10, 0.5, 9),
            "candidate_b": (1.0, 0.20, 0.2, 8),
        },
    }
    for rate, page_scale in ((360, 8), (384, 10)):
        for group, policy_values in specification.items():
            for policy in policies:
                crossed, churn, local, pages = policy_values[policy]
                if rate == 360 and group == 0 and policy == "candidate_a":
                    churn = 0.01
                quality_rows.append(
                    _quality_row(
                        rate=rate,
                        group=group,
                        policy=policy,
                        logit_kl=float(group + pages / 100),
                        local=local,
                        pages=pages * page_scale,
                        first_crossing=(1 if crossed else -1),
                    )
                )
                propagation_rows.append(
                    _propagation_row(
                        rate=rate,
                        group=group,
                        policy=policy,
                        distance=1,
                        crossed=crossed,
                        churn=churn,
                    )
                )
    return pd.DataFrame(quality_rows), pd.DataFrame(propagation_rows)


def test_deployable_remedies_keep_safe_incumbent_and_require_strict_repair() -> None:
    quality, propagation = _remedy_fixture()

    rows, summary = deployable_d1_target_remedies(quality, propagation, ((360, 384),))

    same = rows[
        rows["comparison_kind"].eq("same_numeric_rate") & rows["selection_rate"].eq(360)
    ]
    repair = same[same["selector"].eq("strict_pr13_incumbent_repair")].set_index(
        "group"
    )
    assert repair.loc[0, "selected_policy"] == PR13_POLICY
    assert repair.loc[0, "selector_reason"] == "safe_incumbent_held"
    assert repair.loc[1, "selected_policy"] == "candidate_b"
    assert repair.loc[1, "selector_reason"] == "strictly_fewer_d1_crossings"
    assert repair.loc[2, "selected_policy"] == PR13_POLICY
    assert repair.loc[2, "selector_reason"] == "no_strict_d1_repair"
    assert not bool(repair["crossing_introduced"].any())

    global_rows = same[same["selector"].eq("global_hard_d1_local")].set_index("group")
    assert global_rows.loc[0, "selected_policy"] == "candidate_a"
    assert global_rows.loc[1, "selected_policy"] == "candidate_b"
    assert set(summary["scope"]) == {
        "all_layers_requests",
        "by_layer",
        "by_request",
        "by_layer_request",
    }


def test_deployable_remedies_emit_metadata_and_adjacent_cap_comparisons() -> None:
    quality, propagation = _remedy_fixture()

    rows, _ = deployable_d1_target_remedies(quality, propagation, ((360, 384),))

    metadata = rows[rows["comparison_kind"].eq("metadata_matched_low_vs_high_pr13")]
    assert set(metadata["selection_rate"]) == {360}
    assert set(metadata["selected_source_rate"]) == {360}
    assert set(metadata["reference_rate"]) == {384}
    cap = rows[rows["comparison_kind"].eq("adjacent_rate_early_stop_cap")]
    assert set(cap["selection_rate"]) == {384}
    assert set(cap["reference_rate"]) == {384}
    assert bool(cap["selected_lower_rate"].any())
    assert bool(
        (cap["selected_group_pages"] <= cap["reference_selected_group_pages"]).all()
    )


def test_deployable_remedy_selection_does_not_use_terminal_kl() -> None:
    quality, propagation = _remedy_fixture()
    original, _ = deployable_d1_target_remedies(quality, propagation, ((360, 384),))
    changed = quality.copy()
    changed["logit_kl"] = np.arange(len(changed), dtype=np.float64)[::-1] * 1000
    changed["live_minus_frozen_logit_kl"] = -changed["logit_kl"]

    rerun, _ = deployable_d1_target_remedies(changed, propagation, ((360, 384),))

    identity = [
        "comparison_kind",
        "selector",
        "selection_rate",
        "injection_layer",
        "group",
        "selected_source_rate",
        "selected_policy",
    ]
    assert original[identity].equals(rerun[identity])
    assert not bool(original["terminal_kl_used_for_selection"].any())
