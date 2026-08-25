from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oracle_study.d1_downstream_outcome_diagnosis import (
    downstream_outcome_diagnostics,
)


LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _fixtures() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = pd.DataFrame(
        [
            {
                "injection_layer": 4,
                "group": 7,
                "request_id": "request-0",
                "position": 11,
                "panel_index": 3,
                "selection_class": "adjacent_rate_reversal",
                "selection_detail": "synthetic",
            }
        ]
    )
    terminal_kl = (0.10, 0.20, 0.15, 0.25, 0.20)
    changed_by_lambda = (
        (False, False, False),
        (False, True, False),
        (False, False, True),
        (False, True, True),
        (False, False, False),
    )
    mass_by_lambda = (
        (0.0, 0.0, 0.0),
        (0.0, 0.10, 0.0),
        (0.0, 0.0, 0.20),
        (0.0, 0.15, 0.10),
        (0.0, 0.0, 0.0),
    )
    identity = {
        "injection_layer": 4,
        "group": 7,
        "request_id": "request-0",
        "position": 11,
    }
    quality = []
    routes = []
    for index, interpolation_lambda in enumerate(LAMBDAS):
        candidate_id = f"path-lambda-{index}"
        quality.append(
            {
                **identity,
                "candidate_id": candidate_id,
                "candidate_kind": "interpolation_control",
                "policy": candidate_id,
                "rate_pages_per_expert": 384,
                "route_mode": "live",
                "logit_kl": terminal_kl[index],
                "interpolation_kind": "linear",
                "interpolation_lambda": interpolation_lambda,
                "interpolation_low_rate": 360,
                "interpolation_high_rate": 384,
            }
        )
        for distance, (changed, mass) in enumerate(
            zip(changed_by_lambda[index], mass_by_lambda[index]),
            start=1,
        ):
            routes.append(
                {
                    **identity,
                    "candidate_id": candidate_id,
                    "route_mode": "live",
                    "distance_from_injection": distance,
                    "route_membership_changed": changed,
                    "membership_pairs_changed": int(changed),
                    "router_mass_churn": mass,
                    "centered_router_logit_mse": mass**2,
                    "hidden_mse": mass / 10.0,
                }
            )
    return pd.DataFrame(quality), pd.DataFrame(routes), panel


def test_downstream_outcome_rows_expose_delayed_threshold_cascades() -> None:
    quality, routes, panel = _fixtures()

    rows, paths = downstream_outcome_diagnostics(quality, routes, panel)

    assert len(rows) == 5
    assert not rows["d1_membership_changed"].any()
    assert rows["first_crossing_distance"].tolist() == [-1, 2, 3, 2, -1]
    assert rows["cumulative_membership_changed_layers"].tolist() == [0, 1, 1, 2, 0]
    assert np.allclose(
        rows["cumulative_router_mass_churn"],
        [0.0, 0.1, 0.2, 0.25, 0.0],
    )
    assert rows["downstream_outcome_only"].all()

    assert len(paths) == 1
    path = paths.iloc[0]
    assert path["d1_crossing_transition_count"] == 0
    assert path["any_downstream_crossing_transition_count"] == 2
    assert path["cumulative_changed_layer_transition_count"] == 3
    assert path["first_crossing_distance_transition_count"] == 4
    assert path["d1_constant_across_path"]
    assert path["downstream_burden_varies_across_path"]
    assert path["terminal_kl_nonmonotone"]
    assert path["terminal_nonmonotone_with_constant_d1"]
    assert path["terminal_nonmonotone_with_downstream_burden_transition"]
    assert np.isfinite(path["spearman_terminal_kl_vs_cumulative_mass"])
    assert path["terminal_kl_used_as_outcome_only"]
    assert path["downstream_route_burden_used_as_outcome_only"]


def test_downstream_outcome_rejects_noncontiguous_tail_distances() -> None:
    quality, routes, panel = _fixtures()
    bad_routes = routes[
        ~(
            routes["candidate_id"].eq("path-lambda-0")
            & routes["distance_from_injection"].eq(2)
        )
    ]

    with pytest.raises(ValueError, match="not contiguous"):
        downstream_outcome_diagnostics(quality, bad_routes, panel)
