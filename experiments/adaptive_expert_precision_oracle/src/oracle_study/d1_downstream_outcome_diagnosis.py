"""Descriptive downstream outcomes of D1-scoped Experiment A allocations.

The full tail is observed to explain delayed threshold cascades.  These values
are never allocator inputs and do not define a D2--D4 objective or model.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


TOKEN_KEYS = ["injection_layer", "group", "request_id", "position"]


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _variation(values: pd.Series) -> bool:
    array = values.to_numpy(np.float64)
    return bool(len(array) > 1 and np.ptp(array) > 0.0)


def _safe_correlation(left: pd.Series, right: pd.Series, method: str) -> float:
    if not _variation(left) or not _variation(right):
        return 0.0
    value = float(left.corr(right, method=method))
    return value if np.isfinite(value) else 0.0


def _sign_changes(values: np.ndarray) -> int:
    difference = np.diff(np.asarray(values, np.float64))
    signs = np.sign(difference)
    signs = signs[signs != 0]
    return int(np.count_nonzero(signs[1:] != signs[:-1])) if len(signs) > 1 else 0


def _attach_panel(frame: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    fields = [*TOKEN_KEYS, "panel_index", "selection_class", "selection_detail"]
    _require_columns(panel, fields, "panel")
    metadata = panel[fields]
    if metadata.duplicated(TOKEN_KEYS).any():
        raise ValueError("panel token/layer identity is not unique")
    result = frame.merge(metadata, on=TOKEN_KEYS, validate="many_to_one")
    if len(result) != len(frame):
        raise ValueError("a downstream row was not admitted by the frozen panel")
    return result


def downstream_outcome_diagnostics(
    quality: pd.DataFrame,
    routes: pd.DataFrame,
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate full-tail route outcomes and five-point path transitions.

    The first output has one live row per candidate/token.  The second has one
    row per linear interpolation path and reports descriptive associations
    with terminal KL.  No row selects a candidate.
    """

    quality_fields = [
        "candidate_kind",
        "policy",
        "rate_pages_per_expert",
        "logit_kl",
        "interpolation_kind",
        "interpolation_lambda",
        "interpolation_low_rate",
        "interpolation_high_rate",
    ]
    route_fields = [
        "distance_from_injection",
        "route_membership_changed",
        "membership_pairs_changed",
        "router_mass_churn",
        "centered_router_logit_mse",
        "hidden_mse",
    ]
    identity = [*TOKEN_KEYS, "candidate_id"]
    _require_columns(quality, [*identity, "route_mode", *quality_fields], "quality")
    _require_columns(routes, [*identity, "route_mode", *route_fields], "routes")
    live_quality = quality[quality["route_mode"].astype(str).eq("live")][
        [*identity, *quality_fields]
    ].copy()
    live_routes = routes[routes["route_mode"].astype(str).eq("live")][
        [*identity, *route_fields]
    ].copy()
    if live_quality.duplicated(identity).any():
        raise ValueError("downstream quality identity is not unique")
    if live_routes.duplicated([*identity, "distance_from_injection"]).any():
        raise ValueError("downstream route identity is not unique")

    records = []
    for key, frame in live_routes.groupby(identity, sort=True, dropna=False):
        frame = frame.sort_values("distance_from_injection", kind="stable")
        distances = frame["distance_from_injection"].to_numpy(np.int64)
        if not np.array_equal(distances, np.arange(1, len(frame) + 1)):
            raise ValueError("downstream route distances are not contiguous from D1")
        changed = frame["route_membership_changed"].astype(bool).to_numpy()
        changed_distances = distances[changed]
        mass = frame["router_mass_churn"].to_numpy(np.float64)
        records.append(
            {
                **dict(zip(identity, key)),
                "observed_downstream_layers": len(frame),
                "d1_membership_changed": bool(changed[0]),
                "d1_membership_pairs_changed": int(
                    frame.iloc[0]["membership_pairs_changed"]
                ),
                "d1_router_mass_churn": float(mass[0]),
                "any_downstream_membership_changed": bool(np.any(changed)),
                "cumulative_membership_changed_layers": int(changed.sum()),
                "post_d1_membership_changed_layers": int(changed[1:].sum()),
                "cumulative_membership_pairs_changed": int(
                    frame["membership_pairs_changed"].sum()
                ),
                "cumulative_router_mass_churn": float(mass.sum()),
                "mean_router_mass_churn": float(mass.mean()),
                "maximum_router_mass_churn": float(mass.max(initial=0.0)),
                "cumulative_centered_router_logit_mse": float(
                    frame["centered_router_logit_mse"].sum()
                ),
                "maximum_hidden_mse": float(frame["hidden_mse"].max()),
                "first_crossing_distance": (
                    int(changed_distances[0]) if len(changed_distances) else -1
                ),
                "last_crossing_distance": (
                    int(changed_distances[-1]) if len(changed_distances) else -1
                ),
                "delayed_first_crossing": bool(
                    len(changed_distances) and int(changed_distances[0]) > 1
                ),
                "downstream_outcome_only": True,
            }
        )
    rows = pd.DataFrame(records).merge(live_quality, on=identity, validate="one_to_one")
    if len(rows) != len(live_quality):
        raise ValueError("downstream route grid is incomplete")
    rows = (
        _attach_panel(rows, panel)
        .sort_values([*TOKEN_KEYS, "candidate_id"], kind="stable")
        .reset_index(drop=True)
    )

    linear = rows[
        rows["candidate_kind"].eq("interpolation_control")
        & rows["interpolation_kind"].eq("linear")
    ].copy()
    path_keys = [
        *TOKEN_KEYS,
        "interpolation_low_rate",
        "interpolation_high_rate",
    ]
    path_records = []
    for key, frame in linear.groupby(path_keys, sort=True, dropna=False):
        frame = frame.sort_values("interpolation_lambda", kind="stable")
        lambdas = frame["interpolation_lambda"].to_numpy(np.float64)
        if not np.array_equal(lambdas, np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])):
            raise ValueError("a downstream interpolation path is incomplete")
        terminal = frame["logit_kl"].to_numpy(np.float64)
        d1 = frame["d1_membership_changed"].astype(int).to_numpy()
        any_cross = frame["any_downstream_membership_changed"].astype(int).to_numpy()
        changed_layers = frame["cumulative_membership_changed_layers"].to_numpy(
            np.float64
        )
        cumulative_mass = frame["cumulative_router_mass_churn"].to_numpy(np.float64)
        first_crossing = frame["first_crossing_distance"].to_numpy(np.float64)
        terminal_turns = _sign_changes(terminal)
        d1_transitions = int(np.count_nonzero(d1[1:] != d1[:-1]))
        downstream_transitions = int(np.count_nonzero(any_cross[1:] != any_cross[:-1]))
        changed_layer_transitions = int(
            np.count_nonzero(changed_layers[1:] != changed_layers[:-1])
        )
        mass_transitions = int(
            np.count_nonzero(cumulative_mass[1:] != cumulative_mass[:-1])
        )
        path_records.append(
            {
                **dict(zip(path_keys, key)),
                "selection_class": str(frame.iloc[0]["selection_class"]),
                "terminal_kl_discrete_turning_points": terminal_turns,
                "d1_crossing_transition_count": d1_transitions,
                "any_downstream_crossing_transition_count": downstream_transitions,
                "cumulative_changed_layer_transition_count": changed_layer_transitions,
                "cumulative_mass_transition_count": mass_transitions,
                "first_crossing_distance_transition_count": int(
                    np.count_nonzero(first_crossing[1:] != first_crossing[:-1])
                ),
                "d1_constant_across_path": bool(d1_transitions == 0),
                "downstream_burden_varies_across_path": bool(
                    changed_layer_transitions > 0 or mass_transitions > 0
                ),
                "terminal_kl_nonmonotone": bool(terminal_turns > 0),
                "terminal_nonmonotone_with_constant_d1": bool(
                    terminal_turns > 0 and d1_transitions == 0
                ),
                "terminal_nonmonotone_with_downstream_burden_transition": bool(
                    terminal_turns > 0
                    and (changed_layer_transitions > 0 or mass_transitions > 0)
                ),
                "pearson_terminal_kl_vs_cumulative_changed_layers": _safe_correlation(
                    frame["logit_kl"],
                    frame["cumulative_membership_changed_layers"],
                    "pearson",
                ),
                "spearman_terminal_kl_vs_cumulative_changed_layers": _safe_correlation(
                    frame["logit_kl"],
                    frame["cumulative_membership_changed_layers"],
                    "spearman",
                ),
                "pearson_terminal_kl_vs_cumulative_mass": _safe_correlation(
                    frame["logit_kl"],
                    frame["cumulative_router_mass_churn"],
                    "pearson",
                ),
                "spearman_terminal_kl_vs_cumulative_mass": _safe_correlation(
                    frame["logit_kl"],
                    frame["cumulative_router_mass_churn"],
                    "spearman",
                ),
                "terminal_kl_used_as_outcome_only": True,
                "downstream_route_burden_used_as_outcome_only": True,
            }
        )
    paths = (
        pd.DataFrame(path_records)
        .sort_values(path_keys, kind="stable")
        .reset_index(drop=True)
    )
    return rows, paths
