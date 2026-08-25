"""GPU-free analysis primitives for the focused Experiment A D1 panel.

Terminal KL is retained strictly as an observed outcome.  The module does not
construct a D2--D4 objective, predictor, policy, or runtime controller.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
from typing import Any

import numpy as np
import pandas as pd


SCHEMA = "pr13_d1_experiment_a_diagnostic_analysis_v1"
PR13_POLICY = "pr13_router_square"
FIXED_D1_POLICY = "calibration_selected_fixed_d1"
TOKEN_KEYS = ["injection_layer", "group", "request_id", "position"]
CANDIDATE_KEYS = [
    *TOKEN_KEYS,
    "rate_pages_per_expert",
    "candidate_id",
    "candidate_kind",
    "policy",
]
RATE_PAIRS = ((360, 384), (725, 749))
ROUTE_MODES = ("live", "frozen_set_live_weights", "fully_frozen")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _quantile(values: pd.Series, q: float) -> float:
    array = values.to_numpy(np.float64)
    return float(np.quantile(array, q)) if len(array) else 0.0


def _variation(values: pd.Series) -> bool:
    array = values.to_numpy(np.float64)
    return bool(len(array) > 1 and np.ptp(array) > 0.0)


def _safe_correlation(left: pd.Series, right: pd.Series, method: str) -> float:
    if not _variation(left) or not _variation(right):
        return 0.0
    value = float(left.corr(right, method=method))
    return value if np.isfinite(value) else 0.0


def _attach_panel(frame: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    _require_columns(panel, [*TOKEN_KEYS, "selection_class", "panel_index"], "panel")
    metadata = panel[
        [*TOKEN_KEYS, "panel_index", "selection_class", "selection_detail"]
    ]
    if metadata.duplicated(TOKEN_KEYS).any():
        raise ValueError("panel token/layer identity is not unique")
    result = frame.merge(metadata, on=TOKEN_KEYS, validate="many_to_one")
    if len(result) != len(frame):
        raise ValueError("an analysis row was not admitted by the frozen panel")
    return result


def _scope_groups(
    frame: pd.DataFrame,
    base: Sequence[str],
) -> list[tuple[str, list[str], Any]]:
    return [
        ("all_panel", list(base), frame.groupby(list(base), sort=True, dropna=False)),
        (
            "by_layer",
            ["injection_layer", *base],
            frame.groupby(["injection_layer", *base], sort=True, dropna=False),
        ),
        (
            "by_cohort",
            ["selection_class", *base],
            frame.groupby(["selection_class", *base], sort=True, dropna=False),
        ),
    ]


def route_mode_decomposition(
    quality: pd.DataFrame,
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Form the exact arithmetic three-route-mode KL decomposition."""

    fields = [
        "logit_kl",
        "delta_nll",
        "final_hidden_mse",
        "post_token_full_cache_mse",
        "live_selected_local_qenergy_damage",
        "planned_delta_l2",
        "bf16_effective_delta_l2",
    ]
    _require_columns(quality, [*CANDIDATE_KEYS, "route_mode", *fields], "quality")
    pieces = {}
    for mode in ROUTE_MODES:
        selected = quality[quality["route_mode"].astype(str).eq(mode)][
            CANDIDATE_KEYS + fields
        ].copy()
        if selected.duplicated(CANDIDATE_KEYS).any():
            raise ValueError(f"duplicate quality identity for route mode {mode}")
        pieces[mode] = selected.rename(
            columns={field: f"{mode}_{field}" for field in fields}
        )
    rows = pieces["live"]
    for mode in ROUTE_MODES[1:]:
        rows = rows.merge(pieces[mode], on=CANDIDATE_KEYS, validate="one_to_one")
    if any(len(pieces[mode]) != len(rows) for mode in ROUTE_MODES):
        raise ValueError("route-mode quality grids are incomplete")
    rows = _attach_panel(rows, panel)
    rows["membership_execution_component_kl"] = (
        rows["live_logit_kl"] - rows["frozen_set_live_weights_logit_kl"]
    )
    rows["router_weight_component_kl"] = (
        rows["frozen_set_live_weights_logit_kl"] - rows["fully_frozen_logit_kl"]
    )
    rows["continuous_fully_frozen_component_kl"] = rows["fully_frozen_logit_kl"]
    rows["decomposition_closure_error"] = (
        rows["continuous_fully_frozen_component_kl"]
        + rows["router_weight_component_kl"]
        + rows["membership_execution_component_kl"]
        - rows["live_logit_kl"]
    )
    if float(rows["decomposition_closure_error"].abs().max()) > 1e-15:
        raise RuntimeError("route-mode KL arithmetic decomposition does not close")
    rows = rows.sort_values([*TOKEN_KEYS, "candidate_id"], kind="stable").reset_index(
        drop=True
    )

    summaries = []
    for scope, columns, grouped in _scope_groups(
        rows,
        ["candidate_kind", "candidate_id", "rate_pages_per_expert"],
    ):
        summary = grouped.agg(
            tokens=("group", "size"),
            mean_live_kl=("live_logit_kl", "mean"),
            median_live_kl=("live_logit_kl", "median"),
            p90_live_kl=("live_logit_kl", lambda value: _quantile(value, 0.90)),
            mean_membership_component_kl=("membership_execution_component_kl", "mean"),
            median_membership_component_kl=(
                "membership_execution_component_kl",
                "median",
            ),
            positive_membership_component_fraction=(
                "membership_execution_component_kl",
                lambda value: float((value > 0).mean()),
            ),
            mean_router_weight_component_kl=("router_weight_component_kl", "mean"),
            positive_router_weight_component_fraction=(
                "router_weight_component_kl",
                lambda value: float((value > 0).mean()),
            ),
            mean_fully_frozen_component_kl=(
                "continuous_fully_frozen_component_kl",
                "mean",
            ),
            maximum_closure_error=(
                "decomposition_closure_error",
                lambda value: float(value.abs().max()),
            ),
        ).reset_index()
        summary.insert(0, "scope", scope)
        if "injection_layer" not in summary:
            summary.insert(1, "injection_layer", -1)
        if "selection_class" not in summary:
            summary.insert(2, "selection_class", "all")
        summaries.append(summary)
    summary = (
        pd.concat(summaries, ignore_index=True)
        .sort_values(
            ["scope", "injection_layer", "selection_class", "candidate_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    return rows, summary


def immediate_d1_severity(
    quality: pd.DataFrame,
    routes: pd.DataFrame,
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Join exact immediate-D1 route severity to terminal outcomes."""

    quality_fields = [
        "logit_kl",
        "delta_nll",
        "live_selected_local_qenergy_damage",
        "selected_group_pages",
        "planned_delta_l2",
        "bf16_effective_delta_l2",
        "bf16_collapsed_nonzero_fraction",
        "interpolation_kind",
        "interpolation_lambda",
        "interpolation_scale",
        "interpolation_low_rate",
        "interpolation_high_rate",
        "candidate_kind",
        "policy",
        "rate_pages_per_expert",
    ]
    route_fields = [
        "route_membership_changed",
        "membership_pairs_changed",
        "router_mass_churn",
        "left_reference_routing_mass",
        "entered_candidate_routing_mass",
        "baseline_rank8_rank9_margin",
        "candidate_rank8_rank9_margin",
        "baseline_set_margin_under_candidate_logits",
        "centered_router_logit_mse",
        "centered_router_logit_max_abs",
        "hidden_mse",
    ]
    _require_columns(
        quality,
        [*TOKEN_KEYS, "candidate_id", "route_mode", *quality_fields],
        "quality",
    )
    _require_columns(
        routes,
        [
            *TOKEN_KEYS,
            "candidate_id",
            "route_mode",
            "distance_from_injection",
            *route_fields,
        ],
        "routes",
    )
    live_quality = quality[quality["route_mode"].astype(str).eq("live")][
        [*TOKEN_KEYS, "candidate_id", *quality_fields]
    ].copy()
    d1 = routes[
        routes["route_mode"].astype(str).eq("live")
        & routes["distance_from_injection"].eq(1)
    ][[*TOKEN_KEYS, "candidate_id", *route_fields]].copy()
    identity = [*TOKEN_KEYS, "candidate_id"]
    if live_quality.duplicated(identity).any() or d1.duplicated(identity).any():
        raise ValueError("immediate-D1 identity is not unique")
    rows = live_quality.merge(d1, on=identity, validate="one_to_one")
    if len(rows) != len(live_quality):
        raise ValueError("immediate-D1 route grid is incomplete")
    rows = _attach_panel(rows, panel)
    rows["route_membership_changed"] = rows["route_membership_changed"].astype(bool)
    rows["centered_router_logit_rmse"] = np.sqrt(
        rows["centered_router_logit_mse"].to_numpy(np.float64)
    )
    rows["boundary_violation_depth"] = np.maximum(
        -rows["baseline_set_margin_under_candidate_logits"].to_numpy(np.float64),
        0.0,
    )
    rows["boundary_margin_reduction"] = (
        rows["baseline_rank8_rank9_margin"]
        - rows["baseline_set_margin_under_candidate_logits"]
    )
    rows["changed_routing_mass"] = 0.5 * (
        rows["left_reference_routing_mass"] + rows["entered_candidate_routing_mass"]
    )
    rows = rows.sort_values([*TOKEN_KEYS, "candidate_id"], kind="stable").reset_index(
        drop=True
    )

    metrics = [
        "route_membership_changed",
        "membership_pairs_changed",
        "router_mass_churn",
        "changed_routing_mass",
        "boundary_violation_depth",
        "boundary_margin_reduction",
        "centered_router_logit_rmse",
        "centered_router_logit_max_abs",
        "hidden_mse",
        "live_selected_local_qenergy_damage",
        "planned_delta_l2",
        "bf16_collapsed_nonzero_fraction",
    ]
    correlation_rows = []
    scopes = [("all_panel", (), rows)]
    scopes.extend(
        ("by_layer", (int(key),), value)
        for key, value in rows.groupby("injection_layer", sort=True)
    )
    scopes.extend(
        ("by_cohort", (str(key),), value)
        for key, value in rows.groupby("selection_class", sort=True)
    )
    for scope, key, frame in scopes:
        for metric in metrics:
            left = frame[metric].astype(float)
            right = frame["logit_kl"].astype(float)
            correlation_rows.append(
                {
                    "scope": scope,
                    "injection_layer": int(key[0]) if scope == "by_layer" else -1,
                    "selection_class": str(key[0]) if scope == "by_cohort" else "all",
                    "metric": metric,
                    "rows": len(frame),
                    "metric_varies": _variation(left),
                    "terminal_kl_varies": _variation(right),
                    "pearson_with_terminal_kl": _safe_correlation(
                        left, right, "pearson"
                    ),
                    "spearman_with_terminal_kl": _safe_correlation(
                        left, right, "spearman"
                    ),
                    "terminal_kl_used_as_outcome_only": True,
                }
            )
    correlations = (
        pd.DataFrame(correlation_rows)
        .sort_values(
            ["scope", "injection_layer", "selection_class", "metric"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    summaries = []
    for scope, columns, grouped in _scope_groups(
        rows,
        ["candidate_kind", "candidate_id", "rate_pages_per_expert"],
    ):
        summary = grouped.agg(
            tokens=("group", "size"),
            d1_crossing_fraction=("route_membership_changed", "mean"),
            mean_membership_pairs_changed=("membership_pairs_changed", "mean"),
            mean_router_mass_churn=("router_mass_churn", "mean"),
            p90_router_mass_churn=(
                "router_mass_churn",
                lambda value: _quantile(value, 0.90),
            ),
            mean_centered_router_logit_rmse=("centered_router_logit_rmse", "mean"),
            mean_boundary_violation_depth=("boundary_violation_depth", "mean"),
            mean_d1_hidden_mse=("hidden_mse", "mean"),
            mean_terminal_kl=("logit_kl", "mean"),
            median_terminal_kl=("logit_kl", "median"),
            p90_terminal_kl=("logit_kl", lambda value: _quantile(value, 0.90)),
        ).reset_index()
        summary.insert(0, "scope", scope)
        if "injection_layer" not in summary:
            summary.insert(1, "injection_layer", -1)
        if "selection_class" not in summary:
            summary.insert(2, "selection_class", "all")
        summaries.append(summary)
    summary = (
        pd.concat(summaries, ignore_index=True)
        .sort_values(
            ["scope", "injection_layer", "selection_class", "candidate_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    return rows, correlations, summary


def _sign_changes(values: np.ndarray) -> int:
    difference = np.diff(np.asarray(values, np.float64))
    signs = np.sign(difference)
    signs = signs[signs != 0]
    return int(np.count_nonzero(signs[1:] != signs[:-1])) if len(signs) > 1 else 0


def interpolation_turning_points(
    severity: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Describe D1 path turning points without using KL as an allocator input."""

    required = [
        *TOKEN_KEYS,
        "candidate_id",
        "candidate_kind",
        "interpolation_kind",
        "interpolation_lambda",
        "interpolation_low_rate",
        "interpolation_high_rate",
        "logit_kl",
        "route_membership_changed",
        "router_mass_churn",
        "centered_router_logit_mse",
        "live_selected_local_qenergy_damage",
        "planned_delta_l2",
        "bf16_effective_delta_l2",
        "selection_class",
    ]
    _require_columns(severity, required, "severity")
    points = severity[severity["candidate_kind"].eq("interpolation_control")][
        required
    ].copy()
    if points.empty:
        raise ValueError("diagnostic output contains no interpolation controls")
    points = points.sort_values(
        [
            *TOKEN_KEYS,
            "interpolation_low_rate",
            "interpolation_kind",
            "interpolation_lambda",
        ],
        kind="stable",
    ).reset_index(drop=True)
    linear = points[points["interpolation_kind"].eq("linear")].copy()
    path_keys = [*TOKEN_KEYS, "interpolation_low_rate", "interpolation_high_rate"]
    turning_rows = []
    for key, frame in linear.groupby(path_keys, sort=True):
        frame = frame.sort_values("interpolation_lambda", kind="stable")
        lambdas = frame["interpolation_lambda"].to_numpy(np.float64)
        if not np.array_equal(lambdas, np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])):
            raise ValueError("a fixed-D1 interpolation path is incomplete")
        d1_sorted = frame.sort_values(
            [
                "route_membership_changed",
                "router_mass_churn",
                "centered_router_logit_mse",
                "live_selected_local_qenergy_damage",
                "interpolation_lambda",
            ],
            kind="stable",
        )
        d1_choice = d1_sorted.iloc[0]
        outcome_best = frame.sort_values(
            ["logit_kl", "interpolation_lambda"],
            kind="stable",
        ).iloc[0]
        crossing = frame["route_membership_changed"].astype(int).to_numpy()
        crossed_lambdas = lambdas[crossing > 0]
        kl = frame["logit_kl"].to_numpy(np.float64)
        local = frame["live_selected_local_qenergy_damage"].to_numpy(np.float64)
        norm = frame["planned_delta_l2"].to_numpy(np.float64)
        turning_rows.append(
            {
                **dict(zip(path_keys, key)),
                "selection_class": str(frame.iloc[0]["selection_class"]),
                "d1_lexicographic_lambda": float(d1_choice["interpolation_lambda"]),
                "d1_lexicographic_terminal_kl_outcome": float(d1_choice["logit_kl"]),
                "observed_minimum_kl_lambda": float(
                    outcome_best["interpolation_lambda"]
                ),
                "observed_minimum_terminal_kl": float(outcome_best["logit_kl"]),
                "terminal_kl_not_used_by_d1_lexicographic_choice": True,
                "first_crossing_lambda": (
                    float(crossed_lambdas[0]) if len(crossed_lambdas) else -1.0
                ),
                "crossing_transition_count": int(
                    np.count_nonzero(crossing[1:] != crossing[:-1])
                ),
                "terminal_kl_discrete_turning_points": _sign_changes(kl),
                "local_qenergy_discrete_turning_points": _sign_changes(local),
                "planned_norm_discrete_turning_points": _sign_changes(norm),
                "terminal_kl_monotone_nonincreasing": bool(np.all(np.diff(kl) <= 0.0)),
                "local_qenergy_monotone_nonincreasing": bool(
                    np.all(np.diff(local) <= 0.0)
                ),
                "planned_norm_monotone_nonincreasing": bool(
                    np.all(np.diff(norm) <= 0.0)
                ),
                "high_minus_low_terminal_kl": float(kl[-1] - kl[0]),
                "high_minus_low_local_qenergy": float(local[-1] - local[0]),
                "high_minus_low_planned_norm": float(norm[-1] - norm[0]),
                "interior_kl_above_both_endpoints": bool(
                    np.max(kl[1:-1]) > max(kl[0], kl[-1])
                ),
                "interior_kl_below_both_endpoints": bool(
                    np.min(kl[1:-1]) < min(kl[0], kl[-1])
                ),
            }
        )
    turning = (
        pd.DataFrame(turning_rows)
        .sort_values(path_keys, kind="stable")
        .reset_index(drop=True)
    )

    summaries = []
    for scope, group_columns in (
        ("all_panel", ["interpolation_low_rate", "interpolation_high_rate"]),
        (
            "by_cohort",
            ["selection_class", "interpolation_low_rate", "interpolation_high_rate"],
        ),
        (
            "by_layer",
            ["injection_layer", "interpolation_low_rate", "interpolation_high_rate"],
        ),
    ):
        summary = (
            turning.groupby(group_columns, sort=True, dropna=False)
            .agg(
                paths=("group", "size"),
                mean_high_minus_low_terminal_kl=("high_minus_low_terminal_kl", "mean"),
                median_high_minus_low_terminal_kl=(
                    "high_minus_low_terminal_kl",
                    "median",
                ),
                high_worse_fraction=(
                    "high_minus_low_terminal_kl",
                    lambda value: float((value > 0).mean()),
                ),
                terminal_kl_nonmonotone_fraction=(
                    "terminal_kl_discrete_turning_points",
                    lambda value: float((value > 0).mean()),
                ),
                crossing_transition_fraction=(
                    "crossing_transition_count",
                    lambda value: float((value > 0).mean()),
                ),
                d1_choice_matches_observed_kl_min_fraction=(
                    "d1_lexicographic_lambda",
                    lambda value: 0.0,
                ),
                interior_above_endpoints_fraction=(
                    "interior_kl_above_both_endpoints",
                    "mean",
                ),
                interior_below_endpoints_fraction=(
                    "interior_kl_below_both_endpoints",
                    "mean",
                ),
            )
            .reset_index()
        )
        # Named aggregation cannot compare two columns; calculate the match
        # fraction from the same deterministic groups explicitly.
        match = (
            turning.assign(
                _match=turning["d1_lexicographic_lambda"].eq(
                    turning["observed_minimum_kl_lambda"]
                )
            )
            .groupby(group_columns, sort=True, dropna=False)["_match"]
            .mean()
            .reset_index(name="_match")
        )
        summary = (
            summary.drop(columns="d1_choice_matches_observed_kl_min_fraction")
            .merge(
                match,
                on=group_columns,
                validate="one_to_one",
            )
            .rename(columns={"_match": "d1_choice_matches_observed_kl_min_fraction"})
        )
        summary.insert(0, "scope", scope)
        if "injection_layer" not in summary:
            summary.insert(1, "injection_layer", -1)
        if "selection_class" not in summary:
            summary.insert(2, "selection_class", "all")
        summaries.append(summary)
    summary = (
        pd.concat(summaries, ignore_index=True)
        .sort_values(
            ["scope", "injection_layer", "selection_class", "interpolation_low_rate"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    return points, turning, summary


def page_direction_norm_explanation(
    severity: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use path geometry and norm matching to separate direction from norm."""

    identity = TOKEN_KEYS
    rows = []
    for low, high in RATE_PAIRS:
        linear = severity[
            severity["candidate_kind"].eq("interpolation_control")
            & severity["interpolation_kind"].eq("linear")
            & severity["interpolation_low_rate"].eq(low)
            & severity["interpolation_high_rate"].eq(high)
        ]
        rescaled = severity[
            severity["candidate_kind"].eq("interpolation_control")
            & severity["interpolation_kind"].eq("high_rescaled_to_low_norm")
            & severity["interpolation_low_rate"].eq(low)
            & severity["interpolation_high_rate"].eq(high)
        ]
        fixed_low = severity[
            severity["candidate_id"].eq(f"rate_{low}__{FIXED_D1_POLICY}")
        ]
        fixed_high = severity[
            severity["candidate_id"].eq(f"rate_{high}__{FIXED_D1_POLICY}")
        ]
        endpoint = {}
        for lam, label in ((0.0, "low"), (0.5, "mid"), (1.0, "high")):
            selected = linear[linear["interpolation_lambda"].eq(lam)][
                [
                    *identity,
                    "selection_class",
                    "logit_kl",
                    "route_membership_changed",
                    "router_mass_churn",
                    "live_selected_local_qenergy_damage",
                    "planned_delta_l2",
                    "bf16_effective_delta_l2",
                ]
            ].rename(
                columns={
                    column: f"{label}_{column}"
                    for column in (
                        "logit_kl",
                        "route_membership_changed",
                        "router_mass_churn",
                        "live_selected_local_qenergy_damage",
                        "planned_delta_l2",
                        "bf16_effective_delta_l2",
                    )
                }
            )
            endpoint[label] = selected
        combined = (
            endpoint["low"]
            .merge(
                endpoint["mid"].drop(columns="selection_class"),
                on=identity,
                validate="one_to_one",
            )
            .merge(
                endpoint["high"].drop(columns="selection_class"),
                on=identity,
                validate="one_to_one",
            )
        )
        rescale_fields = [
            *identity,
            "logit_kl",
            "route_membership_changed",
            "router_mass_churn",
            "live_selected_local_qenergy_damage",
            "planned_delta_l2",
            "bf16_effective_delta_l2",
            "bf16_collapsed_nonzero_fraction",
        ]
        combined = combined.merge(
            rescaled[rescale_fields].rename(
                columns={
                    column: f"rescaled_{column}"
                    for column in rescale_fields
                    if column not in identity
                }
            ),
            on=identity,
            validate="one_to_one",
        )
        page_fields = [*identity, "selected_group_pages"]
        combined = combined.merge(
            fixed_low[page_fields].rename(
                columns={"selected_group_pages": "low_selected_group_pages"}
            ),
            on=identity,
            validate="one_to_one",
        ).merge(
            fixed_high[page_fields].rename(
                columns={"selected_group_pages": "high_selected_group_pages"}
            ),
            on=identity,
            validate="one_to_one",
        )
        low_norm = combined["low_planned_delta_l2"].to_numpy(np.float64)
        mid_norm = combined["mid_planned_delta_l2"].to_numpy(np.float64)
        high_norm = combined["high_planned_delta_l2"].to_numpy(np.float64)
        dot = 2.0 * mid_norm**2 - 0.5 * (low_norm**2 + high_norm**2)
        denominator = low_norm * high_norm
        cosine = np.ones(len(combined), np.float64)
        valid = denominator > 0.0
        cosine[valid] = np.clip(dot[valid] / denominator[valid], -1.0, 1.0)
        increment_squared = np.maximum(
            2.0 * (low_norm**2 + high_norm**2) - 4.0 * mid_norm**2,
            0.0,
        )
        combined.insert(0, "low_rate", low)
        combined.insert(1, "high_rate", high)
        combined["low_high_delta_cosine"] = cosine
        combined["low_high_delta_angle_degrees"] = np.degrees(np.arccos(cosine))
        combined["high_minus_low_delta_l2"] = np.sqrt(increment_squared)
        combined["high_to_low_planned_norm_ratio"] = np.divide(
            high_norm,
            low_norm,
            out=np.ones_like(high_norm),
            where=low_norm > 0.0,
        )
        combined["selected_group_pages_high_minus_low"] = (
            combined["high_selected_group_pages"] - combined["low_selected_group_pages"]
        )
        combined["observed_high_minus_low_terminal_kl"] = (
            combined["high_logit_kl"] - combined["low_logit_kl"]
        )
        combined["direction_at_matched_norm_terminal_kl"] = (
            combined["rescaled_logit_kl"] - combined["low_logit_kl"]
        )
        combined["norm_along_high_direction_terminal_kl"] = (
            combined["high_logit_kl"] - combined["rescaled_logit_kl"]
        )
        combined["kl_component_closure_error"] = (
            combined["direction_at_matched_norm_terminal_kl"]
            + combined["norm_along_high_direction_terminal_kl"]
            - combined["observed_high_minus_low_terminal_kl"]
        )
        combined["rescaled_to_low_planned_norm_relative_error"] = np.divide(
            np.abs(
                combined["rescaled_planned_delta_l2"] - combined["low_planned_delta_l2"]
            ),
            combined["low_planned_delta_l2"],
            out=np.zeros(len(combined), np.float64),
            where=combined["low_planned_delta_l2"].to_numpy(np.float64) > 0.0,
        )
        direction_abs = combined["direction_at_matched_norm_terminal_kl"].abs()
        norm_abs = combined["norm_along_high_direction_terminal_kl"].abs()
        combined["larger_absolute_kl_component"] = np.where(
            direction_abs > norm_abs,
            "direction",
            np.where(norm_abs > direction_abs, "norm", "tie"),
        )
        combined["local_improved_but_terminal_kl_worsened"] = (
            combined["high_live_selected_local_qenergy_damage"]
            < combined["low_live_selected_local_qenergy_damage"]
        ) & (combined["observed_high_minus_low_terminal_kl"] > 0.0)
        rows.append(combined)
    detail = (
        pd.concat(rows, ignore_index=True)
        .sort_values(
            ["low_rate", *TOKEN_KEYS],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    if float(detail["kl_component_closure_error"].abs().max()) > 1e-15:
        raise RuntimeError("direction/norm terminal-KL decomposition does not close")

    summaries = []
    for scope, columns in (
        ("all_panel", ["low_rate", "high_rate"]),
        ("by_layer", ["injection_layer", "low_rate", "high_rate"]),
        ("by_cohort", ["selection_class", "low_rate", "high_rate"]),
    ):
        summary = (
            detail.groupby(columns, sort=True, dropna=False)
            .agg(
                tokens=("group", "size"),
                mean_selected_pages_change=(
                    "selected_group_pages_high_minus_low",
                    "mean",
                ),
                mean_delta_cosine=("low_high_delta_cosine", "mean"),
                p10_delta_cosine=(
                    "low_high_delta_cosine",
                    lambda value: _quantile(value, 0.10),
                ),
                mean_delta_angle_degrees=("low_high_delta_angle_degrees", "mean"),
                mean_high_to_low_norm_ratio=("high_to_low_planned_norm_ratio", "mean"),
                mean_high_minus_low_terminal_kl=(
                    "observed_high_minus_low_terminal_kl",
                    "mean",
                ),
                mean_direction_matched_norm_terminal_kl=(
                    "direction_at_matched_norm_terminal_kl",
                    "mean",
                ),
                mean_norm_along_direction_terminal_kl=(
                    "norm_along_high_direction_terminal_kl",
                    "mean",
                ),
                direction_dominant_fraction=(
                    "larger_absolute_kl_component",
                    lambda value: float((value == "direction").mean()),
                ),
                norm_dominant_fraction=(
                    "larger_absolute_kl_component",
                    lambda value: float((value == "norm").mean()),
                ),
                paradox_fraction=("local_improved_but_terminal_kl_worsened", "mean"),
                maximum_norm_match_relative_error=(
                    "rescaled_to_low_planned_norm_relative_error",
                    "max",
                ),
            )
            .reset_index()
        )
        summary.insert(0, "scope", scope)
        if "injection_layer" not in summary:
            summary.insert(1, "injection_layer", -1)
        if "selection_class" not in summary:
            summary.insert(2, "selection_class", "all")
        summaries.append(summary)
    summary = (
        pd.concat(summaries, ignore_index=True)
        .sort_values(
            ["scope", "injection_layer", "selection_class", "low_rate"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    return detail, summary


def bf16_ulp_analysis(
    quality: pd.DataFrame,
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fields = [
        "planned_delta_l2",
        "bf16_effective_delta_l2",
        "bf16_effective_to_planned_norm_ratio",
        "bf16_quantization_error_mse",
        "bf16_quantization_error_max_abs",
        "bf16_quantization_error_ulp_mean",
        "bf16_quantization_error_ulp_p95",
        "bf16_quantization_error_ulp_max",
        "planned_below_half_ulp_fraction",
        "planned_nonzero_coordinates",
        "bf16_changed_coordinates",
        "bf16_collapsed_nonzero_coordinates",
        "bf16_collapsed_nonzero_fraction",
        "logit_kl",
        "interpolation_kind",
    ]
    _require_columns(
        quality,
        [*CANDIDATE_KEYS, "route_mode", *fields],
        "quality",
    )
    detail = quality[quality["route_mode"].astype(str).eq("live")][
        [*CANDIDATE_KEYS, *fields]
    ].copy()
    detail = (
        _attach_panel(detail, panel)
        .sort_values(
            [*TOKEN_KEYS, "candidate_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    detail["any_bf16_collapse"] = detail["bf16_collapsed_nonzero_coordinates"].gt(0)

    summaries = []
    for scope, columns, grouped in _scope_groups(
        detail,
        ["candidate_kind", "interpolation_kind", "rate_pages_per_expert"],
    ):
        summary = grouped.agg(
            candidates=("candidate_id", "size"),
            any_collapse_fraction=("any_bf16_collapse", "mean"),
            mean_collapsed_nonzero_fraction=("bf16_collapsed_nonzero_fraction", "mean"),
            p95_collapsed_nonzero_fraction=(
                "bf16_collapsed_nonzero_fraction",
                lambda value: _quantile(value, 0.95),
            ),
            maximum_collapsed_nonzero_fraction=(
                "bf16_collapsed_nonzero_fraction",
                "max",
            ),
            mean_planned_below_half_ulp_fraction=(
                "planned_below_half_ulp_fraction",
                "mean",
            ),
            mean_effective_to_planned_norm_ratio=(
                "bf16_effective_to_planned_norm_ratio",
                "mean",
            ),
            p95_quantization_error_ulp=(
                "bf16_quantization_error_ulp_p95",
                lambda value: _quantile(value, 0.95),
            ),
            maximum_quantization_error_ulp=("bf16_quantization_error_ulp_max", "max"),
            mean_quantization_error_mse=("bf16_quantization_error_mse", "mean"),
        ).reset_index()
        summary.insert(0, "scope", scope)
        if "injection_layer" not in summary:
            summary.insert(1, "injection_layer", -1)
        if "selection_class" not in summary:
            summary.insert(2, "selection_class", "all")
        summaries.append(summary)
    summary = (
        pd.concat(summaries, ignore_index=True)
        .sort_values(
            [
                "scope",
                "injection_layer",
                "selection_class",
                "candidate_kind",
                "interpolation_kind",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    associations = []
    for metric in (
        "bf16_collapsed_nonzero_fraction",
        "planned_below_half_ulp_fraction",
        "bf16_effective_to_planned_norm_ratio",
        "bf16_quantization_error_mse",
        "bf16_quantization_error_ulp_p95",
    ):
        associations.append(
            {
                "scope": "all_panel",
                "metric": metric,
                "rows": len(detail),
                "pearson_with_terminal_kl": _safe_correlation(
                    detail[metric], detail["logit_kl"], "pearson"
                ),
                "spearman_with_terminal_kl": _safe_correlation(
                    detail[metric], detail["logit_kl"], "spearman"
                ),
                "terminal_kl_used_as_outcome_only": True,
            }
        )
    association = (
        pd.DataFrame(associations)
        .sort_values("metric", kind="stable")
        .reset_index(drop=True)
    )
    return detail, summary, association


def cohort_policy_comparisons(
    severity: pd.DataFrame,
    decomposition: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair fixed D1 and PR13 inside each enriched panel cohort."""

    fields = [
        "logit_kl",
        "route_membership_changed",
        "membership_pairs_changed",
        "router_mass_churn",
        "centered_router_logit_mse",
        "hidden_mse",
        "live_selected_local_qenergy_damage",
        "selected_group_pages",
        "planned_delta_l2",
    ]
    base = severity[severity["candidate_kind"].eq("allocation_policy")]
    key = [*TOKEN_KEYS, "rate_pages_per_expert"]
    pr13 = base[base["policy"].eq(PR13_POLICY)][
        key + ["selection_class", *fields]
    ].rename(columns={field: f"pr13_{field}" for field in fields})
    fixed = base[base["policy"].eq(FIXED_D1_POLICY)][key + fields].rename(
        columns={field: f"fixed_{field}" for field in fields}
    )
    paired = pr13.merge(fixed, on=key, validate="one_to_one")
    for field in fields:
        paired[f"fixed_minus_pr13_{field}"] = paired[f"fixed_{field}"].astype(
            float
        ) - paired[f"pr13_{field}"].astype(float)
    decomp_fields = [
        "membership_execution_component_kl",
        "router_weight_component_kl",
        "continuous_fully_frozen_component_kl",
    ]
    for policy, prefix in ((PR13_POLICY, "pr13"), (FIXED_D1_POLICY, "fixed")):
        selected = decomposition[decomposition["policy"].eq(policy)][
            key + decomp_fields
        ].rename(columns={field: f"{prefix}_{field}" for field in decomp_fields})
        paired = paired.merge(selected, on=key, validate="one_to_one")
    for field in decomp_fields:
        paired[f"fixed_minus_pr13_{field}"] = (
            paired[f"fixed_{field}"] - paired[f"pr13_{field}"]
        )
    pr13_cross = paired["pr13_route_membership_changed"].astype(bool)
    fixed_cross = paired["fixed_route_membership_changed"].astype(bool)
    paired["crossing_event"] = np.select(
        [
            pr13_cross & ~fixed_cross,
            ~pr13_cross & fixed_cross,
            pr13_cross & fixed_cross,
        ],
        ["prevented", "introduced", "both_crossed"],
        default="both_safe",
    )
    paired = paired.sort_values(["selection_class", *key], kind="stable").reset_index(
        drop=True
    )

    summary = (
        paired.groupby(
            ["selection_class", "rate_pages_per_expert", "crossing_event"],
            sort=True,
            dropna=False,
        )
        .agg(
            tokens=("group", "size"),
            mean_fixed_minus_pr13_terminal_kl=("fixed_minus_pr13_logit_kl", "mean"),
            median_fixed_minus_pr13_terminal_kl=("fixed_minus_pr13_logit_kl", "median"),
            p90_fixed_minus_pr13_terminal_kl=(
                "fixed_minus_pr13_logit_kl",
                lambda value: _quantile(value, 0.90),
            ),
            improved_fraction=(
                "fixed_minus_pr13_logit_kl",
                lambda value: float((value < 0).mean()),
            ),
            mean_fixed_minus_pr13_d1_mass=(
                "fixed_minus_pr13_router_mass_churn",
                "mean",
            ),
            mean_fixed_minus_pr13_centered_logit_mse=(
                "fixed_minus_pr13_centered_router_logit_mse",
                "mean",
            ),
            mean_fixed_minus_pr13_local_qenergy=(
                "fixed_minus_pr13_live_selected_local_qenergy_damage",
                "mean",
            ),
            mean_fixed_minus_pr13_membership_component_kl=(
                "fixed_minus_pr13_membership_execution_component_kl",
                "mean",
            ),
            mean_fixed_minus_pr13_weight_component_kl=(
                "fixed_minus_pr13_router_weight_component_kl",
                "mean",
            ),
            mean_fixed_minus_pr13_fully_frozen_component_kl=(
                "fixed_minus_pr13_continuous_fully_frozen_component_kl",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            ["selection_class", "rate_pages_per_expert", "crossing_event"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    return paired, summary
