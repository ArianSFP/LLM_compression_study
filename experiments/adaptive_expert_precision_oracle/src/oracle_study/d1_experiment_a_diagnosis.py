"""Deterministic offline diagnostics for cached-decode D1 Experiment A.

The functions in this module are deliberately GPU-free.  They consume the
finalized quality/propagation tables and the already-materialized source policy
banks.  Terminal KL and routes after D1 are descriptive outcomes only and are
never used to select an allocation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


SCHEMA = "pr13_d1_experiment_a_offline_diagnosis_v1"
PR13_POLICY = "pr13_router_square"
RATE_PAIRS = ((360, 384), (725, 749))

TOKEN_KEYS = [
    "injection_layer",
    "group",
    "request_id",
    "position",
]
RATE_TOKEN_KEYS = ["rate_pages_per_expert", *TOKEN_KEYS]
POLICY_TOKEN_KEYS = [*RATE_TOKEN_KEYS, "policy"]


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _quantile(values: pd.Series, q: float) -> float:
    array = values.to_numpy(np.float64)
    return float(np.quantile(array, q)) if len(array) else float("nan")


def _first_crossing_distance(frame: pd.DataFrame) -> pd.Series:
    first = frame["first_route_membership_change_layer"].to_numpy(np.float64)
    layer = frame["injection_layer"].to_numpy(np.float64)
    return pd.Series(np.where(first >= 0, first - layer, -1), index=frame.index)


def crossing_conditional_effects(
    quality: pd.DataFrame,
    propagation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair every live policy with PR13 and classify its immediate D1 event."""

    _require_columns(
        quality,
        [
            *POLICY_TOKEN_KEYS,
            "route_mode",
            "logit_kl",
            "live_minus_frozen_logit_kl",
            "live_selected_local_qenergy_damage",
            "selected_group_pages",
            "first_route_membership_change_layer",
        ],
        "quality",
    )
    _require_columns(
        propagation,
        [
            *POLICY_TOKEN_KEYS,
            "route_mode",
            "distance_from_injection",
            "route_membership_change_fraction",
            "router_mass_churn",
        ],
        "propagation",
    )
    live = quality[quality["route_mode"].astype(str).eq("live")].copy()
    live["first_crossing_distance"] = _first_crossing_distance(live)
    immediate = propagation[
        propagation["route_mode"].astype(str).eq("live")
        & propagation["distance_from_injection"].eq(1)
    ][
        POLICY_TOKEN_KEYS + ["route_membership_change_fraction", "router_mass_churn"]
    ].copy()
    if immediate.duplicated(POLICY_TOKEN_KEYS).any():
        raise ValueError("propagation has duplicate live D1 identities")
    joined = live.merge(immediate, on=POLICY_TOKEN_KEYS, validate="one_to_one")

    reference_columns = [
        "logit_kl",
        "live_minus_frozen_logit_kl",
        "live_selected_local_qenergy_damage",
        "selected_group_pages",
        "first_crossing_distance",
        "route_membership_change_fraction",
        "router_mass_churn",
    ]
    reference = joined[joined["policy"].astype(str).eq(PR13_POLICY)][
        RATE_TOKEN_KEYS + reference_columns
    ].rename(columns={name: f"pr13_{name}" for name in reference_columns})
    if reference.duplicated(RATE_TOKEN_KEYS).any():
        raise ValueError("quality has duplicate live PR13 identities")
    paired = joined[~joined["policy"].astype(str).eq(PR13_POLICY)].merge(
        reference,
        on=RATE_TOKEN_KEYS,
        validate="many_to_one",
    )
    candidate_crossed = paired["route_membership_change_fraction"].to_numpy() > 0
    pr13_crossed = paired["pr13_route_membership_change_fraction"].to_numpy() > 0
    paired["crossing_event"] = np.select(
        [
            pr13_crossed & ~candidate_crossed,
            ~pr13_crossed & candidate_crossed,
            pr13_crossed & candidate_crossed,
        ],
        ["prevented", "introduced", "both_crossed"],
        default="both_safe",
    )
    paired["logit_kl_minus_pr13"] = paired["logit_kl"] - paired["pr13_logit_kl"]
    paired["live_minus_frozen_kl_minus_pr13"] = (
        paired["live_minus_frozen_logit_kl"] - paired["pr13_live_minus_frozen_logit_kl"]
    )
    paired["local_qenergy_minus_pr13"] = (
        paired["live_selected_local_qenergy_damage"]
        - paired["pr13_live_selected_local_qenergy_damage"]
    )
    paired["d1_mass_churn_minus_pr13"] = (
        paired["router_mass_churn"] - paired["pr13_router_mass_churn"]
    )
    paired = paired.sort_values(
        [*RATE_TOKEN_KEYS, "policy"], kind="stable"
    ).reset_index(drop=True)

    summaries: list[pd.DataFrame] = []
    for scope, group_columns in (
        ("all_layers", ["rate_pages_per_expert", "policy", "crossing_event"]),
        (
            "by_layer",
            [
                "rate_pages_per_expert",
                "injection_layer",
                "policy",
                "crossing_event",
            ],
        ),
    ):
        result = (
            paired.groupby(group_columns, sort=True, dropna=False)
            .agg(
                tokens=("group", "size"),
                mean_logit_kl_minus_pr13=("logit_kl_minus_pr13", "mean"),
                median_logit_kl_minus_pr13=("logit_kl_minus_pr13", "median"),
                p90_logit_kl_minus_pr13=(
                    "logit_kl_minus_pr13",
                    lambda values: _quantile(values, 0.90),
                ),
                p95_logit_kl_minus_pr13=(
                    "logit_kl_minus_pr13",
                    lambda values: _quantile(values, 0.95),
                ),
                max_logit_kl_minus_pr13=("logit_kl_minus_pr13", "max"),
                improved_token_fraction=(
                    "logit_kl_minus_pr13",
                    lambda values: float((values.to_numpy() < 0).mean()),
                ),
                mean_live_minus_frozen_kl_minus_pr13=(
                    "live_minus_frozen_kl_minus_pr13",
                    "mean",
                ),
                mean_local_qenergy_minus_pr13=(
                    "local_qenergy_minus_pr13",
                    "mean",
                ),
                mean_d1_mass_churn_minus_pr13=(
                    "d1_mass_churn_minus_pr13",
                    "mean",
                ),
            )
            .reset_index()
        )
        result.insert(0, "scope", scope)
        if "injection_layer" not in result:
            result.insert(2, "injection_layer", -1)
        summaries.append(result)
    summary = (
        pd.concat(summaries, ignore_index=True)
        .sort_values(
            [
                "scope",
                "rate_pages_per_expert",
                "injection_layer",
                "policy",
                "crossing_event",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    return paired, summary


def adjacent_rate_nonmonotonicity(
    quality: pd.DataFrame,
    rate_pairs: Sequence[tuple[int, int]] = RATE_PAIRS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair the metadata-adjusted and unadjusted caps without inventing pairs."""

    fields = [
        "logit_kl",
        "live_minus_frozen_logit_kl",
        "live_selected_local_qenergy_damage",
        "realized_injected_layer_output_mse",
        "selected_group_pages",
        "first_route_membership_change_layer",
    ]
    _require_columns(
        quality,
        [*POLICY_TOKEN_KEYS, "route_mode", *fields],
        "quality",
    )
    pair_keys = [*TOKEN_KEYS, "policy", "route_mode"]
    parts: list[pd.DataFrame] = []
    for low_rate, high_rate in rate_pairs:
        low = quality[quality["rate_pages_per_expert"].eq(int(low_rate))][
            pair_keys + fields
        ].copy()
        high = quality[quality["rate_pages_per_expert"].eq(int(high_rate))][
            pair_keys + fields
        ].copy()
        if low.duplicated(pair_keys).any() or high.duplicated(pair_keys).any():
            raise ValueError(
                f"duplicate quality identity in rate pair {low_rate}->{high_rate}"
            )
        low = low.rename(columns={name: f"low_{name}" for name in fields})
        high = high.rename(columns={name: f"high_{name}" for name in fields})
        paired = low.merge(high, on=pair_keys, validate="one_to_one")
        if len(paired) != len(low) or len(paired) != len(high):
            raise ValueError(f"incomplete quality rate pair {low_rate}->{high_rate}")
        paired.insert(0, "low_rate", int(low_rate))
        paired.insert(1, "high_rate", int(high_rate))
        paired["logit_kl_delta_high_minus_low"] = (
            paired["high_logit_kl"] - paired["low_logit_kl"]
        )
        paired["local_qenergy_delta_high_minus_low"] = (
            paired["high_live_selected_local_qenergy_damage"]
            - paired["low_live_selected_local_qenergy_damage"]
        )
        paired["injected_mse_delta_high_minus_low"] = (
            paired["high_realized_injected_layer_output_mse"]
            - paired["low_realized_injected_layer_output_mse"]
        )
        paired["live_minus_frozen_kl_delta_high_minus_low"] = (
            paired["high_live_minus_frozen_logit_kl"]
            - paired["low_live_minus_frozen_logit_kl"]
        )
        paired["pages_delta_high_minus_low"] = (
            paired["high_selected_group_pages"] - paired["low_selected_group_pages"]
        )
        paired["local_improved"] = (
            paired["local_qenergy_delta_high_minus_low"].to_numpy() < 0
        )
        paired["kl_worsened"] = paired["logit_kl_delta_high_minus_low"].to_numpy() > 0
        paired["local_improved_but_kl_worsened"] = (
            paired["local_improved"] & paired["kl_worsened"]
        )
        parts.append(paired)
    rows = (
        pd.concat(parts, ignore_index=True)
        .sort_values(
            ["low_rate", "injection_layer", "group", "policy", "route_mode"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    summaries: list[pd.DataFrame] = []
    for scope, group_columns in (
        ("all_layers", ["low_rate", "high_rate", "policy", "route_mode"]),
        (
            "by_layer",
            [
                "low_rate",
                "high_rate",
                "injection_layer",
                "policy",
                "route_mode",
            ],
        ),
    ):
        result = (
            rows.groupby(group_columns, sort=True, dropna=False)
            .agg(
                tokens=("group", "size"),
                mean_logit_kl_delta=("logit_kl_delta_high_minus_low", "mean"),
                median_logit_kl_delta=("logit_kl_delta_high_minus_low", "median"),
                p90_logit_kl_delta=(
                    "logit_kl_delta_high_minus_low",
                    lambda values: _quantile(values, 0.90),
                ),
                p95_logit_kl_delta=(
                    "logit_kl_delta_high_minus_low",
                    lambda values: _quantile(values, 0.95),
                ),
                max_logit_kl_delta=("logit_kl_delta_high_minus_low", "max"),
                mean_local_qenergy_delta=(
                    "local_qenergy_delta_high_minus_low",
                    "mean",
                ),
                mean_injected_mse_delta=(
                    "injected_mse_delta_high_minus_low",
                    "mean",
                ),
                mean_live_minus_frozen_kl_delta=(
                    "live_minus_frozen_kl_delta_high_minus_low",
                    "mean",
                ),
                local_improved_fraction=("local_improved", "mean"),
                kl_worsened_fraction=("kl_worsened", "mean"),
                paradox_fraction=("local_improved_but_kl_worsened", "mean"),
                positive_kl_damage=(
                    "logit_kl_delta_high_minus_low",
                    lambda values: float(
                        np.maximum(values.to_numpy(np.float64), 0).sum()
                    ),
                ),
            )
            .reset_index()
        )
        result.insert(0, "scope", scope)
        if "injection_layer" not in result:
            result.insert(3, "injection_layer", -1)
        summaries.append(result)
    summary = (
        pd.concat(summaries, ignore_index=True)
        .sort_values(
            ["scope", "low_rate", "injection_layer", "policy", "route_mode"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    return rows, summary


def _positive_share(array: np.ndarray, count: int) -> float:
    positive = np.sort(array[array > 0])[::-1]
    total = float(positive.sum())
    if total == 0:
        return 0.0
    return float(positive[:count].sum() / total)


def heavy_tail_summaries(
    crossing_rows: pd.DataFrame,
    adjacent_rows: pd.DataFrame,
    *,
    top_events: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize concentration of regressions for policy and cap comparisons."""

    _require_columns(
        crossing_rows,
        [*RATE_TOKEN_KEYS, "policy", "logit_kl_minus_pr13"],
        "crossing_rows",
    )
    _require_columns(
        adjacent_rows,
        [
            "low_rate",
            "high_rate",
            *TOKEN_KEYS,
            "policy",
            "route_mode",
            "logit_kl_delta_high_minus_low",
        ],
        "adjacent_rows",
    )
    policy_events = crossing_rows[
        RATE_TOKEN_KEYS + ["policy", "crossing_event", "logit_kl_minus_pr13"]
    ].copy()
    policy_events.insert(0, "comparison_kind", "policy_minus_pr13")
    policy_events["comparison_rate_low"] = policy_events["rate_pages_per_expert"]
    policy_events["comparison_rate_high"] = policy_events["rate_pages_per_expert"]
    policy_events["route_mode"] = "live"
    policy_events["damage"] = policy_events["logit_kl_minus_pr13"]

    rate_events = adjacent_rows[
        [
            "low_rate",
            "high_rate",
            *TOKEN_KEYS,
            "policy",
            "route_mode",
            "logit_kl_delta_high_minus_low",
        ]
    ].copy()
    rate_events.insert(0, "comparison_kind", "higher_rate_minus_lower_rate")
    rate_events = rate_events.rename(
        columns={
            "low_rate": "comparison_rate_low",
            "high_rate": "comparison_rate_high",
        }
    )
    rate_events["rate_pages_per_expert"] = rate_events["comparison_rate_high"]
    rate_events["crossing_event"] = "not_applicable"
    rate_events["damage"] = rate_events["logit_kl_delta_high_minus_low"]
    events = pd.concat(
        [
            policy_events[
                [
                    "comparison_kind",
                    "comparison_rate_low",
                    "comparison_rate_high",
                    "rate_pages_per_expert",
                    *TOKEN_KEYS,
                    "policy",
                    "route_mode",
                    "crossing_event",
                    "damage",
                ]
            ],
            rate_events[
                [
                    "comparison_kind",
                    "comparison_rate_low",
                    "comparison_rate_high",
                    "rate_pages_per_expert",
                    *TOKEN_KEYS,
                    "policy",
                    "route_mode",
                    "crossing_event",
                    "damage",
                ]
            ],
        ],
        ignore_index=True,
    )
    base_groups = [
        "comparison_kind",
        "comparison_rate_low",
        "comparison_rate_high",
        "policy",
        "route_mode",
    ]
    summary_parts: list[pd.DataFrame] = []
    for scope, groups in (
        ("all_layers", base_groups),
        ("by_layer", [*base_groups, "injection_layer"]),
    ):
        records: list[dict[str, Any]] = []
        for keys, part in events.groupby(groups, sort=True, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            record = dict(zip(groups, keys, strict=True))
            values = part["damage"].to_numpy(np.float64)
            record.update(
                {
                    "scope": scope,
                    "injection_layer": int(record.get("injection_layer", -1)),
                    "events": int(len(values)),
                    "mean_damage": float(values.mean()),
                    "median_damage": float(np.quantile(values, 0.50)),
                    "p90_damage": float(np.quantile(values, 0.90)),
                    "p95_damage": float(np.quantile(values, 0.95)),
                    "p99_damage": float(np.quantile(values, 0.99)),
                    "minimum_damage": float(values.min()),
                    "maximum_damage": float(values.max()),
                    "positive_fraction": float((values > 0).mean()),
                    "positive_damage_sum": float(np.maximum(values, 0).sum()),
                    "top1_positive_share": _positive_share(values, 1),
                    "top3_positive_share": _positive_share(values, 3),
                    "top5_positive_share": _positive_share(values, 5),
                }
            )
            records.append(record)
        summary_parts.append(pd.DataFrame.from_records(records))
    summary = (
        pd.concat(summary_parts, ignore_index=True)
        .sort_values(
            [
                "scope",
                "comparison_kind",
                "comparison_rate_low",
                "injection_layer",
                "policy",
                "route_mode",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    rank_groups = base_groups
    ranked = events.sort_values(
        [
            *rank_groups,
            "damage",
            "injection_layer",
            "request_id",
            "position",
            "group",
        ],
        ascending=[True] * len(rank_groups) + [False, True, True, True, True],
        kind="stable",
    ).copy()
    ranked["damage_rank"] = ranked.groupby(rank_groups, sort=False).cumcount() + 1
    ranked = ranked[ranked["damage_rank"].le(int(top_events))].reset_index(drop=True)
    return summary, ranked


def _live_d1_candidates(
    quality: pd.DataFrame,
    propagation: pd.DataFrame,
) -> pd.DataFrame:
    """Join live terminal outcomes to exact same-host D1 observations."""

    quality_fields = [
        "logit_kl",
        "live_minus_frozen_logit_kl",
        "live_selected_local_qenergy_damage",
        "selected_group_pages",
        "realized_injected_layer_output_mse",
    ]
    d1_fields = ["route_membership_change_fraction", "router_mass_churn"]
    _require_columns(
        quality, [*POLICY_TOKEN_KEYS, "route_mode", *quality_fields], "quality"
    )
    _require_columns(
        propagation,
        [
            *POLICY_TOKEN_KEYS,
            "route_mode",
            "distance_from_injection",
            *d1_fields,
        ],
        "propagation",
    )
    live_quality = quality[quality["route_mode"].astype(str).eq("live")][
        POLICY_TOKEN_KEYS + quality_fields
    ].copy()
    d1 = propagation[
        propagation["route_mode"].astype(str).eq("live")
        & propagation["distance_from_injection"].eq(1)
    ][POLICY_TOKEN_KEYS + d1_fields].copy()
    if live_quality.duplicated(POLICY_TOKEN_KEYS).any():
        raise ValueError("quality has duplicate live policy identities")
    if d1.duplicated(POLICY_TOKEN_KEYS).any():
        raise ValueError("propagation has duplicate live D1 identities")
    candidates = live_quality.merge(d1, on=POLICY_TOKEN_KEYS, validate="one_to_one")
    if len(candidates) != len(live_quality):
        raise ValueError("not every live quality row has an exact D1 observation")
    candidates["d1_crossed"] = (
        candidates["route_membership_change_fraction"].to_numpy(np.float64) > 0
    )
    return candidates


def _pr13_reference_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    reference = candidates[candidates["policy"].astype(str).eq(PR13_POLICY)].copy()
    if reference.duplicated(RATE_TOKEN_KEYS).any():
        raise ValueError("candidate grid has duplicate PR13 identities")
    if len(reference) != len(candidates[RATE_TOKEN_KEYS].drop_duplicates()):
        raise ValueError("candidate grid is missing a PR13 incumbent")
    return reference


def _choose_same_rate_remedies(candidates: pd.DataFrame) -> pd.DataFrame:
    """Apply the two D1-only target ablations independently at each rate."""

    records: list[pd.Series] = []
    order = [
        "route_membership_change_fraction",
        "router_mass_churn",
        "live_selected_local_qenergy_damage",
        "absolute_page_change_from_incumbent",
        "policy",
    ]
    for _, part in candidates.groupby(RATE_TOKEN_KEYS, sort=True, dropna=False):
        part = part.copy()
        incumbent_rows = part[part["policy"].astype(str).eq(PR13_POLICY)]
        if len(incumbent_rows) != 1:
            raise ValueError("each token/rate group must have exactly one PR13 row")
        incumbent = incumbent_rows.iloc[0]
        part["absolute_page_change_from_incumbent"] = np.abs(
            part["selected_group_pages"].to_numpy(np.int64)
            - int(incumbent["selected_group_pages"])
        )
        incumbent = incumbent.copy()
        incumbent["absolute_page_change_from_incumbent"] = 0
        if not bool(incumbent["d1_crossed"]):
            repair = incumbent.copy()
            eligible_repairs = 0
            reason = "safe_incumbent_held"
        else:
            eligible = part[
                part["route_membership_change_fraction"].to_numpy(np.float64)
                < float(incumbent["route_membership_change_fraction"])
            ]
            eligible_repairs = int(len(eligible))
            if eligible.empty:
                repair = incumbent.copy()
                reason = "no_strict_d1_repair"
            else:
                repair = eligible.sort_values(order, kind="stable").iloc[0].copy()
                reason = "strictly_fewer_d1_crossings"
        repair["selector"] = "strict_pr13_incumbent_repair"
        repair["selector_reason"] = reason
        repair["eligible_repair_candidates"] = eligible_repairs
        repair["candidates_considered"] = int(len(part))
        repair["selection_rate"] = int(incumbent["rate_pages_per_expert"])
        records.append(repair)

        global_choice = part.sort_values(order, kind="stable").iloc[0].copy()
        global_choice["selector"] = "global_hard_d1_local"
        global_choice["selector_reason"] = "global_lexicographic_minimum"
        global_choice["eligible_repair_candidates"] = int(
            (
                part["route_membership_change_fraction"].to_numpy(np.float64)
                < float(incumbent["route_membership_change_fraction"])
            ).sum()
        )
        global_choice["candidates_considered"] = int(len(part))
        global_choice["selection_rate"] = int(incumbent["rate_pages_per_expert"])
        records.append(global_choice)
    return pd.DataFrame.from_records(records).reset_index(drop=True)


def _choose_adjacent_cap_remedies(
    candidates: pd.DataFrame,
    rate_pairs: Sequence[tuple[int, int]],
) -> pd.DataFrame:
    """Choose from adjacent low/high banks under the high-rate page cap."""

    records: list[pd.Series] = []
    order = [
        "route_membership_change_fraction",
        "router_mass_churn",
        "live_selected_local_qenergy_damage",
        "absolute_page_change_from_incumbent",
        "rate_pages_per_expert",
        "policy",
    ]
    for low_rate, high_rate in rate_pairs:
        pool = candidates[
            candidates["rate_pages_per_expert"].isin([int(low_rate), int(high_rate)])
        ].copy()
        for _, part in pool.groupby(TOKEN_KEYS, sort=True, dropna=False):
            incumbent_rows = part[
                part["rate_pages_per_expert"].eq(int(high_rate))
                & part["policy"].astype(str).eq(PR13_POLICY)
            ]
            if len(incumbent_rows) != 1:
                raise ValueError("adjacent cap group has no unique high-rate PR13 row")
            incumbent = incumbent_rows.iloc[0]
            page_cap = int(incumbent["selected_group_pages"])
            eligible = part[part["selected_group_pages"].le(page_cap)].copy()
            if eligible.empty:
                raise ValueError("adjacent cap selector has no candidate under its cap")
            eligible["absolute_page_change_from_incumbent"] = np.abs(
                eligible["selected_group_pages"].to_numpy(np.int64) - page_cap
            )
            choice = eligible.sort_values(order, kind="stable").iloc[0].copy()
            choice["selector"] = "adjacent_rate_hard_d1_early_stop"
            choice["selector_reason"] = "global_lexicographic_minimum_under_cap"
            choice["eligible_repair_candidates"] = int(
                (
                    eligible["route_membership_change_fraction"].to_numpy(np.float64)
                    < float(incumbent["route_membership_change_fraction"])
                ).sum()
            )
            choice["candidates_considered"] = int(len(eligible))
            choice["selection_rate"] = int(high_rate)
            records.append(choice)
    return pd.DataFrame.from_records(records).reset_index(drop=True)


def _attach_remedy_reference(
    choices: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    comparison_kind: str,
    reference_rate_by_selection_rate: Mapping[int, int],
) -> pd.DataFrame:
    """Attach PR13 outcomes after selection, preserving the no-KL-leak boundary."""

    parts: list[pd.DataFrame] = []
    outcome_fields = [
        "logit_kl",
        "live_minus_frozen_logit_kl",
        "live_selected_local_qenergy_damage",
        "selected_group_pages",
        "realized_injected_layer_output_mse",
        "route_membership_change_fraction",
        "router_mass_churn",
        "d1_crossed",
    ]
    for selection_rate, reference_rate in sorted(
        reference_rate_by_selection_rate.items()
    ):
        source = choices[choices["selection_rate"].eq(int(selection_rate))]
        if source.empty:
            continue
        selected = source.copy()
        ref = reference[reference["rate_pages_per_expert"].eq(int(reference_rate))][
            TOKEN_KEYS + outcome_fields
        ].rename(columns={name: f"reference_{name}" for name in outcome_fields})
        if ref.duplicated(TOKEN_KEYS).any():
            raise ValueError("PR13 comparison reference has duplicate token identities")
        selected = selected.merge(ref, on=TOKEN_KEYS, validate="many_to_one")
        if len(selected) != len(source):
            raise ValueError("PR13 comparison reference has an incomplete token grid")
        selected = selected.rename(
            columns={
                "rate_pages_per_expert": "selected_source_rate",
                "policy": "selected_policy",
            }
        )
        selected["comparison_kind"] = str(comparison_kind)
        selected["reference_rate"] = int(reference_rate)
        selected["reference_policy"] = PR13_POLICY
        selected["selected_lower_rate"] = selected["selected_source_rate"].lt(
            int(reference_rate)
        )
        selected["selected_pr13"] = selected["selected_policy"].astype(str).eq(
            PR13_POLICY
        ) & selected["selected_source_rate"].eq(int(reference_rate))
        selected["crossing_prevented"] = selected["reference_d1_crossed"].astype(
            bool
        ) & ~selected["d1_crossed"].astype(bool)
        selected["crossing_introduced"] = ~selected["reference_d1_crossed"].astype(
            bool
        ) & selected["d1_crossed"].astype(bool)
        for name in (
            "logit_kl",
            "live_minus_frozen_logit_kl",
            "live_selected_local_qenergy_damage",
            "selected_group_pages",
            "realized_injected_layer_output_mse",
            "route_membership_change_fraction",
            "router_mass_churn",
        ):
            selected[f"{name}_minus_reference"] = (
                selected[name] - selected[f"reference_{name}"]
            )
        selected["terminal_kl_used_for_selection"] = False
        parts.append(selected)
    if not parts:
        raise ValueError(f"empty remedy comparison: {comparison_kind}")
    return pd.concat(parts, ignore_index=True)


def _remedy_summaries(rows: pd.DataFrame) -> pd.DataFrame:
    base = ["comparison_kind", "selector", "selection_rate", "reference_rate"]
    parts: list[pd.DataFrame] = []
    for scope, groups in (
        ("all_layers_requests", base),
        ("by_layer", [*base, "injection_layer"]),
        ("by_request", [*base, "request_id"]),
        ("by_layer_request", [*base, "injection_layer", "request_id"]),
    ):
        result = (
            rows.groupby(groups, sort=True, dropna=False)
            .agg(
                tokens=("group", "size"),
                mean_logit_kl=("logit_kl", "mean"),
                reference_mean_logit_kl=("reference_logit_kl", "mean"),
                mean_logit_kl_minus_reference=("logit_kl_minus_reference", "mean"),
                median_logit_kl_minus_reference=("logit_kl_minus_reference", "median"),
                p95_logit_kl_minus_reference=(
                    "logit_kl_minus_reference",
                    lambda values: _quantile(values, 0.95),
                ),
                maximum_logit_kl_minus_reference=("logit_kl_minus_reference", "max"),
                improved_token_fraction=(
                    "logit_kl_minus_reference",
                    lambda values: float((values.to_numpy(np.float64) < 0).mean()),
                ),
                mean_live_minus_frozen_logit_kl=("live_minus_frozen_logit_kl", "mean"),
                mean_live_minus_frozen_kl_minus_reference=(
                    "live_minus_frozen_logit_kl_minus_reference",
                    "mean",
                ),
                selected_d1_crossing_tokens=("d1_crossed", "sum"),
                reference_d1_crossing_tokens=("reference_d1_crossed", "sum"),
                crossings_prevented=("crossing_prevented", "sum"),
                crossings_introduced=("crossing_introduced", "sum"),
                mean_d1_membership_change=("route_membership_change_fraction", "mean"),
                mean_d1_membership_change_minus_reference=(
                    "route_membership_change_fraction_minus_reference",
                    "mean",
                ),
                mean_d1_mass_churn=("router_mass_churn", "mean"),
                mean_d1_mass_churn_minus_reference=(
                    "router_mass_churn_minus_reference",
                    "mean",
                ),
                mean_live_local_qenergy=("live_selected_local_qenergy_damage", "mean"),
                mean_live_local_qenergy_minus_reference=(
                    "live_selected_local_qenergy_damage_minus_reference",
                    "mean",
                ),
                mean_selected_pages=("selected_group_pages", "mean"),
                mean_selected_pages_minus_reference=(
                    "selected_group_pages_minus_reference",
                    "mean",
                ),
                selected_lower_rate_fraction=("selected_lower_rate", "mean"),
                selected_pr13_fraction=("selected_pr13", "mean"),
            )
            .reset_index()
        )
        result.insert(0, "scope", scope)
        if "injection_layer" not in result:
            result.insert(5, "injection_layer", -1)
        if "request_id" not in result:
            result.insert(6, "request_id", "all")
        parts.append(result)
    summary = pd.concat(parts, ignore_index=True)
    return summary.sort_values(
        [
            "scope",
            "comparison_kind",
            "selector",
            "selection_rate",
            "reference_rate",
            "injection_layer",
            "request_id",
        ],
        kind="stable",
    ).reset_index(drop=True)


def deployable_d1_target_remedies(
    quality: pd.DataFrame,
    propagation: pd.DataFrame,
    rate_pairs: Sequence[tuple[int, int]] = RATE_PAIRS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate D1-only target remedies over the five executed policies.

    Exact full-model D1 membership and mass plus live local qenergy are
    selection inputs. Terminal KL and live-minus-frozen KL are attached only
    after each choice, making this a target ablation rather than a terminal
    quality oracle.
    """

    candidates = _live_d1_candidates(quality, propagation)
    reference = _pr13_reference_rows(candidates)
    same_rate_choices = _choose_same_rate_remedies(candidates)
    rates = sorted(map(int, candidates["rate_pages_per_expert"].unique()))
    rows = [
        _attach_remedy_reference(
            same_rate_choices,
            reference,
            comparison_kind="same_numeric_rate",
            reference_rate_by_selection_rate={rate: rate for rate in rates},
        )
    ]
    rows.append(
        _attach_remedy_reference(
            same_rate_choices,
            reference,
            comparison_kind="metadata_matched_low_vs_high_pr13",
            reference_rate_by_selection_rate={
                int(low): int(high) for low, high in rate_pairs
            },
        )
    )
    cap_choices = _choose_adjacent_cap_remedies(candidates, rate_pairs)
    rows.append(
        _attach_remedy_reference(
            cap_choices,
            reference,
            comparison_kind="adjacent_rate_early_stop_cap",
            reference_rate_by_selection_rate={
                int(high): int(high) for _, high in rate_pairs
            },
        )
    )
    selected = (
        pd.concat(rows, ignore_index=True)
        .sort_values(
            [
                "comparison_kind",
                "selector",
                "selection_rate",
                "injection_layer",
                "group",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    return selected, _remedy_summaries(selected)


_BIT_COUNT = np.asarray([0, 1, 1, 2, 1, 2, 2, 3], dtype=np.int8)


def source_adjacent_rate_geometry(
    banks: Mapping[tuple[int, int], Any],
    rate_pairs: Sequence[tuple[int, int]] = RATE_PAIRS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Measure state nesting and combined-delta direction across adjacent caps.

    ``banks`` values follow :class:`DecodePolicyBank`'s public attributes.  A
    structural protocol is used so focused tests can supply tiny synthetic
    banks without constructing model artifacts.
    """

    expert_records: list[dict[str, Any]] = []
    delta_records: list[dict[str, Any]] = []
    layers = sorted({int(layer) for layer, _ in banks})
    for low_rate, high_rate in rate_pairs:
        for layer in layers:
            if (layer, int(low_rate)) not in banks or (
                layer,
                int(high_rate),
            ) not in banks:
                continue
            low = banks[(layer, int(low_rate))]
            high = banks[(layer, int(high_rate))]
            low_identity = low.identity.sort_values("group", kind="stable").reset_index(
                drop=True
            )
            high_identity = high.identity.sort_values(
                "group", kind="stable"
            ).reset_index(drop=True)
            identity_columns = ["group", "request_id", "position"]
            if not low_identity[identity_columns].equals(
                high_identity[identity_columns]
            ):
                raise ValueError(
                    f"source identities differ at layer {layer}, pair {low_rate}->{high_rate}"
                )
            if not np.array_equal(low.expert_ids, high.expert_ids):
                raise ValueError(
                    f"source expert IDs differ at layer {layer}, pair {low_rate}->{high_rate}"
                )
            policies = sorted(
                set(low.selected_states).intersection(high.selected_states)
            )
            if not policies:
                raise ValueError("source policy banks have no common policies")
            for policy in policies:
                low_states = np.asarray(low.selected_states[policy], np.int8)
                high_states = np.asarray(high.selected_states[policy], np.int8)
                if low_states.shape != high_states.shape or low_states.ndim != 3:
                    raise ValueError(f"source state shape differs for {policy}")
                if low_states.shape[2] != 512:
                    raise ValueError(f"source state vector width changed for {policy}")
                removed = low_states & (~high_states & 7)
                added = high_states & (~low_states & 7)
                low_pages = _BIT_COUNT[low_states].sum(axis=2)
                high_pages = _BIT_COUNT[high_states].sum(axis=2)
                removed_bits = _BIT_COUNT[removed].sum(axis=2)
                added_bits = _BIT_COUNT[added].sum(axis=2)
                changed_units = (low_states != high_states).sum(axis=2)
                removed_units = (removed != 0).sum(axis=2)
                groups, experts, _ = low_states.shape
                for group in range(groups):
                    identity = low_identity.iloc[group]
                    for rank in range(experts):
                        expert_records.append(
                            {
                                "low_rate": int(low_rate),
                                "high_rate": int(high_rate),
                                "injection_layer": int(layer),
                                "group": int(identity["group"]),
                                "request_id": str(identity["request_id"]),
                                "position": int(identity["position"]),
                                "policy": str(policy),
                                "router_rank": int(rank + 1),
                                "expert_id": int(low.expert_ids[group, rank]),
                                "low_selected_pages": int(low_pages[group, rank]),
                                "high_selected_pages": int(high_pages[group, rank]),
                                "page_delta_high_minus_low": int(
                                    high_pages[group, rank] - low_pages[group, rank]
                                ),
                                "removed_projection_bits": int(
                                    removed_bits[group, rank]
                                ),
                                "added_projection_bits": int(added_bits[group, rank]),
                                "changed_state_units": int(changed_units[group, rank]),
                                "units_with_removed_bits": int(
                                    removed_units[group, rank]
                                ),
                                "expert_is_nested": bool(
                                    removed_bits[group, rank] == 0
                                ),
                            }
                        )
                low_delta = np.asarray(low.deltas[policy], np.float64)
                high_delta = np.asarray(high.deltas[policy], np.float64)
                if low_delta.shape != high_delta.shape or low_delta.shape[0] != groups:
                    raise ValueError(f"source delta shape differs for {policy}")
                for group in range(groups):
                    identity = low_identity.iloc[group]
                    lo = low_delta[group]
                    hi = high_delta[group]
                    increment = hi - lo
                    lo_norm = float(np.linalg.norm(lo))
                    hi_norm = float(np.linalg.norm(hi))
                    increment_norm = float(np.linalg.norm(increment))
                    cosine = (
                        float(np.dot(lo, hi) / (lo_norm * hi_norm))
                        if lo_norm > 0 and hi_norm > 0
                        else float("nan")
                    )
                    increment_alignment = (
                        float(np.dot(lo, increment) / (lo_norm * increment_norm))
                        if lo_norm > 0 and increment_norm > 0
                        else float("nan")
                    )
                    denominator = lo_norm + increment_norm
                    cancellation = (
                        float(1.0 - hi_norm / denominator) if denominator > 0 else 0.0
                    )
                    delta_records.append(
                        {
                            "low_rate": int(low_rate),
                            "high_rate": int(high_rate),
                            "injection_layer": int(layer),
                            "group": int(identity["group"]),
                            "request_id": str(identity["request_id"]),
                            "position": int(identity["position"]),
                            "policy": str(policy),
                            "low_delta_norm": lo_norm,
                            "high_delta_norm": hi_norm,
                            "delta_norm_ratio_high_over_low": (
                                float(hi_norm / lo_norm)
                                if lo_norm > 0
                                else float("nan")
                            ),
                            "adjacent_delta_cosine": cosine,
                            "increment_norm": increment_norm,
                            "increment_over_low_norm": (
                                float(increment_norm / lo_norm)
                                if lo_norm > 0
                                else float("nan")
                            ),
                            "increment_alignment_with_low": increment_alignment,
                            "triangle_cancellation_fraction": cancellation,
                            "high_delta_norm_is_lower": bool(hi_norm < lo_norm),
                        }
                    )
    expert_rows = (
        pd.DataFrame.from_records(expert_records)
        .sort_values(
            ["low_rate", "injection_layer", "group", "policy", "router_rank"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    delta_rows = (
        pd.DataFrame.from_records(delta_records)
        .sort_values(["low_rate", "injection_layer", "group", "policy"], kind="stable")
        .reset_index(drop=True)
    )
    if expert_rows.empty or delta_rows.empty:
        raise ValueError("source geometry grid is empty")
    expert_summary = (
        expert_rows.groupby(
            ["low_rate", "high_rate", "injection_layer", "policy"], sort=True
        )
        .agg(
            experts=("expert_id", "size"),
            experts_with_page_decrease_fraction=(
                "page_delta_high_minus_low",
                lambda values: float((values.to_numpy() < 0).mean()),
            ),
            non_nested_expert_fraction=(
                "expert_is_nested",
                lambda values: float((~values.to_numpy(bool)).mean()),
            ),
            removed_projection_bits=("removed_projection_bits", "sum"),
            added_projection_bits=("added_projection_bits", "sum"),
            changed_state_units=("changed_state_units", "sum"),
            mean_page_delta=("page_delta_high_minus_low", "mean"),
        )
        .reset_index()
    )
    delta_summary = (
        delta_rows.groupby(
            ["low_rate", "high_rate", "injection_layer", "policy"], sort=True
        )
        .agg(
            tokens=("group", "size"),
            mean_adjacent_delta_cosine=("adjacent_delta_cosine", "mean"),
            p10_adjacent_delta_cosine=(
                "adjacent_delta_cosine",
                lambda values: _quantile(values.dropna(), 0.10),
            ),
            minimum_adjacent_delta_cosine=("adjacent_delta_cosine", "min"),
            mean_increment_over_low_norm=("increment_over_low_norm", "mean"),
            mean_increment_alignment_with_low=(
                "increment_alignment_with_low",
                "mean",
            ),
            mean_triangle_cancellation_fraction=(
                "triangle_cancellation_fraction",
                "mean",
            ),
            high_delta_norm_lower_fraction=("high_delta_norm_is_lower", "mean"),
        )
        .reset_index()
    )
    summary = expert_summary.merge(
        delta_summary,
        on=["low_rate", "high_rate", "injection_layer", "policy"],
        validate="one_to_one",
    )
    summary["removed_projection_bit_fraction"] = summary["removed_projection_bits"] / (
        summary["experts"] * 512 * 3
    )
    return expert_rows, delta_rows, summary
