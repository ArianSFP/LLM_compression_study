"""Offline source-slice to live full-model D1 transfer diagnostics.

This module deliberately performs no policy selection.  It asks whether the
source-cell D1 labels used to construct a policy transfer to the finalized
same-host full-model cached-decode execution.  Terminal KL is retained only
for the narrow, predeclared patched-versus-unpatched fixed-policy audit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


SCHEMA = "pr13_d1_source_live_transfer_diagnosis_v1"
PR13_POLICY = "pr13_router_square"
FIXED_POLICY = "calibration_selected_fixed_d1"
POLICIES = (
    PR13_POLICY,
    "exact_combined_local",
    FIXED_POLICY,
    "frontier_companion_d1",
    "exact_token_oracle",
)
THRESHOLDS = (0.0, 1.0 / 64.0, 2.0 / 64.0, 4.0 / 64.0)

TOKEN_KEYS = ["injection_layer", "group", "request_id", "position"]
RATE_TOKEN_KEYS = ["rate_pages_per_expert", *TOKEN_KEYS]
POLICY_TOKEN_KEYS = [*RATE_TOKEN_KEYS, "policy"]


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _safe_fraction(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _correlation(left: pd.Series, right: pd.Series) -> float:
    x = left.to_numpy(np.float64)
    y = right.to_numpy(np.float64)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(left: pd.Series, right: pd.Series) -> float:
    return _correlation(left.rank(method="average"), right.rank(method="average"))


def binary_roc_auc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    """Return tie-aware ROC AUC without a scikit-learn dependency."""

    truth = np.asarray(labels, dtype=bool)
    values = np.asarray(scores, dtype=np.float64)
    if truth.shape != values.shape:
        raise ValueError("labels and scores must have the same shape")
    valid = np.isfinite(values)
    truth = truth[valid]
    values = values[valid]
    positive = int(truth.sum())
    negative = int((~truth).sum())
    if positive == 0 or negative == 0:
        return float("nan")
    ranks = pd.Series(values).rank(method="average").to_numpy(np.float64)
    rank_sum = float(ranks[truth].sum())
    return float((rank_sum - positive * (positive + 1) / 2.0) / (positive * negative))


def source_policy_name(
    provenance: Mapping[str, Any],
    policy: str,
    group: int,
) -> str:
    """Map a finalized tail policy to its raw source-cell policy name."""

    if policy == PR13_POLICY:
        return str(provenance["pr13_source_policy"])
    if policy == "exact_combined_local":
        return str(provenance["local_source_policy"])
    if policy == FIXED_POLICY:
        return str(provenance["fixed_d1_source_policy"])
    if policy == "frontier_companion_d1":
        return str(provenance["companion_d1_source_policy"])
    if policy == "exact_token_oracle":
        by_group = provenance["exact_token_oracle_source_policy_by_group"]
        return str(by_group[str(int(group))])
    raise KeyError(f"unknown finalized policy: {policy}")


def materialize_source_labels(
    manifest: Mapping[str, Any],
    source_cells: Mapping[tuple[int, int], pd.DataFrame],
    patch_groups: pd.DataFrame,
) -> pd.DataFrame:
    """Materialize one effective source D1 label for every policy/token cell.

    Raw source rows describe the original 3090 slice.  For the twelve
    source/full-model Q4 route mismatches, the effective allocation and its
    source-side label came from the checked-in same-host patch; those rows
    override the raw allocation labels while retaining the raw baseline margin
    in separate audit columns.
    """

    patch_required = [
        "layer",
        "rate_pages_per_expert",
        "group",
        "request_id",
        "position",
        "policy",
        "exact_d1_crossed",
        "baseline_rank8_rank9_margin",
        "candidate_labeled_set_margin",
    ]
    _require_columns(patch_groups, patch_required, "patch_groups")
    patch = patch_groups.copy()
    patch_key = [
        "layer",
        "rate_pages_per_expert",
        "group",
        "request_id",
        "position",
        "policy",
    ]
    if patch.duplicated(patch_key).any():
        raise ValueError("patch_groups has duplicate policy identities")
    patch_lookup = patch.set_index(patch_key)
    patched_identities = set(
        map(
            tuple,
            patch[["layer", "group", "request_id", "position"]]
            .drop_duplicates()
            .itertuples(index=False, name=None),
        )
    )

    rows: list[dict[str, Any]] = []
    cells = manifest.get("cells", {})
    for (layer, rate), frame in sorted(source_cells.items()):
        cell_key = f"layer_{int(layer):02d}_rate_{int(rate)}"
        if cell_key not in cells:
            raise KeyError(f"source cell absent from manifest: {cell_key}")
        provenance = cells[cell_key]["policy_provenance"]
        required = [
            "layer",
            "rate_pages_per_expert",
            "group",
            "request_id",
            "position",
            "policy",
            "exact_d1_crossed",
            "baseline_rank8_rank9_margin",
            "candidate_labeled_set_margin",
            "membership_pairs_changed",
            "routing_mass_lost",
        ]
        _require_columns(frame, required, f"source cell {cell_key}")
        lookup = frame.set_index(["group", "policy"])
        groups = (
            frame[["group", "request_id", "position"]]
            .drop_duplicates()
            .sort_values("group", kind="stable")
        )
        if groups["group"].duplicated().any():
            raise ValueError(f"source group identity is ambiguous: {cell_key}")
        for identity in groups.itertuples(index=False):
            group = int(identity.group)
            request_id = str(identity.request_id)
            position = int(identity.position)
            identity_key = (int(layer), group, request_id, position)
            for policy in POLICIES:
                source_policy = source_policy_name(provenance, policy, group)
                try:
                    raw = lookup.loc[(group, source_policy)]
                except KeyError as exc:
                    raise KeyError(
                        f"missing raw source row: {cell_key}, group={group}, "
                        f"policy={source_policy}"
                    ) from exc
                if isinstance(raw, pd.DataFrame):
                    raise ValueError(
                        f"duplicate raw source policy row: {cell_key}, "
                        f"group={group}, policy={source_policy}"
                    )
                record: dict[str, Any] = {
                    "schema": SCHEMA,
                    "injection_layer": int(layer),
                    "rate_pages_per_expert": int(rate),
                    "group": group,
                    "request_id": request_id,
                    "position": position,
                    "policy": policy,
                    "raw_source_policy": source_policy,
                    "allocation_patched": identity_key in patched_identities,
                    "source_metric_origin": "raw_source_cell",
                    "raw_source_exact_d1_crossed": bool(raw["exact_d1_crossed"]),
                    "raw_source_baseline_rank8_rank9_margin": float(
                        raw["baseline_rank8_rank9_margin"]
                    ),
                    "raw_source_candidate_labeled_set_margin": float(
                        raw["candidate_labeled_set_margin"]
                    ),
                    "source_exact_d1_crossed": bool(raw["exact_d1_crossed"]),
                    "source_baseline_rank8_rank9_margin": float(
                        raw["baseline_rank8_rank9_margin"]
                    ),
                    "source_candidate_labeled_set_margin": float(
                        raw["candidate_labeled_set_margin"]
                    ),
                    "source_membership_pairs_changed": int(
                        raw["membership_pairs_changed"]
                    ),
                    "source_routing_mass_lost": float(raw["routing_mass_lost"]),
                }
                patch_key_value = (
                    int(layer),
                    int(rate),
                    group,
                    request_id,
                    position,
                    policy,
                )
                if patch_key_value in patch_lookup.index:
                    replacement = patch_lookup.loc[patch_key_value]
                    record.update(
                        {
                            "source_metric_origin": "same_host_patch",
                            "source_exact_d1_crossed": bool(
                                replacement["exact_d1_crossed"]
                            ),
                            "source_baseline_rank8_rank9_margin": float(
                                replacement["baseline_rank8_rank9_margin"]
                            ),
                            "source_candidate_labeled_set_margin": float(
                                replacement["candidate_labeled_set_margin"]
                            ),
                            "source_membership_pairs_changed": int(
                                replacement["membership_pairs_changed"]
                            ),
                            "source_routing_mass_lost": float(
                                replacement["routing_mass_lost"]
                            ),
                        }
                    )
                elif identity_key in patched_identities:
                    raise ValueError(
                        "patched allocation identity lacks a policy row: "
                        f"{patch_key_value}"
                    )
                rows.append(record)
    result = pd.DataFrame(rows)
    if result.duplicated(POLICY_TOKEN_KEYS).any():
        raise ValueError("materialized source labels have duplicate identities")
    return result.sort_values(POLICY_TOKEN_KEYS, kind="stable").reset_index(drop=True)


def assemble_transfer_rows(
    quality: pd.DataFrame,
    propagation: pd.DataFrame,
    route_audit: pd.DataFrame,
    pro_d1_baseline: pd.DataFrame,
    source_labels: pd.DataFrame,
) -> pd.DataFrame:
    """Join source labels to finalized live full-model D1 observations."""

    _require_columns(
        quality,
        [
            *POLICY_TOKEN_KEYS,
            "route_mode",
            "logit_kl",
            "stored_to_live_delta_mse",
            "stored_to_live_delta_cosine",
            "source_stored_delta_mse",
            "live_injected_delta_mse",
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
    _require_columns(
        route_audit,
        [
            "layer",
            "group",
            "request_id",
            "position",
            "route_set_equal",
            "route_order_equal",
            "source_labeled_margin_on_live_logits",
            "live_rank8_rank9_margin",
        ],
        "route_audit",
    )
    _require_columns(
        pro_d1_baseline,
        [
            "injection_layer",
            "group",
            "request_id",
            "position",
            "pro_d1_rank8_rank9_margin",
        ],
        "pro_d1_baseline",
    )
    live_quality = quality[quality["route_mode"].astype(str).eq("live")].copy()
    if live_quality.duplicated(POLICY_TOKEN_KEYS).any():
        raise ValueError("quality has duplicate live policy identities")
    d1 = propagation[
        propagation["route_mode"].astype(str).eq("live")
        & propagation["distance_from_injection"].eq(1)
    ][
        POLICY_TOKEN_KEYS + ["route_membership_change_fraction", "router_mass_churn"]
    ].copy()
    if d1.duplicated(POLICY_TOKEN_KEYS).any():
        raise ValueError("propagation has duplicate live D1 identities")
    rows = source_labels.merge(
        live_quality[
            POLICY_TOKEN_KEYS
            + [
                "logit_kl",
                "stored_to_live_delta_mse",
                "stored_to_live_delta_cosine",
                "source_stored_delta_mse",
                "live_injected_delta_mse",
            ]
        ],
        on=POLICY_TOKEN_KEYS,
        validate="one_to_one",
    ).merge(d1, on=POLICY_TOKEN_KEYS, validate="one_to_one")
    audit = route_audit.rename(
        columns={
            "layer": "injection_layer",
            "source_labeled_margin_on_live_logits": "injection_layer_source_labeled_margin_on_live_logits",
            "live_rank8_rank9_margin": "injection_layer_live_rank8_rank9_margin",
        }
    )
    audit_keys = ["injection_layer", "group", "request_id", "position"]
    if audit.duplicated(audit_keys).any():
        raise ValueError("route_audit has duplicate token identities")
    rows = rows.merge(
        audit[
            audit_keys
            + [
                "route_set_equal",
                "route_order_equal",
                "injection_layer_source_labeled_margin_on_live_logits",
                "injection_layer_live_rank8_rank9_margin",
            ]
        ],
        on=audit_keys,
        validate="many_to_one",
    )
    if pro_d1_baseline.duplicated(audit_keys).any():
        raise ValueError("pro_d1_baseline has duplicate token identities")
    rows = rows.merge(
        pro_d1_baseline[audit_keys + ["pro_d1_rank8_rank9_margin"]],
        on=audit_keys,
        validate="many_to_one",
    )
    rows["actual_live_exact_d1_crossed"] = (
        rows["route_membership_change_fraction"].to_numpy(np.float64) > 0
    )
    rows["source_live_crossing_agree"] = rows["source_exact_d1_crossed"].to_numpy(
        bool
    ) == rows["actual_live_exact_d1_crossed"].to_numpy(bool)
    rows["raw_source_live_crossing_agree"] = rows[
        "raw_source_exact_d1_crossed"
    ].to_numpy(bool) == rows["actual_live_exact_d1_crossed"].to_numpy(bool)
    rows["source_margin_error_vs_pro_baseline"] = (
        rows["raw_source_baseline_rank8_rank9_margin"]
        - rows["pro_d1_rank8_rank9_margin"]
    )
    rows["terminal_kl_used_for_selection"] = False
    return rows.sort_values(POLICY_TOKEN_KEYS, kind="stable").reset_index(drop=True)


def _confusion(frame: pd.DataFrame, predicted: np.ndarray) -> dict[str, Any]:
    actual = frame["actual_live_exact_d1_crossed"].to_numpy(bool)
    predicted = np.asarray(predicted, dtype=bool)
    tp = int(np.sum(predicted & actual))
    fp = int(np.sum(predicted & ~actual))
    tn = int(np.sum(~predicted & ~actual))
    fn = int(np.sum(~predicted & actual))
    return {
        "rows": int(len(frame)),
        "actual_crossings": int(actual.sum()),
        "predicted_crossings": int(predicted.sum()),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "accuracy": _safe_fraction(tp + tn, len(frame)),
        "precision": _safe_fraction(tp, tp + fp),
        "recall": _safe_fraction(tp, tp + fn),
        "specificity": _safe_fraction(tn, tn + fp),
        "false_negative_rate": _safe_fraction(fn, tp + fn),
        "false_positive_rate": _safe_fraction(fp, fp + tn),
        "predicted_positive_fraction": _safe_fraction(tp + fp, len(frame)),
        "observed_positive_fraction": _safe_fraction(tp + fn, len(frame)),
        "observed_crossing_if_predicted_safe": _safe_fraction(fn, fn + tn),
        "observed_crossing_if_predicted_unsafe": _safe_fraction(tp, tp + fp),
    }


def crossing_transfer_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize source-label confusion and margin ranking by policy/rate."""

    records: list[dict[str, Any]] = []
    groupings: list[tuple[str, list[str]]] = [
        ("policy_rate", ["policy", "rate_pages_per_expert"]),
        ("policy_all_rates", ["policy"]),
        ("all_policies_rates", []),
    ]
    for scope, columns in groupings:
        grouped = (
            [((), rows)]
            if not columns
            else rows.groupby(columns, sort=True, dropna=False)
        )
        for key, frame in grouped:
            values = key if isinstance(key, tuple) else (key,)
            labels = dict(zip(columns, values))
            confusion = _confusion(
                frame, frame["source_exact_d1_crossed"].to_numpy(bool)
            )
            raw_confusion = _confusion(
                frame, frame["raw_source_exact_d1_crossed"].to_numpy(bool)
            )
            record = {
                "schema": SCHEMA,
                "scope": scope,
                "policy": str(labels.get("policy", "__all__")),
                "rate_pages_per_expert": int(labels.get("rate_pages_per_expert", -1)),
                **confusion,
                "raw_source_accuracy": raw_confusion["accuracy"],
                "raw_source_false_negative_rate": raw_confusion["false_negative_rate"],
                "patched_rows": int(frame["allocation_patched"].sum()),
                "candidate_margin_risk_roc_auc": binary_roc_auc(
                    frame["actual_live_exact_d1_crossed"],
                    -frame["source_candidate_labeled_set_margin"],
                ),
                "raw_candidate_margin_risk_roc_auc": binary_roc_auc(
                    frame["actual_live_exact_d1_crossed"],
                    -frame["raw_source_candidate_labeled_set_margin"],
                ),
                "mean_source_candidate_margin": float(
                    frame["source_candidate_labeled_set_margin"].mean()
                ),
                "mean_actual_d1_router_mass_churn": float(
                    frame["router_mass_churn"].mean()
                ),
                "terminal_kl_used_for_selection": False,
            }
            records.append(record)
    return (
        pd.DataFrame(records)
        .sort_values(["scope", "policy", "rate_pages_per_expert"], kind="stable")
        .reset_index(drop=True)
    )


def threshold_calibration(
    rows: pd.DataFrame,
    thresholds: Sequence[float] = THRESHOLDS,
) -> pd.DataFrame:
    """Audit candidate-margin safety thresholds without selecting by KL."""

    records: list[dict[str, Any]] = []
    for (policy, rate), frame in rows.groupby(
        ["policy", "rate_pages_per_expert"], sort=True, dropna=False
    ):
        margin = frame["source_candidate_labeled_set_margin"].to_numpy(np.float64)
        for threshold in thresholds:
            confusion = _confusion(frame, margin <= float(threshold))
            records.append(
                {
                    "schema": SCHEMA,
                    "policy": str(policy),
                    "rate_pages_per_expert": int(rate),
                    "margin_threshold": float(threshold),
                    **confusion,
                    "candidate_margin_risk_roc_auc": binary_roc_auc(
                        frame["actual_live_exact_d1_crossed"], -margin
                    ),
                    "terminal_kl_used_for_selection": False,
                }
            )
    return (
        pd.DataFrame(records)
        .sort_values(
            ["policy", "rate_pages_per_expert", "margin_threshold"], kind="stable"
        )
        .reset_index(drop=True)
    )


def baseline_margin_transfer_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Compare raw-source and exact-PRO D1 Q4 rank-8/rank-9 margins."""

    baseline = rows.drop_duplicates(["rate_pages_per_expert", *TOKEN_KEYS]).copy()
    baseline["patch_stratum"] = np.where(
        baseline["allocation_patched"], "patched", "unpatched"
    )
    records: list[dict[str, Any]] = []
    for rate in [-1, *sorted(map(int, baseline["rate_pages_per_expert"].unique()))]:
        rate_frame = (
            baseline
            if rate == -1
            else baseline[baseline["rate_pages_per_expert"].eq(rate)]
        )
        for stratum in ("all", "patched", "unpatched"):
            frame = (
                rate_frame
                if stratum == "all"
                else rate_frame[rate_frame["patch_stratum"].eq(stratum)]
            )
            if frame.empty:
                continue
            source = frame["raw_source_baseline_rank8_rank9_margin"]
            pro = frame["pro_d1_rank8_rank9_margin"]
            error = source.to_numpy(np.float64) - pro.to_numpy(np.float64)
            records.append(
                {
                    "schema": SCHEMA,
                    "rate_pages_per_expert": int(rate),
                    "patch_stratum": stratum,
                    "groups": int(len(frame)),
                    "pearson": _correlation(source, pro),
                    "spearman": _spearman(source, pro),
                    "mae": float(np.mean(np.abs(error))),
                    "bias_source_minus_pro": float(np.mean(error)),
                    "maximum_absolute_error": float(np.max(np.abs(error))),
                    "source_zero_margin_fraction": float(
                        np.mean(source.to_numpy(np.float64) == 0)
                    ),
                    "pro_zero_margin_fraction": float(
                        np.mean(pro.to_numpy(np.float64) == 0)
                    ),
                }
            )
    return (
        pd.DataFrame(records)
        .sort_values(["rate_pages_per_expert", "patch_stratum"], kind="stable")
        .reset_index(drop=True)
    )


def delta_fidelity_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize stored-delta transfer separately from route-label transfer."""

    records: list[dict[str, Any]] = []
    for (policy, rate), frame in rows.groupby(
        ["policy", "rate_pages_per_expert"], sort=True, dropna=False
    ):
        mse = frame["stored_to_live_delta_mse"].to_numpy(np.float64)
        cosine = frame["stored_to_live_delta_cosine"].to_numpy(np.float64)
        records.append(
            {
                "schema": SCHEMA,
                "policy": str(policy),
                "rate_pages_per_expert": int(rate),
                "rows": int(len(frame)),
                "mean_stored_to_live_delta_mse": float(np.mean(mse)),
                "median_stored_to_live_delta_mse": float(np.median(mse)),
                "p95_stored_to_live_delta_mse": float(np.quantile(mse, 0.95)),
                "maximum_stored_to_live_delta_mse": float(np.max(mse)),
                "mean_stored_to_live_delta_cosine": float(np.mean(cosine)),
                "p05_stored_to_live_delta_cosine": float(np.quantile(cosine, 0.05)),
                "minimum_stored_to_live_delta_cosine": float(np.min(cosine)),
                "fraction_cosine_at_least_0_999": float(np.mean(cosine >= 0.999)),
                "mean_source_stored_delta_mse": float(
                    frame["source_stored_delta_mse"].mean()
                ),
                "mean_live_injected_delta_mse": float(
                    frame["live_injected_delta_mse"].mean()
                ),
                "crossing_label_agreement": float(
                    frame["source_live_crossing_agree"].mean()
                ),
            }
        )
    return (
        pd.DataFrame(records)
        .sort_values(["policy", "rate_pages_per_expert"], kind="stable")
        .reset_index(drop=True)
    )


def patched_fixed_pr13_kl_audit(
    rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair fixed D1 and PR13 KL, stratified only by patch identity.

    This is an outcome audit, not a candidate-selection rule.
    """

    fixed = rows[rows["policy"].astype(str).eq(FIXED_POLICY)][
        RATE_TOKEN_KEYS
        + [
            "allocation_patched",
            "logit_kl",
            "actual_live_exact_d1_crossed",
            "source_exact_d1_crossed",
        ]
    ].copy()
    pr13 = rows[rows["policy"].astype(str).eq(PR13_POLICY)][
        RATE_TOKEN_KEYS + ["logit_kl", "actual_live_exact_d1_crossed"]
    ].rename(
        columns={
            "logit_kl": "pr13_logit_kl",
            "actual_live_exact_d1_crossed": "pr13_actual_live_exact_d1_crossed",
        }
    )
    paired = fixed.merge(pr13, on=RATE_TOKEN_KEYS, validate="one_to_one")
    paired["patch_stratum"] = np.where(
        paired["allocation_patched"], "patched", "unpatched"
    )
    paired["fixed_logit_kl_minus_pr13"] = paired["logit_kl"] - paired["pr13_logit_kl"]
    paired["fixed_prevented_pr13_crossing"] = (
        paired["pr13_actual_live_exact_d1_crossed"]
        & ~paired["actual_live_exact_d1_crossed"]
    )
    paired["fixed_introduced_crossing"] = (
        ~paired["pr13_actual_live_exact_d1_crossed"]
        & paired["actual_live_exact_d1_crossed"]
    )
    paired["terminal_kl_used_for_selection"] = False
    paired.insert(0, "schema", SCHEMA)

    records: list[dict[str, Any]] = []
    for rate in [-1, *sorted(map(int, paired["rate_pages_per_expert"].unique()))]:
        rate_frame = (
            paired if rate == -1 else paired[paired["rate_pages_per_expert"].eq(rate)]
        )
        for stratum in ("patched", "unpatched"):
            frame = rate_frame[rate_frame["patch_stratum"].eq(stratum)]
            if frame.empty:
                continue
            delta = frame["fixed_logit_kl_minus_pr13"].to_numpy(np.float64)
            records.append(
                {
                    "schema": SCHEMA,
                    "rate_pages_per_expert": int(rate),
                    "patch_stratum": stratum,
                    "rows": int(len(frame)),
                    "mean_fixed_logit_kl_minus_pr13": float(np.mean(delta)),
                    "median_fixed_logit_kl_minus_pr13": float(np.median(delta)),
                    "p90_fixed_logit_kl_minus_pr13": float(np.quantile(delta, 0.90)),
                    "maximum_fixed_logit_kl_minus_pr13": float(np.max(delta)),
                    "fixed_improved_fraction": float(np.mean(delta < 0)),
                    "fixed_prevented_pr13_crossings": int(
                        frame["fixed_prevented_pr13_crossing"].sum()
                    ),
                    "fixed_introduced_crossings": int(
                        frame["fixed_introduced_crossing"].sum()
                    ),
                    "source_live_fixed_crossing_agreement": float(
                        (
                            frame["source_exact_d1_crossed"].to_numpy(bool)
                            == frame["actual_live_exact_d1_crossed"].to_numpy(bool)
                        ).mean()
                    ),
                    "terminal_kl_used_for_selection": False,
                }
            )
    summary = (
        pd.DataFrame(records)
        .sort_values(["rate_pages_per_expert", "patch_stratum"], kind="stable")
        .reset_index(drop=True)
    )
    return (
        paired.sort_values(RATE_TOKEN_KEYS, kind="stable").reset_index(drop=True),
        summary,
    )
