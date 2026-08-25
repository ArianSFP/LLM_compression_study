from __future__ import annotations

import numpy as np
import pandas as pd

from oracle_study.d1_source_live_transfer import (
    FIXED_POLICY,
    POLICIES,
    PR13_POLICY,
    assemble_transfer_rows,
    baseline_margin_transfer_summary,
    binary_roc_auc,
    crossing_transfer_summary,
    materialize_source_labels,
    patched_fixed_pr13_kl_audit,
    threshold_calibration,
)


def _manifest() -> dict[str, object]:
    provenance = {
        "pr13_source_policy": "raw_pr13",
        "local_source_policy": "raw_local",
        "fixed_d1_source_policy": "raw_fixed",
        "companion_d1_source_policy": "raw_companion",
        "exact_token_oracle_source_policy_by_group": {"0": "raw_eta0", "1": "raw_eta1"},
    }
    return {
        "cells": {
            "layer_00_rate_384": {"policy_provenance": provenance},
        }
    }


def _source_cell() -> pd.DataFrame:
    mapping = {
        0: {
            "raw_pr13": (False, 0.03125),
            "raw_local": (False, 0.02000),
            "raw_fixed": (False, 0.015625),
            "raw_companion": (True, -0.015625),
            "raw_eta0": (False, 0.04000),
            "raw_eta1": (False, 0.05000),
        },
        1: {
            "raw_pr13": (True, -0.01000),
            "raw_local": (False, 0.01000),
            "raw_fixed": (False, 0.00500),
            "raw_companion": (False, 0.03000),
            "raw_eta0": (False, 0.04000),
            "raw_eta1": (False, 0.00800),
        },
    }
    rows = []
    for group, policies in mapping.items():
        for policy, (crossed, margin) in policies.items():
            rows.append(
                {
                    "layer": 0,
                    "rate_pages_per_expert": 384,
                    "group": group,
                    "request_id": f"r{group}",
                    "position": group + 1,
                    "policy": policy,
                    "exact_d1_crossed": crossed,
                    "baseline_rank8_rank9_margin": 0.02 + group * 0.01,
                    "candidate_labeled_set_margin": margin,
                    "membership_pairs_changed": int(crossed),
                    "routing_mass_lost": 0.1 if crossed else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _patch() -> pd.DataFrame:
    rows = []
    for policy in POLICIES:
        rows.append(
            {
                "layer": 0,
                "rate_pages_per_expert": 384,
                "group": 1,
                "request_id": "r1",
                "position": 2,
                "policy": policy,
                "exact_d1_crossed": policy == FIXED_POLICY,
                "baseline_rank8_rank9_margin": 0.015625,
                "candidate_labeled_set_margin": (
                    -0.01 if policy == FIXED_POLICY else 0.02
                ),
                "membership_pairs_changed": int(policy == FIXED_POLICY),
                "routing_mass_lost": 0.2 if policy == FIXED_POLICY else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _tail_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    quality = []
    propagation = []
    for group in (0, 1):
        for index, policy in enumerate(POLICIES):
            quality.append(
                {
                    "injection_layer": 0,
                    "rate_pages_per_expert": 384,
                    "group": group,
                    "request_id": f"r{group}",
                    "position": group + 1,
                    "policy": policy,
                    "route_mode": "live",
                    "logit_kl": group + index / 10,
                    "stored_to_live_delta_mse": 1e-9 * (index + 1),
                    "stored_to_live_delta_cosine": 0.999 - index * 1e-5,
                    "source_stored_delta_mse": 1e-4,
                    "live_injected_delta_mse": 1.1e-4,
                }
            )
            crossed = (group == 0 and policy == "frontier_companion_d1") or (
                group == 1 and policy == FIXED_POLICY
            )
            propagation.append(
                {
                    "injection_layer": 0,
                    "rate_pages_per_expert": 384,
                    "group": group,
                    "request_id": f"r{group}",
                    "position": group + 1,
                    "policy": policy,
                    "route_mode": "live",
                    "distance_from_injection": 1,
                    "route_membership_change_fraction": float(crossed),
                    "router_mass_churn": 0.1 if crossed else 0.0,
                }
            )
    audit = pd.DataFrame(
        [
            {
                "layer": 0,
                "group": group,
                "request_id": f"r{group}",
                "position": group + 1,
                "route_set_equal": group == 0,
                "route_order_equal": group == 0,
                "source_labeled_margin_on_live_logits": 0.02,
                "live_rank8_rank9_margin": 99.0,
            }
            for group in (0, 1)
        ]
    )
    return pd.DataFrame(quality), pd.DataFrame(propagation), audit


def _rows() -> pd.DataFrame:
    source = materialize_source_labels(
        _manifest(), {(0, 384): _source_cell()}, _patch()
    )
    quality, propagation, audit = _tail_frames()
    pro_d1 = pd.DataFrame(
        [
            {
                "injection_layer": 0,
                "group": 0,
                "request_id": "r0",
                "position": 1,
                "pro_d1_rank8_rank9_margin": 0.021,
            },
            {
                "injection_layer": 0,
                "group": 1,
                "request_id": "r1",
                "position": 2,
                "pro_d1_rank8_rank9_margin": 0.031,
            },
        ]
    )
    return assemble_transfer_rows(quality, propagation, audit, pro_d1, source)


def test_materialize_source_labels_applies_patch_but_retains_raw_margin() -> None:
    source = materialize_source_labels(
        _manifest(), {(0, 384): _source_cell()}, _patch()
    )
    fixed = source[(source["group"].eq(1)) & source["policy"].eq(FIXED_POLICY)].iloc[0]
    assert fixed["source_metric_origin"] == "same_host_patch"
    assert bool(fixed["source_exact_d1_crossed"])
    assert not bool(fixed["raw_source_exact_d1_crossed"])
    assert fixed["source_candidate_labeled_set_margin"] == -0.01
    assert fixed["raw_source_candidate_labeled_set_margin"] == 0.005


def test_auc_is_tie_aware_and_margin_risk_uses_negative_margin() -> None:
    assert binary_roc_auc([False, False, True, True], [0.0, 0.5, 0.5, 1.0]) == 0.875
    rows = _rows()
    summary = crossing_transfer_summary(rows)
    fixed = summary[
        summary["scope"].eq("policy_rate") & summary["policy"].eq(FIXED_POLICY)
    ].iloc[0]
    assert fixed["rows"] == 2
    assert fixed["accuracy"] == 1.0
    assert fixed["candidate_margin_risk_roc_auc"] == 1.0


def test_threshold_calibration_keeps_ranking_separate_from_threshold() -> None:
    calibrated = threshold_calibration(_rows())
    fixed = calibrated[calibrated["policy"].eq(FIXED_POLICY)]
    at_zero = fixed[fixed["margin_threshold"].eq(0.0)].iloc[0]
    at_one_64 = fixed[fixed["margin_threshold"].eq(1 / 64)].iloc[0]
    assert at_zero["true_positive"] == 1
    assert at_zero["false_positive"] == 0
    assert at_one_64["false_positive"] == 1
    assert (
        at_zero["candidate_margin_risk_roc_auc"]
        == at_one_64["candidate_margin_risk_roc_auc"]
    )
    assert not bool(fixed["terminal_kl_used_for_selection"].any())


def test_baseline_summary_never_mixes_injection_router_with_d1_router() -> None:
    rows = _rows()
    assert set(rows["injection_layer_live_rank8_rank9_margin"]) == {99.0}
    assert set(rows["pro_d1_rank8_rank9_margin"]) == {0.021, 0.031}
    summary = baseline_margin_transfer_summary(rows)
    overall = summary[
        summary["rate_pages_per_expert"].eq(-1) & summary["patch_stratum"].eq("all")
    ].iloc[0]
    assert overall["groups"] == 2
    assert np.isclose(overall["mae"], 0.001)
    assert "source_pro_route_set_agreement" not in summary


def test_patch_kl_audit_pairs_fixed_to_pr13_without_selecting() -> None:
    paired, summary = patched_fixed_pr13_kl_audit(_rows())
    assert len(paired) == 2
    assert set(summary["patch_stratum"]) == {"patched", "unpatched"}
    patched = summary[
        summary["rate_pages_per_expert"].eq(-1) & summary["patch_stratum"].eq("patched")
    ].iloc[0]
    assert np.isclose(patched["mean_fixed_logit_kl_minus_pr13"], 0.2)
    assert not bool(paired["terminal_kl_used_for_selection"].any())
