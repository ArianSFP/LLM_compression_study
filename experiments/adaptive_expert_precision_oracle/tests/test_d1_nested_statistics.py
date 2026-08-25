from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oracle_study.d1_nested_statistics import (
    holm_adjust,
    paired_distribution_summary,
    paired_request_cluster_bootstrap_ci,
    paired_request_differences,
    paired_sign_flip_test,
    request_equal_arm_means,
    request_layer_equal_means,
    validate_paired_alignment,
)


def _raw_panel() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    values = {
        "r0": {
            "reference": {0: [0.0, 0.0], 1: [0.0]},
            "candidate": {0: [0.0, 2.0], 1: [9.0]},
        },
        "r1": {
            "reference": {0: [1.0, 1.0], 1: [1.0]},
            "candidate": {0: [3.0, 5.0], 1: [7.0]},
        },
    }
    for request, arms in values.items():
        for arm, layers in arms.items():
            for layer, observations in layers.items():
                for observation, value in enumerate(observations):
                    records.append(
                        {
                            "request_id": request,
                            "split": "evaluation",
                            "arm": arm,
                            "rate_pages_per_expert": 360,
                            "injection_layer": layer,
                            "observation": observation,
                            "logit_kl": value,
                        }
                    )
    return pd.DataFrame.from_records(records)


def test_layer_then_request_equal_aggregation_is_not_row_weighted() -> None:
    raw = _raw_panel()
    request_rows = request_layer_equal_means(
        raw,
        ["logit_kl"],
        observation_cols=["observation"],
        expected_arms=["candidate", "reference"],
        expected_rates=[360],
        expected_layers=[0, 1],
    )

    r0_candidate = request_rows[
        request_rows["request_id"].eq("r0")
        & request_rows["arm"].eq("candidate")
    ].iloc[0]
    assert r0_candidate["logit_kl"] == 5.0  # ((0 + 2) / 2 + 9) / 2
    assert r0_candidate["logit_kl"] != pytest.approx(11.0 / 3.0)
    assert r0_candidate["layers_averaged"] == 2

    arm_rows = request_equal_arm_means(request_rows, ["logit_kl"])
    candidate = arm_rows[arm_rows["arm"].eq("candidate")].iloc[0]
    # r0 contributes 5 and r1 contributes ((3 + 5) / 2 + 7) / 2 = 5.5.
    assert candidate["logit_kl"] == 5.25
    assert candidate["requests_averaged"] == 2


def test_alignment_supports_declared_arm_specific_rates() -> None:
    rows = []
    for request in ("r0", "r1"):
        for arm, rates in {"pr13": (360, 384), "nested": (360,)}.items():
            for rate in rates:
                for layer in (0, 1):
                    rows.append(
                        {
                            "request_id": request,
                            "split": "evaluation",
                            "arm": arm,
                            "rate_pages_per_expert": rate,
                            "injection_layer": layer,
                        }
                    )
    frame = pd.DataFrame(rows)
    validate_paired_alignment(
        frame,
        expected_arm_rates={"pr13": [360, 384], "nested": [360]},
        expected_layers=[0, 1],
    )

    missing = frame.drop(
        frame[
            frame["request_id"].eq("r1")
            & frame["arm"].eq("nested")
            & frame["injection_layer"].eq(1)
        ].index
    )
    with pytest.raises(ValueError, match="unequal|layers"):
        validate_paired_alignment(
            missing,
            expected_arm_rates={"pr13": [360, 384], "nested": [360]},
            expected_layers=[0, 1],
        )


@pytest.mark.parametrize("failure", ["split", "duplicate", "arm", "rate"])
def test_alignment_rejects_split_duplicates_and_incomplete_grids(failure: str) -> None:
    frame = _raw_panel()
    if failure == "split":
        frame.loc[
            frame["request_id"].eq("r0") & frame["arm"].eq("candidate"),
            "split",
        ] = "calibration"
        match = "more than one split"
    elif failure == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        match = "duplicate"
    elif failure == "arm":
        frame = frame.drop(frame[frame["request_id"].eq("r1") & frame["arm"].eq("candidate")].index)
        match = "arm/rate grid"
    else:
        frame.loc[frame["request_id"].eq("r1"), "rate_pages_per_expert"] = 384
        match = "arm/rate grid"
    with pytest.raises(ValueError, match=match):
        validate_paired_alignment(
            frame,
            observation_cols=["observation"],
            expected_arms=["candidate", "reference"],
            expected_rates=[360],
            expected_layers=[0, 1],
        )


def test_product_rate_pairing_forms_one_difference_per_request() -> None:
    request_rows = pd.DataFrame(
        [
            {
                "request_id": request,
                "split": "evaluation",
                "arm": arm,
                "rate_pages_per_expert": rate,
                "logit_kl": value,
            }
            for request, candidate, reference in (("r0", 1.0, 2.5), ("r1", 4.0, 3.0))
            for arm, rate, value in (
                ("nested_d1", 360, candidate),
                ("pr13", 384, reference),
            )
        ]
    )
    paired = paired_request_differences(
        request_rows,
        ["logit_kl"],
        candidate_arm="nested_d1",
        reference_arm="pr13",
        candidate_rate=360,
        reference_rate=384,
        split="evaluation",
        expected_arm_rates={"nested_d1": [360], "pr13": [384]},
    )

    assert paired["request_id"].tolist() == ["r0", "r1"]
    assert paired["logit_kl_difference"].tolist() == [-1.5, 1.0]
    assert paired["candidate_rate"].eq(360).all()
    assert paired["reference_rate"].eq(384).all()


def _differences(values: list[float], split: str = "evaluation") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "request_id": [f"r{index:02d}" for index in range(len(values))],
            "split": [split] * len(values),
            "difference": values,
        }
    )


def test_request_cluster_bootstrap_is_seeded_and_uses_request_means() -> None:
    frame = _differences([-2.0, -1.0, 1.0, 4.0])
    first = paired_request_cluster_bootstrap_ci(
        frame, value_col="difference", resamples=2_000, seed=31
    )
    second = paired_request_cluster_bootstrap_ci(
        frame.sample(frac=1.0, random_state=7),
        value_col="difference",
        resamples=2_000,
        seed=31,
    )

    assert first == second
    assert first.estimate == 0.5
    assert first.requests == 4
    assert first.resamples == 2_000
    assert first.lower <= first.estimate <= first.upper

    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="one row per independent request"):
        paired_request_cluster_bootstrap_ci(
            duplicated, value_col="difference", resamples=100, seed=0
        )


def test_exact_and_monte_carlo_sign_flip_are_correct_and_deterministic() -> None:
    exact = paired_sign_flip_test(
        _differences([-1.0, -2.0]),
        value_col="difference",
        alternative="less",
        seed=99,
    )
    assert exact.method == "exact"
    assert exact.statistic == -1.5
    assert exact.p_value == 0.25
    assert exact.permutations == 4

    many = _differences([float(index + 1) for index in range(25)])
    first = paired_sign_flip_test(
        many,
        value_col="difference",
        monte_carlo_draws=5_000,
        seed=17,
    )
    second = paired_sign_flip_test(
        many.sample(frac=1.0, random_state=8),
        value_col="difference",
        monte_carlo_draws=5_000,
        seed=17,
    )
    assert first == second
    assert first.method == "monte_carlo"
    assert first.permutations == 5_000
    assert 0.0 < first.p_value <= 1.0


def test_sign_flip_and_bootstrap_refuse_to_pool_splits() -> None:
    frame = pd.concat(
        [_differences([-1.0, 0.0], "calibration"), _differences([1.0, 2.0], "evaluation")],
        ignore_index=True,
    )
    frame["request_id"] = ["c0", "c1", "e0", "e1"]
    with pytest.raises(ValueError, match="cannot pool"):
        paired_sign_flip_test(frame, value_col="difference")
    with pytest.raises(ValueError, match="cannot pool"):
        paired_request_cluster_bootstrap_ci(
            frame, value_col="difference", resamples=100
        )


def test_holm_adjustment_preserves_input_order_and_controls_monotonicity() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03, 0.002])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.008])
    assert holm_adjust([]).shape == (0,)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        holm_adjust([0.2, 1.1])


def test_paired_distribution_summary_reports_request_level_tail() -> None:
    summary = paired_distribution_summary(
        _differences([-2.0, -1.0, 0.0, 4.0]), value_col="difference"
    )
    assert summary.requests == 4
    assert summary.mean == 0.25
    assert summary.median == -0.5
    assert summary.win_fraction == 0.5
    assert summary.p90 == pytest.approx(2.8)
    assert summary.p95 == pytest.approx(3.4)
    assert summary.maximum == 4.0

    higher = paired_distribution_summary(
        _differences([-2.0, -1.0, 0.0, 4.0]),
        value_col="difference",
        improvement="higher",
    )
    assert higher.win_fraction == 0.25
