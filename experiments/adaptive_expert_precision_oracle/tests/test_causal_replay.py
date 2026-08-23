from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oracle_study.causal_replay import (
    canonical_states,
    categorical_kl,
    mean_squared_error,
    projection_responses,
    qenergy_damage,
    reconstruct_group_from_responses,
    rms_normalize,
    route_boundary_metrics,
    route_metrics,
    selected_pages,
    token_quality_metrics,
)


def _decoded(seed: int = 7) -> tuple[np.ndarray, dict[int, tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]]]:
    rng = np.random.default_rng(seed)
    activation = rng.normal(size=4)
    result = {}
    for expert in range(8):
        q4 = (
            rng.normal(size=(3, 4)),
            rng.normal(size=(3, 4)),
            rng.normal(size=(5, 3)),
        )
        q2 = tuple(value + rng.normal(scale=0.08, size=value.shape) for value in q4)
        result[expert] = (q2, q4)
    return activation, result


def test_canonical_states_and_page_accounting() -> None:
    value = [0, 1, 2, 3, 4, 5, 6, 7]
    text = json.dumps(value, separators=(",", ":"))
    assert np.array_equal(canonical_states(text, units=8), value)
    assert selected_pages(value) == 12
    with pytest.raises(ValueError, match="canonical"):
        canonical_states(json.dumps(value), units=8)
    with pytest.raises(ValueError, match="values"):
        canonical_states([0, 1], units=8)


def test_reconstruction_sign_qenergy_and_all_q4_gate() -> None:
    activation, decoded = _decoded()
    responses = projection_responses(activation, np.arange(8), decoded)
    states = [np.asarray([(expert + unit) % 8 for unit in range(3)]) for expert in range(8)]
    weights = np.arange(1, 9, dtype=np.float64)
    weights /= weights.sum()
    proxy = np.arange(10, dtype=np.float64).reshape(5, 2) / 13.0
    result = reconstruct_group_from_responses(responses, weights, states, proxy, 0.25)
    assert np.array_equal(result.delta, -result.residual)
    assert result.all_q4_max_abs_error < 1e-14
    assert result.group_damage == pytest.approx(qenergy_damage(result.residual, proxy, 0.25))
    assert np.allclose(
        result.expert_damages,
        [qenergy_damage(error, proxy, 0.25) for error in result.expert_residuals],
    )
    assert np.array_equal(result.selected_pages, [selected_pages(value) for value in states])


def test_all_q4_selected_states_produce_zero_delta() -> None:
    activation, decoded = _decoded(19)
    responses = projection_responses(activation, np.arange(8), decoded)
    states = [np.full(3, 7, dtype=np.int64) for _ in range(8)]
    result = reconstruct_group_from_responses(
        responses, np.full(8, 0.125), states, None, 0.0,
    )
    assert np.max(np.abs(result.delta)) < 1e-14
    assert np.max(np.abs(result.expert_damages)) < 1e-28
    assert result.group_damage < 1e-28


def test_metrics_have_expected_identity_values() -> None:
    hidden = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    logits = np.asarray([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
    assert mean_squared_error(hidden, hidden) == 0.0
    assert np.allclose(np.mean(rms_normalize(hidden) ** 2, axis=-1), 1.0, atol=1e-6)
    assert categorical_kl(logits, logits) == pytest.approx(0.0, abs=1e-15)
    route = route_metrics(logits, logits, top_k=1)
    assert route["top8_overlap"] == 1.0
    assert route["route_churn"] == 0.0
    assert route["route_top1_agreement"] == 1.0
    assert route["route_ordered_top8_agreement"] == 1.0
    quality = token_quality_metrics(logits, logits, np.asarray([2, 0]))
    assert quality["delta_nll"] == 0.0
    assert quality["ppl_ratio"] == 1.0
    assert quality["logit_kl"] == pytest.approx(0.0, abs=1e-15)
    assert quality["top1_agreement"] == 1.0


def test_metric_shape_validation() -> None:
    with pytest.raises(ValueError, match="proxy"):
        qenergy_damage(np.ones(3), np.ones((4, 1)), 1.0)
    with pytest.raises(ValueError, match="top_k"):
        route_metrics(np.ones((2, 3)), np.ones((2, 3)), top_k=3)
    with pytest.raises(ValueError, match="labels"):
        token_quality_metrics(np.ones((2, 3)), np.ones((2, 3)), np.ones(1))


def test_route_boundary_decomposition_and_mass_churn() -> None:
    reference = np.asarray([
        [4.0, 3.0, 2.0, 1.0],
        [4.0, 3.0, 2.0, 1.0],
        [4.0, 3.0, 2.0, 1.0],
    ])
    candidate = np.asarray([
        [4.0, 3.0, 2.0, 1.0],
        [3.0, 4.0, 2.0, 1.0],
        [4.0, 1.0, 3.0, 2.0],
    ])
    metrics = route_boundary_metrics(reference, candidate, top_k=2)
    assert metrics["route_no_change_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["route_order_only_change_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["route_membership_change_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["route_top1_change_fraction"] == pytest.approx(1.0 / 3.0)
    assert 0.0 < metrics["router_mass_churn"] < 1.0

    identity = route_boundary_metrics(reference, reference, top_k=2)
    assert identity == {
        "route_no_change_fraction": 1.0,
        "route_order_only_change_fraction": 0.0,
        "route_membership_change_fraction": 0.0,
        "route_top1_change_fraction": 0.0,
        "router_mass_churn": 0.0,
    }
