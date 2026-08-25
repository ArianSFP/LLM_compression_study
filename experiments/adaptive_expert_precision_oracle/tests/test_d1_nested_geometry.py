from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "src"))

from oracle_study.d1_nested_allocation import (  # noqa: E402
    apply_page_move,
    legal_add_moves,
)
from oracle_study.d1_nested_geometry import TokenStateGeometry  # noqa: E402
from oracle_study.split_interaction_field import (  # noqa: E402
    SplitProjectionResponses,
)


def _geometry(seed: int = 4) -> TokenStateGeometry:
    rng = np.random.default_rng(seed)
    responses = tuple(
        SplitProjectionResponses(
            hidden=rng.normal(size=(512, 4)),
            down2=rng.normal(scale=0.03, size=(512, 9)),
            down4=rng.normal(scale=0.03, size=(512, 9)),
        )
        for _ in range(8)
    )
    weights = np.arange(1, 9, dtype=np.float64)
    weights /= weights.sum()
    return TokenStateGeometry(
        responses, weights, rng.normal(size=(9, 3)), 0.7,
    )


def test_vectorized_move_scores_equal_exact_state_reconstruction() -> None:
    geometry = _geometry()
    states = np.zeros((8, 512), np.uint8)
    moves = legal_add_moves(states)[:37]
    observed = geometry.score_moves(states, moves, chunk_size=7)
    expected = np.asarray([
        geometry.local_damage(apply_page_move(states, move)) for move in moves
    ])
    np.testing.assert_allclose(observed, expected, rtol=2e-13, atol=2e-13)


def test_signed_effects_are_exact_directional_differences() -> None:
    geometry = _geometry(7)
    states = np.zeros((8, 512), np.uint8)
    moves = legal_add_moves(states)[100:117]
    gradients = np.arange(45, dtype=np.float64).reshape(5, 9) / 17.0
    observed = geometry.signed_move_effects(
        states, moves, gradients, chunk_size=4,
    )
    baseline = geometry.output_delta(states)
    expected = np.stack([
        gradients @ (geometry.output_delta(apply_page_move(states, move)) - baseline)
        for move in moves
    ])
    np.testing.assert_allclose(observed, expected, rtol=2e-13, atol=2e-13)


def test_token_dependent_hidden_response_changes_page_effect() -> None:
    first = _geometry(9)
    second = _geometry(10)
    states = np.zeros((8, 512), np.uint8)
    move = legal_add_moves(states)[0]
    assert not np.array_equal(
        first.move_output_delta(states, move),
        second.move_output_delta(states, move),
    )


def test_all_q2_and_legacy_metrics_are_finite() -> None:
    geometry = _geometry(11)
    weights = np.full(8, 1.0 / 8.0)
    states = np.zeros((8, 512), np.uint8)
    assert geometry.all_q2_damage() == geometry.local_damage(states)
    assert np.isfinite(geometry.legacy_additive_damage(states, weights))


def test_raw_bf16_execution_weights_are_retained_without_renormalization() -> None:
    normalized = _geometry(13)
    raw = np.asarray(normalized.execution_weights) * 1.002
    geometry = TokenStateGeometry(
        normalized.responses,
        raw,
        normalized.proxy,
        normalized.beta,
    )
    np.testing.assert_array_equal(geometry.execution_weights, raw)
    assert geometry.execution_sum_tolerance == 0.003
    states = np.zeros((8, 512), np.uint8)
    np.testing.assert_allclose(
        geometry.output_delta(states),
        1.002 * normalized.output_delta(states),
        rtol=2e-13,
        atol=2e-13,
    )
    with pytest.raises(ValueError, match="sum tolerance"):
        TokenStateGeometry(
            normalized.responses,
            raw,
            normalized.proxy,
            normalized.beta,
            execution_sum_tolerance=0.001,
        )
