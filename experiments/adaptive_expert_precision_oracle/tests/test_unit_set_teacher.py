from __future__ import annotations

import numpy as np
import pytest

from oracle_study.unit_set_teacher import (
    down_metric_gram,
    exact_contained_target_fixed_greedy,
    exact_full_target_fixed_greedy,
    selected_exclusion_regret,
    set_damage,
    set_gain,
    unit_correction_gram,
)


def qinner(left: np.ndarray, right: np.ndarray, proxy: np.ndarray, beta: float) -> float:
    return float(left @ right + beta * ((left @ proxy) @ (right @ proxy)))


def test_hidden_scaled_down_gram_matches_explicit_corrections_with_proxy() -> None:
    rng = np.random.default_rng(91)
    output, units = 13, 8
    down2 = rng.normal(size=(output, units))
    down4 = down2 + 0.3 * rng.normal(size=(output, units))
    h2, h4 = rng.normal(size=(2, units))
    proxy = rng.normal(size=(output, 3))
    beta = 0.27
    metadata = down_metric_gram(down2, down4, proxy=proxy, beta=beta, dtype=np.float64)
    actual = unit_correction_gram(h2, h4, metadata, dtype=np.float64)
    corrections = down4.T * h4[:, None] - down2.T * h2[:, None]
    expected = np.array([
        [qinner(left, right, proxy, beta) for right in corrections]
        for left in corrections
    ])
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    assert metadata.teacher_bytes == 3 * units * units * 8


def test_set_damage_and_gain_match_explicit_residual() -> None:
    rng = np.random.default_rng(92)
    corrections = rng.normal(size=(7, 11))
    gram = corrections @ corrections.T
    mask = np.array([1, 0, 1, 0, 0, 1, 0], dtype=bool)
    residual = corrections[~mask].sum(axis=0)
    assert set_damage(gram, mask) == pytest.approx(float(residual @ residual))
    target = corrections.sum(axis=0)
    assert set_gain(gram, mask) == pytest.approx(float(target @ target - residual @ residual))


def test_full_target_fixed_greedy_matches_direct_correlation_updates() -> None:
    rng = np.random.default_rng(93)
    corrections = rng.normal(size=(9, 12))
    gram = corrections @ corrections.T
    path = exact_full_target_fixed_greedy(gram, 6)
    residual = corrections.sum(axis=0)
    expected_order, expected_gain = [], []
    available = set(range(len(corrections)))
    for _ in range(6):
        candidates = []
        for unit in sorted(available):
            vector = corrections[unit]
            gain = 2.0 * float(vector @ residual) - float(vector @ vector)
            candidates.append((gain, -unit, unit))
        gain, _, unit = max(candidates)
        expected_order.append(unit)
        expected_gain.append(gain)
        residual -= corrections[unit]
        available.remove(unit)
    assert path.order.tolist() == expected_order
    np.testing.assert_allclose(path.marginal_gain, expected_gain, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(path.remaining_damage[-1], residual @ residual)


def test_full_and_contained_target_reranking_are_explicitly_distinct() -> None:
    # Candidate 0 is aligned with the omitted target while candidate 1 best
    # reconstructs the candidate-only target.
    corrections = np.array([[1.0, 0.0], [0.0, 0.6], [8.0, 0.0]])
    gram = corrections @ corrections.T
    full = exact_full_target_fixed_greedy(gram, 1, candidates=[0, 1])
    contained = exact_contained_target_fixed_greedy(gram, [0, 1], 1)
    assert full.order.tolist() == [0]
    assert contained.order.tolist() == [0]  # contained target still prefers unit 0 here
    omitted = np.array([[0.0, 8.0]])
    corrections = np.concatenate((corrections[:2], omitted), axis=0)
    gram = corrections @ corrections.T
    full = exact_full_target_fixed_greedy(gram, 1, candidates=[0, 1])
    contained = exact_contained_target_fixed_greedy(gram, [0, 1], 1)
    assert full.order.tolist() == [1]
    assert contained.order.tolist() == [0]


def test_exclusion_regret_is_nonnegative_and_boundary_limited() -> None:
    diagonal = np.diag([9.0, 7.0, 4.0, 1.0])
    units, regret = selected_exclusion_regret(diagonal, 2, boundary_width=1)
    assert units.tolist() == [1]
    assert regret.shape == (1,)
    assert regret[0] >= 0.0
