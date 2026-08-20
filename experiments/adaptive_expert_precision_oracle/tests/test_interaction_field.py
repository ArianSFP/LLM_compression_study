from __future__ import annotations

from itertools import product

import numpy as np

from oracle_study.interaction_field import (
    assemble_joint_gram,
    build_interaction_field,
    encode_interaction_factor,
    factor_exact_proxy_plus_tail,
    factor_joint_gram,
    field_coordinate_descent,
    field_forward_greedy,
    field_local_search,
    make_encoded_factor_self_safe,
    make_factor_self_safe,
    quantize_interaction_factor,
    relaxed_field_descent,
    round_relaxed_pages,
    walsh_hadamard,
)
from oracle_study.neuron_selector import unit_score_metadata
from oracle_study.set_utility_oracles import STATE_PAGE_COSTS
from oracle_study.unit_set_teacher import down_metric_gram


def _problem(seed: int = 7, *, output: int = 7, units: int = 4):
    rng = np.random.default_rng(seed)
    down2 = rng.normal(size=(output, units))
    down4 = down2 + 0.35 * rng.normal(size=(output, units))
    proxy = rng.normal(size=(output, 2))
    beta = 0.4
    h2 = rng.normal(size=units)
    h4 = h2 + 0.5 * rng.normal(size=units)
    gram = down_metric_gram(down2, down4, proxy=proxy, beta=beta, dtype=np.float64)
    abc = unit_score_metadata(down2, down4, proxy=proxy, beta=beta)
    return down2, down4, proxy, beta, h2, h4, gram, abc


def _exact_damage(down2, down4, proxy, beta, h2, h4, states):
    columns = []
    for unit, state in enumerate(states):
        if state == 0:
            columns.append(h4[unit] * down4[:, unit] - h2[unit] * down2[:, unit])
        elif state == 1:
            columns.append(h4[unit] * (down4[:, unit] - down2[:, unit]))
        elif state == 2:
            columns.append((h4[unit] - h2[unit]) * down4[:, unit])
        else:
            columns.append(np.zeros(down2.shape[0]))
    residual = np.sum(columns, axis=0)
    return float(residual @ residual + beta * np.sum((residual @ proxy) ** 2))


def test_joint_gram_and_full_factor_reconstruct_exact_geometry():
    down2, down4, proxy, beta, _, _, gram, _ = _problem()
    direct = np.concatenate((down4, down2), axis=1).T
    projected = direct @ proxy
    joint = assemble_joint_gram(gram)
    np.testing.assert_allclose(joint, direct @ direct.T + beta * (projected @ projected.T))
    factor = factor_joint_gram(gram, 2 * gram.units, dtype=np.float64, tolerance=1e-12)
    np.testing.assert_allclose(factor.joint() @ factor.joint().T, joint, rtol=1e-10, atol=1e-10)


def test_exact_proxy_plus_full_euclidean_tail_reconstructs_metric():
    down2, down4, proxy, beta, _, _, gram, _ = _problem()
    factor = factor_exact_proxy_plus_tail(
        down2, down4, proxy, beta, 2 * gram.units, dtype=np.float64, tolerance=1e-12,
    )
    np.testing.assert_allclose(
        factor.joint() @ factor.joint().T,
        assemble_joint_gram(gram),
        rtol=1e-10,
        atol=1e-10,
    )
    assert factor.exact_rank == proxy.shape[1]


def test_full_factor_field_equals_direct_qenergy_for_every_state():
    down2, down4, proxy, beta, h2, h4, gram, abc = _problem(units=3)
    factor = factor_joint_gram(gram, 2 * gram.units, dtype=np.float64, tolerance=1e-12)
    field = build_interaction_field(factor, h2, h4, abc)
    for states in product(range(4), repeat=gram.units):
        np.testing.assert_allclose(
            field.damage(states),
            _exact_damage(down2, down4, proxy, beta, h2, h4, states),
            rtol=1e-9,
            atol=1e-9,
        )
    np.testing.assert_allclose(field.self_residual, 0.0, atol=2e-10)


def test_truncated_factor_keeps_every_local_state_exact():
    _, _, _, _, h2, h4, gram, abc = _problem()
    factor = factor_joint_gram(gram, 2, dtype=np.float64)
    field = build_interaction_field(factor, h2, h4, abc)
    for unit in range(field.units):
        for state in range(4):
            states = np.full(field.units, 3, np.int64)
            states[unit] = state
            np.testing.assert_allclose(field.damage(states), field.local_damage[unit, state], atol=1e-10)
    assert np.min(field.self_residual) >= -1e-9


def test_discrete_solvers_are_budgeted_and_monotone_in_field_damage():
    _, _, _, _, h2, h4, gram, abc = _problem(units=6, output=9)
    field = build_interaction_field(factor_joint_gram(gram, 4, dtype=np.float64), h2, h4, abc)
    budget = 9
    forward = field_forward_greedy(field, [budget]).snapshots[budget]
    assert forward.pages <= budget
    coordinate = field_coordinate_descent(field, forward.states, budget, max_sweeps=8)
    assert coordinate.pages <= budget
    assert coordinate.damage <= forward.damage + 1e-10
    repaired = field_local_search(field, coordinate.states, budget, shortlist_size=6)
    assert repaired.pages <= budget
    assert repaired.damage <= coordinate.damage + 1e-10


def test_relaxation_is_feasible_lower_bound_and_rounds_within_budget(monkeypatch):
    _, _, _, _, h2, h4, gram, abc = _problem(units=3, output=6)
    field = build_interaction_field(factor_joint_gram(gram, 3, dtype=np.float64), h2, h4, abc)
    budget = 4
    monkeypatch.setattr(
        np.linalg, "norm",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("spectral setup forbidden")),
    )
    relaxed = relaxed_field_descent(field, budget, max_iterations=5000, tolerance=1e-12)
    np.testing.assert_allclose(relaxed.probabilities.sum(axis=1), 1.0, atol=1e-10)
    assert relaxed.expected_pages <= budget + 1e-8
    discrete = min(
        field.damage(states)
        for states in product(range(4), repeat=field.units)
        if int(STATE_PAGE_COSTS[np.asarray(states)].sum()) <= budget
    )
    assert relaxed.objective <= discrete + 1e-7
    rounded = round_relaxed_pages(relaxed.probabilities, budget)
    assert field.pages(rounded) <= budget


def test_hadamard_and_quantized_factor_accounting_are_deterministic():
    _, _, _, _, _, _, gram, _ = _problem(units=4)
    factor = factor_joint_gram(gram, 4, dtype=np.float64)
    transform = walsh_hadamard(4)
    np.testing.assert_allclose(transform @ transform.T, np.eye(4), atol=1e-12)
    rotated = factor.joint() @ transform
    np.testing.assert_allclose(rotated @ rotated.T, factor.joint() @ factor.joint().T, atol=1e-10)
    first = quantize_interaction_factor(factor, "int4_per_row", hadamard_rotate=True)
    second = quantize_interaction_factor(factor, "int4_per_row", hadamard_rotate=True)
    np.testing.assert_array_equal(first.l4, second.l4)
    np.testing.assert_array_equal(first.l2, second.l2)
    encoded = encode_interaction_factor(factor, "int4_per_row", hadamard_rotate=True)
    np.testing.assert_array_equal(first.l4, encoded.decode().l4)
    np.testing.assert_array_equal(first.l2, encoded.decode().l2)
    assert encoded.payload_bytes == (
        (2 * factor.units * factor.rank + 1) // 2 + 4 * factor.units + 4
    )
    assert first.payload_bytes == encoded.payload_bytes


def test_self_safe_quantization_preserves_psd_local_residuals():
    _, _, _, _, h2, h4, gram, abc = _problem(units=5)
    factor = factor_joint_gram(gram, 4, dtype=np.float64)
    quantized = quantize_interaction_factor(factor, "int4_per_row", hadamard_rotate=True)
    safe, shrink = make_factor_self_safe(quantized, abc)
    assert 0.0 <= shrink <= 1.0
    field = build_interaction_field(safe, h2, h4, abc)
    assert np.min(field.self_residual) >= -1e-8
    encoded = encode_interaction_factor(factor, "int4_per_row", hadamard_rotate=True)
    safe_encoded, stored_shrink = make_encoded_factor_self_safe(encoded, abc)
    assert 0.0 <= stored_shrink <= 1.0
    stored_field = build_interaction_field(safe_encoded.decode(), h2, h4, abc)
    assert np.min(stored_field.self_residual) >= -1e-8
