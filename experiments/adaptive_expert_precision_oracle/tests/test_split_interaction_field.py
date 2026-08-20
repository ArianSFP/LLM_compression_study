from __future__ import annotations

from itertools import product

import numpy as np

from oracle_study.interaction_field import (
    build_interaction_field, factor_joint_gram, field_coordinate_descent,
    field_local_search,
)
from oracle_study.neuron_selector import factorized_unit_outputs, unit_score_metadata
from oracle_study.split_interaction_field import (
    INHERITED_FOUR_STATE_MAP, SPLIT_ONLY_STATES, SPLIT_STATE_PAGE_COSTS,
    build_split_interaction_field, map_four_state_to_split,
    split_coordinate_descent, split_diagonal_dp_seed, split_exact_damage,
    build_split_gram_field, split_gram_coordinate_descent,
    split_gram_local_search,
    split_local_search, split_projection_responses, split_state_output,
)
from oracle_study.unit_set_teacher import down_metric_gram


def _problem(seed=19, *, units=4, width=6, output=7):
    rng = np.random.default_rng(seed)
    g2 = rng.normal(size=(units, width))
    u2 = rng.normal(size=(units, width))
    d2 = rng.normal(size=(output, units))
    g4 = g2 + .3 * rng.normal(size=(units, width))
    u4 = u2 + .3 * rng.normal(size=(units, width))
    d4 = d2 + .3 * rng.normal(size=(output, units))
    x = rng.normal(size=width)
    proxy = rng.normal(size=(output, 2))
    beta = .35
    q2, q4 = (g2, u2, d2), (g4, u4, d4)
    responses = split_projection_responses(q2, q4, x)
    gram = down_metric_gram(d2, d4, proxy=proxy, beta=beta, dtype=np.float64)
    abc = unit_score_metadata(d2, d4, proxy=proxy, beta=beta)
    return q2, q4, x, responses, proxy, beta, gram, abc


def test_hidden_order_costs_and_four_state_embedding():
    q2, q4, x, responses, *_ = _problem()
    old = factorized_unit_outputs(q2, q4, x)
    np.testing.assert_allclose(responses.hidden[:, 0], old.h2)
    np.testing.assert_allclose(responses.hidden[:, 3], old.h4)
    np.testing.assert_array_equal(SPLIT_STATE_PAGE_COSTS, [0, 1, 1, 2, 1, 2, 2, 3])
    np.testing.assert_array_equal(INHERITED_FOUR_STATE_MAP, [0, 6, 1, 7])
    assert SPLIT_ONLY_STATES == {2, 3, 4, 5}
    rows = (old.y00, old.y10, old.y01, old.y11)
    for states in product(range(4), repeat=responses.units):
        expected = sum(rows[state][unit] for unit, state in enumerate(states))
        np.testing.assert_allclose(
            split_state_output(responses, map_four_state_to_split(states)), expected,
        )


def test_full_factor_is_exact_for_every_eight_state_combination():
    _, _, _, responses, proxy, beta, gram, abc = _problem(units=3)
    factor = factor_joint_gram(gram, 2 * gram.units, dtype=np.float64, tolerance=1e-12)
    field = build_split_interaction_field(factor, responses.hidden, abc)
    for states in product(range(8), repeat=responses.units):
        np.testing.assert_allclose(
            field.damage(states),
            split_exact_damage(responses, states, proxy=proxy, beta=beta),
            rtol=1e-9, atol=1e-9,
        )
    np.testing.assert_allclose(field.self_residual, 0.0, atol=3e-10)


def test_split_field_equals_old_field_on_embedded_states():
    q2, q4, x, responses, _, _, gram, abc = _problem(units=3)
    old_outputs = factorized_unit_outputs(q2, q4, x)
    factor = factor_joint_gram(gram, 3, dtype=np.float64)
    old = build_interaction_field(factor, old_outputs.h2, old_outputs.h4, abc)
    split = build_split_interaction_field(factor, responses.hidden, abc)
    for states in product(range(4), repeat=responses.units):
        mapped = map_four_state_to_split(states)
        np.testing.assert_allclose(split.damage(mapped), old.damage(states))
        assert split.pages(mapped) == old.pages(states)


def test_eight_state_exact_optimum_cannot_be_worse():
    _, _, _, responses, proxy, beta, _, _ = _problem(units=3)
    for budget in range(10):
        old = min(
            split_exact_damage(responses, map_four_state_to_split(s), proxy=proxy, beta=beta)
            for s in product(range(4), repeat=3)
            if SPLIT_STATE_PAGE_COSTS[map_four_state_to_split(s)].sum() <= budget
        )
        new = min(
            split_exact_damage(responses, s, proxy=proxy, beta=beta)
            for s in product(range(8), repeat=3)
            if SPLIT_STATE_PAGE_COSTS[np.asarray(s)].sum() <= budget
        )
        assert new <= old + 1e-10


def test_diagonal_dp_and_split_solvers_are_budgeted_and_monotone():
    _, _, _, responses, _, _, gram, abc = _problem(units=7, output=9)
    field = build_split_interaction_field(
        factor_joint_gram(gram, 4, dtype=np.float64), responses.hidden, abc,
    )
    budget = 10
    seed = split_diagonal_dp_seed(field, budget)
    coordinate = split_coordinate_descent(field, seed, budget, max_sweeps=8)
    repaired = split_local_search(field, coordinate.states, budget, shortlist_size=6)
    repeated = split_local_search(field, coordinate.states, budget, shortlist_size=6)
    assert coordinate.pages <= budget and repaired.pages <= budget
    assert coordinate.damage <= field.damage(seed) + 1e-10
    assert repaired.damage <= coordinate.damage + 1e-10
    assert coordinate.interaction_dot_macs == 2 * field.units * field.rank * coordinate.sweeps
    np.testing.assert_array_equal(repaired.states, repeated.states)
    assert repaired.candidate_moves_per_pass == 7 * field.units
    assert repaired.maximum_shortlist == 42


def test_inherited_four_state_solution_is_a_feasible_split_warm_start():
    q2, q4, x, responses, _, _, gram, abc = _problem(units=8, output=10)
    old_outputs = factorized_unit_outputs(q2, q4, x)
    factor = factor_joint_gram(gram, 5, dtype=np.float64)
    old_field = build_interaction_field(factor, old_outputs.h2, old_outputs.h4, abc)
    old = field_coordinate_descent(old_field, np.zeros(8, np.int64), 12)
    old = field_local_search(old_field, old.states, 12, shortlist_size=6)
    split = build_split_interaction_field(factor, responses.hidden, abc)
    mapped = map_four_state_to_split(old.states)
    improved = split_coordinate_descent(split, mapped, 12)
    improved = split_local_search(split, improved.states, 12, shortlist_size=6)
    assert split.pages(mapped) == old.pages
    assert improved.damage <= split.damage(mapped) + 1e-10



def test_exact_gram_field_matches_direct_damage_and_local_terms():
    _, _, _, responses, proxy, beta, gram, abc = _problem(units=3)
    field = build_split_gram_field(gram, responses.hidden, abc)
    for states in product(range(8), repeat=3):
        np.testing.assert_allclose(
            field.damage(states),
            split_exact_damage(responses, states, proxy=proxy, beta=beta),
            rtol=1e-10, atol=1e-10,
        )
    for unit in range(3):
        for state in range(8):
            states = np.full(3, 7, np.int64)
            states[unit] = state
            np.testing.assert_allclose(
                field.damage(states), field.local_damage[unit, state], atol=1e-10,
            )


def test_exact_gram_full_action_solver_dominates_restricted_four_state():
    _, _, _, responses, _, _, gram, abc = _problem(units=9, output=11)
    field = build_split_gram_field(gram, responses.hidden, abc)
    budget = 13
    restricted = tuple(INHERITED_FOUR_STATE_MAP.tolist())
    four_seed = split_diagonal_dp_seed(field, budget, restricted)
    four = split_gram_coordinate_descent(
        field, four_seed, budget, allowed_states=restricted,
    )
    four = split_gram_local_search(
        field, four.states, budget, allowed_states=restricted, shortlist_size=6,
    )
    full_seed = split_diagonal_dp_seed(field, budget)
    candidates = {
        "diagonal": split_gram_coordinate_descent(field, full_seed, budget),
        "inherited": split_gram_coordinate_descent(field, four.states, budget),
    }
    selected = min(candidates.values(), key=lambda trace: trace.damage)
    full = split_gram_local_search(
        field, selected.states, budget, shortlist_size=6,
    )
    assert full.pages <= budget
    assert full.damage <= four.damage + 1e-9



def test_diagonal_dp_is_exact_for_the_eight_state_self_objective():
    _, _, _, responses, _, _, gram, abc = _problem(units=3)
    field = build_split_interaction_field(
        factor_joint_gram(gram, 2, dtype=np.float64), responses.hidden, abc,
    )
    budget = 5
    states = split_diagonal_dp_seed(field, budget)
    observed = sum(
        field.local_damage[unit, state] for unit, state in enumerate(states)
    )
    expected = min(
        sum(field.local_damage[unit, state] for unit, state in enumerate(candidate))
        for candidate in product(range(8), repeat=3)
        if SPLIT_STATE_PAGE_COSTS[np.asarray(candidate)].sum() <= budget
    )
    np.testing.assert_allclose(observed, expected)
    assert field.pages(states) <= budget
