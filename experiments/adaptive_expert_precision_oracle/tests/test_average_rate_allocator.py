from __future__ import annotations

from itertools import product

import numpy as np

from oracle_study.average_rate_allocator import (
    AllocationTrace,
    RateOption,
    allocation_aware_rate_frontiers,
    bounded_group_exchange_allocate,
    exact_group_option_allocate,
    global_group_dual_bound,
    lagrangian_rate_frontier,
    multiple_choice_allocate,
    qmetric_features,
    randomized_joint_metric_factor,
)
from oracle_study.interaction_field import factor_joint_gram
from oracle_study.neuron_selector import unit_score_metadata
from oracle_study.split_interaction_field import (
    SPLIT_STATE_PAGE_COSTS,
    build_split_interaction_field,
    split_coordinate_descent,
    split_diagonal_dp_seed,
    split_local_search,
    split_projection_responses,
)
from oracle_study.unit_set_teacher import down_metric_gram


def _problem(seed=7, *, units=5, width=7, output=9):
    rng = np.random.default_rng(seed)
    q2 = (
        rng.normal(size=(units, width)),
        rng.normal(size=(units, width)),
        rng.normal(size=(output, units)),
    )
    q4 = tuple(value + .25 * rng.normal(size=value.shape) for value in q2)
    x = rng.normal(size=width)
    proxy = rng.normal(size=(output, 2))
    beta = .3
    responses = split_projection_responses(q2, q4, x)
    gram = down_metric_gram(q2[2], q4[2], proxy=proxy, beta=beta, dtype=np.float64)
    abc = unit_score_metadata(q2[2], q4[2], proxy=proxy, beta=beta)
    field = build_split_interaction_field(
        factor_joint_gram(gram, min(2 * units, 6), dtype=np.float64),
        responses.hidden,
        abc,
    )
    return q2, q4, proxy, beta, gram, field


def _option(pages: int, damage: float, state: int = 0) -> RateOption:
    return RateOption(
        pages, damage, np.asarray([state], np.int64), "test", 0.0, 0, 0,
    )


def test_randomized_factor_recovers_full_small_joint_gram_on_cpu():
    q2, q4, proxy, beta, gram, _ = _problem(units=3, output=8)
    factor = randomized_joint_metric_factor(
        q2[2], q4[2], proxy, beta, rank=6, oversample=0,
        power_iterations=1, seed=11, device="cpu",
    )
    joint = np.block([[gram.q44, gram.q42], [gram.q42.T, gram.q22]])
    decoded = np.concatenate((factor.l4, factor.l2), axis=0)
    np.testing.assert_allclose(decoded @ decoded.T, joint, rtol=2e-5, atol=2e-5)


def test_priced_split_solvers_reduce_pages_and_preserve_objective():
    *_, field = _problem(units=7, output=10)
    unpriced_seed = split_diagonal_dp_seed(field, 21)
    priced_seed = split_diagonal_dp_seed(field, 21, page_price=1e12)
    assert field.pages(priced_seed) == 0
    unpriced = split_coordinate_descent(field, unpriced_seed, 21)
    priced = split_coordinate_descent(field, unpriced.states, 21, page_price=1e12)
    priced = split_local_search(field, priced.states, 21, page_price=1e12)
    assert priced.pages == 0


def test_lagrangian_frontier_is_deterministic_nondominated_and_anchored():
    *_, field = _problem(units=8, output=11)
    kwargs = dict(
        maximum_pages=24,
        target_pages=(6, 12, 18),
        price_ratios=(16.0, 4.0, 1.0, .25, .0),
        coordinate_sweeps=4,
        local_shortlist=4,
        local_swap_units=2,
        local_max_passes=3,
    )
    left_trace = lagrangian_rate_frontier(field, **kwargs)
    right_trace = lagrangian_rate_frontier(field, **kwargs)
    left, right = left_trace.options, right_trace.options
    assert [(o.pages, o.damage) for o in left] == [(o.pages, o.damage) for o in right]
    assert all(left[i].pages < left[i + 1].pages for i in range(len(left) - 1))
    assert all(left[i].damage > left[i + 1].damage for i in range(len(left) - 1))
    for option in left:
        assert option.pages == field.pages(option.states)
        np.testing.assert_allclose(option.damage, field.damage(option.states))
    assert any(option.pages <= 6 for option in left)
    assert left[-1].pages <= 24
    assert left_trace.anchor_solves == 5
    assert left_trace.price_solves == 5
    assert left_trace.coordinate_sweeps > 0 and left_trace.local_passes > 0


def test_multiple_choice_allocator_matches_brute_force():
    frontiers = (
        (_option(0, 9), _option(2, 4), _option(4, 1)),
        (_option(0, 8), _option(3, 2), _option(5, 0)),
        (_option(0, 3), _option(1, 2), _option(2, 1)),
    )
    weights = np.asarray([1.0, .5, 2.0])
    budget = 7
    trace = multiple_choice_allocate(frontiers, budget, weights)
    expected = min(
        (
            sum(frontiers[e][choice[e]].damage * weights[e] for e in range(3)),
            sum(frontiers[e][choice[e]].pages for e in range(3)),
            choice,
        )
        for choice in product(*(range(len(frontier)) for frontier in frontiers))
        if sum(frontiers[e][choice[e]].pages for e in range(3)) <= budget
    )
    np.testing.assert_array_equal(trace.option_indices, expected[2])
    assert trace.pages == expected[1]
    np.testing.assert_allclose(trace.objective, expected[0])


def test_exact_two_expert_allocator_matches_global_combined_energy():
    frontiers = (
        (_option(0, 0), _option(2, 0), _option(4, 0)),
        (_option(0, 0), _option(3, 0), _option(5, 0)),
    )
    residuals = (
        np.asarray([[3.0, 0.0], [1.0, 1.0], [0.0, 0.0]]),
        np.asarray([[-2.0, 1.0], [0.0, 1.0], [0.0, 0.0]]),
    )
    alpha = np.asarray([.7, .3])
    features = tuple(qmetric_features(value, None, 0.0) for value in residuals)
    trace = exact_group_option_allocate(
        frontiers, features, alpha, 5, np.asarray([0, 0]),
        max_coordinate_sweeps=4, max_pair_passes=4,
    )
    expected = min(
        (
            float((
                alpha[0] * residuals[0][left]
                + alpha[1] * residuals[1][right]
            ) @ (
                alpha[0] * residuals[0][left]
                + alpha[1] * residuals[1][right]
            )),
            left,
            right,
        )
        for left, right in product(range(3), repeat=2)
        if frontiers[0][left].pages + frontiers[1][right].pages <= 5
    )
    np.testing.assert_array_equal(trace.option_indices, expected[1:])
    np.testing.assert_allclose(trace.objective, expected[0])


def test_exact_self_priced_dp_matches_brute_force():
    *_, field = _problem(units=3)
    price = .17
    states = split_diagonal_dp_seed(field, 7, page_price=price)
    observed = sum(
        field.local_damage[unit, state]
        for unit, state in enumerate(states)
    ) + price * field.pages(states)
    expected = min(
        sum(field.local_damage[unit, state] for unit, state in enumerate(candidate))
        + price * int(SPLIT_STATE_PAGE_COSTS[np.asarray(candidate)].sum())
        for candidate in product(range(8), repeat=3)
        if SPLIT_STATE_PAGE_COSTS[np.asarray(candidate)].sum() <= 7
    )
    np.testing.assert_allclose(observed, expected)

def test_allocation_aware_frontier_repairs_only_selected_columns_deterministically():
    *_, left_field = _problem(seed=31, units=7, output=10)
    *_, right_field = _problem(seed=37, units=7, output=10)
    kwargs = dict(
        maximum_pages=21,
        target_pages=(0, 6, 12, 21),
        price_ratios=(8.0, 2.0, .5, .125, 0.0),
        coordinate_sweeps=4,
        local_shortlist=4,
        local_swap_units=2,
        local_max_passes=3,
    )
    frontiers = (
        lagrangian_rate_frontier(left_field, **kwargs).options,
        lagrangian_rate_frontier(right_field, **kwargs).options,
    )
    weights = np.asarray([0.7, 0.3])
    coarse = multiple_choice_allocate(frontiers, 23, weights)
    call = dict(
        page_budget=23,
        burst_cap_pages=21,
        weights=weights,
        coordinate_sweeps=4,
        local_shortlist=4,
        local_swap_units=2,
        local_max_passes=3,
        max_rounds=3,
        adaptive_price_multipliers=(.5, 2.0),
    )
    first = allocation_aware_rate_frontiers(
        (left_field, right_field), frontiers, **call,
    )
    second = allocation_aware_rate_frontiers(
        (left_field, right_field), frontiers, **call,
    )
    assert first.rounds >= 1
    assert first.selected_repairs == 2 * first.rounds
    assert first.adaptive_price_solves >= 0
    assert first.allocation.objective <= coarse.objective + 1e-12
    assert first.allocation.pages <= 23
    assert [
        (option.pages, option.damage, option.states.tolist())
        for frontier in first.frontiers for option in frontier
    ] == [
        (option.pages, option.damage, option.states.tolist())
        for frontier in second.frontiers for option in frontier
    ]


def test_four_expert_exchange_and_global_dual_bound_bracket_brute_force():
    frontiers = tuple((
        _option(0, 0.0, 0),
        _option(1, 0.0, 1),
        _option(2, 0.0, 2),
    ) for _ in range(4))
    raw = (
        np.asarray([[2.0, 0.0], [1.0, 1.0], [0.0, .2]]),
        np.asarray([[-1.8, .1], [-.8, 1.0], [0.0, .2]]),
        np.asarray([[.2, 2.0], [1.0, .9], [.2, 0.0]]),
        np.asarray([[.1, -1.8], [1.0, -.7], [.2, 0.0]]),
    )
    alpha = np.asarray([.4, .3, .2, .1])
    seed_residual = sum(
        (alpha[e] * raw[e][0] for e in range(4)), np.zeros(2),
    )
    seed = AllocationTrace(
        np.zeros(4, np.int64), 0, float(seed_residual @ seed_residual),
    )
    improved = bounded_group_exchange_allocate(
        frontiers, raw, alpha, 5, seed,
        exchange_sizes=(4,), shortlist_size=3, max_passes=2,
    )
    expected = min(
        (
            float(sum(
                (alpha[e] * raw[e][choice[e]] for e in range(4)),
                np.zeros(2),
            ) @ sum(
                (alpha[e] * raw[e][choice[e]] for e in range(4)),
                np.zeros(2),
            )),
            choice,
        )
        for choice in product(range(3), repeat=4)
        if sum(frontiers[e][choice[e]].pages for e in range(4)) <= 5
    )
    np.testing.assert_allclose(improved.objective, expected[0], atol=1e-12)
    bound = global_group_dual_bound(
        frontiers, raw, alpha, 5, improved,
        max_iterations=256, relative_tolerance=1e-9,
    )
    assert 0.0 <= bound.lower_bound <= expected[0] + 1e-10
    assert expected[0] <= bound.upper_bound + 1e-10
    np.testing.assert_allclose(
        bound.absolute_gap,
        bound.upper_bound - bound.lower_bound,
    )
