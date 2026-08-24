from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oracle_study.average_rate_allocator import RateOption
from oracle_study.d1_route_objective import (
    D1BoundaryProblem,
    adaptive_outsider_ranks,
    additive_local_damage_limit,
    build_d1_boundary_problem,
    d1_group_option_allocate,
    directional_route_loss,
    functional_swap_severity,
    exact_candidate_logit_vjps,
    option_boundary_effects,
    routing_mass_severity,
    run_d1_oracle_policies,
    token_option_deltas,
)
from oracle_study.split_interaction_field import split_projection_responses


def _option(pages: int, states: list[int]) -> RateOption:
    return RateOption(
        pages=pages,
        damage=0.0,
        states=np.asarray(states, np.int64),
        source="test",
        page_price=0.0,
        coordinate_sweeps=0,
        local_passes=0,
    )


def _one_boundary() -> D1BoundaryProblem:
    return D1BoundaryProblem(
        selected_expert_ids=np.asarray([0]),
        outsider_expert_ids=np.asarray([1]),
        q4_margins=np.asarray([1.0]),
        margin_sensitivities=np.asarray([[1.0, 0.0]]),
        severity=np.asarray([1.0]),
        temperature=np.asarray([0.1]),
        safety_margin=np.asarray([0.0]),
    )


def test_boundary_problem_uses_exact_q4_rank_pairs_and_unique_logit_gradients() -> None:
    logits = np.asarray([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.9, 1.8])
    gradients = np.arange(30, dtype=np.float64).reshape(10, 3)
    problem = build_d1_boundary_problem(
        logits,
        gradients,
        selected_ranks=(7, 8),
        outsider_ranks=(9, 10),
        temperature=0.2,
    )
    assert problem.boundaries == 4
    assert np.array_equal(problem.selected_expert_ids, [6, 6, 7, 7])
    assert np.array_equal(problem.outsider_expert_ids, [8, 9, 8, 9])
    np.testing.assert_allclose(
        problem.margin_sensitivities,
        gradients[problem.selected_expert_ids] - gradients[problem.outsider_expert_ids],
    )
    assert np.all(problem.q4_margins > 0.0)


def test_adaptive_outsider_pool_and_additive_guardrail() -> None:
    logits = np.asarray([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0,
                         2.9, 2.8, 2.1, 2.0, 1.95])
    assert adaptive_outsider_ranks(logits, 0.25) == (9, 10)
    assert adaptive_outsider_ranks(logits, 2.0, max_outsiders=3) == (9, 10, 11)
    assert additive_local_damage_limit(0.01, 2.0, 0.05) == pytest.approx(0.11)


def test_exact_option_deltas_are_token_dependent() -> None:
    q2 = (
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        np.asarray([[0.5, 0.0], [0.0, 0.5]]),
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
    )
    q4 = (
        np.asarray([[2.0, 0.0], [0.0, 2.0]]),
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        np.asarray([[2.0, 0.0], [0.0, 2.0]]),
    )
    frontier = ((_option(0, [0, 0]), _option(6, [7, 7])),)
    first = token_option_deltas(
        (split_projection_responses(q2, q4, np.asarray([1.0, 0.0])),),
        frontier,
    )[0]
    second = token_option_deltas(
        (split_projection_responses(q2, q4, np.asarray([0.0, 1.0])),),
        frontier,
    )[0]
    assert not np.array_equal(first[0], second[0])
    np.testing.assert_allclose(first[1], 0.0, atol=1e-15)
    np.testing.assert_allclose(second[1], 0.0, atol=1e-15)


def test_boundary_projection_uses_execution_weights_and_signed_effects() -> None:
    problem = _one_boundary()
    deltas = (
        np.asarray([[-2.0, 0.0], [0.0, 0.0]]),
        np.asarray([[0.0, 0.0], [1.0, 0.0]]),
    )
    effects = option_boundary_effects(deltas, [0.75, 0.25], problem)
    np.testing.assert_allclose(effects[0].reshape(-1), [-1.5, 0.0])
    np.testing.assert_allclose(effects[1].reshape(-1), [0.0, 0.25])
    destabilizing = directional_route_loss([1.0], [-1.5], temperature=0.1)
    stabilizing = directional_route_loss([1.0], [0.5], temperature=0.1)
    assert destabilizing > stabilizing


def test_d1_allocator_uses_pair_exchange_under_local_guardrail() -> None:
    problem = _one_boundary()
    frontiers = (
        (_option(0, [0]), _option(1, [1])),
        (_option(0, [0]), _option(1, [1])),
    )
    effects = (
        np.asarray([[-2.0], [0.0]]),
        np.asarray([[0.0], [0.1]]),
    )
    local_features = (
        np.asarray([[1.0, 0.0], [0.0, 0.0]]),
        np.asarray([[0.0, 1.0], [0.0, 0.0]]),
    )
    trace = d1_group_option_allocate(
        frontiers,
        effects,
        problem,
        page_budget=1,
        seed_indices=np.asarray([0, 1]),
        local_option_features=local_features,
        local_router_weights=np.ones(2),
        local_damage_limit=1.0,
        max_coordinate_sweeps=3,
        max_pair_passes=3,
    )
    np.testing.assert_array_equal(trace.option_indices, [1, 0])
    assert trace.pages == 1
    assert trace.local_damage == pytest.approx(1.0)
    assert trace.predicted_crossings == 0
    assert trace.pair_passes >= 1


def test_swap_severity_ablations_are_nonnegative_and_functional() -> None:
    logits = np.asarray([4.0, 3.0, 2.0])
    problem = D1BoundaryProblem(
        selected_expert_ids=np.asarray([0, 1]),
        outsider_expert_ids=np.asarray([2, 2]),
        q4_margins=np.asarray([2.0, 1.0]),
        margin_sensitivities=np.ones((2, 2)),
        severity=np.ones(2),
        temperature=np.ones(2),
        safety_margin=np.zeros(2),
    )
    mass = routing_mass_severity(logits, problem)
    outputs = np.asarray([[2.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    functional = functional_swap_severity(logits, outputs, problem)
    assert np.all(mass > 0.0)
    np.testing.assert_allclose(functional, mass * [4.0, 1.0])


def test_d1_seed_must_pass_guardrail() -> None:
    with pytest.raises(ValueError, match="guardrail"):
        d1_group_option_allocate(
            ((_option(0, [0]),),),
            (np.asarray([[-2.0]]),),
            _one_boundary(),
            0,
            [0],
            local_option_features=(np.asarray([[2.0]]),),
            local_router_weights=[1.0],
            local_damage_limit=1.0,
        )


def test_experiment_a_replays_each_final_policy_exactly_once() -> None:
    problem = _one_boundary()
    frontiers = (
        (_option(0, [0]), _option(1, [1])),
        (_option(0, [0]), _option(1, [1])),
    )
    effects = (
        np.asarray([[-2.0], [0.0]]),
        np.asarray([[0.0], [0.1]]),
    )
    local = (
        np.asarray([[1.0, 0.0], [0.0, 0.0]]),
        np.asarray([[0.0, 1.0], [0.0, 0.0]]),
    )
    reference = np.arange(10, dtype=np.float64)
    calls = []

    final_reference = np.asarray([3.0, 2.0, 1.0])

    def replay(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        calls.append(indices.copy())
        return reference.copy(), final_reference.copy()

    results = run_d1_oracle_policies(
        frontiers,
        effects,
        {"strict": problem, "mass": problem},
        1,
        [0, 1],
        local_option_features=local,
        local_router_weights=[1.0, 1.0],
        local_damage_limit=1.0,
        q4_d1_logits=reference,
        exact_replay=replay,
        q4_final_logits=final_reference,
    )
    assert [result.policy for result in results] == ["mass", "strict"]
    assert len(calls) == 2
    assert all(result.exact_top8_agreement == 1.0 for result in results)
    assert all(result.final_logit_kl == 0.0 for result in results)


def test_exact_candidate_vjps_select_same_token_only() -> None:
    hidden = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4).requires_grad_(True)
    weight = torch.asarray([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 0.0],
    ])
    logits = hidden @ weight.T
    gradients = exact_candidate_logit_vjps(logits, hidden, 1, [2, 0])
    np.testing.assert_allclose(gradients, weight[[2, 0]].numpy())
