from __future__ import annotations

from itertools import product

import numpy as np
import pytest
import torch

import oracle_study.set_utility_oracles as set_utility_oracles
from oracle_study.neuron_selector import FactorizedUnitOutputs
from oracle_study.set_utility_oracles import (
    STATE_00,
    STATE_D,
    STATE_G,
    STATE_GD,
    build_nested_template_bank,
    charged_template_candidates,
    charged_top1_template_candidates,
    charged_top2_union_candidates,
    exact_hybrid_unit_base_gram_greedy,
    exact_hybrid_unit_fixed_greedy,
    exact_monotone_unit_local_search,
    hybrid_state_output,
    hybrid_state_pages,
    utility_weighted_jaccard_distance,
)
from oracle_study.sparse_streaming import qenergy


def _outputs(seed: int = 0, units: int = 4, output: int = 5) -> FactorizedUnitOutputs:
    rng = np.random.default_rng(seed)
    y00 = rng.normal(size=(units, output))
    y10 = y00 + rng.normal(size=(units, output))
    y01 = y00 + rng.normal(size=(units, output))
    y11 = y00 + rng.normal(size=(units, output))
    return FactorizedUnitOutputs(y00, y10, y01, y11, rng.normal(size=units), rng.normal(size=units))


def _brute_forward(outputs: FactorizedUnitOutputs, budget: int, proxy: np.ndarray, beta: float):
    contributions = np.stack((outputs.y00, outputs.y10, outputs.y01, outputs.y11))
    transitions = ((STATE_00, STATE_G, 2), (STATE_00, STATE_D, 1),
                   (STATE_00, STATE_GD, 3), (STATE_G, STATE_GD, 1),
                   (STATE_D, STATE_GD, 2))
    states = np.zeros(outputs.units, dtype=np.int64)
    residual = outputs.target_output - outputs.base_output
    pages = 0
    actions = []
    while pages < budget:
        candidates = []
        for transition, (source, destination, cost) in enumerate(transitions):
            for unit in np.flatnonzero(states == source).tolist():
                if pages + cost > budget:
                    continue
                vector = contributions[destination, unit] - contributions[source, unit]
                gain = qenergy(residual, proxy=proxy, beta=beta) - qenergy(
                    residual - vector, proxy=proxy, beta=beta,
                )
                action_id = transition * outputs.units + unit
                candidates.append((gain / cost, gain, -cost, -action_id,
                                   action_id, unit, source, destination, cost, vector))
        if not candidates:
            break
        *_, action_id, unit, source, destination, cost, vector = max(candidates, key=lambda row: row[:4])
        gain = qenergy(residual, proxy=proxy, beta=beta) - qenergy(
            residual - vector, proxy=proxy, beta=beta,
        )
        residual -= vector
        states[unit] = destination
        pages += cost
        actions.append((action_id, source, destination, cost, pages, gain))
    return states, residual, actions


def test_hybrid_forward_matches_direct_nonidentity_proxy_and_endpoint() -> None:
    outputs = _outputs(18, units=5, output=7)
    proxy = np.random.default_rng(9).normal(size=(7, 3))
    beta = 0.43
    expected_states, expected_residual, expected_actions = _brute_forward(
        outputs, 3 * outputs.units, proxy, beta,
    )
    trace = exact_hybrid_unit_fixed_greedy(
        outputs, 3 * outputs.units, proxy=proxy, beta=beta,
    )
    assert trace.pages == 3 * outputs.units
    assert np.array_equal(trace.states, expected_states)
    assert np.all(trace.states == STATE_GD)
    assert [action.action_id for action in trace.actions] == [row[0] for row in expected_actions]
    np.testing.assert_allclose(
        [action.marginal_gain for action in trace.actions],
        [row[-1] for row in expected_actions], rtol=1e-12, atol=1e-12,
    )
    np.testing.assert_allclose(trace.output, outputs.target_output, rtol=1e-12, atol=1e-12)
    assert trace.residual_energy == pytest.approx(qenergy(expected_residual, proxy=proxy, beta=beta), abs=1e-12)


def test_hybrid_direct_coherent_action_budget_tie_and_validation() -> None:
    zeros = np.zeros((1, 1))
    direct = FactorizedUnitOutputs(zeros, np.array([[-5.0]]), np.array([[-4.0]]),
                                   np.array([[3.0]]), np.zeros(1), np.zeros(1))
    trace = exact_hybrid_unit_fixed_greedy(direct, 3)
    assert [(action.from_state, action.to_state) for action in trace.actions] == [(STATE_00, STATE_GD)]
    assert trace.residual_energy == pytest.approx(0.0)

    tied = FactorizedUnitOutputs(np.zeros((2, 1)), np.zeros((2, 1)),
                                 np.zeros((2, 1)), np.zeros((2, 1)),
                                 np.zeros(2), np.zeros(2))
    one_page = exact_hybrid_unit_fixed_greedy(tied, 1)
    assert one_page.actions[0].unit == 0
    assert one_page.actions[0].to_state == STATE_D
    assert hybrid_state_pages(one_page.states) == 1
    with pytest.raises(ValueError, match="page_budget"):
        exact_hybrid_unit_fixed_greedy(tied, 7)


def test_hybrid_state_output_matches_manual_state_sum() -> None:
    outputs = _outputs(4, units=4, output=3)
    states = np.array([STATE_00, STATE_G, STATE_D, STATE_GD])
    expected = outputs.y00[0] + outputs.y10[1] + outputs.y01[2] + outputs.y11[3]
    np.testing.assert_allclose(hybrid_state_output(outputs, states), expected)
    assert hybrid_state_pages(states) == 6


def test_torch_base_gram_matches_numpy_path_under_dense_and_proxy_metric() -> None:
    outputs = _outputs(91, units=6, output=8)
    rng = np.random.default_rng(92)
    factor = rng.normal(size=(8, 8))
    metric = factor.T @ factor + 0.1 * np.eye(8)
    proxy = rng.normal(size=(8, 3))
    beta = 0.23
    expected = exact_hybrid_unit_fixed_greedy(
        outputs, 18, metric=metric, proxy=proxy, beta=beta,
    )
    actual = exact_hybrid_unit_base_gram_greedy(
        outputs, [0, 5, 18], metric=metric, proxy=proxy, beta=beta,
        device="cpu", dtype=torch.float64,
    )
    assert actual.order.tolist() == [action.action_id for action in expected.actions]
    np.testing.assert_allclose(
        actual.gains, [action.marginal_gain for action in expected.actions],
        rtol=2e-12, atol=2e-12,
    )
    assert actual.cumulative_pages.tolist() == [
        action.cumulative_pages for action in expected.actions
    ]
    assert np.all(actual.terminal_states == STATE_GD)
    assert actual.terminal_pages == 18
    assert actual.primitive_gram_shape == (24, 24)
    assert actual.gram_device == "cpu"
    assert actual.gram_dtype == "float64"
    assert actual.primitive_gram_device_bytes == 24 * 24 * 8
    assert actual.candidate_evaluations >= len(actual.actions)
    timings = (
        actual.gram_build_seconds,
        actual.gram_transfer_seconds,
        actual.greedy_seconds,
        actual.snapshot_seconds,
    )
    assert all(value >= 0.0 for value in timings)
    assert actual.total_seconds >= sum(timings)
    np.testing.assert_allclose(actual.terminal_output, outputs.target_output, atol=1e-12)
    assert actual.terminal_residual_energy == pytest.approx(0.0, abs=1e-20)
    assert actual.snapshots[18].best_prefix == len(actual.actions)
    assert actual.snapshots[18].residual_energy == pytest.approx(0.0, abs=1e-20)


def test_base_gram_shared_path_returns_exact_best_profitable_prefixes() -> None:
    outputs = _outputs(93, units=5, output=7)
    proxy = np.random.default_rng(94).normal(size=(7, 2))
    beta = 0.17
    budgets = [0, 1, 4, 7, 10]
    trace = exact_hybrid_unit_base_gram_greedy(
        outputs, budgets, proxy=proxy, beta=beta,
        device="cpu", dtype=torch.float64,
    )
    expected_path = exact_hybrid_unit_fixed_greedy(
        outputs, max(budgets), proxy=proxy, beta=beta,
    )
    assert trace.order.tolist() == [action.action_id for action in expected_path.actions]
    prefix_pages = np.asarray((0, *trace.cumulative_pages.tolist()))
    direct_energies = []
    for prefix in range(len(trace.actions) + 1):
        states = np.zeros(outputs.units, dtype=np.int64)
        for action in trace.actions[:prefix]:
            assert states[action.unit] == action.from_state
            states[action.unit] = action.to_state
        direct_energies.append(qenergy(
            outputs.target_output - hybrid_state_output(outputs, states),
            proxy=proxy, beta=beta,
        ))
    np.testing.assert_allclose(
        trace.path_residual_energies, direct_energies, rtol=2e-12, atol=2e-12,
    )
    for budget in budgets:
        feasible = np.flatnonzero(prefix_pages <= budget)
        best = min(
            feasible.tolist(),
            key=lambda prefix: (direct_energies[prefix], prefix),
        )
        snapshot = trace.snapshots[budget]
        assert snapshot.best_prefix == best
        assert snapshot.pages == prefix_pages[best]
        assert snapshot.order == tuple(trace.order[:best].tolist())
        assert snapshot.residual_energy == pytest.approx(
            direct_energies[best], rel=2e-12, abs=2e-12,
        )


def test_base_gram_shared_path_keeps_zero_prefix_when_early_actions_are_harmful() -> None:
    zeros = np.zeros((1, 1))
    outputs = FactorizedUnitOutputs(
        zeros, np.array([[-5.0]]), np.array([[-4.0]]), np.array([[3.0]]),
        np.zeros(1), np.zeros(1),
    )
    first = exact_hybrid_unit_base_gram_greedy(
        outputs, [1, 2, 3], device="cpu", dtype=torch.float64,
    )
    second = exact_hybrid_unit_base_gram_greedy(
        outputs, [1, 2, 3], device="cpu", dtype=torch.float64,
    )
    assert first.order.tolist() == second.order.tolist() == [2]
    for budget in (1, 2):
        assert first.snapshots[budget].best_prefix == 0
        assert first.snapshots[budget].pages == 0
        assert not first.snapshots[budget].profitable
    assert first.snapshots[3].best_prefix == 1
    assert first.snapshots[3].states.tolist() == [STATE_GD]
    assert first.snapshots[3].profitable
    np.testing.assert_array_equal(
        first.snapshots[3].states, second.snapshots[3].states,
    )
    np.testing.assert_array_equal(first.gains, second.gains)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_base_gram_path_matches_cpu_deterministically() -> None:
    outputs = _outputs(95, units=5, output=7)
    proxy = np.random.default_rng(96).normal(size=(7, 3))
    kwargs = dict(
        page_budgets=[4, 15], proxy=proxy, beta=0.29, dtype=torch.float64,
    )
    cpu = exact_hybrid_unit_base_gram_greedy(outputs, device="cpu", **kwargs)
    first = exact_hybrid_unit_base_gram_greedy(outputs, device="cuda", **kwargs)
    second = exact_hybrid_unit_base_gram_greedy(outputs, device="cuda", **kwargs)
    assert first.order.tolist() == second.order.tolist() == cpu.order.tolist()
    np.testing.assert_allclose(first.gains, cpu.gains, rtol=2e-11, atol=2e-11)
    np.testing.assert_array_equal(
        first.snapshots[15].states, second.snapshots[15].states,
    )


def test_direct_row_norms_match_pairwise_diagonal_under_nonidentity_qmetric() -> None:
    rng = np.random.default_rng(44)
    rows = rng.normal(size=(17, 9))
    factor = rng.normal(size=(9, 9))
    metric = factor.T @ factor + 0.2 * np.eye(9)
    proxy = rng.normal(size=(9, 4))
    beta = 0.37
    expected = np.diag(set_utility_oracles._metric_pairwise(
        rows, metric=metric, proxy=proxy, beta=beta,
    ))
    actual = set_utility_oracles._metric_row_norms(
        rows, metric=metric, proxy=proxy, beta=beta,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


def test_real_size_candidate_scoring_never_builds_full_candidate_gram(monkeypatch) -> None:
    """Guard the 512-unit x 2048-output selector against quadratic allocation."""
    rng = np.random.default_rng(45)
    units, output = 512, 2048
    y00 = np.zeros((units, output), dtype=np.float64)
    y11 = rng.normal(scale=0.01, size=(units, output))
    outputs = FactorizedUnitOutputs(
        y00, y00.copy(), y00.copy(), y11,
        np.zeros(units), np.zeros(units),
    )
    proxy = rng.normal(scale=0.02, size=(output, 4))
    original_pairwise = set_utility_oracles._metric_pairwise
    observed_rows: list[int] = []

    def guarded_pairwise(rows, **kwargs):
        count = int(np.asarray(rows).shape[0])
        observed_rows.append(count)
        assert count <= 16, "large candidate Gram allocation regressed"
        return original_pairwise(rows, **kwargs)

    monkeypatch.setattr(set_utility_oracles, "_metric_pairwise", guarded_pairwise)
    forward = exact_hybrid_unit_fixed_greedy(
        outputs, 3, proxy=proxy, beta=0.1,
    )
    assert forward.pages == 3
    # The forward path scores 3 * 512 eligible actions but needs no pairwise
    # Gram.  Local search builds one only after taking two moves per signed
    # page-delta bucket (at most twelve rows here).
    assert observed_rows == []
    seed = np.zeros(units, dtype=np.int64)
    seed[0] = STATE_GD
    exact_monotone_unit_local_search(
        outputs, seed, 3, shortlist_size=2, max_swap_units=2,
        max_passes=1, proxy=proxy, beta=0.1,
    )
    assert observed_rows and max(observed_rows) <= 12


def test_local_search_is_monotone_and_exactly_three_swap_local_optimum() -> None:
    outputs = _outputs(71, units=4, output=6)
    proxy = np.random.default_rng(72).normal(size=(6, 2))
    beta = 0.31
    seeds = {
        "coherent": np.array([STATE_GD, STATE_GD, STATE_00, STATE_00]),
        "factorized": np.array([STATE_G, STATE_D, STATE_GD, STATE_00]),
        "hybrid": np.array([STATE_D, STATE_G, STATE_D, STATE_G]),
    }
    result = exact_monotone_unit_local_search(
        outputs, seeds, 6, shortlist_size=None, max_swap_units=3,
        proxy=proxy, beta=beta,
    )
    for trace in result.seed_traces.values():
        assert trace.final_energy <= trace.seed_energy + 1e-12
        assert trace.final_pages == hybrid_state_pages(trace.seed_states)
        for step in trace.steps:
            assert 1 <= len(step.moves) <= 3
            assert sum(move.page_delta for move in step.moves) == 0
            assert step.energy_after < step.energy_before
    assert result.residual_energy <= min(trace.seed_energy for trace in result.seed_traces.values()) + 1e-12

    # With all moves admitted, no page-balanced state reassignment touching at
    # most three units can improve the returned basin.
    target = outputs.target_output
    base_energy = qenergy(target - result.output, proxy=proxy, beta=beta)
    for candidate_tuple in product(range(4), repeat=outputs.units):
        candidate = np.asarray(candidate_tuple)
        if hybrid_state_pages(candidate) != result.pages:
            continue
        if np.count_nonzero(candidate != result.states) > 3:
            continue
        candidate_energy = qenergy(
            target - hybrid_state_output(outputs, candidate), proxy=proxy, beta=beta,
        )
        assert candidate_energy >= base_energy - 1e-10


def test_local_search_shortlist_and_ties_are_deterministic() -> None:
    outputs = _outputs(80, units=5, output=4)
    seed = np.array([STATE_GD, STATE_G, STATE_D, STATE_00, STATE_00])
    first = exact_monotone_unit_local_search(
        outputs, seed, 6, shortlist_size=2, max_swap_units=3,
    )
    second = exact_monotone_unit_local_search(
        outputs, seed, 6, shortlist_size=2, max_swap_units=3,
    )
    assert np.array_equal(first.states, second.states)
    assert first.seed_traces["seed"].steps == second.seed_traces["seed"].steps
    assert first.pages == 6
    with pytest.raises(ValueError, match="exceeds"):
        exact_monotone_unit_local_search(outputs, np.full(5, STATE_GD), 6)


def test_weighted_jaccard_and_nested_per_expert_medoid_bank() -> None:
    first = np.array([1, 1, 0, 0], dtype=bool)
    second = np.array([0, 1, 1, 0], dtype=bool)
    assert utility_weighted_jaccard_distance(first, second, [1, 2, 3, 4]) == pytest.approx(4 / 6)

    masks = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
    ], dtype=bool)
    bank = build_nested_template_bank(
        np.array([7, 7, 7, 9, 9]), masks, np.ones_like(masks, dtype=float), 3,
    )
    expert7 = bank.for_expert(7)
    assert expert7.sample_indices == (1, 0, 2)
    assert np.array_equal(expert7.prefix(1), masks[[1]])
    assert np.array_equal(expert7.prefix(2), masks[[1, 0]])
    assert bank.for_expert(9).sample_indices[0] == 3
    assert all(index < 3 for index in expert7.sample_indices)
    with pytest.raises(KeyError):
        bank.for_expert(11)


def test_template_top1_top2_repair_and_physical_accounting() -> None:
    templates = np.array([
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1],
    ], dtype=bool)
    utility = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    top1 = charged_top1_template_candidates(
        templates, utility, applied_units=2,
    )
    assert top1.template_indices == (0,)
    assert top1.candidate_units == (0, 1)
    assert top1.pages == 6
    assert top1.bytes_read == 3072
    assert top1.retained_utility == pytest.approx(9 / 15)
    assert top1.overfetch == pytest.approx(1.0)

    # (0, 1) and (0, 2) retain equal utility, but the former is charged for
    # only three unique units and therefore wins the deterministic tie.
    top2 = charged_top2_union_candidates(
        templates, utility, applied_units=3,
    )
    assert top2.template_indices == (0, 1)
    assert top2.candidate_units == (0, 1, 2)
    assert top2.pages == 9
    assert top2.retained_utility == pytest.approx(12 / 15)

    repaired = charged_top2_union_candidates(
        templates, utility, repair_count=1, applied_units=3,
    )
    assert repaired.template_indices == (0, 2)
    assert repaired.repair_units == (2,)
    assert repaired.candidate_units == (0, 1, 2, 3, 4)
    assert repaired.pages == 15
    assert repaired.bytes_read == 7680
    assert repaired.retained_utility == pytest.approx(1.0)
    assert repaired.overfetch == pytest.approx(5 / 3)

    fixed_repair = charged_template_candidates(
        templates, [0, 1], repair_order=[0, 1, 3, 4], repair_count=1,
        utility=utility, applied_units=3,
    )
    assert fixed_repair.repair_units == (3,)
    assert fixed_repair.candidate_units == (0, 1, 2, 3)
    assert fixed_repair.pages == 12
    assert fixed_repair.retained_utility == pytest.approx(14 / 15)
    assert fixed_repair.overfetch == pytest.approx(4 / 3)

    explicit = charged_template_candidates(
        templates, [0], repair_order=[0, 1, 4, 3], repair_count=2,
        utility=utility,
    )
    assert explicit.repair_units == (4, 3)
    assert explicit.pages == 12
