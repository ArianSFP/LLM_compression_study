from __future__ import annotations

import numpy as np

from oracle_study.mxfp4_selective import BitAction
from oracle_study.sparse_streaming import (
    NEURON_REFINEMENT_PAYLOAD_BYTES,
    activation_action_scores,
    activation_shortlist,
    apply_gate_up_tiles,
    apply_neuron_actions,
    base_gram_candidate_greedy,
    brute_force_nested_best,
    exact_neuron_marginal_greedy,
    exact_neuron_marginal_per_page_greedy,
    exact_tile_marginal_greedy,
    fixed_vector_greedy,
    fixed_vector_greedy_candidates,
    gate_up_tiles,
    neuron_levels_from_actions,
    neuron_major_decomposition,
    physical_accounting,
    rank_unit_proxies,
    select_neuron_actions_under_page_budget,
    shortlist_containment_metrics,
    tile_action_page_map,
    tile_pages,
    tile_payload_bytes,
    unit_packet_page_map,
)


def _levels(seed: int = 7, *, units: int = 5, width: int = 7, output: int = 6):
    rng = np.random.default_rng(seed)
    q2 = (
        rng.normal(size=(units, width)),
        rng.normal(size=(units, width)),
        rng.normal(size=(output, units)),
    )
    q3 = tuple(value + 0.15 * rng.normal(size=value.shape) for value in q2)
    q4 = tuple(value + 0.08 * rng.normal(size=value.shape) for value in q3)
    return (q2, q3, q4), rng.normal(size=width)


def test_neuron_major_complete_endpoint_and_mixed_levels_are_exact() -> None:
    levels, activation = _levels()
    decomposition = neuron_major_decomposition(levels, activation)
    path = exact_neuron_marginal_greedy(decomposition)
    assert len(path) == 2 * decomposition.units
    np.testing.assert_allclose(apply_neuron_actions(decomposition, path), decomposition.target_output, atol=1e-11)

    precision = np.array([2, 3, 4, 3, 2])
    actions = [
        BitAction(unit, stage, 0.0)
        for unit, level in enumerate(precision)
        for stage in range(1, level - 1)
    ]
    np.testing.assert_array_equal(neuron_levels_from_actions(actions, 5), precision)
    expected = decomposition.levels[precision - 2, np.arange(5)].sum(axis=0)
    np.testing.assert_allclose(apply_neuron_actions(decomposition, actions), expected, atol=1e-12)


def test_fixed_vector_greedy_matches_brute_force_on_orthogonal_fixture() -> None:
    # Orthogonal fixed vectors make marginal greedy globally optimal at every
    # nested action count, giving a clean exhaustive regression oracle.
    magnitudes = np.array([5.0, 3.0, 2.0, 1.5, 1.0, 0.5])
    contributions = np.zeros((2, 3, 6), dtype=np.float64)
    for action, magnitude in enumerate(magnitudes):
        contributions.reshape(6, 6)[action, action] = magnitude
    target = contributions.sum(axis=(0, 1))
    path = fixed_vector_greedy(contributions, target)
    for count in range(7):
        levels = neuron_levels_from_actions(path[:count], 3) - 2
        best_levels, best_damage = brute_force_nested_best(contributions, target, count)
        approximation = sum(
            (contributions[:depth, unit].sum(axis=0) for unit, depth in enumerate(levels)),
            start=np.zeros(6),
        )
        damage = np.sum((target - approximation) ** 2)
        np.testing.assert_allclose(damage, best_damage, atol=1e-12)
        np.testing.assert_array_equal(levels, best_levels)


def test_neuron_packet_budget_and_one_bpw_accounting() -> None:
    units = 512
    page_map = unit_packet_page_map(units)
    path = [
        BitAction(unit, stage, float(units - unit))
        for unit in range(256)
        for stage in (1, 2)
    ]
    selected = select_neuron_actions_under_page_budget(path, units, page_budget=768)
    assert len(selected.actions) == 512
    assert len(selected.pages) == 768
    accounting = physical_accounting(
        selected.actions,
        page_map,
        NEURON_REFINEMENT_PAYLOAD_BYTES,
        coordinates=units,
        expert_weight_count=3 * 512 * 2048,
    )
    assert accounting.physical_bpw == 1.0
    assert accounting.logical_bpw == 1.0
    assert accounting.page_amplification == 1.0

    q3_only = select_neuron_actions_under_page_budget([BitAction(0, 1, 1.0)], units, page_budget=2)
    q3_accounting = physical_accounting(
        q3_only.actions,
        page_map,
        NEURON_REFINEMENT_PAYLOAD_BYTES,
        coordinates=units,
        expert_weight_count=3 * 512 * 2048,
    )
    assert q3_accounting.unique_pages == 2
    np.testing.assert_allclose(q3_accounting.page_amplification, 4 / 3)


def test_exact_marginal_per_page_is_nested_and_reaches_full_endpoint() -> None:
    levels, activation = _levels(41, units=4)
    decomposition = neuron_major_decomposition(levels, activation)
    selected = exact_neuron_marginal_per_page_greedy(decomposition, page_budget=3 * decomposition.units)
    assert len(selected.actions) == 2 * decomposition.units
    assert len(selected.pages) == 3 * decomposition.units
    depth = np.zeros(decomposition.units, dtype=np.uint8)
    paid: set[int] = set()
    page_map = unit_packet_page_map(decomposition.units)
    for action, recorded_cost in zip(selected.actions, selected.incremental_pages):
        assert action.stage == depth[action.coordinate] + 1
        action_id = (action.stage - 1) * decomposition.units + action.coordinate
        new_pages = set(page_map[action_id]) - paid
        assert len(new_pages) == recorded_cost
        assert recorded_cost == (2 if action.stage == 1 else 1)
        paid.update(new_pages)
        depth[action.coordinate] += 1
    np.testing.assert_allclose(apply_neuron_actions(decomposition, selected.actions), decomposition.target_output, atol=1e-11)


def test_all_page_aligned_tile_shapes_cover_once_and_reconstruct_q4() -> None:
    rng = np.random.default_rng(11)
    hidden, width = 64, 128
    gate2 = rng.normal(size=(hidden, width))
    up2 = rng.normal(size=(hidden, width))
    gate4 = gate2 + rng.normal(scale=0.1, size=gate2.shape)
    up4 = up2 + rng.normal(scale=0.1, size=up2.shape)
    activation = rng.normal(size=width)
    for shape in ((16, 64), (32, 32), (64, 16)):
        assert tile_payload_bytes(shape) == 512
        tiles = tile_pages(hidden, width, shape)
        coverage = np.zeros((hidden, width), dtype=np.uint8)
        for tile in tiles:
            coverage[tile.row_start:tile.row_stop, tile.column_start:tile.column_stop] += 1
        np.testing.assert_array_equal(coverage, np.ones_like(coverage))
        gate, up = apply_gate_up_tiles(
            gate2, up2, gate4, up4, activation, [tile.page for tile in tiles],
            tile_shape=shape, tiles=tiles,
        )
        np.testing.assert_allclose(gate, gate4 @ activation, atol=1e-10)
        np.testing.assert_allclose(up, up4 @ activation, atol=1e-10)
        assert len(tile_action_page_map(tiles)) == len(tiles)


def test_exact_tile_first_choice_matches_brute_force_damage() -> None:
    rng = np.random.default_rng(19)
    gate2 = rng.normal(size=(2, 2))
    up2 = rng.normal(size=(2, 2))
    gate4 = gate2 + rng.normal(scale=0.3, size=(2, 2))
    up4 = up2 + rng.normal(scale=0.3, size=(2, 2))
    down = rng.normal(size=(3, 2))
    activation = rng.normal(size=2)
    tiles = gate_up_tiles(2, 2, (1, 1))
    greedy = exact_tile_marginal_greedy(
        gate2, up2, gate4, up4, down, activation,
        tile_shape=(1, 1), tiles=tiles, max_pages=1,
    )
    target_gate, target_up = gate4 @ activation, up4 @ activation
    target = down @ ((target_gate / (1 + np.exp(-target_gate))) * target_up)
    damages = []
    for tile in tiles:
        gate, up = apply_gate_up_tiles(
            gate2, up2, gate4, up4, activation, [tile], tile_shape=(1, 1), tiles=tiles,
        )
        output = down @ ((gate / (1 + np.exp(-gate))) * up)
        damages.append(np.sum((target - output) ** 2))
    assert greedy[0].tile.page == int(np.argmin(damages))


def test_activation_shortlist_closure_and_containment_metrics() -> None:
    activation = np.array([0.2, 3.0, 1.0])
    deltas = np.zeros((2, 3, 3), dtype=np.float64)
    deltas[0] = np.eye(3)
    deltas[1] = 0.5 * np.eye(3)
    scores = activation_action_scores(activation, deltas, rule="magnitude")
    # Asking for the two top raw actions chooses Q3/Q4 of coordinate one;
    # closure is exact and does not add duplicates.
    shortlist = activation_shortlist(scores, 2)
    np.testing.assert_array_equal(shortlist.action_ids, [1, 4])
    exact = fixed_vector_greedy(deltas * activation[None, :, None])
    metrics = shortlist_containment_metrics(
        exact, shortlist, coordinates=3, exact_count=2,
        contributions=deltas * activation[None, :, None],
        action_pages={action: (action,) for action in range(6)},
    )
    assert metrics.exact_action_recall == 1.0
    assert metrics.importance_weighted_recall == 1.0
    assert metrics.support_only_exact_gain_retained == 1.0
    assert metrics.constrained_exact_gain_retained == 1.0
    assert metrics.exact_gain_retained == 1.0
    assert metrics.pages_required == 2
    assert metrics.physical_bytes == 1024


def test_candidate_rescoring_is_distinct_from_support_only_containment() -> None:
    contributions = np.zeros((2, 2, 4), dtype=np.float64)
    contributions[0, 0, 0] = 3.0
    contributions[0, 1, 1] = 2.0
    contributions[1, 0, 2] = 0.1
    contributions[1, 1, 3] = 0.1
    target = contributions.sum(axis=(0, 1))
    exact = fixed_vector_greedy(contributions, target)
    assert exact[0].coordinate == 0
    constrained = fixed_vector_greedy_candidates(contributions, target, [1], max_actions=1)
    assert [(action.coordinate, action.stage) for action in constrained] == [(1, 1)]
    metrics = shortlist_containment_metrics(
        exact, [1], coordinates=2, exact_count=1, contributions=contributions, target=target,
    )
    assert metrics.support_only_exact_gain_retained == 0.0
    np.testing.assert_allclose(metrics.constrained_exact_gain_retained, 4 / 9)
    np.testing.assert_allclose(metrics.exact_gain_retained, 4 / 9)


def test_base_gram_candidates_match_direct_vectors_nonidentity_geometry() -> None:
    rng = np.random.default_rng(2718)
    coordinates, output = 11, 7
    directions = rng.normal(size=(2 * coordinates, output))
    transform = rng.normal(size=(output, output))
    metric = np.eye(output) + 0.2 * (transform @ transform.T)
    root = np.linalg.cholesky(metric)
    features = directions @ root
    gram = np.asarray((directions @ metric) @ directions.T, np.float32)
    activation = np.asarray(rng.normal(size=coordinates), np.float32)
    expanded = np.concatenate((activation, activation))
    contributions = (features * expanded[:, None]).reshape(2, coordinates, output)
    target = contributions.sum(axis=(0, 1))
    candidates = [
        BitAction(1, 2, 0.0), coordinates + 4, 0, 7,
        BitAction(9, 2, 0.0), 3, coordinates + 6,
    ]
    pages = unit_packet_page_map(coordinates)
    direct = fixed_vector_greedy_candidates(
        contributions, target, candidates,
        max_actions=7, action_pages=pages, page_budget=9,
    )
    based = base_gram_candidate_greedy(
        gram, activation, candidates,
        max_actions=7, action_pages=pages, page_budget=9,
    )
    assert [(a.coordinate, a.stage) for a in based] == [
        (a.coordinate, a.stage) for a in direct
    ]
    np.testing.assert_allclose(
        [action.score for action in based], [action.score for action in direct],
        rtol=2e-5, atol=3e-5,
    )
    paid: set[int] = set()
    expected_costs = []
    for action in based:
        action_id = (action.stage - 1) * coordinates + action.coordinate
        new_pages = set(pages[action_id]) - paid
        expected_costs.append(len(new_pages))
        paid.update(new_pages)
    assert based.incremental_pages == tuple(expected_costs)
    assert based.cumulative_pages == tuple(np.cumsum(expected_costs))
    assert based.pages == tuple(sorted(paid))


def test_base_gram_single_page_fast_path_matches_direct_for_every_budget() -> None:
    """Shared coordinate pages retain the exact residual-aware semantics."""
    rng = np.random.default_rng(314159)
    coordinates, output = 9, 8
    features = np.asarray(rng.normal(size=(2 * coordinates, output)), np.float32)
    gram = np.asarray(features @ features.T, np.float32)
    activation = np.asarray(rng.normal(size=coordinates), np.float32)
    expanded = np.concatenate((activation, activation))
    contributions = (features * expanded[:, None]).reshape(2, coordinates, output)
    target = contributions.sum(axis=(0, 1))
    # Both planes for three adjacent coordinates share one physical page,
    # matching the locked coordinate-packet structure at toy scale.
    pages = {
        stage * coordinates + coordinate: (coordinate // 3,)
        for stage in range(2) for coordinate in range(coordinates)
    }
    for budget in range(4):
        direct = fixed_vector_greedy_candidates(
            contributions, target, range(2 * coordinates),
            action_pages=pages, page_budget=budget,
        )
        based = base_gram_candidate_greedy(
            gram, activation, range(2 * coordinates),
            action_pages=pages, page_budget=budget,
        )
        assert [(action.coordinate, action.stage) for action in based] == [
            (action.coordinate, action.stage) for action in direct
        ]
        np.testing.assert_allclose(
            [action.score for action in based],
            [action.score for action in direct],
            rtol=2e-5, atol=3e-5,
        )
        assert len(based.pages) <= budget


def test_base_gram_candidate_closure_nesting_and_deterministic_ties() -> None:
    # A lone Q4 candidate closes over its Q3 prerequisite.
    closed = base_gram_candidate_greedy(
        np.eye(4), np.array([2.0, 1.0]), [BitAction(1, 2, 0.0)],
        action_pages={1: (7,), 3: (8,)}, page_budget=2,
    )
    assert [(action.coordinate, action.stage) for action in closed] == [(1, 1), (1, 2)]
    assert closed.incremental_pages == (1, 1)
    assert closed.cumulative_pages == (1, 2)

    # Reversed candidate input must not affect equal-score global-ID ties.
    complete = base_gram_candidate_greedy(
        np.eye(6), np.ones(3), list(reversed(range(6))),
        action_pages={action: (action,) for action in range(6)},
    )
    assert [(action.coordinate, action.stage) for action in complete] == [
        (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2),
    ]
    assert complete.actions_for_page_budget(3) == complete.actions[:3]
    assert complete.cumulative_pages[-1] == 6


def test_base_gram_full_candidates_reach_exact_endpoint() -> None:
    rng = np.random.default_rng(99)
    coordinates, output = 8, 10
    features = rng.normal(size=(2 * coordinates, output))
    gram = features @ features.T
    activation = rng.normal(size=coordinates)
    expanded = np.concatenate((activation, activation))
    contributions = (features * expanded[:, None]).reshape(2, coordinates, output)
    target = contributions.sum(axis=(0, 1))
    based = base_gram_candidate_greedy(gram, activation, range(2 * coordinates))
    direct = fixed_vector_greedy_candidates(
        contributions, target, range(2 * coordinates),
    )
    assert [(a.coordinate, a.stage) for a in based] == [(a.coordinate, a.stage) for a in direct]
    assert len(based) == 2 * coordinates
    reconstructed = np.zeros(output)
    depth = np.zeros(coordinates, dtype=np.uint8)
    for action in based:
        assert action.stage == depth[action.coordinate] + 1
        depth[action.coordinate] += 1
        reconstructed += contributions[action.stage - 1, action.coordinate]
    np.testing.assert_allclose(reconstructed, target, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(depth, np.full(coordinates, 2))


def test_unit_proxy_rankings_are_deterministic_permutations() -> None:
    levels, activation = _levels(31, units=4)
    rankings = rank_unit_proxies(levels, activation)
    assert set(rankings) == {
        "independent_unit_correction_norm",
        "q2_intermediate_magnitude",
        "down_weight_aware",
        "first_order_sensitivity",
    }
    for ranking in rankings.values():
        assert ranking.score.shape == (4,)
        np.testing.assert_array_equal(np.sort(ranking.order), np.arange(4))
        np.testing.assert_array_equal(ranking.order, np.lexsort((np.arange(4), -ranking.score)))
