from __future__ import annotations

import numpy as np

from oracle_study.interaction_allocator import (
    action_ids,
    block_refresh_from_gram,
    build_base_gram,
    encode_factor,
    exact_marginal_fixed_greedy_from_gram,
    low_rank_factor,
    low_rank_marginal_fixed_greedy,
    page_aware_fixed_greedy,
    recovery_curve_from_gram,
    separate_plane_pages,
)
from oracle_study.mxfp4_selective import canonical_selector_label, exact_marginal_order


def _assert_nested_complete(order, coordinates: int) -> None:
    assert len(order) == 2 * coordinates
    depth = np.zeros(coordinates, np.uint8)
    for action in order:
        assert action.stage == depth[action.coordinate] + 1
        depth[action.coordinate] += 1
    assert np.all(depth == 2)


def test_base_gram_matches_existing_exact_random() -> None:
    rng = np.random.default_rng(20260818)
    deltas = rng.normal(size=(2, 29, 17)).astype(np.float32)
    activation = rng.normal(size=29).astype(np.float32)
    contributions = deltas * activation[None, :, None]
    old = exact_marginal_order(contributions)
    gram = build_base_gram(deltas, device="cpu")
    new = exact_marginal_fixed_greedy_from_gram(gram, activation)
    np.testing.assert_array_equal(action_ids(old, 29), action_ids(new, 29))
    _assert_nested_complete(new, 29)
    assert recovery_curve_from_gram(new, gram, activation)[-1] == 1.0


def test_base_gram_matches_direct_metric_nonidentity() -> None:
    rng = np.random.default_rng(9)
    deltas = rng.normal(size=(2, 23, 11)).astype(np.float32)
    activation = rng.normal(size=23).astype(np.float32)
    transform = rng.normal(size=(11, 6)).astype(np.float32)
    metric = np.eye(11, dtype=np.float32) + 0.2 * (transform @ transform.T)
    gram = build_base_gram(deltas, metric, device="cpu")
    # np.linalg.cholesky returns L with G=L@L.T; row features use D.T@L.
    transformed = np.einsum("sco,op->scp", deltas, np.linalg.cholesky(metric).astype(np.float32))
    direct = exact_marginal_order(transformed * activation[None, :, None])
    based = exact_marginal_fixed_greedy_from_gram(gram, activation)
    np.testing.assert_array_equal(action_ids(direct, 23), action_ids(based, 23))


def test_real_sized_fixture_crossings_and_endpoint() -> None:
    rng = np.random.default_rng(71)
    # Gate/up geometry has 2 x 2048 actions. A compact output rank keeps the
    # regression fast while preserving the real action count and crossings.
    deltas = rng.normal(size=(2, 2048, 8)).astype(np.float32)
    activation = rng.normal(size=2048).astype(np.float32)
    gram = build_base_gram(deltas, device="cpu")
    old = exact_marginal_order(deltas * activation[None, :, None])
    new = exact_marginal_fixed_greedy_from_gram(gram, activation)
    # The first residual-bearing prefix is identical. Once this deliberately
    # rank-8 fixture reaches machine-zero residual, later tie order is not
    # meaningful; crossing counts and the complete endpoint remain identical.
    np.testing.assert_array_equal(action_ids(old[:100], 2048), action_ids(new[:100], 2048))
    old_curve = recovery_curve_from_gram(old, gram, activation)
    new_curve = recovery_curve_from_gram(new, gram, activation)
    old_crossings = [int(np.flatnonzero(old_curve >= target)[0]) + 1 for target in (0.8, 0.9, 0.95, 0.99)]
    new_crossings = [int(np.flatnonzero(new_curve >= target)[0]) + 1 for target in (0.8, 0.9, 0.95, 0.99)]
    assert old_crossings == new_crossings == sorted(new_crossings)
    assert old_curve[-1] == new_curve[-1] == 1.0


def test_block_refresh_is_nested_and_exact_at_one() -> None:
    rng = np.random.default_rng(17)
    deltas = rng.normal(size=(2, 31, 9)).astype(np.float32)
    activation = rng.normal(size=31).astype(np.float32)
    gram = build_base_gram(deltas, device="cpu")
    exact = exact_marginal_fixed_greedy_from_gram(gram, activation)
    refreshed = block_refresh_from_gram(gram, activation, 1, shortlist_size=64)
    np.testing.assert_array_equal(action_ids(exact, 31), action_ids(refreshed, 31))
    _assert_nested_complete(block_refresh_from_gram(gram, activation, 8, shortlist_size=16), 31)


def test_low_rank_exact_when_rank_spans_features_and_encodings_complete() -> None:
    rng = np.random.default_rng(23)
    deltas = rng.normal(size=(2, 19, 7)).astype(np.float32)
    activation = rng.normal(size=19).astype(np.float32)
    gram = build_base_gram(deltas, device="cpu")
    features = deltas.reshape(38, 7)
    factor = low_rank_factor(features, 7, "truncated_svd")
    exact = exact_marginal_fixed_greedy_from_gram(gram, activation)
    approximate = low_rank_marginal_fixed_greedy(factor, np.diag(gram), activation)
    np.testing.assert_array_equal(action_ids(exact, 19), action_ids(approximate, 19))
    for encoding in ("fp16", "int8"):
        _assert_nested_complete(
            low_rank_marginal_fixed_greedy(encode_factor(factor, encoding), np.diag(gram), activation), 19
        )


def test_page_masks_charge_pages_and_respect_nested_eligibility() -> None:
    rng = np.random.default_rng(29)
    deltas = rng.normal(size=(2, 16, 6)).astype(np.float32)
    activation = rng.normal(size=16).astype(np.float32)
    gram = build_base_gram(deltas, device="cpu")
    pages = separate_plane_pages(np.arange(16), per_page=8)
    selected = page_aware_fixed_greedy(gram, activation, pages, page_budget=4)
    assert len(selected.pages) <= 4
    depth = np.zeros(16, np.uint8)
    for action in selected.actions:
        assert action.stage == depth[action.coordinate] + 1
        depth[action.coordinate] += 1


def test_paid_page_reuses_actions_that_become_nested_eligible() -> None:
    gram = np.eye(4, dtype=np.float32)
    activation = np.array([2.0, 1.0], dtype=np.float32)
    # Page 1 is paid for stage two of coordinate zero before stage one of
    # coordinate one is selected. Its second action must subsequently be
    # applied without charging page 1 again.
    pages = {0: (0,), 1: (2, 3), 2: (1,)}
    chosen = page_aware_fixed_greedy(gram, activation, pages, page_budget=3)
    assert chosen.pages == (0, 1, 2)
    assert [(action.coordinate, action.stage) for action in chosen.actions] == [
        (0, 1), (0, 2), (1, 1), (1, 2),
    ]


def test_incomplete_physical_subset_is_not_forced_to_complete_recovery() -> None:
    gram = np.eye(4, dtype=np.float32)
    activation = np.array([2.0, 1.0], dtype=np.float32)
    order = exact_marginal_fixed_greedy_from_gram(gram, activation)
    curve = recovery_curve_from_gram(order[:1], gram, activation)
    np.testing.assert_allclose(curve, [0.4], rtol=0, atol=1e-7)


def test_old_serialized_exact_label_is_readable() -> None:
    assert canonical_selector_label("exact_marginal_omp") == "exact_marginal_fixed_greedy"
