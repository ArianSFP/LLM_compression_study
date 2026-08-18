from __future__ import annotations

import numpy as np

from oracle_study.mxfp4_selective import (
    backward_elimination_order,
    coordinate_to_page,
    diagonal_nested_order,
    exact_marginal_order,
    pairwise_coselection_layout,
    select_under_page_budget,
)


def test_nested_order_never_promotes_before_parent() -> None:
    order = diagonal_nested_order((np.asarray([1.0, 4.0]), np.asarray([10.0, 2.0])))
    seen = set()
    for action in order:
        if action.stage == 2:
            assert action.coordinate in seen
        else:
            seen.add(action.coordinate)


def test_exact_orders_are_nested_and_complete() -> None:
    rng = np.random.default_rng(4)
    contributions = rng.normal(size=(2, 8, 5)).astype(np.float32)
    for order in (exact_marginal_order(contributions), backward_elimination_order(contributions)):
        assert len(order) == 16
        depth = np.zeros(8, np.uint8)
        for action in order:
            assert action.stage == depth[action.coordinate] + 1
            depth[action.coordinate] += 1


def test_exact_page_charge_matches_selected_ids() -> None:
    layout = np.arange(16, dtype=np.int64)
    order = diagonal_nested_order((np.arange(16, dtype=float)[::-1], np.ones(16)))
    selected, pages = select_under_page_budget(order, layout, 64, 512, 512)
    assert len(pages) == 1
    assert all(coordinate_to_page(layout, a.coordinate, a.stage, 64, 512) in pages for a in selected)
    assert all(a.coordinate < 8 for a in selected)


def test_pairwise_layout_groups_identical_masks() -> None:
    incidence = np.asarray([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]], np.uint8)
    layout = pairwise_coselection_layout(incidence, 2)
    groups = [set(layout[i:i + 2].tolist()) for i in range(0, 4, 2)]
    assert {0, 1} in groups and {2, 3} in groups
