"""Low-memory exact-self seeds for compact joint interaction solvers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .split_interaction_field import SPLIT_STATE_PAGE_COSTS, SplitInteractionField


__all__ = ["LagrangianSeedTrace", "lagrangian_self_seed"]


@dataclass(frozen=True)
class LagrangianSeedTrace:
    """Independent-price seed and its explicit bounded work."""

    states: np.ndarray
    pages: int
    page_price: float
    price_iterations: int
    state_evaluations: int
    greedy_evaluations: int


def lagrangian_self_seed(
    field: SplitInteractionField,
    budget_pages: int,
    *,
    price_iterations: int = 48,
) -> LagrangianSeedTrace:
    """Build a near-budget self-only seed without a budget-sized DP table.

    A scalar page price makes all unit choices independent. Bisection finds
    the smallest observed feasible price, then one deterministic marginal-gain
    pass spends useful slack. Runtime storage is O(units), not
    O(units*budget), and all reads are from the resident exact-self table.
    """

    budget = int(budget_pages)
    iterations = int(price_iterations)
    if budget < 0 or budget > 3 * field.units or iterations < 1:
        raise ValueError("invalid Lagrangian seed budget or iteration count")
    local = np.asarray(field.local_damage, np.float64)
    costs = np.asarray(SPLIT_STATE_PAGE_COSTS, np.int64)
    if local.shape != (field.units, 8):
        raise ValueError("field local table changed")
    evaluations = 0

    def choose(price: float) -> tuple[np.ndarray, int]:
        nonlocal evaluations
        scores = local + float(price) * costs[None, :]
        states = np.argmin(scores, axis=1).astype(np.int64)
        evaluations += int(scores.size)
        return states, int(costs[states].sum())

    zero, zero_pages = choose(0.0)
    if zero_pages <= budget:
        return LagrangianSeedTrace(zero, zero_pages, 0.0, 0, evaluations, 0)
    scale = max(float(np.max(local, initial=0.0)), 1.0)
    low, high = 0.0, scale
    feasible, pages = choose(high)
    expansions = 0
    while pages > budget:
        low, high = high, 2.0 * high
        feasible, pages = choose(high)
        expansions += 1
        if expansions > 64 or not np.isfinite(high):
            raise RuntimeError("could not bracket the self-only page price")
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        candidate, candidate_pages = choose(middle)
        if candidate_pages <= budget:
            high, feasible, pages = middle, candidate, candidate_pages
        else:
            low = middle

    slack = budget - pages
    moves: list[tuple[float, float, int, int, int]] = []
    greedy_evaluations = 0
    for unit, source in enumerate(feasible.tolist()):
        source_cost = int(costs[source])
        source_damage = float(local[unit, source])
        for destination in range(8):
            delta_pages = int(costs[destination]) - source_cost
            if delta_pages <= 0 or delta_pages > slack:
                continue
            gain = source_damage - float(local[unit, destination])
            greedy_evaluations += 1
            if gain > 0.0:
                moves.append(
                    (-gain / delta_pages, -gain, unit, destination, delta_pages)
                )
    moves.sort()
    used: set[int] = set()
    for _, _, unit, destination, delta_pages in moves:
        if unit in used or delta_pages > slack:
            continue
        feasible[unit] = destination
        pages += delta_pages
        slack -= delta_pages
        used.add(unit)
        if slack == 0:
            break
    return LagrangianSeedTrace(
        feasible.copy(), pages, float(high), iterations + expansions + 1,
        evaluations, greedy_evaluations,
    )
