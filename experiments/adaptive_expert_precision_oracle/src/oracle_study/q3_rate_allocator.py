"""Compact frontiers and pooled allocation for gate/up Q3 page layouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .average_rate_allocator import AllocationTrace, RateOption, multiple_choice_allocate
from .q3_gate_up_field import (
    Q3InteractionField, Q3PageLayout, Q3_IDEAL_COST_QUANTA,
    Q3_STATE_GATE_LEVEL, Q3_STATE_UP_LEVEL, plane_action_id,
    q3_additive_state_costs, q3_coordinate_descent, q3_local_search,
)


__all__ = [
    "Q3DiagonalDPTable", "Q3RateFrontierTrace", "Q3AllocationAwareTrace",
    "build_q3_diagonal_dp_table", "q3_diagonal_dp_seed", "q3_rate_frontier",
    "q3_allocation_aware_rate_frontiers", "training_plane_incidence",
]


@dataclass(frozen=True)
class Q3DiagonalDPTable:
    """Reusable exact-self table for one additive Q3 layout."""

    choices: np.ndarray
    final_damage: np.ndarray
    state_costs: np.ndarray

    @property
    def units(self) -> int:
        return int(self.choices.shape[0])

    @property
    def maximum_quanta(self) -> int:
        return int(self.choices.shape[1] - 1)

    def seed(self, budget_quanta: int, *, page_price: float = 0.0) -> np.ndarray:
        budget = int(budget_quanta)
        price = float(page_price)
        if budget < 0 or budget > self.maximum_quanta:
            raise ValueError("Q3 quantum budget lies outside the DP table")
        if not np.isfinite(price) or price < 0.0:
            raise ValueError("page_price must be finite and nonnegative")
        objective = self.final_damage[:budget + 1] + price * np.arange(budget + 1)
        quanta = int(np.argmin(objective))
        if not np.isfinite(objective[quanta]):
            raise RuntimeError("Q3 diagonal allocation has no feasible state")
        states = np.empty(self.units, np.int64)
        for unit in range(self.units - 1, -1, -1):
            state = int(self.choices[unit, quanta])
            if state < 0:
                raise RuntimeError("Q3 diagonal allocation backtrack failed")
            states[unit] = state
            quanta -= int(self.state_costs[state])
        return states


def _allowed_states(values: Sequence[int]) -> tuple[int, ...]:
    states = tuple(sorted({int(value) for value in values}))
    if not states or states[0] < 0 or states[-1] >= 18:
        raise ValueError("allowed Q3 states must be a nonempty subset of [0,17]")
    return states


def build_q3_diagonal_dp_table(
    field: Q3InteractionField, layout: Q3PageLayout, maximum_quanta: int,
    allowed_states: Sequence[int] = tuple(range(18)),
) -> Q3DiagonalDPTable:
    """Solve exact per-unit self damage once for every additive-layout budget."""
    costs = q3_additive_state_costs(layout)
    if costs is None:
        raise ValueError("Q3 diagonal DP requires an additive page layout")
    maximum = int(maximum_quanta)
    if maximum < 0 or maximum > int(np.max(costs)) * field.units:
        raise ValueError("Q3 quantum budget lies outside the additive layout")
    allowed = _allowed_states(allowed_states)
    dynamic = np.full(maximum + 1, np.inf, np.float64)
    dynamic[0] = 0.0
    choices = np.full((field.units, maximum + 1), -1, np.int8)
    local = np.asarray(field.local_damage, np.float64)
    for unit in range(field.units):
        updated = np.full(maximum + 1, np.inf, np.float64)
        unit_choice = choices[unit]
        for state in allowed:
            cost = int(costs[state])
            candidate = dynamic[:maximum + 1 - cost] + local[unit, state]
            current = updated[cost:]
            previous = unit_choice[cost:]
            better = candidate < current - 1e-15
            tied = np.isfinite(candidate) & np.isfinite(current)
            difference = np.zeros_like(candidate)
            np.subtract(candidate, current, out=difference, where=tied)
            tied &= np.abs(difference) <= 1e-15
            better |= tied & ((previous < 0) | (state < previous))
            current[better] = candidate[better]
            previous[better] = state
        dynamic = updated
    return Q3DiagonalDPTable(choices, dynamic, costs)


def q3_diagonal_dp_seed(
    field: Q3InteractionField, layout: Q3PageLayout, budget_quanta: int,
    allowed_states: Sequence[int] = tuple(range(18)), *, page_price: float = 0.0,
) -> np.ndarray:
    """Return one exact-self Q3 seed for an additive page layout."""
    return build_q3_diagonal_dp_table(
        field, layout, int(budget_quanta), allowed_states,
    ).seed(int(budget_quanta), page_price=float(page_price))


@dataclass(frozen=True)
class Q3RateFrontierTrace:
    options: tuple[RateOption, ...]
    coordinate_sweeps: int
    local_passes: int
    anchor_solves: int
    price_solves: int
    state_comparisons: int
    diagonal_dp_tables: int
    diagonal_dp_state_updates: int


@dataclass(frozen=True)
class Q3AllocationAwareTrace:
    frontiers: tuple[tuple[RateOption, ...], ...]
    allocation: AllocationTrace
    rounds: int
    selected_repairs: int
    adaptive_price_solves: int
    coordinate_sweeps: int
    local_passes: int


def _key(option: RateOption) -> tuple[float, int, str, tuple[int, ...]]:
    return (float(option.damage), int(option.pages), str(option.source),
            tuple(np.asarray(option.states, np.int64).tolist()))


def _pareto(options: Sequence[RateOption]) -> tuple[RateOption, ...]:
    by_cost: dict[int, RateOption] = {}
    for option in options:
        previous = by_cost.get(int(option.pages))
        if previous is None or _key(option) < _key(previous):
            by_cost[int(option.pages)] = option
    output = []
    best = np.inf
    for cost in sorted(by_cost):
        option = by_cost[cost]
        if float(option.damage) < best - 1e-12:
            output.append(option)
            best = float(option.damage)
    if not output or int(output[0].pages) != 0:
        raise RuntimeError("Q3 frontier lost its zero-cost endpoint")
    return tuple(output)


def q3_rate_frontier(
    field: Q3InteractionField, layout: Q3PageLayout, *, maximum_quanta: int,
    target_quanta: Sequence[int], price_ratios: Sequence[float],
    coordinate_sweeps: int, local_shortlist: int, local_swap_units: int,
    local_max_passes: int, price_local_max_passes: int = 0,
    allowed_states: Sequence[int] = tuple(range(18)),
) -> Q3RateFrontierTrace:
    """Generate price solutions plus exact-self-DP-seeded hard anchors."""
    maximum = int(maximum_quanta)
    if maximum < 0 or maximum > 6 * field.units:
        raise ValueError("maximum Q3 cost lies outside [0,6*units]")
    targets = tuple(sorted({0, maximum, *(int(value) for value in target_quanta)}))
    if targets[0] < 0 or targets[-1] > maximum:
        raise ValueError("target cost exceeds the frontier maximum")
    ratios = tuple(float(value) for value in price_ratios)
    if not ratios or tuple(sorted(ratios, reverse=True)) != ratios:
        raise ValueError("price ratios must be a nonempty descending sequence")
    if any(not np.isfinite(value) or value < 0 for value in ratios):
        raise ValueError("price ratios must be finite and nonnegative")

    allowed_states = _allowed_states(allowed_states)
    zero = np.zeros(field.units, np.int64)
    scale = max(field.damage(zero), 1e-30) / max(maximum, 1)
    options: list[RateOption] = [RateOption(
        0, field.damage(zero), zero.copy(), "zero_endpoint", np.inf, 0, 0,
    )]
    warm = zero.copy()
    sweeps = passes = 0
    for ratio in ratios:
        price = float(ratio) * scale
        coordinate = q3_coordinate_descent(
            field, layout, warm, maximum, max_sweeps=int(coordinate_sweeps),
            page_price=price, allowed_states=allowed_states,
        )
        repaired = q3_local_search(
            field, layout, coordinate.states, maximum,
            shortlist_size=int(local_shortlist), max_swap_units=int(local_swap_units),
            max_passes=int(price_local_max_passes), page_price=price,
            allowed_states=allowed_states,
        )
        warm = repaired.states.copy()
        sweeps += int(coordinate.sweeps)
        passes += int(repaired.passes)
        options.append(RateOption(
            int(repaired.cost_quanta), float(repaired.damage), repaired.states.copy(),
            "lagrangian_price_path", price, int(coordinate.sweeps), int(repaired.passes),
        ))

    coarse = _pareto(options)
    additive_costs = q3_additive_state_costs(layout)
    diagonal = None if additive_costs is None else build_q3_diagonal_dp_table(
        field, layout, maximum, allowed_states,
    )
    for target in targets:
        if diagonal is None:
            feasible = [item for item in coarse if int(item.pages) <= int(target)]
            seed = min(
                feasible,
                key=lambda item: (float(item.damage), -int(item.pages), _key(item)),
            )
            source = f"hard_anchor_{target}"
        else:
            states = diagonal.seed(int(target))
            seed = RateOption(
                int(layout.cost_quanta(states)), float(field.damage(states)),
                states.copy(), f"exact_self_dp_seed_{target}", 0.0, 0, 0,
            )
            options.append(seed)
            source = f"exact_self_dp_repaired_{target}"
        coordinate = q3_coordinate_descent(
            field, layout, seed.states, int(target), max_sweeps=int(coordinate_sweeps),
            allowed_states=allowed_states,
        )
        repaired = q3_local_search(
            field, layout, coordinate.states, int(target),
            shortlist_size=int(local_shortlist), max_swap_units=int(local_swap_units),
            max_passes=int(local_max_passes), allowed_states=allowed_states,
        )
        sweeps += int(coordinate.sweeps)
        passes += int(repaired.passes)
        options.append(RateOption(
            int(repaired.cost_quanta), float(repaired.damage), repaired.states.copy(),
            source, 0.0, int(coordinate.sweeps), int(repaired.passes),
        ))
    diagonal_updates = 0 if diagonal is None else field.units * sum(
        maximum + 1 - int(additive_costs[state]) for state in allowed_states
    )
    return Q3RateFrontierTrace(
        _pareto(options), sweeps, passes, len(targets), len(ratios),
        field.units * 18 * (len(ratios) + len(targets)),
        int(diagonal is not None), int(diagonal_updates),
    )


def _signature(frontiers: Sequence[Sequence[RateOption]], allocation: AllocationTrace):
    return tuple((int(frontiers[index][int(option)].pages),
                  tuple(frontiers[index][int(option)].states.tolist()))
                 for index, option in enumerate(allocation.option_indices))


def _marginal_price(frontier: Sequence[RateOption], selected: int) -> float | None:
    current = frontier[int(selected)]
    slopes = []
    if selected > 0:
        lower = frontier[selected - 1]
        width = int(current.pages) - int(lower.pages)
        if width > 0:
            slopes.append((float(lower.damage) - float(current.damage)) / width)
    if selected + 1 < len(frontier):
        upper = frontier[selected + 1]
        width = int(upper.pages) - int(current.pages)
        if width > 0:
            slopes.append((float(current.damage) - float(upper.damage)) / width)
    values = [value for value in slopes if np.isfinite(value) and value > 0]
    return None if not values else float(np.exp(np.mean(np.log(values))))


def q3_allocation_aware_rate_frontiers(
    fields: Sequence[Q3InteractionField], layouts: Sequence[Q3PageLayout],
    frontiers: Sequence[Sequence[RateOption]], *, group_budget_quanta: int,
    burst_cap_quanta: int, weights: Sequence[float] | np.ndarray,
    coordinate_sweeps: int, local_shortlist: int, local_swap_units: int,
    local_max_passes: int, max_rounds: int = 3,
    adaptive_price_multipliers: Sequence[float] = (.5, 2.0),
    adaptive_price_local_passes: int = 0,
    allowed_states: Sequence[int] = tuple(range(18)),
) -> Q3AllocationAwareTrace:
    """Repair only allocator-selected columns and sample nearby prices."""
    if not fields or len(fields) != len(layouts) or len(fields) != len(frontiers):
        raise ValueError("fields, layouts, and frontiers must align")
    burst = int(burst_cap_quanta)
    working = tuple(_pareto([item for item in frontier if int(item.pages) <= burst])
                    for frontier in frontiers)
    allocation = multiple_choice_allocate(working, int(group_budget_quanta), weights)
    rounds = repairs = price_solves = sweeps = passes = 0
    for round_index in range(int(max_rounds)):
        before = _signature(working, allocation)
        updated = []
        for expert, (field, layout) in enumerate(zip(fields, layouts)):
            selected = int(allocation.option_indices[expert])
            option = working[expert][selected]
            coordinate = q3_coordinate_descent(
                field, layout, option.states, int(option.pages),
                max_sweeps=int(coordinate_sweeps), allowed_states=allowed_states,
            )
            repaired = q3_local_search(
                field, layout, coordinate.states, int(option.pages),
                shortlist_size=int(local_shortlist), max_swap_units=int(local_swap_units),
                max_passes=int(local_max_passes), allowed_states=allowed_states,
            )
            additions = [RateOption(
                int(repaired.cost_quanta), float(repaired.damage), repaired.states.copy(),
                f"selected_hard_repair_round_{round_index + 1}", 0.0,
                int(coordinate.sweeps), int(repaired.passes),
            )]
            repairs += 1
            sweeps += int(coordinate.sweeps)
            passes += int(repaired.passes)
            center = _marginal_price(working[expert], selected)
            if center is not None:
                for multiplier in adaptive_price_multipliers:
                    price = center * float(multiplier)
                    priced = q3_coordinate_descent(
                        field, layout, repaired.states, burst,
                        max_sweeps=int(coordinate_sweeps), page_price=price,
                        allowed_states=allowed_states,
                    )
                    priced_repair = q3_local_search(
                        field, layout, priced.states, burst,
                        shortlist_size=int(local_shortlist), max_swap_units=int(local_swap_units),
                        max_passes=int(adaptive_price_local_passes), page_price=price,
                        allowed_states=allowed_states,
                    )
                    additions.append(RateOption(
                        int(priced_repair.cost_quanta), float(priced_repair.damage),
                        priced_repair.states.copy(), f"adaptive_price_round_{round_index + 1}",
                        price, int(priced.sweeps), int(priced_repair.passes),
                    ))
                    price_solves += 1
                    sweeps += int(priced.sweeps)
                    passes += int(priced_repair.passes)
            updated.append(_pareto((*working[expert], *additions)))
        working = tuple(updated)
        allocation = multiple_choice_allocate(working, int(group_budget_quanta), weights)
        rounds = round_index + 1
        if _signature(working, allocation) == before:
            break
    return Q3AllocationAwareTrace(working, allocation, rounds, repairs, price_solves, sweeps, passes)


def training_plane_incidence(
    fields: Sequence[Q3InteractionField], *, price_ratios: Sequence[float],
) -> np.ndarray:
    """Generate cheap training-only action masks from independent exact-self states."""
    rows = []
    for field in fields:
        scale = max(float(np.sum(field.local_damage[:, 0])), 1e-30) / (6 * field.units)
        for ratio in price_ratios:
            price = float(ratio) * scale
            scores = np.asarray(field.local_damage) + price * Q3_IDEAL_COST_QUANTA[None, :]
            states = np.argmin(scores, axis=1)
            row = np.zeros(4 * field.units, np.uint8)
            for unit, state in enumerate(states.tolist()):
                for stage in range(1, int(Q3_STATE_GATE_LEVEL[state]) + 1):
                    row[plane_action_id("gate", stage, unit, field.units)] = 1
                for stage in range(1, int(Q3_STATE_UP_LEVEL[state]) + 1):
                    row[plane_action_id("up", stage, unit, field.units)] = 1
            rows.append(row)
    if not rows:
        raise ValueError("training incidence needs at least one field")
    return np.stack(rows)
