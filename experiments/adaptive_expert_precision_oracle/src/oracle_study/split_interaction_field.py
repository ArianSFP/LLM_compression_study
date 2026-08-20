"""Eight-state split gate/up/down low-rank interaction fields."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np

from .interaction_field import JointInteractionFactor
from .neuron_selector import UnitScoreMetadata
from .set_utility_oracles import UnitStateMove
from .unit_set_teacher import DownMetricGram


SPLIT_STATE_PAGE_COSTS = np.asarray([0, 1, 1, 2, 1, 2, 2, 3], np.int64)
SPLIT_STATE_GATE_HIGH = np.asarray(
    [False, False, False, False, True, True, True, True],
)
SPLIT_STATE_UP_HIGH = np.asarray(
    [False, False, True, True, False, False, True, True],
)
SPLIT_STATE_DOWN_HIGH = np.asarray(
    [False, True, False, True, False, True, False, True],
)
SPLIT_STATE_HIDDEN_INDEX = (
    2 * SPLIT_STATE_GATE_HIGH.astype(np.int64)
    + SPLIT_STATE_UP_HIGH.astype(np.int64)
)
# Four-state order is 00, G(=gate+up), D, GD.
INHERITED_FOUR_STATE_MAP = np.asarray([0, 6, 1, 7], np.int64)
SPLIT_ONLY_STATES = frozenset((2, 3, 4, 5))


__all__ = [
    "SPLIT_STATE_PAGE_COSTS",
    "SPLIT_STATE_GATE_HIGH",
    "SPLIT_STATE_UP_HIGH",
    "SPLIT_STATE_DOWN_HIGH",
    "SPLIT_STATE_HIDDEN_INDEX",
    "INHERITED_FOUR_STATE_MAP",
    "SPLIT_ONLY_STATES",
    "SplitProjectionResponses",
    "SplitInteractionField",
    "SplitCoordinateTrace",
    "SplitLocalSearchTrace",
    "SplitGramField",
    "SplitGramCoordinateTrace",
    "SplitGramLocalSearchTrace",
    "split_projection_responses",
    "split_state_output",
    "split_exact_damage",
    "build_split_interaction_field",
    "map_four_state_to_split",
    "split_diagonal_dp_seed",
    "split_coordinate_descent",
    "split_local_search",
    "build_split_gram_field",
    "split_gram_coordinate_descent",
    "split_gram_local_search",
]


def _silu(value: np.ndarray) -> np.ndarray:
    value64 = np.asarray(value, np.float64)
    result = np.empty_like(value64)
    positive = value64 >= 0.0
    result[positive] = value64[positive] / (1.0 + np.exp(-value64[positive]))
    exponential = np.exp(value64[~positive])
    result[~positive] = value64[~positive] * exponential / (1.0 + exponential)
    return result


def _validate_states(states: Sequence[int] | np.ndarray, units: int) -> np.ndarray:
    result = np.asarray(states, np.int64).reshape(-1)
    if result.shape != (int(units),) or np.any((result < 0) | (result > 7)):
        raise ValueError("split states must contain one value in [0,7] per unit")
    return result.copy()


@dataclass(frozen=True)
class SplitProjectionResponses:
    """Four hidden responses h22,h24,h42,h44 and two down matrices."""

    hidden: np.ndarray
    down2: np.ndarray
    down4: np.ndarray

    def __post_init__(self) -> None:
        hidden = np.asarray(self.hidden)
        down2 = np.asarray(self.down2)
        down4 = np.asarray(self.down4)
        if hidden.ndim != 2 or hidden.shape[1] != 4:
            raise ValueError("hidden responses must have shape [unit,4]")
        if down2.ndim != 2 or down4.shape != down2.shape or down2.shape[0] != hidden.shape[0]:
            raise ValueError("down matrices must have equal [unit,output] shapes")
        if not all(np.all(np.isfinite(value)) for value in (hidden, down2, down4)):
            raise ValueError("split projection responses must be finite")

    @property
    def units(self) -> int:
        return int(np.asarray(self.hidden).shape[0])

    @property
    def target_hidden(self) -> np.ndarray:
        return np.asarray(self.hidden)[:, 3]

    @property
    def target_output(self) -> np.ndarray:
        return self.target_hidden @ np.asarray(self.down4)


def split_projection_responses(
    q2: Sequence[np.ndarray], q4: Sequence[np.ndarray], activation: np.ndarray,
) -> SplitProjectionResponses:
    """Evaluate h22,h24,h42,h44 with separate gate/up refinements."""
    if len(q2) != 3 or len(q4) != 3:
        raise ValueError("q2/q4 must contain gate, up, and down matrices")
    x = np.asarray(activation, np.float64).reshape(-1)
    gate2, up2, down2 = (np.asarray(value, np.float64) for value in q2)
    gate4, up4, down4 = (np.asarray(value, np.float64) for value in q4)
    if gate2.shape != gate4.shape or up2.shape != up4.shape or down2.shape != down4.shape:
        raise ValueError("Q2/Q4 projection shapes changed")
    if gate2.shape[1] != x.size or up2.shape[1] != x.size:
        raise ValueError("activation width disagrees with gate/up matrices")
    g2, g4 = gate2 @ x, gate4 @ x
    u2, u4 = up2 @ x, up4 @ x
    hidden = np.stack(
        (_silu(g2) * u2, _silu(g2) * u4, _silu(g4) * u2, _silu(g4) * u4),
        axis=1,
    )
    return SplitProjectionResponses(hidden, down2.T, down4.T)


def map_four_state_to_split(states: Sequence[int] | np.ndarray) -> np.ndarray:
    value = np.asarray(states, np.int64).reshape(-1)
    if np.any((value < 0) | (value > 3)):
        raise ValueError("four-state values must lie in [0,3]")
    return INHERITED_FOUR_STATE_MAP[value]


def split_state_output(
    responses: SplitProjectionResponses, states: Sequence[int] | np.ndarray,
) -> np.ndarray:
    state = _validate_states(states, responses.units)
    rows = np.arange(responses.units)
    hidden = np.asarray(responses.hidden, np.float64)[
        rows, SPLIT_STATE_HIDDEN_INDEX[state]
    ]
    low = ~SPLIT_STATE_DOWN_HIGH[state]
    output = np.zeros(np.asarray(responses.down2).shape[1], np.float64)
    if np.any(low):
        output += hidden[low] @ np.asarray(responses.down2, np.float64)[low]
    if np.any(~low):
        output += hidden[~low] @ np.asarray(responses.down4, np.float64)[~low]
    return output


def split_exact_damage(
    responses: SplitProjectionResponses,
    states: Sequence[int] | np.ndarray,
    *,
    proxy: np.ndarray | None = None,
    beta: float = 0.0,
) -> float:
    residual = responses.target_output - split_state_output(responses, states)
    damage = float(residual @ residual)
    if proxy is not None and float(beta) != 0.0:
        projected = residual @ np.asarray(proxy, np.float64)
        damage += float(beta) * float(projected @ projected)
    return damage


@dataclass(frozen=True)
class SplitInteractionField:
    """Exact-self, approximate-cross objective for eight split states."""

    rho: np.ndarray
    coefficients: np.ndarray
    local_damage: np.ndarray
    self_residual: np.ndarray
    factor: JointInteractionFactor
    clipped_local_values: int = 0

    def __post_init__(self) -> None:
        rho = np.asarray(self.rho)
        coefficients = np.asarray(self.coefficients)
        local = np.asarray(self.local_damage)
        residual = np.asarray(self.self_residual)
        if rho.ndim != 3 or rho.shape[1] != 8:
            raise ValueError("rho must have shape [unit,8,rank]")
        if coefficients.shape != (rho.shape[0], 8, 2):
            raise ValueError("coefficients must have shape [unit,8,2]")
        if local.shape != rho.shape[:2] or residual.shape != local.shape:
            raise ValueError("local/self tables must have shape [unit,8]")
        if rho.shape[0] != self.factor.units or rho.shape[2] != self.factor.rank:
            raise ValueError("field and factor shapes disagree")
        if not all(np.all(np.isfinite(value)) for value in (rho, coefficients, local, residual)):
            raise ValueError("split interaction field contains non-finite values")

    @property
    def units(self) -> int:
        return int(np.asarray(self.rho).shape[0])

    @property
    def rank(self) -> int:
        return int(np.asarray(self.rho).shape[2])

    def pages(self, states: Sequence[int] | np.ndarray) -> int:
        state = _validate_states(states, self.units)
        return int(SPLIT_STATE_PAGE_COSTS[state].sum())

    def damage(self, states: Sequence[int] | np.ndarray) -> float:
        state = _validate_states(states, self.units)
        rows = np.arange(self.units)
        signature = np.asarray(self.rho)[rows, state].sum(axis=0)
        return float(signature @ signature + np.asarray(self.self_residual)[rows, state].sum())


def build_split_interaction_field(
    factor: JointInteractionFactor,
    hidden: np.ndarray,
    metadata: UnitScoreMetadata,
    *,
    negative_tolerance: float = 1e-7,
) -> SplitInteractionField:
    """Build the eight-state field with unchanged L4/L2 and A/B/C."""
    values = np.asarray(hidden, np.float64)
    if values.shape != (factor.units, 4) or metadata.units != factor.units:
        raise ValueError("hidden responses, A/B/C, and factor units disagree")
    target = values[:, 3]
    a = np.asarray(metadata.a, np.float64)
    b = np.asarray(metadata.b, np.float64)
    c = np.asarray(metadata.c, np.float64)
    l4 = np.asarray(factor.l4, np.float64)
    l2 = np.asarray(factor.l2, np.float64)
    rho = np.empty((factor.units, 8, factor.rank), np.float64)
    coefficients = np.empty((factor.units, 8, 2), np.float64)
    damage = np.empty((factor.units, 8), np.float64)
    for state in range(8):
        h = values[:, int(SPLIT_STATE_HIDDEN_INDEX[state])]
        if bool(SPLIT_STATE_DOWN_HIGH[state]):
            first, second = target - h, np.zeros_like(target)
            rho[:, state] = first[:, None] * l4
            damage[:, state] = first * first * a
        else:
            first, second = target, -h
            rho[:, state] = first[:, None] * l4 + second[:, None] * l2
            damage[:, state] = (
                target * target * a - 2.0 * target * h * b + h * h * c
            )
        coefficients[:, state, 0] = first
        coefficients[:, state, 1] = second
    scale = max(float(np.max(np.abs(damage), initial=0.0)), 1.0)
    if float(np.min(damage, initial=0.0)) < -float(negative_tolerance) * scale:
        raise ValueError("A/B/C produced materially negative split-state damage")
    clipped = int(np.count_nonzero(damage < 0.0))
    damage = np.maximum(damage, 0.0)
    norms = np.einsum("usr,usr->us", rho, rho, optimize=True)
    return SplitInteractionField(
        rho, coefficients, damage, damage - norms, factor, clipped,
    )


def split_diagonal_dp_seed(
    field: SplitInteractionField,
    budget_pages: int,
    allowed_states: Sequence[int] = tuple(range(8)),
) -> np.ndarray:
    """Solve the exact-self multiple-choice allocation."""
    budget = int(budget_pages)
    if budget < 0 or budget > 3 * field.units:
        raise ValueError("page budget lies outside [0,3*units]")
    dynamic = np.full((field.units + 1, budget + 1), np.inf, np.float64)
    choice = np.full((field.units, budget + 1), -1, np.int8)
    previous = np.full((field.units, budget + 1), -1, np.int32)
    dynamic[0, 0] = 0.0
    local = np.asarray(field.local_damage)
    allowed = tuple(sorted({int(state) for state in allowed_states}))
    if not allowed or any(state < 0 or state > 7 for state in allowed):
        raise ValueError("allowed states must be a nonempty subset of [0,7]")
    for unit in range(field.units):
        for used in np.flatnonzero(np.isfinite(dynamic[unit])).tolist():
            for state in allowed:
                cost = int(SPLIT_STATE_PAGE_COSTS[state])
                target = used + int(cost)
                if target > budget:
                    continue
                candidate = float(dynamic[unit, used] + local[unit, state])
                current = float(dynamic[unit + 1, target])
                if candidate < current - 1e-15 or (
                    abs(candidate - current) <= 1e-15
                    and (choice[unit, target] < 0 or state < int(choice[unit, target]))
                ):
                    dynamic[unit + 1, target] = candidate
                    choice[unit, target] = state
                    previous[unit, target] = used
    final_pages = min(
        range(budget + 1),
        key=lambda pages: (float(dynamic[field.units, pages]), pages),
    )
    if not np.isfinite(dynamic[field.units, final_pages]):
        raise RuntimeError("diagonal allocation has no feasible state")
    states = np.empty(field.units, np.int64)
    pages = int(final_pages)
    for unit in range(field.units - 1, -1, -1):
        state = int(choice[unit, pages])
        if state < 0:
            raise RuntimeError("diagonal allocation backtrack failed")
        states[unit] = state
        pages = int(previous[unit, pages])
    return states


@dataclass(frozen=True)
class SplitCoordinateTrace:
    states: np.ndarray
    pages: int
    damage: float
    sweeps: int
    accepted_moves: int
    interaction_dot_macs: int


def split_coordinate_descent(
    field: SplitInteractionField,
    seed: Sequence[int] | np.ndarray,
    budget_pages: int,
    *,
    max_sweeps: int = 8,
    tolerance: float = 1e-12,
) -> SplitCoordinateTrace:
    """Eight-state coordinate descent using two rank-r dots per unit."""
    budget = int(budget_pages)
    states = _validate_states(seed, field.units)
    pages = field.pages(states)
    if budget < 0 or budget > 3 * field.units or pages > budget:
        raise ValueError("seed exceeds the page budget")
    rho = np.asarray(field.rho)
    coefficients = np.asarray(field.coefficients)
    local = np.asarray(field.local_damage)
    l4 = np.asarray(field.factor.l4, np.float64)
    l2 = np.asarray(field.factor.l2, np.float64)
    rows = np.arange(field.units)
    residual = rho[rows, states].sum(axis=0)
    damage = field.damage(states)
    accepted = 0
    completed = 0
    for sweep in range(int(max_sweeps)):
        changed = False
        for unit in range(field.units):
            source = int(states[unit])
            residual_without = residual - rho[unit, source]
            base_pages = pages - int(SPLIT_STATE_PAGE_COSTS[source])
            allowed = base_pages + SPLIT_STATE_PAGE_COSTS <= budget
            dots = np.asarray((
                residual_without @ l4[unit],
                residual_without @ l2[unit],
            ))
            scores = 2.0 * (coefficients[unit] @ dots) + local[unit]
            destination = int(np.argmin(np.where(allowed, scores, np.inf)))
            if destination == source:
                continue
            proposal = states.copy()
            proposal[unit] = destination
            candidate_damage = field.damage(proposal)
            if candidate_damage < damage - float(tolerance):
                states = proposal
                pages = base_pages + int(SPLIT_STATE_PAGE_COSTS[destination])
                residual = residual_without + rho[unit, destination]
                damage = candidate_damage
                accepted += 1
                changed = True
        completed = sweep + 1
        if not changed:
            break
    return SplitCoordinateTrace(
        states.copy(), pages, float(damage), completed, accepted,
        2 * field.units * field.rank * completed,
    )


@dataclass(frozen=True)
class SplitLocalSearchTrace:
    states: np.ndarray
    pages: int
    damage: float
    passes: int
    accepted_bundles: int
    candidate_moves_per_pass: int
    maximum_shortlist: int


def split_local_search(
    field: SplitInteractionField,
    seed: Sequence[int] | np.ndarray,
    budget_pages: int,
    *,
    shortlist_size: int = 8,
    max_swap_units: int = 3,
    max_passes: int = 12,
    tolerance: float = 1e-12,
) -> SplitLocalSearchTrace:
    """Bounded 1/2/3-unit repair over seven alternatives per unit."""
    states = _validate_states(seed, field.units)
    budget = int(budget_pages)
    pages = field.pages(states)
    if pages > budget:
        raise ValueError("seed exceeds page budget")
    rho = np.asarray(field.rho)
    self_term = np.asarray(field.self_residual)
    rows = np.arange(field.units)
    residual = rho[rows, states].sum(axis=0)
    damage = field.damage(states)
    accepted, completed = 0, 0
    candidate_count = 7 * field.units
    maximum_shortlist = 7 * int(shortlist_size)
    for pass_index in range(int(max_passes)):
        moves: list[tuple[UnitStateMove, np.ndarray, float]] = []
        for unit, source in enumerate(states.tolist()):
            for destination in range(8):
                if destination == source:
                    continue
                page_delta = int(
                    SPLIT_STATE_PAGE_COSTS[destination]
                    - SPLIT_STATE_PAGE_COSTS[source]
                )
                delta = rho[unit, destination] - rho[unit, source]
                change = (
                    2.0 * float(residual @ delta) + float(delta @ delta)
                    + float(self_term[unit, destination] - self_term[unit, source])
                )
                moves.append((
                    UnitStateMove(unit, source, destination, page_delta),
                    delta,
                    -change,
                ))
        shortlisted: list[tuple[UnitStateMove, np.ndarray, float]] = []
        for delta_pages in range(-3, 4):
            group = [move for move in moves if move[0].page_delta == delta_pages]
            group.sort(key=lambda item: (-item[2], item[0].unit, item[0].to_state))
            shortlisted.extend(group[: int(shortlist_size)])
        if not shortlisted:
            break
        deltas = np.stack([item[1] for item in shortlisted])
        pairwise = deltas @ deltas.T
        best_gain = float("-inf")
        best_moves: tuple[int, ...] | None = None
        best_key: tuple[UnitStateMove, ...] | None = None
        for size in range(1, int(max_swap_units) + 1):
            for chosen in combinations(range(len(shortlisted)), size):
                state_moves = tuple(sorted(shortlisted[index][0] for index in chosen))
                if len({move.unit for move in state_moves}) != size:
                    continue
                next_pages = pages + sum(move.page_delta for move in state_moves)
                if next_pages < 0 or next_pages > budget:
                    continue
                gain = sum(shortlisted[index][2] for index in chosen)
                gain -= 2.0 * sum(
                    pairwise[left, right]
                    for left, right in combinations(chosen, 2)
                )
                if gain > best_gain + tolerance or (
                    abs(gain - best_gain) <= tolerance
                    and (best_key is None or state_moves < best_key)
                ):
                    best_gain, best_moves, best_key = float(gain), chosen, state_moves
        if best_moves is None or best_gain <= tolerance:
            break
        proposal = states.copy()
        for index in best_moves:
            move = shortlisted[index][0]
            proposal[move.unit] = move.to_state
        proposal_damage = field.damage(proposal)
        if proposal_damage >= damage - tolerance:
            break
        states = proposal
        pages = field.pages(states)
        residual = rho[rows, states].sum(axis=0)
        damage = proposal_damage
        accepted += 1
        completed = pass_index + 1
    return SplitLocalSearchTrace(
        states.copy(), pages, float(damage), completed, accepted,
        candidate_count, maximum_shortlist,
    )



@dataclass(frozen=True)
class SplitGramField:
    """Exact eight-state objective backed by the three static Gram blocks."""

    coefficients: np.ndarray
    local_damage: np.ndarray
    gram: DownMetricGram

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients)
        local = np.asarray(self.local_damage)
        if coefficients.shape != (self.gram.units, 8, 2):
            raise ValueError("Gram coefficients must have shape [unit,8,2]")
        if local.shape != (self.gram.units, 8):
            raise ValueError("Gram local damage must have shape [unit,8]")
        if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(local)):
            raise ValueError("Gram field contains non-finite values")

    @property
    def units(self) -> int:
        return int(self.gram.units)

    def pages(self, states: Sequence[int] | np.ndarray) -> int:
        state = _validate_states(states, self.units)
        return int(SPLIT_STATE_PAGE_COSTS[state].sum())

    def damage(self, states: Sequence[int] | np.ndarray) -> float:
        state = _validate_states(states, self.units)
        rows = np.arange(self.units)
        selected = np.asarray(self.coefficients)[rows, state]
        high, low = selected[:, 0], selected[:, 1]
        q44 = np.asarray(self.gram.q44, np.float64)
        q42 = np.asarray(self.gram.q42, np.float64)
        q22 = np.asarray(self.gram.q22, np.float64)
        return float(
            high @ q44 @ high
            + 2.0 * high @ q42 @ low
            + low @ q22 @ low
        )


def build_split_gram_field(
    gram: DownMetricGram,
    hidden: np.ndarray,
    metadata: UnitScoreMetadata,
) -> SplitGramField:
    """Build an exact eight-state field without any new Gram block."""
    values = np.asarray(hidden, np.float64)
    if values.shape != (gram.units, 4) or metadata.units != gram.units:
        raise ValueError("hidden responses, A/B/C, and Gram units disagree")
    target = values[:, 3]
    coefficients = np.empty((gram.units, 8, 2), np.float64)
    local = np.empty((gram.units, 8), np.float64)
    a = np.asarray(metadata.a, np.float64)
    b = np.asarray(metadata.b, np.float64)
    c = np.asarray(metadata.c, np.float64)
    for state in range(8):
        h = values[:, int(SPLIT_STATE_HIDDEN_INDEX[state])]
        if bool(SPLIT_STATE_DOWN_HIGH[state]):
            coefficients[:, state, 0] = target - h
            coefficients[:, state, 1] = 0.0
            local[:, state] = (target - h) ** 2 * a
        else:
            coefficients[:, state, 0] = target
            coefficients[:, state, 1] = -h
            local[:, state] = (
                target * target * a - 2.0 * target * h * b + h * h * c
            )
    return SplitGramField(coefficients, np.maximum(local, 0.0), gram)


def _allowed_states(states: Sequence[int]) -> tuple[int, ...]:
    allowed = tuple(sorted({int(state) for state in states}))
    if not allowed or any(state < 0 or state > 7 for state in allowed):
        raise ValueError("allowed states must be a nonempty subset of [0,7]")
    return allowed


def _gram_state(
    field: SplitGramField, states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = np.arange(field.units)
    selected = np.asarray(field.coefficients)[rows, states]
    high, low = selected[:, 0].copy(), selected[:, 1].copy()
    q44 = np.asarray(field.gram.q44, np.float64)
    q42 = np.asarray(field.gram.q42, np.float64)
    q22 = np.asarray(field.gram.q22, np.float64)
    correlation4 = q44 @ high + q42 @ low
    correlation2 = q42.T @ high + q22 @ low
    return high, low, correlation4, correlation2


@dataclass(frozen=True)
class SplitGramCoordinateTrace:
    states: np.ndarray
    pages: int
    damage: float
    sweeps: int
    accepted_moves: int


def split_gram_coordinate_descent(
    field: SplitGramField,
    seed: Sequence[int] | np.ndarray,
    budget_pages: int,
    *,
    allowed_states: Sequence[int] = tuple(range(8)),
    max_sweeps: int = 8,
    tolerance: float = 1e-12,
) -> SplitGramCoordinateTrace:
    """Exact coordinate descent with O(1) state scores after Gram updates."""
    allowed = _allowed_states(allowed_states)
    budget = int(budget_pages)
    states = _validate_states(seed, field.units)
    if any(int(state) not in allowed for state in states):
        raise ValueError("seed contains a forbidden state")
    pages = field.pages(states)
    if pages > budget:
        raise ValueError("seed exceeds page budget")
    high, low, correlation4, correlation2 = _gram_state(field, states)
    q44 = np.asarray(field.gram.q44, np.float64)
    q42 = np.asarray(field.gram.q42, np.float64)
    q22 = np.asarray(field.gram.q22, np.float64)
    coefficients = np.asarray(field.coefficients)
    local = np.asarray(field.local_damage)
    damage = field.damage(states)
    accepted, completed = 0, 0
    for sweep in range(int(max_sweeps)):
        changed = False
        for unit in range(field.units):
            source = int(states[unit])
            base_pages = pages - int(SPLIT_STATE_PAGE_COSTS[source])
            source_high, source_low = high[unit], low[unit]
            without4 = (
                correlation4[unit]
                - q44[unit, unit] * source_high
                - q42[unit, unit] * source_low
            )
            without2 = (
                correlation2[unit]
                - q42[unit, unit] * source_high
                - q22[unit, unit] * source_low
            )
            best = source
            best_score = (
                2.0 * (source_high * without4 + source_low * without2)
                + local[unit, source]
            )
            for destination in allowed:
                if base_pages + int(SPLIT_STATE_PAGE_COSTS[destination]) > budget:
                    continue
                candidate = coefficients[unit, destination]
                score = (
                    2.0 * (candidate[0] * without4 + candidate[1] * without2)
                    + local[unit, destination]
                )
                if score < best_score - tolerance or (
                    abs(score - best_score) <= tolerance and destination < best
                ):
                    best, best_score = destination, float(score)
            if best == source:
                continue
            destination = coefficients[unit, best]
            delta_high = float(destination[0] - source_high)
            delta_low = float(destination[1] - source_low)
            proposal_damage = (
                damage
                + 2.0 * delta_high * correlation4[unit]
                + 2.0 * delta_low * correlation2[unit]
                + delta_high * delta_high * q44[unit, unit]
                + 2.0 * delta_high * delta_low * q42[unit, unit]
                + delta_low * delta_low * q22[unit, unit]
            )
            if proposal_damage < damage - tolerance:
                states[unit] = best
                pages = base_pages + int(SPLIT_STATE_PAGE_COSTS[best])
                high[unit], low[unit] = destination
                correlation4 += q44[:, unit] * delta_high + q42[:, unit] * delta_low
                correlation2 += q42[unit, :] * delta_high + q22[:, unit] * delta_low
                damage = float(proposal_damage)
                accepted += 1
                changed = True
        completed = sweep + 1
        if not changed:
            break
    exact_damage = field.damage(states)
    if not np.isclose(damage, exact_damage, rtol=1e-9, atol=1e-8):
        raise RuntimeError("incremental Gram coordinate damage lost parity")
    return SplitGramCoordinateTrace(
        states.copy(), pages, exact_damage, completed, accepted,
    )


@dataclass(frozen=True)
class SplitGramLocalSearchTrace:
    states: np.ndarray
    pages: int
    damage: float
    passes: int
    accepted_bundles: int


def split_gram_local_search(
    field: SplitGramField,
    seed: Sequence[int] | np.ndarray,
    budget_pages: int,
    *,
    allowed_states: Sequence[int] = tuple(range(8)),
    shortlist_size: int = 8,
    max_swap_units: int = 3,
    max_passes: int = 12,
    tolerance: float = 1e-12,
) -> SplitGramLocalSearchTrace:
    """Exact bounded repair using Gram entries instead of output vectors."""
    allowed = _allowed_states(allowed_states)
    states = _validate_states(seed, field.units)
    if any(int(state) not in allowed for state in states):
        raise ValueError("seed contains a forbidden state")
    budget = int(budget_pages)
    pages = field.pages(states)
    if pages > budget:
        raise ValueError("seed exceeds page budget")
    high, low, correlation4, correlation2 = _gram_state(field, states)
    q44 = np.asarray(field.gram.q44, np.float64)
    q42 = np.asarray(field.gram.q42, np.float64)
    q22 = np.asarray(field.gram.q22, np.float64)
    coefficients = np.asarray(field.coefficients)
    damage = field.damage(states)
    accepted, completed = 0, 0
    for pass_index in range(int(max_passes)):
        moves = []
        for unit, source in enumerate(states.tolist()):
            for destination in allowed:
                if destination == source:
                    continue
                delta = coefficients[unit, destination] - coefficients[unit, source]
                dh, dl = float(delta[0]), float(delta[1])
                change = (
                    2.0 * dh * correlation4[unit]
                    + 2.0 * dl * correlation2[unit]
                    + dh * dh * q44[unit, unit]
                    + 2.0 * dh * dl * q42[unit, unit]
                    + dl * dl * q22[unit, unit]
                )
                move = UnitStateMove(
                    unit, source, destination,
                    int(SPLIT_STATE_PAGE_COSTS[destination] - SPLIT_STATE_PAGE_COSTS[source]),
                )
                moves.append((move, dh, dl, -float(change)))
        shortlisted = []
        for delta_pages in range(-3, 4):
            group = [move for move in moves if move[0].page_delta == delta_pages]
            group.sort(key=lambda item: (-item[3], item[0].unit, item[0].to_state))
            shortlisted.extend(group[: int(shortlist_size)])
        best_gain, best_moves, best_key = float("-inf"), None, None
        for size in range(1, int(max_swap_units) + 1):
            for chosen in combinations(range(len(shortlisted)), size):
                state_moves = tuple(sorted(shortlisted[index][0] for index in chosen))
                if len({move.unit for move in state_moves}) != size:
                    continue
                next_pages = pages + sum(move.page_delta for move in state_moves)
                if next_pages < 0 or next_pages > budget:
                    continue
                gain = sum(shortlisted[index][3] for index in chosen)
                for left, right in combinations(chosen, 2):
                    left_move, left_h, left_l, _ = shortlisted[left]
                    right_move, right_h, right_l, _ = shortlisted[right]
                    i, j = left_move.unit, right_move.unit
                    cross = (
                        left_h * right_h * q44[i, j]
                        + left_h * right_l * q42[i, j]
                        + left_l * right_h * q42[j, i]
                        + left_l * right_l * q22[i, j]
                    )
                    gain -= 2.0 * cross
                if gain > best_gain + tolerance or (
                    abs(gain - best_gain) <= tolerance
                    and (best_key is None or state_moves < best_key)
                ):
                    best_gain, best_moves, best_key = float(gain), chosen, state_moves
        if best_moves is None or best_gain <= tolerance:
            break
        proposal = states.copy()
        for index in best_moves:
            move = shortlisted[index][0]
            proposal[move.unit] = move.to_state
        proposal_damage = field.damage(proposal)
        if proposal_damage >= damage - tolerance:
            break
        states = proposal
        pages = field.pages(states)
        high, low, correlation4, correlation2 = _gram_state(field, states)
        damage = proposal_damage
        accepted += 1
        completed = pass_index + 1
    return SplitGramLocalSearchTrace(
        states.copy(), pages, float(damage), completed, accepted,
    )
