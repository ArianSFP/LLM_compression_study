"""Gate/up Q3 interaction fields with exact physical half-plane page costs.

The 18 states combine gate and up levels in ``{Q2,Q3,Q4}`` with down in
``{Q2,Q4}``.  Exact A/B/C self terms and the existing L4/L2 factor remain
sufficient.  Costs are integer 256-byte quanta.  Deployable layouts pair two
gate/up refinement planes into one 512-byte page and charge the exact union;
the ideal control charges each plane independently.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np

from .interaction_field import JointInteractionFactor
from .neuron_selector import UnitScoreMetadata


Q3_STATE_GATE_LEVEL = np.repeat(np.arange(3, dtype=np.int64), 6)
Q3_STATE_UP_LEVEL = np.tile(np.repeat(np.arange(3, dtype=np.int64), 2), 3)
Q3_STATE_DOWN_HIGH = np.tile(np.asarray([False, True]), 9)
Q3_STATE_HIDDEN_INDEX = 3 * Q3_STATE_GATE_LEVEL + Q3_STATE_UP_LEVEL
Q3_IDEAL_COST_QUANTA = (
    Q3_STATE_GATE_LEVEL + Q3_STATE_UP_LEVEL
    + 2 * Q3_STATE_DOWN_HIGH.astype(np.int64)
)
Q3_TARGET_STATE = 17
Q3_INHERITED_EIGHT_STATE_MAP = np.asarray(
    [0, 1, 4, 5, 12, 13, 16, 17], np.int64,
)

__all__ = [
    "Q3_STATE_GATE_LEVEL", "Q3_STATE_UP_LEVEL", "Q3_STATE_DOWN_HIGH",
    "Q3_STATE_HIDDEN_INDEX", "Q3_IDEAL_COST_QUANTA", "Q3_TARGET_STATE",
    "Q3_INHERITED_EIGHT_STATE_MAP", "Q3ProjectionResponses",
    "Q3InteractionField", "Q3PageLayout", "Q3CoordinateTrace",
    "Q3LocalSearchTrace", "q3_projection_responses", "q3_state_output",
    "q3_exact_damage", "build_q3_interaction_field", "plane_action_id",
    "fixed_gate_up_layout", "fit_coselection_layouts",
    "q3_coordinate_descent", "q3_local_search",
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
    value = np.asarray(states, np.int64).reshape(-1)
    if value.shape != (int(units),) or np.any((value < 0) | (value >= 18)):
        raise ValueError("Q3 states must contain one value in [0,17] per unit")
    return value.copy()


def plane_action_id(projection: str, stage: int, unit: int, units: int = 512) -> int:
    """Return the canonical gate/up half-plane action ID."""
    if projection not in ("gate", "up") or int(stage) not in (1, 2):
        raise ValueError("a plane action is gate/up stage 1/2")
    if int(unit) < 0 or int(unit) >= int(units):
        raise ValueError("unit lies outside the layout")
    projection_index = 0 if projection == "gate" else 1
    return ((2 * projection_index + int(stage) - 1) * int(units) + int(unit))


def _state_action_ids(state: int, unit: int, units: int) -> tuple[int, ...]:
    output: list[int] = []
    for stage in range(1, int(Q3_STATE_GATE_LEVEL[int(state)]) + 1):
        output.append(plane_action_id("gate", stage, unit, units))
    for stage in range(1, int(Q3_STATE_UP_LEVEL[int(state)]) + 1):
        output.append(plane_action_id("up", stage, unit, units))
    return tuple(output)


@dataclass(frozen=True)
class Q3ProjectionResponses:
    """Nine hidden responses h22..h44 and Q2/Q4 down matrices."""

    hidden: np.ndarray
    down2: np.ndarray
    down4: np.ndarray

    def __post_init__(self) -> None:
        hidden = np.asarray(self.hidden)
        down2 = np.asarray(self.down2)
        down4 = np.asarray(self.down4)
        if hidden.ndim != 2 or hidden.shape[1] != 9:
            raise ValueError("hidden responses must have shape [unit,9]")
        if down2.ndim != 2 or down4.shape != down2.shape or down2.shape[0] != hidden.shape[0]:
            raise ValueError("down matrices must have equal [unit,output] shapes")
        if not all(np.all(np.isfinite(value)) for value in (hidden, down2, down4)):
            raise ValueError("Q3 projection responses must be finite")

    @property
    def units(self) -> int:
        return int(np.asarray(self.hidden).shape[0])

    @property
    def target_hidden(self) -> np.ndarray:
        return np.asarray(self.hidden)[:, 8]

    @property
    def target_output(self) -> np.ndarray:
        return self.target_hidden @ np.asarray(self.down4)


def q3_projection_responses(
    q2: Sequence[np.ndarray], q3: Sequence[np.ndarray], q4: Sequence[np.ndarray],
    activation: np.ndarray,
) -> Q3ProjectionResponses:
    """Evaluate all 3x3 gate/up combinations and both down endpoints."""
    if len(q2) != 3 or len(q3) != 3 or len(q4) != 3:
        raise ValueError("Q2/Q3/Q4 must contain gate, up, and down matrices")
    x = np.asarray(activation, np.float64).reshape(-1)
    matrices = [tuple(np.asarray(item, np.float64) for item in level) for level in (q2, q3, q4)]
    if any(matrices[level][0].shape != matrices[0][0].shape for level in range(3)):
        raise ValueError("gate shapes changed across precision")
    if any(matrices[level][1].shape != matrices[0][1].shape for level in range(3)):
        raise ValueError("up shapes changed across precision")
    if any(matrices[level][2].shape != matrices[0][2].shape for level in range(3)):
        raise ValueError("down shapes changed across precision")
    gates = [level[0] @ x for level in matrices]
    ups = [level[1] @ x for level in matrices]
    hidden = np.stack([
        _silu(gates[gate]) * ups[up]
        for gate in range(3) for up in range(3)
    ], axis=1)
    return Q3ProjectionResponses(hidden, matrices[0][2].T, matrices[2][2].T)


def q3_state_output(
    responses: Q3ProjectionResponses, states: Sequence[int] | np.ndarray,
) -> np.ndarray:
    state = _validate_states(states, responses.units)
    rows = np.arange(responses.units)
    hidden = np.asarray(responses.hidden, np.float64)[rows, Q3_STATE_HIDDEN_INDEX[state]]
    high = Q3_STATE_DOWN_HIGH[state]
    output = np.zeros(np.asarray(responses.down2).shape[1], np.float64)
    if np.any(~high):
        output += hidden[~high] @ np.asarray(responses.down2, np.float64)[~high]
    if np.any(high):
        output += hidden[high] @ np.asarray(responses.down4, np.float64)[high]
    return output


def q3_exact_damage(
    responses: Q3ProjectionResponses, states: Sequence[int] | np.ndarray,
    *, proxy: np.ndarray | None = None, beta: float = 0.0,
) -> float:
    residual = responses.target_output - q3_state_output(responses, states)
    damage = float(residual @ residual)
    if proxy is not None and float(beta) != 0.0:
        projected = residual @ np.asarray(proxy, np.float64)
        damage += float(beta) * float(projected @ projected)
    return damage


@dataclass(frozen=True)
class Q3InteractionField:
    """Exact-self, approximate-cross objective for 18 Q3 gate/up states."""

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
        if rho.ndim != 3 or rho.shape[1] != 18:
            raise ValueError("rho must have shape [unit,18,rank]")
        if coefficients.shape != (rho.shape[0], 18, 2):
            raise ValueError("coefficients must have shape [unit,18,2]")
        if local.shape != rho.shape[:2] or residual.shape != local.shape:
            raise ValueError("local/self tables must have shape [unit,18]")
        if rho.shape[0] != self.factor.units or rho.shape[2] != self.factor.rank:
            raise ValueError("field and factor shapes disagree")
        if not all(np.all(np.isfinite(value)) for value in (rho, coefficients, local, residual)):
            raise ValueError("Q3 interaction field contains non-finite values")

    @property
    def units(self) -> int:
        return int(np.asarray(self.rho).shape[0])

    @property
    def rank(self) -> int:
        return int(np.asarray(self.rho).shape[2])

    def damage(self, states: Sequence[int] | np.ndarray) -> float:
        state = _validate_states(states, self.units)
        rows = np.arange(self.units)
        signature = np.asarray(self.rho)[rows, state].sum(axis=0)
        return float(signature @ signature + np.asarray(self.self_residual)[rows, state].sum())


def build_q3_interaction_field(
    factor: JointInteractionFactor, hidden: np.ndarray, metadata: UnitScoreMetadata,
    *, negative_tolerance: float = 1e-7,
) -> Q3InteractionField:
    """Build all 18 states from unchanged L4/L2 and exact A/B/C."""
    values = np.asarray(hidden, np.float64)
    if values.shape != (factor.units, 9) or metadata.units != factor.units:
        raise ValueError("hidden responses, A/B/C, and factor units disagree")
    target = values[:, 8]
    a = np.asarray(metadata.a, np.float64)
    b = np.asarray(metadata.b, np.float64)
    c = np.asarray(metadata.c, np.float64)
    l4 = np.asarray(factor.l4, np.float64)
    l2 = np.asarray(factor.l2, np.float64)
    rho = np.empty((factor.units, 18, factor.rank), np.float64)
    coefficients = np.empty((factor.units, 18, 2), np.float64)
    damage = np.empty((factor.units, 18), np.float64)
    for state in range(18):
        h = values[:, int(Q3_STATE_HIDDEN_INDEX[state])]
        if bool(Q3_STATE_DOWN_HIGH[state]):
            first, second = target - h, np.zeros_like(target)
            rho[:, state] = first[:, None] * l4
            damage[:, state] = first * first * a
        else:
            first, second = target, -h
            rho[:, state] = first[:, None] * l4 + second[:, None] * l2
            damage[:, state] = target * target * a - 2.0 * target * h * b + h * h * c
        coefficients[:, state, 0] = first
        coefficients[:, state, 1] = second
    scale = max(float(np.max(np.abs(damage), initial=0.0)), 1.0)
    if float(np.min(damage, initial=0.0)) < -float(negative_tolerance) * scale:
        raise ValueError("A/B/C produced materially negative Q3-state damage")
    clipped = int(np.count_nonzero(damage < 0.0))
    damage = np.maximum(damage, 0.0)
    norms = np.einsum("usr,usr->us", rho, rho, optimize=True)
    return Q3InteractionField(rho, coefficients, damage, damage - norms, factor, clipped)


@dataclass(frozen=True)
class Q3PageLayout:
    """One ideal layout or one/two complete physical gate/up replicas."""

    layout_id: str
    units: int
    action_pages: np.ndarray | None
    learned: bool = False
    fixed: bool = False

    def __post_init__(self) -> None:
        units = int(self.units)
        if units < 1:
            raise ValueError("layout must contain units")
        if self.action_pages is None:
            return
        pages = np.asarray(self.action_pages, np.int64)
        if pages.ndim != 2 or pages.shape[1] != 4 * units:
            raise ValueError("physical action_pages must have shape [replica,4*units]")
        if pages.shape[0] not in (1, 2) or np.any(pages < 0):
            raise ValueError("physical layout supports one or two nonnegative replicas")
        for replica in pages:
            counts = np.bincount(replica)
            if counts.shape[0] != 2 * units or np.any(counts != 2):
                raise ValueError("every physical page must pair exactly two half-planes")

    @property
    def ideal(self) -> bool:
        return self.action_pages is None

    @property
    def replicas(self) -> int:
        return 0 if self.ideal else int(np.asarray(self.action_pages).shape[0])

    @cached_property
    def state_page_ids(self) -> np.ndarray:
        """Unique gate/up page IDs for every replica, unit, and state."""
        if self.ideal:
            raise ValueError("ideal half-planes do not have physical page IDs")
        table = np.full((self.replicas, self.units, 18, 4), -1, np.int64)
        pages = np.asarray(self.action_pages, np.int64)
        for replica in range(self.replicas):
            for unit in range(self.units):
                for state in range(18):
                    actions = _state_action_ids(state, unit, self.units)
                    if actions:
                        values = np.unique(pages[replica, np.asarray(actions, np.int64)])
                        table[replica, unit, state, :len(values)] = values
        table.setflags(write=False)
        return table

    @cached_property
    def state_page_multiplicity(self) -> np.ndarray:
        """Action multiplicities for ``state_page_ids`` (paired actions may coincide)."""
        if self.ideal:
            raise ValueError("ideal half-planes do not have physical page IDs")
        table = np.zeros((self.replicas, self.units, 18, 4), np.int8)
        pages = np.asarray(self.action_pages, np.int64)
        for replica in range(self.replicas):
            for unit in range(self.units):
                for state in range(18):
                    actions = _state_action_ids(state, unit, self.units)
                    if actions:
                        _, counts = np.unique(
                            pages[replica, np.asarray(actions, np.int64)], return_counts=True,
                        )
                        table[replica, unit, state, :len(counts)] = counts
        table.setflags(write=False)
        return table

    def action_ids(self, states: Sequence[int] | np.ndarray) -> np.ndarray:
        state = _validate_states(states, self.units)
        actions = [
            action for unit, value in enumerate(state.tolist())
            for action in _state_action_ids(value, unit, self.units)
        ]
        return np.asarray(actions, np.int64)

    def page_union(self, states: Sequence[int] | np.ndarray, replica: int) -> np.ndarray:
        if self.ideal:
            raise ValueError("ideal half-planes do not have 512-byte page IDs")
        if int(replica) < 0 or int(replica) >= self.replicas:
            raise ValueError("replica lies outside the layout")
        state = _validate_states(states, self.units)
        actions = self.action_ids(state)
        gate_up = np.unique(np.asarray(self.action_pages)[int(replica), actions])
        down = np.flatnonzero(Q3_STATE_DOWN_HIGH[state]) + 2 * self.units
        return np.sort(np.concatenate((gate_up, down.astype(np.int64))))

    def cost_quanta(self, states: Sequence[int] | np.ndarray) -> int:
        state = _validate_states(states, self.units)
        if self.ideal:
            return int(Q3_IDEAL_COST_QUANTA[state].sum())
        return 2 * min(len(self.page_union(state, replica)) for replica in range(self.replicas))

    def selected_replica(self, states: Sequence[int] | np.ndarray) -> int:
        if self.ideal:
            return -1
        lengths = [len(self.page_union(states, replica)) for replica in range(self.replicas)]
        return int(np.argmin(lengths))


def fixed_gate_up_layout(units: int = 512) -> Q3PageLayout:
    """Pair gate/up planes for the same unit and refinement stage."""
    pages = np.empty((1, 4 * int(units)), np.int64)
    for stage in (1, 2):
        for unit in range(int(units)):
            page = (stage - 1) * int(units) + unit
            pages[0, plane_action_id("gate", stage, unit, units)] = page
            pages[0, plane_action_id("up", stage, unit, units)] = page
    return Q3PageLayout("fixed_gate_up_same_unit_stage", int(units), pages, fixed=True)


def monolithic_q2q4_layout(units: int = 512) -> Q3PageLayout:
    """Pair both refinement stages of each projection as the PR13 page."""
    pages = np.empty((1, 4 * int(units)), np.int64)
    for projection_index, projection in enumerate(("gate", "up")):
        for unit in range(int(units)):
            page = projection_index * int(units) + unit
            pages[0, plane_action_id(projection, 1, unit, units)] = page
            pages[0, plane_action_id(projection, 2, unit, units)] = page
    return Q3PageLayout("monolithic_q2q4_projection_page", int(units), pages, fixed=True)


def _greedy_pairing(
    affinity: np.ndarray, counts: np.ndarray,
    forbidden: dict[int, int] | None = None,
) -> np.ndarray:
    actions = int(affinity.shape[0])
    remaining = np.ones(actions, dtype=bool)
    page_for = np.full(actions, -1, np.int64)
    order = np.lexsort((np.arange(actions), -np.asarray(counts)))
    page = 0
    for anchor in order.tolist():
        if not remaining[anchor]:
            continue
        remaining[anchor] = False
        candidates = np.flatnonzero(remaining)
        if not len(candidates):
            raise RuntimeError("odd number of half-plane actions")
        scores = np.asarray(affinity[anchor, candidates], np.float64)
        if forbidden is not None and anchor in forbidden and len(candidates) > 1:
            scores[candidates == int(forbidden[anchor])] = -np.inf
        best_value = float(np.max(scores))
        tied = candidates[np.flatnonzero(scores == best_value)]
        if len(tied) > 1:
            best_count = np.max(np.asarray(counts)[tied])
            tied = tied[np.asarray(counts)[tied] == best_count]
        partner = int(np.min(tied))
        remaining[partner] = False
        page_for[anchor] = page_for[partner] = page
        page += 1
    return page_for


def fit_coselection_layouts(
    incidence: np.ndarray, *, units: int = 512, replicas: int = 2,
) -> np.ndarray:
    """Fit one or two deterministic training-only co-selection pairings."""
    selected = np.asarray(incidence)
    if selected.ndim != 2 or selected.shape[1] != 4 * int(units):
        raise ValueError("incidence must have shape [sample,4*units]")
    if selected.shape[0] < 1 or int(replicas) not in (1, 2):
        raise ValueError("layout fit needs samples and one/two replicas")
    if not np.all((selected == 0) | (selected == 1)):
        raise ValueError("incidence must be binary")
    values = selected.astype(np.float32, copy=False)
    counts = values.sum(axis=0)
    affinity = values.T @ values
    np.fill_diagonal(affinity, -np.inf)
    first = _greedy_pairing(affinity, counts)
    output = [first]
    if int(replicas) == 2:
        forbidden: dict[int, int] = {}
        for page in range(2 * int(units)):
            pair = np.flatnonzero(first == page)
            forbidden[int(pair[0])] = int(pair[1])
            forbidden[int(pair[1])] = int(pair[0])
        output.append(_greedy_pairing(affinity, counts, forbidden))
    return np.stack(output)


class _PageTracker:
    def __init__(self, layout: Q3PageLayout, states: np.ndarray) -> None:
        self.layout = layout
        self.states = states.copy()
        if layout.ideal:
            self.quanta = int(Q3_IDEAL_COST_QUANTA[states].sum())
            return
        self.counts = np.zeros((layout.replicas, 2 * layout.units), np.int16)
        self.unique = np.zeros(layout.replicas, np.int64)
        self.down = int(np.count_nonzero(Q3_STATE_DOWN_HIGH[states]))
        for unit, state in enumerate(states.tolist()):
            for replica in range(layout.replicas):
                for action in _state_action_ids(state, unit, layout.units):
                    page = int(np.asarray(layout.action_pages)[replica, action])
                    self.counts[replica, page] += 1
        self.unique[:] = np.count_nonzero(self.counts, axis=1)
        self.quanta = 2 * (self.down + int(np.min(self.unique)))

    def candidate_costs(self, unit: int) -> np.ndarray:
        """Return exact costs for all 18 single-unit destinations at once."""
        unit = int(unit)
        source = int(self.states[unit])
        if self.layout.ideal:
            return self.quanta + Q3_IDEAL_COST_QUANTA - Q3_IDEAL_COST_QUANTA[source]
        down = (
            self.down - int(Q3_STATE_DOWN_HIGH[source])
            + Q3_STATE_DOWN_HIGH.astype(np.int64)
        )
        replica_unique = []
        for replica in range(self.layout.replicas):
            table = self.layout.state_page_ids[replica, unit]
            source_ids = table[source]
            source_valid = source_ids >= 0
            source_ids = source_ids[source_valid]
            source_multiplicity = self.layout.state_page_multiplicity[
                replica, unit, source, source_valid,
            ]
            base_unique = int(self.unique[replica])
            if len(source_ids):
                base_unique -= int(np.count_nonzero(self.counts[replica, source_ids] == source_multiplicity))
            valid = table >= 0
            safe = np.where(valid, table, 0)
            source_removal = np.sum(
                (safe[..., None] == source_ids.reshape(1, 1, -1))
                * source_multiplicity.reshape(1, 1, -1), axis=-1,
            ) if len(source_ids) else np.zeros_like(safe)
            counts_after_removal = self.counts[replica, safe] - source_removal
            added = np.count_nonzero(valid & (counts_after_removal == 0), axis=1)
            replica_unique.append(base_unique + added)
        return 2 * (down + np.min(np.stack(replica_unique), axis=0))

    def candidate(self, moves: Iterable[tuple[int, int]]) -> int:
        changes = tuple((int(unit), int(state)) for unit, state in moves)
        if len(changes) == 1:
            unit, state = changes[0]
            return int(self.candidate_costs(unit)[state])
        if self.layout.ideal:
            return int(self.quanta + sum(
                int(Q3_IDEAL_COST_QUANTA[state] - Q3_IDEAL_COST_QUANTA[self.states[unit]])
                for unit, state in changes
            ))
        down = self.down + sum(
            int(Q3_STATE_DOWN_HIGH[state]) - int(Q3_STATE_DOWN_HIGH[self.states[unit]])
            for unit, state in changes
        )
        replica_costs = []
        pages = np.asarray(self.layout.action_pages)
        for replica in range(self.layout.replicas):
            delta: Counter[int] = Counter()
            for unit, state in changes:
                source = int(self.states[unit])
                for action in _state_action_ids(source, unit, self.layout.units):
                    delta[int(pages[replica, action])] -= 1
                for action in _state_action_ids(state, unit, self.layout.units):
                    delta[int(pages[replica, action])] += 1
            unique = int(self.unique[replica])
            for page, change in delta.items():
                before = int(self.counts[replica, page])
                after = before + int(change)
                if after < 0:
                    raise RuntimeError("page tracker underflow")
                unique += int(after > 0) - int(before > 0)
            replica_costs.append(unique)
        return 2 * (down + min(replica_costs))

    def apply(self, unit: int, destination: int) -> None:
        unit, destination = int(unit), int(destination)
        source = int(self.states[unit])
        expected = self.candidate(((unit, destination),))
        if not self.layout.ideal:
            pages = np.asarray(self.layout.action_pages)
            self.down += int(Q3_STATE_DOWN_HIGH[destination]) - int(Q3_STATE_DOWN_HIGH[source])
            for replica in range(self.layout.replicas):
                delta: Counter[int] = Counter()
                for action in _state_action_ids(source, unit, self.layout.units):
                    delta[int(pages[replica, action])] -= 1
                for action in _state_action_ids(destination, unit, self.layout.units):
                    delta[int(pages[replica, action])] += 1
                for page, change in delta.items():
                    before = int(self.counts[replica, page])
                    after = before + int(change)
                    self.counts[replica, page] = after
                    self.unique[replica] += int(after > 0) - int(before > 0)
        self.states[unit] = destination
        self.quanta = expected


@dataclass(frozen=True)
class Q3CoordinateTrace:
    states: np.ndarray
    cost_quanta: int
    damage: float
    sweeps: int
    accepted_moves: int
    interaction_dot_macs: int


def q3_coordinate_descent(
    field: Q3InteractionField, layout: Q3PageLayout,
    seed: Sequence[int] | np.ndarray, budget_quanta: int, *,
    max_sweeps: int = 8, page_price: float = 0.0,
    allowed_states: Sequence[int] = tuple(range(18)), tolerance: float = 1e-12,
) -> Q3CoordinateTrace:
    """Incremental 18-state coordinate descent with exact page-union cost."""
    states = _validate_states(seed, field.units)
    allowed = tuple(sorted({int(value) for value in allowed_states}))
    if not allowed or any(value < 0 or value >= 18 for value in allowed):
        raise ValueError("allowed states must be a nonempty subset of [0,17]")
    if any(int(value) not in allowed for value in states):
        raise ValueError("seed contains a forbidden state")
    tracker = _PageTracker(layout, states)
    budget = int(budget_quanta)
    if tracker.quanta > budget:
        raise ValueError("seed exceeds the cost budget")
    price = float(page_price)
    if not np.isfinite(price) or price < 0.0:
        raise ValueError("page_price must be finite and nonnegative")
    rho = np.asarray(field.rho)
    coefficients = np.asarray(field.coefficients)
    local = np.asarray(field.local_damage)
    l4 = np.asarray(field.factor.l4, np.float64)
    l2 = np.asarray(field.factor.l2, np.float64)
    rows = np.arange(field.units)
    residual = rho[rows, states].sum(axis=0)
    damage = field.damage(states)
    objective = damage + price * tracker.quanta
    accepted = completed = 0
    allowed_mask = np.zeros(18, bool)
    allowed_mask[list(allowed)] = True
    for sweep in range(int(max_sweeps)):
        changed = False
        for unit in range(field.units):
            source = int(states[unit])
            residual_without = residual - rho[unit, source]
            dots = np.asarray((residual_without @ l4[unit], residual_without @ l2[unit]))
            scores = 2.0 * (coefficients[unit] @ dots) + local[unit]
            costs = tracker.candidate_costs(unit)
            candidates = np.where(allowed_mask & (costs <= budget), scores + price * costs, np.inf)
            destination = int(np.argmin(candidates))
            if destination == source:
                continue
            candidate_damage = damage + float(scores[destination] - scores[source])
            candidate_objective = candidate_damage + price * int(costs[destination])
            if candidate_objective < objective - float(tolerance):
                tracker.apply(unit, destination)
                states[unit] = destination
                residual = residual_without + rho[unit, destination]
                damage, objective = candidate_damage, candidate_objective
                accepted += 1
                changed = True
        completed = sweep + 1
        if not changed:
            break
    exact = field.damage(states)
    exact_cost = layout.cost_quanta(states)
    if exact_cost != tracker.quanta or not np.isclose(exact, damage, rtol=1e-9, atol=1e-8):
        raise RuntimeError("incremental Q3 coordinate state lost parity")
    if not np.isclose(objective, exact + price * exact_cost, rtol=1e-9, atol=1e-8):
        raise RuntimeError("incremental Q3 coordinate objective lost parity")
    return Q3CoordinateTrace(states.copy(), exact_cost, float(exact), completed, accepted,
                             2 * field.units * field.rank * completed)


@dataclass(frozen=True)
class Q3LocalSearchTrace:
    states: np.ndarray
    cost_quanta: int
    damage: float
    passes: int
    accepted_bundles: int
    maximum_shortlist: int


def q3_local_search(
    field: Q3InteractionField, layout: Q3PageLayout,
    seed: Sequence[int] | np.ndarray, budget_quanta: int, *,
    shortlist_size: int = 8, max_swap_units: int = 3, max_passes: int = 6,
    page_price: float = 0.0, allowed_states: Sequence[int] = tuple(range(18)),
    tolerance: float = 1e-12,
) -> Q3LocalSearchTrace:
    """Bounded 1/2/3-unit repair with exact non-additive page-union changes."""
    states = _validate_states(seed, field.units)
    allowed = tuple(sorted({int(value) for value in allowed_states}))
    if not allowed or any(value < 0 or value >= 18 for value in allowed):
        raise ValueError("invalid allowed states")
    tracker = _PageTracker(layout, states)
    budget = int(budget_quanta)
    if tracker.quanta > budget:
        raise ValueError("seed exceeds the cost budget")
    price = float(page_price)
    rho = np.asarray(field.rho)
    self_term = np.asarray(field.self_residual)
    rows = np.arange(field.units)
    residual = rho[rows, states].sum(axis=0)
    damage = field.damage(states)
    objective = damage + price * tracker.quanta
    accepted = evaluated = maximum_shortlist = 0
    allowed_mask = np.zeros(18, bool)
    allowed_mask[list(allowed)] = True
    destinations = np.arange(18, dtype=np.int64)[None, :]
    for pass_index in range(int(max_passes)):
        evaluated = pass_index + 1
        delta = rho - rho[rows, states][:, None, :]
        change = (
            2.0 * np.einsum("usr,r->us", delta, residual, optimize=True)
            + np.einsum("usr,usr->us", delta, delta, optimize=True)
            + self_term - self_term[rows, states][:, None]
        )
        costs = np.stack([tracker.candidate_costs(unit) for unit in range(field.units)])
        cost_delta = costs - tracker.quanta
        gain = -change - price * cost_delta
        valid = allowed_mask[None, :] & (destinations != states[:, None])

        def best_moves(mask: np.ndarray) -> list[tuple]:
            units, state_values = np.nonzero(valid & mask)
            if not len(units):
                return []
            order = np.lexsort((state_values, units, -gain[units, state_values]))[:int(shortlist_size)]
            return [
                (int(units[index]), int(states[units[index]]), int(state_values[index]),
                 delta[units[index], state_values[index]], int(cost_delta[units[index], state_values[index]]),
                 float(gain[units[index], state_values[index]]))
                for index in order
            ]

        nonpositive = best_moves(cost_delta <= 0)
        positive = best_moves(cost_delta > 0)
        shortlisted = nonpositive + positive
        maximum_shortlist = max(maximum_shortlist, len(shortlisted))
        if not shortlisted:
            break
        deltas = np.stack([item[3] for item in shortlisted])
        pairwise = deltas @ deltas.T
        best_objective = objective
        best_choice = None
        best_damage = damage
        best_cost = tracker.quanta
        for size in range(1, int(max_swap_units) + 1):
            for chosen in combinations(range(len(shortlisted)), size):
                selected = [shortlisted[index] for index in chosen]
                if len({item[0] for item in selected}) != size:
                    continue
                cost = tracker.candidate((item[0], item[2]) for item in selected)
                if cost > budget:
                    continue
                change = -sum(item[5] + price * item[4] for item in selected)
                change += 2.0 * sum(pairwise[left, right] for left, right in combinations(chosen, 2))
                candidate_damage = damage + float(change)
                candidate_objective = candidate_damage + price * cost
                key = tuple((item[0], item[2]) for item in selected)
                if (candidate_objective < best_objective - tolerance
                        or (abs(candidate_objective - best_objective) <= tolerance
                            and best_choice is not None and key < best_choice[0])):
                    best_objective = candidate_objective
                    best_choice = (key, selected)
                    best_damage, best_cost = candidate_damage, cost
        if best_choice is None:
            break
        for item in best_choice[1]:
            tracker.apply(item[0], item[2])
            states[item[0]] = item[2]
        damage, objective = best_damage, best_objective
        residual = rho[rows, states].sum(axis=0)
        if tracker.quanta != best_cost:
            raise RuntimeError("bundle page tracker lost parity")
        accepted += 1
    exact = field.damage(states)
    exact_cost = layout.cost_quanta(states)
    if exact_cost != tracker.quanta or not np.isclose(exact, damage, rtol=1e-9, atol=1e-8):
        raise RuntimeError("Q3 local-search result lost parity")
    return Q3LocalSearchTrace(states.copy(), exact_cost, float(exact), evaluated,
                              accepted, maximum_shortlist)
