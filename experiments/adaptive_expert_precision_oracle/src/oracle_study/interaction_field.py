"""Low-rank signed interaction fields for four-state MXFP4 unit actions.

The complete teacher uses three 512 by 512 down-column Gram blocks.  This
module compresses their joint 1024 by 1024 PSD geometry while retaining exact
per-unit state energies from A/B/C.  Consequently only cross-unit terms are
approximated.  The full teacher Gram is never consulted by the optimizers in
this module; callers may use it after selection for evaluation only.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np

from .neuron_selector import UnitScoreMetadata
from .set_utility_oracles import (
    STATE_00,
    STATE_D,
    STATE_G,
    STATE_GD,
    STATE_PAGE_COSTS,
    UnitStateMove,
)
from .unit_set_teacher import DownMetricGram


__all__ = [
    "JointInteractionFactor",
    "EncodedInteractionFactor",
    "InteractionField",
    "FieldAction",
    "FieldBudgetSnapshot",
    "FieldForwardTrace",
    "FieldCoordinateTrace",
    "FieldLocalSearchTrace",
    "RelaxedFieldTrace",
    "assemble_joint_gram",
    "factor_joint_gram",
    "factor_exact_proxy_plus_tail",
    "truncate_interaction_factor",
    "build_interaction_field",
    "field_forward_greedy",
    "field_coordinate_descent",
    "field_local_search",
    "relaxed_field_descent",
    "round_relaxed_pages",
    "walsh_hadamard",
    "quantize_interaction_factor",
    "encode_interaction_factor",
    "make_encoded_factor_self_safe",
    "make_factor_self_safe",
]


_FORWARD_TRANSITIONS = (
    (STATE_00, STATE_G, 2),
    (STATE_00, STATE_D, 1),
    (STATE_00, STATE_GD, 3),
    (STATE_G, STATE_GD, 1),
    (STATE_D, STATE_GD, 2),
)


def _canonical_columns(matrix: np.ndarray) -> np.ndarray:
    result = np.asarray(matrix, np.float64).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return result


def _validate_states(states: Sequence[int] | np.ndarray, units: int) -> np.ndarray:
    value = np.asarray(states, np.int64).reshape(-1)
    if value.shape != (int(units),) or np.any((value < 0) | (value > 3)):
        raise ValueError("states must contain one value in {00,G,D,GD} per unit")
    return value.copy()


def assemble_joint_gram(metadata: DownMetricGram) -> np.ndarray:
    """Return the symmetric joint Gram ``[[Q44,Q42],[Q42.T,Q22]]``."""
    q44 = np.asarray(metadata.q44, np.float64)
    q42 = np.asarray(metadata.q42, np.float64)
    q22 = np.asarray(metadata.q22, np.float64)
    joint = np.block([[q44, q42], [q42.T, q22]])
    return 0.5 * (joint + joint.T)


def _pivoted_cholesky(matrix: np.ndarray, rank: int, tolerance: float) -> np.ndarray:
    gram = np.asarray(matrix, np.float64)
    size = gram.shape[0]
    factor = np.zeros((size, int(rank)), np.float64)
    residual = np.maximum(np.diag(gram).copy(), 0.0)
    used = 0
    for column in range(int(rank)):
        pivot = int(np.argmax(residual))
        pivot_value = float(residual[pivot])
        if pivot_value <= tolerance:
            break
        previous = factor[:, :column] @ factor[pivot, :column]
        values = (gram[:, pivot] - previous) / np.sqrt(pivot_value)
        factor[:, column] = values
        residual = np.maximum(residual - values * values, 0.0)
        used = column + 1
    return _canonical_columns(factor[:, :used])


@dataclass(frozen=True)
class JointInteractionFactor:
    """Signed Q4/Q2 row signatures and exact serialized-cost accounting."""

    l4: np.ndarray
    l2: np.ndarray
    method: str
    tail_rank: int
    exact_rank: int = 0
    encoding: str = "fp32"
    storage_bytes: int | None = None

    def __post_init__(self) -> None:
        high = np.asarray(self.l4)
        low = np.asarray(self.l2)
        if high.ndim != 2 or low.shape != high.shape:
            raise ValueError("L4 and L2 must have equal [unit, rank] shapes")
        if not np.all(np.isfinite(high)) or not np.all(np.isfinite(low)):
            raise ValueError("interaction signatures must be finite")
        if self.tail_rank < 0 or self.exact_rank < 0:
            raise ValueError("factor ranks must be nonnegative")
        if self.tail_rank + self.exact_rank != high.shape[1]:
            raise ValueError("tail_rank + exact_rank must equal stored factor width")

    @property
    def units(self) -> int:
        return int(np.asarray(self.l4).shape[0])

    @property
    def rank(self) -> int:
        return int(np.asarray(self.l4).shape[1])

    @property
    def decoded_bytes(self) -> int:
        return int(np.asarray(self.l4).nbytes + np.asarray(self.l2).nbytes)

    @property
    def payload_bytes(self) -> int:
        return self.decoded_bytes if self.storage_bytes is None else int(self.storage_bytes)

    def joint(self) -> np.ndarray:
        return np.concatenate((np.asarray(self.l4), np.asarray(self.l2)), axis=0)


@dataclass(frozen=True)
class EncodedInteractionFactor:
    """Physically packed signed rows with FP16 scales and one FP32 shrink."""

    packed_codes: np.ndarray
    row_scales: np.ndarray
    units: int
    rank: int
    method: str
    tail_rank: int
    exact_rank: int
    encoding: str
    global_scale: np.float32 = np.float32(1.0)

    def __post_init__(self) -> None:
        codes = np.asarray(self.packed_codes)
        scales = np.asarray(self.row_scales)
        if codes.dtype != np.uint8 or codes.ndim != 1:
            raise ValueError("packed codes must be a flat uint8 array")
        if scales.dtype != np.float16 or scales.shape != (2 * int(self.units),):
            raise ValueError("row scales must be FP16 with one value per Q4/Q2 row")
        if self.rank < 1 or self.tail_rank + self.exact_rank != self.rank:
            raise ValueError("encoded factor ranks are inconsistent")
        values = 2 * int(self.units) * int(self.rank)
        expected = values if self.encoding.startswith("int8") else (values + 1) // 2
        if codes.size != expected:
            raise ValueError("packed code byte count disagrees with encoding")
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
            raise ValueError("encoded row scales must be positive and finite")
        if not np.isfinite(self.global_scale) or self.global_scale < 0:
            raise ValueError("global shrink must be nonnegative and finite")

    @property
    def payload_bytes(self) -> int:
        return int(self.packed_codes.nbytes + self.row_scales.nbytes + 4)

    def decode(self) -> JointInteractionFactor:
        values = 2 * self.units * self.rank
        if self.encoding.startswith("int8"):
            quantized = np.asarray(self.packed_codes).view(np.int8).astype(np.float32)
        elif self.encoding.startswith("int4"):
            packed = np.asarray(self.packed_codes, np.uint8)
            unsigned = np.empty(2 * packed.size, np.uint8)
            unsigned[0::2] = packed & 0x0F
            unsigned[1::2] = packed >> 4
            signed = unsigned[:values].astype(np.int8)
            signed[signed >= 8] -= 16
            quantized = signed.astype(np.float32)
        else:
            raise ValueError(f"unsupported encoded factor: {self.encoding}")
        joint = quantized.reshape(2 * self.units, self.rank)
        joint *= np.asarray(self.row_scales, np.float32)[:, None]
        joint *= np.float32(self.global_scale)
        return JointInteractionFactor(
            l4=joint[: self.units],
            l2=joint[self.units :],
            method=self.method,
            tail_rank=self.tail_rank,
            exact_rank=self.exact_rank,
            encoding=self.encoding,
            storage_bytes=self.payload_bytes,
        )


def factor_joint_gram(
    metadata: DownMetricGram,
    rank: int,
    *,
    method: str = "eigh",
    tolerance: float = 1e-10,
    dtype: np.dtype | type = np.float32,
) -> JointInteractionFactor:
    """Factor the static joint PSD teacher geometry at a bounded rank."""
    requested = int(rank)
    if requested < 0 or requested > 2 * metadata.units:
        raise ValueError("rank lies outside the joint Gram dimension")
    joint = assemble_joint_gram(metadata)
    if requested == 0:
        raw = np.zeros((joint.shape[0], 0), np.float64)
    elif method == "eigh":
        values, vectors = np.linalg.eigh(joint)
        order = np.argsort(values)[::-1]
        chosen = order[:requested]
        positive = values[chosen] > float(tolerance)
        chosen = chosen[positive]
        raw = vectors[:, chosen] * np.sqrt(np.maximum(values[chosen], 0.0))[None, :]
        raw = _canonical_columns(raw)
    elif method == "pivoted_cholesky":
        raw = _pivoted_cholesky(joint, requested, float(tolerance))
    else:
        raise ValueError(f"unsupported factor method: {method}")
    units = metadata.units
    decoded = np.asarray(raw, dtype=dtype)
    return JointInteractionFactor(
        l4=decoded[:units], l2=decoded[units:], method=method,
        tail_rank=int(decoded.shape[1]), encoding=str(np.dtype(dtype)),
    )


def factor_exact_proxy_plus_tail(
    down2: np.ndarray,
    down4: np.ndarray,
    proxy: np.ndarray,
    beta: float,
    tail_rank: int,
    *,
    method: str = "eigh",
    tolerance: float = 1e-10,
    dtype: np.dtype | type = np.float32,
) -> JointInteractionFactor:
    """Append exact ``sqrt(beta) P.T d`` coordinates to a Euclidean tail."""
    d2 = np.asarray(down2, np.float64)
    d4 = np.asarray(down4, np.float64)
    p = np.asarray(proxy, np.float64)
    if d2.ndim != 2 or d4.shape != d2.shape:
        raise ValueError("down2/down4 must have equal [output, unit] shapes")
    if p.ndim == 1:
        p = p[:, None]
    if p.ndim != 2 or p.shape[0] != d2.shape[0]:
        raise ValueError("proxy must have shape [output, exact_rank]")
    if beta < 0.0:
        raise ValueError("beta must be nonnegative")
    euclidean = DownMetricGram(
        q44=d4.T @ d4, q42=d4.T @ d2, q22=d2.T @ d2,
    )
    tail = factor_joint_gram(
        euclidean, int(tail_rank), method=method, tolerance=tolerance, dtype=np.float64,
    )
    scale = np.sqrt(float(beta))
    exact4 = scale * (d4.T @ p)
    exact2 = scale * (d2.T @ p)
    high = np.concatenate((np.asarray(tail.l4), exact4), axis=1)
    low = np.concatenate((np.asarray(tail.l2), exact2), axis=1)
    return JointInteractionFactor(
        l4=np.asarray(high, dtype=dtype), l2=np.asarray(low, dtype=dtype),
        method=f"exact_proxy_plus_{method}_tail", tail_rank=tail.rank,
        exact_rank=int(p.shape[1]), encoding=str(np.dtype(dtype)),
    )


def truncate_interaction_factor(
    factor: JointInteractionFactor,
    tail_rank: int,
    *,
    dtype: np.dtype | type | None = None,
) -> JointInteractionFactor:
    """Take a tail prefix while retaining every appended exact coordinate."""
    requested = int(tail_rank)
    if requested < 0 or requested > factor.tail_rank:
        raise ValueError("tail_rank lies outside the fitted factor")
    tail = np.arange(requested, dtype=np.int64)
    exact = np.arange(factor.tail_rank, factor.rank, dtype=np.int64)
    columns = np.concatenate((tail, exact))
    target = np.asarray(factor.l4).dtype if dtype is None else np.dtype(dtype)
    return JointInteractionFactor(
        l4=np.asarray(np.asarray(factor.l4)[:, columns], dtype=target),
        l2=np.asarray(np.asarray(factor.l2)[:, columns], dtype=target),
        method=factor.method,
        tail_rank=requested,
        exact_rank=factor.exact_rank,
        encoding=str(np.dtype(target)),
    )


@dataclass(frozen=True)
class InteractionField:
    """Activation-dependent four-state residual signatures and self energies."""

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
        if rho.ndim != 3 or rho.shape[1] != 4:
            raise ValueError("rho must have shape [unit,4,rank]")
        if coefficients.shape != (rho.shape[0], 4, 2):
            raise ValueError("coefficients must have shape [unit,4,2]")
        if local.shape != rho.shape[:2] or residual.shape != local.shape:
            raise ValueError("local/self tables must have shape [unit,4]")
        if rho.shape[0] != self.factor.units or rho.shape[2] != self.factor.rank:
            raise ValueError("field and factor shapes disagree")
        if not all(np.all(np.isfinite(value)) for value in (rho, coefficients, local, residual)):
            raise ValueError("interaction field contains non-finite values")

    @property
    def units(self) -> int:
        return int(np.asarray(self.rho).shape[0])

    @property
    def rank(self) -> int:
        return int(np.asarray(self.rho).shape[2])

    def pages(self, states: Sequence[int] | np.ndarray) -> int:
        state = _validate_states(states, self.units)
        return int(STATE_PAGE_COSTS[state].sum())

    def residual_signature(self, states: Sequence[int] | np.ndarray) -> np.ndarray:
        state = _validate_states(states, self.units)
        return np.asarray(self.rho)[np.arange(self.units), state].sum(axis=0)

    def damage(self, states: Sequence[int] | np.ndarray) -> float:
        state = _validate_states(states, self.units)
        indices = np.arange(self.units)
        signature = np.asarray(self.rho)[indices, state].sum(axis=0)
        return float(signature @ signature + np.asarray(self.self_residual)[indices, state].sum())


def build_interaction_field(
    factor: JointInteractionFactor,
    h2: np.ndarray,
    h4: np.ndarray,
    metadata: UnitScoreMetadata,
    *,
    negative_tolerance: float = 1e-7,
) -> InteractionField:
    """Construct the exact-diagonal, approximate-cross four-state objective."""
    low = np.asarray(h2, np.float64).reshape(-1)
    high = np.asarray(h4, np.float64).reshape(-1)
    if low.shape != high.shape or low.size != factor.units or metadata.units != factor.units:
        raise ValueError("hidden states, A/B/C and factor units disagree")
    a = np.asarray(metadata.a, np.float64)
    b = np.asarray(metadata.b, np.float64)
    c = np.asarray(metadata.c, np.float64)
    l4 = np.asarray(factor.l4, np.float64)
    l2 = np.asarray(factor.l2, np.float64)
    rho = np.stack((
        high[:, None] * l4 - low[:, None] * l2,
        high[:, None] * (l4 - l2),
        (high - low)[:, None] * l4,
        np.zeros_like(l4),
    ), axis=1)
    coefficients = np.stack((
        np.stack((high, -low), axis=1),
        np.stack((high, -high), axis=1),
        np.stack((high - low, np.zeros_like(high)), axis=1),
        np.zeros((factor.units, 2), np.float64),
    ), axis=1)
    damage = np.stack((
        high * high * a - 2.0 * high * low * b + low * low * c,
        high * high * (a - 2.0 * b + c),
        (high - low) ** 2 * a,
        np.zeros_like(high),
    ), axis=1)
    scale = max(float(np.max(np.abs(damage), initial=0.0)), 1.0)
    if float(np.min(damage, initial=0.0)) < -float(negative_tolerance) * scale:
        raise ValueError("A/B/C produced materially negative local state damage")
    clipped = int(np.count_nonzero(damage < 0.0))
    damage = np.maximum(damage, 0.0)
    signature_norm = np.einsum("usr,usr->us", rho, rho, optimize=True)
    self_residual = damage - signature_norm
    return InteractionField(
        rho=np.asarray(rho, np.float64), coefficients=coefficients, local_damage=damage,
        self_residual=np.asarray(self_residual, np.float64), factor=factor,
        clipped_local_values=clipped,
    )


@dataclass(frozen=True)
class FieldAction:
    unit: int
    from_state: int
    to_state: int
    incremental_pages: int
    cumulative_pages: int
    marginal_gain: float


@dataclass(frozen=True)
class FieldBudgetSnapshot:
    states: np.ndarray
    pages: int
    damage: float
    prefix: int


@dataclass(frozen=True)
class FieldForwardTrace:
    actions: tuple[FieldAction, ...]
    snapshots: Mapping[int, FieldBudgetSnapshot]
    candidate_evaluations: int


def field_forward_greedy(
    field: InteractionField,
    page_budgets: Sequence[int] | int,
) -> FieldForwardTrace:
    """Grow one residual-aware hybrid path and retain each budget's best prefix."""
    requested = (
        (int(page_budgets),) if isinstance(page_budgets, (int, np.integer))
        else tuple(sorted({int(value) for value in page_budgets}))
    )
    if not requested or any(value < 0 or value > 3 * field.units for value in requested):
        raise ValueError("page budgets must be nonempty and lie in [0,3*units]")
    states = np.full(field.units, STATE_00, np.int64)
    indices = np.arange(field.units)
    rho = np.asarray(field.rho)
    self_term = np.asarray(field.self_residual)
    u = rho[indices, states].sum(axis=0)
    damage = field.damage(states)
    pages = 0
    actions: list[FieldAction] = []
    path_states = [states.copy()]
    path_pages = [0]
    path_damage = [damage]
    evaluations = 0
    maximum = max(requested)
    while pages < maximum:
        candidates: list[tuple[float, float, int, int, int, int, int, np.ndarray]] = []
        for transition_id, (source, destination, cost) in enumerate(_FORWARD_TRANSITIONS):
            if pages + cost > maximum:
                continue
            units = np.flatnonzero(states == source)
            if units.size == 0:
                continue
            delta = rho[units, destination] - rho[units, source]
            delta_self = self_term[units, destination] - self_term[units, source]
            change = 2.0 * (delta @ u) + np.einsum("ij,ij->i", delta, delta) + delta_self
            gains = -change
            evaluations += int(units.size)
            for offset, unit in enumerate(units.tolist()):
                action_id = transition_id * field.units + int(unit)
                gain = float(gains[offset])
                candidates.append((
                    gain / cost, gain, -cost, -action_id, int(unit), source,
                    destination, delta[offset],
                ))
        if not candidates:
            break
        chosen = max(candidates, key=lambda value: value[:4])
        _, gain, neg_cost, _, unit, source, destination, delta = chosen
        cost = -int(neg_cost)
        states[unit] = destination
        u += delta
        pages += cost
        damage -= float(gain)
        actions.append(FieldAction(unit, source, destination, cost, pages, float(gain)))
        path_states.append(states.copy())
        path_pages.append(pages)
        path_damage.append(field.damage(states))
    snapshots: dict[int, FieldBudgetSnapshot] = {}
    for budget in requested:
        eligible = [index for index, value in enumerate(path_pages) if value <= budget]
        best = min(eligible, key=lambda index: (path_damage[index], path_pages[index], index))
        snapshots[budget] = FieldBudgetSnapshot(
            states=path_states[best].copy(), pages=int(path_pages[best]),
            damage=float(path_damage[best]), prefix=int(best),
        )
    return FieldForwardTrace(tuple(actions), snapshots, evaluations)


@dataclass(frozen=True)
class FieldCoordinateTrace:
    states: np.ndarray
    pages: int
    damage: float
    penalized_objective: float
    sweeps: int
    accepted_moves: int
    interaction_dot_macs: int


def field_coordinate_descent(
    field: InteractionField,
    seed: Sequence[int] | np.ndarray,
    budget_pages: int,
    *,
    page_price: float = 0.0,
    max_sweeps: int = 8,
    tolerance: float = 1e-12,
) -> FieldCoordinateTrace:
    """Coordinate descent using two signed row dots per unit and four states."""
    budget = int(budget_pages)
    states = _validate_states(seed, field.units)
    pages = field.pages(states)
    if pages > budget or budget < 0 or budget > 3 * field.units:
        raise ValueError("seed exceeds the page budget")
    rho = np.asarray(field.rho)
    coefficients = np.asarray(field.coefficients)
    l4 = np.asarray(field.factor.l4, np.float64)
    l2 = np.asarray(field.factor.l2, np.float64)
    q = np.asarray(field.local_damage)
    indices = np.arange(field.units)
    u = rho[indices, states].sum(axis=0)
    damage = field.damage(states)
    objective = damage + float(page_price) * pages
    accepted = 0
    completed = 0
    for sweep in range(int(max_sweeps)):
        changed = False
        for unit in range(field.units):
            source = int(states[unit])
            u_minus = u - rho[unit, source]
            base_pages = pages - int(STATE_PAGE_COSTS[source])
            allowed = base_pages + STATE_PAGE_COSTS <= budget
            basis_dots = np.asarray((u_minus @ l4[unit], u_minus @ l2[unit]))
            scores = (
                2.0 * (coefficients[unit] @ basis_dots)
                + q[unit]
                + float(page_price) * STATE_PAGE_COSTS
            )
            scores = np.where(allowed, scores, np.inf)
            destination = int(np.argmin(scores))
            if destination == source:
                continue
            candidate_pages = base_pages + int(STATE_PAGE_COSTS[destination])
            candidate_u = u_minus + rho[unit, destination]
            candidate_damage = float(
                candidate_u @ candidate_u
                + np.asarray(field.self_residual)[indices, np.where(indices == unit, destination, states)].sum()
            )
            candidate_objective = candidate_damage + float(page_price) * candidate_pages
            if candidate_objective < objective - float(tolerance):
                states[unit] = destination
                pages = candidate_pages
                u = candidate_u
                damage = field.damage(states)
                objective = damage + float(page_price) * pages
                accepted += 1
                changed = True
        completed = sweep + 1
        if not changed:
            break
    return FieldCoordinateTrace(
        states.copy(), pages, float(damage), float(objective), completed, accepted,
        2 * field.units * field.rank * completed,
    )


@dataclass(frozen=True)
class FieldLocalSearchTrace:
    states: np.ndarray
    pages: int
    damage: float
    passes: int
    accepted_bundles: int


def field_local_search(
    field: InteractionField,
    seed: Sequence[int] | np.ndarray,
    budget_pages: int,
    *,
    shortlist_size: int = 8,
    max_swap_units: int = 3,
    max_passes: int = 16,
    page_balanced: bool = False,
    tolerance: float = 1e-12,
) -> FieldLocalSearchTrace:
    """Bounded one/two/three-unit repair scored only by the compressed field."""
    states = _validate_states(seed, field.units)
    budget = int(budget_pages)
    pages = field.pages(states)
    if pages > budget:
        raise ValueError("seed exceeds page budget")
    rho = np.asarray(field.rho)
    self_term = np.asarray(field.self_residual)
    indices = np.arange(field.units)
    u = rho[indices, states].sum(axis=0)
    damage = field.damage(states)
    accepted = 0
    completed = 0
    for pass_index in range(int(max_passes)):
        moves: list[tuple[UnitStateMove, np.ndarray, float]] = []
        for unit, source in enumerate(states.tolist()):
            for destination in range(4):
                if destination == source:
                    continue
                page_delta = int(STATE_PAGE_COSTS[destination] - STATE_PAGE_COSTS[source])
                delta = rho[unit, destination] - rho[unit, source]
                change = (
                    2.0 * float(u @ delta) + float(delta @ delta)
                    + float(self_term[unit, destination] - self_term[unit, source])
                )
                moves.append((UnitStateMove(unit, source, destination, page_delta), delta, -change))
        shortlisted: list[tuple[UnitStateMove, np.ndarray, float]] = []
        for delta_pages in sorted({move[0].page_delta for move in moves}):
            group = [move for move in moves if move[0].page_delta == delta_pages]
            group.sort(key=lambda item: (-item[2], item[0].unit, item[0].to_state))
            shortlisted.extend(group[:int(shortlist_size)])
        if not shortlisted:
            break
        pairwise = np.stack([item[1] for item in shortlisted]) @ np.stack(
            [item[1] for item in shortlisted]
        ).T
        best_gain = float("-inf")
        best_moves: tuple[int, ...] | None = None
        best_key: tuple[UnitStateMove, ...] | None = None
        for size in range(1, int(max_swap_units) + 1):
            for chosen in combinations(range(len(shortlisted)), size):
                state_moves = tuple(sorted(shortlisted[index][0] for index in chosen))
                if len({move.unit for move in state_moves}) != size:
                    continue
                net_pages = sum(move.page_delta for move in state_moves)
                if page_balanced and net_pages != 0:
                    continue
                next_pages = pages + net_pages
                if next_pages < 0 or next_pages > budget:
                    continue
                gain = sum(shortlisted[index][2] for index in chosen)
                gain -= 2.0 * sum(pairwise[left, right] for left, right in combinations(chosen, 2))
                if (
                    gain > best_gain + tolerance
                    or (abs(gain - best_gain) <= tolerance and (best_key is None or state_moves < best_key))
                ):
                    best_gain = float(gain)
                    best_moves = chosen
                    best_key = state_moves
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
        u = rho[indices, states].sum(axis=0)
        damage = proposal_damage
        accepted += 1
        completed = pass_index + 1
    return FieldLocalSearchTrace(states.copy(), pages, float(damage), completed, accepted)


def _simplex_projection(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, np.float64)
    ordered = np.sort(value)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    valid = ordered - cumulative / (np.arange(len(value)) + 1) > 0.0
    rho = int(np.flatnonzero(valid)[-1])
    threshold = cumulative[rho] / (rho + 1)
    return np.maximum(value - threshold, 0.0)


def _project_relaxed(values: np.ndarray, budget_pages: float) -> np.ndarray:
    raw = np.asarray(values, np.float64)
    pages = STATE_PAGE_COSTS.astype(np.float64)

    def project_at(dual: float) -> np.ndarray:
        return np.stack([_simplex_projection(row - dual * pages) for row in raw])

    unconstrained = project_at(0.0)
    if float(np.sum(unconstrained @ pages)) <= float(budget_pages) + 1e-10:
        return unconstrained
    low, high = 0.0, 1.0
    while float(np.sum(project_at(high) @ pages)) > float(budget_pages):
        high *= 2.0
    for _ in range(80):
        middle = 0.5 * (low + high)
        if float(np.sum(project_at(middle) @ pages)) > float(budget_pages):
            low = middle
        else:
            high = middle
    return project_at(high)


@dataclass(frozen=True)
class RelaxedFieldTrace:
    probabilities: np.ndarray
    objective: float
    expected_pages: float
    iterations: int
    converged: bool


def relaxed_field_descent(
    field: InteractionField,
    budget_pages: float,
    *,
    max_iterations: int = 1000,
    tolerance: float = 1e-10,
    step_size: float | None = None,
) -> RelaxedFieldTrace:
    """Projected-gradient lower bound for the compressed-field objective."""
    budget = float(budget_pages)
    if budget < 0.0 or budget > 3.0 * field.units:
        raise ValueError("relaxed page budget lies outside [0,3*units]")
    rho = np.asarray(field.rho, np.float64)
    linear = np.asarray(field.self_residual, np.float64)
    probabilities = np.zeros((field.units, 4), np.float64)
    probabilities[:, STATE_00] = 1.0
    if step_size is None:
        frobenius_squared = float(np.einsum("usr,usr->", rho, rho, optimize=True))
        step = 1.0 / max(2.0 * frobenius_squared, 1.0)
    else:
        step = float(step_size)
    converged = False
    iterations = 0
    for iteration in range(int(max_iterations)):
        u = np.einsum("us,usr->r", probabilities, rho, optimize=True)
        gradient = 2.0 * np.einsum("usr,r->us", rho, u, optimize=True) + linear
        updated = _project_relaxed(probabilities - step * gradient, budget)
        difference = float(np.max(np.abs(updated - probabilities), initial=0.0))
        probabilities = updated
        iterations = iteration + 1
        if difference <= float(tolerance):
            converged = True
            break
    u = np.einsum("us,usr->r", probabilities, rho, optimize=True)
    objective = float(u @ u + np.sum(probabilities * linear))
    expected_pages = float(np.sum(probabilities * STATE_PAGE_COSTS[None, :]))
    return RelaxedFieldTrace(probabilities, objective, expected_pages, iterations, converged)


def round_relaxed_pages(probabilities: np.ndarray, budget_pages: int) -> np.ndarray:
    """Deterministically round categorical probabilities with exact page DP."""
    values = np.asarray(probabilities, np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or np.any(values < 0.0):
        raise ValueError("probabilities must have shape [unit,4] and be nonnegative")
    budget = int(budget_pages)
    units = values.shape[0]
    if budget < 0 or budget > 3 * units:
        raise ValueError("page budget lies outside [0,3*units]")
    score = np.log(np.maximum(values, 1e-30))
    dynamic = np.full((units + 1, budget + 1), -np.inf, np.float64)
    choice = np.full((units, budget + 1), -1, np.int8)
    dynamic[0, 0] = 0.0
    for unit in range(units):
        for used in range(budget + 1):
            if not np.isfinite(dynamic[unit, used]):
                continue
            for state, cost in enumerate(STATE_PAGE_COSTS.tolist()):
                target = used + int(cost)
                if target > budget:
                    continue
                candidate = dynamic[unit, used] + score[unit, state]
                current = dynamic[unit + 1, target]
                if candidate > current + 1e-15 or (
                    abs(candidate - current) <= 1e-15
                    and (choice[unit, target] < 0 or state < choice[unit, target])
                ):
                    dynamic[unit + 1, target] = candidate
                    choice[unit, target] = state
    final_pages = int(np.argmax(dynamic[units]))
    states = np.empty(units, np.int64)
    pages = final_pages
    for unit in range(units - 1, -1, -1):
        state = int(choice[unit, pages])
        if state < 0:
            raise RuntimeError("relaxed rounding backtrack failed")
        states[unit] = state
        pages -= int(STATE_PAGE_COSTS[state])
    return states


def walsh_hadamard(rank: int) -> np.ndarray:
    """Return the deterministic normalized Walsh-Hadamard matrix."""
    size = int(rank)
    if size < 1 or size & (size - 1):
        raise ValueError("Hadamard rank must be a positive power of two")
    matrix = np.ones((1, 1), np.float64)
    while matrix.shape[0] < size:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix / np.sqrt(size)




def encode_interaction_factor(
    factor: JointInteractionFactor,
    encoding: str,
    *,
    hadamard_rotate: bool = False,
) -> EncodedInteractionFactor:
    """Physically pack signed row codes and their stored FP16 scales."""
    joint = np.asarray(factor.joint(), np.float64)
    if hadamard_rotate:
        joint = joint @ walsh_hadamard(factor.rank)
    if encoding not in {"int8_per_row", "int4_per_row"}:
        raise ValueError("encoding must be int8_per_row or int4_per_row")
    maximum = 127.0 if encoding == "int8_per_row" else 7.0
    raw_scale = np.max(np.abs(joint), axis=1) / maximum
    minimum = np.float64(np.nextafter(np.float16(0.0), np.float16(1.0)))
    target_scale = np.where(raw_scale > 0.0, np.maximum(raw_scale, minimum), 1.0)
    row_scales = target_scale.astype(np.float16)
    rounds_down = row_scales.astype(np.float64) < target_scale
    row_scales[rounds_down] = np.nextafter(
        row_scales[rounds_down], np.float16(np.inf),
    )
    if not np.all(np.isfinite(row_scales)):
        raise ValueError("interaction row scale overflowed FP16")
    decoded_scale = row_scales.astype(np.float64)[:, None]
    quantized = np.clip(
        np.rint(joint / decoded_scale), -maximum, maximum,
    ).astype(np.int8)
    flattened = quantized.reshape(-1)
    if encoding == "int8_per_row":
        packed = flattened.view(np.uint8).copy()
    else:
        unsigned = (flattened.astype(np.int16) & 0x0F).astype(np.uint8)
        if unsigned.size & 1:
            unsigned = np.concatenate((unsigned, np.zeros(1, np.uint8)))
        packed = (unsigned[0::2] | (unsigned[1::2] << 4)).astype(np.uint8)
    suffix = "_hadamard" if hadamard_rotate else ""
    return EncodedInteractionFactor(
        packed_codes=packed,
        row_scales=row_scales,
        units=factor.units,
        rank=factor.rank,
        method=factor.method,
        tail_rank=factor.tail_rank,
        exact_rank=factor.exact_rank,
        encoding=encoding + suffix,
    )
def quantize_interaction_factor(
    factor: JointInteractionFactor,
    encoding: str,
    *,
    hadamard_rotate: bool = False,
) -> JointInteractionFactor:
    """Quantize row signatures with physical packed-byte accounting."""
    return encode_interaction_factor(
        factor, encoding, hadamard_rotate=hadamard_rotate,
    ).decode()


def make_factor_self_safe(
    factor: JointInteractionFactor,
    metadata: UnitScoreMetadata,
    *,
    tolerance: float = 1e-10,
) -> tuple[JointInteractionFactor, float]:
    """Globally shrink a decoded factor until every local 2x2 residual is PSD."""
    if factor.units != metadata.units:
        raise ValueError("factor and A/B/C units disagree")
    a = np.asarray(metadata.a, np.float64)
    b = np.asarray(metadata.b, np.float64)
    c = np.asarray(metadata.c, np.float64)
    l4 = np.asarray(factor.l4, np.float64)
    l2 = np.asarray(factor.l2, np.float64)

    def safe(scale: float) -> bool:
        for unit in range(factor.units):
            exact = np.asarray(((a[unit], b[unit]), (b[unit], c[unit])), np.float64)
            rows = np.stack((l4[unit], l2[unit])) * float(scale)
            residual = exact - rows @ rows.T
            if float(np.min(np.linalg.eigvalsh(0.5 * (residual + residual.T)))) < -tolerance:
                return False
        return True

    if safe(1.0):
        return factor, 1.0
    low, high = 0.0, 1.0
    for _ in range(80):
        middle = 0.5 * (low + high)
        if safe(middle):
            low = middle
        else:
            high = middle
    scale = low
    return JointInteractionFactor(
        l4=np.asarray(l4 * scale, np.float32), l2=np.asarray(l2 * scale, np.float32),
        method=factor.method, tail_rank=factor.tail_rank, exact_rank=factor.exact_rank,
        encoding=factor.encoding + "_self_safe", storage_bytes=factor.storage_bytes,
    ), float(scale)



def make_encoded_factor_self_safe(
    encoded: EncodedInteractionFactor,
    metadata: UnitScoreMetadata,
    *,
    tolerance: float = 1e-10,
) -> tuple[EncodedInteractionFactor, float]:
    """Store a conservative FP32 shrink that keeps every local residual PSD."""
    _, relative = make_factor_self_safe(
        encoded.decode(), metadata, tolerance=tolerance,
    )
    target = float(encoded.global_scale) * float(relative)
    stored = np.float32(target)
    if float(stored) > target:
        stored = np.nextafter(stored, np.float32(0.0))
    for _ in range(16):
        candidate = EncodedInteractionFactor(
            packed_codes=np.asarray(encoded.packed_codes).copy(),
            row_scales=np.asarray(encoded.row_scales).copy(),
            units=encoded.units,
            rank=encoded.rank,
            method=encoded.method,
            tail_rank=encoded.tail_rank,
            exact_rank=encoded.exact_rank,
            encoding=encoded.encoding,
            global_scale=stored,
        )
        _, post = make_factor_self_safe(
            candidate.decode(), metadata, tolerance=tolerance,
        )
        if post == 1.0:
            return candidate, float(stored)
        stored = np.nextafter(stored, np.float32(0.0))
    raise RuntimeError("could not store a self-safe global interaction shrink")
