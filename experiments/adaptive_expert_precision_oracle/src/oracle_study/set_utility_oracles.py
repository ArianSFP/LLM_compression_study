"""Deterministic reference oracles for coherent-unit set utility studies.

These routines consume the exact decoded ``00/G/D/GD`` unit contributions
from :mod:`oracle_study.neuron_selector`.  They do not alter the locked codec
or refit any correction coefficient.  The implementations deliberately
favour explicit, testable semantics over throughput; accelerated experiment
runners can use them as correctness references.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, combinations_with_replacement, product
from time import perf_counter
from typing import Hashable, Mapping, Sequence

import numpy as np
import torch

from .neuron_selector import FactorizedUnitOutputs
from .sparse_streaming import qenergy


__all__ = [
    "STATE_00",
    "STATE_G",
    "STATE_D",
    "STATE_GD",
    "STATE_NAMES",
    "STATE_PAGE_COSTS",
    "HybridForwardAction",
    "HybridForwardTrace",
    "HybridBudgetSnapshot",
    "HybridBaseGramTrace",
    "UnitStateMove",
    "LocalSearchStep",
    "SeedLocalSearchTrace",
    "MonotoneLocalSearchResult",
    "ExpertTemplateBank",
    "NestedTemplateBank",
    "ChargedTemplateCandidates",
    "hybrid_state_pages",
    "hybrid_state_output",
    "exact_hybrid_unit_fixed_greedy",
    "exact_hybrid_unit_base_gram_greedy",
    "exact_monotone_unit_local_search",
    "utility_weighted_jaccard_distance",
    "build_nested_template_bank",
    "charged_template_candidates",
    "best_charged_template_candidates",
    "charged_top1_template_candidates",
    "charged_top2_union_candidates",
]


STATE_00 = 0
STATE_G = 1
STATE_D = 2
STATE_GD = 3
STATE_NAMES = ("00", "G", "D", "GD")
STATE_PAGE_COSTS = np.asarray((0, 2, 1, 3), dtype=np.int64)

_FORWARD_TRANSITIONS = (
    (STATE_00, STATE_G, 2),
    (STATE_00, STATE_D, 1),
    (STATE_00, STATE_GD, 3),
    (STATE_G, STATE_GD, 1),
    (STATE_D, STATE_GD, 2),
)


def _state_contributions(outputs: FactorizedUnitOutputs) -> np.ndarray:
    """Return exact contributions with shape ``[state, unit, output]``."""
    return np.stack((outputs.y00, outputs.y10, outputs.y01, outputs.y11)).astype(
        np.float64, copy=False,
    )


def _validate_states(states: Sequence[int] | np.ndarray, units: int) -> np.ndarray:
    value = np.asarray(states, dtype=np.int64).reshape(-1)
    if value.shape != (int(units),) or np.any((value < STATE_00) | (value > STATE_GD)):
        raise ValueError("states must contain one value in {00, G, D, GD} per unit")
    return value.copy()


def hybrid_state_pages(states: Sequence[int] | np.ndarray) -> int:
    """Return physical 512-byte pages charged by a hybrid unit state vector."""
    value = np.asarray(states, dtype=np.int64).reshape(-1)
    if np.any((value < STATE_00) | (value > STATE_GD)):
        raise ValueError("hybrid states must lie in [0, 3]")
    return int(STATE_PAGE_COSTS[value].sum())


def hybrid_state_output(
    outputs: FactorizedUnitOutputs, states: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Evaluate the exact complete-expert output for hybrid unit states."""
    chosen = _validate_states(states, outputs.units)
    contributions = _state_contributions(outputs)
    return np.asarray(
        contributions[chosen, np.arange(outputs.units)].sum(axis=0), dtype=np.float64,
    )


def _validate_metric_arguments(
    output_size: int,
    metric: np.ndarray | None,
    proxy: np.ndarray | None,
    beta: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if beta < 0.0:
        raise ValueError("beta must be nonnegative")
    dense = None if metric is None else np.asarray(metric, dtype=np.float64)
    if dense is not None and dense.shape != (output_size, output_size):
        raise ValueError("metric shape does not match output width")
    future = None if proxy is None else np.asarray(proxy, dtype=np.float64)
    if future is not None:
        if future.ndim == 1:
            future = future[:, None]
        if future.ndim != 2 or future.shape[0] != output_size:
            raise ValueError("proxy must have shape [output, proxy_width]")
    if beta != 0.0 and future is None:
        raise ValueError("nonzero beta requires a proxy")
    return dense, future


def _metric_row_inner(
    rows: np.ndarray,
    vector: np.ndarray,
    *,
    metric: np.ndarray | None,
    proxy: np.ndarray | None,
    beta: float,
) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float64)
    target = np.asarray(vector, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or values.shape[1] != target.size:
        raise ValueError("row/vector widths disagree")
    result = values @ target if metric is None else (values @ metric) @ target
    if proxy is not None and beta:
        result = result + float(beta) * ((values @ proxy) @ (target @ proxy))
    return np.asarray(result, dtype=np.float64)


def _metric_pairwise(
    rows: np.ndarray,
    *,
    metric: np.ndarray | None,
    proxy: np.ndarray | None,
    beta: float,
) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float64)
    result = values @ values.T if metric is None else (values @ metric) @ values.T
    if proxy is not None and beta:
        projected = values @ proxy
        result = result + float(beta) * (projected @ projected.T)
    return np.asarray(result, dtype=np.float64)


def _metric_row_norms(
    rows: np.ndarray,
    *,
    metric: np.ndarray | None,
    proxy: np.ndarray | None,
    beta: float,
) -> np.ndarray:
    """Return qmetric row energies without materialising a row Gram."""
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("rows must be a matrix")
    if metric is None:
        result = np.einsum("ij,ij->i", values, values, optimize=True)
    else:
        result = np.einsum("ij,ij->i", values @ metric, values, optimize=True)
    if proxy is not None and beta:
        projected = values @ proxy
        result = result + float(beta) * np.einsum(
            "ij,ij->i", projected, projected, optimize=True,
        )
    return np.asarray(result, dtype=np.float64)


def _energy(
    residual: np.ndarray,
    *,
    metric: np.ndarray | None,
    proxy: np.ndarray | None,
    beta: float,
) -> float:
    return qenergy(residual, metric=metric, proxy=proxy, beta=beta)


@dataclass(frozen=True)
class HybridForwardAction:
    """One fixed-coefficient state transition on the hybrid greedy path."""

    action_id: int
    unit: int
    from_state: int
    to_state: int
    incremental_pages: int
    cumulative_pages: int
    marginal_gain: float


@dataclass(frozen=True)
class HybridForwardTrace:
    """Terminal state and full transition trace of hybrid forward greedy."""

    actions: tuple[HybridForwardAction, ...]
    states: np.ndarray
    output: np.ndarray
    residual_energy: float
    pages: int


@dataclass(frozen=True)
class HybridBudgetSnapshot:
    """Best profitable prefix of one shared hybrid path for a page budget."""

    requested_pages: int
    best_prefix: int
    pages: int
    states: np.ndarray
    output: np.ndarray
    residual_energy: float
    cumulative_gain: float
    order: tuple[int, ...]
    profitable: bool


@dataclass(frozen=True)
class HybridBaseGramTrace:
    """GPU-base-Gram hybrid path and exact per-budget prefix snapshots."""

    actions: tuple[HybridForwardAction, ...]
    order: np.ndarray
    gains: np.ndarray
    incremental_pages: np.ndarray
    cumulative_pages: np.ndarray
    path_residual_energies: np.ndarray
    snapshots: Mapping[int, HybridBudgetSnapshot]
    terminal_states: np.ndarray
    terminal_output: np.ndarray
    terminal_residual_energy: float
    terminal_pages: int
    gram_device: str
    gram_dtype: str
    primitive_gram_shape: tuple[int, int]
    primitive_gram_device_bytes: int
    candidate_evaluations: int
    gram_build_seconds: float
    gram_transfer_seconds: float
    greedy_seconds: float
    snapshot_seconds: float
    total_seconds: float


def exact_hybrid_unit_fixed_greedy(
    outputs: FactorizedUnitOutputs,
    page_budget: int,
    *,
    metric: np.ndarray | None = None,
    proxy: np.ndarray | None = None,
    beta: float = 0.0,
) -> HybridForwardTrace:
    """Run exact marginal-per-page greedy over ``00/G/D/GD`` states.

    The direct three-page ``00 -> GD`` coherent action is eligible alongside
    the one- and two-page partial actions.  Complementary ``G -> GD`` and
    ``D -> GD`` actions remain eligible after a partial choice.  Marginals are
    evaluated against the current residual in the exact requested metric;
    coefficients are fixed at one and are never least-squares refit.

    Ties are resolved by higher efficiency, higher raw gain, lower page cost,
    then lower transition-major action ID.  Negative marginals are retained
    so a full ``3 * units`` budget always reaches the exact GD endpoint.
    """
    budget = int(page_budget)
    maximum = 3 * outputs.units
    if budget < 0 or budget > maximum:
        raise ValueError(f"page_budget must lie within [0, {maximum}]")
    contributions = _state_contributions(outputs)
    dense, future = _validate_metric_arguments(outputs.output_size, metric, proxy, beta)
    target = np.asarray(outputs.target_output, dtype=np.float64)
    states = np.full(outputs.units, STATE_00, dtype=np.int64)
    current = np.asarray(outputs.base_output, dtype=np.float64).copy()
    residual = target - current
    pages = 0
    trace: list[HybridForwardAction] = []

    while pages < budget:
        candidates: list[tuple[int, int, int, int, int, np.ndarray]] = []
        for transition, (source, destination, cost) in enumerate(_FORWARD_TRANSITIONS):
            for unit in np.flatnonzero(states == source).tolist():
                if pages + cost <= budget:
                    action_id = transition * outputs.units + int(unit)
                    delta = contributions[destination, unit] - contributions[source, unit]
                    candidates.append((action_id, unit, source, destination, cost, delta))
        if not candidates:
            break
        vectors = np.stack([candidate[-1] for candidate in candidates])
        dots = _metric_row_inner(
            vectors, residual, metric=dense, proxy=future, beta=beta,
        )
        norms = _metric_row_norms(
            vectors, metric=dense, proxy=future, beta=beta,
        )
        gains = 2.0 * dots - norms
        best_index = max(
            range(len(candidates)),
            key=lambda index: (
                float(gains[index]) / candidates[index][4],
                float(gains[index]),
                -candidates[index][4],
                -candidates[index][0],
            ),
        )
        action_id, unit, source, destination, cost, delta = candidates[best_index]
        gain = float(gains[best_index])
        states[unit] = destination
        current += delta
        residual -= delta
        pages += cost
        trace.append(HybridForwardAction(
            action_id=action_id,
            unit=unit,
            from_state=source,
            to_state=destination,
            incremental_pages=cost,
            cumulative_pages=pages,
            marginal_gain=gain,
        ))

    return HybridForwardTrace(
        actions=tuple(trace),
        states=states,
        output=current,
        residual_energy=_energy(
            residual, metric=dense, proxy=future, beta=beta,
        ),
        pages=pages,
    )


def _primitive_transition_vectors(outputs: FactorizedUnitOutputs) -> np.ndarray:
    """Return the algebraically sufficient four-transition basis."""
    return np.stack((
        outputs.y10 - outputs.y00,
        outputs.y01 - outputs.y00,
        outputs.y11 - outputs.y10,
        outputs.y11 - outputs.y01,
    )).astype(np.float64, copy=False)


def _states_at_hybrid_prefix(
    actions: Sequence[HybridForwardAction], units: int, prefix: int,
) -> np.ndarray:
    states = np.full(int(units), STATE_00, dtype=np.int64)
    for action in actions[:int(prefix)]:
        if states[action.unit] != action.from_state:
            raise RuntimeError("hybrid path violates nested state eligibility")
        states[action.unit] = action.to_state
    return states


def exact_hybrid_unit_base_gram_greedy(
    outputs: FactorizedUnitOutputs,
    page_budgets: Sequence[int] | int,
    *,
    metric: np.ndarray | None = None,
    proxy: np.ndarray | None = None,
    beta: float = 0.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> HybridBaseGramTrace:
    """Run one shared hybrid path from a single GPU primitive-action Gram.

    Four fixed primitive vectors per unit are sufficient: 00->G, 00->D,
    G->GD and D->GD.  The coherent 00->GD candidate is scored and applied as
    the algebraic sum of the first and third vectors.  One
    [4 * unit, 4 * unit] qmetric Gram is built on device and transferred once;
    the deterministic CPU loop then updates primitive correlations by
    subtracting selected Gram columns.

    All requested budgets share the path grown for their maximum.  A snapshot
    is the lowest-direct-qenergy prefix whose cumulative physical pages do not
    exceed that budget, with the zero-action prefix available whenever every
    feasible correction prefix is unprofitable.  The terminal fields retain
    the actual end of the shared path even when a snapshot selects an earlier
    profitable prefix.
    """
    total_started = perf_counter()
    if isinstance(page_budgets, (int, np.integer)):
        requested = (int(page_budgets),)
    else:
        requested = tuple(sorted({int(value) for value in page_budgets}))
    if not requested:
        raise ValueError("at least one page budget is required")
    maximum_pages = 3 * outputs.units
    if any(value < 0 or value > maximum_pages for value in requested):
        raise ValueError(f"page budgets must lie within [0, {maximum_pages}]")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("base-Gram dtype must be torch.float32 or torch.float64")
    dense, future = _validate_metric_arguments(
        outputs.output_size, metric, proxy, beta,
    )
    work_device = torch.device(
        "cuda" if device is None and torch.cuda.is_available()
        else "cpu" if device is None else device
    )
    primitive_cpu = _primitive_transition_vectors(outputs)
    units, output_size = outputs.units, outputs.output_size
    gram_build_started = perf_counter()
    with torch.no_grad():
        primitive = torch.as_tensor(
            primitive_cpu.reshape(4 * units, output_size),
            dtype=dtype,
            device=work_device,
        )
        if dense is None:
            gram = primitive @ primitive.T
        else:
            dense_tensor = torch.as_tensor(dense, dtype=dtype, device=work_device)
            gram = (primitive @ dense_tensor) @ primitive.T
        if future is not None and beta:
            proxy_tensor = torch.as_tensor(future, dtype=dtype, device=work_device)
            projected = primitive @ proxy_tensor
            gram = gram + float(beta) * (projected @ projected.T)
        if work_device.type == "cuda":
            torch.cuda.synchronize(work_device)
        gram_build_seconds = perf_counter() - gram_build_started
        gram_device_bytes = int(gram.numel() * gram.element_size())
        # Exactly one large device-to-host transfer.  Conversion to float64
        # happens after arrival so FP32 launch mode does not double traffic.
        gram_transfer_started = perf_counter()
        gram_cpu = gram.detach().cpu().numpy().astype(np.float64, copy=False)
        gram_transfer_seconds = perf_counter() - gram_transfer_started

    # The target residual is the sum of all direct 00->GD vectors, represented
    # in this basis by one coefficient on the 00->G and G->GD blocks.
    greedy_started = perf_counter()
    target_coefficients = np.zeros(4 * units, dtype=np.float64)
    target_coefficients[:units] = 1.0
    target_coefficients[2 * units:3 * units] = 1.0
    correlation = gram_cpu @ target_coefficients
    states = np.full(units, STATE_00, dtype=np.int64)
    residual = np.asarray(outputs.target_output - outputs.base_output, dtype=np.float64)
    initial_energy = _energy(
        residual, metric=dense, proxy=future, beta=beta,
    )
    path_energies: list[float] = [initial_energy]
    actions: list[HybridForwardAction] = []
    cumulative_pages: list[int] = []
    pages = 0
    candidate_evaluations = 0
    maximum_requested = max(requested)
    flat_primitive = primitive_cpu.reshape(4 * units, output_size)

    while pages < maximum_requested:
        candidates: list[
            tuple[float, float, int, int, int, int, int, int, tuple[int, ...]]
        ] = []
        for transition, (source, destination, cost) in enumerate(_FORWARD_TRANSITIONS):
            if pages + cost > maximum_requested:
                continue
            for raw_unit in np.flatnonzero(states == source).tolist():
                unit = int(raw_unit)
                action_id = transition * units + unit
                if transition == 0:
                    primitive_ids = (unit,)
                elif transition == 1:
                    primitive_ids = (units + unit,)
                elif transition == 2:
                    primitive_ids = (unit, 2 * units + unit)
                elif transition == 3:
                    primitive_ids = (2 * units + unit,)
                else:
                    primitive_ids = (3 * units + unit,)
                action_correlation = float(sum(correlation[index] for index in primitive_ids))
                action_norm = float(sum(
                    gram_cpu[left, right]
                    for left in primitive_ids for right in primitive_ids
                ))
                gain = 2.0 * action_correlation - action_norm
                candidates.append((
                    gain / cost, gain, -cost, -action_id,
                    action_id, unit, source, destination, primitive_ids,
                ))
        if not candidates:
            break
        candidate_evaluations += len(candidates)
        chosen = max(candidates, key=lambda value: value[:4])
        _, gain, _, _, action_id, unit, source, destination, primitive_ids = chosen
        cost = int(STATE_PAGE_COSTS[destination] - STATE_PAGE_COSTS[source])
        delta = flat_primitive[list(primitive_ids)].sum(axis=0)
        residual -= delta
        correlation -= gram_cpu[:, list(primitive_ids)].sum(axis=1)
        states[unit] = destination
        pages += cost
        actions.append(HybridForwardAction(
            action_id=action_id,
            unit=unit,
            from_state=source,
            to_state=destination,
            incremental_pages=cost,
            cumulative_pages=pages,
            marginal_gain=float(gain),
        ))
        cumulative_pages.append(pages)
        path_energies.append(_energy(
            residual, metric=dense, proxy=future, beta=beta,
        ))

    terminal_states = states.copy()
    terminal_output = hybrid_state_output(outputs, terminal_states)
    terminal_energy = _energy(
        outputs.target_output - terminal_output,
        metric=dense, proxy=future, beta=beta,
    )
    if pages == maximum_pages and np.all(terminal_states == STATE_GD):
        # The exact mixed-state evaluator is authoritative at the complete
        # endpoint and removes harmless telescoping roundoff from path replay.
        path_energies[-1] = terminal_energy

    greedy_seconds = perf_counter() - greedy_started
    snapshot_started = perf_counter()
    action_tuple = tuple(actions)
    prefix_pages = np.asarray((0, *cumulative_pages), dtype=np.int64)
    path_energy_array = np.asarray(path_energies, dtype=np.float64)
    snapshots: dict[int, HybridBudgetSnapshot] = {}
    for budget in requested:
        feasible = np.flatnonzero(prefix_pages <= budget)
        best_prefix = min(
            feasible.tolist(),
            key=lambda prefix: (float(path_energy_array[prefix]), int(prefix)),
        )
        snapshot_states = _states_at_hybrid_prefix(
            action_tuple, units, best_prefix,
        )
        snapshot_output = hybrid_state_output(outputs, snapshot_states)
        snapshot_energy = _energy(
            outputs.target_output - snapshot_output,
            metric=dense, proxy=future, beta=beta,
        )
        # Direct output evaluation is also used for the profitability decision;
        # the base prefix therefore wins if numerical Gram error invents gain.
        if snapshot_energy >= initial_energy and best_prefix != 0:
            best_prefix = 0
            snapshot_states = np.full(units, STATE_00, dtype=np.int64)
            snapshot_output = np.asarray(outputs.base_output, dtype=np.float64)
            snapshot_energy = initial_energy
        gain = float(initial_energy - snapshot_energy)
        snapshots[budget] = HybridBudgetSnapshot(
            requested_pages=budget,
            best_prefix=best_prefix,
            pages=int(prefix_pages[best_prefix]),
            states=snapshot_states,
            output=snapshot_output,
            residual_energy=float(snapshot_energy),
            cumulative_gain=gain,
            order=tuple(action.action_id for action in action_tuple[:best_prefix]),
            profitable=bool(gain > 0.0),
        )

    snapshot_seconds = perf_counter() - snapshot_started
    total_seconds = perf_counter() - total_started
    return HybridBaseGramTrace(
        actions=action_tuple,
        order=np.asarray([action.action_id for action in actions], dtype=np.int64),
        gains=np.asarray([action.marginal_gain for action in actions], dtype=np.float64),
        incremental_pages=np.asarray(
            [action.incremental_pages for action in actions], dtype=np.int64,
        ),
        cumulative_pages=np.asarray(cumulative_pages, dtype=np.int64),
        path_residual_energies=path_energy_array,
        snapshots=snapshots,
        terminal_states=terminal_states,
        terminal_output=terminal_output,
        terminal_residual_energy=float(terminal_energy),
        terminal_pages=pages,
        gram_device=str(work_device),
        gram_dtype=str(dtype).removeprefix("torch."),
        primitive_gram_shape=tuple(int(value) for value in gram_cpu.shape),
        primitive_gram_device_bytes=gram_device_bytes,
        candidate_evaluations=candidate_evaluations,
        gram_build_seconds=float(gram_build_seconds),
        gram_transfer_seconds=float(gram_transfer_seconds),
        greedy_seconds=float(greedy_seconds),
        snapshot_seconds=float(snapshot_seconds),
        total_seconds=float(total_seconds),
    )


@dataclass(frozen=True, order=True)
class UnitStateMove:
    """One unit reassignment considered by exact local search."""

    unit: int
    from_state: int
    to_state: int
    page_delta: int


@dataclass(frozen=True)
class LocalSearchStep:
    moves: tuple[UnitStateMove, ...]
    gain: float
    energy_before: float
    energy_after: float
    pages_after: int


@dataclass(frozen=True)
class SeedLocalSearchTrace:
    seed_name: str
    seed_states: np.ndarray
    seed_energy: float
    final_states: np.ndarray
    final_energy: float
    final_pages: int
    steps: tuple[LocalSearchStep, ...]


@dataclass(frozen=True)
class MonotoneLocalSearchResult:
    """Best monotone result across all supplied coherent/partial seeds."""

    seed_name: str
    states: np.ndarray
    output: np.ndarray
    residual_energy: float
    pages: int
    seed_traces: Mapping[str, SeedLocalSearchTrace]


@dataclass(frozen=True)
class _ScoredMove:
    move: UnitStateMove
    vector: np.ndarray
    individual_gain: float


def _normalise_seeds(
    seeds: Mapping[str, Sequence[int] | np.ndarray] | Sequence[int] | np.ndarray,
    units: int,
) -> dict[str, np.ndarray]:
    if isinstance(seeds, Mapping):
        if not seeds:
            raise ValueError("at least one seed is required")
        return {
            str(name): _validate_states(value, units)
            for name, value in sorted(seeds.items(), key=lambda item: str(item[0]))
        }
    values = np.asarray(seeds, dtype=np.int64)
    if values.ndim == 1:
        return {"seed": _validate_states(values, units)}
    if values.ndim == 2 and values.shape[1] == units and values.shape[0] > 0:
        return {
            f"seed_{index:03d}": _validate_states(row, units)
            for index, row in enumerate(values)
        }
    raise ValueError("seeds must be one state vector, a matrix, or a named mapping")


def _shortlisted_moves(
    contributions: np.ndarray,
    states: np.ndarray,
    residual: np.ndarray,
    shortlist_size: int | None,
    *,
    metric: np.ndarray | None,
    proxy: np.ndarray | None,
    beta: float,
) -> list[_ScoredMove]:
    raw: list[tuple[UnitStateMove, np.ndarray]] = []
    for unit, source in enumerate(states.tolist()):
        for destination in range(4):
            if destination == source:
                continue
            delta_pages = int(STATE_PAGE_COSTS[destination] - STATE_PAGE_COSTS[source])
            move = UnitStateMove(unit, source, destination, delta_pages)
            raw.append((move, contributions[destination, unit] - contributions[source, unit]))
    vectors = np.stack([value[1] for value in raw])
    dots = _metric_row_inner(vectors, residual, metric=metric, proxy=proxy, beta=beta)
    norms = _metric_row_norms(vectors, metric=metric, proxy=proxy, beta=beta)
    gains = 2.0 * dots - norms
    scored = [_ScoredMove(move, vector, float(gain)) for (move, vector), gain in zip(raw, gains)]

    if shortlist_size is None:
        return scored
    limit = int(shortlist_size)
    if limit < 1:
        raise ValueError("shortlist_size must be positive or None")
    # Shortlist independently by signed page delta.  This prevents the
    # necessary removal half of a balanced swap from being crowded out by
    # individually attractive additions.
    result: list[_ScoredMove] = []
    for page_delta in sorted({candidate.move.page_delta for candidate in scored}):
        group = [candidate for candidate in scored if candidate.move.page_delta == page_delta]
        group.sort(key=lambda value: (
            -value.individual_gain, value.move.unit, value.move.to_state,
        ))
        result.extend(group[:limit])
    return result


def _grouped_balanced_combinations(
    candidates: Sequence[_ScoredMove], size: int,
):
    """Yield deterministic unique-unit combinations whose page delta is zero."""
    groups: dict[int, list[int]] = {}
    for index, candidate in enumerate(candidates):
        groups.setdefault(candidate.move.page_delta, []).append(index)
    deltas = sorted(groups)
    for signature in combinations_with_replacement(deltas, size):
        if sum(signature) != 0:
            continue
        counts = {delta: signature.count(delta) for delta in set(signature)}
        choices = [
            combinations(groups[delta], counts[delta])
            for delta in sorted(counts)
        ]
        for parts in product(*choices):
            indices = tuple(sorted(index for part in parts for index in part))
            units = [candidates[index].move.unit for index in indices]
            if len(set(units)) == size:
                yield indices


def _candidate_combinations(
    candidates: Sequence[_ScoredMove],
    size: int,
    *,
    page_balanced: bool,
    current_pages: int,
    budget_pages: int,
):
    source = (
        _grouped_balanced_combinations(candidates, size)
        if page_balanced else combinations(range(len(candidates)), size)
    )
    for indices in source:
        moves = [candidates[index].move for index in indices]
        if len({move.unit for move in moves}) != size:
            continue
        next_pages = current_pages + sum(move.page_delta for move in moves)
        if next_pages < 0 or next_pages > budget_pages:
            continue
        yield tuple(indices)


def _improve_seed(
    outputs: FactorizedUnitOutputs,
    name: str,
    seed: np.ndarray,
    budget_pages: int,
    shortlist_size: int | None,
    max_swap_units: int,
    max_passes: int,
    improvement_tolerance: float,
    page_balanced: bool,
    *,
    metric: np.ndarray | None,
    proxy: np.ndarray | None,
    beta: float,
) -> SeedLocalSearchTrace:
    contributions = _state_contributions(outputs)
    target = np.asarray(outputs.target_output, dtype=np.float64)
    states = seed.copy()
    current = hybrid_state_output(outputs, states)
    residual = target - current
    pages = hybrid_state_pages(states)
    seed_energy = _energy(residual, metric=metric, proxy=proxy, beta=beta)
    energy = seed_energy
    steps: list[LocalSearchStep] = []

    for _ in range(max_passes):
        candidates = _shortlisted_moves(
            contributions, states, residual, shortlist_size,
            metric=metric, proxy=proxy, beta=beta,
        )
        if not candidates:
            break
        vectors = np.stack([candidate.vector for candidate in candidates])
        pairwise = _metric_pairwise(vectors, metric=metric, proxy=proxy, beta=beta)
        best_gain = float("-inf")
        best_key: tuple[UnitStateMove, ...] | None = None
        best_indices: tuple[int, ...] | None = None
        for size in range(1, max_swap_units + 1):
            for indices in _candidate_combinations(
                candidates, size, page_balanced=page_balanced,
                current_pages=pages, budget_pages=budget_pages,
            ):
                gain = sum(candidates[index].individual_gain for index in indices)
                gain -= 2.0 * sum(
                    pairwise[left, right]
                    for left, right in combinations(indices, 2)
                )
                key = tuple(sorted(candidates[index].move for index in indices))
                if (
                    gain > best_gain + improvement_tolerance
                    or (
                        abs(gain - best_gain) <= improvement_tolerance
                        and (best_key is None or key < best_key)
                    )
                ):
                    best_gain = float(gain)
                    best_key = key
                    best_indices = indices
        if best_indices is None or best_gain <= improvement_tolerance:
            break
        correction = np.sum([candidates[index].vector for index in best_indices], axis=0)
        next_residual = residual - correction
        next_energy = _energy(next_residual, metric=metric, proxy=proxy, beta=beta)
        exact_gain = energy - next_energy
        if exact_gain <= improvement_tolerance:
            break
        before = energy
        chosen = tuple(sorted((candidates[index].move for index in best_indices)))
        for move in chosen:
            states[move.unit] = move.to_state
        pages = hybrid_state_pages(states)
        residual = next_residual
        energy = next_energy
        steps.append(LocalSearchStep(
            moves=chosen,
            gain=float(exact_gain),
            energy_before=float(before),
            energy_after=float(energy),
            pages_after=pages,
        ))

    return SeedLocalSearchTrace(
        seed_name=name,
        seed_states=seed.copy(),
        seed_energy=float(seed_energy),
        final_states=states.copy(),
        final_energy=float(energy),
        final_pages=pages,
        steps=tuple(steps),
    )


def exact_monotone_unit_local_search(
    outputs: FactorizedUnitOutputs,
    seeds: Mapping[str, Sequence[int] | np.ndarray] | Sequence[int] | np.ndarray,
    budget_pages: int,
    *,
    shortlist_size: int | None = 16,
    max_swap_units: int = 3,
    max_passes: int = 100,
    page_balanced: bool = True,
    improvement_tolerance: float = 1e-12,
    metric: np.ndarray | None = None,
    proxy: np.ndarray | None = None,
    beta: float = 0.0,
) -> MonotoneLocalSearchResult:
    """Exactly rescore deterministic one/two/three-unit local state swaps.

    Each seed is improved independently and the best final basin is returned.
    Primitive moves are shortlisted *per signed page delta*, after which every
    feasible combination of up to ``max_swap_units`` distinct units is scored
    with its exact fixed-coefficient joint marginal.  With the default
    ``page_balanced=True``, every accepted bundle has zero net page change.

    A step is accepted only when direct complete-residual qenergy decreases,
    making every seed trace monotone and the returned result never worse than
    the best coherent, factorized, or hybrid seed supplied by the caller.
    """
    budget = int(budget_pages)
    if budget < 0 or budget > 3 * outputs.units:
        raise ValueError(f"budget_pages must lie within [0, {3 * outputs.units}]")
    swaps = int(max_swap_units)
    if swaps < 1 or swaps > 3:
        raise ValueError("max_swap_units must lie in [1, 3]")
    passes = int(max_passes)
    if passes < 0:
        raise ValueError("max_passes must be nonnegative")
    tolerance = float(improvement_tolerance)
    if tolerance < 0.0:
        raise ValueError("improvement_tolerance must be nonnegative")
    dense, future = _validate_metric_arguments(outputs.output_size, metric, proxy, beta)
    normalised = _normalise_seeds(seeds, outputs.units)
    for name, seed in normalised.items():
        if hybrid_state_pages(seed) > budget:
            raise ValueError(f"seed {name!r} exceeds the physical page budget")

    traces = {
        name: _improve_seed(
            outputs, name, seed, budget, shortlist_size, swaps, passes,
            tolerance, bool(page_balanced), metric=dense, proxy=future, beta=beta,
        )
        for name, seed in normalised.items()
    }
    best = min(traces.values(), key=lambda value: (value.final_energy, value.seed_name))
    output = hybrid_state_output(outputs, best.final_states)
    return MonotoneLocalSearchResult(
        seed_name=best.seed_name,
        states=best.final_states.copy(),
        output=output,
        residual_energy=best.final_energy,
        pages=best.final_pages,
        seed_traces=traces,
    )


def utility_weighted_jaccard_distance(
    first_mask: Sequence[bool] | np.ndarray,
    second_mask: Sequence[bool] | np.ndarray,
    first_utility: Sequence[float] | np.ndarray,
    second_utility: Sequence[float] | np.ndarray | None = None,
) -> float:
    """Symmetric utility-weighted Jaccard distance between two supports.

    When utilities differ by invocation, their arithmetic mean supplies the
    symmetric coordinate weight.  Two supports with zero weighted union have
    distance zero.
    """
    first = np.asarray(first_mask, dtype=bool).reshape(-1)
    second = np.asarray(second_mask, dtype=bool).reshape(-1)
    left = np.asarray(first_utility, dtype=np.float64).reshape(-1)
    right = left if second_utility is None else np.asarray(second_utility, dtype=np.float64).reshape(-1)
    if first.shape != second.shape or first.shape != left.shape or left.shape != right.shape:
        raise ValueError("masks and utility vectors must have equal widths")
    if np.any(~np.isfinite(left)) or np.any(~np.isfinite(right)) or np.any(left < 0) or np.any(right < 0):
        raise ValueError("utility weights must be finite and nonnegative")
    weights = 0.5 * (left + right)
    union = float(weights[first | second].sum())
    if union == 0.0:
        return 0.0
    intersection = float(weights[first & second].sum())
    return float(1.0 - intersection / union)


@dataclass(frozen=True)
class ExpertTemplateBank:
    expert_id: Hashable
    sample_indices: tuple[int, ...]
    masks: np.ndarray

    def prefix(self, templates: int) -> np.ndarray:
        count = int(templates)
        if count < 0 or count > len(self.sample_indices):
            raise ValueError("template prefix is outside the fitted bank")
        return np.asarray(self.masks[:count], dtype=bool).copy()


@dataclass(frozen=True)
class NestedTemplateBank:
    """Per-expert nested medoid then farthest-first support templates."""

    experts: Mapping[Hashable, ExpertTemplateBank]
    units: int

    def for_expert(self, expert_id: Hashable) -> ExpertTemplateBank:
        if expert_id not in self.experts:
            raise KeyError(f"expert {expert_id!r} has no fitted templates")
        return self.experts[expert_id]


def _python_scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def build_nested_template_bank(
    expert_ids: Sequence[Hashable] | np.ndarray,
    masks: Sequence[Sequence[bool]] | np.ndarray,
    utilities: Sequence[Sequence[float]] | np.ndarray,
    max_templates: int,
) -> NestedTemplateBank:
    """Fit per-expert nested utility-Jaccard medoid/farthest templates.

    The first support is the exact sample medoid (minimum summed symmetric
    utility-weighted Jaccard distance).  Every later support is the sample
    farthest from its nearest chosen support.  Duplicate supports are skipped,
    so all prefixes are nested and add a physically distinct candidate mask.
    Global sample index resolves every tie.
    """
    ids = np.asarray(expert_ids).reshape(-1)
    support = np.asarray(masks, dtype=bool)
    weight = np.asarray(utilities, dtype=np.float64)
    if support.ndim != 2 or weight.shape != support.shape or ids.shape != (support.shape[0],):
        raise ValueError("expert_ids, masks and utilities have inconsistent shapes")
    if support.shape[0] == 0:
        raise ValueError("at least one training mask is required")
    if np.any(~np.isfinite(weight)) or np.any(weight < 0):
        raise ValueError("utilities must be finite and nonnegative")
    limit = int(max_templates)
    if limit < 1:
        raise ValueError("max_templates must be positive")

    ordered_experts = sorted({_python_scalar(value) for value in ids.tolist()}, key=lambda value: (type(value).__name__, repr(value)))
    banks: dict[Hashable, ExpertTemplateBank] = {}
    for expert in ordered_experts:
        global_indices = np.flatnonzero(ids == expert).astype(np.int64)
        local_masks = support[global_indices]
        local_weights = weight[global_indices]
        count = len(global_indices)
        distances = np.zeros((count, count), dtype=np.float64)
        for left in range(count):
            for right in range(left + 1, count):
                distance = utility_weighted_jaccard_distance(
                    local_masks[left], local_masks[right],
                    local_weights[left], local_weights[right],
                )
                distances[left, right] = distances[right, left] = distance
        medoid_cost = distances.sum(axis=1)
        first = min(range(count), key=lambda index: (medoid_cost[index], int(global_indices[index])))
        chosen = [first]
        chosen_masks = {local_masks[first].tobytes()}
        unique_count = len({row.tobytes() for row in local_masks})
        while len(chosen) < min(limit, unique_count):
            candidates = [
                index for index in range(count)
                if local_masks[index].tobytes() not in chosen_masks
            ]
            next_index = max(
                candidates,
                key=lambda index: (
                    float(np.min(distances[index, chosen])),
                    -int(global_indices[index]),
                ),
            )
            chosen.append(next_index)
            chosen_masks.add(local_masks[next_index].tobytes())
        selected_global = tuple(int(global_indices[index]) for index in chosen)
        banks[expert] = ExpertTemplateBank(
            expert_id=expert,
            sample_indices=selected_global,
            masks=np.asarray(support[list(selected_global)], dtype=bool).copy(),
        )
    return NestedTemplateBank(experts=banks, units=int(support.shape[1]))


@dataclass(frozen=True)
class ChargedTemplateCandidates:
    """Candidate mask with exact physical charge and utility accounting."""

    candidate_mask: np.ndarray
    template_indices: tuple[int, ...]
    repair_units: tuple[int, ...]
    pages: int
    bytes_read: int
    retained_utility: float
    overfetch: float | None

    @property
    def candidate_units(self) -> tuple[int, ...]:
        return tuple(np.flatnonzero(self.candidate_mask).astype(int).tolist())


def charged_template_candidates(
    templates: Sequence[Sequence[bool]] | np.ndarray,
    template_indices: Sequence[int],
    *,
    repair_order: Sequence[int] = (),
    repair_count: int = 0,
    utility: Sequence[float] | np.ndarray | None = None,
    pages_per_unit: int = 3,
    page_bytes: int = 512,
    applied_units: int | None = None,
) -> ChargedTemplateCandidates:
    """Union selected templates, add unique repairs, and charge every page."""
    masks = np.asarray(templates, dtype=bool)
    if masks.ndim != 2 or masks.shape[0] == 0:
        raise ValueError("templates must be a nonempty [template, unit] matrix")
    chosen = tuple(int(value) for value in template_indices)
    if not chosen or len(set(chosen)) != len(chosen) or any(value < 0 or value >= len(masks) for value in chosen):
        raise ValueError("template_indices must be distinct valid template rows")
    repairs_requested = int(repair_count)
    if repairs_requested < 0:
        raise ValueError("repair_count must be nonnegative")
    pages_each = int(pages_per_unit)
    bytes_each = int(page_bytes)
    if pages_each < 1 or bytes_each < 1:
        raise ValueError("page accounting values must be positive")
    candidate = np.any(masks[list(chosen)], axis=0)
    repairs: list[int] = []
    if repairs_requested:
        for raw_unit in repair_order:
            unit = int(raw_unit)
            if unit < 0 or unit >= masks.shape[1]:
                raise ValueError("repair unit is outside the template width")
            if candidate[unit]:
                continue
            candidate[unit] = True
            repairs.append(unit)
            if len(repairs) == repairs_requested:
                break
    if len(repairs) < repairs_requested:
        raise ValueError("repair_order does not contain enough new candidate units")
    count = int(candidate.sum())
    pages = count * pages_each
    values = None if utility is None else np.asarray(utility, dtype=np.float64).reshape(-1)
    if values is not None:
        if values.shape != (masks.shape[1],) or np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError("utility must be one finite nonnegative value per unit")
        total = float(values.sum())
        retained = 1.0 if total == 0.0 else float(values[candidate].sum() / total)
    else:
        retained = float("nan")
    if applied_units is None:
        overfetch = None
    else:
        applied = int(applied_units)
        if applied < 1:
            raise ValueError("applied_units must be positive")
        overfetch = count / applied
    return ChargedTemplateCandidates(
        candidate_mask=candidate.copy(),
        template_indices=chosen,
        repair_units=tuple(repairs),
        pages=pages,
        bytes_read=pages * bytes_each,
        retained_utility=retained,
        overfetch=overfetch,
    )


def best_charged_template_candidates(
    templates: Sequence[Sequence[bool]] | np.ndarray,
    utility: Sequence[float] | np.ndarray,
    *,
    union_size: int = 1,
    repair_count: int = 0,
    pages_per_unit: int = 3,
    page_bytes: int = 512,
    applied_units: int | None = None,
) -> ChargedTemplateCandidates:
    """Return the validation-oracle top-1/top-2 union plus utility repairs.

    Template IDs and repairs are selected jointly by final retained utility.
    Use :func:`charged_template_candidates` when the template prediction is
    fixed before an independent repair model runs.
    """
    masks = np.asarray(templates, dtype=bool)
    values = np.asarray(utility, dtype=np.float64).reshape(-1)
    if masks.ndim != 2 or masks.shape[0] == 0 or values.shape != (masks.shape[1],):
        raise ValueError("templates/utility shapes disagree")
    size = int(union_size)
    if size not in (1, 2):
        raise ValueError("union_size must be one or two")
    size = min(size, len(masks))
    stable_utility_order = np.lexsort((np.arange(values.size), -values))
    results = [
        charged_template_candidates(
            masks, indices, repair_order=stable_utility_order,
            repair_count=repair_count, utility=values,
            pages_per_unit=pages_per_unit, page_bytes=page_bytes,
            applied_units=applied_units,
        )
        for indices in combinations(range(len(masks)), size)
    ]
    return min(results, key=lambda result: (
        -result.retained_utility,
        result.pages,
        result.template_indices,
        result.repair_units,
    ))


def charged_top1_template_candidates(
    templates: Sequence[Sequence[bool]] | np.ndarray,
    utility: Sequence[float] | np.ndarray,
    **kwargs,
) -> ChargedTemplateCandidates:
    """Convenience wrapper for the best charged single-template oracle."""
    return best_charged_template_candidates(templates, utility, union_size=1, **kwargs)


def charged_top2_union_candidates(
    templates: Sequence[Sequence[bool]] | np.ndarray,
    utility: Sequence[float] | np.ndarray,
    **kwargs,
) -> ChargedTemplateCandidates:
    """Convenience wrapper for the best charged two-template union oracle."""
    return best_charged_template_candidates(templates, utility, union_size=2, **kwargs)
