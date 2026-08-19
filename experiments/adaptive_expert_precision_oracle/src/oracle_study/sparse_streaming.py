"""NumPy reference machinery for activation-dependent sparse weight streaming.

This module deliberately operates on *decoded* Q2/Q3/Q4 weight arrays.  It
does not encode, refit, or otherwise alter the locked embedded MXFP4 codec.
The implementations favour explicit, deterministic semantics over speed so a
GPU runner can use them as correctness oracles.

The two action spaces implemented here are:

* neuron-major packets, where one nested action replaces the complete gate
  row, up row, and down column of one SwiGLU unit at the next precision; and
* page-aligned gate/up tiles, where one page refines a rectangular joint tile
  directly from Q2 to Q4.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from .mxfp4_selective import BitAction


DEFAULT_PAGE_BYTES = 512
NEURON_WEIGHTS_PER_MATRIX = 2048
NEURON_REFINEMENT_PAYLOAD_BYTES = 768


def silu(value: np.ndarray) -> np.ndarray:
    """Numerically stable SiLU evaluated in float64."""
    x = np.asarray(value, dtype=np.float64)
    sigmoid = np.empty_like(x)
    positive = x >= 0
    sigmoid[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exponential = np.exp(x[~positive])
    sigmoid[~positive] = exponential / (1.0 + exponential)
    return x * sigmoid


def silu_prime(value: np.ndarray) -> np.ndarray:
    """Numerically stable derivative of SiLU."""
    x = np.asarray(value, dtype=np.float64)
    sigmoid = np.empty_like(x)
    positive = x >= 0
    sigmoid[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exponential = np.exp(x[~positive])
    sigmoid[~positive] = exponential / (1.0 + exponential)
    return sigmoid + x * sigmoid * (1.0 - sigmoid)


def _validate_level(
    level: Sequence[np.ndarray], activation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(level) != 3:
        raise ValueError("a precision level must be (gate, up, down)")
    gate, up, down = (np.asarray(value, dtype=np.float64) for value in level)
    x = np.asarray(activation, dtype=np.float64).reshape(-1)
    if gate.ndim != 2 or up.shape != gate.shape:
        raise ValueError("gate and up must be equal-shaped [unit, input] matrices")
    if gate.shape[1] != x.size:
        raise ValueError("activation width does not match gate/up input width")
    if down.ndim != 2 or down.shape[1] != gate.shape[0]:
        raise ValueError("down must have shape [output, unit]")
    return gate, up, down, x


def unit_contributions(
    levels: Sequence[Sequence[np.ndarray]], activation: np.ndarray,
) -> np.ndarray:
    """Return exact per-unit outputs for decoded precision ``levels``.

    Parameters
    ----------
    levels:
        Usually ``[(G2, U2, D2), (G3, U3, D3), (G4, U4, D4)]``.  Gate and
        up are ``[unit, input]`` and down is ``[output, unit]``.
    activation:
        The actual H0 expert input.

    Returns
    -------
    np.ndarray
        ``[level, unit, output]`` where summing over units gives the exact
        complete-expert output at that precision.
    """
    if not levels:
        raise ValueError("at least one precision level is required")
    result: list[np.ndarray] = []
    reference_shapes: tuple[tuple[int, ...], ...] | None = None
    for level in levels:
        gate, up, down, x = _validate_level(level, activation)
        shapes = (gate.shape, up.shape, down.shape)
        if reference_shapes is None:
            reference_shapes = shapes
        elif shapes != reference_shapes:
            raise ValueError("all precision levels must have identical weight shapes")
        hidden = silu(gate @ x) * (up @ x)
        result.append(hidden[:, None] * down.T)
    return np.asarray(result, dtype=np.float64)


@dataclass(frozen=True)
class NeuronMajorDecomposition:
    """Exact Q2/Q3/Q4 unit outputs and their two nested refinements."""

    levels: np.ndarray  # [3, unit, output]

    def __post_init__(self) -> None:
        values = np.asarray(self.levels)
        if values.ndim != 3 or values.shape[0] != 3:
            raise ValueError("levels must have shape [3, unit, output]")

    @property
    def units(self) -> int:
        return int(self.levels.shape[1])

    @property
    def base_output(self) -> np.ndarray:
        return np.asarray(self.levels[0].sum(axis=0), dtype=np.float64)

    @property
    def target_output(self) -> np.ndarray:
        return np.asarray(self.levels[2].sum(axis=0), dtype=np.float64)

    @property
    def refinements(self) -> np.ndarray:
        return np.diff(np.asarray(self.levels, dtype=np.float64), axis=0)

    def output_for_levels(self, precision: np.ndarray) -> np.ndarray:
        chosen = np.asarray(precision, dtype=np.int64).reshape(-1)
        if chosen.size != self.units or np.any((chosen < 2) | (chosen > 4)):
            raise ValueError("precision must contain one value in {2,3,4} per unit")
        return np.asarray(self.levels[chosen - 2, np.arange(self.units)].sum(axis=0), dtype=np.float64)


def neuron_major_decomposition(
    levels: Sequence[Sequence[np.ndarray]], activation: np.ndarray,
) -> NeuronMajorDecomposition:
    """Build the exact three-level neuron-major decomposition."""
    return NeuronMajorDecomposition(unit_contributions(levels, activation))


def _metric_terms(
    rows: np.ndarray,
    vector: np.ndarray,
    *,
    metric: np.ndarray | None,
    proxy: np.ndarray | None,
    beta: float,
) -> np.ndarray:
    """Return row-wise ``<rows, vector>`` in the qenergy metric."""
    values = np.asarray(rows, dtype=np.float64)
    target = np.asarray(vector, dtype=np.float64).reshape(-1)
    if values.shape[-1] != target.size:
        raise ValueError("vector width does not match contribution output width")
    result = values @ target
    if metric is not None:
        g = np.asarray(metric, dtype=np.float64)
        if g.shape != (target.size, target.size):
            raise ValueError("metric shape does not match output width")
        # ``metric`` replaces the Euclidean term, while proxy/beta augments it.
        result = (values @ g) @ target
    if proxy is not None and beta:
        p = np.asarray(proxy, dtype=np.float64)
        if p.ndim == 1:
            p = p[:, None]
        if p.ndim != 2 or p.shape[0] != target.size:
            raise ValueError("proxy must have shape [output, proxy_width]")
        result = result + float(beta) * ((values @ p) @ (target @ p))
    return np.asarray(result, dtype=np.float64)


def qenergy(
    vector: np.ndarray,
    *,
    metric: np.ndarray | None = None,
    proxy: np.ndarray | None = None,
    beta: float = 0.0,
) -> float:
    """Squared error under an optional dense and/or future-proxy metric."""
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    return float(_metric_terms(value[None, :], value, metric=metric, proxy=proxy, beta=beta)[0])


def _row_qenergies(
    rows: np.ndarray,
    *,
    metric: np.ndarray | None,
    proxy: np.ndarray | None,
    beta: float,
) -> np.ndarray:
    """Vectorized qenergy diagonal for a matrix of row vectors."""
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("rows must be a matrix")
    output = values.shape[1]
    if metric is None:
        result = np.einsum("ij,ij->i", values, values, optimize=True)
    else:
        g = np.asarray(metric, dtype=np.float64)
        if g.shape != (output, output):
            raise ValueError("metric shape does not match output width")
        result = np.einsum("ij,jk,ik->i", values, g, values, optimize=True)
    if proxy is not None and beta:
        p = np.asarray(proxy, dtype=np.float64)
        if p.ndim == 1:
            p = p[:, None]
        if p.ndim != 2 or p.shape[0] != output:
            raise ValueError("proxy must have shape [output, proxy_width]")
        projected = values @ p
        result = result + float(beta) * np.einsum("ij,ij->i", projected, projected, optimize=True)
    return np.asarray(result, dtype=np.float64)


def fixed_vector_greedy(
    contributions: np.ndarray,
    target: np.ndarray | None = None,
    *,
    metric: np.ndarray | None = None,
    proxy: np.ndarray | None = None,
    beta: float = 0.0,
    max_actions: int | None = None,
) -> list[BitAction]:
    """Exact fixed-coefficient marginal greedy for nested vector actions.

    ``contributions`` has shape ``[stage, coordinate, output]``.  Only the
    next stage of a coordinate is eligible.  Selected coefficients are fixed
    at one and are never least-squares refit.  If ``target`` is omitted, the
    complete refinement endpoint (the sum of all action vectors) is used.

    This direct residual implementation is intentionally independent of the
    dense action Gram and is suitable as a CPU oracle for shortlist rescoring.
    """
    values = np.asarray(contributions, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 1:
        raise ValueError("contributions must have shape [stage, coordinate, output]")
    stages, coordinates, output = values.shape
    flat = values.reshape(stages * coordinates, output)
    residual = flat.sum(axis=0) if target is None else np.asarray(target, dtype=np.float64).reshape(-1).copy()
    if residual.size != output:
        raise ValueError("target width does not match contribution output width")
    norms = _row_qenergies(flat, metric=metric, proxy=proxy, beta=beta)

    limit = stages * coordinates if max_actions is None else min(max(int(max_actions), 0), stages * coordinates)
    depth = np.zeros(coordinates, dtype=np.int64)
    selected = np.zeros(stages * coordinates, dtype=bool)
    result: list[BitAction] = []
    for _ in range(limit):
        dots = _metric_terms(flat, residual, metric=metric, proxy=proxy, beta=beta)
        marginal = 2.0 * dots - norms
        eligible = np.zeros(stages * coordinates, dtype=bool)
        for stage in range(stages):
            eligible[stage * coordinates:(stage + 1) * coordinates] = depth == stage
        marginal[~eligible | selected] = -np.inf
        chosen = int(np.argmax(marginal))
        if not np.isfinite(marginal[chosen]):
            break
        stage = chosen // coordinates + 1
        coordinate = chosen % coordinates
        result.append(BitAction(coordinate, stage, float(marginal[chosen])))
        selected[chosen] = True
        depth[coordinate] = stage
        residual -= flat[chosen]
    return result


def exact_neuron_marginal_greedy(
    decomposition: NeuronMajorDecomposition,
    *,
    metric: np.ndarray | None = None,
    proxy: np.ndarray | None = None,
    beta: float = 0.0,
    max_actions: int | None = None,
) -> list[BitAction]:
    """Exact nested Q2->Q3->Q4 neuron-packet selection."""
    return fixed_vector_greedy(
        decomposition.refinements,
        decomposition.target_output - decomposition.base_output,
        metric=metric,
        proxy=proxy,
        beta=beta,
        max_actions=max_actions,
    )


@dataclass(frozen=True)
class VariableCostSelection:
    """Nested actions selected under a unique-page budget."""

    actions: tuple[BitAction, ...]
    pages: tuple[int, ...]
    incremental_pages: tuple[int, ...]


def variable_cost_fixed_vector_greedy(
    contributions: np.ndarray,
    target: np.ndarray,
    action_pages: Mapping[int, Sequence[int]],
    page_budget: int,
    *,
    metric: np.ndarray | None = None,
    proxy: np.ndarray | None = None,
    beta: float = 0.0,
    max_actions: int | None = None,
) -> VariableCostSelection:
    """Exact residual greedy ranked by marginal gain per incremental page.

    Action IDs are stage-major.  A page already paid by an earlier action is
    free, which is what gives a neuron Q3 action cost two pages and its nested
    Q4 action an incremental cost of one.  The residual score is exact in the
    requested qenergy metric; only the greedy choice is heuristic.
    """
    values = np.asarray(contributions, dtype=np.float64)
    endpoint = np.asarray(target, dtype=np.float64).reshape(-1)
    if values.ndim != 3 or values.shape[0] < 1 or values.shape[-1] != endpoint.size:
        raise ValueError("contributions/target shapes do not define nested vector actions")
    stages, coordinates, output = values.shape
    flat = values.reshape(stages * coordinates, output)
    for action in range(len(flat)):
        if action not in action_pages:
            raise ValueError(f"missing physical page mapping for action {action}")
    residual = endpoint.copy()
    norms = _row_qenergies(flat, metric=metric, proxy=proxy, beta=beta)
    depth = np.zeros(coordinates, dtype=np.int64)
    selected = np.zeros(len(flat), dtype=bool)
    paid: set[int] = set()
    result: list[BitAction] = []
    costs: list[int] = []
    budget = max(int(page_budget), 0)
    limit = len(flat) if max_actions is None else min(max(int(max_actions), 0), len(flat))
    for _ in range(limit):
        dots = _metric_terms(flat, residual, metric=metric, proxy=proxy, beta=beta)
        marginal = 2.0 * dots - norms
        best: tuple[float, float, int, int] | None = None
        best_pages: set[int] | None = None
        for action in range(len(flat)):
            stage = action // coordinates
            coordinate = action % coordinates
            if selected[action] or depth[coordinate] != stage:
                continue
            pages = {int(page) for page in action_pages[action]}
            new_pages = pages - paid
            cost = len(new_pages)
            if len(paid) + cost > budget:
                continue
            gain = float(marginal[action])
            # Zero-cost positive work is always preferred.  The final tie
            # fields favour larger raw gain, lower cost, then lower action ID.
            if cost == 0:
                efficiency = float("inf") if gain >= 0 else float("-inf")
            else:
                efficiency = gain / cost
            candidate = (efficiency, gain, -cost, -action)
            if best is None or candidate > best:
                best = candidate
                best_pages = new_pages
        if best is None or best_pages is None:
            break
        action = -best[3]
        stage = action // coordinates + 1
        coordinate = action % coordinates
        result.append(BitAction(coordinate, stage, float(marginal[action])))
        selected[action] = True
        depth[coordinate] = stage
        residual -= flat[action]
        paid.update(best_pages)
        costs.append(len(best_pages))
    return VariableCostSelection(tuple(result), tuple(sorted(paid)), tuple(costs))


def exact_neuron_marginal_per_page_greedy(
    decomposition: NeuronMajorDecomposition,
    page_budget: int,
    *,
    metric: np.ndarray | None = None,
    proxy: np.ndarray | None = None,
    beta: float = 0.0,
    max_actions: int | None = None,
) -> VariableCostSelection:
    """Exact neuron-vector marginal greedy normalized by incremental pages."""
    return variable_cost_fixed_vector_greedy(
        decomposition.refinements,
        decomposition.target_output - decomposition.base_output,
        unit_packet_page_map(decomposition.units),
        page_budget,
        metric=metric,
        proxy=proxy,
        beta=beta,
        max_actions=max_actions,
    )


def neuron_levels_from_actions(
    actions: Iterable[BitAction], units: int, *, strict: bool = True,
) -> np.ndarray:
    """Convert nested actions to one precision in ``{2,3,4}`` per unit."""
    depth = np.zeros(int(units), dtype=np.int64)
    for action in actions:
        coordinate = int(action.coordinate)
        stage = int(action.stage)
        if not 0 <= coordinate < units or not 1 <= stage <= 2:
            raise ValueError("neuron action is outside the Q2/Q3/Q4 action space")
        if stage != depth[coordinate] + 1:
            if strict:
                raise ValueError("neuron actions are not nested or contain a duplicate")
            continue
        depth[coordinate] = stage
    return depth + 2


def apply_neuron_actions(
    decomposition: NeuronMajorDecomposition, actions: Iterable[BitAction],
) -> np.ndarray:
    """Return the exact mixed-precision expert output for ``actions``."""
    precision = neuron_levels_from_actions(actions, decomposition.units)
    return decomposition.output_for_levels(precision)


@dataclass(frozen=True)
class UnitProxyRanking:
    score: np.ndarray
    order: np.ndarray


def _stable_descending(values: np.ndarray) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    ids = np.arange(scores.size, dtype=np.int64)
    return ids[np.lexsort((ids, -scores))]


def rank_unit_proxies(
    levels: Sequence[Sequence[np.ndarray]],
    activation: np.ndarray,
    *,
    metric: np.ndarray | None = None,
    proxy: np.ndarray | None = None,
    beta: float = 0.0,
) -> dict[str, UnitProxyRanking]:
    """Score/rank the principal cheap neuron-major selector proxies.

    The returned keys correspond to independent exact unit correction norm,
    Q2 intermediate magnitude, Q2-activation times down-delta energy, and a
    first-order gate/up/down sensitivity around Q2.
    """
    if len(levels) != 3:
        raise ValueError("Q2, Q3, and Q4 levels are required")
    contributions = unit_contributions(levels, activation)
    gate2, up2, down2, x = _validate_level(levels[0], activation)
    gate4, up4, down4, _ = _validate_level(levels[2], activation)
    g2 = gate2 @ x
    u2 = up2 @ x
    h2 = silu(g2) * u2

    exact_delta = contributions[2] - contributions[0]
    exact_norm = _row_qenergies(exact_delta, metric=metric, proxy=proxy, beta=beta)
    down_delta = down4 - down2
    down_weight_aware = h2 ** 2 * _row_qenergies(
        down_delta.T, metric=metric, proxy=proxy, beta=beta,
    )
    delta_gate = (gate4 - gate2) @ x
    delta_up = (up4 - up2) @ x
    delta_hidden = u2 * silu_prime(g2) * delta_gate + silu(g2) * delta_up
    first_order_vectors = down2.T * delta_hidden[:, None] + down_delta.T * h2[:, None]
    first_order = _row_qenergies(first_order_vectors, metric=metric, proxy=proxy, beta=beta)
    scores = {
        "independent_unit_correction_norm": exact_norm,
        "q2_intermediate_magnitude": np.abs(h2),
        "down_weight_aware": down_weight_aware,
        "first_order_sensitivity": first_order,
    }
    return {
        name: UnitProxyRanking(np.asarray(score, dtype=np.float64), _stable_descending(score))
        for name, score in scores.items()
    }


@dataclass(frozen=True, order=True)
class GateUpTile:
    page: int
    row_start: int
    row_stop: int
    column_start: int
    column_stop: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.row_stop - self.row_start, self.column_stop - self.column_start

    @property
    def weights_per_matrix(self) -> int:
        rows, columns = self.shape
        return rows * columns


@dataclass(frozen=True)
class TileAction:
    tile: GateUpTile
    score: float


def gate_up_tiles(
    hidden_units: int, input_width: int, tile_shape: tuple[int, int],
) -> tuple[GateUpTile, ...]:
    """Return a deterministic row-major tiling without partial edge tiles."""
    tile_rows, tile_columns = (int(value) for value in tile_shape)
    if min(hidden_units, input_width, tile_rows, tile_columns) < 1:
        raise ValueError("matrix and tile dimensions must be positive")
    if hidden_units % tile_rows or input_width % tile_columns:
        raise ValueError("tile shape must divide the gate/up matrix exactly")
    result: list[GateUpTile] = []
    page = 0
    for row in range(0, hidden_units, tile_rows):
        for column in range(0, input_width, tile_columns):
            result.append(GateUpTile(page, row, row + tile_rows, column, column + tile_columns))
            page += 1
    return tuple(result)


def tile_payload_bytes(
    tile_shape: tuple[int, int], *, matrices: int = 2, refinement_bits: int = 2,
) -> int:
    """Exact packed payload for a joint gate/up refinement tile."""
    rows, columns = (int(value) for value in tile_shape)
    bits = rows * columns * int(matrices) * int(refinement_bits)
    if bits <= 0 or bits % 8:
        raise ValueError("tile payload must be a positive whole number of bytes")
    return bits // 8


def tile_pages(
    hidden_units: int,
    input_width: int,
    tile_shape: tuple[int, int],
    *,
    page_bytes: int = DEFAULT_PAGE_BYTES,
) -> tuple[GateUpTile, ...]:
    """Return tiles after proving that every tile is exactly one page."""
    payload = tile_payload_bytes(tile_shape)
    if payload != int(page_bytes):
        raise ValueError(f"tile payload is {payload} bytes, not one {page_bytes}-byte page")
    return gate_up_tiles(hidden_units, input_width, tile_shape)


def _resolve_tiles(
    selected: Iterable[int | GateUpTile], tiles: Sequence[GateUpTile],
) -> tuple[GateUpTile, ...]:
    by_page = {tile.page: tile for tile in tiles}
    result: list[GateUpTile] = []
    seen: set[int] = set()
    for value in selected:
        tile = value if isinstance(value, GateUpTile) else by_page.get(int(value))
        if tile is None or tile.page not in by_page or by_page[tile.page] != tile:
            raise ValueError("selected tile is not part of the supplied tiling")
        if tile.page in seen:
            raise ValueError("a tile page was selected more than once")
        seen.add(tile.page)
        result.append(tile)
    return tuple(result)


def apply_gate_up_tiles(
    gate_q2: np.ndarray,
    up_q2: np.ndarray,
    gate_q4: np.ndarray,
    up_q4: np.ndarray,
    activation: np.ndarray,
    selected: Iterable[int | GateUpTile],
    *,
    tile_shape: tuple[int, int] = (32, 32),
    tiles: Sequence[GateUpTile] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply selected exact Q2->Q4 joint tile pages to gate/up matvecs."""
    gate2 = np.asarray(gate_q2, dtype=np.float64)
    up2 = np.asarray(up_q2, dtype=np.float64)
    gate4 = np.asarray(gate_q4, dtype=np.float64)
    up4 = np.asarray(up_q4, dtype=np.float64)
    x = np.asarray(activation, dtype=np.float64).reshape(-1)
    if gate2.ndim != 2 or up2.shape != gate2.shape or gate4.shape != gate2.shape or up4.shape != gate2.shape:
        raise ValueError("all gate/up matrices must have the same two-dimensional shape")
    if gate2.shape[1] != x.size:
        raise ValueError("activation width does not match gate/up input width")
    layout = tuple(tiles) if tiles is not None else gate_up_tiles(*gate2.shape, tile_shape)
    chosen = _resolve_tiles(selected, layout)
    gate = gate2 @ x
    up = up2 @ x
    for tile in chosen:
        rows = slice(tile.row_start, tile.row_stop)
        columns = slice(tile.column_start, tile.column_stop)
        gate[rows] += (gate4[rows, columns] - gate2[rows, columns]) @ x[columns]
        up[rows] += (up4[rows, columns] - up2[rows, columns]) @ x[columns]
    return gate, up


def tile_expert_output(
    gate_q2: np.ndarray,
    up_q2: np.ndarray,
    gate_q4: np.ndarray,
    up_q4: np.ndarray,
    down: np.ndarray,
    activation: np.ndarray,
    selected: Iterable[int | GateUpTile],
    *,
    tile_shape: tuple[int, int] = (32, 32),
    tiles: Sequence[GateUpTile] | None = None,
) -> np.ndarray:
    """Complete expert output with mixed Q2/Q4 gate/up tile pages."""
    gate, up = apply_gate_up_tiles(
        gate_q2, up_q2, gate_q4, up_q4, activation, selected,
        tile_shape=tile_shape, tiles=tiles,
    )
    down_matrix = np.asarray(down, dtype=np.float64)
    if down_matrix.ndim != 2 or down_matrix.shape[1] != gate.size:
        raise ValueError("down must have shape [output, hidden_unit]")
    return np.asarray(down_matrix @ (silu(gate) * up), dtype=np.float64)


def exact_tile_marginal_greedy(
    gate_q2: np.ndarray,
    up_q2: np.ndarray,
    gate_q4: np.ndarray,
    up_q4: np.ndarray,
    down: np.ndarray,
    activation: np.ndarray,
    *,
    tile_shape: tuple[int, int] = (32, 32),
    tiles: Sequence[GateUpTile] | None = None,
    metric: np.ndarray | None = None,
    proxy: np.ndarray | None = None,
    beta: float = 0.0,
    max_pages: int | None = None,
) -> list[TileAction]:
    """Exact nonlinear page marginal greedy using actual SwiGLU outputs."""
    gate2 = np.asarray(gate_q2, dtype=np.float64)
    up2 = np.asarray(up_q2, dtype=np.float64)
    gate4 = np.asarray(gate_q4, dtype=np.float64)
    up4 = np.asarray(up_q4, dtype=np.float64)
    down_matrix = np.asarray(down, dtype=np.float64)
    x = np.asarray(activation, dtype=np.float64).reshape(-1)
    layout = tuple(tiles) if tiles is not None else gate_up_tiles(*gate2.shape, tile_shape)
    # Validation and baseline computation are shared with the public apply API.
    gate, up = apply_gate_up_tiles(gate2, up2, gate4, up4, x, (), tile_shape=tile_shape, tiles=layout)
    target = down_matrix @ (silu(gate4 @ x) * (up4 @ x))
    output = down_matrix @ (silu(gate) * up)
    current_damage = qenergy(target - output, metric=metric, proxy=proxy, beta=beta)
    remaining = {tile.page: tile for tile in layout}
    limit = len(layout) if max_pages is None else min(max(int(max_pages), 0), len(layout))
    result: list[TileAction] = []
    for _ in range(limit):
        best: tuple[float, int, GateUpTile, np.ndarray, np.ndarray, np.ndarray, float] | None = None
        for page in sorted(remaining):
            tile = remaining[page]
            rows = slice(tile.row_start, tile.row_stop)
            columns = slice(tile.column_start, tile.column_stop)
            candidate_gate = gate[rows] + (gate4[rows, columns] - gate2[rows, columns]) @ x[columns]
            candidate_up = up[rows] + (up4[rows, columns] - up2[rows, columns]) @ x[columns]
            hidden_delta = silu(candidate_gate) * candidate_up - silu(gate[rows]) * up[rows]
            candidate_output = output + down_matrix[:, rows] @ hidden_delta
            damage = qenergy(target - candidate_output, metric=metric, proxy=proxy, beta=beta)
            gain = current_damage - damage
            candidate = (float(gain), -page, tile, candidate_gate, candidate_up, candidate_output, float(damage))
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        if best is None:
            break
        gain, negative_page, tile, candidate_gate, candidate_up, output, current_damage = best
        rows = slice(tile.row_start, tile.row_stop)
        gate[rows] = candidate_gate
        up[rows] = candidate_up
        del remaining[-negative_page]
        result.append(TileAction(tile, gain))
    return result


def activation_action_scores(
    activation: np.ndarray,
    deltas: np.ndarray,
    *,
    rule: str = "magnitude",
    metric: np.ndarray | None = None,
    proxy: np.ndarray | None = None,
    beta: float = 0.0,
) -> np.ndarray:
    """Score input-coordinate refinement actions from the actual activation.

    ``deltas`` is ``[stage, input_coordinate, projection_output]``.  Supported
    rules are ``magnitude`` and ``activation_weighted_delta_norm`` (also
    accepted as ``weight_aware`` or ``future_proxy_weighted``).
    """
    values = np.asarray(deltas, dtype=np.float64)
    x = np.asarray(activation, dtype=np.float64).reshape(-1)
    if values.ndim != 3 or values.shape[1] != x.size:
        raise ValueError("deltas must have shape [stage, input_coordinate, output]")
    if rule in {"magnitude", "abs_activation"}:
        return np.broadcast_to(np.abs(x)[None, :], values.shape[:2]).copy()
    if rule not in {"activation_weighted_delta_norm", "weight_aware", "future_proxy_weighted"}:
        raise ValueError(f"unknown activation shortlist rule: {rule}")
    flat = values.reshape(-1, values.shape[-1])
    norm = np.asarray([
        qenergy(row, metric=metric, proxy=proxy, beta=beta) for row in flat
    ]).reshape(values.shape[:2])
    return x[None, :] ** 2 * norm


def close_nested_candidates(action_ids: Iterable[int], coordinates: int, stages: int = 2) -> np.ndarray:
    """Add every prerequisite plane while preserving deterministic order."""
    n = int(coordinates)
    s = int(stages)
    if n < 1 or s < 1:
        raise ValueError("coordinates and stages must be positive")
    result: list[int] = []
    seen: set[int] = set()
    for value in action_ids:
        action = int(value)
        if not 0 <= action < n * s:
            raise ValueError("candidate action ID is outside the action space")
        stage = action // n
        coordinate = action % n
        for prerequisite in range(stage + 1):
            closed = prerequisite * n + coordinate
            if closed not in seen:
                seen.add(closed)
                result.append(closed)
    return np.asarray(result, dtype=np.int64)


@dataclass(frozen=True)
class ActivationShortlist:
    raw_action_ids: np.ndarray
    action_ids: np.ndarray
    scores: np.ndarray

    @property
    def overfetch_factor(self) -> float:
        return len(self.action_ids) / max(len(self.raw_action_ids), 1)


def activation_shortlist(scores: np.ndarray, count: int) -> ActivationShortlist:
    """Take a top-K action shortlist and close its nested prerequisites."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("scores must have shape [stage, coordinate]")
    stages, coordinates = values.shape
    flat = values.reshape(-1)
    ids = np.arange(flat.size, dtype=np.int64)
    ranking = ids[np.lexsort((ids, -flat))]
    raw = ranking[: min(max(int(count), 0), len(ranking))]
    closed = close_nested_candidates(raw, coordinates, stages)
    return ActivationShortlist(raw, closed, values.copy())


def fixed_vector_greedy_candidates(
    contributions: np.ndarray,
    target: np.ndarray,
    candidate_actions: Iterable[int | BitAction],
    *,
    metric: np.ndarray | None = None,
    proxy: np.ndarray | None = None,
    beta: float = 0.0,
    max_actions: int | None = None,
    action_pages: Mapping[int, Sequence[int]] | None = None,
    page_budget: int | None = None,
) -> list[BitAction]:
    """Rerun exact nested selection inside an overcomplete shortlist.

    Candidate IDs are stage-major and are closed over prerequisites before
    selection.  Returned ``BitAction`` objects retain global coordinate/stage
    identities.  ``max_actions`` gives matched logical-budget evaluation;
    supplying ``action_pages`` and ``page_budget`` additionally enforces a
    matched unique-page budget.
    """
    values = np.asarray(contributions, dtype=np.float64)
    endpoint = np.asarray(target, dtype=np.float64).reshape(-1)
    if values.ndim != 3 or values.shape[0] < 1 or values.shape[-1] != endpoint.size:
        raise ValueError("contributions/target shapes do not define nested vector actions")
    stages, coordinates, output = values.shape
    raw_ids = [_action_id(action, coordinates) for action in candidate_actions]
    closed = close_nested_candidates(raw_ids, coordinates, stages)
    allowed = np.zeros(stages * coordinates, dtype=bool)
    allowed[closed] = True
    flat = values.reshape(stages * coordinates, output)
    norms = _row_qenergies(flat, metric=metric, proxy=proxy, beta=beta)
    residual = endpoint.copy()
    depth = np.zeros(coordinates, dtype=np.int64)
    selected = np.zeros(len(flat), dtype=bool)
    paid: set[int] = set()
    if page_budget is not None and action_pages is None:
        raise ValueError("action_pages are required when page_budget is supplied")
    limit = len(closed) if max_actions is None else min(max(int(max_actions), 0), len(closed))
    result: list[BitAction] = []
    for _ in range(limit):
        dots = _metric_terms(flat, residual, metric=metric, proxy=proxy, beta=beta)
        marginal = 2.0 * dots - norms
        eligible = np.zeros(len(flat), dtype=bool)
        for stage in range(stages):
            eligible[stage * coordinates:(stage + 1) * coordinates] = depth == stage
        eligible &= allowed & ~selected
        if page_budget is not None:
            for action in np.flatnonzero(eligible):
                assert action_pages is not None
                if action not in action_pages:
                    raise ValueError(f"missing physical page mapping for action {action}")
                new_pages = {int(page) for page in action_pages[action]} - paid
                if len(paid) + len(new_pages) > int(page_budget):
                    eligible[action] = False
        marginal[~eligible] = -np.inf
        chosen = int(np.argmax(marginal))
        if not np.isfinite(marginal[chosen]):
            break
        stage = chosen // coordinates + 1
        coordinate = chosen % coordinates
        result.append(BitAction(coordinate, stage, float(marginal[chosen])))
        selected[chosen] = True
        depth[coordinate] = stage
        residual -= flat[chosen]
        if action_pages is not None:
            paid.update(int(page) for page in action_pages.get(chosen, ()))
    return result


def _action_id(action: int | BitAction, coordinates: int) -> int:
    if isinstance(action, BitAction):
        return (int(action.stage) - 1) * int(coordinates) + int(action.coordinate)
    return int(action)


@dataclass(frozen=True)
class BaseGramCandidateSelection:
    """Sequence-like exact shortlist path with physical-page trace."""

    actions: tuple[BitAction, ...]
    incremental_pages: tuple[int, ...]
    cumulative_pages: tuple[int, ...]
    pages: tuple[int, ...]

    def __iter__(self):
        return iter(self.actions)

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, index):
        return self.actions[index]

    def actions_for_page_budget(self, page_budget: int) -> tuple[BitAction, ...]:
        """Return the full-path prefix costing no more than ``page_budget``."""
        count = int(np.searchsorted(self.cumulative_pages, int(page_budget), side="right"))
        return self.actions[:count]


def base_gram_candidate_greedy(
    gram: np.ndarray,
    activation: np.ndarray,
    candidate_actions: Iterable[int | BitAction],
    *,
    max_actions: int | None = None,
    action_pages: Mapping[int, Sequence[int]] | None = None,
    page_budget: int | None = None,
) -> BaseGramCandidateSelection:
    """Exact fixed-coefficient shortlist selection from one reusable Gram.

    ``gram`` is the activation-independent ``K = D.T @ G @ D`` in
    stage-major order.  The length-``n`` activation is expanded to
    ``v = concat(a, a)``.  Candidate global action IDs are closed over Q3
    prerequisites, then selected with

    ``2*a_j*(K@v)_j - a_j**2*K[j,j]``.

    Selecting ``j`` sets ``v_j=0`` through the exact correlation update
    ``corr -= a_j*K[:,j]``.  No action vectors or dense activation-specific
    Gram are rebuilt.  Ties use ``np.argmax`` over global stage-major IDs,
    matching the existing exact fixed-marginal order.  When ``action_pages``
    is supplied, the returned cumulative trace can reprice every prefix of a
    single full path; ``page_budget`` optionally constrains selection itself.
    """
    # ``build_base_gram`` returns a CPU float32 array. Keep the selector in
    # that dtype so every activation reuses the resident Gram instead of
    # allocating and copying a 2n-by-2n float64 matrix. This deliberately
    # matches ``exact_marginal_fixed_greedy_from_gram``; float64 callers pay a
    # one-time downcast, while the deployment path is zero-copy here.
    k = np.asarray(gram, dtype=np.float32)
    if k.ndim != 2 or k.shape[0] != k.shape[1] or k.shape[0] % 2:
        raise ValueError("gram must be square with two equal action planes")
    coordinates = k.shape[0] // 2
    a = np.asarray(activation, dtype=np.float32).reshape(-1)
    if a.size != coordinates:
        raise ValueError(f"activation has {a.size} values; expected {coordinates}")
    coefficients = np.concatenate((a, a))
    raw_ids = [_action_id(action, coordinates) for action in candidate_actions]
    closed = close_nested_candidates(raw_ids, coordinates, 2)
    allowed = np.zeros(2 * coordinates, dtype=bool)
    allowed[closed] = True
    if page_budget is not None and action_pages is None:
        raise ValueError("action_pages are required when page_budget is supplied")
    if action_pages is not None:
        missing = [int(action) for action in closed if int(action) not in action_pages]
        if missing:
            raise ValueError(f"missing physical page mapping for action {missing[0]}")

    correlation = k @ coefficients
    # The diagonal penalty and coefficient multiplier are invariant over the
    # path. Hoisting them avoids two temporary 2n-vectors for every selected
    # action (4,096 iterations for a full gate/up projection).
    twice_coefficients = np.float32(2.0) * coefficients
    diagonal_penalty = coefficients * coefficients * np.diag(k)
    marginal = np.empty(2 * coordinates, dtype=np.float32)
    update = np.empty(2 * coordinates, dtype=np.float32)

    # Nested eligibility changes in exactly two positions after each choice:
    # remove the selected action and, for Q3, expose its Q4 successor. Keep
    # that state incrementally instead of rebuilding it from ``depth`` on
    # every greedy iteration.
    nested_eligible = np.zeros(2 * coordinates, dtype=bool)
    nested_eligible[:coordinates] = allowed[:coordinates]

    # The locked coordinate-packet layout maps every action to exactly one
    # page. Once a page budget is full, feasibility then reduces to one
    # vectorized lookup. Retain the general set-based path for multi-page
    # packets (for example neuron-major Q3 actions), preserving its semantics.
    page_sets: tuple[frozenset[int], ...] | None = None
    single_page_ids: np.ndarray | None = None
    action_on_paid_page: np.ndarray | None = None
    if action_pages is not None:
        normalized: list[frozenset[int]] = []
        for action in range(2 * coordinates):
            if allowed[action]:
                if action not in action_pages:
                    raise ValueError(f"missing physical page mapping for action {action}")
                normalized.append(frozenset(map(int, action_pages[action])))
            else:
                normalized.append(frozenset())
        page_sets = tuple(normalized)
        if all(len(page_sets[action]) == 1 for action in closed):
            single_page_ids = np.full(2 * coordinates, -1, dtype=np.int64)
            for action in closed:
                single_page_ids[int(action)] = next(iter(page_sets[int(action)]))
            action_on_paid_page = np.zeros(2 * coordinates, dtype=bool)

    paid: set[int] = set()
    actions: list[BitAction] = []
    incremental_pages: list[int] = []
    cumulative_pages: list[int] = []
    limit = len(closed) if max_actions is None else min(max(int(max_actions), 0), len(closed))
    physical_limit = None if page_budget is None else max(int(page_budget), 0)
    for _ in range(limit):
        np.multiply(twice_coefficients, correlation, out=marginal)
        marginal -= diagonal_penalty
        eligible = nested_eligible
        if physical_limit is not None:
            assert page_sets is not None
            if single_page_ids is not None:
                # With one page per action every new page is affordable until
                # the limit is reached. Afterwards only co-resident actions
                # remain feasible. Avoid mutating ``nested_eligible``.
                if len(paid) >= physical_limit:
                    assert action_on_paid_page is not None
                    np.copyto(update, marginal)
                    update[~nested_eligible | ~action_on_paid_page] = -np.inf
                    scores = update
                else:
                    marginal[~nested_eligible] = -np.inf
                    scores = marginal
            else:
                # Multi-page actions require the original aggregate union
                # check. Copy the mask so affordability never corrupts the
                # persistent nested-eligibility state.
                eligible = nested_eligible.copy()
                for action in np.flatnonzero(eligible):
                    new_pages = page_sets[int(action)] - paid
                    if len(paid) + len(new_pages) > physical_limit:
                        eligible[action] = False
                marginal[~eligible] = -np.inf
                scores = marginal
        else:
            marginal[~nested_eligible] = -np.inf
            scores = marginal
        chosen = int(np.argmax(scores))
        if not np.isfinite(scores[chosen]):
            break
        stage = chosen // coordinates + 1
        coordinate = chosen % coordinates
        actions.append(BitAction(coordinate, stage, float(scores[chosen])))
        nested_eligible[chosen] = False
        if stage == 1 and allowed[coordinates + coordinate]:
            nested_eligible[coordinates + coordinate] = True
        np.multiply(k[:, chosen], coefficients[chosen], out=update)
        correlation -= update
        new_pages: set[int] = set()
        if page_sets is not None:
            new_pages = set(page_sets[chosen] - paid)
            paid.update(new_pages)
            if single_page_ids is not None and new_pages:
                assert action_on_paid_page is not None
                page = single_page_ids[chosen]
                action_on_paid_page |= single_page_ids == page
        incremental_pages.append(len(new_pages))
        cumulative_pages.append(len(paid))
    return BaseGramCandidateSelection(
        tuple(actions), tuple(incremental_pages), tuple(cumulative_pages), tuple(sorted(paid)),
    )


def _support_gain(
    contributions: np.ndarray,
    target: np.ndarray,
    action_ids: Iterable[int],
    *,
    metric: np.ndarray | None,
    proxy: np.ndarray | None,
    beta: float,
) -> float:
    values = np.asarray(contributions, dtype=np.float64)
    flat = values.reshape(-1, values.shape[-1])
    ids = np.asarray(list(action_ids), dtype=np.int64)
    approximation = np.zeros(flat.shape[1], dtype=np.float64) if not len(ids) else flat[ids].sum(axis=0)
    baseline = qenergy(target, metric=metric, proxy=proxy, beta=beta)
    residual = qenergy(np.asarray(target, np.float64) - approximation, metric=metric, proxy=proxy, beta=beta)
    return baseline - residual


@dataclass(frozen=True)
class ShortlistContainment:
    exact_actions: int
    candidate_actions: int
    matched_actions: int
    exact_action_recall: float
    importance_weighted_recall: float
    support_only_exact_gain_retained: float
    constrained_exact_gain_retained: float
    q3_recall: float
    q4_recall: float
    pages_required: int
    physical_bytes: int

    @property
    def exact_gain_retained(self) -> float:
        """Compatibility name for the scientifically relevant rescored gain."""
        return self.constrained_exact_gain_retained


def shortlist_containment_metrics(
    exact_order: Sequence[BitAction],
    candidates: ActivationShortlist | Iterable[int | BitAction],
    *,
    coordinates: int,
    exact_count: int | None = None,
    contributions: np.ndarray | None = None,
    target: np.ndarray | None = None,
    action_pages: Mapping[int, Sequence[int]] | None = None,
    selection_page_budget: int | None = None,
    page_bytes: int = DEFAULT_PAGE_BYTES,
    metric: np.ndarray | None = None,
    proxy: np.ndarray | None = None,
    beta: float = 0.0,
) -> ShortlistContainment:
    """Measure support/gain containment of an activation-derived shortlist."""
    prefix = list(exact_order[: len(exact_order) if exact_count is None else max(int(exact_count), 0)])
    exact_ids = [_action_id(action, coordinates) for action in prefix]
    source = candidates.action_ids if isinstance(candidates, ActivationShortlist) else candidates
    candidate_ids = close_nested_candidates(
        [_action_id(action, coordinates) for action in source], coordinates, 2,
    ).tolist()
    candidate_set = set(candidate_ids)
    matched = [action for action in exact_ids if action in candidate_set]
    weights = np.asarray([max(float(action.score), 0.0) for action in prefix], dtype=np.float64)
    if not np.any(weights > 0):
        weights = np.ones(len(prefix), dtype=np.float64)
    matched_weight = sum(weights[i] for i, action in enumerate(exact_ids) if action in candidate_set)

    plane_recall: list[float] = []
    for stage in (0, 1):
        plane = [action for action in exact_ids if action // coordinates == stage]
        plane_recall.append(sum(action in candidate_set for action in plane) / max(len(plane), 1))

    support_gain_retained = float("nan")
    constrained_gain_retained = float("nan")
    if contributions is not None:
        values = np.asarray(contributions, dtype=np.float64)
        endpoint = values.sum(axis=(0, 1)) if target is None else np.asarray(target, dtype=np.float64)
        exact_gain = _support_gain(values, endpoint, exact_ids, metric=metric, proxy=proxy, beta=beta)
        matched_gain = _support_gain(values, endpoint, matched, metric=metric, proxy=proxy, beta=beta)
        support_gain_retained = matched_gain / exact_gain if exact_gain > 0 else float("nan")
        constrained = fixed_vector_greedy_candidates(
            values,
            endpoint,
            candidate_ids,
            metric=metric,
            proxy=proxy,
            beta=beta,
            max_actions=len(exact_ids),
            action_pages=action_pages,
            page_budget=selection_page_budget,
        )
        constrained_ids = [_action_id(action, coordinates) for action in constrained]
        constrained_gain = _support_gain(
            values, endpoint, constrained_ids, metric=metric, proxy=proxy, beta=beta,
        )
        constrained_gain_retained = constrained_gain / exact_gain if exact_gain > 0 else float("nan")

    pages: set[int] = set()
    if action_pages is not None:
        for action in candidate_set:
            pages.update(int(page) for page in action_pages.get(action, ()))
    return ShortlistContainment(
        exact_actions=len(exact_ids),
        candidate_actions=len(candidate_set),
        matched_actions=len(matched),
        exact_action_recall=len(matched) / max(len(exact_ids), 1),
        importance_weighted_recall=float(matched_weight / max(float(weights.sum()), 1e-30)),
        support_only_exact_gain_retained=float(support_gain_retained),
        constrained_exact_gain_retained=float(constrained_gain_retained),
        q3_recall=float(plane_recall[0]),
        q4_recall=float(plane_recall[1]),
        pages_required=len(pages),
        physical_bytes=len(pages) * int(page_bytes),
    )


def support_jaccard(
    left: Iterable[int | BitAction], right: Iterable[int | BitAction], *, coordinates: int,
) -> float:
    """Jaccard support stability, usable for gate/up or adjacent tokens."""
    a = {_action_id(value, coordinates) for value in left}
    b = {_action_id(value, coordinates) for value in right}
    return len(a & b) / max(len(a | b), 1)


def unit_packet_page_map(units: int) -> dict[int, tuple[int, ...]]:
    """Map nested neuron actions to the three-page unit-major packet.

    Page ``3*i`` is gate+up Q3, ``3*i+1`` is gate+up Q4, and ``3*i+2``
    contains both down planes.  Consequently Q3 pays two pages and Q4 adds
    only one further page once nesting is respected.
    """
    count = int(units)
    if count < 1:
        raise ValueError("units must be positive")
    result: dict[int, tuple[int, ...]] = {}
    for unit in range(count):
        result[unit] = (3 * unit, 3 * unit + 2)
        result[count + unit] = (3 * unit + 1, 3 * unit + 2)
    return result


def tile_action_page_map(tiles: Sequence[GateUpTile]) -> dict[int, tuple[int, ...]]:
    """Map each tile action ID to its single physical page."""
    return {int(tile.page): (int(tile.page),) for tile in tiles}


@dataclass(frozen=True)
class PhysicalAccounting:
    logical_actions: int
    unique_pages: int
    physical_bytes: int
    logical_payload_bytes: int
    logical_bpw: float
    physical_bpw: float
    page_amplification: float


def physical_accounting(
    selected_actions: Iterable[int | BitAction],
    action_pages: Mapping[int, Sequence[int]],
    logical_payload_bytes: int | Mapping[int, int] | Callable[[int], int],
    *,
    coordinates: int,
    expert_weight_count: int,
    page_bytes: int = DEFAULT_PAGE_BYTES,
) -> PhysicalAccounting:
    """Account exact logical payload and paid pages for selected actions."""
    action_ids = [_action_id(action, coordinates) for action in selected_actions]
    if expert_weight_count <= 0 or page_bytes <= 0:
        raise ValueError("expert_weight_count and page_bytes must be positive")
    pages: set[int] = set()
    logical = 0
    for action in action_ids:
        if action not in action_pages:
            raise ValueError(f"missing physical page mapping for action {action}")
        pages.update(int(page) for page in action_pages[action])
        if isinstance(logical_payload_bytes, Mapping):
            logical += int(logical_payload_bytes[action])
        elif callable(logical_payload_bytes):
            logical += int(logical_payload_bytes(action))
        else:
            logical += int(logical_payload_bytes)
    physical = len(pages) * int(page_bytes)
    return PhysicalAccounting(
        logical_actions=len(action_ids),
        unique_pages=len(pages),
        physical_bytes=physical,
        logical_payload_bytes=logical,
        logical_bpw=8.0 * logical / expert_weight_count,
        physical_bpw=8.0 * physical / expert_weight_count,
        page_amplification=physical / max(logical, 1),
    )


@dataclass(frozen=True)
class BudgetedSelection:
    actions: tuple[BitAction, ...]
    pages: tuple[int, ...]


def select_neuron_actions_under_page_budget(
    order: Iterable[BitAction], units: int, page_budget: int,
) -> BudgetedSelection:
    """Apply a nested neuron path while charging its actual unique pages."""
    count = int(units)
    mapping = unit_packet_page_map(count)
    depth = np.zeros(count, dtype=np.int64)
    pages: set[int] = set()
    selected: list[BitAction] = []
    for action in order:
        coordinate = int(action.coordinate)
        stage = int(action.stage)
        if not 0 <= coordinate < count or stage != depth[coordinate] + 1 or stage > 2:
            continue
        action_id = (stage - 1) * count + coordinate
        candidate_pages = pages | set(mapping[action_id])
        if len(candidate_pages) > max(int(page_budget), 0):
            continue
        pages = candidate_pages
        depth[coordinate] = stage
        selected.append(action)
    return BudgetedSelection(tuple(selected), tuple(sorted(pages)))


def brute_force_nested_best(
    contributions: np.ndarray,
    target: np.ndarray,
    action_count: int,
    *,
    metric: np.ndarray | None = None,
    proxy: np.ndarray | None = None,
    beta: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Tiny-fixture oracle over all nested per-coordinate precision levels.

    This is intentionally exponential and should only be used in tests or
    very small scientific audits.
    """
    values = np.asarray(contributions, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("contributions must have shape [stage, coordinate, output]")
    stages, coordinates, output = values.shape
    wanted = int(action_count)
    if not 0 <= wanted <= stages * coordinates:
        raise ValueError("action_count is outside the action space")
    best_levels: np.ndarray | None = None
    best_damage = float("inf")
    for level_tuple in product(range(stages + 1), repeat=coordinates):
        if sum(level_tuple) != wanted:
            continue
        approximation = np.zeros(output, dtype=np.float64)
        for coordinate, depth in enumerate(level_tuple):
            approximation += values[:depth, coordinate].sum(axis=0)
        damage = qenergy(np.asarray(target, np.float64) - approximation, metric=metric, proxy=proxy, beta=beta)
        levels_array = np.asarray(level_tuple, dtype=np.int64)
        if damage < best_damage - 1e-15 or (
            abs(damage - best_damage) <= 1e-15
            and (best_levels is None or tuple(levels_array) < tuple(best_levels))
        ):
            best_damage = float(damage)
            best_levels = levels_array
    if best_levels is None:
        raise AssertionError("no nested support has the requested action count")
    return best_levels, best_damage
