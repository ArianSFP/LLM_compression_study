"""Torch selectors for sparse activation-dependent expert refinement.

This module only changes the allocation of an already locked Q2->Q4 suffix.
It deliberately accepts decoded Q2 and Q4 weights: no codec coefficient,
tree, or checkpoint parameter is fitted here.

Two exact action spaces are provided:

* fixed hidden-unit output corrections, selected with exact qenergy
  marginals; and
* one-page paired gate/up tiles plus one-page down columns, selected while
  recomputing the nonlinear SwiGLU state.

The implementation keeps the large candidate calculations on the device of
the input tensors.  It therefore runs unchanged on CPU for small regression
tests and on CUDA for the experiment runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


VALID_TILE_SHAPES = ((16, 64), (32, 32), (64, 16))


@dataclass(frozen=True)
class SelectionTrace:
    """A deterministic selector path and exact output snapshots.

    ``order`` and ``gains`` are returned on CPU because every greedy choice
    is already a synchronization point and the experiment writers consume
    these arrays as metadata.  Snapshots are also detached CPU float32
    tensors, so retaining a trace cannot accidentally retain a CUDA graph or
    the expert weight tensors.
    """

    order: torch.Tensor
    gains: torch.Tensor
    snapshots: dict[int, torch.Tensor]


@dataclass(frozen=True)
class ProgressiveNeuronTrace:
    """Progressive neuron actions and exact physical-page accounting.

    ``order`` is a CPU int64 tensor with columns ``[coordinate, stage]``;
    stages one and two respectively mean Q2->Q3 and Q3->Q4.  Gains are the
    actual fixed-vector qenergy marginals, not gain-per-page scores.
    """

    order: torch.Tensor
    gains: torch.Tensor
    incremental_pages: torch.Tensor
    cumulative_pages: torch.Tensor
    snapshots: dict[int, torch.Tensor]

    @property
    def coordinates(self) -> torch.Tensor:
        return self.order[:, 0]

    @property
    def stages(self) -> torch.Tensor:
        return self.order[:, 1]


@dataclass(frozen=True)
class MixedPageGeometry:
    """Page-ID geometry for paired gate/up tiles and down columns."""

    hidden_size: int
    input_size: int
    tile_rows: int
    tile_columns: int

    @property
    def row_blocks(self) -> int:
        return self.hidden_size // self.tile_rows

    @property
    def column_blocks(self) -> int:
        return self.input_size // self.tile_columns

    @property
    def tile_pages(self) -> int:
        return self.row_blocks * self.column_blocks

    @property
    def total_pages(self) -> int:
        return self.tile_pages + self.hidden_size

    def decode(self, page_id: int) -> tuple[str, int, int | None]:
        """Decode a page ID as ``(kind, row/unit, column-or-None)``."""
        page = int(page_id)
        if page < 0 or page >= self.total_pages:
            raise ValueError(f"page ID {page} is outside [0, {self.total_pages})")
        if page < self.tile_pages:
            return "tile", page // self.column_blocks, page % self.column_blocks
        return "down", page - self.tile_pages, None


def _work_device(first: torch.Tensor, device: torch.device | str | None) -> torch.device:
    if device is None:
        return first.device
    return torch.device(device)


def _float_tensor(
    value: torch.Tensor | Sequence[float],
    *,
    device: torch.device,
) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _validate_beta_proxy(
    proxy: torch.Tensor | Sequence[float] | None,
    beta: float,
    output_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    if beta < 0.0:
        raise ValueError("beta must be nonnegative so qenergy is positive semidefinite")
    if proxy is None:
        if beta != 0.0:
            raise ValueError("a qenergy proxy is required when beta is nonzero")
        return None
    result = _float_tensor(proxy, device=device)
    if result.ndim == 1:
        result = result[:, None]
    if result.ndim != 2 or result.shape[0] != output_size:
        raise ValueError(f"proxy must have shape [{output_size}] or [{output_size}, rank]")
    return result


def qinner(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    proxy: torch.Tensor | None = None,
    beta: float = 0.0,
) -> torch.Tensor:
    """Row-wise ``left.T @ (I + beta*P*P.T) @ right``.

    The final dimension is the output dimension.  Prefix dimensions follow
    normal Torch broadcasting.  In particular, ``left`` may be a candidate
    matrix ``[actions, output]`` while ``right`` is one residual vector.
    """
    ordinary = torch.sum(left * right, dim=-1)
    if proxy is None or beta == 0.0:
        return ordinary
    p = proxy[:, None] if proxy.ndim == 1 else proxy
    left_projected = left @ p
    right_projected = right @ p
    return ordinary + float(beta) * torch.sum(left_projected * right_projected, dim=-1)


def qenergy(
    value: torch.Tensor,
    *,
    proxy: torch.Tensor | None = None,
    beta: float = 0.0,
) -> torch.Tensor:
    """Row-wise qenergy under ``G = I + beta*p*p.T``."""
    return qinner(value, value, proxy=proxy, beta=beta)


def _budgets(values: Iterable[int] | None, maximum: int) -> tuple[int, ...]:
    if values is None:
        return (maximum,)
    result = tuple(sorted(set(int(value) for value in values)))
    if any(value < 0 or value > maximum for value in result):
        raise ValueError(f"budgets must be within [0, {maximum}]")
    return result


def _cpu_snapshot(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu", dtype=torch.float32).clone()


def exact_unit_fixed_greedy(
    corrections: torch.Tensor,
    *,
    budgets: Iterable[int] | None = None,
    base_output: torch.Tensor | Sequence[float] | None = None,
    target_correction: torch.Tensor | Sequence[float] | None = None,
    proxy: torch.Tensor | Sequence[float] | None = None,
    beta: float = 0.0,
    device: torch.device | str | None = None,
) -> SelectionTrace:
    """Select fixed hidden-unit output corrections by exact marginal gain.

    ``corrections[i]`` is normally ``y_i^(4) - y_i^(2)``.  With the default
    target, their sum is the exact Q2->Q4 expert correction.  A separate
    ``target_correction`` is accepted for controlled projection studies.

    This is fixed-coefficient marginal greedy, not OMP: selecting a unit
    subtracts its correction from the residual and never refits any selected
    coefficient.  ``budgets`` count unit packets; a neuron-major packet is
    three 512-byte physical pages in the proposed storage layout.
    """
    if not isinstance(corrections, torch.Tensor):
        corrections = torch.as_tensor(corrections)
    work_device = _work_device(corrections, device)
    vectors = _float_tensor(corrections, device=work_device)
    if vectors.ndim != 2 or vectors.shape[0] < 1 or vectors.shape[1] < 1:
        raise ValueError("corrections must have shape [units, output]")
    units, output_size = map(int, vectors.shape)
    requested = _budgets(budgets, units)
    maximum = max(requested, default=0)
    p = _validate_beta_proxy(proxy, float(beta), output_size, work_device)

    if target_correction is None:
        target = vectors.sum(dim=0)
    else:
        target = _float_tensor(target_correction, device=work_device).reshape(-1)
        if target.numel() != output_size:
            raise ValueError("target_correction does not match correction vectors")
    if base_output is None:
        output = torch.zeros_like(target)
    else:
        output = _float_tensor(base_output, device=work_device).reshape(-1).clone()
        if output.numel() != output_size:
            raise ValueError("base_output does not match correction vectors")

    # The 512x512 Gram is compact, and a column update avoids rereading all
    # 512-D vectors after every choice.
    gram = vectors @ vectors.T
    if p is not None and beta != 0.0:
        projected = vectors @ p
        gram = gram + float(beta) * (projected @ projected.T)
    target_correlation = vectors @ target
    if p is not None and beta != 0.0:
        target_correlation = target_correlation + float(beta) * ((vectors @ p) @ (target @ p))
    correlation = target_correlation.clone()
    diagonal = torch.diagonal(gram)
    selected = torch.zeros(units, dtype=torch.bool, device=work_device)
    order: list[int] = []
    gains: list[float] = []
    snapshots: dict[int, torch.Tensor] = {}
    if 0 in requested:
        snapshots[0] = _cpu_snapshot(output)

    for count in range(1, maximum + 1):
        marginal = 2.0 * correlation - diagonal
        marginal = marginal.masked_fill(selected, -torch.inf)
        chosen = int(torch.argmax(marginal).item())
        gain = float(marginal[chosen].item())
        selected[chosen] = True
        order.append(chosen)
        gains.append(gain)
        output.add_(vectors[chosen])
        correlation.sub_(gram[:, chosen])
        if count in requested:
            snapshots[count] = _cpu_snapshot(output)

    return SelectionTrace(
        order=torch.tensor(order, dtype=torch.int64),
        gains=torch.tensor(gains, dtype=torch.float64),
        snapshots=snapshots,
    )


def exact_progressive_neuron_page_greedy(
    corrections: torch.Tensor,
    *,
    page_budgets: Iterable[int] | None = None,
    base_output: torch.Tensor | Sequence[float] | None = None,
    target_correction: torch.Tensor | Sequence[float] | None = None,
    proxy: torch.Tensor | Sequence[float] | None = None,
    beta: float = 0.0,
    device: torch.device | str | None = None,
) -> ProgressiveNeuronTrace:
    """Exact progressive neuron greedy under physical page budgets.

    ``corrections`` has shape ``[2, units, output]`` and normally contains
    ``c23 = y3-y2`` followed by ``c34 = y4-y3``.  Stage two is eligible only
    after stage one for the same unit.  In the coherent neuron packet, stage
    one pays the gate/up-Q3 page and the shared two-plane down page (two
    pages); stage two then pays only the gate/up-Q4 page (one incremental
    page).

    One qenergy Gram is built on ``device``.  Thereafter exact residual
    marginals use a Gram-column correlation update, and selection maximizes
    marginal gain per incremental physical page.  Equal scores choose the
    lower stage-major action ID on both CPU and CUDA.  A requested budget
    skipped by a two-page action receives the latest output attainable
    without exceeding that budget.
    """
    if not isinstance(corrections, torch.Tensor):
        corrections = torch.as_tensor(corrections)
    work_device = _work_device(corrections, device)
    values = _float_tensor(corrections, device=work_device)
    if values.ndim != 3 or values.shape[0] != 2 or values.shape[1] < 1 or values.shape[2] < 1:
        raise ValueError("corrections must have shape [2, units, output]")
    units = int(values.shape[1])
    output_size = int(values.shape[2])
    requested = _budgets(page_budgets, 3 * units)
    maximum_pages = max(requested, default=0)
    p = _validate_beta_proxy(proxy, float(beta), output_size, work_device)
    vectors = values.reshape(2 * units, output_size)

    if target_correction is None:
        target = vectors.sum(dim=0)
    else:
        target = _float_tensor(target_correction, device=work_device).reshape(-1)
        if target.numel() != output_size:
            raise ValueError("target_correction does not match correction vectors")
    if base_output is None:
        output = torch.zeros_like(target)
    else:
        output = _float_tensor(base_output, device=work_device).reshape(-1).clone()
        if output.numel() != output_size:
            raise ValueError("base_output does not match correction vectors")

    gram = vectors @ vectors.T
    target_correlation = vectors @ target
    if p is not None and beta != 0.0:
        projected = vectors @ p
        gram = gram + float(beta) * (projected @ projected.T)
        target_correlation = target_correlation + float(beta) * (projected @ (target @ p))
    correlation = target_correlation.clone()
    diagonal = torch.diagonal(gram)
    depth = torch.zeros(units, dtype=torch.int64, device=work_device)
    selected = torch.zeros(2 * units, dtype=torch.bool, device=work_device)
    page_cost = torch.cat((
        torch.full((units,), 2, dtype=torch.int64, device=work_device),
        torch.ones(units, dtype=torch.int64, device=work_device),
    ))

    order: list[tuple[int, int]] = []
    gains: list[float] = []
    incremental_pages: list[int] = []
    cumulative_pages: list[int] = []
    snapshots: dict[int, torch.Tensor] = {}
    pending = 0
    current_pages = 0
    while pending < len(requested) and requested[pending] == 0:
        snapshots[requested[pending]] = _cpu_snapshot(output)
        pending += 1

    while current_pages < maximum_pages and len(order) < 2 * units:
        marginal = 2.0 * correlation - diagonal
        eligible = torch.zeros(2 * units, dtype=torch.bool, device=work_device)
        eligible[:units] = depth == 0
        eligible[units:] = depth == 1
        fits = current_pages + page_cost <= maximum_pages
        eligible &= fits & ~selected
        if not bool(torch.any(eligible).item()):
            break
        efficiency = marginal / page_cost.to(dtype=marginal.dtype)
        efficiency = efficiency.masked_fill(~eligible, -torch.inf)
        # Match the NumPy oracle's tuple exactly:
        # (efficiency, raw marginal, -cost, -stage-major action ID).
        best_efficiency = torch.max(efficiency)
        tied = eligible & (efficiency == best_efficiency)
        tied_gain = marginal.masked_fill(~tied, -torch.inf)
        best_gain = torch.max(tied_gain)
        tied &= marginal == best_gain
        tied_cost = page_cost.masked_fill(~tied, torch.iinfo(page_cost.dtype).max)
        best_cost = torch.min(tied_cost)
        tied &= page_cost == best_cost
        # torch.argmax returns the first maximum, hence the lowest action ID.
        chosen = int(torch.argmax(tied.to(dtype=torch.int8)).item())
        cost = int(page_cost[chosen].item())
        next_pages = current_pages + cost
        # Budgets crossed by a two-page action get the previous exact state.
        while pending < len(requested) and requested[pending] < next_pages:
            snapshots[requested[pending]] = _cpu_snapshot(output)
            pending += 1

        stage = chosen // units + 1
        coordinate = chosen % units
        gain = float(marginal[chosen].item())
        output.add_(vectors[chosen])
        correlation.sub_(gram[:, chosen])
        selected[chosen] = True
        depth[coordinate] = stage
        current_pages = next_pages
        order.append((coordinate, stage))
        gains.append(gain)
        incremental_pages.append(cost)
        cumulative_pages.append(current_pages)
        while pending < len(requested) and requested[pending] == current_pages:
            snapshots[requested[pending]] = _cpu_snapshot(output)
            pending += 1

    while pending < len(requested):
        snapshots[requested[pending]] = _cpu_snapshot(output)
        pending += 1

    order_tensor = torch.tensor(order, dtype=torch.int64).reshape(-1, 2)
    return ProgressiveNeuronTrace(
        order=order_tensor,
        gains=torch.tensor(gains, dtype=torch.float64),
        incremental_pages=torch.tensor(incremental_pages, dtype=torch.int64),
        cumulative_pages=torch.tensor(cumulative_pages, dtype=torch.int64),
        snapshots=snapshots,
    )


@dataclass
class _MixedInputs:
    base_gate: torch.Tensor
    base_up: torch.Tensor
    base_down: torch.Tensor
    refined_gate: torch.Tensor
    refined_up: torch.Tensor
    refined_down: torch.Tensor
    activation: torch.Tensor
    proxy: torch.Tensor | None
    beta: float
    geometry: MixedPageGeometry


@dataclass
class _MixedState:
    gate: torch.Tensor
    up: torch.Tensor
    hidden: torch.Tensor
    down: torch.Tensor
    output: torch.Tensor
    target: torch.Tensor
    tile_selected: torch.Tensor
    down_selected: torch.Tensor


def _mixed_inputs(
    base_gate: torch.Tensor,
    base_up: torch.Tensor,
    base_down: torch.Tensor,
    refined_gate: torch.Tensor,
    refined_up: torch.Tensor,
    refined_down: torch.Tensor,
    activation: torch.Tensor,
    tile_shape: tuple[int, int],
    proxy: torch.Tensor | Sequence[float] | None,
    beta: float,
    device: torch.device | str | None,
) -> _MixedInputs:
    if not isinstance(base_gate, torch.Tensor):
        base_gate = torch.as_tensor(base_gate)
    work_device = _work_device(base_gate, device)
    g2 = _float_tensor(base_gate, device=work_device)
    u2 = _float_tensor(base_up, device=work_device)
    d2 = _float_tensor(base_down, device=work_device)
    g4 = _float_tensor(refined_gate, device=work_device)
    u4 = _float_tensor(refined_up, device=work_device)
    d4 = _float_tensor(refined_down, device=work_device)
    x = _float_tensor(activation, device=work_device).reshape(-1)
    if g2.ndim != 2 or u2.shape != g2.shape or g4.shape != g2.shape or u4.shape != g2.shape:
        raise ValueError("gate/up matrices must share shape [hidden, input]")
    hidden_size, input_size = map(int, g2.shape)
    if d2.ndim != 2 or d2.shape[1] != hidden_size or d4.shape != d2.shape:
        raise ValueError("down matrices must share shape [output, hidden]")
    if x.numel() != input_size:
        raise ValueError("activation does not match the gate/up input dimension")
    shape = (int(tile_shape[0]), int(tile_shape[1]))
    if shape not in VALID_TILE_SHAPES:
        raise ValueError(f"tile_shape must be one of {VALID_TILE_SHAPES}")
    rows, columns = shape
    if hidden_size % rows or input_size % columns:
        raise ValueError("gate/up dimensions must be divisible by tile_shape")
    geometry = MixedPageGeometry(hidden_size, input_size, rows, columns)
    p = _validate_beta_proxy(proxy, float(beta), int(d2.shape[0]), work_device)
    return _MixedInputs(g2, u2, d2, g4, u4, d4, x, p, float(beta), geometry)


def _initial_state(values: _MixedInputs) -> _MixedState:
    gate = values.base_gate @ values.activation
    up = values.base_up @ values.activation
    hidden = F.silu(gate) * up
    down = values.base_down.clone()
    output = down @ hidden
    target_gate = values.refined_gate @ values.activation
    target_up = values.refined_up @ values.activation
    target_hidden = F.silu(target_gate) * target_up
    target = values.refined_down @ target_hidden
    return _MixedState(
        gate=gate,
        up=up,
        hidden=hidden,
        down=down,
        output=output,
        target=target,
        tile_selected=torch.zeros(values.geometry.tile_pages, dtype=torch.bool, device=gate.device),
        down_selected=torch.zeros(values.geometry.hidden_size, dtype=torch.bool, device=gate.device),
    )


def _tile_activation_deltas(values: _MixedInputs) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute each tile's additive gate/up activation correction."""
    geometry = values.geometry
    row_blocks = geometry.row_blocks
    column_blocks = geometry.column_blocks
    rows, columns = geometry.tile_rows, geometry.tile_columns
    delta_gate = (values.refined_gate - values.base_gate).reshape(row_blocks, rows, column_blocks, columns)
    delta_up = (values.refined_up - values.base_up).reshape(row_blocks, rows, column_blocks, columns)
    # Reshape interleaves row elements before column blocks.  Permuting makes
    # the page axes explicit as [row block, column block, row within block].
    delta_gate = delta_gate.permute(0, 2, 1, 3)
    delta_up = delta_up.permute(0, 2, 1, 3)
    x_blocks = values.activation.reshape(column_blocks, columns)
    gate_effect = torch.einsum("abrc,bc->abr", delta_gate, x_blocks)
    up_effect = torch.einsum("abrc,bc->abr", delta_up, x_blocks)
    return gate_effect.contiguous(), up_effect.contiguous()


def _candidate_corrections(
    values: _MixedInputs,
    state: _MixedState,
    tile_gate_effect: torch.Tensor,
    tile_up_effect: torch.Tensor,
) -> torch.Tensor:
    """Return exact current output corrections for every physical page."""
    geometry = values.geometry
    rows = geometry.tile_rows
    gate_blocks = state.gate.reshape(geometry.row_blocks, rows)
    up_blocks = state.up.reshape(geometry.row_blocks, rows)
    hidden_blocks = state.hidden.reshape(geometry.row_blocks, rows)
    candidate_hidden = F.silu(gate_blocks[:, None, :] + tile_gate_effect)
    candidate_hidden = candidate_hidden * (up_blocks[:, None, :] + tile_up_effect)
    delta_hidden = candidate_hidden - hidden_blocks[:, None, :]
    down_blocks = state.down.reshape(state.down.shape[0], geometry.row_blocks, rows)
    tile_correction = torch.einsum("oar,abr->abo", down_blocks, delta_hidden)
    tile_correction = tile_correction.reshape(geometry.tile_pages, state.output.numel())
    tile_correction = tile_correction.masked_fill(state.tile_selected[:, None], 0.0)

    down_delta = values.refined_down - state.down
    down_correction = (down_delta * state.hidden[None, :]).T.contiguous()
    down_correction = down_correction.masked_fill(state.down_selected[:, None], 0.0)
    return torch.cat((tile_correction, down_correction), dim=0)


def _apply_page(
    page_id: int,
    values: _MixedInputs,
    state: _MixedState,
    tile_gate_effect: torch.Tensor,
    tile_up_effect: torch.Tensor,
) -> None:
    """Apply one page and update the exact nonlinear mixed-precision state."""
    geometry = values.geometry
    if page_id < geometry.tile_pages:
        row_block = page_id // geometry.column_blocks
        column_block = page_id % geometry.column_blocks
        start = row_block * geometry.tile_rows
        stop = start + geometry.tile_rows
        old_hidden = state.hidden[start:stop].clone()
        state.gate[start:stop].add_(tile_gate_effect[row_block, column_block])
        state.up[start:stop].add_(tile_up_effect[row_block, column_block])
        new_hidden = F.silu(state.gate[start:stop]) * state.up[start:stop]
        delta_hidden = new_hidden - old_hidden
        state.hidden[start:stop] = new_hidden
        state.output.add_(state.down[:, start:stop] @ delta_hidden)
        state.tile_selected[page_id] = True
        return

    unit = page_id - geometry.tile_pages
    delta_column = values.refined_down[:, unit] - state.down[:, unit]
    state.output.add_(delta_column * state.hidden[unit])
    state.down[:, unit] = values.refined_down[:, unit]
    state.down_selected[unit] = True


def _actual_gain(old_residual: torch.Tensor, new_residual: torch.Tensor, values: _MixedInputs) -> float:
    old_energy = qenergy(old_residual, proxy=values.proxy, beta=values.beta)
    new_energy = qenergy(new_residual, proxy=values.proxy, beta=values.beta)
    return float((old_energy - new_energy).item())


def exact_mixed_page_greedy(
    base_gate: torch.Tensor,
    base_up: torch.Tensor,
    base_down: torch.Tensor,
    refined_gate: torch.Tensor,
    refined_up: torch.Tensor,
    refined_down: torch.Tensor,
    activation: torch.Tensor,
    *,
    tile_shape: tuple[int, int] = (32, 32),
    page_budgets: Iterable[int] | None = None,
    refresh_interval: int = 1,
    proxy: torch.Tensor | Sequence[float] | None = None,
    beta: float = 0.0,
    device: torch.device | str | None = None,
) -> SelectionTrace:
    """Exact nonlinear page greedy for paired gate/up tiles and down columns.

    Tile pages are numbered row-major, followed by one page per down column.
    At a global refresh, every unselected page is evaluated against the
    current exact SwiGLU state and qenergy residual.  ``refresh_interval=1``
    is exact marginal greedy.  Larger intervals take that many pages from
    the refreshed ranking, applying each to the exact state before the next
    global refresh; this isolates the compute/recovery refresh trade-off.

    One tile or down action is exactly one 512-byte physical page.  Returned
    snapshots are therefore keyed by physical page count, not logical bits.
    """
    if refresh_interval < 1:
        raise ValueError("refresh_interval must be positive")
    values = _mixed_inputs(
        base_gate, base_up, base_down, refined_gate, refined_up, refined_down,
        activation, tile_shape, proxy, float(beta), device,
    )
    requested = _budgets(page_budgets, values.geometry.total_pages)
    maximum = max(requested, default=0)
    state = _initial_state(values)
    tile_gate_effect, tile_up_effect = _tile_activation_deltas(values)
    snapshots: dict[int, torch.Tensor] = {}
    if 0 in requested:
        snapshots[0] = _cpu_snapshot(state.output)
    order: list[int] = []
    gains: list[float] = []

    while len(order) < maximum:
        correction = _candidate_corrections(values, state, tile_gate_effect, tile_up_effect)
        residual = state.target - state.output
        marginal = 2.0 * qinner(correction, residual, proxy=values.proxy, beta=values.beta)
        marginal -= qenergy(correction, proxy=values.proxy, beta=values.beta)
        selected = torch.cat((state.tile_selected, state.down_selected))
        marginal = marginal.masked_fill(selected, -torch.inf)
        # stable=True makes equal-score page IDs deterministic on CPU/CUDA.
        ranking = torch.argsort(marginal, descending=True, stable=True)
        block = min(int(refresh_interval), maximum - len(order))
        for page_tensor in ranking[:block]:
            page = int(page_tensor.item())
            old_residual = state.target - state.output
            _apply_page(page, values, state, tile_gate_effect, tile_up_effect)
            new_residual = state.target - state.output
            order.append(page)
            gains.append(_actual_gain(old_residual, new_residual, values))
            if len(order) in requested:
                snapshots[len(order)] = _cpu_snapshot(state.output)

    return SelectionTrace(
        order=torch.tensor(order, dtype=torch.int64),
        gains=torch.tensor(gains, dtype=torch.float64),
        snapshots=snapshots,
    )


def _silu_prime(value: torch.Tensor) -> torch.Tensor:
    sigmoid = torch.sigmoid(value)
    return sigmoid * (1.0 + value * (1.0 - sigmoid))


def _first_order_proxy_scores(
    values: _MixedInputs,
    state: _MixedState,
    tile_gate_effect: torch.Tensor,
    tile_up_effect: torch.Tensor,
) -> torch.Tensor:
    geometry = values.geometry
    rows = geometry.tile_rows
    gate = state.gate.reshape(geometry.row_blocks, rows)
    up = state.up.reshape(geometry.row_blocks, rows)
    delta_hidden = (
        (up * _silu_prime(gate))[:, None, :] * tile_gate_effect
        + F.silu(gate)[:, None, :] * tile_up_effect
    )
    down_blocks = values.base_down.reshape(values.base_down.shape[0], geometry.row_blocks, rows)
    tile_correction = torch.einsum("oar,abr->abo", down_blocks, delta_hidden)
    tile_correction = tile_correction.reshape(geometry.tile_pages, state.output.numel())
    down_correction = ((values.refined_down - values.base_down) * state.hidden[None, :]).T
    correction = torch.cat((tile_correction, down_correction), dim=0)
    residual = state.target - state.output
    return (
        2.0 * qinner(correction, residual, proxy=values.proxy, beta=values.beta)
        - qenergy(correction, proxy=values.proxy, beta=values.beta)
    )


def _wina_proxy_scores(
    values: _MixedInputs,
    state: _MixedState,
) -> torch.Tensor:
    """Activation x weight x downstream-sensitivity static page scores."""
    geometry = values.geometry
    rows, columns = geometry.tile_rows, geometry.tile_columns
    row_blocks, column_blocks = geometry.row_blocks, geometry.column_blocks
    x2 = values.activation.square().reshape(column_blocks, columns)
    delta_gate = (values.refined_gate - values.base_gate).reshape(row_blocks, rows, column_blocks, columns)
    delta_up = (values.refined_up - values.base_up).reshape(row_blocks, rows, column_blocks, columns)
    delta_gate = delta_gate.permute(0, 2, 1, 3)
    delta_up = delta_up.permute(0, 2, 1, 3)

    gate_factor2 = (state.up * _silu_prime(state.gate)).square().reshape(row_blocks, rows)
    up_factor2 = F.silu(state.gate).square().reshape(row_blocks, rows)
    down_columns = values.base_down.T
    downstream = qenergy(down_columns, proxy=values.proxy, beta=values.beta).reshape(row_blocks, rows)
    local_gate = torch.einsum("abrc,bc->abr", delta_gate.square(), x2)
    local_up = torch.einsum("abrc,bc->abr", delta_up.square(), x2)
    tile_score = torch.sum(
        downstream[:, None, :] * (
            gate_factor2[:, None, :] * local_gate + up_factor2[:, None, :] * local_up
        ),
        dim=-1,
    ).reshape(-1)
    down_correction = ((values.refined_down - values.base_down) * state.hidden[None, :]).T
    down_score = qenergy(down_correction, proxy=values.proxy, beta=values.beta)
    return torch.cat((tile_score, down_score), dim=0)


def static_proxy_mixed_page_order(
    base_gate: torch.Tensor,
    base_up: torch.Tensor,
    base_down: torch.Tensor,
    refined_gate: torch.Tensor,
    refined_up: torch.Tensor,
    refined_down: torch.Tensor,
    activation: torch.Tensor,
    *,
    tile_shape: tuple[int, int] = (32, 32),
    page_budgets: Iterable[int] | None = None,
    proxy_method: str = "first_order",
    proxy: torch.Tensor | Sequence[float] | None = None,
    beta: float = 0.0,
    device: torch.device | str | None = None,
) -> SelectionTrace:
    """Apply a static first-order or WINA page order with exact state updates.

    Scores are computed only at the all-Q2 state.  The stored ``gains`` are
    nevertheless the realized qenergy gains after exact nonlinear
    application, which makes curves directly comparable with dynamic greedy.
    """
    values = _mixed_inputs(
        base_gate, base_up, base_down, refined_gate, refined_up, refined_down,
        activation, tile_shape, proxy, float(beta), device,
    )
    requested = _budgets(page_budgets, values.geometry.total_pages)
    maximum = max(requested, default=0)
    state = _initial_state(values)
    tile_gate_effect, tile_up_effect = _tile_activation_deltas(values)
    method = str(proxy_method).lower().replace("-", "_")
    if method in {"first_order", "first_order_swiglu"}:
        score = _first_order_proxy_scores(values, state, tile_gate_effect, tile_up_effect)
    elif method in {"wina", "wina_style", "activation_weight"}:
        score = _wina_proxy_scores(values, state)
    else:
        raise ValueError("proxy_method must be 'first_order' or 'wina'")
    ranking = torch.argsort(score, descending=True, stable=True)[:maximum]

    snapshots: dict[int, torch.Tensor] = {}
    if 0 in requested:
        snapshots[0] = _cpu_snapshot(state.output)
    order: list[int] = []
    gains: list[float] = []
    for page_tensor in ranking:
        page = int(page_tensor.item())
        old_residual = state.target - state.output
        _apply_page(page, values, state, tile_gate_effect, tile_up_effect)
        new_residual = state.target - state.output
        order.append(page)
        gains.append(_actual_gain(old_residual, new_residual, values))
        if len(order) in requested:
            snapshots[len(order)] = _cpu_snapshot(state.output)

    return SelectionTrace(
        order=torch.tensor(order, dtype=torch.int64),
        gains=torch.tensor(gains, dtype=torch.float64),
        snapshots=snapshots,
    )


__all__ = [
    "MixedPageGeometry",
    "ProgressiveNeuronTrace",
    "SelectionTrace",
    "VALID_TILE_SHAPES",
    "exact_mixed_page_greedy",
    "exact_progressive_neuron_page_greedy",
    "exact_unit_fixed_greedy",
    "qenergy",
    "qinner",
    "static_proxy_mixed_page_order",
]
