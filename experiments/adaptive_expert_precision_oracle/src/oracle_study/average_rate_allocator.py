"""Average-rate allocation over compact split interaction fields.

The routines in this module never pool unrelated requests.  A scheduling
group is one captured token/layer and its eight routed experts.  Each expert
first exposes a compact nondominated rate--distortion frontier; a
multiple-choice dynamic program then enforces the group-wide page budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .interaction_field import JointInteractionFactor
from .split_interaction_field import (
    SPLIT_STATE_PAGE_COSTS,
    SplitInteractionField,
    build_split_diagonal_dp_table,
    split_coordinate_descent,
    split_local_search,
)


__all__ = [
    "RateOption",
    "RateFrontierTrace",
    "AllocationTrace",
    "randomized_joint_metric_factor",
    "lagrangian_rate_frontier",
    "multiple_choice_allocate",
    "exact_group_option_allocate",
    "qmetric_features",
]


@dataclass(frozen=True)
class RateOption:
    """One expert state on a compressed rate--distortion frontier."""

    pages: int
    damage: float
    states: np.ndarray
    source: str
    page_price: float
    coordinate_sweeps: int
    local_passes: int


@dataclass(frozen=True)
class RateFrontierTrace:
    """Nondominated options plus work performed to construct them."""

    options: tuple[RateOption, ...]
    coordinate_sweeps: int
    local_passes: int
    anchor_solves: int
    price_solves: int
    diagonal_dp_state_evaluations: int


@dataclass(frozen=True)
class AllocationTrace:
    """One deterministic multiple-choice group allocation."""

    option_indices: np.ndarray
    pages: int
    objective: float
    coordinate_sweeps: int = 0
    pair_passes: int = 0


def _canonical_columns(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, np.float64).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0:
            result[:, column] *= -1.0
    return result


def randomized_joint_metric_factor(
    down2: np.ndarray,
    down4: np.ndarray,
    proxy: np.ndarray,
    beta: float,
    rank: int,
    *,
    oversample: int = 8,
    power_iterations: int = 2,
    seed: int = 0,
    device: str = "cuda:0",
) -> JointInteractionFactor:
    """Approximate the leading joint Q2/Q4 Gram eigenspace matrix-free.

    The joint Gram is never materialized.  Each subspace iteration applies
    ``W.T @ (I + beta P P.T) @ W`` to a narrow block, where
    ``W=[D4,D2]``.  This reduces factor fitting from a dense 1024-square
    eigendecomposition to a handful of 16-column matrix products.
    """

    import torch

    d2 = np.asarray(down2, np.float32)
    d4 = np.asarray(down4, np.float32)
    p = np.asarray(proxy, np.float32)
    if d2.shape != d4.shape or d2.ndim != 2:
        raise ValueError("down2/down4 must have equal [output,unit] shapes")
    if p.ndim == 1:
        p = p[:, None]
    if p.ndim != 2 or p.shape[0] != d2.shape[0]:
        raise ValueError("proxy must have shape [output,rank]")
    if beta < 0:
        raise ValueError("beta must be nonnegative")
    requested = int(rank)
    width = min(d2.shape[1] * 2, requested + int(oversample))
    if requested < 1 or width < requested:
        raise ValueError("invalid randomized factor rank")
    if power_iterations < 0:
        raise ValueError("power_iterations must be nonnegative")

    runtime = torch.device(device)
    joint = torch.as_tensor(np.concatenate((d4, d2), axis=1), device=runtime)
    proxy_t = torch.as_tensor(p, device=runtime)
    rng = np.random.default_rng(int(seed))
    omega = rng.choice(
        np.asarray([-1.0, 1.0], np.float32),
        size=(joint.shape[1], width),
    ) / np.float32(np.sqrt(float(width)))
    omega = np.asarray(omega, np.float32)
    basis = torch.as_tensor(omega, device=runtime)

    def gram_multiply(right: "torch.Tensor") -> "torch.Tensor":
        projected = joint @ right
        if float(beta) != 0.0 and proxy_t.shape[1]:
            projected = projected + float(beta) * (proxy_t @ (proxy_t.T @ projected))
        return joint.T @ projected

    basis, _ = torch.linalg.qr(gram_multiply(basis), mode="reduced")
    for _ in range(int(power_iterations)):
        basis, _ = torch.linalg.qr(gram_multiply(basis), mode="reduced")
    small = basis.T @ gram_multiply(basis)
    small = (small + small.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(small)
    order = torch.argsort(eigenvalues, descending=True)[:requested]
    values = torch.clamp(eigenvalues[order], min=0.0)
    factor = basis @ eigenvectors[:, order]
    factor = factor * torch.sqrt(values)[None, :]
    decoded = _canonical_columns(factor.detach().cpu().numpy()).astype(np.float32)
    units = d2.shape[1]
    return JointInteractionFactor(
        l4=decoded[:units],
        l2=decoded[units:],
        method="matrix_free_randomized_joint_eigh",
        tail_rank=decoded.shape[1],
        exact_rank=0,
        encoding="float32",
    )


def _option_key(option: RateOption) -> tuple[float, int, str, tuple[int, ...]]:
    return (
        float(option.damage),
        int(option.pages),
        str(option.source),
        tuple(np.asarray(option.states, np.int64).tolist()),
    )


def _pareto_options(options: Sequence[RateOption]) -> tuple[RateOption, ...]:
    by_pages: dict[int, RateOption] = {}
    for option in options:
        previous = by_pages.get(int(option.pages))
        if previous is None or _option_key(option) < _option_key(previous):
            by_pages[int(option.pages)] = option
    result: list[RateOption] = []
    best = float("inf")
    for pages in sorted(by_pages):
        option = by_pages[pages]
        if option.damage < best - 1e-12:
            result.append(option)
            best = float(option.damage)
    if not result:
        raise RuntimeError("rate frontier is empty")
    return tuple(result)


def lagrangian_rate_frontier(
    field: SplitInteractionField,
    *,
    maximum_pages: int,
    target_pages: Sequence[int],
    price_ratios: Sequence[float],
    coordinate_sweeps: int,
    local_shortlist: int,
    local_swap_units: int,
    local_max_passes: int,
    price_local_max_passes: int = 0,
) -> RateFrontierTrace:
    """Build a compact interaction-aware frontier with warm price sweeps.

    Prices are normalized by all-Q2 damage per maximum page.  Exact-self DP
    anchors guarantee the requested uniform operating points; the descending
    price path supplies intermediate nondominated choices for pooled MCKP.
    """

    maximum = int(maximum_pages)
    if maximum < 0 or maximum > 3 * field.units:
        raise ValueError("maximum_pages lies outside the split state range")
    anchors = sorted({0, maximum, *(int(value) for value in target_pages)})
    if anchors[0] < 0 or anchors[-1] > maximum:
        raise ValueError("target page anchor exceeds maximum_pages")
    ratios = tuple(float(value) for value in price_ratios)
    if not ratios or any(not np.isfinite(value) or value < 0 for value in ratios):
        raise ValueError("price ratios must be finite and nonnegative")
    if tuple(sorted(ratios, reverse=True)) != ratios:
        raise ValueError("price ratios must be descending")

    all_zero = np.zeros(field.units, np.int64)
    scale = max(field.damage(all_zero), 1e-30) / max(maximum, 1)
    options: list[RateOption] = []
    total_coordinate_sweeps = 0
    total_local_passes = 0
    diagonal = build_split_diagonal_dp_table(field, maximum)

    for pages in anchors:
        seed = diagonal.seed(pages)
        coordinate = split_coordinate_descent(
            field, seed, pages, max_sweeps=int(coordinate_sweeps),
        )
        repaired = split_local_search(
            field,
            coordinate.states,
            pages,
            shortlist_size=int(local_shortlist),
            max_swap_units=int(local_swap_units),
            max_passes=int(local_max_passes),
        )
        total_coordinate_sweeps += int(coordinate.sweeps)
        total_local_passes += int(repaired.passes)
        options.append(RateOption(
            pages=int(repaired.pages), damage=float(repaired.damage),
            states=repaired.states.copy(), source=f"hard_anchor_{pages}",
            page_price=0.0, coordinate_sweeps=int(coordinate.sweeps),
            local_passes=int(repaired.passes),
        ))

    warm = all_zero
    for ratio in ratios:
        price = float(ratio) * scale
        diagonal_seed = diagonal.seed(maximum, page_price=price)
        traces = (
            split_coordinate_descent(
                field, warm, maximum, page_price=price,
                max_sweeps=int(coordinate_sweeps),
            ),
            split_coordinate_descent(
                field, diagonal_seed, maximum, page_price=price,
                max_sweeps=int(coordinate_sweeps),
            ),
        )
        coordinate = min(
            traces,
            key=lambda item: (
                float(item.damage) + price * int(item.pages),
                int(item.pages),
                tuple(item.states.tolist()),
            ),
        )
        repaired = split_local_search(
            field,
            coordinate.states,
            maximum,
            page_price=price,
            shortlist_size=int(local_shortlist),
            max_swap_units=int(local_swap_units),
            max_passes=int(price_local_max_passes),
        )
        total_coordinate_sweeps += sum(int(item.sweeps) for item in traces)
        total_local_passes += int(repaired.passes)
        warm = repaired.states.copy()
        options.append(RateOption(
            pages=int(repaired.pages), damage=float(repaired.damage),
            states=repaired.states.copy(), source="lagrangian_price_path",
            page_price=price,
            coordinate_sweeps=sum(int(item.sweeps) for item in traces),
            local_passes=int(repaired.passes),
        ))
    return RateFrontierTrace(
        _pareto_options(options), total_coordinate_sweeps, total_local_passes,
        len(anchors), len(ratios),
        field.units * int(sum(
            maximum + 1 - int(cost) for cost in SPLIT_STATE_PAGE_COSTS
        )),
    )


def multiple_choice_allocate(
    frontiers: Sequence[Sequence[RateOption]],
    page_budget: int,
    weights: Sequence[float] | np.ndarray,
) -> AllocationTrace:
    """Exact MCKP over compact per-expert predicted-damage frontiers."""

    count = len(frontiers)
    importance = np.asarray(weights, np.float64).reshape(-1)
    if importance.shape != (count,) or np.any(~np.isfinite(importance)) or np.any(importance < 0):
        raise ValueError("weights must be finite, nonnegative, and match frontiers")
    budget = int(page_budget)
    if budget < 0:
        raise ValueError("page budget must be nonnegative")
    dynamic = np.full(budget + 1, np.inf, np.float64)
    dynamic[0] = 0.0
    choices = np.full((count, budget + 1), -1, np.int32)
    previous = np.full((count, budget + 1), -1, np.int32)
    for expert, frontier in enumerate(frontiers):
        if not frontier:
            raise ValueError("every expert frontier must be nonempty")
        updated = np.full_like(dynamic, np.inf)
        finite = np.flatnonzero(np.isfinite(dynamic))
        for option_index, option in enumerate(frontier):
            pages = int(option.pages)
            if pages < 0 or pages > budget:
                continue
            source = finite[finite + pages <= budget]
            target = source + pages
            candidate = dynamic[source] + importance[expert] * float(option.damage)
            better = candidate < updated[target] - 1e-15
            ties = np.abs(candidate - updated[target]) <= 1e-15
            better |= ties & ((choices[expert, target] < 0) | (option_index < choices[expert, target]))
            selected = np.flatnonzero(better)
            if selected.size:
                positions = target[selected]
                updated[positions] = candidate[selected]
                choices[expert, positions] = option_index
                previous[expert, positions] = source[selected]
        dynamic = updated
    used = min(
        np.flatnonzero(np.isfinite(dynamic)).tolist(),
        key=lambda pages: (float(dynamic[pages]), pages),
    )
    selected = np.empty(count, np.int64)
    cursor = int(used)
    for expert in range(count - 1, -1, -1):
        option = int(choices[expert, cursor])
        if option < 0:
            raise RuntimeError("MCKP backtrack failed")
        selected[expert] = option
        cursor = int(previous[expert, cursor])
    return AllocationTrace(selected, int(used), float(dynamic[used]))


def qmetric_features(
    residuals: np.ndarray, proxy: np.ndarray | None, beta: float,
) -> np.ndarray:
    """Embed residual rows so Euclidean dot products equal qmetric products."""

    values = np.asarray(residuals, np.float64)
    if values.ndim != 2:
        raise ValueError("residuals must have shape [options,output]")
    if proxy is None or float(beta) == 0.0:
        return values
    p = np.asarray(proxy, np.float64)
    if p.ndim == 1:
        p = p[:, None]
    if p.ndim != 2 or p.shape[0] != values.shape[1] or beta < 0:
        raise ValueError("proxy/beta do not match residuals")
    return np.concatenate((values, np.sqrt(float(beta)) * (values @ p)), axis=1)


def exact_group_option_allocate(
    frontiers: Sequence[Sequence[RateOption]],
    option_features: Sequence[np.ndarray],
    router_weights: Sequence[float] | np.ndarray,
    page_budget: int,
    seed_indices: Sequence[int] | np.ndarray,
    *,
    max_coordinate_sweeps: int = 8,
    max_pair_passes: int = 8,
    tolerance: float = 1e-12,
) -> AllocationTrace:
    """Optimize exact combined-MoE qenergy over precomputed options.

    This is a validation-only oracle.  It starts from a deployable additive
    MCKP allocation, performs exact option-coordinate updates, then exact
    two-expert swaps to exchange pages without transient budget overflow.
    """

    count = len(frontiers)
    alpha = np.asarray(router_weights, np.float64).reshape(-1)
    selected = np.asarray(seed_indices, np.int64).reshape(-1).copy()
    if alpha.shape != (count,) or selected.shape != (count,):
        raise ValueError("router weights and seed must match expert count")
    if np.any(~np.isfinite(alpha)):
        raise ValueError("router weights must be finite")
    features = []
    for expert, (frontier, raw) in enumerate(zip(frontiers, option_features)):
        value = np.asarray(raw, np.float64)
        if value.ndim != 2 or value.shape[0] != len(frontier):
            raise ValueError("option features do not match frontier")
        if selected[expert] < 0 or selected[expert] >= len(frontier):
            raise ValueError("seed option index is out of range")
        features.append(value * alpha[expert])
    budget = int(page_budget)

    def total_pages(indices: np.ndarray) -> int:
        return int(sum(
            int(frontiers[expert][int(option)].pages)
            for expert, option in enumerate(indices)
        ))

    pages = total_pages(selected)
    if pages > budget:
        raise ValueError("seed allocation exceeds page budget")
    residual = sum(
        (features[expert][int(selected[expert])] for expert in range(count)),
        np.zeros(features[0].shape[1], np.float64),
    )
    objective = float(residual @ residual)
    completed = 0
    for sweep in range(int(max_coordinate_sweeps)):
        changed = False
        for expert in range(count):
            source = int(selected[expert])
            without = residual - features[expert][source]
            base_pages = pages - int(frontiers[expert][source].pages)
            candidates = np.stack([without + row for row in features[expert]])
            energy = np.einsum("ij,ij->i", candidates, candidates, optimize=True)
            feasible = np.asarray([
                base_pages + int(option.pages) <= budget
                for option in frontiers[expert]
            ])
            destination = int(np.argmin(np.where(feasible, energy, np.inf)))
            if energy[destination] < objective - float(tolerance):
                selected[expert] = destination
                pages = base_pages + int(frontiers[expert][destination].pages)
                residual = candidates[destination]
                objective = float(energy[destination])
                changed = True
        completed = sweep + 1
        if not changed:
            break

    pair_completed = 0
    for pair_pass in range(int(max_pair_passes)):
        best: tuple[float, int, int, int, int, int, np.ndarray] | None = None
        for left in range(count):
            old_left = int(selected[left])
            for right in range(left + 1, count):
                old_right = int(selected[right])
                without = residual - features[left][old_left] - features[right][old_right]
                base_pages = pages - int(frontiers[left][old_left].pages) - int(
                    frontiers[right][old_right].pages
                )
                for li, left_option in enumerate(frontiers[left]):
                    remaining = budget - base_pages - int(left_option.pages)
                    if remaining < 0:
                        continue
                    right_ids = [
                        ri for ri, option in enumerate(frontiers[right])
                        if int(option.pages) <= remaining
                    ]
                    if not right_ids:
                        continue
                    candidates = (
                        without[None, :]
                        + features[left][li][None, :]
                        + features[right][right_ids]
                    )
                    energy = np.einsum("ij,ij->i", candidates, candidates, optimize=True)
                    position = int(np.argmin(energy))
                    ri = int(right_ids[position])
                    candidate = (
                        float(energy[position]), left, right, li, ri,
                        base_pages + int(left_option.pages) + int(frontiers[right][ri].pages),
                        candidates[position].copy(),
                    )
                    if best is None or candidate[:5] < best[:5]:
                        best = candidate
        pair_completed = pair_pass + 1
        if best is None or best[0] >= objective - float(tolerance):
            break
        objective, left, right, li, ri, pages, residual = best
        selected[left], selected[right] = li, ri
    return AllocationTrace(
        selected.copy(), int(pages), float(objective), completed, pair_completed,
    )
