"""Joint top-8 low-rate optimization over split MXFP4 unit states.

PR #13 builds one rate--distortion frontier per routed expert and only then
optimizes the exact combined output over those finite columns.  This module
constructs the strict unit-level continuation: all routed unit fields share
one residual signature, one pooled page budget, and one exact-self dynamic
program.  The existing per-expert factors and A/B/C streams are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .interaction_field import JointInteractionFactor
from .split_interaction_field import (
    SplitInteractionField,
    build_split_diagonal_dp_table,
    split_coordinate_descent,
    split_local_search,
)


__all__ = [
    "JointFieldTrace",
    "combine_split_interaction_fields",
    "exact_qmetric_factor",
    "optimize_joint_field",
    "split_group_states",
]


@dataclass(frozen=True)
class JointFieldTrace:
    """Best deterministic joint-field result and its bounded work."""

    states: np.ndarray
    pages: int
    damage: float
    seed_name: str
    seeds_evaluated: int
    coordinate_sweeps: int
    coordinate_accepted_moves: int
    local_passes: int
    local_accepted_bundles: int


def combine_split_interaction_fields(
    fields: Sequence[SplitInteractionField],
    router_weights: Sequence[float] | np.ndarray,
) -> SplitInteractionField:
    """Compose aligned expert factors into one exact-self top-8 field.

    If expert ``e`` has router weight ``alpha_e``, its signatures scale by
    ``alpha_e`` and its exact local energies scale by ``alpha_e**2``.  The
    resulting objective retains every within-unit self term exactly and adds
    signed within- and cross-expert interactions in the common factor basis.
    """

    if not fields:
        raise ValueError("at least one expert field is required")
    weights = np.asarray(router_weights, np.float64).reshape(-1)
    if weights.shape != (len(fields),) or np.any(~np.isfinite(weights)):
        raise ValueError("router weights must be finite and match the fields")
    if np.any(weights < 0.0):
        raise ValueError("router weights must be nonnegative")
    rank = fields[0].rank
    if any(field.rank != rank for field in fields):
        raise ValueError("all expert factors must use one aligned rank")

    l4 = np.concatenate([
        weight * np.asarray(field.factor.l4, np.float64)
        for field, weight in zip(fields, weights)
    ])
    l2 = np.concatenate([
        weight * np.asarray(field.factor.l2, np.float64)
        for field, weight in zip(fields, weights)
    ])
    rho = np.concatenate([
        weight * np.asarray(field.rho, np.float64)
        for field, weight in zip(fields, weights)
    ])
    coefficients = np.concatenate([
        np.asarray(field.coefficients, np.float64) for field in fields
    ])
    local = np.concatenate([
        weight * weight * np.asarray(field.local_damage, np.float64)
        for field, weight in zip(fields, weights)
    ])
    self_residual = np.concatenate([
        weight * weight * np.asarray(field.self_residual, np.float64)
        for field, weight in zip(fields, weights)
    ])
    factor = JointInteractionFactor(
        l4=l4,
        l2=l2,
        method="aligned_router_weighted_top8_joint_field",
        tail_rank=int(fields[0].factor.tail_rank),
        exact_rank=int(fields[0].factor.exact_rank),
        encoding=fields[0].factor.encoding,
        storage_bytes=sum(int(field.factor.payload_bytes) for field in fields),
    )
    return SplitInteractionField(
        rho=rho,
        coefficients=coefficients,
        local_damage=local,
        self_residual=self_residual,
        factor=factor,
        clipped_local_values=sum(int(field.clipped_local_values) for field in fields),
    )


def exact_qmetric_factor(
    down2_rows: np.ndarray,
    down4_rows: np.ndarray,
    proxy: np.ndarray | None,
    beta: float,
) -> JointInteractionFactor:
    """Embed down rows so factor inner products equal the exact qmetric.

    This is a validation-only ceiling.  Its coordinate axes are common across
    experts, so composing its fields recovers exact cross-expert interactions.
    """

    d2 = np.asarray(down2_rows, np.float64)
    d4 = np.asarray(down4_rows, np.float64)
    if d2.ndim != 2 or d4.shape != d2.shape:
        raise ValueError("down rows must have equal [unit, output] shapes")
    if not np.isfinite(beta) or float(beta) < 0.0:
        raise ValueError("beta must be finite and nonnegative")
    p = np.empty((d2.shape[1], 0), np.float64)
    if proxy is not None:
        p = np.asarray(proxy, np.float64)
        if p.ndim == 1:
            p = p[:, None]
        if p.ndim != 2 or p.shape[0] != d2.shape[1]:
            raise ValueError("proxy must have shape [output, rank]")
    if p.shape[1] and float(beta) != 0.0:
        scale = np.sqrt(float(beta))
        high = np.concatenate((d4, scale * (d4 @ p)), axis=1)
        low = np.concatenate((d2, scale * (d2 @ p)), axis=1)
    else:
        high, low = d4.copy(), d2.copy()
    return JointInteractionFactor(
        l4=high,
        l2=low,
        method="exact_qmetric_output_embedding",
        tail_rank=d2.shape[1],
        exact_rank=(p.shape[1] if float(beta) != 0.0 else 0),
        encoding="float64_validation_oracle",
    )


def optimize_joint_field(
    field: SplitInteractionField,
    budget_pages: int,
    *,
    named_seeds: Sequence[tuple[str, np.ndarray]] = (),
    coordinate_sweeps: int = 12,
    local_shortlist: int = 16,
    local_swap_units: int = 3,
    local_max_passes: int = 12,
) -> JointFieldTrace:
    """Run a global exact-self seed plus bounded joint interaction repair."""

    budget = int(budget_pages)
    diagonal = build_split_diagonal_dp_table(field, budget)
    seeds: list[tuple[str, np.ndarray]] = [
        ("global_exact_self_dp", diagonal.seed(budget)),
    ]
    seen = {tuple(seeds[0][1].tolist())}
    for name, raw in named_seeds:
        state = np.asarray(raw, np.int64).reshape(-1).copy()
        if state.shape != (field.units,):
            raise ValueError(f"joint seed has the wrong shape: {name}")
        if field.pages(state) > budget:
            raise ValueError(f"joint seed exceeds the page budget: {name}")
        key = tuple(state.tolist())
        if key not in seen:
            seeds.append((str(name), state))
            seen.add(key)

    best = None
    for name, seed in seeds:
        coordinate = split_coordinate_descent(
            field,
            seed,
            budget,
            max_sweeps=int(coordinate_sweeps),
        )
        local = split_local_search(
            field,
            coordinate.states,
            budget,
            shortlist_size=int(local_shortlist),
            max_swap_units=int(local_swap_units),
            max_passes=int(local_max_passes),
        )
        key = (
            float(local.damage),
            int(local.pages),
            str(name),
            tuple(local.states.tolist()),
        )
        record = (key, name, coordinate, local)
        if best is None or key < best[0]:
            best = record
    if best is None:
        raise RuntimeError("joint-field optimizer produced no result")
    _, name, coordinate, local = best
    return JointFieldTrace(
        states=local.states.copy(),
        pages=int(local.pages),
        damage=float(local.damage),
        seed_name=str(name),
        seeds_evaluated=len(seeds),
        coordinate_sweeps=int(coordinate.sweeps),
        coordinate_accepted_moves=int(coordinate.accepted_moves),
        local_passes=int(local.passes),
        local_accepted_bundles=int(local.accepted_bundles),
    )


def split_group_states(states: np.ndarray, experts: int) -> tuple[np.ndarray, ...]:
    """Split one flattened joint state vector back into expert unit vectors."""

    value = np.asarray(states, np.int64).reshape(-1)
    count = int(experts)
    if count < 1 or value.size % count:
        raise ValueError("joint state count is not divisible by expert count")
    units = value.size // count
    return tuple(value[index * units : (index + 1) * units].copy() for index in range(count))
