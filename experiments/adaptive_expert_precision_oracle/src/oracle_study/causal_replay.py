"""Causal replay primitives for PR #13 selected precision states.

The functions in this module are deliberately independent of Transformers.
They reconstruct the routed-expert error selected by PR #13 and provide the
metrics used by the downstream replay runner.  Model loading and hooks live in
the runner so reconstruction parity can be tested without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping, Sequence

import numpy as np

from .split_interaction_field import (
    SPLIT_STATE_PAGE_COSTS,
    SplitProjectionResponses,
    split_projection_responses,
    split_state_output,
)


UNITS = 512
TOP_K = 8


def canonical_states(value: str | Sequence[int] | np.ndarray, units: int = UNITS) -> np.ndarray:
    """Parse one canonical PR #13 state vector and fail closed on drift."""

    if isinstance(value, str):
        loaded = json.loads(value)
        canonical = json.dumps(loaded, separators=(",", ":"))
        if value != canonical:
            raise ValueError("selected_states is not canonical compact JSON")
    else:
        loaded = value
    states = np.asarray(loaded, np.int64).reshape(-1)
    if states.shape != (int(units),) or np.any((states < 0) | (states > 7)):
        raise ValueError(f"selected_states must contain {int(units)} values in [0,7]")
    return states.copy()


def selected_pages(states: Sequence[int] | np.ndarray) -> int:
    value = np.asarray(states, np.int64).reshape(-1)
    if np.any((value < 0) | (value > 7)):
        raise ValueError("split state lies outside [0,7]")
    return int(SPLIT_STATE_PAGE_COSTS[value].sum())


def qenergy_damage(
    error: np.ndarray,
    proxy: np.ndarray | None,
    beta: float,
) -> float:
    """Evaluate the exact PR #13 Euclidean-plus-proxy quadratic metric."""

    value = np.asarray(error, np.float64).reshape(-1)
    damage = float(value @ value)
    if proxy is not None and float(beta) != 0.0:
        projection = np.asarray(proxy, np.float64)
        if projection.ndim == 1:
            projection = projection[:, None]
        if projection.ndim != 2 or projection.shape[0] != value.size:
            raise ValueError("proxy does not match error width")
        if float(beta) < 0.0:
            raise ValueError("beta must be non-negative")
        projected = value @ projection
        damage += float(beta) * float(projected @ projected)
    return damage


@dataclass(frozen=True)
class GroupReconstruction:
    """One routed top-8 group's exact and approximate output relationship."""

    delta: np.ndarray
    residual: np.ndarray
    expert_residuals: np.ndarray
    expert_damages: np.ndarray
    selected_pages: np.ndarray
    group_damage: float
    all_q4_max_abs_error: float

    def __post_init__(self) -> None:
        delta = np.asarray(self.delta)
        residual = np.asarray(self.residual)
        expert_residuals = np.asarray(self.expert_residuals)
        expert_damages = np.asarray(self.expert_damages)
        pages = np.asarray(self.selected_pages)
        if delta.ndim != 1 or residual.shape != delta.shape:
            raise ValueError("delta and residual must be equal-width vectors")
        if expert_residuals.shape != (TOP_K, delta.size):
            raise ValueError("expert residuals must be [8, output]")
        if expert_damages.shape != (TOP_K,) or pages.shape != (TOP_K,):
            raise ValueError("expert damages/pages must contain eight values")
        if not np.allclose(delta, -residual, rtol=0.0, atol=0.0):
            raise ValueError("delta must be approximate minus exact")


def projection_responses(
    activation: np.ndarray,
    expert_ids: Sequence[int] | np.ndarray,
    decoded: Mapping[int, tuple[Sequence[np.ndarray], Sequence[np.ndarray]]],
) -> tuple[SplitProjectionResponses, ...]:
    """Build reusable Q2/Q4 responses for the routed experts in one group."""

    ids = np.asarray(expert_ids, np.int64).reshape(-1)
    if ids.shape != (TOP_K,) or len(set(ids.tolist())) != TOP_K:
        raise ValueError("expert_ids must contain eight unique experts")
    result = []
    for expert in ids.tolist():
        if int(expert) not in decoded:
            raise KeyError(f"missing decoded expert {int(expert)}")
        q2, q4 = decoded[int(expert)]
        result.append(split_projection_responses(q2, q4, activation))
    return tuple(result)


def reconstruct_group_from_responses(
    responses: Sequence[SplitProjectionResponses],
    router_weights: Sequence[float] | np.ndarray,
    state_vectors: Sequence[Sequence[int] | np.ndarray],
    proxy: np.ndarray | None,
    beta: float,
) -> GroupReconstruction:
    """Reconstruct the local error selected by one PR #13 group allocation."""

    if len(responses) != TOP_K or len(state_vectors) != TOP_K:
        raise ValueError("reconstruction requires eight expert responses and states")
    weights = np.asarray(router_weights, np.float64).reshape(-1)
    if (
        weights.shape != (TOP_K,)
        or np.any(weights <= 0.0)
        or not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=5e-7)
    ):
        raise ValueError("router_weights must be positive normalized top-8 weights")
    residuals = []
    damages = []
    pages = []
    all_q4_error = 0.0
    for response, state_value in zip(responses, state_vectors):
        states = canonical_states(state_value, response.units)
        target = np.asarray(response.target_output, np.float64)
        approximate = split_state_output(response, states)
        residual = target - approximate
        residuals.append(residual)
        damages.append(qenergy_damage(residual, proxy, beta))
        pages.append(selected_pages(states))
        q4 = split_state_output(
            response, np.full(response.units, 7, dtype=np.int64),
        )
        all_q4_error = max(
            all_q4_error,
            float(np.max(np.abs(target - q4), initial=0.0)),
        )
    expert_residuals = np.stack(residuals)
    combined = np.einsum("e,eo->o", weights, expert_residuals, optimize=True)
    return GroupReconstruction(
        delta=np.asarray(-combined, np.float64),
        residual=np.asarray(combined, np.float64),
        expert_residuals=expert_residuals,
        expert_damages=np.asarray(damages, np.float64),
        selected_pages=np.asarray(pages, np.int64),
        group_damage=qenergy_damage(combined, proxy, beta),
        all_q4_max_abs_error=float(all_q4_error),
    )


def rms_normalize(
    value: np.ndarray,
    weight: np.ndarray | None = None,
    epsilon: float = 1e-6,
) -> np.ndarray:
    source = np.asarray(value, np.float64)
    denominator = np.sqrt(np.mean(source * source, axis=-1, keepdims=True) + float(epsilon))
    result = source / denominator
    if weight is not None:
        result = result * np.asarray(weight, np.float64)
    return result


def mean_squared_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    error = np.asarray(candidate, np.float64) - np.asarray(reference, np.float64)
    return float(np.mean(error * error))


def cosine_distance(reference: np.ndarray, candidate: np.ndarray) -> float:
    first = np.asarray(reference, np.float64).reshape(-1)
    second = np.asarray(candidate, np.float64).reshape(-1)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator == 0.0:
        return 0.0 if np.array_equal(first, second) else 1.0
    return float(1.0 - np.clip(float(first @ second) / denominator, -1.0, 1.0))


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, np.float64)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=-1, keepdims=True)


def categorical_kl(reference_logits: np.ndarray, candidate_logits: np.ndarray) -> float:
    p = softmax(reference_logits)
    q = softmax(candidate_logits)
    tiny = np.finfo(np.float64).tiny
    values = np.sum(p * (np.log(np.maximum(p, tiny)) - np.log(np.maximum(q, tiny))), axis=-1)
    return float(np.mean(values))


def route_metrics(
    reference_logits: np.ndarray,
    candidate_logits: np.ndarray,
    top_k: int = TOP_K,
) -> dict[str, float]:
    reference = np.asarray(reference_logits, np.float64)
    candidate = np.asarray(candidate_logits, np.float64)
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("router logits must be equal [tokens,experts] matrices")
    if top_k < 1 or top_k >= reference.shape[1]:
        raise ValueError("top_k must lie in [1, experts-1]")
    reference_ids = np.argsort(reference, axis=1, kind="stable")[:, -top_k:]
    candidate_ids = np.argsort(candidate, axis=1, kind="stable")[:, -top_k:]
    overlap = np.asarray([
        len(set(a.tolist()) & set(b.tolist())) / float(top_k)
        for a, b in zip(reference_ids, candidate_ids)
    ])
    sorted_reference = np.sort(reference, axis=1)
    sorted_candidate = np.sort(candidate, axis=1)
    reference_margin = sorted_reference[:, -top_k] - sorted_reference[:, -top_k - 1]
    candidate_margin = sorted_candidate[:, -top_k] - sorted_candidate[:, -top_k - 1]
    return {
        "router_logit_kl": categorical_kl(reference, candidate),
        "top8_overlap": float(np.mean(overlap)),
        "route_churn": float(1.0 - np.mean(overlap)),
        "route_top1_agreement": float(
            np.mean(reference_ids[:, -1] == candidate_ids[:, -1])
        ),
        "route_ordered_top8_agreement": float(
            np.mean(np.all(reference_ids == candidate_ids, axis=1))
        ),
        "router_margin_mean": float(np.mean(candidate_margin)),
        "router_margin_change_mean": float(np.mean(candidate_margin - reference_margin)),
    }


def route_boundary_metrics(
    reference_logits: np.ndarray,
    candidate_logits: np.ndarray,
    top_k: int = TOP_K,
) -> dict[str, float]:
    """Describe discrete route changes and probability mass moved between experts.

    The existing :func:`route_metrics` intentionally preserves the pilot schema.
    This companion provides the more diagnostic decomposition needed by the
    same-host frozen-route controls.
    """

    reference = np.asarray(reference_logits, np.float64)
    candidate = np.asarray(candidate_logits, np.float64)
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("router logits must be equal [tokens,experts] matrices")
    if top_k < 1 or top_k >= reference.shape[1]:
        raise ValueError("top_k must lie in [1, experts-1]")

    reference_ids = np.argsort(reference, axis=1, kind="stable")[:, -top_k:][:, ::-1]
    candidate_ids = np.argsort(candidate, axis=1, kind="stable")[:, -top_k:][:, ::-1]
    same_order = np.all(reference_ids == candidate_ids, axis=1)
    same_set = np.asarray([
        set(first.tolist()) == set(second.tolist())
        for first, second in zip(reference_ids, candidate_ids)
    ])
    same_top1 = reference_ids[:, 0] == candidate_ids[:, 0]

    reference_probability = softmax(reference)
    candidate_probability = softmax(candidate)
    reference_top = np.take_along_axis(reference_probability, reference_ids, axis=1)
    candidate_top = np.take_along_axis(candidate_probability, candidate_ids, axis=1)
    reference_top /= reference_top.sum(axis=1, keepdims=True)
    candidate_top /= candidate_top.sum(axis=1, keepdims=True)
    reference_sparse = np.zeros_like(reference_probability)
    candidate_sparse = np.zeros_like(candidate_probability)
    np.put_along_axis(reference_sparse, reference_ids, reference_top, axis=1)
    np.put_along_axis(candidate_sparse, candidate_ids, candidate_top, axis=1)
    mass_churn = 0.5 * np.sum(
        np.abs(reference_sparse - candidate_sparse), axis=1,
    )

    return {
        "route_no_change_fraction": float(np.mean(same_order)),
        "route_order_only_change_fraction": float(np.mean(same_set & ~same_order)),
        "route_membership_change_fraction": float(np.mean(~same_set)),
        "route_top1_change_fraction": float(np.mean(~same_top1)),
        "router_mass_churn": float(np.mean(mass_churn)),
    }


def token_quality_metrics(
    reference_logits: np.ndarray,
    candidate_logits: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    reference = np.asarray(reference_logits, np.float64)
    candidate = np.asarray(candidate_logits, np.float64)
    targets = np.asarray(labels, np.int64).reshape(-1)
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("language-model logits must be equal [tokens,vocab] matrices")
    if targets.shape != (reference.shape[0],):
        raise ValueError("labels must contain one target per logit row")
    reference_log_prob = reference - np.logaddexp.reduce(reference, axis=-1, keepdims=True)
    candidate_log_prob = candidate - np.logaddexp.reduce(candidate, axis=-1, keepdims=True)
    rows = np.arange(targets.size)
    delta_nll = float(np.mean(reference_log_prob[rows, targets] - candidate_log_prob[rows, targets]))
    return {
        "delta_nll": delta_nll,
        "ppl_ratio": float(np.exp(delta_nll)),
        "logit_kl": categorical_kl(reference, candidate),
        "top1_agreement": float(np.mean(np.argmax(reference, axis=-1) == np.argmax(candidate, axis=-1))),
    }
