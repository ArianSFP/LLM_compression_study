"""Pure helpers for same-host causal dose and routing controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


PRE_RESIDUAL = "pre_residual_routed_replacement"
POST_XPLUS_BF16 = "post_xplus_bf16_delta"
POST_XPLUS_FP32 = "post_xplus_fp32_once"
INJECTION_MODES = (PRE_RESIDUAL, POST_XPLUS_BF16, POST_XPLUS_FP32)

LIVE_ROUTES = "live"
FROZEN_SET = "frozen_set_live_weights"
FULLY_FROZEN = "fully_frozen"
ROUTE_MODES = (LIVE_ROUTES, FROZEN_SET, FULLY_FROZEN)


@dataclass(frozen=True, order=True)
class ControlCase:
    """One intervention case in the predeclared sentinel grid."""

    injection_mode: str
    route_mode: str
    alpha: float

    def __post_init__(self) -> None:
        if self.injection_mode not in INJECTION_MODES:
            raise ValueError(f"unknown injection mode: {self.injection_mode}")
        if self.route_mode not in ROUTE_MODES:
            raise ValueError(f"unknown route mode: {self.route_mode}")
        if not np.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError("alpha must be non-negative finite")


def build_control_cases(
    dose_alphas: Sequence[float],
    *,
    include_zero_gates: bool = True,
) -> tuple[ControlCase, ...]:
    """Build the frozen scientific grid without duplicate alpha-one cases."""

    positive = tuple(float(value) for value in dose_alphas)
    if not positive or any(not np.isfinite(value) or value <= 0.0 for value in positive):
        raise ValueError("dose alphas must be positive finite")
    if len(set(positive)) != len(positive):
        raise ValueError("dose alphas must be unique")
    cases = []
    for route_mode in ROUTE_MODES:
        if include_zero_gates:
            cases.append(ControlCase(PRE_RESIDUAL, route_mode, 0.0))
        cases.extend(
            ControlCase(PRE_RESIDUAL, route_mode, alpha)
            for alpha in positive
        )
    for injection_mode in (POST_XPLUS_BF16, POST_XPLUS_FP32):
        cases.append(ControlCase(injection_mode, LIVE_ROUTES, 1.0))
    return tuple(cases)


def selector_and_execution_router_weights(
    weights: Sequence[float] | np.ndarray,
    *,
    historical_sum_atol: float,
    execution_sum_atol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate PR #13 normalized weights from BF16 backend execution weights."""

    execution = np.asarray(weights, np.float64).reshape(-1)
    if (
        execution.shape != (8,)
        or np.any(~np.isfinite(execution))
        or np.any(execution <= 0.0)
    ):
        raise ValueError("router execution weights must be eight positive finite values")
    if (
        not np.isfinite(historical_sum_atol)
        or historical_sum_atol < 0.0
        or not np.isfinite(execution_sum_atol)
        or execution_sum_atol < historical_sum_atol
    ):
        raise ValueError("router sum tolerances are invalid")
    total = float(execution.sum())
    error = abs(total - 1.0)
    if error <= historical_sum_atol:
        selector = execution.copy()
    elif error <= execution_sum_atol:
        selector = execution / total
    else:
        raise ValueError(
            f"router execution weights exceed the sum tolerance: {error:.9g}",
        )
    if not np.isclose(selector.sum(), 1.0, rtol=0.0, atol=historical_sum_atol):
        raise RuntimeError("normalized selector weights do not sum to one")
    return selector, execution


def _ordered_bfloat16(bits: np.ndarray) -> np.ndarray:
    value = np.asarray(bits, np.uint16)
    unsigned = value.astype(np.uint32)
    return np.where(
        (unsigned & 0x8000) != 0,
        0xFFFF - unsigned,
        unsigned + 0x8000,
    ).astype(np.int64)


def bfloat16_ulp_metrics(
    reference_bits: np.ndarray,
    candidate_bits: np.ndarray,
    numeric_equal: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Summarize BF16 representable-step changes from raw uint16 bit patterns."""

    reference = np.asarray(reference_bits, np.uint16)
    candidate = np.asarray(candidate_bits, np.uint16)
    if reference.shape != candidate.shape:
        raise ValueError("BF16 bit arrays must have equal shape")
    if reference.size == 0:
        raise ValueError("BF16 bit arrays must be non-empty")
    distance = np.abs(_ordered_bfloat16(candidate) - _ordered_bfloat16(reference))
    if numeric_equal is not None:
        equal = np.asarray(numeric_equal, bool)
        if equal.shape != reference.shape:
            raise ValueError("numeric equality mask shape changed")
        distance = np.where(equal, 0, distance)
    changed = distance != 0
    return {
        "coordinates": int(distance.size),
        "unchanged_fraction": float(np.mean(~changed)),
        "one_ulp_fraction": float(np.mean(distance == 1)),
        "multi_ulp_fraction": float(np.mean(distance > 1)),
        "mean_ulp_distance": float(np.mean(distance)),
        "max_ulp_distance": int(np.max(distance, initial=0)),
    }


def first_changed_layer(
    layers: Iterable[tuple[int, float]],
    *,
    threshold: float = 0.0,
) -> int | None:
    """Return the first layer whose non-negative change fraction exceeds a threshold."""

    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    first = None
    for layer, fraction in layers:
        value = float(fraction)
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("change fractions must be finite values in [0,1]")
        if value > threshold and (first is None or int(layer) < first):
            first = int(layer)
    return first
