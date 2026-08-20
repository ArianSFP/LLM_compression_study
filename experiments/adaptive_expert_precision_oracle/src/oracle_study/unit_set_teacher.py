"""Training-only coherent-unit interaction teachers.

The deployable A/B/C statistic stores only each correction's diagonal energy.
Training against complete-expert damage also needs cross-unit interactions.
Three static down-column Gram terms and the Q2/Q4 hidden scalars reconstruct
those interactions without storing a 512 by 2048 correction per activation.

The dense matrices here are teacher/oracle state, never deployable metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DownMetricGram:
    """Static down-column terms under M = I + beta P P.T."""

    q44: np.ndarray
    q42: np.ndarray
    q22: np.ndarray

    def __post_init__(self) -> None:
        values = tuple(np.asarray(value) for value in (self.q44, self.q42, self.q22))
        if any(value.ndim != 2 or value.shape[0] != value.shape[1] for value in values):
            raise ValueError("down metric terms must be square matrices")
        if len({value.shape for value in values}) != 1:
            raise ValueError("down metric terms must have equal shapes")

    @property
    def units(self) -> int:
        return int(np.asarray(self.q44).shape[0])

    @property
    def teacher_bytes(self) -> int:
        return int(sum(np.asarray(value).nbytes for value in (self.q44, self.q42, self.q22)))


def down_metric_gram(
    down2: np.ndarray,
    down4: np.ndarray,
    *,
    proxy: np.ndarray | None = None,
    beta: float = 0.0,
    dtype: np.dtype | type = np.float32,
) -> DownMetricGram:
    """Build three static down-column Gram terms."""

    d2 = np.asarray(down2, dtype=np.float64)
    d4 = np.asarray(down4, dtype=np.float64)
    if d2.ndim != 2 or d4.shape != d2.shape:
        raise ValueError("down2/down4 must have equal [output, unit] shapes")
    if beta < 0.0 or (beta != 0.0 and proxy is None):
        raise ValueError("nonnegative beta requires a proxy when nonzero")
    q44 = d4.T @ d4
    q42 = d4.T @ d2
    q22 = d2.T @ d2
    if proxy is not None and beta != 0.0:
        p = np.asarray(proxy, dtype=np.float64)
        if p.ndim == 1:
            p = p[:, None]
        if p.ndim != 2 or p.shape[0] != d2.shape[0]:
            raise ValueError("proxy output width does not match down matrices")
        projected4 = d4.T @ p
        projected2 = d2.T @ p
        q44 += float(beta) * (projected4 @ projected4.T)
        q42 += float(beta) * (projected4 @ projected2.T)
        q22 += float(beta) * (projected2 @ projected2.T)
    return DownMetricGram(
        q44=np.asarray(q44, dtype=dtype),
        q42=np.asarray(q42, dtype=dtype),
        q22=np.asarray(q22, dtype=dtype),
    )


def unit_correction_gram(
    h2: np.ndarray,
    h4: np.ndarray,
    metadata: DownMetricGram,
    *,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Reconstruct K[i,j] = c_i.T M c_j from hidden scalars."""

    low = np.asarray(h2, dtype=np.float64).reshape(-1)
    high = np.asarray(h4, dtype=np.float64).reshape(-1)
    if low.shape != high.shape or low.size != metadata.units:
        raise ValueError("hidden scalars do not match down Gram metadata")
    q44 = np.asarray(metadata.q44, dtype=np.float64)
    q42 = np.asarray(metadata.q42, dtype=np.float64)
    q22 = np.asarray(metadata.q22, dtype=np.float64)
    gram = (
        high[:, None] * q44 * high[None, :]
        - high[:, None] * q42 * low[None, :]
        - low[:, None] * q42.T * high[None, :]
        + low[:, None] * q22 * low[None, :]
    )
    return np.asarray(0.5 * (gram + gram.T), dtype=dtype)


def set_damage(gram: np.ndarray, applied_mask: np.ndarray) -> float:
    """Return exact remaining complete-expert qenergy for a mask."""

    matrix = np.asarray(gram, dtype=np.float64)
    mask = np.asarray(applied_mask, dtype=np.float64).reshape(-1)
    if matrix.shape != (mask.size, mask.size):
        raise ValueError("mask and Gram have incompatible shapes")
    residual = 1.0 - mask
    return float(residual @ matrix @ residual)


def set_gain(gram: np.ndarray, applied_mask: np.ndarray) -> float:
    matrix = np.asarray(gram, dtype=np.float64)
    ones = np.ones(matrix.shape[0], dtype=np.float64)
    return float(ones @ matrix @ ones) - set_damage(matrix, applied_mask)


@dataclass(frozen=True)
class UnitTeacherPath:
    order: np.ndarray
    marginal_gain: np.ndarray
    cumulative_gain: np.ndarray
    remaining_damage: np.ndarray

    def mask(self, units: int, count: int) -> np.ndarray:
        if count < 0 or count > len(self.order):
            raise ValueError("count lies outside the teacher path")
        result = np.zeros(int(units), dtype=bool)
        result[np.asarray(self.order[:count], dtype=np.int64)] = True
        return result

    def best_prefix(self, maximum_count: int) -> int:
        maximum = min(int(maximum_count), len(self.order))
        values = np.concatenate(([0.0], np.asarray(self.cumulative_gain[:maximum], np.float64)))
        return int(np.argmax(values))


def _candidate_mask(units: int, candidates: Iterable[int] | None) -> np.ndarray:
    if candidates is None:
        return np.ones(units, dtype=bool)
    result = np.zeros(units, dtype=bool)
    indices = np.asarray(tuple(int(value) for value in candidates), dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= units) or len(np.unique(indices)) != len(indices):
        raise ValueError("candidate IDs must be unique and in range")
    result[indices] = True
    return result


def _fixed_greedy(
    gram: np.ndarray,
    count: int,
    *,
    candidates: Iterable[int] | None,
    contained_target: bool,
) -> UnitTeacherPath:
    matrix = np.asarray(gram, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Gram must be square")
    units = matrix.shape[0]
    allowed = _candidate_mask(units, candidates)
    if count < 0 or count > int(np.sum(allowed)):
        raise ValueError("count exceeds eligible candidates")
    residual_coefficients = allowed.astype(np.float64) if contained_target else np.ones(units)
    correlation = matrix @ residual_coefficients
    diagonal = np.diag(matrix)
    order: list[int] = []
    gains: list[float] = []
    eligible = allowed.copy()
    for _ in range(int(count)):
        marginal = 2.0 * correlation - diagonal
        chosen = int(np.argmax(np.where(eligible, marginal, -np.inf)))
        order.append(chosen)
        gains.append(float(marginal[chosen]))
        eligible[chosen] = False
        residual_coefficients[chosen] = 0.0
        correlation -= matrix[:, chosen]
    cumulative = np.cumsum(np.asarray(gains, dtype=np.float64))
    initial = allowed.astype(np.float64) if contained_target else np.ones(units)
    base_damage = float(initial @ matrix @ initial)
    return UnitTeacherPath(
        order=np.asarray(order, dtype=np.int64),
        marginal_gain=np.asarray(gains, dtype=np.float64),
        cumulative_gain=cumulative,
        remaining_damage=base_damage - cumulative,
    )


def exact_full_target_fixed_greedy(
    gram: np.ndarray,
    count: int,
    *,
    candidates: Iterable[int] | None = None,
) -> UnitTeacherPath:
    """Fixed greedy against the full Q4 target residual.

    Candidate restriction changes only what may be applied; omitted
    corrections remain in the target. This is a teacher oracle unless a
    deployable synopsis provides the omitted target effects.
    """

    return _fixed_greedy(
        gram, count, candidates=candidates, contained_target=False,
    )


def exact_contained_target_fixed_greedy(
    gram: np.ndarray,
    candidates: Iterable[int],
    count: int,
) -> UnitTeacherPath:
    """Fixed greedy using only fetched corrections as its target.

    This can run after candidate packets are fetched, but ignores unfetched
    corrections and is not full-target marginal reranking.
    """

    return _fixed_greedy(
        gram, count, candidates=candidates, contained_target=True,
    )


def selected_exclusion_regret(
    gram: np.ndarray,
    apply_count: int,
    *,
    boundary_width: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure teacher-set gain lost when selected units are forbidden."""

    matrix = np.asarray(gram, dtype=np.float64)
    baseline = exact_full_target_fixed_greedy(matrix, int(apply_count))
    start = 0 if boundary_width is None else max(0, int(apply_count) - int(boundary_width))
    selected = np.asarray(baseline.order[start:apply_count], dtype=np.int64)
    baseline_gain = float(baseline.cumulative_gain[apply_count - 1]) if apply_count else 0.0
    regrets = []
    all_units = np.arange(matrix.shape[0], dtype=np.int64)
    for forbidden in selected:
        allowed = all_units[all_units != int(forbidden)]
        alternative = exact_full_target_fixed_greedy(
            matrix, int(apply_count), candidates=allowed,
        )
        alternative_gain = (
            float(alternative.cumulative_gain[apply_count - 1]) if apply_count else 0.0
        )
        regrets.append(max(0.0, baseline_gain - alternative_gain))
    return selected, np.asarray(regrets, dtype=np.float64)


__all__ = [
    "DownMetricGram",
    "UnitTeacherPath",
    "down_metric_gram",
    "exact_contained_target_fixed_greedy",
    "exact_full_target_fixed_greedy",
    "selected_exclusion_regret",
    "set_damage",
    "set_gain",
    "unit_correction_gram",
]
