"""Exact token-dependent geometry for physical nested D1 page states.

Static page geometry alone does not determine a page's effect.  Gate/up/down
responses are synthesized for the current activation, and every one-bit move
is evaluated from its current complete unit state.  The vectorized scorers are
oracle conveniences; a runtime predictor is intentionally out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .causal_replay import qenergy_damage
from .d1_nested_allocation import (
    EXPERTS_PER_GROUP,
    PhysicalPageMove,
    validate_states,
)
from .split_interaction_field import (
    SPLIT_STATE_DOWN_HIGH,
    SPLIT_STATE_HIDDEN_INDEX,
    SplitProjectionResponses,
    split_state_output,
)


__all__ = ["TokenStateGeometry"]


@dataclass(frozen=True)
class TokenStateGeometry:
    """Eight-expert exact response geometry for one token/layer group."""

    responses: tuple[SplitProjectionResponses, ...]
    execution_weights: np.ndarray
    proxy: np.ndarray | None
    beta: float
    execution_sum_tolerance: float = 0.003
    # One geometry instance belongs to one token/layer group. Complete state
    # tables recur heavily across the five oracle arms and eta ablations, so
    # retaining their exact synthesized output is both bounded and much
    # cheaper than re-running all eight expert reconstructions. These caches
    # are performance-only: keys contain the complete physical state bytes
    # and returned arrays are immutable.
    _output_delta_cache: dict[bytes, np.ndarray] = field(
        default_factory=dict, init=False, repr=False, compare=False,
    )
    _local_damage_cache: dict[bytes, float] = field(
        default_factory=dict, init=False, repr=False, compare=False,
    )
    _legacy_damage_cache: dict[tuple[bytes, bytes], float] = field(
        default_factory=dict, init=False, repr=False, compare=False,
    )
    _output_delta_cache_hits: list[int] = field(
        default_factory=lambda: [0], init=False, repr=False, compare=False,
    )
    _local_damage_cache_hits: list[int] = field(
        default_factory=lambda: [0], init=False, repr=False, compare=False,
    )
    _legacy_damage_cache_hits: list[int] = field(
        default_factory=lambda: [0], init=False, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        responses = tuple(self.responses)
        if len(responses) != EXPERTS_PER_GROUP:
            raise ValueError("token geometry requires eight expert responses")
        if any(response.units != 512 for response in responses):
            raise ValueError("token geometry requires 512 units per expert")
        widths = {int(np.asarray(response.down2).shape[1]) for response in responses}
        if len(widths) != 1:
            raise ValueError("expert output widths changed")
        weights = np.asarray(self.execution_weights, np.float64).reshape(-1)
        tolerance = float(self.execution_sum_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("execution-weight sum tolerance must be finite nonnegative")
        if (
            weights.shape != (EXPERTS_PER_GROUP,)
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
            or not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=tolerance)
        ):
            raise ValueError("execution weights must be positive raw top-8 weights within sum tolerance")
        beta = float(self.beta)
        if not np.isfinite(beta) or beta < 0.0:
            raise ValueError("beta must be finite nonnegative")
        proxy = None if self.proxy is None else np.asarray(self.proxy, np.float64)
        width = next(iter(widths))
        if proxy is not None:
            if proxy.ndim == 1:
                proxy = proxy[:, None]
            if proxy.ndim != 2 or proxy.shape[0] != width or np.any(~np.isfinite(proxy)):
                raise ValueError("proxy must be finite [output,rank]")
            proxy = proxy.copy()
            proxy.setflags(write=False)
        weights = weights.copy()
        weights.setflags(write=False)
        object.__setattr__(self, "responses", responses)
        object.__setattr__(self, "execution_weights", weights)
        object.__setattr__(self, "proxy", proxy)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "execution_sum_tolerance", tolerance)

    @property
    def output_width(self) -> int:
        return int(np.asarray(self.responses[0].down2).shape[1])

    def expert_output(self, expert: int, states: Any) -> np.ndarray:
        index = int(expert)
        if index < 0 or index >= EXPERTS_PER_GROUP:
            raise ValueError("expert row lies outside routed group")
        row = np.asarray(states)
        if row.shape != (512,):
            raise ValueError("expert state row must contain 512 units")
        return np.asarray(split_state_output(self.responses[index], row), np.float64)

    def output_delta(self, states: Any) -> np.ndarray:
        """Return routed approximate-minus-Q4 output before BF16-once rounding."""

        value = validate_states(states)
        key = np.asarray(value, np.uint8).tobytes(order="C")
        cached = self._output_delta_cache.get(key)
        if cached is not None:
            self._output_delta_cache_hits[0] += 1
            return cached
        result = np.zeros(self.output_width, np.float64)
        for expert, response in enumerate(self.responses):
            approximate = split_state_output(response, value[expert])
            result += float(self.execution_weights[expert]) * (
                np.asarray(approximate, np.float64)
                - np.asarray(response.target_output, np.float64)
            )
        result.setflags(write=False)
        self._output_delta_cache[key] = result
        return result

    def local_damage(self, states: Any) -> float:
        value = validate_states(states)
        key = np.asarray(value, np.uint8).tobytes(order="C")
        cached = self._local_damage_cache.get(key)
        if cached is not None:
            self._local_damage_cache_hits[0] += 1
            return cached
        damage = qenergy_damage(self.output_delta(value), self.proxy, self.beta)
        self._local_damage_cache[key] = damage
        return damage

    def local_damages_from_output_deltas(
        self,
        output_deltas: Any,
        *,
        chunk_size: int = 256,
    ) -> np.ndarray:
        """Evaluate the exact qmetric for a batch of synthesized errors.

        This is the vectorized equivalent of calling ``qenergy_damage`` for
        every row. It is used by matched-swap search after forming the exact
        candidate error as incumbent + removal + addition; no approximation
        or route signal enters the local guard.
        """

        values = np.asarray(output_deltas, np.float64)
        if values.ndim == 1:
            values = values[None, :]
        chunk = int(chunk_size)
        if (
            values.ndim != 2
            or values.shape[1] != self.output_width
            or np.any(~np.isfinite(values))
            or chunk < 1
        ):
            raise ValueError(
                "output deltas must be finite [candidate,output] and chunk positive"
            )
        result = np.empty(values.shape[0], np.float64)
        for start in range(0, values.shape[0], chunk):
            selected = values[start : start + chunk]
            scores = np.einsum("ij,ij->i", selected, selected, optimize=True)
            if self.proxy is not None and self.beta != 0.0:
                projected = selected @ self.proxy
                scores += self.beta * np.einsum(
                    "ij,ij->i", projected, projected, optimize=True,
                )
            result[start : start + selected.shape[0]] = scores
        return result

    def cache_stats(self) -> dict[str, int]:
        """Return non-scientific counters for runner timing evidence."""

        return {
            "output_delta_states": len(self._output_delta_cache),
            "output_delta_hits": int(self._output_delta_cache_hits[0]),
            "local_damage_states": len(self._local_damage_cache),
            "local_damage_hits": int(self._local_damage_cache_hits[0]),
            "legacy_damage_states": len(self._legacy_damage_cache),
            "legacy_damage_hits": int(self._legacy_damage_cache_hits[0]),
        }

    def all_q2_damage(self) -> float:
        return self.local_damage(np.zeros((8, 512), np.uint8))

    def legacy_additive_damage(
        self, states: Any, selector_weights: Sequence[float] | np.ndarray,
    ) -> float:
        """Return the historical additive router-square damage for reporting."""

        value = validate_states(states)
        weights = np.asarray(selector_weights, np.float64).reshape(-1)
        if weights.shape != (8,) or np.any(weights <= 0.0):
            raise ValueError("selector weights must contain eight positive values")
        key = (
            np.asarray(value, np.uint8).tobytes(order="C"),
            np.asarray(weights, np.float64).tobytes(order="C"),
        )
        cached = self._legacy_damage_cache.get(key)
        if cached is not None:
            self._legacy_damage_cache_hits[0] += 1
            return cached
        total = 0.0
        for expert, response in enumerate(self.responses):
            residual = (
                np.asarray(response.target_output, np.float64)
                - split_state_output(response, value[expert])
            )
            total += float(weights[expert] ** 2) * qenergy_damage(
                residual, self.proxy, self.beta,
            )
        result = float(total)
        self._legacy_damage_cache[key] = result
        return result

    def _unit_output(self, expert: int, unit: int, state: int) -> np.ndarray:
        response = self.responses[int(expert)]
        hidden = float(np.asarray(response.hidden, np.float64)[
            int(unit), int(SPLIT_STATE_HIDDEN_INDEX[int(state)])
        ])
        matrix = response.down4 if bool(SPLIT_STATE_DOWN_HIGH[int(state)]) else response.down2
        return hidden * np.asarray(matrix, np.float64)[int(unit)]

    def _move_output_delta_unchecked(
        self,
        value: np.ndarray,
        move: PhysicalPageMove,
    ) -> np.ndarray:
        if int(value[move.expert, move.unit]) != move.source_state:
            raise ValueError("page move is stale for token geometry")
        return float(self.execution_weights[move.expert]) * (
            self._unit_output(
                move.expert, move.unit, move.destination_state,
            )
            - self._unit_output(move.expert, move.unit, move.source_state)
        )

    def move_output_delta(self, states: Any, move: PhysicalPageMove) -> np.ndarray:
        value = validate_states(states)
        return self._move_output_delta_unchecked(value, move)

    def move_output_deltas(
        self, states: Any, moves: Sequence[PhysicalPageMove],
    ) -> np.ndarray:
        value = validate_states(states)
        rows = []
        for move in moves:
            if int(value[move.expert, move.unit]) != move.source_state:
                raise ValueError("page move sequence contains a stale move")
            rows.append(self._move_output_delta_unchecked(value, move))
        if not rows:
            return np.empty((0, self.output_width), np.float64)
        return np.stack(rows)

    def score_moves(
        self,
        states: Any,
        moves: Sequence[PhysicalPageMove],
        *,
        chunk_size: int = 256,
    ) -> np.ndarray:
        """Score all one-bit destinations with exact combined local qmetric."""

        value = validate_states(states)
        candidates = tuple(moves)
        chunk = int(chunk_size)
        if chunk < 1:
            raise ValueError("chunk_size must be positive")
        baseline = self.output_delta(value)
        baseline_proxy = None if self.proxy is None else baseline @ self.proxy
        result = np.empty(len(candidates), np.float64)
        for start in range(0, len(candidates), chunk):
            selected = candidates[start : start + chunk]
            changes = self.move_output_deltas(value, selected)
            values = changes + baseline[None, :]
            scores = np.einsum("ij,ij->i", values, values, optimize=True)
            if self.proxy is not None and self.beta != 0.0:
                projected = changes @ self.proxy + baseline_proxy[None, :]
                scores += self.beta * np.einsum(
                    "ij,ij->i", projected, projected, optimize=True,
                )
            result[start : start + len(selected)] = scores
        return result

    def signed_move_effects(
        self,
        states: Any,
        moves: Sequence[PhysicalPageMove],
        sensitivities: np.ndarray,
        *,
        chunk_size: int = 256,
    ) -> np.ndarray:
        """Project exact token-dependent page effects onto D1 adjoints."""

        gradients = np.asarray(sensitivities, np.float64)
        if (
            gradients.ndim != 2
            or gradients.shape[1] != self.output_width
            or np.any(~np.isfinite(gradients))
        ):
            raise ValueError("D1 sensitivities must be finite [boundary,output]")
        candidates = tuple(moves)
        chunk = int(chunk_size)
        if chunk < 1:
            raise ValueError("chunk_size must be positive")
        result = np.empty((len(candidates), gradients.shape[0]), np.float64)
        value = validate_states(states)
        for start in range(0, len(candidates), chunk):
            selected = candidates[start : start + chunk]
            changes = self.move_output_deltas(value, selected)
            result[start : start + len(selected)] = changes @ gradients.T
        return result
