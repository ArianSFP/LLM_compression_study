"""Opt-in, decision-inert timing helpers for the nested D1 oracle runner.

The scientific artifacts deliberately exclude these measurements.  The
runner may write a separate profiling sidecar whose row order and labels are
canonical, while the measured wall durations are naturally host dependent.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from time import perf_counter
from typing import Any, Callable, Iterator, Mapping, Sequence


PROFILE_SCHEMA = "pr13_d1_nested_internal_timing_profile_v1"

_Scalar = str | int | float | bool | None


def _labels(value: Mapping[str, _Scalar] | None) -> tuple[tuple[str, _Scalar], ...]:
    if value is None:
        return ()
    rows: list[tuple[str, _Scalar]] = []
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        if not key:
            raise ValueError("profile label names must be nonempty")
        if not isinstance(raw_value, (str, int, float, bool, type(None))):
            raise TypeError("profile labels must be JSON scalar values")
        rows.append((key, raw_value))
    rows.sort(key=lambda item: item[0])
    return tuple(rows)


class TimingProfile:
    """Accumulate opt-in call counts and wall time under canonical labels."""

    def __init__(
        self,
        enabled: bool,
        *,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.enabled = bool(enabled)
        self._clock = clock
        self._rows: dict[
            tuple[str, tuple[tuple[str, _Scalar], ...]], dict[str, float | int]
        ] = {}

    def add(
        self,
        stage: str,
        seconds: float,
        *,
        calls: int = 1,
        items: int = 0,
        labels: Mapping[str, _Scalar] | None = None,
    ) -> None:
        """Add one aggregate without reading the clock when disabled."""

        if not self.enabled:
            return
        name = str(stage)
        duration = float(seconds)
        call_count = int(calls)
        item_count = int(items)
        if (
            not name
            or duration < 0.0
            or call_count < 0
            or item_count < 0
        ):
            raise ValueError("profile aggregate values must be nonnegative")
        key = (name, _labels(labels))
        row = self._rows.setdefault(
            key, {"seconds": 0.0, "calls": 0, "items": 0},
        )
        row["seconds"] = float(row["seconds"]) + duration
        row["calls"] = int(row["calls"]) + call_count
        row["items"] = int(row["items"]) + item_count

    @contextmanager
    def measure(
        self,
        stage: str,
        *,
        calls: int = 1,
        items: int = 0,
        labels: Mapping[str, _Scalar] | None = None,
    ) -> Iterator[None]:
        """Measure a block; disabled profiles add no clock calls."""

        if not self.enabled:
            yield
            return
        started = self._clock()
        try:
            yield
        finally:
            self.add(
                stage,
                max(0.0, self._clock() - started),
                calls=calls,
                items=items,
                labels=labels,
            )

    def snapshot(self) -> dict[str, Any]:
        """Return a canonical JSON-compatible representation."""

        entries = []
        for (stage, labels), aggregate in sorted(
            self._rows.items(),
            key=lambda item: (
                item[0][0],
                json.dumps(dict(item[0][1]), sort_keys=True, separators=(",", ":")),
            ),
        ):
            calls = int(aggregate["calls"])
            seconds = float(aggregate["seconds"])
            entries.append({
                "stage": stage,
                "labels": dict(labels),
                "calls": calls,
                "items": int(aggregate["items"]),
                "seconds": seconds,
                "mean_seconds_per_call": None if calls == 0 else seconds / calls,
            })
        return {
            "schema": PROFILE_SCHEMA,
            "enabled": self.enabled,
            "entries": entries,
        }


class ProfiledGeometry:
    """Transparent geometry proxy timing existing callback boundaries."""

    def __init__(
        self,
        geometry: Any,
        profile: TimingProfile,
        labels: Mapping[str, _Scalar],
    ) -> None:
        self._geometry = geometry
        self._profile = profile
        self._labels = dict(labels)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._geometry, name)

    def local_damage(self, states: Any) -> float:
        with self._profile.measure("local_damage", labels=self._labels):
            return self._geometry.local_damage(states)

    def score_moves(self, states: Any, moves: Sequence[Any], **kwargs: Any) -> Any:
        with self._profile.measure(
            "move_scoring",
            items=len(moves),
            labels=self._labels,
        ):
            return self._geometry.score_moves(states, moves, **kwargs)

    def signed_move_effects(
        self,
        states: Any,
        moves: Sequence[Any],
        sensitivities: Any,
        **kwargs: Any,
    ) -> Any:
        with self._profile.measure(
            "signed_move_effect_scoring",
            items=len(moves),
            labels=self._labels,
        ):
            return self._geometry.signed_move_effects(
                states, moves, sensitivities, **kwargs,
            )

    def output_delta(self, states: Any) -> Any:
        with self._profile.measure("output_delta", labels=self._labels):
            return self._geometry.output_delta(states)


def profiled_callable(
    function: Callable[..., Any],
    profile: TimingProfile,
    stage: str,
    *,
    labels: Mapping[str, _Scalar],
) -> Callable[..., Any]:
    """Return a transparent timed callback used only in profiling runs."""

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with profile.measure(stage, labels=labels):
            return function(*args, **kwargs)

    return wrapped


__all__ = [
    "PROFILE_SCHEMA",
    "TimingProfile",
    "ProfiledGeometry",
    "profiled_callable",
]
