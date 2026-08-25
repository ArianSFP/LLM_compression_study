from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "src"))

from oracle_study.d1_nested_profile import (  # noqa: E402
    PROFILE_SCHEMA,
    ProfiledGeometry,
    TimingProfile,
    profiled_callable,
)


class _Clock:
    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return next(self.values)


class _Geometry:
    output_width = 3

    @staticmethod
    def local_damage(states):
        return float(np.asarray(states).sum())

    @staticmethod
    def score_moves(states, moves, **kwargs):
        return np.arange(len(moves), dtype=np.float64)

    @staticmethod
    def signed_move_effects(states, moves, sensitivities, **kwargs):
        return np.ones((len(moves), np.asarray(sensitivities).shape[0]))

    @staticmethod
    def output_delta(states):
        return np.asarray(states, np.float64)


def test_disabled_profile_is_clock_and_output_inert() -> None:
    clock = _Clock([])
    profile = TimingProfile(False, clock=clock)
    with profile.measure("ignored", labels={"group": 0}):
        pass
    profile.add("ignored", 1.0)
    assert clock.calls == 0
    assert profile.snapshot() == {
        "schema": PROFILE_SCHEMA,
        "enabled": False,
        "entries": [],
    }


def test_profile_snapshot_has_canonical_rows_and_exact_aggregates() -> None:
    clock = _Clock([10.0, 10.25, 20.0, 20.5])
    profile = TimingProfile(True, clock=clock)
    with profile.measure("z", items=4, labels={"window": 32, "arm": "arm4"}):
        pass
    with profile.measure("a", labels={"group": 1}):
        pass
    profile.add("z", 0.75, calls=3, items=6, labels={"arm": "arm4", "window": 32})

    snapshot = profile.snapshot()
    assert [row["stage"] for row in snapshot["entries"]] == ["a", "z"]
    assert snapshot["entries"][0] == {
        "stage": "a",
        "labels": {"group": 1},
        "calls": 1,
        "items": 0,
        "seconds": 0.5,
        "mean_seconds_per_call": 0.5,
    }
    assert snapshot["entries"][1] == {
        "stage": "z",
        "labels": {"arm": "arm4", "window": 32},
        "calls": 4,
        "items": 10,
        "seconds": 1.0,
        "mean_seconds_per_call": 0.25,
    }


def test_geometry_and_callable_wrappers_preserve_values() -> None:
    clock = _Clock([float(index) for index in range(12)])
    profile = TimingProfile(True, clock=clock)
    raw = _Geometry()
    wrapped = ProfiledGeometry(raw, profile, {"arm": "arm4"})
    states = np.asarray([1.0, 2.0, 3.0])

    assert wrapped.output_width == raw.output_width
    assert wrapped.local_damage(states) == raw.local_damage(states)
    np.testing.assert_array_equal(
        wrapped.score_moves(states, ("a", "b")),
        raw.score_moves(states, ("a", "b")),
    )
    np.testing.assert_array_equal(
        wrapped.signed_move_effects(states, ("a",), np.ones((2, 3))),
        raw.signed_move_effects(states, ("a",), np.ones((2, 3))),
    )
    np.testing.assert_array_equal(
        wrapped.output_delta(states), raw.output_delta(states),
    )
    identity = profiled_callable(lambda value: value, profile, "callback", labels={"arm": "arm4"})
    assert identity(states) is states

    rows = {row["stage"]: row for row in profile.snapshot()["entries"]}
    assert rows["move_scoring"]["items"] == 2
    assert rows["signed_move_effect_scoring"]["items"] == 1
    assert rows["callback"]["calls"] == 1
