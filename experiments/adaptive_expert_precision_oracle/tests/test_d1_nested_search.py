from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "src"))

from oracle_study.d1_nested_allocation import (  # noqa: E402
    physical_subset,
    state_page_count,
)
from oracle_study.d1_nested_search import (  # noqa: E402
    build_screened_d1_boundaries,
    exact_all_expert_d1_metrics,
    iterative_exact_repair,
    predicted_signed_margin_metrics,
    shortlist_matched_page_swaps,
    stable_top8,
)
from oracle_study.d1_nested_geometry import TokenStateGeometry  # noqa: E402
from oracle_study.split_interaction_field import (  # noqa: E402
    SplitProjectionResponses,
)


def _baseline_logits() -> np.ndarray:
    return -np.arange(256, dtype=np.float64) / 10.0


def test_stable_top8_uses_expert_index_for_exact_ties() -> None:
    tied = np.zeros(256, np.float64)
    assert stable_top8(tied).tolist() == list(range(8))
    tied[[17, 42]] = 1.0
    assert stable_top8(tied).tolist() == [17, 42, 0, 1, 2, 3, 4, 5]


def test_exact_all_256_metrics_pair_strongest_entrant_with_weakest_target() -> None:
    baseline = _baseline_logits()
    candidate = baseline.copy()
    candidate[6] = -2.0
    candidate[7] = -3.0
    candidate[8] = -0.50
    candidate[9] = -0.55

    metrics = exact_all_expert_d1_metrics(baseline, candidate)

    assert metrics.membership_crossings == 2
    assert metrics.entering_outsiders.tolist() == [8, 9]
    assert metrics.lost_target_experts.tolist() == [7, 6]
    assert metrics.violation_depth == pytest.approx((2.5) + (1.45))
    probability = np.exp(baseline - baseline.max())
    probability /= probability.sum()
    assert metrics.routing_mass_lost == pytest.approx(probability[[7, 6]].sum())
    assert metrics.target_set_margin == pytest.approx(-2.5)
    assert not metrics.safe

    safe = exact_all_expert_d1_metrics(baseline, baseline)
    assert safe.safe
    assert safe.membership_crossings == 0
    assert safe.violation_depth == 0.0
    assert safe.routing_mass_lost == 0.0
    assert safe.target_set_margin == pytest.approx(0.1)


def test_screened_projection_matches_explicit_candidate_logit_margins() -> None:
    rng = np.random.default_rng(18)
    baseline = _baseline_logits()
    sensitivities = rng.normal(size=(256, 3))
    delta = np.asarray([0.2, -0.3, 0.1])
    boundaries = build_screened_d1_boundaries(baseline, sensitivities)

    predicted = predicted_signed_margin_metrics(boundaries, delta)
    candidate = baseline + sensitivities @ delta
    expected = (
        candidate[boundaries.selected_expert_ids]
        - candidate[boundaries.outsider_expert_ids]
    )

    assert boundaries.selected_expert_ids.reshape(3, 8)[:, 0].tolist() == [5, 6, 7]
    assert boundaries.outsider_expert_ids.reshape(3, 8)[0].tolist() == list(range(8, 16))
    np.testing.assert_allclose(predicted.predicted_margins, expected, rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(
        predicted.signed_effects,
        expected - boundaries.baseline_margins,
        rtol=0.0,
        atol=2e-15,
    )


class _StubGeometry:
    output_width = 1

    def __init__(self, *, forbidden_unit: int | None = None) -> None:
        self.forbidden_unit = forbidden_unit
        self.signed_calls = 0

    def output_delta(self, states: np.ndarray) -> np.ndarray:
        return np.zeros(1, np.float64)

    @staticmethod
    def _addition_priority(move: object) -> float:
        if move.direction == "remove":
            return 0.0
        bit_rank = {1: 0.003, 2: 0.002, 4: 0.001}[move.bit]
        if move.expert == 0 and move.unit == 3:
            return 10.0 + bit_rank
        if move.expert == 0 and move.unit == 4:
            return 9.0 + bit_rank
        if move.expert == 0 and move.unit == 2:
            return 8.0 + bit_rank
        return 1.0 - 1e-5 * (move.expert * 512 + move.unit)

    def signed_move_effects(
        self,
        states: np.ndarray,
        moves: tuple[object, ...],
        sensitivities: np.ndarray,
    ) -> np.ndarray:
        self.signed_calls += 1
        values = np.asarray([self._addition_priority(move) for move in moves])
        return np.repeat(values[:, None], sensitivities.shape[0], axis=1)

    def local_damage(self, states: np.ndarray) -> float:
        if self.forbidden_unit is not None and int(states[0, self.forbidden_unit]) != 0:
            return 10.0
        if int(states[0, 3]) != 0:
            return 0.1
        if int(states[0, 4]) != 0:
            return 0.2
        return 0.3


def _screen() -> object:
    sensitivities = np.arange(256, dtype=np.float64)[:, None] / 256.0
    return build_screened_d1_boundaries(_baseline_logits(), sensitivities)


def _incumbent_and_core() -> tuple[np.ndarray, np.ndarray]:
    incumbent = np.zeros((8, 512), np.uint8)
    incumbent[0, 0] = 1
    incumbent[0, 1] = 1
    core = np.zeros_like(incumbent)
    return incumbent, core


def test_swap_shortlist_preserves_core_pages_distinct_units_and_exact_guard() -> None:
    incumbent, core = _incumbent_and_core()
    geometry = _StubGeometry(forbidden_unit=3)
    first = shortlist_matched_page_swaps(
        incumbent,
        core,
        geometry,
        _screen(),
        local_damage_limit=1.0,
    )
    second = shortlist_matched_page_swaps(
        incumbent,
        core,
        geometry,
        _screen(),
        local_damage_limit=1.0,
    )

    assert 0 < len(first) <= 8
    assert geometry.signed_calls == 4
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left.states, right.states)
        assert left.removal.sort_key == right.removal.sort_key
        assert left.addition.sort_key == right.addition.sort_key
    for candidate in first:
        assert physical_subset(core, candidate.states)
        assert state_page_count(candidate.states) == state_page_count(incumbent)
        assert (candidate.removal.expert, candidate.removal.unit) != (
            candidate.addition.expert,
            candidate.addition.unit,
        )
        assert candidate.local_damage <= 1.0
        assert int(candidate.states[0, 3]) == 0  # rejected by exact local guard


def test_vectorized_swap_shortlist_matches_scalar_reference_across_guards() -> None:
    rng = np.random.default_rng(712)
    responses = tuple(
        SplitProjectionResponses(
            hidden=rng.normal(size=(512, 4)),
            down2=rng.normal(scale=0.03, size=(512, 9)),
            down4=rng.normal(scale=0.03, size=(512, 9)),
        )
        for _ in range(8)
    )
    weights = np.arange(1, 9, dtype=np.float64)
    weights /= weights.sum()
    geometry = TokenStateGeometry(
        responses, weights, rng.normal(size=(9, 3)), 0.7,
    )

    class ScalarReference:
        """Expose only the pre-optimization geometry callback surface."""

        output_width = geometry.output_width

        @staticmethod
        def output_delta(states):
            return geometry.output_delta(states)

        @staticmethod
        def signed_move_effects(states, moves, sensitivities):
            return geometry.signed_move_effects(states, moves, sensitivities)

        @staticmethod
        def local_damage(states):
            return geometry.local_damage(states)

    incumbent = rng.integers(0, 8, size=(8, 512), dtype=np.uint8)
    core = np.bitwise_and(
        incumbent,
        rng.integers(0, 8, size=(8, 512), dtype=np.uint8),
    )
    sensitivities = rng.normal(size=(256, 9))
    boundaries = build_screened_d1_boundaries(_baseline_logits(), sensitivities)
    baseline_damage = geometry.local_damage(incumbent)
    for guard in (0.0, baseline_damage * 0.95, baseline_damage, baseline_damage * 1.2):
        optimized = shortlist_matched_page_swaps(
            incumbent,
            core,
            geometry,
            boundaries,
            local_damage_limit=guard,
            move_limit=12,
            candidate_limit=8,
        )
        reference = shortlist_matched_page_swaps(
            incumbent,
            core,
            ScalarReference(),  # type: ignore[arg-type]
            boundaries,
            local_damage_limit=guard,
            move_limit=12,
            candidate_limit=8,
        )
        assert len(optimized) == len(reference)
        for observed, expected in zip(optimized, reference, strict=True):
            np.testing.assert_array_equal(observed.states, expected.states)
            assert observed.removal.sort_key == expected.removal.sort_key
            assert observed.addition.sort_key == expected.addition.sort_key
            assert observed.local_damage == expected.local_damage
            np.testing.assert_array_equal(
                observed.predicted.predicted_margins,
                expected.predicted.predicted_margins,
            )


def _candidate_logits_for_crossings(crossings: int, *, shallow: bool = False) -> np.ndarray:
    baseline = _baseline_logits()
    if crossings == 0:
        return baseline
    candidate = baseline.copy()
    candidate[7] = -0.70 if shallow else -2.0
    candidate[8] = -0.65 if shallow else -0.50
    if crossings >= 2:
        candidate[6] = -2.5
        candidate[9] = -0.55
    return candidate


def test_iterative_repair_accepts_only_strict_exact_reductions_and_keeps_core() -> None:
    incumbent, core = _incumbent_and_core()
    geometry = _StubGeometry()
    baseline = _baseline_logits()

    def exact_replay(states: np.ndarray) -> object:
        helpful = int(states[0, 3] != 0) + int(states[0, 4] != 0)
        return exact_all_expert_d1_metrics(
            baseline, _candidate_logits_for_crossings(2 - helpful),
        )

    result = iterative_exact_repair(
        incumbent,
        core,
        geometry,
        _screen(),
        local_damage_limit=1.0,
        exact_replay=exact_replay,
        arm="arm4",
    )

    crossings = [result.initial_metrics.membership_crossings]
    crossings.extend(step.metrics.membership_crossings for step in result.steps)
    assert crossings == [2, 1, 0]
    assert result.stop_reason == "d1_safe"
    assert all(right < left for left, right in zip(crossings, crossings[1:]))
    assert physical_subset(core, result.final_state)
    assert state_page_count(result.final_state) == state_page_count(incumbent)


def test_arm5_uses_exact_severity_only_while_unsafe_and_safe_is_immutable() -> None:
    incumbent, core = _incumbent_and_core()
    baseline = _baseline_logits()

    def exact_replay(states: np.ndarray) -> object:
        if int(states[0, 3]) != 0:
            logits = _candidate_logits_for_crossings(1, shallow=False)
        elif int(states[0, 4]) != 0:
            logits = _candidate_logits_for_crossings(1, shallow=True)
        else:
            logits = _candidate_logits_for_crossings(2)
        return exact_all_expert_d1_metrics(baseline, logits)

    arm4 = iterative_exact_repair(
        incumbent,
        core,
        _StubGeometry(),
        _screen(),
        local_damage_limit=1.0,
        exact_replay=exact_replay,
        arm="arm4",
        maximum_rounds=1,
    )
    arm5 = iterative_exact_repair(
        incumbent,
        core,
        _StubGeometry(),
        _screen(),
        local_damage_limit=1.0,
        exact_replay=exact_replay,
        arm="arm5",
        maximum_rounds=1,
    )
    assert arm4.steps[0].addition.unit == 3  # lower exact local damage
    assert arm5.steps[0].addition.unit == 4  # shallower exact violation

    safe_state = incumbent.copy()
    safe_state[0, 0] = 0
    safe_state[0, 1] = 0
    safe_state[0, 3] = 1
    safe_state[0, 4] = 1
    calls = 0

    def safe_replay(states: np.ndarray) -> object:
        nonlocal calls
        calls += 1
        return exact_all_expert_d1_metrics(baseline, baseline)

    safe = iterative_exact_repair(
        safe_state,
        core,
        _StubGeometry(),
        _screen(),
        local_damage_limit=1.0,
        exact_replay=safe_replay,
        arm="arm5",
    )
    assert calls == 1
    assert safe.steps == ()
    assert safe.stop_reason == "incumbent_d1_safe"
    np.testing.assert_array_equal(safe.final_state, safe_state)
