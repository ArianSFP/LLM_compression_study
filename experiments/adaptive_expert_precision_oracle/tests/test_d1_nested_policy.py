from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "src"))

from oracle_study.d1_nested_allocation import (  # noqa: E402
    DOWN_BIT,
    add_only_local_completion,
    apply_page_move,
    physical_subset,
    state_page_count,
)
from oracle_study.d1_nested_policy import (  # noqa: E402
    Arm3HighPathHint,
    NestedPolicyMemo,
    canonical_add_only_moves,
    orchestrate_nested_policy_pair,
    replay_add_only_moves,
)
from oracle_study.d1_nested_search import (  # noqa: E402
    build_screened_d1_boundaries,
    exact_all_expert_d1_metrics,
)


IDS = np.arange(10, 18, dtype=np.int64)


def _states(*units: int) -> np.ndarray:
    result = np.zeros((8, 512), np.int64)
    for unit in units:
        result[0, int(unit)] |= DOWN_BIT
    return result


def _baseline_logits() -> np.ndarray:
    return -np.arange(256, dtype=np.float64) / 10.0


def _candidate_logits(crossings: int, *, shallow: bool = False) -> np.ndarray:
    baseline = _baseline_logits()
    if crossings == 0:
        return baseline
    result = baseline.copy()
    result[7] = -0.70 if shallow else -2.0
    result[8] = -0.65 if shallow else -0.50
    if crossings >= 2:
        result[6] = -2.5
        result[9] = -0.55
    return result


def _metrics(crossings: int, *, shallow: bool = False):
    baseline = _baseline_logits()
    candidate = _candidate_logits(crossings, shallow=shallow)
    return exact_all_expert_d1_metrics(
        baseline,
        candidate,
        baseline_top8=np.argsort(-baseline, kind="stable")[:8],
        candidate_top8=np.argsort(-candidate, kind="stable")[:8],
    )


def _screen():
    sensitivities = np.arange(256, dtype=np.float64)[:, None] / 256.0
    baseline = _baseline_logits()
    order = np.argsort(-baseline, kind="stable")
    return build_screened_d1_boundaries(
        baseline,
        sensitivities,
        selected_expert_ids=order[5:8],
        outsider_expert_ids=order[8:16],
    )


class _ToyGeometry:
    output_width = 1

    def __init__(
        self,
        target_weights: dict[tuple[int, int, int], float],
        *,
        addition_priority: dict[int, float] | None = None,
        extra_cost: float = 1.0,
    ) -> None:
        self.target_weights = dict(target_weights)
        self.addition_priority = dict(addition_priority or {})
        self.extra_cost = float(extra_cost)

    @staticmethod
    def _active(states: np.ndarray) -> set[tuple[int, int, int]]:
        value = np.asarray(states, np.int64)
        result = set()
        for expert, unit in np.argwhere(value != 0):
            state = int(value[expert, unit])
            for bit in (1, 2, 4):
                if state & bit:
                    result.add((int(expert), int(unit), bit))
        return result

    def local_damage(self, states: np.ndarray) -> float:
        active = self._active(states)
        target = set(self.target_weights)
        missing = sum(self.target_weights[key] for key in target - active)
        extra = self.extra_cost * len(active - target)
        return float(missing + extra)

    def score_moves(self, states, moves):
        return np.asarray([
            self.local_damage(apply_page_move(states, move)) for move in moves
        ])

    def output_delta(self, states):
        return np.zeros(1, np.float64)

    def signed_move_effects(self, states, moves, sensitivities):
        values = []
        for move in moves:
            if move.direction == "remove":
                value = 0.0
            elif (move.expert, move.unit, move.bit) not in self.target_weights:
                value = 0.001
            else:
                value = self.addition_priority.get(move.unit, 0.001)
            values.append(value)
        return np.repeat(
            np.asarray(values, np.float64)[:, None],
            np.asarray(sensitivities).shape[0],
            axis=1,
        )


class _StagedToyGeometry(_ToyGeometry):
    """Expose a second repair page only after accepting the first one."""

    def signed_move_effects(self, states, moves, sensitivities):
        values = []
        first_present = bool(np.asarray(states, np.int64)[0, 3] & DOWN_BIT)
        for move in moves:
            page = (move.expert, move.unit, move.bit)
            if move.direction == "remove":
                value = 0.0
            elif page == (0, 3, DOWN_BIT) and not first_present:
                value = 20.0
            elif page == (0, 4, DOWN_BIT) and first_present:
                value = 20.0
            elif page == (0, 4, DOWN_BIT):
                value = -20.0
            else:
                value = 0.001
            values.append(value)
        return np.repeat(
            np.asarray(values, np.float64)[:, None],
            np.asarray(sensitivities).shape[0],
            axis=1,
        )


class _CountingToyGeometry(_ToyGeometry):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.score_move_calls = 0

    def score_moves(self, states, moves):
        self.score_move_calls += 1
        return super().score_moves(states, moves)


def _run(
    core: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    geometry: _ToyGeometry,
    replay,
    *,
    arm: str = "arm4",
    eta: float = 0.1,
    j_q2: float = 10.0,
    refresh: int = 2,
    memo: NestedPolicyMemo | None = None,
    high_path_hint: Arm3HighPathHint | None = None,
):
    return orchestrate_nested_policy_pair(
        core,
        low,
        high,
        IDS,
        geometry,
        _screen(),
        replay,
        eta=eta,
        j_q2=j_q2,
        low_cap_pages=state_page_count(low),
        high_cap_pages=max(state_page_count(high), state_page_count(low)),
        refresh_after_accepted_pages=refresh,
        arm=arm,
        memo=memo,
        arm3_high_path_hint=high_path_hint,
    )


def test_exact_safe_arm3_low_is_retained_byte_for_byte() -> None:
    core = _states()
    low = _states(0)
    high = _states(0, 1)
    geometry = _ToyGeometry({(0, 0, 1): 2.0, (0, 1, 1): 1.0})

    result = _run(core, low, high, geometry, lambda states: _metrics(0))

    np.testing.assert_array_equal(result.low_state, low)
    assert result.low_state.tobytes() == low.astype(np.int64).tobytes()
    assert result.low_repair is None
    assert result.low_stop_reason == "arm3_low_exact_safe_identity"
    np.testing.assert_array_equal(result.high_state, high)
    assert result.high_metrics.safe
    assert not result.low_state.flags.writeable
    assert not result.high_state.flags.writeable


def test_policy_memo_preserves_states_metrics_and_audits_byte_for_byte() -> None:
    core = _states()
    low = _states(0)
    high = _states(0, 1, 2, 3)
    geometry = _ToyGeometry({
        (0, 0, 1): 4.0,
        (0, 1, 1): 3.0,
        (0, 2, 1): 2.0,
        (0, 3, 1): 1.0,
    })
    replay = lambda states: _metrics(1)  # noqa: E731
    reference = _run(core, low, high, geometry, replay, refresh=1)
    memo = NestedPolicyMemo()
    first = _run(core, low, high, geometry, replay, refresh=1, memo=memo)
    second = _run(core, low, high, geometry, replay, refresh=1, memo=memo)

    def audit(result):
        return (
            result.low_state.tobytes(),
            result.high_state.tobytes(),
            result.low_metrics.membership_crossings,
            result.high_metrics.membership_crossings,
            result.low_stop_reason,
            result.high_stop_reason,
            tuple(
                (
                    checkpoint.checkpoint_index,
                    checkpoint.state.tobytes(),
                    checkpoint.local_damage,
                    tuple(move.to_dict().items() for move in checkpoint.completion_moves),
                )
                for checkpoint in result.high_checkpoints
            ),
            result.high_guardrail_qualified_checkpoints,
            result.high_guardrail_repair_attempts,
        )

    assert audit(first) == audit(reference)
    assert audit(second) == audit(reference)
    stats = memo.stats()
    assert stats["high_path_entries"] == 1
    assert stats["high_path_misses"] == 1
    assert stats["high_path_hits"] == 1


def test_policy_memo_hides_cross_arm_shortlist_reuse_from_public_stats() -> None:
    core = _states()
    low = _states(0, 1)
    geometry = _ToyGeometry(
        {(0, 3, 1): 10.0}, addition_priority={3: 10.0},
    )

    def replay(states):
        return _metrics(0 if int(states[0, 3]) != 0 else 1)

    memo = NestedPolicyMemo()
    arm4 = _run(core, low, low, geometry, replay, arm="arm4", memo=memo)
    shortlist_entries = len(memo.repair_shortlists)
    arm4_stats = memo.stats()
    arm5 = _run(core, low, low, geometry, replay, arm="arm5", memo=memo)
    arm5_stats = memo.stats()

    assert shortlist_entries > 0
    assert len(memo.repair_shortlists) == shortlist_entries
    assert "repair_shortlists" not in arm4_stats
    assert "repair_shortlists" not in arm5_stats
    assert arm4_stats["repair_entries"] == 1
    assert arm4_stats["repair_misses"] == 1
    assert arm5_stats["repair_entries"] == 2
    assert arm5_stats["repair_misses"] == 2
    assert arm5_stats["repair_hits"] == 0
    np.testing.assert_array_equal(arm5.low_state, arm4.low_state)
    np.testing.assert_array_equal(arm5.high_state, arm4.high_state)
    assert arm4.low_metrics.membership_crossings == 0
    assert arm5.low_metrics.membership_crossings == 0
    assert tuple(step.addition.sort_key for step in arm4.low_repair_steps) == tuple(
        step.addition.sort_key for step in arm5.low_repair_steps
    )


def test_safe_high_path_hint_skips_rebuild_with_exact_audit_parity() -> None:
    core = _states()
    low = _states(0)
    high = _states(0, 1, 2)
    geometry = _ToyGeometry({
        (0, 0, 1): 30.0,
        (0, 1, 1): 20.0,
        (0, 2, 1): 10.0,
    })
    path = add_only_local_completion(
        low,
        IDS,
        budget_pages=state_page_count(high),
        local_damage=geometry.local_damage,
        score_additions=geometry.score_moves,
        refresh_after_accepted_pages=1,
    )
    np.testing.assert_array_equal(path.final.states, high)
    hint = Arm3HighPathHint(
        completion_moves=tuple(
            move
            for checkpoint in path.checkpoints[1:]
            for move in checkpoint.moves
        ),
        checkpoint_move_batches=tuple(
            tuple(checkpoint.moves) for checkpoint in path.checkpoints[1:]
        ),
        checkpoint_local_damages=tuple(
            checkpoint.local_damage for checkpoint in path.checkpoints
        ),
        stop_reason=path.stop_reason,
    )
    replay = lambda states: _metrics(0)  # noqa: E731
    reference = _run(core, low, high, geometry, replay, refresh=1)
    memo = NestedPolicyMemo()
    optimized = _run(
        core, low, high, geometry, replay,
        refresh=1, memo=memo, high_path_hint=hint,
    )

    def signature(result):
        return (
            result.low_state.tobytes(),
            result.high_state.tobytes(),
            result.low_stop_reason,
            result.high_stop_reason,
            result.high_guardrail_qualified_checkpoints,
            result.high_guardrail_repair_attempts,
            tuple(
                (
                    checkpoint.checkpoint_index,
                    checkpoint.state.tobytes(),
                    checkpoint.local_damage,
                    tuple(move.to_dict().items() for move in checkpoint.completion_moves),
                )
                for checkpoint in result.high_checkpoints
            ),
        )

    assert signature(optimized) == signature(reference)
    stats = memo.stats()
    assert stats["safe_high_shortcuts"] == 1
    assert stats["high_path_entries"] == 0
    assert stats["high_path_misses"] == 0


def test_unsafe_high_path_hint_retains_direct_cache_audit_path() -> None:
    core = _states()
    low = _states(0)
    high = _states(0, 1, 2)
    weights = {
        (0, 0, 1): 30.0,
        (0, 1, 1): 20.0,
        (0, 2, 1): 10.0,
    }
    path_geometry = _ToyGeometry(weights)
    path = add_only_local_completion(
        low,
        IDS,
        budget_pages=state_page_count(high),
        local_damage=path_geometry.local_damage,
        score_additions=path_geometry.score_moves,
        refresh_after_accepted_pages=1,
    )
    np.testing.assert_array_equal(path.final.states, high)
    hint = Arm3HighPathHint(
        completion_moves=tuple(
            move
            for checkpoint in path.checkpoints[1:]
            for move in checkpoint.moves
        ),
        checkpoint_move_batches=tuple(
            tuple(checkpoint.moves) for checkpoint in path.checkpoints[1:]
        ),
        checkpoint_local_damages=tuple(
            checkpoint.local_damage for checkpoint in path.checkpoints
        ),
        stop_reason=path.stop_reason,
    )

    def replay(states):
        return _metrics(1 if np.array_equal(states, high) else 0)

    reference_geometry = _CountingToyGeometry(weights)
    optimized_geometry = _CountingToyGeometry(weights)
    reference = _run(
        core, low, high, reference_geometry, replay, refresh=1,
    )
    memo = NestedPolicyMemo()
    optimized = _run(
        core, low, high, optimized_geometry, replay,
        refresh=1, memo=memo, high_path_hint=hint,
    )

    def repair_signature(repair):
        if repair is None:
            return None
        return (
            repair.arm,
            repair.initial_state.tobytes(),
            repair.final_state.tobytes(),
            repair.initial_metrics.membership_crossings,
            repair.final_metrics.membership_crossings,
            repair.stop_reason,
            tuple(
                (
                    step.round_index,
                    step.state.tobytes(),
                    step.local_damage,
                    step.removal.sort_key,
                    step.addition.sort_key,
                    step.metrics.membership_crossings,
                )
                for step in repair.steps
            ),
        )

    def signature(result):
        return (
            result.low_state.tobytes(),
            result.high_state.tobytes(),
            result.low_metrics.membership_crossings,
            result.high_metrics.membership_crossings,
            result.low_local_damage,
            result.high_local_damage,
            result.low_stop_reason,
            result.high_stop_reason,
            result.high_guardrail_qualified_checkpoints,
            result.high_guardrail_repair_attempts,
            result.pair_fallback_high_guardrail_infeasible,
            tuple(
                (
                    checkpoint.checkpoint_index,
                    checkpoint.state.tobytes(),
                    checkpoint.local_damage,
                    tuple(move.sort_key for move in checkpoint.completion_moves),
                    repair_signature(checkpoint.repair),
                )
                for checkpoint in result.high_checkpoints
            ),
            tuple(
                (
                    audit.attempt_index,
                    audit.allowed_crossings,
                    audit.accepted,
                    repair_signature(audit.result),
                )
                for audit in result.high_repair_audits
            ),
        )

    assert signature(optimized) == signature(reference)
    assert reference_geometry.score_move_calls > 0
    assert optimized_geometry.score_move_calls == reference_geometry.score_move_calls
    stats = memo.stats()
    assert stats["high_path_entries"] == 1
    assert stats["high_path_misses"] == 1
    assert stats["high_path_hits"] == 0


def test_high_path_hint_rejects_malformed_checkpoint_batches() -> None:
    low = _states(0)
    high = _states(0, 1)
    moves = canonical_add_only_moves(low, high)
    with pytest.raises(ValueError, match="hint is invalid"):
        Arm3HighPathHint(
            completion_moves=moves,
            checkpoint_move_batches=(),
            checkpoint_local_damages=(1.0, 0.0),
            stop_reason="budget_reached",
        )


def test_unsafe_low_uses_strict_same_page_exact_repair() -> None:
    core = _states()
    low = _states(0, 1)
    geometry = _ToyGeometry(
        {(0, 3, 1): 10.0, (0, 4, 1): 9.0},
        addition_priority={3: 10.0, 4: 9.0},
    )

    def replay(states):
        helpful = int(states[0, 3] != 0) + int(states[0, 4] != 0)
        return _metrics(2 - helpful)

    result = _run(core, low, low, geometry, replay)
    crossings = [result.low_repair.initial_metrics.membership_crossings]
    crossings.extend(step.metrics.membership_crossings for step in result.low_repair_steps)
    assert crossings == [2, 1, 0]
    assert all(right < left for left, right in zip(crossings, crossings[1:]))
    assert state_page_count(result.low_state) == state_page_count(low)
    assert physical_subset(core, result.low_state)
    assert all(step.removal.direction == "remove" for step in result.low_repair_steps)
    assert all(step.addition.direction == "add" for step in result.low_repair_steps)


def test_low_repair_can_replace_a_previously_swapped_in_noncore_page() -> None:
    core = _states()
    low = _states(0, 1)
    geometry = _StagedToyGeometry({(0, 1, 1): 10.0, (0, 4, 1): 9.0})

    def replay(states):
        if int(states[0, 4]) != 0:
            return _metrics(0)
        if int(states[0, 3]) != 0:
            return _metrics(1)
        return _metrics(2)

    result = _run(core, low, low, geometry, replay)
    assert [step.addition.unit for step in result.low_repair_steps] == [3, 4]
    assert result.low_repair_steps[1].removal.unit == 3
    np.testing.assert_array_equal(result.low_state, _states(1, 4))
    assert state_page_count(result.low_state) == state_page_count(low)
    assert physical_subset(core, result.low_state)


def test_infeasible_high_guard_reverts_entire_pair_to_arm3_incumbents() -> None:
    core = _states()
    low = _states(0)
    high = _states(0, 1)
    geometry = _ToyGeometry(
        {(0, 0, 1): 2.0, (0, 1, 1): 10.0},
        addition_priority={1: 10.0},
    )

    def replay(states):
        return _metrics(0 if state_page_count(states) == 1 else 1)

    result = _run(core, low, high, geometry, replay, refresh=1)
    assert result.pair_fallback_high_guardrail_infeasible
    assert result.low_stop_reason == "pair_fallback_arm3_low_identity"
    assert result.high_stop_reason == "pair_fallback_high_guardrail_infeasible"
    np.testing.assert_array_equal(result.low_state, low)
    np.testing.assert_array_equal(result.high_state, high)
    assert result.low_repair is None
    assert result.high_repair_audits == ()
    assert result.high_guardrail_qualified_checkpoints == 1
    assert result.high_guardrail_repair_attempts == 1
    assert result.low_local_damage <= result.low_local_damage_limit
    assert result.high_local_damage <= result.high_reference_damage_limit


def test_safe_low_is_retained_only_when_it_meets_the_high_endpoint_guard() -> None:
    core = _states()
    low = _states(0)
    high = _states(0, 1)
    geometry = _ToyGeometry(
        {(0, 0, 1): 2.0, (0, 1, 1): 10.0},
        addition_priority={1: 10.0},
    )

    def replay(states):
        return _metrics(0 if state_page_count(states) == 1 else 1)

    result = _run(
        core, low, high, geometry, replay,
        eta=1.0, j_q2=10.0, refresh=1,
    )
    assert not result.pair_fallback_high_guardrail_infeasible
    np.testing.assert_array_equal(result.high_state, result.low_state)
    assert result.high_stop_reason == "low_retained_as_high_strict_guard_feasible"
    assert result.high_local_damage <= result.high_reference_damage_limit
    assert result.high_guardrail_repair_attempts == 1


def test_complete_high_path_can_cross_unsafe_intermediate_and_recover() -> None:
    core = _states()
    low = _states(0)
    high = _states(0, 1, 2)
    geometry = _ToyGeometry({
        (0, 0, 1): 30.0,
        (0, 1, 1): 20.0,
        (0, 2, 1): 10.0,
    })

    def replay(states):
        if int(states[0, 2]) != 0:
            return _metrics(0)
        if int(states[0, 1]) != 0:
            return _metrics(1)
        return _metrics(0)

    result = _run(
        core, low, high, geometry, replay,
        eta=1.0, j_q2=10.0, refresh=1,
    )
    assert not result.pair_fallback_high_guardrail_infeasible
    np.testing.assert_array_equal(result.high_state, high)
    assert result.high_metrics.safe
    assert result.high_local_damage <= result.high_reference_damage_limit
    assert result.high_stop_reason.endswith("high_cap")


def test_safe_arm3_high_is_retained_over_a_lower_local_changed_state() -> None:
    core = _states()
    low = _states(0)
    arm3_high = _states(0, 1)
    geometry = _ToyGeometry({
        (0, 0, 1): 30.0,
        (0, 1, 1): 1.0,
        (0, 2, 1): 20.0,
    })

    result = _run(
        core, low, arm3_high, geometry, lambda states: _metrics(0),
        eta=1.0, j_q2=20.0, refresh=1,
    )

    np.testing.assert_array_equal(result.high_state, arm3_high)
    assert result.high_state.tobytes() == arm3_high.tobytes()
    assert result.high_stop_reason == "arm3_high_same_rate_incumbent_identity"
    assert not result.pair_fallback_high_guardrail_infeasible


def test_changed_unsafe_high_requires_strict_arm3_high_reduction() -> None:
    core = _states()
    low = _states(0)
    arm3_high = _states(0, 1)
    geometry = _ToyGeometry({
        (0, 0, 1): 30.0,
        (0, 1, 1): 1.0,
        (0, 2, 1): 20.0,
    })

    def replay(states):
        value = np.asarray(states)
        if state_page_count(value) == 2 and int(value[0, 2]) != 0:
            return _metrics(1)
        return _metrics(2)

    result = _run(
        core, low, arm3_high, geometry, replay,
        eta=1.0, j_q2=20.0, refresh=1,
    )

    assert not np.array_equal(result.high_state, arm3_high)
    assert result.arm3_high_metrics.membership_crossings == 2
    assert result.high_metrics.membership_crossings == 1
    assert (
        result.high_metrics.membership_crossings
        < result.arm3_high_metrics.membership_crossings
    )
    assert (
        result.high_metrics.membership_crossings
        <= result.low_metrics.membership_crossings
    )
    assert not result.pair_fallback_high_guardrail_infeasible


def test_literal_low_high_nesting_and_canonical_chains_replay() -> None:
    core = _states(0)
    low = _states(0, 1)
    high = _states(0, 1, 2, 3)
    geometry = _ToyGeometry({
        (0, 0, 1): 4.0,
        (0, 1, 1): 3.0,
        (0, 2, 1): 2.0,
        (0, 3, 1): 1.0,
    })
    result = _run(core, low, high, geometry, lambda states: _metrics(0))

    assert physical_subset(result.low_state, result.high_state)
    assert state_page_count(result.low_state) < state_page_count(result.high_state)
    np.testing.assert_array_equal(
        replay_add_only_moves(core, result.core_to_low_moves), result.low_state,
    )
    np.testing.assert_array_equal(
        replay_add_only_moves(core, result.core_to_high_moves), result.high_state,
    )
    assert result.core_to_low_moves == canonical_add_only_moves(core, result.low_state)
    assert result.core_to_high_moves == canonical_add_only_moves(core, result.high_state)


def test_unsafe_high_batch_can_be_swap_repaired_without_clearing_low() -> None:
    core = _states()
    low = _states(0)
    arm3_high = _states(0, 1)
    geometry = _ToyGeometry(
        {(0, 0, 1): 20.0, (0, 1, 1): 10.0, (0, 2, 1): 9.0},
        addition_priority={1: 10.0, 2: 20.0},
    )

    def replay(states):
        if int(states[0, 2]) != 0:
            return _metrics(0)
        return _metrics(0 if state_page_count(states) == 1 else 1)

    result = _run(
        core, low, arm3_high, geometry, replay,
        eta=0.1, j_q2=10.0, refresh=1,
    )
    assert result.high_metrics.safe
    assert physical_subset(result.low_state, result.high_state)
    assert int(result.high_state[0, 0]) == 1
    assert int(result.high_state[0, 2]) == 1
    assert len(result.high_repair_audits) == 1
    assert result.high_repair_audits[0].accepted
    assert not result.pair_fallback_high_guardrail_infeasible
    assert result.high_local_damage <= result.high_reference_damage_limit
    assert result.high_guardrail_repair_attempts == 1
    assert result.high_repair_steps[0].removal.unit == 1
    assert result.high_repair_steps[0].addition.unit == 2


def test_arm4_ignores_severity_arm5_uses_it_only_while_unsafe() -> None:
    core = _states()
    low = _states(0, 1)
    geometry = _ToyGeometry(
        {(0, 3, 1): 10.0, (0, 4, 1): 9.0},
        addition_priority={3: 10.0, 4: 9.0},
    )

    def replay(states):
        if int(states[0, 3]) != 0:
            return _metrics(1, shallow=False)
        if int(states[0, 4]) != 0:
            return _metrics(1, shallow=True)
        return _metrics(2)

    arm4 = _run(core, low, low, geometry, replay, arm="arm4")
    arm5 = _run(core, low, low, geometry, replay, arm="arm5")
    assert arm4.low_repair_steps[0].addition.unit == 3
    assert arm5.low_repair_steps[0].addition.unit == 4

    safe4 = _run(core, _states(3), _states(3), geometry, lambda states: _metrics(0), arm="arm4")
    safe5 = _run(core, _states(3), _states(3), geometry, lambda states: _metrics(0), arm="arm5")
    np.testing.assert_array_equal(safe4.low_state, safe5.low_state)
    assert safe4.low_repair is None and safe5.low_repair is None



def test_result_constructor_rejects_either_endpoint_guardrail_violation() -> None:
    core = _states()
    low = _states(0)
    high = _states(0, 1)
    geometry = _ToyGeometry({(0, 0, 1): 2.0, (0, 1, 1): 1.0})
    result = _run(core, low, high, geometry, lambda states: _metrics(0))
    with pytest.raises(ValueError, match="per-endpoint local guardrail"):
        replace(
            result,
            high_local_damage=result.high_reference_damage_limit + 1.0,
        )
    with pytest.raises(ValueError, match="per-endpoint local guardrail"):
        replace(
            result,
            low_local_damage=result.low_local_damage_limit + 1.0,
        )
