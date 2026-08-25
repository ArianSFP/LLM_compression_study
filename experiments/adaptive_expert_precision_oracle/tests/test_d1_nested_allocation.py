from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oracle_study.d1_nested_allocation import (
    AllocationCheckpoint,
    AllocationPathTrace,
    DOWN_BIT,
    EXPERTS_PER_GROUP,
    GATE_BIT,
    PhysicalPageMove,
    RouteRepairMetrics,
    STATE_POPCOUNT,
    UNITS_PER_EXPERT,
    UP_BIT,
    add_only_local_completion,
    align_states_to_expert_ids,
    apply_page_move,
    apply_page_moves,
    expert_page_counts,
    hard_saturating_route_repair,
    legal_add_moves,
    legal_remove_moves,
    physical_added_pages,
    physical_page_distance,
    physical_removed_pages,
    physical_subset,
    reverse_local_prune_to_reserve,
    state_page_count,
    validate_expert_id_alignment,
    validate_page_parity,
    validate_states,
)


IDS = np.arange(10, 10 + EXPERTS_PER_GROUP, dtype=np.int64)


def _states() -> np.ndarray:
    return np.zeros((EXPERTS_PER_GROUP, UNITS_PER_EXPERT), np.int64)


def _damage_from_target(target: np.ndarray, weights: np.ndarray | None = None):
    desired = validate_states(target)
    cost = np.ones(
        (EXPERTS_PER_GROUP, UNITS_PER_EXPERT, 3), np.float64,
    ) if weights is None else np.asarray(weights, np.float64)

    def damage(states: np.ndarray) -> float:
        value = validate_states(states)
        missing = desired & ((~value) & 7)
        extra = value & ((~desired) & 7)
        total = 0.0
        for slot, bit in enumerate((DOWN_BIT, UP_BIT, GATE_BIT)):
            total += float(cost[:, :, slot][(missing & bit) != 0].sum())
            total += float(cost[:, :, slot][(extra & bit) != 0].sum())
        return total

    return damage


def _checkpoint(name: str, states: np.ndarray, damage: float = 0.0) -> AllocationCheckpoint:
    return AllocationCheckpoint(name, 0, IDS, states, damage)


def test_exact_three_bit_page_accounting_and_parity() -> None:
    states = _states()
    states[0, :8] = np.arange(8)
    np.testing.assert_array_equal(STATE_POPCOUNT, [0, 1, 1, 2, 1, 2, 2, 3])
    assert state_page_count(states) == 12
    assert expert_page_counts(states).tolist() == [12, 0, 0, 0, 0, 0, 0, 0]
    validate_page_parity(
        states,
        expected_group_pages=12,
        expected_expert_pages=[12, 0, 0, 0, 0, 0, 0, 0],
    )
    with pytest.raises(ValueError, match="group pages"):
        validate_page_parity(states, expected_group_pages=11)
    with pytest.raises(ValueError, match="shape"):
        validate_states(np.zeros((8, 511), np.int64))
    with pytest.raises(ValueError, match="integer"):
        validate_states(np.zeros((8, 512), np.float64))


def test_expert_alignment_is_explicit_and_lossless() -> None:
    states = _states()
    for row in range(EXPERTS_PER_GROUP):
        states[row, 0] = row % 8
    reversed_ids = IDS[::-1]
    with pytest.raises(ValueError, match="expert_id"):
        validate_expert_id_alignment(IDS, reversed_ids)
    aligned = align_states_to_expert_ids(states[::-1], reversed_ids, IDS)
    np.testing.assert_array_equal(aligned, states)
    with pytest.raises(ValueError, match="sets differ"):
        align_states_to_expert_ids(states, IDS, IDS + 100)


def test_subset_distance_and_directional_page_counts_are_exact() -> None:
    low = _states()
    high = _states()
    low[0, 0] = DOWN_BIT
    high[0, 0] = DOWN_BIT | UP_BIT
    high[1, 3] = GATE_BIT
    assert physical_subset(low, high)
    assert not physical_subset(high, low)
    assert physical_page_distance(low, high) == 2
    assert physical_added_pages(low, high) == 2
    assert physical_removed_pages(low, high) == 0
    assert physical_removed_pages(high, low) == 2


def test_legal_one_bit_moves_reject_stale_and_same_unit_batches() -> None:
    states = np.full((EXPERTS_PER_GROUP, UNITS_PER_EXPERT), 7, np.int64)
    states[0, 0] = 0
    adds = legal_add_moves(states)
    assert [(move.bit, move.projection) for move in adds] == [
        (DOWN_BIT, "down"),
        (UP_BIT, "up"),
        (GATE_BIT, "gate"),
    ]
    updated = apply_page_move(states, adds[0])
    assert updated[0, 0] == DOWN_BIT
    with pytest.raises(ValueError, match="stale"):
        apply_page_move(updated, adds[0])
    with pytest.raises(ValueError, match="distinct expert units"):
        apply_page_moves(states, adds[:2])

    removes = legal_remove_moves(updated, lower_states=states)
    assert len(removes) == 1
    assert removes[0].direction == "remove"
    np.testing.assert_array_equal(apply_page_move(updated, removes[0]), states)
    with pytest.raises(ValueError, match="exactly"):
        PhysicalPageMove(0, 0, DOWN_BIT, 0, DOWN_BIT | UP_BIT)


def test_reverse_local_pruning_is_a_common_d1_blind_physical_subset() -> None:
    high = _states()
    high[0, 0] = 7
    high[0, 1] = DOWN_BIT | UP_BIT
    weights = np.ones((8, 512, 3), np.float64)
    weights[0, 0] = [9.0, 8.0, 7.0]
    weights[0, 1] = [1.0, 2.0, 3.0]
    local = _damage_from_target(high, weights)
    path = reverse_local_prune_to_reserve(
        high,
        IDS,
        budget_pages=5,
        repair_window_pages=2,
        local_damage=local,
    )
    assert path.target_pages == 3
    assert path.stop_reason == "reserve_cap_reached"
    assert path.final.group_pages == 3
    assert len(path.checkpoints) == 3
    assert [checkpoint.moves[0].sort_key for checkpoint in path.checkpoints[1:]] == [
        (0, 1, DOWN_BIT, 3, 2),
        (0, 1, UP_BIT, 2, 0),
    ]
    assert physical_subset(path.final.states, high)
    assert physical_page_distance(path.final.states, high) == 2

    repeat = reverse_local_prune_to_reserve(
        high, IDS, budget_pages=5, repair_window_pages=2, local_damage=local,
    )
    assert repeat.to_dict() == path.to_dict()


def test_reverse_local_vectorized_scores_have_an_exact_parity_gate() -> None:
    high = _states()
    high[0, 0] = DOWN_BIT
    local = _damage_from_target(high)

    def wrong_score(_states, moves):
        return np.zeros(len(moves), np.float64)

    with pytest.raises(RuntimeError, match="parity"):
        reverse_local_prune_to_reserve(
            high,
            IDS,
            budget_pages=1,
            repair_window_pages=1,
            local_damage=local,
            score_removals=wrong_score,
        )


def test_add_only_completion_freezes_core_and_stops_before_useless_traffic() -> None:
    core = _states()
    core[0, 0] = DOWN_BIT
    target = core.copy()
    target[0, 1] = UP_BIT
    target[1, 0] = GATE_BIT
    weights = np.ones((8, 512, 3), np.float64)
    weights[0, 1, 1] = 3.0
    weights[1, 0, 2] = 2.0
    local = _damage_from_target(target, weights)
    path = add_only_local_completion(
        core,
        IDS,
        budget_pages=4,
        local_damage=local,
    )
    assert path.final.group_pages == 3
    assert path.stop_reason == "no_positive_local_gain"
    assert len(path.checkpoints) == 3
    assert physical_subset(core, path.final.states)
    assert path.final.local_damage == 0.0
    for previous, current in zip(path.checkpoints, path.checkpoints[1:]):
        assert current.group_pages == previous.group_pages + 1
        assert physical_subset(previous.states, current.states)


def test_add_only_completion_respects_upper_mask_and_d1_admissibility_callback() -> None:
    core = _states()
    upper = core.copy()
    upper[0, 0] = DOWN_BIT | UP_BIT
    target = upper.copy()
    local = _damage_from_target(target)

    def only_up(_states, moves):
        return np.asarray([move.bit == UP_BIT for move in moves], dtype=bool)

    path = add_only_local_completion(
        core,
        IDS,
        budget_pages=2,
        upper_states=upper,
        local_damage=local,
        admissible_additions=only_up,
    )
    assert path.final.states[0, 0] == UP_BIT
    assert path.stop_reason == "no_admissible_addition"
    assert physical_subset(path.final.states, upper)


def test_checkpoint_and_path_serialization_are_exact_and_replayable() -> None:
    initial = _states()
    move = legal_add_moves(initial)[0]
    final = apply_page_move(initial, move)
    checkpoints = (
        AllocationCheckpoint("add_only_completion", 0, IDS, initial, 1.0),
        AllocationCheckpoint("add_only_completion", 1, IDS, final, 0.0, (move,)),
    )
    path = AllocationPathTrace(
        "add_only_completion", 1, "budget_reached", checkpoints,
    )
    encoded = json.loads(json.dumps(path.to_dict()))
    restored = AllocationPathTrace.from_dict(encoded)
    assert restored.to_dict() == path.to_dict()
    assert not restored.final.states.flags.writeable
    assert not restored.final.expert_ids.flags.writeable

    tampered = json.loads(json.dumps(encoded))
    tampered["checkpoints"][1]["states"][0][0] = UP_BIT
    with pytest.raises(ValueError):
        AllocationPathTrace.from_dict(tampered)


def test_hard_saturating_repair_holds_safe_incumbent() -> None:
    incumbent = _states()
    alternative = incumbent.copy()
    alternative[0, 0] = DOWN_BIT
    candidates = (
        _checkpoint("candidate", incumbent, 2.0),
        _checkpoint("candidate", alternative, 0.0),
    )
    choice = hard_saturating_route_repair(
        ("incumbent", "alternative"),
        candidates,
        (
            RouteRepairMetrics(0, 100.0, 2.0),
            RouteRepairMetrics(0, 0.0, 0.0),
        ),
        require_matched_pages=False,
    )
    assert choice.index == 0
    assert choice.reason == "incumbent_d1_safe"


def test_arm5_severity_is_used_only_while_a_repair_remains_unsafe() -> None:
    incumbent = _states()
    unsafe_low_severity = incumbent.copy()
    unsafe_low_severity[0, 0] = DOWN_BIT
    unsafe_high_severity = incumbent.copy()
    unsafe_high_severity[0, 1] = DOWN_BIT
    safe_low_local = incumbent.copy()
    safe_low_local[0, 2] = DOWN_BIT
    safe_high_local = incumbent.copy()
    safe_high_local[0, 3] = DOWN_BIT
    candidates = tuple(_checkpoint("candidate", value) for value in (
        incumbent,
        unsafe_low_severity,
        unsafe_high_severity,
        safe_low_local,
        safe_high_local,
    ))

    # Among still-unsafe one-crossing repairs, Arm5 severity precedes local.
    unsafe_choice = hard_saturating_route_repair(
        ("incumbent", "low_severity", "high_severity"),
        candidates[:3],
        (
            RouteRepairMetrics(2, 0.0, 0.0),
            RouteRepairMetrics(1, 0.1, 10.0),
            RouteRepairMetrics(1, 0.9, 0.0),
        ),
        require_matched_pages=False,
    )
    assert unsafe_choice.name == "low_severity"

    # Once both candidates are safe, their Arm5 severity is ignored and the
    # lower local-damage state wins.
    safe_choice = hard_saturating_route_repair(
        ("incumbent", "safe_low_local", "safe_high_local"),
        (candidates[0], candidates[3], candidates[4]),
        (
            RouteRepairMetrics(2, 0.0, 5.0),
            RouteRepairMetrics(0, 100.0, 1.0),
            RouteRepairMetrics(0, 0.0, 2.0),
        ),
        require_matched_pages=False,
    )
    assert safe_choice.name == "safe_low_local"


def test_route_repair_enforces_frozen_core_page_budget_and_distance() -> None:
    core = _states()
    core[0, 0] = DOWN_BIT
    incumbent = core.copy()
    incumbent[0, 1] = UP_BIT
    candidate = core.copy()
    candidate[0, 2] = GATE_BIT
    candidates = (
        _checkpoint("candidate", incumbent),
        _checkpoint("candidate", candidate),
    )
    metrics = (RouteRepairMetrics(1, 1.0, 1.0), RouteRepairMetrics(0, 0.0, 1.0))
    choice = hard_saturating_route_repair(
        ("incumbent", "repair"),
        candidates,
        metrics,
        frozen_core=_checkpoint("core", core),
        page_budget=2,
        maximum_page_distance=2,
    )
    assert choice.name == "repair"

    broken = candidate.copy()
    broken[0, 0] = 0
    with pytest.raises(ValueError, match="frozen-core"):
        hard_saturating_route_repair(
            ("incumbent", "broken"),
            (candidates[0], _checkpoint("candidate", broken)),
            metrics,
            frozen_core=_checkpoint("core", core),
            page_budget=2,
            require_matched_pages=False,
        )
    with pytest.raises(ValueError, match="distance"):
        hard_saturating_route_repair(
            ("incumbent", "repair"),
            candidates,
            metrics,
            maximum_page_distance=1,
        )


def test_reverse_local_accepts_high_reference_above_low_budget() -> None:
    high = _states()
    high[0, :3] = 7
    local = _damage_from_target(high)
    assert state_page_count(high) == 9
    path = reverse_local_prune_to_reserve(
        high,
        IDS,
        budget_pages=7,
        repair_window_pages=2,
        local_damage=local,
    )
    assert path.target_pages == 5
    assert path.final.group_pages == 5
    assert physical_subset(path.final.states, high)
    assert physical_page_distance(path.final.states, high) == 4


def test_reverse_local_batched_refresh_is_nested_replayable_and_deterministic() -> None:
    high = _states()
    high[0, :6] = DOWN_BIT
    local = _damage_from_target(high)
    path = reverse_local_prune_to_reserve(
        high,
        IDS,
        budget_pages=4,
        repair_window_pages=2,
        local_damage=local,
        refresh_after_accepted_pages=4,
    )
    assert path.target_pages == 2
    assert path.final.group_pages == 2
    assert len(path.checkpoints) == 2
    assert [move.unit for move in path.final.moves] == [0, 1, 2, 3]
    assert len({(move.expert, move.unit) for move in path.final.moves}) == 4
    np.testing.assert_array_equal(
        apply_page_moves(path.checkpoints[0].states, path.final.moves),
        path.final.states,
    )
    assert physical_subset(path.final.states, high)
    restored = AllocationPathTrace.from_dict(
        json.loads(json.dumps(path.to_dict())),
    )
    assert restored.to_dict() == path.to_dict()
    repeat = reverse_local_prune_to_reserve(
        high,
        IDS,
        budget_pages=4,
        repair_window_pages=2,
        local_damage=local,
        refresh_after_accepted_pages=4,
    )
    assert repeat.to_dict() == path.to_dict()


def test_add_only_batched_refresh_is_nested_replayable_and_deterministic() -> None:
    core = _states()
    target = core.copy()
    target[0, :6] = DOWN_BIT
    local = _damage_from_target(target)
    path = add_only_local_completion(
        core,
        IDS,
        budget_pages=4,
        local_damage=local,
        refresh_after_accepted_pages=4,
    )
    assert path.stop_reason == "budget_reached"
    assert path.final.group_pages == 4
    assert len(path.checkpoints) == 2
    assert [move.unit for move in path.final.moves] == [0, 1, 2, 3]
    assert len({(move.expert, move.unit) for move in path.final.moves}) == 4
    np.testing.assert_array_equal(
        apply_page_moves(path.checkpoints[0].states, path.final.moves),
        path.final.states,
    )
    assert physical_subset(core, path.final.states)
    repeat = add_only_local_completion(
        core,
        IDS,
        budget_pages=4,
        local_damage=local,
        refresh_after_accepted_pages=4,
    )
    assert repeat.to_dict() == path.to_dict()
