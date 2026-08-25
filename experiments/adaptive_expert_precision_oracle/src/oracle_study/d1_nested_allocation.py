"""Physical, state-direct allocation primitives for the D1 Experiment A oracle.

This module deliberately knows nothing about frontier option indices or the
sequential runtime controller.  One state row describes the three physical
refinement pages for each of 512 units in one routed expert.  Search policy is
provided through callbacks so a runner can supply exact combined-qmetric
scores and D1-only screening or replay gates without changing page mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

import numpy as np


EXPERTS_PER_GROUP = 8
UNITS_PER_EXPERT = 512
DOWN_BIT = 1
UP_BIT = 2
GATE_BIT = 4
VALID_STATE_MASK = DOWN_BIT | UP_BIT | GATE_BIT
PROJECTION_BITS = (
    (DOWN_BIT, "down"),
    (UP_BIT, "up"),
    (GATE_BIT, "gate"),
)
STATE_POPCOUNT = np.asarray([0, 1, 1, 2, 1, 2, 2, 3], np.int64)

CHECKPOINT_SCHEMA = "pr13_d1_physical_allocation_checkpoint_v1"
PATH_SCHEMA = "pr13_d1_physical_allocation_path_v1"

StateScore = Callable[[np.ndarray], float]
MoveScore = Callable[
    [np.ndarray, tuple["PhysicalPageMove", ...]],
    Sequence[float] | np.ndarray,
]
MoveAdmissibility = Callable[
    [np.ndarray, tuple["PhysicalPageMove", ...]],
    Sequence[bool] | np.ndarray,
]


__all__ = [
    "EXPERTS_PER_GROUP",
    "UNITS_PER_EXPERT",
    "DOWN_BIT",
    "UP_BIT",
    "GATE_BIT",
    "VALID_STATE_MASK",
    "PROJECTION_BITS",
    "STATE_POPCOUNT",
    "PhysicalPageMove",
    "AllocationCheckpoint",
    "AllocationPathTrace",
    "RouteRepairMetrics",
    "RouteRepairSelection",
    "validate_states",
    "validate_expert_ids",
    "validate_expert_id_alignment",
    "align_states_to_expert_ids",
    "expert_page_counts",
    "state_page_count",
    "validate_page_parity",
    "physical_subset",
    "physical_page_distance",
    "physical_removed_pages",
    "physical_added_pages",
    "legal_add_moves",
    "legal_remove_moves",
    "apply_page_move",
    "apply_page_moves",
    "reverse_local_prune_to_reserve",
    "add_only_local_completion",
    "hard_saturating_route_repair",
]


def _integer_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype == np.bool_ or not np.issubdtype(
        raw.dtype, np.integer,
    ):
        raise ValueError(f"{name} must be an integer array with shape {shape}")
    return np.asarray(raw, np.int64).copy()


def _freeze(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value).copy()
    result.setflags(write=False)
    return result


def validate_states(states: Any) -> np.ndarray:
    """Return a copy of one complete physical ``[8,512]`` state table."""

    result = _integer_array(
        states, (EXPERTS_PER_GROUP, UNITS_PER_EXPERT), "states",
    )
    if np.any((result < 0) | (result > VALID_STATE_MASK)):
        raise ValueError("physical states must lie in [0,7]")
    return result


def validate_expert_ids(expert_ids: Any) -> np.ndarray:
    """Return eight distinct, nonnegative expert IDs in routed row order."""

    result = _integer_array(expert_ids, (EXPERTS_PER_GROUP,), "expert_ids")
    if np.any(result < 0) or len(set(result.tolist())) != EXPERTS_PER_GROUP:
        raise ValueError("expert_ids must contain eight distinct nonnegative IDs")
    return result


def validate_expert_id_alignment(reference_ids: Any, candidate_ids: Any) -> None:
    """Reject implicit router-rank alignment when physical expert IDs differ."""

    reference = validate_expert_ids(reference_ids)
    candidate = validate_expert_ids(candidate_ids)
    if not np.array_equal(reference, candidate):
        raise ValueError("physical state rows are not aligned by expert_id")


def align_states_to_expert_ids(
    states: Any,
    expert_ids: Any,
    target_expert_ids: Any,
) -> np.ndarray:
    """Explicitly reorder state rows onto a target expert-ID order."""

    value = validate_states(states)
    source = validate_expert_ids(expert_ids)
    target = validate_expert_ids(target_expert_ids)
    if set(source.tolist()) != set(target.tolist()):
        raise ValueError("source and target expert-ID sets differ")
    rows = {int(expert): index for index, expert in enumerate(source.tolist())}
    return value[[rows[int(expert)] for expert in target.tolist()]].copy()


def expert_page_counts(states: Any) -> np.ndarray:
    value = validate_states(states)
    return STATE_POPCOUNT[value].sum(axis=1, dtype=np.int64)


def state_page_count(states: Any) -> int:
    return int(expert_page_counts(states).sum())


def validate_page_parity(
    states: Any,
    *,
    expected_group_pages: int | None = None,
    expected_expert_pages: Any | None = None,
) -> tuple[np.ndarray, int]:
    """Validate recorded page counts against exact three-bit popcount."""

    per_expert = expert_page_counts(states)
    group = int(per_expert.sum())
    if expected_expert_pages is not None:
        recorded = _integer_array(
            expected_expert_pages, (EXPERTS_PER_GROUP,), "expected_expert_pages",
        )
        if np.any(recorded < 0) or not np.array_equal(recorded, per_expert):
            raise ValueError("recorded expert pages disagree with physical states")
    if expected_group_pages is not None:
        expected = int(expected_group_pages)
        if expected < 0 or expected != group:
            raise ValueError("recorded group pages disagree with physical states")
    return per_expert.copy(), group


def physical_subset(child: Any, parent: Any) -> bool:
    low = validate_states(child)
    high = validate_states(parent)
    removed = low & ((~high) & VALID_STATE_MASK)
    return not bool(np.any(removed))


def physical_page_distance(left: Any, right: Any) -> int:
    first = validate_states(left)
    second = validate_states(right)
    return int(STATE_POPCOUNT[first ^ second].sum())


def physical_removed_pages(source: Any, destination: Any) -> int:
    before = validate_states(source)
    after = validate_states(destination)
    return int(STATE_POPCOUNT[before & ((~after) & VALID_STATE_MASK)].sum())


def physical_added_pages(source: Any, destination: Any) -> int:
    before = validate_states(source)
    after = validate_states(destination)
    return int(STATE_POPCOUNT[after & ((~before) & VALID_STATE_MASK)].sum())


@dataclass(frozen=True)
class PhysicalPageMove:
    """One legal add or remove of one projection page."""

    expert: int
    unit: int
    bit: int
    source_state: int
    destination_state: int

    def __post_init__(self) -> None:
        expert = int(self.expert)
        unit = int(self.unit)
        bit = int(self.bit)
        source = int(self.source_state)
        destination = int(self.destination_state)
        if expert < 0 or expert >= EXPERTS_PER_GROUP:
            raise ValueError("page move expert lies outside the routed group")
        if unit < 0 or unit >= UNITS_PER_EXPERT:
            raise ValueError("page move unit lies outside the expert")
        if bit not in {DOWN_BIT, UP_BIT, GATE_BIT}:
            raise ValueError("page move bit must be down=1, up=2, or gate=4")
        if source < 0 or source > VALID_STATE_MASK or destination < 0 or destination > VALID_STATE_MASK:
            raise ValueError("page move states must lie in [0,7]")
        if (source ^ destination) != bit:
            raise ValueError("page move must change exactly its declared bit")
        is_add = (source & bit) == 0 and destination == (source | bit)
        is_remove = (source & bit) != 0 and destination == (source & ~bit)
        if not (is_add or is_remove):
            raise ValueError("page move is neither a legal add nor remove")
        object.__setattr__(self, "expert", expert)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "bit", bit)
        object.__setattr__(self, "source_state", source)
        object.__setattr__(self, "destination_state", destination)

    @property
    def direction(self) -> str:
        return "add" if self.destination_state & self.bit else "remove"

    @property
    def page_delta(self) -> int:
        return 1 if self.direction == "add" else -1

    @property
    def projection(self) -> str:
        return dict(PROJECTION_BITS)[self.bit]

    @property
    def sort_key(self) -> tuple[int, int, int, int, int]:
        return (
            self.expert,
            self.unit,
            self.bit,
            self.source_state,
            self.destination_state,
        )

    def to_dict(self) -> dict[str, int | str]:
        return {
            "expert": self.expert,
            "unit": self.unit,
            "projection": self.projection,
            "bit": self.bit,
            "source_state": self.source_state,
            "destination_state": self.destination_state,
            "page_delta": self.page_delta,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PhysicalPageMove":
        result = cls(
            expert=int(value["expert"]),
            unit=int(value["unit"]),
            bit=int(value["bit"]),
            source_state=int(value["source_state"]),
            destination_state=int(value["destination_state"]),
        )
        if str(value.get("projection", result.projection)) != result.projection:
            raise ValueError("serialized page-move projection changed")
        if int(value.get("page_delta", result.page_delta)) != result.page_delta:
            raise ValueError("serialized page-move charge changed")
        return result


def _move_for(expert: int, unit: int, bit: int, source: int, direction: str) -> PhysicalPageMove:
    destination = source | bit if direction == "add" else source & ~bit
    return PhysicalPageMove(expert, unit, bit, source, destination)


def _build_legal_move_templates(
    direction: str,
) -> tuple[tuple[tuple[tuple[PhysicalPageMove, ...], ...], ...], ...]:
    """Intern every immutable one-unit move in canonical enumeration order.

    Local prune/completion paths enumerate the same physical move values after
    every refresh. Constructing hundreds of thousands of identical
    dataclass objects in each path is pure Python overhead. These tables are
    built once when the module is imported (before the runner forks its Arm-3
    workers), so their immutable pages can also be shared by forked children.
    Masked enumeration retains the explicit reference loop below because its
    legal set depends on a second state table.
    """

    if direction not in {"add", "remove"}:
        raise ValueError("move-template direction must be add or remove")
    return tuple(
        tuple(
            tuple(
                tuple(
                    _move_for(expert, unit, bit, source, direction)
                    for bit, _ in PROJECTION_BITS
                    if (
                        not source & bit
                        if direction == "add"
                        else bool(source & bit)
                    )
                )
                for source in range(VALID_STATE_MASK + 1)
            )
            for unit in range(UNITS_PER_EXPERT)
        )
        for expert in range(EXPERTS_PER_GROUP)
    )


_LEGAL_ADD_MOVE_TEMPLATES = _build_legal_move_templates("add")
_LEGAL_REMOVE_MOVE_TEMPLATES = _build_legal_move_templates("remove")


def legal_add_moves(states: Any, *, upper_states: Any | None = None) -> tuple[PhysicalPageMove, ...]:
    """Enumerate add-one-bit moves in deterministic expert/unit/bit order."""

    value = validate_states(states)
    upper = None if upper_states is None else validate_states(upper_states)
    if upper is not None and not physical_subset(value, upper):
        raise ValueError("current states are not a subset of the addition upper mask")
    result: list[PhysicalPageMove] = []
    if upper is None:
        for expert in range(EXPERTS_PER_GROUP):
            templates = _LEGAL_ADD_MOVE_TEMPLATES[expert]
            for unit in range(UNITS_PER_EXPERT):
                result.extend(templates[unit][int(value[expert, unit])])
        return tuple(result)
    for expert in range(EXPERTS_PER_GROUP):
        for unit in range(UNITS_PER_EXPERT):
            source = int(value[expert, unit])
            for bit, _ in PROJECTION_BITS:
                if source & bit:
                    continue
                if upper is not None and not int(upper[expert, unit]) & bit:
                    continue
                result.append(_move_for(expert, unit, bit, source, "add"))
    return tuple(result)


def legal_remove_moves(
    states: Any,
    *,
    lower_states: Any | None = None,
) -> tuple[PhysicalPageMove, ...]:
    """Enumerate remove-one-bit moves without clearing a supplied lower mask."""

    value = validate_states(states)
    lower = None if lower_states is None else validate_states(lower_states)
    if lower is not None and not physical_subset(lower, value):
        raise ValueError("removal lower mask is not a subset of current states")
    result: list[PhysicalPageMove] = []
    if lower is None:
        for expert in range(EXPERTS_PER_GROUP):
            templates = _LEGAL_REMOVE_MOVE_TEMPLATES[expert]
            for unit in range(UNITS_PER_EXPERT):
                result.extend(templates[unit][int(value[expert, unit])])
        return tuple(result)
    for expert in range(EXPERTS_PER_GROUP):
        for unit in range(UNITS_PER_EXPERT):
            source = int(value[expert, unit])
            for bit, _ in PROJECTION_BITS:
                if not source & bit:
                    continue
                if lower is not None and int(lower[expert, unit]) & bit:
                    continue
                result.append(_move_for(expert, unit, bit, source, "remove"))
    return tuple(result)


def apply_page_move(states: Any, move: PhysicalPageMove) -> np.ndarray:
    return apply_page_moves(states, (move,))


def apply_page_moves(states: Any, moves: Sequence[PhysicalPageMove]) -> np.ndarray:
    """Apply a stale-checked batch; no two actions may touch one expert-unit."""

    result = validate_states(states)
    seen: set[tuple[int, int]] = set()
    for raw in moves:
        if not isinstance(raw, PhysicalPageMove):
            raise TypeError("page move batches require PhysicalPageMove values")
        key = (raw.expert, raw.unit)
        if key in seen:
            raise ValueError("batched page moves must target distinct expert units")
        seen.add(key)
        if int(result[key]) != raw.source_state:
            raise ValueError("physical page move is stale")
        result[key] = raw.destination_state
    return result


def _state_payload(expert_ids: np.ndarray, states: np.ndarray) -> dict[str, Any]:
    return {
        "expert_ids": np.asarray(expert_ids, np.int64).tolist(),
        "states": np.asarray(states, np.int64).tolist(),
    }


def _state_sha256(expert_ids: np.ndarray, states: np.ndarray) -> str:
    encoded = json.dumps(
        _state_payload(expert_ids, states),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AllocationCheckpoint:
    """One exact, complete physical state and the moves from its predecessor."""

    phase: str
    step: int
    expert_ids: np.ndarray
    states: np.ndarray
    local_damage: float
    moves: tuple[PhysicalPageMove, ...] = ()

    def __post_init__(self) -> None:
        phase = str(self.phase)
        step = int(self.step)
        ids = validate_expert_ids(self.expert_ids)
        states = validate_states(self.states)
        damage = float(self.local_damage)
        moves = tuple(self.moves)
        if not phase or step < 0:
            raise ValueError("checkpoint phase/step is invalid")
        if not np.isfinite(damage) or damage < 0.0:
            raise ValueError("checkpoint local damage must be finite and nonnegative")
        if any(not isinstance(move, PhysicalPageMove) for move in moves):
            raise TypeError("checkpoint moves must be physical page moves")
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "expert_ids", _freeze(ids))
        object.__setattr__(self, "states", _freeze(states))
        object.__setattr__(self, "local_damage", damage)
        object.__setattr__(self, "moves", moves)

    @property
    def expert_pages(self) -> np.ndarray:
        return expert_page_counts(self.states)

    @property
    def group_pages(self) -> int:
        return state_page_count(self.states)

    @property
    def state_sha256(self) -> str:
        return _state_sha256(self.expert_ids, self.states)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CHECKPOINT_SCHEMA,
            "phase": self.phase,
            "step": self.step,
            **_state_payload(self.expert_ids, self.states),
            "expert_pages": self.expert_pages.tolist(),
            "group_pages": self.group_pages,
            "local_damage": self.local_damage,
            "moves": [move.to_dict() for move in self.moves],
            "state_sha256": self.state_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AllocationCheckpoint":
        if str(value.get("schema")) != CHECKPOINT_SCHEMA:
            raise ValueError("unexpected physical checkpoint schema")
        result = cls(
            phase=str(value["phase"]),
            step=int(value["step"]),
            expert_ids=np.asarray(value["expert_ids"]),
            states=np.asarray(value["states"]),
            local_damage=float(value["local_damage"]),
            moves=tuple(PhysicalPageMove.from_dict(item) for item in value["moves"]),
        )
        validate_page_parity(
            result.states,
            expected_group_pages=int(value["group_pages"]),
            expected_expert_pages=np.asarray(value["expert_pages"]),
        )
        if str(value.get("state_sha256")) != result.state_sha256:
            raise ValueError("serialized checkpoint state digest changed")
        return result


@dataclass(frozen=True)
class AllocationPathTrace:
    """A replayable prune or add-only path with complete checkpoints."""

    phase: str
    target_pages: int
    stop_reason: str
    checkpoints: tuple[AllocationCheckpoint, ...]

    def __post_init__(self) -> None:
        phase = str(self.phase)
        target = int(self.target_pages)
        reason = str(self.stop_reason)
        checkpoints = tuple(self.checkpoints)
        if phase not in {"reverse_local_prune", "add_only_completion"}:
            raise ValueError("physical allocation path phase is invalid")
        if target < 0 or not reason or not checkpoints:
            raise ValueError("physical allocation path metadata is invalid")
        first = checkpoints[0]
        if first.phase != phase or first.step != 0 or first.moves:
            raise ValueError("physical allocation path has an invalid initial checkpoint")
        for expected_step, checkpoint in enumerate(checkpoints):
            if checkpoint.phase != phase or checkpoint.step != expected_step:
                raise ValueError("physical allocation checkpoint sequence changed")
            validate_expert_id_alignment(first.expert_ids, checkpoint.expert_ids)
            if expected_step == 0:
                continue
            previous = checkpoints[expected_step - 1]
            if not checkpoint.moves:
                raise ValueError("noninitial checkpoint has no physical move")
            replayed = apply_page_moves(previous.states, checkpoint.moves)
            if not np.array_equal(replayed, checkpoint.states):
                raise ValueError("checkpoint moves do not reproduce its physical state")
            if phase == "reverse_local_prune":
                if any(move.direction != "remove" for move in checkpoint.moves):
                    raise ValueError("reverse-local path contains an add move")
                if not physical_subset(checkpoint.states, previous.states):
                    raise ValueError("reverse-local path is not monotonically decreasing")
            else:
                if any(move.direction != "add" for move in checkpoint.moves):
                    raise ValueError("add-only path contains a remove move")
                if not physical_subset(previous.states, checkpoint.states):
                    raise ValueError("add-only path cleared a frozen physical page")
        if checkpoints[-1].group_pages > target:
            raise ValueError("physical allocation path stopped above its target cap")
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "target_pages", target)
        object.__setattr__(self, "stop_reason", reason)
        object.__setattr__(self, "checkpoints", checkpoints)

    @property
    def final(self) -> AllocationCheckpoint:
        return self.checkpoints[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PATH_SCHEMA,
            "phase": self.phase,
            "target_pages": self.target_pages,
            "stop_reason": self.stop_reason,
            "checkpoints": [checkpoint.to_dict() for checkpoint in self.checkpoints],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AllocationPathTrace":
        if str(value.get("schema")) != PATH_SCHEMA:
            raise ValueError("unexpected physical allocation path schema")
        return cls(
            phase=str(value["phase"]),
            target_pages=int(value["target_pages"]),
            stop_reason=str(value["stop_reason"]),
            checkpoints=tuple(
                AllocationCheckpoint.from_dict(item)
                for item in value["checkpoints"]
            ),
        )


def _finite_nonnegative_score(score: Any, name: str) -> float:
    result = float(score)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must return finite nonnegative local damage")
    return result


def _read_only_states(states: np.ndarray) -> np.ndarray:
    return _freeze(validate_states(states))


def _candidate_local_scores(
    states: np.ndarray,
    moves: tuple[PhysicalPageMove, ...],
    local_damage: StateScore,
    score_moves: MoveScore | None,
) -> np.ndarray:
    if not moves:
        return np.empty(0, np.float64)
    current = _read_only_states(states)
    if score_moves is None:
        result = np.asarray([
            _finite_nonnegative_score(
                local_damage(apply_page_move(current, move)), "local_damage",
            )
            for move in moves
        ], np.float64)
    else:
        result = np.asarray(score_moves(current, moves), np.float64).reshape(-1)
        if result.shape != (len(moves),):
            raise ValueError("move-score callback must return one value per move")
        if np.any(~np.isfinite(result)) or np.any(result < 0.0):
            raise ValueError("move-score callback returned invalid local damage")
    return result


def _admissible_mask(
    states: np.ndarray,
    moves: tuple[PhysicalPageMove, ...],
    callback: MoveAdmissibility | None,
) -> np.ndarray:
    if callback is None:
        return np.ones(len(moves), dtype=bool)
    result = np.asarray(callback(_read_only_states(states), moves))
    if result.shape != (len(moves),) or result.dtype != np.bool_:
        raise ValueError("move-admissibility callback must return one boolean per move")
    return result.copy()


def _best_move(moves: tuple[PhysicalPageMove, ...], scores: np.ndarray) -> int:
    return min(
        range(len(moves)),
        key=lambda index: (float(scores[index]), moves[index].sort_key),
    )


def _best_distinct_move_batch(
    moves: tuple[PhysicalPageMove, ...],
    scores: np.ndarray,
    maximum: int,
    *,
    eligible: np.ndarray | None = None,
) -> tuple[PhysicalPageMove, ...]:
    """Take a deterministic score-ordered batch with distinct expert-units."""

    limit = int(maximum)
    if limit < 1:
        raise ValueError("physical move-batch size must be positive")
    allowed = (
        np.ones(len(moves), dtype=bool)
        if eligible is None
        else np.asarray(eligible)
    )
    if allowed.shape != (len(moves),) or allowed.dtype != np.bool_:
        raise ValueError("physical move eligibility must be one boolean per move")
    # Legal move enumeration, and every admissibility-filtered subsequence of
    # it, is already in canonical sort-key order. A stable C-level score sort
    # therefore implements exactly the historical `(score, sort_key)` order
    # without constructing and sorting thousands of Python key tuples. Keep
    # the generic reference path for callers that supply noncanonical moves.
    canonical = all(
        left.sort_key <= right.sort_key
        for left, right in zip(moves, moves[1:])
    )
    order: Sequence[int]
    if canonical:
        order = np.argsort(scores, kind="stable")
    else:
        order = sorted(
            range(len(moves)),
            key=lambda index: (float(scores[index]), moves[index].sort_key),
        )
    selected: list[PhysicalPageMove] = []
    seen: set[tuple[int, int]] = set()
    for index in order:
        move = moves[index]
        key = (move.expert, move.unit)
        if not bool(allowed[index]) or key in seen:
            continue
        selected.append(move)
        seen.add(key)
        if len(selected) == limit:
            break
    return tuple(selected)


def _checkpoint(
    phase: str,
    step: int,
    expert_ids: np.ndarray,
    states: np.ndarray,
    local_damage: float,
    moves: tuple[PhysicalPageMove, ...] = (),
) -> AllocationCheckpoint:
    return AllocationCheckpoint(
        phase=phase,
        step=step,
        expert_ids=expert_ids,
        states=states,
        local_damage=local_damage,
        moves=moves,
    )


def reverse_local_prune_to_reserve(
    high_reference: Any,
    expert_ids: Any,
    *,
    budget_pages: int,
    repair_window_pages: int,
    local_damage: StateScore,
    score_removals: MoveScore | None = None,
    score_tolerance: float = 1e-9,
    refresh_after_accepted_pages: int = 1,
) -> AllocationPathTrace:
    """D1-blind reverse-local pruning from ``B`` to reserve cap ``B-W``.

    Every arm must share this same core.  The callback surface contains no D1
    value: it ranks physical one-page removals solely by the resulting local
    damage.  A vectorized runner may provide ``score_removals``; the scalar
    ``local_damage`` callback remains the exact checkpoint parity gate.  A
    refresh interval larger than one applies a deterministic distinct-unit
    batch before rescoring.
    """

    states = validate_states(high_reference)
    ids = validate_expert_ids(expert_ids)
    budget = int(budget_pages)
    window = int(repair_window_pages)
    tolerance = float(score_tolerance)
    refresh = int(refresh_after_accepted_pages)
    if (
        budget < 0
        or window < 0
        or window > budget
        or tolerance < 0.0
        or refresh < 1
    ):
        raise ValueError("reverse-local budget/window/tolerance is invalid")
    pages = state_page_count(states)
    target = budget - window
    score = _finite_nonnegative_score(local_damage(_read_only_states(states)), "local_damage")
    checkpoints = [_checkpoint("reverse_local_prune", 0, ids, states, score)]
    if pages <= target:
        return AllocationPathTrace(
            "reverse_local_prune", target,
            "reference_already_within_reserve_cap", tuple(checkpoints),
        )
    while pages > target:
        moves = legal_remove_moves(states)
        if not moves:
            raise RuntimeError("reverse-local pruning exhausted physical pages above cap")
        scores = _candidate_local_scores(
            states, moves, local_damage, score_removals,
        )
        selected = _best_distinct_move_batch(
            moves, scores, min(refresh, pages - target),
        )
        if not selected:
            raise RuntimeError("reverse-local pruning could not form a physical batch")
        candidate = apply_page_moves(states, selected)
        exact = _finite_nonnegative_score(
            local_damage(_read_only_states(candidate)), "local_damage",
        )
        if len(selected) == 1:
            chosen = moves.index(selected[0])
            if not np.isclose(
                exact, float(scores[chosen]), rtol=tolerance, atol=tolerance,
            ):
                raise RuntimeError("reverse-local move score lost exact local-damage parity")
        states, score = candidate, exact
        pages -= len(selected)
        if state_page_count(states) != pages or not physical_subset(states, high_reference):
            raise RuntimeError("reverse-local physical subset/page invariant failed")
        checkpoints.append(_checkpoint(
            "reverse_local_prune", len(checkpoints), ids, states, score, selected,
        ))
    return AllocationPathTrace(
        "reverse_local_prune", target, "reserve_cap_reached", tuple(checkpoints),
    )


def add_only_local_completion(
    frozen_core: Any,
    expert_ids: Any,
    *,
    budget_pages: int,
    local_damage: StateScore,
    score_additions: MoveScore | None = None,
    admissible_additions: MoveAdmissibility | None = None,
    upper_states: Any | None = None,
    minimum_local_improvement: float = 0.0,
    score_tolerance: float = 1e-9,
    refresh_after_accepted_pages: int = 1,
) -> AllocationPathTrace:
    """Greedily complete a frozen core using only useful admissible additions.

    The runner may use ``admissible_additions`` for a D1-only screen or exact
    replay gate.  The physical core itself never evaluates a downstream or
    terminal objective.  It stops below budget when no allowed one-page move
    has strictly positive exact local-qenergy gain.  A refresh interval larger
    than one scores once, takes a deterministic distinct-unit batch, and then
    recomputes exact combined local damage for the resulting checkpoint.
    """

    core = validate_states(frozen_core)
    states = core.copy()
    ids = validate_expert_ids(expert_ids)
    upper = None if upper_states is None else validate_states(upper_states)
    if upper is not None and not physical_subset(core, upper):
        raise ValueError("frozen core is not a subset of the completion upper mask")
    budget = int(budget_pages)
    minimum = float(minimum_local_improvement)
    tolerance = float(score_tolerance)
    refresh = int(refresh_after_accepted_pages)
    if budget < 0 or minimum < 0.0 or tolerance < 0.0 or refresh < 1:
        raise ValueError("completion budget/improvement/tolerance is invalid")
    pages = state_page_count(states)
    if pages > budget:
        raise ValueError("frozen core exceeds the completion budget")
    score = _finite_nonnegative_score(local_damage(_read_only_states(states)), "local_damage")
    checkpoints = [_checkpoint("add_only_completion", 0, ids, states, score)]
    stop_reason = "budget_reached" if pages == budget else ""
    while pages < budget:
        moves = legal_add_moves(states, upper_states=upper)
        if not moves:
            stop_reason = "no_legal_addition"
            break
        admissible = _admissible_mask(states, moves, admissible_additions)
        if not np.any(admissible):
            stop_reason = "no_admissible_addition"
            break
        retained = tuple(move for move, keep in zip(moves, admissible.tolist()) if keep)
        scores = _candidate_local_scores(
            states, retained, local_damage, score_additions,
        )
        useful = scores < score - minimum
        selected = _best_distinct_move_batch(
            retained,
            scores,
            min(refresh, budget - pages),
            eligible=useful,
        )
        if not selected:
            stop_reason = "no_positive_local_gain"
            break
        # Single-page refresh preserves the exact candidate-score parity gate.
        # For a stale-score batch, its exact combined score is not equal to any
        # one candidate score. Retain the longest deterministic prefix that
        # still gives positive exact local recovery.
        accepted: tuple[PhysicalPageMove, ...] = ()
        candidate = states
        exact = score
        for prefix_length in range(len(selected), 0, -1):
            proposed = selected[:prefix_length]
            proposed_states = apply_page_moves(states, proposed)
            proposed_exact = _finite_nonnegative_score(
                local_damage(_read_only_states(proposed_states)), "local_damage",
            )
            if proposed_exact < score - minimum:
                accepted = proposed
                candidate = proposed_states
                exact = proposed_exact
                break
        if not accepted:
            stop_reason = "no_positive_local_gain"
            break
        if len(selected) == 1:
            chosen = retained.index(selected[0])
            if not np.isclose(
                exact, float(scores[chosen]), rtol=tolerance, atol=tolerance,
            ):
                raise RuntimeError("add-only move score lost exact local-damage parity")
        states, score = candidate, exact
        pages += len(accepted)
        if (
            state_page_count(states) != pages
            or not physical_subset(core, states)
            or (upper is not None and not physical_subset(states, upper))
        ):
            raise RuntimeError("add-only frozen-core/page invariant failed")
        checkpoints.append(_checkpoint(
            "add_only_completion", len(checkpoints), ids, states, score, accepted,
        ))
    if not stop_reason:
        stop_reason = "budget_reached"
    return AllocationPathTrace(
        "add_only_completion", budget, stop_reason, tuple(checkpoints),
    )


@dataclass(frozen=True)
class RouteRepairMetrics:
    """D1-only metrics supplied for one already-frozen repair candidate."""

    d1_crossings: int
    arm5_severity: float
    local_damage: float

    def __post_init__(self) -> None:
        crossings = int(self.d1_crossings)
        severity = float(self.arm5_severity)
        damage = float(self.local_damage)
        if crossings < 0:
            raise ValueError("D1 crossing count must be nonnegative")
        if not np.isfinite(severity) or severity < 0.0:
            raise ValueError("Arm5 D1 severity must be finite and nonnegative")
        if not np.isfinite(damage) or damage < 0.0:
            raise ValueError("repair local damage must be finite and nonnegative")
        object.__setattr__(self, "d1_crossings", crossings)
        object.__setattr__(self, "arm5_severity", severity)
        object.__setattr__(self, "local_damage", damage)


@dataclass(frozen=True)
class RouteRepairSelection:
    index: int
    name: str
    reason: str


def hard_saturating_route_repair(
    candidate_names: Sequence[str],
    candidates: Sequence[AllocationCheckpoint],
    metrics: Sequence[RouteRepairMetrics],
    *,
    incumbent_index: int = 0,
    frozen_core: AllocationCheckpoint | None = None,
    page_budget: int | None = None,
    maximum_page_distance: int | None = None,
    require_matched_pages: bool = True,
) -> RouteRepairSelection:
    """Select a bounded repair without replacing a safe D1 incumbent.

    The candidate sequence is supplied and already frozen; this function does
    not generate or mutate states.  An unsafe incumbent can be replaced only
    by a candidate with strictly fewer D1 crossings.  Arm5 severity is used
    only for candidates that remain unsafe.  Once a candidate is D1-safe,
    route severity is saturated and local damage becomes the first tie-break.
    """

    names = tuple(map(str, candidate_names))
    frozen = tuple(candidates)
    values = tuple(metrics)
    count = len(names)
    incumbent = int(incumbent_index)
    if (
        count == 0
        or len(frozen) != count
        or len(values) != count
        or len(set(names)) != count
        or any(not name for name in names)
        or incumbent < 0
        or incumbent >= count
    ):
        raise ValueError("frozen route-repair candidate sequence is invalid")
    if any(not isinstance(item, AllocationCheckpoint) for item in frozen):
        raise TypeError("repair candidates must be allocation checkpoints")
    if any(not isinstance(item, RouteRepairMetrics) for item in values):
        raise TypeError("repair metrics must be RouteRepairMetrics values")
    reference = frozen[incumbent]
    budget = None if page_budget is None else int(page_budget)
    radius = None if maximum_page_distance is None else int(maximum_page_distance)
    if budget is not None and budget < 0:
        raise ValueError("repair page budget must be nonnegative")
    if radius is not None and radius < 0:
        raise ValueError("repair page distance must be nonnegative")
    if frozen_core is not None:
        validate_expert_id_alignment(reference.expert_ids, frozen_core.expert_ids)
    for checkpoint in frozen:
        validate_expert_id_alignment(reference.expert_ids, checkpoint.expert_ids)
        if budget is not None and checkpoint.group_pages > budget:
            raise ValueError("repair candidate exceeds the page budget")
        if bool(require_matched_pages) and checkpoint.group_pages != reference.group_pages:
            raise ValueError("repair candidate does not match incumbent page count")
        if frozen_core is not None and not physical_subset(
            frozen_core.states, checkpoint.states,
        ):
            raise ValueError("repair candidate clears a frozen-core page")
        if radius is not None and physical_page_distance(
            reference.states, checkpoint.states,
        ) > radius:
            raise ValueError("repair candidate exceeds the physical page-distance bound")
    incumbent_metrics = values[incumbent]
    if incumbent_metrics.d1_crossings == 0:
        return RouteRepairSelection(incumbent, names[incumbent], "incumbent_d1_safe")
    eligible = [
        index for index, value in enumerate(values)
        if value.d1_crossings < incumbent_metrics.d1_crossings
    ]
    if not eligible:
        return RouteRepairSelection(
            incumbent, names[incumbent], "no_strict_crossing_repair",
        )

    def key(index: int) -> tuple[Any, ...]:
        value = values[index]
        # Arm5 is a route-severity ablation, not a reward after certification.
        unsafe_severity = value.arm5_severity if value.d1_crossings > 0 else 0.0
        return (
            value.d1_crossings,
            unsafe_severity,
            value.local_damage,
            abs(frozen[index].group_pages - reference.group_pages),
            physical_page_distance(reference.states, frozen[index].states),
            names[index],
            index,
        )

    selected = min(eligible, key=key)
    return RouteRepairSelection(selected, names[selected], "strict_crossing_repair")
