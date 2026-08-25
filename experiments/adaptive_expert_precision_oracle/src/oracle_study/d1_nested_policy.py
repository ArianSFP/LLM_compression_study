"""D1-only nested policy orchestration for one low/high rate pair.

The policy composes physical state allocation with exact all-expert D1 repair.
Terminal quality and downstream routes are deliberately absent. Returned
allocations have canonical add-only paths from one common core; matched
remove/add swaps remain a separate audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .d1_nested_allocation import (
    PROJECTION_BITS,
    PhysicalPageMove,
    add_only_local_completion,
    apply_page_move,
    apply_page_moves,
    physical_subset,
    state_page_count,
    validate_expert_ids,
    validate_states,
)
from .d1_nested_geometry import TokenStateGeometry
from .d1_nested_search import (
    ExactD1Metrics,
    ExactRepairResult,
    ExactRepairStep,
    ScreenedD1Boundaries,
    iterative_exact_repair,
)


ExactReplay = Callable[[np.ndarray], ExactD1Metrics]


__all__ = [
    "NestedPolicyCheckpoint",
    "PolicyRepairAudit",
    "NestedPolicyPairResult",
    "canonical_add_only_moves",
    "replay_add_only_moves",
    "orchestrate_nested_policy_pair",
]


def _freeze(value: np.ndarray, dtype: np.dtype | type = np.int64) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _finite_damage(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _same_metrics(left: ExactD1Metrics, right: ExactD1Metrics) -> bool:
    return (
        left.membership_crossings == right.membership_crossings
        and np.array_equal(left.baseline_top8, right.baseline_top8)
        and np.array_equal(left.candidate_top8, right.candidate_top8)
        and np.array_equal(left.lost_target_experts, right.lost_target_experts)
        and np.array_equal(left.entering_outsiders, right.entering_outsiders)
        and left.violation_depth == right.violation_depth
        and left.routing_mass_lost == right.routing_mass_lost
        and left.target_set_margin == right.target_set_margin
    )


def canonical_add_only_moves(
    source_states: Any,
    destination_states: Any,
) -> tuple[PhysicalPageMove, ...]:
    """Return the deterministic expert/unit/down-up-gate add-only chain."""

    source = validate_states(source_states)
    destination = validate_states(destination_states)
    if not physical_subset(source, destination):
        raise ValueError("canonical add-only source is not a subset of destination")
    current = source.copy()
    result: list[PhysicalPageMove] = []
    for expert in range(current.shape[0]):
        for unit in range(current.shape[1]):
            target = int(destination[expert, unit])
            for bit, _ in PROJECTION_BITS:
                state = int(current[expert, unit])
                if state & bit or not target & bit:
                    continue
                move = PhysicalPageMove(
                    expert=expert,
                    unit=unit,
                    bit=bit,
                    source_state=state,
                    destination_state=state | bit,
                )
                current = apply_page_move(current, move)
                result.append(move)
    if not np.array_equal(current, destination):
        raise RuntimeError("canonical add-only chain did not reach destination")
    return tuple(result)


def replay_add_only_moves(
    source_states: Any,
    moves: Sequence[PhysicalPageMove],
) -> np.ndarray:
    """Replay an ordered add-only chain, retaining same-unit dependencies."""

    current = validate_states(source_states)
    for move in moves:
        if not isinstance(move, PhysicalPageMove) or move.direction != "add":
            raise ValueError("canonical chain contains a non-add physical move")
        current = apply_page_move(current, move)
    return current


def _validate_repair_chain(
    result: ExactRepairResult,
    frozen_core: np.ndarray,
) -> None:
    current = validate_states(result.initial_state)
    pages = state_page_count(current)
    if not physical_subset(frozen_core, current):
        raise ValueError("repair initial state clears its frozen core")
    metrics = result.initial_metrics
    for expected_round, step in enumerate(result.steps, start=1):
        if step.round_index != expected_round:
            raise ValueError("repair audit round indices are not consecutive")
        candidate = apply_page_moves(current, (step.removal, step.addition))
        if not np.array_equal(candidate, step.state):
            raise ValueError("repair swap audit does not replay its state")
        if (
            state_page_count(candidate) != pages
            or not physical_subset(frozen_core, candidate)
            or step.metrics.membership_crossings >= metrics.membership_crossings
        ):
            raise ValueError("repair audit violates page/core/strict-D1 invariants")
        current = candidate
        metrics = step.metrics
    if (
        not np.array_equal(current, result.final_state)
        or not _same_metrics(metrics, result.final_metrics)
    ):
        raise ValueError("repair audit does not reproduce final state/metrics")


@dataclass(frozen=True)
class NestedPolicyCheckpoint:
    """One accepted high checkpoint after additions and optional swap repair."""

    checkpoint_index: int
    state: np.ndarray
    metrics: ExactD1Metrics
    local_damage: float
    completion_moves: tuple[PhysicalPageMove, ...] = ()
    repair: ExactRepairResult | None = None

    def __post_init__(self) -> None:
        index = int(self.checkpoint_index)
        state = validate_states(self.state)
        damage = _finite_damage(self.local_damage, "checkpoint local damage")
        moves = tuple(self.completion_moves)
        if index < 0 or not isinstance(self.metrics, ExactD1Metrics):
            raise ValueError("nested-policy checkpoint metadata is invalid")
        if any(
            not isinstance(move, PhysicalPageMove) or move.direction != "add"
            for move in moves
        ):
            raise ValueError("nested-policy completion moves must be add-only")
        if self.repair is not None and not isinstance(
            self.repair, ExactRepairResult,
        ):
            raise TypeError("checkpoint repair must be ExactRepairResult")
        object.__setattr__(self, "checkpoint_index", index)
        object.__setattr__(self, "state", _freeze(state))
        object.__setattr__(self, "local_damage", damage)
        object.__setattr__(self, "completion_moves", moves)

    @property
    def pages(self) -> int:
        return state_page_count(self.state)


@dataclass(frozen=True)
class PolicyRepairAudit:
    """One accepted or rejected high-path matched-swap repair attempt."""

    attempt_index: int
    allowed_crossings: int
    accepted: bool
    result: ExactRepairResult

    def __post_init__(self) -> None:
        index = int(self.attempt_index)
        allowed = int(self.allowed_crossings)
        accepted = bool(self.accepted)
        if (
            index < 1
            or allowed < 0
            or not isinstance(self.result, ExactRepairResult)
        ):
            raise ValueError("policy repair audit is invalid")
        restored = self.result.final_metrics.membership_crossings <= allowed
        if restored != accepted:
            raise ValueError("repair acceptance disagrees with D1 allowance")
        object.__setattr__(self, "attempt_index", index)
        object.__setattr__(self, "allowed_crossings", allowed)
        object.__setattr__(self, "accepted", accepted)


@dataclass(frozen=True)
class NestedPolicyPairResult:
    """Immutable D1-only result for one physical low/high allocation pair."""

    arm: str
    expert_ids: np.ndarray
    common_core: np.ndarray
    arm3_low_state: np.ndarray
    arm3_high_state: np.ndarray
    low_state: np.ndarray
    high_state: np.ndarray
    low_metrics: ExactD1Metrics
    high_metrics: ExactD1Metrics
    arm3_high_metrics: ExactD1Metrics
    arm3_low_damage: float
    arm3_high_damage: float
    low_local_damage: float
    high_local_damage: float
    low_local_damage_limit: float
    high_reference_damage_limit: float
    eta: float
    j_q2: float
    low_cap_pages: int
    high_cap_pages: int
    refresh_after_accepted_pages: int
    low_stop_reason: str
    high_stop_reason: str
    core_to_low_moves: tuple[PhysicalPageMove, ...]
    core_to_high_moves: tuple[PhysicalPageMove, ...]
    low_repair: ExactRepairResult | None
    high_checkpoints: tuple[NestedPolicyCheckpoint, ...]
    high_repair_audits: tuple[PolicyRepairAudit, ...]
    pair_fallback_high_guardrail_infeasible: bool = False
    high_guardrail_qualified_checkpoints: int = 0
    high_guardrail_repair_attempts: int = 0

    def __post_init__(self) -> None:
        arm = str(self.arm)
        ids = validate_expert_ids(self.expert_ids)
        core = validate_states(self.common_core)
        arm3_low = validate_states(self.arm3_low_state)
        arm3_high = validate_states(self.arm3_high_state)
        low = validate_states(self.low_state)
        high = validate_states(self.high_state)
        low_cap = int(self.low_cap_pages)
        high_cap = int(self.high_cap_pages)
        refresh = int(self.refresh_after_accepted_pages)
        eta = _finite_damage(self.eta, "eta")
        j_q2 = _finite_damage(self.j_q2, "J_Q2")
        reasons = (str(self.low_stop_reason), str(self.high_stop_reason))
        low_moves = tuple(self.core_to_low_moves)
        high_moves = tuple(self.core_to_high_moves)
        checkpoints = tuple(self.high_checkpoints)
        audits = tuple(self.high_repair_audits)
        if not isinstance(
            self.pair_fallback_high_guardrail_infeasible, (bool, np.bool_),
        ):
            raise TypeError("pair fallback flag must be boolean")
        fallback = bool(self.pair_fallback_high_guardrail_infeasible)
        qualified = int(self.high_guardrail_qualified_checkpoints)
        repair_attempts = int(self.high_guardrail_repair_attempts)
        damage_names = (
            "arm3 low damage",
            "arm3 high damage",
            "low local damage",
            "high local damage",
            "low local damage limit",
            "high reference damage limit",
        )
        damages = tuple(
            _finite_damage(value, name)
            for value, name in zip((
                self.arm3_low_damage,
                self.arm3_high_damage,
                self.low_local_damage,
                self.high_local_damage,
                self.low_local_damage_limit,
                self.high_reference_damage_limit,
            ), damage_names, strict=True)
        )
        if arm not in {"arm4", "arm5"}:
            raise ValueError("nested policy arm must be arm4 or arm5")
        if (
            low_cap < 0
            or high_cap < low_cap
            or refresh < 1
            or not all(reasons)
            or not isinstance(self.low_metrics, ExactD1Metrics)
            or not isinstance(self.high_metrics, ExactD1Metrics)
            or not isinstance(self.arm3_high_metrics, ExactD1Metrics)
            or qualified < 0
            or repair_attempts < 0
        ):
            raise ValueError("nested policy scalar metadata is invalid")
        if not (
            physical_subset(core, arm3_low)
            and physical_subset(arm3_low, arm3_high)
            and physical_subset(core, low)
            and physical_subset(low, high)
        ):
            raise ValueError("nested policy states violate literal nesting")
        if (
            state_page_count(arm3_low) > low_cap
            or state_page_count(arm3_high) > high_cap
            or state_page_count(low) != state_page_count(arm3_low)
            or state_page_count(high) > high_cap
        ):
            raise ValueError("nested policy violates cap or matched-low pages")
        tolerance = 1e-12
        if (
            abs(damages[4] - (damages[0] + eta * j_q2)) > tolerance
            or abs(damages[5] - (damages[1] + eta * j_q2)) > tolerance
            or damages[2] > damages[4] + tolerance
            or damages[3] > damages[5] + tolerance
        ):
            raise ValueError("nested policy violates its per-endpoint local guardrail")
        if not fallback and (
            self.high_metrics.membership_crossings
            > self.low_metrics.membership_crossings
            or (self.low_metrics.safe and not self.high_metrics.safe)
        ):
            raise ValueError("nested high endpoint worsens exact D1 membership")
        high_changed = not np.array_equal(high, arm3_high)
        if not high_changed and not _same_metrics(
            self.high_metrics, self.arm3_high_metrics,
        ):
            raise ValueError("identical Arm3/high states changed exact D1 metrics")
        if not fallback:
            if self.arm3_high_metrics.safe and high_changed:
                raise ValueError("nested policy changed a D1-safe Arm3-high incumbent")
            if (
                not self.arm3_high_metrics.safe
                and high_changed
                and self.high_metrics.membership_crossings
                >= self.arm3_high_metrics.membership_crossings
            ):
                raise ValueError(
                    "nested high change lacks a strict same-rate crossing reduction"
                )
        if fallback and (
            not np.array_equal(low, arm3_low)
            or not np.array_equal(high, arm3_high)
            or self.low_repair is not None
            or reasons != (
                "pair_fallback_arm3_low_identity",
                "pair_fallback_high_guardrail_infeasible",
            )
        ):
            raise ValueError("high-guardrail fallback changed its Arm3 pair")
        if not np.array_equal(replay_add_only_moves(core, low_moves), low):
            raise ValueError("canonical core-to-low chain does not replay")
        if not np.array_equal(replay_add_only_moves(core, high_moves), high):
            raise ValueError("canonical core-to-high chain does not replay")
        if self.low_repair is not None:
            if (
                not isinstance(self.low_repair, ExactRepairResult)
                or self.low_repair.arm != arm
                or not np.array_equal(self.low_repair.initial_state, arm3_low)
                or not np.array_equal(self.low_repair.final_state, low)
            ):
                raise ValueError("low repair disagrees with selected low state")
            _validate_repair_chain(self.low_repair, core)
        if not checkpoints or any(
            not isinstance(item, NestedPolicyCheckpoint) for item in checkpoints
        ):
            raise ValueError("nested high path requires exact checkpoints")
        first = checkpoints[0]
        if (
            first.checkpoint_index != 0
            or first.completion_moves
            or first.repair is not None
            or not np.array_equal(first.state, low)
            or not _same_metrics(first.metrics, self.low_metrics)
        ):
            raise ValueError("nested high path has an invalid low checkpoint")
        previous = first
        for expected_index, checkpoint in enumerate(checkpoints[1:], start=1):
            if checkpoint.checkpoint_index != expected_index:
                raise ValueError("nested high checkpoint indices changed")
            before_repair = replay_add_only_moves(
                previous.state, checkpoint.completion_moves,
            )
            if checkpoint.repair is None:
                if not np.array_equal(before_repair, checkpoint.state):
                    raise ValueError("high completion checkpoint does not replay")
            else:
                repair = checkpoint.repair
                if (
                    repair.arm != arm
                    or not np.array_equal(repair.initial_state, before_repair)
                    or not np.array_equal(repair.final_state, checkpoint.state)
                ):
                    raise ValueError("high repair disagrees with checkpoint")
                _validate_repair_chain(repair, low)
            if (
                not physical_subset(low, checkpoint.state)
                or checkpoint.pages <= previous.pages
                or checkpoint.pages > high_cap
                or (
                    not fallback
                    and checkpoint.metrics.membership_crossings
                    > previous.metrics.membership_crossings
                )
            ):
                raise ValueError("accepted high checkpoint violates invariants")
            previous = checkpoint
        if (
            not np.array_equal(previous.state, high)
            or not _same_metrics(previous.metrics, self.high_metrics)
            or previous.local_damage != damages[3]
            or first.local_damage != damages[2]
        ):
            raise ValueError("nested high final checkpoint changed")
        if (not fallback and repair_attempts != len(audits)) or (
            fallback and audits
        ):
            raise ValueError("high-guardrail repair audit count changed")
        for expected, audit in enumerate(audits, start=1):
            if (
                not isinstance(audit, PolicyRepairAudit)
                or audit.attempt_index != expected
                or audit.result.arm != arm
            ):
                raise ValueError("high repair audit sequence changed")
            _validate_repair_chain(audit.result, low)
        object.__setattr__(self, "arm", arm)
        object.__setattr__(self, "expert_ids", _freeze(ids))
        object.__setattr__(self, "common_core", _freeze(core))
        object.__setattr__(self, "arm3_low_state", _freeze(arm3_low))
        object.__setattr__(self, "arm3_high_state", _freeze(arm3_high))
        object.__setattr__(self, "low_state", _freeze(low))
        object.__setattr__(self, "high_state", _freeze(high))
        object.__setattr__(self, "arm3_low_damage", damages[0])
        object.__setattr__(self, "arm3_high_damage", damages[1])
        object.__setattr__(self, "low_local_damage", damages[2])
        object.__setattr__(self, "high_local_damage", damages[3])
        object.__setattr__(self, "low_local_damage_limit", damages[4])
        object.__setattr__(self, "high_reference_damage_limit", damages[5])
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "j_q2", j_q2)
        object.__setattr__(self, "low_cap_pages", low_cap)
        object.__setattr__(self, "high_cap_pages", high_cap)
        object.__setattr__(self, "refresh_after_accepted_pages", refresh)
        object.__setattr__(self, "low_stop_reason", reasons[0])
        object.__setattr__(self, "high_stop_reason", reasons[1])
        object.__setattr__(self, "core_to_low_moves", low_moves)
        object.__setattr__(self, "core_to_high_moves", high_moves)
        object.__setattr__(self, "high_checkpoints", checkpoints)
        object.__setattr__(self, "high_repair_audits", audits)
        object.__setattr__(
            self, "pair_fallback_high_guardrail_infeasible", fallback,
        )
        object.__setattr__(self, "high_guardrail_qualified_checkpoints", qualified)
        object.__setattr__(self, "high_guardrail_repair_attempts", repair_attempts)

    @property
    def low_repair_steps(self) -> tuple[ExactRepairStep, ...]:
        return () if self.low_repair is None else self.low_repair.steps

    @property
    def high_repair_steps(self) -> tuple[ExactRepairStep, ...]:
        return tuple(
            step
            for audit in self.high_repair_audits
            for step in audit.result.steps
        )


def _flatten_completion_moves(path: Any) -> tuple[PhysicalPageMove, ...]:
    return tuple(
        move
        for checkpoint in path.checkpoints[1:]
        for move in checkpoint.moves
    )


def orchestrate_nested_policy_pair(
    common_core: Any,
    arm3_low_endpoint: Any,
    arm3_high_endpoint: Any,
    expert_ids: Any,
    geometry: TokenStateGeometry,
    boundaries: ScreenedD1Boundaries,
    exact_replay: ExactReplay,
    *,
    eta: float,
    j_q2: float,
    low_cap_pages: int,
    high_cap_pages: int,
    refresh_after_accepted_pages: int,
    arm: str,
    maximum_repair_rounds: int = 8,
    move_limit: int = 32,
    candidate_limit: int = 8,
) -> NestedPolicyPairResult:
    """Build one conservative Arm4/Arm5 low/high nested policy pair."""

    core = validate_states(common_core)
    arm3_low = validate_states(arm3_low_endpoint)
    arm3_high = validate_states(arm3_high_endpoint)
    ids = validate_expert_ids(expert_ids)
    policy = str(arm)
    eta_value = _finite_damage(eta, "eta")
    q2 = _finite_damage(j_q2, "J_Q2")
    low_cap = int(low_cap_pages)
    high_cap = int(high_cap_pages)
    refresh = int(refresh_after_accepted_pages)
    if policy not in {"arm4", "arm5"}:
        raise ValueError("nested policy arm must be arm4 or arm5")
    if not isinstance(boundaries, ScreenedD1Boundaries):
        raise TypeError("boundaries must be ScreenedD1Boundaries")
    if not callable(exact_replay):
        raise TypeError("exact_replay must be callable")
    if (
        low_cap < 0
        or high_cap < low_cap
        or refresh < 1
        or state_page_count(arm3_low) > low_cap
        or state_page_count(arm3_high) > high_cap
        or not physical_subset(core, arm3_low)
        or not physical_subset(arm3_low, arm3_high)
    ):
        raise ValueError("Arm3 endpoints do not define a valid nested cap pair")

    replay_cache: dict[bytes, ExactD1Metrics] = {}
    baseline_top8: np.ndarray | None = None

    def replay(states: np.ndarray) -> ExactD1Metrics:
        nonlocal baseline_top8
        value = validate_states(states)
        key = np.asarray(value, np.uint8).tobytes(order="C")
        if key not in replay_cache:
            metrics = exact_replay(_freeze(value))
            if not isinstance(metrics, ExactD1Metrics):
                raise TypeError("exact_replay must return ExactD1Metrics")
            replay_cache[key] = metrics
        metrics = replay_cache[key]
        if baseline_top8 is None:
            baseline_top8 = np.asarray(metrics.baseline_top8).copy()
        elif not np.array_equal(baseline_top8, metrics.baseline_top8):
            raise ValueError("exact replay changed the Q4 D1 target set")
        return metrics

    def local(states: np.ndarray) -> float:
        return _finite_damage(
            geometry.local_damage(states), "geometry local damage",
        )

    def move_scores(
        states: np.ndarray,
        moves: tuple[PhysicalPageMove, ...],
    ) -> np.ndarray:
        values = np.asarray(
            geometry.score_moves(states, moves), np.float64,
        ).reshape(-1)
        if (
            values.shape != (len(moves),)
            or np.any(~np.isfinite(values))
            or np.any(values < 0.0)
        ):
            raise ValueError("geometry move scores are invalid")
        return values

    arm3_low_damage = local(arm3_low)
    arm3_high_damage = local(arm3_high)
    low_limit = arm3_low_damage + eta_value * q2
    high_reference_limit = arm3_high_damage + eta_value * q2
    initial_low_metrics = replay(arm3_low)
    initial_high_metrics = (
        initial_low_metrics
        if np.array_equal(arm3_high, arm3_low)
        else replay(arm3_high)
    )
    low_repair: ExactRepairResult | None = None
    if initial_low_metrics.safe:
        low_state = arm3_low.copy()
        low_metrics = initial_low_metrics
        low_reason = "arm3_low_exact_safe_identity"
    else:
        low_repair = iterative_exact_repair(
            arm3_low,
            core,
            geometry,
            boundaries,
            local_damage_limit=low_limit,
            exact_replay=replay,
            arm=policy,
            maximum_rounds=maximum_repair_rounds,
            move_limit=move_limit,
            candidate_limit=candidate_limit,
        )
        low_state = np.asarray(low_repair.final_state).copy()
        low_metrics = low_repair.final_metrics
        low_reason = f"low_repair_{low_repair.stop_reason}"
    low_damage = local(low_state)
    if (
        state_page_count(low_state) != state_page_count(arm3_low)
        or not physical_subset(core, low_state)
        or low_damage > low_limit + 1e-12
    ):
        raise RuntimeError("selected low state violates repair invariants")

    high_checkpoints: list[NestedPolicyCheckpoint] = [NestedPolicyCheckpoint(
        checkpoint_index=0,
        state=low_state,
        metrics=low_metrics,
        local_damage=low_damage,
    )]
    endpoint_path = add_only_local_completion(
        low_state,
        ids,
        budget_pages=high_cap,
        local_damage=local,
        score_additions=move_scores,
        refresh_after_accepted_pages=refresh,
    )
    path_checkpoints = tuple(endpoint_path.checkpoints)
    path_pages = tuple(checkpoint.group_pages for checkpoint in path_checkpoints)
    if any(right <= left for left, right in zip(path_pages, path_pages[1:])):
        raise RuntimeError("complete high local path does not increase physical pages")
    qualified_indices = tuple(
        index for index, checkpoint in enumerate(path_checkpoints)
        if float(checkpoint.local_damage) <= high_reference_limit + 1e-12
    )
    high_audits: list[PolicyRepairAudit] = []
    repair_attempts = 0
    selected: tuple[
        int,
        np.ndarray,
        ExactD1Metrics,
        float,
        tuple[PhysicalPageMove, ...],
        ExactRepairResult | None,
    ] | None = None

    scan_candidates: list[
        tuple[int, np.ndarray, float, tuple[PhysicalPageMove, ...]]
    ] = []
    for path_index in qualified_indices:
        checkpoint = path_checkpoints[path_index]
        scan_candidates.append((
            path_index,
            np.asarray(checkpoint.states).copy(),
            float(checkpoint.local_damage),
            tuple(
                move
                for prior in path_checkpoints[1 : path_index + 1]
                for move in prior.moves
            ),
        ))
    if (
        physical_subset(low_state, arm3_high)
        and arm3_high_damage <= high_reference_limit + 1e-12
        and not any(
            np.array_equal(candidate[1], arm3_high)
            for candidate in scan_candidates
        )
    ):
        scan_candidates.append((
            -1,
            arm3_high.copy(),
            arm3_high_damage,
            canonical_add_only_moves(low_state, arm3_high),
        ))
    scan_candidates.sort(key=lambda candidate: (
        -state_page_count(candidate[1]),
        candidate[2],
        np.asarray(candidate[1], np.uint8).tobytes(order="C"),
    ))

    # Scan the complete local path plus a separately reachable Arm3-high
    # identity in highest-page, lowest-local deterministic order.
    for path_index, raw_state, raw_damage, completion_moves in scan_candidates:
        if np.array_equal(raw_state, low_state):
            raw_metrics = low_metrics
        elif np.array_equal(raw_state, arm3_high):
            raw_metrics = initial_high_metrics
        else:
            raw_metrics = replay(raw_state)
        candidate_state = raw_state
        candidate_metrics = raw_metrics
        candidate_damage = raw_damage
        repair: ExactRepairResult | None = None
        raw_is_arm3_high = np.array_equal(raw_state, arm3_high)
        if initial_high_metrics.safe:
            if not raw_is_arm3_high:
                continue
            same_rate_allowed_crossings = 0
        else:
            same_rate_allowed_crossings = (
                initial_high_metrics.membership_crossings
                if raw_is_arm3_high
                else initial_high_metrics.membership_crossings - 1
            )
        allowed_crossings = min(
            low_metrics.membership_crossings,
            same_rate_allowed_crossings,
        )
        if raw_metrics.membership_crossings > allowed_crossings:
            if initial_high_metrics.safe:
                continue
            repair_attempts += 1
            repair = iterative_exact_repair(
                raw_state,
                low_state,
                geometry,
                boundaries,
                local_damage_limit=high_reference_limit,
                exact_replay=replay,
                arm=policy,
                maximum_rounds=maximum_repair_rounds,
                move_limit=move_limit,
                candidate_limit=candidate_limit,
            )
            restored = (
                repair.final_metrics.membership_crossings
                <= allowed_crossings
            )
            high_audits.append(PolicyRepairAudit(
                attempt_index=len(high_audits) + 1,
                allowed_crossings=allowed_crossings,
                accepted=restored,
                result=repair,
            ))
            if not restored:
                continue
            candidate_state = np.asarray(repair.final_state).copy()
            candidate_metrics = repair.final_metrics
            candidate_damage = local(candidate_state)
        if (
            candidate_metrics.membership_crossings
            > low_metrics.membership_crossings
            or (
                initial_high_metrics.safe
                and not np.array_equal(candidate_state, arm3_high)
            )
            or (
                not initial_high_metrics.safe
                and not np.array_equal(candidate_state, arm3_high)
                and candidate_metrics.membership_crossings
                >= initial_high_metrics.membership_crossings
            )
            or candidate_damage > high_reference_limit + 1e-12
            or not physical_subset(low_state, candidate_state)
        ):
            raise RuntimeError("strict high checkpoint acceptance invariant failed")
        selected = (
            path_index,
            candidate_state,
            candidate_metrics,
            candidate_damage,
            completion_moves,
            repair,
        )
        break

    pair_fallback = selected is None
    if pair_fallback:
        # A repaired low that cannot be extended within the high endpoint's
        # own Arm3-relative guard is not a publishable nested pair. Revert both
        # endpoints to their D1-blind Arm3 incumbents so the fixed grid remains
        # complete without weakening either endpoint's local control.
        low_state = arm3_low.copy()
        low_metrics = initial_low_metrics
        low_damage = arm3_low_damage
        low_repair = None
        low_reason = "pair_fallback_arm3_low_identity"
        high_state = arm3_high.copy()
        high_metrics = (
            low_metrics if np.array_equal(high_state, low_state)
            else replay(high_state)
        )
        high_damage = arm3_high_damage
        high_reason = "pair_fallback_high_guardrail_infeasible"
        high_audits = []
        high_checkpoints = [NestedPolicyCheckpoint(
            checkpoint_index=0,
            state=low_state,
            metrics=low_metrics,
            local_damage=low_damage,
        )]
        fallback_moves = canonical_add_only_moves(low_state, high_state)
        if fallback_moves:
            high_checkpoints.append(NestedPolicyCheckpoint(
                checkpoint_index=1,
                state=high_state,
                metrics=high_metrics,
                local_damage=high_damage,
                completion_moves=fallback_moves,
            ))
    else:
        (
            path_index,
            high_state,
            high_metrics,
            high_damage,
            completion_moves,
            selected_repair,
        ) = selected
        if path_index != 0:
            high_checkpoints.append(NestedPolicyCheckpoint(
                checkpoint_index=1,
                state=high_state,
                metrics=high_metrics,
                local_damage=high_damage,
                completion_moves=completion_moves,
                repair=selected_repair,
            ))
        if path_index == -1:
            high_reason = "arm3_high_same_rate_incumbent_identity"
        elif path_index == 0:
            high_reason = "low_retained_as_high_strict_guard_feasible"
        elif path_index == len(path_checkpoints) - 1:
            suffix = (
                "high_cap" if path_checkpoints[path_index].group_pages == high_cap
                else str(endpoint_path.stop_reason)
            )
            high_reason = (
                f"completion_endpoint_strict_guard_repaired_{suffix}"
                if selected_repair is not None
                else f"completion_endpoint_strict_guard_nonworsening_{suffix}"
            )
        else:
            high_reason = (
                "highest_page_strict_guard_repaired_checkpoint"
                if selected_repair is not None
                else "highest_page_strict_guard_feasible_checkpoint"
            )


    return NestedPolicyPairResult(
        arm=policy,
        expert_ids=ids,
        common_core=core,
        arm3_low_state=arm3_low,
        arm3_high_state=arm3_high,
        low_state=low_state,
        high_state=high_state,
        low_metrics=low_metrics,
        high_metrics=high_metrics,
        arm3_high_metrics=initial_high_metrics,
        arm3_low_damage=arm3_low_damage,
        arm3_high_damage=arm3_high_damage,
        low_local_damage=low_damage,
        high_local_damage=high_damage,
        low_local_damage_limit=low_limit,
        high_reference_damage_limit=high_reference_limit,
        eta=eta_value,
        j_q2=q2,
        low_cap_pages=low_cap,
        high_cap_pages=high_cap,
        refresh_after_accepted_pages=refresh,
        low_stop_reason=low_reason,
        high_stop_reason=high_reason,
        core_to_low_moves=canonical_add_only_moves(core, low_state),
        core_to_high_moves=canonical_add_only_moves(core, high_state),
        low_repair=low_repair,
        high_checkpoints=tuple(high_checkpoints),
        high_repair_audits=tuple(high_audits),
        pair_fallback_high_guardrail_infeasible=pair_fallback,
        high_guardrail_qualified_checkpoints=len(qualified_indices),
        high_guardrail_repair_attempts=repair_attempts,
    )
