"""D1-only screening and exact repair search for nested physical states.

The signed-margin screen in this module is only a bounded shortlist.  Exact
all-256 next-router replay is required before a state can be accepted, and a
safe incumbent is immutable.  No downstream router or terminal outcome is an
input to these APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .d1_nested_allocation import (
    PhysicalPageMove,
    apply_page_moves,
    legal_add_moves,
    legal_remove_moves,
    physical_subset,
    state_page_count,
    validate_states,
)
from .d1_nested_geometry import TokenStateGeometry


EXPERT_COUNT = 256
TOP_K = 8
SCREEN_SELECTED_RANKS = (6, 7, 8)
SCREEN_OUTSIDER_RANKS = tuple(range(9, 17))


__all__ = [
    "EXPERT_COUNT",
    "TOP_K",
    "ExactD1Metrics",
    "ScreenedD1Boundaries",
    "PredictedMarginMetrics",
    "MatchedSwapCandidate",
    "ExactRepairStep",
    "ExactRepairResult",
    "stable_top8",
    "exact_all_expert_d1_metrics",
    "build_screened_d1_boundaries",
    "predicted_signed_margin_metrics",
    "shortlist_matched_page_swaps",
    "iterative_exact_repair",
]


def _freeze(value: np.ndarray, dtype: np.dtype | type | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _finite_logits(value: Sequence[float] | np.ndarray, *, exact_256: bool) -> np.ndarray:
    logits = np.asarray(value, np.float64)
    expected = (EXPERT_COUNT,) if exact_256 else None
    if logits.ndim != 1 or logits.size < TOP_K or (expected is not None and logits.shape != expected):
        suffix = " exactly 256" if exact_256 else " at least eight"
        raise ValueError(f"router logits must contain{suffix} finite expert values")
    if np.any(~np.isfinite(logits)):
        raise ValueError("router logits must be finite")
    return logits.copy()


def stable_top8(logits: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return stable descending top-8 expert IDs for one router vector."""

    values = _finite_logits(logits, exact_256=False)
    return np.argsort(-values, kind="stable")[:TOP_K].astype(np.int64, copy=True)


def _executed_top8(value: Sequence[int] | np.ndarray, field: str) -> np.ndarray:
    ids = np.asarray(value)
    if (
        ids.shape != (TOP_K,)
        or ids.dtype.kind not in "iu"
        or len(set(map(int, ids))) != TOP_K
        or np.any((ids < 0) | (ids >= EXPERT_COUNT))
    ):
        raise ValueError(f"{field} must contain eight unique executed IDs in [0,255]")
    return np.asarray(ids, np.int64).copy()


@dataclass(frozen=True)
class ExactD1Metrics:
    baseline_top8: np.ndarray
    candidate_top8: np.ndarray
    lost_target_experts: np.ndarray
    entering_outsiders: np.ndarray
    membership_crossings: int
    violation_depth: float
    routing_mass_lost: float
    target_set_margin: float

    def __post_init__(self) -> None:
        baseline = np.asarray(self.baseline_top8, np.int64).reshape(-1)
        candidate = np.asarray(self.candidate_top8, np.int64).reshape(-1)
        lost = np.asarray(self.lost_target_experts, np.int64).reshape(-1)
        entered = np.asarray(self.entering_outsiders, np.int64).reshape(-1)
        crossings = int(self.membership_crossings)
        depth = float(self.violation_depth)
        mass = float(self.routing_mass_lost)
        margin = float(self.target_set_margin)
        if (
            baseline.shape != (TOP_K,)
            or candidate.shape != (TOP_K,)
            or len(set(baseline.tolist())) != TOP_K
            or len(set(candidate.tolist())) != TOP_K
            or np.any((baseline < 0) | (baseline >= EXPERT_COUNT))
            or np.any((candidate < 0) | (candidate >= EXPERT_COUNT))
        ):
            raise ValueError("exact D1 top-8 labels are invalid")
        expected_lost = set(baseline.tolist()).difference(candidate.tolist())
        expected_entered = set(candidate.tolist()).difference(baseline.tolist())
        if (
            crossings < 0
            or crossings > TOP_K
            or lost.shape != (crossings,)
            or entered.shape != (crossings,)
            or set(lost.tolist()) != expected_lost
            or set(entered.tolist()) != expected_entered
        ):
            raise ValueError("exact D1 crossing labels are inconsistent")
        if (
            not np.isfinite(depth)
            or depth < 0.0
            or not np.isfinite(mass)
            or mass < 0.0
            or mass > 1.0 + 1e-12
            or not np.isfinite(margin)
        ):
            raise ValueError("exact D1 severity metrics are invalid")
        object.__setattr__(self, "baseline_top8", _freeze(baseline))
        object.__setattr__(self, "candidate_top8", _freeze(candidate))
        object.__setattr__(self, "lost_target_experts", _freeze(lost))
        object.__setattr__(self, "entering_outsiders", _freeze(entered))
        object.__setattr__(self, "membership_crossings", crossings)
        object.__setattr__(self, "violation_depth", depth)
        object.__setattr__(self, "routing_mass_lost", min(1.0, mass))
        object.__setattr__(self, "target_set_margin", margin)

    @property
    def d1_crossings(self) -> int:
        return self.membership_crossings

    @property
    def summed_violation_depth(self) -> float:
        return self.violation_depth

    @property
    def labeled_target_set_margin(self) -> float:
        return self.target_set_margin

    @property
    def safe(self) -> bool:
        return self.membership_crossings == 0


def _subset_order(
    expert_ids: np.ndarray,
    logits: np.ndarray,
    *,
    descending: bool,
) -> np.ndarray:
    ids = np.asarray(expert_ids, np.int64).reshape(-1)
    primary = -logits[ids] if descending else logits[ids]
    return ids[np.lexsort((ids, primary))]


def exact_all_expert_d1_metrics(
    baseline_logits: Sequence[float] | np.ndarray,
    candidate_logits: Sequence[float] | np.ndarray,
    *,
    baseline_top8: Sequence[int] | np.ndarray,
    candidate_top8: Sequence[int] | np.ndarray,
) -> ExactD1Metrics:
    """Measure exact labeled top-8 damage using explicit executed route IDs.

    The caller supplies IDs produced by the authoritative execution stack;
    logits are never re-ranked with incompatible NumPy tie semantics.

    Entering outsiders are paired strongest-first with lost target experts
    weakest-first under candidate logits.  The summed positive logit gaps form
    violation depth.  Routing mass is the baseline full-softmax probability of
    target experts that the candidate loses.
    """

    baseline = _finite_logits(baseline_logits, exact_256=True)
    candidate = _finite_logits(candidate_logits, exact_256=True)
    target = _executed_top8(baseline_top8, "baseline_top8")
    observed = _executed_top8(candidate_top8, "candidate_top8")
    target_set = set(target.tolist())
    observed_set = set(observed.tolist())
    lost_raw = np.asarray(sorted(target_set.difference(observed_set)), np.int64)
    entered_raw = np.asarray(sorted(observed_set.difference(target_set)), np.int64)
    lost = _subset_order(lost_raw, candidate, descending=False)
    entered = _subset_order(entered_raw, candidate, descending=True)
    crossings = int(lost.size)
    if crossings:
        pair_depth = np.maximum(candidate[entered] - candidate[lost], 0.0)
        violation = float(pair_depth.sum())
    else:
        violation = 0.0
    shifted = baseline - float(np.max(baseline))
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    mass = float(probabilities[lost].sum()) if crossings else 0.0
    outsider_mask = np.ones(EXPERT_COUNT, dtype=bool)
    outsider_mask[target] = False
    target_margin = float(np.min(candidate[target]) - np.max(candidate[outsider_mask]))
    return ExactD1Metrics(
        baseline_top8=target,
        candidate_top8=observed,
        lost_target_experts=lost,
        entering_outsiders=entered,
        membership_crossings=crossings,
        violation_depth=violation,
        routing_mass_lost=mass,
        target_set_margin=target_margin,
    )


@dataclass(frozen=True)
class ScreenedD1Boundaries:
    selected_expert_ids: np.ndarray
    outsider_expert_ids: np.ndarray
    baseline_margins: np.ndarray
    margin_sensitivities: np.ndarray

    def __post_init__(self) -> None:
        selected = np.asarray(self.selected_expert_ids, np.int64).reshape(-1)
        outsiders = np.asarray(self.outsider_expert_ids, np.int64).reshape(-1)
        margins = np.asarray(self.baseline_margins, np.float64).reshape(-1)
        sensitivities = np.asarray(self.margin_sensitivities, np.float64)
        count = len(SCREEN_SELECTED_RANKS) * len(SCREEN_OUTSIDER_RANKS)
        if (
            selected.shape != (count,)
            or outsiders.shape != (count,)
            or margins.shape != (count,)
            or sensitivities.ndim != 2
            or sensitivities.shape[0] != count
            or sensitivities.shape[1] < 1
            or np.any((selected < 0) | (selected >= EXPERT_COUNT))
            or np.any((outsiders < 0) | (outsiders >= EXPERT_COUNT))
            or np.any(selected == outsiders)
            or np.any(~np.isfinite(margins))
            or np.any(margins < 0.0)
            or np.any(~np.isfinite(sensitivities))
        ):
            raise ValueError("screened D1 boundary arrays are invalid")
        object.__setattr__(self, "selected_expert_ids", _freeze(selected))
        object.__setattr__(self, "outsider_expert_ids", _freeze(outsiders))
        object.__setattr__(self, "baseline_margins", _freeze(margins))
        object.__setattr__(self, "margin_sensitivities", _freeze(sensitivities))

    @property
    def boundaries(self) -> int:
        return int(self.baseline_margins.size)

    @property
    def output_width(self) -> int:
        return int(self.margin_sensitivities.shape[1])


def build_screened_d1_boundaries(
    baseline_logits: Sequence[float] | np.ndarray,
    candidate_logit_sensitivities: np.ndarray,
    *,
    selected_expert_ids: Sequence[int] | np.ndarray,
    outsider_expert_ids: Sequence[int] | np.ndarray,
) -> ScreenedD1Boundaries:
    """Build the explicit executed ranks 6--8 versus eight-outsider screen."""

    logits = _finite_logits(baseline_logits, exact_256=True)
    gradients = np.asarray(candidate_logit_sensitivities, np.float64)
    if (
        gradients.ndim != 2
        or gradients.shape[0] != EXPERT_COUNT
        or gradients.shape[1] < 1
        or np.any(~np.isfinite(gradients))
    ):
        raise ValueError(
            "candidate-logit sensitivities must be finite [256,output]"
        )
    selected_unique = np.asarray(selected_expert_ids)
    outsider_unique = np.asarray(outsider_expert_ids)
    if (
        selected_unique.shape != (len(SCREEN_SELECTED_RANKS),)
        or outsider_unique.shape != (len(SCREEN_OUTSIDER_RANKS),)
        or selected_unique.dtype.kind not in "iu"
        or outsider_unique.dtype.kind not in "iu"
        or len(set(map(int, selected_unique))) != selected_unique.size
        or len(set(map(int, outsider_unique))) != outsider_unique.size
        or bool(set(map(int, selected_unique)) & set(map(int, outsider_unique)))
        or np.any((selected_unique < 0) | (selected_unique >= EXPERT_COUNT))
        or np.any((outsider_unique < 0) | (outsider_unique >= EXPERT_COUNT))
    ):
        raise ValueError("screened executed expert ID pools are invalid")
    selected_unique = np.asarray(selected_unique, np.int64)
    outsider_unique = np.asarray(outsider_unique, np.int64)
    selected = np.repeat(selected_unique, outsider_unique.size)
    outsiders = np.tile(outsider_unique, selected_unique.size)
    return ScreenedD1Boundaries(
        selected_expert_ids=selected,
        outsider_expert_ids=outsiders,
        baseline_margins=logits[selected] - logits[outsiders],
        margin_sensitivities=gradients[selected] - gradients[outsiders],
    )


@dataclass(frozen=True)
class PredictedMarginMetrics:
    signed_effects: np.ndarray
    predicted_margins: np.ndarray
    predicted_membership_crossings: int
    violated_boundaries: int
    violation_depth: float
    minimum_margin: float

    def __post_init__(self) -> None:
        effects = np.asarray(self.signed_effects, np.float64).reshape(-1)
        margins = np.asarray(self.predicted_margins, np.float64).reshape(-1)
        crossings = int(self.predicted_membership_crossings)
        violations = int(self.violated_boundaries)
        depth = float(self.violation_depth)
        minimum = float(self.minimum_margin)
        if (
            effects.size == 0
            or margins.shape != effects.shape
            or np.any(~np.isfinite(effects))
            or np.any(~np.isfinite(margins))
            or crossings < 0
            or violations < 0
            or violations > margins.size
            or not np.isfinite(depth)
            or depth < 0.0
            or not np.isfinite(minimum)
        ):
            raise ValueError("predicted signed-margin metrics are invalid")
        object.__setattr__(self, "signed_effects", _freeze(effects))
        object.__setattr__(self, "predicted_margins", _freeze(margins))
        object.__setattr__(self, "predicted_membership_crossings", crossings)
        object.__setattr__(self, "violated_boundaries", violations)
        object.__setattr__(self, "violation_depth", depth)
        object.__setattr__(self, "minimum_margin", minimum)


def _predicted_from_effects(
    boundaries: ScreenedD1Boundaries,
    signed_effects: np.ndarray,
) -> PredictedMarginMetrics:
    effects = np.asarray(signed_effects, np.float64).reshape(-1)
    if effects.shape != (boundaries.boundaries,) or np.any(~np.isfinite(effects)):
        raise ValueError("signed effects must contain one finite value per boundary")
    margins = np.asarray(boundaries.baseline_margins) + effects
    violated = margins < 0.0
    selected_crossings = 0
    for expert in np.unique(boundaries.selected_expert_ids):
        selected_crossings += int(np.any(
            violated[np.asarray(boundaries.selected_expert_ids) == expert]
        ))
    return PredictedMarginMetrics(
        signed_effects=effects,
        predicted_margins=margins,
        predicted_membership_crossings=selected_crossings,
        violated_boundaries=int(violated.sum()),
        violation_depth=float(np.maximum(-margins, 0.0).sum()),
        minimum_margin=float(np.min(margins)),
    )


def predicted_signed_margin_metrics(
    boundaries: ScreenedD1Boundaries,
    token_output_delta: Sequence[float] | np.ndarray,
) -> PredictedMarginMetrics:
    """Project one approximate-minus-Q4 token output delta onto the screen."""

    if not isinstance(boundaries, ScreenedD1Boundaries):
        raise TypeError("boundaries must be ScreenedD1Boundaries")
    delta = np.asarray(token_output_delta, np.float64).reshape(-1)
    if delta.shape != (boundaries.output_width,) or np.any(~np.isfinite(delta)):
        raise ValueError("token output delta must be finite and match output width")
    return _predicted_from_effects(boundaries, boundaries.margin_sensitivities @ delta)


def _predicted_key(metrics: PredictedMarginMetrics) -> tuple[float | int, ...]:
    return (
        metrics.predicted_membership_crossings,
        metrics.violated_boundaries,
        metrics.violation_depth,
        -metrics.minimum_margin,
    )


def _rank_move_indices(
    moves: tuple[PhysicalPageMove, ...],
    move_effects: np.ndarray,
    incumbent_effects: np.ndarray,
    boundaries: ScreenedD1Boundaries,
    limit: int,
) -> tuple[int, ...]:
    return tuple(sorted(
        range(len(moves)),
        key=lambda index: (
            *_predicted_key(_predicted_from_effects(
                boundaries, incumbent_effects + move_effects[index],
            )),
            moves[index].sort_key,
        ),
    )[:limit])


@dataclass(frozen=True)
class MatchedSwapCandidate:
    states: np.ndarray
    removal: PhysicalPageMove
    addition: PhysicalPageMove
    predicted: PredictedMarginMetrics
    local_damage: float

    def __post_init__(self) -> None:
        states = validate_states(self.states)
        damage = float(self.local_damage)
        if (
            not isinstance(self.removal, PhysicalPageMove)
            or self.removal.direction != "remove"
            or not isinstance(self.addition, PhysicalPageMove)
            or self.addition.direction != "add"
            or (self.removal.expert, self.removal.unit)
            == (self.addition.expert, self.addition.unit)
            or not isinstance(self.predicted, PredictedMarginMetrics)
            or not np.isfinite(damage)
            or damage < 0.0
        ):
            raise ValueError("matched swap candidate is invalid")
        object.__setattr__(self, "states", _freeze(states, np.int64))
        object.__setattr__(self, "local_damage", damage)

    @property
    def sort_key(self) -> tuple[object, ...]:
        return (
            *_predicted_key(self.predicted),
            self.local_damage,
            self.removal.sort_key,
            self.addition.sort_key,
        )


def shortlist_matched_page_swaps(
    incumbent_states: np.ndarray,
    frozen_core: np.ndarray,
    geometry: TokenStateGeometry,
    boundaries: ScreenedD1Boundaries,
    *,
    local_damage_limit: float,
    move_limit: int = 32,
    candidate_limit: int = 8,
    local_tolerance: float = 1e-12,
) -> tuple[MatchedSwapCandidate, ...]:
    """Return a bounded signed-margin shortlist of exact-local legal swaps.

    This function is not a route certificate.  It ranks at most 32 removals
    and 32 additions, combines their distinct-unit pairs, and returns at most
    eight states.  Callers must exact-replay all 256 router logits.
    """

    incumbent = validate_states(incumbent_states)
    core = validate_states(frozen_core)
    if not physical_subset(core, incumbent):
        raise ValueError("frozen core must be a physical subset of incumbent")
    if not isinstance(boundaries, ScreenedD1Boundaries):
        raise TypeError("boundaries must be ScreenedD1Boundaries")
    if int(getattr(geometry, "output_width")) != boundaries.output_width:
        raise ValueError("token geometry and D1 sensitivity widths differ")
    move_cap = int(move_limit)
    result_cap = int(candidate_limit)
    guard = float(local_damage_limit)
    tolerance = float(local_tolerance)
    if (
        move_cap != move_limit
        or not 1 <= move_cap <= 32
        or result_cap != candidate_limit
        or not 1 <= result_cap <= 8
        or not np.isfinite(guard)
        or guard < 0.0
        or not np.isfinite(tolerance)
        or tolerance < 0.0
    ):
        raise ValueError("swap shortlist limits or local guard are invalid")
    removals = legal_remove_moves(incumbent, lower_states=core)
    additions = legal_add_moves(incumbent)
    if not removals or not additions:
        return ()
    sensitivities = np.asarray(boundaries.margin_sensitivities)
    vectorized_local = all(hasattr(geometry, name) for name in (
        "move_output_deltas", "local_damages_from_output_deltas",
    ))
    # Project all legal moves in bounded chunks for ranking. Full-width move
    # deltas are synthesized only for the <=32+32 retained moves below; keeping
    # every legal [move,hidden] row would add hundreds of MiB at model width.
    removal_effects = np.asarray(
        geometry.signed_move_effects(incumbent, removals, sensitivities),
        np.float64,
    )
    addition_effects = np.asarray(
        geometry.signed_move_effects(incumbent, additions, sensitivities),
        np.float64,
    )
    if (
        removal_effects.shape != (len(removals), boundaries.boundaries)
        or addition_effects.shape != (len(additions), boundaries.boundaries)
        or np.any(~np.isfinite(removal_effects))
        or np.any(~np.isfinite(addition_effects))
    ):
        raise ValueError("token geometry returned invalid signed move effects")
    incumbent_delta = np.asarray(geometry.output_delta(incumbent), np.float64).reshape(-1)
    if incumbent_delta.shape != (boundaries.output_width,) or np.any(~np.isfinite(incumbent_delta)):
        raise ValueError("token geometry returned an invalid incumbent delta")
    incumbent_effects = sensitivities @ incumbent_delta
    removal_indices = _rank_move_indices(
        removals, removal_effects, incumbent_effects, boundaries,
        min(move_cap, len(removals)),
    )
    addition_indices = _rank_move_indices(
        additions, addition_effects, incumbent_effects, boundaries,
        min(move_cap, len(additions)),
    )
    incumbent_pages = state_page_count(incumbent)
    pair_indices = tuple(
        (remove_index, add_index)
        for remove_index in removal_indices
        for add_index in addition_indices
        if (
            removals[remove_index].expert,
            removals[remove_index].unit,
        ) != (
            additions[add_index].expert,
            additions[add_index].unit,
        )
    )
    vector_local: np.ndarray | None = None
    if vectorized_local and pair_indices:
        selected_removal_deltas = dict(zip(
            removal_indices,
            np.asarray(geometry.move_output_deltas(
                incumbent, tuple(removals[index] for index in removal_indices),
            ), np.float64),
            strict=True,
        ))
        selected_addition_deltas = dict(zip(
            addition_indices,
            np.asarray(geometry.move_output_deltas(
                incumbent, tuple(additions[index] for index in addition_indices),
            ), np.float64),
            strict=True,
        ))
        pair_deltas = np.stack([
            incumbent_delta
            + selected_removal_deltas[remove_index]
            + selected_addition_deltas[add_index]
            for remove_index, add_index in pair_indices
        ])
        vector_local = np.asarray(
            geometry.local_damages_from_output_deltas(pair_deltas), np.float64,
        ).reshape(-1)
        if (
            vector_local.shape != (len(pair_indices),)
            or np.any(~np.isfinite(vector_local))
            or np.any(vector_local < 0.0)
        ):
            raise ValueError("token geometry returned invalid vector local damage")

    # Sort lightweight specifications first. On the optimized path complete
    # 8x512 state tables are materialized and scalar-parity checked only for
    # finalists, rather than for every member of the Cartesian product.
    specifications: list[
        tuple[tuple[float | int, ...], int, int, PredictedMarginMetrics, float]
    ] = []
    for pair_offset, (remove_index, add_index) in enumerate(pair_indices):
        removal = removals[remove_index]
        addition = additions[add_index]
        combined = (
            incumbent_effects
            + removal_effects[remove_index]
            + addition_effects[add_index]
        )
        predicted = _predicted_from_effects(boundaries, combined)
        local = float("nan") if vector_local is None else float(
            vector_local[pair_offset]
        )
        if vector_local is not None and (not np.isfinite(local) or local < 0.0):
            raise ValueError("token geometry returned invalid vector local damage")
        # Local damage cannot affect ordering across distinct predicted D1
        # prefixes. Defer the exact scalar guard and local tie-break until a
        # complete equal-prefix group can still enter the top-k. This retains
        # reference semantics even in an adversarial floating-point tie.
        key = _predicted_key(predicted)
        specifications.append((
            key, remove_index, add_index, predicted, local,
        ))
    specifications.sort(key=lambda item: (
        item[0],
        removals[item[1]].sort_key,
        additions[item[2]].sort_key,
    ))

    retained: list[MatchedSwapCandidate] = []
    offset = 0
    while offset < len(specifications) and len(retained) < result_cap:
        prefix = specifications[offset][0]
        end = offset + 1
        while end < len(specifications) and specifications[end][0] == prefix:
            end += 1
        equal_prefix: list[MatchedSwapCandidate] = []
        for _, remove_index, add_index, predicted, vector_score in specifications[
            offset:end
        ]:
            removal = removals[remove_index]
            addition = additions[add_index]
            candidate_states = apply_page_moves(incumbent, (removal, addition))
            if (
                state_page_count(candidate_states) != incumbent_pages
                or not physical_subset(core, candidate_states)
            ):
                raise RuntimeError(
                    "matched swap violated core or page-count invariants"
                )
            scalar_score = float(geometry.local_damage(candidate_states))
            if not np.isfinite(scalar_score) or scalar_score < 0.0:
                raise ValueError("token geometry returned invalid scalar local damage")
            if vector_local is not None and not np.isclose(
                scalar_score, vector_score, rtol=2e-12, atol=2e-12,
            ):
                raise RuntimeError(
                    "vectorized matched-swap qmetric lost scalar parity"
                )
            if scalar_score > guard + tolerance:
                continue
            equal_prefix.append(MatchedSwapCandidate(
                states=candidate_states,
                removal=removal,
                addition=addition,
                predicted=predicted,
                local_damage=scalar_score,
            ))
        equal_prefix.sort(key=lambda candidate: candidate.sort_key)
        retained.extend(equal_prefix[: result_cap - len(retained)])
        offset = end
    return tuple(retained)


ExactReplay = Callable[[np.ndarray], ExactD1Metrics]


@dataclass(frozen=True)
class ExactRepairStep:
    round_index: int
    state: np.ndarray
    removal: PhysicalPageMove
    addition: PhysicalPageMove
    metrics: ExactD1Metrics
    local_damage: float

    def __post_init__(self) -> None:
        index = int(self.round_index)
        state = validate_states(self.state)
        damage = float(self.local_damage)
        if (
            index < 1
            or not isinstance(self.metrics, ExactD1Metrics)
            or not np.isfinite(damage)
            or damage < 0.0
        ):
            raise ValueError("exact repair step is invalid")
        object.__setattr__(self, "round_index", index)
        object.__setattr__(self, "state", _freeze(state, np.int64))
        object.__setattr__(self, "local_damage", damage)


@dataclass(frozen=True)
class ExactRepairResult:
    arm: str
    initial_state: np.ndarray
    final_state: np.ndarray
    initial_metrics: ExactD1Metrics
    final_metrics: ExactD1Metrics
    steps: tuple[ExactRepairStep, ...]
    stop_reason: str

    def __post_init__(self) -> None:
        arm = str(self.arm)
        initial = validate_states(self.initial_state)
        final = validate_states(self.final_state)
        steps = tuple(self.steps)
        reason = str(self.stop_reason)
        if (
            arm not in {"arm4", "arm5"}
            or not isinstance(self.initial_metrics, ExactD1Metrics)
            or not isinstance(self.final_metrics, ExactD1Metrics)
            or any(not isinstance(step, ExactRepairStep) for step in steps)
            or not reason
        ):
            raise ValueError("exact repair result is invalid")
        object.__setattr__(self, "arm", arm)
        object.__setattr__(self, "initial_state", _freeze(initial, np.int64))
        object.__setattr__(self, "final_state", _freeze(final, np.int64))
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "stop_reason", reason)


def _exact_metrics(exact_replay: ExactReplay, states: np.ndarray) -> ExactD1Metrics:
    frozen = _freeze(validate_states(states), np.int64)
    result = exact_replay(frozen)
    if not isinstance(result, ExactD1Metrics):
        raise TypeError("exact_replay must return ExactD1Metrics")
    return result


def iterative_exact_repair(
    incumbent_states: np.ndarray,
    frozen_core: np.ndarray,
    geometry: TokenStateGeometry,
    boundaries: ScreenedD1Boundaries,
    *,
    local_damage_limit: float,
    exact_replay: ExactReplay,
    arm: str = "arm4",
    maximum_rounds: int = 8,
    move_limit: int = 32,
    candidate_limit: int = 8,
) -> ExactRepairResult:
    """Iteratively accept only strict exact all-256 crossing reductions."""

    policy = str(arm)
    rounds = int(maximum_rounds)
    if policy not in {"arm4", "arm5"}:
        raise ValueError("arm must be 'arm4' or 'arm5'")
    if rounds != maximum_rounds or not 1 <= rounds <= TOP_K:
        raise ValueError("maximum_rounds must be an integer in [1,8]")
    initial = validate_states(incumbent_states)
    core = validate_states(frozen_core)
    if not physical_subset(core, initial):
        raise ValueError("frozen core must be a subset of repair incumbent")
    pages = state_page_count(initial)
    current = initial.copy()
    current_metrics = _exact_metrics(exact_replay, current)
    initial_metrics = current_metrics
    target_labels = np.asarray(current_metrics.baseline_top8).copy()
    if current_metrics.safe:
        return ExactRepairResult(
            policy, initial, initial, initial_metrics, initial_metrics, (),
            "incumbent_d1_safe",
        )
    steps: list[ExactRepairStep] = []
    stop_reason = "maximum_rounds"
    for round_index in range(1, rounds + 1):
        candidates = shortlist_matched_page_swaps(
            current,
            core,
            geometry,
            boundaries,
            local_damage_limit=local_damage_limit,
            move_limit=move_limit,
            candidate_limit=candidate_limit,
        )
        if not candidates:
            stop_reason = "no_shortlist_candidate"
            break
        eligible: list[tuple[int, MatchedSwapCandidate, ExactD1Metrics]] = []
        for index, candidate in enumerate(candidates):
            metrics = _exact_metrics(exact_replay, candidate.states)
            if not np.array_equal(metrics.baseline_top8, target_labels):
                raise ValueError("exact replay changed the labeled baseline target set")
            if metrics.membership_crossings < current_metrics.membership_crossings:
                eligible.append((index, candidate, metrics))
        if not eligible:
            stop_reason = "no_strict_exact_crossing_repair"
            break

        def selection_key(
            item: tuple[int, MatchedSwapCandidate, ExactD1Metrics],
        ) -> tuple[object, ...]:
            index, candidate, metrics = item
            if policy == "arm5" and not metrics.safe:
                severity = (metrics.violation_depth, metrics.routing_mass_lost)
            else:
                severity = (0.0, 0.0)
            return (
                metrics.membership_crossings,
                *severity,
                candidate.local_damage,
                candidate.removal.sort_key,
                candidate.addition.sort_key,
                index,
            )

        _, selected, selected_metrics = min(eligible, key=selection_key)
        if selected_metrics.membership_crossings >= current_metrics.membership_crossings:
            raise RuntimeError("exact repair acceptance was not a strict reduction")
        if (
            state_page_count(selected.states) != pages
            or not physical_subset(core, selected.states)
        ):
            raise RuntimeError("accepted exact repair violated physical invariants")
        current = np.asarray(selected.states).copy()
        current_metrics = selected_metrics
        steps.append(ExactRepairStep(
            round_index=round_index,
            state=current,
            removal=selected.removal,
            addition=selected.addition,
            metrics=current_metrics,
            local_damage=selected.local_damage,
        ))
        if current_metrics.safe:
            stop_reason = "d1_safe"
            break
    return ExactRepairResult(
        arm=policy,
        initial_state=initial,
        final_state=current,
        initial_metrics=initial_metrics,
        final_metrics=current_metrics,
        steps=tuple(steps),
        stop_reason=stop_reason,
    )
