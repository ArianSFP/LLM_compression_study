"""Exact D1 route-boundary objectives over PR #13 frontier options.

D1 is the same token's next-layer router, ``(t, l) -> (t, l + 1)``.
This module is deliberately oracle-only: exact Q4 router margins, exact
token-dependent option deltas, and exact D1 sensitivities are accepted as
inputs.  None of those quantities is a deployable runtime input.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Mapping, Sequence

import numpy as np

from .average_rate_allocator import RateOption
from .split_interaction_field import SplitProjectionResponses, split_state_output


__all__ = [
    "D1BoundaryProblem",
    "D1AllocationTrace",
    "D1OraclePolicyResult",
    "adaptive_outsider_ranks",
    "additive_local_damage_limit",
    "build_d1_boundary_problem",
    "token_option_deltas",
    "option_boundary_effects",
    "directional_route_loss",
    "directional_route_hinge_loss",
    "incumbent_repair_choice",
    "routing_mass_severity",
    "functional_swap_severity",
    "exact_candidate_logit_vjps",
    "d1_group_option_allocate",
    "run_d1_oracle_policies",
]


def _vector_parameter(
    value: float | Sequence[float] | np.ndarray,
    size: int,
    name: str,
    *,
    positive: bool,
) -> np.ndarray:
    result = np.asarray(value, np.float64)
    if result.ndim == 0:
        result = np.full(int(size), float(result), np.float64)
    else:
        result = result.reshape(-1)
    if result.shape != (int(size),) or np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be finite scalar or one value per boundary")
    if positive and np.any(result <= 0.0):
        raise ValueError(f"{name} must be positive")
    if not positive and np.any(result < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return result


@dataclass(frozen=True)
class D1BoundaryProblem:
    """Exact-Q4 selected/outsider boundary labels for one token and layer."""

    selected_expert_ids: np.ndarray
    outsider_expert_ids: np.ndarray
    q4_margins: np.ndarray
    margin_sensitivities: np.ndarray
    severity: np.ndarray
    temperature: np.ndarray
    safety_margin: np.ndarray

    def __post_init__(self) -> None:
        selected = np.asarray(self.selected_expert_ids, np.int64).reshape(-1)
        outsiders = np.asarray(self.outsider_expert_ids, np.int64).reshape(-1)
        margins = np.asarray(self.q4_margins, np.float64).reshape(-1)
        gradients = np.asarray(self.margin_sensitivities, np.float64)
        severity = np.asarray(self.severity, np.float64).reshape(-1)
        temperature = np.asarray(self.temperature, np.float64).reshape(-1)
        safety = np.asarray(self.safety_margin, np.float64).reshape(-1)
        boundaries = margins.size
        if boundaries == 0:
            raise ValueError("D1 boundary problem must be nonempty")
        if selected.shape != (boundaries,) or outsiders.shape != (boundaries,):
            raise ValueError("boundary expert IDs must align with margins")
        if gradients.ndim != 2 or gradients.shape[0] != boundaries:
            raise ValueError("margin sensitivities must be [boundaries,hidden]")
        if any(value.shape != (boundaries,) for value in (severity, temperature, safety)):
            raise ValueError("D1 loss vectors must align with margins")
        if not all(np.all(np.isfinite(value)) for value in (
            margins, gradients, severity, temperature, safety,
        )):
            raise ValueError("D1 boundary values must be finite")
        # BF16 router logits can tie exactly at the top-k boundary. The
        # baseline top-k operation still supplies a deterministic membership
        # label, so zero is a valid (maximally fragile) boundary.
        if np.any(margins < 0.0):
            raise ValueError("exact-Q4 selected/outsider margins must be nonnegative")
        if np.any(severity < 0.0) or np.any(temperature <= 0.0) or np.any(safety < 0.0):
            raise ValueError("D1 loss scales are invalid")

    @property
    def boundaries(self) -> int:
        return int(np.asarray(self.q4_margins).size)

    @property
    def hidden_size(self) -> int:
        return int(np.asarray(self.margin_sensitivities).shape[1])


@dataclass(frozen=True)
class D1AllocationTrace:
    """Deterministic coordinate/pair solution of the directional D1 oracle."""

    option_indices: np.ndarray
    pages: int
    route_loss: float
    local_damage: float
    predicted_margins: np.ndarray
    predicted_crossings: int
    coordinate_sweeps: int
    pair_passes: int

    def __post_init__(self) -> None:
        selected = np.asarray(self.option_indices, np.int64).reshape(-1)
        margins = np.asarray(self.predicted_margins, np.float64).reshape(-1)
        if selected.size == 0 or margins.size == 0:
            raise ValueError("D1 allocation trace must be nonempty")
        if self.pages < 0 or self.predicted_crossings < 0:
            raise ValueError("D1 allocation counts must be nonnegative")
        if not all(np.isfinite(value) for value in (self.route_loss, self.local_damage)):
            raise ValueError("D1 allocation objectives must be finite")


@dataclass(frozen=True)
class D1OraclePolicyResult:
    """One selected policy and its single exact final D1 execution."""

    policy: str
    allocation: D1AllocationTrace
    exact_d1_logits: np.ndarray
    exact_top8_agreement: float
    exact_final_logits: np.ndarray | None = None
    final_logit_kl: float | None = None

    def __post_init__(self) -> None:
        logits = np.asarray(self.exact_d1_logits)
        if not self.policy or logits.ndim != 1 or logits.size < 9:
            raise ValueError("D1 oracle policy result is invalid")
        if not 0.0 <= float(self.exact_top8_agreement) <= 1.0:
            raise ValueError("exact top-8 agreement must lie in [0,1]")
        if (self.exact_final_logits is None) != (self.final_logit_kl is None):
            raise ValueError("final logits and KL must either both be present or absent")
        if self.exact_final_logits is not None:
            final = np.asarray(self.exact_final_logits)
            kl = float(self.final_logit_kl)
            if (
                final.ndim != 1
                or final.size < 2
                or np.any(~np.isfinite(final))
                or not np.isfinite(kl)
                or kl < -1e-12
            ):
                raise ValueError("exact final logits/KL are invalid")


def adaptive_outsider_ranks(
    q4_logits: Sequence[float] | np.ndarray,
    margin_threshold: float,
    *,
    max_outsiders: int = 8,
) -> tuple[int, ...]:
    """Choose exact-Q4 outsider ranks close enough to threaten rank eight.

    Returned ranks are one-based. The closest outsider (rank 9) is always
    included so an otherwise quiet group still has a well-defined boundary.
    """

    logits = np.asarray(q4_logits, np.float64).reshape(-1)
    threshold = float(margin_threshold)
    cap = int(max_outsiders)
    if (
        logits.size < 9
        or np.any(~np.isfinite(logits))
        or not np.isfinite(threshold)
        or threshold < 0.0
        or cap < 1
    ):
        raise ValueError("adaptive outsider-pool inputs are invalid")
    order = np.argsort(-logits, kind="stable")
    boundary = float(logits[order[7]])
    selected = [
        rank
        for rank in range(9, logits.size + 1)
        if boundary - float(logits[order[rank - 1]]) <= threshold
    ][:cap]
    return tuple(selected if selected else [9])


def additive_local_damage_limit(
    local_optimal_damage: float,
    q2_damage: float,
    eta: float,
) -> float:
    """Return ``J_local* + eta J_Q2`` with explicit numerical checks."""

    optimum = float(local_optimal_damage)
    baseline = float(q2_damage)
    fraction = float(eta)
    if (
        not all(np.isfinite(value) for value in (optimum, baseline, fraction))
        or optimum < 0.0
        or baseline < 0.0
        or fraction < 0.0
    ):
        raise ValueError("additive local-guard values must be finite nonnegative")
    return optimum + fraction * baseline


def build_d1_boundary_problem(
    q4_logits: Sequence[float] | np.ndarray,
    logit_sensitivities: np.ndarray,
    *,
    selected_ranks: Sequence[int] = (6, 7, 8),
    outsider_ranks: Sequence[int] = tuple(range(9, 17)),
    severity: float | Sequence[float] | np.ndarray = 1.0,
    temperature: float | Sequence[float] | np.ndarray = 1.0,
    safety_margin: float | Sequence[float] | np.ndarray = 0.0,
) -> D1BoundaryProblem:
    """Build selected-versus-outsider D1 pairs from exact Q4 labels.

    Ranks are one-based and refer to the descending exact-Q4 router order.
    Individual-logit sensitivities are accepted so the unique expert VJPs can
    be reused; pair sensitivities are formed by subtraction.
    """

    logits = np.asarray(q4_logits, np.float64).reshape(-1)
    gradients = np.asarray(logit_sensitivities, np.float64)
    if logits.size < 9 or gradients.ndim != 2 or gradients.shape[0] != logits.size:
        raise ValueError("Q4 logits/sensitivities must cover at least nine experts")
    if not np.all(np.isfinite(logits)) or not np.all(np.isfinite(gradients)):
        raise ValueError("Q4 logits/sensitivities must be finite")
    selected_rank = tuple(int(value) for value in selected_ranks)
    outsider_rank = tuple(int(value) for value in outsider_ranks)
    if (
        not selected_rank
        or not outsider_rank
        or min(selected_rank) < 1
        or max(outsider_rank) > logits.size
        or any(value > 8 for value in selected_rank)
        or any(value <= 8 for value in outsider_rank)
        or len(set(selected_rank)) != len(selected_rank)
        or len(set(outsider_rank)) != len(outsider_rank)
    ):
        raise ValueError("D1 selected/outsider ranks are invalid")
    order = np.argsort(-logits, kind="stable")
    selected_unique = order[np.asarray(selected_rank) - 1]
    outsider_unique = order[np.asarray(outsider_rank) - 1]
    pairs = tuple(product(selected_unique.tolist(), outsider_unique.tolist()))
    selected = np.asarray([left for left, _ in pairs], np.int64)
    outsiders = np.asarray([right for _, right in pairs], np.int64)
    margins = logits[selected] - logits[outsiders]
    pair_gradients = gradients[selected] - gradients[outsiders]
    count = len(pairs)
    return D1BoundaryProblem(
        selected_expert_ids=selected,
        outsider_expert_ids=outsiders,
        q4_margins=margins,
        margin_sensitivities=pair_gradients,
        severity=_vector_parameter(severity, count, "severity", positive=False),
        temperature=_vector_parameter(
            temperature, count, "temperature", positive=True,
        ),
        safety_margin=_vector_parameter(
            safety_margin, count, "safety_margin", positive=False,
        ),
    )


def token_option_deltas(
    responses: Sequence[SplitProjectionResponses],
    frontiers: Sequence[Sequence[RateOption]],
) -> tuple[np.ndarray, ...]:
    """Return exact token-dependent ``approximate - Q4`` option deltas.

    ``responses`` is computed from the current token activation.  Consequently
    the returned effect of a physical refinement option is explicitly a
    function of the token, even though its page geometry is static.
    """

    if len(responses) != len(frontiers) or not responses:
        raise ValueError("responses and frontiers must be nonempty and aligned")
    result = []
    for response, frontier in zip(responses, frontiers):
        if not frontier:
            raise ValueError("every expert frontier must be nonempty")
        target = np.asarray(response.target_output, np.float64)
        result.append(np.stack([
            split_state_output(response, option.states) - target
            for option in frontier
        ]))
    return tuple(result)


def option_boundary_effects(
    option_deltas: Sequence[np.ndarray],
    execution_router_weights: Sequence[float] | np.ndarray,
    problem: D1BoundaryProblem,
) -> tuple[np.ndarray, ...]:
    """Project exact token option deltas onto signed D1 margins."""

    weights = np.asarray(execution_router_weights, np.float64).reshape(-1)
    if weights.shape != (len(option_deltas),) or np.any(~np.isfinite(weights)):
        raise ValueError("execution router weights must match routed experts")
    gradients = np.asarray(problem.margin_sensitivities, np.float64)
    result = []
    for expert, raw in enumerate(option_deltas):
        value = np.asarray(raw, np.float64)
        if value.ndim != 2 or value.shape[1] != problem.hidden_size:
            raise ValueError("option deltas must be [options,hidden]")
        result.append(weights[expert] * (value @ gradients.T))
    return tuple(result)


def directional_route_loss(
    q4_margins: Sequence[float] | np.ndarray,
    signed_effect: Sequence[float] | np.ndarray,
    *,
    severity: float | Sequence[float] | np.ndarray = 1.0,
    temperature: float | Sequence[float] | np.ndarray = 1.0,
    safety_margin: float | Sequence[float] | np.ndarray = 0.0,
) -> float:
    """Evaluate the one-sided softplus boundary risk stably."""

    margins = np.asarray(q4_margins, np.float64).reshape(-1)
    effect = np.asarray(signed_effect, np.float64).reshape(-1)
    if margins.shape != effect.shape or margins.size == 0:
        raise ValueError("margins and signed effects must be equal nonempty vectors")
    count = margins.size
    cost = _vector_parameter(severity, count, "severity", positive=False)
    tau = _vector_parameter(temperature, count, "temperature", positive=True)
    safe = _vector_parameter(safety_margin, count, "safety_margin", positive=False)
    argument = (safe - margins - effect) / tau
    return float(np.sum(cost * np.logaddexp(0.0, argument)))


def directional_route_hinge_loss(
    q4_margins: Sequence[float] | np.ndarray,
    signed_effect: Sequence[float] | np.ndarray,
    *,
    severity: float | Sequence[float] | np.ndarray = 1.0,
    safety_margin: float | Sequence[float] | np.ndarray = 0.0,
    squared: bool = True,
) -> float:
    """Return a one-sided loss that is exactly zero after certification.

    Unlike softplus, this objective gives no reward for pushing an already
    safe boundary farther away from its exact-Q4 margin.  Once all robust
    margins are safe, the allocator's existing secondary local-error term
    determines the choice.
    """

    margins = np.asarray(q4_margins, np.float64).reshape(-1)
    effect = np.asarray(signed_effect, np.float64).reshape(-1)
    if margins.shape != effect.shape or margins.size == 0:
        raise ValueError("margins and signed effects must be equal nonempty vectors")
    count = margins.size
    cost = _vector_parameter(severity, count, "severity", positive=False)
    safe = _vector_parameter(safety_margin, count, "safety_margin", positive=False)
    deficit = np.maximum(0.0, safe - margins - effect)
    if bool(squared):
        deficit = deficit * deficit
    return float(cost @ deficit)


def incumbent_repair_choice(
    policy_names: Sequence[str],
    predicted_crossings: Sequence[int] | np.ndarray,
    route_mass_churn: Sequence[float] | np.ndarray,
    centered_logit_mse: Sequence[float] | np.ndarray,
    local_damage: Sequence[float] | np.ndarray,
    pages: Sequence[int] | np.ndarray,
    *,
    incumbent_policy: str,
    boundary_violation: Sequence[float] | np.ndarray | None = None,
) -> int:
    """Choose a minimal D1 repair without globally replacing the incumbent.

    A safe incumbent is retained.  An unsafe incumbent can only be replaced
    by a candidate with strictly fewer predicted crossings.  This prevents
    a D1 objective from introducing membership errors merely to improve an
    aggregate soft score.  Robust boundary-violation depth is the first
    severity tie-break because the exact panel found it more informative than
    raw routing mass among candidates that already crossed.  Router-mass
    churn, centered-logit fidelity, local damage, traffic change, and policy
    name provide the remaining deterministic tie-breaks.
    """

    names = tuple(map(str, policy_names))
    count = len(names)
    if count == 0 or len(set(names)) != count or str(incumbent_policy) not in names:
        raise ValueError("repair candidates and incumbent policy are invalid")
    crossings = np.asarray(predicted_crossings, np.int64).reshape(-1)
    mass = np.asarray(route_mass_churn, np.float64).reshape(-1)
    centered = np.asarray(centered_logit_mse, np.float64).reshape(-1)
    local = np.asarray(local_damage, np.float64).reshape(-1)
    selected_pages = np.asarray(pages, np.int64).reshape(-1)
    violation = (
        np.zeros(count, np.float64)
        if boundary_violation is None
        else np.asarray(boundary_violation, np.float64).reshape(-1)
    )
    if any(value.shape != (count,) for value in (
        crossings, mass, centered, local, selected_pages, violation,
    )):
        raise ValueError("repair candidate metrics must align with policy names")
    if (
        np.any(crossings < 0)
        or np.any(selected_pages < 0)
        or any(np.any(~np.isfinite(value)) or np.any(value < 0.0) for value in (
            violation, mass, centered, local,
        ))
    ):
        raise ValueError("repair candidate metrics must be finite nonnegative")
    incumbent = names.index(str(incumbent_policy))
    incumbent_crossings = int(crossings[incumbent])
    if incumbent_crossings == 0:
        return incumbent
    eligible = [
        index for index in range(count)
        if int(crossings[index]) < incumbent_crossings
    ]
    if not eligible:
        return incumbent
    incumbent_pages = int(selected_pages[incumbent])
    return min(
        eligible,
        key=lambda index: (
            int(crossings[index]),
            float(violation[index]),
            float(mass[index]),
            float(centered[index]),
            float(local[index]),
            abs(int(selected_pages[index]) - incumbent_pages),
            names[index],
        ),
    )


def routing_mass_severity(
    q4_logits: Sequence[float] | np.ndarray,
    problem: D1BoundaryProblem,
) -> np.ndarray:
    """Price each possible swap by the exact-Q4 selected expert's route mass."""

    logits = np.asarray(q4_logits, np.float64).reshape(-1)
    if logits.size <= int(np.max(problem.outsider_expert_ids)):
        raise ValueError("router logits do not cover D1 boundary IDs")
    shifted = logits - float(np.max(logits))
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    return probabilities[np.asarray(problem.selected_expert_ids, np.int64)]


def functional_swap_severity(
    q4_logits: Sequence[float] | np.ndarray,
    expert_outputs: np.ndarray,
    problem: D1BoundaryProblem,
) -> np.ndarray:
    """Price swaps by selected route mass times exact expert-output distance."""

    outputs = np.asarray(expert_outputs, np.float64)
    if outputs.ndim != 2:
        raise ValueError("expert outputs must be [experts,hidden]")
    selected = np.asarray(problem.selected_expert_ids, np.int64)
    outsiders = np.asarray(problem.outsider_expert_ids, np.int64)
    if outputs.shape[0] <= max(int(selected.max()), int(outsiders.max())):
        raise ValueError("expert outputs do not cover D1 boundary IDs")
    difference = outputs[selected] - outputs[outsiders]
    distance = np.einsum("ij,ij->i", difference, difference, optimize=True)
    return routing_mass_severity(q4_logits, problem) * distance


def exact_candidate_logit_vjps(
    router_logits: object,
    layer_output: object,
    position: int,
    candidate_ids: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Materialize exact same-token candidate-logit VJPs for oracle labels.

    ``router_logits`` must be the differentiable next-layer router output
    produced from ``layer_output``. Only the gradient with respect to the
    same token's layer output is returned. This function is intentionally
    based on autograd and must not be called by the runtime controller.
    """

    import torch

    if not isinstance(router_logits, torch.Tensor) or not isinstance(layer_output, torch.Tensor):
        raise TypeError("D1 VJP inputs must be torch tensors")
    logits = router_logits
    hidden = layer_output
    if logits.ndim == 3 and logits.shape[0] == 1:
        logits = logits[0]
    if hidden.ndim == 3 and hidden.shape[0] == 1:
        hidden_view = hidden[0]
    elif hidden.ndim == 2:
        hidden_view = hidden
    else:
        raise ValueError("layer output must be [tokens,hidden] or [1,tokens,hidden]")
    if logits.ndim != 2 or logits.shape[0] != hidden_view.shape[0]:
        raise ValueError("router logits must be [tokens,experts]")
    token = int(position)
    ids = np.asarray(candidate_ids, np.int64).reshape(-1)
    if (
        token < 0
        or token >= logits.shape[0]
        or ids.size == 0
        or len(set(ids.tolist())) != ids.size
        or np.any(ids < 0)
        or np.any(ids >= logits.shape[1])
    ):
        raise ValueError("D1 VJP position/candidate IDs are invalid")
    if not hidden.requires_grad:
        raise ValueError("layer output must require gradients")
    rows = []
    for offset, expert in enumerate(ids.tolist()):
        gradient = torch.autograd.grad(
            logits[token, int(expert)],
            hidden,
            retain_graph=offset + 1 < ids.size,
            create_graph=False,
            allow_unused=False,
        )[0]
        view = gradient[0] if gradient.ndim == 3 else gradient
        rows.append(view[token].detach().float().cpu().numpy())
    return np.stack(rows)


def _combined_local(
    weighted_features: Sequence[np.ndarray], indices: np.ndarray,
) -> tuple[np.ndarray, float]:
    combined = sum(
        (weighted_features[expert][int(option)] for expert, option in enumerate(indices)),
        np.zeros(weighted_features[0].shape[1], np.float64),
    )
    return combined, float(combined @ combined)


def _strictly_better(
    candidate: tuple[float, float, int],
    current: tuple[float, float, int],
    tolerance: float,
) -> bool:
    if candidate[0] < current[0] - tolerance:
        return True
    if abs(candidate[0] - current[0]) <= tolerance:
        if candidate[1] < current[1] - tolerance:
            return True
        if abs(candidate[1] - current[1]) <= tolerance:
            return candidate[2] < current[2]
    return False


def d1_group_option_allocate(
    frontiers: Sequence[Sequence[RateOption]],
    option_effects: Sequence[np.ndarray],
    problem: D1BoundaryProblem,
    page_budget: int,
    seed_indices: Sequence[int] | np.ndarray,
    *,
    local_option_features: Sequence[np.ndarray],
    local_router_weights: Sequence[float] | np.ndarray,
    local_damage_limit: float,
    max_coordinate_sweeps: int = 8,
    max_pair_passes: int = 8,
    tolerance: float = 1e-12,
    route_loss_kind: str = "softplus",
) -> D1AllocationTrace:
    """Minimize exact directional D1 risk with an additive local guardrail.

    Search operates only on complete expert frontier options.  Coordinate
    updates are followed by two-expert exchanges, matching the existing
    validation-only combined-qenergy oracle's bounded search structure.
    """

    count = len(frontiers)
    selected = np.asarray(seed_indices, np.int64).reshape(-1).copy()
    local_weights = np.asarray(local_router_weights, np.float64).reshape(-1)
    if count == 0 or selected.shape != (count,) or local_weights.shape != (count,):
        raise ValueError("D1 allocation inputs must align with expert count")
    if len(option_effects) != count or len(local_option_features) != count:
        raise ValueError("D1 option effects/features must align with frontiers")
    loss_kind = str(route_loss_kind)
    if loss_kind not in {"softplus", "squared_hinge"}:
        raise ValueError("D1 route loss kind must be softplus or squared_hinge")
    budget = int(page_budget)
    limit = float(local_damage_limit)
    if budget < 0 or not np.isfinite(limit) or limit < 0.0:
        raise ValueError("D1 page budget/local guardrail is invalid")

    effects: list[np.ndarray] = []
    weighted_local: list[np.ndarray] = []
    for expert, frontier in enumerate(frontiers):
        route = np.asarray(option_effects[expert], np.float64)
        local = np.asarray(local_option_features[expert], np.float64)
        if (
            not frontier
            or route.shape != (len(frontier), problem.boundaries)
            or local.ndim != 2
            or local.shape[0] != len(frontier)
            or selected[expert] < 0
            or selected[expert] >= len(frontier)
        ):
            raise ValueError("D1 frontier option arrays are inconsistent")
        if not np.all(np.isfinite(route)) or not np.all(np.isfinite(local)):
            raise ValueError("D1 option arrays must be finite")
        effects.append(route)
        weighted_local.append(local_weights[expert] * local)
    if len({value.shape[1] for value in weighted_local}) != 1:
        raise ValueError("local feature widths changed across experts")

    def pages_for(indices: np.ndarray) -> int:
        return int(sum(
            int(frontiers[expert][int(option)].pages)
            for expert, option in enumerate(indices)
        ))

    pages = pages_for(selected)
    if pages > budget:
        raise ValueError("D1 seed allocation exceeds page budget")
    combined_effect = sum(
        (effects[expert][int(option)] for expert, option in enumerate(selected)),
        np.zeros(problem.boundaries, np.float64),
    )
    combined_local, local_damage = _combined_local(weighted_local, selected)
    if local_damage > limit + tolerance:
        raise ValueError("D1 seed allocation violates the local guardrail")

    def route_loss(effect: np.ndarray) -> float:
        if loss_kind == "squared_hinge":
            return directional_route_hinge_loss(
                problem.q4_margins,
                effect,
                severity=problem.severity,
                safety_margin=problem.safety_margin,
                squared=True,
            )
        return directional_route_loss(
            problem.q4_margins,
            effect,
            severity=problem.severity,
            temperature=problem.temperature,
            safety_margin=problem.safety_margin,
        )

    loss = route_loss(combined_effect)
    coordinate_completed = 0
    for sweep in range(int(max_coordinate_sweeps)):
        changed = False
        for expert in range(count):
            source = int(selected[expert])
            route_without = combined_effect - effects[expert][source]
            local_without = combined_local - weighted_local[expert][source]
            base_pages = pages - int(frontiers[expert][source].pages)
            best = (loss, local_damage, pages, source, combined_effect, combined_local)
            for destination, option in enumerate(frontiers[expert]):
                candidate_pages = base_pages + int(option.pages)
                if candidate_pages > budget:
                    continue
                candidate_local = local_without + weighted_local[expert][destination]
                candidate_damage = float(candidate_local @ candidate_local)
                if candidate_damage > limit + tolerance:
                    continue
                candidate_effect = route_without + effects[expert][destination]
                candidate_loss = route_loss(candidate_effect)
                if _strictly_better(
                    (candidate_loss, candidate_damage, candidate_pages),
                    best[:3],
                    tolerance,
                ):
                    best = (
                        candidate_loss, candidate_damage, candidate_pages,
                        destination, candidate_effect, candidate_local,
                    )
            if int(best[3]) != source:
                loss, local_damage, pages = map(float, best[:3])
                pages = int(pages)
                selected[expert] = int(best[3])
                combined_effect = np.asarray(best[4], np.float64)
                combined_local = np.asarray(best[5], np.float64)
                changed = True
        coordinate_completed = sweep + 1
        if not changed:
            break

    pair_completed = 0
    for pair_pass in range(int(max_pair_passes)):
        best_pair = None
        best_key = (loss, local_damage, pages)
        for left in range(count):
            old_left = int(selected[left])
            for right in range(left + 1, count):
                old_right = int(selected[right])
                route_without = (
                    combined_effect - effects[left][old_left] - effects[right][old_right]
                )
                local_without = (
                    combined_local
                    - weighted_local[left][old_left]
                    - weighted_local[right][old_right]
                )
                base_pages = (
                    pages
                    - int(frontiers[left][old_left].pages)
                    - int(frontiers[right][old_right].pages)
                )
                for li, left_option in enumerate(frontiers[left]):
                    remaining = budget - base_pages - int(left_option.pages)
                    if remaining < 0:
                        continue
                    for ri, right_option in enumerate(frontiers[right]):
                        if int(right_option.pages) > remaining:
                            continue
                        candidate_pages = (
                            base_pages + int(left_option.pages) + int(right_option.pages)
                        )
                        candidate_local = (
                            local_without + weighted_local[left][li] + weighted_local[right][ri]
                        )
                        candidate_damage = float(candidate_local @ candidate_local)
                        if candidate_damage > limit + tolerance:
                            continue
                        candidate_effect = (
                            route_without + effects[left][li] + effects[right][ri]
                        )
                        candidate_loss = route_loss(candidate_effect)
                        candidate_key = (
                            candidate_loss, candidate_damage, candidate_pages,
                        )
                        if _strictly_better(candidate_key, best_key, tolerance):
                            best_key = candidate_key
                            best_pair = (
                                left, right, li, ri, candidate_effect, candidate_local,
                            )
        pair_completed = pair_pass + 1
        if best_pair is None:
            break
        left, right, li, ri, combined_effect, combined_local = best_pair
        selected[left], selected[right] = int(li), int(ri)
        loss, local_damage, pages = best_key

    predicted = np.asarray(problem.q4_margins) + combined_effect
    return D1AllocationTrace(
        option_indices=selected.copy(),
        pages=int(pages),
        route_loss=float(loss),
        local_damage=float(local_damage),
        predicted_margins=predicted.copy(),
        predicted_crossings=int(np.count_nonzero(predicted <= 0.0)),
        coordinate_sweeps=int(coordinate_completed),
        pair_passes=int(pair_completed),
    )


def run_d1_oracle_policies(
    frontiers: Sequence[Sequence[RateOption]],
    option_effects: Sequence[np.ndarray],
    policy_problems: Mapping[str, D1BoundaryProblem],
    page_budget: int,
    seed_indices: Sequence[int] | np.ndarray,
    *,
    local_option_features: Sequence[np.ndarray],
    local_router_weights: Sequence[float] | np.ndarray,
    local_damage_limit: float,
    q4_d1_logits: Sequence[float] | np.ndarray,
    exact_replay: Callable[
        [np.ndarray], np.ndarray | tuple[np.ndarray, np.ndarray]
    ],
    q4_final_logits: Sequence[float] | np.ndarray | None = None,
    max_coordinate_sweeps: int = 8,
    max_pair_passes: int = 8,
    route_loss_kind: str = "softplus",
) -> tuple[D1OraclePolicyResult, ...]:
    """Select each D1 policy, then execute its final allocation exactly once.

    This is the Experiment A boundary: there is no crude starting state,
    speculative execution, omitted-tail prediction, or route certificate.
    ``exact_replay`` receives final option indices and must return the actual
    next-layer router logits produced by that allocation. If
    ``q4_final_logits`` is supplied, the same one-shot callback must return
    ``(d1_logits, final_logits)`` so terminal KL is not obtained from a second
    execution.
    """

    reference = np.asarray(q4_d1_logits, np.float64).reshape(-1)
    if reference.size < 9 or np.any(~np.isfinite(reference)):
        raise ValueError("exact Q4 D1 logits must cover at least nine experts")
    if not policy_problems:
        raise ValueError("at least one D1 oracle policy is required")
    final_reference = None
    if q4_final_logits is not None:
        final_reference = np.asarray(q4_final_logits, np.float64).reshape(-1)
        if final_reference.size < 2 or np.any(~np.isfinite(final_reference)):
            raise ValueError("exact Q4 final logits are invalid")
    reference_top = set(np.argsort(-reference, kind="stable")[:8].tolist())
    results = []
    for policy in sorted(policy_problems):
        trace = d1_group_option_allocate(
            frontiers,
            option_effects,
            policy_problems[policy],
            page_budget,
            seed_indices,
            local_option_features=local_option_features,
            local_router_weights=local_router_weights,
            local_damage_limit=local_damage_limit,
            max_coordinate_sweeps=max_coordinate_sweeps,
            max_pair_passes=max_pair_passes,
            route_loss_kind=route_loss_kind,
        )
        replayed = exact_replay(trace.option_indices.copy())
        final_logits = None
        if isinstance(replayed, tuple):
            if len(replayed) != 2 or final_reference is None:
                raise ValueError("paired exact replay requires Q4 final logits")
            logits = np.asarray(replayed[0], np.float64).reshape(-1)
            final_logits = np.asarray(replayed[1], np.float64).reshape(-1)
            if (
                final_logits.shape != final_reference.shape
                or np.any(~np.isfinite(final_logits))
            ):
                raise ValueError("exact final replay logits changed shape or became non-finite")
        else:
            if final_reference is not None:
                raise ValueError("exact replay omitted final logits required for KL")
            logits = np.asarray(replayed, np.float64).reshape(-1)
        if logits.shape != reference.shape or np.any(~np.isfinite(logits)):
            raise ValueError("exact D1 replay logits changed shape or became non-finite")
        candidate_top = set(np.argsort(-logits, kind="stable")[:8].tolist())
        final_kl = None
        if final_logits is not None:
            reference_shifted = final_reference - float(np.max(final_reference))
            candidate_shifted = final_logits - float(np.max(final_logits))
            reference_log_probability = reference_shifted - np.log(
                np.exp(reference_shifted).sum()
            )
            candidate_log_probability = candidate_shifted - np.log(
                np.exp(candidate_shifted).sum()
            )
            probability = np.exp(reference_log_probability)
            final_kl = max(0.0, float(
                probability @ (reference_log_probability - candidate_log_probability)
            ))
        results.append(D1OraclePolicyResult(
            policy=str(policy),
            allocation=trace,
            exact_d1_logits=logits.copy(),
            exact_top8_agreement=len(reference_top & candidate_top) / 8.0,
            exact_final_logits=(None if final_logits is None else final_logits.copy()),
            final_logit_kl=final_kl,
        ))
    return tuple(results)
