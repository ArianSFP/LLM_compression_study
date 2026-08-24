"""Token-dependent latent primitives for the sequential D1 controller.

Static metadata describes low-rank page geometry.  Dynamic token responses
produce ``c_p(x)``.  The module contains no exact-Q4 routes, exact omitted
page effects, or exact D1 VJPs; those belong only to oracle labels and
evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .split_interaction_field import (
    SPLIT_STATE_DOWN_HIGH,
    SPLIT_STATE_HIDDEN_INDEX,
)


__all__ = [
    "TokenLatentExpert",
    "RouteCertificate",
    "PrecisionTransition",
    "TensorTransitionBatch",
    "token_hidden_responses",
    "combined_remaining_latent",
    "project_candidate_sensitivities",
    "rmsnorm_router_sensitivities",
    "predict_corrected_logits",
    "certify_topk",
    "signed_effect_metrics",
    "corrected_logit_metrics",
    "score_single_page_transitions",
    "score_single_page_transitions_torch",
    "apply_precision_transition",
    "apply_precision_transitions",
    "apply_precision_transitions_torch",
]


PROJECTION_BITS = ((0, "down"), (1, "up"), (2, "gate"))


def _silu(value: np.ndarray) -> np.ndarray:
    source = np.asarray(value, np.float64)
    result = np.empty_like(source)
    positive = source >= 0.0
    result[positive] = source[positive] / (1.0 + np.exp(-source[positive]))
    exponential = np.exp(source[~positive])
    result[~positive] = source[~positive] * exponential / (1.0 + exponential)
    return result


def token_hidden_responses(
    gate2: Sequence[float] | np.ndarray,
    up2: Sequence[float] | np.ndarray,
    gate4: Sequence[float] | np.ndarray,
    up4: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Build dynamic ``h22,h24,h42,h44`` from exact or predicted responses.

    At runtime ``gate2``/``up2`` are available from resident execution while
    ``gate4``/``up4`` may be supplied by a token-conditioned response model.
    Applying SiLU and multiplication here preserves the conditional gate/up
    interaction instead of treating refinement pages as static additive atoms.
    """

    values = [np.asarray(value, np.float64).reshape(-1) for value in (
        gate2, up2, gate4, up4,
    )]
    if not values[0].size or any(value.shape != values[0].shape for value in values):
        raise ValueError("gate/up responses must be equal nonempty vectors")
    if not all(np.all(np.isfinite(value)) for value in values):
        raise ValueError("gate/up responses must be finite")
    g2, u2, g4, u4 = values
    return np.stack(
        (_silu(g2) * u2, _silu(g2) * u4, _silu(g4) * u2, _silu(g4) * u4),
        axis=1,
    )


def _states(value: Sequence[int] | np.ndarray, units: int) -> np.ndarray:
    result = np.asarray(value, np.int64).reshape(-1)
    if result.shape != (int(units),) or np.any((result < 0) | (result > 7)):
        raise ValueError("precision states must contain one value in [0,7] per unit")
    return result.copy()


@dataclass(frozen=True)
class TokenLatentExpert:
    """One expert's static down geometry and dynamic token responses."""

    down2_coefficients: np.ndarray
    down4_coefficients: np.ndarray
    hidden_responses: np.ndarray

    def __post_init__(self) -> None:
        down2 = np.asarray(self.down2_coefficients)
        down4 = np.asarray(self.down4_coefficients)
        hidden = np.asarray(self.hidden_responses)
        if down2.ndim != 2 or down4.shape != down2.shape:
            raise ValueError("down latent coefficients must be equal [unit,rank] arrays")
        if hidden.shape != (down2.shape[0], 4):
            raise ValueError("dynamic hidden responses must be [unit,4]")
        if not all(np.all(np.isfinite(value)) for value in (down2, down4, hidden)):
            raise ValueError("token latent expert values must be finite")

    @property
    def units(self) -> int:
        return int(np.asarray(self.down2_coefficients).shape[0])

    @property
    def rank(self) -> int:
        return int(np.asarray(self.down2_coefficients).shape[1])

    def unit_state_latent(self, unit: int, state: int) -> np.ndarray:
        """Return one unit's token-dependent output latent in one state."""

        row, precision = int(unit), int(state)
        if row < 0 or row >= self.units or precision < 0 or precision > 7:
            raise ValueError("unit/state lies outside the token latent expert")
        hidden = float(np.asarray(self.hidden_responses)[
            row, int(SPLIT_STATE_HIDDEN_INDEX[precision])
        ])
        coefficients = (
            np.asarray(self.down4_coefficients, np.float64)
            if bool(SPLIT_STATE_DOWN_HIGH[precision])
            else np.asarray(self.down2_coefficients, np.float64)
        )
        return hidden * coefficients[row]

    def state_latent(self, states: Sequence[int] | np.ndarray) -> np.ndarray:
        """Return the token's complete expert output latent for a state vector."""

        state = _states(states, self.units)
        rows = np.arange(self.units)
        hidden = np.asarray(self.hidden_responses, np.float64)[
            rows, SPLIT_STATE_HIDDEN_INDEX[state]
        ]
        coefficients = np.where(
            SPLIT_STATE_DOWN_HIGH[state, None],
            np.asarray(self.down4_coefficients, np.float64),
            np.asarray(self.down2_coefficients, np.float64),
        )
        return np.sum(hidden[:, None] * coefficients, axis=0)

    def remaining_to_q4(self, states: Sequence[int] | np.ndarray) -> np.ndarray:
        """Return dynamic ``Q4 - current`` latent correction for this token."""

        state = _states(states, self.units)
        target = np.full(self.units, 7, np.int64)
        return self.state_latent(target) - self.state_latent(state)


@dataclass(frozen=True)
class RouteCertificate:
    """Joint top-k interval certificate over the evaluated candidate pool."""

    top_ids: np.ndarray
    certified: bool
    lower_selected: float
    upper_outsider: float
    certificate_gap: float
    pool_complete: bool


@dataclass(frozen=True)
class PrecisionTransition:
    """One legal single-page gate/up/down state transition."""

    expert: int
    unit: int
    projection: str
    source_state: int
    destination_state: int
    route_risk_reduction: float
    uncertainty_reduction: float
    local_gain: float
    score_per_byte: float


@dataclass(frozen=True)
class TensorTransitionBatch:
    """Device-resident shortlist returned by the fused tensor scorer."""

    expert: object
    unit: object
    projection_slot: object
    source_state: object
    destination_state: object
    route_risk_reduction: object
    uncertainty_reduction: object
    local_gain: object
    score_per_byte: object

    def __len__(self) -> int:
        return int(self.expert.numel())


def combined_remaining_latent(
    experts: Sequence[TokenLatentExpert],
    state_vectors: Sequence[Sequence[int] | np.ndarray],
    execution_router_weights: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Combine token-dependent omitted-tail corrections using execution weights."""

    if len(experts) != len(state_vectors) or not experts:
        raise ValueError("experts and state vectors must be nonempty and aligned")
    ranks = {expert.rank for expert in experts}
    weights = np.asarray(execution_router_weights, np.float64).reshape(-1)
    if len(ranks) != 1 or weights.shape != (len(experts),):
        raise ValueError("expert ranks/router weights do not align")
    if np.any(~np.isfinite(weights)):
        raise ValueError("execution router weights must be finite")
    result = np.zeros(experts[0].rank, np.float64)
    for weight, expert, states in zip(weights, experts, state_vectors):
        result += float(weight) * expert.remaining_to_q4(states)
    return result


def project_candidate_sensitivities(
    output_basis: np.ndarray,
    candidate_sensitivities: np.ndarray,
) -> np.ndarray:
    """Project dynamic candidate-logit sensitivities into the output latent."""

    basis = np.asarray(output_basis, np.float64)
    gradients = np.asarray(candidate_sensitivities, np.float64)
    if basis.ndim != 2 or gradients.ndim != 2 or gradients.shape[1] != basis.shape[0]:
        raise ValueError("output basis and candidate sensitivities do not align")
    if not np.all(np.isfinite(basis)) or not np.all(np.isfinite(gradients)):
        raise ValueError("output basis/sensitivities must be finite")
    return gradients @ basis


def rmsnorm_router_sensitivities(
    hidden: Sequence[float] | np.ndarray,
    rmsnorm_weight: Sequence[float] | np.ndarray,
    router_rows: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Analytic direct-residual RMSNorm/router sensitivity approximation.

    This omits the next token-mixer Jacobian. Its approximation error must be
    included in calibrated runtime uncertainty; exact full VJPs remain oracle
    labels only.
    """

    value = np.asarray(hidden, np.float64).reshape(-1)
    weight = np.asarray(rmsnorm_weight, np.float64).reshape(-1)
    rows = np.asarray(router_rows, np.float64)
    if value.size == 0 or weight.shape != value.shape or rows.ndim != 2:
        raise ValueError("RMSNorm/router sensitivity shapes are invalid")
    if rows.shape[1] != value.size or not all(np.all(np.isfinite(item)) for item in (
        value, weight, rows,
    )):
        raise ValueError("RMSNorm/router sensitivity values are invalid")
    if not np.isfinite(epsilon) or float(epsilon) <= 0.0:
        raise ValueError("RMSNorm epsilon must be positive finite")
    rms = float(np.sqrt(np.mean(value * value) + float(epsilon)))
    scaled_rows = rows * weight[None, :]
    dot = scaled_rows @ value
    return scaled_rows / rms - (
        dot[:, None] * value[None, :] / (value.size * rms ** 3)
    )


def predict_corrected_logits(
    crude_candidate_logits: Sequence[float] | np.ndarray,
    projected_sensitivities: np.ndarray,
    latent_correction: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Predict candidate logits after a dynamic latent output correction."""

    logits = np.asarray(crude_candidate_logits, np.float64).reshape(-1)
    projected = np.asarray(projected_sensitivities, np.float64)
    correction = np.asarray(latent_correction, np.float64).reshape(-1)
    if projected.shape != (logits.size, correction.size):
        raise ValueError("candidate logits/sensitivities/latent correction do not align")
    if not all(np.all(np.isfinite(value)) for value in (logits, projected, correction)):
        raise ValueError("corrected-logit inputs must be finite")
    return logits + projected @ correction


def certify_topk(
    predicted_logits: Sequence[float] | np.ndarray,
    error_radius: Sequence[float] | np.ndarray,
    *,
    top_k: int = 8,
    candidate_ids: Sequence[int] | np.ndarray | None = None,
    excluded_upper_bound: float | None = None,
) -> RouteCertificate:
    """Certify a predicted top-k set with calibrated joint logit intervals."""

    logits = np.asarray(predicted_logits, np.float64).reshape(-1)
    radius = np.asarray(error_radius, np.float64).reshape(-1)
    if logits.shape != radius.shape or logits.size <= int(top_k):
        raise ValueError("predicted logits/radii do not support the requested top-k")
    if np.any(~np.isfinite(logits)) or np.any(~np.isfinite(radius)) or np.any(radius < 0):
        raise ValueError("predicted logits/radii must be finite and radii nonnegative")
    ids = (
        np.arange(logits.size, dtype=np.int64)
        if candidate_ids is None
        else np.asarray(candidate_ids, np.int64).reshape(-1)
    )
    if ids.shape != logits.shape or len(set(ids.tolist())) != ids.size:
        raise ValueError("candidate IDs must be unique and align with logits")
    order = np.argsort(-logits, kind="stable")
    selected = order[: int(top_k)]
    outsiders = order[int(top_k) :]
    lower = float(np.min(logits[selected] - radius[selected]))
    upper = float(np.max(logits[outsiders] + radius[outsiders]))
    complete = excluded_upper_bound is None
    if excluded_upper_bound is not None:
        excluded = float(excluded_upper_bound)
        if not np.isfinite(excluded):
            raise ValueError("excluded outsider upper bound must be finite")
        upper = max(upper, excluded)
    gap = lower - upper
    return RouteCertificate(
        top_ids=ids[selected].copy(),
        certified=bool(gap > 0.0),
        lower_selected=lower,
        upper_outsider=upper,
        certificate_gap=float(gap),
        pool_complete=bool(complete),
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    result = np.empty(values.size, np.float64)
    cursor = 0
    while cursor < values.size:
        end = cursor + 1
        while end < values.size and values[order[end]] == values[order[cursor]]:
            end += 1
        result[order[cursor:end]] = 0.5 * (cursor + end - 1)
        cursor = end
    return result


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def signed_effect_metrics(
    predicted: np.ndarray,
    exact: np.ndarray,
    *,
    error_radius: np.ndarray | None = None,
    top_pages: int = 128,
) -> dict[str, float | int]:
    """Evaluate token-dependent signed page-effect prediction directly."""

    estimate = np.asarray(predicted, np.float64).reshape(-1)
    target = np.asarray(exact, np.float64).reshape(-1)
    if estimate.shape != target.shape or estimate.size == 0:
        raise ValueError("predicted/exact signed effects must be equal nonempty arrays")
    if np.any(~np.isfinite(estimate)) or np.any(~np.isfinite(target)):
        raise ValueError("signed effects must be finite")
    nonzero = target != 0.0
    count = min(max(int(top_pages), 1), target.size)
    predicted_top = set(np.argsort(-np.abs(estimate), kind="stable")[:count].tolist())
    exact_top = set(np.argsort(-np.abs(target), kind="stable")[:count].tolist())
    error = estimate - target
    result: dict[str, float | int] = {
        "effects": int(target.size),
        "pearson_correlation": _correlation(estimate, target),
        "spearman_correlation": _correlation(
            _average_ranks(estimate), _average_ranks(target),
        ),
        "sign_accuracy": (
            float(np.mean(np.sign(estimate[nonzero]) == np.sign(target[nonzero])))
            if np.any(nonzero) else 1.0
        ),
        "top_page_overlap": len(predicted_top & exact_top) / float(count),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "max_abs_error": float(np.max(np.abs(error))),
    }
    if error_radius is not None:
        radius = np.asarray(error_radius, np.float64).reshape(-1)
        if radius.shape != target.shape or np.any(~np.isfinite(radius)) or np.any(radius < 0):
            raise ValueError("signed-effect error radii are invalid")
        result["interval_coverage"] = float(np.mean(np.abs(error) <= radius))
        result["mean_error_radius"] = float(np.mean(radius))
    return result


def corrected_logit_metrics(
    predicted: np.ndarray,
    exact: np.ndarray,
    *,
    error_radius: np.ndarray | None = None,
) -> dict[str, float]:
    """Measure corrected-logit error and optional uncertainty calibration."""

    estimate = np.asarray(predicted, np.float64)
    target = np.asarray(exact, np.float64)
    if estimate.shape != target.shape or estimate.size == 0:
        raise ValueError("predicted/exact corrected logits must align")
    error = estimate - target
    if np.any(~np.isfinite(error)):
        raise ValueError("corrected logits must be finite")
    result = {
        "corrected_logit_mae": float(np.mean(np.abs(error))),
        "corrected_logit_rmse": float(np.sqrt(np.mean(error * error))),
        "corrected_logit_max_abs_error": float(np.max(np.abs(error))),
        "corrected_logit_correlation": _correlation(estimate.reshape(-1), target.reshape(-1)),
    }
    if error_radius is not None:
        radius = np.asarray(error_radius, np.float64)
        if radius.shape != target.shape or np.any(~np.isfinite(radius)) or np.any(radius < 0):
            raise ValueError("corrected-logit error radii are invalid")
        result["corrected_logit_interval_coverage"] = float(
            np.mean(np.abs(error) <= radius)
        )
        result["corrected_logit_mean_radius"] = float(np.mean(radius))
    return result


def score_single_page_transitions(
    experts: Sequence[TokenLatentExpert],
    state_vectors: Sequence[Sequence[int] | np.ndarray],
    execution_router_weights: Sequence[float] | np.ndarray,
    risk_adjoint: Sequence[float] | np.ndarray,
    *,
    uncertainty_reduction: np.ndarray | None = None,
    local_gain: np.ndarray | None = None,
    page_bytes: int = 512,
    shortlist: int | None = None,
    unique_units: bool = False,
) -> tuple[PrecisionTransition, ...]:
    """Rank legal one-page transitions using dynamic ``c_p(x)`` effects.

    Arithmetic is vectorized over units and the full numeric ordering is
    formed before Python objects are materialized.  Runtime callers should
    request only their next batch through ``shortlist`` and set
    ``unique_units`` so every action in that batch remains valid until the
    next response refresh.  ``shortlist=None`` preserves the exhaustive
    research/evaluation surface.
    """

    if len(experts) != len(state_vectors) or not experts:
        raise ValueError("experts/state vectors must be nonempty and aligned")
    ranks = {expert.rank for expert in experts}
    units = {expert.units for expert in experts}
    weights = np.asarray(execution_router_weights, np.float64).reshape(-1)
    adjoint = np.asarray(risk_adjoint, np.float64).reshape(-1)
    if len(ranks) != 1 or len(units) != 1 or weights.shape != (len(experts),):
        raise ValueError("transition expert dimensions/router weights changed")
    if adjoint.shape != (experts[0].rank,) or np.any(~np.isfinite(adjoint)):
        raise ValueError("risk adjoint does not match latent rank")
    if int(page_bytes) <= 0:
        raise ValueError("page_bytes must be positive")
    if shortlist is not None and int(shortlist) < 1:
        raise ValueError("transition shortlist must be positive")
    shape = (len(experts), experts[0].units, 3)
    uncertainty = (
        np.zeros(shape, np.float64)
        if uncertainty_reduction is None
        else np.asarray(uncertainty_reduction, np.float64)
    )
    local = (
        np.zeros(shape, np.float64)
        if local_gain is None
        else np.asarray(local_gain, np.float64)
    )
    if uncertainty.shape != shape or local.shape != shape:
        raise ValueError("transition uncertainty/local gains must be [experts,units,3]")
    if (
        np.any(~np.isfinite(uncertainty))
        or np.any(~np.isfinite(local))
        or np.any(uncertainty < 0.0)
    ):
        raise ValueError("transition uncertainty/local values are invalid")

    expert_columns = []
    unit_columns = []
    slot_columns = []
    source_columns = []
    destination_columns = []
    reduction_columns = []
    uncertainty_columns = []
    local_columns = []
    for expert_index, (expert, state_value, weight) in enumerate(
        zip(experts, state_vectors, weights)
    ):
        states = _states(state_value, expert.units)
        rows = np.arange(expert.units, dtype=np.int64)
        source_hidden = np.asarray(expert.hidden_responses, np.float64)[
            rows, SPLIT_STATE_HIDDEN_INDEX[states]
        ]
        source_coefficients = np.where(
            SPLIT_STATE_DOWN_HIGH[states, None],
            np.asarray(expert.down4_coefficients, np.float64),
            np.asarray(expert.down2_coefficients, np.float64),
        )
        source_latent = source_hidden[:, None] * source_coefficients
        for slot, (bit, _) in enumerate(PROJECTION_BITS):
            valid = (states & (1 << bit)) == 0
            if not np.any(valid):
                continue
            units_valid = rows[valid]
            source_valid = states[valid]
            destination = source_valid | (1 << bit)
            destination_hidden = np.asarray(expert.hidden_responses, np.float64)[
                units_valid, SPLIT_STATE_HIDDEN_INDEX[destination]
            ]
            destination_coefficients = np.where(
                SPLIT_STATE_DOWN_HIGH[destination, None],
                np.asarray(expert.down4_coefficients, np.float64)[valid],
                np.asarray(expert.down2_coefficients, np.float64)[valid],
            )
            delta = float(weight) * (
                destination_hidden[:, None] * destination_coefficients
                - source_latent[valid]
            )
            expert_columns.append(np.full(units_valid.size, expert_index, np.int64))
            unit_columns.append(units_valid)
            slot_columns.append(np.full(units_valid.size, slot, np.int64))
            source_columns.append(source_valid)
            destination_columns.append(destination)
            reduction_columns.append(-(delta @ adjoint))
            uncertainty_columns.append(uncertainty[expert_index, valid, slot])
            local_columns.append(local[expert_index, valid, slot])
    if not expert_columns:
        return ()
    expert_id = np.concatenate(expert_columns)
    unit_id = np.concatenate(unit_columns)
    slot_id = np.concatenate(slot_columns)
    source = np.concatenate(source_columns)
    destination = np.concatenate(destination_columns)
    reduction = np.concatenate(reduction_columns)
    uncertainty_value = np.concatenate(uncertainty_columns)
    local_value = np.concatenate(local_columns)
    score = (reduction + uncertainty_value) / int(page_bytes)
    order = np.lexsort((
        destination,
        unit_id,
        expert_id,
        -local_value,
        -uncertainty_value,
        -score,
    ))
    limit = None if shortlist is None else int(shortlist)
    if bool(unique_units):
        retained = []
        seen = set()
        for index in order.tolist():
            key = (int(expert_id[index]), int(unit_id[index]))
            if key in seen:
                continue
            seen.add(key)
            retained.append(index)
            if limit is not None and len(retained) >= limit:
                break
        order = np.asarray(retained, np.int64)
    elif limit is not None:
        order = order[:limit]
    return tuple(
        PrecisionTransition(
            expert=int(expert_id[index]),
            unit=int(unit_id[index]),
            projection=PROJECTION_BITS[int(slot_id[index])][1],
            source_state=int(source[index]),
            destination_state=int(destination[index]),
            route_risk_reduction=float(reduction[index]),
            uncertainty_reduction=float(uncertainty_value[index]),
            local_gain=float(local_value[index]),
            score_per_byte=float(score[index]),
        )
        for index in order.tolist()
    )


def score_single_page_transitions_torch(
    down2_coefficients: object,
    down4_coefficients: object,
    hidden_responses: object,
    state_vectors: object,
    execution_router_weights: object,
    risk_adjoint: object,
    *,
    uncertainty_reduction: object | None = None,
    local_gain: object | None = None,
    page_bytes: int = 512,
    shortlist: int = 128,
) -> TensorTransitionBatch:
    """Score one distinct-unit batch without leaving the tensor device.

    Inputs are dense ``[experts,units,...]`` tensors.  The best legal
    projection transition for each expert/unit is selected first, followed by
    a device ``topk`` over units.  This is the runtime path; the exhaustive
    NumPy scorer remains the reference implementation and analysis surface.
    """

    import torch

    tensors = (
        down2_coefficients,
        down4_coefficients,
        hidden_responses,
        state_vectors,
        execution_router_weights,
        risk_adjoint,
    )
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("tensor transition scoring requires torch tensors")
    down2 = down2_coefficients
    down4 = down4_coefficients
    hidden = hidden_responses
    states = state_vectors
    weights = execution_router_weights
    adjoint = risk_adjoint
    if (
        down2.ndim != 3
        or down4.shape != down2.shape
        or hidden.shape != down2.shape[:2] + (4,)
        or states.shape != down2.shape[:2]
        or weights.shape != (down2.shape[0],)
        or adjoint.shape != (down2.shape[2],)
    ):
        raise ValueError("tensor transition shapes must be [expert,unit,rank]")
    device = down2.device
    if any(value.device != device for value in tensors[1:]):
        raise ValueError("tensor transition inputs must share one device")
    if states.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("tensor precision states must be integral")
    if bool(torch.any((states < 0) | (states > 7)).item()):
        raise ValueError("tensor precision states must lie in [0,7]")
    batch = int(shortlist)
    if int(page_bytes) <= 0 or batch < 1:
        raise ValueError("page bytes/transition shortlist must be positive")
    shape = down2.shape[:2] + (3,)
    uncertainty = (
        torch.zeros(shape, dtype=down2.dtype, device=device)
        if uncertainty_reduction is None else uncertainty_reduction
    )
    local = (
        torch.zeros(shape, dtype=down2.dtype, device=device)
        if local_gain is None else local_gain
    )
    if (
        not isinstance(uncertainty, torch.Tensor)
        or not isinstance(local, torch.Tensor)
        or uncertainty.shape != shape
        or local.shape != shape
        or uncertainty.device != device
        or local.device != device
    ):
        raise ValueError("tensor uncertainty/local gains must be [expert,unit,3]")

    long_states = states.long()
    bits = torch.tensor((1, 2, 4), dtype=torch.long, device=device)
    destination = torch.bitwise_or(long_states[..., None], bits)
    valid = torch.bitwise_and(long_states[..., None], bits) == 0
    source_hidden_index = torch.bitwise_right_shift(long_states, 1)
    destination_hidden_index = torch.bitwise_right_shift(destination, 1)
    source_hidden = torch.gather(
        hidden, 2, source_hidden_index[..., None],
    ).squeeze(-1)
    destination_hidden = torch.gather(hidden, 2, destination_hidden_index)
    source_coefficients = torch.where(
        torch.bitwise_and(long_states, 1).bool()[..., None], down4, down2,
    )
    destination_coefficients = torch.where(
        torch.bitwise_and(destination, 1).bool()[..., None],
        down4[:, :, None, :],
        down2[:, :, None, :],
    )
    delta = (
        destination_hidden[..., None] * destination_coefficients
        - source_hidden[..., None, None] * source_coefficients[:, :, None, :]
    ) * weights[:, None, None, None]
    reduction = -torch.einsum("eupr,r->eup", delta, adjoint)
    score = (reduction + uncertainty) / int(page_bytes)
    score = torch.where(valid, score, torch.full_like(score, -torch.inf))
    best_score, best_slot = torch.max(score, dim=2)
    flat_score = best_score.reshape(-1)
    available = int(torch.count_nonzero(torch.isfinite(flat_score)).item())
    count = min(batch, available)
    if count == 0:
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        empty_value = torch.empty(0, dtype=down2.dtype, device=device)
        return TensorTransitionBatch(
            empty_long, empty_long, empty_long, empty_long, empty_long,
            empty_value, empty_value, empty_value, empty_value,
        )
    selected_score, flat = torch.topk(flat_score, count, sorted=True)
    units = down2.shape[1]
    expert_id = torch.div(flat, units, rounding_mode="floor")
    unit_id = torch.remainder(flat, units)
    slot = best_slot.reshape(-1)[flat]
    source = long_states[expert_id, unit_id]
    selected_destination = destination[expert_id, unit_id, slot]
    return TensorTransitionBatch(
        expert=expert_id,
        unit=unit_id,
        projection_slot=slot,
        source_state=source,
        destination_state=selected_destination,
        route_risk_reduction=reduction[expert_id, unit_id, slot],
        uncertainty_reduction=uncertainty[expert_id, unit_id, slot],
        local_gain=local[expert_id, unit_id, slot],
        score_per_byte=selected_score,
    )


def apply_precision_transition(
    state_vectors: Sequence[Sequence[int] | np.ndarray],
    transition: PrecisionTransition,
) -> tuple[np.ndarray, ...]:
    """Apply one previously scored transition and reject stale actions."""

    result = tuple(np.asarray(value, np.int64).reshape(-1).copy() for value in state_vectors)
    expert = int(transition.expert)
    unit = int(transition.unit)
    if expert < 0 or expert >= len(result) or unit < 0 or unit >= result[expert].size:
        raise ValueError("transition target lies outside state vectors")
    if int(result[expert][unit]) != int(transition.source_state):
        raise ValueError("precision transition is stale")
    result[expert][unit] = int(transition.destination_state)
    return result


def apply_precision_transitions(
    state_vectors: Sequence[Sequence[int] | np.ndarray],
    transitions: Sequence[PrecisionTransition],
) -> tuple[np.ndarray, ...]:
    """Apply one scored batch whose transitions target distinct expert units."""

    result = tuple(np.asarray(value, np.int64).reshape(-1).copy() for value in state_vectors)
    seen = set()
    for transition in transitions:
        expert = int(transition.expert)
        unit = int(transition.unit)
        key = (expert, unit)
        if key in seen:
            raise ValueError("batched transitions must target distinct expert units")
        seen.add(key)
        if expert < 0 or expert >= len(result) or unit < 0 or unit >= result[expert].size:
            raise ValueError("transition target lies outside state vectors")
        if int(result[expert][unit]) != int(transition.source_state):
            raise ValueError("precision transition is stale")
        result[expert][unit] = int(transition.destination_state)
    return result


def apply_precision_transitions_torch(
    state_vectors: object,
    transitions: TensorTransitionBatch,
) -> object:
    """Apply a device-resident distinct-unit batch and reject stale states."""

    import torch

    if not isinstance(state_vectors, torch.Tensor):
        raise TypeError("tensor transition application requires a torch tensor")
    result = state_vectors.clone()
    if len(transitions) == 0:
        return result
    expert = transitions.expert.long()
    unit = transitions.unit.long()
    if (
        expert.device != result.device
        or unit.device != result.device
        or bool(torch.any(expert < 0).item())
        or bool(torch.any(expert >= result.shape[0]).item())
        or bool(torch.any(unit < 0).item())
        or bool(torch.any(unit >= result.shape[1]).item())
    ):
        raise ValueError("tensor transition target lies outside state vectors")
    packed = expert * result.shape[1] + unit
    if int(torch.unique(packed).numel()) != len(transitions):
        raise ValueError("tensor transitions must target distinct expert units")
    current = result[expert, unit]
    source = transitions.source_state.to(dtype=current.dtype)
    if bool(torch.any(current != source).item()):
        raise ValueError("tensor precision transition is stale")
    result[expert, unit] = transitions.destination_state.to(dtype=result.dtype)
    return result
