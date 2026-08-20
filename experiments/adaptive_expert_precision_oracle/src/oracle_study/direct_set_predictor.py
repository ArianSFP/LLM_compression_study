"""Compact coherent-unit predictor and exact set-utility distillation tools.

The predictor in this module consumes only deployable Q2-side features.  A
dense 512 by 512 correction Gram may be supplied to the *training losses* and
teacher evaluators, but is deliberately absent from :class:`forward`.  This
keeps the scientific boundary between a training-only H0 oracle and a
deployable candidate predictor explicit.

The systems interface is also explicit.  A predictor fetches a fixed-size
candidate set, after which two different rerankers can be audited:

* contained-target reranking is information-feasible after fetch, but exact
  interaction reranking remains compute-expensive and is not assumed to be a
  generally deployable cheap path;
* full-target restricted reranking uses effects of omitted corrections and is
  a teacher upper bound, not a deployable selector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .set_utility_selector import hard_forward_soft_backward_mask
from .unit_set_teacher import (
    exact_contained_target_fixed_greedy,
    exact_full_target_fixed_greedy,
    set_damage,
    set_gain,
)


HeadMode = Literal["single", "joint_nested"]
Reduction = Literal["none", "mean", "sum"]


def _finite_tensor(value: torch.Tensor, name: str) -> None:
    if not torch.is_floating_point(value):
        raise ValueError(f"{name} must be floating point")
    if not bool(torch.all(torch.isfinite(value)).item()):
        raise ValueError(f"{name} contains a non-finite value")


def _stable_descending_order(values: torch.Tensor) -> torch.Tensor:
    return torch.argsort(values, dim=-1, descending=True, stable=True)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError("weighted reduction expects one value per sample")
    if weights is None:
        return values.mean()
    weight = torch.as_tensor(weights, dtype=values.dtype, device=values.device)
    if weight.shape != values.shape or not bool(torch.all(torch.isfinite(weight)).item()):
        raise ValueError("sample weights must be finite with one value per sample")
    if bool(torch.any(weight < 0).item()) or float(weight.sum()) <= 0.0:
        raise ValueError("sample weights must be nonnegative with positive total")
    return torch.sum(values * weight) / torch.sum(weight)


def _reduce(values: torch.Tensor, reduction: Reduction, weights: torch.Tensor | None) -> torch.Tensor:
    if reduction == "none":
        if weights is not None:
            raise ValueError("sample weights require a mean or sum reduction")
        return values
    if reduction == "mean":
        return _weighted_mean(values, weights)
    if reduction == "sum":
        if weights is None:
            return values.sum()
        weight = torch.as_tensor(weights, dtype=values.dtype, device=values.device)
        if weight.shape != values.shape or bool(torch.any(weight < 0).item()):
            raise ValueError("sample weights must be nonnegative with one value per sample")
        return torch.sum(values * weight)
    raise ValueError(f"unsupported reduction: {reduction}")


def _broadcast_unit_component(
    value: torch.Tensor | None,
    *,
    batch: int,
    units: int,
    width: int,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor | None:
    if width == 0:
        if value is not None and value.numel():
            raise ValueError(f"{name} was supplied but its configured width is zero")
        return None
    if value is None:
        raise ValueError(f"{name} is required by this predictor configuration")
    component = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if component.ndim == 2 and component.shape == (units, width):
        component = component.unsqueeze(0).expand(batch, -1, -1)
    if component.shape != (batch, units, width):
        raise ValueError(f"{name} must have shape [batch, unit, {width}] or [unit, {width}]")
    _finite_tensor(component, name)
    return component


@dataclass(frozen=True)
class PredictorOutput:
    """Candidate and application logits.

    In ``single`` mode both fields reference the same ordered score.  In
    ``joint_nested`` mode inference constructs a candidate set that contains
    every selected application unit, irrespective of raw-head disagreement.
    """

    candidate_logits: torch.Tensor
    apply_logits: torch.Tensor
    head_mode: HeadMode

    def __post_init__(self) -> None:
        if self.candidate_logits.shape != self.apply_logits.shape:
            raise ValueError("candidate and apply logits must have equal shapes")
        if self.candidate_logits.ndim != 2:
            raise ValueError("predictor logits must have shape [batch, unit]")
        if self.head_mode not in ("single", "joint_nested"):
            raise ValueError(f"unsupported head mode: {self.head_mode}")

        _finite_tensor(self.candidate_logits, "candidate_logits")
        _finite_tensor(self.apply_logits, "apply_logits")

@dataclass(frozen=True)
class PredictorAccounting:
    trainable_parameters: int
    parameter_metadata_bytes: int
    static_abc_metadata_bytes: int
    external_feature_metadata_bytes: int
    total_metadata_bytes: int
    linear_macs_per_invocation: int
    context_reduction_additions: int
    unit_count: int


class CompactCoherentUnitPredictor(nn.Module):
    """A shared per-unit MLP conditioned by a compact invocation context.

    The context is the mean of the deployable per-unit features concatenated
    with an optional global Q2-side feature/sketch.  Parameter count therefore
    does not scale with the number of experts or units.  Static A/B/C score
    metadata and PQ-predicted gate/up responses can be appended without any
    teacher-side input entering inference.
    """

    def __init__(
        self,
        q2_unit_dim: int,
        *,
        global_dim: int = 0,
        pq_delta_dim: int = 0,
        abc_dim: int = 0,
        hidden_dim: int = 24,
        unit_count: int = 512,
        head_mode: HeadMode = "single",
    ) -> None:
        super().__init__()
        dimensions = (q2_unit_dim, global_dim, pq_delta_dim, abc_dim)
        if any(int(value) != value or value < 0 for value in dimensions):
            raise ValueError("feature dimensions must be nonnegative integers")
        if q2_unit_dim < 1 or hidden_dim < 1 or unit_count < 2:
            raise ValueError("q2_unit_dim, hidden_dim and unit_count must be positive")
        if head_mode not in ("single", "joint_nested"):
            raise ValueError(f"unsupported head mode: {head_mode}")
        self.q2_unit_dim = int(q2_unit_dim)
        self.global_dim = int(global_dim)
        self.pq_delta_dim = int(pq_delta_dim)
        self.abc_dim = int(abc_dim)
        self.hidden_dim = int(hidden_dim)
        self.unit_count = int(unit_count)
        self.head_mode: HeadMode = head_mode
        self.unit_input_dim = self.q2_unit_dim + self.pq_delta_dim + self.abc_dim

        self.local_projection = nn.Linear(self.unit_input_dim, self.hidden_dim)
        self.context_projection = nn.Linear(
            self.unit_input_dim + self.global_dim, self.hidden_dim,
        )
        self.hidden_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        if head_mode == "single":
            self.ordered_head = nn.Linear(self.hidden_dim, 1)
        else:
            self.candidate_head = nn.Linear(self.hidden_dim, 1)
            self.apply_head = nn.Linear(self.hidden_dim, 1)

    def _features(
        self,
        q2_unit_features: torch.Tensor,
        *,
        pq_delta_response: torch.Tensor | None,
        abc: torch.Tensor | None,
    ) -> torch.Tensor:
        q2 = q2_unit_features
        if q2.ndim != 3 or q2.shape[1:] != (self.unit_count, self.q2_unit_dim):
            raise ValueError(
                f"q2_unit_features must have shape [batch, {self.unit_count}, {self.q2_unit_dim}]",
            )
        _finite_tensor(q2, "q2_unit_features")
        pieces = [q2]
        pq = _broadcast_unit_component(
            pq_delta_response, batch=q2.shape[0], units=self.unit_count,
            width=self.pq_delta_dim, reference=q2, name="pq_delta_response",
        )
        static = _broadcast_unit_component(
            abc, batch=q2.shape[0], units=self.unit_count,
            width=self.abc_dim, reference=q2, name="abc",
        )
        if pq is not None:
            pieces.append(pq)
        if static is not None:
            pieces.append(static)
        return torch.cat(pieces, dim=-1)

    def forward(
        self,
        q2_unit_features: torch.Tensor,
        *,
        global_features: torch.Tensor | None = None,
        pq_delta_response: torch.Tensor | None = None,
        abc: torch.Tensor | None = None,
    ) -> PredictorOutput:
        unit = self._features(
            q2_unit_features, pq_delta_response=pq_delta_response, abc=abc,
        )
        batch = unit.shape[0]
        if self.global_dim:
            if global_features is None:
                raise ValueError("global_features are required by this predictor configuration")
            global_value = torch.as_tensor(
                global_features, dtype=unit.dtype, device=unit.device,
            )
            if global_value.shape != (batch, self.global_dim):
                raise ValueError(f"global_features must have shape [batch, {self.global_dim}]")
            _finite_tensor(global_value, "global_features")
            context_input = torch.cat((unit.mean(dim=1), global_value), dim=-1)
        else:
            if global_features is not None and global_features.numel():
                raise ValueError("global_features were supplied but global_dim is zero")
            context_input = unit.mean(dim=1)
        hidden = F.silu(
            self.local_projection(unit) + self.context_projection(context_input).unsqueeze(1),
        )
        hidden = F.silu(self.hidden_projection(hidden))
        if self.head_mode == "single":
            ordered = self.ordered_head(hidden).squeeze(-1)
            return PredictorOutput(ordered, ordered, "single")
        return PredictorOutput(
            self.candidate_head(hidden).squeeze(-1),
            self.apply_head(hidden).squeeze(-1),
            "joint_nested",
        )

    def accounting(
        self,
        *,
        parameter_bits: int = 16,
        abc_bits: int = 16,
        external_feature_metadata_bytes: int = 0,
    ) -> PredictorAccounting:
        """Return exact stored-parameter and linear-MAC accounting.

        Biases are charged as parameters but do not add MACs.  ABC is charged
        as expert-specific static metadata.  PQ/codebook bytes are supplied by
        their encoder so they cannot be accidentally inferred from a dense
        logical tensor.
        """
        if parameter_bits < 1 or abc_bits < 1 or external_feature_metadata_bytes < 0:
            raise ValueError("metadata precisions/bytes must be positive")
        parameters = sum(int(value.numel()) for value in self.parameters())
        parameter_bytes = (parameters * int(parameter_bits) + 7) // 8
        abc_bytes = (self.unit_count * self.abc_dim * int(abc_bits) + 7) // 8
        heads = 1 if self.head_mode == "single" else 2
        macs = (
            self.unit_count * self.unit_input_dim * self.hidden_dim
            + (self.unit_input_dim + self.global_dim) * self.hidden_dim
            + self.unit_count * self.hidden_dim * self.hidden_dim
            + heads * self.unit_count * self.hidden_dim
        )
        reduction_additions = self.unit_input_dim * (self.unit_count - 1)
        external = int(external_feature_metadata_bytes)
        return PredictorAccounting(
            trainable_parameters=parameters,
            parameter_metadata_bytes=parameter_bytes,
            static_abc_metadata_bytes=abc_bytes,
            external_feature_metadata_bytes=external,
            total_metadata_bytes=parameter_bytes + abc_bytes + external,
            linear_macs_per_invocation=int(macs),
            context_reduction_additions=int(reduction_additions),
            unit_count=self.unit_count,
        )


@dataclass(frozen=True)
class NestedMasks:
    candidate: torch.Tensor
    apply: torch.Tensor


def nested_candidate_apply_masks(
    output: PredictorOutput,
    *,
    candidate_count: int = 256,
    apply_count: int = 192,
) -> NestedMasks:
    """Build deterministic fixed-cardinality masks with apply subset candidate."""
    units = output.apply_logits.shape[-1]
    if not (0 <= apply_count <= candidate_count <= units):
        raise ValueError("counts must satisfy 0 <= apply <= candidate <= units")
    apply_order = _stable_descending_order(output.apply_logits)
    candidate_order = _stable_descending_order(output.candidate_logits)
    apply_mask = torch.zeros_like(output.apply_logits, dtype=torch.bool)
    if apply_count:
        apply_mask.scatter_(1, apply_order[:, :apply_count], True)
    candidate_mask = apply_mask.clone()
    needed = candidate_count - apply_count
    if needed:
        for row in range(candidate_mask.shape[0]):
            eligible = candidate_order[row][~apply_mask[row, candidate_order[row]]]
            candidate_mask[row, eligible[:needed]] = True
    return NestedMasks(candidate=candidate_mask, apply=apply_mask)


def exclusion_regret_unit_weights(regret: torch.Tensor | None) -> torch.Tensor | None:
    """Convert nonnegative exclusion regret to mean-one importance weights."""
    if regret is None:
        return None
    value = torch.as_tensor(regret)
    _finite_tensor(value, "exclusion_regret")
    if value.ndim != 2 or bool(torch.any(value < 0).item()):
        raise ValueError("exclusion regret must be nonnegative [batch, unit]")
    positive = value > 0
    count = positive.sum(dim=-1, keepdim=True)
    mean = value.sum(dim=-1, keepdim=True) / count.clamp_min(1)
    scaled = torch.where(positive, value / mean.clamp_min(torch.finfo(value.dtype).tiny), torch.zeros_like(value))
    weights = 1.0 + scaled
    return weights / weights.mean(dim=-1, keepdim=True)


def teacher_marginal_regression_loss(
    logits: torch.Tensor,
    teacher_marginal: torch.Tensor,
    *,
    exclusion_regret: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Huber regression to per-invocation standardized teacher marginals."""
    if logits.shape != teacher_marginal.shape or logits.ndim != 2:
        raise ValueError("logits and teacher marginals must be equal [batch, unit] tensors")
    _finite_tensor(logits, "logits")
    teacher = torch.as_tensor(teacher_marginal, dtype=logits.dtype, device=logits.device)
    _finite_tensor(teacher, "teacher_marginal")
    target = (teacher - teacher.mean(dim=-1, keepdim=True)) / teacher.std(
        dim=-1, keepdim=True, unbiased=False,
    ).clamp_min(torch.finfo(logits.dtype).eps)
    element = F.smooth_l1_loss(logits, target, reduction="none")
    unit_weight = exclusion_regret_unit_weights(
        None if exclusion_regret is None else torch.as_tensor(
            exclusion_regret, dtype=logits.dtype, device=logits.device,
        ),
    )
    if unit_weight is not None:
        element = element * unit_weight
    return _reduce(element.mean(dim=-1), reduction, sample_weight)


def listwise_marginal_kl_loss(
    logits: torch.Tensor,
    teacher_marginal: torch.Tensor,
    *,
    temperature: float = 1.0,
    exclusion_regret: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Listwise KL, optionally tilting teacher mass by exclusion regret."""
    if logits.shape != teacher_marginal.shape or logits.ndim != 2:
        raise ValueError("logits and teacher marginals must be equal [batch, unit] tensors")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    teacher = torch.as_tensor(teacher_marginal, dtype=logits.dtype, device=logits.device)
    _finite_tensor(logits, "logits")
    _finite_tensor(teacher, "teacher_marginal")
    teacher_score = teacher / float(temperature)
    unit_weight = exclusion_regret_unit_weights(
        None if exclusion_regret is None else torch.as_tensor(
            exclusion_regret, dtype=logits.dtype, device=logits.device,
        ),
    )
    if unit_weight is not None:
        teacher_score = teacher_score + torch.log(unit_weight.clamp_min(torch.finfo(logits.dtype).tiny))
    teacher_probability = torch.softmax(teacher_score, dim=-1)
    teacher_log_probability = torch.log(teacher_probability.clamp_min(torch.finfo(logits.dtype).tiny))
    predicted_log_probability = torch.log_softmax(logits / float(temperature), dim=-1)
    per_sample = torch.sum(
        teacher_probability * (teacher_log_probability - predicted_log_probability), dim=-1,
    )
    return _reduce(per_sample, reduction, sample_weight)


def _validate_teacher_order(order: torch.Tensor, units: int) -> None:
    if order.ndim != 2 or order.shape[1] != units or order.dtype not in (
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
    ):
        raise ValueError("teacher order must be an integer [batch, unit] permutation")
    expected = torch.arange(units, device=order.device).expand_as(order)
    if not bool(torch.all(torch.sort(order.to(torch.int64), dim=-1).values == expected).item()):
        raise ValueError("each teacher order row must be a permutation")


def boundary_pair_ranking_loss(
    logits: torch.Tensor,
    teacher_order: torch.Tensor,
    *,
    boundaries: Sequence[int] = (192, 256),
    boundary_width: int = 16,
    exclusion_regret: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Pairwise logistic loss for units straddling streaming boundaries."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, unit]")
    units = logits.shape[1]
    order = torch.as_tensor(teacher_order, device=logits.device)
    if order.shape[0] != logits.shape[0]:
        raise ValueError("teacher order batch does not match logits")
    _validate_teacher_order(order, units)
    if boundary_width < 1:
        raise ValueError("boundary_width must be positive")
    counts = tuple(int(value) for value in boundaries)
    if not counts or any(value <= 0 or value >= units for value in counts):
        raise ValueError("boundaries must lie strictly inside the unit range")
    unit_weight = exclusion_regret_unit_weights(
        None if exclusion_regret is None else torch.as_tensor(
            exclusion_regret, dtype=logits.dtype, device=logits.device,
        ),
    )
    losses = []
    for boundary in counts:
        width = min(int(boundary_width), boundary, units - boundary)
        positive_ids = order[:, boundary - width:boundary]
        negative_ids = order[:, boundary:boundary + width]
        positive = torch.gather(logits, 1, positive_ids)
        negative = torch.gather(logits, 1, negative_ids)
        pair = F.softplus(-(positive.unsqueeze(2) - negative.unsqueeze(1)))
        if unit_weight is not None:
            positive_weight = torch.gather(unit_weight, 1, positive_ids).unsqueeze(2)
            pair = pair * positive_weight
            per_sample = pair.sum(dim=(1, 2)) / (positive_weight.sum(dim=(1, 2)) * width)
        else:
            per_sample = pair.mean(dim=(1, 2))
        losses.append(per_sample)
    combined = torch.stack(losses, dim=0).mean(dim=0)
    return _reduce(combined, reduction, sample_weight)


@dataclass(frozen=True)
class CandidateCoverageLossResult:
    loss: torch.Tensor
    candidate_mask: torch.Tensor
    teacher_apply_mask: torch.Tensor
    per_sample_missed_fraction: torch.Tensor


def candidate_coverage_loss(
    candidate_logits: torch.Tensor,
    teacher_order: torch.Tensor,
    *,
    apply_logits: torch.Tensor | None = None,
    candidate_count: int = 256,
    teacher_apply_count: int = 192,
    temperature: float = 1.0,
    exclusion_regret: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
) -> CandidateCoverageLossResult:
    """Penalize fetched candidates that omit teacher top-apply units.

    The forward mask contains exactly ``candidate_count`` units. The target is
    coverage of the teacher top ``teacher_apply_count`` support, not the
    counterfactual objective obtained by applying every fetched packet.
    Straight-through cardinality gradients make the discrete coverage loss
    trainable while keeping its reported forward value physically exact.
    """
    if candidate_logits.ndim != 2:
        raise ValueError("candidate logits must have shape [batch, unit]")
    batch, units = candidate_logits.shape
    apply_score = candidate_logits if apply_logits is None else apply_logits
    if apply_score.shape != candidate_logits.shape:
        raise ValueError("apply logits shape does not match candidate logits")
    _finite_tensor(candidate_logits, "candidate_logits")
    _finite_tensor(apply_score, "apply_logits")
    if not (0 < teacher_apply_count <= candidate_count < units):
        raise ValueError("counts must satisfy 0 < teacher apply <= candidate < units")
    order = torch.as_tensor(teacher_order, device=candidate_logits.device)
    if order.shape[0] != batch:
        raise ValueError("teacher order batch does not match candidate logits")
    _validate_teacher_order(order, units)
    if apply_logits is None:
        candidate_mask = hard_forward_soft_backward_mask(
            candidate_logits, int(candidate_count), temperature=temperature,
        )
    else:
        nested = nested_candidate_apply_masks(
            PredictorOutput(
                candidate_logits=candidate_logits,
                apply_logits=apply_score,
                head_mode="joint_nested",
            ),
            candidate_count=int(candidate_count),
            apply_count=int(teacher_apply_count),
        )
        hard_candidate = nested.candidate.to(dtype=candidate_logits.dtype)
        apply_surrogate = hard_forward_soft_backward_mask(
            apply_score, int(teacher_apply_count), temperature=temperature,
        )
        repair_count = int(candidate_count) - int(teacher_apply_count)
        repair_rows = []
        for row in range(batch):
            eligible = torch.nonzero(~nested.apply[row], as_tuple=False).flatten()
            local = hard_forward_soft_backward_mask(
                candidate_logits[row, eligible].unsqueeze(0),
                repair_count,
                temperature=temperature,
            ).squeeze(0)
            repair_rows.append(
                torch.zeros_like(candidate_logits[row]).scatter(0, eligible, local)
            )
        repair_surrogate = torch.stack(repair_rows, dim=0)
        surrogate = apply_surrogate + repair_surrogate
        candidate_mask = hard_candidate + surrogate - surrogate.detach()
    teacher_mask = torch.zeros_like(candidate_logits)
    teacher_mask.scatter_(1, order[:, :int(teacher_apply_count)], 1.0)
    importance = teacher_mask
    regret_weight = exclusion_regret_unit_weights(
        None if exclusion_regret is None else torch.as_tensor(
            exclusion_regret, dtype=candidate_logits.dtype, device=candidate_logits.device,
        ),
    )
    if regret_weight is not None:
        if regret_weight.shape != candidate_logits.shape:
            raise ValueError("exclusion regret shape does not match candidate logits")
        importance = importance * regret_weight
    denominator = importance.sum(dim=-1).clamp_min(torch.finfo(candidate_logits.dtype).eps)
    retained = (candidate_mask * importance).sum(dim=-1) / denominator
    missed = 1.0 - retained
    return CandidateCoverageLossResult(
        loss=_weighted_mean(missed, sample_weight),
        candidate_mask=candidate_mask,
        teacher_apply_mask=teacher_mask,
        per_sample_missed_fraction=missed,
    )


def exact_gram_set_damage(
    teacher_gram: torch.Tensor,
    mask: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Compute ``(1-m).T K (1-m)`` without correction vectors."""
    gram = torch.as_tensor(teacher_gram, dtype=mask.dtype, device=mask.device)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if gram.ndim == 2:
        gram = gram.unsqueeze(0)
    if mask.ndim != 2 or gram.shape != (mask.shape[0], mask.shape[1], mask.shape[1]):
        raise ValueError("Gram/mask shapes must be [batch, unit, unit] and [batch, unit]")
    _finite_tensor(mask, "mask")
    _finite_tensor(gram, "teacher_gram")
    if not bool(torch.allclose(gram, gram.transpose(-1, -2), rtol=1e-4, atol=1e-6)):
        raise ValueError("teacher Gram must be symmetric")
    residual = 1.0 - mask
    damage = torch.einsum("bi,bij,bj->b", residual, gram, residual)
    return _reduce(damage, reduction, sample_weight)


@dataclass(frozen=True)
class HardSetLossResult:
    loss: torch.Tensor
    mask: torch.Tensor
    per_sample_damage: torch.Tensor


def hard_fixed_cardinality_gram_set_loss(
    logits: torch.Tensor,
    teacher_gram: torch.Tensor,
    count: int,
    *,
    temperature: float = 1.0,
    sample_weight: torch.Tensor | None = None,
    normalize_by_full_damage: bool = False,
) -> HardSetLossResult:
    """Hard-forward fixed-cardinality exact set loss with soft gradients."""
    mask = hard_forward_soft_backward_mask(logits, int(count), temperature=temperature)
    damage = exact_gram_set_damage(teacher_gram, mask, reduction="none")
    if normalize_by_full_damage:
        gram = torch.as_tensor(teacher_gram, dtype=logits.dtype, device=logits.device)
        ones = torch.ones_like(logits)
        baseline = torch.einsum("bi,bij,bj->b", ones, gram, ones).abs()
        damage = damage / baseline.clamp_min(torch.finfo(logits.dtype).eps)
    return HardSetLossResult(
        loss=_weighted_mean(damage, sample_weight),
        mask=mask,
        per_sample_damage=damage,
    )


def request_balanced_sample_weights(
    request_ids: Sequence[str],
    base_sample_weight: torch.Tensor | np.ndarray | None = None,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Give every request equal total weight, then normalize sample mean to one."""
    identifiers = tuple(str(value) for value in request_ids)
    if not identifiers:
        raise ValueError("at least one request is required")
    counts: dict[str, int] = {}
    for identifier in identifiers:
        counts[identifier] = counts.get(identifier, 0) + 1
    if base_sample_weight is None:
        balanced = torch.tensor(
            [1.0 / counts[identifier] for identifier in identifiers], dtype=dtype, device=device,
        )
    else:
        base = torch.as_tensor(base_sample_weight, dtype=dtype, device=device)
        if base.shape != (len(identifiers),) or not bool(torch.all(torch.isfinite(base)).item()):
            raise ValueError("base sample weights must be finite with one value per sample")
        if bool(torch.any(base < 0).item()):
            raise ValueError("base sample weights must be nonnegative")
        group_total = {
            identifier: sum(float(base[index]) for index, value in enumerate(identifiers) if value == identifier)
            for identifier in counts
        }
        if any(total <= 0.0 for total in group_total.values()):
            raise ValueError("every request must have positive total base sample weight")
        denominator = torch.tensor(
            [group_total[identifier] for identifier in identifiers], dtype=dtype, device=device,
        )
        balanced = base / denominator
    if float(balanced.sum()) <= 0:
        raise ValueError("combined sample weights must have positive total")
    return balanced * (len(balanced) / balanced.sum())


def validate_request_split_isolation(
    request_ids: Sequence[str], split_labels: Sequence[str],
) -> None:
    if len(request_ids) != len(split_labels) or not request_ids:
        raise ValueError("request IDs and split labels must have equal nonzero length")
    assignments: dict[str, str] = {}
    for request, split in zip(request_ids, split_labels, strict=True):
        identifier, label = str(request), str(split)
        if not label:
            raise ValueError("split labels cannot be empty")
        previous = assignments.setdefault(identifier, label)
        if previous != label:
            raise ValueError(f"request {identifier!r} appears in both {previous!r} and {label!r}")


@dataclass(frozen=True)
class DirectSetTrainingData:
    q2_unit_features: torch.Tensor
    teacher_marginal: torch.Tensor
    teacher_gram: torch.Tensor
    request_ids: tuple[str, ...]
    split_labels: tuple[str, ...]
    teacher_order: torch.Tensor | None = None
    global_features: torch.Tensor | None = None
    pq_delta_response: torch.Tensor | None = None
    abc: torch.Tensor | None = None
    exclusion_regret: torch.Tensor | None = None
    sample_weight: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.q2_unit_features.ndim != 3:
            raise ValueError("q2_unit_features must have shape [sample, unit, feature]")
        samples, units = self.q2_unit_features.shape[:2]
        if self.teacher_marginal.shape != (samples, units):
            raise ValueError("teacher_marginal shape does not match features")
        if self.teacher_gram.shape != (samples, units, units):
            raise ValueError("teacher_gram shape does not match features")
        if len(self.request_ids) != samples or len(self.split_labels) != samples:
            raise ValueError("request/split labels must match the sample count")
        validate_request_split_isolation(self.request_ids, self.split_labels)
        if self.teacher_order is not None:
            _validate_teacher_order(self.teacher_order, units)
            if self.teacher_order.shape[0] != samples:
                raise ValueError("teacher order sample count does not match features")
        for name in ("global_features", "pq_delta_response", "abc", "exclusion_regret"):
            value = getattr(self, name)
            if value is not None and value.shape[0] not in (samples, units if name == "abc" and value.ndim == 2 else -1):
                raise ValueError(f"{name} has an incompatible sample dimension")
        if self.sample_weight is not None and self.sample_weight.shape != (samples,):
            raise ValueError("sample_weight must have shape [sample]")

    @property
    def samples(self) -> int:
        return int(self.q2_unit_features.shape[0])

    @property
    def units(self) -> int:
        return int(self.q2_unit_features.shape[1])


@dataclass(frozen=True)
class DirectSetTrainingConfig:
    epochs: int = 100
    learning_rate: float = 3e-3
    weight_decay: float = 0.0
    apply_count: int = 192
    candidate_count: int = 256
    boundary_width: int = 16
    temperature: float = 1.0
    marginal_weight: float = 0.2
    listwise_weight: float = 1.0
    boundary_weight: float = 1.0
    set_weight: float = 1.0
    candidate_set_weight: float = 0.25
    seed: int = 20260820
    candidate_coverage_weight: float = 0.0

@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    marginal: torch.Tensor
    listwise: torch.Tensor
    boundary: torch.Tensor
    apply_set: torch.Tensor
    candidate_set: torch.Tensor
    candidate_coverage: torch.Tensor


def _resolved_teacher_order(data: DirectSetTrainingData, indices: torch.Tensor) -> torch.Tensor:
    if data.teacher_order is not None:
        return data.teacher_order[indices]
    return _stable_descending_order(data.teacher_marginal[indices])


def _model_inputs(data: DirectSetTrainingData, indices: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor | None]:
    def selected(value: torch.Tensor | None, *, static_allowed: bool = False) -> torch.Tensor | None:
        if value is None:
            return None
        if static_allowed and value.ndim == 2 and value.shape[0] == data.units:
            return value.to(device)
        return value[indices].to(device)
    return {
        "q2_unit_features": data.q2_unit_features[indices].to(device),
        "global_features": selected(data.global_features),
        "pq_delta_response": selected(data.pq_delta_response),
        "abc": selected(data.abc, static_allowed=True),
    }


def direct_set_distillation_loss(
    output: PredictorOutput,
    teacher_marginal: torch.Tensor,
    teacher_order: torch.Tensor,
    teacher_gram: torch.Tensor,
    config: DirectSetTrainingConfig,
    *,
    exclusion_regret: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
) -> LossBreakdown:
    units = output.apply_logits.shape[1]
    if not (0 < config.apply_count < config.candidate_count < units):
        raise ValueError("training counts must satisfy 0 < apply < candidate < units")
    marginal = teacher_marginal_regression_loss(
        output.apply_logits, teacher_marginal,
        exclusion_regret=exclusion_regret, sample_weight=sample_weight,
    )
    apply_listwise = listwise_marginal_kl_loss(
        output.apply_logits, teacher_marginal, temperature=config.temperature,
        exclusion_regret=exclusion_regret, sample_weight=sample_weight,
    )
    candidate_listwise = listwise_marginal_kl_loss(
        output.candidate_logits, teacher_marginal, temperature=config.temperature,
        exclusion_regret=exclusion_regret, sample_weight=sample_weight,
    )
    listwise = 0.5 * (apply_listwise + candidate_listwise)
    apply_boundary = boundary_pair_ranking_loss(
        output.apply_logits, teacher_order, boundaries=(config.apply_count,),
        boundary_width=config.boundary_width, exclusion_regret=exclusion_regret,
        sample_weight=sample_weight,
    )
    candidate_boundary = boundary_pair_ranking_loss(
        output.candidate_logits, teacher_order, boundaries=(config.candidate_count,),
        boundary_width=config.boundary_width, exclusion_regret=exclusion_regret,
        sample_weight=sample_weight,
    )
    boundary = 0.5 * (apply_boundary + candidate_boundary)
    apply_set = hard_fixed_cardinality_gram_set_loss(
        output.apply_logits, teacher_gram, config.apply_count,
        temperature=config.temperature, sample_weight=sample_weight,
        normalize_by_full_damage=True,
    ).loss
    candidate_set = hard_fixed_cardinality_gram_set_loss(
        output.candidate_logits, teacher_gram, config.candidate_count,
        temperature=config.temperature, sample_weight=sample_weight,
        normalize_by_full_damage=True,
    ).loss
    coverage = candidate_coverage_loss(
        output.candidate_logits, teacher_order,
        apply_logits=output.apply_logits if output.head_mode == "joint_nested" else None,
        candidate_count=config.candidate_count,
        teacher_apply_count=config.apply_count,
        temperature=config.temperature,
        exclusion_regret=exclusion_regret,
        sample_weight=sample_weight,
    ).loss
    candidate_set_weight = config.candidate_set_weight if output.head_mode == "single" else 0.0
    total = (
        config.marginal_weight * marginal
        + config.listwise_weight * listwise
        + config.boundary_weight * boundary
        + config.set_weight * apply_set
        + candidate_set_weight * candidate_set
        + config.candidate_coverage_weight * coverage
    )
    return LossBreakdown(
        total, marginal, listwise, boundary, apply_set, candidate_set, coverage,
    )


@dataclass(frozen=True)
class TrainingEpoch:
    epoch: int
    train_loss: float
    validation_loss: float | None


@dataclass(frozen=True)
class DirectSetTrainingResult:
    history: tuple[TrainingEpoch, ...]
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]

    @property
    def initial_train_loss(self) -> float:
        return self.history[0].train_loss

    @property
    def final_train_loss(self) -> float:
        return self.history[-1].train_loss


def train_direct_set_predictor(
    model: CompactCoherentUnitPredictor,
    data: DirectSetTrainingData,
    config: DirectSetTrainingConfig,
    *,
    train_split: str = "train",
    validation_split: str = "validation",
) -> DirectSetTrainingResult:
    """Deterministic full-batch fit with request-isolated split selection."""
    if config.epochs < 1 or config.learning_rate <= 0 or config.weight_decay < 0:
        raise ValueError("training epochs/rates are invalid")
    if data.units != model.unit_count:
        raise ValueError("training data and predictor unit counts differ")
    train_indices = tuple(i for i, value in enumerate(data.split_labels) if value == train_split)
    validation_indices = tuple(i for i, value in enumerate(data.split_labels) if value == validation_split)
    if not train_indices:
        raise ValueError(f"no samples belong to training split {train_split!r}")
    # Isolation is rechecked at the training boundary so callers cannot bypass
    # it by constructing an object through nonstandard serialization.
    validate_request_split_isolation(data.request_ids, data.split_labels)

    device = next(model.parameters()).device
    train_index = torch.tensor(train_indices, dtype=torch.int64)
    validation_index = torch.tensor(validation_indices, dtype=torch.int64)
    train_request = [data.request_ids[index] for index in train_indices]
    base_weight = None if data.sample_weight is None else data.sample_weight[train_index]
    train_weight = request_balanced_sample_weights(
        train_request, base_weight, dtype=model.local_projection.weight.dtype, device=device,
    )

    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.manual_seed(int(config.seed))
    torch.use_deterministic_algorithms(True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config.learning_rate), weight_decay=float(config.weight_decay),
    )

    def loss_for(indices: torch.Tensor, weights: torch.Tensor | None) -> LossBreakdown:
        inputs = _model_inputs(data, indices, device)
        output = model(**inputs)  # type: ignore[arg-type]
        marginal = data.teacher_marginal[indices].to(device=device, dtype=output.apply_logits.dtype)
        gram = data.teacher_gram[indices].to(device=device, dtype=output.apply_logits.dtype)
        order = _resolved_teacher_order(data, indices).to(device)
        regret = None if data.exclusion_regret is None else data.exclusion_regret[indices].to(
            device=device, dtype=output.apply_logits.dtype,
        )
        return direct_set_distillation_loss(
            output, marginal, order, gram, config,
            exclusion_regret=regret, sample_weight=weights,
        )

    history: list[TrainingEpoch] = []
    try:
        model.train()
        with torch.no_grad():
            initial = float(loss_for(train_index, train_weight).total)
        history.append(TrainingEpoch(0, initial, None))
        for epoch in range(1, int(config.epochs) + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(train_index, train_weight).total
            loss.backward()
            if not all(
                parameter.grad is None or bool(torch.all(torch.isfinite(parameter.grad)).item())
                for parameter in model.parameters()
            ):
                raise FloatingPointError("non-finite predictor gradient")
            optimizer.step()
            model.eval()
            with torch.no_grad():
                train_loss = float(loss_for(train_index, train_weight).total)
                validation_loss: float | None = None
                if validation_indices:
                    validation_request = [data.request_ids[index] for index in validation_indices]
                    validation_base = None if data.sample_weight is None else data.sample_weight[validation_index]
                    validation_weight = request_balanced_sample_weights(
                        validation_request, validation_base,
                        dtype=model.local_projection.weight.dtype, device=device,
                    )
                    validation_loss = float(loss_for(validation_index, validation_weight).total)
            history.append(TrainingEpoch(epoch, train_loss, validation_loss))
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
    return DirectSetTrainingResult(tuple(history), train_indices, validation_indices)


@dataclass(frozen=True)
class SelectionMetrics:
    selected_count: int
    support_recall: float
    full_target_gain: float
    correction_recovery: float
    teacher_gain_retained: float


@dataclass(frozen=True)
class CandidateRerankEvaluation:
    predicted_candidate_ids: np.ndarray
    predicted_apply_ids: np.ndarray
    contained_target_rerank_ids: np.ndarray
    full_target_restricted_ids: np.ndarray
    teacher_ids: np.ndarray
    candidate_teacher_support_recall: float
    predicted_apply: SelectionMetrics
    contained_target_rerank: SelectionMetrics
    full_target_restricted: SelectionMetrics


def _selection_metrics(
    gram: np.ndarray, selected: np.ndarray, teacher: np.ndarray,
) -> SelectionMetrics:
    units = gram.shape[0]
    mask = np.zeros(units, dtype=bool)
    mask[np.asarray(selected, dtype=np.int64)] = True
    teacher_mask = np.zeros(units, dtype=bool)
    teacher_mask[np.asarray(teacher, dtype=np.int64)] = True
    gain = float(set_gain(gram, mask))
    teacher_gain = float(set_gain(gram, teacher_mask))
    base = float(set_damage(gram, np.zeros(units, dtype=bool)))
    recovery = gain / base if abs(base) > 1e-12 else (1.0 if abs(gain) <= 1e-12 else float("nan"))
    retained = gain / teacher_gain if abs(teacher_gain) > 1e-12 else (
        1.0 if abs(gain - teacher_gain) <= 1e-12 else float("nan")
    )
    recall = float(np.sum(mask & teacher_mask) / max(len(teacher), 1))
    return SelectionMetrics(len(selected), recall, gain, recovery, retained)


def evaluate_candidate_reranking(
    output: PredictorOutput,
    teacher_gram: torch.Tensor | np.ndarray,
    *,
    candidate_count: int = 256,
    apply_count: int = 192,
) -> tuple[CandidateRerankEvaluation, ...]:
    """Audit prediction, information-feasible contained rerank, and teacher rerank.

    Exact contained interaction reranking may still be too compute-expensive
    for deployment.
    """
    masks = nested_candidate_apply_masks(
        output, candidate_count=candidate_count, apply_count=apply_count,
    )
    grams = np.asarray(torch.as_tensor(teacher_gram).detach().cpu(), dtype=np.float64)
    if grams.ndim == 2:
        grams = grams[None, ...]
    if grams.shape != (
        output.apply_logits.shape[0], output.apply_logits.shape[1], output.apply_logits.shape[1],
    ):
        raise ValueError("teacher Gram batch does not match predictor logits")
    candidate_masks = masks.candidate.detach().cpu().numpy()
    apply_masks = masks.apply.detach().cpu().numpy()
    evaluations = []
    for row, gram in enumerate(grams):
        candidate = np.flatnonzero(candidate_masks[row]).astype(np.int64)
        predicted = np.flatnonzero(apply_masks[row]).astype(np.int64)
        contained = exact_contained_target_fixed_greedy(
            gram, candidate, apply_count,
        ).order
        restricted = exact_full_target_fixed_greedy(
            gram, apply_count, candidates=candidate,
        ).order
        teacher = exact_full_target_fixed_greedy(gram, apply_count).order
        candidate_recall = float(np.intersect1d(candidate, teacher).size / max(apply_count, 1))
        evaluations.append(CandidateRerankEvaluation(
            predicted_candidate_ids=candidate,
            predicted_apply_ids=predicted,
            contained_target_rerank_ids=np.asarray(contained, dtype=np.int64),
            full_target_restricted_ids=np.asarray(restricted, dtype=np.int64),
            teacher_ids=np.asarray(teacher, dtype=np.int64),
            candidate_teacher_support_recall=candidate_recall,
            predicted_apply=_selection_metrics(gram, predicted, teacher),
            contained_target_rerank=_selection_metrics(gram, contained, teacher),
            full_target_restricted=_selection_metrics(gram, restricted, teacher),
        ))
    return tuple(evaluations)


__all__ = [
    "CandidateRerankEvaluation",
    "CompactCoherentUnitPredictor",
    "DirectSetTrainingConfig",
    "CandidateCoverageLossResult",
    "DirectSetTrainingData",
    "DirectSetTrainingResult",
    "HardSetLossResult",
    "LossBreakdown",
    "NestedMasks",
    "PredictorAccounting",
    "PredictorOutput",
    "SelectionMetrics",
    "boundary_pair_ranking_loss",
    "direct_set_distillation_loss",
    "candidate_coverage_loss",
    "evaluate_candidate_reranking",
    "exact_gram_set_damage",
    "exclusion_regret_unit_weights",
    "hard_fixed_cardinality_gram_set_loss",
    "listwise_marginal_kl_loss",
    "nested_candidate_apply_masks",
    "request_balanced_sample_weights",
    "teacher_marginal_regression_loss",
    "train_direct_set_predictor",
    "validate_request_split_isolation",
]
