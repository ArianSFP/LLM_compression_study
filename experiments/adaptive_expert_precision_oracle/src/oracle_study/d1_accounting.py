"""Transparent compute and byte accounting for the D1 studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence


EXPERT_WEIGHTS = 3_145_728
PAGE_BYTES = 512
EXPERTS_PER_GROUP = 8
UNITS = 512
HIDDEN_SIZE = 2048
EXPERTS_PER_LAYER = 256
PR13_FACTOR_BYTES = 6_148
PR13_ABC_BYTES = 3_084
PR13_PRIMARY_SELECTOR_MACS = 51_343_360


@dataclass(frozen=True)
class ControllerAccounting:
    rank: int
    candidate_logits: int
    refreshes: int
    direct_candidate_macs: int
    one_adjoint_macs: int
    response_synthesis_macs: int
    direct_total_macs: int
    one_adjoint_total_macs: int
    pr13_primary_selector_macs: int
    direct_fraction_of_pr13: float
    one_adjoint_fraction_of_pr13: float
    current_metadata_bpw: float
    joint_metadata_bpw: float
    joint_uncertainty_metadata_bpw: float
    current_model_metadata_mib: float
    joint_model_metadata_mib: float
    joint_uncertainty_model_metadata_mib: float
    current_group_metadata_kib: float
    joint_group_metadata_kib: float
    joint_uncertainty_group_metadata_kib: float
    matched_pages: dict[str, int]
    matched_pages_with_uncertainty: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ProvisionalPassAccounting:
    context_tokens: int
    average_macs: float
    all_d1_macs: float
    average_packed_megabytes: float
    average_bf16_megabytes: float
    fraction_of_normal_layer: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def metadata_bpw(
    bytes_per_expert: float,
    *,
    expert_weights: int = EXPERT_WEIGHTS,
) -> float:
    if bytes_per_expert < 0.0 or int(expert_weights) <= 0:
        raise ValueError("metadata byte/weight counts are invalid")
    return 8.0 * float(bytes_per_expert) / int(expert_weights)


def matched_integral_pages(
    total_bpw: float,
    bytes_per_expert: float,
    *,
    expert_weights: int = EXPERT_WEIGHTS,
    page_bytes: int = PAGE_BYTES,
) -> int:
    """Maximum whole correction pages after charging all selector metadata."""

    total_bytes = float(total_bpw) * int(expert_weights) / 8.0
    remaining = total_bytes - float(bytes_per_expert)
    if not math.isfinite(remaining) or remaining < 0.0 or int(page_bytes) <= 0:
        raise ValueError("charged bpw cannot contain the supplied metadata")
    return int(math.floor((remaining + 1e-9) / int(page_bytes)))


def controller_accounting(
    charged_bpw: Sequence[float],
    *,
    rank: int = 8,
    candidate_logits: int = 16,
    refreshes: int = 6,
    response_synthesis_bytes_per_expert: int = 10_240,
    shared_bytes_per_layer: int = 98_304,
    uncertainty_bytes_per_expert: int = 1_600,
    response_synthesis_macs_per_expert: int = 10_240,
    pr13_selector_macs: int = PR13_PRIMARY_SELECTOR_MACS,
) -> ControllerAccounting:
    """Estimate a rank-r dynamic controller and matched all-in page budgets."""

    r, candidates, scans = int(rank), int(candidate_logits), int(refreshes)
    if r < 1 or candidates < 9 or scans < 1:
        raise ValueError("controller rank/candidate/refresh counts are invalid")
    current = PR13_FACTOR_BYTES + PR13_ABC_BYTES
    amortized_shared = float(shared_bytes_per_layer) / EXPERTS_PER_LAYER
    joint = (
        PR13_FACTOR_BYTES
        + PR13_ABC_BYTES
        + int(response_synthesis_bytes_per_expert)
        + amortized_shared
    )
    joint_uncertainty = joint + int(uncertainty_bytes_per_expert)
    pages = EXPERTS_PER_GROUP * 3 * UNITS
    # Dense scoring projects every page onto every candidate-logit direction
    # at every refresh.  The one-adjoint form first collapses the current
    # directional risk into one latent vector and scans each page once.
    projection = HIDDEN_SIZE * r * candidates
    direct = pages * r * candidates * scans + projection
    adjoint = pages * r * scans + projection
    synthesis = EXPERTS_PER_GROUP * int(response_synthesis_macs_per_expert)
    baseline_macs = int(pr13_selector_macs)
    if synthesis < 0 or baseline_macs <= 0:
        raise ValueError("response-synthesis/PR13 compute counts are invalid")
    direct_total = direct + synthesis
    adjoint_total = adjoint + synthesis
    rates = tuple(float(value) for value in charged_bpw)
    if not rates or any(not math.isfinite(value) or value <= 0.0 for value in rates):
        raise ValueError("charged bpw values must be positive finite")
    matched = {
        f"{value:.12g}": matched_integral_pages(value, joint)
        for value in rates
    }
    matched_uncertainty = {
        f"{value:.12g}": matched_integral_pages(value, joint_uncertainty)
        for value in rates
    }
    current_model = current * EXPERTS_PER_LAYER * 40
    joint_model = (
        (joint - amortized_shared) * EXPERTS_PER_LAYER * 40
        + int(shared_bytes_per_layer) * 40
    )
    joint_uncertainty_model = joint_model + (
        int(uncertainty_bytes_per_expert) * EXPERTS_PER_LAYER * 40
    )
    return ControllerAccounting(
        rank=r,
        candidate_logits=candidates,
        refreshes=scans,
        direct_candidate_macs=int(direct),
        one_adjoint_macs=int(adjoint),
        response_synthesis_macs=int(synthesis),
        direct_total_macs=int(direct_total),
        one_adjoint_total_macs=int(adjoint_total),
        pr13_primary_selector_macs=baseline_macs,
        direct_fraction_of_pr13=direct_total / float(baseline_macs),
        one_adjoint_fraction_of_pr13=adjoint_total / float(baseline_macs),
        current_metadata_bpw=metadata_bpw(current),
        joint_metadata_bpw=metadata_bpw(joint),
        joint_uncertainty_metadata_bpw=metadata_bpw(joint_uncertainty),
        current_model_metadata_mib=current_model / float(1 << 20),
        joint_model_metadata_mib=joint_model / float(1 << 20),
        joint_uncertainty_model_metadata_mib=(
            joint_uncertainty_model / float(1 << 20)
        ),
        current_group_metadata_kib=(current * EXPERTS_PER_GROUP) / 1024.0,
        joint_group_metadata_kib=(joint * EXPERTS_PER_GROUP) / 1024.0,
        joint_uncertainty_group_metadata_kib=(
            joint_uncertainty * EXPERTS_PER_GROUP / 1024.0
        ),
        matched_pages=matched,
        matched_pages_with_uncertainty=matched_uncertainty,
    )


def provisional_decode_accounting(context_tokens: int) -> ProvisionalPassAccounting:
    """Estimate one invalidated D1 pre-MoE pass over the 29/10 hybrid mix.

    Traffic values are optimistic decimal-MB lower bounds.  Packed accounting
    assumes packed linear-attention projections, BF16 full-attention weights,
    and includes minimum state/cache traffic.  The BF16 path matches the
    dequantized same-host scientific implementation.
    """

    context = int(context_tokens)
    if context < 1:
        raise ValueError("decode context must be positive")
    linear_macs = 35_815_424.0
    full_macs = 27_787_264.0 + 8_192.0 * context
    average = (29.0 * linear_macs + 10.0 * full_macs) / 39.0
    routed_and_shared_moe = 28_313_600.0
    normal = average + routed_and_shared_moe
    linear_state_bytes = 4_194_304.0
    packed_linear_weights = 19_010_000.0
    bf16_linear_weights = 68_490_000.0
    full_weights = 55_575_000.0
    full_cache = 2_048.0 * context
    packed = (
        29.0 * (packed_linear_weights + linear_state_bytes)
        + 10.0 * (full_weights + full_cache)
    ) / 39.0
    bf16 = (
        29.0 * (bf16_linear_weights + linear_state_bytes)
        + 10.0 * (full_weights + full_cache)
    ) / 39.0
    return ProvisionalPassAccounting(
        context_tokens=context,
        average_macs=average,
        all_d1_macs=39.0 * average,
        average_packed_megabytes=packed / 1e6,
        average_bf16_megabytes=bf16 / 1e6,
        fraction_of_normal_layer=average / normal,
    )
