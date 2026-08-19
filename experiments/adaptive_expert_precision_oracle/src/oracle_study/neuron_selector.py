"""Factorized neuron actions and compact residual-response predictors.

The routines in this module operate on decoded weights from the locked
Q2->Q3->Q4 hierarchy.  They never refit the codec or change its selected
trees.  Selection coefficients are fixed: every chosen transition installs
an exact decoded state and no least-squares coefficient refit is performed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from .sparse_streaming import silu


G_FROM_BASE = 0
D_FROM_BASE = 1
D_AFTER_G = 2
G_AFTER_D = 3
TRANSITION_NAMES = ("G", "D", "D_after_G", "G_after_D")
TRANSITION_PAGE_COSTS = (2, 1, 1, 2)


def _as_level(
    level: Sequence[np.ndarray], activation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(level) != 3:
        raise ValueError("a precision level must be (gate, up, down)")
    gate, up, down = (np.asarray(value, dtype=np.float64) for value in level)
    x = np.asarray(activation, dtype=np.float64).reshape(-1)
    if gate.ndim != 2 or up.shape != gate.shape:
        raise ValueError("gate and up must have equal [unit, input] shapes")
    if gate.shape[1] != x.size:
        raise ValueError("activation width does not match gate/up")
    if down.ndim != 2 or down.shape[1] != gate.shape[0]:
        raise ValueError("down must have shape [output, unit]")
    return gate, up, down, x


@dataclass(frozen=True)
class FactorizedUnitOutputs:
    """The four exact Q2/Q4 gate/up-versus-down states for every unit."""

    y00: np.ndarray
    y10: np.ndarray
    y01: np.ndarray
    y11: np.ndarray
    h2: np.ndarray
    h4: np.ndarray

    def __post_init__(self) -> None:
        arrays = tuple(np.asarray(value) for value in (self.y00, self.y10, self.y01, self.y11))
        if any(value.ndim != 2 for value in arrays) or len({value.shape for value in arrays}) != 1:
            raise ValueError("factorized outputs must share [unit, output] shape")
        units = arrays[0].shape[0]
        if np.asarray(self.h2).shape != (units,) or np.asarray(self.h4).shape != (units,):
            raise ValueError("hidden states must have one scalar per unit")

    @property
    def units(self) -> int:
        return int(np.asarray(self.y00).shape[0])

    @property
    def output_size(self) -> int:
        return int(np.asarray(self.y00).shape[1])

    @property
    def base_output(self) -> np.ndarray:
        return np.asarray(self.y00, np.float64).sum(axis=0)

    @property
    def target_output(self) -> np.ndarray:
        return np.asarray(self.y11, np.float64).sum(axis=0)

    @property
    def transitions(self) -> np.ndarray:
        """Return transitions in the order named by ``TRANSITION_NAMES``."""
        return np.stack((
            self.y10 - self.y00,
            self.y01 - self.y00,
            self.y11 - self.y10,
            self.y11 - self.y01,
        )).astype(np.float64, copy=False)


def factorized_unit_outputs(
    q2: Sequence[np.ndarray], q4: Sequence[np.ndarray], activation: np.ndarray,
) -> FactorizedUnitOutputs:
    """Construct exact ``00/G/D/GD`` unit outputs without cross-unit mixing."""
    gate2, up2, down2, x2 = _as_level(q2, activation)
    gate4, up4, down4, x4 = _as_level(q4, activation)
    if (gate2.shape, up2.shape, down2.shape) != (gate4.shape, up4.shape, down4.shape):
        raise ValueError("Q2 and Q4 matrices must have identical shapes")
    if not np.array_equal(x2, x4):
        raise ValueError("Q2 and Q4 activations disagree")
    h2 = silu(gate2 @ x2) * (up2 @ x2)
    h4 = silu(gate4 @ x2) * (up4 @ x2)
    return FactorizedUnitOutputs(
        y00=down2.T * h2[:, None],
        y10=down2.T * h4[:, None],
        y01=down4.T * h2[:, None],
        y11=down4.T * h4[:, None],
        h2=h2,
        h4=h4,
    )


@dataclass(frozen=True)
class FactorizedSelectionTrace:
    order: torch.Tensor
    gains: torch.Tensor
    incremental_pages: torch.Tensor
    cumulative_pages: torch.Tensor
    snapshots: Mapping[int, torch.Tensor]


def _work_device(value: torch.Tensor, device: torch.device | str | None) -> torch.device:
    return value.device if device is None else torch.device(device)


def _float_tensor(value: torch.Tensor | np.ndarray | Sequence[float], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def exact_factorized_neuron_fixed_greedy(
    outputs: FactorizedUnitOutputs,
    *,
    page_budgets: Iterable[int],
    proxy: torch.Tensor | np.ndarray | None = None,
    beta: float = 0.0,
    device: torch.device | str | None = None,
) -> FactorizedSelectionTrace:
    """Exact marginal-per-page greedy over factorized G/D unit states.

    State ``00`` exposes a two-page gate/up action and a one-page down action.
    Selecting either exposes only the complementary action for that unit.
    Marginals are updated from one fixed transition Gram; selected
    coefficients are never refit.  Requested budgets share one path up to the
    largest budget, matching a deployable static path.
    """
    requested = tuple(sorted(set(int(value) for value in page_budgets)))
    maximum_pages = 3 * outputs.units
    if any(value < 0 or value > maximum_pages for value in requested):
        raise ValueError(f"page budgets must lie within [0, {maximum_pages}]")
    transitions_cpu = np.asarray(outputs.transitions, dtype=np.float32)
    transitions = torch.as_tensor(transitions_cpu)
    work_device = _work_device(transitions, device)
    vectors = transitions.to(device=work_device).reshape(4 * outputs.units, outputs.output_size)
    p = None if proxy is None else _float_tensor(proxy, work_device)
    if p is not None:
        if p.ndim == 1:
            p = p[:, None]
        if p.ndim != 2 or p.shape[0] != outputs.output_size:
            raise ValueError("proxy must have shape [output, rank]")
    if beta < 0.0 or (beta != 0.0 and p is None):
        raise ValueError("nonzero beta requires a proxy and beta must be nonnegative")

    target = _float_tensor(outputs.target_output - outputs.base_output, work_device)
    current_output = _float_tensor(outputs.base_output, work_device).clone()
    gram = vectors @ vectors.T
    correlation = vectors @ target
    if p is not None and beta != 0.0:
        projected = vectors @ p
        target_projected = target @ p
        gram = gram + float(beta) * (projected @ projected.T)
        correlation = correlation + float(beta) * (projected @ target_projected)
    diagonal = torch.diagonal(gram)
    costs = torch.tensor(
        [2] * outputs.units + [1] * outputs.units + [1] * outputs.units + [2] * outputs.units,
        dtype=torch.int64,
        device=work_device,
    )
    state = torch.zeros(outputs.units, dtype=torch.int8, device=work_device)

    order: list[int] = []
    gains: list[float] = []
    incremental: list[int] = []
    cumulative: list[int] = []
    snapshots: dict[int, torch.Tensor] = {}
    pending = 0
    current_pages = 0
    while pending < len(requested) and requested[pending] == 0:
        snapshots[0] = current_output.detach().cpu().clone()
        pending += 1

    maximum_requested = max(requested, default=0)
    while current_pages < maximum_requested and len(order) < 2 * outputs.units:
        eligible = torch.zeros(4 * outputs.units, dtype=torch.bool, device=work_device)
        eligible[:outputs.units] = state == 0
        eligible[outputs.units:2 * outputs.units] = state == 0
        eligible[2 * outputs.units:3 * outputs.units] = state == 1
        eligible[3 * outputs.units:] = state == 2
        eligible &= current_pages + costs <= maximum_requested
        if not bool(torch.any(eligible).item()):
            break
        marginal = 2.0 * correlation - diagonal
        efficiency = (marginal / costs.to(marginal.dtype)).masked_fill(~eligible, -torch.inf)
        best_efficiency = torch.max(efficiency)
        tied = eligible & (efficiency == best_efficiency)
        best_gain = torch.max(marginal.masked_fill(~tied, -torch.inf))
        tied &= marginal == best_gain
        best_cost = torch.min(costs.masked_fill(~tied, torch.iinfo(costs.dtype).max))
        tied &= costs == best_cost
        chosen = int(torch.argmax(tied.to(torch.int8)).item())
        cost = int(costs[chosen].item())
        next_pages = current_pages + cost
        while pending < len(requested) and requested[pending] < next_pages:
            snapshots[requested[pending]] = current_output.detach().cpu().clone()
            pending += 1

        block = chosen // outputs.units
        unit = chosen % outputs.units
        if block == G_FROM_BASE:
            state[unit] = 1
        elif block == D_FROM_BASE:
            state[unit] = 2
        else:
            state[unit] = 3
        gain = float(marginal[chosen].item())
        current_output.add_(vectors[chosen])
        correlation.sub_(gram[:, chosen])
        current_pages = next_pages
        order.append(chosen)
        gains.append(gain)
        incremental.append(cost)
        cumulative.append(current_pages)
        while pending < len(requested) and requested[pending] == current_pages:
            snapshots[requested[pending]] = current_output.detach().cpu().clone()
            pending += 1

    while pending < len(requested):
        snapshots[requested[pending]] = current_output.detach().cpu().clone()
        pending += 1
    return FactorizedSelectionTrace(
        order=torch.tensor(order, dtype=torch.int64),
        gains=torch.tensor(gains, dtype=torch.float64),
        incremental_pages=torch.tensor(incremental, dtype=torch.int64),
        cumulative_pages=torch.tensor(cumulative, dtype=torch.int64),
        snapshots=snapshots,
    )


@dataclass(frozen=True)
class UnitScoreMetadata:
    """Three scalar qmetric terms per unit for exact complete-packet scores."""

    a: np.ndarray
    b: np.ndarray
    c: np.ndarray

    def __post_init__(self) -> None:
        arrays = tuple(np.asarray(value).reshape(-1) for value in (self.a, self.b, self.c))
        if len({value.shape for value in arrays}) != 1:
            raise ValueError("A, B and C must have one equal-shaped vector each")

    @property
    def units(self) -> int:
        return int(np.asarray(self.a).size)


def unit_score_metadata(
    down2: np.ndarray,
    down4: np.ndarray,
    *,
    proxy: np.ndarray | None = None,
    beta: float = 0.0,
) -> UnitScoreMetadata:
    """Precompute ``d4'Gd4``, ``d4'Gd2`` and ``d2'Gd2`` per unit."""
    d2 = np.asarray(down2, dtype=np.float64).T
    d4 = np.asarray(down4, dtype=np.float64).T
    if d2.ndim != 2 or d4.shape != d2.shape:
        raise ValueError("down2/down4 must have equal [output, unit] shapes")
    a = np.einsum("ij,ij->i", d4, d4, optimize=True)
    b = np.einsum("ij,ij->i", d4, d2, optimize=True)
    c = np.einsum("ij,ij->i", d2, d2, optimize=True)
    if proxy is not None and beta:
        p = np.asarray(proxy, dtype=np.float64)
        if p.ndim == 1:
            p = p[:, None]
        if p.ndim != 2 or p.shape[0] != d2.shape[1]:
            raise ValueError("proxy output width does not match down matrices")
        p2 = d2 @ p
        p4 = d4 @ p
        a += float(beta) * np.einsum("ij,ij->i", p4, p4, optimize=True)
        b += float(beta) * np.einsum("ij,ij->i", p4, p2, optimize=True)
        c += float(beta) * np.einsum("ij,ij->i", p2, p2, optimize=True)
    return UnitScoreMetadata(a=a, b=b, c=c)


def complete_unit_scores(
    h2: np.ndarray, h4: np.ndarray, metadata: UnitScoreMetadata,
) -> np.ndarray:
    """Exact independent qenergy score from hidden scalars and A/B/C."""
    low = np.asarray(h2, dtype=np.float64).reshape(-1)
    high = np.asarray(h4, dtype=np.float64).reshape(-1)
    if low.shape != high.shape or low.size != metadata.units:
        raise ValueError("hidden vectors and metadata have different unit counts")
    score = (
        np.asarray(metadata.a, np.float64) * high * high
        - 2.0 * np.asarray(metadata.b, np.float64) * high * low
        + np.asarray(metadata.c, np.float64) * low * low
    )
    return np.maximum(score, 0.0)


@dataclass(frozen=True)
class EncodedArray:
    decoded: np.ndarray
    storage_bytes: int
    encoding: str


def encode_array(value: np.ndarray, encoding: str) -> EncodedArray:
    """Quantize selector metadata and return its decoded runtime value."""
    array = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError("selector metadata contains a non-finite input")
    if encoding == "fp16":
        decoded = array.astype(np.float16).astype(np.float32)
        if not np.all(np.isfinite(decoded)):
            raise ValueError("FP16 selector metadata overflowed")
        return EncodedArray(decoded, 2 * array.size, encoding)
    if encoding == "int8_per_row":
        matrix = array.reshape(1, -1) if array.ndim == 1 else array
        raw_scale = np.max(np.abs(matrix), axis=1, keepdims=True) / 127.0
        minimum_scale = np.float32(np.nextafter(np.float16(0.0), np.float16(1.0)))
        scale = np.where(raw_scale > 0.0, np.maximum(raw_scale, minimum_scale), 1.0)
        if np.any(scale > np.finfo(np.float16).max):
            raise ValueError("row-scaled INT8 selector scale overflowed FP16")
        scale = scale.astype(np.float16)
        quantized = np.clip(np.rint(matrix / scale.astype(np.float32)), -127, 127).astype(np.int8)
        decoded = quantized.astype(np.float32) * scale.astype(np.float32)
        if not np.all(np.isfinite(decoded)):
            raise ValueError("row-scaled INT8 selector metadata overflowed")
        return EncodedArray(decoded.reshape(array.shape), quantized.size + scale.size * 2, encoding)
    if encoding in {"fp8_e4m3fn", "fp8_e4m3fn_per_row"}:
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("this Torch build does not provide float8_e4m3fn")
        matrix = array.reshape(1, -1) if array.ndim == 1 else array
        raw_scale = np.max(np.abs(matrix), axis=1, keepdims=True) / 448.0
        # The scale itself is stored in FP16. Clamp nonzero rows to the
        # smallest representable positive value before casting so tiny fitted
        # response rows quantize to zero instead of dividing by an underflowed
        # zero scale. True zero rows retain the exact zero payload.
        minimum_scale = np.float32(np.nextafter(np.float16(0.0), np.float16(1.0)))
        scale = np.where(raw_scale > 0.0, np.maximum(raw_scale, minimum_scale), 1.0)
        if np.any(scale > np.finfo(np.float16).max):
            raise ValueError("row-scaled FP8 selector scale overflowed FP16")
        scale = scale.astype(np.float16)
        normalized = matrix / scale.astype(np.float32)
        tensor = torch.as_tensor(normalized).to(torch.float8_e4m3fn)
        decoded = tensor.to(torch.float32).cpu().numpy() * scale.astype(np.float32)
        if not np.all(np.isfinite(decoded)):
            raise ValueError("row-scaled FP8 selector metadata overflowed")
        return EncodedArray(decoded.reshape(array.shape), array.size + scale.size * 2, encoding)
    raise ValueError(f"unsupported encoding: {encoding}")


def encode_unit_score_metadata(
    metadata: UnitScoreMetadata, encoding: str = "fp16",
) -> tuple[UnitScoreMetadata, int]:
    decoded: list[np.ndarray] = []
    storage_bytes = 0
    for value in (metadata.a, metadata.b, metadata.c):
        array = np.asarray(value, dtype=np.float32)
        scale = np.float32(np.max(np.abs(array), initial=0.0))
        normalized = array / max(float(scale), 1e-30)
        encoded = encode_array(normalized, encoding)
        decoded.append(encoded.decoded.astype(np.float32) * scale)
        # One FP32 scale keeps the FP16 payload finite for every expert.
        storage_bytes += encoded.storage_bytes + 4
    return UnitScoreMetadata(*decoded), storage_bytes


@dataclass(frozen=True)
class ResponseModel:
    """Layer-shared analysis transforms and expert-specific synthesis factors."""

    analysis_gate: np.ndarray
    analysis_up: np.ndarray
    synthesis_gate: Mapping[int, np.ndarray]
    synthesis_up: Mapping[int, np.ndarray]
    rank: int
    basis_kind: str
    fit_diagnostics: Mapping[str, float]

    @property
    def shared_analysis(self) -> bool:
        return self.analysis_gate is self.analysis_up or np.array_equal(self.analysis_gate, self.analysis_up)


def _canonical_subspace(gram: np.ndarray, rank: int) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(gram, dtype=np.float64))
    order = np.argsort(eigenvalues)[::-1][:rank]
    basis = eigenvectors[:, order]
    for column in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0.0:
            basis[:, column] *= -1.0
    return basis


def _fit_analysis(
    x: np.ndarray,
    responses: Sequence[np.ndarray],
    rank: int,
    ridge_relative: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    samples = np.asarray(x, dtype=np.float64)
    if samples.ndim != 2 or not responses:
        raise ValueError("training activations and response matrices are required")
    if rank < 1 or rank > min(samples.shape):
        raise ValueError("rank exceeds the identifiable training subspace")
    gram = np.zeros((samples.shape[0], samples.shape[0]), dtype=np.float64)
    for response in responses:
        value = np.asarray(response, dtype=np.float64)
        if value.ndim != 2 or value.shape[0] != samples.shape[0]:
            raise ValueError("response sample count does not match activations")
        gram += value @ value.T
    desired = _canonical_subspace(gram, rank)
    kernel = samples @ samples.T
    ridge = float(ridge_relative) * max(float(np.trace(kernel)) / max(len(kernel), 1), 1e-30)
    coefficients = np.linalg.solve(kernel + ridge * np.eye(len(kernel)), desired)
    analysis = (coefficients.T @ samples).astype(np.float32)
    latent = samples @ analysis.T
    return analysis, latent, ridge


def _fit_synthesis(latent: np.ndarray, response: np.ndarray) -> np.ndarray:
    coefficient, *_ = np.linalg.lstsq(
        np.asarray(latent, dtype=np.float64), np.asarray(response, dtype=np.float64), rcond=None,
    )
    return coefficient.T.astype(np.float32)


def fit_activation_response_model(
    training_activation: np.ndarray,
    residuals: Mapping[int, tuple[np.ndarray, np.ndarray]],
    *,
    rank: int,
    separate_gate_up: bool,
    ridge_relative: float = 1e-4,
    basis_kind: str = "layer_shared",
) -> ResponseModel:
    """Fit response-optimal shared transforms using training requests only."""
    x = np.asarray(training_activation, dtype=np.float64)
    if x.ndim != 2 or not residuals:
        raise ValueError("training_activation and residual expert matrices are required")
    responses_gate = {expert: x @ np.asarray(pair[0], np.float64).T for expert, pair in residuals.items()}
    responses_up = {expert: x @ np.asarray(pair[1], np.float64).T for expert, pair in residuals.items()}
    if separate_gate_up:
        bg, zg, ridge_g = _fit_analysis(x, list(responses_gate.values()), rank, ridge_relative)
        bu, zu, ridge_u = _fit_analysis(x, list(responses_up.values()), rank, ridge_relative)
    else:
        bg, zg, ridge_g = _fit_analysis(
            x, [*responses_gate.values(), *responses_up.values()], rank, ridge_relative,
        )
        bu, zu, ridge_u = bg, zg, ridge_g
    ag = {expert: _fit_synthesis(zg, response) for expert, response in responses_gate.items()}
    au = {expert: _fit_synthesis(zu, response) for expert, response in responses_up.items()}
    numerator = 0.0
    denominator = 0.0
    for expert in sorted(residuals):
        for actual, predicted in (
            (responses_gate[expert], zg @ ag[expert].T),
            (responses_up[expert], zu @ au[expert].T),
        ):
            numerator += float(np.square(actual - predicted).sum())
            denominator += float(np.square(actual).sum())
    return ResponseModel(
        analysis_gate=bg,
        analysis_up=bu,
        synthesis_gate=ag,
        synthesis_up=au,
        rank=int(rank),
        basis_kind=str(basis_kind),
        fit_diagnostics={
            "training_response_relative_mse": numerator / max(denominator, 1e-30),
            "ridge_gate": ridge_g,
            "ridge_up": ridge_u,
            "training_samples": float(x.shape[0]),
        },
    )


def encode_response_model(
    model: ResponseModel,
    *,
    synthesis_encoding: str,
) -> tuple[ResponseModel, dict[str, int]]:
    """Encode FP16 shared transforms and selected expert synthesis factors."""
    bg = encode_array(model.analysis_gate, "fp16")
    if model.shared_analysis:
        bu = bg
        analysis_bytes = bg.storage_bytes
    else:
        bu = encode_array(model.analysis_up, "fp16")
        analysis_bytes = bg.storage_bytes + bu.storage_bytes
    ag_encoded = {expert: encode_array(value, synthesis_encoding) for expert, value in model.synthesis_gate.items()}
    au_encoded = {expert: encode_array(value, synthesis_encoding) for expert, value in model.synthesis_up.items()}
    encoded_model = ResponseModel(
        analysis_gate=bg.decoded,
        analysis_up=bu.decoded,
        synthesis_gate={expert: value.decoded for expert, value in ag_encoded.items()},
        synthesis_up={expert: value.decoded for expert, value in au_encoded.items()},
        rank=model.rank,
        basis_kind=model.basis_kind,
        fit_diagnostics=model.fit_diagnostics,
    )
    return encoded_model, {
        "layer_analysis_bytes": int(analysis_bytes),
        "expert_synthesis_bytes": int(
            sum(value.storage_bytes for value in (*ag_encoded.values(), *au_encoded.values()))
            / max(len(ag_encoded), 1)
        ),
    }


def predict_residual_response(
    model: ResponseModel, expert: int, activation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(activation, dtype=np.float32).reshape(-1)
    if expert not in model.synthesis_gate or expert not in model.synthesis_up:
        raise KeyError(f"expert {expert} is absent from the response model")
    zg = np.asarray(model.analysis_gate, np.float32) @ x
    zu = zg if model.shared_analysis else np.asarray(model.analysis_up, np.float32) @ x
    return (
        np.asarray(model.synthesis_gate[expert], np.float32) @ zg,
        np.asarray(model.synthesis_up[expert], np.float32) @ zu,
    )


def predict_q4_hidden(
    model: ResponseModel,
    expert: int,
    activation: np.ndarray,
    gate2_response: np.ndarray,
    up2_response: np.ndarray,
) -> np.ndarray:
    delta_gate, delta_up = predict_residual_response(model, expert, activation)
    return silu(np.asarray(gate2_response, np.float64) + delta_gate) * (
        np.asarray(up2_response, np.float64) + delta_up
    )


def descending_order(score: np.ndarray) -> np.ndarray:
    values = np.asarray(score, dtype=np.float64).reshape(-1)
    return np.lexsort((np.arange(values.size, dtype=np.int64), -values))


def utility_weighted_recall(score: np.ndarray, predicted_order: np.ndarray, count: int) -> float:
    values = np.maximum(np.asarray(score, dtype=np.float64).reshape(-1), 0.0)
    k = min(max(int(count), 0), values.size)
    if k == 0:
        return 1.0
    oracle = descending_order(values)[:k]
    predicted = np.asarray(predicted_order, dtype=np.int64).reshape(-1)[:k]
    return float(values[predicted].sum() / max(values[oracle].sum(), 1e-30))


def ndcg_at_k(score: np.ndarray, predicted_order: np.ndarray, count: int) -> float:
    values = np.maximum(np.asarray(score, dtype=np.float64).reshape(-1), 0.0)
    k = min(max(int(count), 0), values.size)
    if k == 0:
        return 1.0
    discount = 1.0 / np.log2(np.arange(k, dtype=np.float64) + 2.0)
    predicted = np.asarray(predicted_order, dtype=np.int64).reshape(-1)[:k]
    ideal = descending_order(values)[:k]
    return float((values[predicted] @ discount) / max(values[ideal] @ discount, 1e-30))


__all__ = [
    "D_AFTER_G",
    "D_FROM_BASE",
    "EncodedArray",
    "FactorizedSelectionTrace",
    "FactorizedUnitOutputs",
    "G_AFTER_D",
    "G_FROM_BASE",
    "ResponseModel",
    "TRANSITION_NAMES",
    "TRANSITION_PAGE_COSTS",
    "UnitScoreMetadata",
    "complete_unit_scores",
    "descending_order",
    "encode_array",
    "encode_response_model",
    "encode_unit_score_metadata",
    "exact_factorized_neuron_fixed_greedy",
    "factorized_unit_outputs",
    "fit_activation_response_model",
    "ndcg_at_k",
    "predict_q4_hidden",
    "predict_residual_response",
    "unit_score_metadata",
    "utility_weighted_recall",
]
