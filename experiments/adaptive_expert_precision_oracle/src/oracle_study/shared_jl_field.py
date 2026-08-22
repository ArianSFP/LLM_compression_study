"""Globally aligned JL interaction signatures with exact local geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .interaction_field import (
    EncodedInteractionFactor,
    JointInteractionFactor,
    encode_interaction_factor,
    make_factor_self_safe,
    make_encoded_factor_self_safe,
    walsh_hadamard,
)
from .neuron_selector import UnitScoreMetadata


@dataclass(frozen=True)
class SharedJLFactor:
    factor: JointInteractionFactor
    encoded: EncodedInteractionFactor | None
    self_safe_scale: float
    maximum_calibration_error: float


def quantized_shared_factor_with_pair_calibration(
    factor: JointInteractionFactor,
    metadata: UnitScoreMetadata,
    *,
    encoding: str,
    hadamard_rotate: bool = True,
) -> SharedJLFactor:
    """Quantize a wide shared field, then store a tiny local pair correction.

    Sign rows use one bit per latent value. Ternary rows use two bits and an
    exact least-squares support threshold. One FP16 scale per row and one FP16
    two-by-two transform per unit restore the local Q4/Q2 Gram after decoding.
    """
    if encoding not in {"sign_per_row", "ternary_per_row"}:
        raise ValueError("encoding must be sign_per_row or ternary_per_row")
    joint = np.asarray(factor.joint(), np.float64)
    if hadamard_rotate:
        joint = joint @ walsh_hadamard(factor.rank)
    rows, rank = joint.shape
    codes = np.empty_like(joint, np.int8)
    raw_scales = np.empty(rows, np.float64)
    for row in range(rows):
        values = joint[row]
        if encoding == "sign_per_row":
            raw_scales[row] = float(np.mean(np.abs(values)))
            codes[row] = np.where(values >= 0.0, 1, -1)
        else:
            order = np.argsort(-np.abs(values), kind="stable")
            sorted_abs = np.abs(values[order])
            cumulative = np.cumsum(sorted_abs)
            count = int(np.argmax(
                cumulative * cumulative / np.arange(1, rank + 1)
            )) + 1
            raw_scales[row] = float(cumulative[count - 1] / count)
            codes[row].fill(0)
            chosen = order[:count]
            codes[row, chosen] = np.where(values[chosen] >= 0.0, 1, -1)
    minimum = np.float64(np.nextafter(np.float16(0.0), np.float16(1.0)))
    target_scales = np.maximum(raw_scales, minimum)
    stored_scales = target_scales.astype(np.float16)
    rounds_down = stored_scales.astype(np.float64) < target_scales
    stored_scales[rounds_down] = np.nextafter(
        stored_scales[rounds_down], np.float16(np.inf),
    )
    decoded = codes.astype(np.float64) * stored_scales.astype(np.float64)[:, None]
    a = np.asarray(metadata.a, np.float64)
    b = np.asarray(metadata.b, np.float64)
    c = np.asarray(metadata.c, np.float64)
    transforms = np.empty((factor.units, 2, 2), np.float16)
    calibrated = np.empty_like(decoded)
    maximum_error = 0.0
    for unit in range(factor.units):
        pair = np.stack((decoded[unit], decoded[factor.units + unit]))
        target = np.asarray(((a[unit], b[unit]), (b[unit], c[unit])), np.float64)
        transform = (
            _matrix_square_root(target, False)
            @ _matrix_square_root(pair @ pair.T, True)
        )
        transforms[unit] = transform.astype(np.float16)
        restored = transforms[unit].astype(np.float64) @ pair
        calibrated[unit], calibrated[factor.units + unit] = restored
        maximum_error = max(
            maximum_error, float(np.max(np.abs(restored @ restored.T - target)))
        )
    bits = 1 if encoding == "sign_per_row" else 2
    packed_code_bytes = (2 * factor.units * rank * bits + 7) // 8
    storage_bytes = packed_code_bytes + stored_scales.nbytes + transforms.nbytes + 4
    decoded_factor = JointInteractionFactor(
        l4=calibrated[: factor.units].astype(np.float32),
        l2=calibrated[factor.units :].astype(np.float32),
        method=factor.method + "_pair_calibrated",
        tail_rank=factor.tail_rank,
        exact_rank=factor.exact_rank,
        encoding=encoding + ("_hadamard" if hadamard_rotate else ""),
        storage_bytes=storage_bytes,
    )
    safe, scale = make_factor_self_safe(decoded_factor, metadata)
    margin = np.float32(1.0 - 1e-6)
    safe = JointInteractionFactor(
        l4=np.asarray(safe.l4 * margin, np.float32),
        l2=np.asarray(safe.l2 * margin, np.float32),
        method=safe.method,
        tail_rank=safe.tail_rank,
        exact_rank=safe.exact_rank,
        encoding=safe.encoding + "_rounding_safe",
        storage_bytes=safe.storage_bytes,
    )
    return SharedJLFactor(safe, None, float(scale) * float(margin), maximum_error)


def rademacher_projection(width: int, rank: int, seed: int) -> np.ndarray:
    """Create a deterministic dense signed JL projection."""
    input_width, output_rank = int(width), int(rank)
    if input_width < 1 or output_rank < 1:
        raise ValueError("JL dimensions must be positive")
    rng = np.random.default_rng(int(seed))
    signs = rng.integers(0, 2, size=(input_width, output_rank), dtype=np.int8)
    signs = signs.astype(np.float32) * 2.0 - 1.0
    return signs / np.float32(np.sqrt(output_rank))


def _matrix_square_root(matrix: np.ndarray, inverse: bool) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(symmetric)
    floor = np.finfo(np.float64).eps * max(float(np.max(values)), 1.0)
    if inverse:
        transformed = 1.0 / np.sqrt(np.maximum(values, floor))
    else:
        transformed = np.sqrt(np.maximum(values, 0.0))
    return (vectors * transformed[None, :]) @ vectors.T


def calibrated_shared_factor(
    tail_l4: np.ndarray,
    tail_l2: np.ndarray,
    proxy_l4: np.ndarray,
    proxy_l2: np.ndarray,
    metadata: UnitScoreMetadata,
    *,
    encoding: str | None,
) -> SharedJLFactor:
    """Calibrate each Q4/Q2 row pair to its exact A/B/C Gram, then encode.

    The shared sketch supplies globally aligned cross-unit coordinates.  A
    two-by-two whitening/coloring transform makes each unit's local down Gram
    exactly equal to its stored A/B/C values before quantization.
    """
    high_tail = np.asarray(tail_l4, np.float64)
    low_tail = np.asarray(tail_l2, np.float64)
    high_proxy = np.asarray(proxy_l4, np.float64)
    low_proxy = np.asarray(proxy_l2, np.float64)
    if high_tail.shape != low_tail.shape or high_tail.ndim != 2:
        raise ValueError("tail signatures must have equal [unit,rank] shapes")
    if high_proxy.shape != low_proxy.shape or high_proxy.shape[0] != high_tail.shape[0]:
        raise ValueError("proxy signatures must share the unit dimension")
    raw_high = np.concatenate((high_tail, high_proxy), axis=1)
    raw_low = np.concatenate((low_tail, low_proxy), axis=1)
    if raw_high.shape[1] < 2 or metadata.units != raw_high.shape[0]:
        raise ValueError("calibrated field shape changed")
    a = np.asarray(metadata.a, np.float64)
    b = np.asarray(metadata.b, np.float64)
    c = np.asarray(metadata.c, np.float64)
    high = np.empty_like(raw_high)
    low = np.empty_like(raw_low)
    maximum_error = 0.0
    for unit in range(raw_high.shape[0]):
        rows = np.stack((raw_high[unit], raw_low[unit]))
        target = np.asarray(((a[unit], b[unit]), (b[unit], c[unit])), np.float64)
        transform = _matrix_square_root(target, False) @ _matrix_square_root(rows @ rows.T, True)
        calibrated = transform @ rows
        high[unit], low[unit] = calibrated
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(calibrated @ calibrated.T - target))),
        )
    factor = JointInteractionFactor(
        l4=high.astype(np.float32),
        l2=low.astype(np.float32),
        method="shared_rademacher_jl_exact_abc_calibrated",
        tail_rank=high_tail.shape[1],
        exact_rank=high_proxy.shape[1],
        encoding="fp32_control" if encoding is None else str(encoding),
    )
    if encoding is None:
        return SharedJLFactor(factor, None, 1.0, maximum_error)
    encoded = encode_interaction_factor(
        factor, str(encoding), hadamard_rotate=True,
    )
    safe, scale = make_encoded_factor_self_safe(encoded, metadata)
    return SharedJLFactor(safe.decode(), safe, scale, maximum_error)
