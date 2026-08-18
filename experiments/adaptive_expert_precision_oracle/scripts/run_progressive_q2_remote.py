#!/usr/bin/env python3
"""Run the progressive projection pilot with a deterministic 2-bit resident base."""

from __future__ import annotations

import math
import numpy as np

import oracle_study.quant as quant
import run_progressive_remote as implementation


def q2_quantize(
    weights: np.ndarray,
    group_size: int = 64,
    activation_second_moment: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Activation-diagonal-MSE four-level signed quantizer with levels ±1,±3."""
    weights = np.asarray(weights, dtype=np.float32)
    if weights.ndim != 2:
        raise ValueError("weights must be a matrix")
    out, width = weights.shape
    moments = np.ones(width, dtype=np.float64) if activation_second_moment is None else np.asarray(activation_second_moment, dtype=np.float64)
    if moments.shape != (width,) or np.any(moments < 0):
        raise ValueError("bad activation second moment")
    result = np.empty_like(weights)
    scales = np.empty((out, math.ceil(width / group_size)), dtype=np.float32)
    for group_index, start in enumerate(range(0, width, group_size)):
        stop = min(start + group_size, width)
        block = weights[:, start:stop].astype(np.float64)
        weight = moments[start:stop][None, :]
        scale = np.maximum(np.sum(np.abs(block) * weight, axis=1) / max(float(np.sum(weight)), 1e-30), 1e-12)
        code = np.ones_like(block)
        for _ in range(8):
            magnitude = np.where(np.abs(block) < 2.0 * scale[:, None], 1.0, 3.0)
            code = np.where(block >= 0, magnitude, -magnitude)
            numerator = np.sum(block * code * weight, axis=1)
            denominator = np.sum(code * code * weight, axis=1)
            scale = np.maximum(numerator / np.maximum(denominator, 1e-30), 1e-12)
        scales[:, group_index] = scale.astype(np.float32)
        result[:, start:stop] = (code * scale[:, None]).astype(np.float32)
    return result, scales


if __name__ == "__main__":
    quant.binary_quantize = q2_quantize
    implementation.main()
