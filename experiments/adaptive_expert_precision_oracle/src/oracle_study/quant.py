"""Deterministic weight/atom quantizers and byte accounting."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _groups(width: int, group_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, width, group_size):
        yield start, min(start + group_size, width)


def binary_quantize(
    weights: np.ndarray,
    group_size: int = 64,
    activation_second_moment: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-output-row signed binary quantization.

    With no activation statistics, scales are mean absolute weights. With a
    second moment, each group scale is the diagonal-covariance functional MSE
    optimum against ``weights`` with signs fixed to ``sign(weights)``.
    """
    weights = np.asarray(weights, dtype=np.float32)
    if weights.ndim != 2:
        raise ValueError("weights must be a matrix")
    out, width = weights.shape
    moments = None
    if activation_second_moment is not None:
        moments = np.asarray(activation_second_moment, dtype=np.float64)
        if moments.shape != (width,) or np.any(moments < 0):
            raise ValueError("bad activation second moment")
    result = np.empty_like(weights)
    scales = np.empty((out, math.ceil(width / group_size)), dtype=np.float32)
    signs = np.where(weights >= 0, 1.0, -1.0).astype(np.float32)
    for group_index, (start, stop) in enumerate(_groups(width, group_size)):
        block = weights[:, start:stop]
        if moments is None:
            scale = np.mean(np.abs(block), axis=1)
        else:
            weight = moments[start:stop][None, :]
            numerator = np.sum(block * signs[:, start:stop] * weight, axis=1)
            denominator = max(float(np.sum(weight)), 1e-30)
            scale = numerator / denominator
        scale = np.maximum(scale, 0.0).astype(np.float32)
        scales[:, group_index] = scale
        result[:, start:stop] = signs[:, start:stop] * scale[:, None]
    return result, scales


def binary_storage_bytes(shape: tuple[int, int], group_size: int = 64) -> dict[str, int | float]:
    out, width = shape
    weight_bits = out * width
    scale_bytes = out * math.ceil(width / group_size) * 2
    header_bytes = 64
    total = math.ceil(weight_bits / 8) + scale_bytes + header_bytes
    return {
        "payload_bytes": math.ceil(weight_bits / 8),
        "scale_bytes": scale_bytes,
        "header_bytes": header_bytes,
        "total_bytes": total,
        "effective_bpw": 8.0 * total / (out * width),
    }


def progressive_quantize(atom: np.ndarray, bits: int) -> tuple[np.ndarray, float]:
    """Nested signed-prefix quantization backed by one int16 master code.

    Each lower precision is obtained by clearing low two's-complement bits of
    the same int16 code. Therefore 2/4/8-bit payloads are strict prefixes of
    the 16-bit representation and the final stage returns the int16 master.
    """
    atom = np.asarray(atom, dtype=np.float32)
    if bits not in (0, 2, 4, 8, 16):
        raise ValueError("bits must be one of 0,2,4,8,16")
    maximum = float(np.max(np.abs(atom))) if atom.size else 0.0
    scale = maximum / 32767.0 if maximum > 0 else 1.0
    master = np.clip(np.rint(atom / scale), -32767, 32767).astype(np.int16)
    if bits == 0:
        return np.zeros_like(atom), scale
    shift = 16 - bits
    restored = ((master.astype(np.int32) >> shift) << shift).astype(np.int16)
    return restored.astype(np.float32) * scale, scale


def progressive_packet_bytes(length: int, bits: int, metadata_bytes: int = 8) -> int:
    if bits == 0:
        return 0
    return math.ceil(length * bits / 8) + 2 + metadata_bytes


def page_ids_for_atoms(
    atom_ids: np.ndarray,
    packet_bytes: int,
    page_size: int,
    layout_order: np.ndarray | None = None,
) -> np.ndarray:
    atom_ids = np.asarray(atom_ids, dtype=np.int64)
    if layout_order is None:
        positions = atom_ids
    else:
        inverse = np.empty_like(layout_order)
        inverse[np.asarray(layout_order, dtype=np.int64)] = np.arange(len(layout_order))
        positions = inverse[atom_ids]
    starts = positions * int(packet_bytes)
    ends = starts + int(packet_bytes) - 1
    pages: list[int] = []
    for start, end in zip(starts.tolist(), ends.tolist()):
        pages.extend(range(start // page_size, end // page_size + 1))
    return np.unique(np.asarray(pages, dtype=np.int64))


def co_selection_layout(binary_selection: np.ndarray) -> np.ndarray:
    """Greedy training-only adjacency layout from a selection incidence matrix."""
    selected = np.asarray(binary_selection, dtype=np.float32)
    if selected.ndim != 2:
        raise ValueError("selection incidence must be [samples, atoms]")
    count = selected.sum(axis=0)
    affinity = selected.T @ selected
    remaining = set(range(selected.shape[1]))
    if not remaining:
        return np.empty(0, dtype=np.int64)
    first = int(np.argmax(count))
    order = [first]
    remaining.remove(first)
    while remaining:
        last = order[-1]
        candidate = max(remaining, key=lambda j: (float(affinity[last, j]), float(count[j]), -j))
        order.append(candidate)
        remaining.remove(candidate)
    return np.asarray(order, dtype=np.int64)
