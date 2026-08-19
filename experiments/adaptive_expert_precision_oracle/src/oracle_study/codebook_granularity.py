"""Local-codebook progressive MXFP4 codecs used by the H0 granularity study.

The module operates on exact compressed-tensors MXFP4 leaf codes.  An exact
codec changes only the balanced leaf-to-(parent, r1, r2) bijection and its Q2/Q3
FP16 reconstruction tables; Q4 always decodes through the checkpoint E2M1
leaf table.  The fitting routines consume diagonal functional masses, which
keeps assignment/update steps reproducible and cheap enough for a bounded
four-layer oracle study.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Iterable

import numpy as np

from .mxfp4_embed import LEAF_VALUES, MXFP4LeafTensor, maps_from_order


FORMAT_MAGIC = b"CGBMXF4\0"
FORMAT_VERSION = 1
FORMAT_HEADER_BYTES = 64


@dataclass(frozen=True)
class ProgressiveBook:
    """One 32-byte exact-MXFP4 progressive reconstruction table."""

    parent: np.ndarray
    r1: np.ndarray
    r2: np.ndarray
    c2: np.ndarray
    c3: np.ndarray
    label: str = "learned"

    def __post_init__(self) -> None:
        parent = np.asarray(self.parent, dtype=np.uint8)
        r1 = np.asarray(self.r1, dtype=np.uint8)
        r2 = np.asarray(self.r2, dtype=np.uint8)
        if parent.shape != (16,) or r1.shape != (16,) or r2.shape != (16,):
            raise ValueError("book maps must contain 16 leaves")
        if np.asarray(self.c2).shape != (4,) or np.asarray(self.c3).shape != (4, 2):
            raise ValueError("invalid centroid table")
        triples = {(int(parent[i]), int(r1[i]), int(r2[i])) for i in range(16)}
        if len(triples) != 16:
            raise ValueError("book is not a balanced exact-leaf bijection")
        counts = np.bincount(parent, minlength=4)
        if not np.array_equal(counts, np.full(4, 4)):
            raise ValueError("each parent must own four exact leaves")

    def normalized_table(self, level: int) -> np.ndarray:
        if level == 2:
            return np.asarray(self.c2, np.float32)[np.asarray(self.parent, np.uint8)]
        if level == 3:
            return np.asarray(self.c3, np.float32)[
                np.asarray(self.parent, np.uint8), np.asarray(self.r1, np.uint8)
            ]
        if level == 4:
            return LEAF_VALUES.copy()
        raise ValueError(level)

    def distortion_vector(self, level: int) -> np.ndarray:
        return (LEAF_VALUES.astype(np.float64) - self.normalized_table(level)) ** 2

    def to_bytes(self) -> bytes:
        digits = (
            np.asarray(self.parent, np.uint8)
            | (np.asarray(self.r1, np.uint8) << np.uint8(2))
            | (np.asarray(self.r2, np.uint8) << np.uint8(3))
        )
        packed = digits[0::2] | (digits[1::2] << np.uint8(4))
        centroids = np.concatenate((np.asarray(self.c2).reshape(-1), np.asarray(self.c3).reshape(-1)))
        payload = packed.tobytes() + centroids.astype("<f2").tobytes()
        if len(payload) != 32:
            raise AssertionError(len(payload))
        return payload

    @classmethod
    def from_bytes(cls, payload: bytes, label: str = "serialized") -> "ProgressiveBook":
        if len(payload) != 32:
            raise ValueError("a progressive book is exactly 32 bytes")
        packed = np.frombuffer(payload[:8], np.uint8)
        digits = np.empty(16, np.uint8)
        digits[0::2] = packed & np.uint8(15)
        digits[1::2] = packed >> np.uint8(4)
        values = np.frombuffer(payload[8:], dtype="<f2").astype(np.float32)
        return cls(digits & 3, (digits >> 2) & 1, (digits >> 3) & 1, values[:4], values[4:].reshape(4, 2), label)

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "parent": np.asarray(self.parent, np.uint8).tolist(),
            "r1": np.asarray(self.r1, np.uint8).tolist(),
            "r2": np.asarray(self.r2, np.uint8).tolist(),
            "c2": np.asarray(self.c2, np.float32).tolist(),
            "c3": np.asarray(self.c3, np.float32).tolist(),
            "serialized_bytes": 32,
        }


@dataclass(frozen=True)
class LearnedScalarBook:
    """Unbalanced learned-leaf diagnostic; it is deliberately not code exact."""

    parent: np.ndarray
    r1: np.ndarray
    r2: np.ndarray
    c2: np.ndarray
    c3: np.ndarray
    c4: np.ndarray
    label: str = "learned_16_leaf"

    def __post_init__(self) -> None:
        if np.asarray(self.parent).shape != (16,) or np.asarray(self.r1).shape != (16,) or np.asarray(self.r2).shape != (16,):
            raise ValueError("learned leaf maps must contain 16 source symbols")
        if np.asarray(self.c2).shape != (4,) or np.asarray(self.c3).shape != (4, 2) or np.asarray(self.c4).shape != (4, 2, 2):
            raise ValueError("invalid learned hierarchy tables")

    def normalized_table(self, level: int) -> np.ndarray:
        parent = np.asarray(self.parent, np.uint8)
        r1 = np.asarray(self.r1, np.uint8)
        r2 = np.asarray(self.r2, np.uint8)
        if level == 2:
            return np.asarray(self.c2, np.float32)[parent]
        if level == 3:
            return np.asarray(self.c3, np.float32)[parent, r1]
        if level == 4:
            return np.asarray(self.c4, np.float32)[parent, r1, r2]
        raise ValueError(level)

    def distortion_vector(self, level: int) -> np.ndarray:
        return (LEAF_VALUES.astype(np.float64) - self.normalized_table(level)) ** 2

    def to_bytes(self) -> bytes:
        digits = (
            np.asarray(self.parent, np.uint8)
            | (np.asarray(self.r1, np.uint8) << np.uint8(2))
            | (np.asarray(self.r2, np.uint8) << np.uint8(3))
        )
        packed = digits[0::2] | (digits[1::2] << np.uint8(4))
        centroids = np.concatenate((np.asarray(self.c2).reshape(-1), np.asarray(self.c3).reshape(-1), np.asarray(self.c4).reshape(-1)))
        payload = packed.tobytes() + centroids.astype("<f2").tobytes()
        if len(payload) != 64:
            raise AssertionError(len(payload))
        return payload


def _weighted_kmeans_1d(values: np.ndarray, weights: np.ndarray, clusters: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    targets = (np.arange(clusters) + 0.5) * cumulative[-1] / clusters
    centers = values[order[np.searchsorted(cumulative, targets, side="left")]].astype(np.float64)
    assignment = np.zeros(len(values), np.uint8)
    for _ in range(64):
        updated = np.argmin((values[:, None] - centers[None, :]) ** 2, axis=1).astype(np.uint8)
        if np.array_equal(updated, assignment):
            assignment = updated
            break
        assignment = updated
        for cluster in range(clusters):
            mask = assignment == cluster
            if np.any(mask) and weights[mask].sum() > 0:
                centers[cluster] = np.average(values[mask], weights=weights[mask])
    return assignment, centers


def fit_learned_scalar_book(masses: np.ndarray, label: str = "learned_16_leaf") -> LearnedScalarBook:
    """Fit an unbalanced 4x2x2 scalar tree over source MXFP4 values."""
    weights = np.maximum(np.asarray(masses, np.float64), 1e-30)
    parent, c2 = _weighted_kmeans_1d(LEAF_VALUES.astype(np.float64), weights, 4)
    r1 = np.zeros(16, np.uint8)
    r2 = np.zeros(16, np.uint8)
    c3 = np.zeros((4, 2), np.float64)
    c4 = np.zeros((4, 2, 2), np.float64)
    for p in range(4):
        leaves = np.flatnonzero(parent == p)
        if not len(leaves):
            c3[p] = c2[p]
            c4[p] = c2[p]
            continue
        child, child_centers = _weighted_kmeans_1d(LEAF_VALUES[leaves].astype(np.float64), weights[leaves], 2)
        r1[leaves] = child
        c3[p] = child_centers
        for bit in range(2):
            grandchildren = leaves[child == bit]
            if not len(grandchildren):
                c4[p, bit] = child_centers[bit]
                continue
            terminal, terminal_centers = _weighted_kmeans_1d(
                LEAF_VALUES[grandchildren].astype(np.float64), weights[grandchildren], 2,
            )
            r2[grandchildren] = terminal
            c4[p, bit] = terminal_centers
    return LearnedScalarBook(
        parent, r1, r2, c2.astype(np.float16).astype(np.float32),
        c3.astype(np.float16).astype(np.float32), c4.astype(np.float16).astype(np.float32), label,
    )


def book_from_order(order: Iterable[int], masses: np.ndarray, label: str = "learned") -> ProgressiveBook:
    parent, r1, r2 = maps_from_order(order)
    weights = np.asarray(masses, np.float64)
    if weights.shape != (16,):
        raise ValueError("leaf masses must have shape (16,)")
    c2 = np.empty(4, np.float32)
    c3 = np.empty((4, 2), np.float32)
    for p in range(4):
        leaves = np.flatnonzero(parent == p)
        local = weights[leaves]
        c2[p] = np.average(LEAF_VALUES[leaves], weights=local) if local.sum() > 0 else np.mean(LEAF_VALUES[leaves])
        for bit in range(2):
            child = leaves[r1[leaves] == bit]
            child_weight = weights[child]
            c3[p, bit] = np.average(LEAF_VALUES[child], weights=child_weight) if child_weight.sum() > 0 else np.mean(LEAF_VALUES[child])
    # The serialized representation, not an unrounded fitting table, is scored.
    return ProgressiveBook(parent, r1, r2, c2.astype(np.float16).astype(np.float32), c3.astype(np.float16).astype(np.float32), label)


def candidate_orders(seed: int, count: int = 96) -> list[tuple[int, ...]]:
    """Deterministic balanced-tree search pool with useful fixed seeds."""
    fixed = [
        tuple(range(16)),
        tuple(np.argsort(LEAF_VALUES, kind="stable").tolist()),
        (0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15),
    ]
    rng = np.random.default_rng(seed)
    found = {value: None for value in fixed}
    while len(found) < max(count, len(fixed)):
        found[tuple(map(int, rng.permutation(16)))] = None
    return list(found)[:count]


def fit_best_book(
    masses: np.ndarray,
    orders: Iterable[Iterable[int]],
    q2_tolerance: float,
    label: str,
) -> ProgressiveBook:
    books = [book_from_order(order, masses, label) for order in orders]
    q2 = np.asarray([np.dot(masses, book.distortion_vector(2)) for book in books], np.float64)
    best = float(q2.min())
    eligible = np.flatnonzero(q2 <= best * (1.0 + q2_tolerance) + 1e-24)
    q3 = np.asarray([np.dot(masses, books[i].distortion_vector(3)) for i in eligible], np.float64)
    return books[int(eligible[int(np.argmin(q3))])]


def _kmeans_seed_assignments(masses: np.ndarray, families: int, seed: int) -> np.ndarray:
    """Small deterministic k-means on normalized leaf histograms."""
    values = np.asarray(masses, np.float32)
    total = values.sum(axis=1, keepdims=True)
    features = values / np.maximum(total, 1e-30)
    rng = np.random.default_rng(seed)
    if len(features) > 65536:
        sampled = features[rng.choice(len(features), 65536, replace=False)]
    else:
        sampled = features
    centers = np.empty((families, 16), np.float32)
    centers[0] = sampled[int(rng.integers(len(sampled)))]
    closest = np.sum((sampled - centers[0]) ** 2, axis=1)
    for index in range(1, families):
        chosen = int(np.argmax(closest))
        centers[index] = sampled[chosen]
        closest = np.minimum(closest, np.sum((sampled - centers[index]) ** 2, axis=1))
    for _ in range(6):
        assignment = np.argmin(
            np.sum(features[:, None, :] ** 2, axis=2)
            - 2.0 * features @ centers.T
            + np.sum(centers[None, :, :] ** 2, axis=2),
            axis=1,
        )
        for index in range(families):
            selected = features[assignment == index]
            if len(selected):
                centers[index] = selected.mean(axis=0)
    return assignment.astype(np.uint16)


def assign_books(masses: np.ndarray, books: list[ProgressiveBook], q2_tolerance: float) -> np.ndarray:
    errors2 = np.stack([book.distortion_vector(2) for book in books], axis=1)
    errors3 = np.stack([book.distortion_vector(3) for book in books], axis=1)
    d2 = np.asarray(masses, np.float64) @ errors2
    d3 = np.asarray(masses, np.float64) @ errors3
    best2 = np.min(d2, axis=1, keepdims=True)
    scale = np.maximum(np.asarray(masses, np.float64).sum(axis=1, keepdims=True), 1.0)
    eligible = d2 <= best2 * (1.0 + q2_tolerance) + 1e-14 * scale
    d3[~eligible] = np.inf
    return np.argmin(d3, axis=1).astype(np.uint16)


def fit_family_books(
    masses: np.ndarray,
    families: int,
    q2_tolerance: float,
    seed: int,
    *,
    validation_masses: np.ndarray | None = None,
    order_count: int = 96,
    max_iterations: int = 8,
) -> tuple[list[ProgressiveBook], np.ndarray, list[dict[str, float]]]:
    """Alternating block assignment and balanced hierarchy update."""
    values = np.asarray(masses, np.float64)
    if values.ndim != 2 or values.shape[1] != 16:
        raise ValueError("masses must have shape [units, 16]")
    if not 1 <= families <= 256:
        raise ValueError(families)
    assignments = _kmeans_seed_assignments(values, families, seed)
    validation = values if validation_masses is None else np.asarray(validation_masses, np.float64)
    if validation.shape != values.shape:
        raise ValueError("validation masses must align one-to-one with training units")
    orders = candidate_orders(seed + 17, order_count)
    books: list[ProgressiveBook] = []
    history: list[dict[str, float]] = []
    previous: np.ndarray | None = None
    best_books: list[ProgressiveBook] | None = None
    best_assignments: np.ndarray | None = None
    best_validation_q2 = np.inf
    best_validation_q3 = np.inf
    stale = 0
    snapshots: list[tuple[float, float, list[ProgressiveBook], np.ndarray]] = []
    for iteration in range(max_iterations):
        books = []
        for family in range(families):
            selected = values[assignments == family]
            aggregate = selected.sum(axis=0) if len(selected) else values[(family * 104729 + seed) % len(values)]
            books.append(fit_best_book(aggregate, orders, q2_tolerance, f"family_{family:03d}"))
        updated = assign_books(values, books, q2_tolerance)
        q2 = sum(float(np.dot(values[updated == i].sum(axis=0), book.distortion_vector(2))) for i, book in enumerate(books))
        q3 = sum(float(np.dot(values[updated == i].sum(axis=0), book.distortion_vector(3))) for i, book in enumerate(books))
        validation_q2 = sum(float(np.dot(validation[updated == i].sum(axis=0), book.distortion_vector(2))) for i, book in enumerate(books))
        validation_q3 = sum(float(np.dot(validation[updated == i].sum(axis=0), book.distortion_vector(3))) for i, book in enumerate(books))
        counts = np.bincount(updated, minlength=families).astype(np.float64)
        probability = counts[counts > 0] / max(counts.sum(), 1.0)
        entropy = float(-np.sum(probability * np.log(probability)))
        history.append({
            "iteration": float(iteration), "q2_distortion": q2, "q3_distortion": q3,
            "validation_q2_distortion": validation_q2, "validation_q3_distortion": validation_q3,
            "changed_fraction": 1.0 if previous is None else float(np.mean(updated != previous)),
            "entropy_nats": entropy, "effective_families": float(np.exp(entropy)),
        })
        snapshots.append((validation_q2, validation_q3, list(books), updated.copy()))
        improves_q2 = validation_q2 < best_validation_q2 * (1.0 - 1e-9)
        within_q2 = validation_q2 <= best_validation_q2 * (1.0 + q2_tolerance) + 1e-24
        improves_q3 = validation_q3 < best_validation_q3 * (1.0 - 1e-9)
        if best_books is None or improves_q2 or (within_q2 and improves_q3):
            best_validation_q2 = min(best_validation_q2, validation_q2)
            best_validation_q3 = validation_q3
            best_books = list(books)
            best_assignments = updated.copy()
            stale = 0
        else:
            stale += 1
        if previous is not None and np.array_equal(updated, previous):
            assignments = updated
            break
        if stale >= 2:
            break
        previous = assignments
        assignments = updated
    assert best_books is not None and best_assignments is not None
    minimum_q2 = min(value[0] for value in snapshots)
    eligible = [value for value in snapshots if value[0] <= minimum_q2 * (1.0 + q2_tolerance) + 1e-24]
    _, _, selected_books, selected_assignments = min(eligible, key=lambda value: (value[1], value[0]))
    return selected_books, selected_assignments, history


def block_leaf_masses(
    tensor: MXFP4LeafTensor,
    importance: np.ndarray,
    group_size: int,
) -> np.ndarray:
    """Sufficient weighted leaf statistics for 16/32/64-weight storage groups."""
    if group_size not in (16, 32, 64):
        raise ValueError(group_size)
    codes = np.asarray(tensor.codes, np.uint8)
    weight = np.broadcast_to(np.asarray(importance, np.float64), codes.shape)
    weight = weight * tensor.expanded_scales.astype(np.float64) ** 2
    if codes.shape[1] % group_size:
        raise ValueError("group size does not divide the storage row")
    grouped_codes = codes.reshape(-1, group_size)
    grouped_weight = weight.reshape(-1, group_size)
    result = np.zeros((len(grouped_codes), 16), np.float64)
    rows = np.arange(len(grouped_codes))[:, None]
    np.add.at(result, (np.broadcast_to(rows, grouped_codes.shape), grouped_codes), grouped_weight)
    return result


def vector_leaf_masses(tensor: MXFP4LeafTensor, importance: np.ndarray, projection: str) -> np.ndarray:
    codes = np.asarray(tensor.codes, np.uint8)
    weight = np.broadcast_to(np.asarray(importance, np.float64), codes.shape)
    weight = weight * tensor.expanded_scales.astype(np.float64) ** 2
    if projection in ("gate", "up"):
        result = np.zeros((codes.shape[0], 16), np.float64)
        rows = np.arange(codes.shape[0])[:, None]
        np.add.at(result, (np.broadcast_to(rows, codes.shape), codes), weight)
        return result
    if projection == "down":
        result = np.zeros((codes.shape[1], 16), np.float64)
        columns = np.arange(codes.shape[1])[None, :]
        np.add.at(result, (np.broadcast_to(columns, codes.shape), codes), weight)
        return result
    raise ValueError(projection)


def fit_independent_books(
    masses: np.ndarray, q2_tolerance: float, seed: int, order_count: int = 48,
) -> list[ProgressiveBook]:
    orders = candidate_orders(seed, order_count)
    return [fit_best_book(row, orders, q2_tolerance, f"vector_{i:04d}") for i, row in enumerate(np.asarray(masses))]


def unpack_selector(payload: bytes, count: int, bits: int) -> np.ndarray:
    raw = np.frombuffer(payload, np.uint8)
    unpacked = np.unpackbits(raw, bitorder="little")[: count * bits].reshape(count, bits)
    weights = (np.uint16(1) << np.arange(bits, dtype=np.uint16))[None, :]
    return np.sum(unpacked.astype(np.uint16) * weights, axis=1, dtype=np.uint16)


def pack_selector(values: np.ndarray, bits: int) -> bytes:
    ids = np.asarray(values, np.uint16).reshape(-1)
    if np.any(ids >= (1 << bits)):
        raise ValueError("selector does not fit requested bit width")
    planes = ((ids[:, None] >> np.arange(bits, dtype=np.uint16)) & 1).astype(np.uint8)
    return np.packbits(planes.reshape(-1), bitorder="little").tobytes()


def decode_with_books(
    tensor: MXFP4LeafTensor,
    books: list[ProgressiveBook],
    selectors: np.ndarray | None,
    group_size: int,
    projection: str,
    level: int,
    modifiers: np.ndarray | None = None,
) -> np.ndarray:
    """Decode an exact codec at Q2/Q3/Q4 from its serialized semantics."""
    if level == 4 and not hasattr(books[0], "c4"):
        return tensor.dequantize()
    codes = np.asarray(tensor.codes, np.uint8)
    if selectors is None:
        vector_ids = np.arange(len(books), dtype=np.int64)
        if projection in ("gate", "up"):
            book_grid = np.broadcast_to(vector_ids[:, None], codes.shape)
        elif projection == "down":
            book_grid = np.broadcast_to(vector_ids[None, :], codes.shape)
        else:
            raise ValueError(projection)
    else:
        per_row = codes.shape[1] // group_size
        ids = np.asarray(selectors, np.int64).reshape(codes.shape[0], per_row)
        book_grid = np.repeat(ids, group_size, axis=1)
    parent = np.stack([book.parent for book in books])
    r1 = np.stack([book.r1 for book in books])
    c2 = np.stack([book.c2 for book in books])
    c3 = np.stack([book.c3 for book in books])
    p = parent[book_grid, codes]
    if level == 2:
        normalized = c2[book_grid, p]
        modifier_columns = (0, 1)
    elif level == 3:
        bit = r1[book_grid, codes]
        normalized = c3[book_grid, p, bit]
        modifier_columns = (2, 3)
    elif level == 4:
        bit = r1[book_grid, codes]
        r2 = np.stack([book.r2 for book in books])
        c4 = np.stack([book.c4 for book in books])
        normalized = c4[book_grid, p, bit, r2[book_grid, codes]]
        modifier_columns = None
    else:
        raise ValueError(level)
    if modifiers is not None and modifier_columns is not None:
        modifier = np.asarray(modifiers, np.float32)
        if projection in ("gate", "up"):
            scale = modifier[:, modifier_columns[0]][:, None]
            offset = modifier[:, modifier_columns[1]][:, None]
        else:
            scale = modifier[:, modifier_columns[0]][None, :]
            offset = modifier[:, modifier_columns[1]][None, :]
        normalized = normalized * scale + offset
    return normalized.astype(np.float32) * tensor.expanded_scales


def fit_vector_modifiers(
    tensor: MXFP4LeafTensor,
    books: list[ProgressiveBook],
    selectors: np.ndarray,
    group_size: int,
    projection: str,
    importance: np.ndarray,
) -> np.ndarray:
    """Fit FP16 affine Q2 and Q3 modifiers (eight bytes per vector)."""
    target = LEAF_VALUES[tensor.codes].astype(np.float64)
    scale_weight = tensor.expanded_scales.astype(np.float64) ** 2
    weight = np.broadcast_to(np.asarray(importance, np.float64), target.shape) * scale_weight
    result = np.empty(((tensor.shape[0] if projection in ("gate", "up") else tensor.shape[1]), 4), np.float32)
    for level, columns in ((2, (0, 1)), (3, (2, 3))):
        base = decode_with_books(tensor, books, selectors, group_size, projection, level) / tensor.expanded_scales
        if projection in ("gate", "up"):
            axes = 1
        else:
            axes = 0
        sw = np.sum(weight, axis=axes)
        sx = np.sum(weight * base, axis=axes)
        sy = np.sum(weight * target, axis=axes)
        sxx = np.sum(weight * base * base, axis=axes)
        sxy = np.sum(weight * base * target, axis=axes)
        determinant = sw * sxx - sx * sx
        fitted_scale = np.ones_like(determinant)
        np.divide(sw * sxy - sx * sy, determinant, out=fitted_scale, where=np.abs(determinant) > 1e-20)
        fitted_offset = np.zeros_like(sw)
        np.divide(sy - fitted_scale * sx, sw, out=fitted_offset, where=sw > 0)
        result[:, columns[0]] = np.clip(fitted_scale, 0.5, 1.5)
        result[:, columns[1]] = np.clip(fitted_offset, -1.0, 1.0)
    return result.astype(np.float16).astype(np.float32)


def book_code_arrays(
    codes: np.ndarray, books: list[ProgressiveBook], selectors: np.ndarray | None,
    group_size: int, projection: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(codes, np.uint8)
    if selectors is None:
        ids = np.arange(len(books), dtype=np.int64)
        grid = np.broadcast_to(ids[:, None], values.shape) if projection in ("gate", "up") else np.broadcast_to(ids[None, :], values.shape)
    else:
        ids = np.asarray(selectors, np.int64).reshape(values.shape[0], values.shape[1] // group_size)
        grid = np.repeat(ids, group_size, axis=1)
    parent = np.stack([book.parent for book in books])[grid, values]
    r1 = np.stack([book.r1 for book in books])[grid, values]
    r2 = np.stack([book.r2 for book in books])[grid, values]
    return parent.astype(np.uint8), r1.astype(np.uint8), r2.astype(np.uint8)


def serialize_metadata_file(
    path: Path,
    design_id: int,
    selector_bits: int,
    selectors: dict[str, np.ndarray],
    modifiers: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Write the exact per-expert resident metadata bytes with a fixed header."""
    projection_ids = {"gate": 0, "up": 1, "down": 2}
    body = bytearray()
    entries = []
    for projection in ("gate", "up", "down"):
        while len(body) % 16:
            body.append(0)
        offset = FORMAT_HEADER_BYTES + len(body)
        payload = pack_selector(selectors[projection], selector_bits) if selector_bits else b""
        if modifiers is not None:
            payload += np.asarray(modifiers[projection], dtype="<f2").tobytes()
        body.extend(payload)
        entries.append((projection_ids[projection], 3 if modifiers is not None else 1, offset, len(payload)))
    while len(body) % 16:
        body.append(0)
    header = bytearray(FORMAT_HEADER_BYTES)
    struct.pack_into("<8sHHHHI", header, 0, FORMAT_MAGIC, FORMAT_VERSION, int(design_id), int(selector_bits), len(entries), len(body))
    for index, (projection, kind, offset, length) in enumerate(entries):
        struct.pack_into("<BBHII", header, 20 + 12 * index, projection, kind, 0, offset, length)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(header) + bytes(body))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest, "header_bytes": FORMAT_HEADER_BYTES, "padding_bytes": len(body) - sum(item[3] for item in entries)}


def serialize_books(path: Path, books: Iterable[ProgressiveBook]) -> dict[str, Any]:
    values = list(books)
    payload = b"".join(book.to_bytes() for book in values)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"path": str(path), "bytes": len(payload), "books": len(values), "bytes_per_book": 0 if not values else len(values[0].to_bytes()), "sha256": hashlib.sha256(payload).hexdigest()}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
