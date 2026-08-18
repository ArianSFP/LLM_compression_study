"""Lossless compressed-tensors MXFP4 parsing and embedded 2+1+1-bit trees.

The authoritative leaf format is the compressed-tensors ``mxfp4-pack-quantized``
layout: two adjacent E2M1 leaf codes per byte (even input coordinate in the low
nibble), plus one biased E8M0 exponent per 32 consecutive input coordinates.
This module never quantizes a floating-point matrix.  It only reorganizes exact
leaf codes already present in a checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


BLOCK_SIZE = 32
LEAF_VALUES = np.asarray(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)
UNIQUE_E2M1_VALUES = np.asarray(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
     0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=np.float32,
)


def decode_e8m0(scale_codes: np.ndarray) -> np.ndarray:
    """Decode biased E8M0 exponents used by compressed-tensors MXFP4."""
    codes = np.asarray(scale_codes, dtype=np.uint8)
    return np.exp2(codes.astype(np.int16) - 127).astype(np.float32)


def unpack_leaf_codes(packed: np.ndarray, original_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Unpack adjacent low/high nibbles without changing their code values."""
    values = np.asarray(packed, dtype=np.uint8)
    codes = np.empty((*values.shape[:-1], values.shape[-1] * 2), dtype=np.uint8)
    codes[..., 0::2] = values & np.uint8(0x0F)
    codes[..., 1::2] = values >> np.uint8(4)
    if original_shape is not None and codes.shape != original_shape:
        raise ValueError(f"unpacked shape {codes.shape} != {original_shape}")
    return codes


def pack_leaf_codes(codes: np.ndarray) -> np.ndarray:
    """Pack exact leaf codes using the compressed-tensors nibble convention."""
    values = np.asarray(codes, dtype=np.uint8)
    if values.shape[-1] % 2:
        raise ValueError("MXFP4 input width must be even")
    if np.any(values > 15):
        raise ValueError("MXFP4 leaf code outside [0, 15]")
    return values[..., 0::2] | (values[..., 1::2] << np.uint8(4))


def pack_1bit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint8)
    if np.any(values > 1):
        raise ValueError("bitplane values must be binary")
    return np.packbits(values, axis=-1, bitorder="little")


def unpack_1bit(packed: np.ndarray, count: int) -> np.ndarray:
    return np.unpackbits(np.asarray(packed, dtype=np.uint8), axis=-1, count=count, bitorder="little")


def pack_2bit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint8)
    if values.shape[-1] % 4:
        raise ValueError("two-bit parent stream length must be divisible by four")
    if np.any(values > 3):
        raise ValueError("parent code outside [0, 3]")
    shaped = values.reshape(*values.shape[:-1], -1, 4)
    shifts = np.asarray([0, 2, 4, 6], dtype=np.uint8)
    return np.bitwise_or.reduce(shaped << shifts, axis=-1)


def unpack_2bit(packed: np.ndarray, count: int) -> np.ndarray:
    if count % 4:
        raise ValueError("two-bit parent stream length must be divisible by four")
    source = np.asarray(packed, dtype=np.uint8)
    shifts = np.asarray([0, 2, 4, 6], dtype=np.uint8)
    return ((source[..., None] >> shifts) & np.uint8(3)).reshape(*source.shape[:-1], count)


@dataclass(frozen=True)
class MXFP4LeafTensor:
    """Exact codes/scales for one expert projection."""

    codes: np.ndarray
    scale_codes: np.ndarray
    tensor_name: str
    packed_sha256: str | None = None
    scale_sha256: str | None = None

    def __post_init__(self) -> None:
        codes = np.asarray(self.codes)
        scales = np.asarray(self.scale_codes)
        if codes.ndim != 2 or scales.ndim != 2:
            raise ValueError("codes and scale codes must be matrices")
        if codes.shape[-1] != scales.shape[-1] * BLOCK_SIZE:
            raise ValueError(f"scale geometry mismatch: {codes.shape} vs {scales.shape}")
        if codes.dtype != np.uint8 or scales.dtype != np.uint8:
            raise TypeError("packed leaf codes and E8M0 scales must remain uint8")
        if np.any(codes > 15):
            raise ValueError("invalid MXFP4 leaf code")

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(map(int, self.codes.shape))

    @property
    def scales(self) -> np.ndarray:
        return decode_e8m0(self.scale_codes)

    @property
    def expanded_scales(self) -> np.ndarray:
        return np.repeat(self.scales, BLOCK_SIZE, axis=-1)

    def dequantize(self) -> np.ndarray:
        return self.expanded_scales * LEAF_VALUES[self.codes]

    def exact_streams(self) -> tuple[np.ndarray, np.ndarray]:
        return pack_leaf_codes(self.codes), self.scale_codes.copy()

    def storage_bytes(self) -> dict[str, int | float]:
        weights = int(np.prod(self.shape))
        packed = weights // 2
        scales = weights // BLOCK_SIZE
        return {
            "weights": weights,
            "packed_leaf_bytes": packed,
            "scale_bytes": scales,
            "total_bytes": packed + scales,
            "effective_bpw": 8.0 * (packed + scales) / weights,
        }


@dataclass(frozen=True)
class EmbeddedTree:
    """Balanced bijection from 16 leaf codes to a 2+1+1 hierarchy."""

    leaf_to_parent: np.ndarray
    leaf_to_r1: np.ndarray
    leaf_to_r2: np.ndarray
    c2: np.ndarray
    c3: np.ndarray
    name: str
    centroid_mode: str

    def __post_init__(self) -> None:
        p = np.asarray(self.leaf_to_parent, dtype=np.uint8)
        r1 = np.asarray(self.leaf_to_r1, dtype=np.uint8)
        r2 = np.asarray(self.leaf_to_r2, dtype=np.uint8)
        if p.shape != (16,) or r1.shape != (16,) or r2.shape != (16,):
            raise ValueError("leaf maps must have length 16")
        if np.any(p > 3) or np.any(r1 > 1) or np.any(r2 > 1):
            raise ValueError("invalid hierarchy digit")
        triples = {(int(p[i]), int(r1[i]), int(r2[i])) for i in range(16)}
        if len(triples) != 16:
            raise ValueError("hierarchy is not a leaf-code bijection")
        if np.asarray(self.c2).shape != (4,) or np.asarray(self.c3).shape != (4, 2):
            raise ValueError("invalid reconstruction table shape")

    @property
    def leaf_by_bits(self) -> np.ndarray:
        inverse = np.empty((4, 2, 2), dtype=np.uint8)
        for leaf in range(16):
            inverse[self.leaf_to_parent[leaf], self.leaf_to_r1[leaf], self.leaf_to_r2[leaf]] = leaf
        return inverse

    def bitplanes(self, leaves: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        codes = np.asarray(leaves, dtype=np.uint8)
        return self.leaf_to_parent[codes], self.leaf_to_r1[codes], self.leaf_to_r2[codes]

    def decode_normalized(self, leaves: np.ndarray, level: int) -> np.ndarray:
        p, r1, _ = self.bitplanes(leaves)
        if level == 2:
            return np.asarray(self.c2, dtype=np.float32)[p]
        if level == 3:
            return np.asarray(self.c3, dtype=np.float32)[p, r1]
        if level == 4:
            return LEAF_VALUES[np.asarray(leaves, dtype=np.uint8)]
        raise ValueError(f"unsupported embedded level {level}")

    def decode(self, tensor: MXFP4LeafTensor, level: int) -> np.ndarray:
        return tensor.expanded_scales * self.decode_normalized(tensor.codes, level)

    def reconstruct_leaf_codes(self, leaves: np.ndarray) -> np.ndarray:
        p, r1, r2 = self.bitplanes(leaves)
        return self.leaf_by_bits[p, r1, r2]

    def metadata_bytes(self) -> int:
        # 16 four-bit leaf-map entries plus FP16 reconstruction tables.
        return 8 + (4 + 8) * 2

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "centroid_mode": self.centroid_mode,
            "leaf_to_parent": np.asarray(self.leaf_to_parent).tolist(),
            "leaf_to_r1": np.asarray(self.leaf_to_r1).tolist(),
            "leaf_to_r2": np.asarray(self.leaf_to_r2).tolist(),
            "leaf_by_bits": self.leaf_by_bits.tolist(),
            "c2": np.asarray(self.c2, dtype=np.float32).tolist(),
            "c3": np.asarray(self.c3, dtype=np.float32).tolist(),
            "metadata_bytes": self.metadata_bytes(),
        }


def maps_from_order(order: Iterable[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.asarray(list(order), dtype=np.uint8)
    if order.shape != (16,) or set(map(int, order)) != set(range(16)):
        raise ValueError("tree order must be a permutation of leaf codes")
    parent = np.empty(16, dtype=np.uint8)
    r1 = np.empty(16, dtype=np.uint8)
    r2 = np.empty(16, dtype=np.uint8)
    for position, leaf in enumerate(order):
        parent[leaf] = position // 4
        r1[leaf] = (position % 4) // 2
        r2[leaf] = position % 2
    return parent, r1, r2


NATURAL_ORDER = tuple(range(16))
NUMERIC_ORDER = tuple(np.argsort(LEAF_VALUES, kind="stable").tolist())
MAGNITUDE_INTERLEAVED_ORDER = (0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15)


def tree_orders(kind: str, seed: int, random_candidates: int = 256) -> list[tuple[int, ...]]:
    """Generate a deterministic bounded family of balanced candidate trees."""
    fixed = [NATURAL_ORDER, NUMERIC_ORDER, MAGNITUDE_INTERLEAVED_ORDER]
    if kind == "natural":
        return [NATURAL_ORDER]
    rng = np.random.default_rng(seed)
    found: dict[tuple[int, ...], None] = {value: None for value in fixed}
    positive = np.arange(8, dtype=np.uint8)
    negative = np.arange(8, 16, dtype=np.uint8)
    for _ in range(random_candidates):
        if kind == "sign_preserving":
            order = tuple(np.concatenate((rng.permutation(positive), rng.permutation(negative))).tolist())
        elif kind == "unrestricted":
            order = tuple(rng.permutation(16).tolist())
        else:
            raise ValueError(kind)
        found[order] = None
    return list(found)


def leaf_importance_statistics(
    tensor: MXFP4LeafTensor,
    importance: np.ndarray | None = None,
) -> np.ndarray:
    """Return diagonal functional mass per exact leaf code.

    ``importance`` is per-weight non-negative sensitivity.  Multiplication by
    the squared exact block scale makes the statistic appropriate for fitting
    normalized reconstruction centroids.
    """
    if importance is None:
        weight = np.ones(tensor.shape, dtype=np.float64)
    else:
        weight = np.broadcast_to(np.asarray(importance, dtype=np.float64), tensor.shape)
        if np.any(weight < 0):
            raise ValueError("importance must be non-negative")
    weight = weight * tensor.expanded_scales.astype(np.float64) ** 2
    return np.bincount(tensor.codes.reshape(-1), weights=weight.reshape(-1), minlength=16).astype(np.float64)


def combine_leaf_statistics(items: Iterable[np.ndarray]) -> np.ndarray:
    rows = [np.asarray(value, dtype=np.float64) for value in items]
    if not rows:
        return np.ones(16, dtype=np.float64)
    return np.sum(rows, axis=0)


def _centroid(leaves: np.ndarray, masses: np.ndarray, constrained: bool) -> float:
    weights = masses[leaves]
    if float(weights.sum()) <= 0:
        value = float(np.mean(LEAF_VALUES[leaves]))
    else:
        value = float(np.sum(weights * LEAF_VALUES[leaves]) / weights.sum())
    if constrained:
        value = float(UNIQUE_E2M1_VALUES[np.argmin(np.abs(UNIQUE_E2M1_VALUES - value))])
    return value


def fit_tree(
    order: Iterable[int],
    masses: np.ndarray,
    name: str,
    centroid_mode: str = "free_fp16",
) -> EmbeddedTree:
    masses = np.asarray(masses, dtype=np.float64)
    if masses.shape != (16,):
        raise ValueError("leaf masses must have shape (16,)")
    constrained = centroid_mode == "e2m1"
    if centroid_mode not in ("free_fp16", "e2m1"):
        raise ValueError(centroid_mode)
    parent, r1, r2 = maps_from_order(order)
    c2 = np.empty(4, dtype=np.float32)
    c3 = np.empty((4, 2), dtype=np.float32)
    for p in range(4):
        leaves = np.flatnonzero(parent == p)
        c2[p] = _centroid(leaves, masses, constrained)
        for bit in range(2):
            child = leaves[r1[leaves] == bit]
            c3[p, bit] = _centroid(child, masses, constrained)
    if centroid_mode == "free_fp16":
        # Stored centroids are FP16; fitting must evaluate that exact decode.
        c2 = c2.astype(np.float16).astype(np.float32)
        c3 = c3.astype(np.float16).astype(np.float32)
    return EmbeddedTree(parent, r1, r2, c2, c3, name, centroid_mode)


def tree_distortion(tree: EmbeddedTree, masses: np.ndarray, level: int) -> float:
    approx = tree.decode_normalized(np.arange(16, dtype=np.uint8), level)
    return float(np.sum(np.asarray(masses, dtype=np.float64) * (LEAF_VALUES - approx) ** 2))


def select_tree(
    orders: Iterable[Iterable[int]],
    masses: np.ndarray,
    name: str,
    centroid_mode: str,
    q2_tolerance: float = 1e-3,
) -> EmbeddedTree:
    """Lexicographically select Q2, then Q3 within a Q2 tolerance."""
    candidates = [fit_tree(order, masses, name, centroid_mode) for order in orders]
    q2 = np.asarray([tree_distortion(tree, masses, 2) for tree in candidates])
    best = float(q2.min())
    eligible = np.flatnonzero(q2 <= best * (1.0 + q2_tolerance) + 1e-30)
    q3 = np.asarray([tree_distortion(candidates[i], masses, 3) for i in eligible])
    return candidates[int(eligible[int(np.argmin(q3))])]


def load_compressed_mxfp4_expert(
    checkpoint: Path,
    weight_map: dict[str, str],
    layer: int,
    expert: int,
    projection: str,
) -> MXFP4LeafTensor:
    """Losslessly load one per-expert compressed-tensors MXFP4 projection."""
    from safetensors import safe_open

    if projection not in ("gate", "up", "down"):
        raise ValueError(projection)
    prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}_proj"
    packed_name = prefix + ".weight_packed"
    scale_name = prefix + ".weight_scale"
    if packed_name not in weight_map or scale_name not in weight_map:
        raise KeyError(f"missing exact MXFP4 pair for {prefix}")
    packed_path = checkpoint / weight_map[packed_name]
    scale_path = checkpoint / weight_map[scale_name]
    if not packed_path.is_file() or not scale_path.is_file():
        raise FileNotFoundError((packed_path, scale_path))
    with safe_open(packed_path, framework="np") as handle:
        packed = np.asarray(handle.get_tensor(packed_name))
    with safe_open(scale_path, framework="np") as handle:
        scales = np.asarray(handle.get_tensor(scale_name))
    if packed.dtype != np.uint8 or scales.dtype != np.uint8:
        raise TypeError(f"non-MXFP4 storage for {prefix}: {packed.dtype}, {scales.dtype}")
    codes = unpack_leaf_codes(packed)
    expected = (512, 2048) if projection in ("gate", "up") else (2048, 512)
    if codes.shape != expected or scales.shape != (expected[0], expected[1] // BLOCK_SIZE):
        raise ValueError(f"unexpected compressed-tensors geometry for {prefix}: {codes.shape}, {scales.shape}")
    return MXFP4LeafTensor(codes.copy(), scales.copy(), prefix)


def audit_checkpoint(checkpoint: Path, layers: Iterable[int]) -> dict[str, Any]:
    """Fail closed unless every routed expert projection is exact packed MXFP4."""
    index_path = checkpoint / "model.safetensors.index.json"
    config_path = checkpoint / "config.json"
    index = json.loads(index_path.read_text())
    config = json.loads(config_path.read_text())
    weight_map = index["weight_map"]
    quant = config.get("quantization_config", {})
    errors: list[str] = []
    if quant.get("quant_method") != "compressed-tensors":
        errors.append("quant_method is not compressed-tensors")
    if quant.get("format") != "mxfp4-pack-quantized":
        errors.append("format is not mxfp4-pack-quantized")
    group = next(iter(quant.get("config_groups", {}).values()), {})
    weights = group.get("weights", {})
    if weights.get("num_bits") != 4 or weights.get("group_size") != BLOCK_SIZE or weights.get("type") != "float":
        errors.append(f"unexpected weight quantization geometry: {weights}")
    missing: list[str] = []
    required = 0
    for layer, expert, projection in itertools.product(map(int, layers), range(256), ("gate", "up", "down")):
        prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}_proj"
        for suffix in ("weight_packed", "weight_scale"):
            required += 1
            key = prefix + "." + suffix
            if key not in weight_map:
                missing.append(key)
    if missing:
        errors.append(f"missing {len(missing)} packed tensors")
    return {
        "passed": not errors,
        "checkpoint": str(checkpoint),
        "format": quant.get("format"),
        "group_size": weights.get("group_size"),
        "layers": list(map(int, layers)),
        "experts_per_layer": 256,
        "required_tensor_entries": required,
        "missing_count": len(missing),
        "missing_sample": missing[:20],
        "errors": errors,
    }
