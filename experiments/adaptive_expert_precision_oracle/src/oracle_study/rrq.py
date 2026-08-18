"""Deterministic recurrent residual quantization and exact serialization.

This is an RRQ implementation for the oracle study, not a bit-exact
reproduction of arXiv:2608.04048v1. Each stage owns independent codes and
parameters; the next residual is constructed from the stored-parameter decode.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Sequence

import numpy as np


CODEBOOKS = {
    "midrise": np.asarray([-3.0, -1.0, 1.0, 3.0], dtype=np.float32),
    "signed_zero": np.asarray([-2.0, -1.0, 0.0, 1.0], dtype=np.float32),
}


@dataclass(frozen=True)
class QuantizerSpec:
    name: str
    group_size: int
    scale_storage: str = "fp16"


@dataclass
class Int2Encoding:
    packed_codes: np.ndarray
    stored_scales: np.ndarray | None
    fitted_scales: np.ndarray | None
    zero_points: np.ndarray | None
    stored_centroids: np.ndarray | None
    fitted_centroids: np.ndarray | None
    reconstruction: np.ndarray
    shape: tuple[int, int]
    group_size: int
    quantizer: str
    scale_storage: str

    @property
    def groups(self) -> int:
        return self.shape[0] * math.ceil(self.shape[1] / self.group_size)

    def stream_bytes(self) -> dict[str, int]:
        return {
            "header": 64,
            "codes": int(self.packed_codes.nbytes),
            "scales": 0 if self.stored_scales is None else int(self.stored_scales.nbytes),
            "zero_points": 0 if self.zero_points is None else int(self.zero_points.nbytes),
            "centroids": 0 if self.stored_centroids is None else int(self.stored_centroids.nbytes),
        }

    def serialized_bytes(self) -> int:
        return int(sum(self.stream_bytes().values()))

    def physical_read_bytes(self, page_size: int) -> int:
        # Header/offset metadata is local at execution; it still counts toward
        # external capacity, but code/parameter streams are the fetched bytes.
        return int(sum(
            math.ceil(value / page_size) * page_size
            for key, value in self.stream_bytes().items()
            if key != "header" and value
        ))


@dataclass
class RRQSeries:
    base: np.ndarray
    stages: list[Int2Encoding]
    residual_norms: list[float]

    def prefix(self, depth: int) -> np.ndarray:
        if depth < 0 or depth > len(self.stages):
            raise ValueError(depth)
        value = np.asarray(self.base, dtype=np.float32).copy()
        for stage in self.stages[:depth]:
            value += stage.reconstruction
        return value


def pack_q2_codes(codes: np.ndarray) -> np.ndarray:
    flat = np.asarray(codes, dtype=np.uint8).reshape(-1)
    if np.any(flat > 3):
        raise ValueError("two-bit codes must lie in [0, 3]")
    padded = np.zeros(math.ceil(len(flat) / 4) * 4, dtype=np.uint8)
    padded[: len(flat)] = flat
    grouped = padded.reshape(-1, 4)
    return (grouped[:, 0] | (grouped[:, 1] << 2) | (grouped[:, 2] << 4) | (grouped[:, 3] << 6)).astype(np.uint8)


def unpack_q2_codes(packed: np.ndarray, count: int) -> np.ndarray:
    source = np.asarray(packed, dtype=np.uint8).reshape(-1)
    result = np.empty(len(source) * 4, dtype=np.uint8)
    for offset in range(4):
        result[offset::4] = (source >> (2 * offset)) & 3
    return result[:count]


def _float32_to_bf16(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def _bf16_to_float32(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def _store_float(values: np.ndarray, storage: str) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(values, dtype=np.float32)
    if storage == "fp16":
        stored = source.astype(np.float16)
        return stored, stored.astype(np.float32)
    if storage == "bf16":
        stored = _float32_to_bf16(source)
        return stored, _bf16_to_float32(stored)
    if storage == "fp32":
        return source.copy(), source.copy()
    raise ValueError(storage)


def _decoded_float(stored: np.ndarray, storage: str) -> np.ndarray:
    if storage == "fp16":
        return np.asarray(stored, dtype=np.float16).astype(np.float32)
    if storage == "bf16":
        return _bf16_to_float32(stored)
    if storage == "fp32":
        return np.asarray(stored, dtype=np.float32)
    raise ValueError(storage)


def _group_matrix(matrix: np.ndarray, group_size: int) -> tuple[np.ndarray, int]:
    value = np.asarray(matrix, dtype=np.float32)
    if value.ndim != 2:
        raise ValueError("RRQ target must be a matrix")
    out, width = value.shape
    groups = math.ceil(width / group_size)
    padded = np.zeros((out, groups * group_size), dtype=np.float32)
    padded[:, :width] = value
    return padded.reshape(out * groups, group_size), width


def _ungroup_matrix(groups: np.ndarray, shape: tuple[int, int], group_size: int) -> np.ndarray:
    out, width = shape
    return np.asarray(groups, dtype=np.float32).reshape(out, -1)[:, :width].copy()


def _group_weights(width: int, out: int, group_size: int, moments: np.ndarray | None) -> np.ndarray:
    groups = math.ceil(width / group_size)
    padded = np.zeros(groups * group_size, dtype=np.float64)
    if moments is None:
        padded[:width] = 1.0
    else:
        supplied = np.asarray(moments, dtype=np.float64)
        if supplied.shape != (width,) or np.any(supplied < 0):
            raise ValueError("bad activation moments")
        padded[:width] = supplied
    return np.broadcast_to(padded.reshape(1, groups, group_size), (out, groups, group_size)).reshape(out * groups, group_size)


def _initial_scales(values: np.ndarray, weights: np.ndarray, codebook: np.ndarray) -> list[np.ndarray]:
    maximum = np.max(np.abs(values), axis=1)
    mean = np.sum(np.abs(values) * weights, axis=1) / np.maximum(np.sum(weights, axis=1), 1e-30)
    rms = np.sqrt(np.sum(values * values * weights, axis=1) / np.maximum(np.sum(weights, axis=1), 1e-30))
    largest = max(float(np.max(np.abs(codebook))), 1.0)
    return [
        np.maximum(maximum / largest, 1e-12),
        np.maximum(maximum / 2.0, 1e-12),
        np.maximum(mean, 1e-12),
        np.maximum(rms / math.sqrt(2.0), 1e-12),
    ]


def _fit_scaled_codebook(values: np.ndarray, weights: np.ndarray, codebook: np.ndarray, optimize: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    best_error = np.full(len(values), np.inf, dtype=np.float64)
    best_scale = np.ones(len(values), dtype=np.float32)
    best_codes = np.zeros_like(values, dtype=np.uint8)
    starts = _initial_scales(values, weights, codebook)
    if not optimize:
        starts = starts[:1]
    for initial in starts:
        scale = initial.astype(np.float64)
        for _ in range(8 if optimize else 1):
            distance = np.abs(values[:, :, None] - scale[:, None, None] * codebook[None, None, :])
            codes = np.argmin(distance, axis=2).astype(np.uint8)
            chosen = codebook[codes]
            if optimize:
                scale = np.maximum(
                    np.sum(weights * values * chosen, axis=1) / np.maximum(np.sum(weights * chosen * chosen, axis=1), 1e-30),
                    1e-12,
                )
        recon = codebook[codes] * scale[:, None]
        error = np.sum(weights * (values - recon) ** 2, axis=1)
        better = error < best_error
        best_error[better] = error[better]
        best_scale[better] = scale[better].astype(np.float32)
        best_codes[better] = codes[better]
    return best_codes, best_scale, best_error


def _fit_affine(values: np.ndarray, weights: np.ndarray, optimize: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not optimize:
        minimum, maximum = np.min(values, axis=1), np.max(values, axis=1)
        scale = np.maximum((maximum - minimum) / 3.0, 1e-12)
        zero = np.clip(np.rint(-minimum / scale), 0, 3).astype(np.uint8)
        codes = np.clip(np.rint(values / scale[:, None]) + zero[:, None], 0, 3).astype(np.uint8)
        recon = (codes.astype(np.float32) - zero[:, None]) * scale[:, None]
        error = np.sum(weights * (values - recon) ** 2, axis=1)
        return codes, scale.astype(np.float32), zero, error
    best_error = np.full(len(values), np.inf, dtype=np.float64)
    best_scale = np.ones(len(values), dtype=np.float32)
    best_zero = np.zeros(len(values), dtype=np.uint8)
    best_codes = np.zeros_like(values, dtype=np.uint8)
    for zero in range(4):
        codebook = np.arange(4, dtype=np.float32) - float(zero)
        codes, scale, error = _fit_scaled_codebook(values, weights, codebook, optimize=True)
        better = error < best_error
        best_error[better] = error[better]
        best_scale[better] = scale[better]
        best_zero[better] = zero
        best_codes[better] = codes[better]
    return best_codes, best_scale, best_zero, best_error


def _fit_centroids(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroids = np.quantile(values, [0.0, 1.0 / 3, 2.0 / 3, 1.0], axis=1).T.astype(np.float32)
    codes = np.zeros_like(values, dtype=np.uint8)
    for _ in range(32):
        new_codes = np.argmin(np.abs(values[:, :, None] - centroids[:, None, :]), axis=2).astype(np.uint8)
        changed = not np.array_equal(new_codes, codes)
        codes = new_codes
        for index in range(4):
            selected = codes == index
            numerator = np.sum(np.where(selected, weights * values, 0.0), axis=1)
            denominator = np.sum(np.where(selected, weights, 0.0), axis=1)
            nonempty = denominator > 0
            centroids[nonempty, index] = (numerator[nonempty] / denominator[nonempty]).astype(np.float32)
        centroids.sort(axis=1)
        if not changed:
            break
    codes = np.argmin(np.abs(values[:, :, None] - centroids[:, None, :]), axis=2).astype(np.uint8)
    return codes, centroids


def quantize_int2_stage(
    target: np.ndarray,
    group_size: int,
    mode: str,
    activation_second_moment: np.ndarray | None = None,
    scale_storage: str = "fp16",
) -> Int2Encoding:
    matrix = np.asarray(target, dtype=np.float32)
    grouped, width = _group_matrix(matrix, group_size)
    weights = _group_weights(width, matrix.shape[0], group_size, activation_second_moment)
    zero_points = None
    stored_centroids = fitted_centroids = None
    fitted_scales = stored_scales = None
    if mode in ("midrise_mse", "midrise_rtn", "signed_zero_mse", "signed_zero_rtn"):
        family = "midrise" if mode.startswith("midrise") else "signed_zero"
        codes, fitted_scales, _ = _fit_scaled_codebook(grouped, weights, CODEBOOKS[family], mode.endswith("mse"))
        stored_scales, decoded = _store_float(fitted_scales, scale_storage)
        reconstruction = CODEBOOKS[family][codes] * decoded[:, None]
    elif mode in ("affine_mse", "affine_rtn"):
        codes, fitted_scales, zero_points, _ = _fit_affine(grouped, weights, mode.endswith("mse"))
        stored_scales, decoded = _store_float(fitted_scales, scale_storage)
        reconstruction = (codes.astype(np.float32) - zero_points[:, None]) * decoded[:, None]
    elif mode == "centroid4":
        codes, fitted_centroids = _fit_centroids(grouped, weights)
        stored_centroids, decoded = _store_float(fitted_centroids, scale_storage)
        reconstruction = np.take_along_axis(decoded, codes, axis=1)
    else:
        raise ValueError(mode)
    return Int2Encoding(
        pack_q2_codes(codes.reshape(-1)), stored_scales, fitted_scales, zero_points,
        stored_centroids, fitted_centroids,
        _ungroup_matrix(reconstruction, matrix.shape, group_size), tuple(matrix.shape),
        int(group_size), mode, scale_storage,
    )


def encode_legacy_q2(
    weights: np.ndarray,
    group_size: int = 64,
    activation_second_moment: np.ndarray | None = None,
    scale_storage: str = "fp16",
) -> Int2Encoding:
    """Encode the repository's activation-aware ±1/±3 Q2 exactly."""
    matrix = np.asarray(weights, dtype=np.float32)
    grouped, width = _group_matrix(matrix, group_size)
    importance = _group_weights(width, matrix.shape[0], group_size, activation_second_moment)
    scale = np.maximum(np.sum(np.abs(grouped) * importance, axis=1) / np.maximum(np.sum(importance, axis=1), 1e-30), 1e-12)
    values = np.ones_like(grouped)
    for _ in range(8):
        magnitude = np.where(np.abs(grouped) < 2.0 * scale[:, None], 1.0, 3.0)
        values = np.where(grouped >= 0, magnitude, -magnitude)
        scale = np.maximum(
            np.sum(grouped * values * importance, axis=1) / np.maximum(np.sum(values * values * importance, axis=1), 1e-30),
            1e-12,
        )
    codes = np.searchsorted(CODEBOOKS["midrise"], values).astype(np.uint8)
    stored, decoded = _store_float(scale.astype(np.float32), scale_storage)
    reconstruction = CODEBOOKS["midrise"][codes] * decoded[:, None]
    return Int2Encoding(
        pack_q2_codes(codes.reshape(-1)), stored, scale.astype(np.float32), None, None, None,
        _ungroup_matrix(reconstruction, matrix.shape, group_size), tuple(matrix.shape),
        int(group_size), "legacy_q2", scale_storage,
    )


def decode_encoding(encoding: Int2Encoding) -> np.ndarray:
    # Codes include the row-local padding introduced when the input width is
    # not divisible by the group size.  Decode that exact serialized stream,
    # then strip padding in _ungroup_matrix.
    count = encoding.groups * encoding.group_size
    codes = unpack_q2_codes(encoding.packed_codes, count).reshape(encoding.groups, encoding.group_size)
    if encoding.stored_centroids is not None:
        centroids = _decoded_float(encoding.stored_centroids, encoding.scale_storage)
        return _ungroup_matrix(np.take_along_axis(centroids, codes, axis=1), encoding.shape, encoding.group_size)
    scales = _decoded_float(encoding.stored_scales, encoding.scale_storage)
    if encoding.zero_points is not None:
        recon = (codes.astype(np.float32) - encoding.zero_points[:, None]) * scales[:, None]
    else:
        family = "midrise" if encoding.quantizer in ("legacy_q2", "midrise_mse", "midrise_rtn") else "signed_zero"
        recon = CODEBOOKS[family][codes] * scales[:, None]
    return _ungroup_matrix(recon, encoding.shape, encoding.group_size)


def build_rrq_series(
    base: np.ndarray,
    reference: np.ndarray,
    specs: Sequence[QuantizerSpec],
    activation_second_moment: np.ndarray | None = None,
) -> RRQSeries:
    current = np.asarray(base, dtype=np.float32).copy()
    target = np.asarray(reference, dtype=np.float32)
    residual = target - current
    norms = [float(np.linalg.norm(residual.astype(np.float64)))]
    stages: list[Int2Encoding] = []
    for spec in specs:
        stage = quantize_int2_stage(residual, spec.group_size, spec.name, activation_second_moment, spec.scale_storage)
        decoded = decode_encoding(stage)
        np.testing.assert_array_equal(decoded, stage.reconstruction)
        stages.append(stage)
        current += decoded
        residual = target - current
        norms.append(float(np.linalg.norm(residual.astype(np.float64))))
    return RRQSeries(np.asarray(base, dtype=np.float32), stages, norms)


def build_atom_rrq_series(
    target_atoms: np.ndarray,
    specs: Sequence[QuantizerSpec],
) -> RRQSeries:
    """RRQ a [output, atom] dictionary in independently fetchable packets."""
    target = np.asarray(target_atoms, dtype=np.float32).T.copy()
    return build_rrq_series(np.zeros_like(target), target, specs, None)


def atom_stage_matrix(series: RRQSeries, stage: int) -> np.ndarray:
    """Return one decoded atom-RRQ stage as [output, atom]."""
    if stage < 1 or stage > len(series.stages):
        raise ValueError(stage)
    return series.stages[stage - 1].reconstruction.T.copy()


def atom_packet_bytes(encoding: Int2Encoding, page_size: int) -> tuple[int, int]:
    """Logical and page-aligned bytes for one independently fetchable atom."""
    atoms, width = encoding.shape
    groups = math.ceil(width / encoding.group_size)
    code_bytes = math.ceil(width / 4)
    scale_item = 0 if encoding.stored_scales is None else encoding.stored_scales.dtype.itemsize
    centroid_item = 0 if encoding.stored_centroids is None else encoding.stored_centroids.dtype.itemsize
    logical = code_bytes + groups * scale_item
    logical += groups if encoding.zero_points is not None else 0
    logical += groups * 4 * centroid_item
    expected = encoding.serialized_bytes() - encoding.stream_bytes()["header"]
    if logical * atoms != expected:
        raise AssertionError((logical, atoms, expected, encoding.stream_bytes()))
    return int(logical), int(math.ceil(logical / page_size) * page_size)


def serialize_encoding(root: Path, label: str, encoding: Int2Encoding) -> dict[str, object]:
    directory = Path(root) / label
    directory.mkdir(parents=True, exist_ok=True)
    header = struct.pack(
        "<8sIIIIII32s",
        b"ORCLRRQ1", encoding.shape[0], encoding.shape[1], encoding.group_size,
        len(encoding.packed_codes), encoding.groups, 1,
        encoding.quantizer.encode()[:32].ljust(32, b"\0"),
    )
    if len(header) != 64:
        raise AssertionError(len(header))
    streams: dict[str, bytes] = {"header": header, "codes": encoding.packed_codes.tobytes(order="C")}
    if encoding.stored_scales is not None:
        streams["scales"] = encoding.stored_scales.tobytes(order="C")
    if encoding.zero_points is not None:
        streams["zero_points"] = encoding.zero_points.tobytes(order="C")
    if encoding.stored_centroids is not None:
        streams["centroids"] = encoding.stored_centroids.tobytes(order="C")
    files = {}
    expected = encoding.stream_bytes()
    for name, payload in streams.items():
        path = directory / f"{name}.bin"
        path.write_bytes(payload)
        files[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(payload).hexdigest()}
        if path.stat().st_size != expected[name]:
            raise AssertionError((name, path.stat().st_size, expected[name]))
    manifest = {
        "label": label, "shape": list(encoding.shape), "group_size": encoding.group_size,
        "quantizer": encoding.quantizer, "scale_storage": encoding.scale_storage,
        "files": files, "serialized_bytes": sum(v["bytes"] for v in files.values()),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def scale_diagnostics(encoding: Int2Encoding) -> dict[str, float | int]:
    original = encoding.fitted_scales if encoding.fitted_scales is not None else encoding.fitted_centroids.reshape(-1)
    stored = encoding.stored_scales if encoding.stored_scales is not None else encoding.stored_centroids
    decoded = _decoded_float(stored, encoding.scale_storage).reshape(-1)
    original = np.asarray(original, dtype=np.float32).reshape(-1)
    tiny = np.float32(2 ** -14) if encoding.scale_storage == "fp16" else np.float32(2 ** -126)
    relative = np.abs(decoded - original) / np.maximum(np.abs(original), 1e-30)
    return {
        "minimum": float(decoded.min()), "maximum": float(decoded.max()),
        "p1": float(np.quantile(decoded, .01)), "median": float(np.median(decoded)), "p99": float(np.quantile(decoded, .99)),
        "zeros": int(np.sum(decoded == 0)), "subnormals": int(np.sum((np.abs(decoded) > 0) & (np.abs(decoded) < tiny))),
        "median_relative_rounding_error": float(np.median(relative)), "maximum_relative_rounding_error": float(np.max(relative)),
    }
