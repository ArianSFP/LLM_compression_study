"""Correctness references for structured set-utility selector metadata.

This module deliberately favors deterministic, inspectable implementations
over throughput.  It provides three independent building blocks for follow-up
allocator studies without changing the locked Q2->Q3->Q4 codec:

* activation-second-moment-weighted residual product quantization in 32-value
  input blocks, with layer-shared codebooks and physically packed 4-bit codes;
* row-scaled INT2, ternary, and binary-sign metadata encodings; and
* fixed-cardinality straight-through masks and the exact proxy qmetric set
  objective used to distill coherent unit selections.

For normalized residual code ``z``, decoded E8M0 scale ``s``, and block
second moment ``M``, the PQ fit minimizes exactly
``sum_i s_i**2 * (z_i-c_i)' M (z_i-c_i)`` within each input block.  ``M`` is
the uncentred moment ``E[x x']`` appearing in decoded response distortion.
The block-local objective deliberately omits cross-block activation
covariance, so it is not the complete-row response objective when blocks are
correlated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch


PQ_BLOCK_SIZE = 32
PQ_CODEBOOK_SIZE = 16
RowEncoding = Literal["int2", "ternary", "sign_scale"]
Reduction = Literal["none", "mean", "sum"]
BlockScaleProvenance = Literal["none", "new_e8m0_metadata", "locked_resident_e8m0"]


def _require_finite(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains a non-finite value")
    return array


def _e8m0_codes_and_scales(block_scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate decoded finite E8M0 values and recover their exact bytes."""
    scales = _require_finite(np.asarray(block_scales, dtype=np.float64), "block scales")
    if np.any(scales <= 0.0):
        raise ValueError("E8M0 block scales must be strictly positive")
    exponent = np.rint(np.log2(scales)).astype(np.int16)
    if np.any(exponent < -127) or np.any(exponent > 127):
        raise ValueError("block scale lies outside the finite E8M0 exponent range")
    decoded = np.exp2(exponent).astype(np.float32)
    if not np.array_equal(scales.astype(np.float32), decoded):
        raise ValueError("block scales must be exact decoded E8M0 powers of two")
    return (exponent + 127).astype(np.uint8), decoded


def _normalized_weight_blocks(
    weights: np.ndarray, block_scales: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    blocks = weights.shape[-1] // PQ_BLOCK_SIZE
    vectors = np.asarray(weights, dtype=np.float64).reshape(-1, blocks, PQ_BLOCK_SIZE)
    if block_scales is None:
        return vectors, None, None
    expected = (*weights.shape[:-1], blocks)
    supplied = np.asarray(block_scales)
    if supplied.shape != expected:
        raise ValueError(f"block scales must have shape {expected}")
    codes, decoded = _e8m0_codes_and_scales(supplied)
    flat_scales = decoded.reshape(-1, blocks).astype(np.float64)
    return vectors / flat_scales[:, :, None], codes, decoded


def _pack_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    """Pack unsigned fixed-width codes, least-significant field first."""
    if bits not in (1, 2, 4) or 8 % bits:
        raise ValueError("bits must be one of 1, 2, or 4")
    values = np.asarray(codes, dtype=np.uint8).reshape(-1)
    limit = 1 << bits
    if np.any(values >= limit):
        raise ValueError(f"a {bits}-bit code exceeds its representable range")
    per_byte = 8 // bits
    packed = np.zeros((values.size + per_byte - 1) // per_byte, dtype=np.uint8)
    if values.size:
        positions = np.arange(values.size, dtype=np.int64)
        shifted = values.astype(np.uint16) << ((positions % per_byte) * bits)
        np.bitwise_or.at(packed, positions // per_byte, shifted.astype(np.uint8))
    return packed


def _unpack_codes(packed: np.ndarray, count: int, bits: int) -> np.ndarray:
    if bits not in (1, 2, 4) or 8 % bits:
        raise ValueError("bits must be one of 1, 2, or 4")
    payload = np.asarray(packed, dtype=np.uint8).reshape(-1)
    per_byte = 8 // bits
    expected = (int(count) + per_byte - 1) // per_byte
    if payload.size != expected:
        raise ValueError(f"packed payload has {payload.size} bytes; expected {expected}")
    positions = np.arange(int(count), dtype=np.int64)
    return ((payload[positions // per_byte] >> ((positions % per_byte) * bits)) & ((1 << bits) - 1)).astype(
        np.uint8, copy=False,
    )


def _weighted_squared_distances(
    vectors: np.ndarray, centroids: np.ndarray, second_moment: np.ndarray,
) -> np.ndarray:
    difference = np.asarray(vectors, np.float64)[:, None, :] - np.asarray(centroids, np.float64)[None, :, :]
    distances = np.einsum("nkd,de,nke->nk", difference, second_moment, difference, optimize=True)
    return np.maximum(distances, 0.0)


def _assign_codewords(
    vectors: np.ndarray, centroids: np.ndarray, second_moment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    distances = _weighted_squared_distances(vectors, centroids, second_moment)
    codes = np.argmin(distances, axis=1).astype(np.uint8)
    return codes, distances[np.arange(len(vectors)), codes]


def _deterministic_weighted_kmeans(
    vectors: np.ndarray,
    second_moment: np.ndarray,
    *,
    sample_weights: np.ndarray | None = None,
    codebook_size: int,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted Lloyd fit with deterministic farthest-first initialization.

    Codeword zero is fixed at zero.  Besides making an all-zero residual exact,
    this guarantees that an optional second additive stage can decline to
    change any vector.
    """
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != PQ_BLOCK_SIZE:
        raise ValueError(f"vectors must have shape [count, {PQ_BLOCK_SIZE}]")
    if values.shape[0] < codebook_size - 1:
        raise ValueError("at least codebook_size - 1 residual vectors are required")
    weights = np.ones(values.shape[0], dtype=np.float64) if sample_weights is None else np.asarray(
        sample_weights, dtype=np.float64,
    )
    if weights.shape != (values.shape[0],) or not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("sample weights must be finite, positive, and aligned with residual vectors")
    centroids = np.zeros((codebook_size, PQ_BLOCK_SIZE), dtype=np.float64)
    chosen = np.zeros(values.shape[0], dtype=bool)
    minimum_distance = _weighted_squared_distances(values, centroids[:1], second_moment)[:, 0]
    for slot in range(1, codebook_size):
        candidate_score = minimum_distance * weights
        candidate_score[chosen] = -np.inf
        index = int(np.argmax(candidate_score))
        chosen[index] = True
        centroids[slot] = values[index]
        distance = _weighted_squared_distances(values, centroids[slot:slot + 1], second_moment)[:, 0]
        minimum_distance = np.minimum(minimum_distance, distance)

    previous: np.ndarray | None = None
    for _ in range(max_iterations):
        assignment, _ = _assign_codewords(values, centroids, second_moment)
        if previous is not None and np.array_equal(assignment, previous):
            break
        previous = assignment
        # Zero is a deliberately fixed codeword; all other non-empty clusters
        # use the scale-squared mean, which is optimal for a common PSD metric.
        for code in range(1, codebook_size):
            selected = assignment == code
            members = values[selected]
            if len(members):
                centroids[code] = np.average(members, axis=0, weights=weights[selected])
        centroids[0].fill(0.0)
    assignment, _ = _assign_codewords(values, centroids, second_moment)
    return centroids.astype(np.float32), assignment


@dataclass(frozen=True)
class BlockPQCodebooks:
    """Layer-shared additive residual codebooks.

    ``centroids`` has shape ``[stage, input_block, 16, 32]``.  The activation
    second moments are fit-only metadata and are not charged to deployment
    storage.
    """

    centroids: np.ndarray
    activation_second_moment: np.ndarray
    normalized_by_block_scale: bool = False

    def __post_init__(self) -> None:
        value = _require_finite(np.asarray(self.centroids), "PQ centroids")
        moment = _require_finite(np.asarray(self.activation_second_moment), "activation second moment")
        if value.ndim != 4 or value.shape[0] not in (1, 2):
            raise ValueError("centroids must have one or two additive stages")
        if value.shape[2:] != (PQ_CODEBOOK_SIZE, PQ_BLOCK_SIZE):
            raise ValueError("centroids must have shape [stage, block, 16, 32]")
        if moment.shape != (value.shape[1], PQ_BLOCK_SIZE, PQ_BLOCK_SIZE):
            raise ValueError("activation second moments do not match the codebook blocks")
        if not np.allclose(value[:, :, 0], 0.0):
            raise ValueError("PQ codeword zero must be the additive no-op")

    @property
    def stages(self) -> int:
        return int(np.asarray(self.centroids).shape[0])

    @property
    def blocks(self) -> int:
        return int(np.asarray(self.centroids).shape[1])

    @property
    def input_width(self) -> int:
        return self.blocks * PQ_BLOCK_SIZE


def fit_activation_weighted_block_pq(
    residual_weights: np.ndarray,
    training_activations: np.ndarray,
    *,
    block_scales: np.ndarray | None = None,
    stages: int = 1,
    max_iterations: int = 50,
) -> BlockPQCodebooks:
    """Fit deterministic layer-shared 32-D response-weighted codebooks.

    All leading residual-weight dimensions are pooled, so passing stacked
    gate/up residuals from many experts produces one layer-shared codebook.
    One codebook is retained per input block because activation statistics and
    MXFP4 block positions differ across the input dimension.  When decoded
    E8M0 ``block_scales`` are supplied, clustering operates on residual codes
    normalized by those scales.  This avoids confounding code shape with the
    locked codec's per-row/block magnitude.  Farthest-first seeds and
    Lloyd centroids use per-row ``scale**2`` weights, while assignments remain
    metric-nearest because that positive weight is constant across a row's
    candidates.  Thus each block minimizes exact decoded response distortion
    ``sum_i scale_i**2 * error_i' M_block error_i``.  Independent block
    moments omit cross-block covariance by construction.
    """
    weights = _require_finite(np.asarray(residual_weights, dtype=np.float64), "residual weights")
    activation = _require_finite(np.asarray(training_activations, dtype=np.float64), "training activations")
    if weights.ndim < 2 or weights.shape[-1] % PQ_BLOCK_SIZE:
        raise ValueError(f"residual weight width must be divisible by {PQ_BLOCK_SIZE}")
    if activation.ndim != 2 or activation.shape[1] != weights.shape[-1] or not len(activation):
        raise ValueError("training activations must have shape [sample, residual input width]")
    if stages not in (1, 2):
        raise ValueError("stages must be one or two")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    blocks = weights.shape[-1] // PQ_BLOCK_SIZE
    vectors, _, decoded_scales = _normalized_weight_blocks(weights, block_scales)
    sample_weights = np.ones(vectors.shape[:2], dtype=np.float64)
    if decoded_scales is not None:
        sample_weights = np.square(decoded_scales.reshape(vectors.shape[:2]).astype(np.float64))
    xblocks = activation.reshape(len(activation), blocks, PQ_BLOCK_SIZE)
    moments = np.einsum("nbd,nbe->bde", xblocks, xblocks, optimize=True) / float(len(activation))
    moments = (moments + np.swapaxes(moments, 1, 2)) * 0.5
    residual = vectors.copy()
    centroids = np.zeros((stages, blocks, PQ_CODEBOOK_SIZE, PQ_BLOCK_SIZE), dtype=np.float32)
    for stage in range(stages):
        for block in range(blocks):
            fitted, codes = _deterministic_weighted_kmeans(
                residual[:, block], moments[block],
                sample_weights=sample_weights[:, block],
                codebook_size=PQ_CODEBOOK_SIZE, max_iterations=max_iterations,
            )
            centroids[stage, block] = fitted
            residual[:, block] -= fitted[codes]
    return BlockPQCodebooks(
        centroids=centroids,
        activation_second_moment=moments,
        normalized_by_block_scale=block_scales is not None,
    )


@dataclass(frozen=True)
class EncodedBlockResidualPQ:
    """FP16 shared codebooks and nibble-packed per-matrix code indices."""

    stored_codebooks: np.ndarray
    packed_codes: np.ndarray
    code_shape: tuple[int, int, int]
    weight_shape: tuple[int, ...]
    block_scale_codes: np.ndarray | None = None
    block_scale_provenance: BlockScaleProvenance = "none"

    def __post_init__(self) -> None:
        codebooks = np.asarray(self.stored_codebooks)
        packed = np.asarray(self.packed_codes)
        if codebooks.dtype != np.float16 or codebooks.ndim != 4:
            raise ValueError("stored PQ codebooks must be a four-dimensional FP16 array")
        if codebooks.shape[0] not in (1, 2) or codebooks.shape[2:] != (PQ_CODEBOOK_SIZE, PQ_BLOCK_SIZE):
            raise ValueError("stored PQ codebooks must have shape [stage, block, 16, 32]")
        if packed.dtype != np.uint8 or packed.ndim != 1:
            raise ValueError("packed PQ codes must be a flat uint8 array")
        if len(self.weight_shape) < 2 or self.weight_shape[-1] != codebooks.shape[1] * PQ_BLOCK_SIZE:
            raise ValueError("weight shape and PQ codebook width disagree")
        vectors = int(np.prod(self.weight_shape[:-1], dtype=np.int64))
        expected_shape = (vectors, codebooks.shape[1], codebooks.shape[0])
        if tuple(self.code_shape) != expected_shape:
            raise ValueError(f"code shape must be {expected_shape}")
        if packed.size != (int(np.prod(expected_shape, dtype=np.int64)) + 1) // 2:
            raise ValueError("packed PQ code payload has the wrong byte length")
        if not np.all(np.isfinite(codebooks)):
            raise ValueError("stored PQ codebooks contain a non-finite value")
        if self.block_scale_provenance not in ("none", "new_e8m0_metadata", "locked_resident_e8m0"):
            raise ValueError(f"unsupported block-scale provenance: {self.block_scale_provenance}")
        if self.block_scale_codes is None:
            if self.block_scale_provenance != "none":
                raise ValueError("block-scale provenance requires E8M0 scale codes")
        else:
            scale_codes = np.asarray(self.block_scale_codes)
            expected_scales = (*self.weight_shape[:-1], codebooks.shape[1])
            if scale_codes.dtype != np.uint8 or scale_codes.shape != expected_scales:
                raise ValueError(f"E8M0 scale codes must be uint8 with shape {expected_scales}")
            if self.block_scale_provenance == "none":
                raise ValueError("E8M0 scale codes require explicit provenance")

    @property
    def layer_shared_codebook_bytes(self) -> int:
        return int(np.asarray(self.stored_codebooks).nbytes)

    @property
    def encoded_index_bytes(self) -> int:
        return int(np.asarray(self.packed_codes).nbytes)

    @property
    def block_scale_bytes_read(self) -> int:
        return 0 if self.block_scale_codes is None else int(np.asarray(self.block_scale_codes).nbytes)

    @property
    def new_block_scale_metadata_bytes(self) -> int:
        if self.block_scale_provenance == "locked_resident_e8m0":
            return 0
        return self.block_scale_bytes_read

    @property
    def total_storage_bytes(self) -> int:
        return (
            self.layer_shared_codebook_bytes + self.encoded_index_bytes
            + self.new_block_scale_metadata_bytes
        )


def encode_block_residual_pq(
    residual_weights: np.ndarray,
    codebooks: BlockPQCodebooks,
    *,
    block_scales: np.ndarray | None = None,
    block_scale_provenance: BlockScaleProvenance | None = None,
) -> EncodedBlockResidualPQ:
    """Assign residual rows to FP16 codebooks and physically pack indices."""
    weights = _require_finite(np.asarray(residual_weights, dtype=np.float64), "residual weights")
    if weights.ndim < 2 or weights.shape[-1] != codebooks.input_width:
        raise ValueError("residual weights do not match the codebook input width")
    stored = np.asarray(codebooks.centroids, dtype=np.float16)
    if not np.all(np.isfinite(stored)):
        raise ValueError("PQ codebook overflowed FP16 storage")
    decoded_codebooks = stored.astype(np.float64)
    vectors, scale_codes, _ = _normalized_weight_blocks(weights, block_scales)
    if codebooks.normalized_by_block_scale != (scale_codes is not None):
        raise ValueError("fit and encode must agree on block-scale normalization")
    if scale_codes is None:
        if block_scale_provenance is not None:
            raise ValueError("block-scale provenance was supplied without block scales")
        provenance: BlockScaleProvenance = "none"
    else:
        provenance = "new_e8m0_metadata" if block_scale_provenance is None else block_scale_provenance
        if provenance not in ("new_e8m0_metadata", "locked_resident_e8m0"):
            raise ValueError("scaled PQ requires new or locked-resident E8M0 provenance")
    residual = vectors.copy()
    codes = np.empty((len(vectors), codebooks.blocks, codebooks.stages), dtype=np.uint8)
    for stage in range(codebooks.stages):
        for block in range(codebooks.blocks):
            assigned, _ = _assign_codewords(
                residual[:, block], decoded_codebooks[stage, block],
                np.asarray(codebooks.activation_second_moment[block], np.float64),
            )
            codes[:, block, stage] = assigned
            residual[:, block] -= decoded_codebooks[stage, block, assigned]
    return EncodedBlockResidualPQ(
        stored_codebooks=stored,
        packed_codes=_pack_codes(codes, 4),
        code_shape=tuple(int(value) for value in codes.shape),
        weight_shape=tuple(int(value) for value in weights.shape),
        block_scale_codes=scale_codes,
        block_scale_provenance=provenance,
    )


def block_pq_codes(encoded: EncodedBlockResidualPQ) -> np.ndarray:
    """Return logical indices; this decoded view is not charged as storage."""
    count = int(np.prod(encoded.code_shape, dtype=np.int64))
    return _unpack_codes(encoded.packed_codes, count, 4).reshape(encoded.code_shape)


def decode_block_residual_pq(encoded: EncodedBlockResidualPQ) -> np.ndarray:
    codebooks = np.asarray(encoded.stored_codebooks, dtype=np.float32)
    codes = block_pq_codes(encoded)
    reconstructed = np.zeros((encoded.code_shape[0], encoded.code_shape[1], PQ_BLOCK_SIZE), dtype=np.float32)
    for stage in range(encoded.code_shape[2]):
        for block in range(encoded.code_shape[1]):
            reconstructed[:, block] += codebooks[stage, block, codes[:, block, stage]]
    if encoded.block_scale_codes is not None:
        scales = np.exp2(
            np.asarray(encoded.block_scale_codes, dtype=np.int16).reshape(encoded.code_shape[:2]) - 127,
        ).astype(np.float32)
        reconstructed *= scales[:, :, None]
    return reconstructed.reshape(encoded.weight_shape)


@dataclass(frozen=True)
class PQResponseAccounting:
    """Exact logical work and traffic for the reference LUT evaluation."""

    layer_shared_codebook_bytes: int
    encoded_index_bytes: int
    new_block_scale_metadata_bytes: int
    block_scale_bytes_read: int
    activation_bytes: int
    lut_write_bytes: int
    lut_read_bytes: int
    codebook_lut_macs: int
    scale_multiplications: int
    response_accumulation_additions: int

    @property
    def total_metadata_bytes(self) -> int:
        return (
            self.layer_shared_codebook_bytes + self.encoded_index_bytes
            + self.new_block_scale_metadata_bytes
        )

    @property
    def total_bytes_read(self) -> int:
        return (
            self.layer_shared_codebook_bytes + self.encoded_index_bytes + self.block_scale_bytes_read
            + self.activation_bytes + self.lut_read_bytes
        )


@dataclass(frozen=True)
class PQResponseEvaluation:
    response: np.ndarray
    lookup_table: np.ndarray
    accounting: PQResponseAccounting


def evaluate_block_pq_response_lut(
    encoded: EncodedBlockResidualPQ,
    activation: np.ndarray,
    *,
    activation_bytes_per_value: int | None = None,
    lut_bytes_per_value: int = 4,
) -> PQResponseEvaluation:
    """Evaluate all encoded row responses through shared codeword dot products."""
    raw_activation = _require_finite(np.asarray(activation), "activation")
    x = np.asarray(raw_activation, dtype=np.float32).reshape(-1)
    if x.size != encoded.weight_shape[-1]:
        raise ValueError("activation width does not match encoded residual weights")
    if activation_bytes_per_value is None:
        activation_bytes_per_value = int(raw_activation.dtype.itemsize)
    if activation_bytes_per_value < 1 or lut_bytes_per_value < 1:
        raise ValueError("byte widths must be positive")
    codebooks = np.asarray(encoded.stored_codebooks, dtype=np.float32)
    xblocks = x.reshape(encoded.code_shape[1], PQ_BLOCK_SIZE)
    lut = np.einsum("sbkd,bd->sbk", codebooks, xblocks, optimize=True).astype(np.float32, copy=False)
    codes = block_pq_codes(encoded)
    response = np.zeros(encoded.code_shape[0], dtype=np.float32)
    scales = None
    if encoded.block_scale_codes is not None:
        scales = np.exp2(
            np.asarray(encoded.block_scale_codes, dtype=np.int16).reshape(encoded.code_shape[:2]) - 127,
        ).astype(np.float32)
    for block in range(encoded.code_shape[1]):
        block_response = np.zeros(encoded.code_shape[0], dtype=np.float32)
        for stage in range(encoded.code_shape[2]):
            block_response += lut[stage, block, codes[:, block, stage]]
        if scales is not None:
            block_response *= scales[:, block]
        response += block_response

    terms_per_row = encoded.code_shape[1] * encoded.code_shape[2]
    accounting = PQResponseAccounting(
        layer_shared_codebook_bytes=encoded.layer_shared_codebook_bytes,
        encoded_index_bytes=encoded.encoded_index_bytes,
        new_block_scale_metadata_bytes=encoded.new_block_scale_metadata_bytes,
        block_scale_bytes_read=encoded.block_scale_bytes_read,
        activation_bytes=x.size * int(activation_bytes_per_value),
        lut_write_bytes=lut.size * int(lut_bytes_per_value),
        lut_read_bytes=encoded.code_shape[0] * terms_per_row * int(lut_bytes_per_value),
        codebook_lut_macs=lut.size * PQ_BLOCK_SIZE,
        scale_multiplications=(
            0 if encoded.block_scale_codes is None else encoded.code_shape[0] * encoded.code_shape[1]
        ),
        response_accumulation_additions=encoded.code_shape[0] * max(terms_per_row - 1, 0),
    )
    return PQResponseEvaluation(
        response=response.reshape(encoded.weight_shape[:-1]), lookup_table=lut, accounting=accounting,
    )


@dataclass(frozen=True)
class QuantizedRows:
    """Physically packed low-bit rows with one stored FP16 scale per row."""

    packed_codes: np.ndarray
    scales: np.ndarray
    shape: tuple[int, int]
    encoding: RowEncoding

    def __post_init__(self) -> None:
        if self.encoding not in ("int2", "ternary", "sign_scale"):
            raise ValueError(f"unsupported row encoding: {self.encoding}")
        if len(self.shape) != 2 or min(self.shape) < 1:
            raise ValueError("quantized row shape must be a non-empty matrix")
        packed = np.asarray(self.packed_codes)
        scales = np.asarray(self.scales)
        if packed.dtype != np.uint8 or packed.ndim != 1:
            raise ValueError("packed row codes must be a flat uint8 array")
        if scales.dtype != np.float16 or scales.shape != (self.shape[0],):
            raise ValueError("one FP16 scale is required per row")
        bits = 1 if self.encoding == "sign_scale" else 2
        expected = (self.shape[0] * self.shape[1] * bits + 7) // 8
        if packed.size != expected:
            raise ValueError("packed row-code payload has the wrong byte length")
        if not np.all(np.isfinite(scales)):
            raise ValueError("row scales contain a non-finite value")

    @property
    def bits_per_value(self) -> int:
        return 1 if self.encoding == "sign_scale" else 2

    @property
    def payload_bytes(self) -> int:
        return int(np.asarray(self.packed_codes).nbytes)

    @property
    def scale_bytes(self) -> int:
        return int(np.asarray(self.scales).nbytes)

    @property
    def storage_bytes(self) -> int:
        return self.payload_bytes + self.scale_bytes

    @property
    def dense_matvec_macs(self) -> int:
        return self.shape[0] * self.shape[1]


def _stored_fp16_scale(value: float) -> np.float16:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("row scale must be finite and nonnegative")
    if value == 0.0:
        return np.float16(0.0)
    minimum = float(np.nextafter(np.float16(0.0), np.float16(1.0)))
    stored = np.float16(max(value, minimum))
    if not np.isfinite(stored):
        raise ValueError("row scale overflowed FP16")
    return stored


def _fit_row_levels(row: np.ndarray, levels: np.ndarray) -> tuple[np.ndarray, np.float16]:
    values = np.asarray(row, dtype=np.float64)
    if not np.any(values):
        zero = int(np.argmin(np.abs(levels)))
        return np.full(values.shape, zero, dtype=np.uint8), np.float16(0.0)
    maximum = float(np.max(np.abs(values)))
    nonzero_level = np.abs(levels[np.nonzero(levels)])
    seeds = [
        maximum / max(float(np.max(np.abs(levels))), 1e-30),
        float(np.mean(np.abs(values))) / max(float(np.mean(nonzero_level)), 1e-30),
    ]
    positive = values[values > 0.0]
    negative = values[values < 0.0]
    if len(positive):
        seeds.append(float(np.max(positive)) / max(float(np.max(levels)), 1e-30))
    if len(negative):
        seeds.append(float(-np.min(negative)) / max(float(-np.min(levels)), 1e-30))
    best: tuple[float, int, np.ndarray, np.float16] | None = None
    for seed_index, seed in enumerate(seeds):
        scale = max(seed, 1e-30)
        for _ in range(32):
            codes = np.argmin(np.square(values[:, None] - scale * levels[None, :]), axis=1)
            quantized = levels[codes]
            denominator = float(quantized @ quantized)
            updated = float(values @ quantized) / denominator if denominator else 0.0
            updated = max(updated, 0.0)
            if abs(updated - scale) <= 1e-12 * max(scale, 1.0):
                scale = updated
                break
            scale = updated
        stored = _stored_fp16_scale(scale)
        decoded_scale = float(stored)
        codes = np.argmin(np.square(values[:, None] - decoded_scale * levels[None, :]), axis=1).astype(np.uint8)
        error = float(np.square(values - decoded_scale * levels[codes]).sum())
        candidate = (error, seed_index, codes, stored)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    assert best is not None
    return best[2], best[3]


def quantize_low_bit_rows(matrix: np.ndarray, encoding: RowEncoding) -> QuantizedRows:
    """Quantize a synthesis matrix row-wise and return its packed payload."""
    value = _require_finite(np.asarray(matrix, dtype=np.float64), "row matrix")
    if value.ndim != 2 or min(value.shape) < 1:
        raise ValueError("matrix must be a non-empty two-dimensional array")
    if encoding == "int2":
        levels = np.array([-2.0, -1.0, 0.0, 1.0], dtype=np.float64)
        bits = 2
    elif encoding == "ternary":
        levels = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
        bits = 2
    elif encoding == "sign_scale":
        levels = np.array([-1.0, 1.0], dtype=np.float64)
        bits = 1
    else:
        raise ValueError(f"unsupported row encoding: {encoding}")
    codes = np.empty(value.shape, dtype=np.uint8)
    scales = np.empty(value.shape[0], dtype=np.float16)
    for row in range(value.shape[0]):
        codes[row], scales[row] = _fit_row_levels(value[row], levels)
    return QuantizedRows(
        packed_codes=_pack_codes(codes, bits), scales=scales,
        shape=(int(value.shape[0]), int(value.shape[1])), encoding=encoding,
    )


def decode_low_bit_rows(encoded: QuantizedRows) -> np.ndarray:
    levels = {
        "int2": np.array([-2.0, -1.0, 0.0, 1.0], dtype=np.float32),
        "ternary": np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        "sign_scale": np.array([-1.0, 1.0], dtype=np.float32),
    }[encoded.encoding]
    count = encoded.shape[0] * encoded.shape[1]
    codes = _unpack_codes(encoded.packed_codes, count, encoded.bits_per_value).reshape(encoded.shape)
    if np.any(codes >= len(levels)):
        raise ValueError(f"packed {encoded.encoding} payload contains an unused code")
    return levels[codes] * np.asarray(encoded.scales, dtype=np.float32)[:, None]


class _SoftFixedCardinality(torch.autograd.Function):
    """Logistic cardinality projection with its implicit exact Jacobian."""

    @staticmethod
    def forward(ctx: object, logits: torch.Tensor, count: int, temperature: float) -> torch.Tensor:
        width = logits.shape[-1]
        if count == 0:
            soft = torch.zeros_like(logits)
        elif count == width:
            soft = torch.ones_like(logits)
        else:
            work = logits.to(torch.float64)
            margin = max(40.0, 20.0 + 2.0 * float(np.log(max(width, 2)))) * temperature
            lower = work.amin(dim=-1, keepdim=True) - margin
            upper = work.amax(dim=-1, keepdim=True) + margin
            for _ in range(80):
                threshold = (lower + upper) * 0.5
                candidate = torch.sigmoid((work - threshold) / temperature)
                too_many = candidate.sum(dim=-1, keepdim=True) > count
                lower = torch.where(too_many, threshold, lower)
                upper = torch.where(too_many, upper, threshold)
            soft = torch.sigmoid((work - (lower + upper) * 0.5) / temperature).to(logits.dtype)
        ctx.temperature = temperature
        ctx.save_for_backward(soft)
        return soft

    @staticmethod
    def backward(ctx: object, gradient: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        (soft,) = ctx.saved_tensors
        weight = soft * (1.0 - soft) / ctx.temperature
        denominator = weight.sum(dim=-1, keepdim=True)
        weighted_mean = (gradient * weight).sum(dim=-1, keepdim=True) / denominator.clamp_min(
            torch.finfo(weight.dtype).tiny,
        )
        result = weight * (gradient - weighted_mean)
        result = torch.where(denominator > 0.0, result, torch.zeros_like(result))
        return result, None, None


def soft_fixed_cardinality_mask(
    logits: torch.Tensor, count: int, *, temperature: float = 1.0,
) -> torch.Tensor:
    """Return a differentiable mask whose rows sum to ``count``."""
    if not torch.is_floating_point(logits) or logits.ndim < 1:
        raise ValueError("logits must be a floating tensor with at least one dimension")
    if not bool(torch.all(torch.isfinite(logits)).item()):
        raise ValueError("logits contain a non-finite value")
    if int(count) != count or count < 0 or count > logits.shape[-1]:
        raise ValueError("count must lie between zero and the final logits width")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    return _SoftFixedCardinality.apply(logits, int(count), float(temperature))


def hard_forward_soft_backward_mask(
    logits: torch.Tensor, count: int, *, temperature: float = 1.0,
) -> torch.Tensor:
    """Use deterministic exact top-k forward and cardinality-soft backward."""
    soft = soft_fixed_cardinality_mask(logits, count, temperature=temperature)
    hard = torch.zeros_like(logits)
    if count:
        # Stable ordering makes equal-score ties choose the lowest unit index.
        order = torch.argsort(logits, dim=-1, descending=True, stable=True)
        hard.scatter_(-1, order[..., :int(count)], 1.0)
    return hard + soft - soft.detach()


def exact_qmetric_set_loss(
    correction_vectors: torch.Tensor,
    mask: torch.Tensor,
    *,
    target_correction: torch.Tensor | None = None,
    proxy: torch.Tensor | None = None,
    beta: float = 0.0,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Exact ``D(target - sum_i mask_i c_i)`` for the proxy qmetric.

    The metric is ``G = I + beta * proxy @ proxy.T``.  Omitting the target
    uses the complete coherent correction ``sum_i c_i``.  Leading dimensions
    broadcast, allowing either invocation-specific or shared correction
    vectors.
    """
    if not torch.is_floating_point(correction_vectors) or correction_vectors.ndim < 2:
        raise ValueError("correction_vectors must be a floating [..., unit, output] tensor")
    if not torch.is_floating_point(mask) or mask.ndim < 1:
        raise ValueError("mask must be a floating [..., unit] tensor")
    if mask.shape[-1] != correction_vectors.shape[-2]:
        raise ValueError("mask and correction vectors have different unit counts")
    if beta < 0.0 or not np.isfinite(beta):
        raise ValueError("beta must be finite and nonnegative")
    if beta != 0.0 and proxy is None:
        raise ValueError("nonzero beta requires a proxy")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError(f"unsupported reduction: {reduction}")
    if not bool(torch.all(torch.isfinite(correction_vectors)).item()) or not bool(torch.all(torch.isfinite(mask)).item()):
        raise ValueError("set-loss inputs contain a non-finite value")

    selected = (mask.unsqueeze(-1) * correction_vectors).sum(dim=-2)
    target = correction_vectors.sum(dim=-2) if target_correction is None else target_correction
    target = torch.as_tensor(target, dtype=correction_vectors.dtype, device=correction_vectors.device)
    if target.shape[-1] != correction_vectors.shape[-1]:
        raise ValueError("target correction has the wrong output width")
    residual = target - selected
    loss = torch.sum(residual * residual, dim=-1)
    if proxy is not None and beta:
        p = torch.as_tensor(proxy, dtype=correction_vectors.dtype, device=correction_vectors.device)
        if p.ndim == 1:
            p = p[:, None]
        if p.ndim != 2 or p.shape[0] != correction_vectors.shape[-1]:
            raise ValueError("proxy must have shape [output, rank]")
        projected = residual @ p
        loss = loss + float(beta) * torch.sum(projected * projected, dim=-1)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


__all__ = [
    "BlockPQCodebooks",
    "BlockScaleProvenance",
    "EncodedBlockResidualPQ",
    "PQ_BLOCK_SIZE",
    "PQ_CODEBOOK_SIZE",
    "PQResponseAccounting",
    "PQResponseEvaluation",
    "QuantizedRows",
    "block_pq_codes",
    "decode_block_residual_pq",
    "decode_low_bit_rows",
    "encode_block_residual_pq",
    "evaluate_block_pq_response_lut",
    "exact_qmetric_set_loss",
    "fit_activation_weighted_block_pq",
    "hard_forward_soft_backward_mask",
    "quantize_low_bit_rows",
    "soft_fixed_cardinality_mask",
]
