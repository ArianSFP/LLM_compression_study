"""Refinement-aware orthogonal transforms and sparse fixed-code selectors.

The resident matrix is deliberately absent from this module.  If ``R`` is a
streamed refinement and ``Q`` has orthonormal columns, the transformed actions
are the columns of ``R @ Q`` with activation coefficients ``Q.T @ x``.  This
keeps the experiment focused on the off-device path and makes every storage
and truncation approximation explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def canonicalize_columns(matrix: np.ndarray) -> np.ndarray:
    """Return a copy with a deterministic sign for every column."""
    value = np.asarray(matrix, np.float64).copy()
    if value.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    for column in range(value.shape[1]):
        pivot = int(np.argmax(np.abs(value[:, column])))
        if value[pivot, column] < 0.0:
            value[:, column] *= -1.0
    return value


def correction_eigenbasis(
    residuals: np.ndarray | Sequence[np.ndarray],
    metric: np.ndarray | None = None,
    *,
    rank: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Right-singular basis of one or more jointly scored refinements.

    Multiple residuals are interpreted as disjoint output blocks.  With one
    residual and output metric ``G``, this diagonalizes ``R.T @ G @ R``.
    The returned eigenvalues are descending and the basis has shape
    ``[input, rank]``.  A thin basis is sufficient whenever the stacked
    residual has no component outside its span.
    """
    if isinstance(residuals, np.ndarray):
        matrices = (np.asarray(residuals, np.float64),)
    else:
        matrices = tuple(np.asarray(value, np.float64) for value in residuals)
    if not matrices or any(value.ndim != 2 for value in matrices):
        raise ValueError("residuals must contain two-dimensional matrices")
    width = matrices[0].shape[1]
    if any(value.shape[1] != width for value in matrices):
        raise ValueError("residual input widths disagree")
    if metric is not None and len(matrices) != 1:
        raise ValueError("an explicit metric is supported for one residual")

    if metric is None:
        weighted = np.concatenate(matrices, axis=0)
    else:
        gram = np.asarray(metric, np.float64)
        if gram.shape != (matrices[0].shape[0], matrices[0].shape[0]):
            raise ValueError("metric shape disagrees with residual output")
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (gram + gram.T))
        if float(eigenvalues.min(initial=0.0)) < -1e-9:
            raise ValueError("metric must be positive semidefinite")
        root = eigenvectors @ (
            np.sqrt(np.maximum(eigenvalues, 0.0))[:, None] * eigenvectors.T
        )
        weighted = root @ matrices[0]

    maximum = min(weighted.shape)
    wanted = maximum if rank is None else int(rank)
    if wanted < 0 or wanted > maximum:
        raise ValueError(f"rank must lie in [0,{maximum}]")
    if wanted == 0:
        return np.empty((width, 0), np.float64), np.empty(0, np.float64)
    _, singular, right = np.linalg.svd(weighted, full_matrices=False)
    basis = canonicalize_columns(right[:wanted].T)
    return basis, singular[:wanted] ** 2


def activation_pca_basis(
    activations: np.ndarray,
    *,
    rank: int | None = None,
    center: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit an uncentered orthogonal activation basis by default.

    Centering is opt-in because a nonzero mean would otherwise require a
    separately represented affine correction at inference time.
    """
    values = np.asarray(activations, np.float64)
    if values.ndim != 2 or not len(values):
        raise ValueError("activations must be a nonempty [sample,input] matrix")
    if center:
        values = values - values.mean(axis=0, keepdims=True)
    width = values.shape[1]
    wanted = width if rank is None else int(rank)
    if wanted < 0 or wanted > width:
        raise ValueError(f"rank must lie in [0,{width}]")
    covariance = values.T @ values / float(len(values))
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    order = np.argsort(eigenvalues, kind="stable")[::-1][:wanted]
    basis = canonicalize_columns(eigenvectors[:, order])
    return basis, np.maximum(eigenvalues[order], 0.0)


def transform_refinement(
    residual: np.ndarray, basis: np.ndarray, activation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return transformed columns ``R Q`` and coefficients ``Q.T x``."""
    matrix = np.asarray(residual, np.float64)
    q = np.asarray(basis, np.float64)
    x = np.asarray(activation, np.float64).reshape(-1)
    if matrix.ndim != 2 or q.ndim != 2:
        raise ValueError("residual and basis must be matrices")
    if matrix.shape[1] != q.shape[0] or q.shape[0] != x.size:
        raise ValueError("residual, basis, and activation widths disagree")
    return matrix @ q, q.T @ x


def fixed_contributions(
    residual: np.ndarray, basis: np.ndarray, activation: np.ndarray,
) -> np.ndarray:
    """Return one fixed output correction row per transformed action."""
    atoms, coefficients = transform_refinement(residual, basis, activation)
    return atoms.T * coefficients[:, None]


def row_quadratic_energy(
    rows: np.ndarray, metric: np.ndarray | None = None,
) -> np.ndarray:
    value = np.asarray(rows, np.float64)
    if value.ndim != 2:
        raise ValueError("rows must be a matrix")
    if metric is None:
        return np.einsum("ij,ij->i", value, value)
    gram = np.asarray(metric, np.float64)
    if gram.shape != (value.shape[1], value.shape[1]):
        raise ValueError("metric shape disagrees with row width")
    return np.einsum("ij,jk,ik->i", value, gram, value)


def diagonal_order(
    contributions: np.ndarray, metric: np.ndarray | None = None,
) -> np.ndarray:
    """Order fixed actions by independent quadratic utility."""
    energy = row_quadratic_energy(contributions, metric)
    return np.lexsort((np.arange(len(energy)), -energy)).astype(np.int64)


def exact_fixed_coefficient_order(
    contributions: np.ndarray,
    target: np.ndarray | None = None,
    metric: np.ndarray | None = None,
    *,
    maximum: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact marginal greedy order without coefficient refitting.

    This is the fixed-coefficient comparator often called OMP in the earlier
    study.  ``target`` defaults to the sum of every contribution.
    """
    rows = np.asarray(contributions, np.float64)
    if rows.ndim != 2:
        raise ValueError("contributions must be [action,output]")
    count = len(rows) if maximum is None else min(int(maximum), len(rows))
    if count < 0:
        raise ValueError("maximum must be nonnegative")
    desired = rows.sum(axis=0) if target is None else np.asarray(target, np.float64).reshape(-1)
    if desired.size != rows.shape[1]:
        raise ValueError("target width disagrees with contributions")
    gram = rows @ rows.T if metric is None else rows @ np.asarray(metric, np.float64) @ rows.T
    correlation = rows @ desired if metric is None else rows @ np.asarray(metric, np.float64) @ desired
    diagonal = np.diag(gram).copy()
    used = np.zeros(len(rows), bool)
    order = np.empty(count, np.int64)
    gains = np.empty(count, np.float64)
    for step in range(count):
        marginal = 2.0 * correlation - diagonal
        marginal[used] = -np.inf
        best = float(np.max(marginal))
        chosen = int(np.flatnonzero(marginal == best)[0])
        order[step] = chosen
        gains[step] = best
        used[chosen] = True
        correlation -= gram[:, chosen]
    return order, gains


def recovery_at_counts(
    contributions: np.ndarray,
    order: Sequence[int],
    counts: Iterable[int],
    target: np.ndarray | None = None,
    metric: np.ndarray | None = None,
) -> dict[int, float]:
    """Quadratic target recovery for prefixes of one action order."""
    rows = np.asarray(contributions, np.float64)
    desired = rows.sum(axis=0) if target is None else np.asarray(target, np.float64).reshape(-1)
    requested = sorted(set(map(int, counts)))
    if any(value < 0 or value > len(order) for value in requested):
        raise ValueError("requested count is outside the available order")
    denominator = float(row_quadratic_energy(desired[None, :], metric)[0])
    approximation = np.zeros_like(desired)
    result: dict[int, float] = {}
    pending = 0
    if requested and requested[0] == 0:
        result[0] = 0.0 if denominator > 0.0 else 1.0
        pending = 1
    for step, action in enumerate(map(int, order), start=1):
        approximation += rows[action]
        while pending < len(requested) and requested[pending] == step:
            error = desired - approximation
            damage = float(row_quadratic_energy(error[None, :], metric)[0])
            result[step] = 1.0 - damage / max(denominator, 1e-30)
            pending += 1
        if pending == len(requested):
            break
    return result


@dataclass(frozen=True)
class SymmetricColumnQuantization:
    """Decoded columnwise symmetric quantization plus physical accounting."""

    decoded: np.ndarray
    scales: np.ndarray
    bits: int
    code_bytes_per_column: int
    metadata_bytes: int

    @property
    def columns(self) -> int:
        return int(np.asarray(self.decoded).shape[1])

    @property
    def code_bytes(self) -> int:
        return self.columns * int(self.code_bytes_per_column)


def quantize_columns_symmetric(matrix: np.ndarray, bits: int) -> SymmetricColumnQuantization:
    """Quantize every action column with one FP16 scale.

    Codes are physically bit-packed in the accounting.  The decoded matrix is
    retained only for numerical evaluation; selectors must separately account
    for page rounding when combining projection payloads.
    """
    value = np.asarray(matrix, np.float32)
    if value.ndim != 2 or bits not in (2, 4, 8):
        raise ValueError("matrix must be two-dimensional and bits must be 2, 4, or 8")
    maximum = np.max(np.abs(value), axis=0)
    if bits == 2:
        # Use all four physical codewords.  Odd signed levels are symmetric
        # around zero; alternating assignment and least-squares scale updates
        # give the per-column MSE optimum for the current assignments.
        active = maximum > 0.0
        fitted_scale = np.where(active, maximum / 3.0, 0.0).astype(np.float32)
        assigned = np.ones(value.shape, dtype=np.int8)
        for _ in range(8):
            safe_scale = np.where(fitted_scale > 0.0, fitted_scale, 1.0)
            normalized = np.abs(value / safe_scale[None, :])
            magnitude = np.where(normalized < 2.0, 1, 3).astype(np.int8)
            assigned = np.where(value < 0.0, -magnitude, magnitude).astype(np.int8)
            numerator = np.sum(value * assigned, axis=0, dtype=np.float64)
            denominator = np.sum(
                assigned.astype(np.float64) ** 2, axis=0, dtype=np.float64,
            )
            fitted_scale = np.where(
                active, numerator / np.maximum(denominator, 1.0), 0.0,
            ).astype(np.float32)
        scale = fitted_scale.astype(np.float16)
    else:
        qmax = (1 << (bits - 1)) - 1
        scale = np.where(maximum > 0.0, maximum / float(qmax), 1.0).astype(np.float16)
    # A nonzero column can have a max/qmax below the FP16 subnormal range.
    # Clamp to the smallest physically storable positive scale: this preserves
    # finite decoding and is the closest representable symmetric quantizer.
    underflow = (maximum > 0.0) & (scale == 0.0)
    if np.any(underflow):
        scale = scale.copy()
        scale[underflow] = np.nextafter(
            np.float16(0.0), np.float16(1.0), dtype=np.float16,
        )
    decoded_scale = scale.astype(np.float32)
    if bits == 2:
        safe_scale = np.where(decoded_scale > 0.0, decoded_scale, 1.0)
        normalized = np.abs(value / safe_scale[None, :])
        magnitude = np.where(normalized < 2.0, 1, 3).astype(np.int8)
        codes = np.where(value < 0.0, -magnitude, magnitude).astype(np.int8)
    else:
        codes = np.clip(
            np.rint(value / decoded_scale[None, :]), -qmax, qmax,
        ).astype(np.int8)
    decoded = codes.astype(np.float32) * decoded_scale[None, :]
    code_bytes = (value.shape[0] * bits + 7) // 8
    return SymmetricColumnQuantization(
        decoded=decoded,
        scales=scale,
        bits=int(bits),
        code_bytes_per_column=int(code_bytes),
        metadata_bytes=int(scale.nbytes),
    )


def offdiagonal_fraction(matrices: np.ndarray) -> float:
    """Squared Frobenius fraction outside the diagonal."""
    value = np.asarray(matrices, np.float64)
    if value.ndim == 2:
        value = value[None, :, :]
    if value.ndim != 3 or value.shape[1] != value.shape[2]:
        raise ValueError("matrices must have shape [matrix,n,n]")
    total = float(np.sum(value * value))
    diagonal = float(np.sum(np.diagonal(value, axis1=1, axis2=2) ** 2))
    return (total - diagonal) / max(total, 1e-30)


def orthogonal_joint_diagonalize(
    matrices: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    sweeps: int = 8,
    angle_tolerance: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Jacobi approximate joint diagonalization of symmetric matrices."""
    values = np.asarray(matrices, np.float64).copy()
    if values.ndim != 3 or values.shape[1] != values.shape[2] or not len(values):
        raise ValueError("matrices must have shape [matrix,n,n]")
    values = 0.5 * (values + values.transpose(0, 2, 1))
    n = values.shape[1]
    weight = np.ones(len(values), np.float64) if weights is None else np.asarray(weights, np.float64)
    if weight.shape != (len(values),) or np.any(weight < 0.0) or not np.any(weight > 0.0):
        raise ValueError("weights must be nonnegative with one positive entry")
    weight = weight / weight.sum()
    basis = np.eye(n, dtype=np.float64)
    initial = offdiagonal_fraction(np.sqrt(weight)[:, None, None] * values)
    rotations = 0
    completed = 0
    for sweep in range(max(int(sweeps), 0)):
        maximum_angle = 0.0
        for left in range(n - 1):
            for right in range(left + 1, n):
                a = values[:, left, left] - values[:, right, right]
                b = 2.0 * values[:, left, right]
                two_by_two = np.asarray([
                    [np.sum(weight * b * b), -np.sum(weight * a * b)],
                    [-np.sum(weight * a * b), np.sum(weight * a * a)],
                ])
                _, vectors = np.linalg.eigh(two_by_two)
                cosine2, sine2 = vectors[:, 0]
                if cosine2 < 0.0:
                    cosine2, sine2 = -cosine2, -sine2
                angle = 0.5 * float(np.arctan2(sine2, cosine2))
                maximum_angle = max(maximum_angle, abs(angle))
                if abs(angle) <= float(angle_tolerance):
                    continue
                cosine, sine = float(np.cos(angle)), float(np.sin(angle))
                old_left = basis[:, left].copy(); old_right = basis[:, right].copy()
                basis[:, left] = cosine * old_left + sine * old_right
                basis[:, right] = -sine * old_left + cosine * old_right

                column_left = values[:, :, left].copy(); column_right = values[:, :, right].copy()
                values[:, :, left] = cosine * column_left + sine * column_right
                values[:, :, right] = -sine * column_left + cosine * column_right
                row_left = values[:, left, :].copy(); row_right = values[:, right, :].copy()
                values[:, left, :] = cosine * row_left + sine * row_right
                values[:, right, :] = -sine * row_left + cosine * row_right
                rotations += 1
        completed = sweep + 1
        if maximum_angle <= float(angle_tolerance):
            break
    basis = canonicalize_columns(basis)
    final = offdiagonal_fraction(np.sqrt(weight)[:, None, None] * values)
    return basis, values, {
        "initial_offdiagonal_fraction": float(initial),
        "final_offdiagonal_fraction": float(final),
        "sweeps": int(completed),
        "rotations": int(rotations),
    }


def block_joint_diagonalize(
    residuals: Sequence[np.ndarray],
    *,
    block_size: int = 32,
    weights: np.ndarray | None = None,
    sweeps: int = 8,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Fit a shared block-diagonal AJD basis directly from refinements."""
    matrices = tuple(np.asarray(value, np.float64) for value in residuals)
    if not matrices or any(value.ndim != 2 for value in matrices):
        raise ValueError("residuals must contain matrices")
    width = matrices[0].shape[1]
    size = int(block_size)
    if size <= 0 or width % size or any(value.shape[1] != width for value in matrices):
        raise ValueError("block size must divide the common input width")
    blocks: list[np.ndarray] = []
    initial = 0.0; final = 0.0; rotations = 0; completed = 0
    for start in range(0, width, size):
        grams = np.stack([
            value[:, start:start + size].T @ value[:, start:start + size]
            for value in matrices
        ])
        basis, _, facts = orthogonal_joint_diagonalize(
            grams, weights=weights, sweeps=sweeps,
        )
        blocks.append(basis)
        initial += float(facts["initial_offdiagonal_fraction"])
        final += float(facts["final_offdiagonal_fraction"])
        rotations += int(facts["rotations"])
        completed = max(completed, int(facts["sweeps"]))
    count = len(blocks)
    return np.stack(blocks), {
        "blocks": int(count),
        "block_size": int(size),
        "mean_initial_offdiagonal_fraction": initial / count,
        "mean_final_offdiagonal_fraction": final / count,
        "maximum_sweeps": int(completed),
        "rotations": int(rotations),
    }


def materialize_block_basis(blocks: np.ndarray) -> np.ndarray:
    """Materialize ``diag(Q_0,...,Q_b)`` for evaluation."""
    value = np.asarray(blocks, np.float64)
    if value.ndim != 3 or value.shape[1] != value.shape[2]:
        raise ValueError("blocks must have shape [block,size,size]")
    count, size, _ = value.shape
    result = np.zeros((count * size, count * size), np.float64)
    for index in range(count):
        start = index * size
        result[start:start + size, start:start + size] = value[index]
    return result
