"""Output-side low-rank correction fitting and serialized quantization."""

from __future__ import annotations

import math
import numpy as np
import torch


def output_basis(outputs: np.ndarray, rank: int, device: str = "cuda") -> tuple[np.ndarray, np.ndarray]:
    y = torch.as_tensor(np.asarray(outputs, dtype=np.float32), device=device)
    q = min(int(rank) + 16, min(y.shape))
    _, singular, vectors = torch.pca_lowrank(y, q=q, center=False, niter=4)
    keep = min(int(rank), vectors.shape[1])
    return vectors[:, :keep].cpu().numpy().astype(np.float32), singular[:keep].cpu().numpy().astype(np.float32)


def raw_residual_basis(residuals: np.ndarray, rank: int, device: str = "cuda") -> tuple[np.ndarray, np.ndarray]:
    matrix = np.concatenate([np.asarray(value, dtype=np.float32) for value in residuals], axis=1)
    value = torch.as_tensor(matrix, device=device)
    q = min(int(rank) + 16, min(value.shape))
    u, singular, _ = torch.pca_lowrank(value, q=q, center=False, niter=4)
    keep = min(int(rank), u.shape[1])
    return u[:, :keep].cpu().numpy().astype(np.float32), singular[:keep].cpu().numpy().astype(np.float32)


def proxy_weight_outputs(outputs: np.ndarray, proxy: np.ndarray, beta: float) -> np.ndarray:
    y = np.asarray(outputs, dtype=np.float32)
    p = np.asarray(proxy, dtype=np.float32)
    q, _ = np.linalg.qr(p)
    return y + (math.sqrt(1.0 + float(beta)) - 1.0) * (y @ q) @ q.T


def proxy_output_basis(
    outputs: np.ndarray,
    proxy: np.ndarray,
    beta: float,
    rank: int,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a metric-weighted output subspace and return raw-space atoms.

    PCA is performed after right multiplication by G**1/2, where
    G = I + beta QQ'.  Its right singular vectors therefore live in the
    transformed coordinates.  Map them back with G**-1/2 and QR-orthogonalize
    the raw-space span before deriving expert coefficients.
    """
    p = np.asarray(proxy, dtype=np.float32)
    q, _ = np.linalg.qr(p)
    weighted = proxy_weight_outputs(outputs, proxy, beta)
    transformed, singular = output_basis(weighted, rank, device)
    inverse_factor = 1.0 / math.sqrt(1.0 + float(beta)) - 1.0
    raw = transformed + inverse_factor * q @ (q.T @ transformed)
    raw, _ = np.linalg.qr(raw)
    return raw[:, : transformed.shape[1]].astype(np.float32), singular


def projected_coefficients(basis: np.ndarray, residual: np.ndarray) -> np.ndarray:
    return (np.asarray(basis, dtype=np.float32).T @ np.asarray(residual, dtype=np.float32)).astype(np.float32)


def activation_weighted_coefficients(
    basis: np.ndarray,
    residual: np.ndarray,
    hidden: np.ndarray,
    proxy: np.ndarray | None = None,
    beta: float = 0.0,
    ridge: float = 1e-4,
) -> np.ndarray:
    u = np.asarray(basis, dtype=np.float64)
    r = np.asarray(residual, dtype=np.float64)
    h = np.asarray(hidden, dtype=np.float64)
    if proxy is None or beta == 0:
        target = (h @ r.T) @ u
    else:
        p = np.asarray(proxy, dtype=np.float64)
        gu = u + beta * p @ (p.T @ u)
        gram = u.T @ gu
        target = (h @ r.T) @ gu @ np.linalg.pinv(gram)
    covariance = h.T @ h
    scale = float(np.trace(covariance) / max(covariance.shape[0], 1))
    regularizer = max(scale * ridge, 1e-12)
    if h.shape[0] < h.shape[1]:
        fitted = h.T @ np.linalg.solve(h @ h.T + np.eye(h.shape[0]) * regularizer, target)
    else:
        fitted = np.linalg.solve(
            covariance + np.eye(covariance.shape[0]) * regularizer,
            h.T @ target,
        )
    return fitted.T.astype(np.float32)


def quantize_coefficients(matrix: np.ndarray, bits: int, group_size: int = 64) -> tuple[np.ndarray, dict[str, int]]:
    value = np.asarray(matrix, dtype=np.float32)
    if bits == 16:
        restored = torch.from_numpy(value).to(torch.bfloat16).float().numpy()
        payload = value.size * 2
        return restored, {"payload_bytes": payload, "scale_bytes": 0, "header_bytes": 64, "total_bytes": payload + 64}
    if bits not in (4, 8):
        raise ValueError(bits)
    qmax = 2 ** (bits - 1) - 1
    restored = np.empty_like(value)
    groups = math.ceil(value.shape[1] / group_size)
    for row in range(value.shape[0]):
        for group in range(groups):
            start = group * group_size
            stop = min(start + group_size, value.shape[1])
            block = value[row, start:stop]
            maximum = float(np.max(np.abs(block))) if block.size else 0.0
            scale = maximum / qmax if maximum else 1.0
            code = np.clip(np.rint(block / scale), -qmax, qmax)
            restored[row, start:stop] = code * scale
    payload = math.ceil(value.size * bits / 8)
    scales = value.shape[0] * groups * 2
    return restored, {"payload_bytes": payload, "scale_bytes": scales, "header_bytes": 64, "total_bytes": payload + scales + 64}


def low_rank_reconstruct(basis: np.ndarray, coefficients: np.ndarray, hidden: np.ndarray) -> np.ndarray:
    return np.asarray(hidden, dtype=np.float32) @ np.asarray(coefficients, dtype=np.float32).T @ np.asarray(basis, dtype=np.float32).T


def kmeans(features: np.ndarray, clusters: int, seed: int, iterations: int = 30) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(features, dtype=np.float32)
    if clusters > len(value):
        raise ValueError("more clusters than samples")
    rng = np.random.default_rng(seed)
    centres = value[rng.choice(len(value), size=clusters, replace=False)].copy()
    assignment = np.zeros(len(value), dtype=np.int64)
    for _ in range(iterations):
        distance = ((value[:, None] - centres[None]) ** 2).sum(axis=2)
        updated = np.argmin(distance, axis=1)
        if np.array_equal(updated, assignment):
            break
        assignment = updated
        for cluster in range(clusters):
            members = value[assignment == cluster]
            if len(members):
                centres[cluster] = members.mean(axis=0)
    return assignment, centres
