"""Layer-shared analysis/synthesis transforms fitted on training data only."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch


@dataclass
class Transform:
    name: str
    analysis: np.ndarray  # [K, d]
    synthesis: np.ndarray  # [d, K]
    eigenvalues: np.ndarray | None = None
    retained_rank: int | None = None
    condition_number: float | None = None

    def validate(self, atol: float = 3e-3) -> None:
        d = self.synthesis.shape[0]
        if self.analysis.shape[1] != d or self.analysis.shape[0] != self.synthesis.shape[1]:
            raise ValueError(f"incompatible transform shapes for {self.name}")
        if self.analysis.shape[0] == d:
            error = np.linalg.norm(self.synthesis @ self.analysis - np.eye(d), ord="fro") / math.sqrt(d)
            if error > atol:
                raise ValueError(f"{self.name} full-rank reconstruction error {error}")


def covariance(samples: np.ndarray, shrinkage: float = 1e-4) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float32)
    second = (x.T @ x) / max(len(x), 1)
    mean_diag = float(np.trace(second) / second.shape[0])
    return second + np.eye(second.shape[0], dtype=np.float32) * (shrinkage * max(mean_diag, 1e-12))


def _eigh(matrix: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    value = torch.as_tensor(matrix, dtype=torch.float32, device=device)
    eigenvalues, eigenvectors = torch.linalg.eigh(value)
    order = torch.argsort(eigenvalues, descending=True)
    return eigenvalues[order].cpu().numpy(), eigenvectors[:, order].cpu().numpy()


def identity_transform(dimension: int) -> Transform:
    eye = np.eye(dimension, dtype=np.float32)
    return Transform("native", eye, eye)


def random_orthogonal_transform(dimension: int, seed: int) -> Transform:
    if dimension & (dimension - 1):
        rng = np.random.default_rng(seed)
        q, _ = np.linalg.qr(rng.standard_normal((dimension, dimension), dtype=np.float32))
        basis = q.astype(np.float32)
    else:
        basis = np.asarray([[1.0]], dtype=np.float32)
        while basis.shape[0] < dimension:
            basis = np.block([[basis, basis], [basis, -basis]])
        basis /= math.sqrt(dimension)
        rng = np.random.default_rng(seed)
        basis = basis[rng.permutation(dimension)][:, rng.permutation(dimension)]
        basis *= rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=dimension)[None, :]
    return Transform("random", basis.T.copy(), basis.copy())


def pca_transform(samples: np.ndarray, device: str = "cuda") -> Transform:
    c = covariance(samples)
    values, basis = _eigh(c, device)
    return Transform("pca", basis.T.copy(), basis.copy(), eigenvalues=values)


def generalized_transform(
    samples: np.ndarray,
    residual_operator: np.ndarray,
    device: str = "cuda",
    shrinkage: float = 1e-4,
    rank_tolerance: float = 1e-5,
) -> Transform:
    c = covariance(samples, shrinkage)
    values_c, vectors_c = _eigh(c, device)
    keep = values_c > max(float(values_c[0]) * rank_tolerance, 1e-12)
    retained = int(np.sum(keep))
    root = (vectors_c[:, keep] * np.sqrt(values_c[keep])[None, :]) @ vectors_c[:, keep].T
    inverse_root = (vectors_c[:, keep] * (1.0 / np.sqrt(values_c[keep]))[None, :]) @ vectors_c[:, keep].T
    kernel = root @ np.asarray(residual_operator, dtype=np.float32) @ root
    lambdas, u = _eigh((kernel + kernel.T) * 0.5, device)
    synthesis = root @ u
    analysis = u.T @ inverse_root
    result = Transform(
        "generalized", analysis.astype(np.float32), synthesis.astype(np.float32),
        eigenvalues=lambdas.astype(np.float32), retained_rank=retained,
        condition_number=float(values_c[0] / max(values_c[keep][-1], 1e-30)),
    )
    result.validate(atol=2e-2)
    return result


def learned_orthogonal_transform(
    samples: np.ndarray,
    residual_operator: np.ndarray,
    seed: int,
    device: str = "cuda",
) -> Transform:
    """Orthogonal spectral relaxation of the sparse residual-action objective.

    A request bootstrap supplies the stated random seed. The orthogonal
    eigenbasis concentrates the symmetrized activation/residual energy. This
    is a stable closed-form relaxation, not an unconstrained dense SGD fit.
    """
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(samples), size=len(samples))
    c = covariance(np.asarray(samples)[indices], shrinkage=3e-4)
    h = np.asarray(residual_operator, dtype=np.float32)
    objective = (c @ h + h @ c) * 0.5
    values, basis = _eigh((objective + objective.T) * 0.5, device)
    result = Transform(f"learned_s{seed}", basis.T.copy(), basis.copy(), eigenvalues=values)
    result.validate()
    return result


def two_view_overcomplete(first: Transform, second: Transform) -> Transform:
    if first.synthesis.shape != second.synthesis.shape:
        raise ValueError("views must have equal full-rank shape")
    analysis = np.concatenate([first.analysis, second.analysis], axis=0)
    synthesis = 0.5 * np.concatenate([first.synthesis, second.synthesis], axis=1)
    return Transform("overcomplete_2x", analysis.astype(np.float32), synthesis.astype(np.float32))
