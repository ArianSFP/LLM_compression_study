#!/usr/bin/env python3
"""Phase A wrapper with a strictly invertible regularized generalized basis."""

from __future__ import annotations

import numpy as np
import torch

import oracle_study.bases as bases
import run_phase_a_remote as implementation


def generalized_full_rank(
    samples: np.ndarray,
    residual_operator: np.ndarray,
    device: str = "cuda",
    shrinkage: float = 1e-4,
    rank_tolerance: float = 1e-5,
) -> bases.Transform:
    covariance = bases.covariance(samples, shrinkage)
    eigenvalues, eigenvectors = bases._eigh(covariance, device)
    floor = max(float(eigenvalues[0]) * rank_tolerance, 1e-12)
    regularized = np.maximum(eigenvalues, floor)
    root = (eigenvectors * np.sqrt(regularized)[None, :]) @ eigenvectors.T
    inverse_root = (eigenvectors * (1.0 / np.sqrt(regularized))[None, :]) @ eigenvectors.T
    kernel = root @ np.asarray(residual_operator, dtype=np.float32) @ root
    lambdas, vectors = bases._eigh((kernel + kernel.T) * 0.5, device)
    synthesis = root @ vectors
    analysis = vectors.T @ inverse_root
    result = bases.Transform(
        "generalized",
        analysis.astype(np.float32),
        synthesis.astype(np.float32),
        eigenvalues=lambdas.astype(np.float32),
        retained_rank=int(np.sum(eigenvalues > floor)),
        condition_number=float(regularized[0] / regularized[-1]),
    )
    result.validate(atol=2e-2)
    return result


if __name__ == "__main__":
    bases.generalized_transform = generalized_full_rank
    implementation.main()
