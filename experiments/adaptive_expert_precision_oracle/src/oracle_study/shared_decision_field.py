"""Compact common-coordinate fields fitted to exact joint decisions.

The exact joint oracle embeds every down column in the full output qmetric.
That representation is useful as a teacher but is far too large to read at
runtime.  This module fits one layer-shared orthonormal basis to training-only
qmetric residual trajectories, then projects every expert's Q2/Q4 down rows
into those common coordinates.  Exact A/B/C streams continue to provide all
same-unit terms; the compact field is used only for signed cross-unit and
cross-expert interactions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .average_rate_allocator import qmetric_features
from .interaction_field import JointInteractionFactor


__all__ = [
    "SharedDecisionBasis",
    "fit_shared_decision_basis",
    "project_shared_interaction_factor",
]


def _canonical_columns(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, np.float64).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return result


@dataclass(frozen=True)
class SharedDecisionBasis:
    """Layer-shared qmetric feature basis and its frozen fit diagnostics."""

    components: np.ndarray
    fit_rows: int
    fit_rank: int
    normalized_rows: bool
    explained_energy: float
    method: str = "training_residual_qmetric_svd"

    def __post_init__(self) -> None:
        value = np.asarray(self.components)
        if value.ndim != 2 or value.shape[1] != int(self.fit_rank):
            raise ValueError("shared basis shape disagrees with fit rank")
        if int(self.fit_rows) < int(self.fit_rank) or self.fit_rank < 1:
            raise ValueError("shared basis has insufficient fit rows")
        if not np.all(np.isfinite(value)):
            raise ValueError("shared basis contains non-finite values")
        gram = value.T.astype(np.float64) @ value.astype(np.float64)
        if not np.allclose(gram, np.eye(self.fit_rank), rtol=1e-5, atol=1e-6):
            raise ValueError("shared basis columns are not orthonormal")
        if not np.isfinite(self.explained_energy) or not 0.0 <= self.explained_energy <= 1.0:
            raise ValueError("shared basis explained energy is invalid")

    @property
    def feature_width(self) -> int:
        return int(np.asarray(self.components).shape[0])


def fit_shared_decision_basis(
    residuals: np.ndarray,
    proxy: np.ndarray | None,
    beta: float,
    rank: int,
    *,
    normalize_rows: bool = True,
) -> SharedDecisionBasis:
    """Fit a deterministic common basis to training-only residual fields.

    Row normalization gives every routed token/layer trajectory equal weight,
    rather than allowing a few high-damage examples to determine the basis.
    The SVD is performed in the exact augmented qmetric feature space, so the
    fitted coordinates can allocate their small rank between ordinary output
    cancellation and the existing future-proxy term.
    """

    raw = np.asarray(residuals, np.float64)
    if raw.ndim != 2 or raw.shape[0] < 1 or not np.all(np.isfinite(raw)):
        raise ValueError("training residuals must be a finite nonempty matrix")
    features = qmetric_features(raw, proxy, beta)
    if normalize_rows:
        norms = np.linalg.norm(features, axis=1)
        if np.any(norms <= 0.0) or not np.all(np.isfinite(norms)):
            raise ValueError("training residuals contain a zero qmetric row")
        features = features / norms[:, None]
    requested = int(rank)
    if requested < 1 or requested > min(features.shape):
        raise ValueError("shared basis rank lies outside the training matrix")
    _, singular, right = np.linalg.svd(features, full_matrices=False)
    components = _canonical_columns(right[:requested].T)
    energy = singular * singular
    explained = float(np.sum(energy[:requested]) / np.sum(energy))
    return SharedDecisionBasis(
        components=np.asarray(components, np.float32),
        fit_rows=int(features.shape[0]),
        fit_rank=requested,
        normalized_rows=bool(normalize_rows),
        explained_energy=explained,
    )


def project_shared_interaction_factor(
    down2_rows: np.ndarray,
    down4_rows: np.ndarray,
    proxy: np.ndarray | None,
    beta: float,
    basis: SharedDecisionBasis,
    *,
    dtype: np.dtype | type = np.float32,
) -> JointInteractionFactor:
    """Project one expert into a layer-shared decision coordinate system."""

    d2 = np.asarray(down2_rows, np.float64)
    d4 = np.asarray(down4_rows, np.float64)
    if d2.ndim != 2 or d4.shape != d2.shape:
        raise ValueError("down rows must have equal [unit,output] shapes")
    feature2 = qmetric_features(d2, proxy, beta)
    feature4 = qmetric_features(d4, proxy, beta)
    components = np.asarray(basis.components, np.float64)
    if feature2.shape[1] != components.shape[0]:
        raise ValueError("basis and qmetric feature widths disagree")
    return JointInteractionFactor(
        l4=np.asarray(feature4 @ components, dtype=dtype),
        l2=np.asarray(feature2 @ components, dtype=dtype),
        method=basis.method,
        tail_rank=int(basis.fit_rank),
        exact_rank=0,
        encoding=str(np.dtype(dtype)),
    )
