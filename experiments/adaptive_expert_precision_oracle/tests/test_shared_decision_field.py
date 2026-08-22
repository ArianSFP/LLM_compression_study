from __future__ import annotations

import numpy as np
import pytest

from oracle_study.average_rate_allocator import qmetric_features
from oracle_study.shared_decision_field import (
    fit_shared_decision_basis,
    project_shared_interaction_factor,
)


def test_shared_basis_is_deterministic_common_qmetric_projection() -> None:
    rng = np.random.default_rng(19)
    residuals = rng.normal(size=(12, 7))
    proxy = rng.normal(size=(7, 2))
    first = fit_shared_decision_basis(residuals, proxy, 0.3, 4)
    second = fit_shared_decision_basis(residuals, proxy, 0.3, 4)
    assert np.array_equal(first.components, second.components)
    assert np.allclose(first.components.T @ first.components, np.eye(4), atol=1e-6)
    assert 0.0 < first.explained_energy <= 1.0

    d2 = rng.normal(size=(5, 7))
    d4 = rng.normal(size=(5, 7))
    factor = project_shared_interaction_factor(d2, d4, proxy, 0.3, first)
    expected2 = qmetric_features(d2, proxy, 0.3) @ first.components
    expected4 = qmetric_features(d4, proxy, 0.3) @ first.components
    assert np.allclose(factor.l2, expected2, rtol=2e-7, atol=2e-7)
    assert np.allclose(factor.l4, expected4, rtol=2e-7, atol=2e-7)
    assert factor.rank == 4


def test_row_normalization_prevents_one_large_residual_dominating() -> None:
    residuals = np.asarray(((1000.0, 0.0), (0.0, 1.0), (0.0, 1.0)))
    normalized = fit_shared_decision_basis(
        residuals, None, 0.0, 1, normalize_rows=True,
    )
    unnormalized = fit_shared_decision_basis(
        residuals, None, 0.0, 1, normalize_rows=False,
    )
    assert abs(float(normalized.components[1, 0])) > 0.99
    assert abs(float(unnormalized.components[0, 0])) > 0.99


@pytest.mark.parametrize("bad_rank", [0, 4])
def test_shared_basis_rejects_invalid_rank(bad_rank: int) -> None:
    with pytest.raises(ValueError, match="rank"):
        fit_shared_decision_basis(np.eye(3), None, 0.0, bad_rank)


def test_shared_projection_rejects_unaligned_feature_width() -> None:
    basis = fit_shared_decision_basis(np.eye(4), None, 0.0, 2)
    with pytest.raises(ValueError, match="width"):
        project_shared_interaction_factor(
            np.ones((2, 3)), np.ones((2, 3)), None, 0.0, basis,
        )
