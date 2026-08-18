import numpy as np

from oracle_study.low_rank import low_rank_reconstruct, projected_coefficients, proxy_output_basis


def test_full_output_basis_reconstructs_down_residual_action():
    rng = np.random.default_rng(7)
    residual = rng.standard_normal((9, 5), dtype=np.float32)
    hidden = rng.standard_normal((6, 5), dtype=np.float32)
    basis = np.eye(9, dtype=np.float32)
    coefficients = projected_coefficients(basis, residual)
    actual = low_rank_reconstruct(basis, coefficients, hidden)
    expected = hidden @ residual.T
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_proxy_basis_is_returned_in_raw_output_coordinates():
    outputs = np.zeros((64, 4), dtype=np.float32)
    outputs[:, 0] = np.linspace(-10, 10, 64)
    outputs[:, 1] = np.tile([-1.0, 1.0], 32)
    proxy = np.zeros((4, 1), dtype=np.float32)
    proxy[1, 0] = 1.0
    basis, _ = proxy_output_basis(outputs, proxy, beta=256.0, rank=1, device="cpu")
    assert basis.shape == (4, 1)
    np.testing.assert_allclose(basis.T @ basis, np.ones((1, 1)), atol=1e-5)
    assert abs(float(basis[1, 0])) > abs(float(basis[0, 0]))
