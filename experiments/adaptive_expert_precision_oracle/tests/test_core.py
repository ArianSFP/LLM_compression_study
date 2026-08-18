from __future__ import annotations

import numpy as np

from oracle_study.bases import identity_transform, random_orthogonal_transform, two_view_overcomplete
from oracle_study.capture import split_for_sequence
from oracle_study.quant import (
    binary_quantize,
    binary_storage_bytes,
    page_ids_for_atoms,
    progressive_packet_bytes,
    progressive_quantize,
)


def test_request_split_is_deterministic_and_exclusive():
    values = [split_for_sequence(f"seq-{i}", 7) for i in range(100)]
    assert values == [split_for_sequence(f"seq-{i}", 7) for i in range(100)]
    assert set(values) == {"train", "validation", "test"}


def test_binary_quantizer_deterministic_and_accounted():
    rng = np.random.default_rng(3)
    weights = rng.standard_normal((7, 130), dtype=np.float32)
    q1, scales1 = binary_quantize(weights, 64)
    q2, scales2 = binary_quantize(weights, 64)
    np.testing.assert_array_equal(q1, q2)
    np.testing.assert_array_equal(scales1, scales2)
    accounting = binary_storage_bytes(weights.shape, 64)
    assert accounting["total_bytes"] == accounting["payload_bytes"] + accounting["scale_bytes"] + 64
    assert accounting["effective_bpw"] > 1.0


def test_full_basis_reconstructs_residual_action():
    rng = np.random.default_rng(5)
    residual = rng.standard_normal((11, 32), dtype=np.float32)
    x = rng.standard_normal(32, dtype=np.float32)
    for transform in (identity_transform(32), random_orthogonal_transform(32, 4)):
        atoms = residual @ transform.synthesis
        code = transform.analysis @ x
        np.testing.assert_allclose(atoms @ code, residual @ x, rtol=2e-5, atol=2e-5)


def test_overcomplete_full_reconstruction():
    rng = np.random.default_rng(8)
    residual = rng.standard_normal((9, 32), dtype=np.float32)
    x = rng.standard_normal(32, dtype=np.float32)
    transform = two_view_overcomplete(random_orthogonal_transform(32, 1), random_orthogonal_transform(32, 2))
    np.testing.assert_allclose(
        (residual @ transform.synthesis) @ (transform.analysis @ x),
        residual @ x,
        rtol=3e-5,
        atol=3e-5,
    )


def test_progressive_prefix_nesting_and_final_tolerance():
    rng = np.random.default_rng(9)
    atom = rng.standard_normal(257, dtype=np.float32)
    recon = {bits: progressive_quantize(atom, bits)[0] for bits in (0, 2, 4, 8, 16)}
    errors = [np.linalg.norm(atom - recon[bits]) for bits in (0, 2, 4, 8, 16)]
    assert all(a >= b for a, b in zip(errors, errors[1:]))
    assert errors[-1] / np.linalg.norm(atom) < 1e-4
    assert progressive_packet_bytes(257, 4) > 0


def test_page_rounding_counts_cross_page_packets():
    pages = page_ids_for_atoms(np.asarray([0, 1, 8]), packet_bytes=600, page_size=4096)
    np.testing.assert_array_equal(pages, np.asarray([0, 1]))
