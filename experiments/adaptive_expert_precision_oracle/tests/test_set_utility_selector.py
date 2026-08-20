from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from oracle_study.set_utility_selector import (
    BlockPQCodebooks,
    block_pq_codes,
    decode_block_residual_pq,
    decode_low_bit_rows,
    encode_block_residual_pq,
    evaluate_block_pq_response_lut,
    exact_qmetric_set_loss,
    fit_activation_weighted_block_pq,
    hard_forward_soft_backward_mask,
    quantize_low_bit_rows,
    soft_fixed_cardinality_mask,
)


def _weighted_block_distortion(
    original: np.ndarray, reconstructed: np.ndarray, activation: np.ndarray,
) -> float:
    difference = (np.asarray(original) - np.asarray(reconstructed)).reshape(-1, original.shape[-1] // 32, 32)
    xblocks = np.asarray(activation).reshape(len(activation), original.shape[-1] // 32, 32)
    moment = np.einsum("nbd,nbe->bde", xblocks, xblocks) / len(xblocks)
    return float(np.einsum("vbd,bde,vbe->", difference, moment, difference))


def test_activation_weighted_pq_is_deterministic_and_second_moment_exact() -> None:
    rng = np.random.default_rng(20260820)
    weights = rng.normal(size=(2, 19, 64)).astype(np.float32)
    activation = rng.normal(size=(73, 64)).astype(np.float32)
    activation[:, :32] *= np.linspace(0.05, 2.0, 32)

    first = fit_activation_weighted_block_pq(weights, activation, stages=1, max_iterations=20)
    second = fit_activation_weighted_block_pq(weights, activation, stages=1, max_iterations=20)
    np.testing.assert_array_equal(first.centroids, second.centroids)
    np.testing.assert_array_equal(first.activation_second_moment, second.activation_second_moment)
    assert first.centroids.shape == (1, 2, 16, 32)
    assert np.array_equal(first.centroids[:, :, 0], np.zeros((1, 2, 32), np.float32))

    # E[((w-c)'x_b)^2] is exactly the second-moment quadratic form used by
    # the fit (tested per block so cross-block covariance cannot enter).
    encoded = encode_block_residual_pq(weights, first)
    decoded = decode_block_residual_pq(encoded)
    difference = (weights - decoded).reshape(-1, 2, 32)
    for block in range(2):
        empirical = np.square(difference[:, block] @ activation[:, block * 32:(block + 1) * 32].T).mean(axis=1)
        quadratic = np.einsum(
            "vd,de,ve->v", difference[:, block], first.activation_second_moment[block], difference[:, block],
        )
        np.testing.assert_allclose(quadratic, empirical, rtol=2e-6, atol=2e-6)


def test_second_additive_pq_stage_cannot_increase_weighted_distortion() -> None:
    rng = np.random.default_rng(13)
    weights = rng.normal(size=(3, 17, 64)).astype(np.float32)
    activation = rng.normal(size=(64, 64)).astype(np.float32)
    one = fit_activation_weighted_block_pq(weights, activation, stages=1, max_iterations=25)
    two = fit_activation_weighted_block_pq(weights, activation, stages=2, max_iterations=25)
    decoded_one = decode_block_residual_pq(encode_block_residual_pq(weights, one))
    decoded_two = decode_block_residual_pq(encode_block_residual_pq(weights, two))
    assert _weighted_block_distortion(weights, decoded_two, activation) <= (
        _weighted_block_distortion(weights, decoded_one, activation) + 2e-5
    )


@pytest.mark.parametrize("stages", [1, 2])
def test_pq_nibble_storage_decode_and_lut_response_parity(stages: int) -> None:
    rng = np.random.default_rng(8 + stages)
    blocks, vectors = 2, 18
    weights = rng.normal(size=(2, 9, blocks * 32)).astype(np.float32)
    activation_fit = rng.normal(size=(48, blocks * 32)).astype(np.float32)
    model = fit_activation_weighted_block_pq(weights, activation_fit, stages=stages, max_iterations=15)
    encoded = encode_block_residual_pq(weights, model)
    logical_code_count = vectors * blocks * stages
    assert encoded.layer_shared_codebook_bytes == stages * blocks * 16 * 32 * 2
    assert encoded.encoded_index_bytes == math.ceil(logical_code_count / 2)
    assert block_pq_codes(encoded).shape == (vectors, blocks, stages)
    assert np.all(block_pq_codes(encoded) < 16)

    x = rng.normal(size=blocks * 32).astype(np.float32)
    evaluation = evaluate_block_pq_response_lut(encoded, x)
    expected = decode_block_residual_pq(encoded) @ x
    np.testing.assert_allclose(evaluation.response, expected, rtol=2e-6, atol=2e-6)
    accounting = evaluation.accounting
    assert accounting.layer_shared_codebook_bytes == encoded.layer_shared_codebook_bytes
    assert accounting.encoded_index_bytes == encoded.encoded_index_bytes
    assert accounting.new_block_scale_metadata_bytes == 0
    assert accounting.block_scale_bytes_read == 0
    assert accounting.activation_bytes == blocks * 32 * 4
    assert accounting.codebook_lut_macs == stages * blocks * 16 * 32
    assert accounting.scale_multiplications == 0
    assert accounting.lut_write_bytes == stages * blocks * 16 * 4
    assert accounting.lut_read_bytes == vectors * stages * blocks * 4
    assert accounting.response_accumulation_additions == vectors * (stages * blocks - 1)
    assert accounting.total_metadata_bytes == encoded.total_storage_bytes


def test_pq_codebooks_are_reusable_layer_shared_metadata() -> None:
    rng = np.random.default_rng(22)
    training_weights = rng.normal(size=(32, 32)).astype(np.float32)
    activation = rng.normal(size=(40, 32)).astype(np.float32)
    model = fit_activation_weighted_block_pq(training_weights, activation)
    left = encode_block_residual_pq(rng.normal(size=(7, 32)), model)
    right = encode_block_residual_pq(rng.normal(size=(11, 32)), model)
    np.testing.assert_array_equal(left.stored_codebooks, right.stored_codebooks)
    assert left.layer_shared_codebook_bytes == right.layer_shared_codebook_bytes == 16 * 32 * 2
    assert left.encoded_index_bytes == math.ceil(7 / 2)
    assert right.encoded_index_bytes == math.ceil(11 / 2)



def test_scale_normalized_pq_avoids_e8m0_magnitude_confounding() -> None:
    rng = np.random.default_rng(812)
    patterns = rng.normal(size=(10, 32)).astype(np.float32)
    scales = np.repeat(np.asarray([2.0 ** -8, 1.0, 2.0 ** 8], np.float32), len(patterns))[:, None]
    normalized = np.tile(patterns, (3, 1))
    weights = normalized * scales
    block_scales = scales.copy()
    activation = rng.normal(size=(96, 32)).astype(np.float32)

    normalized_model = fit_activation_weighted_block_pq(
        weights, activation, block_scales=block_scales, max_iterations=25,
    )
    repeated_model = fit_activation_weighted_block_pq(
        weights, activation, block_scales=block_scales, max_iterations=25,
    )
    decoded_model = fit_activation_weighted_block_pq(weights, activation, max_iterations=25)
    np.testing.assert_array_equal(normalized_model.centroids, repeated_model.centroids)
    assert normalized_model.normalized_by_block_scale
    assert not decoded_model.normalized_by_block_scale

    normalized_encoded = encode_block_residual_pq(
        weights, normalized_model, block_scales=block_scales,
        block_scale_provenance="locked_resident_e8m0",
    )
    decoded_encoded = encode_block_residual_pq(weights, decoded_model)
    normalized_reconstruction = decode_block_residual_pq(normalized_encoded) / scales
    decoded_reconstruction = decode_block_residual_pq(decoded_encoded) / scales
    normalized_error = float(np.square(normalized - normalized_reconstruction).mean())
    confounded_error = float(np.square(normalized - decoded_reconstruction).mean())
    assert normalized_error < confounded_error * 1e-3



def test_scale_squared_fit_moves_centroid_and_reduces_decoded_response_error() -> None:
    rows = 30
    normalized = np.zeros((rows, 32), np.float32)
    normalized[:, 0] = np.linspace(0.2, 6.0, rows)
    scales = np.ones((rows, 1), np.float32)
    important = 15
    scales[important, 0] = 8.0
    weights = normalized * scales
    activation = np.zeros((64, 32), np.float32)
    activation[:, 0] = np.linspace(-2.0, 2.0, len(activation))

    weighted = fit_activation_weighted_block_pq(
        weights, activation, block_scales=scales, max_iterations=50,
    )
    unweighted_fit = fit_activation_weighted_block_pq(normalized, activation, max_iterations=50)
    unweighted = BlockPQCodebooks(
        unweighted_fit.centroids, unweighted_fit.activation_second_moment,
        normalized_by_block_scale=True,
    )
    weighted_decoded = decode_block_residual_pq(encode_block_residual_pq(
        weights, weighted, block_scales=scales,
        block_scale_provenance="locked_resident_e8m0",
    ))
    unweighted_decoded = decode_block_residual_pq(encode_block_residual_pq(
        weights, unweighted, block_scales=scales,
        block_scale_provenance="locked_resident_e8m0",
    ))
    target = float(normalized[important, 0])
    weighted_centroid = float(weighted_decoded[important, 0] / scales[important, 0])
    unweighted_centroid = float(unweighted_decoded[important, 0] / scales[important, 0])
    assert abs(target - weighted_centroid) < abs(target - unweighted_centroid) * 0.05
    weighted_response_error = float(np.square((weights - weighted_decoded) @ activation.T).mean())
    unweighted_response_error = float(np.square((weights - unweighted_decoded) @ activation.T).mean())
    assert weighted_response_error < unweighted_response_error * 0.25


def test_scale_aware_pq_parity_and_resident_provenance_accounting() -> None:
    rng = np.random.default_rng(913)
    rows, blocks, stages = 24, 2, 2
    scales = np.exp2(rng.integers(-10, 11, size=(rows, blocks))).astype(np.float32)
    normalized = rng.normal(size=(rows, blocks, 32)).astype(np.float32)
    weights = (normalized * scales[:, :, None]).reshape(rows, blocks * 32)
    activation_fit = rng.normal(size=(80, blocks * 32)).astype(np.float32)
    model = fit_activation_weighted_block_pq(
        weights, activation_fit, block_scales=scales, stages=stages, max_iterations=20,
    )
    repeated = fit_activation_weighted_block_pq(
        weights, activation_fit, block_scales=scales, stages=stages, max_iterations=20,
    )
    np.testing.assert_array_equal(model.centroids, repeated.centroids)
    charged = encode_block_residual_pq(weights, model, block_scales=scales)
    resident = encode_block_residual_pq(
        weights, model, block_scales=scales,
        block_scale_provenance="locked_resident_e8m0",
    )
    assert charged.new_block_scale_metadata_bytes == rows * blocks
    assert resident.new_block_scale_metadata_bytes == 0
    assert charged.block_scale_bytes_read == resident.block_scale_bytes_read == rows * blocks
    assert charged.total_storage_bytes == resident.total_storage_bytes + rows * blocks
    np.testing.assert_array_equal(charged.block_scale_codes, resident.block_scale_codes)
    np.testing.assert_array_equal(decode_block_residual_pq(charged), decode_block_residual_pq(resident))

    x = rng.normal(size=blocks * 32).astype(np.float32)
    evaluation = evaluate_block_pq_response_lut(resident, x)
    np.testing.assert_allclose(evaluation.response, decode_block_residual_pq(resident) @ x, rtol=2e-6, atol=2e-5)
    accounting = evaluation.accounting
    assert accounting.new_block_scale_metadata_bytes == 0
    assert accounting.block_scale_bytes_read == rows * blocks
    assert accounting.scale_multiplications == rows * blocks
    assert accounting.total_metadata_bytes == resident.total_storage_bytes
    assert accounting.total_bytes_read == (
        resident.layer_shared_codebook_bytes + resident.encoded_index_bytes + rows * blocks
        + blocks * 32 * 4 + rows * blocks * stages * 4
    )


def test_scaled_pq_requires_exact_e8m0_values_and_matching_fit_mode() -> None:
    weights = np.zeros((16, 32), np.float32)
    activation = np.ones((8, 32), np.float32)
    with pytest.raises(ValueError, match="powers of two"):
        fit_activation_weighted_block_pq(weights, activation, block_scales=np.full((16, 1), 1.5))
    scaled = fit_activation_weighted_block_pq(weights, activation, block_scales=np.ones((16, 1)))
    plain = fit_activation_weighted_block_pq(weights, activation)
    with pytest.raises(ValueError, match="agree"):
        encode_block_residual_pq(weights, scaled)
    with pytest.raises(ValueError, match="agree"):
        encode_block_residual_pq(weights, plain, block_scales=np.ones((16, 1)))
    with pytest.raises(ValueError, match="without block scales"):
        encode_block_residual_pq(weights, plain, block_scale_provenance="locked_resident_e8m0")


def test_pq_rejects_wrong_block_width_and_too_few_training_vectors() -> None:
    with pytest.raises(ValueError, match="divisible"):
        fit_activation_weighted_block_pq(np.zeros((20, 33)), np.zeros((5, 33)))
    with pytest.raises(ValueError, match="at least"):
        fit_activation_weighted_block_pq(np.zeros((14, 32)), np.zeros((5, 32)))
    centroids = np.zeros((1, 1, 16, 32), np.float32)
    with pytest.raises(ValueError, match="second moments"):
        BlockPQCodebooks(centroids, np.zeros((2, 32, 32)))


@pytest.mark.parametrize(
    ("encoding", "bits"),
    [("int2", 2), ("ternary", 2), ("sign_scale", 1)],
)
def test_low_bit_row_quantizers_pack_exactly_and_roundtrip_deterministically(
    encoding: str, bits: int,
) -> None:
    rng = np.random.default_rng(41)
    value = rng.normal(size=(5, 13)).astype(np.float32)
    first = quantize_low_bit_rows(value, encoding)  # type: ignore[arg-type]
    second = quantize_low_bit_rows(value, encoding)  # type: ignore[arg-type]
    np.testing.assert_array_equal(first.packed_codes, second.packed_codes)
    np.testing.assert_array_equal(first.scales, second.scales)
    assert first.payload_bytes == math.ceil(value.size * bits / 8)
    assert first.scale_bytes == value.shape[0] * 2
    assert first.storage_bytes == math.ceil(value.size * bits / 8) + value.shape[0] * 2
    assert first.dense_matvec_macs == value.size
    decoded = decode_low_bit_rows(first)
    assert decoded.shape == value.shape
    assert decoded.dtype == np.float32
    assert np.all(np.isfinite(decoded))
    assert np.square(value - decoded).sum() < np.square(value).sum()


def test_sign_scale_uses_l2_optimal_mean_absolute_row_scale() -> None:
    value = np.array([[1.0, -2.0, 3.0, -4.0], [0.0, 0.0, 0.0, 0.0]], np.float32)
    encoded = quantize_low_bit_rows(value, "sign_scale")
    decoded = decode_low_bit_rows(encoded)
    assert float(encoded.scales[0]) == pytest.approx(2.5, abs=2e-3)
    assert float(encoded.scales[1]) == 0.0
    np.testing.assert_allclose(decoded[0], [2.5, -2.5, 2.5, -2.5], atol=2e-3)
    np.testing.assert_array_equal(decoded[1], np.zeros(4, np.float32))


@pytest.mark.parametrize("encoding", ["int2", "ternary", "sign_scale"])
def test_low_bit_row_quantizers_handle_tiny_and_zero_rows(encoding: str) -> None:
    value = np.array([[1e-9, -2e-9, 0.0], [0.0, 0.0, 0.0]], np.float32)
    encoded = quantize_low_bit_rows(value, encoding)  # type: ignore[arg-type]
    decoded = decode_low_bit_rows(encoded)
    assert np.all(np.isfinite(decoded))
    np.testing.assert_array_equal(decoded[1], np.zeros(3, np.float32))


def test_hard_forward_mask_is_exact_topk_with_deterministic_ties() -> None:
    logits = torch.tensor([[3.0, 3.0, -1.0, 2.0], [0.0, 2.0, 1.0, 4.0]], dtype=torch.float64)
    mask = hard_forward_soft_backward_mask(logits, 2, temperature=0.7)
    np.testing.assert_array_equal(mask.detach().numpy(), [[1, 1, 0, 0], [0, 1, 0, 1]])
    np.testing.assert_array_equal(mask.sum(dim=-1).numpy(), [2, 2])


def test_soft_fixed_cardinality_has_exact_sum_and_constrained_gradient() -> None:
    logits = torch.tensor(
        [[-1.2, 0.4, 2.1, 0.3, -0.7], [3.0, -2.0, 0.0, 1.0, 0.2]],
        dtype=torch.float64,
        requires_grad=True,
    )
    soft = soft_fixed_cardinality_mask(logits, 2, temperature=0.8)
    torch.testing.assert_close(soft.sum(dim=-1), torch.full((2,), 2.0, dtype=torch.float64), atol=1e-12, rtol=0)
    upstream = torch.tensor([[1.0, -0.5, 2.0, 0.1, -1.0], [-2.0, 0.4, 0.9, 3.0, 0.2]], dtype=torch.float64)
    (soft * upstream).sum().backward()
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))
    assert torch.any(torch.abs(logits.grad) > 1e-8)
    # Adding a constant to every row cannot alter a cardinality projection.
    torch.testing.assert_close(logits.grad.sum(dim=-1), torch.zeros(2, dtype=torch.float64), atol=1e-12, rtol=0)


@pytest.mark.parametrize("count", [0, 4])
def test_straight_through_cardinality_endpoints_have_zero_gradient(count: int) -> None:
    logits = torch.arange(4, dtype=torch.float64, requires_grad=True)
    mask = hard_forward_soft_backward_mask(logits, count)
    assert float(mask.detach().sum()) == count
    (mask * torch.arange(1, 5, dtype=torch.float64)).sum().backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


def test_exact_qmetric_set_loss_matches_explicit_nonidentity_metric() -> None:
    generator = torch.Generator().manual_seed(91)
    batch, units, output, rank = 3, 7, 9, 4
    corrections = torch.randn((batch, units, output), generator=generator, dtype=torch.float64)
    mask = torch.rand((batch, units), generator=generator, dtype=torch.float64, requires_grad=True)
    target = torch.randn((batch, output), generator=generator, dtype=torch.float64)
    proxy = torch.randn((output, rank), generator=generator, dtype=torch.float64)
    beta = 0.37
    actual = exact_qmetric_set_loss(
        corrections, mask, target_correction=target, proxy=proxy, beta=beta, reduction="none",
    )
    residual = target - (mask.unsqueeze(-1) * corrections).sum(dim=1)
    metric = torch.eye(output, dtype=torch.float64) + beta * (proxy @ proxy.T)
    expected = torch.einsum("bi,ij,bj->b", residual, metric, residual)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    actual.sum().backward()
    assert mask.grad is not None and torch.all(torch.isfinite(mask.grad))


def test_exact_qmetric_set_loss_default_target_has_complete_endpoint() -> None:
    corrections = torch.tensor(
        [[1.0, 2.0, -1.0], [-0.5, 1.0, 3.0], [2.0, -2.0, 0.25]], dtype=torch.float64,
    )
    full = torch.ones(3, dtype=torch.float64)
    empty = torch.zeros(3, dtype=torch.float64)
    assert float(exact_qmetric_set_loss(corrections, full)) == pytest.approx(0.0, abs=1e-15)
    expected = torch.square(corrections.sum(dim=0)).sum()
    torch.testing.assert_close(
        exact_qmetric_set_loss(corrections, empty, reduction="sum"), expected,
    )
    with pytest.raises(ValueError, match="requires a proxy"):
        exact_qmetric_set_loss(corrections, empty, beta=0.1)
