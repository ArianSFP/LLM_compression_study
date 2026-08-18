from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from oracle_study.rrq import (
    QuantizerSpec,
    build_rrq_series,
    decode_encoding,
    encode_legacy_q2,
    pack_q2_codes,
    quantize_int2_stage,
    serialize_encoding,
    unpack_q2_codes,
)
from run_progressive_q2_remote import q2_quantize


def test_q2_code_pack_round_trip_with_padding() -> None:
    source = np.asarray([0, 1, 2, 3, 3, 2, 1], dtype=np.uint8)
    packed = pack_q2_codes(source)
    assert packed.nbytes == 2
    np.testing.assert_array_equal(unpack_q2_codes(packed, len(source)), source)


def test_nondivisible_width_decodes_exact_serialized_codes() -> None:
    rng = np.random.default_rng(12)
    target = rng.normal(size=(3, 70)).astype(np.float32)
    encoding = quantize_int2_stage(target, 64, "signed_zero_mse")
    assert encoding.groups == 6
    assert encoding.packed_codes.nbytes == 96
    np.testing.assert_array_equal(decode_encoding(encoding), encoding.reconstruction)


def test_zero_capable_codebooks_represent_zero_exactly() -> None:
    target = np.zeros((4, 128), dtype=np.float32)
    for mode in ("signed_zero_rtn", "signed_zero_mse", "affine_rtn", "affine_mse", "centroid4"):
        encoding = quantize_int2_stage(target, 64, mode)
        np.testing.assert_array_equal(decode_encoding(encoding), target)


def test_legacy_q2_uses_serialized_fp16_scales() -> None:
    rng = np.random.default_rng(13)
    target = rng.normal(size=(5, 128)).astype(np.float32)
    encoding = encode_legacy_q2(target, 64, scale_storage="fp16")
    assert encoding.stored_scales.dtype == np.float16
    np.testing.assert_array_equal(decode_encoding(encoding), encoding.reconstruction)
    # The fitted FP32 values are deliberately distinct from their stored form
    # for this sample, proving the execution path does not retain ideal scales.
    assert np.any(encoding.fitted_scales != encoding.stored_scales.astype(np.float32))


def test_legacy_q2_fp32_reproduces_historical_implementation() -> None:
    rng = np.random.default_rng(131)
    target = rng.normal(size=(7, 130)).astype(np.float32)
    moments = np.exp(rng.normal(size=130)).astype(np.float64)
    historical, _ = q2_quantize(target, 64, moments)
    serialized = encode_legacy_q2(target, 64, moments, scale_storage="fp32")
    np.testing.assert_allclose(serialized.reconstruction, historical, rtol=2e-6, atol=2e-6)


def test_rrq_stages_quantize_the_decoded_remaining_residual() -> None:
    rng = np.random.default_rng(14)
    reference = rng.normal(size=(8, 128)).astype(np.float32)
    base = encode_legacy_q2(reference, 64).reconstruction
    specs = [
        QuantizerSpec("signed_zero_mse", 64),
        QuantizerSpec("affine_mse", 128),
        QuantizerSpec("midrise_mse", 64),
    ]
    series = build_rrq_series(base, reference, specs)
    running = base.copy()
    for depth, stage in enumerate(series.stages, start=1):
        running += decode_encoding(stage)
        np.testing.assert_array_equal(series.prefix(depth), running)
    assert series.residual_norms[-1] < series.residual_norms[0]


def test_rrq_gemv_is_sum_of_serialized_stage_gemvs() -> None:
    rng = np.random.default_rng(15)
    reference = rng.normal(size=(9, 64)).astype(np.float32)
    x = rng.normal(size=64).astype(np.float32)
    base = encode_legacy_q2(reference, 64).reconstruction
    series = build_rrq_series(base, reference, [QuantizerSpec("affine_mse", 64)] * 3)
    expected = base @ x
    for stage in series.stages:
        expected += decode_encoding(stage) @ x
    np.testing.assert_allclose(series.prefix(3) @ x, expected, rtol=2e-6, atol=2e-6)


def test_serialized_size_hashes_and_page_rounding(tmp_path: Path) -> None:
    rng = np.random.default_rng(16)
    target = rng.normal(size=(4, 130)).astype(np.float32)
    encoding = quantize_int2_stage(target, 64, "affine_mse")
    # 4 rows * 3 padded groups: 192 code bytes, 24 scale bytes,
    # 12 byte-aligned zero points, and one 64-byte header.
    assert encoding.stream_bytes() == {
        "header": 64,
        "codes": 192,
        "scales": 24,
        "zero_points": 12,
        "centroids": 0,
    }
    manifest = serialize_encoding(tmp_path, "stage", encoding)
    assert manifest["serialized_bytes"] == 292
    assert encoding.physical_read_bytes(512) == 3 * 512
    for item in manifest["files"].values():
        assert len(item["sha256"]) == 64


def test_scale_storage_formats_have_exact_accounting() -> None:
    rng = np.random.default_rng(17)
    target = rng.normal(size=(4, 128)).astype(np.float32)
    fp16 = quantize_int2_stage(target, 64, "signed_zero_mse", scale_storage="fp16")
    bf16 = quantize_int2_stage(target, 64, "signed_zero_mse", scale_storage="bf16")
    fp32 = quantize_int2_stage(target, 64, "signed_zero_mse", scale_storage="fp32")
    assert fp16.stored_scales.nbytes == bf16.stored_scales.nbytes
    assert fp32.stored_scales.nbytes == 2 * fp16.stored_scales.nbytes
    np.testing.assert_array_equal(decode_encoding(fp16), fp16.reconstruction)
    np.testing.assert_array_equal(decode_encoding(bf16), bf16.reconstruction)
    np.testing.assert_array_equal(decode_encoding(fp32), fp32.reconstruction)


def test_full_stage_rates_include_scale_and_zero_point_bytes() -> None:
    shape = (512, 2048)
    target = np.ones(shape, dtype=np.float32)
    g64 = quantize_int2_stage(target, 64, "signed_zero_mse")
    g128 = quantize_int2_stage(target, 128, "signed_zero_mse")
    affine64 = quantize_int2_stage(target, 64, "affine_mse")
    assert 8 * (g64.serialized_bytes() - 64) / target.size == 2.25
    assert 8 * (g128.serialized_bytes() - 64) / target.size == 2.125
    # Actual implementation uses one byte-aligned zero point per group.
    assert 8 * (affine64.serialized_bytes() - 64) / target.size == 2.375
