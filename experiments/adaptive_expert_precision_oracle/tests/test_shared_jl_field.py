import numpy as np

from oracle_study.interaction_field import JointInteractionFactor
from oracle_study.neuron_selector import UnitScoreMetadata
from oracle_study.shared_jl_field import (
    calibrated_shared_factor,
    quantized_shared_factor_with_pair_calibration,
    rademacher_projection,
)


def test_rademacher_projection_is_deterministic_and_normalized():
    first = rademacher_projection(17, 8, 91)
    second = rademacher_projection(17, 8, 91)
    assert np.array_equal(first, second)
    assert np.allclose(
        np.sort(np.unique(first)),
        np.asarray((-1.0 / np.sqrt(8), 1.0 / np.sqrt(8))),
    )


def test_calibration_recovers_exact_local_gram_and_int4_is_self_safe():
    rng = np.random.default_rng(13)
    units, tail, proxy = 7, 12, 4
    d4 = rng.normal(size=(units, 31))
    d2 = rng.normal(size=(units, 31))
    metric_proxy = rng.normal(size=(31, proxy)) / np.sqrt(31)
    beta = 0.2
    metadata = UnitScoreMetadata(
        np.einsum("ud,ud->u", d4, d4) + beta * np.sum((d4 @ metric_proxy) ** 2, axis=1),
        np.einsum("ud,ud->u", d4, d2) + beta * np.sum((d4 @ metric_proxy) * (d2 @ metric_proxy), axis=1),
        np.einsum("ud,ud->u", d2, d2) + beta * np.sum((d2 @ metric_proxy) ** 2, axis=1),
    )
    projection = rademacher_projection(31, tail, 7)
    root = np.sqrt(beta)
    fp32 = calibrated_shared_factor(
        d4 @ projection, d2 @ projection,
        root * (d4 @ metric_proxy), root * (d2 @ metric_proxy),
        metadata, encoding=None,
    )
    for unit in range(units):
        rows = np.stack((fp32.factor.l4[unit], fp32.factor.l2[unit])).astype(np.float64)
        target = np.asarray((
            (metadata.a[unit], metadata.b[unit]),
            (metadata.b[unit], metadata.c[unit]),
        ))
        assert np.allclose(rows @ rows.T, target, rtol=2e-6, atol=2e-5)
    encoded = calibrated_shared_factor(
        d4 @ projection, d2 @ projection,
        root * (d4 @ metric_proxy), root * (d2 @ metric_proxy),
        metadata, encoding="int4_per_row",
    )
    assert encoded.encoded is not None
    assert encoded.encoded.payload_bytes == units * (tail + proxy) + 4 * units + 4
    assert 0.0 < encoded.self_safe_scale <= 1.0
    for unit in range(units):
        rows = np.stack((encoded.factor.l4[unit], encoded.factor.l2[unit])).astype(np.float64)
        target = np.asarray((
            (metadata.a[unit], metadata.b[unit]),
            (metadata.b[unit], metadata.c[unit]),
        ))
        assert np.min(np.linalg.eigvalsh(target - rows @ rows.T)) >= -1e-7



def test_sign_and_ternary_pair_calibration_are_compact_and_self_safe():
    rng = np.random.default_rng(27)
    units, rank = 9, 16
    high = rng.normal(size=(units, rank)).astype(np.float32)
    low = rng.normal(size=(units, rank)).astype(np.float32)
    metadata = UnitScoreMetadata(
        np.sum(high.astype(np.float64) ** 2, axis=1),
        np.sum(high.astype(np.float64) * low.astype(np.float64), axis=1),
        np.sum(low.astype(np.float64) ** 2, axis=1),
    )
    source = JointInteractionFactor(high, low, "synthetic", rank)
    for encoding, bits in (("sign_per_row", 1), ("ternary_per_row", 2)):
        result = quantized_shared_factor_with_pair_calibration(
            source, metadata, encoding=encoding,
        )
        expected = (
            (2 * units * rank * bits + 7) // 8
            + 2 * units * 2
            + units * 4 * 2
            + 4
        )
        assert result.factor.payload_bytes == expected
        assert 0.0 < result.self_safe_scale <= 1.0
        for unit in range(units):
            rows = np.stack((
                result.factor.l4[unit], result.factor.l2[unit],
            )).astype(np.float64)
            target = np.asarray((
                (metadata.a[unit], metadata.b[unit]),
                (metadata.b[unit], metadata.c[unit]),
            ))
            assert np.min(np.linalg.eigvalsh(target - rows @ rows.T)) >= -1e-7
