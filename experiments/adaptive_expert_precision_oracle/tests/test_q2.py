from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from oracle_study.quant import binary_quantize
from run_progressive_q2_remote import q2_quantize


def test_q2_is_deterministic_and_improves_training_objective() -> None:
    rng = np.random.default_rng(20260817)
    weights = rng.normal(size=(16, 128)).astype(np.float32)
    moments = np.exp(rng.normal(size=128)).astype(np.float64)
    first, scales_first = q2_quantize(weights, 64, moments)
    second, scales_second = q2_quantize(weights, 64, moments)
    binary, _ = binary_quantize(weights, 64, moments)
    assert np.array_equal(first, second)
    assert np.array_equal(scales_first, scales_second)
    q2_damage = np.sum((weights - first) ** 2 * moments[None, :])
    binary_damage = np.sum((weights - binary) ** 2 * moments[None, :])
    assert q2_damage < binary_damage
