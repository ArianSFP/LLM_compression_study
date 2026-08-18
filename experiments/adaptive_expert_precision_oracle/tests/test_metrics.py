from __future__ import annotations

import numpy as np
import torch

from oracle_study.metrics import exact_greedy_approximations, metric_rows, router_metrics


def test_exact_greedy_full_support_recovers_correction():
    rng = np.random.default_rng(12)
    atoms = torch.from_numpy(rng.standard_normal((7, 9), dtype=np.float32))
    codes = torch.from_numpy(rng.standard_normal((3, 9), dtype=np.float32))
    correction = codes @ atoms.T
    approximations, choices = exact_greedy_approximations(atoms, codes, correction, [0, 9])
    assert choices.shape == (3, 9)
    np.testing.assert_allclose(approximations[9].numpy(), correction.numpy(), rtol=2e-5, atol=2e-5)


def test_metrics_keep_negative_recovery_and_router_sets():
    reference = np.asarray([[1.0, 0.0]], dtype=np.float32)
    base = np.asarray([[0.0, 0.0]], dtype=np.float32)
    bad = np.asarray([[-2.0, 0.0]], dtype=np.float32)
    assert metric_rows(reference, base, bad)["recovery"][0] < 0
    logits = np.arange(20, dtype=np.float32)[None, :]
    stable = router_metrics(logits, logits.copy())
    assert stable["top8_recall"][0] == 1.0
    assert stable["boundary_flip"][0] == 0.0
