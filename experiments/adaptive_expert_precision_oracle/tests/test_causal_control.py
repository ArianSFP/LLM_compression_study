from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from oracle_study.causal_control import (
    FULLY_FROZEN,
    LIVE_ROUTES,
    POST_XPLUS_BF16,
    POST_XPLUS_FP32,
    PRE_RESIDUAL,
    ROUTE_MODES,
    bfloat16_ulp_metrics,
    build_control_cases,
    first_changed_layer,
    selector_and_execution_router_weights,
)
from run_same_host_causal_controls import _pre_residual_start


def test_control_grid_is_complete_and_deduplicated() -> None:
    alphas = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
    cases = build_control_cases(alphas)
    assert len(cases) == 3 * (1 + len(alphas)) + 2
    assert len(set(cases)) == len(cases)
    assert sum(case.alpha == 0.0 for case in cases) == len(ROUTE_MODES)
    assert {
        (case.injection_mode, case.route_mode)
        for case in cases if case.injection_mode != PRE_RESIDUAL
    } == {
        (POST_XPLUS_BF16, LIVE_ROUTES),
        (POST_XPLUS_FP32, LIVE_ROUTES),
    }
    assert any(
        case.injection_mode == PRE_RESIDUAL
        and case.route_mode == FULLY_FROZEN
        and case.alpha == 4.0
        for case in cases
    )
    with pytest.raises(ValueError, match="unique"):
        build_control_cases((1.0, 1.0))


def test_bfloat16_ulp_metrics_use_monotonic_signed_codes() -> None:
    reference = np.asarray([0x3F80, 0xBF80, 0x0000, 0x8000], np.uint16)
    candidate = np.asarray([0x3F81, 0xBF7F, 0x0002, 0x0000], np.uint16)
    numeric_equal = np.asarray([False, False, False, True])
    metrics = bfloat16_ulp_metrics(reference, candidate, numeric_equal)
    assert metrics["coordinates"] == 4
    assert metrics["unchanged_fraction"] == 0.25
    assert metrics["one_ulp_fraction"] == 0.5
    assert metrics["multi_ulp_fraction"] == 0.25
    assert metrics["max_ulp_distance"] == 2


def test_first_changed_layer_validation() -> None:
    assert first_changed_layer([(7, 0.0), (9, 0.1), (8, 0.2)]) == 8
    assert first_changed_layer([(7, 0.0), (9, 0.0)]) is None
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        first_changed_layer([(0, 1.1)])


def test_selector_and_execution_router_weights_preserve_backend_values() -> None:
    historical = np.full(8, 0.125, np.float64)
    selector, execution = selector_and_execution_router_weights(
        historical,
        historical_sum_atol=5e-7,
        execution_sum_atol=0.003,
    )
    assert np.array_equal(selector, historical)
    assert np.array_equal(execution, historical)

    rounded = np.asarray(
        [0.25, 0.20, 0.15, 0.125, 0.10, 0.075, 0.05, 0.048],
        np.float64,
    )
    selector, execution = selector_and_execution_router_weights(
        rounded,
        historical_sum_atol=5e-7,
        execution_sum_atol=0.003,
    )
    assert np.array_equal(execution, rounded)
    assert np.isclose(selector.sum(), 1.0, rtol=0.0, atol=5e-7)
    assert not np.array_equal(selector, execution)
    with pytest.raises(ValueError, match="sum tolerance"):
        selector_and_execution_router_weights(
            rounded * 0.9,
            historical_sum_atol=5e-7,
            execution_sum_atol=0.003,
        )


class _DummyExperts(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * 2


class _DummyMlp(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = _DummyExperts()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        routed = self.experts(value)
        shared = value * 3
        return routed + shared


class _DummyModel:
    def __init__(self) -> None:
        self._embedding = torch.nn.Embedding(2, 2, dtype=torch.bfloat16)
        layer = SimpleNamespace(mlp=_DummyMlp())
        language_model = SimpleNamespace(layers=[layer])
        self.model = SimpleNamespace(language_model=language_model)

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self._embedding


def test_pre_residual_path_replaces_only_routed_expert_output() -> None:
    model = _DummyModel()
    x = torch.tensor(
        [[1.0, -2.0], [0.5, 4.0]], dtype=torch.bfloat16,
    )
    residual = torch.tensor(
        [[10.0, 20.0], [-5.0, 1.0]], dtype=torch.bfloat16,
    )
    routed = x * 2
    delta = np.asarray(
        [[0.25, -0.5], [1.0, -2.0]], dtype=np.float32,
    )
    baseline = {
        "x": {0: x.float().numpy()},
        "residual": {0: residual.float().numpy()},
        "routed": {0: routed.float().numpy()},
    }
    start, facts = _pre_residual_start(model, baseline, 0, delta, 1.0)
    replacement = (
        routed.float() + torch.as_tensor(delta, dtype=torch.float32)
    ).to(torch.bfloat16)
    expected = residual.unsqueeze(0) + (
        replacement.unsqueeze(0) + x.unsqueeze(0) * 3
    )
    assert torch.equal(start, expected)
    assert facts["routed_replacement_applied"] is True
    assert facts["routed_repeat_max_abs"] == 0.0
