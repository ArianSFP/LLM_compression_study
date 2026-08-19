"""Focused tests for the validation-only tile continuation machinery."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_sparse_streaming_tile_followup as followup
from oracle_study import sparse_streaming_cuda as cuda


def _fixture() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(20260819)
    hidden, inputs, outputs = 64, 64, 9
    gate2 = torch.randn(hidden, inputs, generator=generator) * 0.1
    up2 = torch.randn(hidden, inputs, generator=generator) * 0.1
    down2 = torch.randn(outputs, hidden, generator=generator) * 0.1
    gate4 = gate2 + torch.randn(hidden, inputs, generator=generator) * 0.02
    up4 = up2 + torch.randn(hidden, inputs, generator=generator) * 0.02
    down4 = down2 + torch.randn(outputs, hidden, generator=generator) * 0.02
    activation = torch.randn(inputs, generator=generator)
    proxy = torch.randn(outputs, generator=generator)
    return gate2, up2, down2, gate4, up4, down4, activation, proxy


def _output(gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return down @ (F.silu(gate @ x) * (up @ x))


def test_cartesian_selector_factorizes_tile_scores_and_reaches_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fixture()
    # Four 32x32 tiles followed by 64 down pages.  The expected Cartesian
    # tile scores are outer([10, 6], [5, 11]) / 16, hence pages 1,3,0,2.
    mocked = torch.cat((torch.tensor([1.0, 9.0, 4.0, 2.0]), torch.zeros(64)))
    monkeypatch.setattr(cuda, "_wina_proxy_scores", lambda *_: mocked.clone())
    geometry = cuda.MixedPageGeometry(64, 64, 32, 32)
    trace = cuda.static_cartesian_mixed_page_order(
        *values[:7], tile_shape=(32, 32),
        page_budgets=[4, geometry.total_pages], proxy=values[7], beta=0.1,
        device="cpu",
    )
    assert trace.order[:4].tolist() == [1, 3, 0, 2]
    assert len(set(trace.order.tolist())) == geometry.total_pages
    torch.testing.assert_close(
        trace.snapshots[geometry.total_pages],
        _output(values[3], values[4], values[5], values[6]),
        rtol=2e-5, atol=2e-5,
    )


def test_followup_grid_is_exactly_the_missing_predeclared_cells() -> None:
    identities = {(spec.tile_shape, spec.predeclared_selector) for spec in followup.FOLLOWUP_SPECS}
    assert identities == {
        ((16, 64), "exact_dynamic_tile_marginal"),
        ((16, 64), "activation_energy_x_weight"),
        ((16, 64), "cartesian_hidden_input_blocks"),
        ((32, 32), "cartesian_hidden_input_blocks"),
        ((64, 16), "exact_dynamic_tile_marginal"),
        ((64, 16), "activation_energy_x_weight"),
        ((64, 16), "cartesian_hidden_input_blocks"),
    }
    assert followup.REUSED_32_SELECTORS == {
        "exact_dynamic_tile_marginal": "exact_dynamic_tile_marginal_refresh_1",
        "activation_energy_x_weight": "static_wina",
    }


def test_followup_design_fails_closed_on_split_or_grid_change() -> None:
    config = {
        "page_size_bytes": 512,
        "activation_shortlist_unit": "input_coordinates",
        "coordinate_score_aggregation": "sum_stage_energy",
        "shortlist_physical_layout": "canonical_paired_planes_four_coordinates_per_projection_page",
        "tile_pilot_shape": [32, 32],
        "conditional_followup_tile_shapes": [[16, 64], [32, 32], [64, 16]],
        "conditional_followup_tile_selectors": [
            "exact_dynamic_tile_marginal", "activation_energy_x_weight",
            "cartesian_hidden_input_blocks",
        ],
        "promotion_policy": {"selection_split": "validation", "no_test_tuning": True, "shortlist_min_overfetch": 1.0},
    }
    followup.validate_design(config)
    config["promotion_policy"]["selection_split"] = "test"
    with pytest.raises(RuntimeError, match="validation selection"):
        followup.validate_design(config)
