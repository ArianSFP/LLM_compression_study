from __future__ import annotations

import pytest
import numpy as np
import torch
import torch.nn.functional as F

from oracle_study.sparse_streaming_cuda import (
    MixedPageGeometry,
    exact_mixed_page_greedy,
    exact_progressive_neuron_page_greedy,
    exact_unit_fixed_greedy,
    qenergy,
    static_proxy_mixed_page_order,
)
from oracle_study.sparse_streaming import (
    NeuronMajorDecomposition,
    exact_neuron_marginal_per_page_greedy,
    variable_cost_fixed_vector_greedy,
)


def _metric_inner(left: torch.Tensor, right: torch.Tensor, proxy: torch.Tensor, beta: float) -> torch.Tensor:
    if proxy.ndim == 1:
        proxy_term = torch.dot(left, proxy) * torch.dot(right, proxy)
    else:
        proxy_term = torch.sum((left @ proxy) * (right @ proxy))
    return torch.dot(left, right) + beta * proxy_term


def test_exact_unit_fixed_greedy_matches_direct_qenergy_reference() -> None:
    generator = torch.Generator().manual_seed(20260819)
    corrections = torch.randn(9, 7, generator=generator)
    proxy = torch.randn(7, 3, generator=generator)
    beta = 0.37
    trace = exact_unit_fixed_greedy(
        corrections,
        budgets=[0, 1, 4, 9],
        base_output=torch.linspace(-0.3, 0.2, 7),
        proxy=proxy,
        beta=beta,
        device="cpu",
    )

    residual = corrections.sum(dim=0)
    available = list(range(len(corrections)))
    reference: list[int] = []
    for _ in range(len(corrections)):
        scores = [
            2.0 * _metric_inner(corrections[j], residual, proxy, beta)
            - _metric_inner(corrections[j], corrections[j], proxy, beta)
            for j in available
        ]
        position = int(torch.argmax(torch.stack(scores)).item())
        chosen = available.pop(position)
        reference.append(chosen)
        residual -= corrections[chosen]
    assert trace.order.tolist() == reference
    torch.testing.assert_close(
        trace.snapshots[9],
        torch.linspace(-0.3, 0.2, 7) + corrections.sum(dim=0),
    )


def test_exact_unit_custom_target_and_gain_telescope() -> None:
    corrections = torch.tensor(
        [[1.0, 0.0], [0.0, 2.0], [-0.5, 0.25]],
        dtype=torch.float32,
    )
    target = torch.tensor([0.25, 1.5])
    trace = exact_unit_fixed_greedy(corrections, target_correction=target, budgets=[0, 3])
    initial = qenergy(target)
    final = qenergy(target - trace.snapshots[3])
    torch.testing.assert_close(trace.gains.sum(), (initial - final).double())
    assert sorted(trace.order.tolist()) == [0, 1, 2]


def test_progressive_neuron_page_greedy_matches_numpy_proxy_oracle() -> None:
    generator = torch.Generator().manual_seed(1917)
    corrections = torch.randn(2, 9, 13, generator=generator) * 0.2
    base = torch.randn(13, generator=generator) * 0.1
    proxy = torch.randn(13, 4, generator=generator)
    beta = 0.23
    budgets = [0, 1, 2, 3, 4, 7, 13, 27]
    trace = exact_progressive_neuron_page_greedy(
        corrections,
        page_budgets=budgets,
        base_output=base,
        proxy=proxy,
        beta=beta,
        device="cpu",
    )

    correction_np = corrections.numpy().astype(np.float64)
    levels = np.stack((
        np.zeros_like(correction_np[0]),
        correction_np[0],
        correction_np[0] + correction_np[1],
    ))
    numpy_trace = exact_neuron_marginal_per_page_greedy(
        NeuronMajorDecomposition(levels),
        page_budget=27,
        proxy=proxy.numpy().astype(np.float64),
        beta=beta,
    )
    expected_order = [[action.coordinate, action.stage] for action in numpy_trace.actions]
    assert trace.order.tolist() == expected_order
    torch.testing.assert_close(
        trace.gains,
        torch.tensor([action.score for action in numpy_trace.actions], dtype=torch.float64),
        rtol=2e-5,
        atol=2e-5,
    )
    assert trace.incremental_pages.tolist() == [2 if stage == 1 else 1 for _, stage in expected_order]
    assert trace.cumulative_pages.tolist() == list(np.cumsum(trace.incremental_pages.numpy()))

    depth = torch.zeros(9, dtype=torch.int64)
    approximation = base.clone()
    position = 0
    for budget in budgets:
        while position < len(trace.order) and int(trace.cumulative_pages[position]) <= budget:
            coordinate, stage = map(int, trace.order[position].tolist())
            assert stage == int(depth[coordinate]) + 1
            depth[coordinate] = stage
            approximation += corrections[stage - 1, coordinate]
            position += 1
        torch.testing.assert_close(trace.snapshots[budget], approximation, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        trace.snapshots[27],
        base + corrections.sum(dim=(0, 1)),
        rtol=2e-5,
        atol=2e-5,
    )
    assert torch.all(depth == 2)


@pytest.mark.parametrize("budget", [1, 2, 3, 5, 8, 13, 18])
def test_progressive_neuron_matches_numpy_at_each_physical_budget(budget: int) -> None:
    generator = torch.Generator().manual_seed(4242)
    corrections = torch.randn(2, 6, 8, generator=generator) * 0.17
    proxy = torch.randn(8, 3, generator=generator)
    beta = 0.31
    torch_trace = exact_progressive_neuron_page_greedy(
        corrections, page_budgets=[0, budget], proxy=proxy, beta=beta,
    )
    correction_np = corrections.numpy().astype(np.float64)
    levels = np.stack((
        np.zeros_like(correction_np[0]), correction_np[0], correction_np.sum(axis=0),
    ))
    numpy_trace = exact_neuron_marginal_per_page_greedy(
        NeuronMajorDecomposition(levels),
        page_budget=budget,
        proxy=proxy.numpy().astype(np.float64),
        beta=beta,
    )
    expected = [[action.coordinate, action.stage] for action in numpy_trace.actions]
    assert torch_trace.order.tolist() == expected
    assert torch_trace.incremental_pages.tolist() == [
        2 if action.stage == 1 else 1 for action in numpy_trace.actions
    ]
    assert len(torch_trace.cumulative_pages) == len(expected)
    if expected:
        assert int(torch_trace.cumulative_pages[-1]) <= budget
    approximation = torch.zeros(8)
    for coordinate, stage in expected:
        approximation += corrections[stage - 1, coordinate]
    torch.testing.assert_close(torch_trace.snapshots[budget], approximation, rtol=2e-5, atol=2e-5)


def test_progressive_page_snapshots_hold_last_state_across_unfilled_budget() -> None:
    corrections = torch.zeros(2, 2, 4)
    corrections[0, 0, 0] = 10.0
    corrections[0, 1, 1] = 8.0
    base = torch.tensor([0.5, -0.5, 0.25, -0.25])
    first = exact_progressive_neuron_page_greedy(
        corrections, page_budgets=[0, 1, 2, 3, 4], base_output=base,
    )
    second = exact_progressive_neuron_page_greedy(
        corrections, page_budgets=[0, 1, 2, 3, 4], base_output=base,
    )
    assert first.order.tolist() == [[0, 1], [1, 1]]
    assert first.incremental_pages.tolist() == [2, 2]
    assert first.cumulative_pages.tolist() == [2, 4]
    torch.testing.assert_close(first.snapshots[0], base)
    torch.testing.assert_close(first.snapshots[1], base)
    torch.testing.assert_close(first.snapshots[3], first.snapshots[2])
    torch.testing.assert_close(first.order, second.order)
    torch.testing.assert_close(first.gains, second.gains)


def test_progressive_cross_stage_efficiency_tie_matches_numpy_tuple() -> None:
    # After action (unit 0, stage 1), the remaining stage-1 action has
    # gain/cost = -2/2 and the newly eligible stage-2 action has -1/1.
    # Equal efficiency must be broken by higher raw gain, selecting stage 2
    # despite its higher global action ID.  This catches an ID-only tie-break.
    corrections = torch.zeros(2, 2, 4)
    corrections[0, 0, 0] = 1.0
    corrections[0, 1, 1] = 1.0
    corrections[1, 0, 2] = 1.0
    corrections[1, 1, 3] = 1.0
    target = torch.tensor([2.5, -0.5, 0.0, 0.0])
    torch_trace = exact_progressive_neuron_page_greedy(
        corrections, page_budgets=[6], target_correction=target,
    )
    numpy_trace = variable_cost_fixed_vector_greedy(
        corrections.numpy(),
        target.numpy(),
        {
            0: (0, 2), 1: (3, 5),
            2: (1, 2), 3: (4, 5),
        },
        page_budget=6,
    )
    expected = [[action.coordinate, action.stage] for action in numpy_trace.actions]
    assert expected[:2] == [[0, 1], [0, 2]]
    assert torch_trace.order.tolist() == expected
    torch.testing.assert_close(
        torch_trace.gains,
        torch.tensor([action.score for action in numpy_trace.actions], dtype=torch.float64),
    )


def _fixture() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(811)
    hidden, inputs, outputs = 64, 64, 11
    gate2 = torch.randn(hidden, inputs, generator=generator) * 0.12
    up2 = torch.randn(hidden, inputs, generator=generator) * 0.12
    down2 = torch.randn(outputs, hidden, generator=generator) * 0.12
    gate4 = gate2 + torch.randn(hidden, inputs, generator=generator) * 0.025
    up4 = up2 + torch.randn(hidden, inputs, generator=generator) * 0.025
    down4 = down2 + torch.randn(outputs, hidden, generator=generator) * 0.025
    activation = torch.randn(inputs, generator=generator)
    proxy = torch.randn(outputs, generator=generator)
    return gate2, up2, down2, gate4, up4, down4, activation, proxy


def _output(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    activation: torch.Tensor,
) -> torch.Tensor:
    return down @ (F.silu(gate @ activation) * (up @ activation))


def _single_page_output(values: tuple[torch.Tensor, ...], shape: tuple[int, int], page: int) -> torch.Tensor:
    gate2, up2, down2, gate4, up4, down4, activation, _ = values
    geometry = MixedPageGeometry(gate2.shape[0], gate2.shape[1], *shape)
    gate = gate2.clone()
    up = up2.clone()
    down = down2.clone()
    kind, first, second = geometry.decode(page)
    if kind == "tile":
        assert second is not None
        row_start = first * shape[0]
        row_stop = row_start + shape[0]
        column_start = second * shape[1]
        column_stop = column_start + shape[1]
        gate[row_start:row_stop, column_start:column_stop] = gate4[
            row_start:row_stop, column_start:column_stop
        ]
        up[row_start:row_stop, column_start:column_stop] = up4[
            row_start:row_stop, column_start:column_stop
        ]
    else:
        down[:, first] = down4[:, first]
    return _output(gate, up, down, activation)


@pytest.mark.parametrize("shape", [(16, 64), (32, 32), (64, 16)])
def test_exact_mixed_first_page_matches_exhaustive_nonlinear_qenergy(shape: tuple[int, int]) -> None:
    values = _fixture()
    gate2, up2, down2, gate4, up4, down4, activation, proxy = values
    geometry = MixedPageGeometry(64, 64, *shape)
    base = _output(gate2, up2, down2, activation)
    target = _output(gate4, up4, down4, activation)
    residual = target - base
    beta = 0.2
    scores = []
    for page in range(geometry.total_pages):
        correction = _single_page_output(values, shape, page) - base
        score = 2.0 * _metric_inner(correction, residual, proxy, beta)
        score -= _metric_inner(correction, correction, proxy, beta)
        scores.append(score)
    expected = int(torch.argmax(torch.stack(scores)).item())

    trace = exact_mixed_page_greedy(
        gate2, up2, down2, gate4, up4, down4, activation,
        tile_shape=shape,
        page_budgets=[0, 1],
        proxy=proxy,
        beta=beta,
        device="cpu",
    )
    assert trace.order.tolist() == [expected]
    torch.testing.assert_close(trace.snapshots[1], _single_page_output(values, shape, expected))


def test_exact_mixed_full_path_reaches_refined_endpoint_and_snapshots_pages() -> None:
    gate2, up2, down2, gate4, up4, down4, activation, proxy = _fixture()
    geometry = MixedPageGeometry(64, 64, 32, 32)
    budgets = [0, 1, 7, geometry.total_pages]
    trace = exact_mixed_page_greedy(
        gate2, up2, down2, gate4, up4, down4, activation,
        tile_shape=(32, 32),
        page_budgets=budgets,
        refresh_interval=1,
        proxy=proxy,
        beta=0.13,
    )
    assert len(trace.order) == geometry.total_pages
    assert len(set(trace.order.tolist())) == geometry.total_pages
    assert sorted(trace.snapshots) == budgets
    torch.testing.assert_close(
        trace.snapshots[geometry.total_pages],
        _output(gate4, up4, down4, activation),
        rtol=2e-5,
        atol=2e-5,
    )
    initial_residual = _output(gate4, up4, down4, activation) - trace.snapshots[0]
    final_residual = _output(gate4, up4, down4, activation) - trace.snapshots[geometry.total_pages]
    expected_gain = qenergy(initial_residual, proxy=proxy, beta=0.13)
    expected_gain -= qenergy(final_residual, proxy=proxy, beta=0.13)
    torch.testing.assert_close(trace.gains.sum(), expected_gain.double(), rtol=2e-5, atol=2e-5)


def test_block_refresh_keeps_exact_state_and_unique_pages() -> None:
    gate2, up2, down2, gate4, up4, down4, activation, _ = _fixture()
    geometry = MixedPageGeometry(64, 64, 16, 64)
    trace = exact_mixed_page_greedy(
        gate2, up2, down2, gate4, up4, down4, activation,
        tile_shape=(16, 64),
        page_budgets=[9, geometry.total_pages],
        refresh_interval=4,
    )
    assert len(set(trace.order.tolist())) == geometry.total_pages
    torch.testing.assert_close(
        trace.snapshots[geometry.total_pages],
        _output(gate4, up4, down4, activation),
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.parametrize("method", ["first_order", "wina"])
def test_static_proxy_orders_apply_exact_pages_and_reach_endpoint(method: str) -> None:
    gate2, up2, down2, gate4, up4, down4, activation, proxy = _fixture()
    geometry = MixedPageGeometry(64, 64, 64, 16)
    trace = static_proxy_mixed_page_order(
        gate2, up2, down2, gate4, up4, down4, activation,
        tile_shape=(64, 16),
        page_budgets=[0, 5, geometry.total_pages],
        proxy_method=method,
        proxy=proxy,
        beta=0.07,
        device="cpu",
    )
    assert len(set(trace.order.tolist())) == geometry.total_pages
    torch.testing.assert_close(
        trace.snapshots[geometry.total_pages],
        _output(gate4, up4, down4, activation),
        rtol=2e-5,
        atol=2e-5,
    )


def test_validation_rejects_nonphysical_geometry_and_qenergy_without_proxy() -> None:
    values = _fixture()
    with pytest.raises(ValueError, match="tile_shape"):
        exact_mixed_page_greedy(*values[:7], tile_shape=(8, 128), page_budgets=[1])
    with pytest.raises(ValueError, match="proxy"):
        exact_unit_fixed_greedy(torch.ones(2, 3), budgets=[1], beta=0.1)
