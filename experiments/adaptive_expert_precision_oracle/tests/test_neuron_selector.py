from __future__ import annotations

import numpy as np
import pytest
import torch

from oracle_study.neuron_selector import (
    D_AFTER_G,
    D_FROM_BASE,
    G_AFTER_D,
    G_FROM_BASE,
    FactorizedUnitOutputs,
    complete_unit_scores,
    descending_order,
    encode_array,
    encode_response_model,
    encode_unit_score_metadata,
    exact_factorized_neuron_fixed_greedy,
    factorized_unit_outputs,
    fit_activation_response_model,
    ndcg_at_k,
    predict_q4_hidden,
    predict_residual_response,
    unit_score_metadata,
    utility_weighted_recall,
)


def qenergy(value: np.ndarray, proxy: np.ndarray, beta: float) -> float:
    x = np.asarray(value, np.float64)
    return float(x @ x + beta * np.square(x @ proxy).sum())


def brute_factorized(outputs: FactorizedUnitOutputs, proxy: np.ndarray, beta: float):
    vectors = outputs.transitions
    state = np.zeros(outputs.units, np.int64)
    residual = outputs.target_output - outputs.base_output
    result = []
    pages = 0
    while len(result) < 2 * outputs.units:
        candidates = []
        for unit in range(outputs.units):
            blocks = (G_FROM_BASE, D_FROM_BASE) if state[unit] == 0 else (
                (D_AFTER_G,) if state[unit] == 1 else (G_AFTER_D,) if state[unit] == 2 else ()
            )
            for block in blocks:
                correction = vectors[block, unit]
                gain = qenergy(residual, proxy, beta) - qenergy(residual - correction, proxy, beta)
                cost = (2, 1, 1, 2)[block]
                action_id = block * outputs.units + unit
                candidates.append((gain / cost, gain, -cost, -action_id, action_id, block, unit, cost))
        *_, action_id, block, unit, cost = max(candidates)
        correction = vectors[block, unit]
        gain = qenergy(residual, proxy, beta) - qenergy(residual - correction, proxy, beta)
        residual -= correction
        state[unit] = 1 if block == G_FROM_BASE else 2 if block == D_FROM_BASE else 3
        pages += cost
        result.append((action_id, gain, pages))
    return result


def test_factorized_outputs_sum_to_exact_mixed_states() -> None:
    rng = np.random.default_rng(4)
    units, inputs, outputs = 5, 7, 6
    q2 = (rng.normal(size=(units, inputs)), rng.normal(size=(units, inputs)), rng.normal(size=(outputs, units)))
    q4 = tuple(value + 0.2 * rng.normal(size=value.shape) for value in q2)
    x = rng.normal(size=inputs)
    value = factorized_unit_outputs(q2, q4, x)
    gate2, up2, down2 = q2
    gate4, up4, down4 = q4
    h2 = (gate2 @ x) / (1.0 + np.exp(-(gate2 @ x))) * (up2 @ x)
    h4 = (gate4 @ x) / (1.0 + np.exp(-(gate4 @ x))) * (up4 @ x)
    np.testing.assert_allclose(value.y00.sum(0), down2 @ h2, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(value.y10.sum(0), down2 @ h4, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(value.y01.sum(0), down4 @ h2, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(value.y11.sum(0), down4 @ h4, rtol=1e-11, atol=1e-11)


def test_factorized_gpu_greedy_matches_direct_nonidentity_metric_and_endpoint() -> None:
    rng = np.random.default_rng(18)
    units, output = 6, 9
    y00 = rng.normal(size=(units, output))
    y10 = y00 + rng.normal(size=(units, output))
    y01 = y00 + rng.normal(size=(units, output))
    y11 = y10 + y01 - y00 + 0.1 * rng.normal(size=(units, output))
    values = FactorizedUnitOutputs(y00, y10, y01, y11, rng.normal(size=units), rng.normal(size=units))
    proxy = rng.normal(size=(output, 3))
    beta = 0.37
    expected = brute_factorized(values, proxy, beta)
    trace = exact_factorized_neuron_fixed_greedy(
        values, page_budgets=[3 * units], proxy=proxy, beta=beta, device="cpu",
    )
    assert trace.order.tolist() == [row[0] for row in expected]
    np.testing.assert_allclose(trace.gains.numpy(), [row[1] for row in expected], rtol=2e-5, atol=2e-5)
    assert trace.cumulative_pages.tolist() == [row[2] for row in expected]
    assert trace.cumulative_pages[-1] == 3 * units
    np.testing.assert_allclose(trace.snapshots[3 * units].numpy(), values.target_output, rtol=2e-6, atol=2e-6)
    seen = np.zeros(units, np.int64)
    for action in trace.order.tolist():
        block, unit = divmod(action, units)
        if block in (G_FROM_BASE, D_FROM_BASE):
            assert seen[unit] == 0
            seen[unit] = 1 if block == G_FROM_BASE else 2
        elif block == D_AFTER_G:
            assert seen[unit] == 1
            seen[unit] = 3
        else:
            assert seen[unit] == 2
            seen[unit] = 3
    assert np.all(seen == 3)


def test_factorized_greedy_snapshots_never_exceed_budget() -> None:
    rng = np.random.default_rng(7)
    units, output = 4, 5
    y00 = rng.normal(size=(units, output))
    values = FactorizedUnitOutputs(
        y00, y00 + rng.normal(size=(units, output)),
        y00 + rng.normal(size=(units, output)), y00 + rng.normal(size=(units, output)),
        rng.normal(size=units), rng.normal(size=units),
    )
    trace = exact_factorized_neuron_fixed_greedy(values, page_budgets=range(3 * units + 1), device="cpu")
    for budget in range(3 * units + 1):
        assert budget in trace.snapshots
        prefix = int(np.sum(trace.cumulative_pages.numpy() <= budget))
        expected = values.base_output.copy()
        for action in trace.order[:prefix].tolist():
            block, unit = divmod(action, units)
            expected += values.transitions[block, unit]
        np.testing.assert_allclose(trace.snapshots[budget].numpy(), expected, rtol=2e-6, atol=2e-6)


def test_hidden_scalar_sufficient_statistic_matches_full_correction_qenergy() -> None:
    rng = np.random.default_rng(2)
    output, units = 11, 8
    down2 = rng.normal(size=(output, units))
    down4 = down2 + 0.4 * rng.normal(size=(output, units))
    h2 = rng.normal(size=units)
    h4 = rng.normal(size=units)
    proxy = rng.normal(size=(output, 4))
    beta = 0.19
    metadata = unit_score_metadata(down2, down4, proxy=proxy, beta=beta)
    score = complete_unit_scores(h2, h4, metadata)
    correction = down4.T * h4[:, None] - down2.T * h2[:, None]
    expected = np.array([qenergy(row, proxy, beta) for row in correction])
    np.testing.assert_allclose(score, expected, rtol=1e-11, atol=1e-11)


def test_fp16_scalar_metadata_is_compact_and_accurate() -> None:
    rng = np.random.default_rng(3)
    metadata = unit_score_metadata(rng.normal(size=(17, 9)), rng.normal(size=(17, 9)))
    encoded, storage = encode_unit_score_metadata(metadata, "fp16")
    assert storage == 3 * (9 * 2 + 4)
    for actual, approximate in zip((metadata.a, metadata.b, metadata.c), (encoded.a, encoded.b, encoded.c)):
        np.testing.assert_allclose(approximate, actual, rtol=8e-4, atol=8e-4)


def test_activation_response_fit_recovers_low_rank_response_on_heldout() -> None:
    rng = np.random.default_rng(5)
    samples, inputs, units, rank = 24, 6, 5, 2
    x = rng.normal(size=(samples, inputs))
    shared = rng.normal(size=(rank, inputs))
    residuals = {}
    for expert in (4, 9):
        residuals[expert] = (
            rng.normal(size=(units, rank)) @ shared,
            rng.normal(size=(units, rank)) @ shared,
        )
    model = fit_activation_response_model(
        x, residuals, rank=rank, separate_gate_up=False, ridge_relative=1e-10,
    )
    heldout = rng.normal(size=inputs)
    for expert, (gate, up) in residuals.items():
        predicted_gate, predicted_up = predict_residual_response(model, expert, heldout)
        np.testing.assert_allclose(predicted_gate, gate @ heldout, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(predicted_up, up @ heldout, rtol=2e-6, atol=2e-6)
    assert model.fit_diagnostics["training_response_relative_mse"] < 1e-12


def test_separate_response_fit_and_hidden_prediction() -> None:
    rng = np.random.default_rng(51)
    samples, inputs, units = 30, 7, 4
    x = rng.normal(size=(samples, inputs))
    gate_basis = rng.normal(size=(2, inputs))
    up_basis = rng.normal(size=(2, inputs))
    residuals = {3: (rng.normal(size=(units, 2)) @ gate_basis, rng.normal(size=(units, 2)) @ up_basis)}
    model = fit_activation_response_model(x, residuals, rank=2, separate_gate_up=True, ridge_relative=1e-10)
    heldout = rng.normal(size=inputs)
    gate2 = rng.normal(size=units)
    up2 = rng.normal(size=units)
    predicted = predict_q4_hidden(model, 3, heldout, gate2, up2)
    dg, du = residuals[3][0] @ heldout, residuals[3][1] @ heldout
    expected = (gate2 + dg) / (1.0 + np.exp(-(gate2 + dg))) * (up2 + du)
    np.testing.assert_allclose(predicted, expected, rtol=3e-6, atol=3e-6)


def test_response_encoding_accounts_shared_and_expert_bytes() -> None:
    rng = np.random.default_rng(8)
    x = rng.normal(size=(16, 5))
    residuals = {2: (rng.normal(size=(4, 5)), rng.normal(size=(4, 5)))}
    model = fit_activation_response_model(x, residuals, rank=3, separate_gate_up=False)
    encoded, accounting = encode_response_model(model, synthesis_encoding="int8_per_row")
    assert accounting["layer_analysis_bytes"] == 3 * 5 * 2
    assert accounting["expert_synthesis_bytes"] == 2 * (4 * 3 + 4 * 2)
    assert encoded.shared_analysis
    assert encode_array(np.ones((2, 3), np.float32), "fp16").storage_bytes == 12


def test_ranking_metrics_are_exact_for_perfect_and_worse_orders() -> None:
    score = np.array([9.0, 4.0, 1.0, 0.0])
    perfect = descending_order(score)
    reverse = perfect[::-1]
    assert utility_weighted_recall(score, perfect, 2) == pytest.approx(1.0)
    assert ndcg_at_k(score, perfect, 3) == pytest.approx(1.0)
    assert utility_weighted_recall(score, reverse, 2) < 0.1
    assert ndcg_at_k(score, reverse, 3) < 0.5


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="Torch has no float8")
def test_fp8_encoding_uses_one_byte_per_value() -> None:
    value = np.array([[0.1, -2.0, 5.5]], np.float32)
    encoded = encode_array(value, "fp8_e4m3fn_per_row")
    assert encoded.storage_bytes == value.size + 2
    assert np.all(np.isfinite(encoded.decoded))
