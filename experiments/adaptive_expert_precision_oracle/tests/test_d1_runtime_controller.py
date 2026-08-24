from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oracle_study.d1_runtime_controller import (
    TokenLatentExpert,
    apply_precision_transition,
    apply_precision_transitions,
    apply_precision_transitions_torch,
    certify_topk,
    combined_remaining_latent,
    corrected_logit_metrics,
    predict_corrected_logits,
    project_candidate_sensitivities,
    rmsnorm_router_sensitivities,
    score_single_page_transitions,
    score_single_page_transitions_torch,
    signed_effect_metrics,
    token_hidden_responses,
)


def _expert(hidden: np.ndarray | None = None) -> TokenLatentExpert:
    return TokenLatentExpert(
        down2_coefficients=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        down4_coefficients=np.asarray([[2.0, 0.0], [0.0, 2.0]]),
        hidden_responses=(
            np.asarray([
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
            ])
            if hidden is None else np.asarray(hidden, np.float64)
        ),
    )


def test_dynamic_hidden_responses_preserve_gate_up_interactions() -> None:
    first = token_hidden_responses([1.0, 0.0], [2.0, 3.0], [2.0, 0.0], [4.0, 1.0])
    second = token_hidden_responses([0.0, 1.0], [2.0, 3.0], [0.0, 2.0], [4.0, 1.0])
    assert first.shape == (2, 4)
    assert not np.array_equal(first, second)
    assert first[0, 0] != first[0, 3]


def test_page_latent_is_token_dependent_and_conditional_on_state() -> None:
    first = _expert()
    second = _expert(np.asarray([
        [2.0, 4.0, 6.0, 8.0],
        [1.0, 1.5, 2.0, 2.5],
    ]))
    first_down_effect = first.unit_state_latent(0, 1) - first.unit_state_latent(0, 0)
    second_down_effect = second.unit_state_latent(0, 1) - second.unit_state_latent(0, 0)
    assert not np.array_equal(first_down_effect, second_down_effect)
    gate_then_down = first.unit_state_latent(0, 5) - first.unit_state_latent(0, 4)
    assert not np.array_equal(first_down_effect, gate_then_down)


def test_combined_tail_and_corrected_logits_can_flip_crude_route() -> None:
    experts = (_expert(), _expert())
    states = (np.asarray([0, 0]), np.asarray([7, 7]))
    remaining = combined_remaining_latent(experts, states, [0.75, 0.25])
    np.testing.assert_allclose(remaining, [5.25, 6.0])
    basis = np.eye(2)
    gradients = np.asarray([[1.0, 0.0], [0.0, -0.25]])
    projected = project_candidate_sensitivities(basis, gradients)
    corrected = predict_corrected_logits([1.0, 2.0], projected, remaining)
    assert corrected[0] > corrected[1]


def test_topk_certificate_requires_outsider_and_excluded_pool_separation() -> None:
    logits = np.asarray([5.0, 4.0, 3.0, 2.0])
    radius = np.full(4, 0.1)
    passed = certify_topk(logits, radius, top_k=2, candidate_ids=[10, 11, 12, 13])
    assert passed.certified is True
    assert set(passed.top_ids.tolist()) == {10, 11}
    assert passed.pool_complete is True
    failed = certify_topk(
        logits,
        radius,
        top_k=2,
        candidate_ids=[10, 11, 12, 13],
        excluded_upper_bound=4.5,
    )
    assert failed.certified is False
    assert failed.pool_complete is False


def test_signed_effect_and_corrected_logit_metrics_report_calibration() -> None:
    exact = np.asarray([3.0, -2.0, 1.0, -0.5])
    predicted = np.asarray([2.8, -2.1, 0.9, 0.2])
    metrics = signed_effect_metrics(
        predicted,
        exact,
        error_radius=np.asarray([0.3, 0.2, 0.2, 0.8]),
        top_pages=2,
    )
    assert metrics["pearson_correlation"] > 0.9
    assert metrics["sign_accuracy"] == 0.75
    assert metrics["top_page_overlap"] == 1.0
    assert metrics["interval_coverage"] == 1.0
    logits = corrected_logit_metrics(
        predicted.reshape(2, 2),
        exact.reshape(2, 2),
        error_radius=np.ones((2, 2)),
    )
    assert logits["corrected_logit_interval_coverage"] == 1.0
    assert logits["corrected_logit_rmse"] > 0.0


def test_transition_ranking_uses_dynamic_signed_effect_and_rejects_stale_action() -> None:
    experts = (_expert(),)
    states = (np.asarray([0, 7]),)
    actions = score_single_page_transitions(
        experts,
        states,
        [1.0],
        risk_adjoint=[-1.0, 0.0],
        uncertainty_reduction=np.zeros((1, 2, 3)),
        local_gain=np.ones((1, 2, 3)),
    )
    assert len(actions) == 3
    assert actions[0].unit == 0
    updated = apply_precision_transition(states, actions[0])
    assert updated[0][0] == actions[0].destination_state
    with pytest.raises(ValueError, match="stale"):
        apply_precision_transition(updated, actions[0])


def test_transition_shortlist_is_unique_and_batch_safe() -> None:
    experts = (_expert(), _expert())
    states = (np.asarray([0, 0]), np.asarray([0, 0]))
    actions = score_single_page_transitions(
        experts,
        states,
        [0.6, 0.4],
        risk_adjoint=[-1.0, -0.5],
        shortlist=3,
        unique_units=True,
    )
    assert len(actions) == 3
    assert len({(action.expert, action.unit) for action in actions}) == 3
    updated = apply_precision_transitions(states, actions)
    assert sum(np.count_nonzero(value) for value in updated) == 3
    with pytest.raises(ValueError, match="distinct"):
        apply_precision_transitions(states, (actions[0], actions[0]))


def test_tensor_transition_shortlist_matches_reference_scores() -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(7)
    down2 = rng.normal(size=(2, 5, 3))
    down4 = rng.normal(size=(2, 5, 3))
    hidden = rng.normal(size=(2, 5, 4))
    states = np.asarray([
        [0, 1, 2, 3, 4],
        [0, 2, 4, 6, 7],
    ], np.int64)
    weights = np.asarray([0.6, 0.4])
    adjoint = rng.normal(size=3)
    uncertainty = rng.uniform(0.0, 0.1, size=(2, 5, 3))
    local = rng.normal(size=(2, 5, 3))
    experts = tuple(
        TokenLatentExpert(down2[index], down4[index], hidden[index])
        for index in range(2)
    )
    reference = score_single_page_transitions(
        experts,
        tuple(states),
        weights,
        adjoint,
        uncertainty_reduction=uncertainty,
        local_gain=local,
        shortlist=6,
        unique_units=True,
    )
    batch = score_single_page_transitions_torch(
        torch.asarray(down2),
        torch.asarray(down4),
        torch.asarray(hidden),
        torch.asarray(states),
        torch.asarray(weights),
        torch.asarray(adjoint),
        uncertainty_reduction=torch.asarray(uncertainty),
        local_gain=torch.asarray(local),
        shortlist=6,
    )
    assert len(batch) == len(reference) == 6
    np.testing.assert_array_equal(
        batch.expert.cpu().numpy(), [item.expert for item in reference],
    )
    np.testing.assert_array_equal(
        batch.unit.cpu().numpy(), [item.unit for item in reference],
    )
    np.testing.assert_allclose(
        batch.score_per_byte.cpu().numpy(),
        [item.score_per_byte for item in reference],
    )
    updated = apply_precision_transitions_torch(torch.asarray(states), batch)
    assert int(torch.count_nonzero(updated != torch.asarray(states))) == 6
    with pytest.raises(ValueError, match="stale"):
        apply_precision_transitions_torch(updated, batch)


def test_analytic_rmsnorm_router_sensitivity_matches_finite_difference() -> None:
    hidden = np.asarray([0.7, -1.1, 0.3])
    norm_weight = np.asarray([1.2, 0.8, 1.5])
    router = np.asarray([[0.5, -0.2, 0.9], [-0.1, 0.7, 0.2]])
    epsilon = 1e-5
    analytic = rmsnorm_router_sensitivities(
        hidden, norm_weight, router, epsilon=epsilon,
    )

    def logits(value: np.ndarray) -> np.ndarray:
        normalized = value / np.sqrt(np.mean(value * value) + epsilon)
        return router @ (normalized * norm_weight)

    finite = np.empty_like(analytic)
    step = 1e-6
    for column in range(hidden.size):
        offset = np.zeros_like(hidden)
        offset[column] = step
        finite[:, column] = (logits(hidden + offset) - logits(hidden - offset)) / (2 * step)
    np.testing.assert_allclose(analytic, finite, rtol=2e-6, atol=2e-6)
