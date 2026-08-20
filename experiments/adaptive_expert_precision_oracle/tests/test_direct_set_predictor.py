from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from oracle_study.direct_set_predictor import (
    CompactCoherentUnitPredictor,
    DirectSetTrainingConfig,
    DirectSetTrainingData,
    PredictorOutput,
    boundary_pair_ranking_loss,
    candidate_coverage_loss,
    direct_set_distillation_loss,
    evaluate_candidate_reranking,
    exact_gram_set_damage,
    hard_fixed_cardinality_gram_set_loss,
    listwise_marginal_kl_loss,
    nested_candidate_apply_masks,
    request_balanced_sample_weights,
    train_direct_set_predictor,
)


def test_exact_gram_set_loss_matches_explicit_corrections_and_hard_cardinality() -> None:
    generator = torch.Generator().manual_seed(20260820)
    batch, units, output = 3, 7, 11
    corrections = torch.randn(batch, units, output, generator=generator, dtype=torch.float64)
    gram = corrections @ corrections.transpose(-1, -2)
    logits = torch.randn(batch, units, generator=generator, dtype=torch.float64, requires_grad=True)

    result = hard_fixed_cardinality_gram_set_loss(
        logits, gram, 3, temperature=0.8,
    )
    torch.testing.assert_close(
        result.mask.detach().sum(dim=-1), torch.full((batch,), 3.0, dtype=torch.float64),
        atol=0, rtol=0,
    )
    assert torch.all((result.mask.detach() == 0) | (result.mask.detach() == 1))
    residual = ((1.0 - result.mask).unsqueeze(-1) * corrections).sum(dim=1)
    expected = torch.square(residual).sum(dim=-1)
    torch.testing.assert_close(result.per_sample_damage, expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(
        exact_gram_set_damage(gram, result.mask, reduction="none"), expected,
        atol=1e-10, rtol=1e-10,
    )
    result.loss.backward()
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))
    assert torch.any(torch.abs(logits.grad) > 1e-10)


def test_joint_masks_are_exactly_nested_even_when_heads_disagree() -> None:
    output = PredictorOutput(
        candidate_logits=torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0, 4.0]]),
        apply_logits=torch.tensor([[0.0, 1.0, 2.0, 3.0, 20.0, 19.0]]),
        head_mode="joint_nested",
    )
    masks = nested_candidate_apply_masks(output, candidate_count=4, apply_count=2)
    assert masks.candidate.sum().item() == 4
    assert masks.apply.sum().item() == 2
    assert torch.all(~masks.apply | masks.candidate)
    np.testing.assert_array_equal(torch.nonzero(masks.apply[0], as_tuple=False).flatten().numpy(), [4, 5])
    np.testing.assert_array_equal(torch.nonzero(masks.candidate[0], as_tuple=False).flatten().numpy(), [0, 1, 4, 5])


def test_candidate_coverage_prefers_teacher_apply_support_deterministically() -> None:
    order = torch.tensor([[0, 1, 2, 3, 4, 5]])
    good = torch.tensor([[9.0, 8.0, 7.0, 0.0, -1.0, -2.0]], requires_grad=True)
    bad = torch.tensor([[0.0, -1.0, 9.0, 8.0, 7.0, -2.0]], requires_grad=True)
    first = candidate_coverage_loss(
        good, order, candidate_count=3, teacher_apply_count=2,
    )
    second = candidate_coverage_loss(
        good, order, candidate_count=3, teacher_apply_count=2,
    )
    missed = candidate_coverage_loss(
        bad, order, candidate_count=3, teacher_apply_count=2,
    )
    assert float(first.loss.detach()) == pytest.approx(0.0, abs=0.0)
    assert float(missed.loss.detach()) == pytest.approx(1.0, abs=0.0)
    assert first.candidate_mask.detach().sum().item() == 3
    torch.testing.assert_close(first.candidate_mask, second.candidate_mask, atol=0, rtol=0)
    missed.loss.backward()
    assert bad.grad is not None and torch.all(torch.isfinite(bad.grad))

def test_joint_objective_uses_physical_nested_union_for_coverage() -> None:
    order = torch.tensor([[0, 1, 2, 3, 4, 5]])
    output = PredictorOutput(
        candidate_logits=torch.tensor([[0.0, -1.0, 9.0, 8.0, 7.0, -2.0]]),
        apply_logits=torch.tensor([[9.0, 8.0, 0.0, -1.0, -2.0, -3.0]]),
        head_mode="joint_nested",
    )
    teacher_marginal = torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0, 4.0]])
    gram = torch.diag_embed(torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0, 4.0]]))
    config = DirectSetTrainingConfig(
        apply_count=2,
        candidate_count=3,
        boundary_width=1,
        marginal_weight=0.0,
        listwise_weight=0.0,
        boundary_weight=0.0,
        set_weight=1.0,
        candidate_set_weight=1_000.0,
        candidate_coverage_weight=2.0,
    )
    loss = direct_set_distillation_loss(
        output, teacher_marginal, order, gram, config,
    )
    torch.testing.assert_close(
        loss.total, loss.apply_set + 2.0 * loss.candidate_coverage,
    )
    assert float(loss.candidate_coverage) == pytest.approx(0.0)
    assert DirectSetTrainingConfig().candidate_coverage_weight == 0.0

def test_joint_coverage_catches_standalone_topk_false_pass_and_reaches_both_heads() -> None:
    order = torch.tensor([[0, 1, 2, 3, 4, 5]])
    candidate = torch.tensor(
        [[9.0, 8.0, 7.0, 0.0, -1.0, -2.0]], requires_grad=True,
    )
    apply = torch.tensor(
        [[0.0, -1.0, -2.0, -3.0, 10.0, 9.0]], requires_grad=True,
    )
    standalone = candidate_coverage_loss(
        candidate, order, candidate_count=3, teacher_apply_count=2,
    )
    nested = candidate_coverage_loss(
        candidate, order, apply_logits=apply,
        candidate_count=3, teacher_apply_count=2,
    )
    assert float(standalone.loss.detach()) == pytest.approx(0.0, abs=0.0)
    assert float(nested.loss.detach()) == pytest.approx(0.5, abs=0.0)
    np.testing.assert_array_equal(
        torch.nonzero(
            nested.candidate_mask.detach()[0], as_tuple=False,
        ).flatten().numpy(),
        [0, 4, 5],
    )
    nested.loss.backward()
    assert candidate.grad is not None and torch.any(torch.abs(candidate.grad) > 0)
    assert apply.grad is not None and torch.any(torch.abs(apply.grad) > 0)




def test_predictor_consumes_only_configured_deployable_features_and_accounts_exactly() -> None:
    model = CompactCoherentUnitPredictor(
        3, global_dim=4, pq_delta_dim=2, abc_dim=3,
        hidden_dim=8, unit_count=512, head_mode="joint_nested",
    )
    output = model(
        torch.zeros(2, 512, 3),
        global_features=torch.zeros(2, 4),
        pq_delta_response=torch.zeros(2, 512, 2),
        # Static per-expert metadata broadcasts across invocations.
        abc=torch.zeros(512, 3),
    )
    assert output.candidate_logits.shape == (2, 512)
    assert output.apply_logits.shape == (2, 512)
    accounting = model.accounting(
        parameter_bits=16, abc_bits=16, external_feature_metadata_bytes=111,
    )
    assert accounting.trainable_parameters == 266
    assert accounting.parameter_metadata_bytes == 532
    assert accounting.static_abc_metadata_bytes == 3_072
    assert accounting.external_feature_metadata_bytes == 111
    assert accounting.total_metadata_bytes == 3_715
    assert accounting.linear_macs_per_invocation == 73_824
    assert accounting.context_reduction_additions == 8 * 511
    with pytest.raises(TypeError, match="teacher_gram"):
        # There is intentionally no teacher Gram/correction argument in the
        # deployable forward signature.
        model(torch.zeros(2, 512, 3), teacher_gram=torch.eye(512))  # type: ignore[call-arg]


def test_listwise_and_boundary_losses_preserve_teacher_order() -> None:
    teacher = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0]], dtype=torch.float64)
    exact = listwise_marginal_kl_loss(teacher, teacher)
    reversed_loss = listwise_marginal_kl_loss(torch.flip(teacher, dims=(1,)), teacher)
    assert float(exact) == pytest.approx(0.0, abs=1e-14)
    assert float(reversed_loss) > 1.0
    order = torch.tensor([[0, 1, 2, 3, 4]])
    good = boundary_pair_ranking_loss(teacher, order, boundaries=(2,), boundary_width=2)
    bad = boundary_pair_ranking_loss(-teacher, order, boundaries=(2,), boundary_width=2)
    assert float(good) < float(bad)


def test_request_balancing_and_split_leakage_guard() -> None:
    weights = request_balanced_sample_weights(("a", "a", "a", "b"))
    assert float(weights[:3].sum()) == pytest.approx(float(weights[3]))
    q2 = torch.zeros(2, 4, 1)
    weighted = request_balanced_sample_weights(
        ("a", "a", "a", "b"), torch.tensor([9.0, 1.0, 2.0, 7.0]),
    )
    assert float(weighted[:3].sum()) == pytest.approx(float(weighted[3]))
    marginal = torch.zeros(2, 4)
    gram = torch.eye(4).expand(2, -1, -1).clone()
    with pytest.raises(ValueError, match="appears in both"):
        DirectSetTrainingData(
            q2, marginal, gram,
            request_ids=("same", "same"),
            split_labels=("train", "validation"),
        )


def _tiny_training_fixture() -> DirectSetTrainingData:
    generator = torch.Generator().manual_seed(17)
    samples, units = 10, 6
    features = torch.randn(samples, units, 2, generator=generator)
    teacher = 2.5 * features[..., 0] - 0.6 * features[..., 1]
    positive = torch.nn.functional.softplus(teacher) + 0.2
    gram = torch.diag_embed(positive)
    order = torch.argsort(positive, dim=-1, descending=True, stable=True)
    regret = torch.zeros_like(teacher)
    regret.scatter_(1, order[:, :2], positive.gather(1, order[:, :2]))
    return DirectSetTrainingData(
        q2_unit_features=features,
        teacher_marginal=teacher,
        teacher_gram=gram,
        teacher_order=order,
        exclusion_regret=regret,
        request_ids=("tr0", "tr0", "tr1", "tr1", "tr2", "tr2", "va0", "va0", "va1", "va1"),
        split_labels=("train",) * 6 + ("validation",) * 4,
    )


def test_tiny_training_is_deterministic_improves_and_never_uses_validation_rows() -> None:
    data = _tiny_training_fixture()
    torch.manual_seed(99)
    first = CompactCoherentUnitPredictor(
        2, hidden_dim=7, unit_count=6, head_mode="joint_nested",
    )
    second = CompactCoherentUnitPredictor(
        2, hidden_dim=7, unit_count=6, head_mode="joint_nested",
    )
    second.load_state_dict(copy.deepcopy(first.state_dict()))
    config = DirectSetTrainingConfig(
        epochs=35,
        learning_rate=0.025,
        apply_count=2,
        candidate_count=3,
        boundary_width=1,
        marginal_weight=0.3,
        listwise_weight=1.0,
        boundary_weight=0.5,
        set_weight=0.3,
        candidate_set_weight=0.0,
        candidate_coverage_weight=0.5,
        seed=1234,
    )
    first_result = train_direct_set_predictor(first, data, config)
    second_result = train_direct_set_predictor(second, data, config)
    assert first_result.train_indices == (0, 1, 2, 3, 4, 5)
    assert first_result.validation_indices == (6, 7, 8, 9)
    assert first_result.final_train_loss < first_result.initial_train_loss * 0.7
    assert [row.train_loss for row in first_result.history] == pytest.approx(
        [row.train_loss for row in second_result.history], abs=0, rel=0,
    )
    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_candidate_evaluation_keeps_contained_and_full_target_rerank_distinct() -> None:
    # Candidate corrections are 1 and 2 while an omitted correction is -10.
    # The contained target (sum=3) selects unit 1; the full target (sum=-7)
    # selects unit 0 from the same fetched set.
    correction = np.array([1.0, 2.0, -10.0])
    gram = np.outer(correction, correction)
    output = PredictorOutput(
        candidate_logits=torch.tensor([[3.0, 2.0, 0.0]]),
        apply_logits=torch.tensor([[3.0, 2.0, 0.0]]),
        head_mode="single",
    )
    result = evaluate_candidate_reranking(
        output, gram, candidate_count=2, apply_count=1,
    )[0]
    np.testing.assert_array_equal(result.predicted_candidate_ids, [0, 1])
    np.testing.assert_array_equal(result.contained_target_rerank_ids, [1])
    np.testing.assert_array_equal(result.full_target_restricted_ids, [0])
    np.testing.assert_array_equal(result.teacher_ids, [2])
    assert result.contained_target_rerank.full_target_gain != (
        result.full_target_restricted.full_target_gain
    )
    assert result.candidate_teacher_support_recall == 0.0
