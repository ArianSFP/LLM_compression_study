from __future__ import annotations

import numpy as np

from oracle_study.interaction_field import factor_joint_gram
from oracle_study.low_rate_joint_field import (
    combine_split_interaction_fields,
    exact_qmetric_factor,
    optimize_joint_field,
    split_group_states,
)
from oracle_study.neuron_selector import unit_score_metadata
from oracle_study.split_interaction_field import (
    build_split_interaction_field,
    split_exact_damage,
    split_projection_responses,
    split_state_output,
)
from oracle_study.unit_set_teacher import down_metric_gram


def _expert(seed: int, *, units: int = 4, width: int = 6, output: int = 8):
    rng = np.random.default_rng(seed)
    q2 = (
        rng.normal(size=(units, width)),
        rng.normal(size=(units, width)),
        rng.normal(size=(output, units)),
    )
    q4 = tuple(value + 0.2 * rng.normal(size=value.shape) for value in q2)
    x = rng.normal(size=width)
    return q2, q4, x


def test_combined_full_rank_field_equals_exact_weighted_group_damage():
    proxy = np.random.default_rng(21).normal(size=(8, 2))
    beta = 0.3
    weights = np.asarray([0.65, 0.35])
    fields = []
    responses = []
    states = []
    for index, seed in enumerate((3, 7)):
        q2, q4, x = _expert(seed)
        response = split_projection_responses(q2, q4, x)
        factor = exact_qmetric_factor(response.down2, response.down4, proxy, beta)
        abc = unit_score_metadata(q2[2], q4[2], proxy=proxy, beta=beta)
        fields.append(build_split_interaction_field(factor, response.hidden, abc))
        responses.append(response)
        states.append(np.asarray([index, 2, 5, 7], np.int64))
    combined = combine_split_interaction_fields(fields, weights)
    flattened = np.concatenate(states)
    residual = sum(
        (
            weights[index]
            * (
                response.target_output
                - split_state_output(response, states[index])
            )
            for index, response in enumerate(responses)
        ),
        np.zeros(8, np.float64),
    )
    exact = float(residual @ residual)
    projected = residual @ proxy
    exact += beta * float(projected @ projected)
    np.testing.assert_allclose(combined.damage(flattened), exact, rtol=1e-10, atol=1e-10)


def test_combined_aligned_factor_matches_weighted_field_formula():
    proxy = np.random.default_rng(31).normal(size=(8, 2))
    beta = 0.2
    weights = np.asarray([0.7, 0.3])
    fields, states = [], []
    for index, seed in enumerate((11, 13)):
        q2, q4, x = _expert(seed)
        response = split_projection_responses(q2, q4, x)
        gram = down_metric_gram(q2[2], q4[2], proxy=proxy, beta=beta, dtype=np.float64)
        factor = factor_joint_gram(gram, 5, dtype=np.float64)
        abc = unit_score_metadata(q2[2], q4[2], proxy=proxy, beta=beta)
        field = build_split_interaction_field(factor, response.hidden, abc)
        fields.append(field)
        states.append(np.asarray([0, 2 + index, 5, 7], np.int64))
    combined = combine_split_interaction_fields(fields, weights)
    signature = sum(
        (
            weights[index]
            * fields[index].rho[np.arange(fields[index].units), states[index]].sum(axis=0)
            for index in range(2)
        ),
        np.zeros(fields[0].rank),
    )
    rows = np.arange(fields[0].units)
    self_terms = sum(
        weights[index] ** 2
        * fields[index].self_residual[rows, states[index]].sum()
        for index in range(2)
    )
    np.testing.assert_allclose(
        combined.damage(np.concatenate(states)),
        float(signature @ signature + self_terms),
    )


def test_joint_optimizer_is_deterministic_feasible_and_seed_dominant():
    proxy = np.random.default_rng(41).normal(size=(8, 2))
    fields = []
    for seed in (17, 19):
        q2, q4, x = _expert(seed, units=5)
        response = split_projection_responses(q2, q4, x)
        factor = exact_qmetric_factor(response.down2, response.down4, proxy, 0.1)
        abc = unit_score_metadata(q2[2], q4[2], proxy=proxy, beta=0.1)
        fields.append(build_split_interaction_field(factor, response.hidden, abc))
    field = combine_split_interaction_fields(fields, (0.55, 0.45))
    inherited = np.asarray([0, 1, 2, 4, 7, 0, 1, 2, 4, 7], np.int64)
    kwargs = dict(
        named_seeds=(("inherited", inherited),),
        coordinate_sweeps=5,
        local_shortlist=5,
        local_swap_units=3,
        local_max_passes=5,
    )
    first = optimize_joint_field(field, 15, **kwargs)
    second = optimize_joint_field(field, 15, **kwargs)
    np.testing.assert_array_equal(first.states, second.states)
    assert first.pages <= 15
    assert first.damage <= field.damage(inherited) + 1e-12
    assert first.damage == field.damage(first.states)
    assert len(split_group_states(first.states, 2)) == 2


def test_exact_qmetric_single_expert_matches_existing_exact_damage():
    q2, q4, x = _expert(23)
    proxy = np.random.default_rng(29).normal(size=(8, 3))
    beta = 0.4
    response = split_projection_responses(q2, q4, x)
    factor = exact_qmetric_factor(response.down2, response.down4, proxy, beta)
    abc = unit_score_metadata(q2[2], q4[2], proxy=proxy, beta=beta)
    field = build_split_interaction_field(factor, response.hidden, abc)
    states = np.asarray([0, 3, 5, 7], np.int64)
    np.testing.assert_allclose(
        field.damage(states),
        split_exact_damage(response, states, proxy=proxy, beta=beta),
        rtol=1e-10,
        atol=1e-10,
    )
