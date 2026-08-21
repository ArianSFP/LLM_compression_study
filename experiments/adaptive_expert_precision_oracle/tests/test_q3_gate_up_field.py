import numpy as np

from oracle_study.interaction_field import JointInteractionFactor
from oracle_study.neuron_selector import UnitScoreMetadata
from oracle_study.q3_gate_up_field import (
    Q3_IDEAL_COST_QUANTA, Q3_INHERITED_EIGHT_STATE_MAP,
    Q3_STATE_DOWN_HIGH, Q3_STATE_GATE_LEVEL, Q3_STATE_UP_LEVEL,
    Q3PageLayout, _PageTracker, build_q3_interaction_field, fit_coselection_layouts,
    fixed_gate_up_layout, plane_action_id, q3_coordinate_descent,
    monolithic_q2q4_layout, q3_exact_damage, q3_local_search,
    q3_projection_responses,
)


def _case(seed=4, units=8, width=12, rank=4):
    rng = np.random.default_rng(seed)
    q2 = tuple(rng.normal(size=(units, width)) for _ in range(2)) + (rng.normal(size=(width, units)),)
    q3 = tuple(q2[index] + .08 * rng.normal(size=q2[index].shape) for index in range(3))
    q4 = tuple(q3[index] + .08 * rng.normal(size=q3[index].shape) for index in range(3))
    x = rng.normal(size=width)
    responses = q3_projection_responses(q2, q3, q4, x)
    joint = np.concatenate((q4[2], q2[2]), axis=1)
    _, values, vectors = np.linalg.svd(joint, full_matrices=False)
    rows = vectors[:rank].T * values[:rank]
    factor = JointInteractionFactor(rows[:units], rows[units:], "test", rank, 0, "fp64")
    d4, d2 = q4[2], q2[2]
    metadata = UnitScoreMetadata(
        np.sum(d4 * d4, axis=0), np.sum(d4 * d2, axis=0), np.sum(d2 * d2, axis=0)
    )
    return responses, build_q3_interaction_field(factor, responses.hidden, metadata)


def test_state_contract_and_exact_self_terms():
    assert len(Q3_STATE_GATE_LEVEL) == 18
    assert len({(int(g), int(u), bool(d)) for g, u, d in zip(
        Q3_STATE_GATE_LEVEL, Q3_STATE_UP_LEVEL, Q3_STATE_DOWN_HIGH
    )}) == 18
    assert Q3_IDEAL_COST_QUANTA.tolist() == [
        g + u + 2 * int(d) for g, u, d in zip(
            Q3_STATE_GATE_LEVEL, Q3_STATE_UP_LEVEL, Q3_STATE_DOWN_HIGH
        )
    ]
    responses, field = _case(rank=8)
    for unit in range(field.units):
        for state in range(18):
            states = np.full(field.units, 17, np.int64)
            states[unit] = state
            assert np.isclose(field.damage(states), q3_exact_damage(responses, states), rtol=1e-9, atol=1e-8)


def test_inherited_eight_states_embed_old_action_space():
    triples = [
        (int(Q3_STATE_GATE_LEVEL[state]), int(Q3_STATE_UP_LEVEL[state]), bool(Q3_STATE_DOWN_HIGH[state]))
        for state in Q3_INHERITED_EIGHT_STATE_MAP
    ]
    assert triples == [(0, 0, False), (0, 0, True), (0, 2, False), (0, 2, True),
                       (2, 0, False), (2, 0, True), (2, 2, False), (2, 2, True)]


def test_fixed_layout_and_ideal_cost_are_exact():
    layout = fixed_gate_up_layout(8)
    states = np.asarray([17, 16, 12, 6, 2, 1, 0, 8])
    assert layout.cost_quanta(states) == 2 * len(layout.page_union(states, 0))
    ideal = Q3PageLayout("ideal", 8, None)
    assert ideal.cost_quanta(states) == int(Q3_IDEAL_COST_QUANTA[states].sum())
    gate = plane_action_id("gate", 1, 3, 8)
    up = plane_action_id("up", 1, 3, 8)
    assert layout.action_pages[0, gate] == layout.action_pages[0, up]
    old = monolithic_q2q4_layout(8)
    assert old.action_pages[0, plane_action_id("gate", 1, 3, 8)] == old.action_pages[0, plane_action_id("gate", 2, 3, 8)]
    assert old.action_pages[0, plane_action_id("up", 1, 3, 8)] == old.action_pages[0, plane_action_id("up", 2, 3, 8)]


def test_training_layouts_are_deterministic_diverse_and_valid():
    rng = np.random.default_rng(12)
    incidence = (rng.random((30, 32)) < .25).astype(np.uint8)
    first = fit_coselection_layouts(incidence, units=8, replicas=2)
    second = fit_coselection_layouts(incidence, units=8, replicas=2)
    assert np.array_equal(first, second)
    assert not np.array_equal(first[0], first[1])
    layout = Q3PageLayout("learned2", 8, first, learned=True)
    states = rng.integers(0, 18, size=8)
    assert layout.cost_quanta(states) == 2 * min(
        len(layout.page_union(states, replica)) for replica in range(2)
    )


def test_coordinate_and_local_search_respect_unique_page_budget():
    _, field = _case()
    incidence = np.eye(4 * field.units, dtype=np.uint8)
    pages = fit_coselection_layouts(incidence, units=field.units, replicas=2)
    layout = Q3PageLayout("learned2", field.units, pages, learned=True)
    seed = np.zeros(field.units, np.int64)
    coordinate = q3_coordinate_descent(field, layout, seed, 24, max_sweeps=8)
    repaired = q3_local_search(field, layout, coordinate.states, 24, max_passes=3)
    assert coordinate.cost_quanta <= 24
    assert repaired.cost_quanta <= 24
    assert repaired.damage <= coordinate.damage + 1e-9
    assert repaired.cost_quanta == layout.cost_quanta(repaired.states)


def test_replicated_layout_never_costs_more_than_first_layout():
    rng = np.random.default_rng(21)
    incidence = (rng.random((40, 32)) < .3).astype(np.uint8)
    pages = fit_coselection_layouts(incidence, units=8, replicas=2)
    one = Q3PageLayout("one", 8, pages[:1], learned=True)
    two = Q3PageLayout("two", 8, pages, learned=True)
    for _ in range(50):
        states = rng.integers(0, 18, size=8)
        assert two.cost_quanta(states) <= one.cost_quanta(states)


def test_vectorized_single_unit_costs_match_full_page_union():
    rng = np.random.default_rng(22)
    incidence = (rng.random((40, 32)) < .3).astype(np.uint8)
    pages = fit_coselection_layouts(incidence, units=8, replicas=2)
    layouts = (
        Q3PageLayout("ideal", 8, None),
        fixed_gate_up_layout(8),
        monolithic_q2q4_layout(8),
        Q3PageLayout("learned2", 8, pages, learned=True),
    )
    for layout in layouts:
        states = rng.integers(0, 18, size=8)
        tracker = _PageTracker(layout, states)
        for unit in range(8):
            expected = []
            for destination in range(18):
                candidate = states.copy()
                candidate[unit] = destination
                expected.append(layout.cost_quanta(candidate))
