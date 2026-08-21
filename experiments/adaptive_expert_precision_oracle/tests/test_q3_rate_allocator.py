import numpy as np

from oracle_study.average_rate_allocator import multiple_choice_allocate
from oracle_study.interaction_field import JointInteractionFactor
from oracle_study.neuron_selector import UnitScoreMetadata
from oracle_study.q3_gate_up_field import (
    Q3_INHERITED_EIGHT_STATE_MAP, Q3PageLayout, build_q3_interaction_field,
    fixed_gate_up_layout, monolithic_q2q4_layout,
)
from oracle_study.q3_rate_allocator import (
    q3_allocation_aware_rate_frontiers, q3_rate_frontier,
    training_plane_incidence,
)


def _field(seed, units=8, rank=4):
    rng = np.random.default_rng(seed)
    l4 = rng.normal(size=(units, rank)) * .2
    l2 = rng.normal(size=(units, rank)) * .2
    factor = JointInteractionFactor(l4, l2, "test", rank, 0, "fp64")
    hidden = rng.normal(size=(units, 9))
    hidden[:, 8] = rng.normal(size=units)
    a = rng.uniform(.5, 2, units)
    c = rng.uniform(.5, 2, units)
    b = np.clip(rng.normal(scale=.2, size=units), -np.sqrt(a*c), np.sqrt(a*c))
    return build_q3_interaction_field(factor, hidden, UnitScoreMetadata(a, b, c))


def test_frontier_is_pareto_and_uses_exact_layout_cost():
    field = _field(1)
    layout = fixed_gate_up_layout(field.units)
    trace = q3_rate_frontier(
        field, layout, maximum_quanta=48, target_quanta=[12, 24, 36],
        price_ratios=[16, 4, 1, .25, 0], coordinate_sweeps=4,
        local_shortlist=4, local_swap_units=2, local_max_passes=2,
    )
    costs = [int(item.pages) for item in trace.options]
    damage = [float(item.damage) for item in trace.options]
    assert costs == sorted(set(costs))
    assert all(damage[index] > damage[index + 1] for index in range(len(damage)-1))
    assert all(layout.cost_quanta(item.states) == item.pages for item in trace.options)


def test_selected_column_repair_preserves_budget():
    fields = [_field(index + 2) for index in range(3)]
    layouts = [Q3PageLayout("ideal", field.units, None) for field in fields]
    frontiers = [q3_rate_frontier(
        field, layout, maximum_quanta=48, target_quanta=[16, 32],
        price_ratios=[8, 2, .5, 0], coordinate_sweeps=3,
        local_shortlist=3, local_swap_units=2, local_max_passes=1,
    ).options for field, layout in zip(fields, layouts)]
    coarse = multiple_choice_allocate(frontiers, 70, np.ones(3))
    refined = q3_allocation_aware_rate_frontiers(
        fields, layouts, frontiers, group_budget_quanta=70,
        burst_cap_quanta=48, weights=np.ones(3), coordinate_sweeps=3,
        local_shortlist=3, local_swap_units=2, local_max_passes=2,
        max_rounds=2,
    )
    assert refined.allocation.pages <= 70
    assert refined.allocation.objective <= coarse.objective + 1e-9


def test_training_incidence_is_binary_and_has_four_planes_per_unit():
    fields = [_field(9), _field(10)]
    incidence = training_plane_incidence(fields, price_ratios=[8, 1, 0])
    assert incidence.shape == (6, 4 * fields[0].units)
    assert set(np.unique(incidence)) <= {0, 1}


def test_same_solver_reference_never_admits_q3_states():
    field = _field(31)
    trace = q3_rate_frontier(
        field, monolithic_q2q4_layout(field.units), maximum_quanta=48,
        target_quanta=[24], price_ratios=[8, 2, .5, 0],
        coordinate_sweeps=4, local_shortlist=4, local_swap_units=2,
        local_max_passes=2, allowed_states=Q3_INHERITED_EIGHT_STATE_MAP,
    )
    allowed = set(map(int, Q3_INHERITED_EIGHT_STATE_MAP))
    assert all(set(map(int, option.states)) <= allowed for option in trace.options)
