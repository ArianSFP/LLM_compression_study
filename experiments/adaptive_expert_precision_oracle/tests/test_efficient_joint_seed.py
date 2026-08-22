from __future__ import annotations

import numpy as np

from oracle_study.efficient_joint_seed import lagrangian_self_seed
from oracle_study.interaction_field import JointInteractionFactor
from oracle_study.split_interaction_field import (
    SPLIT_STATE_PAGE_COSTS,
    SplitInteractionField,
)


def _field(local: np.ndarray) -> SplitInteractionField:
    units = int(local.shape[0])
    factor = JointInteractionFactor(
        np.zeros((units, 1)), np.zeros((units, 1)), "zero", 1,
    )
    return SplitInteractionField(
        rho=np.zeros((units, 8, 1)),
        coefficients=np.zeros((units, 8, 2)),
        local_damage=np.asarray(local, np.float64),
        self_residual=np.asarray(local, np.float64),
        factor=factor,
    )


def test_lagrangian_seed_is_deterministic_feasible_and_low_memory() -> None:
    rng = np.random.default_rng(8)
    local = rng.uniform(0.1, 2.0, size=(64, 8))
    local[:, 7] = 0.0
    field = _field(local)
    first = lagrangian_self_seed(field, 48, price_iterations=32)
    second = lagrangian_self_seed(field, 48, price_iterations=32)
    assert np.array_equal(first.states, second.states)
    assert first.pages == int(SPLIT_STATE_PAGE_COSTS[first.states].sum())
    assert first.pages <= 48
    assert first.state_evaluations <= (32 + 66) * 64 * 8
    assert first.states.nbytes == 64 * 8


def test_lagrangian_seed_spends_pages_on_highest_self_gain() -> None:
    local = np.full((3, 8), 10.0)
    local[:, 0] = 5.0
    local[0, 1] = 0.0
    local[1, 1] = 1.0
    local[2, 1] = 2.0
    trace = lagrangian_self_seed(_field(local), 2)
    assert trace.pages == 2
    assert trace.states.tolist() == [1, 1, 0]
