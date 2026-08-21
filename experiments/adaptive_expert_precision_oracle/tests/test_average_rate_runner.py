from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from oracle_study.average_rate_allocator import AllocationTrace, RateOption


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_average_rate_allocation",
    ROOT / "scripts/run_average_rate_allocation.py",
)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


def _config():
    return json.loads(
        (ROOT / "configs/qwen36_mxfp4_average_rate_allocation.json").read_text()
    )


def test_frozen_contract_accepts_only_declared_grid():
    config = _config()
    runner._validate_contract(config)
    changed = dict(config)
    changed["mean_correction_page_budgets"] = [384, 576, 750, 768]
    with pytest.raises(RuntimeError, match="mean rate grid"):
        runner._validate_contract(changed)


def test_group_plan_uses_complete_top8_validation_rows_only():
    config = _config()
    layers = np.repeat(np.asarray(config["layers"], np.int64), 32)
    rows = len(layers)
    data = {
        "layer": layers,
        "split": np.asarray(["validation"] * rows),
        "request_id": np.asarray([f"r{index // 4}" for index in range(rows)]),
        "sequence_id": np.asarray([f"s{index // 4}" for index in range(rows)]),
        "position": np.arange(rows),
        "expert_ids": np.tile(np.arange(8), (rows, 1)),
        "router_weights": np.tile(np.ones(8) / 8, (rows, 1)),
    }
    plan = runner._plan(data, config)
    assert len(plan) == 128
    assert len({tuple(row[name] for name in runner.GROUP_IDENTITY) for row in plan}) == 128
    assert all(len(row["experts"]) == 8 for row in plan)
    data["expert_ids"][0, 1] = 0
    with pytest.raises(RuntimeError, match="eight unique"):
        runner._plan(data, config)


def _option(pages: int, damage: float, state: int) -> RateOption:
    return RateOption(
        pages, damage, np.asarray([state], np.int64), "synthetic", 0.0, 1, 1,
    )


def test_group_row_uses_exact_overall_bpw_and_router_weighted_energy():
    config = _config()
    frontiers = tuple((
        _option(0, 4.0, 0), _option(1, 1.0, 7),
    ) for _ in range(8))
    features = tuple(np.asarray([[2.0, 0.0], [1.0, 0.0]]) for _ in range(8))
    exact = tuple(np.asarray([4.0, 1.0]) for _ in range(8))
    base = tuple(value[0] for value in features)
    weights = np.ones(8) / 8
    allocation = AllocationTrace(np.ones(8, np.int64), 8, 8.0)
    metadata = {
        "capture_source": "exact_checkpoint",
        "evaluation_split": "validation",
        "request_id": "r",
        "sequence_id": "s",
        "position": 3,
        "layer": 4,
        "experts": list(range(8)),
    }
    metadata_bytes = 9232
    group, expert_rows = runner._allocation_rows(
        metadata,
        config["primary_factor_id"],
        frontiers,
        features,
        exact,
        base,
        weights,
        384,
        768,
        "pooled_router_square_compressed",
        allocation,
        1.0,
        .2,
        .01,
        123,
        456,
        11,
        7,
        metadata_bytes,
        config,
    )
    assert len(expert_rows) == 8
    assert group["actual_group_pages"] == 8
    assert group["average_actual_correction_bpw"] == pytest.approx(8 / (8 * 768))
    assert group["average_allowed_total_bpw"] == pytest.approx(
        8 * (metadata_bytes + 384 * 512) / runner.EXPERT_WEIGHTS
    )
    assert group["group_recovery"] == pytest.approx(.75)
    assert group["router_square_additive_recovery"] == pytest.approx(.75)
    assert group["selector_dp_state_evaluations"] == 456
    assert group["selector_coordinate_sweeps"] == 11
    assert group["selector_local_passes"] == 7
    assert group["promotable"] is False
    assert group["continuation_eligible"] is True


def test_frontier_compute_accounting_uses_actual_work():
    config = _config()

    class Trace:
        coordinate_sweeps = 11
        local_passes = 7

    rank = 8
    observed = runner._frontier_compute_macs(rank, Trace(), config)
    shortlist = 7 * config["local_shortlist"]
    expected = (
        16 * 512 * rank + 24 * 512
        + (2 * 512 * rank + 16 * 512) * 11
        + 7 * (14 * 512 * rank + shortlist * shortlist * rank)
    )
    assert observed == expected


def test_uniform_allocation_respects_each_expert_cap():
    frontiers = tuple((
        _option(0, 10.0, 0), _option(3, 4.0, 1), _option(6, 1.0, 7),
    ) for _ in range(8))
    trace = runner._uniform_allocation(frontiers, 4)
    np.testing.assert_array_equal(trace.option_indices, np.ones(8, np.int64))
    assert trace.pages == 24
