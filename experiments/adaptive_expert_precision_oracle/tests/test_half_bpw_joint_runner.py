from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import run_half_bpw_joint_field as runner
from oracle_study.split_interaction_field import split_projection_responses


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/qwen36_mxfp4_half_bpw_joint_field.json"


def test_frozen_half_bpw_contract_is_exact():
    config = json.loads(CONFIG.read_text())
    runner._validate_config(config)
    assert config["group_page_budget"] == 8 * 384
    assert config["streamed_mean_pages_per_expert"] / 768 == 0.5
    assert tuple(config["policies"]) == runner.POLICIES
    assert config["base_pr13_commit"] == "dfba3748e51d6916dc1f28cb1d3b3250188adbbe"
    assert runner.base.proxy_gradients is runner.pr13.base.proxy_gradients


def test_contract_rejects_rate_worker_and_policy_drift():
    original = json.loads(CONFIG.read_text())
    for name, value in (
        ("streamed_mean_pages_per_expert", 383),
        ("evaluation_workers", 25),
        ("policies", list(reversed(runner.POLICIES))),
    ):
        changed = dict(original)
        changed[name] = value
        with pytest.raises(ValueError):
            runner._validate_config(changed)


def test_exact_group_damage_matches_manual_weighted_residual():
    rng = np.random.default_rng(9)
    proxy = rng.normal(size=(7, 2))
    beta = 0.25
    weights = np.asarray([0.6, 0.4])
    responses = []
    states = []
    for _ in range(2):
        q2 = (
            rng.normal(size=(4, 5)),
            rng.normal(size=(4, 5)),
            rng.normal(size=(7, 4)),
        )
        q4 = tuple(value + 0.1 * rng.normal(size=value.shape) for value in q2)
        responses.append(split_projection_responses(q2, q4, rng.normal(size=5)))
        states.append(rng.integers(0, 8, size=4, dtype=np.int64))
    observed = runner._exact_group_damage(
        tuple(responses), weights, np.concatenate(states), proxy, beta,
    )
    residual = sum(
        (
            weights[index]
            * (
                response.target_output
                - runner.split_state_output(response, states[index])
            )
            for index, response in enumerate(responses)
        ),
        np.zeros(7),
    )
    expected = float(residual @ residual + beta * ((residual @ proxy) @ (residual @ proxy)))
    np.testing.assert_allclose(observed, expected)


def test_witness_loader_requires_complete_top8_grid(tmp_path: Path):
    config = json.loads(CONFIG.read_text())
    identities = []
    expert_rows = []
    group_rows = []
    for group in range(config["expected_validation_groups"]):
        identity = {
            "capture_source": "exact_checkpoint",
            "evaluation_split": "validation",
            "request_id": f"r{group}",
            "position": group,
            "layer": config["layers"][group % 4],
        }
        identities.append(identity)
        group_rows.append({
            **identity,
            "factor_config_id": config["base_witness_factor_id"],
            "allocation_policy": config["base_witness_policy"],
            "mean_budget_pages_per_expert": 384,
            "burst_cap_pages_per_expert": 1536,
            "group_recovery": 0.97,
        })
        for rank in range(1, 9):
            expert_rows.append({
                **identity,
                "router_rank": rank,
                "factor_config_id": config["base_witness_factor_id"],
                "allocation_policy": config["base_witness_policy"],
                "mean_budget_pages_per_expert": 384,
                "burst_cap_pages_per_expert": 1536,
                "selected_states": json.dumps([rank - 1] * 512),
            })
    pd.DataFrame(expert_rows).to_parquet(tmp_path / runner.pr13.EXPERT_ALLOCATION)
    pd.DataFrame(group_rows).to_parquet(tmp_path / runner.pr13.GROUP_FRONTIER)
    witnesses, recoveries = runner._load_witnesses(config, tmp_path)
    assert len(witnesses) == len(recoveries) == 128
    first = next(iter(witnesses.values()))
    assert first.shape == (4096,)
    assert first[:512].tolist() == [0] * 512
    broken = pd.DataFrame(expert_rows[:-1])
    broken.to_parquet(tmp_path / runner.pr13.EXPERT_ALLOCATION)
    with pytest.raises(RuntimeError):
        runner._load_witnesses(config, tmp_path)


def test_analytical_solver_work_grows_with_rank_and_search():
    small = runner._solver_macs(4, 4096, 2, 1, 8)
    wide = runner._solver_macs(16, 4096, 2, 1, 8)
    searched = runner._solver_macs(4, 4096, 4, 3, 16)
    assert 0 < small < wide
    assert searched > small
