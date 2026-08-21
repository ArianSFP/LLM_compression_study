from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from oracle_study.interaction_field import factor_joint_gram
from oracle_study.neuron_selector import factorized_unit_outputs, unit_score_metadata
from oracle_study.split_interaction_field import (
    build_split_interaction_field, split_projection_responses,
)
from oracle_study.unit_set_teacher import down_metric_gram


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "run_split_interaction_field_study",
    SCRIPTS / "run_split_interaction_field_study.py",
)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _config():
    return json.loads(
        (ROOT / "configs/qwen36_mxfp4_split_interaction_field.json").read_text()
    )


def _tiny(seed=4, units=8, width=5, output=9):
    rng = np.random.default_rng(seed)
    g2, u2, d2 = (
        rng.normal(size=(units, width)),
        rng.normal(size=(units, width)),
        rng.normal(size=(output, units)),
    )
    g4 = g2 + .2 * rng.normal(size=g2.shape)
    u4 = u2 + .2 * rng.normal(size=u2.shape)
    d4 = d2 + .2 * rng.normal(size=d2.shape)
    x = rng.normal(size=width)
    proxy = rng.normal(size=(output, 2))
    beta = .3
    q2, q4 = (g2, u2, d2), (g4, u4, d4)
    abc = unit_score_metadata(d2, d4, proxy=proxy, beta=beta)
    gram = down_metric_gram(d2, d4, proxy=proxy, beta=beta, dtype=np.float64)
    factor = factor_joint_gram(gram, 4, dtype=np.float64)
    return q2, q4, x, proxy, beta, abc, factor


def test_frozen_contract_uses_24_single_thread_workers_and_four_factors():
    config = _config()
    old = json.loads(
        (ROOT / "configs/qwen36_mxfp4_set_utility_distillation.json").read_text()
    )
    runner._validate_contract(config, old)
    assert config["evaluation_workers"] == 24
    assert config["worker_blas_threads"] == 1
    assert len(config["factor_config_ids"]) == 4


def test_rank8_primary_worst_case_stays_below_compute_gate():
    diagnostics = {
        "allowed_state_count": 8, "coordinate_sweeps": 16,
        "local_evaluated_passes": 12, "local_maximum_shortlist": 56,
    }
    total, components = runner._split_compute(rank=8, diagnostics=diagnostics)
    assert total == sum(components.values())
    assert total < 1_572_864
    assert components["local_search_macs"] > components["coordinate_macs"]
    restricted = dict(diagnostics, allowed_state_count=4, local_maximum_shortlist=24)
    four_total, _ = runner._split_compute(rank=8, diagnostics=restricted)
    assert four_total < total


def test_state_summary_reports_split_only_and_separate_pages():
    states = np.asarray([0, 1, 2, 3, 4, 5, 6, 7])
    summary = runner._state_summary(states)
    assert json.loads(summary["state_counts"]) == [1] * 8
    assert summary["split_only_units"] == 4
    assert summary["gate_high_units"] == 4
    assert summary["up_high_units"] == 4
    assert summary["down_high_units"] == 4


def test_old_reference_and_split_solver_smoke(monkeypatch):
    q2, q4, x, _, _, abc, factor = _tiny()
    monkeypatch.setattr(runner, "UNITS", 8)
    config = _config()
    config["coordinate_sweeps"] = 4
    config["local_max_passes"] = 3
    old_outputs = factorized_unit_outputs(q2, q4, x)
    old, _ = runner._old_solution(old_outputs, abc, factor, 12, config)
    responses = split_projection_responses(q2, q4, x)
    field = build_split_interaction_field(factor, responses.hidden, abc)
    primary = runner._split_solver(field, 12, old.states, config, inherited=False)
    restricted = runner._split_solver(
        field, 12, old.states, config, inherited=False,
        allowed_states=tuple(runner.INHERITED_FOUR_STATE_MAP.tolist()),
    )
    warm = runner._split_solver(field, 12, old.states, config, inherited=True)
    assert primary[0].pages <= 12
    assert restricted[0].pages <= 12
    assert set(restricted[0].states.tolist()).issubset(
        set(runner.INHERITED_FOUR_STATE_MAP.tolist())
    )
    assert warm[0].pages <= 12
    assert warm[0].damage <= field.damage(
        np.asarray([0, 6, 1, 7], np.int64)[old.states]
    ) + 1e-10


def test_cpu_capacity_records_quota_and_visible_topology():
    facts = runner._cpu_capacity_facts()
    assert set(facts) == {
        "cpu_quota_raw", "quota_cores", "affinity_logical_cpus", "os_cpu_count",
    }
    assert facts["affinity_logical_cpus"] >= 1
