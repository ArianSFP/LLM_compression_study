from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import analyze_half_bpw_joint_field as analyzer
import run_half_bpw_joint_field as runner


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(tmp_path: Path) -> tuple[Path, Path]:
    config = json.loads(
        (ROOT / "configs/qwen36_mxfp4_half_bpw_joint_field.json").read_text()
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    validation = tmp_path / "validation"
    validation.mkdir()
    rows, states = [], []
    recoveries = {
        runner.POLICIES[0]: 0.970,
        runner.POLICIES[1]: 0.996,
        runner.POLICIES[2]: 0.997,
        runner.POLICIES[3]: 0.998,
    }
    selected = json.dumps([0] * 4096, separators=(",", ":"))
    state_hash = hashlib.sha256(bytes(4096)).hexdigest()
    for index in range(128):
        identity = {
            "capture_source": "exact_checkpoint",
            "evaluation_split": "validation",
            "request_id": f"request-{index // 32}",
            "position": index,
            "layer": [0, 4, 20, 39][index // 32],
        }
        for policy in runner.POLICIES:
            recovery = recoveries[policy]
            damage = 100.0 * (1.0 - recovery)
            metadata = 7184 if policy == runner.POLICIES[1] else 19468
            rows.append({
                **identity,
                "allocation_policy": policy,
                "streamed_bpw": 0.5,
                "allowed_total_bpw": 0.5 + 8 * metadata / 3145728,
                "actual_group_pages": 3000,
                "metadata_bpw": 8 * metadata / 3145728,
                "group_base_qenergy_damage": 100.0,
                "group_exact_qenergy_damage": damage,
                "group_recovery": recovery,
                "recovery_delta_vs_pr13": recovery - 0.970,
                "damage_ratio_vs_pr13": damage / 3.0,
                "selected_states_sha256": state_hash,
                "selector_compute_macs": 1000,
                "coordinate_sweeps": 3,
                "local_passes": 2,
            })
            states.append({
                **identity,
                "allocation_policy": policy,
                "selected_states_sha256": state_hash,
                "selected_states": selected,
            })
    group_path = validation / runner.GROUP_FRONTIER
    state_path = validation / runner.STATE_EVIDENCE
    accounting_path = validation / runner.ACCOUNTING
    pd.DataFrame(rows).to_parquet(group_path, index=False)
    pd.DataFrame(states).to_parquet(state_path, index=False)
    accounting_path.write_text(json.dumps({"schema_version": 1}) + "\n")
    facts = {
        "completed": True,
        "completed_layers": [0, 4, 20, 39],
        "failures": [],
        "failure_history": [],
        "config_sha256": _sha(config_path),
        "runner_sha256": _sha(ROOT / "scripts/run_half_bpw_joint_field.py"),
        "joint_core_sha256": _sha(ROOT / "src/oracle_study/low_rate_joint_field.py"),
        "group_frontier_sha256": _sha(group_path),
        "state_evidence_sha256": _sha(state_path),
        "accounting_sha256": _sha(accounting_path),
    }
    (validation / runner.RUN_FACTS).write_text(json.dumps(facts) + "\n")
    return config_path, validation


def test_analysis_emits_strict_target_decision_and_manifest(tmp_path: Path):
    config, validation = _bundle(tmp_path)
    output = tmp_path / "analysis"
    analyzer.analyze(config, validation, output)
    decision = json.loads((output / analyzer.DECISION).read_text())
    assert decision["status"] == "continue"
    assert decision["primary_pass"] is True
    assert decision["exact_ceiling_pass"] is True
    summary = pd.read_csv(output / analyzer.SUMMARY)
    assert len(summary) == 4
    assert set(summary["groups"]) == {128}
    manifest = json.loads((output / analyzer.MANIFEST).read_text())
    assert manifest["decision_sha256"] == _sha(output / analyzer.DECISION)
    assert set(manifest["generated"]) == {
        analyzer.SUMMARY, analyzer.LAYER, analyzer.DECISION,
        analyzer.REPORT, analyzer.PLOT_PNG, analyzer.PLOT_SVG,
    }


def test_analysis_rejects_exact_oracle_regression(tmp_path: Path):
    config, validation = _bundle(tmp_path)
    frame = pd.read_parquet(validation / runner.GROUP_FRONTIER)
    mask = frame["allocation_policy"] == runner.POLICIES[3]
    frame.loc[mask, "group_exact_qenergy_damage"] = 4.0
    frame.loc[mask, "group_recovery"] = 0.96
    frame.to_parquet(validation / runner.GROUP_FRONTIER, index=False)
    facts_path = validation / runner.RUN_FACTS
    facts = json.loads(facts_path.read_text())
    facts["group_frontier_sha256"] = _sha(validation / runner.GROUP_FRONTIER)
    facts_path.write_text(json.dumps(facts) + "\n")
    with pytest.raises(RuntimeError, match="does not dominate"):
        analyzer.analyze(config, validation, tmp_path / "analysis")
