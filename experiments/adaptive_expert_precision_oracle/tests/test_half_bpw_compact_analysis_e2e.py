from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import analyze_half_bpw_compact_field as analyzer
import run_half_bpw_compact_field as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/qwen36_mxfp4_half_bpw_compact_field.json"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def test_analyzer_attributes_same_factor_dp_pass_to_seed_gap(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text())
    fit = tmp_path / "fit"
    validation = tmp_path / "validation"
    output = tmp_path / "analysis"
    fit.mkdir()
    validation.mkdir()

    dependencies = runner._dependencies()
    fit_facts = {**dependencies, "completed": True, "failures": []}
    manifest = {**dependencies, "completed": True}
    _write_json(fit / runner.FIT_FACTS, fit_facts)
    _write_json(fit / runner.FIT_MANIFEST, manifest)

    payloads = {
        runner.POLICIES[0]: 6148,
        runner.POLICIES[1]: 6148,
        runner.POLICIES[2]: 6148,
        runner.POLICIES[3]: 32768,
        runner.POLICIES[4]: 10244,
        runner.POLICIES[5]: 0,
    }
    recoveries = {
        runner.POLICIES[0]: 0.97,
        runner.POLICIES[1]: 0.98,
        runner.POLICIES[2]: 0.996,
        runner.POLICIES[3]: 0.985,
        runner.POLICIES[4]: 0.988,
        runner.POLICIES[5]: 0.999,
    }
    rows = []
    states = []
    for group in range(128):
        identity = {
            "capture_source": "exact_checkpoint",
            "evaluation_split": "validation",
            "request_id": f"request-{group}",
            "position": group,
            "layer": (0, 4, 20, 39)[group // 32],
        }
        for policy in runner.POLICIES:
            factor_bytes = payloads[policy]
            metadata = factor_bytes + 3084
            if policy == runner.POLICIES[2]:
                seed_name = "global_exact_self_dp"
                evaluations = 8 * 4096 * (3072 + 1)
                backpointers = 4096 * (3072 + 1)
            elif policy in (
                runner.POLICIES[1], runner.POLICIES[3], runner.POLICIES[4],
            ):
                seed_name = "global_self_lagrangian"
                evaluations = 1_638_400
                backpointers = 0
            else:
                seed_name = "frozen_control"
                evaluations = 0
                backpointers = 0
            rows.append({
                **identity,
                "allocation_policy": policy,
                "streamed_bpw": 0.5,
                "actual_group_pages": 3072,
                "average_actual_streamed_bpw": 0.5,
                "factor_payload_bytes_per_expert": factor_bytes,
                "combined_metadata_bytes_per_expert": metadata,
                "metadata_bpw": 8.0 * metadata / 3_145_728.0,
                "allowed_total_bpw": 0.5 + 8.0 * metadata / 3_145_728.0,
                "logical_group_bytes_read": 3072 * 512 + 8 * metadata,
                "group_base_qenergy_damage": 1.0,
                "group_exact_qenergy_damage": 1.0 - recoveries[policy],
                "group_recovery": recoveries[policy],
                "damage_ratio_vs_pr13": 1.0,
                "state_agreement_with_exact_teacher": 0.5,
                "selector_compute_macs": 10_000_000,
                "selector_seed_state_evaluations": evaluations,
                "selector_seed_backpointer_bytes": backpointers,
                "selector_wall_seconds": 0.01,
                "coordinate_sweeps": 2,
                "local_passes": 1,
                "validation_oracle_seed_used": False,
                "seed_name": seed_name,
            })
            states.append({**identity, "allocation_policy": policy})
    pd.DataFrame(rows).to_parquet(validation / runner.GROUP_FRONTIER, index=False)
    pd.DataFrame(states).to_parquet(validation / runner.STATE_EVIDENCE, index=False)
    _write_json(validation / runner.ACCOUNTING, {
        "policies": list(runner.POLICIES),
        "schema_version": 2,
        "primary_group_sidecar_bytes": 73856,
    })
    facts = {
        **dependencies,
        "completed": True,
        "failures": [],
        "failure_history": [],
        "config_sha256": analyzer._sha256(CONFIG),
        "fit_facts_sha256": analyzer._sha256(fit / runner.FIT_FACTS),
        "fit_manifest_sha256": analyzer._sha256(fit / runner.FIT_MANIFEST),
        "group_frontier_sha256": analyzer._sha256(validation / runner.GROUP_FRONTIER),
        "state_evidence_sha256": analyzer._sha256(validation / runner.STATE_EVIDENCE),
        "accounting_sha256": analyzer._sha256(validation / runner.ACCOUNTING),
    }
    _write_json(validation / runner.RUN_FACTS, facts)

    analyzer.analyze(CONFIG, fit, validation, output)
    decision = json.loads((output / analyzer.DECISION).read_text())
    assert decision["status"] == "seed_efficiency_gap"
    assert decision["primary_median_seed_backpointer_bytes"] == 0
    assert decision["dp_control_p10_recovery"] == 0.996
    report = (output / analyzer.REPORT).read_text()
    assert "12,587,008-byte exact-self DP backpointer" in report
    manifest_result = json.loads((output / analyzer.MANIFEST).read_text())
    assert manifest_result["sources"]["efficient_seed_core_sha256"]["path"] == (
        "src/oracle_study/efficient_joint_seed.py"
    )
