from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from oracle_study.split_interaction_field_analysis import (
    promotion_payload, sha256, summary_table, validate_evidence_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_split_interaction_field_study.py"
CORE = ROOT / "src/oracle_study/split_interaction_field.py"
CONFIG = ROOT / "configs/qwen36_mxfp4_split_interaction_field.json"


def _states(budget: int) -> tuple[str, str]:
    states = np.zeros(512, np.int64)
    states[: budget // 3] = 7
    counts = np.bincount(states, minlength=8)
    return (
        json.dumps(states.tolist(), separators=(",", ":")),
        json.dumps(counts.tolist(), separators=(",", ":")),
    )


def _bundle(tmp_path: Path):
    config = json.loads(CONFIG.read_text())
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True))
    validation = tmp_path / "validation"
    validation.mkdir()
    rows = []
    factors = config["factor_config_ids"]
    primary_factors = set(config["primary_factor_config_ids"])
    payloads = {
        "exact_proxy_plus_eigh_tail_tail0_exact4_fp32": 16_384,
        "joint_eigh_tail8_exact0_fp32": 32_768,
        "joint_eigh_tail8_exact0_int8_per_row": 10_244,
        "joint_eigh_tail8_exact0_int4_per_row_hadamard": 6_148,
    }

    def ordinary_row(identity, factor, solver, budget, regime, recovery, same, old, macs):
        factor_bytes = payloads[factor]
        metadata_bytes = factor_bytes + 3_084
        state_text, count_text = _states(budget)
        allowed = 4 if solver in {
            config["pr11_four_state_solver_id"], config["four_state_reference_solver_id"],
        } else 8
        fixed = regime == "fixed_correction_budget"
        return {
            **identity, "budget_regime": regime,
            "physical_budget_pages": budget, "physical_pages": 3 * (budget // 3),
            "physical_budget_bpw": budget / 768,
            "strict_all_in_total_bytes": np.nan if fixed else 393_216,
            "budget_total_bytes": metadata_bytes + budget * 512,
            "budget_total_bpw": 8 * (metadata_bytes + budget * 512) / 3_145_728,
            "strict_all_in_budget_pass": np.nan if fixed else True,
            "strict_all_in_slack_bytes": np.nan if fixed else 393_216 - metadata_bytes - budget * 512,
            "factor_config_id": factor, "factor_family": "joint_eigh", "total_rank": 8,
            "factor_payload_bytes": factor_bytes, "combined_metadata_bytes": metadata_bytes,
            "combined_metadata_bpw": 8 * metadata_bytes / 3_145_728,
            "solver": solver, "selected_seed": "synthetic", "recovery": recovery,
            "four_state_reference_recovery": same, "pr11_four_state_recovery": old,
            "recovery_gain_vs_four_state_reference": recovery - same,
            "recovery_gain_vs_pr11_four_state": recovery - old,
            "seed_optimizer_gain_vs_pr11_four_state": same - old,
            "four_state_reference_kind": "compressed_four_state_exact_self_dp_same_solver",
            "pr10_exact_hybrid_recovery": .95 if fixed else np.nan,
            "set_gain_retention_vs_pr10_exact_hybrid": recovery / .95 if fixed else np.nan,
            "compressed_predicted_damage": 1 - recovery, "exact_qenergy_damage": 1 - recovery,
            "damage_prediction_error_over_base": 0.0, "selector_runtime_ms": 2.0,
            "field_build_runtime_ms": 0.2,
            "selector_compute_macs": macs, "selector_compute_gate_pass": macs < 1_572_864,
            "selector_compute_accounting_bound": "complete",
            "coordinate_sweeps": 8 if solver != config["pr11_four_state_solver_id"] else 0,
            "selected_seed_coordinate_sweeps": 4 if solver != config["pr11_four_state_solver_id"] else 0,
            "coordinate_accepted_moves": 12 if solver != config["pr11_four_state_solver_id"] else 0,
            "local_evaluated_passes": 4 if solver != config["pr11_four_state_solver_id"] else 0,
            "local_accepted_bundles": 2 if solver != config["pr11_four_state_solver_id"] else 0,
            "local_candidate_moves_per_pass": (allowed - 1) * 512,
            "local_maximum_shortlist": (allowed - 1) * 8,
            "allowed_state_count": allowed, "seed_count": 2,
            "uses_exact_mixed_h4": True, "exact_gram_oracle": False,
            "promotable": fixed and solver == config["primary_solver_id"] and factor in primary_factors,
            "selection_regime": "h0_exact_mixed_h4_split_action_geometry_ceiling",
            "selector_non_mac_operations": "control", "selected_states": state_text,
            "state_counts": count_text, "split_only_units": 0,
            "gate_high_units": budget // 3, "up_high_units": budget // 3,
            "down_high_units": budget // 3, "field_build_macs": 1,
            "coordinate_macs": 1, "local_search_macs": 1,
            "inherited_pr11_selector_macs": 0,
        }

    for invocation in range(69):
        identity = {
            "capture_source": "exact_checkpoint", "evaluation_split": "validation",
            "request_id": f"req-{invocation // 23}", "position": invocation,
            "layer": [0, 4, 20, 39][invocation % 4], "expert_id": invocation % 3,
        }
        old = .94 + invocation * 1e-5
        same = old + .001
        for budget in config["page_budgets"]:
            for factor in factors:
                for solver, gain, macs in (
                    (config["pr11_four_state_solver_id"], old - same, 800_000),
                    (config["four_state_reference_solver_id"], 0.0, 700_000),
                    (config["primary_solver_id"], .002, 1_200_000),
                    (config["warm_start_solver_id"], .003, 1_400_000),
                ):
                    rows.append(ordinary_row(
                        identity, factor, solver, budget, "fixed_correction_budget",
                        same + gain, same, old, macs,
                    ))
            state_text, count_text = _states(budget)
            exact_four = .91 + invocation * 1e-5
            for solver, gain, allowed in (
                (config["exact_four_state_solver_id"], 0.0, 4),
                (config["exact_eight_state_solver_id"], .004, 8),
            ):
                recovery = exact_four + gain
                rows.append({
                    **identity, "budget_regime": "fixed_correction_budget",
                    "physical_budget_pages": budget, "physical_pages": 3 * (budget // 3),
                    "physical_budget_bpw": budget / 768, "strict_all_in_total_bytes": np.nan,
                    "budget_total_bytes": np.nan, "budget_total_bpw": np.nan,
                    "strict_all_in_budget_pass": np.nan, "strict_all_in_slack_bytes": np.nan,
                    "factor_config_id": "exact_full_down_gram_training_only",
                    "factor_family": "exact_full_down_gram", "total_rank": np.nan,
                    "factor_payload_bytes": np.nan, "combined_metadata_bytes": np.nan,
                    "combined_metadata_bpw": np.nan, "solver": solver,
                    "selected_seed": "synthetic", "recovery": recovery,
                    "four_state_reference_recovery": exact_four, "pr11_four_state_recovery": .90,
                    "recovery_gain_vs_four_state_reference": gain,
                    "recovery_gain_vs_pr11_four_state": recovery - .90,
                    "seed_optimizer_gain_vs_pr11_four_state": exact_four - .90,
                    "four_state_reference_kind": "exact_full_gram_four_state_same_solver",
                    "pr10_exact_hybrid_recovery": .95,
                    "set_gain_retention_vs_pr10_exact_hybrid": recovery / .95,
                    "compressed_predicted_damage": 1 - recovery,
                    "exact_qenergy_damage": 1 - recovery,
                    "damage_prediction_error_over_base": 0.0, "selector_runtime_ms": 2.0,
                    "field_build_runtime_ms": 0.0,
            "field_build_runtime_ms": 0.2,
                    "selector_compute_macs": np.nan, "selector_compute_gate_pass": np.nan,
                    "selector_compute_accounting_bound": "oracle", "coordinate_sweeps": 8,
                    "selected_seed_coordinate_sweeps": 4, "coordinate_accepted_moves": 12,
                    "local_evaluated_passes": 4, "local_accepted_bundles": 2,
                    "local_candidate_moves_per_pass": 0, "local_maximum_shortlist": 0,
                    "allowed_state_count": allowed, "seed_count": 2,
                    "uses_exact_mixed_h4": True, "exact_gram_oracle": True,
                    "promotable": False,
                    "selection_regime": "h0_exact_mixed_h4_split_action_geometry_ceiling",
                    "selector_non_mac_operations": "control", "selected_states": state_text,
                    "state_counts": count_text, "split_only_units": 0,
                    "gate_high_units": budget // 3, "up_high_units": budget // 3,
                    "down_high_units": budget // 3,
                })
        for factor in factors:
            budget = config["strict_all_in_page_budgets"][factor]
            for solver, gain, macs in (
                (config["four_state_reference_solver_id"], 0.0, 700_000),
                (config["primary_solver_id"], .002, 1_200_000),
            ):
                rows.append(ordinary_row(
                    identity, factor, solver, budget, "strict_all_in_one_bpw",
                    same + gain, same, old, macs,
                ))
    frame = pd.DataFrame(rows)
    frontier = validation / "split_interaction_field_frontier.parquet"
    frame.to_parquet(frontier, index=False)
    facts = {
        "completed": True, "failures": [], "completed_work_units": [f"unit-{i}" for i in range(12)],
        "test_rows_admitted": False, "test_scientific_values_used": False,
        "selection_split": "validation", "config_sha256": sha256(config_path),
        "runner_sha256": sha256(RUNNER), "split_core_sha256": sha256(CORE),
        "pr11_run_facts_sha256": config["pr11_run_facts_sha256"],
        "pr11_factor_manifest_sha256": config["pr11_factor_manifest_sha256"],
        "pr11_frontier_sha256": config["pr11_frontier_sha256"],
        "frontier_sha256": sha256(frontier), "frontier_rows": len(frame),
        "validation_scope_audit": {"expected_unique_invocations": 69, "observed_unique_invocations": 69, "observed_layer_expert_cells": 12},
        "state_page_costs": config["state_page_costs"],
        "inherited_four_state_map": config["inherited_four_state_map"],
        "strict_all_in_total_bytes": 393_216,
        "strict_all_in_page_budgets": config["strict_all_in_page_budgets"],
        "incremental_coordinate_updates": True,
    }
    (validation / "run_facts.json").write_text(json.dumps(facts, sort_keys=True))
    return config_path, validation, config, frame

def test_validate_summary_and_promotion_contract(tmp_path):
    config_path, validation, config, _ = _bundle(tmp_path)
    loaded, facts, frame = validate_evidence_bundle(
        config_path, validation, runner_path=RUNNER, core_path=CORE,
    )
    summary = summary_table(frame)
    promotion = promotion_payload(loaded, facts, frame, summary)
    assert len(frame) == 69 * 62
    assert promotion["status"] == "continue_to_predicted_mixed_hidden_interface"
    assert promotion["selected_validation_configuration"] is None
    assert promotion["passing_validation_configurations"] == sorted(config["primary_factor_config_ids"])
    exact = promotion["exact_gram_one_bpw"]
    assert exact["recovery_gain_median"] == pytest.approx(.004)
    assert promotion["recommended_strict_all_in_factor"]["factor_config_id"] == (
        "joint_eigh_tail8_exact0_int4_per_row_hadamard"
    )
    assert promotion["int8_minus_int4_all_in_recovery"]["median"] == pytest.approx(0.0)
    attributed = summary[(
        (summary["budget_regime"] == "fixed_correction_budget")
        & (summary["solver"] == config["primary_solver_id"])
        & (summary["physical_budget_pages"] == 768)
    )].iloc[0]
    assert attributed["recovery_gain_median"] == pytest.approx(.002)
    assert attributed["seed_optimizer_gain_median"] == pytest.approx(.001)
    assert attributed["gain_vs_pr11_median"] == pytest.approx(.003)


def test_validation_fails_on_state_page_tampering(tmp_path):
    config_path, validation, _, frame = _bundle(tmp_path)
    frame.loc[0, "physical_pages"] -= 1
    frontier = validation / "split_interaction_field_frontier.parquet"
    frame.to_parquet(frontier, index=False)
    facts_path = validation / "run_facts.json"
    facts = json.loads(facts_path.read_text())
    facts["frontier_sha256"] = sha256(frontier)
    facts_path.write_text(json.dumps(facts))
    with pytest.raises(RuntimeError, match="state/page"):
        validate_evidence_bundle(
            config_path, validation, runner_path=RUNNER, core_path=CORE,
        )



def test_validation_fails_on_strict_all_in_byte_tampering(tmp_path):
    config_path, validation, _, frame = _bundle(tmp_path)
    index = frame.index[frame["budget_regime"] == "strict_all_in_one_bpw"][0]
    frame.loc[index, "budget_total_bytes"] += 1
    frontier = validation / "split_interaction_field_frontier.parquet"
    frame.to_parquet(frontier, index=False)
    facts_path = validation / "run_facts.json"
    facts = json.loads(facts_path.read_text())
    facts["frontier_sha256"] = sha256(frontier)
    facts_path.write_text(json.dumps(facts))
    with pytest.raises(RuntimeError, match="total budget bytes"):
        validate_evidence_bundle(
            config_path, validation, runner_path=RUNNER, core_path=CORE,
        )

def test_analyzer_end_to_end(tmp_path, monkeypatch):
    config_path, validation, _, _ = _bundle(tmp_path)
    output = tmp_path / "analysis_a"
    spec = importlib.util.spec_from_file_location(
        "analyze_split_interaction_field_study",
        ROOT / "scripts/analyze_split_interaction_field_study.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        sys, "argv",
        ["analyze", "--config", str(config_path), "--validation-dir", str(validation),
         "--output", str(output)],
    )
    module.main()
    promotion = json.loads((output / "split_interaction_field_promotions.json").read_text())
    manifest = json.loads((output / "analysis_manifest.json").read_text())
    assert len(promotion["passing_validation_configurations"]) == 2
    assert len(manifest["outputs"]) == 11
    assert (output / "SPLIT_INTERACTION_FIELD_REPORT.md").stat().st_size > 1000
    assert b"NaN" not in (output / "split_interaction_field_analysis.json").read_bytes()

    output_b = tmp_path / "analysis_b"
    monkeypatch.setattr(
        sys, "argv",
        ["analyze", "--config", str(config_path), "--validation-dir", str(validation),
         "--output", str(output_b)],
    )
    module.main()
    assert {
        path.name: path.read_bytes() for path in output.iterdir()
    } == {
        path.name: path.read_bytes() for path in output_b.iterdir()
    }
