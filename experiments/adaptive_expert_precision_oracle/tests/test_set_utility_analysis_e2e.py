from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oracle_study.set_utility_analysis import (
    CONTAINED,
    FULL_TARGET,
    INDEPENDENT,
    PREDICTED,
    freeze_validation_promotions,
    recompute_accounting,
    sha256,
    validate_evidence_bundle,
)


EXPERIMENT = Path(__file__).parents[1]

FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "set_utility_analysis_fixtures", Path(__file__).with_name("test_set_utility_analysis.py"),
)
assert FIXTURE_SPEC is not None and FIXTURE_SPEC.loader is not None
fixtures = importlib.util.module_from_spec(FIXTURE_SPEC)
FIXTURE_SPEC.loader.exec_module(fixtures)

ANALYZER_SPEC = importlib.util.spec_from_file_location(
    "analyze_set_utility_distillation_e2e",
    EXPERIMENT / "scripts" / "analyze_set_utility_distillation.py",
)
assert ANALYZER_SPEC is not None and ANALYZER_SPEC.loader is not None
analyzer = importlib.util.module_from_spec(ANALYZER_SPEC)
ANALYZER_SPEC.loader.exec_module(analyzer)


def test_candidate_rerank_json_nulls_only_declared_support_template_fields() -> None:
    frame = pd.DataFrame([
        {
            "selector_family": "support_template",
            "selector_compute_additions": np.nan,
            "selector_scale_multiplications": np.nan,
            "rerank_incremental_compute_macs": 17,
        },
        {
            "selector_family": "block_pq_residual_synopsis",
            "selector_compute_additions": 23,
            "selector_scale_multiplications": 29,
            "rerank_incremental_compute_macs": 31,
        },
    ])
    records = analyzer.candidate_rerank_json_records(frame)
    assert pd.isna(frame.loc[0, "selector_compute_additions"])
    assert pd.isna(frame.loc[0, "selector_scale_multiplications"])
    assert records[0]["selector_compute_additions"] is None
    assert records[0]["selector_scale_multiplications"] is None
    assert records[0]["rerank_incremental_compute_macs"] == 17
    assert records[1]["selector_compute_additions"] == 23
    assert records[1]["selector_scale_multiplications"] == 29

    outside_family = frame.copy()
    outside_family.loc[1, "selector_compute_additions"] = np.nan
    with pytest.raises(RuntimeError, match="missing outside a support-template"):
        analyzer.candidate_rerank_json_records(outside_family)

    non_finite_present = frame.copy()
    non_finite_present.loc[0, "selector_compute_additions"] = np.inf
    with pytest.raises(RuntimeError, match="non-finite"):
        analyzer.candidate_rerank_json_records(non_finite_present)


def test_synthetic_fit_validation_accounting_freeze_and_report_end_to_end(tmp_path: Path) -> None:
    config = fixtures.locked_scope_config()
    config.update({
        "base_pr9_commit": "fixture-base",
        "pq_additive_stages": [1],
        "reference": {
            **config["reference"],
            "revision": "fixture-revision",
        },
    })
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True))
    dependency_paths = {
        "runner_sha256": EXPERIMENT / "scripts/run_set_utility_distillation.py",
        "teacher_core_sha256": EXPERIMENT / "src/oracle_study/unit_set_teacher.py",
        "oracle_core_sha256": EXPERIMENT / "src/oracle_study/set_utility_oracles.py",
        "selector_core_sha256": EXPERIMENT / "src/oracle_study/set_utility_selector.py",
        "direct_predictor_core_sha256": EXPERIMENT / "src/oracle_study/direct_set_predictor.py",
        "neuron_selector_core_sha256": EXPERIMENT / "src/oracle_study/neuron_selector.py",
        "prior_runner_sha256": EXPERIMENT / "scripts/run_neuron_selector_distillation.py",
        "base_runner_sha256": EXPERIMENT / "scripts/run_sparse_streaming_study.py",
        "sparse_streaming_core_sha256": EXPERIMENT / "src/oracle_study/sparse_streaming.py",
        "mxfp4_embed_core_sha256": EXPERIMENT / "src/oracle_study/mxfp4_embed.py",
        "selective_pages_runner_sha256": EXPERIMENT / "scripts/run_mxfp4_selective_pages.py",
        "phase_a_runner_sha256": EXPERIMENT / "scripts/run_phase_a_remote.py",
    }
    dependency_hashes = {key: sha256(path) for key, path in dependency_paths.items()}

    fit_dir = tmp_path / "fit"
    fit_dir.mkdir()
    fit_layers = {}
    for layer_text, experts in config["locked_exact_sampled_experts"].items():
        layer = int(layer_text)
        shard = fit_dir / f"set_utility_fit_layer_{layer}.npz"
        shard.write_bytes(f"synthetic immutable fit shard {layer}".encode())
        sidecar_record = {
            "layer": layer,
            "shard": shard.name,
            "shard_sha256": sha256(shard),
            "shard_bytes": shard.stat().st_size,
            "array_count": 0,
            "fit_split": "train",
            "validation_or_test_rows_used": False,
            "runtime_provenance": fixtures.runtime_provenance(),
            "teacher_rows_by_cohort": {cohort: 10 for cohort in fixtures.FIT_COHORTS},
            "template_cohort_example_counts_by_expert": {
                str(expert): {cohort: 10 for cohort in fixtures.FIT_COHORTS}
                for expert in experts
            },
            "template_entries": [
                {
                    "cohort": cohort,
                    "expert_id": int(expert),
                    "available_templates": 8,
                    "requested_template_counts": [8],
                    "training_examples": 10,
                    "pairing_semantics": (
                        "actual_routed_occurrences"
                        if cohort == fixtures.FIT_COHORTS[0]
                        else "synthetic_all_x_by_expert_augmentation"
                    ),
                }
                for expert in experts
                for cohort in fixtures.FIT_COHORTS
            ],
            "selector_entries": [
                {**fixtures.pq_entry(), "expert_id": int(expert)} for expert in experts
            ],
        }
        sidecar = fit_dir / f"set_utility_fit_layer_{layer}.json"
        sidecar.write_text(json.dumps(sidecar_record, sort_keys=True))
        fit_layers[layer_text] = {**sidecar_record, "sidecar_sha256": sha256(sidecar)}
    fit_manifest = {
        "schema_version": 1,
        "run_id": config["run_id"],
        "config_sha256": sha256(config_path),
        "fit_split": "train",
        "validation_or_test_rows_used": False,
        "test_rows_admitted": False,
        "test_rows_used": False,
        "monolithic_source_contains_test_rows": True,
        "cross_capture_request_id_audit": fixtures.cross_capture_request_id_audit(),
        "direct_candidate_head_supervision": fixtures.JOINT_CANDIDATE_COVERAGE_SEMANTICS,
        "runtime_provenance": fixtures.runtime_provenance(),
        "layers": fit_layers,
    }
    manifest_path = fit_dir / "set_utility_fit_manifest.json"
    manifest_path.write_text(json.dumps(fit_manifest, sort_keys=True))
    fit_facts = {
        "completed": True,
        "failures": [],
        "fit_split": "train",
        "validation_or_test_rows_used": False,
        "test_rows_admitted": False,
        "test_rows_used": False,
        "monolithic_source_contains_test_rows": True,
        "request_separation_verified": True,
        "cross_capture_request_id_audit": fixtures.cross_capture_request_id_audit(),
        "direct_candidate_head_supervision": fixtures.JOINT_CANDIDATE_COVERAGE_SEMANTICS,
        "runtime_provenance": fixtures.runtime_provenance(),
        "codec_locked": True,
        "config_sha256": sha256(config_path),
        "exact_capture_sha256": config["locked_capture_sha256"]["exact_checkpoint"],
        "cross_capture_sha256": config["locked_capture_sha256"]["cross_reference"],
        "tree_sha256": config["locked_tree_sha256"],
        "checkpoint_config_sha256": config["reference"]["config_sha256"],
        "checkpoint_index_sha256": config["reference"]["index_sha256"],
        "manifest_sha256": sha256(manifest_path),
        **dependency_hashes,
    }
    fit_facts_path = fit_dir / "fit_facts.json"
    fit_facts_path.write_text(json.dumps(fit_facts, sort_keys=True))

    validation_dir = tmp_path / "validation"
    validation_dir.mkdir()
    hybrid_rows = []
    template_rows = []
    selector_rows = []
    candidate_rows = []
    hybrid_recovery = {
        "coherent_independent_abc_pr9_control": 0.938,
        "coherent_exact_set_fixed_greedy_teacher": 0.940,
        "factorized_fixed_greedy_pr9_control": 0.939,
        "direct_GD_hybrid_forward": 0.942,
        "hybrid_forward_plus_1_2_3_unit_local_search": 0.946,
    }
    for identity in fixtures.locked_validation_identities():
        for family, recovery in hybrid_recovery.items():
            for budget in (0.5, 0.75, 1.0):
                pages = int(round(768 * budget))
                hybrid_rows.append({
                    **identity,
                    "selector_family": family,
                    "selection_regime": "h0_exact_oracle",
                    "physical_budget_bpw": budget,
                    "physical_pages": pages,
                    "physical_bytes": pages * 512,
                    "physical_bpw": budget,
                    "page_amplification": 1.0,
                    "recovery": recovery,
                })
        for cohort in fixtures.FIT_COHORTS:
            for template_selector in fixtures.TEMPLATE_SELECTORS:
              for repair_count in (0, 16, 32, 64):
                selector_id = (
                    f"template:{cohort}:K8:"
                    f"{template_selector}:repair{repair_count}"
                )
                template_rows.append({
                    **identity,
                    "template_cohort": cohort,
                    "pairing_semantics": (
                        "actual_routed_occurrences"
                        if cohort == fixtures.FIT_COHORTS[0]
                        else "synthetic_all_x_by_expert_augmentation"
                    ),
                    "template_selector": template_selector,
                    "template_count_requested": 8,
                    "template_count_effective": 8,
                    "repair_count": repair_count,
                    "candidate_units": 256,
                    "candidate_cap_units": 256,
                    "candidate_cap_pass": True,
                    "physical_pages": 768,
                    "physical_bytes": 393_216,
                    "physical_bpw": 1.0,
                    "candidate_overfetch": 4 / 3,
                    "validation_oracle_path_utility_retained": 0.96,
                    "selection_regime": "h0_validation_template_oracle_not_classifier",
                    "promotable": False,
                })
                for semantics in (PREDICTED, INDEPENDENT, CONTAINED, FULL_TARGET):
                    candidate = {
                        **fixtures.candidate_row(
                            semantics, payloads_already_read=False,
                            q2_gate_up_scales_already_read=False,
                        ),
                        **identity,
                        "selector_config_id": selector_id,
                        "selector_family": "support_template",
                        "candidate_cap_units": 256,
                        "candidate_cap_pass": True,
                        "template_bank_metadata_bytes": 512,
                        "selector_compute_macs": 0,
                        "selector_abc_score_units": 0,
                        "selector_abc_score_compute_macs": 0,
                        "selector_abc_score_macs_per_unit": 0,
                        "selector_compute_additions": np.nan,
                        "selector_scale_multiplications": np.nan,
                        "selector_bytes_read": 512 + 393_216,
                        "promotable": False,
                    }
                    candidate["total_selector_compute_macs"] = candidate[
                        "rerank_incremental_compute_macs"
                    ]
                    candidate["total_selector_bytes_read"] = (
                        candidate["selector_bytes_read"]
                        + max(candidate["rerank_incremental_bytes_read"] - 393_216, 0)
                    )
                    candidate_rows.append(candidate)
        selector_rows.append({**fixtures.selector_row(), **identity})
        for semantics in (PREDICTED, INDEPENDENT, CONTAINED, FULL_TARGET):
            candidate = {**fixtures.candidate_row(semantics), **identity}
            if semantics == PREDICTED:
                candidate.update(
                    recovery=0.90,
                    oracle_set_gain_retention=0.90,
                    set_gain=0.90 * candidate["oracle_set_gain"],
                    selected_units=json.dumps(list(range(64, 256))),
                )
            candidate_rows.append(candidate)

    artifact_rows = {
        "hybrid_oracle_frontier": hybrid_rows,
        "template_frontier": template_rows,
        "selector_frontier": selector_rows,
        "candidate_frontier": candidate_rows,
    }
    artifact_names = {
        "hybrid_oracle_frontier": "hybrid_oracle_frontier.parquet",
        "template_frontier": "support_template_frontier.parquet",
        "selector_frontier": "pq_high_rank_selector_frontier.parquet",
        "candidate_frontier": "candidate_set_rerank_frontier.parquet",
    }
    for key, rows in artifact_rows.items():
        pd.DataFrame(rows).to_parquet(validation_dir / artifact_names[key], index=False)

    locked_ids = fixtures.locked_validation_identities()
    locked_cells = sorted({(row["layer"], row["expert_id"]) for row in locked_ids})
    validation_scope_audit = {
        "expected_unique_invocations": 69,
        "observed_unique_invocations": 69,
        "expected_layer_expert_cells": 12,
        "observed_layer_expert_cells": 12,
        "locked_layer_expert_cells": [list(cell) for cell in locked_cells],
        "observed_layer_expert_cell_records": [
            {
                "layer": layer,
                "expert_id": expert,
                "invocations": sum(
                    row["layer"] == layer and row["expert_id"] == expert
                    for row in locked_ids
                ),
            }
            for layer, expert in locked_cells
        ],
        "validation_scope_complete": True,
    }
    run_facts = {
        "completed": True,
        "failures": [],
        "run_id": config["run_id"],
        "evaluation_split": "validation",
        "config_sha256": sha256(config_path),
        "capture_sha256": config["locked_capture_sha256"]["exact_checkpoint"],
        "tree_sha256": config["locked_tree_sha256"],
        "checkpoint_config_sha256": config["reference"]["config_sha256"],
        "checkpoint_index_sha256": config["reference"]["index_sha256"],
        "fit_manifest_sha256": sha256(manifest_path),
        "fit_facts_sha256": sha256(fit_facts_path),
        "request_separation_verified": True,
        "cross_capture_request_id_audit": fixtures.cross_capture_request_id_audit(),
        "direct_candidate_head_supervision": fixtures.JOINT_CANDIDATE_COVERAGE_SEMANTICS,
        "runtime_provenance": fixtures.runtime_provenance(),
        "validation_scope_audit": validation_scope_audit,
        "test_rows_admitted": False,
        "test_rows_used": False,
        "monolithic_source_contains_test_rows": True,
        "codec_locked": True,
        "expected_unique_invocations": 69,
        "observed_unique_invocations": 69,
        "expected_layer_expert_cells": 12,
        "completed_work_units": [
            f"validation:{layer}:{expert}"
            for layer, experts in config["locked_exact_sampled_experts"].items()
            for expert in experts
        ],
        "artifact_row_counts": {key: len(rows) for key, rows in artifact_rows.items()},
        **dependency_hashes,
    }
    (validation_dir / "run_facts.json").write_text(json.dumps(run_facts, sort_keys=True))

    evidence = validate_evidence_bundle(config_path, fit_dir, validation_dir, EXPERIMENT)
    accounted = {
        kind: recompute_accounting(kind, frame, evidence.config, evidence.selector_index)
        for kind, frame in evidence.frames.items()
    }
    promotion, _ = freeze_validation_promotions(evidence, accounted)
    assert promotion["selection_split"] == "validation"
    assert promotion["test_scientific_rows_or_values_consulted"] is False
    assert promotion["test_metadata_audit"] == fixtures.TEST_METADATA_AUDIT
    assert promotion["h0_late_candidate_policy"]["selected_validation_configuration"][
        "rerank_semantics"
    ] == INDEPENDENT

    output = tmp_path / "analysis"
    analyzer.analyze(argparse.Namespace(
        config=config_path,
        fit_dir=fit_dir,
        validation_dir=validation_dir,
        output=output,
    ))
    frozen = json.loads((output / "set_utility_promotions.json").read_text())
    assert frozen["h4_predictor_trained_or_evaluated"] is False
    assert frozen["contained_interaction_systems_oracle_used_for_promotion"] is False
    assert frozen["support_template_oracle_used_for_selector_promotion"] is False
    report = (output / "SET_UTILITY_DISTILLATION_REPORT.md").read_text()
    assert "exact-checkpoint validation-only" in report
    assert "H0-late" in report
    assert "not a global" in report
    assert "4,108" in report
    assert "candidate_head_supervision" not in report
    assert "supplies and optimizes no" in report
    assert "exclusion-regret supervision" in report
    assert "physically" in report and "No test scientific tensor or value" in report
    assert "`split`" in report and "`request_id`" in report
    assert "both charge the 1,024 output-scale" in report
    assert "exactly 69 unique validation" in report
    assert "stale `promotable=true`" not in report
    assert "intentional top-2 unions above 256" in report
    manifest = json.loads((output / "analysis_manifest.json").read_text())
    assert manifest["monolithic_capture_archives_physically_loaded_by_runner"] is True
    assert manifest["test_split_and_request_metadata_audited"] is True
    assert manifest["test_metadata_audit"] == fixtures.TEST_METADATA_AUDIT
    assert manifest["test_scientific_rows_or_values_admitted_or_used"] is False
    assert manifest["selector_grid_audit"]["locked_unique_invocations"] == 69
    assert manifest["selector_grid_audit"]["locked_layer_expert_cells"] == 12
    assert manifest["selector_grid_audit"]["complete"] is True
    assert manifest["selector_grid_audit"]["hybrid_grid"]["expected_rows"] == 1_035
    assert manifest["selector_grid_audit"]["template_grid"]["expected_template_rows"] == 1_656
    assert manifest["selector_grid_audit"]["template_grid"][
        "expected_support_candidate_rows"
    ] == 6_624
    assert manifest["selector_grid_audit"]["template_grid"][
        "explicitly_nonpromotable"
    ] is True
    assert "candidate_rerank_frontier.svg" in manifest["outputs"]
    assert "candidate_rerank_accounting.csv" in manifest["outputs"]
    rerank_csv = output / "candidate_rerank_accounting.csv"
    assert manifest["outputs"][rerank_csv.name] == {
        "bytes": rerank_csv.stat().st_size,
        "sha256": sha256(rerank_csv),
    }
    rerank = pd.read_csv(rerank_csv)
    support = rerank[rerank["selector_family"] == "support_template"]
    assert support["selector_compute_additions"].isna().all()
    assert support["selector_scale_multiplications"].isna().all()
    pq = rerank[rerank["selector_family"] == "block_pq_residual_synopsis"]
    by_semantics = {
        semantic: pq[pq["rerank_semantics"] == semantic].iloc[0]
        for semantic in (PREDICTED, INDEPENDENT, CONTAINED, FULL_TARGET)
    }
    assert bool(by_semantics[PREDICTED]["rerank_accounting_is_lower_bound"]) is False
    assert bool(by_semantics[INDEPENDENT]["rerank_accounting_is_lower_bound"]) is False
    assert bool(by_semantics[CONTAINED]["rerank_accounting_is_lower_bound"]) is True
    assert by_semantics[CONTAINED]["rerank_accounting_bound"] == (
        fixtures.INTERACTION_RERANK_ACCOUNTING_BOUND
    )
    assert by_semantics[FULL_TARGET]["rerank_unaccounted_overhead"] == (
        fixtures.INTERACTION_RERANK_UNACCOUNTED
    )
    assert int(by_semantics[INDEPENDENT]["rerank_q2_gate_up_parent_code_bytes_read"]) == 262_144
    assert int(by_semantics[FULL_TARGET]["rerank_q2_gate_up_parent_code_bytes_read"]) == 524_288
    assert int(by_semantics[INDEPENDENT]["rerank_q2_gate_up_scale_bytes_read"]) == 0
    assert int(by_semantics[INDEPENDENT]["rerank_q4_response_scale_multiplications"]) == 32_768
    assert manifest["analysis_sources"]["analyzer_wrapper"]["path"] == (
        "scripts/analyze_set_utility_distillation.py"
    )
    assert manifest["analysis_sources"]["analyzer_core"]["path"] == (
        "src/oracle_study/set_utility_analysis.py"
    )
    assert manifest["analysis_sources"]["analyzer_wrapper"]["sha256"] == sha256(
        EXPERIMENT / "scripts/analyze_set_utility_distillation.py"
    )
    accounting = json.loads((output / "selector_compute_storage_accounting.json").read_text())
    assert accounting["schema_version"] == 5
    assert accounting["candidate_rerank_optional_null_contract"] == {
        "csv_encoding": "empty_cell_preserving_raw_missing_value",
        "fields": ["selector_compute_additions", "selector_scale_multiplications"],
        "json_encoding": "null",
        "reason": "not_applicable_validation_oracle_has_no_deployable_selector",
        "selector_family": "support_template",
    }
    support_json = [
        row for row in accounting["candidate_rerank_rows"]
        if row["selector_family"] == "support_template"
    ]
    assert support_json
    assert all(row["selector_compute_additions"] is None for row in support_json)
    assert all(row["selector_scale_multiplications"] is None for row in support_json)
    assert all(row["rerank_incremental_compute_macs"] is not None for row in support_json)
    assert accounting["q4_response_resident_payload_contract"] == {
        "candidate_packet_bytes_per_unit": 1_536,
        "candidate_packet_is_suffix_only": True,
        "q2_parent_code_bytes_per_unit": 1_024,
        "q2_scale_bytes_per_unit": 128,
        "scale_bytes_deduplicated_when_selector_already_read": True,
        "scale_multiplications_per_unit": 128,
    }
    assert accounting["candidate_rerank_rows"]
    assert accounting["audited_selector_accounting"][
        "high_rank_low_bit_output_scale_multiplications_per_invocation"
    ] == 1_024
    assert accounting["audited_selector_accounting"]["standalone_q2_unit_feature_bytes"] == 3_072
    assert accounting["audited_selector_accounting"]["standalone_abc_score_compute_macs"] == 3_072
    assert accounting["audited_selector_accounting"]["serialized_abc_payload_charged_exactly"] is True
    caveat = accounting["known_raw_accounting_caveats"][
        "contained_and_full_target_interaction_paths"
    ]
    assert caveat["accounting_bound"] == fixtures.INTERACTION_RERANK_ACCOUNTING_BOUND
    assert caveat["missing_from_charge"] == fixtures.INTERACTION_RERANK_UNACCOUNTED
    assert "raw_annotation_authority" not in accounting
