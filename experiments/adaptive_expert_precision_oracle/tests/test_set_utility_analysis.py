from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oracle_study.set_utility_analysis import (
    ABC_SCORE_MAC_CONVENTION,
    CANDIDATE_PACKET_CONTENTS,
    CONTAINED,
    FULL_TARGET,
    HYBRID_FAMILIES,
    INDEPENDENT,
    INTERACTION_RERANK_ACCOUNTING_BOUND,
    INTERACTION_RERANK_UNACCOUNTED,
    LOCKED_HYBRID_BUDGETS,
    FIT_COHORTS,
    JOINT_CANDIDATE_COVERAGE_SEMANTICS,
    PREDICTED,
    RERANK_NON_MAC_CONVENTION,
    TEMPLATE_SELECTORS,
    TEST_METADATA_AUDIT,
    ValidatedEvidence,
    _configured_selector_contract,
    _manifest_accounting,
    _promotion_exclusion_reasons,
    _validate_code_hashes,
    _validate_candidate_rerank_accounting,
    _validate_fit_bundle,
    _validate_hybrid_grid,
    _validate_locked_validation_scope,
    _validate_selector_grid_coverage,
    _validate_selector_promotion_contract,
    _validate_template_grid,
    freeze_validation_promotions,
    load_validation_frame,
    recompute_accounting,
    sha256,
    summarize,
    validate_config_contract,
)


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_set_utility_distillation.py"
SPEC = importlib.util.spec_from_file_location("analyze_set_utility_distillation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)

EXPERIMENT = Path(__file__).parents[1]


def test_all_executable_dependency_hashes_are_mandatory() -> None:
    paths = {
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
    facts = {key: sha256(path) for key, path in paths.items()}
    _validate_code_hashes(facts, EXPERIMENT)
    for key in paths:
        incomplete = dict(facts)
        incomplete.pop(key)
        with pytest.raises(RuntimeError, match=key):
            _validate_code_hashes(incomplete, EXPERIMENT)


def config() -> dict:
    return {
        "run_id": "fixture",
        "layers": [0],
        "page_size_bytes": 512,
        "expert_weights": 3_145_728,
        "experts_per_layer": 256,
        "reference_bpw": 4.25,
        "suffix_bpw_per_complete_representation": 2.0,
        "selection_split": "validation",
        "evaluation_splits": ["validation"],
        "fit_splits": ["train"],
        "no_test_rows_admitted": True,
        "no_test_rows_used": True,
        "no_test_tuning": True,
        "candidate_units": 256,
        "applied_units": 192,
        "expected_validation_unique_invocations": 69,
        "expected_validation_layer_expert_cells": 12,
        "expected_cross_capture_shared_request_ids": 0,
        "direct_candidate_head_supervision": JOINT_CANDIDATE_COVERAGE_SEMANTICS,
        "template_training_cohorts": list(FIT_COHORTS),
        "pq_block_size": 32,
        "pq_codebook_size": 16,
        "locked_tree_sha256": "c" * 64,
        "locked_capture_sha256": {
            "exact_checkpoint": "a" * 64,
            "cross_reference": "b" * 64,
        },
        "reference": {"config_sha256": "d" * 64, "index_sha256": "e" * 64},
        "promotion_gates": {
            "candidate_median_recovery": 0.94,
            "candidate_p10_recovery": 0.88,
            "candidate_p10_set_gain_retention": 0.97,
            "max_selector_metadata_bpw": 0.35,
            "max_selector_compute_macs": 1_572_864,
            "max_storage_multiplier": 5.0,
            "hybrid_min_median_improvement_at_1bpw": 0.005,
            "template_top1_median_gain_retention": 0.95,
            "template_top2_median_gain_retention": 0.98,
            "template_top2_p10_gain_retention": 0.95,
        },
    }


def identity(index: int = 0) -> dict:
    return {
        "capture_source": "exact_checkpoint",
        "evaluation_split": "validation",
        "request_id": f"v{index}",
        "position": index,
        "layer": 0,
        "expert_id": 7,
    }


def locked_scope_config() -> dict:
    value = config()
    value.update({
        "layers": [0, 4, 20, 39],
        "locked_exact_sampled_experts": {
            "0": {"7": "cold", "8": "median", "9": "hot"},
            "4": {"17": "cold", "18": "median", "19": "hot"},
            "20": {"27": "cold", "28": "median", "29": "hot"},
            "39": {"37": "cold", "38": "median", "39": "hot"},
        },
        "pq_additive_stages": [],
        "high_rank_low_bit_controls": [],
        "direct_predictor_training_cohorts": [],
        "hybrid_budgets_bpw": [0.5, 0.75, 1.0],
        "template_counts_primary": [8],
        "template_counts_augmented": [8],
        "template_repair_counts": [0, 16, 32, 64],
    })
    return value


def locked_validation_identities(*, omit_last_cell: bool = False) -> list[dict]:
    cfg = locked_scope_config()
    cells = [
        (int(layer), int(expert))
        for layer, experts in cfg["locked_exact_sampled_experts"].items()
        for expert in experts
    ]
    if omit_last_cell:
        cells = cells[:-1]
    rows = []
    for index in range(69):
        layer, expert = cells[index % len(cells)]
        rows.append({
            "capture_source": "exact_checkpoint",
            "evaluation_split": "validation",
            "request_id": f"locked-v{index}",
            "position": index,
            "layer": layer,
            "expert_id": expert,
        })
    return rows


def locked_template_manifest() -> dict:
    cfg = locked_scope_config()
    return {
        "layers": {
            str(layer): {
                "template_cohort_example_counts_by_expert": {
                    str(expert): {cohort: 8 for cohort in FIT_COHORTS}
                    for expert in experts
                },
                "template_entries": [
                    {
                        "cohort": cohort,
                        "expert_id": int(expert),
                        "available_templates": 8,
                        "requested_template_counts": [8],
                        "training_examples": 8,
                        "pairing_semantics": (
                            "actual_routed_occurrences"
                            if cohort == FIT_COHORTS[0]
                            else "synthetic_all_x_by_expert_augmentation"
                        ),
                    }
                    for expert in experts
                    for cohort in FIT_COHORTS
                ],
            }
            for layer, experts in cfg["locked_exact_sampled_experts"].items()
        },
    }


def runtime_provenance() -> dict:
    return {
        "device": "cpu",
        "gpu_names_used": [],
        "torch_version": "fixture-torch",
        "torch_cuda_runtime": None,
        "numpy_version": np.__version__,
        "python_version": "3.fixture",
        "host": "fixture-host",
    }


def cross_capture_request_id_audit() -> dict:
    return {
        "exact_unique_request_ids": 10,
        "cross_unique_request_ids": 10,
        "shared_request_id_count": 0,
        "expected_shared_request_id_count": 0,
        "expected_shared_request_ids_match": True,
        "shared_same_split_request_id_counts": {"train": 0, "validation": 0, "test": 0},
        "shared_same_split_request_ids": {"train": [], "validation": [], "test": []},
        "incompatible_overlap_count": 0,
        "incompatible_overlaps": [],
        "cross_source_train_to_nontrain_overlap_count": 0,
        "cross_source_train_to_nontrain_overlaps": [],
        "split_compatible": True,
    }


def pq_entry() -> dict:
    return {
        "config_id": "block_pq_s1_separate_gate_up",
        "selector_family": "block_pq_residual_synopsis",
        "expert_id": 7,
        "stages": 1,
        "expert_specific_bytes": 32_768,
        "layer_shared_bytes": 131_072,
        "resident_scale_bytes_read": 65_536,
    }


def base_accounting(*, fetched: int, applied: int, direct: bool = False) -> dict:
    cfg = config()
    pq = pq_entry()
    if direct:
        unit_input, hidden = 8, 24
        model_macs = 512 * unit_input * hidden + unit_input * hidden + 512 * hidden * hidden + 512 * hidden
        model_additions = unit_input * 511
        expert = pq["expert_specific_bytes"]
        shared = pq["layer_shared_bytes"] + 10_100
        macs = 65_536 + model_macs
        abc_score_units = abc_score_macs = abc_score_macs_per_unit = 0
        additions = 64_512 + model_additions
        scales = 65_536
        selector_bytes = 495_616 + 10_000 + 100 + 2 * 512 * (3 + 2) + 3_084
    else:
        expert, shared = pq["expert_specific_bytes"], pq["layer_shared_bytes"]
        abc_score_units = 512
        abc_score_macs_per_unit = 6
        abc_score_macs = abc_score_units * abc_score_macs_per_unit
        macs, additions, scales = 65_536 + abc_score_macs, 64_512, 65_536
        selector_bytes = 495_616 + 3_072 + 3_084
    abc = 3_084
    metadata = expert + abc + shared / 256
    physical_bytes = fetched * 3 * 512
    logical_bytes = applied * 3 * 512
    return {
        "physical_pages": fetched * 3,
        "physical_bytes": physical_bytes,
        "physical_bpw": 8 * physical_bytes / cfg["expert_weights"],
        "logical_actions": applied,
        "logical_bytes": logical_bytes,
        "logical_bpw": 8 * logical_bytes / cfg["expert_weights"],
        "page_amplification": physical_bytes / logical_bytes,
        "candidate_overfetch": fetched / applied,
        "expert_specific_metadata_bytes": expert,
        "layer_shared_metadata_bytes": shared,
        "layer_shared_amortized_bytes_per_expert": shared / 256,
        "abc_metadata_bytes": abc,
        "selector_metadata_bytes_per_expert": metadata,
        "selector_metadata_bpw": 8 * metadata / cfg["expert_weights"],
        "selector_compute_macs": macs,
        "selector_abc_score_units": abc_score_units,
        "selector_abc_score_compute_macs": abc_score_macs,
        "selector_abc_score_macs_per_unit": abc_score_macs_per_unit,
        "selector_abc_score_mac_convention": ABC_SCORE_MAC_CONVENTION,
        "selector_compute_additions": additions,
        "selector_scale_multiplications": scales,
        "selector_bytes_read": selector_bytes + (physical_bytes if fetched == 256 else 0),
        "selector_bytes_read_semantics": (
            "logical_unique_payload_plus_LUT_reads_not_measured_DRAM_traffic"
        ),
        "selector_activation_payload_already_read": True,
        "selector_activation_payload_bytes_read": 4_096,
        "selector_q2_unit_feature_bytes_read": 3_072,
        "selector_abc_metadata_bytes_read": abc,
        "suffix_storage_bpw": 2.0,
        "storage_multiplier": (4.25 + 2.0 + 8 * metadata / cfg["expert_weights"]) / 4.25,
    }


def selector_row(*, direct: bool = False) -> dict:
    return {
        **identity(),
        "selector_config_id": "direct" if direct else pq_entry()["config_id"],
        "selector_family": "direct_set_predictor" if direct else "block_pq_residual_synopsis",
        "selection_regime": "deployable_h0_direct_set_predictor" if direct else "deployable_h0_synopsis_direct_score",
        "applied_units": 192,
        "recovery": 0.9,
        **base_accounting(fetched=192, applied=192, direct=direct),
    }


def candidate_row(
    semantics: str, *, direct: bool = False, payloads_already_read: bool = True,
    q2_gate_up_scales_already_read: bool = True,
) -> dict:
    candidates = 256
    proxy_rank = 4
    if semantics == PREDICTED:
        q4_units, workspace_bytes, gram_units, lower_bound = 0, 0, 0, False
        regime, role = "deployable_predicted_application", "primary_predictor_output"
    elif semantics == INDEPENDENT:
        q4_units, workspace_bytes, gram_units, lower_bound = candidates, 0, 0, False
        regime, role = "deployable_after_candidate_fetch", "primary_realistic_h0_rerank"
    elif semantics == CONTAINED:
        q4_units = gram_units = candidates
        workspace_bytes, lower_bound = candidates * 2048 * 2, True
        regime, role = "information_deployable_but_expensive_interaction_oracle", "systems_oracle_not_promotion_primary"
    else:
        q4_units = gram_units = 512
        workspace_bytes, lower_bound = 512 * 2048 * 2, True
        regime, role = "teacher_oracle_not_deployable", "teacher_information_fixed_greedy_control"
    has_interaction = gram_units > 0
    is_independent = semantics == INDEPENDENT
    q4_macs = 2 * q4_units * 2048
    down_macs = 2 * gram_units * 2048
    euclidean_macs = gram_units * gram_units * 2048
    projection_macs = gram_units * 2048 * (proxy_rank if has_interaction else 0)
    proxy_gram_macs = gram_units * gram_units * (proxy_rank if has_interaction else 0)
    gram_macs = euclidean_macs + projection_macs + proxy_gram_macs
    abc_macs_per_unit = 6 if is_independent else 0
    incremental_macs = q4_macs + abc_macs_per_unit * q4_units + down_macs + gram_macs
    packet_bytes = q4_units * 1536
    proxy_bytes = 2048 * proxy_rank * 4 if has_interaction else 0
    activation_bytes = 0 if (not q4_units or payloads_already_read) else 4_096
    q2_bytes = 0 if (not q4_units or payloads_already_read) else 3_072
    abc_bytes = 0 if (not is_independent or payloads_already_read) else 3_084
    q2_parent_code_bytes = q4_units * 1_024
    q2_scale_bytes = 0 if q2_gate_up_scales_already_read else q4_units * 128
    q4_scale_multiplications = q4_units * 128
    incremental_bytes = (
        packet_bytes + activation_bytes + q2_bytes + abc_bytes
        + q2_parent_code_bytes + q2_scale_bytes
        + workspace_bytes + gram_units * gram_units * 4 + proxy_bytes
    )
    accounting = base_accounting(fetched=256, applied=192, direct=direct)
    total_macs = accounting["selector_compute_macs"] + incremental_macs
    total_bytes = accounting["selector_bytes_read"] + max(
        incremental_bytes - candidates * 1536, 0,
    )
    return {
        **identity(),
        "selector_config_id": "direct" if direct else pq_entry()["config_id"],
        "selector_family": "direct_set_predictor" if direct else "block_pq_residual_synopsis",
        "candidate_units": candidates,
        "applied_units": 192,
        "rerank_semantics": semantics,
        "rerank_role": role,
        "selection_regime": regime,
        "recovery": 0.95,
        "set_gain": 1.0,
        "oracle_set_gain": 1.0 / 0.98,
        "oracle_set_gain_retention": 0.98,
        "candidate_unit_ids": json.dumps(list(range(candidates))),
        "selected_units": json.dumps(list(range(192))),
        "oracle_selected_units": json.dumps(list(range(256, 448))),
        "activation_payload_already_read": payloads_already_read,
        "q2_payload_already_read": payloads_already_read,
        "abc_payload_already_read": payloads_already_read,
        "rerank_activation_payload_bytes_read": activation_bytes,
        "rerank_q2_unit_feature_bytes_read": q2_bytes,
        "rerank_abc_metadata_bytes_read": abc_bytes,
        "rerank_candidate_packet_bytes_read": packet_bytes,
        "rerank_candidate_packet_contents": CANDIDATE_PACKET_CONTENTS,
        "rerank_q2_gate_up_parent_code_bytes_read": q2_parent_code_bytes,
        "rerank_q2_gate_up_scale_bytes_read": q2_scale_bytes,
        "rerank_q2_gate_up_scale_payload_already_read": q2_gate_up_scales_already_read,
        "rerank_q2_gate_up_parent_payload_required": q4_units > 0,
        "rerank_q2_gate_up_parent_payload_accounted": True,
        "rerank_incremental_compute_macs": incremental_macs,
        "rerank_incremental_bytes_read": incremental_bytes,
        "rerank_packet_bytes_already_in_selector_bytes_read": True,
        "rerank_q4_response_units": q4_units,
        "rerank_q4_response_compute_macs": q4_macs,
        "rerank_q4_response_scale_multiplications": q4_scale_multiplications,
        "rerank_down_correction_compute_macs": down_macs,
        "rerank_correction_workspace_bytes": workspace_bytes,
        "rerank_interaction_gram_units": gram_units,
        "rerank_interaction_gram_euclidean_compute_macs": euclidean_macs,
        "rerank_interaction_proxy_rank": proxy_rank if has_interaction else 0,
        "rerank_interaction_proxy_projection_compute_macs": projection_macs,
        "rerank_interaction_proxy_gram_compute_macs": proxy_gram_macs,
        "rerank_interaction_proxy_scale_multiplications": gram_units * gram_units,
        "rerank_proxy_metadata_bytes_read": proxy_bytes,
        "rerank_interaction_gram_compute_macs": gram_macs,
        "rerank_abc_score_macs_per_unit": abc_macs_per_unit,
        "rerank_abc_score_mac_convention": ABC_SCORE_MAC_CONVENTION,
        "rerank_non_mac_operations": RERANK_NON_MAC_CONVENTION,
        "rerank_resident_q2_down_payload_required": has_interaction,
        "rerank_resident_q2_down_payload_accounted": not has_interaction,
        "rerank_accounting_is_lower_bound": lower_bound,
        "rerank_accounting_bound": (
            INTERACTION_RERANK_ACCOUNTING_BOUND
            if lower_bound else "complete_under_declared_logical_payload_and_MAC_conventions"
        ),
        "rerank_unaccounted_overhead": (
            INTERACTION_RERANK_UNACCOUNTED
            if lower_bound else "none_under_declared_logical_payload_and_MAC_conventions"
        ),
        **accounting,
        "total_selector_compute_macs": total_macs,
        "total_selector_bytes_read": total_bytes,
    }


def direct_entry() -> dict:
    unit_input, hidden = 8, 24
    return {
        "config_id": "direct",
        "selector_family": "direct_set_predictor",
        "experts": [7, 9],
        "q2_unit_dim": 3,
        "pq_delta_dim": 2,
        "abc_dim": 3,
        "hidden_dim": hidden,
        "head_mode": "single",
        "pq_runtime_response_precision": "fp16_round_then_fp32_reference",
        "linear_macs": 512 * unit_input * hidden + unit_input * hidden + 512 * hidden * hidden + 512 * hidden,
        "context_additions": unit_input * 511,
        "model_parameter_bytes_fp16": 10_000,
        "normalization_bytes_fp16": 100,
        "layer_shared_bytes": 10_100,
        "abc_bytes": 3_084,
        "pq_config_by_expert": {"7": pq_entry()["config_id"], "9": pq_entry()["config_id"]},
    }


def test_contract_and_whole_file_loader_refuse_test_and_duplicate_rows(tmp_path: Path) -> None:
    validate_config_contract(config())
    row = {
        **identity(), "selector_family": "coherent_exact_set_fixed_greedy_teacher",
        "selection_regime": "h0_exact_oracle", "physical_budget_bpw": 1.0,
        "physical_pages": 768, "physical_bytes": 393_216, "physical_bpw": 1.0,
        "page_amplification": 1.0, "recovery": 0.95,
    }
    path = tmp_path / "hybrid.parquet"
    pd.DataFrame([row]).to_parquet(path, index=False)
    assert len(load_validation_frame(path, "hybrid")) == 1
    leaked = {**row, "evaluation_split": "test", "request_id": "test"}
    pd.DataFrame([row, leaked]).to_parquet(path, index=False)
    with pytest.raises(RuntimeError, match="non-validation"):
        load_validation_frame(path, "hybrid")
    pd.DataFrame([row, row]).to_parquet(path, index=False)
    with pytest.raises(RuntimeError, match="duplicate scientific"):
        load_validation_frame(path, "hybrid")


def test_locked_scope_refuses_self_consistent_reduced_plan_and_missing_expert_cell() -> None:
    complete = pd.DataFrame(locked_validation_identities())
    frames = {kind: complete for kind in ("hybrid", "template", "selector", "candidate")}
    with pytest.raises(RuntimeError, match="exactly 69"):
        _validate_locked_validation_scope(
            locked_scope_config(),
            {"expected_unique_invocations": 68, "observed_unique_invocations": 68},
            frames,
        )

    missing = pd.DataFrame(locked_validation_identities(omit_last_cell=True))
    missing_frames = {
        kind: missing for kind in ("hybrid", "template", "selector", "candidate")
    }
    with pytest.raises(RuntimeError, match="all 12 locked"):
        _validate_locked_validation_scope(
            locked_scope_config(),
            {"expected_unique_invocations": 69, "observed_unique_invocations": 69},
            missing_frames,
        )


def test_hybrid_grid_refuses_wholly_missing_family_budget_combination() -> None:
    identities = locked_validation_identities()
    hybrid = pd.DataFrame([
        {
            **row,
            "selector_family": family,
            "physical_budget_bpw": budget,
        }
        for row in identities
        for family in HYBRID_FAMILIES
        for budget in LOCKED_HYBRID_BUDGETS
    ])
    by_cell = {
        cell: {
            tuple(row[column] for column in (
                "capture_source", "evaluation_split", "request_id", "position", "layer", "expert_id",
            ))
            for row in identities if (row["layer"], row["expert_id"]) == cell
        }
        for cell in {
            (row["layer"], row["expert_id"]) for row in identities
        }
    }
    assert _validate_hybrid_grid(locked_scope_config(), hybrid, by_cell)["complete"] is True
    missing = hybrid[
        ~(
            (hybrid["selector_family"] == "factorized_fixed_greedy_pr9_control")
            & np.isclose(hybrid["physical_budget_bpw"], 0.75)
        )
    ]
    with pytest.raises(RuntimeError, match="hybrid family/budget grid"):
        _validate_hybrid_grid(locked_scope_config(), missing, by_cell)


def test_template_grid_refuses_wholly_missing_selector_repair_combination() -> None:
    cfg = locked_scope_config()
    identities = locked_validation_identities()
    identity_columns = (
        "capture_source", "evaluation_split", "request_id", "position", "layer", "expert_id",
    )
    by_cell = {
        cell: {
            tuple(row[column] for column in identity_columns)
            for row in identities if (row["layer"], row["expert_id"]) == cell
        }
        for cell in {(row["layer"], row["expert_id"]) for row in identities}
    }
    template = pd.DataFrame([
        {
            **row,
            "template_cohort": cohort,
            "pairing_semantics": (
                "actual_routed_occurrences"
                if cohort == FIT_COHORTS[0]
                else "synthetic_all_x_by_expert_augmentation"
            ),
            "template_selector": selector,
            "template_count_requested": 8,
            "template_count_effective": 8,
            "repair_count": repair,
            "candidate_units": 256,
            "candidate_cap_units": 256,
            "candidate_cap_pass": True,
            "promotable": False,
        }
        for row in identities
        for cohort in FIT_COHORTS
        for selector in TEMPLATE_SELECTORS
        for repair in (0, 16, 32, 64)
    ])
    support = pd.DataFrame([
        {
            **row,
            "selector_config_id": (
                f"template:{cohort}:K8:{selector}:repair{repair}"
            ),
            "selector_family": "support_template",
            "candidate_units": 256,
            "candidate_cap_units": 256,
            "candidate_cap_pass": True,
            "promotable": False,
            "rerank_semantics": semantic,
        }
        for row in identities
        for cohort in FIT_COHORTS
        for selector in TEMPLATE_SELECTORS
        for repair in (0, 16, 32, 64)
        for semantic in (PREDICTED, INDEPENDENT, CONTAINED, FULL_TARGET)
    ])
    frames = {"template": template, "candidate": support}
    audit = _validate_template_grid(
        cfg, locked_template_manifest(), frames, by_cell,
    )
    assert audit["explicitly_nonpromotable"] is True
    zero_routed_manifest = locked_template_manifest()
    zero_routed_manifest["layers"]["0"]["template_cohort_example_counts_by_expert"]["7"][
        FIT_COHORTS[0]
    ] = 0
    zero_routed_manifest["layers"]["0"]["template_entries"] = [
        entry for entry in zero_routed_manifest["layers"]["0"]["template_entries"]
        if not (int(entry["expert_id"]) == 7 and entry["cohort"] == FIT_COHORTS[0])
    ]
    routed_cell = (
        (pd.to_numeric(template["layer"]) == 0)
        & (pd.to_numeric(template["expert_id"]) == 7)
        & (template["template_cohort"].astype(str) == FIT_COHORTS[0])
    )
    routed_support_cell = (
        (pd.to_numeric(support["layer"]) == 0)
        & (pd.to_numeric(support["expert_id"]) == 7)
        & support["selector_config_id"].astype(str).str.startswith(
            f"template:{FIT_COHORTS[0]}:"
        )
    )
    zero_audit = _validate_template_grid(
        cfg,
        zero_routed_manifest,
        {
            "template": template.loc[~routed_cell].copy(),
            "candidate": support.loc[~routed_support_cell].copy(),
        },
        by_cell,
    )
    assert zero_audit["complete"] is True
    with pytest.raises(RuntimeError, match="unrecognized or duplicate grid row"):
        _validate_template_grid(cfg, zero_routed_manifest, frames, by_cell)
    over_cap_template = template.copy()
    over_cap_support = support.copy()
    top2 = over_cap_template["template_selector"] == "top2_union_validation_oracle"
    support_top2 = over_cap_support["selector_config_id"].str.contains(
        "top2_union_validation_oracle"
    )
    over_cap_template.loc[top2, ["candidate_units", "candidate_cap_pass"]] = [300, False]
    over_cap_support.loc[
        support_top2, ["candidate_units", "candidate_cap_pass"]
    ] = [300, False]
    over_cap_audit = _validate_template_grid(
        cfg,
        locked_template_manifest(),
        {"template": over_cap_template, "candidate": over_cap_support},
        by_cell,
    )
    assert over_cap_audit["over_cap_rows_retained_as_nonpromotable_evidence"] > 0
    missing = template[
        ~(
            (template["template_selector"] == "top2_union_validation_oracle")
            & (template["repair_count"] == 64)
        )
    ]
    with pytest.raises(RuntimeError, match="template grid is incomplete"):
        _validate_template_grid(
            cfg, locked_template_manifest(), {"template": missing, "candidate": support}, by_cell,
        )


def test_selector_grid_refuses_wholly_missing_manifest_declared_config() -> None:
    cfg = locked_scope_config()
    cfg["pq_additive_stages"] = [1]
    identities = locked_validation_identities()
    hybrid = pd.DataFrame(identities)
    scope_frames = {
        kind: hybrid for kind in ("hybrid", "template", "selector", "candidate")
    }
    by_cell = _validate_locked_validation_scope(
        cfg,
        {"expected_unique_invocations": 69, "observed_unique_invocations": 69},
        scope_frames,
    )
    selector_index = {}
    for layer, expert in by_cell:
        selector_index[(layer, expert, pq_entry()["config_id"])] = {
            **pq_entry(), "expert_id": expert,
        }
    selector = pd.DataFrame([
        {
            **row,
            "selector_config_id": pq_entry()["config_id"],
            "selector_family": "block_pq_residual_synopsis",
        }
        for row in identities
    ])
    candidate = pd.DataFrame([
        {
            **row,
            "selector_config_id": pq_entry()["config_id"],
            "selector_family": "block_pq_residual_synopsis",
            "rerank_semantics": semantic,
        }
        for row in identities
        for semantic in (PREDICTED, INDEPENDENT, CONTAINED, FULL_TARGET)
    ])
    candidate = pd.concat([
        candidate,
        pd.DataFrame([{
            **identities[0],
            "selector_config_id": "template_oracle",
            "selector_family": "support_template",
            "rerank_semantics": PREDICTED,
        }]),
    ], ignore_index=True)
    audit = _validate_selector_grid_coverage(
        cfg,
        selector_index,
        {"selector": selector, "candidate": candidate},
        by_cell,
    )
    assert audit["complete"] is True
    for layer, expert in by_cell:
        selector_index[(layer, expert, "unconfigured_extra")] = {
            "config_id": "unconfigured_extra",
            "selector_family": "block_pq_residual_synopsis",
        }
    with pytest.raises(RuntimeError, match="contains extras"):
        _validate_selector_grid_coverage(
            cfg,
            selector_index,
            {"selector": selector, "candidate": candidate},
            by_cell,
        )


def test_frozen_selector_contract_explicitly_enumerates_pq_high_rank_and_direct_grid() -> None:
    frozen = json.loads(
        (EXPERIMENT / "configs/qwen36_mxfp4_set_utility_distillation.json").read_text()
    )
    contract = set(_configured_selector_contract(frozen))
    assert {
        ("block_pq_s1_separate_gate_up", "block_pq_residual_synopsis"),
        ("block_pq_s2_separate_gate_up", "block_pq_residual_synopsis"),
        ("high_rank_r64_fp8_e4m3fn_per_row", "high_rank_low_bit_linear_response"),
        ("high_rank_r64_int8_per_row", "high_rank_low_bit_linear_response"),
        ("high_rank_r128_int2", "high_rank_low_bit_linear_response"),
        ("high_rank_r128_ternary", "high_rank_low_bit_linear_response"),
        ("high_rank_r256_ternary", "high_rank_low_bit_linear_response"),
    }.issubset(contract)
    direct = {config_id for config_id, family in contract if family == "direct_set_predictor"}
    assert len(direct) == 10
    for cohort in frozen["direct_predictor_training_cohorts"]:
        assert sum(config_id.startswith(f"direct_{cohort}_") for config_id in direct) == 5


def test_separate_gate_up_pq_accounting_recomputes_factor_two_luts() -> None:
    frame = pd.DataFrame([selector_row()])
    index = {(0, 7, pq_entry()["config_id"]): pq_entry()}
    result = recompute_accounting("selector", frame, config(), index)
    assert result.loc[0, "selector_compute_macs_recomputed"] == 68_608
    assert result.loc[0, "selector_abc_score_compute_macs_recomputed"] == 3_072
    assert result.loc[0, "selector_scale_multiplications_recomputed"] == 65_536
    assert result.loc[0, "selector_bytes_read_recomputed"] == 501_772
    bad = frame.copy()
    bad.loc[0, "selector_compute_macs"] = 32_768
    with pytest.raises(RuntimeError, match="selector_compute_macs"):
        recompute_accounting("selector", bad, config(), index)


def test_shared_direct_entry_resolves_each_expert_and_linked_pq_cost() -> None:
    frame = pd.DataFrame([selector_row(direct=True)])
    index = {
        (0, 7, pq_entry()["config_id"]): pq_entry(),
        (0, 7, "direct"): direct_entry(),
    }
    result = recompute_accounting("selector", frame, config(), index)
    expected_model = direct_entry()["linear_macs"]
    assert result.loc[0, "selector_compute_macs_recomputed"] == expected_model + 65_536
    assert result.loc[0, "expert_specific_metadata_bytes"] == 32_768
    assert result.loc[0, "layer_shared_metadata_bytes"] == 141_172

    wrong_precision = dict(direct_entry())
    wrong_precision["pq_runtime_response_precision"] = "fp32_unrounded"
    with pytest.raises(RuntimeError, match="PQ runtime precision"):
        recompute_accounting(
            "selector",
            frame,
            config(),
            {
                (0, 7, pq_entry()["config_id"]): pq_entry(),
                (0, 7, "direct"): wrong_precision,
            },
        )


def test_direct_feature_traffic_uses_exact_serialized_abc_bytes_not_dimension() -> None:
    entry = dict(direct_entry())
    unit_input, hidden = 9, int(entry["hidden_dim"])
    entry.update({
        "abc_dim": 4,
        "abc_bytes": 4_108,
        "linear_macs": (
            512 * unit_input * hidden + unit_input * hidden
            + 512 * hidden * hidden + 512 * hidden
        ),
        "context_additions": unit_input * 511,
    })
    row = selector_row(direct=True)
    row["selector_compute_macs"] += 512 * hidden + hidden
    row["selector_compute_additions"] += 511
    row["selector_bytes_read"] += 2 * 512
    row["selector_abc_metadata_bytes_read"] += 1_024
    row["abc_metadata_bytes"] += 1_024
    row["selector_metadata_bytes_per_expert"] += 1_024
    row["selector_metadata_bpw"] = (
        8 * row["selector_metadata_bytes_per_expert"] / config()["expert_weights"]
    )
    row["storage_multiplier"] = (
        4.25 + 2.0 + row["selector_metadata_bpw"]
    ) / 4.25
    index = {
        (0, 7, pq_entry()["config_id"]): pq_entry(),
        (0, 7, "direct"): entry,
    }
    result = recompute_accounting("selector", pd.DataFrame([row]), config(), index)
    assert result.loc[0, "selector_bytes_read_recomputed"] == row["selector_bytes_read"]
    assert result.loc[0, "abc_metadata_bytes"] == 4_108


def test_summary_omits_all_nan_inapplicable_metric_but_rejects_partial_nan() -> None:
    rows = [
        {**identity(index), "selector_config_id": "direct", "recovery": 0.9, "response_mse": np.nan}
        for index in range(3)
    ]
    summary = summarize(
        pd.DataFrame(rows), ["selector_config_id"], ["recovery", "response_mse"],
    )
    assert summary.loc[0, "recovery_n"] == 3
    assert "response_mse_n" not in summary

    rows[0]["response_mse"] = 0.25
    with pytest.raises(RuntimeError, match="mixes finite and non-finite"):
        summarize(pd.DataFrame(rows), ["selector_config_id"], ["response_mse"])


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ({"promotable": False}, "explicit_promotable_false"),
        ({"control_only": True}, "explicit_control_only_true"),
        ({"control_role": "diagnostic_only"}, "control_role:diagnostic_only"),
        (
            {"sampled_expert_scope": "transductive_sampled_expert_diagnostic_not_bankwide"},
            "transductive_sampled_expert_diagnostic",
        ),
        (
            {"static_unit_bias_semantics": "train_only;transductive_sampled_expert_upper_bound"},
            "transductive_sampled_expert_upper_bound",
        ),
    ],
)
def test_explicit_control_only_manifest_markers_are_nonpromotable(
    entry: dict, expected: str,
) -> None:
    assert expected in _promotion_exclusion_reasons(entry)


def test_frozen_control_manifest_entries_require_nonpromotion_markers() -> None:
    with pytest.raises(RuntimeError, match="explicitly nonpromotable"):
        _validate_selector_promotion_contract(
            (0, 7, "high_rank_r64_int8_per_row"),
            {
                "selector_family": "high_rank_low_bit_linear_response",
                "sampled_expert_scope": "transductive_sampled_expert_diagnostic",
            },
        )
    with pytest.raises(RuntimeError, match="static-unit-bias"):
        _validate_selector_promotion_contract(
            (0, 7, "direct_x_q2_abc_static_unit_bias_control"),
            {
                "selector_family": "direct_set_predictor",
                "pq_delta_dim": 0,
                "pq_runtime_response_precision": "not_applicable",
                "static_unit_bias_semantics": "none",
            },
        )


def test_fit_manifest_indexes_shared_direct_selector_for_every_declared_expert(tmp_path: Path) -> None:
    cfg = config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg))
    shard = tmp_path / "set_utility_fit_layer_0.npz"
    shard.write_bytes(b"immutable fixture")
    sidecar_record = {
        "layer": 0,
        "shard": shard.name,
        "shard_sha256": sha256(shard),
        "shard_bytes": shard.stat().st_size,
        "array_count": 0,
        "fit_split": "train",
        "validation_or_test_rows_used": False,
        "runtime_provenance": runtime_provenance(),
        "selector_entries": [direct_entry()],
    }
    sidecar = tmp_path / "set_utility_fit_layer_0.json"
    sidecar.write_text(json.dumps(sidecar_record, sort_keys=True))
    manifest = {
        "run_id": "fixture", "config_sha256": sha256(config_path), "fit_split": "train",
        "validation_or_test_rows_used": False, "test_rows_admitted": False, "test_rows_used": False,
        "monolithic_source_contains_test_rows": True,
        "cross_capture_request_id_audit": cross_capture_request_id_audit(),
        "direct_candidate_head_supervision": JOINT_CANDIDATE_COVERAGE_SEMANTICS,
        "runtime_provenance": runtime_provenance(),
        "layers": {"0": {**sidecar_record, "sidecar_sha256": sha256(sidecar)}},
    }
    manifest_path = tmp_path / "set_utility_fit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    facts = {
        "completed": True, "failures": [], "fit_split": "train",
        "validation_or_test_rows_used": False, "test_rows_admitted": False, "test_rows_used": False,
        "monolithic_source_contains_test_rows": True,
        "request_separation_verified": True, "codec_locked": True,
        "cross_capture_request_id_audit": cross_capture_request_id_audit(),
        "direct_candidate_head_supervision": JOINT_CANDIDATE_COVERAGE_SEMANTICS,
        "runtime_provenance": runtime_provenance(),
        "config_sha256": sha256(config_path), "exact_capture_sha256": "a" * 64,
        "cross_capture_sha256": "b" * 64, "tree_sha256": "c" * 64,
        "checkpoint_config_sha256": "d" * 64, "checkpoint_index_sha256": "e" * 64,
        "manifest_sha256": sha256(manifest_path),
    }
    (tmp_path / "fit_facts.json").write_text(json.dumps(facts))
    _, _, selector_index = _validate_fit_bundle(cfg, config_path, tmp_path)
    assert selector_index[(0, 7, "direct")]["selector_family"] == "direct_set_predictor"
    assert selector_index[(0, 9, "direct")]["selector_family"] == "direct_set_predictor"


def test_four_candidate_interfaces_recompute_incremental_and_total_work() -> None:
    frame = pd.DataFrame([candidate_row(value) for value in (PREDICTED, INDEPENDENT, CONTAINED, FULL_TARGET)])
    index = {(0, 7, pq_entry()["config_id"]): pq_entry()}
    result = recompute_accounting("candidate", frame, config(), index).set_index("rerank_semantics")
    assert result.loc[INDEPENDENT, "rerank_incremental_compute_macs_recomputed"] == 1_050_112
    assert result.loc[INDEPENDENT, "total_selector_compute_macs"] == 1_118_720
    assert result.loc[INDEPENDENT, "rerank_q2_gate_up_parent_code_bytes_read_recomputed"] == 262_144
    assert result.loc[INDEPENDENT, "rerank_q2_gate_up_scale_bytes_read_recomputed"] == 0
    assert result.loc[INDEPENDENT, "rerank_q4_response_scale_multiplications_recomputed"] == 32_768
    assert result.loc[CONTAINED, "rerank_incremental_compute_macs_recomputed"] == 138_674_176
    assert result.loc[CONTAINED, "rerank_accounting_is_lower_bound"]
    assert result.loc[PREDICTED, "total_selector_compute_macs"] == 68_608
    assert result.loc[PREDICTED, "total_selector_bytes_read"] == result.loc[
        PREDICTED, "selector_bytes_read"
    ]
    assert result.loc[FULL_TARGET, "rerank_incremental_compute_macs_recomputed"] == (
        2 * 512 * 2048 + 2 * 512 * 2048
        + 512 * 512 * 2048 + 512 * 2048 * 4 + 512 * 512 * 4
    )
    assert result.loc[FULL_TARGET, "rerank_incremental_bytes_read_recomputed"] == (
        512 * 1536 + 512 * 1_024 + 512 * 2048 * 2
        + 512 * 512 * 4 + 2048 * 4 * 4
    )
    assert result.loc[FULL_TARGET, "rerank_correction_workspace_bytes_recomputed"] == (
        512 * 2048 * 2
    )
    assert result.loc[FULL_TARGET, "rerank_accounting_is_lower_bound"]
    bad = frame.copy()
    bad.loc[bad["rerank_semantics"] == FULL_TARGET, "rerank_q4_response_units"] = 256
    with pytest.raises(RuntimeError, match="rerank_q4_response_units"):
        recompute_accounting("candidate", bad, config(), index)


def test_rerank_conditionally_reads_activation_q2_and_exact_abc_payloads() -> None:
    frame = pd.DataFrame([
        candidate_row(value, payloads_already_read=False)
        for value in (PREDICTED, INDEPENDENT, CONTAINED, FULL_TARGET)
    ])
    arrays = _validate_candidate_rerank_accounting(frame)
    by_semantics = {
        semantic: position
        for position, semantic in enumerate(frame["rerank_semantics"].astype(str))
    }
    predicted = by_semantics[PREDICTED]
    independent = by_semantics[INDEPENDENT]
    contained = by_semantics[CONTAINED]
    assert arrays["activation_bytes"][predicted] == 0
    assert arrays["q2_bytes"][predicted] == 0
    assert arrays["abc_bytes"][predicted] == 0
    assert arrays["activation_bytes"][independent] == 4_096
    assert arrays["q2_bytes"][independent] == 3_072
    assert arrays["abc_bytes"][independent] == 3_084
    assert arrays["activation_bytes"][contained] == 4_096
    assert arrays["q2_bytes"][contained] == 3_072
    assert arrays["abc_bytes"][contained] == 0


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("candidate_unit_ids", json.dumps(list(range(255)) + [511]), "different candidates"),
        ("candidate_unit_ids", json.dumps([0] * 256), "duplicate units"),
        ("selected_units", json.dumps(list(range(191)) + [511]), "not a subset"),
        ("oracle_selected_units", json.dumps(list(range(191))), "cardinality"),
        ("oracle_set_gain", 123.0, "different oracle gains"),
        ("rerank_q2_gate_up_parent_code_bytes_read", 0, "parent_code_bytes"),
        ("rerank_q2_gate_up_scale_bytes_read", 1, "scale_bytes"),
        ("rerank_q4_response_scale_multiplications", 0, "scale_multiplications"),
        ("rerank_q2_gate_up_parent_payload_accounted", False, "parent_payload_accounted"),
    ],
)
def test_candidate_support_and_oracle_evidence_fail_closed(
    field: str, replacement: object, message: str,
) -> None:
    frame = pd.DataFrame([
        candidate_row(value) for value in (PREDICTED, INDEPENDENT, CONTAINED, FULL_TARGET)
    ])
    target = frame["rerank_semantics"] == CONTAINED
    frame.loc[target, field] = replacement
    if field == "oracle_set_gain":
        frame.loc[target, "set_gain"] = float(replacement) * 0.98
    with pytest.raises(RuntimeError, match=message):
        _validate_candidate_rerank_accounting(frame)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("candidate_units", 255, "fetch exactly 256"),
        ("applied_units", 191, "apply exactly 192"),
    ],
)
def test_candidate_interface_refuses_nonexact_fetched_or_applied_counts(
    field: str, value: int, message: str,
) -> None:
    frame = pd.DataFrame([
        candidate_row(semantic)
        for semantic in (PREDICTED, INDEPENDENT, CONTAINED, FULL_TARGET)
    ])
    frame.loc[frame["rerank_semantics"] == PREDICTED, field] = value
    with pytest.raises(RuntimeError, match=message):
        _validate_candidate_rerank_accounting(frame, config())


def test_full_target_matches_unrestricted_oracle_when_oracle_is_fetched() -> None:
    frame = pd.DataFrame([
        candidate_row(value) for value in (PREDICTED, INDEPENDENT, CONTAINED, FULL_TARGET)
    ])
    oracle = json.dumps(list(range(192)))
    frame["oracle_selected_units"] = oracle
    frame["oracle_set_gain"] = 1.0
    frame["oracle_set_gain_retention"] = 1.0
    _validate_candidate_rerank_accounting(frame)
    bad = frame.copy()
    bad.loc[bad["rerank_semantics"] == FULL_TARGET, "selected_units"] = json.dumps(
        list(range(64, 256)),
    )
    with pytest.raises(RuntimeError, match="unrestricted oracle path"):
        _validate_candidate_rerank_accounting(bad)


def test_rerank_charges_q2_gate_up_scales_when_selector_did_not_read_them() -> None:
    frame = pd.DataFrame([
        candidate_row(semantic, q2_gate_up_scales_already_read=False)
        for semantic in (PREDICTED, INDEPENDENT, CONTAINED, FULL_TARGET)
    ])
    arrays = _validate_candidate_rerank_accounting(frame, config())
    semantics = list(frame["rerank_semantics"].astype(str))
    assert arrays["q2_scale_bytes"][semantics.index(PREDICTED)] == 0
    assert arrays["q2_scale_bytes"][semantics.index(INDEPENDENT)] == 256 * 128
    assert arrays["q2_scale_bytes"][semantics.index(FULL_TARGET)] == 512 * 128


def promotion_fixture() -> tuple[ValidatedEvidence, dict[str, pd.DataFrame]]:
    ids = [identity(index) for index in range(10)]
    hybrid = []
    template = []
    candidate = []
    for row in ids:
        hybrid.extend([
            {**row, "selector_family": "coherent_independent_abc_pr9_control", "physical_budget_bpw": 1.0, "recovery": 0.938},
            {**row, "selector_family": "coherent_exact_set_fixed_greedy_teacher", "physical_budget_bpw": 1.0, "recovery": 0.94},
            {**row, "selector_family": "hybrid_forward_plus_1_2_3_unit_local_search", "physical_budget_bpw": 1.0, "recovery": 0.946},
        ])
        template.append({
            **row, "template_cohort": "routed_exact_train_primary",
            "pairing_semantics": "actual_routed_occurrences",
            "template_selector": "top1_template_validation_oracle",
            "template_count_requested": 8, "template_count_effective": 8, "repair_count": 0,
            "candidate_units": 192, "candidate_overfetch": 1.0,
            "validation_oracle_path_utility_retained": 0.96,
        })
        for semantics in (PREDICTED, INDEPENDENT, CONTAINED, FULL_TARGET):
            if semantics == PREDICTED:
                recovery, gain, total, regime, role = 0.90, 0.90, 68_608, "deployable_predicted_application", "primary_predictor_output"
            elif semantics == INDEPENDENT:
                recovery, gain, total, regime, role = 0.95, 0.98, 1_118_720, "deployable_after_candidate_fetch", "primary_realistic_h0_rerank"
            elif semantics == CONTAINED:
                recovery, gain, total, regime, role = 0.999, 1.01, 138_742_784, "information_deployable_but_expensive_interaction_oracle", "systems_oracle_not_promotion_primary"
            else:
                recovery, gain, total, regime, role = 0.999, 1.02, 546_376_704, "teacher_oracle_not_deployable", "teacher_information_fixed_greedy_control"
            candidate.append({
                **row, "selector_config_id": "pq", "selector_family": "block_pq_residual_synopsis",
                "rerank_semantics": semantics, "selection_regime": regime, "rerank_role": role,
                "recovery": recovery, "oracle_set_gain_retention": gain,
                "selector_metadata_bpw": 0.1, "selector_compute_macs": 68_608,
                "rerank_incremental_compute_macs": total - 68_608,
                "total_selector_compute_macs": total, "storage_multiplier": 1.5,
            })
    frames = {
        "hybrid": pd.DataFrame(hybrid),
        "template": pd.DataFrame(template),
        "selector": pd.DataFrame([{**row, "selector_config_id": "pq", "selector_family": "block_pq_residual_synopsis", "selection_regime": "deployable_h0_synopsis_direct_score"} for row in ids]),
        "candidate": pd.DataFrame(candidate),
    }
    evidence = ValidatedEvidence(
        config=config(), config_sha256="f" * 64, fit_facts={}, fit_manifest={},
        run_facts={"expected_unique_invocations": 10, "observed_unique_invocations": 10, "fit_manifest_sha256": "0" * 64},
        selector_index={}, frames=frames, evidence_sha256={"candidate": "1" * 64},
    )
    return evidence, frames


def test_promotion_uses_predicted_or_independent_not_expensive_contained_control() -> None:
    evidence, frames = promotion_fixture()
    payload, tables = freeze_validation_promotions(evidence, frames)
    selected = payload["h0_late_candidate_policy"]["selected_validation_configuration"]
    assert selected["rerank_semantics"] == INDEPENDENT
    assert payload["contained_interaction_systems_oracle_used_for_promotion"] is False
    assert payload["activation_timing"].startswith("actual_x")
    assert tables["hybrid_gate"].iloc[0]["passes"]
    assert not set(tables["hybrid_gate"]["selector_family"]) & {
        "coherent_independent_abc_pr9_control",
        "coherent_exact_set_fixed_greedy_teacher",
    }
    assert tables["hybrid_gate"].iloc[0]["paired_delta_vs_pr9_median"] == pytest.approx(0.008)
    assert tables["hybrid_gate"].iloc[0]["paired_delta_vs_exact_set_median"] == pytest.approx(0.006)


def test_over_cap_top2_template_union_is_reported_but_never_passes() -> None:
    evidence, frames = promotion_fixture()
    template = frames["template"].copy()
    template["template_selector"] = "top2_union_validation_oracle"
    template["candidate_units"] = 300
    template["candidate_overfetch"] = 300 / 192
    template["validation_oracle_path_utility_retained"] = 0.999
    payload, tables = freeze_validation_promotions(
        evidence, {**frames, "template": template},
    )
    assert not tables["template_gate"]["passes"].any()
    assert payload["template_existence_oracle"]["status"] == "negative"
    assert payload["template_existence_oracle"]["deployable_selector_promotion_eligible"] is False
    assert payload["support_template_oracle_used_for_selector_promotion"] is False


def test_compute_gate_is_strict_and_includes_incremental_rerank_macs() -> None:
    evidence, frames = promotion_fixture()
    candidate = frames["candidate"].copy()
    independent = candidate["rerank_semantics"] == INDEPENDENT
    candidate.loc[independent, "total_selector_compute_macs"] = 1_572_864
    candidate.loc[independent, "rerank_incremental_compute_macs"] = 1_572_864 - 68_608
    frames = {**frames, "candidate": candidate}
    payload, gates = freeze_validation_promotions(evidence, frames)
    assert payload["h0_late_candidate_policy"]["status"] == "stop"
    independent_gate = gates["selector_gate"][gates["selector_gate"]["rerank_semantics"] == INDEPENDENT]
    assert not bool(independent_gate.iloc[0]["passes_compute"])


def test_transductive_static_bias_control_is_reported_but_never_promoted() -> None:
    evidence, frames = promotion_fixture()
    candidate = frames["candidate"].copy()
    candidate["selector_config_id"] = "transductive_static_bias"
    candidate["selector_family"] = "direct_set_predictor"
    frames = {**frames, "candidate": candidate}
    entry = {
        "config_id": "transductive_static_bias",
        "selector_family": "direct_set_predictor",
        "static_unit_bias_semantics": (
            "train_only_per_expert_frequency;transductive_sampled_expert_upper_bound"
        ),
    }
    evidence = ValidatedEvidence(
        config=evidence.config,
        config_sha256=evidence.config_sha256,
        fit_facts=evidence.fit_facts,
        fit_manifest=evidence.fit_manifest,
        run_facts=evidence.run_facts,
        selector_index={(0, 7, "transductive_static_bias"): entry},
        frames=frames,
        evidence_sha256=evidence.evidence_sha256,
    )
    payload, gates = freeze_validation_promotions(evidence, frames)
    assert payload["h0_late_candidate_policy"]["status"] == "stop"
    control = gates["selector_gate"]
    assert len(control) == 2
    assert not control["promotion_eligible_selector"].any()
    assert control["promotion_exclusion_reason"].str.contains(
        "transductive_sampled_expert_upper_bound",
    ).all()
    assert not control["passes"].any()


def test_transductive_high_rank_diagnostic_is_reported_but_never_promoted() -> None:
    evidence, frames = promotion_fixture()
    candidate = frames["candidate"].copy()
    candidate["selector_config_id"] = "high_rank_control"
    candidate["selector_family"] = "high_rank_low_bit_linear_response"
    frames = {**frames, "candidate": candidate}
    entry = {
        "config_id": "high_rank_control",
        "selector_family": "high_rank_low_bit_linear_response",
        "promotable": False,
        "sampled_expert_scope": (
            "transductive_sampled_expert_diagnostic_not_bankwide_generalization"
        ),
    }
    evidence = ValidatedEvidence(
        config=evidence.config,
        config_sha256=evidence.config_sha256,
        fit_facts=evidence.fit_facts,
        fit_manifest=evidence.fit_manifest,
        run_facts=evidence.run_facts,
        selector_index={(0, 7, "high_rank_control"): entry},
        frames=frames,
        evidence_sha256=evidence.evidence_sha256,
    )
    payload, gates = freeze_validation_promotions(evidence, frames)
    assert payload["h0_late_candidate_policy"]["status"] == "stop"
    assert payload["high_rank_low_bit_diagnostic_used_for_promotion"] is False
    control = gates["selector_gate"]
    assert not control["promotion_eligible_selector"].any()
    assert control["promotion_exclusion_reason"].str.contains(
        "transductive_sampled_expert_diagnostic",
    ).all()
    assert control["promotion_exclusion_reason"].str.contains(
        "explicit_promotable_false",
    ).all()


def test_high_rank_manifest_accounting_charges_both_output_scale_vectors() -> None:
    frame = pd.DataFrame([{
        "layer": 0,
        "expert_id": 7,
        "selector_config_id": "high_rank_control",
        "selector_family": "high_rank_low_bit_linear_response",
    }])
    entry = {
        "config_id": "high_rank_control",
        "selector_family": "high_rank_low_bit_linear_response",
        "rank": 64,
        "expert_specific_bytes": 65_536,
        "layer_shared_bytes": 262_144,
        "abc_bytes": 3_084,
    }
    _, _, macs, additions, scales, bytes_read, abc, activation_read = _manifest_accounting(
        frame,
        {(0, 7, "high_rank_control"): entry},
        config(),
    )
    assert macs.tolist() == [64 * (2048 + 2 * 512) + 3_072]
    assert additions.tolist() == [0]
    assert scales.tolist() == [1_024]
    assert bytes_read.tolist() == [262_144 + 65_536 + 4_096 + 3_072 + 3_084]
    assert abc.tolist() == [3_084]
    assert activation_read.astype(bool).tolist() == [True]


def test_save_figure_normalizes_svg_and_report_states_no_h4_or_global_optimum(tmp_path: Path) -> None:
    fig, axis = analyzer.plt.subplots()
    axis.plot([0, 1], [1, 0])
    _, svg = analyzer.save_figure(fig, tmp_path, "fixture")
    assert all(line == line.rstrip() for line in svg.read_text().splitlines())
    source = SCRIPT.read_text()
    assert "not a global" in source
    assert "not H4 prefetch" in source
    assert "138.7M" in source
    assert "four non-interchangeable interfaces" in source
    assert "4,108" in source
    assert "candidate_head_supervision" not in source
    assert "supplies and optimizes no" in source
    assert "logical unique payload" in source
    assert "both charge the 1,024 output-scale" in source
    assert "exactly 69 unique validation" in source
    assert "No test scientific tensor or value is admitted" in source
    assert "`split`" in source and "`request_id`" in source
    assert "resident Q2-down parent-code/scale payload" in source
    assert "candidate_rerank_accounting.csv" in source
    assert "The 14 generated CSVs" in source
    assert "stale `promotable=true`" not in source
