"""Strict validation-only analysis for coherent-unit set-utility studies.

This module is intentionally independent of the experiment runner.  It reads
the runner's immutable facts and evidence, recomputes all deterministic
systems arithmetic, and freezes promotion decisions without admitting a test
row.  In particular, it keeps four candidate interfaces distinct:

* predicted application with no reranking;
* exact independent A/B/C reranking within fetched candidates;
* contained-target interaction reranking, information-feasible from fetched pages but a roughly 138.7M-MAC H0 systems oracle;
* full-target restricted reranking (nondeployable teacher-information control).

Every exact path here is ``exact_marginal_fixed_greedy``: coefficients are
fixed and there is no least-squares refit. It is not a globally optimal top-k
solver. Consequently, a restricted path may occasionally beat the
unrestricted fixed-greedy path through path dependence; reported gain ratios
are intentionally not clipped at one.

The support-template results are existence oracles.  They are never relabelled
as a deployable classifier, and routed training occurrences remain separate
from synthetic all-activation/expert pairings.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


IDENTITY = (
    "capture_source", "evaluation_split", "request_id", "position", "layer", "expert_id",
)
ARTIFACT_FILES = {
    "hybrid": "hybrid_oracle_frontier.parquet",
    "template": "support_template_frontier.parquet",
    "selector": "pq_high_rank_selector_frontier.parquet",
    "candidate": "candidate_set_rerank_frontier.parquet",
}
SCIENTIFIC_KEYS = {
    "hybrid": (*IDENTITY, "selector_family", "physical_budget_bpw"),
    "template": (
        *IDENTITY, "template_cohort", "pairing_semantics", "template_selector",
        "template_count_requested", "template_count_effective", "repair_count",
    ),
    "selector": (*IDENTITY, "selector_config_id", "selector_family", "selection_regime"),
    "candidate": (
        *IDENTITY, "selector_config_id", "selector_family", "candidate_units",
        "applied_units", "rerank_semantics", "selection_regime",
    ),
}
REQUIRED_COLUMNS = {
    "hybrid": {
        *IDENTITY, "selector_family", "selection_regime", "physical_budget_bpw",
        "physical_pages", "physical_bytes", "physical_bpw", "page_amplification", "recovery",
    },
    "template": {
        *IDENTITY, "template_cohort", "pairing_semantics", "template_selector",
        "template_count_requested", "template_count_effective", "candidate_units",
        "physical_pages", "physical_bytes", "physical_bpw", "candidate_overfetch",
        "validation_oracle_path_utility_retained", "selection_regime",
    },
    "selector": {
        *IDENTITY, "selector_config_id", "selector_family", "selection_regime",
        "applied_units", "recovery", "physical_pages", "physical_bytes", "physical_bpw",
        "logical_actions", "logical_bytes", "logical_bpw", "page_amplification",
        "expert_specific_metadata_bytes", "layer_shared_metadata_bytes",
        "layer_shared_amortized_bytes_per_expert", "abc_metadata_bytes",
        "selector_metadata_bytes_per_expert", "selector_metadata_bpw",
        "selector_compute_macs", "selector_compute_additions", "selector_scale_multiplications",
        "selector_abc_score_units", "selector_abc_score_compute_macs",
        "selector_abc_score_macs_per_unit", "selector_abc_score_mac_convention",
        "selector_bytes_read", "selector_activation_payload_already_read",
        "selector_bytes_read_semantics",
        "selector_activation_payload_bytes_read", "selector_q2_unit_feature_bytes_read",
        "selector_abc_metadata_bytes_read",
        "suffix_storage_bpw", "storage_multiplier",
    },
    "candidate": {
        *IDENTITY, "selector_config_id", "selector_family", "candidate_units",
        "applied_units", "rerank_semantics", "selection_regime", "recovery", "set_gain",
        "oracle_set_gain_retention", "rerank_role", "rerank_incremental_compute_macs",
        "rerank_incremental_bytes_read", "rerank_packet_bytes_already_in_selector_bytes_read",
        "candidate_unit_ids", "selected_units", "oracle_selected_units", "oracle_set_gain",
        "activation_payload_already_read", "q2_payload_already_read",
        "abc_payload_already_read", "rerank_activation_payload_bytes_read",
        "rerank_q2_unit_feature_bytes_read", "rerank_abc_metadata_bytes_read",
        "rerank_candidate_packet_bytes_read",
        "rerank_q2_gate_up_parent_code_bytes_read",
        "rerank_q2_gate_up_scale_bytes_read",
        "rerank_q2_gate_up_scale_payload_already_read",
        "rerank_q2_gate_up_parent_payload_required",
        "rerank_q2_gate_up_parent_payload_accounted",
        "rerank_q4_response_units", "rerank_q4_response_compute_macs",
        "rerank_q4_response_scale_multiplications",
        "rerank_correction_workspace_bytes", "rerank_down_correction_compute_macs",
        "rerank_interaction_gram_units", "rerank_interaction_gram_compute_macs",
        "rerank_interaction_gram_euclidean_compute_macs",
        "rerank_interaction_proxy_rank", "rerank_interaction_proxy_projection_compute_macs",
        "rerank_interaction_proxy_gram_compute_macs",
        "rerank_interaction_proxy_scale_multiplications", "rerank_proxy_metadata_bytes_read",
        "rerank_abc_score_macs_per_unit", "rerank_abc_score_mac_convention",
        "rerank_non_mac_operations", "rerank_candidate_packet_contents",
        "rerank_resident_q2_down_payload_required",
        "rerank_resident_q2_down_payload_accounted", "rerank_accounting_is_lower_bound",
        "rerank_accounting_bound", "rerank_unaccounted_overhead",
        "selector_compute_macs", "selector_bytes_read",
        "total_selector_compute_macs", "total_selector_bytes_read",
        "physical_pages", "physical_bytes", "physical_bpw",
        "logical_actions", "logical_bytes", "logical_bpw", "page_amplification",
    },
}

UNITS = 512
INPUTS = 2048
FP16_BYTES = 2
Q2_UNIT_FEATURE_BYTES = 3 * UNITS * FP16_BYTES
ACTIVATION_PAYLOAD_BYTES = INPUTS * FP16_BYTES
UNIT_PACKET_BYTES = 3 * 512
Q2_GATE_UP_PARENT_CODE_BYTES_PER_UNIT = 1_024
Q2_GATE_UP_SCALE_BYTES_PER_UNIT = 128
Q4_RESPONSE_SCALE_MULTIPLICATIONS_PER_UNIT = 128
LOCKED_PROXY_RANK = 4
SELECTOR_BYTE_SEMANTICS = (
    "logical_unique_payload_plus_LUT_reads_not_measured_DRAM_traffic"
)
SELECTOR_MAC_SEMANTICS = (
    "analytical_linear_algebraic_macs_excluding_nonlinear_silu_log_sqrt_topk_sort_"
    "control_flow_and_runtime_overhead"
)
ABC_SCORE_MACS_PER_UNIT = 6
ABC_SCORE_MAC_CONVENTION = (
    "six_scalar_multiplications_per_unit_for_"
    "A_h4_squared_minus_2B_h4_h2_plus_C_h2_squared;_minus_2_is_prefolded_"
    "into_B_without_changing_metadata_size;_two_scalar_additions_are_not_MACs"
)
RERANK_NON_MAC_CONVENTION = (
    "SiLU,_elementwise_hidden_products,_scalar_additions,_clamp,_deterministic_"
    "top-k/sort,_Q2-parent/suffix_table_decode,_E8M0_decode,_reported_Q4-response_"
    "block-scale_multiplications,_and_fixed-greedy_control_are_excluded_from_the_"
    "MAC_metric"
)
CANDIDATE_PACKET_CONTENTS = (
    "1536-byte/unit_locked_Q2-to-Q4_suffix_only:_gate/up=1024_bytes,_down=512_bytes;_"
    "excludes_resident_Q2_weights/scales,_ABC,_activation,_and_future-proxy_metadata"
)
INTERACTION_RERANK_ACCOUNTING_BOUND = (
    "precise_lower_bound_excludes_resident_Q2-down_parent-code/scale_payload,_"
    "Q2-down_tree/E8M0_decode_and_block-scale_application,_fixed-greedy_initial_"
    "correlation,_and_per-selection_vector_updates"
)
INTERACTION_RERANK_UNACCOUNTED = (
    "resident_Q2-down_parent-code/scale_payload,_tree/E8M0_decode,_and_block-scale_"
    "application;_fixed-greedy_initial_correlation_and_per-selection_vector_updates;_"
    "operations_listed_in_rerank_non_mac_operations"
)

PREDICTED = "predicted_direct_no_rerank"
INDEPENDENT = "exact_independent_abc_within_fetched_candidates"
CONTAINED = "contained_fetched_target_exact_h0"
FULL_TARGET = "full_target_restricted_teacher_oracle"
CANDIDATE_SEMANTICS = (PREDICTED, INDEPENDENT, CONTAINED, FULL_TARGET)
LOCKED_VALIDATION_INVOCATIONS = 69
LOCKED_VALIDATION_CELLS = 12
LOCKED_HYBRID_BUDGETS = (0.5, 0.75, 1.0)
HYBRID_FAMILIES = (
    "coherent_independent_abc_pr9_control",
    "coherent_exact_set_fixed_greedy_teacher",
    "factorized_fixed_greedy_pr9_control",
    "direct_GD_hybrid_forward",
    "hybrid_forward_plus_1_2_3_unit_local_search",
)
TEMPLATE_SELECTORS = (
    "top1_template_validation_oracle",
    "top2_union_validation_oracle",
)
LOCKED_TEMPLATE_REPAIRS = (0, 16, 32, 64)
FIT_COHORTS = (
    "routed_exact_train_primary",
    "exact_train_all_x_augmentation",
    "combined_exact_cross_train_all_x_augmentation",
)
JOINT_CANDIDATE_COVERAGE_SEMANTICS = (
    "hard_forward_top192_apply_head_union_top64_candidate_head_excluding_apply_"
    "supervised_for_teacher_top192_coverage"
)
RUNTIME_PROVENANCE_FIELDS = (
    "device", "gpu_names_used", "torch_version", "torch_cuda_runtime",
    "numpy_version", "python_version", "host",
)
TEST_METADATA_AUDIT = {
    "arrays_inspected": ["request_id", "split"],
    "split_labels_covered": ["train", "validation", "test"],
    "purpose": "within_capture_request_separation_and_cross_capture_split_compatibility_only",
    "test_scientific_tensors_or_values_admitted_evaluated_or_used": False,
    "test_metrics_computed": False,
    "test_tuning_selection_or_decision_use": False,
}
DIRECT_SELECTOR_VARIANTS = (
    ("q2_abc_control", "listwise_boundary", "single"),
    ("q2_abc_pq1", "listwise_boundary", "single"),
    ("q2_abc_pq1", "listwise_boundary_hard_set", "single"),
    ("q2_abc_pq1_joint_nested", "listwise_boundary_hard_set", "joint_nested"),
    ("q2_abc_static_unit_bias_control", "listwise_boundary_hard_set", "single"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array) or not np.all(np.isfinite(array)):
        raise RuntimeError("summary values are empty or non-finite")
    return {
        "n": int(len(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _plain(value: Any) -> Any:
    """Convert numpy/pandas scalars to canonical JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if not isinstance(value, (str, bytes)) and pd.isna(value):
        return None
    return value


def _require_bool(value: Mapping[str, Any], key: str, expected: bool) -> None:
    if value.get(key) is not expected:
        raise RuntimeError(f"immutable boolean {key} is {value.get(key)!r}, expected {expected!r}")


def _require_equal(value: Mapping[str, Any], key: str, expected: Any) -> None:
    if value.get(key) != expected:
        raise RuntimeError(f"immutable fact {key} is {value.get(key)!r}, expected {expected!r}")


def validate_config_contract(config: Mapping[str, Any]) -> None:
    """Validate the fail-closed contract used by both runner and analyzer."""
    _require_equal(config, "selection_split", "validation")
    _require_equal(config, "evaluation_splits", ["validation"])
    _require_equal(config, "fit_splits", ["train"])
    _require_bool(config, "no_test_tuning", True)
    no_test_admitted = config.get("no_test_rows_admitted", config.get("no_test_rows_loaded"))
    if no_test_admitted is not True:
        raise RuntimeError("configuration must prohibit admitting test rows")
    if "no_test_rows_used" in config:
        _require_bool(config, "no_test_rows_used", True)
    if int(config["page_size_bytes"]) != 512:
        raise RuntimeError("page size is not the locked 512 bytes")
    if int(config["expert_weights"]) != 3_145_728:
        raise RuntimeError("expert weight count is not the locked 3,145,728")
    if int(config["candidate_units"]) != 256 or int(config["applied_units"]) != 192:
        raise RuntimeError("candidate interface must fetch 256 and apply 192 coherent units")
    if int(config.get("expected_validation_unique_invocations", -1)) != LOCKED_VALIDATION_INVOCATIONS:
        raise RuntimeError("configuration must lock exactly 69 validation invocations")
    if int(config.get("expected_validation_layer_expert_cells", -1)) != LOCKED_VALIDATION_CELLS:
        raise RuntimeError("configuration must lock exactly 12 validation layer/expert cells")
    if tuple(map(str, config.get("template_training_cohorts", ()))) != FIT_COHORTS:
        raise RuntimeError("configuration changed the three exact template cohorts")
    if str(config.get("direct_candidate_head_supervision")) != JOINT_CANDIDATE_COVERAGE_SEMANTICS:
        raise RuntimeError("configuration changed joint candidate-head supervision")
    if int(config.get("expected_cross_capture_shared_request_ids", -1)) < 0:
        raise RuntimeError("configuration has a negative cross-capture overlap expectation")


def _validated_runtime_provenance(value: Any, label: str) -> dict[str, Any]:
    """Return one exact runtime-provenance record or fail closed."""
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} runtime provenance is absent or malformed")
    if set(value) != set(RUNTIME_PROVENANCE_FIELDS):
        raise RuntimeError(
            f"{label} runtime provenance fields changed: "
            f"{sorted(value)} != {sorted(RUNTIME_PROVENANCE_FIELDS)}"
        )
    result = {key: _plain(value[key]) for key in RUNTIME_PROVENANCE_FIELDS}
    if not isinstance(result["gpu_names_used"], list):
        raise RuntimeError(f"{label} gpu_names_used is not a list")
    return result


def _validated_cross_capture_request_audit(
    value: Any, config: Mapping[str, Any], label: str,
) -> dict[str, Any]:
    """Validate the exact no-cross-split-leakage facts serialized by fit."""
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} cross-capture request audit is absent")
    expected = int(config.get("expected_cross_capture_shared_request_ids", -1))
    if expected < 0:
        raise RuntimeError("configuration lacks a nonnegative shared-request expectation")
    required = {
        "split_compatible": True,
        "expected_shared_request_ids_match": True,
        "shared_request_id_count": expected,
        "expected_shared_request_id_count": expected,
        "incompatible_overlap_count": 0,
        "cross_source_train_to_nontrain_overlap_count": 0,
        "incompatible_overlaps": [],
        "cross_source_train_to_nontrain_overlaps": [],
    }
    for key, expected_value in required.items():
        if value.get(key) != expected_value:
            raise RuntimeError(
                f"{label} cross-capture request audit changed {key}: "
                f"{value.get(key)!r} != {expected_value!r}"
            )
    labels = {"train", "validation", "test"}
    counts = value.get("shared_same_split_request_id_counts")
    identifiers = value.get("shared_same_split_request_ids")
    if (
        not isinstance(counts, Mapping) or set(counts) != labels
        or not isinstance(identifiers, Mapping) or set(identifiers) != labels
    ):
        raise RuntimeError(f"{label} cross-capture same-split audit keys changed")
    for split in labels:
        ids = identifiers[split]
        count = counts[split]
        if (
            type(count) is not int or count < 0 or not isinstance(ids, list)
            or len(ids) != count or not all(isinstance(item, str) for item in ids)
        ):
            raise RuntimeError(
                f"{label} cross-capture same-split audit is malformed for {split}"
            )
    if sum(int(counts[split]) for split in labels) != expected:
        raise RuntimeError(f"{label} cross-capture same-split counts do not total expected overlap")
    return _plain(value)


def _locked_validation_cells(config: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    """Resolve the immutable exact-checkpoint validation layer/expert cells."""
    mapping = config.get("locked_exact_sampled_experts")
    if not isinstance(mapping, Mapping):
        raise RuntimeError("configuration lacks locked_exact_sampled_experts")
    configured_layers = set(map(int, config["layers"]))
    if set(map(int, mapping)) != configured_layers:
        raise RuntimeError("locked exact expert layers differ from configured layers")
    cells: list[tuple[int, int]] = []
    for layer_text, experts in mapping.items():
        if not isinstance(experts, Mapping) or not experts:
            raise RuntimeError(f"locked exact expert scope is empty for layer {layer_text}")
        cells.extend((int(layer_text), int(expert)) for expert in experts)
    if len(cells) != LOCKED_VALIDATION_CELLS or len(set(cells)) != LOCKED_VALIDATION_CELLS:
        raise RuntimeError(
            "locked exact validation scope must contain exactly "
            f"{LOCKED_VALIDATION_CELLS} distinct layer/expert cells"
        )
    return tuple(sorted(cells))


def _identity_set(frame: pd.DataFrame) -> set[tuple[Any, ...]]:
    return set(frame[list(IDENTITY)].drop_duplicates().itertuples(index=False, name=None))


def _validate_locked_validation_scope(
    config: Mapping[str, Any], facts: Mapping[str, Any], frames: Mapping[str, pd.DataFrame],
) -> dict[tuple[int, int], set[tuple[Any, ...]]]:
    """Reject a self-consistent but reduced runner plan or a missing expert cell."""
    expected = int(facts.get("expected_unique_invocations", -1))
    observed = int(facts.get("observed_unique_invocations", -1))
    if expected != LOCKED_VALIDATION_INVOCATIONS or observed != LOCKED_VALIDATION_INVOCATIONS:
        raise RuntimeError(
            "exact-checkpoint validation scope must contain exactly "
            f"{LOCKED_VALIDATION_INVOCATIONS} unique invocations; "
            f"run facts declare expected={expected}, observed={observed}"
        )
    locked_cells = set(_locked_validation_cells(config))
    hybrid_identities = _identity_set(frames["hybrid"])
    if len(hybrid_identities) != LOCKED_VALIDATION_INVOCATIONS:
        raise RuntimeError(
            "hybrid evidence does not contain the locked 69 unique validation invocations"
        )
    by_cell: dict[tuple[int, int], set[tuple[Any, ...]]] = {
        cell: {identity for identity in hybrid_identities if identity[4:6] == cell}
        for cell in locked_cells
    }
    observed_cells = {identity[4:6] for identity in hybrid_identities}
    if observed_cells != locked_cells or any(not values for values in by_cell.values()):
        missing = sorted(locked_cells - observed_cells)
        extra = sorted(observed_cells - locked_cells)
        raise RuntimeError(
            "validation evidence does not cover all 12 locked layer/expert cells; "
            f"missing={missing}, extra={extra}"
        )
    for kind, frame in frames.items():
        unknown = _identity_set(frame) - hybrid_identities
        if unknown:
            raise RuntimeError(
                f"{kind} evidence contains invocation identities outside the locked hybrid scope"
            )
    return by_cell


def _validate_serialized_validation_scope_audit(
    config: Mapping[str, Any], facts: Mapping[str, Any],
    invocations_by_cell: Mapping[tuple[int, int], set[tuple[Any, ...]]],
) -> None:
    """Bind the runner's pre-science plan audit to independently observed evidence."""
    audit = facts.get("validation_scope_audit")
    if not isinstance(audit, Mapping):
        raise RuntimeError("validation run facts omit validation_scope_audit")
    expected_cells = list(map(list, sorted(invocations_by_cell)))
    expected_records = [
        {"layer": layer, "expert_id": expert, "invocations": len(invocations_by_cell[(layer, expert)])}
        for layer, expert in sorted(invocations_by_cell)
    ]
    expected_scalars = {
        "expected_unique_invocations": LOCKED_VALIDATION_INVOCATIONS,
        "observed_unique_invocations": LOCKED_VALIDATION_INVOCATIONS,
        "expected_layer_expert_cells": LOCKED_VALIDATION_CELLS,
        "observed_layer_expert_cells": LOCKED_VALIDATION_CELLS,
        "validation_scope_complete": True,
    }
    for key, expected in expected_scalars.items():
        if audit.get(key) != expected:
            raise RuntimeError(f"validation_scope_audit changed {key}")
    if audit.get("locked_layer_expert_cells") != expected_cells:
        raise RuntimeError("validation_scope_audit locked cells differ from evidence")
    observed_records = sorted(
        audit.get("observed_layer_expert_cell_records", []),
        key=lambda item: (int(item["layer"]), int(item["expert_id"])),
    )
    if observed_records != expected_records:
        raise RuntimeError("validation_scope_audit per-cell invocation counts differ from evidence")
    if int(facts.get("expected_layer_expert_cells", -1)) != LOCKED_VALIDATION_CELLS:
        raise RuntimeError("validation run facts changed expected_layer_expert_cells")


def _configured_selector_contract(
    config: Mapping[str, Any],
) -> tuple[tuple[str, str], ...]:
    """Enumerate the frozen selector configurations that fit must materialize."""
    expected: set[tuple[str, str]] = set()
    for stages in map(int, config.get("pq_additive_stages", ())):
        expected.add((f"block_pq_s{stages}_separate_gate_up", "block_pq_residual_synopsis"))
    for record in config.get("high_rank_low_bit_controls", ()):
        rank = int(record["rank"])
        if rank in {64, 128, 256}:
            expected.add((
                f"high_rank_r{rank}_{record['encoding']}",
                "high_rank_low_bit_linear_response",
            ))
    for cohort in map(str, config.get("direct_predictor_training_cohorts", ())):
        for feature, objective, head in DIRECT_SELECTOR_VARIANTS:
            expected.add((
                f"direct_{cohort}_{feature}_{objective}_{head}",
                "direct_set_predictor",
            ))
    return tuple(sorted(expected))


def _validate_selector_promotion_contract(
    key: tuple[int, int, str], entry: Mapping[str, Any],
) -> None:
    """Require immutable exclusion evidence for every predeclared control."""
    family = str(entry.get("selector_family"))
    config_id = str(key[2])
    if family == "high_rank_low_bit_linear_response":
        if entry.get("promotable") is not False:
            raise RuntimeError(f"high-rank control is not explicitly nonpromotable for {key}")
        scope = str(entry.get("sampled_expert_scope", ""))
        if "transductive_sampled_expert_diagnostic" not in scope:
            raise RuntimeError(f"high-rank control lacks transductive scope for {key}")
    is_static_bias = "q2_abc_static_unit_bias_control" in config_id
    static_semantics = str(entry.get("static_unit_bias_semantics", ""))
    if is_static_bias and "transductive_sampled_expert_upper_bound" not in static_semantics:
        raise RuntimeError(f"static-unit-bias control lacks its nonpromotion marker for {key}")
    if family == "direct_set_predictor":
        pq_dim = int(entry.get("pq_delta_dim", 0))
        expected_precision = (
            "fp16_round_then_fp32_reference" if pq_dim else "not_applicable"
        )
        if str(entry.get("pq_runtime_response_precision")) != expected_precision:
            raise RuntimeError(f"direct predictor PQ runtime precision changed for {key}")


def _validate_selector_grid_coverage(
    config: Mapping[str, Any],
    selector_index: Mapping[tuple[int, int, str], Mapping[str, Any]],
    frames: Mapping[str, pd.DataFrame],
    invocations_by_cell: Mapping[tuple[int, int], set[tuple[Any, ...]]],
) -> dict[str, Any]:
    """Match every validation-scoped manifest selector to its complete evidence grid."""
    locked_cells = set(invocations_by_cell)
    frozen = _configured_selector_contract(config)
    for layer, expert in sorted(locked_cells):
        observed_contract = {
            (config_id, str(entry.get("selector_family")))
            for (local_layer, local_expert, config_id), entry in selector_index.items()
            if (local_layer, local_expert) == (layer, expert)
        }
        if observed_contract != set(frozen):
            raise RuntimeError(
                "validation-scoped fit selector contract is incomplete or contains extras for "
                f"{(layer, expert)}; missing={sorted(set(frozen) - observed_contract)}, "
                f"extra={sorted(observed_contract - set(frozen))}"
            )
        for config_id, family in frozen:
            key = (layer, expert, config_id)
            entry = selector_index.get(key)
            if entry is None:
                raise RuntimeError(
                    "fit manifest is missing frozen configured selector "
                    f"{config_id!r} for validation cell {(layer, expert)}"
                )
            if str(entry.get("selector_family")) != family:
                raise RuntimeError(f"fit manifest selector family changed for {key}")
            if (
                family == "high_rank_low_bit_linear_response"
                and entry.get("analysis_design_rank_identifiable") is not True
            ):
                raise RuntimeError(f"high-rank fit is not marked identifiable for {key}")
            _validate_selector_promotion_contract(key, entry)

    manifest_entries = {
        key: entry for key, entry in selector_index.items() if key[:2] in locked_cells
    }
    if not manifest_entries:
        raise RuntimeError("fit manifest has no validation-scoped selector entries")

    for kind in ("selector", "candidate"):
        frame = frames[kind]
        if kind == "candidate":
            # Support-template rows are separately declared validation
            # existence oracles, not fitted selector-manifest entries.
            frame = frame[frame["selector_family"].astype(str) != "support_template"]
        for row in frame[["layer", "expert_id", "selector_config_id", "selector_family"]].drop_duplicates().itertuples(index=False):
            key = (int(row.layer), int(row.expert_id), str(row.selector_config_id))
            entry = manifest_entries.get(key)
            if entry is None:
                raise RuntimeError(f"{kind} row has no validation-scoped fit selector entry: {key}")
            if str(row.selector_family) != str(entry.get("selector_family")):
                raise RuntimeError(f"{kind} selector family changed for {key}")

    selector = frames["selector"]
    candidate = frames["candidate"]
    for (layer, expert, config_id), entry in sorted(manifest_entries.items()):
        family = str(entry.get("selector_family"))
        expected_identities = invocations_by_cell[(layer, expert)]
        direct_rows = selector[
            (pd.to_numeric(selector["layer"]) == layer)
            & (pd.to_numeric(selector["expert_id"]) == expert)
            & (selector["selector_config_id"].astype(str) == config_id)
            & (selector["selector_family"].astype(str) == family)
        ]
        if len(direct_rows) != len(expected_identities) or _identity_set(direct_rows) != expected_identities:
            raise RuntimeError(
                "selector evidence is incomplete for manifest selector "
                f"{(layer, expert, config_id, family)}"
            )
        if family == "direct_set_predictor":
            expected_precision = (
                "fp16_round_then_fp32_reference"
                if int(entry.get("pq_delta_dim", 0)) else "not_applicable"
            )
            if "pq_runtime_response_precision" not in direct_rows or set(
                direct_rows["pq_runtime_response_precision"].astype(str)
            ) != {expected_precision}:
                raise RuntimeError(
                    f"selector evidence changed direct PQ runtime precision for "
                    f"{(layer, expert, config_id)}"
                )
        candidate_rows = candidate[
            (pd.to_numeric(candidate["layer"]) == layer)
            & (pd.to_numeric(candidate["expert_id"]) == expert)
            & (candidate["selector_config_id"].astype(str) == config_id)
            & (candidate["selector_family"].astype(str) == family)
        ]
        observed_semantics = set(candidate_rows["rerank_semantics"].astype(str))
        if observed_semantics != set(CANDIDATE_SEMANTICS):
            raise RuntimeError(
                f"candidate evidence for manifest selector {(layer, expert, config_id, family)} "
                f"has semantics {sorted(observed_semantics)}, expected {sorted(CANDIDATE_SEMANTICS)}"
            )
        if family == "direct_set_predictor":
            expected_precision = (
                "fp16_round_then_fp32_reference"
                if int(entry.get("pq_delta_dim", 0)) else "not_applicable"
            )
            if "pq_runtime_response_precision" not in candidate_rows or set(
                candidate_rows["pq_runtime_response_precision"].astype(str)
            ) != {expected_precision}:
                raise RuntimeError(
                    f"candidate evidence changed direct PQ runtime precision for "
                    f"{(layer, expert, config_id)}"
                )
        for semantic in CANDIDATE_SEMANTICS:
            local = candidate_rows[candidate_rows["rerank_semantics"].astype(str) == semantic]
            if len(local) != len(expected_identities) or _identity_set(local) != expected_identities:
                raise RuntimeError(
                    "candidate evidence is incomplete for manifest selector/semantic "
                    f"{(layer, expert, config_id, family, semantic)}"
                )
    return {
        "locked_unique_invocations": LOCKED_VALIDATION_INVOCATIONS,
        "locked_layer_expert_cells": LOCKED_VALIDATION_CELLS,
        "validation_scoped_manifest_selector_entries": len(manifest_entries),
        "validation_scoped_manifest_selector_configs": len({
            (key[2], str(entry.get("selector_family")))
            for key, entry in manifest_entries.items()
        }),
        "frozen_configured_selector_configs": len(frozen),
        "candidate_semantics": list(CANDIDATE_SEMANTICS),
        "complete": True,
    }


def _validate_hybrid_grid(
    config: Mapping[str, Any], frame: pd.DataFrame,
    invocations_by_cell: Mapping[tuple[int, int], set[tuple[Any, ...]]],
) -> dict[str, Any]:
    """Require the exact five-family by three-budget broad hybrid grid."""
    configured_budgets = tuple(map(float, config.get("hybrid_budgets_bpw", ())))
    if configured_budgets != LOCKED_HYBRID_BUDGETS:
        raise RuntimeError(
            f"hybrid budgets changed: {configured_budgets} != {LOCKED_HYBRID_BUDGETS}"
        )
    expected_identities = set().union(*invocations_by_cell.values())
    expected_grid = {
        (family, budget) for family in HYBRID_FAMILIES for budget in LOCKED_HYBRID_BUDGETS
    }
    observed_grid = set(zip(
        frame["selector_family"].astype(str),
        pd.to_numeric(frame["physical_budget_bpw"]).astype(float),
    ))
    if observed_grid != expected_grid:
        raise RuntimeError(
            "hybrid family/budget grid is incomplete or contains extras; "
            f"missing={sorted(expected_grid - observed_grid)}, "
            f"extra={sorted(observed_grid - expected_grid)}"
        )
    for family, budget in sorted(expected_grid):
        local = frame[
            (frame["selector_family"].astype(str) == family)
            & np.isclose(pd.to_numeric(frame["physical_budget_bpw"]), budget)
        ]
        if len(local) != LOCKED_VALIDATION_INVOCATIONS or _identity_set(local) != expected_identities:
            raise RuntimeError(
                f"hybrid grid lacks exact 69-invocation coverage for {(family, budget)}"
            )
    return {
        "families": list(HYBRID_FAMILIES),
        "budgets_bpw": list(LOCKED_HYBRID_BUDGETS),
        "expected_rows": (
            len(HYBRID_FAMILIES) * len(LOCKED_HYBRID_BUDGETS)
            * LOCKED_VALIDATION_INVOCATIONS
        ),
        "complete": True,
    }


def _template_counts_for_cohort(config: Mapping[str, Any], cohort: str) -> tuple[int, ...]:
    cohorts = tuple(map(str, config.get("template_training_cohorts", ())))
    if not cohorts or cohort not in cohorts:
        raise RuntimeError(f"template cohort is not frozen in config: {cohort}")
    key = "template_counts_primary" if cohort == cohorts[0] else "template_counts_augmented"
    counts = tuple(map(int, config.get(key, ())))
    if not counts or len(set(counts)) != len(counts):
        raise RuntimeError(f"template K grid is empty or duplicated for cohort {cohort}")
    return counts


def _validate_template_grid(
    config: Mapping[str, Any], fit_manifest: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    invocations_by_cell: Mapping[tuple[int, int], set[tuple[Any, ...]]],
) -> dict[str, Any]:
    """Require every manifest/config template oracle combination and rerank interface."""
    cohorts = tuple(map(str, config.get("template_training_cohorts", ())))
    if cohorts != FIT_COHORTS:
        raise RuntimeError(
            f"template training cohorts changed: {cohorts} != {FIT_COHORTS}"
        )
    repairs = tuple(map(int, config.get("template_repair_counts", ())))
    if repairs != LOCKED_TEMPLATE_REPAIRS:
        raise RuntimeError(
            f"template repair grid changed: {repairs} != {LOCKED_TEMPLATE_REPAIRS}"
        )
    candidate_cap = int(config["candidate_units"])
    template_index: dict[tuple[int, int, str], dict[str, Any]] = {}
    locked_cells = set(invocations_by_cell)
    example_counts: dict[tuple[int, int, str], int] = {}
    for layer_text, record in fit_manifest.get("layers", {}).items():
        layer = int(layer_text)
        layer_experts = {expert for local_layer, expert in locked_cells if local_layer == layer}
        raw_counts = record.get("template_cohort_example_counts_by_expert")
        if not isinstance(raw_counts, Mapping) or set(map(int, raw_counts)) != layer_experts:
            raise RuntimeError(
                f"template cohort example-count expert scope changed for layer {layer}"
            )
        for expert_text, cohort_counts in raw_counts.items():
            expert = int(expert_text)
            if not isinstance(cohort_counts, Mapping) or set(cohort_counts) != set(FIT_COHORTS):
                raise RuntimeError(
                    f"template cohort example-count keys changed for {(layer, expert)}"
                )
            for cohort in FIT_COHORTS:
                count = cohort_counts[cohort]
                if type(count) is not int or int(count) < 0:
                    raise RuntimeError(
                        f"template cohort example count is invalid for {(layer, expert, cohort)}"
                    )
                example_counts[(layer, expert, cohort)] = int(count)
        for entry in record.get("template_entries", []):
            key = (layer, int(entry["expert_id"]), str(entry["cohort"]))
            if key[:2] not in locked_cells:
                continue
            if key in template_index:
                raise RuntimeError(f"duplicate validation-scoped template manifest entry: {key}")
            template_index[key] = dict(entry)
    for layer, expert in sorted(locked_cells):
        observed_cohorts = {
            cohort for local_layer, local_expert, cohort in template_index
            if (local_layer, local_expert) == (layer, expert)
        }
        expected_cohorts = {FIT_COHORTS[1], FIT_COHORTS[2]}
        routed_examples = example_counts.get((layer, expert, FIT_COHORTS[0]), -1)
        if routed_examples > 0:
            expected_cohorts.add(FIT_COHORTS[0])
        if observed_cohorts != expected_cohorts:
            raise RuntimeError(
                f"fit manifest template cohort scope disagrees with explicit example counts "
                f"for {(layer, expert)}; observed={sorted(observed_cohorts)}, "
                f"expected={sorted(expected_cohorts)}"
            )
        for cohort in (FIT_COHORTS[1], FIT_COHORTS[2]):
            if example_counts.get((layer, expert, cohort), 0) <= 0:
                raise RuntimeError(
                    f"synthetic template cohort has no training examples for {(layer, expert, cohort)}"
                )
    for (layer, expert, cohort), entry in sorted(template_index.items()):
        requested = tuple(map(int, entry.get("requested_template_counts", ())))
        expected_counts = _template_counts_for_cohort(config, cohort)
        if requested != expected_counts:
            raise RuntimeError(
                f"fit manifest template K grid changed for {(layer, expert, cohort)}"
            )
        if int(entry.get("available_templates", 0)) <= 0:
            raise RuntimeError(f"fit manifest has no templates for {(layer, expert, cohort)}")
        if int(entry.get("training_examples", -1)) != example_counts[(layer, expert, cohort)]:
            raise RuntimeError(
                f"template entry/example-count evidence differs for {(layer, expert, cohort)}"
            )
        expected_pairing = (
            "actual_routed_occurrences" if cohort == cohorts[0]
            else "synthetic_all_x_by_expert_augmentation"
        )
        if str(entry.get("pairing_semantics")) != expected_pairing:
            raise RuntimeError(f"template pairing semantics changed for {(layer, expert, cohort)}")

    template = frames["template"]
    support = frames["candidate"][
        frames["candidate"]["selector_family"].astype(str) == "support_template"
    ]
    for name, local in (("template", template), ("support-template candidate", support)):
        if "promotable" not in local or not (local["promotable"] == False).all():  # noqa: E712
            raise RuntimeError(f"{name} evidence is not explicitly nonpromotable")
        if "candidate_cap_units" not in local or not (
            pd.to_numeric(local["candidate_cap_units"]) == candidate_cap
        ).all():
            raise RuntimeError(f"{name} evidence changed the declared candidate cap")
        if "candidate_cap_pass" not in local:
            raise RuntimeError(f"{name} evidence omits candidate-cap arithmetic")
        expected_cap_pass = pd.to_numeric(local["candidate_units"]) <= candidate_cap
        if not np.array_equal(
            local["candidate_cap_pass"].astype(bool).to_numpy(), expected_cap_pass.to_numpy(),
        ):
            raise RuntimeError(f"{name} candidate_cap_pass disagrees with charged candidate units")

    template_groups = {
        (int(key[0]), int(key[1]), str(key[2]), int(key[3]), str(key[4]), int(key[5])): local
        for key, local in template.groupby([
            "layer", "expert_id", "template_cohort", "template_count_requested",
            "template_selector", "repair_count",
        ], sort=False, dropna=False)
    }
    support_groups = {
        (int(key[0]), int(key[1]), str(key[2]), str(key[3])): local
        for key, local in support.groupby([
            "layer", "expert_id", "selector_config_id", "rerank_semantics",
        ], sort=False, dropna=False)
    }

    def candidate_units_by_identity(local: pd.DataFrame) -> dict[tuple[Any, ...], int]:
        return {
            tuple(row[:-1]): int(row[-1])
            for row in local[[*IDENTITY, "candidate_units"]].itertuples(index=False, name=None)
        }

    expected_template_rows = 0
    expected_support_rows = 0
    for (layer, expert, cohort), entry in sorted(template_index.items()):
        identities = invocations_by_cell[(layer, expert)]
        pairing = str(entry["pairing_semantics"])
        available = int(entry["available_templates"])
        expected_combinations = {
            (count, selector, repair)
            for count in map(int, entry["requested_template_counts"])
            for selector in TEMPLATE_SELECTORS
            for repair in repairs
        }
        observed_combinations = {
            (count, selector, repair)
            for local_layer, local_expert, local_cohort, count, selector, repair in template_groups
            if (local_layer, local_expert, local_cohort) == (layer, expert, cohort)
        }
        if observed_combinations != expected_combinations:
            raise RuntimeError(
                f"template grid is incomplete for {(layer, expert, cohort)}; "
                f"missing={sorted(expected_combinations - observed_combinations)}, "
                f"extra={sorted(observed_combinations - expected_combinations)}"
            )
        for count, selector, repair in sorted(expected_combinations):
            local = template_groups[(layer, expert, cohort, count, selector, repair)]
            if len(local) != len(identities) or _identity_set(local) != identities:
                raise RuntimeError(
                    "template combination lacks exact per-cell invocation coverage for "
                    f"{(layer, expert, cohort, count, selector, repair)}"
                )
            if set(pd.to_numeric(local["template_count_effective"]).astype(int)) != {
                min(count, available)
            }:
                raise RuntimeError("template effective K differs from fit-manifest availability")
            if set(local["pairing_semantics"].astype(str)) != {pairing}:
                raise RuntimeError("template pairing semantics differ from the fit manifest")
            selector_id = f"template:{cohort}:K{count}:{selector}:repair{repair}"
            observed_semantics = {
                semantic
                for local_layer, local_expert, local_id, semantic in support_groups
                if (local_layer, local_expert, local_id) == (layer, expert, selector_id)
            }
            if observed_semantics != set(CANDIDATE_SEMANTICS):
                raise RuntimeError(f"support-template candidate semantics are incomplete for {selector_id}")
            template_candidate_units = candidate_units_by_identity(local)
            for semantic in CANDIDATE_SEMANTICS:
                semantic_rows = support_groups[(layer, expert, selector_id, semantic)]
                if len(semantic_rows) != len(identities) or _identity_set(semantic_rows) != identities:
                    raise RuntimeError(
                        "support-template candidate interface lacks exact per-cell coverage for "
                        f"{(layer, expert, selector_id, semantic)}"
                    )
                if candidate_units_by_identity(semantic_rows) != template_candidate_units:
                    raise RuntimeError(
                        f"template and candidate artifacts disagree on charged units for {selector_id}"
                    )
            expected_template_rows += len(identities)
            expected_support_rows += len(identities) * len(CANDIDATE_SEMANTICS)
    if len(template) != expected_template_rows or len(support) != expected_support_rows:
        raise RuntimeError("template artifacts contain an unrecognized or duplicate grid row")
    return {
        "configured_training_cohorts": list(cohorts),
        "manifest_present_training_cohorts": sorted({key[2] for key in template_index}),
        "template_selectors": list(TEMPLATE_SELECTORS),
        "repair_counts": list(repairs),
        "candidate_cap_units": candidate_cap,
        "over_cap_rows_retained_as_nonpromotable_evidence": int(
            (pd.to_numeric(template["candidate_units"]) > candidate_cap).sum()
        ),
        "expected_template_rows": expected_template_rows,
        "expected_support_candidate_rows": expected_support_rows,
        "explicitly_nonpromotable": True,
        "complete": True,
    }


def _validate_fit_bundle(
    config: Mapping[str, Any], config_path: Path, fit_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[tuple[int, int, str], dict[str, Any]]]:
    facts_path = fit_dir / "fit_facts.json"
    manifest_path = fit_dir / "set_utility_fit_manifest.json"
    facts = json.loads(facts_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    _require_bool(facts, "completed", True)
    if facts.get("failures") not in ([], None):
        raise RuntimeError("fit facts contain a current failure")
    _require_equal(facts, "fit_split", "train")
    _require_bool(facts, "validation_or_test_rows_used", False)
    _require_bool(facts, "test_rows_admitted", False)
    _require_bool(facts, "test_rows_used", False)
    _require_bool(facts, "monolithic_source_contains_test_rows", True)
    _require_bool(facts, "request_separation_verified", True)
    _require_bool(facts, "codec_locked", True)
    config_hash = sha256(config_path)
    _require_equal(facts, "config_sha256", config_hash)
    _require_equal(facts, "exact_capture_sha256", config["locked_capture_sha256"]["exact_checkpoint"])
    _require_equal(facts, "cross_capture_sha256", config["locked_capture_sha256"]["cross_reference"])
    _require_equal(facts, "tree_sha256", config["locked_tree_sha256"])
    _require_equal(facts, "checkpoint_config_sha256", config["reference"]["config_sha256"])
    _require_equal(facts, "checkpoint_index_sha256", config["reference"]["index_sha256"])
    _require_equal(facts, "manifest_sha256", sha256(manifest_path))

    _require_equal(manifest, "run_id", config["run_id"])
    _require_equal(manifest, "config_sha256", config_hash)
    _require_equal(manifest, "fit_split", "train")
    _require_bool(manifest, "validation_or_test_rows_used", False)
    _require_bool(manifest, "test_rows_admitted", False)
    _require_bool(manifest, "test_rows_used", False)
    _require_bool(manifest, "monolithic_source_contains_test_rows", True)
    facts_audit = _validated_cross_capture_request_audit(
        facts.get("cross_capture_request_id_audit"), config, "fit facts",
    )
    manifest_audit = _validated_cross_capture_request_audit(
        manifest.get("cross_capture_request_id_audit"), config, "fit manifest",
    )
    if facts_audit != manifest_audit:
        raise RuntimeError("fit facts/manifest cross-capture request audits differ")
    _require_equal(
        facts, "direct_candidate_head_supervision", JOINT_CANDIDATE_COVERAGE_SEMANTICS,
    )
    _require_equal(
        manifest, "direct_candidate_head_supervision", JOINT_CANDIDATE_COVERAGE_SEMANTICS,
    )
    fit_runtime = _validated_runtime_provenance(
        facts.get("runtime_provenance"), "fit facts",
    )
    if _validated_runtime_provenance(
        manifest.get("runtime_provenance"), "fit manifest",
    ) != fit_runtime:
        raise RuntimeError("fit facts/manifest runtime provenance differs")
    if set(map(int, manifest.get("layers", {}))) != set(map(int, config["layers"])):
        raise RuntimeError("fit manifest layers differ from the frozen configuration")

    selector_index: dict[tuple[int, int, str], dict[str, Any]] = {}
    for layer_text, record in manifest["layers"].items():
        layer = int(layer_text)
        shard_name = Path(str(record["shard"]))
        if shard_name.is_absolute() or shard_name.name != str(shard_name):
            raise RuntimeError("fit shard path is not a safe local basename")
        shard = fit_dir / shard_name
        sidecar = fit_dir / f"set_utility_fit_layer_{layer}.json"
        if sha256(shard) != str(record["shard_sha256"]):
            raise RuntimeError(f"fit shard {layer} hash changed")
        if shard.stat().st_size != int(record["shard_bytes"]):
            raise RuntimeError(f"fit shard {layer} byte count changed")
        if sha256(sidecar) != str(record["sidecar_sha256"]):
            raise RuntimeError(f"fit sidecar {layer} hash changed")
        sidecar_value = json.loads(sidecar.read_text())
        if sidecar_value != {key: value for key, value in record.items() if key != "sidecar_sha256"}:
            raise RuntimeError(f"fit sidecar {layer} differs from its manifest record")
        if sidecar_value.get("fit_split") != "train" or sidecar_value.get("validation_or_test_rows_used") is not False:
            raise RuntimeError(f"fit sidecar {layer} admits non-training evidence")
        if _validated_runtime_provenance(
            sidecar_value.get("runtime_provenance"), f"fit sidecar {layer}",
        ) != fit_runtime:
            raise RuntimeError(f"fit sidecar {layer} runtime provenance differs")
        for entry in record.get("selector_entries", []):
            experts = entry.get("experts", [entry.get("expert_id")])
            if not experts or any(expert is None for expert in experts):
                raise RuntimeError("fit selector entry has no expert scope")
            for expert in map(int, experts):
                key = (layer, expert, str(entry["config_id"]))
                if key in selector_index:
                    raise RuntimeError(f"duplicate fit selector identity: {key}")
                selector_index[key] = dict(entry)
    return facts, manifest, selector_index


def _validate_code_hashes(facts: Mapping[str, Any], experiment: Path) -> None:
    mapping = {
        "runner_sha256": experiment / "scripts/run_set_utility_distillation.py",
        "teacher_core_sha256": experiment / "src/oracle_study/unit_set_teacher.py",
        "oracle_core_sha256": experiment / "src/oracle_study/set_utility_oracles.py",
        "selector_core_sha256": experiment / "src/oracle_study/set_utility_selector.py",
        "direct_predictor_core_sha256": experiment / "src/oracle_study/direct_set_predictor.py",
        "neuron_selector_core_sha256": experiment / "src/oracle_study/neuron_selector.py",
        "prior_runner_sha256": experiment / "scripts/run_neuron_selector_distillation.py",
        "base_runner_sha256": experiment / "scripts/run_sparse_streaming_study.py",
        "sparse_streaming_core_sha256": experiment / "src/oracle_study/sparse_streaming.py",
        "mxfp4_embed_core_sha256": experiment / "src/oracle_study/mxfp4_embed.py",
        "selective_pages_runner_sha256": experiment / "scripts/run_mxfp4_selective_pages.py",
        "phase_a_runner_sha256": experiment / "scripts/run_phase_a_remote.py",
    }
    for key, path in mapping.items():
        if not path.exists():
            raise RuntimeError(f"executable dependency is missing for {key}: {path}")
        if key not in facts:
            raise RuntimeError(f"missing executable dependency hash: {key}")
        if facts[key] != sha256(path):
            raise RuntimeError(f"executable dependency hash changed for {key}")


def load_validation_frame(path: Path, kind: str) -> pd.DataFrame:
    """Load a whole artifact and fail if *any* non-validation row is present."""
    if kind not in ARTIFACT_FILES:
        raise ValueError(f"unknown evidence kind: {kind}")
    frame = pd.read_parquet(path)
    missing = REQUIRED_COLUMNS[kind] - set(frame.columns)
    # repair_count was added after the initial schema; absence means zero and
    # is made explicit before defining scientific identity.
    if kind == "template" and missing == {"repair_count"}:
        frame = frame.copy()
        frame["repair_count"] = 0
        missing.clear()
    if missing:
        raise RuntimeError(f"{path.name} is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise RuntimeError(f"validation evidence is empty: {path}")
    if set(frame["evaluation_split"].astype(str)) != {"validation"}:
        raise RuntimeError(f"{path.name} contains a non-validation row")
    if set(frame["capture_source"].astype(str)) != {"exact_checkpoint"}:
        raise RuntimeError(f"{path.name} contains non-exact-checkpoint evidence")
    keys = list(SCIENTIFIC_KEYS[kind])
    if frame[keys].isnull().any().any():
        raise RuntimeError(f"{path.name} has a null scientific identity")
    if frame.duplicated(keys, keep=False).any():
        duplicate = frame.loc[frame.duplicated(keys, keep=False), keys].iloc[0].to_dict()
        raise RuntimeError(f"duplicate scientific identity in {path.name}: {duplicate}")
    for column in ("recovery", "physical_bpw", "selector_metadata_bpw", "storage_multiplier"):
        if column in frame and not np.all(np.isfinite(pd.to_numeric(frame[column]))):
            raise RuntimeError(f"{path.name} contains non-finite {column}")
    return frame


@dataclass(frozen=True)
class ValidatedEvidence:
    config: dict[str, Any]
    config_sha256: str
    fit_facts: dict[str, Any]
    fit_manifest: dict[str, Any]
    run_facts: dict[str, Any]
    selector_index: dict[tuple[int, int, str], dict[str, Any]]
    frames: dict[str, pd.DataFrame]
    evidence_sha256: dict[str, str]
    selector_grid_audit: dict[str, Any] | None = None


def validate_evidence_bundle(
    config_path: Path, fit_dir: Path, validation_dir: Path, experiment: Path,
) -> ValidatedEvidence:
    config_path, fit_dir, validation_dir = map(Path, (config_path, fit_dir, validation_dir))
    config = json.loads(config_path.read_text())
    validate_config_contract(config)
    fit_facts, fit_manifest, selector_index = _validate_fit_bundle(config, config_path, fit_dir)
    _validate_code_hashes(fit_facts, Path(experiment))
    facts_path = validation_dir / "run_facts.json"
    facts = json.loads(facts_path.read_text())
    _require_bool(facts, "completed", True)
    if facts.get("failures") != []:
        raise RuntimeError("validation facts contain failures")
    _require_equal(facts, "run_id", config["run_id"])
    _require_equal(facts, "evaluation_split", "validation")
    _require_equal(facts, "config_sha256", sha256(config_path))
    _require_equal(facts, "capture_sha256", config["locked_capture_sha256"]["exact_checkpoint"])
    _require_equal(facts, "tree_sha256", config["locked_tree_sha256"])
    _require_equal(facts, "checkpoint_config_sha256", config["reference"]["config_sha256"])
    _require_equal(facts, "checkpoint_index_sha256", config["reference"]["index_sha256"])
    _require_equal(facts, "fit_manifest_sha256", sha256(fit_dir / "set_utility_fit_manifest.json"))
    _require_equal(facts, "fit_facts_sha256", sha256(fit_dir / "fit_facts.json"))
    _require_bool(facts, "request_separation_verified", True)
    _require_bool(facts, "test_rows_admitted", False)
    _require_bool(facts, "test_rows_used", False)
    _require_bool(facts, "monolithic_source_contains_test_rows", True)
    _require_bool(facts, "codec_locked", True)
    run_audit = _validated_cross_capture_request_audit(
        facts.get("cross_capture_request_id_audit"), config, "validation run facts",
    )
    fit_audit = _validated_cross_capture_request_audit(
        fit_facts.get("cross_capture_request_id_audit"), config, "fit facts",
    )
    if run_audit != fit_audit:
        raise RuntimeError("fit/validation cross-capture request audits differ")
    _require_equal(
        facts, "direct_candidate_head_supervision", JOINT_CANDIDATE_COVERAGE_SEMANTICS,
    )
    fit_runtime = _validated_runtime_provenance(
        fit_facts.get("runtime_provenance"), "fit facts",
    )
    if _validated_runtime_provenance(
        facts.get("runtime_provenance"), "validation run facts",
    ) != fit_runtime:
        raise RuntimeError("fit/validation runtime provenance differs")
    if int(facts.get("observed_unique_invocations", -1)) != int(facts.get("expected_unique_invocations", -2)):
        raise RuntimeError("validation invocation coverage is incomplete")
    _validate_code_hashes(facts, Path(experiment))

    frames: dict[str, pd.DataFrame] = {}
    hashes: dict[str, str] = {}
    for kind, name in ARTIFACT_FILES.items():
        path = validation_dir / name
        frames[kind] = load_validation_frame(path, kind)
        hashes[kind] = sha256(path)
        counts = facts.get("artifact_row_counts", {})
        runner_key = {
            "hybrid": "hybrid_oracle_frontier",
            "template": "template_frontier",
            "selector": "selector_frontier",
            "candidate": "candidate_frontier",
        }[kind]
        if int(counts.get(runner_key, -1)) != len(frames[kind]):
            raise RuntimeError(f"run-facts row count differs for {kind}")

    invocations_by_cell = _validate_locked_validation_scope(config, facts, frames)
    _validate_serialized_validation_scope_audit(config, facts, invocations_by_cell)
    selector_grid = _validate_selector_grid_coverage(
        config, selector_index, frames, invocations_by_cell,
    )
    hybrid_grid = _validate_hybrid_grid(config, frames["hybrid"], invocations_by_cell)
    template_grid = _validate_template_grid(
        config, fit_manifest, frames, invocations_by_cell,
    )
    selector_grid_audit = {
        **selector_grid,
        "hybrid_grid": hybrid_grid,
        "template_grid": template_grid,
    }
    if any(str(value).lower() == "test" for value in facts.get("completed_work_units", [])):
        raise RuntimeError("completed work units include test")
    return ValidatedEvidence(
        config=dict(config), config_sha256=sha256(config_path), fit_facts=fit_facts,
        fit_manifest=fit_manifest, run_facts=facts, selector_index=selector_index,
        frames=frames, evidence_sha256=hashes, selector_grid_audit=selector_grid_audit,
    )


def _assert_close(frame: pd.DataFrame, column: str, expected: np.ndarray, *, atol: float = 1e-10) -> None:
    if column not in frame:
        raise RuntimeError(f"accounting column {column} is absent")
    observed = pd.to_numeric(frame[column]).to_numpy(np.float64)
    desired = np.asarray(expected, np.float64)
    if not np.allclose(observed, desired, rtol=1e-10, atol=atol):
        index = int(np.flatnonzero(~np.isclose(observed, desired, rtol=1e-10, atol=atol))[0])
        raise RuntimeError(
            f"accounting mismatch for {column} at row {index}: {observed[index]} != {desired[index]}",
        )


def _assert_bool_array(frame: pd.DataFrame, column: str, expected: np.ndarray) -> None:
    """Reject truthy integers/strings as well as semantically wrong booleans."""
    if column not in frame:
        raise RuntimeError(f"accounting column {column} is absent")
    raw = frame[column].tolist()
    if not all(isinstance(value, (bool, np.bool_)) for value in raw):
        raise RuntimeError(f"accounting column {column} is not strictly boolean")
    observed = np.asarray(raw, bool)
    desired = np.asarray(expected, bool)
    if not np.array_equal(observed, desired):
        index = int(np.flatnonzero(observed != desired)[0])
        raise RuntimeError(
            f"accounting mismatch for {column} at row {index}: "
            f"{observed[index]} != {desired[index]}",
        )


def _manifest_accounting(
    frame: pd.DataFrame,
    selector_index: Mapping[tuple[int, int, str], Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    """Reconstruct storage/work from immutable fit records, including linked PQ."""
    outputs: list[list[float]] = [[] for _ in range(8)]
    blocks = int(config["expert_weights"]) // (3 * 512 * int(config["pq_block_size"]))
    if blocks != 64:
        raise RuntimeError("PQ block accounting no longer describes a 2048-wide expert")

    def pq_cost(entry: Mapping[str, Any]) -> tuple[int, int, int, int, int, int]:
        stages = int(entry["stages"])
        expert = int(entry["expert_specific_bytes"])
        shared = int(entry["layer_shared_bytes"])
        # Gate and up have separate layer-shared codebook/LUT banks.
        macs = 2 * stages * blocks * int(config["pq_codebook_size"]) * int(config["pq_block_size"])
        additions = 2 * 512 * (blocks * stages - 1)
        scales = 2 * 512 * blocks
        bytes_read = (
            shared + expert + int(entry["resident_scale_bytes_read"]) + 2 * 2048
            + 2 * 512 * blocks * stages * 4
        )
        return expert, shared, macs, additions, scales, bytes_read

    for row in frame.itertuples(index=False):
        key = (int(row.layer), int(row.expert_id), str(row.selector_config_id))
        if key not in selector_index:
            raise RuntimeError(f"selector row has no immutable fit-manifest entry: {key}")
        entry = selector_index[key]
        family = str(row.selector_family)
        if family != str(entry.get("selector_family")):
            raise RuntimeError(f"selector family changed for {key}")
        abc = int(entry.get("abc_bytes", 3 * (512 * 2 + 4)))
        if family == "block_pq_residual_synopsis":
            expert, shared, macs, additions, scales, bytes_read = pq_cost(entry)
            macs += UNITS * ABC_SCORE_MACS_PER_UNIT
            # Direct scoring consumes resident Q2 g/u/h and the exact encoded
            # A/B/C payload in addition to the PQ residual-response synopsis.
            bytes_read += Q2_UNIT_FEATURE_BYTES + abc
            activation_payload_already_read = True
        elif family == "high_rank_low_bit_linear_response":
            rank = int(entry["rank"])
            expert = int(entry["expert_specific_bytes"])
            shared = int(entry["layer_shared_bytes"])
            macs = rank * (2048 + 2 * 512) + UNITS * ABC_SCORE_MACS_PER_UNIT
            additions = 0
            scales = 2 * 512
            bytes_read = (
                shared + expert + ACTIVATION_PAYLOAD_BYTES
                + Q2_UNIT_FEATURE_BYTES + abc
            )
            activation_payload_already_read = True
        elif family == "direct_set_predictor":
            units = 512
            unit_input = int(entry["q2_unit_dim"]) + int(entry.get("pq_delta_dim", 0)) + int(entry.get("abc_dim", 0))
            hidden = int(entry["hidden_dim"])
            heads = 1 if str(entry.get("head_mode", "single")) == "single" else 2
            predictor_macs = (
                units * unit_input * hidden + unit_input * hidden
                + units * hidden * hidden + heads * units * hidden
            )
            predictor_additions = unit_input * (units - 1)
            if predictor_macs != int(entry["linear_macs"]) or predictor_additions != int(entry["context_additions"]):
                raise RuntimeError(f"direct predictor architecture arithmetic changed for {key}")
            expert = 0
            shared = int(entry["layer_shared_bytes"])
            macs, additions, scales = predictor_macs, predictor_additions, 0
            bytes_read = int(entry["model_parameter_bytes_fp16"]) + int(entry["normalization_bytes_fp16"])
            pq_dim = int(entry.get("pq_delta_dim", 0))
            expected_pq_precision = (
                "fp16_round_then_fp32_reference" if pq_dim else "not_applicable"
            )
            if str(entry.get("pq_runtime_response_precision")) != expected_pq_precision:
                raise RuntimeError(
                    f"direct predictor PQ runtime precision contract changed for {key}"
                )
            activation_payload_already_read = False
            if pq_dim:
                pq_id = str(entry["pq_config_by_expert"][str(int(row.expert_id))])
                pq_key = (int(row.layer), int(row.expert_id), pq_id)
                if pq_key not in selector_index:
                    raise RuntimeError(f"direct predictor linked PQ entry is absent: {pq_key}")
                pq_expert, pq_shared, pq_macs, pq_additions, pq_scales, pq_bytes = pq_cost(
                    selector_index[pq_key],
                )
                expert += pq_expert
                shared += pq_shared
                macs += pq_macs
                additions += pq_additions
                scales += pq_scales
                bytes_read += pq_bytes
                activation_payload_already_read = True
            # A/B/C may include per-vector scales and optional static bias.
            # Charge the exact immutable payload, not abc_dim*512*FP16.
            bytes_read += (
                FP16_BYTES * units * (int(entry["q2_unit_dim"]) + pq_dim)
                + abc
            )
        else:
            raise RuntimeError(f"no independent MAC formula for selector family {family}")
        for target, value in zip(outputs, (
            expert, shared, macs, additions, scales, bytes_read, abc,
            activation_payload_already_read,
        )):
            target.append(float(value))
    return tuple(np.asarray(value, np.float64) for value in outputs)  # type: ignore[return-value]


def _validate_selector_abc_score_accounting(frame: pd.DataFrame) -> None:
    """Validate the separately serialized direct-score component of selector work."""
    required = {
        "selector_abc_score_units", "selector_abc_score_compute_macs",
        "selector_abc_score_macs_per_unit", "selector_abc_score_mac_convention",
    }
    missing = required - set(frame)
    if missing:
        raise RuntimeError(f"selector ABC-score accounting fields are absent: {sorted(missing)}")
    standalone = frame["selector_family"].astype(str) != "direct_set_predictor"
    expected_units = standalone.astype(np.int64).to_numpy() * UNITS
    expected_macs_per_unit = standalone.astype(np.int64).to_numpy() * ABC_SCORE_MACS_PER_UNIT
    expected_macs = expected_units * ABC_SCORE_MACS_PER_UNIT
    _assert_close(frame, "selector_abc_score_units", expected_units)
    _assert_close(frame, "selector_abc_score_compute_macs", expected_macs)
    _assert_close(frame, "selector_abc_score_macs_per_unit", expected_macs_per_unit)
    if set(frame["selector_abc_score_mac_convention"].astype(str)) != {
        ABC_SCORE_MAC_CONVENTION
    }:
        raise RuntimeError("selector ABC score MAC convention changed")


def _manifest_q2_gate_up_scales_already_read(
    frame: pd.DataFrame,
    selector_index: Mapping[tuple[int, int, str], Mapping[str, Any]],
) -> np.ndarray:
    """Resolve selector-side scale availability without trusting evidence rows."""
    result: list[bool] = []
    for row in frame.itertuples(index=False):
        key = (int(row.layer), int(row.expert_id), str(row.selector_config_id))
        entry = selector_index.get(key)
        if entry is None:
            raise RuntimeError(f"selector row has no immutable fit-manifest entry: {key}")
        family = str(entry.get("selector_family"))
        if family == "block_pq_residual_synopsis":
            result.append(True)
        elif family == "high_rank_low_bit_linear_response":
            result.append(False)
        elif family == "direct_set_predictor":
            result.append(bool(int(entry.get("pq_delta_dim", 0))))
        else:
            raise RuntimeError(f"no Q2 gate/up scale-read contract for selector family {family}")
    return np.asarray(result, bool)


def _unit_id_tuple(value: Any, *, field: str, row: int, expected: int) -> tuple[int, ...]:
    """Decode one canonical JSON unit list and enforce its physical cardinality."""
    if not isinstance(value, str):
        raise RuntimeError(f"candidate field {field} at row {row} is not canonical JSON text")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"candidate field {field} at row {row} is invalid JSON") from error
    if not isinstance(decoded, list) or len(decoded) != int(expected):
        raise RuntimeError(
            f"candidate field {field} at row {row} has cardinality "
            f"{len(decoded) if isinstance(decoded, list) else 'non-list'}, expected {expected}",
        )
    if not all(type(unit) is int for unit in decoded):
        raise RuntimeError(f"candidate field {field} at row {row} contains a non-integer unit")
    units = tuple(decoded)
    if len(set(units)) != len(units):
        raise RuntimeError(f"candidate field {field} at row {row} contains duplicate units")
    if any(unit < 0 or unit >= UNITS for unit in units):
        raise RuntimeError(f"candidate field {field} at row {row} contains an out-of-range unit")
    return units


def _validate_candidate_set_evidence(frame: pd.DataFrame) -> None:
    """Validate fetched/applied/oracle sets before accepting set-utility metrics."""
    candidates_by_row: list[tuple[int, ...]] = []
    selected_by_row: list[tuple[int, ...]] = []
    oracle_by_row: list[tuple[int, ...]] = []
    groups: dict[tuple[Any, ...], list[int]] = {}
    group_columns = (*IDENTITY, "selector_config_id", "selector_family")
    for position, row in enumerate(frame.itertuples(index=False)):
        candidates = _unit_id_tuple(
            row.candidate_unit_ids, field="candidate_unit_ids", row=position,
            expected=int(row.candidate_units),
        )
        selected = _unit_id_tuple(
            row.selected_units, field="selected_units", row=position,
            expected=int(row.applied_units),
        )
        oracle = _unit_id_tuple(
            row.oracle_selected_units, field="oracle_selected_units", row=position,
            expected=int(row.applied_units),
        )
        if not set(selected).issubset(candidates):
            raise RuntimeError(f"selected_units at row {position} are not a subset of candidates")
        oracle_gain = float(row.oracle_set_gain)
        gain = float(row.set_gain)
        retention = float(row.oracle_set_gain_retention)
        if not np.isfinite(oracle_gain) or oracle_gain <= 0.0:
            raise RuntimeError(f"oracle_set_gain at row {position} is not finite and positive")
        if not np.isfinite(gain) or not np.isfinite(retention):
            raise RuntimeError(f"set-gain evidence at row {position} is non-finite")
        if not np.isclose(retention, gain / oracle_gain, rtol=1e-10, atol=1e-10):
            raise RuntimeError(f"oracle_set_gain_retention at row {position} is inconsistent")
        candidates_by_row.append(candidates)
        selected_by_row.append(selected)
        oracle_by_row.append(oracle)
        key = tuple(getattr(row, column) for column in group_columns)
        groups.setdefault(key, []).append(position)

    for key, positions in groups.items():
        semantics = {str(frame.iloc[position]["rerank_semantics"]) for position in positions}
        if len(positions) != len(CANDIDATE_SEMANTICS) or semantics != set(CANDIDATE_SEMANTICS):
            raise RuntimeError(f"candidate-set evidence lacks four semantics for {key}")
        if len({candidates_by_row[position] for position in positions}) != 1:
            raise RuntimeError(f"the four rerank semantics use different candidates for {key}")
        if len({oracle_by_row[position] for position in positions}) != 1:
            raise RuntimeError(f"the four rerank semantics use different oracle selections for {key}")
        oracle_gains = np.asarray(
            [float(frame.iloc[position]["oracle_set_gain"]) for position in positions],
            np.float64,
        )
        if not np.allclose(oracle_gains, oracle_gains[0], rtol=1e-12, atol=1e-12):
            raise RuntimeError(f"the four rerank semantics use different oracle gains for {key}")

        # Equal applied sets must have equal exact gain, independently of the
        # interface label or ordering used to produce them.
        gain_by_set: dict[frozenset[int], float] = {}
        for position in positions:
            selected_set = frozenset(selected_by_row[position])
            gain = float(frame.iloc[position]["set_gain"])
            if selected_set in gain_by_set and not np.isclose(
                gain, gain_by_set[selected_set], rtol=1e-10, atol=1e-10,
            ):
                raise RuntimeError(f"identical selected sets have different gains for {key}")
            gain_by_set[selected_set] = gain

        # If the fetched universe contains the unrestricted fixed-greedy
        # oracle path, restricting that same deterministic path cannot change
        # its selected set or exact set gain.
        full_position = next(
            position for position in positions
            if str(frame.iloc[position]["rerank_semantics"]) == FULL_TARGET
        )
        oracle_set = set(oracle_by_row[full_position])
        if oracle_set.issubset(candidates_by_row[full_position]):
            if set(selected_by_row[full_position]) != oracle_set:
                raise RuntimeError("full-target restricted path disagrees with unrestricted oracle path")
            if not np.isclose(
                float(frame.iloc[full_position]["set_gain"]),
                float(frame.iloc[full_position]["oracle_set_gain"]),
                rtol=1e-10, atol=1e-10,
            ):
                raise RuntimeError("full-target restricted gain disagrees with oracle_set_gain")


def _validate_candidate_rerank_accounting(
    frame: pd.DataFrame, config: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Recompute exact declared rerank charges for all four candidate interfaces."""
    expected_candidates = 256 if config is None else int(config["candidate_units"])
    expected_applied = 192 if config is None else int(config["applied_units"])
    applied = pd.to_numeric(frame["applied_units"]).to_numpy(np.int64)
    if not np.all(applied == expected_applied):
        raise RuntimeError(
            f"candidate evidence must apply exactly {expected_applied} coherent units"
        )
    non_template = frame["selector_family"].astype(str) != "support_template"
    fetched = pd.to_numeric(frame["candidate_units"]).to_numpy(np.int64)
    if not np.all(fetched[non_template.to_numpy()] == expected_candidates):
        raise RuntimeError(
            f"non-template candidate evidence must fetch exactly {expected_candidates} units"
        )
    _validate_candidate_set_evidence(frame)
    flag_columns = (
        "activation_payload_already_read", "q2_payload_already_read",
        "abc_payload_already_read", "rerank_q2_gate_up_scale_payload_already_read",
    )
    flags: dict[str, np.ndarray] = {}
    for column in flag_columns:
        raw = np.asarray(frame[column].tolist(), dtype=object)
        if not all(isinstance(value, (bool, np.bool_)) for value in raw):
            raise RuntimeError(f"{column} is not strictly boolean")
        flags[column] = raw.astype(bool)
    values: dict[str, list[float]] = {
        "q4_units": [], "q4_macs": [], "workspace_bytes": [], "down_macs": [],
        "gram_units": [], "euclidean_gram_macs": [], "proxy_rank": [],
        "proxy_projection_macs": [], "proxy_gram_macs": [], "proxy_scales": [],
        "proxy_bytes": [], "gram_macs": [], "abc_macs_per_unit": [],
        "activation_bytes": [], "q2_bytes": [], "abc_bytes": [], "packet_bytes": [],
        "q2_parent_code_bytes": [], "q2_scale_bytes": [], "q4_scale_multiplications": [],
        "incremental_macs": [], "incremental_bytes": [],
    }
    lower_bounds: list[bool] = []
    down_required: list[bool] = []
    down_accounted: list[bool] = []
    gate_up_parent_required: list[bool] = []
    gate_up_parent_accounted: list[bool] = []
    bounds: list[str] = []
    unaccounted: list[str] = []
    for position, row in enumerate(frame.itertuples(index=False)):
        candidates = int(row.candidate_units)
        semantics = str(row.rerank_semantics)
        if semantics == PREDICTED:
            q4_units, workspace_bytes, gram_units, lower_bound = 0, 0, 0, False
        elif semantics == INDEPENDENT:
            q4_units, workspace_bytes, gram_units, lower_bound = candidates, 0, 0, False
        elif semantics == CONTAINED:
            q4_units = gram_units = candidates
            workspace_bytes, lower_bound = candidates * INPUTS * FP16_BYTES, True
        elif semantics == FULL_TARGET:
            q4_units = gram_units = UNITS
            workspace_bytes, lower_bound = UNITS * INPUTS * FP16_BYTES, True
        else:
            raise RuntimeError(f"unknown candidate rerank semantics: {semantics}")

        is_independent = semantics == INDEPENDENT
        has_interaction = gram_units > 0
        q4_macs = 2 * q4_units * INPUTS
        abc_macs_per_unit = ABC_SCORE_MACS_PER_UNIT if is_independent else 0
        down_macs = 2 * gram_units * INPUTS
        proxy_rank = LOCKED_PROXY_RANK if has_interaction else 0
        euclidean_gram_macs = gram_units * gram_units * INPUTS
        proxy_projection_macs = gram_units * INPUTS * proxy_rank
        proxy_gram_macs = gram_units * gram_units * proxy_rank
        proxy_scales = gram_units * gram_units
        proxy_bytes = INPUTS * proxy_rank * 4
        gram_macs = euclidean_gram_macs + proxy_projection_macs + proxy_gram_macs
        incremental_macs = (
            q4_macs + abc_macs_per_unit * q4_units + down_macs + gram_macs
        )
        activation_bytes = (
            ACTIVATION_PAYLOAD_BYTES
            if q4_units and not flags["activation_payload_already_read"][position] else 0
        )
        q2_bytes = (
            Q2_UNIT_FEATURE_BYTES
            if q4_units and not flags["q2_payload_already_read"][position] else 0
        )
        abc_bytes = (
            3_084
            if is_independent and not flags["abc_payload_already_read"][position] else 0
        )
        packet_bytes = q4_units * UNIT_PACKET_BYTES
        q2_parent_code_bytes = q4_units * Q2_GATE_UP_PARENT_CODE_BYTES_PER_UNIT
        q2_scale_bytes = (
            0 if flags["rerank_q2_gate_up_scale_payload_already_read"][position]
            else q4_units * Q2_GATE_UP_SCALE_BYTES_PER_UNIT
        )
        q4_scale_multiplications = (
            q4_units * Q4_RESPONSE_SCALE_MULTIPLICATIONS_PER_UNIT
        )
        incremental_bytes = (
            packet_bytes + activation_bytes + q2_bytes + abc_bytes
            + q2_parent_code_bytes + q2_scale_bytes
            + workspace_bytes + gram_units * gram_units * 4 + proxy_bytes
        )
        for key, value in (
            ("q4_units", q4_units), ("q4_macs", q4_macs),
            ("workspace_bytes", workspace_bytes), ("down_macs", down_macs),
            ("gram_units", gram_units), ("euclidean_gram_macs", euclidean_gram_macs),
            ("proxy_rank", proxy_rank), ("proxy_projection_macs", proxy_projection_macs),
            ("proxy_gram_macs", proxy_gram_macs), ("proxy_scales", proxy_scales),
            ("proxy_bytes", proxy_bytes), ("gram_macs", gram_macs),
            ("abc_macs_per_unit", abc_macs_per_unit),
            ("activation_bytes", activation_bytes), ("q2_bytes", q2_bytes),
            ("abc_bytes", abc_bytes), ("packet_bytes", packet_bytes),
            ("q2_parent_code_bytes", q2_parent_code_bytes),
            ("q2_scale_bytes", q2_scale_bytes),
            ("q4_scale_multiplications", q4_scale_multiplications),
            ("incremental_macs", incremental_macs),
            ("incremental_bytes", incremental_bytes),
        ):
            values[key].append(float(value))
        lower_bounds.append(lower_bound)
        down_required.append(has_interaction)
        down_accounted.append(not has_interaction)
        gate_up_parent_required.append(q4_units > 0)
        gate_up_parent_accounted.append(True)
        bounds.append(
            INTERACTION_RERANK_ACCOUNTING_BOUND
            if lower_bound else "complete_under_declared_logical_payload_and_MAC_conventions"
        )
        unaccounted.append(
            INTERACTION_RERANK_UNACCOUNTED
            if lower_bound else "none_under_declared_logical_payload_and_MAC_conventions"
        )
    arrays = {key: np.asarray(value, np.float64) for key, value in values.items()}
    for column, key in (
        ("rerank_q4_response_units", "q4_units"),
        ("rerank_q4_response_compute_macs", "q4_macs"),
        ("rerank_down_correction_compute_macs", "down_macs"),
        ("rerank_correction_workspace_bytes", "workspace_bytes"),
        ("rerank_interaction_gram_units", "gram_units"),
        ("rerank_interaction_gram_euclidean_compute_macs", "euclidean_gram_macs"),
        ("rerank_interaction_proxy_rank", "proxy_rank"),
        ("rerank_interaction_proxy_projection_compute_macs", "proxy_projection_macs"),
        ("rerank_interaction_proxy_gram_compute_macs", "proxy_gram_macs"),
        ("rerank_interaction_proxy_scale_multiplications", "proxy_scales"),
        ("rerank_proxy_metadata_bytes_read", "proxy_bytes"),
        ("rerank_interaction_gram_compute_macs", "gram_macs"),
        ("rerank_abc_score_macs_per_unit", "abc_macs_per_unit"),
        ("rerank_activation_payload_bytes_read", "activation_bytes"),
        ("rerank_q2_unit_feature_bytes_read", "q2_bytes"),
        ("rerank_abc_metadata_bytes_read", "abc_bytes"),
        ("rerank_candidate_packet_bytes_read", "packet_bytes"),
        ("rerank_q2_gate_up_parent_code_bytes_read", "q2_parent_code_bytes"),
        ("rerank_q2_gate_up_scale_bytes_read", "q2_scale_bytes"),
        ("rerank_q4_response_scale_multiplications", "q4_scale_multiplications"),
        ("rerank_incremental_compute_macs", "incremental_macs"),
        ("rerank_incremental_bytes_read", "incremental_bytes"),
    ):
        _assert_close(frame, column, arrays[key])
    if not np.array_equal(
        frame["rerank_accounting_is_lower_bound"].astype(bool).to_numpy(),
        np.asarray(lower_bounds, bool),
    ):
        raise RuntimeError("rerank lower-bound markers disagree with interface semantics")
    _assert_bool_array(
        frame, "rerank_resident_q2_down_payload_required", np.asarray(down_required, bool),
    )
    _assert_bool_array(
        frame, "rerank_resident_q2_down_payload_accounted", np.asarray(down_accounted, bool),
    )
    _assert_bool_array(
        frame, "rerank_q2_gate_up_parent_payload_required",
        np.asarray(gate_up_parent_required, bool),
    )
    _assert_bool_array(
        frame, "rerank_q2_gate_up_parent_payload_accounted",
        np.asarray(gate_up_parent_accounted, bool),
    )
    if list(frame["rerank_accounting_bound"].astype(str)) != bounds:
        raise RuntimeError("rerank accounting-bound declarations changed")
    if list(frame["rerank_unaccounted_overhead"].astype(str)) != unaccounted:
        raise RuntimeError("rerank unaccounted-overhead declarations changed")
    _assert_bool_array(
        frame, "rerank_packet_bytes_already_in_selector_bytes_read",
        np.ones(len(frame), bool),
    )
    if set(frame["rerank_abc_score_mac_convention"].astype(str)) != {ABC_SCORE_MAC_CONVENTION}:
        raise RuntimeError("ABC score MAC convention changed")
    if set(frame["rerank_non_mac_operations"].astype(str)) != {RERANK_NON_MAC_CONVENTION}:
        raise RuntimeError("rerank non-MAC operation declaration changed")
    if set(frame["rerank_candidate_packet_contents"].astype(str)) != {CANDIDATE_PACKET_CONTENTS}:
        raise RuntimeError("candidate packet contents declaration changed")
    return arrays


def recompute_accounting(
    kind: str, frame: pd.DataFrame, config: Mapping[str, Any],
    selector_index: Mapping[tuple[int, int, str], Mapping[str, Any]],
) -> pd.DataFrame:
    """Validate and append independently recomputed accounting columns."""
    result = frame.copy()
    weights = int(config["expert_weights"])
    page_bytes = int(config["page_size_bytes"])
    if kind == "hybrid":
        pages = pd.to_numeric(result["physical_pages"]).to_numpy(np.int64)
        physical_bytes = pages * page_bytes
        _assert_close(result, "physical_bytes", physical_bytes)
        _assert_close(result, "physical_bpw", 8.0 * physical_bytes / weights)
        _assert_close(result, "page_amplification", np.ones(len(result)))
        budget_pages = np.rint(pd.to_numeric(result["physical_budget_bpw"]) * weights / (8 * page_bytes))
        if np.any(pages > budget_pages):
            raise RuntimeError("hybrid oracle exceeds its charged page budget")
        result["physical_bytes_recomputed"] = physical_bytes
        result["physical_bpw_recomputed"] = 8.0 * physical_bytes / weights
        return result

    if kind == "template":
        fetched = pd.to_numeric(result["candidate_units"]).to_numpy(np.int64)
        applied = np.full(len(result), int(config["applied_units"]), np.int64)
    elif kind == "selector":
        fetched = pd.to_numeric(result["applied_units"]).to_numpy(np.int64)
        if not np.all(fetched == int(config["applied_units"])):
            raise RuntimeError(
                f"selector evidence must apply exactly {int(config['applied_units'])} units"
            )
        applied = fetched.copy()
    elif kind == "candidate":
        fetched = pd.to_numeric(result["candidate_units"]).to_numpy(np.int64)
        applied = pd.to_numeric(result["applied_units"]).to_numpy(np.int64)
    else:
        raise ValueError(f"unknown accounting kind: {kind}")
    pages = 3 * fetched
    physical_bytes = pages * page_bytes
    logical_actions = applied
    logical_bytes = 3 * applied * page_bytes
    physical_bpw = 8.0 * physical_bytes / weights
    logical_bpw = 8.0 * logical_bytes / weights
    amplification = physical_bytes / logical_bytes
    overfetch = fetched / applied
    for column, expected in (
        ("physical_pages", pages), ("physical_bytes", physical_bytes),
        ("physical_bpw", physical_bpw), ("candidate_overfetch", overfetch),
    ):
        if column in result:
            _assert_close(result, column, expected)
    if kind in ("selector", "candidate"):
        for column, expected in (
            ("logical_actions", logical_actions), ("logical_bytes", logical_bytes),
            ("logical_bpw", logical_bpw), ("page_amplification", amplification),
        ):
            _assert_close(result, column, expected)
    result["physical_pages_recomputed"] = pages
    result["physical_bytes_recomputed"] = physical_bytes
    result["physical_bpw_recomputed"] = physical_bpw
    result["logical_actions_recomputed"] = logical_actions
    result["logical_bytes_recomputed"] = logical_bytes
    result["logical_bpw_recomputed"] = logical_bpw
    result["page_amplification_recomputed"] = amplification
    result["candidate_overfetch_recomputed"] = overfetch

    rerank: dict[str, np.ndarray] | None = None
    if kind == "candidate":
        rerank = _validate_candidate_rerank_accounting(result, config)
        for column, key in (
            ("rerank_q4_response_units_recomputed", "q4_units"),
            ("rerank_q4_response_compute_macs_recomputed", "q4_macs"),
            ("rerank_down_correction_compute_macs_recomputed", "down_macs"),
            ("rerank_correction_workspace_bytes_recomputed", "workspace_bytes"),
            ("rerank_interaction_gram_units_recomputed", "gram_units"),
            ("rerank_interaction_gram_euclidean_compute_macs_recomputed", "euclidean_gram_macs"),
            ("rerank_interaction_proxy_rank_recomputed", "proxy_rank"),
            ("rerank_interaction_proxy_projection_compute_macs_recomputed", "proxy_projection_macs"),
            ("rerank_interaction_proxy_gram_compute_macs_recomputed", "proxy_gram_macs"),
            ("rerank_interaction_proxy_scale_multiplications_recomputed", "proxy_scales"),
            ("rerank_proxy_metadata_bytes_read_recomputed", "proxy_bytes"),
            ("rerank_interaction_gram_compute_macs_recomputed", "gram_macs"),
            ("rerank_abc_score_macs_per_unit_recomputed", "abc_macs_per_unit"),
            ("rerank_activation_payload_bytes_read_recomputed", "activation_bytes"),
            ("rerank_q2_unit_feature_bytes_read_recomputed", "q2_bytes"),
            ("rerank_abc_metadata_bytes_read_recomputed", "abc_bytes"),
            ("rerank_candidate_packet_bytes_read_recomputed", "packet_bytes"),
            ("rerank_q2_gate_up_parent_code_bytes_read_recomputed", "q2_parent_code_bytes"),
            ("rerank_q2_gate_up_scale_bytes_read_recomputed", "q2_scale_bytes"),
            ("rerank_q4_response_scale_multiplications_recomputed", "q4_scale_multiplications"),
            ("rerank_incremental_compute_macs_recomputed", "incremental_macs"),
            ("rerank_incremental_bytes_read_recomputed", "incremental_bytes"),
        ):
            result[column] = rerank[key]
        support_rows = result["selector_family"].astype(str) == "support_template"
        if support_rows.any():
            support = result.loc[support_rows]
            if "template_bank_metadata_bytes" not in support:
                raise RuntimeError("support-template candidate rows omit mask-bank metadata")
            base_compute = np.zeros(len(support), np.float64)
            base_bytes = (
                pd.to_numeric(support["template_bank_metadata_bytes"]).to_numpy(np.float64)
                + pd.to_numeric(support["physical_bytes"]).to_numpy(np.float64)
            )
            _assert_close(support, "selector_compute_macs", base_compute)
            _assert_close(support, "selector_bytes_read", base_bytes)
            _assert_bool_array(
                support, "activation_payload_already_read",
                np.zeros(len(support), bool),
            )
            _assert_bool_array(
                support, "q2_payload_already_read", np.zeros(len(support), bool),
            )
            _assert_bool_array(
                support, "abc_payload_already_read", np.zeros(len(support), bool),
            )
            _assert_bool_array(
                support, "rerank_q2_gate_up_scale_payload_already_read",
                np.zeros(len(support), bool),
            )
            support_incremental_macs = rerank["incremental_macs"][support_rows.to_numpy()]
            support_incremental_bytes = rerank["incremental_bytes"][support_rows.to_numpy()]
            support_packets = fetched[support_rows.to_numpy()] * 3 * page_bytes
            support_total_macs = base_compute + support_incremental_macs
            support_total_bytes = base_bytes + np.maximum(
                support_incremental_bytes - support_packets, 0,
            )
            _assert_close(support, "total_selector_compute_macs", support_total_macs)
            _assert_close(support, "total_selector_bytes_read", support_total_bytes)
            result.loc[support_rows, "total_selector_compute_macs_recomputed"] = support_total_macs
            result.loc[support_rows, "total_selector_bytes_read_recomputed"] = support_total_bytes

    if kind not in ("selector", "candidate"):
        return result
    selector_rows = result["selector_family"].astype(str) != "support_template"
    local = result.loc[selector_rows].copy()
    if local.empty:
        return result
    _validate_selector_abc_score_accounting(local)
    standalone_score = local["selector_family"].astype(str) != "direct_set_predictor"
    score_units = standalone_score.astype(np.int64).to_numpy() * UNITS
    result.loc[local.index, "selector_abc_score_units_recomputed"] = score_units
    result.loc[local.index, "selector_abc_score_compute_macs_recomputed"] = (
        score_units * ABC_SCORE_MACS_PER_UNIT
    )
    result.loc[local.index, "selector_abc_score_macs_per_unit_recomputed"] = (
        standalone_score.astype(np.int64).to_numpy() * ABC_SCORE_MACS_PER_UNIT
    )
    if kind == "candidate":
        _assert_bool_array(
            local, "rerank_q2_gate_up_scale_payload_already_read",
            _manifest_q2_gate_up_scales_already_read(local, selector_index),
        )
    expert, shared, macs, additions, scales, bytes_read, abc, activation_read = _manifest_accounting(
        local, selector_index, config,
    )
    activation_column = (
        "selector_activation_payload_already_read"
        if kind == "selector" else "activation_payload_already_read"
    )
    _assert_bool_array(local, activation_column, activation_read.astype(bool))
    if "selector_bytes_read_semantics" not in local or set(
        local["selector_bytes_read_semantics"].astype(str)
    ) != {SELECTOR_BYTE_SEMANTICS}:
        raise RuntimeError("selector logical-byte accounting semantics changed")
    if kind == "selector":
        _assert_close(
            local, "selector_activation_payload_bytes_read",
            activation_read.astype(np.int64) * ACTIVATION_PAYLOAD_BYTES,
        )
        _assert_close(
            local, "selector_q2_unit_feature_bytes_read",
            np.full(len(local), Q2_UNIT_FEATURE_BYTES),
        )
        _assert_close(local, "selector_abc_metadata_bytes_read", abc)
    else:
        _assert_bool_array(local, "q2_payload_already_read", np.ones(len(local), bool))
        _assert_bool_array(local, "abc_payload_already_read", np.ones(len(local), bool))
    amortized = shared / float(config["experts_per_layer"])
    metadata = expert + abc + amortized
    metadata_bpw = 8.0 * metadata / weights
    storage = (
        float(config["reference_bpw"]) + float(config["suffix_bpw_per_complete_representation"])
        + metadata_bpw
    ) / float(config["reference_bpw"])
    expected_selector_bytes = bytes_read + (fetched[selector_rows.to_numpy()] * 3 * page_bytes if kind == "candidate" else 0)
    for column, expected in (
        ("expert_specific_metadata_bytes", expert),
        ("layer_shared_metadata_bytes", shared),
        ("layer_shared_amortized_bytes_per_expert", amortized),
        ("abc_metadata_bytes", abc),
        ("selector_metadata_bytes_per_expert", metadata),
        ("selector_metadata_bpw", metadata_bpw),
        ("selector_compute_macs", macs),
        ("selector_compute_additions", additions),
        ("selector_scale_multiplications", scales),
        ("selector_bytes_read", expected_selector_bytes),
        ("suffix_storage_bpw", np.full(len(local), float(config["suffix_bpw_per_complete_representation"]))),
        ("storage_multiplier", storage),
    ):
        _assert_close(local, column, expected, atol=1e-8)

    total_macs = macs.copy()
    total_bytes = expected_selector_bytes.copy()
    if kind == "candidate":
        assert rerank is not None
        incremental_macs_array = rerank["incremental_macs"][selector_rows.to_numpy()]
        incremental_bytes_array = rerank["incremental_bytes"][selector_rows.to_numpy()]
        total_macs += incremental_macs_array
        packet_already_charged = fetched[selector_rows.to_numpy()] * 3 * page_bytes
        total_bytes += np.maximum(incremental_bytes_array - packet_already_charged, 0)
        _assert_close(local, "total_selector_compute_macs", total_macs)
        _assert_close(local, "total_selector_bytes_read", total_bytes)
    for column, values in (
        ("selector_metadata_bpw_recomputed", metadata_bpw),
        ("selector_compute_macs_recomputed", macs),
        ("selector_compute_additions_recomputed", additions),
        ("selector_scale_multiplications_recomputed", scales),
        ("selector_bytes_read_recomputed", expected_selector_bytes),
        ("activation_payload_already_read_recomputed", activation_read.astype(bool)),
        ("total_selector_compute_macs_recomputed", total_macs),
        ("total_selector_bytes_read_recomputed", total_bytes),
        ("storage_multiplier_recomputed", storage),
    ):
        result.loc[selector_rows, column] = values
    return result


def summarize(
    frame: pd.DataFrame, groups: Sequence[str], metrics: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, local in frame.groupby(list(groups), dropna=False, sort=True):
        values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(groups, values))
        row["unique_invocations"] = int(len(local[list(IDENTITY)].drop_duplicates()))
        row["unique_requests"] = int(local["request_id"].astype(str).nunique())
        for metric in metrics:
            if metric in local:
                numeric = pd.to_numeric(local[metric], errors="coerce").to_numpy(np.float64)
                finite = np.isfinite(numeric)
                # Some selector families do not produce response-reconstruction
                # diagnostics at all (the direct set predictor is the important
                # case), so parquet represents that whole group as NaN. Omit an
                # inapplicable all-NaN metric, but fail closed on a partially
                # missing group rather than silently changing its denominator.
                if not bool(np.any(finite)):
                    continue
                if not bool(np.all(finite)):
                    raise RuntimeError(
                        f"summary metric {metric!r} mixes finite and non-finite values "
                        f"within group {row!r}",
                    )
                row.update({
                    f"{metric}_{name}": value
                    for name, value in quantiles(numeric).items()
                })
        rows.append(row)
    return pd.DataFrame(rows)


def _promotion_exclusion_reasons(entry: Mapping[str, Any]) -> tuple[str, ...]:
    """Return explicit manifest reasons that make a selector control-only.

    This deliberately does not infer intent from accuracy or a family name.
    Promotion is blocked only by an explicit nonpromotable/control-only marker
    or by the named sampled-expert static-bias upper bound.
    """
    reasons: list[str] = []
    if entry.get("promotable") is False:
        reasons.append("explicit_promotable_false")
    if entry.get("control_only") is True:
        reasons.append("explicit_control_only_true")
    for key in ("promotion_role", "selector_role", "control_role"):
        role = str(entry.get(key, "")).strip().lower()
        if role in {"control_only", "diagnostic_only", "nonpromotable", "non_promotable"}:
            reasons.append(f"{key}:{role}")
    static_semantics = str(entry.get("static_unit_bias_semantics", "")).lower()
    if "transductive_sampled_expert_upper_bound" in static_semantics:
        reasons.append("transductive_sampled_expert_upper_bound")
    sampled_scope = str(entry.get("sampled_expert_scope", "")).lower()
    if "transductive_sampled_expert_diagnostic" in sampled_scope:
        reasons.append("transductive_sampled_expert_diagnostic")
    return tuple(sorted(set(reasons)))


def _selector_promotion_status(
    selector_index: Mapping[tuple[int, int, str], Mapping[str, Any]],
    selector_config_id: str,
    selector_family: str,
) -> tuple[bool, str]:
    entries = [
        entry
        for (_, _, config_id), entry in selector_index.items()
        if str(config_id) == str(selector_config_id)
        and str(entry.get("selector_family")) == str(selector_family)
    ]
    if not entries:
        # Synthetic unit fixtures may provide already-accounted frames without
        # a fit manifest. Production evidence is independently required to have
        # a manifest entry for every non-template selector row.
        return True, "none"
    reasons = sorted({reason for entry in entries for reason in _promotion_exclusion_reasons(entry)})
    return not reasons, ";".join(reasons) if reasons else "none"


def _complete_config_coverage(frame: pd.DataFrame, expected: int, keys: Sequence[str]) -> None:
    for key, local in frame.groupby(list(keys), dropna=False):
        count = len(local[list(IDENTITY)].drop_duplicates())
        if count != expected:
            raise RuntimeError(f"incomplete exact-validation coverage for {key}: {count} != {expected}")


def freeze_validation_promotions(
    evidence: ValidatedEvidence, accounted: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Apply frozen gates using exact-checkpoint validation evidence only."""
    for name, frame in accounted.items():
        if set(frame["evaluation_split"].astype(str)) != {"validation"}:
            raise RuntimeError(f"{name} contains non-validation evidence")
    config = evidence.config
    expected = int(evidence.run_facts["expected_unique_invocations"])
    gates = config["promotion_gates"]

    hybrid = accounted["hybrid"]
    one = hybrid[np.isclose(pd.to_numeric(hybrid["physical_budget_bpw"]), 1.0)]
    _complete_config_coverage(one, expected, ["selector_family", "physical_budget_bpw"])
    pr9_family = "coherent_independent_abc_pr9_control"
    exact_family = "coherent_exact_set_fixed_greedy_teacher"
    baseline_families = {pr9_family, exact_family}
    pr9 = one[one["selector_family"].astype(str) == pr9_family]
    exact = one[one["selector_family"].astype(str) == exact_family]
    challengers = one[~one["selector_family"].astype(str).isin(baseline_families)]
    if len(pr9) != expected or len(exact) != expected or challengers.empty:
        raise RuntimeError("one-bpw hybrid comparison lacks an explicit coherent baseline or challenger")
    hybrid_rows = []
    for family, local in challengers.groupby("selector_family", sort=True):
        paired = exact[[*IDENTITY, "recovery"]].merge(
            pr9[[*IDENTITY, "recovery"]], on=list(IDENTITY), how="inner",
            suffixes=("_exact_set_coherent", "_pr9_independent"), validate="one_to_one",
        ).merge(
            local[[*IDENTITY, "recovery"]], on=list(IDENTITY), how="inner",
            validate="one_to_one",
        ).rename(columns={"recovery": "recovery_hybrid"})
        if len(paired) != expected:
            raise RuntimeError(f"hybrid family {family} lacks paired one-bpw baseline evidence")
        pr9_q = quantiles(paired["recovery_pr9_independent"])
        exact_q = quantiles(paired["recovery_exact_set_coherent"])
        hybrid_q = quantiles(paired["recovery_hybrid"])
        delta_exact = quantiles(paired["recovery_hybrid"] - paired["recovery_exact_set_coherent"])
        delta_pr9 = quantiles(paired["recovery_hybrid"] - paired["recovery_pr9_independent"])
        hybrid_rows.append({
            "selector_family": family,
            "pr9_independent_median": pr9_q["median"],
            "exact_set_coherent_median": exact_q["median"],
            "hybrid_median": hybrid_q["median"],
            "median_recovery_improvement_vs_exact_set_coherent": (
                float(hybrid_q["median"]) - float(exact_q["median"])
            ),
            "paired_delta_vs_exact_set_p10": delta_exact["p10"],
            "paired_delta_vs_exact_set_median": delta_exact["median"],
            "paired_delta_vs_pr9_p10": delta_pr9["p10"],
            "paired_delta_vs_pr9_median": delta_pr9["median"],
        })
    hybrid_gate = float(gates.get("hybrid_min_median_improvement_at_1bpw", 0.005))
    hybrid_table = pd.DataFrame(hybrid_rows).sort_values(
        ["hybrid_median", "paired_delta_vs_exact_set_p10", "selector_family"],
        ascending=[False, False, True],
    )
    hybrid_table["passes"] = (
        hybrid_table["median_recovery_improvement_vs_exact_set_coherent"] >= hybrid_gate
    )
    hybrid_best = hybrid_table.iloc[0].to_dict()

    template = accounted["template"]
    template_groups = [
        "template_cohort", "pairing_semantics", "template_selector",
        "template_count_requested", "template_count_effective", "repair_count",
    ]
    template_table = summarize(
        template, template_groups,
        ["validation_oracle_path_utility_retained", "candidate_units", "candidate_overfetch"],
    )
    top1_gate = float(gates.get("template_top1_median_gain_retention", 0.95))
    top2_median_gate = float(gates.get("template_top2_median_gain_retention", 0.98))
    top2_p10_gate = float(gates.get("template_top2_p10_gain_retention", 0.95))
    template_table["candidate_max"] = template_table["candidate_units_max"]
    template_table["complete_validation_coverage"] = template_table["unique_invocations"] == expected
    is_top1 = template_table["template_selector"].astype(str).str.startswith("top1")
    is_top2 = template_table["template_selector"].astype(str).str.startswith("top2")
    template_table["passes"] = (
        is_top1
        & (template_table["validation_oracle_path_utility_retained_median"] >= top1_gate)
        & (template_table["candidate_max"] <= 256)
        & template_table["complete_validation_coverage"]
    ) | (
        is_top2
        & (template_table["validation_oracle_path_utility_retained_median"] >= top2_median_gate)
        & (template_table["validation_oracle_path_utility_retained_p10"] >= top2_p10_gate)
        & (template_table["candidate_max"] <= 256)
        & template_table["complete_validation_coverage"]
    )

    candidate = accounted["candidate"]
    deployable = candidate[candidate["selector_family"].astype(str) != "support_template"].copy()
    required = set(CANDIDATE_SEMANTICS)
    for key, local in deployable.groupby(["selector_config_id", "selector_family"], sort=True):
        observed = set(local["rerank_semantics"].astype(str))
        if observed != required:
            raise RuntimeError(f"candidate config {key} has semantics {sorted(observed)}, expected {sorted(required)}")
        _complete_config_coverage(local, expected, ["rerank_semantics"])
    candidate_table = summarize(
        deployable,
        ["selector_config_id", "selector_family", "rerank_semantics", "selection_regime", "rerank_role"],
        [
            "recovery", "oracle_set_gain_retention", "selector_metadata_bpw",
            "selector_compute_macs", "rerank_incremental_compute_macs",
            "total_selector_compute_macs", "storage_multiplier",
        ],
    )
    primary = candidate_table[candidate_table["rerank_semantics"].isin((PREDICTED, INDEPENDENT))].copy()
    eligibility = {
        (str(config_id), str(family)): _selector_promotion_status(
            evidence.selector_index, str(config_id), str(family),
        )
        for config_id, family in primary[["selector_config_id", "selector_family"]]
        .drop_duplicates().itertuples(index=False, name=None)
    }
    primary["promotion_eligible_selector"] = [
        eligibility[(str(config_id), str(family))][0]
        for config_id, family in primary[["selector_config_id", "selector_family"]]
        .itertuples(index=False, name=None)
    ]
    primary["promotion_exclusion_reason"] = [
        eligibility[(str(config_id), str(family))][1]
        for config_id, family in primary[["selector_config_id", "selector_family"]]
        .itertuples(index=False, name=None)
    ]
    primary["passes_recovery"] = (
        (primary["recovery_median"] >= float(gates["candidate_median_recovery"]))
        & (primary["recovery_p10"] >= float(gates["candidate_p10_recovery"]))
    )
    primary["passes_set_gain"] = (
        primary["oracle_set_gain_retention_p10"] >= float(gates["candidate_p10_set_gain_retention"])
    )
    primary["passes_metadata"] = (
        primary["selector_metadata_bpw_max"] <= float(gates["max_selector_metadata_bpw"])
    )
    primary["passes_compute"] = (
        primary["total_selector_compute_macs_max"] < float(gates["max_selector_compute_macs"])
    )
    primary["passes_storage"] = (
        primary["storage_multiplier_max"] < float(gates.get("max_storage_multiplier", 5.0))
    )
    primary["passes"] = primary[[
        "passes_recovery", "passes_set_gain", "passes_metadata", "passes_compute", "passes_storage",
    ]].all(axis=1) & primary["promotion_eligible_selector"]
    passing = primary[primary["passes"]].sort_values(
        ["oracle_set_gain_retention_p10", "recovery_p10", "recovery_median",
         "total_selector_compute_macs_max", "selector_metadata_bpw_max", "selector_config_id"],
        ascending=[False, False, False, True, True, True],
    )
    selected = passing.iloc[0].to_dict() if len(passing) else None

    template_pass = template_table[template_table["passes"]].sort_values(
        ["validation_oracle_path_utility_retained_p10", "validation_oracle_path_utility_retained_median",
         "candidate_max", "template_cohort", "template_selector"],
        ascending=[False, False, True, True, True],
    )
    template_best = template_pass.iloc[0].to_dict() if len(template_pass) else None
    payload = {
        "schema_version": 2,
        "generated_by": "oracle_study.set_utility_analysis.freeze_validation_promotions",
        "run_id": config["run_id"],
        "selection_split": "validation",
        "selection_capture_source": "exact_checkpoint",
        "test_scientific_rows_or_values_consulted": False,
        "test_split_and_request_metadata_audited": True,
        "test_metadata_audit": TEST_METADATA_AUDIT,
        "monolithic_capture_archives_physically_loaded_by_runner": True,
        "h4_predictor_trained_or_evaluated": False,
        "activation_timing": "actual_x_and_q2_features_at_H0; late_external_fetch; not_H4_prefetch",
        "config_sha256": evidence.config_sha256,
        "fit_manifest_sha256": evidence.run_facts["fit_manifest_sha256"],
        "validation_run_facts": {
            "expected_unique_invocations": expected,
            "observed_unique_invocations": int(evidence.run_facts["observed_unique_invocations"]),
            "request_separation_verified": True, "codec_locked": True,
        },
        "validation_evidence_sha256": dict(evidence.evidence_sha256),
        "gates": {
            "candidate_median_recovery": float(gates["candidate_median_recovery"]),
            "candidate_p10_recovery": float(gates["candidate_p10_recovery"]),
            "candidate_p10_set_gain_retention": float(gates["candidate_p10_set_gain_retention"]),
            "max_selector_metadata_bpw": float(gates["max_selector_metadata_bpw"]),
            "max_total_selector_compute_macs_strict": float(gates["max_selector_compute_macs"]),
            "hybrid_min_median_improvement_at_1bpw": hybrid_gate,
            "template_top1_median_gain_retention": top1_gate,
            "template_top2_median_gain_retention": top2_median_gate,
            "template_top2_p10_gain_retention": top2_p10_gate,
            "template_candidate_limit": 256,
        },
        "hybrid_oracle": {
            "status": "continue" if bool(hybrid_best["passes"]) else "retire",
            "best_validation_result": hybrid_best,
        },
        "template_existence_oracle": {
            "status": "regime_exists" if template_best else "negative",
            "best_validation_result": template_best,
            "deployable_classifier_evaluated": False,
            "deployable_selector_promotion_eligible": False,
        },
        "h0_late_candidate_policy": {
            "status": "promote_to_new_sealed_holdout" if selected else "stop",
            "selected_validation_configuration": selected,
            "latency_solution_claimed": False,
            "h4_prefetch_selector_claimed": False,
        },
        "contained_interaction_systems_oracle_used_for_promotion": False,
        "full_target_restricted_teacher_information_control_used_for_promotion": False,
        "high_rank_low_bit_diagnostic_used_for_promotion": False,
        "support_template_oracle_used_for_selector_promotion": False,
        "predicted_or_independent_primary_only": True,
        "new_sealed_holdout_required_before_confirmatory_claim": bool(selected),
    }
    return _plain(payload), {
        "hybrid_gate": hybrid_table, "template_gate": template_table,
        "candidate_interfaces": candidate_table, "selector_gate": primary,
    }


def validate_promotion_payload(payload: Mapping[str, Any], evidence: ValidatedEvidence) -> None:
    """Fail if a serialized decision no longer matches immutable evidence."""
    if (
        payload.get("schema_version") != 2
        or payload.get("selection_split") != "validation"
        or payload.get("test_scientific_rows_or_values_consulted") is not False
        or payload.get("test_split_and_request_metadata_audited") is not True
        or payload.get("test_metadata_audit") != TEST_METADATA_AUDIT
        or payload.get("monolithic_capture_archives_physically_loaded_by_runner") is not True
    ):
        raise RuntimeError("promotion payload is not validation-only")
    if payload.get("config_sha256") != evidence.config_sha256:
        raise RuntimeError("promotion payload config hash changed")
    if payload.get("validation_evidence_sha256") != evidence.evidence_sha256:
        raise RuntimeError("promotion payload evidence hashes changed")
    recomputed, _ = freeze_validation_promotions(
        evidence,
        {
            kind: recompute_accounting(kind, frame, evidence.config, evidence.selector_index)
            for kind, frame in evidence.frames.items()
        },
    )
    for key in (
        "hybrid_oracle", "template_existence_oracle", "h0_late_candidate_policy",
        "contained_interaction_systems_oracle_used_for_promotion",
        "full_target_restricted_teacher_information_control_used_for_promotion",
        "high_rank_low_bit_diagnostic_used_for_promotion",
        "support_template_oracle_used_for_selector_promotion",
        "predicted_or_independent_primary_only", "new_sealed_holdout_required_before_confirmatory_claim",
    ):
        if payload.get(key) != recomputed.get(key):
            raise RuntimeError(f"stored promotion decision changed for {key}")
