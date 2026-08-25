#!/usr/bin/env python3
"""Frozen post-outcome inference for nested D1 Experiment A.

This analyzer has no allocator entry point. It authenticates both the
finalized outcome tables and the exact sealed allocation manifest, then gives
each independent evaluation request one vote after averaging layers equally.

Confidence intervals and multiplicity are deliberately separate: the 95%
percentile bootstrap intervals are nominal and two-sided; Holm correction is
applied only to the four one-sided paired sign-flip p-values. A primary gate
requires both a nominal upper confidence limit below zero and a Holm-adjusted
p-value below 0.05.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "src"))

from oracle_study.d1_nested_artifacts import (  # noqa: E402
    canonical_sha256,
    decode_state,
    load_sealed_allocation_manifest,
    physical_subset,
    state_page_count,
)
from oracle_study.d1_nested_outcomes import (  # noqa: E402
    CACHE_FILE,
    CONTRAST_FILE,
    OUTCOME_SCHEMA,
    PROPAGATION_FILE,
    QUALITY_FILE,
    RUN_FACTS_FILE,
    ZERO_FILE,
    file_sha256,
    load_outcome_plan,
    request_layer_means,
    require_zero_dose_parity,
    route_mode_contrasts,
    validate_outcome_config,
    validate_plan_against_config,
)
from oracle_study.d1_nested_statistics import (  # noqa: E402
    holm_adjust,
    paired_distribution_summary,
    paired_request_cluster_bootstrap_ci,
    paired_request_differences,
    paired_sign_flip_test,
)
from run_same_host_causal_controls import atomic_json, atomic_parquet  # noqa: E402


REQUEST_MEANS = "d1_nested_outcome_request_means.parquet"
ROUTE_MODE_REQUEST_CONTRASTS = "d1_nested_outcome_route_mode_request_contrasts.parquet"
SUMMARY = "d1_nested_outcome_descriptive_summary.parquet"
PRIMARY_REQUEST_DIFFERENCES = "d1_nested_primary_request_differences.parquet"
PRIMARY_REQUEST_LAYER_DIFFERENCES = "d1_nested_primary_request_layer_differences.parquet"
PRIMARY_INFERENCE = "d1_nested_primary_inference.parquet"
PRIMARY_STRATA = "d1_nested_primary_strata.parquet"
SECONDARY_REQUEST_DIFFERENCES = "d1_nested_secondary_request_differences.parquet"
SECONDARY_SUMMARY = "d1_nested_secondary_descriptive_summary.parquet"
LIVE_FROZEN_DISTRIBUTIONS = "d1_nested_live_minus_fully_frozen_distributions.parquet"
ALLOCATION_ROWS = "d1_nested_allocation_rows.parquet"
ALLOCATION_COMPARISONS = "d1_nested_allocation_comparisons.parquet"
ALLOCATION_ARM_SUMMARY = "d1_nested_allocation_arm_summary.parquet"
ALLOCATION_SIDE_SUMMARY = "d1_nested_allocation_side_summary.parquet"
NESTING_ROWS = "d1_nested_nesting_rows.parquet"
PROMOTION = "d1_nested_experiment_a_promotion.json"
ANALYSIS_FACTS = "d1_nested_outcome_analysis_facts.json"
HOST_FACTS = "d1_nested_outcome_host_facts.json"

ARM_PR13 = "independent_pr13"
ARM_STRICT = "strict_frozen_candidate_incumbent_repair"
ARM_LOCAL = "nested_local_only"
ARM_D1 = "nested_d1_safe"
ARM_SEVERITY = "nested_d1_safe_violation_mass"

# This is the D1 next-layer mixer, not the injection layer's own mixer.
D1_NEXT_MIXER = {
    0: "linear_attention",
    1: "linear_attention",
    4: "linear_attention",
    6: "full_attention",
    12: "linear_attention",
    23: "linear_attention",
}

PRIMARY_SPECS = (
    ("nested_d1_safe_vs_nested_local_only_at_360", ARM_D1, 360, ARM_LOCAL, 360, False),
    ("nested_d1_safe_vs_nested_local_only_at_725", ARM_D1, 725, ARM_LOCAL, 725, False),
    (
        "nested_d1_safe_360_vs_independent_pr13_384_product_matched",
        ARM_D1, 360, ARM_PR13, 384, True,
    ),
    (
        "nested_d1_safe_725_vs_independent_pr13_749_product_matched",
        ARM_D1, 725, ARM_PR13, 749, True,
    ),
)
POPCOUNT = np.asarray([0, 1, 1, 2, 1, 2, 2, 3], dtype=np.int8)
MATERIALIZED_CONFIG_NAME = "qwen36_mxfp4_d1_nested_safe_allocation_20260825_v2.json"
MATERIALIZED_CONFIG_TOP_LEVEL_FIELDS = frozenset({
    "run_id", "schema", "experiment_stage", "authenticated_capture_protocol",
    "implementation_parent_commit", "base_all_layer_pr13_commit",
    "base_same_host_controls_commit", "checkpoint", "checkpoint_revision",
    "checkpoint_index_sha256", "checkpoint_config_sha256",
    "tokenizer_json_sha256", "selected_trees", "selected_tree_sha256",
    "fit_dir", "pr13_config", "request_source", "decode_position",
    "injection_layers", "d1_definition",
    "d2_d4_objectives_models_or_selector_inputs_present",
    "downstream_routes_and_terminal_kl_role", "traffic_pairs",
    "metadata_accounting", "physical_state", "arms",
    "strict_frozen_candidate_bank", "common_core", "local_guardrail",
    "d1_screen", "exact_d1_gate", "calibration", "allocation_phase",
    "outcome_phase", "statistics", "promotion_gates",
    "hardware_execution_path", "output_root", "experiment_b_started",
    "runtime_predictor_in_scope", "generated_rollout_in_scope",
    "joint_all_layer_compression_in_scope", "scientific_boundary",
    "configuration_provenance", "authenticated_capture_artifacts",
    "authenticated_capture_execution_stack",
    "authenticated_pr13_inputs", "reference_high_core_seal", "atomic_resume",
})
SECONDARY_FAMILIES = (
    ("nested_local_only_vs_independent_pr13_same_numeric_cap", ARM_LOCAL, ARM_PR13),
    ("strict_repair_vs_independent_pr13_same_numeric_cap", ARM_STRICT, ARM_PR13),
    ("severity_ablation_vs_nested_d1_safe", ARM_SEVERITY, ARM_D1),
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _validate_materialized_v2_config(config: Mapping[str, Any]) -> None:
    """Require the standalone v2 provenance and execution contract in full."""

    if set(config) != MATERIALIZED_CONFIG_TOP_LEVEL_FIELDS:
        raise RuntimeError("materialized v2 top-level schema changed")
    if config.get("run_id") != "qwen36_mxfp4_d1_nested_safe_allocation_20260825_v2":
        raise RuntimeError("analyzer requires the materialized v2 run ID")
    if config.get("output_root") != "/workspace/pr13_d1_nested_safe_oracle_20260825_v2":
        raise RuntimeError("materialized v2 output root changed")
    if tuple(config.get("arms", ())) != (
        ARM_PR13, ARM_STRICT, ARM_LOCAL, ARM_D1, ARM_SEVERITY,
    ):
        raise RuntimeError("materialized v2 arm declaration changed")
    if tuple(map(int, config.get("injection_layers", ()))) != tuple(D1_NEXT_MIXER):
        raise RuntimeError("materialized v2 injection layers changed")
    request_source = config.get("request_source")
    if not isinstance(request_source, Mapping) or (
        request_source.get("calibration_requests") != 32
        or request_source.get("evaluation_requests") != 96
        or request_source.get("total_requests") != 128
        or request_source.get("calibration_requests_per_domain") != 4
        or request_source.get("evaluation_requests_per_domain") != 12
        or request_source.get("development_overlap_required") != 0
    ):
        raise RuntimeError("materialized v2 request split changed")
    traffic = config.get("traffic_pairs")
    if not isinstance(traffic, list) or [
        (
            int(row["metadata_matched_pages_per_expert"]),
            int(row["pr13_reference_pages_per_expert"]),
        )
        for row in traffic
    ] != [(360, 384), (725, 749)]:
        raise RuntimeError("materialized v2 traffic pairs changed")
    if (
        config.get("d1_definition") != "same_decode_token_next_layer_router_only"
        or config.get("d2_d4_objectives_models_or_selector_inputs_present") is not False
        or config.get("downstream_routes_and_terminal_kl_role")
        != "sealed_outcome_phase_only"
        or config.get("runtime_predictor_in_scope") is not False
    ):
        raise RuntimeError("materialized v2 D1-only scientific boundary changed")
    expected_provenance = {
        "materialization": "standalone_full_config_v1",
        "parent_path": "configs/qwen36_mxfp4_d1_nested_safe_allocation_20260825_v1.json",
        "parent_sha256": "fef7b21ecdebcfb68d3f63c6a10575db4ce1163d72ca410e107626c34d061480",
    }
    if config.get("configuration_provenance") != expected_provenance:
        raise RuntimeError("materialized v2 parent provenance changed")
    capture = config.get("authenticated_capture_artifacts")
    required_capture = {
        "directory": "/workspace/pr13_d1_nested_safe_oracle_20260825_v1/captures_authenticated",
        "facts_path": "/workspace/pr13_d1_nested_safe_oracle_20260825_v1/captures_authenticated/capture_facts_all.json",
        "facts_schema": "pr13_d1_nested_exact_decode_capture_facts_v1",
        "facts_sha256": "65a1ae5dc5a149d8855f4bef4001181058ed66a4df5302877e6a0e859d543643",
        "split": "all",
        "requests": 128,
        "files": 128,
        "manifest_sha256": "a2934ea02e3ee9c01f566c12969e81baf83b0bf30ee18d0fadeb5bff46a0c546",
        "manifest_facts_sha256": "43732f6b4efd3d26df382936ea40fd1d5ea04d0ac9231810c10433f4dcbb9b2c",
    }
    if capture != required_capture:
        raise RuntimeError("materialized v2 authenticated capture inventory changed")
    expected_capture_stack = {
        "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "torch": "2.8.0+cu128",
        "cuda": "12.8",
        "route_operation": (
            "raw_logits_fp32_plus_anchor_fp32_cuda_then_softmax_fp32_then_"
            "torch_topk_k8_sorted"
        ),
        "captured_ordered_router_ids_authoritative": True,
        "allocation_runtime_must_match_gpu_torch_cuda_exactly": True,
        "all_selected_request_d1_targets_preflight_required": True,
    }
    if config.get("authenticated_capture_execution_stack") != expected_capture_stack:
        raise RuntimeError("materialized v2 capture execution stack changed")
    pr13 = config.get("authenticated_pr13_inputs")
    if not isinstance(pr13, Mapping) or (
        pr13.get("config_path") != config.get("pr13_config")
        or pr13.get("fit_dir") != config.get("fit_dir")
        or pr13.get("tree_sha256") != config.get("selected_tree_sha256")
        or pr13.get("checkpoint_config_sha256") != config.get("checkpoint_config_sha256")
        or pr13.get("checkpoint_index_sha256") != config.get("checkpoint_index_sha256")
        or pr13.get("config_sha256")
        != "56ec0ba55610a0542603d758597129fd3180fd944d973d1b367b7a0bc1a5d006"
        or pr13.get("factor_manifest_schema") != "average_rate_all_layers_distributed_v1"
        or pr13.get("factor_manifest_sha256")
        != "60204faca5d9d7974e035ff05088938af942fbd5e0d313c030dddc6748157069"
    ):
        raise RuntimeError("materialized v2 PR13 input inventory changed")
    expected_reference = {
        "required": True,
        "reference_arm": ARM_PR13,
        "reference_endpoint": "paired_reference_high",
        "applies_to_arms": [ARM_LOCAL, ARM_D1, ARM_SEVERITY],
        "literal_subset_and_state_hash_required": True,
    }
    if config.get("reference_high_core_seal") != expected_reference:
        raise RuntimeError("materialized v2 reference-high core seal changed")
    expected_resume = {
        "schema": "pr13_d1_nested_atomic_layer_resume_v1",
        "full_request_split_only": True,
        "facts_written_last": True,
        "scientific_files_per_layer": [
            "nested_candidates.pkl", "nested_candidate_metrics.parquet",
            "nested_slice_parity.parquet",
        ],
        "completed_global_requires_exact_configured_layer_set": True,
        "mixed_input_or_code_identity_rejected": True,
    }
    if config.get("atomic_resume") != expected_resume:
        raise RuntimeError("materialized v2 atomic resume contract changed")
    statistics = config.get("statistics")
    if not isinstance(statistics, Mapping) or (
        statistics.get("unit_of_independence") != "request"
        or statistics.get("request_weighting") != "equal"
        or statistics.get("within_request_layer_weighting") != "equal"
        or statistics.get("cluster_bootstrap_resamples") != 10_000
        or statistics.get("paired_sign_flip_resamples") != 100_000
        or statistics.get("bootstrap_seed") != 20260825
        or statistics.get("multiplicity") != "holm_within_primary_family"
    ):
        raise RuntimeError("materialized v2 statistical contract changed")
    gates = config.get("promotion_gates")
    if not isinstance(gates, Mapping) or (
        gates.get("all_engineering_gates_required") is not True
        or gates.get("nested_d1_safe_exact_crossing_rate_below_nested_local_only")
        is not True
        or gates.get("nested_local_safe_made_unsafe_allowed") != 0
        or gates.get("local_guardrail_violations_allowed") != 0
        or gates.get("experiment_b_starts_on_gate_failure") is not False
        or gates.get("paired_p95_kl_regression_maximum") != 0.0
        or gates.get("paired_single_request_kl_regression_maximum") != 0.002839
    ):
        raise RuntimeError("materialized v2 promotion gates changed")
    hardware = config.get("hardware_execution_path")
    if not isinstance(hardware, Mapping) or (
        hardware.get("cpu_worker_blas_threads") != 1
        or hardware.get("cpu_worker_processes_allowed_range") != [1, 64]
        or hardware.get("cpu_worker_processes_default") != 24
        or hardware.get("cpu_worker_process_count_is_semantics_inert") is not True
    ):
        raise RuntimeError("materialized v2 CPU execution contract changed")


def _load_authenticated_tables(root: Path) -> tuple[
    dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
]:
    facts = load_json(root / RUN_FACTS_FILE)
    if facts.get("schema") != OUTCOME_SCHEMA or facts.get("completed") is not True:
        raise RuntimeError("nested outcome run facts are incomplete")
    if (
        facts.get("allocation_manifest_was_authenticated_before_access") is not True
        or facts.get("allocation_selected_from_terminal_outcomes") is not False
        or facts.get("no_allocation_decisions_in_outcome_runner") is not True
        or facts.get("experiment_b_started") is not False
    ):
        raise RuntimeError("nested outcome allocation firewall changed")
    outputs = facts.get("outputs")
    if not isinstance(outputs, dict):
        raise RuntimeError("nested outcome output inventory is absent")
    host_record = facts.get("host_facts")
    host_path = root / HOST_FACTS
    if (
        not isinstance(host_record, dict)
        or host_record.get("file") != HOST_FACTS
        or not host_path.is_file()
        or host_record.get("sha256") != file_sha256(host_path)
        or int(host_record.get("bytes", -1)) != host_path.stat().st_size
    ):
        raise RuntimeError("nested outcome host provenance changed")
    frames = []
    for name in (QUALITY_FILE, PROPAGATION_FILE, CACHE_FILE, ZERO_FILE, CONTRAST_FILE):
        path = root / name
        record = outputs.get(name)
        if (
            not isinstance(record, dict)
            or not path.is_file()
            or record.get("sha256") != file_sha256(path)
            or int(record.get("bytes", -1)) != path.stat().st_size
        ):
            raise RuntimeError(f"finalized outcome artifact changed: {path}")
        frame = pd.read_parquet(path)
        if len(frame) != int(record.get("rows", -1)):
            raise RuntimeError(f"finalized outcome row count changed: {path}")
        frames.append(frame)
    return (facts, *frames)


def _load_authenticated_manifest(
    facts: Mapping[str, Any], manifest_path: Path, seal_path: Path,
) -> dict[str, Any]:
    """Authenticate the allocation seal and every outcome-side identity pin."""

    pins = facts.get("input_pins")
    if not isinstance(pins, Mapping):
        raise RuntimeError("outcome input pins are absent")
    required = (
        "allocation_manifest_sha256", "frozen_calibration_spec_sha256",
        "allocation_identities_sha256", "allocations",
    )
    if any(key not in pins for key in required):
        raise RuntimeError("outcome allocation pins are incomplete")
    manifest = load_sealed_allocation_manifest(
        manifest_path=Path(manifest_path),
        seal_path=Path(seal_path),
        expected_manifest_sha256=str(pins["allocation_manifest_sha256"]),
        expected_frozen_calibration_spec_sha256=str(
            pins["frozen_calibration_spec_sha256"]
        ),
    )
    allocations = manifest["allocations"]
    if len(allocations) != int(pins["allocations"]):
        raise RuntimeError("sealed allocation count differs from outcome pin")
    identities = canonical_sha256([
        [
            str(row["arm"]), int(row["rate"]), int(row["layer"]),
            str(row["request_id"]), str(row["selected_state"]["sha256"]),
        ]
        for row in sorted(
            allocations,
            key=lambda item: (
                str(item["arm"]), int(item["rate"]), int(item["layer"]),
                str(item["request_id"]),
            ),
        )
    ])
    if identities != pins["allocation_identities_sha256"]:
        raise RuntimeError("sealed allocation identities differ from outcome pin")
    return manifest


def _authenticate_config_and_grid(
    facts: Mapping[str, Any], config_path: Path,
    manifest_path: Path, seal_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pins = facts.get("input_pins")
    if not isinstance(pins, Mapping):
        raise RuntimeError("outcome input pins are absent")
    if file_sha256(config_path) != pins.get("allocation_config_sha256"):
        raise RuntimeError("allocation config bytes differ from outcome pin")
    config = load_json(config_path)
    validate_outcome_config(config)
    _validate_materialized_v2_config(config)
    if canonical_sha256(config) != pins.get("allocation_config_canonical_sha256"):
        raise RuntimeError("allocation config canonical digest differs from outcome pin")
    manifest = _load_authenticated_manifest(facts, manifest_path, seal_path)
    plan = load_outcome_plan(
        manifest_path=manifest_path,
        seal_path=seal_path,
        expected_manifest_sha256=str(pins["allocation_manifest_sha256"]),
        expected_frozen_calibration_spec_sha256=str(
            pins["frozen_calibration_spec_sha256"]
        ),
    )
    validate_plan_against_config(
        plan, config, config_file_sha256=file_sha256(config_path),
    )
    declared = tuple(config["statistics"]["primary_contrasts"])
    expected = tuple(spec[0] for spec in PRIMARY_SPECS)
    if declared != expected:
        raise RuntimeError("frozen primary contrast declaration changed")
    secondary = tuple(config["statistics"]["secondary_contrasts"])
    if secondary != tuple(family[0] for family in SECONDARY_FAMILIES):
        raise RuntimeError("frozen secondary contrast declaration changed")
    if set(map(int, config["injection_layers"])) != set(D1_NEXT_MIXER):
        raise RuntimeError("D1 next-layer mixer stratum map changed")
    return config, manifest


def _require_quality_allocation_identity(
    quality: pd.DataFrame,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
) -> None:
    """Require every logical outcome row to name its exact sealed state."""

    columns = [
        "split", "prompt_sha256", "domain", "position", "arm",
        "rate_pages_per_expert", "injection_layer", "request_id",
        "selected_state_sha256", "allocation_manifest_sha256",
    ]
    missing = sorted(set(columns).difference(quality.columns))
    if missing:
        raise RuntimeError(f"quality allocation identity columns are absent: {missing}")
    if not quality["allocation_manifest_sha256"].eq(manifest_sha256).all():
        raise RuntimeError("quality rows name a different allocation manifest")
    observed = set(quality[[
        "split", "prompt_sha256", "domain", "position", "arm",
        "rate_pages_per_expert", "injection_layer", "request_id",
        "selected_state_sha256",
    ]].drop_duplicates().itertuples(index=False, name=None))
    expected = {
        (
            str(row["split"]), str(row["prompt_sha256"]), str(row["domain"]),
            int(row["position"]), str(row["arm"]), int(row["rate"]),
            int(row["layer"]), str(row["request_id"]),
            str(row["selected_state"]["sha256"]),
        )
        for row in manifest["allocations"]
    }
    if observed != expected:
        raise RuntimeError("quality rows do not match the sealed allocation identities")


def _candidate_tail_aggregates(propagation: pd.DataFrame) -> pd.DataFrame:
    identity = [
        "split", "arm", "rate_pages_per_expert", "injection_layer",
        "request_id", "position", "route_mode",
    ]
    required = set(identity + [
        "route_membership_change_fraction", "router_mass_churn", "hidden_mse",
    ])
    missing = sorted(required.difference(propagation.columns))
    if missing:
        raise ValueError(f"propagation table is missing columns: {missing}")
    return propagation.groupby(identity, sort=True, dropna=False).agg(
        mean_downstream_route_membership_change_fraction=(
            "route_membership_change_fraction", "mean",
        ),
        mean_downstream_router_mass_churn=("router_mass_churn", "mean"),
        mean_downstream_hidden_mse=("hidden_mse", "mean"),
        maximum_downstream_hidden_mse=("hidden_mse", "max"),
        downstream_layers_observed=("observation_layer", "nunique"),
    ).reset_index()


def _candidate_cache_aggregates(cache: pd.DataFrame) -> pd.DataFrame:
    identity = [
        "split", "arm", "rate_pages_per_expert", "injection_layer",
        "request_id", "position", "route_mode",
    ]
    values = [
        "full_cache_mse", "attention_kv_mse", "deltanet_conv_mse",
        "deltanet_recurrent_mse",
    ]
    missing = sorted(set(identity + values).difference(cache.columns))
    if missing:
        raise ValueError(f"cache table is missing columns: {missing}")
    if cache.duplicated(identity).any():
        raise ValueError("cache table repeats a candidate identity")
    return cache[identity + values].copy()


def _contrast_request_means(contrasts: pd.DataFrame) -> pd.DataFrame:
    identity = [
        "split", "request_id", "arm", "rate_pages_per_expert", "injection_layer",
    ]
    values = [
        "live_minus_fully_frozen_logit_kl",
        "live_minus_frozen_set_live_weights_logit_kl",
        "frozen_set_live_weights_minus_fully_frozen_logit_kl",
    ]
    missing = sorted(set(identity + values).difference(contrasts.columns))
    if missing:
        raise ValueError(f"route-mode contrast table is missing columns: {missing}")
    if contrasts.empty or contrasts.duplicated(identity).any():
        raise ValueError("route-mode contrasts are empty or duplicated")
    layers = set(map(int, contrasts["injection_layer"].unique()))
    cells: set[tuple[str, int]] | None = None
    for request_id, request in contrasts.groupby("request_id", sort=False):
        observed = set(zip(
            request["arm"].astype(str), request["rate_pages_per_expert"].astype(int),
            strict=True,
        ))
        if cells is None:
            cells = observed
        elif observed != cells:
            raise ValueError(f"route-mode contrast grid differs for request {request_id}")
        for cell, part in request.groupby(["arm", "rate_pages_per_expert"], sort=False):
            if set(map(int, part["injection_layer"])) != layers:
                raise ValueError(
                    f"route-mode contrast layer grid differs for request {request_id}, {cell}"
                )
    keys = ["split", "request_id", "arm", "rate_pages_per_expert"]
    result = contrasts.groupby(keys, sort=True, dropna=False)[values].mean().reset_index()
    result["layers_averaged"] = len(layers)
    return result


def _summary_rows(request_means: pd.DataFrame, value_columns: Sequence[str]) -> pd.DataFrame:
    rows = []
    keys = ["split", "arm", "rate_pages_per_expert", "route_mode"]
    for identity, part in request_means.groupby(keys, sort=True, dropna=False):
        for metric in value_columns:
            values = part[metric].to_numpy(np.float64)
            rows.append({
                **dict(zip(keys, identity, strict=True)),
                "metric": metric,
                "requests": len(values),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p90": float(np.quantile(values, 0.90)),
                "p95": float(np.quantile(values, 0.95)),
                "maximum": float(np.max(values)),
            })
    return pd.DataFrame(rows)


def _state(raw: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(decode_state(raw), dtype=np.uint8)


def _bit_delta(left: np.ndarray, right: np.ndarray) -> tuple[int, int]:
    """Return pages removed from left and pages added to reach right."""

    left = np.asarray(left, np.uint8)
    right = np.asarray(right, np.uint8)
    removed = int(POPCOUNT[np.bitwise_and(left, np.bitwise_not(right) & 7)].sum())
    added = int(POPCOUNT[np.bitwise_and(right, np.bitwise_not(left) & 7)].sum())
    return removed, added


def _declared_rate_pairs(config: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    return tuple(
        (
            int(row["metadata_matched_pages_per_expert"]),
            int(row["pr13_reference_pages_per_expert"]),
        )
        for row in config["traffic_pairs"]
    )


def _require_reference_high_core_seals(
    config: Mapping[str, Any],
    raw_index: Mapping[tuple[str, int, int, str], Mapping[str, Any]],
    states: Mapping[tuple[str, int, int, str], np.ndarray],
) -> set[tuple[str, int, int, str]]:
    pairs = _declared_rate_pairs(config)
    rate_to_high = {
        rate: high for low, high in pairs for rate in (low, high)
    }
    expected_fields = {
        "schema", "reference_arm", "reference_rate", "reference_state_sha256",
        "reference_pages", "common_core_state_sha256", "common_core_pages",
        "common_core_is_literal_subset", "removed_pages", "added_pages",
    }
    verified: set[tuple[str, int, int, str]] = set()
    for key, raw in raw_index.items():
        arm, rate, layer, request_id = key
        if arm not in {ARM_LOCAL, ARM_D1, ARM_SEVERITY}:
            continue
        if rate not in rate_to_high:
            raise RuntimeError(f"nested allocation has undeclared rate {rate}")
        high_rate = rate_to_high[rate]
        reference_key = (ARM_PR13, high_rate, layer, request_id)
        reference = raw_index.get(reference_key)
        if reference is None:
            raise RuntimeError(f"nested allocation lacks paired PR13 high {reference_key}")
        for field in ("split", "prompt_sha256", "domain", "position"):
            if raw.get(field) != reference.get(field):
                raise RuntimeError(
                    f"nested core and paired PR13 high {field} identities differ"
                )
        if tuple(map(int, raw["expert_ids"])) != tuple(map(int, reference["expert_ids"])):
            raise RuntimeError("nested core and paired PR13 high expert rows differ")
        wrapper = raw.get("parameter")
        candidate = wrapper.get("candidate") if isinstance(wrapper, Mapping) else None
        seal = (
            candidate.get("paired_pr13_high_core_seal")
            if isinstance(candidate, Mapping) else None
        )
        if not isinstance(seal, Mapping) or set(seal) != expected_fields:
            raise RuntimeError("nested allocation lacks its complete paired PR13-high seal")
        core = _state(raw["freeze_state"])
        reference_state = states[reference_key]
        if not physical_subset(core, reference_state):
            raise RuntimeError("nested common core is not a subset of paired PR13 high")
        reference_pages = state_page_count(reference_state)
        core_pages = state_page_count(core)
        expected = {
            "schema": "pr13_d1_paired_reference_high_core_seal_v1",
            "reference_arm": ARM_PR13,
            "reference_rate": high_rate,
            "reference_state_sha256": str(reference["selected_state"]["sha256"]),
            "reference_pages": reference_pages,
            "common_core_state_sha256": str(raw["freeze_state"]["sha256"]),
            "common_core_pages": core_pages,
            "common_core_is_literal_subset": True,
            "removed_pages": reference_pages - core_pages,
            "added_pages": 0,
        }
        if dict(seal) != expected:
            raise RuntimeError("nested paired PR13-high seal does not reproduce exactly")
        verified.add(key)
    expected_cells = sum(
        key[0] in {ARM_LOCAL, ARM_D1, ARM_SEVERITY} for key in raw_index
    )
    if len(verified) != expected_cells:
        raise RuntimeError("not every nested allocation has a verified PR13-high seal")
    return verified


def _allocation_tables(
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return compact allocation, incumbent-comparison, and nesting tables."""

    rows: list[dict[str, Any]] = []
    states: dict[tuple[str, int, int, str], np.ndarray] = {}
    raw_index: dict[tuple[str, int, int, str], Mapping[str, Any]] = {}
    for raw in manifest["allocations"]:
        key = (
            str(raw["arm"]), int(raw["rate"]), int(raw["layer"]),
            str(raw["request_id"]),
        )
        selected = _state(raw["selected_state"])
        freeze = _state(raw["freeze_state"])
        states[key] = selected
        raw_index[key] = raw
        parameter = raw.get("parameter", {})
        candidate = parameter.get("candidate", {}) if isinstance(parameter, Mapping) else {}
        frozen = parameter.get("frozen", {}) if isinstance(parameter, Mapping) else {}
        endpoint = str(candidate.get("endpoint", "not_applicable"))
        stop_key = "low_stop_reason" if endpoint == "low" else "high_stop_reason"
        selected_pages = state_page_count(selected)
        freeze_pages = state_page_count(freeze)
        cap = int(raw["selected_state"]["page_cap"])
        crossings = int(raw["d1_crossings"])
        if (
            cap != key[1] * 8
            or int(raw["freeze_state"]["page_cap"]) != cap
            or selected_pages > cap
            or freeze_pages > selected_pages
            or not physical_subset(freeze, selected)
        ):
            raise RuntimeError("sealed allocation physical-state accounting changed")
        if not 0 <= crossings <= 8:
            raise RuntimeError("sealed allocation D1 crossing count is outside 0..8")
        expert_ids = tuple(map(int, raw["expert_ids"]))
        if len(expert_ids) != 8 or len(set(expert_ids)) != 8:
            raise RuntimeError("sealed allocation expert identity is invalid")
        rows.append({
            "split": str(raw["split"]),
            "arm": key[0],
            "rate_pages_per_expert": key[1],
            "injection_layer": key[2],
            "d1_next_layer_mixer": D1_NEXT_MIXER[key[2]],
            "request_id": key[3],
            "domain": str(raw.get("domain", "unknown")),
            "position": int(raw.get("position", -1)),
            "selected_state_sha256": str(raw["selected_state"]["sha256"]),
            "freeze_state_sha256": str(raw["freeze_state"]["sha256"]),
            "selected_pages": selected_pages,
            "page_cap": cap,
            "unused_pages": cap - selected_pages,
            "early_stop": selected_pages < cap,
            "freeze_pages": freeze_pages,
            "pages_added_after_freeze": selected_pages - freeze_pages,
            "d1_crossings": crossings,
            "d1_unsafe": crossings > 0,
            "d1_violation_depth": float(raw["d1_violation_depth"]),
            "d1_routing_mass_churn": float(raw["d1_routing_mass_churn"]),
            "d1_labeled_margin": float(raw["d1_labeled_margin"]),
            "local_damage": float(raw["local_damage"]),
            "all_q2_damage": float(raw["all_q2_damage"]),
            "repair_window_pages": int(frozen.get(
                "repair_window_pages", candidate.get("repair_window", -1),
            )),
            "eta": float(frozen.get("eta", candidate.get("eta", np.nan))),
            "endpoint": endpoint,
            "completion_stop_reason": str(candidate.get(
                stop_key, candidate.get("completion_stop_reason", "not_applicable"),
            )),
            "pair_fallback_high_guardrail_infeasible": bool(candidate.get(
                "pair_fallback_high_guardrail_infeasible", False,
            )),
        })
    verified_reference_seals = _require_reference_high_core_seals(
        config, raw_index, states,
    )
    allocations = pd.DataFrame(rows)
    allocations["paired_pr13_high_core_seal_authenticated"] = [
        (str(row.arm), int(row.rate_pages_per_expert), int(row.injection_layer),
         str(row.request_id)) in verified_reference_seals
        if str(row.arm) in {ARM_LOCAL, ARM_D1, ARM_SEVERITY} else True
        for row in allocations.itertuples(index=False)
    ]
    allocations = allocations.sort_values(
        ["request_id", "injection_layer", "rate_pages_per_expert", "arm"],
        kind="stable",
    ).reset_index(drop=True)

    comparisons: list[dict[str, Any]] = []
    for candidate_arm, reference_arm in (
        (ARM_STRICT, ARM_PR13),
        (ARM_D1, ARM_LOCAL),
        (ARM_SEVERITY, ARM_LOCAL),
    ):
        for key, candidate_state in states.items():
            arm, rate, layer, request_id = key
            if arm != candidate_arm:
                continue
            reference_key = (reference_arm, rate, layer, request_id)
            if reference_key not in states:
                raise RuntimeError(f"sealed allocation lacks incumbent {reference_key}")
            candidate_raw = raw_index[key]
            reference_raw = raw_index[reference_key]
            removed, added = _bit_delta(states[reference_key], candidate_state)
            reference_crossings = int(reference_raw["d1_crossings"])
            candidate_crossings = int(candidate_raw["d1_crossings"])
            wrapper = candidate_raw.get("parameter", {})
            parameter = wrapper.get("candidate", {}) if isinstance(wrapper, Mapping) else {}
            frozen = wrapper.get("frozen", {}) if isinstance(wrapper, Mapping) else {}
            eta = float(frozen.get("eta", parameter.get("eta", 0.0)))
            local_limit = (
                float(reference_raw["local_damage"])
                + eta * float(candidate_raw["all_q2_damage"])
            )
            endpoint = str(parameter.get("endpoint", "not_applicable"))
            fallback_eligible = bool(
                candidate_arm in {ARM_D1, ARM_SEVERITY} and endpoint == "low"
            )
            comparisons.append({
                "split": str(candidate_raw["split"]),
                "comparison_arm": candidate_arm,
                "incumbent_arm": reference_arm,
                "rate_pages_per_expert": rate,
                "injection_layer": layer,
                "d1_next_layer_mixer": D1_NEXT_MIXER[layer],
                "request_id": request_id,
                "domain": str(candidate_raw.get("domain", "unknown")),
                "endpoint": endpoint,
                "incumbent_d1_crossings": reference_crossings,
                "candidate_d1_crossings": candidate_crossings,
                "crossings_prevented": max(0, reference_crossings - candidate_crossings),
                "crossings_introduced": max(0, candidate_crossings - reference_crossings),
                "incumbent_d1_unsafe": reference_crossings > 0,
                "candidate_d1_unsafe": candidate_crossings > 0,
                "safe_to_unsafe": reference_crossings == 0 and candidate_crossings > 0,
                "unsafe_to_safe": reference_crossings > 0 and candidate_crossings == 0,
                "strict_crossing_reduction": candidate_crossings < reference_crossings,
                "candidate_d1_violation_depth": float(candidate_raw["d1_violation_depth"]),
                "candidate_d1_routing_mass_churn": float(
                    candidate_raw["d1_routing_mass_churn"]
                ),
                "candidate_local_damage": float(candidate_raw["local_damage"]),
                "incumbent_local_damage": float(reference_raw["local_damage"]),
                "local_damage_limit": local_limit,
                "local_guardrail_violation": bool(
                    candidate_arm in {ARM_D1, ARM_SEVERITY}
                    and float(candidate_raw["local_damage"]) > local_limit + 1e-12
                ),
                "physical_state_changed": removed + added > 0,
                "pages_removed_from_incumbent": removed,
                "pages_added_to_incumbent": added,
                "physical_page_hamming_distance": removed + added,
                "selected_pages": int(candidate_raw["selected_state"]["page_count"]),
                "page_cap": int(candidate_raw["selected_state"]["page_cap"]),
                "early_stop": int(candidate_raw["selected_state"]["page_count"])
                < int(candidate_raw["selected_state"]["page_cap"]),
                "pair_fallback_counted": bool(
                    fallback_eligible
                    and parameter.get("pair_fallback_high_guardrail_infeasible", False)
                ),
                "pair_fallback_eligible": fallback_eligible,
            })
    comparison_frame = pd.DataFrame(comparisons).sort_values(
        ["comparison_arm", "rate_pages_per_expert", "request_id", "injection_layer"],
        kind="stable",
    ).reset_index(drop=True)

    nesting: list[dict[str, Any]] = []
    for chain in manifest["nesting_chains"]:
        members = chain["members"]
        if len(members) != 2:
            raise RuntimeError("each nesting chain must contain exactly low and high")
        for low_ref, high_ref in zip(members, members[1:]):
            low_key = (
                str(low_ref["arm"]), int(low_ref["rate"]), int(low_ref["layer"]),
                str(low_ref["request_id"]),
            )
            high_key = (
                str(high_ref["arm"]), int(high_ref["rate"]), int(high_ref["layer"]),
                str(high_ref["request_id"]),
            )
            if (
                high_key[0] != low_key[0]
                or high_key[2] != low_key[2]
                or high_key[3] != low_key[3]
            ):
                raise RuntimeError("nesting chain changes arm/layer/request identity")
            low_state, high_state = states[low_key], states[high_key]
            removed, added = _bit_delta(low_state, high_state)
            low_raw, high_raw = raw_index[low_key], raw_index[high_key]
            for field in ("split", "prompt_sha256", "domain", "position"):
                if low_raw.get(field) != high_raw.get(field):
                    raise RuntimeError(f"nesting chain changes {field} identity")
            cores_identical = bool(np.array_equal(
                _state(low_raw["freeze_state"]), _state(high_raw["freeze_state"]),
            ))
            low_crossings = int(low_raw["d1_crossings"])
            nesting.append({
                "chain_name": str(chain["name"]),
                "split": str(low_raw["split"]),
                "arm": low_key[0],
                "low_rate": low_key[1],
                "high_rate": high_key[1],
                "injection_layer": low_key[2],
                "d1_next_layer_mixer": D1_NEXT_MIXER[low_key[2]],
                "request_id": low_key[3],
                "domain": str(low_raw.get("domain", "unknown")),
                "literal_physical_subset": bool(physical_subset(low_state, high_state)),
                "common_core_identical_low_to_high": cores_identical,
                "pages_removed_low_to_high": removed,
                "pages_added_low_to_high": added,
                "low_d1_crossings": low_crossings,
                "low_d1_safe": low_crossings == 0,
                "pages_added_after_d1_safe_low": added if low_crossings == 0 else 0,
                "additions_classified_as_after_safety": low_crossings == 0,
                "low_pages": state_page_count(low_state),
                "high_pages": state_page_count(high_state),
                "high_early_stop": state_page_count(high_state)
                < int(high_raw["selected_state"]["page_cap"]),
            })
    nesting_frame = pd.DataFrame(nesting).sort_values(
        ["arm", "low_rate", "request_id", "injection_layer"], kind="stable",
    ).reset_index(drop=True)
    return allocations, comparison_frame, nesting_frame


def _nesting_grid_audit(
    config: Mapping[str, Any],
    allocations: pd.DataFrame,
    nesting: pd.DataFrame,
) -> dict[str, Any]:
    nested_arms = (ARM_LOCAL, ARM_D1, ARM_SEVERITY)
    pairs = _declared_rate_pairs(config)
    layers = tuple(map(int, config["injection_layers"]))
    evaluation = allocations[allocations["split"].eq("evaluation")]
    requests = tuple(sorted(map(str, evaluation["request_id"].unique())))
    expected_requests = int(config["request_source"]["evaluation_requests"])
    if len(requests) != expected_requests:
        raise RuntimeError("nesting grid evaluation request count changed")
    if set(map(int, evaluation["injection_layer"].unique())) != set(layers):
        raise RuntimeError("nesting grid injection-layer set changed")
    identity_columns = [
        "arm", "low_rate", "high_rate", "injection_layer", "request_id",
    ]
    if nesting.duplicated(identity_columns).any():
        raise RuntimeError("nesting grid repeats an arm/pair/layer/request identity")
    observed = set(nesting[identity_columns].itertuples(index=False, name=None))
    expected = {
        (arm, low, high, layer, request)
        for arm in nested_arms
        for low, high in pairs
        for request in requests
        for layer in layers
    }
    if observed != expected or len(nesting) != len(expected):
        raise RuntimeError(
            "nesting grid is not the exact three-arm/two-pair/request/layer product"
        )
    if not nesting["split"].eq("evaluation").all():
        raise RuntimeError("nesting grid contains a non-evaluation chain")
    if not nesting["literal_physical_subset"].all() or not (
        nesting["pages_removed_low_to_high"].eq(0).all()
    ):
        raise RuntimeError("nesting grid contains a non-nested physical pair")
    nested_allocations = evaluation[evaluation["arm"].isin(nested_arms)]
    if not nested_allocations["paired_pr13_high_core_seal_authenticated"].all():
        raise RuntimeError("nesting grid contains an unauthenticated PR13-high core seal")
    if not nesting["common_core_identical_low_to_high"].all():
        raise RuntimeError("nesting grid changes its common core across endpoints")
    for low, high in pairs:
        pair = nested_allocations[
            nested_allocations["rate_pages_per_expert"].isin((low, high))
        ]
        core_counts = pair.groupby(
            ["injection_layer", "request_id"], sort=False,
        )["freeze_state_sha256"].nunique()
        if core_counts.ne(1).any():
            raise RuntimeError("nested arms do not share one common core per pair/cell")
    return {
        "nested_arms": len(nested_arms),
        "declared_rate_pairs": len(pairs),
        "evaluation_requests": len(requests),
        "injection_layers": len(layers),
        "expected_chains": len(expected),
        "observed_chains": len(nesting),
        "unique_identity_grid_complete": True,
        "rates_order_and_pair_membership_exact": True,
        "literal_physical_nesting_complete": True,
        "common_core_identity_complete": True,
        "paired_pr13_high_core_seals_complete": True,
    }


def _allocation_summary(comparisons: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    strata = [("overall", "all", comparisons)]
    for column, kind in (
        ("domain", "domain"),
        ("injection_layer", "layer"),
        ("d1_next_layer_mixer", "attention"),
    ):
        strata.extend(
            (kind, str(value), part)
            for value, part in comparisons.groupby(column, sort=True, dropna=False)
        )
    for stratum_type, stratum, frame in strata:
        for identity, part in frame.groupby(
            ["comparison_arm", "incumbent_arm", "rate_pages_per_expert"], sort=True,
        ):
            fallback_denominator = int(part["pair_fallback_eligible"].sum())
            rows.append({
                "stratum_type": stratum_type,
                "stratum": stratum,
                "comparison_arm": identity[0],
                "incumbent_arm": identity[1],
                "rate_pages_per_expert": int(identity[2]),
                "cells": len(part),
                "requests": int(part["request_id"].nunique()),
                "candidate_crossing_events": int(part["candidate_d1_unsafe"].sum()),
                "incumbent_crossing_events": int(part["incumbent_d1_unsafe"].sum()),
                "candidate_crossing_rate": float(part["candidate_d1_unsafe"].mean()),
                "incumbent_crossing_rate": float(part["incumbent_d1_unsafe"].mean()),
                "crossings_prevented": int(part["crossings_prevented"].sum()),
                "crossings_introduced": int(part["crossings_introduced"].sum()),
                "safe_to_unsafe_events": int(part["safe_to_unsafe"].sum()),
                "unsafe_to_safe_events": int(part["unsafe_to_safe"].sum()),
                "physical_state_changes": int(part["physical_state_changed"].sum()),
                "physical_state_change_fraction": float(
                    part["physical_state_changed"].mean()
                ),
                "mean_pages_removed_from_incumbent": float(
                    part["pages_removed_from_incumbent"].mean()
                ),
                "mean_pages_added_to_incumbent": float(
                    part["pages_added_to_incumbent"].mean()
                ),
                "maximum_page_hamming_distance": int(
                    part["physical_page_hamming_distance"].max()
                ),
                "local_guardrail_violations": int(
                    part["local_guardrail_violation"].sum()
                ),
                "early_stop_fraction": float(part["early_stop"].mean()),
                "pair_fallback_pairs": int(part["pair_fallback_counted"].sum()),
                "pair_fallback_eligible_pairs": fallback_denominator,
                "pair_fallback_fraction": (
                    float(part["pair_fallback_counted"].sum()) / fallback_denominator
                    if fallback_denominator else 0.0
                ),
            })
    return pd.DataFrame(rows).sort_values(
        ["stratum_type", "stratum", "comparison_arm", "rate_pages_per_expert"],
        kind="stable",
    ).reset_index(drop=True)


def _allocation_arm_summary(allocations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    strata = [("overall", "all", allocations)]
    for column, kind in (
        ("domain", "domain"),
        ("injection_layer", "layer"),
        ("d1_next_layer_mixer", "attention"),
    ):
        strata.extend(
            (kind, str(value), part)
            for value, part in allocations.groupby(column, sort=True, dropna=False)
        )
    for stratum_type, stratum, frame in strata:
        for (arm, rate), part in frame.groupby(
            ["arm", "rate_pages_per_expert"], sort=True,
        ):
            rows.append({
                "stratum_type": stratum_type,
                "stratum": stratum,
                "arm": arm,
                "rate_pages_per_expert": int(rate),
                "cells": len(part),
                "requests": int(part["request_id"].nunique()),
                "d1_crossing_events": int(part["d1_unsafe"].sum()),
                "d1_crossing_rate": float(part["d1_unsafe"].mean()),
                "d1_crossings_total": int(part["d1_crossings"].sum()),
                "mean_d1_violation_depth": float(part["d1_violation_depth"].mean()),
                "mean_d1_routing_mass_churn": float(
                    part["d1_routing_mass_churn"].mean()
                ),
                "mean_local_damage": float(part["local_damage"].mean()),
                "mean_selected_pages": float(part["selected_pages"].mean()),
                "mean_unused_pages": float(part["unused_pages"].mean()),
                "early_stop_cells": int(part["early_stop"].sum()),
                "early_stop_fraction": float(part["early_stop"].mean()),
            })
    return pd.DataFrame(rows).sort_values(
        ["stratum_type", "stratum", "arm", "rate_pages_per_expert"],
        kind="stable",
    ).reset_index(drop=True)


def _expected_arm_rates(config: Mapping[str, Any]) -> dict[str, list[int]]:
    rates = sorted({
        int(row[field])
        for row in config["traffic_pairs"]
        for field in (
            "metadata_matched_pages_per_expert",
            "pr13_reference_pages_per_expert",
        )
    })
    return {str(arm): rates for arm in config["arms"]}


def _primary_request_inference(
    request_means: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    statistics = config["statistics"]
    bootstrap_draws = int(statistics["cluster_bootstrap_resamples"])
    sign_draws = int(statistics["paired_sign_flip_resamples"])
    seed = int(statistics["bootstrap_seed"])
    if bootstrap_draws != 10_000 or sign_draws != 100_000:
        raise RuntimeError("frozen primary resample counts changed")
    live = request_means[
        request_means["split"].eq("evaluation")
        & request_means["route_mode"].eq("live")
    ].copy()
    expected = _expected_arm_rates(config)
    domains = live[["request_id", "domain"]].drop_duplicates()
    if domains.groupby("request_id")["domain"].nunique().gt(1).any():
        raise RuntimeError("request domain changed across primary cells")
    paired_parts: list[pd.DataFrame] = []
    inference_rows: list[dict[str, Any]] = []
    for index, spec in enumerate(PRIMARY_SPECS):
        name, candidate_arm, candidate_rate, reference_arm, reference_rate, product = spec
        paired = paired_request_differences(
            live,
            ["logit_kl"],
            candidate_arm=candidate_arm,
            reference_arm=reference_arm,
            candidate_rate=candidate_rate,
            reference_rate=reference_rate,
            split="evaluation",
            expected_arm_rates=expected,
        ).merge(domains, on="request_id", validate="one_to_one")
        paired.insert(0, "contrast", name)
        paired["route_mode"] = "live"
        paired["product_matched"] = bool(product)
        paired_parts.append(paired)
        value_col = "logit_kl_difference"
        distribution = paired_distribution_summary(paired, value_col=value_col)
        ci = paired_request_cluster_bootstrap_ci(
            paired,
            value_col=value_col,
            confidence=0.95,
            resamples=bootstrap_draws,
            seed=seed + index,
        )
        sign = paired_sign_flip_test(
            paired,
            value_col=value_col,
            alternative="less",
            monte_carlo_draws=sign_draws,
            seed=seed + 10_000 + index,
        )
        inference_rows.append({
            "contrast": name,
            "candidate_arm": candidate_arm,
            "candidate_rate": int(candidate_rate),
            "reference_arm": reference_arm,
            "reference_rate": int(reference_rate),
            "product_matched": bool(product),
            "route_mode": "live",
            "metric": "logit_kl",
            "difference_direction": "candidate_minus_reference",
            "requests": distribution.requests,
            "mean": distribution.mean,
            "median": distribution.median,
            "win_fraction": distribution.win_fraction,
            "p90_regression": distribution.p90,
            "p95_regression": distribution.p95,
            "maximum_regression": distribution.maximum,
            "bootstrap_estimate": ci.estimate,
            "bootstrap_lower": ci.lower,
            "bootstrap_upper": ci.upper,
            "bootstrap_confidence": ci.confidence,
            "bootstrap_resamples": ci.resamples,
            "bootstrap_seed": ci.seed,
            "bootstrap_interval": "two_sided_nominal_percentile",
            "bootstrap_interval_multiplicity_adjusted": False,
            "sign_flip_statistic": sign.statistic,
            "sign_flip_p_raw": sign.p_value,
            "sign_flip_alternative": sign.alternative,
            "sign_flip_method": sign.method,
            "sign_flip_permutations": sign.permutations,
            "sign_flip_nonzero_requests": sign.nonzero_requests,
            "sign_flip_seed": sign.seed,
        })
    inference = pd.DataFrame(inference_rows)
    inference["sign_flip_p_holm"] = holm_adjust(
        inference["sign_flip_p_raw"].to_numpy(np.float64)
    )
    inference["nominal_ci_upper_below_zero"] = inference["bootstrap_upper"] < 0.0
    inference["holm_sign_flip_p_below_0_05"] = inference["sign_flip_p_holm"] < 0.05
    inference["primary_inferential_gate_pass"] = (
        inference["nominal_ci_upper_below_zero"]
        & inference["holm_sign_flip_p_below_0_05"]
    )
    return pd.concat(paired_parts, ignore_index=True), inference


def _secondary_request_summaries(
    request_means: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Declared failure-interpretation contrasts; never used for promotion."""

    live = request_means[
        request_means["split"].eq("evaluation")
        & request_means["route_mode"].eq("live")
    ].copy()
    expected = _expected_arm_rates(config)
    domains = live[["request_id", "domain"]].drop_duplicates()
    paired_parts: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    rates = sorted({rate for declared in expected.values() for rate in declared})
    for family, candidate_arm, reference_arm in SECONDARY_FAMILIES:
        for rate in rates:
            paired = paired_request_differences(
                live,
                ["logit_kl"],
                candidate_arm=candidate_arm,
                reference_arm=reference_arm,
                candidate_rate=rate,
                reference_rate=rate,
                split="evaluation",
                expected_arm_rates=expected,
            ).merge(domains, on="request_id", validate="one_to_one")
            contrast = f"{family}_at_{rate}"
            paired.insert(0, "contrast", contrast)
            paired.insert(0, "contrast_family", family)
            paired["route_mode"] = "live"
            paired["analysis_role"] = "secondary_descriptive_failure_interpretation"
            paired_parts.append(paired)
            distribution = paired_distribution_summary(
                paired, value_col="logit_kl_difference",
            )
            summaries.append({
                "contrast_family": family,
                "contrast": contrast,
                "candidate_arm": candidate_arm,
                "reference_arm": reference_arm,
                "rate_pages_per_expert": rate,
                "requests": distribution.requests,
                "mean": distribution.mean,
                "median": distribution.median,
                "win_fraction": distribution.win_fraction,
                "p90_regression": distribution.p90,
                "p95_regression": distribution.p95,
                "maximum_regression": distribution.maximum,
                "inferential_status": "secondary_descriptive_no_promotion_use",
            })
    return (
        pd.concat(paired_parts, ignore_index=True),
        pd.DataFrame(summaries).sort_values(
            ["contrast_family", "rate_pages_per_expert"], kind="stable",
        ).reset_index(drop=True),
    )


def _live_frozen_distribution_summary(
    contrast_means: pd.DataFrame,
) -> pd.DataFrame:
    value_col = "live_minus_fully_frozen_logit_kl"
    rows: list[dict[str, Any]] = []
    for identity, part in contrast_means.groupby(
        ["split", "arm", "rate_pages_per_expert"], sort=True, dropna=False,
    ):
        distribution = paired_distribution_summary(part, value_col=value_col)
        rows.append({
            "split": identity[0],
            "arm": identity[1],
            "rate_pages_per_expert": int(identity[2]),
            "metric": value_col,
            "difference_direction": "live_minus_fully_frozen",
            "requests": distribution.requests,
            "mean": distribution.mean,
            "median": distribution.median,
            "negative_fraction": distribution.win_fraction,
            "p90": distribution.p90,
            "p95": distribution.p95,
            "maximum": distribution.maximum,
            "inferential_status": "paired_descriptive_failure_interpretation",
            "used_for_promotion": False,
        })
    return pd.DataFrame(rows).sort_values(
        ["split", "arm", "rate_pages_per_expert"], kind="stable",
    ).reset_index(drop=True)


def _layer_differences(
    quality: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    live = quality[
        quality["split"].eq("evaluation") & quality["route_mode"].eq("live")
    ].copy()
    rows: list[pd.DataFrame] = []
    keys = ["split", "request_id", "injection_layer", "position", "domain"]
    for spec in PRIMARY_SPECS:
        name, candidate_arm, candidate_rate, reference_arm, reference_rate, product = spec
        candidate = live[
            live["arm"].eq(candidate_arm)
            & live["rate_pages_per_expert"].eq(candidate_rate)
        ][keys + ["logit_kl"]].rename(columns={"logit_kl": "candidate_logit_kl"})
        reference = live[
            live["arm"].eq(reference_arm)
            & live["rate_pages_per_expert"].eq(reference_rate)
        ][keys + ["logit_kl"]].rename(columns={"logit_kl": "reference_logit_kl"})
        paired = candidate.merge(reference, on=keys, validate="one_to_one")
        if paired.empty:
            raise RuntimeError(f"primary layer contrast is empty: {name}")
        paired["contrast"] = name
        paired["candidate_arm"] = candidate_arm
        paired["candidate_rate"] = int(candidate_rate)
        paired["reference_arm"] = reference_arm
        paired["reference_rate"] = int(reference_rate)
        paired["product_matched"] = bool(product)
        paired["d1_next_layer_mixer"] = paired["injection_layer"].map(D1_NEXT_MIXER)
        if paired["d1_next_layer_mixer"].isna().any():
            raise RuntimeError("primary layer row lacks its D1 next-layer mixer label")
        paired["logit_kl_difference"] = (
            paired["candidate_logit_kl"] - paired["reference_logit_kl"]
        )
        rows.append(paired)
    result = pd.concat(rows, ignore_index=True)
    expected_rows = (
        len(PRIMARY_SPECS)
        * int(config["request_source"]["evaluation_requests"])
        * len(config["injection_layers"])
    )
    if len(result) != expected_rows:
        raise RuntimeError("primary request/layer contrast grid is incomplete")
    return result.sort_values(
        ["contrast", "request_id", "injection_layer"], kind="stable",
    ).reset_index(drop=True)


def _strata_summary(layer_differences: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for contrast, contrast_rows in layer_differences.groupby("contrast", sort=True):
        groups: list[tuple[str, str, pd.DataFrame]] = []
        for domain, part in contrast_rows.groupby("domain", sort=True):
            request = part.groupby(["split", "request_id"], sort=True)[
                "logit_kl_difference"
            ].mean().reset_index()
            groups.append(("domain", str(domain), request))
        for layer, part in contrast_rows.groupby("injection_layer", sort=True):
            groups.append(("layer", str(int(layer)), part[
                ["split", "request_id", "logit_kl_difference"]
            ]))
        for mixer, part in contrast_rows.groupby("d1_next_layer_mixer", sort=True):
            request = part.groupby(["split", "request_id"], sort=True)[
                "logit_kl_difference"
            ].mean().reset_index()
            groups.append(("attention", str(mixer), request))
        for kind, value, part in groups:
            summary = paired_distribution_summary(
                part, value_col="logit_kl_difference",
            )
            rows.append({
                "contrast": contrast,
                "stratum_type": kind,
                "stratum": value,
                "requests": summary.requests,
                "mean": summary.mean,
                "median": summary.median,
                "win_fraction": summary.win_fraction,
                "p90_regression": summary.p90,
                "p95_regression": summary.p95,
                "maximum_regression": summary.maximum,
                "inferential_status": "descriptive_only_no_multiplicity_claim",
            })
    return pd.DataFrame(rows).sort_values(
        ["contrast", "stratum_type", "stratum"], kind="stable",
    ).reset_index(drop=True)


def _promotion_payload(
    config: Mapping[str, Any],
    inference: pd.DataFrame,
    allocations: pd.DataFrame,
    comparisons: pd.DataFrame,
    nesting: pd.DataFrame,
) -> dict[str, Any]:
    d1 = comparisons[comparisons["comparison_arm"].eq(ARM_D1)]
    if d1.empty:
        raise RuntimeError("allocation comparisons contain no nested D1 arm")
    crossing_by_rate = {
        str(int(rate)): {
            "candidate_exact_crossings_sum": int(part["candidate_d1_crossings"].sum()),
            "incumbent_exact_crossings_sum": int(part["incumbent_d1_crossings"].sum()),
            "candidate_exact_crossings_mean": float(part["candidate_d1_crossings"].mean()),
            "incumbent_exact_crossings_mean": float(part["incumbent_d1_crossings"].mean()),
            "exact_crossings_nonworse": bool(
                part["candidate_d1_crossings"].sum()
                <= part["incumbent_d1_crossings"].sum()
            ),
            "candidate_binary_unsafe_rate": float(part["candidate_d1_unsafe"].mean()),
            "incumbent_binary_unsafe_rate": float(part["incumbent_d1_unsafe"].mean()),
            "binary_unsafe_strictly_below": bool(
                part["candidate_d1_unsafe"].mean()
                < part["incumbent_d1_unsafe"].mean()
            ),
            "binary_unsafe_nonworse": bool(
                part["candidate_d1_unsafe"].mean()
                <= part["incumbent_d1_unsafe"].mean()
            ),
        }
        for rate, part in d1.groupby("rate_pages_per_expert", sort=True)
    }
    nesting_audit = _nesting_grid_audit(config, allocations, nesting)
    candidate_exact_sum = int(d1["candidate_d1_crossings"].sum())
    incumbent_exact_sum = int(d1["incumbent_d1_crossings"].sum())
    candidate_exact_mean = float(d1["candidate_d1_crossings"].mean())
    incumbent_exact_mean = float(d1["incumbent_d1_crossings"].mean())
    candidate_binary_rate = float(d1["candidate_d1_unsafe"].mean())
    incumbent_binary_rate = float(d1["incumbent_d1_unsafe"].mean())
    engineering: dict[str, Any] = {
        "finalized_outcome_artifacts_authenticated": True,
        "all_zero_dose_route_and_cache_gates_bit_exact": True,
        "stored_route_mode_contrasts_reproduced_exactly": True,
        "sealed_allocation_authenticated": True,
        "nesting_grid": nesting_audit,
        "exact_nesting_identity_grid_complete": bool(
            nesting_audit["unique_identity_grid_complete"]
        ),
        "paired_pr13_high_core_seals_complete": bool(
            nesting_audit["paired_pr13_high_core_seals_complete"]
        ),
        "nested_common_core_identity_complete": bool(
            nesting_audit["common_core_identity_complete"]
        ),
        "literal_physical_nesting_rate": 1.0,
        "literal_physical_nesting_complete": True,
        "nested_d1_pooled_exact_crossings_sum": candidate_exact_sum,
        "nested_local_pooled_exact_crossings_sum": incumbent_exact_sum,
        "nested_d1_pooled_exact_crossings_mean": candidate_exact_mean,
        "nested_local_pooled_exact_crossings_mean": incumbent_exact_mean,
        "nested_d1_exact_crossings_sum_strictly_below_local_pooled": bool(
            candidate_exact_sum < incumbent_exact_sum
        ),
        "nested_d1_exact_crossings_mean_strictly_below_local_pooled": bool(
            candidate_exact_mean < incumbent_exact_mean
        ),
        "nested_d1_exact_crossings_nonworse_at_every_rate": bool(
            all(row["exact_crossings_nonworse"] for row in crossing_by_rate.values())
        ),
        "nested_d1_pooled_binary_unsafe_rate": candidate_binary_rate,
        "nested_local_pooled_binary_unsafe_rate": incumbent_binary_rate,
        "nested_d1_binary_unsafe_rate_strictly_below_local_pooled": bool(
            candidate_binary_rate < incumbent_binary_rate
        ),
        "nested_d1_binary_unsafe_rate_nonworse_at_every_rate": bool(
            all(row["binary_unsafe_nonworse"] for row in crossing_by_rate.values())
        ),
        "crossing_rates_by_rate": crossing_by_rate,
        "nested_local_safe_made_unsafe_events": int(d1["safe_to_unsafe"].sum()),
        "nested_local_safe_made_unsafe_gate_pass": bool(
            d1["safe_to_unsafe"].sum() == 0
        ),
        "local_guardrail_violations": int(d1["local_guardrail_violation"].sum()),
        "local_guardrail_gate_pass": bool(
            d1["local_guardrail_violation"].sum() == 0
        ),
    }
    engineering["all_engineering_gates_pass"] = bool(
        engineering["finalized_outcome_artifacts_authenticated"]
        and engineering["all_zero_dose_route_and_cache_gates_bit_exact"]
        and engineering["stored_route_mode_contrasts_reproduced_exactly"]
        and engineering["sealed_allocation_authenticated"]
        and engineering["exact_nesting_identity_grid_complete"]
        and engineering["paired_pr13_high_core_seals_complete"]
        and engineering["nested_common_core_identity_complete"]
        and engineering["literal_physical_nesting_complete"]
        and engineering[
            "nested_d1_exact_crossings_sum_strictly_below_local_pooled"
        ]
        and engineering[
            "nested_d1_exact_crossings_mean_strictly_below_local_pooled"
        ]
        and engineering["nested_d1_exact_crossings_nonworse_at_every_rate"]
        and engineering["nested_d1_binary_unsafe_rate_strictly_below_local_pooled"]
        and engineering["nested_d1_binary_unsafe_rate_nonworse_at_every_rate"]
        and engineering["nested_local_safe_made_unsafe_gate_pass"]
        and engineering["local_guardrail_gate_pass"]
    )
    indexed = {str(row["contrast"]): row for row in inference.to_dict("records")}
    product_names = [spec[0] for spec in PRIMARY_SPECS if spec[-1]]
    primary_gate = bool(inference["primary_inferential_gate_pass"].all())
    product_ci_gate = bool(all(
        float(indexed[name]["bootstrap_upper"]) < 0.0 for name in product_names
    ))
    p95_limit = float(config["promotion_gates"]["paired_p95_kl_regression_maximum"])
    maximum_limit = float(
        config["promotion_gates"]["paired_single_request_kl_regression_maximum"]
    )
    p95_gate = bool(inference["p95_regression"].le(p95_limit).all())
    maximum_gate = bool(inference["maximum_regression"].le(maximum_limit).all())
    inference_gates = {
        "primary_nominal_ci_and_holm_sign_flip_gate_pass": primary_gate,
        "product_matched_nominal_ci_upper_below_zero_at_both_rates": product_ci_gate,
        "paired_p95_regression_limit": p95_limit,
        "paired_p95_regression_gate_pass": p95_gate,
        "paired_single_request_regression_limit": maximum_limit,
        "paired_single_request_regression_gate_pass": maximum_gate,
    }
    passed = bool(
        engineering["all_engineering_gates_pass"]
        and primary_gate and product_ci_gate and p95_gate and maximum_gate
    )
    return {
        "schema": "pr13_d1_nested_experiment_a_promotion_v1",
        "status": "passed" if passed else "failed",
        "experiment_b_promotion_gate_passed": passed,
        "experiment_b_started": False,
        "decision_does_not_start_experiment_b": True,
        "primary_metric": "live current-token logit KL candidate-minus-reference",
        "unit_of_independence": "request",
        "within_request_layer_weighting": "equal",
        "confidence_interval_rule": (
            "two-sided nominal 95% percentile request-cluster bootstrap; "
            "intervals are not Holm-adjusted"
        ),
        "multiplicity_rule": (
            "Holm adjustment applies only to the four one-sided paired "
            "sign-flip p-values"
        ),
        "primary_gate_rule": (
            "each nominal bootstrap upper limit below zero and each "
            "Holm-adjusted one-sided sign-flip p-value below 0.05"
        ),
        "secondary_contrasts_role": (
            "descriptive failure interpretation only; never used for promotion"
        ),
        "live_minus_fully_frozen_role": (
            "paired descriptive failure interpretation only; never used for promotion"
        ),
        "engineering_gates": engineering,
        "inference_gates": inference_gates,
        "primary_contrasts": [
            {
                "contrast": str(row["contrast"]),
                "mean": float(row["mean"]),
                "nominal_bootstrap_lower": float(row["bootstrap_lower"]),
                "nominal_bootstrap_upper": float(row["bootstrap_upper"]),
                "raw_sign_flip_p": float(row["sign_flip_p_raw"]),
                "holm_sign_flip_p": float(row["sign_flip_p_holm"]),
                "p95_regression": float(row["p95_regression"]),
                "maximum_regression": float(row["maximum_regression"]),
                "pass": bool(row["primary_inferential_gate_pass"]),
            }
            for row in inference.to_dict("records")
        ],
    }


def analyze(
    root: Path,
    output: Path,
    *,
    config_path: Path,
    allocation_manifest: Path,
    allocation_seal: Path,
) -> None:
    facts, quality, propagation, cache, zero, stored_contrasts = (
        _load_authenticated_tables(root)
    )
    require_zero_dose_parity(zero)
    config, manifest = _authenticate_config_and_grid(
        facts, config_path, allocation_manifest, allocation_seal,
    )
    _require_quality_allocation_identity(
        quality, manifest, str(facts["input_pins"]["allocation_manifest_sha256"]),
    )
    recomputed_contrasts = route_mode_contrasts(quality)
    order = [
        "split", "arm", "rate_pages_per_expert", "injection_layer",
        "request_id", "position",
    ]
    stored = stored_contrasts.sort_values(order, kind="stable").reset_index(drop=True)
    recomputed = recomputed_contrasts.sort_values(order, kind="stable").reset_index(drop=True)
    if list(stored.columns) != list(recomputed.columns) or not stored.equals(recomputed):
        raise RuntimeError("stored route-mode contrasts do not reproduce exactly")

    identity = [
        "split", "arm", "rate_pages_per_expert", "injection_layer",
        "request_id", "position", "route_mode",
    ]
    enriched = quality.merge(
        _candidate_tail_aggregates(propagation), on=identity, validate="one_to_one",
    ).merge(
        _candidate_cache_aggregates(cache), on=identity, validate="one_to_one",
    )
    values = [
        "logit_kl", "delta_nll", "final_hidden_mse",
        "mean_downstream_route_membership_change_fraction",
        "mean_downstream_router_mass_churn", "mean_downstream_hidden_mse",
        "maximum_downstream_hidden_mse", "full_cache_mse", "attention_kv_mse",
        "deltanet_conv_mse", "deltanet_recurrent_mse",
    ]
    request_means = request_layer_means(enriched, values)
    request_domains = quality[["request_id", "domain"]].drop_duplicates()
    if request_domains.groupby("request_id")["domain"].nunique().gt(1).any():
        raise RuntimeError("one request appears in multiple domains")
    request_means = request_means.merge(
        request_domains, on="request_id", validate="many_to_one",
    )
    contrast_means = _contrast_request_means(recomputed_contrasts)
    summary = _summary_rows(request_means, values)
    allocation_rows, allocation_comparisons, nesting_rows = _allocation_tables(
        manifest, config,
    )
    allocation_arm_summary = _allocation_arm_summary(allocation_rows)
    allocation_summary = _allocation_summary(allocation_comparisons)
    paired, inference = _primary_request_inference(request_means, config)
    secondary_paired, secondary_summary = _secondary_request_summaries(
        request_means, config,
    )
    live_frozen_distributions = _live_frozen_distribution_summary(contrast_means)
    layer_differences = _layer_differences(quality, config)
    strata = _strata_summary(layer_differences)
    promotion = _promotion_payload(
        config, inference, allocation_rows, allocation_comparisons, nesting_rows,
    )

    output.mkdir(parents=True, exist_ok=True)
    tables = (
        (REQUEST_MEANS, request_means),
        (ROUTE_MODE_REQUEST_CONTRASTS, contrast_means),
        (SUMMARY, summary),
        (PRIMARY_REQUEST_DIFFERENCES, paired),
        (PRIMARY_REQUEST_LAYER_DIFFERENCES, layer_differences),
        (PRIMARY_INFERENCE, inference),
        (PRIMARY_STRATA, strata),
        (SECONDARY_REQUEST_DIFFERENCES, secondary_paired),
        (SECONDARY_SUMMARY, secondary_summary),
        (LIVE_FROZEN_DISTRIBUTIONS, live_frozen_distributions),
        (ALLOCATION_ROWS, allocation_rows),
        (ALLOCATION_COMPARISONS, allocation_comparisons),
        (ALLOCATION_ARM_SUMMARY, allocation_arm_summary),
        (ALLOCATION_SIDE_SUMMARY, allocation_summary),
        (NESTING_ROWS, nesting_rows),
    )
    for name, frame in tables:
        atomic_parquet(output / name, frame)
    atomic_json(output / PROMOTION, promotion)
    outputs = {
        name: {
            "sha256": file_sha256(output / name),
            "bytes": (output / name).stat().st_size,
            "rows": len(frame),
        }
        for name, frame in tables
    }
    outputs[PROMOTION] = {
        "sha256": file_sha256(output / PROMOTION),
        "bytes": (output / PROMOTION).stat().st_size,
    }
    atomic_json(output / ANALYSIS_FACTS, {
        "schema": "pr13_d1_nested_cached_decode_outcome_analysis_v2",
        "completed": True,
        "source_run_facts_sha256": file_sha256(root / RUN_FACTS_FILE),
        "source_allocation_manifest_sha256": facts["input_pins"][
            "allocation_manifest_sha256"
        ],
        "source_allocation_seal_sha256": file_sha256(allocation_seal),
        "source_allocation_config_sha256": file_sha256(config_path),
        "outputs": outputs,
        "unit_of_independence": "request",
        "within_request_layer_weighting": "equal",
        "primary_bootstrap": {
            "resamples": 10_000,
            "confidence": 0.95,
            "type": "two_sided_nominal_percentile",
            "multiplicity_adjusted": False,
        },
        "primary_sign_flip": {
            "alternative": "less",
            "monte_carlo_draws": 100_000,
            "multiplicity": "Holm within the four-test primary family",
        },
        "ci_vs_holm_clarification": (
            "The percentile confidence intervals are nominal and are not "
            "Holm-adjusted. Holm correction applies only to the four sign-flip "
            "p-values. Each primary gate requires both conditions."
        ),
        "attention_stratum_semantics": "same-token D1 next-layer mixer",
        "terminal_metrics_are_outcomes_only": True,
        "terminal_metrics_used_for_allocation_selection": False,
        "secondary_contrasts_are_descriptive_only": True,
        "secondary_contrasts_used_for_promotion": False,
        "live_minus_fully_frozen_distributions_are_descriptive_only": True,
        "live_minus_fully_frozen_distributions_used_for_promotion": False,
        "allocation_manifest_authenticated_again_before_analysis": True,
        "promotion_decision_does_not_start_experiment_b": True,
        "experiment_b_started": False,
        "promotion_status": promotion["status"],
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcome-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            EXPERIMENT / "configs"
            / MATERIALIZED_CONFIG_NAME
        ),
    )
    parser.add_argument("--allocation-manifest", type=Path, required=True)
    parser.add_argument("--allocation-seal", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.output = args.output or args.outcome_root / "analysis"
    return args


def main() -> None:
    args = parse_args()
    analyze(
        args.outcome_root,
        args.output,
        config_path=args.config,
        allocation_manifest=args.allocation_manifest,
        allocation_seal=args.allocation_seal,
    )


if __name__ == "__main__":
    main()
