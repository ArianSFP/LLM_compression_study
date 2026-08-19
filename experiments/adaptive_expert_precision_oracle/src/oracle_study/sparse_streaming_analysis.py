"""Fail-closed analysis and validation for the sparse-streaming study.

Promotion decisions in this module are functions of validation rows only.  Test
rows are joined to a frozen decision afterwards and can never change the
selected configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_FILES = {
    "neuron": "neuron_major_frontier.parquet",
    "shortlist": "activation_shortlist_containment.parquet",
    "labels": "exact_action_labels.parquet",
    "tile": "tile_streaming_frontier.parquet",
    "stability": "support_stability.parquet",
    "concentration": "activation_concentration.parquet",
}
RUN_FACTS = "run_facts.json"

_DEDUP_COMMON_ID = (
    "capture_source", "evaluation_split", "request_id", "position", "layer", "expert_id",
)

DEDUPLICATION_KEYS = {
    "neuron": _DEDUP_COMMON_ID + (
        "objective", "selection_regime", "action_family", "representation", "selector",
        "physical_budget_bpw",
    ),
    "shortlist": _DEDUP_COMMON_ID + (
        "objective", "selection_regime", "projection", "score_method", "shortlist_size",
        "refresh_block", "physical_budget_bpw",
    ),
    "labels": _DEDUP_COMMON_ID + ("projection", "action_rank"),
    "tile": _DEDUP_COMMON_ID + (
        "objective", "selection_regime", "action_family", "tile_shape", "selector",
        "physical_budget_bpw",
    ),
    "stability": _DEDUP_COMMON_ID + (
        "projection", "score_method", "shortlist_size", "relationship", "token_gap",
    ),
    "concentration": _DEDUP_COMMON_ID + ("top_k",),
}

SCIENTIFIC_EVIDENCE_COLUMNS = {
    "neuron": (
        "physical_pages", "physical_bytes", "logical_actions", "logical_bytes", "logical_bpw", "recovery",
        "suffix_storage_bpw", "selector_metadata_bpw",
        "selector_metadata_bytes_per_expert", "storage_multiplier", "page_amplification",
    ),
    "shortlist": (
        "physical_pages", "physical_bytes", "logical_actions", "logical_bytes", "logical_bpw", "recovery", "candidate_pages",
        "fetched_pages", "candidate_overfetch_factor", "exact_gain_retained",
        "exact_action_recall", "importance_weighted_recall", "recovery_diagonal",
        "recovery_exact", "recovery_constrained", "recovery_exact_matched_fetched_pages",
        "recovery_diagonal_matched_fetched_pages",
        "fraction_exact_over_diagonal_gain_retained_matched_fetched_pages",
        "suffix_storage_bpw", "selector_metadata_bpw",
        "selector_metadata_bytes_per_expert", "storage_multiplier", "page_amplification",
    ),
    "labels": (
        "action_id", "coordinate", "refinement_stage", "marginal_gain", "physical_page_id",
    ),
    "tile": (
        "physical_pages", "physical_bytes", "logical_actions", "logical_bytes", "logical_bpw", "recovery",
        "suffix_storage_bpw", "selector_metadata_bpw",
        "selector_metadata_bytes_per_expert", "storage_multiplier", "page_amplification",
    ),
    "stability": ("support_jaccard", "importance_weighted_overlap"),
    "concentration": (
        "top_k_energy_fraction", "exact_zero_fraction", "coordinates_for_90pct_energy",
    ),
}

COMMON_ID = (
    "capture_source",
    "evaluation_split",
    "request_id",
    "position",
    "layer",
    "expert_id",
)
RECOVERY_COMMON = COMMON_ID + (
    "objective",
    "selection_regime",
    "physical_budget_bpw",
    "physical_pages",
    "physical_bytes",
    "logical_actions",
    "logical_bytes",
    "logical_bpw",
    "recovery",
    "selector_runtime_ms",
    "selector_bytes_read",
    "suffix_storage_bpw",
    "selector_metadata_bpw",
    "selector_metadata_bytes_per_expert",
    "storage_multiplier",
    "page_amplification",
)


@dataclass(frozen=True)
class TableSpec:
    required: tuple[str, ...]
    numeric: tuple[str, ...]
    nonnegative: tuple[str, ...] = ()
    integer: tuple[str, ...] = ()


TABLE_SPECS: dict[str, TableSpec] = {
    "neuron": TableSpec(
        required=RECOVERY_COMMON + ("action_family", "representation", "selector"),
        numeric=(
            "position", "layer", "expert_id", "physical_budget_bpw", "physical_pages",
            "physical_bytes", "logical_actions", "logical_bytes", "logical_bpw", "recovery", "selector_runtime_ms",
            "selector_bytes_read", "storage_multiplier", "page_amplification",
            "suffix_storage_bpw", "selector_metadata_bpw", "selector_metadata_bytes_per_expert",
        ),
        nonnegative=(
            "position", "layer", "expert_id", "physical_budget_bpw", "physical_pages",
            "physical_bytes", "logical_actions", "logical_bytes", "logical_bpw", "selector_runtime_ms",
            "selector_bytes_read", "storage_multiplier", "page_amplification",
            "suffix_storage_bpw", "selector_metadata_bpw", "selector_metadata_bytes_per_expert",
        ),
        integer=(
            "position", "layer", "expert_id", "physical_pages", "physical_bytes",
            "logical_actions", "logical_bytes", "selector_bytes_read", "selector_metadata_bytes_per_expert",
        ),
    ),
    "shortlist": TableSpec(
        required=RECOVERY_COMMON + (
            "projection", "score_method", "shortlist_size",
            "refresh_block", "candidate_overfetch_factor", "physical_budget_bpw",
            "candidate_pages", "fetched_pages", "exact_gain_retained",
            "exact_action_recall", "importance_weighted_recall", "recovery_diagonal",
            "recovery_exact", "recovery_constrained", "selector_runtime_ms",
            "selector_bytes_read", "recovery_exact_matched_fetched_pages",
            "recovery_diagonal_matched_fetched_pages",
            "fraction_exact_over_diagonal_gain_retained_matched_fetched_pages",
        ),
        numeric=(
            "position", "layer", "expert_id", "physical_pages", "physical_bytes",
            "logical_actions", "logical_bytes", "logical_bpw", "recovery", "shortlist_size", "refresh_block",
            "candidate_overfetch_factor", "physical_budget_bpw", "candidate_pages",
            "fetched_pages", "exact_gain_retained", "exact_action_recall",
            "importance_weighted_recall", "recovery_diagonal", "recovery_exact",
            "recovery_constrained", "selector_runtime_ms", "selector_bytes_read",
            "recovery_exact_matched_fetched_pages",
            "recovery_diagonal_matched_fetched_pages",
            "fraction_exact_over_diagonal_gain_retained_matched_fetched_pages",
            "suffix_storage_bpw", "selector_metadata_bpw",
            "selector_metadata_bytes_per_expert", "storage_multiplier", "page_amplification",
        ),
        nonnegative=(
            "position", "layer", "expert_id", "physical_pages", "physical_bytes",
            "logical_actions", "logical_bytes", "logical_bpw", "shortlist_size", "refresh_block",
            "candidate_overfetch_factor", "physical_budget_bpw", "candidate_pages",
            "fetched_pages", "exact_action_recall", "importance_weighted_recall",
            "selector_runtime_ms", "selector_bytes_read", "suffix_storage_bpw",
            "selector_metadata_bpw", "selector_metadata_bytes_per_expert",
            "storage_multiplier", "page_amplification",
        ),
        integer=(
            "position", "layer", "expert_id", "physical_pages", "physical_bytes",
            "logical_actions", "logical_bytes", "shortlist_size", "refresh_block", "candidate_pages",
            "fetched_pages", "selector_bytes_read", "selector_metadata_bytes_per_expert",
        ),
    ),
    "labels": TableSpec(
        required=COMMON_ID + (
            "projection", "action_rank", "action_id", "coordinate", "refinement_stage",
            "marginal_gain", "physical_page_id",
        ),
        numeric=(
            "position", "layer", "expert_id", "action_rank", "action_id", "coordinate",
            "refinement_stage", "marginal_gain", "physical_page_id",
        ),
        nonnegative=(
            "position", "layer", "expert_id", "action_rank", "action_id", "coordinate",
            "refinement_stage", "physical_page_id",
        ),
        integer=(
            "position", "layer", "expert_id", "action_rank", "action_id", "coordinate",
            "refinement_stage", "physical_page_id",
        ),
    ),
    "tile": TableSpec(
        required=RECOVERY_COMMON + ("action_family", "tile_shape", "selector"),
        numeric=(
            "position", "layer", "expert_id", "physical_budget_bpw", "physical_pages",
            "physical_bytes", "logical_actions", "logical_bytes", "logical_bpw", "recovery", "selector_runtime_ms",
            "selector_bytes_read", "storage_multiplier", "page_amplification",
            "suffix_storage_bpw", "selector_metadata_bpw", "selector_metadata_bytes_per_expert",
        ),
        nonnegative=(
            "position", "layer", "expert_id", "physical_budget_bpw", "physical_pages",
            "physical_bytes", "logical_actions", "logical_bytes", "logical_bpw", "selector_runtime_ms",
            "selector_bytes_read", "storage_multiplier", "page_amplification",
            "suffix_storage_bpw", "selector_metadata_bpw", "selector_metadata_bytes_per_expert",
        ),
        integer=(
            "position", "layer", "expert_id", "physical_pages", "physical_bytes",
            "logical_actions", "logical_bytes", "selector_bytes_read", "selector_metadata_bytes_per_expert",
        ),
    ),
    "stability": TableSpec(
        required=COMMON_ID + (
            "projection", "score_method", "shortlist_size", "token_gap",
            "relationship", "support_jaccard", "importance_weighted_overlap",
        ),
        numeric=(
            "position", "layer", "expert_id", "shortlist_size", "token_gap",
            "support_jaccard", "importance_weighted_overlap",
        ),
        nonnegative=(
            "position", "layer", "expert_id", "shortlist_size", "token_gap",
            "support_jaccard", "importance_weighted_overlap",
        ),
        integer=("position", "layer", "expert_id", "shortlist_size", "token_gap"),
    ),
    "concentration": TableSpec(
        required=COMMON_ID + (
            "top_k", "top_k_energy_fraction", "exact_zero_fraction",
            "coordinates_for_90pct_energy",
        ),
        numeric=(
            "position", "layer", "expert_id", "top_k", "top_k_energy_fraction",
            "exact_zero_fraction", "coordinates_for_90pct_energy",
        ),
        nonnegative=(
            "position", "layer", "expert_id", "top_k", "top_k_energy_fraction",
            "exact_zero_fraction", "coordinates_for_90pct_energy",
        ),
        integer=(
            "position", "layer", "expert_id", "top_k", "coordinates_for_90pct_energy",
        ),
    ),
}

ALLOWED_SPLITS = {"train", "validation", "test"}
ALLOWED_OBJECTIVES = {
    "isolated_projection_qenergy",
    "exact_sequential_complete_expert_qenergy",
}
REGIME_PREFIX_TO_CATEGORY = {
    "h0_oracle_": "h0_oracle",
    "h0_resident_proxy_": "h0_proxy",
    "h0_deployable_metadata_": "deployable_h0_proxy",
}


class AnalysisValidationError(ValueError):
    """Raised when an input cannot support a scientifically valid analysis."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def schema_document() -> dict[str, Any]:
    return {
        "version": 1,
        "input_files": INPUT_FILES,
        "run_facts": {
            "required": [
                "run_id", "config_sha256", "reference", "locked_tree_sha256",
                "locked_capture_sha256", "page_size_bytes", "codec_locked",
                "request_separation_verified", "run_mode", "completed",
                "expected_unique_invocations", "observed_unique_invocations",
            ],
            "reference_required": ["revision", "config_sha256", "index_sha256"],
        },
        "tables": {
            name: {
                "required": list(spec.required),
                "numeric": list(spec.numeric),
                "nonnegative": list(spec.nonnegative),
                "integer": list(spec.integer),
            }
            for name, spec in TABLE_SPECS.items()
        },
        "enums": {
            "evaluation_split": sorted(ALLOWED_SPLITS),
            "objective": sorted(ALLOWED_OBJECTIVES),
            "selection_regime_prefixes": sorted(REGIME_PREFIX_TO_CATEGORY),
            "derived_selection_category": sorted(set(REGIME_PREFIX_TO_CATEGORY.values())),
        },
    }


def _require_keys(value: Mapping[str, Any], keys: Iterable[str], context: str) -> None:
    missing = sorted(set(keys).difference(value))
    if missing:
        raise AnalysisValidationError(f"{context} missing required keys: {missing}")


def validate_run_facts(facts: Mapping[str, Any], config: Mapping[str, Any], config_sha256: str) -> None:
    _require_keys(
        facts,
        (
            "run_id", "config_sha256", "reference", "locked_tree_sha256",
            "locked_capture_sha256", "page_size_bytes", "codec_locked",
            "request_separation_verified", "run_mode", "completed",
            "expected_unique_invocations", "observed_unique_invocations",
        ),
        RUN_FACTS,
    )
    reference = facts["reference"]
    if not isinstance(reference, Mapping):
        raise AnalysisValidationError("run_facts.json reference must be an object")
    _require_keys(reference, ("revision", "config_sha256", "index_sha256"), "run_facts.reference")
    expected_ref = config["reference"]
    comparisons = {
        "config_sha256": (facts["config_sha256"], config_sha256),
        "reference.revision": (reference["revision"], expected_ref["revision"]),
        "reference.config_sha256": (reference["config_sha256"], expected_ref["config_sha256"]),
        "reference.index_sha256": (reference["index_sha256"], expected_ref["index_sha256"]),
        "locked_tree_sha256": (facts["locked_tree_sha256"], config["locked_tree_sha256"]),
        "locked_capture_sha256": (facts["locked_capture_sha256"], config["locked_capture_sha256"]),
        "page_size_bytes": (facts["page_size_bytes"], config["page_size_bytes"]),
    }
    mismatches = [name for name, (actual, expected) in comparisons.items() if actual != expected]
    if mismatches:
        details = ", ".join(
            f"{name}: got {comparisons[name][0]!r}, expected {comparisons[name][1]!r}"
            for name in mismatches
        )
        raise AnalysisValidationError(f"run provenance mismatch: {details}")
    if facts["codec_locked"] is not True:
        raise AnalysisValidationError("run_facts.json must assert codec_locked=true")
    if facts["request_separation_verified"] is not True:
        raise AnalysisValidationError("run_facts.json must assert request_separation_verified=true")
    if facts["run_mode"] not in {"pilot", "full"}:
        raise AnalysisValidationError("run_facts.json run_mode must be pilot or full")
    if facts["completed"] is not True:
        raise AnalysisValidationError("run_facts.json completed must be true; partial checkpoints are rejected")
    expected_invocations = int(facts["expected_unique_invocations"])
    observed_invocations = int(facts["observed_unique_invocations"])
    if expected_invocations < 0 or observed_invocations < 0 or observed_invocations > expected_invocations:
        raise AnalysisValidationError("run_facts invocation counts must satisfy 0 <= observed <= expected")
    if observed_invocations != expected_invocations:
        raise AnalysisValidationError(
            "completed run_facts must have observed_unique_invocations == expected_unique_invocations"
        )


def validate_table(name: str, frame: pd.DataFrame, page_size_bytes: int) -> None:
    spec = TABLE_SPECS[name]
    if frame.empty:
        raise AnalysisValidationError(f"{INPUT_FILES[name]} is empty")
    missing = sorted(set(spec.required).difference(frame.columns))
    if missing:
        raise AnalysisValidationError(f"{INPUT_FILES[name]} missing required columns: {missing}")
    if frame[list(spec.required)].isna().any().any():
        bad = frame[list(spec.required)].columns[frame[list(spec.required)].isna().any()].tolist()
        raise AnalysisValidationError(f"{INPUT_FILES[name]} contains null required values in: {bad}")
    for column in spec.numeric:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise AnalysisValidationError(f"{INPUT_FILES[name]}.{column} must be finite numeric")
        if column in spec.nonnegative and (values < 0).any():
            raise AnalysisValidationError(f"{INPUT_FILES[name]}.{column} must be nonnegative")
        if column in spec.integer and not np.equal(values, np.floor(values)).all():
            raise AnalysisValidationError(f"{INPUT_FILES[name]}.{column} must be integral")
    splits = set(frame["evaluation_split"].astype(str).unique())
    if not splits.issubset(ALLOWED_SPLITS):
        raise AnalysisValidationError(f"{INPUT_FILES[name]} has invalid evaluation_split values: {sorted(splits - ALLOWED_SPLITS)}")
    if "objective" in frame:
        objectives = set(frame["objective"].astype(str).unique())
        if not objectives.issubset(ALLOWED_OBJECTIVES):
            raise AnalysisValidationError(f"{INPUT_FILES[name]} has invalid objective values: {sorted(objectives - ALLOWED_OBJECTIVES)}")
    if "selection_regime" in frame:
        regimes = set(frame["selection_regime"].astype(str).unique())
        invalid = sorted(
            regime for regime in regimes
            if not any(regime.startswith(prefix) for prefix in REGIME_PREFIX_TO_CATEGORY)
        )
        if invalid:
            raise AnalysisValidationError(
                f"{INPUT_FILES[name]} has invalid detailed selection_regime values: {invalid}"
            )
    if name in {"neuron", "tile", "shortlist"}:
        expected = frame["physical_pages"].astype(np.int64) * int(page_size_bytes)
        if not np.array_equal(expected.to_numpy(), frame["physical_bytes"].astype(np.int64).to_numpy()):
            raise AnalysisValidationError(f"{INPUT_FILES[name]} physical_bytes must equal physical_pages * page_size_bytes")
        logical_denominator_weights = 2048 * 512 if name == "shortlist" else 3 * 2048 * 512
        expected_logical_bpw = (
            frame["logical_bytes"].astype(np.float64) * 8.0 / logical_denominator_weights
        )
        if not np.allclose(
            frame["logical_bpw"].to_numpy(dtype=float),
            expected_logical_bpw.to_numpy(dtype=float),
            rtol=0,
            atol=1e-12,
        ):
            raise AnalysisValidationError(
                f"{INPUT_FILES[name]} logical_bpw must equal "
                "8 * logical_bytes / the declared expert-or-projection weight denominator"
            )
        expected_amplification = frame["physical_bytes"].to_numpy(dtype=float) / np.maximum(
            frame["logical_bytes"].to_numpy(dtype=float), 1.0
        )
        if not np.allclose(
            frame["page_amplification"].to_numpy(dtype=float),
            expected_amplification,
            rtol=0,
            atol=1e-12,
        ):
            raise AnalysisValidationError(
                f"{INPUT_FILES[name]} page_amplification must equal "
                "physical_bytes / max(logical_bytes, 1); fetched shortlist pages may exceed applied payload"
            )
    if name == "shortlist":
        if not np.array_equal(
            frame["physical_pages"].astype(np.int64).to_numpy(),
            frame["fetched_pages"].astype(np.int64).to_numpy(),
        ):
            raise AnalysisValidationError(
                "activation_shortlist_containment.parquet physical_pages must equal fetched_pages; "
                "candidate prefetch is charged even when an action is not applied"
            )
    if name == "labels" and not set(frame["refinement_stage"].astype(int).unique()).issubset({1, 2}):
        raise AnalysisValidationError("exact_action_labels.parquet refinement_stage must be 1 or 2")
    if name == "stability":
        allowed_relationships = {"same_invocation_gate_up", "adjacent_token_same_sequence"}
        relationships = set(frame["relationship"].astype(str).unique())
        if not relationships.issubset(allowed_relationships):
            raise AnalysisValidationError(
                f"support stability has invalid relationships: {sorted(relationships - allowed_relationships)}"
            )
        same = frame["relationship"] == "same_invocation_gate_up"
        adjacent = frame["relationship"] == "adjacent_token_same_sequence"
        if (frame.loc[same, "token_gap"] != 0).any():
            raise AnalysisValidationError("same-invocation gate/up overlap must use token_gap=0")
        if (frame.loc[adjacent, "token_gap"] != 1).any():
            raise AnalysisValidationError("adjacent-token stability must use token_gap=1")
        if (frame.loc[adjacent, "capture_source"] != "exact_checkpoint").any():
            raise AnalysisValidationError(
                "adjacent-token stability claims are valid only on the fresh exact-checkpoint capture"
            )
    if name == "concentration":
        for column in ("top_k_energy_fraction", "exact_zero_fraction"):
            if (frame[column] > 1.0 + 1e-12).any():
                raise AnalysisValidationError(f"activation_concentration.parquet {column} must be <= 1")


def deduplicate_scientific_rows(
    name: str, frame: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = list(DEDUPLICATION_KEYS[name])
    evidence = list(SCIENTIFIC_EVIDENCE_COLUMNS[name])
    duplicate_mask = frame.duplicated(keys, keep=False)
    duplicate_groups = 0
    for key, values in frame.loc[duplicate_mask].groupby(keys, dropna=False, sort=False):
        duplicate_groups += 1
        first = values.iloc[0]
        for column in evidence:
            candidate = values[column]
            if pd.api.types.is_numeric_dtype(candidate):
                agrees = np.allclose(
                    candidate.to_numpy(dtype=float), float(first[column]),
                    rtol=1e-6, atol=1e-8, equal_nan=True,
                )
            else:
                agrees = bool((candidate.astype(str) == str(first[column])).all())
            if not agrees:
                raise AnalysisValidationError(
                    f"{INPUT_FILES[name]} duplicate scientific key {key!r} disagrees in "
                    f"evidence column {column!r}"
                )
    mode_rank = frame["source_run_mode"].map({"full": 0, "pilot": 1})
    ordered = frame.assign(_mode_rank=mode_rank, _input_order=np.arange(len(frame))).sort_values(
        ["_mode_rank", "_input_order"], kind="mergesort"
    )
    deduplicated = ordered.drop_duplicates(keys, keep="first").sort_values("_input_order")
    kept_modes = deduplicated["source_run_mode"].value_counts().to_dict()
    result = deduplicated.drop(columns=["_mode_rank", "_input_order"]).reset_index(drop=True)
    stats = {
        "pre_rows": int(len(frame)),
        "post_rows": int(len(result)),
        "rows_removed": int(len(frame) - len(result)),
        "duplicate_scientific_key_groups": int(duplicate_groups),
        "pre_rows_by_run_mode": {
            str(key): int(value) for key, value in frame.source_run_mode.value_counts().items()
        },
        "kept_rows_by_run_mode": {str(key): int(value) for key, value in kept_modes.items()},
        "preference": "full_over_pilot",
        "numeric_evidence_tolerance": {"rtol": 1e-6, "atol": 1e-8},
    }
    return result, stats


def load_and_validate_inputs(
    input_dirs: Sequence[Path], config_path: Path
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    if not input_dirs:
        raise AnalysisValidationError("at least one --input directory is required")
    if not config_path.is_file():
        raise AnalysisValidationError(f"configuration does not exist: {config_path}")
    config = json.loads(config_path.read_text())
    _require_keys(
        config,
        (
            "run_id", "layers", "page_size_bytes", "reference", "locked_tree_sha256",
            "locked_capture_sha256", "promotion_policy",
            "primary_metric",
            "activation_shortlist_unit", "coordinate_score_aggregation",
            "shortlist_physical_layout", "tile_pilot_shape",
        ),
        "sparse-streaming config",
    )
    frozen_design = {
        "activation_shortlist_unit": "input_coordinates",
        "coordinate_score_aggregation": "sum_stage_energy",
        "shortlist_physical_layout": "canonical_paired_planes_four_coordinates_per_projection_page",
        "tile_pilot_shape": [32, 32],
    }
    wrong_design = {
        key: {"actual": config[key], "expected": expected}
        for key, expected in frozen_design.items() if config[key] != expected
    }
    if wrong_design:
        raise AnalysisValidationError(f"frozen sparse-streaming design changed: {wrong_design}")
    config_digest = sha256_path(config_path)
    facts_rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {str(config_path): config_digest}
    frames: dict[str, list[pd.DataFrame]] = {name: [] for name in INPUT_FILES}
    for directory in input_dirs:
        if not directory.is_dir():
            raise AnalysisValidationError(f"input directory does not exist: {directory}")
        facts_path = directory / RUN_FACTS
        if not facts_path.is_file():
            raise AnalysisValidationError(f"missing required input: {facts_path}")
        facts = json.loads(facts_path.read_text())
        validate_run_facts(facts, config, config_digest)
        facts_rows.append(facts)
        hashes[str(facts_path)] = sha256_path(facts_path)
        directory_tables: dict[str, pd.DataFrame] = {}
        for name, filename in INPUT_FILES.items():
            path = directory / filename
            if not path.is_file():
                raise AnalysisValidationError(f"completed input directory is missing required table: {path}")
            table = pd.read_parquet(path)
            table["source_run_mode"] = facts["run_mode"]
            directory_tables[name] = table
            frames[name].append(table)
            hashes[str(path)] = sha256_path(path)
        observed_from_union = len(
            pd.concat(
                [table[list(COMMON_ID)] for table in directory_tables.values()],
                ignore_index=True,
            ).drop_duplicates()
        )
        if observed_from_union != int(facts["observed_unique_invocations"]):
            raise AnalysisValidationError(
                f"{directory} scientific-table union has {observed_from_union} unique "
                f"invocations but run_facts reports {facts['observed_unique_invocations']}"
            )
    merged = {name: pd.concat(values, ignore_index=True) for name, values in frames.items()}
    deduplication: dict[str, Any] = {}
    for name, frame in merged.items():
        validate_table(name, frame, int(config["page_size_bytes"]))
        merged[name], deduplication[name] = deduplicate_scientific_rows(name, frame)
        frame = merged[name]
        if "selection_regime" in frame:
            frame["selection_category"] = frame["selection_regime"].map(selection_category)
    reference_bpw = float(config.get("reference_bpw", 4.25))
    expert_weights = 3 * 2048 * 512
    for name in ("neuron", "shortlist", "tile"):
        frame = merged[name]
        expected_metadata_bpw = frame["selector_metadata_bytes_per_expert"] * 8.0 / expert_weights
        if not np.allclose(
            frame["selector_metadata_bpw"], expected_metadata_bpw, rtol=0, atol=1e-12
        ):
            raise AnalysisValidationError(
                f"{INPUT_FILES[name]} selector_metadata_bpw does not match "
                "selector_metadata_bytes_per_expert * 8 / expert_weights"
            )
        expected_multiplier = (
            reference_bpw + frame["suffix_storage_bpw"] + frame["selector_metadata_bpw"]
        ) / reference_bpw
        if not np.allclose(frame["storage_multiplier"], expected_multiplier, rtol=0, atol=1e-12):
            raise AnalysisValidationError(
                f"{INPUT_FILES[name]} storage_multiplier must equal "
                "(reference_bpw + suffix_storage_bpw + selector_metadata_bpw) / reference_bpw; "
                "resident-parent bpw is not the denominator"
            )
    config["_analysis_deduplication"] = deduplication
    return merged, facts_rows, config, hashes


def selection_category(regime: str) -> str:
    value = str(regime)
    for prefix, category in REGIME_PREFIX_TO_CATEGORY.items():
        if value.startswith(prefix):
            return category
    raise AnalysisValidationError(f"unknown detailed selection regime: {value!r}")


def _q10(values: pd.Series) -> float:
    return float(values.quantile(0.1))


def _candidate_records(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    value_column: str,
) -> pd.DataFrame:
    return (
        frame.groupby(list(group_columns), dropna=False)
        .agg(
            n=(value_column, "size"),
            p10=(value_column, _q10),
            median=(value_column, "median"),
            p90=(value_column, lambda x: float(x.quantile(0.9))),
            median_physical_pages=("physical_pages", "median") if "physical_pages" in frame else ("candidate_pages", "median"),
            median_runtime_ms=("selector_runtime_ms", "median"),
            median_selector_bytes=("selector_bytes_read", "median"),
        )
        .reset_index()
    )


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    # Promotion gates are recomputed by the runner from these serialized
    # aggregates. Preserve enough precision that JSON rendering cannot move a
    # value across a frozen threshold after the analyzer has set its flag.
    return json.loads(frame.to_json(orient="records", double_precision=15))


def dataframe_sha256(frame: pd.DataFrame) -> str:
    columns = sorted(column for column in frame.columns if column != "source_run_mode")
    ordered = frame[columns].sort_values(columns, kind="mergesort").reset_index(drop=True)
    payload = ordered.to_csv(index=False, float_format="%.17g").encode()
    return hashlib.sha256(payload).hexdigest()


def _best_per_group(
    candidates: pd.DataFrame,
    partition: Sequence[str],
    sort_columns: Sequence[str],
    ascending: Sequence[bool],
) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    ordered = candidates.sort_values(list(sort_columns), ascending=list(ascending), kind="mergesort")
    return ordered.groupby(list(partition), dropna=False, sort=True).head(1).reset_index(drop=True)


def require_validation_coverage(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    expected_layers: Sequence[int],
    minimum_rows_per_layer: int,
    minimum_requests_per_layer: int,
    context: str,
) -> None:
    expected = set(map(int, expected_layers))
    if not expected:
        raise AnalysisValidationError("configuration must name at least one audited layer")
    if int(minimum_rows_per_layer) <= 0:
        raise AnalysisValidationError("promotion_min_validation_invocations_per_layer must be positive")
    if int(minimum_requests_per_layer) <= 0:
        raise AnalysisValidationError("promotion_min_validation_requests_per_layer must be positive")
    failures: list[str] = []
    for key, values in frame.groupby(list(group_columns), dropna=False, sort=True):
        counts = values.groupby("layer").size().to_dict()
        requests = values.groupby("layer").request_id.nunique().to_dict()
        missing = sorted(expected.difference(map(int, counts)))
        under = {
            int(layer): int(counts.get(layer, 0))
            for layer in sorted(expected)
            if int(counts.get(layer, 0)) < int(minimum_rows_per_layer)
        }
        under_requests = {
            int(layer): int(requests.get(layer, 0))
            for layer in sorted(expected)
            if int(requests.get(layer, 0)) < int(minimum_requests_per_layer)
        }
        if missing or under or under_requests:
            failures.append(
                f"candidate={key!r}, missing={missing}, rows_per_layer_below_min={under}, "
                f"requests_per_layer_below_min={under_requests}"
            )
    if failures:
        raise AnalysisValidationError(
            f"{context} lacks frozen validation coverage across all audited layers: "
            + "; ".join(failures[:8])
        )


def validation_coverage_records(
    frame: pd.DataFrame, group_columns: Sequence[str], family: str
) -> list[dict[str, Any]]:
    values = (
        frame.groupby(list(group_columns) + ["layer"], dropna=False)
        .agg(
            validation_invocations=("request_id", "size"),
            validation_requests=("request_id", "nunique"),
        )
        .reset_index()
    )
    values.insert(0, "frontier", family)
    return _records(values)


def choose_validation_promotions(
    tables: Mapping[str, pd.DataFrame], config: Mapping[str, Any], config_sha256: str, input_hashes: Mapping[str, str]
) -> dict[str, Any]:
    policy = config["promotion_policy"]
    if policy.get("selection_split") != "validation" or policy.get("no_test_tuning") is not True:
        raise AnalysisValidationError("promotion policy must specify validation selection and no_test_tuning=true")
    _require_keys(
        policy,
        (
            "promotion_min_validation_invocations_per_layer",
            "promotion_min_validation_requests_per_layer",
            "shortlist_overfetch_statistic",
            "shortlist_min_overfetch",
        ),
        "promotion_policy",
    )
    if policy["shortlist_overfetch_statistic"] != "p90":
        raise AnalysisValidationError(
            "promotion_policy.shortlist_overfetch_statistic must be 'p90' for this analyzer"
        )
    expected_layers = list(map(int, config["layers"]))
    minimum_rows = int(policy["promotion_min_validation_invocations_per_layer"])
    minimum_requests = int(policy["promotion_min_validation_requests_per_layer"])

    neuron = tables["neuron"]
    unit_validation = neuron[
        (neuron["capture_source"] == "exact_checkpoint")
        &
        (neuron["evaluation_split"] == "validation")
        & np.isclose(neuron["physical_budget_bpw"], 1.0)
        & (neuron["objective"] == "exact_sequential_complete_expert_qenergy")
    ]
    if unit_validation.empty:
        raise AnalysisValidationError("neuron frontier lacks validation complete-expert rows at 1.0 physical bpw")
    unit_groups = [
        "action_family", "representation", "selector", "selection_category",
        "selection_regime", "objective",
    ]
    require_validation_coverage(
        unit_validation, unit_groups, expected_layers, minimum_rows, minimum_requests,
        "neuron-major promotion",
    )
    unit_stats = _candidate_records(unit_validation, unit_groups, "recovery")
    unit_stats["passes_median"] = unit_stats["median"] >= float(policy["unit_go_at_1bpw_median"])
    unit_stats["passes_p10"] = unit_stats["p10"] >= float(policy["unit_go_at_1bpw_p10"])
    unit_eligible = unit_stats[unit_stats.passes_median & unit_stats.passes_p10]
    unit_selected = _best_per_group(
        unit_eligible,
        ["action_family", "selection_category"],
        ["action_family", "selection_category", "median", "p10", "median_runtime_ms", "selector", "representation"],
        [True, True, False, False, True, True, True],
    )

    shortlist = tables["shortlist"]
    shortlist_validation = shortlist[
        (shortlist["capture_source"] == "exact_checkpoint")
        & (shortlist["evaluation_split"] == "validation")
        & np.isclose(shortlist["physical_budget_bpw"], 1.0)
    ]
    if shortlist_validation.empty:
        raise AnalysisValidationError("activation shortlist frontier lacks validation rows")
    shortlist_groups = [
        "projection", "score_method", "shortlist_size", "refresh_block",
        "selection_category", "selection_regime",
    ]
    require_validation_coverage(
        shortlist_validation, shortlist_groups, expected_layers, minimum_rows, minimum_requests,
        "activation-shortlist promotion",
    )
    shortlist_primary = "fraction_exact_over_diagonal_gain_retained_matched_fetched_pages"
    short_stats = _candidate_records(shortlist_validation, shortlist_groups, shortlist_primary)
    overfetch_stats = (
        shortlist_validation.groupby(shortlist_groups, dropna=False)
        .candidate_overfetch_factor.agg(
            p10_candidate_overfetch=_q10,
            median_candidate_overfetch="median",
            p90_candidate_overfetch=lambda x: float(x.quantile(0.9)),
            max_candidate_overfetch="max",
        )
        .reset_index()
    )
    short_stats = short_stats.merge(
        overfetch_stats, on=shortlist_groups, how="left", validate="one_to_one"
    )
    short_stats["passes_median"] = short_stats["median"] >= float(policy["shortlist_min_exact_gain_retention_median"])
    short_stats["passes_p10"] = short_stats["p10"] >= float(policy["shortlist_min_exact_gain_retention_p10"])
    short_stats["passes_candidates"] = short_stats["shortlist_size"] <= int(policy["shortlist_max_candidates"])
    short_stats["passes_min_overfetch"] = (
        short_stats["p10_candidate_overfetch"] >= float(policy["shortlist_min_overfetch"])
    )
    short_stats["passes_max_overfetch"] = (
        short_stats["p90_candidate_overfetch"] <= float(policy["shortlist_max_overfetch"])
    )
    short_stats["passes_overfetch"] = (
        short_stats["passes_min_overfetch"] & short_stats["passes_max_overfetch"]
    )
    short_eligible = short_stats[
        short_stats.passes_median & short_stats.passes_p10
        & short_stats.passes_candidates & short_stats.passes_overfetch
    ]
    short_selected = _best_per_group(
        short_eligible,
        ["projection", "selection_category"],
        [
            "projection", "selection_category", "median", "p10",
            "p90_candidate_overfetch", "p10_candidate_overfetch",
            "max_candidate_overfetch", "shortlist_size",
            "refresh_block", "median_runtime_ms", "score_method",
        ],
        [True, True, False, False, True, False, True, True, True, True, True],
    )

    tile = tables["tile"]
    tile_validation = tile[
        (tile["capture_source"] == "exact_checkpoint")
        & (tile["evaluation_split"] == "validation")
        & (tile["objective"] == "exact_sequential_complete_expert_qenergy")
    ]
    baseline_name = str(policy.get("tile_baseline_action_family", "input_coordinate_baseline"))
    baseline = tile_validation[
        (tile_validation["action_family"] == baseline_name)
        & np.isclose(tile_validation["physical_budget_bpw"], 1.0)
    ]
    if baseline.empty:
        raise AnalysisValidationError(
            f"tile frontier lacks validation baseline action_family={baseline_name!r} at 1.0 physical bpw"
        )
    input_baseline_median = float(baseline["recovery"].median())
    baseline_pages = float(baseline["physical_pages"].median())
    locked_pr6_median = float(
        config["locked_pr6_baselines_at_1_physical_bpw"]["h0_hybrid_representation"]["median"]
    )
    candidate_rows = tile_validation[tile_validation["action_family"] != baseline_name]
    if candidate_rows.empty:
        raise AnalysisValidationError("tile frontier contains no non-baseline validation candidate rows")
    tile_groups = [
        "action_family", "tile_shape", "selector", "selection_category",
        "selection_regime", "objective",
    ]
    require_validation_coverage(
        tile_validation, tile_groups + ["physical_budget_bpw"], expected_layers,
        minimum_rows, minimum_requests, "tile promotion",
    )
    one_bpw = candidate_rows[np.isclose(candidate_rows["physical_budget_bpw"], 1.0)]
    if one_bpw.empty:
        raise AnalysisValidationError("tile frontier lacks candidate validation rows at 1.0 physical bpw")
    tile_stats = _candidate_records(one_bpw, tile_groups, "recovery")
    tile_stats["recovery_point_gain_vs_locked_pr6_h0_hybrid"] = (
        tile_stats["median"] - locked_pr6_median
    )
    curve = (
        candidate_rows.groupby(tile_groups + ["physical_budget_bpw"], dropna=False)
        .agg(median_recovery=("recovery", "median"), median_pages=("physical_pages", "median"))
        .reset_index()
    )
    page_reductions: list[dict[str, Any]] = []
    for key, values in curve.groupby(tile_groups, dropna=False):
        reaches = values[values["median_recovery"] >= input_baseline_median].sort_values("median_pages")
        pages = float(reaches.iloc[0]["median_pages"]) if len(reaches) else float("inf")
        record = dict(zip(tile_groups, key if isinstance(key, tuple) else (key,)))
        record["page_reduction_at_matched_recovery"] = (
            1.0 - pages / baseline_pages if np.isfinite(pages) and baseline_pages > 0 else float("-inf")
        )
        page_reductions.append(record)
    tile_stats = tile_stats.merge(pd.DataFrame(page_reductions), on=tile_groups, how="left", validate="one_to_one")
    tile_stats["passes_recovery_gain"] = (
        tile_stats["recovery_point_gain_vs_locked_pr6_h0_hybrid"]
        >= float(policy["tile_min_recovery_point_gain"])
    )
    tile_stats["passes_page_reduction"] = tile_stats["page_reduction_at_matched_recovery"] >= float(policy["tile_min_page_reduction_at_matched_recovery"])
    tile_eligible = tile_stats[tile_stats.passes_recovery_gain | tile_stats.passes_page_reduction]
    tile_selected = _best_per_group(
        tile_eligible,
        ["action_family", "selection_category"],
        [
            "action_family", "selection_category",
            "recovery_point_gain_vs_locked_pr6_h0_hybrid",
            "page_reduction_at_matched_recovery", "median", "median_runtime_ms", "tile_shape", "selector",
        ],
        [True, True, False, False, False, True, True, True],
    )

    return {
        "schema_version": 1,
        "generated_by": "oracle_study.sparse_streaming_analysis.choose_validation_promotions",
        "config_sha256": config_sha256,
        "provenance": {
            "config_sha256": config_sha256,
            "reference_revision": config["reference"]["revision"],
            "checkpoint_config_sha256": config["reference"]["config_sha256"],
            "checkpoint_index_sha256": config["reference"]["index_sha256"],
            "tree_sha256": config["locked_tree_sha256"],
            "capture_sha256": config["locked_capture_sha256"],
        },
        "validation_evidence_sha256": {
            "neuron_major": dataframe_sha256(unit_validation),
            "activation_shortlist": dataframe_sha256(shortlist_validation),
            "tile_streaming": dataframe_sha256(tile_validation),
        },
        "nonvalidation_input_hashes_excluded_from_frozen_decision": True,
        "selection_split": "validation",
        "selection_capture_source": "exact_checkpoint",
        "cross_reference_role": "secondary_sensitivity_only",
        "test_rows_consulted_for_selection": False,
        "policy": policy,
        "minimum_validation_coverage": {
            "layers": expected_layers,
            "invocations_per_layer": minimum_rows,
            "distinct_requests_per_layer": minimum_requests,
        },
        "validation_coverage": [
            *validation_coverage_records(unit_validation, unit_groups, "neuron_major"),
            *validation_coverage_records(shortlist_validation, shortlist_groups, "activation_shortlist"),
            *validation_coverage_records(
                tile_validation, tile_groups + ["physical_budget_bpw"], "tile_streaming"
            ),
        ],
        "neuron_major": {
            "status": "promote" if len(unit_selected) else "stop",
            "all_validation_candidates": _records(unit_stats),
            "promoted": _records(unit_selected),
        },
        "activation_shortlist": {
            "status": "promote" if len(short_selected) else "stop",
            "all_validation_candidates": _records(short_stats),
            "promoted": _records(short_selected),
        },
        "tile_streaming": {
            "status": "promote" if len(tile_selected) else "stop",
            "baseline_action_family": baseline_name,
            "input_coordinate_baseline_validation_median_at_1bpw": input_baseline_median,
            "baseline_validation_median_pages_at_1bpw": baseline_pages,
            "locked_pr6_h0_hybrid_median_at_1bpw": locked_pr6_median,
            "all_validation_candidates": _records(tile_stats),
            "promoted": _records(tile_selected),
        },
    }


def frozen_json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def frozen_json_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(frozen_json_text(payload).encode("utf-8")).hexdigest()


def validate_full_run_promotions(
    facts: Sequence[Mapping[str, Any]], promotions: Mapping[str, Any]
) -> str:
    """Bind every full cohort to the one validation-frozen decision payload.

    The payload is regenerated solely from the pilot's exact-checkpoint
    validation rows. Its canonical hash is therefore also the expected hash
    of the pilot-analysis ``sparse_streaming_promotions.json``. Pilot runner
    facts intentionally have no promotion hash; every full runner fact must
    carry this exact digest.
    """
    expected = frozen_json_sha256(promotions)
    full = [item for item in facts if item.get("run_mode") == "full"]
    if not full:
        return expected
    observed: list[str] = []
    for item in full:
        digest = item.get("validation_promotions_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise AnalysisValidationError(
                "full run_facts must record a canonical validation_promotions_sha256"
            )
        observed.append(digest)
    distinct = sorted(set(observed))
    if len(distinct) != 1:
        raise AnalysisValidationError(
            "all full run directories must use the same frozen validation promotion artifact"
        )
    if distinct[0] != expected:
        raise AnalysisValidationError(
            "full-run validation_promotions_sha256 does not match the promotion payload "
            "regenerated from exact-checkpoint validation evidence"
        )
    return expected


def write_frozen_json(path: Path, payload: Mapping[str, Any]) -> None:
    rendered = frozen_json_text(payload)
    if path.exists() and path.read_text() != rendered:
        raise AnalysisValidationError(
            f"refusing to overwrite frozen decision with different contents: {path}"
        )
    path.write_text(rendered)


def summarize_frontiers(tables: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    neuron_groups = [
        "source_run_mode", "capture_source", "evaluation_split", "objective", "selection_category", "selection_regime",
        "action_family", "representation", "selector", "physical_budget_bpw",
    ]
    neuron = (
        tables["neuron"].groupby(neuron_groups, dropna=False)
        .agg(
            n=("recovery", "size"), p10_recovery=("recovery", _q10),
            median_recovery=("recovery", "median"), p90_recovery=("recovery", lambda x: float(x.quantile(0.9))),
            median_physical_pages=("physical_pages", "median"),
            median_logical_actions=("logical_actions", "median"),
            median_page_amplification=("page_amplification", "median"),
            median_runtime_ms=("selector_runtime_ms", "median"),
            median_selector_bytes=("selector_bytes_read", "median"),
        ).reset_index()
    )
    neuron_by_layer = (
        tables["neuron"].groupby(
            [*neuron_groups[:3], "layer", *neuron_groups[3:]], dropna=False
        )
        .agg(
            n=("recovery", "size"), p10_recovery=("recovery", _q10),
            median_recovery=("recovery", "median"),
            p90_recovery=("recovery", lambda x: float(x.quantile(0.9))),
            median_physical_pages=("physical_pages", "median"),
            median_logical_actions=("logical_actions", "median"),
            median_logical_bytes=("logical_bytes", "median"),
            median_page_amplification=("page_amplification", "median"),
        ).reset_index()
    )
    short_groups = [
        "source_run_mode", "capture_source", "evaluation_split", "projection", "selection_category", "selection_regime", "score_method",
        "shortlist_size", "refresh_block", "physical_budget_bpw",
    ]
    shortlist = (
        tables["shortlist"].groupby(short_groups, dropna=False)
        .agg(
            n=("fraction_exact_over_diagonal_gain_retained_matched_fetched_pages", "size"),
            p10_exact_gain_retained_matched_fetched_pages=(
                "fraction_exact_over_diagonal_gain_retained_matched_fetched_pages", _q10
            ),
            median_exact_gain_retained_matched_fetched_pages=(
                "fraction_exact_over_diagonal_gain_retained_matched_fetched_pages", "median"
            ),
            p90_exact_gain_retained_matched_fetched_pages=(
                "fraction_exact_over_diagonal_gain_retained_matched_fetched_pages",
                lambda x: float(x.quantile(0.9)),
            ),
            median_planned_prefix_exact_gain_retained=("exact_gain_retained", "median"),
            median_action_recall=("exact_action_recall", "median"),
            median_importance_weighted_recall=("importance_weighted_recall", "median"),
            median_candidate_pages=("candidate_pages", "median"),
            median_fetched_pages=("fetched_pages", "median"),
            p10_candidate_overfetch=("candidate_overfetch_factor", _q10),
            median_candidate_overfetch=("candidate_overfetch_factor", "median"),
            p90_candidate_overfetch=("candidate_overfetch_factor", lambda x: float(x.quantile(0.9))),
            max_candidate_overfetch=("candidate_overfetch_factor", "max"),
            median_runtime_ms=("selector_runtime_ms", "median"),
            median_selector_bytes=("selector_bytes_read", "median"),
        ).reset_index()
    )
    shortlist_by_layer = (
        tables["shortlist"].groupby(
            [*short_groups[:3], "layer", *short_groups[3:]], dropna=False
        )
        .agg(
            n=("fraction_exact_over_diagonal_gain_retained_matched_fetched_pages", "size"),
            p10_exact_gain_retained_matched_fetched_pages=(
                "fraction_exact_over_diagonal_gain_retained_matched_fetched_pages", _q10
            ),
            median_exact_gain_retained_matched_fetched_pages=(
                "fraction_exact_over_diagonal_gain_retained_matched_fetched_pages", "median"
            ),
            p90_exact_gain_retained_matched_fetched_pages=(
                "fraction_exact_over_diagonal_gain_retained_matched_fetched_pages",
                lambda x: float(x.quantile(0.9)),
            ),
            median_fetched_pages=("fetched_pages", "median"),
            median_logical_actions=("logical_actions", "median"),
            median_logical_bytes=("logical_bytes", "median"),
            p90_candidate_overfetch=(
                "candidate_overfetch_factor", lambda x: float(x.quantile(0.9))
            ),
        ).reset_index()
    )
    tile_groups = [
        "source_run_mode", "capture_source", "evaluation_split", "objective", "selection_category", "selection_regime", "action_family",
        "tile_shape", "selector", "physical_budget_bpw",
    ]
    tile = (
        tables["tile"].groupby(tile_groups, dropna=False)
        .agg(
            n=("recovery", "size"), p10_recovery=("recovery", _q10),
            median_recovery=("recovery", "median"), p90_recovery=("recovery", lambda x: float(x.quantile(0.9))),
            median_physical_pages=("physical_pages", "median"),
            median_logical_actions=("logical_actions", "median"),
            median_page_amplification=("page_amplification", "median"),
            median_runtime_ms=("selector_runtime_ms", "median"),
            median_selector_bytes=("selector_bytes_read", "median"),
        ).reset_index()
    )
    tile_by_layer = (
        tables["tile"].groupby(
            [*tile_groups[:3], "layer", *tile_groups[3:]], dropna=False
        )
        .agg(
            n=("recovery", "size"), p10_recovery=("recovery", _q10),
            median_recovery=("recovery", "median"),
            p90_recovery=("recovery", lambda x: float(x.quantile(0.9))),
            median_physical_pages=("physical_pages", "median"),
            median_logical_actions=("logical_actions", "median"),
            median_logical_bytes=("logical_bytes", "median"),
            median_page_amplification=("page_amplification", "median"),
        ).reset_index()
    )
    stability_groups = [
        "source_run_mode", "capture_source", "evaluation_split", "layer", "projection", "score_method",
        "shortlist_size", "relationship", "token_gap",
    ]
    stability = (
        tables["stability"].groupby(stability_groups, dropna=False)
        .agg(
            n=("support_jaccard", "size"), p10_support_jaccard=("support_jaccard", _q10),
            median_support_jaccard=("support_jaccard", "median"),
            p10_importance_weighted_overlap=("importance_weighted_overlap", _q10),
            median_importance_weighted_overlap=("importance_weighted_overlap", "median"),
        ).reset_index()
    )
    label_groups = [
        "source_run_mode", "capture_source", "evaluation_split", "layer", "projection",
    ]
    rank_columns = sorted(
        column for column in tables["labels"].columns if column.startswith("coordinate_rank_")
    )
    label_records: list[dict[str, Any]] = []
    for key, values in tables["labels"].groupby(label_groups, dropna=False):
        occurrences = values[["request_id", "position", "expert_id"]].drop_duplicates()
        stage_one = int((values.refinement_stage == 1).sum())
        stage_two = int((values.refinement_stage == 2).sum())
        total = int(len(values))
        record = dict(zip(label_groups, key if isinstance(key, tuple) else (key,)))
        record.update({
            "selected_actions": total,
            "distinct_requests": int(values.request_id.nunique()),
            "unique_occurrences": int(len(occurrences)),
            "experts": int(values.expert_id.nunique()),
            "actions_per_occurrence": total / max(len(occurrences), 1),
            "q2_q3_actions": stage_one,
            "q3_q4_actions": stage_two,
            "q2_q3_fraction": stage_one / max(total, 1),
            "q3_q4_fraction": stage_two / max(total, 1),
            "median_marginal_gain": float(values.marginal_gain.median()),
        })
        for column in rank_columns:
            record[f"{column}_p10"] = float(values[column].quantile(0.1))
            record[f"{column}_median"] = float(values[column].median())
            record[f"{column}_p90"] = float(values[column].quantile(0.9))
        label_records.append(record)
    labels = pd.DataFrame(label_records)
    concentration = (
        tables["concentration"].groupby(
            [
                "source_run_mode", "capture_source", "evaluation_split", "layer", "top_k",
            ],
            dropna=False,
        ).agg(
            n=("top_k_energy_fraction", "size"),
            p10_top_k_energy_fraction=("top_k_energy_fraction", _q10),
            median_top_k_energy_fraction=("top_k_energy_fraction", "median"),
            p90_top_k_energy_fraction=("top_k_energy_fraction", lambda x: float(x.quantile(0.9))),
            median_exact_zero_fraction=("exact_zero_fraction", "median"),
            p10_coordinates_for_90pct_energy=("coordinates_for_90pct_energy", _q10),
            median_coordinates_for_90pct_energy=("coordinates_for_90pct_energy", "median"),
            p90_coordinates_for_90pct_energy=(
                "coordinates_for_90pct_energy", lambda x: float(x.quantile(0.9))
            ),
        ).reset_index()
    )
    return {
        "neuron_major_summary.csv": neuron,
        "neuron_major_by_layer_summary.csv": neuron_by_layer,
        "activation_shortlist_summary.csv": shortlist,
        "activation_shortlist_by_layer_summary.csv": shortlist_by_layer,
        "tile_streaming_summary.csv": tile,
        "tile_streaming_by_layer_summary.csv": tile_by_layer,
        "support_stability_summary.csv": stability,
        "exact_action_label_summary.csv": labels,
        "activation_concentration_summary.csv": concentration,
    }


def _savefig(fig: plt.Figure, directory: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(directory / f"{name}.png", dpi=180, metadata={"Software": "oracle_study"})
    svg = directory / f"{name}.svg"
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def preferred_test_rows(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame[
        (frame.capture_source == "exact_checkpoint")
        & (frame.evaluation_split == "test")
    ]
    if (values.source_run_mode == "full").any():
        return values[values.source_run_mode == "full"]
    return values[values.source_run_mode == "pilot"]


def make_plots(tables: Mapping[str, pd.DataFrame], config: Mapping[str, Any], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    neuron = tables["neuron"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    isolated = preferred_test_rows(tables["shortlist"]).copy()
    isolated["fetched_projection_bpw"] = 8.0 * isolated.physical_bytes / (2048 * 512)
    for label, column in (
        ("diagonal", "recovery_diagonal_matched_fetched_pages"),
        ("exact", "recovery_exact_matched_fetched_pages"),
    ):
        curve = isolated.groupby("fetched_projection_bpw")[column].median()
        axes[0].plot(curve.index, curve.values, marker="o", label=label)
    for method, rows in isolated.groupby("score_method"):
        curve = rows.groupby("fetched_projection_bpw").recovery.median()
        axes[0].plot(curve.index, curve.values, marker="o", label=f"constrained/{method}")
    axes[0].set_title("isolated projection qenergy recovery")
    complete = preferred_test_rows(neuron)
    complete = complete[complete.objective == "exact_sequential_complete_expert_qenergy"]
    for (family, category, selector, regime), rows in complete.groupby(
        ["action_family", "selection_category", "selector", "selection_regime"]
    ):
        curve = rows.groupby("physical_budget_bpw").recovery.median()
        axes[1].plot(curve.index, curve.values, marker="o", label=f"{family}/{category}/{selector}/{regime}")
    axes[1].set_title("exact sequential complete-expert qenergy recovery")
    axes[0].set_xlabel("Fetched physical bpw (one-projection denominator)")
    axes[1].set_xlabel("Fetched physical bpw (complete-expert denominator)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=6)
    axes[0].set_ylabel("Median expert-output qenergy recovery")
    _savefig(fig, directory, "01_isolated_vs_complete_qenergy")

    shortlist = tables["shortlist"]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    values = preferred_test_rows(shortlist)
    for (projection, category, method, regime), rows in values.groupby(
        ["projection", "selection_category", "score_method", "selection_regime"]
    ):
        curve = rows.groupby("shortlist_size").fraction_exact_over_diagonal_gain_retained_matched_fetched_pages.median()
        ax.plot(curve.index, curve.values, marker="o", label=f"{projection}/{category}/{method}/{regime}")
    ax.axhline(float(config["promotion_policy"]["shortlist_min_exact_gain_retention_median"]), color="black", ls="--", lw=1)
    ax.axvline(int(config["promotion_policy"]["shortlist_max_candidates"]), color="black", ls=":", lw=1)
    ax.set(xlabel="Prefetched input coordinates (K)", ylabel="Median exact-over-diagonal gain retained")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6)
    _savefig(fig, directory, "02_activation_shortlist_containment")

    tile = tables["tile"]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    values = preferred_test_rows(tile)
    values = values[values.objective == "exact_sequential_complete_expert_qenergy"]
    for (family, shape, category, selector, regime), rows in values.groupby(
        ["action_family", "tile_shape", "selection_category", "selector", "selection_regime"]
    ):
        curve = rows.groupby("physical_budget_bpw").recovery.median()
        ax.plot(curve.index, curve.values, marker="o", label=f"{family}/{shape}/{category}/{selector}/{regime}")
    ax.set(xlabel="Fetched physical correction bpw", ylabel="Median complete-expert qenergy recovery")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6)
    _savefig(fig, directory, "03_tile_complete_expert_frontier")

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for label, frame in (("neuron", neuron), ("tile", tile)):
        values = preferred_test_rows(frame)
        ax.scatter(values.logical_actions, values.physical_pages, s=9, alpha=0.35, label=label)
    ax.set(xlabel="Logical actions applied", ylabel="Fetched 512-byte physical pages")
    ax.grid(alpha=0.25)
    ax.legend()
    _savefig(fig, directory, "04_logical_actions_vs_physical_pages")

    stability = tables["stability"]
    stability = preferred_test_rows(stability)
    stability = stability[stability.relationship == "adjacent_token_same_sequence"]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for (projection, method), rows in stability.groupby(["projection", "score_method"]):
        curve = rows.groupby("shortlist_size").support_jaccard.median()
        ax.plot(curve.index, curve.values, marker="o", label=f"{projection}/{method}")
    ax.set(
        xlabel="Configured support size (proxy K coordinates; exact page budget)",
        ylabel="Median support Jaccard at true token gap 1",
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    _savefig(fig, directory, "05_true_adjacent_support_stability")

    concentration = preferred_test_rows(tables["concentration"])
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    for layer, rows in concentration.groupby("layer"):
        curve = rows.groupby("top_k").top_k_energy_fraction.median()
        axes[0].plot(curve.index, curve.values, marker="o", label=f"layer {layer}")
    axes[0].set(
        xlabel="Largest-magnitude activation coordinates K",
        ylabel="Median activation energy fraction",
    )
    by_layer = concentration.groupby("layer").coordinates_for_90pct_energy.median()
    axes[1].bar([str(int(layer)) for layer in by_layer.index], by_layer.values)
    axes[1].set(
        xlabel="Layer", ylabel="Median coordinates required for 90% activation energy",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    _savefig(fig, directory, "06_activation_concentration")


def compute_accounting(
    tables: Mapping[str, pd.DataFrame], config: Mapping[str, Any], facts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    page_size = int(config["page_size_bytes"])
    expert_weights = 3 * 2048 * 512
    pages_per_bpw = expert_weights // 8 // page_size
    observed: dict[str, Any] = {}
    for name in ("neuron", "shortlist", "tile"):
        values = tables[name]
        observed[name] = {
            "rows": int(len(values)),
            "physical_pages_min_max": [int(values.physical_pages.min()), int(values.physical_pages.max())],
            "physical_bytes_min_max": [int(values.physical_bytes.min()), int(values.physical_bytes.max())],
            "logical_actions_min_max": [int(values.logical_actions.min()), int(values.logical_actions.max())],
            "logical_bytes_min_max": [int(values.logical_bytes.min()), int(values.logical_bytes.max())],
            "logical_bpw_min_max": [float(values.logical_bpw.min()), float(values.logical_bpw.max())],
            "selector_runtime_ms_min_max": [float(values.selector_runtime_ms.min()), float(values.selector_runtime_ms.max())],
            "selector_bytes_read_min_max": [int(values.selector_bytes_read.min()), int(values.selector_bytes_read.max())],
            "storage_multiplier_min_max": [float(values.storage_multiplier.min()), float(values.storage_multiplier.max())],
            "suffix_storage_bpw_min_max": [float(values.suffix_storage_bpw.min()), float(values.suffix_storage_bpw.max())],
            "selector_metadata_bpw_min_max": [float(values.selector_metadata_bpw.min()), float(values.selector_metadata_bpw.max())],
            "selector_metadata_bytes_per_expert_min_max": [
                int(values.selector_metadata_bytes_per_expert.min()),
                int(values.selector_metadata_bytes_per_expert.max()),
            ],
            "selector_metadata_bytes_all_40x256_experts_max": int(
                values.selector_metadata_bytes_per_expert.max() * 40 * 256
            ),
            "page_amplification_median": float(values.page_amplification.median()),
        }
    values = tables["shortlist"]
    observed["activation_shortlist"] = {
        "rows": int(len(values)),
        "selector_runtime_ms_min_max": [float(values.selector_runtime_ms.min()), float(values.selector_runtime_ms.max())],
        "selector_bytes_read_min_max": [int(values.selector_bytes_read.min()), int(values.selector_bytes_read.max())],
        "candidate_pages_min_max": [int(values.candidate_pages.min()), int(values.candidate_pages.max())],
        "fetched_pages_min_max": [int(values.fetched_pages.min()), int(values.fetched_pages.max())],
    }
    concentration = tables["concentration"]
    observed["activation_concentration"] = {
        "rows": int(len(concentration)),
        "top_k_values": sorted(map(int, concentration.top_k.unique())),
        "exact_zero_fraction_min_max": [
            float(concentration.exact_zero_fraction.min()),
            float(concentration.exact_zero_fraction.max()),
        ],
        "coordinates_for_90pct_energy_min_max": [
            int(concentration.coordinates_for_90pct_energy.min()),
            int(concentration.coordinates_for_90pct_energy.max()),
        ],
    }
    max_storage = max(
        float(tables["neuron"].storage_multiplier.max()),
        float(tables["shortlist"].storage_multiplier.max()),
        float(tables["tile"].storage_multiplier.max()),
    )
    cap = float(config.get("external_storage_cap", 5.0))
    reference_bpw = float(config.get("reference_bpw", 4.25))
    representation_suffix_bpw = float(config.get("suffix_bpw_per_complete_representation", 2.0))
    representation_multiplier = (reference_bpw + representation_suffix_bpw) / reference_bpw
    return {
        "schema_version": 1,
        "units": {"bytes": "bytes", "selector_runtime": "milliseconds", "budget": "physical correction bpw"},
        "selector_measurement_scope": {
            "selector_runtime_ms": "one complete selector path/frontier construction for the invocation",
            "selector_bytes_read": "unique selector tensor footprint lower bound; not a hardware traffic counter and excludes repeated scans/Gram-column reads",
            "budget_rows": "repeat the same full-path/full-frontier cost; do not sum rows or interpret as incremental per-budget latency",
        },
        "analytical_geometry": {
            "expert_weights": expert_weights,
            "page_size_bytes": page_size,
            "pages_per_one_physical_bpw": pages_per_bpw,
            "full_q4_neuron_packet_bytes": 1536,
            "full_q4_neuron_packet_pages": 3,
            "progressive_neuron_q3_pages": 2,
            "progressive_neuron_q4_increment_pages": 1,
            "hidden_units_refinable_at_one_bpw_full_packets": 256,
            "gate_up_32x32_q2_to_q4_tile_bytes": 512,
            "gate_up_32x32_q2_to_q4_tile_pages": 1,
            "complete_tile_plus_down_endpoint_pages": 1536,
            "complete_suffix_endpoint_bpw": 2.0,
            "storage_multiplier_convention": "(reference_bpw + suffix_bpw + selector_metadata_bpw) / reference_bpw",
            "reference_bpw": reference_bpw,
            "one_representation_suffix_bpw": representation_suffix_bpw,
            "one_representation_multiplier_before_selector_metadata": representation_multiplier,
            "resident_parent_bpw_not_used_as_multiplier_denominator": True,
        },
        "frozen_protocol": {
            "primary_metric": config["primary_metric"],
            "activation_shortlist_unit": config["activation_shortlist_unit"],
            "coordinate_score_aggregation": config["coordinate_score_aggregation"],
            "shortlist_physical_layout": config["shortlist_physical_layout"],
            "tile_pilot_shape": config["tile_pilot_shape"],
            "promotion_min_validation_invocations_per_layer": config[
                "promotion_policy"
            ][
                "promotion_min_validation_invocations_per_layer"
            ],
            "promotion_min_validation_requests_per_layer": config[
                "promotion_policy"
            ][
                "promotion_min_validation_requests_per_layer"
            ],
        },
        "pilot_full_deduplication": config["_analysis_deduplication"],
        "observed_not_inferred": observed,
        "maximum_observed_storage_multiplier": max_storage,
        "external_storage_cap": cap,
        "passes_external_storage_cap": bool(max_storage < cap),
        "run_ids": sorted({str(item["run_id"]) for item in facts}),
        "test_masks_used_for_promotion_or_layout_training": False,
        "task_accuracy_measured": False,
    }


def _promotion_test_rows(frame: pd.DataFrame, promoted: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> pd.DataFrame:
    values = frame[frame.evaluation_split == "test"]
    selected: list[pd.DataFrame] = []
    for record in promoted:
        mask = np.ones(len(values), dtype=bool)
        for key in keys:
            mask &= values[key].astype(str).to_numpy() == str(record[key])
        selected.append(values[mask])
    return pd.concat(selected, ignore_index=True) if selected else values.iloc[0:0]


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "No configuration passed the predeclared validation gate."
    shown = frame[list(columns)].copy()
    for column in shown.select_dtypes(include=[np.number]).columns:
        shown[column] = shown[column].map(lambda value: f"{float(value):.4f}")

    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        text = str(value).replace("\r\n", "\n").replace("\r", "\n")
        return text.replace("|", r"\|").replace("\n", "<br>")

    header = "| " + " | ".join(cell(column) for column in shown.columns) + " |"
    separator = "| " + " | ".join("---" for _ in shown.columns) + " |"
    body = [
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in shown.itertuples(index=False, name=None)
    ]
    return "\n".join((header, separator, *body))


def build_report(
    tables: Mapping[str, pd.DataFrame], promotions: Mapping[str, Any], config: Mapping[str, Any]
) -> str:
    unit_keys = [
        "action_family", "representation", "selector", "selection_category",
        "selection_regime", "objective",
    ]
    unit_test = _promotion_test_rows(tables["neuron"], promotions["neuron_major"]["promoted"], unit_keys)
    unit_test_summary = (
        unit_test.groupby(["source_run_mode", "capture_source"] + unit_keys, dropna=False).agg(
            n=("recovery", "size"), p10=("recovery", _q10), median=("recovery", "median"),
            p90=("recovery", lambda x: float(x.quantile(0.9))),
            pages=("physical_pages", "median"), actions=("logical_actions", "median"),
        ).reset_index()
    ) if len(unit_test) else pd.DataFrame()
    short_keys = [
        "projection", "score_method", "shortlist_size", "refresh_block",
        "selection_category", "selection_regime",
    ]
    short_test = _promotion_test_rows(tables["shortlist"], promotions["activation_shortlist"]["promoted"], short_keys)
    short_summary = (
        short_test.groupby(
            ["source_run_mode", "capture_source"] + short_keys + ["physical_budget_bpw"], dropna=False
        ).agg(
            n=("fraction_exact_over_diagonal_gain_retained_matched_fetched_pages", "size"),
            p10=("fraction_exact_over_diagonal_gain_retained_matched_fetched_pages", _q10),
            median=("fraction_exact_over_diagonal_gain_retained_matched_fetched_pages", "median"),
            fetched_pages=("fetched_pages", "median"),
            p10_overfetch=("candidate_overfetch_factor", _q10),
            median_overfetch=("candidate_overfetch_factor", "median"),
            p90_overfetch=("candidate_overfetch_factor", lambda x: float(x.quantile(0.9))),
            max_overfetch=("candidate_overfetch_factor", "max"),
        ).reset_index()
    ) if len(short_test) else pd.DataFrame()
    tile_keys = [
        "action_family", "tile_shape", "selector", "selection_category",
        "selection_regime", "objective",
    ]
    tile_test = _promotion_test_rows(tables["tile"], promotions["tile_streaming"]["promoted"], tile_keys)
    tile_summary = (
        tile_test[np.isclose(tile_test.physical_budget_bpw, 1.0)].groupby(["source_run_mode", "capture_source"] + tile_keys, dropna=False).agg(
            n=("recovery", "size"), p10=("recovery", _q10), median=("recovery", "median"),
            p90=("recovery", lambda x: float(x.quantile(0.9))), pages=("physical_pages", "median"),
        ).reset_index()
    ) if len(tile_test) else pd.DataFrame()
    baseline = config.get("locked_pr6_baselines_at_1_physical_bpw", {})
    return f"""# Activation-Dependent Sparse Weight Streaming Study

Run: `{config['run_id']}`

## Scope and claim boundary

This study measures **expert-output qenergy recovery**, not task accuracy. It does not measure logits, routing quality, token quality, perplexity, or downstream benchmarks. The exact checkpoint, embedded Q2→Q3→Q4 codec, selected trees, and train/validation/test request split remain locked.

Every selector retains a precise `selection_regime`; a derived category groups it as `h0_oracle`, `h0_proxy`, or `deployable_h0_proxy` without erasing whether it reads the full suffix, runs late after resident Q2, or needs deployable metadata. H0 choices may use the current activation or exact suffix corrections and are not H4 prefetch claims. No H4 predictor is trained here. The broader cross-reference capture is a secondary sensitivity check; primary claims use the fresh exact-checkpoint capture.

## Promotion protocol

`sparse_streaming_promotions.json` is frozen from validation rows only. Test rows were joined only after configuration IDs had been selected. A changed decision cannot overwrite the frozen JSON. The exact action labels remain evidence, not training data for any held-out decision in this analysis.

Every candidate must cover layers `{config['layers']}` with at least **{config['promotion_policy']['promotion_min_validation_invocations_per_layer']} invocations and {config['promotion_policy']['promotion_min_validation_requests_per_layer']} distinct requests per layer**. `promotion_validation_coverage.csv` records the actual count for every candidate/layer, and each promoted validation record reports its total `n`.

Promotion gates use the frozen pooled p10/median rules, but a broad-effect claim requires the separate layer audit. `neuron_major_by_layer_summary.csv`, `activation_shortlist_by_layer_summary.csv`, and `tile_streaming_by_layer_summary.csv` report n/p10/median/p90 and physical/logical accounting for every layer, split, capture source, and selector identity. Coverage alone is never presented as evidence of consistent effect size.

## Completed inputs and pilot/full deduplication

Only run directories with `completed=true` and matching expected/observed unique-invocation counts are accepted. Pilot and full rows are keyed by invocation plus scientific configuration. Duplicate recovery, support, and action evidence must agree within the frozen tolerance; the full row is then retained and the pilot copy dropped. Pre/post counts are: `{json.dumps(config['_analysis_deduplication'], sort_keys=True)}`. Duplicate invocations are never silently reweighted.

## Method context, not evidence for this result

- [Prox](https://arxiv.org/abs/2607.27591) motivates using the SwiGLU intermediate state as a channel-salience signal. Prox studies channel omission in a different serving design; it does not establish the packet-recovery result measured here.
- [WINA](https://arxiv.org/abs/2505.19427) and its [reference implementation](https://github.com/microsoft/wina) motivate activation-times-weight proxy scores. WINA's exact orthogonality argument does not hold for these unrotated correction columns, so the score is treated only as a heuristic.
- [R-Sparse](https://arxiv.org/abs/2504.19449) motivates combining concentrated input support with another structured representation. It is not evidence that the correction-action Gram is sparse.
- [TEAL](https://arxiv.org/abs/2408.14690) and its [reference implementation](https://github.com/FasterDecoding/TEAL) motivate a separately labeled resident-Q2 sparsification control. That intervention changes resident computation and is not conflated with lossless selective suffix streaming.

## Measured activation concentration

`activation_concentration.parquet` and `activation_concentration_summary.csv` regenerate the heavy-tail diagnostics from the locked captures: exact-zero fraction, top-K energy fraction, and coordinates required for 90% energy. No previously quoted 0.069%/81%/762 figure is treated as evidence unless it is reproduced in these artifacts. Plot 06 shows the measured held-out curves and layer medians.

## Neuron-major progressive refinement

Neuron packets make 512 SwiGLU hidden units the decision axis. Full Q2→Q4 refinement costs three 512-byte pages per unit; the progressive form costs two pages for Q3 and one additional page for Q4. At one correction bpw, 256 complete unit packets fit.

Held-out promoted configurations:

{_markdown_table(unit_test_summary, ['source_run_mode', 'capture_source', 'action_family', 'representation', 'selector', 'selection_category', 'selection_regime', 'n', 'p10', 'median', 'p90', 'pages', 'actions']) if len(unit_test_summary) else 'No neuron-major configuration passed both validation median and p10 gates.'}

The table reports exact sequential complete-expert recovery separately from isolated projection rows in the CSV and plots. Logical actions are not treated as physical traffic.

## Activation shortlist containment and exact H0 rescoring

Candidate prefetch pages are charged even when a candidate is not applied. The primary gate uses `fraction_exact_over_diagonal_gain_retained_matched_fetched_pages`: exact, diagonal, and constrained recovery are all repriced to the same fetched-page count. Planned-prefix containment remains secondary. Exact rescoring within a fetched shortlist remains an H0 oracle unless an independently deployable mechanism supplies both the candidate set and a representation of the omitted target residual.

The frozen shortlist unit is `{config['activation_shortlist_unit']}`. Both refinement planes are aggregated by `{config['coordinate_score_aggregation']}` and mapped through `{config['shortlist_physical_layout']}`. Thus a configured K means K input coordinates, not K plane-actions, and candidate traffic is repriced to the canonical physical page union.

Held-out promoted configurations:

{_markdown_table(short_summary, ['source_run_mode', 'capture_source', 'projection', 'score_method', 'shortlist_size', 'refresh_block', 'selection_category', 'selection_regime', 'physical_budget_bpw', 'n', 'p10', 'median', 'fetched_pages', 'p10_overfetch', 'median_overfetch', 'p90_overfetch', 'max_overfetch']) if len(short_summary) else 'No shortlist configuration passed the predeclared utility, candidate-count, p10-minimum-overfetch, and p90-maximum-overfetch gates.'}

Raw support recall is secondary to retained exact-over-diagonal utility at matched fetched-page budgets. To prevent an underfilled candidate set from passing this matched-page metric trivially, promotion additionally requires p10 candidate overfetch to be at least **{config['promotion_policy']['shortlist_min_overfetch']:.2f}**; the bandwidth ceiling requires p90 to be at most **{config['promotion_policy']['shortlist_max_overfetch']:.2f}**. P10, median, p90, and maximum overfetch remain visible in the frozen decision and summaries.

## Page-aligned tiles and hybrids

The frozen pilot shape is `{config['tile_pilot_shape'][0]}×{config['tile_pilot_shape'][1]}`. One 32×32 paired gate/up Q2→Q4 tile is exactly one 512-byte page. Tile selection is nonlinear and is evaluated through actual SwiGLU reconstruction. The ≥3-point validation gate is measured against the locked PR #6 all-layer one-bpw H0-hybrid median **{config['locked_pr6_baselines_at_1_physical_bpw']['h0_hybrid_representation']['median']:.4f}**. The separately recomputed `input_coordinate_baseline` curve is used only for the alternative 25% matched-recovery page-reduction gate. No test row chooses a shape.

Held-out one-bpw promoted configurations:

{_markdown_table(tile_summary, ['source_run_mode', 'capture_source', 'action_family', 'tile_shape', 'selector', 'selection_category', 'selection_regime', 'n', 'p10', 'median', 'p90', 'pages']) if len(tile_summary) else 'No unrestricted gate/up-tile plus down-column path passed the validation recovery-gain or matched-recovery page-reduction gate.'}

## Physical accounting

`sparse_streaming_compute_storage_accounting.json` distinguishes logical actions, applied pages, fetched candidate pages, selector bytes, and external representation storage. Complete endpoint storage is accounted from the locked suffix representation; action counts never stand in for physical performance. The configured PR #6 one-bpw controls are preserved in the configuration: `{json.dumps(baseline, sort_keys=True)}`.

`selector_runtime_ms` is measured for one complete selector path/frontier construction per invocation. `selector_bytes_read` is a reproducible **unique tensor-footprint lower bound**, not a hardware traffic counter: it excludes cache-dependent repeated marginal scans and Gram-column reads. The same values are repeated on each budget snapshot row; rows must not be summed or read as incremental per-budget costs. Regimes `h0_oracle_full_suffix`, `h0_oracle_full_target_residual`, and `h0_oracle_suffix_metadata_heavy` inspect the full decoded suffix or target residual. Exact/dynamic and static tile selectors in those regimes are H0 oracles, not deployable selectors. Late resident-Q2 proxies are separately labeled and may require a second pass before suffix selection.

## Support stability

`support_stability_summary.csv` separates same-invocation gate/up overlap (`token_gap=0`) from true adjacent-token pairs (`token_gap=1`). Adjacent claims use only the fresh exact-checkpoint capture; the stride-16 broader capture is deliberately excluded from adjacency claims.

## Interpretation rule

Promotion failure is a negative result, not permission to tune on test. A positive H0 oracle shows headroom only. A deployable-H0 claim additionally requires a `deployable_h0_proxy` row to pass the same validation gate and retain its result on the frozen held-out evaluation while respecting fetched pages, selector compute, and storage. H4 prefetch feasibility remains a separate future question.
"""


def analyze(
    input_dirs: Sequence[Path], config_path: Path, output_dir: Path, *, validate_only: bool = False
) -> dict[str, Any]:
    tables, facts, config, input_hashes = load_and_validate_inputs(input_dirs, config_path)
    promotions = choose_validation_promotions(tables, config, sha256_path(config_path), input_hashes)
    promotion_sha256 = validate_full_run_promotions(facts, promotions)
    if validate_only:
        return {
            "validated": True,
            "run_ids": sorted({item["run_id"] for item in facts}),
            "promotions": promotions,
            "validation_promotions_sha256": promotion_sha256,
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    promotions_path = output_dir / "sparse_streaming_promotions.json"
    write_frozen_json(promotions_path, promotions)
    summaries = summarize_frontiers(tables)
    summaries["promotion_validation_coverage.csv"] = pd.DataFrame(
        promotions["validation_coverage"]
    )
    for filename, frame in summaries.items():
        frame.to_csv(output_dir / filename, index=False)
    plots = output_dir / "plots"
    make_plots(tables, config, plots)
    accounting = compute_accounting(tables, config, facts)
    (output_dir / "sparse_streaming_compute_storage_accounting.json").write_text(
        json.dumps(accounting, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (output_dir / "SPARSE_STREAMING_ALLOCATOR_REPORT.md").write_text(
        build_report(tables, promotions, config)
    )
    manifest = {
        "schema_version": 1,
        "config_sha256": sha256_path(config_path),
        "validation_promotions_sha256": promotion_sha256,
        "input_sha256": dict(sorted(input_hashes.items())),
        "outputs": {},
    }
    output_paths = [
        promotions_path,
        *(output_dir / name for name in summaries),
        output_dir / "sparse_streaming_compute_storage_accounting.json",
        output_dir / "SPARSE_STREAMING_ALLOCATOR_REPORT.md",
        *sorted(plots.glob("*.png")),
        *sorted(plots.glob("*.svg")),
    ]
    manifest["outputs"] = {
        str(path.relative_to(output_dir)): sha256_path(path) for path in sorted(output_paths)
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest
