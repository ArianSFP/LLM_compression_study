"""Outcome-only admission and analysis for nested D1 Experiment A.

This module is deliberately CPU-only.  It authenticates an allocation seal
before exposing complete selected states to a downstream-tail runner, joins
those states to the independently pinned request manifest, and computes
paired route-mode outcome summaries.  It contains no allocation search or D1
policy-selection code.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .d1_nested_artifacts import (
    canonical_sha256,
    decode_state,
    load_sealed_allocation_manifest,
    state_page_count,
    validate_frozen_calibration_spec,
)


ROUTE_MODES = ("live", "frozen_set_live_weights", "fully_frozen")
ALLOCATION_CONFIG_SCHEMA = "pr13_d1_nested_safe_allocation_config_v1"
ALLOCATION_CONFIG_RUN_ID = (
    "qwen36_mxfp4_d1_nested_safe_allocation_20260825_v2"
)
ALLOCATION_CONFIG_V1_PATH = (
    "configs/qwen36_mxfp4_d1_nested_safe_allocation_20260825_v1.json"
)
ALLOCATION_CONFIG_FILE_SHA256 = (
    "f800d43c047af5ef34d3872d5c04856ded12d3509718d1a1f46af2e40d34a599"
)
ALLOCATION_CONFIG_CANONICAL_SHA256 = (
    "927f847b20dd68ad91d43ccf46c4e70f05f4aec750dbb6b1fbbfc2a7bcdcf223"
)
AUTHENTICATED_CAPTURE_CONFIG_PATH = (
    "configs/qwen36_mxfp4_d1_nested_safe_oracle_20260825_v1.json"
)
AUTHENTICATED_CAPTURE_CONFIG_SHA256 = (
    "c37eb2c63de64dba87baa44b0ef57fc2206b32db8f82dfc2d49a9e9da0da1713"
)
OUTCOME_SCHEMA = "pr13_d1_nested_cached_decode_outcomes_v1"
QUALITY_FILE = "d1_nested_outcome_quality.parquet"
PROPAGATION_FILE = "d1_nested_outcome_propagation.parquet"
CACHE_FILE = "d1_nested_outcome_cache.parquet"
ZERO_FILE = "d1_nested_outcome_zero_gates.parquet"
CONTRAST_FILE = "d1_nested_outcome_route_mode_contrasts.parquet"
RUN_FACTS_FILE = "d1_nested_outcome_run_facts.json"


@dataclass(frozen=True)
class OutcomeAllocation:
    """One authenticated, allocation-only current-token state."""

    arm: str
    rate: int
    layer: int
    request_id: str
    split: str
    expert_ids: tuple[int, ...]
    selected_state: np.ndarray
    selected_pages: int
    page_cap: int
    prompt_sha256: str | None
    domain: str | None
    position: int | None
    state_sha256: str

    @property
    def identity(self) -> tuple[str, int, int, str]:
        return (self.arm, self.rate, self.layer, self.request_id)


@dataclass(frozen=True)
class OutcomePlan:
    """Authenticated allocations plus immutable seal identities."""

    allocations: tuple[OutcomeAllocation, ...]
    allocation_manifest_sha256: str
    frozen_calibration_spec_sha256: str
    allocation_config_file_sha256: str
    request_manifest_sha256: str
    request_manifest_facts_sha256: str
    allocation_config_canonical_sha256: str
    calibration_candidate_code_identity: dict[str, Any]

    def for_split(self, split: str) -> tuple[OutcomeAllocation, ...]:
        if split not in {"calibration", "evaluation"}:
            raise ValueError("outcome split must be calibration or evaluation")
        selected = tuple(row for row in self.allocations if row.split == split)
        if not selected:
            raise ValueError(f"sealed allocation manifest contains no {split} rows")
        return selected


@dataclass(frozen=True)
class OutcomeRequest:
    request_id: str
    split: str
    domain: str
    prompt_sha256: str
    prompt_token_ids: tuple[int, ...]
    position: int
    next_token_id: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"frozen calibration {field} is invalid")
    return value


def _required_candidate_code_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema", "files", "canonical_sha256",
    }:
        raise ValueError("calibration candidate code identity fields changed")
    if value.get("schema") != "pr13_d1_nested_code_bundle_v1":
        raise ValueError("calibration candidate code identity schema changed")
    files = value.get("files")
    if isinstance(files, bool) or not isinstance(files, int) or files < 1:
        raise ValueError("calibration candidate code identity file count is invalid")
    return {
        "schema": "pr13_d1_nested_code_bundle_v1",
        "files": int(files),
        "canonical_sha256": _required_sha256(
            value.get("canonical_sha256"),
            field="calibration candidate code canonical SHA-256",
        ),
    }


def _require_contract_fields(
    config: Mapping[str, Any],
    field: str,
    expected: Mapping[str, Any],
) -> Mapping[str, Any]:
    value = config.get(field)
    if not isinstance(value, Mapping):
        raise ValueError(f"frozen {field} contract is absent")
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"frozen {field} contract changed: {key}")
    return value


def validate_outcome_config(config: Mapping[str, Any]) -> None:
    """Require the frozen Experiment-A outcome boundary, not policy knobs."""

    if config.get("schema") != ALLOCATION_CONFIG_SCHEMA:
        raise ValueError("unexpected nested Experiment A allocation config schema")
    if config.get("experiment_stage") != "A":
        raise ValueError("nested outcome runner is restricted to Experiment A")
    if config.get("run_id") != ALLOCATION_CONFIG_RUN_ID:
        raise ValueError("outcome runner requires the immutable v2 allocation config")
    capture = config.get("authenticated_capture_protocol")
    if (
        not isinstance(capture, Mapping)
        or set(capture) != {"path", "sha256"}
        or capture.get("path") != AUTHENTICATED_CAPTURE_CONFIG_PATH
        or capture.get("sha256") != AUTHENTICATED_CAPTURE_CONFIG_SHA256
    ):
        raise ValueError("authenticated capture protocol pin changed")
    _require_contract_fields(config, "configuration_provenance", {
        "materialization": "standalone_full_config_v1",
        "parent_path": ALLOCATION_CONFIG_V1_PATH,
        "parent_sha256": (
            "fef7b21ecdebcfb68d3f63c6a10575db4ce1163d72ca410e107626c34d061480"
        ),
    })
    request_source = config.get("request_source")
    if not isinstance(request_source, Mapping):
        raise ValueError("frozen request source contract is absent")
    capture_artifacts = _require_contract_fields(
        config,
        "authenticated_capture_artifacts",
        {
            "directory": (
                "/workspace/pr13_d1_nested_safe_oracle_20260825_v1/"
                "captures_authenticated"
            ),
            "facts_path": (
                "/workspace/pr13_d1_nested_safe_oracle_20260825_v1/"
                "captures_authenticated/capture_facts_all.json"
            ),
            "facts_schema": "pr13_d1_nested_exact_decode_capture_facts_v1",
            "facts_sha256": (
                "65a1ae5dc5a149d8855f4bef4001181058ed66a4df5302877e6a0e859d543643"
            ),
            "split": "all",
            "requests": 128,
            "files": 128,
            "manifest_sha256": (
                "a2934ea02e3ee9c01f566c12969e81baf83b0bf30ee18d0fadeb5bff46a0c546"
            ),
            "manifest_facts_sha256": (
                "43732f6b4efd3d26df382936ea40fd1d5ea04d0ac9231810c10433f4dcbb9b2c"
            ),
        },
    )
    if capture_artifacts.get("requests") != request_source.get("total_requests"):
        raise ValueError("authenticated capture request count changed")
    _require_contract_fields(config, "authenticated_capture_execution_stack", {
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
    })
    pr13_inputs = _require_contract_fields(
        config,
        "authenticated_pr13_inputs",
        {
            "config_path": "configs/qwen36_mxfp4_average_rate_all_layers.json",
            "config_sha256": (
                "56ec0ba55610a0542603d758597129fd3180fd944d973d1b367b7a0bc1a5d006"
            ),
            "fit_dir": "/workspace/pr13_average_rate_all_layers_20260822_v1/results/fit",
            "factor_manifest_path": (
                "/workspace/pr13_average_rate_all_layers_20260822_v1/results/fit/"
                "average_rate_factor_manifest.json"
            ),
            "factor_manifest_schema": "average_rate_all_layers_distributed_v1",
            "factor_manifest_sha256": (
                "60204faca5d9d7974e035ff05088938af942fbd5e0d313c030dddc6748157069"
            ),
            "tree_sha256": (
                "da4675181fd81c87bd2d3db7ff2d3dea1f075db4ea98f23ab90f14943ee2bfe8"
            ),
            "checkpoint_config_sha256": (
                "52411f11bf654a1a5b3bfb15c84be894ee57d97e95dbdd37656f6c75846edc1a"
            ),
            "checkpoint_index_sha256": (
                "842c9ba65c2bebe47cc834eb8e8a744b7ba8f610b9fc8cd751cbc7a4564d39fb"
            ),
        },
    )
    if any((
        pr13_inputs.get("config_path") != config.get("pr13_config"),
        pr13_inputs.get("fit_dir") != config.get("fit_dir"),
        pr13_inputs.get("tree_sha256") != config.get("selected_tree_sha256"),
        pr13_inputs.get("checkpoint_config_sha256")
        != config.get("checkpoint_config_sha256"),
        pr13_inputs.get("checkpoint_index_sha256")
        != config.get("checkpoint_index_sha256"),
    )):
        raise ValueError("authenticated PR13 inputs disagree with the root config")
    _require_contract_fields(config, "reference_high_core_seal", {
        "required": True,
        "reference_arm": "independent_pr13",
        "reference_endpoint": "paired_reference_high",
        "applies_to_arms": [
            "nested_local_only", "nested_d1_safe",
            "nested_d1_safe_violation_mass",
        ],
        "literal_subset_and_state_hash_required": True,
    })
    _require_contract_fields(config, "atomic_resume", {
        "schema": "pr13_d1_nested_atomic_layer_resume_v1",
        "full_request_split_only": True,
        "facts_written_last": True,
        "scientific_files_per_layer": [
            "nested_candidates.pkl", "nested_candidate_metrics.parquet",
            "nested_slice_parity.parquet",
        ],
        "completed_global_requires_exact_configured_layer_set": True,
        "mixed_input_or_code_identity_rejected": True,
    })
    outcome = config.get("outcome_phase")
    if not isinstance(outcome, Mapping):
        raise ValueError("outcome phase is absent")
    expected = {
        "requires_sealed_allocation_manifest": True,
        "route_modes": list(ROUTE_MODES),
        "terminal_metric_scope": "current_token_only",
        "post_token_complete_cache_retained": True,
        "zero_dose_bit_exact_gate": True,
        "candidate_cache_reused": False,
    }
    for key, value in expected.items():
        if outcome.get(key) != value:
            raise ValueError(f"frozen outcome contract changed: {key}")
    if config.get("experiment_b_started") is not False:
        raise ValueError("Experiment B must remain paused")
    if config.get("generated_rollout_in_scope") is not False:
        raise ValueError("generated rollout is outside this outcome study")
    if config.get("joint_all_layer_compression_in_scope") is not False:
        raise ValueError("joint all-layer compression is outside this outcome study")
    if canonical_sha256(config) != ALLOCATION_CONFIG_CANONICAL_SHA256:
        raise ValueError(
            "immutable v2 allocation config canonical SHA-256 changed"
        )


def _optional_string(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a non-empty string when present")
    return value


def _optional_integer(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer when present")
    return int(value)


def load_outcome_plan(
    *,
    manifest_path: Path,
    seal_path: Path,
    expected_manifest_sha256: str | None = None,
    expected_frozen_calibration_spec_sha256: str | None = None,
) -> OutcomePlan:
    """Authenticate the raw seal/manifest before decoding any selected state."""

    manifest = load_sealed_allocation_manifest(
        manifest_path=Path(manifest_path),
        seal_path=Path(seal_path),
        expected_manifest_sha256=expected_manifest_sha256,
        expected_frozen_calibration_spec_sha256=(
            expected_frozen_calibration_spec_sha256
        ),
    )
    digest = file_sha256(Path(manifest_path))
    frozen_digest = validate_frozen_calibration_spec(manifest["frozen_calibration"])
    frozen_spec = manifest["frozen_calibration"]["spec"]
    config_file_sha256 = _required_sha256(
        frozen_spec.get("config_file_sha256"),
        field="allocation config file SHA-256",
    )
    frozen_config_canonical_sha256 = _required_sha256(
        frozen_spec.get("config_canonical_sha256"),
        field="allocation config canonical SHA-256",
    )
    request_manifest_sha256 = _required_sha256(
        frozen_spec.get("request_manifest_sha256"),
        field="request manifest SHA-256",
    )
    request_manifest_facts_sha256 = _required_sha256(
        frozen_spec.get("request_manifest_facts_sha256"),
        field="request manifest facts SHA-256",
    )
    calibration_candidate_code_identity = _required_candidate_code_identity(
        frozen_spec.get("calibration_candidate_code_identity")
    )
    allocation_inputs = manifest.get("allocation_inputs")
    if not isinstance(allocation_inputs, Mapping):
        raise ValueError("allocation manifest inputs must be an object")
    config_canonical_sha256 = _required_sha256(
        allocation_inputs.get("config_canonical_sha256"),
        field="allocation manifest config canonical SHA-256",
    )
    if config_canonical_sha256 != frozen_config_canonical_sha256:
        raise ValueError(
            "allocation manifest and frozen calibration config digests differ"
        )
    allocations: list[OutcomeAllocation] = []
    for index, raw in enumerate(manifest["allocations"]):
        state = np.asarray(decode_state(raw["selected_state"]), dtype=np.uint8)
        if state.shape != (8, 512):
            raise ValueError(f"allocation {index} selected state shape changed")
        page_cap = int(raw["selected_state"]["page_cap"])
        selected_pages = state_page_count(state)
        if selected_pages > page_cap or page_cap != int(raw["rate"]) * 8:
            raise ValueError(f"allocation {index} has inconsistent physical traffic")
        allocations.append(OutcomeAllocation(
            arm=str(raw["arm"]),
            rate=int(raw["rate"]),
            layer=int(raw["layer"]),
            request_id=str(raw["request_id"]),
            split=str(raw["split"]),
            expert_ids=tuple(map(int, raw["expert_ids"])),
            selected_state=state,
            selected_pages=selected_pages,
            page_cap=page_cap,
            prompt_sha256=_optional_string(
                raw.get("prompt_sha256"), field=f"allocations[{index}].prompt_sha256",
            ),
            domain=_optional_string(
                raw.get("domain"), field=f"allocations[{index}].domain",
            ),
            position=_optional_integer(
                raw.get("position"), field=f"allocations[{index}].position",
            ),
            state_sha256=hashlib.sha256(state.tobytes(order="C")).hexdigest(),
        ))
    identities = [row.identity for row in allocations]
    if len(set(identities)) != len(identities):
        raise ValueError("outcome plan repeats an allocation identity")
    return OutcomePlan(
        allocations=tuple(allocations),
        allocation_manifest_sha256=digest,
        frozen_calibration_spec_sha256=frozen_digest,
        allocation_config_file_sha256=config_file_sha256,
        request_manifest_sha256=request_manifest_sha256,
        request_manifest_facts_sha256=request_manifest_facts_sha256,
        allocation_config_canonical_sha256=config_canonical_sha256,
        calibration_candidate_code_identity=(
            calibration_candidate_code_identity
        ),
    )


def join_request_manifest(
    allocations: Sequence[OutcomeAllocation],
    request_rows: Sequence[Mapping[str, Any]],
) -> dict[str, OutcomeRequest]:
    """Join allocation identities to the pinned input-only request rows."""

    indexed: dict[str, OutcomeRequest] = {}
    for index, raw in enumerate(request_rows):
        request_id = str(raw.get("request_id", ""))
        if not request_id or request_id in indexed:
            raise ValueError("request manifest has an empty or duplicate request_id")
        tokens_raw = raw.get("prompt_token_ids")
        if not isinstance(tokens_raw, list) or not tokens_raw:
            raise ValueError(f"request row {index} has no token sequence")
        if any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in tokens_raw
        ):
            raise ValueError(f"request row {index} token IDs are invalid")
        position = raw.get("decode_position")
        next_token = raw.get("next_token_id")
        if (
            isinstance(position, bool) or not isinstance(position, int)
            or position < 1 or position + 1 >= len(tokens_raw)
            or isinstance(next_token, bool) or not isinstance(next_token, int)
            or next_token != tokens_raw[position + 1]
        ):
            raise ValueError(f"request row {index} decode label is invalid")
        split = str(raw.get("split", ""))
        if split not in {"calibration", "evaluation"}:
            raise ValueError(f"request row {index} split is invalid")
        domain = str(raw.get("domain", ""))
        prompt_sha = str(raw.get("prompt_sha256", ""))
        if not domain or len(prompt_sha) != 64:
            raise ValueError(f"request row {index} metadata is invalid")
        indexed[request_id] = OutcomeRequest(
            request_id=request_id,
            split=split,
            domain=domain,
            prompt_sha256=prompt_sha,
            prompt_token_ids=tuple(map(int, tokens_raw)),
            position=int(position),
            next_token_id=int(next_token),
        )

    admitted_ids = {row.request_id for row in allocations}
    missing = admitted_ids.difference(indexed)
    if missing:
        raise ValueError(f"sealed allocations reference unknown requests: {sorted(missing)}")
    for row in allocations:
        request = indexed[row.request_id]
        if request.split != row.split:
            raise ValueError(f"allocation/request split mismatch for {row.request_id}")
        if row.prompt_sha256 is not None and row.prompt_sha256 != request.prompt_sha256:
            raise ValueError(f"allocation/request prompt mismatch for {row.request_id}")
        if row.domain is not None and row.domain != request.domain:
            raise ValueError(f"allocation/request domain mismatch for {row.request_id}")
        if row.position is not None and row.position != request.position:
            raise ValueError(f"allocation/request position mismatch for {row.request_id}")
    return {request_id: indexed[request_id] for request_id in sorted(admitted_ids)}


def validate_complete_allocation_grid(
    allocations: Sequence[OutcomeAllocation],
) -> tuple[tuple[str, int, int], ...]:
    """Require every admitted request to carry the identical arm/rate/layer grid."""

    if not allocations:
        raise ValueError("outcome allocation grid is empty")
    request_splits: dict[str, str] = {}
    by_request: dict[str, set[tuple[str, int, int]]] = {}
    for row in allocations:
        previous = request_splits.setdefault(row.request_id, row.split)
        if previous != row.split:
            raise ValueError("one request appears in both splits")
        grid = by_request.setdefault(row.request_id, set())
        key = (row.arm, row.rate, row.layer)
        if key in grid:
            raise ValueError("allocation grid contains a duplicate request cell")
        grid.add(key)
    template: set[tuple[str, int, int]] | None = None
    for request_id in sorted(by_request):
        if template is None:
            template = by_request[request_id]
        elif by_request[request_id] != template:
            raise ValueError(f"allocation grid is incomplete for request {request_id}")
    assert template is not None
    return tuple(sorted(template, key=lambda value: (value[2], value[0], value[1])))


def validate_plan_against_config(
    plan: OutcomePlan,
    config: Mapping[str, Any],
    *,
    config_file_sha256: str | None = None,
) -> None:
    """Bind the sealed evaluation grid to the frozen arm/rate/layer contract."""

    validate_outcome_config(config)
    if plan.allocation_config_file_sha256 != ALLOCATION_CONFIG_FILE_SHA256:
        raise ValueError(
            "sealed allocation manifest is not bound to the immutable raw v2 config"
        )
    if (
        config_file_sha256 is not None
        and plan.allocation_config_file_sha256
        != _required_sha256(config_file_sha256, field="supplied config file SHA-256")
    ):
        raise ValueError(
            "sealed allocation manifest is not bound to the raw allocation config"
        )
    observed_canonical_sha256 = canonical_sha256(config)
    if plan.allocation_config_canonical_sha256 != observed_canonical_sha256:
        raise ValueError(
            "sealed allocation manifest is not bound to the frozen allocation config"
        )
    allocations = plan.allocations
    if not allocations or {row.split for row in allocations} != {"evaluation"}:
        raise ValueError("sealed outcome plan must contain evaluation allocations only")
    expected_arms = tuple(map(str, config.get("arms", ())))
    expected_layers = tuple(map(int, config.get("injection_layers", ())))
    pair_rows = config.get("traffic_pairs")
    if not expected_arms or not expected_layers or not isinstance(pair_rows, list):
        raise ValueError("frozen arm/rate/layer grid is absent")
    expected_rates = tuple(sorted({
        int(row[field])
        for row in pair_rows
        for field in (
            "metadata_matched_pages_per_expert",
            "pr13_reference_pages_per_expert",
        )
    }))
    observed_arms = {row.arm for row in allocations}
    observed_rates = {row.rate for row in allocations}
    observed_layers = {row.layer for row in allocations}
    if observed_arms != set(expected_arms):
        raise ValueError("sealed outcome arm grid differs from the frozen config")
    if observed_rates != set(expected_rates):
        raise ValueError("sealed outcome rate grid differs from the frozen config")
    if observed_layers != set(expected_layers):
        raise ValueError("sealed outcome layer grid differs from the frozen config")
    requests = {row.request_id for row in allocations}
    expected_requests = int(config["request_source"]["evaluation_requests"])
    if len(requests) != expected_requests:
        raise ValueError(
            f"sealed evaluation request count changed: {len(requests)} != {expected_requests}"
        )
    observed_grid = validate_complete_allocation_grid(allocations)
    expected_grid = {
        (arm, rate, layer)
        for arm in expected_arms
        for rate in expected_rates
        for layer in expected_layers
    }
    if set(observed_grid) != expected_grid:
        raise ValueError("sealed allocation grid is not the frozen full Cartesian grid")
    expected_rows = len(requests) * len(expected_grid)
    if len(allocations) != expected_rows:
        raise ValueError("sealed allocation row count differs from the frozen grid")


def route_mode_contrasts(quality: pd.DataFrame) -> pd.DataFrame:
    """Return one paired row per allocation without selecting on outcomes."""

    identity = [
        "split", "arm", "rate_pages_per_expert", "injection_layer",
        "request_id", "position",
    ]
    required = set(identity + ["route_mode", "logit_kl"])
    missing = sorted(required.difference(quality.columns))
    if missing:
        raise ValueError(f"quality table is missing columns: {missing}")
    if quality.empty or quality[identity + ["route_mode"]].isna().any().any():
        raise ValueError("quality route-mode identity is empty or incomplete")
    if quality.duplicated(identity + ["route_mode"]).any():
        raise ValueError("quality route-mode identity is not unique")
    modes = set(map(str, quality["route_mode"].unique()))
    if modes != set(ROUTE_MODES):
        raise ValueError(f"quality route modes changed: {sorted(modes)}")
    pivot = quality.pivot(index=identity, columns="route_mode", values="logit_kl")
    if pivot.isna().any().any():
        raise ValueError("a candidate is missing a paired route-mode outcome")
    result = pivot.reset_index().rename(columns={
        "live": "live_logit_kl",
        "frozen_set_live_weights": "frozen_set_live_weights_logit_kl",
        "fully_frozen": "fully_frozen_logit_kl",
    })
    result["live_minus_fully_frozen_logit_kl"] = (
        result["live_logit_kl"] - result["fully_frozen_logit_kl"]
    )
    result["live_minus_frozen_set_live_weights_logit_kl"] = (
        result["live_logit_kl"] - result["frozen_set_live_weights_logit_kl"]
    )
    result["frozen_set_live_weights_minus_fully_frozen_logit_kl"] = (
        result["frozen_set_live_weights_logit_kl"]
        - result["fully_frozen_logit_kl"]
    )
    return result.sort_values(identity, kind="stable").reset_index(drop=True)


def require_zero_dose_parity(zero: pd.DataFrame) -> None:
    """Fail unless outputs and every classified hybrid-cache commit are exact."""

    required_boolean = (
        "all_exact",
        "paired_baseline_full_cache_bit_identical",
        "post_token_full_cache_bit_identical",
        "post_token_attention_kv_bit_identical",
        "post_token_deltanet_conv_bit_identical",
        "post_token_deltanet_recurrent_bit_identical",
    )
    required = set(required_boolean + ("route_mode", "routed_repeat_max_abs"))
    missing = sorted(required.difference(zero.columns))
    if missing:
        raise ValueError(f"zero-dose table is missing columns: {missing}")
    if zero.empty or set(map(str, zero["route_mode"].unique())) != set(ROUTE_MODES):
        raise ValueError("zero-dose route-mode grid is incomplete")
    cell_identity = ["split", "injection_layer", "request_id", "position"]
    if set(cell_identity).issubset(zero.columns):
        if zero.duplicated(cell_identity + ["route_mode"]).any():
            raise ValueError("zero-dose table repeats a route-mode identity")
        for identity, rows in zero.groupby(cell_identity, sort=False, dropna=False):
            if set(map(str, rows["route_mode"])) != set(ROUTE_MODES):
                raise ValueError(f"zero-dose route-mode grid is incomplete for {identity}")
    for column in required_boolean:
        values = zero[column].tolist()
        if any(
            not isinstance(value, (bool, np.bool_)) or not bool(value)
            for value in values
        ):
            raise RuntimeError(f"zero-dose parity failed: {column}")
    repeat = zero["routed_repeat_max_abs"].to_numpy(np.float64)
    if np.any(~np.isfinite(repeat)) or np.any(repeat != 0.0):
        raise RuntimeError("zero-dose routed replacement is not bit-repeatable")


def request_layer_means(
    quality: pd.DataFrame,
    value_columns: Sequence[str],
) -> pd.DataFrame:
    """Average declared layers equally, retaining route modes and requests."""

    identity = [
        "split", "request_id", "arm", "rate_pages_per_expert",
        "route_mode", "injection_layer",
    ]
    missing = sorted(set(identity).union(value_columns).difference(quality.columns))
    if missing:
        raise ValueError(f"quality table is missing columns: {missing}")
    if quality.empty or quality.duplicated(identity).any():
        raise ValueError("quality table is empty or has duplicate candidate rows")
    numeric = quality[list(value_columns)].to_numpy(np.float64)
    if np.any(~np.isfinite(numeric)):
        raise ValueError("quality outcomes must be finite")
    expected_layers = set(map(int, quality["injection_layer"].unique()))
    expected_cells: set[tuple[str, int, str]] | None = None
    for request_id, rows in quality.groupby("request_id", sort=False):
        cells = set(zip(
            rows["arm"].astype(str),
            rows["rate_pages_per_expert"].astype(int),
            rows["route_mode"].astype(str),
            strict=True,
        ))
        if expected_cells is None:
            expected_cells = cells
        elif cells != expected_cells:
            raise ValueError(f"quality grid is incomplete for request {request_id}")
        for cell, part in rows.groupby(
            ["arm", "rate_pages_per_expert", "route_mode"], sort=False,
        ):
            if set(map(int, part["injection_layer"])) != expected_layers:
                raise ValueError(
                    f"quality layer grid is incomplete for request {request_id}, cell {cell}"
                )
    keys = ["split", "request_id", "arm", "rate_pages_per_expert", "route_mode"]
    result = (
        quality.groupby(keys, sort=True, dropna=False)[list(value_columns)]
        .mean()
        .reset_index()
    )
    result["layers_averaged"] = len(expected_layers)
    return result


def outcome_input_facts(plan: OutcomePlan) -> dict[str, Any]:
    """Small deterministic identity record suitable for run facts."""

    return {
        "allocation_manifest_sha256": plan.allocation_manifest_sha256,
        "frozen_calibration_spec_sha256": plan.frozen_calibration_spec_sha256,
        "allocation_config_file_sha256": plan.allocation_config_file_sha256,
        "allocation_config_canonical_sha256": (
            plan.allocation_config_canonical_sha256
        ),
        "calibration_candidate_code_identity": dict(
            plan.calibration_candidate_code_identity
        ),
        "allocation_identities_sha256": canonical_sha256([
            [row.arm, row.rate, row.layer, row.request_id, row.state_sha256]
            for row in sorted(plan.allocations, key=lambda item: item.identity)
        ]),
        "allocations": len(plan.allocations),
    }


def physical_execution_identity(
    allocation: OutcomeAllocation,
    delta: np.ndarray,
) -> dict[str, str]:
    """Hash the exact expert/state mapping and live float32 injected delta.

    Arm, rate and allocator parameters are intentionally excluded.  They are
    logical labels; identical physical states and deltas within one
    request/layer require only one downstream execution.
    """

    state = np.asarray(allocation.selected_state, np.uint8)
    if state.shape != (8, 512) or len(allocation.expert_ids) != 8:
        raise ValueError("physical execution identity requires an [8,512] state")
    if len(set(allocation.expert_ids)) != 8:
        raise ValueError("physical execution identity requires unique expert IDs")
    canonical_rows = sorted(
        zip(allocation.expert_ids, state, strict=True), key=lambda pair: int(pair[0]),
    )
    state_payload = b"".join(
        int(expert).to_bytes(2, "little", signed=False)
        + np.ascontiguousarray(row, dtype=np.uint8).tobytes(order="C")
        for expert, row in canonical_rows
    )
    value = np.ascontiguousarray(np.asarray(delta, np.float32).reshape(-1))
    if value.size == 0 or np.any(~np.isfinite(value)):
        raise ValueError("physical execution delta must be nonempty and finite")
    state_digest = hashlib.sha256(state_payload).hexdigest()
    delta_digest = hashlib.sha256(value.tobytes(order="C")).hexdigest()
    execution_digest = canonical_sha256({
        "schema": "pr13_d1_nested_physical_execution_identity_v1",
        "request_id": allocation.request_id,
        "layer": allocation.layer,
        "expert_state_sha256": state_digest,
        "float32_delta_sha256": delta_digest,
    })
    return {
        "physical_execution_id": execution_digest,
        "physical_expert_state_sha256": state_digest,
        "physical_delta_sha256": delta_digest,
    }
