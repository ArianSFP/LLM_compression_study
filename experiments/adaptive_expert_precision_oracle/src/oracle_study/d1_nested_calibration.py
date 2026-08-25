"""Calibration and immutable sealing for nested D1 Experiment A.

This module is the allocation/outcome firewall.  It loads trusted, internal
per-layer candidate pickles, admits only the frozen D1/local candidate schema,
selects one repair-window/eta pair on calibration requests, and converts the
untouched evaluation allocations to the strict canonical artifact contract.

Selection is deliberately traffic-pair-level: Arm 4 is calibrated at the
metadata-matched low endpoint and its parameters are inherited by the paired
high endpoint and by Arm 5.  Choosing the two endpoints independently could
change their common core and would make literal low-to-high nesting impossible.
No terminal quality, router after D1, final hidden, or cache outcome is admitted.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .d1_nested_artifacts import (
    ALLOCATION_MANIFEST_SCHEMA,
    ArtifactValidationError,
    atomic_write_json,
    canonical_json_bytes,
    canonical_sha256,
    encode_state,
    freeze_calibration_spec,
    physical_subset,
    reject_outcome_leakage,
    state_page_count,
    validate_allocation_manifest,
    validate_frozen_calibration_spec,
    write_sealed_allocation_manifest,
)


CANDIDATE_SCHEMA = "pr13_d1_nested_safe_oracle_layer_candidates_v1"
CALIBRATION_EVIDENCE_SCHEMA = "pr13_d1_nested_calibration_evidence_v1"
CALIBRATION_SPEC_SCHEMA = "pr13_d1_nested_pair_calibration_spec_v1"
ALLOCATION_RECORD_SCHEMA = "pr13_d1_nested_sealed_allocation_record_v1"

ARM_PR13 = "independent_pr13"
ARM_STRICT = "strict_frozen_candidate_incumbent_repair"
ARM_LOCAL = "nested_local_only"
ARM_D1 = "nested_d1_safe"
ARM_D1_SEVERITY = "nested_d1_safe_violation_mass"
FIVE_ARMS = (ARM_PR13, ARM_STRICT, ARM_LOCAL, ARM_D1, ARM_D1_SEVERITY)
NESTED_ARMS = (ARM_LOCAL, ARM_D1, ARM_D1_SEVERITY)

SELECTION_ORDER = (
    "zero_safe_to_unsafe_events",
    "fewest_unresolved_exact_d1_crossings",
    "smallest_repair_window_pages",
    "smallest_eta",
    "lowest_mean_exact_combined_local_qenergy",
    "deterministic_parameter_hash",
)

AUTHENTICATED_CAPTURE_CONFIG_PATH = (
    "configs/qwen36_mxfp4_d1_nested_safe_oracle_20260825_v1.json"
)
AUTHENTICATED_CAPTURE_CONFIG_SHA256 = (
    "c37eb2c63de64dba87baa44b0ef57fc2206b32db8f82dfc2d49a9e9da0da1713"
)

CANDIDATE_FIELDS = {
    "schema",
    "arm",
    "rate",
    "layer",
    "group",
    "request_id",
    "prompt_sha256",
    "domain",
    "split",
    "position",
    "expert_ids",
    "selector_weights",
    "execution_weights",
    "common_core_states",
    "selected_states",
    "moves",
    "selected_pages",
    "core_pages",
    "local_damage",
    "all_q2_damage",
    "legacy_additive_damage",
    "d1_candidate_logits",
    "d1_crossings",
    "d1_violation_depth",
    "d1_routing_mass_churn",
    "d1_labeled_margin",
    "parameter",
    "state_sha256",
}

MOVE_REQUIRED_FIELDS = {
    "expert", "unit", "bit", "source_state", "destination_state",
}
MOVE_OPTIONAL_FIELDS = {"step", "expert_id", "projection", "page_delta"}
MOVE_FIELDS = MOVE_REQUIRED_FIELDS | MOVE_OPTIONAL_FIELDS
HEX_64 = re.compile(r"^[0-9a-f]{64}$")


__all__ = [
    "CANDIDATE_SCHEMA",
    "CALIBRATION_EVIDENCE_SCHEMA",
    "CALIBRATION_SPEC_SCHEMA",
    "ALLOCATION_RECORD_SCHEMA",
    "FIVE_ARMS",
    "CandidateSource",
    "CandidateRecord",
    "CandidateCorpus",
    "CalibrationSelection",
    "load_layer_candidate_pickles",
    "select_and_freeze_calibration",
    "write_calibration_selection",
    "validate_calibration_selection",
    "build_evaluation_allocation_manifest",
    "write_sealed_evaluation_allocations",
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: Any, field: str) -> str:
    result = str(value).lower()
    if not HEX_64.fullmatch(result):
        raise ArtifactValidationError(f"{field} must be a lowercase SHA-256")
    return result


def _integer(value: Any, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ArtifactValidationError(f"{field} must be an integer")
    result = int(value)
    if result < minimum:
        raise ArtifactValidationError(f"{field} must be at least {minimum}")
    return result


def _finite(value: Any, field: str, *, nonnegative: bool = True) -> float:
    result = float(value)
    if not np.isfinite(result) or (nonnegative and result < 0.0):
        suffix = " nonnegative" if nonnegative else ""
        raise ArtifactValidationError(f"{field} must be finite{suffix}")
    return result


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ArtifactValidationError(f"{field} must be a nonempty trimmed string")
    return value


def _json_value(value: Any, field: str) -> Any:
    def convert(item: Any) -> Any:
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Mapping):
            return {str(key): convert(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(child) for child in item]
        return item

    converted = convert(value)
    try:
        return json.loads(canonical_json_bytes(converted))
    except ArtifactValidationError as exc:
        raise ArtifactValidationError(f"{field} is not canonical JSON: {exc}") from exc


def _state(value: Any, field: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (8, 512) or raw.dtype.kind not in "iu":
        raise ArtifactValidationError(f"{field} must be an integer [8,512] state")
    if np.any((raw < 0) | (raw > 7)):
        raise ArtifactValidationError(f"{field} contains a state outside [0,7]")
    result = np.asarray(raw, np.uint8).copy()
    result.setflags(write=False)
    return result


def _weights(value: Any, field: str) -> tuple[float, ...]:
    result = np.asarray(value, np.float64).reshape(-1)
    if result.shape != (8,) or np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise ArtifactValidationError(f"{field} must contain eight positive weights")
    return tuple(map(float, result))


def _expert_ids(value: Any) -> tuple[int, ...]:
    array = np.asarray(value)
    if array.shape != (8,) or array.dtype.kind not in "iu":
        raise ArtifactValidationError("expert_ids must contain eight integers")
    result = tuple(map(int, array))
    if len(set(result)) != 8 or min(result) < 0 or max(result) >= 256:
        raise ArtifactValidationError("expert_ids must be eight unique IDs in [0,255]")
    return result


def _normalise_moves(
    value: Any,
    *,
    core: np.ndarray,
    selected: np.ndarray,
    expert_ids: tuple[int, ...],
) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        raise ArtifactValidationError("candidate moves must be a sequence")
    current = np.asarray(core, np.uint8).copy()
    result: list[dict[str, Any]] = []
    names = {1: "down", 2: "up", 4: "gate"}
    for index, raw in enumerate(value):
        path = f"moves[{index}]"
        if not isinstance(raw, Mapping):
            raise ArtifactValidationError(f"{path} must be an object")
        missing = MOVE_REQUIRED_FIELDS - set(raw)
        extra = set(raw) - MOVE_FIELDS
        if missing or extra:
            raise ArtifactValidationError(
                f"{path} fields differ; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        expert = _integer(raw["expert"], f"{path}.expert")
        unit = _integer(raw["unit"], f"{path}.unit")
        bit = _integer(raw["bit"], f"{path}.bit", 1)
        source = _integer(raw["source_state"], f"{path}.source_state")
        destination = _integer(raw["destination_state"], f"{path}.destination_state")
        if expert >= 8 or unit >= 512 or bit not in names:
            raise ArtifactValidationError(f"{path} indexes or projection bit are invalid")
        if source > 7 or destination > 7 or int(current[expert, unit]) != source:
            raise ArtifactValidationError(f"{path} breaks the literal state chain")
        if source & bit or destination != source | bit:
            raise ArtifactValidationError(f"{path} is not one add-only page")
        move = {
            "step": index,
            "expert": expert,
            "expert_id": expert_ids[expert],
            "unit": unit,
            "bit": bit,
            "projection": names[bit],
            "source_state": source,
            "destination_state": destination,
            "page_delta": 1,
        }
        for key in MOVE_OPTIONAL_FIELDS & set(raw):
            if key in {"step", "expert_id", "page_delta"}:
                expected = move[key]
                if _integer(raw[key], f"{path}.{key}") != expected:
                    raise ArtifactValidationError(f"{path}.{key} changed")
            elif str(raw[key]) != move[key]:
                raise ArtifactValidationError(f"{path}.projection changed")
        current[expert, unit] = destination
        result.append(move)
    if not np.array_equal(current, selected):
        raise ArtifactValidationError("candidate moves do not reproduce selected state")
    return tuple(result)


@dataclass(frozen=True)
class CandidateSource:
    path: str
    sha256: str
    byte_count: int
    layer: int
    split: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "bytes": self.byte_count,
            "layer": self.layer,
            "split": self.split,
        }


@dataclass(frozen=True)
class CandidateRecord:
    arm: str
    rate: int
    layer: int
    group: int
    request_id: str
    prompt_sha256: str
    domain: str
    split: str
    position: int
    expert_ids: tuple[int, ...]
    selector_weights: tuple[float, ...]
    execution_weights: tuple[float, ...]
    common_core: np.ndarray
    selected_state: np.ndarray
    moves: tuple[dict[str, Any], ...]
    selected_pages: int
    core_pages: int
    local_damage: float
    all_q2_damage: float
    legacy_additive_damage: float
    d1_crossings: int
    d1_violation_depth: float
    d1_routing_mass_churn: float
    d1_labeled_margin: float
    parameter: dict[str, Any]
    state_sha256: str
    source: CandidateSource
    source_index: int

    @property
    def cell(self) -> tuple[int, str]:
        return self.layer, self.request_id

    @property
    def identity(self) -> tuple[str, int, int, str]:
        return self.arm, self.rate, self.layer, self.request_id

    @property
    def source_locator_sha256(self) -> str:
        return canonical_sha256({
            "source_sha256": self.source.sha256,
            "source_index": self.source_index,
        })


@dataclass(frozen=True)
class CandidateCorpus:
    records: tuple[CandidateRecord, ...]
    sources: tuple[CandidateSource, ...]
    split: str


@dataclass(frozen=True)
class CalibrationSelection:
    frozen_calibration: dict[str, Any]
    evidence: dict[str, Any]

    @property
    def sha256(self) -> str:
        return validate_frozen_calibration_spec(self.frozen_calibration)


def _candidate_record(
    raw: Any,
    *,
    source: CandidateSource,
    index: int,
) -> CandidateRecord:
    path = f"{source.path}[{index}]"
    if not isinstance(raw, Mapping):
        raise ArtifactValidationError(f"candidate {path} must be an object")
    reject_outcome_leakage(raw, path=f"$.candidate[{index}]")
    missing = CANDIDATE_FIELDS - set(raw)
    extra = set(raw) - CANDIDATE_FIELDS
    if missing or extra:
        raise ArtifactValidationError(
            f"candidate fields differ; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if raw["schema"] != CANDIDATE_SCHEMA:
        raise ArtifactValidationError(f"candidate {path} has an unexpected schema")
    arm = _text(raw["arm"], f"{path}.arm")
    if arm not in FIVE_ARMS:
        raise ArtifactValidationError(f"candidate {path} has an undeclared arm")
    rate = _integer(raw["rate"], f"{path}.rate", 1)
    layer = _integer(raw["layer"], f"{path}.layer")
    if layer != source.layer:
        raise ArtifactValidationError(f"candidate {path} changed its source layer")
    split = _text(raw["split"], f"{path}.split")
    if split != source.split:
        raise ArtifactValidationError(f"candidate {path} changed its source split")
    expert_ids = _expert_ids(raw["expert_ids"])
    core = _state(raw["common_core_states"], f"{path}.common_core_states")
    selected = _state(raw["selected_states"], f"{path}.selected_states")
    if not physical_subset(core, selected):
        raise ArtifactValidationError(f"candidate {path} clears its common core")
    selected_pages = _integer(raw["selected_pages"], f"{path}.selected_pages")
    core_pages = _integer(raw["core_pages"], f"{path}.core_pages")
    if selected_pages != state_page_count(selected) or core_pages != state_page_count(core):
        raise ArtifactValidationError(f"candidate {path} page counts changed")
    if selected_pages > rate * 8:
        raise ArtifactValidationError(f"candidate {path} exceeds its pooled page cap")
    state_digest = _sha256(raw["state_sha256"], f"{path}.state_sha256")
    observed_digest = hashlib.sha256(selected.tobytes(order="C")).hexdigest()
    if state_digest != observed_digest:
        raise ArtifactValidationError(f"candidate {path} state SHA-256 changed")
    logits = np.asarray(raw["d1_candidate_logits"], np.float64).reshape(-1)
    if logits.shape != (256,) or np.any(~np.isfinite(logits)):
        raise ArtifactValidationError(f"candidate {path} lacks all 256 finite D1 logits")
    crossings = _integer(raw["d1_crossings"], f"{path}.d1_crossings")
    if crossings > 8:
        raise ArtifactValidationError(f"candidate {path} has more than eight crossings")
    parameter = _json_value(raw["parameter"], f"{path}.parameter")
    if not isinstance(parameter, dict):
        raise ArtifactValidationError(f"candidate {path}.parameter must be an object")
    reject_outcome_leakage(parameter, path=f"$.candidate[{index}].parameter")
    moves = _normalise_moves(
        raw["moves"], core=core, selected=selected, expert_ids=expert_ids,
    )
    return CandidateRecord(
        arm=arm,
        rate=rate,
        layer=layer,
        group=_integer(raw["group"], f"{path}.group"),
        request_id=_text(raw["request_id"], f"{path}.request_id"),
        prompt_sha256=_sha256(raw["prompt_sha256"], f"{path}.prompt_sha256"),
        domain=_text(raw["domain"], f"{path}.domain"),
        split=split,
        position=_integer(raw["position"], f"{path}.position"),
        expert_ids=expert_ids,
        selector_weights=_weights(raw["selector_weights"], f"{path}.selector_weights"),
        execution_weights=_weights(raw["execution_weights"], f"{path}.execution_weights"),
        common_core=core,
        selected_state=selected,
        moves=moves,
        selected_pages=selected_pages,
        core_pages=core_pages,
        local_damage=_finite(raw["local_damage"], f"{path}.local_damage"),
        all_q2_damage=_finite(raw["all_q2_damage"], f"{path}.all_q2_damage"),
        legacy_additive_damage=_finite(
            raw["legacy_additive_damage"], f"{path}.legacy_additive_damage",
        ),
        d1_crossings=crossings,
        d1_violation_depth=_finite(
            raw["d1_violation_depth"], f"{path}.d1_violation_depth",
        ),
        d1_routing_mass_churn=_finite(
            raw["d1_routing_mass_churn"], f"{path}.d1_routing_mass_churn",
        ),
        d1_labeled_margin=_finite(
            raw["d1_labeled_margin"], f"{path}.d1_labeled_margin",
            nonnegative=False,
        ),
        parameter=parameter,
        state_sha256=state_digest,
        source=source,
        source_index=index,
    )


def load_layer_candidate_pickles(
    paths: Sequence[Path],
    *,
    expected_split: str,
    expected_file_sha256: Mapping[str, str],
    expected_layers: Sequence[int] | None = None,
) -> CandidateCorpus:
    """Load and validate trusted internal per-layer pickle artifacts.

    Pickle is intentionally limited to same-project trusted artifacts.  File
    hashes are computed before deserialization and may be externally pinned.
    """

    split = str(expected_split)
    if split not in {"calibration", "evaluation"}:
        raise ArtifactValidationError("expected_split must be calibration or evaluation")
    if not isinstance(expected_file_sha256, Mapping) or not expected_file_sha256:
        raise ArtifactValidationError("candidate pickle hashes must be externally pinned")
    supplied = tuple(Path(path) for path in paths)
    if not supplied or len({path.resolve() for path in supplied}) != len(supplied):
        raise ArtifactValidationError("candidate pickle paths must be nonempty and unique")
    expected = None if expected_layers is None else tuple(sorted(map(int, expected_layers)))
    records: list[CandidateRecord] = []
    sources: list[CandidateSource] = []
    observed_layers: list[int] = []
    for path in sorted(supplied, key=lambda item: str(item)):
        match = re.search(r"layer_(\d+)", str(path.parent))
        if match is None:
            raise ArtifactValidationError(f"candidate path does not identify a layer: {path}")
        layer = int(match.group(1))
        digest = _sha256_file(path)
        logical_path = f"{split}/layer_{layer:02d}/{path.name}"
        pinned = expected_file_sha256.get(str(path))
        if pinned is None:
            pinned = expected_file_sha256.get(logical_path)
        if pinned is None or digest != _sha256(pinned, f"expected hash for {path}"):
            raise ArtifactValidationError(f"candidate pickle hash mismatch: {path}")
        source = CandidateSource(
            path=logical_path,
            sha256=digest,
            byte_count=path.stat().st_size,
            layer=layer,
            split=split,
        )
        with path.open("rb") as handle:
            payload = pickle.load(handle)  # noqa: S301 - trusted, hash-pinned artifacts
            if handle.read(1):
                raise ArtifactValidationError(f"candidate pickle has trailing bytes: {path}")
        if not isinstance(payload, list) or not payload:
            raise ArtifactValidationError(f"candidate pickle must contain a nonempty list: {path}")
        sources.append(source)
        observed_layers.append(layer)
        records.extend(
            _candidate_record(raw, source=source, index=index)
            for index, raw in enumerate(payload)
        )
    if len(set(observed_layers)) != len(observed_layers):
        raise ArtifactValidationError("more than one candidate pickle was supplied per layer")
    if expected is not None and tuple(sorted(observed_layers)) != expected:
        raise ArtifactValidationError(
            f"candidate layers differ; expected={expected}, observed={tuple(sorted(observed_layers))}"
        )
    return CandidateCorpus(tuple(records), tuple(sources), split)


@dataclass(frozen=True)
class _TrafficPair:
    low_rate: int
    high_rate: int
    low_cap: int
    high_cap: int

    @property
    def pair_id(self) -> str:
        return f"{self.low_rate}_to_{self.high_rate}"


@dataclass(frozen=True)
class _ConfigContract:
    run_id: str
    layers: tuple[int, ...]
    pairs: tuple[_TrafficPair, ...]
    windows: tuple[int, ...]
    etas: tuple[float, ...]
    calibration_requests: int
    evaluation_requests: int
    authenticated_capture_path: str
    authenticated_capture_sha256: str
    config_canonical_sha256: str

    @property
    def rates(self) -> tuple[int, ...]:
        return tuple(sorted(
            {rate for pair in self.pairs for rate in (pair.low_rate, pair.high_rate)}
        ))


def _config_contract(config: Mapping[str, Any]) -> _ConfigContract:
    if not isinstance(config, Mapping):
        raise ArtifactValidationError("nested experiment config must be an object")
    if (
        config.get("schema") != "pr13_d1_nested_safe_allocation_config_v1"
        or config.get("experiment_stage") != "A"
        or config.get("d1_definition") != "same_decode_token_next_layer_router_only"
        or config.get("d2_d4_objectives_models_or_selector_inputs_present") is not False
        or config.get("downstream_routes_and_terminal_kl_role")
        != "sealed_outcome_phase_only"
        or config.get("experiment_b_started") is not False
        or config.get("runtime_predictor_in_scope") is not False
        or config.get("generated_rollout_in_scope") is not False
        or config.get("joint_all_layer_compression_in_scope") is not False
    ):
        raise ArtifactValidationError("config changed the frozen D1-only Experiment A boundary")
    if tuple(config.get("arms", ())) != FIVE_ARMS:
        raise ArtifactValidationError("config changed the frozen five-arm ordering")
    calibration = config.get("calibration")
    if not isinstance(calibration, Mapping) or (
        calibration.get("terminal_kl_used") is not False
        or calibration.get("downstream_routes_used") is not False
        or calibration.get("selection_per_rate") is not False
        or calibration.get("selection_per_traffic_pair") is not True
        or calibration.get("metadata_matched_low_endpoint_selects_shared_pair_parameters")
        is not True
        or calibration.get("parameters_frozen_before_evaluation_allocation") is not True
        or tuple(calibration.get("selection_order", ())) != SELECTION_ORDER
    ):
        raise ArtifactValidationError("config changed the frozen calibration firewall")
    capture_protocol = config.get("authenticated_capture_protocol")
    if (
        not isinstance(capture_protocol, Mapping)
        or set(capture_protocol) != {"path", "sha256"}
        or capture_protocol.get("path") != AUTHENTICATED_CAPTURE_CONFIG_PATH
        or capture_protocol.get("sha256") != AUTHENTICATED_CAPTURE_CONFIG_SHA256
    ):
        raise ArtifactValidationError(
            "allocation config changed its authenticated capture protocol"
        )
    layers_raw = config.get("injection_layers")
    if not isinstance(layers_raw, list) or not layers_raw:
        raise ArtifactValidationError("config injection_layers must be nonempty")
    layers = tuple(map(int, layers_raw))
    if len(set(layers)) != len(layers) or min(layers) < 0 or max(layers) >= 39:
        raise ArtifactValidationError("config injection layers are invalid or repeated")
    core = config.get("common_core")
    guard = config.get("local_guardrail")
    d1_screen = config.get("d1_screen")
    if (
        not isinstance(core, Mapping)
        or not isinstance(guard, Mapping)
        or not isinstance(d1_screen, Mapping)
    ):
        raise ArtifactValidationError(
            "config lacks common-core, local-guard, or D1-screen grids"
        )
    if (
        d1_screen.get("raw_slice_router_anchor_max_abs") != 0.0625
        or d1_screen.get("captured_full_model_baseline_logit_anchor_applied")
        is not True
        or d1_screen.get("zero_delta_after_anchor_must_equal_captured_logits")
        is not True
        or d1_screen.get("candidate_page_shortlist") != 32
        or d1_screen.get("maximum_exact_finalists") != 8
    ):
        raise ArtifactValidationError(
            "config changed the authenticated D1 router-anchor/search contract"
        )
    decode = config.get("decode_position")
    allocation_phase = config.get("allocation_phase")
    exact_gate = config.get("exact_d1_gate")
    if (
        not isinstance(decode, Mapping)
        or decode.get("exact_prefill") is not True
        or decode.get("query_length") != 1
        or decode.get("positions_per_request") != 1
        or not isinstance(allocation_phase, Mapping)
        or allocation_phase.get("terminal_logits_captured") is not False
        or allocation_phase.get("downstream_layers_after_d1_executed") is not False
        or allocation_phase.get("pair_fallback_flag_serialized") is not True
        or allocation_phase.get(
            "high_guardrail_qualified_checkpoint_count_serialized"
        ) is not True
        or allocation_phase.get(
            "high_guardrail_repair_attempt_count_serialized"
        ) is not True
        or allocation_phase.get(
            "pair_fallback_frequency_reported_once_per_policy_pair"
        ) is not True
        or not isinstance(exact_gate, Mapping)
        or exact_gate.get("all_256_router_logits_checked") is not True
        or exact_gate.get("safe_incumbent_is_immutable") is not True
        or exact_gate.get("unsafe_replacement_requires_strictly_fewer_membership_changes")
        is not True
        or exact_gate.get("high_completion_path_scan")
        != "complete_add_only_local_path_highest_pages_then_lowest_local"
        or exact_gate.get("high_checkpoint_admission")
        != "strict_high_endpoint_local_guard_and_exact_d1_nonworsening_against_selected_low"
        or exact_gate.get("same_rate_high_incumbent_semantics")
        != "safe_arm3_high_byte_identical_else_changed_high_requires_strict_crossing_reduction"
        or exact_gate.get("high_guardrail_infeasible_action")
        != "revert_entire_low_high_policy_pair_byte_for_byte_to_arm3"
        or exact_gate.get("pair_fallback_preserves_fixed_grid") is not True
        or core.get("d1_blind") is not True
        or guard.get("form")
        != "J_candidate_le_J_nested_local_endpoint_plus_eta_times_J_all_Q2"
    ):
        raise ArtifactValidationError("config changed exact-prefill D1 allocation semantics")
    windows_raw = core.get("repair_window_pages_per_group_grid")
    etas_raw = guard.get("eta_grid")
    if not isinstance(windows_raw, list) or not windows_raw:
        raise ArtifactValidationError("repair-window grid must be nonempty")
    if not isinstance(etas_raw, list) or not etas_raw:
        raise ArtifactValidationError("eta grid must be nonempty")
    windows = tuple(_integer(value, "repair window") for value in windows_raw)
    etas = tuple(_finite(value, "eta") for value in etas_raw)
    if len(set(windows)) != len(windows) or len(set(etas)) != len(etas):
        raise ArtifactValidationError("repair-window and eta grids must not repeat")
    pair_rows = config.get("traffic_pairs")
    if not isinstance(pair_rows, list) or not pair_rows:
        raise ArtifactValidationError("traffic_pairs must be nonempty")
    pairs: list[_TrafficPair] = []
    seen_rates: set[int] = set()
    for index, raw in enumerate(pair_rows):
        if not isinstance(raw, Mapping):
            raise ArtifactValidationError(f"traffic_pairs[{index}] must be an object")
        low = _integer(
            raw.get("metadata_matched_pages_per_expert"),
            f"traffic_pairs[{index}].metadata_matched_pages_per_expert", 1,
        )
        high = _integer(
            raw.get("pr13_reference_pages_per_expert"),
            f"traffic_pairs[{index}].pr13_reference_pages_per_expert", 1,
        )
        low_cap = _integer(
            raw.get("metadata_matched_group_cap"),
            f"traffic_pairs[{index}].metadata_matched_group_cap", 1,
        )
        high_cap = _integer(
            raw.get("pr13_reference_group_cap"),
            f"traffic_pairs[{index}].pr13_reference_group_cap", 1,
        )
        if low >= high or low_cap != 8 * low or high_cap != 8 * high:
            raise ArtifactValidationError("traffic pair rates or pooled caps changed")
        if {low, high} & seen_rates:
            raise ArtifactValidationError("traffic-pair rates overlap")
        seen_rates.update((low, high))
        pairs.append(_TrafficPair(low, high, low_cap, high_cap))
    request_source = config.get("request_source")
    if not isinstance(request_source, Mapping):
        raise ArtifactValidationError("config request_source is absent")
    calibration_count = _integer(
        request_source.get("calibration_requests"), "calibration request count", 1,
    )
    evaluation_count = _integer(
        request_source.get("evaluation_requests"), "evaluation request count", 1,
    )
    return _ConfigContract(
        run_id=_text(config.get("run_id"), "config run_id"),
        layers=layers,
        pairs=tuple(pairs),
        windows=windows,
        etas=etas,
        calibration_requests=calibration_count,
        evaluation_requests=evaluation_count,
        authenticated_capture_path=AUTHENTICATED_CAPTURE_CONFIG_PATH,
        authenticated_capture_sha256=AUTHENTICATED_CAPTURE_CONFIG_SHA256,
        config_canonical_sha256=canonical_sha256(config),
    )


def _parameter_window(record: CandidateRecord, *, required: bool) -> int | None:
    values = [
        record.parameter[key]
        for key in ("repair_window", "repair_window_pages")
        if key in record.parameter
    ]
    if not values:
        if required:
            raise ArtifactValidationError(
                f"{record.identity} lacks its repair-window parameter"
            )
        return None
    parsed = tuple(_integer(value, "candidate repair window") for value in values)
    if len(set(parsed)) != 1:
        raise ArtifactValidationError(f"{record.identity} has inconsistent window aliases")
    return parsed[0]


def _parameter_eta(record: CandidateRecord, *, required: bool) -> float | None:
    if "eta" not in record.parameter:
        if required:
            raise ArtifactValidationError(f"{record.identity} lacks its eta parameter")
        return None
    return _finite(record.parameter["eta"], "candidate eta")


def _expected_endpoint(record: CandidateRecord, pair: _TrafficPair) -> None:
    if "endpoint" not in record.parameter:
        return
    expected = "low" if record.rate == pair.low_rate else "high"
    if record.parameter["endpoint"] != expected:
        raise ArtifactValidationError(f"{record.identity} changed its pair endpoint label")


def _request_cohort(
    corpus: CandidateCorpus,
    contract: _ConfigContract,
    *,
    expected_count: int,
    expected_request_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    observed_layers = {record.layer for record in corpus.records}
    if observed_layers != set(contract.layers):
        raise ArtifactValidationError(
            f"candidate layer set changed: {sorted(observed_layers)}"
        )
    request_ids = tuple(sorted({record.request_id for record in corpus.records}))
    if expected_request_ids is not None:
        expected = tuple(sorted(map(str, expected_request_ids)))
        if len(set(expected)) != len(expected):
            raise ArtifactValidationError("expected request IDs repeat")
        if request_ids != expected:
            raise ArtifactValidationError("candidate request cohort changed")
    if len(request_ids) != expected_count:
        raise ArtifactValidationError(
            f"candidate request count changed: expected {expected_count}, got {len(request_ids)}"
        )
    metadata: dict[str, tuple[str, str, int]] = {}
    for record in corpus.records:
        current = (record.prompt_sha256, record.domain, record.position)
        prior = metadata.setdefault(record.request_id, current)
        if prior != current:
            raise ArtifactValidationError(
                f"request {record.request_id!r} changed prompt/domain/position metadata"
            )
    return request_ids


def _source_rows(sources: Iterable[CandidateSource]) -> list[dict[str, Any]]:
    return [
        source.to_dict()
        for source in sorted(sources, key=lambda item: (item.layer, item.path))
    ]


def _index_unique(
    records: Iterable[CandidateRecord],
    key: Any,
    *,
    label: str,
) -> dict[Any, CandidateRecord]:
    result: dict[Any, CandidateRecord] = {}
    for record in records:
        identity = key(record)
        if identity in result:
            raise ArtifactValidationError(f"duplicate {label}: {identity}")
        result[identity] = record
    return result


def select_and_freeze_calibration(
    corpus: CandidateCorpus,
    config: Mapping[str, Any],
    *,
    expected_request_ids: Sequence[str],
    config_file_sha256: str,
    request_manifest_sha256: str,
    request_manifest_facts_sha256: str,
    local_tolerance: float = 1e-12,
) -> CalibrationSelection:
    """Select pair-level W/eta from Arm 4 calibration D1/local evidence only."""

    if not isinstance(corpus, CandidateCorpus) or corpus.split != "calibration":
        raise ArtifactValidationError("calibration selection requires calibration pickles")
    contract = _config_contract(config)
    request_ids = _request_cohort(
        corpus, contract,
        expected_count=contract.calibration_requests,
        expected_request_ids=expected_request_ids,
    )
    tolerance = _finite(local_tolerance, "local tolerance")
    expected_cells = {
        (layer, request_id)
        for layer in contract.layers
        for request_id in request_ids
    }
    evidence_rows: list[dict[str, Any]] = []
    selected_pairs: list[dict[str, Any]] = []
    for pair in contract.pairs:
        arm3_records = [
            record for record in corpus.records
            if record.arm == ARM_LOCAL and record.rate == pair.low_rate
        ]
        arm4_records = [
            record for record in corpus.records
            if record.arm == ARM_D1 and record.rate == pair.low_rate
        ]
        arm3 = _index_unique(
            arm3_records,
            lambda record: (_parameter_window(record, required=True), record.cell),
            label=f"Arm3 calibration cell at rate {pair.low_rate}",
        )
        arm4 = _index_unique(
            arm4_records,
            lambda record: (
                _parameter_window(record, required=True),
                _parameter_eta(record, required=True),
                record.cell,
            ),
            label=f"Arm4 calibration cell at rate {pair.low_rate}",
        )
        observed_arm3_grid = {key[0] for key in arm3}
        observed_arm4_grid = {(key[0], key[1]) for key in arm4}
        if observed_arm3_grid != set(contract.windows):
            raise ArtifactValidationError(
                f"Arm3 repair-window grid changed at rate {pair.low_rate}"
            )
        if observed_arm4_grid != {
            (window, eta) for window in contract.windows for eta in contract.etas
        }:
            raise ArtifactValidationError(
                f"Arm4 W/eta grid changed at rate {pair.low_rate}"
            )
        for record in (*arm3_records, *arm4_records):
            _expected_endpoint(record, pair)
        for window in contract.windows:
            if {key[1] for key in arm3 if key[0] == window} != expected_cells:
                raise ArtifactValidationError(
                    f"Arm3 calibration coverage changed for rate={pair.low_rate}, W={window}"
                )
            for eta in contract.etas:
                if {
                    key[2] for key in arm4 if key[:2] == (window, eta)
                } != expected_cells:
                    raise ArtifactValidationError(
                        f"Arm4 calibration coverage changed for rate={pair.low_rate}, "
                        f"W={window}, eta={eta}"
                    )
                safe_to_unsafe = 0
                unresolved = 0
                local_values: list[float] = []
                safe_state_changes = 0
                for cell in sorted(expected_cells):
                    local = arm3[(window, cell)]
                    candidate = arm4[(window, eta, cell)]
                    if (
                        local.prompt_sha256 != candidate.prompt_sha256
                        or local.expert_ids != candidate.expert_ids
                        or local.domain != candidate.domain
                        or local.position != candidate.position
                        or local.selector_weights != candidate.selector_weights
                        or local.execution_weights != candidate.execution_weights
                        or not np.array_equal(local.common_core, candidate.common_core)
                    ):
                        raise ArtifactValidationError(
                            f"Arm3/Arm4 calibration identity or common core changed at {cell}"
                        )
                    if candidate.all_q2_damage != local.all_q2_damage:
                        raise ArtifactValidationError(
                            f"Arm3/Arm4 J_Q2 changed at calibration cell {cell}"
                        )
                    local_limit = local.local_damage + float(eta) * local.all_q2_damage
                    if candidate.local_damage > local_limit + tolerance:
                        raise ArtifactValidationError(
                            f"Arm4 local guardrail violation at rate={pair.low_rate}, "
                            f"W={window}, eta={eta}, cell={cell}"
                        )
                    if local.d1_crossings == 0:
                        safe_to_unsafe += int(candidate.d1_crossings > 0)
                        safe_state_changes += int(
                            not np.array_equal(local.selected_state, candidate.selected_state)
                        )
                    changed = not np.array_equal(
                        local.selected_state, candidate.selected_state,
                    )
                    if local.d1_crossings == 0 and changed:
                        raise ArtifactValidationError(
                            f"Arm4 changed a D1-safe Arm3 calibration incumbent at {cell}"
                        )
                    if (
                        local.d1_crossings > 0
                        and changed
                        and candidate.d1_crossings >= local.d1_crossings
                    ):
                        raise ArtifactValidationError(
                            f"Arm4 calibration repair lacks a strict crossing reduction at {cell}"
                        )
                    if not changed and candidate.d1_crossings != local.d1_crossings:
                        raise ArtifactValidationError(
                            f"identical Arm3/Arm4 calibration states changed D1 labels at {cell}"
                        )
                    unresolved += candidate.d1_crossings
                    local_values.append(candidate.local_damage)
                parameter = {
                    "repair_window_pages": window,
                    "eta": float(eta),
                }
                evidence_rows.append({
                    "pair_id": pair.pair_id,
                    "metadata_matched_rate": pair.low_rate,
                    "reference_rate": pair.high_rate,
                    **parameter,
                    "safe_to_unsafe_events": safe_to_unsafe,
                    "safe_arm3_state_changes": safe_state_changes,
                    "unresolved_exact_d1_crossings": unresolved,
                    "mean_exact_combined_local_qenergy": float(np.mean(local_values)),
                    "cells": len(local_values),
                    "parameter_sha256": canonical_sha256(parameter),
                })
        pair_rows = [row for row in evidence_rows if row["pair_id"] == pair.pair_id]
        ordered = sorted(pair_rows, key=lambda row: (
            int(row["safe_to_unsafe_events"]),
            int(row["unresolved_exact_d1_crossings"]),
            int(row["repair_window_pages"]),
            float(row["eta"]),
            float(row["mean_exact_combined_local_qenergy"]),
            str(row["parameter_sha256"]),
        ))
        selected = ordered[0]
        if int(selected["safe_to_unsafe_events"]) != 0:
            raise ArtifactValidationError(
                f"no zero-safe-to-unsafe calibration choice exists for {pair.pair_id}"
            )
        selected_pairs.append({
            "pair_id": pair.pair_id,
            "metadata_matched_rate": pair.low_rate,
            "reference_rate": pair.high_rate,
            "repair_window_pages": int(selected["repair_window_pages"]),
            "eta": float(selected["eta"]),
            "selected_parameter_sha256": str(selected["parameter_sha256"]),
            "selection_source_arm": ARM_D1,
            "selection_source_endpoint": "metadata_matched_low",
            "applies_to_arms": [ARM_LOCAL, ARM_D1, ARM_D1_SEVERITY],
            "applies_to_rates": [pair.low_rate, pair.high_rate],
        })
    for row in evidence_rows:
        row["selected"] = any(
            row["pair_id"] == choice["pair_id"]
            and row["parameter_sha256"] == choice["selected_parameter_sha256"]
            for choice in selected_pairs
        )
    evidence_payload = {
        "schema": CALIBRATION_EVIDENCE_SCHEMA,
        "config_canonical_sha256": contract.config_canonical_sha256,
        "selection_order": list(SELECTION_ORDER),
        "selection_scope": "arm4_metadata_matched_low_endpoint_per_traffic_pair",
        "request_ids": list(request_ids),
        "layers": list(contract.layers),
        "candidate_sources": _source_rows(corpus.sources),
        "rows": sorted(
            evidence_rows,
            key=lambda row: (
                row["metadata_matched_rate"], row["repair_window_pages"], row["eta"],
            ),
        ),
    }
    evidence = {
        **evidence_payload,
        "sha256": canonical_sha256(evidence_payload),
    }
    file_digest = _sha256(config_file_sha256, "allocation config file sha256")
    request_digest = _sha256(request_manifest_sha256, "request manifest sha256")
    request_facts_digest = _sha256(
        request_manifest_facts_sha256, "request manifest facts sha256",
    )
    rates: dict[str, dict[str, Any]] = {}
    for pair in selected_pairs:
        for endpoint, rate in zip(
            ("metadata_matched_low", "paired_reference_high"),
            pair["applies_to_rates"], strict=True,
        ):
            rates[str(rate)] = {
                "pair_id": pair["pair_id"],
                "endpoint": endpoint,
                "selection_source_rate": pair["metadata_matched_rate"],
                "repair_window_pages": pair["repair_window_pages"],
                "eta": pair["eta"],
                "selected_parameter_sha256": pair["selected_parameter_sha256"],
            }
    spec = {
        "schema": CALIBRATION_SPEC_SCHEMA,
        "run_id": contract.run_id,
        "config_canonical_sha256": contract.config_canonical_sha256,
        "config_file_sha256": file_digest,
        "request_manifest_sha256": request_digest,
        "request_manifest_facts_sha256": request_facts_digest,
        "authenticated_capture_protocol": {
            "path": contract.authenticated_capture_path,
            "sha256": contract.authenticated_capture_sha256,
        },
        "clarification": (
            "traffic_pair_shared_w_eta_selected_on_arm4_metadata_matched_low_"
            "endpoint_to_preserve_literal_common_core_nesting"
        ),
        "selection_order": list(SELECTION_ORDER),
        "selection_scope": "traffic_pair_not_independent_endpoint",
        "arm5_inherits_arm4_parameters": True,
        "layers": list(contract.layers),
        "calibration_request_ids": list(request_ids),
        "calibration_request_ids_sha256": canonical_sha256(list(request_ids)),
        "calibration_prompt_sha256": sorted({
            record.prompt_sha256 for record in corpus.records
        }),
        "calibration_candidate_sources": _source_rows(corpus.sources),
        "calibration_evidence_sha256": evidence["sha256"],
        "traffic_pairs": selected_pairs,
        "rates": rates,
    }
    frozen = freeze_calibration_spec(spec)
    return CalibrationSelection(frozen_calibration=frozen, evidence=evidence)


def write_calibration_selection(
    selection: CalibrationSelection,
    *,
    frozen_path: Path,
    evidence_path: Path,
) -> dict[str, Any]:
    """Canonically write evidence first and its authenticated frozen spec last."""

    if not isinstance(selection, CalibrationSelection):
        raise TypeError("selection must be CalibrationSelection")
    frozen = selection.frozen_calibration
    frozen_digest = validate_frozen_calibration_spec(frozen)
    evidence = selection.evidence
    reject_outcome_leakage(evidence, path="$.calibration_evidence")
    if not isinstance(evidence, Mapping) or evidence.get("schema") != CALIBRATION_EVIDENCE_SCHEMA:
        raise ArtifactValidationError("calibration evidence schema changed")
    evidence_payload = {key: value for key, value in evidence.items() if key != "sha256"}
    evidence_digest = canonical_sha256(evidence_payload)
    if (
        evidence.get("sha256") != evidence_digest
        or frozen["spec"].get("calibration_evidence_sha256") != evidence_digest
    ):
        raise ArtifactValidationError("calibration evidence hash changed")
    frozen_path = Path(frozen_path)
    evidence_path = Path(evidence_path)
    if frozen_path.resolve() == evidence_path.resolve():
        raise ArtifactValidationError("frozen selection and evidence paths must differ")
    atomic_write_json(evidence_path, evidence)
    atomic_write_json(frozen_path, frozen)
    return {
        "frozen_calibration_spec_sha256": frozen_digest,
        "frozen_file_sha256": canonical_sha256(frozen),
        "calibration_evidence_sha256": evidence_digest,
        "frozen_file": frozen_path.name,
        "evidence_file": evidence_path.name,
    }


def validate_calibration_selection(
    frozen_calibration: Mapping[str, Any] | CalibrationSelection,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate a frozen selection and bind it to the supplied config."""

    frozen = (
        frozen_calibration.frozen_calibration
        if isinstance(frozen_calibration, CalibrationSelection)
        else frozen_calibration
    )
    validate_frozen_calibration_spec(frozen)
    spec = frozen["spec"]
    contract = _config_contract(config)
    if (
        spec.get("schema") != CALIBRATION_SPEC_SCHEMA
        or spec.get("run_id") != contract.run_id
        or spec.get("config_canonical_sha256") != contract.config_canonical_sha256
        or spec.get("authenticated_capture_protocol") != {
            "path": contract.authenticated_capture_path,
            "sha256": contract.authenticated_capture_sha256,
        }
        or tuple(spec.get("selection_order", ())) != SELECTION_ORDER
        or spec.get("selection_scope") != "traffic_pair_not_independent_endpoint"
        or spec.get("arm5_inherits_arm4_parameters") is not True
        or tuple(spec.get("layers", ())) != contract.layers
    ):
        raise ArtifactValidationError("frozen calibration selection changed its contract")
    calibration_request_ids = spec.get("calibration_request_ids")
    calibration_prompts = spec.get("calibration_prompt_sha256")
    if (
        not isinstance(calibration_request_ids, list)
        or any(not isinstance(value, str) or not value for value in calibration_request_ids)
        or calibration_request_ids != sorted(calibration_request_ids)
        or len(set(calibration_request_ids)) != contract.calibration_requests
        or spec.get("calibration_request_ids_sha256")
        != canonical_sha256(calibration_request_ids)
        or not isinstance(calibration_prompts, list)
        or any(not isinstance(value, str) or not HEX_64.fullmatch(value) for value in calibration_prompts)
        or calibration_prompts != sorted(calibration_prompts)
        or len(set(calibration_prompts)) != contract.calibration_requests
    ):
        raise ArtifactValidationError("frozen calibration request cohort changed")
    sources = spec.get("calibration_candidate_sources")
    if not isinstance(sources, list) or len(sources) != len(contract.layers):
        raise ArtifactValidationError("frozen calibration candidate sources changed")
    source_layers: set[int] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping) or set(source) != {
            "path", "sha256", "bytes", "layer", "split",
        }:
            raise ArtifactValidationError(
                f"frozen calibration source {index} fields changed"
            )
        if source.get("split") != "calibration":
            raise ArtifactValidationError("frozen candidate source crossed splits")
        _text(source.get("path"), f"calibration source {index} path")
        _sha256(source.get("sha256"), f"calibration source {index} sha256")
        _integer(source.get("bytes"), f"calibration source {index} bytes", 1)
        source_layers.add(_integer(
            source.get("layer"), f"calibration source {index} layer",
        ))
    if source_layers != set(contract.layers):
        raise ArtifactValidationError("frozen calibration source layers changed")
    _sha256(spec.get("calibration_evidence_sha256"), "calibration evidence sha256")
    _sha256(spec.get("config_file_sha256"), "allocation config file sha256")
    _sha256(spec.get("request_manifest_sha256"), "request manifest sha256")
    _sha256(
        spec.get("request_manifest_facts_sha256"), "request manifest facts sha256",
    )
    pair_rows = spec.get("traffic_pairs")
    rates = spec.get("rates")
    if not isinstance(pair_rows, list) or not isinstance(rates, Mapping):
        raise ArtifactValidationError("frozen calibration lacks traffic-pair parameters")
    indexed = {
        str(row.get("pair_id")): row
        for row in pair_rows
        if isinstance(row, Mapping)
    }
    if len(indexed) != len(pair_rows) or set(indexed) != {
        pair.pair_id for pair in contract.pairs
    }:
        raise ArtifactValidationError("frozen calibration traffic-pair identities changed")
    expected_rates: dict[str, tuple[_TrafficPair, str]] = {}
    for pair in contract.pairs:
        row = indexed[pair.pair_id]
        window = _integer(row.get("repair_window_pages"), "frozen repair window")
        eta = _finite(row.get("eta"), "frozen eta")
        parameter = {"repair_window_pages": window, "eta": eta}
        if (
            window not in contract.windows
            or eta not in contract.etas
            or row.get("metadata_matched_rate") != pair.low_rate
            or row.get("reference_rate") != pair.high_rate
            or row.get("selection_source_arm") != ARM_D1
            or row.get("selection_source_endpoint") != "metadata_matched_low"
            or row.get("applies_to_arms") != [ARM_LOCAL, ARM_D1, ARM_D1_SEVERITY]
            or row.get("applies_to_rates") != [pair.low_rate, pair.high_rate]
            or row.get("selected_parameter_sha256") != canonical_sha256(parameter)
        ):
            raise ArtifactValidationError(f"frozen parameters changed for {pair.pair_id}")
        expected_rates[str(pair.low_rate)] = (pair, "metadata_matched_low")
        expected_rates[str(pair.high_rate)] = (pair, "paired_reference_high")
    if set(rates) != set(expected_rates):
        raise ArtifactValidationError("frozen per-rate parameter map changed")
    for rate, (pair, endpoint) in expected_rates.items():
        entry = rates[rate]
        selected = indexed[pair.pair_id]
        if not isinstance(entry, Mapping) or any((
            entry.get("pair_id") != pair.pair_id,
            entry.get("endpoint") != endpoint,
            entry.get("selection_source_rate") != pair.low_rate,
            entry.get("repair_window_pages") != selected["repair_window_pages"],
            entry.get("eta") != selected["eta"],
            entry.get("selected_parameter_sha256")
            != selected["selected_parameter_sha256"],
        )):
            raise ArtifactValidationError(f"frozen rate {rate} drifted from its pair")
    return json.loads(canonical_json_bytes(frozen))


def _selected_rate_parameter(
    frozen: Mapping[str, Any], rate: int,
) -> tuple[int, float, str]:
    entry = frozen["spec"]["rates"].get(str(rate))
    if not isinstance(entry, Mapping):
        raise ArtifactValidationError(f"frozen calibration has no rate {rate}")
    return (
        _integer(entry.get("repair_window_pages"), "selected repair window"),
        _finite(entry.get("eta"), "selected eta"),
        _text(entry.get("pair_id"), "selected pair_id"),
    )


def _matches_frozen_parameter(
    record: CandidateRecord,
    frozen: Mapping[str, Any],
) -> bool:
    if record.arm in {ARM_PR13, ARM_STRICT}:
        return True
    window, eta, _ = _selected_rate_parameter(frozen, record.rate)
    if _parameter_window(record, required=True) != window:
        return False
    if record.arm == ARM_LOCAL:
        return True
    return _parameter_eta(record, required=True) == eta


def _same_current_token_identity(
    records: Sequence[CandidateRecord],
) -> None:
    first = records[0]
    for record in records[1:]:
        if (
            record.prompt_sha256 != first.prompt_sha256
            or record.domain != first.domain
            or record.position != first.position
            or record.expert_ids != first.expert_ids
            or record.selector_weights != first.selector_weights
            or record.execution_weights != first.execution_weights
        ):
            raise ArtifactValidationError(
                f"five-arm current-token identity changed at {(first.rate, first.cell)}"
            )


def _changed(left: CandidateRecord, right: CandidateRecord) -> bool:
    return not np.array_equal(left.selected_state, right.selected_state)


def _same_exact_d1(left: CandidateRecord, right: CandidateRecord) -> bool:
    return (
        left.d1_crossings == right.d1_crossings
        and left.d1_violation_depth == right.d1_violation_depth
        and left.d1_routing_mass_churn == right.d1_routing_mass_churn
        and left.d1_labeled_margin == right.d1_labeled_margin
    )


def _validate_selected_cell(
    by_arm: Mapping[str, CandidateRecord],
    *,
    eta: float,
    tolerance: float,
) -> None:
    records = [by_arm[arm] for arm in FIVE_ARMS]
    _same_current_token_identity(records)
    pr13 = by_arm[ARM_PR13]
    strict = by_arm[ARM_STRICT]
    local = by_arm[ARM_LOCAL]
    d1 = by_arm[ARM_D1]
    severity = by_arm[ARM_D1_SEVERITY]
    if pr13.d1_crossings == 0:
        if strict.d1_crossings != 0 or _changed(pr13, strict):
            raise ArtifactValidationError("strict repair changed a D1-safe PR13 incumbent")
    elif _changed(pr13, strict) and strict.d1_crossings >= pr13.d1_crossings:
        raise ArtifactValidationError("strict repair lacks a strict exact crossing reduction")
    if not _changed(pr13, strict) and not _same_exact_d1(pr13, strict):
        raise ArtifactValidationError("identical PR13/strict states changed exact D1 metrics")
    if not (
        np.array_equal(local.common_core, d1.common_core)
        and np.array_equal(local.common_core, severity.common_core)
    ):
        raise ArtifactValidationError("nested arms do not share the frozen common core")
    for candidate in (d1, severity):
        if candidate.all_q2_damage != local.all_q2_damage:
            raise ArtifactValidationError("nested policy changed its token J_Q2")
        limit = local.local_damage + eta * local.all_q2_damage
        if candidate.local_damage > limit + tolerance:
            raise ArtifactValidationError("selected nested policy violates its local guardrail")
        if local.d1_crossings == 0:
            if candidate.d1_crossings != 0 or _changed(local, candidate):
                raise ArtifactValidationError("nested policy changed a D1-safe Arm3 endpoint")
        elif _changed(local, candidate) and candidate.d1_crossings >= local.d1_crossings:
            raise ArtifactValidationError(
                "nested repair changed Arm3 without a strict exact crossing reduction"
            )
        if not _changed(local, candidate) and not _same_exact_d1(local, candidate):
            raise ArtifactValidationError("identical nested states changed exact D1 metrics")


def _allocation_record(
    record: CandidateRecord,
    *,
    frozen_digest: str,
    frozen: Mapping[str, Any],
) -> dict[str, Any]:
    cap = record.rate * 8
    parameter: dict[str, Any] = {"candidate": record.parameter}
    if record.arm in NESTED_ARMS:
        window, eta, pair_id = _selected_rate_parameter(frozen, record.rate)
        parameter["frozen"] = {
            "pair_id": pair_id,
            "repair_window_pages": window,
            "eta": eta,
        }
    return {
        "schema": ALLOCATION_RECORD_SCHEMA,
        "arm": record.arm,
        "rate": record.rate,
        "layer": record.layer,
        "request_id": record.request_id,
        "split": "evaluation",
        "calibration_spec_sha256": frozen_digest,
        "prompt_sha256": record.prompt_sha256,
        "domain": record.domain,
        "position": record.position,
        "expert_ids": list(record.expert_ids),
        "selector_weights": list(record.selector_weights),
        "execution_weights": list(record.execution_weights),
        "freeze_state": encode_state(record.common_core, page_cap=cap),
        "selected_state": encode_state(record.selected_state, page_cap=cap),
        "moves": [dict(move) for move in record.moves],
        "parameter": parameter,
        "local_damage": record.local_damage,
        "all_q2_damage": record.all_q2_damage,
        "legacy_additive_damage": record.legacy_additive_damage,
        "d1_crossings": record.d1_crossings,
        "d1_violation_depth": record.d1_violation_depth,
        "d1_routing_mass_churn": record.d1_routing_mass_churn,
        "d1_labeled_margin": record.d1_labeled_margin,
        "candidate_source": {
            "path": record.source.path,
            "sha256": record.source.sha256,
            "index": record.source_index,
            "locator_sha256": record.source_locator_sha256,
            "selected_state_sha256": record.state_sha256,
        },
    }


def _allocation_reference(record: CandidateRecord) -> dict[str, Any]:
    return {
        "arm": record.arm,
        "rate": record.rate,
        "layer": record.layer,
        "request_id": record.request_id,
    }


def build_evaluation_allocation_manifest(
    corpus: CandidateCorpus,
    config: Mapping[str, Any],
    frozen_calibration: Mapping[str, Any] | CalibrationSelection,
    *,
    expected_request_ids: Sequence[str],
    local_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Select the frozen five-arm evaluation grid and build a strict manifest."""

    if not isinstance(corpus, CandidateCorpus) or corpus.split != "evaluation":
        raise ArtifactValidationError("allocation sealing requires evaluation pickles")
    contract = _config_contract(config)
    frozen = validate_calibration_selection(frozen_calibration, config)
    frozen_digest = validate_frozen_calibration_spec(frozen)
    request_ids = _request_cohort(
        corpus, contract,
        expected_count=contract.evaluation_requests,
        expected_request_ids=expected_request_ids,
    )
    calibration_ids = set(frozen["spec"]["calibration_request_ids"])
    calibration_prompts = set(frozen["spec"]["calibration_prompt_sha256"])
    evaluation_prompts = {record.prompt_sha256 for record in corpus.records}
    if calibration_ids & set(request_ids) or calibration_prompts & evaluation_prompts:
        raise ArtifactValidationError(
            "calibration and evaluation request/prompt cohorts overlap"
        )
    tolerance = _finite(local_tolerance, "local tolerance")
    allowed_rates = set(contract.rates)
    if any(record.rate not in allowed_rates for record in corpus.records):
        raise ArtifactValidationError("evaluation candidates contain an undeclared rate")
    off_frozen = [
        record.identity
        for record in corpus.records
        if not _matches_frozen_parameter(record, frozen)
    ]
    if off_frozen:
        raise ArtifactValidationError(
            "evaluation allocation contains parameters not frozen by calibration"
        )
    selected = list(corpus.records)
    indexed = _index_unique(
        selected,
        lambda record: record.identity,
        label="frozen evaluation allocation identity",
    )
    expected_identities = {
        (arm, rate, layer, request_id)
        for arm in FIVE_ARMS
        for rate in contract.rates
        for layer in contract.layers
        for request_id in request_ids
    }
    if set(indexed) != expected_identities:
        missing = expected_identities - set(indexed)
        extra = set(indexed) - expected_identities
        raise ArtifactValidationError(
            f"frozen five-arm evaluation grid differs; missing={len(missing)}, "
            f"extra={len(extra)}"
        )
    for rate in contract.rates:
        _, eta, _ = _selected_rate_parameter(frozen, rate)
        for layer in contract.layers:
            for request_id in request_ids:
                by_arm = {
                    arm: indexed[(arm, rate, layer, request_id)]
                    for arm in FIVE_ARMS
                }
                _validate_selected_cell(by_arm, eta=eta, tolerance=tolerance)
    chains: list[dict[str, Any]] = []
    for pair in contract.pairs:
        for arm in NESTED_ARMS:
            for layer in contract.layers:
                for request_id in request_ids:
                    low = indexed[(arm, pair.low_rate, layer, request_id)]
                    high = indexed[(arm, pair.high_rate, layer, request_id)]
                    if (
                        low.expert_ids != high.expert_ids
                        or not np.array_equal(low.common_core, high.common_core)
                        or not physical_subset(low.selected_state, high.selected_state)
                    ):
                        raise ArtifactValidationError(
                            f"literal nested pair failed for {arm}, {pair.pair_id}, "
                            f"layer={layer}, request={request_id}"
                        )
                    chains.append({
                        "name": (
                            f"{arm}__{pair.pair_id}__layer_{layer:02d}__"
                            f"request_{request_id}"
                        ),
                        "members": [
                            _allocation_reference(low),
                            _allocation_reference(high),
                        ],
                    })
    ordered = sorted(
        indexed.values(),
        key=lambda record: (
            record.request_id, record.layer, record.rate, FIVE_ARMS.index(record.arm),
        ),
    )
    manifest = {
        "schema": ALLOCATION_MANIFEST_SCHEMA,
        "run_id": contract.run_id,
        "frozen_calibration": frozen,
        "allocations": [
            _allocation_record(
                record, frozen_digest=frozen_digest, frozen=frozen,
            )
            for record in ordered
        ],
        "nesting_chains": chains,
        "allocation_inputs": {
            "config_canonical_sha256": contract.config_canonical_sha256,
            "evaluation_candidate_sources": _source_rows(corpus.sources),
            "evaluation_request_ids": list(request_ids),
            "evaluation_request_ids_sha256": canonical_sha256(list(request_ids)),
            "traffic_pair_parameter_clarification": (
                "low_endpoint_arm4_selection_shared_with_high_endpoint_and_arm5"
            ),
            "five_arm_grid_complete": True,
        },
    }
    return validate_allocation_manifest(manifest)


def write_sealed_evaluation_allocations(
    corpus: CandidateCorpus,
    config: Mapping[str, Any],
    frozen_calibration: Mapping[str, Any] | CalibrationSelection,
    *,
    manifest_path: Path,
    seal_path: Path,
    expected_request_ids: Sequence[str],
    local_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Build, validate, atomically write, and checksum-seal evaluation states."""

    manifest = build_evaluation_allocation_manifest(
        corpus,
        config,
        frozen_calibration,
        expected_request_ids=expected_request_ids,
        local_tolerance=local_tolerance,
    )
    return write_sealed_allocation_manifest(
        manifest,
        manifest_path=Path(manifest_path),
        seal_path=Path(seal_path),
    )
