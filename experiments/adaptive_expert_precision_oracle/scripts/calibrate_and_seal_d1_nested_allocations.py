#!/usr/bin/env python3
"""Calibrate and checksum-seal nested D1 Experiment A allocations.

This is deliberately a narrow allocation-side CLI.  Every file that can alter
selection is byte-hash pinned before parsing, the request cohort is recovered
from the pinned selected-request manifest, and the six candidate pickles are
reconstructed from the immutable layer list rather than discovered by a glob.
Terminal and downstream outcome artifacts are neither accepted nor opened.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "src"))

from oracle_study.d1_nested_artifacts import (  # noqa: E402
    ArtifactValidationError,
    canonical_json_bytes,
    validate_frozen_calibration_spec,
)
from oracle_study.d1_nested_calibration import (  # noqa: E402
    CANDIDATE_SCHEMA,
    load_layer_candidate_pickles,
    select_and_freeze_calibration,
    write_calibration_selection,
    write_sealed_evaluation_allocations,
)


ALLOCATION_CONFIG_SCHEMA = "pr13_d1_nested_safe_allocation_config_v1"
REQUEST_MANIFEST_SCHEMA = "pr13_d1_nested_request_manifest_v1"
REQUEST_FACTS_SCHEMA = "pr13_d1_nested_request_manifest_facts_v1"
PICKLE_NAME = "nested_candidates.pkl"
HEX_64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ProtocolInputs:
    config: dict[str, Any]
    config_sha256: str
    request_manifest_sha256: str
    request_manifest_facts_sha256: str
    request_ids: tuple[str, ...]
    manifest_request_ids: tuple[str, ...]
    calibration_request_ids: tuple[str, ...]
    layers: tuple[int, ...]


@dataclass(frozen=True)
class CandidateInputs:
    paths: tuple[Path, ...]
    sha256_by_path: dict[str, str]
    run_facts_sha256: str
    code_identity: dict[str, Any] | None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ArtifactValidationError(f"{field} must be a SHA-256 string")
    result = value.lower()
    if not HEX_64.fullmatch(result):
        raise ArtifactValidationError(
            f"{field} must contain exactly 64 hexadecimal digits"
        )
    return result


def _verified_bytes(path: Path, expected_sha256: str, field: str) -> tuple[bytes, str]:
    path = Path(path)
    expected = _sha256(expected_sha256, f"{field} expected SHA-256")
    if not path.is_file():
        raise ArtifactValidationError(f"{field} not found: {path}")
    observed = _sha256_file(path)
    if observed != expected:
        raise ArtifactValidationError(
            f"{field} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return path.read_bytes(), observed


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactValidationError(f"JSON object repeats key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ArtifactValidationError(f"non-finite JSON constant is forbidden: {value}")


def _decode_json(payload: bytes, field: str) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except ArtifactValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"{field} is not strict UTF-8 JSON: {exc}") from exc


def _load_pinned_json(path: Path, expected_sha256: str, field: str) -> tuple[Any, str]:
    payload, digest = _verified_bytes(path, expected_sha256, field)
    return _decode_json(payload, field), digest


def _load_pinned_jsonl(
    path: Path, expected_sha256: str, field: str,
) -> tuple[list[Any], str, int]:
    payload, digest = _verified_bytes(path, expected_sha256, field)
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ArtifactValidationError(f"{field} is not UTF-8") from exc
    lines = text.splitlines(keepends=True)
    if not lines or any(not line.strip() for line in lines):
        raise ArtifactValidationError(f"{field} must be contiguous nonblank JSONL")
    if any(not line.endswith(("\n", "\r")) for line in lines):
        raise ArtifactValidationError(f"{field} must end every JSONL row with a newline")
    rows = [
        _decode_json(line.encode("utf-8"), f"{field} line {index}")
        for index, line in enumerate(lines, start=1)
    ]
    return rows, digest, len(payload)


def _integer(value: Any, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ArtifactValidationError(f"{field} must be an integer >= {minimum}")
    return value


def _finite(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"{field} must be a finite number") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ArtifactValidationError(f"{field} must be finite and nonnegative")
    return result


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ArtifactValidationError(f"{field} must be a nonempty trimmed string")
    return value


def _string_set_sha256(values: Sequence[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _config_layers(config: Mapping[str, Any]) -> tuple[int, ...]:
    if config.get("schema") != ALLOCATION_CONFIG_SCHEMA:
        raise ArtifactValidationError("allocation config schema changed")
    raw = config.get("injection_layers")
    if not isinstance(raw, list):
        raise ArtifactValidationError("allocation config injection_layers must be a list")
    layers = tuple(_integer(value, "injection layer") for value in raw)
    if len(layers) != 6 or len(set(layers)) != 6:
        raise ArtifactValidationError(
            "allocation sealing requires exactly six distinct configured layers"
        )
    return layers


def _split_counts(config: Mapping[str, Any]) -> dict[str, int]:
    source = config.get("request_source")
    if not isinstance(source, Mapping):
        raise ArtifactValidationError("allocation config request_source is absent")
    return {
        "calibration": _integer(
            source.get("calibration_requests"), "calibration request count", 1,
        ),
        "evaluation": _integer(
            source.get("evaluation_requests"), "evaluation request count", 1,
        ),
    }


def _config_grid(config: Mapping[str, Any]) -> tuple[list[int], list[float], list[int]]:
    core = config.get("common_core")
    guard = config.get("local_guardrail")
    pairs = config.get("traffic_pairs")
    if not isinstance(core, Mapping) or not isinstance(guard, Mapping):
        raise ArtifactValidationError("allocation config lacks search grids")
    if not isinstance(pairs, list) or not pairs:
        raise ArtifactValidationError("allocation config traffic_pairs is absent")
    windows_raw = core.get("repair_window_pages_per_group_grid")
    etas_raw = guard.get("eta_grid")
    if not isinstance(windows_raw, list) or not isinstance(etas_raw, list):
        raise ArtifactValidationError("allocation config repair-window/eta grids changed")
    windows = [_integer(value, "repair window") for value in windows_raw]
    etas = [_finite(value, "eta") for value in etas_raw]
    rates: list[int] = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, Mapping):
            raise ArtifactValidationError(f"traffic pair {index} is not an object")
        rates.extend((
            _integer(
                pair.get("metadata_matched_pages_per_expert"),
                f"traffic pair {index} low rate", 1,
            ),
            _integer(
                pair.get("pr13_reference_pages_per_expert"),
                f"traffic pair {index} high rate", 1,
            ),
        ))
    if len(set(windows)) != len(windows) or len(set(etas)) != len(etas):
        raise ArtifactValidationError("allocation search grids repeat values")
    if len(set(rates)) != len(rates):
        raise ArtifactValidationError("allocation traffic-pair rates overlap")
    return windows, etas, sorted(rates)


def _load_protocol_inputs(args: argparse.Namespace, split: str) -> ProtocolInputs:
    config_raw, config_digest = _load_pinned_json(
        args.config, args.config_sha256, "allocation config",
    )
    if not isinstance(config_raw, dict):
        raise ArtifactValidationError("allocation config must be a JSON object")
    layers = _config_layers(config_raw)
    counts = _split_counts(config_raw)

    rows, manifest_digest, manifest_bytes = _load_pinned_jsonl(
        args.request_manifest,
        args.request_manifest_sha256,
        "selected request manifest",
    )
    request_ids: list[str] = []
    prompt_hashes: list[str] = []
    split_ids: dict[str, list[str]] = {"calibration": [], "evaluation": []}
    split_prompts: dict[str, list[str]] = {"calibration": [], "evaluation": []}
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping) or row.get("schema") != REQUEST_MANIFEST_SCHEMA:
            raise ArtifactValidationError(
                f"selected request manifest row {index} has an invalid schema"
            )
        row_split = row.get("split")
        if row_split not in split_ids:
            raise ArtifactValidationError(
                f"selected request manifest row {index} has an invalid split"
            )
        request_id = _text(row.get("request_id"), f"manifest row {index} request_id")
        prompt_sha = _sha256(
            row.get("prompt_sha256"), f"manifest row {index} prompt_sha256",
        )
        _text(row.get("domain"), f"manifest row {index} domain")
        _integer(row.get("decode_position"), f"manifest row {index} decode_position")
        request_ids.append(request_id)
        prompt_hashes.append(prompt_sha)
        split_ids[row_split].append(request_id)
        split_prompts[row_split].append(prompt_sha)
    if len(set(request_ids)) != len(request_ids):
        raise ArtifactValidationError("selected request manifest repeats a request ID")
    if len(set(prompt_hashes)) != len(prompt_hashes):
        raise ArtifactValidationError("selected request manifest repeats a prompt hash")
    for name, expected in counts.items():
        if len(split_ids[name]) != expected:
            raise ArtifactValidationError(
                f"selected request manifest {name} count changed: "
                f"expected {expected}, got {len(split_ids[name])}"
            )

    facts, facts_digest = _load_pinned_json(
        args.request_manifest_facts,
        args.request_manifest_facts_sha256,
        "selected request manifest facts",
    )
    if not isinstance(facts, Mapping) or (
        facts.get("schema") != REQUEST_FACTS_SCHEMA or facts.get("completed") is not True
    ):
        raise ArtifactValidationError("selected request manifest facts schema changed")
    output = facts.get("output")
    fact_counts = facts.get("counts")
    identities = facts.get("split_identity_hashes")
    invariants = facts.get("invariants")
    if not all(isinstance(value, Mapping) for value in (
        output, fact_counts, identities, invariants,
    )):
        raise ArtifactValidationError("selected request manifest facts are incomplete")
    if (
        output.get("sha256") != manifest_digest
        or output.get("bytes") != manifest_bytes
        or output.get("rows") != len(rows)
        or fact_counts.get("selected_rows") != len(rows)
        or fact_counts.get("calibration_rows") != counts["calibration"]
        or fact_counts.get("evaluation_rows") != counts["evaluation"]
    ):
        raise ArtifactValidationError("request manifest facts do not bind its exact bytes")
    expected_identities = {
        "calibration_request_ids_sha256": _string_set_sha256(
            split_ids["calibration"]
        ),
        "calibration_prompt_sha256_set_sha256": _string_set_sha256(
            split_prompts["calibration"]
        ),
        "evaluation_request_ids_sha256": _string_set_sha256(
            split_ids["evaluation"]
        ),
        "evaluation_prompt_sha256_set_sha256": _string_set_sha256(
            split_prompts["evaluation"]
        ),
    }
    if any(identities.get(key) != value for key, value in expected_identities.items()):
        raise ArtifactValidationError("request manifest split identity hashes changed")
    required_invariants = {
        "exactly_one_decode_position_per_request": True,
        "calibration_evaluation_request_overlap": 0,
        "calibration_evaluation_prompt_overlap": 0,
        "duplicate_selected_request_ids": 0,
        "duplicate_selected_prompt_hashes": 0,
        "model_or_gpu_used": False,
        "terminal_quality_labels_read_or_used": False,
    }
    if any(invariants.get(key) != value for key, value in required_invariants.items()):
        raise ArtifactValidationError("request manifest facts invariants changed")
    selected = tuple(sorted(split_ids[split]))
    return ProtocolInputs(
        config=config_raw,
        config_sha256=config_digest,
        request_manifest_sha256=manifest_digest,
        request_manifest_facts_sha256=facts_digest,
        request_ids=selected,
        manifest_request_ids=tuple(split_ids[split]),
        calibration_request_ids=tuple(sorted(split_ids["calibration"])),
        layers=layers,
    )


def _validate_v2_candidate_run_provenance(
    facts: Mapping[str, Any],
    protocol: ProtocolInputs,
    split: str,
) -> tuple[dict[str, Any], str] | None:
    seal = protocol.config.get("reference_high_core_seal")
    if not isinstance(seal, Mapping) or seal.get("required") is not True:
        return None
    capture = protocol.config.get("authenticated_capture_artifacts")
    pr13 = protocol.config.get("authenticated_pr13_inputs")
    capture_protocol = protocol.config.get("authenticated_capture_protocol")
    resume = protocol.config.get("atomic_resume")
    if not all(isinstance(value, Mapping) for value in (
        capture, pr13, capture_protocol, resume,
    )):
        raise ArtifactValidationError("v2 candidate provenance config is incomplete")
    if (
        resume.get("schema") != "pr13_d1_nested_atomic_layer_resume_v1"
        or resume.get("facts_written_last") is not True
        or resume.get("completed_global_requires_exact_configured_layer_set")
        is not True
        or facts.get("atomic_resume_protocol")
        != "validated_facts_last_per_layer_exact_six_layer_merge_v1"
        or facts.get("run_id") != protocol.config.get("run_id")
        or facts.get("pilot_nonpromotable") is not False
        or facts.get("completed_for_sealing") is not True
    ):
        raise ArtifactValidationError("v2 candidate atomic resume contract changed")
    pins = facts.get("input_pins")
    if not isinstance(pins, Mapping):
        raise ArtifactValidationError("v2 candidate run lacks authenticated input pins")
    expected = {
        "schema": "pr13_d1_nested_allocation_input_pins_v1",
        "allocation_config_sha256": protocol.config_sha256,
        "capture_protocol_config_sha256": capture_protocol.get("sha256"),
        "request_manifest_sha256": protocol.request_manifest_sha256,
        "request_manifest_facts_sha256":
            protocol.request_manifest_facts_sha256,
        "capture_facts_sha256": capture.get("facts_sha256"),
        "checkpoint_config_sha256":
            protocol.config.get("checkpoint_config_sha256"),
        "checkpoint_index_sha256":
            protocol.config.get("checkpoint_index_sha256"),
        "tokenizer_json_sha256":
            protocol.config.get("tokenizer_json_sha256"),
        "selected_tree_sha256": protocol.config.get("selected_tree_sha256"),
        "pr13_config_sha256": pr13.get("config_sha256"),
        "factor_manifest_sha256": pr13.get("factor_manifest_sha256"),
    }
    for key, value in expected.items():
        if pins.get(key) != value:
            raise ArtifactValidationError(
                f"v2 candidate input pin changed: {key}"
            )
    factor_layers = pins.get("factor_layers")
    if (
        not isinstance(factor_layers, Mapping)
        or set(factor_layers) != {str(layer) for layer in protocol.layers}
    ):
        raise ArtifactValidationError("v2 candidate factor-layer pins changed")
    for layer, record in factor_layers.items():
        if (
            not isinstance(record, Mapping)
            or set(record) != {"factor", "sidecar"}
            or any(
                not isinstance(record.get(kind), Mapping)
                or set(record[kind]) != {"sha256", "bytes"}
                or not HEX_64.fullmatch(str(record[kind].get("sha256", "")))
                or _integer(
                    record[kind].get("bytes"),
                    f"factor layer {layer} {kind} bytes",
                    1,
                ) < 1
                for kind in ("factor", "sidecar")
            )
        ):
            raise ArtifactValidationError("v2 candidate factor file pin changed")
    code = pins.get("code_identity")
    if (
        not isinstance(code, Mapping)
        or code.get("schema") != "pr13_d1_nested_code_bundle_v1"
        or _integer(code.get("files"), "candidate code file count", 1) < 1
        or not HEX_64.fullmatch(str(code.get("canonical_sha256", "")))
    ):
        raise ArtifactValidationError("v2 candidate code identity changed")
    identity_sha = _sha256(
        facts.get("run_identity_sha256"), "candidate run identity SHA-256",
    )
    identity = facts.get("run_identity")
    if (
        not isinstance(identity, Mapping)
        or identity.get("schema")
        != "pr13_d1_nested_allocation_run_identity_v1"
        or identity.get("run_id") != protocol.config.get("run_id")
        or identity.get("split") != split
        or identity.get("request_ids") != list(protocol.manifest_request_ids)
        or identity.get("input_pins") != pins
        or hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
        != identity_sha
    ):
        raise ArtifactValidationError("v2 candidate run identity changed")
    return dict(pins), identity_sha


def _discover_candidate_inputs(
    args: argparse.Namespace,
    protocol: ProtocolInputs,
    split: str,
    *,
    expected_frozen_sha256: str | None = None,
) -> CandidateInputs:
    facts, facts_digest = _load_pinned_json(
        args.candidate_run_facts,
        args.candidate_run_facts_sha256,
        f"{split} candidate run facts",
    )
    if not isinstance(facts, Mapping) or (
        facts.get("completed") is not True
        or facts.get("schema") != CANDIDATE_SCHEMA
        or facts.get("split") != split
        or facts.get("test_rows_admitted_or_used") is not False
    ):
        raise ArtifactValidationError(f"{split} candidate run facts changed contract")
    v2_provenance = _validate_v2_candidate_run_provenance(facts, protocol, split)
    layer_facts = facts.get("layers")
    expected_keys = {str(layer) for layer in protocol.layers}
    if not isinstance(layer_facts, Mapping) or set(layer_facts) != expected_keys:
        raise ArtifactValidationError(
            f"{split} candidate run facts do not contain exactly six configured layers"
        )
    windows, etas, rates = _config_grid(protocol.config)
    root = Path(args.candidate_root)
    paths: list[Path] = []
    pins: dict[str, str] = {}
    for layer in protocol.layers:
        entry = layer_facts[str(layer)]
        if not isinstance(entry, Mapping) or (
            entry.get("completed") is not True
            or entry.get("schema") != CANDIDATE_SCHEMA
            or entry.get("split") != split
            or entry.get("layer") != layer
            or entry.get("allocation_uses_terminal_or_downstream_outcome") is not False
            or entry.get("candidate_execution_stops_at_d1_router") is not True
        ):
            raise ArtifactValidationError(
                f"candidate layer {layer} facts changed the D1-only contract"
            )
        if v2_provenance is not None:
            input_pins, run_identity_sha = v2_provenance
            if (
                entry.get("run_id") != protocol.config.get("run_id")
                or entry.get("run_identity_sha256") != run_identity_sha
                or entry.get("input_pins") != input_pins
                or entry.get("atomic_resume_protocol")
                != "facts_last_exact_three_file_inventory_v1"
                or entry.get("paired_reference_high_core_sealed") is not True
                or entry.get("pilot_nonpromotable") is not False
                or entry.get("completed_for_sealing") is not True
                or entry.get("partial_request_cohort") is not False
            ):
                raise ArtifactValidationError(
                    f"candidate layer {layer} v2 provenance changed"
                )
        entry_etas = entry.get("eta_grid")
        entry_rates = entry.get("rates")
        if not isinstance(entry_etas, list) or not isinstance(entry_rates, list):
            raise ArtifactValidationError(
                f"candidate layer {layer} search grids must be lists"
            )
        if (
            entry.get("requests") != len(protocol.request_ids)
            or entry.get("request_ids") != list(protocol.manifest_request_ids)
            or entry.get("repair_windows") != windows
            or [_finite(value, f"candidate layer {layer} eta") for value in entry_etas]
            != etas
            or sorted(
                _integer(value, f"candidate layer {layer} rate", 1)
                for value in entry_rates
            )
            != rates
            or entry.get("rows") != entry.get("expected_rows")
            or _integer(entry.get("rows"), f"candidate layer {layer} rows", 1) < 1
        ):
            raise ArtifactValidationError(
                f"candidate layer {layer} cohort or search grid changed"
            )
        frozen_facts = entry.get("frozen_calibration")
        if split == "calibration" and frozen_facts is not None:
            raise ArtifactValidationError("calibration candidate facts contain a frozen spec")
        if split == "evaluation":
            if not isinstance(frozen_facts, Mapping) or (
                frozen_facts.get("frozen_calibration_spec_sha256")
                != expected_frozen_sha256
            ):
                raise ArtifactValidationError(
                    f"evaluation candidate layer {layer} used another frozen spec"
                )
        files = entry.get("files")
        if not isinstance(files, Mapping) or not isinstance(
            files.get(PICKLE_NAME), Mapping,
        ):
            raise ArtifactValidationError(
                f"candidate layer {layer} facts lack {PICKLE_NAME} provenance"
            )
        if v2_provenance is not None and set(files) != {
            "nested_candidates.pkl",
            "nested_candidate_metrics.parquet",
            "nested_slice_parity.parquet",
        }:
            raise ArtifactValidationError(
                f"candidate layer {layer} scientific file inventory changed"
            )
        file_facts = files[PICKLE_NAME]
        pinned = _sha256(
            file_facts.get("sha256"), f"candidate layer {layer} pickle SHA-256",
        )
        path = root / split / f"layer_{layer:02d}" / PICKLE_NAME
        if not path.is_file() or path.stat().st_size != file_facts.get("bytes"):
            raise ArtifactValidationError(
                f"candidate layer {layer} pickle is missing or has changed size"
            )
        observed = _sha256_file(path)
        if observed != pinned:
            raise ArtifactValidationError(
                f"candidate layer {layer} pickle differs from pinned run facts"
            )
        paths.append(path)
        pins[str(path)] = pinned
    code_identity = (
        None if v2_provenance is None
        else dict(v2_provenance[0]["code_identity"])
    )
    return CandidateInputs(tuple(paths), pins, facts_digest, code_identity)


def _bind_frozen_spec(
    path: Path,
    expected_sha256: str,
    protocol: ProtocolInputs,
    calibration_request_ids: Sequence[str],
) -> tuple[dict[str, Any], str, str]:
    payload, file_digest = _verified_bytes(
        path, expected_sha256, "frozen calibration spec",
    )
    frozen = _decode_json(payload, "frozen calibration spec")
    if not isinstance(frozen, dict):
        raise ArtifactValidationError("frozen calibration spec must be an object")
    if canonical_json_bytes(frozen) != payload:
        raise ArtifactValidationError(
            "frozen calibration spec must use the canonical JSON byte representation"
        )
    frozen_digest = validate_frozen_calibration_spec(frozen)
    spec = frozen.get("spec")
    expected_calibration = tuple(sorted(map(str, calibration_request_ids)))
    if not isinstance(spec, Mapping) or any((
        spec.get("config_file_sha256") != protocol.config_sha256,
        spec.get("request_manifest_sha256") != protocol.request_manifest_sha256,
        spec.get("request_manifest_facts_sha256")
        != protocol.request_manifest_facts_sha256,
        tuple(spec.get("calibration_request_ids", ())) != expected_calibration,
    )):
        raise ArtifactValidationError(
            "frozen calibration spec is not bound to the pinned config/request cohort"
        )
    return frozen, frozen_digest, file_digest

def _require_matching_candidate_code_identity(
    frozen: Mapping[str, Any],
    evaluation_code_identity: Mapping[str, Any] | None,
) -> None:
    """Fail closed if calibration and evaluation used different code bundles."""

    spec = frozen.get("spec")
    calibration = (
        spec.get("calibration_candidate_code_identity")
        if isinstance(spec, Mapping)
        else None
    )
    if calibration is None and evaluation_code_identity is None:
        return
    for label, identity in (
        ("calibration", calibration),
        ("evaluation", evaluation_code_identity),
    ):
        if (
            not isinstance(identity, Mapping)
            or set(identity) != {"schema", "files", "canonical_sha256"}
            or identity.get("schema") != "pr13_d1_nested_code_bundle_v1"
            or _integer(identity.get("files"), f"{label} code file count", 1) < 1
            or not HEX_64.fullmatch(str(identity.get("canonical_sha256", "")))
        ):
            raise ArtifactValidationError(
                f"{label} candidate code identity is missing or malformed"
            )
    if dict(calibration) != dict(evaluation_code_identity):
        raise ArtifactValidationError(
            "evaluation candidate code identity differs from frozen calibration"

        )

def _reject_output_aliases(
    outputs: Sequence[Path], inputs: Sequence[Path],
) -> None:
    resolved_outputs = tuple(Path(path).resolve() for path in outputs)
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise ArtifactValidationError("output paths must be distinct")
    resolved_inputs = {Path(path).resolve() for path in inputs}
    if set(resolved_outputs) & resolved_inputs:
        raise ArtifactValidationError("an output path aliases an authenticated input")


def run_calibrate(args: argparse.Namespace) -> dict[str, Any]:
    protocol = _load_protocol_inputs(args, "calibration")
    candidates = _discover_candidate_inputs(args, protocol, "calibration")
    _reject_output_aliases(
        (args.frozen_output, args.evidence_output),
        (
            args.config,
            args.request_manifest,
            args.request_manifest_facts,
            args.candidate_run_facts,
            *candidates.paths,
        ),
    )
    corpus = load_layer_candidate_pickles(
        candidates.paths,
        expected_split="calibration",
        expected_file_sha256=candidates.sha256_by_path,
        expected_layers=protocol.layers,
    )
    selection = select_and_freeze_calibration(
        corpus,
        protocol.config,
        expected_request_ids=protocol.request_ids,
        config_file_sha256=protocol.config_sha256,
        request_manifest_sha256=protocol.request_manifest_sha256,
        request_manifest_facts_sha256=protocol.request_manifest_facts_sha256,
        calibration_candidate_code_identity=candidates.code_identity,
    )
    written = write_calibration_selection(
        selection,
        frozen_path=args.frozen_output,
        evidence_path=args.evidence_output,
    )
    return {
        "completed": True,
        "mode": "calibrate",
        "requests": len(protocol.request_ids),
        "layers": list(protocol.layers),
        "candidate_run_facts_sha256": candidates.run_facts_sha256,
        **written,
    }


def run_seal_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    protocol = _load_protocol_inputs(args, "evaluation")
    frozen, frozen_digest, frozen_file_digest = _bind_frozen_spec(
        args.frozen_spec,
        args.frozen_spec_sha256,
        protocol,
        protocol.calibration_request_ids,
    )
    candidates = _discover_candidate_inputs(
        args,
        protocol,
        "evaluation",
        expected_frozen_sha256=frozen_digest,
    )
    _require_matching_candidate_code_identity(
        frozen, candidates.code_identity,
    )
    _reject_output_aliases(
        (args.manifest_output, args.seal_output),
        (
            args.config,
            args.request_manifest,
            args.request_manifest_facts,
            args.candidate_run_facts,
            args.frozen_spec,
            *candidates.paths,
        ),
    )
    corpus = load_layer_candidate_pickles(
        candidates.paths,
        expected_split="evaluation",
        expected_file_sha256=candidates.sha256_by_path,
        expected_layers=protocol.layers,
    )
    seal = write_sealed_evaluation_allocations(
        corpus,
        protocol.config,
        frozen,
        manifest_path=args.manifest_output,
        seal_path=args.seal_output,
        expected_request_ids=protocol.request_ids,
    )
    return {
        "completed": True,
        "mode": "seal-evaluation",
        "requests": len(protocol.request_ids),
        "layers": list(protocol.layers),
        "frozen_calibration_file_sha256": frozen_file_digest,
        "frozen_calibration_spec_sha256": frozen_digest,
        "candidate_run_facts_sha256": candidates.run_facts_sha256,
        "seal": seal,
    }


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--request-manifest", required=True, type=Path)
    parser.add_argument("--request-manifest-sha256", required=True)
    parser.add_argument("--request-manifest-facts", required=True, type=Path)
    parser.add_argument("--request-manifest-facts-sha256", required=True)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--candidate-run-facts", required=True, type=Path)
    parser.add_argument("--candidate-run-facts-sha256", required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    calibrate = subparsers.add_parser(
        "calibrate", help="select W/eta on calibration requests and freeze them",
    )
    _add_common_arguments(calibrate)
    calibrate.add_argument("--frozen-output", required=True, type=Path)
    calibrate.add_argument("--evidence-output", required=True, type=Path)

    evaluation = subparsers.add_parser(
        "seal-evaluation", help="seal evaluation allocations using frozen W/eta",
    )
    _add_common_arguments(evaluation)
    evaluation.add_argument("--frozen-spec", required=True, type=Path)
    evaluation.add_argument("--frozen-spec-sha256", required=True)
    evaluation.add_argument("--manifest-output", required=True, type=Path)
    evaluation.add_argument("--seal-output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = (
            run_calibrate(args)
            if args.mode == "calibrate"
            else run_seal_evaluation(args)
        )
    except (ArtifactValidationError, FileNotFoundError, OSError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
