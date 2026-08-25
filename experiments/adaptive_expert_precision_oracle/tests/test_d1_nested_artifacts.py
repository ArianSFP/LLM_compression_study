from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "src"))

from oracle_study.d1_nested_artifacts import (
    ALLOCATION_MANIFEST_SCHEMA,
    ArtifactValidationError,
    allocation_manifest_sha256,
    canonical_json_bytes,
    canonical_sha256,
    decode_state,
    encode_state,
    freeze_calibration_spec,
    load_sealed_allocation_manifest,
    validate_allocation_manifest,
    write_sealed_allocation_manifest,
)


def _zero_state() -> list[list[int]]:
    return [[0 for _ in range(512)] for _ in range(8)]


def _with_bits(*moves: tuple[int, int, int]) -> list[list[int]]:
    state = _zero_state()
    for expert, unit, bit in moves:
        state[expert][unit] |= bit
    return state


def _moves(*moves: tuple[int, int, int]) -> list[dict[str, int | str]]:
    current: dict[tuple[int, int], int] = {}
    result: list[dict[str, int | str]] = []
    names = {1: "down", 2: "up", 4: "gate"}
    for step, (expert, unit, bit) in enumerate(moves):
        source = current.get((expert, unit), 0)
        destination = source | bit
        result.append(
            {
                "step": step,
                "expert": expert,
                "expert_id": expert + 10,
                "unit": unit,
                "bit": bit,
                "projection": names[bit],
                "source_state": source,
                "destination_state": destination,
                "page_delta": 1,
            }
        )
        current[(expert, unit)] = destination
    return result


def _allocation(
    *,
    frozen_digest: str,
    rate: int,
    split: str,
    selected: list[list[int]],
    moves: list[dict[str, int | str]],
    arm: str = "nested_d1_safe",
    layer: int = 0,
    request_id: str = "request-000",
) -> dict[str, object]:
    cap = rate * 8
    return {
        "arm": arm,
        "rate": rate,
        "layer": layer,
        "request_id": request_id,
        "split": split,
        "calibration_spec_sha256": frozen_digest,
        "expert_ids": list(range(10, 18)),
        "freeze_state": encode_state(_zero_state(), page_cap=cap),
        "selected_state": encode_state(selected, page_cap=cap),
        "moves": moves,
    }


def _reference(allocation: dict[str, object]) -> dict[str, object]:
    return {
        "arm": allocation["arm"],
        "rate": allocation["rate"],
        "layer": allocation["layer"],
        "request_id": allocation["request_id"],
    }


def _manifest() -> dict[str, object]:
    frozen = freeze_calibration_spec(
        {
            "rates": {
                "360": {"repair_window_pages": 16, "eta": 0.0005},
                "384": {"repair_window_pages": 16, "eta": 0.0005},
            },
            "selection_order": ["safe_to_unsafe", "crossings", "local_qenergy"],
        }
    )
    low = _allocation(
        frozen_digest=str(frozen["sha256"]),
        rate=360,
        split="evaluation",
        selected=_with_bits((0, 0, 1)),
        moves=_moves((0, 0, 1)),
    )
    high = _allocation(
        frozen_digest=str(frozen["sha256"]),
        rate=384,
        split="evaluation",
        selected=_with_bits((0, 0, 1), (3, 511, 4)),
        moves=_moves((0, 0, 1), (3, 511, 4)),
    )
    return {
        "schema": ALLOCATION_MANIFEST_SCHEMA,
        "run_id": "nested-a-test",
        "frozen_calibration": frozen,
        "allocations": [low, high],
        "nesting_chains": [
            {
                "name": "low_to_high",
                "members": [_reference(low), _reference(high)],
            }
        ],
        "allocation_inputs": {"checkpoint_sha256": "a" * 64},
    }


def test_canonical_serialization_ignores_mapping_insertion_order() -> None:
    first = {"z": 1, "a": {"y": [3, 2, 1], "x": True}}
    second = {"a": {"x": True, "y": [3, 2, 1]}, "z": 1}
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert canonical_sha256(first) == canonical_sha256(second)


def test_state_hex_is_lossless_and_popcount_is_sealed() -> None:
    state = _with_bits((0, 0, 1), (0, 0, 4), (7, 511, 2))
    encoded = encode_state(state, page_cap=3)
    assert encoded["page_count"] == 3
    assert encoded["expert_page_counts"] == [2, 0, 0, 0, 0, 0, 0, 1]
    assert [list(row) for row in decode_state(encoded)] == state

    count_tamper = deepcopy(encoded)
    count_tamper["page_count"] = 2
    with pytest.raises(ArtifactValidationError, match="popcount"):
        decode_state(count_tamper)

    digest_tamper = deepcopy(encoded)
    digest_tamper["data_hex"] = "00" + str(encoded["data_hex"])[2:]
    with pytest.raises(ArtifactValidationError, match="SHA-256"):
        decode_state(digest_tamper)

    with pytest.raises(ArtifactValidationError, match="exceeds declared cap"):
        encode_state(state, page_cap=2)


@pytest.mark.parametrize(
    "leakage",
    [
        {"terminal_logits": [1.0]},
        {"metrics": {"logit_kl": 0.1}},
        {"metrics": [{"delta_nll": 0.2}]},
        {"outcome": {"final_hidden_state": [0.0]}},
        {"audit": {"downstream_routes": [1, 2]}},
        {"audit": {"post_token_cache_mse": 0.0}},
        {"audit": {"downstream": {"routes": [1, 2]}}},
        {"audit": {"post_token_cache": {"outcomes": {"mse": 0.0}}}},
        {"terminal": {"logits": [1.0]}},
        {"final": {"hidden": [0.0]}},
    ],
)
def test_manifest_rejects_recursive_terminal_outcome_leakage(
    leakage: dict[str, object],
) -> None:
    manifest = _manifest()
    manifest["deep_provenance"] = {"nested": [leakage]}
    with pytest.raises(ArtifactValidationError, match="outcome leakage"):
        validate_allocation_manifest(manifest)


def test_frozen_calibration_spec_digest_is_immutable() -> None:
    manifest = _manifest()
    frozen = manifest["frozen_calibration"]
    assert isinstance(frozen, dict)
    spec = frozen["spec"]
    assert isinstance(spec, dict)
    rates = spec["rates"]
    assert isinstance(rates, dict)
    rate = rates["360"]
    assert isinstance(rate, dict)
    rate["eta"] = 0.001
    with pytest.raises(ArtifactValidationError, match="frozen calibration spec SHA-256"):
        validate_allocation_manifest(manifest)


def test_literal_nesting_chain_failure_is_rejected() -> None:
    manifest = _manifest()
    allocations = manifest["allocations"]
    assert isinstance(allocations, list)
    high = allocations[1]
    assert isinstance(high, dict)
    high["selected_state"] = encode_state(_with_bits((0, 0, 2)), page_cap=384 * 8)
    high["moves"] = _moves((0, 0, 2))
    with pytest.raises(ArtifactValidationError, match="literal bitwise nesting"):
        validate_allocation_manifest(manifest)


def test_nesting_chain_cannot_be_omitted_or_change_common_core() -> None:
    absent = _manifest()
    absent["nesting_chains"] = []
    with pytest.raises(ArtifactValidationError, match="non-empty"):
        validate_allocation_manifest(absent)

    changed_core = _manifest()
    allocations = changed_core["allocations"]
    assert isinstance(allocations, list)
    high = allocations[1]
    assert isinstance(high, dict)
    alternate_core = _with_bits((7, 510, 1))
    high["freeze_state"] = encode_state(alternate_core, page_cap=384 * 8)
    high["selected_state"] = encode_state(
        _with_bits((0, 0, 1), (3, 511, 4), (7, 510, 1)),
        page_cap=384 * 8,
    )
    high["moves"] = _moves((0, 0, 1), (3, 511, 4))
    with pytest.raises(ArtifactValidationError, match="frozen common core"):
        validate_allocation_manifest(changed_core)


@pytest.mark.parametrize(
    "bad_move",
    [
        {
            "step": 0,
            "expert": 0,
            "unit": 0,
            "bit": 3,
            "source_state": 0,
            "destination_state": 3,
        },
        {
            "step": 0,
            "expert": 0,
            "unit": 0,
            "bit": 1,
            "source_state": 1,
            "destination_state": 0,
        },
    ],
)
def test_move_chain_rejects_multibit_or_non_add_move(
    bad_move: dict[str, int],
) -> None:
    manifest = _manifest()
    allocations = manifest["allocations"]
    assert isinstance(allocations, list)
    low = allocations[0]
    assert isinstance(low, dict)
    low["moves"] = [bad_move]
    with pytest.raises(ArtifactValidationError, match="one bit|add-only|literal chain"):
        validate_allocation_manifest(manifest)


def test_allocation_identity_uniqueness_and_request_split_consistency() -> None:
    duplicate = _manifest()
    allocations = duplicate["allocations"]
    assert isinstance(allocations, list)
    allocations.append(deepcopy(allocations[0]))
    with pytest.raises(ArtifactValidationError, match="duplicate arm/rate/layer/request"):
        validate_allocation_manifest(duplicate)

    split_drift = _manifest()
    allocations = split_drift["allocations"]
    assert isinstance(allocations, list)
    high = allocations[1]
    assert isinstance(high, dict)
    high["split"] = "calibration"
    with pytest.raises(ArtifactValidationError, match="both calibration and evaluation"):
        validate_allocation_manifest(split_drift)


def test_seal_detects_manifest_tamper_before_tail(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest_path = tmp_path / "allocations.json"
    seal_path = tmp_path / "allocations.seal.json"
    seal = write_sealed_allocation_manifest(
        manifest,
        manifest_path=manifest_path,
        seal_path=seal_path,
    )
    loaded = load_sealed_allocation_manifest(
        manifest_path=manifest_path,
        seal_path=seal_path,
        expected_manifest_sha256=str(seal["allocation_manifest_sha256"]),
        expected_frozen_calibration_spec_sha256=str(
            seal["frozen_calibration_spec_sha256"]
        ),
    )
    assert allocation_manifest_sha256(loaded) == seal["allocation_manifest_sha256"]
    assert manifest_path.read_bytes() == canonical_json_bytes(loaded)

    decoded = json.loads(manifest_path.read_text())
    decoded["run_id"] = "tampered-after-allocation"
    manifest_path.write_bytes(canonical_json_bytes(decoded))
    with pytest.raises(ArtifactValidationError, match="sealed allocation manifest SHA-256"):
        load_sealed_allocation_manifest(
            manifest_path=manifest_path,
            seal_path=seal_path,
        )
