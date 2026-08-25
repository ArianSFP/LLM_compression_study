"""Pure-CPU sealing and validation for nested D1 allocation artifacts.

Allocation selection and downstream outcome execution are deliberately kept on
opposite sides of this module.  A tail runner should only consume allocations
through :func:`load_sealed_allocation_manifest`, which authenticates the raw
manifest bytes before parsing them and then revalidates every physical state,
move, nesting chain, split, and frozen calibration identity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


EXPERTS_PER_GROUP = 8
UNITS_PER_EXPERT = 512
STATE_BYTES = EXPERTS_PER_GROUP * UNITS_PER_EXPERT
VALID_STATE_MASK = 7
PROJECTION_BITS = {1: "down", 2: "up", 4: "gate"}
STATE_ENCODING = "uint8_hex_c_order_8x512_v1"

FROZEN_CALIBRATION_SCHEMA = "pr13_d1_frozen_calibration_spec_v1"
ALLOCATION_MANIFEST_SCHEMA = "pr13_d1_nested_allocation_manifest_v1"
ALLOCATION_SEAL_SCHEMA = "pr13_d1_nested_allocation_manifest_seal_v1"

SPLITS = {"calibration", "evaluation"}
IDENTITY_FIELDS = ("arm", "rate", "layer", "request_id")
STATE_FIELDS = {
    "encoding",
    "shape",
    "data_hex",
    "sha256",
    "expert_page_counts",
    "page_count",
    "page_cap",
}
MOVE_FIELDS = {
    "expert",
    "unit",
    "bit",
    "source_state",
    "destination_state",
}


__all__ = [
    "EXPERTS_PER_GROUP",
    "UNITS_PER_EXPERT",
    "STATE_BYTES",
    "VALID_STATE_MASK",
    "PROJECTION_BITS",
    "STATE_ENCODING",
    "FROZEN_CALIBRATION_SCHEMA",
    "ALLOCATION_MANIFEST_SCHEMA",
    "ALLOCATION_SEAL_SCHEMA",
    "ArtifactValidationError",
    "canonical_json_bytes",
    "canonical_sha256",
    "atomic_write_bytes",
    "atomic_write_json",
    "reject_outcome_leakage",
    "encode_state",
    "decode_state",
    "state_page_count",
    "physical_subset",
    "freeze_calibration_spec",
    "validate_frozen_calibration_spec",
    "validate_allocation_manifest",
    "allocation_manifest_sha256",
    "write_sealed_allocation_manifest",
    "load_sealed_allocation_manifest",
]


class ArtifactValidationError(ValueError):
    """Raised when an allocation artifact violates the sealed protocol."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the single canonical UTF-8 JSON representation used for hashes."""

    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"value is not canonical JSON: {exc}") from exc
    return text.encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _normalise_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ArtifactValidationError(f"{field} must be a SHA-256 string")
    result = value.lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ArtifactValidationError(f"{field} must contain 64 hexadecimal digits")
    return result


def _stage_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        return Path(handle.name)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    temporary: Path | None = None
    try:
        temporary = _stage_bytes(path, payload)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(Path(path), canonical_json_bytes(value))


def _normalise_key(key: str) -> str:
    with_word_boundaries = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    return re.sub(r"[^a-z0-9]+", "_", with_word_boundaries.lower()).strip("_")


def _leakage_reason(tokens: tuple[str, ...]) -> str | None:
    token_set = set(tokens)
    if any(token.startswith("nll") or token.startswith("kl") for token in tokens):
        return "terminal KL/NLL"
    if ({"terminal", "final"} & token_set) and any(
        token.startswith("logit") for token in tokens
    ):
        return "terminal logits"
    if "final" in token_set and any(token.startswith("hidden") for token in tokens):
        return "final hidden state"
    if "downstream" in token_set and any(
        token.startswith(("route", "router", "routing")) for token in tokens
    ):
        return "downstream routes"
    if any(token.startswith("cache") for token in tokens) and any(
        token in token_set
        for token in {
            "outcome",
            "outcomes",
            "error",
            "errors",
            "mse",
            "difference",
            "differences",
            "diff",
            "drift",
            "delta",
            "commit",
            "commits",
            "final",
            "post",
        }
    ):
        return "cache outcomes"
    return None


def reject_outcome_leakage(
    value: Any,
    *,
    path: str = "$",
    _branch_tokens: tuple[str, ...] = (),
) -> None:
    """Reject forbidden outcome-bearing keys at any recursive depth."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ArtifactValidationError(f"non-string JSON key at {path}")
            key_tokens = tuple(
                token for token in _normalise_key(key).split("_") if token
            )
            branch_tokens = _branch_tokens + key_tokens
            reason = _leakage_reason(branch_tokens)
            if reason is not None:
                raise ArtifactValidationError(
                    f"outcome leakage key at {path}.{key}: {reason} is sealed from allocation"
                )
            reject_outcome_leakage(
                child,
                path=f"{path}.{key}",
                _branch_tokens=branch_tokens,
            )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            reject_outcome_leakage(
                child,
                path=f"{path}[{index}]",
                _branch_tokens=_branch_tokens,
            )


def _integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactValidationError(f"{field} must be an integer")
    if value < minimum:
        raise ArtifactValidationError(f"{field} must be at least {minimum}")
    return value


def _nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ArtifactValidationError(
            f"{field} must be a non-empty string without edge whitespace"
        )
    return value


def _state_rows(value: Any) -> tuple[tuple[int, ...], ...]:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ArtifactValidationError("state must be a [8,512] integer sequence")
    if len(value) != EXPERTS_PER_GROUP:
        raise ArtifactValidationError("state must have exactly 8 expert rows")
    rows: list[tuple[int, ...]] = []
    for expert, row in enumerate(value):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise ArtifactValidationError(f"state expert row {expert} is not a sequence")
        if len(row) != UNITS_PER_EXPERT:
            raise ArtifactValidationError(
                f"state expert row {expert} must contain exactly 512 units"
            )
        result: list[int] = []
        for unit, item in enumerate(row):
            if isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= VALID_STATE_MASK:
                raise ArtifactValidationError(
                    f"state[{expert}][{unit}] must be a uint8 value in [0,7]"
                )
            result.append(item)
        rows.append(tuple(result))
    return tuple(rows)


def _raw_state_bytes(state: Any) -> bytes:
    rows = _state_rows(state)
    return bytes(item for row in rows for item in row)


def _state_counts(raw: bytes) -> tuple[list[int], int]:
    per_expert = [
        sum(value.bit_count() for value in raw[start : start + UNITS_PER_EXPERT])
        for start in range(0, STATE_BYTES, UNITS_PER_EXPERT)
    ]
    return per_expert, sum(per_expert)


def encode_state(state: Any, *, page_cap: int) -> dict[str, Any]:
    """Encode one complete physical state losslessly in deterministic hex."""

    cap = _integer(page_cap, field="page_cap")
    raw = _raw_state_bytes(state)
    expert_counts, total = _state_counts(raw)
    if total > cap:
        raise ArtifactValidationError(
            f"state page popcount {total} exceeds declared cap {cap}"
        )
    return {
        "encoding": STATE_ENCODING,
        "shape": [EXPERTS_PER_GROUP, UNITS_PER_EXPERT],
        "data_hex": raw.hex(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "expert_page_counts": expert_counts,
        "page_count": total,
        "page_cap": cap,
    }


def decode_state(value: Any) -> tuple[tuple[int, ...], ...]:
    """Validate and decode a lossless physical-state object."""

    if not isinstance(value, Mapping):
        raise ArtifactValidationError("encoded state must be a JSON object")
    if set(value) != STATE_FIELDS:
        missing = sorted(STATE_FIELDS - set(value))
        extra = sorted(set(value) - STATE_FIELDS)
        raise ArtifactValidationError(
            f"encoded state fields differ; missing={missing}, extra={extra}"
        )
    if value["encoding"] != STATE_ENCODING:
        raise ArtifactValidationError("unexpected physical-state encoding")
    if value["shape"] != [EXPERTS_PER_GROUP, UNITS_PER_EXPERT]:
        raise ArtifactValidationError("encoded physical state shape is not [8,512]")
    data_hex = value["data_hex"]
    if (
        not isinstance(data_hex, str)
        or len(data_hex) != STATE_BYTES * 2
        or any(character not in "0123456789abcdef" for character in data_hex)
    ):
        raise ArtifactValidationError("physical-state data_hex is not canonical lossless hex")
    raw = bytes.fromhex(data_hex)
    if any(item > VALID_STATE_MASK for item in raw):
        raise ArtifactValidationError("encoded physical state contains values outside [0,7]")
    digest = _normalise_sha256(value["sha256"], field="state sha256")
    if digest != hashlib.sha256(raw).hexdigest():
        raise ArtifactValidationError("physical-state SHA-256 mismatch")
    expert_counts, total = _state_counts(raw)
    recorded_expert_counts = value["expert_page_counts"]
    if (
        not isinstance(recorded_expert_counts, list)
        or len(recorded_expert_counts) != EXPERTS_PER_GROUP
        or any(isinstance(item, bool) or not isinstance(item, int) for item in recorded_expert_counts)
        or recorded_expert_counts != expert_counts
    ):
        raise ArtifactValidationError("expert page counts disagree with state popcount")
    recorded_total = _integer(value["page_count"], field="state page_count")
    cap = _integer(value["page_cap"], field="state page_cap")
    if recorded_total != total:
        raise ArtifactValidationError("group page count disagrees with state popcount")
    if total > cap:
        raise ArtifactValidationError(
            f"state page popcount {total} exceeds declared cap {cap}"
        )
    return tuple(
        tuple(raw[start : start + UNITS_PER_EXPERT])
        for start in range(0, STATE_BYTES, UNITS_PER_EXPERT)
    )


def state_page_count(state: Any) -> int:
    raw = _raw_state_bytes(state)
    return _state_counts(raw)[1]


def physical_subset(low: Any, high: Any) -> bool:
    low_raw = _raw_state_bytes(low)
    high_raw = _raw_state_bytes(high)
    return all((left & ((~right) & VALID_STATE_MASK)) == 0 for left, right in zip(low_raw, high_raw))


def freeze_calibration_spec(spec: Any) -> dict[str, Any]:
    """Return an immutable canonical copy and digest of calibration-only inputs."""

    if not isinstance(spec, Mapping):
        raise ArtifactValidationError("calibration spec must be a JSON object")
    reject_outcome_leakage(spec, path="$.frozen_calibration.spec")
    canonical_spec = json.loads(canonical_json_bytes(spec))
    return {
        "schema": FROZEN_CALIBRATION_SCHEMA,
        "spec": canonical_spec,
        "sha256": canonical_sha256(canonical_spec),
    }


def validate_frozen_calibration_spec(value: Any) -> str:
    if not isinstance(value, Mapping) or set(value) != {"schema", "spec", "sha256"}:
        raise ArtifactValidationError("invalid frozen calibration spec object")
    if value["schema"] != FROZEN_CALIBRATION_SCHEMA:
        raise ArtifactValidationError("unexpected frozen calibration spec schema")
    if not isinstance(value["spec"], Mapping):
        raise ArtifactValidationError("frozen calibration spec payload must be an object")
    reject_outcome_leakage(value["spec"], path="$.frozen_calibration.spec")
    recorded = _normalise_sha256(value["sha256"], field="frozen calibration spec sha256")
    observed = canonical_sha256(value["spec"])
    if recorded != observed:
        raise ArtifactValidationError("frozen calibration spec SHA-256 mismatch")
    return observed


def _expert_ids(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != EXPERTS_PER_GROUP:
        raise ArtifactValidationError("expert_ids must contain exactly eight IDs")
    result = tuple(_integer(item, field="expert_id") for item in value)
    if len(set(result)) != EXPERTS_PER_GROUP:
        raise ArtifactValidationError("expert_ids must be unique")
    return result


def _identity(value: Mapping[str, Any], *, path: str) -> tuple[str, int, int, str]:
    return (
        _nonempty_string(value.get("arm"), field=f"{path}.arm"),
        _integer(value.get("rate"), field=f"{path}.rate", minimum=1),
        _integer(value.get("layer"), field=f"{path}.layer"),
        _nonempty_string(value.get("request_id"), field=f"{path}.request_id"),
    )


def _validate_move_chain(
    allocation: Mapping[str, Any],
    *,
    expert_ids: tuple[int, ...],
    freeze_state: tuple[tuple[int, ...], ...],
    selected_state: tuple[tuple[int, ...], ...],
    path: str,
) -> None:
    moves = allocation.get("moves")
    if not isinstance(moves, list):
        raise ArtifactValidationError(f"{path}.moves must be a list")
    current = [list(row) for row in freeze_state]
    for index, move in enumerate(moves):
        move_path = f"{path}.moves[{index}]"
        if not isinstance(move, Mapping):
            raise ArtifactValidationError(f"{move_path} must be an object")
        missing = MOVE_FIELDS - set(move)
        if missing:
            raise ArtifactValidationError(f"{move_path} is missing fields {sorted(missing)}")
        if "step" in move and _integer(move["step"], field=f"{move_path}.step") != index:
            raise ArtifactValidationError(f"{move_path}.step is not sequential")
        expert = _integer(move["expert"], field=f"{move_path}.expert")
        unit = _integer(move["unit"], field=f"{move_path}.unit")
        bit = _integer(move["bit"], field=f"{move_path}.bit", minimum=1)
        source = _integer(move["source_state"], field=f"{move_path}.source_state")
        destination = _integer(
            move["destination_state"], field=f"{move_path}.destination_state"
        )
        if expert >= EXPERTS_PER_GROUP or unit >= UNITS_PER_EXPERT:
            raise ArtifactValidationError(f"{move_path} index lies outside [8,512]")
        if bit not in PROJECTION_BITS:
            raise ArtifactValidationError(f"{move_path} must change one bit in {{1,2,4}}")
        if source > VALID_STATE_MASK or destination > VALID_STATE_MASK:
            raise ArtifactValidationError(f"{move_path} state lies outside [0,7]")
        if current[expert][unit] != source:
            raise ArtifactValidationError(f"{move_path} source_state breaks the literal chain")
        if source & bit or destination != (source | bit) or (source ^ destination) != bit:
            raise ArtifactValidationError(
                f"{move_path} is not an add-only one-bit move after freeze"
            )
        if "expert_id" in move and _integer(
            move["expert_id"], field=f"{move_path}.expert_id"
        ) != expert_ids[expert]:
            raise ArtifactValidationError(f"{move_path}.expert_id is not row-aligned")
        if "page_delta" in move and _integer(
            move["page_delta"], field=f"{move_path}.page_delta", minimum=1
        ) != 1:
            raise ArtifactValidationError(f"{move_path}.page_delta must be exactly one")
        if "projection" in move and move["projection"] != PROJECTION_BITS[bit]:
            raise ArtifactValidationError(f"{move_path}.projection disagrees with bit")
        current[expert][unit] = destination
    if tuple(tuple(row) for row in current) != selected_state:
        raise ArtifactValidationError(f"{path}.moves do not reproduce selected_state exactly")
    if state_page_count(selected_state) != state_page_count(freeze_state) + len(moves):
        raise ArtifactValidationError(f"{path}.moves do not have exact one-page accounting")


def _validate_allocation(
    value: Any,
    *,
    frozen_digest: str,
    path: str,
) -> tuple[tuple[str, int, int, str], dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ArtifactValidationError(f"{path} must be an object")
    identity = _identity(value, path=path)
    arm, rate, layer, request_id = identity
    del arm, layer, request_id
    split = value.get("split")
    if split not in SPLITS:
        raise ArtifactValidationError(f"{path}.split must be calibration or evaluation")
    recorded_frozen = _normalise_sha256(
        value.get("calibration_spec_sha256"),
        field=f"{path}.calibration_spec_sha256",
    )
    if recorded_frozen != frozen_digest:
        raise ArtifactValidationError(f"{path} does not use the frozen calibration spec")
    expert_ids = _expert_ids(value.get("expert_ids"))
    freeze_state = decode_state(value.get("freeze_state"))
    selected_state = decode_state(value.get("selected_state"))
    expected_cap = rate * EXPERTS_PER_GROUP
    if value["freeze_state"]["page_cap"] != expected_cap:
        raise ArtifactValidationError(f"{path}.freeze_state cap is not rate*8")
    if value["selected_state"]["page_cap"] != expected_cap:
        raise ArtifactValidationError(f"{path}.selected_state cap is not rate*8")
    if not physical_subset(freeze_state, selected_state):
        raise ArtifactValidationError(f"{path}.freeze_state is not a literal subset")
    _validate_move_chain(
        value,
        expert_ids=expert_ids,
        freeze_state=freeze_state,
        selected_state=selected_state,
        path=path,
    )
    return identity, {
        "split": split,
        "expert_ids": expert_ids,
        "freeze_state": freeze_state,
        "selected_state": selected_state,
        "selected_page_cap": value["selected_state"]["page_cap"],
    }


def validate_allocation_manifest(value: Any) -> dict[str, Any]:
    """Validate a complete allocation-only manifest and return a canonical copy."""

    if not isinstance(value, Mapping):
        raise ArtifactValidationError("allocation manifest must be a JSON object")
    reject_outcome_leakage(value)
    required = {"schema", "frozen_calibration", "allocations", "nesting_chains"}
    missing = required - set(value)
    if missing:
        raise ArtifactValidationError(f"allocation manifest is missing {sorted(missing)}")
    if value["schema"] != ALLOCATION_MANIFEST_SCHEMA:
        raise ArtifactValidationError("unexpected allocation manifest schema")
    frozen_digest = validate_frozen_calibration_spec(value["frozen_calibration"])
    allocations = value["allocations"]
    if not isinstance(allocations, list) or not allocations:
        raise ArtifactValidationError("allocations must be a non-empty list")

    indexed: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    request_splits: dict[str, str] = {}
    for index, allocation in enumerate(allocations):
        identity, metadata = _validate_allocation(
            allocation,
            frozen_digest=frozen_digest,
            path=f"$.allocations[{index}]",
        )
        if identity in indexed:
            raise ArtifactValidationError(
                "duplicate arm/rate/layer/request allocation identity"
            )
        indexed[identity] = metadata
        request_id = identity[3]
        prior_split = request_splits.setdefault(request_id, metadata["split"])
        if prior_split != metadata["split"]:
            raise ArtifactValidationError(
                f"request {request_id!r} appears in both calibration and evaluation"
            )

    chains = value["nesting_chains"]
    if not isinstance(chains, list) or not chains:
        raise ArtifactValidationError("nesting_chains must be a non-empty list")
    chain_names: set[str] = set()
    for chain_index, chain in enumerate(chains):
        path = f"$.nesting_chains[{chain_index}]"
        if not isinstance(chain, Mapping):
            raise ArtifactValidationError(f"{path} must be an object")
        name = _nonempty_string(chain.get("name"), field=f"{path}.name")
        if name in chain_names:
            raise ArtifactValidationError("duplicate nesting-chain name")
        chain_names.add(name)
        members = chain.get("members")
        if not isinstance(members, list) or len(members) < 2:
            raise ArtifactValidationError(f"{path}.members must contain at least two states")
        identities: list[tuple[str, int, int, str]] = []
        for member_index, member in enumerate(members):
            member_path = f"{path}.members[{member_index}]"
            if not isinstance(member, Mapping) or set(member) != set(IDENTITY_FIELDS):
                raise ArtifactValidationError(
                    f"{member_path} must contain exactly {list(IDENTITY_FIELDS)}"
                )
            identity = _identity(member, path=member_path)
            if identity not in indexed:
                raise ArtifactValidationError(f"{member_path} references no allocation")
            identities.append(identity)
        if len(set(identities)) != len(identities):
            raise ArtifactValidationError(f"{path} repeats an allocation")
        arms = {identity[0] for identity in identities}
        layers = {identity[2] for identity in identities}
        requests = {identity[3] for identity in identities}
        rates = [identity[1] for identity in identities]
        if len(arms) != 1 or len(layers) != 1 or len(requests) != 1:
            raise ArtifactValidationError(
                f"{path} must keep arm, layer, and request fixed"
            )
        if any(right <= left for left, right in zip(rates, rates[1:])):
            raise ArtifactValidationError(f"{path} rates must increase literally")
        for low_identity, high_identity in zip(identities, identities[1:]):
            low = indexed[low_identity]
            high = indexed[high_identity]
            if low["split"] != high["split"]:
                raise ArtifactValidationError(f"{path} crosses request splits")
            if low["expert_ids"] != high["expert_ids"]:
                raise ArtifactValidationError(
                    f"{path} state rows are not aligned by literal expert ID"
                )
            if low["freeze_state"] != high["freeze_state"]:
                raise ArtifactValidationError(
                    f"{path} members do not share one literal frozen common core"
                )
            if not physical_subset(low["selected_state"], high["selected_state"]):
                raise ArtifactValidationError(f"{path} violates literal bitwise nesting")
            if low["selected_page_cap"] > high["selected_page_cap"]:
                raise ArtifactValidationError(f"{path} page caps decrease")

    return json.loads(canonical_json_bytes(value))


def allocation_manifest_sha256(value: Any) -> str:
    validated = validate_allocation_manifest(value)
    return canonical_sha256(validated)


def _json_load_no_duplicates(payload: bytes, *, path: Path) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in pairs:
            if key in result:
                raise ArtifactValidationError(f"duplicate JSON key {key!r} in {path}")
            result[key] = child
        return result

    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=object_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"invalid UTF-8 JSON in {path}: {exc}") from exc


def write_sealed_allocation_manifest(
    manifest: Any,
    *,
    manifest_path: Path,
    seal_path: Path,
) -> dict[str, Any]:
    """Validate, canonically write, and checksum-seal an allocation manifest."""

    manifest_path = Path(manifest_path)
    seal_path = Path(seal_path)
    if manifest_path.resolve() == seal_path.resolve():
        raise ArtifactValidationError("manifest and seal paths must differ")
    validated = validate_allocation_manifest(manifest)
    manifest_payload = canonical_json_bytes(validated)
    digest = hashlib.sha256(manifest_payload).hexdigest()
    frozen_digest = validate_frozen_calibration_spec(validated["frozen_calibration"])
    seal = {
        "schema": ALLOCATION_SEAL_SCHEMA,
        "allocation_manifest_file": manifest_path.name,
        "allocation_manifest_bytes": len(manifest_payload),
        "allocation_manifest_sha256": digest,
        "frozen_calibration_spec_sha256": frozen_digest,
    }
    seal_payload = canonical_json_bytes(seal)
    manifest_temporary: Path | None = None
    seal_temporary: Path | None = None
    try:
        manifest_temporary = _stage_bytes(manifest_path, manifest_payload)
        seal_temporary = _stage_bytes(seal_path, seal_payload)
        os.replace(manifest_temporary, manifest_path)
        manifest_temporary = None
        os.replace(seal_temporary, seal_path)
        seal_temporary = None
    finally:
        for temporary in (manifest_temporary, seal_temporary):
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return seal


def load_sealed_allocation_manifest(
    *,
    manifest_path: Path,
    seal_path: Path,
    expected_manifest_sha256: str | None = None,
    expected_frozen_calibration_spec_sha256: str | None = None,
) -> dict[str, Any]:
    """Authenticate raw manifest bytes before returning allocations to a tail."""

    manifest_path = Path(manifest_path)
    seal_path = Path(seal_path)
    seal_payload = seal_path.read_bytes()
    seal = _json_load_no_duplicates(seal_payload, path=seal_path)
    if canonical_json_bytes(seal) != seal_payload:
        raise ArtifactValidationError("allocation seal is not canonical JSON")
    required_seal = {
        "schema",
        "allocation_manifest_file",
        "allocation_manifest_bytes",
        "allocation_manifest_sha256",
        "frozen_calibration_spec_sha256",
    }
    if not isinstance(seal, Mapping) or set(seal) != required_seal:
        raise ArtifactValidationError("invalid allocation seal fields")
    if seal["schema"] != ALLOCATION_SEAL_SCHEMA:
        raise ArtifactValidationError("unexpected allocation seal schema")
    if seal["allocation_manifest_file"] != manifest_path.name:
        raise ArtifactValidationError("allocation seal names a different manifest")

    # This check deliberately precedes JSON parsing or allocation access.
    manifest_payload = manifest_path.read_bytes()
    observed_digest = hashlib.sha256(manifest_payload).hexdigest()
    sealed_digest = _normalise_sha256(
        seal["allocation_manifest_sha256"], field="sealed allocation manifest sha256"
    )
    if observed_digest != sealed_digest:
        raise ArtifactValidationError("sealed allocation manifest SHA-256 mismatch")
    if expected_manifest_sha256 is not None and observed_digest != _normalise_sha256(
        expected_manifest_sha256, field="expected allocation manifest sha256"
    ):
        raise ArtifactValidationError("allocation manifest does not match expected digest")
    if seal["allocation_manifest_bytes"] != len(manifest_payload):
        raise ArtifactValidationError("sealed allocation manifest byte count mismatch")

    manifest = _json_load_no_duplicates(manifest_payload, path=manifest_path)
    if canonical_json_bytes(manifest) != manifest_payload:
        raise ArtifactValidationError("allocation manifest is not canonical JSON")
    validated = validate_allocation_manifest(manifest)
    frozen_digest = validate_frozen_calibration_spec(validated["frozen_calibration"])
    sealed_frozen = _normalise_sha256(
        seal["frozen_calibration_spec_sha256"],
        field="sealed frozen calibration spec sha256",
    )
    if frozen_digest != sealed_frozen:
        raise ArtifactValidationError("sealed frozen calibration spec SHA-256 mismatch")
    if (
        expected_frozen_calibration_spec_sha256 is not None
        and frozen_digest
        != _normalise_sha256(
            expected_frozen_calibration_spec_sha256,
            field="expected frozen calibration spec sha256",
        )
    ):
        raise ArtifactValidationError(
            "frozen calibration spec does not match expected digest"
        )
    return validated
