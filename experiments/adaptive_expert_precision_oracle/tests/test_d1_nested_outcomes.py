from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import pytest


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "src"))

from oracle_study.d1_nested_artifacts import (  # noqa: E402
    ALLOCATION_MANIFEST_SCHEMA,
    ArtifactValidationError,
    canonical_sha256,
    encode_state,
    freeze_calibration_spec,
    write_sealed_allocation_manifest,
)
from oracle_study.d1_nested_outcomes import (  # noqa: E402
    ALLOCATION_CONFIG_CANONICAL_SHA256,
    ALLOCATION_CONFIG_FILE_SHA256,
    ROUTE_MODES,
    join_request_manifest,
    load_outcome_plan,
    outcome_input_facts,
    request_layer_means,
    require_zero_dose_parity,
    route_mode_contrasts,
    validate_complete_allocation_grid,
    validate_outcome_config,
    validate_plan_against_config,
)


def _state(*bits: tuple[int, int, int]) -> list[list[int]]:
    result = [[0] * 512 for _ in range(8)]
    for expert, unit, bit in bits:
        result[expert][unit] |= bit
    return result


def _moves(*bits: tuple[int, int, int]) -> list[dict[str, object]]:
    names = {1: "down", 2: "up", 4: "gate"}
    current: dict[tuple[int, int], int] = {}
    result = []
    for step, (expert, unit, bit) in enumerate(bits):
        source = current.get((expert, unit), 0)
        destination = source | bit
        result.append({
            "step": step,
            "expert": expert,
            "expert_id": expert + 10,
            "unit": unit,
            "bit": bit,
            "projection": names[bit],
            "source_state": source,
            "destination_state": destination,
            "page_delta": 1,
        })
        current[(expert, unit)] = destination
    return result


def _allocation(
    frozen: str,
    *,
    request_id: str,
    layer: int,
    rate: int,
    selected: list[list[int]],
    moves: list[dict[str, object]],
) -> dict[str, object]:
    cap = rate * 8
    return {
        "arm": "nested_d1_safe",
        "rate": rate,
        "layer": layer,
        "request_id": request_id,
        "split": "evaluation",
        "domain": "code",
        "prompt_sha256": "a" * 64,
        "position": 2,
        "calibration_spec_sha256": frozen,
        "expert_ids": list(range(10, 18)),
        "freeze_state": encode_state(_state(), page_cap=cap),
        "selected_state": encode_state(selected, page_cap=cap),
        "moves": moves,
    }


def _manifest(requests: tuple[str, ...] = ("7",)) -> dict[str, object]:
    frozen = freeze_calibration_spec({
        "window": 16,
        "eta": 0.0005,
        "config_file_sha256": "d" * 64,
        "config_canonical_sha256": "e" * 64,
        "request_manifest_sha256": "b" * 64,
        "request_manifest_facts_sha256": "c" * 64,
        "calibration_candidate_code_identity": {
            "schema": "pr13_d1_nested_code_bundle_v1",
            "files": 1,
            "canonical_sha256": "f" * 64,
        },
    })
    allocations = []
    chains = []
    for request_id in requests:
        for layer in (0, 1):
            low = _allocation(
                str(frozen["sha256"]), request_id=request_id, layer=layer,
                rate=360, selected=_state((0, 0, 1)), moves=_moves((0, 0, 1)),
            )
            high = _allocation(
                str(frozen["sha256"]), request_id=request_id, layer=layer,
                rate=384, selected=_state((0, 0, 1), (1, 1, 2)),
                moves=_moves((0, 0, 1), (1, 1, 2)),
            )
            allocations.extend((low, high))
            chains.append({
                "name": f"{request_id}-{layer}",
                "members": [
                    {key: low[key] for key in ("arm", "rate", "layer", "request_id")},
                    {key: high[key] for key in ("arm", "rate", "layer", "request_id")},
                ],
            })
    return {
        "schema": ALLOCATION_MANIFEST_SCHEMA,
        "frozen_calibration": frozen,
        "allocation_inputs": {"config_canonical_sha256": "e" * 64},
        "allocations": allocations,
        "nesting_chains": chains,
    }


def _seal(tmp_path: Path, requests: tuple[str, ...] = ("7",)):
    manifest_path = tmp_path / "allocations.json"
    seal_path = tmp_path / "allocations.seal.json"
    seal = write_sealed_allocation_manifest(
        _manifest(requests), manifest_path=manifest_path, seal_path=seal_path,
    )
    return manifest_path, seal_path, seal


def _request(request_id: str = "7") -> dict[str, object]:
    return {
        "request_id": request_id,
        "split": "evaluation",
        "domain": "code",
        "prompt_sha256": "a" * 64,
        "prompt_token_ids": [10, 11, 12, 13],
        "decode_position": 2,
        "next_token_id": 13,
    }


def test_outcome_plan_requires_authentic_seal_and_decodes_full_state(
    tmp_path: Path,
) -> None:
    manifest, seal_path, seal = _seal(tmp_path)
    plan = load_outcome_plan(
        manifest_path=manifest,
        seal_path=seal_path,
        expected_manifest_sha256=str(seal["allocation_manifest_sha256"]),
        expected_frozen_calibration_spec_sha256=str(
            seal["frozen_calibration_spec_sha256"]
        ),
    )
    assert len(plan.allocations) == 4
    assert plan.allocations[0].selected_state.shape == (8, 512)
    assert plan.allocations[0].selected_state.dtype.name == "uint8"
    assert plan.allocations[0].selected_pages == 1
    assert len(plan.allocations[0].state_sha256) == 64
    assert plan.allocation_config_file_sha256 == "d" * 64
    assert plan.request_manifest_sha256 == "b" * 64
    assert plan.request_manifest_facts_sha256 == "c" * 64
    assert plan.calibration_candidate_code_identity == {
        "schema": "pr13_d1_nested_code_bundle_v1",
        "files": 1,
        "canonical_sha256": "f" * 64,
    }
    input_facts = outcome_input_facts(plan)
    assert input_facts["allocation_config_file_sha256"] == "d" * 64
    assert input_facts["allocation_config_canonical_sha256"] == "e" * 64
    assert input_facts["calibration_candidate_code_identity"] == (
        plan.calibration_candidate_code_identity
    )

    manifest.write_bytes(manifest.read_bytes() + b"\n")
    with pytest.raises(ArtifactValidationError, match="SHA-256 mismatch"):
        load_outcome_plan(manifest_path=manifest, seal_path=seal_path)


def test_request_join_checks_split_prompt_domain_position_and_label(
    tmp_path: Path,
) -> None:
    manifest, seal, _ = _seal(tmp_path)
    plan = load_outcome_plan(manifest_path=manifest, seal_path=seal)
    joined = join_request_manifest(plan.allocations, [_request()])
    assert joined["7"].next_token_id == 13

    wrong = _request()
    wrong["decode_position"] = 1
    wrong["next_token_id"] = 12
    with pytest.raises(ValueError, match="position mismatch"):
        join_request_manifest(plan.allocations, [wrong])


def test_allocation_grid_rejects_one_missing_request_cell(tmp_path: Path) -> None:
    manifest, seal, _ = _seal(tmp_path, ("7", "8"))
    plan = load_outcome_plan(manifest_path=manifest, seal_path=seal)
    grid = validate_complete_allocation_grid(plan.allocations)
    assert len(grid) == 4
    incomplete = list(plan.allocations)
    incomplete.pop()
    with pytest.raises(ValueError, match="incomplete"):
        validate_complete_allocation_grid(incomplete)


def _quality() -> pd.DataFrame:
    rows = []
    values = {
        "live": 0.8,
        "frozen_set_live_weights": 0.5,
        "fully_frozen": 0.2,
    }
    for request in ("7", "8"):
        for layer in (0, 1):
            for mode in ROUTE_MODES:
                rows.append({
                    "split": "evaluation",
                    "arm": "nested_d1_safe",
                    "rate_pages_per_expert": 360,
                    "injection_layer": layer,
                    "request_id": request,
                    "position": 2,
                    "route_mode": mode,
                    "logit_kl": values[mode] + layer,
                    "delta_nll": values[mode] / 10 + layer,
                    "final_hidden_mse": values[mode] / 100 + layer,
                })
    return pd.DataFrame(rows)


def test_route_mode_contrasts_pair_all_three_controls() -> None:
    paired = route_mode_contrasts(_quality())
    assert len(paired) == 4
    assert paired["live_minus_fully_frozen_logit_kl"].tolist() == pytest.approx(
        [0.6] * 4,
    )
    assert paired["live_minus_frozen_set_live_weights_logit_kl"].tolist() == pytest.approx(
        [0.3] * 4,
    )
    missing = _quality().query("route_mode != 'fully_frozen'")
    with pytest.raises(ValueError, match="route modes changed"):
        route_mode_contrasts(missing)


def test_zero_gate_requires_every_hybrid_cache_commit() -> None:
    rows = []
    for mode in ROUTE_MODES:
        rows.append({
            "route_mode": mode,
            "all_exact": True,
            "paired_baseline_full_cache_bit_identical": True,
            "post_token_full_cache_bit_identical": True,
            "post_token_attention_kv_bit_identical": True,
            "post_token_deltanet_conv_bit_identical": True,
            "post_token_deltanet_recurrent_bit_identical": True,
            "routed_repeat_max_abs": 0.0,
        })
    frame = pd.DataFrame(rows)
    require_zero_dose_parity(frame)
    broken = frame.copy()
    broken.loc[0, "post_token_deltanet_recurrent_bit_identical"] = False
    with pytest.raises(RuntimeError, match="deltanet_recurrent"):
        require_zero_dose_parity(broken)

    malformed = frame.copy().astype({"all_exact": object})
    malformed.loc[0, "all_exact"] = "False"
    with pytest.raises(RuntimeError, match="all_exact"):
        require_zero_dose_parity(malformed)


def test_request_means_retain_route_mode_and_weight_layers_equally() -> None:
    means = request_layer_means(
        _quality(), ("logit_kl", "delta_nll", "final_hidden_mse"),
    )
    assert len(means) == 2 * len(ROUTE_MODES)
    live = means.query("request_id == '7' and route_mode == 'live'").iloc[0]
    assert live["logit_kl"] == pytest.approx(1.3)
    assert live["layers_averaged"] == 2

    incomplete = _quality().drop(index=[0]).reset_index(drop=True)
    with pytest.raises(ValueError, match="layer grid is incomplete"):
        request_layer_means(incomplete, ("logit_kl",))


def test_checked_in_config_has_exact_outcome_only_contract() -> None:
    path = (
        EXPERIMENT / "configs"
        / "qwen36_mxfp4_d1_nested_safe_allocation_20260825_v2.json"
    )
    config = json.loads(path.read_text())
    validate_outcome_config(config)
    changed = deepcopy(config)
    changed["outcome_phase"]["route_modes"] = ["live", "fully_frozen"]
    with pytest.raises(ValueError, match="route_modes"):
        validate_outcome_config(changed)
    changed = deepcopy(config)
    changed["authenticated_capture_protocol"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="capture protocol"):
        validate_outcome_config(changed)

    changed = deepcopy(config)
    changed["scientific_boundary"] += " changed"
    with pytest.raises(ValueError, match="canonical SHA-256"):
        validate_outcome_config(changed)

    v1 = json.loads((
        EXPERIMENT / "configs"
        / "qwen36_mxfp4_d1_nested_safe_allocation_20260825_v1.json"
    ).read_text())
    with pytest.raises(ValueError, match="immutable v2"):
        validate_outcome_config(v1)


@pytest.mark.parametrize(
    ("block", "field"),
    [
        ("configuration_provenance", "parent_sha256"),
        ("authenticated_capture_artifacts", "facts_sha256"),
        ("authenticated_capture_execution_stack", "cuda"),
        ("authenticated_pr13_inputs", "factor_manifest_sha256"),
        ("reference_high_core_seal", "literal_subset_and_state_hash_required"),
        ("atomic_resume", "mixed_input_or_code_identity_rejected"),
    ],
)
def test_v2_outcome_config_rejects_authenticated_contract_drift(
    block: str,
    field: str,
) -> None:
    path = (
        EXPERIMENT / "configs"
        / "qwen36_mxfp4_d1_nested_safe_allocation_20260825_v2.json"
    )
    changed = json.loads(path.read_text())
    value = changed[block][field]
    changed[block][field] = not value if isinstance(value, bool) else "0" * 64
    with pytest.raises(ValueError, match=block):
        validate_outcome_config(changed)


def test_v2_plan_preserves_raw_and_canonical_config_identities() -> None:
    path = (
        EXPERIMENT / "configs"
        / "qwen36_mxfp4_d1_nested_safe_allocation_20260825_v2.json"
    )
    raw = path.read_bytes()
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    config = json.loads(raw)
    canonical = canonical_sha256(config)
    assert raw_sha256 == ALLOCATION_CONFIG_FILE_SHA256
    assert canonical == ALLOCATION_CONFIG_CANONICAL_SHA256
    rates = sorted({
        int(pair[field])
        for pair in config["traffic_pairs"]
        for field in (
            "metadata_matched_pages_per_expert",
            "pr13_reference_pages_per_expert",
        )
    })
    allocations = tuple(
        SimpleNamespace(
            arm=arm,
            rate=rate,
            layer=layer,
            request_id=f"request-{request:03d}",
            split="evaluation",
        )
        for request in range(config["request_source"]["evaluation_requests"])
        for arm in config["arms"]
        for rate in rates
        for layer in config["injection_layers"]
    )
    plan = SimpleNamespace(
        allocations=allocations,
        allocation_config_file_sha256=raw_sha256,
        allocation_config_canonical_sha256=canonical,
    )
    validate_plan_against_config(
        plan, config, config_file_sha256=raw_sha256,
    )

    with pytest.raises(ValueError, match="raw allocation config"):
        validate_plan_against_config(
            plan, config, config_file_sha256="0" * 64,
        )
    wrong_canonical = SimpleNamespace(
        allocations=allocations,
        allocation_config_file_sha256=raw_sha256,
        allocation_config_canonical_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="frozen allocation config"):
        validate_plan_against_config(
            wrong_canonical, config, config_file_sha256=raw_sha256,
        )
