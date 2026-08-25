from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "src"))

from oracle_study.d1_nested_artifacts import (  # noqa: E402
    ArtifactValidationError,
    freeze_calibration_spec,
    load_sealed_allocation_manifest,
)
from oracle_study.d1_nested_calibration import (  # noqa: E402
    ARM_D1,
    ARM_D1_SEVERITY,
    ARM_LOCAL,
    ARM_PR13,
    ARM_STRICT,
    CANDIDATE_SCHEMA,
    FIVE_ARMS,
    build_evaluation_allocation_manifest,
    load_layer_candidate_pickles,
    select_and_freeze_calibration,
    validate_calibration_selection,
    write_calibration_selection,
    write_sealed_evaluation_allocations,
)


def _config() -> dict[str, object]:
    return {
        "run_id": "nested-calibration-test",
        "schema": "pr13_d1_nested_safe_allocation_config_v1",
        "experiment_stage": "A",
        "authenticated_capture_protocol": {
            "path": "configs/qwen36_mxfp4_d1_nested_safe_oracle_20260825_v1.json",
            "sha256": "c37eb2c63de64dba87baa44b0ef57fc2206b32db8f82dfc2d49a9e9da0da1713",
        },
        "d1_definition": "same_decode_token_next_layer_router_only",
        "d2_d4_objectives_models_or_selector_inputs_present": False,
        "downstream_routes_and_terminal_kl_role": "sealed_outcome_phase_only",
        "experiment_b_started": False,
        "runtime_predictor_in_scope": False,
        "generated_rollout_in_scope": False,
        "joint_all_layer_compression_in_scope": False,
        "arms": list(FIVE_ARMS),
        "injection_layers": [0],
        "traffic_pairs": [{
            "metadata_matched_pages_per_expert": 2,
            "pr13_reference_pages_per_expert": 3,
            "metadata_matched_group_cap": 16,
            "pr13_reference_group_cap": 24,
        }],
        "decode_position": {
            "exact_prefill": True, "query_length": 1, "positions_per_request": 1,
        },
        "allocation_phase": {
            "terminal_logits_captured": False,
            "downstream_layers_after_d1_executed": False,
            "pair_fallback_flag_serialized": True,
            "high_guardrail_qualified_checkpoint_count_serialized": True,
            "high_guardrail_repair_attempt_count_serialized": True,
            "pair_fallback_frequency_reported_once_per_policy_pair": True,
        },
        "exact_d1_gate": {
            "all_256_router_logits_checked": True,
            "safe_incumbent_is_immutable": True,
            "unsafe_replacement_requires_strictly_fewer_membership_changes": True,
            "high_completion_path_scan": "complete_add_only_local_path_highest_pages_then_lowest_local",
            "high_checkpoint_admission": "strict_high_endpoint_local_guard_and_exact_d1_nonworsening_against_selected_low",
            "same_rate_high_incumbent_semantics": "safe_arm3_high_byte_identical_else_changed_high_requires_strict_crossing_reduction",
            "high_guardrail_infeasible_action": "revert_entire_low_high_policy_pair_byte_for_byte_to_arm3",
            "pair_fallback_preserves_fixed_grid": True,
        },
        "common_core": {
            "d1_blind": True,
            "repair_window_pages_per_group_grid": [1, 2],
        },
        "d1_screen": {
            "raw_slice_router_anchor_max_abs": 0.0625,
            "captured_full_model_baseline_logit_anchor_applied": True,
            "zero_delta_after_anchor_must_equal_captured_logits": True,
            "candidate_page_shortlist": 32,
            "maximum_exact_finalists": 8,
        },
        "local_guardrail": {
            "form": "J_candidate_le_J_nested_local_endpoint_plus_eta_times_J_all_Q2",
            "eta_grid": [0.0, 0.1],
        },
        "request_source": {
            "calibration_requests": 1,
            "evaluation_requests": 1,
        },
        "calibration": {
            "terminal_kl_used": False,
            "downstream_routes_used": False,
            "selection_per_rate": False,
            "selection_per_traffic_pair": True,
            "metadata_matched_low_endpoint_selects_shared_pair_parameters": True,
            "selection_order": [
                "zero_safe_to_unsafe_events",
                "fewest_unresolved_exact_d1_crossings",
                "smallest_repair_window_pages",
                "smallest_eta",
                "lowest_mean_exact_combined_local_qenergy",
                "deterministic_parameter_hash",
            ],
            "parameters_frozen_before_evaluation_allocation": True,
        },
    }


def _state(*pages: tuple[int, int, int]) -> np.ndarray:
    result = np.zeros((8, 512), np.uint8)
    for expert, unit, bit in pages:
        result[expert, unit] |= bit
    return result


def _moves(source: np.ndarray, destination: np.ndarray) -> list[dict[str, object]]:
    current = np.asarray(source, np.uint8).copy()
    result = []
    names = {1: "down", 2: "up", 4: "gate"}
    for expert in range(8):
        for unit in range(512):
            for bit in (1, 2, 4):
                if current[expert, unit] & bit or not destination[expert, unit] & bit:
                    continue
                before = int(current[expert, unit])
                after = before | bit
                result.append({
                    "expert": expert,
                    "unit": unit,
                    "bit": bit,
                    "projection": names[bit],
                    "source_state": before,
                    "destination_state": after,
                })
                current[expert, unit] = after
    np.testing.assert_array_equal(current, destination)
    return result


def _page_count(state: np.ndarray) -> int:
    return sum(int(value).bit_count() for value in state.reshape(-1))


def _row(
    *,
    split: str,
    arm: str,
    rate: int,
    selected: np.ndarray,
    core: np.ndarray | None = None,
    parameter: dict[str, object] | None = None,
    crossings: int = 1,
    local_damage: float = 1.0,
    request_id: str = "request-0",
) -> dict[str, object]:
    selected = np.asarray(selected, np.uint8)
    frozen = selected.copy() if core is None else np.asarray(core, np.uint8)
    target_ids = list(range(8))
    candidate_ids = target_ids.copy()
    if crossings:
        candidate_ids[-crossings:] = list(range(8, 8 + crossings))
    prompt_sha = hashlib.sha256(request_id.encode()).hexdigest()
    return {
        "schema": CANDIDATE_SCHEMA,
        "arm": arm,
        "rate": rate,
        "layer": 0,
        "group": 0,
        "request_id": request_id,
        "prompt_sha256": prompt_sha,
        "domain": "reasoning",
        "split": split,
        "position": 8,
        "expert_ids": list(range(8)),
        "selector_weights": [0.125] * 8,
        "execution_weights": [0.125] * 8,
        "common_core_states": frozen,
        "selected_states": selected,
        "moves": _moves(frozen, selected),
        "selected_pages": _page_count(selected),
        "core_pages": _page_count(frozen),
        "local_damage": local_damage,
        "all_q2_damage": 1.0,
        "legacy_additive_damage": local_damage + 0.1,
        "d1_candidate_logits": np.linspace(1.0, -1.0, 256, dtype=np.float32),
        "d1_target_ids": target_ids,
        "d1_candidate_ids": candidate_ids,
        "d1_crossings": crossings,
        "d1_violation_depth": float(crossings) * 0.2,
        "d1_routing_mass_churn": float(crossings) * 0.01,
        "d1_labeled_margin": 0.1 if crossings == 0 else -0.1,
        "parameter": {} if parameter is None else parameter,
        "state_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
    }


def _write_pickle(
    tmp_path: Path, split: str, rows: list[dict[str, object]],
) -> Path:
    path = tmp_path / split / "layer_00" / "nested_candidates.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def _load(path: Path, split: str):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return load_layer_candidate_pickles(
        [path],
        expected_split=split,
        expected_layers=[0],
        expected_file_sha256={str(path): digest},
    )


def _select(
    corpus,
    config: dict[str, object] | None = None,
    code_identity: dict[str, object] | None = None,
):
    return select_and_freeze_calibration(
        corpus,
        _config() if config is None else config,
        expected_request_ids=["request-0"],
        config_file_sha256="a" * 64,
        request_manifest_sha256="b" * 64,
        request_manifest_facts_sha256="c" * 64,
        calibration_candidate_code_identity=code_identity,
    )


def _calibration_rows() -> list[dict[str, object]]:
    core = _state()
    local_state = _state((0, 0, 1))
    repaired_state = _state((0, 1, 2))
    rows: list[dict[str, object]] = []
    for window in (1, 2):
        rows.append(_row(
            split="calibration",
            arm=ARM_LOCAL,
            rate=2,
            core=core,
            selected=local_state,
            parameter={"repair_window": window, "endpoint": "low"},
            crossings=1,
        ))
        for eta in (0.0, 0.1):
            crossings = 0 if (window, eta) == (2, 0.1) else 1
            rows.append(_row(
                split="calibration",
                arm=ARM_D1,
                rate=2,
                core=core,
                selected=repaired_state if crossings == 0 else local_state,
                parameter={
                    "repair_window": window,
                    "eta": eta,
                    "endpoint": "low",
                },
                crossings=crossings,
                local_damage=1.05 if eta else 1.0,
            ))
    return rows


def _evaluation_rows(
    *, d1_local_damage: float = 1.05, request_id: str = "request-eval",
) -> list[dict[str, object]]:
    core = _state()
    rows: list[dict[str, object]] = []
    for rate, endpoint in ((2, "low"), (3, "high")):
        pr13_state = _state((1, 0, 1)) if rate == 2 else _state(
            (1, 0, 1), (1, 1, 4),
        )
        local_state = _state((0, 0, 1)) if rate == 2 else _state(
            (0, 0, 1), (0, 2, 4),
        )
        repaired_state = _state((0, 1, 2)) if rate == 2 else _state(
            (0, 1, 2), (0, 2, 4),
        )
        rows.extend((
            _row(
                split="evaluation", arm=ARM_PR13, rate=rate,
                selected=pr13_state, crossings=0,
                parameter={"independent_column_generated": True},
                request_id=request_id,
            ),
            _row(
                split="evaluation", arm=ARM_STRICT, rate=rate,
                selected=pr13_state, crossings=0,
                parameter={"frozen_candidate_bank": True},
                request_id=request_id,
            ),
            _row(
                split="evaluation", arm=ARM_LOCAL, rate=rate,
                core=core, selected=local_state, crossings=1,
                parameter={"repair_window": 2, "endpoint": endpoint},
                request_id=request_id,
            ),
            _row(
                split="evaluation", arm=ARM_D1, rate=rate,
                core=core, selected=repaired_state, crossings=0,
                local_damage=d1_local_damage,
                parameter={"repair_window": 2, "eta": 0.1, "endpoint": endpoint},
                request_id=request_id,
            ),
            _row(
                split="evaluation", arm=ARM_D1_SEVERITY, rate=rate,
                core=core, selected=repaired_state, crossings=0,
                local_damage=d1_local_damage,
                parameter={"repair_window": 2, "eta": 0.1, "endpoint": endpoint},
                request_id=request_id,
            ),
        ))
    return rows


def _selection(tmp_path: Path):
    path = _write_pickle(tmp_path, "calibration", _calibration_rows())
    return _select(_load(path, "calibration"))


def test_calibration_selects_arm4_low_endpoint_and_freezes_pair_parameters(
    tmp_path: Path,
) -> None:
    selection = _selection(tmp_path)
    spec = selection.frozen_calibration["spec"]
    assert spec["traffic_pairs"][0]["repair_window_pages"] == 2
    assert spec["traffic_pairs"][0]["eta"] == pytest.approx(0.1)
    assert spec["rates"]["2"]["repair_window_pages"] == 2
    assert spec["rates"]["3"]["repair_window_pages"] == 2
    assert spec["rates"]["3"]["selection_source_rate"] == 2
    assert spec["arm5_inherits_arm4_parameters"] is True
    assert sum(row["selected"] for row in selection.evidence["rows"]) == 1
    assert selection.sha256 == selection.frozen_calibration["sha256"]
    facts = write_calibration_selection(
        selection,
        frozen_path=tmp_path / "frozen.json",
        evidence_path=tmp_path / "evidence.json",
    )
    assert facts["frozen_calibration_spec_sha256"] == selection.sha256
    assert (tmp_path / "frozen.json").is_file()
    assert (tmp_path / "evidence.json").is_file()


def test_calibration_rejects_incomplete_grid_and_candidate_outcome_leakage(
    tmp_path: Path,
) -> None:
    incomplete = _write_pickle(
        tmp_path / "incomplete", "calibration", _calibration_rows()[:-1],
    )
    corpus = _load(incomplete, "calibration")
    with pytest.raises(ArtifactValidationError, match="grid|coverage"):
        _select(corpus)

    leaked_rows = _calibration_rows()
    leaked_rows[0]["terminal_logits"] = [0.0]
    leaked = _write_pickle(tmp_path / "leaked", "calibration", leaked_rows)
    with pytest.raises(ArtifactValidationError, match="outcome leakage"):
        _load(leaked, "calibration")


def test_candidate_pickle_hash_can_be_externally_pinned(tmp_path: Path) -> None:
    path = _write_pickle(tmp_path, "calibration", _calibration_rows())
    with pytest.raises(ArtifactValidationError, match="hash mismatch"):
        load_layer_candidate_pickles(
            [path],
            expected_split="calibration",
            expected_layers=[0],
            expected_file_sha256={str(path): "0" * 64},
        )

    corpus = _load(path, "calibration")
    ambiguous = deepcopy(_config())
    ambiguous["calibration"]["selection_per_rate"] = True
    ambiguous["calibration"]["selection_per_traffic_pair"] = False
    with pytest.raises(ArtifactValidationError, match="calibration firewall"):
        _select(corpus, ambiguous)

    capture_drift = deepcopy(_config())
    capture_drift["authenticated_capture_protocol"]["sha256"] = "f" * 64
    with pytest.raises(ArtifactValidationError, match="authenticated capture"):
        _select(corpus, capture_drift)

    for key, value in (
        ("raw_slice_router_anchor_max_abs", 0.0626),
        ("captured_full_model_baseline_logit_anchor_applied", False),
        ("zero_delta_after_anchor_must_equal_captured_logits", False),
        ("candidate_page_shortlist", 64),
        ("maximum_exact_finalists", 9),
    ):
        anchor_drift = deepcopy(_config())
        anchor_drift["d1_screen"][key] = value
        with pytest.raises(ArtifactValidationError, match="router-anchor/search"):
            _select(corpus, anchor_drift)

    for section, key, value in (
        ("exact_d1_gate", "pair_fallback_preserves_fixed_grid", False),
        (
            "exact_d1_gate",
            "same_rate_high_incumbent_semantics",
            "changed",
        ),
        ("allocation_phase", "pair_fallback_flag_serialized", False),
        (
            "allocation_phase",
            "pair_fallback_frequency_reported_once_per_policy_pair",
            False,
        ),
    ):
        fallback_drift = deepcopy(_config())
        fallback_drift[section][key] = value
        with pytest.raises(
            ArtifactValidationError, match="exact-prefill D1 allocation semantics"
        ):
            _select(corpus, fallback_drift)


def test_frozen_rate_cannot_drift_from_its_traffic_pair(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    changed_spec = deepcopy(selection.frozen_calibration["spec"])
    changed_spec["rates"]["3"]["repair_window_pages"] = 1
    changed = freeze_calibration_spec(changed_spec)
    with pytest.raises(ArtifactValidationError, match="drifted from its pair"):
        validate_calibration_selection(changed, _config())


def test_evaluation_grid_is_converted_to_canonical_manifest_and_sealed(
    tmp_path: Path,
) -> None:
    selection = _selection(tmp_path / "cal")
    path = _write_pickle(tmp_path, "evaluation", _evaluation_rows())
    corpus = _load(path, "evaluation")
    manifest = build_evaluation_allocation_manifest(
        corpus, _config(), selection, expected_request_ids=["request-eval"],
    )
    assert len(manifest["allocations"]) == 10
    assert len(manifest["nesting_chains"]) == 3
    assert {
        row["arm"] for row in manifest["allocations"]
    } == set(FIVE_ARMS)
    nested = [
        row for row in manifest["allocations"] if row["arm"] == ARM_D1
    ]
    assert {row["parameter"]["frozen"]["repair_window_pages"] for row in nested} == {2}
    assert {row["parameter"]["frozen"]["eta"] for row in nested} == {0.1}

    manifest_path = tmp_path / "sealed" / "allocations.json"
    seal_path = tmp_path / "sealed" / "allocations.seal.json"
    seal = write_sealed_evaluation_allocations(
        corpus,
        _config(),
        selection,
        manifest_path=manifest_path,
        seal_path=seal_path,
        expected_request_ids=["request-eval"],
    )
    loaded = load_sealed_allocation_manifest(
        manifest_path=manifest_path,
        seal_path=seal_path,
        expected_manifest_sha256=seal["allocation_manifest_sha256"],
        expected_frozen_calibration_spec_sha256=selection.sha256,
    )
    assert loaded == manifest


def test_evaluation_seal_rejects_local_guard_and_split_misuse(tmp_path: Path) -> None:
    selection = _selection(tmp_path / "cal")
    bad_path = _write_pickle(
        tmp_path / "bad", "evaluation", _evaluation_rows(d1_local_damage=2.0),
    )
    bad = _load(bad_path, "evaluation")
    with pytest.raises(ArtifactValidationError, match="local guardrail"):
        build_evaluation_allocation_manifest(
            bad, _config(), selection, expected_request_ids=["request-eval"],
        )

    calibration_path = _write_pickle(
        tmp_path / "wrong", "calibration", _calibration_rows(),
    )
    calibration = _load(calibration_path, "calibration")
    with pytest.raises(ArtifactValidationError, match="evaluation pickles"):
        build_evaluation_allocation_manifest(
            calibration, _config(), selection, expected_request_ids=["request-0"],
        )

    overlap_path = _write_pickle(
        tmp_path / "overlap", "evaluation", _evaluation_rows(request_id="request-0"),
    )
    overlap = _load(overlap_path, "evaluation")
    with pytest.raises(ArtifactValidationError, match="cohorts overlap"):
        build_evaluation_allocation_manifest(
            overlap, _config(), selection, expected_request_ids=["request-0"],
        )

    off_grid_rows = _evaluation_rows()
    off_grid_rows.append(_row(
        split="evaluation",
        arm=ARM_LOCAL,
        rate=2,
        core=_state(),
        selected=_state((0, 0, 1)),
        crossings=1,
        parameter={"repair_window": 1, "endpoint": "low"},
        request_id="request-eval",
    ))
    off_grid_path = _write_pickle(
        tmp_path / "off-grid", "evaluation", off_grid_rows,
    )
    off_grid = _load(off_grid_path, "evaluation")
    with pytest.raises(ArtifactValidationError, match="not frozen"):
        build_evaluation_allocation_manifest(
            off_grid, _config(), selection, expected_request_ids=["request-eval"],
        )


def test_frozen_calibration_carries_candidate_code_identity(
    tmp_path: Path,
) -> None:
    identity = {
        "schema": "pr13_d1_nested_code_bundle_v1",
        "files": 17,
        "canonical_sha256": "d" * 64,
    }
    path = _write_pickle(tmp_path, "calibration", _calibration_rows())
    selection = _select(
        _load(path, "calibration"), code_identity=identity,
    )
    assert (
        selection.frozen_calibration["spec"]
        ["calibration_candidate_code_identity"]
    ) == identity
    validated = validate_calibration_selection(
        selection.frozen_calibration, _config(),
    )
    assert validated["spec"]["calibration_candidate_code_identity"] == identity


def test_calibration_and_sealing_reject_inconsistent_authoritative_target_ids(
    tmp_path: Path,
) -> None:
    calibration_rows = _calibration_rows()
    calibration_rows[-1]["d1_target_ids"] = [1, 0, 2, 3, 4, 5, 6, 7]
    calibration_path = _write_pickle(
        tmp_path / "calibration-target-drift",
        "calibration",
        calibration_rows,
    )
    with pytest.raises(
        ArtifactValidationError,
        match="authoritative d1_target_ids changed across arms/rates",
    ):
        _select(_load(calibration_path, "calibration"))

    selection = _selection(tmp_path / "selection")
    evaluation_rows = _evaluation_rows()
    evaluation_rows[-1]["d1_target_ids"] = [1, 0, 2, 3, 4, 5, 6, 7]
    evaluation_path = _write_pickle(
        tmp_path / "evaluation-target-drift",
        "evaluation",
        evaluation_rows,
    )
    with pytest.raises(
        ArtifactValidationError,
        match="authoritative d1_target_ids changed across arms/rates",
    ):
        build_evaluation_allocation_manifest(
            _load(evaluation_path, "evaluation"),
            _config(),
            selection,
            expected_request_ids=["request-eval"],
        )
