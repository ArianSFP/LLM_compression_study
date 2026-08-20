from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_set_utility_distillation.py"
SPEC = importlib.util.spec_from_file_location("run_set_utility_distillation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def tiny_contract() -> dict:
    return {
        "layers": [0, 4, 20, 39],
        "selection_split": "validation",
        "evaluation_splits": ["validation"],
        "fit_splits": ["train"],
        "no_test_rows_admitted": True,
        "no_test_rows_used": True,
        "no_test_tuning": True,
        "page_size_bytes": 512,
        "expert_weights": 3_145_728,
        "pq_codebook_size": 16,
        "pq_block_size": 32,
        "candidate_units": 256,
        "applied_units": 192,
        "teacher_path_length": 512,
        "max_validation_invocations_per_expert": 16,
        "max_hybrid_oracle_invocations_per_expert": 16,
        "expected_validation_unique_invocations": 69,
        "expected_validation_layer_expert_cells": 12,
        "expected_cross_capture_shared_request_ids": 0,
        "direct_predictor_candidate_set_weight": 0.0,
        "direct_predictor_candidate_coverage_weight": 1.0,
        "direct_candidate_head_supervision": runner.JOINT_CANDIDATE_COVERAGE_SEMANTICS,
        "template_training_cohorts": list(runner.FIT_COHORTS),
        "locked_exact_sampled_experts": {
            "0": {"1": "cold", "2": "median", "3": "hot"},
            "4": {"4": "cold", "5": "median", "6": "hot"},
            "20": {"7": "cold", "8": "median", "9": "hot"},
            "39": {"10": "cold", "11": "median", "12": "hot"},
        },
        "locked_tree_sha256": "a" * 64,
        "locked_capture_sha256": {"exact_checkpoint": "b" * 64, "cross_reference": "c" * 64},
        "reference": {"config_sha256": "d" * 64, "index_sha256": "e" * 64},
    }


def full_split_fixture() -> dict[str, np.ndarray]:
    return {
        "split": np.array(["train", "validation", "test", "train"]),
        "request_id": np.array(["train-a", "val-a", "test-a", "train-b"]),
        "x": np.arange(12).reshape(4, 3),
        "layer": np.zeros(4, np.int64),
    }


def test_contract_has_no_test_mode_and_fixed_candidate_interface() -> None:
    runner.validate_study_contract(tiny_contract())
    for key, value in (
        ("evaluation_splits", ["validation", "test"]),
        ("fit_splits", ["train", "validation"]),
        ("no_test_rows_admitted", False),
        ("candidate_units", 240),
        ("expected_validation_unique_invocations", 68),
        ("expected_validation_layer_expert_cells", 11),
        ("direct_candidate_head_supervision", "stale"),
    ):
        changed = {**tiny_contract(), key: value}
        with pytest.raises(RuntimeError):
            runner.validate_study_contract(changed)


def test_cross_capture_request_audit_allows_only_same_split_sharing() -> None:
    exact = {
        "split": np.array(["train", "validation", "test"]),
        "request_id": np.array(["shared-train", "exact-val", "exact-test"]),
    }
    cross = {
        "split": np.array(["train", "validation", "test"]),
        "request_id": np.array(["shared-train", "cross-val", "cross-test"]),
    }
    audit = runner.audit_cross_capture_request_ids(
        exact, cross, expected_shared_request_ids=1,
    )
    assert audit["split_compatible"] is True
    assert audit["shared_request_id_count"] == 1
    assert audit["shared_same_split_request_id_counts"] == {
        "train": 1, "validation": 0, "test": 0,
    }
    assert audit["incompatible_overlaps"] == []
    runner.require_locked_request_audit(
        {"cross_capture_request_id_audit": audit},
        {"cross_capture_request_id_audit": audit},
        {"expected_cross_capture_shared_request_ids": 1},
    )


def test_cross_capture_request_audit_rejects_split_mismatch_and_locked_count() -> None:
    exact = {
        "split": np.array(["train", "validation", "test"]),
        "request_id": np.array(["shared", "exact-val", "exact-test"]),
    }
    mismatch = {
        "split": np.array(["validation", "train", "test"]),
        "request_id": np.array(["shared", "cross-train", "cross-test"]),
    }
    with pytest.raises(RuntimeError, match="cross-capture request split mismatch"):
        runner.audit_cross_capture_request_ids(exact, mismatch)
    same = {
        "split": np.array(["train", "validation", "test"]),
        "request_id": np.array(["shared", "cross-val", "cross-test"]),
    }
    with pytest.raises(RuntimeError, match="shared request count"):
        runner.audit_cross_capture_request_ids(
            exact, same, expected_shared_request_ids=0,
        )


def test_capture_admission_verifies_requests_then_removes_test_rows() -> None:
    admitted = runner.admit_capture_rows(full_split_fixture(), ["train", "validation"])
    assert admitted["split"].tolist() == ["train", "validation", "train"]
    assert admitted["request_id"].tolist() == ["train-a", "val-a", "train-b"]
    np.testing.assert_array_equal(admitted["x"], np.array([[0, 1, 2], [3, 4, 5], [9, 10, 11]]))
    with pytest.raises(RuntimeError, match="cannot admit test"):
        runner.admit_capture_rows(full_split_fixture(), ["test"])


def test_capture_admission_rejects_request_level_leakage_before_filtering() -> None:
    data = full_split_fixture()
    data["request_id"][2] = "train-a"
    with pytest.raises(RuntimeError, match="request split leakage"):
        runner.admit_capture_rows(data, ["train"])


def test_template_fit_never_aligns_unit_ids_across_experts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "UNITS", 6)
    rows = []
    for expert, supports in ((3, ([0, 1], [0, 2])), (9, ([4, 5], [3, 5]))):
        for support in supports:
            mask = np.zeros(6, bool)
            mask[list(support)] = True
            utility = np.arange(1, 7, dtype=np.float32)
            rows.append({
                "cohort": runner.FIT_COHORTS[0], "expert_id": expert,
                "activation_source": "exact_checkpoint",
                "request_id": f"train-{expert}-{len(rows)}",
                "mask192_packed": np.packbits(mask, bitorder="little"),
                "path_utility": utility,
            })
    arrays: dict[str, np.ndarray] = {}
    config = {"template_counts_primary": [1, 2, 4], "template_counts_augmented": [1, 2, 4]}
    index = runner.fit_template_arrays(rows, config, arrays)
    assert {entry["expert_id"] for entry in index} == {3, 9}
    assert len({entry["array_key"] for entry in index}) == 2
    for entry in index:
        assert entry["pairing_semantics"] == "actual_routed_occurrences"
        unpacked = np.unpackbits(arrays[entry["array_key"]], axis=1, bitorder="little")[:, :6]
        if entry["expert_id"] == 3:
            assert not unpacked[:, 3:].any()
        else:
            assert not unpacked[:, :3].any()


def test_full_target_and_contained_candidate_reranks_are_not_conflated() -> None:
    corrections = np.array([[1.0, 0.0], [0.0, 0.6], [0.0, 8.0]])
    gram = corrections @ corrections.T
    rows = runner.candidate_rerank_records(
        gram, [0, 1], 1, independent_scores=[0.1, 1.0, 999.0],
        activation_payload_already_read=False,
        q2_payload_already_read=False,
        abc_payload_already_read=False,
        abc_metadata_bytes=30,
        proxy_rank=2,
    )
    assert len(rows) == 4
    by_semantics = {row["rerank_semantics"]: row for row in rows}
    assert by_semantics["predicted_direct_no_rerank"]["selection_regime"] == "deployable_predicted_application"
    independent = by_semantics["exact_independent_abc_within_fetched_candidates"]
    full = by_semantics["full_target_restricted_teacher_oracle"]
    contained = by_semantics["contained_fetched_target_exact_h0"]
    units = len(gram)
    proxy_rank = 2
    assert json.loads(independent["selected_units"]) == [1]
    assert json.loads(full["selected_units"]) == [1]
    assert json.loads(contained["selected_units"]) == [0]
    for row in rows:
        assert json.loads(row["candidate_unit_ids"]) == [0, 1]
        assert json.loads(row["oracle_selected_units"]) == [2]
        assert row["oracle_set_gain"] == pytest.approx(rows[0]["oracle_set_gain"])
    assert independent["rerank_role"] == "primary_realistic_h0_rerank"
    assert independent["rerank_incremental_compute_macs"] == 2 * 2 * runner.INPUTS + 6 * 2
    assert independent["rerank_incremental_bytes_read"] == (
        2 * runner.UNIT_PACKET_BYTES + runner.ACTIVATION_PAYLOAD_BYTES + 18 + 30
        + 2 * runner.Q2_GATE_UP_PARENT_CODE_BYTES_PER_UNIT
        + 2 * runner.Q2_GATE_UP_SCALE_BYTES_PER_UNIT
    )
    assert independent["rerank_activation_payload_bytes_read"] == runner.ACTIVATION_PAYLOAD_BYTES
    assert independent["rerank_q2_unit_feature_bytes_read"] == 18
    assert independent["rerank_abc_metadata_bytes_read"] == 30
    assert independent["rerank_abc_score_macs_per_unit"] == 6
    assert independent["rerank_q2_gate_up_parent_payload_required"] is True
    assert independent["rerank_q2_gate_up_parent_payload_accounted"] is True
    assert independent["rerank_q2_gate_up_parent_code_bytes_read"] == 2 * 1_024
    assert independent["rerank_q2_gate_up_scale_bytes_read"] == 2 * 128
    assert independent["rerank_q4_response_scale_multiplications"] == 2 * 128
    assert independent["rerank_candidate_packet_bytes_read"] == 2 * 1_536
    assert independent["rerank_q2_gate_up_scale_payload_already_read"] is False
    assert "suffix_only" in independent["rerank_candidate_packet_contents"]
    assert independent["rerank_abc_score_mac_convention"] == runner.ABC_SCORE_MAC_CONVENTION
    assert independent["rerank_score_semantics"] == runner.INDEPENDENT_ABC_SCORE_SEMANTICS
    assert full["selection_regime"] == "teacher_oracle_not_deployable"
    assert full["rerank_q4_response_units"] == units
    assert full["rerank_q4_response_compute_macs"] == 2 * units * runner.INPUTS
    assert full["rerank_correction_workspace_bytes"] == units * runner.INPUTS * 2
    assert full["rerank_interaction_gram_units"] == units
    assert full["rerank_incremental_compute_macs"] == (
        2 * units * runner.INPUTS
        + 2 * units * runner.INPUTS
        + units * units * runner.INPUTS
        + units * runner.INPUTS * proxy_rank
        + units * units * proxy_rank
    )
    assert full["rerank_incremental_bytes_read"] == (
        units * runner.UNIT_PACKET_BYTES
        + runner.ACTIVATION_PAYLOAD_BYTES
        + 3 * units * 2
        + units * runner.Q2_GATE_UP_PARENT_CODE_BYTES_PER_UNIT
        + units * runner.Q2_GATE_UP_SCALE_BYTES_PER_UNIT
        + units * runner.INPUTS * 2
        + units * units * 4
        + runner.INPUTS * proxy_rank * 4
    )
    assert full["rerank_accounting_is_lower_bound"] is True
    assert full["rerank_accounting_bound"] == runner.INTERACTION_RERANK_ACCOUNTING_BOUND
    assert contained["rerank_q4_response_units"] == 2
    assert contained["rerank_correction_workspace_bytes"] == 2 * runner.INPUTS * 2
    assert contained["rerank_accounting_is_lower_bound"] is True
    assert contained["selection_regime"] == "information_deployable_but_expensive_interaction_oracle"


@pytest.mark.parametrize(
    "candidates,applied,predicted,match",
    [
        ([0, 0], 1, None, "unique"),
        ([0, 3], 1, None, "out-of-range"),
        ([0, 1, 2], 2, [0, 0], "unique"),
        ([0, 1], 1, [2], "candidate subset"),
    ],
)
def test_candidate_rerank_rejects_invalid_support_evidence(
    candidates: list[int], applied: int, predicted: list[int] | None, match: str,
) -> None:
    with pytest.raises((ValueError, RuntimeError), match=match):
        runner.candidate_rerank_records(
            np.eye(3), candidates, applied,
            independent_scores=np.ones(3),
            predicted_apply=predicted,
            activation_payload_already_read=True,
            q2_payload_already_read=True,
            abc_payload_already_read=True,
            abc_metadata_bytes=30,
            proxy_rank=2,
        )


def test_selector_score_payload_charges_exact_q2_and_abc_once() -> None:
    base = {
        "bytes_read": 100,
        "activation_payload_already_read": True,
        "q2_payload_already_read": False, "abc_payload_already_read": False,
        "activation_payload_bytes_read": runner.ACTIVATION_PAYLOAD_BYTES,
        "q2_unit_feature_bytes_read": 0, "abc_metadata_bytes_read": 0,
    }
    charged = runner._charge_selector_score_payload(
        base, units=runner.UNITS, abc_metadata_bytes=3_084,
    )
    assert charged["bytes_read"] == 100 + runner.Q2_UNIT_FEATURE_BYTES + 3_084
    assert charged["q2_unit_feature_bytes_read"] == runner.Q2_UNIT_FEATURE_BYTES
    assert charged["abc_metadata_bytes_read"] == 3_084
    assert charged["q2_payload_already_read"] and charged["abc_payload_already_read"]
    assert charged["activation_payload_bytes_read"] == runner.ACTIVATION_PAYLOAD_BYTES
    repeated = runner._charge_selector_score_payload(
        charged, units=runner.UNITS, abc_metadata_bytes=3_084,
    )
    assert repeated == charged


def test_primary_independent_abc_uses_runtime_rounding_when_ranking_flips() -> None:
    activation = np.array([0.6502975821495056, 0.5861281752586365], np.float32)
    q2_gate = np.array([
        [1.740587830543518, -0.16979125142097473],
        [2.375706672668457, -0.06615936756134033],
    ], np.float32)
    q2_up = np.array([
        [0.754574179649353, -2.3771915435791016],
        [-4.1568474769592285, 3.723717212677002],
    ], np.float32)
    q4_gate = np.array([
        [1.7577482461929321, -0.19148051738739014],
        [2.3697657585144043, 0.042647428810596466],
    ], np.float32)
    q4_up = np.array([
        [0.7853489518165588, -2.5601401329040527],
        [-4.093222618103027, 3.5527141094207764],
    ], np.float32)
    down2 = np.zeros((2, 2), np.float32)
    down4 = np.eye(2, dtype=np.float32)
    metadata = runner.unit_score_metadata(down2, down4)
    encoded, abc_bytes = runner.encode_unit_score_metadata(metadata, "fp16")

    teacher = runner.factorized_unit_outputs(
        (q2_gate, q2_up, down2), (q4_gate, q4_up, down4), activation,
    )
    teacher_score = runner.complete_unit_scores(teacher.h2, teacher.h4, encoded)
    runtime_score, runtime = runner.runtime_independent_abc_scores(
        q2_gate, q2_up, q4_gate, q4_up, activation, encoded,
    )
    assert np.lexsort((np.arange(2), -teacher_score)).tolist() == [0, 1]
    assert np.lexsort((np.arange(2), -runtime_score)).tolist() == [1, 0]
    np.testing.assert_array_equal(
        runtime["runtime_activation"], activation.astype(np.float16).astype(np.float32),
    )

    rows = runner.candidate_rerank_records(
        np.eye(2), [0, 1], 1, independent_scores=runtime_score,
        activation_payload_already_read=False,
        q2_payload_already_read=False,
        abc_payload_already_read=False,
        abc_metadata_bytes=abc_bytes,
        proxy_rank=1,
    )
    independent = next(
        row for row in rows
        if row["rerank_semantics"] == "exact_independent_abc_within_fetched_candidates"
    )
    assert json.loads(independent["selected_units"]) == [1]
    assert independent["rerank_score_semantics"] == runner.INDEPENDENT_ABC_SCORE_SEMANTICS
    assert "full_vector_reference_only_fetched_units_charged" in (
        independent["rerank_score_semantics"]
    )


def test_candidate_total_bytes_retain_packets_without_double_charging() -> None:
    packets = 256 * runner.UNIT_PACKET_BYTES
    selector = 12_345 + packets
    assert runner.total_candidate_bytes_read(selector, 0, 256) == selector
    assert runner.total_candidate_bytes_read(selector, packets, 256) == selector
    assert runner.total_candidate_bytes_read(selector, packets + 777, 256) == selector + 777
    all_units = runner.UNITS * runner.UNIT_PACKET_BYTES
    assert runner.total_candidate_bytes_read(selector, all_units + 999, 256) == (
        selector + (runner.UNITS - 256) * runner.UNIT_PACKET_BYTES + 999
    )


def test_one_bpw_fetch_accounting_charges_all_256_packets() -> None:
    config = {
        "experts_per_layer": 256,
        "suffix_bpw_per_complete_representation": 2.0,
        "reference_bpw": 4.25,
    }
    fields = runner.accounting_fields(
        config, expert_metadata_bytes=32_768, layer_shared_bytes=65_536,
        abc_bytes=3_084, compute_macs=131_072, compute_additions=65_536,
        selector_bytes_read=90_000, fetched_units=256, applied_units=192,
    )
    assert fields["physical_pages"] == 768
    assert fields["physical_bytes"] == 393_216
    assert fields["physical_bpw"] == pytest.approx(1.0)
    assert fields["logical_bpw"] == pytest.approx(0.75)
    assert fields["page_amplification"] == pytest.approx(4 / 3)
    assert fields["layer_shared_amortized_bytes_per_expert"] == 256
    assert fields["storage_multiplier"] < 5.0


def test_fit_layer_loader_checks_hash_size_and_array_count(tmp_path: Path) -> None:
    shard = tmp_path / "set_utility_fit_layer_0.npz"
    np.savez_compressed(shard, weights=np.arange(7, dtype=np.float16))
    record = {
        "shard": shard.name, "shard_sha256": runner.sha256(shard),
        "shard_bytes": shard.stat().st_size, "array_count": 1,
    }
    arrays, loaded = runner.load_fit_layer(tmp_path, {"layers": {"0": record}}, 0)
    np.testing.assert_array_equal(arrays["weights"], np.arange(7, dtype=np.float16))
    assert loaded == record
    with pytest.raises(RuntimeError, match="changed"):
        runner.load_fit_layer(tmp_path, {"layers": {"0": {**record, "shard_sha256": "0" * 64}}}, 0)


def test_empty_checkpoint_artifacts_keep_declared_schema(tmp_path: Path) -> None:
    for name, filename in runner.ARTIFACTS.items():
        path = tmp_path / filename
        runner.atomic_parquet(path, [])
        frame = runner.pd.read_parquet(path)
        assert frame.empty
        assert list(frame.columns) == runner.ARTIFACT_SCHEMAS[name]


def test_runtime_provenance_rejects_device_gpu_and_runtime_drift() -> None:
    current = runner.runtime_provenance("cpu")
    runner.require_runtime_provenance_match(current, current, "fixture")
    replacements = {
        "device": "cuda:7",
        "gpu_names_used": ["different GPU"],
        "torch_version": "changed-torch",
        "torch_cuda_runtime": "changed-cuda",
        "numpy_version": "changed-numpy",
        "python_version": "changed-python",
        "host": "different-host",
    }
    for field, value in replacements.items():
        recorded = {**current, field: value}
        with pytest.raises(RuntimeError, match=f"mismatch for {field}"):
            runner.require_runtime_provenance_match(recorded, current, "fit resume")
    missing = dict(current)
    missing.pop("torch_version")
    with pytest.raises(RuntimeError, match="lacks"):
        runner.require_runtime_provenance_match(missing, current, "validation resume")


def test_resume_discards_uncommitted_expert_rows_and_checks_provenance(tmp_path: Path) -> None:
    expected = {"run_id": "bounded", "capture_sha256": "a" * 64}
    facts = {
        **expected, "completed": False,
        "completed_work_units": ["validation:0:3"],
        "failures": [{"type": "Interrupted", "message": "fixture"}],
    }
    (tmp_path / "run_facts.json").write_text(json.dumps(facts))
    for name, filename in runner.ARTIFACTS.items():
        runner.atomic_parquet(tmp_path / filename, [
            {"layer": 0, "expert_id": 3, "marker": f"{name}:keep"},
            {"layer": 4, "expert_id": 9, "marker": f"{name}:drop"},
        ])
    state, completed, history = runner.load_checkpoint_state(tmp_path, expected)
    assert completed == ["validation:0:3"]
    assert history == facts["failures"]
    assert all(len(rows) == 1 and rows[0]["marker"].endswith(":keep") for rows in state.values())
    with pytest.raises(RuntimeError, match="resume provenance mismatch"):
        runner.load_checkpoint_state(tmp_path, {**expected, "run_id": "changed"})


def test_evaluation_plan_only_emits_exact_validation_rows() -> None:
    data = {
        "layer": np.zeros(5, np.int64),
        "split": np.array(["train", "validation", "validation", "train", "validation"]),
        "expert_ids": np.array([[3, 9], [3, 8], [9, 3], [3, 1], [4, 9]], np.int64),
    }
    config = {
        "layers": [0], "max_validation_invocations_per_expert": 2,
        "locked_exact_sampled_experts": {"0": {"3": "cold", "9": "hot"}},
    }
    plan = runner.evaluation_plan(data, config)
    by_expert = {expert: (records.tolist(), ranks.tolist()) for _, expert, _, records, ranks in plan}
    assert by_expert == {3: ([1, 2], [0, 1]), 9: ([2, 4], [0, 1])}
    data["split"][0] = "test"
    with pytest.raises(RuntimeError, match="forbidden split"):
        runner.evaluation_plan(data, config)


def test_validation_plan_audit_fails_closed_on_count_or_cell_drift() -> None:
    config = {
        "locked_exact_sampled_experts": {"0": {"3": "cold", "9": "hot"}},
        "expected_validation_unique_invocations": 4,
        "expected_validation_layer_expert_cells": 2,
    }
    plan = [
        (0, 3, "cold", np.array([1, 2]), np.array([0, 1])),
        (0, 9, "hot", np.array([2, 4]), np.array([0, 1])),
    ]
    audit = runner.audit_validation_plan(plan, config)
    assert audit["validation_scope_complete"] is True
    assert audit["observed_unique_invocations"] == 4
    assert audit["observed_layer_expert_cells"] == 2
    with pytest.raises(RuntimeError, match="validation invocations"):
        runner.audit_validation_plan(plan[:-1] + [
            (0, 9, "hot", np.array([2]), np.array([0])),
        ], config)
    with pytest.raises(RuntimeError, match="observed validation cells"):
        runner.audit_validation_plan([plan[0], (0, 8, *plan[1][2:])], config)



def test_train_row_sampler_round_robins_requests_before_second_positions() -> None:
    local = {
        "split": np.array(["train"] * 6 + ["validation"]),
        "request_id": np.array(["a", "a", "a", "b", "b", "c", "v"]),
        "position": np.array([1, 2, 3, 1, 2, 1, 1]),
    }
    rows = runner.stable_rows(local, "train", 4)
    assert rows.tolist() == [0, 3, 5, 1]
    assert set(local["request_id"][rows[:3]]) == {"a", "b", "c"}


def test_runner_pq_uses_separate_scale_aware_gate_up_banks() -> None:
    rng = np.random.default_rng(991)
    rows, width = 16, 32
    q2_gate = rng.normal(size=(rows, width)).astype(np.float32)
    q2_up = rng.normal(size=(rows, width)).astype(np.float32)
    q4_gate = q2_gate + 0.25 * rng.normal(size=(rows, width)).astype(np.float32)
    q4_up = q2_up + 0.25 * rng.normal(size=(rows, width)).astype(np.float32)
    dummy_down = np.zeros((width, rows), np.float32)
    matrices = {
        7: {
            "gate": [q2_gate, (q2_gate + q4_gate) / 2, q4_gate],
            "up": [q2_up, (q2_up + q4_up) / 2, q4_up],
            "down": [dummy_down, dummy_down, dummy_down],
        }
    }
    scales = {
        7: {
            "gate": np.ones((rows, 1), np.float32),
            "up": np.full((rows, 1), 2.0, np.float32),
        }
    }
    arrays: dict[str, np.ndarray] = {}
    entries = runner._pq_fit(
        matrices, scales, rng.normal(size=(24, width)).astype(np.float32),
        {"pq_additive_stages": [1], "pq_max_iterations": 2}, arrays,
    )
    assert len(entries) == 1
    entry = entries[0]
    assert entry["codebook_sharing"] == "separate_gate_up_layer_shared"
    assert entry["block_scale_provenance"] == "locked_resident_e8m0"
    assert "scale_squared" in entry["pq_fit_objective"]
    assert "cross_block_covariance_omitted" in entry["pq_fit_objective"]
    assert np.isfinite(entry["training_whole_response_relative_mse"])
    assert entry["pq_whole_response_diagnostic_rows"] == 24
    assert entry["new_scale_metadata_bytes"] == 0
    assert entry["resident_scale_bytes_read"] == 32
    assert entry["expert_specific_bytes"] == 16
    assert entry["layer_shared_bytes"] == 2048
    gate, up, accounting = runner._selector_prediction(
        arrays, entry, rng.normal(size=width).astype(np.float32),
    )
    assert gate.shape == up.shape == (rows,)
    assert accounting["compute_macs"] == 1024
    assert accounting["scale_multiplications"] == 32
    assert accounting["bytes_read"] >= entry["resident_scale_bytes_read"]



def test_static_unit_bias_adds_exactly_one_fp16_scalar_per_unit() -> None:
    config = {
        "experts_per_layer": 256,
        "suffix_bpw_per_complete_representation": 2.0,
        "reference_bpw": 4.25,
    }
    common = dict(
        config=config, expert_metadata_bytes=0, layer_shared_bytes=10_000,
        compute_macs=100, compute_additions=10, selector_bytes_read=20_000,
        fetched_units=256, applied_units=192,
    )
    control = runner.accounting_fields(abc_bytes=3_084, **common)
    static = runner.accounting_fields(abc_bytes=4_108, **common)
    assert static["selector_metadata_bytes_per_expert"] - control["selector_metadata_bytes_per_expert"] == 1_024
    assert static["abc_metadata_bytes"] - control["abc_metadata_bytes"] == runner.UNITS * 2


def test_stable_model_seed_depends_on_full_identity_and_repeats() -> None:
    first = runner._stable_model_seed(7, 4, "cohort", "pq1", "set", "single")
    assert first == runner._stable_model_seed(7, 4, "cohort", "pq1", "set", "single")
    assert first != runner._stable_model_seed(7, 4, "cohort", "pq1", "set", "joint_nested")


def test_hybrid_forward_best_prefix_can_reject_a_negative_terminal_action() -> None:
    from types import SimpleNamespace
    from oracle_study.neuron_selector import FactorizedUnitOutputs

    outputs = FactorizedUnitOutputs(
        y00=np.array([[0.0], [0.0]]),
        y10=np.array([[10.0], [0.0]]),
        y01=np.array([[10.0], [0.0]]),
        y11=np.array([[10.0], [-9.0]]),
        h2=np.zeros(2),
        h4=np.ones(2),
    )
    action = SimpleNamespace(
        unit=0, from_state=0, to_state=3, cumulative_pages=3,
    )
    trace = SimpleNamespace(actions=(action,), pages=3)
    best = runner._best_hybrid_forward_prefix(
        outputs, trace, proxy=np.zeros((1, 1)), beta=0.0,
    )
    assert best["actions"] == 0
    assert best["pages"] == 0
    assert best["residual_energy"] == pytest.approx(1.0)
    np.testing.assert_array_equal(best["states"], np.zeros(2, np.int64))



def test_contract_rejects_truncated_locked_hash() -> None:
    config = tiny_contract()
    config["locked_tree_sha256"] = "a" * 56
    with pytest.raises(RuntimeError, match="64 lowercase hex"):
        runner.validate_study_contract(config)



def test_hidden16_joint_pq_model_stays_under_strict_primary_compute_gate() -> None:
    model = runner.CompactCoherentUnitPredictor(
        3, pq_delta_dim=2, abc_dim=3, hidden_dim=16,
        unit_count=runner.UNITS, head_mode="joint_nested",
    )
    model_macs = model.accounting(parameter_bits=16, abc_bits=16).linear_macs_per_invocation
    pq1_macs = 2 * 64 * 16 * 32
    independent_rerank_macs = 256 * (2 * runner.INPUTS + 6)
    assert model_macs + pq1_macs + independent_rerank_macs <= 1_572_864


def test_high_rank_row_scaled_synthesis_charges_all_1024_scales() -> None:
    rank = 2
    arrays = {
        "analysis": np.zeros((rank, runner.INPUTS), np.float16),
        "gate": np.zeros((runner.UNITS, rank), np.float32),
        "up": np.zeros((runner.UNITS, rank), np.float32),
    }
    entry = {
        "selector_family": "high_rank_low_bit_linear_response",
        "rank": rank,
        "analysis_key": "analysis",
        "layer_shared_bytes": arrays["analysis"].nbytes,
        "expert_specific_bytes": arrays["gate"].nbytes + arrays["up"].nbytes,
        "projection_arrays": {
            "gate": {"decoded_key": "gate"},
            "up": {"decoded_key": "up"},
        },
    }
    _, _, accounting = runner._selector_prediction(
        arrays, entry, np.zeros(runner.INPUTS, np.float32),
    )
    assert accounting["scale_multiplications"] == 2 * runner.UNITS == 1_024


def test_every_high_rank_transductive_fit_is_nonpromotable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    train_x = np.eye(4, dtype=np.float32)
    q2 = np.zeros((3, 4), np.float32)
    q4_gate = np.arange(12, dtype=np.float32).reshape(3, 4) / 10
    q4_up = np.flip(q4_gate, axis=1).copy()
    matrices = {
        7: {
            "gate": [q2, q2, q4_gate],
            "up": [q2, q2, q4_up],
        },
    }

    def responses(x, residuals, _device):
        return (
            {expert: x @ pair[0].T for expert, pair in residuals.items()},
            {expert: x @ pair[1].T for expert, pair in residuals.items()},
        )

    monkeypatch.setattr(runner.prior, "gpu_responses", responses)
    monkeypatch.setattr(
        runner.prior, "canonical_subspace",
        lambda _responses, maximum, _device: np.eye(4, dtype=np.float32)[:maximum],
    )
    monkeypatch.setattr(
        runner.prior, "analysis_from_subspace",
        lambda _x, desired, _ridge: desired,
    )
    monkeypatch.setattr(
        runner.prior, "syntheses",
        lambda latent, response_bank, _device: {
            name: np.linalg.lstsq(latent, response, rcond=None)[0].T
            for name, response in response_bank.items()
        },
    )
    monkeypatch.setattr(
        runner.prior, "encode_array",
        lambda value, _encoding: SimpleNamespace(
            decoded=np.asarray(value, np.float32), storage_bytes=np.asarray(value).nbytes,
        ),
    )
    entries = runner._high_rank_fit(
        matrices, train_x,
        {
            "high_rank_low_bit_controls": [{"rank": 2, "encoding": "int8_per_row"}],
            "ridge_relative": 1e-4,
        },
        runner.torch.device("cpu"), {},
    )
    assert entries and all(entry["promotable"] is False for entry in entries)


def test_locked_config_and_raw_candidate_semantics_are_unambiguous() -> None:
    config_path = (
        Path(__file__).parents[1]
        / "configs/qwen36_mxfp4_set_utility_distillation.json"
    )
    config = json.loads(config_path.read_text())
    runner.validate_study_contract(config)
    assert config["expected_validation_unique_invocations"] == 69
    assert config["expected_validation_layer_expert_cells"] == 12
    assert config["expected_cross_capture_shared_request_ids"] == 0
    assert config["direct_candidate_head_supervision"] == (
        runner.JOINT_CANDIDATE_COVERAGE_SEMANTICS
    )
    source = SCRIPT.read_text()
    assert source.count('"candidate_head_supervision":') == 1
    assert "no candidate hard-set or top192 coverage loss" not in source


def test_standalone_selector_score_charge_is_emitted_and_idempotent() -> None:
    base = {
        "compute_macs": 17, "bytes_read": 100,
        "activation_payload_already_read": True,
        "q2_payload_already_read": True, "abc_payload_already_read": True,
        "activation_payload_bytes_read": runner.ACTIVATION_PAYLOAD_BYTES,
        "q2_unit_feature_bytes_read": runner.Q2_UNIT_FEATURE_BYTES,
        "abc_metadata_bytes_read": 3_084,
    }
    charged = runner._charge_selector_score_payload(
        base, units=runner.UNITS, abc_metadata_bytes=3_084,
    )
    assert charged["abc_score_units"] == runner.UNITS
    assert charged["abc_score_compute_macs"] == 3_072
    assert charged["compute_macs"] == 17 + 3_072
    assert runner._charge_selector_score_payload(
        charged, units=runner.UNITS, abc_metadata_bytes=3_084,
    ) == charged
    direct = runner._charge_selector_score_payload(
        base, units=runner.UNITS, abc_metadata_bytes=3_084, abc_score_required=False,
    )
    assert direct["abc_score_units"] == direct["abc_score_compute_macs"] == 0
    assert direct["compute_macs"] == 17
    emitted = runner.accounting_fields(
        {"experts_per_layer": 256, "suffix_bpw_per_complete_representation": 2.0, "reference_bpw": 4.25},
        expert_metadata_bytes=0, layer_shared_bytes=0, abc_bytes=3_084,
        compute_macs=charged["compute_macs"], compute_additions=0,
        selector_bytes_read=charged["bytes_read"], fetched_units=256, applied_units=192,
        abc_score_units=charged["abc_score_units"],
        abc_score_compute_macs=charged["abc_score_compute_macs"],
    )
    assert emitted["selector_abc_score_compute_macs"] == 3_072
    assert emitted["selector_abc_score_mac_convention"] == runner.ABC_SCORE_MAC_CONVENTION


def test_candidate_q2_scale_bytes_deduplicate_but_scale_work_remains() -> None:
    common = dict(
        gram=np.eye(3), candidates=[0, 1], applied=1, independent_scores=np.ones(3),
        activation_payload_already_read=True, q2_payload_already_read=True,
        abc_payload_already_read=True, abc_metadata_bytes=30, proxy_rank=1,
    )
    charged_rows = runner.candidate_rerank_records(
        **common, q2_gate_up_scale_payload_already_read=False,
    )
    deduped_rows = runner.candidate_rerank_records(
        **common, q2_gate_up_scale_payload_already_read=True,
    )
    charged = next(row for row in charged_rows if row["rerank_semantics"].startswith("exact_independent"))
    deduped = next(row for row in deduped_rows if row["rerank_semantics"].startswith("exact_independent"))
    assert charged["rerank_candidate_packet_bytes_read"] == 2 * runner.UNIT_PACKET_BYTES
    assert deduped["rerank_candidate_packet_bytes_read"] == charged["rerank_candidate_packet_bytes_read"]
    assert charged["rerank_q2_gate_up_parent_code_bytes_read"] == 2 * 1_024
    assert deduped["rerank_q2_gate_up_parent_code_bytes_read"] == 2 * 1_024
    assert charged["rerank_q2_gate_up_scale_bytes_read"] == 2 * 128
    assert deduped["rerank_q2_gate_up_scale_bytes_read"] == 0
    assert charged["rerank_incremental_bytes_read"] - deduped["rerank_incremental_bytes_read"] == 2 * 128
    assert charged["rerank_q4_response_scale_multiplications"] == 2 * 128
    assert deduped["rerank_q4_response_scale_multiplications"] == 2 * 128
    predicted = next(row for row in deduped_rows if row["rerank_semantics"] == "predicted_direct_no_rerank")
    assert predicted["rerank_q2_gate_up_parent_payload_required"] is False
    assert predicted["rerank_q2_gate_up_parent_payload_accounted"] is True
    contained = next(row for row in charged_rows if row["rerank_semantics"].startswith("contained"))
    assert contained["rerank_resident_q2_down_payload_required"] is True
    assert contained["rerank_resident_q2_down_payload_accounted"] is False
    assert "Q2-down_parent-code/scale_payload" in contained["rerank_accounting_bound"]


def test_pq_response_feature_contract_rounds_in_fit_and_runtime() -> None:
    raw = np.array([[1.0003, -0.3333], [123.456, 0.00012345]], np.float32)
    rounded = runner._fp16_round_widen_pq_response(raw)
    expected = raw.astype(np.float16).astype(np.float32)
    assert rounded.dtype == np.float32
    np.testing.assert_array_equal(rounded, expected)
    assert not np.array_equal(rounded, raw)
    source = SCRIPT.read_text()
    assert "pq_raw = _fp16_round_widen_pq_response(" in source
    assert "pq_value = _fp16_round_widen_pq_response(" in source
    assert source.count('"pq_runtime_response_precision": (') == 1
    assert '"fp16_round_then_fp32_reference" if use_pq else "not_applicable"' in source


def test_template_cohort_counts_preserve_explicit_zero_routed_cell() -> None:
    routed, exact_synthetic, mixed_synthetic = runner.FIT_COHORTS
    rows = [
        {"expert_id": 3, "cohort": routed},
        {"expert_id": 3, "cohort": exact_synthetic},
        {"expert_id": 3, "cohort": mixed_synthetic},
        {"expert_id": 9, "cohort": exact_synthetic},
        {"expert_id": 9, "cohort": mixed_synthetic},
    ]
    counts = runner.template_cohort_example_counts_by_expert(rows, [9, 3])
    assert list(counts) == ["3", "9"]
    assert counts["3"][routed] == 1
    assert counts["9"][routed] == 0
    assert counts["3"][exact_synthetic] > 0 and counts["9"][exact_synthetic] > 0
    assert counts["3"][mixed_synthetic] > 0 and counts["9"][mixed_synthetic] > 0


def test_primary_compute_gate_is_strict() -> None:
    assert runner._strict_primary_compute_gate_pass(99, 100) is True
    assert runner._strict_primary_compute_gate_pass(100, 100) is False
    assert runner._strict_primary_compute_gate_pass(101, 100) is False
