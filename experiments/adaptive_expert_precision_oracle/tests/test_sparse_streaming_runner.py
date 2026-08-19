"""Focused fail-closed tests for the sparse-streaming study runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_sparse_streaming_study as runner
from oracle_study.mxfp4_selective import BitAction

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config() -> dict:
    return {
        "run_id": "synthetic-run",
        "layers": [4],
        "page_size_bytes": 512,
        "reference_bpw": 4.25,
        "suffix_bpw_per_complete_representation": 2.0,
        "activation_shortlist_unit": "input_coordinates",
        "coordinate_score_aggregation": "sum_stage_energy",
        "shortlist_physical_layout": "canonical_paired_planes_four_coordinates_per_projection_page",
        "tile_pilot_shape": [32, 32],
        "pilot_validation_invocations_per_expert": 16,
        "pilot_test_invocations_per_expert": 1,
        "max_test_invocations_per_expert": 24,
        "pilot_expert_per_layer": {"4": 154},
        "locked_exact_sampled_experts_from_pr4": {
            "4": {"17": "cold", "110": "median", "154": "hot"},
        },
        "locked_cross_sampled_experts_from_pr6": {
            "4": {"156": "cold", "176": "median", "134": "hot"},
        },
        "promotion_policy": {
            "promotion_min_validation_invocations_per_layer": 4,
            "promotion_min_validation_requests_per_layer": 2,
            "unit_go_at_1bpw_median": 0.90,
            "unit_go_at_1bpw_p10": 0.85,
            "shortlist_min_overfetch": 1.0,
            "shortlist_min_exact_gain_retention_median": 0.95,
            "shortlist_min_exact_gain_retention_p10": 0.90,
            "shortlist_max_candidates": 512,
            "shortlist_max_overfetch": 1.25,
            "shortlist_overfetch_statistic": "p90",
            "tile_min_recovery_point_gain": 0.03,
            "tile_min_page_reduction_at_matched_recovery": 0.25,
        },
        "reference": {
            "revision": "locked-revision",
            "config_sha256": "checkpoint-config",
            "index_sha256": "checkpoint-index",
        },
        "locked_capture_sha256": {
            "exact_checkpoint": "fresh-capture",
            "cross_reference": "cross-capture",
        },
        "locked_tree_sha256": "tree-hash",
    }


def _identity(split: str = "validation") -> dict:
    return {
        "capture_source": "exact_checkpoint",
        "evaluation_split": split,
        "request_id": f"{split}-request",
        "position": 3,
        "layer": 4,
        "expert_id": 154,
    }


def test_capture_hash_and_request_split_checks_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = tmp_path / "capture.npz"
    capture.write_bytes(b"capture")
    trees = tmp_path / "trees.json"
    trees.write_text("{}\n")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n")
    (checkpoint / "model.safetensors.index.json").write_text("{}\n")
    config_path = tmp_path / "study.json"
    config_path.write_text("{}\n")
    config = _config()
    config["locked_capture_sha256"]["exact_checkpoint"] = _sha(capture)
    config["locked_tree_sha256"] = _sha(trees)
    config["reference"]["config_sha256"] = _sha(checkpoint / "config.json")
    config["reference"]["index_sha256"] = _sha(checkpoint / "model.safetensors.index.json")
    monkeypatch.setattr(runner, "audit_checkpoint", lambda *_: {"passed": True})

    audit, hashes = runner.verify_locked_inputs(
        config, config_path, capture, checkpoint, trees, "exact_checkpoint",
    )
    assert audit["passed"]
    assert hashes["capture_sha256"] == _sha(capture)
    capture.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="capture changed"):
        runner.verify_locked_inputs(
            config, config_path, capture, checkpoint, trees, "exact_checkpoint",
        )

    split_data = {
        "split": np.asarray(["train", "validation", "test"]),
        "request_id": np.asarray(["train-r", "validation-r", "test-r"]),
    }
    assert runner.verify_request_separation(split_data) == {
        "train": 1, "validation": 1, "test": 1,
    }
    split_data["request_id"][-1] = "train-r"
    with pytest.raises(RuntimeError, match="split leakage"):
        runner.verify_request_separation(split_data)
    with pytest.raises(RuntimeError, match="split labels changed"):
        runner.verify_request_separation({
            "split": np.asarray(["train", "validation", "shadow"]),
            "request_id": np.asarray(["a", "b", "c"]),
        })


def test_coordinate_k_semantics_page_closure_and_skip_repricing() -> None:
    coordinates = 16
    score = np.zeros(coordinates)
    score[[0, 8]] = [10.0, 9.0]
    chosen, shortlisted = runner.coordinate_shortlist(score, 2, coordinates)
    np.testing.assert_array_equal(chosen, [0, 8])
    np.testing.assert_array_equal(shortlisted, [0, 8, 16, 24])
    page_map = runner.canonical_coordinate_page_map(coordinates)
    expanded, pages = runner.close_coordinate_shortlist_to_pages(
        shortlisted, coordinates, page_map,
    )
    assert pages == {0, 2}
    assert len(expanded) == 16
    assert set(expanded) == (
        set(range(4)) | set(range(8, 12))
        | set(range(16, 20)) | set(range(24, 28))
    )

    order = [BitAction(0, 1, 3.0), BitAction(4, 1, 2.0), BitAction(0, 2, 1.0)]
    selected, paid = runner.path_under_page_budget(order, coordinates, 1, page_map)
    assert [(a.coordinate, a.stage) for a in selected] == [(0, 1), (0, 2)]
    assert paid == {0}


def test_exact_matched_page_control_reruns_residual_instead_of_filtering() -> None:
    # A PSD action Gram where the unrestricted path visits page 0 between
    # the two nested actions on page 1.  Post-hoc filtering incorrectly keeps
    # page-1 stage 2 with a marginal computed after the omitted page-0 action.
    vectors = np.asarray([
        [0.42776996, -0.57083756],
        [2.6544607, -1.6085450],
        [0.6617157, -0.14342594],
        [-0.3545064, 1.0663588],
    ], dtype=np.float32)
    gram = vectors @ vectors.T
    activation = np.asarray([-1.8179220, -0.9846762], dtype=np.float32)
    coordinates = 2
    pages = {
        stage * coordinates + coordinate: (coordinate,)
        for stage in range(2) for coordinate in range(coordinates)
    }
    unrestricted = runner.exact_marginal_fixed_greedy_from_gram(gram, activation)
    filtered, _ = runner.path_under_page_budget(
        unrestricted, coordinates, 1, pages,
    )
    constrained, paid = runner.residual_consistent_exact_under_page_budget(
        gram, activation, coordinates, 1, pages,
    )

    def ids(actions: list[BitAction]) -> list[int]:
        return [
            (int(action.stage) - 1) * coordinates + int(action.coordinate)
            for action in actions
        ]

    assert ids(filtered) == [1, 3]
    assert ids(constrained) == [1]
    assert paid == {1}
    contributions = vectors * np.concatenate((activation, activation))[:, None]
    target = contributions.sum(axis=0)

    def realized_recovery(actions: list[BitAction]) -> float:
        selected = ids(actions)
        approximation = contributions[selected].sum(axis=0) if selected else np.zeros_like(target)
        return 1.0 - float(np.sum((target - approximation) ** 2) / np.sum(target ** 2))

    assert realized_recovery(constrained) > realized_recovery(filtered)


def test_physical_bpw_page_and_metadata_arithmetic() -> None:
    assert runner.physical_page_budget(1.0) == 768
    assert runner.projection_page_budget(1.0) == 256
    config = _config()
    fields = runner.storage_fields(config, metadata_bytes=1028)
    expected_metadata_bpw = 1028 * 8 / runner.EXPERT_WEIGHTS
    assert fields["selector_metadata_bpw"] == pytest.approx(expected_metadata_bpw)
    assert fields["storage_multiplier"] == pytest.approx(
        (4.25 + 2.0 + expected_metadata_bpw) / 4.25,
    )
    standard = runner.standard_frontier_fields(
        physical_budget_bpw=1.0,
        pages=768,
        logical_actions=512,
        recovery_value=0.9,
        runtime_seconds=0.001,
        selector_bytes_read=4096,
        multiplier=float(fields["storage_multiplier"]),
        logical_payload_bytes=512 * 768,
    )
    assert standard["physical_bytes"] == 768 * 512
    assert standard["page_amplification"] == 1.0


def test_pilot_record_plan_keeps_validation_and_test_separate() -> None:
    config = _config()
    data = {
        "split": np.asarray(["validation"] * 4 + ["test"] * 2),
        "request_id": np.asarray(["v0", "v1", "v1", "v2", "t0", "t1"]),
        "expert_ids": np.asarray([
            [154, 1], [110, 2], [110, 3], [110, 4], [154, 5], [154, 6],
        ]),
    }
    validation = runner.layer_record_plan(
        data, 4, "validation", "pilot", "exact_checkpoint", config,
    )
    validation_records = np.concatenate([records for _, _, records, _ in validation])
    assert len(validation_records) == 4
    assert set(data["split"][validation_records]) == {"validation"}
    assert len(set(data["request_id"][validation_records])) >= 2

    sanity = runner.layer_record_plan(
        data, 4, "test", "pilot", "exact_checkpoint", config,
    )
    sanity_records = np.concatenate([records for _, _, records, _ in sanity])
    assert len(sanity_records) == 1
    assert set(data["split"][sanity_records]) == {"test"}
    assert set(validation_records).isdisjoint(set(sanity_records))

    # Cross-reference strata are a frozen PR6 cohort and do not depend on
    # any counts or values in the current capture rows.
    assert runner.strata_for_layer(data, 4, "cross_reference", config) == {
        156: "cold", 176: "median", 134: "hot",
    }


def test_resume_provenance_binds_mode_source_and_promotion_hash() -> None:
    hashes = {"config_file_sha256": "config", "capture_sha256": "capture"}
    existing = {
        "config_sha256": "config",
        "input_hashes": hashes,
        "run_mode": "full",
        "source": "exact_checkpoint",
        "validation_promotions_sha256": "promotion",
    }
    runner.validate_resume_provenance(
        existing, hashes, mode="full", source="exact_checkpoint",
        promotion_sha256="promotion",
    )
    for changed in (
        {"mode": "pilot"},
        {"source": "cross_reference"},
        {"promotion_sha256": "different"},
    ):
        arguments = {
            "mode": changed.get("mode", "full"),
            "source": changed.get("source", "exact_checkpoint"),
            "promotion_sha256": changed.get("promotion_sha256", "promotion"),
        }
        with pytest.raises(RuntimeError, match="resume checkpoint provenance"):
            runner.validate_resume_provenance(existing, hashes, **arguments)


def test_analyzer_promotions_are_validation_only_and_fail_closed() -> None:
    config = _config()
    hashes = {
        "config_file_sha256": "study-config",
        "config_sha256": "checkpoint-config",
        "index_sha256": "checkpoint-index",
        "tree_sha256": "tree-hash",
    }
    promoted_neuron = {
        "action_family": "neuron_major",
        "representation": "direct_q2_q4_complete_unit_packet_3pages",
        "selector": "q2_hidden_magnitude",
        "selection_category": "h0_proxy",
        "selection_regime": "h0_resident_proxy_late",
        "objective": "exact_sequential_complete_expert_qenergy",
        "n": 4,
        "median": 0.95,
        "p10": 0.90,
        "passes_median": True,
        "passes_p10": True,
    }
    payload = {
        "schema_version": 1,
        "generated_by": "oracle_study.sparse_streaming_analysis.choose_validation_promotions",
        "config_sha256": "study-config",
        "provenance": {
            "config_sha256": "study-config",
            "reference_revision": "locked-revision",
            "checkpoint_config_sha256": "checkpoint-config",
            "checkpoint_index_sha256": "checkpoint-index",
            "tree_sha256": "tree-hash",
            "capture_sha256": config["locked_capture_sha256"],
        },
        "validation_evidence_sha256": {
            "neuron_major": "a" * 64,
            "activation_shortlist": "b" * 64,
            "tile_streaming": "c" * 64,
        },
        "selection_split": "validation",
        "selection_capture_source": "exact_checkpoint",
        "test_rows_consulted_for_selection": False,
        "minimum_validation_coverage": {
            "layers": [4], "invocations_per_layer": 4, "distinct_requests_per_layer": 2,
        },
        "validation_coverage": [{
            "frontier": "neuron_major",
            **{
                name: promoted_neuron[name]
                for name in (
                    "action_family", "representation", "selector", "selection_category",
                    "selection_regime", "objective",
                )
            },
            "layer": 4,
            "validation_invocations": 4,
            "validation_requests": 2,
        }],
        "neuron_major": {
            "status": "promote", "all_validation_candidates": [promoted_neuron],
            "promoted": [promoted_neuron],
        },
        "activation_shortlist": {
            "status": "stop", "all_validation_candidates": [], "promoted": [],
        },
        "tile_streaming": {
            "status": "stop", "all_validation_candidates": [], "promoted": [],
        },
    }
    plan = runner.validate_and_parse_promotions(
        payload, config, hashes, "exact_checkpoint",
    )
    assert plan["neuron_major"]["enabled"]
    assert plan["neuron_major"]["proxy_selectors"] == ["q2_hidden_magnitude"]
    assert not plan["activation_shortlist"]["enabled"]
    payload["neuron_major"]["all_validation_candidates"][0]["passes_p10"] = "false"
    with pytest.raises(RuntimeError, match="gate passes_p10 disagrees"):
        runner.validate_and_parse_promotions(payload, config, hashes, "exact_checkpoint")
    payload["neuron_major"]["all_validation_candidates"][0]["passes_p10"] = True
    payload["neuron_major"]["all_validation_candidates"][0]["p10"] = 0.0
    with pytest.raises(RuntimeError, match="gate passes_p10 disagrees"):
        runner.validate_and_parse_promotions(payload, config, hashes, "exact_checkpoint")
    payload["neuron_major"]["all_validation_candidates"][0]["p10"] = 0.90
    payload["validation_evidence_sha256"]["neuron_major"] = "z" * 64
    with pytest.raises(RuntimeError, match="evidence hashes"):
        runner.validate_and_parse_promotions(payload, config, hashes, "exact_checkpoint")
    payload["validation_evidence_sha256"]["neuron_major"] = "a" * 64
    payload["test_rows_consulted_for_selection"] = True
    with pytest.raises(RuntimeError, match="test_rows_consulted"):
        runner.validate_and_parse_promotions(payload, config, hashes, "exact_checkpoint")


def test_all_promotion_family_gates_are_recomputed_from_aggregate_metrics() -> None:
    policy = _config()["promotion_policy"]
    shortlist = {
        "median": 0.96,
        "p10": 0.91,
        "shortlist_size": 512,
        "p10_candidate_overfetch": 1.0,
        "p90_candidate_overfetch": 1.25,
        "passes_median": True,
        "passes_p10": True,
        "passes_candidates": True,
        "passes_min_overfetch": True,
        "passes_max_overfetch": True,
        "passes_overfetch": True,
    }
    runner._validate_promoted_candidate_gates("activation_shortlist", shortlist, policy)
    shortlist["p90_candidate_overfetch"] = 1.26
    with pytest.raises(RuntimeError, match="passes_max_overfetch disagrees"):
        runner._validate_promoted_candidate_gates("activation_shortlist", shortlist, policy)

    tile = {
        "recovery_point_gain_vs_locked_pr6_h0_hybrid": 0.031,
        "page_reduction_at_matched_recovery": None,
        "passes_recovery_gain": True,
        "passes_page_reduction": False,
    }
    runner._validate_promoted_candidate_gates("tile_streaming", tile, policy)
    tile["recovery_point_gain_vs_locked_pr6_h0_hybrid"] = 0.01
    with pytest.raises(RuntimeError, match="passes_recovery_gain disagrees"):
        runner._validate_promoted_candidate_gates("tile_streaming", tile, policy)


def test_prevalidation_failure_preserves_existing_run_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config(), sort_keys=True) + "\n")
    output = tmp_path / "existing-run"
    output.mkdir()
    facts_path = output / "run_facts.json"
    original = {"sentinel": "existing transaction journal", "completed": True}
    facts_path.write_text(json.dumps(original, sort_keys=True) + "\n")
    monkeypatch.setattr(
        runner,
        "verify_locked_inputs",
        lambda *args: ({}, {
            "config_file_sha256": "config",
            "capture_sha256": "capture",
            "tree_sha256": "tree",
            "config_sha256": "checkpoint-config",
            "index_sha256": "checkpoint-index",
        }),
    )
    monkeypatch.setattr(sys, "argv", [
        "run_sparse_streaming_study.py",
        "--config", str(config_path),
        "--captures", str(tmp_path / "capture.npz"),
        "--checkpoint", str(tmp_path / "checkpoint"),
        "--trees", str(tmp_path / "trees.json"),
        "--output", str(output),
        "--source", "exact_checkpoint",
        "--mode", "full",
        "--device", "cpu",
    ])
    with pytest.raises(RuntimeError, match="full mode requires analyzer-generated"):
        runner.main()
    assert json.loads(facts_path.read_text()) == original


def test_best_utility_prefix_keeps_profitable_nested_pair() -> None:
    actions = (
        BitAction(0, 1, -1.0),
        BitAction(0, 2, 3.0),
        BitAction(1, 1, -4.0),
    )
    selected = runner.best_utility_prefix(actions)
    assert [(action.coordinate, action.stage) for action in selected] == [(0, 1), (0, 2)]
    assert runner.best_utility_prefix((BitAction(0, 1, -1.0),)) == ()


def test_baseline_call_concentration_and_atomic_output_schemas(tmp_path: Path) -> None:
    rng = np.random.default_rng(19)
    q2 = [rng.normal(size=(4, 4)), rng.normal(size=(4, 4)), rng.normal(size=(4, 4))]
    q4 = [value + 0.05 * rng.normal(size=value.shape) for value in q2]
    trace = runner.input_coordinate_baseline_trace(
        q2[0], q2[1], q2[2], q4[0], q4[1], q4[2],
        rng.normal(size=4), [0, 6], np.eye(4), 0.2, torch.device("cpu"),
    )
    assert 6 in trace.snapshots
    assert len(trace.order) == 6

    concentration = runner.activation_concentration_rows(
        np.asarray([4.0, 3.0, 0.0, 0.0]), _identity(),
    )
    assert len(concentration) == len(runner.ACTIVATION_CONCENTRATION_TOP_K)
    assert concentration[0]["top_k_energy_fraction"] == 1.0
    assert concentration[0]["exact_zero_fraction"] == 0.5
    assert concentration[0]["coordinates_for_90pct_energy"] == 2

    state = {name: [] for name in runner.ARTIFACT_FILES}
    state["activation_concentration"] = concentration
    facts = {
        "run_id": "synthetic-run",
        "run_mode": "pilot",
        "config_sha256": "study-config",
        "reference": _config()["reference"],
        "locked_tree_sha256": "tree-hash",
        "locked_capture_sha256": _config()["locked_capture_sha256"],
        "page_size_bytes": 512,
        "codec_locked": True,
        "request_separation_verified": True,
        "completed": False,
        "expected_unique_invocations": 1,
        "observed_unique_invocations": 0,
    }
    runner.flush_checkpoint(tmp_path, state, [], facts)
    for name, filename in runner.ARTIFACT_FILES.items():
        path = tmp_path / filename
        assert path.exists()
        frame = pd.read_parquet(path)
        assert set(runner.ARTIFACT_SCHEMAS[name]).issubset(frame.columns)
    saved_facts = json.loads((tmp_path / "run_facts.json").read_text())
    for key in (
        "run_id", "run_mode", "config_sha256", "reference", "locked_tree_sha256",
        "locked_capture_sha256", "page_size_bytes", "codec_locked",
        "request_separation_verified", "expected_unique_invocations",
    ):
        assert key in saved_facts


def test_resume_drops_rows_outside_committed_expert_transactions(tmp_path: Path) -> None:
    committed = {
        **_identity("validation"),
        "top_k": 64,
        "top_k_energy_fraction": 0.8,
        "exact_zero_fraction": 0.0,
        "coordinates_for_90pct_energy": 10,
    }
    partial = {
        **committed,
        "evaluation_split": "test",
        "request_id": "partial-test-request",
        "position": 4,
    }
    pd.DataFrame([committed, partial]).to_parquet(
        tmp_path / runner.ARTIFACT_FILES["activation_concentration"], index=False,
    )
    state, cache, discarded = runner.output_state(
        tmp_path, ["validation:4:154"],
    )
    assert cache == []
    assert len(state["activation_concentration"]) == 1
    assert state["activation_concentration"][0]["evaluation_split"] == "validation"
    assert discarded["activation_concentration"] == 1
