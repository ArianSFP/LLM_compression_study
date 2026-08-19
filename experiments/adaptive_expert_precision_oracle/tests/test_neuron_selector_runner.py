from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_neuron_selector_distillation.py"
SPEC = importlib.util.spec_from_file_location("run_neuron_selector_distillation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def tiny_config() -> dict:
    return {
        "max_validation_invocations_per_expert": 2,
        "max_test_invocations_per_expert": 3,
        "layers": [0],
        "locked_exact_sampled_experts": {"0": {"2": "cold", "5": "hot"}},
        "locked_cross_sampled_experts": {"0": {"7": "cold"}},
        "reference_bpw": 4.25,
        "suffix_bpw_per_complete_representation": 2.0,
    }


def test_evaluation_plan_covers_every_locked_expert_occurrence() -> None:
    data = {
        "layer": np.zeros(4, np.int64),
        "split": np.array(["validation", "validation", "validation", "test"]),
        "expert_ids": np.array([[2, 5], [5, 2], [9, 5], [2, 1]], np.int64),
    }
    plan = runner.evaluation_plan(data, tiny_config(), "exact_checkpoint", "validation")
    by_expert = {expert: (records.tolist(), ranks.tolist()) for _, expert, _, records, ranks in plan}
    assert by_expert == {2: ([0, 1], [0, 1]), 5: ([0, 1], [1, 0])}


def test_frontier_accounting_distinguishes_fetched_and_applied_bytes() -> None:
    fields = runner.frontier_fields(
        tiny_config(), pages=768, logical_actions=192,
        logical_bytes=192 * runner.UNIT_PACKET_BYTES,
        recovery=0.94, runtime_seconds=0.002,
        selector_bytes_read=1234, metadata_bytes=130_000,
    )
    assert fields["physical_bpw"] == pytest.approx(1.0)
    assert fields["logical_bpw"] == pytest.approx(0.75)
    assert fields["page_amplification"] == pytest.approx(4 / 3)
    assert fields["selector_metadata_bpw"] == pytest.approx(8 * 130_000 / runner.EXPERT_WEIGHTS)
    assert fields["storage_multiplier"] < 1.6


def test_model_arrays_support_shared_and_per_expert_analysis() -> None:
    shared_entry = {
        "config_id": "shared", "experts": [2, 5],
        "per_expert_analysis": False, "shared_analysis": True,
    }
    bundle = {
        "shared__bg": np.arange(12, dtype=np.float16).reshape(3, 4),
        "shared__ag": np.zeros((2, 6, 3), np.float16),
        "shared__au": np.ones((2, 6, 3), np.float16),
    }
    bg, bu, ag, au = runner.model_arrays(bundle, shared_entry, 5)
    assert np.array_equal(bg, bu)
    assert ag.shape == au.shape == (6, 3)
    per_entry = {
        "config_id": "per", "experts": [2, 5],
        "per_expert_analysis": True, "shared_analysis": False,
    }
    bundle.update({
        "per__bg": np.arange(24, dtype=np.float16).reshape(2, 3, 4),
        "per__ag": np.zeros((2, 6, 3), np.float16),
        "per__au": np.ones((2, 6, 3), np.float16),
    })
    bg, bu, _, _ = runner.model_arrays(bundle, per_entry, 5)
    assert np.array_equal(bg, bundle["per__bg"][1])
    assert np.array_equal(bg, bu)


def test_promotion_parser_fails_closed_on_test_tuning(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n")
    value = {
        "schema_version": 1,
        "selection_split": "validation",
        "test_rows_consulted_for_selection": True,
        "config_sha256": runner.sha256(config_path),
        "fit_bundle_sha256": "a" * 64,
    }
    path = tmp_path / "promotions.json"
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="test tuning"):
        runner.promotion_config_ids(path, {"_config_path": str(config_path)}, "a" * 64)


def test_pr7_comparator_hash_verification_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("evidence\n")
    monkeypatch.setattr(runner, "EXPERIMENT", tmp_path)
    with pytest.raises(RuntimeError, match="changed"):
        runner.verify_pr7_comparators({
            "pr7_comparators": {"x": {"path": "artifact.json", "sha256": "0" * 64}}
        })


def test_checkpoint_resume_keeps_only_committed_expert_transactions(tmp_path: Path) -> None:
    expected = {"run_id": "r", "capture_sha256": "a" * 64}
    facts = {
        **expected,
        "completed": False,
        "completed_work_units": ["validation:0:2"],
        "failures": [{"type": "Interrupted", "message": "fixture"}],
    }
    (tmp_path / "run_facts.json").write_text(json.dumps(facts))
    for name, filename in runner.ARTIFACTS.items():
        rows = [
            {"evaluation_split": "validation", "layer": 0, "expert_id": 2, "marker": f"{name}-keep"},
            {"evaluation_split": "validation", "layer": 4, "expert_id": 9, "marker": f"{name}-drop"},
        ]
        runner.atomic_parquet(tmp_path / filename, rows)
    state, completed, history = runner.load_checkpoint_state(tmp_path, expected, "validation")
    assert completed == ["validation:0:2"]
    assert history == facts["failures"]
    for rows in state.values():
        assert len(rows) == 1
        assert rows[0]["marker"].endswith("-keep")


def test_checkpoint_resume_rejects_provenance_mismatch(tmp_path: Path) -> None:
    (tmp_path / "run_facts.json").write_text(json.dumps({
        "run_id": "old", "completed": False, "completed_work_units": [], "failures": [],
    }))
    with pytest.raises(RuntimeError, match="resume provenance mismatch"):
        runner.load_checkpoint_state(tmp_path, {"run_id": "new"}, "validation")


def test_empty_artifacts_retain_minimum_schema(tmp_path: Path) -> None:
    for name, filename in runner.ARTIFACTS.items():
        path = tmp_path / filename
        runner.atomic_parquet(path, [])
        frame = runner.pd.read_parquet(path)
        assert frame.empty
        assert set(runner.ARTIFACT_SCHEMAS[name]).issubset(frame.columns)


def test_evaluate_invocation_emits_factorized_predictor_and_fetched_accounting() -> None:
    rng = np.random.default_rng(28)
    units, inputs, outputs, rank = 4, 5, 6, 2
    q2 = {
        "gate": rng.normal(size=(units, inputs)),
        "up": rng.normal(size=(units, inputs)),
        "down": rng.normal(size=(outputs, units)),
    }
    q4 = {name: value + 0.1 * rng.normal(size=value.shape) for name, value in q2.items()}
    matrices = {name: [q2[name], (q2[name] + q4[name]) / 2.0, q4[name]] for name in q2}
    entry = {
        "config_id": "tiny", "layer": 0, "training_cohort": "train",
        "basis_variant": "layer_shared_joint_gate_up", "rank": rank,
        "synthesis_encoding": "fp16", "experts": [7],
        "shared_analysis": True, "per_expert_analysis": False,
        "analysis_bytes_per_expert": 0, "layer_analysis_bytes": 2 * rank * inputs,
        "expert_synthesis_bytes": 2 * 2 * units * rank,
    }
    bundle = {
        "tiny__bg": rng.normal(size=(rank, inputs)).astype(np.float16),
        "tiny__ag": rng.normal(size=(1, units, rank)).astype(np.float16),
        "tiny__au": rng.normal(size=(1, units, rank)).astype(np.float16),
    }
    config_value = {
        "coherent_unit_counts": [2],
        "factorized_physical_budgets_bpw": [6 / runner.PAGES_PER_BPW],
        "candidate_applied_units": 2,
        "candidate_unit_counts": [2, 3],
        "reference_bpw": 4.25,
        "suffix_bpw_per_complete_representation": 2.0,
    }
    rows = runner.evaluate_invocation(
        matrices=matrices, activation=rng.normal(size=inputs),
        proxy=rng.normal(size=(outputs, 2)), beta=0.2,
        metadata={"expert_id": 7}, config=config_value,
        device=runner.torch.device("cpu"), entries=[entry], bundle=bundle,
    )
    assert len(rows["factorized_neuron_frontier"]) == 2
    assert len(rows["response_predictor_frontier"]) == 1
    assert len(rows["candidate_rerank_frontier"]) == 2
    assert len(rows["response_prediction_diagnostics"]) == 1
    candidate = rows["candidate_rerank_frontier"][1]
    assert candidate["physical_pages"] == 9
    assert candidate["logical_bytes"] == 2 * runner.UNIT_PACKET_BYTES
    assert candidate["page_amplification"] == pytest.approx(1.5)
    assert np.isfinite(candidate["recovery"])


def test_layer_bundle_loader_verifies_hash_size_count_and_layer(tmp_path: Path) -> None:
    path = tmp_path / "response_fit_layer_0.npz"
    np.savez_compressed(path, l0__weights=np.arange(6, dtype=np.float16))
    record = {
        "path": path.name,
        "sha256": runner.sha256(path),
        "bytes": path.stat().st_size,
        "array_count": 1,
    }
    values = runner.load_bundle_layer(tmp_path, {"layers": {"0": record}}, 0)
    np.testing.assert_array_equal(values["l0__weights"], np.arange(6, dtype=np.float16))
    bad = {"layers": {"0": {**record, "sha256": "0" * 64}}}
    with pytest.raises(RuntimeError, match="changed"):
        runner.load_bundle_layer(tmp_path, bad, 0)
    foreign = tmp_path / "foreign.npz"
    np.savez_compressed(foreign, l4__weights=np.ones(2, dtype=np.float16))
    foreign_record = {
        "path": foreign.name, "sha256": runner.sha256(foreign),
        "bytes": foreign.stat().st_size, "array_count": 1,
    }
    with pytest.raises(RuntimeError, match="foreign layer"):
        runner.load_bundle_layer(tmp_path, {"layers": {"0": foreign_record}}, 0)


def test_gpu_canonical_subspace_matches_reference_projection() -> None:
    rng = np.random.default_rng(81)
    responses = [rng.normal(size=(13, 7)), rng.normal(size=(13, 5))]
    actual = runner.canonical_subspace(responses, 4, runner.torch.device("cpu"))
    gram = sum(value @ value.T for value in responses)
    _, vectors = np.linalg.eigh(gram)
    expected = vectors[:, -4:]
    np.testing.assert_allclose(
        actual @ actual.T, expected @ expected.T, rtol=2e-5, atol=2e-5,
    )


@pytest.mark.parametrize("samples,inputs", [(7, 11), (14, 6)])
def test_shared_ridge_factorization_matches_independent_reference(
    samples: int, inputs: int,
) -> None:
    rng = np.random.default_rng(samples + inputs)
    x = rng.normal(size=(samples, inputs)).astype(np.float32)
    first = rng.normal(size=(samples, 3)).astype(np.float32)
    second = rng.normal(size=(samples, 2)).astype(np.float32)
    actual = runner.analyses_from_subspaces(
        x, {"first": first, "second": second}, 1e-3, runner.torch.device("cpu"),
    )
    np.testing.assert_allclose(
        actual["first"], runner.analysis_from_subspace(x, first, 1e-3),
        rtol=2e-3, atol=2e-3,
    )
    np.testing.assert_allclose(
        actual["second"], runner.analysis_from_subspace(x, second, 1e-3),
        rtol=2e-3, atol=2e-3,
    )


def test_batched_syntheses_match_independent_least_squares() -> None:
    rng = np.random.default_rng(118)
    latent = rng.normal(size=(31, 5)).astype(np.float32)
    responses = {
        "gate": rng.normal(size=(31, 7)).astype(np.float32),
        "up": rng.normal(size=(31, 4)).astype(np.float32),
    }
    actual = runner.syntheses(latent, responses, runner.torch.device("cpu"))
    for name, response in responses.items():
        np.testing.assert_allclose(
            actual[name], runner.synthesis(latent, response),
            rtol=2e-4, atol=2e-4,
        )
