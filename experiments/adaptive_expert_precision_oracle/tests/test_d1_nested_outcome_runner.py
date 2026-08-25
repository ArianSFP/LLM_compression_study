from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import ast
import hashlib
import json
import sys

import numpy as np
import pytest
import torch


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(EXPERIMENT / "scripts")]

from oracle_study.d1_nested_outcomes import OutcomeAllocation  # noqa: E402
import run_d1_nested_outcomes as runner  # noqa: E402


def _write_json(path: Path, value: object) -> str:
    payload = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_raw_request_inputs_must_match_both_frozen_calibration_hashes(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "requests.jsonl"
    original = (
        json.dumps({"request_id": "7", "prompt_token_ids": [10, 11, 12]})
        + "\n"
    ).encode()
    manifest.write_bytes(original)
    original_manifest_sha = hashlib.sha256(original).hexdigest()
    facts = tmp_path / "requests_facts.json"
    original_facts_sha = _write_json(
        facts,
        {"output": {"sha256": original_manifest_sha, "bytes": len(original)}},
    )
    plan = SimpleNamespace(
        request_manifest_sha256=original_manifest_sha,
        request_manifest_facts_sha256=original_facts_sha,
    )
    assert runner._authenticate_raw_request_inputs(
        plan, manifest_path=manifest, manifest_facts_path=facts,
    ) == {
        "manifest_sha256": original_manifest_sha,
        "manifest_facts_sha256": original_facts_sha,
    }

    altered = (
        json.dumps({"request_id": "7", "prompt_token_ids": [10, 99, 12]})
        + "\n"
    ).encode()
    manifest.write_bytes(altered)
    altered_manifest_sha = hashlib.sha256(altered).hexdigest()
    altered_facts_sha = _write_json(
        facts,
        {"output": {"sha256": altered_manifest_sha, "bytes": len(altered)}},
    )
    with pytest.raises(ValueError, match="request manifest does not match"):
        runner._authenticate_raw_request_inputs(
            plan, manifest_path=manifest, manifest_facts_path=facts,
        )

    manifest.write_bytes(original)
    facts.write_text(json.dumps({"self_consistent": True}) + "\n")
    with pytest.raises(ValueError, match="manifest facts do not match"):
        runner._authenticate_raw_request_inputs(
            plan, manifest_path=manifest, manifest_facts_path=facts,
        )
    assert altered_facts_sha != original_facts_sha


def test_substituted_tokens_are_rejected_before_request_rows_are_parsed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_manifest = tmp_path / "original_requests.jsonl"
    original_payload = (
        json.dumps({"request_id": "7", "prompt_token_ids": [10, 11, 12]})
        + "\n"
    ).encode()
    original_manifest.write_bytes(original_payload)
    original_manifest_sha = hashlib.sha256(original_payload).hexdigest()
    original_facts = tmp_path / "original_requests_facts.json"
    original_facts_sha = _write_json(
        original_facts,
        {"output": {"sha256": original_manifest_sha}},
    )

    supplied_manifest = tmp_path / "supplied_requests.jsonl"
    supplied_payload = (
        json.dumps({"request_id": "7", "prompt_token_ids": [10, 99, 12]})
        + "\n"
    ).encode()
    supplied_manifest.write_bytes(supplied_payload)
    supplied_manifest_sha = hashlib.sha256(supplied_payload).hexdigest()
    supplied_facts = tmp_path / "supplied_requests_facts.json"
    _write_json(
        supplied_facts,
        {"output": {"sha256": supplied_manifest_sha}},
    )

    capture = tmp_path / "capture.json"
    capture_sha = _write_json(capture, {})
    config = tmp_path / "allocation.json"
    config_sha = _write_json(config, {
        "authenticated_capture_protocol": {
            "path": capture.name,
            "sha256": capture_sha,
        },
    })
    plan = SimpleNamespace(
        request_manifest_sha256=original_manifest_sha,
        request_manifest_facts_sha256=original_facts_sha,
    )
    token_rows_exposed = False

    def forbidden_parser(**_kwargs):
        nonlocal token_rows_exposed
        token_rows_exposed = True
        raise AssertionError("request token rows were parsed before raw authentication")

    monkeypatch.setattr(runner, "EXPERIMENT", tmp_path)
    monkeypatch.setattr(runner, "FROZEN_ALLOCATION_CONFIG_PATH", config)
    monkeypatch.setattr(runner, "ALLOCATION_CONFIG_FILE_SHA256", config_sha)
    monkeypatch.setattr(runner, "AUTHENTICATED_CAPTURE_CONFIG_PATH", capture.name)
    monkeypatch.setattr(runner, "AUTHENTICATED_CAPTURE_CONFIG_SHA256", capture_sha)
    monkeypatch.setattr(runner, "validate_outcome_config", lambda _config: None)
    monkeypatch.setattr(runner, "load_outcome_plan", lambda **_kwargs: plan)
    monkeypatch.setattr(
        runner, "validate_plan_against_config", lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runner, "_authenticate_outcome_code_bundle", lambda _plan: {},
    )
    monkeypatch.setattr(runner, "validate_capture_inputs", forbidden_parser)
    args = SimpleNamespace(
        config=config, request_manifest=supplied_manifest,
        request_manifest_facts=supplied_facts,
        allocation_manifest=tmp_path / "allocations.json",
        allocation_seal=tmp_path / "allocations.seal.json",
        expected_allocation_sha256="a" * 64,
        expected_calibration_sha256="b" * 64,
        splits=["evaluation"], layers=None,
    )
    with pytest.raises(ValueError, match="request manifest does not match"):
        runner._load_inputs(args)
    assert token_rows_exposed is False


def test_outcome_code_bundle_is_canonical_and_covers_execution_sources() -> None:
    bundle = runner._outcome_code_bundle()
    payload = {
        key: bundle[key]
        for key in (
            "schema", "scope", "roots", "files", "members",
            "allocation_candidate_code_identity",
        )
    }
    assert bundle["sha256"] == runner.canonical_sha256(payload)
    assert bundle["files"] == len(bundle["members"])
    members = {row["path"] for row in bundle["members"]}
    assert runner.OUTCOME_CODE_REQUIRED_MEMBERS.issubset(members)
    inventory = {
        row["path"]: {"sha256": row["sha256"], "bytes": row["bytes"]}
        for row in bundle["members"]
    }
    assert bundle["allocation_candidate_code_identity"] == {
        "schema": "pr13_d1_nested_code_bundle_v1",
        "files": len(inventory),
        "canonical_sha256": runner.canonical_sha256(inventory),
    }


def test_outcome_code_must_equal_sealed_allocation_bundle() -> None:
    bundle = runner._outcome_code_bundle()
    expected = dict(bundle["allocation_candidate_code_identity"])
    admitted = runner._authenticate_outcome_code_bundle(
        SimpleNamespace(calibration_candidate_code_identity=expected),
    )
    assert admitted["allocation_candidate_code_identity"] == expected

    changed = dict(expected)
    changed["canonical_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="sealed allocation code identity"):
        runner._authenticate_outcome_code_bundle(
            SimpleNamespace(calibration_candidate_code_identity=changed),
        )


def test_resume_rejects_a_different_outcome_code_bundle(tmp_path: Path) -> None:
    paths = runner._cell_paths(tmp_path, "evaluation", 0)
    paths["root"].mkdir(parents=True)
    recorded_pins = {
        "stable_input": "same",
        "outcome_code_bundle": {"sha256": "a" * 64},
    }
    _write_json(paths["facts"], {
        "schema": runner.OUTCOME_SCHEMA,
        "completed": True,
        "split": "evaluation",
        "injection_layer": 0,
        "allocations": 1,
        "input_pins": recorded_pins,
        "route_modes": list(runner.ROUTE_MODES),
        "one_exact_pretoken_cache_per_request_across_all_layers": True,
        "one_paired_native_baseline_per_request_across_all_layers": True,
        "logical_rows_fanned_out_from_physical_execution": True,
        "every_zero_route_mode_physically_executed": True,
    })
    current_pins = {
        **recorded_pins,
        "outcome_code_bundle": {"sha256": "b" * 64},
    }
    allocation = SimpleNamespace(request_id="7")
    with pytest.raises(RuntimeError, match="cannot authenticate resume"):
        runner._completed_cell(
            paths,
            split="evaluation",
            layer=0,
            pins=current_pins,
            allocations=(allocation,),
        )


def _capture_execution_stack() -> dict[str, object]:
    config = json.loads((
        EXPERIMENT / "configs"
        / "qwen36_mxfp4_d1_nested_safe_allocation_20260825_v2.json"
    ).read_text())
    return dict(config["authenticated_capture_execution_stack"])


def test_exact_capture_execution_stack_is_admitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = _capture_execution_stack()
    monkeypatch.setattr(runner.torch, "__version__", stack["torch"])
    monkeypatch.setattr(runner.torch.version, "cuda", stack["cuda"])
    observed = runner._validate_execution_stack(
        {"authenticated_capture_execution_stack": stack},
        {"gpu": stack["gpu"]},
    )
    assert observed == stack


@pytest.mark.parametrize("field", ["gpu", "torch", "cuda"])
def test_capture_execution_stack_mismatch_fails_closed(
    field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = _capture_execution_stack()
    hardware = {"gpu": stack["gpu"]}
    torch_version = str(stack["torch"])
    cuda_version = str(stack["cuda"])
    if field == "gpu":
        hardware["gpu"] = "NVIDIA GeForce RTX 3090"
    elif field == "torch":
        torch_version = "0.0.0"
    else:
        cuda_version = "0.0"
    monkeypatch.setattr(runner.torch, "__version__", torch_version)
    monkeypatch.setattr(runner.torch.version, "cuda", cuda_version)
    with pytest.raises(RuntimeError, match="execution stack mismatch"):
        runner._validate_execution_stack(
            {"authenticated_capture_execution_stack": stack}, hardware,
        )


class _PrefixObserver:
    def __init__(self) -> None:
        self.calls: list[tuple[int, object]] = []

    def run(self, _model, _encoded, position: int, prefix: object):
        self.calls.append((position, prefix))
        return SimpleNamespace(cache={"position": position}, elapsed_seconds=position + 0.5)


def test_exact_prefix_advances_once_and_retains_only_pretoken_cache() -> None:
    observer = _PrefixObserver()
    prefix, elapsed = runner.advance_exact_prefix(
        observer, object(), {}, position=3,
    )
    assert [position for position, _ in observer.calls] == [0, 1, 2]
    assert observer.calls[0][1] is None
    assert observer.calls[1][1] == {"position": 0}
    assert observer.calls[2][1] == {"position": 1}
    assert prefix == {"position": 2}
    assert elapsed == pytest.approx(4.5)


def test_selected_state_is_reordered_to_live_route_and_uses_live_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_ids = tuple(range(10, 18))
    live_ids = np.asarray(source_ids[::-1], np.int64)
    states = np.zeros((8, 512), np.uint8)
    states[:, 0] = np.arange(8, dtype=np.uint8)
    allocation = OutcomeAllocation(
        arm="nested_d1_safe",
        rate=360,
        layer=0,
        request_id="7",
        split="evaluation",
        expert_ids=source_ids,
        selected_state=states,
        selected_pages=int(np.unpackbits(states).sum()),
        page_cap=2880,
        prompt_sha256="a" * 64,
        domain="code",
        position=2,
        state_sha256="b" * 64,
    )
    weights = np.arange(1, 9, dtype=np.float32)
    weights /= weights.sum()
    targets = [np.asarray([float(expert)], np.float32) for expert in live_ids]
    reference = float(sum(weight * target[0] for weight, target in zip(weights, targets)))
    baseline = SimpleNamespace(
        router_ids={0: torch.as_tensor(live_ids)},
        router_scores={0: torch.as_tensor(weights)},
        routed=torch.as_tensor([reference], dtype=torch.float32),
    )
    responses = [SimpleNamespace(target_output=target) for target in targets]
    monkeypatch.setattr(
        runner,
        "split_state_output",
        lambda _response, state: np.asarray([float(state[0])], np.float32),
    )
    delta, facts = runner._selected_state_delta(
        allocation, baseline, {}, responses, q4_routed_max_abs_atol=1e-5,
    )
    reordered_values = np.arange(7, -1, -1, dtype=np.float64)
    expected = float(np.dot(weights, reordered_values - live_ids))
    assert delta.tolist() == pytest.approx([expected])
    assert facts["source_and_live_route_set_equal"] is True
    assert facts["source_and_live_route_order_equal"] is False
    assert facts["activation_dependent_page_effects"] is True

    baseline.router_ids[0] = torch.as_tensor([99, *live_ids[1:]])
    with pytest.raises(RuntimeError, match="route differs"):
        runner._selected_state_delta(
            allocation, baseline, {}, responses, q4_routed_max_abs_atol=1e-5,
        )


def test_outcome_runner_has_no_allocator_or_candidate_pickle_dependency() -> None:
    path = EXPERIMENT / "scripts" / "run_d1_nested_outcomes.py"
    source = path.read_text()
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "run_d1_nested_safe_oracle" not in imports
    assert "oracle_study.d1_nested_policy" not in imports
    assert "oracle_study.d1_nested_search" not in imports
    assert "nested_candidates.pkl" not in source
    assert "d1_candidate_logits" not in source
    assert "_run_candidate(" in source
    assert "advance_exact_prefix_native(" in source
    assert "one_exact_pretoken_cache_per_request_across_all_layers" in source
    run_phase = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_phase"
    )
    run_source = ast.get_source_segment(source, run_phase)
    assert run_source is not None
    stack_gate = run_source.index("_validate_execution_stack(")
    assert stack_gate < run_source.index("_completed_cell(")
    assert stack_gate < run_source.index("_load_model(")


class _MultiBaselineObserver:
    def __init__(self, layers: tuple[int, ...]) -> None:
        self.layers = layers
        self.calls = 0
        self.active = layers[0]
        self.x_by_layer: dict[int, torch.Tensor] = {}
        self.routed_by_layer: dict[int, torch.Tensor] = {}

    def set_injection_layer(self, layer: int) -> None:
        self.active = layer

    def run(self, _model, _encoded, _position, _prefix):
        self.calls += 1
        self.x_by_layer = {
            layer: torch.as_tensor([float(layer)]) for layer in self.layers
        }
        self.routed_by_layer = {
            layer: torch.as_tensor([float(layer + 100)]) for layer in self.layers
        }
        return runner.DecodeObservation(
            hidden={}, router_logits={}, router_scores={}, router_ids={},
            x=self.x_by_layer[self.active],
            routed=self.routed_by_layer[self.active],
            logits=torch.as_tensor([1.0]), cache={"stable": True},
            elapsed_seconds=0.25,
        )


def test_request_major_baseline_runs_twice_total_for_all_injection_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layers = (0, 1, 4, 6, 12, 23)
    observer = _MultiBaselineObserver(layers)
    monkeypatch.setattr(runner, "clone_decode_cache", lambda value: value.copy())
    monkeypatch.setattr(
        runner,
        "_observation_maxima",
        lambda _first, _second: {
            "hidden_max_abs": 0.0,
            "router_logits_max_abs": 0.0,
            "router_scores_max_abs": 0.0,
            "router_ids_equal": True,
            "x_max_abs": 0.0,
            "routed_max_abs": 0.0,
            "terminal_logits_max_abs": 0.0,
        },
    )
    monkeypatch.setattr(
        runner,
        "classified_cache_metrics",
        lambda _first, _second: {"full_cache_bit_identical": True},
    )
    views, facts, elapsed = runner._baseline_bundle(
        observer, object(), {}, 33, {"prefix": True}, layers,
    )
    assert observer.calls == 2
    assert set(views) == set(layers)
    assert [float(views[layer].x.item()) for layer in layers] == list(map(float, layers))
    assert facts["paired_baseline_shared_across_injection_layers"] is True
    assert elapsed == pytest.approx(0.5)


def test_physical_identity_deduplicates_logical_arm_and_rate_labels() -> None:
    ids = tuple(range(10, 18))
    state = np.zeros((8, 512), np.uint8)
    state[:, 0] = np.arange(8, dtype=np.uint8)

    def allocation(
        arm: str, rate: int, expert_ids: tuple[int, ...], selected: np.ndarray,
    ) -> OutcomeAllocation:
        return OutcomeAllocation(
            arm=arm, rate=rate, layer=4, request_id="7", split="evaluation",
            expert_ids=expert_ids, selected_state=selected,
            selected_pages=int(np.unpackbits(selected).sum()), page_cap=rate * 8,
            prompt_sha256="a" * 64, domain="code", position=33,
            state_sha256="b" * 64,
        )

    first = allocation("nested_local_only", 360, ids, state)
    second = allocation("nested_d1_safe", 384, ids[::-1], state[::-1].copy())
    delta = np.asarray([0.0, -0.0, 0.25], np.float32)
    first_identity = runner.physical_execution_identity(first, delta)
    second_identity = runner.physical_execution_identity(second, delta.copy())
    assert first_identity == second_identity
    changed = runner.physical_execution_identity(
        second, np.asarray([0.0, 0.0, 0.5], np.float32),
    )
    assert changed["physical_execution_id"] != first_identity["physical_execution_id"]


def test_zero_frozen_route_identity_proof_matches_hook_computation() -> None:
    logits = torch.linspace(-2.0, 2.0, 16, dtype=torch.float32).reshape(1, -1)
    ids = torch.topk(logits, 8, dim=-1).indices
    probabilities = torch.softmax(logits, dtype=torch.float, dim=-1)
    scores = torch.gather(probabilities, -1, ids)
    scores = scores / scores.sum(dim=-1, keepdim=True)
    baseline = SimpleNamespace(
        router_logits={layer: logits.clone() for layer in range(5, 40)},
        router_ids={layer: ids.clone() for layer in range(5, 40)},
        router_scores={layer: scores.clone() for layer in range(5, 40)},
    )
    proof = runner._prove_zero_frozen_routes_are_identity(
        baseline, 4, torch.device("cpu"),
    )
    assert proof[
        "zero_frozen_set_live_weight_all_downstream_scores_bit_identical"
    ] is True
    assert proof["zero_frozen_set_live_weight_max_abs"] == 0.0

    baseline.router_scores[12] = scores.clone()
    baseline.router_scores[12][0, 0] = 0.0
    with pytest.raises(RuntimeError, match="not a bitwise identity"):
        runner._prove_zero_frozen_routes_are_identity(
            baseline, 4, torch.device("cpu"),
        )
