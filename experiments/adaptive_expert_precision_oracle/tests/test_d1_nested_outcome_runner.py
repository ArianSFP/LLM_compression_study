from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import ast
import sys

import numpy as np
import pytest
import torch


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(EXPERIMENT / "scripts")]

from oracle_study.d1_nested_outcomes import OutcomeAllocation  # noqa: E402
import run_d1_nested_outcomes as runner  # noqa: E402


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
