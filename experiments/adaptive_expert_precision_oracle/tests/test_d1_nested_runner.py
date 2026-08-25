from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / "scripts"), str(EXPERIMENT / "src")]

from oracle_study.average_rate_allocator import (  # noqa: E402
    RateOption,
    exact_group_option_allocate,
    multiple_choice_allocate,
)
from oracle_study.d1_nested_artifacts import (  # noqa: E402
    canonical_sha256,
    freeze_calibration_spec,
)
from oracle_study.d1_nested_policy import canonical_add_only_moves  # noqa: E402
from oracle_study.d1_nested_search import (  # noqa: E402
    ExactD1Metrics,
    exact_all_expert_d1_metrics,
    stable_top8,
)
import run_d1_nested_safe_oracle as runner  # noqa: E402


def _option(pages: int, damage: float, state: int) -> RateOption:
    states = np.zeros(512, np.uint8)
    states[0] = np.uint8(state)
    return RateOption(
        pages=pages,
        damage=damage,
        states=states,
        source="test",
        page_price=0.0,
        coordinate_sweeps=0,
        local_passes=0,
    )


class Geometry:
    def local_damage(self, _states: np.ndarray) -> float:
        return 1.25

    def all_q2_damage(self) -> float:
        return 2.5

    def legacy_additive_damage(
        self, _states: np.ndarray, _weights: np.ndarray,
    ) -> float:
        return 1.5


def _logits() -> tuple[np.ndarray, np.ndarray]:
    target = np.linspace(3.0, -3.0, 256, dtype=np.float64)
    candidate = target.copy()
    candidate[8] = target[7] + 0.75
    return target, candidate


def test_raw_slice_route_is_diagnostic_but_anchored_route_is_hard_gate() -> None:
    target = np.arange(8, dtype=np.int64)
    raw_order_swap = target.copy()
    raw_order_swap[5:7] = raw_order_swap[5:7][::-1]
    parity = runner._route_anchor_parity(
        raw_order_swap, target.copy(), target,
    )
    assert parity == {
        "raw_cached_vs_full_ordered_top8_equal": False,
        "raw_cached_vs_full_top8_set_equal": True,
        "anchored_cached_vs_full_ordered_top8_equal": True,
    }

    raw_membership_swap = target.copy()
    raw_membership_swap[-1] = 8
    parity = runner._route_anchor_parity(
        raw_membership_swap, target.copy(), target,
    )
    assert parity["raw_cached_vs_full_ordered_top8_equal"] is False
    assert parity["raw_cached_vs_full_top8_set_equal"] is False

    wrong_anchor = target.copy()
    wrong_anchor[-1] = 8
    with pytest.raises(RuntimeError, match="anchored CUDA"):
        runner._route_anchor_parity(target, wrong_anchor, target)

    duplicate = target.copy()
    duplicate[-1] = duplicate[-2]
    with pytest.raises(ValueError, match="eight unique"):
        runner._route_anchor_parity(duplicate, target, target)



def test_v2_config_order_claims_match_implemented_selector_keys() -> None:
    config = runner.load_json(runner.FROZEN_ALLOCATION_CONFIG_PATH)
    runner._validate_deterministic_order_contract(config)
    assert (
        config["implementation_parent_commit"]
        == "7a2adea1dd3d188914a3c803aaebf51411b9c7e0"
    )
    assert "refresh_after_accepted_pages" not in config["d1_screen"]
    assert "beam_width" not in config["d1_screen"]

    changed = runner.load_json(runner.FROZEN_ALLOCATION_CONFIG_PATH)
    changed["common_core"]["deterministic_tie_break"][1] = "expert_id"
    with pytest.raises(RuntimeError, match="deterministic ordering"):
        runner._validate_deterministic_order_contract(changed)

    changed = runner.load_json(runner.FROZEN_ALLOCATION_CONFIG_PATH)
    changed["exact_d1_gate"]["arm4_unsafe_tie_break"][-1] = (
        "deterministic_state_sha256"
    )
    with pytest.raises(RuntimeError, match="deterministic ordering"):
        runner._validate_deterministic_order_contract(changed)

def test_state_record_uses_exact_target_logits_for_all_severity_fields() -> None:
    target, candidate = _logits()
    candidate_ids = stable_top8(candidate)
    expected = exact_all_expert_d1_metrics(
        target, candidate,
        baseline_top8=stable_top8(target),
        candidate_top8=candidate_ids,
    )
    target_ids = stable_top8(target)
    states = np.zeros((8, 512), np.uint8)
    group = {
        "layer": 0,
        "group": 0,
        "request_id": "18446744073709551615",
        "prompt_sha256": "a" * 64,
        "domain": "code",
        "split": "calibration",
        "position": 8,
        "experts": np.arange(8),
        "execution_weights": np.full(8, 0.125),
    }
    record = runner._state_record(
        arm=runner.ARM_LOCAL,
        rate=360,
        group=group,
        common_core=states,
        states=states,
        moves=(),
        geometry=Geometry(),
        selector_weights=np.full(8, 0.125),
        d1_logits=candidate,
        d1_candidate_ids=candidate_ids,
        target_logits=target,
        target_ids=target_ids,
        parameter={},
    )
    assert record["d1_crossings"] == expected.membership_crossings == 1
    assert record["d1_violation_depth"] == pytest.approx(expected.violation_depth)
    assert record["d1_routing_mass_churn"] == pytest.approx(
        expected.routing_mass_lost
    )
    assert record["d1_labeled_margin"] == pytest.approx(expected.target_set_margin)
    np.testing.assert_array_equal(record["d1_target_ids"], target_ids)
    np.testing.assert_array_equal(record["d1_candidate_ids"], candidate_ids)

    wrong_ids = target_ids.copy()
    wrong_ids[-1] = wrong_ids[-2]
    with pytest.raises(ValueError, match="baseline_top8"):
        runner._state_record(
            arm=runner.ARM_LOCAL,
            rate=360,
            group=group,
            common_core=states,
            states=states,
            moves=(),
            geometry=Geometry(),
            selector_weights=np.full(8, 0.125),
            d1_logits=candidate,
            target_logits=target,
            d1_candidate_ids=candidate_ids,
            target_ids=wrong_ids,
            parameter={},
        )


def test_trace_states_uses_nested_allocation_indices() -> None:
    frontiers = tuple(
        (_option(0, 1.0, 0), _option(1, 0.0, 1)) for _ in range(8)
    )
    selected = np.asarray([1, 0, 1, 0, 1, 0, 1, 0], np.int64)
    trace = SimpleNamespace(
        frontiers=frontiers,
        option_indices=np.zeros(8, np.int64),
        allocation=SimpleNamespace(option_indices=selected),
    )
    states = runner._trace_states(trace)
    np.testing.assert_array_equal(states[:, 0], selected.astype(np.uint8))


def test_refined_frontiers_reproduce_pr13_mckp_and_exact_local_states() -> None:
    frontiers = tuple(
        (
            _option(0, float(12 - expert), 0),
            _option(2, 0.0, 3),
        )
        for expert in range(8)
    )
    features = tuple(
        np.asarray([[float(8 - expert)], [0.0]], np.float64)
        for expert in range(8)
    )
    selector = np.arange(1, 9, dtype=np.float64)
    selector /= selector.sum()
    execution = np.arange(8, 0, -1, dtype=np.float64)
    execution /= execution.sum()
    group = {
        "selector_weights": selector,
        "execution_weights": execution,
    }
    config = {
        "group_exact_coordinate_sweeps": 8,
        "group_exact_pair_passes": 8,
    }
    refined = {
        "frontiers": frontiers,
        "features": features,
        "column_generation_rounds": 3,
        "column_generation_repairs": 24,
    }
    expected_pr13 = multiple_choice_allocate(frontiers, 8, selector * selector)
    expected_local = exact_group_option_allocate(
        frontiers,
        features,
        execution,
        8,
        expected_pr13.option_indices,
        max_coordinate_sweeps=8,
        max_pair_passes=8,
    )

    pr13_states, local_states, facts = runner._pr13_and_local_states(
        refined, group, config, 1,
    )

    np.testing.assert_array_equal(
        pr13_states[:, 0],
        3 * (expected_pr13.option_indices == 1).astype(np.uint8),
    )
    np.testing.assert_array_equal(
        local_states[:, 0],
        3 * (expected_local.option_indices == 1).astype(np.uint8),
    )
    assert facts == {
        "column_generation_rounds": 3,
        "column_generation_repairs": 24,
        "pr13_pages": expected_pr13.pages,
        "frontier_local_pages": expected_local.pages,
    }


def test_rate_precompute_batches_all_groups_and_propagates_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    groups = [{"group": 0}, {"group": 1}]
    coarse = [{"coarse": 0}, {"coarse": 1}]
    calls: list[tuple[object, ...]] = []
    active_limits = 0
    entered_limits: list[int] = []

    class OneThreadLimit:
        def __enter__(self):
            nonlocal active_limits
            active_limits += 1
            entered_limits.append(active_limits)
            return self

        def __exit__(self, exc_type, exc, traceback):
            nonlocal active_limits
            active_limits -= 1
            return False

    def limit(*, limits):
        assert limits == 1
        return OneThreadLimit()

        assert active_limits == 1
    def build(groups_arg, experts, proxy, beta, config, workers, layer):
        calls.append(("build", len(groups_arg), workers, layer))
        return coarse

    def refine(
        geometries, groups_arg, proxy, beta, config, rate, workers, layer,
    ):
        assert active_limits == 1
        assert geometries is coarse
        calls.append(("refine", rate, len(groups_arg), workers, layer))
        return [
            {"group": index, "rate": rate} for index in range(len(groups_arg))
        ]

    def allocate(refined, group, config, rate):
        assert refined == {"group": group["group"], "rate": rate}
        states = np.full((8, 512), rate + group["group"], np.int64)
        return states, states.copy(), {"rate": rate}

    monkeypatch.setattr(runner.base, "_build_geometries", build)
    monkeypatch.setattr(runner.base, "_refine_geometries", refine)
    monkeypatch.setattr(runner, "threadpool_limits", limit)
    monkeypatch.setattr(runner, "_pr13_and_local_states", allocate)
    proxy = np.zeros((1, 1), np.float32)
    geometries, states = runner._precompute_rate_states(
        groups,
        {},
        proxy,
        0.0,
        {},
        [360, 749],
        7,
        0,
    )

    assert geometries is coarse
    assert calls == [
        ("build", 2, 7, 0),
        ("refine", 360, 2, 7, 0),
        ("refine", 749, 2, 7, 0),
    ]
    assert [sorted(group_states) for group_states in states] == [
        [360, 749], [360, 749],
    ]
    assert entered_limits == [1, 1, 1]
    assert active_limits == 0
    assert int(states[1][749][0][0, 0]) == 750


class _BankGeometry:
    output_width = 1

    def __init__(self, damages: dict[int, float]) -> None:
        self.damages = damages

    @staticmethod
    def marker(states: np.ndarray) -> int:
        active = np.flatnonzero(np.asarray(states)[0])
        return -1 if active.size == 0 else int(active[0])

    def local_damage(self, states: np.ndarray) -> float:
        return float(self.damages[self.marker(states)])


def _marked_state(unit: int) -> np.ndarray:
    state = np.zeros((8, 512), np.uint8)
    state[0, unit] = 1
    return state


def _exact(crossings: int, mass: float = 0.0) -> ExactD1Metrics:
    target = np.arange(8, dtype=np.int64)
    observed = target.copy()
    if crossings:
        observed[-crossings:] = np.arange(8, 8 + crossings, dtype=np.int64)
    return ExactD1Metrics(
        baseline_top8=target,
        candidate_top8=observed,
        lost_target_experts=target[-crossings:] if crossings else np.asarray([], np.int64),
        entering_outsiders=(
            np.arange(8, 8 + crossings, dtype=np.int64)
            if crossings else np.asarray([], np.int64)
        ),
        membership_crossings=crossings,
        violation_depth=float(crossings),
        routing_mass_lost=float(mass),
        target_set_margin=-float(crossings),
    )


def test_strict_five_state_bank_is_frozen_and_safe_incumbent_saturates() -> None:
    states = [_marked_state(unit) for unit in range(5)]
    geometry = _BankGeometry({0: 5.0, 1: 4.0, 2: 3.0, 3: 1.0, 4: 2.0})
    observed: list[int] = []

    def safe_replay(state: np.ndarray) -> ExactD1Metrics:
        observed.append(geometry.marker(state))
        return _exact(0)

    safe = runner._strict_frozen_bank(
        rate=1,
        expert_ids=np.arange(8),
        pr13_state=states[0],
        exact_local_state=states[1],
        arm3_endpoints={16: states[2], 32: states[3], 64: states[4]},
        geometry=geometry,  # type: ignore[arg-type]
        exact_replay=safe_replay,
    )
    assert observed == [0, 1, 2, 3, 4]
    assert safe.selected_index == 0
    assert safe.reason == "incumbent_d1_safe"
    assert safe.names == (
        "independent_pr13_incumbent",
        "independent_exact_combined_local",
        "nested_local_window_16",
        "nested_local_window_32",
        "nested_local_window_64",
    )

    def unsafe_replay(state: np.ndarray) -> ExactD1Metrics:
        marker = geometry.marker(state)
        return _exact(1, 0.2) if marker == 0 else _exact(0)

    repaired = runner._strict_frozen_bank(
        rate=1,
        expert_ids=np.arange(8),
        pr13_state=states[0],
        exact_local_state=states[1],
        arm3_endpoints={16: states[2], 32: states[3], 64: states[4]},
        geometry=geometry,  # type: ignore[arg-type]
        exact_replay=unsafe_replay,
    )
    # Once safe, routing severity saturates and exact local damage selects W=32.
    assert repaired.selected_name == "nested_local_window_32"
    assert repaired.reason == "strict_crossing_repair"


def test_outer_exact_replay_cache_keys_complete_state_bytes() -> None:
    target = np.linspace(3.0, -3.0, 256, dtype=np.float32)
    calls: list[float] = []

    class Context:
        target_logits = target

        target_ids = stable_top8(target)
        def replay_route(self, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            calls.append(float(delta[0]))
            candidate = target.copy()
            if delta[0] > 0:
                candidate[8] = candidate[7] + 1.0
            return candidate, stable_top8(candidate)

    class OutputGeometry:
        def output_delta(self, states: np.ndarray) -> np.ndarray:
            return np.asarray([np.count_nonzero(states)], np.float32)

    cache = runner.ExactReplayCache(Context(), OutputGeometry())  # type: ignore[arg-type]
    zero = np.zeros((8, 512), np.uint8)
    one = zero.copy()
    one[0, 0] = 1
    assert cache(zero).membership_crossings == 0
    assert cache(zero.copy()).membership_crossings == 0
    assert cache.logits(one).shape == (256,)
    assert cache(one).membership_crossings == 1
    assert calls == [0.0, 1.0]
    assert cache.executions == 2
    assert cache.hits == 2


def _grid_config() -> dict[str, object]:
    return {
        "run_id": "nested-test",
        "injection_layers": [0, 1],
        "traffic_pairs": [
            {
                "metadata_matched_pages_per_expert": 360,
                "pr13_reference_pages_per_expert": 384,
            },
            {
                "metadata_matched_pages_per_expert": 725,
                "pr13_reference_pages_per_expert": 749,
            },
        ],
        "common_core": {"repair_window_pages_per_group_grid": [16, 32, 64]},
        "local_guardrail": {"eta_grid": [0.0, 0.0001, 0.001]},
    }


def test_calibration_and_authenticated_evaluation_row_grids() -> None:
    config = _grid_config()
    calibration, facts = runner._policy_grid(config, "calibration", None)
    assert facts is None
    assert runner._expected_rows_per_group([360, 384, 725, 749], calibration) == 92
    with pytest.raises(ValueError, match="requires authenticated"):
        runner._policy_grid(config, "evaluation", None)

    choices = [
        (360, 384, 32, 0.0001),
        (725, 749, 64, 0.001),
    ]
    pair_rows = []
    for low, high, window, eta in choices:
        parameter = {"repair_window_pages": window, "eta": eta}
        pair_rows.append({
            "metadata_matched_rate": low,
            "reference_rate": high,
            "repair_window_pages": window,
            "eta": eta,
            "selected_parameter_sha256": canonical_sha256(parameter),
            "applies_to_rates": [low, high],
            "applies_to_arms": [
                runner.ARM_LOCAL, runner.ARM_D1, runner.ARM_D1_SEVERITY,
            ],
        })
    frozen = freeze_calibration_spec({
        "run_id": config["run_id"],
        "config_canonical_sha256": runner.canonical_sha256(config),
        "layers": config["injection_layers"],
        "arm5_inherits_arm4_parameters": True,
        "calibration_evidence_sha256": "a" * 64,
        "traffic_pairs": pair_rows,
    })
    evaluation, authenticated = runner._policy_grid(
        config, "evaluation", frozen,
    )
    assert evaluation == {
        (360, 384): {32: (0.0001,)},
        (725, 749): {64: (0.001,)},
    }
    assert authenticated is not None
    assert len(authenticated["frozen_calibration_spec_sha256"]) == 64
    assert runner._expected_rows_per_group(
        [360, 384, 725, 749], evaluation,
    ) == 20
    with pytest.raises(ValueError, match="rejects"):
        runner._policy_grid(config, "calibration", frozen)


def test_compact_arm3_pair_retains_only_endpoints_and_canonical_chains() -> None:
    core = np.zeros((8, 512), np.uint8)
    low = core.copy()
    low[0, 0] = 1
    high = low.copy()
    high[1, 1] = 2
    compact = runner.CompactLocalPair(
        low_rate=1,
        high_rate=2,
        repair_window=1,
        common_core=core,
        low_state=low,
        high_state=high,
        core_damage=3.0,
        low_damage=2.0,
        high_damage=1.0,
        core_stop_reason="reserve_cap_reached",
        low_stop_reason="budget_reached",
        high_stop_reason="budget_reached",
        core_to_low_moves=canonical_add_only_moves(core, low),
        core_to_high_moves=canonical_add_only_moves(core, high),
        low_to_high_moves=canonical_add_only_moves(low, high),
        high_path_checkpoint_move_batches=(
            canonical_add_only_moves(low, high),
        ),
        high_path_checkpoint_damages=(2.0, 1.0),
    )
    assert compact.common_core.dtype == np.uint8
    assert not compact.common_core.flags.writeable
    np.testing.assert_array_equal(
        runner._replay_add_chain(compact.common_core, compact.core_to_high_moves),
        compact.high_state,
    )
    assert not hasattr(compact, "checkpoints")



def test_high_guardrail_fallback_audit_counts_pairs_not_endpoint_rows() -> None:
    rows = []
    for arm, fallback, qualified, attempts in (
        (runner.ARM_D1, True, 2, 3),
        (runner.ARM_D1_SEVERITY, False, 4, 1),
    ):
        for endpoint in ("low", "high"):
            rows.append({
                "arm": arm,
                "parameter": {
                    "endpoint": endpoint,
                    "pair_fallback_high_guardrail_infeasible": fallback,
                    "high_guardrail_qualified_checkpoints": qualified,
                    "high_guardrail_repair_attempts": attempts,
                },
            })
    audit = runner._high_guardrail_audit(rows)
    assert audit == {
        "policy_pairs": 2,
        "fallback_pairs": 1,
        "fallback_fraction": 0.5,
        "qualified_checkpoints_total": 6,
        "repair_attempts_total": 4,
        "by_arm": {
            runner.ARM_D1: {
                "policy_pairs": 1,
                "fallback_pairs": 1,
                "fallback_fraction": 1.0,
            },
            runner.ARM_D1_SEVERITY: {
                "policy_pairs": 1,
                "fallback_pairs": 0,
                "fallback_fraction": 0.0,
            },
        },
        "counting_unit": "policy_pair_low_endpoint_only",
    }


def test_scientific_memo_audit_excludes_host_dependent_durations() -> None:
    stats = {
        "high_path_entries": 2,
        "high_path_hits": 7,
        "high_path_misses": 2,
        "high_path_build_seconds": 12.345,
        "repair_entries": 3,
        "repair_hits": 5,
        "repair_misses": 3,
        "repair_build_seconds": 67.89,
        "safe_high_shortcuts": 9,
    }
    audit = runner._deterministic_policy_memo_audit(stats)
    assert audit == {
        key: int(value)
        for key, value in stats.items()
        if not key.endswith("_seconds")
    }
    assert not any(key.endswith("_seconds") for key in audit)
    with pytest.raises(ValueError, match="timing schema"):
        runner._deterministic_policy_memo_audit({
            key: value for key, value in stats.items()
            if key != "repair_build_seconds"
        })


def test_profile_sidecar_is_disabled_path_and_scientific_artifact_inert(
    tmp_path: Path,
) -> None:
    config = {"run_id": "profile-inertness"}
    disabled = runner.TimingProfile(False)
    assert runner._write_profile_sidecar(
        disabled,
        tmp_path,
        split="calibration",
        layer=0,
        requests=1,
        config=config,
    ) is None
    assert list(tmp_path.iterdir()) == []

    enabled = runner.TimingProfile(True)
    enabled.add("probe", 0.25, labels={"group": 0})
    path = runner._write_profile_sidecar(
        enabled,
        tmp_path,
        split="calibration",
        layer=0,
        requests=1,
        config=config,
    )
    assert path == tmp_path / "profiles" / "calibration" / "layer_00_timings.json"
    assert path.is_file()
    assert not (tmp_path / "calibration").exists()
    payload = runner.load_json(path)
    assert payload["included_in_scientific_layer_facts"] is False
    assert payload["scientific_candidate_rows_modified"] is False
