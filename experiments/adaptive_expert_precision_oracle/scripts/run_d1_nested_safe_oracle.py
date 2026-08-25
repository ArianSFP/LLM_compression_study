#!/usr/bin/env python3
"""Physical nested D1-safe Experiment A oracle.

Allocation is restricted to local and same-token next-layer D1 information.
This runner never computes a terminal logit or executes a candidate beyond the
next router.  It writes per-layer allocation candidates which are calibrated
and sealed before the separate downstream outcome runner may consume them.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import gc
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import platform
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(EXPERIMENT / "scripts")]

from oracle_study.average_rate_allocator import (  # noqa: E402
    exact_group_option_allocate,
    multiple_choice_allocate,
)
from oracle_study.causal_control import selector_and_execution_router_weights  # noqa: E402
from oracle_study.d1_decode import (  # noqa: E402
    cache_state_metrics,
    clone_decode_cache,
    mutation_safe_cached_autograd,
)
from oracle_study.d1_layer_slice import load_d1_layer_slice  # noqa: E402
from oracle_study.d1_nested_allocation import (  # noqa: E402
    AllocationCheckpoint,
    PhysicalPageMove,
    RouteRepairMetrics,
    add_only_local_completion,
    hard_saturating_route_repair,
    physical_added_pages,
    physical_removed_pages,
    physical_subset,
    reverse_local_prune_to_reserve,
    state_page_count,
    validate_states,
)
from oracle_study.d1_nested_geometry import TokenStateGeometry  # noqa: E402
from oracle_study.d1_nested_artifacts import (  # noqa: E402
    validate_frozen_calibration_spec,
)
from oracle_study.d1_nested_policy import (  # noqa: E402
    Arm3HighPathHint,
    NestedPolicyMemo,
    canonical_add_only_moves,
    orchestrate_nested_policy_pair,
)
from oracle_study.d1_nested_profile import (  # noqa: E402
    ProfiledGeometry,
    TimingProfile,
    profiled_callable,
)
from oracle_study.d1_nested_search import (  # noqa: E402
    ExactD1Metrics,
    build_screened_d1_boundaries,
    exact_all_expert_d1_metrics,
    stable_top8,
)
import capture_d1_nested_exact_decode as capture_auth  # noqa: E402
import run_d1_slice_oracle_pilot as base  # noqa: E402


SCHEMA = "pr13_d1_nested_safe_oracle_layer_candidates_v1"
ARM_PR13 = "independent_pr13"
ARM_STRICT = "strict_frozen_candidate_incumbent_repair"
ARM_LOCAL = "nested_local_only"
ARM_D1 = "nested_d1_safe"
ARM_D1_SEVERITY = "nested_d1_safe_violation_mass"
STRICT_BANK_WINDOWS = (16, 32, 64)
FROZEN_ALLOCATION_CONFIG_PATH = (
    EXPERIMENT / "configs"
    / "qwen36_mxfp4_d1_nested_safe_allocation_20260825_v2.json"
)
SCIENTIFIC_LAYER_FILES = (
    "nested_candidates.pkl",
    "nested_candidate_metrics.parquet",
    "nested_slice_parity.parquet",
)


@dataclass(frozen=True)
class CompactLocalPair:
    """Compact D1-blind core/endpoint result returned across a worker pipe."""

    low_rate: int
    high_rate: int
    repair_window: int
    common_core: np.ndarray
    low_state: np.ndarray
    high_state: np.ndarray
    core_damage: float
    low_damage: float
    high_damage: float
    core_stop_reason: str
    low_stop_reason: str
    high_stop_reason: str
    core_to_low_moves: tuple[PhysicalPageMove, ...]
    core_to_high_moves: tuple[PhysicalPageMove, ...]
    low_to_high_moves: tuple[PhysicalPageMove, ...]
    high_path_checkpoint_damages: tuple[float, ...]

    def __post_init__(self) -> None:
        low_rate = int(self.low_rate)
        high_rate = int(self.high_rate)
        window = int(self.repair_window)
        core = np.asarray(validate_states(self.common_core), np.uint8)
        low = np.asarray(validate_states(self.low_state), np.uint8)
        high = np.asarray(validate_states(self.high_state), np.uint8)
        damages = tuple(map(float, (
            self.core_damage, self.low_damage, self.high_damage,
        )))
        reasons = tuple(map(str, (
            self.core_stop_reason, self.low_stop_reason, self.high_stop_reason,
        )))
        low_moves = tuple(self.core_to_low_moves)
        high_moves = tuple(self.core_to_high_moves)
        low_to_high = tuple(self.low_to_high_moves)
        high_path_damages = tuple(map(float, self.high_path_checkpoint_damages))
        if (
            low_rate < 1
            or high_rate <= low_rate
            or window < 0
            or any(not np.isfinite(value) or value < 0.0 for value in damages)
            or any(not value for value in reasons)
            or not high_path_damages
            or any(
                not np.isfinite(value) or value < 0.0
                for value in high_path_damages
            )
            or not physical_subset(core, low)
            or not physical_subset(low, high)
            or state_page_count(low) > 8 * low_rate
            or state_page_count(high) > 8 * high_rate
        ):
            raise ValueError("compact Arm3 pair violates its physical contract")
        if not np.array_equal(_replay_add_chain(core, low_moves), low):
            raise ValueError("compact Arm3 low canonical chain does not replay")
        if not np.array_equal(_replay_add_chain(core, high_moves), high):
            raise ValueError("compact Arm3 high canonical chain does not replay")
        if not np.array_equal(_replay_add_chain(low, low_to_high), high):
            raise ValueError("compact Arm3 low-to-high path does not replay")
        if not np.isclose(
            high_path_damages[-1], damages[2], rtol=0.0, atol=1e-12,
        ):
            raise ValueError("compact Arm3 high-path endpoint damage changed")
        for value in (core, low, high):
            value.setflags(write=False)
        object.__setattr__(self, "low_rate", low_rate)
        object.__setattr__(self, "high_rate", high_rate)
        object.__setattr__(self, "repair_window", window)
        object.__setattr__(self, "common_core", core)
        object.__setattr__(self, "low_state", low)
        object.__setattr__(self, "high_state", high)
        object.__setattr__(self, "core_damage", damages[0])
        object.__setattr__(self, "low_damage", damages[1])
        object.__setattr__(self, "high_damage", damages[2])
        object.__setattr__(self, "core_stop_reason", reasons[0])
        object.__setattr__(self, "low_stop_reason", reasons[1])
        object.__setattr__(self, "high_stop_reason", reasons[2])
        object.__setattr__(self, "core_to_low_moves", low_moves)
        object.__setattr__(self, "core_to_high_moves", high_moves)
        object.__setattr__(self, "low_to_high_moves", low_to_high)
        object.__setattr__(
            self, "high_path_checkpoint_damages", high_path_damages,
        )


@dataclass(frozen=True)
class FrozenBankResult:
    names: tuple[str, ...]
    checkpoints: tuple[AllocationCheckpoint, ...]
    metrics: tuple[ExactD1Metrics, ...]
    selected_index: int
    selected_name: str
    reason: str


class ExactReplayCache:
    """One byte-keyed exact D1 replay cache shared by every arm for a token."""

    def __init__(
        self,
        context: Any,
        geometry: TokenStateGeometry,
        *,
        profile: TimingProfile | None = None,
        profile_labels: Mapping[str, str | int | float | bool | None] | None = None,
    ) -> None:
        self.context = context
        self.geometry = geometry
        self.profile = profile
        self.profile_labels = dict(profile_labels or {})
        self._cache: dict[bytes, tuple[ExactD1Metrics, np.ndarray, np.ndarray]] = {}
        self.executions = 0
        self.hits = 0
        self._target = np.asarray(context.target_logits, np.float64)
        self._target_ids = np.asarray(context.target_ids, np.int64)

    @staticmethod
    def key(states: np.ndarray) -> bytes:
        return np.asarray(validate_states(states), np.uint8).tobytes(order="C")

    def evaluate(
        self, states: np.ndarray,
    ) -> tuple[ExactD1Metrics, np.ndarray, np.ndarray]:
        value = validate_states(states)
        key = self.key(value)
        if key in self._cache:
            self.hits += 1
            if self.profile is not None:
                self.profile.add(
                    "exact_replay_cache_hit",
                    0.0,
                    labels=self.profile_labels,
                )
            return self._cache[key]
        def execute() -> tuple[np.ndarray, np.ndarray]:
            logits, candidate_ids = self.context.replay_route(
                self.geometry.output_delta(value),
            )
            return (
                np.asarray(logits, np.float32).reshape(-1),
                np.asarray(candidate_ids, np.int64).reshape(-1),
            )
        if self.profile is not None:
            with self.profile.measure(
                "exact_replay_execution",
                labels=self.profile_labels,
            ):
                logits, candidate_ids = execute()
        else:
            logits, candidate_ids = execute()
        metrics = exact_all_expert_d1_metrics(
            self._target, logits,
            baseline_top8=self._target_ids,
            candidate_top8=candidate_ids,
        )
        frozen_logits = logits.copy()
        frozen_logits.setflags(write=False)
        frozen_ids = candidate_ids.copy()
        frozen_ids.setflags(write=False)
        self._cache[key] = (metrics, frozen_logits, frozen_ids)
        self.executions += 1
        return self._cache[key]

    def __call__(self, states: np.ndarray) -> ExactD1Metrics:
        return self.evaluate(states)[0]

    def logits(self, states: np.ndarray) -> np.ndarray:
        return self.evaluate(states)[1]

    def ids(self, states: np.ndarray) -> np.ndarray:
        return self.evaluate(states)[2]

    @property
    def states(self) -> int:
        return len(self._cache)


def _replay_add_chain(
    source: np.ndarray, moves: Sequence[PhysicalPageMove],
) -> np.ndarray:
    current = validate_states(source)
    for move in moves:
        if move.direction != "add":
            raise ValueError("canonical physical chain contains a removal")
        if int(current[move.expert, move.unit]) != move.source_state:
            raise ValueError("canonical physical chain is stale")
        current[move.expert, move.unit] = move.destination_state
    return current


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    result = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not result or any(not isinstance(row, dict) for row in result):
        raise ValueError("request manifest must contain JSON objects")
    return result


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)

def _write_profile_sidecar(
    profile: TimingProfile,
    output_dir: Path,
    *,
    split: str,
    layer: int,
    requests: int,
    config: Mapping[str, Any],
) -> Path | None:
    """Write host timing outside scientific artifacts, or nothing when off."""

    if not profile.enabled:
        return None
    profile_path = (
        output_dir / "profiles" / str(split)
        / f"layer_{int(layer):02d}_timings.json"
    )
    payload = profile.snapshot()
    payload.update({
        "split": str(split),
        "layer": int(layer),
        "requests": int(requests),
        "config_canonical_sha256": canonical_sha256(config),
        "scientific_candidate_rows_modified": False,
        "included_in_scientific_layer_facts": False,
        "durations_are_host_dependent": True,
        "arm3_items_are_three_completion_passes_per_forked_pair": True,
    })
    atomic_json(profile_path, payload)
    return profile_path



def atomic_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def atomic_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(list(rows)).to_parquet(temporary, index=False)
    temporary.replace(path)


def _code_identity() -> dict[str, Any]:
    roots = (EXPERIMENT / "scripts", EXPERIMENT / "src" / "oracle_study")
    files = sorted(
        path for root in roots for path in root.rglob("*.py") if path.is_file()
    )
    inventory = {
        str(path.relative_to(EXPERIMENT)): {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in files
    }
    return {
        "schema": "pr13_d1_nested_code_bundle_v1",
        "files": len(inventory),
        "canonical_sha256": canonical_sha256(inventory),
    }


def _assert_file_pin(
    path: Path,
    expected_sha256: Any,
    label: str,
    *,
    expected_bytes: Any | None = None,
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    observed_bytes = path.stat().st_size
    observed_sha = sha256(path)
    if observed_sha != str(expected_sha256):
        raise RuntimeError(f"{label} SHA-256 mismatch")
    if expected_bytes is not None and observed_bytes != int(expected_bytes):
        raise RuntimeError(f"{label} byte count mismatch")
    return {"sha256": observed_sha, "bytes": observed_bytes}

def _cuda_softmax_top8(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror the model's authoritative FP32 CUDA softmax/top-k route."""

    values = logits.float()
    if (
        not values.is_cuda
        or values.shape != (256,)
        or not bool(torch.isfinite(values).all().item())
    ):
        raise RuntimeError("route execution requires 256 finite FP32 CUDA logits")
    probabilities = torch.softmax(values, dim=-1, dtype=torch.float32)
    expert_ids = torch.topk(
        probabilities, 8, dim=-1, largest=True, sorted=True,
    ).indices
    return probabilities, expert_ids


def _captured_route_cuda_preflight(
    capture_dir: Path,
    request_rows: Sequence[Mapping[str, Any]],
    layers: Sequence[int],
) -> dict[str, Any]:
    """Authenticate captured ordered D1 routes before frontier construction."""

    contexts = 0
    ordered_matches = 0
    numpy_stable_order_mismatches = 0
    boundary_probability_ties = 0
    with torch.inference_mode():
        for row in request_rows:
            request_id = str(row["request_id"])
            position = int(row["decode_position"])
            path = capture_auth.capture_path(capture_dir, request_id)
            with np.load(path, allow_pickle=False) as source:
                for layer in layers:
                    target_layer = int(layer) + 1
                    logits = np.asarray(
                        source["router_logits"][target_layer, position],
                        np.float32,
                    )
                    captured_ids = np.asarray(
                        source["router_ids"][target_layer, position],
                        np.int64,
                    )
                    tensor = torch.as_tensor(
                        logits, device="cuda:0", dtype=torch.float32,
                    )
                    probabilities, executed = _cuda_softmax_top8(tensor)
                    observed = executed.detach().cpu().numpy()
                    contexts += 1
                    if not np.array_equal(observed, captured_ids):
                        raise RuntimeError(
                            "captured ordered route CUDA preflight failed: "
                            f"request={request_id}, injection_layer={int(layer)}, "
                            f"target_layer={target_layer}, captured={captured_ids.tolist()}, "
                            f"executed={observed.tolist()}"
                        )
                    ordered_matches += 1
                    numpy_stable_order_mismatches += int(
                        not np.array_equal(stable_top8(logits), captured_ids)
                    )
                    mask = torch.ones(256, dtype=torch.bool, device=tensor.device)
                    mask[executed] = False
                    outside_max = torch.max(probabilities[mask])
                    selected_min = torch.min(probabilities[executed])
                    boundary_probability_ties += int(
                        bool(torch.eq(selected_min, outside_max).item())
                    )
    expected = len(request_rows) * len(tuple(layers))
    if contexts != expected or ordered_matches != expected:
        raise RuntimeError("captured route CUDA preflight coverage changed")
    return {
        "schema": "pr13_d1_captured_route_cuda_preflight_v1",
        "operation_role": "captured_unanchored_target_route_authentication",
        "operation": (
            "raw_logits_fp32_cuda_then_softmax_fp32_then_"
            "torch_topk_k8_sorted"
        ),
        "requests": len(request_rows),
        "target_injection_layers": list(map(int, layers)),
        "contexts": contexts,
        "ordered_matches": ordered_matches,
        "all_ordered_ids_exact": True,
        "captured_ordered_router_ids_authoritative": True,
        "numpy_stable_order_mismatches": numpy_stable_order_mismatches,
        "boundary_probability_ties": boundary_probability_ties,
    }



def _validate_deterministic_order_contract(config: Mapping[str, Any]) -> None:
    """Bind documentary ordering claims to the implemented selector keys."""

    common = config.get("common_core")
    screen = config.get("d1_screen")
    gate = config.get("exact_d1_gate")
    allocation = config.get("allocation_phase")
    expected_common = [
        "least_exact_combined_local_damage_increase",
        "routed_expert_slot",
        "unit",
        "projection_bit",
    ]
    expected_arm4 = [
        "fewest_membership_changes",
        "lowest_exact_combined_local_qenergy",
        "lowest_removal_sort_key_routed_slot_unit_projection_bit_source_state_destination_state",
        "lowest_addition_sort_key_routed_slot_unit_projection_bit_source_state_destination_state",
        "lowest_shortlist_index",
    ]
    expected_arm5 = [
        "fewest_membership_changes",
        "lowest_boundary_violation_depth_if_candidate_remains_unsafe_else_zero",
        "lowest_baseline_routing_mass_lost_if_candidate_remains_unsafe_else_zero",
        "lowest_exact_combined_local_qenergy",
        "lowest_removal_sort_key_routed_slot_unit_projection_bit_source_state_destination_state",
        "lowest_addition_sort_key_routed_slot_unit_projection_bit_source_state_destination_state",
        "lowest_shortlist_index",
    ]
    if (
        not isinstance(common, Mapping)
        or common.get("deterministic_tie_break") != expected_common
        or not isinstance(screen, Mapping)
        or "refresh_after_accepted_pages" in screen
        or "beam_width" in screen
        or not isinstance(gate, Mapping)
        or gate.get("arm4_unsafe_tie_break") != expected_arm4
        or gate.get("arm5_unsafe_tie_break") != expected_arm5
        or not isinstance(allocation, Mapping)
        or allocation.get("planned_fp32_and_bf16_once_effective_deltas_recorded")
        is not False
    ):
        raise RuntimeError(
            "immutable deterministic ordering/effective-delta contract changed"
        )


def _authenticate_allocation_inputs(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    dict[int, tuple[Path, Path]],
    dict[str, Any],
]:
    """Authenticate every scientific input before a model or factor is loaded."""

    if not FROZEN_ALLOCATION_CONFIG_PATH.is_file():
        raise FileNotFoundError(FROZEN_ALLOCATION_CONFIG_PATH)
    allocation_config_sha = sha256(args.config)
    if allocation_config_sha != sha256(FROZEN_ALLOCATION_CONFIG_PATH):
        raise RuntimeError(
            "allocation config bytes differ from the checked-in immutable v2 config"
        )
    config = load_json(args.config)
    if (
        config.get("schema") != "pr13_d1_nested_safe_allocation_config_v1"
        or config.get("experiment_stage") != "A"
        or config.get("run_id")
        != "qwen36_mxfp4_d1_nested_safe_allocation_20260825_v2"
    ):
        raise RuntimeError("allocation config is not the frozen v2 Experiment A run")
    _validate_deterministic_order_contract(config)
    pilot = bool(args.pilot)
    production_root = Path(str(config["output_root"])).resolve()
    output_root = Path(args.output_dir).resolve()
    if pilot:
        if args.request_limit is None:
            raise ValueError("--pilot requires an explicit --request-limit")
        pilot_root = production_root / "pilots"
        try:
            relative_pilot = output_root.relative_to(pilot_root)
        except ValueError as exc:
            raise RuntimeError(
                "pilot output must be below <v2 output_root>/pilots"
            ) from exc
        if not relative_pilot.parts:
            raise RuntimeError("pilot output requires a named child directory")
    elif args.request_limit is not None:
        raise ValueError("production v2 allocation rejects partial request cohorts")
    elif output_root != production_root:
        raise RuntimeError("production output differs from the immutable v2 run root")
    hardware_path = config.get("hardware_execution_path")
    if not isinstance(hardware_path, Mapping) or (
        hardware_path.get("cpu_worker_blas_threads") != 1
        or hardware_path.get("cpu_worker_processes_allowed_range") != [1, 64]
        or hardware_path.get("cpu_worker_processes_default") != 24
        or hardware_path.get("cpu_worker_process_count_is_semantics_inert")
        is not True
        or not 1 <= int(args.workers) <= 64
    ):
        raise RuntimeError("immutable CPU worker/thread hardware path changed")
    if str(args.device) not in {"cuda", "cuda:0"}:
        raise RuntimeError("immutable hardware path requires CUDA device zero")

    capture_protocol = config.get("authenticated_capture_protocol")
    if not isinstance(capture_protocol, Mapping):
        raise RuntimeError("authenticated capture protocol pin is absent")
    capture_config_path = EXPERIMENT / str(capture_protocol.get("path", ""))
    _assert_file_pin(
        capture_config_path,
        capture_protocol.get("sha256"),
        "capture protocol config",
    )
    capture_config, all_rows, capture_hashes = capture_auth.validate_capture_inputs(
        config_path=capture_config_path,
        manifest_path=args.manifest,
        manifest_facts_path=args.manifest_facts,
        frozen_config_path=capture_config_path,
    )
    for field in (
        "checkpoint_revision",
        "checkpoint_index_sha256",
        "checkpoint_config_sha256",
        "tokenizer_json_sha256",
        "selected_tree_sha256",
    ):
        if config.get(field) != capture_config.get(field):
            raise RuntimeError(f"allocation/capture protocol mismatch: {field}")
    capture_hardware = capture_config.get("hardware_execution_path")
    if (
        config.get("injection_layers") != capture_config.get("injection_layers")
        or config.get("decode_position") != capture_config.get("decode_position")
        or config.get("request_source") != capture_config.get("request_source")
        or not isinstance(capture_hardware, Mapping)
        or any(
            hardware_path.get(key) != value
            for key, value in capture_hardware.items()
        )
    ):
        raise RuntimeError("allocation/capture scientific protocol changed")

    checkpoint_pins = capture_auth.validate_checkpoint_pins(
        capture_config, args.checkpoint,
    )
    tokenizer_pin = _assert_file_pin(
        args.checkpoint / "tokenizer.json",
        config["tokenizer_json_sha256"],
        "checkpoint tokenizer",
    )
    tree_pin = _assert_file_pin(
        args.trees, config["selected_tree_sha256"], "selected expert tree",
    )
    pr13_inputs = config.get("authenticated_pr13_inputs")
    if not isinstance(pr13_inputs, Mapping):
        raise RuntimeError("authenticated PR13 input pins are absent")
    pr13_config_pin = _assert_file_pin(
        args.pr13_config,
        pr13_inputs.get("config_sha256"),
        "PR13 allocation config",
    )
    pr13_config = load_json(args.pr13_config)
    if (
        pr13_inputs.get("tree_sha256") != tree_pin["sha256"]
        or pr13_inputs.get("checkpoint_config_sha256")
        != checkpoint_pins["checkpoint_config_sha256"]
        or pr13_inputs.get("checkpoint_index_sha256")
        != checkpoint_pins["checkpoint_index_sha256"]
    ):
        raise RuntimeError("v2 PR13 checkpoint/tree pins changed")


    capture_inputs = config.get("authenticated_capture_artifacts")
    if not isinstance(capture_inputs, Mapping):
        raise RuntimeError("authenticated capture artifact pins are absent")
    if (
        capture_inputs.get("facts_schema") != capture_auth.FACTS_SCHEMA
        or capture_inputs.get("split") != "all"
        or capture_inputs.get("requests") != len(all_rows)
        or capture_inputs.get("files") != len(all_rows)
        or capture_inputs.get("manifest_sha256")
        != capture_hashes["manifest_sha256"]
        or capture_inputs.get("manifest_facts_sha256")
        != capture_hashes["manifest_facts_sha256"]
    ):
        raise RuntimeError("v2 capture-artifact pins changed")

    capture_facts_pin = _assert_file_pin(
        args.capture_facts,
        capture_inputs.get("facts_sha256"),
        "capture ledger",
    )
    capture_facts = load_json(args.capture_facts)
    layers = tuple(map(int, config["injection_layers"]))
    router_layers = tuple(sorted(set(layers + tuple(layer + 1 for layer in layers))))
    expected_names = {
        capture_auth.capture_path(args.capture_dir, str(row["request_id"])).name
        for row in all_rows
    }
    files = capture_facts.get("files")
    repeat_gates = capture_facts.get("repeat_gates")
    if (
        capture_facts.get("schema") != capture_auth.FACTS_SCHEMA
        or capture_facts.get("completed") is not True
        or capture_facts.get("split") != "all"
        or capture_facts.get("requests") != len(all_rows)
        or capture_facts.get("request_ids")
        != [str(row["request_id"]) for row in all_rows]
        or capture_facts.get("config_sha256") != capture_hashes["config_sha256"]
        or capture_facts.get("manifest_sha256") != capture_hashes["manifest_sha256"]
        or capture_facts.get("manifest_facts_sha256")
        != capture_hashes["manifest_facts_sha256"]
        or capture_facts.get("checkpoint_config_sha256")
        != checkpoint_pins["checkpoint_config_sha256"]
        or capture_facts.get("checkpoint_index_sha256")
        != checkpoint_pins["checkpoint_index_sha256"]
        or capture_facts.get("layers") != list(layers)
        or capture_facts.get("array_model_layers") != 40
        or capture_facts.get("array_hidden_size") != 2048
        or capture_facts.get("array_routed_experts") != 256
        or capture_facts.get("all_current_token_repeats_exact") is not True
        or capture_facts.get("exact_prefix_progression") is not True
        or capture_facts.get("one_isolated_decode_token_per_request") is not True
        or capture_facts.get("terminal_logits_retained_or_written") is not False
        or capture_facts.get("candidate_state_or_delta_used") is not False
        or capture_facts.get("downstream_candidate_outcome_used") is not False
        or not isinstance(files, Mapping)
        or set(files) != expected_names
        or not isinstance(repeat_gates, list)
        or len(repeat_gates) != len(all_rows)
    ):
        raise RuntimeError("capture ledger violates the immutable exact-decode contract")
    if (
        any(not isinstance(row, Mapping) for row in repeat_gates)
        or [str(row.get("request_id")) for row in repeat_gates]
        != [str(row["request_id"]) for row in all_rows]
        or any(row.get("all_exact") is not True for row in repeat_gates)
    ):
        raise RuntimeError("capture repeat-gate ledger changed")
    actual_names = {
        path.name for path in Path(args.capture_dir).iterdir()
        if path.is_file() and path.suffix == ".npz"
    }
    if actual_names != expected_names:
        raise RuntimeError("capture directory NPZ inventory differs from its ledger")

    full_split_rows = [
        row for row in all_rows if str(row["split"]) == str(args.split)
    ]
    expected_split_requests = int(
        config["request_source"][f"{args.split}_requests"]
    )
    if len(full_split_rows) != expected_split_requests:
        raise RuntimeError("selected split request cohort changed")
    if pilot:
        limit = int(args.request_limit)
        if not 1 <= limit < expected_split_requests:
            raise ValueError(
                "pilot request-limit must be smaller than the complete split"
            )
        split_rows = full_split_rows[:limit]
    else:
        split_rows = full_split_rows
    for row in split_rows:
        path = capture_auth.capture_path(
            args.capture_dir, str(row["request_id"]),
        )
        record = files.get(path.name)
        if not isinstance(record, Mapping):
            raise RuntimeError(f"capture ledger lacks {path.name}")
        _assert_file_pin(
            path,
            record.get("sha256"),
            f"capture archive {path.name}",
            expected_bytes=record.get("bytes"),
        )
        if record.get("request_manifest_row_sha256") != canonical_sha256(dict(row)):
            raise RuntimeError(f"capture row identity changed: {path.name}")
        repeat = capture_auth.validate_capture_archive(
            path,
            row,
            input_hashes=capture_hashes,
            injection_layers=layers,
            router_layers=router_layers,
            model_layers=40,
            hidden_size=2048,
            num_experts=256,
        )
        if record.get("repeat_gate_sha256") != canonical_sha256(repeat):
            raise RuntimeError(f"capture repeat gate changed: {path.name}")


    capture_environment = capture_facts.get("environment")
    capture_stack = config.get("authenticated_capture_execution_stack")
    hardware = capture_auth._gpu_gate(config)
    runtime = {
        **hardware,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    if (
        not isinstance(capture_environment, Mapping)
        or not isinstance(capture_stack, Mapping)
        or capture_stack.get("route_operation")
        != (
            "raw_logits_fp32_plus_anchor_fp32_cuda_then_softmax_fp32_"
            "then_torch_topk_k8_sorted"
        )
        or capture_stack.get("captured_ordered_router_ids_authoritative") is not True
        or capture_stack.get("allocation_runtime_must_match_gpu_torch_cuda_exactly")
        is not True
        or capture_stack.get("all_selected_request_d1_targets_preflight_required")
        is not True
        or any(
            str(capture_stack.get(key)) != str(capture_environment.get(key))
            for key in ("gpu", "torch", "cuda")
        )
        or any(
            str(runtime.get(key)) != str(capture_environment.get(key))
            for key in ("gpu", "torch", "cuda")
        )
    ):
        raise RuntimeError(
            "allocation GPU/Torch/CUDA stack differs from authenticated capture"
        )
    authenticated_capture_stack = {
        **{
            key: str(capture_environment[key])
            for key in ("gpu", "torch", "cuda")
        },
        "candidate_route_operation_role": "anchored_candidate_route_execution",
        "candidate_route_operation": str(capture_stack["route_operation"]),
        "target_preflight_operation_role": (
            "captured_unanchored_target_route_authentication"
        ),
        "target_preflight_operation": (
            "raw_logits_fp32_cuda_then_softmax_fp32_then_torch_topk_k8_sorted"
        ),
    }
    route_preflight = _captured_route_cuda_preflight(
        args.capture_dir, split_rows, layers,
    )
    factor_manifest_pin = _assert_file_pin(
        args.factor_manifest,
        pr13_inputs.get("factor_manifest_sha256"),
        "PR13 factor manifest",
    )
    factor_manifest = load_json(args.factor_manifest)
    manifest_layers = factor_manifest.get("layers")
    if (
        factor_manifest.get("schema")
        != pr13_inputs.get("factor_manifest_schema")
        or factor_manifest.get("completed") is not True
        or factor_manifest.get("config_sha256") != pr13_config_pin["sha256"]
        or factor_manifest.get("tree_sha256") != tree_pin["sha256"]
        or factor_manifest.get("checkpoint_config_sha256")
        != checkpoint_pins["checkpoint_config_sha256"]
        or factor_manifest.get("checkpoint_index_sha256")
        != checkpoint_pins["checkpoint_index_sha256"]
        or not isinstance(manifest_layers, Mapping)
        or set(manifest_layers) != {str(layer) for layer in range(40)}
    ):
        raise RuntimeError("PR13 factor manifest changed its authenticated contract")
    factor_paths: dict[int, tuple[Path, Path]] = {}
    factor_layer_pins: dict[str, Any] = {}
    for layer in layers:
        record = manifest_layers.get(str(layer))
        if not isinstance(record, Mapping):
            raise RuntimeError(f"factor manifest lacks layer {layer}")
        factor_path = Path(args.fit_dir) / str(record.get("file", ""))
        sidecar_path = Path(args.fit_dir) / str(record.get("sidecar", ""))
        if factor_path.parent.resolve() != Path(args.fit_dir).resolve():
            raise RuntimeError("factor manifest contains an unsafe factor filename")
        if sidecar_path.parent.resolve() != Path(args.fit_dir).resolve():
            raise RuntimeError("factor manifest contains an unsafe sidecar filename")
        factor_pin = _assert_file_pin(
            factor_path,
            record.get("sha256"),
            f"PR13 factor layer {layer}",
            expected_bytes=record.get("bytes"),
        )
        sidecar_pin = _assert_file_pin(
            sidecar_path,
            record.get("sidecar_sha256"),
            f"PR13 factor sidecar layer {layer}",
        )
        sidecar = load_json(sidecar_path)
        if sidecar.get("sha256") != factor_pin["sha256"]:
            raise RuntimeError(f"PR13 factor sidecar payload mismatch at layer {layer}")
        factor_paths[layer] = (factor_path, sidecar_path)
        factor_layer_pins[str(layer)] = {
            "factor": factor_pin,
            "sidecar": sidecar_pin,
        }

    code_identity = _code_identity()
    input_pins = {
        "schema": "pr13_d1_nested_allocation_input_pins_v1",
        "allocation_config_sha256": allocation_config_sha,
        "capture_protocol_config_sha256": capture_hashes["config_sha256"],
        "request_manifest_sha256": capture_hashes["manifest_sha256"],
        "request_manifest_facts_sha256": capture_hashes["manifest_facts_sha256"],
        "capture_facts_sha256": capture_facts_pin["sha256"],
        "checkpoint_config_sha256": checkpoint_pins["checkpoint_config_sha256"],
        "checkpoint_index_sha256": checkpoint_pins["checkpoint_index_sha256"],
        "tokenizer_json_sha256": tokenizer_pin["sha256"],
        "selected_tree_sha256": tree_pin["sha256"],
        "pr13_config_sha256": pr13_config_pin["sha256"],
        "factor_manifest_sha256": factor_manifest_pin["sha256"],
        "factor_layers": factor_layer_pins,
        "authenticated_capture_execution_stack": authenticated_capture_stack,
        "captured_route_cuda_preflight": route_preflight,
        "code_identity": code_identity,
    }
    frozen_sha = (
        None
        if args.frozen_calibration is None
        else sha256(args.frozen_calibration)
    )
    run_identity = {
        "schema": "pr13_d1_nested_allocation_run_identity_v1",
        "run_id": str(config["run_id"]),
        "split": str(args.split),
        "request_ids": [str(row["request_id"]) for row in split_rows],
        "input_pins": input_pins,
        "hardware_runtime": runtime,
        "captured_route_cuda_preflight": route_preflight,
        "frozen_calibration_file_sha256": frozen_sha,
        "pilot_nonpromotable": pilot,
        "completed_for_sealing": not pilot,
        "output_dir": str(output_root),
        "cpu_worker_processes": int(args.workers),
        "cpu_worker_blas_threads": 1,
    }
    return (
        config,
        pr13_config,
        split_rows,
        input_pins,
        factor_paths,
        {
            "hardware_runtime": runtime,
            "captured_route_cuda_preflight": route_preflight,
            "run_identity_sha256": canonical_sha256(run_identity),
            "run_identity": run_identity,
        },
    )


def capture_path(directory: Path, request_id: str) -> Path:
    path = capture_auth.capture_path(directory, request_id)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_capture(path: Path, layer: int) -> dict[str, np.ndarray]:
    return base._load_capture(path, int(layer))


def _selected_requests(
    manifest: Path, split: str, request_limit: int | None,
) -> list[dict[str, Any]]:
    rows = [row for row in load_jsonl(manifest) if str(row["split"]) == split]
    if request_limit is not None:
        rows = rows[: int(request_limit)]
    if not rows:
        raise RuntimeError("selected request split is empty")
    return rows


def _groups(
    request_rows: Sequence[Mapping[str, Any]],
    captures: Mapping[str, Mapping[str, np.ndarray]],
    layer: int,
) -> list[dict[str, Any]]:
    groups = []
    for row in request_rows:
        request_id = str(row["request_id"])
        capture = captures[request_id]
        position = int(row["decode_position"])
        if position != len(capture["input_ids"]) - 1:
            raise RuntimeError("capture does not end at its declared decode token")
        ids = np.asarray(capture["router_ids"][layer, position], np.int64)
        selector, execution = selector_and_execution_router_weights(
            capture["router_scores"][layer, position],
            historical_sum_atol=5e-7,
            execution_sum_atol=0.003,
        )
        groups.append({
            "group": len(groups),
            "request_id": request_id,
            "prompt_sha256": str(row["prompt_sha256"]),
            "domain": str(row["domain"]),
            "split": str(row["split"]),
            "position": position,
            "layer": int(layer),
            "experts": ids,
            "selector_weights": selector,
            "execution_weights": execution,
            "activation": np.asarray(capture["x"][position], np.float32),
        })
    return groups


def _trace_states(trace: Any) -> np.ndarray:
    allocation = getattr(trace, "allocation", trace)
    states = np.stack([
        np.asarray(frontier[int(index)].states, np.uint8)
        for frontier, index in zip(trace.frontiers, allocation.option_indices)
    ])
    return validate_states(states)


def _pr13_and_local_states(
    refined: Mapping[str, Any],
    group: Mapping[str, Any],
    config: Mapping[str, Any],
    rate: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Allocate PR13/local states from one already-refined rate frontier."""

    budget = 8 * int(rate)
    frontiers = tuple(refined["frontiers"])
    features = tuple(refined["features"])
    allocation = multiple_choice_allocate(
        frontiers,
        budget,
        np.asarray(group["selector_weights"], np.float64) ** 2,
    )
    local = exact_group_option_allocate(
        frontiers,
        features,
        np.asarray(group["execution_weights"], np.float64),
        budget,
        allocation.option_indices,
        max_coordinate_sweeps=int(config["group_exact_coordinate_sweeps"]),
        max_pair_passes=int(config["group_exact_pair_passes"]),
    )
    pr13_states = np.stack([
        np.asarray(frontier[int(index)].states, np.uint8)
        for frontier, index in zip(frontiers, allocation.option_indices)
    ])
    local_states = np.stack([
        np.asarray(frontier[int(index)].states, np.uint8)
        for frontier, index in zip(frontiers, local.option_indices)
    ])
    return validate_states(pr13_states), validate_states(local_states), {
        "column_generation_rounds": int(refined["column_generation_rounds"]),
        "column_generation_repairs": int(refined["column_generation_repairs"]),
        "pr13_pages": state_page_count(pr13_states),
        "frontier_local_pages": state_page_count(local_states),
    }


_COLLAPSE_CONTEXT: tuple[
    Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]], Mapping[str, Any], int,
] | None = None


def _collapse_rate_task(
    group_index: int,
) -> tuple[int, tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    if _COLLAPSE_CONTEXT is None:
        raise RuntimeError("forked rate-collapse context is absent")
    refined, groups, config, rate = _COLLAPSE_CONTEXT
    with threadpool_limits(limits=1):
        result = _pr13_and_local_states(
            refined[int(group_index)], groups[int(group_index)], config, rate,
        )
    return int(group_index), result


def _collapse_refined_rate(
    refined: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    rate: int,
    workers: int,
    layer: int,
) -> list[tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    """Fork final MCKP/exact-local collapse across already-refined groups."""

    if len(refined) != len(groups) or not groups:
        raise ValueError("rate-collapse refined/group grids differ or are empty")
    if int(workers) == 1 or len(groups) == 1:
        return [
            _pr13_and_local_states(item, group, config, int(rate))
            for item, group in zip(refined, groups, strict=True)
        ]
    global _COLLAPSE_CONTEXT
    _COLLAPSE_CONTEXT = (refined, groups, config, int(rate))
    results: list[tuple[np.ndarray, np.ndarray, dict[str, Any]] | None] = [
        None for _ in groups
    ]
    fork = mp.get_context("fork")
    try:
        with ProcessPoolExecutor(
            max_workers=min(int(workers), len(groups)), mp_context=fork,
        ) as pool:
            futures = {
                pool.submit(_collapse_rate_task, index): index
                for index in range(len(groups))
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                index, result = future.result()
                if index != futures[future] or results[index] is not None:
                    raise RuntimeError("forked rate-collapse identity changed")
                results[index] = result
                if completed % 8 == 0 or completed == len(groups):
                    print(
                        f"[layer {layer}] collapsed rate={rate} "
                        f"{completed}/{len(groups)} groups",
                        flush=True,
                    )
    finally:
        _COLLAPSE_CONTEXT = None
    if any(value is None for value in results):
        raise RuntimeError("forked rate-collapse grid was incomplete")
    return list(results)  # type: ignore[arg-type]


def _precompute_rate_states(
    groups: Sequence[Mapping[str, Any]],
    experts: Mapping[int, Mapping[str, Any]],
    proxy: np.ndarray,
    beta: float,
    config: Mapping[str, Any],
    rates: Sequence[int],
    workers: int,
    layer: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[int, tuple[np.ndarray, np.ndarray, dict[str, Any]]]],
]:
    """Build/refine/collapse all groups in worker batches, retaining states only."""

    with threadpool_limits(limits=1):
        geometries = base._build_geometries(
            groups, experts, proxy, beta, config, int(workers), int(layer),
        )
    if len(geometries) != len(groups):
        raise RuntimeError("coarse geometry grid does not match request groups")
    state_grid: list[
        dict[int, tuple[np.ndarray, np.ndarray, dict[str, Any]]]
    ] = [dict() for _ in groups]
    for rate in map(int, rates):
        with threadpool_limits(limits=1):
            refined = base._refine_geometries(
                geometries,
                groups,
                proxy,
                beta,
                config,
                rate,
                int(workers),
                int(layer),
            )
        collapsed = _collapse_refined_rate(
            refined, groups, config, rate, int(workers), int(layer),
        )
        for index, result in enumerate(collapsed):
            state_grid[index][rate] = result
        del refined, collapsed
    return geometries, state_grid


_ARM3_CONTEXT: tuple[
    Sequence[Mapping[str, Any]],
    Sequence[Mapping[str, Any]],
    Sequence[Mapping[int, tuple[np.ndarray, np.ndarray, dict[str, Any]]]],
    np.ndarray,
    float,
    Sequence[tuple[int, int]],
    int,
] | None = None


def _compact_local_pair_task(
    job: tuple[int, int, int],
) -> tuple[int, int, int, CompactLocalPair]:
    if _ARM3_CONTEXT is None:
        raise RuntimeError("forked Arm3 context is absent")
    group_index, pair_index, window = map(int, job)
    geometries, groups, rate_grid, proxy, beta, pairs, refresh = _ARM3_CONTEXT
    group = groups[group_index]
    low_rate, high_rate = pairs[pair_index]
    coarse = geometries[group_index]
    geometry = TokenStateGeometry(
        tuple(coarse["responses"]),
        np.asarray(group["execution_weights"], np.float64),
        proxy,
        beta,
    )
    high_reference = rate_grid[group_index][high_rate][0]
    with threadpool_limits(limits=1):
        core, low, high = _local_nested_pair(
            high_reference,
            np.asarray(group["experts"], np.int64),
            geometry,
            low_rate,
            high_rate,
            window,
            refresh,
        )
    common = np.asarray(core.final.states, np.uint8)
    low_state = np.asarray(low.final.states, np.uint8)
    high_state = np.asarray(high.final.states, np.uint8)
    result = CompactLocalPair(
        low_rate=low_rate,
        high_rate=high_rate,
        repair_window=window,
        common_core=common,
        low_state=low_state,
        high_state=high_state,
        core_damage=float(core.final.local_damage),
        low_damage=float(low.final.local_damage),
        high_damage=float(high.final.local_damage),
        core_stop_reason=str(core.stop_reason),
        low_stop_reason=str(low.stop_reason),
        high_stop_reason=str(high.stop_reason),
        core_to_low_moves=canonical_add_only_moves(common, low_state),
        core_to_high_moves=canonical_add_only_moves(common, high_state),
        low_to_high_moves=tuple(
            move
            for checkpoint in high.checkpoints[1:]
            for move in checkpoint.moves
        ),
        high_path_checkpoint_damages=tuple(
            float(checkpoint.local_damage) for checkpoint in high.checkpoints
        ),
    )
    return group_index, pair_index, window, result


def _precompute_arm3_pairs(
    geometries: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    rate_grid: Sequence[Mapping[int, tuple[np.ndarray, np.ndarray, dict[str, Any]]]],
    proxy: np.ndarray,
    beta: float,
    pair_specs: Sequence[tuple[int, int]],
    repair_windows: Sequence[int],
    refresh: int,
    workers: int,
    layer: int,
) -> list[dict[tuple[int, int, int], CompactLocalPair]]:
    """Precompute every D1-blind local pair before loading the D1 GPU slice."""

    if len(geometries) != len(groups) or len(rate_grid) != len(groups):
        raise ValueError("Arm3 precompute grids differ")
    windows = tuple(map(int, repair_windows))
    jobs = [
        (group_index, pair_index, window)
        for group_index in range(len(groups))
        for pair_index in range(len(pair_specs))
        for window in windows
    ]
    if not jobs:
        raise ValueError("Arm3 precompute job grid is empty")
    global _ARM3_CONTEXT
    _ARM3_CONTEXT = (
        geometries, groups, rate_grid, np.asarray(proxy), float(beta),
        tuple(pair_specs), int(refresh),
    )
    results: list[dict[tuple[int, int, int], CompactLocalPair]] = [
        {} for _ in groups
    ]
    try:
        if int(workers) == 1 or len(jobs) == 1:
            completed_rows = map(_compact_local_pair_task, jobs)
            for group_index, pair_index, window, result in completed_rows:
                results[group_index][(
                    result.low_rate, result.high_rate, window,
                )] = result
        else:
            fork = mp.get_context("fork")
            with ProcessPoolExecutor(
                max_workers=min(int(workers), len(jobs)), mp_context=fork,
            ) as pool:
                futures = {pool.submit(_compact_local_pair_task, job): job for job in jobs}
                for completed, future in enumerate(as_completed(futures), start=1):
                    group_index, pair_index, window, result = future.result()
                    if (group_index, pair_index, window) != futures[future]:
                        raise RuntimeError("forked Arm3 job identity changed")
                    key = (result.low_rate, result.high_rate, window)
                    if key in results[group_index]:
                        raise RuntimeError("forked Arm3 result repeated")
                    results[group_index][key] = result
                    if completed % 16 == 0 or completed == len(jobs):
                        print(
                            f"[layer {layer}] Arm3 local pairs "
                            f"{completed}/{len(jobs)}",
                            flush=True,
                        )
    finally:
        _ARM3_CONTEXT = None
    expected = len(pair_specs) * len(windows)
    if any(len(group) != expected for group in results):
        raise RuntimeError("forked Arm3 result grid was incomplete")
    return results


@dataclass
class D1Context:
    model_slice: Any
    prefix_cache: Any
    baseline_logits: np.ndarray
    baseline_ids: np.ndarray
    target_logits: np.ndarray
    target_ids: np.ndarray
    logit_anchor: np.ndarray
    sensitivities: np.ndarray
    candidate_ids: np.ndarray
    current_residual: np.ndarray
    current_x: np.ndarray
    current_routed: np.ndarray
    parity: dict[str, Any]

    def replay_route(self, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hidden = self.model_slice.compose_current_output(
            self.current_residual[None, None, :],
            self.current_x[None, None, :],
            self.current_routed[None, None, :],
            np.asarray(delta, np.float32)[None, None, :],
        )
        with torch.inference_mode():
            logits, _, _, _ = self.model_slice.next_router_outputs_decode(
                hidden, clone_decode_cache(self.prefix_cache),
                int(self.parity["position"]),
            )
            anchor = torch.as_tensor(
                self.logit_anchor,
                device=logits.device,
                dtype=torch.float32,
            )
            anchored = logits[0, 0].float() + anchor
            _, executed_ids = _cuda_softmax_top8(anchored)
        return (
            anchored.detach().cpu().numpy().astype(np.float32, copy=False),
            executed_ids.detach().cpu().numpy().astype(np.int64, copy=False),
        )

    def replay_logits(self, delta: np.ndarray) -> np.ndarray:
        return self.replay_route(delta)[0]


def _d1_context(
    model_slice: Any,
    capture: Mapping[str, np.ndarray],
    layer: int,
    position: int,
) -> D1Context:
    """Build the exact-prefix D1 context from authenticated stored hidden state."""

    stored_hidden = torch.as_tensor(
        np.asarray(capture["hidden"][layer], np.float32),
        device=model_slice.device,
        dtype=torch.bfloat16,
    ).unsqueeze(0)
    if stored_hidden.ndim != 3 or position >= stored_hidden.shape[1]:
        raise RuntimeError("stored injection-layer hidden state has invalid shape")
    stored_current = stored_hidden[:, position : position + 1]
    zero = np.zeros_like(np.asarray(capture["routed"][position], np.float32))
    singleton_current = model_slice.compose_current_output(
        np.asarray(capture["residual"][position], np.float32)[None, None, :],
        np.asarray(capture["x"][position], np.float32)[None, None, :],
        np.asarray(capture["routed"][position], np.float32)[None, None, :],
        zero[None, None, :],
    ).detach()
    if not torch.equal(singleton_current, stored_current):
        raise RuntimeError(
            "singleton zero-delta composition differs from authenticated stored hidden"
        )
    with torch.inference_mode():
        prefix = model_slice.decode_prefix_cache(stored_hidden[:, :position])
        stored_logits, _, stored_ids, stored_cache = (
            model_slice.next_router_outputs_decode(
                stored_current, clone_decode_cache(prefix), position,
            )
        )
        singleton_logits, _, singleton_ids, singleton_cache = (
            model_slice.next_router_outputs_decode(
                singleton_current, clone_decode_cache(prefix), position,
            )
        )
    cache_parity = cache_state_metrics(stored_cache, singleton_cache, layer + 1)
    if (
        not torch.equal(stored_logits, singleton_logits)
        or not torch.equal(stored_ids, singleton_ids)
        or not cache_parity.bit_identical
    ):
        raise RuntimeError("stored/singleton exact D1 decode is not bit-identical")
    raw_cuda = stored_logits[0, 0].detach().float()
    raw_baseline_logits = raw_cuda.cpu().numpy()
    baseline_ids = stored_ids[0, 0].detach().cpu().numpy()
    target_logits = np.asarray(capture["router_logits"][layer + 1, position], np.float32)
    target_ids = np.asarray(capture["router_ids"][layer + 1, position], np.int64)
    logit_anchor = np.asarray(target_logits - raw_baseline_logits, np.float32)
    raw_max_abs = float(np.max(np.abs(logit_anchor), initial=0.0))
    anchored_cuda = raw_cuda + torch.as_tensor(
        logit_anchor, device=raw_cuda.device, dtype=torch.float32,
    )
    target_probabilities, anchored_ids_cuda = _cuda_softmax_top8(anchored_cuda)
    anchored_baseline = anchored_cuda.detach().cpu().numpy()
    anchored_ids = anchored_ids_cuda.detach().cpu().numpy()
    if raw_max_abs > 0.0625 or not np.array_equal(anchored_baseline, target_logits):
        raise RuntimeError("raw D1 slice exceeds or fails its constant-logit anchor gate")
    if not np.array_equal(baseline_ids, target_ids):
        raise RuntimeError("stored-hidden cached decode changes ordered target IDs")
    if not np.array_equal(anchored_ids, target_ids):
        raise RuntimeError("anchored CUDA softmax/top-k differs from captured target IDs")
    baseline_logits = anchored_baseline
    outsider_mask = torch.ones(
        256, dtype=torch.bool, device=target_probabilities.device,
    )
    outsider_mask[anchored_ids_cuda] = False
    outsider_probabilities = target_probabilities.masked_fill(
        ~outsider_mask, -torch.inf,
    )
    outsiders = torch.topk(
        outsider_probabilities, 8, largest=True, sorted=True,
    ).indices.detach().cpu().numpy()
    candidate_ids = np.concatenate((target_ids[5:8], outsiders))
    current = stored_current.detach().clone().requires_grad_(True)
    with mutation_safe_cached_autograd():
        logits, _ = model_slice.next_router_logits_decode(
            current, clone_decode_cache(prefix), position,
        )
        values = logits[0, 0]
        gradients = []
        for offset, expert in enumerate(candidate_ids.tolist()):
            gradient = torch.autograd.grad(
                values[int(expert)], current,
                retain_graph=offset + 1 < len(candidate_ids),
                create_graph=False,
            )[0]
            gradients.append(gradient[0, 0].detach().float().cpu().numpy())
    parity = {
        "position": int(position),
        "stored_current_hidden_bit_identical": True,
        "singleton_zero_current_hidden_bit_identical": True,
        "stored_singleton_router_bit_identical": True,
        "stored_singleton_cache_bit_identical": bool(cache_parity.bit_identical),
        "raw_cached_vs_full_router_max_abs": raw_max_abs,
        "constant_logit_anchor_applied": bool(raw_max_abs > 0.0),
        "anchored_cached_vs_full_router_max_abs": 0.0,
        "anchored_cached_vs_full_router_bit_identical": True,
        "cached_vs_full_ordered_top8_equal": True,
    }
    return D1Context(
        model_slice=model_slice,
        prefix_cache=prefix,
        baseline_logits=baseline_logits,
        baseline_ids=baseline_ids,
        target_logits=target_logits,
        target_ids=target_ids,
        logit_anchor=logit_anchor,
        sensitivities=np.stack(gradients),
        candidate_ids=candidate_ids,
        current_residual=np.asarray(capture["residual"][position], np.float32),
        current_x=np.asarray(capture["x"][position], np.float32),
        current_routed=np.asarray(capture["routed"][position], np.float32),
        parity=parity,
    )


def _state_record(
    *,
    arm: str,
    rate: int,
    group: Mapping[str, Any],
    common_core: np.ndarray,
    states: np.ndarray,
    moves: Sequence[PhysicalPageMove] | Sequence[Mapping[str, Any]],
    geometry: TokenStateGeometry,
    selector_weights: np.ndarray,
    d1_logits: np.ndarray,
    target_logits: np.ndarray,
    d1_candidate_ids: np.ndarray,
    target_ids: np.ndarray,
    parameter: Mapping[str, Any],
) -> dict[str, Any]:
    exact = exact_all_expert_d1_metrics(
        target_logits,
        d1_logits,
        baseline_top8=target_ids,
        candidate_top8=d1_candidate_ids,
    )
    if not np.array_equal(exact.baseline_top8, np.asarray(target_ids, np.int64)):
        raise RuntimeError("stored target IDs changed from the authoritative route")
    move_rows = [
        move.to_dict() if isinstance(move, PhysicalPageMove) else dict(move)
        for move in moves
    ]
    return {
        "schema": SCHEMA,
        "arm": str(arm),
        "rate": int(rate),
        "layer": int(group["layer"]),
        "group": int(group["group"]),
        "request_id": str(group["request_id"]),
        "prompt_sha256": str(group["prompt_sha256"]),
        "domain": str(group["domain"]),
        "split": str(group["split"]),
        "position": int(group["position"]),
        "expert_ids": np.asarray(group["experts"], np.int64).tolist(),
        "selector_weights": np.asarray(selector_weights, np.float64).tolist(),
        "execution_weights": np.asarray(group["execution_weights"], np.float64).tolist(),
        "common_core_states": validate_states(common_core),
        "selected_states": validate_states(states),
        "moves": move_rows,
        "selected_pages": state_page_count(states),
        "core_pages": state_page_count(common_core),
        "local_damage": geometry.local_damage(states),
        "all_q2_damage": geometry.all_q2_damage(),
        "legacy_additive_damage": geometry.legacy_additive_damage(
            states, selector_weights,
        ),
        "d1_candidate_logits": np.asarray(d1_logits, np.float32),
        "d1_crossings": int(exact.membership_crossings),
        "d1_violation_depth": float(exact.violation_depth),
        "d1_candidate_ids": np.asarray(d1_candidate_ids, np.int64),
        "d1_target_ids": np.asarray(target_ids, np.int64),
        "d1_routing_mass_churn": float(exact.routing_mass_lost),
        "d1_labeled_margin": float(exact.target_set_margin),
        "parameter": dict(parameter),
        "state_sha256": hashlib.sha256(
            validate_states(states).astype(np.uint8).tobytes(),
        ).hexdigest(),
    }


def _local_nested_pair(
    high_reference: np.ndarray,
    expert_ids: np.ndarray,
    geometry: TokenStateGeometry,
    low_rate: int,
    high_rate: int,
    window: int,
    refresh: int,
) -> tuple[Any, Any, Any]:
    core_trace = reverse_local_prune_to_reserve(
        high_reference,
        expert_ids,
        budget_pages=8 * int(low_rate),
        repair_window_pages=int(window),
        local_damage=geometry.local_damage,
        score_removals=geometry.score_moves,
        refresh_after_accepted_pages=int(refresh),
    )
    low_trace = add_only_local_completion(
        core_trace.final.states,
        expert_ids,
        budget_pages=8 * int(low_rate),
        local_damage=geometry.local_damage,
        score_additions=geometry.score_moves,
        refresh_after_accepted_pages=int(refresh),
    )
    high_trace = add_only_local_completion(
        low_trace.final.states,
        expert_ids,
        budget_pages=8 * int(high_rate),
        local_damage=geometry.local_damage,
        score_additions=geometry.score_moves,
        refresh_after_accepted_pages=int(refresh),
    )
    if not (
        physical_subset(core_trace.final.states, low_trace.final.states)
        and physical_subset(low_trace.final.states, high_trace.final.states)
    ):
        raise RuntimeError("nested local physical chain failed")
    return core_trace, low_trace, high_trace



def _state_digest(states: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(validate_states(states), np.uint8).tobytes(order="C"),
    ).hexdigest()


def _paired_reference_high_core_seal(
    common_core: np.ndarray,
    reference_high: np.ndarray,
    *,
    reference_high_rate: int,
) -> dict[str, Any]:
    core = validate_states(common_core)
    reference = validate_states(reference_high)
    added = physical_added_pages(reference, core)
    removed = physical_removed_pages(reference, core)
    if added != 0 or not physical_subset(core, reference):
        raise RuntimeError("common core is not a literal subset of paired PR13 high")
    reference_pages = state_page_count(reference)
    core_pages = state_page_count(core)
    if removed != reference_pages - core_pages:
        raise RuntimeError("reference-high/core page accounting changed")
    return {
        "schema": "pr13_d1_paired_reference_high_core_seal_v1",
        "reference_arm": ARM_PR13,
        "reference_rate": int(reference_high_rate),
        "reference_state_sha256": _state_digest(reference),
        "reference_pages": int(reference_pages),
        "common_core_state_sha256": _state_digest(core),
        "common_core_pages": int(core_pages),
        "common_core_is_literal_subset": True,
        "removed_pages": int(removed),
        "added_pages": 0,
    }


def _full_d1_boundaries(context: D1Context) -> Any:
    sensitivities = np.asarray(context.sensitivities, np.float64)
    candidate_ids = np.asarray(context.candidate_ids, np.int64).reshape(-1)
    if (
        sensitivities.ndim != 2
        or sensitivities.shape[0] != candidate_ids.size
        or len(set(candidate_ids.tolist())) != candidate_ids.size
        or np.any((candidate_ids < 0) | (candidate_ids >= 256))
    ):
        raise ValueError("D1 context candidate sensitivities are invalid")
    full = np.zeros((256, sensitivities.shape[1]), np.float64)
    full[candidate_ids] = sensitivities
    return build_screened_d1_boundaries(
        context.target_logits,
        full,
        selected_expert_ids=np.asarray(context.target_ids, np.int64)[5:8],
        outsider_expert_ids=candidate_ids[3:],
    )


def _checkpoint_for_state(
    name: str,
    expert_ids: np.ndarray,
    states: np.ndarray,
    geometry: TokenStateGeometry,
    step: int,
) -> AllocationCheckpoint:
    value = validate_states(states)
    return AllocationCheckpoint(
        phase=str(name),
        step=int(step),
        expert_ids=np.asarray(expert_ids, np.int64),
        states=value,
        local_damage=float(geometry.local_damage(value)),
    )


def _strict_frozen_bank(
    *,
    rate: int,
    expert_ids: np.ndarray,
    pr13_state: np.ndarray,
    exact_local_state: np.ndarray,
    arm3_endpoints: Mapping[int, np.ndarray],
    geometry: TokenStateGeometry,
    exact_replay: Callable[[np.ndarray], ExactD1Metrics],
) -> FrozenBankResult:
    """Freeze the exact five-state Arm2 bank before any exact replay."""

    if tuple(sorted(map(int, arm3_endpoints))) != STRICT_BANK_WINDOWS:
        raise ValueError("strict Arm2 bank requires frozen W=16,32,64 endpoints")
    names = (
        "independent_pr13_incumbent",
        "independent_exact_combined_local",
        *(f"nested_local_window_{window}" for window in STRICT_BANK_WINDOWS),
    )
    states = (
        validate_states(pr13_state),
        validate_states(exact_local_state),
        *(validate_states(arm3_endpoints[window]) for window in STRICT_BANK_WINDOWS),
    )
    # Construction of every immutable checkpoint intentionally precedes the
    # first D1 callback: no replay can influence which candidates enter bank.
    checkpoints = tuple(
        _checkpoint_for_state(name, expert_ids, state, geometry, index)
        for index, (name, state) in enumerate(zip(names, states, strict=True))
    )
    exact = tuple(exact_replay(checkpoint.states) for checkpoint in checkpoints)
    values = tuple(
        RouteRepairMetrics(
            d1_crossings=metric.membership_crossings,
            arm5_severity=metric.routing_mass_lost,
            local_damage=checkpoint.local_damage,
        )
        for checkpoint, metric in zip(checkpoints, exact, strict=True)
    )
    selected = hard_saturating_route_repair(
        names,
        checkpoints,
        values,
        incumbent_index=0,
        page_budget=8 * int(rate),
        require_matched_pages=False,
    )
    return FrozenBankResult(
        names=names,
        checkpoints=checkpoints,
        metrics=exact,
        selected_index=selected.index,
        selected_name=selected.name,
        reason=selected.reason,
    )


def _metric_audit(metric: ExactD1Metrics) -> dict[str, Any]:
    return {
        "membership_crossings": int(metric.membership_crossings),
        "violation_depth": float(metric.violation_depth),
        "routing_mass_lost": float(metric.routing_mass_lost),
        "target_set_margin": float(metric.target_set_margin),
        "lost_target_experts": metric.lost_target_experts.tolist(),
        "entering_outsiders": metric.entering_outsiders.tolist(),
    }


def _repair_audit(result: Any | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "arm": str(result.arm),
        "initial_state_sha256": _state_digest(result.initial_state),
        "final_state_sha256": _state_digest(result.final_state),
        "initial_metrics": _metric_audit(result.initial_metrics),
        "final_metrics": _metric_audit(result.final_metrics),
        "stop_reason": str(result.stop_reason),
        "steps": [
            {
                "round_index": int(step.round_index),
                "state_sha256": _state_digest(step.state),
                "removal": step.removal.to_dict(),
                "addition": step.addition.to_dict(),
                "metrics": _metric_audit(step.metrics),
                "local_damage": float(step.local_damage),
            }
            for step in result.steps
        ],
    }


def _strict_bank_parameter(result: FrozenBankResult) -> dict[str, Any]:
    return {
        "frozen_candidate_bank": True,
        "all_states_frozen_before_exact_d1_selection": True,
        "candidate_names": list(result.names),
        "candidate_state_sha256": [
            _state_digest(checkpoint.states) for checkpoint in result.checkpoints
        ],
        "candidate_metrics": [
            {
                **_metric_audit(metric),
                "local_damage": float(checkpoint.local_damage),
                "pages": int(checkpoint.group_pages),
            }
            for checkpoint, metric in zip(
                result.checkpoints, result.metrics, strict=True,
            )
        ],
        "incumbent_name": result.names[0],
        "selected_candidate": result.selected_name,
        "selection_reason": result.reason,
        "safe_incumbent_identity_retained": bool(
            result.metrics[0].safe and result.selected_index == 0
        ),
        "selected_index": int(result.selected_index),
    }


def _policy_parameter(
    result: Any,
    *,
    repair_window: int,
    eta: float,
    endpoint: str,
    paired_reference_high_core_seal: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "repair_window": int(repair_window),
        "eta": float(eta),
        "endpoint": str(endpoint),
        "policy_arm": str(result.arm),
        "low_stop_reason": str(result.low_stop_reason),
        "high_stop_reason": str(result.high_stop_reason),
        "arm3_low_damage": float(result.arm3_low_damage),
        "arm3_high_damage": float(result.arm3_high_damage),
        "low_local_damage_limit": float(result.low_local_damage_limit),
        "high_reference_damage_limit": float(result.high_reference_damage_limit),
        "pair_fallback_high_guardrail_infeasible": bool(
            result.pair_fallback_high_guardrail_infeasible
        ),
        "high_guardrail_qualified_checkpoints": int(
            result.high_guardrail_qualified_checkpoints
        ),
        "high_guardrail_repair_attempts": int(
            result.high_guardrail_repair_attempts
        ),
        "strict_per_endpoint_local_guardrail_satisfied": True,
        "low_metrics": _metric_audit(result.low_metrics),
        "high_metrics": _metric_audit(result.high_metrics),
        "low_repair": _repair_audit(result.low_repair),
        "high_checkpoints": [
            {
                "checkpoint_index": int(checkpoint.checkpoint_index),
                "state_sha256": _state_digest(checkpoint.state),
                "metrics": _metric_audit(checkpoint.metrics),
                "local_damage": float(checkpoint.local_damage),
                "completion_moves": [
                    move.to_dict() for move in checkpoint.completion_moves
                ],
                "repair": _repair_audit(checkpoint.repair),
            }
            for checkpoint in result.high_checkpoints
        ],
        "high_repair_audits": [
            {
                "attempt_index": int(audit.attempt_index),
                "allowed_crossings": int(audit.allowed_crossings),
                "accepted": bool(audit.accepted),
                "result": _repair_audit(audit.result),
            }
            for audit in result.high_repair_audits
        ],
        "canonical_core_to_selected_chain": True,
        "paired_pr13_high_core_seal": dict(paired_reference_high_core_seal),
    }


def _high_guardrail_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Count pair fallbacks once through each policy's low endpoint row."""

    pair_rows = [
        row for row in rows
        if row.get("arm") in {ARM_D1, ARM_D1_SEVERITY}
        and isinstance(row.get("parameter"), Mapping)
        and row["parameter"].get("endpoint") == "low"
    ]
    fallbacks = sum(
        bool(row["parameter"].get("pair_fallback_high_guardrail_infeasible"))
        for row in pair_rows
    )
    by_arm = {
        arm: {
            "policy_pairs": sum(row["arm"] == arm for row in pair_rows),
            "fallback_pairs": sum(
                row["arm"] == arm
                and bool(row["parameter"].get(
                    "pair_fallback_high_guardrail_infeasible",
                ))
                for row in pair_rows
            ),
        }
        for arm in (ARM_D1, ARM_D1_SEVERITY)
    }
    for value in by_arm.values():
        value["fallback_fraction"] = (
            0.0 if value["policy_pairs"] == 0
            else value["fallback_pairs"] / value["policy_pairs"]
        )
    return {
        "policy_pairs": len(pair_rows),
        "fallback_pairs": fallbacks,
        "fallback_fraction": (
            0.0 if not pair_rows else fallbacks / len(pair_rows)
        ),
        "qualified_checkpoints_total": sum(
            int(row["parameter"].get("high_guardrail_qualified_checkpoints", 0))
            for row in pair_rows
        ),
        "repair_attempts_total": sum(
            int(row["parameter"].get("high_guardrail_repair_attempts", 0))
            for row in pair_rows
        ),
        "by_arm": by_arm,
        "counting_unit": "policy_pair_low_endpoint_only",
    }


def _deterministic_policy_memo_audit(
    stats: Mapping[str, float | int],
) -> dict[str, int]:
    """Retain reproducible memo counts; timing belongs only in profile sidecars."""

    timing_keys = {"high_path_build_seconds", "repair_build_seconds"}
    missing = timing_keys.difference(stats)
    if missing:
        raise ValueError(f"policy memo timing schema is incomplete: {sorted(missing)}")
    result: dict[str, int] = {}
    for key, value in stats.items():
        if key in timing_keys:
            continue
        if isinstance(value, bool) or int(value) != value or int(value) < 0:
            raise ValueError("deterministic policy memo counters must be nonnegative integers")
        result[str(key)] = int(value)
    return result


def _config_windows_etas(
    config: Mapping[str, Any],
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    windows = tuple(map(
        int, config["common_core"]["repair_window_pages_per_group_grid"],
    ))
    etas = tuple(map(float, config["local_guardrail"]["eta_grid"]))
    if (
        windows != STRICT_BANK_WINDOWS
        or not etas
        or len(set(windows)) != len(windows)
        or len(set(etas)) != len(etas)
        or any(window < 0 for window in windows)
        or any(not np.isfinite(eta) or eta < 0.0 for eta in etas)
    ):
        raise ValueError("configured repair-window/eta grids are invalid")
    return windows, etas


def _policy_grid(
    config: Mapping[str, Any],
    split: str,
    frozen_calibration: Mapping[str, Any] | None,
) -> tuple[dict[tuple[int, int], dict[int, tuple[float, ...]]], dict[str, Any] | None]:
    """Return full calibration or authenticated selected evaluation grid."""

    pairs = tuple(
        (
            int(item["metadata_matched_pages_per_expert"]),
            int(item["pr13_reference_pages_per_expert"]),
        )
        for item in config["traffic_pairs"]
    )
    windows, etas = _config_windows_etas(config)
    if split == "calibration":
        if frozen_calibration is not None:
            raise ValueError("calibration split rejects a frozen-parameter override")
        return {
            pair: {window: etas for window in windows}
            for pair in pairs
        }, None
    if split != "evaluation" or frozen_calibration is None:
        raise ValueError("evaluation split requires authenticated frozen calibration")
    digest = validate_frozen_calibration_spec(frozen_calibration)
    spec = frozen_calibration.get("spec")
    if not isinstance(spec, Mapping):
        raise ValueError("frozen calibration spec is absent")
    if (
        spec.get("run_id") != config.get("run_id")
        or spec.get("config_canonical_sha256") != canonical_sha256(config)
        or tuple(spec.get("layers", ()))
        != tuple(map(int, config["injection_layers"]))
        or spec.get("arm5_inherits_arm4_parameters") is not True
    ):
        raise ValueError("frozen calibration is not bound to this allocation config")
    pair_rows = spec.get("traffic_pairs")
    if not isinstance(pair_rows, list):
        raise ValueError("frozen calibration traffic-pair selections are absent")
    indexed = {
        (
            int(row.get("metadata_matched_rate", -1)),
            int(row.get("reference_rate", -1)),
        ): row
        for row in pair_rows if isinstance(row, Mapping)
    }
    if set(indexed) != set(pairs) or len(indexed) != len(pair_rows):
        raise ValueError("frozen calibration traffic-pair grid changed")
    grid: dict[tuple[int, int], dict[int, tuple[float, ...]]] = {}
    for pair in pairs:
        row = indexed[pair]
        window = int(row.get("repair_window_pages", -1))
        eta = float(row.get("eta", float("nan")))
        parameter = {"repair_window_pages": window, "eta": eta}
        if (
            window not in windows
            or eta not in etas
            or row.get("selected_parameter_sha256") != canonical_sha256(parameter)
            or row.get("applies_to_rates") != list(pair)
            or row.get("applies_to_arms")
            != [ARM_LOCAL, ARM_D1, ARM_D1_SEVERITY]
        ):
            raise ValueError("frozen calibration selected parameter changed")
        grid[pair] = {window: (eta,)}
    return grid, {
        "frozen_calibration_spec_sha256": str(digest),
        "calibration_evidence_sha256": str(
            spec.get("calibration_evidence_sha256", "")
        ),
    }



def _expected_rows_per_group(
    rates: Sequence[int],
    policy_grid: Mapping[tuple[int, int], Mapping[int, Sequence[float]]],
) -> int:
    rows = 2 * len(set(map(int, rates)))
    for window_grid in policy_grid.values():
        rows += 2 * len(window_grid)
        rows += 4 * sum(len(tuple(etas)) for etas in window_grid.values())
    return int(rows)



def run_layer(args: argparse.Namespace, layer: int) -> dict[str, Any]:
    started = time.perf_counter()
    profile = TimingProfile(bool(getattr(args, "profile_timings", False)))
    layer_profile_labels = {"layer": int(layer), "split": str(args.split)}
    if not all(hasattr(args, name) for name in (
        "authenticated_config",
        "authenticated_pr13_config",
        "authenticated_request_rows",
        "authenticated_factor_paths",
        "input_pins",
        "run_identity_sha256",
        "hardware_runtime",
    )):
        raise RuntimeError("run_layer requires successful authenticated preflight")
    config = args.authenticated_config
    pr13_config = args.authenticated_pr13_config
    frozen_raw = (
        None if args.frozen_calibration is None
        else load_json(args.frozen_calibration)
    )
    policy_grid, frozen_facts = _policy_grid(config, args.split, frozen_raw)
    config_windows, config_etas = _config_windows_etas(config)
    if tuple(map(int, args.repair_windows)) != config_windows:
        raise ValueError("CLI repair windows differ from the immutable config grid")
    request_rows = list(args.authenticated_request_rows)
    captures = {
        str(row["request_id"]): load_capture(
            capture_path(args.capture_dir, str(row["request_id"])), layer,
        ) for row in request_rows
    }
    groups = _groups(request_rows, captures, layer)
    factor_path, factor_sidecar = args.authenticated_factor_paths[int(layer)]
    if load_json(factor_sidecar)["sha256"] != sha256(factor_path):
        raise RuntimeError("authenticated PR13 factor sidecar payload changed")
    arrays = dict(np.load(factor_path, allow_pickle=False))
    proxy = np.asarray(arrays["proxy"], np.float32)
    beta = float(np.asarray(arrays["beta"]).reshape(-1)[0])
    experts = base._prepare_experts(
        args, layer, groups, arrays, proxy, beta, pr13_config,
    )
    pair_specs = [
        (
            int(item["metadata_matched_pages_per_expert"]),
            int(item["pr13_reference_pages_per_expert"]),
        ) for item in config["traffic_pairs"]
    ]
    rates = sorted({rate for pair in pair_specs for rate in pair})
    with profile.measure(
        "rate_state_precompute",
        calls=len(groups) * len(rates),
        labels=layer_profile_labels,
    ):
        geometries, rate_state_grid = _precompute_rate_states(
            groups,
            experts,
            proxy,
            beta,
            pr13_config,
            rates,
            int(args.workers),
            int(layer),
        )
    refresh = int(config["common_core"]["refresh_after_accepted_pages"])
    arm3_jobs = len(groups) * len(pair_specs) * len(config_windows)
    with profile.measure(
        "arm3_local_pair_precompute",
        calls=arm3_jobs,
        items=3 * arm3_jobs,
        labels={
            **layer_profile_labels,
            "completion_passes_per_pair": 3,
            "scope": "forked_batch_including_process_overhead",
        },
    ):
        arm3_grid = _precompute_arm3_pairs(
            geometries,
            groups,
            rate_state_grid,
            proxy,
            beta,
            pair_specs,
            config_windows,
            refresh,
            int(args.workers),
            int(layer),
        )
    del arrays, experts
    gc.collect()
    with profile.measure("d1_slice_load", labels=layer_profile_labels):
        model_slice = load_d1_layer_slice(
            args.checkpoint, layer, device=args.device,
        )

    rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    replay_executions = 0
    replay_hits = 0
    screen = config["d1_screen"]
    move_limit = min(32, int(screen["candidate_page_shortlist"]))
    candidate_limit = min(8, int(screen["maximum_exact_finalists"]))
    expected_per_group = _expected_rows_per_group(rates, policy_grid)
    for group_index, (group, coarse) in enumerate(
        zip(groups, geometries, strict=True),
    ):
        request_id = str(group["request_id"])
        group_profile_labels = {
            **layer_profile_labels,
            "group": int(group["group"]),
            "request_id": request_id,
        }
        group_started = time.perf_counter() if profile.enabled else 0.0
        with profile.measure("d1_context", labels=group_profile_labels):
            context = _d1_context(
                model_slice, captures[request_id], layer, int(group["position"]),
            )
        token_geometry = TokenStateGeometry(
            tuple(coarse["responses"]),
            np.asarray(group["execution_weights"], np.float64),
            proxy,
            beta,
        )
        boundaries = _full_d1_boundaries(context)
        replay = ExactReplayCache(
            context,
            token_geometry,
            profile=profile if profile.enabled else None,
            profile_labels=group_profile_labels,
        )
        policy_memo = NestedPolicyMemo()
        rate_states = rate_state_grid[group_index]
        local_pairs = arm3_grid[group_index]
        first_row = len(rows)

        # Arm 1: exactly one independent PR13 incumbent at each numeric rate.
        for rate in rates:
            state = rate_states[rate][0]
            arm_profile_labels = {
                **group_profile_labels,
                "arm": ARM_PR13,
                "rate": int(rate),
            }
            arm_started = time.perf_counter() if profile.enabled else 0.0
            rows.append(_state_record(
                arm=ARM_PR13,
                rate=rate,
                group=group,
                common_core=state,
                states=state,
                moves=(),
                geometry=token_geometry,
                selector_weights=np.asarray(group["selector_weights"]),
                d1_logits=replay.logits(state),
                d1_candidate_ids=replay.ids(state),
                target_logits=context.target_logits,
                target_ids=context.target_ids,
                parameter={
                    "independent_column_generated": True,
                    "frontier_facts": dict(rate_states[rate][2]),
                },
            ))
            profile.add(
                "arm_execution",
                time.perf_counter() - arm_started if profile.enabled else 0.0,
                labels=arm_profile_labels,
            )


        # Arm 2: freeze all five complete states at a rate before replaying any.
        for rate in rates:
            arm_profile_labels = {
                **group_profile_labels,
                "arm": ARM_STRICT,
                "rate": int(rate),
            }
            arm_started = time.perf_counter() if profile.enabled else 0.0
            pair = next(pair for pair in pair_specs if rate in pair)
            endpoints = {
                window: (
                    local_pairs[(pair[0], pair[1], window)].low_state
                    if rate == pair[0]
                    else local_pairs[(pair[0], pair[1], window)].high_state
                )
                for window in STRICT_BANK_WINDOWS
            }
            bank = _strict_frozen_bank(
                rate=rate,
                expert_ids=np.asarray(group["experts"], np.int64),
                pr13_state=rate_states[rate][0],
                exact_local_state=rate_states[rate][1],
                arm3_endpoints=endpoints,
                geometry=token_geometry,
                exact_replay=replay,
            )
            selected = bank.checkpoints[bank.selected_index]
            rows.append(_state_record(
                arm=ARM_STRICT,
                rate=rate,
                group=group,
                common_core=selected.states,
                states=selected.states,
                moves=(),
                geometry=token_geometry,
                selector_weights=np.asarray(group["selector_weights"]),
                d1_logits=replay.logits(selected.states),
                d1_candidate_ids=replay.ids(selected.states),
                target_logits=context.target_logits,
                target_ids=context.target_ids,
                parameter=_strict_bank_parameter(bank),
            ))
            profile.add(
                "arm_execution",
                time.perf_counter() - arm_started if profile.enabled else 0.0,
                labels=arm_profile_labels,
            )


        # Arms 3--5 share each D1-blind physical core and Arm3 endpoints.
        for pair in pair_specs:
            low_rate, high_rate = pair
            for window, eta_values in policy_grid[pair].items():
                local = local_pairs[(low_rate, high_rate, int(window))]
                paired_reference_seal = _paired_reference_high_core_seal(
                    local.common_core,
                    rate_states[high_rate][0],
                    reference_high_rate=high_rate,
                )
                arm_profile_labels = {
                    **group_profile_labels,
                    "arm": ARM_LOCAL,
                    "low_rate": int(low_rate),
                    "high_rate": int(high_rate),
                    "repair_window": int(window),
                }
                arm_started = time.perf_counter() if profile.enabled else 0.0

                rows.append(_state_record(
                    arm=ARM_LOCAL,
                    rate=low_rate,
                    group=group,
                    common_core=local.common_core,
                    states=local.low_state,
                    moves=local.core_to_low_moves,
                    geometry=token_geometry,
                    selector_weights=np.asarray(group["selector_weights"]),
                    d1_logits=replay.logits(local.low_state),
                    d1_candidate_ids=replay.ids(local.low_state),
                    target_logits=context.target_logits,
                    target_ids=context.target_ids,
                    parameter={
                        "repair_window": int(window),
                        "endpoint": "low",
                        "core_stop_reason": local.core_stop_reason,
                        "completion_stop_reason": local.low_stop_reason,
                        "paired_pr13_high_core_seal": dict(paired_reference_seal),
                    },
                ))
                rows.append(_state_record(
                    arm=ARM_LOCAL,
                    rate=high_rate,
                    group=group,
                    common_core=local.common_core,
                    states=local.high_state,
                    moves=local.core_to_high_moves,
                    geometry=token_geometry,
                    selector_weights=np.asarray(group["selector_weights"]),
                    d1_logits=replay.logits(local.high_state),
                    d1_candidate_ids=replay.ids(local.high_state),
                    target_logits=context.target_logits,
                    target_ids=context.target_ids,
                    parameter={
                        "repair_window": int(window),
                        "endpoint": "high",
                        "core_stop_reason": local.core_stop_reason,
                        "completion_stop_reason": local.high_stop_reason,
                        "paired_pr13_high_core_seal": dict(paired_reference_seal),
                    },
                ))
                profile.add(
                    "arm_execution",
                    time.perf_counter() - arm_started if profile.enabled else 0.0,
                    calls=2,
                    labels=arm_profile_labels,
                )
                for eta in eta_values:
                    for policy_name, arm_name in (
                        ("arm4", ARM_D1), ("arm5", ARM_D1_SEVERITY),
                    ):
                        policy_labels = {
                            **group_profile_labels,
                            "arm": arm_name,
                            "low_rate": int(low_rate),
                            "high_rate": int(high_rate),
                            "repair_window": int(window),
                            "eta": float(eta),
                        }
                        policy_geometry = (
                            ProfiledGeometry(
                                token_geometry, profile, policy_labels,
                            )
                            if profile.enabled else token_geometry
                        )
                        policy_replay = (
                            profiled_callable(
                                replay,
                                profile,
                                "policy_exact_replay_callback",
                                labels=policy_labels,
                            )
                            if profile.enabled else replay
                        )
                        memo_before = policy_memo.stats()
                        policy_started = (
                            time.perf_counter() if profile.enabled else 0.0
                        )
                        result = orchestrate_nested_policy_pair(
                            local.common_core,
                            local.low_state,
                            local.high_state,
                            np.asarray(group["experts"], np.int64),
                            policy_geometry,
                            boundaries,
                            policy_replay,
                            eta=float(eta),
                            j_q2=token_geometry.all_q2_damage(),
                            low_cap_pages=8 * low_rate,
                            high_cap_pages=8 * high_rate,
                            refresh_after_accepted_pages=refresh,
                            arm=policy_name,
                            maximum_repair_rounds=8,
                            move_limit=move_limit,
                            candidate_limit=candidate_limit,
                            memo=policy_memo,
                            arm3_high_path_hint=Arm3HighPathHint(
                                completion_moves=local.low_to_high_moves,
                                checkpoint_local_damages=(
                                    local.high_path_checkpoint_damages
                                ),
                                stop_reason=local.high_stop_reason,
                            ),
                        )
                        memo_after = policy_memo.stats()
                        profile.add(
                            "policy_pair",
                            time.perf_counter() - policy_started if profile.enabled else 0.0,
                            labels=policy_labels,
                        )
                        profile.add(
                            "local_completion",
                            float(memo_after["high_path_build_seconds"])
                            - float(memo_before["high_path_build_seconds"]),
                            calls=int(memo_after["high_path_misses"])
                            - int(memo_before["high_path_misses"]),
                            labels=policy_labels,
                        )
                        profile.add(
                            "local_completion_cache_hit",
                            0.0,
                            calls=int(memo_after["high_path_hits"])
                            - int(memo_before["high_path_hits"]),
                            labels=policy_labels,
                        )
                        profile.add(
                            "exact_repair_shortlist",
                            float(memo_after["repair_build_seconds"])
                            - float(memo_before["repair_build_seconds"]),
                            calls=int(memo_after["repair_misses"])
                            - int(memo_before["repair_misses"]),
                            labels=policy_labels,
                        )
                        profile.add(
                            "exact_repair_cache_hit",
                            0.0,
                            calls=int(memo_after["repair_hits"])
                            - int(memo_before["repair_hits"]),
                            labels=policy_labels,
                        )
                        for endpoint, rate, state, moves in (
                            (
                                "low", low_rate, result.low_state,
                                result.core_to_low_moves,
                            ),
                            (
                                "high", high_rate, result.high_state,
                                result.core_to_high_moves,
                            ),
                        ):
                            rows.append(_state_record(
                                arm=arm_name,
                                rate=rate,
                                group=group,
                                common_core=result.common_core,
                                states=state,
                                moves=moves,
                                geometry=token_geometry,
                                selector_weights=np.asarray(
                                    group["selector_weights"],
                                ),
                                d1_logits=replay.logits(state),
                                d1_candidate_ids=replay.ids(state),
                                target_logits=context.target_logits,
                                target_ids=context.target_ids,
                                parameter=_policy_parameter(
                                    result,
                                    repair_window=int(window),
                                    eta=float(eta),
                                    endpoint=endpoint,
                                    paired_reference_high_core_seal=paired_reference_seal,
                                ),
                            ))
        emitted = len(rows) - first_row
        if emitted != expected_per_group:
            raise RuntimeError(
                f"candidate row grid changed: expected {expected_per_group}, got {emitted}"
            )
        replay_executions += replay.executions
        replay_hits += replay.hits
        policy_memo_stats = policy_memo.stats()
        policy_memo_audit = _deterministic_policy_memo_audit(
            policy_memo_stats,
        )
        geometry_cache_stats = token_geometry.cache_stats()
        parity_rows.append({
            "request_id": request_id,
            "layer": layer,
            **context.parity,
            "computed_vjp_sensitivity_rows": int(context.candidate_ids.size),
            "expanded_d1_sensitivity_rows": 256,
            "exact_replay_unique_states": int(replay.states),
            "exact_replay_executions": int(replay.executions),
            "exact_replay_cache_hits": int(replay.hits),
            **{f"policy_memo_{key}": value for key, value in policy_memo_audit.items()},
            **{f"geometry_cache_{key}": value for key, value in geometry_cache_stats.items()},
        })
        print(
            f"[nested allocation] layer={layer} group={group['group'] + 1}/"
            f"{len(groups)} rows={emitted} replays={replay.executions} "
            f"cache_hits={replay.hits} "
            f"high_path_cache={policy_memo_stats['high_path_hits']}/"
            f"{policy_memo_stats['high_path_misses']}",
            flush=True,
        )
        profile.add(
            "group_total",
            time.perf_counter() - group_started if profile.enabled else 0.0,
            labels=group_profile_labels,
        )

        del context, replay, boundaries
        torch.cuda.empty_cache()

    expected_rows = expected_per_group * len(groups)
    if len(rows) != expected_rows:
        raise RuntimeError("final candidate row count differs from immutable grid")
    layer_dir = args.output_dir / args.split / f"layer_{layer:02d}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    atomic_pickle(layer_dir / "nested_candidates.pkl", rows)
    summary_rows = [{
        **{
            key: value for key, value in row.items()
            if key not in {
                "common_core_states", "selected_states", "moves", "parameter",
                "d1_candidate_logits", "selector_weights", "execution_weights",
            }
        },
        "parameter_json": json.dumps(
            row["parameter"], sort_keys=True, separators=(",", ":"),
        ),
    } for row in rows]
    atomic_parquet(layer_dir / "nested_candidate_metrics.parquet", summary_rows)
    atomic_parquet(layer_dir / "nested_slice_parity.parquet", parity_rows)
    rows_by_arm = {
        arm: sum(row["arm"] == arm for row in rows)
        for arm in (ARM_PR13, ARM_STRICT, ARM_LOCAL, ARM_D1, ARM_D1_SEVERITY)
    }
    facts = {
        "completed": True,
        "schema": SCHEMA,
        "split": args.split,
        "layer": layer,
        "run_id": str(config["run_id"]),
        "run_identity_sha256": str(args.run_identity_sha256),
        "input_pins": dict(args.input_pins),
        "captured_route_cuda_preflight": dict(args.captured_route_cuda_preflight),
        "atomic_resume_protocol": "facts_last_exact_three_file_inventory_v1",
        "paired_reference_high_core_sealed": True,
        "pilot_nonpromotable": bool(args.pilot),
        "completed_for_sealing": not bool(args.pilot),
        "partial_request_cohort": bool(args.pilot),
        "requests": len(groups),
        "request_ids": [str(row["request_id"]) for row in request_rows],
        "repair_windows": list(config_windows),
        "eta_grid": list(config_etas),
        "evaluated_policy_grid": {
            f"{low}_to_{high}": {
                str(window): list(map(float, etas))
                for window, etas in windows.items()
            }
            for (low, high), windows in policy_grid.items()
        },
        "rates": rates,
        "rows": len(rows),
        "expected_rows": expected_rows,
        "expected_rows_per_group": expected_per_group,
        "rows_by_arm": rows_by_arm,
        "high_guardrail_audit": _high_guardrail_audit(rows),
        "exact_replay_executions": replay_executions,
        "exact_replay_cache_hits": replay_hits,
        "allocation_uses_terminal_or_downstream_outcome": False,
        "candidate_execution_stops_at_d1_router": True,
        "arm3_precomputed_before_gpu_d1": True,
        "pr13_and_exact_local_collapsed_in_workers": True,
        "frozen_calibration": (
            None if frozen_facts is None else {
                **frozen_facts,
                "path": str(args.frozen_calibration.resolve()),
                "file_sha256": sha256(args.frozen_calibration),
            }
        ),
        "wall_seconds": time.perf_counter() - started,
        "files": {
            name: _assert_file_pin(layer_dir / name, sha256(layer_dir / name), name)
            for name in SCIENTIFIC_LAYER_FILES
        },
        "environment": {
            "host": platform.node(),
            **dict(args.hardware_runtime),
        },
    }
    atomic_json(layer_dir / "nested_layer_facts.json", facts)
    profile_path = _write_profile_sidecar(
        profile,
        args.output_dir,
        split=args.split,
        layer=layer,
        requests=len(groups),
        config=config,
    )
    if profile_path is not None:
        print(
            f"[nested profile] wrote {profile_path}",
            flush=True,
        )
    del model_slice, geometries, rate_state_grid, arm3_grid
    gc.collect()
    torch.cuda.empty_cache()
    return facts


def _expected_layer_contract(
    args: argparse.Namespace,
    layer: int,
) -> dict[str, Any]:
    config = args.authenticated_config
    frozen_raw = (
        None if args.frozen_calibration is None
        else load_json(args.frozen_calibration)
    )
    policy_grid, frozen_facts = _policy_grid(config, args.split, frozen_raw)
    windows, etas = _config_windows_etas(config)
    rates = sorted({
        int(rate)
        for pair in config["traffic_pairs"]
        for rate in (
            pair["metadata_matched_pages_per_expert"],
            pair["pr13_reference_pages_per_expert"],
        )
    })
    per_group = _expected_rows_per_group(rates, policy_grid)
    requests = list(args.authenticated_request_rows)
    frozen_expected = (
        None if frozen_facts is None else {
            **frozen_facts,
            "path": str(args.frozen_calibration.resolve()),
            "file_sha256": sha256(args.frozen_calibration),
        }
    )
    return {
        "completed": True,
        "schema": SCHEMA,
        "split": str(args.split),
        "layer": int(layer),
        "run_id": str(config["run_id"]),
        "run_identity_sha256": str(args.run_identity_sha256),
        "input_pins": dict(args.input_pins),
        "captured_route_cuda_preflight": dict(args.captured_route_cuda_preflight),
        "atomic_resume_protocol": "facts_last_exact_three_file_inventory_v1",
        "paired_reference_high_core_sealed": True,
        "pilot_nonpromotable": bool(args.pilot),
        "completed_for_sealing": not bool(args.pilot),
        "partial_request_cohort": bool(args.pilot),
        "requests": len(requests),
        "request_ids": [str(row["request_id"]) for row in requests],
        "repair_windows": list(windows),
        "eta_grid": list(etas),
        "evaluated_policy_grid": {
            f"{low}_to_{high}": {
                str(window): list(map(float, values))
                for window, values in grid.items()
            }
            for (low, high), grid in policy_grid.items()
        },
        "rates": rates,
        "expected_rows": per_group * len(requests),
        "expected_rows_per_group": per_group,
        "allocation_uses_terminal_or_downstream_outcome": False,
        "candidate_execution_stops_at_d1_router": True,
        "frozen_calibration": frozen_expected,
    }


def _validate_completed_layer(
    args: argparse.Namespace,
    layer: int,
) -> dict[str, Any] | None:
    layer_dir = args.output_dir / args.split / f"layer_{int(layer):02d}"
    facts_path = layer_dir / "nested_layer_facts.json"
    allowed_partial = {
        *SCIENTIFIC_LAYER_FILES,
        *(f"{name}.tmp" for name in SCIENTIFIC_LAYER_FILES),
        "nested_layer_facts.json.tmp",
    }
    if not facts_path.is_file():
        if layer_dir.is_dir():
            unexpected = {
                path.name for path in layer_dir.iterdir()
                if path.is_file() and path.name not in allowed_partial
            }
            if unexpected:
                raise RuntimeError(
                    f"incomplete layer {layer} contains unexpected files: "
                    f"{sorted(unexpected)}"
                )
        return None

    actual_names = {
        path.name for path in layer_dir.iterdir() if path.is_file()
    }
    expected_names = {*SCIENTIFIC_LAYER_FILES, facts_path.name}
    if actual_names != expected_names:
        raise RuntimeError(
            f"completed layer {layer} has a noncanonical file inventory"
        )
    facts = load_json(facts_path)
    expected = _expected_layer_contract(args, layer)
    for key, value in expected.items():
        if facts.get(key) != value:
            raise RuntimeError(
                f"completed layer {layer} resume identity changed: {key}"
            )
    if (
        facts.get("rows") != expected["expected_rows"]
        or facts.get("expected_rows") != expected["expected_rows"]
        or not isinstance(facts.get("rows_by_arm"), Mapping)
        or sum(map(int, facts["rows_by_arm"].values())) != expected["expected_rows"]
    ):
        raise RuntimeError(f"completed layer {layer} candidate row grid changed")
    files = facts.get("files")
    if not isinstance(files, Mapping) or set(files) != set(SCIENTIFIC_LAYER_FILES):
        raise RuntimeError(f"completed layer {layer} file ledger changed")
    for name in SCIENTIFIC_LAYER_FILES:
        record = files[name]
        if not isinstance(record, Mapping):
            raise RuntimeError(f"completed layer {layer} file record changed")
        _assert_file_pin(
            layer_dir / name,
            record.get("sha256"),
            f"completed layer {layer} {name}",
            expected_bytes=record.get("bytes"),
        )
    environment = facts.get("environment")
    if (
        not isinstance(environment, Mapping)
        or {
            key: value for key, value in environment.items() if key != "host"
        } != dict(args.hardware_runtime)
    ):
        raise RuntimeError(f"completed layer {layer} runtime identity changed")
    return facts


def _complete_run_payload(
    args: argparse.Namespace,
    layers: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "completed": True,
        "schema": SCHEMA,
        "split": str(args.split),
        "run_id": str(args.authenticated_config["run_id"]),
        "run_identity_sha256": str(args.run_identity_sha256),
        "run_identity": dict(args.run_identity),
        "pilot_nonpromotable": bool(args.pilot),
        "completed_for_sealing": not bool(args.pilot),
        "input_pins": dict(args.input_pins),
        "captured_route_cuda_preflight": dict(args.captured_route_cuda_preflight),
        "atomic_resume_protocol": (
            "pilot_nonpromotable_facts_last_requested_layer_merge_v1"
            if args.pilot else
            "validated_facts_last_per_layer_exact_six_layer_merge_v1"
        ),
        "layers": dict(layers),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "test_rows_admitted_or_used": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-facts", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trees", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--factor-manifest", type=Path, required=True)
    parser.add_argument("--pr13-config", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--capture-facts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("calibration", "evaluation"), required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--repair-windows", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--request-limit", type=int)
    parser.add_argument("--frozen-calibration", type=Path)
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="run a partial, isolated, explicitly nonpromotable pilot",
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--profile-timings",
        action="store_true",
        help="write a separate non-scientific internal timing sidecar",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.request_limit is not None and args.request_limit < 1:
        raise ValueError("request-limit must be positive")
    if args.workers < 1 or args.workers > 64:
        raise ValueError("workers must lie in [1,64]")
    if any(window < 0 for window in args.repair_windows):
        raise ValueError("repair windows must be nonnegative")
    if args.split == "calibration" and args.frozen_calibration is not None:
        raise ValueError("calibration split rejects --frozen-calibration")
    if args.split == "evaluation" and args.frozen_calibration is None:
        raise ValueError("evaluation split requires --frozen-calibration")

    (
        config,
        pr13_config,
        request_rows,
        input_pins,
        factor_paths,
        run_facts,
    ) = _authenticate_allocation_inputs(args)
    configured_layers = tuple(map(int, config["injection_layers"]))
    requested_layers = tuple(map(int, args.layers))
    if (
        len(set(requested_layers)) != len(requested_layers)
        or not set(requested_layers).issubset(configured_layers)
    ):
        raise ValueError("CLI layers must be unique configured injection layers")
    target_layers = requested_layers if args.pilot else configured_layers
    if tuple(map(int, args.repair_windows)) != _config_windows_etas(config)[0]:
        raise ValueError("CLI repair windows differ from the immutable config grid")

    args.authenticated_config = config
    args.authenticated_pr13_config = pr13_config
    args.authenticated_request_rows = request_rows
    args.authenticated_factor_paths = factor_paths
    args.input_pins = input_pins
    args.run_identity = run_facts["run_identity"]
    args.run_identity_sha256 = run_facts["run_identity_sha256"]
    args.hardware_runtime = run_facts["hardware_runtime"]
    args.captured_route_cuda_preflight = run_facts["captured_route_cuda_preflight"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for layer in requested_layers:
        existing = _validate_completed_layer(args, layer)
        if existing is not None:
            print(
                f"[nested resume] reused split={args.split} layer={layer}",
                flush=True,
            )
            continue
        run_layer(args, layer)

    completed: dict[str, dict[str, Any]] = {}
    for layer in target_layers:
        layer_facts = _validate_completed_layer(args, layer)
        if layer_facts is not None:
            completed[str(layer)] = layer_facts
    global_path = args.output_dir / f"nested_run_facts_{args.split}.json"
    if len(completed) != len(target_layers):
        if global_path.exists():
            raise RuntimeError(
                "completed global run facts coexist with an incomplete layer grid"
            )
        pending = [
            layer for layer in target_layers if str(layer) not in completed
        ]
        print(
            f"[nested resume] split={args.split} pending_layers={pending}",
            flush=True,
        )
        return

    payload = _complete_run_payload(args, completed)
    if global_path.is_file():
        if load_json(global_path) != payload:
            raise RuntimeError("existing completed run facts changed sealed merge identity")
        print(f"[nested resume] reused {global_path}", flush=True)
        return
    atomic_json(global_path, payload)


if __name__ == "__main__":
    main()
