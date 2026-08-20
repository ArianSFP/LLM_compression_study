#!/usr/bin/env python3
"""Exact-H4 ceiling for block-exact low-rank MXFP4 interaction fields."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(EXPERIMENT / "scripts")]

from oracle_study.interaction_field import (  # noqa: E402
    EncodedInteractionFactor,
    JointInteractionFactor,
    build_interaction_field,
    encode_interaction_factor,
    factor_exact_proxy_plus_tail,
    factor_joint_gram,
    field_coordinate_descent,
    field_forward_greedy,
    field_local_search,
    make_encoded_factor_self_safe,
    relaxed_field_descent,
    round_relaxed_pages,
    truncate_interaction_factor,
)
from oracle_study.neuron_selector import (  # noqa: E402
    complete_unit_scores,
    factorized_unit_outputs,
    unit_score_metadata,
)
from oracle_study.set_utility_oracles import (  # noqa: E402
    STATE_GD,
    STATE_PAGE_COSTS,
    hybrid_state_output,
)
from oracle_study.unit_set_teacher import down_metric_gram  # noqa: E402
from run_mxfp4_selective_pages import tree_from_record  # noqa: E402
import run_set_utility_distillation as prior  # noqa: E402
import run_sparse_streaming_study as base  # noqa: E402


IDENTITY = (
    "capture_source", "evaluation_split", "request_id", "position", "layer", "expert_id",
)
FRONTIER = "interaction_field_frontier.parquet"
MANIFEST = "interaction_field_factor_manifest.json"
FACTS = "run_facts.json"
ABC_ESTABLISHED_BYTES = 3_084
UNITS = 512
COMPUTE_COMPONENTS = (
    "field_build",
    "independent_seed",
    "forward",
    "relaxation_setup",
    "relaxation_iterations",
    "coordinate",
    "local",
)

_EVALUATION_CONTEXT: dict[str, Any] | None = None


def _atomic_json(path: Path, value: Any) -> None:
    prior.atomic_json(path, value)


def _cpu_capacity_facts() -> dict[str, Any]:
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    raw = cpu_max.read_text().strip() if cpu_max.exists() else "unavailable"
    quota_cores: float | None = None
    parts = raw.split()
    if len(parts) == 2 and parts[0] != "max":
        quota_cores = float(parts[0]) / float(parts[1])
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return {
        "cpu_max": raw,
        "quota_cores": quota_cores,
        "affinity_logical_cpus": affinity,
        "os_cpu_count": os.cpu_count(),
    }


def _worker_thread_environment() -> dict[str, str | None]:
    return {
        name: os.environ.get(name)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    }


def _atomic_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    temporary.replace(path)


def _factor_id(family: str, tail_rank: int, exact_rank: int, encoding: str) -> str:
    return f"{family}_tail{int(tail_rank)}_exact{int(exact_rank)}_{encoding}"


def _validate_contract(config: Mapping[str, Any], pr10: Mapping[str, Any]) -> None:
    if config.get("base_pr10_commit") != "54ebf5b64ec04c4605222a8f96d04d0f61e1dcdd":
        raise RuntimeError("interaction study is not stacked on the frozen PR #10 head")
    if config.get("fit_split") != "train" or config.get("selection_split") != "validation":
        raise RuntimeError("interaction study split contract changed")
    if config.get("no_test_rows_admitted") is not True:
        raise RuntimeError("test-row admission must remain forbidden")
    if list(map(int, config.get("page_budgets", []))) != [384, 576, 768]:
        raise RuntimeError("page budget grid changed")
    if int(config.get("expected_validation_unique_invocations", -1)) != int(
        pr10["expected_validation_unique_invocations"]
    ):
        raise RuntimeError("validation invocation contract changed")
    if int(config.get("expected_validation_layer_expert_cells", -1)) != int(
        pr10["expected_validation_layer_expert_cells"]
    ):
        raise RuntimeError("validation cell contract changed")
    if list(config.get("deployment_seed_basins", [])) != [
        "all_00", "independent_abc",
    ]:
        raise RuntimeError("deployment seed contract changed")
    if list(config.get("geometry_seed_basins", [])) != [
        "all_00", "independent_abc", "compressed_residual_forward",
        "continuous_relaxation_round",
    ]:
        raise RuntimeError("geometry seed contract changed")
    if (
        config.get("primary_solver_id") != "two_seed_coordinate_plus_local_repair"
        or config.get("geometry_solver_id") != "four_seed_coordinate_plus_local_repair"
    ):
        raise RuntimeError("solver identity contract changed")
    if int(config.get("evaluation_workers", -1)) != 24:
        raise RuntimeError("evaluation worker contract changed")
    if config.get("worker_start_method") != "fork":
        raise RuntimeError("worker start-method contract changed")
    if int(config.get("worker_blas_threads", -1)) != 1:
        raise RuntimeError("worker BLAS-thread contract changed")
    if int(config.get("abc_metadata_bytes_per_expert", -1)) != ABC_ESTABLISHED_BYTES:
        raise RuntimeError("established PR #10 A/B/C payload changed")


def _load_pr10_baselines(path: Path, config: Mapping[str, Any]) -> dict[tuple[Any, ...], float]:
    frame = pd.read_parquet(path)
    families = {
        "hybrid_forward_plus_1_2_3_unit_local_search": "exact_hybrid",
        "coherent_exact_set_fixed_greedy_teacher": "coherent_exact_set",
        "coherent_independent_abc_pr9_control": "pr9_independent",
    }
    frame = frame[frame["selector_family"].isin(families)].copy()
    frame["baseline"] = frame["selector_family"].map(families)
    expected = int(config["expected_validation_unique_invocations"]) * 3 * len(families)
    if len(frame) != expected:
        raise RuntimeError(f"PR #10 baseline rows {len(frame)} != {expected}")
    result: dict[tuple[Any, ...], float] = {}
    for row in frame.to_dict("records"):
        identity = tuple(row[name] for name in IDENTITY)
        key = identity + (int(round(float(row["physical_budget_bpw"]) * 768)), row["baseline"])
        if key in result:
            raise RuntimeError("duplicate PR #10 baseline identity")
        recovery = float(row["recovery"])
        if not np.isfinite(recovery):
            raise RuntimeError("non-finite PR #10 baseline")
        result[key] = recovery
    return result


def _store_decoded(
    arrays: dict[str, np.ndarray], prefix: str, factor: JointInteractionFactor,
) -> dict[str, Any]:
    l4_key, l2_key = f"{prefix}__l4", f"{prefix}__l2"
    arrays[l4_key] = np.asarray(factor.l4, np.float32)
    arrays[l2_key] = np.asarray(factor.l2, np.float32)
    return {
        "storage_kind": "decoded_fp32_geometry_ceiling",
        "l4_key": l4_key,
        "l2_key": l2_key,
        "factor_payload_bytes": int(arrays[l4_key].nbytes + arrays[l2_key].nbytes),
        "self_safe_global_scale": 1.0,
    }


def _store_encoded(
    arrays: dict[str, np.ndarray], prefix: str, encoded: EncodedInteractionFactor,
) -> dict[str, Any]:
    code_key, scale_key, global_key = (
        f"{prefix}__packed_codes", f"{prefix}__row_scales", f"{prefix}__global_scale",
    )
    arrays[code_key] = np.asarray(encoded.packed_codes, np.uint8)
    arrays[scale_key] = np.asarray(encoded.row_scales, np.float16)
    arrays[global_key] = np.asarray([encoded.global_scale], np.float32)
    physical = arrays[code_key].nbytes + arrays[scale_key].nbytes + arrays[global_key].nbytes
    if physical != encoded.payload_bytes:
        raise RuntimeError("encoded interaction payload accounting changed")
    return {
        "storage_kind": "physically_packed_signed_rows",
        "packed_codes_key": code_key,
        "row_scales_key": scale_key,
        "global_scale_key": global_key,
        "factor_payload_bytes": int(physical),
        "self_safe_global_scale": float(encoded.global_scale),
    }


def _fit_cell_factors(
    q2: tuple[np.ndarray, ...], q4: tuple[np.ndarray, ...], proxy: np.ndarray,
    beta: float, config: Mapping[str, Any], layer: int, expert: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    maximum = max(map(int, config["tail_ranks"]))
    gram = down_metric_gram(q2[2], q4[2], proxy=proxy, beta=beta, dtype=np.float64)
    abc = unit_score_metadata(q2[2], q4[2], proxy=proxy, beta=beta)
    fitted = {
        "joint_eigh": factor_joint_gram(gram, maximum, method="eigh", dtype=np.float64),
        "joint_pivoted_cholesky": factor_joint_gram(
            gram, maximum, method="pivoted_cholesky", dtype=np.float64,
        ),
        "exact_proxy_plus_eigh_tail": factor_exact_proxy_plus_tail(
            q2[2], q4[2], proxy, beta, maximum, method="eigh", dtype=np.float64,
        ),
    }
    arrays: dict[str, np.ndarray] = {}
    entries: list[dict[str, Any]] = []
    for family in config["geometry_factor_families"]:
        ranks = (
            config["exact_proxy_tail_ranks"]
            if family == "exact_proxy_plus_eigh_tail"
            else config["tail_ranks"]
        )
        for tail_rank in map(int, ranks):
            factor = truncate_interaction_factor(fitted[family], tail_rank, dtype=np.float32)
            config_id = _factor_id(family, tail_rank, factor.exact_rank, "fp32")
            prefix = f"l{layer}_e{expert}_{config_id}"
            storage = _store_decoded(arrays, prefix, factor)
            entries.append({
                "layer": layer, "expert_id": expert, "factor_config_id": config_id,
                "factor_family": family, "tail_rank": tail_rank,
                "exact_rank": factor.exact_rank, "total_rank": factor.rank,
                "encoding": "fp32", "hadamard_rotated": False,
                "geometry_ceiling": True, "quantized_deployable_sidecar": False,
                **storage,
            })
    for spec in config["quantized_factor_specs"]:
        family = str(spec["source"])
        tail_rank = int(spec["tail_rank"])
        factor = truncate_interaction_factor(fitted[family], tail_rank, dtype=np.float64)
        encoded = encode_interaction_factor(
            factor, str(spec["encoding"]), hadamard_rotate=bool(spec["hadamard"]),
        )
        encoded, shrink = make_encoded_factor_self_safe(encoded, abc)
        suffix = str(spec["encoding"]) + ("_hadamard" if spec["hadamard"] else "")
        config_id = _factor_id(family, tail_rank, factor.exact_rank, suffix)
        prefix = f"l{layer}_e{expert}_{config_id}"
        storage = _store_encoded(arrays, prefix, encoded)
        entries.append({
            "layer": layer, "expert_id": expert, "factor_config_id": config_id,
            "factor_family": family, "tail_rank": tail_rank,
            "exact_rank": factor.exact_rank, "total_rank": factor.rank,
            "encoding": str(spec["encoding"]),
            "hadamard_rotated": bool(spec["hadamard"]),
            "geometry_ceiling": False, "quantized_deployable_sidecar": True,
            "self_safe_shrink_unrounded": float(shrink), **storage,
        })
    identifiers = [entry["factor_config_id"] for entry in entries]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("duplicate factor configuration ID")
    return arrays, entries


def _load_factor(arrays: Mapping[str, np.ndarray], entry: Mapping[str, Any]) -> JointInteractionFactor:
    if entry["storage_kind"] == "decoded_fp32_geometry_ceiling":
        return JointInteractionFactor(
            l4=np.asarray(arrays[entry["l4_key"]], np.float32),
            l2=np.asarray(arrays[entry["l2_key"]], np.float32),
            method=str(entry["factor_family"]), tail_rank=int(entry["tail_rank"]),
            exact_rank=int(entry["exact_rank"]), encoding="fp32",
            storage_bytes=int(entry["factor_payload_bytes"]),
        )
    encoded = EncodedInteractionFactor(
        packed_codes=np.asarray(arrays[entry["packed_codes_key"]], np.uint8),
        row_scales=np.asarray(arrays[entry["row_scales_key"]], np.float16),
        units=UNITS, rank=int(entry["total_rank"]), method=str(entry["factor_family"]),
        tail_rank=int(entry["tail_rank"]), exact_rank=int(entry["exact_rank"]),
        encoding=str(entry["encoding"]) + ("_hadamard" if entry["hadamard_rotated"] else ""),
        global_scale=np.float32(np.asarray(arrays[entry["global_scale_key"]]).reshape(-1)[0]),
    )
    if encoded.payload_bytes != int(entry["factor_payload_bytes"]):
        raise RuntimeError("loaded factor payload changed")
    return encoded.decode()


def _exact_damage(outputs: Any, states: np.ndarray, proxy: np.ndarray, beta: float) -> float:
    residual = np.asarray(outputs.target_output, np.float64) - hybrid_state_output(outputs, states)
    return float(base.numpy_qenergy(residual, proxy=proxy, beta=beta))


def _row(
    metadata: Mapping[str, Any], entry: Mapping[str, Any], solver: str,
    states: np.ndarray, pages: int, predicted_damage: float, exact_damage: float,
    base_damage: float, baselines: Mapping[str, float], runtime_seconds: float,
    selector_macs: int, compute_bound: str, seed: str,
    compute_components: Mapping[str, int],
    forward_candidate_evaluations: int,
    relaxation_iterations: int,
    coordinate_sweeps: int,
    local_evaluated_passes: int,
) -> dict[str, Any]:
    components = {
        name: int(compute_components.get(name, 0)) for name in COMPUTE_COMPONENTS
    }
    if set(compute_components) - set(COMPUTE_COMPONENTS):
        raise ValueError("unknown selector-compute component")
    if any(value < 0 for value in components.values()):
        raise ValueError("selector-compute components must be nonnegative")
    if sum(components.values()) != int(selector_macs):
        raise ValueError("selector-compute components do not sum to the total")
    recovery = 1.0 - exact_damage / base_damage
    teacher = baselines["exact_hybrid"]
    return {
        **metadata, **{key: entry[key] for key in (
            "factor_config_id", "factor_family", "tail_rank", "exact_rank", "total_rank",
            "encoding", "hadamard_rotated", "geometry_ceiling",
            "quantized_deployable_sidecar", "factor_payload_bytes", "self_safe_global_scale",
        )},
        "solver": solver, "selected_seed": seed, "physical_pages": int(pages),
        "physical_budget_pages": int(metadata["physical_budget_pages"]),
        "recovery": recovery,
        "set_gain_retention_vs_pr10_exact_hybrid": recovery / teacher,
        "recovery_gap_vs_pr10_exact_hybrid": recovery - teacher,
        "pr10_exact_hybrid_recovery": teacher,
        "pr10_coherent_exact_set_recovery": baselines["coherent_exact_set"],
        "pr10_independent_recovery": baselines["pr9_independent"],
        "compressed_predicted_damage": float(predicted_damage),
        "exact_qenergy_damage": float(exact_damage),
        "damage_prediction_error_over_base": (predicted_damage - exact_damage) / base_damage,
        "selector_runtime_ms": 1000.0 * runtime_seconds,
        "selector_compute_macs": int(selector_macs),
        **{
            f"selector_{name}_compute_macs": value
            for name, value in components.items()
        },
        "selector_compute_accounting_bound": compute_bound,
        "forward_candidate_evaluations": int(forward_candidate_evaluations),
        "relaxation_iterations": int(relaxation_iterations),
        "coordinate_sweeps": int(coordinate_sweeps),
        "local_evaluated_passes": int(local_evaluated_passes),
        "selector_non_mac_operations": (
            "state_scalar_additions_projection_simplex_sort_exact_page_DP_round_"
            "topk_and_control_flow_excluded_from_analytical_MAC_metric"
        ),
        "abc_metadata_bytes": ABC_ESTABLISHED_BYTES,
        "combined_metadata_bytes": int(entry["factor_payload_bytes"]) + ABC_ESTABLISHED_BYTES,
        "combined_metadata_bpw": 8.0 * (
            int(entry["factor_payload_bytes"]) + ABC_ESTABLISHED_BYTES
        ) / 3_145_728,
        "uses_exact_h4": True, "uses_exact_abc_self_terms": True,
        "selection_regime": "h0_exact_h4_geometry_ceiling",
    }


def _evaluate_factor(
    outputs: Any, abc: Any, factor: JointInteractionFactor, entry: Mapping[str, Any],
    metadata: Mapping[str, Any], proxy: np.ndarray, beta: float,
    baseline_map: Mapping[tuple[Any, ...], float], config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    field = build_interaction_field(factor, outputs.h2, outputs.h4, abc)
    rank = factor.rank
    relaxation_step = 1.0 / max(
        2.0 * float(np.einsum("usr,usr->", field.rho, field.rho, optimize=True)),
        1.0,
    )
    budgets = list(map(int, config["page_budgets"]))
    started = time.perf_counter()
    forward = field_forward_greedy(field, budgets)
    forward_seconds = time.perf_counter() - started
    all_00 = np.zeros(UNITS, np.int64)
    base_damage = _exact_damage(outputs, all_00, proxy, beta)
    identity = tuple(metadata[name] for name in IDENTITY)
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        baseline = {
            name: float(baseline_map[identity + (budget, name)])
            for name in ("exact_hybrid", "coherent_exact_set", "pr9_independent")
        }
        count = min(UNITS, budget // 3)
        independent = all_00.copy()
        order = np.lexsort((
            np.arange(UNITS), -complete_unit_scores(outputs.h2, outputs.h4, abc),
        ))
        independent[order[:count]] = STATE_GD
        relaxed_started = time.perf_counter()
        relaxed = relaxed_field_descent(
            field, budget, max_iterations=int(config["relaxation_iterations"]),
            tolerance=float(config["relaxation_tolerance"]), step_size=relaxation_step,
        )
        relaxed_states = round_relaxed_pages(relaxed.probabilities, budget)
        relaxed_seconds = time.perf_counter() - relaxed_started
        forward_state = forward.snapshots[budget].states
        seeds = {
            "all_00": all_00,
            "compressed_residual_forward": forward_state,
            "continuous_relaxation_round": relaxed_states,
            "independent_abc": independent,
        }
        cheap_started = time.perf_counter()
        cheap_traces = {
            name: field_coordinate_descent(
                field, seeds[name], budget,
                max_sweeps=int(config["coordinate_sweeps"]),
            )
            for name in ("all_00", "independent_abc")
        }
        cheap_seed_name, cheap_coordinate = min(
            cheap_traces.items(), key=lambda item: (item[1].damage, item[0]),
        )
        cheap_coordinate_seconds = time.perf_counter() - cheap_started
        cheap_local_started = time.perf_counter()
        cheap_repaired = field_local_search(
            field, cheap_coordinate.states, budget,
            shortlist_size=int(config["local_shortlist"]),
            max_swap_units=int(config["local_swap_units"]),
            max_passes=int(config["local_max_passes"]), page_balanced=False,
        )
        cheap_local_seconds = time.perf_counter() - cheap_local_started
        coordinate_started = time.perf_counter()
        traces = {
            name: field_coordinate_descent(
                field, seed, budget, max_sweeps=int(config["coordinate_sweeps"]),
            )
            for name, seed in sorted(seeds.items())
        }
        seed_name, coordinate = min(
            traces.items(), key=lambda item: (item[1].damage, item[0]),
        )
        coordinate_seconds = time.perf_counter() - coordinate_started
        local_started = time.perf_counter()
        repaired = field_local_search(
            field, coordinate.states, budget,
            shortlist_size=int(config["local_shortlist"]),
            max_swap_units=int(config["local_swap_units"]),
            max_passes=int(config["local_max_passes"]), page_balanced=False,
        )
        local_seconds = time.perf_counter() - local_started
        field_build_macs = 8 * UNITS * rank + 12 * UNITS
        independent_seed_macs = 6 * UNITS
        relaxation_setup_macs = 4 * UNITS * rank
        forward_macs = 2 * rank * int(forward.candidate_evaluations)
        relaxation_macs = 8 * UNITS * rank * int(relaxed.iterations)
        coordinate_sweeps = sum(int(trace.sweeps) for trace in traces.values())
        coordinate_macs = (
            2 * UNITS * rank * coordinate_sweeps
            + 8 * UNITS * coordinate_sweeps
        )
        evaluated_local_passes = min(
            int(config["local_max_passes"]), int(repaired.passes) + 1,
        )
        maximum_shortlist = 7 * int(config["local_shortlist"])
        local_macs = evaluated_local_passes * (
            6 * UNITS * rank + maximum_shortlist * maximum_shortlist * rank
        )
        cheap_coordinate_sweeps = sum(
            int(trace.sweeps) for trace in cheap_traces.values()
        )
        cheap_coordinate_macs = (
            2 * UNITS * rank * cheap_coordinate_sweeps
            + 8 * UNITS * cheap_coordinate_sweeps
        )
        cheap_evaluated_local_passes = min(
            int(config["local_max_passes"]), int(cheap_repaired.passes) + 1,
        )
        cheap_local_macs = cheap_evaluated_local_passes * (
            6 * UNITS * rank + maximum_shortlist * maximum_shortlist * rank
        )
        common_metadata = {**metadata, "physical_budget_pages": budget}
        zero_counters = {
            "forward_candidate_evaluations": 0,
            "relaxation_iterations": 0,
            "coordinate_sweeps": 0,
            "local_evaluated_passes": 0,
        }
        control_states = (
            ("compressed_residual_forward", forward_state, forward.snapshots[budget].damage,
             forward_seconds, field_build_macs + forward_macs,
             "complete_field_build_plus_shared_max_budget_forward_upper_bound", "forward",
             {"field_build": field_build_macs, "forward": forward_macs},
             {**zero_counters,
              "forward_candidate_evaluations": int(forward.candidate_evaluations)}),
            ("continuous_relaxation_round", relaxed_states, field.damage(relaxed_states),
             relaxed_seconds, field_build_macs + relaxation_setup_macs + relaxation_macs,
             "Frobenius_safe_step_setup_plus_projected_gradient_logical_MACs", "relaxation",
             {
                 "field_build": field_build_macs,
                 "relaxation_setup": relaxation_setup_macs,
                 "relaxation_iterations": relaxation_macs,
             }, {**zero_counters,
                 "relaxation_iterations": int(relaxed.iterations)}),
            ("four_seed_coordinate", coordinate.states, coordinate.damage,
             forward_seconds + relaxed_seconds + coordinate_seconds,
             field_build_macs + independent_seed_macs + forward_macs
             + relaxation_setup_macs + relaxation_macs + coordinate_macs,
             "complete_field_and_all_seed_construction_plus_two_dot_coordinate_MACs", seed_name,
             {
                 "field_build": field_build_macs,
                 "independent_seed": independent_seed_macs,
                 "forward": forward_macs,
                 "relaxation_setup": relaxation_setup_macs,
                 "relaxation_iterations": relaxation_macs,
                 "coordinate": coordinate_macs,
             }, {
                 **zero_counters,
                 "forward_candidate_evaluations": int(forward.candidate_evaluations),
                 "relaxation_iterations": int(relaxed.iterations),
                 "coordinate_sweeps": coordinate_sweeps,
             }),
        )
        if entry["factor_config_id"] in set(config["solver_control_factor_ids"]):
            for solver, states, predicted, seconds, macs, bound, seed, components, counters in control_states:
                rows.append(_row(
                    common_metadata, entry, solver, states, int(np.sum(STATE_PAGE_COSTS[states])),
                    predicted, _exact_damage(outputs, states, proxy, beta), base_damage,
                    baseline, seconds, macs, bound, seed, components, **counters,
                ))
        cheap_components = {
            "field_build": field_build_macs,
            "independent_seed": independent_seed_macs,
            "coordinate": cheap_coordinate_macs,
            "local": cheap_local_macs,
        }
        rows.append(_row(
            common_metadata, entry, "two_seed_coordinate_plus_local_repair",
            cheap_repaired.states, cheap_repaired.pages, cheap_repaired.damage,
            _exact_damage(outputs, cheap_repaired.states, proxy, beta),
            base_damage, baseline,
            cheap_coordinate_seconds + cheap_local_seconds,
            sum(cheap_components.values()),
            "complete_field_build_plus_two_seed_coordinate_and_conservative_local_upper_bound",
            cheap_seed_name, cheap_components,
            forward_candidate_evaluations=0,
            relaxation_iterations=0,
            coordinate_sweeps=cheap_coordinate_sweeps,
            local_evaluated_passes=cheap_evaluated_local_passes,
        ))
        primary_components = {
            "field_build": field_build_macs,
            "independent_seed": independent_seed_macs,
            "forward": forward_macs,
            "relaxation_setup": relaxation_setup_macs,
            "relaxation_iterations": relaxation_macs,
            "coordinate": coordinate_macs,
            "local": local_macs,
        }
        rows.append(_row(
            common_metadata, entry, "four_seed_coordinate_plus_local_repair",
            repaired.states, repaired.pages, repaired.damage,
            _exact_damage(outputs, repaired.states, proxy, beta), base_damage, baseline,
            forward_seconds + relaxed_seconds + coordinate_seconds + local_seconds,
            sum(primary_components.values()),
            "complete_field_build_plus_conservative_local_upper_bound_and_all_seed_costs",
            seed_name, primary_components,
            forward_candidate_evaluations=int(forward.candidate_evaluations),
            relaxation_iterations=int(relaxed.iterations),
            coordinate_sweeps=coordinate_sweeps,
            local_evaluated_passes=evaluated_local_passes,
        ))
    return rows


def _evaluate_observation_task(
    task: tuple[int, int],
) -> tuple[int, int, list[dict[str, Any]]]:
    """Evaluate one invocation using fork-shared immutable cell state."""
    if _EVALUATION_CONTEXT is None:
        raise RuntimeError("parallel evaluation context was not initialized")
    cell_index, observation_index = map(int, task)
    cell = _EVALUATION_CONTEXT["cells"][cell_index]
    activation, metadata = cell["observations"][observation_index]
    outputs = factorized_unit_outputs(cell["q2"], cell["q4"], activation)
    rows: list[dict[str, Any]] = []
    for entry, factor in cell["factors"]:
        rows.extend(_evaluate_factor(
            outputs, cell["abc"], factor, entry, metadata,
            cell["proxy"], cell["beta"], _EVALUATION_CONTEXT["baselines"],
            _EVALUATION_CONTEXT["config"],
        ))
    return cell_index, observation_index, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pr10-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trees", type=Path, required=True)
    parser.add_argument("--exact-captures", type=Path, required=True)
    parser.add_argument("--pr10-validation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    pr10 = json.loads(args.pr10_config.read_text())
    _validate_contract(config, pr10)
    if prior.sha256(args.pr10_config) != config["pr10_config_sha256"]:
        raise RuntimeError("PR #10 config hash changed")
    hybrid_path = args.pr10_validation_dir / "hybrid_oracle_frontier.parquet"
    pr10_facts_path = args.pr10_validation_dir / "run_facts.json"
    if prior.sha256(hybrid_path) != config["pr10_validation_hybrid_sha256"]:
        raise RuntimeError("PR #10 hybrid evidence hash changed")
    if prior.sha256(pr10_facts_path) != config["pr10_validation_facts_sha256"]:
        raise RuntimeError("PR #10 validation facts hash changed")
    baselines = _load_pr10_baselines(hybrid_path, config)
    audit, hashes = base.verify_locked_inputs(
        pr10, args.pr10_config, args.exact_captures, args.checkpoint, args.trees,
        "exact_checkpoint",
    )
    data = prior.load_capture_admitted(args.exact_captures, ["train", "validation"])
    if "test" in set(map(str, np.unique(data["split"]))):
        raise RuntimeError("test row entered interaction study")
    plan = prior.evaluation_plan(data, pr10)
    scope = prior.audit_validation_plan(plan, pr10)
    worker_environment = _worker_thread_environment()
    expected_threads = str(int(config["worker_blas_threads"]))
    if set(worker_environment.values()) != {expected_threads}:
        raise RuntimeError("worker BLAS thread environment is not frozen")
    cpu_capacity = _cpu_capacity_facts()
    quota = cpu_capacity["quota_cores"]
    if quota is not None and quota < float(config["evaluation_workers"]):
        raise RuntimeError(
            f"CPU quota {quota} is below {config['evaluation_workers']} workers"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    facts_path = args.output / FACTS
    facts = {
        "completed": False, "failures": [], "completed_work_units": [],
        "run_id": config["run_id"], "study_scope": config["study_scope"],
        "selection_split": "validation", "test_rows_admitted": False,
        "test_scientific_values_used": False, "codec_locked": True,
        "config_sha256": prior.sha256(args.config),
        "pr10_config_sha256": prior.sha256(args.pr10_config),
        "pr10_validation_hybrid_sha256": prior.sha256(hybrid_path),
        "pr10_validation_facts_sha256": prior.sha256(pr10_facts_path),
        "capture_sha256": hashes["capture_sha256"], "tree_sha256": hashes["tree_sha256"],
        "checkpoint_config_sha256": hashes["config_sha256"],
        "checkpoint_index_sha256": hashes["index_sha256"], "checkpoint_audit": audit,
        "validation_scope_audit": scope, "runtime_provenance": prior.runtime_provenance(args.device),
        "runner_sha256": prior.sha256(Path(__file__).resolve()),
        "interaction_core_sha256": prior.sha256(
            EXPERIMENT / "src/oracle_study/interaction_field.py"
        ),
        "pr10_runner_sha256": prior.sha256(
            EXPERIMENT / "scripts/run_set_utility_distillation.py"
        ),
        "no_post_hoc_factor_or_solver_selection": True,
        "evaluation_workers": int(config["evaluation_workers"]),
        "worker_start_method": str(config["worker_start_method"]),
        "worker_blas_threads": int(config["worker_blas_threads"]),
        "worker_thread_environment": worker_environment,
        "cpu_capacity": cpu_capacity,
    }
    _atomic_json(facts_path, facts)
    try:
        index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
        tree_records = json.loads(args.trees.read_text())
        trees = {name: tree_from_record(tree_records[name]) for name in prior.PROJECTIONS}
        factor_entries: list[dict[str, Any]] = []
        layer_arrays: dict[int, dict[str, np.ndarray]] = {}
        cell_cache: dict[tuple[int, int], tuple[Any, Any, np.ndarray, float]] = {}
        for layer, expert, _, _, _ in plan:
            local = prior.layer_view(data, layer)
            proxy, proxy_facts = base.proxy_gradients(
                local["xplus"], [local[f"h{h}_router_logits"] for h in range(1, 5)],
                local["split"],
            )
            beta = float(proxy_facts["beta"])
            matrices = prior.decode_expert(args.checkpoint, index, trees, layer, expert)
            q2 = tuple(matrices[name][0] for name in prior.PROJECTIONS)
            q4 = tuple(matrices[name][2] for name in prior.PROJECTIONS)
            arrays, entries = _fit_cell_factors(q2, q4, proxy, beta, config, layer, expert)
            layer_arrays.setdefault(layer, {}).update(arrays)
            factor_entries.extend(entries)
            cell_cache[(layer, expert)] = (matrices, unit_score_metadata(
                q2[2], q4[2], proxy=proxy, beta=beta,
            ), proxy, beta)
        layer_records = {}
        for layer, arrays in sorted(layer_arrays.items()):
            path = args.output / f"interaction_field_factors_layer_{layer}.npz"
            prior.atomic_npz(path, arrays)
            layer_records[str(layer)] = {
                "file": path.name, "sha256": prior.sha256(path),
                "bytes": path.stat().st_size, "array_count": len(arrays),
            }
        manifest = {
            "run_id": config["run_id"], "fit_split": "train",
            "validation_or_test_scientific_values_used": False,
            "config_sha256": facts["config_sha256"],
            "runner_sha256": facts["runner_sha256"],
            "interaction_core_sha256": facts["interaction_core_sha256"],
            "entries": factor_entries, "layers": layer_records,
        }
        manifest_path = args.output / MANIFEST
        _atomic_json(manifest_path, manifest)
        facts["factor_manifest_sha256"] = prior.sha256(manifest_path)
        facts["factor_config_count_per_expert"] = len(factor_entries) // 12
        _atomic_json(facts_path, facts)

        rows: list[dict[str, Any]] = []
        entries_by_cell = {
            (layer, expert): [
                entry for entry in factor_entries
                if int(entry["layer"]) == layer and int(entry["expert_id"]) == expert
            ]
            for layer, expert, *_ in plan
        }
        global _EVALUATION_CONTEXT
        cells: list[dict[str, Any]] = []
        tasks: list[tuple[int, int]] = []
        expected_observations: list[int] = []
        for cell_index, (layer, expert, stratum, records, ranks) in enumerate(plan):
            local = prior.layer_view(data, layer)
            matrices, abc, proxy, beta = cell_cache[(layer, expert)]
            arrays = layer_arrays[layer]
            q2 = tuple(matrices[name][0] for name in prior.PROJECTIONS)
            q4 = tuple(matrices[name][2] for name in prior.PROJECTIONS)
            observations = []
            for record, router_rank in zip(records, ranks):
                metadata = base.common_metadata(
                    "exact_checkpoint", "validation", local, int(record), int(router_rank),
                    layer, expert, stratum,
                )
                observations.append((
                    np.asarray(local["x"][record], np.float32),
                    metadata,
                ))
            factors = tuple(
                (entry, _load_factor(arrays, entry))
                for entry in entries_by_cell[(layer, expert)]
            )
            cells.append({
                "layer": layer, "expert": expert, "q2": q2, "q4": q4,
                "abc": abc, "proxy": proxy, "beta": beta,
                "factors": factors, "observations": tuple(observations),
            })
            expected_observations.append(len(observations))
            tasks.extend((cell_index, index) for index in range(len(observations)))

        _EVALUATION_CONTEXT = {
            "cells": tuple(cells), "baselines": baselines, "config": config,
        }
        facts["parallel_task_count"] = len(tasks)
        facts["effective_evaluation_workers"] = min(
            int(config["evaluation_workers"]), len(tasks),
        )
        _atomic_json(facts_path, facts)
        pending: list[dict[int, list[dict[str, Any]]]] = [dict() for _ in cells]
        next_cell = 0
        worker_count = int(facts["effective_evaluation_workers"])
        context = mp.get_context(str(config["worker_start_method"]))
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
            futures = {
                executor.submit(_evaluate_observation_task, task): task
                for task in tasks
            }
            for future in as_completed(futures):
                cell_index, observation_index, cell_rows = future.result()
                expected_task = futures[future]
                if (cell_index, observation_index) != expected_task:
                    raise RuntimeError("parallel worker returned the wrong task identity")
                if observation_index in pending[cell_index]:
                    raise RuntimeError("parallel worker returned a duplicate task")
                pending[cell_index][observation_index] = cell_rows
                while next_cell < len(cells):
                    expected_count = expected_observations[next_cell]
                    if len(pending[next_cell]) != expected_count:
                        break
                    if set(pending[next_cell]) != set(range(expected_count)):
                        raise RuntimeError("parallel cell observation grid changed")
                    for index in range(expected_count):
                        rows.extend(pending[next_cell][index])
                    layer, expert = cells[next_cell]["layer"], cells[next_cell]["expert"]
                    work_unit = f"validation:{layer}:{expert}"
                    facts["completed_work_units"].append(work_unit)
                    _atomic_parquet(args.output / FRONTIER, rows)
                    facts["frontier_rows"] = len(rows)
                    _atomic_json(facts_path, facts)
                    gc.collect()
                    next_cell += 1

        _EVALUATION_CONTEXT = None
        if next_cell != len(cells):
            raise RuntimeError("parallel evaluation did not close every cell")
        expected_primary = int(config["expected_validation_unique_invocations"]) * 3 * (
            len(config["tail_ranks"]) * 2
            + len(config["exact_proxy_tail_ranks"])
            + len(config["quantized_factor_specs"])
        )
        primary = [row for row in rows if row["solver"] == config["primary_solver_id"]]
        geometry = [row for row in rows if row["solver"] == config["geometry_solver_id"]]
        if len(primary) != expected_primary:
            raise RuntimeError(f"primary grid {len(primary)} != {expected_primary}")
        if len(geometry) != expected_primary:
            raise RuntimeError(f"geometry grid {len(geometry)} != {expected_primary}")
        identities = {tuple(row[name] for name in IDENTITY) for row in primary}
        if len(identities) != int(config["expected_validation_unique_invocations"]):
            raise RuntimeError("primary validation identity coverage changed")
        geometry_identities = {tuple(row[name] for name in IDENTITY) for row in geometry}
        if geometry_identities != identities:
            raise RuntimeError("geometry validation identity coverage changed")
        facts["completed"] = True
        facts["frontier_rows"] = len(rows)
        facts["frontier_sha256"] = prior.sha256(args.output / FRONTIER)
        facts["factor_manifest_sha256"] = prior.sha256(manifest_path)
        facts["completed_unix"] = time.time()
        _atomic_json(facts_path, facts)
    except Exception as error:
        facts["failures"].append({"type": type(error).__name__, "message": str(error)})
        _atomic_json(facts_path, facts)
        raise


if __name__ == "__main__":
    main()
