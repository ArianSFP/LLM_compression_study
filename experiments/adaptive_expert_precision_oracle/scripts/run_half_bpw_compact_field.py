#!/usr/bin/env python3
"""Fit and evaluate bandwidth-bounded shared joint interaction fields."""

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
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(EXPERIMENT / "scripts")]

import run_average_rate_allocation as pr13
import run_half_bpw_joint_field as teacher
import run_sparse_streaming_study as base
from oracle_study.interaction_field import (
    EncodedInteractionFactor,
    JointInteractionFactor,
    encode_interaction_factor,
    make_encoded_factor_self_safe,
)
from oracle_study.efficient_joint_seed import lagrangian_self_seed
from oracle_study.low_rate_joint_field import (
    JointFieldTrace,
    combine_split_interaction_fields,
    exact_qmetric_factor,
    optimize_joint_field,
    split_group_states,
)
from oracle_study.shared_decision_field import (
    SharedDecisionBasis,
    fit_shared_decision_basis,
    project_shared_interaction_factor,
)
from oracle_study.split_interaction_field import (
    SPLIT_STATE_DOWN_HIGH,
    SPLIT_STATE_HIDDEN_INDEX,
    SPLIT_STATE_PAGE_COSTS,
    build_split_diagonal_dp_table,
    build_split_interaction_field,
    split_coordinate_descent,
    split_local_search,
    split_projection_responses,
    split_state_output,
)


FIT_FACTS = "compact_half_bpw_fit_facts.json"
FIT_MANIFEST = "compact_half_bpw_factor_manifest.json"
FIT_LAYER = "compact_half_bpw_factor_layer_{layer}.npz"
FIT_SIDECAR = "compact_half_bpw_factor_layer_{layer}.json"
RUN_FACTS = "compact_half_bpw_run_facts.json"
GROUP_FRONTIER = "compact_half_bpw_group_frontier.parquet"
STATE_EVIDENCE = "compact_half_bpw_state_evidence.parquet"
ACCOUNTING = "compact_half_bpw_accounting.json"
GROUP_IDENTITY = teacher.GROUP_IDENTITY
POLICIES = (
    "frozen_pr13_exact_combined_witness",
    "shared_decision_rank8_int4",
    "shared_decision_rank8_int4_dp_control",
    "shared_decision_rank8_fp32",
    "shared_decision_rank16_int4",
    "frozen_joint_exact_qmetric_oracle",
)
_FIT_CONTEXT: dict[str, Any] | None = None
_EVAL_CONTEXT: dict[str, Any] | None = None


def _sha256(path: Path) -> str:
    return teacher._sha256(path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    teacher._atomic_json(path, payload)


def _atomic_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(list(rows)).to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _dependencies() -> dict[str, str]:
    return {
        "runner_sha256": _sha256(Path(__file__)),
        "teacher_runner_sha256": _sha256(Path(teacher.__file__)),
        "joint_core_sha256": _sha256(
            EXPERIMENT / "src/oracle_study/low_rate_joint_field.py"
        ),
        "shared_basis_core_sha256": _sha256(
            EXPERIMENT / "src/oracle_study/shared_decision_field.py"
        ),
        "efficient_seed_core_sha256": _sha256(
            EXPERIMENT / "src/oracle_study/efficient_joint_seed.py"
        ),
        "split_core_sha256": _sha256(
            EXPERIMENT / "src/oracle_study/split_interaction_field.py"
        ),
        "interaction_core_sha256": _sha256(
            EXPERIMENT / "src/oracle_study/interaction_field.py"
        ),
    }


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("fit_split") != "train" or config.get("selection_split") != "validation":
        raise ValueError("compact field requires train-only fit and validation-only selection")
    expected = {
        "layers": [0, 4, 20, 39],
        "experts_per_group": 8,
        "units_per_expert": 512,
        "experts_per_layer": 256,
        "expected_train_groups_per_layer": 70,
        "expected_validation_groups_per_layer": 32,
        "expected_validation_groups": 128,
        "streamed_mean_pages_per_expert": 384,
        "group_page_budget": 3072,
        "page_bytes": 512,
        "expert_weights": 3_145_728,
        "abc_metadata_bytes_per_expert": 3084,
        "trajectory_exclusion_units": 64,
        "primary_rank": 8,
        "control_rank": 16,
        "primary_factor_payload_bytes_per_expert": 6148,
        "control_factor_payload_bytes_per_expert": 10244,
        "compact_seed": "lagrangian_primary_with_dp_control",
        "policies": list(POLICIES),
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"compact-field contract changed: {name}")
    if int(config["evaluation_workers"]) < 1 or int(config["fit_workers"]) < 1:
        raise ValueError("worker counts must be positive")
    if int(config["worker_blas_threads"]) != 1:
        raise ValueError("workers must use one BLAS thread")


def _plan_split(
    data: Mapping[str, np.ndarray], config: Mapping[str, Any], split: str,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    expected = (
        int(config["expected_train_groups_per_layer"])
        if split == "train"
        else int(config["expected_validation_groups_per_layer"])
    )
    counts = {}
    for layer in map(int, config["layers"]):
        local = pr13.prior.layer_view(data, layer)
        records = np.flatnonzero(np.asarray(local["split"]).astype(str) == split)
        counts[layer] = len(records)
        for record in records.tolist():
            experts = np.asarray(local["expert_ids"][record], np.int64)
            weights = np.asarray(local["router_weights"][record], np.float64)
            if experts.shape != (8,) or len(set(experts.tolist())) != 8:
                raise RuntimeError(f"{split} group is not eight unique routed experts")
            if weights.shape != (8,) or np.any(weights <= 0.0) or not np.isclose(
                weights.sum(), 1.0, atol=5e-7,
            ):
                raise RuntimeError(f"{split} router weights changed")
            groups.append({
                "capture_source": "exact_checkpoint",
                "evaluation_split": split,
                "request_id": str(local["request_id"][record]),
                "sequence_id": str(local["sequence_id"][record]),
                "position": int(local["position"][record]),
                "layer": layer,
                "record": int(record),
                "experts": experts.tolist(),
                "router_weights": weights.tolist(),
            })
    if counts != {layer: expected for layer in map(int, config["layers"])}:
        raise RuntimeError(f"{split} group counts changed: {counts}")
    identities = {tuple(group[name] for name in GROUP_IDENTITY) for group in groups}
    if len(identities) != len(groups):
        raise RuntimeError(f"duplicate {split} group identity")
    return groups


def _group_residual(responses: Sequence[Any], weights: np.ndarray, states: np.ndarray) -> np.ndarray:
    per_expert = split_group_states(states, len(responses))
    return sum(
        (
            weights[index]
            * (response.target_output - split_state_output(response, per_expert[index]))
            for index, response in enumerate(responses)
        ),
        np.zeros(np.asarray(responses[0].target_output).shape[0], np.float64),
    )


def _unit_residual(response: Any, unit: int, state: int) -> np.ndarray:
    hidden = float(response.hidden[unit, int(SPLIT_STATE_HIDDEN_INDEX[state])])
    current = (
        response.down4[unit] if bool(SPLIT_STATE_DOWN_HIGH[state])
        else response.down2[unit]
    )
    return response.target_hidden[unit] * response.down4[unit] - hidden * current


def _fit_teacher_group(task: int) -> tuple[int, np.ndarray, dict[str, Any]]:
    if _FIT_CONTEXT is None:
        raise RuntimeError("fit worker lacks context")
    started = time.perf_counter()
    group = _FIT_CONTEXT["groups"][int(task)]
    weights = np.asarray(group["router_weights"], np.float64)
    activation = np.asarray(_FIT_CONTEXT["local"]["x"][int(group["record"])], np.float64)
    responses = tuple(
        split_projection_responses(
            _FIT_CONTEXT["experts"][int(expert)]["q2"],
            _FIT_CONTEXT["experts"][int(expert)]["q4"],
            activation,
        )
        for expert in group["experts"]
    )
    proxy = _FIT_CONTEXT["proxy"]
    beta = float(_FIT_CONTEXT["beta"])
    fields = []
    for response, expert in zip(responses, group["experts"]):
        factor = exact_qmetric_factor(response.down2, response.down4, proxy, beta)
        fields.append(build_split_interaction_field(
            factor, response.hidden, _FIT_CONTEXT["experts"][int(expert)]["abc"],
        ))
    joint = combine_split_interaction_fields(tuple(fields), weights)
    budget = int(_FIT_CONTEXT["config"]["group_page_budget"])
    diagonal = build_split_diagonal_dp_table(joint, budget).seed(budget)
    trace = optimize_joint_field(
        joint,
        budget,
        coordinate_sweeps=int(_FIT_CONTEXT["config"]["coordinate_sweeps"]),
        local_shortlist=int(_FIT_CONTEXT["config"]["local_shortlist"]),
        local_swap_units=int(_FIT_CONTEXT["config"]["local_swap_units"]),
        local_max_passes=int(_FIT_CONTEXT["config"]["local_max_passes"]),
    )
    base = np.zeros(joint.units, np.int64)
    final_residual = _group_residual(responses, weights, trace.states)
    residuals = [
        _group_residual(responses, weights, base),
        _group_residual(responses, weights, diagonal),
        final_residual,
    ]
    rows = np.arange(joint.units)
    local_energy = np.asarray(joint.local_damage)[rows, trace.states]
    shortlist = np.argsort(-local_energy, kind="stable")[: int(
        _FIT_CONTEXT["config"]["trajectory_exclusion_units"]
    )]
    states_by_expert = split_group_states(trace.states, 8)
    for flat in shortlist.tolist():
        expert_index, unit = divmod(int(flat), 512)
        state = int(states_by_expert[expert_index][unit])
        residuals.append(
            final_residual
            - weights[expert_index] * _unit_residual(
                responses[expert_index], unit, state,
            )
        )
    values = np.stack(residuals)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("training trajectory contains non-finite residuals")
    return int(task), values, {
        "group_index": int(task),
        "rows": int(values.shape[0]),
        "teacher_pages": int(trace.pages),
        "teacher_sweeps": int(trace.coordinate_sweeps),
        "teacher_local_passes": int(trace.local_passes),
        "wall_seconds": time.perf_counter() - started,
    }


def _encoded_arrays(encoded: EncodedInteractionFactor) -> tuple[np.ndarray, np.ndarray, np.float32]:
    return (
        np.asarray(encoded.packed_codes, np.uint8),
        np.asarray(encoded.row_scales, np.float16),
        np.float32(encoded.global_scale),
    )


def _fit_layer(
    config: Mapping[str, Any], checkpoint: Path, index: Mapping[str, str],
    trees: Mapping[str, Any], layer: int, local: Mapping[str, np.ndarray],
    groups: list[dict[str, Any]], proxy: np.ndarray, beta: float, device: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    active = sorted({int(expert) for group in groups for expert in group["experts"]})
    experts = {}
    for expert in active:
        decoded = pr13.prior.decode_expert(checkpoint, dict(index), trees, layer, expert)
        q2 = tuple(decoded[name][0] for name in pr13.PROJECTIONS)
        q4 = tuple(decoded[name][2] for name in pr13.PROJECTIONS)
        experts[expert] = {
            "q2": q2,
            "q4": q4,
            "abc": pr13.unit_score_metadata(q2[2], q4[2], proxy=proxy, beta=beta),
        }
    global _FIT_CONTEXT
    _FIT_CONTEXT = {
        "groups": tuple(groups), "local": local, "experts": experts,
        "proxy": proxy, "beta": beta, "config": config,
    }
    pending = {}
    context = mp.get_context(str(config["worker_start_method"]))
    workers = min(int(config["fit_workers"]), len(groups))
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = {executor.submit(_fit_teacher_group, task): task for task in range(len(groups))}
        for future in as_completed(futures):
            task, residuals, diagnostic = future.result()
            if task != futures[future]:
                raise RuntimeError("fit worker identity changed")
            pending[task] = (residuals, diagnostic)
    trajectory = np.concatenate([pending[task][0] for task in range(len(groups))])
    diagnostics = [pending[task][1] for task in range(len(groups))]
    _FIT_CONTEXT = None
    del experts
    gc.collect()

    basis = fit_shared_decision_basis(
        trajectory, proxy, beta, int(config["control_rank"]), normalize_rows=True,
    )
    basis8 = SharedDecisionBasis(
        components=np.asarray(basis.components[:, : int(config["primary_rank"])], np.float32),
        fit_rows=basis.fit_rows,
        fit_rank=int(config["primary_rank"]),
        normalized_rows=basis.normalized_rows,
        explained_energy=0.0,
    )
    # Recompute prefix energy from the fitted trajectory rather than assigning
    # an invalid diagnostic to the immutable dataclass contract.
    from oracle_study.average_rate_allocator import qmetric_features
    features = qmetric_features(trajectory, proxy, beta)
    features /= np.linalg.norm(features, axis=1)[:, None]
    captured8 = float(np.linalg.norm(features @ basis8.components) ** 2 / np.linalg.norm(features) ** 2)
    basis8 = SharedDecisionBasis(
        components=basis8.components,
        fit_rows=basis8.fit_rows,
        fit_rank=basis8.fit_rank,
        normalized_rows=True,
        explained_energy=captured8,
    )

    arrays: dict[str, list[np.ndarray] | np.ndarray] = {
        "rank8_packed_codes": [], "rank8_row_scales": [], "rank8_global_scales": [],
        "rank16_packed_codes": [], "rank16_row_scales": [], "rank16_global_scales": [],
        "rank8_fp32_l4": [], "rank8_fp32_l2": [],
    }
    shrink8, shrink16 = [], []
    down_tree = trees["down"]
    for expert in range(int(config["experts_per_layer"])):
        leaves = pr13.load_compressed_mxfp4_expert(
            checkpoint, dict(index), layer, expert, "down",
        )
        down2 = down_tree.decode(leaves, 2)
        down4 = down_tree.decode(leaves, 4)
        abc = pr13.unit_score_metadata(down2, down4, proxy=proxy, beta=beta)
        raw8 = project_shared_interaction_factor(
            down2.T, down4.T, proxy, beta, basis8,
        )
        raw16 = project_shared_interaction_factor(
            down2.T, down4.T, proxy, beta, basis,
        )
        encoded8, scale8 = make_encoded_factor_self_safe(
            encode_interaction_factor(raw8, "int4_per_row", hadamard_rotate=True), abc,
        )
        encoded16, scale16 = make_encoded_factor_self_safe(
            encode_interaction_factor(raw16, "int4_per_row", hadamard_rotate=True), abc,
        )
        p8, s8, g8 = _encoded_arrays(encoded8)
        p16, s16, g16 = _encoded_arrays(encoded16)
        arrays["rank8_packed_codes"].append(p8)
        arrays["rank8_row_scales"].append(s8)
        arrays["rank8_global_scales"].append(np.asarray(g8))
        arrays["rank16_packed_codes"].append(p16)
        arrays["rank16_row_scales"].append(s16)
        arrays["rank16_global_scales"].append(np.asarray(g16))
        arrays["rank8_fp32_l4"].append(np.asarray(raw8.l4, np.float32))
        arrays["rank8_fp32_l2"].append(np.asarray(raw8.l2, np.float32))
        shrink8.append(float(scale8))
        shrink16.append(float(scale16))
    result = {name: np.stack(values) for name, values in arrays.items()}
    result.update({
        "basis_components": np.asarray(basis.components, np.float32),
        "proxy": np.asarray(proxy, np.float32),
        "beta": np.asarray([beta], np.float64),
    })
    if result["rank8_packed_codes"].shape != (256, 4096):
        raise RuntimeError("rank8 physical code shape changed")
    if result["rank16_packed_codes"].shape != (256, 8192):
        raise RuntimeError("rank16 physical code shape changed")
    return result, {
        "layer": int(layer),
        "train_groups": len(groups),
        "trajectory_rows": int(trajectory.shape[0]),
        "trajectory_exclusion_units": int(config["trajectory_exclusion_units"]),
        "rank8_explained_training_energy": captured8,
        "rank16_explained_training_energy": float(basis.explained_energy),
        "rank8_self_safe_scale_min": float(np.min(shrink8)),
        "rank16_self_safe_scale_min": float(np.min(shrink16)),
        "teacher_group_wall_seconds": float(sum(item["wall_seconds"] for item in diagnostics)),
    }


def _encoded(
    arrays: Mapping[str, np.ndarray], prefix: str, expert: int, rank: int,
) -> JointInteractionFactor:
    return EncodedInteractionFactor(
        packed_codes=np.asarray(arrays[f"{prefix}_packed_codes"][expert], np.uint8),
        row_scales=np.asarray(arrays[f"{prefix}_row_scales"][expert], np.float16),
        units=512,
        rank=int(rank),
        method=f"shared_decision_rank{rank}",
        tail_rank=int(rank),
        exact_rank=0,
        encoding="int4_per_row_hadamard",
        global_scale=np.float32(arrays[f"{prefix}_global_scales"][expert]),
    ).decode()


def _load_factors(arrays: Mapping[str, np.ndarray], expert: int) -> dict[str, JointInteractionFactor]:
    fp32 = JointInteractionFactor(
        l4=np.asarray(arrays["rank8_fp32_l4"][expert], np.float32),
        l2=np.asarray(arrays["rank8_fp32_l2"][expert], np.float32),
        method="shared_decision_rank8_fp32",
        tail_rank=8,
        encoding="fp32",
        storage_bytes=32768,
    )
    rank8 = _encoded(arrays, "rank8", expert, 8)
    result = {
        POLICIES[1]: rank8,
        POLICIES[2]: rank8,
        POLICIES[3]: fp32,
        POLICIES[4]: _encoded(arrays, "rank16", expert, 16),
    }
    expected = {
        POLICIES[1]: 6148, POLICIES[2]: 6148,
        POLICIES[3]: 32768, POLICIES[4]: 10244,
    }
    if {name: factor.payload_bytes for name, factor in result.items()} != expected:
        raise RuntimeError("compact factor payload contract changed")
    return result


def _verify_teacher(
    config: Mapping[str, Any], directory: Path,
) -> tuple[dict[tuple[Any, ...], np.ndarray], dict[tuple[Any, ...], dict[str, float]]]:
    files = {
        teacher.RUN_FACTS: config["teacher_run_facts_sha256"],
        teacher.GROUP_FRONTIER: config["teacher_group_frontier_sha256"],
        teacher.STATE_EVIDENCE: config["teacher_state_evidence_sha256"],
        teacher.ACCOUNTING: config["teacher_accounting_sha256"],
    }
    for name, expected in files.items():
        path = directory / name
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"exact-joint teacher artifact changed: {name}")
    facts = json.loads((directory / teacher.RUN_FACTS).read_text())
    if facts.get("completed") is not True or facts.get("failures"):
        raise RuntimeError("exact-joint teacher is incomplete")
    state = pd.read_parquet(directory / teacher.STATE_EVIDENCE)
    group = pd.read_parquet(directory / teacher.GROUP_FRONTIER)
    state = state[state["allocation_policy"] == "joint_exact_qmetric_oracle"]
    group = group[group["allocation_policy"] == "joint_exact_qmetric_oracle"]
    if len(state) != 128 or len(group) != 128:
        raise RuntimeError("exact-joint teacher row count changed")
    states = {
        tuple(row[name] for name in GROUP_IDENTITY): np.asarray(
            json.loads(row["selected_states"]), np.int64,
        )
        for row in state.to_dict("records")
    }
    metrics = {
        tuple(row[name] for name in GROUP_IDENTITY): {
            "recovery": float(row["group_recovery"]),
            "damage": float(row["group_exact_qenergy_damage"]),
        }
        for row in group.to_dict("records")
    }
    if set(states) != set(metrics):
        raise RuntimeError("exact-joint teacher identities changed")
    return states, metrics


def _optimize_lagrangian(
    field: Any, config: Mapping[str, Any],
) -> tuple[JointFieldTrace, Any]:
    """Run the primary linear-memory seed and existing incremental repairs."""

    budget = int(config["group_page_budget"])
    seed = lagrangian_self_seed(field, budget)
    coordinate = split_coordinate_descent(
        field, seed.states, budget,
        max_sweeps=int(config["coordinate_sweeps"]),
    )
    local = split_local_search(
        field, coordinate.states, budget,
        shortlist_size=int(config["local_shortlist"]),
        max_swap_units=int(config["local_swap_units"]),
        max_passes=int(config["local_max_passes"]),
    )
    return JointFieldTrace(
        states=local.states.copy(),
        pages=int(local.pages),
        damage=float(local.damage),
        seed_name="global_self_lagrangian",
        seeds_evaluated=1,
        coordinate_sweeps=int(coordinate.sweeps),
        coordinate_accepted_moves=int(coordinate.accepted_moves),
        local_passes=int(local.passes),
        local_accepted_bundles=int(local.accepted_bundles),
    ), seed


def _evaluate_group(task: int):
    if _EVAL_CONTEXT is None:
        raise RuntimeError("evaluation worker lacks context")
    started = time.perf_counter()
    group = _EVAL_CONTEXT["groups"][int(task)]
    identity = tuple(group[name] for name in GROUP_IDENTITY)
    weights = np.asarray(group["router_weights"], np.float64)
    activation = np.asarray(_EVAL_CONTEXT["local"]["x"][int(group["record"])], np.float64)
    responses = tuple(
        split_projection_responses(
            _EVAL_CONTEXT["experts"][int(expert)]["q2"],
            _EVAL_CONTEXT["experts"][int(expert)]["q4"],
            activation,
        )
        for expert in group["experts"]
    )
    proxy, beta = _EVAL_CONTEXT["proxy"], float(_EVAL_CONTEXT["beta"])
    base = np.zeros(4096, np.int64)
    base_damage = teacher._exact_group_damage(responses, weights, base, proxy, beta)
    witness = np.asarray(_EVAL_CONTEXT["witnesses"][identity], np.int64)
    exact_state = np.asarray(_EVAL_CONTEXT["teacher_states"][identity], np.int64)
    records = [
        (POLICIES[0], witness, 0, 0, 0, 0, 0, 0.0, "frozen_pr13_state"),
    ]
    for policy in POLICIES[1:5]:
        fields = tuple(
            build_split_interaction_field(
                _EVAL_CONTEXT["experts"][int(expert)]["factors"][policy],
                responses[index].hidden,
                _EVAL_CONTEXT["experts"][int(expert)]["abc"],
            )
            for index, expert in enumerate(group["experts"])
        )
        joint = combine_split_interaction_fields(fields, weights)
        solve_started = time.perf_counter()
        if policy == POLICIES[2]:
            trace = optimize_joint_field(
                joint,
                int(_EVAL_CONTEXT["config"]["group_page_budget"]),
                coordinate_sweeps=int(_EVAL_CONTEXT["config"]["coordinate_sweeps"]),
                local_shortlist=int(_EVAL_CONTEXT["config"]["local_shortlist"]),
                local_swap_units=int(_EVAL_CONTEXT["config"]["local_swap_units"]),
                local_max_passes=int(_EVAL_CONTEXT["config"]["local_max_passes"]),
            )
            seed_evaluations = 8 * 4096 * (3072 + 1)
            seed_backpointer_bytes = 4096 * (3072 + 1)
        else:
            trace, seed_trace = _optimize_lagrangian(joint, _EVAL_CONTEXT["config"])
            seed_evaluations = (
                int(seed_trace.state_evaluations)
                + int(seed_trace.greedy_evaluations)
            )
            seed_backpointer_bytes = 0
        solve_seconds = time.perf_counter() - solve_started
        expected_seed = (
            "global_exact_self_dp" if policy == POLICIES[2]
            else "global_self_lagrangian"
        )
        if trace.seed_name != expected_seed or trace.seeds_evaluated != 1:
            raise RuntimeError("compact solver admitted a nondeployable seed")
        records.append((
            policy, trace.states, trace.coordinate_sweeps, trace.local_passes,
            teacher._solver_macs(
                joint.rank, joint.units, trace.coordinate_sweeps,
                trace.local_passes, int(_EVAL_CONTEXT["config"]["local_shortlist"]),
            ), seed_evaluations, seed_backpointer_bytes, solve_seconds,
            trace.seed_name,
        ))
    records.append((
        POLICIES[5], exact_state, 0, 0, 0, 0, 0, 0.0,
        "frozen_exact_teacher",
    ))
    payloads = {
        POLICIES[0]: 6148, POLICIES[1]: 6148, POLICIES[2]: 6148,
        POLICIES[3]: 32768, POLICIES[4]: 10244, POLICIES[5]: 0,
    }
    rows, state_rows = [], []
    for (
        policy, states, sweeps, passes, macs, seed_evaluations,
        seed_backpointer_bytes, solve_seconds, seed,
    ) in records:
        pages = int(SPLIT_STATE_PAGE_COSTS[np.asarray(states, np.int64)].sum())
        if pages > int(_EVAL_CONTEXT["config"]["group_page_budget"]):
            raise RuntimeError("compact policy exceeded pooled page budget")
        damage = teacher._exact_group_damage(responses, weights, states, proxy, beta)
        if policy == POLICIES[5]:
            frozen = _EVAL_CONTEXT["teacher_metrics"][identity]
            if not np.isclose(damage, frozen["damage"], rtol=1e-9, atol=1e-8):
                raise RuntimeError("exact-joint teacher damage changed")
        recovery = 1.0 - damage / base_damage
        metadata = int(_EVAL_CONTEXT["config"]["abc_metadata_bytes_per_expert"]) + payloads[policy]
        total_bytes = pages * 512 + 8 * metadata
        row = {
            **{name: group[name] for name in GROUP_IDENTITY},
            "sequence_id": group["sequence_id"],
            "allocation_policy": policy,
            "streamed_bpw": 0.5,
            "group_page_budget": 3072,
            "actual_group_pages": pages,
            "average_actual_streamed_bpw": pages / (8.0 * 768.0),
            "factor_payload_bytes_per_expert": payloads[policy],
            "combined_metadata_bytes_per_expert": metadata,
            "metadata_bpw": 8.0 * metadata / 3_145_728,
            "allowed_total_bpw": 0.5 + 8.0 * metadata / 3_145_728,
            "logical_group_bytes_read": total_bytes,
            "group_base_qenergy_damage": base_damage,
            "group_exact_qenergy_damage": damage,
            "group_recovery": recovery,
            "damage_ratio_vs_pr13": damage / max(
                teacher._exact_group_damage(responses, weights, witness, proxy, beta), 1e-30,
            ),
            "state_agreement_with_exact_teacher": float(np.mean(states == exact_state)),
            "coordinate_sweeps": int(sweeps),
            "local_passes": int(passes),
            "selector_compute_macs": int(macs),
            "selector_seed_state_evaluations": int(seed_evaluations),
            "selector_seed_backpointer_bytes": int(seed_backpointer_bytes),
            "selector_wall_seconds": float(solve_seconds),
            "validation_oracle_seed_used": False,
            "seed_name": seed,
            "validation_only": True,
            "deployable_payload_accounting": policy in POLICIES[1:5],
        }
        rows.append(row)
        state_rows.append({
            **{name: group[name] for name in GROUP_IDENTITY},
            "sequence_id": group["sequence_id"],
            "allocation_policy": policy,
            "selected_states_sha256": teacher._state_hash(states),
            "selected_states": json.dumps(np.asarray(states, np.int64).tolist(), separators=(",", ":")),
        })
    return int(task), rows, state_rows, {
        "group_index": int(task), "wall_seconds": time.perf_counter() - started,
    }


def _base_inputs(args: argparse.Namespace, config: Mapping[str, Any]):
    base_config, frozen_hashes = teacher._verify_pr13(config, args)
    audit, input_hashes = pr13._locked_inputs(
        base_config, args.pr10_config, args.exact_captures, args.checkpoint, args.trees,
    )
    pr13._verify_base(base_config, args.pr12_config, args.pr12_dir)
    return base_config, frozen_hashes, audit, input_hashes


def _fit(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    _validate_config(config)
    base_config, frozen_hashes, audit, inputs = _base_inputs(args, config)
    threads = pr13._thread_contract({
        **base_config,
        "evaluation_workers": int(config["fit_workers"]),
        "worker_blas_threads": int(config["worker_blas_threads"]),
    })
    runtime = pr13.prior.runtime_provenance(args.device)
    dependencies = _dependencies()
    locked = {
        "run_id": config["run_id"], "phase": "fit", "fit_split": "train",
        "validation_rows_admitted_or_used": False,
        "test_scientific_rows_admitted_or_used": False,
        "config_sha256": _sha256(args.config), "capture_sha256": inputs["capture_sha256"],
        "tree_sha256": inputs["tree_sha256"], "runtime_provenance": runtime,
        **frozen_hashes, **dependencies,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    facts_path = args.output / FIT_FACTS
    if facts_path.exists():
        facts = json.loads(facts_path.read_text())
        for name, value in locked.items():
            if facts.get(name) != value:
                raise RuntimeError(f"compact fit resume provenance changed: {name}")
        if facts.get("failures") or facts.get("failure_history"):
            raise RuntimeError("failed compact fit cannot resume")
        if facts.get("completed") is True:
            return
    else:
        facts = {
            **locked, "completed": False, "completed_layers": [], "failures": [],
            "failure_history": [], "checkpoint_audit": audit,
            "worker_thread_environment": threads,
        }
        _atomic_json(facts_path, facts)
    data = pr13.prior.load_capture_admitted(args.exact_captures, ["train"])
    if set(map(str, np.unique(data["split"]))) != {"train"}:
        raise RuntimeError("compact fit admitted non-training rows")
    groups = _plan_split(data, config, "train")
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    tree_records = json.loads(args.trees.read_text())
    trees = {name: pr13.tree_from_record(tree_records[name]) for name in pr13.PROJECTIONS}
    layer_records = {}
    try:
        for layer in map(int, config["layers"]):
            if layer in set(map(int, facts["completed_layers"])):
                continue
            local = pr13.prior.layer_view(data, layer)
            proxy, proxy_facts = base.proxy_gradients(
                local["xplus"], [local[f"h{h}_router_logits"] for h in range(1, 5)], local["split"],
            )
            beta = float(proxy_facts["beta"])
            arrays, diagnostic = _fit_layer(
                config, args.checkpoint, index, trees, layer, local,
                [group for group in groups if int(group["layer"]) == layer],
                proxy, beta, args.device,
            )
            shard = args.output / FIT_LAYER.format(layer=layer)
            _atomic_npz(shard, arrays)
            sidecar = args.output / FIT_SIDECAR.format(layer=layer)
            sidecar_payload = {
                **diagnostic, "file": shard.name, "bytes": shard.stat().st_size,
                "sha256": _sha256(shard), "runtime_provenance": runtime,
                "fit_split": "train", "validation_rows_admitted_or_used": False,
                "test_scientific_rows_admitted_or_used": False,
            }
            _atomic_json(sidecar, sidecar_payload)
            layer_records[str(layer)] = {
                "file": shard.name, "bytes": shard.stat().st_size,
                "sha256": sidecar_payload["sha256"], "sidecar": sidecar.name,
                "sidecar_sha256": _sha256(sidecar),
            }
            facts["completed_layers"] = sorted(set(facts["completed_layers"]) | {layer})
            _atomic_json(facts_path, facts)
        manifest = {
            **locked, "completed": True, "layers": layer_records,
            "factor_payload_bytes": {
                POLICIES[1]: 6148, POLICIES[2]: 6148,
                POLICIES[3]: 32768, POLICIES[4]: 10244,
            },
            "primary_combined_metadata_bytes_per_expert": 9232,
        }
        _atomic_json(args.output / FIT_MANIFEST, manifest)
        facts["completed"] = True
        facts["factor_manifest_sha256"] = _sha256(args.output / FIT_MANIFEST)
        _atomic_json(facts_path, facts)
    except Exception as error:
        record = {"type": type(error).__name__, "message": str(error)}
        facts["failures"] = [record]
        facts.setdefault("failure_history", []).append(record)
        _atomic_json(facts_path, facts)
        raise


def _evaluate(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    _validate_config(config)
    base_config, frozen_hashes, audit, inputs = _base_inputs(args, config)
    threads = pr13._thread_contract({
        **base_config, "evaluation_workers": int(config["evaluation_workers"]),
        "worker_blas_threads": int(config["worker_blas_threads"]),
    })
    runtime = pr13.prior.runtime_provenance(args.device)
    dependencies = _dependencies()
    manifest_path = args.fit_dir / FIT_MANIFEST
    facts_path = args.fit_dir / FIT_FACTS
    if not manifest_path.is_file() or not facts_path.is_file():
        raise RuntimeError("compact fit bundle is missing")
    manifest = json.loads(manifest_path.read_text())
    fit_facts = json.loads(facts_path.read_text())
    if manifest.get("completed") is not True or fit_facts.get("completed") is not True:
        raise RuntimeError("compact fit bundle is incomplete")
    teacher_states, teacher_metrics = _verify_teacher(config, args.teacher_dir)
    witnesses, recoveries = teacher._load_witnesses(config, args.pr13_validation_dir)
    data = pr13.prior.load_capture_admitted(args.exact_captures, ["train", "validation"])
    if "test" in set(map(str, np.unique(data["split"]))):
        raise RuntimeError("test row entered compact evaluation")
    groups = _plan_split(data, config, "validation")
    if set(tuple(group[name] for name in GROUP_IDENTITY) for group in groups) != set(teacher_states):
        raise RuntimeError("compact validation identities differ from teacher")
    locked = {
        "run_id": config["run_id"], "phase": "evaluate", "selection_split": "validation",
        "test_scientific_rows_admitted_or_used": False,
        "config_sha256": _sha256(args.config), "capture_sha256": inputs["capture_sha256"],
        "tree_sha256": inputs["tree_sha256"], "runtime_provenance": runtime,
        "fit_facts_sha256": _sha256(facts_path), "fit_manifest_sha256": _sha256(manifest_path),
        **frozen_hashes, **dependencies,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    run_path = args.output / RUN_FACTS
    if run_path.exists():
        facts = json.loads(run_path.read_text())
        for name, value in locked.items():
            if facts.get(name) != value:
                raise RuntimeError(f"compact evaluation resume changed: {name}")
        if facts.get("failures") or facts.get("failure_history"):
            raise RuntimeError("failed compact evaluation cannot resume")
        if facts.get("completed") is True:
            return
    else:
        facts = {
            **locked, "completed": False, "completed_layers": [], "failures": [],
            "failure_history": [], "checkpoint_audit": audit,
            "worker_thread_environment": threads,
        }
        _atomic_json(run_path, facts)
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    tree_records = json.loads(args.trees.read_text())
    trees = {name: pr13.tree_from_record(tree_records[name]) for name in pr13.PROJECTIONS}
    rows, states, diagnostics = [], [], []
    try:
        for layer in map(int, config["layers"]):
            local = pr13.prior.layer_view(data, layer)
            proxy, proxy_facts = base.proxy_gradients(
                local["xplus"], [local[f"h{h}_router_logits"] for h in range(1, 5)], local["split"],
            )
            beta = float(proxy_facts["beta"])
            record = manifest["layers"][str(layer)]
            shard = args.fit_dir / record["file"]
            if _sha256(shard) != record["sha256"] or shard.stat().st_size != int(record["bytes"]):
                raise RuntimeError("compact fit shard changed")
            arrays = dict(np.load(shard, allow_pickle=False))
            if not np.array_equal(np.asarray(arrays["proxy"], np.float32), np.asarray(proxy, np.float32)):
                raise RuntimeError("train proxy changed")
            layer_groups = [group for group in groups if int(group["layer"]) == layer]
            active = sorted({int(expert) for group in layer_groups for expert in group["experts"]})
            experts = {}
            for expert in active:
                decoded = pr13.prior.decode_expert(args.checkpoint, index, trees, layer, expert)
                q2 = tuple(decoded[name][0] for name in pr13.PROJECTIONS)
                q4 = tuple(decoded[name][2] for name in pr13.PROJECTIONS)
                experts[expert] = {
                    "q2": q2, "q4": q4,
                    "abc": pr13.unit_score_metadata(q2[2], q4[2], proxy=proxy, beta=beta),
                    "factors": _load_factors(arrays, expert),
                }
            global _EVAL_CONTEXT
            _EVAL_CONTEXT = {
                "groups": tuple(layer_groups), "local": local, "experts": experts,
                "proxy": proxy, "beta": beta, "config": config,
                "witnesses": witnesses, "recoveries": recoveries,
                "teacher_states": teacher_states, "teacher_metrics": teacher_metrics,
            }
            pending = {}
            context = mp.get_context(str(config["worker_start_method"]))
            workers = min(int(config["evaluation_workers"]), len(layer_groups))
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
                futures = {executor.submit(_evaluate_group, task): task for task in range(len(layer_groups))}
                for future in as_completed(futures):
                    task, local_rows, local_states, diagnostic = future.result()
                    if task != futures[future]:
                        raise RuntimeError("evaluation worker identity changed")
                    pending[task] = (local_rows, local_states, diagnostic)
            for task in range(len(layer_groups)):
                local_rows, local_states, diagnostic = pending[task]
                rows.extend(local_rows); states.extend(local_states); diagnostics.append(diagnostic)
            _atomic_parquet(args.output / GROUP_FRONTIER, rows)
            _atomic_parquet(args.output / STATE_EVIDENCE, states)
            facts["completed_layers"] = sorted(set(facts["completed_layers"]) | {layer})
            _atomic_json(run_path, facts)
            _EVAL_CONTEXT = None
            del experts, arrays
            gc.collect()
        expected = 128 * len(POLICIES)
        if len(rows) != expected or len(states) != expected:
            raise RuntimeError("compact output row grid changed")
        frame = pd.DataFrame(rows)
        if frame[list(GROUP_IDENTITY) + ["allocation_policy"]].duplicated().any():
            raise RuntimeError("duplicate compact policy row")
        accounting = {
            "schema_version": 2, "group_rows": len(rows), "state_rows": len(states),
            "policies": list(POLICIES), "streamed_bpw": 0.5,
            "primary_factor_payload_bytes_per_expert": 6148,
            "primary_combined_metadata_bytes_per_expert": 9232,
            "primary_group_sidecar_bytes": 73856,
            "diagnostics": diagnostics,
        }
        _atomic_json(args.output / ACCOUNTING, accounting)
        facts["completed"] = True
        facts["group_frontier_sha256"] = _sha256(args.output / GROUP_FRONTIER)
        facts["state_evidence_sha256"] = _sha256(args.output / STATE_EVIDENCE)
        facts["accounting_sha256"] = _sha256(args.output / ACCOUNTING)
        _atomic_json(run_path, facts)
    except Exception as error:
        record = {"type": type(error).__name__, "message": str(error)}
        facts["failures"] = [record]
        facts.setdefault("failure_history", []).append(record)
        _atomic_json(run_path, facts)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("fit", "evaluate"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pr13-config", type=Path, required=True)
    parser.add_argument("--pr13-fit-dir", type=Path, required=True)
    parser.add_argument("--pr13-validation-dir", type=Path, required=True)
    parser.add_argument("--pr12-config", type=Path, required=True)
    parser.add_argument("--pr12-dir", type=Path, required=True)
    parser.add_argument("--pr10-config", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path)
    parser.add_argument("--fit-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trees", type=Path, required=True)
    parser.add_argument("--exact-captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.phase == "fit":
        _fit(args, config)
    else:
        if args.teacher_dir is None or args.fit_dir is None:
            raise ValueError("evaluate requires --teacher-dir and --fit-dir")
        _evaluate(args, config)


if __name__ == "__main__":
    main()
