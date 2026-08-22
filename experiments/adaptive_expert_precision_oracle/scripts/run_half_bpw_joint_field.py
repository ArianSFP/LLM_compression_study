#!/usr/bin/env python3
"""Evaluate a joint top-8 unit field at exactly 0.5 streamed bpw."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import hashlib
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
sys.path.insert(0, str(EXPERIMENT / "src"))
sys.path.insert(0, str(EXPERIMENT / "scripts"))

import run_average_rate_allocation as pr13
import run_sparse_streaming_study as base
from oracle_study.average_rate_allocator import qmetric_features
from oracle_study.low_rate_joint_field import (
    combine_split_interaction_fields,
    exact_qmetric_factor,
    optimize_joint_field,
    split_group_states,
)
from oracle_study.split_interaction_field import (
    SPLIT_STATE_PAGE_COSTS,
    build_split_interaction_field,
    split_projection_responses,
    split_state_output,
)


GROUP_FRONTIER = "half_bpw_joint_group_frontier.parquet"
STATE_EVIDENCE = "half_bpw_joint_state_evidence.parquet"
RUN_FACTS = "half_bpw_joint_run_facts.json"
ACCOUNTING = "half_bpw_joint_accounting.json"
GROUP_IDENTITY = (
    "capture_source", "evaluation_split", "request_id", "position", "layer",
)
POLICIES = (
    "frozen_pr13_exact_combined_witness",
    "joint_rank4_proxy_int4",
    "joint_rank4_proxy_fp32",
    "joint_exact_qmetric_oracle",
)
_CONTEXT: dict[str, Any] | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _atomic_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    temporary.replace(path)


def _dependencies() -> dict[str, str]:
    paths = {
        "runner_sha256": Path(__file__).resolve(),
        "joint_core_sha256": EXPERIMENT / "src/oracle_study/low_rate_joint_field.py",
        "base_pr13_runner_sha256": EXPERIMENT / "scripts/run_average_rate_allocation.py",
        "base_pr13_allocator_sha256": EXPERIMENT / "src/oracle_study/average_rate_allocator.py",
        "split_core_sha256": EXPERIMENT / "src/oracle_study/split_interaction_field.py",
        "interaction_core_sha256": EXPERIMENT / "src/oracle_study/interaction_field.py",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _validate_config(config: Mapping[str, Any]) -> None:
    if list(map(int, config["layers"])) != [0, 4, 20, 39]:
        raise ValueError("layer grid changed")
    if int(config["experts_per_group"]) != 8 or int(config["units_per_expert"]) != 512:
        raise ValueError("top-8/unit contract changed")
    if int(config["streamed_mean_pages_per_expert"]) != 384:
        raise ValueError("primary streamed rate is not exactly 0.5 bpw")
    if int(config["group_page_budget"]) != 8 * 384:
        raise ValueError("pooled group page budget changed")
    if tuple(map(str, config["policies"])) != POLICIES:
        raise ValueError("policy grid changed")
    if int(config["evaluation_workers"]) != 24:
        raise ValueError("worker count changed")
    if config.get("selection_split") != "validation":
        raise ValueError("selection split changed")
    if config.get("no_test_scientific_values_used") is not True:
        raise ValueError("test scientific-use guard changed")


def _verify_pr13(
    config: Mapping[str, Any], args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, str]]:
    paths = {
        "base_pr13_config_sha256": args.pr13_config,
        "base_pr13_runner_sha256": EXPERIMENT / "scripts/run_average_rate_allocation.py",
        "base_pr13_allocator_sha256": EXPERIMENT / "src/oracle_study/average_rate_allocator.py",
        "base_pr13_fit_facts_sha256": args.pr13_fit_dir / pr13.FIT_FACTS,
        "base_pr13_fit_manifest_sha256": args.pr13_fit_dir / pr13.FIT_MANIFEST,
        "base_pr13_validation_facts_sha256": args.pr13_validation_dir / pr13.RUN_FACTS,
        "base_pr13_group_frontier_sha256": args.pr13_validation_dir / pr13.GROUP_FRONTIER,
        "base_pr13_expert_allocation_sha256": args.pr13_validation_dir / pr13.EXPERT_ALLOCATION,
    }
    observed = {}
    for name, path in paths.items():
        value = _sha256(path)
        if value != str(config[name]):
            raise RuntimeError(f"immutable PR13 dependency changed: {name}")
        observed[name] = value
    base_config = json.loads(args.pr13_config.read_text())
    if base_config.get("run_id") != "qwen36_mxfp4_average_rate_allocation_frontier_closure_20260821_v2":
        raise RuntimeError("immutable PR13 run ID changed")
    fit_facts = json.loads((args.pr13_fit_dir / pr13.FIT_FACTS).read_text())
    manifest = json.loads((args.pr13_fit_dir / pr13.FIT_MANIFEST).read_text())
    validation = json.loads((args.pr13_validation_dir / pr13.RUN_FACTS).read_text())
    if any(record.get("completed") is not True for record in (fit_facts, manifest, validation)):
        raise RuntimeError("immutable PR13 evidence is incomplete")
    if fit_facts.get("failures") or validation.get("failures"):
        raise RuntimeError("immutable PR13 evidence records a failure")
    return base_config, observed


def _identity(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(record[name] for name in GROUP_IDENTITY)


def _load_witnesses(
    config: Mapping[str, Any], validation_dir: Path,
) -> tuple[dict[tuple[Any, ...], np.ndarray], dict[tuple[Any, ...], float]]:
    expert = pd.read_parquet(validation_dir / pr13.EXPERT_ALLOCATION)
    group = pd.read_parquet(validation_dir / pr13.GROUP_FRONTIER)
    filters = (
        (expert["factor_config_id"].astype(str) == str(config["base_witness_factor_id"]))
        & (expert["allocation_policy"].astype(str) == str(config["base_witness_policy"]))
        & (expert["mean_budget_pages_per_expert"].astype(int) == int(config["base_witness_mean_pages"]))
        & (expert["burst_cap_pages_per_expert"].astype(int) == int(config["base_witness_burst_pages"]))
    )
    selected = expert.loc[filters].copy()
    if len(selected) != int(config["expected_validation_groups"]) * 8:
        raise RuntimeError("PR13 witness expert row count changed")
    witnesses = {}
    for raw_identity, frame in selected.groupby(list(GROUP_IDENTITY), sort=False):
        local = frame.sort_values("router_rank")
        if local["router_rank"].astype(int).tolist() != list(range(1, 9)):
            raise RuntimeError("PR13 witness router-rank grid changed")
        states = [
            np.asarray(json.loads(value), np.int64)
            for value in local["selected_states"].astype(str)
        ]
        if any(state.shape != (512,) for state in states):
            raise RuntimeError("PR13 witness state shape changed")
        witnesses[tuple(raw_identity)] = np.concatenate(states)
    group_filter = (
        (group["factor_config_id"].astype(str) == str(config["base_witness_factor_id"]))
        & (group["allocation_policy"].astype(str) == str(config["base_witness_policy"]))
        & (group["mean_budget_pages_per_expert"].astype(int) == int(config["base_witness_mean_pages"]))
        & (group["burst_cap_pages_per_expert"].astype(int) == int(config["base_witness_burst_pages"]))
    )
    baseline = group.loc[group_filter]
    if len(baseline) != int(config["expected_validation_groups"]):
        raise RuntimeError("PR13 witness group row count changed")
    recoveries = {
        _identity(row): float(row["group_recovery"])
        for row in baseline.to_dict("records")
    }
    if set(witnesses) != set(recoveries):
        raise RuntimeError("PR13 witness identity sets changed")
    return witnesses, recoveries


def _state_hash(states: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(states, np.uint8).tobytes()).hexdigest()


def _exact_group_damage(
    responses: tuple[Any, ...], weights: np.ndarray, states: np.ndarray,
    proxy: np.ndarray, beta: float,
) -> float:
    per_expert = split_group_states(states, len(responses))
    residual = sum(
        (
            weights[index]
            * (
                response.target_output
                - split_state_output(response, per_expert[index])
            )
            for index, response in enumerate(responses)
        ),
        np.zeros(np.asarray(responses[0].target_output).shape[0], np.float64),
    )
    feature = qmetric_features(residual[None, :], proxy, beta)[0]
    return float(feature @ feature)


def _solver_macs(rank: int, units: int, sweeps: int, passes: int, shortlist: int) -> int:
    coordinate = (2 * units * rank + 16 * units) * int(sweeps)
    maximum_shortlist = 7 * int(shortlist)
    local = int(passes) * (
        14 * units * rank + maximum_shortlist * maximum_shortlist * rank
    )
    build = 16 * units * rank + 24 * units
    return int(build + coordinate + local)


def _evaluate_group(task: int):
    if _CONTEXT is None:
        raise RuntimeError("evaluation worker lacks its context")
    started = time.perf_counter()
    group = _CONTEXT["groups"][int(task)]
    config = _CONTEXT["config"]
    local = _CONTEXT["local"]
    activation = np.asarray(local["x"][int(group["record"])], np.float32)
    weights = np.asarray(group["router_weights"], np.float64)
    responses = tuple(
        split_projection_responses(
            _CONTEXT["experts"][int(expert)]["q2"],
            _CONTEXT["experts"][int(expert)]["q4"],
            activation,
        )
        for expert in group["experts"]
    )
    identity = _identity(group)
    witness = np.asarray(_CONTEXT["witnesses"][identity], np.int64)
    budget = int(config["group_page_budget"])
    witness_pages = int(SPLIT_STATE_PAGE_COSTS[witness].sum())
    if witness_pages > budget:
        raise RuntimeError("frozen PR13 witness exceeds the pooled budget")
    proxy = _CONTEXT["proxy"]
    beta = float(_CONTEXT["beta"])
    base_states = np.zeros(8 * 512, np.int64)
    base_damage = _exact_group_damage(responses, weights, base_states, proxy, beta)
    witness_damage = _exact_group_damage(responses, weights, witness, proxy, beta)
    observed_recovery = 1.0 - witness_damage / base_damage
    if not np.isclose(
        observed_recovery, float(_CONTEXT["recoveries"][identity]),
        rtol=1e-9, atol=1e-9,
    ):
        raise RuntimeError("frozen PR13 witness no longer reproduces")

    result: list[tuple[str, np.ndarray, int, float, float, int, int, int, int, str]] = [(
        POLICIES[0], witness.copy(), witness_pages, witness_damage, 0.0,
        0, 0, 0, 0, "frozen_pr13_state",
    )]
    for policy, factor_id in (
        (POLICIES[1], str(config["primary_factor_id"])),
        (POLICIES[2], str(config["control_factor_id"])),
    ):
        fields = tuple(
            build_split_interaction_field(
                _CONTEXT["experts"][int(expert)]["factors"][factor_id],
                responses[index].hidden,
                _CONTEXT["experts"][int(expert)]["abc"],
            )
            for index, expert in enumerate(group["experts"])
        )
        joint = combine_split_interaction_fields(fields, weights)
        trace = optimize_joint_field(
            joint,
            budget,
            named_seeds=(("frozen_pr13_witness", witness),),
            coordinate_sweeps=int(config["coordinate_sweeps"]),
            local_shortlist=int(config["local_shortlist"]),
            local_swap_units=int(config["local_swap_units"]),
            local_max_passes=int(config["local_max_passes"]),
        )
        exact = _exact_group_damage(responses, weights, trace.states, proxy, beta)
        macs = _solver_macs(
            joint.rank, joint.units, trace.coordinate_sweeps,
            trace.local_passes, int(config["local_shortlist"]),
        )
        result.append((
            policy, trace.states, trace.pages, exact, trace.damage, macs,
            trace.coordinate_sweeps, trace.local_passes,
            trace.local_accepted_bundles, trace.seed_name,
        ))

    exact_fields = []
    for index, expert in enumerate(group["experts"]):
        response = responses[index]
        factor = exact_qmetric_factor(response.down2, response.down4, proxy, beta)
        exact_fields.append(build_split_interaction_field(
            factor, response.hidden, _CONTEXT["experts"][int(expert)]["abc"],
        ))
    exact_joint = combine_split_interaction_fields(tuple(exact_fields), weights)
    exact_trace = optimize_joint_field(
        exact_joint,
        budget,
        named_seeds=(("frozen_pr13_witness", witness),),
        coordinate_sweeps=int(config["coordinate_sweeps"]),
        local_shortlist=int(config["local_shortlist"]),
        local_swap_units=int(config["local_swap_units"]),
        local_max_passes=int(config["local_max_passes"]),
    )
    exact_damage = _exact_group_damage(
        responses, weights, exact_trace.states, proxy, beta,
    )
    if not np.isclose(exact_damage, exact_trace.damage, rtol=1e-8, atol=1e-7):
        raise RuntimeError("exact joint field lost qmetric parity")
    if exact_damage > witness_damage + float(config["dominance_tolerance"]):
        raise RuntimeError("exact joint solver regressed its immutable PR13 seed")
    result.append((
        POLICIES[3], exact_trace.states, exact_trace.pages, exact_damage,
        exact_trace.damage, _solver_macs(
            exact_joint.rank, exact_joint.units, exact_trace.coordinate_sweeps,
            exact_trace.local_passes, int(config["local_shortlist"]),
        ), exact_trace.coordinate_sweeps, exact_trace.local_passes,
        exact_trace.local_accepted_bundles, exact_trace.seed_name,
    ))

    rows, states_rows = [], []
    for (
        policy, states, pages, exact, predicted, macs, sweeps, passes,
        bundles, seed_name,
    ) in result:
        if policy == POLICIES[0]:
            factor_bytes, factor_rank, promotable = 6148, 8, False
        elif policy == POLICIES[1]:
            factor_bytes, factor_rank, promotable = 4100, 4, True
        elif policy == POLICIES[2]:
            factor_bytes, factor_rank, promotable = 16384, 4, False
        else:
            factor_bytes, factor_rank, promotable = 0, exact_joint.rank, False
        metadata = int(config["abc_metadata_bytes_per_expert"]) + factor_bytes
        recovery = 1.0 - float(exact) / base_damage
        row = {
            **{name: group[name] for name in GROUP_IDENTITY},
            "sequence_id": group["sequence_id"],
            "allocation_policy": policy,
            "streamed_mean_pages_per_expert": int(config["streamed_mean_pages_per_expert"]),
            "streamed_bpw": 0.5,
            "group_page_budget": budget,
            "actual_group_pages": int(pages),
            "average_actual_streamed_bpw": int(pages) / (8.0 * 768.0),
            "factor_rank": int(factor_rank),
            "factor_payload_bytes_per_expert": int(factor_bytes),
            "combined_metadata_bytes_per_expert": metadata,
            "metadata_bpw": 8.0 * metadata / int(config["expert_weights"]),
            "allowed_total_bpw": 0.5 + 8.0 * metadata / int(config["expert_weights"]),
            "group_base_qenergy_damage": base_damage,
            "group_exact_qenergy_damage": float(exact),
            "group_recovery": recovery,
            "compressed_predicted_damage": float(predicted),
            "recovery_delta_vs_pr13": recovery - observed_recovery,
            "damage_ratio_vs_pr13": float(exact) / max(witness_damage, 1e-30),
            "selected_states_sha256": _state_hash(states),
            "state_counts": json.dumps(
                np.bincount(states, minlength=8).tolist(), separators=(",", ":"),
            ),
            "seed_name": seed_name,
            "coordinate_sweeps": int(sweeps),
            "local_passes": int(passes),
            "local_accepted_bundles": int(bundles),
            "selector_compute_macs": int(macs),
            "promotable_geometry_control": bool(promotable),
            "exact_cross_expert_information_used_for_selection": policy == POLICIES[3],
            "validation_only": True,
        }
        rows.append(row)
        states_rows.append({
            **{name: group[name] for name in GROUP_IDENTITY},
            "sequence_id": group["sequence_id"],
            "allocation_policy": policy,
            "selected_states_sha256": row["selected_states_sha256"],
            "selected_states": json.dumps(states.tolist(), separators=(",", ":")),
        })
    return int(task), rows, states_rows, {
        "group_index": int(task),
        "wall_seconds": time.perf_counter() - started,
    }


def _evaluate(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    _validate_config(config)
    base_config, frozen_hashes = _verify_pr13(config, args)
    audit, input_hashes = pr13._locked_inputs(
        base_config, args.pr10_config, args.exact_captures,
        args.checkpoint, args.trees,
    )
    pr13._verify_base(base_config, args.pr12_config, args.pr12_dir)
    threads = pr13._thread_contract({
        **base_config,
        "evaluation_workers": int(config["evaluation_workers"]),
        "worker_blas_threads": int(config["worker_blas_threads"]),
    })
    dependencies = _dependencies()
    runtime = pr13.prior.runtime_provenance(args.device)
    witnesses, recoveries = _load_witnesses(config, args.pr13_validation_dir)
    data = pr13.prior.load_capture_admitted(args.exact_captures, ["train", "validation"])
    if "test" in set(map(str, np.unique(data["split"]))):
        raise RuntimeError("test row entered the low-rate continuation")
    groups = pr13._plan(data, base_config)
    if set(_identity(group) for group in groups) != set(witnesses):
        raise RuntimeError("current validation plan differs from the PR13 witness")
    manifest_path = args.pr13_fit_dir / pr13.FIT_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    frozen = {
        "run_id": config["run_id"],
        "phase": "evaluate",
        "selection_split": "validation",
        "test_scientific_rows_admitted_or_used": False,
        "validation_groups": len(groups),
        "policies_per_group": len(POLICIES),
        "config_sha256": _sha256(args.config),
        "capture_sha256": input_hashes["capture_sha256"],
        "tree_sha256": input_hashes["tree_sha256"],
        "checkpoint_config_sha256": input_hashes["config_sha256"],
        "checkpoint_index_sha256": input_hashes["index_sha256"],
        "runtime_provenance": runtime,
        "evaluation_workers": int(config["evaluation_workers"]),
        "worker_thread_environment": threads,
        **frozen_hashes,
        **dependencies,
    }
    group_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    if args.output.exists():
        facts_path = args.output / RUN_FACTS
        if not facts_path.is_file():
            raise RuntimeError("output exists without resumable facts")
        facts = json.loads(facts_path.read_text())
        for name, value in frozen.items():
            if facts.get(name) != value:
                raise RuntimeError(f"resume provenance changed: {name}")
        if facts.get("failures") or facts.get("failure_history"):
            raise RuntimeError("failed evaluation cannot be resumed")
        if facts.get("completed") is True:
            return
        completed = set(map(int, facts.get("completed_layers", [])))
        if completed:
            group_rows = pd.read_parquet(args.output / GROUP_FRONTIER).to_dict("records")
            state_rows = pd.read_parquet(args.output / STATE_EVIDENCE).to_dict("records")
            diagnostics = list(facts.get("diagnostics", []))
    else:
        args.output.mkdir(parents=True)
        facts = {
            "completed": False,
            "completed_layers": [],
            "failures": [],
            "failure_history": [],
            "checkpoint_audit": audit,
            "cpu_capacity": pr13._cpu_capacity(),
            "diagnostics": [],
            **frozen,
        }
        _atomic_json(args.output / RUN_FACTS, facts)
    index = json.loads(
        (args.checkpoint / "model.safetensors.index.json").read_text()
    )["weight_map"]
    tree_records = json.loads(args.trees.read_text())
    trees = {
        name: pr13.tree_from_record(tree_records[name]) for name in pr13.PROJECTIONS
    }
    try:
        for layer in map(int, config["layers"]):
            if layer in set(map(int, facts["completed_layers"])):
                continue
            local = pr13.prior.layer_view(data, layer)
            proxy, proxy_facts = base.proxy_gradients(
                local["xplus"],
                [local[f"h{h}_router_logits"] for h in range(1, 5)],
                local["split"],
            )
            beta = float(proxy_facts["beta"])
            record = manifest["layers"][str(layer)]
            shard = args.pr13_fit_dir / record["file"]
            sidecar = args.pr13_fit_dir / record["sidecar"]
            if (
                _sha256(shard) != record["sha256"]
                or shard.stat().st_size != int(record["bytes"])
                or _sha256(sidecar) != record["sidecar_sha256"]
            ):
                raise RuntimeError("immutable PR13 factor shard changed")
            arrays = dict(np.load(shard, allow_pickle=False))
            if not np.array_equal(np.asarray(arrays["proxy"], np.float32), np.asarray(proxy, np.float32)):
                raise RuntimeError("train-derived proxy changed")
            layer_groups = [group for group in groups if int(group["layer"]) == layer]
            active = sorted({int(expert) for group in layer_groups for expert in group["experts"]})
            experts = {}
            for expert in active:
                decoded = pr13.prior.decode_expert(
                    args.checkpoint, index, trees, layer, expert,
                )
                q2 = tuple(decoded[name][0] for name in pr13.PROJECTIONS)
                q4 = tuple(decoded[name][2] for name in pr13.PROJECTIONS)
                experts[expert] = {
                    "q2": q2,
                    "q4": q4,
                    "abc": pr13.unit_score_metadata(
                        q2[2], q4[2], proxy=proxy, beta=beta,
                    ),
                    "factors": pr13._load_layer_factors(arrays, base_config, expert),
                }
            global _CONTEXT
            _CONTEXT = {
                "groups": tuple(layer_groups),
                "local": local,
                "experts": experts,
                "proxy": proxy,
                "beta": beta,
                "config": config,
                "witnesses": witnesses,
                "recoveries": recoveries,
            }
            pending = {}
            context = mp.get_context(str(config["worker_start_method"]))
            workers = min(int(config["evaluation_workers"]), len(layer_groups))
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
                futures = {
                    executor.submit(_evaluate_group, task): task
                    for task in range(len(layer_groups))
                }
                for future in as_completed(futures):
                    observed_task, rows, state_values, diagnostic = future.result()
                    if observed_task != futures[future]:
                        raise RuntimeError("worker group identity changed")
                    pending[observed_task] = (rows, state_values, diagnostic)
            for task in range(len(layer_groups)):
                rows, state_values, diagnostic = pending[task]
                group_rows.extend(rows)
                state_rows.extend(state_values)
                diagnostics.append(diagnostic)
            _atomic_parquet(args.output / GROUP_FRONTIER, group_rows)
            _atomic_parquet(args.output / STATE_EVIDENCE, state_rows)
            facts["completed_layers"] = sorted(
                set(map(int, facts["completed_layers"])) | {layer}
            )
            facts["observed_group_rows"] = len(group_rows)
            facts["observed_state_rows"] = len(state_rows)
            facts["diagnostics"] = diagnostics
            _atomic_json(args.output / RUN_FACTS, facts)
            _CONTEXT = None
            del experts, arrays
            gc.collect()
        expected = int(config["expected_validation_groups"]) * len(POLICIES)
        if len(group_rows) != expected or len(state_rows) != expected:
            raise RuntimeError("completed row grid changed")
        frame = pd.DataFrame(group_rows)
        identity_columns = list(GROUP_IDENTITY) + ["allocation_policy"]
        if frame[identity_columns].duplicated().any():
            raise RuntimeError("duplicate policy/group row")
        if set(frame["allocation_policy"].astype(str)) != set(POLICIES):
            raise RuntimeError("policy coverage changed")
        if np.any(frame["actual_group_pages"].astype(int) > int(config["group_page_budget"])):
            raise RuntimeError("pooled page budget exceeded")
        accounting = {
            "schema_version": 1,
            "group_rows": len(group_rows),
            "state_rows": len(state_rows),
            "validation_groups": int(config["expected_validation_groups"]),
            "policies": list(POLICIES),
            "streamed_mean_pages_per_expert": 384,
            "streamed_bpw": 0.5,
            "group_page_budget": 3072,
            "diagnostics": diagnostics,
        }
        _atomic_json(args.output / ACCOUNTING, accounting)
        facts["completed"] = True
        facts["group_frontier_sha256"] = _sha256(args.output / GROUP_FRONTIER)
        facts["state_evidence_sha256"] = _sha256(args.output / STATE_EVIDENCE)
        facts["accounting_sha256"] = _sha256(args.output / ACCOUNTING)
        _atomic_json(args.output / RUN_FACTS, facts)
    except Exception as error:
        record = {"type": type(error).__name__, "message": str(error)}
        facts["failures"] = [record]
        facts.setdefault("failure_history", []).append(record)
        _atomic_json(args.output / RUN_FACTS, facts)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pr13-config", type=Path, required=True)
    parser.add_argument("--pr13-fit-dir", type=Path, required=True)
    parser.add_argument("--pr13-validation-dir", type=Path, required=True)
    parser.add_argument("--pr12-config", type=Path, required=True)
    parser.add_argument("--pr12-dir", type=Path, required=True)
    parser.add_argument("--pr10-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trees", type=Path, required=True)
    parser.add_argument("--exact-captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    _evaluate(args, json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
