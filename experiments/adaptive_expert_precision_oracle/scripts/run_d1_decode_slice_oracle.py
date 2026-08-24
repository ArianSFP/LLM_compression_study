#!/usr/bin/env python3
"""Exact-prefix, cached single-token D1 layer-slice oracle.

This is the decode-only successor to ``run_d1_slice_oracle_pilot.py``.  It
reuses that runner's exact complete-option frontier construction, but every
D1 gradient and replay starts from an exact prefix cache and injects only one
current token.  No prompt-position delta is allowed to influence another.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import pickle
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(EXPERIMENT / "scripts")]

from oracle_study.d1_decode import (  # noqa: E402
    cache_state_metrics,
    clone_decode_cache,
    mutation_safe_cached_autograd,
    tensor_state_metrics,
)
from oracle_study.d1_layer_slice import load_d1_layer_slice  # noqa: E402
from oracle_study.d1_route_objective import option_boundary_effects  # noqa: E402
import run_d1_slice_oracle_pilot as base  # noqa: E402


SCHEMA = "pr13_d1_exact_prefill_decode_slice_oracle_v2"
POLICY_PR13 = base.POLICY_PR13
POLICY_REGENERATED = base.POLICY_REGENERATED
POLICY_LOCAL = base.POLICY_LOCAL
POLICY_LOCAL_REFINED = base.POLICY_LOCAL_REFINED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trees", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--same-host-layers-dir", type=Path, required=True)
    parser.add_argument("--pr13-config", type=Path, required=True)
    parser.add_argument("--same-host-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--rates", type=int, nargs="+", required=True)
    parser.add_argument("--eta", type=float, nargs="+", required=True)
    parser.add_argument("--temperature", type=float, nargs="+", default=[0.0625])
    parser.add_argument("--minimum-prefix-tokens", type=int, default=1)
    parser.add_argument("--coarse-geometry-cache-dir", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--column-generated-frontiers", action="store_true")
    parser.add_argument(
        "--historical-pr13-mode",
        choices=("required", "if_available", "omit"),
        default="required",
    )
    return parser.parse_args()


def _coarse_geometry_identity(
    args: argparse.Namespace,
    layer: int,
    requests: Sequence[str],
    groups: Sequence[Mapping[str, Any]],
    factor_path: Path,
) -> dict[str, Any]:
    return {
        "schema": "pr13_d1_coarse_complete_option_geometry_cache_v1",
        "layer": int(layer),
        "group_identity": [
            [str(group["request_id"]), int(group["position"])] for group in groups
        ],
        "input_hashes": {
            "trees": base.sha256(args.trees),
            "factor": base.sha256(factor_path),
            "pr13_config": base.sha256(args.pr13_config),
            "base_runner": base.sha256(Path(base.__file__).resolve()),
            **{
                f"capture_{request_id}": base.sha256(
                    base._capture_path(args.capture_dir, request_id)
                )
                for request_id in requests
            },
        },
    }


def _atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _decode_groups(
    captures: Mapping[str, Mapping[str, np.ndarray]],
    layer: int,
    same_host_config: Mapping[str, Any],
    minimum_prefix_tokens: int,
) -> list[dict[str, Any]]:
    all_groups = base._groups(captures, layer, same_host_config)
    result = []
    for group in all_groups:
        if int(group["position"]) < int(minimum_prefix_tokens):
            continue
        record = dict(group)
        record["group"] = len(result)
        result.append(record)
    expected = int(same_host_config["expected_groups_per_layer"]) - (
        len(same_host_config["validation_request_ids"]) * int(minimum_prefix_tokens)
    )
    if len(result) != expected:
        raise RuntimeError(f"decode group count changed: {len(result)} != {expected}")
    return result


def _local_hidden(model_slice: Any, capture: Mapping[str, np.ndarray]) -> torch.Tensor:
    first = model_slice.compose_current_output(
        capture["residual"], capture["x"], capture["routed"],
    )
    second = model_slice.compose_current_output(
        capture["residual"], capture["x"], capture["routed"],
    )
    if not torch.equal(first, second):
        raise RuntimeError("exact current-layer reconstruction is not bit-repeatable")
    return first.detach()


def _paired_decode_baselines(
    model_slice: Any,
    captures: Mapping[str, Mapping[str, np.ndarray]],
    groups: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, torch.Tensor],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    list[dict[str, Any]],
]:
    hidden_by_request = {
        request_id: _local_hidden(model_slice, capture)
        for request_id, capture in captures.items()
    }
    full_logits = {}
    with torch.inference_mode():
        for request_id, hidden in hidden_by_request.items():
            logits, _, _ = model_slice.next_router_outputs(
                hidden, captures[request_id]["attention_mask"],
            )
            full_logits[request_id] = logits.detach()
    baseline_logits: dict[int, np.ndarray] = {}
    baseline_ids: dict[int, np.ndarray] = {}
    parity_rows = []
    for group in groups:
        index = int(group["group"])
        request_id = str(group["request_id"])
        position = int(group["position"])
        hidden = hidden_by_request[request_id]
        with torch.no_grad():
            prefix = model_slice.decode_prefix_cache(hidden[:, :position])
            first_cache = clone_decode_cache(prefix)
            second_cache = clone_decode_cache(prefix)
            first_logits, _, first_ids, first_final = (
                model_slice.next_router_outputs_decode(
                    hidden[:, position : position + 1], first_cache, position,
                )
            )
            second_logits, _, second_ids, second_final = (
                model_slice.next_router_outputs_decode(
                    hidden[:, position : position + 1], second_cache, position,
                )
            )
        cache_repeat = cache_state_metrics(
            first_final, second_final, model_slice.layer + 1,
        )
        if (
            not torch.equal(first_logits, second_logits)
            or not torch.equal(first_ids, second_ids)
            or not cache_repeat.bit_identical
        ):
            raise RuntimeError("cached decode baseline is not bit-repeatable")
        values = first_logits[0, 0].detach().float().cpu().numpy()
        ids = first_ids[0, 0].detach().cpu().numpy()
        baseline_logits[index] = values
        baseline_ids[index] = ids
        full_values = full_logits[request_id][0, position]
        full_ids = torch.topk(full_values, 8).indices.detach().cpu().numpy()
        parity_rows.append({
            "schema": SCHEMA,
            "layer": int(model_slice.layer),
            "next_layer": int(model_slice.layer) + 1,
            "next_mixer": str(model_slice.next_mixer_type),
            "group": index,
            "request_id": request_id,
            "position": position,
            "prefix_tokens": position,
            "paired_repeat_bit_identical": True,
            **cache_repeat.to_dict("paired_cache"),
            "cached_vs_full_router_max_abs": float(
                torch.max(torch.abs(first_logits[0, 0].float() - full_values.float())).item()
            ),
            "cached_vs_full_top8_set_equal": bool(
                set(ids.tolist()) == set(full_ids.tolist())
            ),
        })
    return hidden_by_request, baseline_logits, baseline_ids, parity_rows


def _decode_problems_and_effects(
    model_slice: Any,
    hidden_by_request: Mapping[str, torch.Tensor],
    groups: Sequence[Mapping[str, Any]],
    geometries: Sequence[Mapping[str, Any]],
    baseline_ids: Mapping[int, np.ndarray],
    temperatures: Sequence[float],
) -> tuple[dict[float, list[Any]], list[tuple[np.ndarray, ...]], list[dict[str, Any]]]:
    problems = {float(tau): [None] * len(groups) for tau in temperatures}
    effects: list[tuple[np.ndarray, ...] | None] = [None] * len(groups)
    rows = []
    for group in groups:
        index = int(group["group"])
        request_id = str(group["request_id"])
        position = int(group["position"])
        hidden = hidden_by_request[request_id]
        with torch.no_grad():
            prefix = model_slice.decode_prefix_cache(hidden[:, :position])
        current = hidden[:, position : position + 1].detach().clone().requires_grad_(True)
        # Native cached DeltaNet commits convolution/recurrent tensors in place.
        # Clone tensors as autograd saves them so the exact native forward can
        # commit its private cache without invalidating its backward graph.
        with mutation_safe_cached_autograd():
            logits, _ = model_slice.next_router_logits_decode(
                current, clone_decode_cache(prefix), position,
            )
            values = logits[0, 0]
            values_np = values.detach().float().cpu().numpy()
            exact_top8 = np.asarray(baseline_ids[index], np.int64)
            selected_set = set(exact_top8.tolist())
            outsiders = np.asarray(sorted(
                (expert for expert in range(len(values_np)) if expert not in selected_set),
                key=lambda expert: (-float(values_np[expert]), int(expert)),
            )[:8], np.int64)
            candidate_ids = np.concatenate((exact_top8[5:8], outsiders))
            sensitivities = np.zeros((values_np.size, current.shape[-1]), np.float64)
            started = time.perf_counter()
            for expert in candidate_ids.tolist():
                gradient = torch.autograd.grad(
                    values[int(expert)], current,
                    retain_graph=True,
                    create_graph=False,
                )[0]
                sensitivities[int(expert)] = (
                    gradient[0, 0].detach().float().cpu().numpy()
                )
            elapsed = time.perf_counter() - started
        for tau in temperatures:
            problems[float(tau)][index] = base._exact_boundary_problem(
                values_np, sensitivities, exact_top8, float(tau),
            )
        effects[index] = option_boundary_effects(
            geometries[index]["option_deltas"],
            np.asarray(group["execution_weights"], np.float64),
            problems[float(temperatures[0])][index],
        )
        rows.append({
            "schema": SCHEMA,
            "layer": int(group["layer"]),
            "next_layer": int(group["layer"]) + 1,
            "next_mixer": str(model_slice.next_mixer_type),
            "group": index,
            "request_id": request_id,
            "position": position,
            "prefix_tokens": position,
            "rank8_rank9_margin": float(values_np[exact_top8[7]] - values_np[outsiders[0]]),
            "exact_vjp_candidates": int(len(candidate_ids)),
            "exact_vjp_seconds": elapsed,
            "cache_conditioned": True,
            "native_cached_forward": True,
            "autograd_saved_tensors_cloned": True,
        })
        print(
            f"[decode VJP] layer={group['layer']} group={index + 1}/{len(groups)} "
            f"position={position} seconds={elapsed:.3f}",
            flush=True,
        )
        del logits, current, prefix
    if any(value is None for value in effects):
        raise RuntimeError("cached D1 option effects were incomplete")
    if any(any(value is None for value in values) for values in problems.values()):
        raise RuntimeError("cached D1 boundary problems were incomplete")
    return (
        {tau: list(values) for tau, values in problems.items()},
        list(effects),  # type: ignore[arg-type]
        rows,
    )


def _policy_deltas(
    groups: Sequence[Mapping[str, Any]],
    geometries: Sequence[Mapping[str, Any]],
    selections: Mapping[tuple[str, int], Sequence[Any]],
    allocation_rows: Sequence[Mapping[str, Any]],
    historical_path: Path,
    rates: Sequence[int],
    historical_mode: str,
) -> tuple[dict[tuple[str, int], np.ndarray], dict[tuple[str, int, int], float]]:
    allocation_lookup = {
        (str(row["policy"]), int(row["rate_pages_per_expert"]), int(row["group"])): row
        for row in allocation_rows
    }
    with np.load(historical_path, allow_pickle=False) as historical:
        historical_rates = np.asarray(historical["mean_budget_pages_per_expert"], np.int64)
        historical_request = np.asarray(historical["request_id"]).astype(str)
        historical_position = np.asarray(historical["position"], np.int64)
        historical_delta = np.asarray(historical["delta"], np.float32)
        historical_damage = np.asarray(historical["group_damage"], np.float64)
    rate_index = {int(rate): index for index, rate in enumerate(historical_rates)}
    generated = sorted({str(policy) for policy, _ in selections})
    deltas = {}
    damage = {}
    for rate in map(int, rates):
        available = rate in rate_index
        if historical_mode == "required" and not available:
            raise RuntimeError(f"historical delta omits rate {rate}")
        policies = list(generated)
        if historical_mode != "omit" and available:
            policies.append(POLICY_PR13)
        for policy in sorted(policies):
            complete = np.zeros((len(groups), 2048), np.float32)
            if policy == POLICY_PR13:
                for group in groups:
                    mask = (
                        (historical_request == str(group["request_id"]))
                        & (historical_position == int(group["position"]))
                    )
                    found = np.flatnonzero(mask)
                    if len(found) != 1:
                        raise RuntimeError("historical decode group is not unique")
                    source = int(found[0])
                    index = int(group["group"])
                    complete[index] = historical_delta[rate_index[rate], source]
                    damage[(policy, rate, index)] = float(
                        historical_damage[rate_index[rate], source]
                    )
            else:
                for group, geometry, trace in zip(
                    groups, geometries, selections[(policy, rate)],
                ):
                    index = int(group["group"])
                    complete[index] = base._delta_for_trace(
                        geometry,
                        np.asarray(group["execution_weights"], np.float64),
                        trace,
                    )
                    damage[(policy, rate, index)] = float(
                        allocation_lookup[(policy, rate, index)]["local_qenergy_damage"]
                    )
            deltas[(policy, rate)] = complete
    return deltas, damage


def _decode_exact_replays(
    model_slice: Any,
    captures: Mapping[str, Mapping[str, np.ndarray]],
    hidden_by_request: Mapping[str, torch.Tensor],
    groups: Sequence[Mapping[str, Any]],
    deltas: Mapping[tuple[str, int], np.ndarray],
    damage: Mapping[tuple[str, int, int], float],
) -> list[dict[str, Any]]:
    rows = []
    policies_by_rate: dict[int, list[str]] = {}
    for policy, rate in deltas:
        policies_by_rate.setdefault(int(rate), []).append(str(policy))
    for group in groups:
        index = int(group["group"])
        request_id = str(group["request_id"])
        position = int(group["position"])
        capture = captures[request_id]
        hidden = hidden_by_request[request_id]
        with torch.no_grad():
            prefix = model_slice.decode_prefix_cache(hidden[:, :position])
            baseline_logits, _, baseline_ids, baseline_cache = (
                model_slice.next_router_outputs_decode(
                    hidden[:, position : position + 1],
                    clone_decode_cache(prefix),
                    position,
                )
            )
            for rate, policies in sorted(policies_by_rate.items()):
                for policy in sorted(policies):
                    candidate_hidden = model_slice.compose_current_output(
                        capture["residual"][position : position + 1],
                        capture["x"][position : position + 1],
                        capture["routed"][position : position + 1],
                        deltas[(policy, rate)][index],
                    )
                    candidate_logits, _, candidate_ids, candidate_cache = (
                        model_slice.next_router_outputs_decode(
                            candidate_hidden,
                            clone_decode_cache(prefix),
                            position,
                        )
                    )
                    cache_metrics = cache_state_metrics(
                        baseline_cache, candidate_cache, model_slice.layer + 1,
                    )
                    rows.append({
                        "schema": SCHEMA,
                        "layer": int(group["layer"]),
                        "next_layer": int(group["layer"]) + 1,
                        "next_mixer": str(model_slice.next_mixer_type),
                        "group": index,
                        "request_id": request_id,
                        "position": position,
                        "prefix_tokens": position,
                        "policy": policy,
                        "rate_pages_per_expert": int(rate),
                        "local_qenergy_damage": damage[(policy, rate, index)],
                        **tensor_state_metrics(
                            hidden[:, position : position + 1],
                            candidate_hidden,
                            "current_hidden",
                        ),
                        **cache_metrics.to_dict("next_mixer_cache"),
                        **base._route_metrics(
                            baseline_logits[0, 0].float().cpu().numpy(),
                            candidate_logits[0, 0].float().cpu().numpy(),
                            baseline_ids[0, 0].cpu().numpy(),
                            candidate_ids[0, 0].cpu().numpy(),
                        ),
                    })
        print(
            f"[decode replay] layer={group['layer']} group={index + 1}/{len(groups)}",
            flush=True,
        )
    return rows


def _token_oracle(exact: pd.DataFrame, allocations: pd.DataFrame) -> pd.DataFrame:
    candidates = exact[exact["policy"].str.startswith("d1_strict_")].copy()
    pages = allocations[[
        "group", "policy", "rate_pages_per_expert", "selected_group_pages",
    ]].drop_duplicates()
    candidates = candidates.merge(
        pages,
        on=["group", "policy", "rate_pages_per_expert"],
        validate="many_to_one",
    )
    choices = []
    for _, frame in candidates.groupby(
        ["layer", "group", "rate_pages_per_expert"], sort=True,
    ):
        choices.append(frame.sort_values([
            "exact_d1_crossed", "membership_pairs_changed", "routing_mass_lost",
            "local_qenergy_damage", "selected_group_pages", "policy",
        ], kind="stable").iloc[0].to_dict())
    return pd.DataFrame(choices)


def run_layer(args: argparse.Namespace, layer: int) -> dict[str, Any]:
    started = time.perf_counter()
    pr13_config = base.load_json(args.pr13_config)
    same_host_config = base.load_json(args.same_host_config)
    requests = list(map(str, same_host_config["validation_request_ids"]))
    captures = {
        request_id: base._load_capture(
            base._capture_path(args.capture_dir, request_id), int(layer),
        )
        for request_id in requests
    }
    groups = _decode_groups(
        captures, int(layer), same_host_config, int(args.minimum_prefix_tokens),
    )
    factor_path = args.fit_dir / f"average_rate_factor_layer_{int(layer)}.npz"
    factor_sidecar = args.fit_dir / f"average_rate_factor_layer_{int(layer)}.json"
    if base.load_json(factor_sidecar)["sha256"] != base.sha256(factor_path):
        raise RuntimeError("factor sidecar hash mismatch")
    arrays = dict(np.load(factor_path, allow_pickle=False))
    proxy = np.asarray(arrays["proxy"], np.float32)
    beta = float(np.asarray(arrays["beta"]).reshape(-1)[0])
    cache_identity = _coarse_geometry_identity(
        args, int(layer), requests, groups, factor_path,
    )
    cache_path = None
    geometry_cache_status = "disabled"
    if args.coarse_geometry_cache_dir is not None:
        cache_path = (
            args.coarse_geometry_cache_dir
            / f"coarse_geometry_layer_{int(layer):02d}.pkl"
        )
    if cache_path is not None and cache_path.is_file():
        with cache_path.open("rb") as handle:
            cached = pickle.load(handle)
        if cached.get("identity") != cache_identity:
            raise RuntimeError(f"coarse geometry cache identity changed: {cache_path}")
        geometries = cached["geometries"]
        if len(geometries) != len(groups):
            raise RuntimeError("coarse geometry cache group count changed")
        geometry_cache_status = "loaded"
        print(f"[layer {layer}] loaded coarse geometry cache {cache_path}", flush=True)
    else:
        experts = base._prepare_experts(
            args, int(layer), groups, arrays, proxy, beta, pr13_config,
        )
        geometries = base._build_geometries(
            groups, experts, proxy, beta, pr13_config, args.workers, int(layer),
        )
        del experts
        if cache_path is not None:
            _atomic_pickle(cache_path, {
                "identity": cache_identity,
                "geometries": geometries,
            })
            geometry_cache_status = "created"
            print(f"[layer {layer}] wrote coarse geometry cache {cache_path}", flush=True)
    local_policy = POLICY_LOCAL
    if args.column_generated_frontiers:
        geometries = base._refine_geometries(
            geometries, groups, proxy, beta, pr13_config,
            int(args.rates[0]), args.workers, int(layer),
        )
        local_policy = POLICY_LOCAL_REFINED
    del arrays
    gc.collect()

    model_slice = load_d1_layer_slice(args.checkpoint, int(layer), device=args.device)
    hidden, baseline_logits, baseline_ids, parity_rows = _paired_decode_baselines(
        model_slice, captures, groups,
    )
    problems, effects, gradient_rows = _decode_problems_and_effects(
        model_slice, hidden, groups, geometries, baseline_ids,
        list(map(float, args.temperature)),
    )
    selections, allocation_rows, expert_rows = base._allocation_policies(
        groups, geometries, problems, effects,
        list(map(int, args.rates)), list(map(float, args.eta)),
        pr13_config, local_policy,
    )
    historical_path = (
        args.same_host_layers_dir / f"same_host_delta_layer_{int(layer):02d}.npz"
    )
    deltas, damage = _policy_deltas(
        groups, geometries, selections, allocation_rows, historical_path,
        list(map(int, args.rates)), str(args.historical_pr13_mode),
    )
    exact_rows = _decode_exact_replays(
        model_slice, captures, hidden, groups, deltas, damage,
    )
    layer_dir = args.output_dir / f"layer_{int(layer):02d}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    exact = pd.DataFrame(exact_rows)
    allocations = pd.DataFrame(allocation_rows)
    base.atomic_parquet(layer_dir / "d1_decode_exact_route_metrics.parquet", exact)
    base.atomic_parquet(layer_dir / "d1_decode_allocation_groups.parquet", allocations)
    base.atomic_parquet(layer_dir / "d1_decode_allocation_experts.parquet", expert_rows)
    base.atomic_parquet(layer_dir / "d1_decode_vjp_metrics.parquet", gradient_rows)
    base.atomic_parquet(layer_dir / "d1_decode_cache_parity.parquet", parity_rows)
    oracle = _token_oracle(exact, allocations)
    base.atomic_parquet(layer_dir / "d1_decode_token_oracle.parquet", oracle)
    base.atomic_npz(
        layer_dir / "d1_decode_selected_deltas.npz",
        schema=np.asarray(SCHEMA),
        layer=np.asarray(int(layer), np.int64),
        **{f"{policy}__rate_{rate}": value for (policy, rate), value in deltas.items()},
    )
    summary = (
        exact.groupby(["layer", "policy", "rate_pages_per_expert"], sort=True)
        .agg(
            groups=("group", "size"),
            exact_d1_crossing_rate=("exact_d1_crossed", "mean"),
            mean_local_qenergy_damage=("local_qenergy_damage", "mean"),
            mean_current_hidden_mse=("current_hidden_mse", "mean"),
            mean_next_mixer_cache_mse=("next_mixer_cache_mse", "mean"),
        )
        .reset_index()
    )
    base.atomic_parquet(layer_dir / "d1_decode_policy_summary.parquet", summary)
    facts = {
        "completed": True,
        "schema": SCHEMA,
        "layer": int(layer),
        "next_layer": int(layer) + 1,
        "next_mixer": str(model_slice.next_mixer_type),
        "groups": len(groups),
        "minimum_prefix_tokens": int(args.minimum_prefix_tokens),
        "rates": list(map(int, args.rates)),
        "eta": list(map(float, args.eta)),
        "temperature": list(map(float, args.temperature)),
        "frontier_mode": "column_generated" if args.column_generated_frontiers else "coarse",
        "historical_pr13_mode": str(args.historical_pr13_mode),
        "coarse_geometry_cache": {
            "status": geometry_cache_status,
            "path": None if cache_path is None else str(cache_path),
            "sha256": (
                None if cache_path is None else base.sha256(cache_path)
            ),
            "identity": cache_identity,
        },
        "policies": sorted(exact["policy"].unique().tolist()),
        "paired_cached_baseline_bit_identical": bool(
            all(row["paired_repeat_bit_identical"] for row in parity_rows)
        ),
        "cache_state_parity_required": True,
        "single_current_token_injection": True,
        "prompt_position_coupling_allowed": False,
        "wall_seconds": time.perf_counter() - started,
        "files": {
            path.name: {"sha256": base.sha256(path), "bytes": path.stat().st_size}
            for path in sorted(layer_dir.iterdir())
            if path.is_file() and path.name != "d1_decode_layer_facts.json"
        },
        "environment": {
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "input_hashes": {
            "checkpoint_index": base.sha256(
                args.checkpoint / "model.safetensors.index.json"
            ),
            "trees": base.sha256(args.trees),
            "pr13_config": base.sha256(args.pr13_config),
            "same_host_config": base.sha256(args.same_host_config),
            "factor": base.sha256(factor_path),
            "historical_delta": base.sha256(historical_path),
            **{
                f"capture_{request_id}": base.sha256(
                    base._capture_path(args.capture_dir, request_id)
                )
                for request_id in requests
            },
        },
        "test_rows_admitted_or_used": False,
        "scientific_boundary": (
            "Exact reconstructed-prefix cached single-token D1 slice surrogate. "
            "No full-model exact-prefill handoff, terminal KL, temporal rollout, "
            "D2-D4 objective, or runtime-controller claim."
        ),
    }
    base.atomic_json(layer_dir / "d1_decode_layer_facts.json", facts)
    print(summary.to_string(index=False), flush=True)
    del model_slice, geometries, hidden
    torch.cuda.empty_cache()
    gc.collect()
    return facts


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.workers > 32:
        raise ValueError("workers must lie in [1,32]")
    if args.minimum_prefix_tokens < 1:
        raise ValueError("decode-only pilot requires at least one exact prefix token")
    if any(rate < 1 or rate > 1536 for rate in args.rates):
        raise ValueError("rates must lie in the physical page range")
    if any(value < 0.0 for value in args.eta):
        raise ValueError("eta must be nonnegative")
    if any(value <= 0.0 for value in args.temperature):
        raise ValueError("temperature must be positive")
    if args.column_generated_frontiers and len(args.rates) != 1:
        raise ValueError("column-generated mode requires one rate per invocation")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    facts = [run_layer(args, int(layer)) for layer in args.layers]
    base.atomic_json(args.output_dir / "d1_decode_pilot_run_facts.json", {
        "completed": True,
        "schema": SCHEMA,
        "layers": {str(row["layer"]): row for row in facts},
        "runner_sha256": base.sha256(Path(__file__).resolve()),
        "decode_core_sha256": base.sha256(
            EXPERIMENT / "src/oracle_study/d1_decode.py"
        ),
        "slice_core_sha256": base.sha256(
            EXPERIMENT / "src/oracle_study/d1_layer_slice.py"
        ),
        "objective_core_sha256": base.sha256(
            EXPERIMENT / "src/oracle_study/d1_route_objective.py"
        ),
        "test_rows_admitted_or_used": False,
    })


if __name__ == "__main__":
    main()
