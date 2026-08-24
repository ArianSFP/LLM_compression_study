#!/usr/bin/env python3
"""Matched-rate D1 route oracle over exact PR #13 frontier options.

This pilot deliberately separates the scientific allocation objective from the
sequential runtime controller. It evaluates complete frontier options at a
fixed all-in page budget, then executes each final request-level allocation
once through a paired 3090 layer slice.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import gc
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(EXPERIMENT / "scripts")]

from oracle_study.average_rate_allocator import (  # noqa: E402
    exact_group_option_allocate,
    lagrangian_rate_frontier,
    multiple_choice_allocate,
)
from oracle_study.causal_control import (  # noqa: E402
    selector_and_execution_router_weights,
)
from oracle_study.d1_layer_slice import load_d1_layer_slice  # noqa: E402
from oracle_study.d1_route_objective import (  # noqa: E402
    D1BoundaryProblem,
    additive_local_damage_limit,
    d1_group_option_allocate,
    option_boundary_effects,
    token_option_deltas,
)
from oracle_study.neuron_selector import unit_score_metadata  # noqa: E402
from oracle_study.split_interaction_field import (  # noqa: E402
    build_split_interaction_field,
    split_projection_responses,
)
from run_mxfp4_selective_pages import tree_from_record  # noqa: E402
import run_average_rate_allocation as pr13  # noqa: E402
import run_set_utility_distillation as set_study  # noqa: E402


SCHEMA = "pr13_d1_layer_slice_oracle_v1"
POLICY_PR13 = "pr13_router_square_column_generated"
POLICY_REGENERATED = "regenerated_router_square_column_generated"
POLICY_LOCAL = "exact_combined_local_coarse_frontier"
POLICY_LOCAL_REFINED = "exact_combined_local_column_generated_frontier"
PAGE_BYTES = 512
_GEOMETRY_CONTEXT: dict[str, Any] | None = None
_REFINE_CONTEXT: dict[str, Any] | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_parquet(path: Path, rows: Sequence[Mapping[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


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
    parser.add_argument("--rates", type=int, nargs="+", default=[384, 749])
    parser.add_argument("--eta", type=float, nargs="+", default=[0.05])
    parser.add_argument("--temperature", type=float, nargs="+", default=[0.0625])
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--column-generated-frontiers", action="store_true")
    parser.add_argument(
        "--historical-pr13-mode",
        choices=("required", "if_available", "omit"),
        default="required",
        help=(
            "Whether preserved historical PR13 deltas must be replayed. Use "
            "'omit' for explicitly non-historical matched-metadata page caps."
        ),
    )
    return parser.parse_args()


def _capture_path(directory: Path, request_id: str) -> Path:
    for path in (
        directory / f"{request_id}.npz",
        directory / "baseline" / f"{request_id}.npz",
    ):
        if path.exists():
            return path
    raise FileNotFoundError(f"missing capture for {request_id}")


def _load_capture(path: Path, layer: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {
            "input_ids": np.asarray(source["input_ids"]),
            "attention_mask": (
                np.asarray(source["attention_mask"])
                if "attention_mask" in source.files
                else np.ones(len(source["input_ids"]), np.int64)
            ),
            "hidden": np.asarray(source["hidden"], np.float32),
            "router_logits": np.asarray(source["router_logits"], np.float32),
            "router_scores": np.asarray(source["router_scores"], np.float32),
            "router_ids": np.asarray(source["router_ids"], np.int64),
            "residual": np.asarray(source[f"residual_layer_{layer:02d}"], np.float32),
            "x": np.asarray(source[f"x_layer_{layer:02d}"], np.float32),
            "routed": np.asarray(source[f"routed_layer_{layer:02d}"], np.float32),
        }


def _groups(
    captures: Mapping[str, Mapping[str, np.ndarray]],
    layer: int,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result = []
    excluded = int(config["exclude_final_positions"])
    for request_id in map(str, config["validation_request_ids"]):
        capture = captures[request_id]
        for position in range(len(capture["input_ids"]) - excluded):
            ids = np.asarray(capture["router_ids"][layer, position], np.int64)
            selector, execution = selector_and_execution_router_weights(
                capture["router_scores"][layer, position],
                historical_sum_atol=float(config["historical_router_sum_atol"]),
                execution_sum_atol=float(config["same_host_execution_router_sum_atol"]),
            )
            result.append({
                "group": len(result),
                "request_id": request_id,
                "position": position,
                "layer": int(layer),
                "experts": ids,
                "selector_weights": selector,
                "execution_weights": execution,
                "activation": np.asarray(capture["x"][position], np.float32),
            })
    if len(result) != int(config["expected_groups_per_layer"]):
        raise RuntimeError("validation group count changed")
    return result


def _primary_factor(
    arrays: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    expert: int,
) -> Any:
    rank = int(config["factor_fit_rank"])
    encoded = pr13.EncodedInteractionFactor(
        packed_codes=np.asarray(arrays["primary_packed_codes"][expert], np.uint8),
        row_scales=np.asarray(arrays["primary_row_scales"][expert], np.float16),
        units=pr13.UNITS,
        rank=rank,
        method="primary",
        tail_rank=rank,
        exact_rank=0,
        encoding="int4_hadamard",
        global_scale=np.float32(arrays["primary_global_scales"][expert]),
    )
    return encoded.decode()


def _prepare_experts(
    args: argparse.Namespace,
    layer: int,
    groups: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    proxy: np.ndarray,
    beta: float,
    pr13_config: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    index = load_json(args.checkpoint / "model.safetensors.index.json")["weight_map"]
    tree_records = load_json(args.trees)
    trees = {
        name: tree_from_record(tree_records[name])
        for name in pr13.PROJECTIONS
    }
    active = sorted({
        int(expert) for group in groups for expert in group["experts"]
    })

    def prepare(expert: int) -> tuple[int, dict[str, Any]]:
        decoded = set_study.decode_expert(
            args.checkpoint, index, trees, layer, expert,
        )
        q2 = tuple(decoded[name][0] for name in pr13.PROJECTIONS)
        q4 = tuple(decoded[name][2] for name in pr13.PROJECTIONS)
        return expert, {
            "q2": q2,
            "q4": q4,
            "abc": unit_score_metadata(q2[2], q4[2], proxy=proxy, beta=beta),
            "factor": _primary_factor(arrays, pr13_config, expert),
        }

    print(f"[layer {layer}] decoding {len(active)} active experts", flush=True)
    result: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(args.workers, len(active))) as executor:
        futures = {executor.submit(prepare, expert): expert for expert in active}
        for completed, future in enumerate(as_completed(futures), start=1):
            expert, cell = future.result()
            result[int(expert)] = cell
            if completed % 16 == 0 or completed == len(active):
                print(
                    f"[layer {layer}] decoded {completed}/{len(active)} experts",
                    flush=True,
                )
    return result


def _group_geometry(
    group: Mapping[str, Any],
    experts: Mapping[int, Mapping[str, Any]],
    proxy: np.ndarray,
    beta: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    responses = tuple(
        split_projection_responses(
            experts[int(expert)]["q2"],
            experts[int(expert)]["q4"],
            np.asarray(group["activation"], np.float32),
        )
        for expert in group["experts"]
    )
    frontiers = []
    fields = []
    features = []
    exact_damage = []
    for expert, response in zip(group["experts"], responses):
        cell = experts[int(expert)]
        field = build_split_interaction_field(
            cell["factor"], response.hidden, cell["abc"],
        )
        trace = lagrangian_rate_frontier(
            field,
            maximum_pages=int(config["frontier_maximum_pages"]),
            target_pages=config["frontier_target_pages"],
            price_ratios=config["frontier_price_ratios"],
            coordinate_sweeps=int(config["coordinate_sweeps"]),
            local_shortlist=int(config["local_shortlist"]),
            local_swap_units=int(config["local_swap_units"]),
            local_max_passes=int(config["local_max_passes"]),
            price_local_max_passes=int(config["price_local_max_passes"]),
        )
        values, damage = pr13._exact_option_data(response, trace.options, proxy, beta)
        fields.append(field)
        frontiers.append(trace.options)
        features.append(values)
        exact_damage.append(damage)
    return {
        "responses": responses,
        "fields": tuple(fields),
        "frontiers": tuple(frontiers),
        "features": tuple(features),
        "exact_damage": tuple(exact_damage),
        "option_deltas": token_option_deltas(responses, frontiers),
    }


def _geometry_task(group_index: int) -> tuple[int, dict[str, Any]]:
    if _GEOMETRY_CONTEXT is None:
        raise RuntimeError("forked geometry context is not initialized")
    index = int(group_index)
    group = _GEOMETRY_CONTEXT["groups"][index]
    return index, _group_geometry(
        group,
        _GEOMETRY_CONTEXT["experts"],
        _GEOMETRY_CONTEXT["proxy"],
        _GEOMETRY_CONTEXT["beta"],
        _GEOMETRY_CONTEXT["config"],
    )

def _build_geometries(
    groups: Sequence[Mapping[str, Any]],
    experts: Mapping[int, Mapping[str, Any]],
    proxy: np.ndarray,
    beta: float,
    config: Mapping[str, Any],
    workers: int,
    layer: int,
) -> list[dict[str, Any]]:
    global _GEOMETRY_CONTEXT
    print(f"[layer {layer}] building {len(groups)} exact option geometries", flush=True)
    result: list[dict[str, Any] | None] = [None] * len(groups)
    _GEOMETRY_CONTEXT = {
        "groups": tuple(groups),
        "experts": experts,
        "proxy": proxy,
        "beta": float(beta),
        "config": config,
    }
    try:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(groups)),
            mp_context=mp.get_context("fork"),
        ) as executor:
            futures = {
                executor.submit(_geometry_task, index): index
                for index in range(len(groups))
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                group_index, geometry = future.result()
                if int(group_index) != int(futures[future]):
                    raise RuntimeError("forked geometry identity changed")
                result[int(group_index)] = geometry
                if completed % 8 == 0 or completed == len(groups):
                    print(
                        f"[layer {layer}] built {completed}/{len(groups)} geometries",
                        flush=True,
                    )
    finally:
        _GEOMETRY_CONTEXT = None
    if any(value is None for value in result):
        raise RuntimeError("frontier geometry construction was incomplete")
    return list(result)  # type: ignore[arg-type]


def _refine_geometry_task(group_index: int) -> tuple[int, dict[str, Any]]:
    if _REFINE_CONTEXT is None:
        raise RuntimeError("forked refinement context is not initialized")
    index = int(group_index)
    geometry = _REFINE_CONTEXT["geometries"][index]
    group = _REFINE_CONTEXT["groups"][index]
    config = _REFINE_CONTEXT["config"]
    trace = pr13.allocation_aware_rate_frontiers(
        geometry["fields"],
        geometry["frontiers"],
        page_budget=8 * int(_REFINE_CONTEXT["rate"]),
        burst_cap_pages=int(config["frontier_maximum_pages"]),
        weights=np.asarray(group["selector_weights"], np.float64) ** 2,
        coordinate_sweeps=int(config["coordinate_sweeps"]),
        local_shortlist=int(config["local_shortlist"]),
        local_swap_units=int(config["local_swap_units"]),
        local_max_passes=int(config["local_max_passes"]),
        max_rounds=int(config["column_generation_max_rounds"]),
        adaptive_price_multipliers=config["column_generation_price_multipliers"],
        adaptive_price_local_passes=int(
            config["column_generation_price_local_max_passes"]
        ),
    )
    features = []
    exact_damage = []
    for response, options in zip(geometry["responses"], trace.frontiers):
        values, damage = pr13._exact_option_data(
            response,
            options,
            _REFINE_CONTEXT["proxy"],
            _REFINE_CONTEXT["beta"],
        )
        features.append(values)
        exact_damage.append(damage)
    result = dict(geometry)
    result.update({
        "frontiers": trace.frontiers,
        "features": tuple(features),
        "exact_damage": tuple(exact_damage),
        "option_deltas": token_option_deltas(
            geometry["responses"], trace.frontiers,
        ),
        "column_generation_rounds": int(trace.rounds),
        "column_generation_repairs": int(trace.selected_repairs),
    })
    return index, result


def _refine_geometries(
    geometries: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    proxy: np.ndarray,
    beta: float,
    config: Mapping[str, Any],
    rate: int,
    workers: int,
    layer: int,
) -> list[dict[str, Any]]:
    global _REFINE_CONTEXT
    print(
        f"[layer {layer}] column-generating {len(groups)} rate-{rate} geometries",
        flush=True,
    )
    result: list[dict[str, Any] | None] = [None] * len(groups)
    _REFINE_CONTEXT = {
        "geometries": tuple(geometries),
        "groups": tuple(groups),
        "proxy": proxy,
        "beta": float(beta),
        "config": config,
        "rate": int(rate),
    }
    try:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(groups)),
            mp_context=mp.get_context("fork"),
        ) as executor:
            futures = {
                executor.submit(_refine_geometry_task, index): index
                for index in range(len(groups))
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                group_index, geometry = future.result()
                if int(group_index) != int(futures[future]):
                    raise RuntimeError("forked refinement identity changed")
                result[int(group_index)] = geometry
                if completed % 8 == 0 or completed == len(groups):
                    print(
                        f"[layer {layer}] refined {completed}/{len(groups)} geometries",
                        flush=True,
                    )
    finally:
        _REFINE_CONTEXT = None
    if any(value is None for value in result):
        raise RuntimeError("column-generated geometry grid was incomplete")
    return list(result)  # type: ignore[arg-type]


def _paired_baselines(
    model_slice: Any,
    captures: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    list[dict[str, Any]],
]:
    hidden_by_request = {}
    logits_by_request = {}
    ids_by_request = {}
    parity_rows = []
    for request_id, capture in captures.items():
        with torch.inference_mode():
            hidden_first = model_slice.compose_current_output(
                capture["residual"], capture["x"], capture["routed"],
            )
            logits_first, _, ids_first = model_slice.next_router_outputs(
                hidden_first, capture["attention_mask"],
            )
            hidden_second = model_slice.compose_current_output(
                capture["residual"], capture["x"], capture["routed"],
            )
            logits_second, _, ids_second = model_slice.next_router_outputs(
                hidden_second, capture["attention_mask"],
            )
        local_repeat = bool(
            torch.equal(hidden_first, hidden_second)
            and torch.equal(logits_first, logits_second)
            and torch.equal(ids_first, ids_second)
        )
        if not local_repeat:
            raise RuntimeError("paired 3090 zero-dose slice is not bit-repeatable")
        local_hidden = hidden_first.detach()
        local_logits = logits_first.detach()
        local_ids = ids_first.detach()
        hidden_by_request[request_id] = local_hidden
        logits_by_request[request_id] = local_logits
        ids_by_request[request_id] = local_ids
        stored_hidden = torch.as_tensor(
            capture["hidden"][model_slice.layer],
            device=local_hidden.device,
            dtype=torch.bfloat16,
        ).unsqueeze(0)
        stored_logits = np.asarray(
            capture["router_logits"][model_slice.layer + 1], np.float32,
        )
        stored_ids = np.asarray(
            capture["router_ids"][model_slice.layer + 1], np.int64,
        )
        local_logits_np = local_logits[0].float().cpu().numpy()
        local_ids_np = local_ids[0].cpu().numpy()
        parity_rows.append({
            "request_id": request_id,
            "paired_repeat_bit_identical": local_repeat,
            "stored_current_hidden_max_abs": float(
                torch.max(torch.abs(local_hidden.float() - stored_hidden.float())).item()
            ),
            "stored_next_router_max_abs": float(
                np.max(np.abs(local_logits_np - stored_logits), initial=0.0)
            ),
            "stored_next_top8_set_exact_fraction": float(np.mean([
                set(left.tolist()) == set(right.tolist())
                for left, right in zip(local_ids_np, stored_ids)
            ])),
        })
    return hidden_by_request, logits_by_request, ids_by_request, parity_rows

def _exact_boundary_problem(
    values: np.ndarray,
    sensitivities: np.ndarray,
    exact_top8: np.ndarray,
    temperature: float,
) -> D1BoundaryProblem:
    selected_top8 = np.asarray(exact_top8, np.int64).reshape(-1)
    if selected_top8.shape != (8,) or len(set(selected_top8.tolist())) != 8:
        raise RuntimeError("paired GPU router did not return eight unique experts")
    selected = selected_top8[5:8]
    selected_set = set(selected_top8.tolist())
    outsiders = np.asarray(sorted(
        (expert for expert in range(len(values)) if expert not in selected_set),
        key=lambda expert: (-float(values[expert]), int(expert)),
    )[:8], np.int64)
    left = np.repeat(selected, len(outsiders))
    right = np.tile(outsiders, len(selected))
    margins = np.asarray(values[left] - values[right], np.float64)
    if np.any(margins < -1e-12):
        raise RuntimeError("exact GPU top-k membership contradicts router logits")
    margins = np.maximum(margins, 0.0)
    count = len(margins)
    return D1BoundaryProblem(
        selected_expert_ids=left,
        outsider_expert_ids=right,
        q4_margins=margins,
        margin_sensitivities=sensitivities[left] - sensitivities[right],
        severity=np.ones(count, np.float64),
        temperature=np.full(count, float(temperature), np.float64),
        safety_margin=np.zeros(count, np.float64),
    )


def _problems_and_effects(
    model_slice: Any,
    captures: Mapping[str, Mapping[str, np.ndarray]],
    groups: Sequence[Mapping[str, Any]],
    geometries: Sequence[Mapping[str, Any]],
    hidden_by_request: Mapping[str, torch.Tensor],
    ids_by_request: Mapping[str, torch.Tensor],
    temperatures: Sequence[float],
) -> tuple[
    dict[float, list[Any]], list[tuple[np.ndarray, ...]], list[dict[str, Any]]
]:
    problems = {float(tau): [None] * len(groups) for tau in temperatures}
    effects: list[tuple[np.ndarray, ...] | None] = [None] * len(groups)
    gradient_rows = []
    groups_by_request: dict[str, list[Mapping[str, Any]]] = {}
    for group in groups:
        groups_by_request.setdefault(str(group["request_id"]), []).append(group)

    for request_id, request_groups in groups_by_request.items():
        hidden = hidden_by_request[request_id].detach().clone().requires_grad_(True)
        logits = model_slice.next_router_logits(
            hidden, captures[request_id]["attention_mask"],
        )
        for group in request_groups:
            group_index = int(group["group"])
            position = int(group["position"])
            values = logits[0, position]
            values_np = values.detach().float().cpu().numpy()
            exact_top8 = ids_by_request[request_id][0, position].cpu().numpy()
            selected_set = set(exact_top8.tolist())
            outsiders = np.asarray(sorted(
                (expert for expert in range(len(values_np)) if expert not in selected_set),
                key=lambda expert: (-float(values_np[expert]), int(expert)),
            )[:8], np.int64)
            candidate_ids = np.concatenate((exact_top8[5:8], outsiders))
            sensitivities = np.zeros(
                (values_np.size, hidden.shape[-1]), np.float64,
            )
            started = time.perf_counter()
            for expert in candidate_ids.tolist():
                gradient = torch.autograd.grad(
                    values[int(expert)], hidden,
                    retain_graph=True,
                    create_graph=False,
                )[0]
                sensitivities[int(expert)] = (
                    gradient[0, position].detach().float().cpu().numpy()
                )
            gradient_seconds = time.perf_counter() - started
            for tau in temperatures:
                problems[float(tau)][group_index] = _exact_boundary_problem(
                    values_np,
                    sensitivities,
                    exact_top8,
                    float(tau),
                )
            reference_problem = problems[float(temperatures[0])][group_index]
            effects[group_index] = option_boundary_effects(
                geometries[group_index]["option_deltas"],
                group["execution_weights"],
                reference_problem,
            )
            boundary_margin = float(values_np[exact_top8[7]] - values_np[outsiders[0]])
            gradient_rows.append({
                "group": group_index,
                "request_id": request_id,
                "position": position,
                "rank8_rank9_margin": boundary_margin,
                "exact_vjp_candidates": int(len(candidate_ids)),
                "exact_vjp_seconds": gradient_seconds,
            })
            print(
                f"[D1 VJP] request={request_id} position={position} "
                f"margin={boundary_margin:.6g} "
                f"seconds={gradient_seconds:.3f}",
                flush=True,
            )
        del logits, hidden
        torch.cuda.empty_cache()
    if any(value is None for value in effects):
        raise RuntimeError("D1 option effects were incomplete")
    if any(any(value is None for value in rows) for rows in problems.values()):
        raise RuntimeError("D1 boundary problems were incomplete")
    return (
        {tau: list(rows) for tau, rows in problems.items()},
        list(effects),  # type: ignore[arg-type]
        gradient_rows,
    )


def _combined_damage(
    features: Sequence[np.ndarray],
    weights: np.ndarray,
    indices: np.ndarray,
) -> float:
    combined = sum(
        (
            float(weights[expert]) * features[expert][int(option)]
            for expert, option in enumerate(indices)
        ),
        np.zeros(features[0].shape[1], np.float64),
    )
    return float(combined @ combined)


def _allocation_policies(
    groups: Sequence[Mapping[str, Any]],
    geometries: Sequence[Mapping[str, Any]],
    problems: Mapping[float, Sequence[Any]],
    effects: Sequence[tuple[np.ndarray, ...]],
    rates: Sequence[int],
    etas: Sequence[float],
    config: Mapping[str, Any],
    local_policy: str,
) -> tuple[
    dict[tuple[str, int], list[Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    selections: dict[tuple[str, int], list[Any]] = {}
    group_rows = []
    expert_rows = []
    for rate in rates:
        selections[(local_policy, int(rate))] = [None] * len(groups)
        if local_policy == POLICY_LOCAL_REFINED:
            selections[(POLICY_REGENERATED, int(rate))] = [None] * len(groups)
        for tau in problems:
            for eta in etas:
                name = f"d1_strict_tau_{tau:g}_eta_{eta:g}"
                selections[(name, int(rate))] = [None] * len(groups)

    for group, geometry in zip(groups, geometries):
        group_index = int(group["group"])
        frontiers = geometry["frontiers"]
        features = geometry["features"]
        selector_weights = np.asarray(group["selector_weights"], np.float64)
        zero_indices = np.asarray([
            next(i for i, option in enumerate(frontier) if int(option.pages) == 0)
            for frontier in frontiers
        ], np.int64)
        q2_damage = _combined_damage(features, selector_weights, zero_indices)
        for rate in rates:
            budget = 8 * int(rate)
            additive = multiple_choice_allocate(
                frontiers, budget, selector_weights * selector_weights,
            )
            if local_policy == POLICY_LOCAL_REFINED:
                selections[(POLICY_REGENERATED, int(rate))][group_index] = additive
            local = exact_group_option_allocate(
                frontiers,
                features,
                selector_weights,
                budget,
                additive.option_indices,
                max_coordinate_sweeps=int(config["group_exact_coordinate_sweeps"]),
                max_pair_passes=int(config["group_exact_pair_passes"]),
            )
            selections[(local_policy, int(rate))][group_index] = local
            policy_values = [(local_policy, local, 0.0, 0.0)]
            if local_policy == POLICY_LOCAL_REFINED:
                policy_values.insert(
                    0, (POLICY_REGENERATED, additive, 0.0, 0.0),
                )
            for tau, problem_rows in problems.items():
                for eta in etas:
                    name = f"d1_strict_tau_{tau:g}_eta_{eta:g}"
                    limit = additive_local_damage_limit(
                        local.objective, q2_damage, float(eta),
                    )
                    d1 = d1_group_option_allocate(
                        frontiers,
                        effects[group_index],
                        problem_rows[group_index],
                        budget,
                        local.option_indices,
                        local_option_features=features,
                        local_router_weights=selector_weights,
                        local_damage_limit=limit,
                        max_coordinate_sweeps=int(config["group_exact_coordinate_sweeps"]),
                        max_pair_passes=int(config["group_exact_pair_passes"]),
                    )
                    selections[(name, int(rate))][group_index] = d1
                    policy_values.append((name, d1, float(tau), float(eta)))
            for policy, trace, tau, eta in policy_values:
                indices = np.asarray(trace.option_indices, np.int64)
                local_damage = _combined_damage(features, selector_weights, indices)
                group_rows.append({
                    "schema": SCHEMA,
                    "layer": int(group["layer"]),
                    "group": group_index,
                    "request_id": str(group["request_id"]),
                    "position": int(group["position"]),
                    "policy": policy,
                    "rate_pages_per_expert": int(rate),
                    "group_page_budget": budget,
                    "selected_group_pages": int(trace.pages),
                    "local_qenergy_damage": local_damage,
                    "local_q2_damage": q2_damage,
                    "local_optimal_damage": float(local.objective),
                    "local_guard_eta": eta,
                    "temperature": tau,
                    "predicted_route_loss": (
                        float(trace.route_loss) if hasattr(trace, "route_loss") else np.nan
                    ),
                    "predicted_crossings": (
                        int(trace.predicted_crossings)
                        if hasattr(trace, "predicted_crossings") else -1
                    ),
                })
                for router_rank, (expert, option_index) in enumerate(
                    zip(group["experts"], indices), start=1,
                ):
                    option = frontiers[router_rank - 1][int(option_index)]
                    expert_rows.append({
                        "schema": SCHEMA,
                        "layer": int(group["layer"]),
                        "group": group_index,
                        "request_id": str(group["request_id"]),
                        "position": int(group["position"]),
                        "policy": policy,
                        "rate_pages_per_expert": int(rate),
                        "router_rank": router_rank,
                        "expert_id": int(expert),
                        "selector_router_weight": float(
                            group["selector_weights"][router_rank - 1]
                        ),
                        "execution_router_weight": float(
                            group["execution_weights"][router_rank - 1]
                        ),
                        "selected_option_index": int(option_index),
                        "selected_pages": int(option.pages),
                        "selected_states": json.dumps(
                            np.asarray(option.states, np.int64).tolist(),
                            separators=(",", ":"),
                        ),
                    })
        print(
            f"[allocation] group {group_index + 1}/{len(groups)} complete",
            flush=True,
        )
    if any(any(value is None for value in rows) for rows in selections.values()):
        raise RuntimeError("allocation grid was incomplete")
    return selections, group_rows, expert_rows


def _delta_for_trace(
    geometry: Mapping[str, Any],
    execution_weights: np.ndarray,
    trace: Any,
) -> np.ndarray:
    indices = np.asarray(trace.option_indices, np.int64)
    return sum(
        (
            float(execution_weights[expert])
            * geometry["option_deltas"][expert][int(option)]
            for expert, option in enumerate(indices)
        ),
        np.zeros(2048, np.float64),
    ).astype(np.float32)


def _route_metrics(
    baseline_logits: np.ndarray,
    candidate_logits: np.ndarray,
    baseline_ids: np.ndarray,
    candidate_ids: np.ndarray,
) -> dict[str, float | int | bool]:
    baseline_ids = np.asarray(baseline_ids, np.int64).reshape(8)
    candidate_ids = np.asarray(candidate_ids, np.int64).reshape(8)
    baseline_set = set(baseline_ids.tolist())
    candidate_set = set(candidate_ids.tolist())
    shifted = baseline_logits.astype(np.float64) - float(np.max(baseline_logits))
    probability = np.exp(shifted)
    probability /= probability.sum()
    lost = baseline_set - candidate_set
    entered = candidate_set - baseline_set
    outsider_ids = list(set(range(len(baseline_logits))) - baseline_set)
    baseline_margin = float(
        baseline_logits[baseline_ids[7]] - np.max(baseline_logits[outsider_ids])
    )
    candidate_labeled_margin = float(
        np.min(candidate_logits[list(baseline_set)])
        - np.max(candidate_logits[list(set(range(len(candidate_logits))) - baseline_set)])
    )
    return {
        "exact_d1_crossed": bool(baseline_set != candidate_set),
        "exact_top8_agreement": len(baseline_set & candidate_set) / 8.0,
        "membership_pairs_changed": len(lost),
        "routing_mass_lost": float(sum(probability[index] for index in lost)),
        "baseline_rank8_rank9_margin": baseline_margin,
        "candidate_labeled_set_margin": candidate_labeled_margin,
        "entered_experts": json.dumps(sorted(entered), separators=(",", ":")),
        "left_experts": json.dumps(sorted(lost), separators=(",", ":")),
    }


def _exact_replays(
    model_slice: Any,
    captures: Mapping[str, Mapping[str, np.ndarray]],
    groups: Sequence[Mapping[str, Any]],
    geometries: Sequence[Mapping[str, Any]],
    selections: Mapping[tuple[str, int], Sequence[Any]],
    historical_path: Path,
    rates: Sequence[int],
    baseline_logits: Mapping[str, torch.Tensor],
    baseline_ids: Mapping[str, torch.Tensor],
    allocation_group_rows: Sequence[Mapping[str, Any]],
    historical_pr13_mode: str = "required",
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    allocation_lookup = {
        (
            str(row["policy"]),
            int(row["rate_pages_per_expert"]),
            int(row["group"]),
        ): row
        for row in allocation_group_rows
    }
    with np.load(historical_path, allow_pickle=False) as historical:
        historical_rates = np.asarray(
            historical["mean_budget_pages_per_expert"], np.int64,
        )
        historical_request = np.asarray(historical["request_id"]).astype(str)
        historical_position = np.asarray(historical["position"], np.int64)
        historical_delta = np.asarray(historical["delta"], np.float32)
        historical_damage = np.asarray(historical["group_damage"], np.float64)
    historical_rate_index = {
        int(rate): index for index, rate in enumerate(historical_rates.tolist())
    }

    if historical_pr13_mode not in ("required", "if_available", "omit"):
        raise ValueError("invalid historical PR13 replay mode")
    delta_by_policy: dict[str, np.ndarray] = {}
    generated_policies = sorted({policy for policy, _ in selections})
    metric_rows = []
    groups_by_request = {
        request_id: [group for group in groups if group["request_id"] == request_id]
        for request_id in captures
    }
    for rate in rates:
        historical_available = int(rate) in historical_rate_index
        if historical_pr13_mode == "required" and not historical_available:
            raise RuntimeError(f"historical delta omits rate {rate}")
        policies = list(generated_policies)
        if historical_pr13_mode != "omit" and historical_available:
            policies.append(POLICY_PR13)
        policies = sorted(policies)
        for policy in policies:
            complete = np.zeros((len(groups), 2048), np.float32)
            local_damage_by_group: dict[int, float] = {}
            if policy == POLICY_PR13:
                source_rate = historical_rate_index[int(rate)]
                for group in groups:
                    mask = (
                        (historical_request == str(group["request_id"]))
                        & (historical_position == int(group["position"]))
                    )
                    indices = np.flatnonzero(mask)
                    if len(indices) != 1:
                        raise RuntimeError("historical group identity is not unique")
                    source = int(indices[0])
                    complete[int(group["group"])] = historical_delta[source_rate, source]
                    local_damage_by_group[int(group["group"])] = float(
                        historical_damage[source_rate, source]
                    )
            else:
                traces = selections[(policy, int(rate))]
                for group, geometry, trace in zip(groups, geometries, traces):
                    complete[int(group["group"])] = _delta_for_trace(
                        geometry,
                        np.asarray(group["execution_weights"], np.float64),
                        trace,
                    )
                    local_damage_by_group[int(group["group"])] = float(
                        allocation_lookup[
                            (policy, int(rate), int(group["group"]))
                        ]["local_qenergy_damage"]
                    )
            delta_by_policy[f"{policy}__rate_{int(rate)}"] = complete
            for request_id, request_groups in groups_by_request.items():
                capture = captures[request_id]
                request_delta = np.zeros_like(capture["routed"], np.float32)
                for group in request_groups:
                    request_delta[int(group["position"])] = complete[
                        int(group["group"])
                    ]
                with torch.inference_mode():
                    candidate_hidden = model_slice.compose_current_output(
                        capture["residual"],
                        capture["x"],
                        capture["routed"],
                        request_delta,
                    )
                    candidate_logits_tensor, _, candidate_ids_tensor = (
                        model_slice.next_router_outputs(
                            candidate_hidden,
                            capture["attention_mask"],
                        )
                    )
                candidate_logits = (
                    candidate_logits_tensor[0].detach().float().cpu().numpy()
                )
                candidate_route_ids = candidate_ids_tensor[0].detach().cpu().numpy()
                reference_logits = (
                    baseline_logits[request_id][0].detach().float().cpu().numpy()
                )
                reference_route_ids = baseline_ids[request_id][0].detach().cpu().numpy()
                for group in request_groups:
                    position = int(group["position"])
                    metric_rows.append({
                        "schema": SCHEMA,
                        "layer": int(group["layer"]),
                        "next_layer": int(group["layer"]) + 1,
                        "group": int(group["group"]),
                        "request_id": request_id,
                        "position": position,
                        "policy": policy,
                        "rate_pages_per_expert": int(rate),
                        "local_qenergy_damage": local_damage_by_group[
                            int(group["group"])
                        ],
                        **_route_metrics(
                            reference_logits[position],
                            candidate_logits[position],
                            reference_route_ids[position],
                            candidate_route_ids[position],
                        ),
                    })
            print(
                f"[exact replay] rate={rate} policy={policy} complete",
                flush=True,
            )
    return metric_rows, delta_by_policy


def run_layer(args: argparse.Namespace, layer: int) -> dict[str, Any]:
    started = time.perf_counter()
    pr13_config = load_json(args.pr13_config)
    same_host_config = load_json(args.same_host_config)
    requests = list(map(str, same_host_config["validation_request_ids"]))
    captures = {
        request_id: _load_capture(
            _capture_path(args.capture_dir, request_id), int(layer),
        )
        for request_id in requests
    }
    groups = _groups(captures, int(layer), same_host_config)
    factor_path = args.fit_dir / f"average_rate_factor_layer_{int(layer)}.npz"
    factor_sidecar = args.fit_dir / f"average_rate_factor_layer_{int(layer)}.json"
    if load_json(factor_sidecar)["sha256"] != sha256(factor_path):
        raise RuntimeError("factor sidecar hash mismatch")
    arrays = dict(np.load(factor_path, allow_pickle=False))
    proxy = np.asarray(arrays["proxy"], np.float32)
    beta = float(np.asarray(arrays["beta"]).reshape(-1)[0])
    experts = _prepare_experts(
        args, int(layer), groups, arrays, proxy, beta, pr13_config,
    )
    geometries = _build_geometries(
        groups, experts, proxy, beta, pr13_config, args.workers, int(layer),
    )
    local_policy = POLICY_LOCAL
    if args.column_generated_frontiers:
        geometries = _refine_geometries(
            geometries, groups, proxy, beta, pr13_config,
            int(args.rates[0]), args.workers, int(layer),
        )
        local_policy = POLICY_LOCAL_REFINED
    del experts, arrays
    gc.collect()

    model_slice = load_d1_layer_slice(
        args.checkpoint, int(layer), device=args.device,
    )
    hidden, baseline_logits, baseline_ids, parity_rows = _paired_baselines(
        model_slice, captures,
    )
    problems, effects, gradient_rows = _problems_and_effects(
        model_slice,
        captures,
        groups,
        geometries,
        hidden,
        baseline_ids,
        list(map(float, args.temperature)),
    )
    selections, allocation_rows, expert_rows = _allocation_policies(
        groups,
        geometries,
        problems,
        effects,
        list(map(int, args.rates)),
        list(map(float, args.eta)),
        pr13_config,
        local_policy,
    )
    historical_path = (
        args.same_host_layers_dir
        / f"same_host_delta_layer_{int(layer):02d}.npz"
    )
    exact_rows, deltas = _exact_replays(
        model_slice,
        captures,
        groups,
        geometries,
        selections,
        historical_path,
        list(map(int, args.rates)),
        baseline_logits,
        baseline_ids,
        allocation_rows,
        args.historical_pr13_mode,
    )
    layer_dir = args.output_dir / f"layer_{int(layer):02d}"
    atomic_parquet(layer_dir / "d1_exact_route_metrics.parquet", exact_rows)
    atomic_parquet(layer_dir / "d1_allocation_groups.parquet", allocation_rows)
    atomic_parquet(layer_dir / "d1_allocation_experts.parquet", expert_rows)
    atomic_parquet(layer_dir / "d1_vjp_metrics.parquet", gradient_rows)
    atomic_parquet(layer_dir / "d1_paired_parity.parquet", parity_rows)
    atomic_npz(
        layer_dir / "d1_selected_deltas.npz",
        schema=np.asarray(SCHEMA),
        layer=np.asarray(int(layer), np.int64),
        **{name: value for name, value in deltas.items()},
    )
    exact_frame = pd.DataFrame(exact_rows)
    summary = (
        exact_frame.groupby(
            ["layer", "policy", "rate_pages_per_expert"], sort=True,
        )
        .agg(
            groups=("group", "size"),
            exact_d1_crossing_rate=("exact_d1_crossed", "mean"),
            mean_exact_top8_agreement=("exact_top8_agreement", "mean"),
            mean_membership_pairs_changed=("membership_pairs_changed", "mean"),
            mean_routing_mass_lost=("routing_mass_lost", "mean"),
            mean_local_qenergy_damage=("local_qenergy_damage", "mean"),
        )
        .reset_index()
    )
    atomic_parquet(layer_dir / "d1_policy_summary.parquet", summary)
    facts = {
        "schema": SCHEMA,
        "layer": int(layer),
        "next_layer": int(layer) + 1,
        "groups": len(groups),
        "active_experts": len({
            int(expert) for group in groups for expert in group["experts"]
        }),
        "rates": list(map(int, args.rates)),
        "eta": list(map(float, args.eta)),
        "temperature": list(map(float, args.temperature)),
        "frontier_mode": (
            "column_generated" if args.column_generated_frontiers else "coarse"
        ),
        "historical_pr13_mode": str(args.historical_pr13_mode),
        "historical_pr13_replayed": bool(
            str(args.historical_pr13_mode) != "omit"
            and all(int(rate) in (384, 576, 749, 768) for rate in args.rates)
        ),
        "policies": sorted(exact_frame["policy"].unique().tolist()),
        "paired_local_baseline_required": True,
        "cross_device_capture_drift_quantified_not_counted_as_allocator_switching": True,
        "parity_rows": parity_rows,
        "wall_seconds": time.perf_counter() - started,
        "files": {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sorted(layer_dir.iterdir())
            if path.is_file() and path.name != "d1_layer_facts.json"
        },
        "environment": {
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "input_hashes": {
            "checkpoint_index": sha256(
                args.checkpoint / "model.safetensors.index.json"
            ),
            "trees": sha256(args.trees),
            "pr13_config": sha256(args.pr13_config),
            "same_host_config": sha256(args.same_host_config),
            "factor": sha256(factor_path),
            "historical_delta": sha256(historical_path),
            **{
                f"capture_{request_id}": sha256(
                    _capture_path(args.capture_dir, request_id)
                )
                for request_id in requests
            },
        },
        "test_rows_admitted_or_used": False,
        "scientific_boundary": (
            "D1 same-token next-layer-router layer-slice pilot only; paired "
            "3090 Q4 baseline; no D2-D4 objective and no terminal-KL claim."
        ),
    }
    atomic_json(layer_dir / "d1_layer_facts.json", facts)
    print(summary.to_string(index=False), flush=True)
    del model_slice, geometries, hidden, baseline_logits, baseline_ids
    torch.cuda.empty_cache()
    gc.collect()
    return facts


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.workers > 32:
        raise ValueError("workers must lie in [1,32]")
    if any(rate < 1 or rate > 1536 for rate in args.rates):
        raise ValueError("rates must lie in the physical [1,1536] page range")
    if any(value < 0.0 for value in args.eta):
        raise ValueError("eta must be nonnegative")
    if any(value <= 0.0 for value in args.temperature):
        raise ValueError("temperature must be positive")
    if args.column_generated_frontiers and len(args.rates) != 1:
        raise ValueError(
            "column-generated mode requires exactly one rate per invocation"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    facts = [run_layer(args, int(layer)) for layer in args.layers]
    atomic_json(args.output_dir / "d1_pilot_run_facts.json", {
        "schema": SCHEMA,
        "completed": True,
        "layers": {str(row["layer"]): row for row in facts},
        "runner_sha256": sha256(Path(__file__).resolve()),
        "slice_core_sha256": sha256(
            EXPERIMENT / "src/oracle_study/d1_layer_slice.py"
        ),
        "objective_core_sha256": sha256(
            EXPERIMENT / "src/oracle_study/d1_route_objective.py"
        ),
    })


if __name__ == "__main__":
    main()
