#!/usr/bin/env python3
"""Compact cross-layer average-rate ceiling over frozen PR #13 factors.

This is deliberately an exploratory probe.  It does not mutate the frozen
PR #13 bundle: it rebuilds the existing compact rank-8 per-expert frontiers,
injects their published selected states as dominance anchors, and asks whether
the same total page budget is better spent across several token/layer groups.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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

from oracle_study.average_rate_allocator import (  # noqa: E402
    RateOption,
    exact_group_option_allocate,
    lagrangian_rate_frontier,
    multiple_choice_allocate,
)
from oracle_study.split_interaction_field import (  # noqa: E402
    build_split_interaction_field,
    split_projection_responses,
)
from run_mxfp4_selective_pages import tree_from_record  # noqa: E402
import run_average_rate_allocation as pr13  # noqa: E402
import run_sparse_streaming_study as sparse  # noqa: E402


PRIMARY = "matrix_free_joint_rank8_int4_per_row_hadamard"
PR13_GROUP_GEOMETRY_BYTES = 8 * (6_148 + 3_084)
PR13_GROUP_COMPUTE_MACS = 79_712_256
GROUP_WIDTH = 8
UNITS = 512
_WORK_CONTEXT: dict[str, Any] | None = None


def _parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result):
        raise ValueError("integer list must be nonempty and unique")
    return result


def _load_shard(path: Path, expected_split: str, layer: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        data = {name: np.asarray(loaded[name]) for name in loaded.files}
    if set(np.asarray(data["split"]).astype(str)) != {expected_split}:
        raise RuntimeError(f"capture split changed: {path}")
    if set(map(int, np.unique(data["layer"]))) != {int(layer)}:
        raise RuntimeError(f"capture layer changed: {path}")
    return data


def _validation_groups(data: Mapping[str, np.ndarray], count: int) -> tuple[dict[str, Any], ...]:
    order = sorted(
        range(len(data["split"])),
        key=lambda index: (
            str(data["request_id"][index]), int(data["position"][index]), index,
        ),
    )[: int(count)]
    result = []
    for record in order:
        experts = np.asarray(data["expert_ids"][record], np.int64)
        weights = np.asarray(data["router_weights"][record], np.float64)
        if experts.shape != (8,) or len(set(experts.tolist())) != 8:
            raise RuntimeError("validation group is not eight unique experts")
        if weights.shape != (8,) or not np.isclose(weights.sum(), 1.0, atol=5e-7):
            raise RuntimeError("router weights changed")
        result.append({
            "record": int(record),
            "request_id": str(data["request_id"][record]),
            "position": int(data["position"][record]),
            "experts": experts.tolist(),
            "weights": weights.tolist(),
        })
    return tuple(result)


def _option_key(option: RateOption) -> tuple[int, float, tuple[int, ...]]:
    return (
        int(option.pages), float(option.damage),
        tuple(np.asarray(option.states, np.int64).tolist()),
    )


def _pareto(options: Sequence[RateOption]) -> tuple[RateOption, ...]:
    by_state: dict[tuple[int, ...], RateOption] = {}
    for option in options:
        state = tuple(np.asarray(option.states, np.int64).tolist())
        old = by_state.get(state)
        if old is None or _option_key(option) < _option_key(old):
            by_state[state] = option
    by_pages: dict[int, RateOption] = {}
    for option in by_state.values():
        pages = int(option.pages)
        old = by_pages.get(pages)
        if old is None or float(option.damage) < float(old.damage) - 1e-12:
            by_pages[pages] = option
    kept: list[RateOption] = []
    best = np.inf
    for pages in sorted(by_pages):
        option = by_pages[pages]
        if float(option.damage) < best - 1e-12:
            kept.append(option)
            best = float(option.damage)
    if not kept or int(kept[0].pages) != 0:
        raise RuntimeError("frontier lost zero-page endpoint")
    return tuple(kept)


def _inherited_states(
    frame: pd.DataFrame, group: Mapping[str, Any], expert: int,
) -> tuple[tuple[str, int, np.ndarray], ...]:
    mask = (
        frame["request_id"].astype(str).eq(str(group["request_id"]))
        & frame["position"].astype(int).eq(int(group["position"]))
        & frame["expert_id"].astype(int).eq(int(expert))
        & frame["factor_config_id"].eq(PRIMARY)
        & frame["burst_cap_pages_per_expert"].astype(int).eq(1536)
        & frame["allocation_policy"].isin([
            "pooled_router_square_column_generated",
            "pooled_exact_combined_moe_column_generated_local",
        ])
    )
    rows = frame.loc[mask]
    result = []
    for row in rows.itertuples(index=False):
        states = np.asarray(json.loads(row.selected_states), np.int64)
        if states.shape != (UNITS,):
            raise RuntimeError("inherited state vector shape changed")
        result.append((
            f"inherited_{row.allocation_policy}_{int(row.mean_budget_pages_per_expert)}",
            int(row.selected_pages), states,
        ))
    return tuple(result)


def _chosen_record(
    frontiers: Sequence[Sequence[RateOption]], features: Sequence[np.ndarray],
    allocation: Any, weights: np.ndarray, predicted_base: float,
    exact_base: float, source: str,
) -> dict[str, Any]:
    indices = np.asarray(allocation.option_indices, np.int64)
    selected = [features[e][int(index)] for e, index in enumerate(indices)]
    combined = sum(
        (weights[e] * selected[e] for e in range(GROUP_WIDTH)),
        np.zeros(selected[0].shape, np.float64),
    )
    exact_damage = float(combined @ combined)
    predicted_damage = float(sum(
        weights[e] * weights[e]
        * float(frontiers[e][int(index)].damage)
        for e, index in enumerate(indices)
    ))
    pages = int(sum(
        int(frontiers[e][int(index)].pages) for e, index in enumerate(indices)
    ))
    return {
        "source": source,
        "pages": pages,
        "exact_damage": exact_damage,
        "exact_normalized_damage": exact_damage / exact_base,
        "recovery": 1.0 - exact_damage / exact_base,
        "predicted_damage": predicted_damage,
        "predicted_normalized_damage": predicted_damage / predicted_base,
    }


def _pareto_records(rows: Sequence[Mapping[str, Any]], objective: str) -> list[dict[str, Any]]:
    by_pages: dict[int, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        pages = int(row["pages"])
        old = by_pages.get(pages)
        if old is None or float(row[objective]) < float(old[objective]) - 1e-15:
            by_pages[pages] = row
    result = []
    best = np.inf
    for pages in sorted(by_pages):
        row = by_pages[pages]
        if float(row[objective]) < best - 1e-15:
            result.append(row)
            best = float(row[objective])
    return result


def _evaluate_group(task: int) -> tuple[int, dict[str, Any]]:
    if _WORK_CONTEXT is None:
        raise RuntimeError("worker context is unset")
    group = _WORK_CONTEXT["groups"][int(task)]
    config = _WORK_CONTEXT["config"]
    activation = np.asarray(
        _WORK_CONTEXT["validation"]["x"][int(group["record"])], np.float32,
    )
    weights = np.asarray(group["weights"], np.float64)
    frontiers, features = [], []
    responses_by_expert = []
    for expert in map(int, group["experts"]):
        cell = _WORK_CONTEXT["experts"][expert]
        responses = split_projection_responses(cell["q2"], cell["q4"], activation)
        field = build_split_interaction_field(cell["factor"], responses.hidden, cell["abc"])
        trace = lagrangian_rate_frontier(
            field,
            maximum_pages=1536,
            target_pages=_WORK_CONTEXT["anchors"],
            price_ratios=config["frontier_price_ratios"],
            coordinate_sweeps=int(_WORK_CONTEXT["coordinate_sweeps"]),
            local_shortlist=int(config["local_shortlist"]),
            local_swap_units=int(config["local_swap_units"]),
            local_max_passes=int(_WORK_CONTEXT["local_passes"]),
            price_local_max_passes=0,
        )
        options = list(trace.options)
        for source, pages, states in _inherited_states(
            _WORK_CONTEXT["inherited"], group, expert,
        ):
            options.append(RateOption(
                pages=pages, damage=float(field.damage(states)),
                states=states.copy(), source=source, page_price=0.0,
                coordinate_sweeps=0, local_passes=0,
            ))
        options = _pareto(options)
        option_features, _ = pr13._exact_option_data(
            responses, options, _WORK_CONTEXT["proxy"], _WORK_CONTEXT["beta"],
        )
        frontiers.append(options)
        features.append(option_features)
        responses_by_expert.append(responses)

    zero_features = [value[0] for value in features]
    exact_zero = sum(
        (weights[e] * zero_features[e] for e in range(GROUP_WIDTH)),
        np.zeros(zero_features[0].shape, np.float64),
    )
    exact_base = float(exact_zero @ exact_zero)
    predicted_base = float(sum(
        weights[e] * weights[e] * float(frontiers[e][0].damage)
        for e in range(GROUP_WIDTH)
    ))
    if exact_base <= 0.0 or predicted_base <= 0.0:
        raise RuntimeError("group base damage is not positive")

    compressed_rows, exact_rows = [], []
    for mean_pages in _WORK_CONTEXT["mean_grid"]:
        budget = GROUP_WIDTH * int(mean_pages)
        compressed = multiple_choice_allocate(frontiers, budget, weights * weights)
        compressed_rows.append(_chosen_record(
            frontiers, features, compressed, weights, predicted_base, exact_base,
            f"compressed_mean_{mean_pages}",
        ))
        exact = exact_group_option_allocate(
            frontiers, features, weights, budget, compressed.option_indices,
            max_coordinate_sweeps=int(_WORK_CONTEXT["group_sweeps"]),
            max_pair_passes=int(_WORK_CONTEXT["group_pair_passes"]),
        )
        exact_rows.append(_chosen_record(
            frontiers, features, exact, weights, predicted_base, exact_base,
            f"exact_mean_{mean_pages}",
        ))
    return int(task), {
        "request_id": str(group["request_id"]),
        "position": int(group["position"]),
        "layer": int(_WORK_CONTEXT["layer"]),
        "compressed": _pareto_records(compressed_rows, "predicted_normalized_damage"),
        "exact": _pareto_records(exact_rows, "exact_normalized_damage"),
    }


def _cross_layer_allocate(
    groups: Sequence[Mapping[str, Any]], policy: str, budget: int,
) -> tuple[list[int], int]:
    objective = (
        "predicted_normalized_damage" if policy == "compressed"
        else "exact_normalized_damage"
    )
    dynamic = np.full(int(budget) + 1, np.inf, np.float64)
    dynamic[0] = 0.0
    choices = np.full((len(groups), int(budget) + 1), -1, np.int16)
    previous = np.full((len(groups), int(budget) + 1), -1, np.int32)
    for group_index, group in enumerate(groups):
        updated = np.full_like(dynamic, np.inf)
        finite = np.flatnonzero(np.isfinite(dynamic))
        for option_index, option in enumerate(group[policy]):
            pages = int(option["pages"])
            source = finite[finite + pages <= int(budget)]
            target = source + pages
            candidate = dynamic[source] + float(option[objective])
            better = candidate < updated[target] - 1e-15
            selected = np.flatnonzero(better)
            if selected.size:
                positions = target[selected]
                updated[positions] = candidate[selected]
                choices[group_index, positions] = int(option_index)
                previous[group_index, positions] = source[selected]
        dynamic = updated
    feasible = np.flatnonzero(np.isfinite(dynamic))
    used = min(feasible.tolist(), key=lambda pages: (float(dynamic[pages]), pages))
    selected: list[int] = [0] * len(groups)
    cursor = int(used)
    for group_index in range(len(groups) - 1, -1, -1):
        option = int(choices[group_index, cursor])
        if option < 0:
            raise RuntimeError("cross-layer allocation backtrack failed")
        selected[group_index] = option
        cursor = int(previous[group_index, cursor])
    return selected, int(used)


def _summarize(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, np.float64)
    return {
        "p10": float(np.quantile(array, 0.1)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p90": float(np.quantile(array, 0.9)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trees", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="0,4,15,18,20,23,35,39")
    parser.add_argument("--groups-per-layer", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--mean-grid", default="0,96,192,288,384,480,576,672,768,960,1152,1344,1536")
    parser.add_argument("--anchors", default="0,96,192,288,384,480,576,768,1152,1536")
    parser.add_argument("--coordinate-sweeps", type=int, default=4)
    parser.add_argument("--local-passes", type=int, default=8)
    parser.add_argument("--group-sweeps", type=int, default=4)
    parser.add_argument("--group-pair-passes", type=int, default=4)
    args = parser.parse_args()
    layers = _parse_ints(args.layers)
    mean_grid = _parse_ints(args.mean_grid)
    anchors = _parse_ints(args.anchors)
    config = json.loads(args.config.read_text())
    if str(config["primary_factor_id"]) != PRIMARY:
        raise RuntimeError("primary PR #13 factor changed")
    tree_records = json.loads(args.trees.read_text())
    trees = {name: tree_from_record(tree_records[name]) for name in pr13.PROJECTIONS}
    index = json.loads(
        (args.checkpoint / "model.safetensors.index.json").read_text()
    )["weight_map"]
    all_groups: list[dict[str, Any]] = []
    started = time.perf_counter()
    for layer in layers:
        train = _load_shard(
            args.capture_dir / f"train_layer_{layer:02d}.npz", "train", layer,
        )
        validation = _load_shard(
            args.capture_dir / f"validation_layer_{layer:02d}.npz",
            "validation", layer,
        )
        groups = _validation_groups(validation, args.groups_per_layer)
        proxy, proxy_facts = sparse.proxy_gradients(
            train["xplus"], [train[f"h{h}_router_logits"] for h in range(1, 5)],
            train["split"],
        )
        beta = float(proxy_facts["beta"])
        with np.load(
            args.fit_dir / pr13.FIT_LAYER.format(layer=layer), allow_pickle=False,
        ) as loaded:
            factor_arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
        active = sorted({int(expert) for group in groups for expert in group["experts"]})
        experts = {}
        for expert in active:
            decoded = pr13.prior.decode_expert(
                args.checkpoint, index, trees, int(layer), int(expert),
            )
            q2 = tuple(decoded[name][0] for name in pr13.PROJECTIONS)
            q4 = tuple(decoded[name][2] for name in pr13.PROJECTIONS)
            experts[expert] = {
                "q2": q2,
                "q4": q4,
                "abc": pr13.unit_score_metadata(q2[2], q4[2], proxy=proxy, beta=beta),
                "factor": pr13._load_layer_factors(
                    factor_arrays, config, int(expert),
                )[PRIMARY],
            }
        inherited = pd.read_parquet(
            args.validation_dir / "layers"
            / f"average_rate_expert_allocation_layer_{layer}.parquet",
        )
        global _WORK_CONTEXT
        _WORK_CONTEXT = {
            "layer": int(layer), "groups": groups, "validation": validation,
            "experts": experts, "proxy": proxy, "beta": beta,
            "config": config, "inherited": inherited,
            "mean_grid": mean_grid, "anchors": anchors,
            "coordinate_sweeps": int(args.coordinate_sweeps),
            "local_passes": int(args.local_passes),
            "group_sweeps": int(args.group_sweeps),
            "group_pair_passes": int(args.group_pair_passes),
        }
        pending: dict[int, dict[str, Any]] = {}
        context = mp.get_context("fork")
        workers = min(int(args.workers), len(groups))
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            futures = {executor.submit(_evaluate_group, i): i for i in range(len(groups))}
            for future in as_completed(futures):
                index_value, result = future.result()
                pending[index_value] = result
        all_groups.extend(pending[index_value] for index_value in range(len(groups)))
        _WORK_CONTEXT = None
        print(json.dumps({
            "layer": int(layer), "groups_complete": len(groups),
            "elapsed_seconds": time.perf_counter() - started,
        }), flush=True)

    by_token: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for group in all_groups:
        by_token.setdefault(
            (str(group["request_id"]), int(group["position"])), [],
        ).append(group)
    if any(len(groups) != len(layers) for groups in by_token.values()):
        raise RuntimeError("selected token identities are not aligned across layers")

    summaries: dict[str, Any] = {}
    budget_per_group = GROUP_WIDTH * 384
    for policy in ("compressed", "exact"):
        pooled_recovery, uniform_recovery, selected_pages = [], [], []
        for token_groups in by_token.values():
            token_groups.sort(key=lambda row: int(row["layer"]))
            selected, used = _cross_layer_allocate(
                token_groups, policy, len(token_groups) * budget_per_group,
            )
            selected_pages.append(used)
            for group, option_index in zip(token_groups, selected):
                pooled_recovery.append(float(group[policy][option_index]["recovery"]))
                candidates = [
                    row for row in group[policy]
                    if int(row["pages"]) <= budget_per_group
                ]
                uniform_recovery.append(max(float(row["recovery"]) for row in candidates))
        summaries[policy] = {
            "uniform_0p5": _summarize(uniform_recovery),
            "cross_layer_average_0p5": _summarize(pooled_recovery),
            "paired_recovery_delta": _summarize(
                np.asarray(pooled_recovery) - np.asarray(uniform_recovery)
            ),
            "actual_pages_total": int(sum(selected_pages)),
            "allowed_pages_total": int(
                len(by_token) * len(layers) * budget_per_group
            ),
        }
    payload = {
        "schema": "half_bpw_cross_layer_compact_probe_v1",
        "scope": {
            "layers": list(layers),
            "groups_per_layer": int(args.groups_per_layer),
            "tokens": len(by_token),
            "groups": len(all_groups),
            "mean_correction_pages_per_expert": 384,
            "mean_correction_bpw": 0.5,
        },
        "cost_contract": {
            "group_geometry_bytes": PR13_GROUP_GEOMETRY_BYTES,
            "geometry_ratio_vs_pr13": 1.0,
            "base_pr13_group_compute_macs": PR13_GROUP_COMPUTE_MACS,
            "cross_layer_dp_is_small_control_overhead": True,
        },
        "summaries": summaries,
        "groups": all_groups,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), **summaries}, indent=2), flush=True)


if __name__ == "__main__":
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    main()
