#!/usr/bin/env python3
"""Evaluate two all-layer H1 response predictors through frozen PR #13 allocation."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch


LAYERS = tuple(range(40))
HORIZON = 1
BUDGET = 749
RATE = 0.9987386067708334
PRIMARY_FACTOR = "matrix_free_joint_rank8_int4_per_row_hadamard"
PRIMARY_POLICY = "pooled_router_square_column_generated"
BURST_CAP = 1536


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


class Recorder:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, event: str, **fields: Any) -> None:
        row = {"event": event, "unix": time.time(), **fields}
        with self.path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("pr13_h1_all_layer_allocator_helper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.HORIZON = HORIZON
    module.BUDGETS = (BUDGET,)
    module.RATES = {BUDGET: RATE}
    return module


def load_rows(root: Path, layer: int) -> dict[str, np.ndarray]:
    pieces: dict[str, list[np.ndarray]] = {}
    for split in ("train", "validation"):
        with np.load(root / f"{split}_layer_{layer:02d}.npz", allow_pickle=False) as loaded:
            for name in loaded.files:
                pieces.setdefault(name, []).append(np.asarray(loaded[name]))
    return {name: np.concatenate(values, axis=0) for name, values in pieces.items()}


def load_oracle(root: Path) -> dict[tuple[str, int, int, int], dict[str, Any]]:
    groups = pd.read_parquet(root / "average_rate_group_frontier.parquet")
    experts = pd.read_parquet(root / "average_rate_expert_allocation.parquet")
    common_group = (
        (groups["factor_config_id"] == PRIMARY_FACTOR)
        & (groups["allocation_policy"] == PRIMARY_POLICY)
        & (groups["burst_cap_pages_per_expert"] == BURST_CAP)
        & (groups["mean_budget_pages_per_expert"] == BUDGET)
        & (groups["position"] >= HORIZON)
        & groups["layer"].isin(LAYERS)
    )
    common_expert = (
        (experts["factor_config_id"] == PRIMARY_FACTOR)
        & (experts["allocation_policy"] == PRIMARY_POLICY)
        & (experts["burst_cap_pages_per_expert"] == BURST_CAP)
        & (experts["mean_budget_pages_per_expert"] == BUDGET)
        & (experts["position"] >= HORIZON)
        & experts["layer"].isin(LAYERS)
    )
    group_frame = groups[common_group]
    expert_frame = experts[common_expert]
    if len(group_frame) != 1160 or len(expert_frame) != 1160 * 8:
        raise RuntimeError(f"canonical all-layer H1 subset changed: {len(group_frame)}/{len(expert_frame)}")
    result = {}
    for _, group in group_frame.iterrows():
        key = (str(group.request_id), int(group.position), int(group.layer), BUDGET)
        local = expert_frame[
            (expert_frame.request_id == key[0])
            & (expert_frame.position == key[1])
            & (expert_frame.layer == key[2])
        ].sort_values("router_rank")
        if len(local) != 8:
            raise RuntimeError(f"canonical expert width changed: {key}")
        result[key] = {
            "damage": float(group.group_exact_qenergy_damage),
            "base_damage": float(group.group_base_qenergy_damage),
            "recovery": float(group.group_recovery),
            "states": [np.asarray(json.loads(value), np.int64) for value in local.selected_states],
            "pages": [int(value) for value in local.selected_pages],
            "experts": [int(value) for value in local.expert_id],
        }
    return result


def expand_compact(value: torch.Tensor, active: list[int]) -> np.ndarray:
    dense = np.zeros((value.shape[0], 256, 4, 512), np.float64)
    dense[:, np.asarray(active, np.int64)] = value.numpy().astype(np.float64)
    return dense


def allocate_layer(
    layer: int,
    experiment: Path,
    exact_root: Path,
    checkpoint: Path,
    checkpoint_index: Mapping[str, str],
    trees: Mapping[str, Any],
    factor_dir: Path,
    config: Mapping[str, Any],
    oracle: Mapping[tuple[str, int, int, int], Any],
    variants: tuple[str, str],
    workers: int,
    helper: Any,
    runner: Any,
    base: Any,
    prior: Any,
    output: Path,
    recorder: Recorder,
) -> pd.DataFrame:
    from oracle_study.average_rate_allocator import (
        allocation_aware_rate_frontiers,
        lagrangian_rate_frontier,
        qmetric_features,
    )
    from oracle_study.neuron_selector import unit_score_metadata
    from oracle_study.split_interaction_field import (
        SPLIT_STATE_DOWN_HIGH,
        SPLIT_STATE_GATE_HIGH,
        SPLIT_STATE_UP_HIGH,
        SplitProjectionResponses,
        build_split_interaction_field,
        split_state_output,
    )

    dataset = torch.load(experiment / f"dataset_layer_{layer}.pt", map_location="cpu", weights_only=False)
    prediction = torch.load(
        experiment / f"prediction_layer_{layer}_{variants[0]}.pt",
        map_location="cpu",
        weights_only=True,
    ).float()
    predictions = {variants[0]: prediction, variants[1]: prediction}
    active = [int(value) for value in dataset["active_experts"]]
    active_index = {expert: index for index, expert in enumerate(active)}
    exact_hidden = expand_compact(dataset["exact_hidden_compact"], active)
    variant_hidden = {variant: expand_compact(prediction, active) for variant, prediction in predictions.items()}
    exact_data = load_rows(exact_root, layer)
    local = runner.prior.layer_view(exact_data, layer)
    proxy, proxy_facts = base.proxy_gradients(
        local["xplus"], [local[f"h{h}_router_logits"] for h in range(1, 5)], local["split"],
    )
    beta = float(proxy_facts["beta"])
    arrays = dict(np.load(factor_dir / f"average_rate_factor_layer_{layer}.npz", allow_pickle=False))
    factors, abc = {}, {}
    down2, down4 = [], []
    for expert in active:
        factors[expert] = runner._load_layer_factors(arrays, config, expert)[PRIMARY_FACTOR]
        decoded = prior.decode_expert(checkpoint, checkpoint_index, trees, layer, expert)
        down2.append(decoded["down"][0].copy())
        down4.append(decoded["down"][2].copy())
        del decoded
    down2_np = np.stack(down2).astype(np.float64)
    down4_np = np.stack(down4).astype(np.float64)
    for expert in active:
        index = active_index[expert]
        abc[expert] = unit_score_metadata(
            down2_np[index], down4_np[index], proxy=proxy, beta=beta,
        )
    validation_indices = [index for index, split in enumerate(dataset["split"]) if split == "validation"]
    if len(validation_indices) != 29:
        raise RuntimeError(f"H1 validation count changed for layer {layer}")
    helper._ALLOC_CONTEXT = {
        "dataset": dataset,
        "validation_indices": validation_indices,
        "exact_hidden": exact_hidden,
        "learned_hidden": variant_hidden[variants[0]],
        "parent_hidden": variant_hidden[variants[1]],
        "down2": down2_np,
        "down4": down4_np,
        "active_index": active_index,
        "factors": factors,
        "abc": abc,
        "proxy": np.asarray(proxy, np.float64),
        "beta": beta,
        "config": config,
        "oracle": oracle,
        "SplitProjectionResponses": SplitProjectionResponses,
        "build_split_interaction_field": build_split_interaction_field,
        "split_state_output": split_state_output,
        "qmetric_features": qmetric_features,
        "lagrangian_rate_frontier": lagrangian_rate_frontier,
        "allocation_aware_rate_frontiers": allocation_aware_rate_frontiers,
        "state_gate": SPLIT_STATE_GATE_HIGH,
        "state_up": SPLIT_STATE_UP_HIGH,
        "state_down": SPLIT_STATE_DOWN_HIGH,
    }
    started = time.time()
    context = mp.get_context("fork")
    with context.Pool(min(workers, len(validation_indices))) as pool:
        groups = pool.map(helper.evaluate_allocation_group, range(len(validation_indices)))
    rows = [row for group in groups for row in group]
    helper._ALLOC_CONTEXT = None
    frame = pd.DataFrame(rows)
    frame["method"] = frame["method"].replace({
        "learned_response": variants[0],
        "parent_activation_q4_oracle": variants[1],
    })
    compact = frame.drop(columns=["selected_states", "oracle_states"])
    compact.to_parquet(output / f"allocation_layer_{layer}.parquet", index=False)
    recorder("allocation_layer_completed", layer=layer, variants=list(variants), rows=len(frame), seconds=time.time() - started)
    del dataset, predictions, exact_hidden, variant_hidden, exact_data, arrays, factors, abc
    del down2, down4, down2_np, down4_np, frame
    gc.collect()
    return compact


def summarize(frame: pd.DataFrame) -> list[dict[str, Any]]:
    result = []
    for method, local in frame.groupby("method", sort=True):
        recovered = local["base_qenergy_damage"] - local["exact_qenergy_damage"]
        oracle_recovered = local["base_qenergy_damage"] - local["oracle_exact_qenergy_damage"]
        oracle_median = float(local["oracle_qenergy_recovery"].median())
        median = float(local["qenergy_recovery"].median())
        p10 = float(local["qenergy_recovery"].quantile(0.10))
        result.append({
            "method": method,
            "groups": len(local),
            "overall_bpw": RATE,
            "oracle_utility_retained": float(recovered.sum() / oracle_recovered.sum()),
            "normalized_allocation_regret": float(
                (local["exact_qenergy_damage"] - local["oracle_exact_qenergy_damage"]).sum()
                / oracle_recovered.sum()
            ),
            "qenergy_recovery_p10": p10,
            "qenergy_recovery_median": median,
            "qenergy_recovery_p90": float(local["qenergy_recovery"].quantile(0.90)),
            "oracle_qenergy_recovery_p10": float(local["oracle_qenergy_recovery"].quantile(0.10)),
            "oracle_qenergy_recovery_median": oracle_median,
            "median_retention_ratio": median / oracle_median,
            "meets_90_percent_p10": bool(p10 >= 0.90),
            "page_jaccard_mean": float(local["page_jaccard"].mean()),
            "state_overlap_mean": float(local["state_overlap"].mean()),
            "maximum_oracle_rescore_base_relative_error": float(
                local["canonical_oracle_damage_base_relative_error"].max()
            ),
        })
    return result


def per_layer_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    result = []
    for (method, layer), local in frame.groupby(["method", "layer"], sort=True):
        result.append({
            "method": method,
            "layer": int(layer),
            "groups": len(local),
            "qenergy_recovery_p10": float(local["qenergy_recovery"].quantile(0.10)),
            "qenergy_recovery_median": float(local["qenergy_recovery"].median()),
            "qenergy_recovery_mean": float(local["qenergy_recovery"].mean()),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=Path("/workspace/pr13_h1_all_layers_predictor_20260822_v1"))
    parser.add_argument("--variants", default="q2_asym_b512,b1_asym_b512_tail16")
    parser.add_argument("--layers", default="0-39")
    parser.add_argument("--workers", type=int, default=29)
    parser.add_argument("--exact-root", type=Path, default=Path("/workspace/all_layer_split_interaction_study/capture"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/workspace/pr13_average_rate_all_layers_20260822_v1/inputs/checkpoint"))
    parser.add_argument("--trees", type=Path, default=Path("/workspace/codebook_granularity_study/locked/selected_trees.json"))
    parser.add_argument("--oracle-root", type=Path, default=Path("/workspace/set_utility_study/experiments/adaptive_expert_precision_oracle"))
    parser.add_argument("--base-config", type=Path, default=Path("/workspace/set_utility_study/experiments/adaptive_expert_precision_oracle/configs/qwen36_mxfp4_average_rate_allocation.json"))
    parser.add_argument("--factor-dir", type=Path, default=Path("/workspace/pr13_average_rate_all_layers_20260822_v1/results/fit"))
    parser.add_argument("--canonical", type=Path, default=Path("/workspace/pr13_average_rate_all_layers_20260822_v1/results/validation"))
    parser.add_argument("--helper", type=Path, default=Path("/workspace/pr13_h4_response_field_pilot_run.py"))
    parser.add_argument("--output", type=Path, default=Path("/workspace/pr13_h1_all_layers_predictor_20260822_v1/allocation_q2asym_tail16"))
    args = parser.parse_args()
    variants = tuple(args.variants.split(","))
    if len(variants) != 2 or variants[0] == variants[1]:
        raise ValueError("--variants must contain two distinct comma-separated names")
    layers = LAYERS if args.layers == "0-39" else tuple(int(value) for value in args.layers.split(","))
    args.output.mkdir(parents=True, exist_ok=True)
    recorder = Recorder(args.output / "run.jsonl")
    sys.path[:0] = [str(args.oracle_root / "src"), str(args.oracle_root / "scripts")]
    import run_average_rate_allocation as runner
    import run_set_utility_distillation as prior
    import run_sparse_streaming_study as base
    from run_mxfp4_selective_pages import tree_from_record
    helper = load_module(args.helper)
    config = json.loads(args.base_config.read_text())
    oracle = load_oracle(args.canonical)
    checkpoint_index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    tree_records = json.loads(args.trees.read_text())
    trees = {name: tree_from_record(tree_records[name]) for name in ("gate", "up", "down")}
    completed = {
        int(path.stem.split("_")[-1]): path
        for path in args.output.glob("allocation_layer_*.parquet")
    }
    recorder("allocation_started", variants=list(variants), layers=list(layers), completed_layers=sorted(completed), oracle_groups=len(oracle))
    for layer in layers:
        if layer in completed:
            continue
        allocate_layer(
            layer, args.experiment, args.exact_root, args.checkpoint,
            checkpoint_index, trees, args.factor_dir, config, oracle, variants,
            args.workers, helper, runner, base, prior, args.output, recorder,
        )
        frames = [pd.read_parquet(path) for path in sorted(args.output.glob("allocation_layer_*.parquet"))]
        frame = pd.concat(frames, ignore_index=True)
        atomic_json(args.output / "allocation_summary.json", {
            "completed_layers": sorted(set(frame.layer.astype(int))),
            "summary": summarize(frame),
            "per_layer": per_layer_summary(frame),
        })
    frames = [pd.read_parquet(path) for path in sorted(args.output.glob("allocation_layer_*.parquet"))]
    frame = pd.concat(frames, ignore_index=True)
    frame.to_parquet(args.output / "allocation_results.parquet", index=False)
    payload = {
        "completed_layers": sorted(set(frame.layer.astype(int))),
        "summary": summarize(frame),
        "per_layer": per_layer_summary(frame),
    }
    atomic_json(args.output / "allocation_summary.json", payload)
    recorder("allocation_completed", **payload)


if __name__ == "__main__":
    main()
