#!/usr/bin/env python3
"""Export resident Q4 plus a fixed per-token sparse stream budget on predicted routes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


HIDDEN, UNITS, BLOCK, BLOCKS = 2048, 512, 32, 64
DENOMINATOR = 2 * HIDDEN * UNITS


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def aligned_capture(root: Path, layer: int, dataset: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    captured: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray]] = {}
    for split in ("train", "validation"):
        with np.load(root / f"{split}_layer_{layer:02d}.npz", allow_pickle=False) as data:
            for index in range(len(data["position"])):
                key = (str(data["request_id"][index]), int(data["position"][index]), str(data["prefix_hash"][index]))
                if key in captured:
                    raise RuntimeError(f"duplicate identity {key}")
                captured[key] = (
                    np.asarray(data["x"][index], np.float32),
                    np.asarray(data["router_logits"][index], np.float32),
                )
    values = [
        captured[(str(request), int(position), str(prefix))]
        for request, position, prefix in zip(dataset["request_id"], dataset["target_position"], dataset["prefix_hash"])
    ]
    return torch.from_numpy(np.stack([value[0] for value in values])), torch.from_numpy(np.stack([value[1] for value in values]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--resident-count", type=int, default=96)
    parser.add_argument("--block-budget", type=int, default=136)
    parser.add_argument("--parent-capture", type=Path, required=True)
    parser.add_argument("--damage-scores", type=Path, default=Path("/workspace/hybrid_q4_damage_frontier_v2/damage_scores.pt"))
    parser.add_argument("--source-experiment", type=Path, default=Path("/workspace/pr13_h1_all_layers_predictor_20260822_v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--low-helper", type=Path, default=Path("/workspace/pr13_low_bpw_pilot.py"))
    parser.add_argument("--sparse-helper", type=Path, default=Path("/workspace/run_sparse_refinement_pilot.py"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/workspace/codebook_granularity_study/inputs/checkpoint"))
    parser.add_argument("--trees", type=Path, default=Path("/workspace/codebook_granularity_study/locked/selected_trees.json"))
    parser.add_argument("--oracle-root", type=Path, default=Path("/workspace/set_utility_study/experiments/adaptive_expert_precision_oracle"))
    args = parser.parse_args()
    if args.block_budget > 17 * 8:
        raise ValueError("block budget exceeds the frozen 0.531315 bpw payload")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    low = load_module("adaptive_stream_low", args.low_helper)
    sparse = load_module("adaptive_stream_sparse", args.sparse_helper)
    sys.path[:0] = [str(args.oracle_root / "src"), str(args.oracle_root / "scripts")]
    import run_set_utility_distillation as prior
    from run_mxfp4_selective_pages import tree_from_record

    dataset_path = args.source_experiment / f"dataset_layer_{args.layer}.pt"
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    active = [int(value) for value in dataset["active_experts"]]
    active_index = {expert: index for index, expert in enumerate(active)}
    parent_x_cpu, parent_logits = aligned_capture(args.parent_capture, args.layer, dataset)
    parent_x = parent_x_cpu.to(device)
    predicted_routes = torch.topk(parent_logits, 8, dim=-1).indices
    damage = torch.load(args.damage_scores, map_location="cpu", weights_only=True)
    resident_global = torch.topk(damage[args.layer], args.resident_count).indices.long()
    resident_set = set(resident_global.tolist())
    resident_local = torch.tensor([expert in resident_set for expert in active], dtype=torch.bool, device=device)

    checkpoint_index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    records = json.loads(args.trees.read_text())
    trees = {name: tree_from_record(records[name]) for name in ("gate", "up", "down")}
    weights = {name: np.empty((len(active), UNITS, HIDDEN), np.float32) for name in ("gate2", "gate4", "up2", "up4")}
    for local_index, expert in enumerate(active):
        decoded = prior.decode_expert(args.checkpoint, checkpoint_index, trees, args.layer, expert)
        for name in ("gate", "up"):
            weights[name + "2"][local_index] = decoded[name][0]
            weights[name + "4"][local_index] = decoded[name][2]
        if local_index % 16 == 15 or local_index + 1 == len(active):
            print(json.dumps({"event": "decode", "completed": local_index + 1, "total": len(active)}), flush=True)
        del decoded
    delta_gate = weights["gate4"] - weights["gate2"]
    delta_up = weights["up4"] - weights["up2"]
    gate2 = low.project(weights["gate2"], parent_x, device)
    up2 = low.project(weights["up2"], parent_x, device)
    score_gate, score_up = sparse.score_blocks(delta_gate, delta_up, parent_x, "block_norm", device)
    score = score_gate + score_up
    selected = torch.zeros_like(score)
    actual_blocks = []
    for row in range(len(parent_x)):
        streamed = [int(expert) for expert in predicted_routes[row] if int(expert) not in resident_set]
        if not streamed:
            actual_blocks.append(0)
            continue
        base, remainder = divmod(args.block_budget, len(streamed))
        used = 0
        for rank, expert in enumerate(streamed):
            count = min(BLOCKS, base + int(rank < remainder))
            local_index = active_index.get(expert)
            if local_index is not None and count:
                indices = torch.topk(score[row, local_index], count).indices
                selected[row, local_index, indices] = 1
                used += count
        actual_blocks.append(used)
    gate_sparse = sparse.partial_projection(delta_gate, parent_x, selected, device)
    up_sparse = sparse.partial_projection(delta_up, parent_x, selected, device)
    gate_full = low.project(weights["gate4"], parent_x, device) - gate2
    up_full = low.project(weights["up4"], parent_x, device) - up2
    gate_residual = torch.where(resident_local[None, :, None], gate_full, gate_sparse)
    up_residual = torch.where(resident_local[None, :, None], up_full, up_sparse)
    prediction = low.hidden(gate2, gate2 + gate_residual, up2, up2 + up_residual).half().cpu()

    validation = torch.tensor([value == "validation" for value in dataset["split"]])
    routed = dataset["expert_ids"][validation]
    routed_local = torch.tensor([[active_index[int(expert)] for expert in row] for row in routed])
    metrics = low.metric_rows(dataset["exact_hidden_compact"][validation].float(), prediction[validation].float(), routed_local)
    predicted_validation = predicted_routes[validation]
    route_overlap = torch.stack([torch.isin(predicted_validation[row], routed[row]).sum() for row in range(len(routed))]).float() / 8
    block_bytes = BLOCK * UNITS * 2 * 2 // 8
    max_bytes_top8 = args.block_budget * (block_bytes + 1)
    stream_bpw = (max_bytes_top8 / 8) * 8 / DENOMINATOR
    destination_dataset = args.output / f"dataset_layer_{args.layer}.pt"
    variant = f"resident{args.resident_count}_adaptive_predroute_b{args.block_budget}"
    destination_prediction = args.output / f"prediction_layer_{args.layer}_{variant}.pt"
    if not destination_dataset.exists():
        destination_dataset.hardlink_to(dataset_path)
    torch.save(prediction, destination_prediction)
    manifest = {
        "schema": "pr13_h1_resident_q4_adaptive_predicted_route_stream_v1",
        "layer": args.layer,
        "variant": variant,
        "resident_experts_per_layer": args.resident_count,
        "resident_fraction": args.resident_count / 256,
        "resident_expert_bpw_including_q2_scales": 2.25 + 2 * args.resident_count / 256,
        "parent_capture": str(args.parent_capture),
        "route_source": "resident-rollout router logits; no exact-route block assignment",
        "validation_route_set_overlap_mean": float(route_overlap.mean()),
        "validation_route_set_overlap_p10": float(torch.quantile(route_overlap, 0.10)),
        "block_budget_per_top8": args.block_budget,
        "actual_blocks_mean_all_rows": float(np.mean(actual_blocks)),
        "actual_blocks_max": int(max(actual_blocks)),
        "maximum_streamed_bytes_top8": max_bytes_top8,
        "maximum_streamed_bpw": stream_bpw,
        "metrics": metrics,
    }
    (args.output / "export_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "export_completed", **manifest}), flush=True)


if __name__ == "__main__":
    main()
