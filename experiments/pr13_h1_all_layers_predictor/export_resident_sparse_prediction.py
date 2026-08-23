#!/usr/bin/env python3
"""Export static damage-resident Q4 plus dynamic sparse-block H1 responses."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--blocks", type=int, default=17)
    parser.add_argument("--resident-count", type=int, default=40)
    parser.add_argument("--damage-scores", type=Path, default=Path("/workspace/hybrid_q4_damage_frontier_v2/damage_scores.pt"))
    parser.add_argument("--source-experiment", type=Path, default=Path("/workspace/pr13_h1_all_layers_predictor_20260822_v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--low-helper", type=Path, default=Path("/workspace/pr13_low_bpw_pilot.py"))
    parser.add_argument("--sparse-helper", type=Path, default=Path("/workspace/run_sparse_refinement_pilot.py"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/workspace/pr13_average_rate_all_layers_20260822_v1/inputs/checkpoint"))
    parser.add_argument("--trees", type=Path, default=Path("/workspace/codebook_granularity_study/locked/selected_trees.json"))
    parser.add_argument("--oracle-root", type=Path, default=Path("/workspace/set_utility_study/experiments/adaptive_expert_precision_oracle"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    low = load_module("low_bpw_resident_helpers", args.low_helper)
    sparse = load_module("sparse_resident_helpers", args.sparse_helper)
    sys.path[:0] = [str(args.oracle_root / "src"), str(args.oracle_root / "scripts")]
    import run_set_utility_distillation as prior
    from run_mxfp4_selective_pages import tree_from_record

    dataset_path = args.source_experiment / f"dataset_layer_{args.layer}.pt"
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    active = [int(value) for value in dataset["active_experts"]]
    parent_x = dataset["parent_target_x"].float().to(device)
    damage = torch.load(args.damage_scores, map_location="cpu", weights_only=True)
    resident_global = torch.topk(damage[args.layer], args.resident_count).indices.long()
    resident_set = set(resident_global.tolist())
    resident_local = torch.tensor([expert in resident_set for expert in active], dtype=torch.bool, device=device)

    checkpoint_index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    records = json.loads(args.trees.read_text())
    trees = {name: tree_from_record(records[name]) for name in ("gate", "up", "down")}
    weights = {name: np.empty((len(active), low.UNITS, low.HIDDEN), np.float32) for name in ("gate2", "gate4", "up2", "up4")}
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
    selected = sparse.select_top(score_gate + score_up, args.blocks)
    gate_residual = sparse.partial_projection(delta_gate, parent_x, selected, device)
    up_residual = sparse.partial_projection(delta_up, parent_x, selected, device)
    gate_full = low.project(weights["gate4"], parent_x, device) - gate2
    up_full = low.project(weights["up4"], parent_x, device) - up2
    gate_residual = torch.where(resident_local[None, :, None], gate_full, gate_residual)
    up_residual = torch.where(resident_local[None, :, None], up_full, up_residual)
    prediction = low.hidden(gate2, gate2 + gate_residual, up2, up2 + up_residual).half().cpu()

    validation = torch.tensor([value == "validation" for value in dataset["split"]])
    active_index = {expert: index for index, expert in enumerate(active)}
    routed = dataset["expert_ids"][validation]
    routed_local = torch.tensor([[active_index[int(expert)] for expert in row] for row in routed])
    metrics = low.metric_rows(dataset["exact_hidden_compact"][validation].float(), prediction[validation].float(), routed_local)
    hit_rate = float(torch.isin(routed, resident_global).float().mean())

    destination_dataset = args.output / f"dataset_layer_{args.layer}.pt"
    destination_prediction = args.output / f"prediction_layer_{args.layer}_resident{args.resident_count}_sparse_block17.pt"
    if not destination_dataset.exists():
        destination_dataset.hardlink_to(dataset_path)
    torch.save(prediction, destination_prediction)
    stream_payload = sparse.payload_block(args.blocks, len(active), True)
    manifest = {
        "schema": "pr13_h1_resident_q4_dynamic_sparse_block_prediction_v1",
        "layer": args.layer,
        "resident_experts_per_layer": args.resident_count,
        "resident_fraction": args.resident_count / 256,
        "resident_expert_bpw_including_q2_scales": 2.25 + 2 * args.resident_count / 256,
        "active_experts": len(active),
        "active_resident_experts": int(resident_local.sum()),
        "validation_route_slot_resident_hit_rate": hit_rate,
        "prediction_shape": list(prediction.shape),
        "stream_payload": stream_payload,
        "metrics": metrics,
    }
    (args.output / "export_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "export_completed", **manifest}), flush=True)


if __name__ == "__main__":
    main()
