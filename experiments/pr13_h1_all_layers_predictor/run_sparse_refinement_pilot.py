#!/usr/bin/env python3
"""Screen exact sparse Q2->Q4 refinement sketches under 0.55 streamed bpw."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


HIDDEN = 2048
UNITS = 512
BLOCK = 32
BLOCKS = HIDDEN // BLOCK
DENOMINATOR = 2 * HIDDEN * UNITS


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("low_bpw_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def payload_block(blocks: int, experts: int, shared_indices: bool) -> dict[str, Any]:
    weight_bytes = blocks * BLOCK * UNITS * 2 * 2 // 8
    index_bytes = blocks if shared_indices else 2 * blocks
    per_expert = weight_bytes + index_bytes
    return payload_record(per_expert, experts, {"selected_input_blocks": blocks, "shared_indices": shared_indices})


def payload_rows(rows: int, experts: int, shared_indices: bool) -> dict[str, Any]:
    weight_bytes = rows * HIDDEN * 2 * 2 // 8
    index_bytes = rows * 2 if shared_indices else rows * 4
    per_expert = weight_bytes + index_bytes
    return payload_record(per_expert, experts, {"selected_output_rows": rows, "shared_indices": shared_indices})


def payload_record(per_expert: int, experts: int, extra: dict[str, Any]) -> dict[str, Any]:
    bpw = per_expert * 8 / DENOMINATOR
    if bpw > 0.55 + 1e-12:
        raise RuntimeError(bpw)
    return {
        **extra,
        "streamed_bytes_per_expert": per_expert,
        "streamed_bytes_top8": per_expert * 8,
        "streamed_bytes_all_active_experts": per_expert * experts,
        "streamed_bpw": bpw,
    }


def partial_projection(delta: np.ndarray, x: torch.Tensor, selected: torch.Tensor, device: torch.device) -> torch.Tensor:
    # selected [samples, experts, blocks]
    x_blocks = x.reshape(len(x), BLOCKS, BLOCK)
    output = torch.empty((len(x), delta.shape[0], UNITS), dtype=torch.float32, device=device)
    for expert in range(delta.shape[0]):
        weight = torch.from_numpy(delta[expert]).to(device).reshape(UNITS, BLOCKS, BLOCK)
        contribution = torch.einsum("nbd,obd->nbo", x_blocks, weight)
        output[:, expert] = (contribution * selected[:, expert, :, None]).sum(dim=1)
        del weight, contribution
    torch.cuda.empty_cache()
    return output


def score_blocks(delta_gate: np.ndarray, delta_up: np.ndarray, x: torch.Tensor, mode: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x_blocks = x.reshape(len(x), BLOCKS, BLOCK)
    gate = torch.from_numpy(delta_gate).to(device).square().sum(dim=1).reshape(len(delta_gate), BLOCKS, BLOCK)
    up = torch.from_numpy(delta_up).to(device).square().sum(dim=1).reshape(len(delta_up), BLOCKS, BLOCK)
    if mode == "block_norm":
        activation = x_blocks.square().sum(-1)
        score_gate = activation[:, None] * gate.sum(-1)[None]
        score_up = activation[:, None] * up.sum(-1)[None]
    elif mode == "column_norm":
        activation = x_blocks.square()
        score_gate = torch.einsum("nbd,ebd->neb", activation, gate)
        score_up = torch.einsum("nbd,ebd->neb", activation, up)
    else:
        raise ValueError(mode)
    del gate, up
    return score_gate, score_up


def select_top(score: torch.Tensor, count: int) -> torch.Tensor:
    indices = torch.topk(score, count, dim=-1).indices
    result = torch.zeros_like(score, dtype=torch.float32)
    result.scatter_(-1, indices, 1.0)
    return result


def run_layer(args: argparse.Namespace, layer: int, helper: Any, prior: Any, checkpoint_index: Mapping[str, str], trees: Mapping[str, Any], recorder: Recorder, device: torch.device) -> dict[str, Any]:
    started = time.time()
    exact = helper.load_layer(args.exact_root, layer)
    parent = helper.load_layer(args.parent_root, layer)
    parent_ids = helper.identity_map(parent)
    rows = np.flatnonzero(exact["position"] >= 1)
    exact_x = torch.tensor(exact["x"][rows], dtype=torch.float32, device=device)
    parent_x = torch.tensor(np.stack([
        parent["x"][parent_ids[(str(exact["request_id"][i]), int(exact["position"][i]), str(exact["prefix_hash"][i]))]]
        for i in rows
    ]), dtype=torch.float32, device=device)
    splits = np.asarray(exact["split"][rows]).astype(str)
    train = torch.from_numpy(splits == "train").to(device)
    validation = torch.from_numpy(splits == "validation").to(device)
    routed_experts = np.asarray(exact["expert_ids"][rows])[splits == "validation"]
    active = sorted(set(routed_experts.ravel().tolist()))
    active_index = {expert: index for index, expert in enumerate(active)}
    routed_local = torch.tensor([[active_index[int(expert)] for expert in row] for row in routed_experts], dtype=torch.int64, device=device)
    recorder("alignment", layer=layer, train_rows=int(train.sum()), validation_rows=int(validation.sum()), active_experts=len(active))

    weights = {name: np.empty((len(active), UNITS, HIDDEN), np.float32) for name in ("gate2", "gate4", "up2", "up4")}
    for local_index, expert in enumerate(active):
        decoded = prior.decode_expert(args.checkpoint, checkpoint_index, trees, layer, expert)
        for name in ("gate", "up"):
            weights[name + "2"][local_index] = decoded[name][0]
            weights[name + "4"][local_index] = decoded[name][2]
        if local_index % 16 == 15 or local_index + 1 == len(active):
            recorder("decode", layer=layer, completed=local_index + 1, total=len(active))
        del decoded
    delta_gate = weights["gate4"] - weights["gate2"]
    delta_up = weights["up4"] - weights["up2"]
    exact_projection = {name: helper.project(weights[name], exact_x, device) for name in weights}
    truth = helper.hidden(exact_projection["gate2"], exact_projection["gate4"], exact_projection["up2"], exact_projection["up4"])
    parent_projection = {name: helper.project(weights[name], parent_x, device) for name in weights}
    parent_control = helper.hidden(parent_projection["gate2"], parent_projection["gate4"], parent_projection["up2"], parent_projection["up4"])
    parent_g2, parent_u2 = parent_projection["gate2"], parent_projection["up2"]
    parent_b = helper.silu(parent_g2) * parent_u2
    results = [{
        "name": "full_q4_parent_state_control",
        "payload": {"streamed_bpw": 2.0},
        "metrics": helper.metric_rows(truth[validation], parent_control[validation], routed_local),
    }]

    def add(name: str, predicted: torch.Tensor, payload: dict[str, Any], resident_metadata_bytes_per_expert: int) -> None:
        record = {
            "name": name,
            "payload": payload,
            "resident_metadata_bytes_per_expert": resident_metadata_bytes_per_expert,
            "metrics": helper.metric_rows(truth[validation], predicted, routed_local),
        }
        results.append(record)
        recorder("candidate", layer=layer, result=record)

    # Fixed paired output rows selected by in-domain training response-error reduction.
    truth_components = helper.components(truth)
    control_components = helper.components(parent_control)
    base_components = torch.zeros_like(control_components)
    base_components[:, :, 0] = parent_b
    baseline_error = (truth_components[train] - base_components[train]).square().sum(dim=(0, 2))
    paired_error = (truth_components[train] - control_components[train]).square().sum(dim=(0, 2))
    paired_gain = baseline_error - paired_error
    paired_indices = torch.topk(paired_gain, args.rows, dim=-1).indices
    paired_mask = torch.zeros((len(active), UNITS), dtype=torch.bool, device=device)
    paired_mask.scatter_(1, paired_indices, True)
    prediction_components = base_components[validation].clone()
    prediction_components = torch.where(paired_mask[None, :, None, :], control_components[validation], prediction_components)
    add(f"fixed_paired_output_rows_{args.rows}", helper.hidden_from_components(prediction_components), payload_rows(args.rows, len(active), True), args.rows * 2)
    del prediction_components

    # Independently selected gate/up rows; iota is available only at their intersection.
    gate_state = base_components.clone()
    gate_state[:, :, 1] = control_components[:, :, 1]
    up_state = base_components.clone()
    up_state[:, :, 2] = control_components[:, :, 2]
    gate_gain = baseline_error - (truth_components[train] - gate_state[train]).square().sum(dim=(0, 2))
    up_gain = baseline_error - (truth_components[train] - up_state[train]).square().sum(dim=(0, 2))
    gate_mask = torch.zeros((len(active), UNITS), dtype=torch.bool, device=device)
    up_mask = torch.zeros_like(gate_mask)
    gate_mask.scatter_(1, torch.topk(gate_gain, args.rows, dim=-1).indices, True)
    up_mask.scatter_(1, torch.topk(up_gain, args.rows, dim=-1).indices, True)
    prediction_components = base_components[validation].clone()
    prediction_components[:, :, 1] = torch.where(gate_mask[None], control_components[validation, :, 1], prediction_components[:, :, 1])
    prediction_components[:, :, 2] = torch.where(up_mask[None], control_components[validation, :, 2], prediction_components[:, :, 2])
    both_mask = gate_mask & up_mask
    prediction_components[:, :, 3] = torch.where(both_mask[None], control_components[validation, :, 3], prediction_components[:, :, 3])
    add(f"fixed_independent_output_rows_{args.rows}", helper.hidden_from_components(prediction_components), payload_rows(args.rows, len(active), False), args.rows * 4)
    del prediction_components

    # Activation-shaped input-block selection using compact resident norm metadata.
    for mode, metadata_bytes in (("block_norm", 2 * BLOCKS * 2), ("column_norm", 2 * HIDDEN * 2)):
        score_gate, score_up = score_blocks(delta_gate, delta_up, parent_x[validation], mode, device)
        for shared in (True, False):
            if shared:
                common = select_top(score_gate + score_up, args.blocks)
                selected_gate, selected_up = common, common
            else:
                selected_gate = select_top(score_gate, args.blocks)
                selected_up = select_top(score_up, args.blocks)
            gate_residual = partial_projection(delta_gate, parent_x[validation], selected_gate, device)
            up_residual = partial_projection(delta_up, parent_x[validation], selected_up, device)
            prediction = helper.hidden(parent_g2[validation], parent_g2[validation] + gate_residual, parent_u2[validation], parent_u2[validation] + up_residual)
            add(f"dynamic_{mode}_{'shared' if shared else 'independent'}_{args.blocks}_blocks", prediction, payload_block(args.blocks, len(active), shared), metadata_bytes)
            del gate_residual, up_residual, prediction, selected_gate, selected_up
        del score_gate, score_up
    result = {"layer": layer, "active_experts": len(active), "results": results, "seconds": time.time() - started}
    # Multi-resolution union: exact full refinements for concentrated rows and
    # activation-selected blocks for every remaining row.
    score_gate, score_up = score_blocks(delta_gate, delta_up, parent_x[validation], "block_norm", device)
    for selected_rows, selected_blocks in ((108, 4), (76, 8), (44, 12)):
        row_indices = torch.topk(paired_gain, selected_rows, dim=-1).indices
        row_mask = torch.zeros((len(active), UNITS), dtype=torch.bool, device=device)
        row_mask.scatter_(1, row_indices, True)
        common = select_top(score_gate + score_up, selected_blocks)
        gate_partial = partial_projection(delta_gate, parent_x[validation], common, device)
        up_partial = partial_projection(delta_up, parent_x[validation], common, device)
        gate_full = parent_projection["gate4"][validation] - parent_g2[validation]
        up_full = parent_projection["up4"][validation] - parent_u2[validation]
        gate_residual = torch.where(row_mask[None], gate_full, gate_partial)
        up_residual = torch.where(row_mask[None], up_full, up_partial)
        prediction = helper.hidden(parent_g2[validation], parent_g2[validation] + gate_residual, parent_u2[validation], parent_u2[validation] + up_residual)
        row_bytes = selected_rows * HIDDEN * 2 * 2 // 8 + selected_rows * 2
        block_bytes = selected_blocks * BLOCK * UNITS * 2 * 2 // 8 + selected_blocks
        hybrid_payload = payload_record(row_bytes + block_bytes, len(active), {
            "selected_output_rows": selected_rows,
            "selected_input_blocks": selected_blocks,
            "shared_indices": True,
        })
        add(f"hybrid_{selected_rows}_rows_{selected_blocks}_blocks", prediction, hybrid_payload, 2 * BLOCKS * 2 + selected_rows * 2)
        del row_indices, row_mask, common, gate_partial, up_partial, gate_residual, up_residual, prediction
    del score_gate, score_up
    recorder("layer_completed", **result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default="14")
    parser.add_argument("--rows", type=int, default=140)
    parser.add_argument("--blocks", type=int, default=17)
    parser.add_argument("--helper", type=Path, default=Path("/workspace/pr13_low_bpw_pilot.py"))
    parser.add_argument("--exact-root", type=Path, default=Path("/workspace/all_layer_split_interaction_study/capture"))
    parser.add_argument("--parent-root", type=Path, default=Path("/workspace/embedded_parent_capture_20260821_v1"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/workspace/pr13_average_rate_all_layers_20260822_v1/inputs/checkpoint"))
    parser.add_argument("--trees", type=Path, default=Path("/workspace/codebook_granularity_study/locked/selected_trees.json"))
    parser.add_argument("--oracle-root", type=Path, default=Path("/workspace/set_utility_study/experiments/adaptive_expert_precision_oracle"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    layers = tuple(int(value) for value in args.layers.split(","))
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    recorder = Recorder(args.output / "run.jsonl")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    helper = load_module(args.helper)
    source, scripts = args.oracle_root / "src", args.oracle_root / "scripts"
    sys.path[:0] = [str(source), str(scripts)]
    import run_set_utility_distillation as prior
    from run_mxfp4_selective_pages import tree_from_record
    checkpoint_index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    records = json.loads(args.trees.read_text())
    trees = {name: tree_from_record(records[name]) for name in ("gate", "up", "down")}
    recorder("screen_started", layers=layers, rows=args.rows, blocks=args.blocks, gpu=torch.cuda.get_device_name(0), budget_bpw=0.55)
    results = [run_layer(args, layer, helper, prior, checkpoint_index, trees, recorder, device) for layer in layers]
    atomic_json(args.output / "result.json", {"schema": "pr13_h1_sparse_exact_refinement_screen_v1", "layers": results})
    recorder("screen_completed", result=str(args.output / "result.json"))


if __name__ == "__main__":
    main()
