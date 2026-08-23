#!/usr/bin/env python3
"""Screen local nonlinear response maps from resident Q2 gate/up projections."""

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
DENOMINATOR = 2 * HIDDEN * UNITS
SEED = 42


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


def chebyshev(value: torch.Tensor, maximum: int) -> list[torch.Tensor]:
    result = [torch.ones_like(value)]
    if maximum:
        result.append(value)
    for _ in range(2, maximum + 1):
        result.append(2 * value * result[-1] - result[-2])
    return result


def feature_pairs(degree: int, limit: int | None = None) -> list[tuple[int, int]]:
    pairs = [(i, j) for total in range(degree + 1) for i in range(total + 1) for j in (total - i,)]
    if limit is not None and len(pairs) > limit:
        # Remove the most extreme univariate terms first; retain mixed interactions.
        removable = sorted(range(len(pairs)), key=lambda index: (min(pairs[index]), -max(pairs[index])))
        remove = set(removable[:len(pairs) - limit])
        pairs = [pair for index, pair in enumerate(pairs) if index not in remove]
    return pairs


def make_features(gate: torch.Tensor, up: torch.Tensor, statistics: tuple[torch.Tensor, ...], pairs: list[tuple[int, int]]) -> torch.Tensor:
    mean_g, scale_g, mean_u, scale_u = statistics
    g = ((gate - mean_g) * scale_g / 3.0).clamp(-1.0, 1.0)
    u = ((up - mean_u) * scale_u / 3.0).clamp(-1.0, 1.0)
    maximum = max(max(pair) for pair in pairs)
    tg, tu = chebyshev(g, maximum), chebyshev(u, maximum)
    return torch.stack([tg[i] * tu[j] for i, j in pairs], dim=-1)


def statistics(gate: torch.Tensor, up: torch.Tensor) -> tuple[torch.Tensor, ...]:
    mean_g, mean_u = gate.mean(0), up.mean(0)
    scale_g = gate.std(0).clamp_min(1e-4).reciprocal()
    scale_u = up.std(0).clamp_min(1e-4).reciprocal()
    return mean_g, scale_g, mean_u, scale_u


def fit_and_predict(
    train_gate: torch.Tensor,
    train_up: torch.Tensor,
    train_target: torch.Tensor,
    eval_gate: torch.Tensor,
    eval_up: torch.Tensor,
    pairs: list[tuple[int, int]],
    alpha: float,
    expert_batch: int,
) -> torch.Tensor:
    outputs = []
    experts = train_gate.shape[1]
    rank = len(pairs)
    for start in range(0, experts, expert_batch):
        stop = min(experts, start + expert_batch)
        gate = train_gate[:, start:stop]
        up = train_up[:, start:stop]
        stats = statistics(gate, up)
        x = make_features(gate, up, stats, pairs).permute(1, 2, 0, 3).contiguous()
        y = train_target[:, start:stop].permute(1, 3, 0, 2).contiguous()
        # x [experts, units, rows, features], y [experts, units, rows, components]
        gram = x.transpose(-1, -2) @ x
        cross = x.transpose(-1, -2) @ y
        identity = torch.eye(rank, device=x.device)
        coefficient = torch.linalg.solve(gram + identity * (alpha * len(train_gate)), cross)
        coefficient_q = coefficient.half().float()
        eval_x = make_features(eval_gate[:, start:stop], eval_up[:, start:stop], stats, pairs).permute(1, 2, 0, 3)
        prediction = eval_x @ coefficient_q
        outputs.append(prediction.permute(2, 0, 3, 1).contiguous())
        del gate, up, stats, x, y, gram, cross, coefficient, coefficient_q, eval_x, prediction
        torch.cuda.empty_cache()
    return torch.cat(outputs, dim=1)


def payload(features: int, experts: int) -> dict[str, Any]:
    # FP16 coefficients for four components plus FP16 mean/invstd for gate/up.
    per_expert = features * 4 * UNITS * 2 + 4 * UNITS * 2
    bpw = per_expert * 8 / DENOMINATOR
    if bpw > 0.55 + 1e-12:
        raise RuntimeError((features, bpw))
    return {
        "features": features,
        "streamed_bytes_per_expert": per_expert,
        "streamed_bytes_top8": per_expert * 8,
        "streamed_bytes_all_active_experts": per_expert * experts,
        "streamed_bpw": bpw,
        "macs_top8": features * 4 * UNITS * 8,
    }


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
    requests = np.asarray(exact["request_id"][rows]).astype(str)
    fit_np = (splits == "train") & (requests != args.tune_request)
    tune_np = (splits == "train") & (requests == args.tune_request)
    all_train_np = splits == "train"
    validation_np = splits == "validation"
    routed_experts = np.asarray(exact["expert_ids"][rows])[validation_np]
    active = sorted(set(routed_experts.ravel().tolist()))
    active_index = {expert: index for index, expert in enumerate(active)}
    routed_local = torch.tensor([[active_index[int(expert)] for expert in row] for row in routed_experts], dtype=torch.int64, device=device)
    recorder("alignment", layer=layer, fit_rows=int(fit_np.sum()), tune_rows=int(tune_np.sum()), validation_rows=int(validation_np.sum()), active_experts=len(active))

    weights = {name: np.empty((len(active), UNITS, HIDDEN), np.float32) for name in ("gate2", "gate4", "up2", "up4")}
    for local_index, expert in enumerate(active):
        decoded = prior.decode_expert(args.checkpoint, checkpoint_index, trees, layer, expert)
        for name in ("gate", "up"):
            weights[name + "2"][local_index] = decoded[name][0]
            weights[name + "4"][local_index] = decoded[name][2]
        if local_index % 16 == 15 or local_index + 1 == len(active):
            recorder("decode", layer=layer, completed=local_index + 1, total=len(active))
        del decoded
    exact_projection = {name: helper.project(weights[name], exact_x, device) for name in weights}
    truth = helper.hidden(exact_projection["gate2"], exact_projection["gate4"], exact_projection["up2"], exact_projection["up4"])
    truth_components = helper.components(truth)
    parent_gate = helper.project(weights["gate2"], parent_x, device)
    parent_up = helper.project(weights["up2"], parent_x, device)
    parent_b = helper.silu(parent_gate) * parent_up
    target = truth_components.clone()
    target[:, :, 0] -= parent_b
    masks = {name: torch.from_numpy(value).to(device) for name, value in (("fit", fit_np), ("tune", tune_np), ("train", all_train_np), ("validation", validation_np))}
    results = []
    definitions = [(2, None), (4, None), (6, None), (7, 34)]
    for degree, limit in definitions:
        pairs = feature_pairs(degree, limit)
        best = None
        for alpha in args.alphas:
            tune_prediction = fit_and_predict(parent_gate[masks["fit"]], parent_up[masks["fit"]], target[masks["fit"]], parent_gate[masks["tune"]], parent_up[masks["tune"]], pairs, alpha, args.expert_batch)
            error = (tune_prediction - target[masks["tune"]]).square()
            relative = float(error.sum() / target[masks["tune"]].square().sum().clamp_min(1e-30))
            if best is None or relative < best[0]:
                best = (relative, alpha, float(error.mean()))
            del tune_prediction, error
        assert best is not None
        prediction = fit_and_predict(parent_gate[masks["train"]], parent_up[masks["train"]], target[masks["train"]], parent_gate[masks["validation"]], parent_up[masks["validation"]], pairs, best[1], args.expert_batch)
        prediction[:, :, 0] += parent_b[masks["validation"]]
        prediction_hidden = helper.hidden_from_components(prediction)
        record = {
            "name": f"local_chebyshev_d{degree}_k{len(pairs)}",
            "degree": degree,
            "alpha": best[1],
            "tune_relative_mse": best[0],
            "tune_mse": best[2],
            "payload": payload(len(pairs), len(active)),
            "metrics": helper.metric_rows(truth[masks["validation"]], prediction_hidden, routed_local),
        }
        results.append(record)
        recorder("candidate", layer=layer, result=record)
        del prediction, prediction_hidden
    result = {"layer": layer, "active_experts": len(active), "results": results, "seconds": time.time() - started}
    recorder("layer_completed", **result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default="14")
    parser.add_argument("--helper", type=Path, default=Path("/workspace/pr13_low_bpw_pilot.py"))
    parser.add_argument("--exact-root", type=Path, default=Path("/workspace/all_layer_split_interaction_study/capture"))
    parser.add_argument("--parent-root", type=Path, default=Path("/workspace/embedded_parent_capture_20260821_v1"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/workspace/pr13_average_rate_all_layers_20260822_v1/inputs/checkpoint"))
    parser.add_argument("--trees", type=Path, default=Path("/workspace/codebook_granularity_study/locked/selected_trees.json"))
    parser.add_argument("--oracle-root", type=Path, default=Path("/workspace/set_utility_study/experiments/adaptive_expert_precision_oracle"))
    parser.add_argument("--tune-request", default="mxfp4-confirm-004")
    parser.add_argument("--alphas", default="0.0001,0.001,0.01,0.1,1")
    parser.add_argument("--expert-batch", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.alphas = tuple(float(value) for value in args.alphas.split(","))
    layers = tuple(int(value) for value in args.layers.split(","))
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    recorder = Recorder(args.output / "run.jsonl")
    torch.manual_seed(SEED)
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
    recorder("screen_started", layers=layers, alphas=args.alphas, gpu=torch.cuda.get_device_name(0), budget_bpw=0.55)
    results = [run_layer(args, layer, helper, prior, checkpoint_index, trees, recorder, device) for layer in layers]
    atomic_json(args.output / "result.json", {"schema": "pr13_h1_local_response_map_screen_v1", "layers": results})
    recorder("screen_completed", result=str(args.output / "result.json"))


if __name__ == "__main__":
    main()
