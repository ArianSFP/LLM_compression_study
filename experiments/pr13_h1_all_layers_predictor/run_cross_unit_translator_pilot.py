#!/usr/bin/env python3
"""Fit legal 2-bit cross-unit Q2-projection to Q4-residual translators."""

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


def quantize_four_centroids(weight: torch.Tensor, iterations: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    # weight [output, input]; four FP16 centroids per output row.
    mean = weight.mean(dim=-1, keepdim=True)
    std = weight.std(dim=-1, keepdim=True).clamp_min(1e-8)
    offsets = torch.tensor((-1.5, -0.5, 0.5, 1.5), device=weight.device)
    centroids = mean + std * offsets
    for _ in range(iterations):
        code = (weight[:, :, None] - centroids[:, None, :]).abs().argmin(dim=-1)
        one_hot = torch.nn.functional.one_hot(code, 4).to(weight.dtype)
        sums = (one_hot * weight[:, :, None]).sum(dim=1)
        counts = one_hot.sum(dim=1)
        updated = sums / counts.clamp_min(1)
        centroids = torch.where(counts > 0, updated, centroids)
    centroids_q = centroids.half().float()
    code = (weight[:, :, None] - centroids_q[:, None, :]).abs().argmin(dim=-1)
    dequant = torch.gather(centroids_q, 1, code)
    return dequant, centroids_q


def fit_head(
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_tune: torch.Tensor,
    y_tune: torch.Tensor,
    x_validation: torch.Tensor,
    alphas: tuple[float, ...],
) -> tuple[torch.Tensor, dict[str, Any]]:
    mean_x = x_fit.mean(0)
    mean_y = y_fit.mean(0)
    x = x_fit - mean_x
    y = y_fit - mean_y
    gram = x.T @ x
    cross = x.T @ y
    scale = torch.trace(gram) / UNITS
    identity = torch.eye(UNITS, device=x.device)
    best = None
    for alpha in alphas:
        transpose = torch.linalg.solve(gram + identity * (scale * alpha), cross)
        dequant, centroids = quantize_four_centroids(transpose.T)
        bias = (y_fit - x_fit @ dequant.T).mean(0).half().float()
        tune_prediction = x_tune @ dequant.T + bias
        error = (tune_prediction - y_tune).square()
        relative = float(error.sum() / y_tune.square().sum().clamp_min(1e-30))
        if best is None or relative < best[0]:
            best = (relative, alpha, dequant, centroids, bias, float(error.mean()))
        del transpose, dequant, centroids, bias, tune_prediction, error
    assert best is not None
    prediction = x_validation @ best[2].T + best[4]
    return prediction, {"alpha": best[1], "tune_relative_mse": best[0], "tune_mse": best[5]}


def payload(experts: int) -> dict[str, Any]:
    # Two 512x512 two-bit code matrices, 4 FP16 centroids/output, FP16 bias/output.
    codes = 2 * UNITS * UNITS * 2 // 8
    centroids = 2 * UNITS * 4 * 2
    biases = 2 * UNITS * 2
    per_expert = codes + centroids + biases
    bpw = per_expert * 8 / DENOMINATOR
    if bpw > 0.55:
        raise RuntimeError(bpw)
    return {
        "streamed_bytes_per_expert": per_expert,
        "streamed_bytes_top8": per_expert * 8,
        "streamed_bytes_all_active_experts": per_expert * experts,
        "streamed_bpw": bpw,
        "macs_top8": 2 * UNITS * UNITS * 8,
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
    validation_np = np.asarray(exact["split"][rows]).astype(str) == "validation"
    validation = torch.from_numpy(validation_np).to(device)
    routed_experts = np.asarray(exact["expert_ids"][rows])[validation_np]
    active = sorted(set(routed_experts.ravel().tolist()))
    active_index = {expert: index for index, expert in enumerate(active)}
    routed_local = torch.tensor([[active_index[int(expert)] for expert in row] for row in routed_experts], dtype=torch.int64, device=device)
    recorder("alignment", layer=layer, validation_rows=int(validation.sum()), active_experts=len(active))

    weights = {name: np.empty((len(active), UNITS, HIDDEN), np.float32) for name in ("gate2", "gate4", "up2", "up4")}
    for local_index, expert in enumerate(active):
        decoded = prior.decode_expert(args.checkpoint, checkpoint_index, trees, layer, expert)
        for name in ("gate", "up"):
            weights[name + "2"][local_index] = decoded[name][0]
            weights[name + "4"][local_index] = decoded[name][2]
        if local_index % 16 == 15 or local_index + 1 == len(active):
            recorder("decode", layer=layer, completed=local_index + 1, total=len(active))
        del decoded

    # HARP request-disjoint fitting/tuning states; target is the deterministic
    # Q4-Q2 weight response on the parent state, avoiding precision-domain leakage.
    request_index = torch.load(args.fitting_root / "request_index.pt", map_location="cpu", weights_only=True)
    available = torch.load(args.fitting_root / "target_available.pt", map_location="cpu", weights_only=True)[:, 0, layer]
    generator = torch.Generator().manual_seed(SEED)
    order = torch.randperm(int(request_index.max()) + 1, generator=generator)
    fit_mask = available & torch.isin(request_index, order[:224])
    tune_mask = available & torch.isin(request_index, order[224:])
    store = torch.load(args.fitting_root / "parent_target.pt", map_location="cpu", mmap=True, weights_only=True)
    fit_x = store[fit_mask, 0, layer].float().to(device)
    tune_x = store[tune_mask, 0, layer].float().to(device)
    del store
    both = torch.cat((fit_x, tune_x), dim=0)
    split_at = len(fit_x)
    fitting_projection = {name: helper.project(weights[name], both, device) for name in weights}

    exact_projection = {name: helper.project(weights[name], exact_x[validation], device) for name in weights}
    truth = helper.hidden(exact_projection["gate2"], exact_projection["gate4"], exact_projection["up2"], exact_projection["up4"])
    parent_g2 = helper.project(weights["gate2"], parent_x[validation], device)
    parent_u2 = helper.project(weights["up2"], parent_x[validation], device)
    predicted_gate = torch.empty_like(parent_g2)
    predicted_up = torch.empty_like(parent_u2)
    tuning = {"gate": [], "up": []}
    for expert in range(len(active)):
        for name, destination, validation_input in (("gate", predicted_gate, parent_g2), ("up", predicted_up, parent_u2)):
            input_all = fitting_projection[name + "2"][:, expert]
            target_all = fitting_projection[name + "4"][:, expert] - input_all
            prediction, record = fit_head(input_all[:split_at], target_all[:split_at], input_all[split_at:], target_all[split_at:], validation_input[:, expert], args.alphas)
            destination[:, expert] = prediction
            tuning[name].append(record)
        if expert % 8 == 7 or expert + 1 == len(active):
            recorder("fit_progress", layer=layer, completed=expert + 1, total=len(active))
    prediction = helper.hidden(parent_g2, parent_g2 + predicted_gate, parent_u2, parent_u2 + predicted_up)
    record = {
        "name": "cross_unit_2bit_four_centroid",
        "payload": payload(len(active)),
        "metrics": helper.metric_rows(truth, prediction, routed_local),
        "tuning": {
            name: {
                "relative_mse_mean": float(np.mean([row["tune_relative_mse"] for row in values])),
                "alpha_counts": {str(alpha): sum(row["alpha"] == alpha for row in values) for alpha in args.alphas},
            }
            for name, values in tuning.items()
        },
    }
    recorder("candidate", layer=layer, result=record)
    result = {"layer": layer, "active_experts": len(active), "result": record, "seconds": time.time() - started}
    recorder("layer_completed", **result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default="14")
    parser.add_argument("--alphas", default="0.01,0.1,1")
    parser.add_argument("--helper", type=Path, default=Path("/workspace/pr13_low_bpw_pilot.py"))
    parser.add_argument("--exact-root", type=Path, default=Path("/workspace/all_layer_split_interaction_study/capture"))
    parser.add_argument("--parent-root", type=Path, default=Path("/workspace/embedded_parent_capture_20260821_v1"))
    parser.add_argument("--fitting-root", type=Path, default=Path("/workspace/harp_fitting_mxfp4_parent_pairs_v1"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/workspace/pr13_average_rate_all_layers_20260822_v1/inputs/checkpoint"))
    parser.add_argument("--trees", type=Path, default=Path("/workspace/codebook_granularity_study/locked/selected_trees.json"))
    parser.add_argument("--oracle-root", type=Path, default=Path("/workspace/set_utility_study/experiments/adaptive_expert_precision_oracle"))
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
    atomic_json(args.output / "result.json", {"schema": "pr13_h1_cross_unit_translator_screen_v1", "layers": results})
    recorder("screen_completed", result=str(args.output / "result.json"))


if __name__ == "__main__":
    main()
