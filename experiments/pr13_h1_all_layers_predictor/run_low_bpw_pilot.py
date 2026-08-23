#!/usr/bin/env python3
"""Quick <=0.55 streamed-bpw H1 response-field predictor pilot.

The resident input is the all-Q2 H1 activation.  Expert-specific streamed payload
is counted exactly over the gate+up residual-weight denominator (2*512*2048).
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


HIDDEN = 2048
UNITS = 512
HORIZON = 1
SEED = 42
RESIDUAL_WEIGHT_COUNT = 2 * UNITS * HIDDEN


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


def load_h4_helpers(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("pr13_h4_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import helper module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_layer(root: Path, layer: int) -> dict[str, np.ndarray]:
    pieces: dict[str, list[np.ndarray]] = defaultdict(list)
    for split in ("train", "validation"):
        with np.load(root / f"{split}_layer_{layer:02d}.npz", allow_pickle=False) as loaded:
            for name in loaded.files:
                pieces[name].append(np.asarray(loaded[name]))
    return {name: np.concatenate(values, axis=0) for name, values in pieces.items()}


def identity_map(data: Mapping[str, np.ndarray]) -> dict[tuple[str, int, str], int]:
    result = {}
    for index in range(len(data["position"])):
        key = (str(data["request_id"][index]), int(data["position"][index]), str(data["prefix_hash"][index]))
        if key in result:
            raise RuntimeError(f"duplicate identity {key}")
        result[key] = index
    return result


def silu(value: torch.Tensor) -> torch.Tensor:
    return value * torch.sigmoid(value)


def project(weights: np.ndarray, activations: torch.Tensor, device: torch.device) -> torch.Tensor:
    shape = weights.shape
    tensor = torch.from_numpy(weights).to(device)
    result = (tensor.reshape(shape[0] * shape[1], shape[2]) @ activations.T).T
    result = result.reshape(len(activations), shape[0], shape[1])
    del tensor
    torch.cuda.empty_cache()
    return result


def hidden(g2: torch.Tensor, g4: torch.Tensor, u2: torch.Tensor, u4: torch.Tensor) -> torch.Tensor:
    return torch.stack((silu(g2) * u2, silu(g2) * u4, silu(g4) * u2, silu(g4) * u4), dim=-2)


def components(hidden_value: torch.Tensor) -> torch.Tensor:
    h22, h24, h42, h44 = hidden_value.unbind(dim=-2)
    return torch.stack((h22, h42 - h22, h24 - h22, h44 - h42 - h24 + h22), dim=-2)


def hidden_from_components(value: torch.Tensor) -> torch.Tensor:
    b, delta_g, delta_u, interaction = value.unbind(dim=-2)
    return torch.stack((b, b + delta_u, b + delta_g, b + delta_g + delta_u + interaction), dim=-2)


def metric_rows(truth: torch.Tensor, prediction: torch.Tensor, routed_local: torch.Tensor) -> list[dict[str, Any]]:
    names = ("b", "delta_g", "delta_u", "iota")
    rows = torch.arange(len(routed_local), device=truth.device)[:, None]
    target = components(truth)[rows, routed_local].double()
    pred = components(prediction)[rows, routed_local].double()
    result = []
    for index, name in enumerate(names):
        t = target[:, :, index]
        p = pred[:, :, index]
        cosine = (t * p).sum(-1) / (t.square().sum(-1).sqrt() * p.square().sum(-1).sqrt()).clamp_min(1e-30)
        error = (p - t).square()
        result.append({
            "component": name,
            "cosine": float(cosine.mean()),
            "mse": float(error.mean()),
            "relative_mse": float(error.sum() / t.square().sum().clamp_min(1e-30)),
            "target_rms": float(t.square().mean().sqrt()),
            "vectors": int(t.shape[0] * t.shape[1]),
        })
    return result


def payload(rank_sum: int, heads: int, active_experts: int) -> dict[str, Any]:
    per_expert = rank_sum * UNITS + heads * UNITS * 4
    bpw = per_expert * 8 / RESIDUAL_WEIGHT_COUNT
    if bpw > 0.55 + 1e-12:
        raise RuntimeError(f"candidate exceeds budget: {bpw:.9f} bpw")
    return {
        "streamed_bytes_per_expert": per_expert,
        "streamed_bytes_top8": per_expert * 8,
        "streamed_bytes_all_active_experts": per_expert * active_experts,
        "streamed_bpw": bpw,
        "rank_sum": rank_sum,
        "heads": heads,
    }


def quantize_rows(head: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (head.abs().amax(dim=-1) / 127.0).clamp_min(1e-12)
    qhead = torch.round(head / scale[:, None]).clamp(-127, 127).to(torch.int8)
    scale_q = scale.half().float()
    return qhead, scale_q


def operator_residual(
    x: torch.Tensor,
    mean: torch.Tensor,
    basis: torch.Tensor,
    delta: np.ndarray,
    rank: int,
    device: torch.device,
) -> torch.Tensor:
    mean_q = mean.half().float()
    basis_q = basis[:, :rank].half().float()
    z = (x - mean_q) @ basis_q
    flat = torch.from_numpy(delta).to(device).reshape(-1, HIDDEN)
    head = flat @ basis_q
    bias = (flat @ mean_q).half().float()
    qhead, scale = quantize_rows(head)
    result = z @ (qhead.float() * scale[:, None]).T + bias
    result = result.reshape(len(x), delta.shape[0], UNITS)
    del flat, head, bias, qhead, scale, z
    torch.cuda.empty_cache()
    return result


def operator_components(
    x: torch.Tensor,
    mean: torch.Tensor,
    basis: torch.Tensor,
    delta_gate: np.ndarray,
    delta_up: np.ndarray,
    rank: int,
    g2: torch.Tensor,
    u2: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    gate = operator_residual(x, mean, basis, delta_gate, rank, device)
    up = operator_residual(x, mean, basis, delta_up, rank, device)
    result = components(hidden(g2, g2 + gate, u2, u2 + up))
    del gate, up
    torch.cuda.empty_cache()
    return result


def fit_direct(
    z_fit_all: torch.Tensor,
    z_tune_all: torch.Tensor,
    y_fit: torch.Tensor,
    y_tune: torch.Tensor,
    ranks: list[int],
    alphas: tuple[float, ...],
) -> tuple[list[dict[str, torch.Tensor]], dict[str, Any]]:
    if y_fit.shape[2] != len(ranks):
        raise ValueError("one rank required per component head")
    artifacts = []
    tuning = []
    for component_index, rank in enumerate(ranks):
        z_fit = z_fit_all[:, :rank]
        z_tune = z_tune_all[:, :rank]
        target_fit = y_fit[:, :, component_index].reshape(len(y_fit), -1)
        target_tune = y_tune[:, :, component_index].reshape(len(y_tune), -1)
        bias = target_fit.mean(dim=0)
        centered = target_fit - bias
        gram = z_fit.T @ z_fit
        cross = z_fit.T @ centered
        scale = torch.trace(gram) / rank
        best = None
        for alpha in alphas:
            head = torch.linalg.solve(gram + torch.eye(rank, device=gram.device) * (scale * alpha), cross).T
            qhead, qscale = quantize_rows(head)
            prediction = z_tune @ (qhead.float() * qscale[:, None]).T + bias.half().float()
            error = (prediction - target_tune).square()
            relative = float(error.sum() / target_tune.square().sum().clamp_min(1e-30))
            row = {"alpha": alpha, "relative_mse": relative, "mse": float(error.mean())}
            if best is None or relative < best[0]:
                best = (relative, qhead.cpu(), qscale.half().cpu(), bias.half().cpu(), row)
            del head, qhead, qscale, prediction, error
        assert best is not None
        artifacts.append({"qhead": best[1], "scale": best[2], "bias": best[3], "rank": torch.tensor(rank)})
        tuning.append({"component_index": component_index, "rank": rank, **best[4]})
        del z_fit, z_tune, target_fit, target_tune, centered, gram, cross, bias
        gc.collect()
        torch.cuda.empty_cache()
    return artifacts, {"components": tuning}


def apply_direct(z: torch.Tensor, artifacts: list[dict[str, torch.Tensor]], experts: int, device: torch.device) -> torch.Tensor:
    outputs = []
    for artifact in artifacts:
        rank = int(artifact["rank"])
        qhead = artifact["qhead"].to(device)
        scale = artifact["scale"].to(device).float()
        bias = artifact["bias"].to(device).float()
        prediction = z[:, :rank] @ (qhead.float() * scale[:, None]).T + bias
        outputs.append(prediction.reshape(len(z), experts, UNITS))
        del qhead, scale, bias, prediction
    result = torch.stack(outputs, dim=-2)
    torch.cuda.empty_cache()
    return result


def run_layer(args: argparse.Namespace, layer: int, prior: Any, checkpoint_index: Mapping[str, str], trees: Mapping[str, Any], recorder: Recorder, device: torch.device) -> dict[str, Any]:
    started = time.time()
    exact_data = load_layer(args.exact_root, layer)
    parent_data = load_layer(args.parent_root, layer)
    parent_ids = identity_map(parent_data)
    rows = np.flatnonzero(exact_data["position"] >= HORIZON)
    exact_x = torch.tensor(exact_data["x"][rows], dtype=torch.float32, device=device)
    parent_x = torch.tensor(np.stack([
        parent_data["x"][parent_ids[(str(exact_data["request_id"][i]), int(exact_data["position"][i]), str(exact_data["prefix_hash"][i]))]]
        for i in rows
    ]), dtype=torch.float32, device=device)
    validation_np = np.asarray(exact_data["split"][rows]).astype(str) == "validation"
    validation = torch.from_numpy(validation_np).to(device)
    routed_experts = np.asarray(exact_data["expert_ids"][rows])[validation_np]
    active = sorted(set(routed_experts.ravel().tolist()))
    active_index = {expert: index for index, expert in enumerate(active)}
    routed_local = torch.tensor([[active_index[int(expert)] for expert in row] for row in routed_experts], dtype=torch.int64, device=device)
    recorder("layer_alignment", layer=layer, rows=len(rows), validation_rows=int(validation_np.sum()), active_experts=len(active))

    weights = {name: np.empty((len(active), UNITS, HIDDEN), np.float32) for name in ("gate2", "gate4", "up2", "up4")}
    for local_index, expert in enumerate(active):
        decoded = prior.decode_expert(args.checkpoint, checkpoint_index, trees, layer, expert)
        for name in ("gate", "up"):
            weights[name + "2"][local_index] = decoded[name][0]
            weights[name + "4"][local_index] = decoded[name][2]
        del decoded
        if local_index % 16 == 15 or local_index + 1 == len(active):
            recorder("decode_progress", layer=layer, completed=local_index + 1, total=len(active))
    delta_gate = weights["gate4"] - weights["gate2"]
    delta_up = weights["up4"] - weights["up2"]

    # Fixed request-disjoint HARP split.
    request_index = torch.load(args.fitting_root / "request_index.pt", map_location="cpu", weights_only=True)
    available = torch.load(args.fitting_root / "target_available.pt", map_location="cpu", weights_only=True)[:, 0, layer]
    generator = torch.Generator().manual_seed(SEED)
    request_order = torch.randperm(int(request_index.max()) + 1, generator=generator)
    fit_requests, tune_requests = request_order[:224], request_order[224:]
    fit_mask = available & torch.isin(request_index, fit_requests)
    tune_mask = available & torch.isin(request_index, tune_requests)
    parent_store = torch.load(args.fitting_root / "parent_target.pt", map_location="cpu", mmap=True, weights_only=True)
    exact_store = torch.load(args.fitting_root / "exact_target.pt", map_location="cpu", mmap=True, weights_only=True)
    fit_parent = parent_store[fit_mask, 0, layer].float().to(device)
    tune_parent = parent_store[tune_mask, 0, layer].float().to(device)
    fit_exact = exact_store[fit_mask, 0, layer].float().to(device)
    tune_exact = exact_store[tune_mask, 0, layer].float().to(device)
    del parent_store, exact_store

    mean = fit_parent.mean(dim=0)
    centered = fit_parent - mean
    _, singular, basis = torch.pca_lowrank(centered, q=136, center=False, niter=4)
    mean_q = mean.half().float()
    basis_q = basis.half().float()
    z_fit = (fit_parent - mean_q) @ basis_q
    z_tune = (tune_parent - mean_q) @ basis_q
    z_validation = (parent_x[validation] - mean_q) @ basis_q
    recorder("encoder_fitted", layer=layer, fit_rows=len(fit_parent), tune_rows=len(tune_parent), captured_variance=float(singular.square().sum() / centered.square().sum()))
    del centered, singular

    # Exact training/tuning response fields and resident-Q2 baselines.
    both_exact = torch.cat((fit_exact, tune_exact), dim=0)
    both_parent = torch.cat((fit_parent, tune_parent), dim=0)
    split_at = len(fit_exact)
    exact_proj = {name: project(weights[name], both_exact, device) for name in ("gate2", "gate4", "up2", "up4")}
    exact_components_all = components(hidden(exact_proj["gate2"], exact_proj["gate4"], exact_proj["up2"], exact_proj["up4"]))
    parent_g2_all = project(weights["gate2"], both_parent, device)
    parent_u2_all = project(weights["up2"], both_parent, device)
    parent_b_all = silu(parent_g2_all) * parent_u2_all
    fit_components, tune_components = exact_components_all[:split_at], exact_components_all[split_at:]
    fit_b_parent, tune_b_parent = parent_b_all[:split_at], parent_b_all[split_at:]
    del both_exact, both_parent, exact_proj, exact_components_all

    exact_val_proj = {name: project(weights[name], exact_x[validation], device) for name in ("gate2", "gate4", "up2", "up4")}
    truth_validation = hidden(exact_val_proj["gate2"], exact_val_proj["gate4"], exact_val_proj["up2"], exact_val_proj["up4"])
    parent_g2_val = project(weights["gate2"], parent_x[validation], device)
    parent_u2_val = project(weights["up2"], parent_x[validation], device)
    parent_b_val = silu(parent_g2_val) * parent_u2_val
    parent_control = hidden(parent_g2_val, project(weights["gate4"], parent_x[validation], device), parent_u2_val, project(weights["up4"], parent_x[validation], device))
    results: list[dict[str, Any]] = [{
        "name": "resident_q2_activation_exact_q4_control",
        "metrics": metric_rows(truth_validation, parent_control, routed_local),
        "payload": payload(0, 0, len(active)),
    }]

    def add_result(name: str, predicted_components: torch.Tensor, rank_sum: int, heads: int, tuning: Any = None) -> None:
        record = {
            "name": name,
            "metrics": metric_rows(truth_validation, hidden_from_components(predicted_components), routed_local),
            "payload": payload(rank_sum, heads, len(active)),
        }
        if tuning is not None:
            record["tuning"] = tuning
        results.append(record)
        recorder("candidate_completed", layer=layer, **record)

    # 1) Structured two-head operator baseline: rank 136 + rank 136.
    op136_val = operator_components(parent_x[validation], mean, basis, delta_gate, delta_up, 136, parent_g2_val, parent_u2_val, device)
    add_result("operator_gate_up_r136", op136_val, 272, 2)
    del op136_val

    # 2) Direct delta_g/delta_u/iota heads: rank 89 each; analytic resident b.
    art, tuning = fit_direct(z_fit, z_tune, fit_components[:, :, 1:], tune_components[:, :, 1:], [89, 89, 89], args.ridge_alphas)
    direct = apply_direct(z_validation, art, len(active), device)
    pred = torch.cat((parent_b_val[:, :, None], direct), dim=-2)
    add_result("direct_deltas_r89", pred, 267, 3, tuning)
    del art, direct, pred

    # 3) Four direct heads: b correction and three exact interaction components.
    fit_target4 = torch.cat(((fit_components[:, :, :1] - fit_b_parent[:, :, None]), fit_components[:, :, 1:]), dim=-2)
    tune_target4 = torch.cat(((tune_components[:, :, :1] - tune_b_parent[:, :, None]), tune_components[:, :, 1:]), dim=-2)
    art, tuning = fit_direct(z_fit, z_tune, fit_target4, tune_target4, [66, 66, 66, 66], args.ridge_alphas)
    direct = apply_direct(z_validation, art, len(active), device)
    direct[:, :, 0] += parent_b_val
    add_result("direct_bcorr_deltas_r66", direct, 264, 4, tuning)
    del fit_target4, tune_target4, art, direct

    # 4/5) Structured operator plus a small directly-fitted iota correction.
    for op_rank, correction_rank in ((120, 29), (128, 13)):
        fit_op = operator_components(fit_parent, mean, basis, delta_gate, delta_up, op_rank, parent_g2_all[:split_at], parent_u2_all[:split_at], device)
        tune_op = operator_components(tune_parent, mean, basis, delta_gate, delta_up, op_rank, parent_g2_all[split_at:], parent_u2_all[split_at:], device)
        val_op = operator_components(parent_x[validation], mean, basis, delta_gate, delta_up, op_rank, parent_g2_val, parent_u2_val, device)
        art, tuning = fit_direct(z_fit, z_tune, fit_components[:, :, 3:4] - fit_op[:, :, 3:4], tune_components[:, :, 3:4] - tune_op[:, :, 3:4], [correction_rank], args.ridge_alphas)
        correction = apply_direct(z_validation, art, len(active), device)
        val_op[:, :, 3:4] += correction
        add_result(f"operator_r{op_rank}_iota_correction_r{correction_rank}", val_op, 2 * op_rank + correction_rank, 3, tuning)
        del fit_op, tune_op, val_op, art, correction

    # 6) More balanced hybrid: rank-96 operator plus rank-23 corrections to all deltas.
    fit_op = operator_components(fit_parent, mean, basis, delta_gate, delta_up, 96, parent_g2_all[:split_at], parent_u2_all[:split_at], device)
    tune_op = operator_components(tune_parent, mean, basis, delta_gate, delta_up, 96, parent_g2_all[split_at:], parent_u2_all[split_at:], device)
    val_op = operator_components(parent_x[validation], mean, basis, delta_gate, delta_up, 96, parent_g2_val, parent_u2_val, device)
    art, tuning = fit_direct(z_fit, z_tune, fit_components[:, :, 1:] - fit_op[:, :, 1:], tune_components[:, :, 1:] - tune_op[:, :, 1:], [23, 23, 23], args.ridge_alphas)
    correction = apply_direct(z_validation, art, len(active), device)
    val_op[:, :, 1:] += correction
    add_result("operator_r96_delta_corrections_r23", val_op, 261, 5, tuning)
    del fit_op, tune_op, val_op, art, correction

    result = {
        "layer": layer,
        "fit_rows": int(fit_mask.sum()),
        "tune_rows": int(tune_mask.sum()),
        "validation_rows": int(validation_np.sum()),
        "active_validation_experts": len(active),
        "parent_activation_mse_pr13": float((parent_x[validation] - exact_x[validation]).square().mean()),
        "candidates": results,
        "seconds": time.time() - started,
    }
    recorder("layer_completed", **result)
    del fit_parent, tune_parent, fit_exact, tune_exact, fit_components, tune_components, fit_b_parent, tune_b_parent
    del parent_g2_all, parent_u2_all, parent_b_all, truth_validation, parent_g2_val, parent_u2_val, parent_b_val
    gc.collect()
    torch.cuda.empty_cache()
    return result


def aggregate(layers: list[dict[str, Any]]) -> dict[str, Any]:
    names = [candidate["name"] for candidate in layers[0]["candidates"]]
    output = []
    for name in names:
        records = [next(candidate for candidate in layer["candidates"] if candidate["name"] == name) for layer in layers]
        metrics = []
        for component in ("b", "delta_g", "delta_u", "iota"):
            rows = [next(metric for metric in record["metrics"] if metric["component"] == component) for record in records]
            metrics.append({
                "component": component,
                "cosine_mean": float(np.mean([row["cosine"] for row in rows])),
                "mse_mean": float(np.mean([row["mse"] for row in rows])),
                "relative_mse_mean": float(np.mean([row["relative_mse"] for row in rows])),
            })
        output.append({"name": name, "payload": records[0]["payload"], "metrics": metrics})
    return {"candidates": output}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default="14")
    parser.add_argument("--exact-root", type=Path, default=Path("/workspace/all_layer_split_interaction_study/capture"))
    parser.add_argument("--parent-root", type=Path, default=Path("/workspace/embedded_parent_capture_20260821_v1"))
    parser.add_argument("--fitting-root", type=Path, default=Path("/workspace/harp_fitting_mxfp4_parent_pairs_v1"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/workspace/pr13_average_rate_all_layers_20260822_v1/inputs/checkpoint"))
    parser.add_argument("--trees", type=Path, default=Path("/workspace/codebook_granularity_study/locked/selected_trees.json"))
    parser.add_argument("--oracle-root", type=Path, default=Path("/workspace/set_utility_study/experiments/adaptive_expert_precision_oracle"))
    parser.add_argument("--h4-helper", type=Path, default=Path("/workspace/pr13_h4_response_field_pilot_run.py"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge-alphas", default="1e-4,1e-3,1e-2,1e-1")
    args = parser.parse_args()
    args.ridge_alphas = tuple(float(value) for value in args.ridge_alphas.split(","))
    layers = tuple(int(value) for value in args.layers.split(","))
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    recorder = Recorder(args.output / "run.jsonl")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    source, scripts = args.oracle_root / "src", args.oracle_root / "scripts"
    sys.path[:0] = [str(source), str(scripts)]
    import run_set_utility_distillation as prior
    from run_mxfp4_selective_pages import tree_from_record
    load_h4_helpers(args.h4_helper)
    checkpoint_index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    records = json.loads(args.trees.read_text())
    trees = {name: tree_from_record(records[name]) for name in ("gate", "up", "down")}
    recorder("pilot_started", layers=layers, gpu=torch.cuda.get_device_name(0), budget_bpw=0.55, ridge_alphas=args.ridge_alphas)
    layer_results = []
    for layer in layers:
        layer_results.append(run_layer(args, layer, prior, checkpoint_index, trees, recorder, device))
    result = {"schema": "pr13_h1_low_bpw_response_pilot_v1", "budget_bpw": 0.55, "layers": layer_results, "aggregate": aggregate(layer_results)}
    atomic_json(args.output / "result.json", result)
    recorder("pilot_completed", result=str(args.output / "result.json"))


if __name__ == "__main__":
    main()
