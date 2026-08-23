#!/usr/bin/env python3
"""Screen efficient Q4-Q2 response projections on all 40 causal H1 layers."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


LAYERS = tuple(range(40))
HIDDEN = 2048
UNITS = 512
HORIZON = 1
TAIL_RANK = 16


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
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
    spec = importlib.util.spec_from_file_location("pr13_h1_four_layer_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_rows(root: Path, layer: int) -> dict[str, np.ndarray]:
    pieces: dict[str, list[np.ndarray]] = {}
    for split in ("train", "validation"):
        with np.load(root / f"{split}_layer_{layer:02d}.npz", allow_pickle=False) as loaded:
            for name in loaded.files:
                pieces.setdefault(name, []).append(np.asarray(loaded[name]))
    return {name: np.concatenate(values, axis=0) for name, values in pieces.items()}


def assert_alignment(exact: dict[str, np.ndarray], parent: dict[str, np.ndarray], layer: int) -> None:
    for name in ("request_id", "position", "prefix_hash", "split", "layer"):
        if not np.array_equal(exact[name], parent[name]):
            raise RuntimeError(f"layer {layer} parent/exact {name} alignment changed")


def masked_mean(value: np.ndarray, mask: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    count = mask.sum(axis=-1)
    total = np.where(mask, value, 0.0).sum(axis=-1, dtype=np.float32)
    return np.where(count > 0, total / np.maximum(count, 1), fallback).astype(np.float32)


def binary_quantize(delta: np.ndarray, block: int) -> tuple[np.ndarray, dict[str, torch.Tensor]]:
    grouped = delta.reshape(len(delta), UNITS, HIDDEN // block, block)
    sign = grouped >= 0
    magnitude = np.abs(grouped)
    overall = magnitude.mean(axis=-1, dtype=np.float32)
    positive = masked_mean(magnitude, sign, overall)
    negative = masked_mean(magnitude, ~sign, overall)
    reconstructed = np.where(sign, positive[..., None], -negative[..., None]).reshape(delta.shape).astype(np.float32)
    packed = np.packbits(sign.reshape(len(delta), UNITS, HIDDEN), axis=-1, bitorder="little")
    return reconstructed, {
        "sign_bits": torch.from_numpy(packed.copy()),
        "scale_fp16": torch.from_numpy(np.stack((negative, positive), axis=-1)).half(),
    }


def pack_two_bit(code: np.ndarray) -> np.ndarray:
    local = code.reshape(len(code), UNITS, HIDDEN // 4, 4).astype(np.uint8, copy=False)
    return (
        local[..., 0]
        | (local[..., 1] << 2)
        | (local[..., 2] << 4)
        | (local[..., 3] << 6)
    )


def two_bit_quantize(
    delta: np.ndarray,
    block: int,
    asymmetric: bool,
) -> tuple[np.ndarray, dict[str, torch.Tensor]]:
    grouped = delta.reshape(len(delta), UNITS, HIDDEN // block, block)
    sign = grouped >= 0
    magnitude = np.abs(grouped)
    overall = magnitude.mean(axis=-1, dtype=np.float32)
    if asymmetric:
        positive_mean = masked_mean(magnitude, sign, overall)
        negative_mean = masked_mean(magnitude, ~sign, overall)
        positive_high = sign & (magnitude >= positive_mean[..., None])
        negative_high = (~sign) & (magnitude >= negative_mean[..., None])
        positive_low = sign & ~positive_high
        negative_low = (~sign) & ~negative_high
        neg_hi = masked_mean(magnitude, negative_high, negative_mean)
        neg_lo = masked_mean(magnitude, negative_low, negative_mean)
        pos_lo = masked_mean(magnitude, positive_low, positive_mean)
        pos_hi = masked_mean(magnitude, positive_high, positive_mean)
        levels = np.stack((neg_hi, neg_lo, pos_lo, pos_hi), axis=-1)
    else:
        high = magnitude >= overall[..., None]
        low_level = masked_mean(magnitude, ~high, overall)
        high_level = masked_mean(magnitude, high, overall)
        neg_hi, neg_lo, pos_lo, pos_hi = high_level, low_level, low_level, high_level
        levels = np.stack((low_level, high_level), axis=-1)
        positive_high = sign & high
        negative_high = (~sign) & high
    code = np.where(sign, 2, 1).astype(np.uint8)
    code[positive_high] = 3
    code[negative_high] = 0
    reconstructed = np.choose(
        code,
        (-neg_hi[..., None], -neg_lo[..., None], pos_lo[..., None], pos_hi[..., None]),
    ).reshape(delta.shape).astype(np.float32)
    return reconstructed, {
        "code_2bit": torch.from_numpy(pack_two_bit(code).copy()),
        "level_fp16": torch.from_numpy(levels).half(),
    }


def tensor_bytes(value: Any) -> int:
    return value.numel() * value.element_size() if isinstance(value, torch.Tensor) else 0


def save_prediction(
    layer: int,
    variant: str,
    prediction: torch.Tensor,
    model: dict[str, Any],
    exact_hidden: torch.Tensor,
    validation: torch.Tensor,
    routed: torch.Tensor,
    helper: Any,
    output: Path,
    streamed_per_expert: int,
) -> dict[str, Any]:
    prediction_path = output / f"prediction_layer_{layer}_{variant}.pt"
    model_path = output / f"model_layer_{layer}_{variant}.pt"
    atomic_save(prediction.half().cpu(), prediction_path)
    atomic_save(model, model_path)
    metrics = helper.routed_metrics(exact_hidden[validation], prediction[validation], routed)
    return {
        "variant": variant,
        "metrics": metrics,
        "streamed_bytes_per_expert": streamed_per_expert,
        "streamed_bytes_top8": 8 * streamed_per_expert,
        "prediction": str(prediction_path),
        "prediction_sha256": sha256(prediction_path),
        "model": str(model_path),
        "model_sha256": sha256(model_path),
    }


def build_layer(
    layer: int,
    exact_root: Path,
    parent_root: Path,
    checkpoint: Path,
    checkpoint_index: dict[str, str],
    trees: dict[str, Any],
    output: Path,
    helper: Any,
    prior: Any,
    device: torch.device,
    recorder: Recorder,
) -> dict[str, Any]:
    started = time.time()
    exact = load_rows(exact_root, layer)
    parent = load_rows(parent_root, layer)
    assert_alignment(exact, parent, layer)
    admitted = exact["position"] >= HORIZON
    if int(admitted.sum()) != 93:
        raise RuntimeError(f"layer {layer} H1 rows changed: {admitted.sum()}")
    split = exact["split"][admitted]
    validation_np = split == "validation"
    if int(validation_np.sum()) != 29 or int((~validation_np).sum()) != 64:
        raise RuntimeError(f"layer {layer} H1 split changed")
    exact_x = torch.from_numpy(exact["x"][admitted]).float().to(device)
    parent_x = torch.from_numpy(parent["x"][admitted]).float().to(device)
    expert_ids = exact["expert_ids"][admitted]
    active = sorted(set(expert_ids[validation_np].ravel().tolist()))
    active_index = {expert: index for index, expert in enumerate(active)}
    routed = torch.tensor(
        [[active_index[int(expert)] for expert in row] for row in expert_ids[validation_np]],
        dtype=torch.int64,
        device=device,
    )
    validation = torch.from_numpy(validation_np).to(device)
    recorder("layer_alignment", layer=layer, rows=93, train=64, validation=29, active_experts=len(active))

    gate2 = np.empty((len(active), UNITS, HIDDEN), np.float32)
    up2 = np.empty_like(gate2)
    deltas = {name: np.empty_like(gate2) for name in ("gate", "up")}
    for index, expert in enumerate(active):
        decoded = prior.decode_expert(checkpoint, checkpoint_index, trees, layer, int(expert))
        gate2[index] = decoded["gate"][0]
        up2[index] = decoded["up"][0]
        deltas["gate"][index] = decoded["gate"][2] - decoded["gate"][0]
        deltas["up"][index] = decoded["up"][2] - decoded["up"][0]
        del decoded
    recorder("weights_decoded", layer=layer, active_experts=len(active))

    g2_exact = helper.project(gate2, exact_x, device)
    u2_exact = helper.project(up2, exact_x, device)
    dg_exact = helper.project(deltas["gate"], exact_x, device)
    du_exact = helper.project(deltas["up"], exact_x, device)
    exact_hidden = helper.hidden(g2_exact, g2_exact + dg_exact, u2_exact, u2_exact + du_exact)
    g2_parent = helper.project(gate2, parent_x, device)
    u2_parent = helper.project(up2, parent_x, device)
    exact_parent_residual = {
        name: helper.project(delta, parent_x, device) for name, delta in deltas.items()
    }

    variants: list[dict[str, Any]] = []
    binary_predictions: dict[int, dict[str, torch.Tensor]] = {512: {}, 256: {}}
    binary_recon_512: dict[str, np.ndarray] = {}
    binary_model_512: dict[str, Any] = {
        "schema": "pr13_h1_all_layer_binary_asymmetric_v1",
        "layer": layer,
        "block_size": 512,
        "active_experts": torch.tensor(active, dtype=torch.int16),
    }
    for block in (512, 256):
        artifact: dict[str, Any] = {
            "schema": "pr13_h1_all_layer_binary_asymmetric_v1",
            "layer": layer,
            "block_size": block,
            "active_experts": torch.tensor(active, dtype=torch.int16),
        }
        for name in ("gate", "up"):
            reconstructed, packed = binary_quantize(deltas[name], block)
            binary_predictions[block][name] = helper.project(reconstructed, parent_x, device)
            artifact[f"{name}_sign_bits"] = packed["sign_bits"]
            artifact[f"{name}_scale_fp16"] = packed["scale_fp16"]
            if block == 512:
                binary_recon_512[name] = reconstructed
                binary_model_512[f"{name}_sign_bits"] = packed["sign_bits"]
                binary_model_512[f"{name}_scale_fp16"] = packed["scale_fp16"]
        prediction = helper.hidden(
            g2_parent, g2_parent + binary_predictions[block]["gate"],
            u2_parent, u2_parent + binary_predictions[block]["up"],
        )
        streamed = sum(tensor_bytes(value) for key, value in artifact.items() if key.endswith("sign_bits") or key.endswith("scale_fp16"))
        record = save_prediction(
            layer, f"b1_asym_b{block}", prediction, artifact, exact_hidden,
            validation, routed, helper, output, streamed // len(active),
        )
        variants.append(record)
        recorder("variant_completed", layer=layer, **record)
        del prediction, artifact
        gc.collect()

    train_parent = parent_x[~validation]
    mean = train_parent.mean(0)
    centered = train_parent - mean
    _, _, basis = torch.pca_lowrank(centered, q=TAIL_RANK, center=False, niter=5)
    mean_q = mean.half().float()
    basis_q = basis.half().float()
    z = (parent_x - mean_q) @ basis_q
    tail_model: dict[str, Any] = {
        **binary_model_512,
        "schema": "pr13_h1_all_layer_binary_asymmetric_tail_v1",
        "tail_rank": TAIL_RANK,
        "mean_fp16": mean_q.half().cpu(),
        "basis_fp16": basis_q.half().cpu(),
    }
    tail_predictions: dict[str, torch.Tensor] = {}
    for name in ("gate", "up"):
        tail = torch.from_numpy(deltas[name] - binary_recon_512[name]).to(device).reshape(-1, HIDDEN)
        head = tail @ basis_q
        bias = tail @ mean_q
        scale = (head.abs().amax(-1) / 127.0).clamp_min(1e-12)
        qhead = torch.round(head / scale[:, None]).clamp(-127, 127).to(torch.int8)
        scale_q = scale.half().float()
        bias_q = bias.half().float()
        tail_prediction = z @ (qhead.float() * scale_q[:, None]).T + bias_q
        tail_predictions[name] = binary_predictions[512][name] + tail_prediction.reshape(len(parent_x), len(active), UNITS)
        tail_model[f"{name}_tail_head_int8"] = qhead.reshape(len(active), UNITS, TAIL_RANK).cpu()
        tail_model[f"{name}_tail_scale_fp16"] = scale_q.reshape(len(active), UNITS).half().cpu()
        tail_model[f"{name}_tail_bias_fp16"] = bias_q.reshape(len(active), UNITS).half().cpu()
        del tail, head, bias, scale, qhead, scale_q, bias_q, tail_prediction
    prediction = helper.hidden(
        g2_parent, g2_parent + tail_predictions["gate"],
        u2_parent, u2_parent + tail_predictions["up"],
    )
    streamed_keys = ("sign_bits", "scale_fp16", "head_int8", "bias_fp16")
    streamed = sum(
        tensor_bytes(value) for key, value in tail_model.items()
        if any(token in key for token in streamed_keys) and key not in {"mean_fp16", "basis_fp16"}
    )
    record = save_prediction(
        layer, "b1_asym_b512_tail16", prediction, tail_model, exact_hidden,
        validation, routed, helper, output, streamed // len(active),
    )
    record["shared_resident_bytes"] = tensor_bytes(tail_model["mean_fp16"]) + tensor_bytes(tail_model["basis_fp16"])
    variants.append(record)
    recorder("variant_completed", layer=layer, **record)
    del prediction, tail_predictions, tail_model, centered, z, basis, mean, mean_q, basis_q
    gc.collect()
    torch.cuda.empty_cache()

    for asymmetric in (False, True):
        artifact = {
            "schema": "pr13_h1_all_layer_two_bit_residual_v1",
            "layer": layer,
            "block_size": 512,
            "asymmetric_levels": asymmetric,
            "active_experts": torch.tensor(active, dtype=torch.int16),
        }
        residuals = {}
        for name in ("gate", "up"):
            reconstructed, packed = two_bit_quantize(deltas[name], 512, asymmetric)
            residuals[name] = helper.project(reconstructed, parent_x, device)
            artifact[f"{name}_code_2bit"] = packed["code_2bit"]
            artifact[f"{name}_level_fp16"] = packed["level_fp16"]
            del reconstructed
        prediction = helper.hidden(
            g2_parent, g2_parent + residuals["gate"],
            u2_parent, u2_parent + residuals["up"],
        )
        streamed = sum(tensor_bytes(value) for key, value in artifact.items() if key.endswith("code_2bit") or key.endswith("level_fp16"))
        variant = "q2_sym_b512" if not asymmetric else "q2_asym_b512"
        record = save_prediction(
            layer, variant, prediction, artifact, exact_hidden,
            validation, routed, helper, output, streamed // len(active),
        )
        variants.append(record)
        recorder("variant_completed", layer=layer, **record)
        del prediction, artifact, residuals
        gc.collect()
        torch.cuda.empty_cache()

    dataset = {
        "schema": "pr13_h1_all_layer_response_dataset_v1",
        "layer": layer,
        "horizon": HORIZON,
        "request_id": exact["request_id"][admitted].tolist(),
        "split": split.tolist(),
        "target_position": torch.from_numpy(exact["position"][admitted].copy()),
        "source_position": torch.from_numpy(exact["position"][admitted].copy()) - HORIZON,
        "prefix_hash": exact["prefix_hash"][admitted].tolist(),
        "expert_ids": torch.from_numpy(expert_ids.copy()),
        "router_weights": torch.from_numpy(exact["router_weights"][admitted].copy()).float(),
        "active_experts": active,
        "exact_hidden_compact": exact_hidden.float().cpu(),
        "exact_target_activation_auxiliary": exact_x.half().cpu(),
        "parent_target_x": parent_x.half().cpu(),
        "variant_predictions": {record["variant"]: record["prediction"] for record in variants},
    }
    dataset_path = output / f"dataset_layer_{layer}.pt"
    atomic_save(dataset, dataset_path)
    diagnostics = {}
    for name in ("gate", "up"):
        target = exact_parent_residual[name]
        for block in (512, 256):
            error = (binary_predictions[block][name] - target).square()
            diagnostics[f"b1_b{block}_{name}_relative_mse_parent"] = float(error.sum() / target.square().sum().clamp_min(1e-30))
    result = {
        "layer": layer,
        "rows": 93,
        "train_rows": 64,
        "validation_rows": 29,
        "active_validation_experts": len(active),
        "parent_activation_mse": float((parent_x - exact_x).square().mean()),
        "variants": variants,
        "dataset": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
        "diagnostics": diagnostics,
        "seconds": time.time() - started,
    }
    recorder("layer_completed", **result)
    del exact_hidden, exact_x, parent_x, gate2, up2, deltas, g2_exact, u2_exact, dg_exact, du_exact
    del g2_parent, u2_parent, exact_parent_residual, binary_predictions, binary_recon_512, dataset
    gc.collect()
    torch.cuda.empty_cache()
    return result


def aggregate(layers: list[dict[str, Any]]) -> dict[str, Any]:
    variants = [record["variant"] for record in layers[0]["variants"]]
    result = []
    for variant in variants:
        records = [next(item for item in layer["variants"] if item["variant"] == variant) for layer in layers]
        components = []
        for component in ("b", "delta_g", "delta_u", "iota"):
            rows = [next(item for item in record["metrics"] if item["component"] == component) for record in records]
            components.append({
                "component": component,
                "cosine_mean": float(np.mean([row["cosine"] for row in rows])),
                "mse_mean": float(np.mean([row["mse"] for row in rows])),
                "relative_mse_mean": float(np.mean([row["relative_mse"] for row in rows])),
                "cosine_p10_across_layers": float(np.quantile([row["cosine"] for row in rows], 0.10)),
            })
        result.append({
            "variant": variant,
            "components": components,
            "streamed_bytes_per_expert": records[0]["streamed_bytes_per_expert"],
            "streamed_bytes_top8_per_layer": records[0]["streamed_bytes_top8"],
            "shared_resident_bytes_per_layer": records[0].get("shared_resident_bytes", 0),
        })
    return {"variants": result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact-root", type=Path, default=Path("/workspace/all_layer_split_interaction_study/capture"))
    parser.add_argument("--parent-root", type=Path, default=Path("/workspace/embedded_parent_capture_20260821_v1"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/workspace/pr13_average_rate_all_layers_20260822_v1/inputs/checkpoint"))
    parser.add_argument("--trees", type=Path, default=Path("/workspace/codebook_granularity_study/locked/selected_trees.json"))
    parser.add_argument("--oracle-root", type=Path, default=Path("/workspace/set_utility_study/experiments/adaptive_expert_precision_oracle"))
    parser.add_argument("--helper", type=Path, default=Path("/workspace/pr13_h1_four_layer_helpers.py"))
    parser.add_argument("--output", type=Path, default=Path("/workspace/pr13_h1_all_layers_predictor_20260822_v1"))
    parser.add_argument("--layers", default="0-39")
    args = parser.parse_args()
    if args.output.exists() and not (args.output / "run.jsonl").exists():
        raise FileExistsError(f"refusing ambiguous output {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    recorder = Recorder(args.output / "run.jsonl")
    if args.layers == "0-39":
        layers = LAYERS
    else:
        layers = tuple(int(value) for value in args.layers.split(","))
    sys.path[:0] = [str(args.oracle_root / "src"), str(args.oracle_root / "scripts")]
    import run_set_utility_distillation as prior
    from run_mxfp4_selective_pages import tree_from_record
    helper = load_module(args.helper)
    checkpoint_index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    tree_records = json.loads(args.trees.read_text())
    trees = {name: tree_from_record(tree_records[name]) for name in ("gate", "up", "down")}
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    existing = {}
    summary_path = args.output / "screen_result.json"
    if summary_path.exists():
        existing = {int(item["layer"]): item for item in json.loads(summary_path.read_text())["layers"]}
    recorder("screen_started", layers=list(layers), existing_layers=sorted(existing), h1=True)
    for layer in layers:
        if layer in existing:
            continue
        existing[layer] = build_layer(
            layer, args.exact_root, args.parent_root, args.checkpoint,
            checkpoint_index, trees, args.output, helper, prior, device, recorder,
        )
        payload = {
            "schema": "pr13_h1_all_layer_response_screen_v1",
            "layers": [existing[index] for index in sorted(existing)],
        }
        payload["aggregate"] = aggregate(payload["layers"])
        atomic_json(summary_path, payload)
    recorder("screen_completed", layers=sorted(existing), result_sha256=sha256(summary_path))


if __name__ == "__main__":
    main()
