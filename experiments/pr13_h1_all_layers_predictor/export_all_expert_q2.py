#!/usr/bin/env python3
"""Compile route-independent all-256-expert asymmetric 2-bit residuals."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

HIDDEN = 2048
UNITS = 512
EXPERTS = 256
BLOCK = 512


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def atomic_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_bytes(value: torch.Tensor) -> int:
    return value.numel() * value.element_size()


def unpack_code(packed: torch.Tensor) -> np.ndarray:
    value = packed.numpy()
    code = np.stack(tuple((value >> shift) & 3 for shift in (0, 2, 4, 6)), axis=-1)
    return code.reshape(*value.shape[:-1], HIDDEN).astype(np.uint8)


def export_layer(
    layer: int,
    args: argparse.Namespace,
    checkpoint_index: dict[str, str],
    trees: dict[str, Any],
    screen: Any,
    calibration: Any,
    prior: Any,
    device: torch.device,
) -> dict[str, Any]:
    dataset = torch.load(
        args.experiment / f"dataset_layer_{layer}.pt",
        map_location="cpu", weights_only=False,
    )
    parent_x = dataset["parent_target_x"].float().to(device)
    exact_x = dataset["exact_target_activation_auxiliary"].float().to(device)
    train = torch.tensor(
        [value == "train" for value in dataset["split"]],
        dtype=torch.bool, device=device,
    )
    evaluated_variant = (
        "q2_asym_b512_actcal_a1p0" if args.calibrated else "q2_asym_b512"
    )
    evaluated = torch.load(
        args.experiment / f"model_layer_{layer}_{evaluated_variant}.pt",
        map_location="cpu", weights_only=False,
    )
    evaluated_experts = evaluated["active_experts"].long()
    codes = {name: [] for name in ("gate", "up")}
    tables = {name: [] for name in ("gate", "up")}
    for begin in range(0, EXPERTS, args.chunk_experts):
        end = min(begin + args.chunk_experts, EXPERTS)
        deltas = {
            name: np.empty((end - begin, UNITS, HIDDEN), np.float32)
            for name in ("gate", "up")
        }
        for local, expert in enumerate(range(begin, end)):
            decoded = prior.decode_expert(
                args.checkpoint, checkpoint_index, trees, layer, expert
            )
            deltas["gate"][local] = decoded["gate"][2] - decoded["gate"][0]
            deltas["up"][local] = decoded["up"][2] - decoded["up"][0]
        for name in ("gate", "up"):
            _, packed = screen.two_bit_quantize(deltas[name], BLOCK, True)
            code_packed = packed["code_2bit"].contiguous()
            initial = packed["level_fp16"].float().numpy()
            if args.calibrated:
                signed = initial.copy()
                signed[..., :2] *= -1.0
                _, learned = calibration.calibrated_projection(
                    parent_x, exact_x, deltas[name], unpack_code(code_packed),
                    signed, train, (args.alpha,), end - begin, device,
                )
                table = learned[args.alpha].contiguous()
            else:
                table = packed["level_fp16"].contiguous()
            codes[name].append(code_packed)
            tables[name].append(table)
        del deltas
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({
            "event": "export_chunk", "layer": layer,
            "begin": begin, "end": end, "calibrated": args.calibrated,
        }), flush=True)
    artifact: dict[str, Any] = {
        "schema": (
            "pr13_h1_all_expert_activation_calibrated_q2_residual_v1"
            if args.calibrated
            else "pr13_h1_all_expert_q2_residual_v1"
        ),
        "layer": layer,
        "horizon": 1,
        "block_size": BLOCK,
        "asymmetric_levels": True,
        "experts": torch.arange(EXPERTS, dtype=torch.int16),
        "resident_parent": "embedded_parent_mxfp4_all_layer_sharded_v1",
        "calibration_rows": int(train.sum()) if args.calibrated else 0,
        "ridge_alpha": args.alpha if args.calibrated else None,
    }
    for name in ("gate", "up"):
        artifact[f"{name}_code_2bit"] = torch.cat(codes[name], dim=0)
        key = f"{name}_signed_level_fp16" if args.calibrated else f"{name}_level_fp16"
        artifact[key] = torch.cat(tables[name], dim=0)
        evaluated_key = key
        selected = evaluated_experts
        if not torch.equal(
            artifact[f"{name}_code_2bit"][selected],
            evaluated[f"{name}_code_2bit"],
        ):
            raise RuntimeError(f"layer {layer} {name} code active-subset audit failed")
        if not torch.equal(artifact[key][selected], evaluated[evaluated_key]):
            raise RuntimeError(f"layer {layer} {name} table active-subset audit failed")
    payload_keys = [
        key for key, value in artifact.items()
        if isinstance(value, torch.Tensor) and (
            key.endswith("code_2bit") or key.endswith("level_fp16")
        )
    ]
    payload_bytes = sum(tensor_bytes(artifact[key]) for key in payload_keys)
    expected = 557056 * EXPERTS
    if payload_bytes != expected:
        raise RuntimeError(f"layer {layer} payload changed: {payload_bytes}/{expected}")
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"all_expert_q2_layer_{layer}.pt"
    atomic_save(artifact, path)
    result = {
        "layer": layer,
        "path": str(path),
        "sha256": sha256(path),
        "payload_bytes": payload_bytes,
        "bytes_per_expert": payload_bytes // EXPERTS,
        "top8_streamed_bytes": 8 * payload_bytes // EXPERTS,
        "active_subset_experts_audited": len(evaluated_experts),
        "active_subset_bit_exact": True,
    }
    print(json.dumps({"event": "export_layer_completed", **result}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrated", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--layers", default="0-39")
    parser.add_argument("--chunk-experts", type=int, default=4)
    parser.add_argument(
        "--experiment", type=Path,
        default=Path("/workspace/pr13_h1_all_layers_predictor_20260822_v1"),
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("/workspace/pr13_average_rate_all_layers_20260822_v1/inputs/checkpoint"),
    )
    parser.add_argument(
        "--trees", type=Path,
        default=Path("/workspace/codebook_granularity_study/locked/selected_trees.json"),
    )
    parser.add_argument(
        "--oracle-root", type=Path,
        default=Path("/workspace/set_utility_study/experiments/adaptive_expert_precision_oracle"),
    )
    parser.add_argument(
        "--screen", type=Path,
        default=Path("/workspace/pr13_h1_run_all_layer_screen.py"),
    )
    parser.add_argument(
        "--calibration", type=Path,
        default=Path("/workspace/pr13_h1_run_activation_calibrated_screen.py"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("/workspace/pr13_h1_all_layers_predictor_20260822_v1/final_all_expert_q2"),
    )
    args = parser.parse_args()
    layers = tuple(range(40)) if args.layers == "0-39" else tuple(
        int(value) for value in args.layers.split(",")
    )
    device = torch.device("cuda")
    screen = load_module("pr13_h1_export_screen", args.screen)
    calibration = load_module("pr13_h1_export_calibration", args.calibration)
    sys.path[:0] = [str(args.oracle_root / "src"), str(args.oracle_root / "scripts")]
    import run_set_utility_distillation as prior
    from run_mxfp4_selective_pages import tree_from_record
    checkpoint_index = json.loads(
        (args.checkpoint / "model.safetensors.index.json").read_text()
    )["weight_map"]
    records = json.loads(args.trees.read_text())
    trees = {name: tree_from_record(records[name]) for name in ("gate", "up", "down")}
    completed = {
        int(path.stem.split("_")[-1]): path
        for path in args.output.glob("all_expert_q2_layer_*.pt")
    }
    outputs = []
    for layer in layers:
        if layer in completed:
            path = completed[layer]
            artifact = torch.load(path, map_location="cpu", weights_only=False)
            payload = sum(
                tensor_bytes(value) for key, value in artifact.items()
                if isinstance(value, torch.Tensor) and (
                    key.endswith("code_2bit") or key.endswith("level_fp16")
                )
            )
            outputs.append({
                "layer": layer, "path": str(path), "sha256": sha256(path),
                "payload_bytes": payload, "bytes_per_expert": payload // EXPERTS,
                "top8_streamed_bytes": 8 * payload // EXPERTS,
                "resumed": True,
            })
            continue
        outputs.append(export_layer(
            layer, args, checkpoint_index, trees, screen, calibration, prior, device
        ))
        atomic_json(args.output / "manifest.json", {
            "schema": "pr13_h1_all_expert_q2_manifest_v1",
            "calibrated": args.calibrated,
            "ridge_alpha": args.alpha if args.calibrated else None,
            "layers": outputs,
            "total_payload_bytes": sum(item["payload_bytes"] for item in outputs),
        })
    manifest = {
        "schema": "pr13_h1_all_expert_q2_manifest_v1",
        "calibrated": args.calibrated,
        "ridge_alpha": args.alpha if args.calibrated else None,
        "layers": outputs,
        "total_payload_bytes": sum(item["payload_bytes"] for item in outputs),
        "bytes_per_expert_per_layer": 557056,
        "top8_streamed_bytes_per_layer": 4456448,
        "shared_resident_bytes_per_layer": 0,
    }
    atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps({"event": "export_completed", **manifest}), flush=True)


if __name__ == "__main__":
    main()
