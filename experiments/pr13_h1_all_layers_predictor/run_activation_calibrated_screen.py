#!/usr/bin/env python3
"""Activation-calibrate fixed 1/2-bit Q4-Q2 residual codes.

Only the tiny per-block lookup tables are refit, so deployment traffic and
matvec structure remain unchanged. Exact future activations are calibration
targets only; inference consumes the causal resident-Q2 H1 state.
"""

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unpack_binary(model: dict[str, Any], name: str) -> tuple[np.ndarray, np.ndarray]:
    packed = model[f"{name}_sign_bits"].numpy()
    code = np.unpackbits(packed, axis=-1, count=HIDDEN, bitorder="little").astype(np.uint8)
    signed = model[f"{name}_scale_fp16"].float().numpy()
    signed[..., 0] *= -1.0
    return code, signed


def unpack_two_bit(model: dict[str, Any], name: str) -> tuple[np.ndarray, np.ndarray]:
    packed = model[f"{name}_code_2bit"].numpy()
    code = np.stack(tuple((packed >> shift) & 3 for shift in (0, 2, 4, 6)), axis=-1)
    code = code.reshape(*packed.shape[:-1], HIDDEN).astype(np.uint8)
    signed = model[f"{name}_level_fp16"].float().numpy()
    signed[..., :2] *= -1.0
    return code, signed


def code_features(x: torch.Tensor, code: torch.Tensor, levels: int) -> torch.Tensor:
    """Return [T,E,U,B*K] lookup-sum features for an expert chunk."""
    blocks = HIDDEN // BLOCK
    x_block = x.reshape(len(x), blocks, BLOCK)
    code_block = code.reshape(*code.shape[:-1], blocks, BLOCK)
    columns = []
    for level in range(levels):
        mask = (code_block == level).to(x.dtype)
        columns.append(torch.einsum("tbd,eubd->teub", x_block, mask))
    return torch.stack(columns, dim=-1).reshape(len(x), len(code), UNITS, blocks * levels)


@torch.inference_mode()
def calibrated_projection(
    parent_x: torch.Tensor,
    exact_x: torch.Tensor,
    delta: np.ndarray,
    code: np.ndarray,
    prior_levels: np.ndarray,
    train: torch.Tensor,
    alphas: tuple[float, ...],
    chunk_experts: int,
    device: torch.device,
) -> tuple[dict[float, torch.Tensor], dict[float, torch.Tensor]]:
    outputs = {
        alpha: torch.empty((len(parent_x), len(delta), UNITS), dtype=torch.float32)
        for alpha in alphas
    }
    learned = {
        alpha: torch.empty(prior_levels.shape, dtype=torch.float16)
        for alpha in alphas
    }
    levels = prior_levels.shape[-1]
    width = (HIDDEN // BLOCK) * levels
    eye = torch.eye(width, dtype=torch.float32, device=device)
    train_parent = parent_x[train]
    train_exact = exact_x[train]
    train_rows = int(train.sum())
    for begin in range(0, len(delta), chunk_experts):
        end = min(begin + chunk_experts, len(delta))
        local_code = torch.from_numpy(code[begin:end]).to(device)
        train_feature = code_features(train_parent, local_code, levels)
        all_feature = code_features(parent_x, local_code, levels)
        target = torch.einsum(
            "td,eud->teu", train_exact, torch.from_numpy(delta[begin:end]).to(device)
        )
        feature = train_feature.permute(1, 2, 0, 3).reshape(-1, train_rows, width)
        target_flat = target.permute(1, 2, 0).reshape(-1, train_rows, 1)
        gram = feature.transpose(1, 2) @ feature
        rhs = feature.transpose(1, 2) @ target_flat
        prior = torch.from_numpy(prior_levels[begin:end]).to(device).reshape(-1, width, 1)
        ridge_scale = gram.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-8)
        for alpha in alphas:
            penalty = (float(alpha) * ridge_scale)[:, None, None]
            solution = torch.linalg.solve(gram + penalty * eye, rhs + penalty * prior)
            solution_q = solution.squeeze(-1).half().float()
            learned[alpha][begin:end] = solution_q.reshape(
                end - begin, UNITS, HIDDEN // BLOCK, levels
            ).cpu().half()
            outputs[alpha][:, begin:end] = torch.einsum(
                "teup,eup->teu",
                all_feature,
                solution_q.reshape(end - begin, UNITS, width),
            ).cpu()
        del local_code, train_feature, all_feature, target, feature, target_flat, gram, rhs, prior
        gc.collect()
        torch.cuda.empty_cache()
    return outputs, learned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
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
        "--helper", type=Path,
        default=Path("/workspace/pr13_h1_four_layer_helpers.py"),
    )
    parser.add_argument("--alphas", default="0.0001,0.001,0.01,0.1,1.0")
    parser.add_argument("--chunk-experts", type=int, default=4)
    args = parser.parse_args()
    alphas = tuple(float(value) for value in args.alphas.split(","))
    device = torch.device("cuda")
    helper = load_module("pr13_h1_calibration_helpers", args.helper)
    sys.path[:0] = [str(args.oracle_root / "src"), str(args.oracle_root / "scripts")]
    import run_set_utility_distillation as prior
    from run_mxfp4_selective_pages import tree_from_record

    dataset = torch.load(
        args.experiment / f"dataset_layer_{args.layer}.pt",
        map_location="cpu", weights_only=False,
    )
    active = [int(value) for value in dataset["active_experts"]]
    parent_x = dataset["parent_target_x"].float().to(device)
    exact_x = dataset["exact_target_activation_auxiliary"].float().to(device)
    train_cpu = torch.tensor([value == "train" for value in dataset["split"]], dtype=torch.bool)
    train = train_cpu.to(device)
    validation_cpu = ~train_cpu
    active_index = {expert: index for index, expert in enumerate(active)}
    routed = torch.tensor(
        [[active_index[int(expert)] for expert in row] for row in dataset["expert_ids"][validation_cpu]],
        dtype=torch.int64,
    )
    checkpoint_index = json.loads(
        (args.checkpoint / "model.safetensors.index.json").read_text()
    )["weight_map"]
    records = json.loads(args.trees.read_text())
    trees = {name: tree_from_record(records[name]) for name in ("gate", "up", "down")}
    gate2 = np.empty((len(active), UNITS, HIDDEN), np.float32)
    up2 = np.empty_like(gate2)
    deltas = {name: np.empty_like(gate2) for name in ("gate", "up")}
    for index, expert in enumerate(active):
        decoded = prior.decode_expert(args.checkpoint, checkpoint_index, trees, args.layer, expert)
        gate2[index] = decoded["gate"][0]
        up2[index] = decoded["up"][0]
        deltas["gate"][index] = decoded["gate"][2] - decoded["gate"][0]
        deltas["up"][index] = decoded["up"][2] - decoded["up"][0]
    g2 = helper.project(gate2, parent_x, device).cpu()
    u2 = helper.project(up2, parent_x, device).cpu()
    exact_hidden = dataset["exact_hidden_compact"].float()

    records_out = []
    for source, bits in (("b1_asym_b512", 1), ("q2_asym_b512", 2)):
        base_model = torch.load(
            args.experiment / f"model_layer_{args.layer}_{source}.pt",
            map_location="cpu", weights_only=False,
        )
        projections, levels_by_name = {}, {}
        for name in ("gate", "up"):
            if bits == 1:
                code, initial = unpack_binary(base_model, name)
            else:
                code, initial = unpack_two_bit(base_model, name)
            projections[name], levels_by_name[name] = calibrated_projection(
                parent_x, exact_x, deltas[name], code, initial, train, alphas,
                args.chunk_experts, device,
            )
        for alpha in alphas:
            prediction = helper.hidden(
                g2, g2 + projections["gate"][alpha],
                u2, u2 + projections["up"][alpha],
            )
            token = str(alpha).replace(".", "p")
            variant = f"{source}_actcal_a{token}"
            prediction_path = args.experiment / f"prediction_layer_{args.layer}_{variant}.pt"
            model_path = args.experiment / f"model_layer_{args.layer}_{variant}.pt"
            artifact = dict(base_model)
            artifact.update(
                schema="pr13_h1_activation_calibrated_lookup_v1",
                source_variant=source,
                ridge_alpha=alpha,
                calibration_rows=int(train.sum()),
            )
            for name in ("gate", "up"):
                artifact[f"{name}_signed_level_fp16"] = levels_by_name[name][alpha]
                artifact.pop(f"{name}_scale_fp16", None)
                artifact.pop(f"{name}_level_fp16", None)
            atomic_save(prediction.half(), prediction_path)
            atomic_save(artifact, model_path)
            metrics = helper.routed_metrics(
                exact_hidden[validation_cpu], prediction[validation_cpu], routed
            )
            streamed = 278528 if bits == 1 else 557056
            record = {
                "layer": args.layer,
                "variant": variant,
                "source_variant": source,
                "bits": bits,
                "ridge_alpha": alpha,
                "train_rows": int(train.sum()),
                "validation_rows": int(validation_cpu.sum()),
                "streamed_bytes_per_expert": streamed,
                "streamed_bytes_top8": streamed * 8,
                "metrics": metrics,
                "prediction": str(prediction_path),
                "prediction_sha256": sha256(prediction_path),
                "model": str(model_path),
                "model_sha256": sha256(model_path),
            }
            records_out.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    output = args.experiment / f"activation_calibration_layer_{args.layer}.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({
        "layer": args.layer, "alphas": alphas, "results": records_out,
    }, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)


if __name__ == "__main__":
    main()
