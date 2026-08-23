#!/usr/bin/env python3
"""Train an in-domain shared, layer-conditioned resident H1 state corrector."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


HIDDEN = 2048
LAYERS = 40
SEED = 42


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


def load_layer(root: Path, layer: int) -> dict[str, np.ndarray]:
    pieces: dict[str, list[np.ndarray]] = defaultdict(list)
    for split in ("train", "validation"):
        with np.load(root / f"{split}_layer_{layer:02d}.npz", allow_pickle=False) as loaded:
            for name in loaded.files:
                pieces[name].append(np.asarray(loaded[name]))
    return {name: np.concatenate(values, axis=0) for name, values in pieces.items()}


def identity_map(data: Mapping[str, np.ndarray]) -> dict[tuple[str, int, str], int]:
    return {
        (str(data["request_id"][i]), int(data["position"][i]), str(data["prefix_hash"][i])): i
        for i in range(len(data["position"]))
    }


def build_dataset(exact_root: Path, parent_root: Path) -> dict[str, torch.Tensor | list[str]]:
    target_parent, source_parent, target_exact, layer_ids, splits, requests = [], [], [], [], [], []
    for layer in range(LAYERS):
        exact = load_layer(exact_root, layer)
        parent = load_layer(parent_root, layer)
        parent_ids = identity_map(parent)
        exact_by_request_position = {
            (str(exact["request_id"][i]), int(exact["position"][i])): i
            for i in range(len(exact["position"]))
        }
        for i in np.flatnonzero(exact["position"] >= 1):
            request = str(exact["request_id"][i])
            position = int(exact["position"][i])
            target_key = (request, position, str(exact["prefix_hash"][i]))
            source_i = exact_by_request_position[(request, position - 1)]
            source_key = (request, position - 1, str(exact["prefix_hash"][source_i]))
            target_parent.append(parent["x"][parent_ids[target_key]])
            source_parent.append(parent["x"][parent_ids[source_key]])
            target_exact.append(exact["x"][i])
            layer_ids.append(layer)
            splits.append(str(exact["split"][i]))
            requests.append(request)
    return {
        "target_parent": torch.tensor(np.stack(target_parent), dtype=torch.float32),
        "source_parent": torch.tensor(np.stack(source_parent), dtype=torch.float32),
        "target_exact": torch.tensor(np.stack(target_exact), dtype=torch.float32),
        "layer": torch.tensor(layer_ids, dtype=torch.int64),
        "split": splits,
        "request": requests,
    }


class Corrector(nn.Module):
    def __init__(self, width: int, depth: int, use_source: bool):
        super().__init__()
        self.width = width
        self.depth = depth
        self.use_source = use_source
        self.target_encoder = nn.Linear(HIDDEN, width, bias=False)
        self.source_encoder = nn.Linear(HIDDEN, width, bias=False) if use_source else None
        self.layer_embedding = nn.Embedding(LAYERS, width)
        self.blocks = nn.ModuleList(nn.Linear(width, width) for _ in range(depth - 1))
        self.decoder = nn.Linear(width, HIDDEN, bias=False)
        self.layer_bias = nn.Embedding(LAYERS, HIDDEN)
        nn.init.normal_(self.target_encoder.weight, std=0.02 / (HIDDEN ** 0.5))
        if self.source_encoder is not None:
            nn.init.normal_(self.source_encoder.weight, std=0.02 / (HIDDEN ** 0.5))
        nn.init.zeros_(self.layer_embedding.weight)
        for block in self.blocks:
            nn.init.orthogonal_(block.weight, gain=0.1)
            nn.init.zeros_(block.bias)
        nn.init.normal_(self.decoder.weight, std=1e-4)
        nn.init.zeros_(self.layer_bias.weight)

    def forward(self, target_parent: torch.Tensor, source_parent: torch.Tensor, layer: torch.Tensor) -> torch.Tensor:
        target_norm = F.rms_norm(target_parent, (HIDDEN,))
        latent = self.target_encoder(target_norm) + self.layer_embedding(layer)
        if self.source_encoder is not None:
            source_norm = F.rms_norm(source_parent, (HIDDEN,))
            latent = latent + self.source_encoder(source_norm)
        latent = F.silu(latent)
        for block in self.blocks:
            latent = latent + F.silu(block(latent))
        residual = self.decoder(latent) + self.layer_bias(layer)
        return target_parent + residual


def metric(target: torch.Tensor, prediction: torch.Tensor, layers: torch.Tensor) -> dict[str, Any]:
    error = (prediction.float() - target.float()).square()
    cosine = F.cosine_similarity(prediction.float(), target.float(), dim=-1)
    by_layer = []
    for layer in range(LAYERS):
        mask = layers == layer
        by_layer.append({
            "layer": layer,
            "mse": float(error[mask].mean()),
            "cosine": float(cosine[mask].mean()),
        })
    layer_mses = np.asarray([row["mse"] for row in by_layer])
    return {
        "mse": float(error.mean()),
        "cosine": float(cosine.mean()),
        "layer_mse_p10": float(np.quantile(layer_mses, 0.1)),
        "layer_mse_median": float(np.median(layer_mses)),
        "layer_mse_p90": float(np.quantile(layer_mses, 0.9)),
        "layers_below_0_1": int((layer_mses < 0.1).sum()),
        "by_layer": by_layer,
    }


@torch.no_grad()
def evaluate(model: nn.Module, data: dict[str, torch.Tensor], indices: torch.Tensor, device: torch.device, batch_size: int) -> dict[str, Any]:
    model.eval()
    predictions = []
    for start in range(0, len(indices), batch_size):
        local = indices[start:start + batch_size]
        predictions.append(model(
            data["target_parent"][local].to(device),
            data["source_parent"][local].to(device),
            data["layer"][local].to(device),
        ).cpu())
    prediction = torch.cat(predictions)
    return metric(data["target_exact"][indices], prediction, data["layer"][indices])


def train_one(args: argparse.Namespace, config: dict[str, Any], data: dict[str, Any], fit: torch.Tensor, tune: torch.Tensor, validation: torch.Tensor, recorder: Recorder, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(SEED)
    model = Corrector(config["width"], config["depth"], config["use_source"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_mse = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    generator = torch.Generator().manual_seed(SEED)
    started = time.time()
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        order = fit[torch.randperm(len(fit), generator=generator)]
        total_loss = 0.0
        for start in range(0, len(order), args.batch_size):
            local = order[start:start + args.batch_size]
            target_parent = data["target_parent"][local].to(device)
            source_parent = data["source_parent"][local].to(device)
            target_exact = data["target_exact"][local].to(device)
            layer = data["layer"][local].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = model(target_parent, source_parent, layer)
                loss = (prediction.float() - target_exact).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss) * len(local)
        if epoch % args.eval_every == 0 or epoch == 1:
            tune_metric = evaluate(model, data, tune, device, args.eval_batch_size)
            recorder("epoch", config=config["name"], epoch=epoch, fit_mse=total_loss / len(fit), tune_mse=tune_metric["mse"])
            if tune_metric["mse"] < best_mse - args.min_delta:
                best_mse = tune_metric["mse"]
                best_state = {key: value.detach().half().cpu().clone() for key, value in model.state_dict().items()}
                best_epoch = epoch
                stale = 0
            else:
                stale += args.eval_every
            if stale >= args.patience:
                break
    assert best_state is not None
    model.load_state_dict({key: value.float() for key, value in best_state.items()})
    tune_metric = evaluate(model, data, tune, device, args.eval_batch_size)
    validation_metric = evaluate(model, data, validation, device, args.eval_batch_size)
    parameter_count = sum(value.numel() for value in model.parameters())
    result = {
        **config,
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "parameter_count": parameter_count,
        "resident_bytes_fp16": parameter_count * 2,
        "streamed_bpw": 0.0,
        "macs_per_activation": int((1 + int(config["use_source"])) * HIDDEN * config["width"] + (config["depth"] - 1) * config["width"] ** 2 + config["width"] * HIDDEN),
        "tune": tune_metric,
        "validation": validation_metric,
        "seconds": time.time() - started,
    }
    torch.save({"schema": "shared_layer_conditioned_h1_corrector_v1", "config": config, "state_dict_fp16": best_state, "result": result}, args.output / f"{config['name']}.pt")
    recorder("architecture_completed", result=result)
    del model, optimizer, best_state
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact-root", type=Path, default=Path("/workspace/all_layer_split_interaction_study/capture"))
    parser.add_argument("--parent-root", type=Path, default=Path("/workspace/embedded_parent_capture_20260821_v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configs", default="target_w128,target_source_w128,target_source_w256,target_source_w256_d2,target_source_w512")
    parser.add_argument("--tune-request", default="mxfp4-confirm-004")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    recorder = Recorder(args.output / "run.jsonl")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0")
    data = build_dataset(args.exact_root, args.parent_root)
    split = np.asarray(data["split"])
    request = np.asarray(data["request"])
    fit = torch.from_numpy(np.flatnonzero((split == "train") & (request != args.tune_request))).long()
    tune = torch.from_numpy(np.flatnonzero((split == "train") & (request == args.tune_request))).long()
    validation = torch.from_numpy(np.flatnonzero(split == "validation")).long()
    baseline = {
        "name": "uncorrected_parent",
        "streamed_bpw": 0.0,
        "resident_bytes_fp16": 0,
        "tune": metric(data["target_exact"][tune], data["target_parent"][tune], data["layer"][tune]),
        "validation": metric(data["target_exact"][validation], data["target_parent"][validation], data["layer"][validation]),
    }
    definitions = {
        "target_w128": {"name": "target_w128", "width": 128, "depth": 1, "use_source": False},
        "target_source_w128": {"name": "target_source_w128", "width": 128, "depth": 1, "use_source": True},
        "target_source_w256": {"name": "target_source_w256", "width": 256, "depth": 1, "use_source": True},
        "target_source_w256_d2": {"name": "target_source_w256_d2", "width": 256, "depth": 2, "use_source": True},
        "target_source_w512": {"name": "target_source_w512", "width": 512, "depth": 1, "use_source": True},
    }
    configs = [definitions[name] for name in args.configs.split(",")]
    recorder("training_started", gpu=torch.cuda.get_device_name(0), fit_rows=len(fit), tune_rows=len(tune), validation_rows=len(validation), baseline=baseline, configs=configs)
    results = [train_one(args, config, data, fit, tune, validation, recorder, device) for config in configs]
    atomic_json(args.output / "result.json", {"schema": "pr13_shared_h1_corrector_screen_v1", "baseline": baseline, "results": results})
    recorder("training_completed", result=str(args.output / "result.json"))


if __name__ == "__main__":
    main()
