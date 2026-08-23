#!/usr/bin/env python3
"""Bounded H4 response-field prediction pilot for compression-study PR #13.

The temporal contract is deliberately explicit: source position t is paired
with target position t+4.  Predictor inputs come only from the aligned natural
2-bit-parent rollout.  The exact target activation is serialized as auxiliary
data and is used only to construct/scored exact response targets.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


LAYERS = (0, 4, 20, 39)
EXPERTS = 256
UNITS = 512
HIDDEN = 2048
HORIZON = 4
COMPONENTS = ("b", "delta_g", "delta_u", "iota")
BUDGETS = (384, 576, 749)
RATES = {384: 0.523478, 576: 0.773478, 749: 0.998739}
PRIMARY_FACTOR = "matrix_free_joint_rank8_int4_per_row_hadamard"
PRIMARY_POLICY = "pooled_router_square_column_generated"
BURST_CAP = 1536
TUNE_REQUEST = "mxfp4-confirm-004"
SEED = 20260822

EXPECTED_HASHES = {
    "capture": "52bc9eb2d726e014efef77f24e2f6eaf4401b5f7abd7dddb2d84f1a55483a931",
    "trees": "da4675181fd81c87bd2d3db7ff2d3dea1f075db4ea98f23ab90f14943ee2bfe8",
    "config": "a889dbf9c5ebdf79de0e7e7b7d65165d9f39782d72d669e073c80966e5d83edf",
    "runner": "55dd6f1b28b3dea36913943d5094b007176ddeabf8c12925dfbcf200118acaae",
    "allocator": "30ef9e95970533903754b2e68d299f8113a1c2c00f583a981bc3cc89498ba75b",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


class Recorder:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __call__(self, event: str, **values: Any) -> None:
        record = {"time_unix": time.time(), "event": event, **values}
        line = json.dumps(record, sort_keys=True)
        print(line, flush=True)
        with self.path.open("a") as handle:
            handle.write(line + "\n")


def components_from_hidden(hidden: torch.Tensor) -> torch.Tensor:
    """Convert code order [h22,h24,h42,h44] to [b,dg,du,iota]."""
    h22, h24, h42, h44 = hidden.unbind(dim=-2)
    return torch.stack(
        (h22, h42 - h22, h24 - h22, h44 - h42 - h24 + h22),
        dim=-2,
    )


def hidden_from_components(components: np.ndarray | torch.Tensor) -> Any:
    b, delta_g, delta_u, iota = [components[..., index, :] for index in range(4)]
    stack = (
        b,
        b + delta_u,
        b + delta_g,
        b + delta_g + delta_u + iota,
    )
    if isinstance(components, torch.Tensor):
        return torch.stack(stack, dim=-2)
    return np.stack(stack, axis=-2)


def silu(value: torch.Tensor) -> torch.Tensor:
    return F.silu(value)


@dataclass(frozen=True)
class ModelConfig:
    input_size: int = HIDDEN * 2 + EXPERTS
    width: int = 64
    rank: int = 16
    dropout: float = 0.05


class ResponseFieldPredictor(nn.Module):
    """c -> z -> expert/unit response components with a resident-Q2 b skip."""

    def __init__(
        self,
        config: ModelConfig,
        input_mean: torch.Tensor,
        input_scale: torch.Tensor,
        output_bias: torch.Tensor,
        initial_head: torch.Tensor,
    ) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("input_mean", input_mean.float())
        self.register_buffer("input_scale", input_scale.float())
        self.encoder = nn.Sequential(
            nn.Linear(config.input_size, config.width),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.width, config.rank),
        )
        self.output_bias = nn.Parameter(output_bias.float())
        self.output_head = nn.Parameter(initial_head.float())

    def encode(self, context: torch.Tensor) -> torch.Tensor:
        normalized = (context.float() - self.input_mean) / self.input_scale
        return self.encoder(normalized)

    def forward(self, context: torch.Tensor, q2_b: torch.Tensor) -> torch.Tensor:
        z = self.encode(context)
        residual = self.output_bias[None] + torch.einsum(
            "nr,ecur->necu", z, self.output_head,
        )
        base = torch.zeros_like(residual)
        base[:, :, 0] = q2_b.float()
        return base + residual


def response_loss(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    pred = prediction.float()
    truth = target.float()
    dot = (pred * truth).sum(dim=-1)
    pred_norm = pred.square().sum(dim=-1).clamp_min(1e-12).sqrt()
    truth_norm = truth.square().sum(dim=-1).clamp_min(1e-12).sqrt()
    cosine = dot / (pred_norm * truth_norm).clamp_min(1e-8)
    cosine_loss = (1.0 - cosine).mean()
    magnitude = (torch.log(pred_norm + 1e-6) - torch.log(truth_norm + 1e-6)).square().mean()
    relative = (
        (pred - truth).square().mean(dim=-1)
        / truth.square().mean(dim=-1).clamp_min(1e-8)
    ).clamp_max(20.0).mean()
    total = cosine_loss + 0.05 * magnitude + 0.01 * relative
    return total, {
        "loss": float(total.detach()),
        "cosine_loss": float(cosine_loss.detach()),
        "magnitude_loss": float(magnitude.detach()),
        "relative_mse_stabilizer": float(relative.detach()),
    }


@torch.no_grad()
def evaluate_loss(
    model: ResponseFieldPredictor,
    context: torch.Tensor,
    q2_b: torch.Tensor,
    target: torch.Tensor,
    indices: torch.Tensor,
    device: torch.device,
    batch_size: int = 3,
) -> dict[str, float]:
    model.eval()
    weighted = defaultdict(float)
    rows = 0
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(
                context[selected].to(device), q2_b[selected].to(device),
            )
        _, metrics = response_loss(prediction, target[selected].to(device))
        count = len(selected)
        for name, value in metrics.items():
            weighted[name] += value * count
        rows += count
    return {name: value / max(rows, 1) for name, value in weighted.items()}


def pca_initialization(
    target: torch.Tensor,
    q2_b: torch.Tensor,
    fit_indices: torch.Tensor,
    rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train-only PCA initializer for the fixed rank-z output family."""
    truth = target[fit_indices].to(device).float()
    base = torch.zeros_like(truth)
    base[:, :, 0] = q2_b[fit_indices].to(device).float()
    residual = truth - base
    bias = residual.mean(dim=0)
    centered = residual - bias
    component_scale = truth.square().mean(dim=(0, 1, 3)).sqrt().clamp_min(1e-4)
    standardized = centered / component_scale[None, None, :, None]
    flat = standardized.flatten(1)
    gram = flat @ flat.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)[:rank]
    values = eigenvalues[order].clamp_min(1e-8)
    left = eigenvectors[:, order]
    singular = values.sqrt()
    right = (left.T @ flat) / singular[:, None]
    scores = left * singular[None]
    score_scale = scores.std(dim=0).clamp_min(1e-4)
    z_target = scores / score_scale[None]
    raw_head = right.reshape(rank, EXPERTS, 4, UNITS)
    raw_head = raw_head * component_scale[None, None, :, None]
    raw_head = raw_head * score_scale[:, None, None, None]
    head = raw_head.permute(1, 2, 3, 0).contiguous()
    return bias.cpu(), head.cpu(), z_target.cpu()


def train_layer_model(
    layer: int,
    dataset: Mapping[str, Any],
    output: Path,
    recorder: Recorder,
    device: torch.device,
) -> dict[str, Any]:
    request_ids = list(dataset["request_id"])
    split = list(dataset["split"])
    fit_indices = torch.tensor([
        index for index, (name, request) in enumerate(zip(split, request_ids))
        if name == "train" and request != TUNE_REQUEST
    ])
    tune_indices = torch.tensor([
        index for index, (name, request) in enumerate(zip(split, request_ids))
        if name == "train" and request == TUNE_REQUEST
    ])
    if len(fit_indices) != 37 or len(tune_indices) != 9:
        raise RuntimeError(f"layer {layer} fit/tune rows changed: {len(fit_indices)}/{len(tune_indices)}")

    context = torch.cat(
        (
            dataset["parent_target_x"],
            dataset["parent_source_x"],
            dataset["parent_target_router_logits"],
        ),
        dim=-1,
    ).float()
    target = components_from_hidden(dataset["exact_hidden"].float())
    q2_b = dataset["parent_hidden"][:, :, 0].float()
    input_mean = context[fit_indices].mean(dim=0)
    input_scale = context[fit_indices].std(dim=0).clamp_min(1e-3)
    config = ModelConfig()
    output_bias, initial_head, z_target = pca_initialization(
        target, q2_b, fit_indices, config.rank, device,
    )
    model = ResponseFieldPredictor(
        config, input_mean, input_scale, output_bias, initial_head,
    ).to(device)

    # Fit only c->z to the train-only PCA scores before optimizing the stated
    # response loss. This is initialization, not a second architecture.
    encoder_optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=2e-3, weight_decay=0.02)
    generator = torch.Generator().manual_seed(SEED + layer)
    fit_lookup = {int(value): position for position, value in enumerate(fit_indices.tolist())}
    model.train()
    for step in range(240):
        order = fit_indices[torch.randperm(len(fit_indices), generator=generator)]
        selected = order[: min(8, len(order))]
        prediction = model.encode(context[selected].to(device))
        local_indices = torch.tensor([fit_lookup[int(value)] for value in selected])
        loss = F.mse_loss(prediction.float(), z_target[local_indices].to(device).float())
        encoder_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        encoder_optimizer.step()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    initial_tune = evaluate_loss(model, context, q2_b, target, tune_indices, device)
    best_metric = initial_tune["cosine_loss"] + 0.05 * initial_tune["magnitude_loss"]
    best_epoch = 0
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    history = [{"epoch": 0, "tune": initial_tune}]
    recorder(
        "training_started", layer=layer, parameters=parameter_count,
        fit_rows=len(fit_indices), tune_rows=len(tune_indices), initial_tune=initial_tune,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, betas=(0.9, 0.95), weight_decay=0.01)
    maximum_epochs, minimum_epochs, patience = 80, 8, 6
    stale = 0
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        order = fit_indices[torch.randperm(len(fit_indices), generator=generator)]
        totals = defaultdict(float)
        seen = 0
        for start in range(0, len(order), 2):
            selected = order[start : start + 2]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(context[selected].to(device), q2_b[selected].to(device))
            loss, metrics = response_loss(prediction, target[selected].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for name, value in metrics.items():
                totals[name] += value * len(selected)
            seen += len(selected)
        tune = evaluate_loss(model, context, q2_b, target, tune_indices, device)
        train_metrics = {name: value / seen for name, value in totals.items()}
        selection = tune["cosine_loss"] + 0.05 * tune["magnitude_loss"]
        improved = selection < best_metric - 1e-4
        if improved:
            best_metric = selection
            best_epoch = epoch
            stale = 0
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        else:
            stale += 1
        record = {"epoch": epoch, "train": train_metrics, "tune": tune, "improved": improved}
        history.append(record)
        recorder("epoch", layer=layer, **record)
        if epoch >= minimum_epochs and stale >= patience:
            recorder("plateau_stop", layer=layer, epoch=epoch, best_epoch=best_epoch, stale=stale)
            break

    model.load_state_dict(best_state)
    final_fit = evaluate_loss(model, context, q2_b, target, fit_indices, device)
    final_tune = evaluate_loss(model, context, q2_b, target, tune_indices, device)
    model.eval()
    prediction_parts = []
    with torch.no_grad():
        for start in range(0, len(context), 3):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction_parts.append(model(
                    context[start : start + 3].to(device),
                    q2_b[start : start + 3].to(device),
                ).float().cpu())
    predictions = torch.cat(prediction_parts)
    checkpoint = {
        "schema": "pr13_h4_response_predictor_rank16_v1",
        "layer": layer,
        "model_config": asdict(config),
        "model": best_state,
        "best_epoch": best_epoch,
        "selection_metric": "cosine_loss + 0.05 * log_magnitude_loss",
        "fit_metrics": final_fit,
        "tune_metrics": final_tune,
        "parameter_count": parameter_count,
        "input_contract": ["parent_target_x", "parent_source_x", "parent_target_router_logits"],
        "target_contract": list(COMPONENTS),
        "true_activation_is_input": False,
    }
    checkpoint_path = output / f"model_layer_{layer}.pt"
    predictions_path = output / f"predicted_components_layer_{layer}.pt"
    history_path = output / f"history_layer_{layer}.json"
    atomic_torch_save(checkpoint, checkpoint_path)
    atomic_torch_save(predictions.half(), predictions_path)
    atomic_json(history_path, history)
    result = {
        "layer": layer,
        "best_epoch": best_epoch,
        "epochs_run": history[-1]["epoch"],
        "parameter_count": parameter_count,
        "fit": final_fit,
        "tune": final_tune,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "predictions": str(predictions_path),
        "predictions_sha256": sha256(predictions_path),
        "history_sha256": sha256(history_path),
    }
    recorder("training_completed", **result)
    del model, best_state, target, predictions
    gc.collect()
    torch.cuda.empty_cache()
    return result


def identity_map(data: Mapping[str, np.ndarray]) -> dict[tuple[str, int, str], int]:
    result = {}
    for index in range(len(data["position"])):
        key = (
            str(data["request_id"][index]),
            int(data["position"][index]),
            str(data["prefix_hash"][index]),
        )
        if key in result:
            raise RuntimeError(f"duplicate capture identity: {key}")
        result[key] = index
    return result


def load_parent_layer(parent_root: Path, layer: int) -> dict[str, np.ndarray]:
    pieces: dict[str, list[np.ndarray]] = defaultdict(list)
    for split in ("train", "validation"):
        with np.load(parent_root / f"{split}_layer_{layer:02d}.npz", allow_pickle=False) as loaded:
            for name in loaded.files:
                pieces[name].append(np.asarray(loaded[name]))
    return {name: np.concatenate(values, axis=0) for name, values in pieces.items()}


def eligible_rows(local: Mapping[str, np.ndarray]) -> list[int]:
    split = np.asarray(local["split"]).astype(str)
    return [
        index for index in range(len(split))
        if split[index] in {"train", "validation"} and int(local["position"][index]) >= HORIZON
    ]


def gpu_projection(weights: np.ndarray, activations: torch.Tensor, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(weights).to(device)
    flat = tensor.reshape(EXPERTS * UNITS, HIDDEN)
    projected = (flat @ activations.T).T.reshape(len(activations), EXPERTS, UNITS)
    del tensor, flat
    torch.cuda.empty_cache()
    return projected


def build_layer_dataset(
    layer: int,
    exact_data: Mapping[str, np.ndarray],
    parent_root: Path,
    checkpoint: Path,
    index: Mapping[str, str],
    trees: Mapping[str, Any],
    output: Path,
    recorder: Recorder,
    device: torch.device,
    prior: Any,
) -> dict[str, Any]:
    started = time.time()
    local = prior.layer_view(exact_data, layer)
    parent = load_parent_layer(parent_root, layer)
    parent_by_identity = identity_map(parent)
    exact_by_request_position = {
        (str(local["request_id"][index]), int(local["position"][index])): index
        for index in range(len(local["position"]))
    }
    rows = eligible_rows(local)
    if len(rows) != 66:
        raise RuntimeError(f"layer {layer} H4-admissible rows changed: {len(rows)}")
    parent_target, parent_source, parent_router = [], [], []
    exact_target, request_ids, splits, positions, prefixes = [], [], [], [], []
    expert_ids, router_weights = [], []
    for record in rows:
        request = str(local["request_id"][record])
        position = int(local["position"][record])
        target_key = (request, position, str(local["prefix_hash"][record]))
        source_record = exact_by_request_position.get((request, position - HORIZON))
        if source_record is None:
            raise RuntimeError("H4 source identity is missing")
        source_key = (
            request,
            position - HORIZON,
            str(local["prefix_hash"][source_record]),
        )
        target_parent_index = parent_by_identity[target_key]
        source_parent_index = parent_by_identity[source_key]
        if str(parent["split"][target_parent_index]) != str(local["split"][record]):
            raise RuntimeError("parent/exact split alignment changed")
        parent_target.append(parent["x"][target_parent_index])
        parent_source.append(parent["x"][source_parent_index])
        parent_router.append(parent["router_logits"][target_parent_index])
        exact_target.append(local["x"][record])
        request_ids.append(request)
        splits.append(str(local["split"][record]))
        positions.append(position)
        prefixes.append(str(local["prefix_hash"][record]))
        expert_ids.append(local["expert_ids"][record])
        router_weights.append(local["router_weights"][record])

    active = sorted(set(np.asarray(expert_ids, np.int64)[np.asarray(splits) == "validation"].ravel().tolist()))
    gate2 = np.empty((EXPERTS, UNITS, HIDDEN), np.float32)
    gate4 = np.empty_like(gate2)
    up2 = np.empty_like(gate2)
    up4 = np.empty_like(gate2)
    down2, down4 = {}, {}
    decode_started = time.time()
    for expert in range(EXPERTS):
        decoded = prior.decode_expert(checkpoint, index, trees, layer, expert)
        gate2[expert] = decoded["gate"][0]
        gate4[expert] = decoded["gate"][2]
        up2[expert] = decoded["up"][0]
        up4[expert] = decoded["up"][2]
        if expert in active:
            down2[expert] = torch.from_numpy(decoded["down"][0].copy())
            down4[expert] = torch.from_numpy(decoded["down"][2].copy())
        if expert % 32 == 31:
            recorder(
                "decode_progress", layer=layer, experts=expert + 1,
                seconds=time.time() - decode_started,
            )
        del decoded

    exact_tensor = torch.from_numpy(np.stack(exact_target).astype(np.float32)).to(device)
    parent_tensor = torch.from_numpy(np.stack(parent_target).astype(np.float32)).to(device)
    activations = torch.cat((exact_tensor, parent_tensor), dim=0)
    projections = []
    for name, weights in (("gate2", gate2), ("gate4", gate4), ("up2", up2), ("up4", up4)):
        projection_started = time.time()
        projections.append(gpu_projection(weights, activations, device))
        recorder("projection_completed", layer=layer, projection=name, seconds=time.time() - projection_started)
    g2, g4, u2, u4 = projections
    hidden = torch.stack(
        (silu(g2) * u2, silu(g2) * u4, silu(g4) * u2, silu(g4) * u4),
        dim=-2,
    )
    exact_hidden = hidden[: len(rows)].float().cpu()
    parent_hidden = hidden[len(rows) :].float().cpu()

    # A float64 single-cell audit guards the GPU vectorization/order contract.
    audit_expert = int(expert_ids[0][0])
    decoded_audit = prior.decode_expert(checkpoint, index, trees, layer, audit_expert)
    from oracle_study.split_interaction_field import split_projection_responses
    reference = split_projection_responses(
        tuple(decoded_audit[name][0] for name in ("gate", "up", "down")),
        tuple(decoded_audit[name][2] for name in ("gate", "up", "down")),
        np.asarray(exact_target[0], np.float32),
    ).hidden
    observed = exact_hidden[0, audit_expert].numpy().T
    audit_relative_mse = float(np.square(observed - reference).sum() / max(np.square(reference).sum(), 1e-30))
    if audit_relative_mse > 2e-10:
        raise RuntimeError(f"vectorized response audit failed: {audit_relative_mse}")

    dataset = {
        "schema": "pr13_h4_response_examples_v1",
        "layer": layer,
        "horizon": HORIZON,
        "request_id": request_ids,
        "split": splits,
        "target_position": torch.tensor(positions, dtype=torch.int64),
        "source_position": torch.tensor(positions, dtype=torch.int64) - HORIZON,
        "prefix_hash": prefixes,
        "expert_ids": torch.from_numpy(np.stack(expert_ids).astype(np.int64)),
        "router_weights": torch.from_numpy(np.stack(router_weights).astype(np.float32)),
        "parent_target_x": torch.from_numpy(np.stack(parent_target).astype(np.float32)),
        "parent_source_x": torch.from_numpy(np.stack(parent_source).astype(np.float32)),
        "parent_target_router_logits": torch.from_numpy(np.stack(parent_router).astype(np.float32)),
        "exact_target_activation_auxiliary": torch.from_numpy(np.stack(exact_target).astype(np.float32)),
        "exact_hidden": exact_hidden,
        "parent_hidden": parent_hidden,
        "input_fields": ["parent_target_x", "parent_source_x", "parent_target_router_logits"],
        "auxiliary_only_fields": ["exact_target_activation_auxiliary"],
        "target_fields": list(COMPONENTS),
        "branch_conditional": True,
    }
    runtime = {
        "schema": "pr13_h4_allocator_runtime_layer_v1",
        "layer": layer,
        "active_experts": active,
        "down2": torch.stack([down2[expert] for expert in active]),
        "down4": torch.stack([down4[expert] for expert in active]),
    }
    dataset_path = output / f"dataset_layer_{layer}.pt"
    runtime_path = output / f"allocator_runtime_layer_{layer}.pt"
    atomic_torch_save(dataset, dataset_path)
    atomic_torch_save(runtime, runtime_path)
    result = {
        "layer": layer,
        "rows": len(rows),
        "train_rows": sum(value == "train" for value in splits),
        "validation_rows": sum(value == "validation" for value in splits),
        "active_validation_experts": len(active),
        "response_vectorization_relative_mse": audit_relative_mse,
        "dataset": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
        "runtime": str(runtime_path),
        "runtime_sha256": sha256(runtime_path),
        "seconds": time.time() - started,
    }
    recorder("dataset_layer_completed", **result)
    del gate2, gate4, up2, up4, projections, hidden, exact_hidden, parent_hidden
    del exact_tensor, parent_tensor, activations, down2, down4, runtime, dataset
    gc.collect()
    torch.cuda.empty_cache()
    return result


def metric_rows(
    target: torch.Tensor,
    prediction: torch.Tensor,
    row_indices: torch.Tensor,
    routed_experts: torch.Tensor,
    method: str,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    rows = torch.arange(len(row_indices))[:, None]
    truth = target[row_indices][rows, routed_experts]
    pred = prediction[row_indices][rows, routed_experts]
    result = []
    accumulators = {}
    for component, name in enumerate(COMPONENTS):
        t = truth[:, :, component].double()
        p = pred[:, :, component].double()
        dot = (t * p).sum(dim=-1)
        tnorm = t.square().sum(dim=-1).sqrt()
        pnorm = p.square().sum(dim=-1).sqrt()
        cosine = dot / (tnorm * pnorm).clamp_min(1e-20)
        error = (p - t).square()
        result.append({
            "method": method,
            "component": name,
            "vectors": int(t.shape[0] * t.shape[1]),
            "cosine": float(cosine.mean()),
            "mse": float(error.mean()),
            "relative_mse": float(error.sum() / t.square().sum().clamp_min(1e-30)),
            "target_rms": float(t.square().mean().sqrt()),
        })
        accumulators[name] = torch.tensor([
            cosine.sum(), cosine.numel(), error.sum(), t.square().sum(), error.numel()
        ], dtype=torch.float64)
    return result, accumulators


_ALLOC_CONTEXT: dict[str, Any] | None = None


def exact_feature(responses: Any, states: np.ndarray, qmetric_features: Any, split_state_output: Any, proxy: np.ndarray, beta: float) -> np.ndarray:
    residual = np.asarray(responses.target_output, np.float64) - split_state_output(responses, states)
    return qmetric_features(residual[None], proxy, beta)[0]


def page_overlap(states_a: np.ndarray, states_b: np.ndarray, gate: np.ndarray, up: np.ndarray, down: np.ndarray) -> float:
    pages_a = np.stack((gate[states_a], up[states_a], down[states_a]), axis=-1).reshape(-1)
    pages_b = np.stack((gate[states_b], up[states_b], down[states_b]), axis=-1).reshape(-1)
    union = np.count_nonzero(pages_a | pages_b)
    return 1.0 if union == 0 else float(np.count_nonzero(pages_a & pages_b) / union)


def evaluate_allocation_group(task: int) -> list[dict[str, Any]]:
    if _ALLOC_CONTEXT is None:
        raise RuntimeError("allocation worker context missing")
    c = _ALLOC_CONTEXT
    dataset = c["dataset"]
    record = c["validation_indices"][task]
    experts = dataset["expert_ids"][record].numpy().astype(np.int64)
    weights = dataset["router_weights"][record].numpy().astype(np.float64)
    request = dataset["request_id"][record]
    position = int(dataset["target_position"][record])
    layer = int(dataset["layer"])
    exact_responses = []
    for expert in experts:
        active_index = c["active_index"][int(expert)]
        exact_responses.append(c["SplitProjectionResponses"](
            c["exact_hidden"][record, expert].T,
            c["down2"][active_index].T,
            c["down4"][active_index].T,
        ))
    base_features = [
        exact_feature(
            response, np.zeros(UNITS, np.int64), c["qmetric_features"],
            c["split_state_output"], c["proxy"], c["beta"],
        )
        for response in exact_responses
    ]
    base_combined = sum(
        (weights[index] * feature for index, feature in enumerate(base_features)),
        np.zeros_like(base_features[0]),
    )
    base_damage = float(base_combined @ base_combined)
    rows = []
    # One-method wrapper; scientific arithmetic below remains frozen.
    methods = {"learned_response": c["learned_hidden"]}
    for method, hidden_source in methods.items():
        fields, coarse_frontiers = [], []
        predicted_responses = []
        for rank, expert in enumerate(experts):
            active_index = c["active_index"][int(expert)]
            response = c["SplitProjectionResponses"](
                hidden_source[record, expert].T,
                c["down2"][active_index].T,
                c["down4"][active_index].T,
            )
            predicted_responses.append(response)
            field = c["build_split_interaction_field"](
                c["factors"][int(expert)], response.hidden, c["abc"][int(expert)],
            )
            trace = c["lagrangian_rate_frontier"](
                field,
                maximum_pages=int(c["config"]["frontier_maximum_pages"]),
                target_pages=c["config"]["frontier_target_pages"],
                price_ratios=c["config"]["frontier_price_ratios"],
                coordinate_sweeps=int(c["config"]["coordinate_sweeps"]),
                local_shortlist=int(c["config"]["local_shortlist"]),
                local_swap_units=int(c["config"]["local_swap_units"]),
                local_max_passes=int(c["config"]["local_max_passes"]),
                price_local_max_passes=int(c["config"]["price_local_max_passes"]),
            )
            fields.append(field)
            coarse_frontiers.append(tuple(
                option for option in trace.options if int(option.pages) <= BURST_CAP
            ))
        for budget in BUDGETS:
            refined = c["allocation_aware_rate_frontiers"](
                fields, coarse_frontiers,
                page_budget=8 * budget,
                burst_cap_pages=BURST_CAP,
                weights=weights * weights,
                coordinate_sweeps=int(c["config"]["coordinate_sweeps"]),
                local_shortlist=int(c["config"]["local_shortlist"]),
                local_swap_units=int(c["config"]["local_swap_units"]),
                local_max_passes=int(c["config"]["local_max_passes"]),
                max_rounds=int(c["config"]["column_generation_max_rounds"]),
                adaptive_price_multipliers=c["config"]["column_generation_price_multipliers"],
                adaptive_price_local_passes=int(c["config"]["column_generation_price_local_max_passes"]),
            )
            selected_states, selected_pages, features = [], [], []
            for router_rank, option_index in enumerate(refined.allocation.option_indices):
                option = refined.frontiers[router_rank][int(option_index)]
                states = np.asarray(option.states, np.int64)
                selected_states.append(states)
                selected_pages.append(int(option.pages))
                features.append(exact_feature(
                    exact_responses[router_rank], states, c["qmetric_features"],
                    c["split_state_output"], c["proxy"], c["beta"],
                ))
            combined = sum(
                (weights[index] * feature for index, feature in enumerate(features)),
                np.zeros_like(features[0]),
            )
            damage = float(combined @ combined)
            recovery = 1.0 - damage / base_damage
            oracle = c["oracle"][(request, position, layer, budget)]
            oracle_states = oracle["states"]
            canonical_relative_error = abs(base_damage - oracle["base_damage"]) / oracle["base_damage"]
            oracle_features = [
                exact_feature(
                    exact_responses[index], oracle_states[index], c["qmetric_features"],
                    c["split_state_output"], c["proxy"], c["beta"],
                )
                for index in range(8)
            ]
            oracle_combined = sum(
                (weights[index] * feature for index, feature in enumerate(oracle_features)),
                np.zeros_like(oracle_features[0]),
            )
            rescored_oracle_damage = float(oracle_combined @ oracle_combined)
            oracle_score_relative_error = abs(rescored_oracle_damage - oracle["damage"]) / max(oracle["damage"], 1e-30)
            oracle_score_base_relative_error = abs(rescored_oracle_damage - oracle["damage"]) / base_damage
            if canonical_relative_error > 2e-5 or oracle_score_base_relative_error > 5e-7:
                raise RuntimeError(
                    f"canonical qenergy audit failed {canonical_relative_error}/{oracle_score_relative_error}"
                )
            state_overlap = float(np.mean([
                np.mean(selected_states[index] == oracle_states[index]) for index in range(8)
            ]))
            jaccard = float(np.mean([
                page_overlap(
                    selected_states[index], oracle_states[index],
                    c["state_gate"], c["state_up"], c["state_down"],
                )
                for index in range(8)
            ]))
            page_count_match = float(np.mean([
                selected_pages[index] == oracle["pages"][index] for index in range(8)
            ]))
            rows.append({
                "method": method,
                "request_id": request,
                "target_position": position,
                "source_position": position - HORIZON,
                "layer": layer,
                "mean_budget_pages": budget,
                "overall_bpw": RATES[budget],
                "exact_qenergy_damage": damage,
                "base_qenergy_damage": base_damage,
                "qenergy_recovery": recovery,
                "oracle_exact_qenergy_damage": rescored_oracle_damage,
                "oracle_qenergy_recovery": 1.0 - rescored_oracle_damage / base_damage,
                "actual_group_pages": int(sum(selected_pages)),
                "oracle_actual_group_pages": int(sum(oracle["pages"])),
                "state_overlap": state_overlap,
                "page_jaccard": jaccard,
                "expert_page_count_match": page_count_match,
                "selected_states": [value.tolist() for value in selected_states],
                "selected_pages": selected_pages,
                "oracle_states": [value.tolist() for value in oracle_states],
                "oracle_pages": oracle["pages"],
                "canonical_base_relative_error": canonical_relative_error,
                "canonical_oracle_damage_relative_error": oracle_score_relative_error,
                "canonical_oracle_damage_base_relative_error": oracle_score_base_relative_error,
            })
    return rows


def load_oracle(canonical: Path) -> dict[tuple[str, int, int, int], dict[str, Any]]:
    groups = pd.read_parquet(canonical / "average_rate_group_frontier.parquet")
    experts = pd.read_parquet(canonical / "average_rate_expert_allocation.parquet")
    filters = (
        (groups["factor_config_id"] == PRIMARY_FACTOR)
        & (groups["allocation_policy"] == PRIMARY_POLICY)
        & (groups["burst_cap_pages_per_expert"] == BURST_CAP)
        & groups["mean_budget_pages_per_expert"].isin(BUDGETS)
        & (groups["position"] >= HORIZON)
    )
    group_frame = groups[filters].copy()
    expert_filters = (
        (experts["factor_config_id"] == PRIMARY_FACTOR)
        & (experts["allocation_policy"] == PRIMARY_POLICY)
        & (experts["burst_cap_pages_per_expert"] == BURST_CAP)
        & experts["mean_budget_pages_per_expert"].isin(BUDGETS)
        & (experts["position"] >= HORIZON)
    )
    expert_frame = experts[expert_filters].copy()
    if len(group_frame) != 80 * 3 or len(expert_frame) != 80 * 3 * 8:
        raise RuntimeError(f"canonical H4 subset changed: {len(group_frame)}/{len(expert_frame)}")
    result = {}
    for _, group in group_frame.iterrows():
        key = (
            str(group.request_id), int(group.position), int(group.layer),
            int(group.mean_budget_pages_per_expert),
        )
        local = expert_frame[
            (expert_frame.request_id == key[0])
            & (expert_frame.position == key[1])
            & (expert_frame.layer == key[2])
            & (expert_frame.mean_budget_pages_per_expert == key[3])
        ].sort_values("router_rank")
        if len(local) != 8:
            raise RuntimeError(f"canonical expert allocation width changed: {key}")
        result[key] = {
            "damage": float(group.group_exact_qenergy_damage),
            "base_damage": float(group.group_base_qenergy_damage),
            "recovery": float(group.group_recovery),
            "states": [np.asarray(json.loads(value), np.int64) for value in local.selected_states],
            "pages": [int(value) for value in local.selected_pages],
            "experts": [int(value) for value in local.expert_id],
        }
    return result


def allocate_layer(
    layer: int,
    dataset: Mapping[str, Any],
    runtime: Mapping[str, Any],
    predictions: torch.Tensor,
    exact_data: Mapping[str, np.ndarray],
    factor_dir: Path,
    config: Mapping[str, Any],
    oracle: Mapping[tuple[str, int, int, int], Any],
    output: Path,
    recorder: Recorder,
    workers: int,
    runner: Any,
    base: Any,
) -> list[dict[str, Any]]:
    from oracle_study.average_rate_allocator import (
        allocation_aware_rate_frontiers, lagrangian_rate_frontier, qmetric_features,
    )
    from oracle_study.neuron_selector import unit_score_metadata
    from oracle_study.split_interaction_field import (
        SPLIT_STATE_DOWN_HIGH, SPLIT_STATE_GATE_HIGH, SPLIT_STATE_UP_HIGH,
        SplitProjectionResponses, build_split_interaction_field, split_state_output,
    )

    local = runner.prior.layer_view(exact_data, layer)
    proxy, proxy_facts = base.proxy_gradients(
        local["xplus"], [local[f"h{h}_router_logits"] for h in range(1, 5)], local["split"],
    )
    beta = float(proxy_facts["beta"])
    arrays = dict(np.load(factor_dir / f"average_rate_factor_layer_{layer}.npz", allow_pickle=False))
    active = [int(value) for value in runtime["active_experts"]]
    active_index = {expert: index for index, expert in enumerate(active)}
    factors, abc = {}, {}
    for expert in active:
        factors[expert] = runner._load_layer_factors(arrays, config, expert)[PRIMARY_FACTOR]
        index = active_index[expert]
        abc[expert] = unit_score_metadata(
            runtime["down2"][index].numpy(), runtime["down4"][index].numpy(),
            proxy=proxy, beta=beta,
        )
    target_components = components_from_hidden(dataset["exact_hidden"].float())
    learned_hidden = hidden_from_components(predictions.float()).numpy().astype(np.float64)
    parent_hidden = dataset["parent_hidden"].numpy().astype(np.float64)
    validation_indices = [index for index, value in enumerate(dataset["split"]) if value == "validation"]
    if len(validation_indices) != 20:
        raise RuntimeError(f"layer {layer} validation group count changed")

    global _ALLOC_CONTEXT
    _ALLOC_CONTEXT = {
        "dataset": dataset,
        "validation_indices": validation_indices,
        "exact_hidden": dataset["exact_hidden"].numpy().astype(np.float64),
        "learned_hidden": learned_hidden,
        "parent_hidden": parent_hidden,
        "down2": runtime["down2"].numpy().astype(np.float64),
        "down4": runtime["down4"].numpy().astype(np.float64),
        "active_index": active_index,
        "factors": factors,
        "abc": abc,
        "proxy": np.asarray(proxy, np.float64),
        "beta": beta,
        "config": config,
        "oracle": oracle,
        "SplitProjectionResponses": SplitProjectionResponses,
        "build_split_interaction_field": build_split_interaction_field,
        "split_state_output": split_state_output,
        "qmetric_features": qmetric_features,
        "lagrangian_rate_frontier": lagrangian_rate_frontier,
        "allocation_aware_rate_frontiers": allocation_aware_rate_frontiers,
        "state_gate": SPLIT_STATE_GATE_HIGH,
        "state_up": SPLIT_STATE_UP_HIGH,
        "state_down": SPLIT_STATE_DOWN_HIGH,
    }
    started = time.time()
    context = mp.get_context("fork")
    pool_workers = min(workers, len(validation_indices))
    with context.Pool(pool_workers) as pool:
        results = pool.map(evaluate_allocation_group, range(len(validation_indices)))
    rows = [row for group in results for row in group]
    path = output / f"allocation_layer_{layer}.parquet"
    pd.DataFrame(rows).drop(columns=["selected_states", "oracle_states"]).to_parquet(path, index=False)
    states_path = output / f"allocation_states_layer_{layer}.json"
    atomic_json(states_path, rows)
    recorder(
        "allocation_layer_completed", layer=layer, rows=len(rows),
        seconds=time.time() - started, parquet_sha256=sha256(path),
        states_sha256=sha256(states_path), workers=pool_workers,
    )
    _ALLOC_CONTEXT = None
    del target_components, learned_hidden, parent_hidden, arrays, factors, abc
    gc.collect()
    return rows


def aggregate_component_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Per-layer records are already useful; the final overall table is computed
    # directly in main from concatenated routed tensors.
    return records


def summarize_allocations(frame: pd.DataFrame) -> list[dict[str, Any]]:
    result = []
    for (method, budget), local in frame.groupby(["method", "mean_budget_pages"], sort=True):
        recovered = local["base_qenergy_damage"] - local["exact_qenergy_damage"]
        oracle_recovered = local["base_qenergy_damage"] - local["oracle_exact_qenergy_damage"]
        retained = float(recovered.sum() / oracle_recovered.sum())
        regret = float((local["exact_qenergy_damage"] - local["oracle_exact_qenergy_damage"]).sum() / oracle_recovered.sum())
        result.append({
            "method": method,
            "mean_budget_pages": int(budget),
            "overall_bpw": float(local["overall_bpw"].iloc[0]),
            "groups": len(local),
            "oracle_utility_retained": retained,
            "normalized_allocation_regret": regret,
            "qenergy_recovery_p10": float(local["qenergy_recovery"].quantile(0.10)),
            "qenergy_recovery_median": float(local["qenergy_recovery"].median()),
            "oracle_qenergy_recovery_p10": float(local["oracle_qenergy_recovery"].quantile(0.10)),
            "oracle_qenergy_recovery_median": float(local["oracle_qenergy_recovery"].median()),
            "page_jaccard_mean": float(local["page_jaccard"].mean()),
            "state_overlap_mean": float(local["state_overlap"].mean()),
            "expert_page_count_match_mean": float(local["expert_page_count_match"].mean()),
            "maximum_oracle_rescore_relative_error": float(local["canonical_oracle_damage_relative_error"].max()),
            "maximum_oracle_rescore_base_relative_error": float(local["canonical_oracle_damage_base_relative_error"].max()),
        })
    return result


def verify_inputs(args: argparse.Namespace, runner_path: Path, allocator_path: Path) -> dict[str, Any]:
    observed = {
        "capture": sha256(args.exact_capture),
        "trees": sha256(args.trees),
        "config": sha256(args.config),
        "runner": sha256(runner_path),
        "allocator": sha256(allocator_path),
    }
    for name, expected in EXPECTED_HASHES.items():
        if observed[name] != expected:
            raise RuntimeError(f"frozen input changed: {name} {observed[name]} != {expected}")
    parent_manifest = args.parent_root / "capture_manifest.json"
    return {
        "hashes": observed,
        "parent_manifest": str(parent_manifest),
        "parent_manifest_sha256": sha256(parent_manifest),
        "checkpoint_config_sha256": sha256(args.checkpoint / "config.json"),
        "checkpoint_index_sha256": sha256(args.checkpoint / "model.safetensors.index.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-root", type=Path, default=Path("/workspace/set_utility_study/experiments/adaptive_expert_precision_oracle"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/workspace/qwen36_mxfp4_candidate"))
    parser.add_argument("--trees", type=Path, default=Path("/workspace/codebook_granularity_study/locked/selected_trees.json"))
    parser.add_argument("--exact-capture", type=Path, default=Path("/workspace/codebook_granularity_study/inputs/qwen36_exact_confirm_captures.npz"))
    parser.add_argument("--parent-root", type=Path, default=Path("/workspace/embedded_parent_capture_20260821_v1"))
    parser.add_argument("--pr13-results", type=Path, default=Path("/workspace/set_utility_study/results/qwen36_mxfp4_average_rate_allocation_frontier_closure_20260821_v2"))
    parser.add_argument("--config", type=Path, default=Path("/workspace/set_utility_study/experiments/adaptive_expert_precision_oracle/configs/qwen36_mxfp4_average_rate_allocation.json"))
    parser.add_argument("--output", type=Path, default=Path("/workspace/pr13_h4_response_field_pilot_20260822_v1"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    recorder = Recorder(args.output / "run.jsonl")
    started = time.time()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")

    scripts = args.oracle_root / "scripts"
    source = args.oracle_root / "src"
    sys.path[:0] = [str(source), str(scripts)]
    import run_average_rate_allocation as runner
    import run_set_utility_distillation as prior
    import run_sparse_streaming_study as base
    from run_mxfp4_selective_pages import tree_from_record

    provenance = verify_inputs(
        args,
        scripts / "run_average_rate_allocation.py",
        source / "oracle_study/average_rate_allocator.py",
    )
    with args.config.open() as handle:
        config = json.load(handle)
    runner._validate_contract(config)
    recorder(
        "pilot_started", pid=os.getpid(), gpu=torch.cuda.get_device_name(0),
        torch=torch.__version__, layers=LAYERS, budgets=BUDGETS,
        true_activation_is_input=False, branch_conditional=True, provenance=provenance,
    )
    exact_data = prior.load_capture_admitted(args.exact_capture, ["train", "validation"])
    if "test" in set(np.asarray(exact_data["split"]).astype(str)):
        raise RuntimeError("test row entered pilot")
    checkpoint_index = json.loads(
        (args.checkpoint / "model.safetensors.index.json").read_text()
    )["weight_map"]
    tree_records = json.loads(args.trees.read_text())
    trees = {name: tree_from_record(tree_records[name]) for name in ("gate", "up", "down")}

    dataset_records = []
    for layer in LAYERS:
        path = args.output / f"dataset_layer_{layer}.pt"
        runtime_path = args.output / f"allocator_runtime_layer_{layer}.pt"
        if args.resume and path.is_file() and runtime_path.is_file():
            record = {
                "layer": layer, "dataset": str(path), "dataset_sha256": sha256(path),
                "runtime": str(runtime_path), "runtime_sha256": sha256(runtime_path),
                "resumed": True,
            }
            recorder("dataset_layer_resumed", **record)
        else:
            record = build_layer_dataset(
                layer, exact_data, args.parent_root, args.checkpoint,
                checkpoint_index, trees, args.output, recorder, device, prior,
            )
        dataset_records.append(record)
    atomic_json(args.output / "dataset_manifest.json", dataset_records)

    training_records = []
    for layer in LAYERS:
        checkpoint_path = args.output / f"model_layer_{layer}.pt"
        predictions_path = args.output / f"predicted_components_layer_{layer}.pt"
        if args.resume and checkpoint_path.is_file() and predictions_path.is_file():
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            record = {
                "layer": layer, "best_epoch": int(state["best_epoch"]),
                "parameter_count": int(state["parameter_count"]),
                "fit": state["fit_metrics"], "tune": state["tune_metrics"],
                "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
                "predictions": str(predictions_path), "predictions_sha256": sha256(predictions_path),
                "resumed": True,
            }
            recorder("training_resumed", **record)
        else:
            dataset = torch.load(
                args.output / f"dataset_layer_{layer}.pt", map_location="cpu", weights_only=False,
            )
            record = train_layer_model(layer, dataset, args.output, recorder, device)
            del dataset
            gc.collect()
        training_records.append(record)
    atomic_json(args.output / "training_summary.json", training_records)

    oracle = load_oracle(args.pr13_results / "validation_exact_h4")
    all_component_rows = []
    component_tensors: dict[tuple[str, str], torch.Tensor] = defaultdict(lambda: torch.zeros(5, dtype=torch.float64))
    field_ablation = defaultdict(lambda: torch.zeros(2, dtype=torch.float64))
    allocation_rows = []
    for layer in LAYERS:
        dataset = torch.load(
            args.output / f"dataset_layer_{layer}.pt", map_location="cpu", weights_only=False,
        )
        runtime = torch.load(
            args.output / f"allocator_runtime_layer_{layer}.pt", map_location="cpu", weights_only=False,
        )
        predictions = torch.load(
            args.output / f"predicted_components_layer_{layer}.pt", map_location="cpu", weights_only=True,
        ).float()
        target = components_from_hidden(dataset["exact_hidden"].float())
        parent_components = components_from_hidden(dataset["parent_hidden"].float())
        q2_components = torch.zeros_like(target)
        q2_components[:, :, 0] = dataset["parent_hidden"][:, :, 0]
        validation = torch.tensor([index for index, value in enumerate(dataset["split"]) if value == "validation"])
        routed = dataset["expert_ids"][validation]
        for method, value in (
            ("learned_response", predictions),
            ("parent_activation_q4_oracle", parent_components),
            ("resident_q2_zero_refinement", q2_components),
        ):
            layer_rows, accumulators = metric_rows(target, value, validation, routed, method)
            for row in layer_rows:
                row["layer"] = layer
            all_component_rows.extend(layer_rows)
            for component, accumulator in accumulators.items():
                component_tensors[(method, component)] += accumulator

        rows_index = torch.arange(len(validation))[:, None]
        truth_routed = target[validation][rows_index, routed].double()
        pred_routed = predictions[validation][rows_index, routed].double()
        exact_hidden = hidden_from_components(truth_routed)
        predicted_hidden = hidden_from_components(pred_routed)
        base_error = (predicted_hidden - exact_hidden).square().sum()
        field_ablation["baseline"] += torch.tensor([base_error, exact_hidden.square().sum()])
        for component in range(4):
            ablated = pred_routed.clone()
            ablated[:, :, component] = truth_routed[:, :, component]
            ablated_hidden = hidden_from_components(ablated)
            field_ablation[COMPONENTS[component]] += torch.tensor([
                (ablated_hidden - exact_hidden).square().sum(), exact_hidden.square().sum(),
            ])

        rows = allocate_layer(
            layer, dataset, runtime, predictions, exact_data,
            args.pr13_results / "fit", config, oracle, args.output,
            recorder, args.workers, runner, base,
        )
        allocation_rows.extend(rows)
        del dataset, runtime, predictions, target, parent_components, q2_components
        gc.collect()

    overall_components = []
    for (method, component), values in sorted(component_tensors.items()):
        cosine_sum, vector_count, error, energy, element_count = values
        overall_components.append({
            "method": method,
            "component": component,
            "vectors": 80 * 8,
            "cosine": float(cosine_sum / vector_count.clamp_min(1)),
            "mse": float(error / element_count),
            "relative_mse": float(error / energy.clamp_min(1e-30)),
            "target_rms": float((energy / element_count).sqrt()),
        })
    component_frame = pd.DataFrame(all_component_rows)
    component_frame.to_parquet(args.output / "component_metrics_by_layer.parquet", index=False)
    pd.DataFrame(overall_components).to_csv(args.output / "component_metrics_overall.csv", index=False)
    ablation_summary = {}
    baseline_error = float(field_ablation["baseline"][0])
    for name, values in field_ablation.items():
        relative = float(values[0] / values[1].clamp_min(1e-30))
        ablation_summary[name] = {
            "reconstructed_hidden_relative_mse": relative,
            "error_reduction_vs_learned": (
                0.0 if name == "baseline" else 1.0 - float(values[0]) / max(baseline_error, 1e-30)
            ),
        }

    allocation_frame = pd.DataFrame(allocation_rows)
    allocation_frame.drop(columns=["selected_states", "oracle_states"]).to_parquet(
        args.output / "allocation_results.parquet", index=False,
    )
    allocation_summary = summarize_allocations(allocation_frame)
    atomic_json(args.output / "allocation_summary.json", allocation_summary)
    atomic_json(args.output / "field_component_oracle_substitution.json", ablation_summary)

    result = {
        "schema": "pr13_h4_response_field_pilot_result_v1",
        "completed": True,
        "bounded_scope": {
            "layers": list(LAYERS),
            "horizon": HORIZON,
            "branch_conditional": True,
            "train_groups": 184,
            "validation_groups": 80,
            "test_rows_admitted_or_used": False,
            "architecture_sweep": False,
            "factor": PRIMARY_FACTOR,
            "policy": PRIMARY_POLICY,
            "budgets": list(BUDGETS),
        },
        "provenance": provenance,
        "training": training_records,
        "component_metrics": overall_components,
        "field_component_oracle_substitution": ablation_summary,
        "allocation": allocation_summary,
        "wall_seconds": time.time() - started,
    }
    atomic_json(args.output / "result.json", result)
    manifest = {}
    for path in sorted(args.output.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            manifest[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    atomic_json(args.output / "artifact_manifest.json", manifest)
    recorder(
        "pilot_completed", wall_seconds=time.time() - started,
        result_sha256=sha256(args.output / "result.json"),
        artifact_manifest_sha256=sha256(args.output / "artifact_manifest.json"),
    )


if __name__ == "__main__":
    main()
