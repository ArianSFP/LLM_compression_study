#!/usr/bin/env python3
"""Fit compact resident H1 Q2-to-exact activation correctors."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


HIDDEN = 2048
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


def metrics(target: torch.Tensor, prediction: torch.Tensor) -> dict[str, float]:
    error = prediction - target
    cosine = (target * prediction).sum(-1) / (target.square().sum(-1).sqrt() * prediction.square().sum(-1).sqrt()).clamp_min(1e-30)
    return {
        "mse": float(error.square().mean()),
        "relative_mse": float(error.square().sum() / target.square().sum().clamp_min(1e-30)),
        "cosine": float(cosine.mean()),
    }


def run_layer(args: argparse.Namespace, layer: int, recorder: Recorder, device: torch.device) -> dict[str, Any]:
    started = time.time()
    request_index = torch.load(args.fitting_root / "request_index.pt", map_location="cpu", weights_only=True)
    available = torch.load(args.fitting_root / "target_available.pt", map_location="cpu", weights_only=True)[:, 0, layer]
    generator = torch.Generator().manual_seed(SEED)
    request_order = torch.randperm(int(request_index.max()) + 1, generator=generator)
    fit_mask = available & torch.isin(request_index, request_order[:224])
    tune_mask = available & torch.isin(request_index, request_order[224:])
    parent_store = torch.load(args.fitting_root / "parent_target.pt", map_location="cpu", mmap=True, weights_only=True)
    exact_store = torch.load(args.fitting_root / "exact_target.pt", map_location="cpu", mmap=True, weights_only=True)
    fit_parent = parent_store[fit_mask, 0, layer].float().to(device)
    tune_parent = parent_store[tune_mask, 0, layer].float().to(device)
    fit_exact = exact_store[fit_mask, 0, layer].float().to(device)
    tune_exact = exact_store[tune_mask, 0, layer].float().to(device)
    del parent_store, exact_store

    exact_data = load_layer(args.exact_root, layer)
    parent_data = load_layer(args.parent_root, layer)
    parent_ids = identity_map(parent_data)
    rows = np.flatnonzero((exact_data["position"] >= 1) & (np.asarray(exact_data["split"]).astype(str) == "validation"))
    validation_exact = torch.tensor(exact_data["x"][rows], dtype=torch.float32, device=device)
    validation_parent = torch.tensor(np.stack([
        parent_data["x"][parent_ids[(str(exact_data["request_id"][i]), int(exact_data["position"][i]), str(exact_data["prefix_hash"][i]))]]
        for i in rows
    ]), dtype=torch.float32, device=device)

    fit_mean = fit_parent.mean(0)
    residual = fit_exact - fit_parent
    residual_mean = residual.mean(0)
    x = fit_parent - fit_mean
    y = residual - residual_mean
    tune_x = tune_parent - fit_mean
    validation_x = validation_parent - fit_mean
    results = [{
        "name": "uncorrected_parent",
        "rank": 0,
        "resident_bytes_per_layer": 0,
        "tune": metrics(tune_exact, tune_parent),
        "validation": metrics(validation_exact, validation_parent),
    }]

    # Tiny diagonal affine control with shrink selected only on fitting tune data.
    diagonal = (x * y).sum(0) / x.square().sum(0).clamp_min(1e-12)
    best_diag = None
    for shrink in (0.0, 0.25, 0.5, 0.75, 1.0):
        tune_prediction = tune_parent + tune_x * diagonal * shrink + residual_mean * shrink
        tune_metric = metrics(tune_exact, tune_prediction)
        if best_diag is None or tune_metric["mse"] < best_diag[0]:
            best_diag = (tune_metric["mse"], shrink, tune_metric)
    assert best_diag is not None
    shrink = best_diag[1]
    validation_prediction = validation_parent + validation_x * diagonal * shrink + residual_mean * shrink
    results.append({
        "name": "diagonal_affine",
        "rank": HIDDEN,
        "shrink": shrink,
        "resident_bytes_per_layer": HIDDEN * 2 * 2,
        "macs_per_activation": HIDDEN,
        "tune": best_diag[2],
        "validation": metrics(validation_exact, validation_prediction),
    })
    recorder("diagonal_completed", layer=layer, result=results[-1])

    # Output PCA supplies a compact residual decoder. Ridge fits its input map.
    maximum_rank = max(args.ranks)
    _, singular, output_basis = torch.pca_lowrank(y, q=maximum_rank, center=False, niter=5)
    total_variance = y.square().sum()
    gram = x.T @ x
    gram_scale = torch.trace(gram) / HIDDEN
    identity = torch.eye(HIDDEN, device=device)
    target_coordinates = y @ output_basis
    cross = x.T @ target_coordinates
    solved = {}
    for alpha in args.alphas:
        solved[alpha] = torch.linalg.solve(gram + identity * (gram_scale * alpha), cross)
        recorder("ridge_solved", layer=layer, alpha=alpha)
    for rank in args.ranks:
        basis_q = output_basis[:, :rank].half().float()
        best = None
        for alpha in args.alphas:
            encoder_q = solved[alpha][:, :rank].half().float()
            tune_prediction = tune_parent + (tune_x @ encoder_q) @ basis_q.T + residual_mean.half().float()
            tune_metric = metrics(tune_exact, tune_prediction)
            if best is None or tune_metric["mse"] < best[0]:
                best = (tune_metric["mse"], alpha, tune_metric, encoder_q)
        assert best is not None
        validation_prediction = validation_parent + (validation_x @ best[3]) @ basis_q.T + residual_mean.half().float()
        record = {
            "name": f"lowrank_residual_r{rank}",
            "rank": rank,
            "alpha": best[1],
            "captured_residual_variance": float(singular[:rank].square().sum() / total_variance),
            "resident_bytes_per_layer": int((HIDDEN * rank * 2 + HIDDEN) * 2),
            "macs_per_activation": int(2 * HIDDEN * rank),
            "tune": best[2],
            "validation": metrics(validation_exact, validation_prediction),
        }
        results.append(record)
        recorder("rank_completed", layer=layer, result=record)
    result = {
        "layer": layer,
        "fit_rows": int(fit_mask.sum()),
        "tune_rows": int(tune_mask.sum()),
        "validation_rows": len(rows),
        "results": results,
        "seconds": time.time() - started,
    }
    recorder("layer_completed", **result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default="14")
    parser.add_argument("--ranks", default="32,64,128,256,384,512")
    parser.add_argument("--alphas", default="0.001,0.01,0.1,1,10")
    parser.add_argument("--fitting-root", type=Path, default=Path("/workspace/harp_fitting_mxfp4_parent_pairs_v1"))
    parser.add_argument("--exact-root", type=Path, default=Path("/workspace/all_layer_split_interaction_study/capture"))
    parser.add_argument("--parent-root", type=Path, default=Path("/workspace/embedded_parent_capture_20260821_v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.ranks = tuple(int(value) for value in args.ranks.split(","))
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
    recorder("screen_started", layers=layers, ranks=args.ranks, alphas=args.alphas, gpu=torch.cuda.get_device_name(0))
    results = [run_layer(args, layer, recorder, device) for layer in layers]
    atomic_json(args.output / "result.json", {"schema": "h1_resident_state_corrector_screen_v1", "layers": results})
    recorder("screen_completed", result=str(args.output / "result.json"))


if __name__ == "__main__":
    main()
