#!/usr/bin/env python3
"""Align a resident rollout capture to PR #13 H1 datasets and score state MSE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def load_capture(root: Path, layer: int) -> dict[tuple[str, int, str], np.ndarray]:
    result: dict[tuple[str, int, str], np.ndarray] = {}
    for split in ("train", "validation"):
        with np.load(root / f"{split}_layer_{layer:02d}.npz", allow_pickle=False) as data:
            for index in range(len(data["position"])):
                key = (
                    str(data["request_id"][index]),
                    int(data["position"][index]),
                    str(data["prefix_hash"][index]),
                )
                if key in result:
                    raise RuntimeError(f"duplicate identity {key}")
                result[key] = np.asarray(data["x"][index], np.float32)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--layers", default="0-39")
    args = parser.parse_args()
    layers = range(40) if args.layers == "0-39" else [int(value) for value in args.layers.split(",")]
    rows = []
    for layer in layers:
        dataset = torch.load(
            args.dataset_root / f"dataset_layer_{layer}.pt",
            map_location="cpu", weights_only=False,
        )
        captured = load_capture(args.capture_root, layer)
        resident = torch.from_numpy(np.stack([
            captured[(str(request), int(position), str(prefix))]
            for request, position, prefix in zip(
                dataset["request_id"], dataset["target_position"], dataset["prefix_hash"]
            )
        ]))
        exact = dataset["exact_target_activation_auxiliary"].float()
        parent = dataset["parent_target_x"].float()
        for split in ("train", "validation"):
            selected = torch.tensor([value == split for value in dataset["split"]])
            exact_local = exact[selected]
            parent_local = parent[selected]
            resident_local = resident[selected]
            target_square = float(exact_local.square().mean())
            row = {
                "layer": layer,
                "split": split,
                "rows": int(selected.sum()),
                "target_mean_square": target_square,
                "q2_parent_mse": float((parent_local - exact_local).square().mean()),
                "resident_parent_mse": float((resident_local - exact_local).square().mean()),
                "resident_relative_mse": float((resident_local - exact_local).square().mean() / target_square),
            }
            row["mse_reduction_fraction"] = 1 - row["resident_parent_mse"] / row["q2_parent_mse"]
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    for split in ("train", "validation"):
        local = [row for row in rows if row["split"] == split]
        summary = {
            "event": "summary",
            "split": split,
            "layers": len(local),
            "q2_parent_mse_mean": float(np.mean([row["q2_parent_mse"] for row in local])),
            "resident_parent_mse_mean": float(np.mean([row["resident_parent_mse"] for row in local])),
            "resident_parent_mse_p90_layer": float(np.quantile([row["resident_parent_mse"] for row in local], 0.90)),
        }
        summary["mse_reduction_fraction"] = 1 - summary["resident_parent_mse_mean"] / summary["q2_parent_mse_mean"]
        print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
