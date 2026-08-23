#!/usr/bin/env python3
"""Measure resident-rollout top-8 route overlap on aligned PR #13 H1 rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--damage-scores", type=Path, required=True)
    parser.add_argument("--resident-count", type=int, required=True)
    args = parser.parse_args()
    dataset = torch.load(args.dataset, map_location="cpu", weights_only=False)
    layer = int(dataset["layer"])
    captured = {}
    for split in ("train", "validation"):
        with np.load(args.capture_root / f"{split}_layer_{layer:02d}.npz", allow_pickle=False) as data:
            for index in range(len(data["position"])):
                key = (str(data["request_id"][index]), int(data["position"][index]), str(data["prefix_hash"][index]))
                captured[key] = np.asarray(data["router_logits"][index], np.float32)
    logits = torch.from_numpy(np.stack([
        captured[(str(request), int(position), str(prefix))]
        for request, position, prefix in zip(dataset["request_id"], dataset["target_position"], dataset["prefix_hash"])
    ]))
    predicted = torch.topk(logits, 8, dim=-1).indices
    exact = dataset["expert_ids"]
    resident = torch.topk(
        torch.load(args.damage_scores, map_location="cpu", weights_only=True)[layer],
        args.resident_count,
    ).indices
    validation = torch.tensor([value == "validation" for value in dataset["split"]])
    predicted, exact = predicted[validation], exact[validation]
    overlap = torch.stack([torch.isin(predicted[row], exact[row]).sum() for row in range(len(exact))])
    exact_resident = torch.isin(exact, resident)
    predicted_resident = torch.isin(predicted, resident)
    payload_slots = 17 * 8
    nonresident = 8 - predicted_resident.sum(dim=-1)
    result = {
        "layer": layer,
        "validation_rows": len(exact),
        "route_set_overlap_mean": float(overlap.float().mean() / 8),
        "route_set_overlap_p10": float(torch.quantile(overlap.float() / 8, 0.10)),
        "exact_resident_slot_rate": float(exact_resident.float().mean()),
        "predicted_resident_slot_rate": float(predicted_resident.float().mean()),
        "predicted_nonresident_routes_mean": float(nonresident.float().mean()),
        "blocks_per_predicted_nonresident_mean": float(torch.minimum(torch.full_like(nonresident, 64), payload_slots // nonresident.clamp_min(1)).float().mean()),
        "rows_with_no_nonresident_predicted_routes": int((nonresident == 0).sum()),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
