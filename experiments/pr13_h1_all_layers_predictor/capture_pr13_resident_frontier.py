#!/usr/bin/env python3
"""Capture several damage-ranked PR #13 resident rollouts from one model load."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import capture_embedded_parent as capture


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--damage-scores", type=Path, required=True)
    parser.add_argument("--resident-counts", default="96,64")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-cpu-memory", default="118GiB")
    args = parser.parse_args()
    counts = sorted({int(value) for value in args.resident_counts.split(",")}, reverse=True)
    if not counts or any(not 0 < value < 256 for value in counts):
        raise ValueError("resident counts must be unique integers in [1, 255]")
    outputs = [args.output_root / f"resident{count:03d}" for count in counts]
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite a resident frontier output")
    capture._validate_immutable_prompt_contract()
    started = time.time()
    checkpoint = capture.verify_checkpoint_metadata(args.checkpoint)
    damage = torch.load(args.damage_scores, map_location="cpu", weights_only=True)
    if tuple(damage.shape) != (40, 256):
        raise RuntimeError(f"damage score shape changed: {tuple(damage.shape)}")
    rankings = [torch.argsort(damage[layer], descending=True).tolist() for layer in range(40)]
    model, tokenizer, device_mode, load_memory = capture._load_model(
        args.checkpoint,
        cpu_only=True,
        cpu_offload=False,
        max_gpu_memory="20GiB",
        max_cpu_memory=args.max_cpu_memory,
    )
    previous = {(layer, expert) for layer in range(40) for expert in range(256)}
    for count, output in zip(counts, outputs):
        resident = {
            (layer, int(expert))
            for layer, ranking in enumerate(rankings)
            for expert in ranking[:count]
        }
        conversion = capture.replace_routed_experts_with_embedded_parent(
            model, args.checkpoint, convert_experts=previous - resident,
        )
        rows = capture._capture_rows(model, tokenizer, max_length=64)
        shards, identity_hash = capture.validate_and_shard_rows(rows)
        conversion.update({
            "resident_experts_per_layer": count,
            "resident_fraction": count / 256,
            "effective_expert_bpw_including_parent_scales": 2.25 + 2 * count / 256,
            "selection": "per-layer training-only Q4-to-Q2 activation-damage ranking",
        })
        manifest = capture.publish_capture(
            output,
            shards,
            identity_grid_sha256=identity_hash,
            checkpoint=checkpoint,
            runtime_provenance=capture._runtime_facts(device_mode),
            hardware=capture._hardware_facts(),
            model_load_memory=load_memory,
            parent_conversion=conversion,
            started_unix=started,
        )
        previous = resident
        print(json.dumps({
            "event": "resident_frontier_point_complete",
            "resident_count": count,
            "identity_grid_sha256": manifest["identity_grid_sha256"],
            "wall_seconds": time.time() - started,
        }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
