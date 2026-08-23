#!/usr/bin/env python3
"""Capture PR #13 prompts with a damage-ranked static Q4 resident set."""

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
    parser.add_argument("--resident-count", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cpu-memory", default="118GiB")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 < args.resident_count < 256:
        raise ValueError("resident-count must be in [1, 255]")
    capture._validate_immutable_prompt_contract()
    started = time.time()
    checkpoint = capture.verify_checkpoint_metadata(args.checkpoint)
    damage = torch.load(args.damage_scores, map_location="cpu", weights_only=True)
    if tuple(damage.shape) != (40, 256):
        raise RuntimeError(f"damage score shape changed: {tuple(damage.shape)}")
    resident = {
        (layer, int(expert))
        for layer in range(40)
        for expert in torch.topk(damage[layer], args.resident_count).indices
    }
    convert = {
        (layer, expert)
        for layer in range(40)
        for expert in range(256)
        if (layer, expert) not in resident
    }
    model, tokenizer, device_mode, load_memory = capture._load_model(
        args.checkpoint,
        cpu_only=True,
        cpu_offload=False,
        max_gpu_memory="20GiB",
        max_cpu_memory=args.max_cpu_memory,
    )
    conversion = capture.replace_routed_experts_with_embedded_parent(
        model, args.checkpoint, convert_experts=convert,
    )
    rows = capture._capture_rows(model, tokenizer, max_length=64)
    shards, identity_hash = capture.validate_and_shard_rows(rows)
    conversion.update({
        "resident_experts_per_layer": args.resident_count,
        "resident_fraction": args.resident_count / 256,
        "effective_expert_bpw_including_parent_scales": 2.25 + 2 * args.resident_count / 256,
        "selection": "per-layer training-only Q4-to-Q2 activation-damage ranking",
    })
    manifest = capture.publish_capture(
        args.output,
        shards,
        identity_grid_sha256=identity_hash,
        checkpoint=checkpoint,
        runtime_provenance=capture._runtime_facts(device_mode),
        hardware=capture._hardware_facts(),
        model_load_memory=load_memory,
        parent_conversion=conversion,
        started_unix=started,
    )
    print(json.dumps({
        "event": "resident_capture_complete",
        "resident_count": args.resident_count,
        "identity_grid_sha256": manifest["identity_grid_sha256"],
        "wall_seconds": time.time() - started,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
