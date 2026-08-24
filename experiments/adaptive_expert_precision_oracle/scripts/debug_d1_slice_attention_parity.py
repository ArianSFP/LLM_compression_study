#!/usr/bin/env python3
"""Diagnose cached full-attention D1 slice parity without allocating pages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(EXPERIMENT / "scripts")]

from oracle_study.d1_decode import clone_decode_cache  # noqa: E402
from oracle_study.d1_layer_slice import load_d1_layer_slice  # noqa: E402
from run_d1_tail_same_host_allocation_patch import (  # noqa: E402
    _capture_record,
    _hidden_prefix,
    _load_capture,
)


def _route(
    model_slice,
    current: torch.Tensor,
    prefix,
    position: int,
    attention_mask: torch.Tensor | None,
) -> np.ndarray:
    cache = clone_decode_cache(prefix)
    residual = current
    normalized = model_slice.next_input_norm(current)
    positions = torch.full(
        (4, 1, 1), int(position), device=current.device, dtype=torch.long,
    )
    mixed, _ = model_slice.next_mixer(
        hidden_states=normalized,
        position_embeddings=model_slice.next_rotary(current, positions[1:]),
        attention_mask=attention_mask,
        position_ids=positions[0],
        past_key_values=cache,
    )
    logits = model_slice.next_router(
        model_slice.next_post_norm(residual + mixed)
    )[0]
    return logits.reshape(-1).detach().float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--group", type=int, required=True)
    args = parser.parse_args()

    capture = _load_capture(args.capture_dir, args.layer)
    record = _capture_record(capture, args.group)
    request_id = str(capture["request_id"][record])
    position = int(capture["position"][record])
    model_slice = load_d1_layer_slice(args.checkpoint, args.layer, device="cuda")
    if model_slice.next_mixer_type != "full_attention":
        raise RuntimeError("diagnostic requires a full-attention D1 layer")
    with torch.no_grad():
        prefix = model_slice.decode_prefix_cache(
            _hidden_prefix(capture, request_id, position)
        )
        current = model_slice.compose_current_output(
            capture["residual"][record],
            capture["x"][record],
            capture["routed"][record],
        ).reshape(1, 1, -1)
        reference = np.asarray(capture["d1_router_logits"][record], np.float32)
        zero_bf16 = torch.zeros(
            (1, 1, 1, position + 1),
            dtype=torch.bfloat16,
            device=current.device,
        )
        zero_float32 = zero_bf16.float()
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            create_causal_mask,
        )
        positions = torch.full(
            (1, 1), position, device=current.device, dtype=torch.long,
        )
        created = create_causal_mask(
            config=model_slice.next_config,
            inputs_embeds=current,
            attention_mask=torch.ones(
                (1, position + 1), device=current.device, dtype=torch.long,
            ),
            past_key_values=clone_decode_cache(prefix),
            position_ids=positions,
        )
        variants = {
            "none": None,
            "zero_bf16": zero_bf16,
            "zero_float32": zero_float32,
            "create_causal_mask": created,
        }
        results = {}
        for name, mask in variants.items():
            observed = _route(model_slice, current, prefix, position, mask)
            difference = observed.astype(np.float64) - reference.astype(np.float64)
            results[name] = {
                "max_abs": float(np.max(np.abs(difference), initial=0.0)),
                "mse": float(np.mean(difference * difference)),
                "bit_identical": bool(np.array_equal(observed, reference)),
                "route_order_equal": bool(np.array_equal(
                    np.argsort(-observed, kind="stable")[:8],
                    np.asarray(capture["d1_router_ids"][record], np.int64),
                )),
                "mask_shape": None if mask is None else list(mask.shape),
                "mask_dtype": None if mask is None else str(mask.dtype),
                "mask_min": None if mask is None else float(mask.min()),
                "mask_max": None if mask is None else float(mask.max()),
            }
    print(json.dumps({
        "layer": args.layer,
        "group": args.group,
        "request_id": request_id,
        "position": position,
        "variants": results,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
