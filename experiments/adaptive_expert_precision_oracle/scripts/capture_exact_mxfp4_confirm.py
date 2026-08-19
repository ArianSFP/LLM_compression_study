#!/usr/bin/env python3
"""Small deterministic text-only capture from the exact MXFP4 checkpoint.

This confirmation path loads the checkpoint through Transformers'
compressed-tensors integration with ``dequantize=True``.  Thus the numerical
weights are decoded directly from the checkpoint's packed E2M1 leaf codes and
E8M0 scales; no dequantize/requantize conversion is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, CompressedTensorsConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForConditionalGeneration


PROMPTS = [
    "Explain why a binary search needs a sorted input in two concise paragraphs.",
    "Write a Python function that merges two already sorted integer lists.",
    "A train travels 135 km in 90 minutes. Compute its average speed in km/h and show the arithmetic.",
    "Compare optimistic and pessimistic concurrency control for a small database.",
    "What causes the seasons on Earth? Correct the common distance-from-the-Sun misconception.",
    "Design three edge cases for a UTF-8 streaming parser.",
    "Summarize the difference between precision and recall using a medical screening example.",
    "Give a short proof that the square root of two is irrational.",
    "Create a SQL query that returns the top three products by revenue in each category.",
    "Explain how a page cache can reduce storage reads but increase tail latency after eviction.",
    "Translate the following requirement into pseudocode: retry twice with exponential backoff, then fail closed.",
    "List the major assumptions behind ordinary least squares and one symptom of each violation.",
]


def split_for(index: int) -> str:
    return "train" if index < 6 else ("validation" if index < 9 else "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 4, 20, 39])
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--cpu-offload", action="store_true",
        help="dispatch overflow layers to CPU while preserving prompts, hooks, and split",
    )
    parser.add_argument(
        "--cpu-only", action="store_true",
        help="load the dequantized graph wholly on CPU (avoids Accelerate offload hooks)",
    )
    parser.add_argument("--max-gpu-memory", default="20GiB")
    args = parser.parse_args()
    started = time.time()
    full_config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    quantization = CompressedTensorsConfig.from_dict(full_config.quantization_config)
    quantization.dequantize = True
    full_config.text_config._attn_implementation = "eager"
    full_config.text_config._experts_implementation = "eager"
    device_map: Any = {"": 0}
    maximum_memory = None
    if args.cpu_only:
        device_map = {"": "cpu"}
    elif args.cpu_offload:
        device_map = "auto"
        maximum_memory = {0: args.max_gpu_memory, "cpu": "256GiB"}
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        args.checkpoint, config=full_config, dtype=torch.bfloat16, local_files_only=True,
        low_cpu_mem_usage=True, device_map=device_map, max_memory=maximum_memory,
        quantization_config=quantization,
    )
    model.eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    selected_layers = set(args.layers)
    captured: dict[int, dict[str, torch.Tensor]] = {}
    handles = []
    for layer_id, layer in enumerate(model.model.language_model.layers):
        if layer_id not in selected_layers:
            continue
        captured[layer_id] = {}
        def norm_hook(_module: Any, _inputs: Any, output: torch.Tensor, layer_id: int = layer_id) -> None:
            captured[layer_id]["x"] = output.detach().float().cpu()
        def gate_hook(_module: Any, _inputs: Any, output: torch.Tensor, layer_id: int = layer_id) -> None:
            captured[layer_id]["router_logits"] = output[0].detach().float().cpu().unsqueeze(0)
        def layer_hook(_module: Any, _inputs: Any, output: Any, layer_id: int = layer_id) -> None:
            value = output[0] if isinstance(output, tuple) else output
            captured[layer_id]["xplus"] = value.detach().float().cpu()
        handles.extend([
            layer.post_attention_layernorm.register_forward_hook(norm_hook),
            layer.mlp.gate.register_forward_hook(gate_hook),
            layer.register_forward_hook(layer_hook),
        ])
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for request_index, prompt in enumerate(PROMPTS):
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length, add_special_tokens=True)
            input_device = model.get_input_embeddings().weight.device
            encoded = {key: value.to(input_device) for key, value in encoded.items()}
            captured = {layer: {} for layer in selected_layers}
            outputs = model(**encoded, use_cache=False, return_dict=True)
            tokens = encoded["input_ids"][0].cpu().numpy()
            length = int(tokens.shape[0])
            if length <= 4:
                raise RuntimeError("prompt too short for H1-H4 confirmation")
            for layer in sorted(selected_layers):
                item = captured[layer]
                if set(item) != {"x", "router_logits", "xplus"}:
                    raise RuntimeError(f"incomplete hooks for layer {layer}: {item.keys()}")
                x = item["x"][0].numpy(); logits = item["router_logits"][0].numpy(); xplus = item["xplus"][0].numpy()
                probabilities = torch.softmax(torch.from_numpy(logits), dim=-1)
                weights, ids = torch.topk(probabilities, 8, dim=-1)
                weights = weights / weights.sum(dim=-1, keepdim=True)
                for position in range(length - 4):
                    prefix = tokens[:position + 1].tobytes()
                    row = {
                        "sequence_id": f"mxfp4-confirm-{request_index:03d}", "request_id": f"mxfp4-confirm-{request_index:03d}",
                        "position": position, "layer": layer, "block_type": str(getattr(model.model.language_model.layers[layer], "block_type", "unknown")),
                        "split": split_for(request_index), "prefix_hash": hashlib.sha256(prefix).hexdigest(),
                        "x": x[position], "xplus": xplus[position], "router_logits": logits[position],
                        "expert_ids": ids[position].numpy(), "router_weights": weights[position].numpy(),
                    }
                    for horizon in range(1, 5):
                        row[f"h{horizon}_xplus"] = xplus[position + horizon]
                        row[f"h{horizon}_router_logits"] = logits[position + horizon]
                        row[f"h{horizon}_expert_ids"] = ids[position + horizon].numpy()
                    rows.append(row)
            print(json.dumps({"captured_request": request_index, "tokens": length, "rows": len(rows)}), flush=True)
    for handle in handles:
        handle.remove()
    scalar = ("sequence_id", "request_id", "position", "layer", "block_type", "split", "prefix_hash")
    arrays: dict[str, np.ndarray] = {name: np.asarray([row[name] for row in rows]) for name in scalar}
    tensor_names = ("x", "xplus", "router_logits", "expert_ids", "router_weights") + tuple(f"h{h}_{name}" for h in range(1, 5) for name in ("xplus", "router_logits", "expert_ids"))
    for name in tensor_names:
        arrays[name] = np.stack([row[name] for row in rows])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    manifest = {
        "schema": "exact_mxfp4_transformers_confirm_v1", "checkpoint": str(args.checkpoint), "revision": args.revision,
        "config_sha256": sha256(args.checkpoint / "config.json"), "index_sha256": sha256(args.checkpoint / "model.safetensors.index.json"),
        "loader": "Transformers compressed-tensors dequantize=True; exact packed leaf decode, no requantization",
        "device_map": (
            "CPU resident" if args.cpu_only
            else ("auto CPU offload" if args.cpu_offload else "single CUDA device")
        ),
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "host": platform.node(),
        "layers": sorted(selected_layers), "requests": len(PROMPTS), "rows": len(rows),
        "split_requests": {name: sum(split_for(i) == name for i in range(len(PROMPTS))) for name in ("train", "validation", "test")},
        "split_rows": {name: sum(row["split"] == name for row in rows) for name in ("train", "validation", "test")},
        "started_unix": started, "completed_unix": time.time(),
    }
    manifest["wall_seconds"] = manifest["completed_unix"] - started
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
