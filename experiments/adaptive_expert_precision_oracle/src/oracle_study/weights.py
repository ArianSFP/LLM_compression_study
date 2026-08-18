"""Load exact BF16 source and production GGUF expert matrices read-only."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open


def _hf_tensor(hf_root: Path, tensor_name: str) -> torch.Tensor:
    index = json.loads((hf_root / "model.safetensors.index.json").read_text())
    shard = hf_root / index["weight_map"][tensor_name]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(tensor_name)


def load_hf_experts(hf_root: Path, layer: int, expert_ids: list[int]) -> dict[str, np.ndarray]:
    gate_up_name = f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"
    down_name = f"model.language_model.layers.{layer}.mlp.experts.down_proj"
    gate_up = _hf_tensor(hf_root, gate_up_name)[expert_ids].float().numpy()
    down = _hf_tensor(hf_root, down_name)[expert_ids].float().numpy()
    if gate_up.shape[1:] != (1024, 2048) or down.shape[1:] != (2048, 512):
        raise ValueError(f"unexpected expert shapes gate_up={gate_up.shape} down={down.shape}")
    result = {
        "gate": gate_up[:, :512].copy(),
        "up": gate_up[:, 512:].copy(),
        "down": down.copy(),
    }
    del gate_up, down
    gc.collect()
    return result


def load_gguf_experts(reader: Any, layer: int, expert_ids: list[int], gguf_module: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    names = {
        "gate": f"blk.{layer}.ffn_gate_exps.weight",
        "up": f"blk.{layer}.ffn_up_exps.weight",
        "down": f"blk.{layer}.ffn_down_exps.weight",
    }
    by_name = {tensor.name: tensor for tensor in reader.tensors}
    result: dict[str, np.ndarray] = {}
    facts: dict[str, Any] = {}
    for projection, name in names.items():
        tensor = by_name[name]
        decoded = gguf_module.dequantize(tensor.data, tensor.tensor_type)
        selected = np.asarray(decoded[expert_ids], dtype=np.float32).copy()
        expected = (len(expert_ids), 512, 2048) if projection in ("gate", "up") else (len(expert_ids), 2048, 512)
        if selected.shape != expected:
            raise ValueError(f"GGUF orientation mismatch {name}: {selected.shape} != {expected}")
        result[projection] = selected
        facts[projection] = {
            "tensor_name": name,
            "quantization_type_id": int(tensor.tensor_type),
            "quantization_type": str(tensor.tensor_type).split(".")[-1],
            "tensor_storage_bytes": int(tensor.data.nbytes),
            "bytes_per_expert": int(tensor.data.nbytes // 256),
            "effective_bpw": float(8 * tensor.data.nbytes / (256 * np.prod(expected[1:]))),
        }
        del decoded, selected
        gc.collect()
    return result, facts


def alignment_statistics(bf16: dict[str, np.ndarray], q4: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    facts: dict[str, dict[str, float]] = {}
    for name in ("gate", "up", "down"):
        source = bf16[name].astype(np.float64).reshape(-1)
        quant = q4[name].astype(np.float64).reshape(-1)
        delta = quant - source
        facts[name] = {
            "bf16_q4_cosine": float(np.dot(source, quant) / max(np.linalg.norm(source) * np.linalg.norm(quant), 1e-30)),
            "relative_rmse": float(np.sqrt(np.mean(delta**2)) / max(np.sqrt(np.mean(source**2)), 1e-30)),
            "max_abs_error": float(np.max(np.abs(delta))),
        }
    return facts


def expert_output(weights: dict[str, np.ndarray], x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gate = weights["gate"] @ x
    up = weights["up"] @ x
    hidden = (gate / (1.0 + np.exp(-gate))) * up
    output = weights["down"] @ hidden
    return gate, up, hidden, output
