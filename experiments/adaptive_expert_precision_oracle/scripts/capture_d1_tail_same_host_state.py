#!/usr/bin/env python3
"""Capture compact PRO same-host states for D1 allocation regeneration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(EXPERIMENT / "scripts")]

from oracle_study.d1_decode_tail import (  # noqa: E402
    sha256,
    validate_cached_decode_tail_config,
)
from run_d1_downstream_tail_kl import _gpu_gate, _policy_banks  # noqa: E402
from run_same_host_causal_controls import (  # noqa: E402
    _encoded_requests,
    _load_model,
    atomic_json,
)


SCHEMA = "pr13_d1_tail_same_host_allocation_capture_v1"
FACTS = "d1_tail_same_host_allocation_capture_facts.json"


@dataclass
class CaptureObservation:
    hidden: dict[int, torch.Tensor]
    x: dict[int, torch.Tensor]
    residual: dict[int, torch.Tensor]
    routed: dict[int, torch.Tensor]
    router_logits: dict[int, torch.Tensor]
    router_scores: dict[int, torch.Tensor]
    router_ids: dict[int, torch.Tensor]
    cache: Any
    elapsed_seconds: float


class AllocationCaptureObserver:
    """Capture only the six allocation layers during exact cached decode."""

    def __init__(self, model: Any, layers: list[int]) -> None:
        self.layers = tuple(map(int, layers))
        self.handles = []
        modules = model.model.language_model.layers
        for layer_id in self.layers:
            layer = modules[layer_id]

            def layer_hook(
                _module: Any,
                _inputs: Any,
                output: Any,
                layer_id: int = layer_id,
            ) -> None:
                value = output[0] if isinstance(output, tuple) else output
                self.hidden[layer_id] = value.detach().cpu().clone()

            def norm_hook(
                _module: Any,
                inputs: Any,
                output: torch.Tensor,
                layer_id: int = layer_id,
            ) -> None:
                self.x[layer_id] = output.detach().cpu().clone()
                self.residual[layer_id] = inputs[0].detach().cpu().clone()

            def routed_hook(
                _module: Any,
                _inputs: Any,
                output: torch.Tensor,
                layer_id: int = layer_id,
            ) -> None:
                self.routed[layer_id] = output.detach().cpu().clone()

            def gate_hook(
                _module: Any,
                _inputs: Any,
                output: Any,
                layer_id: int = layer_id,
            ) -> None:
                if not isinstance(output, tuple) or len(output) < 3:
                    raise RuntimeError("router output contract changed")
                self.router_logits[layer_id] = output[0].detach().cpu().clone()
                self.router_scores[layer_id] = output[1].detach().cpu().clone()
                self.router_ids[layer_id] = output[2].detach().cpu().clone()

            self.handles.append(layer.register_forward_hook(layer_hook))
            self.handles.append(
                layer.post_attention_layernorm.register_forward_hook(norm_hook)
            )
            self.handles.append(layer.mlp.experts.register_forward_hook(routed_hook))
            self.handles.append(layer.mlp.gate.register_forward_hook(gate_hook))
        self.reset()

    def reset(self) -> None:
        self.hidden = {}
        self.x = {}
        self.residual = {}
        self.routed = {}
        self.router_logits = {}
        self.router_scores = {}
        self.router_ids = {}

    def run(
        self,
        model: Any,
        encoded: Mapping[str, torch.Tensor],
        position: int,
        prefix_cache: Any | None,
    ) -> CaptureObservation:
        self.reset()
        device = encoded["input_ids"].device
        position = int(position)
        started = time.perf_counter()
        with torch.inference_mode():
            output = model(
                input_ids=encoded["input_ids"][:, position : position + 1],
                attention_mask=encoded["attention_mask"][:, : position + 1],
                past_key_values=prefix_cache,
                cache_position=torch.as_tensor(
                    [position], device=device, dtype=torch.long,
                ),
                use_cache=True,
                return_dict=True,
            )
        torch.cuda.synchronize()
        expected = set(self.layers)
        for name, values in (
            ("hidden", self.hidden),
            ("x", self.x),
            ("residual", self.residual),
            ("routed", self.routed),
            ("router_logits", self.router_logits),
            ("router_scores", self.router_scores),
            ("router_ids", self.router_ids),
        ):
            if set(values) != expected:
                raise RuntimeError(f"incomplete same-host capture for {name}")
        return CaptureObservation(
            hidden=self.hidden,
            x=self.x,
            residual=self.residual,
            routed=self.routed,
            router_logits=self.router_logits,
            router_scores=self.router_scores,
            router_ids=self.router_ids,
            cache=output.past_key_values,
            elapsed_seconds=time.perf_counter() - started,
        )

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("configuration must be a JSON object")
    return value


def _flat(value: torch.Tensor, dtype: np.dtype[Any]) -> np.ndarray:
    return value.reshape(-1, value.shape[-1]).float().numpy().astype(dtype)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def capture(
    config: Mapping[str, Any],
    config_path: Path,
    checkpoint: Path,
    output: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    hardware = _gpu_gate(config)
    _, _, _, banks = _policy_banks(config)
    layers = list(map(int, config["injection_layers"]))
    observed_layers = sorted(set(layers + [layer + 1 for layer in layers]))
    rates = list(map(int, config["page_caps"]))
    references = {layer: banks[(layer, rates[0])] for layer in layers}
    identity = references[layers[0]].identity
    group_lookup = {
        (str(row.request_id), int(row.position)): int(row.group)
        for row in identity.itertuples(index=False)
    }

    model, tokenizer, load_seconds = _load_model(checkpoint)
    encoded = _encoded_requests(tokenizer, model, config)
    observer = AllocationCaptureObserver(model, observed_layers)
    layer_rows: dict[int, list[dict[str, Any]]] = {
        layer: [] for layer in layers
    }
    baseline_seconds = 0.0
    mismatch_rows = []
    try:
        for request_id in map(str, config["validation_request_ids"]):
            request = encoded[request_id]
            prefix = None
            final_position = int(
                config["position_mask"]["admitted_positions_per_request"][request_id]
            )
            for position in range(final_position + 1):
                observation = observer.run(model, request, position, prefix)
                baseline_seconds += observation.elapsed_seconds
                admitted = position > 0
                group = group_lookup[(request_id, position)] if admitted else -1
                for layer in layers:
                    live_ids = _flat(
                        observation.router_ids[layer], np.float32,
                    ).astype(np.int64).reshape(8)
                    record = {
                        "request_id": request_id,
                        "position": position,
                        "group": group,
                        "admitted": admitted,
                        "hidden": _flat(observation.hidden[layer], np.float32).reshape(2048),
                        "x": _flat(observation.x[layer], np.float32).reshape(2048),
                        "residual": _flat(observation.residual[layer], np.float32).reshape(2048),
                        "routed": _flat(observation.routed[layer], np.float32).reshape(2048),
                        "router_logits": _flat(
                            observation.router_logits[layer], np.float32,
                        ).reshape(256),
                        "router_scores": _flat(
                            observation.router_scores[layer], np.float32,
                        ).reshape(8),
                        "router_ids": live_ids,
                        "d1_router_logits": _flat(
                            observation.router_logits[layer + 1], np.float32,
                        ).reshape(256),
                        "d1_router_scores": _flat(
                            observation.router_scores[layer + 1], np.float32,
                        ).reshape(8),
                        "d1_router_ids": _flat(
                            observation.router_ids[layer + 1], np.float32,
                        ).astype(np.int64).reshape(8),
                    }
                    layer_rows[layer].append(record)
                    if admitted:
                        source_ids = np.asarray(
                            references[layer].expert_ids[group], np.int64,
                        )
                        if set(source_ids.tolist()) != set(live_ids.tolist()):
                            mismatch_rows.append({
                                "layer": layer,
                                "group": group,
                                "request_id": request_id,
                                "position": position,
                                "source_ids": source_ids.tolist(),
                                "live_ids": live_ids.tolist(),
                            })
                prefix = observation.cache
                del observation
            gc.collect()
    finally:
        observer.close()
        del encoded, tokenizer, model
        gc.collect()
        torch.cuda.empty_cache()

    output.mkdir(parents=True, exist_ok=True)
    files = {}
    expected_records = sum(
        int(value) + 1
        for value in config["position_mask"]["admitted_positions_per_request"].values()
    )
    for layer, rows in layer_rows.items():
        if len(rows) != expected_records:
            raise RuntimeError("same-host allocation capture grid is incomplete")
        path = output / f"same_host_allocation_layer_{layer:02d}.npz"
        _atomic_npz(
            path,
            schema=np.asarray(SCHEMA),
            layer=np.asarray(layer, np.int64),
            request_id=np.asarray([row["request_id"] for row in rows]),
            position=np.asarray([row["position"] for row in rows], np.int64),
            group=np.asarray([row["group"] for row in rows], np.int64),
            admitted=np.asarray([row["admitted"] for row in rows], bool),
            hidden=np.stack([row["hidden"] for row in rows]).astype(np.float32),
            x=np.stack([row["x"] for row in rows]).astype(np.float32),
            residual=np.stack([row["residual"] for row in rows]).astype(np.float32),
            routed=np.stack([row["routed"] for row in rows]).astype(np.float32),
            router_logits=np.stack(
                [row["router_logits"] for row in rows],
            ).astype(np.float32),
            router_scores=np.stack(
                [row["router_scores"] for row in rows],
            ).astype(np.float32),
            router_ids=np.stack([row["router_ids"] for row in rows]).astype(np.int64),
            d1_router_logits=np.stack(
                [row["d1_router_logits"] for row in rows],
            ).astype(np.float32),
            d1_router_scores=np.stack(
                [row["d1_router_scores"] for row in rows],
            ).astype(np.float32),
            d1_router_ids=np.stack(
                [row["d1_router_ids"] for row in rows],
            ).astype(np.int64),
        )
        files[path.name] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "records": len(rows),
        }
    facts = {
        "completed": True,
        "schema": SCHEMA,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "checkpoint_index_sha256": sha256(
            checkpoint / "model.safetensors.index.json"
        ),
        "layers": layers,
        "observed_layers": observed_layers,
        "requests": list(map(str, config["validation_request_ids"])),
        "records_per_layer": expected_records,
        "admitted_groups_per_layer": int(
            config["position_mask"]["expected_groups_per_layer"]
        ),
        "source_route_set_mismatches": len(mismatch_rows),
        "mismatches": mismatch_rows,
        "files": files,
        "model_load_seconds": load_seconds,
        "baseline_seconds": baseline_seconds,
        "wall_seconds": time.perf_counter() - started,
        "exact_prefix_progression": True,
        "candidate_state_or_delta_used": False,
        "environment": {
            **hardware,
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "test_rows_admitted_or_used": False,
    }
    atomic_json(output / FACTS, facts)
    return facts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_json(args.config)
    validate_cached_decode_tail_config(config)
    print(json.dumps(
        capture(config, args.config, args.checkpoint, args.output),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
