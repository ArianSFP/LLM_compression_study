#!/usr/bin/env python3
"""GPU causal impulse replay for the PR #13 all-layer reconstruction audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(EXPERIMENT / "scripts")]

from oracle_study.causal_replay import (  # noqa: E402
    cosine_distance,
    mean_squared_error,
    rms_normalize,
    route_metrics,
    token_quality_metrics,
)
from capture_exact_mxfp4_confirm import PROMPTS  # noqa: E402
from run_causal_downstream_replay import (  # noqa: E402
    AUDIT_FACTS,
    AUDIT_MANIFEST,
    LAYERS,
    load_json,
    sha256,
    validate_config,
)


SCHEMA = "pr13_causal_single_layer_impulse_v1"
BASELINE_FILE = "baseline_reference.npz"
BASELINE_FACTS = "baseline_replay_facts.json"
PROPAGATION_LAYER = "impulse_propagation_layer_{layer:02d}.parquet"
QUALITY_LAYER = "impulse_quality_layer_{layer:02d}.parquet"
IMPULSE_SIDECAR = "impulse_layer_{layer:02d}.json"
IMPULSE_FAILURE = "impulse_layer_{layer:02d}.failure.json"
IMPULSE_FACTS = "impulse_run_facts.json"
IMPULSE_PROPAGATION = "impulse_propagation.parquet"
IMPULSE_QUALITY = "impulse_quality.parquet"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def atomic_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(list(rows)).to_parquet(temporary, index=False)
    temporary.replace(path)


def runtime_provenance() -> dict[str, Any]:
    return {
        "host": platform.node(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_driver": torch.cuda.driver_version() if hasattr(torch.cuda, "driver_version") else None,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
    }


def _request_number(request_id: str) -> int:
    prefix = "mxfp4-confirm-"
    if not str(request_id).startswith(prefix):
        raise ValueError(f"unexpected request ID {request_id}")
    value = int(str(request_id)[len(prefix):])
    if value < 0 or value >= len(PROMPTS):
        raise ValueError(f"request ID is outside prompt source: {request_id}")
    return value


def _load_model(checkpoint: Path) -> tuple[Any, Any, float]:
    from transformers import AutoConfig, AutoTokenizer, CompressedTensorsConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeForConditionalGeneration,
    )

    started = time.perf_counter()
    full_config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    quantization = CompressedTensorsConfig.from_dict(full_config.quantization_config)
    quantization.dequantize = True
    full_config.text_config._attn_implementation = "eager"
    full_config.text_config._experts_implementation = "eager"
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        checkpoint,
        config=full_config,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        device_map={"": 0},
        quantization_config=quantization,
    )
    model.eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    torch.cuda.synchronize()
    return model, tokenizer, time.perf_counter() - started


class ForwardObserver:
    """Capture complete layer outputs and router logits for one forward."""

    def __init__(self, model: Any) -> None:
        self.hidden: dict[int, torch.Tensor] = {}
        self.router: dict[int, torch.Tensor] = {}
        self.handles = []
        layers = model.model.language_model.layers
        for layer_id, layer in enumerate(layers):
            def layer_hook(
                _module: Any,
                _inputs: Any,
                output: torch.Tensor,
                layer_id: int = layer_id,
            ) -> None:
                self.hidden[layer_id] = output.detach().float().cpu()

            def gate_hook(
                _module: Any,
                _inputs: Any,
                output: Any,
                layer_id: int = layer_id,
            ) -> None:
                logits = output[0] if isinstance(output, tuple) else output
                self.router[layer_id] = logits.detach().float().cpu()

            self.handles.append(layer.register_forward_hook(layer_hook))
            self.handles.append(layer.mlp.gate.register_forward_hook(gate_hook))

    def run(self, model: Any, encoded: Mapping[str, torch.Tensor]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray, float]:
        self.hidden = {}
        self.router = {}
        started = time.perf_counter()
        with torch.inference_mode():
            output = model(**encoded, use_cache=False, return_dict=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if set(self.hidden) != set(LAYERS) or set(self.router) != set(LAYERS):
            raise RuntimeError("forward hooks did not capture all forty layers")
        hidden = {
            layer: self.hidden[layer][0].numpy()
            for layer in LAYERS
        }
        router = {
            layer: self.router[layer].reshape(encoded["input_ids"].shape[1], -1).numpy()
            for layer in LAYERS
        }
        logits = output.logits[0].detach().float().cpu().numpy()
        del output
        return hidden, router, logits, elapsed

    def run_tail(
        self,
        model: Any,
        hidden_states: torch.Tensor,
        start_layer: int,
        context: Mapping[str, Any],
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray, float]:
        """Continue after ``start_layer`` from a captured complete layer output."""

        self.hidden = {}
        self.router = {}
        text_model = model.model.language_model
        started = time.perf_counter()
        with torch.inference_mode():
            value = hidden_states
            for layer_id in range(start_layer + 1, len(text_model.layers)):
                value = text_model.layers[layer_id](
                    value,
                    position_embeddings=context["position_embeddings"],
                    attention_mask=context["causal_masks"][
                        text_model.config.layer_types[layer_id]
                    ],
                    position_ids=context["text_position_ids"],
                    past_key_values=None,
                    use_cache=False,
                )
            normalized = text_model.norm(value)
            logits = model.lm_head(normalized)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        expected = set(range(start_layer + 1, len(text_model.layers)))
        if set(self.hidden) != expected or set(self.router) != expected:
            raise RuntimeError("tail hooks did not capture every downstream layer")
        hidden = {
            layer: self.hidden[layer][0].numpy()
            for layer in expected
        }
        sequence_length = int(hidden_states.shape[1])
        router = {
            layer: self.router[layer].reshape(sequence_length, -1).numpy()
            for layer in expected
        }
        result_logits = logits[0].detach().float().cpu().numpy()
        del normalized, logits
        return hidden, router, result_logits, elapsed

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def _encoded_requests(tokenizer: Any, model: Any, config: Mapping[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
    result = {}
    device = model.get_input_embeddings().weight.device
    for request_id in config["validation_request_ids"]:
        prompt = PROMPTS[_request_number(request_id)]
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=64,
            add_special_tokens=True,
        )
        result[str(request_id)] = {name: value.to(device) for name, value in encoded.items()}
    return result


def _validate_capture_identity(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    encoded: Mapping[str, Mapping[str, torch.Tensor]],
) -> None:
    manifest = load_json(args.capture_dir / "capture_manifest.json")
    prompt_source = EXPERIMENT / "scripts" / "capture_exact_mxfp4_confirm.py"
    if manifest["source"]["immutable_prompt_source"]["sha256"] != sha256(prompt_source):
        raise RuntimeError("immutable prompt source changed")
    for layer in LAYERS:
        record = manifest["splits"]["validation"][str(layer)]
        path = args.capture_dir / record["file"]
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"capture validation shard {layer} changed")
        with np.load(path, allow_pickle=False) as loaded:
            for request_id in config["validation_request_ids"]:
                mask = loaded["request_id"].astype(str) == str(request_id)
                positions = loaded["position"][mask].astype(np.int64)
                tokens = encoded[str(request_id)]["input_ids"][0].detach().cpu().numpy()
                if len(positions) == 0 or int(positions.max()) + 5 != len(tokens):
                    raise RuntimeError(f"token length does not match capture for {request_id}")
                for position, expected in zip(positions, loaded["prefix_hash"][mask].astype(str)):
                    actual = hashlib.sha256(tokens[: int(position) + 1].tobytes()).hexdigest()
                    if actual != expected:
                        raise RuntimeError("token prefix hash changed")


def _capture_comparison(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    baseline: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    manifest = load_json(args.capture_dir / "capture_manifest.json")
    squared = 0.0
    count = 0
    xplus_max = 0.0
    router_max = 0.0
    exact_routes = 0
    route_rows = 0
    for layer in LAYERS:
        record = manifest["splits"]["validation"][str(layer)]
        with np.load(args.capture_dir / record["file"], allow_pickle=False) as loaded:
            for row in range(len(loaded["position"])):
                request_id = str(loaded["request_id"][row])
                position = int(loaded["position"][row])
                hidden = np.asarray(baseline[request_id]["hidden"][layer][position], np.float64)
                expected_hidden = np.asarray(loaded["xplus"][row], np.float64)
                error = hidden - expected_hidden
                squared += float(error @ error)
                count += error.size
                xplus_max = max(xplus_max, float(np.max(np.abs(error), initial=0.0)))
                logits = np.asarray(baseline[request_id]["router"][layer][position], np.float64)
                expected_logits = np.asarray(loaded["router_logits"][row], np.float64)
                router_max = max(
                    router_max,
                    float(np.max(np.abs(logits - expected_logits), initial=0.0)),
                )
                ids = np.argsort(logits, kind="stable")[-8:][::-1]
                exact_routes += int(np.array_equal(ids, loaded["expert_ids"][row]))
                route_rows += 1
    return {
        "capture_xplus_rmse": float(np.sqrt(squared / count)),
        "capture_xplus_max_abs": xplus_max,
        "capture_router_max_abs": router_max,
        "capture_route_exact_fraction": exact_routes / float(route_rows),
        "capture_rows": float(route_rows),
    }


def _zero_hook(_module: Any, _inputs: Any, output: torch.Tensor) -> torch.Tensor:
    return output + torch.zeros_like(output)


def baseline_phase(args: argparse.Namespace) -> None:
    config = args.config_data
    facts_path = args.output / BASELINE_FACTS
    reference_path = args.output / BASELINE_FILE
    if facts_path.exists() or reference_path.exists():
        if not facts_path.is_file() or not reference_path.is_file():
            raise RuntimeError("partial baseline replay artifact exists")
        facts = load_json(facts_path)
        if facts.get("completed") is not True or facts.get("reference_sha256") != sha256(reference_path):
            raise RuntimeError("baseline replay resume artifact changed")
        return
    reconstruction = load_json(args.reconstruction_dir / AUDIT_FACTS)
    if reconstruction.get("completed") is not True or reconstruction.get("causal_replay_admitted") is not True:
        raise RuntimeError("reconstruction audit has not admitted causal replay")
    _validate_reconstruction_identity(args, config, reconstruction)
    model, tokenizer, load_seconds = _load_model(args.checkpoint)
    encoded = _encoded_requests(tokenizer, model, config)
    _validate_capture_identity(args, config, encoded)
    observer = ForwardObserver(model)
    baseline: dict[str, dict[str, Any]] = {}
    forward_seconds = {}
    try:
        for request_id, request in encoded.items():
            hidden, router, logits, elapsed = observer.run(model, request)
            baseline[request_id] = {
                "hidden": np.stack([hidden[layer] for layer in LAYERS]),
                "router": np.stack([router[layer] for layer in LAYERS]),
                "logits": logits,
                "input_ids": request["input_ids"][0].detach().cpu().numpy(),
            }
            forward_seconds[request_id] = elapsed
        comparison = _capture_comparison(args, config, baseline)
        zero_handles = [
            layer.register_forward_hook(_zero_hook, prepend=True)
            for layer in model.model.language_model.layers
        ]
        no_op_max = 0.0
        no_op_router_max = 0.0
        no_op_logit_max = 0.0
        try:
            for request_id, request in encoded.items():
                hidden, router, logits, _ = observer.run(model, request)
                no_op_max = max(
                    no_op_max,
                    max(
                        float(np.max(np.abs(hidden[layer] - baseline[request_id]["hidden"][layer]), initial=0.0))
                        for layer in LAYERS
                    ),
                )
                no_op_router_max = max(
                    no_op_router_max,
                    max(
                        float(np.max(np.abs(router[layer] - baseline[request_id]["router"][layer]), initial=0.0))
                        for layer in LAYERS
                    ),
                )
                no_op_logit_max = max(
                    no_op_logit_max,
                    float(np.max(np.abs(logits - baseline[request_id]["logits"]), initial=0.0)),
                )
        finally:
            for handle in zero_handles:
                handle.remove()
        arrays: dict[str, np.ndarray] = {}
        for request_id, values in baseline.items():
            suffix = f"request_{_request_number(request_id):03d}"
            arrays[f"{suffix}_input_ids"] = values["input_ids"]
            arrays[f"{suffix}_hidden"] = values["hidden"]
            arrays[f"{suffix}_router_logits"] = values["router"]
            arrays[f"{suffix}_final_logits"] = values["logits"]
        atomic_npz(reference_path, arrays)
        no_op_limit = float(config["no_op_replay_max_abs_atol"])
        gates = {
            "capture_xplus_rmse": comparison["capture_xplus_rmse"]
            <= float(config["baseline_capture_xplus_rmse_atol"]),
            "capture_xplus_max_abs": comparison["capture_xplus_max_abs"]
            <= float(config["baseline_capture_xplus_max_abs_atol"]),
            "capture_router_max_abs": comparison["capture_router_max_abs"]
            <= float(config["baseline_capture_router_max_abs_atol"]),
            "capture_route_exact_fraction": comparison["capture_route_exact_fraction"]
            >= float(config["baseline_capture_route_exact_fraction_min"]),
            "no_op_hidden": no_op_max <= no_op_limit,
            "no_op_router": no_op_router_max <= no_op_limit,
            "no_op_logits": no_op_logit_max <= no_op_limit,
        }
        facts = {
            "completed": True,
            "schema": SCHEMA,
            "run_id": config["run_id"],
            "phase": "baseline-replay",
            **comparison,
            "no_op_hidden_max_abs": no_op_max,
            "no_op_router_max_abs": no_op_router_max,
            "no_op_logit_max_abs": no_op_logit_max,
            "gates": gates,
            "replay_admitted": bool(all(gates.values())),
            "reference_file": reference_path.name,
            "reference_bytes": reference_path.stat().st_size,
            "reference_sha256": sha256(reference_path),
            "model_load_seconds": load_seconds,
            "forward_seconds": forward_seconds,
            "cuda_allocated_gib": torch.cuda.memory_allocated() / 2**30,
            "cuda_reserved_gib": torch.cuda.memory_reserved() / 2**30,
            "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "test_scientific_rows_admitted_or_used": False,
            "runtime_provenance": runtime_provenance(),
        }
        atomic_json(facts_path, facts)
    finally:
        observer.close()


def _validate_reconstruction_identity(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    facts: Mapping[str, Any],
) -> None:
    if facts.get("run_id") != config["run_id"]:
        raise RuntimeError("reconstruction run ID changed")
    for name in (
        "checkpoint_config_sha256",
        "checkpoint_index_sha256",
        "capture_manifest_sha256",
        "selected_tree_sha256",
        "expert_allocation_sha256",
        "group_frontier_sha256",
        "average_rate_run_facts_sha256",
        "factor_manifest_sha256",
    ):
        if facts.get(name) != config.get(name):
            raise RuntimeError(f"reconstruction provenance changed for {name}")


def _load_baseline(path: Path, config: Mapping[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    with np.load(path, allow_pickle=False) as loaded:
        for request_id in config["validation_request_ids"]:
            suffix = f"request_{_request_number(request_id):03d}"
            result[str(request_id)] = {
                "input_ids": np.asarray(loaded[f"{suffix}_input_ids"]),
                "hidden": np.asarray(loaded[f"{suffix}_hidden"]),
                "router": np.asarray(loaded[f"{suffix}_router_logits"]),
                "logits": np.asarray(loaded[f"{suffix}_final_logits"]),
            }
    return result



def _load_full_capture_sequences(
    capture_dir: Path,
    config: Mapping[str, Any],
    encoded: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, dict[str, np.ndarray]]:
    """Assemble all token positions from base rows plus the final H1--H4 rows."""

    manifest = load_json(capture_dir / "capture_manifest.json")
    result: dict[str, dict[str, np.ndarray]] = {}
    for request_id in config["validation_request_ids"]:
        sequence_length = int(encoded[str(request_id)]["input_ids"].shape[1])
        result[str(request_id)] = {
            "hidden": np.empty(
                (len(LAYERS), sequence_length, int(config["hidden_size"])), np.float32,
            ),
            "router": np.empty(
                (len(LAYERS), sequence_length, 256), np.float32,
            ),
        }
    for layer in LAYERS:
        record = manifest["splits"]["validation"][str(layer)]
        path = capture_dir / record["file"]
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"capture validation shard {layer} changed")
        with np.load(path, allow_pickle=False) as loaded:
            for request_id in config["validation_request_ids"]:
                request_id = str(request_id)
                mask = loaded["request_id"].astype(str) == request_id
                rows = np.flatnonzero(mask)
                order = np.argsort(loaded["position"][rows])
                rows = rows[order]
                positions = np.asarray(loaded["position"][rows], np.int64)
                sequence_length = result[request_id]["hidden"].shape[1]
                expected = np.arange(sequence_length - 4, dtype=np.int64)
                if not np.array_equal(positions, expected):
                    raise RuntimeError(
                        f"capture positions are not a complete H4 prefix for {request_id}",
                    )
                hidden = result[request_id]["hidden"][layer]
                router = result[request_id]["router"][layer]
                hidden[positions] = np.asarray(loaded["xplus"][rows], np.float32)
                router[positions] = np.asarray(loaded["router_logits"][rows], np.float32)
                last = int(rows[-1])
                for horizon in range(1, 5):
                    hidden[int(positions[-1]) + horizon] = np.asarray(
                        loaded[f"h{horizon}_xplus"][last], np.float32,
                    )
                    router[int(positions[-1]) + horizon] = np.asarray(
                        loaded[f"h{horizon}_router_logits"][last], np.float32,
                    )
                    overlap = rows[:-horizon]
                    if len(overlap):
                        if not np.array_equal(
                            loaded[f"h{horizon}_xplus"][overlap],
                            loaded["xplus"][rows[horizon:]],
                        ):
                            raise RuntimeError("xplus horizon overlap changed")
                        if not np.array_equal(
                            loaded[f"h{horizon}_router_logits"][overlap],
                            loaded["router_logits"][rows[horizon:]],
                        ):
                            raise RuntimeError("router horizon overlap changed")
    return result


def _tail_context(
    model: Any,
    encoded: Mapping[str, torch.Tensor],
    hidden_states: torch.Tensor,
) -> dict[str, Any]:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        create_causal_mask,
        create_recurrent_attention_mask,
    )

    text_model = model.model.language_model
    batch, sequence_length = hidden_states.shape[:2]
    position_ids = torch.arange(
        sequence_length, device=hidden_states.device, dtype=torch.long,
    ).view(1, 1, -1).expand(4, batch, -1)
    text_position_ids = position_ids[0]
    rope_position_ids = position_ids[1:]
    mask_kwargs = {
        "config": text_model.config,
        "inputs_embeds": hidden_states,
        "attention_mask": encoded.get("attention_mask"),
        "past_key_values": None,
        "position_ids": text_position_ids,
    }
    return {
        "text_position_ids": text_position_ids,
        "position_embeddings": text_model.rotary_emb(hidden_states, rope_position_ids),
        "causal_masks": {
            "full_attention": create_causal_mask(**mask_kwargs),
            "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
        },
    }

def _impulse_paths(output: Path, layer: int) -> tuple[Path, Path, Path, Path]:
    root = output / "layers"
    return (
        root / PROPAGATION_LAYER.format(layer=layer),
        root / QUALITY_LAYER.format(layer=layer),
        root / IMPULSE_SIDECAR.format(layer=layer),
        root / IMPULSE_FAILURE.format(layer=layer),
    )


def _delta_for_request(
    shard: Mapping[str, np.ndarray],
    rate_index: int,
    request_id: str,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    delta = np.zeros((sequence_length, shard["delta"].shape[-1]), np.float32)
    mask = np.asarray(shard["request_id"]).astype(str) == str(request_id)
    positions = np.asarray(shard["position"][mask], np.int64)
    delta[positions] = np.asarray(shard["delta"][rate_index, mask], np.float32)
    return delta, positions


def impulse_phase(args: argparse.Namespace) -> None:
    config = args.config_data
    baseline_facts = load_json(args.output / BASELINE_FACTS)
    baseline_path = args.output / BASELINE_FILE
    no_op_gates = ("no_op_hidden", "no_op_router", "no_op_logits")
    if not all(baseline_facts.get("gates", {}).get(name) is True for name in no_op_gates):
        raise RuntimeError("same-process no-op gates did not admit paired tail replay")
    if baseline_facts.get("reference_sha256") != sha256(baseline_path):
        raise RuntimeError("baseline reference changed")
    reconstruction = load_json(args.reconstruction_dir / AUDIT_FACTS)
    _validate_reconstruction_identity(args, config, reconstruction)
    reconstruction_manifest = load_json(args.reconstruction_dir / AUDIT_MANIFEST)
    model, tokenizer, load_seconds = _load_model(args.checkpoint)
    encoded = _encoded_requests(tokenizer, model, config)
    _validate_capture_identity(args, config, encoded)
    captured = _load_full_capture_sequences(args.capture_dir, config, encoded)
    device = model.get_input_embeddings().weight.device
    contexts = {}
    for request_id in config["validation_request_ids"]:
        request_id = str(request_id)
        context_hidden = torch.as_tensor(
            captured[request_id]["hidden"][0],
            device=device,
            dtype=torch.bfloat16,
        ).unsqueeze(0)
        contexts[request_id] = _tail_context(
            model, encoded[request_id], context_hidden,
        )
    observer = ForwardObserver(model)
    selected_layers = tuple(args.layers if args.layers else LAYERS)
    try:
        for injection_layer in selected_layers:
            if injection_layer not in LAYERS:
                raise RuntimeError("impulse layer is outside range(40)")
            propagation_path, quality_path, sidecar_path, failure_path = _impulse_paths(
                args.output, injection_layer,
            )
            if failure_path.exists():
                raise RuntimeError(f"prior impulse failure must be reviewed: {failure_path}")
            if propagation_path.exists() or quality_path.exists() or sidecar_path.exists():
                if not all(path.is_file() for path in (propagation_path, quality_path, sidecar_path)):
                    raise RuntimeError("partial impulse artifact exists")
                sidecar = load_json(sidecar_path)
                if (
                    sidecar.get("propagation_sha256") != sha256(propagation_path)
                    or sidecar.get("quality_sha256") != sha256(quality_path)
                ):
                    raise RuntimeError("impulse resume artifact changed")
                continue
            started = time.perf_counter()
            try:
                record = reconstruction_manifest["layers"][str(injection_layer)]
                reconstruction_path = args.reconstruction_dir / record["file"]
                if record["sha256"] != sha256(reconstruction_path):
                    raise RuntimeError("reconstruction delta shard changed")
                with np.load(reconstruction_path, allow_pickle=False) as loaded:
                    shard = {name: np.asarray(loaded[name]) for name in loaded.files}
                rates = np.asarray(shard["mean_budget_pages_per_expert"], np.int64)
                wanted_rates = list(map(int, config["initial_impulse_mean_pages"]))
                rate_indices = {
                    rate: int(np.flatnonzero(rates == rate)[0])
                    for rate in wanted_rates
                }
                propagation_rows: list[dict[str, Any]] = []
                quality_rows: list[dict[str, Any]] = []
                realized_errors = []
                baseline_capture_hidden_max = 0.0
                baseline_capture_route_churn_max = 0.0
                baseline_forward_seconds = 0.0
                candidate_forward_seconds = 0.0
                tail_no_op_hidden_max = 0.0
                tail_no_op_router_max = 0.0
                tail_no_op_logits_max = 0.0
                for request_id_value in config["validation_request_ids"]:
                    request_id = str(request_id_value)
                    request = encoded[request_id]
                    sequence_length = int(request["input_ids"].shape[1])
                    captured_hidden = captured[request_id]["hidden"]
                    captured_router = captured[request_id]["router"]
                    base_start = torch.as_tensor(
                        captured_hidden[injection_layer],
                        device=device,
                        dtype=torch.bfloat16,
                    ).unsqueeze(0)
                    base_start_array = base_start[0].float().cpu().numpy()
                    (
                        base_tail_hidden,
                        base_tail_router,
                        base_logits,
                        elapsed,
                    ) = observer.run_tail(
                        model, base_start, injection_layer, contexts[request_id],
                    )
                    baseline_forward_seconds += elapsed
                    (
                        repeated_tail_hidden,
                        repeated_tail_router,
                        repeated_logits,
                        elapsed,
                    ) = observer.run_tail(
                        model, base_start, injection_layer, contexts[request_id],
                    )
                    baseline_forward_seconds += elapsed
                    hidden_repeat_error = max(
                        (
                            float(
                                np.max(
                                    np.abs(
                                        repeated_tail_hidden[layer]
                                        - base_tail_hidden[layer]
                                    ),
                                    initial=0.0,
                                ),
                            )
                            for layer in base_tail_hidden
                        ),
                        default=0.0,
                    )
                    router_repeat_error = max(
                        (
                            float(
                                np.max(
                                    np.abs(
                                        repeated_tail_router[layer]
                                        - base_tail_router[layer]
                                    ),
                                    initial=0.0,
                                ),
                            )
                            for layer in base_tail_router
                        ),
                        default=0.0,
                    )
                    logits_repeat_error = float(
                        np.max(np.abs(repeated_logits - base_logits), initial=0.0),
                    )
                    tail_no_op_hidden_max = max(
                        tail_no_op_hidden_max, hidden_repeat_error,
                    )
                    tail_no_op_router_max = max(
                        tail_no_op_router_max, router_repeat_error,
                    )
                    tail_no_op_logits_max = max(
                        tail_no_op_logits_max, logits_repeat_error,
                    )
                    if (
                        hidden_repeat_error != 0.0
                        or router_repeat_error != 0.0
                        or logits_repeat_error != 0.0
                    ):
                        raise RuntimeError("paired tail is not exactly repeatable")
                    base_hidden = {
                        injection_layer: base_start_array,
                        **base_tail_hidden,
                    }
                    base_router = {
                        injection_layer: np.asarray(
                            captured_router[injection_layer], np.float32,
                        ),
                        **base_tail_router,
                    }
                    baseline_capture_metrics = {}
                    for observation_layer in range(injection_layer, 40):
                        capture_hidden = np.asarray(
                            captured_hidden[observation_layer], np.float32,
                        )
                        paired_hidden = np.asarray(
                            base_hidden[observation_layer], np.float32,
                        )
                        capture_route = route_metrics(
                            np.asarray(captured_router[observation_layer], np.float32),
                            np.asarray(base_router[observation_layer], np.float32),
                        )
                        hidden_max = float(
                            np.max(np.abs(paired_hidden - capture_hidden), initial=0.0),
                        )
                        baseline_capture_hidden_max = max(
                            baseline_capture_hidden_max, hidden_max,
                        )
                        baseline_capture_route_churn_max = max(
                            baseline_capture_route_churn_max,
                            float(capture_route["route_churn"]),
                        )
                        baseline_capture_metrics[observation_layer] = {
                            "baseline_capture_hidden_mse": mean_squared_error(
                                capture_hidden, paired_hidden,
                            ),
                            "baseline_capture_hidden_max_abs": hidden_max,
                            **{
                                f"baseline_capture_{name}": value
                                for name, value in capture_route.items()
                            },
                        }
                    for rate, rate_index in rate_indices.items():
                        intended_delta, positions = _delta_for_request(
                            shard, rate_index, request_id, sequence_length,
                        )
                        delta_tensor = torch.as_tensor(
                            intended_delta,
                            device=device,
                            dtype=torch.bfloat16,
                        ).unsqueeze(0)
                        candidate_start = base_start + delta_tensor
                        realized_delta = (
                            candidate_start[0].float().cpu().numpy()
                            - base_start_array
                        )
                        realized_errors.append(
                            float(
                                np.max(
                                    np.abs(realized_delta - intended_delta),
                                    initial=0.0,
                                ),
                            ),
                        )
                        (
                            candidate_tail_hidden,
                            candidate_tail_router,
                            candidate_logits,
                            elapsed,
                        ) = observer.run_tail(
                            model,
                            candidate_start,
                            injection_layer,
                            contexts[request_id],
                        )
                        candidate_forward_seconds += elapsed
                        candidate_hidden = {
                            injection_layer: candidate_start[0].float().cpu().numpy(),
                            **candidate_tail_hidden,
                        }
                        candidate_router = {
                            injection_layer: np.asarray(
                                captured_router[injection_layer], np.float32,
                            ),
                            **candidate_tail_router,
                        }
                        intended_energy = float(
                            np.mean(
                                np.sum(
                                    intended_delta[positions].astype(np.float64) ** 2,
                                    axis=-1,
                                ),
                            ),
                        )
                        realized_energy = float(
                            np.mean(
                                np.sum(
                                    realized_delta[positions].astype(np.float64) ** 2,
                                    axis=-1,
                                ),
                            ),
                        )
                        intended_energy_all_tokens = float(
                            np.mean(
                                np.sum(intended_delta.astype(np.float64) ** 2, axis=-1),
                            ),
                        )
                        realized_energy_all_tokens = float(
                            np.mean(
                                np.sum(realized_delta.astype(np.float64) ** 2, axis=-1),
                            ),
                        )
                        if (
                            realized_energy <= 0.0
                            or realized_energy_all_tokens <= 0.0
                        ):
                            raise RuntimeError("realized impulse energy is not positive")
                        for observation_layer in range(injection_layer, 40):
                            reference_hidden = np.asarray(
                                base_hidden[observation_layer], np.float32,
                            )
                            comparison_hidden = np.asarray(
                                candidate_hidden[observation_layer], np.float32,
                            )
                            error = (
                                comparison_hidden.astype(np.float64)
                                - reference_hidden.astype(np.float64)
                            )
                            all_energy = float(np.mean(np.sum(error * error, axis=-1)))
                            captured_energy = float(
                                np.mean(
                                    np.sum(
                                        error[positions] * error[positions],
                                        axis=-1,
                                    ),
                                ),
                            )
                            routes = route_metrics(
                                np.asarray(base_router[observation_layer], np.float32),
                                np.asarray(candidate_router[observation_layer], np.float32),
                            )
                            propagation_rows.append({
                                "injection_layer": injection_layer,
                                "observation_layer": observation_layer,
                                "layer_distance": observation_layer - injection_layer,
                                "request_id": request_id,
                                "mean_budget_pages_per_expert": rate,
                                "charged_bpw": float(config["charged_bpw"][str(rate)]),
                                "sequence_tokens": sequence_length,
                                "injected_positions": len(positions),
                                "intended_delta_energy": intended_energy,
                                "realized_delta_energy": realized_energy,
                                "hidden_error_energy_all_tokens": all_energy,
                                "intended_delta_energy_all_tokens": (
                                    intended_energy_all_tokens
                                ),
                                "realized_delta_energy_all_tokens": realized_energy_all_tokens,
                                "hidden_error_energy_captured_positions": captured_energy,
                                "propagation_coefficient_all_tokens": (
                                    all_energy / realized_energy_all_tokens
                                ),
                                "propagation_coefficient_captured_positions": (
                                    captured_energy / realized_energy
                                ),
                                "hidden_mse": mean_squared_error(
                                    reference_hidden, comparison_hidden,
                                ),
                                "rmsnorm_normalized_mse": mean_squared_error(
                                    rms_normalize(reference_hidden),
                                    rms_normalize(comparison_hidden),
                                ),
                                "cosine_distance": cosine_distance(
                                    reference_hidden, comparison_hidden,
                                ),
                                **routes,
                                **baseline_capture_metrics[observation_layer],
                            })
                        labels = request["input_ids"][0, 1:].detach().cpu().numpy()
                        quality = token_quality_metrics(
                            np.asarray(base_logits[:-1], np.float32),
                            np.asarray(candidate_logits[:-1], np.float32),
                            labels,
                        )
                        quality_rows.append({
                            "injection_layer": injection_layer,
                            "request_id": request_id,
                            "mean_budget_pages_per_expert": rate,
                            "charged_bpw": float(config["charged_bpw"][str(rate)]),
                            "sequence_tokens": sequence_length,
                            "label_tokens": len(labels),
                            "intended_delta_energy": intended_energy,
                            "realized_delta_energy": realized_energy,
                            **quality,
                            "intended_delta_energy_all_tokens": intended_energy_all_tokens,
                            "realized_delta_energy_all_tokens": realized_energy_all_tokens,
                        })
                atomic_parquet(propagation_path, propagation_rows)
                atomic_parquet(quality_path, quality_rows)
                sidecar = {
                    "completed": True,
                    "schema": SCHEMA,
                    "run_id": config["run_id"],
                    "phase": "single-layer-paired-tail-frozen-error-impulse",
                    "injection_layer": injection_layer,
                    "rates": wanted_rates,
                    "requests": list(config["validation_request_ids"]),
                    "propagation_rows": len(propagation_rows),
                    "quality_rows": len(quality_rows),
                    "propagation_file": propagation_path.name,
                    "propagation_bytes": propagation_path.stat().st_size,
                    "propagation_sha256": sha256(propagation_path),
                    "quality_file": quality_path.name,
                    "quality_bytes": quality_path.stat().st_size,
                    "quality_sha256": sha256(quality_path),
                    "reconstruction_sha256": record["sha256"],
                    "baseline_reference_sha256": baseline_facts["reference_sha256"],
                    "max_realized_vs_intended_delta_abs": max(
                        realized_errors, default=0.0,
                    ),
                    "max_paired_baseline_vs_capture_hidden_abs": (
                        baseline_capture_hidden_max
                    ),
                    "max_paired_baseline_vs_capture_route_churn": (
                        baseline_capture_route_churn_max
                    ),
                    "model_load_seconds_for_process": load_seconds,
                    "paired_baseline_forward_seconds": baseline_forward_seconds,
                    "tail_no_op_hidden_max_abs": tail_no_op_hidden_max,
                    "tail_no_op_router_max_abs": tail_no_op_router_max,
                    "tail_no_op_logits_max_abs": tail_no_op_logits_max,
                    "candidate_forward_seconds": candidate_forward_seconds,
                    "wall_seconds": time.perf_counter() - started,
                    "test_scientific_rows_admitted_or_used": False,
                    "scientific_boundary": (
                        "Candidate and reference tails use the same GPU process and "
                        "kernel path from the captured layer output. Drift between the "
                        "paired reference tail and the old CPU capture is reported as "
                        "a separate control; this is not bit-identical continuation of "
                        "the old CPU trajectory."
                    ),
                    "runtime_provenance": runtime_provenance(),
                }
                atomic_json(sidecar_path, sidecar)
            except Exception as error:
                atomic_json(failure_path, {
                    "completed": False,
                    "schema": SCHEMA,
                    "phase": "single-layer-paired-tail-frozen-error-impulse",
                    "injection_layer": injection_layer,
                    "error": {"type": type(error).__name__, "message": str(error)},
                })
                raise
    finally:
        observer.close()


def finalize_impulse(args: argparse.Namespace) -> None:
    config = args.config_data
    baseline = load_json(args.output / BASELINE_FACTS)
    no_op_gates = ("no_op_hidden", "no_op_router", "no_op_logits")
    if not all(baseline.get("gates", {}).get(name) is True for name in no_op_gates):
        raise RuntimeError("same-process no-op gates did not pass")
    propagation_frames = []
    quality_frames = []
    layers = {}
    for layer in LAYERS:
        propagation, quality, sidecar_path, failure = _impulse_paths(args.output, layer)
        if failure.exists():
            raise RuntimeError(f"impulse failure is present: {failure}")
        if not all(path.is_file() for path in (propagation, quality, sidecar_path)):
            raise RuntimeError(f"impulse layer {layer} is incomplete")
        sidecar = load_json(sidecar_path)
        if (
            sidecar.get("propagation_sha256") != sha256(propagation)
            or sidecar.get("quality_sha256") != sha256(quality)
        ):
            raise RuntimeError(f"impulse layer {layer} changed")
        propagation_frames.append(pd.read_parquet(propagation))
        quality_frames.append(pd.read_parquet(quality))
        layers[str(layer)] = {
            "propagation_sha256": sidecar["propagation_sha256"],
            "quality_sha256": sidecar["quality_sha256"],
            "sidecar_sha256": sha256(sidecar_path),
        }
    propagation = pd.concat(propagation_frames, ignore_index=True)
    quality = pd.concat(quality_frames, ignore_index=True)
    propagation_path = args.output / IMPULSE_PROPAGATION
    quality_path = args.output / IMPULSE_QUALITY
    temporary = propagation_path.with_suffix(".parquet.tmp")
    propagation.to_parquet(temporary, index=False)
    temporary.replace(propagation_path)
    temporary = quality_path.with_suffix(".parquet.tmp")
    quality.to_parquet(temporary, index=False)
    temporary.replace(quality_path)
    facts = {
        "completed": True,
        "schema": SCHEMA,
        "run_id": config["run_id"],
        "phase": "single-layer-paired-tail-frozen-error-impulse",
        "layers": layers,
        "rates": list(config["initial_impulse_mean_pages"]),
        "requests": list(config["validation_request_ids"]),
        "propagation_rows": len(propagation),
        "quality_rows": len(quality),
        "propagation_file": propagation_path.name,
        "propagation_sha256": sha256(propagation_path),
        "quality_file": quality_path.name,
        "quality_sha256": sha256(quality_path),
        "baseline_reference_sha256": baseline["reference_sha256"],
        "test_scientific_rows_admitted_or_used": False,
        "scientific_boundary": (
            "Isolated paired-tail frozen-error impulse replay on the three validation "
            "prompts. Candidate and reference use one GPU process; old CPU-capture "
            "drift remains a separate control. This is not an end-to-end streamed "
            "model or population perplexity estimate."
        ),
        "runtime_provenance": runtime_provenance(),
    }
    atomic_json(args.output / IMPULSE_FACTS, facts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("baseline", "impulse", "finalize-impulse"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--reconstruction-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", nargs="*", type=int)
    args = parser.parse_args()
    args.config_data = load_json(args.config)
    validate_config(args.config_data)
    return args


def main() -> None:
    args = parse_args()
    if args.phase == "baseline":
        baseline_phase(args)
    elif args.phase == "impulse":
        impulse_phase(args)
    else:
        finalize_impulse(args)


if __name__ == "__main__":
    main()
