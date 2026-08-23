#!/usr/bin/env python3
"""Same-host PR #13 capture, true MoE replacement, and routing controls.

The exact model remains resident for baseline capture, live-activation PR #13
selection, zero-dose parity gates, and every sentinel replay. The primary
intervention replaces the routed-expert tensor before shared-expert composition
and before the decoder residual addition.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(EXPERIMENT / "scripts")]

from oracle_study.causal_control import (  # noqa: E402
    FROZEN_SET,
    FULLY_FROZEN,
    LIVE_ROUTES,
    POST_XPLUS_BF16,
    POST_XPLUS_FP32,
    PRE_RESIDUAL,
    bfloat16_ulp_metrics,
    build_control_cases,
    first_changed_layer,
    selector_and_execution_router_weights,
)
from oracle_study.causal_replay import (  # noqa: E402
    cosine_distance,
    mean_squared_error,
    projection_responses,
    reconstruct_group_from_responses,
    rms_normalize,
    route_boundary_metrics,
    route_metrics,
    token_quality_metrics,
)
from oracle_study.neuron_selector import unit_score_metadata  # noqa: E402
from run_causal_downstream_replay import LAYERS, load_json, sha256  # noqa: E402
from run_mxfp4_selective_pages import tree_from_record  # noqa: E402
import run_average_rate_allocation as pr13  # noqa: E402
import run_set_utility_distillation as set_study  # noqa: E402


SCHEMA = "pr13_same_host_causal_controls_v1"
BASELINE_FACTS = "same_host_baseline_facts.json"
RUN_FACTS = "same_host_control_run_facts.json"
PROPAGATION = "same_host_control_propagation.parquet"
QUALITY = "same_host_control_quality.parquet"
ZERO_GATES = "same_host_zero_dose_gates.parquet"
ALLOCATION = "same_host_allocation_layer_{layer:02d}.parquet"
DELTA = "same_host_delta_layer_{layer:02d}.npz"
LAYER_PROPAGATION = "same_host_propagation_layer_{layer:02d}.parquet"
LAYER_QUALITY = "same_host_quality_layer_{layer:02d}.parquet"
LAYER_ZERO = "same_host_zero_gates_layer_{layer:02d}.parquet"
LAYER_SIDECAR = "same_host_control_layer_{layer:02d}.json"
LAYER_FAILURE = "same_host_control_layer_{layer:02d}.failure.json"


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


def atomic_parquet(path: Path, rows: Sequence[Mapping[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _request_number(request_id: str) -> int:
    prefix = "mxfp4-confirm-"
    if not str(request_id).startswith(prefix):
        raise ValueError(f"unexpected request ID: {request_id}")
    value = int(str(request_id)[len(prefix):])
    if value < 0 or value >= 12:
        raise ValueError(f"request ID is outside immutable prompts: {request_id}")
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


def _encoded_requests(
    tokenizer: Any,
    model: Any,
    config: Mapping[str, Any],
) -> dict[str, dict[str, torch.Tensor]]:
    from capture_exact_mxfp4_confirm import PROMPTS

    result = {}
    device = model.get_input_embeddings().weight.device
    for request_value in config["validation_request_ids"]:
        request_id = str(request_value)
        encoded = tokenizer(
            PROMPTS[_request_number(request_id)],
            return_tensors="pt",
            truncation=True,
            max_length=64,
            add_special_tokens=True,
        )
        result[request_id] = {
            name: value.to(device) for name, value in encoded.items()
        }
    return result


def _tensor_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


class SameHostObserver:
    """Capture exact same-process states around every sentinel MoE residual."""

    def __init__(self, model: Any, sentinel_layers: Sequence[int]) -> None:
        self.sentinel_layers = frozenset(map(int, sentinel_layers))
        self.handles = []
        self.reset()
        for layer_id, layer in enumerate(model.model.language_model.layers):
            def layer_hook(
                _module: Any,
                _inputs: Any,
                output: Any,
                layer_id: int = layer_id,
            ) -> None:
                value = output[0] if isinstance(output, tuple) else output
                self.hidden[layer_id] = value.detach().cpu()

            def gate_hook(
                _module: Any,
                _inputs: Any,
                output: Any,
                layer_id: int = layer_id,
            ) -> None:
                if not isinstance(output, tuple) or len(output) < 3:
                    raise RuntimeError("router must return logits, scores, and IDs")
                self.router_logits[layer_id] = output[0].detach().cpu()
                self.router_scores[layer_id] = output[1].detach().cpu()
                self.router_ids[layer_id] = output[2].detach().cpu()

            self.handles.append(layer.register_forward_hook(layer_hook))
            self.handles.append(layer.mlp.gate.register_forward_hook(gate_hook))
            if layer_id in self.sentinel_layers:
                def norm_pre_hook(
                    _module: Any,
                    inputs: Any,
                    layer_id: int = layer_id,
                ) -> None:
                    self.residual[layer_id] = inputs[0].detach().cpu()

                def norm_hook(
                    _module: Any,
                    _inputs: Any,
                    output: torch.Tensor,
                    layer_id: int = layer_id,
                ) -> None:
                    self.x[layer_id] = output.detach().cpu()

                def experts_hook(
                    _module: Any,
                    _inputs: Any,
                    output: torch.Tensor,
                    layer_id: int = layer_id,
                ) -> None:
                    self.routed[layer_id] = output.detach().cpu()

                self.handles.append(
                    layer.post_attention_layernorm.register_forward_pre_hook(
                        norm_pre_hook,
                    ),
                )
                self.handles.append(
                    layer.post_attention_layernorm.register_forward_hook(norm_hook),
                )
                self.handles.append(
                    layer.mlp.experts.register_forward_hook(experts_hook),
                )

    def reset(self) -> None:
        self.hidden: dict[int, torch.Tensor] = {}
        self.router_logits: dict[int, torch.Tensor] = {}
        self.router_scores: dict[int, torch.Tensor] = {}
        self.router_ids: dict[int, torch.Tensor] = {}
        self.residual: dict[int, torch.Tensor] = {}
        self.x: dict[int, torch.Tensor] = {}
        self.routed: dict[int, torch.Tensor] = {}

    def run(
        self,
        model: Any,
        encoded: Mapping[str, torch.Tensor],
    ) -> tuple[dict[str, Any], float]:
        self.reset()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model(**encoded, use_cache=False, return_dict=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        expected = set(LAYERS)
        if (
            set(self.hidden) != expected
            or set(self.router_logits) != expected
            or set(self.router_scores) != expected
            or set(self.router_ids) != expected
        ):
            raise RuntimeError("same-host capture missed a decoder layer")
        if (
            set(self.residual) != set(self.sentinel_layers)
            or set(self.x) != set(self.sentinel_layers)
            or set(self.routed) != set(self.sentinel_layers)
        ):
            raise RuntimeError("same-host capture missed a sentinel MoE tensor")
        sequence_length = int(encoded["input_ids"].shape[1])
        result = {
            "input_ids": encoded["input_ids"][0].detach().cpu().numpy(),
            "attention_mask": encoded["attention_mask"][0].detach().cpu().numpy(),
            "hidden": np.stack([
                _tensor_numpy(self.hidden[layer][0]) for layer in LAYERS
            ]),
            "router_logits": np.stack([
                _tensor_numpy(
                    self.router_logits[layer].reshape(sequence_length, -1),
                )
                for layer in LAYERS
            ]),
            "router_scores": np.stack([
                _tensor_numpy(
                    self.router_scores[layer].reshape(sequence_length, -1),
                )
                for layer in LAYERS
            ]),
            "router_ids": np.stack([
                self.router_ids[layer].reshape(sequence_length, -1).numpy()
                for layer in LAYERS
            ]).astype(np.int64),
            "logits": _tensor_numpy(output.logits[0]),
            "residual": {
                layer: _tensor_numpy(self.residual[layer][0])
                for layer in self.sentinel_layers
            },
            "x": {
                layer: _tensor_numpy(self.x[layer][0])
                for layer in self.sentinel_layers
            },
            "routed": {
                layer: _tensor_numpy(
                    self.routed[layer].reshape(sequence_length, -1),
                )
                for layer in self.sentinel_layers
            },
        }
        del output
        return result, elapsed

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def _max_capture_difference(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> dict[str, float]:
    result = {}
    for name in ("hidden", "router_logits", "router_scores", "logits"):
        result[name] = float(
            np.max(
                np.abs(
                    np.asarray(first[name], np.float32)
                    - np.asarray(second[name], np.float32)
                ),
                initial=0.0,
            ),
        )
    result["router_ids"] = float(
        np.max(
            np.abs(
                np.asarray(first["router_ids"], np.int64)
                - np.asarray(second["router_ids"], np.int64)
            ),
            initial=0,
        ),
    )
    for name in ("residual", "x", "routed"):
        result[name] = max(
            (
                float(
                    np.max(
                        np.abs(
                            np.asarray(first[name][layer], np.float32)
                            - np.asarray(second[name][layer], np.float32)
                        ),
                        initial=0.0,
                    ),
                )
                for layer in first[name]
            ),
            default=0.0,
        )
    return result


def _baseline_arrays(
    request_id: str,
    capture: Mapping[str, Any],
    sentinel_layers: Sequence[int],
) -> dict[str, np.ndarray]:
    arrays = {
        "request_id": np.asarray(request_id),
        "input_ids": np.asarray(capture["input_ids"]),
        "attention_mask": np.asarray(capture["attention_mask"]),
        "hidden": np.asarray(capture["hidden"], np.float32),
        "router_logits": np.asarray(capture["router_logits"], np.float32),
        "router_scores": np.asarray(capture["router_scores"], np.float32),
        "router_ids": np.asarray(capture["router_ids"], np.int64),
        "logits": np.asarray(capture["logits"], np.float32),
    }
    for layer in sentinel_layers:
        arrays[f"residual_layer_{layer:02d}"] = np.asarray(
            capture["residual"][layer], np.float32,
        )
        arrays[f"x_layer_{layer:02d}"] = np.asarray(
            capture["x"][layer], np.float32,
        )
        arrays[f"routed_layer_{layer:02d}"] = np.asarray(
            capture["routed"][layer], np.float32,
        )
    return arrays


def _write_or_verify_baseline(
    output: Path,
    captures: Mapping[str, Mapping[str, Any]],
    sentinel_layers: Sequence[int],
) -> dict[str, dict[str, Any]]:
    records = {}
    root = output / "baseline"
    for request_id, capture in captures.items():
        path = root / f"{request_id}.npz"
        arrays = _baseline_arrays(request_id, capture, sentinel_layers)
        if path.exists():
            with np.load(path, allow_pickle=False) as loaded:
                if tuple(loaded.files) != tuple(arrays):
                    raise RuntimeError("same-host baseline schema changed on resume")
                for name, value in arrays.items():
                    if not np.array_equal(np.asarray(loaded[name]), value):
                        raise RuntimeError(
                            f"same-host baseline changed on resume: {request_id}/{name}",
                        )
        else:
            atomic_npz(path, arrays)
        records[request_id] = {
            "file": str(path.relative_to(output)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "tokens": int(len(capture["input_ids"])),
        }
    return records


def _validate_contract(args: argparse.Namespace) -> None:
    config = args.config_data
    pr13_config = args.pr13_config_data
    if list(map(int, config["sentinel_layers"])) != [0, 1, 4, 6, 12, 23, 39]:
        raise RuntimeError("sentinel layer grid changed")
    if list(map(int, config["mean_budget_pages_per_expert"])) != [384, 576, 749, 768]:
        raise RuntimeError("rate grid changed")
    if list(map(float, config["dose_alphas"])) != [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]:
        raise RuntimeError("dose grid changed")
    if list(map(str, config["route_modes"])) != [
        LIVE_ROUTES, FROZEN_SET, FULLY_FROZEN,
    ]:
        raise RuntimeError("route control grid changed")
    if sha256(args.pr13_config) != config["pr13_all_layer_config_sha256"]:
        raise RuntimeError("frozen all-layer PR #13 config changed")
    if sha256(EXPERIMENT / "scripts" / "run_average_rate_allocation.py") != config["pr13_runner_sha256"]:
        raise RuntimeError("frozen PR #13 runner changed")
    if sha256(EXPERIMENT / "src" / "oracle_study" / "average_rate_allocator.py") != config["pr13_allocator_sha256"]:
        raise RuntimeError("frozen PR #13 allocator changed")
    if sha256(args.checkpoint / "config.json") != config["checkpoint_config_sha256"]:
        raise RuntimeError("checkpoint config changed")
    if sha256(args.checkpoint / "model.safetensors.index.json") != config["checkpoint_index_sha256"]:
        raise RuntimeError("checkpoint index changed")
    if sha256(args.trees) != config["selected_tree_sha256"]:
        raise RuntimeError("selected tree changed")
    manifest = args.fit_dir / "average_rate_factor_manifest.json"
    if sha256(manifest) != config["factor_manifest_sha256"]:
        raise RuntimeError("all-layer PR #13 factor manifest changed")
    if str(pr13_config["primary_factor_id"]) != str(config["factor_config_id"]):
        raise RuntimeError("primary PR #13 factor changed")
    if int(config["burst_cap_pages_per_expert"]) != 1536:
        raise RuntimeError("burst cap changed")
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise RuntimeError("OMP_NUM_THREADS must be frozen to one")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise RuntimeError("OPENBLAS_NUM_THREADS must be frozen to one")
    if os.environ.get("MKL_NUM_THREADS") != "1":
        raise RuntimeError("MKL_NUM_THREADS must be frozen to one")
    if not torch.cuda.is_available():
        raise RuntimeError("same-host controls require one CUDA GPU")


@dataclass
class SelectorResult:
    allocation: pd.DataFrame
    deltas: np.ndarray
    rates: np.ndarray
    request_ids: np.ndarray
    positions: np.ndarray
    group_damage: np.ndarray
    max_group_damage_error: float
    all_q4_max_abs_error: float
    max_execution_router_sum_error: float
    max_execution_selector_weight_abs_difference: float
    active_experts: int


def _slim_pr13_config(
    config: Mapping[str, Any],
    pr13_config: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(pr13_config)
    primary = str(config["factor_config_id"])
    result["factor_configs"] = [
        dict(record)
        for record in pr13_config["factor_configs"]
        if str(record["factor_id"]) == primary
    ]
    if len(result["factor_configs"]) != 1:
        raise RuntimeError("primary factor specification is not unique")
    result["mean_correction_page_budgets"] = list(
        map(int, config["mean_budget_pages_per_expert"]),
    )
    result["primary_burst_caps_pages"] = [
        int(config["burst_cap_pages_per_expert"]),
    ]
    result["global_bound_mean_pages"] = -1
    result["quantized_control_factor_ids"] = []
    return result


def _selector_groups(
    captures: Mapping[str, Mapping[str, Any]],
    layer: int,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    groups = []
    activation = []
    excluded = int(config["exclude_final_positions"])
    for request_value in config["validation_request_ids"]:
        request_id = str(request_value)
        capture = captures[request_id]
        sequence_length = int(len(capture["input_ids"]))
        for position in range(sequence_length - excluded):
            record = len(groups)
            ids = np.asarray(
                capture["router_ids"][layer, position], np.int64,
            )
            selector_weights, execution_weights = (
                selector_and_execution_router_weights(
                    capture["router_scores"][layer, position],
                    historical_sum_atol=float(
                        config["historical_router_sum_atol"],
                    ),
                    execution_sum_atol=float(
                        config["same_host_execution_router_sum_atol"],
                    ),
                )
            )
            if ids.shape != (8,) or len(set(ids.tolist())) != 8:
                raise RuntimeError("same-host router did not produce eight unique experts")
            groups.append({
                "capture_source": "same_host_exact_gpu",
                "evaluation_split": "validation",
                "request_id": request_id,
                "sequence_id": request_id,
                "position": position,
                "layer": int(layer),
                "record": record,
                "experts": ids.tolist(),
                "router_weights": selector_weights.tolist(),
                "execution_router_weights": execution_weights.tolist(),
            })
            activation.append(np.asarray(capture["x"][layer][position], np.float32))
    if len(groups) != int(config["expected_groups_per_layer"]):
        raise RuntimeError("same-host validation group count changed")
    return groups, {"x": np.stack(activation)}


def _load_primary_factor(
    arrays: Mapping[str, np.ndarray],
    pr13_config: Mapping[str, Any],
    expert: int,
) -> Any:
    """Decode only the frozen primary factor with the original PR #13 schema."""

    rank = int(pr13_config["factor_fit_rank"])
    encoded = pr13.EncodedInteractionFactor(
        packed_codes=np.asarray(
            arrays["primary_packed_codes"][expert], np.uint8,
        ),
        row_scales=np.asarray(
            arrays["primary_row_scales"][expert], np.float16,
        ),
        units=pr13.UNITS,
        rank=rank,
        method="primary",
        tail_rank=rank,
        exact_rank=0,
        encoding="int4_hadamard",
        global_scale=np.float32(
            arrays["primary_global_scales"][expert],
        ),
    )
    factor = encoded.decode()
    primary = str(pr13_config["primary_factor_id"])
    specifications = [
        record for record in pr13_config["factor_configs"]
        if str(record["factor_id"]) == primary
    ]
    if len(specifications) != 1:
        raise RuntimeError("primary factor contract is not unique")
    expected = specifications[0]
    if (
        factor.rank != int(expected["rank"])
        or factor.payload_bytes != int(expected["factor_payload_bytes"])
    ):
        raise RuntimeError("primary factor contract changed")
    return factor


def _select_layer(
    args: argparse.Namespace,
    captures: Mapping[str, Mapping[str, Any]],
    layer: int,
    index: Mapping[str, str],
    tree_records: Mapping[str, Any],
) -> SelectorResult:
    config = args.config_data
    selector_config = _slim_pr13_config(config, args.pr13_config_data)
    factor_path = args.fit_dir / f"average_rate_factor_layer_{layer}.npz"
    factor_sidecar = args.fit_dir / f"average_rate_factor_layer_{layer}.json"
    sidecar = load_json(factor_sidecar)
    if sidecar.get("sha256") != sha256(factor_path):
        raise RuntimeError("factor layer sidecar/hash mismatch")
    arrays = dict(np.load(factor_path, allow_pickle=False))
    proxy = np.asarray(arrays["proxy"], np.float32)
    beta = float(np.asarray(arrays["beta"]).reshape(-1)[0])
    groups, local = _selector_groups(captures, layer, config)
    active = sorted({
        int(expert) for group in groups for expert in group["experts"]
    })
    trees = {
        name: tree_from_record(tree_records[name])
        for name in pr13.PROJECTIONS
    }
    decoded_by_expert = {}
    experts = {}
    primary = str(config["factor_config_id"])

    def prepare(expert: int) -> tuple[int, Any, Any]:
        decoded = set_study.decode_expert(
            args.checkpoint, index, trees, layer, expert,
        )
        q2 = tuple(decoded[name][0] for name in pr13.PROJECTIONS)
        q4 = tuple(decoded[name][2] for name in pr13.PROJECTIONS)
        cell = {
            "q2": q2,
            "q4": q4,
            "abc": unit_score_metadata(q2[2], q4[2], proxy=proxy, beta=beta),
            "factors": {
                primary: _load_primary_factor(
                    arrays, args.pr13_config_data, expert,
                ),
            },
        }
        return expert, (q2, q4), cell

    print(
        f"[layer {layer}] preparing {len(active)} active experts "
        f"with {min(int(args.selector_workers), len(active))} workers",
        flush=True,
    )
    with ThreadPoolExecutor(
        max_workers=min(int(args.selector_workers), len(active)),
    ) as executor:
        futures = {
            executor.submit(prepare, expert): expert for expert in active
        }
        for future in as_completed(futures):
            expert, decoded, cell = future.result()
            if int(expert) != int(futures[future]):
                raise RuntimeError("prepared expert identity changed")
            decoded_by_expert[expert] = decoded
            experts[expert] = cell
    print(f"[layer {layer}] evaluating {len(groups)} selector groups", flush=True)
    pr13._EVAL_CONTEXT = {
        "groups": tuple(groups),
        "local": local,
        "experts": experts,
        "proxy": proxy,
        "beta": beta,
        "config": selector_config,
    }
    completed = {}
    try:
        workers = min(int(args.selector_workers), len(groups))
        if args.selector_backend == "fork":
            executor_context = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp.get_context("fork"),
            )
        elif args.selector_backend == "thread":
            executor_context = ThreadPoolExecutor(max_workers=workers)
        else:
            raise RuntimeError("unknown selector worker backend")
        print(
            f"[layer {layer}] selector backend={args.selector_backend} "
            f"workers={workers}",
            flush=True,
        )
        with executor_context as executor:
            futures = {
                executor.submit(pr13._evaluate_group, task): task
                for task in range(len(groups))
            }
            for future in as_completed(futures):
                task, group_rows, expert_rows, diagnostics = future.result()
                if int(task) != int(futures[future]):
                    raise RuntimeError("selector task identity changed")
                completed[int(task)] = (group_rows, expert_rows, diagnostics)
    finally:
        pr13._EVAL_CONTEXT = None
    group_rows = []
    expert_rows = []
    for task in range(len(groups)):
        rows, experts_for_group, _ = completed[task]
        group_rows.extend(rows)
        expert_rows.extend(experts_for_group)
    group_frame = pd.DataFrame(group_rows)
    expert_frame = pd.DataFrame(expert_rows)
    common = (
        group_frame["factor_config_id"].eq(primary)
        & group_frame["allocation_policy"].eq(config["allocation_policy"])
        & group_frame["burst_cap_pages_per_expert"].eq(
            int(config["burst_cap_pages_per_expert"]),
        )
        & group_frame["mean_budget_pages_per_expert"].isin(
            list(map(int, config["mean_budget_pages_per_expert"])),
        )
    )
    selected_groups = group_frame.loc[common].copy()
    expert_common = (
        expert_frame["factor_config_id"].eq(primary)
        & expert_frame["allocation_policy"].eq(config["allocation_policy"])
        & expert_frame["burst_cap_pages_per_expert"].eq(
            int(config["burst_cap_pages_per_expert"]),
        )
        & expert_frame["mean_budget_pages_per_expert"].isin(
            list(map(int, config["mean_budget_pages_per_expert"])),
        )
    )
    selected_experts = expert_frame.loc[expert_common].copy()
    selected_groups = selected_groups.sort_values(
        ["request_id", "position", "mean_budget_pages_per_expert"],
        kind="stable",
    ).reset_index(drop=True)
    selected_experts = selected_experts.sort_values(
        [
            "request_id", "position", "mean_budget_pages_per_expert",
            "router_rank",
        ],
        kind="stable",
    ).reset_index(drop=True)
    group_lookup = {
        (str(group["request_id"]), int(group["position"])): group
        for group in groups
    }
    selector_values = []
    execution_values = []
    for row in selected_experts.itertuples(index=False):
        group = group_lookup[(str(row.request_id), int(row.position))]
        stored_rank = int(row.router_rank)
        if stored_rank < 1 or stored_rank > 8:
            raise RuntimeError("PR #13 router rank is outside one through eight")
        rank = stored_rank - 1
        selector_values.append(float(group["router_weights"][rank]))
        execution_values.append(
            float(group["execution_router_weights"][rank]),
        )
    selector_values_array = np.asarray(selector_values, np.float64)
    execution_values_array = np.asarray(execution_values, np.float64)
    if not np.array_equal(
        selected_experts["router_weight"].to_numpy(np.float64),
        selector_values_array,
    ):
        raise RuntimeError("selected PR #13 router weights changed")
    selected_experts["execution_router_weight"] = execution_values_array
    selected_experts["execution_selector_weight_abs_difference"] = np.abs(
        execution_values_array - selector_values_array,
    )
    max_execution_router_sum_error = max(
        abs(sum(group["execution_router_weights"]) - 1.0)
        for group in groups
    )
    max_execution_selector_weight_abs_difference = float(
        np.max(np.abs(execution_values_array - selector_values_array)),
    )
    rates = np.asarray(config["mean_budget_pages_per_expert"], np.int64)
    expected_groups = len(groups) * len(rates)
    if len(selected_groups) != expected_groups or len(selected_experts) != 8 * expected_groups:
        raise RuntimeError("same-host selected allocation grid changed")
    deltas = np.empty((len(rates), len(groups), 2048), np.float32)
    damage = np.empty((len(rates), len(groups)), np.float64)
    maximum_error = 0.0
    all_q4_maximum = 0.0
    for group_index, group in enumerate(groups):
        responses = projection_responses(
            local["x"][group_index],
            group["experts"],
            decoded_by_expert,
        )
        for rate_index, rate in enumerate(rates.tolist()):
            group_rows_for_rate = selected_groups[
                selected_groups["request_id"].eq(group["request_id"])
                & selected_groups["position"].eq(group["position"])
                & selected_groups["mean_budget_pages_per_expert"].eq(rate)
            ]
            expert_rows_for_rate = selected_experts[
                selected_experts["request_id"].eq(group["request_id"])
                & selected_experts["position"].eq(group["position"])
                & selected_experts["mean_budget_pages_per_expert"].eq(rate)
            ].sort_values("router_rank", kind="stable")
            if len(group_rows_for_rate) != 1 or len(expert_rows_for_rate) != 8:
                raise RuntimeError("selected same-host allocation identity is not unique")
            result = reconstruct_group_from_responses(
                responses,
                group["router_weights"],
                expert_rows_for_rate["selected_states"].tolist(),
                proxy,
                beta,
            )
            expected = float(
                group_rows_for_rate.iloc[0]["group_exact_qenergy_damage"],
            )
            error = abs(float(result.group_damage) - expected)
            maximum_error = max(maximum_error, error)
            all_q4_maximum = max(
                all_q4_maximum, float(result.all_q4_max_abs_error),
            )
            if not np.isclose(
                result.group_damage,
                expected,
                rtol=float(config["selector_qenergy_rtol"]),
                atol=float(config["selector_qenergy_atol"]),
            ):
                raise RuntimeError("same-host selector qenergy reconstruction failed")
            if not np.array_equal(
                result.selected_pages,
                expert_rows_for_rate["selected_pages"].to_numpy(np.int64),
            ):
                raise RuntimeError("same-host selector page reconstruction failed")
            selector_weights = np.asarray(
                group["router_weights"], np.float64,
            )
            execution_weights = np.asarray(
                group["execution_router_weights"], np.float64,
            )
            if np.array_equal(execution_weights, selector_weights):
                execution_delta = result.delta
            else:
                execution_delta = -np.einsum(
                    "e,eo->o", execution_weights,
                    result.expert_residuals, optimize=True,
                )
            deltas[rate_index, group_index] = execution_delta.astype(np.float32)
            damage[rate_index, group_index] = result.group_damage
    if all_q4_maximum > float(config["selector_all_q4_atol"]):
        raise RuntimeError(
            f"same-host all-Q4 reconstruction exceeded tolerance: "
            f"{all_q4_maximum:.17g}",
        )
    selected_experts.insert(0, "same_host_schema", SCHEMA)
    del experts, decoded_by_expert, arrays
    gc.collect()
    return SelectorResult(
        allocation=selected_experts,
        deltas=deltas,
        rates=rates,
        request_ids=np.asarray([group["request_id"] for group in groups]),
        positions=np.asarray([group["position"] for group in groups], np.int64),
        group_damage=damage,
        max_group_damage_error=maximum_error,
        all_q4_max_abs_error=all_q4_maximum,
        max_execution_router_sum_error=float(
            max_execution_router_sum_error,
        ),
        max_execution_selector_weight_abs_difference=(
            max_execution_selector_weight_abs_difference
        ),
        active_experts=len(active),
    )


class RouteOverride:
    """Override downstream executed routes while preserving live router logits."""

    def __init__(
        self,
        model: Any,
        start_layer: int,
        route_mode: str,
        baseline: Mapping[str, Any],
    ) -> None:
        self.handles = []
        if route_mode == LIVE_ROUTES:
            return
        layers = model.model.language_model.layers
        for layer_id in range(start_layer + 1, len(layers)):
            def hook(
                _module: Any,
                _inputs: Any,
                output: Any,
                layer_id: int = layer_id,
            ) -> Any:
                if not isinstance(output, tuple) or len(output) < 3:
                    raise RuntimeError("router override requires a three-tensor tuple")
                logits, scores, _ = output[:3]
                exact_ids = torch.as_tensor(
                    baseline["router_ids"][layer_id],
                    device=logits.device,
                    dtype=torch.long,
                ).reshape_as(output[2])
                if route_mode == FULLY_FROZEN:
                    exact_scores = torch.as_tensor(
                        baseline["router_scores"][layer_id],
                        device=logits.device,
                        dtype=scores.dtype,
                    ).reshape_as(scores)
                elif route_mode == FROZEN_SET:
                    probabilities = torch.softmax(logits, dtype=torch.float, dim=-1)
                    exact_scores = torch.gather(
                        probabilities, dim=-1, index=exact_ids,
                    )
                    exact_scores = exact_scores / exact_scores.sum(
                        dim=-1, keepdim=True,
                    )
                    exact_scores = exact_scores.to(scores.dtype)
                else:
                    raise RuntimeError(f"unknown route mode: {route_mode}")
                replacement = (logits, exact_scores, exact_ids, *output[3:])
                return replacement

            self.handles.append(
                layers[layer_id].mlp.gate.register_forward_hook(hook),
            )

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


@contextmanager
def _route_override(
    model: Any,
    start_layer: int,
    route_mode: str,
    baseline: Mapping[str, Any],
) -> Iterator[None]:
    override = RouteOverride(model, start_layer, route_mode, baseline)
    try:
        yield
    finally:
        override.close()


def _request_delta(
    selector: SelectorResult,
    rate_index: int,
    request_id: str,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    delta = np.zeros((sequence_length, selector.deltas.shape[-1]), np.float32)
    mask = selector.request_ids.astype(str) == str(request_id)
    positions = selector.positions[mask]
    delta[positions] = selector.deltas[rate_index, mask]
    return delta, positions


def _bf16_bits(value: torch.Tensor) -> np.ndarray:
    source = value.detach().contiguous().cpu()
    if source.dtype != torch.bfloat16:
        raise ValueError("ULP accounting requires a BF16 tensor")
    return source.view(torch.int16).numpy().view(np.uint16)


def _ulp_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    return bfloat16_ulp_metrics(
        _bf16_bits(reference),
        _bf16_bits(candidate),
        reference.detach().cpu().float().numpy()
        == candidate.detach().cpu().float().numpy(),
    )


def _pre_residual_start(
    model: Any,
    baseline: Mapping[str, Any],
    layer: int,
    delta: np.ndarray,
    alpha: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    device = model.get_input_embeddings().weight.device
    module = model.model.language_model.layers[layer]
    x = torch.as_tensor(
        baseline["x"][layer], device=device, dtype=torch.bfloat16,
    ).unsqueeze(0)
    residual = torch.as_tensor(
        baseline["residual"][layer], device=device, dtype=torch.bfloat16,
    ).unsqueeze(0)
    reference_routed = torch.as_tensor(
        baseline["routed"][layer], device=device, dtype=torch.bfloat16,
    )
    scaled = torch.as_tensor(delta, device=device, dtype=torch.float32) * float(alpha)
    replacement = (reference_routed.float() + scaled).to(torch.bfloat16)
    exact_repeat_max = 0.0

    def replace_routed(
        _module: Any,
        _inputs: Any,
        output: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal exact_repeat_max
        reshaped = output.reshape_as(reference_routed)
        exact_repeat_max = float(
            torch.max(torch.abs(reshaped.float() - reference_routed.float())).item(),
        )
        if exact_repeat_max != 0.0:
            raise RuntimeError("same-host routed expert output is not repeatable")
        return replacement.reshape_as(output)

    handle = module.mlp.experts.register_forward_hook(replace_routed)
    try:
        with torch.inference_mode():
            mlp_output = module.mlp(x)
            if isinstance(mlp_output, tuple):
                mlp_output = mlp_output[0]
            start = residual + mlp_output
    finally:
        handle.remove()
    return start, {
        "routed_repeat_max_abs": exact_repeat_max,
        "routed_replacement_applied": True,
        **{
            f"routed_{name}": value
            for name, value in _ulp_metrics(reference_routed, replacement).items()
        },
    }


def _post_xplus_start(
    model: Any,
    baseline: Mapping[str, Any],
    layer: int,
    delta: np.ndarray,
    alpha: float,
    injection_mode: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    device = model.get_input_embeddings().weight.device
    reference = torch.as_tensor(
        baseline["hidden"][layer], device=device, dtype=torch.bfloat16,
    ).unsqueeze(0)
    value = torch.as_tensor(delta, device=device, dtype=torch.float32) * float(alpha)
    if injection_mode == POST_XPLUS_BF16:
        candidate = reference + value.to(torch.bfloat16).unsqueeze(0)
    elif injection_mode == POST_XPLUS_FP32:
        candidate = (reference.float() + value.unsqueeze(0)).to(torch.bfloat16)
    else:
        raise RuntimeError("post-xplus helper received a non-post injection mode")
    return candidate, {
        "routed_replacement_applied": False,
        "routed_repeat_max_abs": 0.0,
        "routed_coordinates": 0,
        "routed_unchanged_fraction": 0.0,
        "routed_one_ulp_fraction": 0.0,
        "routed_multi_ulp_fraction": 0.0,
        "routed_mean_ulp_distance": 0.0,
        "routed_max_ulp_distance": 0,
    }


def _run_candidate(
    observer: ForwardObserver,
    model: Any,
    baseline: Mapping[str, Any],
    context: Mapping[str, Any],
    layer: int,
    delta: np.ndarray,
    alpha: float,
    injection_mode: str,
    route_mode: str,
) -> tuple[torch.Tensor, dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray, dict[str, Any], float]:
    if injection_mode == PRE_RESIDUAL:
        start, injection_facts = _pre_residual_start(
            model, baseline, layer, delta, alpha,
        )
    else:
        start, injection_facts = _post_xplus_start(
            model, baseline, layer, delta, alpha, injection_mode,
        )
    with _route_override(model, layer, route_mode, baseline):
        hidden, router, logits, elapsed = observer.run_tail(
            model, start, layer, context,
        )
    return start, hidden, router, logits, injection_facts, elapsed


def _hidden_and_router(
    baseline: Mapping[str, Any],
    layer: int,
    start: torch.Tensor,
    tail_hidden: Mapping[int, np.ndarray],
    tail_router: Mapping[int, np.ndarray],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    return (
        {layer: _tensor_numpy(start[0]), **dict(tail_hidden)},
        {
            layer: np.asarray(baseline["router_logits"][layer], np.float32),
            **dict(tail_router),
        },
    )


def _zero_reference(
    observer: ForwardObserver,
    model: Any,
    baseline: Mapping[str, Any],
    context: Mapping[str, Any],
    layer: int,
    route_mode: str,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray, dict[str, Any]]:
    sequence_length = int(len(baseline["input_ids"]))
    zero = np.zeros((sequence_length, 2048), np.float32)
    start, hidden, router, logits, injection, elapsed = _run_candidate(
        observer, model, baseline, context, layer, zero, 0.0,
        PRE_RESIDUAL, route_mode,
    )
    complete_hidden, complete_router = _hidden_and_router(
        baseline, layer, start, hidden, router,
    )
    hidden_max = max(
        float(
            np.max(
                np.abs(
                    np.asarray(complete_hidden[observation], np.float32)
                    - np.asarray(baseline["hidden"][observation], np.float32)
                ),
                initial=0.0,
            ),
        )
        for observation in range(layer, 40)
    )
    router_max = max(
        float(
            np.max(
                np.abs(
                    np.asarray(complete_router[observation], np.float32)
                    - np.asarray(baseline["router_logits"][observation], np.float32)
                ),
                initial=0.0,
            ),
        )
        for observation in range(layer, 40)
    )
    logits_max = float(
        np.max(
            np.abs(
                np.asarray(logits, np.float32)
                - np.asarray(baseline["logits"], np.float32)
            ),
            initial=0.0,
        ),
    )
    facts = {
        "hidden_max_abs": hidden_max,
        "router_max_abs": router_max,
        "logits_max_abs": logits_max,
        "elapsed_seconds": elapsed,
        **injection,
    }
    return complete_hidden, complete_router, logits, facts


def _energy(error: np.ndarray, positions: np.ndarray | None = None) -> float:
    value = np.asarray(error, np.float64)
    if positions is not None:
        value = value[np.asarray(positions, np.int64)]
    return float(np.mean(np.sum(value * value, axis=-1)))


def _replay_layer(
    args: argparse.Namespace,
    model: Any,
    observer: ForwardObserver,
    captures: Mapping[str, Mapping[str, Any]],
    encoded: Mapping[str, Mapping[str, torch.Tensor]],
    contexts: Mapping[str, Mapping[str, Any]],
    layer: int,
    selector: SelectorResult,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], float]:
    config = args.config_data
    cases = build_control_cases(config["dose_alphas"], include_zero_gates=False)
    propagation_rows = []
    quality_rows = []
    zero_rows = []
    candidate_seconds = 0.0
    for request_value in config["validation_request_ids"]:
        request_id = str(request_value)
        baseline = captures[request_id]
        sequence_length = int(len(baseline["input_ids"]))
        references = {}
        for route_mode in config["route_modes"]:
            hidden, router, logits, facts = _zero_reference(
                observer, model, baseline, contexts[request_id],
                layer, str(route_mode),
            )
            references[str(route_mode)] = (hidden, router, logits)
            zero_rows.append({
                "injection_layer": layer,
                "request_id": request_id,
                "route_mode": str(route_mode),
                "sequence_tokens": sequence_length,
                **facts,
            })
            if str(route_mode) == LIVE_ROUTES and any(
                float(facts[name]) > float(config["live_zero_dose_max_abs_atol"])
                for name in ("hidden_max_abs", "router_max_abs", "logits_max_abs")
            ):
                raise RuntimeError("live zero-dose replay did not reproduce same-host baseline")
        for rate_index, rate in enumerate(selector.rates.tolist()):
            delta, positions = _request_delta(
                selector, rate_index, request_id, sequence_length,
            )
            intended_energy = _energy(delta, positions)
            intended_all = _energy(delta)
            if intended_energy <= 0.0:
                raise RuntimeError("same-host intended delta energy is not positive")
            for case in cases:
                reference_hidden, reference_router, reference_logits = references[
                    case.route_mode
                ]
                start, tail_hidden, tail_router, candidate_logits, injection_facts, elapsed = _run_candidate(
                    observer,
                    model,
                    baseline,
                    contexts[request_id],
                    layer,
                    delta,
                    case.alpha,
                    case.injection_mode,
                    case.route_mode,
                )
                candidate_seconds += elapsed
                candidate_hidden, candidate_router = _hidden_and_router(
                    baseline, layer, start, tail_hidden, tail_router,
                )
                reference_start = torch.as_tensor(
                    reference_hidden[layer],
                    device=start.device,
                    dtype=torch.bfloat16,
                ).unsqueeze(0)
                layer_ulp = _ulp_metrics(reference_start, start)
                realized = (
                    np.asarray(candidate_hidden[layer], np.float64)
                    - np.asarray(reference_hidden[layer], np.float64)
                )
                realized_energy = _energy(realized, positions)
                realized_all = _energy(realized)
                boundary_by_layer = []
                observation_cache = {}
                for observation in range(layer, 40):
                    reference_value = np.asarray(
                        reference_hidden[observation], np.float32,
                    )
                    candidate_value = np.asarray(
                        candidate_hidden[observation], np.float32,
                    )
                    error = (
                        candidate_value.astype(np.float64)
                        - reference_value.astype(np.float64)
                    )
                    routes = route_metrics(
                        np.asarray(reference_router[observation], np.float32),
                        np.asarray(candidate_router[observation], np.float32),
                    )
                    boundary = route_boundary_metrics(
                        np.asarray(reference_router[observation], np.float32),
                        np.asarray(candidate_router[observation], np.float32),
                    )
                    boundary_by_layer.append((
                        observation,
                        boundary["route_membership_change_fraction"],
                    ))
                    observation_cache[observation] = boundary
                    propagation_rows.append({
                        "injection_layer": layer,
                        "observation_layer": observation,
                        "layer_distance": observation - layer,
                        "request_id": request_id,
                        "mean_budget_pages_per_expert": int(rate),
                        "charged_bpw": float(config["charged_bpw"][str(rate)]),
                        "alpha": float(case.alpha),
                        "injection_mode": case.injection_mode,
                        "route_mode": case.route_mode,
                        "sequence_tokens": sequence_length,
                        "injected_positions": len(positions),
                        "intended_delta_energy": intended_energy,
                        "intended_scaled_delta_energy": (
                            intended_energy * float(case.alpha) ** 2
                        ),
                        "realized_layer_error_energy": realized_energy,
                        "hidden_error_energy_all_tokens": _energy(error),
                        "hidden_error_energy_injected_positions": _energy(
                            error, positions,
                        ),
                        "hidden_mse": mean_squared_error(
                            reference_value, candidate_value,
                        ),
                        "rmsnorm_normalized_mse": mean_squared_error(
                            rms_normalize(reference_value),
                            rms_normalize(candidate_value),
                        ),
                        "cosine_distance": cosine_distance(
                            reference_value, candidate_value,
                        ),
                        **routes,
                        **boundary,
                    })
                labels = np.asarray(baseline["input_ids"][1:], np.int64)
                quality = token_quality_metrics(
                    np.asarray(reference_logits[:-1], np.float32),
                    np.asarray(candidate_logits[:-1], np.float32),
                    labels,
                )
                quality_rows.append({
                    "injection_layer": layer,
                    "request_id": request_id,
                    "mean_budget_pages_per_expert": int(rate),
                    "charged_bpw": float(config["charged_bpw"][str(rate)]),
                    "alpha": float(case.alpha),
                    "injection_mode": case.injection_mode,
                    "route_mode": case.route_mode,
                    "sequence_tokens": sequence_length,
                    "label_tokens": len(labels),
                    "injected_positions": len(positions),
                    "intended_delta_energy": intended_energy,
                    "intended_delta_energy_all_tokens": intended_all,
                    "intended_scaled_delta_energy": (
                        intended_energy * float(case.alpha) ** 2
                    ),
                    "realized_layer_error_energy": realized_energy,
                    "realized_layer_error_energy_all_tokens": realized_all,
                    "first_route_membership_change_layer": first_changed_layer(
                        boundary_by_layer,
                    ),
                    "final_router_mass_churn": observation_cache[39][
                        "router_mass_churn"
                    ],
                    **{
                        f"layer_output_{name}": value
                        for name, value in layer_ulp.items()
                    },
                    **injection_facts,
                    **quality,
                })
    return propagation_rows, quality_rows, zero_rows, candidate_seconds


def _layer_paths(output: Path, layer: int) -> dict[str, Path]:
    root = output / "layers"
    return {
        "allocation": root / ALLOCATION.format(layer=layer),
        "delta": root / DELTA.format(layer=layer),
        "propagation": root / LAYER_PROPAGATION.format(layer=layer),
        "quality": root / LAYER_QUALITY.format(layer=layer),
        "zero": root / LAYER_ZERO.format(layer=layer),
        "sidecar": root / LAYER_SIDECAR.format(layer=layer),
        "failure": root / LAYER_FAILURE.format(layer=layer),
    }


def _runtime_provenance() -> dict[str, Any]:
    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "cpu_count": os.cpu_count(),
    }


def run(args: argparse.Namespace) -> None:
    from run_causal_impulse_replay import (
        ForwardObserver,
        _tail_context,
    )

    _validate_contract(args)
    config = args.config_data
    model, tokenizer, load_seconds = _load_model(args.checkpoint)
    encoded = _encoded_requests(tokenizer, model, config)
    observer = SameHostObserver(model, config["sentinel_layers"])
    captures = {}
    capture_seconds = 0.0
    repeat_maxima = {}
    try:
        for request_value in config["validation_request_ids"]:
            request_id = str(request_value)
            first, elapsed = observer.run(model, encoded[request_id])
            capture_seconds += elapsed
            second, elapsed = observer.run(model, encoded[request_id])
            capture_seconds += elapsed
            differences = _max_capture_difference(first, second)
            if any(value != 0.0 for value in differences.values()):
                raise RuntimeError("same-host full baseline is not exactly repeatable")
            captures[request_id] = first
            repeat_maxima[request_id] = differences
    finally:
        observer.close()
    baseline_records = _write_or_verify_baseline(
        args.output, captures, config["sentinel_layers"],
    )
    baseline_payload = {
        "completed": True,
        "schema": SCHEMA,
        "run_id": config["run_id"],
        "same_process_model_residency": True,
        "requests": baseline_records,
        "repeat_max_abs": repeat_maxima,
        "capture_seconds": capture_seconds,
        "model_load_seconds": load_seconds,
        "runtime_provenance": _runtime_provenance(),
        "test_rows_admitted_or_used": False,
    }
    baseline_path = args.output / BASELINE_FACTS
    if baseline_path.exists():
        previous = load_json(baseline_path)
        stable = dict(baseline_payload)
        stable.pop("capture_seconds")
        stable.pop("model_load_seconds")
        previous_stable = dict(previous)
        previous_stable.pop("capture_seconds", None)
        previous_stable.pop("model_load_seconds", None)
        if previous_stable != stable:
            raise RuntimeError("same-host baseline facts changed on resume")
    else:
        atomic_json(baseline_path, baseline_payload)

    device = model.get_input_embeddings().weight.device
    contexts = {}
    for request_value in config["validation_request_ids"]:
        request_id = str(request_value)
        seed = torch.as_tensor(
            captures[request_id]["hidden"][0],
            device=device,
            dtype=torch.bfloat16,
        ).unsqueeze(0)
        contexts[request_id] = _tail_context(
            model, encoded[request_id], seed,
        )
    tail_observer = ForwardObserver(model)
    index = load_json(
        args.checkpoint / "model.safetensors.index.json",
    )["weight_map"]
    tree_records = load_json(args.trees)
    try:
        for layer in map(int, config["sentinel_layers"]):
            paths = _layer_paths(args.output, layer)
            if paths["failure"].exists():
                raise RuntimeError(f"prior layer failure must be reviewed: {paths['failure']}")
            required = [
                paths["allocation"], paths["delta"], paths["propagation"],
                paths["quality"], paths["zero"], paths["sidecar"],
            ]
            if any(path.exists() for path in required):
                if not all(path.is_file() for path in required):
                    raise RuntimeError("partial same-host layer artifact exists")
                sidecar = load_json(paths["sidecar"])
                for name in ("allocation", "delta", "propagation", "quality", "zero"):
                    if sidecar[f"{name}_sha256"] != sha256(paths[name]):
                        raise RuntimeError("same-host resume artifact changed")
                continue
            started = time.perf_counter()
            try:
                selector = _select_layer(
                    args, captures, layer, index, tree_records,
                )
                atomic_parquet(paths["allocation"], selector.allocation)
                atomic_npz(paths["delta"], {
                    "schema": np.asarray(SCHEMA),
                    "layer": np.asarray(layer, np.int64),
                    "mean_budget_pages_per_expert": selector.rates,
                    "request_id": selector.request_ids,
                    "position": selector.positions,
                    "delta": selector.deltas,
                    "group_damage": selector.group_damage,
                })
                propagation, quality, zero, candidate_seconds = _replay_layer(
                    args, model, tail_observer, captures, encoded,
                    contexts, layer, selector,
                )
                atomic_parquet(paths["propagation"], propagation)
                atomic_parquet(paths["quality"], quality)
                atomic_parquet(paths["zero"], zero)
                sidecar = {
                    "completed": True,
                    "schema": SCHEMA,
                    "run_id": config["run_id"],
                    "layer": layer,
                    "groups": int(config["expected_groups_per_layer"]),
                    "rates": selector.rates.tolist(),
                    "active_experts": selector.active_experts,
                    "max_group_qenergy_abs_error": selector.max_group_damage_error,
                    "all_q4_max_abs_error": selector.all_q4_max_abs_error,
                    "max_execution_router_sum_error": (
                        selector.max_execution_router_sum_error
                    ),
                    "max_execution_selector_weight_abs_difference": (
                        selector.max_execution_selector_weight_abs_difference
                    ),
                    "allocation_rows": len(selector.allocation),
                    "propagation_rows": len(propagation),
                    "quality_rows": len(quality),
                    "zero_gate_rows": len(zero),
                    "candidate_tail_seconds": candidate_seconds,
                    "wall_seconds": time.perf_counter() - started,
                    "test_rows_admitted_or_used": False,
                }
                for name in ("allocation", "delta", "propagation", "quality", "zero"):
                    sidecar[f"{name}_file"] = paths[name].name
                    sidecar[f"{name}_bytes"] = paths[name].stat().st_size
                    sidecar[f"{name}_sha256"] = sha256(paths[name])
                atomic_json(paths["sidecar"], sidecar)
            except Exception as error:
                atomic_json(paths["failure"], {
                    "completed": False,
                    "schema": SCHEMA,
                    "run_id": config["run_id"],
                    "layer": layer,
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                })
                raise
    finally:
        tail_observer.close()


def finalize(args: argparse.Namespace) -> None:
    _validate_contract(args)
    config = args.config_data
    propagation = []
    quality = []
    zero = []
    layers = {}
    for layer in map(int, config["sentinel_layers"]):
        paths = _layer_paths(args.output, layer)
        if paths["failure"].exists():
            raise RuntimeError(f"same-host layer failure is present: {layer}")
        if not paths["sidecar"].is_file():
            raise RuntimeError(f"same-host layer is incomplete: {layer}")
        sidecar = load_json(paths["sidecar"])
        for name in ("allocation", "delta", "propagation", "quality", "zero"):
            if sidecar[f"{name}_sha256"] != sha256(paths[name]):
                raise RuntimeError(f"same-host layer artifact changed: {layer}/{name}")
        propagation.append(pd.read_parquet(paths["propagation"]))
        quality.append(pd.read_parquet(paths["quality"]))
        zero.append(pd.read_parquet(paths["zero"]))
        layers[str(layer)] = sidecar
    propagation_frame = pd.concat(propagation, ignore_index=True)
    quality_frame = pd.concat(quality, ignore_index=True)
    zero_frame = pd.concat(zero, ignore_index=True)
    numeric = propagation_frame.select_dtypes(include=[np.number]).to_numpy(np.float64)
    if not np.all(np.isfinite(numeric)):
        raise RuntimeError("non-finite propagation result")
    numeric_quality = quality_frame.select_dtypes(include=[np.number]).drop(
        columns=["first_route_membership_change_layer"], errors="ignore",
    ).to_numpy(np.float64)
    if not np.all(np.isfinite(numeric_quality)):
        raise RuntimeError("non-finite quality result")
    atomic_parquet(args.output / PROPAGATION, propagation_frame)
    atomic_parquet(args.output / QUALITY, quality_frame)
    atomic_parquet(args.output / ZERO_GATES, zero_frame)
    payload = {
        "completed": True,
        "schema": SCHEMA,
        "run_id": config["run_id"],
        "sentinel_layers": list(map(int, config["sentinel_layers"])),
        "rates": list(map(int, config["mean_budget_pages_per_expert"])),
        "dose_alphas": list(map(float, config["dose_alphas"])),
        "route_modes": list(map(str, config["route_modes"])),
        "propagation_rows": len(propagation_frame),
        "quality_rows": len(quality_frame),
        "zero_gate_rows": len(zero_frame),
        "layers": layers,
        "baseline_facts_sha256": sha256(args.output / BASELINE_FACTS),
        "propagation_sha256": sha256(args.output / PROPAGATION),
        "quality_sha256": sha256(args.output / QUALITY),
        "zero_gates_sha256": sha256(args.output / ZERO_GATES),
        "test_rows_admitted_or_used": False,
        "scientific_boundary": config["scientific_boundary"],
    }
    atomic_json(args.output / RUN_FACTS, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("run", "finalize"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pr13-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trees", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selector-workers", type=int, default=24)
    parser.add_argument(
        "--selector-backend", choices=("fork", "thread"), default=None,
    )
    args = parser.parse_args()
    args.config_data = load_json(args.config)
    args.pr13_config_data = load_json(args.pr13_config)
    if args.selector_workers < 1 or args.selector_workers > 96:
        raise ValueError("selector workers must lie in [1,96]")
    configured_backend = str(args.config_data["selector_worker_backend"])
    if args.selector_backend is None:
        args.selector_backend = configured_backend
    if args.selector_backend != configured_backend:
        raise ValueError("selector backend differs from the frozen config")
    return args


def main() -> None:
    args = parse_args()
    if args.phase == "run":
        run(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
