#!/usr/bin/env python3
"""Exact-prefix, cached single-token downstream-tail KL experiment.

Every candidate starts from a private clone of the complete exact prefix
cache.  It processes one current token, replaces only that token's routed-MoE
output at one layer, retains the post-token cache, and scores only that
current token's terminal distribution.  The legacy no-cache full-sequence
runner lives in ``run_d1_full_sequence_tail_kl_legacy.py``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
import gc
import json
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

from oracle_study.causal_replay import projection_responses, qenergy_damage  # noqa: E402
from oracle_study.d1_decode import cache_state_metrics, clone_decode_cache  # noqa: E402
from oracle_study.split_interaction_field import split_state_output  # noqa: E402
from oracle_study.d1_decode_tail import (  # noqa: E402
    DECODE_POLICIES,
    DecodePolicyBank,
    add_live_minus_frozen,
    assemble_decode_policy_bank,
    calibrate_fixed_d1_policy,
    classified_cache_metrics,
    current_token_quality_metrics,
    downstream_route_rows,
    expected_grid_counts,
    index_decode_cells,
    sha256,
    token_delta,
    validate_cached_decode_tail_config,
)
from run_same_host_causal_controls import (  # noqa: E402
    _encoded_requests,
    _load_model,
    atomic_json,
    atomic_parquet,
)


from run_mxfp4_selective_pages import tree_from_record  # noqa: E402
import run_set_utility_distillation as set_study  # noqa: E402
import run_average_rate_allocation as pr13  # noqa: E402

SCHEMA = "pr13_d1_cached_decode_tail_kl_v2"
CALIBRATION = "d1_cached_decode_tail_fixed_calibration.parquet"
BANK_MANIFEST = "d1_cached_decode_tail_policy_bank_manifest.json"
QUALITY = "d1_cached_decode_tail_quality.parquet"
PROPAGATION = "d1_cached_decode_tail_propagation.parquet"
ZERO = "d1_cached_decode_tail_zero_gates.parquet"
RUN_FACTS = "d1_cached_decode_tail_run_facts.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _np(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


@dataclass
class DecodeObservation:
    hidden: dict[int, torch.Tensor]
    router_logits: dict[int, torch.Tensor]
    router_scores: dict[int, torch.Tensor]
    router_ids: dict[int, torch.Tensor]
    x: torch.Tensor
    routed: torch.Tensor
    logits: torch.Tensor
    cache: Any
    elapsed_seconds: float

    def numpy_hidden(self) -> dict[int, np.ndarray]:
        return {layer: _np(value).reshape(-1) for layer, value in self.hidden.items()}

    def numpy_router(self) -> dict[int, np.ndarray]:
        return {layer: _np(value).reshape(-1) for layer, value in self.router_logits.items()}


class CachedDecodeObserver:
    """Capture every current-token decoder state and one injection tensor."""

    def __init__(self, model: Any, injection_layer: int) -> None:
        self.injection_layer = int(injection_layer)
        self.handles = []
        self.reset()
        layers = model.model.language_model.layers
        for layer_id, layer in enumerate(layers):
            def layer_hook(
                _module: Any,
                _inputs: Any,
                output: Any,
                layer_id: int = layer_id,
            ) -> None:
                value = output[0] if isinstance(output, tuple) else output
                self.hidden[layer_id] = value.detach().cpu().clone()

            def gate_hook(
                _module: Any,
                _inputs: Any,
                output: Any,
                layer_id: int = layer_id,
            ) -> None:
                if not isinstance(output, tuple) or len(output) < 3:
                    raise RuntimeError("router must return logits, scores, and expert IDs")
                self.router_logits[layer_id] = output[0].detach().cpu().clone()
                self.router_scores[layer_id] = output[1].detach().cpu().clone()
                self.router_ids[layer_id] = output[2].detach().cpu().clone()

            self.handles.append(layer.register_forward_hook(layer_hook))
            self.handles.append(layer.mlp.gate.register_forward_hook(gate_hook))
        injection = layers[self.injection_layer]

        def norm_hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
            self.x = output.detach().cpu().clone()

        self.handles.append(
            injection.post_attention_layernorm.register_forward_hook(norm_hook)
        )

        def routed_hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
            self.routed = output.detach().cpu().clone()

        self.handles.append(injection.mlp.experts.register_forward_hook(routed_hook))

    def reset(self) -> None:
        self.hidden: dict[int, torch.Tensor] = {}
        self.router_logits: dict[int, torch.Tensor] = {}
        self.router_scores: dict[int, torch.Tensor] = {}
        self.router_ids: dict[int, torch.Tensor] = {}
        self.x: torch.Tensor | None = None
        self.routed: torch.Tensor | None = None

    def run(
        self,
        model: Any,
        encoded: Mapping[str, torch.Tensor],
        position: int,
        prefix_cache: Any | None,
    ) -> DecodeObservation:
        self.reset()
        device = encoded["input_ids"].device
        position = int(position)
        kwargs: dict[str, Any] = {
            "input_ids": encoded["input_ids"][:, position : position + 1],
            "attention_mask": encoded["attention_mask"][:, : position + 1],
            "past_key_values": prefix_cache,
            "cache_position": torch.as_tensor([position], device=device, dtype=torch.long),
            "use_cache": True,
            "return_dict": True,
        }
        started = time.perf_counter()
        with torch.inference_mode():
            output = model(**kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        expected = set(range(40))
        if (
            set(self.hidden) != expected
            or set(self.router_logits) != expected
            or set(self.router_scores) != expected
            or set(self.router_ids) != expected
            or self.x is None
            or self.routed is None
        ):
            raise RuntimeError("cached decode observer missed a current-token tensor")
        result = DecodeObservation(
            hidden=dict(self.hidden),
            router_logits=dict(self.router_logits),
            router_scores=dict(self.router_scores),
            router_ids=dict(self.router_ids),
            x=self.x,
            routed=self.routed,
            logits=output.logits[0, 0].detach().cpu().clone(),
            cache=output.past_key_values,
            elapsed_seconds=elapsed,
        )
        del output
        return result

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


class RoutedOutputReplacement:
    """Replace one current token before shared-expert composition/residual."""

    def __init__(
        self,
        model: Any,
        layer: int,
        reference_routed: torch.Tensor,
        delta: np.ndarray,
    ) -> None:
        self.reference = reference_routed.detach().cpu().clone()
        self.delta = np.asarray(delta, np.float32)
        self.repeat_max_abs = float("nan")
        self.applied = False

        def hook(_module: Any, _inputs: Any, output: torch.Tensor) -> torch.Tensor:
            reference = self.reference.to(device=output.device, dtype=output.dtype)
            reference = reference.reshape_as(output)
            self.repeat_max_abs = float(
                torch.max(torch.abs(output.float() - reference.float())).item()
            )
            if self.repeat_max_abs != 0.0:
                raise RuntimeError("current-token routed output is not bit-repeatable")
            correction = torch.as_tensor(
                self.delta, device=output.device, dtype=torch.float32,
            ).reshape_as(output)
            self.applied = True
            return (reference.float() + correction).to(output.dtype)

        self.handle = model.model.language_model.layers[int(layer)].mlp.experts.register_forward_hook(hook)

    def close(self) -> None:
        self.handle.remove()
        if not self.applied:
            raise RuntimeError("routed-output replacement hook was not exercised")


class FrozenDecodeRoutes:
    """Freeze executed downstream IDs/weights while retaining live logits."""

    def __init__(
        self,
        model: Any,
        injection_layer: int,
        route_mode: str,
        baseline: DecodeObservation,
    ) -> None:
        self.handles = []
        if route_mode == "live":
            return
        if route_mode != "fully_frozen":
            raise ValueError(f"unknown route mode: {route_mode}")
        layers = model.model.language_model.layers
        for layer_id in range(int(injection_layer) + 1, len(layers)):
            def hook(
                _module: Any,
                _inputs: Any,
                output: Any,
                layer_id: int = layer_id,
            ) -> Any:
                if not isinstance(output, tuple) or len(output) < 3:
                    raise RuntimeError("route freeze requires router tuple output")
                exact_scores = baseline.router_scores[layer_id].to(
                    device=output[1].device, dtype=output[1].dtype,
                ).reshape_as(output[1])
                exact_ids = baseline.router_ids[layer_id].to(
                    device=output[2].device, dtype=output[2].dtype,
                ).reshape_as(output[2])
                return (output[0], exact_scores, exact_ids, *output[3:])

            self.handles.append(layers[layer_id].mlp.gate.register_forward_hook(hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


@contextmanager
def _candidate_hooks(
    model: Any,
    layer: int,
    baseline: DecodeObservation,
    delta: np.ndarray,
    route_mode: str,
) -> Iterator[RoutedOutputReplacement]:
    replacement = RoutedOutputReplacement(model, layer, baseline.routed, delta)
    freezer = FrozenDecodeRoutes(model, layer, route_mode, baseline)
    try:
        yield replacement
    finally:
        freezer.close()
        replacement.close()


def _observation_maxima(
    first: DecodeObservation,
    second: DecodeObservation,
) -> dict[str, float | bool]:
    def maximum(left: Mapping[int, torch.Tensor], right: Mapping[int, torch.Tensor]) -> float:
        return max(
            float(torch.max(torch.abs(left[layer].float() - right[layer].float())).item())
            for layer in left
        )

    return {
        "hidden_max_abs": maximum(first.hidden, second.hidden),
        "router_logits_max_abs": maximum(first.router_logits, second.router_logits),
        "router_scores_max_abs": maximum(first.router_scores, second.router_scores),
        "router_ids_equal": all(
            torch.equal(first.router_ids[layer], second.router_ids[layer])
            for layer in first.router_ids
        ),
        "x_max_abs": float(
            torch.max(torch.abs(first.x.float() - second.x.float())).item()
        ),
        "routed_max_abs": float(
            torch.max(torch.abs(first.routed.float() - second.routed.float())).item()
        ),
        "terminal_logits_max_abs": float(
            torch.max(torch.abs(first.logits.float() - second.logits.float())).item()
        ),
    }


def _run_candidate(
    observer: CachedDecodeObserver,
    model: Any,
    encoded: Mapping[str, torch.Tensor],
    position: int,
    exact_prefix: Any,
    baseline: DecodeObservation,
    layer: int,
    delta: np.ndarray,
    route_mode: str,
) -> tuple[DecodeObservation, dict[str, Any]]:
    private_cache = clone_decode_cache(exact_prefix)
    with _candidate_hooks(model, layer, baseline, delta, route_mode) as replacement:
        candidate = observer.run(model, encoded, position, private_cache)
    return candidate, {
        "routed_replacement_applied": bool(replacement.applied),
        "routed_repeat_max_abs": float(replacement.repeat_max_abs),
    }


def _zero_metrics(
    baseline: DecodeObservation,
    candidate: DecodeObservation,
) -> dict[str, Any]:
    maxima = _observation_maxima(baseline, candidate)
    cache = classified_cache_metrics(baseline.cache, candidate.cache)
    return {
        **maxima,
        **{f"post_token_{name}": value for name, value in cache.items()},
        "all_exact": bool(
            maxima["hidden_max_abs"] == 0.0
            and maxima["router_logits_max_abs"] == 0.0
            and maxima["router_scores_max_abs"] == 0.0
            and maxima["router_ids_equal"]
            and maxima["x_max_abs"] == 0.0
            and maxima["routed_max_abs"] == 0.0
            and maxima["terminal_logits_max_abs"] == 0.0
            and cache["full_cache_bit_identical"]
        ),
    }


def _policy_banks(config: Mapping[str, Any]) -> tuple[
    Mapping[tuple[int, int], Any],
    dict[int, str],
    pd.DataFrame,
    dict[tuple[int, int], DecodePolicyBank],
]:
    cells = index_decode_cells(config)
    fixed, evidence = calibrate_fixed_d1_policy(
        cells,
        config["fixed_d1_calibration"]["calibration_layers"],
        config["page_caps"],
    )
    frozen = config["d1_source_policy_by_rate"]
    for rate, selected in fixed.items():
        if str(frozen[str(rate)]["fixed"]) != str(selected):
            raise RuntimeError(
                f"frozen fixed policy differs from calibration at rate {rate}: "
                f"{frozen[str(rate)]['fixed']} != {selected}"
            )
    banks = {
        key: assemble_decode_policy_bank(
            cell,
            fixed_source_policy=str(frozen[str(key[1])]["fixed"]),
            companion_source_policy=str(frozen[str(key[1])]["companion"]),
        )
        for key, cell in cells.items()
    }
    first_identity = None
    for key, bank in sorted(banks.items()):
        identity = bank.identity[["group", "request_id", "position"]].reset_index(drop=True)
        if first_identity is None:
            first_identity = identity
        elif not identity.equals(first_identity):
            raise RuntimeError(f"decode token identity grid differs in cell {key}")
    return cells, fixed, evidence, banks


def _manifest(
    config: Mapping[str, Any],
    config_path: Path,
    cells: Mapping[tuple[int, int], Any],
    banks: Mapping[tuple[int, int], DecodePolicyBank],
    fixed: Mapping[int, str],
) -> dict[str, Any]:
    implementation = [
        EXPERIMENT / "src/oracle_study/d1_decode.py",
        EXPERIMENT / "src/oracle_study/d1_decode_tail.py",
        EXPERIMENT / "scripts/run_d1_downstream_tail_kl.py",
        EXPERIMENT / "scripts/analyze_d1_downstream_tail_kl.py",
    ]
    records = {}
    for key, cell in sorted(cells.items()):
        bank = banks[key]
        facts = cell.directory / "d1_decode_layer_facts.json"
        records[f"layer_{key[0]:02d}_rate_{key[1]}"] = {
            "directory": str(cell.directory),
            "facts_sha256": sha256(facts),
            "delta_sha256": sha256(cell.directory / "d1_decode_selected_deltas.npz"),
            "allocation_experts_sha256": sha256(cell.directory / "d1_decode_allocation_experts.parquet"),
            "token_oracle_sha256": sha256(cell.directory / "d1_decode_token_oracle.parquet"),
            "groups": int(cell.facts["groups"]),
            "policy_provenance": bank.provenance,
        }
    return {
        "completed": True,
        "schema": SCHEMA,
        "config_sha256": sha256(config_path),
        "implementation_files": {
            str(path.relative_to(EXPERIMENT)): {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in implementation
        },
        "fixed_d1_source_policy_by_rate": {
            str(rate): value for rate, value in sorted(fixed.items())
        },
        "cells": records,
        "expected_grid": expected_grid_counts(config),
        "matched_raw_manifest_sha256": str(config["matched_raw_manifest_sha256"]),
        "live_selected_state_reconstruction_required": True,
        "sequence_shaped_delta_constructed": False,
        "test_rows_admitted_or_used": False,
    }


def validate_phase(args: argparse.Namespace) -> None:
    base_config = EXPERIMENT / str(args.config_data["base_decode_slice_config"])
    if sha256(base_config) != str(
        args.config_data["base_decode_slice_config_sha256"]
    ):
        raise RuntimeError("immutable exact-prefill decode slice config changed")
    cells, fixed, evidence, banks = _policy_banks(args.config_data)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_parquet(args.output / CALIBRATION, evidence)
    atomic_json(
        args.output / BANK_MANIFEST,
        _manifest(args.config_data, args.config, cells, banks, fixed),
    )
    print(
        "[validated] "
        + ", ".join(f"rate {rate}: {policy}" for rate, policy in sorted(fixed.items())),
        flush=True,
    )


def _gpu_gate(config: Mapping[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("cached-decode tail requires CUDA")
    properties = torch.cuda.get_device_properties(0)
    name = str(properties.name)
    memory_gib = int(properties.total_memory) / float(1 << 30)
    required = config["hardware_execution_path"]
    if str(required["required_gpu_name_substring"]) not in name:
        raise RuntimeError(f"wrong GPU for full cached tail: {name}")
    if memory_gib < float(required["minimum_gpu_memory_gib"]):
        raise RuntimeError(f"insufficient GPU memory for full cached tail: {memory_gib:.1f} GiB")
    return {"gpu": name, "gpu_memory_gib": memory_gib}


def _cell_paths(output: Path, layer: int, rate: int) -> dict[str, Path]:
    root = output / "cells" / f"layer_{int(layer):02d}_rate_{int(rate)}"
    return {
        "root": root,
        "quality": root / QUALITY,
        "propagation": root / PROPAGATION,
        "zero": root / ZERO,
        "facts": root / "d1_cached_decode_tail_cell_facts.json",
    }


def _completed_cell(paths: Mapping[str, Path], layer: int, rate: int) -> bool:
    if not paths["facts"].is_file():
        return False
    facts = load_json(paths["facts"])
    if not facts.get("completed") or int(facts["layer"]) != layer or int(facts["rate"]) != rate:
        return False
    for key in ("quality", "propagation", "zero"):
        path = paths[key]
        if not path.is_file() or sha256(path) != facts["files"][path.name]["sha256"]:
            return False
    return True


def _request_rows(bank: DecodePolicyBank, request_id: str) -> pd.DataFrame:
    rows = bank.identity[bank.identity["request_id"].astype(str).eq(str(request_id))]
    rows = rows.sort_values("position", kind="stable").reset_index(drop=True)
    positions = rows["position"].to_numpy(np.int64)
    if len(rows) == 0 or not np.array_equal(
        positions, np.arange(int(positions[0]), int(positions[0]) + len(positions)),
    ) or int(positions[0]) != 1:
        raise RuntimeError("decode positions must be contiguous from absolute position one")
    return rows


def _prepare_live_reconstruction(
    checkpoint: Path,
    trees_path: Path,
    fit_dir: Path,
    layer: int,
    banks: Sequence[DecodePolicyBank],
    workers: int,
) -> tuple[dict[int, tuple[Sequence[np.ndarray], Sequence[np.ndarray]]], np.ndarray, float]:
    index = load_json(checkpoint / "model.safetensors.index.json")["weight_map"]
    tree_records = load_json(trees_path)
    trees = {
        name: tree_from_record(tree_records[name])
        for name in pr13.PROJECTIONS
    }
    active = sorted({
        int(expert)
        for bank in banks
        for expert in bank.expert_ids.reshape(-1).tolist()
    })

    def prepare(expert: int) -> tuple[int, tuple[Sequence[np.ndarray], Sequence[np.ndarray]]]:
        decoded = set_study.decode_expert(checkpoint, index, trees, int(layer), int(expert))
        q2 = tuple(decoded[name][0] for name in pr13.PROJECTIONS)
        q4 = tuple(decoded[name][2] for name in pr13.PROJECTIONS)
        return int(expert), (q2, q4)

    result: dict[int, tuple[Sequence[np.ndarray], Sequence[np.ndarray]]] = {}
    print(f"[layer {layer}] decoding {len(active)} experts for live activation reconstruction", flush=True)
    with ThreadPoolExecutor(max_workers=min(int(workers), len(active))) as executor:
        futures = {executor.submit(prepare, expert): expert for expert in active}
        for completed, future in enumerate(as_completed(futures), start=1):
            expert, decoded = future.result()
            result[expert] = decoded
            if completed % 16 == 0 or completed == len(active):
                print(f"[layer {layer}] decoded {completed}/{len(active)} experts", flush=True)
    factor = fit_dir / f"average_rate_factor_layer_{int(layer)}.npz"
    with np.load(factor, allow_pickle=False) as arrays:
        proxy = np.asarray(arrays["proxy"], np.float32)
        beta = float(np.asarray(arrays["beta"]).reshape(-1)[0])
    return result, proxy, beta


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, np.float64).reshape(-1)
    second = np.asarray(right, np.float64).reshape(-1)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(first @ second / denominator) if denominator else 1.0


def _live_policy_delta(
    bank: DecodePolicyBank,
    policy: str,
    group: int,
    request_id: str,
    position: int,
    baseline: DecodeObservation,
    decoded: Mapping[int, tuple[Sequence[np.ndarray], Sequence[np.ndarray]]],
    proxy: np.ndarray,
    beta: float,
    config: Mapping[str, Any],
    reconstruction_context: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    stored, source = token_delta(
        bank, policy, group=group, request_id=request_id, position=position,
    )
    source_ids = np.asarray(bank.expert_ids[int(group)], np.int64)
    live_ids = baseline.router_ids[int(bank.layer)].reshape(-1).numpy().astype(np.int64)
    if source_ids.shape != (8,) or live_ids.shape != (8,):
        raise RuntimeError("routed expert identity must contain eight values")
    source_set = set(source_ids.tolist())
    live_set = set(live_ids.tolist())
    if source_set != live_set:
        raise RuntimeError(
            f"cached full-model route differs from allocation route at "
            f"layer={bank.layer} group={group}: source={source_ids.tolist()} "
            f"live={live_ids.tolist()}"
        )
    source_rank = {int(expert): rank for rank, expert in enumerate(source_ids.tolist())}
    states = np.stack([
        bank.selected_states[policy][int(group), source_rank[int(expert)]]
        for expert in live_ids.tolist()
    ])
    if "responses" not in reconstruction_context:
        reconstruction_context["responses"] = projection_responses(
            _np(baseline.x).reshape(-1), live_ids, decoded,
        )
    responses = reconstruction_context["responses"]
    weights = _np(baseline.router_scores[int(bank.layer)]).reshape(-1).astype(np.float64)
    if weights.shape != (8,) or np.any(weights <= 0.0):
        raise RuntimeError("live execution router weights are invalid")
    residuals = []
    targets = []
    for response, state in zip(responses, states):
        target = np.asarray(response.target_output, np.float64)
        approximate = np.asarray(split_state_output(response, state), np.float64)
        targets.append(target)
        residuals.append(target - approximate)
    residuals_array = np.stack(residuals)
    live_delta = -np.einsum("e,eo->o", weights, residuals_array, optimize=True)
    q4_routed = np.einsum("e,eo->o", weights, np.stack(targets), optimize=True)
    reference_routed = _np(baseline.routed).reshape(-1).astype(np.float64)
    q4_max_abs = float(np.max(np.abs(q4_routed - reference_routed), initial=0.0))
    tolerance = float(config["live_reconstruction"]["q4_routed_max_abs_atol"])
    if q4_max_abs > tolerance:
        raise RuntimeError(
            f"live Q4 reconstruction exceeds BF16 execution tolerance: {q4_max_abs} > {tolerance}"
        )
    live = np.asarray(live_delta, np.float32)
    difference = live.astype(np.float64) - stored.astype(np.float64)
    return live, {
        "selected_group_pages": int(source["selected_group_pages"]),
        "source_selected_local_qenergy_damage": float(
            source["selected_local_qenergy_damage"]
        ),
        "live_selected_local_qenergy_damage": float(
            qenergy_damage(live, proxy, beta)
        ),
        "source_stored_delta_mse": float(source["injected_delta_mse"]),
        "live_injected_delta_mse": float(np.mean(live.astype(np.float64) ** 2)),
        "stored_to_live_delta_mse": float(np.mean(difference ** 2)),
        "stored_to_live_delta_cosine": _cosine_similarity(stored, live),
        "live_q4_routed_max_abs": q4_max_abs,
        "source_and_live_route_set_equal": True,
        "source_and_live_route_order_equal": bool(np.array_equal(source_ids, live_ids)),
        "live_execution_router_weight_sum": float(weights.sum()),
        "delta_execution": "selected_states_reconstructed_on_live_cached_activation",
    }


def run_cell(
    model: Any,
    encoded_requests: Mapping[str, Mapping[str, torch.Tensor]],
    bank: DecodePolicyBank,
    decoded: Mapping[int, tuple[Sequence[np.ndarray], Sequence[np.ndarray]]],
    proxy: np.ndarray,
    beta: float,
    config: Mapping[str, Any],
    output: Path,
) -> None:
    layer, rate = int(bank.layer), int(bank.rate)
    paths = _cell_paths(output, layer, rate)
    if _completed_cell(paths, layer, rate):
        print(f"[resume] layer={layer} rate={rate}", flush=True)
        return
    observer = CachedDecodeObserver(model, layer)
    quality_rows: list[dict[str, Any]] = []
    propagation_rows: list[dict[str, Any]] = []
    zero_rows: list[dict[str, Any]] = []
    baseline_seconds = 0.0
    candidate_seconds = 0.0
    try:
        for request_id in map(str, config["validation_request_ids"]):
            encoded = encoded_requests[request_id]
            rows = _request_rows(bank, request_id)
            # Token zero is executed exactly once to construct the complete
            # native cache used as the prefix for absolute position one.
            prefix_observation = observer.run(model, encoded, 0, None)
            exact_prefix = prefix_observation.cache
            baseline_seconds += prefix_observation.elapsed_seconds
            del prefix_observation
            for identity in rows.itertuples(index=False):
                group = int(identity.group)
                position = int(identity.position)
                first = observer.run(
                    model, encoded, position, clone_decode_cache(exact_prefix),
                )
                second = observer.run(
                    model, encoded, position, clone_decode_cache(exact_prefix),
                )
                baseline_seconds += first.elapsed_seconds + second.elapsed_seconds
                repeat = _observation_maxima(first, second)
                repeat_cache = classified_cache_metrics(first.cache, second.cache)
                if (
                    any(float(repeat[name]) != 0.0 for name in (
                        "hidden_max_abs", "router_logits_max_abs", "router_scores_max_abs",
                        "x_max_abs", "routed_max_abs", "terminal_logits_max_abs",
                    ))
                    or not bool(repeat["router_ids_equal"])
                    or not bool(repeat_cache["full_cache_bit_identical"])
                ):
                    raise RuntimeError("paired native cached-decode baseline is not bit-identical")

                zero_delta = np.zeros(2048, np.float32)
                for route_mode in map(str, config["route_modes"]):
                    zero_candidate, injection = _run_candidate(
                        observer, model, encoded, position, exact_prefix, first,
                        layer, zero_delta, route_mode,
                    )
                    candidate_seconds += zero_candidate.elapsed_seconds
                    zero_metric = _zero_metrics(first, zero_candidate)
                    if not bool(zero_metric["all_exact"]) or injection["routed_repeat_max_abs"] != 0.0:
                        raise RuntimeError("zero-dose cached decode failed exact parity")
                    zero_rows.append({
                        "schema": SCHEMA,
                        "injection_layer": layer,
                        "rate_pages_per_expert": rate,
                        "group": group,
                        "request_id": request_id,
                        "position": position,
                        "prefix_tokens": position,
                        "route_mode": route_mode,
                        "query_length": 1,
                        "cross_position_delta_coupling": False,
                        "routed_repeat_max_abs": injection["routed_repeat_max_abs"],
                        "paired_baseline_full_cache_bit_identical": repeat_cache["full_cache_bit_identical"],
                        **zero_metric,
                    })
                    del zero_candidate

                baseline_hidden = first.numpy_hidden()
                baseline_router = first.numpy_router()
                next_token_id = int(encoded["input_ids"][0, position + 1].item())
                reconstruction_context: dict[str, Any] = {}
                for policy in DECODE_POLICIES:
                    delta, local = _live_policy_delta(
                        bank, policy, group, request_id, position, first,
                        decoded, proxy, beta, config, reconstruction_context,
                    )
                    for route_mode in map(str, config["route_modes"]):
                        candidate, injection = _run_candidate(
                            observer, model, encoded, position, exact_prefix, first,
                            layer, delta, route_mode,
                        )
                        candidate_seconds += candidate.elapsed_seconds
                        candidate_hidden = candidate.numpy_hidden()
                        candidate_router = candidate.numpy_router()
                        quality = current_token_quality_metrics(
                            _np(first.logits), _np(candidate.logits), next_token_id,
                            baseline_hidden[39], candidate_hidden[39],
                        )
                        route_rows, first_changed = downstream_route_rows(
                            baseline_router, candidate_router,
                            baseline_hidden, candidate_hidden, layer,
                        )
                        cache = classified_cache_metrics(first.cache, candidate.cache)
                        base = {
                            "schema": SCHEMA,
                            "injection_layer": layer,
                            "rate_pages_per_expert": rate,
                            "group": group,
                            "request_id": request_id,
                            "position": position,
                            "prefix_tokens": position,
                            "policy": policy,
                            "route_mode": route_mode,
                            "next_token_id": next_token_id,
                            "query_length": 1,
                            "cross_position_delta_coupling": False,
                        }
                        quality_rows.append({
                            **base,
                            **local,
                            **quality,
                            "realized_injected_layer_output_mse": float(
                                np.mean((baseline_hidden[layer].astype(np.float64) - candidate_hidden[layer].astype(np.float64)) ** 2)
                            ),
                            "first_route_membership_change_layer": (
                                int(first_changed) if first_changed is not None else -1
                            ),
                            "routed_repeat_max_abs": injection["routed_repeat_max_abs"],
                            "candidate_seconds": candidate.elapsed_seconds,
                            **{f"post_token_{name}": value for name, value in cache.items()},
                        })
                        for route in route_rows:
                            observation_layer = int(route["observation_layer"])
                            layer_cache = cache_state_metrics(
                                first.cache, candidate.cache, observation_layer,
                            )
                            propagation_rows.append({
                                **base,
                                **route,
                                **layer_cache.to_dict("post_token_layer_cache"),
                            })
                        del candidate
                # The paired exact baseline becomes the exact prefix for the
                # next absolute token. Candidate caches are never reused.
                exact_prefix = first.cache
                del second
                gc.collect()
    finally:
        observer.close()

    expected_quality = 29 * len(DECODE_POLICIES) * len(config["route_modes"])
    expected_propagation = expected_quality * (39 - layer)
    expected_zero = 29 * len(config["route_modes"])
    if len(quality_rows) != expected_quality:
        raise RuntimeError("cached-decode cell quality grid is incomplete")
    if len(propagation_rows) != expected_propagation:
        raise RuntimeError("cached-decode cell propagation grid is incomplete")
    if len(zero_rows) != expected_zero:
        raise RuntimeError("cached-decode cell zero-dose grid is incomplete")
    paths["root"].mkdir(parents=True, exist_ok=True)
    atomic_parquet(paths["quality"], quality_rows)
    atomic_parquet(paths["propagation"], propagation_rows)
    atomic_parquet(paths["zero"], zero_rows)
    files = {
        paths[key].name: {
            "sha256": sha256(paths[key]),
            "bytes": paths[key].stat().st_size,
        }
        for key in ("quality", "propagation", "zero")
    }
    atomic_json(paths["facts"], {
        "completed": True,
        "schema": SCHEMA,
        "layer": layer,
        "rate": rate,
        "groups": 29,
        "policies": list(DECODE_POLICIES),
        "route_modes": list(map(str, config["route_modes"])),
        "quality_rows": len(quality_rows),
        "propagation_rows": len(propagation_rows),
        "zero_rows": len(zero_rows),
        "baseline_seconds": baseline_seconds,
        "candidate_seconds": candidate_seconds,
        "files": files,
        "sequence_shaped_delta_constructed": False,
        "candidate_cache_reused": False,
        "delta_execution": "selected_states_reconstructed_on_live_cached_activation",
    })
    print(
        f"[complete] layer={layer} rate={rate} quality={len(quality_rows)} "
        f"propagation={len(propagation_rows)} zero={len(zero_rows)}",
        flush=True,
    )


def run_phase(args: argparse.Namespace) -> None:
    hardware = _gpu_gate(args.config_data)
    checkpoint_index = args.checkpoint / "model.safetensors.index.json"
    if sha256(checkpoint_index) != str(args.config_data["checkpoint_index_sha256"]):
        raise RuntimeError("checkpoint index changed")
    if sha256(args.trees) != str(args.config_data["selected_tree_sha256"]):
        raise RuntimeError("selected-tree file changed")
    _, _, _, banks = _policy_banks(args.config_data)
    model, tokenizer, load_seconds = _load_model(args.checkpoint)
    encoded = _encoded_requests(tokenizer, model, args.config_data)
    atomic_json(args.output / "d1_cached_decode_tail_host_facts.json", {
        "schema": SCHEMA,
        "completed": True,
        **hardware,
        "model_load_seconds": load_seconds,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "host": platform.node(),
    })
    try:
        for layer in map(int, args.config_data["injection_layers"]):
            layer_banks = [
                banks[(layer, int(rate))]
                for rate in args.config_data["page_caps"]
            ]
            decoded, proxy, beta = _prepare_live_reconstruction(
                args.checkpoint, args.trees, args.fit_dir, layer,
                layer_banks, args.reconstruction_workers,
            )
            for bank in layer_banks:
                run_cell(
                    model, encoded, bank, decoded, proxy, beta,
                    args.config_data, args.output,
                )
            del decoded, proxy
            gc.collect()
    finally:
        del encoded, tokenizer, model
        gc.collect()
        torch.cuda.empty_cache()


def finalize_phase(args: argparse.Namespace) -> None:
    counts = expected_grid_counts(args.config_data)
    quality_parts = []
    propagation_parts = []
    zero_parts = []
    cell_facts = {}
    for layer in map(int, args.config_data["injection_layers"]):
        for rate in map(int, args.config_data["page_caps"]):
            paths = _cell_paths(args.output, layer, rate)
            if not _completed_cell(paths, layer, rate):
                raise RuntimeError(f"cached-decode tail cell is incomplete: {(layer, rate)}")
            quality_parts.append(pd.read_parquet(paths["quality"]))
            propagation_parts.append(pd.read_parquet(paths["propagation"]))
            zero_parts.append(pd.read_parquet(paths["zero"]))
            cell_facts[f"layer_{layer:02d}_rate_{rate}"] = load_json(paths["facts"])
    quality = add_live_minus_frozen(pd.concat(quality_parts, ignore_index=True))
    propagation = pd.concat(propagation_parts, ignore_index=True)
    zero = pd.concat(zero_parts, ignore_index=True)
    if len(quality) != counts["quality_rows"]:
        raise RuntimeError("final cached-decode quality grid is incomplete")
    if len(propagation) != counts["propagation_rows"]:
        raise RuntimeError("final cached-decode propagation grid is incomplete")
    if len(zero) != counts["logical_zero_dose_rows"]:
        raise RuntimeError("final cached-decode zero-dose grid is incomplete")
    if not bool(zero["all_exact"].all()):
        raise RuntimeError("a final zero-dose cache or output gate failed")
    for frame, name in ((quality, QUALITY), (propagation, PROPAGATION), (zero, ZERO)):
        if frame.empty or frame.isnull().any().any():
            raise RuntimeError(f"final table is empty or contains nulls: {name}")
        atomic_parquet(args.output / name, frame)
    outputs = {
        name: {
            "sha256": sha256(args.output / name),
            "bytes": (args.output / name).stat().st_size,
            "rows": len(frame),
        }
        for name, frame in (
            (QUALITY, quality), (PROPAGATION, propagation), (ZERO, zero),
        )
    }
    atomic_json(args.output / RUN_FACTS, {
        "completed": True,
        "schema": SCHEMA,
        "config_sha256": sha256(args.config),
        "counts": counts,
        "outputs": outputs,
        "cells": cell_facts,
        "zero_dose_all_exact": True,
        "terminal_metric_scope": "current_token_only",
        "exact_prefix_cache_cloned_per_candidate": True,
        "post_token_cache_retained_and_audited": True,
        "sequence_shaped_delta_constructed": False,
        "scientific_boundary": (
            "Three-request exact-prefix cached single-token smoke study. One "
            "injected layer per candidate; no joint all-layer compression, "
            "generated rollout, deployable predictor, or quality claim."
        ),
        "test_rows_admitted_or_used": False,
    })
    print(f"[finalized] {counts}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("validate", "run", "finalize"), required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--trees", type=Path)
    parser.add_argument("--fit-dir", type=Path)
    parser.add_argument("--reconstruction-workers", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.config_data = load_json(args.config)
    validate_cached_decode_tail_config(args.config_data)
    args.output = args.output or Path(str(args.config_data["output_root"]))
    if args.phase == "run" and any(
        value is None for value in (args.checkpoint, args.trees, args.fit_dir)
    ):
        parser.error("--checkpoint, --trees, and --fit-dir are required for --phase run")
    return args


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    if args.phase == "validate":
        validate_phase(args)
    elif args.phase == "run":
        run_phase(args)
    else:
        finalize_phase(args)


if __name__ == "__main__":
    main()
