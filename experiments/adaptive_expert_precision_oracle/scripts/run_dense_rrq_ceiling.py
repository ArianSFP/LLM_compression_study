#!/usr/bin/env python3
"""Bounded dense-prefix RRQ ceiling on the audited Qwen3.6 MoE pilot.

This run deliberately stops before selective packets.  It validates whether
serialized recurrent two-bit stages provide a useful action space for the
existing exact-ID/page allocator.  Quantizer construction and calibration use
training/validation requests only; test results are paired exploratory results
because PR #2 already informed the RRQ hypothesis.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


PROJECTIONS = ("gate", "up", "down")
MATRIX_WEIGHTS = 2048 * 512
EXPERT_WEIGHTS = 3 * MATRIX_WEIGHTS


def silu(value: torch.Tensor) -> torch.Tensor:
    return value * torch.sigmoid(value)


def qenergy(error: torch.Tensor, proxy: torch.Tensor, beta: float) -> torch.Tensor:
    return torch.sum(error * error, dim=-1) + beta * torch.sum((error @ proxy) ** 2, dim=-1)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def save_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    temporary.replace(path)


def spec_label(spec: Any) -> str:
    return f"{spec.name}_g{spec.group_size}_{spec.scale_storage}"


def spec_record(spec: Any) -> dict[str, Any]:
    return {"name": spec.name, "group_size": int(spec.group_size), "scale_storage": spec.scale_storage}


def occurrence(d: dict[str, np.ndarray], expert: int, split: str, maximum: int) -> tuple[np.ndarray, np.ndarray]:
    found = np.argwhere((d["expert_ids"] == expert) & (d["split"][:, None] == split))
    found = found[:maximum]
    if not len(found):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return found[:, 0], found[:, 1]


def physical_stream_bytes(encoding: Any, page_size: int) -> int:
    return encoding.physical_read_bytes(page_size)


def series_prefix_bytes(series: Any, depth: int, page_size: int) -> tuple[int, int]:
    logical = sum(stage.serialized_bytes() - stage.stream_bytes()["header"] for stage in series.stages[:depth])
    physical = sum(physical_stream_bytes(stage, page_size) for stage in series.stages[:depth])
    return int(logical), int(physical)


def series_capacity(series: Any) -> int:
    return int(sum(stage.serialized_bytes() for stage in series.stages))


def choose_search_experts(experts: list[int], strata: dict[int, str], d: dict[str, np.ndarray]) -> list[int]:
    selected = []
    for stratum in ("cold", "median", "hot"):
        candidates = [expert for expert in experts if strata[expert] == stratum]
        candidates.sort(key=lambda expert: int(np.sum(d["expert_ids"][d["split"] == "validation"] == expert)), reverse=True)
        if candidates:
            selected.append(candidates[0])
    return selected


def nested_atom_prefixes(atoms: np.ndarray) -> tuple[list[np.ndarray], list[tuple[int, int]], int]:
    """PR #2 INT16-master high-bit prefixes with stored FP16 scales."""
    value = np.asarray(atoms, dtype=np.float32)
    maximum = np.max(np.abs(value), axis=0, keepdims=True)
    fitted = np.where(maximum > 0, maximum / 32767.0, 1.0).astype(np.float32)
    minimum_fp16 = np.float32(np.nextafter(np.float16(0), np.float16(1)))
    stored = np.maximum(fitted, minimum_fp16).astype(np.float16)
    scale = stored.astype(np.float32)
    master = np.clip(np.rint(value / scale), -32767, 32767).astype(np.int32)
    prefixes = [np.zeros_like(value)]
    bytes_by_depth = [(0, 0)]
    for bits in (2, 4, 8):
        shift = 16 - bits
        restored = ((master >> shift) << shift).astype(np.float32) * scale
        prefixes.append(restored)
        code_bytes = math.ceil(value.size * bits / 8)
        scale_bytes = value.shape[1] * 2
        logical = code_bytes + scale_bytes
        physical = math.ceil(code_bytes / 512) * 512 + math.ceil(scale_bytes / 512) * 512
        bytes_by_depth.append((logical, physical))
    # Capacity is the true INT16 master plus FP16 scales and a fixed header.
    capacity = value.size * 2 + value.shape[1] * 2 + 64
    return prefixes, bytes_by_depth, capacity


def direct_atom_prefixes(atoms: np.ndarray) -> tuple[list[np.ndarray], list[tuple[int, int]], int]:
    """Independent, zero-containing, per-atom MSE quantizers at 2/4/8 bits."""
    value = np.asarray(atoms, dtype=np.float32)
    prefixes = [np.zeros_like(value)]
    bytes_by_depth = [(0, 0)]
    capacity = 64
    for bits in (2, 4, 8):
        qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        scale = np.maximum(np.max(np.abs(value), axis=0) / max(qmax, 1), 1e-12).astype(np.float64)
        for _ in range(12):
            codes = np.clip(np.rint(value / scale[None]), qmin, qmax)
            scale = np.maximum(np.sum(value * codes, axis=0) / np.maximum(np.sum(codes * codes, axis=0), 1e-30), 1e-12)
        minimum_fp16 = np.float32(np.nextafter(np.float16(0), np.float16(1)))
        stored = np.maximum(scale, minimum_fp16).astype(np.float16)
        decoded = stored.astype(np.float32)
        codes = np.clip(np.rint(value / decoded[None]), qmin, qmax)
        prefixes.append((codes * decoded[None]).astype(np.float32))
        code_bytes = math.ceil(value.size * bits / 8)
        scale_bytes = value.shape[1] * 2
        logical = code_bytes + scale_bytes
        physical = math.ceil(code_bytes / 512) * 512 + math.ceil(scale_bytes / 512) * 512
        bytes_by_depth.append((logical, physical))
        capacity += logical + 64
    return prefixes, bytes_by_depth, capacity


def reference_hashes(config: dict[str, Any]) -> dict[str, Any]:
    cache_name = os.environ.get("RRQ_REFERENCE_HASH_CACHE")
    if cache_name and Path(cache_name).exists():
        cached = json.loads(Path(cache_name).read_text())
        return cached.get("reference_hashes", cached)
    result = {}
    for label, raw in (("production_reference", config["remote"]["gguf_model"]), ("capture_manifest", Path(config["remote"]["capture_root"]) / "CORPUS_MANIFEST.json")):
        path = Path(raw)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 << 20), b""):
                digest.update(block)
        result[label] = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    if cache_name:
        Path(cache_name).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def make_search_units(states: dict[int, dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    units = []
    for layer, state in states.items():
        d = state["data"]
        for expert in state["search_experts"]:
            ei = state["experts"].index(expert)
            records, _ = occurrence(d, expert, "validation", maximum)
            if not len(records):
                continue
            x = np.asarray(d["x"][records], dtype=np.float32)
            gate = x @ state["reference"]["gate"][ei].T
            up = x @ state["reference"]["up"][ei].T
            href = (gate / (1.0 + np.exp(-gate))) * up
            units.append({"layer": layer, "expert": expert, "expert_index": ei, "x": x, "href": href, "proxy": state["proxy"]})
    return units


def score_path(
    states: dict[int, dict[str, Any]], units: list[dict[str, Any]], projection: str,
    specs: tuple[Any, ...], moment_kind: str = "primary",
) -> tuple[float, dict[str, float]]:
    from oracle_study.rrq import build_rrq_series
    def score_unit(unit: dict[str, Any]) -> tuple[float, float, int, int]:
        state = states[unit["layer"]]
        ei = unit["expert_index"]
        reference = state["reference"][projection][ei]
        base = state["base_fp16"][projection][ei]
        if projection in ("gate", "up"):
            moment = state["moment_x"]
            inputs = unit["x"]
        else:
            moment = None if moment_kind == "weight_only" else state[moment_kind]
            inputs = unit["href"]
        series = build_rrq_series(base, reference, specs, moment)
        base_error = inputs @ (reference - base).T
        error = inputs @ (reference - series.prefix(len(specs))).T
        if projection == "down":
            proxy = unit["proxy"]
            base_values = np.sum(base_error.astype(np.float64) ** 2, axis=1) + state["proxy_beta"] * np.sum((base_error @ proxy).astype(np.float64) ** 2, axis=1)
            values = np.sum(error.astype(np.float64) ** 2, axis=1) + state["proxy_beta"] * np.sum((error @ proxy).astype(np.float64) ** 2, axis=1)
        else:
            base_values = np.sum(base_error.astype(np.float64) ** 2, axis=1)
            values = np.sum(error.astype(np.float64) ** 2, axis=1)
        result = float(values.sum()), float(base_values.sum()), int(np.sum(values > base_values)), len(values)
        del series
        return result
    workers = max(1, min(int(os.environ.get("RRQ_SEARCH_WORKERS", "4")), len(units)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pieces = list(pool.map(score_unit, units))
    numerator = sum(value[0] for value in pieces)
    denominator = sum(value[1] for value in pieces)
    worsened = sum(value[2] for value in pieces)
    total = sum(value[3] for value in pieces)
    return numerator / max(denominator, 1e-30), {"numerator": numerator, "denominator": denominator, "fraction_worsened": worsened / max(total, 1), "invocations": total}


def search_paths(states: dict[int, dict[str, Any]], units: list[dict[str, Any]], candidates: list[Any], beam_width: int, output: Path) -> dict[str, Any]:
    from oracle_study.rrq import quantize_int2_stage

    def extend(
        projection: str, parents: list[np.ndarray] | None, spec: Any,
    ) -> tuple[float, dict[str, float], list[np.ndarray]]:
        def extend_unit(index_and_unit: tuple[int, dict[str, Any]]) -> tuple[float, float, int, int, np.ndarray]:
            index, unit = index_and_unit
            state = states[unit["layer"]]
            ei = unit["expert_index"]
            reference = state["reference"][projection][ei]
            base = state["base_fp16"][projection][ei]
            current = base if parents is None else parents[index]
            if projection in ("gate", "up"):
                moment = state["moment_x"]
                inputs = unit["x"]
            else:
                moment = None
                inputs = unit["href"]
            stage = quantize_int2_stage(reference - current, spec.group_size, spec.name, moment, spec.scale_storage)
            updated = current + stage.reconstruction
            base_error = inputs @ (reference - base).T
            error = inputs @ (reference - updated).T
            if projection == "down":
                proxy = unit["proxy"]
                base_values = np.sum(base_error.astype(np.float64) ** 2, axis=1) + state["proxy_beta"] * np.sum((base_error @ proxy).astype(np.float64) ** 2, axis=1)
                values = np.sum(error.astype(np.float64) ** 2, axis=1) + state["proxy_beta"] * np.sum((error @ proxy).astype(np.float64) ** 2, axis=1)
            else:
                base_values = np.sum(base_error.astype(np.float64) ** 2, axis=1)
                values = np.sum(error.astype(np.float64) ** 2, axis=1)
            return float(values.sum()), float(base_values.sum()), int(np.sum(values > base_values)), len(values), updated

        workers = max(1, min(int(os.environ.get("RRQ_SEARCH_WORKERS", "4")), len(units)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pieces = list(pool.map(extend_unit, enumerate(units)))
        numerator = sum(value[0] for value in pieces)
        denominator = sum(value[1] for value in pieces)
        worsened = sum(value[2] for value in pieces)
        total = sum(value[3] for value in pieces)
        facts = {"numerator": numerator, "denominator": denominator, "fraction_worsened": worsened / max(total, 1), "invocations": total}
        return numerator / max(denominator, 1e-30), facts, [value[4] for value in pieces]

    result: dict[str, Any] = {}
    for projection in PROJECTIONS:
        homogeneous = []
        first_stage = []
        for spec in candidates:
            path = (spec, spec, spec)
            matrices = None
            for depth in range(1, 4):
                score, facts, matrices = extend(projection, matrices, spec)
                if depth == 1:
                    first_stage.append((score, (spec,), facts, matrices))
            homogeneous.append((score, path, facts))
        homogeneous.sort(key=lambda item: item[0])
        first_stage.sort(key=lambda item: item[0])
        beam = first_stage[:beam_width]
        history = [[{"score": score, "path": [spec_record(v) for v in path], **facts} for score, path, facts, _ in beam]]
        print(json.dumps({"rrq_search": projection, "depth": 1, "best_score": beam[0][0], "path": [spec_label(v) for v in beam[0][1]]}), flush=True)
        for depth in range(2, 4):
            expanded = []
            for _, prefix, _, parent_matrices in beam:
                for spec in candidates:
                    path = prefix + (spec,)
                    score, facts, matrices = extend(projection, parent_matrices, spec)
                    expanded.append((score, path, facts, matrices))
            expanded.sort(key=lambda item: item[0])
            beam = expanded[:beam_width]
            history.append([{"score": score, "path": [spec_record(v) for v in path], **facts} for score, path, facts, _ in beam])
            print(json.dumps({"rrq_search": projection, "depth": depth, "best_score": beam[0][0], "path": [spec_label(v) for v in beam[0][1]]}), flush=True)
        result[projection] = {
            "heterogeneous": [spec_record(v) for v in beam[0][1]],
            "heterogeneous_score": beam[0][0],
            "homogeneous": [spec_record(v) for v in homogeneous[0][1]],
            "homogeneous_score": homogeneous[0][0],
            "homogeneous_grid": [{"score": score, "path": [spec_record(v) for v in path], **facts} for score, path, facts in homogeneous],
            "beam_history": history,
        }
        atomic_json(output / "quantizer_selection.json", result)
    return result


def records_to_specs(records: Iterable[dict[str, Any]]) -> tuple[Any, ...]:
    from oracle_study.rrq import QuantizerSpec
    return tuple(QuantizerSpec(item["name"], int(item["group_size"]), item.get("scale_storage", "fp16")) for item in records)


def sequential_mix_moments(states: dict[int, dict[str, Any]], selected: dict[str, Any], maximum: int) -> None:
    from oracle_study.rrq import build_rrq_series
    gspecs = records_to_specs(selected["gate"]["heterogeneous"])
    uspecs = records_to_specs(selected["up"]["heterogeneous"])
    for layer, state in states.items():
        sumsq = np.zeros(512, dtype=np.float64)
        count = 0
        for ei, expert in enumerate(state["experts"]):
            records, _ = occurrence(state["data"], expert, "train", maximum)
            if not len(records):
                continue
            x = np.asarray(state["data"]["x"][records], dtype=np.float32)
            gs = build_rrq_series(state["base_fp16"]["gate"][ei], state["reference"]["gate"][ei], gspecs, state["moment_x"])
            us = build_rrq_series(state["base_fp16"]["up"][ei], state["reference"]["up"][ei], uspecs, state["moment_x"])
            gout = [x @ gs.prefix(depth).T for depth in range(4)]
            uout = [x @ us.prefix(depth).T for depth in range(4)]
            for kg in range(4):
                for ku in range(4):
                    h = (gout[kg] / (1.0 + np.exp(-gout[kg]))) * uout[ku]
                    sumsq += np.sum(h.astype(np.float64) ** 2, axis=0)
                    count += len(h)
        state["moment_h_sequential_mix"] = sumsq / max(count, 1)
        state["sequential_mix_samples"] = count


def choose_down_calibration(states: dict[int, dict[str, Any]], units: list[dict[str, Any]], selected: dict[str, Any]) -> dict[str, Any]:
    path = records_to_specs(selected["down"]["heterogeneous"])
    outcomes = {}
    for label, key in (("weight_only", "weight_only"), ("authoritative_h", "moment_h_authoritative"), ("sequential_mix", "moment_h_sequential_mix")):
        score, facts = score_path(states, units, "down", path, key)
        outcomes[label] = {"score": score, **facts}
    winner = min(outcomes, key=lambda name: outcomes[name]["score"])
    return {"selected": winner, "validation": outcomes}


def moment_for(state: dict[str, Any], projection: str, calibration: str) -> np.ndarray | None:
    if projection in ("gate", "up"):
        return state["moment_x"]
    return {"weight_only": None, "authoritative_h": state["moment_h_authoritative"], "sequential_mix": state["moment_h_sequential_mix"]}[calibration]


def prepare_rrq_method(state: dict[str, Any], ei: int, paths: dict[str, tuple[Any, ...]], calibration: str, transformed: bool, page_size: int) -> dict[str, Any]:
    from oracle_study.rrq import build_rrq_series
    prepared: dict[str, Any] = {"kind": "rrq", "projection": {}, "resident_transform_bytes": 0}
    for projection in PROJECTIONS:
        reference = state["reference"][projection][ei]
        base = state["base_fp16"][projection][ei]
        if transformed and projection in ("gate", "up"):
            transform = state["transform"]
            atoms = (reference - base) @ transform.synthesis
            moment = state["moment_z"]
            series = build_rrq_series(np.zeros_like(atoms), atoms, paths[projection], moment)
            prefixes = [series.prefix(depth) for depth in range(4)]
            kind = "atoms"
            prepared["resident_transform_bytes"] = int(transform.analysis.nbytes)
        else:
            series = build_rrq_series(base, reference, paths[projection], moment_for(state, projection, calibration))
            prefixes = [series.prefix(depth) for depth in range(4)]
            kind = "matrix"
        prepared["projection"][projection] = {
            "kind": kind,
            "prefixes": prefixes,
            "target": atoms if kind == "atoms" else reference,
            "series": series,
            "bytes": [series_prefix_bytes(series, depth, page_size) for depth in range(4)],
            "capacity": series_capacity(series),
        }
    return prepared


def prepare_atom_control(state: dict[str, Any], ei: int, direct: bool) -> dict[str, Any]:
    prepared: dict[str, Any] = {"kind": "atom_control", "projection": {}, "resident_transform_bytes": int(state["transform"].analysis.nbytes)}
    builder = direct_atom_prefixes if direct else nested_atom_prefixes
    for projection in PROJECTIONS:
        reference = state["reference"][projection][ei]
        base = state["base_fp16"][projection][ei]
        if projection in ("gate", "up"):
            atoms = (reference - base) @ state["transform"].synthesis
            kind = "atoms"
        else:
            atoms = reference - base
            kind = "atoms_native"
        prefixes, bytes_512, capacity = builder(atoms)
        prepared["projection"][projection] = {"kind": kind, "prefixes": prefixes, "bytes_512": bytes_512, "capacity": capacity}
    return prepared


def method_outputs(prepared: dict[str, Any], state: dict[str, Any], ei: int, x: torch.Tensor, page_size: int) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], dict[str, list[tuple[int, int]]]]:
    outputs: dict[str, Any] = {}
    bytes_by_projection: dict[str, list[tuple[int, int]]] = {}
    analysis = torch.from_numpy(state["transform"].analysis).to(x.device)
    z = x @ analysis.T
    for projection in ("gate", "up"):
        item = prepared["projection"][projection]
        base = torch.from_numpy(state["base_fp16"][projection][ei]).to(x.device)
        if item["kind"] == "matrix":
            outputs[projection] = [x @ torch.from_numpy(value).to(x.device).T for value in item["prefixes"]]
        else:
            base_output = x @ base.T
            outputs[projection] = [base_output + z @ torch.from_numpy(value).to(x.device).T for value in item["prefixes"]]
        if prepared["kind"] == "rrq":
            bytes_by_projection[projection] = item["bytes"]
        else:
            bytes_by_projection[projection] = [(logical, math.ceil(logical / page_size) * page_size) if logical else (0, 0) for logical, _ in item["bytes_512"]]
    down_item = prepared["projection"]["down"]
    down_base = torch.from_numpy(state["base_fp16"]["down"][ei]).to(x.device)
    if down_item["kind"] == "matrix":
        down = [torch.from_numpy(value).to(x.device) for value in down_item["prefixes"]]
    else:
        down = [down_base + torch.from_numpy(value).to(x.device) for value in down_item["prefixes"]]
    if prepared["kind"] == "rrq":
        bytes_by_projection["down"] = down_item["bytes"]
    else:
        bytes_by_projection["down"] = [(logical, math.ceil(logical / page_size) * page_size) if logical else (0, 0) for logical, _ in down_item["bytes_512"]]
    return outputs["gate"], outputs["up"], down, bytes_by_projection


def evaluate_method(
    rows: list[dict[str, Any]], projection_rows: list[dict[str, Any]], stage_rows: list[dict[str, Any]],
    method: str, prepared: dict[str, Any], state: dict[str, Any], ei: int, expert: int,
    split: str, records: np.ndarray, ranks: np.ndarray, page_size: int,
) -> None:
    if not len(records):
        return
    device = "cuda"
    d = state["data"]
    x = torch.from_numpy(np.asarray(d["x"][records], dtype=np.float32)).to(device)
    reference = {name: torch.from_numpy(state["reference"][name][ei]).to(device) for name in PROJECTIONS}
    base = {name: torch.from_numpy(state["base_fp16"][name][ei]).to(device) for name in PROJECTIONS}
    gref, uref = x @ reference["gate"].T, x @ reference["up"].T
    href = silu(gref) * uref
    yref = href @ reference["down"].T
    gbase, ubase = x @ base["gate"].T, x @ base["up"].T
    hbase = silu(gbase) * ubase
    ybase = hbase @ base["down"].T
    proxy = torch.from_numpy(state["proxy"]).to(device)
    denominator = qenergy(yref - ybase, proxy, state["proxy_beta"])
    gouts, uouts, down_matrices, byte_table = method_outputs(prepared, state, ei, x, page_size)
    capacity = sum(prepared["projection"][name]["capacity"] for name in PROJECTIONS)
    reference_bytes = sum(state["storage"][name]["bytes_per_expert"] for name in PROJECTIONS)
    storage_multiplier = (reference_bytes + capacity) / reference_bytes
    if storage_multiplier > 5.0 + 1e-9:
        raise RuntimeError(f"5x cap exceeded: {method=} {storage_multiplier=}")
    for projection, outputs, target in (("gate", gouts, gref), ("up", uouts, uref)):
        base_error = target - outputs[0]
        base_damage = torch.sum(base_error * base_error, dim=1)
        for depth in range(4):
            error = target - outputs[depth]
            damage = torch.sum(error * error, dim=1)
            logical, physical = byte_table[projection][depth]
            for sample in range(len(records)):
                projection_rows.append({
                    "request_id": str(d["request_id"][records[sample]]), "sequence_id": str(d["sequence_id"][records[sample]]),
                    "position": int(d["position"][records[sample]]), "split": split, "layer": state["layer"], "expert_id": expert,
                    "expert_stratum": state["strata"][expert], "projection": projection, "method": method, "depth": depth,
                    "logical_bytes": logical, "physical_bytes": physical, "physical_bpw": 8 * physical / MATRIX_WEIGHTS,
                    "recovery": float(1 - damage[sample] / max(float(base_damage[sample]), 1e-20)),
                    "damage": float(damage[sample]), "base_damage": float(base_damage[sample]),
                })
    down_base_error = yref - href @ down_matrices[0].T
    down_base_damage = qenergy(down_base_error, proxy, state["proxy_beta"])
    for depth in range(4):
        ydown = href @ down_matrices[depth].T
        damage = qenergy(yref - ydown, proxy, state["proxy_beta"])
        logical, physical = byte_table["down"][depth]
        for sample in range(len(records)):
            projection_rows.append({
                "request_id": str(d["request_id"][records[sample]]), "sequence_id": str(d["sequence_id"][records[sample]]),
                "position": int(d["position"][records[sample]]), "split": split, "layer": state["layer"], "expert_id": expert,
                "expert_stratum": state["strata"][expert], "projection": "down", "method": method, "depth": depth,
                "down_input": "authoritative", "logical_bytes": logical, "physical_bytes": physical,
                "physical_bpw": 8 * physical / MATRIX_WEIGHTS,
                "recovery": float(1 - damage[sample] / max(float(down_base_damage[sample]), 1e-20)),
                "damage": float(damage[sample]), "base_damage": float(down_base_damage[sample]),
            })
    for kg, ku, kd in itertools.product(range(4), repeat=3):
        h = silu(gouts[kg]) * uouts[ku]
        yhat = h @ down_matrices[kd].T
        damage = qenergy(yref - yhat, proxy, state["proxy_beta"])
        logical = byte_table["gate"][kg][0] + byte_table["up"][ku][0] + byte_table["down"][kd][0]
        physical = byte_table["gate"][kg][1] + byte_table["up"][ku][1] + byte_table["down"][kd][1]
        error = yref - yhat
        extra_flops = 2 * MATRIX_WEIGHTS * (kg + ku + kd)
        if prepared["resident_transform_bytes"] and (kg or ku):
            extra_flops += 2 * 2048 * 2048
        for sample in range(len(records)):
            record = int(records[sample]); rank = int(ranks[sample]); logits = np.sort(d["router_logits"][record])
            rows.append({
                "request_id": str(d["request_id"][record]), "sequence_id": str(d["sequence_id"][record]), "position": int(d["position"][record]),
                "split": split, "layer": state["layer"], "expert_id": expert, "expert_stratum": state["strata"][expert],
                "expert_train_frequency": state["frequencies"][expert], "router_rank": rank + 1,
                "router_coefficient": float(d["router_weights"][record, rank]), "router_boundary_margin": float(logits[-8] - logits[-9]),
                "method": method, "gate_depth": kg, "up_depth": ku, "down_depth": kd, "tuple": f"({kg},{ku},{kd})",
                "logical_bytes": logical, "physical_bytes": physical, "logical_bpw": 8 * logical / EXPERT_WEIGHTS,
                "physical_bpw": 8 * physical / EXPERT_WEIGHTS, "page_amplification": physical / max(logical, 1),
                "gate_physical_bpw": 8 * byte_table["gate"][kg][1] / MATRIX_WEIGHTS,
                "up_physical_bpw": 8 * byte_table["up"][ku][1] / MATRIX_WEIGHTS,
                "down_physical_bpw": 8 * byte_table["down"][kd][1] / MATRIX_WEIGHTS,
                "gate_logical_bpw": 8 * byte_table["gate"][kg][0] / MATRIX_WEIGHTS,
                "up_logical_bpw": 8 * byte_table["up"][ku][0] / MATRIX_WEIGHTS,
                "down_logical_bpw": 8 * byte_table["down"][kd][0] / MATRIX_WEIGHTS,
                "recovery": float(1 - damage[sample] / max(float(denominator[sample]), 1e-20)),
                "damage": float(damage[sample]), "base_damage": float(denominator[sample]),
                "relative_output_error": float(torch.linalg.norm(error[sample]) / max(float(torch.linalg.norm(yref[sample])), 1e-20)),
                "absolute_error": float(torch.sqrt(torch.mean(error[sample] * error[sample]))),
                "cosine_similarity": float(torch.nn.functional.cosine_similarity(yhat[sample][None], yref[sample][None])),
                "external_storage_multiplier": storage_multiplier, "resident_transform_bytes": prepared["resident_transform_bytes"],
                "extra_flops": extra_flops, "extra_flops_reference_expert_ratio": extra_flops / (2 * EXPERT_WEIGHTS),
            })
    if prepared["kind"] == "rrq":
        for projection in PROJECTIONS:
            series = prepared["projection"][projection]["series"]
            inputs = x if projection in ("gate", "up") else href
            if prepared["projection"][projection]["kind"] == "atoms":
                inputs = x @ torch.from_numpy(state["transform"].analysis).to(device).T
            for stage_index, stage in enumerate(series.stages, start=1):
                target_matrix = prepared["projection"][projection]["target"]
                residual_before_np = target_matrix - series.prefix(stage_index - 1)
                residual_after_np = target_matrix - series.prefix(stage_index)
                residual_before = torch.from_numpy(residual_before_np).to(device)
                stage_matrix = torch.from_numpy(stage.reconstruction).to(device)
                before_action = inputs @ residual_before.T
                correction_action = inputs @ stage_matrix.T
                after_action = before_action - correction_action
                if projection == "down":
                    before_energy = qenergy(before_action, proxy, state["proxy_beta"])
                    after_energy = qenergy(after_action, proxy, state["proxy_beta"])
                else:
                    before_energy = torch.sum(before_action * before_action, dim=1)
                    after_energy = torch.sum(after_action * after_action, dim=1)
                cosine = torch.sum(before_action * correction_action, dim=1) / torch.clamp(torch.linalg.norm(before_action, dim=1) * torch.linalg.norm(correction_action, dim=1), min=1e-20)
                from oracle_study.rrq import unpack_q2_codes
                decoded_codes = unpack_q2_codes(stage.packed_codes, stage.groups * stage.group_size)
                unique, counts = np.unique(decoded_codes, return_counts=True)
                code_hist = {str(int(key)): int(value) for key, value in zip(unique, counts)}
                abs_before = np.abs(residual_before_np.astype(np.float64)).reshape(-1)
                abs_after = np.abs(residual_after_np.astype(np.float64)).reshape(-1)
                if stage.zero_points is not None:
                    zero_codes = np.repeat(stage.zero_points.astype(np.uint8), stage.group_size)
                    zero_code_fraction = float(np.mean(decoded_codes == zero_codes))
                elif stage.quantizer.startswith("signed_zero"):
                    zero_code_fraction = float(np.mean(decoded_codes == 2))
                else:
                    zero_code_fraction = float("nan")
                stage_rows.append({
                    "split": split, "layer": state["layer"], "expert_id": expert, "projection": projection, "method": method,
                    "stage": stage_index, "quantizer": stage.quantizer, "group_size": stage.group_size,
                    "weight_contraction": series.residual_norms[stage_index] ** 2 / max(series.residual_norms[stage_index - 1] ** 2, 1e-30),
                    "functional_contraction": float(after_energy.sum() / torch.clamp(before_energy.sum(), min=1e-30)),
                    "fraction_invocations_worsened": float(torch.mean((after_energy > before_energy).float())),
                    "median_stage_residual_cosine": float(torch.median(cosine)), "code_byte_histogram_json": json.dumps(code_hist, sort_keys=True),
                    "zero_code_fraction": zero_code_fraction,
                    "pmr_before": float(abs_before.max() / max(float(np.mean(abs_before)), 1e-30)),
                    "pmr_after": float(abs_after.max() / max(float(np.mean(abs_after)), 1e-30)),
                    "max_range_contraction": float(abs_after.max() / max(float(abs_before.max()), 1e-30)),
                    "logical_bytes": prepared["projection"][projection]["bytes"][stage_index][0] - prepared["projection"][projection]["bytes"][stage_index - 1][0],
                    "physical_bytes": prepared["projection"][projection]["bytes"][stage_index][1] - prepared["projection"][projection]["bytes"][stage_index - 1][1],
                })
    del x, reference, base, gref, uref, href, yref, gbase, ubase, hbase, ybase, proxy, gouts, uouts, down_matrices
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    experiment = args.config.resolve().parents[1]
    sys.path[:0] = [str(experiment / "src"), str(experiment / "scripts"), config["remote"]["gguf_python"]]
    import gguf
    from oracle_study.rrq import QuantizerSpec, build_rrq_series, encode_legacy_q2, scale_diagnostics, serialize_encoding
    from oracle_study.weights import alignment_statistics, load_gguf_experts, load_hf_experts
    from run_phase_a_remote import proxy_gradients, select_experts
    from run_phase_a_v2_remote import generalized_full_rank

    args.output.mkdir(parents=True, exist_ok=True)
    package_root = args.output / "serialized_stage_packages"
    package_root.mkdir(exist_ok=True)
    loaded = np.load(args.captures, allow_pickle=False)
    data = {name: loaded[name] for name in loaded.files}
    split_requests = {split: set(map(str, data["request_id"][data["split"] == split])) for split in ("train", "validation", "test")}
    if any(split_requests[left] & split_requests[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("request-level split leakage")
    reader = gguf.GGUFReader(config["remote"]["gguf_model"])
    states: dict[int, dict[str, Any]] = {}
    facts: dict[str, Any] = {
        "started_unix": time.time(), "run_id": config["run_id"], "host": platform.node(), "platform": platform.platform(),
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "split_requests": {key: len(value) for key, value in split_requests.items()}, "test_status": "exploratory; PR #2 outcomes informed the RRQ hypothesis",
        "reference_hashes": reference_hashes(config), "layers": {},
    }
    for layer in config["layers"]:
        mask = data["layer"] == layer
        d = {name: value[mask] for name, value in data.items()}
        strata, frequencies = select_experts(d["expert_ids"], d["split"], int(config["experts_per_frequency_stratum"]))
        experts = sorted(strata)
        reference, storage = load_gguf_experts(reader, layer, experts, gguf)
        hf = load_hf_experts(Path(config["remote"]["hf_model"]), layer, experts)
        alignment = alignment_statistics(hf, reference)
        del hf
        train_x = np.asarray(d["x"][d["split"] == "train"], dtype=np.float32)
        moment_x = np.mean(train_x.astype(np.float64) ** 2, axis=0)
        base_fp16 = {name: [] for name in PROJECTIONS}; base_fp32 = {name: [] for name in PROJECTIONS}
        for name in ("gate", "up"):
            for matrix in reference[name]:
                base_fp16[name].append(encode_legacy_q2(matrix, int(config["base_group_size"]), moment_x, "fp16").reconstruction)
                base_fp32[name].append(encode_legacy_q2(matrix, int(config["base_group_size"]), moment_x, "fp32").reconstruction)
        train_h = []
        for ei, expert in enumerate(experts):
            records, _ = occurrence(d, expert, "train", int(config["max_train_invocations_per_expert"]))
            if not len(records):
                continue
            x = np.asarray(d["x"][records], dtype=np.float32)
            gate = x @ reference["gate"][ei].T; up = x @ reference["up"][ei].T
            train_h.append((gate / (1.0 + np.exp(-gate))) * up)
        all_h = np.concatenate(train_h)
        moment_h = np.mean(all_h.astype(np.float64) ** 2, axis=0)
        for matrix in reference["down"]:
            base_fp16["down"].append(encode_legacy_q2(matrix, int(config["base_group_size"]), moment_h, "fp16").reconstruction)
            base_fp32["down"].append(encode_legacy_q2(matrix, int(config["base_group_size"]), moment_h, "fp32").reconstruction)
        base_fp16 = {name: np.stack(value) for name, value in base_fp16.items()}; base_fp32 = {name: np.stack(value) for name, value in base_fp32.items()}
        rg = torch.from_numpy(reference["gate"] - base_fp16["gate"]).to("cuda").reshape(-1, 2048)
        ru = torch.from_numpy(reference["up"] - base_fp16["up"]).to("cuda").reshape(-1, 2048)
        operator = ((rg.T @ rg) + (ru.T @ ru)).cpu().numpy() / len(experts)
        del rg, ru; torch.cuda.empty_cache()
        transform = generalized_full_rank(train_x, operator)
        moment_z = np.mean((train_x @ transform.analysis.T).astype(np.float64) ** 2, axis=0)
        proxy, proxy_facts = proxy_gradients(d["xplus"], [d[f"h{h}_router_logits"] for h in range(1, 5)], d["split"])
        states[layer] = {
            "layer": layer, "data": d, "experts": experts, "strata": strata, "frequencies": frequencies,
            "search_experts": choose_search_experts(experts, strata, d), "reference": reference, "storage": storage,
            "base_fp16": base_fp16, "base_fp32": base_fp32, "moment_x": moment_x,
            "moment_h_authoritative": moment_h, "proxy": proxy, "proxy_beta": float(proxy_facts["beta"]),
            "transform": transform, "moment_z": moment_z,
        }
        facts["layers"][str(layer)] = {"experts": experts, "search_experts": states[layer]["search_experts"], "storage": storage, "alignment": alignment, "proxy": proxy_facts}
        atomic_json(args.output / "run_facts.json", facts)
        print(json.dumps({"prepared_layer": layer, "experts": experts, "search": states[layer]["search_experts"]}), flush=True)
    units = make_search_units(states, int(config["max_validation_invocations_per_expert"]))
    candidates = [QuantizerSpec(item["name"], int(item["group_size"]), config["scale_storage_primary"]) for item in config["search_candidates"]]
    selected_path_file = args.output / "quantizer_selection.json"
    selected = json.loads(selected_path_file.read_text()) if selected_path_file.exists() and all(name in json.loads(selected_path_file.read_text()) for name in PROJECTIONS) else search_paths(states, units, candidates, int(config["search_beam_width"]), args.output)
    # Four-centroid diagnostic: a higher-rate scalar upper control, never the
    # deployable primary format.
    centroid_spec = QuantizerSpec("centroid4", 64, config["scale_storage_primary"])
    centroid_scores = {projection: score_path(states, units, projection, (centroid_spec,) * 3, "weight_only" if projection == "down" else "primary")[0] for projection in PROJECTIONS}
    sequential_mix_moments(states, selected, int(config["max_train_invocations_per_expert"]))
    down_calibration = choose_down_calibration(states, units, selected)
    atomic_json(args.output / "quantizer_selection.json", {**selected, "centroid4_diagnostic_scores": centroid_scores, "down_calibration": down_calibration})
    selected["down_calibration"] = down_calibration
    hetero_paths = {name: records_to_specs(selected[name]["heterogeneous"]) for name in PROJECTIONS}
    homogeneous_paths = {name: records_to_specs(selected[name]["homogeneous"]) for name in PROJECTIONS}
    bf16_paths = {name: tuple(QuantizerSpec(spec.name, spec.group_size, "bf16") for spec in path) for name, path in hetero_paths.items()}
    fp32_paths = {name: tuple(QuantizerSpec(spec.name, spec.group_size, "fp32") for spec in path) for name, path in hetero_paths.items()}
    metrics: list[dict[str, Any]] = []
    projection_metrics: list[dict[str, Any]] = []
    stage_metrics: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []
    storage_rows: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []
    completed: set[tuple[int, int]] = set()
    checkpoint = args.output / "dense_tuple_metrics.parquet"
    if checkpoint.exists():
        old = pd.read_parquet(checkpoint)
        metrics = old.to_dict("records")
        completed = set(zip(old["layer"].astype(int), old["expert_id"].astype(int)))
        for path, target in ((args.output / "projection_metrics.parquet", projection_metrics), (args.output / "stage_diagnostics.parquet", stage_metrics), (args.output / "scale_controls.parquet", scale_rows), (args.output / "base_scale_sensitivity.parquet", base_rows)):
            if path.exists(): target.extend(pd.read_parquet(path).to_dict("records"))
    page_size = 512
    for layer, state in states.items():
        for ei, expert in enumerate(state["experts"]):
            if (layer, expert) in completed:
                continue
            methods = {
                "rrq_raw_heterogeneous": prepare_rrq_method(state, ei, hetero_paths, down_calibration["selected"], False, page_size),
                "rrq_raw_homogeneous": prepare_rrq_method(state, ei, homogeneous_paths, "weight_only", False, page_size),
                "rrq_raw_heterogeneous_bf16_scales": prepare_rrq_method(state, ei, bf16_paths, down_calibration["selected"], False, page_size),
                "rrq_raw_heterogeneous_fp32_scales_upper": prepare_rrq_method(state, ei, fp32_paths, down_calibration["selected"], False, page_size),
                "rrq_generalized_heterogeneous": prepare_rrq_method(state, ei, hetero_paths, down_calibration["selected"], True, page_size),
                "nested_int16_prefix_generalized_native": prepare_atom_control(state, ei, direct=False),
                "independent_direct_generalized_native": prepare_atom_control(state, ei, direct=True),
            }
            if down_calibration["selected"] != "weight_only":
                methods["rrq_raw_heterogeneous_weight_only_down"] = prepare_rrq_method(state, ei, hetero_paths, "weight_only", False, page_size)
            for method, prepared in methods.items():
                for split, maximum in (("validation", int(config["max_validation_invocations_per_expert"])), ("test", int(config["max_test_invocations_per_expert"]))):
                    records, ranks = occurrence(state["data"], expert, split, maximum)
                    evaluate_method(metrics, projection_metrics, stage_metrics, method, prepared, state, ei, expert, split, records, ranks, page_size)
                if method.startswith("rrq_"):
                    reference_bytes = sum(state["storage"][name]["bytes_per_expert"] for name in PROJECTIONS)
                    correction_bytes = sum(prepared["projection"][name]["capacity"] for name in PROJECTIONS)
                    storage_rows.append({"layer": layer, "expert_id": expert, "method": method, "reference_fallback_bytes": reference_bytes, "correction_capacity_bytes": correction_bytes, "external_storage_multiplier": (reference_bytes + correction_bytes) / reference_bytes})
            # Selected-series scale-format diagnostic and exact package hashes.
            for projection in PROJECTIONS:
                reference = state["reference"][projection][ei]; base = state["base_fp16"][projection][ei]
                for storage_format in ("fp16", "bf16", "fp32"):
                    specs = tuple(QuantizerSpec(spec.name, spec.group_size, storage_format) for spec in hetero_paths[projection])
                    series = build_rrq_series(base, reference, specs, moment_for(state, projection, down_calibration["selected"]))
                    for stage_index, stage in enumerate(series.stages, 1):
                        scale_rows.append({"layer": layer, "expert_id": expert, "projection": projection, "scale_storage": storage_format, "stage": stage_index, **scale_diagnostics(stage), "residual_norm": series.residual_norms[stage_index]})
                    if storage_format == "fp16":
                        for stage_index, stage in enumerate(series.stages, 1):
                            manifest = serialize_encoding(package_root, f"layer{layer}_expert{expert}_{projection}_stage{stage_index}", stage)
                            manifest["package_root_ephemeral"] = True
                # Serialized-scale base sensitivity uses identical codes and
                # only changes whether fitted scales were rounded to FP16.
                for split in ("validation", "test"):
                    records, _ = occurrence(state["data"], expert, split, int(config["max_test_invocations_per_expert"]))
                    if not len(records): continue
                    inp = np.asarray(state["data"]["x"][records], dtype=np.float32) if projection in ("gate", "up") else None
                    if projection == "down":
                        x = np.asarray(state["data"]["x"][records], dtype=np.float32)
                        g = x @ state["reference"]["gate"][ei].T; u = x @ state["reference"]["up"][ei].T
                        inp = (g / (1 + np.exp(-g))) * u
                    reference_output = inp @ reference.T
                    for base_name, matrix in (("serialized_fp16", state["base_fp16"][projection][ei]), ("historical_fp32", state["base_fp32"][projection][ei])):
                        error = reference_output - inp @ matrix.T
                        base_rows.append({"layer": layer, "expert_id": expert, "projection": projection, "split": split, "base_scale_storage": base_name, "mean_squared_functional_error": float(np.mean(error.astype(np.float64) ** 2))})
            save_parquet(args.output / "dense_tuple_metrics.parquet", metrics)
            save_parquet(args.output / "projection_metrics.parquet", projection_metrics)
            save_parquet(args.output / "stage_diagnostics.parquet", stage_metrics)
            save_parquet(args.output / "scale_controls.parquet", scale_rows)
            save_parquet(args.output / "base_scale_sensitivity.parquet", base_rows)
            pd.DataFrame(storage_rows).to_csv(args.output / "storage_accounting.csv", index=False)
            print(json.dumps({"completed_layer": layer, "expert": expert, "tuple_rows": len(metrics)}), flush=True)
            del methods; gc.collect(); torch.cuda.empty_cache()
    # Dense page-size accounting is bookkeeping: streams are contiguous.
    page_rows = []
    for layer, state in states.items():
        for projection in PROJECTIONS:
            shape = state["reference"][projection][0].shape
            for spec in hetero_paths[projection]:
                probe = build_rrq_series(np.zeros(shape, dtype=np.float32), np.ones(shape, dtype=np.float32), (spec,), None).stages[0]
                logical = probe.serialized_bytes() - 64
                for page in config["page_sizes"]:
                    physical = probe.physical_read_bytes(int(page))
                    page_rows.append({"layer": layer, "projection": projection, "quantizer": spec_label(spec), "page_size": int(page), "logical_bytes": logical, "physical_bytes": physical, "amplification": physical / logical, "stage_effective_physical_bpw": 8 * physical / MATRIX_WEIGHTS})
    pd.DataFrame(page_rows).to_csv(args.output / "dense_page_accounting.csv", index=False)
    facts["quantizer_selection"] = selected
    facts["centroid4_diagnostic_scores"] = centroid_scores
    facts["down_calibration"] = down_calibration
    facts["rows"] = {"tuples": len(metrics), "projection": len(projection_metrics), "stages": len(stage_metrics)}
    facts["wall_seconds"] = time.time() - facts["started_unix"]
    facts["peak_cuda_bytes"] = int(torch.cuda.max_memory_allocated())
    facts["resident_base"] = "Q2 codes/scales resident on Spark only; no external duplicate counted"
    facts["external_capacity"] = "production-reference fallback plus selected three-stage RRQ package; base excluded"
    atomic_json(args.output / "run_facts.json", facts)
    print(json.dumps({"dense_rrq_complete": True, "wall_seconds": facts["wall_seconds"], "rows": facts["rows"]}), flush=True)


if __name__ == "__main__":
    main()
