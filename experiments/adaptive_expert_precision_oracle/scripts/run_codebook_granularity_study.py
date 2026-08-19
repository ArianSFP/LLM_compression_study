#!/usr/bin/env python3
"""Bounded H0 study of 0.125-bpw local metadata for progressive MXFP4.

Stages are intentionally restartable and shardable.  ``fit`` writes immutable
codec artifacts, ``dense`` and ``selective`` may be split by layer across GPUs,
and ``merge`` combines only complete shard files.  No H4 predictor is trained.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
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
MATRIX_WEIGHTS = 512 * 2048
EXPERT_WEIGHTS = 3 * MATRIX_WEIGHTS
PAGE_SIZE = 512


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_parquet(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def silu(value: torch.Tensor) -> torch.Tensor:
    return value * torch.sigmoid(value)


def qenergy(error: torch.Tensor, proxy: torch.Tensor, beta: float) -> torch.Tensor:
    return torch.sum(error * error, dim=-1) + beta * torch.sum((error @ proxy) ** 2, dim=-1)


def occurrence(data: dict[str, np.ndarray], expert: int, split: str, maximum: int) -> tuple[np.ndarray, np.ndarray]:
    found = np.argwhere((data["expert_ids"] == expert) & (data["split"][:, None] == split))[:maximum]
    if not len(found):
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return found[:, 0], found[:, 1]


def select_stratified_experts(
    data: dict[str, np.ndarray], per_stratum: int, minimum: int,
) -> tuple[dict[int, str], dict[int, dict[str, int]]]:
    rows = []
    for expert in range(256):
        counts = {
            split: int(np.sum(data["expert_ids"][data["split"] == split] == expert))
            for split in ("train", "validation", "test")
        }
        if min(counts.values()) >= minimum:
            rows.append((sum(counts.values()), expert, counts))
    if len(rows) < 3 * per_stratum:
        raise RuntimeError(f"only {len(rows)} experts have complete split coverage")
    rows.sort()
    middle = len(rows) // 2
    chosen = {
        "cold": rows[:per_stratum],
        "median": rows[middle - per_stratum // 2:middle - per_stratum // 2 + per_stratum],
        "hot": rows[-per_stratum:],
    }
    strata: dict[int, str] = {}
    counts: dict[int, dict[str, int]] = {}
    for label, values in chosen.items():
        for _, expert, local in values:
            strata[int(expert)] = label
            counts[int(expert)] = local
    return strata, counts


def load_capture(path: Path, layers: Iterable[int]) -> dict[int, dict[str, np.ndarray]]:
    raw = np.load(path, allow_pickle=False)
    result: dict[int, dict[str, np.ndarray]] = {}
    for layer in map(int, layers):
        mask = raw["layer"] == layer
        result[layer] = {name: raw[name][mask] for name in raw.files}
    return result


def baseline_books(path: Path) -> dict[str, list[Any]]:
    from oracle_study.codebook_granularity import ProgressiveBook

    records = json.loads(path.read_text())
    result = {}
    for projection in PROJECTIONS:
        record = records[projection]
        result[projection] = [ProgressiveBook(
            np.asarray(record["leaf_to_parent"], np.uint8),
            np.asarray(record["leaf_to_r1"], np.uint8),
            np.asarray(record["leaf_to_r2"], np.uint8),
            np.asarray(record["c2"], np.float32),
            np.asarray(record["c3"], np.float32),
            "locked_pr4",
        )]
    return result


def prepare_states(
    config: dict[str, Any], captures: Path, checkpoint: Path, layers: Iterable[int],
) -> tuple[dict[int, dict[str, Any]], dict[str, str]]:
    from oracle_study.mxfp4_embed import audit_checkpoint, load_compressed_mxfp4_expert
    from run_phase_a_remote import proxy_gradients

    layer_list = list(map(int, layers))
    audit = audit_checkpoint(checkpoint, layer_list)
    if not audit["passed"]:
        raise RuntimeError(audit)
    config_hash = sha256(checkpoint / "config.json")
    index_hash = sha256(checkpoint / "model.safetensors.index.json")
    expected = config["reference"]
    if config_hash != expected["config_sha256"] or index_hash != expected["index_sha256"]:
        raise RuntimeError("authoritative checkpoint hashes changed")
    capture_hash = sha256(captures)
    expected_capture = config.get("primary_capture", {}).get("sha256")
    if expected_capture and capture_hash != expected_capture:
        raise RuntimeError(f"capture hash changed: {capture_hash} != {expected_capture}")
    weight_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    data_by_layer = load_capture(captures, layer_list)
    states: dict[int, dict[str, Any]] = {}
    for layer in layer_list:
        data = data_by_layer[layer]
        strata, counts = select_stratified_experts(
            data, int(config["experts_per_stratum"]), int(config["minimum_split_occurrences"]),
        )
        proxy, proxy_facts = proxy_gradients(
            data["xplus"], [data[f"h{h}_router_logits"] for h in range(1, 5)], data["split"],
        )
        tensors = {
            expert: {
                projection: load_compressed_mxfp4_expert(checkpoint, weight_map, layer, expert, projection)
                for projection in PROJECTIONS
            }
            for expert in sorted(strata)
        }
        states[layer] = {
            "layer": layer, "data": data, "strata": strata, "counts": counts,
            "experts": sorted(strata), "proxy": proxy, "beta": float(proxy_facts["beta"]),
            "proxy_facts": proxy_facts, "tensors": tensors,
        }
    return states, {"config_sha256": config_hash, "index_sha256": index_hash}


def projection_importance(
    state: dict[str, Any], expert: int, objective: str, maximum: int, fit_split: str = "train",
) -> dict[str, np.ndarray]:
    """Diagonal weight, projection, or complete-expert functional objective."""
    tensors = state["tensors"][expert]
    if objective == "weight_mse":
        return {projection: np.ones(tensor.shape, np.float32) for projection, tensor in tensors.items()}
    records, _ = occurrence(state["data"], expert, fit_split, maximum)
    if not len(records):
        raise RuntimeError(f"no {fit_split} invocation for layer {state['layer']} expert {expert}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.from_numpy(np.asarray(state["data"]["x"][records], np.float32)).to(device)
    matrices = {p: torch.from_numpy(tensors[p].dequantize()).to(device) for p in PROJECTIONS}
    with torch.no_grad():
        gate = x @ matrices["gate"].T
        up = x @ matrices["up"].T
        hidden = silu(gate) * up
        x2 = torch.mean(x * x, dim=0)
        h2 = torch.mean(hidden * hidden, dim=0)
        if objective == "activation_projection":
            result = {
                "gate": x2[None, :].expand(512, 2048).cpu().numpy().copy(),
                "up": x2[None, :].expand(512, 2048).cpu().numpy().copy(),
                "down": h2[None, :].expand(2048, 512).cpu().numpy().copy(),
            }
        elif objective == "expert_functional_proxy":
            proxy = torch.from_numpy(np.asarray(state["proxy"], np.float32)).to(device)
            beta = float(state["beta"])
            down = matrices["down"]
            down_metric_norm = torch.sum(down * down, dim=0) + beta * torch.sum((down.T @ proxy) ** 2, dim=1)
            sigmoid = torch.sigmoid(gate)
            silu_prime = sigmoid + gate * sigmoid * (1.0 - sigmoid)
            gate_factor = down_metric_norm[None, :] * (up * silu_prime) ** 2
            up_factor = down_metric_norm[None, :] * silu(gate) ** 2
            gate_importance = gate_factor.T @ (x * x) / len(x)
            up_importance = up_factor.T @ (x * x) / len(x)
            metric_diagonal = 1.0 + beta * torch.sum(proxy * proxy, dim=1)
            down_importance = metric_diagonal[:, None] * h2[None, :]
            result = {
                "gate": gate_importance.cpu().numpy(), "up": up_importance.cpu().numpy(),
                "down": down_importance.cpu().numpy(),
            }
        else:
            raise ValueError(objective)
    del x, matrices, gate, up, hidden
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {name: np.asarray(value, np.float32) for name, value in result.items()}


def record_key(layer: int, expert: int, projection: str) -> str:
    return f"l{layer:02d}_e{expert:03d}_{projection}"


def safe_key(value: str) -> str:
    return value.replace("/", "__").replace(":", "_").replace("-", "_")


def save_candidate(directory: Path, candidate: dict[str, Any]) -> None:
    """Save research state separately from the charged deployment payloads."""
    from oracle_study.codebook_granularity import serialize_books, serialize_metadata_file

    directory.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    manifest = {key: value for key, value in candidate.items() if key not in ("books", "selectors", "modifiers")}
    manifest["book_scopes"] = {}
    table_facts = []
    for scope, books in candidate["books"].items():
        key = safe_key(scope)
        arrays[f"book_{key}_parent"] = np.stack([book.parent for book in books]).astype(np.uint8)
        arrays[f"book_{key}_r1"] = np.stack([book.r1 for book in books]).astype(np.uint8)
        arrays[f"book_{key}_r2"] = np.stack([book.r2 for book in books]).astype(np.uint8)
        arrays[f"book_{key}_c2"] = np.stack([book.c2 for book in books]).astype(np.float32)
        arrays[f"book_{key}_c3"] = np.stack([book.c3 for book in books]).astype(np.float32)
        if hasattr(books[0], "c4"):
            arrays[f"book_{key}_c4"] = np.stack([book.c4 for book in books]).astype(np.float32)
        manifest["book_scopes"][scope] = key
        table_facts.append({"scope": scope, **serialize_books(directory / "serialized_tables" / f"{key}.bin", books)})
    for key, value in candidate["selectors"].items():
        arrays[f"selector_{safe_key(key)}"] = np.asarray(value, np.uint16)
    for key, value in candidate.get("modifiers", {}).items():
        arrays[f"modifier_{safe_key(key)}"] = np.asarray(value, np.float32)
    np.savez_compressed(directory / "research_state.npz", **arrays)
    manifest["table_files"] = table_facts
    payload_facts = []
    for layer, expert in candidate["experts"]:
        selectors = {}
        modifiers = {}
        for projection in PROJECTIONS:
            key = record_key(layer, expert, projection)
            selectors[projection] = candidate["selectors"].get(key, np.empty(0, np.uint16))
            if key in candidate.get("modifiers", {}):
                modifiers[projection] = candidate["modifiers"][key]
        if int(candidate["selector_bits"]) or modifiers:
            fact = serialize_metadata_file(
                directory / "serialized_metadata" / f"layer_{layer:02d}_expert_{expert:03d}.cgb",
                int(candidate["design_id"]), int(candidate["selector_bits"]), selectors,
                modifiers if modifiers else None,
            )
            payload_facts.append({"layer": layer, "expert": expert, **fact})
    manifest["metadata_files"] = payload_facts
    atomic_json(directory / "manifest.json", manifest)


def load_candidate(directory: Path) -> dict[str, Any]:
    from oracle_study.codebook_granularity import LearnedScalarBook, ProgressiveBook

    manifest = json.loads((directory / "manifest.json").read_text())
    state = np.load(directory / "research_state.npz", allow_pickle=False)
    books = {}
    for scope, key in manifest["book_scopes"].items():
        parents = state[f"book_{key}_parent"]
        r1 = state[f"book_{key}_r1"]
        r2 = state[f"book_{key}_r2"]
        c2 = state[f"book_{key}_c2"]
        c3 = state[f"book_{key}_c3"]
        c4_key = f"book_{key}_c4"
        if c4_key in state.files:
            c4 = state[c4_key]
            books[scope] = [LearnedScalarBook(parents[i], r1[i], r2[i], c2[i], c3[i], c4[i], f"{scope}:{i}") for i in range(len(parents))]
        else:
            books[scope] = [ProgressiveBook(parents[i], r1[i], r2[i], c2[i], c3[i], f"{scope}:{i}") for i in range(len(parents))]
    selectors = {}
    modifiers = {}
    for layer, expert in manifest["experts"]:
        for projection in PROJECTIONS:
            key = record_key(int(layer), int(expert), projection)
            array_key = f"selector_{safe_key(key)}"
            if array_key in state.files:
                selectors[key] = state[array_key]
            modifier_key = f"modifier_{safe_key(key)}"
            if modifier_key in state.files:
                modifiers[key] = state[modifier_key]
    return {**manifest, "books": books, "selectors": selectors, "modifiers": modifiers}


def candidate_scope(candidate: dict[str, Any], layer: int, expert: int, projection: str, stratum: str) -> str:
    sharing = candidate["sharing"]
    if sharing == "baseline_global":
        return f"global/{projection}"
    if sharing == "global_projection":
        return f"global/{projection}"
    if sharing == "layer_projection":
        return f"layer_{layer:02d}/{projection}"
    if sharing == "cluster_projection":
        return f"cluster_{stratum}/{projection}"
    if sharing == "expert_projection" or sharing == "independent_vector":
        return f"layer_{layer:02d}/expert_{expert:03d}/{projection}"
    raise ValueError(sharing)


def decoded_matrices(
    candidate: dict[str, Any], state: dict[str, Any], expert: int,
) -> dict[str, list[np.ndarray]]:
    from oracle_study.codebook_granularity import decode_with_books

    matrices = {}
    for projection in PROJECTIONS:
        key = record_key(state["layer"], expert, projection)
        scope = candidate_scope(candidate, state["layer"], expert, projection, state["strata"][expert])
        selectors = candidate["selectors"].get(key)
        matrices[projection] = [
            decode_with_books(
                state["tensors"][expert][projection], candidate["books"][scope], selectors,
                int(candidate["group_size"]), projection, level, candidate.get("modifiers", {}).get(key),
            )
            for level in (2, 3, 4)
        ]
    return matrices


def stable_seed(base: int, label: str) -> int:
    return int(base) + int(hashlib.sha256(label.encode()).hexdigest()[:8], 16) % 1_000_000


def build_candidate(
    states: dict[int, dict[str, Any]], config: dict[str, Any], *, name: str,
    design: str, design_id: int, objective: str, sharing: str, families: int,
    group_size: int, selector_bits: int, epsilon: float, shrinkage: float = 0.0,
    importance_cache: dict[tuple[int, int, str, str], dict[str, np.ndarray]],
    fit_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    from oracle_study.codebook_granularity import (
        block_leaf_masses, fit_family_books, fit_independent_books,
        fit_vector_modifiers, vector_leaf_masses,
    )

    seed = int(config["seed"])
    experts = [(layer, expert) for layer, state in states.items() for expert in state["experts"]]
    candidate: dict[str, Any] = {
        "name": name, "design": design, "design_id": design_id, "objective": objective,
        "sharing": sharing, "families": families, "group_size": group_size,
        "selector_bits": selector_bits, "q2_tolerance": epsilon, "exact_mxfp4": True,
        "vector_shrinkage_strength": shrinkage,
        "experts": experts, "books": {}, "selectors": {}, "modifiers": {},
    }
    pieces: dict[str, list[tuple[str, np.ndarray, np.ndarray]]] = defaultdict(list)
    independent_context: dict[str, tuple[int, int, str, int]] = {}
    independent_priors: dict[tuple[int, str], list[np.ndarray]] = defaultdict(list)
    for layer, state in states.items():
        for expert in state["experts"]:
            importance = importance_cache[(layer, expert, objective, "train")]
            validation_importance = importance_cache[(layer, expert, objective, "validation")]
            for projection in PROJECTIONS:
                key = record_key(layer, expert, projection)
                scope = candidate_scope(candidate, layer, expert, projection, state["strata"][expert])
                tensor = state["tensors"][expert][projection]
                if sharing == "independent_vector":
                    masses = vector_leaf_masses(tensor, importance[projection], projection)
                    validation_masses = vector_leaf_masses(tensor, validation_importance[projection], projection)
                    train_records, _ = occurrence(
                        state["data"], expert, "train", int(config["max_train_invocations_per_expert"]),
                    )
                    independent_context[scope] = (layer, expert, projection, len(train_records))
                    independent_priors[(layer, projection)].append(masses.sum(axis=0))
                else:
                    masses = block_leaf_masses(tensor, importance[projection], group_size)
                    validation_masses = block_leaf_masses(tensor, validation_importance[projection], group_size)
                pieces[scope].append((key, masses, validation_masses))
    for scope, local_pieces in pieces.items():
        if sharing == "independent_vector":
            if len(local_pieces) != 1:
                raise AssertionError(scope)
            _, masses, validation_masses = local_pieces[0]
            layer, expert, projection, train_invocations = independent_context[scope]
            alpha = 0.0
            fit_masses = masses
            if shrinkage > 0:
                prior = np.sum(independent_priors[(layer, projection)], axis=0)
                prior = prior / max(float(prior.sum()), 1e-30)
                alpha = float(shrinkage / (train_invocations + shrinkage))
                fit_masses = (1.0 - alpha) * masses + alpha * masses.sum(axis=1, keepdims=True) * prior[None, :]
            books = fit_independent_books(
                fit_masses, epsilon, stable_seed(seed, f"{name}:{scope}"),
                int(config.get("independent_order_candidates", 48)),
            )
            candidate["books"][scope] = books
            q2 = sum(float(np.dot(masses[i], book.distortion_vector(2))) for i, book in enumerate(books))
            q3 = sum(float(np.dot(masses[i], book.distortion_vector(3))) for i, book in enumerate(books))
            validation_q2 = sum(float(np.dot(validation_masses[i], book.distortion_vector(2))) for i, book in enumerate(books))
            validation_q3 = sum(float(np.dot(validation_masses[i], book.distortion_vector(3))) for i, book in enumerate(books))
            fit_rows.append({
                "candidate": name, "scope": scope, "iteration": 0, "q2_distortion": q2,
                "q3_distortion": q3, "validation_q2_distortion": validation_q2,
                "validation_q3_distortion": validation_q3, "changed_fraction": 0.0,
                "entropy_nats": math.log(max(len(books), 1)), "effective_families": len(books),
                "train_invocations": train_invocations, "shrinkage_alpha": alpha,
            })
            continue
        combined = np.concatenate([value for _, value, _ in local_pieces], axis=0)
        combined_validation = np.concatenate([value for _, _, value in local_pieces], axis=0)
        books, assignments, history = fit_family_books(
            combined, families, epsilon, stable_seed(seed, f"{name}:{scope}"),
            validation_masses=combined_validation, order_count=int(config.get("family_order_candidates", 96)),
            max_iterations=int(config.get("alternating_iterations", 8)),
        )
        candidate["books"][scope] = books
        offset = 0
        for key, masses, _ in local_pieces:
            candidate["selectors"][key] = assignments[offset:offset + len(masses)].copy()
            offset += len(masses)
        for row in history:
            fit_rows.append({"candidate": name, "scope": scope, **row})
    if design == "C":
        for layer, state in states.items():
            for expert in state["experts"]:
                importance = importance_cache[(layer, expert, objective, "train")]
                for projection in PROJECTIONS:
                    key = record_key(layer, expert, projection)
                    scope = candidate_scope(candidate, layer, expert, projection, state["strata"][expert])
                    candidate["modifiers"][key] = fit_vector_modifiers(
                        state["tensors"][expert][projection], candidate["books"][scope],
                        candidate["selectors"][key], group_size, projection, importance[projection],
                    )
    return candidate


def build_baseline(states: dict[int, dict[str, Any]], trees: Path) -> dict[str, Any]:
    books = baseline_books(trees)
    candidate: dict[str, Any] = {
        "name": "baseline_pr4_2p25", "design": "baseline", "design_id": 0,
        "objective": "activation_projection", "sharing": "baseline_global", "families": 1,
        "group_size": 32, "selector_bits": 0, "q2_tolerance": 0.001,
        "exact_mxfp4": True, "experts": [], "books": {}, "selectors": {}, "modifiers": {},
    }
    for projection in PROJECTIONS:
        candidate["books"][f"global/{projection}"] = books[projection]
    for layer, state in states.items():
        for expert in state["experts"]:
            candidate["experts"].append((layer, expert))
            for projection in PROJECTIONS:
                tensor = state["tensors"][expert][projection]
                candidate["selectors"][record_key(layer, expert, projection)] = np.zeros(
                    tensor.shape[0] * (tensor.shape[1] // 32), np.uint16,
                )
    return candidate


def build_learned_leaf_diagnostic(
    states: dict[int, dict[str, Any]],
    importance_cache: dict[tuple[int, int, str, str], dict[str, np.ndarray]],
) -> dict[str, Any]:
    from oracle_study.codebook_granularity import block_leaf_masses, fit_learned_scalar_book

    candidate: dict[str, Any] = {
        "name": "learned_16_leaf_scalar_global", "design": "learned_16_leaf", "design_id": 6,
        "objective": "expert_functional_proxy", "sharing": "global_projection", "families": 1,
        "group_size": 32, "selector_bits": 0, "q2_tolerance": 0.0,
        "exact_mxfp4": False, "experts": [], "books": {}, "selectors": {}, "modifiers": {},
    }
    for projection in PROJECTIONS:
        masses = []
        for layer, state in states.items():
            for expert in state["experts"]:
                masses.append(block_leaf_masses(
                    state["tensors"][expert][projection],
                    importance_cache[(layer, expert, "expert_functional_proxy", "train")][projection], 32,
                ).sum(axis=0))
        candidate["books"][f"global/{projection}"] = [fit_learned_scalar_book(np.sum(masses, axis=0), projection)]
    for layer, state in states.items():
        for expert in state["experts"]:
            candidate["experts"].append((layer, expert))
            for projection in PROJECTIONS:
                tensor = state["tensors"][expert][projection]
                candidate["selectors"][record_key(layer, expert, projection)] = np.zeros(
                    tensor.shape[0] * (tensor.shape[1] // 32), np.uint16,
                )
    return candidate


def fit_stage(args: argparse.Namespace, config: dict[str, Any]) -> None:
    layers = list(map(int, config["layers"]))
    states, checkpoint_hashes = prepare_states(config, args.captures, args.checkpoint, layers)
    output = args.output
    codec_root = output / "serialized_manifests" / "codecs"
    codec_root.mkdir(parents=True, exist_ok=True)
    fit_path = output / "metrics" / "fit_history.parquet"
    fit_rows: list[dict[str, Any]] = (
        pd.read_parquet(fit_path).to_dict("records") if fit_path.is_file() else []
    )
    importance_cache: dict[tuple[int, int, str, str], dict[str, np.ndarray]] = {}
    objectives = ("weight_mse", "activation_projection", "expert_functional_proxy")
    for layer, state in states.items():
        for expert in state["experts"]:
            for objective in objectives:
                for fit_split, maximum in (("train", int(config["max_train_invocations_per_expert"])), ("validation", int(config["max_validation_invocations_per_expert"]))):
                    importance_cache[(layer, expert, objective, fit_split)] = projection_importance(
                        state, expert, objective, maximum, fit_split,
                    )
            print(json.dumps({"importance_ready": [layer, expert]}), flush=True)

    specifications: list[dict[str, Any]] = [
        dict(name="A_vector_functional", design="A", design_id=1, objective="expert_functional_proxy", sharing="independent_vector", families=512, group_size=32, selector_bits=0, epsilon=0.005),
        dict(name="A_vector_functional_shrunk", design="A", design_id=1, objective="expert_functional_proxy", sharing="independent_vector", families=512, group_size=32, selector_bits=0, epsilon=0.005, shrinkage=float(config.get("vector_shrinkage_strength", 8.0))),
        dict(name="B16_g32_global_weight", design="B", design_id=2, objective="weight_mse", sharing="global_projection", families=16, group_size=32, selector_bits=4, epsilon=0.005),
        dict(name="B16_g32_global_activation", design="B", design_id=2, objective="activation_projection", sharing="global_projection", families=16, group_size=32, selector_bits=4, epsilon=0.005),
        dict(name="B16_g32_global_functional", design="B", design_id=2, objective="expert_functional_proxy", sharing="global_projection", families=16, group_size=32, selector_bits=4, epsilon=0.005),
        dict(name="B16_g32_layer_functional", design="B", design_id=2, objective="expert_functional_proxy", sharing="layer_projection", families=16, group_size=32, selector_bits=4, epsilon=0.005),
        dict(name="B16_g32_cluster_functional", design="B", design_id=2, objective="expert_functional_proxy", sharing="cluster_projection", families=16, group_size=32, selector_bits=4, epsilon=0.005),
        dict(name="B16_g32_expert_functional", design="B", design_id=2, objective="expert_functional_proxy", sharing="expert_projection", families=16, group_size=32, selector_bits=4, epsilon=0.005),
        dict(name="C8_g32_layer_functional", design="C", design_id=3, objective="expert_functional_proxy", sharing="layer_projection", families=8, group_size=32, selector_bits=3, epsilon=0.005),
        dict(name="D4_g16_layer_functional", design="D", design_id=4, objective="expert_functional_proxy", sharing="layer_projection", families=4, group_size=16, selector_bits=2, epsilon=0.005),
        dict(name="E16_g64_layer_functional", design="E", design_id=5, objective="expert_functional_proxy", sharing="layer_projection", families=16, group_size=64, selector_bits=8, epsilon=0.005),
        dict(name="E32_g64_layer_functional", design="E", design_id=5, objective="expert_functional_proxy", sharing="layer_projection", families=32, group_size=64, selector_bits=8, epsilon=0.005),
        dict(name="E64_g64_layer_functional", design="E", design_id=5, objective="expert_functional_proxy", sharing="layer_projection", families=64, group_size=64, selector_bits=8, epsilon=0.005),
    ]
    for epsilon in config["q2_tolerance_sweep"]:
        if abs(float(epsilon) - 0.005) < 1e-12:
            continue
        suffix = str(float(epsilon)).replace(".", "p")
        specifications.append(dict(
            name=f"B16_g32_global_functional_eps_{suffix}", design="B", design_id=2,
            objective="expert_functional_proxy", sharing="global_projection", families=16,
            group_size=32, selector_bits=4, epsilon=float(epsilon),
        ))

    requested = set(args.candidates.split(",")) if args.candidates else None
    if requested is not None:
        known = {"baseline_pr4_2p25", "learned_16_leaf_scalar_global"} | {
            value["name"] for value in specifications
        }
        unknown = requested - known
        if unknown:
            raise ValueError(f"unknown fit candidates: {sorted(unknown)}")
        specifications = [value for value in specifications if value["name"] in requested]

    index = []
    if requested is None or "baseline_pr4_2p25" in requested:
        baseline = build_baseline(states, args.trees)
        if not (codec_root / baseline["name"] / "manifest.json").is_file():
            save_candidate(codec_root / baseline["name"], baseline)
        index.append(baseline["name"])
    if requested is None or "learned_16_leaf_scalar_global" in requested:
        learned = build_learned_leaf_diagnostic(states, importance_cache)
        if not (codec_root / learned["name"] / "manifest.json").is_file():
            save_candidate(codec_root / learned["name"], learned)
        index.append(learned["name"])
    for specification in specifications:
        started = time.time()
        candidate_path = codec_root / specification["name"]
        if candidate_path.joinpath("manifest.json").is_file():
            index.append(specification["name"])
            print(json.dumps({"candidate": specification["name"], "resumed": True}), flush=True)
            continue
        candidate = build_candidate(states, config, importance_cache=importance_cache, fit_rows=fit_rows, **specification)
        save_candidate(candidate_path, candidate)
        index.append(candidate["name"])
        atomic_parquet(output / "metrics" / "fit_history.parquet", fit_rows)
        atomic_json(output / "serialized_manifests" / "candidate_index.json", index)
        print(json.dumps({"candidate": candidate["name"], "fit_seconds": time.time() - started}), flush=True)
        del candidate
        gc.collect()
    first_state = next(iter(states.values()))
    facts = {
        "stage": "fit", "completed_unix": time.time(), "host": platform.node(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoint_hashes": checkpoint_hashes, "capture_sha256": sha256(args.captures),
        "capture_requests": {split: len(np.unique(first_state["data"]["request_id"][first_state["data"]["split"] == split])) for split in ("train", "validation", "test")},
        "layers": {str(layer): {"experts": state["experts"], "strata": state["strata"], "counts": state["counts"], "proxy": state["proxy_facts"]} for layer, state in states.items()},
        "candidates": index,
    }
    atomic_json(output / "serialized_manifests" / "candidate_index.json", index)
    atomic_json(output / "serialized_manifests" / "fit_run_facts.json", facts)


def endpoint_code_exact(candidate: dict[str, Any], state: dict[str, Any], expert: int) -> bool:
    from oracle_study.codebook_granularity import book_code_arrays

    for projection in PROJECTIONS:
        tensor = state["tensors"][expert][projection]
        key = record_key(state["layer"], expert, projection)
        scope = candidate_scope(candidate, state["layer"], expert, projection, state["strata"][expert])
        parent, r1, r2 = book_code_arrays(
            tensor.codes, candidate["books"][scope], candidate["selectors"].get(key),
            int(candidate["group_size"]), projection,
        )
        books = candidate["books"][scope]
        if candidate["selectors"].get(key) is None:
            ids = np.arange(len(books), dtype=np.int64)
            grid = np.broadcast_to(ids[:, None], tensor.codes.shape) if projection in ("gate", "up") else np.broadcast_to(ids[None, :], tensor.codes.shape)
        else:
            selectors = np.asarray(candidate["selectors"][key], np.int64).reshape(
                tensor.shape[0], tensor.shape[1] // int(candidate["group_size"]),
            )
            grid = np.repeat(selectors, int(candidate["group_size"]), axis=1)
        inverse = np.empty((len(books), 4, 2, 2), np.uint8)
        for book_id, book in enumerate(books):
            for leaf in range(16):
                inverse[book_id, book.parent[leaf], book.r1[leaf], book.r2[leaf]] = leaf
        if not np.array_equal(inverse[grid, parent, r1, r2], tensor.codes):
            return False
    return True


def dense_expert_rows(
    candidate: dict[str, Any], state: dict[str, Any], expert: int, config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    matrices_np = decoded_matrices(candidate, state, expert)
    if candidate["exact_mxfp4"] and not endpoint_code_exact(candidate, state, expert):
        raise AssertionError(f"non-exact endpoint for {candidate['name']} layer {state['layer']} expert {expert}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    matrices = {p: [torch.from_numpy(value).to(device) for value in levels] for p, levels in matrices_np.items()}
    reference_matrices = {p: torch.from_numpy(state["tensors"][expert][p].dequantize()).to(device) for p in PROJECTIONS}
    proxy = torch.from_numpy(np.asarray(state["proxy"], np.float32)).to(device)
    dense_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    for split, maximum in (
        ("train", int(config["max_train_invocations_per_expert"])),
        ("validation", int(config["max_validation_invocations_per_expert"])),
        ("test", int(config["max_test_invocations_per_expert"])),
    ):
        records, ranks = occurrence(state["data"], expert, split, maximum)
        if not len(records):
            continue
        x = torch.from_numpy(np.asarray(state["data"]["x"][records], np.float32)).to(device)
        gate = [x @ value.T for value in matrices["gate"]]
        up = [x @ value.T for value in matrices["up"]]
        reference_gate = x @ reference_matrices["gate"].T
        reference_up = x @ reference_matrices["up"].T
        href = silu(reference_gate) * reference_up
        yref = href @ reference_matrices["down"].T
        ylevels = [silu(gate[level]) * up[level] @ matrices["down"][level].T for level in range(3)]
        damages = [qenergy(yref - value, proxy, float(state["beta"])) for value in ylevels]
        reference_energy = qenergy(yref, proxy, float(state["beta"]))
        for projection in PROJECTIONS:
            inputs = x if projection != "down" else href
            target = inputs @ reference_matrices[projection].T
            local = []
            for level in range(3):
                error = target - inputs @ matrices[projection][level].T
                damage = torch.sum(error * error, dim=1) if projection != "down" else qenergy(error, proxy, float(state["beta"]))
                local.append(damage)
            denominator = torch.clamp(local[0], min=1e-30)
            target_norm = torch.clamp(torch.linalg.vector_norm(target, dim=1), min=1e-30)
            for level in range(3):
                error = target - inputs @ matrices[projection][level].T
                for sample, record in enumerate(records):
                    projection_rows.append({
                        "candidate": candidate["name"], "design": candidate["design"],
                        "objective": candidate["objective"], "sharing": candidate["sharing"],
                        "families": candidate["families"], "group_size": candidate["group_size"],
                        "q2_tolerance": candidate["q2_tolerance"], "request_id": str(state["data"]["request_id"][record]),
                        "position": int(state["data"]["position"][record]), "split": split,
                        "layer": state["layer"], "expert_id": expert, "expert_stratum": state["strata"][expert],
                        "projection": projection, "level": level + 2, "damage": float(local[level][sample]),
                        "recovery": float(1.0 - local[level][sample] / denominator[sample]),
                        "relative_error": float(torch.linalg.vector_norm(error[sample]) / target_norm[sample]),
                        "endpoint_exact": bool(level == 2 and candidate["exact_mxfp4"]),
                    })
        base = torch.clamp(damages[0], min=1e-30)
        for level in range(3):
            error = yref - ylevels[level]
            for sample, (record, rank) in enumerate(zip(records, ranks)):
                dense_rows.append({
                    "candidate": candidate["name"], "design": candidate["design"],
                    "objective": candidate["objective"], "sharing": candidate["sharing"],
                    "families": candidate["families"], "group_size": candidate["group_size"],
                    "selector_bits": candidate["selector_bits"], "q2_tolerance": candidate["q2_tolerance"],
                    "request_id": str(state["data"]["request_id"][record]),
                    "sequence_id": str(state["data"]["sequence_id"][record]),
                    "position": int(state["data"]["position"][record]), "split": split,
                    "layer": state["layer"], "expert_id": expert, "expert_stratum": state["strata"][expert],
                    "router_rank": int(rank) + 1, "router_coefficient": float(state["data"]["router_weights"][record, rank]),
                    "activation_norm": float(torch.linalg.vector_norm(x[sample])), "level": level + 2,
                    "damage": float(damages[level][sample]), "base_damage": float(damages[0][sample]),
                    "reference_energy": float(reference_energy[sample]),
                    "recovery": float(1.0 - damages[level][sample] / base[sample]),
                    "relative_output_error": float(torch.linalg.vector_norm(error[sample]) / torch.clamp(torch.linalg.vector_norm(yref[sample]), min=1e-30)),
                    "endpoint_exact": bool(level == 2 and candidate["exact_mxfp4"]),
                })
        del x, gate, up, href, yref, ylevels, damages
    del matrices, reference_matrices, proxy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dense_rows, projection_rows


def dense_stage(args: argparse.Namespace, config: dict[str, Any]) -> None:
    layers = list(map(int, args.layers or config["layers"]))
    states, _ = prepare_states(config, args.captures, args.checkpoint, layers)
    codec_root = args.output / "serialized_manifests" / "codecs"
    names = json.loads((args.output / "serialized_manifests" / "candidate_index.json").read_text())
    if args.candidates:
        selected = set(args.candidates.split(","))
        names = [name for name in names if name in selected]
    rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    for name in names:
        candidate = load_candidate(codec_root / name)
        for layer in layers:
            state = states[layer]
            for expert in state["experts"]:
                local, local_projection = dense_expert_rows(candidate, state, expert, config)
                rows.extend(local)
                projection_rows.extend(local_projection)
                print(json.dumps({"dense": name, "layer": layer, "expert": expert, "rows": len(local)}), flush=True)
        del candidate
        gc.collect()
    suffix = "_".join(map(str, layers))
    atomic_parquet(args.output / "metrics" / "dense_shards" / f"dense_layers_{suffix}.parquet", rows)
    atomic_parquet(args.output / "metrics" / "dense_shards" / f"projection_layers_{suffix}.parquet", projection_rows)


def unconstrained_greedy_from_gram(gram: np.ndarray, activation: np.ndarray) -> list[int]:
    values = np.asarray(gram, np.float32)
    coefficient = np.asarray(activation, np.float32)
    correlation = values @ coefficient
    diagonal = np.diag(values)
    available = np.ones(len(coefficient), bool)
    result = []
    for _ in range(len(coefficient)):
        score = 2.0 * coefficient * correlation - coefficient * coefficient * diagonal
        score[~available] = -np.inf
        chosen = int(np.argmax(score))
        if not np.isfinite(score[chosen]):
            break
        result.append(chosen)
        available[chosen] = False
        correlation -= float(coefficient[chosen]) * values[:, chosen]
    return result


def close_paid_pages(
    paid: set[int], coordinates: int, actions_per_page: int, paired: bool,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    depth = np.zeros(coordinates, np.uint8)
    applied: list[tuple[int, int]] = []
    pages_per_plane = math.ceil(coordinates / actions_per_page)
    changed = True
    while changed:
        changed = False
        if paired:
            for page in sorted(paid):
                start = page * actions_per_page
                for coordinate in range(start, min(start + actions_per_page, coordinates)):
                    if depth[coordinate] == 0:
                        depth[coordinate] = 2
                        applied.extend(((coordinate, 1), (coordinate, 2)))
                        changed = True
        else:
            for stage in (1, 2):
                for page in sorted(value for value in paid if value // pages_per_plane == stage - 1):
                    local_page = page % pages_per_plane
                    start = local_page * actions_per_page
                    for coordinate in range(start, min(start + actions_per_page, coordinates)):
                        if depth[coordinate] == stage - 1:
                            depth[coordinate] = stage
                            applied.append((coordinate, stage))
                            changed = True
    return depth, applied


def page_snapshot(
    order: list[tuple[int, int]], coordinates: int, payload_bytes: int,
    rate: float, paired: bool,
) -> dict[str, Any]:
    effective_payload = payload_bytes * (2 if paired else 1)
    actions_per_page = max(1, PAGE_SIZE // effective_payload)
    pages_per_plane = math.ceil(coordinates / actions_per_page)
    byte_budget = int(math.floor(rate * MATRIX_WEIGHTS / 8 + 1e-9))
    maximum_pages = byte_budget // PAGE_SIZE
    paid: set[int] = set()
    for coordinate, stage in order:
        page = coordinate // actions_per_page if paired else (stage - 1) * pages_per_plane + coordinate // actions_per_page
        if page in paid:
            continue
        if len(paid) >= maximum_pages:
            break
        paid.add(page)
    depth, applied = close_paid_pages(paid, coordinates, actions_per_page, paired)
    logical_bytes = len(applied) * payload_bytes
    return {
        "depth": depth, "pages": np.asarray(sorted(paid), np.uint16),
        "physical_bytes": len(paid) * PAGE_SIZE, "logical_bytes": logical_bytes,
        "q2_q3_upgrades": int(np.sum(depth >= 1)), "q3_q4_upgrades": int(np.sum(depth >= 2)),
    }


def projection_orders(
    deltas: list[np.ndarray], activation: np.ndarray, projection: str,
    proxy: np.ndarray, beta: float,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    from oracle_study.interaction_allocator import (
        build_base_gram, diagonal_from_gram, exact_marginal_fixed_greedy_from_gram,
    )

    action_deltas = np.stack([delta.T for delta in deltas]).astype(np.float32)
    metric = None
    if projection == "down":
        metric = np.eye(action_deltas.shape[-1], dtype=np.float32) + float(beta) * np.asarray(proxy, np.float32) @ np.asarray(proxy, np.float32).T
    gram = build_base_gram(action_deltas, metric, device="cuda" if torch.cuda.is_available() else "cpu")
    if projection in ("gate", "up"):
        separate_actions = exact_marginal_fixed_greedy_from_gram(gram, activation)
    else:
        separate_actions = diagonal_from_gram(gram, activation)
    separate = [(int(value.coordinate), int(value.stage)) for value in separate_actions]
    merged = action_deltas[0] + action_deltas[1]
    if metric is None:
        paired_gram = merged @ merged.T
    else:
        paired_gram = (merged @ metric) @ merged.T
    if projection in ("gate", "up"):
        paired_ids = unconstrained_greedy_from_gram(paired_gram, activation)
    else:
        paired_ids = list(np.argsort(-(np.asarray(activation, np.float64) ** 2 * np.diag(paired_gram)), kind="stable"))
    paired = [(int(value), 1) for value in paired_ids]
    return separate, paired


def selective_invocation(
    candidate: dict[str, Any], state: dict[str, Any], expert: int, record: int, rank: int,
    matrices: dict[str, list[np.ndarray]], orders: dict[str, dict[str, list[tuple[int, int]]]],
    rates: list[float], budgets: list[float], packet: str, split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paired = packet == "paired"
    data = state["data"]
    x = np.asarray(data["x"][record], np.float32)
    gref = matrices["gate"][2] @ x
    uref = matrices["up"][2] @ x
    href = (gref / (1.0 + np.exp(-np.clip(gref, -80, 80)))) * uref
    activations = {"gate": x, "up": x, "down": href}
    snapshots: dict[str, list[dict[str, Any]]] = {}
    deltas = {p: [matrices[p][1] - matrices[p][0], matrices[p][2] - matrices[p][1]] for p in PROJECTIONS}
    for projection in PROJECTIONS:
        payload = matrices[projection][0].shape[0] // 8
        snapshots[projection] = []
        for rate in rates:
            snapshot = page_snapshot(
                orders[projection][packet], matrices[projection][0].shape[1], payload, rate, paired,
            )
            correction = np.zeros(matrices[projection][0].shape[0], np.float32)
            for coordinate, depth in enumerate(snapshot["depth"]):
                if depth >= 1:
                    correction += deltas[projection][0][:, coordinate] * activations[projection][coordinate]
                if depth >= 2:
                    correction += deltas[projection][1][:, coordinate] * activations[projection][coordinate]
            snapshot["correction"] = correction
            snapshots[projection].append(snapshot)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gbase = matrices["gate"][0] @ x
    ubase = matrices["up"][0] @ x
    gate_stack = torch.from_numpy(np.stack([gbase + value["correction"] for value in snapshots["gate"]])).to(device)
    up_stack = torch.from_numpy(np.stack([ubase + value["correction"] for value in snapshots["up"]])).to(device)
    hidden = silu(gate_stack[:, None, :]) * up_stack[None, :, :]
    hidden_flat = hidden.reshape(len(rates) * len(rates), 512)
    down_stack = []
    for snapshot in snapshots["down"]:
        matrix = matrices["down"][0].copy()
        for coordinate, depth in enumerate(snapshot["depth"]):
            if depth >= 1:
                matrix[:, coordinate] += deltas["down"][0][:, coordinate]
            if depth >= 2:
                matrix[:, coordinate] += deltas["down"][1][:, coordinate]
        down_stack.append(matrix)
    down_tensor = torch.from_numpy(np.stack(down_stack)).to(device)
    outputs = torch.einsum("ai,koi->ako", hidden_flat, down_tensor).reshape(len(rates), len(rates), len(rates), 2048)
    reference = torch.from_numpy(matrices["down"][2] @ href).to(device)
    base_hidden = (gbase / (1.0 + np.exp(-np.clip(gbase, -80, 80)))) * ubase
    base_output = torch.from_numpy(matrices["down"][0] @ base_hidden).to(device)
    proxy = torch.from_numpy(np.asarray(state["proxy"], np.float32)).to(device)
    base_damage = float(qenergy(reference - base_output, proxy, float(state["beta"])))
    damage = qenergy(reference[None, None, None, :] - outputs, proxy, float(state["beta"])).cpu().numpy()
    rows = []
    actions = []
    for budget in budgets:
        choices = []
        for ig, iu, idown in np.ndindex(damage.shape):
            physical = sum((snapshots[projection][index]["physical_bytes"] for projection, index in (("gate", ig), ("up", iu), ("down", idown))))
            if 8.0 * physical / EXPERT_WEIGHTS <= budget + 1e-12:
                choices.append((float(damage[ig, iu, idown]), ig, iu, idown, physical))
        value, ig, iu, idown, physical = min(choices)
        selected = {"gate": snapshots["gate"][ig], "up": snapshots["up"][iu], "down": snapshots["down"][idown]}
        logical = sum(item["logical_bytes"] for item in selected.values())
        row = {
            "candidate": candidate["name"], "design": candidate["design"], "packet": packet,
            "request_id": str(data["request_id"][record]), "sequence_id": str(data["sequence_id"][record]),
            "position": int(data["position"][record]), "split": split, "layer": state["layer"],
            "expert_id": expert, "expert_stratum": state["strata"][expert], "router_rank": int(rank) + 1,
            "router_coefficient": float(data["router_weights"][record, rank]), "budget_bpw": budget,
            "logical_bytes": logical, "physical_bytes": physical,
            "logical_bpw": 8.0 * logical / EXPERT_WEIGHTS, "physical_bpw": 8.0 * physical / EXPERT_WEIGHTS,
            "page_amplification": physical / max(logical, 1), "recovery": 1.0 - value / max(base_damage, 1e-30),
            "damage": value, "base_damage": base_damage,
        }
        for projection, snapshot in selected.items():
            row[f"{projection}_bytes"] = snapshot["physical_bytes"]
            row[f"{projection}_q2_q3"] = snapshot["q2_q3_upgrades"]
            row[f"{projection}_q3_q4"] = snapshot["q3_q4_upgrades"]
            actions.append({
                "candidate": candidate["name"], "packet": packet, "request_id": row["request_id"],
                "position": row["position"], "layer": state["layer"], "expert_id": expert,
                "budget_bpw": budget, "projection": projection,
                "depth_u8": snapshot["depth"].tobytes(), "page_ids_u16": snapshot["pages"].tobytes(),
                "physical_bytes": snapshot["physical_bytes"], "logical_bytes": snapshot["logical_bytes"],
            })
        rows.append(row)
    del gate_stack, up_stack, hidden, hidden_flat, down_tensor, outputs, reference, proxy
    return rows, actions


def selective_stage(args: argparse.Namespace, config: dict[str, Any]) -> None:
    layers = list(map(int, args.layers or config["layers"]))
    states, _ = prepare_states(config, args.captures, args.checkpoint, layers)
    codec_root = args.output / "serialized_manifests" / "codecs"
    names = json.loads((args.output / "serialized_manifests" / "candidate_index.json").read_text())
    if args.candidates:
        selected = set(args.candidates.split(","))
        names = [name for name in names if name in selected]
    else:
        raise ValueError("selective stage requires validation-selected --candidates")
    rates = list(map(float, config["selective_projection_rates_bpw"]))
    budgets = list(map(float, config["physical_budgets_bpw"]))
    split = str(args.split)
    max_invocations = int(
        config[
            "max_validation_invocations_per_expert"
            if split == "validation"
            else "max_test_invocations_per_expert"
        ]
    )
    shard_directory = "selective_shards" if split == "test" else "selective_validation_shards"
    suffix = "_".join(map(str, layers))
    for name in names:
        rows: list[dict[str, Any]] = []
        action_rows: list[dict[str, Any]] = []
        candidate = load_candidate(codec_root / name)
        for layer in layers:
            state = states[layer]
            for expert in state["experts"]:
                matrices = decoded_matrices(candidate, state, expert)
                deltas = {p: [matrices[p][1] - matrices[p][0], matrices[p][2] - matrices[p][1]] for p in PROJECTIONS}
                records, ranks = occurrence(state["data"], expert, split, max_invocations)
                for record, rank in zip(records, ranks):
                    x = np.asarray(state["data"]["x"][record], np.float32)
                    gref = matrices["gate"][2] @ x
                    uref = matrices["up"][2] @ x
                    href = (gref / (1.0 + np.exp(-np.clip(gref, -80, 80)))) * uref
                    inputs = {"gate": x, "up": x, "down": href}
                    orders = {}
                    for projection in PROJECTIONS:
                        separate, paired = projection_orders(
                            deltas[projection], inputs[projection], projection,
                            state["proxy"], state["beta"],
                        )
                        orders[projection] = {"separate": separate, "paired": paired}
                    for packet in ("separate", "paired"):
                        local, local_actions = selective_invocation(
                            candidate, state, expert, int(record), int(rank), matrices,
                            orders, rates, budgets, packet, split,
                        )
                        rows.extend(local)
                        action_rows.extend(local_actions)
                    print(json.dumps({"selective": name, "layer": layer, "expert": expert, "request": str(state["data"]["request_id"][record])}), flush=True)
                del matrices, deltas
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        # A complete candidate is an independently mergeable transaction. This
        # bounds memory and preserves completed work if a long remote worker is
        # interrupted while evaluating a later candidate.
        key = safe_key(name)
        atomic_parquet(
            args.output / "metrics" / shard_directory / f"selective_{key}_layers_{suffix}.parquet",
            rows,
        )
        atomic_parquet(
            args.output / "metrics" / shard_directory / f"actions_{key}_layers_{suffix}.parquet",
            action_rows,
        )
        del candidate


def diagnostics_stage(args: argparse.Namespace, config: dict[str, Any]) -> None:
    from oracle_study.mxfp4_embed import LEAF_VALUES

    if not args.candidates or "," in args.candidates:
        raise ValueError("diagnostics requires exactly one selected B candidate")
    layers = list(map(int, args.layers or config["layers"]))
    states, _ = prepare_states(config, args.captures, args.checkpoint, layers)
    candidate = load_candidate(args.output / "serialized_manifests" / "codecs" / args.candidates)
    if candidate["design"] != "B" or int(candidate["group_size"]) != 32:
        raise ValueError("family specialization diagnostic is defined for Design B/G32")
    rows: list[dict[str, Any]] = []
    utilization: list[dict[str, Any]] = []
    for layer, state in states.items():
        for expert in state["experts"]:
            importance = projection_importance(
                state, expert, "expert_functional_proxy", int(config["max_train_invocations_per_expert"]),
            )
            for projection in PROJECTIONS:
                tensor = state["tensors"][expert][projection]
                key = record_key(layer, expert, projection)
                ids = np.asarray(candidate["selectors"][key], np.int64)
                codes = tensor.codes.reshape(-1, 32)
                normalized = LEAF_VALUES[codes]
                scale_codes = tensor.scale_codes.reshape(-1)
                local_importance = np.asarray(importance[projection], np.float32).reshape(-1, 32)
                sample_count = min(len(ids), int(config.get("family_diagnostic_blocks_per_projection", 8192)))
                sample = np.linspace(0, len(ids) - 1, sample_count, dtype=np.int64)
                counts = np.bincount(ids, minlength=int(candidate["families"]))
                probability = counts[counts > 0] / max(counts.sum(), 1)
                entropy = float(-np.sum(probability * np.log(probability)))
                scope = candidate_scope(candidate, layer, expert, projection, state["strata"][expert])
                utilization.append({
                    "candidate": candidate["name"], "scope": scope, "layer": layer, "expert_id": expert,
                    "expert_stratum": state["strata"][expert], "projection": projection,
                    "entropy_nats": entropy, "effective_families": float(np.exp(entropy)),
                    "used_families": int(np.sum(counts > 0)), "blocks": len(ids),
                })
                for block in sample:
                    values = normalized[block]
                    histogram = np.bincount(codes[block], minlength=16) / 32.0
                    nonzero = values[values != 0]
                    rms = float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))
                    row = {
                        "candidate": candidate["name"], "scope": scope, "family_id": int(ids[block]),
                        "layer": layer, "expert_id": expert, "expert_stratum": state["strata"][expert],
                        "projection": projection, "block_index": int(block),
                        "zero_fraction": float(np.mean(values == 0)),
                        "half_fraction": float(np.mean(np.abs(values) == 0.5)),
                        "sign_imbalance": 0.0 if not len(nonzero) else float(np.mean(nonzero > 0) - np.mean(nonzero < 0)),
                        "abs_mean": float(np.mean(np.abs(values))), "variance": float(np.var(values)),
                        "max_rms_ratio": float(np.max(np.abs(values)) / max(rms, 1e-30)),
                        "scale_exponent": int(scale_codes[block]) - 127,
                        "activation_weighted_importance": float(np.mean(local_importance[block])),
                        "distinct_leaf_count": int(np.sum(histogram > 0)),
                    }
                    for leaf in range(16):
                        row[f"leaf_{leaf:02d}_fraction"] = float(histogram[leaf])
                    rows.append(row)
    frame = pd.DataFrame(rows)
    features = [
        "zero_fraction", "half_fraction", "sign_imbalance", "abs_mean", "variance",
        "max_rms_ratio", "scale_exponent", "activation_weighted_importance", "distinct_leaf_count",
    ]
    rng = np.random.default_rng(int(config["seed"]))
    predictability_rows = []
    correct = majority_correct = tested = 0
    # Family IDs are table-local: ID 3 for gate has no semantic relationship
    # to ID 3 for up/down, so predictability must be scored per table scope.
    for scope, scoped in frame.groupby("scope"):
        train_mask = rng.random(len(scoped)) < 0.7
        train = scoped[train_mask]
        test = scoped[~train_mask]
        means = train[features].mean().to_numpy()
        scales = np.maximum(train[features].std().to_numpy(), 1e-12)
        centers = {}
        for family, group in train.groupby("family_id"):
            centers[int(family)] = ((group[features].to_numpy() - means) / scales).mean(axis=0)
        center_ids = np.asarray(sorted(centers), np.int64)
        center_values = np.stack([centers[int(value)] for value in center_ids])
        standardized = (test[features].to_numpy() - means) / scales
        prediction = center_ids[np.argmin(np.sum((standardized[:, None, :] - center_values[None, :, :]) ** 2, axis=2), axis=1)]
        truth = test.family_id.to_numpy()
        majority = int(train.family_id.mode().iloc[0])
        local_correct = int(np.sum(prediction == truth))
        local_majority = int(np.sum(truth == majority))
        correct += local_correct
        majority_correct += local_majority
        tested += len(test)
        predictability_rows.append({
            "scope": scope, "train_blocks": len(train), "test_blocks": len(test),
            "nearest_centroid_accuracy": local_correct / max(len(test), 1),
            "majority_accuracy": local_majority / max(len(test), 1),
        })
    predictability = {
        "candidate": candidate["name"], "features": features,
        "test_blocks": tested, "nearest_centroid_accuracy": correct / max(tested, 1),
        "majority_accuracy": majority_correct / max(tested, 1),
        "scope_results": predictability_rows,
    }
    atomic_parquet(args.output / "metrics" / "family_block_features.parquet", frame)
    pd.DataFrame(utilization).to_csv(args.output / "metrics" / "family_utilization.csv", index=False)
    atomic_json(args.output / "metrics" / "family_predictability.json", predictability)


def merge_stage(args: argparse.Namespace, config: dict[str, Any]) -> None:
    metrics = args.output / "metrics"
    dense_files = sorted((metrics / "dense_shards").glob("dense_layers_*.parquet"))
    projection_files = sorted((metrics / "dense_shards").glob("projection_layers_*.parquet"))
    selective_files = sorted((metrics / "selective_shards").glob("selective_*.parquet"))
    action_files = sorted((metrics / "selective_shards").glob("actions_*.parquet"))
    if dense_files:
        atomic_parquet(metrics / "dense_metrics.parquet", pd.concat([pd.read_parquet(path) for path in dense_files], ignore_index=True))
    if projection_files:
        atomic_parquet(metrics / "projection_metrics.parquet", pd.concat([pd.read_parquet(path) for path in projection_files], ignore_index=True))
    if selective_files:
        atomic_parquet(metrics / "selective_metrics.parquet", pd.concat([pd.read_parquet(path) for path in selective_files], ignore_index=True))
    if action_files:
        atomic_parquet(metrics / "selected_actions.parquet", pd.concat([pd.read_parquet(path) for path in action_files], ignore_index=True))
    atomic_json(metrics / "merge_manifest.json", {
        "completed_unix": time.time(), "dense_shards": [str(path) for path in dense_files],
        "projection_shards": [str(path) for path in projection_files],
        "selective_shards": [str(path) for path in selective_files],
        "action_shards": [str(path) for path in action_files],
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("fit", "dense", "selective", "diagnostics", "merge"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--captures", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--trees", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--candidates", type=str)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    experiment = args.config.resolve().parents[1]
    sys.path[:0] = [str(experiment / "src"), str(experiment / "scripts")]
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage in ("fit", "dense", "selective", "diagnostics"):
        for attribute in ("captures", "checkpoint", "trees"):
            if getattr(args, attribute) is None:
                raise ValueError(f"--{attribute} is required for {args.stage}")
    if args.stage == "fit":
        fit_stage(args, config)
    elif args.stage == "dense":
        dense_stage(args, config)
    elif args.stage == "selective":
        selective_stage(args, config)
    elif args.stage == "diagnostics":
        diagnostics_stage(args, config)
    else:
        merge_stage(args, config)


if __name__ == "__main__":
    main()
