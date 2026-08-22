#!/usr/bin/env python3
"""Fit and evaluate sparse transforms of the streamed gate/up refinement.

This is a basis-ceiling continuation of PR #13.  It keeps the resident Q2
matrices unchanged, fits shared transforms on train only, and evaluates the
locked validation hot-expert cohort used by the earlier OMP-gap study.  Down is
held at Q4 so the nonlinear output metric isolates gate/up refinement quality.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(EXPERIMENT / "scripts")]

from oracle_study.mxfp4_embed import audit_checkpoint, load_compressed_mxfp4_expert  # noqa: E402
from oracle_study.refinement_rotation import (  # noqa: E402
    block_joint_diagonalize,
    canonicalize_columns,
    exact_fixed_coefficient_order,
    materialize_block_basis,
    quantize_columns_symmetric,
    recovery_at_counts,
)
from oracle_study.sparse_streaming import silu, silu_prime  # noqa: E402
from run_mxfp4_selective_pages import occurrence, tree_from_record  # noqa: E402
import run_set_utility_distillation as prior  # noqa: E402
import run_sparse_streaming_study as sparse  # noqa: E402


PROJECTIONS = ("gate", "up", "down")
FIT_FILE = "refinement_rotation_layer_{layer}.npz"
FIT_FACTS = "refinement_rotation_fit_facts.json"
EVAL_ROWS = "refinement_rotation_validation.parquet"
OMP_ROWS = "refinement_rotation_omp_audit.parquet"
EVAL_FACTS = "refinement_rotation_evaluation_facts.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    return {
        "runner_sha256": sha256(Path(__file__).resolve()),
        "rotation_core_sha256": sha256(
            EXPERIMENT / "src/oracle_study/refinement_rotation.py"
        ),
        "mxfp4_core_sha256": sha256(EXPERIMENT / "src/oracle_study/mxfp4_embed.py"),
    }


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def atomic_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(list(rows)).to_parquet(temporary, index=False)
    temporary.replace(path)


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("base_pr13_commit") != "dfba3748e51d6916dc1f28cb1d3b3250188adbbe":
        raise RuntimeError("rotation audit is not based on frozen PR #13")
    if config.get("fit_split") != "train" or config.get("selection_split") != "validation":
        raise RuntimeError("split contract changed")
    if config.get("no_test_rows_admitted") is not True:
        raise RuntimeError("test admission must remain forbidden")
    if list(map(int, config.get("layers", []))) != [0, 4, 20, 39]:
        raise RuntimeError("layer grid changed")
    if int(config.get("input_width", -1)) != 2048 or int(config.get("expert_width", -1)) != 512:
        raise RuntimeError("expert geometry changed")
    if int(config.get("shared_rank", -1)) != 1024:
        raise RuntimeError("shared rank changed")
    if int(config.get("physical_page_bytes", -1)) != 512:
        raise RuntimeError("physical page size changed")
    if int(config.get("paired_gate_up_bits", -1)) != 4:
        raise RuntimeError("paired encoding must remain INT4")
    if int(config.get("full_rank_paired_bits", -1)) != 2:
        raise RuntimeError("full-rank paired encoding must remain INT2")
    if int(config.get("coordinates_per_int2_page", -1)) != 2:
        raise RuntimeError("an INT2 page must contain two fixed coordinate pairs")
    if int(config.get("separate_projection_bits", -1)) != 8:
        raise RuntimeError("separate encoding must remain INT8")


def locked_inputs(
    config: Mapping[str, Any], checkpoint: Path, trees: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    hashes = {
        "checkpoint_config_sha256": sha256(checkpoint / "config.json"),
        "checkpoint_index_sha256": sha256(checkpoint / "model.safetensors.index.json"),
        "tree_sha256": sha256(trees),
    }
    reference = config["reference"]
    if hashes["checkpoint_config_sha256"] != reference["config_sha256"]:
        raise RuntimeError("checkpoint config hash changed")
    if hashes["checkpoint_index_sha256"] != reference["index_sha256"]:
        raise RuntimeError("checkpoint index hash changed")
    if hashes["tree_sha256"] != config["locked_tree_sha256"]:
        raise RuntimeError("selected refinement tree changed")
    audit = audit_checkpoint(checkpoint, config["layers"])
    if audit.get("passed") is not True:
        raise RuntimeError(f"checkpoint audit failed: {audit['errors']}")
    return hashes, audit


def load_trees(path: Path) -> dict[str, Any]:
    records = json.loads(path.read_text())
    return {name: tree_from_record(records[name]) for name in PROJECTIONS}


def decode_expert(
    checkpoint: Path, index: Mapping[str, str], trees: Mapping[str, Any],
    layer: int, expert: int,
) -> dict[str, list[np.ndarray]]:
    return prior.decode_expert(checkpoint, index, trees, layer, expert)


def decode_gate_up_residuals(
    checkpoint: Path, index: Mapping[str, str], trees: Mapping[str, Any],
    layer: int, expert: int,
) -> tuple[np.ndarray, np.ndarray]:
    result = []
    for projection in ("gate", "up"):
        tensor = load_compressed_mxfp4_expert(
            checkpoint, index, layer, expert, projection,
        )
        q2 = trees[projection].decode(tensor, 2)
        q4 = trees[projection].decode(tensor, 4)
        result.append(np.asarray(q4 - q2, np.float32))
    return result[0], result[1]


def route_weights(local: Mapping[str, np.ndarray], experts: int, mixture: float) -> np.ndarray:
    mass = np.zeros(int(experts), np.float64)
    ids = np.asarray(local["expert_ids"], np.int64)
    weights = np.asarray(local["router_weights"], np.float64)
    split = np.asarray(local["split"]).astype(str)
    for row in np.flatnonzero(split == "train"):
        np.add.at(mass, ids[row], weights[row] ** 2)
    if mass.sum() > 0.0:
        mass /= mass.sum()
    else:
        mass.fill(1.0 / len(mass))
    uniform = np.full_like(mass, 1.0 / len(mass))
    return (1.0 - float(mixture)) * mass + float(mixture) * uniform


def torch_eigenbasis(gram: torch.Tensor, rank: int) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = torch.linalg.eigh(0.5 * (gram + gram.T))
    order = torch.argsort(values, descending=True)[:int(rank)]
    basis = vectors[:, order].detach().cpu().numpy()
    basis = canonicalize_columns(basis).astype(np.float32)
    eigenvalues = torch.clamp(values[order], min=0).detach().cpu().numpy().astype(np.float64)
    return basis, eigenvalues


def fit_layer(
    config: Mapping[str, Any], checkpoint: Path, index: Mapping[str, str],
    trees: Mapping[str, Any], local: Mapping[str, np.ndarray], layer: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    started = time.time()
    width = int(config["input_width"])
    rank = int(config["shared_rank"])
    weights = route_weights(
        local, int(config["experts_per_layer"]), float(config["router_uniform_mixture"]),
    )
    train = np.asarray(local["x"], np.float32)[np.asarray(local["split"]).astype(str) == "train"]
    x = torch.as_tensor(train, dtype=torch.float32, device=device)
    activation_gram = x.T @ x / max(len(x), 1)
    activation_full_basis, activation_full_values = torch_eigenbasis(activation_gram, width)
    activation_basis = activation_full_basis[:, :rank]
    activation_values = activation_full_values[:rank]
    del x, activation_gram

    uniform_gram = torch.zeros((width, width), dtype=torch.float32, device=device)
    router_gram = torch.zeros_like(uniform_gram)
    top = np.lexsort((np.arange(len(weights)), -weights))[:int(config["ajd_experts_per_layer"])]
    ajd_residuals: list[np.ndarray] = []
    ajd_weights: list[float] = []
    decode_seconds = 0.0
    gram_seconds = 0.0
    for expert in range(int(config["experts_per_layer"])):
        decode_started = time.perf_counter()
        gate, up = decode_gate_up_residuals(
            checkpoint, index, trees, layer, expert,
        )
        decode_seconds += time.perf_counter() - decode_started
        gram_started = time.perf_counter()
        for residual in (gate, up):
            value = torch.as_tensor(residual, device=device)
            gram = value.T @ value
            uniform_gram.add_(gram, alpha=1.0 / (2.0 * int(config["experts_per_layer"])))
            router_gram.add_(gram, alpha=float(weights[expert]) / 2.0)
        if expert in set(map(int, top)):
            ajd_residuals.extend((gate, up))
            ajd_weights.extend((float(weights[expert]) / 2.0, float(weights[expert]) / 2.0))
        gram_seconds += time.perf_counter() - gram_started
        del gate, up, value, gram
    uniform_full_basis, uniform_full_values = torch_eigenbasis(uniform_gram, width)
    router_full_basis, router_full_values = torch_eigenbasis(router_gram, width)
    uniform_basis = uniform_full_basis[:, :rank]
    uniform_values = uniform_full_values[:rank]
    router_basis = router_full_basis[:, :rank]
    router_values = router_full_values[:rank]

    blocks, ajd_facts = block_joint_diagonalize(
        ajd_residuals,
        block_size=int(config["block_size"]),
        weights=np.asarray(ajd_weights, np.float64),
        sweeps=int(config["ajd_sweeps"]),
    )
    block_full = materialize_block_basis(blocks)
    gram_cpu = router_gram.detach().cpu().numpy().astype(np.float64)
    block_energy = np.einsum("ij,ij->j", block_full, gram_cpu @ block_full)
    block_order = np.lexsort((np.arange(width), -block_energy))
    block_basis = block_full[:, block_order[:rank]].astype(np.float32)
    block_full_basis = block_full[:, block_order].astype(np.float32)
    arrays = {
        "activation_pca_basis": activation_basis.astype(np.float16),
        "activation_pca_values": activation_values,
        "activation_pca_full_basis": activation_full_basis.astype(np.float16),
        "activation_pca_full_values": activation_full_values,
        "refinement_uniform_basis": uniform_basis.astype(np.float16),
        "refinement_uniform_values": uniform_values,
        "refinement_uniform_full_basis": uniform_full_basis.astype(np.float16),
        "refinement_uniform_full_values": uniform_full_values,
        "refinement_router_basis": router_basis.astype(np.float16),
        "refinement_router_values": router_values,
        "refinement_router_full_basis": router_full_basis.astype(np.float16),
        "refinement_router_full_values": router_full_values,
        "block_ajd_basis": block_basis.astype(np.float16),
        "block_ajd_full_basis": block_full_basis.astype(np.float16),
        "block_ajd_blocks": blocks.astype(np.float16),
        "block_ajd_selected_columns": block_order[:rank].astype(np.uint16),
        "block_ajd_full_column_order": block_order.astype(np.uint16),
        "block_ajd_values": block_energy[block_order[:rank]].astype(np.float64),
        "block_ajd_full_values": block_energy[block_order].astype(np.float64),
        "router_expert_weights": weights.astype(np.float64),
        "ajd_experts": np.asarray(top, np.int64),
    }
    facts = {
        "layer": int(layer),
        "train_rows": int(len(train)),
        "experts_fit": int(config["experts_per_layer"]),
        "decode_seconds": float(decode_seconds),
        "gram_seconds": float(gram_seconds),
        "wall_seconds": float(time.time() - started),
        "ajd": ajd_facts,
        "activation_top_rank_energy_fraction": float(
            activation_values.sum()
            / max(float(np.sum(np.asarray(train, np.float64) ** 2) / len(train)), 1e-30)
        ),
        "uniform_top_rank_energy_fraction": float(
            uniform_values.sum() / max(float(torch.trace(uniform_gram).item()), 1e-30)
        ),
        "router_top_rank_energy_fraction": float(
            router_values.sum() / max(float(torch.trace(router_gram).item()), 1e-30)
        ),
    }
    del uniform_gram, router_gram, gram_cpu, block_full, ajd_residuals
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return arrays, facts


def run_fit(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    hashes, audit = locked_inputs(config, args.checkpoint, args.trees)
    data = prior.load_capture_admitted(args.captures, ["train"])
    if set(map(str, np.unique(data["split"]))) != {"train"}:
        raise RuntimeError("fit admitted a non-training row")
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    trees = load_trees(args.trees)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    facts: dict[str, Any] = {
        "completed": False,
        "phase": "fit",
        "run_id": config["run_id"],
        "config_sha256": sha256(args.config),
        "capture_sha256": sha256(args.captures),
        "fit_split": "train",
        "validation_rows_admitted_or_used": False,
        "test_rows_admitted_or_used": False,
        "checkpoint_audit": audit,
        "device": str(device),
        "torch": torch.__version__,
        "layers": {},
        **source_hashes(),
        **hashes,
    }
    atomic_json(args.output / FIT_FACTS, facts)
    for layer in map(int, config["layers"]):
        local = prior.layer_view(data, layer)
        arrays, layer_facts = fit_layer(
            config, args.checkpoint, index, trees, local, layer, device,
        )
        path = args.output / FIT_FILE.format(layer=layer)
        atomic_npz(path, arrays)
        facts["layers"][str(layer)] = {
            **layer_facts,
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        atomic_json(args.output / FIT_FACTS, facts)
        gc.collect()
    facts["completed"] = True
    atomic_json(args.output / FIT_FACTS, facts)


def qinner(left: np.ndarray, right: np.ndarray, proxy: np.ndarray, beta: float) -> float:
    a = np.asarray(left, np.float64).reshape(-1)
    b = np.asarray(right, np.float64).reshape(-1)
    value = float(a @ b)
    if float(beta) != 0.0:
        value += float(beta) * float((a @ proxy) @ (b @ proxy))
    return value


def qenergy(value: np.ndarray, proxy: np.ndarray, beta: float) -> float:
    return qinner(value, value, proxy, beta)


def output_recovery(
    target: np.ndarray, base: np.ndarray, approximation: np.ndarray,
    proxy: np.ndarray, beta: float,
) -> float:
    denominator = qenergy(target - base, proxy, beta)
    return 1.0 - qenergy(target - approximation, proxy, beta) / max(denominator, 1e-30)


def threshold_counts(curve: np.ndarray, targets: Sequence[float]) -> dict[float, int | None]:
    result: dict[float, int | None] = {}
    for target in map(float, targets):
        found = np.flatnonzero(np.asarray(curve) >= target)
        result[target] = None if not len(found) else int(found[0])
    return result


def isolated_curve(
    contributions: np.ndarray, order: np.ndarray, target: np.ndarray | None = None,
) -> np.ndarray:
    rows = np.asarray(contributions, np.float64)
    desired = rows.sum(axis=0) if target is None else np.asarray(target, np.float64).reshape(-1)
    if desired.size != rows.shape[1]:
        raise ValueError("isolated target width disagrees with contributions")
    denominator = float(desired @ desired)
    prefix = np.vstack((np.zeros((1, rows.shape[1])), np.cumsum(rows[order], axis=0)))
    error = desired[None, :] - prefix
    return 1.0 - np.einsum("ij,ij->i", error, error) / max(denominator, 1e-30)


def quantized_joint_representation(
    gate_residual: np.ndarray, up_residual: np.ndarray, basis: np.ndarray, bits: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    gate = quantize_columns_symmetric(np.asarray(gate_residual, np.float32) @ basis, bits)
    up = quantize_columns_symmetric(np.asarray(up_residual, np.float32) @ basis, bits)
    code_bytes = gate.code_bytes_per_column + up.code_bytes_per_column
    return gate.decoded, up.decoded, {
        "action_payload_bytes": int(code_bytes),
        "representation_code_bytes": int(gate.code_bytes + up.code_bytes),
        "representation_scale_bytes": int(gate.metadata_bytes + up.metadata_bytes),
        "representation_bytes": int(gate.code_bytes + up.code_bytes + gate.metadata_bytes + up.metadata_bytes),
    }


def joint_basis_torch(
    gate_residual: np.ndarray, up_residual: np.ndarray, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = torch.as_tensor(
        np.concatenate((gate_residual, up_residual), axis=0),
        dtype=torch.float32, device=device,
    )
    _, singular, right = torch.linalg.svd(matrix, full_matrices=False)
    basis = canonicalize_columns(right.T.detach().cpu().numpy()).astype(np.float32)
    values = (singular * singular).detach().cpu().numpy().astype(np.float64)
    del matrix, singular, right
    return basis, values


def separate_bases_torch(
    gate_residual: np.ndarray, up_residual: np.ndarray, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    result = []
    for residual in (gate_residual, up_residual):
        matrix = torch.as_tensor(residual, dtype=torch.float32, device=device)
        _, _, right = torch.linalg.svd(matrix, full_matrices=False)
        result.append(canonicalize_columns(right.T.detach().cpu().numpy()).astype(np.float32))
        del matrix, right
    return result[0], result[1]


def output_jacobian_basis_torch(
    matrices: Mapping[str, Sequence[np.ndarray]], train_activations: np.ndarray,
    proxy: np.ndarray, beta: float, device: torch.device, rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Train-only eigensystem of the first-order expert-output damage Gram."""
    train = np.asarray(train_activations, np.float32)
    if train.ndim != 2 or not len(train):
        raise ValueError("output-Jacobian fit requires training activations")
    gate4 = torch.as_tensor(matrices["gate"][2], dtype=torch.float32, device=device)
    up4 = torch.as_tensor(matrices["up"][2], dtype=torch.float32, device=device)
    gate_residual = torch.as_tensor(
        matrices["gate"][2] - matrices["gate"][0], dtype=torch.float32, device=device,
    )
    up_residual = torch.as_tensor(
        matrices["up"][2] - matrices["up"][0], dtype=torch.float32, device=device,
    )
    down = torch.as_tensor(matrices["down"][2], dtype=torch.float32, device=device)
    p = torch.as_tensor(proxy, dtype=torch.float32, device=device)
    hidden_metric = down.T @ down
    if float(beta) != 0.0:
        projected = down.T @ p
        hidden_metric.add_(projected @ projected.T, alpha=float(beta))
    x = torch.as_tensor(train, dtype=torch.float32, device=device)
    gates = x @ gate4.T
    ups = x @ up4.T
    sigmoid = torch.sigmoid(gates)
    gate_factor = ups * sigmoid * (1.0 + gates * (1.0 - sigmoid))
    up_factor = torch.nn.functional.silu(gates)
    gram = torch.zeros((train.shape[1], train.shape[1]), dtype=torch.float32, device=device)
    scale = 1.0 / float(len(train))
    for row in range(len(train)):
        combined = gate_factor[row, :, None] * gate_residual
        combined = combined + up_factor[row, :, None] * up_residual
        weighted = hidden_metric @ combined
        gram.addmm_(combined.T, weighted, alpha=scale)
    basis, values = torch_eigenbasis(gram, int(rank))
    del gate4, up4, gate_residual, up_residual, down, p, hidden_metric
    del x, gates, ups, sigmoid, gate_factor, up_factor, gram, combined, weighted
    return basis, values


def native_states(
    gate2: np.ndarray, gate4: np.ndarray, up2: np.ndarray, up4: np.ndarray,
) -> np.ndarray:
    h22 = silu(gate2) * up2
    h24 = silu(gate2) * up4
    h42 = silu(gate4) * up2
    h44 = silu(gate4) * up4
    return np.stack((h22, h24, h42, h44), axis=1)


def native_diagonal_order(
    hidden: np.ndarray, down_columns: np.ndarray, proxy: np.ndarray, beta: float,
) -> np.ndarray:
    base = hidden[:, 0]
    gate_delta = hidden[:, 2] - base
    up_delta = hidden[:, 1] - base
    projected = np.asarray(down_columns, np.float64) @ np.asarray(proxy, np.float64)
    norm = np.einsum("ij,ij->i", down_columns, down_columns)
    norm += float(beta) * np.einsum("ij,ij->i", projected, projected)
    score = np.concatenate((gate_delta * gate_delta * norm, up_delta * up_delta * norm))
    return np.lexsort((np.arange(len(score)), -score)).astype(np.int64)


def apply_native_order(hidden: np.ndarray, order: np.ndarray, count: int) -> np.ndarray:
    units = len(hidden)
    gate = np.zeros(units, bool); up = np.zeros(units, bool)
    for action in map(int, order[:min(int(count), len(order))]):
        if action < units:
            gate[action] = True
        else:
            up[action - units] = True
    state = 2 * gate.astype(np.int64) + up.astype(np.int64)
    return hidden[np.arange(units), state]


def native_exact_greedy(
    hidden: np.ndarray, down_columns: np.ndarray, proxy: np.ndarray, beta: float,
) -> np.ndarray:
    units = len(hidden)
    projected = np.asarray(down_columns, np.float64) @ np.asarray(proxy, np.float64)
    gram = np.asarray(down_columns, np.float64) @ np.asarray(down_columns, np.float64).T
    gram += float(beta) * (projected @ projected.T)
    state = np.zeros(units, np.int64)
    current_hidden = np.asarray(hidden[:, 0], np.float64).copy()
    residual_hidden = np.asarray(hidden[:, 3], np.float64) - current_hidden
    correlation = gram @ residual_hidden
    order = np.empty(2 * units, np.int64)
    for step in range(2 * units):
        gate_eligible = (state == 0) | (state == 1)
        up_eligible = (state == 0) | (state == 2)
        gate_target = np.where(state == 0, hidden[:, 2], hidden[:, 3])
        up_target = np.where(state == 0, hidden[:, 1], hidden[:, 3])
        gate_delta = gate_target - current_hidden
        up_delta = up_target - current_hidden
        gain_gate = 2.0 * gate_delta * correlation - gate_delta * gate_delta * np.diag(gram)
        gain_up = 2.0 * up_delta * correlation - up_delta * up_delta * np.diag(gram)
        gain_gate[~gate_eligible] = -np.inf
        gain_up[~up_eligible] = -np.inf
        gains = np.concatenate((gain_gate, gain_up))
        chosen = int(np.flatnonzero(gains == np.max(gains))[0])
        unit = chosen % units
        delta = gate_delta[unit] if chosen < units else up_delta[unit]
        current_hidden[unit] += delta
        correlation -= delta * gram[:, unit]
        if chosen < units:
            state[unit] = 2 if state[unit] == 0 else 3
        else:
            state[unit] = 1 if state[unit] == 0 else 3
        order[step] = chosen
    return order


def transformed_occurrence(
    *,
    method: str,
    gate_atoms: np.ndarray,
    up_atoms: np.ndarray,
    gate_basis: np.ndarray,
    up_basis: np.ndarray,
    activation: np.ndarray,
    gate2: np.ndarray,
    up2: np.ndarray,
    target_gate_correction: np.ndarray,
    target_up_correction: np.ndarray,
    target_output: np.ndarray,
    base_output: np.ndarray,
    down4: np.ndarray,
    proxy: np.ndarray,
    beta: float,
    budgets: Sequence[int],
    targets: Sequence[float],
    accounting: Mapping[str, int | float | str],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    zg = np.asarray(gate_basis, np.float64).T @ np.asarray(activation, np.float64)
    zu = np.asarray(up_basis, np.float64).T @ np.asarray(activation, np.float64)
    if (
        gate_atoms.shape[1] == up_atoms.shape[1]
        and gate_basis.shape == up_basis.shape
        and (gate_basis is up_basis or np.array_equal(gate_basis, up_basis))
    ):
        contributions = np.concatenate((gate_atoms.T * zg[:, None], up_atoms.T * zu[:, None]), axis=1)
        order = np.lexsort((np.arange(len(contributions)), -np.einsum("ij,ij->i", contributions, contributions)))
        paired = True
    else:
        gate_contribution = gate_atoms.T * zg[:, None]
        up_contribution = up_atoms.T * zu[:, None]
        contributions = np.zeros((len(gate_contribution) + len(up_contribution), gate_atoms.shape[0] + up_atoms.shape[0]), np.float64)
        contributions[:len(gate_contribution), :gate_atoms.shape[0]] = gate_contribution
        contributions[len(gate_contribution):, gate_atoms.shape[0]:] = up_contribution
        order = np.lexsort((np.arange(len(contributions)), -np.einsum("ij,ij->i", contributions, contributions)))
        paired = False
    coordinates_per_action = int(accounting.get("coordinates_per_action", 1))
    if coordinates_per_action <= 0:
        raise ValueError("coordinates_per_action must be positive")
    if coordinates_per_action != 1:
        if not paired:
            raise ValueError("packed coordinate pages require a shared gate/up basis")
        if len(contributions) % coordinates_per_action:
            raise ValueError("coordinate count is not divisible by the fixed page grouping")
        contributions = contributions.reshape(
            -1, coordinates_per_action, contributions.shape[1],
        ).sum(axis=1)
    order = np.lexsort((
        np.arange(len(contributions)),
        -np.einsum("ij,ij->i", contributions, contributions),
    ))
    isolated_target = np.concatenate((
        np.asarray(target_gate_correction, np.float64),
        np.asarray(target_up_correction, np.float64),
    ))
    curve = isolated_curve(contributions, order, isolated_target)
    threshold = threshold_counts(curve, targets)
    rows: list[dict[str, Any]] = []
    outputs: list[np.ndarray] = []
    isolated: list[float] = []
    for budget in map(int, budgets):
        count = min(budget, len(order))
        selected = order[:count]
        if paired:
            gate_delta = contributions[selected, :gate_atoms.shape[0]].sum(axis=0)
            up_delta = contributions[selected, gate_atoms.shape[0]:].sum(axis=0)
        else:
            gate_ids = selected[selected < gate_atoms.shape[1]]
            up_ids = selected[selected >= gate_atoms.shape[1]] - gate_atoms.shape[1]
            gate_delta = (gate_atoms[:, gate_ids] * zg[gate_ids][None, :]).sum(axis=1)
            up_delta = (up_atoms[:, up_ids] * zu[up_ids][None, :]).sum(axis=1)
        hidden = silu(np.asarray(gate2, np.float64) + gate_delta) * (
            np.asarray(up2, np.float64) + up_delta
        )
        output = hidden @ np.asarray(down4, np.float64).T
        outputs.append(output.astype(np.float32))
        isolated.append(float(curve[count]))
        rows.append({
            "method": method,
            "physical_pages": int(budget),
            "logical_actions": int(count),
            "isolated_gate_up_recovery": float(curve[count]),
            "expert_output_recovery": output_recovery(
                target_output, base_output, output, proxy, beta,
            ),
            "actions_at_90pct": threshold[0.9],
            "actions_at_95pct": threshold[0.95],
            "actions_at_99pct": threshold[0.99],
            "full_action_endpoint_recovery": float(curve[-1]),
            **accounting,
        })
    return rows, np.stack(outputs), contributions, np.asarray(order, np.int64)


def prepare_representations(
    config: Mapping[str, Any], matrices: Mapping[str, Sequence[np.ndarray]],
    shared: Mapping[str, np.ndarray], train_activations: np.ndarray,
    proxy: np.ndarray, beta: float, device: torch.device,
) -> tuple[dict[str, dict[str, Any]], np.ndarray]:
    gate_residual = np.asarray(matrices["gate"][2] - matrices["gate"][0], np.float32)
    up_residual = np.asarray(matrices["up"][2] - matrices["up"][0], np.float32)
    representations: dict[str, dict[str, Any]] = {}
    basis_map = {
        "identity_paired_int4": np.eye(int(config["input_width"]), dtype=np.float32),
        "activation_pca_rank1024_int4": shared["activation_pca_basis"],
        "refinement_uniform_rank1024_int4": shared["refinement_uniform_basis"],
        "refinement_router_rank1024_int4": shared["refinement_router_basis"],
        "block_ajd_rank1024_int4": shared["block_ajd_basis"],
        "identity_full_int2": np.eye(int(config["input_width"]), dtype=np.float32),
        "activation_pca_full_int2": shared["activation_pca_full_basis"],
        "refinement_uniform_full_int2": shared["refinement_uniform_full_basis"],
        "refinement_router_full_int2": shared["refinement_router_full_basis"],
        "block_ajd_full_int2": shared["block_ajd_full_basis"],
    }
    joint_basis, joint_values = joint_basis_torch(gate_residual, up_residual, device)
    basis_map["per_expert_joint_rank1024_int4"] = joint_basis.astype(np.float16).astype(np.float32)
    basis_map["per_expert_joint_rank1024_int2"] = basis_map[
        "per_expert_joint_rank1024_int4"
    ]
    output_basis, _ = output_jacobian_basis_torch(
        matrices, train_activations, proxy, beta, device, int(config["shared_rank"]),
    )
    basis_map["per_expert_output_jacobian_train_rank1024_int4"] = (
        output_basis.astype(np.float16).astype(np.float32)
    )
    for method, basis in basis_map.items():
        packed_int2 = method.endswith("_int2")
        bits = int(config["full_rank_paired_bits"] if packed_int2 else config["paired_gate_up_bits"])
        coordinates_per_action = int(config["coordinates_per_int2_page"] if packed_int2 else 1)
        gate_atoms, up_atoms, accounting = quantized_joint_representation(
            gate_residual, up_residual, basis, bits,
        )
        packed_payload_bytes = int(
            accounting["action_payload_bytes"] * coordinates_per_action
        )
        if packed_payload_bytes != int(config["physical_page_bytes"]):
            raise RuntimeError(f"{method} does not form one physical page per action")
        scope = "per_expert" if method.startswith("per_expert") else (
            "fixed" if method.startswith("identity") else "layer_shared"
        )
        if scope == "fixed":
            basis_storage = 0
        elif scope == "per_expert" or not method.startswith("block_ajd"):
            basis_storage = int(basis.size * 2)
        else:
            order_key = (
                "block_ajd_full_column_order" if method.endswith("full_int2")
                else "block_ajd_selected_columns"
            )
            basis_storage = int(
                shared["block_ajd_blocks"].size * 2 + shared[order_key].size * 2
            )
        transform_macs = (
            0 if scope == "fixed"
            else int(np.count_nonzero(basis)) if method.startswith("block_ajd")
            else int(basis.size)
        )
        representations[method] = {
            "gate_atoms": gate_atoms,
            "up_atoms": up_atoms,
            "gate_basis": basis,
            "up_basis": basis,
            "accounting": {
                **accounting,
                "action_payload_bytes": packed_payload_bytes,
                "coordinates_per_action": coordinates_per_action,
                "basis_scope": scope,
                "basis_storage_bytes": basis_storage,
                "transform_macs": transform_macs,
                "physically_encoded": True,
                "budget_semantics": "physical_512_byte_pages",
            },
        }

    representations["per_expert_joint_rank1024_fp32_oracle"] = {
        "gate_atoms": gate_residual @ joint_basis,
        "up_atoms": up_residual @ joint_basis,
        "gate_basis": joint_basis,
        "up_basis": joint_basis,
        "accounting": {
            "action_payload_bytes": int(2 * gate_residual.shape[0] * 4),
            "coordinates_per_action": 1,
            "representation_code_bytes": int(
                (gate_residual.shape[0] + up_residual.shape[0])
                * joint_basis.shape[1] * 4
            ),
            "representation_scale_bytes": 0,
            "representation_bytes": int(
                (gate_residual.shape[0] + up_residual.shape[0])
                * joint_basis.shape[1] * 4
            ),
            "basis_scope": "per_expert",
            "basis_storage_bytes": int(joint_basis.size * 2),
            "transform_macs": int(joint_basis.size),
            "physically_encoded": False,
            "budget_semantics": "action_count_only",
        },
    }

    gate_basis, up_basis = separate_bases_torch(gate_residual, up_residual, device)
    gate_basis = gate_basis.astype(np.float16).astype(np.float32)
    up_basis = up_basis.astype(np.float16).astype(np.float32)
    gate_encoded = quantize_columns_symmetric(
        gate_residual @ gate_basis, int(config["separate_projection_bits"]),
    )
    up_encoded = quantize_columns_symmetric(
        up_residual @ up_basis, int(config["separate_projection_bits"]),
    )
    representations["per_expert_separate_rank512_int8"] = {
        "gate_atoms": gate_encoded.decoded,
        "up_atoms": up_encoded.decoded,
        "gate_basis": gate_basis,
        "up_basis": up_basis,
        "accounting": {
            "action_payload_bytes": 512,
            "coordinates_per_action": 1,
            "representation_code_bytes": int(gate_encoded.code_bytes + up_encoded.code_bytes),
            "representation_scale_bytes": int(gate_encoded.metadata_bytes + up_encoded.metadata_bytes),
            "representation_bytes": int(
                gate_encoded.code_bytes + up_encoded.code_bytes
                + gate_encoded.metadata_bytes + up_encoded.metadata_bytes
            ),
            "basis_scope": "per_expert",
            "basis_storage_bytes": int((gate_basis.size + up_basis.size) * 2),
            "transform_macs": int(gate_basis.size + up_basis.size),
            "physically_encoded": True,
            "budget_semantics": "physical_512_byte_pages",
        },
    }
    return representations, joint_values


def evaluate_invocation(
    config: Mapping[str, Any], matrices: Mapping[str, Sequence[np.ndarray]],
    representations: Mapping[str, Mapping[str, Any]], joint_values: np.ndarray,
    x: np.ndarray, proxy: np.ndarray, beta: float, *, run_omp_audit: bool,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], list[dict[str, Any]]]:
    gate2, gate4 = np.asarray(matrices["gate"][0]), np.asarray(matrices["gate"][2])
    up2, up4 = np.asarray(matrices["up"][0]), np.asarray(matrices["up"][2])
    down4 = np.asarray(matrices["down"][2])
    activation = np.asarray(x, np.float32)
    g2 = gate2 @ activation; g4 = gate4 @ activation
    u2 = up2 @ activation; u4 = up4 @ activation
    hidden_states = native_states(g2, g4, u2, u4)
    target_output = hidden_states[:, 3] @ down4.T
    base_output = hidden_states[:, 0] @ down4.T
    budgets = list(map(int, config["physical_page_budgets"]))
    targets = list(map(float, config["recovery_targets"]))
    rows: list[dict[str, Any]] = []
    outputs: dict[str, np.ndarray] = {}
    omp_rows: list[dict[str, Any]] = []

    down_columns = np.asarray(down4.T, np.float64)
    native_orders = {
        "native_row_diagonal": native_diagonal_order(hidden_states, down_columns, proxy, beta),
        "native_row_exact_greedy": native_exact_greedy(hidden_states, down_columns, proxy, beta),
    }
    gate_target = g4 - g2; up_target = u4 - u2
    isolated_denominator = float(gate_target @ gate_target + up_target @ up_target)
    for method, order in native_orders.items():
        method_outputs = []
        isolated_curve_values = [0.0]
        for count in range(1, len(order) + 1):
            hidden = apply_native_order(hidden_states, order, count)
            gate_selected = np.zeros(len(hidden), bool); up_selected = np.zeros(len(hidden), bool)
            chosen = order[:count]
            gate_selected[chosen[chosen < len(hidden)]] = True
            up_selected[chosen[chosen >= len(hidden)] - len(hidden)] = True
            error = float(np.sum(gate_target[~gate_selected] ** 2) + np.sum(up_target[~up_selected] ** 2))
            isolated_curve_values.append(1.0 - error / max(isolated_denominator, 1e-30))
        curve = np.asarray(isolated_curve_values)
        threshold = threshold_counts(curve, targets)
        for budget in budgets:
            count = min(int(budget), len(order))
            hidden = apply_native_order(hidden_states, order, count)
            output = hidden @ down4.T
            method_outputs.append(output.astype(np.float32))
            rows.append({
                "method": method,
                "physical_pages": int(budget),
                "logical_actions": int(count),
                "isolated_gate_up_recovery": float(curve[count]),
                "expert_output_recovery": output_recovery(target_output, base_output, output, proxy, beta),
                "actions_at_90pct": threshold[0.9],
                "actions_at_95pct": threshold[0.95],
                "actions_at_99pct": threshold[0.99],
                "full_action_endpoint_recovery": float(curve[-1]),
                "action_payload_bytes": 512,
                "coordinates_per_action": 1,
                "representation_code_bytes": 1024 * 512,
                "representation_scale_bytes": 0,
                "representation_bytes": 1024 * 512,
                "basis_scope": "native",
                "basis_storage_bytes": 0,
                "transform_macs": 0,
                "physically_encoded": True,
                "budget_semantics": "physical_512_byte_pages",
            })
        outputs[method] = np.stack(method_outputs)

    for method, representation in representations.items():
        gate_atoms = np.asarray(representation["gate_atoms"])
        up_atoms = np.asarray(representation["up_atoms"])
        gate_basis = np.asarray(representation["gate_basis"])
        up_basis = np.asarray(representation["up_basis"])
        method_rows, method_outputs, contributions, order = transformed_occurrence(
            method=method, gate_atoms=gate_atoms, up_atoms=up_atoms,
            gate_basis=gate_basis, up_basis=up_basis, activation=activation,
            gate2=g2, up2=u2, target_gate_correction=g4 - g2,
            target_up_correction=u4 - u2,
            target_output=target_output, base_output=base_output,
            down4=down4, proxy=proxy, beta=beta, budgets=budgets, targets=targets,
            accounting=representation["accounting"],
        )
        rows.extend(method_rows); outputs[method] = method_outputs
        if run_omp_audit and method in {
            "identity_paired_int4", "block_ajd_rank1024_int4",
            "per_expert_joint_rank1024_int4", "per_expert_joint_rank1024_fp32_oracle",
            "identity_full_int2", "block_ajd_full_int2",
            "per_expert_joint_rank1024_int2",
        }:
            isolated_target = np.concatenate((g4 - g2, u4 - u2))
            exact, _ = exact_fixed_coefficient_order(contributions, target=isolated_target)
            diagonal_curve = isolated_curve(contributions, order, isolated_target)
            exact_curve = isolated_curve(contributions, exact, isolated_target)
            diagonal_threshold = threshold_counts(diagonal_curve, targets)
            exact_threshold = threshold_counts(exact_curve, targets)
            for target in targets:
                omp_rows.append({
                    "method": method,
                    "target_recovery": float(target),
                    "diagonal_actions": diagonal_threshold[target],
                    "exact_greedy_actions": exact_threshold[target],
                    "diagonal_to_exact_ratio": (
                        np.nan if diagonal_threshold[target] is None or exact_threshold[target] in (None, 0)
                        else float(diagonal_threshold[target]) / float(exact_threshold[target])
                    ),
                    "eigenvalue_energy_fraction_at_512": (
                        float(joint_values[:512].sum() / max(joint_values.sum(), 1e-30))
                        if method == "per_expert_joint_rank1024_fp32_oracle" else np.nan
                    ),
                })
    return rows, outputs, omp_rows


def run_evaluate(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    hashes, audit = locked_inputs(config, args.checkpoint, args.trees)
    fit_facts = json.loads((args.fit_dir / FIT_FACTS).read_text())
    if fit_facts.get("completed") is not True:
        raise RuntimeError("rotation fit is incomplete")
    sources = source_hashes()
    for name, value in {
        "run_id": config["run_id"],
        "config_sha256": sha256(args.config),
        "capture_sha256": sha256(args.captures),
        **sources,
        **hashes,
    }.items():
        if fit_facts.get(name) != value:
            raise RuntimeError(f"fit/evaluation provenance changed: {name}")
    data = prior.load_capture_admitted(args.captures, ["train", "validation"])
    if "test" in set(map(str, np.unique(data["split"]))):
        raise RuntimeError("evaluation admitted a test row")
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    trees = load_trees(args.trees)
    device = torch.device(args.device)
    rows: list[dict[str, Any]] = []
    omp_rows: list[dict[str, Any]] = []
    facts: dict[str, Any] = {
        "completed": False,
        "phase": "evaluate",
        "run_id": config["run_id"],
        "config_sha256": sha256(args.config),
        "capture_sha256": sha256(args.captures),
        "fit_facts_sha256": sha256(args.fit_dir / FIT_FACTS),
        "selection_split": "validation",
        "test_rows_admitted_or_used": False,
        "checkpoint_audit": audit,
        "device": str(device),
        "layers": {},
        **sources,
        **hashes,
    }
    atomic_json(args.output / EVAL_FACTS, facts)
    for layer in map(int, config["layers"]):
        layer_started = time.time()
        local = prior.layer_view(data, layer)
        proxy, proxy_facts = sparse.proxy_gradients(
            local["xplus"], [local[f"h{h}_router_logits"] for h in range(1, 5)], local["split"],
        )
        beta = float(proxy_facts["beta"])
        fit_record = fit_facts["layers"][str(layer)]
        fit_path = args.fit_dir / fit_record["file"]
        if sha256(fit_path) != fit_record["sha256"]:
            raise RuntimeError("fit layer hash changed")
        with np.load(fit_path, allow_pickle=False) as loaded:
            shared = {name: np.asarray(loaded[name]) for name in loaded.files}
        expert = int(config["pilot_expert_per_layer"][str(layer)])
        records, ranks = occurrence(
            local, expert, "validation", int(config["maximum_validation_invocations_per_expert"]),
        )
        if not len(records):
            raise RuntimeError(f"layer {layer} expert {expert} has no validation occurrences")
        matrices = decode_expert(args.checkpoint, index, trees, layer, expert)
        representations, joint_values = prepare_representations(
            config, matrices, shared,
            np.asarray(local["x"])[np.asarray(local["split"]).astype(str) == "train"],
            proxy, beta, device,
        )
        layer_rows = 0
        for occurrence_index, (record, router_rank) in enumerate(zip(records, ranks)):
            evaluated, _, omp = evaluate_invocation(
                config, matrices, representations, joint_values,
                local["x"][record], proxy, beta,
                run_omp_audit=occurrence_index == 0,
            )
            metadata = {
                "capture_source": "exact_checkpoint_recapture",
                "evaluation_split": "validation",
                "request_id": str(local["request_id"][record]),
                "position": int(local["position"][record]),
                "layer": int(layer),
                "expert_id": int(expert),
                "router_rank": int(router_rank),
                "occurrence_index": int(occurrence_index),
            }
            rows.extend([{**metadata, **row} for row in evaluated])
            omp_rows.extend([{**metadata, **row} for row in omp])
            layer_rows += len(evaluated)
        facts["layers"][str(layer)] = {
            "expert_id": int(expert),
            "validation_invocations": int(len(records)),
            "rows": int(layer_rows),
            "proxy_facts": proxy_facts,
            "wall_seconds": float(time.time() - layer_started),
        }
        atomic_parquet(args.output / EVAL_ROWS, rows)
        atomic_parquet(args.output / OMP_ROWS, omp_rows)
        atomic_json(args.output / EVAL_FACTS, facts)
        del matrices, shared, representations, joint_values
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    facts["observed_validation_invocations"] = int(
        len({(row["request_id"], row["position"], row["layer"], row["expert_id"]) for row in rows})
    )
    facts["observed_rows"] = int(len(rows))
    facts["omp_rows"] = int(len(omp_rows))
    facts["completed"] = True
    atomic_json(args.output / EVAL_FACTS, facts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("fit", "evaluate"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trees", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.phase == "evaluate" and args.fit_dir is None:
        parser.error("--fit-dir is required for evaluate")
    return args


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    if args.phase == "fit":
        run_fit(args, config)
    else:
        run_evaluate(args, config)


if __name__ == "__main__":
    main()
