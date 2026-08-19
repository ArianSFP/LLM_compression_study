#!/usr/bin/env python3
"""Validation-first factorized-neuron and response-distillation study.

The fit phase consumes training requests only and serializes frozen FP16
response factors.  Evaluation may run either validation without promotions or
test with an analyzer-frozen promotion artifact.  The locked codec, trees,
checkpoint revision and capture split labels are inherited from PR #7.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(EXPERIMENT / "scripts")]

from oracle_study.mxfp4_embed import load_compressed_mxfp4_expert
from oracle_study.neuron_selector_analysis import validate_promotion_payload
from oracle_study.neuron_selector import (
    TRANSITION_NAMES,
    complete_unit_scores,
    descending_order,
    encode_array,
    encode_unit_score_metadata,
    exact_factorized_neuron_fixed_greedy,
    factorized_unit_outputs,
    ndcg_at_k,
    unit_score_metadata,
    utility_weighted_recall,
)
from oracle_study.sparse_streaming import neuron_major_decomposition, silu
from run_mxfp4_selective_pages import occurrence, tree_from_record
import run_sparse_streaming_study as base


PROJECTIONS = ("gate", "up", "down")
PAGE_BYTES = 512
EXPERT_WEIGHTS = 3 * 2048 * 512
PAGES_PER_BPW = EXPERT_WEIGHTS // (8 * PAGE_BYTES)
UNIT_PACKET_BYTES = 3 * PAGE_BYTES
IDENTITY = ("capture_source", "evaluation_split", "request_id", "position", "layer", "expert_id")

ARTIFACTS = {
    "factorized_neuron_frontier": "factorized_neuron_frontier.parquet",
    "response_predictor_frontier": "response_predictor_frontier.parquet",
    "candidate_rerank_frontier": "candidate_rerank_frontier.parquet",
    "response_prediction_diagnostics": "response_prediction_diagnostics.parquet",
}

ARTIFACT_SCHEMAS = {
    "factorized_neuron_frontier": [*IDENTITY, "objective", "action_family", "physical_budget_bpw", "recovery"],
    "response_predictor_frontier": [*IDENTITY, "response_config_id", "training_cohort", "basis_variant", "rank", "synthesis_encoding", "physical_budget_bpw", "recovery"],
    "candidate_rerank_frontier": [*IDENTITY, "response_config_id", "training_cohort", "basis_variant", "rank", "synthesis_encoding", "candidate_units", "applied_units", "recovery"],
    "response_prediction_diagnostics": [*IDENTITY, "response_config_id", "training_cohort", "basis_variant", "rank", "synthesis_encoding", "hidden_q4_relative_mse"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame = pd.DataFrame(list(rows)) if rows else pd.DataFrame(columns=ARTIFACT_SCHEMAS.get(path.stem, []))
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def load_checkpoint_state(
    output: Path, expected_facts: Mapping[str, Any], split: str,
) -> tuple[dict[str, list[dict[str, Any]]], list[str], list[dict[str, Any]]]:
    """Load only fully committed expert transactions from an interrupted run."""
    facts_path = output / "run_facts.json"
    if not facts_path.exists():
        return {name: [] for name in ARTIFACTS}, [], []
    prior = json.loads(facts_path.read_text())
    for key, expected in expected_facts.items():
        if prior.get(key) != expected:
            raise RuntimeError(f"resume provenance mismatch for {key}: {prior.get(key)!r} != {expected!r}")
    if prior.get("completed") is True:
        raise RuntimeError("evaluation output is already complete")
    completed = list(map(str, prior.get("completed_work_units", [])))
    completed_set = set(completed)
    state: dict[str, list[dict[str, Any]]] = {}
    for name, filename in ARTIFACTS.items():
        path = output / filename
        rows = pd.read_parquet(path).to_dict("records") if path.exists() else []
        state[name] = [
            row for row in rows
            if f"{split}:{int(row['layer'])}:{int(row['expert_id'])}" in completed_set
        ]
    history = list(prior.get("failure_history", [])) + list(prior.get("failures", []))
    return state, completed, history


def load_bundle_layer(
    fit_dir: Path, manifest: Mapping[str, Any], layer: int,
) -> dict[str, np.ndarray]:
    record = manifest["layers"][str(int(layer))]
    relative = Path(str(record["path"]))
    if relative.name != str(relative) or relative.is_absolute():
        raise RuntimeError("fit bundle shard path is not a local basename")
    path = fit_dir / relative
    if sha256(path) != str(record["sha256"]):
        raise RuntimeError(f"fit bundle shard {layer} changed")
    if path.stat().st_size != int(record["bytes"]):
        raise RuntimeError(f"fit bundle shard {layer} byte count changed")
    with np.load(path, allow_pickle=False) as shard:
        values = {name: np.asarray(shard[name]) for name in shard.files}
    if len(values) != int(record["array_count"]):
        raise RuntimeError(f"fit bundle shard {layer} array count changed")
    if any(not name.startswith(f"l{int(layer)}__") for name in values):
        raise RuntimeError(f"fit bundle shard {layer} contains a foreign layer")
    return values


def load_capture(path: Path) -> dict[str, np.ndarray]:
    loaded = np.load(path, allow_pickle=False)
    result = {name: loaded[name] for name in loaded.files}
    base.verify_request_separation(result)
    return result


def verify_pr7_comparators(config: Mapping[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, record in config["pr7_comparators"].items():
        path = EXPERIMENT / str(record["path"])
        actual = sha256(path)
        if actual != str(record["sha256"]):
            raise RuntimeError(f"PR7 comparator {name} changed: {actual} != {record['sha256']}")
        hashes[name] = actual
    return hashes


def expert_strata(config: Mapping[str, Any], source: str, layer: int) -> dict[int, str]:
    key = "locked_exact_sampled_experts" if source == "exact_checkpoint" else "locked_cross_sampled_experts"
    return {int(expert): str(label) for expert, label in config[key][str(layer)].items()}


def union_experts(config: Mapping[str, Any], layer: int) -> list[int]:
    return sorted(set(expert_strata(config, "exact_checkpoint", layer)) | set(expert_strata(config, "cross_reference", layer)))


def layer_view(data: Mapping[str, np.ndarray], layer: int) -> dict[str, np.ndarray]:
    mask = np.asarray(data["layer"]) == int(layer)
    return {name: np.asarray(value)[mask] for name, value in data.items()}


def decode_expert(
    checkpoint: Path,
    index: Mapping[str, str],
    trees: Mapping[str, Any],
    layer: int,
    expert: int,
) -> dict[str, list[np.ndarray]]:
    tensors = {
        projection: load_compressed_mxfp4_expert(checkpoint, index, layer, expert, projection)
        for projection in PROJECTIONS
    }
    return {
        projection: [trees[projection].decode(tensors[projection], level) for level in (2, 3, 4)]
        for projection in PROJECTIONS
    }


def gpu_responses(
    x: np.ndarray,
    residuals: Mapping[int, tuple[np.ndarray, np.ndarray]],
    device: torch.device,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    activation = torch.as_tensor(x, dtype=torch.float32, device=device)
    gate: dict[int, np.ndarray] = {}
    up: dict[int, np.ndarray] = {}
    for expert in sorted(residuals):
        dg, du = residuals[expert]
        gate[expert] = (activation @ torch.as_tensor(dg, dtype=torch.float32, device=device).T).cpu().numpy()
        up[expert] = (activation @ torch.as_tensor(du, dtype=torch.float32, device=device).T).cpu().numpy()
    return gate, up


def canonical_subspace(
    responses: Sequence[np.ndarray], rank: int, device: torch.device,
) -> np.ndarray:
    samples = int(np.asarray(responses[0]).shape[0])
    gram = torch.zeros((samples, samples), dtype=torch.float32, device=device)
    for response in responses:
        value = torch.as_tensor(response, dtype=torch.float32, device=device)
        gram.addmm_(value, value.T)
    _, vectors = torch.linalg.eigh(gram)
    result = vectors[:, -int(rank):].flip(1).cpu().numpy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0:
            result[:, column] *= -1
    del gram, vectors
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def analysis_from_subspace(x: np.ndarray, desired: np.ndarray, ridge_relative: float) -> np.ndarray:
    samples = np.asarray(x, np.float64)
    kernel = samples @ samples.T
    ridge = float(ridge_relative) * max(float(np.trace(kernel)) / max(len(kernel), 1), 1e-30)
    coefficient = np.linalg.solve(kernel + ridge * np.eye(len(kernel)), desired)
    return (coefficient.T @ samples).astype(np.float32)


def analyses_from_subspaces(
    x: np.ndarray,
    desired: Mapping[str, np.ndarray],
    ridge_relative: float,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Fit every B transform with one shared ridge factorization."""
    activation = torch.as_tensor(x, dtype=torch.float32, device=device)
    names = list(desired)
    widths = [int(np.asarray(desired[name]).shape[1]) for name in names]
    targets = torch.as_tensor(
        np.concatenate([np.asarray(desired[name], np.float32) for name in names], axis=1),
        dtype=torch.float32, device=device,
    )
    if activation.shape[0] <= activation.shape[1]:
        kernel = activation @ activation.T
        ridge = float(ridge_relative) * max(float(torch.trace(kernel).item()) / max(len(kernel), 1), 1e-30)
        factor = torch.linalg.cholesky(kernel + ridge * torch.eye(len(kernel), device=device))
        coefficients = torch.cholesky_solve(targets, factor)
        combined = coefficients.T @ activation
    else:
        kernel = activation.T @ activation
        ridge = float(ridge_relative) * max(float(torch.trace(kernel).item()) / max(activation.shape[0], 1), 1e-30)
        factor = torch.linalg.cholesky(kernel + ridge * torch.eye(kernel.shape[0], device=device))
        combined = torch.cholesky_solve(activation.T @ targets, factor).T
    chunks = torch.split(combined, widths, dim=0)
    result = {name: chunk.cpu().numpy() for name, chunk in zip(names, chunks)}
    del activation, targets, kernel, factor, combined
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def synthesis(latent: np.ndarray, response: np.ndarray) -> np.ndarray:
    coefficient, *_ = np.linalg.lstsq(
        np.asarray(latent, np.float64), np.asarray(response, np.float64), rcond=None,
    )
    return coefficient.T.astype(np.float32)


def syntheses(
    latent: np.ndarray,
    responses: Mapping[str, np.ndarray],
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Reuse one pseudoinverse for every response sharing a latent basis."""
    design = torch.as_tensor(latent, dtype=torch.float32, device=device)
    inverse = torch.linalg.pinv(design)
    result = {
        name: (inverse @ torch.as_tensor(value, dtype=torch.float32, device=device)).T.cpu().numpy()
        for name, value in responses.items()
    }
    del design, inverse
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def relative_mse(
    gate: Mapping[int, np.ndarray], up: Mapping[int, np.ndarray],
    zg: np.ndarray, zu: np.ndarray,
    ag: Mapping[int, np.ndarray], au: Mapping[int, np.ndarray],
) -> float:
    numerator = denominator = 0.0
    for expert in sorted(gate):
        for actual, predicted in ((gate[expert], zg @ ag[expert].T), (up[expert], zu @ au[expert].T)):
            numerator += float(np.square(actual - predicted).sum())
            denominator += float(np.square(actual).sum())
    return numerator / max(denominator, 1e-30)


def fit_phase(args: argparse.Namespace, config: dict[str, Any]) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    audit_exact, hashes_exact = base.verify_locked_inputs(
        config, args.config, args.exact_captures, args.checkpoint, args.trees, "exact_checkpoint",
    )
    audit_cross, hashes_cross = base.verify_locked_inputs(
        config, args.config, args.cross_captures, args.checkpoint, args.trees, "cross_reference",
    )
    comparator_hashes = verify_pr7_comparators(config)
    exact = load_capture(args.exact_captures)
    cross = load_capture(args.cross_captures)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    records = json.loads(args.trees.read_text())
    trees = {projection: tree_from_record(records[projection]) for projection in PROJECTIONS}
    arrays: dict[str, np.ndarray] = {}
    entries: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    ridge_relative = float(config["ridge_relative"])
    ranks = list(map(int, config["response_ranks"]))

    for layer in map(int, config["layers"]):
        residuals: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for expert in union_experts(config, layer):
            matrices = decode_expert(args.checkpoint, index, trees, layer, expert)
            residuals[expert] = (
                np.asarray(matrices["gate"][2] - matrices["gate"][0], np.float32),
                np.asarray(matrices["up"][2] - matrices["up"][0], np.float32),
            )
        exact_layer = layer_view(exact, layer)
        cross_layer = layer_view(cross, layer)
        exact_train = np.asarray(exact_layer["x"][exact_layer["split"] == "train"], np.float32)
        cross_train = np.asarray(cross_layer["x"][cross_layer["split"] == "train"], np.float32)
        cohorts = {
            "exact_checkpoint_train_only": exact_train,
            "combined_exact_and_cross_train": np.concatenate((exact_train, cross_train), axis=0),
        }
        for cohort_name in config["training_cohorts"]:
            x = cohorts[str(cohort_name)]
            gate_response, up_response = gpu_responses(x, residuals, device)
            max_identifiable = min(x.shape)
            max_rank = min(max(ranks), max_identifiable)
            subspaces = {
                "shared_joint": canonical_subspace(
                    [*gate_response.values(), *up_response.values()], max_rank, device,
                ),
                "shared_gate": canonical_subspace(list(gate_response.values()), max_rank, device),
                "shared_up": canonical_subspace(list(up_response.values()), max_rank, device),
            }
            subspaces.update({
                f"expert_{expert}": canonical_subspace(
                    [gate_response[expert], up_response[expert]], max_rank, device,
                ) for expert in sorted(residuals)
            })
            fitted_analysis = analyses_from_subspaces(x, subspaces, ridge_relative, device)
            for rank in ranks:
                if rank > max_identifiable:
                    diagnostics.append({
                        "layer": layer, "training_cohort": cohort_name, "rank": rank,
                        "basis_variant": "all", "status": "skipped_unidentifiable_rank",
                        "training_samples": int(x.shape[0]),
                    })
                    continue
                for variant in config["response_basis_variants"]:
                    key = f"l{layer}__{cohort_name}__{variant}__r{rank}"
                    experts = sorted(residuals)
                    if variant == "layer_shared_joint_gate_up":
                        bg = fitted_analysis["shared_joint"][:rank]
                        bu = bg
                        zg = zu = x @ bg.T
                        fitted = syntheses(zg, {
                            **{f"gate_{expert}": gate_response[expert] for expert in experts},
                            **{f"up_{expert}": up_response[expert] for expert in experts},
                        }, device)
                        ag = {expert: fitted[f"gate_{expert}"] for expert in experts}
                        au = {expert: fitted[f"up_{expert}"] for expert in experts}
                        bg_encoded = encode_array(bg, "fp16")
                        ag_encoded = {expert: encode_array(value, "fp16") for expert, value in ag.items()}
                        au_encoded = {expert: encode_array(value, "fp16") for expert, value in au.items()}
                        bg = bu = bg_encoded.decoded
                        ag = {expert: value.decoded for expert, value in ag_encoded.items()}
                        au = {expert: value.decoded for expert, value in au_encoded.items()}
                        accounting = {
                            "layer_analysis_bytes": bg_encoded.storage_bytes,
                            "expert_synthesis_bytes": int(
                                sum(value.storage_bytes for value in (*ag_encoded.values(), *au_encoded.values()))
                                / len(experts)
                            ),
                        }
                        model_mse = relative_mse(
                            gate_response, up_response, x @ bg.T, x @ bu.T, ag, au,
                        )
                        arrays[f"{key}__bg"] = bg.astype(np.float16)
                        shared = True
                    elif variant == "layer_shared_separate_gate_up":
                        bg = fitted_analysis["shared_gate"][:rank]
                        bu = fitted_analysis["shared_up"][:rank]
                        zg, zu = x @ bg.T, x @ bu.T
                        fitted_gate = syntheses(
                            zg, {str(expert): gate_response[expert] for expert in experts}, device,
                        )
                        fitted_up = syntheses(
                            zu, {str(expert): up_response[expert] for expert in experts}, device,
                        )
                        ag = {expert: fitted_gate[str(expert)] for expert in experts}
                        au = {expert: fitted_up[str(expert)] for expert in experts}
                        bg_encoded = encode_array(bg, "fp16")
                        bu_encoded = encode_array(bu, "fp16")
                        ag_encoded = {expert: encode_array(value, "fp16") for expert, value in ag.items()}
                        au_encoded = {expert: encode_array(value, "fp16") for expert, value in au.items()}
                        bg, bu = bg_encoded.decoded, bu_encoded.decoded
                        ag = {expert: value.decoded for expert, value in ag_encoded.items()}
                        au = {expert: value.decoded for expert, value in au_encoded.items()}
                        accounting = {
                            "layer_analysis_bytes": bg_encoded.storage_bytes + bu_encoded.storage_bytes,
                            "expert_synthesis_bytes": int(
                                sum(value.storage_bytes for value in (*ag_encoded.values(), *au_encoded.values()))
                                / len(experts)
                            ),
                        }
                        model_mse = relative_mse(
                            gate_response, up_response, x @ bg.T, x @ bu.T, ag, au,
                        )
                        arrays[f"{key}__bg"] = bg.astype(np.float16)
                        arrays[f"{key}__bu"] = bu.astype(np.float16)
                        shared = False
                    elif variant == "per_expert_joint_gate_up_upper_bound":
                        bg_list, ag_list, au_list = [], [], []
                        numerator = denominator = 0.0
                        for expert in experts:
                            bg_e = fitted_analysis[f"expert_{expert}"][:rank]
                            z_e = x @ bg_e.T
                            fitted = syntheses(
                                z_e, {"gate": gate_response[expert], "up": up_response[expert]}, device,
                            )
                            ag_e, au_e = fitted["gate"], fitted["up"]
                            for actual, predicted in (
                                (gate_response[expert], z_e @ ag_e.T),
                                (up_response[expert], z_e @ au_e.T),
                            ):
                                numerator += float(np.square(actual - predicted).sum())
                                denominator += float(np.square(actual).sum())
                            bg_list.append(bg_e.astype(np.float16))
                            ag_list.append(ag_e.astype(np.float16))
                            au_list.append(au_e.astype(np.float16))
                        arrays[f"{key}__bg"] = np.stack(bg_list)
                        bg = arrays[f"{key}__bg"].astype(np.float32)
                        ag = {expert: ag_list[index].astype(np.float32) for index, expert in enumerate(experts)}
                        au = {expert: au_list[index].astype(np.float32) for index, expert in enumerate(experts)}
                        arrays[f"{key}__ag"] = np.stack(ag_list)
                        arrays[f"{key}__au"] = np.stack(au_list)
                        accounting = {
                            "layer_analysis_bytes": 0,
                            "expert_synthesis_bytes": int(2 * 512 * rank * 2),
                        }
                        entries.append({
                            "config_id": key, "layer": layer, "training_cohort": cohort_name,
                            "basis_variant": variant, "rank": rank, "synthesis_encoding": "fp16",
                            "experts": experts, "shared_analysis": False,
                            "per_expert_analysis": True,
                            "analysis_bytes_per_expert": int(rank * x.shape[1] * 2),
                            "layer_analysis_bytes": 0,
                            "expert_synthesis_bytes": accounting["expert_synthesis_bytes"],
                            "training_response_relative_mse": numerator / max(denominator, 1e-30),
                            "training_samples": int(x.shape[0]),
                        })
                        diagnostics.append({**entries[-1], "status": "fit"})
                        continue
                    else:
                        raise RuntimeError(f"unknown response basis variant: {variant}")

                    arrays[f"{key}__ag"] = np.stack([ag[expert] for expert in experts]).astype(np.float16)
                    arrays[f"{key}__au"] = np.stack([au[expert] for expert in experts]).astype(np.float16)
                    if not shared:
                        arrays[f"{key}__bu"] = np.asarray(bu, np.float16)
                    entry = {
                        "config_id": key, "layer": layer, "training_cohort": cohort_name,
                        "basis_variant": variant, "rank": rank, "synthesis_encoding": "fp16",
                        "experts": experts, "shared_analysis": shared,
                        "per_expert_analysis": False,
                        "analysis_bytes_per_expert": 0,
                        "layer_analysis_bytes": int(accounting["layer_analysis_bytes"]),
                        "expert_synthesis_bytes": int(accounting["expert_synthesis_bytes"]),
                        "training_response_relative_mse": float(model_mse),
                        "training_samples": int(x.shape[0]),
                    }
                    entries.append(entry)
                    diagnostics.append({**entry, "status": "fit"})
        del residuals
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # The response subspace is fit once. Quantized synthesis variants reuse
    # exactly the same FP16 layer analysis transform and differ only in the
    # expert-specific A factors, so encoding is validation-safe and cheap.
    fp16_entries = list(entries)
    for parent in fp16_entries:
        if bool(parent["per_expert_analysis"]):
            # This sampled-expert diagnostic is an upper bound, not a
            # bank-wide deployment candidate; FP16 diagnoses whether shared
            # basis error is the bottleneck.
            continue
        parent_key = str(parent["config_id"])
        for encoding in config["synthesis_encodings"]:
            if encoding == "fp16":
                continue
            key = f"{parent_key}__{encoding}"
            arrays[f"{key}__bg"] = arrays[f"{parent_key}__bg"]
            if not bool(parent["shared_analysis"]):
                arrays[f"{key}__bu"] = arrays[f"{parent_key}__bu"]
            encoded_gate = [
                encode_array(matrix.astype(np.float32), str(encoding))
                for matrix in arrays[f"{parent_key}__ag"]
            ]
            encoded_up = [
                encode_array(matrix.astype(np.float32), str(encoding))
                for matrix in arrays[f"{parent_key}__au"]
            ]
            arrays[f"{key}__ag"] = np.stack([value.decoded for value in encoded_gate]).astype(np.float16)
            arrays[f"{key}__au"] = np.stack([value.decoded for value in encoded_up]).astype(np.float16)
            entry = {
                **parent,
                "config_id": key,
                "synthesis_encoding": str(encoding),
                "expert_synthesis_bytes": int(
                    sum(value.storage_bytes for value in (*encoded_gate, *encoded_up))
                    / len(encoded_gate)
                ),
                "parent_fp16_config_id": parent_key,
            }
            entries.append(entry)
            diagnostics.append({**entry, "status": "encoded_from_frozen_fp16_fit"})

    shard_records: dict[str, dict[str, Any]] = {}
    for layer in map(int, config["layers"]):
        prefix = f"l{layer}__"
        layer_arrays = {name: value for name, value in arrays.items() if name.startswith(prefix)}
        shard_path = args.output / f"response_fit_layer_{layer}.npz"
        temporary = shard_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **layer_arrays)
        temporary.replace(shard_path)
        shard_records[str(layer)] = {
            "path": shard_path.name,
            "sha256": sha256(shard_path),
            "bytes": shard_path.stat().st_size,
            "array_count": len(layer_arrays),
        }
    bundle_manifest_path = args.output / "response_fit_bundle_manifest.json"
    atomic_json(bundle_manifest_path, {
        "schema_version": 1, "run_id": config["run_id"],
        "config_sha256": sha256(args.config), "layers": shard_records,
    })
    index_payload = {
        "schema_version": 1,
        "run_id": config["run_id"],
        "config_sha256": sha256(args.config),
        "fit_split": "train",
        "validation_or_test_rows_used": False,
        "bundle_manifest_sha256": sha256(bundle_manifest_path),
        "entries": entries,
    }
    index_path = args.output / "response_fit_index.json"
    atomic_json(index_path, index_payload)
    atomic_parquet(args.output / "response_fit_diagnostics.parquet", diagnostics)
    facts = {
        "completed": True,
        "phase": "fit",
        "run_id": config["run_id"],
        "config_sha256": sha256(args.config),
        "fit_split": "train",
        "validation_or_test_rows_used": False,
        "exact_capture_sha256": hashes_exact["capture_sha256"],
        "cross_capture_sha256": hashes_cross["capture_sha256"],
        "tree_sha256": hashes_exact["tree_sha256"],
        "checkpoint_config_sha256": hashes_exact["config_sha256"],
        "checkpoint_index_sha256": hashes_exact["index_sha256"],
        "checkpoint_audit_exact": audit_exact,
        "checkpoint_audit_cross": audit_cross,
        "pr7_comparator_sha256": comparator_hashes,
        "bundle_sha256": sha256(bundle_manifest_path),
        "bundle_manifest_sha256": sha256(bundle_manifest_path),
        "bundle_shards": shard_records,
        "index_sha256": sha256(index_path),
        "model_entries": len(entries),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "host": platform.node(),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "selector_core_sha256": sha256(EXPERIMENT / "src/oracle_study/neuron_selector.py"),
        "base_runner_sha256": sha256(EXPERIMENT / "scripts/run_sparse_streaming_study.py"),
    }
    atomic_json(args.output / "fit_facts.json", facts)


def metadata_for_record(
    source: str,
    split: str,
    data: Mapping[str, np.ndarray],
    record: int,
    rank: int,
    layer: int,
    expert: int,
    stratum: str,
) -> dict[str, Any]:
    return base.common_metadata(source, split, dict(data), record, rank, layer, expert, stratum)


def evaluation_plan(
    data: Mapping[str, np.ndarray], config: Mapping[str, Any], source: str, split: str,
) -> list[tuple[int, int, str, np.ndarray, np.ndarray]]:
    maximum = (
        int(config["max_validation_invocations_per_expert"])
        if split == "validation" else int(config["max_test_invocations_per_expert"])
    )
    result = []
    for layer in map(int, config["layers"]):
        local = layer_view(data, layer)
        for expert, stratum in sorted(expert_strata(config, source, layer).items()):
            records, ranks = occurrence(local, expert, split, maximum)
            if len(records):
                result.append((layer, expert, stratum, records, ranks))
    return result


def frontier_fields(
    config: Mapping[str, Any], *, pages: int, logical_actions: int,
    logical_bytes: int, recovery: float, runtime_seconds: float,
    selector_bytes_read: int, metadata_bytes: float,
) -> dict[str, Any]:
    physical_bytes = int(pages) * PAGE_BYTES
    metadata_bpw = 8.0 * float(metadata_bytes) / EXPERT_WEIGHTS
    return {
        "physical_pages": int(pages),
        "physical_bytes": physical_bytes,
        "physical_bpw": 8.0 * physical_bytes / EXPERT_WEIGHTS,
        "logical_actions": int(logical_actions),
        "logical_bytes": int(logical_bytes),
        "logical_bpw": 8.0 * int(logical_bytes) / EXPERT_WEIGHTS,
        "recovery": float(recovery),
        "selector_runtime_ms": 1000.0 * float(runtime_seconds),
        "selector_bytes_read": int(selector_bytes_read),
        "selector_bytes_read_semantics": "unique_tensor_footprint_lower_bound_not_hardware_traffic",
        "suffix_storage_bpw": float(config["suffix_bpw_per_complete_representation"]),
        "selector_metadata_bytes_per_expert": float(metadata_bytes),
        "selector_metadata_bpw": metadata_bpw,
        "storage_multiplier": (
            float(config["reference_bpw"])
            + float(config["suffix_bpw_per_complete_representation"])
            + metadata_bpw
        ) / float(config["reference_bpw"]),
        "page_amplification": physical_bytes / max(int(logical_bytes), 1),
    }


def model_arrays(
    bundle: Mapping[str, np.ndarray], entry: Mapping[str, Any], expert: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    key = str(entry["config_id"])
    experts = list(map(int, entry["experts"]))
    index = experts.index(int(expert))
    bg_all = np.asarray(bundle[f"{key}__bg"], np.float32)
    bg = bg_all[index] if bool(entry["per_expert_analysis"]) else bg_all
    if bool(entry["shared_analysis"]) or bool(entry["per_expert_analysis"]):
        bu = bg
    else:
        bu = np.asarray(bundle[f"{key}__bu"], np.float32)
    ag = np.asarray(bundle[f"{key}__ag"], np.float32)[index]
    au = np.asarray(bundle[f"{key}__au"], np.float32)[index]
    return bg, bu, ag, au


def output_for_units(base_output: np.ndarray, corrections: np.ndarray, order: np.ndarray, count: int) -> np.ndarray:
    chosen = np.asarray(order[: int(count)], np.int64)
    return np.asarray(base_output, np.float64) + np.asarray(corrections, np.float64)[chosen].sum(axis=0)


def best_factorized_prefix(
    factorized: Any, trace: Any, budget_pages: int,
) -> tuple[np.ndarray, int, int, np.ndarray, int]:
    """Return the highest-qenergy prefix whose paid pages fit the budget.

    The greedy path may traverse a locally negative prerequisite to unlock a
    jointly profitable complementary transition. We therefore construct the
    whole budget-constrained path, then choose its best cumulative prefix
    (including the empty prefix) rather than blindly applying its last state.
    """
    cumulative_pages = np.asarray(trace.cumulative_pages, np.int64)
    evaluated = int(np.sum(cumulative_pages <= int(budget_pages)))
    cumulative_gain = np.cumsum(np.asarray(trace.gains, np.float64)[:evaluated])
    best_actions = int(np.argmax(np.concatenate(([0.0], cumulative_gain))))
    chosen = np.asarray(trace.order, np.int64)[:best_actions]
    paid = int(cumulative_pages[best_actions - 1]) if best_actions else 0
    vectors = np.asarray(factorized.transitions, np.float64).reshape(
        4 * int(factorized.units), -1,
    )
    output = np.asarray(factorized.base_output, np.float64).copy()
    if best_actions:
        output += vectors[chosen].sum(axis=0)
    return output, best_actions, paid, chosen, evaluated


def evaluate_invocation(
    *,
    matrices: Mapping[str, list[np.ndarray]],
    activation: np.ndarray,
    proxy: np.ndarray,
    beta: float,
    metadata: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
    entries: Sequence[Mapping[str, Any]],
    bundle: Mapping[str, np.ndarray],
) -> dict[str, list[dict[str, Any]]]:
    q2 = tuple(matrices[name][0] for name in PROJECTIONS)
    q4 = tuple(matrices[name][2] for name in PROJECTIONS)
    factorized = factorized_unit_outputs(q2, q4, activation)
    target, base_output = factorized.target_output, factorized.base_output
    corrections = np.asarray(factorized.y11 - factorized.y00, np.float64)
    exact_metadata = unit_score_metadata(q2[2], q4[2], proxy=proxy, beta=beta)
    encoded_metadata, abc_bytes = encode_unit_score_metadata(exact_metadata, "fp16")
    exact_score = complete_unit_scores(factorized.h2, factorized.h4, exact_metadata)
    exact_order = descending_order(exact_score)
    result = {name: [] for name in ARTIFACTS}

    # Controlled coherent-packet baseline, computed through the scalar identity.
    for count in map(int, config["coherent_unit_counts"]):
        started = time.perf_counter()
        output = output_for_units(base_output, corrections, exact_order, count)
        runtime = time.perf_counter() - started
        result["factorized_neuron_frontier"].append({
            **metadata,
            "objective": "exact_sequential_complete_expert_qenergy",
            "action_family": "coherent_complete_unit_packet",
            "representation": "q2_q4_gate_up_down_together_3pages",
            "selector": "exact_independent_unit_qenergy_via_abc",
            "selection_regime": "h0_oracle_exact_hidden_scalar",
            "selection_regime_category": "h0_oracle",
            "physical_budget_bpw": 3 * count / PAGES_PER_BPW,
            "gate_up_actions": count,
            "down_actions": count,
            **frontier_fields(
                config, pages=3 * count, logical_actions=count,
                logical_bytes=count * UNIT_PACKET_BYTES,
                recovery=base.recovery(target, base_output, output, proxy, beta),
                runtime_seconds=runtime, selector_bytes_read=corrections.nbytes,
                metadata_bytes=0,
            ),
        })

    # Each budget gets an independently constrained exact path.
    for budget_bpw in map(float, config["factorized_physical_budgets_bpw"]):
        budget_pages = int(round(budget_bpw * PAGES_PER_BPW))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        trace = exact_factorized_neuron_fixed_greedy(
            factorized, page_budgets=[budget_pages], proxy=proxy, beta=beta, device=device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        runtime = time.perf_counter() - started
        output, actions, paid, chosen, evaluated = best_factorized_prefix(
            factorized, trace, budget_pages,
        )
        blocks = chosen // factorized.units
        gate_actions = int(np.sum(np.isin(blocks, [0, 3])))
        down_actions = int(np.sum(np.isin(blocks, [1, 2])))
        result["factorized_neuron_frontier"].append({
            **metadata,
            "objective": "exact_sequential_complete_expert_qenergy",
            "action_family": "factorized_gate_up_down_unit_states",
            "representation": "G_2pages_D_1page_four_exact_states",
            "selector": "exact_factorized_marginal_per_page_fixed_greedy",
            "selection_regime": "h0_oracle_full_suffix",
            "selection_regime_category": "h0_oracle",
            "physical_budget_bpw": budget_bpw,
            "gate_up_actions": gate_actions,
            "down_actions": down_actions,
            "path_actions_evaluated": evaluated,
            "prefix_policy": "best_exact_qenergy_prefix_under_budget",
            "transition_order": json.dumps([TRANSITION_NAMES[int(block)] for block in blocks]),
            **frontier_fields(
                config, pages=paid, logical_actions=actions, logical_bytes=paid * PAGE_BYTES,
                recovery=base.recovery(target, base_output, output, proxy, beta),
                runtime_seconds=runtime,
                selector_bytes_read=int(factorized.transitions.size * 4 + (4 * factorized.units) ** 2 * 4),
                metadata_bytes=0,
            ),
        })

    gate2_response = np.asarray(q2[0] @ activation, np.float64)
    up2_response = np.asarray(q2[1] @ activation, np.float64)
    exact_delta_gate = np.asarray((q4[0] - q2[0]) @ activation, np.float64)
    exact_delta_up = np.asarray((q4[1] - q2[1]) @ activation, np.float64)
    for entry in entries:
        expert = int(metadata["expert_id"])
        bg, bu, ag, au = model_arrays(bundle, entry, expert)
        started = time.perf_counter()
        zg = bg @ np.asarray(activation, np.float32)
        zu = zg if bool(entry["shared_analysis"]) or bool(entry["per_expert_analysis"]) else bu @ np.asarray(activation, np.float32)
        predicted_delta_gate = ag @ zg
        predicted_delta_up = au @ zu
        predicted_h4 = silu(gate2_response + predicted_delta_gate) * (up2_response + predicted_delta_up)
        predicted_score = complete_unit_scores(factorized.h2, predicted_h4, encoded_metadata)
        predicted_order = descending_order(predicted_score)
        prediction_runtime = time.perf_counter() - started
        analysis_bytes = (
            int(entry["analysis_bytes_per_expert"])
            if bool(entry["per_expert_analysis"])
            else float(entry["layer_analysis_bytes"]) / 256.0
        )
        metadata_bytes = analysis_bytes + int(entry["expert_synthesis_bytes"]) + abc_bytes
        rank = int(entry["rank"])
        macs = (
            (2 if not bool(entry["shared_analysis"]) and not bool(entry["per_expert_analysis"]) else 1)
            * 2048 * rank + 2 * 512 * rank
        )
        config_fields = {
            "response_config_id": str(entry["config_id"]),
            "training_cohort": str(entry["training_cohort"]),
            "basis_variant": str(entry["basis_variant"]),
            "rank": rank,
            "synthesis_encoding": str(entry["synthesis_encoding"]),
            "selector_compute_macs": int(macs),
            "second_pass_required": True,
        }
        denominator_gate = max(float(np.square(exact_delta_gate).sum()), 1e-30)
        denominator_up = max(float(np.square(exact_delta_up).sum()), 1e-30)
        denominator_hidden = max(float(np.square(factorized.h4).sum()), 1e-30)
        result["response_prediction_diagnostics"].append({
            **metadata, **config_fields,
            "delta_gate_relative_mse": float(np.square(predicted_delta_gate - exact_delta_gate).sum()) / denominator_gate,
            "delta_up_relative_mse": float(np.square(predicted_delta_up - exact_delta_up).sum()) / denominator_up,
            "hidden_q4_relative_mse": float(np.square(predicted_h4 - factorized.h4).sum()) / denominator_hidden,
            "unit_score_ndcg_192": ndcg_at_k(exact_score, predicted_order, 192),
            "unit_score_ndcg_256": ndcg_at_k(exact_score, predicted_order, 256),
            "utility_weighted_recall_192": utility_weighted_recall(exact_score, predicted_order, 192),
            "utility_weighted_recall_256": utility_weighted_recall(exact_score, predicted_order, 256),
            "predictor_runtime_ms": 1000.0 * prediction_runtime,
            "selector_metadata_bytes_per_expert": metadata_bytes,
            "selector_metadata_bpw": 8.0 * metadata_bytes / EXPERT_WEIGHTS,
        })
        for count in map(int, config["coherent_unit_counts"]):
            output = output_for_units(base_output, corrections, predicted_order, count)
            result["response_predictor_frontier"].append({
                **metadata, **config_fields,
                "objective": "exact_sequential_complete_expert_qenergy",
                "action_family": "predicted_complete_unit_packet",
                "representation": "q2_q4_gate_up_down_together_3pages",
                "selector": "predicted_h4_abc_independent_unit_score",
                "selection_regime": "deployable_h0_response_predictor_late",
                "selection_regime_category": "deployable_h0_proxy",
                "physical_budget_bpw": 3 * count / PAGES_PER_BPW,
                "unit_count": count,
                "unit_score_ndcg": ndcg_at_k(exact_score, predicted_order, count),
                "utility_weighted_recall": utility_weighted_recall(exact_score, predicted_order, count),
                **frontier_fields(
                    config, pages=3 * count, logical_actions=count,
                    logical_bytes=count * UNIT_PACKET_BYTES,
                    recovery=base.recovery(target, base_output, output, proxy, beta),
                    runtime_seconds=prediction_runtime,
                    selector_bytes_read=int(bg.nbytes + (0 if bu is bg else bu.nbytes) + ag.nbytes + au.nbytes + 2 * 512 * 4),
                    metadata_bytes=metadata_bytes,
                ),
            })
        applied = int(config["candidate_applied_units"])
        oracle_utility = max(float(exact_score[exact_order[:applied]].sum()), 1e-30)
        for candidates in map(int, config["candidate_unit_counts"]):
            fetched = predicted_order[:candidates]
            reranked = fetched[np.lexsort((fetched, -exact_score[fetched]))][:applied]
            output = output_for_units(base_output, corrections, reranked, applied)
            result["candidate_rerank_frontier"].append({
                **metadata, **config_fields,
                "objective": "exact_sequential_complete_expert_qenergy",
                "action_family": "predicted_candidate_units_exact_h0_rerank",
                "representation": "fetched_complete_unit_packets_apply_subset",
                "selector": "predicted_h4_candidates_exact_abc_rerank",
                "selection_regime": "deployable_candidate_prefetch_plus_h0_oracle_rerank",
                "selection_regime_category": "candidate_prefetch_h0_rerank",
                "physical_budget_bpw": 3 * candidates / PAGES_PER_BPW,
                "candidate_units": candidates,
                "applied_units": applied,
                "candidate_overfetch": candidates / applied,
                "oracle_utility_retention": float(exact_score[reranked].sum()) / oracle_utility,
                "oracle_support_recall": len(set(map(int, reranked)) & set(map(int, exact_order[:applied]))) / applied,
                **frontier_fields(
                    config, pages=3 * candidates, logical_actions=applied,
                    logical_bytes=applied * UNIT_PACKET_BYTES,
                    recovery=base.recovery(target, base_output, output, proxy, beta),
                    runtime_seconds=prediction_runtime,
                    selector_bytes_read=int(bg.nbytes + (0 if bu is bg else bu.nbytes) + ag.nbytes + au.nbytes + candidates * UNIT_PACKET_BYTES),
                    metadata_bytes=metadata_bytes,
                ),
            })
    return result


def promotion_config_ids(path: Path, config: Mapping[str, Any], fit_hash: str) -> tuple[set[str], bool]:
    return validate_promotion_payload(
        json.loads(path.read_text()), config,
        config_sha256=sha256(Path(config["_config_path"])),
        fit_bundle_sha256=fit_hash,
    )


def evaluate_phase(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if args.split not in {"validation", "test"}:
        raise RuntimeError("evaluation split must be validation or test")
    if (args.split == "test") != (args.promotions is not None):
        raise RuntimeError("test requires promotions and validation forbids them")
    args.output.mkdir(parents=True, exist_ok=True)
    audit, hashes = base.verify_locked_inputs(
        config, args.config, args.captures, args.checkpoint, args.trees, args.source,
    )
    comparator_hashes = verify_pr7_comparators(config)
    data = load_capture(args.captures)
    bundle_manifest_path = args.fit_dir / "response_fit_bundle_manifest.json"
    index_path = args.fit_dir / "response_fit_index.json"
    fit_facts_path = args.fit_dir / "fit_facts.json"
    fit_facts = json.loads(fit_facts_path.read_text())
    if not fit_facts.get("completed") or fit_facts.get("validation_or_test_rows_used") is not False:
        raise RuntimeError("fit facts are incomplete or permit non-training fit rows")
    bundle_digest = sha256(bundle_manifest_path)
    if fit_facts.get("bundle_sha256") != bundle_digest or fit_facts.get("index_sha256") != sha256(index_path):
        raise RuntimeError("fit bundle manifest/index hash differs from fit facts")
    bundle_manifest = json.loads(bundle_manifest_path.read_text())
    if bundle_manifest.get("config_sha256") != sha256(args.config):
        raise RuntimeError("fit bundle manifest config changed")
    if set(bundle_manifest.get("layers", {})) != set(map(str, config["layers"])):
        raise RuntimeError("fit bundle manifest does not contain every configured layer")
    fit_index = json.loads(index_path.read_text())
    if fit_index.get("fit_split") != "train" or fit_index.get("validation_or_test_rows_used") is not False:
        raise RuntimeError("fit index is not training-only")
    entries = list(fit_index["entries"])
    promote_factorized = True
    promotion_sha = None
    if args.promotions is not None:
        config["_config_path"] = str(args.config)
        selected_ids, promote_factorized = promotion_config_ids(
            args.promotions, config, bundle_digest,
        )
        entries = [entry for entry in entries if str(entry["config_id"]) in selected_ids]
        if not entries and selected_ids:
            raise RuntimeError("promoted response configurations are absent from fit index")
        promotion_sha = sha256(args.promotions)
    loaded_layer: int | None = None
    bundle: dict[str, np.ndarray] = {}
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    tree_records = json.loads(args.trees.read_text())
    trees = {projection: tree_from_record(tree_records[projection]) for projection in PROJECTIONS}
    device = torch.device(args.device)
    facts_path = args.output / "run_facts.json"
    facts = {
        "completed": False,
        "failures": [],
        "phase": "evaluate",
        "run_id": config["run_id"],
        "source": args.source,
        "evaluation_split": args.split,
        "config_sha256": sha256(args.config),
        "capture_sha256": hashes["capture_sha256"],
        "tree_sha256": hashes["tree_sha256"],
        "checkpoint_config_sha256": hashes["config_sha256"],
        "checkpoint_index_sha256": hashes["index_sha256"],
        "checkpoint_audit": audit,
        "fit_bundle_sha256": bundle_digest,
        "fit_index_sha256": sha256(index_path),
        "validation_promotions_sha256": promotion_sha,
        "pr7_comparator_sha256": comparator_hashes,
        "request_separation_verified": True,
        "codec_locked": True,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "host": platform.node(),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "selector_core_sha256": sha256(EXPERIMENT / "src/oracle_study/neuron_selector.py"),
        "base_runner_sha256": sha256(EXPERIMENT / "scripts/run_sparse_streaming_study.py"),
    }
    resume_keys = {key: facts[key] for key in (
        "run_id", "source", "evaluation_split", "config_sha256", "capture_sha256",
        "tree_sha256", "checkpoint_config_sha256", "checkpoint_index_sha256",
        "fit_bundle_sha256", "fit_index_sha256", "validation_promotions_sha256",
        "runner_sha256", "selector_core_sha256", "base_runner_sha256",
    )}
    state, completed, history = load_checkpoint_state(args.output, resume_keys, args.split)
    facts["completed_work_units"] = completed
    facts["failure_history"] = history
    try:
        for layer, expert, stratum, records, ranks in evaluation_plan(data, config, args.source, args.split):
            work_unit = f"{args.split}:{layer}:{expert}"
            if work_unit in set(completed):
                continue
            if loaded_layer != layer:
                bundle = load_bundle_layer(args.fit_dir, bundle_manifest, layer)
                loaded_layer = layer
            local_data = layer_view(data, layer)
            proxy, proxy_facts = base.proxy_gradients(
                local_data["xplus"], [local_data[f"h{h}_router_logits"] for h in range(1, 5)],
                local_data["split"],
            )
            beta = float(proxy_facts["beta"])
            matrices = decode_expert(args.checkpoint, index, trees, layer, expert)
            layer_entries = [
                entry for entry in entries
                if int(entry["layer"]) == layer and expert in set(map(int, entry["experts"]))
            ]
            for record, router_rank in zip(records, ranks):
                metadata = metadata_for_record(
                    args.source, args.split, local_data, int(record), int(router_rank),
                    layer, expert, stratum,
                )
                rows = evaluate_invocation(
                    matrices=matrices, activation=np.asarray(local_data["x"][record], np.float32),
                    proxy=proxy, beta=beta, metadata=metadata, config=config, device=device,
                    entries=layer_entries, bundle=bundle,
                )
                if not promote_factorized:
                    rows["factorized_neuron_frontier"] = [
                        row for row in rows["factorized_neuron_frontier"]
                        if row["action_family"] == "coherent_complete_unit_packet"
                    ]
                for name in ARTIFACTS:
                    state[name].extend(rows[name])
                print(json.dumps({
                    "completed_invocation": {name: metadata[name] for name in IDENTITY},
                    "rows": {name: len(rows[name]) for name in ARTIFACTS},
                }, sort_keys=True), flush=True)
            completed.append(work_unit)
            for name, filename in ARTIFACTS.items():
                atomic_parquet(args.output / filename, state[name])
            facts["completed_work_units"] = completed
            facts["artifact_row_counts"] = {name: len(rows) for name, rows in state.items()}
            atomic_json(facts_path, facts)
            del matrices
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        identities = {
            tuple(row[column] for column in IDENTITY)
            for rows in state.values() for row in rows
        }
        expected = sum(len(records) for _, _, _, records, _ in evaluation_plan(data, config, args.source, args.split))
        if len(identities) != expected:
            raise RuntimeError(f"observed {len(identities)} invocations != expected {expected}")
        facts["expected_unique_invocations"] = expected
        facts["observed_unique_invocations"] = len(identities)
        facts["completed"] = True
        facts["completed_unix"] = time.time()
        atomic_json(facts_path, facts)
    except Exception as error:
        facts["failures"].append({"type": type(error).__name__, "message": str(error), "unix": time.time()})
        atomic_json(facts_path, facts)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("fit", "evaluate"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trees", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--exact-captures", type=Path)
    parser.add_argument("--cross-captures", type=Path)
    parser.add_argument("--captures", type=Path)
    parser.add_argument("--source", choices=("exact_checkpoint", "cross_reference"))
    parser.add_argument("--split", choices=("validation", "test"))
    parser.add_argument("--fit-dir", type=Path)
    parser.add_argument("--promotions", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    if int(config["page_size_bytes"]) != PAGE_BYTES or int(config["expert_weights"]) != EXPERT_WEIGHTS:
        raise RuntimeError("physical accounting constants changed")
    if args.phase == "fit":
        if args.exact_captures is None or args.cross_captures is None:
            raise RuntimeError("fit requires --exact-captures and --cross-captures")
        fit_phase(args, config)
    else:
        if args.captures is None or args.source is None or args.split is None or args.fit_dir is None:
            raise RuntimeError("evaluate requires captures, source, split and fit-dir")
        evaluate_phase(args, config)


if __name__ == "__main__":
    main()
