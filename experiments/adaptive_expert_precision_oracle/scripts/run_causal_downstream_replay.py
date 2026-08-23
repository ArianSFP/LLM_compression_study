#!/usr/bin/env python3
"""Audit and replay PR #13 all-layer selected precision states.

Phase ``audit-layer`` reconstructs the actual routed-MoE error vectors for one
layer and verifies them against the frozen PR #13 qenergy and page accounting.
Phase ``finalize-audit`` seals a complete forty-layer audit.  The expensive
Transformers replay is intentionally a separate phase so no causal claim can
be produced before reconstruction parity passes.
"""

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


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(EXPERIMENT / "scripts")]

from oracle_study.causal_replay import (  # noqa: E402
    canonical_states,
    projection_responses,
    reconstruct_group_from_responses,
)
from oracle_study.mxfp4_embed import load_compressed_mxfp4_expert  # noqa: E402
from run_mxfp4_selective_pages import tree_from_record  # noqa: E402


SCHEMA = "pr13_causal_reconstruction_audit_v1"
PROJECTIONS = ("gate", "up", "down")
LAYERS = tuple(range(40))
AUDIT_SHARD = "reconstruction_layer_{layer:02d}.npz"
AUDIT_SIDECAR = "reconstruction_layer_{layer:02d}.json"
AUDIT_FAILURE = "reconstruction_layer_{layer:02d}.failure.json"
AUDIT_MANIFEST = "reconstruction_manifest.json"
AUDIT_FACTS = "reconstruction_run_facts.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


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


def runtime_provenance() -> dict[str, Any]:
    result: dict[str, Any] = {
        "host": platform.node(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import torch
        result.update({
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })
    except Exception as error:  # pragma: no cover - provenance only
        result["torch_error"] = repr(error)
    return result


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("base_all_layer_commit") != "089deb4bd41eb71864779ced0cbb3db41e9dcb63":
        raise RuntimeError("causal replay is not pinned to commit 089deb4")
    if list(map(int, config.get("layers", []))) != list(LAYERS):
        raise RuntimeError("layer grid changed")
    if config.get("selection_split") != "validation":
        raise RuntimeError("selection split changed")
    if config.get("no_test_rows_admitted_or_used") is not True:
        raise RuntimeError("test-row exclusion changed")
    if list(map(int, config.get("mean_budget_pages_per_expert", []))) != [384, 576, 749, 768]:
        raise RuntimeError("operating-point grid changed")
    if config.get("factor_config_id") != "matrix_free_joint_rank8_int4_per_row_hadamard":
        raise RuntimeError("primary factor changed")
    if config.get("allocation_policy") != "pooled_router_square_column_generated":
        raise RuntimeError("primary allocation policy changed")
    if int(config.get("burst_cap_pages_per_expert", -1)) != 1536:
        raise RuntimeError("burst cap changed")
    if config.get("predicted_h4_mode_in_scope") is not False:
        raise RuntimeError("predicted-H4 mode must remain out of scope")


def verify_frozen_inputs(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, str]:
    paths = {
        "checkpoint_config_sha256": args.checkpoint / "config.json",
        "checkpoint_index_sha256": args.checkpoint / "model.safetensors.index.json",
        "capture_manifest_sha256": args.capture_dir / "capture_manifest.json",
        "selected_tree_sha256": args.trees,
        "expert_allocation_sha256": args.validation_dir / "average_rate_expert_allocation.parquet",
        "group_frontier_sha256": args.validation_dir / "average_rate_group_frontier.parquet",
        "average_rate_run_facts_sha256": args.validation_dir / "average_rate_run_facts.json",
        "factor_manifest_sha256": args.fit_dir / "average_rate_factor_manifest.json",
    }
    hashes = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        value = sha256(path)
        if value != str(config[name]):
            raise RuntimeError(f"frozen input changed for {name}: {value}")
        hashes[name] = value
    manifest = load_json(args.capture_dir / "capture_manifest.json")
    if manifest.get("schema") != config["capture_schema"]:
        raise RuntimeError("capture schema changed")
    if manifest.get("identity_grid_sha256") != config["capture_identity_grid_sha256"]:
        raise RuntimeError("capture identity grid changed")
    if manifest.get("layers") != list(LAYERS):
        raise RuntimeError("capture layer grid changed")
    return hashes


def _capture_layer(args: argparse.Namespace, manifest: Mapping[str, Any], layer: int) -> dict[str, np.ndarray]:
    record = manifest["splits"]["validation"][str(layer)]
    path = args.capture_dir / str(record["file"])
    if sha256(path) != record["sha256"]:
        raise RuntimeError(f"capture shard changed for layer {layer}")
    with np.load(path, allow_pickle=False) as loaded:
        result = {name: np.asarray(loaded[name]) for name in loaded.files}
    if len(result["layer"]) != int(args.config_data["expected_validation_groups_per_layer"]):
        raise RuntimeError("validation row count changed")
    if not np.all(result["layer"] == layer) or not np.all(result["split"].astype(str) == "validation"):
        raise RuntimeError("capture shard identity changed")
    return result


def _trees(path: Path) -> dict[str, Any]:
    records = load_json(path)
    if set(records) != set(PROJECTIONS):
        raise RuntimeError("selected-tree projection set changed")
    return {name: tree_from_record(records[name]) for name in PROJECTIONS}


def _decode_active_experts(
    checkpoint: Path,
    index: Mapping[str, str],
    trees: Mapping[str, Any],
    layer: int,
    experts: Sequence[int],
) -> dict[int, tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]]:
    result = {}
    for expert in experts:
        q2, q4 = [], []
        for projection in PROJECTIONS:
            tensor = load_compressed_mxfp4_expert(
                checkpoint, dict(index), layer, int(expert), projection,
            )
            q2.append(trees[projection].decode(tensor, 2))
            q4.append(trees[projection].decode(tensor, 4))
        result[int(expert)] = (tuple(q2), tuple(q4))
    return result


def _selected_frames(
    args: argparse.Namespace, config: Mapping[str, Any], layer: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    layer_dir = args.validation_dir / "layers"
    expert_path = layer_dir / f"average_rate_expert_allocation_layer_{layer}.parquet"
    group_path = layer_dir / f"average_rate_group_frontier_layer_{layer}.parquet"
    sidecar_path = layer_dir / f"average_rate_evaluation_layer_{layer}.json"
    sidecar = load_json(sidecar_path)
    if sidecar.get("expert_sha256") != sha256(expert_path):
        raise RuntimeError("per-layer expert allocation changed")
    if sidecar.get("group_sha256") != sha256(group_path):
        raise RuntimeError("per-layer group frontier changed")
    experts = pd.read_parquet(expert_path)
    groups = pd.read_parquet(group_path)
    rates = set(map(int, config["mean_budget_pages_per_expert"]))
    common = (
        (experts["factor_config_id"] == config["factor_config_id"])
        & (experts["allocation_policy"] == config["allocation_policy"])
        & (experts["burst_cap_pages_per_expert"] == int(config["burst_cap_pages_per_expert"]))
        & experts["mean_budget_pages_per_expert"].isin(rates)
    )
    experts = experts.loc[common].copy()
    common_group = (
        (groups["factor_config_id"] == config["factor_config_id"])
        & (groups["allocation_policy"] == config["allocation_policy"])
        & (groups["burst_cap_pages_per_expert"] == int(config["burst_cap_pages_per_expert"]))
        & groups["mean_budget_pages_per_expert"].isin(rates)
    )
    groups = groups.loc[common_group].copy()
    expected_groups = int(config["expected_validation_groups_per_layer"]) * len(rates)
    if len(groups) != expected_groups or len(experts) != 8 * expected_groups:
        raise RuntimeError(
            f"selected allocation grid changed: {len(groups)} groups, {len(experts)} experts"
        )
    return experts, groups, sidecar


def _group_key(request_id: str, position: int, rate: int) -> tuple[str, int, int]:
    return str(request_id), int(position), int(rate)


def _audit_paths(output: Path, layer: int) -> tuple[Path, Path, Path]:
    return (
        output / "layers" / AUDIT_SHARD.format(layer=layer),
        output / "layers" / AUDIT_SIDECAR.format(layer=layer),
        output / "layers" / AUDIT_FAILURE.format(layer=layer),
    )


def audit_layer(args: argparse.Namespace) -> None:
    config = args.config_data
    layer = int(args.layer)
    if layer not in LAYERS:
        raise RuntimeError("audit layer is outside range(40)")
    shard_path, sidecar_path, failure_path = _audit_paths(args.output, layer)
    if failure_path.exists():
        raise RuntimeError(f"prior failure must be reviewed: {failure_path}")
    if shard_path.exists() or sidecar_path.exists():
        if not shard_path.is_file() or not sidecar_path.is_file():
            raise RuntimeError("partial reconstruction artifact exists")
        sidecar = load_json(sidecar_path)
        if sidecar.get("sha256") != sha256(shard_path) or sidecar.get("completed") is not True:
            raise RuntimeError("resume artifact changed")
        return
    started = time.perf_counter()
    try:
        manifest = load_json(args.capture_dir / "capture_manifest.json")
        capture = _capture_layer(args, manifest, layer)
        expert_frame, group_frame, source_sidecar = _selected_frames(args, config, layer)
        factor_path = args.fit_dir / f"average_rate_factor_layer_{layer}.npz"
        factor_manifest = load_json(args.fit_dir / "average_rate_factor_manifest.json")
        factor_record = factor_manifest["layers"][str(layer)]
        if factor_record["sha256"] != sha256(factor_path):
            raise RuntimeError("factor layer changed")
        with np.load(factor_path, allow_pickle=False) as factor:
            proxy = np.asarray(factor["proxy"], np.float64)
            beta = float(np.asarray(factor["beta"]).reshape(-1)[0])
        rows_by_identity = {
            (str(request), int(position)): index
            for index, (request, position) in enumerate(zip(capture["request_id"], capture["position"]))
        }
        if len(rows_by_identity) != len(capture["position"]):
            raise RuntimeError("duplicate capture identity")
        expert_groups = {
            _group_key(key[0], key[1], key[2]): value.sort_values("router_rank")
            for key, value in expert_frame.groupby(
                ["request_id", "position", "mean_budget_pages_per_expert"], sort=False,
            )
        }
        group_rows = {
            _group_key(row.request_id, row.position, row.mean_budget_pages_per_expert): row
            for row in group_frame.itertuples(index=False)
        }
        active = sorted(set(map(int, expert_frame["expert_id"].tolist())))
        index = load_json(args.checkpoint / "model.safetensors.index.json")["weight_map"]
        decoded = _decode_active_experts(
            args.checkpoint, index, _trees(args.trees), layer, active,
        )
        rates = np.asarray(config["mean_budget_pages_per_expert"], np.int64)
        row_order = sorted(rows_by_identity, key=lambda key: (key[0], key[1]))
        deltas = np.empty((len(rates), len(row_order), int(config["hidden_size"])), np.float32)
        reconstructed_damage = np.empty((len(rates), len(row_order)), np.float64)
        stored_damage = np.empty_like(reconstructed_damage)
        page_vectors = np.empty((len(rates), len(row_order), 8), np.int64)
        expert_ids_out = np.empty((len(row_order), 8), np.int64)
        router_weights_out = np.empty((len(row_order), 8), np.float64)
        expert_abs_errors: list[float] = []
        group_abs_errors: list[float] = []
        q4_errors: list[float] = []
        for row_index, identity in enumerate(row_order):
            capture_index = rows_by_identity[identity]
            expert_ids = np.asarray(capture["expert_ids"][capture_index], np.int64)
            router_weights = np.asarray(capture["router_weights"][capture_index], np.float64)
            responses = projection_responses(
                capture["x"][capture_index], expert_ids, decoded,
            )
            expert_ids_out[row_index] = expert_ids
            router_weights_out[row_index] = router_weights
            for rate_index, rate in enumerate(rates.tolist()):
                key = _group_key(identity[0], identity[1], rate)
                allocations = expert_groups[key]
                if not np.array_equal(allocations["expert_id"].to_numpy(np.int64), expert_ids):
                    raise RuntimeError("allocation expert order changed")
                if not np.allclose(
                    allocations["router_weight"].to_numpy(np.float64), router_weights,
                    rtol=0.0, atol=5e-8,
                ):
                    raise RuntimeError("allocation router weights changed")
                states = [canonical_states(value) for value in allocations["selected_states"]]
                result = reconstruct_group_from_responses(
                    responses, router_weights, states, proxy, beta,
                )
                expected_pages = allocations["selected_pages"].to_numpy(np.int64)
                if not np.array_equal(result.selected_pages, expected_pages):
                    raise RuntimeError("selected page accounting changed")
                expected_expert = allocations["exact_qenergy_damage"].to_numpy(np.float64)
                difference = np.abs(result.expert_damages - expected_expert)
                expert_abs_errors.extend(difference.tolist())
                if not np.allclose(
                    result.expert_damages, expected_expert,
                    rtol=float(config["reconstruction_damage_rtol"]),
                    atol=float(config["reconstruction_damage_atol"]),
                ):
                    raise RuntimeError(f"expert qenergy parity failed for {key}")
                expected_group = float(group_rows[key].group_exact_qenergy_damage)
                group_error = abs(result.group_damage - expected_group)
                group_abs_errors.append(group_error)
                if not np.isclose(
                    result.group_damage, expected_group,
                    rtol=float(config["reconstruction_damage_rtol"]),
                    atol=float(config["reconstruction_damage_atol"]),
                ):
                    raise RuntimeError(f"group qenergy parity failed for {key}")
                if result.all_q4_max_abs_error > float(config["all_q4_max_abs_atol"]):
                    raise RuntimeError("all-Q4 control is not zero within tolerance")
                q4_errors.append(result.all_q4_max_abs_error)
                deltas[rate_index, row_index] = result.delta.astype(np.float32)
                reconstructed_damage[rate_index, row_index] = result.group_damage
                stored_damage[rate_index, row_index] = expected_group
                page_vectors[rate_index, row_index] = result.selected_pages
        arrays = {
            "mean_budget_pages_per_expert": rates,
            "request_id": np.asarray([key[0] for key in row_order]),
            "position": np.asarray([key[1] for key in row_order], np.int64),
            "layer": np.full(len(row_order), layer, np.int64),
            "expert_ids": expert_ids_out,
            "router_weights": router_weights_out.astype(np.float32),
            "selected_pages": page_vectors,
            "delta": deltas,
            "reconstructed_group_qenergy_damage": reconstructed_damage,
            "stored_group_qenergy_damage": stored_damage,
        }
        atomic_npz(shard_path, arrays)
        sidecar = {
            "completed": True,
            "schema": SCHEMA,
            "run_id": config["run_id"],
            "phase": "audit-reconstruction-layer",
            "layer": layer,
            "rows": len(row_order),
            "rates": rates.tolist(),
            "reconstructions": int(len(row_order) * len(rates)),
            "expert_comparisons": int(len(row_order) * len(rates) * 8),
            "active_experts": len(active),
            "max_expert_qenergy_abs_error": max(expert_abs_errors, default=0.0),
            "max_group_qenergy_abs_error": max(group_abs_errors, default=0.0),
            "max_all_q4_abs_error": max(q4_errors, default=0.0),
            "all_selected_page_vectors_exact": True,
            "test_scientific_rows_admitted_or_used": False,
            "file": shard_path.name,
            "bytes": shard_path.stat().st_size,
            "sha256": sha256(shard_path),
            "source_evaluation_sidecar_sha256": sha256(
                args.validation_dir / "layers" / f"average_rate_evaluation_layer_{layer}.json"
            ),
            "source_factor_sha256": factor_record["sha256"],
            "source_capture_sha256": manifest["splits"]["validation"][str(layer)]["sha256"],
            "wall_seconds": time.perf_counter() - started,
            "runtime_provenance": runtime_provenance(),
        }
        atomic_json(sidecar_path, sidecar)
    except Exception as error:
        atomic_json(failure_path, {
            "completed": False,
            "schema": SCHEMA,
            "phase": "audit-reconstruction-layer",
            "layer": layer,
            "error": {"type": type(error).__name__, "message": str(error)},
        })
        raise


def finalize_audit(args: argparse.Namespace) -> None:
    config = args.config_data
    input_hashes = verify_frozen_inputs(args, config)
    layers = {}
    total_rows = total_reconstructions = total_expert_comparisons = 0
    maximum_expert = maximum_group = maximum_q4 = 0.0
    for layer in LAYERS:
        shard, sidecar_path, failure = _audit_paths(args.output, layer)
        if failure.exists():
            raise RuntimeError(f"reconstruction failure is present: {failure}")
        if not shard.is_file() or not sidecar_path.is_file():
            raise RuntimeError(f"reconstruction layer {layer} is incomplete")
        sidecar = load_json(sidecar_path)
        if sidecar.get("completed") is not True or sidecar.get("sha256") != sha256(shard):
            raise RuntimeError(f"reconstruction layer {layer} changed")
        if sidecar.get("test_scientific_rows_admitted_or_used") is not False:
            raise RuntimeError("test-row exclusion changed")
        total_rows += int(sidecar["rows"])
        total_reconstructions += int(sidecar["reconstructions"])
        total_expert_comparisons += int(sidecar["expert_comparisons"])
        maximum_expert = max(maximum_expert, float(sidecar["max_expert_qenergy_abs_error"]))
        maximum_group = max(maximum_group, float(sidecar["max_group_qenergy_abs_error"]))
        maximum_q4 = max(maximum_q4, float(sidecar["max_all_q4_abs_error"]))
        layers[str(layer)] = {
            "file": str(Path("layers") / shard.name),
            "bytes": shard.stat().st_size,
            "sha256": sha256(shard),
            "sidecar": str(Path("layers") / sidecar_path.name),
            "sidecar_sha256": sha256(sidecar_path),
        }
    if total_rows != int(config["expected_validation_groups"]):
        raise RuntimeError("final reconstruction row count changed")
    expected_reconstructions = total_rows * len(config["mean_budget_pages_per_expert"])
    if total_reconstructions != expected_reconstructions:
        raise RuntimeError("final reconstruction operating-point grid changed")
    manifest = {
        "completed": True,
        "schema": SCHEMA,
        "run_id": config["run_id"],
        "phase": "audit-reconstruction",
        "layers": layers,
        "input_hashes": input_hashes,
        "config_sha256": sha256(args.config),
    }
    manifest_path = args.output / AUDIT_MANIFEST
    atomic_json(manifest_path, manifest)
    facts = {
        "completed": True,
        "schema": SCHEMA,
        "run_id": config["run_id"],
        "phase": "audit-reconstruction",
        "layers": list(LAYERS),
        "validation_rows": total_rows,
        "reconstructions": total_reconstructions,
        "expert_comparisons": total_expert_comparisons,
        "max_expert_qenergy_abs_error": maximum_expert,
        "max_group_qenergy_abs_error": maximum_group,
        "max_all_q4_abs_error": maximum_q4,
        "all_selected_page_vectors_exact": True,
        "test_scientific_rows_admitted_or_used": False,
        "causal_replay_admitted": True,
        "manifest_sha256": sha256(manifest_path),
        "runtime_provenance": runtime_provenance(),
        **input_hashes,
    }
    atomic_json(args.output / AUDIT_FACTS, facts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("audit-layer", "finalize-audit"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--trees", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int)
    args = parser.parse_args()
    args.config_data = load_json(args.config)
    validate_config(args.config_data)
    if args.phase == "audit-layer" and args.layer is None:
        parser.error("audit-layer requires --layer")
    return args


def main() -> None:
    args = parse_args()
    if args.phase == "audit-layer":
        audit_layer(args)
    else:
        finalize_audit(args)


if __name__ == "__main__":
    main()
