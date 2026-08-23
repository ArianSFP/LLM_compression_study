#!/usr/bin/env python3
"""CPU parity audit for the same-host runner's frozen PR #13 selector path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pandas as pd


EXPERIMENT = Path(__file__).resolve().parents[1]

from run_causal_downstream_replay import load_json, sha256
from run_same_host_causal_controls import (
    SCHEMA,
    _select_layer,
)


AUDIT_SCHEMA = "pr13_same_host_selector_historical_parity_v1"
AUDIT_FACTS = "historical_selector_parity_facts.json"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _historical_captures(
    capture_dir: Path,
    manifest: Mapping[str, Any],
    layer: int,
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    record = manifest["splits"]["validation"][str(layer)]
    path = capture_dir / str(record["file"])
    if sha256(path) != record["sha256"]:
        raise RuntimeError("historical validation capture shard changed")
    with np.load(path, allow_pickle=False) as loaded:
        shard = {name: np.asarray(loaded[name]) for name in loaded.files}
    captures = {}
    excluded = int(config["exclude_final_positions"])
    for request_value in config["validation_request_ids"]:
        request_id = str(request_value)
        mask = np.asarray(shard["request_id"]).astype(str) == request_id
        positions = np.asarray(shard["position"][mask], np.int64)
        if positions.size == 0 or not np.array_equal(
            positions, np.arange(positions.size, dtype=np.int64),
        ):
            raise RuntimeError("historical request positions are not contiguous")
        sequence_length = int(positions[-1]) + excluded + 1
        router_ids = np.full((40, sequence_length, 8), -1, np.int64)
        router_scores = np.zeros((40, sequence_length, 8), np.float32)
        activation = np.zeros((sequence_length, 2048), np.float32)
        router_ids[layer, positions] = np.asarray(
            shard["expert_ids"][mask], np.int64,
        )
        router_scores[layer, positions] = np.asarray(
            shard["router_weights"][mask], np.float32,
        )
        activation[positions] = np.asarray(shard["x"][mask], np.float32)
        captures[request_id] = {
            "input_ids": np.zeros(sequence_length, np.int64),
            "router_ids": router_ids,
            "router_scores": router_scores,
            "x": {layer: activation},
        }
    if sum(
        len(capture["input_ids"]) - excluded for capture in captures.values()
    ) != int(config["expected_groups_per_layer"]):
        raise RuntimeError("historical selector audit group count changed")
    return captures


def _expected_allocation(
    validation_dir: Path,
    layer: int,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    path = validation_dir / "average_rate_expert_allocation.parquet"
    columns = [
        "layer", "request_id", "position", "mean_budget_pages_per_expert",
        "burst_cap_pages_per_expert", "factor_config_id", "allocation_policy",
        "router_rank", "expert_id", "router_weight", "selected_pages",
        "selected_states", "exact_qenergy_damage",
    ]
    frame = pd.read_parquet(
        path,
        columns=columns,
        filters=[("layer", "==", int(layer))],
    )
    common = (
        frame["factor_config_id"].eq(config["factor_config_id"])
        & frame["allocation_policy"].eq(config["allocation_policy"])
        & frame["burst_cap_pages_per_expert"].eq(
            int(config["burst_cap_pages_per_expert"]),
        )
        & frame["mean_budget_pages_per_expert"].isin(
            list(map(int, config["mean_budget_pages_per_expert"])),
        )
    )
    return frame.loc[common].sort_values(
        [
            "request_id", "position", "mean_budget_pages_per_expert",
            "router_rank",
        ],
        kind="stable",
    ).reset_index(drop=True)


def _compare_allocation(
    candidate: pd.DataFrame,
    expected: pd.DataFrame,
) -> dict[str, Any]:
    candidate = candidate.sort_values(
        [
            "request_id", "position", "mean_budget_pages_per_expert",
            "router_rank",
        ],
        kind="stable",
    ).reset_index(drop=True)
    if len(candidate) != len(expected):
        raise RuntimeError("historical allocation row count changed")
    exact_fields = [
        "request_id", "position", "mean_budget_pages_per_expert",
        "router_rank", "expert_id", "selected_pages", "selected_states",
    ]
    for name in exact_fields:
        if not np.array_equal(
            candidate[name].to_numpy(), expected[name].to_numpy(),
        ):
            raise RuntimeError(f"historical selected allocation changed: {name}")
    float_fields = ["router_weight", "exact_qenergy_damage"]
    maxima = {}
    for name in float_fields:
        error = np.abs(
            candidate[name].to_numpy(np.float64)
            - expected[name].to_numpy(np.float64)
        )
        maxima[name] = float(np.max(error, initial=0.0))
        if not np.allclose(
            candidate[name].to_numpy(np.float64),
            expected[name].to_numpy(np.float64),
            rtol=1e-9,
            atol=1e-10,
        ):
            raise RuntimeError(f"historical selected allocation float changed: {name}")
    return {
        "rows": len(candidate),
        "exact_fields": exact_fields,
        "max_float_abs_error": maxima,
    }


def _compare_reconstruction(
    selector: Any,
    reconstruction_dir: Path,
    manifest: Mapping[str, Any],
    layer: int,
) -> dict[str, Any]:
    record = manifest["layers"][str(layer)]
    path = reconstruction_dir / str(record["file"])
    if sha256(path) != record["sha256"]:
        raise RuntimeError("historical reconstruction shard changed")
    with np.load(path, allow_pickle=False) as loaded:
        rates = np.asarray(loaded["mean_budget_pages_per_expert"], np.int64)
        request_ids = np.asarray(loaded["request_id"]).astype(str)
        positions = np.asarray(loaded["position"], np.int64)
        delta = np.asarray(loaded["delta"], np.float32)
    if not np.array_equal(rates, selector.rates):
        raise RuntimeError("historical reconstruction rate order changed")
    if not np.array_equal(request_ids, selector.request_ids.astype(str)):
        raise RuntimeError("historical reconstruction request order changed")
    if not np.array_equal(positions, selector.positions):
        raise RuntimeError("historical reconstruction position order changed")
    difference = np.abs(
        delta.astype(np.float64) - selector.deltas.astype(np.float64),
    )
    maximum = float(np.max(difference, initial=0.0))
    if maximum != 0.0:
        raise RuntimeError("historical reconstructed delta is not bit-identical")
    return {
        "reconstructions": int(delta.shape[0] * delta.shape[1]),
        "max_delta_abs_error": maximum,
        "bit_identical": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pr13-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trees", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--reconstruction-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selector-workers", type=int, default=24)
    parser.add_argument("--layers", type=int, nargs="*")
    args = parser.parse_args()
    config = load_json(args.config)
    pr13_config = load_json(args.pr13_config)
    if sha256(args.checkpoint / "config.json") != config["checkpoint_config_sha256"]:
        raise RuntimeError("checkpoint config changed")
    if sha256(args.checkpoint / "model.safetensors.index.json") != config["checkpoint_index_sha256"]:
        raise RuntimeError("checkpoint index changed")
    if sha256(args.trees) != config["selected_tree_sha256"]:
        raise RuntimeError("selected tree changed")
    if sha256(args.fit_dir / "average_rate_factor_manifest.json") != config["factor_manifest_sha256"]:
        raise RuntimeError("factor manifest changed")
    capture_manifest_path = args.capture_dir / "capture_manifest.json"
    if sha256(capture_manifest_path) != config["historical_capture_manifest_sha256"]:
        raise RuntimeError("historical capture manifest changed")
    allocation_path = args.validation_dir / "average_rate_expert_allocation.parquet"
    if sha256(allocation_path) != config["historical_expert_allocation_sha256"]:
        raise RuntimeError("historical expert allocation changed")
    reconstruction_manifest_path = args.reconstruction_dir / "reconstruction_manifest.json"
    if sha256(reconstruction_manifest_path) != config["historical_reconstruction_manifest_sha256"]:
        raise RuntimeError("historical reconstruction manifest changed")
    capture_manifest = load_json(capture_manifest_path)
    reconstruction_manifest = load_json(reconstruction_manifest_path)
    selected_layers = list(
        map(int, args.layers if args.layers else config["sentinel_layers"]),
    )
    index = load_json(
        args.checkpoint / "model.safetensors.index.json",
    )["weight_map"]
    tree_records = load_json(args.trees)
    runner_args = SimpleNamespace(
        config_data=config,
        pr13_config_data=pr13_config,
        checkpoint=args.checkpoint,
        trees=args.trees,
        fit_dir=args.fit_dir,
        selector_workers=args.selector_workers,
    )
    results = {}
    for layer in selected_layers:
        captures = _historical_captures(
            args.capture_dir, capture_manifest, layer, config,
        )
        selector = _select_layer(
            runner_args, captures, layer, index, tree_records,
        )
        allocation = _compare_allocation(
            selector.allocation,
            _expected_allocation(args.validation_dir, layer, config),
        )
        reconstruction = _compare_reconstruction(
            selector, args.reconstruction_dir, reconstruction_manifest, layer,
        )
        results[str(layer)] = {
            "allocation": allocation,
            "reconstruction": reconstruction,
            "max_group_qenergy_abs_error": selector.max_group_damage_error,
            "all_q4_max_abs_error": selector.all_q4_max_abs_error,
            "active_experts": selector.active_experts,
        }
        atomic_json(args.output / f"historical_selector_parity_layer_{layer:02d}.json", {
            "completed": True,
            "schema": AUDIT_SCHEMA,
            "layer": layer,
            **results[str(layer)],
        })
    atomic_json(args.output / AUDIT_FACTS, {
        "completed": True,
        "schema": AUDIT_SCHEMA,
        "same_host_schema_under_audit": SCHEMA,
        "run_id": config["run_id"],
        "layers": selected_layers,
        "results": results,
        "historical_selected_states_bit_identical": True,
        "historical_reconstructed_deltas_bit_identical": True,
        "test_rows_admitted_or_used": False,
        "runtime": {
            "host": platform.node(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "selector_workers": args.selector_workers,
        },
    })


if __name__ == "__main__":
    main()
