#!/usr/bin/env python3
"""Validation-only alternate tile-shape continuation and pilot augmenter."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
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

import run_sparse_streaming_study as base
from oracle_study.mxfp4_embed import load_compressed_mxfp4_expert
from oracle_study.sparse_streaming_cuda import (
    MixedPageGeometry,
    SelectionTrace,
    exact_mixed_page_greedy,
    static_cartesian_mixed_page_order,
    static_proxy_mixed_page_order,
)
from run_mxfp4_selective_pages import tree_from_record
from run_phase_a_remote import proxy_gradients

SOURCE = "exact_checkpoint"
SPLIT = "validation"
FOLLOWUP_FACTS = "tile_shape_followup_facts.json"
FOLLOWUP_ROWS = "alternate_tile_shape_validation_frontier.parquet"


@dataclass(frozen=True)
class FollowupSpec:
    predeclared_selector: str
    selector: str
    tile_shape: tuple[int, int]
    refresh_interval: int

    @property
    def identity(self) -> tuple[str, str]:
        return f"{self.tile_shape[0]}x{self.tile_shape[1]}", self.selector


# Exact/WINA 32x32 validation rows are reused from the immutable parent.
FOLLOWUP_SPECS = (
    FollowupSpec("exact_dynamic_tile_marginal", "exact_dynamic_tile_marginal_refresh_1", (16, 64), 1),
    FollowupSpec("activation_energy_x_weight", "activation_energy_x_weight", (16, 64), 0),
    FollowupSpec("cartesian_hidden_input_blocks", "cartesian_hidden_input_blocks", (16, 64), 0),
    FollowupSpec("cartesian_hidden_input_blocks", "cartesian_hidden_input_blocks", (32, 32), 0),
    FollowupSpec("exact_dynamic_tile_marginal", "exact_dynamic_tile_marginal_refresh_1", (64, 16), 1),
    FollowupSpec("activation_energy_x_weight", "activation_energy_x_weight", (64, 16), 0),
    FollowupSpec("cartesian_hidden_input_blocks", "cartesian_hidden_input_blocks", (64, 16), 0),
)
REUSED_32_SELECTORS = {
    "exact_dynamic_tile_marginal": "exact_dynamic_tile_marginal_refresh_1",
    "activation_energy_x_weight": "static_wina",
}


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    return hashlib.sha256(text.encode()).hexdigest()


def identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row[name] for name in base.IDENTITY_COLUMNS)


def validate_design(config: Mapping[str, Any]) -> None:
    base.validate_config_controls(dict(config))
    shapes = {tuple(map(int, value)) for value in config["conditional_followup_tile_shapes"]}
    if shapes != {(16, 64), (32, 32), (64, 16)}:
        raise RuntimeError(f"conditional tile shapes changed: {shapes}")
    selectors = set(map(str, config["conditional_followup_tile_selectors"]))
    expected = {
        "exact_dynamic_tile_marginal", "activation_energy_x_weight",
        "cartesian_hidden_input_blocks",
    }
    if selectors != expected:
        raise RuntimeError(f"conditional tile selectors changed: {selectors}")
    policy = config["promotion_policy"]
    if policy.get("selection_split") != SPLIT or policy.get("no_test_tuning") is not True:
        raise RuntimeError("follow-up requires validation selection and no_test_tuning=true")


def validate_trigger(
    promotions: Mapping[str, Any], config: Mapping[str, Any], hashes: Mapping[str, str],
) -> dict[str, Any]:
    plan = base.validate_and_parse_promotions(dict(promotions), dict(config), dict(hashes), SOURCE)
    tile = plan["tile_streaming"]
    selected = tile.get("selection")
    if not tile.get("enabled") or not isinstance(selected, Mapping):
        raise RuntimeError("validation did not trigger alternate tile shapes")
    if tuple(map(int, selected.get("tile_shape", ()))) != (32, 32):
        raise RuntimeError("alternate shapes require a promoted 32x32 pilot")
    return dict(selected)


def selector_trace(
    spec: FollowupSpec,
    matrices: Mapping[str, Sequence[np.ndarray]],
    activation: np.ndarray,
    proxy: np.ndarray,
    beta: float,
    page_budgets: Sequence[int],
    device: torch.device,
) -> SelectionTrace:
    arguments = (
        matrices["gate"][0], matrices["up"][0], matrices["down"][0],
        matrices["gate"][2], matrices["up"][2], matrices["down"][2], activation,
    )
    keywords = {
        "tile_shape": spec.tile_shape,
        "page_budgets": page_budgets,
        "proxy": proxy,
        "beta": beta,
        "device": device,
    }
    if spec.predeclared_selector == "exact_dynamic_tile_marginal":
        return exact_mixed_page_greedy(
            *arguments, refresh_interval=spec.refresh_interval, **keywords,
        )
    if spec.predeclared_selector == "activation_energy_x_weight":
        return static_proxy_mixed_page_order(
            *arguments, proxy_method="wina", **keywords,
        )
    if spec.predeclared_selector == "cartesian_hidden_input_blocks":
        return static_cartesian_mixed_page_order(*arguments, **keywords)
    raise RuntimeError(f"unsupported follow-up selector: {spec.predeclared_selector}")


def followup_rows(
    matrices: Mapping[str, Sequence[np.ndarray]],
    activation: np.ndarray,
    proxy: np.ndarray,
    beta: float,
    metadata: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    budgets_bpw = list(map(float, config["physical_budgets_bpw"]))
    budgets = [base.physical_page_budget(value) for value in budgets_bpw]
    requested = sorted(set(budgets))
    base_output, target = base.decoded_complete_outputs(dict(matrices), activation)
    geometry_by_shape = {
        shape: MixedPageGeometry(512, 2048, *shape)
        for shape in {(16, 64), (32, 32), (64, 16)}
    }

    # Prove each alternate packetization reaches the same immutable Q4 endpoint.
    for shape in ((16, 64), (64, 16)):
        geometry = geometry_by_shape[shape]
        endpoint = static_proxy_mixed_page_order(
            matrices["gate"][0], matrices["up"][0], matrices["down"][0],
            matrices["gate"][2], matrices["up"][2], matrices["down"][2], activation,
            tile_shape=shape, page_budgets=[geometry.total_pages],
            proxy_method="wina", proxy=proxy, beta=beta, device=device,
        )
        base.assert_complete_endpoint(endpoint, geometry.total_pages, target, f"follow-up tile {shape}")

    base_storage = base.storage_fields(dict(config))
    selector_input_bytes = int(
        sum(matrices[projection][level].nbytes for projection in base.PROJECTIONS for level in (0, 2))
        + activation.nbytes
    )
    rows: list[dict[str, Any]] = []
    for spec in FOLLOWUP_SPECS:
        geometry = geometry_by_shape[spec.tile_shape]
        base.synchronize(device)
        started = time.perf_counter()
        trace = selector_trace(spec, matrices, activation, proxy, beta, requested, device)
        base.synchronize(device)
        elapsed = time.perf_counter() - started
        base.assert_unique_trace(trace, geometry.total_pages)
        regime = (
            "h0_oracle_full_suffix"
            if spec.predeclared_selector == "exact_dynamic_tile_marginal"
            else "h0_oracle_suffix_metadata_heavy"
        )
        for budget_bpw, budget_pages in zip(budgets_bpw, budgets):
            output = trace.snapshots[budget_pages].numpy()
            chosen = np.asarray(trace.order[:budget_pages], np.int64)
            rows.append({
                **metadata,
                "objective": "exact_sequential_complete_expert_qenergy",
                "selection_regime": regime,
                "selection_regime_category": "h0_oracle",
                "action_family": "gate_up_tile_plus_down_column",
                "representation": "paired_direct_q2_q4_gate_up_tile_plus_down_column",
                "tile_shape": f"{spec.tile_shape[0]}x{spec.tile_shape[1]}",
                "selector": spec.selector,
                "refresh_interval": int(spec.refresh_interval),
                "global_refreshes": (
                    int(np.ceil(budget_pages / max(spec.refresh_interval, 1)))
                    if spec.refresh_interval else 1
                ),
                "tile_pages_selected": int(np.sum(chosen < geometry.tile_pages)),
                "down_pages_selected": int(np.sum(chosen >= geometry.tile_pages)),
                "conditional_followup_selector": spec.predeclared_selector,
                **base_storage,
                **base.standard_frontier_fields(
                    physical_budget_bpw=budget_bpw,
                    pages=budget_pages,
                    logical_actions=budget_pages,
                    recovery_value=base.recovery(target, base_output, output, proxy, beta),
                    runtime_seconds=elapsed,
                    selector_bytes_read=selector_input_bytes,
                    multiplier=float(base_storage["storage_multiplier"]),
                    logical_payload_bytes=budget_pages * base.PAGE_BYTES,
                ),
            })
    return rows


def parent_validation_identities(
    parent: Path, hashes: Mapping[str, str], config: Mapping[str, Any],
) -> set[tuple[Any, ...]]:
    facts = json.loads((parent / "run_facts.json").read_text())
    if (
        facts.get("completed") is not True
        or facts.get("failures") not in (None, [])
        or facts.get("run_mode") != "pilot"
        or facts.get("source") != SOURCE
        or facts.get("input_hashes") != dict(hashes)
        or facts.get("codec_locked") is not True
        or facts.get("request_separation_verified") is not True
        or facts.get("config_sha256") != hashes["config_file_sha256"]
        or facts.get("locked_tree_sha256") != config["locked_tree_sha256"]
        or facts.get("locked_capture_sha256") != config["locked_capture_sha256"]
    ):
        raise RuntimeError("parent pilot provenance is incomplete or differs from locked inputs")
    validation = pd.read_parquet(
        parent / base.ARTIFACT_FILES["tile_streaming_frontier"],
        filters=[("evaluation_split", "==", SPLIT)],
    )
    if set(validation["evaluation_split"].astype(str).unique()) != {SPLIT}:
        raise RuntimeError("predicate-pushed parent read exposed a non-validation row")
    expected_controls = (
        ("32x32", "exact_dynamic_tile_marginal_refresh_1"),
        ("32x32", "static_wina"),
        ("input_coordinate_2columns_per_page", "static_first_order_input_coordinate_baseline"),
    )
    identities: set[tuple[Any, ...]] | None = None
    budgets = len(config["physical_budgets_bpw"])
    for shape, selector in expected_controls:
        values = validation[
            (validation["tile_shape"].astype(str) == shape)
            & (validation["selector"].astype(str) == selector)
        ]
        current = {
            tuple(row[name] for name in base.IDENTITY_COLUMNS)
            for row in values.to_dict("records")
        }
        if len(values) != len(current) * budgets or len(current) != 55:
            raise RuntimeError(f"parent control {shape}/{selector} lacks 55x{budgets} validation rows")
        if identities is None:
            identities = current
        elif current != identities:
            raise RuntimeError("parent validation controls disagree on invocation identities")
    if identities is None:
        raise RuntimeError("parent pilot contains no validation identities")
    return identities


def planned_validation(
    all_data: Mapping[str, np.ndarray], config: Mapping[str, Any],
) -> tuple[dict[int, list[tuple[int, str, np.ndarray, np.ndarray]]], set[tuple[Any, ...]]]:
    plans: dict[int, list[tuple[int, str, np.ndarray, np.ndarray]]] = {}
    identities: set[tuple[Any, ...]] = set()
    for layer in map(int, config["layers"]):
        mask = np.asarray(all_data["layer"]) == layer
        data = {name: np.asarray(value)[mask] for name, value in all_data.items()}
        layer_plan = base.layer_record_plan(data, layer, SPLIT, "pilot", SOURCE, dict(config))
        plans[layer] = layer_plan
        for expert, stratum, records, ranks in layer_plan:
            for record, rank in zip(records, ranks):
                meta = base.common_metadata(
                    SOURCE, SPLIT, data, int(record), int(rank), layer, expert, stratum,
                )
                identities.add(identity(meta))
    return plans, identities


def flush(
    output: Path,
    rows: list[dict[str, Any]],
    facts: dict[str, Any],
    *,
    final: bool,
    schema_parent: Path,
) -> None:
    base.atomic_parquet(
        output / base.ARTIFACT_FILES["tile_streaming_frontier"], rows,
        base.ARTIFACT_SCHEMAS["tile_streaming_frontier"],
    )
    base.atomic_parquet(output / FOLLOWUP_ROWS, rows, base.ARTIFACT_SCHEMAS["tile_streaming_frontier"])
    for name, filename in base.ARTIFACT_FILES.items():
        if name == "tile_streaming_frontier":
            continue
        # Preserve Arrow/pandas numeric dtypes from the immutable parent
        # while predicate-pushing a condition that returns zero rows.  Generic
        # object-typed empties would contaminate merged numeric frontiers.
        empty = pd.read_parquet(
            schema_parent / filename,
            filters=[("layer", "==", -1)],
        )
        if not empty.empty or not set(base.ARTIFACT_SCHEMAS[name]).issubset(empty.columns):
            raise RuntimeError(f"parent schema template failed for {filename}")
        temporary = (output / filename).with_suffix(".parquet.tmp")
        empty.to_parquet(temporary, index=False)
        temporary.replace(output / filename)
    empty_cache = pd.read_parquet(
        schema_parent / "_support_cache.parquet",
        filters=[("layer", "==", -1)],
    )
    if not empty_cache.empty:
        raise RuntimeError("parent support-cache schema template is not empty")
    cache_temporary = output / "_support_cache.parquet.tmp"
    empty_cache.to_parquet(cache_temporary, index=False)
    cache_temporary.replace(output / "_support_cache.parquet")
    facts["artifact_row_counts"] = {
        name: len(rows) if name == "tile_streaming_frontier" else 0
        for name in base.ARTIFACT_FILES
    }
    facts["observed_unique_invocations"] = len({identity(row) for row in rows})
    facts["updated_unix"] = time.time()
    facts["completed"] = bool(final)
    base.atomic_json(output / "run_facts.json", facts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trees", type=Path, required=True)
    parser.add_argument("--parent-pilot", type=Path, required=True)
    parser.add_argument("--trigger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    config = json.loads(args.config.read_text())
    validate_design(config)
    audit, hashes = base.verify_locked_inputs(
        config, args.config, args.captures, args.checkpoint, args.trees, SOURCE,
    )
    trigger_container = json.loads(args.trigger.read_text())
    promotions = trigger_container.get("promotions", trigger_container)
    selected = validate_trigger(promotions, config, hashes)
    parent_ids = parent_validation_identities(args.parent_pilot, hashes, config)
    loaded = np.load(args.captures, allow_pickle=False)
    all_data = {name: loaded[name] for name in loaded.files}
    request_counts = base.verify_request_separation(all_data)
    plans, planned_ids = planned_validation(all_data, config)
    if planned_ids != parent_ids:
        raise RuntimeError("continuation validation identities differ from immutable parent pilot")

    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    records = json.loads(args.trees.read_text())
    trees = {projection: tree_from_record(records[projection]) for projection in base.PROJECTIONS}
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    facts_path = args.output / "run_facts.json"
    trigger_sha = canonical_json_sha256(promotions)
    facts: dict[str, Any] = {
        "run_id": config["run_id"], "run_mode": "pilot",
        "mode": "validation_tile_shape_followup", "source": SOURCE,
        "started_unix": time.time(), "completed": False, "failures": [],
        "codec_locked": True, "request_separation_verified": True,
        "request_split_policy": "validation-only conditional continuation; no test rows",
        "page_size_bytes": int(config["page_size_bytes"]),
        "reference": {
            "revision": config["reference"]["revision"],
            "config_sha256": config["reference"]["config_sha256"],
            "index_sha256": config["reference"]["index_sha256"],
        },
        "locked_tree_sha256": config["locked_tree_sha256"],
        "locked_capture_sha256": config["locked_capture_sha256"],
        "config_sha256": hashes["config_file_sha256"], "input_hashes": hashes,
        "checkpoint_audit": audit, "request_counts_by_split": request_counts,
        "expected_unique_invocations": len(planned_ids),
        "expected_cohort_counts": {SPLIT: len(planned_ids)},
        "runner_sha256": base.sha256(Path(__file__).resolve()),
        "base_runner_sha256": base.sha256(Path(base.__file__).resolve()),
        "selector_core_sha256": base.sha256(
            EXPERIMENT / "src" / "oracle_study" / "sparse_streaming_cuda.py"
        ),
        "parent_pilot_run_facts_sha256": base.sha256(args.parent_pilot / "run_facts.json"),
        "parent_pilot_tile_sha256": base.sha256(
            args.parent_pilot / base.ARTIFACT_FILES["tile_streaming_frontier"]
        ),
        "trigger_validation_promotions_sha256": trigger_sha,
        "trigger_selected_tile": selected,
        "conditional_followup_shapes": [[16, 64], [32, 32], [64, 16]],
        "conditional_followup_selectors": sorted({spec.predeclared_selector for spec in FOLLOWUP_SPECS}),
        "test_rows_evaluated": 0,
        "test_rows_consulted_for_selection": False,
        "host": platform.node(), "torch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    rows: list[dict[str, Any]] = []
    completed: set[str] = set()
    if facts_path.exists():
        existing = json.loads(facts_path.read_text())
        expected_resume = {
            "config_sha256": hashes["config_file_sha256"], "input_hashes": hashes,
            "mode": "validation_tile_shape_followup", "source": SOURCE,
            "trigger_validation_promotions_sha256": trigger_sha,
            "runner_sha256": base.sha256(Path(__file__).resolve()),
            "base_runner_sha256": base.sha256(Path(base.__file__).resolve()),
            "selector_core_sha256": base.sha256(
                EXPERIMENT / "src" / "oracle_study" / "sparse_streaming_cuda.py"
            ),
            "parent_pilot_run_facts_sha256": base.sha256(
                args.parent_pilot / "run_facts.json"
            ),
            "parent_pilot_tile_sha256": base.sha256(
                args.parent_pilot / base.ARTIFACT_FILES["tile_streaming_frontier"]
            ),
        }
        if any(existing.get(name) != value for name, value in expected_resume.items()):
            raise RuntimeError("follow-up resume provenance differs from current locked inputs")
        if existing.get("completed") is True:
            return
        facts.update(existing)
        completed = set(map(str, facts.get("completed_work_units", [])))
        prior = base.load_rows(args.output / base.ARTIFACT_FILES["tile_streaming_frontier"])
        rows = [row for row in prior if f"validation:{int(row['layer'])}:{int(row['expert_id'])}" in completed]
        facts["resumed_unix"] = time.time()

    try:
        for layer in map(int, config["layers"]):
            mask = np.asarray(all_data["layer"]) == layer
            data = {name: np.asarray(value)[mask] for name, value in all_data.items()}
            proxy, proxy_facts = proxy_gradients(
                data["xplus"], [data[f"h{h}_router_logits"] for h in range(1, 5)], data["split"],
            )
            beta = float(proxy_facts["beta"])
            facts.setdefault("proxy_facts_by_layer", {})[str(layer)] = proxy_facts
            for expert, stratum, record_ids, ranks in plans[layer]:
                work_unit = f"validation:{layer}:{expert}"
                if work_unit in completed:
                    continue
                tensors = {
                    projection: load_compressed_mxfp4_expert(
                        args.checkpoint, index, layer, expert, projection,
                    )
                    for projection in base.PROJECTIONS
                }
                matrices = {
                    projection: [trees[projection].decode(tensors[projection], level) for level in (2, 3, 4)]
                    for projection in base.PROJECTIONS
                }
                local: list[dict[str, Any]] = []
                for record, rank in zip(record_ids, ranks):
                    activation = np.asarray(data["x"][record], np.float32)
                    metadata = base.common_metadata(
                        SOURCE, SPLIT, data, int(record), int(rank), layer, expert, stratum,
                    )
                    started = time.perf_counter()
                    local.extend(followup_rows(matrices, activation, proxy, beta, metadata, config, device))
                    print(json.dumps({
                        "completed_validation_followup_invocation": {
                            "layer": layer, "expert_id": expert,
                            "request_id": metadata["request_id"], "position": metadata["position"],
                        },
                        "elapsed_seconds": time.perf_counter() - started,
                        "rows_buffered": len(local),
                    }, sort_keys=True), flush=True)
                rows.extend(local)
                completed.add(work_unit)
                facts["completed_work_units"] = sorted(completed)
                facts["last_completed"] = {
                    "split": SPLIT, "layer": layer, "expert_id": expert, "unix": time.time(),
                }
                flush(
                    args.output, rows, facts, final=False,
                    schema_parent=args.parent_pilot,
                )
                del matrices, tensors
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        observed = {identity(row) for row in rows}
        expected_rows = len(planned_ids) * len(FOLLOWUP_SPECS) * len(config["physical_budgets_bpw"])
        if observed != planned_ids or len(rows) != expected_rows:
            raise RuntimeError(
                f"follow-up endpoint mismatch: identities={len(observed)}/{len(planned_ids)}, "
                f"rows={len(rows)}/{expected_rows}"
            )
        facts["observed_cohort_counts"] = {SPLIT: len(observed)}
        facts["completed_unix"] = time.time()
        flush(
            args.output, rows, facts, final=True,
            schema_parent=args.parent_pilot,
        )
    except Exception as error:
        facts.setdefault("failures", []).append({
            "unix": time.time(), "type": type(error).__name__, "message": str(error),
        })
        facts["completed"] = False
        base.atomic_json(facts_path, facts)
        raise


if __name__ == "__main__":
    main()
