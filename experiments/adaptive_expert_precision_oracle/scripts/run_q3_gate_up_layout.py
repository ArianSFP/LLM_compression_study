#!/usr/bin/env python3
"""Fit training-only Q3 plane layouts and evaluate grouped exact-H4 ceilings."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(EXPERIMENT / "scripts")]

from oracle_study.average_rate_allocator import (  # noqa: E402
    AllocationTrace, RateOption, exact_group_option_allocate,
    multiple_choice_allocate, qmetric_features,
)
from oracle_study.q3_gate_up_field import (  # noqa: E402
    Q3_INHERITED_EIGHT_STATE_MAP, Q3_STATE_DOWN_HIGH, Q3_STATE_GATE_LEVEL,
    Q3_STATE_UP_LEVEL, Q3InteractionField, Q3PageLayout,
    build_q3_interaction_field, fit_coselection_layouts,
    fixed_gate_up_layout, monolithic_q2q4_layout, q3_projection_responses,
    q3_state_output,
)
from oracle_study.q3_rate_allocator import (  # noqa: E402
    q3_allocation_aware_rate_frontiers, q3_rate_frontier,
    training_plane_incidence,
)
from oracle_study.neuron_selector import unit_score_metadata  # noqa: E402
from run_mxfp4_selective_pages import occurrence, tree_from_record  # noqa: E402
import run_average_rate_allocation as pr13  # noqa: E402
import run_set_utility_distillation as prior  # noqa: E402
import run_sparse_streaming_study as base  # noqa: E402


LAYOUT_FACTS = "q3_layout_fit_facts.json"
LAYOUT_MANIFEST = "q3_layout_manifest.json"
LAYOUT_LAYER = "q3_layout_layer_{layer}.npz"
LAYOUT_SIDECAR = "q3_layout_layer_{layer}.json"
RUN_FACTS = "q3_gate_up_run_facts.json"
GROUP_FRONTIER = "q3_gate_up_group_frontier.parquet"
EXPERT_ALLOCATION = "q3_gate_up_expert_allocation.parquet"
ACCOUNTING = "q3_gate_up_accounting.json"
GROUP_IDENTITY = (
    "capture_source", "evaluation_split", "request_id", "position", "layer",
)
PROJECTIONS = ("gate", "up", "down")
UNITS = 512
EXPERT_WEIGHTS = 3_145_728
QUANTUM_BYTES = 256
PAGE_BYTES = 512
BASE_EXACT_BPW = 4.25
_LAYOUT_CONTEXT: dict[str, Any] | None = None
_EVAL_CONTEXT: dict[str, Any] | None = None


def _sha256(path: Path) -> str:
    return prior.sha256(path)


def _bytes_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    prior.atomic_json(path, payload)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _atomic_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(list(rows)).to_parquet(temporary, index=False)
    temporary.replace(path)


def _dependencies() -> dict[str, str]:
    paths = {
        "runner_sha256": Path(__file__).resolve(),
        "q3_field_core_sha256": EXPERIMENT / "src/oracle_study/q3_gate_up_field.py",
        "q3_allocator_core_sha256": EXPERIMENT / "src/oracle_study/q3_rate_allocator.py",
        "average_allocator_core_sha256": EXPERIMENT / "src/oracle_study/average_rate_allocator.py",
        "split_core_sha256": EXPERIMENT / "src/oracle_study/split_interaction_field.py",
        "interaction_core_sha256": EXPERIMENT / "src/oracle_study/interaction_field.py",
        "mxfp4_core_sha256": EXPERIMENT / "src/oracle_study/mxfp4_embed.py",
        "set_utility_runner_sha256": EXPERIMENT / "scripts/run_set_utility_distillation.py",
        "sparse_runner_sha256": EXPERIMENT / "scripts/run_sparse_streaming_study.py",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _cpu_capacity() -> dict[str, Any]:
    quota = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    quota_cores = None
    raw = "unavailable"
    if Path("/sys/fs/cgroup/cpu.max").exists():
        raw = Path("/sys/fs/cgroup/cpu.max").read_text().strip()
        fields = raw.split()
        if len(fields) == 2 and fields[0] != "max":
            quota_cores = float(fields[0]) / float(fields[1])
    elif quota.exists() and period.exists():
        raw = f"{quota.read_text().strip()} {period.read_text().strip()}"
        numerator, denominator = map(int, raw.split())
        if numerator > 0:
            quota_cores = numerator / denominator
    return {
        "cpu_quota_raw": raw, "quota_cores": quota_cores,
        "affinity_logical_cpus": len(os.sched_getaffinity(0)),
        "os_cpu_count": os.cpu_count(),
    }


def _threads(config: Mapping[str, Any]) -> dict[str, str | None]:
    values = {name: os.environ.get(name) for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    )}
    if set(values.values()) != {str(config["worker_blas_threads"])}:
        raise RuntimeError("BLAS thread environment is not frozen")
    capacity = _cpu_capacity()
    if capacity["quota_cores"] is not None:
        ratio = int(config["evaluation_workers"]) / float(capacity["quota_cores"])
        if ratio > float(config["maximum_worker_oversubscription"]) + 1e-12:
            raise RuntimeError("worker oversubscription exceeds its frozen bound")
        capacity["worker_to_quota_ratio"] = ratio
    return values


def _validate_contract(config: Mapping[str, Any]) -> None:
    if config.get("base_pr13_commit") != "dfba3748e51d6916dc1f28cb1d3b3250188adbbe":
        raise RuntimeError("Q3 study is not stacked on frozen PR #13")
    if config.get("fit_split") != "train" or config.get("selection_split") != "validation":
        raise RuntimeError("split contract changed")
    if config.get("no_test_rows_admitted") is not True:
        raise RuntimeError("test admission must remain forbidden")
    if list(map(int, config.get("layers", []))) != [0, 4, 20, 39]:
        raise RuntimeError("layer scope changed")
    if int(config.get("experts_per_group", -1)) != 8 or int(config.get("experts_per_layer", -1)) != 256:
        raise RuntimeError("expert scope changed")
    if int(config.get("expected_validation_groups", -1)) != 128:
        raise RuntimeError("validation group count changed")
    if int(config.get("factor_rank", -1)) != 4 or int(config.get("factor_payload_bytes_per_expert", -1)) != 4100:
        raise RuntimeError("Rank-4 Hadamard factor changed")
    if int(config.get("abc_metadata_bytes_per_expert", -1)) != 3084:
        raise RuntimeError("A/B/C payload changed")
    if int(config.get("cost_quantum_bytes", -1)) != QUANTUM_BYTES or int(config.get("physical_page_bytes", -1)) != PAGE_BYTES:
        raise RuntimeError("physical transfer geometry changed")
    if int(config.get("frontier_maximum_quanta", -1)) != 3072:
        raise RuntimeError("full Q4 endpoint changed")
    if int(config.get("strict_all_in_mean_quanta", -1)) != 1506:
        raise RuntimeError("strict all-in quantum budget changed")
    if int(config.get("evaluation_workers", -1)) != 32 or config.get("worker_start_method") != "fork":
        raise RuntimeError("worker execution contract changed")
    if float(config.get("maximum_worker_oversubscription", -1)) != 1.2:
        raise RuntimeError("worker oversubscription contract changed")
    expected_layouts = [
        "q2q4_monolithic_same_solver_reference",
        "q3_ideal_logical_256_byte_plane_ceiling",
        "q3_physical_fixed_gate_up_pairing",
        "q3_physical_training_coselection_single",
        "q3_physical_training_coselection_replicated2",
    ]
    if list(config.get("layout_controls", [])) != expected_layouts:
        raise RuntimeError("Q3 layout grid changed")
    if config.get("reproduced_pr13_baseline_layout") != expected_layouts[0]:
        raise RuntimeError("reproduced PR13 baseline identity changed")
    frozen_targets = (
        float(config.get("frozen_pr13_rank4_target_recovery_p10", -1)),
        float(config.get("frozen_pr13_rank4_target_recovery_median", -1)),
        float(config.get("frozen_pr13_rank4_target_total_bpw", -1)),
        float(config.get("baseline_reproduction_absolute_tolerance", -1)),
    )
    expected_targets = (
        0.9959026317130112, 0.998869448260264,
        0.9987386067708334, 0.0005,
    )
    if frozen_targets != expected_targets:
        raise RuntimeError("frozen PR13 Rank-4 target changed")
    expected_rate_grid = sorted({
        *range(384, 1281, 32),
        *range(1284, 1537, 4),
        1506, 1507, 1568, 1600, 1632, 1664,
    })
    if list(map(int, config.get("mean_correction_quanta", []))) != expected_rate_grid:
        raise RuntimeError("matched-target Q3 rate grid changed")
    if int(config.get("matched_target_search_resolution_quanta", -1)) != 4:
        raise RuntimeError("matched-target search resolution changed")
    if config.get("legacy_q2q4_fallback_replica") is not True:
        raise RuntimeError("legacy Q2/Q4 fallback replica changed")

    if list(config.get("allocation_policies", [])) != [
        "uniform_per_expert", "pooled_router_square_multi_budget_column_generated",
        "pooled_exact_combined_moe_local_control",
    ]:
        raise RuntimeError("allocation policy grid changed")
    prices = list(map(float, config.get("frontier_price_ratios", [])))
    if not prices or prices != sorted(prices, reverse=True) or prices[-1] != 0.0:
        raise RuntimeError("price path changed")
    if int(config.get("layout_fit_replicas", -1)) != 2:
        raise RuntimeError("replicated layout count changed")


def _base_artifacts(config: Mapping[str, Any], pr13_config: Path, pr13_dir: Path) -> dict[str, str]:
    paths = {
        "base_pr13_config_sha256": pr13_config,
        "base_pr13_runner_sha256": EXPERIMENT / "scripts/run_average_rate_allocation.py",
        "base_pr13_allocator_sha256": EXPERIMENT / "src/oracle_study/average_rate_allocator.py",
        "base_pr13_split_core_sha256": EXPERIMENT / "src/oracle_study/split_interaction_field.py",
        "base_pr13_fit_facts_sha256": pr13_dir / "fit" / pr13.FIT_FACTS,
        "base_pr13_fit_manifest_sha256": pr13_dir / "fit" / pr13.FIT_MANIFEST,
        "base_pr13_validation_facts_sha256": pr13_dir / "validation_exact_h4" / pr13.RUN_FACTS,
        "base_pr13_promotions_sha256": pr13_dir / "analysis" / "average_rate_promotions.json",
    }
    observed = {}
    for name, path in paths.items():
        value = _sha256(path)
        if value != str(config[name]):
            raise RuntimeError(f"frozen PR #13 input changed: {path}")
        observed[name] = value
    manifest = json.loads((pr13_dir / "fit" / pr13.FIT_MANIFEST).read_text())
    facts = json.loads((pr13_dir / "fit" / pr13.FIT_FACTS).read_text())
    if manifest.get("completed") is not True or facts.get("completed") is not True:
        raise RuntimeError("frozen PR #13 factor fit is incomplete")
    if str(config["factor_id"]) not in manifest.get("factor_ids", []):
        raise RuntimeError("frozen fit does not contain the Rank-4 Hadamard factor")
    for layer in map(int, config["layers"]):
        record = manifest["layers"][str(layer)]
        shard = pr13_dir / "fit" / record["file"]
        sidecar = pr13_dir / "fit" / record["sidecar"]
        if _sha256(shard) != record["sha256"] or shard.stat().st_size != int(record["bytes"]):
            raise RuntimeError("PR13 factor shard changed")
        if _sha256(sidecar) != record["sidecar_sha256"]:
            raise RuntimeError("PR13 factor sidecar changed")
    return observed


def _locked_inputs(
    config: Mapping[str, Any], pr10_config: Path, capture: Path,
    checkpoint: Path, trees: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    if _sha256(pr10_config) != str(config["pr10_config_sha256"]):
        raise RuntimeError("frozen PR #10 config changed")
    return base.verify_locked_inputs(
        json.loads(pr10_config.read_text()), pr10_config, capture,
        checkpoint, trees, "exact_checkpoint",
    )


def _factor_layer(
    pr13_dir: Path, manifest: Mapping[str, Any], layer: int,
) -> dict[str, np.ndarray]:
    record = manifest["layers"][str(int(layer))]
    shard = pr13_dir / "fit" / str(record["file"])
    if _sha256(shard) != record["sha256"] or shard.stat().st_size != int(record["bytes"]):
        raise RuntimeError("factor shard changed during Q3 study")
    return dict(np.load(shard, allow_pickle=False))


def _rank4_factor(
    arrays: Mapping[str, np.ndarray], pr13_config: Mapping[str, Any], expert: int,
):
    factors = pr13._load_layer_factors(arrays, pr13_config, int(expert))
    return factors["exact_proxy_rank4_int4_per_row_hadamard"]


def _fit_layout_expert(task: int) -> tuple[int, np.ndarray, int]:
    if _LAYOUT_CONTEXT is None:
        raise RuntimeError("layout context is not initialized")
    expert = int(_LAYOUT_CONTEXT["experts"][int(task)])
    local = _LAYOUT_CONTEXT["local"]
    records, _ = occurrence(
        local, expert, "train",
        int(_LAYOUT_CONTEXT["config"]["layout_fit_max_train_invocations_per_expert"]),
    )
    if not len(records):
        return expert, np.empty((0, 4 * UNITS), np.uint8), 0
    decoded = prior.decode_expert(
        _LAYOUT_CONTEXT["checkpoint"], _LAYOUT_CONTEXT["index"],
        _LAYOUT_CONTEXT["trees"], int(_LAYOUT_CONTEXT["layer"]), expert,
    )
    q2 = tuple(decoded[name][0] for name in PROJECTIONS)
    q3 = tuple(decoded[name][1] for name in PROJECTIONS)
    q4 = tuple(decoded[name][2] for name in PROJECTIONS)
    abc = unit_score_metadata(
        q2[2], q4[2], proxy=_LAYOUT_CONTEXT["proxy"], beta=_LAYOUT_CONTEXT["beta"],
    )
    factor = _rank4_factor(_LAYOUT_CONTEXT["arrays"], _LAYOUT_CONTEXT["pr13_config"], expert)
    fields = []
    for record in records.tolist():
        responses = q3_projection_responses(q2, q3, q4, local["x"][int(record)])
        fields.append(build_q3_interaction_field(factor, responses.hidden, abc))
    incidence = training_plane_incidence(
        fields, price_ratios=_LAYOUT_CONTEXT["config"]["layout_fit_price_ratios"],
    )
    return expert, incidence, len(records)


def _fit_layouts(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    _validate_contract(config)
    base_hashes = _base_artifacts(config, args.pr13_config, args.pr13_dir)
    audit, inputs = _locked_inputs(config, args.pr10_config, args.exact_captures, args.checkpoint, args.trees)
    threads = _threads(config)
    dependency = _dependencies()
    runtime = prior.runtime_provenance(args.device)
    frozen = {
        "run_id": config["run_id"], "phase": "fit_layout", "completed": False,
        "fit_split": "train", "test_scientific_rows_admitted_or_used": False,
        "config_sha256": _sha256(args.config), "capture_sha256": inputs["capture_sha256"],
        "tree_sha256": inputs["tree_sha256"],
        "checkpoint_config_sha256": inputs["config_sha256"],
        "checkpoint_index_sha256": inputs["index_sha256"],
        "runtime_provenance": runtime,
        "worker_thread_environment": threads, **base_hashes, **dependency,
    }
    if args.output.exists():
        facts_path = args.output / LAYOUT_FACTS
        if not facts_path.is_file():
            raise RuntimeError("layout output exists without facts")
        facts = json.loads(facts_path.read_text())
        for name, value in frozen.items():
            if name != "completed" and facts.get(name) != value:
                raise RuntimeError(f"layout resume provenance changed: {name}")
        if facts.get("completed") is True:
            return
    else:
        args.output.mkdir(parents=True)
        facts = {**frozen, "completed_layers": [], "failures": [], "failure_history": [],
                 "checkpoint_audit": audit, "cpu_capacity": _cpu_capacity()}
        _atomic_json(args.output / LAYOUT_FACTS, facts)
    data = prior.load_capture_admitted(args.exact_captures, ["train"])
    pr13_config = json.loads(args.pr13_config.read_text())
    manifest = json.loads((args.pr13_dir / "fit" / pr13.FIT_MANIFEST).read_text())
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    tree_records = json.loads(args.trees.read_text())
    trees = {name: tree_from_record(tree_records[name]) for name in PROJECTIONS}
    try:
        for layer in map(int, config["layers"]):
            if layer in set(map(int, facts["completed_layers"])):
                continue
            local = prior.layer_view(data, layer)
            proxy, proxy_facts = base.proxy_gradients(
                local["xplus"], [local[f"h{h}_router_logits"] for h in range(1, 5)], local["split"],
            )
            arrays = _factor_layer(args.pr13_dir, manifest, layer)
            experts = sorted(set(map(int, np.asarray(local["expert_ids"]).reshape(-1))))
            global _LAYOUT_CONTEXT
            _LAYOUT_CONTEXT = {
                "experts": experts, "local": local, "config": config,
                "checkpoint": args.checkpoint, "index": index, "trees": trees,
                "layer": layer, "proxy": proxy, "beta": float(proxy_facts["beta"]),
                "arrays": arrays, "pr13_config": pr13_config,
            }
            pending = {}
            context = mp.get_context(config["worker_start_method"])
            with ProcessPoolExecutor(
                max_workers=min(int(config["evaluation_workers"]), len(experts)),
                mp_context=context,
            ) as executor:
                futures = {executor.submit(_fit_layout_expert, task): task for task in range(len(experts))}
                for future in as_completed(futures):
                    expert, incidence, count = future.result()
                    pending[int(expert)] = (incidence, int(count))
            rows = [pending[expert][0] for expert in experts if len(pending[expert][0])]
            if not rows:
                raise RuntimeError("training layout incidence is empty")
            incidence = np.concatenate(rows, axis=0)
            action_pages = fit_coselection_layouts(
                incidence, units=UNITS, replicas=int(config["layout_fit_replicas"]),
            ).astype(np.uint16)
            action_counts = incidence.sum(axis=0, dtype=np.uint32)
            pair_affinity = np.empty((2, 2 * UNITS), np.uint32)
            for replica in range(2):
                for page in range(2 * UNITS):
                    pair = np.flatnonzero(action_pages[replica] == page)
                    pair_affinity[replica, page] = np.count_nonzero(
                        incidence[:, int(pair[0])] & incidence[:, int(pair[1])]
                    )
            shard = args.output / LAYOUT_LAYER.format(layer=layer)
            _atomic_npz(shard, {
                "action_pages": action_pages, "action_counts": action_counts,
                "pair_affinity": pair_affinity,
            })
            sidecar = args.output / LAYOUT_SIDECAR.format(layer=layer)
            _atomic_json(sidecar, {
                "layer": layer, "fit_split": "train", "validation_or_test_values_used": False,
                "routed_experts": len(experts), "routed_train_invocations": sum(value[1] for value in pending.values()),
                "incidence_rows": len(incidence), "incidence_sha256": _bytes_sha256(incidence),
                "shard": shard.name, "shard_sha256": _sha256(shard), "shard_bytes": shard.stat().st_size,
                "runtime_provenance": runtime,
            })
            facts["completed_layers"] = sorted(set(map(int, facts["completed_layers"])) | {layer})
            _atomic_json(args.output / LAYOUT_FACTS, facts)
            _LAYOUT_CONTEXT = None
            del arrays, incidence, rows
            gc.collect()
        layers = {}
        for layer in map(int, config["layers"]):
            sidecar_path = args.output / LAYOUT_SIDECAR.format(layer=layer)
            record = json.loads(sidecar_path.read_text())
            shard = args.output / record["shard"]
            if _sha256(shard) != record["shard_sha256"] or shard.stat().st_size != int(record["shard_bytes"]):
                raise RuntimeError("layout shard changed before manifest")
            layers[str(layer)] = {**record, "sidecar_sha256": _sha256(sidecar_path)}
        manifest_path = args.output / LAYOUT_MANIFEST
        _atomic_json(manifest_path, {
            "schema_version": 1, "completed": True, "run_id": config["run_id"],
            "fit_split": "train", "validation_or_test_values_used": False,
            "layout_scope": "layer_shared_training_only_routed_exact_checkpoint_incidence",
            "replicas": 2, "actions_per_layer": 2048, "pages_per_replica": 1024,
            "config_sha256": _sha256(args.config), "runtime_provenance": runtime,
            "capture_sha256": inputs["capture_sha256"], "tree_sha256": inputs["tree_sha256"],
            "checkpoint_config_sha256": inputs["config_sha256"],
            "checkpoint_index_sha256": inputs["index_sha256"],
            "layers": layers, **base_hashes, **dependency,
        })
        facts["completed"] = True
        facts["layout_manifest_sha256"] = _sha256(manifest_path)
        _atomic_json(args.output / LAYOUT_FACTS, facts)
    except Exception as error:
        record = {"type": type(error).__name__, "message": str(error)}
        facts["failures"] = [record]
        facts.setdefault("failure_history", []).append(record)
        _atomic_json(args.output / LAYOUT_FACTS, facts)
        raise


def _native_layouts(action_pages: np.ndarray) -> dict[str, Q3PageLayout]:
    return {
        "q2q4_monolithic_same_solver_reference": monolithic_q2q4_layout(UNITS),
        "q3_ideal_logical_256_byte_plane_ceiling": Q3PageLayout("ideal_logical_256", UNITS, None),
        "q3_physical_fixed_gate_up_pairing": fixed_gate_up_layout(UNITS),
        "q3_physical_training_coselection_single": Q3PageLayout(
            "training_coselection_single", UNITS, np.asarray(action_pages[:1], np.int64), learned=True,
        ),
        "q3_physical_training_coselection_replicated2": Q3PageLayout(
            "training_coselection_replicated2", UNITS, np.asarray(action_pages, np.int64), learned=True,
        ),
    }


def _with_legacy_fallback(layout: Q3PageLayout) -> Q3PageLayout:
    if layout.ideal:
        raise ValueError("ideal layout cannot have a physical fallback replica")
    legacy = monolithic_q2q4_layout(layout.units)
    pages = np.concatenate((
        np.asarray(layout.action_pages, np.int64),
        np.asarray(legacy.action_pages, np.int64),
    ), axis=0)
    return Q3PageLayout(
        f"{layout.layout_id}_plus_legacy_q2q4_fallback",
        layout.units, pages, learned=layout.learned, fixed=layout.fixed,
    )


def _layouts(action_pages: np.ndarray) -> dict[str, Q3PageLayout]:
    native = _native_layouts(action_pages)
    output = dict(native)
    for layout_id in (
        "q3_physical_fixed_gate_up_pairing",
        "q3_physical_training_coselection_single",
        "q3_physical_training_coselection_replicated2",
    ):
        output[layout_id] = _with_legacy_fallback(native[layout_id])
    return output

def _option_data(responses: Any, options: Sequence[RateOption], proxy: np.ndarray, beta: float):
    residuals = np.stack([
        np.asarray(responses.target_output, np.float64) - q3_state_output(responses, option.states)
        for option in options
    ])
    features = qmetric_features(residuals, proxy, beta)
    damage = np.einsum("ij,ij->i", features, features, optimize=True)
    return features, damage


def _pareto(options: Sequence[RateOption]) -> tuple[RateOption, ...]:
    by_cost = {}
    for option in options:
        previous = by_cost.get(int(option.pages))
        key = (float(option.damage), str(option.source), tuple(option.states.tolist()))
        if previous is None or key < (float(previous.damage), str(previous.source), tuple(previous.states.tolist())):
            by_cost[int(option.pages)] = option
    output, best = [], np.inf
    for cost in sorted(by_cost):
        option = by_cost[cost]
        if float(option.damage) < best - 1e-12:
            output.append(option)
            best = float(option.damage)
    return tuple(output)


def _merge_rebased_frontiers(
    field: Q3InteractionField, layout: Q3PageLayout,
    native: Sequence[RateOption], injections: Sequence[tuple[str, Sequence[RateOption]]],
    *, maximum_quanta: int,
) -> tuple[RateOption, ...]:
    """Union candidate states after charging each under the target layout."""
    candidates = list(native)
    witnesses = []
    for label, frontier in injections:
        for option in frontier:
            states = np.asarray(option.states, np.int64)
            cost = int(layout.cost_quanta(states))
            if cost > int(maximum_quanta):
                continue
            mapped = RateOption(
                cost, float(field.damage(states)), states.copy(),
                f"injected_{label}:{option.source}", float(option.page_price),
                int(option.coordinate_sweeps), int(option.local_passes),
            )
            candidates.append(mapped)
            witnesses.append(mapped)
    merged = _pareto(candidates)
    for witness in witnesses:
        if not any(
            int(candidate.pages) <= int(witness.pages)
            and float(candidate.damage) <= float(witness.damage) + 1e-12
            for candidate in merged
        ):
            raise RuntimeError("Q3 frontier lost an injected dominance witness")
    return merged


def _merge_literal_rebased_frontiers(
    field: Q3InteractionField, layout: Q3PageLayout,
    native: Sequence[RateOption], injections: Sequence[tuple[str, Sequence[RateOption]]],
    *, maximum_quanta: int,
) -> tuple[RateOption, ...]:
    """Union exact state witnesses without Pareto-pruning them away.

    This is used only after column generation.  The allocator must see every
    inherited Q2/Q4 state literally: domination by a different compressed
    state is not a reproducible baseline witness.
    """
    candidates = [option for option in native if int(option.pages) <= int(maximum_quanta)]
    witnesses: list[RateOption] = []
    for label, frontier in injections:
        for option in frontier:
            states = np.asarray(option.states, np.int64)
            cost = int(layout.cost_quanta(states))
            if cost > int(maximum_quanta):
                continue
            witness = RateOption(
                cost, float(field.damage(states)), states.copy(),
                f"literal_{label}:{option.source}", float(option.page_price),
                int(option.coordinate_sweeps), int(option.local_passes),
            )
            candidates.append(witness)
            witnesses.append(witness)

    unique: dict[tuple[int, bytes], RateOption] = {}
    for option in candidates:
        key = (int(option.pages), np.asarray(option.states, np.int64).tobytes())
        previous = unique.get(key)
        if previous is None or (
            float(option.damage), str(option.source)
        ) < (float(previous.damage), str(previous.source)):
            unique[key] = option
    merged = tuple(sorted(
        unique.values(),
        key=lambda option: (
            int(option.pages), float(option.damage), str(option.source),
            tuple(np.asarray(option.states, np.int64).tolist()),
        ),
    ))
    for witness in witnesses:
        if not any(
            int(candidate.pages) == int(witness.pages)
            and np.array_equal(candidate.states, witness.states)
            for candidate in merged
        ):
            raise RuntimeError("Q3 frontier lost a literal injected state witness")
    return merged


def _rebased_allocation_witness(
    candidate_frontiers: Sequence[Sequence[RateOption]],
    reference_frontiers: Sequence[Sequence[RateOption]],
    reference: AllocationTrace, weights: np.ndarray,
) -> AllocationTrace:
    """Map one allocation to identical states in another charged layout."""
    importance = np.asarray(weights, np.float64).reshape(-1)
    if importance.shape != (len(candidate_frontiers),):
        raise ValueError("allocation witness weights do not match frontiers")
    selected = []
    for expert, reference_index in enumerate(reference.option_indices.tolist()):
        reference_option = reference_frontiers[expert][int(reference_index)]
        matches = [
            index for index, option in enumerate(candidate_frontiers[expert])
            if np.array_equal(option.states, reference_option.states)
        ]
        if not matches:
            raise RuntimeError("Q3 frontier lacks a literal group-allocation witness")
        selected.append(min(matches, key=lambda index: (
            int(candidate_frontiers[expert][index].pages),
            float(candidate_frontiers[expert][index].damage), index,
        )))
    pages = sum(
        int(candidate_frontiers[expert][index].pages)
        for expert, index in enumerate(selected)
    )
    objective = sum(
        float(importance[expert]) * float(candidate_frontiers[expert][index].damage)
        for expert, index in enumerate(selected)
    )
    return AllocationTrace(np.asarray(selected, np.int64), int(pages), float(objective))


def _prefer_allocation(candidate: AllocationTrace, witness: AllocationTrace) -> AllocationTrace:
    return witness if (
        float(witness.objective), int(witness.pages), tuple(witness.option_indices.tolist())
    ) < (
        float(candidate.objective), int(candidate.pages), tuple(candidate.option_indices.tolist())
    ) else candidate


def _assert_allocation_objective_dominates(
    candidate: AllocationTrace, reference: AllocationTrace, label: str,
) -> None:
    if float(candidate.objective) > float(reference.objective) + 1e-10:
        raise RuntimeError(f"{label} failed compressed-objective dominance")


def _uniform(frontiers: Sequence[Sequence[RateOption]], mean_quanta: int) -> AllocationTrace:
    chosen = []
    for frontier in frontiers:
        feasible = [index for index, option in enumerate(frontier) if int(option.pages) <= int(mean_quanta)]
        chosen.append(min(feasible, key=lambda index: (float(frontier[index].damage), int(frontier[index].pages), index)))
    pages = sum(int(frontiers[expert][index].pages) for expert, index in enumerate(chosen))
    objective = sum(float(frontiers[expert][index].damage) for expert, index in enumerate(chosen))
    return AllocationTrace(np.asarray(chosen, np.int64), pages, objective)


def _layout_descriptor_bytes(layout_id: str, config: Mapping[str, Any]) -> int:
    if layout_id == "q3_physical_training_coselection_single":
        return int(config["layout_descriptor_bytes_per_layer_single"]) // 256
    if layout_id == "q3_physical_training_coselection_replicated2":
        return int(config["layout_descriptor_bytes_per_layer_replicated"]) // 256
    return 0


def _storage_multiplier(layout_id: str, layout: Q3PageLayout, config: Mapping[str, Any]) -> float:
    base_bytes = BASE_EXACT_BPW * EXPERT_WEIGHTS / 8.0
    extra_replica = max(layout.replicas - 1, 0) * (2 * 512 * 2048 * 2 / 8)
    resident = int(config["factor_payload_bytes_per_expert"]) + int(config["abc_metadata_bytes_per_expert"])
    resident += _layout_descriptor_bytes(layout_id, config)
    return float((base_bytes + extra_replica + resident) / base_bytes)


def _allocation_rows(
    group: Mapping[str, Any], layout_id: str, layout: Q3PageLayout,
    frontiers: Sequence[Sequence[RateOption]], features: Sequence[np.ndarray],
    exact: Sequence[np.ndarray], base_features: Sequence[np.ndarray], weights: np.ndarray,
    mean_quanta: int, policy: str, allocation: AllocationTrace,
    runtime_seconds: float, work: Mapping[str, int], config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = np.asarray(allocation.option_indices, np.int64)
    chosen = [features[expert][int(index)] for expert, index in enumerate(selected)]
    combined = sum((weights[expert] * chosen[expert] for expert in range(8)),
                   np.zeros(chosen[0].shape[0], np.float64))
    base_combined = sum((weights[expert] * base_features[expert] for expert in range(8)),
                        np.zeros(base_features[0].shape[0], np.float64))
    damage = float(combined @ combined)
    base_damage = float(base_combined @ base_combined)
    if not np.isfinite(base_damage) or base_damage <= 0:
        raise RuntimeError("group base damage is not positive finite")
    total_quanta = sum(int(frontiers[expert][int(index)].pages) for expert, index in enumerate(selected))
    if total_quanta != int(allocation.pages) or total_quanta > 8 * int(mean_quanta):
        raise RuntimeError("group quantum budget changed")
    descriptor = _layout_descriptor_bytes(layout_id, config)
    metadata = int(config["factor_payload_bytes_per_expert"]) + int(config["abc_metadata_bytes_per_expert"]) + descriptor
    allowed_bytes = 8 * (metadata + int(mean_quanta) * QUANTUM_BYTES)
    actual_bytes = 8 * metadata + total_quanta * QUANTUM_BYTES
    scalar_comparisons = int(work["state_comparisons"])
    rank = int(config["factor_rank"])
    solver_macs = int(work["coordinate_sweeps"]) * (2 * UNITS * rank + 4 * 18 * UNITS)
    solver_macs += int(work["local_passes"]) * (2 * 17 * UNITS * rank)
    group_row = {
        **{name: group[name] for name in GROUP_IDENTITY}, "sequence_id": group["sequence_id"],
        "factor_config_id": config["factor_id"], "layout_id": layout_id,
        "layout_is_physical": not layout.ideal, "layout_replicas": layout.replicas,
        "allocation_policy": policy, "mean_budget_quanta_per_expert": int(mean_quanta),
        "mean_budget_physical_page_equivalent": int(mean_quanta) / 2.0,
        "group_budget_quanta": 8 * int(mean_quanta), "burst_cap_quanta_per_expert": int(config["burst_cap_quanta_per_expert"]),
        "actual_group_quanta": total_quanta, "actual_group_physical_page_equivalent": total_quanta / 2.0,
        "unused_group_quanta": 8 * int(mean_quanta) - total_quanta,
        "factor_payload_bytes_per_expert": int(config["factor_payload_bytes_per_expert"]),
        "abc_metadata_bytes_per_expert": int(config["abc_metadata_bytes_per_expert"]),
        "layout_descriptor_bytes_per_expert": descriptor, "combined_metadata_bytes_per_expert": metadata,
        "combined_metadata_bpw": 8.0 * metadata / EXPERT_WEIGHTS,
        "average_allowed_total_bpw": allowed_bytes / EXPERT_WEIGHTS,
        "average_actual_total_bpw": actual_bytes / EXPERT_WEIGHTS,
        "strict_one_bpw_pass": allowed_bytes <= 8 * int(config["strict_all_in_total_bytes_per_expert"]),
        "group_recovery": 1.0 - damage / base_damage, "group_exact_qenergy_damage": damage,
        "group_base_qenergy_damage": base_damage, "compressed_allocation_objective": float(allocation.objective),
        "selected_expert_quanta": json.dumps([int(frontiers[e][int(i)].pages) for e, i in enumerate(selected)], separators=(",", ":")),
        "router_weights": json.dumps(weights.tolist(), separators=(",", ":")),
        "selector_wall_time_ms": 1000.0 * float(runtime_seconds),
        "selector_frontier_wall_time_ms": 1000.0 * float(work["frontier_wall_seconds"]),
        "selector_allocation_wall_time_ms": 1000.0 * float(work["allocation_wall_seconds"]),
        "selector_compute_macs": solver_macs, "selector_state_comparisons": scalar_comparisons,
        "selector_coordinate_sweeps": int(work["coordinate_sweeps"]),
        "selector_local_passes": int(work["local_passes"]),
        "selector_diagonal_dp_tables": int(work["diagonal_dp_tables"]),
        "selector_diagonal_dp_state_updates": int(work["diagonal_dp_state_updates"]),
        "frontier_refinement_rounds": int(work["refinement_rounds"]),
        "external_storage_multiplier": _storage_multiplier(layout_id, layout, config),
        "exact_h4_gate_up_responses_used": True,
        "exact_cross_expert_information_used": policy == "pooled_exact_combined_moe_local_control",
        "promotable": False,
        "continuation_eligible": layout_id in set(config["primary_physical_layouts"]) and policy == "pooled_router_square_multi_budget_column_generated",
        "selection_regime": "exact_h4_rank4_hadamard_q3_gate_up_geometry_ceiling",
    }
    expert_rows = []
    if policy == "pooled_router_square_multi_budget_column_generated":
        for router_rank, index in enumerate(selected.tolist()):
            option = frontiers[router_rank][index]
            states = np.asarray(option.states, np.uint8)
            selected_replica = layout.selected_replica(states)
            pages = np.empty(0, np.int64) if layout.ideal else layout.page_union(states, selected_replica)
            exact_damage = float(exact[router_rank][index])
            individual_base = float(base_features[router_rank] @ base_features[router_rank])
            expert_rows.append({
                **{name: group[name] for name in GROUP_IDENTITY}, "sequence_id": group["sequence_id"],
                "expert_id": int(group["experts"][router_rank]), "router_rank": router_rank + 1,
                "router_weight": float(weights[router_rank]), "layout_id": layout_id,
                "allocation_policy": policy, "mean_budget_quanta_per_expert": int(mean_quanta),
                "selected_quanta": int(option.pages), "selected_replica": selected_replica,
                "selected_state_counts": json.dumps(np.bincount(states, minlength=18).tolist(), separators=(",", ":")),
                "selected_states_blob": states.tobytes(),
                "selected_states_sha256": _bytes_sha256(states),
                "selected_page_union_sha256": None if layout.ideal else _bytes_sha256(np.asarray(pages, np.int64)),
                "selected_unique_pages": None if layout.ideal else len(pages),
                "selected_gate_q3_or_above_units": int(np.count_nonzero(Q3_STATE_GATE_LEVEL[states] >= 1)),
                "selected_gate_q4_units": int(np.count_nonzero(Q3_STATE_GATE_LEVEL[states] == 2)),
                "selected_up_q3_or_above_units": int(np.count_nonzero(Q3_STATE_UP_LEVEL[states] >= 1)),
                "selected_up_q4_units": int(np.count_nonzero(Q3_STATE_UP_LEVEL[states] == 2)),
                "selected_down_q4_units": int(np.count_nonzero(Q3_STATE_DOWN_HIGH[states])),
                "option_source": option.source, "compressed_predicted_damage": float(option.damage),
                "exact_qenergy_damage": exact_damage, "base_qenergy_damage": individual_base,
                "expert_recovery": 1.0 - exact_damage / individual_base,
            })
    return group_row, expert_rows


def _evaluate_group(task: int):
    if _EVAL_CONTEXT is None:
        raise RuntimeError("evaluation context is not initialized")
    group = _EVAL_CONTEXT["groups"][int(task)]
    config = _EVAL_CONTEXT["config"]
    local = _EVAL_CONTEXT["local"]
    activation = np.asarray(local["x"][int(group["record"])], np.float32)
    weights = np.asarray(group["router_weights"], np.float64)
    expert_ids = list(map(int, group["experts"]))
    layouts = _layouts(_EVAL_CONTEXT["action_pages"])
    native_layouts = _native_layouts(_EVAL_CONTEXT["action_pages"])
    fields, responses = [], []
    for expert in expert_ids:
        cell = _EVAL_CONTEXT["experts"][expert]
        response = q3_projection_responses(cell["q2"], cell["q3"], cell["q4"], activation)
        responses.append(response)
        fields.append(build_q3_interaction_field(cell["factor"], response.hidden, cell["abc"]))

    baseline_id = str(config["reproduced_pr13_baseline_layout"])
    ideal_id = "q3_ideal_logical_256_byte_plane_ceiling"
    maximum = int(config["frontier_maximum_quanta"])
    layout_data = {}
    runtime_start = time.perf_counter()
    for layout_id, layout in layouts.items():
        native_layout = native_layouts[layout_id]
        started = time.perf_counter()
        frontiers, base_features = [], []
        sweeps = passes = comparisons = dp_tables = dp_updates = 0
        allowed = Q3_INHERITED_EIGHT_STATE_MAP if layout_id.startswith("q2q4_") else tuple(range(18))
        for expert in range(8):
            trace = q3_rate_frontier(
                fields[expert], native_layout, maximum_quanta=maximum,
                target_quanta=config["frontier_target_quanta"], price_ratios=config["frontier_price_ratios"],
                coordinate_sweeps=int(config["coordinate_sweeps"]), local_shortlist=int(config["local_shortlist"]),
                local_swap_units=int(config["local_swap_units"]), local_max_passes=int(config["local_max_passes"]),
                price_local_max_passes=int(config["price_local_max_passes"]), allowed_states=allowed,
            )
            values, _ = _option_data(
                responses[expert], trace.options, _EVAL_CONTEXT["proxy"], _EVAL_CONTEXT["beta"],
            )
            zero = next(index for index, option in enumerate(trace.options) if int(option.pages) == 0)
            frontiers.append(_merge_rebased_frontiers(
                fields[expert], layout, (),
                (("native_layout", trace.options),),
                maximum_quanta=maximum,
            ))
            base_features.append(values[zero])
            sweeps += int(trace.coordinate_sweeps); passes += int(trace.local_passes); comparisons += int(trace.state_comparisons)
            dp_tables += int(trace.diagonal_dp_tables)
            dp_updates += int(trace.diagonal_dp_state_updates)
        layout_data[layout_id] = {
            "layout": layout, "frontiers": tuple(frontiers), "base": tuple(base_features),
            "runtime": time.perf_counter() - started,
            "allocation_runtime": 0.0,
            "work": {"coordinate_sweeps": sweeps, "local_passes": passes,
                     "state_comparisons": comparisons, "refinement_rounds": 0,
                     "inherited_candidates_injected": 0,
                     "diagonal_dp_tables": dp_tables,
                     "diagonal_dp_state_updates": dp_updates,
                     "physical_candidates_injected_into_ideal": 0},
        }

    baseline_frontiers = layout_data[baseline_id]["frontiers"]
    for layout_id, data in layout_data.items():
        if layout_id == baseline_id:
            continue
        data["frontiers"] = tuple(
            _merge_rebased_frontiers(
                fields[expert], data["layout"], data["frontiers"][expert],
                (("reproduced_pr13_q2q4", baseline_frontiers[expert]),),
                maximum_quanta=maximum,
            )
            for expert in range(8)
        )
        data["work"]["inherited_candidates_injected"] = sum(
            len(frontier) for frontier in baseline_frontiers
        )

    refinement_order = [
        layout_id for layout_id in config["layout_controls"] if layout_id != ideal_id
    ] + [ideal_id]
    for layout_id in refinement_order:
        data = layout_data[layout_id]
        layout = data["layout"]
        started = time.perf_counter()
        if layout_id == ideal_id:
            injected = 0
            rebuilt = []
            for expert in range(8):
                sources = tuple(
                    (physical_id, layout_data[physical_id]["frontiers"][expert])
                    for physical_id in config["primary_physical_layouts"]
                )
                injected += sum(len(frontier) for _, frontier in sources)
                rebuilt.append(_merge_rebased_frontiers(
                    fields[expert], layout, data["frontiers"][expert], sources,
                    maximum_quanta=maximum,
                ))
            data["frontiers"] = tuple(rebuilt)
            data["work"]["physical_candidates_injected_into_ideal"] = injected
        allowed = Q3_INHERITED_EIGHT_STATE_MAP if layout_id.startswith("q2q4_") else tuple(range(18))
        working = data["frontiers"]
        refinement_rounds = 0
        for _ in range(int(config["column_generation_outer_rounds"])):
            for mean in map(int, config["column_generation_mean_quanta"]):
                refinement = q3_allocation_aware_rate_frontiers(
                    fields, [layout] * 8, working, group_budget_quanta=8 * mean,
                    burst_cap_quanta=int(config["burst_cap_quanta_per_expert"]), weights=weights * weights,
                    coordinate_sweeps=int(config["coordinate_sweeps"]), local_shortlist=int(config["local_shortlist"]),
                    local_swap_units=int(config["local_swap_units"]), local_max_passes=int(config["local_max_passes"]),
                    max_rounds=int(config["column_generation_inner_rounds"]),
                    adaptive_price_multipliers=config["column_generation_price_multipliers"],
                    adaptive_price_local_passes=int(config["column_generation_price_local_max_passes"]),
                    allowed_states=allowed,
                )
                working = refinement.frontiers
                data["work"]["coordinate_sweeps"] += int(refinement.coordinate_sweeps)
                data["work"]["local_passes"] += int(refinement.local_passes)
                data["work"]["state_comparisons"] += 18 * UNITS * int(refinement.coordinate_sweeps)
                refinement_rounds += int(refinement.rounds)
        features, exact = [], []
        for expert in range(8):
            values, damage = _option_data(responses[expert], working[expert], _EVAL_CONTEXT["proxy"], _EVAL_CONTEXT["beta"])
            features.append(values); exact.append(damage)
        data["frontiers"] = working
        data["features"] = tuple(features)
        data["exact"] = tuple(exact)
        data["runtime"] += time.perf_counter() - started
        data["work"]["refinement_rounds"] = refinement_rounds

    final_baseline_frontiers = layout_data[baseline_id]["frontiers"]
    # Column generation Pareto-prunes aggressively.  Reinsert every inherited
    # state literally after refinement, then place every final physical state
    # in the ideal control.  Recompute exact option arrays for the expanded
    # frontiers so row indices remain a closed evidence contract.
    for physical_id in config["primary_physical_layouts"]:
        data = layout_data[physical_id]
        data["frontiers"] = tuple(
            _merge_literal_rebased_frontiers(
                fields[expert], data["layout"], data["frontiers"][expert],
                (("reproduced_pr13_q2q4", final_baseline_frontiers[expert]),),
                maximum_quanta=maximum,
            )
            for expert in range(8)
        )
    ideal = layout_data[ideal_id]
    ideal["frontiers"] = tuple(
        _merge_literal_rebased_frontiers(
            fields[expert], ideal["layout"], ideal["frontiers"][expert],
            tuple(
                (physical_id, layout_data[physical_id]["frontiers"][expert])
                for physical_id in config["primary_physical_layouts"]
            ),
            maximum_quanta=maximum,
        )
        for expert in range(8)
    )
    for data in layout_data.values():
        features, exact = [], []
        for expert in range(8):
            values, damage = _option_data(
                responses[expert], data["frontiers"][expert],
                _EVAL_CONTEXT["proxy"], _EVAL_CONTEXT["beta"],
            )
            features.append(values); exact.append(damage)
        data["features"] = tuple(features)
        data["exact"] = tuple(exact)

    router_cache = {}
    for mean in map(int, config["mean_correction_quanta"]):
        for layout_id, data in layout_data.items():
            started = time.perf_counter()
            router_cache[(layout_id, mean)] = multiple_choice_allocate(
                data["frontiers"], 8 * mean, weights * weights,
            )
            data["allocation_runtime"] += time.perf_counter() - started
        baseline_allocation = router_cache[(baseline_id, mean)]
        for physical_id in config["primary_physical_layouts"]:
            witness = _rebased_allocation_witness(
                layout_data[physical_id]["frontiers"], final_baseline_frontiers,
                baseline_allocation, weights * weights,
            )
            if int(witness.pages) > 8 * mean:
                raise RuntimeError("inherited Q2/Q4 allocation witness exceeds Q3 budget")
            router_cache[(physical_id, mean)] = _prefer_allocation(
                router_cache[(physical_id, mean)], witness,
            )
            _assert_allocation_objective_dominates(
                router_cache[(physical_id, mean)], baseline_allocation,
                f"{physical_id} Q3 at {mean} quanta",
            )

        for physical_id in config["primary_physical_layouts"]:
            physical = router_cache[(physical_id, mean)]
            witness = _rebased_allocation_witness(
                layout_data[ideal_id]["frontiers"],
                layout_data[physical_id]["frontiers"], physical, weights * weights,
            )
            if int(witness.pages) > 8 * mean:
                raise RuntimeError("physical allocation witness exceeds ideal Q3 budget")
            router_cache[(ideal_id, mean)] = _prefer_allocation(
                router_cache[(ideal_id, mean)], witness,
            )
            _assert_allocation_objective_dominates(
                router_cache[(ideal_id, mean)], physical,
                f"ideal Q3 versus {physical_id} at {mean} quanta",
            )

    allocation_cache = {}
    for layout_id in config["layout_controls"]:
        data = layout_data[layout_id]
        for mean in map(int, config["mean_correction_quanta"]):
            started = time.perf_counter()
            router = router_cache[(layout_id, mean)]
            allocations = (
                ("uniform_per_expert", _uniform(data["frontiers"], mean)),
                ("pooled_router_square_multi_budget_column_generated", router),
                ("pooled_exact_combined_moe_local_control", exact_group_option_allocate(
                    data["frontiers"], data["features"], weights, 8 * mean,
                    router.option_indices,
                    max_coordinate_sweeps=int(config["exact_group_coordinate_sweeps"]),
                    max_pair_passes=int(config["exact_group_pair_passes"]),
                )),
            )
            allocation_cache[(layout_id, mean)] = allocations
            data["allocation_runtime"] += time.perf_counter() - started
        data["work"]["frontier_wall_seconds"] = float(data["runtime"])
        data["work"]["allocation_wall_seconds"] = float(data["allocation_runtime"])

    rows, expert_rows = [], []
    for layout_id in config["layout_controls"]:
        data = layout_data[layout_id]
        for mean in map(int, config["mean_correction_quanta"]):
            allocations = allocation_cache[(layout_id, mean)]
            for policy, allocation in allocations:
                row, experts = _allocation_rows(
                    group, layout_id, data["layout"], data["frontiers"], data["features"],
                    data["exact"], data["base"], weights, mean, policy, allocation,
                    data["runtime"] + data["allocation_runtime"], data["work"], config,
                )
                rows.append(row); expert_rows.extend(experts)
    return int(task), rows, expert_rows, {
        "group_index": int(task), "wall_seconds": time.perf_counter() - runtime_start,
        "layouts": {layout_id: {"frontier_options": [len(item) for item in data["frontiers"]],
                                **data["work"]} for layout_id, data in layout_data.items()},
    }


def _evaluate(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    _validate_contract(config)
    base_hashes = _base_artifacts(config, args.pr13_config, args.pr13_dir)
    audit, inputs = _locked_inputs(config, args.pr10_config, args.exact_captures, args.checkpoint, args.trees)
    threads = _threads(config)
    dependency = _dependencies()
    runtime = prior.runtime_provenance(args.device)
    layout_manifest_path = args.layout_dir / LAYOUT_MANIFEST
    layout_facts_path = args.layout_dir / LAYOUT_FACTS
    layout_manifest = json.loads(layout_manifest_path.read_text())
    layout_facts = json.loads(layout_facts_path.read_text())
    if layout_manifest.get("completed") is not True or layout_facts.get("completed") is not True:
        raise RuntimeError("Q3 layout fit is incomplete")
    for name, value in {"config_sha256": _sha256(args.config), "runtime_provenance": runtime,
                        "capture_sha256": inputs["capture_sha256"], "tree_sha256": inputs["tree_sha256"],
                        "checkpoint_config_sha256": inputs["config_sha256"],
                        "checkpoint_index_sha256": inputs["index_sha256"],
                        **base_hashes, **dependency}.items():
        if layout_manifest.get(name) != value or layout_facts.get(name) != value:
            raise RuntimeError(f"layout/evaluation provenance changed: {name}")
    if layout_facts.get("layout_manifest_sha256") != _sha256(layout_manifest_path):
        raise RuntimeError("layout manifest hash changed")
    data = prior.load_capture_admitted(args.exact_captures, ["train", "validation"])
    if "test" in set(map(str, np.unique(data["split"]))):
        raise RuntimeError("test row entered Q3 evaluation")
    groups = pr13._plan(data, config)
    frozen = {
        "run_id": config["run_id"], "phase": "evaluate", "selection_split": "validation",
        "test_scientific_rows_admitted_or_used": False, "validation_groups": len(groups),
        "validation_expert_invocations": 8 * len(groups), "config_sha256": _sha256(args.config),
        "capture_sha256": inputs["capture_sha256"], "tree_sha256": inputs["tree_sha256"],
        "layout_facts_sha256": _sha256(layout_facts_path), "layout_manifest_sha256": _sha256(layout_manifest_path),
        "runtime_provenance": runtime, "evaluation_workers": int(config["evaluation_workers"]),
        "worker_thread_environment": threads, **base_hashes, **dependency,
    }
    group_rows, expert_rows, diagnostics = [], [], []
    if args.output.exists():
        facts_path = args.output / RUN_FACTS
        if not facts_path.is_file():
            raise RuntimeError("evaluation output exists without resumable facts")
        facts = json.loads(facts_path.read_text())
        for name, value in frozen.items():
            if facts.get(name) != value:
                raise RuntimeError(f"evaluation resume provenance changed: {name}")
        if facts.get("completed") is True:
            return
        completed = set(map(int, facts.get("completed_layers", [])))
        if completed:
            group_rows = pd.read_parquet(args.output / GROUP_FRONTIER).to_dict("records")
            expert_rows = pd.read_parquet(args.output / EXPERT_ALLOCATION).to_dict("records")
            diagnostics = list(facts.get("frontier_diagnostics", []))
    else:
        args.output.mkdir(parents=True)
        facts = {"completed": False, "completed_layers": [], "failures": [], "failure_history": [],
                 "checkpoint_audit": audit, "cpu_capacity": _cpu_capacity(), "frontier_diagnostics": [], **frozen}
        _atomic_json(args.output / RUN_FACTS, facts)
    pr13_config = json.loads(args.pr13_config.read_text())
    factor_manifest = json.loads((args.pr13_dir / "fit" / pr13.FIT_MANIFEST).read_text())
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    tree_records = json.loads(args.trees.read_text())
    trees = {name: tree_from_record(tree_records[name]) for name in PROJECTIONS}
    try:
        for layer in map(int, config["layers"]):
            if layer in set(map(int, facts["completed_layers"])):
                continue
            local = prior.layer_view(data, layer)
            proxy, proxy_facts = base.proxy_gradients(
                local["xplus"], [local[f"h{h}_router_logits"] for h in range(1, 5)], local["split"],
            )
            factor_arrays = _factor_layer(args.pr13_dir, factor_manifest, layer)
            layout_record = layout_manifest["layers"][str(layer)]
            layout_shard = args.layout_dir / layout_record["shard"]
            if _sha256(layout_shard) != layout_record["shard_sha256"] or layout_shard.stat().st_size != int(layout_record["shard_bytes"]):
                raise RuntimeError("layout shard changed")
            layout_arrays = dict(np.load(layout_shard, allow_pickle=False))
            layer_groups = [group for group in groups if int(group["layer"]) == layer]
            active = sorted({int(expert) for group in layer_groups for expert in group["experts"]})
            experts = {}
            for expert in active:
                decoded = prior.decode_expert(args.checkpoint, index, trees, layer, expert)
                q2 = tuple(decoded[name][0] for name in PROJECTIONS)
                q3 = tuple(decoded[name][1] for name in PROJECTIONS)
                q4 = tuple(decoded[name][2] for name in PROJECTIONS)
                experts[expert] = {
                    "q2": q2, "q3": q3, "q4": q4,
                    "abc": unit_score_metadata(q2[2], q4[2], proxy=proxy, beta=float(proxy_facts["beta"])),
                    "factor": _rank4_factor(factor_arrays, pr13_config, expert),
                }
            global _EVAL_CONTEXT
            _EVAL_CONTEXT = {"groups": tuple(layer_groups), "local": local, "experts": experts,
                             "proxy": proxy, "beta": float(proxy_facts["beta"]), "config": config,
                             "action_pages": np.asarray(layout_arrays["action_pages"], np.int64)}
            pending = {}
            context = mp.get_context(config["worker_start_method"])
            with ProcessPoolExecutor(max_workers=min(int(config["evaluation_workers"]), len(layer_groups)), mp_context=context) as executor:
                futures = {executor.submit(_evaluate_group, task): task for task in range(len(layer_groups))}
                for future in as_completed(futures):
                    index_value, rows, experts_value, diagnostic = future.result()
                    pending[index_value] = (rows, experts_value, diagnostic)
            for index_value in range(len(layer_groups)):
                rows, experts_value, diagnostic = pending[index_value]
                group_rows.extend(rows); expert_rows.extend(experts_value); diagnostics.append(diagnostic)
            _atomic_parquet(args.output / GROUP_FRONTIER, group_rows)
            _atomic_parquet(args.output / EXPERT_ALLOCATION, expert_rows)
            facts["completed_layers"] = sorted(set(map(int, facts["completed_layers"])) | {layer})
            facts["observed_group_rows"] = len(group_rows); facts["observed_expert_rows"] = len(expert_rows)
            facts["frontier_diagnostics"] = diagnostics
            _atomic_json(args.output / RUN_FACTS, facts)
            _EVAL_CONTEXT = None
            del experts, factor_arrays, layout_arrays
            gc.collect()
        expected_groups = int(config["expected_validation_groups"]) * len(config["layout_controls"]) * len(config["mean_correction_quanta"]) * len(config["allocation_policies"])
        expected_experts = int(config["expected_validation_groups"]) * len(config["layout_controls"]) * len(config["mean_correction_quanta"]) * 8
        if len(group_rows) != expected_groups or len(expert_rows) != expected_experts:
            raise RuntimeError(f"Q3 row grid changed: {len(group_rows)}/{expected_groups}, {len(expert_rows)}/{expected_experts}")
        frame = pd.DataFrame(group_rows)
        keys = list(GROUP_IDENTITY) + ["layout_id", "allocation_policy", "mean_budget_quanta_per_expert"]
        if frame[keys].duplicated().any():
            raise RuntimeError("duplicate Q3 group frontier row")
        if np.any(frame["actual_group_quanta"] > frame["group_budget_quanta"]):
            raise RuntimeError("Q3 group budget exceeded")
        accounting = {
            "schema_version": 1, "group_rows": len(group_rows), "expert_rows": len(expert_rows),
            "validation_groups": int(config["expected_validation_groups"]), "experts_per_group": 8,
            "cost_quantum_bytes": QUANTUM_BYTES, "physical_page_bytes": PAGE_BYTES,
            "factor_payload_bytes_per_expert": int(config["factor_payload_bytes_per_expert"]),
            "abc_metadata_bytes_per_expert": int(config["abc_metadata_bytes_per_expert"]),
            "layout_controls": config["layout_controls"], "allocation_policies": config["allocation_policies"],
            "mean_correction_quanta": config["mean_correction_quanta"],
            "physical_cost_convention": "two_256_byte_planes_per_512_byte_page_charged_by_exact_unique_page_union",
            "replicated_cost_convention": "choose_one_complete_layout_replica_with_minimum_union_per_expert_invocation_no_cross_replica_mixing",
            "legacy_q2q4_fallback_replica": True,
        }
        _atomic_json(args.output / ACCOUNTING, accounting)
        facts["completed"] = True
        facts["group_frontier_sha256"] = _sha256(args.output / GROUP_FRONTIER)
        facts["expert_allocation_sha256"] = _sha256(args.output / EXPERT_ALLOCATION)
        facts["accounting_sha256"] = _sha256(args.output / ACCOUNTING)
        _atomic_json(args.output / RUN_FACTS, facts)
    except Exception as error:
        record = {"type": type(error).__name__, "message": str(error)}
        facts["failures"] = [record]
        facts.setdefault("failure_history", []).append(record)
        _atomic_json(args.output / RUN_FACTS, facts)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("fit-layout", "evaluate"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pr13-config", type=Path, required=True)
    parser.add_argument("--pr13-dir", type=Path, required=True)
    parser.add_argument("--pr10-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trees", type=Path, required=True)
    parser.add_argument("--exact-captures", type=Path, required=True)
    parser.add_argument("--layout-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.phase == "fit-layout":
        _fit_layouts(args, config)
    else:
        if args.layout_dir is None:
            parser.error("--layout-dir is required for evaluation")
        _evaluate(args, config)


if __name__ == "__main__":
    main()
