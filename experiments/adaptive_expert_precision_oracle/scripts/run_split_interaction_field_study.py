#!/usr/bin/env python3
"""Evaluate separate gate/up/down eight-state interaction fields."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(EXPERIMENT / "scripts")]

from oracle_study.interaction_field import (  # noqa: E402
    JointInteractionFactor,
    build_interaction_field,
    field_coordinate_descent,
    field_local_search,
)
from oracle_study.neuron_selector import (  # noqa: E402
    complete_unit_scores,
    factorized_unit_outputs,
    unit_score_metadata,
)
from oracle_study.split_interaction_field import (  # noqa: E402
    INHERITED_FOUR_STATE_MAP,
    SPLIT_ONLY_STATES,
    SPLIT_STATE_DOWN_HIGH,
    SPLIT_STATE_GATE_HIGH,
    SPLIT_STATE_PAGE_COSTS,
    SPLIT_STATE_UP_HIGH,
    build_split_gram_field,
    build_split_interaction_field,
    map_four_state_to_split,
    split_coordinate_descent,
    split_diagonal_dp_seed,
    split_exact_damage,
    split_gram_coordinate_descent,
    split_gram_local_search,
    split_local_search,
    split_projection_responses,
)
from oracle_study.unit_set_teacher import down_metric_gram  # noqa: E402
import run_interaction_field_study as pr11  # noqa: E402
from run_mxfp4_selective_pages import tree_from_record  # noqa: E402
import run_set_utility_distillation as prior  # noqa: E402
import run_sparse_streaming_study as base  # noqa: E402


IDENTITY = (
    "capture_source", "evaluation_split", "request_id", "position",
    "layer", "expert_id",
)
FRONTIER = "split_interaction_field_frontier.parquet"
FACTS = "run_facts.json"
UNITS = 512
WEIGHTS = 3_145_728
ABC_BYTES = 3_084
_EVALUATION_CONTEXT: dict[str, Any] | None = None


def _atomic_json(path: Path, value: Any) -> None:
    prior.atomic_json(path, value)


def _atomic_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    temporary.replace(path)


def _cpu_capacity_facts() -> dict[str, Any]:
    quota_path = Path("/sys/fs/cgroup/cpu.max")
    quota_cores = None
    raw = "unavailable"
    if quota_path.exists():
        raw = quota_path.read_text().strip()
        values = raw.split()
        if len(values) == 2 and values[0] != "max":
            quota_cores = float(values[0]) / float(values[1])
    else:
        quota = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        period = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if quota.exists() and period.exists():
            q, p = int(quota.read_text()), int(period.read_text())
            raw = f"{q} {p}"
            if q > 0:
                quota_cores = q / p
    return {
        "cpu_quota_raw": raw,
        "quota_cores": quota_cores,
        "affinity_logical_cpus": len(os.sched_getaffinity(0)),
        "os_cpu_count": os.cpu_count(),
    }


def _validate_contract(config: Mapping[str, Any], old: Mapping[str, Any]) -> None:
    if config.get("base_pr11_commit") != "b276da7e53d09bdb9450021c6eb4ea7b704debd2":
        raise RuntimeError("study is not stacked on frozen PR #11")
    if config.get("fit_split") != "train" or config.get("selection_split") != "validation":
        raise RuntimeError("split contract changed")
    if config.get("no_test_rows_admitted") is not True:
        raise RuntimeError("test admission must remain forbidden")
    if list(map(int, config.get("page_budgets", []))) != [384, 576, 768]:
        raise RuntimeError("page budgets changed")
    if list(map(int, config.get("state_page_costs", []))) != SPLIT_STATE_PAGE_COSTS.tolist():
        raise RuntimeError("eight-state page costs changed")
    if list(map(int, config.get("inherited_four_state_map", []))) != INHERITED_FOUR_STATE_MAP.tolist():
        raise RuntimeError("four-state embedding changed")
    if int(config.get("expected_validation_unique_invocations", -1)) != int(
        old["expected_validation_unique_invocations"]
    ):
        raise RuntimeError("validation identity contract changed")
    if int(config.get("expected_validation_layer_expert_cells", -1)) != int(
        old["expected_validation_layer_expert_cells"]
    ):
        raise RuntimeError("validation cell contract changed")
    if int(config.get("evaluation_workers", -1)) != 24:
        raise RuntimeError("worker contract changed")
    if config.get("worker_start_method") != "fork" or int(config.get("worker_blas_threads", -1)) != 1:
        raise RuntimeError("worker execution contract changed")
    factors = list(map(str, config.get("factor_config_ids", [])))
    if len(factors) != 4 or len(set(factors)) != 4:
        raise RuntimeError("factor subset changed")


def _load_pr11_references(
    path: Path, config: Mapping[str, Any],
) -> dict[tuple[Any, ...], dict[str, float | int]]:
    frame = pd.read_parquet(path)
    frame = frame[
        (frame["solver"] == "two_seed_coordinate_plus_local_repair")
        & frame["factor_config_id"].isin(config["factor_config_ids"])
    ].copy()
    expected = (
        int(config["expected_validation_unique_invocations"])
        * len(config["page_budgets"])
        * len(config["factor_config_ids"])
    )
    if len(frame) != expected:
        raise RuntimeError(f"PR #11 reference grid {len(frame)} != {expected}")
    result = {}
    for row in frame.to_dict("records"):
        identity = tuple(row[name] for name in IDENTITY)
        key = identity + (int(row["physical_budget_pages"]), str(row["factor_config_id"]))
        if key in result:
            raise RuntimeError("duplicate PR #11 reference")
        result[key] = {
            "recovery": float(row["recovery"]),
            "exact_hybrid": float(row["pr10_exact_hybrid_recovery"]),
            "selector_compute_macs": int(row["selector_compute_macs"]),
        }
    return result


def _state_summary(states: np.ndarray) -> dict[str, Any]:
    value = np.asarray(states, np.int64)
    counts = np.bincount(value, minlength=8)
    split_only = np.isin(value, tuple(SPLIT_ONLY_STATES))
    return {
        "selected_states": json.dumps(value.tolist(), separators=(",", ":")),
        "state_counts": json.dumps(counts.tolist(), separators=(",", ":")),
        "split_only_units": int(np.count_nonzero(split_only)),
        "gate_high_units": int(np.count_nonzero(SPLIT_STATE_GATE_HIGH[value])),
        "up_high_units": int(np.count_nonzero(SPLIT_STATE_UP_HIGH[value])),
        "down_high_units": int(np.count_nonzero(SPLIT_STATE_DOWN_HIGH[value])),
    }


def _old_solution(
    outputs: Any,
    abc: Any,
    factor: JointInteractionFactor,
    budget: int,
    config: Mapping[str, Any],
) -> tuple[Any, str]:
    field = build_interaction_field(factor, outputs.h2, outputs.h4, abc)
    all_00 = np.zeros(UNITS, np.int64)
    independent = all_00.copy()
    count = min(UNITS, int(budget) // 3)
    order = np.lexsort((
        np.arange(UNITS),
        -complete_unit_scores(outputs.h2, outputs.h4, abc),
    ))
    independent[order[:count]] = 3
    traces = {
        "all_00": field_coordinate_descent(
            field, all_00, budget, max_sweeps=int(config["coordinate_sweeps"]),
        ),
        "independent_abc": field_coordinate_descent(
            field, independent, budget, max_sweeps=int(config["coordinate_sweeps"]),
        ),
    }
    name, chosen = min(traces.items(), key=lambda item: (item[1].damage, item[0]))
    repaired = field_local_search(
        field,
        chosen.states,
        budget,
        shortlist_size=int(config["local_shortlist"]),
        max_swap_units=int(config["local_swap_units"]),
        max_passes=int(config["local_max_passes"]),
        page_balanced=False,
    )
    return repaired, name


def _split_solver(
    field: Any,
    budget: int,
    old_states: np.ndarray,
    config: Mapping[str, Any],
    *,
    inherited: bool,
) -> tuple[Any, str, int, int]:
    all_00 = np.zeros(UNITS, np.int64)
    diagonal = split_diagonal_dp_seed(field, budget)
    seeds = {"all_00": all_00, "diagonal_exact_self_dp": diagonal}
    if inherited:
        seeds["inherited_four_state"] = map_four_state_to_split(old_states)
    traces = {
        name: split_coordinate_descent(
            field, seed, budget, max_sweeps=int(config["coordinate_sweeps"]),
        )
        for name, seed in seeds.items()
    }
    name, chosen = min(traces.items(), key=lambda item: (item[1].damage, item[0]))
    repaired = split_local_search(
        field,
        chosen.states,
        budget,
        shortlist_size=int(config["local_shortlist"]),
        max_swap_units=int(config["local_swap_units"]),
        max_passes=int(config["local_max_passes"]),
    )
    sweeps = sum(int(trace.sweeps) for trace in traces.values())
    passes = min(int(config["local_max_passes"]), int(repaired.passes) + 1)
    return repaired, name, sweeps, passes


def _split_compute(
    rank: int, sweeps: int, passes: int,
) -> tuple[int, dict[str, int]]:
    build = 16 * UNITS * rank + 24 * UNITS
    coordinate = (2 * UNITS * rank + 16 * UNITS) * int(sweeps)
    maximum_shortlist = 7 * 8
    local = int(passes) * (
        14 * UNITS * rank + maximum_shortlist * maximum_shortlist * rank
    )
    components = {
        "field_build_macs": int(build),
        "coordinate_macs": int(coordinate),
        "local_search_macs": int(local),
    }
    return sum(components.values()), components


def _row(
    metadata: Mapping[str, Any],
    *,
    factor_entry: Mapping[str, Any] | None,
    solver: str,
    states: np.ndarray,
    budget: int,
    exact_damage: float,
    base_damage: float,
    reference: Mapping[str, float | int],
    predicted_damage: float,
    runtime_seconds: float,
    selector_macs: int | None,
    components: Mapping[str, int],
    seed: str,
    coordinate_sweeps: int,
    local_passes: int,
    exact_gram_oracle: bool,
) -> dict[str, Any]:
    recovery = 1.0 - float(exact_damage) / float(base_damage)
    factor_id = (
        "exact_full_down_gram_training_only"
        if factor_entry is None else str(factor_entry["factor_config_id"])
    )
    factor_bytes = None if factor_entry is None else int(factor_entry["factor_payload_bytes"])
    factor_rank = None if factor_entry is None else int(factor_entry["total_rank"])
    result = {
        **metadata,
        "physical_budget_pages": int(budget),
        "physical_pages": int(SPLIT_STATE_PAGE_COSTS[np.asarray(states)].sum()),
        "physical_budget_bpw": float(budget) / 768.0,
        "factor_config_id": factor_id,
        "factor_family": "exact_full_down_gram" if factor_entry is None else factor_entry["factor_family"],
        "total_rank": factor_rank,
        "factor_payload_bytes": factor_bytes,
        "combined_metadata_bytes": None if factor_bytes is None else factor_bytes + ABC_BYTES,
        "combined_metadata_bpw": None if factor_bytes is None else 8.0 * (factor_bytes + ABC_BYTES) / WEIGHTS,
        "solver": solver,
        "selected_seed": seed,
        "recovery": float(recovery),
        "four_state_reference_recovery": float(reference["recovery"]),
        "pr11_four_state_recovery": float(
            reference.get("pr11_recovery", reference["recovery"])
        ),
        "recovery_gain_vs_four_state_reference": float(
            recovery - float(reference["recovery"])
        ),
        "four_state_reference_kind": str(
            reference.get("kind", "same_factor_pr11")
        ),
        "pr10_exact_hybrid_recovery": float(reference["exact_hybrid"]),
        "set_gain_retention_vs_pr10_exact_hybrid": float(recovery / float(reference["exact_hybrid"])),
        "compressed_predicted_damage": float(predicted_damage),
        "exact_qenergy_damage": float(exact_damage),
        "damage_prediction_error_over_base": float((predicted_damage - exact_damage) / base_damage),
        "selector_runtime_ms": 1000.0 * float(runtime_seconds),
        "selector_compute_macs": selector_macs,
        "selector_compute_gate_pass": (
            None if selector_macs is None
            else int(selector_macs) < 1_572_864
        ),
        "selector_compute_accounting_bound": (
            "training_only_exact_gram_oracle_nonpromotable"
            if exact_gram_oracle
            else "complete_conservative_logical_MAC_upper_bound"
        ),
        "coordinate_sweeps": int(coordinate_sweeps),
        "local_evaluated_passes": int(local_passes),
        "uses_exact_mixed_h4": True,
        "exact_gram_oracle": bool(exact_gram_oracle),
        "promotable": bool(
            not exact_gram_oracle
            and factor_entry is not None
            and factor_entry["factor_config_id"] in _EVALUATION_CONTEXT["config"]["primary_factor_config_ids"]
            and solver == _EVALUATION_CONTEXT["config"]["primary_solver_id"]
        ),
        "selection_regime": "h0_exact_mixed_h4_split_action_geometry_ceiling",
        "selector_non_mac_operations": (
            "exact_self_DP_state_additions_sort_topk_and_control_flow_excluded"
        ),
        **{name: int(value) for name, value in components.items()},
        **_state_summary(states),
    }
    return result


def _exact_gram_rows(
    responses: Any,
    abc: Any,
    gram: Any,
    budget: int,
    metadata: Mapping[str, Any],
    reference: Mapping[str, float | int],
    base_damage: float,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    field = build_split_gram_field(gram, responses.hidden, abc)
    restricted = tuple(INHERITED_FOUR_STATE_MAP.tolist())
    started = time.perf_counter()
    four_seeds = {
        "all_00": np.zeros(UNITS, np.int64),
        "diagonal_exact_self_dp": split_diagonal_dp_seed(field, budget, restricted),
    }
    four_traces = {
        name: split_gram_coordinate_descent(
            field, seed, budget, allowed_states=restricted,
            max_sweeps=int(config["coordinate_sweeps"]),
        )
        for name, seed in four_seeds.items()
    }
    four_name, four = min(four_traces.items(), key=lambda item: (item[1].damage, item[0]))
    four = split_gram_local_search(
        field, four.states, budget, allowed_states=restricted,
        shortlist_size=int(config["local_shortlist"]),
        max_swap_units=int(config["local_swap_units"]),
        max_passes=int(config["local_max_passes"]),
    )
    four_seconds = time.perf_counter() - started
    full_started = time.perf_counter()
    full_seeds = {
        "all_00": np.zeros(UNITS, np.int64),
        "diagonal_exact_self_dp": split_diagonal_dp_seed(field, budget),
        "inherited_exact_four_state": four.states,
    }
    full_traces = {
        name: split_gram_coordinate_descent(
            field, seed, budget, max_sweeps=int(config["coordinate_sweeps"]),
        )
        for name, seed in full_seeds.items()
    }
    full_name, full = min(full_traces.items(), key=lambda item: (item[1].damage, item[0]))
    full = split_gram_local_search(
        field, full.states, budget,
        shortlist_size=int(config["local_shortlist"]),
        max_swap_units=int(config["local_swap_units"]),
        max_passes=int(config["local_max_passes"]),
    )
    full_seconds = time.perf_counter() - full_started
    if full.damage > four.damage + 1e-7 * max(base_damage, 1.0):
        raise RuntimeError("exact eight-state solver lost to embedded four-state result")
    exact_reference = dict(reference)
    exact_reference.update({
        "pr11_recovery": float(reference["recovery"]),
        "recovery": 1.0 - float(four.damage) / float(base_damage),
        "kind": "exact_full_gram_four_state_same_solver",
    })
    rows = []
    for solver, trace, seed, seconds, sweeps in (
        (
            config["exact_four_state_solver_id"], four, four_name, four_seconds,
            sum(item.sweeps for item in four_traces.values()),
        ),
        (
            config["exact_eight_state_solver_id"], full, full_name, full_seconds,
            sum(item.sweeps for item in full_traces.values()),
        ),
    ):
        exact = split_exact_damage(
            responses, trace.states,
            proxy=_EVALUATION_CONTEXT["cell_proxy"], beta=_EVALUATION_CONTEXT["cell_beta"],
        )
        if not np.isclose(trace.damage, exact, rtol=1e-8, atol=1e-6):
            raise RuntimeError("exact Gram solver lost output-space parity")
        rows.append(_row(
            metadata,
            factor_entry=None,
            solver=solver,
            states=trace.states,
            budget=budget,
            exact_damage=exact,
            base_damage=base_damage,
            reference=exact_reference,
            predicted_damage=trace.damage,
            runtime_seconds=seconds,
            selector_macs=None,
            components={},
            seed=seed,
            coordinate_sweeps=sweeps,
            local_passes=int(trace.passes),
            exact_gram_oracle=True,
        ))
    return rows


def _evaluate_task(task: tuple[int, int]) -> tuple[int, int, list[dict[str, Any]]]:
    if _EVALUATION_CONTEXT is None:
        raise RuntimeError("evaluation context is not initialized")
    cell_index, observation_index = map(int, task)
    cell = _EVALUATION_CONTEXT["cells"][cell_index]
    activation, metadata = cell["observations"][observation_index]
    responses = split_projection_responses(cell["q2"], cell["q4"], activation)
    old_outputs = factorized_unit_outputs(cell["q2"], cell["q4"], activation)
    base_damage = split_exact_damage(
        responses, np.zeros(UNITS, np.int64), proxy=cell["proxy"], beta=cell["beta"],
    )
    if not np.isfinite(base_damage) or base_damage <= 0.0:
        raise RuntimeError("base damage must be positive and finite")
    rows = []
    identity = tuple(metadata[name] for name in IDENTITY)
    global_context = _EVALUATION_CONTEXT
    previous_proxy = global_context.get("cell_proxy")
    previous_beta = global_context.get("cell_beta")
    global_context["cell_proxy"], global_context["cell_beta"] = cell["proxy"], cell["beta"]
    try:
        for budget in map(int, global_context["config"]["page_budgets"]):
            exact_reference = global_context["references"][
                identity + (budget, global_context["config"]["factor_config_ids"][0])
            ]
            rows.extend(_exact_gram_rows(
                responses, cell["abc"], cell["gram"], budget, metadata,
                exact_reference, base_damage, global_context["config"],
            ))
            for entry, factor in cell["factors"]:
                reference = global_context["references"][
                    identity + (budget, entry["factor_config_id"])
                ]
                old_started = time.perf_counter()
                old, old_seed = _old_solution(
                    old_outputs, cell["abc"], factor, budget, global_context["config"],
                )
                old_seconds = time.perf_counter() - old_started
                old_split_states = map_four_state_to_split(old.states)
                old_exact = split_exact_damage(
                    responses, old_split_states, proxy=cell["proxy"], beta=cell["beta"],
                )
                old_recovery = 1.0 - old_exact / base_damage
                if not np.isclose(old_recovery, reference["recovery"], rtol=1e-7, atol=2e-8):
                    raise RuntimeError("recomputed four-state reference changed from PR #11")
                old_components = {"inherited_pr11_selector_macs": int(reference["selector_compute_macs"])}
                rows.append(_row(
                    metadata,
                    factor_entry=entry,
                    solver=global_context["config"]["four_state_reference_solver_id"],
                    states=old_split_states,
                    budget=budget,
                    exact_damage=old_exact,
                    base_damage=base_damage,
                    reference=reference,
                    predicted_damage=build_split_interaction_field(
                        factor, responses.hidden, cell["abc"],
                    ).damage(old_split_states),
                    runtime_seconds=old_seconds,
                    selector_macs=int(reference["selector_compute_macs"]),
                    components=old_components,
                    seed=old_seed,
                    coordinate_sweeps=0,
                    local_passes=0,
                    exact_gram_oracle=False,
                ))
                field = build_split_interaction_field(factor, responses.hidden, cell["abc"])
                for inherited, solver in (
                    (False, global_context["config"]["primary_solver_id"]),
                    (True, global_context["config"]["warm_start_solver_id"]),
                ):
                    started = time.perf_counter()
                    trace, seed, sweeps, passes = _split_solver(
                        field, budget, old.states, global_context["config"], inherited=inherited,
                    )
                    seconds = time.perf_counter() - started
                    selector_macs, components = _split_compute(field.rank, sweeps, passes)
                    if inherited:
                        components["inherited_pr11_selector_macs"] = int(
                            reference["selector_compute_macs"]
                        )
                        selector_macs += int(reference["selector_compute_macs"])
                    exact = split_exact_damage(
                        responses, trace.states, proxy=cell["proxy"], beta=cell["beta"],
                    )
                    rows.append(_row(
                        metadata,
                        factor_entry=entry,
                        solver=solver,
                        states=trace.states,
                        budget=budget,
                        exact_damage=exact,
                        base_damage=base_damage,
                        reference=reference,
                        predicted_damage=trace.damage,
                        runtime_seconds=seconds,
                        selector_macs=selector_macs,
                        components=components,
                        seed=seed,
                        coordinate_sweeps=sweeps,
                        local_passes=passes,
                        exact_gram_oracle=False,
                    ))
    finally:
        if previous_proxy is None:
            global_context.pop("cell_proxy", None)
        else:
            global_context["cell_proxy"] = previous_proxy
        if previous_beta is None:
            global_context.pop("cell_beta", None)
        else:
            global_context["cell_beta"] = previous_beta
    return cell_index, observation_index, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pr11-config", type=Path, required=True)
    parser.add_argument("--pr10-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trees", type=Path, required=True)
    parser.add_argument("--exact-captures", type=Path, required=True)
    parser.add_argument("--pr11-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    old = json.loads(args.pr10_config.read_text())
    _validate_contract(config, old)
    frozen = {
        args.pr11_config: config["pr11_config_sha256"],
        EXPERIMENT / "scripts/run_interaction_field_study.py": config["pr11_runner_sha256"],
        EXPERIMENT / "src/oracle_study/interaction_field.py": config["pr11_core_sha256"],
        args.pr11_dir / "run_facts.json": config["pr11_run_facts_sha256"],
        args.pr11_dir / "interaction_field_factor_manifest.json": config["pr11_factor_manifest_sha256"],
        args.pr11_dir / "interaction_field_frontier.parquet": config["pr11_frontier_sha256"],
        args.pr10_config: config["pr10_config_sha256"],
    }
    for path, expected in frozen.items():
        if prior.sha256(path) != expected:
            raise RuntimeError(f"frozen dependency hash changed: {path}")
    references = _load_pr11_references(
        args.pr11_dir / "interaction_field_frontier.parquet", config,
    )
    audit, hashes = base.verify_locked_inputs(
        old, args.pr10_config, args.exact_captures, args.checkpoint, args.trees,
        "exact_checkpoint",
    )
    data = prior.load_capture_admitted(args.exact_captures, ["train", "validation"])
    if "test" in set(map(str, np.unique(data["split"]))):
        raise RuntimeError("test row entered split interaction study")
    plan = prior.evaluation_plan(data, old)
    scope = prior.audit_validation_plan(plan, old)
    threads = {
        name: os.environ.get(name)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    }
    if set(threads.values()) != {str(config["worker_blas_threads"])}:
        raise RuntimeError("worker BLAS thread environment is not frozen")
    capacity = _cpu_capacity_facts()
    if capacity["quota_cores"] is not None and capacity["quota_cores"] < config["evaluation_workers"]:
        raise RuntimeError("CPU quota is below the frozen worker count")
    args.output.mkdir(parents=True, exist_ok=False)
    facts_path = args.output / FACTS
    facts = {
        "completed": False,
        "failures": [],
        "completed_work_units": [],
        "run_id": config["run_id"],
        "study_scope": config["study_scope"],
        "selection_split": "validation",
        "test_rows_admitted": False,
        "test_scientific_values_used": False,
        "validation_scope_audit": scope,
        "checkpoint_audit": audit,
        "capture_sha256": hashes["capture_sha256"],
        "tree_sha256": hashes["tree_sha256"],
        "checkpoint_config_sha256": hashes["config_sha256"],
        "checkpoint_index_sha256": hashes["index_sha256"],
        "config_sha256": prior.sha256(args.config),
        "pr11_run_facts_sha256": prior.sha256(args.pr11_dir / "run_facts.json"),
        "pr11_factor_manifest_sha256": prior.sha256(
            args.pr11_dir / "interaction_field_factor_manifest.json"
        ),
        "pr11_frontier_sha256": prior.sha256(
            args.pr11_dir / "interaction_field_frontier.parquet"
        ),
        "runner_sha256": prior.sha256(Path(__file__).resolve()),
        "split_core_sha256": prior.sha256(
            EXPERIMENT / "src/oracle_study/split_interaction_field.py"
        ),
        "runtime_provenance": prior.runtime_provenance(args.device),
        "cpu_capacity": capacity,
        "evaluation_workers": int(config["evaluation_workers"]),
        "worker_start_method": config["worker_start_method"],
        "worker_thread_environment": threads,
        "state_page_costs": SPLIT_STATE_PAGE_COSTS.tolist(),
        "inherited_four_state_map": INHERITED_FOUR_STATE_MAP.tolist(),
    }
    _atomic_json(facts_path, facts)
    try:
        manifest = json.loads(
            (args.pr11_dir / "interaction_field_factor_manifest.json").read_text()
        )
        selected_entries = [
            entry for entry in manifest["entries"]
            if entry["factor_config_id"] in set(config["factor_config_ids"])
        ]
        if len(selected_entries) != 12 * len(config["factor_config_ids"]):
            raise RuntimeError("selected PR #11 factor grid is incomplete")
        arrays_by_layer = {}
        for layer, record in manifest["layers"].items():
            path = args.pr11_dir / record["file"]
            if prior.sha256(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
                raise RuntimeError("PR #11 factor shard changed")
            arrays_by_layer[int(layer)] = dict(np.load(path, allow_pickle=False))

        index = json.loads(
            (args.checkpoint / "model.safetensors.index.json").read_text()
        )["weight_map"]
        tree_records = json.loads(args.trees.read_text())
        trees = {name: tree_from_record(tree_records[name]) for name in prior.PROJECTIONS}
        cells = []
        tasks = []
        expected_observations = []
        for cell_index, (layer, expert, stratum, records, ranks) in enumerate(plan):
            local = prior.layer_view(data, layer)
            proxy, proxy_facts = base.proxy_gradients(
                local["xplus"], [local[f"h{h}_router_logits"] for h in range(1, 5)],
                local["split"],
            )
            beta = float(proxy_facts["beta"])
            matrices = prior.decode_expert(args.checkpoint, index, trees, layer, expert)
            q2 = tuple(matrices[name][0] for name in prior.PROJECTIONS)
            q4 = tuple(matrices[name][2] for name in prior.PROJECTIONS)
            abc = unit_score_metadata(q2[2], q4[2], proxy=proxy, beta=beta)
            gram = down_metric_gram(q2[2], q4[2], proxy=proxy, beta=beta, dtype=np.float64)
            entries = sorted(
                (
                    entry for entry in selected_entries
                    if int(entry["layer"]) == layer and int(entry["expert_id"]) == expert
                ),
                key=lambda entry: entry["factor_config_id"],
            )
            factors = tuple(
                (entry, pr11._load_factor(arrays_by_layer[layer], entry))
                for entry in entries
            )
            observations = []
            for record, router_rank in zip(records, ranks):
                metadata = base.common_metadata(
                    "exact_checkpoint", "validation", local, int(record), int(router_rank),
                    layer, expert, stratum,
                )
                observations.append((np.asarray(local["x"][record], np.float32), metadata))
            cells.append({
                "layer": layer, "expert": expert, "q2": q2, "q4": q4,
                "abc": abc, "gram": gram, "proxy": proxy, "beta": beta,
                "factors": factors, "observations": tuple(observations),
            })
            expected_observations.append(len(observations))
            tasks.extend((cell_index, offset) for offset in range(len(observations)))

        global _EVALUATION_CONTEXT
        _EVALUATION_CONTEXT = {
            "cells": tuple(cells), "config": config, "references": references,
        }
        facts["parallel_task_count"] = len(tasks)
        facts["effective_evaluation_workers"] = min(config["evaluation_workers"], len(tasks))
        _atomic_json(facts_path, facts)
        rows = []
        pending = [dict() for _ in cells]
        next_cell = 0
        context = mp.get_context(config["worker_start_method"])
        with ProcessPoolExecutor(
            max_workers=facts["effective_evaluation_workers"], mp_context=context,
        ) as executor:
            futures = {executor.submit(_evaluate_task, task): task for task in tasks}
            for future in as_completed(futures):
                cell_index, observation_index, task_rows = future.result()
                if (cell_index, observation_index) != futures[future]:
                    raise RuntimeError("worker task identity changed")
                pending[cell_index][observation_index] = task_rows
                while next_cell < len(cells):
                    if len(pending[next_cell]) != expected_observations[next_cell]:
                        break
                    for offset in range(expected_observations[next_cell]):
                        rows.extend(pending[next_cell][offset])
                    layer = cells[next_cell]["layer"]
                    expert = cells[next_cell]["expert"]
                    facts["completed_work_units"].append(f"validation:{layer}:{expert}")
                    _atomic_parquet(args.output / FRONTIER, rows)
                    facts["frontier_rows"] = len(rows)
                    _atomic_json(facts_path, facts)
                    next_cell += 1
                    gc.collect()
        _EVALUATION_CONTEXT = None
        if next_cell != len(cells):
            raise RuntimeError("parallel evaluation did not close every cell")
        per_identity = 3 * (2 + 3 * len(config["factor_config_ids"]))
        expected_rows = config["expected_validation_unique_invocations"] * per_identity
        if len(rows) != expected_rows:
            raise RuntimeError(f"frontier rows {len(rows)} != {expected_rows}")
        identities = {tuple(row[name] for name in IDENTITY) for row in rows}
        if len(identities) != config["expected_validation_unique_invocations"]:
            raise RuntimeError("validation identity coverage changed")
        primary = [
            row for row in rows
            if row["solver"] == config["primary_solver_id"]
            and row["factor_config_id"] in config["primary_factor_config_ids"]
        ]
        if len(primary) != config["expected_validation_unique_invocations"] * 3 * 2:
            raise RuntimeError("primary grid incomplete")
        facts.update({
            "completed": True,
            "frontier_rows": len(rows),
            "frontier_sha256": prior.sha256(args.output / FRONTIER),
            "completed_unix": time.time(),
        })
        _atomic_json(facts_path, facts)
    except Exception as error:
        facts["failures"].append({"type": type(error).__name__, "message": str(error)})
        _atomic_json(facts_path, facts)
        raise


if __name__ == "__main__":
    main()
