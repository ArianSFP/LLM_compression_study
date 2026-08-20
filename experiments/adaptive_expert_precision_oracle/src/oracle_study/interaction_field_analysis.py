"""Fail-closed analysis for the exact-H4 low-rank interaction-field study."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


IDENTITY = (
    "capture_source", "evaluation_split", "request_id", "position", "layer", "expert_id",
)
PRIMARY_SOLVER = "two_seed_coordinate_plus_local_repair"
GEOMETRY_SOLVER = "four_seed_coordinate_plus_local_repair"
CONTROL_SOLVERS = {
    "compressed_residual_forward",
    "continuous_relaxation_round",
    "four_seed_coordinate",
}
ABC_BYTES = 3_084
EXPERT_WEIGHTS = 3_145_728
UNITS = 512
COMPUTE_COMPONENTS = (
    "field_build", "independent_seed", "forward", "relaxation_setup",
    "relaxation_iterations", "coordinate", "local",
)
OPERATION_COUNTERS = (
    "forward_candidate_evaluations", "relaxation_iterations",
    "coordinate_sweeps", "local_evaluated_passes",
)
EXPECTED_COMPUTE_BOUNDS = {
    "compressed_residual_forward": (
        "complete_field_build_plus_shared_max_budget_forward_upper_bound"
    ),
    "continuous_relaxation_round": (
        "Frobenius_safe_step_setup_plus_projected_gradient_logical_MACs"
    ),
    "four_seed_coordinate": (
        "complete_field_and_all_seed_construction_plus_two_dot_coordinate_MACs"
    ),
    GEOMETRY_SOLVER: (
        "complete_field_build_plus_conservative_local_upper_bound_and_all_seed_costs"
    ),
    PRIMARY_SOLVER: (
        "complete_field_build_plus_two_seed_coordinate_and_conservative_local_upper_bound"
    ),
}
EXPECTED_NON_MAC_CONVENTION = (
    "state_scalar_additions_projection_simplex_sort_exact_page_DP_round_"
    "topk_and_control_flow_excluded_from_analytical_MAC_metric"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values: pd.Series | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, np.float64)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise RuntimeError("summary input must be a nonempty finite vector")
    return {
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
    }


def _factor_id(family: str, tail_rank: int, exact_rank: int, encoding: str) -> str:
    return f"{family}_tail{int(tail_rank)}_exact{int(exact_rank)}_{encoding}"


def expected_factor_ids(config: Mapping[str, Any]) -> set[str]:
    result = {
        _factor_id(family, rank, 0, "fp32")
        for family in ("joint_eigh", "joint_pivoted_cholesky")
        for rank in map(int, config["tail_ranks"])
    }
    result |= {
        _factor_id("exact_proxy_plus_eigh_tail", rank, 4, "fp32")
        for rank in map(int, config["exact_proxy_tail_ranks"])
    }
    for spec in config["quantized_factor_specs"]:
        suffix = str(spec["encoding"]) + ("_hadamard" if spec["hadamard"] else "")
        result.add(_factor_id(str(spec["source"]), int(spec["tail_rank"]), 0, suffix))
    return result


def _physical_factor_bytes(row: Mapping[str, Any]) -> int:
    rank = int(row["total_rank"])
    encoding = str(row["encoding"])
    values = 2 * 512 * rank
    if encoding == "fp32":
        return 4 * values
    if encoding == "int8_per_row":
        return values + 2 * 1024 + 4
    if encoding == "int4_per_row":
        return (values + 1) // 2 + 2 * 1024 + 4
    raise RuntimeError(f"unknown factor encoding {encoding!r}")

def _validate_compute_accounting(
    frontier: pd.DataFrame, config: Mapping[str, Any],
) -> None:
    component_columns = [
        f"selector_{name}_compute_macs" for name in COMPUTE_COMPONENTS
    ]
    integral_columns = ["selector_compute_macs", *component_columns, *OPERATION_COUNTERS]
    values = frontier[integral_columns].to_numpy(np.float64)
    if np.any(values < 0.0) or np.any(values != np.rint(values)):
        raise RuntimeError("selector compute evidence must be nonnegative integers")
    if np.any(
        frontier["selector_compute_macs"].to_numpy(np.int64)
        != frontier[component_columns].sum(axis=1).to_numpy(np.int64)
    ):
        raise RuntimeError("selector compute components do not sum to the total")
    rank = frontier["total_rank"].to_numpy(np.int64)
    solver = frontier["solver"].astype(str)
    if set(solver) != set(EXPECTED_COMPUTE_BOUNDS):
        raise RuntimeError("solver family changed")
    coordinate_mask = solver.isin({
        "four_seed_coordinate", PRIMARY_SOLVER, GEOMETRY_SOLVER,
    }).to_numpy()
    relaxation_mask = solver.isin({
        "continuous_relaxation_round", "four_seed_coordinate", GEOMETRY_SOLVER,
    }).to_numpy()
    forward_mask = solver.isin({
        "compressed_residual_forward", "four_seed_coordinate", GEOMETRY_SOLVER,
    }).to_numpy()
    local_mask = solver.isin({PRIMARY_SOLVER, GEOMETRY_SOLVER}).to_numpy()
    counter_masks = {
        "forward_candidate_evaluations": forward_mask,
        "relaxation_iterations": relaxation_mask,
        "coordinate_sweeps": coordinate_mask,
        "local_evaluated_passes": local_mask,
    }
    for name, mask in counter_masks.items():
        if np.any(frontier[name].to_numpy(np.int64)[~mask] != 0):
            raise RuntimeError(f"{name} is nonzero for an inapplicable solver")
    expected = {
        "field_build": 8 * UNITS * rank + 12 * UNITS,
        "independent_seed": np.where(coordinate_mask, 6 * UNITS, 0),
        "forward": 2 * rank * frontier["forward_candidate_evaluations"].to_numpy(np.int64),
        "relaxation_setup": np.where(relaxation_mask, 4 * UNITS * rank, 0),
        "relaxation_iterations": (
            8 * UNITS * rank * frontier["relaxation_iterations"].to_numpy(np.int64)
        ),
        "coordinate": (
            (2 * UNITS * rank + 8 * UNITS)
            * frontier["coordinate_sweeps"].to_numpy(np.int64)
        ),
        "local": (
            frontier["local_evaluated_passes"].to_numpy(np.int64)
            * (
                6 * UNITS * rank
                + (7 * int(config["local_shortlist"])) ** 2 * rank
            )
        ),
    }
    for name, expected_values in expected.items():
        actual = frontier[f"selector_{name}_compute_macs"].to_numpy(np.int64)
        if np.any(actual != expected_values):

            raise RuntimeError(f"selector {name} compute arithmetic changed")
    expected_bounds = solver.map(EXPECTED_COMPUTE_BOUNDS)
    if np.any(frontier["selector_compute_accounting_bound"].astype(str) != expected_bounds):
        raise RuntimeError("selector compute accounting convention changed")
    if set(frontier["selector_non_mac_operations"].astype(str)) != {
        EXPECTED_NON_MAC_CONVENTION
    }:
        raise RuntimeError("selector non-MAC convention changed")


@dataclass(frozen=True)
class Evidence:
    config: dict[str, Any]
    facts: dict[str, Any]
    manifest: dict[str, Any]
    frontier: pd.DataFrame
    evidence_dir: Path


def validate_evidence(
    config_path: Path,
    evidence_dir: Path,
    *,
    runner_path: Path | None = None,
    core_path: Path | None = None,
) -> Evidence:
    config = json.loads(config_path.read_text())
    facts_path = evidence_dir / "run_facts.json"
    manifest_path = evidence_dir / "interaction_field_factor_manifest.json"
    frontier_path = evidence_dir / "interaction_field_frontier.parquet"
    facts = json.loads(facts_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if facts.get("completed") is not True or facts.get("failures") != []:
        raise RuntimeError("interaction evidence is incomplete or failed")
    if facts.get("test_rows_admitted") is not False or facts.get("test_scientific_values_used") is not False:
        raise RuntimeError("test scientific boundary changed")
    workers = int(config["evaluation_workers"])
    if int(facts.get("evaluation_workers", -1)) != workers:
        raise RuntimeError("evaluation worker provenance changed")
    if facts.get("worker_start_method") != config["worker_start_method"]:
        raise RuntimeError("worker start-method provenance changed")
    if int(facts.get("worker_blas_threads", -1)) != int(config["worker_blas_threads"]):
        raise RuntimeError("worker BLAS-thread provenance changed")
    expected_environment = {
        name: str(int(config["worker_blas_threads"]))
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    }
    if facts.get("worker_thread_environment") != expected_environment:
        raise RuntimeError("worker thread environment changed")
    capacity = facts.get("cpu_capacity", {})
    quota = capacity.get("quota_cores")
    if quota is not None and float(quota) < workers:
        raise RuntimeError("recorded CPU quota is below the worker count")
    expected_tasks = int(config["expected_validation_unique_invocations"])
    if int(facts.get("parallel_task_count", -1)) != expected_tasks:
        raise RuntimeError("parallel evaluation task count changed")
    if int(facts.get("effective_evaluation_workers", -1)) != min(workers, expected_tasks):
        raise RuntimeError("effective evaluation worker count changed")
    if facts.get("config_sha256") != sha256(config_path):
        raise RuntimeError("interaction config hash changed")
    if facts.get("factor_manifest_sha256") != sha256(manifest_path):
        raise RuntimeError("factor manifest hash changed")
    if facts.get("frontier_sha256") != sha256(frontier_path):
        raise RuntimeError("interaction frontier hash changed")
    if runner_path is not None and facts.get("runner_sha256") != sha256(runner_path):
        raise RuntimeError("interaction runner hash changed")
    if core_path is not None and facts.get("interaction_core_sha256") != sha256(core_path):
        raise RuntimeError("interaction core hash changed")
    for key in ("config_sha256", "runner_sha256", "interaction_core_sha256"):
        if manifest.get(key) != facts.get(key):
            raise RuntimeError(f"manifest/facts provenance mismatch for {key}")
    if manifest.get("fit_split") != "train" or manifest.get(
        "validation_or_test_scientific_values_used"
    ) is not False:
        raise RuntimeError("factor fit did not remain train/static only")
    for layer, record in manifest["layers"].items():
        path = evidence_dir / str(record["file"])
        if path.name != str(record["file"]) or sha256(path) != record["sha256"]:
            raise RuntimeError(f"factor shard {layer} changed")
        if path.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"factor shard {layer} byte count changed")
        with np.load(path, allow_pickle=False) as loaded:
            if len(loaded.files) != int(record["array_count"]):
                raise RuntimeError(f"factor shard {layer} array count changed")

    expected_ids = expected_factor_ids(config)
    entries = pd.DataFrame(manifest["entries"])
    if len(entries) != int(config["expected_validation_layer_expert_cells"]) * len(expected_ids):
        raise RuntimeError("factor manifest grid size changed")
    if entries.duplicated(["layer", "expert_id", "factor_config_id"]).any():
        raise RuntimeError("duplicate factor manifest entry")
    for _, group in entries.groupby(["layer", "expert_id"], sort=False):
        if set(group["factor_config_id"].astype(str)) != expected_ids:
            raise RuntimeError("factor manifest configuration grid changed")
    for row in entries.to_dict("records"):
        if int(row["factor_payload_bytes"]) != _physical_factor_bytes(row):
            raise RuntimeError("factor physical payload arithmetic changed")
        if row["encoding"] == "fp32" and row["geometry_ceiling"] is not True:
            raise RuntimeError("FP32 geometry ceiling marker changed")
        if row["encoding"] != "fp32" and row["quantized_deployable_sidecar"] is not True:
            raise RuntimeError("quantized sidecar marker changed")

    frontier = pd.read_parquet(frontier_path)
    required = {
        *IDENTITY, "factor_config_id", "solver", "physical_budget_pages", "physical_pages",
        "recovery", "set_gain_retention_vs_pr10_exact_hybrid",
        "recovery_gap_vs_pr10_exact_hybrid", "pr10_exact_hybrid_recovery",
        "compressed_predicted_damage", "exact_qenergy_damage",
        "damage_prediction_error_over_base", "selector_compute_macs",
        "factor_payload_bytes", "combined_metadata_bytes", "combined_metadata_bpw",
        *(f"selector_{name}_compute_macs" for name in COMPUTE_COMPONENTS),
        *OPERATION_COUNTERS, "selector_compute_accounting_bound",
        "selector_non_mac_operations",
    }
    missing = required - set(frontier.columns)
    if missing:
        raise RuntimeError(f"frontier is missing columns {sorted(missing)}")
    numeric = [
        "recovery", "set_gain_retention_vs_pr10_exact_hybrid",
        "recovery_gap_vs_pr10_exact_hybrid", "pr10_exact_hybrid_recovery",
        "compressed_predicted_damage", "exact_qenergy_damage",
        "damage_prediction_error_over_base", "selector_compute_macs",
        "factor_payload_bytes", "combined_metadata_bytes", "combined_metadata_bpw",
    ]
    numeric += [f"selector_{name}_compute_macs" for name in COMPUTE_COMPONENTS]
    numeric += list(OPERATION_COUNTERS)
    if not np.all(np.isfinite(frontier[numeric].to_numpy(np.float64))):
        raise RuntimeError("frontier contains non-finite scientific/accounting evidence")
    _validate_compute_accounting(frontier, config)
    if set(frontier["capture_source"].astype(str)) != {"exact_checkpoint"}:
        raise RuntimeError("capture source changed")
    if set(frontier["evaluation_split"].astype(str)) != {"validation"}:
        raise RuntimeError("evaluation split changed")
    if set(map(int, frontier["physical_budget_pages"].unique())) != set(
        map(int, config["page_budgets"])
    ):
        raise RuntimeError("frontier budget grid changed")
    if np.any(frontier["physical_pages"] > frontier["physical_budget_pages"]):
        raise RuntimeError("a solver exceeded its physical page budget")
    if np.any(np.abs(
        frontier["set_gain_retention_vs_pr10_exact_hybrid"]
        - frontier["recovery"] / frontier["pr10_exact_hybrid_recovery"]
    ) > 1e-10):
        raise RuntimeError("set-gain retention arithmetic changed")
    if np.any(frontier["combined_metadata_bytes"] != frontier["factor_payload_bytes"] + ABC_BYTES):
        raise RuntimeError("combined metadata byte arithmetic changed")
    expected_bpw = 8.0 * frontier["combined_metadata_bytes"] / EXPERT_WEIGHTS
    if np.any(np.abs(frontier["combined_metadata_bpw"] - expected_bpw) > 1e-12):
        raise RuntimeError("combined metadata bpw arithmetic changed")

    primary = frontier[frontier["solver"] == PRIMARY_SOLVER]
    expected_primary = (
        int(config["expected_validation_unique_invocations"])
        * len(config["page_budgets"]) * len(expected_ids)
    )
    if len(primary) != expected_primary:
        raise RuntimeError(f"primary grid {len(primary)} != {expected_primary}")
    if primary.duplicated([*IDENTITY, "physical_budget_pages", "factor_config_id"]).any():
        raise RuntimeError("duplicate primary field row")
    for identity, group in primary.groupby(list(IDENTITY), sort=False):
        if len(group) != len(config["page_budgets"]) * len(expected_ids):
            raise RuntimeError(f"incomplete primary identity {identity!r}")
        if set(group["factor_config_id"].astype(str)) != expected_ids:
            raise RuntimeError("primary factor grid changed")
    if primary[list(IDENTITY)].drop_duplicates().shape[0] != int(
        config["expected_validation_unique_invocations"]
    ):
        raise RuntimeError("primary invocation coverage changed")

    geometry = frontier[frontier["solver"] == GEOMETRY_SOLVER]
    if len(geometry) != expected_primary:
        raise RuntimeError(f"geometry grid {len(geometry)} != {expected_primary}")
    if geometry.duplicated([*IDENTITY, "physical_budget_pages", "factor_config_id"]).any():
        raise RuntimeError("duplicate geometry field row")
    for identity, group in geometry.groupby(list(IDENTITY), sort=False):
        if len(group) != len(config["page_budgets"]) * len(expected_ids):
            raise RuntimeError(f"incomplete geometry identity {identity!r}")
        if set(group["factor_config_id"].astype(str)) != expected_ids:
            raise RuntimeError("geometry factor grid changed")
    if set(map(tuple, geometry[list(IDENTITY)].to_numpy())) != set(
        map(tuple, primary[list(IDENTITY)].to_numpy())
    ):
        raise RuntimeError("primary/geometry invocation coverage differs")

    controls = frontier[frontier["solver"].isin(CONTROL_SOLVERS)]
    expected_controls = (
        int(config["expected_validation_unique_invocations"])
        * len(config["page_budgets"]) * len(config["solver_control_factor_ids"])
        * len(CONTROL_SOLVERS)
    )
    if len(controls) != expected_controls:
        raise RuntimeError(f"solver-control grid {len(controls)} != {expected_controls}")
    if set(controls["factor_config_id"].astype(str)) != set(config["solver_control_factor_ids"]):
        raise RuntimeError("solver-control factor IDs changed")
    return Evidence(config, facts, manifest, frontier, evidence_dir)


def summarize(evidence: Evidence) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    config = evidence.config
    primary = evidence.frontier[evidence.frontier["solver"] == PRIMARY_SOLVER].copy()
    compared = evidence.frontier[
        evidence.frontier["solver"].isin({PRIMARY_SOLVER, GEOMETRY_SOLVER})
    ].copy()
    factor_rows = []
    for (solver, factor_id, budget), group in compared.groupby(
        ["solver", "factor_config_id", "physical_budget_pages"], sort=True,
    ):
        recovery = quantiles(group["recovery"])
        retention = quantiles(group["set_gain_retention_vs_pr10_exact_hybrid"])
        gap = quantiles(group["recovery_gap_vs_pr10_exact_hybrid"])
        factor_rows.append({
            "factor_config_id": factor_id, "physical_budget_pages": int(budget),
            "solver": solver,
            "factor_family": str(group["factor_family"].iloc[0]),
            "tail_rank": int(group["tail_rank"].iloc[0]),
            "exact_rank": int(group["exact_rank"].iloc[0]),
            "total_rank": int(group["total_rank"].iloc[0]),
            "encoding": str(group["encoding"].iloc[0]),
            "hadamard_rotated": bool(group["hadamard_rotated"].iloc[0]),
            "geometry_ceiling": bool(group["geometry_ceiling"].iloc[0]),
            "factor_payload_bytes": int(group["factor_payload_bytes"].iloc[0]),
            "combined_metadata_bpw": float(group["combined_metadata_bpw"].iloc[0]),
            "selector_compute_macs_max": int(group["selector_compute_macs"].max()),
            **{f"recovery_{key}": value for key, value in recovery.items()},
            **{f"retention_{key}": value for key, value in retention.items()},
            **{f"gap_{key}": value for key, value in gap.items()},
            "geometry_gate_pass": (
                retention["p10"] >= float(config["primary_gate_p10_set_gain_retention"])
                and gap["median"] >= -float(
                    config["primary_gate_median_recovery_gap_vs_exact_hybrid"]
                )
            ),
            "compute_gate_pass": int(group["selector_compute_macs"].max())
            < int(config["selector_compute_gate_macs"]),
            "metadata_gate_pass": float(group["combined_metadata_bpw"].max())
            <= float(config["metadata_gate_bpw"]),
        })
    factor_summary = pd.DataFrame(factor_rows)

    solver_rows = []
    controls = evidence.frontier[
        evidence.frontier["factor_config_id"].isin(config["solver_control_factor_ids"])
    ]
    for (factor_id, budget, solver), group in controls.groupby(
        ["factor_config_id", "physical_budget_pages", "solver"], sort=True,
    ):
        recovery = quantiles(group["recovery"])
        retention = quantiles(group["set_gain_retention_vs_pr10_exact_hybrid"])
        solver_rows.append({
            "factor_config_id": factor_id, "physical_budget_pages": int(budget),
            "solver": solver, **{f"recovery_{k}": v for k, v in recovery.items()},
            **{f"retention_{k}": v for k, v in retention.items()},
            "selector_compute_macs_max": int(group["selector_compute_macs"].max()),
        })
    solver_summary = pd.DataFrame(solver_rows)

    layer_rows = []
    for (factor_id, budget, layer), group in primary.groupby(
        ["factor_config_id", "physical_budget_pages", "layer"], sort=True,
    ):
        recovery = quantiles(group["recovery"])
        retention = quantiles(group["set_gain_retention_vs_pr10_exact_hybrid"])
        layer_rows.append({
            "factor_config_id": factor_id, "physical_budget_pages": int(budget),
            "layer": int(layer), **{f"recovery_{k}": v for k, v in recovery.items()},
            **{f"retention_{k}": v for k, v in retention.items()},
        })
    layer_summary = pd.DataFrame(layer_rows)

    one_bpw = factor_summary[
        (factor_summary["physical_budget_pages"] == 768)
        & (factor_summary["solver"] == GEOMETRY_SOLVER)
    ].copy()
    primary_one_bpw = factor_summary[
        (factor_summary["physical_budget_pages"] == 768)
        & (factor_summary["solver"] == PRIMARY_SOLVER)
    ].copy()
    best = one_bpw.sort_values(
        ["retention_p10", "recovery_median", "combined_metadata_bpw", "factor_config_id"],
        ascending=[False, False, True, True],
    ).iloc[0]
    quantized = one_bpw[one_bpw["encoding"] != "fp32"]
    best_quantized = quantized.sort_values(
        ["retention_p10", "recovery_median", "combined_metadata_bpw", "factor_config_id"],
        ascending=[False, False, True, True],
    ).iloc[0]
    best_primary = primary_one_bpw.sort_values(
        ["retention_p10", "recovery_median", "combined_metadata_bpw", "factor_config_id"],
        ascending=[False, False, True, True],
    ).iloc[0]
    primary_quantized = primary_one_bpw[primary_one_bpw["encoding"] != "fp32"]
    best_primary_quantized = primary_quantized.sort_values(
        ["retention_p10", "recovery_median", "combined_metadata_bpw", "factor_config_id"],
        ascending=[False, False, True, True],
    ).iloc[0]
    fully_gated = primary_quantized[
        primary_quantized["geometry_gate_pass"]
        & primary_quantized["compute_gate_pass"]
        & primary_quantized["metadata_gate_pass"]
    ]
    best_fully_gated = None if fully_gated.empty else fully_gated.sort_values(
        ["retention_p10", "recovery_median", "combined_metadata_bpw", "factor_config_id"],
        ascending=[False, False, True, True],
    ).iloc[0]
    conclusion = {
        "study_scope": config["study_scope"],
        "best_exact_h4_geometry_configuration": str(best["factor_config_id"]),
        "best_exact_h4_geometry_metrics": {
            key: float(best[key]) for key in (
                "recovery_p10", "recovery_median", "retention_p10", "retention_median",
                "gap_median", "combined_metadata_bpw",
            )
        },
        "best_quantized_configuration": str(best_quantized["factor_config_id"]),
        "best_quantized_metrics": {
            key: float(best_quantized[key]) for key in (
                "recovery_p10", "recovery_median", "retention_p10", "retention_median",
                "gap_median", "combined_metadata_bpw",
            )
        },
        "best_two_seed_configuration": str(best_primary["factor_config_id"]),
        "best_two_seed_metrics": {
            key: float(best_primary[key]) for key in (
                "recovery_p10", "recovery_median", "retention_p10", "retention_median",
                "gap_median", "combined_metadata_bpw", "selector_compute_macs_max",
            )
        },
        "best_quantized_two_seed_configuration": str(
            best_primary_quantized["factor_config_id"]
        ),
        "best_quantized_two_seed_metrics": {
            key: float(best_primary_quantized[key]) for key in (
                "recovery_p10", "recovery_median", "retention_p10", "retention_median",
                "gap_median", "combined_metadata_bpw", "selector_compute_macs_max",
            )
        },
        "best_quantized_full_gate_configuration": (
            None if best_fully_gated is None else str(best_fully_gated["factor_config_id"])
        ),
        "best_quantized_full_gate_metrics": (
            None if best_fully_gated is None else {
                key: float(best_fully_gated[key]) for key in (
                    "recovery_p10", "recovery_median", "retention_p10",
                    "retention_median", "gap_median", "combined_metadata_bpw",
                    "selector_compute_macs_max",
                )
            }
        ),
        "any_geometry_gate_pass": bool(one_bpw["geometry_gate_pass"].any()),
        "any_quantized_geometry_gate_pass": bool(quantized["geometry_gate_pass"].any()),
        "any_two_seed_geometry_gate_pass": bool(
            primary_one_bpw["geometry_gate_pass"].any()
        ),
        "any_quantized_two_seed_geometry_gate_pass": bool(
            primary_quantized["geometry_gate_pass"].any()
        ),
        "any_quantized_full_compute_and_metadata_pass": bool((
            primary_quantized["geometry_gate_pass"]
            & primary_quantized["compute_gate_pass"]
            & primary_quantized["metadata_gate_pass"]
        ).any()),
        "deployment_status": "diagnostic_only_exact_h4_no_candidate_or_horizon_claim",
    }
    return factor_summary, solver_summary, layer_summary, conclusion
