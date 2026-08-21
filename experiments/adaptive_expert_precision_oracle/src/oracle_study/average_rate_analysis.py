"""Fail-closed analysis for grouped top-8 average-rate allocation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


GROUP_IDENTITY = (
    "capture_source", "evaluation_split", "request_id", "position", "layer",
)
POLICY_IDENTITY = (
    "factor_config_id", "allocation_policy",
    "mean_budget_pages_per_expert", "burst_cap_pages_per_expert",
)
FIT_FACTS = "average_rate_fit_facts.json"
FIT_MANIFEST = "average_rate_factor_manifest.json"
RUN_FACTS = "average_rate_run_facts.json"
GROUP_FRONTIER = "average_rate_group_frontier.parquet"
EXPERT_ALLOCATION = "average_rate_expert_allocation.parquet"
ACCOUNTING = "average_rate_accounting.json"
EXPERT_WEIGHTS = 3_145_728
PAGE_BYTES = 512


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, np.float64)
    if array.ndim != 1 or not len(array) or np.any(~np.isfinite(array)):
        raise ValueError("summary input must be a nonempty finite vector")
    return {
        "p10": float(np.quantile(array, .1)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, .9)),
        "mean": float(np.mean(array)),
    }


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{name} is missing columns: {missing}")


def _finite(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    _require_columns(frame, columns, name)
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
        if np.any(~np.isfinite(values)):
            raise RuntimeError(f"{name}.{column} contains non-finite evidence")


def _json_ints(value: Any, length: int, name: str) -> list[int]:
    try:
        result = json.loads(str(value))
    except Exception as error:
        raise RuntimeError(f"{name} is not canonical JSON") from error
    if (
        not isinstance(result, list) or len(result) != int(length)
        or any(isinstance(item, bool) or not isinstance(item, int) for item in result)
    ):
        raise RuntimeError(f"{name} must contain {length} integers")
    if json.dumps(result, separators=(",", ":")) != str(value):
        raise RuntimeError(f"{name} is not canonical compact JSON")
    return result


@dataclass(frozen=True)
class EvidenceBundle:
    config: dict[str, Any]
    fit_facts: dict[str, Any]
    fit_manifest: dict[str, Any]
    run_facts: dict[str, Any]
    accounting: dict[str, Any]
    groups: pd.DataFrame
    experts: pd.DataFrame
    input_paths: dict[str, Path]


def expected_grid(config: Mapping[str, Any]) -> set[tuple[str, str, int, int]]:
    primary = str(config["primary_factor_id"])
    control = str(config["compute_control_factor_id"])
    result: set[tuple[str, str, int, int]] = set()
    for mean in map(int, config["mean_correction_page_budgets"]):
        result.add((primary, "uniform_per_expert", mean, mean))
        for burst in map(int, config["primary_burst_caps_pages"]):
            result.add((primary, "pooled_router_square_compressed", mean, burst))
        result.add((primary, "pooled_equal_weight_compressed", mean, 1536))
        result.add((primary, "pooled_exact_combined_moe_oracle", mean, 1536))
    mean = int(config["compute_control_all_in_mean_pages"])
    result |= {
        (control, "uniform_per_expert", mean, mean),
        (control, "pooled_router_square_compressed", mean, 1536),
        (control, "pooled_exact_combined_moe_oracle", mean, 1536),
    }
    return result


def validate_evidence_bundle(
    config_path: Path, fit_dir: Path, validation_dir: Path,
    *, source_paths: Mapping[str, Path] | None = None,
) -> EvidenceBundle:
    """Validate provenance, exact grids, pages, bytes, and grouped identities."""
    config_path, fit_dir, validation_dir = map(Path, (config_path, fit_dir, validation_dir))
    paths = {
        "config": config_path,
        "fit_facts": fit_dir / FIT_FACTS,
        "fit_manifest": fit_dir / FIT_MANIFEST,
        "run_facts": validation_dir / RUN_FACTS,
        "validation_accounting": validation_dir / ACCOUNTING,
        "group_frontier": validation_dir / GROUP_FRONTIER,
        "expert_allocation": validation_dir / EXPERT_ALLOCATION,
    }
    for name, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"missing {name}: {path}")
    config = json.loads(config_path.read_text())
    fit_facts = json.loads(paths["fit_facts"].read_text())
    manifest = json.loads(paths["fit_manifest"].read_text())
    run_facts = json.loads(paths["run_facts"].read_text())
    accounting = json.loads(paths["validation_accounting"].read_text())
    groups = pd.read_parquet(paths["group_frontier"])
    experts = pd.read_parquet(paths["expert_allocation"])

    if fit_facts.get("completed") is not True or manifest.get("completed") is not True:
        raise RuntimeError("factor fit is incomplete")
    if run_facts.get("completed") is not True:
        raise RuntimeError("group evaluation is incomplete")
    for payload, label in ((fit_facts, "fit"), (run_facts, "evaluation")):
        if payload.get("failures") or payload.get("failure_history"):
            raise RuntimeError(f"{label} recorded a failure")
        if payload.get("test_scientific_rows_admitted_or_used") is not False:
            raise RuntimeError(f"{label} test-scientific boundary changed")
    if fit_facts.get("validation_rows_admitted_or_used") is not False:
        raise RuntimeError("fit admitted validation rows")
    if fit_facts.get("fit_split") != "train" or manifest.get("fit_split") != "train":
        raise RuntimeError("fit split changed")
    if run_facts.get("selection_split") != "validation":
        raise RuntimeError("selection split changed")

    hashes = (
        ("factor_manifest_sha256", paths["fit_manifest"], fit_facts),
        ("fit_facts_sha256", paths["fit_facts"], run_facts),
        ("fit_manifest_sha256", paths["fit_manifest"], run_facts),
        ("group_frontier_sha256", paths["group_frontier"], run_facts),
        ("expert_allocation_sha256", paths["expert_allocation"], run_facts),
        ("accounting_sha256", paths["validation_accounting"], run_facts),
    )
    for field, path, payload in hashes:
        if payload.get(field) != sha256(path):
            raise RuntimeError(f"artifact hash changed: {field}")
    if fit_facts.get("config_sha256") != sha256(config_path):
        raise RuntimeError("fit config hash changed")
    if run_facts.get("config_sha256") != sha256(config_path):
        raise RuntimeError("evaluation config hash changed")

    dependencies = (
        "runner_sha256", "allocator_core_sha256", "split_core_sha256",
        "interaction_core_sha256", "selector_core_sha256", "mxfp4_core_sha256",
        "set_utility_runner_sha256", "sparse_runner_sha256",
        "base_pr12_config_sha256", "base_pr12_runner_sha256",
        "base_pr12_run_facts_sha256", "base_pr12_frontier_sha256",
        "capture_sha256", "tree_sha256", "checkpoint_config_sha256",
        "checkpoint_index_sha256", "runtime_provenance",
    )
    for field in dependencies:
        values = fit_facts.get(field), manifest.get(field), run_facts.get(field)
        if any(value is None for value in values) or len({json.dumps(v, sort_keys=True) for v in values}) != 1:
            raise RuntimeError(f"fit/evaluation dependency mismatch: {field}")
    for field, path in (source_paths or {}).items():
        if field not in dependencies or run_facts.get(field) != sha256(Path(path)):
            raise RuntimeError(f"on-disk source hash changed: {field}")

    layers = list(map(int, config["layers"]))
    if fit_facts.get("completed_layers") != layers or run_facts.get("completed_layers") != layers:
        raise RuntimeError("fit/evaluation layer grid is incomplete")
    if set(manifest.get("layers", {})) != set(map(str, layers)):
        raise RuntimeError("fit manifest layer grid changed")
    for layer in layers:
        record = manifest["layers"][str(layer)]
        shard, sidecar = fit_dir / record["file"], fit_dir / record["sidecar"]
        if (
            not shard.is_file() or not sidecar.is_file()
            or shard.stat().st_size != int(record["bytes"])
            or sha256(shard) != record["sha256"]
            or sha256(sidecar) != record["sidecar_sha256"]
        ):
            raise RuntimeError(f"factor artifact changed for layer {layer}")
        arrays = np.load(shard, allow_pickle=False)
        shapes = {
            "primary_packed_codes": (256, 4096),
            "primary_row_scales": (256, 1024),
            "primary_global_scales": (256,),
            "rank4_l4": (256, 512, 4),
            "rank4_l2": (256, 512, 4),
        }
        for name, shape in shapes.items():
            if name not in arrays or arrays[name].shape != shape:
                raise RuntimeError(f"factor shape changed: layer {layer} {name}")

    group_columns = (
        *GROUP_IDENTITY, "sequence_id", *POLICY_IDENTITY,
        "actual_group_pages", "group_page_budget", "unused_group_pages",
        "average_actual_correction_pages", "average_actual_correction_bpw",
        "average_allowed_correction_bpw", "average_allowed_total_bpw",
        "average_actual_total_bpw", "combined_metadata_bytes_per_expert",
        "combined_metadata_bpw", "group_recovery", "group_exact_qenergy_damage",
        "group_base_qenergy_damage", "router_square_additive_recovery",
        "router_square_additive_damage", "router_square_additive_base_damage",
        "selector_runtime_ms", "selector_compute_macs",
        "selector_dp_state_evaluations", "selector_coordinate_sweeps",
        "selector_local_passes", "selected_expert_pages", "router_weights",
        "promotable", "continuation_eligible",
        "exact_cross_expert_information_used",
    )
    expert_columns = (
        *GROUP_IDENTITY, "sequence_id", *POLICY_IDENTITY,
        "expert_id", "router_rank", "router_weight", "selected_pages",
        "selected_states", "selected_state_counts", "exact_qenergy_damage",
        "base_qenergy_damage", "expert_recovery",
    )
    _require_columns(groups, group_columns, "group frontier")
    _require_columns(experts, expert_columns, "expert allocation")
    _finite(groups, (
        "mean_budget_pages_per_expert", "burst_cap_pages_per_expert",
        "actual_group_pages", "group_page_budget", "unused_group_pages",
        "average_actual_correction_pages", "average_actual_correction_bpw",
        "average_allowed_correction_bpw", "average_allowed_total_bpw",
        "average_actual_total_bpw", "combined_metadata_bytes_per_expert",
        "combined_metadata_bpw", "group_recovery", "group_exact_qenergy_damage",
        "group_base_qenergy_damage", "router_square_additive_recovery",
        "router_square_additive_damage", "router_square_additive_base_damage",
        "selector_runtime_ms", "selector_compute_macs",
        "selector_dp_state_evaluations", "selector_coordinate_sweeps",
        "selector_local_passes",
    ), "group frontier")
    _finite(experts, (
        "expert_id", "router_rank", "router_weight", "selected_pages",
        "exact_qenergy_damage", "base_qenergy_damage", "expert_recovery",
    ), "expert allocation")
    expected_rows = int(config["expected_validation_groups"]) * 27
    if len(groups) != expected_rows or len(experts) != 8 * expected_rows:
        raise RuntimeError("result row count changed")
    if groups[list(GROUP_IDENTITY) + list(POLICY_IDENTITY)].duplicated().any():
        raise RuntimeError("duplicate group row")
    if experts[list(GROUP_IDENTITY) + list(POLICY_IDENTITY) + ["router_rank"]].duplicated().any():
        raise RuntimeError("duplicate expert row")

    identities = groups[list(GROUP_IDENTITY)].drop_duplicates()
    if len(identities) != int(config["expected_validation_groups"]):
        raise RuntimeError("validation group identity count changed")
    counts = identities.groupby("layer").size().to_dict()
    expected_counts = {
        layer: int(config["expected_validation_groups_per_layer"]) for layer in layers
    }
    if counts != expected_counts:
        raise RuntimeError(f"validation group layer counts changed: {counts}")
    frozen_grid = expected_grid(config)
    for identity, frame in groups.groupby(list(GROUP_IDENTITY), sort=False):
        observed = set(map(tuple, frame[list(POLICY_IDENTITY)].itertuples(index=False, name=None)))
        if observed != frozen_grid:
            raise RuntimeError(f"policy grid changed for {identity}")

    expert_groups = experts.groupby(
        list(GROUP_IDENTITY) + list(POLICY_IDENTITY), sort=False,
    )
    for row in groups.itertuples(index=False):
        mean, pages = int(row.mean_budget_pages_per_expert), int(row.actual_group_pages)
        metadata = int(row.combined_metadata_bytes_per_expert)
        if int(row.group_page_budget) != 8 * mean or pages > 8 * mean:
            raise RuntimeError("group page cap arithmetic changed")
        if int(row.unused_group_pages) != 8 * mean - pages:
            raise RuntimeError("unused page arithmetic changed")
        checks = (
            (row.average_actual_correction_bpw, pages / (8.0 * 768.0)),
            (row.average_allowed_correction_bpw, mean / 768.0),
            (row.average_actual_total_bpw, (8 * metadata + pages * PAGE_BYTES) / EXPERT_WEIGHTS),
            (row.average_allowed_total_bpw, 8 * (metadata + mean * PAGE_BYTES) / EXPERT_WEIGHTS),
            (row.combined_metadata_bpw, 8.0 * metadata / EXPERT_WEIGHTS),
        )
        if any(not np.isclose(float(value), expected, rtol=0, atol=1e-12) for value, expected in checks):
            raise RuntimeError("group byte/bpw accounting changed")
        if float(row.group_base_qenergy_damage) <= 0 or float(row.router_square_additive_base_damage) <= 0:
            raise RuntimeError("recovery denominator is not positive")
        recovery = 1.0 - float(row.group_exact_qenergy_damage) / float(row.group_base_qenergy_damage)
        additive = 1.0 - float(row.router_square_additive_damage) / float(row.router_square_additive_base_damage)
        if not np.isclose(float(row.group_recovery), recovery, rtol=0, atol=1e-10):
            raise RuntimeError("group recovery arithmetic changed")
        if not np.isclose(float(row.router_square_additive_recovery), additive, rtol=0, atol=1e-10):
            raise RuntimeError("additive recovery arithmetic changed")
        selected_pages = _json_ints(row.selected_expert_pages, 8, "selected_expert_pages")
        if sum(selected_pages) != pages or max(selected_pages) > int(row.burst_cap_pages_per_expert):
            raise RuntimeError("selected expert page accounting changed")
        weights = json.loads(str(row.router_weights))
        if len(weights) != 8 or not np.isclose(sum(map(float, weights)), 1.0, atol=5e-7):
            raise RuntimeError("router weight evidence changed")
        oracle = row.allocation_policy == "pooled_exact_combined_moe_oracle"
        if bool(row.exact_cross_expert_information_used) != oracle or bool(row.promotable):
            raise RuntimeError("oracle/deployability marker changed")
        eligible = row.allocation_policy in {
            "uniform_per_expert", "pooled_router_square_compressed",
        }
        if bool(row.continuation_eligible) != eligible:
            raise RuntimeError("continuation marker changed")
        key = tuple(getattr(row, name) for name in (*GROUP_IDENTITY, *POLICY_IDENTITY))
        frame = expert_groups.get_group(key).sort_values("router_rank")
        if frame["router_rank"].astype(int).tolist() != list(range(1, 9)):
            raise RuntimeError("router-rank group changed")
        if frame["selected_pages"].astype(int).tolist() != selected_pages:
            raise RuntimeError("expert/group selected pages disagree")
        for states_json, counts_json in zip(frame["selected_states"], frame["selected_state_counts"]):
            states = _json_ints(states_json, 512, "selected_states")
            state_counts = _json_ints(counts_json, 8, "selected_state_counts")
            if any(state < 0 or state > 7 for state in states):
                raise RuntimeError("selected state lies outside [0,7]")
            if np.bincount(states, minlength=8).tolist() != state_counts:
                raise RuntimeError("selected state counts changed")

    if int(accounting.get("group_rows", -1)) != len(groups):
        raise RuntimeError("validation accounting group count changed")
    if int(accounting.get("expert_rows", -1)) != len(experts):
        raise RuntimeError("validation accounting expert count changed")
    return EvidenceBundle(
        config, fit_facts, manifest, run_facts, accounting, groups, experts, paths,
    )


def summary_tables(bundle: EvidenceBundle) -> dict[str, pd.DataFrame]:
    """Build accuracy, per-layer, allocation, and runtime tables."""
    groups = bundle.groups.copy()
    uniform = groups[
        groups["allocation_policy"].eq("uniform_per_expert")
    ][list(GROUP_IDENTITY) + [
        "factor_config_id", "mean_budget_pages_per_expert", "group_recovery",
    ]].rename(columns={"group_recovery": "uniform_group_recovery"})
    paired = groups.merge(
        uniform,
        on=list(GROUP_IDENTITY) + ["factor_config_id", "mean_budget_pages_per_expert"],
        how="left", validate="many_to_one",
    )
    if paired["uniform_group_recovery"].isna().any():
        raise RuntimeError("uniform paired comparison is incomplete")
    paired["paired_gain_vs_uniform"] = paired["group_recovery"] - paired["uniform_group_recovery"]
    keys = list(POLICY_IDENTITY)
    rows = []
    for key, frame in paired.groupby(keys, sort=True):
        summaries = {
            "qenergy_recovery": quantiles(frame["group_recovery"]),
            "additive_recovery": quantiles(frame["router_square_additive_recovery"]),
            "actual_total_bpw": quantiles(frame["average_actual_total_bpw"]),
            "paired_gain_vs_uniform": quantiles(frame["paired_gain_vs_uniform"]),
            "selector_runtime_ms": quantiles(frame["selector_runtime_ms"]),
            "actual_correction_pages": quantiles(frame["average_actual_correction_pages"]),
        }
        row = {
            **dict(zip(keys, key)), "groups": len(frame),
            "overall_average_bpw": float(frame["average_allowed_total_bpw"].iloc[0]),
            "selector_compute_macs": int(frame["selector_compute_macs"].iloc[0]),
            "selector_dp_state_evaluations": int(frame["selector_dp_state_evaluations"].iloc[0]),
            "selector_coordinate_sweeps_median": float(np.median(frame["selector_coordinate_sweeps"])),
            "selector_local_passes_median": float(np.median(frame["selector_local_passes"])),
            "continuation_eligible": bool(frame["continuation_eligible"].iloc[0]),
        }
        for prefix, values in summaries.items():
            row.update({f"{prefix}_{name}": value for name, value in values.items()})
        rows.append(row)
    accuracy = pd.DataFrame(rows).sort_values(
        ["factor_config_id", "overall_average_bpw", "allocation_policy",
         "burst_cap_pages_per_expert"],
    ).reset_index(drop=True)

    layer_rows = []
    for key, frame in paired.groupby(keys + ["layer"], sort=True):
        recovery, gain = quantiles(frame["group_recovery"]), quantiles(frame["paired_gain_vs_uniform"])
        row = {
            **dict(zip(keys + ["layer"], key)), "groups": len(frame),
            "overall_average_bpw": float(frame["average_allowed_total_bpw"].iloc[0]),
        }
        row.update({f"qenergy_recovery_{name}": value for name, value in recovery.items()})
        row.update({f"paired_gain_vs_uniform_{name}": value for name, value in gain.items()})
        layer_rows.append(row)
    expert_keys = keys + ["layer", "router_rank"]
    allocation = bundle.experts.groupby(expert_keys, as_index=False).agg(
        expert_invocations=("selected_pages", "size"),
        selected_pages_mean=("selected_pages", "mean"),
        selected_pages_median=("selected_pages", "median"),
        selected_pages_p10=("selected_pages", lambda x: float(np.quantile(x, .1))),
        selected_pages_p90=("selected_pages", lambda x: float(np.quantile(x, .9))),
        expert_recovery_median=("expert_recovery", "median"),
        expert_recovery_p10=("expert_recovery", lambda x: float(np.quantile(x, .1))),
    )
    runtime = accuracy[keys + [
        "selector_runtime_ms_median", "selector_runtime_ms_p90",
        "selector_compute_macs", "selector_dp_state_evaluations",
        "selector_coordinate_sweeps_median", "selector_local_passes_median",
    ]].copy()
    return {
        "accuracy_by_bpw": accuracy,
        "layer_accuracy": pd.DataFrame(layer_rows),
        "allocation_distribution": allocation,
        "runtime": runtime,
    }


def promotion_payload(
    bundle: EvidenceBundle, tables: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    config, rows = bundle.config, tables["accuracy_by_bpw"]
    mask = (
        rows["factor_config_id"].eq(config["primary_factor_id"])
        & rows["allocation_policy"].eq(config["promotion_primary_policy"])
        & rows["mean_budget_pages_per_expert"].eq(int(config["promotion_mean_pages"]))
        & rows["burst_cap_pages_per_expert"].eq(
            int(config["promotion_primary_burst_cap_pages"])
        )
    )
    if int(mask.sum()) != 1:
        raise RuntimeError("promotion-primary summary row is incomplete")
    row = rows[mask].iloc[0]
    median_gate = float(row["paired_gain_vs_uniform_median"]) >= float(
        config["promotion_gate_median_group_recovery_gain"]
    )
    p10_gate = float(row["paired_gain_vs_uniform_p10"]) >= float(
        config["promotion_gate_p10_group_recovery_gain"]
    )
    return {
        "schema_version": 1,
        "status": (
            "continue_to_predicted_h4_average_rate"
            if median_gate and p10_gate else "stop_average_rate_path"
        ),
        "selected_deployable_configuration": None,
        "exact_h4_geometry_ceiling_only": True,
        "primary_factor_config_id": str(config["primary_factor_id"]),
        "primary_policy": str(config["promotion_primary_policy"]),
        "comparison_policy": str(config["promotion_comparison_policy"]),
        "mean_correction_pages_per_expert": int(config["promotion_mean_pages"]),
        "burst_cap_pages_per_expert": int(config["promotion_primary_burst_cap_pages"]),
        "overall_average_bpw": float(row["overall_average_bpw"]),
        "qenergy_recovery_p10": float(row["qenergy_recovery_p10"]),
        "qenergy_recovery_median": float(row["qenergy_recovery_median"]),
        "paired_gain_vs_uniform_p10": float(row["paired_gain_vs_uniform_p10"]),
        "paired_gain_vs_uniform_median": float(row["paired_gain_vs_uniform_median"]),
        "median_gain_gate_pass": median_gate,
        "p10_gain_gate_pass": p10_gate,
        "q3_status": str(config["q3_followup_status"]),
        "q3_required_controls": list(config["q3_required_controls"]),
        "test_scientific_rows_admitted_or_used": False,
    }
