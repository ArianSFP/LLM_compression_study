"""Fail-closed analysis for the gate/up-Q3 physical-layout study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .q3_gate_up_field import (
    Q3_IDEAL_COST_QUANTA, Q3_INHERITED_EIGHT_STATE_MAP,
    Q3_STATE_DOWN_HIGH, Q3_STATE_GATE_LEVEL, Q3_STATE_UP_LEVEL,
    Q3PageLayout, fixed_gate_up_layout,
    monolithic_q2q4_layout,
)


GROUP_IDENTITY = (
    "capture_source", "evaluation_split", "request_id", "position", "layer",
)
PRIMARY_POLICY = "pooled_router_square_multi_budget_column_generated"
EXACT_POLICY = "pooled_exact_combined_moe_local_control"
BASELINE_LAYOUT = "q2q4_monolithic_same_solver_reference"
EXPERT_WEIGHTS = 3_145_728
BASE_EXACT_BPW = 4.25
ROUTER_SUM_ATOL = 8.0 * np.finfo(np.float32).eps

__all__ = [
    "PRIMARY_POLICY", "BASELINE_LAYOUT", "validate_evidence_bundle",
    "accuracy_table", "threshold_table", "promotion_payload",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _bytes_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _require_finite(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    for column in columns:
        if column not in frame:
            raise RuntimeError(f"{name} lacks {column}")
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"{name}.{column} is not finite")


def _require_bool(frame: pd.DataFrame, column: str, expected: bool | None = None) -> None:
    if column not in frame or frame[column].isna().any():
        raise RuntimeError(f"group.{column} is missing")
    values = set(frame[column].map(bool).unique().tolist())
    if expected is not None and values != {bool(expected)}:
        raise RuntimeError(f"group.{column} changed")


def _json_vector(value: str, length: int, name: str) -> np.ndarray:
    try:
        result = np.asarray(json.loads(value), np.float64)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid {name}") from error
    if result.shape != (int(length),) or not np.all(np.isfinite(result)):
        raise RuntimeError(f"invalid {name}")
    return result


def _layouts(action_pages: np.ndarray) -> dict[str, Q3PageLayout]:
    native = {
        BASELINE_LAYOUT: monolithic_q2q4_layout(512),
        "q3_ideal_logical_256_byte_plane_ceiling": Q3PageLayout("ideal", 512, None),
        "q3_physical_fixed_gate_up_pairing": fixed_gate_up_layout(512),
        "q3_physical_training_coselection_single": Q3PageLayout(
            "learned1", 512, np.asarray(action_pages[:1], np.int64), learned=True,
        ),
        "q3_physical_training_coselection_replicated2": Q3PageLayout(
            "learned2", 512, np.asarray(action_pages, np.int64), learned=True,
        ),
    }
    legacy = np.asarray(native[BASELINE_LAYOUT].action_pages, np.int64)
    output = dict(native)
    for layout_id in (
        "q3_physical_fixed_gate_up_pairing",
        "q3_physical_training_coselection_single",
        "q3_physical_training_coselection_replicated2",
    ):
        value = native[layout_id]
        output[layout_id] = Q3PageLayout(
            f"{value.layout_id}_plus_legacy_q2q4_fallback", value.units,
            np.concatenate((np.asarray(value.action_pages, np.int64), legacy), axis=0),
            learned=value.learned, fixed=value.fixed,
        )
    return output

def _expected_layout_accounting(
    layout_id: str, config: Mapping[str, Any],
) -> tuple[int, int, bool, float]:
    if layout_id == "q3_physical_training_coselection_single":
        descriptor = int(config["layout_descriptor_bytes_per_layer_single"]) // 256
    elif layout_id == "q3_physical_training_coselection_replicated2":
        descriptor = int(config["layout_descriptor_bytes_per_layer_replicated"]) // 256
    else:
        descriptor = 0
    replicas = {BASELINE_LAYOUT: 1, "q3_ideal_logical_256_byte_plane_ceiling": 0,
                "q3_physical_fixed_gate_up_pairing": 2,
                "q3_physical_training_coselection_single": 2,
                "q3_physical_training_coselection_replicated2": 3}[layout_id]
    physical = layout_id != "q3_ideal_logical_256_byte_plane_ceiling"
    base_bytes = BASE_EXACT_BPW * EXPERT_WEIGHTS / 8.0
    duplicated = max(replicas - 1, 0) * (2 * 512 * 2048 * 2 / 8)
    resident = (
        int(config["factor_payload_bytes_per_expert"])
        + int(config["abc_metadata_bytes_per_expert"])
        + descriptor
    )
    storage = float((base_bytes + duplicated + resident) / base_bytes)
    return descriptor, replicas, physical, storage


def _validate_expert_states(
    expert: pd.DataFrame, layout_manifest: Mapping[str, Any], layout_dir: Path,
) -> None:
    cached = {}
    for layer in sorted(set(map(int, expert["layer"].unique()))):
        record = layout_manifest["layers"][str(layer)]
        shard = layout_dir / str(record["shard"])
        if _sha256(shard) != str(record["shard_sha256"]) or shard.stat().st_size != int(record["shard_bytes"]):
            raise RuntimeError("layout shard changed during analysis")
        arrays = dict(np.load(shard, allow_pickle=False))
        cached[layer] = _layouts(arrays["action_pages"])
    for row in expert.itertuples(index=False):
        layout = cached[int(row.layer)][str(row.layout_id)]
        counts = np.asarray(json.loads(row.selected_state_counts), np.int64)
        if counts.shape != (18,) or np.any(counts < 0) or int(counts.sum()) != 512:
            raise RuntimeError("selected state-count evidence changed")
        if str(row.layout_id) == BASELINE_LAYOUT:
            forbidden = set(range(18)) - set(map(int, Q3_INHERITED_EIGHT_STATE_MAP))
            if any(int(counts[state]) for state in forbidden):
                raise RuntimeError("eight-state baseline used a Q3-only state")
        blob = row.selected_states_blob
        if not isinstance(blob, (bytes, bytearray, memoryview)) or len(blob) != 512:
            raise RuntimeError("expert row lacks its 512-byte state evidence")
        states = np.frombuffer(blob, dtype=np.uint8).astype(np.int64)
        if np.any(states >= 18) or not np.array_equal(np.bincount(states, minlength=18), counts):
            raise RuntimeError("state blob/count evidence changed")
        if _bytes_sha256(states.astype(np.uint8)) != str(row.selected_states_sha256):
            raise RuntimeError("state hash changed")
        cost = layout.cost_quanta(states)
        replica = layout.selected_replica(states)
        if layout.ideal:
            if cost != int(row.selected_quanta):
                raise RuntimeError("ideal 256-byte quantum accounting changed")
            if replica != -1 or int(row.selected_replica) != -1:
                raise RuntimeError("ideal layout has a physical replica")
            if not pd.isna(row.selected_unique_pages) or not pd.isna(row.selected_page_union_sha256):
                raise RuntimeError("ideal layout has physical page evidence")
        else:
            pages = layout.page_union(states, replica)
            if cost != int(row.selected_quanta) or replica != int(row.selected_replica):
                raise RuntimeError("physical unique-page cost changed")
            if len(pages) != int(row.selected_unique_pages) or 2 * len(pages) != int(row.selected_quanta):
                raise RuntimeError("physical page/quanta relation changed")
            if _bytes_sha256(np.asarray(pages, np.int64)) != str(row.selected_page_union_sha256):
                raise RuntimeError("selected page-union hash changed")
        expected_counts = {
            "selected_gate_q3_or_above_units": np.count_nonzero(Q3_STATE_GATE_LEVEL[states] >= 1),
            "selected_gate_q4_units": np.count_nonzero(Q3_STATE_GATE_LEVEL[states] == 2),
            "selected_up_q3_or_above_units": np.count_nonzero(Q3_STATE_UP_LEVEL[states] >= 1),
            "selected_up_q4_units": np.count_nonzero(Q3_STATE_UP_LEVEL[states] == 2),
            "selected_down_q4_units": np.count_nonzero(Q3_STATE_DOWN_HIGH[states]),
        }
        for name, value in expected_counts.items():
            if int(getattr(row, name)) != int(value):
                raise RuntimeError(f"Q3 state-use counter changed: {name}")


def validate_evidence_bundle(
    config_path: Path, layout_dir: Path, validation_dir: Path,
    *, experiment: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    config = json.loads(config_path.read_text())
    facts_path = validation_dir / "q3_gate_up_run_facts.json"
    accounting_path = validation_dir / "q3_gate_up_accounting.json"
    group_path = validation_dir / "q3_gate_up_group_frontier.parquet"
    expert_path = validation_dir / "q3_gate_up_expert_allocation.parquet"
    layout_facts_path = layout_dir / "q3_layout_fit_facts.json"
    layout_manifest_path = layout_dir / "q3_layout_manifest.json"
    facts = json.loads(facts_path.read_text())
    accounting = json.loads(accounting_path.read_text())
    layout_facts = json.loads(layout_facts_path.read_text())
    layout_manifest = json.loads(layout_manifest_path.read_text())
    if facts.get("completed") is not True or layout_facts.get("completed") is not True or layout_manifest.get("completed") is not True:
        raise RuntimeError("Q3 evidence is incomplete")
    if facts.get("failures") or facts.get("failure_history") or layout_facts.get("failures") or layout_facts.get("failure_history"):
        raise RuntimeError("Q3 evidence contains a failure")
    expected_hashes = {
        "config_sha256": _sha256(config_path),
        "runner_sha256": _sha256(experiment / "scripts/run_q3_gate_up_layout.py"),
        "q3_field_core_sha256": _sha256(experiment / "src/oracle_study/q3_gate_up_field.py"),
        "q3_allocator_core_sha256": _sha256(experiment / "src/oracle_study/q3_rate_allocator.py"),
        "layout_facts_sha256": _sha256(layout_facts_path),
        "layout_manifest_sha256": _sha256(layout_manifest_path),
    }
    for name, value in expected_hashes.items():
        if facts.get(name) != value:
            raise RuntimeError(f"validation source/input hash changed: {name}")
    if layout_facts.get("layout_manifest_sha256") != expected_hashes["layout_manifest_sha256"]:
        raise RuntimeError("layout facts/manifest link changed")
    provenance_names = (
        "config_sha256", "runner_sha256", "q3_field_core_sha256", "q3_allocator_core_sha256",
        "average_allocator_core_sha256", "split_core_sha256", "interaction_core_sha256",
        "mxfp4_core_sha256", "set_utility_runner_sha256", "sparse_runner_sha256",
        "base_pr13_config_sha256", "base_pr13_runner_sha256", "base_pr13_allocator_sha256",
        "base_pr13_split_core_sha256", "base_pr13_fit_facts_sha256",
        "base_pr13_fit_manifest_sha256", "base_pr13_validation_facts_sha256",
        "base_pr13_promotions_sha256", "capture_sha256", "tree_sha256",
        "checkpoint_config_sha256", "checkpoint_index_sha256", "runtime_provenance",
    )
    for name in provenance_names:
        if layout_facts.get(name) != layout_manifest.get(name):
            raise RuntimeError(f"layout facts/manifest provenance changed: {name}")
        if (name not in {"checkpoint_config_sha256", "checkpoint_index_sha256"}
                and facts.get(name) != layout_manifest.get(name)):
            raise RuntimeError(f"layout/evaluation provenance changed: {name}")
    expected_layers = sorted(map(int, config["layers"]))
    if sorted(map(int, facts.get("completed_layers", []))) != expected_layers:
        raise RuntimeError("validation layer closure changed")
    if sorted(map(int, layout_facts.get("completed_layers", []))) != expected_layers:
        raise RuntimeError("layout-fit layer closure changed")
    if sorted(map(int, layout_manifest.get("layers", {}))) != expected_layers:
        raise RuntimeError("layout manifest layer scope changed")
    if facts.get("selection_split") != "validation" or layout_facts.get("fit_split") != "train":
        raise RuntimeError("Q3 split contract changed")
    if facts.get("test_scientific_rows_admitted_or_used") is not False:
        raise RuntimeError("validation test-scientific boundary changed")
    if (layout_facts.get("test_scientific_rows_admitted_or_used") is not False
            or layout_manifest.get("validation_or_test_values_used") is not False):
        raise RuntimeError("layout-fit test-scientific boundary changed")
    if facts.get("validation_groups") != int(config["expected_validation_groups"]) or facts.get("validation_expert_invocations") != int(config["expected_validation_expert_invocations"]):
        raise RuntimeError("validation scope changed")
    for path, field in ((group_path, "group_frontier_sha256"), (expert_path, "expert_allocation_sha256"),
                        (accounting_path, "accounting_sha256")):
        if facts.get(field) != _sha256(path):
            raise RuntimeError(f"validation artifact changed: {path.name}")
    group = pd.read_parquet(group_path)
    expert = pd.read_parquet(expert_path)
    expected_group_rows = int(config["expected_validation_groups"]) * len(config["layout_controls"]) * len(config["mean_correction_quanta"]) * len(config["allocation_policies"])
    expected_expert_rows = int(config["expected_validation_groups"]) * len(config["layout_controls"]) * len(config["mean_correction_quanta"]) * 8
    if len(group) != expected_group_rows or len(expert) != expected_expert_rows:
        raise RuntimeError("Q3 artifact row count changed")
    keys = list(GROUP_IDENTITY) + ["layout_id", "allocation_policy", "mean_budget_quanta_per_expert"]
    if group[keys].duplicated().any():
        raise RuntimeError("duplicate group row")
    observed_grid = set(zip(group.layout_id.astype(str), group.allocation_policy.astype(str),
                            group.mean_budget_quanta_per_expert.astype(int)))
    expected_grid = {(layout, policy, int(mean)) for layout in config["layout_controls"]
                     for policy in config["allocation_policies"] for mean in config["mean_correction_quanta"]}
    if observed_grid != expected_grid:
        raise RuntimeError("Q3 categorical grid changed")
    identities = group[list(GROUP_IDENTITY)].drop_duplicates()
    if len(identities) != int(config["expected_validation_groups"]):
        raise RuntimeError("Q3 validation identity count changed")
    if set(identities.evaluation_split.astype(str)) != {"validation"}:
        raise RuntimeError("Q3 evidence admitted another split")
    if set(identities.capture_source.astype(str)) != {"exact_checkpoint"}:
        raise RuntimeError("Q3 evidence admitted another capture source")
    layer_counts = identities.groupby("layer").size().to_dict()
    expected_layer_counts = {
        int(layer): int(config["expected_validation_groups_per_layer"])
        for layer in config["layers"]
    }
    if layer_counts != expected_layer_counts:
        raise RuntimeError("Q3 per-layer validation scope changed")
    _require_finite(group, (
        "group_recovery", "group_exact_qenergy_damage", "group_base_qenergy_damage",
        "actual_group_quanta", "group_budget_quanta", "average_allowed_total_bpw",
        "average_actual_total_bpw", "combined_metadata_bpw", "selector_wall_time_ms",
        "selector_frontier_wall_time_ms", "selector_allocation_wall_time_ms",
        "selector_compute_macs", "selector_diagonal_dp_tables",
        "selector_diagonal_dp_state_updates", "external_storage_multiplier",
    ), "group")
    if np.any(group.selector_diagonal_dp_tables < 0) or np.any(group.selector_diagonal_dp_state_updates < 0):
        raise RuntimeError("Q3 diagonal-DP operation counters changed")
    if not np.allclose(
        group.selector_wall_time_ms,
        group.selector_frontier_wall_time_ms + group.selector_allocation_wall_time_ms,
        rtol=0, atol=1e-6,
    ):
        raise RuntimeError("Q3 selector wall-time decomposition changed")
    expected_descriptor = group.layout_id.astype(str).map(
        lambda item: _expected_layout_accounting(item, config)[0]
    ).to_numpy(np.int64)
    expected_replicas = group.layout_id.astype(str).map(
        lambda item: _expected_layout_accounting(item, config)[1]
    ).to_numpy(np.int64)
    expected_physical = group.layout_id.astype(str).map(
        lambda item: _expected_layout_accounting(item, config)[2]
    ).to_numpy(bool)
    expected_storage = group.layout_id.astype(str).map(
        lambda item: _expected_layout_accounting(item, config)[3]
    ).to_numpy(np.float64)
    expected_metadata = (
        int(config["factor_payload_bytes_per_expert"])
        + int(config["abc_metadata_bytes_per_expert"])
        + expected_descriptor
    )
    exact_columns = {
        "factor_payload_bytes_per_expert": np.full(len(group), int(config["factor_payload_bytes_per_expert"])),
        "abc_metadata_bytes_per_expert": np.full(len(group), int(config["abc_metadata_bytes_per_expert"])),
        "layout_descriptor_bytes_per_expert": expected_descriptor,
        "combined_metadata_bytes_per_expert": expected_metadata,
        "layout_replicas": expected_replicas,
        "burst_cap_quanta_per_expert": np.full(len(group), int(config["burst_cap_quanta_per_expert"])),
    }
    for name, expected in exact_columns.items():
        if name not in group or not np.array_equal(pd.to_numeric(group[name]).to_numpy(np.int64), expected):
            raise RuntimeError(f"group.{name} accounting changed")
    if not np.array_equal(group.layout_is_physical.map(bool).to_numpy(), expected_physical):
        raise RuntimeError("group.layout_is_physical accounting changed")
    if not np.allclose(group.external_storage_multiplier, expected_storage, rtol=0, atol=1e-15):
        raise RuntimeError("group.external_storage_multiplier accounting changed")
    expected_metadata_bpw = 8.0 * expected_metadata / EXPERT_WEIGHTS
    if not np.allclose(group.combined_metadata_bpw, expected_metadata_bpw, rtol=0, atol=1e-15):
        raise RuntimeError("group.combined_metadata_bpw accounting changed")
    if set(group.factor_config_id.astype(str)) != {str(config["factor_id"])}:
        raise RuntimeError("factor identity changed")
    if set(group.selection_regime.astype(str)) != {"exact_h4_rank4_hadamard_q3_gate_up_geometry_ceiling"}:
        raise RuntimeError("selection regime changed")
    _require_bool(group, "exact_h4_gate_up_responses_used", True)
    if np.any(group.group_base_qenergy_damage <= 0) or np.any(group.actual_group_quanta > group.group_budget_quanta):
        raise RuntimeError("group damage/budget contract changed")
    recovery = 1.0 - group.group_exact_qenergy_damage.to_numpy(np.float64) / group.group_base_qenergy_damage.to_numpy(np.float64)
    if not np.allclose(recovery, group.group_recovery, rtol=0, atol=1e-12):
        raise RuntimeError("group recovery arithmetic changed")
    if not np.array_equal(group.group_budget_quanta.to_numpy(np.int64), 8 * group.mean_budget_quanta_per_expert.to_numpy(np.int64)):
        raise RuntimeError("group quantum-budget arithmetic changed")
    selected_quanta = np.stack([
        _json_vector(value, 8, "selected_expert_quanta").astype(np.int64)
        for value in group.selected_expert_quanta.astype(str)
    ])
    if np.any(selected_quanta < 0) or np.any(selected_quanta > int(config["burst_cap_quanta_per_expert"])):
        raise RuntimeError("selected expert quantum vector changed")
    if not np.array_equal(selected_quanta.sum(axis=1), group.actual_group_quanta.to_numpy(np.int64)):
        raise RuntimeError("selected expert/group quantum total changed")
    router = np.stack([_json_vector(value, 8, "router_weights") for value in group.router_weights.astype(str)])
    if np.any(router < 0) or not np.allclose(router.sum(axis=1), 1.0, rtol=0, atol=ROUTER_SUM_ATOL):
        raise RuntimeError("router-weight evidence changed")
    expected_bpw = (
        pd.to_numeric(group.combined_metadata_bytes_per_expert).to_numpy(np.float64)
        + pd.to_numeric(group.mean_budget_quanta_per_expert).to_numpy(np.float64) * 256
    ) * 8 / int(config["expert_weights"])
    if not np.allclose(expected_bpw, group.average_allowed_total_bpw, rtol=0, atol=1e-15):
        raise RuntimeError("allowed total-bpw arithmetic changed")
    metadata = group.combined_metadata_bytes_per_expert.to_numpy(np.int64)
    actual_bpw = (
        8 * metadata + group.actual_group_quanta.to_numpy(np.int64) * 256
    ) / int(config["expert_weights"])
    if not np.allclose(actual_bpw, group.average_actual_total_bpw, rtol=0, atol=1e-15):
        raise RuntimeError("actual total-bpw arithmetic changed")
    strict = (
        8 * (metadata + group.mean_budget_quanta_per_expert.to_numpy(np.int64) * 256)
        <= 8 * int(config["strict_all_in_total_bytes_per_expert"])
    )
    if not np.array_equal(strict, group.strict_one_bpw_pass.map(bool).to_numpy()):
        raise RuntimeError("strict all-in flag changed")
    _require_bool(group, "promotable", False)
    expected_exact = group.allocation_policy.astype(str) == EXACT_POLICY
    if not np.array_equal(expected_exact, group.exact_cross_expert_information_used.map(bool)):
        raise RuntimeError("exact cross-expert control flag changed")
    expected_continuation = (
        group.layout_id.astype(str).isin(set(config["primary_physical_layouts"]))
        & (group.allocation_policy.astype(str) == PRIMARY_POLICY)
    )
    if not np.array_equal(expected_continuation, group.continuation_eligible.map(bool)):
        raise RuntimeError("continuation eligibility changed")
    expert_key = list(GROUP_IDENTITY) + ["layout_id", "mean_budget_quanta_per_expert", "router_rank"]
    if expert[expert_key].duplicated().any():
        raise RuntimeError("duplicate Q3 expert-allocation row")
    if set(expert.allocation_policy.astype(str)) != {PRIMARY_POLICY} or set(expert.router_rank.astype(int)) != set(range(1, 9)):
        raise RuntimeError("expert allocation policy/rank scope changed")
    expert_grid = expert.groupby(list(GROUP_IDENTITY) + ["layout_id", "mean_budget_quanta_per_expert"]).size()
    expected_expert_groups = int(config["expected_validation_groups"]) * len(config["layout_controls"]) * len(config["mean_correction_quanta"])
    if len(expert_grid) != expected_expert_groups or not np.all(expert_grid.to_numpy() == 8):
        raise RuntimeError("expert allocation grid changed")
    _require_finite(expert, (
        "selected_quanta", "router_weight", "compressed_predicted_damage",
        "exact_qenergy_damage", "base_qenergy_damage", "expert_recovery",
    ), "expert")
    if np.any(expert.base_qenergy_damage <= 0) or np.any(
        (expert.router_weight < 0) | (expert.router_weight > 1)
    ):
        raise RuntimeError("expert damage/router-weight contract changed")
    expected_expert_recovery = 1.0 - expert.exact_qenergy_damage / expert.base_qenergy_damage
    if not np.allclose(expected_expert_recovery, expert.expert_recovery, rtol=0, atol=1e-12):
        raise RuntimeError("expert recovery arithmetic changed")
    expert_weight_sums = expert.groupby(
        list(GROUP_IDENTITY) + ["layout_id", "mean_budget_quanta_per_expert"]
    ).router_weight.sum()
    if not np.allclose(expert_weight_sums, 1.0, rtol=0, atol=ROUTER_SUM_ATOL):
        raise RuntimeError("expert router weights do not sum to one")
    _validate_expert_states(expert, layout_manifest, layout_dir)
    primary = group[group.allocation_policy == PRIMARY_POLICY]
    expert_keys = list(GROUP_IDENTITY) + ["layout_id", "mean_budget_quanta_per_expert"]
    sums = expert.groupby(expert_keys, dropna=False).selected_quanta.sum().reset_index(name="expert_quanta")
    joined = primary.merge(sums, on=expert_keys, validate="one_to_one")
    if len(joined) != len(primary) or not np.array_equal(joined.actual_group_quanta.astype(int), joined.expert_quanta.astype(int)):
        raise RuntimeError("expert/group selected-quanta evidence changed")
    if accounting.get("group_rows") != len(group) or accounting.get("expert_rows") != len(expert):
        raise RuntimeError("accounting row totals changed")
    expected_accounting = {
        "schema_version": 1, "group_rows": len(group), "expert_rows": len(expert),
        "validation_groups": int(config["expected_validation_groups"]), "experts_per_group": 8,
        "cost_quantum_bytes": 256, "physical_page_bytes": 512,
        "factor_payload_bytes_per_expert": int(config["factor_payload_bytes_per_expert"]),
        "abc_metadata_bytes_per_expert": int(config["abc_metadata_bytes_per_expert"]),
        "layout_controls": config["layout_controls"],
        "allocation_policies": config["allocation_policies"],
        "mean_correction_quanta": config["mean_correction_quanta"],
        "physical_cost_convention": "two_256_byte_planes_per_512_byte_page_charged_by_exact_unique_page_union",
        "replicated_cost_convention": "choose_one_complete_layout_replica_with_minimum_union_per_expert_invocation_no_cross_replica_mixing",
        "legacy_q2q4_fallback_replica": True,
    }
    if any(accounting.get(name) != value for name, value in expected_accounting.items()):
        raise RuntimeError("accounting schema/content changed")
    return config, group, expert, {"facts": facts, "accounting": accounting,
                                  "layout_facts": layout_facts, "layout_manifest": layout_manifest}


def accuracy_table(group: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, frame in group.groupby(
        ["layout_id", "allocation_policy", "mean_budget_quanta_per_expert"], sort=True,
    ):
        recovery = frame.group_recovery.to_numpy(np.float64)
        rows.append({
            "layout_id": keys[0], "allocation_policy": keys[1],
            "mean_budget_quanta_per_expert": int(keys[2]),
            "average_allowed_total_bpw": float(frame.average_allowed_total_bpw.iloc[0]),
            "average_actual_total_bpw_mean": float(frame.average_actual_total_bpw.mean()),
            "groups": len(frame), "recovery_p10": float(np.quantile(recovery, .1)),
            "recovery_median": float(np.median(recovery)),
            "recovery_p90": float(np.quantile(recovery, .9)),
            "recovery_mean": float(np.mean(recovery)),
            "remaining_damage_median": float(1.0 - np.median(recovery)),
        })
    result = pd.DataFrame(rows)
    baseline = result[result.layout_id == BASELINE_LAYOUT][[
        "allocation_policy", "mean_budget_quanta_per_expert", "recovery_p10",
        "recovery_median",
    ]].rename(columns={
        "recovery_p10": "baseline_recovery_p10",
        "recovery_median": "baseline_recovery_median",
    })
    result = result.merge(
        baseline, on=["allocation_policy", "mean_budget_quanta_per_expert"],
        how="left", validate="many_to_one",
    )
    baseline_p10_damage = 1.0 - result.baseline_recovery_p10.to_numpy(np.float64)
    baseline_median_damage = 1.0 - result.baseline_recovery_median.to_numpy(np.float64)
    if np.any(baseline_p10_damage <= 0) or np.any(baseline_median_damage <= 0):
        raise RuntimeError("baseline remaining damage is not positive")
    result["p10_remaining_damage_ratio_vs_baseline_same_budget"] = (1.0 - result.recovery_p10) / baseline_p10_damage
    result["median_remaining_damage_ratio_vs_baseline_same_budget"] = (1.0 - result.recovery_median) / baseline_median_damage
    return result


def threshold_table(accuracy: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    primary = accuracy[accuracy.allocation_policy == PRIMARY_POLICY]
    rows = []
    for layout, frame in primary.groupby("layout_id", sort=True):
        frame = frame.sort_values("average_allowed_total_bpw")
        for target in map(float, config["median_targets"]):
            eligible = frame[frame.recovery_median >= target]
            rows.append({"layout_id": layout, "metric": "median", "target": target,
                         "minimum_total_bpw": None if eligible.empty else float(eligible.average_allowed_total_bpw.iloc[0])})
        for target in map(float, config["p10_floor_targets"]):
            eligible = frame[frame.recovery_p10 >= target]
            rows.append({"layout_id": layout, "metric": "p10", "target": target,
                         "minimum_total_bpw": None if eligible.empty else float(eligible.average_allowed_total_bpw.iloc[0])})
    return pd.DataFrame(rows)


def promotion_payload(accuracy: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, Any]:
    primary = accuracy[accuracy.allocation_policy == PRIMARY_POLICY]
    strict = int(config["strict_all_in_mean_quanta"])
    target_p10 = float(config["frozen_pr13_rank4_target_recovery_p10"])
    target_median = float(config["frozen_pr13_rank4_target_recovery_median"])
    target_bpw = float(config["frozen_pr13_rank4_target_total_bpw"])
    tolerance = float(config["baseline_reproduction_absolute_tolerance"])
    search_resolution_quanta = int(config["matched_target_search_resolution_quanta"])
    search_resolution_bpw = (
        8.0 * search_resolution_quanta * int(config["cost_quantum_bytes"])
        / int(config["expert_weights"])
    )
    baseline_rows = primary[(primary.layout_id == BASELINE_LAYOUT) & (primary.mean_budget_quanta_per_expert == strict)]
    if len(baseline_rows) != 1:
        raise RuntimeError("strict baseline summary is not unique")
    baseline = baseline_rows.iloc[0]
    if 1.0 - float(baseline.recovery_median) <= 0 or 1.0 - float(baseline.recovery_p10) <= 0:
        raise RuntimeError("strict baseline remaining damage is not positive")
    baseline_p10_delta = float(baseline.recovery_p10) - target_p10
    baseline_median_delta = float(baseline.recovery_median) - target_median
    reproduction_pass = (
        abs(baseline_p10_delta) <= tolerance and abs(baseline_median_delta) <= tolerance
    )
    comparisons = []
    for layout in config["primary_physical_layouts"]:
        row = primary[(primary.layout_id == layout) & (primary.mean_budget_quanta_per_expert == strict)]
        if len(row) != 1:
            raise RuntimeError("strict physical-layout summary is not unique")
        value = row.iloc[0]
        median_ratio = (1.0 - float(value.recovery_median)) / (1.0 - target_median)
        p10_ratio = (1.0 - float(value.recovery_p10)) / (1.0 - target_p10)
        curve = primary[primary.layout_id == layout].sort_values("average_allowed_total_bpw")
        matched = curve[(curve.recovery_median >= target_median) &
                        (curve.recovery_p10 >= target_p10)]
        matched_row = None if matched.empty else matched.iloc[0]
        matched_bpw = None if matched_row is None else float(matched_row.average_allowed_total_bpw)
        matched_quanta = None if matched_row is None else int(matched_row.mean_budget_quanta_per_expert)
        prior = curve[curve.mean_budget_quanta_per_expert < matched_quanta] if matched_quanta is not None else curve.iloc[0:0]
        prior_row = None if prior.empty else prior.iloc[-1]
        delta = None if matched_bpw is None else matched_bpw - target_bpw
        same_rate_pass = median_ratio <= float(config["promotion_same_rate_remaining_damage_ratio_max"])
        matched_pass = delta is not None and delta <= float(config["promotion_matched_quality_total_bpw_delta_max"])
        comparisons.append({
            "layout_id": layout, "strict_recovery_p10": float(value.recovery_p10),
            "strict_recovery_median": float(value.recovery_median),
            "strict_median_remaining_damage_ratio_vs_frozen_pr13_target": median_ratio,
            "strict_p10_remaining_damage_ratio_vs_frozen_pr13_target": p10_ratio,
            "matched_frozen_pr13_p10_and_median_minimum_total_bpw": matched_bpw,
            "matched_frozen_pr13_mean_quanta": matched_quanta,
            "previous_sampled_total_bpw": None if prior_row is None else float(prior_row.average_allowed_total_bpw),
            "previous_sampled_recovery_p10": None if prior_row is None else float(prior_row.recovery_p10),
            "previous_sampled_recovery_median": None if prior_row is None else float(prior_row.recovery_median),
            "matched_target_search_resolution_quanta": search_resolution_quanta,
            "matched_target_search_resolution_total_bpw": search_resolution_bpw,
            "matched_quality_total_bpw_delta": delta,
            "same_rate_remaining_damage_gate_pass": same_rate_pass,
            "matched_quality_rate_shift_gate_pass": matched_pass,
            "passes": reproduction_pass and (same_rate_pass or matched_pass),
        })
    passing = [item for item in comparisons if item["passes"]]
    selected = None if not passing else min(
        passing, key=lambda item: (
            item["strict_median_remaining_damage_ratio_vs_frozen_pr13_target"],
            np.inf if item["matched_quality_total_bpw_delta"] is None else item["matched_quality_total_bpw_delta"],
            item["layout_id"],
        ),
    )["layout_id"]
    ideal_rows = primary[
        (primary.layout_id == "q3_ideal_logical_256_byte_plane_ceiling")
        & (primary.mean_budget_quanta_per_expert == strict)
    ]
    if len(ideal_rows) != 1:
        raise RuntimeError("strict ideal-layout summary is not unique")
    ideal = ideal_rows.iloc[0]
    witness = max(
        comparisons,
        key=lambda item: (
            item["strict_recovery_median"], item["strict_recovery_p10"],
            item["layout_id"],
        ),
    )
    ideal_diagnostic = {
        "reported_ideal_control_is_certified_global_ceiling": False,
        "candidate_frontier_is_constructive_superset_of_every_physical_frontier": True,
        "compressed_objective_dominance_is_runner_asserted": True,
        "strict_solver_recovery_p10": float(ideal.recovery_p10),
        "strict_solver_recovery_median": float(ideal.recovery_median),
        "strict_feasible_witness_layout": witness["layout_id"],
        "strict_feasible_witness_recovery_p10": witness["strict_recovery_p10"],
        "strict_feasible_witness_recovery_median": witness["strict_recovery_median"],
        "solver_median_recovery_gap_to_feasible_witness": witness["strict_recovery_median"] - float(ideal.recovery_median),
        "solver_p10_recovery_gap_to_feasible_witness": witness["strict_recovery_p10"] - float(ideal.recovery_p10),
        "interpretation": "every physical-layout state is feasible at equal-or-lower cost in the ideal plane space; a lower ideal solver result is optimization headroom, not a representation bound",
    }
    return {
        "schema_version": 3,
        "gate_up_q3_status": "continue_to_predictive_q3_study" if selected else "stop_embedded_gate_up_q3",
        "selected_physical_layout": selected,
        "frozen_pr13_rank4_target": {
            "total_bpw": target_bpw, "recovery_p10": target_p10,
            "recovery_median": target_median,
            "search_resolution_quanta": search_resolution_quanta,
            "search_resolution_total_bpw": search_resolution_bpw,
        },
        "restored_eight_state_baseline": {
            "mean_quanta": strict, "total_bpw": float(baseline.average_allowed_total_bpw),
            "recovery_p10": float(baseline.recovery_p10), "recovery_median": float(baseline.recovery_median),
            "p10_delta_vs_frozen_pr13": baseline_p10_delta,
            "median_delta_vs_frozen_pr13": baseline_median_delta,
            "absolute_tolerance": tolerance,
            "reproduction_pass": reproduction_pass,
        },
        "physical_layout_comparisons": comparisons,
        "ideal_logical_solver_diagnostic": ideal_diagnostic,
        "down_q3_status": "eligible_for_separate_followup" if selected else "deferred_gate_up_q3_did_not_pass",
        "test_scientific_values_admitted_or_used": False,
        "scope": config["study_scope"],
    }
