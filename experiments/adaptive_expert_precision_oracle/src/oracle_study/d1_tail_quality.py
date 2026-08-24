"""Policy assembly and metrics for single-layer D1 downstream-tail replay.

This module contains no model loading.  It validates the immutable layer-slice
artifacts, turns their complete group delta banks into the five comparison
policies, and computes metrics from an exact downstream execution supplied by
the caller.  Keeping artifact selection here makes the oracle-versus-runtime
and calibration-versus-reranking boundaries testable without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .causal_control import first_changed_layer
from .causal_replay import mean_squared_error, route_boundary_metrics, token_quality_metrics


PR13_POLICY = "pr13_router_square"
LOCAL_POLICY = "exact_combined_local"
FIXED_D1_POLICY = "calibration_selected_fixed_d1"
RERANKED_D1_POLICY = "exact_request_reranked_d1"
REPAIRED_D1_POLICY = "exact_request_repaired_d1"
STANDARD_POLICIES = (
    PR13_POLICY,
    LOCAL_POLICY,
    FIXED_D1_POLICY,
    RERANKED_D1_POLICY,
    REPAIRED_D1_POLICY,
)

HISTORICAL_PR13 = "pr13_router_square_column_generated"
REGENERATED_PR13 = "regenerated_router_square_column_generated"
LOCAL_PREFIX = "exact_combined_local_"
D1_PREFIX = "d1_strict_"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256_manifest(root: Path, manifest: Path) -> int:
    """Verify a two-space-delimited sha256sum manifest under ``root``."""

    records = [line for line in manifest.read_text().splitlines() if line.strip()]
    seen: set[str] = set()
    for line in records:
        try:
            expected, relative_name = line.split("  ", 1)
        except ValueError as error:
            raise RuntimeError(f"malformed SHA-256 manifest row: {line!r}") from error
        if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
            raise RuntimeError(f"malformed SHA-256 digest in manifest: {line!r}")
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe path in SHA-256 manifest: {relative_name!r}")
        normalized = relative.as_posix()
        if normalized in seen:
            raise RuntimeError(f"duplicate path in SHA-256 manifest: {relative_name!r}")
        seen.add(normalized)
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"raw expansion artifact changed: {path}")
    return len(records)


def validate_expansion_config(config: Mapping[str, Any]) -> None:
    """Validate the immutable promoted-expansion contract."""

    if config.get("schema") != "pr13_d1_layer_slice_expansion_config_v1":
        raise ValueError("unexpected D1 expansion config schema")
    if list(map(int, config["injection_layers"])) != [0, 1, 4, 6, 12, 23]:
        raise ValueError("promoted D1 layer set changed")
    if list(map(int, config["mean_budget_pages_per_expert"])) != [384, 749]:
        raise ValueError("promoted D1 mechanism rates changed")
    expected_eta = np.asarray(
        [0.0, 0.00005, 0.0001, 0.00025, 0.0005, 0.001], np.float64,
    )
    actual_eta = np.asarray(config["local_guard"]["eta_sweep"], np.float64)
    if not np.array_equal(actual_eta, expected_eta):
        raise ValueError("promoted D1 eta grid changed")
    requests = list(map(str, config["validation_request_ids"]))
    if requests != ["mxfp4-confirm-006", "mxfp4-confirm-007", "mxfp4-confirm-008"]:
        raise ValueError("promoted D1 request set changed")
    mask = config["position_mask"]
    admitted = mask["admitted_positions_per_request"]
    if sum(int(admitted[request]) for request in requests) != 32:
        raise ValueError("promoted D1 admitted-position count changed")
    candidates = config["boundary_candidates"]
    if list(map(int, candidates["selected_router_ranks_one_indexed"])) != [6, 7, 8]:
        raise ValueError("selected boundary ranks changed")
    if list(map(int, candidates["outsider_router_ranks_one_indexed"])) != list(range(9, 17)):
        raise ValueError("outsider boundary ranks changed")
    if float(config["loss"]["temperature"]) != 0.0625:
        raise ValueError("D1 temperature changed")
    search = config["search"]
    if int(search["d1_group_coordinate_sweeps"]) != 8:
        raise ValueError("D1 coordinate sweep count changed")
    if int(search["d1_group_pair_passes"]) != 8:
        raise ValueError("D1 pair pass count changed")
    if int(search["exact_request_repair_coordinate_sweeps"]) != 2:
        raise ValueError("request repair coordinate sweep count changed")
    if int(search["exact_request_repair_pair_passes"]) != 2:
        raise ValueError("request repair pair pass count changed")
    hardware = config["hardware_execution_path"]
    if str(hardware["gpu"]) != "NVIDIA GeForce RTX 3090":
        raise ValueError("promoted D1 hardware path changed")
    if config.get("terminal_tail_quality_in_scope") is not False:
        raise ValueError("the layer-slice config must stop before terminal quality")


def validate_tail_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != "pr13_d1_downstream_tail_kl_config_v1":
        raise ValueError("unexpected D1 tail config schema")
    if config.get("base_d1_expansion_config_sha256") != (
        "de00bd6ada308a7693fc7816fdee3e1e6de1b2d6206a26a7f1dd61a094d0adfd"
    ):
        raise ValueError("tail study is not bound to the immutable expansion config")
    layers = list(map(int, config["injection_layers"]))
    if layers != [0, 1, 4, 6, 12, 23]:
        raise ValueError("tail injection layers changed")
    requests = list(map(str, config["validation_request_ids"]))
    if len(requests) != 3 or len(set(requests)) != 3:
        raise ValueError("tail smoke requires three unique requests")
    mechanism = list(map(int, config["mechanism_page_caps"]))
    matched = list(map(int, config["matched_runtime_metadata_page_caps"]))
    if mechanism != [384, 749] or matched != [360, 725]:
        raise ValueError("tail page-cap grid changed")
    if list(map(str, config["policies"])) != list(STANDARD_POLICIES):
        raise ValueError("tail comparison policy grid changed")
    if list(map(str, config["route_modes"])) != ["live", "fully_frozen"]:
        raise ValueError("tail routing-control grid changed")
    repair_directories = list(map(str, config["existing_repair_cell_dirs"]))
    if len(repair_directories) != 12 or len(set(repair_directories)) != 12:
        raise ValueError("existing exact-repair directory grid changed")
    calibration = config["fixed_d1_calibration"]
    expected_eta = np.asarray(
        [0.0, 0.00005, 0.0001, 0.00025, 0.0005, 0.001], np.float64,
    )
    if not np.array_equal(
        np.asarray(calibration["candidate_eta"], np.float64), expected_eta,
    ):
        raise ValueError("fixed-policy eta grid changed")
    if float(calibration["temperature"]) != 0.0625:
        raise ValueError("fixed-policy D1 temperature changed")
    if list(map(int, calibration["calibration_layers"])) != [0, 12, 23]:
        raise ValueError("fixed-policy calibration layers changed")
    if list(map(int, calibration["held_out_expansion_layers"])) != [1, 4, 6]:
        raise ValueError("fixed-policy held-out layers changed")
    if int(config["minimum_requests_before_quality_claim"]) < 64:
        raise ValueError("quality-claim request gate was weakened")
    hardware = config["hardware_execution_path"]
    if str(hardware["required_gpu_name_substring"]) != "RTX PRO 6000":
        raise ValueError("exact-tail GPU identity gate changed")
    if int(hardware["minimum_gpu_memory_gib"]) < 90:
        raise ValueError("exact-tail GPU memory gate was weakened")


@dataclass(frozen=True)
class OracleCell:
    layer: int
    rate: int
    directory: Path
    facts: Mapping[str, Any]
    metrics: pd.DataFrame
    allocations: pd.DataFrame
    deltas: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class PolicyDeltaBank:
    layer: int
    rate: int
    identity: pd.DataFrame
    deltas: Mapping[str, np.ndarray]
    local_qenergy_damage: Mapping[str, np.ndarray]
    provenance: Mapping[str, Any]


def load_oracle_cell(directory: str | Path) -> OracleCell:
    root = Path(directory)
    facts_path = root / "d1_layer_facts.json"
    facts = json.loads(facts_path.read_text())
    layer = int(facts["layer"])
    rates = list(map(int, facts["rates"]))
    if len(rates) != 1:
        raise ValueError("tail replay requires one rate per oracle cell")
    required = (
        "d1_exact_route_metrics.parquet",
        "d1_allocation_groups.parquet",
        "d1_selected_deltas.npz",
    )
    recorded = facts.get("files", {})
    for name in required:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if name not in recorded or sha256(path) != recorded[name]["sha256"]:
            raise RuntimeError(f"oracle cell hash changed: {root}/{name}")
    metrics = pd.read_parquet(root / required[0])
    allocations = pd.read_parquet(root / required[1])
    with np.load(root / required[2], allow_pickle=False) as loaded:
        deltas = {
            str(name): np.asarray(loaded[name], np.float32)
            for name in loaded.files
            if "__rate_" in str(name)
        }
    expected_shape = (int(facts["groups"]), 2048)
    if not deltas or any(value.shape != expected_shape for value in deltas.values()):
        raise RuntimeError("oracle delta bank shape changed")
    if any(np.any(~np.isfinite(value)) for value in deltas.values()):
        raise RuntimeError("oracle delta bank contains non-finite values")
    return OracleCell(
        layer=layer,
        rate=rates[0],
        directory=root,
        facts=facts,
        metrics=metrics,
        allocations=allocations,
        deltas=deltas,
    )


def index_oracle_cells(directories: Sequence[str | Path]) -> dict[tuple[int, int], OracleCell]:
    result: dict[tuple[int, int], OracleCell] = {}
    for directory in directories:
        cell = load_oracle_cell(directory)
        key = (cell.layer, cell.rate)
        if key in result:
            raise ValueError(f"duplicate oracle cell: {key}")
        result[key] = cell
    return result


def validate_expansion_artifacts(
    config: Mapping[str, Any],
    cells: Mapping[tuple[int, int], OracleCell],
) -> dict[str, Any]:
    """Prove that raw promoted cells implement the immutable expansion config."""

    validate_expansion_config(config)
    raw_root = Path(str(config["raw_result_root"]))
    raw_manifest = raw_root / "artifact_hashes.sha256"
    if sha256(raw_manifest) != str(config["raw_result_manifest_sha256"]):
        raise RuntimeError("promoted expansion raw manifest hash changed")
    manifest_entries = verify_sha256_manifest(raw_root, raw_manifest)
    if manifest_entries != int(config["raw_result_manifest_files"]):
        raise RuntimeError("promoted expansion raw manifest entry count changed")
    layers = list(map(int, config["injection_layers"]))
    rates = list(map(int, config["mean_budget_pages_per_expert"]))
    expected = {(layer, rate) for layer in layers for rate in rates}
    if set(cells) != expected:
        raise RuntimeError(
            f"promoted expansion cell grid changed: "
            f"missing={sorted(expected - set(cells))}, "
            f"unexpected={sorted(set(cells) - expected)}",
        )
    expected_eta = np.asarray(config["local_guard"]["eta_sweep"], np.float64)
    expected_temperature = np.asarray([config["loss"]["temperature"]], np.float64)
    requests = list(map(str, config["validation_request_ids"]))
    admitted = config["position_mask"]["admitted_positions_per_request"]
    records = {}
    for (layer, rate), cell in sorted(cells.items()):
        facts = cell.facts
        if int(facts["groups"]) != int(config["position_mask"]["expected_groups_per_layer"]):
            raise RuntimeError(f"group count changed in cell {(layer, rate)}")
        if str(facts["frontier_mode"]) != "column_generated":
            raise RuntimeError(f"frontier mode changed in cell {(layer, rate)}")
        if not np.array_equal(np.asarray(facts["eta"], np.float64), expected_eta):
            raise RuntimeError(f"eta grid changed in cell {(layer, rate)}")
        if not np.array_equal(
            np.asarray(facts["temperature"], np.float64), expected_temperature,
        ):
            raise RuntimeError(f"temperature changed in cell {(layer, rate)}")
        if str(facts["environment"]["gpu"]) != str(
            config["hardware_execution_path"]["gpu"]
        ):
            raise RuntimeError(f"hardware path changed in cell {(layer, rate)}")
        identity = (
            cell.allocations[["group", "request_id", "position"]]
            .drop_duplicates().sort_values("group", kind="stable")
        )
        if list(dict.fromkeys(identity["request_id"].astype(str))) != requests:
            raise RuntimeError(f"request order changed in cell {(layer, rate)}")
        for request in requests:
            positions = identity[
                identity["request_id"].astype(str).eq(request)
            ]["position"].to_numpy(np.int64)
            expected_positions = np.arange(int(admitted[request]), dtype=np.int64)
            if not np.array_equal(positions, expected_positions):
                raise RuntimeError(
                    f"position mask changed in cell {(layer, rate)} request {request}",
                )
        vjp_path = cell.directory / "d1_vjp_metrics.parquet"
        recorded = facts["files"].get(vjp_path.name)
        if recorded is None or sha256(vjp_path) != recorded["sha256"]:
            raise RuntimeError(f"VJP artifact changed in cell {(layer, rate)}")
        vjp = pd.read_parquet(vjp_path)
        if not np.all(vjp["exact_vjp_candidates"].to_numpy(np.int64) == 11):
            raise RuntimeError(f"boundary candidate width changed in cell {(layer, rate)}")
        records[f"layer_{layer:02d}_rate_{rate}"] = {
            "directory": str(cell.directory),
            "facts_sha256": sha256(cell.directory / "d1_layer_facts.json"),
            "groups": int(facts["groups"]),
            "eta": list(map(float, facts["eta"])),
            "temperature": list(map(float, facts["temperature"])),
            "gpu": str(facts["environment"]["gpu"]),
            "vjp_candidate_width": 11,
        }
    return {
        "completed": True,
        "schema": "pr13_d1_layer_slice_expansion_validation_v1",
        "run_id": str(config["run_id"]),
        "layers": layers,
        "rates": rates,
        "requests": requests,
        "groups_per_cell": int(config["position_mask"]["expected_groups_per_layer"]),
        "raw_manifest": str(raw_manifest),
        "raw_manifest_sha256": sha256(raw_manifest),
        "raw_manifest_entries": manifest_entries,
        "cells": records,
        "test_rows_admitted_or_used": False,
    }


def _candidate_request_choices(cell: OracleCell) -> pd.DataFrame:
    metrics = cell.metrics[cell.metrics["policy"].str.startswith(D1_PREFIX)].copy()
    allocations = cell.allocations[
        cell.allocations["policy"].str.startswith(D1_PREFIX)
    ][[
        "layer", "rate_pages_per_expert", "request_id", "policy",
        "selected_group_pages",
    ]]
    request = (
        metrics.groupby(
            ["layer", "rate_pages_per_expert", "request_id", "policy"],
            sort=True,
        )
        .agg(
            exact_crossings=("exact_d1_crossed", "sum"),
            membership_pairs_changed=("membership_pairs_changed", "sum"),
            routing_mass_lost=("routing_mass_lost", "sum"),
            local_qenergy_damage=("local_qenergy_damage", "sum"),
        )
        .reset_index()
    )
    pages = (
        allocations.groupby(
            ["layer", "rate_pages_per_expert", "request_id", "policy"],
            sort=True,
        )["selected_group_pages"].sum().reset_index(name="selected_pages")
    )
    request = request.merge(
        pages,
        on=["layer", "rate_pages_per_expert", "request_id", "policy"],
        validate="one_to_one",
    )
    choices = []
    for _, frame in request.groupby(
        ["layer", "rate_pages_per_expert", "request_id"], sort=True,
    ):
        chosen = frame.sort_values([
            "exact_crossings",
            "membership_pairs_changed",
            "routing_mass_lost",
            "local_qenergy_damage",
            "selected_pages",
            "policy",
        ], kind="stable").iloc[0]
        choices.append(chosen.to_dict())
    return pd.DataFrame(choices)


def calibrate_fixed_d1_policy(
    cells: Mapping[tuple[int, int], OracleCell],
    calibration_layers: Sequence[int],
    rates: Sequence[int],
) -> tuple[dict[int, str], pd.DataFrame]:
    """Select one complete D1 eta policy per rate on frozen calibration layers."""

    calibration_set = set(map(int, calibration_layers))
    selections: dict[int, str] = {}
    evidence = []
    for rate in map(int, rates):
        metric_parts = []
        allocation_parts = []
        for layer in sorted(calibration_set):
            cell = cells.get((layer, rate))
            if cell is None:
                raise KeyError(f"missing calibration cell {(layer, rate)}")
            metric_parts.append(
                cell.metrics[cell.metrics["policy"].str.startswith(D1_PREFIX)]
            )
            allocation_parts.append(
                cell.allocations[
                    cell.allocations["policy"].str.startswith(D1_PREFIX)
                ]
            )
        metrics = pd.concat(metric_parts, ignore_index=True)
        allocations = pd.concat(allocation_parts, ignore_index=True)
        summary = (
            metrics.groupby("policy", sort=True)
            .agg(
                exact_crossings=("exact_d1_crossed", "sum"),
                membership_pairs_changed=("membership_pairs_changed", "sum"),
                routing_mass_lost=("routing_mass_lost", "sum"),
                local_qenergy_damage=("local_qenergy_damage", "sum"),
            )
            .reset_index()
        )
        pages = (
            allocations.groupby("policy", sort=True)["selected_group_pages"]
            .sum().reset_index(name="selected_pages")
        )
        summary = summary.merge(pages, on="policy", validate="one_to_one")
        summary.insert(0, "rate_pages_per_expert", rate)
        summary["calibration_layers"] = json.dumps(sorted(calibration_set))
        ordered = summary.sort_values([
            "exact_crossings",
            "membership_pairs_changed",
            "routing_mass_lost",
            "local_qenergy_damage",
            "selected_pages",
            "policy",
        ], kind="stable")
        selected = str(ordered.iloc[0]["policy"])
        summary["selected"] = summary["policy"].eq(selected)
        selections[rate] = selected
        evidence.append(summary)
    return selections, pd.concat(evidence, ignore_index=True)


def _policy_key(policy: str, rate: int) -> str:
    return f"{policy}__rate_{int(rate)}"


def _single_source_policy(cell: OracleCell, prefix: str) -> str:
    suffix = f"__rate_{cell.rate}"
    choices = sorted(
        name[: -len(suffix)] for name in cell.deltas
        if name.endswith(suffix) and name.startswith(prefix)
    )
    if len(choices) != 1:
        raise RuntimeError(
            f"expected one {prefix!r} policy in cell {(cell.layer, cell.rate)}; "
            f"found {choices}",
        )
    return choices[0]


def _repair_delta(repair_directory: Path, layer: int, rate: int) -> np.ndarray:
    facts_path = repair_directory / "d1_exact_repair_facts.json"
    facts = json.loads(facts_path.read_text())
    if int(facts["layer"]) != int(layer) or int(facts["rate_pages_per_expert"]) != int(rate):
        raise RuntimeError("repair identity changed")
    path = repair_directory / "d1_exact_repair_delta.npz"
    expected = facts["outputs"][path.name]["sha256"]
    if sha256(path) != expected:
        raise RuntimeError("repair delta hash changed")
    with np.load(path, allow_pickle=False) as loaded:
        result = np.asarray(loaded["delta"], np.float32)
    return result


def _group_local_qenergy(cell: OracleCell, policy: str) -> np.ndarray:
    rows = cell.metrics[
        cell.metrics["policy"].astype(str).eq(str(policy))
        & cell.metrics["rate_pages_per_expert"].eq(int(cell.rate))
    ].sort_values("group", kind="stable")
    expected = int(cell.facts["groups"])
    if len(rows) != expected or not np.array_equal(
        rows["group"].to_numpy(np.int64), np.arange(expected),
    ):
        raise RuntimeError(f"local-qenergy group grid changed for policy {policy}")
    result = rows["local_qenergy_damage"].to_numpy(np.float64)
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise RuntimeError(f"invalid local-qenergy values for policy {policy}")
    return result


def _repair_local_qenergy(
    repair_directory: Path,
    facts: Mapping[str, Any],
    expected_groups: int,
) -> np.ndarray:
    path = repair_directory / "d1_exact_repair_metrics.parquet"
    expected_hash = facts["outputs"][path.name]["sha256"]
    if sha256(path) != expected_hash:
        raise RuntimeError("repair metric hash changed")
    rows = pd.read_parquet(path).sort_values("group", kind="stable")
    if len(rows) != int(expected_groups) or not np.array_equal(
        rows["group"].to_numpy(np.int64), np.arange(int(expected_groups)),
    ):
        raise RuntimeError("repair local-qenergy group grid changed")
    result = rows["local_qenergy_damage"].to_numpy(np.float64)
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise RuntimeError("repair local-qenergy values are invalid")
    return result


def assemble_policy_delta_bank(
    cell: OracleCell,
    repair_directory: str | Path,
    fixed_source_policy: str,
) -> PolicyDeltaBank:
    """Assemble five final policies from one exact oracle cell."""

    rate = int(cell.rate)
    allocation = cell.allocations.copy()
    identity = (
        allocation[["group", "request_id", "position", "layer"]]
        .drop_duplicates()
        .sort_values("group", kind="stable")
        .reset_index(drop=True)
    )
    expected_groups = int(cell.facts["groups"])
    if len(identity) != expected_groups:
        raise RuntimeError("oracle group identity count changed")
    if not np.array_equal(identity["group"].to_numpy(np.int64), np.arange(expected_groups)):
        raise RuntimeError("oracle group IDs are not contiguous")

    historical_key = _policy_key(HISTORICAL_PR13, rate)
    regenerated_key = _policy_key(REGENERATED_PR13, rate)
    if historical_key in cell.deltas:
        pr13_source = HISTORICAL_PR13
    elif regenerated_key in cell.deltas:
        pr13_source = REGENERATED_PR13
    else:
        raise RuntimeError("cell has neither historical nor regenerated router-square delta")
    local_source = _single_source_policy(cell, LOCAL_PREFIX)
    fixed_key = _policy_key(str(fixed_source_policy), rate)
    if fixed_key not in cell.deltas or not str(fixed_source_policy).startswith(D1_PREFIX):
        raise RuntimeError("calibrated fixed D1 source is absent from cell")

    choices = _candidate_request_choices(cell)
    reranked = np.zeros_like(next(iter(cell.deltas.values())), np.float32)
    source_by_request: dict[str, str] = {}
    for row in choices.itertuples(index=False):
        request_id = str(row.request_id)
        source = str(row.policy)
        source_by_request[request_id] = source
        mask = identity["request_id"].astype(str).eq(request_id).to_numpy()
        reranked[mask] = cell.deltas[_policy_key(source, rate)][mask]
    if set(source_by_request) != set(identity["request_id"].astype(str)):
        raise RuntimeError("request reranking omitted a request")

    repair_root = Path(repair_directory)
    repaired = _repair_delta(repair_root, cell.layer, rate)
    if repaired.shape != reranked.shape or np.any(~np.isfinite(repaired)):
        raise RuntimeError("repair delta shape or finiteness changed")
    repair_facts = json.loads((repair_root / "d1_exact_repair_facts.json").read_text())
    pr13_local = _group_local_qenergy(cell, pr13_source)
    local_local = _group_local_qenergy(cell, local_source)
    fixed_local = _group_local_qenergy(cell, str(fixed_source_policy))
    reranked_local = np.zeros(expected_groups, np.float64)
    for request_id, source in source_by_request.items():
        mask = identity["request_id"].astype(str).eq(request_id).to_numpy()
        reranked_local[mask] = _group_local_qenergy(cell, source)[mask]
    repaired_local = _repair_local_qenergy(
        repair_root, repair_facts, expected_groups,
    )
    result = {
        PR13_POLICY: cell.deltas[_policy_key(pr13_source, rate)].copy(),
        LOCAL_POLICY: cell.deltas[_policy_key(local_source, rate)].copy(),
        FIXED_D1_POLICY: cell.deltas[fixed_key].copy(),
        RERANKED_D1_POLICY: reranked,
        REPAIRED_D1_POLICY: repaired,
    }
    if tuple(result) != STANDARD_POLICIES:
        raise RuntimeError("standard policy ordering changed")
    local_qenergy = {
        PR13_POLICY: pr13_local,
        LOCAL_POLICY: local_local,
        FIXED_D1_POLICY: fixed_local,
        RERANKED_D1_POLICY: reranked_local,
        REPAIRED_D1_POLICY: repaired_local,
    }
    return PolicyDeltaBank(
        layer=cell.layer,
        rate=rate,
        identity=identity,
        deltas=result,
        local_qenergy_damage=local_qenergy,
        provenance={
            "pr13_source_policy": pr13_source,
            "pr13_source_kind": (
                "preserved_historical" if pr13_source == HISTORICAL_PR13
                else "regenerated_same_algorithm_nonhistorical_cap"
            ),
            "local_source_policy": local_source,
            "fixed_d1_source_policy": str(fixed_source_policy),
            "reranked_source_policy_by_request": source_by_request,
            "repair_directory": str(Path(repair_directory)),
            "oracle_directory": str(cell.directory),
        },
    )


def request_delta(
    bank: PolicyDeltaBank,
    policy: str,
    request_id: str,
    sequence_length: int,
    hidden_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    if policy not in bank.deltas:
        raise KeyError(policy)
    rows = bank.identity[bank.identity["request_id"].astype(str).eq(str(request_id))]
    positions = rows["position"].to_numpy(np.int64)
    groups = rows["group"].to_numpy(np.int64)
    if len(rows) == 0 or np.any(positions < 0) or np.any(positions >= int(sequence_length)):
        raise RuntimeError("request group positions are invalid")
    result = np.zeros((int(sequence_length), int(hidden_size)), np.float32)
    result[positions] = np.asarray(bank.deltas[policy], np.float32)[groups]
    return result, positions


def request_policy_local_metrics(
    bank: PolicyDeltaBank,
    policy: str,
    request_id: str,
) -> dict[str, float]:
    if policy not in bank.deltas or policy not in bank.local_qenergy_damage:
        raise KeyError(policy)
    rows = bank.identity[bank.identity["request_id"].astype(str).eq(str(request_id))]
    groups = rows["group"].to_numpy(np.int64)
    if len(groups) == 0:
        raise RuntimeError("request has no admitted allocation groups")
    qenergy = np.asarray(bank.local_qenergy_damage[policy], np.float64)[groups]
    delta = np.asarray(bank.deltas[policy], np.float64)[groups]
    return {
        "selected_local_qenergy_damage_sum": float(np.sum(qenergy)),
        "selected_local_qenergy_damage_per_group": float(np.mean(qenergy)),
        "injected_delta_mse": float(np.mean(delta * delta)),
    }


def downstream_tail_metrics(
    *,
    reference_logits: np.ndarray,
    candidate_logits: np.ndarray,
    input_ids: np.ndarray,
    reference_final_hidden: np.ndarray,
    candidate_final_hidden: np.ndarray,
    reference_router: Mapping[int, np.ndarray],
    candidate_router: Mapping[int, np.ndarray],
    injection_layer: int,
    admitted_positions: Sequence[int],
) -> dict[str, Any]:
    """Compute terminal quality and downstream routing metrics for one tail."""

    labels = np.asarray(input_ids, np.int64)[1:]
    quality = token_quality_metrics(
        np.asarray(reference_logits, np.float32)[:-1],
        np.asarray(candidate_logits, np.float32)[:-1],
        labels,
    )
    layers = list(range(int(injection_layer) + 1, 40))
    if any(layer not in reference_router or layer not in candidate_router for layer in layers):
        raise ValueError("downstream router map is incomplete")
    route_rows = []
    first_rows = []
    admitted = np.asarray(admitted_positions, np.int64)
    for layer in layers:
        reference = np.asarray(reference_router[layer], np.float32)
        candidate = np.asarray(candidate_router[layer], np.float32)
        metrics = route_boundary_metrics(reference, candidate)
        admitted_metrics = route_boundary_metrics(reference[admitted], candidate[admitted])
        route_rows.append((layer, metrics, admitted_metrics))
        first_rows.append((layer, admitted_metrics["route_membership_change_fraction"]))
    d1 = route_rows[0][2]
    result: dict[str, Any] = {
        **quality,
        "final_hidden_mse": mean_squared_error(
            np.asarray(reference_final_hidden, np.float32),
            np.asarray(candidate_final_hidden, np.float32),
        ),
        "d1_route_membership_change_fraction": float(
            d1["route_membership_change_fraction"]
        ),
        "d1_router_mass_churn": float(d1["router_mass_churn"]),
        "mean_downstream_route_membership_change_fraction": float(np.mean([
            row[1]["route_membership_change_fraction"] for row in route_rows
        ])),
        "mean_downstream_router_mass_churn": float(np.mean([
            row[1]["router_mass_churn"] for row in route_rows
        ])),
        "final_layer_route_membership_change_fraction": float(
            route_rows[-1][1]["route_membership_change_fraction"]
        ),
        "final_layer_router_mass_churn": float(route_rows[-1][1]["router_mass_churn"]),
        "first_route_membership_change_layer": first_changed_layer(first_rows),
    }
    if any(
        not np.isfinite(float(value))
        for key, value in result.items()
        if key != "first_route_membership_change_layer" and value is not None
    ):
        raise RuntimeError("downstream tail produced a non-finite metric")
    return result


def add_live_minus_frozen(quality: pd.DataFrame) -> pd.DataFrame:
    """Attach live minus fully-frozen KL to every paired quality row."""

    identity = [
        "injection_layer", "request_id", "rate_pages_per_expert", "policy",
    ]
    if quality.duplicated(identity + ["route_mode"]).any():
        raise ValueError("quality route-mode identity is not unique")
    live = quality[quality["route_mode"].eq("live")][identity + ["logit_kl"]].rename(
        columns={"logit_kl": "live_logit_kl"},
    )
    frozen = quality[quality["route_mode"].eq("fully_frozen")][identity + ["logit_kl"]].rename(
        columns={"logit_kl": "fully_frozen_logit_kl"},
    )
    paired = live.merge(frozen, on=identity, validate="one_to_one")
    paired["live_minus_fully_frozen_logit_kl"] = (
        paired["live_logit_kl"] - paired["fully_frozen_logit_kl"]
    )
    return quality.merge(paired, on=identity, validate="many_to_one")
