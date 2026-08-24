"""Policy and metric primitives for exact-prefix cached single-token tails.

Unlike :mod:`oracle_study.d1_tail_quality`, this module never assembles a
sequence-shaped perturbation.  One bank row is one isolated current decode
token behind an exact prefix cache.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .causal_replay import mean_squared_error, route_boundary_metrics, token_quality_metrics
from .d1_decode import cache_tensor_items
from .d1_tail_quality import sha256, verify_sha256_manifest


PR13_POLICY = "pr13_router_square"
LOCAL_POLICY = "exact_combined_local"
FIXED_D1_POLICY = "calibration_selected_fixed_d1"
COMPANION_D1_POLICY = "frontier_companion_d1"
TOKEN_ORACLE_POLICY = "exact_token_oracle"
DECODE_POLICIES = (
    PR13_POLICY,
    LOCAL_POLICY,
    FIXED_D1_POLICY,
    COMPANION_D1_POLICY,
    TOKEN_ORACLE_POLICY,
)

HISTORICAL_PR13 = "pr13_router_square_column_generated"
REGENERATED_PR13 = "regenerated_router_square_column_generated"
LOCAL_SOURCE = "exact_combined_local_column_generated_frontier"
D1_PREFIX = "d1_strict_"


@dataclass(frozen=True)
class DecodeOracleCell:
    layer: int
    rate: int
    directory: Path
    facts: Mapping[str, Any]
    metrics: pd.DataFrame
    allocations: pd.DataFrame
    expert_allocations: pd.DataFrame
    token_oracle: pd.DataFrame
    deltas: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class DecodePolicyBank:
    layer: int
    rate: int
    identity: pd.DataFrame
    deltas: Mapping[str, np.ndarray]
    local_qenergy_damage: Mapping[str, np.ndarray]
    selected_group_pages: Mapping[str, np.ndarray]
    expert_ids: np.ndarray
    selected_states: Mapping[str, np.ndarray]
    provenance: Mapping[str, Any]


def validate_cached_decode_tail_config(config: Mapping[str, Any]) -> None:
    """Reject the old full-sequence contract and validate the decode grid."""

    if config.get("schema") != "pr13_d1_cached_decode_tail_kl_config_v2":
        raise ValueError("unexpected cached-decode tail config schema")
    if list(map(int, config["injection_layers"])) != [0, 1, 4, 6, 12, 23]:
        raise ValueError("cached-decode injection-layer grid changed")
    if list(map(int, config["page_caps"])) != [360, 384, 725, 749]:
        raise ValueError("cached-decode page-cap grid changed")
    if list(map(str, config["policies"])) != list(DECODE_POLICIES):
        raise ValueError("cached-decode policy grid changed")
    if list(map(str, config["route_modes"])) != ["live", "fully_frozen"]:
        raise ValueError("cached-decode route-mode grid changed")
    requests = list(map(str, config["validation_request_ids"]))
    if requests != ["mxfp4-confirm-006", "mxfp4-confirm-007", "mxfp4-confirm-008"]:
        raise ValueError("cached-decode request cohort changed")
    mask = config["position_mask"]
    admitted = {str(key): int(value) for key, value in mask["admitted_positions_per_request"].items()}
    if int(mask["expected_groups_per_layer"]) != 29 or sum(admitted.values()) != 29:
        raise ValueError("cached-decode tail must contain 29 isolated positions")
    if int(mask["minimum_exact_prefix_tokens"]) != 1:
        raise ValueError("cached-decode tail requires a non-empty exact prefix")
    if bool(mask.get("cross_position_delta_coupling", True)):
        raise ValueError("cross-position delta coupling is forbidden")
    contract = config["decode_execution"]
    required = {
        "query_length": 1,
        "candidate_prefix_cache": "deep_clone_complete_exact_prefix",
        "terminal_metric_scope": "current_token_only",
        "retain_post_token_cache": True,
        "provisional_or_speculative_pass": False,
    }
    for name, expected in required.items():
        if contract.get(name) != expected:
            raise ValueError(f"cached-decode execution contract changed: {name}")
    forbidden = {
        "existing_repair_cell_dirs",
        "matched_repair_dir_template",
        "source_capture_dir",
    }
    present = sorted(forbidden.intersection(config))
    if present:
        raise ValueError(f"legacy full-sequence inputs are forbidden: {present}")
    sources = config["d1_source_policy_by_rate"]
    for rate in map(int, config["page_caps"]):
        record = sources.get(str(rate))
        if not isinstance(record, Mapping):
            raise ValueError(f"missing frozen D1 source mapping for rate {rate}")
        fixed = str(record.get("fixed", ""))
        companion = str(record.get("companion", ""))
        if not fixed.startswith(D1_PREFIX) or not companion.startswith(D1_PREFIX):
            raise ValueError(f"invalid D1 source policy mapping for rate {rate}")
        if fixed == companion:
            raise ValueError(f"fixed and companion D1 policies coincide at rate {rate}")
    reconstruction = config["live_reconstruction"]
    if reconstruction.get("candidate_delta") != (
        "complete_selected_states_reevaluated_on_live_cached_activation"
    ):
        raise ValueError("candidate delta must be reconstructed on the live activation")
    if reconstruction.get("stored_slice_delta") != "transfer_diagnostic_only":
        raise ValueError("stored slice deltas must remain diagnostic only")
    if reconstruction.get("route_set_match_required") is not True:
        raise ValueError("live/source routed expert set equality must be required")
    tolerance = float(reconstruction["q4_routed_max_abs_atol"])
    if not (0.0 < tolerance <= 0.125):
        raise ValueError("live Q4 routed-output tolerance is invalid")
    if len(str(config.get("selected_tree_sha256", ""))) != 64:
        raise ValueError("selected-tree digest is missing")
    hardware = config["hardware_execution_path"]
    if str(hardware["required_gpu_name_substring"]) != "RTX PRO 6000":
        raise ValueError("full cached tail requires the PRO 6000 execution class")
    if int(hardware["minimum_gpu_memory_gib"]) < 90:
        raise ValueError("full cached tail memory gate was weakened")
    if int(config["minimum_requests_before_quality_claim"]) < 64:
        raise ValueError("quality-claim request gate was weakened")


def cell_directory(config: Mapping[str, Any], layer: int, rate: int) -> Path:
    mechanism = set(map(int, config["mechanism_page_caps"]))
    matched = set(map(int, config["matched_runtime_metadata_page_caps"]))
    if int(rate) in mechanism:
        template = str(config["mechanism_cell_dir_template"])
    elif int(rate) in matched:
        template = str(config["matched_cell_dir_template"])
    else:
        raise KeyError(f"unconfigured page cap: {rate}")
    return Path(template.format(layer=int(layer), rate=int(rate)))


def _verify_recorded(root: Path, facts: Mapping[str, Any], names: Sequence[str]) -> None:
    recorded = facts.get("files", {})
    for name in names:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if name not in recorded or str(recorded[name].get("sha256")) != sha256(path):
            raise RuntimeError(f"cached-decode oracle artifact changed: {path}")


def load_decode_oracle_cell(directory: str | Path) -> DecodeOracleCell:
    root = Path(directory)
    facts_path = root / "d1_decode_layer_facts.json"
    if not facts_path.is_file():
        raise FileNotFoundError(facts_path)
    facts = json.loads(facts_path.read_text())
    if facts.get("schema") != "pr13_d1_exact_prefill_decode_slice_oracle_v2":
        raise ValueError(f"unexpected decode oracle schema: {root}")
    if not bool(facts.get("single_current_token_injection")):
        raise ValueError("decode cell is not single-token")
    if bool(facts.get("prompt_position_coupling_allowed", True)):
        raise ValueError("decode cell permits cross-position coupling")
    rates = list(map(int, facts["rates"]))
    if len(rates) != 1:
        raise ValueError("one page cap is required per decode oracle cell")
    required = (
        "d1_decode_exact_route_metrics.parquet",
        "d1_decode_allocation_groups.parquet",
        "d1_decode_allocation_experts.parquet",
        "d1_decode_selected_deltas.npz",
        "d1_decode_token_oracle.parquet",
        "d1_decode_cache_parity.parquet",
    )
    _verify_recorded(root, facts, required)
    metrics = pd.read_parquet(root / required[0])
    allocations = pd.read_parquet(root / required[1])
    expert_allocations = pd.read_parquet(root / required[2])
    token_oracle = pd.read_parquet(root / required[4])
    with np.load(root / required[3], allow_pickle=False) as loaded:
        deltas = {
            str(name): np.asarray(loaded[name], np.float32)
            for name in loaded.files
            if "__rate_" in str(name)
        }
    groups = int(facts["groups"])
    expected_shape = (groups, 2048)
    if groups != 29:
        raise RuntimeError("decode oracle group count changed")
    if not deltas or any(value.shape != expected_shape for value in deltas.values()):
        raise RuntimeError("decode delta bank shape changed")
    if any(np.any(~np.isfinite(value)) for value in deltas.values()):
        raise RuntimeError("decode delta bank contains non-finite values")
    return DecodeOracleCell(
        layer=int(facts["layer"]),
        rate=rates[0],
        directory=root,
        facts=facts,
        metrics=metrics,
        allocations=allocations,
        expert_allocations=expert_allocations,
        token_oracle=token_oracle,
        deltas=deltas,
    )


def index_decode_cells(
    config: Mapping[str, Any],
) -> dict[tuple[int, int], DecodeOracleCell]:
    manifest = Path(str(config["matched_raw_manifest"]))
    if sha256(manifest) != str(config["matched_raw_manifest_sha256"]):
        raise RuntimeError("matched-rate raw manifest changed")
    entries = verify_sha256_manifest(manifest.parent, manifest)
    if entries != int(config["matched_raw_manifest_files"]):
        raise RuntimeError(
            f"matched-rate raw manifest count changed: {entries}"
        )
    cells: dict[tuple[int, int], DecodeOracleCell] = {}
    for layer in map(int, config["injection_layers"]):
        for rate in map(int, config["page_caps"]):
            cell = load_decode_oracle_cell(cell_directory(config, layer, rate))
            key = (layer, rate)
            if (cell.layer, cell.rate) != key:
                raise RuntimeError(f"decode cell identity changed: expected {key}")
            cells[key] = cell
    return cells


def _identity(cell: DecodeOracleCell) -> pd.DataFrame:
    columns = ["group", "request_id", "position", "layer"]
    result = (
        cell.allocations[columns]
        .drop_duplicates()
        .sort_values("group", kind="stable")
        .reset_index(drop=True)
    )
    expected = np.arange(int(cell.facts["groups"]), dtype=np.int64)
    if len(result) != len(expected) or not np.array_equal(
        result["group"].to_numpy(np.int64), expected,
    ):
        raise RuntimeError("decode group identity is incomplete or non-contiguous")
    if np.any(result["position"].to_numpy(np.int64) < 1):
        raise RuntimeError("decode group does not have a non-empty exact prefix")
    return result


def _policy_key(policy: str, rate: int) -> str:
    return f"{policy}__rate_{int(rate)}"


def _metric_vector(cell: DecodeOracleCell, policy: str, name: str) -> np.ndarray:
    rows = cell.metrics[
        cell.metrics["policy"].astype(str).eq(str(policy))
        & cell.metrics["rate_pages_per_expert"].eq(int(cell.rate))
    ].sort_values("group", kind="stable")
    expected = np.arange(int(cell.facts["groups"]), dtype=np.int64)
    if len(rows) != len(expected) or not np.array_equal(
        rows["group"].to_numpy(np.int64), expected,
    ):
        raise RuntimeError(f"decode metric grid changed for {policy}")
    values = rows[name].to_numpy()
    if np.issubdtype(values.dtype, np.number) and np.any(~np.isfinite(values)):
        raise RuntimeError(f"non-finite decode metric {name} for {policy}")
    return values


def _page_vector(cell: DecodeOracleCell, policy: str) -> np.ndarray:
    rows = cell.allocations[
        cell.allocations["policy"].astype(str).eq(str(policy))
        & cell.allocations["rate_pages_per_expert"].eq(int(cell.rate))
    ].sort_values("group", kind="stable")
    expected = np.arange(int(cell.facts["groups"]), dtype=np.int64)
    if len(rows) != len(expected) or not np.array_equal(
        rows["group"].to_numpy(np.int64), expected,
    ):
        raise RuntimeError(f"decode allocation grid changed for {policy}")
    return rows["selected_group_pages"].to_numpy(np.int64)


def _expert_policy_arrays(
    cell: DecodeOracleCell,
    policy: str,
) -> tuple[np.ndarray, np.ndarray]:
    rows = cell.expert_allocations[
        cell.expert_allocations["policy"].astype(str).eq(str(policy))
        & cell.expert_allocations["rate_pages_per_expert"].eq(int(cell.rate))
    ].sort_values(["group", "router_rank"], kind="stable")
    groups = int(cell.facts["groups"])
    if len(rows) != groups * 8:
        raise RuntimeError(f"decode expert allocation grid changed for {policy}")
    expected_groups = np.repeat(np.arange(groups, dtype=np.int64), 8)
    expected_ranks = np.tile(np.arange(1, 9, dtype=np.int64), groups)
    if not np.array_equal(rows["group"].to_numpy(np.int64), expected_groups):
        raise RuntimeError(f"decode expert group order changed for {policy}")
    if not np.array_equal(rows["router_rank"].to_numpy(np.int64), expected_ranks):
        raise RuntimeError(f"decode expert rank order changed for {policy}")
    ids = rows["expert_id"].to_numpy(np.int64).reshape(groups, 8)
    states = np.empty((groups, 8, 512), np.int8)
    for index, value in enumerate(rows["selected_states"].astype(str)):
        parsed = np.asarray(json.loads(value), np.int8)
        if parsed.shape != (512,) or np.any(parsed < 0) or np.any(parsed > 7):
            raise RuntimeError(f"invalid complete state vector for {policy}")
        states[index // 8, index % 8] = parsed
    return ids, states


def calibrate_fixed_d1_policy(
    cells: Mapping[tuple[int, int], DecodeOracleCell],
    calibration_layers: Sequence[int],
    rates: Sequence[int],
) -> tuple[dict[int, str], pd.DataFrame]:
    """Select one eta per rate without reading held-out layer labels."""

    selected: dict[int, str] = {}
    evidence: list[pd.DataFrame] = []
    for rate in map(int, rates):
        metric_parts = []
        allocation_parts = []
        for layer in map(int, calibration_layers):
            cell = cells[(layer, rate)]
            metric_parts.append(cell.metrics[cell.metrics["policy"].str.startswith(D1_PREFIX)])
            allocation_parts.append(
                cell.allocations[cell.allocations["policy"].str.startswith(D1_PREFIX)]
            )
        metrics = pd.concat(metric_parts, ignore_index=True)
        allocations = pd.concat(allocation_parts, ignore_index=True)
        summary = (
            metrics.groupby("policy", sort=True)
            .agg(
                exact_crossings=("exact_d1_crossed", "sum"),
                membership_pairs_changed=("membership_pairs_changed", "sum"),
                routing_mass_lost=("routing_mass_lost", "sum"),
                local_qenergy_damage=("local_qenergy_damage", "mean"),
            )
            .reset_index()
        )
        pages = (
            allocations.groupby("policy", sort=True)["selected_group_pages"]
            .mean().reset_index(name="selected_group_pages")
        )
        summary = summary.merge(pages, on="policy", validate="one_to_one")
        summary.insert(0, "rate_pages_per_expert", rate)
        ordered = summary.sort_values(
            [
                "exact_crossings",
                "membership_pairs_changed",
                "routing_mass_lost",
                "local_qenergy_damage",
                "selected_group_pages",
                "policy",
            ],
            kind="stable",
        )
        choice = str(ordered.iloc[0]["policy"])
        summary["selected"] = summary["policy"].astype(str).eq(choice)
        selected[rate] = choice
        evidence.append(summary)
    return selected, pd.concat(evidence, ignore_index=True)


def assemble_decode_policy_bank(
    cell: DecodeOracleCell,
    *,
    fixed_source_policy: str,
    companion_source_policy: str,
) -> DecodePolicyBank:
    """Assemble four fixed policies and a per-token exact-label oracle."""

    rate = int(cell.rate)
    identity = _identity(cell)
    historical_key = _policy_key(HISTORICAL_PR13, rate)
    regenerated_key = _policy_key(REGENERATED_PR13, rate)
    allocation_policies = set(cell.allocations["policy"].astype(str))
    if historical_key in cell.deltas:
        pr13_delta_source = HISTORICAL_PR13
        if HISTORICAL_PR13 in allocation_policies:
            pr13_source = HISTORICAL_PR13
            pr13_kind = "preserved_historical"
        elif REGENERATED_PR13 in allocation_policies and regenerated_key in cell.deltas:
            delta_error = float(np.max(np.abs(
                cell.deltas[historical_key].astype(np.float64)
                - cell.deltas[regenerated_key].astype(np.float64)
            ), initial=0.0))
            if delta_error > 1e-12:
                raise RuntimeError("regenerated PR13 state table exceeds historical roundoff")
            pr13_source = REGENERATED_PR13
            pr13_kind = "preserved_historical_delta_with_audited_regenerated_states_roundoff_atol_1e-12"
        else:
            raise RuntimeError("historical PR13 delta has no executable state table")
    elif regenerated_key in cell.deltas:
        pr13_delta_source = REGENERATED_PR13
        pr13_source = REGENERATED_PR13
        pr13_kind = "regenerated_same_algorithm_nonhistorical_cap"
    else:
        raise RuntimeError("decode cell has no router-square comparator")
    sources = {
        PR13_POLICY: pr13_source,
        LOCAL_POLICY: LOCAL_SOURCE,
        FIXED_D1_POLICY: str(fixed_source_policy),
        COMPANION_D1_POLICY: str(companion_source_policy),
    }
    delta_sources = dict(sources)
    delta_sources[PR13_POLICY] = pr13_delta_source
    for source in sources.values():
        if _policy_key(source, rate) not in cell.deltas:
            raise RuntimeError(f"decode source policy is absent: {source}")

    deltas: dict[str, np.ndarray] = {
        policy: cell.deltas[_policy_key(source, rate)].copy()
        for policy, source in delta_sources.items()
    }
    local = {
        policy: _metric_vector(cell, source, "local_qenergy_damage").astype(np.float64)
        for policy, source in sources.items()
    }
    pages = {
        policy: _page_vector(cell, source)
        for policy, source in sources.items()
    }
    source_arrays = {
        policy: _expert_policy_arrays(cell, source)
        for policy, source in sources.items()
    }
    reference_ids = source_arrays[PR13_POLICY][0]
    for policy, (ids, _) in source_arrays.items():
        if not np.array_equal(ids, reference_ids):
            raise RuntimeError(
                f"source routed-expert identities differ for policy {policy}"
            )
    selected_states: dict[str, np.ndarray] = {
        policy: values[1].copy()
        for policy, values in source_arrays.items()
    }

    oracle = cell.token_oracle.sort_values("group", kind="stable").reset_index(drop=True)
    expected = np.arange(int(cell.facts["groups"]), dtype=np.int64)
    if len(oracle) != len(expected) or not np.array_equal(
        oracle["group"].to_numpy(np.int64), expected,
    ):
        raise RuntimeError("exact token oracle grid changed")
    oracle_delta = np.empty((len(expected), 2048), np.float32)
    oracle_states = np.empty((len(expected), 8, 512), np.int8)
    oracle_sources: dict[int, str] = {}
    for row in oracle.itertuples(index=False):
        group = int(row.group)
        source = str(row.policy)
        if not source.startswith(D1_PREFIX) or _policy_key(source, rate) not in cell.deltas:
            raise RuntimeError(f"invalid exact token oracle source: {source}")
        oracle_delta[group] = cell.deltas[_policy_key(source, rate)][group]
        ids, states = _expert_policy_arrays(cell, source)
        if not np.array_equal(ids[group], reference_ids[group]):
            raise RuntimeError("exact token oracle routed experts changed")
        oracle_states[group] = states[group]
        oracle_sources[group] = source
    deltas[TOKEN_ORACLE_POLICY] = oracle_delta
    selected_states[TOKEN_ORACLE_POLICY] = oracle_states
    local[TOKEN_ORACLE_POLICY] = oracle["local_qenergy_damage"].to_numpy(np.float64)
    pages[TOKEN_ORACLE_POLICY] = oracle["selected_group_pages"].to_numpy(np.int64)
    if tuple(deltas) != DECODE_POLICIES:
        raise RuntimeError("decode policy order changed")
    return DecodePolicyBank(
        layer=cell.layer,
        rate=rate,
        identity=identity,
        deltas=deltas,
        local_qenergy_damage=local,
        selected_group_pages=pages,
        expert_ids=reference_ids.copy(),
        selected_states=selected_states,
        provenance={
            "oracle_directory": str(cell.directory),
            "pr13_source_policy": pr13_source,
            "pr13_delta_source_policy": pr13_delta_source,
            "pr13_source_kind": pr13_kind,
            "local_source_policy": LOCAL_SOURCE,
            "fixed_d1_source_policy": str(fixed_source_policy),
            "companion_d1_source_policy": str(companion_source_policy),
            "exact_token_oracle_source_policy_by_group": oracle_sources,
        },
    )


def token_delta(
    bank: DecodePolicyBank,
    policy: str,
    *,
    group: int,
    request_id: str,
    position: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Return exactly one token delta and its allocation-local diagnostics."""

    if policy not in bank.deltas:
        raise KeyError(policy)
    row = bank.identity[bank.identity["group"].eq(int(group))]
    if len(row) != 1:
        raise RuntimeError("decode token group identity is not unique")
    item = row.iloc[0]
    if str(item["request_id"]) != str(request_id) or int(item["position"]) != int(position):
        raise RuntimeError("decode token identity changed")
    delta = np.asarray(bank.deltas[policy][int(group)], np.float32)
    if delta.shape != (2048,) or np.any(~np.isfinite(delta)):
        raise RuntimeError("isolated decode delta is invalid")
    return delta.copy(), {
        "selected_local_qenergy_damage": float(
            bank.local_qenergy_damage[policy][int(group)]
        ),
        "selected_group_pages": int(bank.selected_group_pages[policy][int(group)]),
        "injected_delta_mse": float(np.mean(delta.astype(np.float64) ** 2)),
    }


def current_token_quality_metrics(
    reference_logits: np.ndarray,
    candidate_logits: np.ndarray,
    next_token_id: int,
    reference_final_hidden: np.ndarray,
    candidate_final_hidden: np.ndarray,
) -> dict[str, float]:
    reference = np.asarray(reference_logits, np.float32).reshape(1, -1)
    candidate = np.asarray(candidate_logits, np.float32).reshape(1, -1)
    result = token_quality_metrics(
        reference, candidate, np.asarray([int(next_token_id)], np.int64),
    )
    result["final_hidden_mse"] = mean_squared_error(
        np.asarray(reference_final_hidden, np.float32),
        np.asarray(candidate_final_hidden, np.float32),
    )
    return result


def downstream_route_rows(
    reference_router: Mapping[int, np.ndarray],
    candidate_router: Mapping[int, np.ndarray],
    reference_hidden: Mapping[int, np.ndarray],
    candidate_hidden: Mapping[int, np.ndarray],
    injection_layer: int,
) -> tuple[list[dict[str, float | int]], int | None]:
    rows: list[dict[str, float | int]] = []
    first: int | None = None
    for layer in range(int(injection_layer) + 1, 40):
        route = route_boundary_metrics(
            np.asarray(reference_router[layer], np.float32).reshape(1, -1),
            np.asarray(candidate_router[layer], np.float32).reshape(1, -1),
        )
        changed = float(route["route_membership_change_fraction"]) > 0.0
        if changed and first is None:
            first = layer
        rows.append({
            "observation_layer": layer,
            "distance_from_injection": layer - int(injection_layer),
            "hidden_mse": mean_squared_error(
                np.asarray(reference_hidden[layer], np.float32),
                np.asarray(candidate_hidden[layer], np.float32),
            ),
            **route,
        })
    return rows, first


def cache_metrics_by_kind(
    reference: Any,
    candidate: Any,
    predicate: Callable[[str], bool] | None = None,
) -> dict[str, float | int | bool]:
    left = dict(cache_tensor_items(reference))
    right = dict(cache_tensor_items(candidate))
    if left.keys() != right.keys():
        raise RuntimeError("post-token cache inventories differ")
    names = [name for name in left if predicate is None or predicate(name)]
    coordinates = 0
    squared = 0.0
    maximum = 0.0
    identical = True
    for name in names:
        if tuple(left[name].shape) != tuple(right[name].shape):
            raise RuntimeError(f"post-token cache shape differs: {name}")
        identical = identical and bool(left[name].equal(right[name]))
        difference = left[name].detach().float() - right[name].detach().float()
        count = int(difference.numel())
        coordinates += count
        if count:
            maximum = max(maximum, float(difference.abs().max().item()))
            squared += float((difference.double() ** 2).sum().item())
    return {
        "tensors": len(names),
        "coordinates": coordinates,
        "bit_identical": identical,
        "max_abs": maximum,
        "mse": squared / coordinates if coordinates else 0.0,
    }


def classified_cache_metrics(reference: Any, candidate: Any) -> dict[str, float | int | bool]:
    kinds = {
        "full_cache": None,
        "attention_kv": lambda name: name.endswith(".keys") or name.endswith(".values"),
        "deltanet_conv": lambda name: ".conv_states." in name,
        "deltanet_recurrent": lambda name: ".recurrent_states." in name,
    }
    result: dict[str, float | int | bool] = {}
    for prefix, predicate in kinds.items():
        for name, value in cache_metrics_by_kind(reference, candidate, predicate).items():
            result[f"{prefix}_{name}"] = value
    return result


def add_live_minus_frozen(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "injection_layer",
        "rate_pages_per_expert",
        "group",
        "request_id",
        "position",
        "policy",
    ]
    live = frame[frame["route_mode"].eq("live")][keys + ["logit_kl"]]
    frozen = frame[frame["route_mode"].eq("fully_frozen")][keys + ["logit_kl"]]
    paired = live.merge(frozen, on=keys, suffixes=("_live", "_frozen"), validate="one_to_one")
    paired["live_minus_frozen_logit_kl"] = paired["logit_kl_live"] - paired["logit_kl_frozen"]
    return frame.merge(
        paired[keys + ["live_minus_frozen_logit_kl"]],
        on=keys,
        validate="many_to_one",
    )


def expected_grid_counts(config: Mapping[str, Any]) -> dict[str, int]:
    layers = list(map(int, config["injection_layers"]))
    rates = list(map(int, config["page_caps"]))
    groups = int(config["position_mask"]["expected_groups_per_layer"])
    policies = len(config["policies"])
    modes = len(config["route_modes"])
    quality = len(layers) * len(rates) * groups * policies * modes
    propagation = len(rates) * groups * policies * modes * sum(39 - layer for layer in layers)
    zero = len(layers) * len(rates) * groups * modes
    return {
        "quality_rows": quality,
        "propagation_rows": propagation,
        "logical_zero_dose_rows": zero,
        "physical_zero_dose_executions_if_rate_deduplicated": len(layers) * groups * modes,
    }
