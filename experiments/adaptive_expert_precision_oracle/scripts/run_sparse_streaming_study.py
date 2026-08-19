#!/usr/bin/env python3
"""Validation-first sparse activation-dependent MXFP4 suffix study.

This runner allocates the locked embedded Q2->Q3->Q4 suffix.  It never fits
trees, changes leaf codes, changes the checkpoint revision, or mixes request
splits.  Pilot mode selects bounded configurations on validation and evaluates
only those configurations (plus named H0 oracles) on held-out test records.
Full mode requires the resulting validation-promotions JSON.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(EXPERIMENT / "scripts")]

from oracle_study.interaction_allocator import (
    build_base_gram,
    diagonal_from_gram,
    exact_marginal_fixed_greedy_from_gram,
)
from oracle_study.mxfp4_embed import audit_checkpoint, load_compressed_mxfp4_expert
from oracle_study.mxfp4_selective import BitAction
from oracle_study.sparse_streaming import (
    base_gram_candidate_greedy,
    neuron_major_decomposition,
    qenergy as numpy_qenergy,
    silu as numpy_silu,
    silu_prime as numpy_silu_prime,
)
from oracle_study.sparse_streaming_cuda import (
    MixedPageGeometry,
    ProgressiveNeuronTrace,
    SelectionTrace,
    exact_mixed_page_greedy,
    exact_progressive_neuron_page_greedy,
    exact_unit_fixed_greedy,
    qenergy as torch_qenergy,
    qinner as torch_qinner,
    static_proxy_mixed_page_order,
)
from run_mxfp4_selective_pages import occurrence, tree_from_record
from run_phase_a_remote import proxy_gradients


PROJECTIONS = ("gate", "up", "down")
SHORTLIST_PROJECTIONS = ("gate", "up")
MATRIX_WEIGHTS = 2048 * 512
EXPERT_WEIGHTS = 3 * MATRIX_WEIGHTS
PAGE_BYTES = 512
PAGES_PER_EXPERT_BPW = EXPERT_WEIGHTS // (8 * PAGE_BYTES)
DIRECT_UNIT_PAYLOAD_BYTES = 1536
COORDINATE_ACTION_PAYLOAD_BYTES = 64
ACTIVATION_CONCENTRATION_TOP_K = (64, 128, 256, 384, 512, 768, 1024, 2048)

IDENTITY_COLUMNS = (
    "capture_source", "evaluation_split", "request_id", "position", "layer", "expert_id",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_parquet(
    path: Path, rows: list[dict[str, Any]], columns: Sequence[str] | None = None,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame = pd.DataFrame(rows)
    if frame.empty and columns is not None:
        frame = pd.DataFrame({name: pd.Series(dtype="object") for name in columns})
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def physical_page_budget(bpw: float) -> int:
    return int(math.floor(float(bpw) * EXPERT_WEIGHTS / (8 * PAGE_BYTES) + 1e-12))


def projection_page_budget(bpw: float) -> int:
    return int(math.floor(float(bpw) * MATRIX_WEIGHTS / (8 * PAGE_BYTES) + 1e-12))


def recovery(
    target: np.ndarray, base: np.ndarray, approximation: np.ndarray,
    proxy: np.ndarray, beta: float,
) -> float:
    denominator = numpy_qenergy(target - base, proxy=proxy, beta=beta)
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise RuntimeError(f"non-positive complete-expert recovery denominator: {denominator}")
    damage = numpy_qenergy(target - approximation, proxy=proxy, beta=beta)
    return 1.0 - damage / denominator


def projection_recovery(contributions: np.ndarray, actions: Sequence[BitAction]) -> float:
    values = np.asarray(contributions, np.float64)
    target = values.sum(axis=(0, 1))
    approximation = np.zeros_like(target)
    for action in actions:
        approximation += values[int(action.stage) - 1, int(action.coordinate)]
    denominator = float(target @ target)
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise RuntimeError(f"non-positive projection recovery denominator: {denominator}")
    residual = target - approximation
    return 1.0 - float(residual @ residual) / denominator


def realized_projection_marginals(
    contributions: np.ndarray, actions: Sequence[BitAction],
) -> tuple[BitAction, ...]:
    """Replay a static nested order against its actual changing residual."""
    values = np.asarray(contributions, np.float64)
    residual = values.sum(axis=(0, 1))
    result: list[BitAction] = []
    for action in actions:
        correction = values[int(action.stage) - 1, int(action.coordinate)]
        gain = 2.0 * float(correction @ residual) - float(correction @ correction)
        result.append(BitAction(int(action.coordinate), int(action.stage), gain))
        residual -= correction
    return tuple(result)


def common_metadata(
    source: str, split: str, data: dict[str, np.ndarray], record: int,
    router_rank: int, layer: int, expert: int, stratum: str,
) -> dict[str, Any]:
    return {
        "capture_source": source,
        "evaluation_split": split,
        "request_id": str(data["request_id"][record]),
        "sequence_id": str(data["sequence_id"][record]),
        "position": int(data["position"][record]),
        "layer": int(layer),
        "expert_id": int(expert),
        "expert_stratum": str(stratum),
        "router_rank": int(router_rank) + 1,
        "router_coefficient": float(data["router_weights"][record, router_rank]),
    }


def verify_request_separation(data: Mapping[str, np.ndarray]) -> dict[str, int]:
    observed_splits = set(map(str, np.unique(data["split"])))
    expected_splits = {"train", "validation", "test"}
    if observed_splits != expected_splits:
        raise RuntimeError(f"capture split labels changed: {observed_splits} != {expected_splits}")
    request_sets = {
        split: set(map(str, data["request_id"][data["split"] == split]))
        for split in ("train", "validation", "test")
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = request_sets[left] & request_sets[right]
        if overlap:
            raise RuntimeError(f"request split leakage between {left} and {right}: {sorted(overlap)[:3]}")
    return {split: len(values) for split, values in request_sets.items()}


def assert_unique_trace(trace: SelectionTrace, maximum_page: int) -> None:
    order = list(map(int, trace.order.tolist()))
    if len(order) != len(set(order)):
        raise RuntimeError("selector returned a duplicate physical page/action")
    if any(page < 0 or page >= int(maximum_page) for page in order):
        raise RuntimeError("selector returned a page/action outside its representation")


def assert_neuron_endpoints(decomposition: Any, exact_trace: SelectionTrace) -> None:
    units = int(decomposition.units)
    if len(exact_trace.order) != units or len(set(map(int, exact_trace.order.tolist()))) != units:
        raise RuntimeError("direct neuron oracle did not select every unit exactly once")
    endpoint = exact_trace.snapshots.get(units)
    if endpoint is None or not np.allclose(
        endpoint.numpy(), decomposition.target_output, rtol=2e-4, atol=2e-3,
    ):
        raise RuntimeError("full direct neuron packet endpoint differs from decoded Q4")
    progressive = decomposition.base_output.copy()
    depth = np.zeros(units, np.uint8)
    pages: set[int] = set()
    for stage in (1, 2):
        for unit in range(units):
            if stage != int(depth[unit]) + 1:
                raise RuntimeError("progressive neuron endpoint lost nesting")
            progressive += decomposition.refinements[stage - 1, unit]
            depth[unit] = stage
            if stage == 1:
                pages.update((3 * unit, 3 * unit + 2))
            else:
                pages.update((3 * unit + 1, 3 * unit + 2))
    if len(pages) != 3 * units or not np.all(depth == 2):
        raise RuntimeError("progressive neuron endpoint did not pay exactly three pages per unit")
    if not np.allclose(progressive, decomposition.target_output, rtol=1e-8, atol=1e-8):
        raise RuntimeError("progressive neuron endpoint differs from decoded Q4")


def assert_progressive_trace(
    decomposition: Any, trace: ProgressiveNeuronTrace,
) -> None:
    units = int(decomposition.units)
    order = np.asarray(trace.order, np.int64)
    if order.shape != (2 * units, 2):
        raise RuntimeError("progressive selector did not return every nested action")
    if len({(int(row[0]), int(row[1])) for row in order}) != 2 * units:
        raise RuntimeError("progressive selector duplicated an action")
    depth = np.zeros(units, np.uint8)
    for (coordinate, stage), cost in zip(order, np.asarray(trace.incremental_pages, np.int64)):
        if int(stage) != int(depth[int(coordinate)]) + 1:
            raise RuntimeError("progressive selector violated nested eligibility")
        expected_cost = 2 if int(stage) == 1 else 1
        if int(cost) != expected_cost:
            raise RuntimeError("progressive selector charged the wrong incremental page cost")
        depth[int(coordinate)] = int(stage)
    cumulative = np.asarray(trace.cumulative_pages, np.int64)
    if len(cumulative) != 2 * units or int(cumulative[-1]) != 3 * units:
        raise RuntimeError("progressive selector did not reach its 3*units-page endpoint")
    endpoint = trace.snapshots.get(3 * units)
    if endpoint is None or not np.allclose(
        endpoint.numpy(), decomposition.target_output, rtol=2e-4, atol=2e-3,
    ):
        raise RuntimeError("progressive selector endpoint differs from decoded Q4")


def verify_locked_inputs(
    config: dict[str, Any], config_path: Path, captures: Path, checkpoint: Path,
    trees: Path, source: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    if source not in config["locked_capture_sha256"]:
        raise RuntimeError(f"no locked capture hash for {source}")
    actual_capture = sha256(captures)
    expected_capture = str(config["locked_capture_sha256"][source])
    if actual_capture != expected_capture:
        raise RuntimeError(f"capture changed: {actual_capture} != {expected_capture}")
    actual_tree = sha256(trees)
    expected_tree = str(config["locked_tree_sha256"])
    if actual_tree != expected_tree:
        raise RuntimeError(f"selected trees changed: {actual_tree} != {expected_tree}")
    checkpoint_hashes = {
        "config_sha256": sha256(checkpoint / "config.json"),
        "index_sha256": sha256(checkpoint / "model.safetensors.index.json"),
    }
    for name, actual in checkpoint_hashes.items():
        expected = str(config["reference"][name])
        if actual != expected:
            raise RuntimeError(f"checkpoint {name} changed: {actual} != {expected}")
    audit = audit_checkpoint(checkpoint, config["layers"])
    if not audit["passed"]:
        raise RuntimeError(f"checkpoint audit failed: {audit}")
    return audit, {
        "config_file_sha256": sha256(config_path),
        "capture_sha256": actual_capture,
        "tree_sha256": actual_tree,
        **checkpoint_hashes,
    }


def canonical_coordinate_page_map(coordinates: int) -> dict[int, tuple[int, ...]]:
    """Frozen identity packet layout; each page stores both planes for four columns."""
    return {
        stage * int(coordinates) + coordinate: (coordinate // 4,)
        for stage in range(2) for coordinate in range(int(coordinates))
    }


def close_coordinate_shortlist_to_pages(
    shortlisted_actions: Iterable[int], coordinates: int,
    action_pages: Mapping[int, Sequence[int]],
) -> tuple[np.ndarray, set[int]]:
    """Expose every co-resident action after paying the frozen page union."""
    requested = np.asarray(tuple(map(int, shortlisted_actions)), np.int64)
    fetched = {
        int(page) for action in requested for page in action_pages[int(action)]
    }
    expanded = np.asarray([
        action for action in range(2 * int(coordinates))
        if any(int(page) in fetched for page in action_pages[action])
    ], np.int64)
    return expanded, fetched


def path_under_page_budget(
    order: Sequence[BitAction], coordinates: int, page_budget: int,
    action_pages: Mapping[int, Sequence[int]],
) -> tuple[list[BitAction], set[int]]:
    depth = np.zeros(int(coordinates), np.uint8)
    pages: set[int] = set()
    selected: list[BitAction] = []
    for action in order:
        coordinate = int(action.coordinate)
        stage = int(action.stage)
        if stage != int(depth[coordinate]) + 1:
            continue
        action_id = (stage - 1) * int(coordinates) + coordinate
        proposed = pages | set(map(int, action_pages[action_id]))
        if len(proposed) > int(page_budget):
            continue
        pages = proposed
        depth[coordinate] = stage
        selected.append(action)
    return selected, pages


def best_utility_prefix(actions: Sequence[BitAction]) -> tuple[BitAction, ...]:
    """Return the nested greedy prefix with maximum cumulative marginal."""
    cumulative = 0.0
    best_gain, best_count = 0.0, 0
    for count, action in enumerate(actions, start=1):
        cumulative += float(action.score)
        if cumulative > best_gain:
            best_gain, best_count = cumulative, count
    return tuple(actions[:best_count])


def residual_consistent_exact_under_page_budget(
    gram: np.ndarray, activation: np.ndarray, coordinates: int, page_budget: int,
    action_pages: Mapping[int, Sequence[int]],
) -> tuple[list[BitAction], set[int]]:
    """Rerun exact fixed-greedy with the physical budget active.

    Filtering an unrestricted exact order after it was scored is not
    residual-consistent: every skipped action changes the residual used to
    score all later actions.  This helper independently runs the base-Gram
    selector with unaffordable new-page actions masked, then retains the
    best cumulative-utility prefix.  The returned support is therefore a
    valid exact fixed-coefficient greedy control at the requested physical
    budget; it is still an action-greedy control, not an exhaustive page-mask
    oracle.
    """
    all_actions = np.arange(2 * int(coordinates), dtype=np.int64)
    trace = base_gram_candidate_greedy(
        gram, activation, all_actions,
        action_pages=action_pages, page_budget=max(int(page_budget), 0),
    )
    selected = list(best_utility_prefix(trace.actions))
    selected_again, pages = path_under_page_budget(
        selected, coordinates, page_budget, action_pages,
    )
    if selected_again != selected or len(pages) > int(page_budget):
        raise RuntimeError("page-constrained exact selector produced an invalid physical prefix")
    return selected, pages


def action_ids(actions: Iterable[BitAction], coordinates: int) -> list[int]:
    return [(int(action.stage) - 1) * int(coordinates) + int(action.coordinate) for action in actions]


def static_unit_output(decomposition: Any, order: Sequence[int], count: int) -> np.ndarray:
    corrections = decomposition.levels[2] - decomposition.levels[0]
    chosen = np.asarray(order[: max(int(count), 0)], np.int64)
    if not len(chosen):
        return decomposition.base_output
    return decomposition.base_output + corrections[chosen].sum(axis=0)


def weighted_jaccard(
    left_ids: np.ndarray, left_scores: np.ndarray,
    right_ids: np.ndarray, right_scores: np.ndarray,
) -> float:
    left = {int(action): max(float(left_scores[int(action)]), 0.0) for action in left_ids}
    right = {int(action): max(float(right_scores[int(action)]), 0.0) for action in right_ids}
    union = set(left) | set(right)
    numerator = sum(min(left.get(action, 0.0), right.get(action, 0.0)) for action in union)
    denominator = sum(max(left.get(action, 0.0), right.get(action, 0.0)) for action in union)
    return numerator / max(denominator, 1e-30)


def storage_multiplier(config: dict[str, Any]) -> float:
    return (
        float(config["reference_bpw"]) + float(config["suffix_bpw_per_complete_representation"])
    ) / float(config["reference_bpw"])


def storage_fields(
    config: dict[str, Any], *, metadata_bytes: int = 0,
) -> dict[str, float | int]:
    suffix = float(config["suffix_bpw_per_complete_representation"])
    metadata_bpw = 8.0 * int(metadata_bytes) / EXPERT_WEIGHTS
    return {
        "suffix_storage_bpw": suffix,
        "selector_metadata_bpw": metadata_bpw,
        "selector_metadata_bytes_per_expert": int(metadata_bytes),
        "storage_multiplier": (float(config["reference_bpw"]) + suffix + metadata_bpw)
        / float(config["reference_bpw"]),
    }


def row_qenergies(rows: np.ndarray, proxy: np.ndarray, beta: float) -> np.ndarray:
    values = np.asarray(rows, np.float64)
    result = np.einsum("ij,ij->i", values, values, optimize=True)
    projected = values @ np.asarray(proxy, np.float64)
    return result + float(beta) * np.einsum("ij,ij->i", projected, projected, optimize=True)


def descending_order(score: np.ndarray) -> np.ndarray:
    values = np.asarray(score, np.float64).reshape(-1)
    ids = np.arange(len(values), dtype=np.int64)
    return ids[np.lexsort((ids, -values))]


def coordinate_scores(
    activation: np.ndarray, deltas: np.ndarray, method: str,
    *, future_metric: np.ndarray | None = None,
) -> np.ndarray:
    """One deterministic score per input coordinate, aggregating both planes."""
    x = np.asarray(activation, np.float64).reshape(-1)
    values = np.asarray(deltas, np.float64)
    if values.shape[:2] != (2, x.size):
        raise ValueError("deltas must be [2, input_coordinate, projection_output]")
    if method == "abs_activation":
        return np.abs(x)
    metric = None if method == "activation_weighted_delta_norm" else future_metric
    if method not in {"activation_weighted_delta_norm", "future_proxy_weighted"}:
        raise ValueError(f"unknown coordinate score method: {method}")
    if method == "future_proxy_weighted" and metric is None:
        raise ValueError("future-proxy score requires a projection-output metric")
    if metric is None:
        stage_energy = np.einsum("sjo,sjo->sj", values, values, optimize=True)
    else:
        stage_energy = np.einsum("sjo,op,sjp->sj", values, metric, values, optimize=True)
    return x * x * stage_energy.sum(axis=0)


def coordinate_shortlist(score: np.ndarray, count: int, coordinates: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(score, np.float64).reshape(-1)
    if values.size != int(coordinates):
        raise ValueError("coordinate score width changed")
    ids = np.arange(values.size, dtype=np.int64)
    chosen = ids[np.lexsort((ids, -values))[: min(max(int(count), 0), len(ids))]]
    actions = np.concatenate((chosen, chosen + int(coordinates)))
    return chosen, actions


def activation_concentration_rows(
    activation: np.ndarray, metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Regenerate heavy-tail diagnostics from each locked activation."""
    values = np.asarray(activation, np.float64).reshape(-1)
    energy = values * values
    total = float(energy.sum())
    ordered = np.sort(energy)[::-1]
    coordinates_for_90pct = int(np.searchsorted(np.cumsum(ordered), 0.9 * total) + 1) if total > 0 else 0
    exact_zero_fraction = float(np.mean(values == 0.0))
    return [
        {**metadata, "top_k": min(int(top_k), len(values)), "top_k_energy_fraction": float(ordered[:top_k].sum() / total) if total > 0 else 0.0, "exact_zero_fraction": exact_zero_fraction, "coordinates_for_90pct_energy": coordinates_for_90pct}
        for top_k in ACTIVATION_CONCENTRATION_TOP_K
    ]


def input_coordinate_baseline_trace(
    gate2: np.ndarray, up2: np.ndarray, down2: np.ndarray,
    gate4: np.ndarray, up4: np.ndarray, down4: np.ndarray,
    activation: np.ndarray, page_budgets: Iterable[int], proxy: np.ndarray,
    beta: float, device: torch.device,
) -> SelectionTrace:
    """Static first-order direct-Q4 input packets plus exact down-column pages.

    A gate/up page stores both refinement planes for two input coordinates
    (2 matrices * 512 rows * 2 coordinates * 2 bits = 512 bytes).  Down uses
    one complete two-plane hidden-unit column per page.  This is a controlled
    input-coordinate baseline, not the PR6 hybrid layout.
    """
    g2 = torch.as_tensor(gate2, dtype=torch.float32, device=device)
    u2 = torch.as_tensor(up2, dtype=torch.float32, device=device)
    d2 = torch.as_tensor(down2, dtype=torch.float32, device=device).clone()
    g4 = torch.as_tensor(gate4, dtype=torch.float32, device=device)
    u4 = torch.as_tensor(up4, dtype=torch.float32, device=device)
    d4 = torch.as_tensor(down4, dtype=torch.float32, device=device)
    x = torch.as_tensor(activation, dtype=torch.float32, device=device)
    p = torch.as_tensor(proxy, dtype=torch.float32, device=device)
    hidden_size, input_size = map(int, g2.shape)
    if input_size % 2:
        raise ValueError("input-coordinate packet baseline requires an even input width")
    gate_up_pages = input_size // 2
    total_pages = gate_up_pages + hidden_size
    requested = tuple(sorted(set(map(int, page_budgets))))
    if any(value < 0 or value > total_pages for value in requested):
        raise ValueError("input-coordinate page budget exceeds the representation endpoint")
    maximum = max(requested, default=0)

    gate = g2 @ x
    up = u2 @ x
    hidden = torch.nn.functional.silu(gate) * up
    output = d2 @ hidden
    target_gate = g4 @ x
    target_up = u4 @ x
    target = d4 @ (torch.nn.functional.silu(target_gate) * target_up)

    delta_gate = (g4 - g2).reshape(hidden_size, gate_up_pages, 2)
    delta_up = (u4 - u2).reshape(hidden_size, gate_up_pages, 2)
    x_pages = x.reshape(gate_up_pages, 2)
    gate_effect = torch.einsum("hpc,pc->ph", delta_gate, x_pages)
    up_effect = torch.einsum("hpc,pc->ph", delta_up, x_pages)
    sigmoid = torch.sigmoid(gate)
    silu_prime = sigmoid * (1.0 + gate * (1.0 - sigmoid))
    hidden_effect = (up * silu_prime)[None, :] * gate_effect
    hidden_effect += torch.nn.functional.silu(gate)[None, :] * up_effect
    gate_up_correction = hidden_effect @ d2.T
    down_correction = ((d4 - d2) * hidden[None, :]).T.contiguous()
    correction = torch.cat((gate_up_correction, down_correction), dim=0)
    residual = target - output
    score = 2.0 * torch_qinner(correction, residual, proxy=p, beta=beta)
    score -= torch_qenergy(correction, proxy=p, beta=beta)
    ranking = torch.argsort(score, descending=True, stable=True)[:maximum]

    snapshots: dict[int, torch.Tensor] = {}
    if 0 in requested:
        snapshots[0] = output.detach().cpu().float().clone()
    order: list[int] = []
    gains: list[float] = []
    for page_tensor in ranking:
        page = int(page_tensor.item())
        old_residual = target - output
        if page < gate_up_pages:
            old_hidden = hidden.clone()
            gate.add_(gate_effect[page])
            up.add_(up_effect[page])
            hidden = torch.nn.functional.silu(gate) * up
            output.add_(d2 @ (hidden - old_hidden))
        else:
            unit = page - gate_up_pages
            delta_column = d4[:, unit] - d2[:, unit]
            output.add_(delta_column * hidden[unit])
            d2[:, unit] = d4[:, unit]
        new_residual = target - output
        gain = torch_qenergy(old_residual, proxy=p, beta=beta)
        gain -= torch_qenergy(new_residual, proxy=p, beta=beta)
        order.append(page)
        gains.append(float(gain.item()))
        if len(order) in requested:
            snapshots[len(order)] = output.detach().cpu().float().clone()
    return SelectionTrace(
        order=torch.tensor(order, dtype=torch.int64),
        gains=torch.tensor(gains, dtype=torch.float64),
        snapshots=snapshots,
    )


def standard_frontier_fields(
    *, physical_budget_bpw: float, pages: int, logical_actions: int,
    recovery_value: float, runtime_seconds: float, selector_bytes_read: int,
    multiplier: float, logical_payload_bytes: int | None = None,
) -> dict[str, Any]:
    physical_bytes = int(pages) * PAGE_BYTES
    logical_bytes = physical_bytes if logical_payload_bytes is None else int(logical_payload_bytes)
    return {
        "physical_budget_bpw": float(physical_budget_bpw),
        "physical_pages": int(pages),
        "physical_bytes": physical_bytes,
        "physical_bpw": 8.0 * physical_bytes / EXPERT_WEIGHTS,
        "logical_actions": int(logical_actions),
        "logical_bytes": logical_bytes,
        "logical_bpw": 8.0 * logical_bytes / EXPERT_WEIGHTS,
        "recovery": float(recovery_value),
        "selector_runtime_ms": 1000.0 * float(runtime_seconds),
        "selector_bytes_read": int(selector_bytes_read),
        "selector_bytes_read_semantics": "unique_tensor_footprint_lower_bound_not_hardware_traffic",
        "storage_multiplier": float(multiplier),
        "page_amplification": physical_bytes / max(logical_bytes, 1),
    }


UNIT_PROXY_NAMES = {
    "independent_unit_correction_norm": "independent_unit_correction_norm",
    "q2_hidden_magnitude": "q2_intermediate_magnitude",
    "weight_aware_down_delta": "down_weight_aware",
    "first_order_gate_up_down_sensitivity": "first_order_sensitivity",
}


UNIT_SELECTION_REGIMES = {
    "exact_unit_output_marginal_fixed_greedy": "h0_oracle_full_suffix",
    "independent_unit_correction_norm": "h0_oracle_full_suffix",
    "q2_hidden_magnitude": "h0_resident_proxy_late",
    "weight_aware_down_delta": "h0_deployable_metadata_late",
    "first_order_gate_up_down_sensitivity": "h0_oracle_full_suffix",
}


def evaluate_neuron_frontier(
    matrices: dict[str, list[np.ndarray]], x: np.ndarray, proxy: np.ndarray, beta: float,
    metadata: dict[str, Any], config: dict[str, Any], device: torch.device,
    selected_proxy: Sequence[str] | None, down_norm_metadata: np.ndarray,
) -> list[dict[str, Any]]:
    levels = [tuple(matrices[p][level] for p in PROJECTIONS) for level in range(3)]
    decomposition = neuron_major_decomposition(levels, x)
    counts = sorted(set(int(value) for value in config["unit_counts"] if int(value) <= decomposition.units))
    trace_counts = sorted(set(counts) | {decomposition.units})
    corrections = torch.as_tensor(
        decomposition.levels[2] - decomposition.levels[0], dtype=torch.float32, device=device,
    )
    synchronize(device)
    started = time.perf_counter()
    exact_trace = exact_unit_fixed_greedy(
        corrections, budgets=trace_counts, base_output=decomposition.base_output,
        proxy=proxy, beta=beta, device=device,
    )
    synchronize(device)
    exact_seconds = time.perf_counter() - started
    assert_neuron_endpoints(decomposition, exact_trace)
    exact_bytes = corrections.numel() * corrections.element_size() + decomposition.units ** 2 * 4
    target = decomposition.target_output
    base = decomposition.base_output
    base_storage = storage_fields(config)
    rows: list[dict[str, Any]] = []
    for count in counts:
        output = exact_trace.snapshots[count].numpy()
        rows.append({
            **metadata,
            "objective": "exact_sequential_complete_expert_qenergy",
            "selection_regime": UNIT_SELECTION_REGIMES["exact_unit_output_marginal_fixed_greedy"],
            "action_family": "neuron_major",
            "representation": "direct_q2_q4_complete_unit_packet_3pages",
            "selector": "exact_unit_output_marginal_fixed_greedy",
            "selection_regime_category": "h0_oracle",
            "second_pass_required": False,
            **base_storage,
            **standard_frontier_fields(
                physical_budget_bpw=3 * count / PAGES_PER_EXPERT_BPW,
                pages=3 * count, logical_actions=count,
                recovery_value=recovery(target, base, output, proxy, beta),
                runtime_seconds=exact_seconds, selector_bytes_read=exact_bytes,
                multiplier=float(base_storage["storage_multiplier"]),
                logical_payload_bytes=count * DIRECT_UNIT_PAYLOAD_BYTES,
            ),
        })

    progressive_budgets = sorted(
        {physical_page_budget(value) for value in map(float, config["physical_budgets_bpw"])}
        | {3 * decomposition.units}
    )
    synchronize(device)
    progressive_started = time.perf_counter()
    progressive_trace = exact_progressive_neuron_page_greedy(
        torch.as_tensor(decomposition.refinements, dtype=torch.float32, device=device),
        page_budgets=progressive_budgets, base_output=decomposition.base_output,
        proxy=proxy, beta=beta, device=device,
    )
    synchronize(device)
    progressive_seconds = time.perf_counter() - progressive_started
    assert_progressive_trace(decomposition, progressive_trace)
    cumulative = np.asarray(progressive_trace.cumulative_pages, np.int64)
    for budget_bpw in map(float, config["physical_budgets_bpw"]):
        budget_pages = physical_page_budget(budget_bpw)
        logical_actions = int(np.sum(cumulative <= budget_pages))
        paid_pages = int(cumulative[logical_actions - 1]) if logical_actions else 0
        if paid_pages > budget_pages:
            raise RuntimeError("progressive neuron selector exceeded its physical page budget")
        rows.append({
            **metadata,
            "objective": "exact_sequential_complete_expert_qenergy",
            "selection_regime": "h0_oracle_full_suffix",
            "selection_regime_category": "h0_oracle",
            "action_family": "neuron_major",
            "representation": "progressive_q3_q4_coherent_unit_packet_2plus1_pages",
            "selector": "exact_progressive_neuron_page_greedy",
            "second_pass_required": False,
            **base_storage,
            **standard_frontier_fields(
                physical_budget_bpw=budget_bpw, pages=paid_pages,
                logical_actions=logical_actions,
                recovery_value=recovery(
                    target, base, progressive_trace.snapshots[budget_pages].numpy(), proxy, beta,
                ),
                runtime_seconds=progressive_seconds,
                selector_bytes_read=decomposition.refinements.nbytes + (2 * decomposition.units) ** 2 * 4,
                multiplier=float(base_storage["storage_multiplier"]),
                logical_payload_bytes=logical_actions * 768,
            ),
        })
    gate2, up2, down2 = levels[0]
    gate4, up4, down4 = levels[2]
    g2 = np.asarray(gate2 @ x, np.float64)
    u2 = np.asarray(up2 @ x, np.float64)
    h2 = numpy_silu(g2) * u2
    selectors = list(UNIT_PROXY_NAMES)
    if selected_proxy is not None:
        selectors = list(selected_proxy)
    for selector in selectors:
        proxy_started = time.perf_counter()
        metadata_bytes = 0
        if selector == "q2_hidden_magnitude":
            score = np.abs(h2)
            bytes_read = h2.nbytes
        elif selector == "weight_aware_down_delta":
            # The FP16 per-unit norm is persistent selector metadata; its
            # offline construction is not charged as H0 selector runtime.
            metadata_bytes = 2 * decomposition.units + 4
            score = h2 * h2 * np.asarray(down_norm_metadata, np.float64)
            bytes_read = h2.nbytes + metadata_bytes
        elif selector == "independent_unit_correction_norm":
            score = row_qenergies(decomposition.levels[2] - decomposition.levels[0], proxy, beta)
            bytes_read = (decomposition.levels[2] - decomposition.levels[0]).nbytes
        elif selector == "first_order_gate_up_down_sensitivity":
            delta_gate = (gate4 - gate2) @ x
            delta_up = (up4 - up2) @ x
            delta_hidden = u2 * numpy_silu_prime(g2) * delta_gate + numpy_silu(g2) * delta_up
            vectors = down2.T * delta_hidden[:, None] + (down4 - down2).T * h2[:, None]
            score = row_qenergies(vectors, proxy, beta)
            bytes_read = sum(value.nbytes for value in (gate2, up2, down2, gate4, up4, down4))
        else:
            raise ValueError(f"unknown unit selector: {selector}")
        ranking = descending_order(score)
        proxy_seconds = time.perf_counter() - proxy_started
        selector_storage = storage_fields(config, metadata_bytes=metadata_bytes)
        for count in counts:
            output = static_unit_output(decomposition, ranking, count)
            rows.append({
                **metadata,
                "objective": "exact_sequential_complete_expert_qenergy",
                "selection_regime": UNIT_SELECTION_REGIMES[selector],
                "selection_regime_category": (
                    "deployable_h0_proxy" if selector == "weight_aware_down_delta"
                    else "h0_proxy" if selector == "q2_hidden_magnitude" else "h0_oracle"
                ),
                "action_family": "neuron_major",
                "representation": "direct_q2_q4_complete_unit_packet_3pages",
                "selector": selector,
                "second_pass_required": selector in {"q2_hidden_magnitude", "weight_aware_down_delta"},
                **selector_storage,
                **standard_frontier_fields(
                    physical_budget_bpw=3 * count / PAGES_PER_EXPERT_BPW,
                    pages=3 * count, logical_actions=count,
                    recovery_value=recovery(target, base, output, proxy, beta),
                    runtime_seconds=proxy_seconds,
                    selector_bytes_read=bytes_read,
                    multiplier=float(selector_storage["storage_multiplier"]),
                    logical_payload_bytes=count * DIRECT_UNIT_PAYLOAD_BYTES,
                ),
            })
    return rows


def future_projection_metric(
    projection: str, matrices: dict[str, list[np.ndarray]], x: np.ndarray,
    proxy: np.ndarray, beta: float,
) -> np.ndarray:
    """First-order complete-output qenergy metric for gate/up activation deltas."""
    gate = np.asarray(matrices["gate"][2] @ x, np.float64)
    up = np.asarray(matrices["up"][2] @ x, np.float64)
    factor = up * numpy_silu_prime(gate) if projection == "gate" else numpy_silu(gate)
    down = np.asarray(matrices["down"][2], np.float64)
    hidden_metric = down.T @ down
    projected = down.T @ np.asarray(proxy, np.float64)
    hidden_metric += float(beta) * (projected @ projected.T)
    return factor[:, None] * hidden_metric * factor[None, :]


def inverse_rank(values: np.ndarray) -> np.ndarray:
    score = np.asarray(values, np.float64).reshape(-1)
    ids = np.arange(len(score), dtype=np.int64)
    order = ids[np.lexsort((ids, -score))]
    result = np.empty(len(score), np.int64)
    result[order] = np.arange(1, len(score) + 1)
    return result


def evaluate_shortlists(
    matrices: dict[str, list[np.ndarray]], deltas: dict[str, np.ndarray], grams: dict[str, np.ndarray],
    x: np.ndarray, proxy: np.ndarray, beta: float, metadata: dict[str, Any],
    config: dict[str, Any], selected: Mapping[str, Any] | None,
    exact_labels: list[dict[str, Any]], support_cache: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selector_storage = storage_fields(config)
    methods = ("abs_activation", "activation_weighted_delta_norm", "future_proxy_weighted")
    coordinates = int(x.size)
    action_pages = canonical_coordinate_page_map(coordinates)
    projection_subset = SHORTLIST_PROJECTIONS if selected is None else tuple(
        projection for projection in SHORTLIST_PROJECTIONS if projection in selected
    )
    future_started = time.perf_counter()
    # This is a shortlist proxy, not an exact recovery calculation. Keep the
    # shared 512x512 construction in native float32 so the bounded pilot does
    # not pay two large FP64 BLAS products per invocation.
    future_gate = np.asarray(matrices["gate"][2] @ x, np.float32)
    future_up = np.asarray(matrices["up"][2] @ x, np.float32)
    future_down = np.asarray(matrices["down"][2], np.float32)
    future_hidden_metric = future_down.T @ future_down
    projected = future_down.T @ np.asarray(proxy, np.float32)
    future_hidden_metric += float(beta) * (projected @ projected.T)
    future_factors = {
        "gate": future_up * numpy_silu_prime(future_gate),
        "up": numpy_silu(future_gate),
    }
    future_metric_seconds = time.perf_counter() - future_started
    for projection in projection_subset:
        gram = grams[projection]
        contributions = np.asarray(deltas[projection], np.float64) * np.asarray(x, np.float64)[None, :, None]
        exact_started = time.perf_counter()
        exact = exact_marginal_fixed_greedy_from_gram(gram, x)
        diagonal = diagonal_from_gram(gram, x)
        exact_seconds = time.perf_counter() - exact_started
        if len(exact) != 2 * coordinates or len(set(action_ids(exact, coordinates))) != 2 * coordinates:
            raise RuntimeError("full projection exact path is incomplete or duplicated")
        exact_physical_cache: dict[int, tuple[list[BitAction], set[int]]] = {}

        def exact_at_pages(page_budget: int) -> tuple[list[BitAction], set[int]]:
            budget = max(int(page_budget), 0)
            if budget not in exact_physical_cache:
                exact_physical_cache[budget] = residual_consistent_exact_under_page_budget(
                    gram, x, coordinates, budget, action_pages,
                )
            return exact_physical_cache[budget]

        factor = future_factors[projection]
        future_metric = factor[:, None] * future_hidden_metric * factor[None, :]
        score_by_method: dict[str, np.ndarray] = {}
        score_seconds: dict[str, float] = {}
        for method in methods:
            score_started = time.perf_counter()
            score_by_method[method] = coordinate_scores(
                x, deltas[projection], method,
                future_metric=future_metric if method == "future_proxy_weighted" else None,
            )
            score_seconds[method] = time.perf_counter() - score_started
        score_seconds["future_proxy_weighted"] += future_metric_seconds / max(len(projection_subset), 1)
        score_bytes = {
            "abs_activation": x.nbytes,
            "activation_weighted_delta_norm": x.nbytes + deltas[projection].nbytes,
            "future_proxy_weighted": x.nbytes + deltas[projection].nbytes + future_metric.nbytes,
        }
        rank_by_method = {method: inverse_rank(score) for method, score in score_by_method.items()}
        for rank, action in enumerate(exact, start=1):
            coordinate = int(action.coordinate)
            exact_labels.append({
                **metadata,
                "projection": projection,
                "action_rank": rank,
                "action_id": (int(action.stage) - 1) * coordinates + coordinate,
                "coordinate": coordinate,
                "refinement_stage": int(action.stage),
                "refinement_plane": "q2_q3" if int(action.stage) == 1 else "q3_q4",
                "marginal_gain": float(action.score),
                "physical_page_id": int(action_pages[coordinate][0]),
                "layout_method": "canonical_locked_coordinate_packets_identity",
                **{f"coordinate_rank_{method}": int(values[coordinate]) for method, values in rank_by_method.items()},
            })

        # Exact-support control at the frozen 1.0 projection-bpw prefix.  The
        # common page-budget surrogate lets gate/up join even when their
        # logical action counts differ.
        exact_support_budget_bpw = 1.0
        exact_support_page_budget = projection_page_budget(exact_support_budget_bpw)
        exact_support, _ = exact_at_pages(exact_support_page_budget)
        exact_support_scores = np.zeros(2 * coordinates, np.float64)
        for action in exact_support:
            action_id = (int(action.stage) - 1) * coordinates + int(action.coordinate)
            exact_support_scores[action_id] = max(float(action.score), 0.0)
        support_cache.append({
            **metadata,
            "projection": projection,
            "score_method": "exact_marginal_fixed_greedy",
            "shortlist_size": exact_support_page_budget,
            "support_action_count": len(exact_support),
            "support_budget_bpw": exact_support_budget_bpw,
            "action_ids": np.asarray(action_ids(exact_support, coordinates), np.int64),
            "action_scores": exact_support_scores,
        })

        shortlist_sizes = list(map(int, config["activation_shortlist_sizes"]))
        method_subset = methods
        if selected is not None:
            choice = selected[projection]
            method_subset = (str(choice["score_method"]),)
            shortlist_sizes = [int(choice["shortlist_size"])]
        for method in method_subset:
            score = score_by_method[method]
            for shortlist_size in shortlist_sizes:
                shortlisted_coordinates, shortlisted_actions = coordinate_shortlist(score, shortlist_size, coordinates)
                candidate_actions, candidate_pages = close_coordinate_shortlist_to_pages(
                    shortlisted_actions, coordinates, action_pages,
                )
                candidate_started = time.perf_counter()
                candidate_path = base_gram_candidate_greedy(
                    gram, x, candidate_actions, action_pages=action_pages,
                )
                candidate_seconds = time.perf_counter() - candidate_started
                best_candidate_actions = best_utility_prefix(candidate_path.actions)
                support_cache.append({
                    **metadata,
                    "projection": projection,
                    "score_method": method,
                    "shortlist_size": int(shortlist_size),
                    "support_action_count": int(len(shortlisted_actions)),
                    "support_budget_bpw": np.nan,
                    "action_ids": shortlisted_actions.copy(),
                    "action_scores": np.concatenate((score, score)),
                })
                for prefix_bpw in map(float, config["support_prefix_bpw"]):
                    budget_pages = projection_page_budget(prefix_bpw)
                    exact_selected, exact_pages = exact_at_pages(budget_pages)
                    diagonal_feasible, _ = path_under_page_budget(
                        diagonal, coordinates, budget_pages, action_pages,
                    )
                    diagonal_selected = list(best_utility_prefix(
                        realized_projection_marginals(contributions, diagonal_feasible),
                    ))
                    if len(exact_pages) > budget_pages:
                        raise RuntimeError("exact projection selector exceeded its physical page budget")
                    constrained, constrained_pages = path_under_page_budget(
                        best_candidate_actions, coordinates, budget_pages, action_pages,
                    )
                    if len(constrained_pages) > budget_pages:
                        raise RuntimeError("constrained shortlist selector exceeded its physical page budget")
                    exact_recovery = projection_recovery(contributions, exact_selected)
                    diagonal_recovery = projection_recovery(contributions, diagonal_selected)
                    constrained_recovery = projection_recovery(contributions, constrained)
                    exact_matched, _ = exact_at_pages(len(candidate_pages))
                    diagonal_matched_feasible, _ = path_under_page_budget(
                        diagonal, coordinates, len(candidate_pages), action_pages,
                    )
                    diagonal_matched = list(best_utility_prefix(
                        realized_projection_marginals(contributions, diagonal_matched_feasible),
                    ))
                    constrained_matched, constrained_matched_pages = path_under_page_budget(
                        best_candidate_actions, coordinates, len(candidate_pages), action_pages,
                    )
                    exact_matched_recovery = projection_recovery(contributions, exact_matched)
                    diagonal_matched_recovery = projection_recovery(contributions, diagonal_matched)
                    constrained_matched_recovery = projection_recovery(contributions, constrained_matched)
                    strict_retention = (
                        (constrained_matched_recovery - diagonal_matched_recovery)
                        / (exact_matched_recovery - diagonal_matched_recovery)
                        if exact_matched_recovery > diagonal_matched_recovery + 1e-12 else 0.0
                    )
                    raw_candidate_set = set(map(int, shortlisted_actions))
                    candidate_set = set(map(int, candidate_actions))
                    exact_ids = action_ids(exact_selected, coordinates)
                    matched = [action for action in exact_ids if action in candidate_set]
                    weights = np.asarray(
                        [max(float(action.score), 0.0) for action in exact_selected], np.float64,
                    )
                    if not np.any(weights > 0):
                        weights = np.ones(len(exact_selected), np.float64)
                    exact_action_recall = len(matched) / max(len(exact_ids), 1)
                    weighted_recall = sum(
                        weights[index] for index, action in enumerate(exact_ids) if action in candidate_set
                    ) / max(float(weights.sum()), 1e-30)
                    raw_exact_action_recall = sum(action in raw_candidate_set for action in exact_ids) / max(len(exact_ids), 1)
                    raw_weighted_recall = sum(weights[index] for index, action in enumerate(exact_ids) if action in raw_candidate_set) / max(float(weights.sum()), 1e-30)
                    gain_retained = constrained_recovery / exact_recovery if exact_recovery > 0 else 0.0
                    relative_to_diagonal = (
                        (constrained_recovery - diagonal_recovery) / (exact_recovery - diagonal_recovery)
                        if exact_recovery > diagonal_recovery + 1e-12 else 0.0
                    )
                    logical_bytes = len(constrained_matched) * COORDINATE_ACTION_PAYLOAD_BYTES
                    standard = standard_frontier_fields(
                        physical_budget_bpw=prefix_bpw, pages=len(candidate_pages),
                        logical_actions=len(constrained_matched), recovery_value=constrained_matched_recovery,
                        runtime_seconds=score_seconds[method] + candidate_seconds,
                        selector_bytes_read=gram.nbytes + score_bytes[method],
                        multiplier=float(selector_storage["storage_multiplier"]),
                        logical_payload_bytes=logical_bytes,
                    )
                    standard["physical_bpw"] = 8.0 * standard["physical_bytes"] / MATRIX_WEIGHTS
                    standard["logical_bpw"] = 8.0 * logical_bytes / MATRIX_WEIGHTS
                    rows.append({
                        **metadata,
                        "objective": "isolated_projection_qenergy",
                        "selection_regime": "h0_oracle_full_target_residual",
                        "selection_regime_category": "h0_oracle",
                        "projection": projection,
                        "score_method": method,
                        "shortlist_size": int(shortlist_size),
                        "shortlist_unit": "input_coordinates",
                        "coordinate_score_aggregation": "magnitude" if method == "abs_activation" else "sum_stage_energy",
                        "candidate_actions": int(len(candidate_actions)),
                        "shortlisted_coordinate_count": int(len(shortlisted_coordinates)),
                        "shortlisted_actions_after_nested_closure": int(len(shortlisted_actions)),
                        "page_closed_candidate_actions": int(len(candidate_actions)),
                        "best_utility_prefix_actions": int(len(best_candidate_actions)),
                        "planned_prefix_logical_actions": int(len(constrained)),
                        "planned_prefix_logical_bytes": int(len(constrained) * COORDINATE_ACTION_PAYLOAD_BYTES),
                        "refresh_block": 1,
                        "candidate_overfetch_factor": len(candidate_pages) / max(len(exact_pages), 1),
                        "exact_gain_retained": float(gain_retained),
                        "fraction_exact_over_diagonal_gain_retained": float(relative_to_diagonal),
                        "exact_action_recall": raw_exact_action_recall,
                        "importance_weighted_recall": raw_weighted_recall,
                        "page_closed_exact_action_recall": exact_action_recall,
                        "page_closed_importance_weighted_recall": weighted_recall,
                        "candidate_pages": len(candidate_pages),
                        "fetched_pages": len(candidate_pages),
                        "applied_pages": len(constrained_matched_pages),
                        "planned_prefix_applied_pages": len(constrained_pages),
                        "recovery_diagonal": diagonal_recovery,
                        "recovery_exact": exact_recovery,
                        "recovery_constrained": constrained_recovery,
                        "recovery_exact_matched_fetched_pages": exact_matched_recovery,
                        "recovery_diagonal_matched_fetched_pages": diagonal_matched_recovery,
                        "recovery_constrained_matched_fetched_pages": constrained_matched_recovery,
                        "fraction_exact_over_diagonal_gain_retained_matched_fetched_pages": strict_retention,
                        "layout_method": "canonical_locked_coordinate_packets_identity",
                        "budget_scope": "isolated_projection",
                        "prefetch_bytes_charged": len(candidate_pages) * PAGE_BYTES,
                        **selector_storage,
                        **standard,
                    })
    return rows


def decoded_complete_outputs(
    matrices: dict[str, list[np.ndarray]], x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    outputs = []
    for level in (0, 2):
        gate = matrices["gate"][level] @ x
        up = matrices["up"][level] @ x
        outputs.append(matrices["down"][level] @ (numpy_silu(gate) * up))
    return np.asarray(outputs[0], np.float64), np.asarray(outputs[1], np.float64)


def assert_complete_endpoint(
    trace: SelectionTrace, total_pages: int, target: np.ndarray, label: str,
) -> None:
    assert_unique_trace(trace, total_pages)
    endpoint = trace.snapshots.get(total_pages)
    if endpoint is None or len(trace.order) != total_pages:
        raise RuntimeError(f"{label} did not visit every physical page")
    if not np.allclose(endpoint.numpy(), target, rtol=2e-4, atol=2e-3):
        raise RuntimeError(f"{label} endpoint differs from decoded Q4")


def tile_specs(
    config: dict[str, Any], selected: Mapping[str, Any] | None,
) -> list[tuple[str, tuple[int, int], int]]:
    if selected is not None:
        chosen_shape = tuple(map(int, selected["tile_shape"]))
        chosen_selector = str(selected["selector"])
        chosen_refresh = int(selected.get("refresh_interval", 1))
        values = [
            ("exact_dynamic_tile_marginal", (32, 32), 1),
            (chosen_selector, chosen_shape, chosen_refresh),
        ]
        return list(dict.fromkeys(values))
    shape = tuple(map(int, config["tile_pilot_shape"]))
    return [
        ("exact_dynamic_tile_marginal", shape, 1),
        ("static_first_order", shape, 0),
        ("static_wina", shape, 0),
    ]


def evaluate_tile_frontier(
    matrices: dict[str, list[np.ndarray]], x: np.ndarray, proxy: np.ndarray, beta: float,
    metadata: dict[str, Any], config: dict[str, Any], device: torch.device,
    selected: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    budgets_bpw = list(map(float, config["physical_budgets_bpw"]))
    page_budgets = [physical_page_budget(value) for value in budgets_bpw]
    base, target = decoded_complete_outputs(matrices, x)
    base_storage = storage_fields(config)
    rows: list[dict[str, Any]] = []
    arrays = [matrices[p][level] for p in PROJECTIONS for level in (0, 2)]
    selector_input_bytes = int(sum(value.nbytes for value in arrays) + x.nbytes)
    checked_shapes: set[tuple[int, int]] = set()

    for selector, shape, refresh in tile_specs(config, selected):
        geometry = MixedPageGeometry(512, 2048, *shape)
        total_pages = geometry.total_pages
        if shape not in checked_shapes:
            # A static full traversal is enough to prove representation and
            # page-ID completeness; the endpoint is order independent.
            endpoint_trace = static_proxy_mixed_page_order(
                matrices["gate"][0], matrices["up"][0], matrices["down"][0],
                matrices["gate"][2], matrices["up"][2], matrices["down"][2], x,
                tile_shape=shape, page_budgets=[total_pages], proxy_method="first_order",
                proxy=proxy, beta=beta, device=device,
            )
            assert_complete_endpoint(endpoint_trace, total_pages, target, f"tile {shape}")
            checked_shapes.add(shape)
        requested = sorted(set(page_budgets))
        synchronize(device)
        started = time.perf_counter()
        if selector == "exact_dynamic_tile_marginal":
            trace = exact_mixed_page_greedy(
                matrices["gate"][0], matrices["up"][0], matrices["down"][0],
                matrices["gate"][2], matrices["up"][2], matrices["down"][2], x,
                tile_shape=shape, page_budgets=requested, refresh_interval=refresh,
                proxy=proxy, beta=beta, device=device,
            )
            label = f"exact_dynamic_tile_marginal_refresh_{refresh}"
            regime = "h0_oracle_full_suffix"
            # Hardware traffic depends on cache residency and kernel fusion.
            # Report the reproducible unique input footprint lower bound and
            # use measured wall time for the actual refresh cost.
            bytes_read = selector_input_bytes
        elif selector in {"static_first_order", "static_wina"}:
            method = "first_order" if selector == "static_first_order" else "wina"
            trace = static_proxy_mixed_page_order(
                matrices["gate"][0], matrices["up"][0], matrices["down"][0],
                matrices["gate"][2], matrices["up"][2], matrices["down"][2], x,
                tile_shape=shape, page_budgets=requested, proxy_method=method,
                proxy=proxy, beta=beta, device=device,
            )
            label = selector
            regime = "h0_oracle_suffix_metadata_heavy"
            bytes_read = selector_input_bytes
        else:
            raise ValueError(f"unknown tile selector: {selector}")
        synchronize(device)
        elapsed = time.perf_counter() - started
        assert_unique_trace(trace, total_pages)
        for budget_bpw, budget_pages in zip(budgets_bpw, page_budgets):
            if budget_pages > total_pages:
                raise RuntimeError("tile budget exceeds representation endpoint")
            output = trace.snapshots[budget_pages].numpy()
            rows.append({
                **metadata,
                "objective": "exact_sequential_complete_expert_qenergy",
                "selection_regime": regime,
                "selection_regime_category": "h0_oracle",
                "action_family": "gate_up_tile_plus_down_column",
                "representation": "paired_direct_q2_q4_gate_up_tile_plus_down_column",
                "tile_shape": f"{shape[0]}x{shape[1]}",
                "selector": label,
                "refresh_interval": int(refresh),
                "global_refreshes": math.ceil(budget_pages / max(refresh, 1)) if refresh else 1,
                "tile_pages_selected": int(np.sum(np.asarray(trace.order[:budget_pages]) < geometry.tile_pages)),
                "down_pages_selected": int(np.sum(np.asarray(trace.order[:budget_pages]) >= geometry.tile_pages)),
                **base_storage,
                **standard_frontier_fields(
                    physical_budget_bpw=budget_bpw, pages=budget_pages,
                    logical_actions=budget_pages,
                    recovery_value=recovery(target, base, output, proxy, beta),
                    runtime_seconds=elapsed, selector_bytes_read=bytes_read,
                    multiplier=float(base_storage["storage_multiplier"]),
                    logical_payload_bytes=budget_pages * PAGE_BYTES,
                ),
            })

    # Controlled input-coordinate action-space baseline in every tile table.
    total_baseline_pages = 2048 // 2 + 512
    baseline_requested = sorted(set(page_budgets) | {total_baseline_pages})
    synchronize(device)
    baseline_started = time.perf_counter()
    baseline = input_coordinate_baseline_trace(
        matrices["gate"][0], matrices["up"][0], matrices["down"][0],
        matrices["gate"][2], matrices["up"][2], matrices["down"][2], x,
        baseline_requested, proxy, beta, device,
    )
    synchronize(device)
    baseline_seconds = time.perf_counter() - baseline_started
    assert_complete_endpoint(baseline, total_baseline_pages, target, "input-coordinate baseline")
    for budget_bpw, budget_pages in zip(budgets_bpw, page_budgets):
        rows.append({
            **metadata,
            "objective": "exact_sequential_complete_expert_qenergy",
            "selection_regime": "h0_oracle_full_suffix",
            "selection_regime_category": "h0_oracle",
            "action_family": "input_coordinate_baseline",
            "representation": "direct_q2_q4_paired_gate_up_input_packets_plus_down_columns",
            "tile_shape": "input_coordinate_2columns_per_page",
            "selector": "static_first_order_input_coordinate_baseline",
            "refresh_interval": 0,
            "global_refreshes": 1,
            "tile_pages_selected": int(np.sum(np.asarray(baseline.order[:budget_pages]) < 1024)),
            "down_pages_selected": int(np.sum(np.asarray(baseline.order[:budget_pages]) >= 1024)),
            **base_storage,
            **standard_frontier_fields(
                physical_budget_bpw=budget_bpw, pages=budget_pages,
                logical_actions=budget_pages,
                recovery_value=recovery(
                    target, base, baseline.snapshots[budget_pages].numpy(), proxy, beta,
                ),
                runtime_seconds=baseline_seconds, selector_bytes_read=selector_input_bytes,
                multiplier=float(base_storage["storage_multiplier"]),
                logical_payload_bytes=budget_pages * PAGE_BYTES,
            ),
        })
    return rows


def support_stability_rows(cache: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_occurrence: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for value in cache:
        key = (
            value["capture_source"], value["evaluation_split"], value["request_id"],
            value["position"], value["layer"], value["expert_id"],
            value["score_method"], value["shortlist_size"],
        )
        by_occurrence[key][value["projection"]] = value
    for values in by_occurrence.values():
        if set(values) != {"gate", "up"}:
            continue
        left, right = values["gate"], values["up"]
        a, b = set(map(int, left["action_ids"])), set(map(int, right["action_ids"]))
        rows.append({
            **{name: left[name] for name in (*IDENTITY_COLUMNS, "sequence_id", "expert_stratum", "router_rank")},
            "projection": "gate_up",
            "score_method": left["score_method"],
            "shortlist_size": int(left["shortlist_size"]),
            "relationship": "same_invocation_gate_up",
            "token_gap": 0,
            "support_action_count": len(a),
            "right_support_action_count": len(b),
            "support_budget_bpw": float(left.get("support_budget_bpw", np.nan)),
            "support_jaccard": len(a & b) / max(len(a | b), 1),
            "importance_weighted_overlap": weighted_jaccard(
                left["action_ids"], left["action_scores"], right["action_ids"], right["action_scores"],
            ),
        })

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for value in cache:
        # Cross-reference capture positions are intentionally strided.  Do not
        # turn an accidental numeric adjacency into an adjacent-token claim.
        if value["capture_source"] != "exact_checkpoint":
            continue
        key = (
            value["capture_source"], value["evaluation_split"], value["sequence_id"],
            value["layer"], value["expert_id"], value["projection"],
            value["score_method"], value["shortlist_size"],
        )
        grouped[key].append(value)
    for values in grouped.values():
        ordered = sorted(values, key=lambda row: (int(row["position"]), str(row["request_id"])))
        for left, right in zip(ordered, ordered[1:]):
            gap = int(right["position"]) - int(left["position"])
            if gap != 1:
                continue
            a, b = set(map(int, left["action_ids"])), set(map(int, right["action_ids"]))
            rows.append({
                **{name: left[name] for name in (*IDENTITY_COLUMNS, "sequence_id", "expert_stratum", "router_rank")},
                "right_request_id": right["request_id"],
                "right_position": int(right["position"]),
                "projection": left["projection"],
                "score_method": left["score_method"],
                "shortlist_size": int(left["shortlist_size"]),
                "relationship": "adjacent_token_same_sequence",
                "token_gap": gap,
                "support_action_count": len(a),
                "right_support_action_count": len(b),
                "support_budget_bpw": float(left.get("support_budget_bpw", np.nan)),
                "support_jaccard": len(a & b) / max(len(a | b), 1),
                "importance_weighted_overlap": weighted_jaccard(
                    left["action_ids"], left["action_scores"], right["action_ids"], right["action_scores"],
                ),
            })
    return rows


STANDARD_FRONTIER_COLUMNS = [
    *IDENTITY_COLUMNS, "objective", "selection_regime", "physical_budget_bpw",
    "physical_pages", "physical_bytes", "logical_actions", "recovery",
    "selector_runtime_ms", "selector_bytes_read", "storage_multiplier", "page_amplification",
]

ARTIFACT_SCHEMAS: dict[str, list[str]] = {
    "neuron_major_frontier": [
        *STANDARD_FRONTIER_COLUMNS, "action_family", "representation", "selector",
        "suffix_storage_bpw", "selector_metadata_bpw", "selector_metadata_bytes_per_expert",
    ],
    "activation_shortlist_containment": [
        *STANDARD_FRONTIER_COLUMNS, "projection", "score_method", "shortlist_size",
        "refresh_block", "candidate_overfetch_factor", "exact_gain_retained",
        "exact_action_recall", "importance_weighted_recall", "candidate_pages", "fetched_pages",
        "recovery_diagonal", "recovery_exact", "recovery_constrained",
        "recovery_exact_matched_fetched_pages", "recovery_diagonal_matched_fetched_pages",
        "fraction_exact_over_diagonal_gain_retained_matched_fetched_pages",
        "suffix_storage_bpw", "selector_metadata_bpw", "selector_metadata_bytes_per_expert",
    ],
    "tile_streaming_frontier": [
        *STANDARD_FRONTIER_COLUMNS, "action_family", "tile_shape", "selector",
        "suffix_storage_bpw", "selector_metadata_bpw", "selector_metadata_bytes_per_expert",
    ],
    "exact_action_labels": [
        *IDENTITY_COLUMNS, "projection", "action_rank", "action_id", "coordinate",
        "refinement_stage", "refinement_plane", "marginal_gain", "physical_page_id",
        "layout_method", "coordinate_rank_abs_activation",
        "coordinate_rank_activation_weighted_delta_norm",
        "coordinate_rank_future_proxy_weighted",
    ],
    "support_stability": [
        *IDENTITY_COLUMNS, "projection", "score_method", "shortlist_size", "relationship", "token_gap",
        "support_jaccard", "importance_weighted_overlap", "support_action_count",
        "right_support_action_count", "support_budget_bpw",
    ],
    "activation_concentration": [
        *IDENTITY_COLUMNS, "top_k", "top_k_energy_fraction", "exact_zero_fraction",
        "coordinates_for_90pct_energy",
    ],
}

ARTIFACT_FILES = {
    "neuron_major_frontier": "neuron_major_frontier.parquet",
    "activation_shortlist_containment": "activation_shortlist_containment.parquet",
    "tile_streaming_frontier": "tile_streaming_frontier.parquet",
    "exact_action_labels": "exact_action_labels.parquet",
    "support_stability": "support_stability.parquet",
    "activation_concentration": "activation_concentration.parquet",
}


def assert_artifact_schema(name: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    missing = set(ARTIFACT_SCHEMAS[name]) - set(rows[0])
    if missing:
        raise RuntimeError(f"{name} is missing required columns: {sorted(missing)}")


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [] if not path.exists() else pd.read_parquet(path).to_dict("records")


def serializable_support_cache(cache: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in cache:
        value = dict(row)
        value["action_ids"] = np.asarray(value["action_ids"], np.int64).tolist()
        value["action_scores"] = np.asarray(value["action_scores"], np.float64).tolist()
        result.append(value)
    return result


def flush_checkpoint(
    output: Path, state: dict[str, list[dict[str, Any]]], support_cache: list[dict[str, Any]],
    facts: dict[str, Any],
) -> None:
    state["support_stability"] = support_stability_rows(support_cache)
    for name, filename in ARTIFACT_FILES.items():
        assert_artifact_schema(name, state[name])
        atomic_parquet(output / filename, state[name], ARTIFACT_SCHEMAS[name])
    atomic_parquet(
        output / "_support_cache.parquet", serializable_support_cache(support_cache),
        [*IDENTITY_COLUMNS, "sequence_id", "projection", "score_method", "shortlist_size",
         "support_action_count", "support_budget_bpw", "action_ids", "action_scores"],
    )
    facts["artifact_row_counts"] = {name: len(state[name]) for name in ARTIFACT_FILES}
    facts["updated_unix"] = time.time()
    atomic_json(output / "run_facts.json", facts)


def strata_for_layer(
    data: dict[str, np.ndarray], layer: int, source: str, config: dict[str, Any],
) -> dict[int, str]:
    key = (
        "locked_exact_sampled_experts_from_pr4"
        if source == "exact_checkpoint"
        else "locked_cross_sampled_experts_from_pr6"
    )
    if key not in config or str(layer) not in config[key]:
        raise RuntimeError(f"missing pre-registered expert strata at {key}.{layer}")
    return {
        int(expert): str(stratum)
        for expert, stratum in config[key][str(layer)].items()
    }


def layer_record_plan(
    data: dict[str, np.ndarray], layer: int, split: str, mode: str,
    source: str, config: dict[str, Any],
) -> list[tuple[int, str, np.ndarray, np.ndarray]]:
    strata = strata_for_layer(data, layer, source, config)
    if mode == "full":
        if split != "test":
            raise RuntimeError("full mode may evaluate only held-out test rows")
        maximum = int(config["max_test_invocations_per_expert"])
        experts = sorted(strata)
    elif split == "test":
        maximum = int(config["pilot_test_invocations_per_expert"])
        if source == "exact_checkpoint":
            experts = [int(config["pilot_expert_per_layer"][str(layer)])]
        else:
            experts = [next(expert for expert, label in strata.items() if label == "hot")]
    elif split == "validation":
        maximum = int(config["pilot_validation_invocations_per_expert"])
        if source == "exact_checkpoint":
            hot = int(config["pilot_expert_per_layer"][str(layer)])
            experts = [hot] + [expert for expert in sorted(strata) if expert != hot]
        else:
            hot = next(expert for expert, label in strata.items() if label == "hot")
            experts = [hot] + [expert for expert in sorted(strata) if expert != hot]
    else:
        raise RuntimeError(f"unsupported split/mode combination: {split}/{mode}")

    policy = config["promotion_policy"]
    minimum_invocations = int(policy["promotion_min_validation_invocations_per_layer"])
    minimum_requests = int(policy["promotion_min_validation_requests_per_layer"])
    result: list[tuple[int, str, np.ndarray, np.ndarray]] = []
    request_ids: set[str] = set()
    invocations = 0
    for expert in experts:
        records, ranks = occurrence(data, expert, split, maximum)
        if len(records):
            result.append((expert, strata[expert], records, ranks))
            invocations += len(records)
            request_ids.update(map(str, data["request_id"][records]))
        if mode == "pilot" and split == "validation":
            if (
                invocations >= minimum_invocations
                and len(request_ids) >= minimum_requests
            ):
                break
        elif mode == "pilot":
            break
    if mode == "pilot" and split == "validation":
        if (
            invocations < minimum_invocations
            or len(request_ids) < minimum_requests
        ):
            raise RuntimeError(f"layer {layer} lacks locked validation promotion coverage")
    return result


def output_state(
    output: Path, completed_work_units: Iterable[str],
) -> tuple[
    dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, int],
]:
    """Load only expert transactions committed by the last facts checkpoint.

    Artifact files are replaced before ``run_facts.json`` during a flush. If
    the process dies between those replacements, the files can contain rows
    for an expert whose work-unit commit was never recorded. Dropping those
    rows on resume makes the facts file the transaction journal and prevents
    a rerun from duplicating partial evidence.
    """
    completed = set(map(str, completed_work_units))

    def committed(row: Mapping[str, Any]) -> bool:
        work_unit = (
            f"{row['evaluation_split']}:{int(row['layer'])}:"
            f"{int(row['expert_id'])}"
        )
        return work_unit in completed

    state: dict[str, list[dict[str, Any]]] = {}
    discarded: dict[str, int] = {}
    for name, filename in ARTIFACT_FILES.items():
        loaded = load_rows(output / filename)
        kept = [row for row in loaded if committed(row)]
        state[name] = kept
        discarded[name] = len(loaded) - len(kept)
    loaded_cache = load_rows(output / "_support_cache.parquet")
    cache = [row for row in loaded_cache if committed(row)]
    discarded["_support_cache"] = len(loaded_cache) - len(cache)
    for row in cache:
        row["action_ids"] = np.asarray(row["action_ids"], np.int64)
        row["action_scores"] = np.asarray(row["action_scores"], np.float64)
    return state, cache, discarded


def run_split(
    *, split: str, mode: str, source: str, config: dict[str, Any], all_data: dict[str, np.ndarray],
    checkpoint: Path, index: dict[str, str], trees: dict[str, Any], device: torch.device,
    promotions: dict[str, Any] | None, state: dict[str, list[dict[str, Any]]],
    support_cache: list[dict[str, Any]], facts: dict[str, Any], output: Path,
) -> None:
    completed = set(map(str, facts.get("completed_work_units", [])))
    for layer in map(int, config["layers"]):
        mask = all_data["layer"] == layer
        data = {name: value[mask] for name, value in all_data.items()}
        proxy, proxy_facts = proxy_gradients(
            data["xplus"], [data[f"h{h}_router_logits"] for h in range(1, 5)], data["split"],
        )
        beta = float(proxy_facts["beta"])
        facts.setdefault("proxy_facts_by_layer", {})[str(layer)] = proxy_facts
        for expert, stratum, records, ranks in layer_record_plan(
            data, layer, split, mode, source, config,
        ):
            work_unit = f"{split}:{layer}:{expert}"
            if work_unit in completed:
                continue
            tensors = {
                projection: load_compressed_mxfp4_expert(checkpoint, index, layer, expert, projection)
                for projection in PROJECTIONS
            }
            matrices = {
                projection: [trees[projection].decode(tensors[projection], level) for level in (2, 3, 4)]
                for projection in PROJECTIONS
            }
            deltas = {
                projection: np.stack((
                    (matrices[projection][1] - matrices[projection][0]).T,
                    (matrices[projection][2] - matrices[projection][1]).T,
                ))
                for projection in PROJECTIONS
            }
            run_shortlist_family = promotions is None or promotions["activation_shortlist"]["enabled"]
            grams = {
                projection: build_base_gram(deltas[projection], device=str(device))
                for projection in SHORTLIST_PROJECTIONS
            } if run_shortlist_family else {}
            metadata_started = time.perf_counter()
            raw_down_norm = row_qenergies(
                (matrices["down"][2] - matrices["down"][0]).T, proxy, beta,
            )
            if not np.all(np.isfinite(raw_down_norm)) or np.any(raw_down_norm < 0.0):
                raise RuntimeError("down-norm selector metadata is non-finite or negative")
            down_norm_scale = np.float32(np.max(raw_down_norm, initial=0.0))
            normalized_down_norm = np.clip(raw_down_norm / max(float(down_norm_scale), 1e-30), 0.0, 1.0).astype(np.float16)
            down_norm_metadata = normalized_down_norm.astype(np.float64) * float(down_norm_scale)
            metadata_error = down_norm_metadata - raw_down_norm
            metadata_seconds = time.perf_counter() - metadata_started
            facts.setdefault("selector_metadata_build_by_expert", {})[
                f"{layer}:{expert}"
            ] = {
                "method": "qenergy_down_delta_norm",
                "encoding": "normalized_fp16_plus_fp32_expert_scale",
                "bytes": int(2 * len(down_norm_metadata) + 4),
                "offline_build_seconds": metadata_seconds,
                "scale_fp32": float(down_norm_scale),
                "clipped_values": 0,
                "max_absolute_quantization_error": float(np.max(np.abs(metadata_error), initial=0.0)),
                "relative_l2_quantization_error": float(np.linalg.norm(metadata_error) / max(np.linalg.norm(raw_down_norm), 1e-30)),
                "fit_split": "none_weight_only",
            }
            local = {
                "neuron_major_frontier": [], "activation_shortlist_containment": [],
                "tile_streaming_frontier": [], "exact_action_labels": [],
                "activation_concentration": [],
            }
            local_support: list[dict[str, Any]] = []
            for record, rank in zip(records, ranks):
                invocation_started = time.perf_counter()
                x = np.asarray(data["x"][record], np.float32)
                metadata = common_metadata(
                    source, split, data, int(record), int(rank), layer, expert, stratum,
                )
                run_neuron = promotions is None or promotions["neuron_major"]["enabled"]
                local["activation_concentration"].extend(activation_concentration_rows(x, metadata))
                run_shortlist = promotions is None or promotions["activation_shortlist"]["enabled"]
                run_tile = promotions is None or promotions["tile_streaming"]["enabled"]
                selected_neuron = None if promotions is None else promotions["neuron_major"]["proxy_selectors"]
                selected_shortlist = None if promotions is None else promotions["activation_shortlist"]["selection"]
                selected_tile = None if promotions is None else promotions["tile_streaming"]["selection"]
                if run_neuron:
                    local["neuron_major_frontier"].extend(evaluate_neuron_frontier(
                        matrices, x, proxy, beta, metadata, config, device,
                        selected_neuron, down_norm_metadata,
                    ))
                if run_shortlist:
                    local["activation_shortlist_containment"].extend(evaluate_shortlists(
                        matrices, deltas, grams, x, proxy, beta, metadata, config, selected_shortlist,
                        local["exact_action_labels"], local_support,
                    ))
                if run_tile:
                    local["tile_streaming_frontier"].extend(evaluate_tile_frontier(
                        matrices, x, proxy, beta, metadata, config, device, selected_tile,
                    ))
                print(json.dumps({
                    "completed_invocation": {
                        "split": split,
                        "layer": int(layer),
                        "expert_id": int(expert),
                        "request_id": metadata["request_id"],
                        "position": int(metadata["position"]),
                    },
                    "elapsed_seconds": time.perf_counter() - invocation_started,
                    "rows_buffered": {name: len(rows) for name, rows in local.items()},
                }, sort_keys=True), flush=True)
            # Commit only a complete expert work unit, so a crash cannot leave
            # a partially represented cohort marked resumable.
            for name, rows in local.items():
                state[name].extend(rows)
            support_cache.extend(local_support)
            completed.add(work_unit)
            facts["completed_work_units"] = sorted(completed)
            facts["last_completed"] = {
                "split": split, "layer": layer, "expert_id": expert, "unix": time.time(),
            }
            flush_checkpoint(output, state, support_cache, facts)
            del matrices, deltas, grams, tensors
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()


def validate_config_controls(config: dict[str, Any]) -> None:
    expected = {
        "page_size_bytes": PAGE_BYTES,
        "activation_shortlist_unit": "input_coordinates",
        "coordinate_score_aggregation": "sum_stage_energy",
        "shortlist_physical_layout": "canonical_paired_planes_four_coordinates_per_projection_page",
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise RuntimeError(f"locked config control changed at {name}: {config.get(name)!r} != {value!r}")
    if tuple(map(int, config["tile_pilot_shape"])) != (32, 32):
        raise RuntimeError("the bounded pilot must start with the page-aligned 32x32 tile")
    minimum_overfetch = float(config["promotion_policy"]["shortlist_min_overfetch"])
    if minimum_overfetch != 1.0:
        raise RuntimeError("shortlist_min_overfetch must remain frozen at 1.0")


def cohort_counts(state: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    values: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
    for name in ARTIFACT_FILES:
        for row in state[name]:
            values[str(row["evaluation_split"])].add(tuple(row[column] for column in IDENTITY_COLUMNS))
    return {split: len(identities) for split, identities in values.items()}


def expected_counts(
    all_data: dict[str, np.ndarray], config: dict[str, Any], source: str, mode: str,
) -> dict[str, int]:
    splits = ("validation", "test") if mode == "pilot" else ("test",)
    result = {split: 0 for split in splits}
    for layer in map(int, config["layers"]):
        mask = all_data["layer"] == layer
        data = {name: value[mask] for name, value in all_data.items()}
        for split in splits:
            result[split] += sum(
                len(records) for _, _, records, _ in layer_record_plan(
                    data, layer, split, mode, source, config,
                )
            )
    return result


def _is_canonical_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _promotion_metric(candidate: Mapping[str, Any], name: str) -> float:
    value = candidate.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"promotion candidate metric {name} is missing or non-numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"promotion candidate metric {name} is non-finite")
    return result


def _require_gate_flag(
    candidate: Mapping[str, Any], name: str, expected: bool,
) -> None:
    if candidate.get(name) is not expected:
        raise RuntimeError(
            f"promotion candidate gate {name} disagrees with its frozen aggregate metric"
        )


def _validate_promoted_candidate_gates(
    family: str, candidate: Mapping[str, Any], policy: Mapping[str, Any],
) -> None:
    if family == "neuron_major":
        expectations = {
            "passes_median": _promotion_metric(candidate, "median")
            >= float(policy["unit_go_at_1bpw_median"]),
            "passes_p10": _promotion_metric(candidate, "p10")
            >= float(policy["unit_go_at_1bpw_p10"]),
        }
    elif family == "activation_shortlist":
        if policy.get("shortlist_overfetch_statistic") != "p90":
            raise RuntimeError("promotion shortlist overfetch statistic changed from p90")
        passes_min_overfetch = (
            _promotion_metric(candidate, "p10_candidate_overfetch")
            >= float(policy["shortlist_min_overfetch"])
        )
        passes_max_overfetch = (
            _promotion_metric(candidate, "p90_candidate_overfetch")
            <= float(policy["shortlist_max_overfetch"])
        )
        expectations = {
            "passes_median": _promotion_metric(candidate, "median")
            >= float(policy["shortlist_min_exact_gain_retention_median"]),
            "passes_p10": _promotion_metric(candidate, "p10")
            >= float(policy["shortlist_min_exact_gain_retention_p10"]),
            "passes_candidates": _promotion_metric(candidate, "shortlist_size")
            <= int(policy["shortlist_max_candidates"]),
            "passes_min_overfetch": passes_min_overfetch,
            "passes_max_overfetch": passes_max_overfetch,
            "passes_overfetch": passes_min_overfetch and passes_max_overfetch,
        }
    elif family == "tile_streaming":
        page_reduction = candidate.get("page_reduction_at_matched_recovery")
        passes_page_reduction = False
        if page_reduction is not None:
            passes_page_reduction = (
                _promotion_metric(candidate, "page_reduction_at_matched_recovery")
                >= float(policy["tile_min_page_reduction_at_matched_recovery"])
            )
        expectations = {
            "passes_recovery_gain": _promotion_metric(
                candidate, "recovery_point_gain_vs_locked_pr6_h0_hybrid",
            ) >= float(policy["tile_min_recovery_point_gain"]),
            "passes_page_reduction": passes_page_reduction,
        }
    else:
        raise RuntimeError(f"unknown promotion family {family}")
    for name, expected in expectations.items():
        _require_gate_flag(candidate, name, bool(expected))
    if family == "tile_streaming":
        passed = expectations["passes_recovery_gain"] or expectations["passes_page_reduction"]
    else:
        passed = all(expectations.values())
    if not passed:
        raise RuntimeError(f"promoted {family} row does not pass its frozen validation gates")


def validate_and_parse_promotions(
    value: dict[str, Any], config: dict[str, Any], hashes: dict[str, str], source: str,
) -> dict[str, Any]:
    if value.get("schema_version") != 1:
        raise RuntimeError("unsupported sparse-streaming promotions schema")
    if value.get("generated_by") != "oracle_study.sparse_streaming_analysis.choose_validation_promotions":
        raise RuntimeError("promotion artifact was not emitted by the locked analyzer path")
    if value.get("config_sha256") != hashes["config_file_sha256"]:
        raise RuntimeError("promotion root config_sha256 changed")
    evidence = value.get("validation_evidence_sha256")
    if not isinstance(evidence, Mapping) or set(evidence) != {
        "neuron_major", "activation_shortlist", "tile_streaming",
    } or any(not _is_canonical_sha256(digest) for digest in evidence.values()):
        raise RuntimeError("promotion validation evidence hashes are missing or malformed")
    policy = config["promotion_policy"]
    expected_coverage = {
        "layers": list(map(int, config["layers"])),
        "invocations_per_layer": int(policy["promotion_min_validation_invocations_per_layer"]),
        "distinct_requests_per_layer": int(policy["promotion_min_validation_requests_per_layer"]),
    }
    if value.get("minimum_validation_coverage") != expected_coverage:
        raise RuntimeError("promotion minimum validation coverage changed")
    coverage = value.get("validation_coverage")
    if not isinstance(coverage, list) or any(not isinstance(row, Mapping) for row in coverage):
        raise RuntimeError("promotion validation coverage is missing or malformed")
    expected_top = {
        "selection_split": "validation",
        "selection_capture_source": "exact_checkpoint",
        "test_rows_consulted_for_selection": False,
    }
    for name, expected in expected_top.items():
        if value.get(name) != expected:
            raise RuntimeError(f"invalid promotion control {name}: {value.get(name)!r}")
    provenance = value.get("provenance", {})
    expected_provenance = {
        "config_sha256": hashes["config_file_sha256"],
        "reference_revision": config["reference"]["revision"],
        "checkpoint_config_sha256": hashes["config_sha256"],
        "checkpoint_index_sha256": hashes["index_sha256"],
        "tree_sha256": hashes["tree_sha256"],
        "capture_sha256": config["locked_capture_sha256"],
    }
    for name, expected in expected_provenance.items():
        if provenance.get(name) != expected:
            raise RuntimeError(f"promotion provenance changed at {name}")
    required_by_family = {
        "neuron_major": (
            "action_family", "representation", "selector", "selection_category", "selection_regime", "objective",
        ),
        "activation_shortlist": (
            "projection", "score_method", "shortlist_size", "refresh_block", "selection_category", "selection_regime",
        ),
        "tile_streaming": (
            "action_family", "tile_shape", "selector", "selection_category", "selection_regime", "objective",
        ),
    }
    frontier_by_family = {
        "neuron_major": "neuron_major",
        "activation_shortlist": "activation_shortlist",
        "tile_streaming": "tile_streaming",
    }
    plan: dict[str, Any] = {"source": source}
    for family in ("neuron_major", "activation_shortlist", "tile_streaming"):
        record = value.get(family)
        if not isinstance(record, Mapping) or record.get("status") not in {"promote", "stop"}:
            raise RuntimeError(f"promotion artifact has invalid {family} status")
        if not isinstance(record.get("all_validation_candidates"), list) or not isinstance(record.get("promoted"), list):
            raise RuntimeError(f"promotion artifact has malformed {family} candidate lists")
        all_candidates = list(record["all_validation_candidates"])
        if any(not isinstance(row, Mapping) for row in all_candidates):
            raise RuntimeError(f"promotion artifact has malformed {family} validation candidates")
        promoted = list(record["promoted"])
        if any(not isinstance(row, Mapping) or not set(required_by_family[family]) <= set(row) for row in promoted):
            raise RuntimeError(f"promotion artifact has incomplete {family} identity")
        if (record["status"] == "stop") != (len(promoted) == 0):
            raise RuntimeError(f"promotion {family} status disagrees with promoted records")
        identity = required_by_family[family]
        for promoted_row in promoted:
            matches = [
                candidate for candidate in all_candidates
                if all(candidate.get(name) == promoted_row.get(name) for name in identity)
            ]
            if len(matches) != 1:
                raise RuntimeError(f"promoted {family} row is not uniquely present in validation candidates")
            candidate = matches[0]
            _validate_promoted_candidate_gates(family, candidate, policy)
            for layer in expected_coverage["layers"]:
                layer_rows = [
                    row for row in coverage
                    if row.get("frontier") == frontier_by_family[family]
                    and int(row.get("layer", -1)) == int(layer)
                    and all(row.get(name) == promoted_row.get(name) for name in identity)
                    and (
                        family != "tile_streaming"
                        or math.isclose(float(row.get("physical_budget_bpw", float("nan"))), 1.0)
                    )
                ]
                if len(layer_rows) != 1:
                    raise RuntimeError(f"promoted {family} row lacks unique layer-{layer} validation coverage")
                evidence = layer_rows[0]
                if (
                    int(evidence.get("validation_invocations", -1))
                    < expected_coverage["invocations_per_layer"]
                    or int(evidence.get("validation_requests", -1))
                    < expected_coverage["distinct_requests_per_layer"]
                ):
                    raise RuntimeError(f"promoted {family} row fails layer-{layer} validation coverage")
        plan[family] = {"enabled": record["status"] == "promote", "promoted": promoted}

    neuron_selectors = [
        str(row["selector"]) for row in plan["neuron_major"]["promoted"]
        if str(row["selector"]) in UNIT_PROXY_NAMES
    ]
    plan["neuron_major"]["proxy_selectors"] = sorted(set(neuron_selectors))

    shortlist: dict[str, Any] = {}
    for row in plan["activation_shortlist"]["promoted"]:
        projection = str(row["projection"])
        if projection in shortlist:
            raise RuntimeError(f"multiple promoted shortlist rows for {projection}")
        shortlist[projection] = {
            "score_method": str(row["score_method"]),
            "shortlist_size": int(row["shortlist_size"]),
            "refresh_block": int(row["refresh_block"]),
        }
    plan["activation_shortlist"]["selection"] = shortlist

    tile_promoted = plan["tile_streaming"]["promoted"]
    if len(tile_promoted) > 1:
        raise RuntimeError("runner currently accepts at most one promoted tile category")
    tile_selection = None
    if tile_promoted:
        row = tile_promoted[0]
        label = str(row["selector"])
        refresh = 0
        selector = label
        prefix = "exact_dynamic_tile_marginal_refresh_"
        if label.startswith(prefix):
            selector = "exact_dynamic_tile_marginal"
            refresh = int(label[len(prefix):])
        tile_selection = {
            "selector": selector,
            "tile_shape": list(map(int, str(row["tile_shape"]).split("x"))),
            "refresh_interval": refresh,
        }
    plan["tile_streaming"]["selection"] = tile_selection
    return plan


def unique_invocation_count(state: dict[str, list[dict[str, Any]]]) -> int:
    identities: set[tuple[Any, ...]] = set()
    for name in ARTIFACT_FILES:
        for row in state[name]:
            identities.add(tuple(row[column] for column in IDENTITY_COLUMNS))
    return len(identities)


def validate_resume_provenance(
    existing: Mapping[str, Any], hashes: Mapping[str, str], *, mode: str,
    source: str, promotion_sha256: str | None,
) -> None:
    if (
        existing.get("config_sha256") != hashes["config_file_sha256"]
        or existing.get("input_hashes") != hashes
        or existing.get("run_mode") != mode
        or existing.get("source") != source
        or existing.get("validation_promotions_sha256") != promotion_sha256
    ):
        raise RuntimeError("resume checkpoint provenance differs from current locked inputs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trees", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", choices=("exact_checkpoint", "cross_reference"), required=True)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--promotions", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    existing_path = args.output / "run_facts.json"
    had_existing_facts = existing_path.exists()
    existing_facts_validated = False
    config = json.loads(args.config.read_text())
    validate_config_controls(config)
    facts: dict[str, Any] = {
        "run_id": config["run_id"],
        "run_mode": args.mode,
        "mode": args.mode,
        "source": args.source,
        "started_unix": time.time(),
        "completed": False,
        "failures": [],
        "codec_locked": True,
        "request_separation_verified": False,
        "page_size_bytes": int(config["page_size_bytes"]),
        "reference": {
            "revision": config["reference"]["revision"],
            "config_sha256": config["reference"]["config_sha256"],
            "index_sha256": config["reference"]["index_sha256"],
        },
        "locked_tree_sha256": config["locked_tree_sha256"],
        "locked_capture_sha256": config["locked_capture_sha256"],
        "request_split_policy": "training fits only training; analyzer promotions only validation; test never selects configuration",
        "selection_regime_enums": sorted(set(UNIT_SELECTION_REGIMES.values()) | {
            "h0_oracle_full_suffix", "h0_oracle_suffix_metadata_heavy",
            "h0_oracle_full_target_residual",
        }),
        "executed_refresh_blocks": [1],
        "executed_activation_shortlist_unit": "input_coordinates",
        "activation_concentration_top_k": list(ACTIVATION_CONCENTRATION_TOP_K),
        "conditional_followup_grids_not_executed": [
            "tile_shapes", "tile_selectors", "shortlist_refresh_blocks",
            "candidate_overfetch_factors", "sparse_graph_neighbours",
            "sparse_graph_base_ranks", "signature_encodings",
        ],
        "host": platform.node(),
        "torch": torch.__version__,
    }
    try:
        audit, hashes = verify_locked_inputs(
            config, args.config, args.captures, args.checkpoint, args.trees, args.source,
        )
        facts["config_sha256"] = hashes["config_file_sha256"]
        facts["input_hashes"] = hashes
        facts["checkpoint_audit"] = audit
        plan: dict[str, Any] | None = None
        if args.mode == "pilot":
            if args.promotions is not None:
                raise RuntimeError("pilot mode must not consume a promotion artifact")
        else:
            if args.promotions is None:
                raise RuntimeError("full mode requires analyzer-generated --promotions")
            payload = json.loads(args.promotions.read_text())
            plan = validate_and_parse_promotions(payload, config, hashes, args.source)
            facts["validation_promotions_sha256"] = sha256(args.promotions)
            facts["promotion_family_status"] = {
                family: bool(plan[family]["enabled"])
                for family in ("neuron_major", "activation_shortlist", "tile_streaming")
            }
        loaded = np.load(args.captures, allow_pickle=False)
        all_data = {name: loaded[name] for name in loaded.files}
        facts["request_counts_by_split"] = verify_request_separation(all_data)
        facts["request_separation_verified"] = True
        expected_by_split = expected_counts(all_data, config, args.source, args.mode)
        facts["expected_cohort_counts"] = expected_by_split
        facts["expected_unique_invocations"] = int(sum(expected_by_split.values()))

        if existing_path.exists():
            existing = json.loads(existing_path.read_text())
            validate_resume_provenance(
                existing, hashes, mode=args.mode, source=args.source,
                promotion_sha256=facts.get("validation_promotions_sha256"),
            )
            existing_facts_validated = True
            if existing.get("completed"):
                return
            facts.update(existing)
            facts["resumed_unix"] = time.time()
        state, support_cache, discarded_partial = output_state(
            args.output, facts.get("completed_work_units", []),
        )
        if any(discarded_partial.values()):
            facts.setdefault("resume_discarded_partial_rows", []).append({
                "unix": time.time(), "rows_by_artifact": discarded_partial,
            })
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        facts["device"] = str(device)
        facts["gpu"] = torch.cuda.get_device_name(device) if device.type == "cuda" else None
        index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
        records = json.loads(args.trees.read_text())
        trees = {projection: tree_from_record(records[projection]) for projection in PROJECTIONS}

        if args.mode == "pilot":
            # The bounded test sanity rows run the same fixed pilot grid, but
            # no decision is made here; only analyze_sparse_streaming.py may
            # freeze validation promotions.
            for split in ("validation", "test"):
                run_split(
                    split=split, mode="pilot", source=args.source, config=config,
                    all_data=all_data, checkpoint=args.checkpoint, index=index, trees=trees,
                    device=device, promotions=None, state=state, support_cache=support_cache,
                    facts=facts, output=args.output,
                )
        else:
            if plan is None:
                raise RuntimeError("full mode promotion plan was not validated")
            run_split(
                split="test", mode="full", source=args.source, config=config,
                all_data=all_data, checkpoint=args.checkpoint, index=index, trees=trees,
                device=device, promotions=plan, state=state, support_cache=support_cache,
                facts=facts, output=args.output,
            )
        observed = unique_invocation_count(state)
        facts["observed_unique_invocations"] = int(observed)
        facts["observed_cohort_counts"] = cohort_counts(state)
        if observed != int(facts["expected_unique_invocations"]):
            raise RuntimeError(
                f"observed unique invocations {observed} != expected {facts['expected_unique_invocations']}"
            )
        facts["completed"] = True
        facts["completed_unix"] = time.time()
        flush_checkpoint(args.output, state, support_cache, facts)
    except Exception as error:
        # A command pointed at an existing directory may fail before its
        # provenance has been accepted (wrong mode/source/promotions, corrupt
        # inputs, or a malformed journal). Preserve that prior transaction
        # journal verbatim. New runs and validated resumes still record their
        # own failure atomically.
        if not had_existing_facts or existing_facts_validated:
            facts.setdefault("failures", []).append({
                "unix": time.time(), "type": type(error).__name__, "message": str(error),
            })
            facts["completed"] = False
            atomic_json(existing_path, facts)
        raise


if __name__ == "__main__":
    main()
