#!/usr/bin/env python3
"""Broad fixed-coefficient allocator audit for the locked MXFP4 hierarchy.

This runner never selects trees, changes leaf codes, or fits on test requests.
It builds one activation-independent base Gram per expert/projection on CUDA,
then reuses that matrix across the requested validation/test invocations.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(EXPERIMENT / "scripts")]

from oracle_study.interaction_allocator import (
    action_ids,
    backward_elimination_from_gram,
    block_refresh_from_gram,
    build_base_gram,
    diagonal_from_gram,
    encode_factor,
    exact_marginal_fixed_greedy_from_gram,
    forward_backward_hybrid_from_gram,
    initial_full_marginal_from_gram,
    low_rank_marginal_fixed_greedy,
    recovery_curve_from_gram,
    support_overlap,
)
from oracle_study.mxfp4_embed import audit_checkpoint, load_compressed_mxfp4_expert
from oracle_study.mxfp4_selective import BitAction
from oracle_study.mxfp4_selective import pairwise_coselection_layout
from run_mxfp4_hierarchy_dense import select_varied_experts
from run_mxfp4_selective_pages import occurrence, silu, tree_from_record
from run_phase_a_remote import proxy_gradients


PROJECTIONS = ("gate", "up", "down")
TARGETS = (0.8, 0.9, 0.95, 0.99)
RANKS = (8, 16, 32, 64, 128)
LOW_RANK_METHODS = ("truncated_svd", "fixed_jl", "activation_weighted_svd")
ENCODINGS = ("fp16", "int8")
MATRIX_WEIGHTS = 2048 * 512
PAGE_SIZE = 512


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def first_crossing(curve: np.ndarray, target: float) -> int:
    found = np.flatnonzero(curve >= target)
    return int(found[0] + 1) if len(found) else int(len(curve))


def plane_order(gram: np.ndarray, activation: np.ndarray, exact: bool) -> tuple[np.ndarray, np.ndarray]:
    """Unnested fixed-coefficient order and recovery for one refinement plane."""
    k = np.asarray(gram, np.float32)
    a = np.asarray(activation, np.float32)
    correlation = k @ a
    denominator = max(float(a.astype(np.float64) @ correlation.astype(np.float64)), 1e-30)
    diagonal = np.diag(k)
    if not exact:
        order = np.argsort(-(a * a * diagonal), kind="stable")
        local_correlation = correlation.copy()
        recovered = 0.0
        curve = []
        for chosen in order:
            chosen = int(chosen)
            gain = 2.0 * a[chosen] * local_correlation[chosen] - a[chosen] ** 2 * diagonal[chosen]
            recovered += float(gain)
            local_correlation -= float(a[chosen]) * k[:, chosen]
            curve.append(recovered / denominator)
        if curve:
            curve[-1] = 1.0
        return order.astype(np.int64), np.asarray(curve)
    selected = np.zeros(len(a), bool)
    order: list[int] = []
    recovered = 0.0
    curve: list[float] = []
    for _ in range(len(a)):
        marginal = 2.0 * a * correlation - a * a * diagonal
        marginal[selected] = -np.inf
        chosen = int(np.argmax(marginal))
        selected[chosen] = True
        recovered += float(marginal[chosen])
        correlation -= float(a[chosen]) * k[:, chosen]
        order.append(chosen)
        curve.append(recovered / denominator)
    curve[-1] = 1.0
    return np.asarray(order, np.int64), np.asarray(curve)


def metric_action_features(
    deltas: np.ndarray, projection: str, proxy: np.ndarray, beta: float,
) -> np.ndarray:
    flat = np.asarray(deltas, np.float32).reshape(-1, deltas.shape[-1])
    if projection != "down":
        return flat
    future = math.sqrt(max(float(beta), 0.0)) * (flat @ np.asarray(proxy, np.float32))
    return np.concatenate((flat, future), axis=1)


def gpu_low_rank_factors(
    features: np.ndarray,
    maximum_rank: int,
    training_inputs: np.ndarray,
    seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return all maximum-rank factors and one full nonzero Gram spectrum."""
    values = torch.from_numpy(np.asarray(features, np.float32)).cuda()
    _, singular, vh = torch.linalg.svd(values, full_matrices=False)
    factors = {"truncated_svd": values @ vh[:maximum_rank].T}
    activation = np.asarray(training_inputs, np.float64)
    rms = np.sqrt(np.mean(activation * activation, axis=0))
    weights = torch.from_numpy(np.concatenate((rms, rms)).astype(np.float32)).cuda()
    _, _, weighted_vh = torch.linalg.svd(values * weights[:, None], full_matrices=False)
    factors["activation_weighted_svd"] = values @ weighted_vh[:maximum_rank].T
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    sketch = torch.randint(
        0, 2, (values.shape[1], maximum_rank), generator=generator, device="cuda", dtype=torch.int8
    ).float()
    sketch = (2.0 * sketch - 1.0) / math.sqrt(maximum_rank)
    factors["fixed_jl"] = values @ sketch
    return (
        {name: factor.cpu().numpy().astype(np.float32) for name, factor in factors.items()},
        singular.square().cpu().numpy().astype(np.float64),
    )


def hybrid_from_paths(
    forward: Sequence[Any], backward: Sequence[Any], forward_actions: int,
) -> list[Any]:
    limit = min(max(int(forward_actions), 0), len(forward))
    result = list(forward[:limit])
    n = len(forward) // 2
    depth = np.zeros(n, np.uint8)
    used = set()
    for action in result:
        depth[action.coordinate] = action.stage
        used.add((action.coordinate, action.stage))
    pending = [a for a in backward if (a.coordinate, a.stage) not in used]
    while pending:
        remainder = []
        progressed = False
        for action in pending:
            if action.stage == int(depth[action.coordinate]) + 1:
                result.append(action)
                depth[action.coordinate] = action.stage
                progressed = True
            else:
                remainder.append(action)
        if not progressed:
            raise AssertionError("hybrid path lost nested eligibility")
        pending = remainder
    return result


def validation_select_hybrid(
    exact: Sequence[Any],
    backward: Sequence[Any],
    gram: np.ndarray,
    activation: np.ndarray,
    rates: Sequence[float],
    fractions: Sequence[float],
) -> tuple[float, list[Any], dict[str, float]]:
    """Choose the static forward/back split using a validation activation."""
    n = gram.shape[0] // 2
    candidates: list[tuple[float, float, list[Any]]] = []
    score_by_fraction: dict[str, float] = {}
    mixed = [float(fraction) for fraction in fractions if 0.0 < float(fraction) < 1.0]
    if not mixed:
        raise ValueError("hybrid validation requires a forward fraction strictly between zero and one")
    for fraction in mixed:
        value = min(max(float(fraction), 0.0), 1.0)
        order = hybrid_from_paths(exact, backward, round(value * 2 * n))
        curve = recovery_curve_from_gram(order, gram, activation)
        counts = [
            min(max(int(round(float(rate) * n)), 1), 2 * n)
            for rate in rates if 0.25 <= float(rate) <= 1.0
        ]
        score = float(np.mean([curve[count - 1] for count in counts]))
        score_by_fraction[f"{value:g}"] = score
        # Deterministic tie break prefers a smaller exact prefix.
        candidates.append((score, -value, order))
    score, negative_fraction, order = max(candidates, key=lambda row: (row[0], row[1]))
    return -negative_fraction, order, score_by_fraction


def selector_paths(
    gram: np.ndarray,
    activation: np.ndarray,
    shortlist: int,
    hybrid_fraction: float,
    promoted_only: bool = False,
) -> list[tuple[str, Sequence[Any], float, int, int]]:
    """Return label, order, seconds, analytic selector bytes, refreshes."""
    actions = gram.shape[0]
    gram_bytes = int(gram.nbytes)
    result = []
    constructors = [
        ("diagonal", lambda: diagonal_from_gram(gram, activation), actions * 4, 0),
    ]
    if not promoted_only:
        constructors.append((
            "initial_full_marginal",
            lambda: initial_full_marginal_from_gram(gram, activation),
            gram_bytes,
            1,
        ))
    refresh_blocks = (16, 8, 4) if promoted_only else (32, 16, 8, 4)
    for block in refresh_blocks:
        constructors.append((
            f"block_refresh_{block}",
            lambda block=block: block_refresh_from_gram(gram, activation, block, shortlist_size=shortlist),
            gram_bytes + actions * actions * 4,
            math.ceil(actions / block),
        ))
    constructors.extend([
        (
            "exact_marginal_fixed_greedy",
            lambda: exact_marginal_fixed_greedy_from_gram(gram, activation),
            gram_bytes + actions * actions * 4,
            actions,
        ),
        (
            "backward_elimination",
            lambda: backward_elimination_from_gram(gram, activation),
            gram_bytes + actions * actions * 4,
            actions,
        ),
    ])
    for label, constructor, bytes_read, refreshes in constructors:
        started = time.perf_counter()
        order = constructor()
        result.append((label, order, time.perf_counter() - started, int(bytes_read), int(refreshes)))
    exact = next(row[1] for row in result if row[0] == "exact_marginal_fixed_greedy")
    backward = next(row[1] for row in result if row[0] == "backward_elimination")
    if not promoted_only:
        hybrid_started = time.perf_counter()
        hybrid = hybrid_from_paths(exact, backward, round(hybrid_fraction * actions))
        result.append((
            "validation_selected_forward_backward_hybrid",
            hybrid,
            time.perf_counter() - hybrid_started,
            2 * (gram_bytes + actions * actions * 4),
            actions,
        ))
    return result


def common_metadata(
    source: str, data: dict[str, np.ndarray], record: int, rank: int,
    layer: int, expert: int, stratum: str, projection: str, split: str,
) -> dict[str, Any]:
    return {
        "capture_source": source,
        "evaluation_split": split,
        "request_id": str(data["request_id"][record]),
        "sequence_id": str(data["sequence_id"][record]),
        "position": int(data["position"][record]),
        "layer": layer,
        "projection": projection,
        "expert_id": expert,
        "expert_stratum": stratum,
        "router_rank": int(rank) + 1,
        "router_coefficient": float(data["router_weights"][record, rank]),
    }


def add_broad_audit(
    output: list[dict[str, Any]],
    common: dict[str, Any],
    gram: np.ndarray,
    activation: np.ndarray,
    diagonal: Sequence[Any],
    exact: Sequence[Any],
    backward: Sequence[Any],
) -> None:
    for label, order in (
        ("diagonal", diagonal),
        ("exact_marginal_fixed_greedy", exact),
        ("backward_elimination", backward),
    ):
        curve = recovery_curve_from_gram(order, gram, activation)
        for target in TARGETS:
            crossing = first_crossing(curve, target)
            output.append({
                **common,
                "refinement_plane": "joint_nested_q2_q3_q4",
                "selector": label,
                "target_recovery": target,
                "actions_required": crossing,
                "logical_bpw_projection": crossing / (gram.shape[0] // 2),
                "recovery_at_crossing": float(curve[crossing - 1]),
                "complete_endpoint": float(curve[-1]),
            })
    n = gram.shape[0] // 2
    for stage, plane in enumerate(("q2_q3", "q3_q4")):
        local = gram[stage * n:(stage + 1) * n, stage * n:(stage + 1) * n]
        for label, is_exact in (("diagonal", False), ("exact_marginal_fixed_greedy", True)):
            _, curve = plane_order(local, activation, is_exact)
            for target in TARGETS:
                crossing = first_crossing(curve, target)
                output.append({
                    **common,
                    "refinement_plane": plane,
                    "selector": label,
                    "target_recovery": target,
                    "actions_required": crossing,
                    "logical_bpw_projection": crossing / n,
                    "recovery_at_crossing": float(curve[crossing - 1]),
                    "complete_endpoint": float(curve[-1]),
                })


def rows_for_paths(
    output: list[dict[str, Any]],
    overlaps: list[dict[str, Any]],
    common: dict[str, Any],
    gram: np.ndarray,
    activation: np.ndarray,
    paths: list[tuple[str, Sequence[Any], float, int, int]],
    rates: Sequence[float],
) -> dict[str, tuple[Sequence[Any], np.ndarray]]:
    n = gram.shape[0] // 2
    cached = {label: (order, recovery_curve_from_gram(order, gram, activation))
              for label, order, _, _, _ in paths}
    exact_order, exact_curve = cached["exact_marginal_fixed_greedy"]
    diagonal_order, diagonal_curve = cached["diagonal"]
    for label, order, seconds, bytes_read, refreshes in paths:
        curve = cached[label][1]
        for rate in rates:
            count = min(max(int(round(float(rate) * n)), 1), 2 * n)
            recovery = float(curve[count - 1])
            denominator = float(exact_curve[count - 1] - diagonal_curve[count - 1])
            retained = float("nan") if denominator <= 1e-12 else (
                recovery - float(diagonal_curve[count - 1])
            ) / denominator
            output.append({
                **common,
                "selector": label,
                "action_count": count,
                "logical_bpw_projection": count / n,
                "recovery": recovery,
                "selector_runtime_seconds_full_path": seconds,
                "selector_bytes_read_full_path": bytes_read,
                "global_refreshes_full_path": refreshes,
                "global_refreshes_to_budget": min(refreshes, math.ceil(count / max(1, (2 * n) // max(refreshes, 1)))),
                "fraction_exact_over_diagonal_gain_retained": retained,
                "complete_endpoint": float(curve[-1]),
            })
            overlaps.append({
                **common,
                "selector": label,
                "action_count": count,
                "logical_bpw_projection": count / n,
                "jaccard_with_exact": support_overlap(order, exact_order, count),
            })
    return cached


def low_rank_rows(
    output: list[dict[str, Any]],
    overlaps: list[dict[str, Any]],
    common: dict[str, Any],
    gram: np.ndarray,
    activation: np.ndarray,
    diagonal_order: Sequence[Any],
    exact_order: Sequence[Any],
    factor_by_method: dict[str, np.ndarray],
    rates: Sequence[float],
    combinations_to_run: Sequence[tuple[str, int, str]],
    physical_layout: np.ndarray,
    payload_bytes: int,
) -> None:
    if not combinations_to_run:
        return
    n = gram.shape[0] // 2
    exact_curve = recovery_curve_from_gram(exact_order, gram, activation)
    diagonal_curve = recovery_curve_from_gram(diagonal_order, gram, activation)
    exact_diag = np.diag(gram)
    inverse = np.empty_like(physical_layout)
    inverse[physical_layout] = np.arange(len(physical_layout), dtype=np.int64)
    per_page = max(PAGE_SIZE // payload_bytes, 1)
    pages_per_plane = math.ceil(n / per_page)
    coordinate_pages = inverse // per_page
    action_pages = np.concatenate((coordinate_pages, coordinate_pages + pages_per_plane))

    def select_at_budget(
        order: Sequence[BitAction], byte_budget: int,
    ) -> tuple[list[BitAction], set[int]]:
        maximum_pages = int(byte_budget) // PAGE_SIZE
        selected: list[BitAction] = []
        paid: set[int] = set()
        depth = np.zeros(n, np.uint8)
        for action in order:
            if action.stage != int(depth[action.coordinate]) + 1:
                continue
            action_id = (action.stage - 1) * n + action.coordinate
            page = int(action_pages[action_id])
            if page not in paid and len(paid) >= maximum_pages:
                continue
            paid.add(page)
            depth[action.coordinate] = action.stage
            selected.append(action)
        return selected, paid

    gram_cuda = torch.from_numpy(np.asarray(gram, np.float32)).cuda()
    expanded = np.concatenate((np.asarray(activation, np.float32),) * 2)
    total_energy = max(
        float(torch.from_numpy(expanded).cuda() @ gram_cuda @ torch.from_numpy(expanded).cuda()),
        1e-30,
    )

    def physical_frontier(
        order: Sequence[BitAction],
    ) -> dict[float, tuple[float, list[BitAction], set[int]]]:
        selections = []
        residuals = []
        for rate in rates:
            byte_budget = int(math.floor(float(rate) * MATRIX_WEIGHTS / 8 + 1e-9))
            selected_actions, selected_pages = select_at_budget(order, byte_budget)
            residual = expanded.copy()
            ids = [(action.stage - 1) * n + action.coordinate for action in selected_actions]
            residual[np.asarray(ids, np.int64)] = 0.0
            residuals.append(residual)
            selections.append((float(rate), selected_actions, selected_pages))
        residual_cuda = torch.from_numpy(np.stack(residuals)).cuda()
        energies = torch.sum((residual_cuda @ gram_cuda) * residual_cuda, dim=1).cpu().numpy()
        return {
            rate: (1.0 - float(energy) / total_energy, selected_actions, selected_pages)
            for (rate, selected_actions, selected_pages), energy in zip(selections, energies)
        }

    diagonal_physical_frontier = physical_frontier(diagonal_order)
    exact_physical_frontier = physical_frontier(exact_order)
    for method, rank, encoding in combinations_to_run:
        factor = factor_by_method[method][:, :rank]
        if method == "fixed_jl":
            factor = factor * math.sqrt(max(RANKS) / rank)
        encoded = encode_factor(factor, encoding)
        started = time.perf_counter()
        maximum_scored = max(
            min(max(int(round(float(rate) * n)), 1), 2 * n)
            for rate in rates if float(rate) < 2.0
        )
        order = low_rank_marginal_fixed_greedy_cuda(
            encoded.decode(), exact_diag, activation, maximum_scored, diagonal_order
        )
        elapsed = time.perf_counter() - started
        curve = recovery_curve_from_gram(order, gram, activation)
        approximate_physical_frontier = physical_frontier(order)
        for rate in rates:
            count = min(max(int(round(float(rate) * n)), 1), 2 * n)
            recovery = float(curve[count - 1])
            logical_denominator = float(exact_curve[count - 1] - diagonal_curve[count - 1])
            retained = float("nan") if logical_denominator <= 1e-12 else (
                recovery - float(diagonal_curve[count - 1])
            ) / logical_denominator
            physical_recovery, physical_selected, physical_pages = approximate_physical_frontier[float(rate)]
            diagonal_physical = diagonal_physical_frontier[float(rate)][0]
            exact_physical = exact_physical_frontier[float(rate)][0]
            physical_denominator = exact_physical - diagonal_physical
            physical_retained = float("nan") if physical_denominator <= 1e-12 else (
                physical_recovery - diagonal_physical
            ) / physical_denominator
            physical_bytes = len(physical_pages) * PAGE_SIZE
            logical_bytes = len(physical_selected) * payload_bytes
            output.append({
                **common,
                "method": method,
                "rank": rank,
                "metadata_encoding": encoding,
                "action_count": count,
                "logical_bpw_projection": count / n,
                "recovery": recovery,
                "fraction_exact_over_diagonal_gain_retained": retained,
                "physical_budget_bpw_projection": float(rate),
                "physical_page_count": len(physical_pages),
                "physical_bytes": physical_bytes,
                "physical_bpw_projection": 8 * physical_bytes / MATRIX_WEIGHTS,
                "logical_action_count_at_physical_budget": len(physical_selected),
                "logical_bpw_at_physical_budget": len(physical_selected) / n,
                "recovery_at_matched_physical_budget": physical_recovery,
                "diagonal_recovery_at_matched_physical_budget": diagonal_physical,
                "exact_recovery_at_matched_physical_budget": exact_physical,
                "fraction_exact_over_diagonal_gain_retained_at_matched_physical_budget": physical_retained,
                "page_amplification": physical_bytes / max(logical_bytes, 1),
                "selector_runtime_seconds_full_path": elapsed,
                "selector_bytes_read_full_path": int(maximum_scored * factor.nbytes),
                "global_refreshes_full_path": 0,
                "global_refreshes_to_budget": 0,
                "metadata_bytes_projection": encoded.metadata_bytes + exact_diag.astype(np.float32).nbytes,
                "metadata_bytes_all_40x256_experts_projection": (
                    encoded.metadata_bytes + exact_diag.astype(np.float32).nbytes
                ) * 40 * 256,
                "complete_endpoint": float(curve[-1]),
            })
            overlaps.append({
                **common,
                "selector": f"{method}_r{rank}_{encoding}",
                "action_count": count,
                "logical_bpw_projection": count / n,
                "jaccard_with_exact": support_overlap(order, exact_order, count),
            })


def low_rank_marginal_fixed_greedy_cuda(
    factor: np.ndarray,
    exact_diag: np.ndarray,
    activation: np.ndarray,
    maximum_scored_actions: int,
    tail_order: Sequence[BitAction],
) -> list[BitAction]:
    """CUDA U·p loop; append an order-irrelevant nested tail for 2-bpw saturation."""
    u = torch.from_numpy(np.asarray(factor, np.float32)).cuda()
    n = u.shape[0] // 2
    base = np.asarray(activation, np.float32).reshape(-1)
    expanded = np.concatenate((base, base))
    a = torch.from_numpy(expanded).cuda()
    diagonal = torch.from_numpy(np.array(exact_diag, dtype=np.float32, copy=True)).cuda()
    p = u.T @ a
    depth = torch.zeros(n, dtype=torch.uint8, device="cuda")
    selected = torch.zeros(2 * n, dtype=torch.bool, device="cuda")
    result: list[BitAction] = []
    for _ in range(min(int(maximum_scored_actions), 2 * n)):
        marginal = 2.0 * a * (u @ p) - a * a * diagonal
        eligible = torch.zeros(2 * n, dtype=torch.bool, device="cuda")
        eligible[:n] = depth == 0
        eligible[n:] = depth == 1
        marginal[~eligible | selected] = -torch.inf
        chosen = int(torch.argmax(marginal))
        score = float(marginal[chosen])
        stage = chosen // n + 1
        coordinate = chosen % n
        result.append(BitAction(coordinate, stage, score))
        selected[chosen] = True
        depth[coordinate] = stage
        p -= a[chosen] * u[chosen]
    used = {(action.coordinate, action.stage) for action in result}
    cpu_depth = np.zeros(n, np.uint8)
    for action in result:
        cpu_depth[action.coordinate] = action.stage
    pending = [action for action in tail_order if (action.coordinate, action.stage) not in used]
    while pending:
        remainder = []
        progressed = False
        for action in pending:
            if action.stage == int(cpu_depth[action.coordinate]) + 1:
                result.append(action)
                cpu_depth[action.coordinate] = action.stage
                progressed = True
            else:
                remainder.append(action)
        if not progressed:
            raise AssertionError("low-rank saturated tail lost nested eligibility")
        pending = remainder
    return result


def training_diagonal_layout(
    gram: np.ndarray, training_inputs: np.ndarray, payload_bytes: int,
) -> np.ndarray:
    """Current PR #4-style co-selection layout fitted only on train masks."""
    n = gram.shape[0] // 2
    masks = []
    for activation in np.asarray(training_inputs, np.float32):
        order = diagonal_from_gram(gram, activation)
        mask = np.zeros(n, np.uint8)
        for action in order[:n]:
            mask[action.coordinate] = 1
        masks.append(mask)
    return pairwise_coselection_layout(
        np.asarray(masks, np.uint8), max(PAGE_SIZE // payload_bytes, 1)
    )


def parse_promotions(path: Path | None) -> dict[str, list[tuple[str, int, str]]]:
    if path is None:
        return {}
    value = json.loads(path.read_text())
    return {
        projection: [(str(row["method"]), int(row["rank"]), str(row["metadata_encoding"])) for row in rows]
        for projection, rows in value["low_rank"].items()
    }


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
    parser.add_argument(
        "--disable-low-rank", action="store_true",
        help="Record spectra but skip low-rank factors and selectors (used after the validation gate fails).",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    audit = audit_checkpoint(args.checkpoint, config["layers"])
    if not audit["passed"]:
        raise RuntimeError(audit)
    expected_revision = config["reference"]["revision"]
    checkpoint_hashes = {
        "config_sha256": sha256(args.checkpoint / "config.json"),
        "index_sha256": sha256(args.checkpoint / "model.safetensors.index.json"),
    }
    for name, actual in checkpoint_hashes.items():
        expected = config["reference"][name]
        if actual != expected:
            raise RuntimeError(f"checkpoint {name} changed: {actual} != {expected}")
    tree_records = json.loads(args.trees.read_text())
    trees = {projection: tree_from_record(tree_records[projection]) for projection in PROJECTIONS}
    loaded = np.load(args.captures, allow_pickle=False)
    all_data = {name: loaded[name] for name in loaded.files}
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    minimum = 1 if args.source == "exact_checkpoint" else 3
    rates = list(map(float, config["projection_rates_bpw"]))[1:]
    promotions = parse_promotions(args.promotions)
    broad_rows: list[dict[str, Any]] = []
    selector_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    low_rank_output: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    facts: dict[str, Any] = {
        "run_id": config["run_id"],
        "source": args.source,
        "mode": args.mode,
        "started_unix": started,
        "host": platform.node(),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "checkpoint_audit": audit,
        "checkpoint_revision_locked": expected_revision,
        "checkpoint_hashes": checkpoint_hashes,
        "tree_sha256": sha256(args.trees),
        "capture_sha256": sha256(args.captures),
        "request_split_policy": "read unchanged from capture; expert fitting uses train, hyperparameter rows use validation, scientific rows use test",
    }
    if args.mode == "pilot" and not args.disable_low_rank:
        combinations_to_run = [(m, r, e) for m in LOW_RANK_METHODS for r in RANKS for e in ENCODINGS]
    else:
        combinations_to_run = []
    for layer in config["layers"]:
        layer_mask = all_data["layer"] == layer
        data = {name: value[layer_mask] for name, value in all_data.items()}
        strata, _ = select_varied_experts(data, 1, minimum)
        if args.source == "exact_checkpoint":
            strata = {
                int(expert): str(stratum)
                for expert, stratum in config["locked_exact_sampled_experts_from_pr4"][str(layer)].items()
            }
        experts = sorted(strata)
        if args.mode == "pilot":
            experts = [experts[-1]]
        proxy, proxy_facts = proxy_gradients(
            data["xplus"], [data[f"h{h}_router_logits"] for h in range(1, 5)], data["split"]
        )
        beta = float(proxy_facts["beta"])
        for expert in experts:
            tensors = {
                projection: load_compressed_mxfp4_expert(args.checkpoint, index, layer, expert, projection)
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
            train_records, _ = occurrence(
                data, expert, "train", int(config["max_train_invocations_per_expert"])
            )
            train_x = np.asarray(data["x"][train_records], np.float32)
            train_gate = torch.from_numpy(train_x).cuda() @ torch.from_numpy(matrices["gate"][2]).cuda().T
            train_up = torch.from_numpy(train_x).cuda() @ torch.from_numpy(matrices["up"][2]).cuda().T
            train_h = (silu(train_gate) * train_up).cpu().numpy()
            training_inputs = {"gate": train_x, "up": train_x, "down": train_h}
            val_records, val_ranks = occurrence(data, expert, "validation", 1 if args.mode == "pilot" else 0)
            test_maximum = 1 if args.mode == "pilot" else int(config["max_test_invocations_per_expert"])
            test_records, test_ranks = occurrence(data, expert, "test", test_maximum)
            for projection in PROJECTIONS:
                features = metric_action_features(deltas[projection], projection, proxy, beta)
                feature_deltas = features.reshape(2, deltas[projection].shape[1], features.shape[1])
                gram_started = time.perf_counter()
                gram = build_base_gram(feature_deltas, device="cuda")
                gram_seconds = time.perf_counter() - gram_started
                factor_by_method: dict[str, np.ndarray] = {}
                spectra: np.ndarray | None = None
                required_methods = LOW_RANK_METHODS if args.mode == "pilot" and not args.disable_low_rank else tuple(
                    sorted({row[0] for row in promotions.get(projection, [])})
                )
                if required_methods:
                    all_factors, spectra = gpu_low_rank_factors(
                        features, max(RANKS), training_inputs[projection],
                        int(config["seed"]) + layer * 257 + expert * 17,
                    )
                    factor_by_method = {name: all_factors[name] for name in required_methods}
                else:
                    values = torch.from_numpy(np.asarray(features, np.float32)).cuda()
                    spectra = torch.linalg.svdvals(values).square().cpu().numpy().astype(np.float64)
                payload_bytes = deltas[projection].shape[-1] // 8
                if required_methods:
                    physical_layout = training_diagonal_layout(
                        gram, training_inputs[projection], payload_bytes
                    )
                else:
                    physical_layout = np.arange(gram.shape[0] // 2, dtype=np.int64)
                total_spectrum = max(float(np.sum(spectra.astype(np.float64))), 1e-30)
                for rank in RANKS:
                    spectrum_rows.append({
                        "capture_source": args.source,
                        "layer": layer,
                        "expert_id": expert,
                        "expert_stratum": strata[expert],
                        "projection": projection,
                        "rank": rank,
                        "spectral_energy_retained": float(np.sum(spectra[:rank], dtype=np.float64) / total_spectrum),
                        "gram_trace": total_spectrum,
                        "gram_bytes": int(gram.nbytes),
                        "gram_build_seconds_cuda": gram_seconds,
                    })
                records: list[tuple[str, np.ndarray, np.ndarray]] = []
                if args.mode == "pilot":
                    records.append(("validation", val_records, val_ranks))
                records.append(("test", test_records, test_ranks))
                validation_selected: list[tuple[str, int, str]] = []
                selected_hybrid_fraction = 0.25
                for split, split_records, split_ranks in records:
                    for record, router_rank in zip(split_records, split_ranks):
                        x = np.asarray(data["x"][record], np.float32)
                        if projection == "down":
                            gate = matrices["gate"][2] @ x
                            up = matrices["up"][2] @ x
                            activation = (gate / (1.0 + np.exp(-np.clip(gate, -80, 80)))) * up
                        else:
                            activation = x
                        common = common_metadata(
                            args.source, data, int(record), int(router_rank), layer, expert,
                            strata[expert], projection, split,
                        )
                        paths = selector_paths(
                            gram, activation, int(config["shortlist_size"]), hybrid_fraction=0.25,
                            promoted_only=args.mode == "full",
                        )
                        if args.mode == "pilot":
                            exact_path = next(row[1] for row in paths if row[0] == "exact_marginal_fixed_greedy")
                            backward_path = next(row[1] for row in paths if row[0] == "backward_elimination")
                            if split == "validation":
                                selected_hybrid_fraction, chosen_hybrid, hybrid_scores = validation_select_hybrid(
                                    exact_path, backward_path, gram, activation, rates,
                                    config["hybrid_forward_fractions"],
                                )
                                facts.setdefault("validation_hybrid_promotions", []).append({
                                    "capture_source": args.source,
                                    "layer": layer,
                                    "expert_id": expert,
                                    "projection": projection,
                                    "selected_forward_fraction": selected_hybrid_fraction,
                                    "scores": hybrid_scores,
                                })
                            else:
                                chosen_hybrid = hybrid_from_paths(
                                    exact_path, backward_path,
                                    round(selected_hybrid_fraction * len(exact_path)),
                                )
                            paths = [
                                row for row in paths
                                if row[0] != "validation_selected_forward_backward_hybrid"
                            ] + [(
                                "validation_selected_forward_backward_hybrid",
                                chosen_hybrid,
                                0.0,
                                2 * (int(gram.nbytes) + gram.shape[0] * gram.shape[0] * 4),
                                gram.shape[0],
                            )]
                        cached = rows_for_paths(
                            selector_rows, overlap_rows, common, gram, activation, paths, rates
                        )
                        if split == "test":
                            add_broad_audit(
                                broad_rows, common, gram, activation,
                                cached["diagonal"][0],
                                cached["exact_marginal_fixed_greedy"][0],
                                cached["backward_elimination"][0],
                            )
                        if args.mode == "pilot" and split == "validation" and not args.disable_low_rank:
                            local_combinations = combinations_to_run
                        elif args.mode == "pilot":
                            local_combinations = validation_selected
                        else:
                            local_combinations = promotions.get(projection, [])
                        low_rank_start = len(low_rank_output)
                        low_rank_rows(
                            low_rank_output, overlap_rows, common, gram, activation,
                            cached["diagonal"][0], cached["exact_marginal_fixed_greedy"][0],
                            factor_by_method, rates, local_combinations,
                            physical_layout, payload_bytes,
                        )
                        if (
                            args.mode == "pilot"
                            and split == "validation"
                            and not args.disable_low_rank
                            and low_rank_start < len(low_rank_output)
                        ):
                            frame = pd.DataFrame(low_rank_output[low_rank_start:])
                            operating = frame[
                                (frame["rank"] <= 64)
                                & (frame["physical_budget_bpw_projection"].isin([0.5, 0.75, 1.0]))
                                & frame[
                                    "fraction_exact_over_diagonal_gain_retained_at_matched_physical_budget"
                                ].notna()
                            ]
                            ranked = (
                                operating.groupby(["method", "rank", "metadata_encoding"], as_index=False)
                                .agg(
                                    retained=(
                                        "fraction_exact_over_diagonal_gain_retained_at_matched_physical_budget",
                                        "mean",
                                    ),
                                    recovery=("recovery_at_matched_physical_budget", "mean"),
                                    metadata_bytes=("metadata_bytes_projection", "first"),
                                )
                                .sort_values(
                                    ["retained", "recovery", "metadata_bytes", "rank"],
                                    ascending=[False, False, True, True],
                                )
                            )
                            # Promote the global validation winner plus the
                            # best candidate from each factor family. This is a
                            # bounded Pareto screen, never a test-based choice.
                            chosen_rows = [] if ranked.empty else [ranked.iloc[0]]
                            for method in LOW_RANK_METHODS:
                                local = ranked[ranked["method"] == method]
                                if len(local):
                                    chosen_rows.append(local.iloc[0])
                            seen = set()
                            validation_selected = []
                            for row in chosen_rows:
                                key = (str(row["method"]), int(row["rank"]), str(row["metadata_encoding"]))
                                if key not in seen:
                                    seen.add(key)
                                    validation_selected.append(key)
                            facts.setdefault("validation_promotions", []).append({
                                "capture_source": args.source,
                                "layer": layer,
                                "expert_id": expert,
                                "projection": projection,
                                "choices": [
                                    {"method": method, "rank": rank, "metadata_encoding": encoding}
                                    for method, rank, encoding in validation_selected
                                ],
                            })
                print(json.dumps({
                    "source": args.source, "mode": args.mode, "layer": layer,
                    "expert": expert, "projection": projection,
                    "broad_rows": len(broad_rows), "selector_rows": len(selector_rows),
                    "low_rank_rows": len(low_rank_output),
                }), flush=True)
                del gram, factor_by_method
                gc.collect()
                torch.cuda.empty_cache()
            atomic_parquet(args.output / "exact_broad_ranking_audit.parquet", broad_rows)
            atomic_parquet(args.output / "selector_refresh_frontier.parquet", selector_rows)
            atomic_parquet(args.output / "gram_spectrum.parquet", spectrum_rows)
            atomic_parquet(args.output / "low_rank_selector_frontier.parquet", low_rank_output)
            pd.DataFrame(overlap_rows).to_csv(args.output / "support_overlap_summary.csv", index=False)
            facts["last_completed"] = {"layer": layer, "expert": expert, "unix": time.time()}
            atomic_json(args.output / "run_facts.json", facts)
    facts["completed_unix"] = time.time()
    facts["wall_seconds"] = facts["completed_unix"] - started
    atomic_json(args.output / "run_facts.json", facts)


if __name__ == "__main__":
    main()
