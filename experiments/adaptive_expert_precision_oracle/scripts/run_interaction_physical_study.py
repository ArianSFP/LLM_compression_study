#!/usr/bin/env python3
"""Training-only physical layouts and exact sequential expert evaluation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(EXPERIMENT / "scripts")]

from oracle_study.interaction_allocator import (
    block_refresh_from_gram,
    build_base_gram,
    diagonal_from_gram,
    exact_marginal_fixed_greedy_from_gram,
    page_aware_fixed_greedy,
    separate_plane_pages,
)
from oracle_study.mxfp4_embed import audit_checkpoint, load_compressed_mxfp4_expert
from oracle_study.mxfp4_selective import (
    BitAction,
    coordinate_to_page,
    pairwise_coselection_layout,
    select_under_page_budget,
)
from oracle_study.page_allocator import hypergraph_layout
from run_interaction_projection_study import metric_action_features, plane_order
from run_mxfp4_hierarchy_dense import select_varied_experts
from run_mxfp4_selective_pages import occurrence, qenergy, silu, tree_from_record
from run_phase_a_remote import proxy_gradients


PROJECTIONS = ("gate", "up", "down")
MATRIX_WEIGHTS = 2048 * 512
EXPERT_WEIGHTS = 3 * MATRIX_WEIGHTS
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


def action_mask(order: Sequence[BitAction], count: int, coordinates: int) -> np.ndarray:
    mask = np.zeros(coordinates, np.uint8)
    for action in order[:count]:
        mask[action.coordinate] = 1
    return mask


def build_training_layouts(
    gram: np.ndarray,
    train_inputs: np.ndarray,
    payload_bytes: int,
) -> tuple[dict[str, list[np.ndarray]], dict[str, Any]]:
    """Fit four layout families from training activations and masks only."""
    n = gram.shape[0] // 2
    diagonal_masks = []
    exact_masks = []
    exact_multirate = []
    exact_orders = []
    for activation in np.asarray(train_inputs, np.float32):
        diagonal = diagonal_from_gram(gram, activation)
        exact = exact_marginal_fixed_greedy_from_gram(gram, activation)
        exact_orders.append(exact)
        diagonal_masks.append(action_mask(diagonal, n, n))
        exact_masks.append(action_mask(exact, n, n))
        for rate in (0.25, 0.5, 0.75, 1.0):
            exact_multirate.append(action_mask(exact, max(1, round(rate * n)), n))
    diagonal_incidence = np.asarray(diagonal_masks, np.uint8)
    exact_incidence = np.asarray(exact_masks, np.uint8)
    multirate_incidence = np.asarray(exact_multirate, np.uint8)
    per_page = max(PAGE_SIZE // payload_bytes, 1)
    current = pairwise_coselection_layout(diagonal_incidence, per_page)
    exact = pairwise_coselection_layout(exact_incidence, per_page)
    balanced = hypergraph_layout(multirate_incidence, per_page)
    norm = np.linalg.norm(np.asarray(train_inputs, np.float64), axis=1)
    threshold = float(np.median(norm))
    clusters = (norm > threshold).astype(np.int64)
    cluster_layouts = []
    for cluster in (0, 1):
        mask = clusters == cluster
        local = exact_incidence[mask] if np.any(mask) else exact_incidence
        cluster_layouts.append(pairwise_coselection_layout(local, per_page))
    return {
        "current_diagonal_trained_coselection": [current],
        "exact_greedy_training_masks": [exact],
        "balanced_hypergraph": [balanced],
        "activation_feature_support_cluster": cluster_layouts,
    }, {
        "training_invocations": len(train_inputs),
        "activation_norm_threshold": threshold,
        "atoms_per_page": per_page,
        "test_masks_used": False,
    }


def choose_layout(
    layouts: dict[str, list[np.ndarray]], method: str,
    activation: np.ndarray, facts: dict[str, Any],
) -> tuple[np.ndarray, int]:
    options = layouts[method]
    if len(options) == 1:
        return options[0], 0
    cluster = int(np.linalg.norm(np.asarray(activation, np.float64)) > facts["activation_norm_threshold"])
    return options[cluster], cluster


def snapshot_separate(
    order: Sequence[BitAction],
    deltas: np.ndarray,
    activation: np.ndarray,
    layout: np.ndarray,
    payload_bytes: int,
    rate: float,
) -> dict[str, Any]:
    byte_budget = int(math.floor(rate * MATRIX_WEIGHTS / 8 + 1e-9))
    actions, pages = select_under_page_budget(order, layout, payload_bytes, PAGE_SIZE, byte_budget)
    correction = np.zeros(deltas.shape[-1], np.float32)
    depth = np.zeros(deltas.shape[1], np.uint8)
    for action in actions:
        correction += deltas[action.stage - 1, action.coordinate] * float(activation[action.coordinate])
        depth[action.coordinate] = action.stage
    return {
        "correction": correction,
        "depth": depth,
        "actions": actions,
        "pages": pages,
        "logical_bytes": len(actions) * payload_bytes,
        "physical_bytes": len(pages) * PAGE_SIZE,
        "representation": "separate_q3_q4_planes",
    }


def paired_exact_order(gram: np.ndarray, activation: np.ndarray) -> list[BitAction]:
    n = gram.shape[0] // 2
    merged_gram = (
        gram[:n, :n] + gram[:n, n:] + gram[n:, :n] + gram[n:, n:]
    )
    ids, _ = plane_order(merged_gram, activation, True)
    return [BitAction(int(coordinate), 1, 0.0) for coordinate in ids]


def snapshot_paired(
    order: Sequence[BitAction],
    deltas: np.ndarray,
    activation: np.ndarray,
    layout: np.ndarray,
    payload_bytes: int,
    rate: float,
) -> dict[str, Any]:
    paired_payload = 2 * payload_bytes
    per_page = max(PAGE_SIZE // paired_payload, 1)
    inverse = np.empty_like(layout)
    inverse[layout] = np.arange(len(layout))
    maximum_pages = int(math.floor(rate * MATRIX_WEIGHTS / 8 + 1e-9)) // PAGE_SIZE
    pages = set()
    actions = []
    for action in order:
        page = int(inverse[action.coordinate]) // per_page
        proposed = pages | {page}
        if len(proposed) > maximum_pages:
            continue
        pages = proposed
        actions.append(action)
    correction = np.zeros(deltas.shape[-1], np.float32)
    depth = np.zeros(deltas.shape[1], np.uint8)
    merged = deltas[0] + deltas[1]
    for action in actions:
        correction += merged[action.coordinate] * float(activation[action.coordinate])
        depth[action.coordinate] = 2
    return {
        "correction": correction,
        "depth": depth,
        "actions": actions,
        "pages": pages,
        "logical_bytes": len(actions) * paired_payload,
        "physical_bytes": len(pages) * PAGE_SIZE,
        "representation": "paired_exact_q2_q4_packets",
    }


def local_damage(
    target: np.ndarray, correction: np.ndarray,
    projection: str, proxy: np.ndarray, beta: float,
) -> float:
    error = np.asarray(target - correction, np.float64)
    value = float(error @ error)
    if projection == "down":
        value += beta * float(np.sum((error @ np.asarray(proxy, np.float64)) ** 2))
    return value


@lru_cache(maxsize=8)
def binary_masks(width: int) -> np.ndarray:
    values = np.arange(1, 1 << width, dtype=np.uint16)[:, None]
    bits = np.arange(width, dtype=np.uint16)[None, :]
    return ((values >> bits) & 1).astype(bool)


def matrix_from_snapshot(
    parent: np.ndarray, deltas: np.ndarray, snapshot: dict[str, Any],
) -> np.ndarray:
    matrix = parent.copy()
    for coordinate, depth in enumerate(snapshot["depth"]):
        if depth >= 1:
            matrix[:, coordinate] += deltas[0, coordinate]
        if depth >= 2:
            matrix[:, coordinate] += deltas[1, coordinate]
    return matrix


def joint_gate_up_shortlist_pilot(
    matrices: dict[str, list[np.ndarray]],
    deltas: dict[str, np.ndarray],
    x: np.ndarray,
    gate_layout: np.ndarray,
    up_layout: np.ndarray,
    down_snapshot: dict[str, Any],
    proxy: np.ndarray,
    beta: float,
    metadata: dict[str, Any],
    *,
    global_refreshes: int = 16,
    pages_per_refresh: int = 4,
    shortlist_size: int = 32,
) -> dict[str, Any]:
    """First-order page shortlist followed by exact SwiGLU/qenergy rescoring.

    This is deliberately a bounded H0 pilot. Each refresh propagates all
    gate/up action effects through the current down reconstruction, shortlists
    pages by aggregate first-order gain, then sequentially chooses pages using
    exhaustive actual-SwiGLU masks (at most 2^8 per separate-plane page).
    """
    started = time.perf_counter()
    n = deltas["gate"].shape[1]
    gate_pages = separate_plane_pages(gate_layout, per_page=8)
    up_pages = separate_plane_pages(up_layout, per_page=8)
    pages = {
        **{("gate", page): ids for page, ids in gate_pages.items()},
        **{("up", page): ids for page, ids in up_pages.items()},
    }
    remaining = set(pages)
    paid: set[tuple[str, int]] = set()
    depths = {"gate": np.zeros(n, np.uint8), "up": np.zeros(n, np.uint8)}
    corrections = {"gate": np.zeros(512, np.float32), "up": np.zeros(512, np.float32)}
    chosen_actions: list[tuple[str, int]] = []
    gbase = matrices["gate"][0] @ x
    ubase = matrices["up"][0] @ x
    down = matrix_from_snapshot(matrices["down"][0], deltas["down"], down_snapshot)
    gate_ref = matrices["gate"][2] @ x
    up_ref = matrices["up"][2] @ x
    href = (gate_ref / (1.0 + np.exp(-np.clip(gate_ref, -80, 80)))) * up_ref
    yref = torch.from_numpy(matrices["down"][2] @ href).cuda()
    down_t = torch.from_numpy(down).cuda()
    proxy_t = torch.from_numpy(np.asarray(proxy, np.float32)).cuda()

    def current_values() -> tuple[np.ndarray, np.ndarray, torch.Tensor, float]:
        gate = gbase + corrections["gate"]
        up = ubase + corrections["up"]
        gate_t = torch.from_numpy(gate).cuda()
        up_t = torch.from_numpy(up).cuda()
        hidden = silu(gate_t) * up_t
        output = down_t @ hidden
        damage = float(qenergy(yref - output, proxy_t, beta))
        return gate, up, output, damage

    def eligible_ids(key: tuple[str, int]) -> list[int]:
        projection = key[0]
        return [
            int(action_id) for action_id in pages[key]
            if int(action_id) // n + 1 == int(depths[projection][int(action_id) % n]) + 1
        ]

    def exact_page_mask(
        key: tuple[str, int], gate: np.ndarray, up: np.ndarray, damage: float,
    ) -> tuple[float, tuple[int, ...]]:
        projection = key[0]
        ids = eligible_ids(key)
        if not ids:
            return 0.0, ()
        action_delta = np.stack([
            deltas[projection][action_id // n, action_id % n] * float(x[action_id % n])
            for action_id in ids
        ]).astype(np.float32)
        masks = binary_masks(len(ids))
        additions = torch.from_numpy(masks.astype(np.float32) @ action_delta).cuda()
        gate_t = torch.from_numpy(gate).cuda()
        up_t = torch.from_numpy(up).cuda()
        if projection == "gate":
            hidden = silu(gate_t[None, :] + additions) * up_t[None, :]
        else:
            hidden = silu(gate_t)[None, :] * (up_t[None, :] + additions)
        outputs = hidden @ down_t.T
        damages = qenergy(yref[None, :] - outputs, proxy_t, beta)
        position = int(torch.argmin(damages))
        gain = damage - float(damages[position])
        if gain <= 0.0:
            return 0.0, ()
        chosen = tuple(np.asarray(ids, np.int64)[masks[position]].tolist())
        return gain, chosen

    for _refresh in range(global_refreshes):
        gate, up, output, damage = current_values()
        residual = yref - output
        gate_sigmoid = 1.0 / (1.0 + np.exp(-np.clip(gate, -80, 80)))
        silu_prime = gate_sigmoid * (1.0 + gate * (1.0 - gate_sigmoid))
        derivative = {"gate": up * silu_prime, "up": gate * gate_sigmoid}
        approximate: list[tuple[float, tuple[str, int]]] = []
        for projection in ("gate", "up"):
            expanded_x = np.concatenate((x, x)).astype(np.float32)
            flat = deltas[projection].reshape(2 * n, 512) * expanded_x[:, None]
            hidden_effect = torch.from_numpy(flat * derivative[projection][None, :]).cuda()
            output_effect = hidden_effect @ down_t.T
            action_gain = (
                qenergy(residual[None, :], proxy_t, beta)
                - qenergy(residual[None, :] - output_effect, proxy_t, beta)
            ).cpu().numpy()
            for key in (value for value in remaining if value[0] == projection):
                ids = eligible_ids(key)
                if ids:
                    approximate.append((float(np.maximum(action_gain[ids], 0.0).sum()), key))
        shortlist = [key for _, key in sorted(approximate, key=lambda row: (-row[0], row[1]))[:shortlist_size]]
        if not shortlist:
            break
        selected_this_refresh = 0
        while selected_this_refresh < pages_per_refresh and shortlist:
            gate, up, _output, damage = current_values()
            rescored = []
            for key in shortlist:
                gain, mask = exact_page_mask(key, gate, up, damage)
                rescored.append((gain, key, mask))
            gain, key, mask = max(rescored, key=lambda row: (row[0], tuple(-v if isinstance(v, int) else v for v in row[1])))
            if gain <= 0.0 or not mask:
                break
            projection = key[0]
            for action_id in sorted(mask, key=lambda value: value // n):
                coordinate = action_id % n
                stage = action_id // n
                corrections[projection] += deltas[projection][stage, coordinate] * float(x[coordinate])
                depths[projection][coordinate] = stage + 1
                chosen_actions.append((projection, action_id))
            paid.add(key)
            remaining.remove(key)
            shortlist.remove(key)
            selected_this_refresh += 1
    _gate, _up, _output, final_damage = current_values()
    ybase = torch.from_numpy(
        matrices["down"][0] @ ((gbase / (1.0 + np.exp(-np.clip(gbase, -80, 80)))) * ubase)
    ).cuda()
    base_damage = float(qenergy(yref - ybase, proxy_t, beta))
    physical_bytes = len(paid) * PAGE_SIZE + int(down_snapshot["physical_bytes"])
    logical_bytes = len(chosen_actions) * 64 + int(down_snapshot["logical_bytes"])
    return {
        **metadata,
        "objective": "joint_gate_up_first_order_shortlist_exact_swiglu_rescore",
        "budget_bpw": 8 * physical_bytes / EXPERT_WEIGHTS,
        "recovery": 1.0 - final_damage / max(base_damage, 1e-30),
        "damage": final_damage,
        "base_damage": base_damage,
        "physical_bytes": physical_bytes,
        "logical_bytes": logical_bytes,
        "physical_bpw": 8 * physical_bytes / EXPERT_WEIGHTS,
        "logical_bpw": 8 * logical_bytes / EXPERT_WEIGHTS,
        "page_amplification": physical_bytes / max(logical_bytes, 1),
        "physical_page_count": physical_bytes // PAGE_SIZE,
        "global_refreshes": global_refreshes,
        "shortlist_size": shortlist_size,
        "pages_per_refresh": pages_per_refresh,
        "joint_gate_up_pages": len(paid),
        "joint_gate_up_actions": len(chosen_actions),
        "selector_runtime_seconds": time.perf_counter() - started,
    }


def evaluate_complete_expert(
    matrices: dict[str, list[np.ndarray]],
    deltas: dict[str, np.ndarray],
    snapshots: dict[str, list[dict[str, Any]]],
    x: np.ndarray,
    proxy: np.ndarray,
    beta: float,
    budgets: Sequence[float],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    gbase = matrices["gate"][0] @ x
    ubase = matrices["up"][0] @ x
    gate = torch.from_numpy(np.stack([gbase + row["correction"] for row in snapshots["gate"]])).cuda()
    up = torch.from_numpy(np.stack([ubase + row["correction"] for row in snapshots["up"]])).cuda()
    hidden = (silu(gate[:, None, :]) * up[None, :, :]).reshape(-1, 512)
    down_matrices = []
    for row in snapshots["down"]:
        matrix = matrices["down"][0].copy()
        for coordinate, depth in enumerate(row["depth"]):
            if depth >= 1:
                matrix[:, coordinate] += deltas["down"][0, coordinate]
            if depth >= 2:
                matrix[:, coordinate] += deltas["down"][1, coordinate]
        down_matrices.append(matrix)
    down = torch.from_numpy(np.stack(down_matrices)).cuda()
    outputs = torch.einsum("ai,koi->ako", hidden, down).reshape(
        len(snapshots["gate"]), len(snapshots["up"]), len(snapshots["down"]), 2048
    )
    gate_ref = matrices["gate"][2] @ x
    up_ref = matrices["up"][2] @ x
    hidden_ref = (gate_ref / (1.0 + np.exp(-np.clip(gate_ref, -80, 80)))) * up_ref
    yref = torch.from_numpy(matrices["down"][2] @ hidden_ref).cuda()
    ybase = torch.from_numpy(
        matrices["down"][0] @ ((gbase / (1.0 + np.exp(-np.clip(gbase, -80, 80)))) * ubase)
    ).cuda()
    proxy_t = torch.from_numpy(np.asarray(proxy, np.float32)).cuda()
    base_damage = float(qenergy(yref - ybase, proxy_t, beta))
    damage = qenergy(yref[None, None, None] - outputs, proxy_t, beta).cpu().numpy()
    rows = []
    for budget in budgets:
        candidates = []
        for ig, iu, idown in np.ndindex(damage.shape):
            chosen = (snapshots["gate"][ig], snapshots["up"][iu], snapshots["down"][idown])
            physical = sum(row["physical_bytes"] for row in chosen)
            if 8 * physical / EXPERT_WEIGHTS <= float(budget) + 1e-12:
                candidates.append((float(damage[ig, iu, idown]), ig, iu, idown, physical))
        value, ig, iu, idown, physical = min(candidates)
        chosen = (snapshots["gate"][ig], snapshots["up"][iu], snapshots["down"][idown])
        logical = sum(row["logical_bytes"] for row in chosen)
        rows.append({
            **metadata,
            "objective": "exact_sequential_complete_expert_9x9x9",
            "budget_bpw": float(budget),
            "recovery": 1.0 - value / max(base_damage, 1e-30),
            "damage": value,
            "base_damage": base_damage,
            "physical_bytes": physical,
            "logical_bytes": logical,
            "physical_bpw": 8 * physical / EXPERT_WEIGHTS,
            "logical_bpw": 8 * logical / EXPERT_WEIGHTS,
            "page_amplification": physical / max(logical, 1),
            "physical_page_count": physical // PAGE_SIZE,
            "gate_rate_index": ig,
            "up_rate_index": iu,
            "down_rate_index": idown,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trees", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", choices=("exact_checkpoint", "cross_reference"), required=True)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    audit = audit_checkpoint(args.checkpoint, config["layers"])
    if not audit["passed"]:
        raise RuntimeError(audit)
    for name, path in (
        ("config_sha256", args.checkpoint / "config.json"),
        ("index_sha256", args.checkpoint / "model.safetensors.index.json"),
    ):
        actual = sha256(path)
        if actual != config["reference"][name]:
            raise RuntimeError(f"checkpoint {name} changed: {actual}")
    trees_json = json.loads(args.trees.read_text())
    trees = {p: tree_from_record(trees_json[p]) for p in PROJECTIONS}
    loaded = np.load(args.captures, allow_pickle=False)
    all_data = {name: loaded[name] for name in loaded.files}
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    minimum = 1 if args.source == "exact_checkpoint" else 3
    rates = list(map(float, config["projection_rates_bpw"]))
    budgets = list(map(float, config["physical_budgets_bpw"]))
    page_rows: list[dict[str, Any]] = []
    packet_rows: list[dict[str, Any]] = []
    layout_facts: list[dict[str, Any]] = []
    started = time.time()
    for layer in config["layers"]:
        mask = all_data["layer"] == layer
        data = {name: value[mask] for name, value in all_data.items()}
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
                p: load_compressed_mxfp4_expert(args.checkpoint, index, layer, expert, p)
                for p in PROJECTIONS
            }
            matrices = {
                p: [trees[p].decode(tensors[p], level) for level in (2, 3, 4)]
                for p in PROJECTIONS
            }
            deltas = {
                p: np.stack(((matrices[p][1] - matrices[p][0]).T, (matrices[p][2] - matrices[p][1]).T))
                for p in PROJECTIONS
            }
            train_records, _ = occurrence(
                data, expert, "train", int(config["layout_training_invocations_per_expert"])
            )
            train_x = np.asarray(data["x"][train_records], np.float32)
            train_gate = torch.from_numpy(train_x).cuda() @ torch.from_numpy(matrices["gate"][2]).cuda().T
            train_up = torch.from_numpy(train_x).cuda() @ torch.from_numpy(matrices["up"][2]).cuda().T
            train_h = (silu(train_gate) * train_up).cpu().numpy()
            train_inputs = {"gate": train_x, "up": train_x, "down": train_h}
            grams = {}
            layouts = {}
            layout_meta = {}
            for projection in PROJECTIONS:
                features = metric_action_features(deltas[projection], projection, proxy, beta)
                grams[projection] = build_base_gram(
                    features.reshape(2, deltas[projection].shape[1], -1), device="cuda"
                )
                payload = deltas[projection].shape[-1] // 8
                layouts[projection], layout_meta[projection] = build_training_layouts(
                    grams[projection], train_inputs[projection], payload
                )
                for method, options in layouts[projection].items():
                    layout_facts.append({
                        "capture_source": args.source, "layer": layer, "expert_id": expert,
                        "projection": projection, "layout_method": method,
                        "layout_replicas": len(options), **layout_meta[projection],
                    })
            test_records, test_ranks = occurrence(
                data, expert, "test",
                1 if args.mode == "pilot" else int(config["max_test_invocations_per_expert"]),
            )
            for record, router_rank in zip(test_records, test_ranks):
                x = np.asarray(data["x"][record], np.float32)
                gate_ref = matrices["gate"][2] @ x
                up_ref = matrices["up"][2] @ x
                hidden_ref = (gate_ref / (1.0 + np.exp(-np.clip(gate_ref, -80, 80)))) * up_ref
                activations = {"gate": x, "up": x, "down": hidden_ref}
                common = {
                    "capture_source": args.source,
                    "request_id": str(data["request_id"][record]),
                    "sequence_id": str(data["sequence_id"][record]),
                    "position": int(data["position"][record]),
                    "layer": layer,
                    "expert_id": expert,
                    "expert_stratum": strata[expert],
                    "router_rank": int(router_rank) + 1,
                }
                # Orders depend on the activation and Gram, not on a physical
                # permutation. Compute them once and reprice the same paths
                # under every training-only layout.
                projection_orders: dict[str, dict[str, Sequence[BitAction]]] = {}
                for projection in PROJECTIONS:
                    activation = activations[projection]
                    diagonal = diagonal_from_gram(grams[projection], activation)
                    exact = exact_marginal_fixed_greedy_from_gram(grams[projection], activation)
                    block16 = block_refresh_from_gram(
                        grams[projection], activation, 16, shortlist_size=128
                    )
                    if projection == "down":
                        # The broad audit did not reproduce a substantial down
                        # gain, so controlled complete-expert paths retain the
                        # PR #4 diagonal down allocator.
                        exact = diagonal
                        block16 = diagonal
                    projection_orders[projection] = {
                        "diagonal": diagonal,
                        "exact": exact,
                        "block16": block16,
                        "paired": paired_exact_order(grams[projection], activation),
                    }
                for layout_method in layouts["gate"]:
                    separate_snapshots: dict[str, list[dict[str, Any]]] = {}
                    block16_snapshots: dict[str, list[dict[str, Any]]] = {}
                    diagonal_snapshots: dict[str, list[dict[str, Any]]] = {}
                    paired_snapshots: dict[str, list[dict[str, Any]]] = {}
                    hybrid_snapshots: dict[str, list[dict[str, Any]]] = {}
                    for projection in PROJECTIONS:
                        activation = activations[projection]
                        layout, replica = choose_layout(
                            layouts[projection], layout_method, activation, layout_meta[projection]
                        )
                        exact = projection_orders[projection]["exact"]
                        diagonal = projection_orders[projection]["diagonal"]
                        block16 = projection_orders[projection]["block16"]
                        paired = projection_orders[projection]["paired"]
                        payload = deltas[projection].shape[-1] // 8
                        separate_snapshots[projection] = [
                            snapshot_separate(exact, deltas[projection], activation, layout, payload, rate)
                            for rate in rates
                        ]
                        diagonal_snapshots[projection] = [
                            snapshot_separate(diagonal, deltas[projection], activation, layout, payload, rate)
                            for rate in rates
                        ]
                        block16_snapshots[projection] = [
                            snapshot_separate(block16, deltas[projection], activation, layout, payload, rate)
                            for rate in rates
                        ]
                        paired_snapshots[projection] = [
                            snapshot_paired(paired, deltas[projection], activation, layout, payload, rate)
                            for rate in rates
                        ]
                        target = (deltas[projection][0] + deltas[projection][1]).T @ activation
                        hybrid_snapshots[projection] = []
                        for separate, paired_value in zip(
                            separate_snapshots[projection], paired_snapshots[projection]
                        ):
                            sd = local_damage(target, separate["correction"], projection, proxy, beta)
                            pdmg = local_damage(target, paired_value["correction"], projection, proxy, beta)
                            chosen = dict(separate if sd <= pdmg else paired_value)
                            chosen["representation"] = "hybrid_h0_best_separate_or_paired"
                            hybrid_snapshots[projection].append(chosen)
                        if args.mode == "pilot" and layout_method in (
                            "current_diagonal_trained_coselection", "exact_greedy_training_masks"
                        ):
                            pages = separate_plane_pages(layout, per_page=max(PAGE_SIZE // payload, 1))
                            # The exhaustive oracle is bounded to a 0.25-bpw
                            # projection pilot; larger points use the exact
                            # fixed path with measured pages.
                            page_budget = int((0.25 * MATRIX_WEIGHTS / 8) // PAGE_SIZE)
                            page_started = time.perf_counter()
                            selected = page_aware_fixed_greedy(
                                grams[projection], activation, pages, page_budget
                            )
                            page_seconds = time.perf_counter() - page_started
                            order_snapshot = separate_snapshots[projection][1]
                            for selector_name, action_count, physical_pages in (
                                ("page_level_residual_aware_mask_oracle", len(selected.actions), len(selected.pages)),
                                ("exact_action_path_repriced_to_pages", len(order_snapshot["actions"]), len(order_snapshot["pages"])),
                            ):
                                page_rows.append({
                                    **common,
                                    "projection": projection,
                                    "layout_method": layout_method,
                                    "layout_replica": replica,
                                    "selector": selector_name,
                                    "budget_bpw_projection": 0.25,
                                    "logical_action_count": action_count,
                                    "physical_page_count": physical_pages,
                                    "physical_bytes": physical_pages * PAGE_SIZE,
                                    "page_amplification": (
                                        physical_pages * PAGE_SIZE / max(action_count * payload, 1)
                                    ),
                                    "selector_runtime_seconds": page_seconds if "oracle" in selector_name else 0.0,
                                    "test_masks_used_to_build_layout": False,
                                })
                            if projection in ("gate", "up"):
                                n = grams[projection].shape[0] // 2
                                merged_gram = (
                                    grams[projection][:n, :n]
                                    + grams[projection][:n, n:]
                                    + grams[projection][n:, :n]
                                    + grams[projection][n:, n:]
                                )
                                paired_full_gram = np.zeros((2 * n, 2 * n), np.float32)
                                paired_full_gram[:n, :n] = merged_gram
                                paired_pages = {
                                    page: tuple(layout[start:start + 4].tolist())
                                    for page, start in enumerate(range(0, n, 4))
                                }
                                paired_page_started = time.perf_counter()
                                paired_selected = page_aware_fixed_greedy(
                                    paired_full_gram, activation, paired_pages, page_budget
                                )
                                page_rows.append({
                                    **common,
                                    "projection": projection,
                                    "layout_method": layout_method,
                                    "layout_replica": replica,
                                    "selector": "paired_page_level_residual_aware_mask_oracle",
                                    "budget_bpw_projection": 0.25,
                                    "logical_action_count": len(paired_selected.actions),
                                    "physical_page_count": len(paired_selected.pages),
                                    "physical_bytes": len(paired_selected.pages) * PAGE_SIZE,
                                    "page_amplification": (
                                        len(paired_selected.pages) * PAGE_SIZE
                                        / max(len(paired_selected.actions) * 2 * payload, 1)
                                    ),
                                    "selector_runtime_seconds": time.perf_counter() - paired_page_started,
                                    "test_masks_used_to_build_layout": False,
                                })
                    if args.mode == "pilot" and layout_method == "exact_greedy_training_masks":
                        gate_layout, _ = choose_layout(
                            layouts["gate"], layout_method, activations["gate"], layout_meta["gate"]
                        )
                        up_layout, _ = choose_layout(
                            layouts["up"], layout_method, activations["up"], layout_meta["up"]
                        )
                        down_index = min(range(len(rates)), key=lambda index: abs(rates[index] - 1.0))
                        packet_rows.append(joint_gate_up_shortlist_pilot(
                            matrices, deltas, x, gate_layout, up_layout,
                            diagonal_snapshots["down"][down_index], proxy, beta, {
                                **common,
                                "layout_method": layout_method,
                                "packet_representation": "separate_q3_q4_planes",
                                "storage_suffix_bpw": 2.0,
                                "external_storage_multiplier": (4.25 + 2.0) / 4.25,
                                "h0_or_deployable": "H0 first-order shortlist plus exact SwiGLU rescore",
                                "projection_selector": "joint_gate_up_first_order_exact_page_rescore_down_diagonal",
                            }
                        ))
                    for representation, snapshots in (
                        ("separate_q3_q4_planes", separate_snapshots),
                        ("paired_exact_q2_q4_packets", paired_snapshots),
                        ("hybrid_both_representations", hybrid_snapshots),
                    ):
                        metadata = {
                            **common,
                            "layout_method": layout_method,
                            "packet_representation": representation,
                            "storage_suffix_bpw": 2.0 if representation != "hybrid_both_representations" else 4.0,
                            "external_storage_multiplier": (
                                (4.25 + (2.0 if representation != "hybrid_both_representations" else 4.0)) / 4.25
                            ),
                            "h0_or_deployable": (
                                "H0 oracle representation choice" if representation == "hybrid_both_representations"
                                else "deployable fixed representation"
                            ),
                            "projection_selector": "gate_up_exact_marginal_fixed_greedy_down_diagonal",
                        }
                        packet_rows.extend(evaluate_complete_expert(
                            matrices, deltas, snapshots, x, proxy, beta, budgets, metadata
                        ))
                    if layout_method == "current_diagonal_trained_coselection":
                        packet_rows.extend(evaluate_complete_expert(
                            matrices, deltas, diagonal_snapshots, x, proxy, beta, budgets, {
                                **common,
                                "layout_method": layout_method,
                                "packet_representation": "separate_q3_q4_planes",
                                "storage_suffix_bpw": 2.0,
                                "external_storage_multiplier": (4.25 + 2.0) / 4.25,
                                "h0_or_deployable": "deployable fixed representation",
                                "projection_selector": "diagonal",
                            }
                        ))
                        packet_rows.extend(evaluate_complete_expert(
                            matrices, deltas, block16_snapshots, x, proxy, beta, budgets, {
                                **common,
                                "layout_method": layout_method,
                                "packet_representation": "separate_q3_q4_planes",
                                "storage_suffix_bpw": 2.0,
                                "external_storage_multiplier": (4.25 + 2.0) / 4.25,
                                "h0_or_deployable": "H0 full-Gram refresh control",
                                "projection_selector": "gate_up_block_refresh_16_down_diagonal",
                            }
                        ))
                print(json.dumps({
                    "source": args.source, "mode": args.mode, "layer": layer,
                    "expert": expert, "request_id": common["request_id"],
                    "page_rows": len(page_rows), "packet_rows": len(packet_rows),
                }), flush=True)
            atomic_parquet(args.output / "action_to_page_frontier.parquet", page_rows)
            atomic_parquet(args.output / "paired_hybrid_frontier.parquet", packet_rows)
            pd.DataFrame(layout_facts).to_csv(args.output / "layout_training_facts.csv", index=False)
            gc.collect()
            torch.cuda.empty_cache()
    accounting = {
        "page_size_bytes": PAGE_SIZE,
        "reference_bpw": 4.25,
        "resident_parent_bpw": 2.25,
        "separate_suffix_bpw": 2.0,
        "paired_suffix_bpw": 2.0,
        "hybrid_suffix_bpw": 4.0,
        "external_storage_multipliers": {
            "reference_plus_separate": (4.25 + 2.0) / 4.25,
            "reference_plus_paired": (4.25 + 2.0) / 4.25,
            "reference_plus_hybrid": (4.25 + 4.0) / 4.25,
        },
        "all_below_5x": True,
        "test_masks_used_for_layouts": False,
        "started_unix": started,
        "completed_unix": time.time(),
    }
    atomic_json(args.output / "selector_compute_storage_accounting.json", accounting)


if __name__ == "__main__":
    main()
