#!/usr/bin/env python3
"""Phases B/C: selective embedded-MXFP4 bits and physical page ceilings."""

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
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECTIONS = ("gate", "up", "down")
MATRIX_WEIGHTS = 2048 * 512
EXPERT_WEIGHTS = 3 * MATRIX_WEIGHTS


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    temporary.replace(path)


def silu(value: torch.Tensor) -> torch.Tensor:
    return value * torch.sigmoid(value)


def qenergy(error: torch.Tensor, proxy: torch.Tensor, beta: float) -> torch.Tensor:
    return torch.sum(error * error, dim=-1) + beta * torch.sum((error @ proxy) ** 2, dim=-1)


def tree_from_record(record: dict[str, Any]) -> Any:
    from oracle_study.mxfp4_embed import EmbeddedTree
    return EmbeddedTree(
        np.asarray(record["leaf_to_parent"], np.uint8), np.asarray(record["leaf_to_r1"], np.uint8),
        np.asarray(record["leaf_to_r2"], np.uint8), np.asarray(record["c2"], np.float32),
        np.asarray(record["c3"], np.float32), str(record["name"]), str(record["centroid_mode"]),
    )


def occurrence(data: dict[str, np.ndarray], expert: int, split: str, maximum: int) -> tuple[np.ndarray, np.ndarray]:
    found = np.argwhere((data["expert_ids"] == expert) & (data["split"][:, None] == split))[:maximum]
    if not len(found):
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return found[:, 0], found[:, 1]


def projection_norms(deltas: list[np.ndarray], proxy: np.ndarray | None, beta: float) -> list[np.ndarray]:
    result = []
    for delta in deltas:
        norm = np.sum(delta.astype(np.float64) ** 2, axis=0)
        if proxy is not None:
            norm += beta * np.sum((delta.T.astype(np.float64) @ proxy.astype(np.float64)) ** 2, axis=1)
        result.append(norm)
    return result


def build_layouts(
    deltas: list[np.ndarray], train_inputs: np.ndarray, payload_bytes: int, page_size: int,
    proxy: np.ndarray | None, beta: float,
) -> dict[str, list[np.ndarray]]:
    from oracle_study.mxfp4_selective import pairwise_coselection_layout
    inputs = np.asarray(train_inputs, np.float32)
    norms = projection_norms(deltas, proxy, beta)
    score = np.stack([inputs.astype(np.float64) ** 2 * value[None, :] for value in norms], axis=1)
    top = max(1, inputs.shape[1] // 4)
    incidence_parts = []
    for stage in range(2):
        mask = np.zeros((len(inputs), inputs.shape[1]), np.uint8)
        ids = np.argpartition(score[:, stage], -top, axis=1)[:, -top:]
        np.put_along_axis(mask, ids, 1, axis=1)
        incidence_parts.append(mask)
    incidence = np.concatenate(incidence_parts, axis=0)
    mean_score = score.mean(axis=(0, 1))
    group_size = max(1, page_size // payload_bytes)
    result: dict[str, list[np.ndarray]] = {
        "naive_1rep": [np.arange(inputs.shape[1], dtype=np.int64)],
        "importance_1rep": [np.argsort(mean_score, kind="stable")[::-1].astype(np.int64)],
        "coselection_1rep": [pairwise_coselection_layout(incidence, group_size)],
    }
    activation_norm = np.linalg.norm(inputs.astype(np.float64), axis=1)
    for replicas in (2, 4):
        boundaries = np.quantile(activation_norm, np.linspace(0, 1, replicas + 1))
        layouts = []
        for index in range(replicas):
            mask = (activation_norm >= boundaries[index]) & (activation_norm <= boundaries[index + 1] if index + 1 == replicas else activation_norm < boundaries[index + 1])
            local = np.concatenate([part[mask] for part in incidence_parts], axis=0)
            if not len(local):
                local = incidence
            layouts.append(pairwise_coselection_layout(local, group_size))
        result[f"coselection_h0_{replicas}rep"] = layouts
    return result


def selected_snapshot(
    deltas: list[np.ndarray], inputs: np.ndarray, layout_options: list[np.ndarray], payload_bytes: int,
    page_size: int, rate: float, proxy: np.ndarray | None, beta: float, paired: bool = False,
) -> dict[str, Any]:
    from oracle_study.mxfp4_selective import BitAction, diagonal_nested_order, select_under_page_budget
    vector = np.asarray(inputs, np.float32)
    if paired:
        merged = deltas[0] + deltas[1]
        contributions = merged * vector[None, :]
        order = [BitAction(int(j), 1, float(value)) for j, value in enumerate(np.sum(contributions.astype(np.float64) ** 2, axis=0))]
        order.sort(key=lambda action: action.score, reverse=True)
        effective_deltas = [merged]
        effective_payload = payload_bytes * 2
    else:
        norms = projection_norms(deltas, proxy, beta)
        order = diagonal_nested_order([vector.astype(np.float64) ** 2 * value for value in norms])
        effective_deltas = deltas
        effective_payload = payload_bytes
    byte_budget = int(math.floor(rate * MATRIX_WEIGHTS / 8 + 1e-9))
    target = (deltas[0] + deltas[1]) @ vector
    best: dict[str, Any] | None = None
    for replica, layout in enumerate(layout_options):
        actions, pages = select_under_page_budget(order, layout, effective_payload, page_size, byte_budget)
        correction = np.zeros(deltas[0].shape[0], np.float32)
        depth = np.zeros(deltas[0].shape[1], np.uint8)
        for action in actions:
            correction += effective_deltas[action.stage - 1][:, action.coordinate] * vector[action.coordinate]
            depth[action.coordinate] = 2 if paired else action.stage
        error = target - correction
        damage = float(np.sum(error.astype(np.float64) ** 2))
        if proxy is not None:
            damage += beta * float(np.sum((error.astype(np.float64) @ proxy.astype(np.float64)) ** 2))
        candidate = {
            "correction": correction, "depth": depth, "actions": actions, "pages": pages,
            "replica": replica, "layout": layout.copy(), "physical_bytes": len(pages) * page_size,
            "logical_bytes": len(actions) * effective_payload, "local_damage": damage,
        }
        if best is None or candidate["local_damage"] < best["local_damage"]:
            best = candidate
    assert best is not None
    return best


def support_audit(contributions: np.ndarray, output: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    from oracle_study.mxfp4_selective import backward_elimination_order, diagonal_nested_order, exact_marginal_order
    norms = [np.sum(value.astype(np.float64) ** 2, axis=1) for value in contributions]
    methods = {
        "diagonal": diagonal_nested_order(norms),
        "exact_marginal_omp": exact_marginal_order(contributions),
        "backward_elimination": backward_elimination_order(contributions),
    }
    target = contributions.sum(axis=(0, 1)).astype(np.float64)
    denominator = max(float(np.sum(target * target)), 1e-30)
    for method, order in methods.items():
        current = np.zeros_like(target)
        crossings = {threshold: None for threshold in (0.8, 0.9, 0.95, 0.99)}
        for index, action in enumerate(order, 1):
            current += contributions[action.stage - 1, action.coordinate]
            recovery = 1.0 - float(np.sum((target - current) ** 2)) / denominator
            for threshold in crossings:
                if crossings[threshold] is None and recovery >= threshold:
                    crossings[threshold] = index
        for threshold, count in crossings.items():
            output.append({**metadata, "ranking": method, "threshold": threshold, "actions_required": count,
                           "logical_bpw_projection": None if count is None else count / contributions.shape[1]})


def evaluate_invocation(
    state: dict[str, Any], expert: int, record: int, rank: int, trees: dict[str, Any], layouts: dict[str, dict[str, list[np.ndarray]]],
    layout_method: str, page_size: int, rates: list[float], budgets: list[float], proxy: np.ndarray, beta: float,
    paired: bool,
) -> tuple[list[dict[str, Any]], dict[float, dict[str, Any]]]:
    device = "cuda"
    data = state["data"]
    tensors = state["tensors"][expert]
    matrices = {p: [trees[p].decode(tensors[p], level) for level in (2, 3, 4)] for p in PROJECTIONS}
    deltas = {p: [matrices[p][1] - matrices[p][0], matrices[p][2] - matrices[p][1]] for p in PROJECTIONS}
    x = np.asarray(data["x"][record], np.float32)
    gref = matrices["gate"][2] @ x; uref = matrices["up"][2] @ x
    href = (gref / (1.0 + np.exp(-np.clip(gref, -80, 80)))) * uref
    input_values = {"gate": x, "up": x, "down": href}
    snapshots: dict[str, list[dict[str, Any]]] = {}
    for projection in PROJECTIONS:
        payload = matrices[projection][0].shape[0] // 8
        local_proxy = proxy if projection == "down" else None
        snapshots[projection] = [selected_snapshot(
            deltas[projection], input_values[projection], layouts[projection][layout_method], payload,
            page_size, rate, local_proxy, beta, paired,
        ) for rate in rates]
    gbase = matrices["gate"][0] @ x; ubase = matrices["up"][0] @ x
    gate = torch.from_numpy(np.stack([gbase + value["correction"] for value in snapshots["gate"]])).to(device)
    up = torch.from_numpy(np.stack([ubase + value["correction"] for value in snapshots["up"]])).to(device)
    h = silu(gate[:, None, :]) * up[None, :, :]
    hflat = h.reshape(len(rates) * len(rates), 512)
    down_stack = []
    for value in snapshots["down"]:
        matrix = matrices["down"][0].copy()
        for coordinate, depth in enumerate(value["depth"]):
            if depth >= 1:
                matrix[:, coordinate] += deltas["down"][0][:, coordinate]
            if depth >= 2:
                matrix[:, coordinate] += deltas["down"][1][:, coordinate]
        down_stack.append(matrix)
    down_t = torch.from_numpy(np.stack(down_stack)).to(device)
    outputs = torch.einsum("ai,koi->ako", hflat, down_t).reshape(len(rates), len(rates), len(rates), 2048)
    yref = torch.from_numpy(matrices["down"][2] @ href).to(device)
    ybase = torch.from_numpy(matrices["down"][0] @ ((gbase / (1.0 + np.exp(-np.clip(gbase, -80, 80)))) * ubase)).to(device)
    proxy_t = torch.from_numpy(proxy).to(device)
    base_damage = float(qenergy(yref - ybase, proxy_t, beta))
    damage = qenergy(yref[None, None, None, :] - outputs, proxy_t, beta).cpu().numpy()
    rows = []
    chosen: dict[float, dict[str, Any]] = {}
    for budget in budgets:
        candidates = []
        for ig, iu, idown in np.ndindex(damage.shape):
            physical = snapshots["gate"][ig]["physical_bytes"] + snapshots["up"][iu]["physical_bytes"] + snapshots["down"][idown]["physical_bytes"]
            if 8 * physical / EXPERT_WEIGHTS <= budget + 1e-12:
                candidates.append((float(damage[ig, iu, idown]), ig, iu, idown, physical))
        best = min(candidates)
        value, ig, iu, idown, physical = best
        selected = {"gate": snapshots["gate"][ig], "up": snapshots["up"][iu], "down": snapshots["down"][idown]}
        logical = sum(v["logical_bytes"] for v in selected.values())
        recovery = 1.0 - value / max(base_damage, 1e-30)
        row = {
            "request_id": str(data["request_id"][record]), "sequence_id": str(data["sequence_id"][record]),
            "position": int(data["position"][record]), "split": "test", "layer": state["layer"], "expert_id": expert,
            "expert_stratum": state["strata"][expert], "router_rank": rank + 1,
            "router_coefficient": float(data["router_weights"][record, rank]), "layout_method": layout_method,
            "page_size": page_size, "paired_planes": paired, "budget_bpw": budget,
            "gate_rate_target": rates[ig], "up_rate_target": rates[iu], "down_rate_target": rates[idown],
            "logical_bytes": logical, "physical_bytes": physical, "logical_bpw": 8 * logical / EXPERT_WEIGHTS,
            "physical_bpw": 8 * physical / EXPERT_WEIGHTS, "page_amplification": physical / max(logical, 1),
            "recovery": recovery, "damage": value, "base_damage": base_damage,
            "gate_actions": len(selected["gate"]["actions"]), "up_actions": len(selected["up"]["actions"]),
            "down_actions": len(selected["down"]["actions"]),
            "gate_replica": selected["gate"]["replica"], "up_replica": selected["up"]["replica"], "down_replica": selected["down"]["replica"],
        }
        rows.append(row)
        chosen[budget] = {"row": row, "selected": selected}
    del gate, up, h, hflat, down_t, outputs, proxy_t
    return rows, chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dense-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    experiment = args.config.resolve().parents[1]
    sys.path[:0] = [str(experiment / "src"), str(experiment / "scripts")]
    from oracle_study.mxfp4_embed import audit_checkpoint, load_compressed_mxfp4_expert
    from oracle_study.mxfp4_selective import coordinate_to_page
    from run_mxfp4_hierarchy_dense import select_varied_experts
    from run_phase_a_remote import proxy_gradients

    args.output.mkdir(parents=True, exist_ok=True)
    audit = audit_checkpoint(args.checkpoint, config["layers"])
    if not audit["passed"]:
        raise RuntimeError(audit)
    tree_records = json.loads((args.dense_output / "selected_trees.json").read_text())
    trees = {p: tree_from_record(tree_records[p]) for p in PROJECTIONS}
    loaded = np.load(args.captures, allow_pickle=False)
    all_data = {name: loaded[name] for name in loaded.files}
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    states = {}
    facts: dict[str, Any] = {
        "run_id": config["run_id"], "started_unix": time.time(), "gpu": torch.cuda.get_device_name(0),
        "host": platform.node(), "checkpoint_audit": audit, "capture_status": config["status"], "layers": {},
        "replica_policy": "H0 best-replica diagnostic; one-replica co-selection is the primary deployable layout",
    }
    for layer in config["layers"]:
        mask = all_data["layer"] == layer
        data = {name: value[mask] for name, value in all_data.items()}
        strata, counts = select_varied_experts(data, int(config["experts_per_stratum"]), int(config["minimum_split_occurrences"]))
        experts = sorted(strata)
        tensors = {e: {p: load_compressed_mxfp4_expert(args.checkpoint, index, layer, e, p) for p in PROJECTIONS} for e in experts}
        proxy, proxy_facts = proxy_gradients(data["xplus"], [data[f"h{h}_router_logits"] for h in range(1, 5)], data["split"])
        states[layer] = {"layer": layer, "data": data, "strata": strata, "experts": experts, "tensors": tensors, "proxy": proxy, "beta": float(proxy_facts["beta"])}
        facts["layers"][str(layer)] = {"experts": experts, "proxy": proxy_facts}
    rates = np.arange(0, 2.0 + 1e-9, float(config["selective_rate_step"])).tolist()
    budgets = list(map(float, config["physical_budgets_bpw"]))
    primary_rows: list[dict[str, Any]] = []
    layout_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    completed: set[tuple[int, int]] = set()
    primary_path = args.output / "selective_metrics.parquet"
    if primary_path.exists():
        old = pd.read_parquet(primary_path); primary_rows = old.to_dict("records")
        completed = set(zip(old["layer"].astype(int), old["expert_id"].astype(int)))
        for path, target in ((args.output / "layout_metrics.parquet", layout_rows), (args.output / "selected_bit_actions.parquet", action_rows), (args.output / "ranking_audit.parquet", support_rows)):
            if path.exists(): target.extend(pd.read_parquet(path).to_dict("records"))
    for layer, state in states.items():
        for expert in state["experts"]:
            if (layer, expert) in completed:
                continue
            tensors = state["tensors"][expert]
            matrices = {p: [trees[p].decode(tensors[p], level) for level in (2, 3, 4)] for p in PROJECTIONS}
            deltas = {p: [matrices[p][1] - matrices[p][0], matrices[p][2] - matrices[p][1]] for p in PROJECTIONS}
            train_records, _ = occurrence(state["data"], expert, "train", int(config["max_train_invocations_per_expert"]))
            train_x = np.asarray(state["data"]["x"][train_records], np.float32)
            gate = torch.from_numpy(train_x).cuda() @ torch.from_numpy(matrices["gate"][2]).cuda().T
            up = torch.from_numpy(train_x).cuda() @ torch.from_numpy(matrices["up"][2]).cuda().T
            train_h = (silu(gate) * up).cpu().numpy()
            layouts_by_page: dict[int, dict[str, dict[str, list[np.ndarray]]]] = {}
            for page_size in map(int, config["page_sizes"]):
                layouts_by_page[page_size] = {}
                for projection, inputs in (("gate", train_x), ("up", train_x), ("down", train_h)):
                    payload = matrices[projection][0].shape[0] // 8
                    layouts_by_page[page_size][projection] = build_layouts(deltas[projection], inputs, payload, page_size, state["proxy"] if projection == "down" else None, state["beta"])
            test_records, test_ranks = occurrence(state["data"], expert, "test", int(config["max_test_invocations_per_expert"]))
            for sample, (record, rank) in enumerate(zip(test_records, test_ranks)):
                rows, chosen = evaluate_invocation(state, expert, int(record), int(rank), trees, layouts_by_page[512], "coselection_1rep", 512, rates, budgets, state["proxy"], state["beta"], False)
                primary_rows.extend(rows)
                for budget, value in chosen.items():
                    for projection, selected in value["selected"].items():
                        actions = selected["actions"]
                        page_ids = [coordinate_to_page(selected["layout"], action.coordinate, action.stage, matrices[projection][0].shape[0] // 8, 512) for action in actions]
                        action_rows.append({
                            "request_id": value["row"]["request_id"], "sequence_id": value["row"]["sequence_id"],
                            "position": value["row"]["position"], "layer": layer, "expert_id": expert,
                            "projection": projection, "budget_bpw": budget, "selected_action_count": len(actions),
                            "coordinate_ids_json": json.dumps([action.coordinate for action in actions]),
                            "refinement_bits_json": json.dumps([action.stage for action in actions]),
                            "action_page_ids_json": json.dumps(page_ids),
                            "selected_page_ids_json": json.dumps(sorted(selected["pages"])),
                            "marginal_scores_json": json.dumps([action.score for action in actions]),
                            "replica_id": selected["replica"],
                        })
                if sample < 1:
                    variants = []
                    for page_size in map(int, config["page_sizes"]):
                        for method in ("naive_1rep", "importance_1rep", "coselection_1rep", "coselection_h0_2rep", "coselection_h0_4rep"):
                            variants.append((page_size, method, False))
                    variants.extend((page_size, "coselection_1rep", True) for page_size in map(int, config["page_sizes"]))
                    for page_size, method, paired in variants:
                        extra, _ = evaluate_invocation(state, expert, int(record), int(rank), trees, layouts_by_page[page_size], method, page_size, rates, budgets, state["proxy"], state["beta"], paired)
                        layout_rows.extend(extra)
                if sample == 0 and expert == state["experts"][-1]:
                    x = np.asarray(state["data"]["x"][record], np.float32)
                    gref = matrices["gate"][2] @ x; uref = matrices["up"][2] @ x
                    href = (gref / (1 + np.exp(-np.clip(gref, -80, 80)))) * uref
                    for projection, inputs in (("gate", x), ("up", x), ("down", href)):
                        contributions = np.stack([delta.T * inputs[:, None] for delta in deltas[projection]])
                        support_audit(contributions, support_rows, {"layer": layer, "expert_id": expert, "projection": projection, "request_id": str(state["data"]["request_id"][record])})
            atomic_parquet(args.output / "selective_metrics.parquet", primary_rows)
            atomic_parquet(args.output / "layout_metrics.parquet", layout_rows)
            atomic_parquet(args.output / "selected_bit_actions.parquet", action_rows)
            atomic_parquet(args.output / "ranking_audit.parquet", support_rows)
            facts["last_completed"] = {"layer": layer, "expert": expert, "unix": time.time()}
            atomic_json(args.output / "run_facts.json", facts)
            print(json.dumps({"completed_layer": layer, "expert": expert, "primary_rows": len(primary_rows), "layout_rows": len(layout_rows)}), flush=True)
            gc.collect(); torch.cuda.empty_cache()
    # Attach paired dense comparator and selective retention ratio eta.
    dense = pd.read_parquet(args.dense_output / "dense_metrics.parquet")
    primary = pd.DataFrame(primary_rows)
    keys = ["request_id", "position", "layer", "expert_id"]
    comparator_rows = []
    for _, row in primary.iterrows():
        matched = dense[(dense["split"] == "test") & (dense["request_id"] == row.request_id) & (dense["position"] == row.position) & (dense["layer"] == row.layer) & (dense["expert_id"] == row.expert_id) & (dense["physical_bpw"] <= row.budget_bpw + 1e-12)]
        if len(matched):
            best = matched.loc[matched.damage.idxmin()]
            denominator = float(best.base_damage - best.damage)
            eta = float("nan") if denominator <= 1e-20 else (float(row.base_damage) - float(row.damage)) / denominator
            comparator_rows.append({**{key: row[key] for key in keys}, "budget_bpw": row.budget_bpw, "selective_recovery": row.recovery,
                                    "dense_recovery": best.recovery, "dense_tuple": best.tuple, "retention_eta": eta})
    pd.DataFrame(comparator_rows).to_parquet(args.output / "selective_dense_retention.parquet", index=False)
    # Storage cap: exact reference fallback plus 1/2/4 copies of a two-bpw suffix.
    storage = []
    for replicas in (1, 2, 4):
        multiplier = (4.25 + 2.0 * replicas) / 4.25
        storage.append({"layout_replicas": replicas, "reference_fallback_bpw": 4.25, "suffix_bpw": 2.0 * replicas, "external_storage_multiplier": multiplier, "within_5x": multiplier <= 5.0})
    pd.DataFrame(storage).to_csv(args.output / "storage_accounting.csv", index=False)
    facts["completed_unix"] = time.time(); facts["wall_seconds"] = facts["completed_unix"] - facts["started_unix"]
    atomic_json(args.output / "run_facts.json", facts)


if __name__ == "__main__":
    main()
