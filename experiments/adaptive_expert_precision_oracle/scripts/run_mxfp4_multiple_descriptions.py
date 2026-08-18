#!/usr/bin/env python3
"""Evaluate A-first, bidirectional A/B, and parity A/B/Q MXFP4 suffixes.

This is a paired H0 codec/action experiment.  It reuses the exact packed MXFP4
endpoint and locked Q2 parent tree from the preceding hierarchy study.  New B
and parity Q3 centroids and page layouts are fit on training requests only.
Every saved action carries its explicit before/after state, description-plane
IDs, and exact charged page IDs.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping

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


def hierarchy_from_record(record: dict[str, Any]) -> Any:
    from oracle_study.mxfp4_descriptions import MultipleDescriptionTree
    base = tree_from_record(record["base"])
    return MultipleDescriptionTree(
        base, np.asarray(record["c3a"], np.float32), np.asarray(record["c3b"], np.float32),
        np.asarray(record["c3q"], np.float32),
    )


def occurrence(data: dict[str, np.ndarray], expert: int, split: str, maximum: int) -> tuple[np.ndarray, np.ndarray]:
    found = np.argwhere((data["expert_ids"] == expert) & (data["split"][:, None] == split))[:maximum]
    if not len(found):
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return found[:, 0], found[:, 1]


def augmented(contributions: np.ndarray, proxy: np.ndarray | None, beta: float) -> np.ndarray:
    values = np.asarray(contributions, np.float32)
    if proxy is None:
        return values
    projected = values.astype(np.float64) @ np.asarray(proxy, np.float64)
    return np.concatenate((values.astype(np.float64), math.sqrt(beta) * projected), axis=1).astype(np.float32)


def contribution_arrays(
    matrices: Mapping[str, np.ndarray], vector: np.ndarray, descriptions: tuple[str, ...],
    proxy: np.ndarray | None, beta: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    x = np.asarray(vector, np.float32)
    raw_target = (matrices["exact"] - matrices["base"]).T * x[:, None]
    raw_single = {name: (matrices[name] - matrices["base"]).T * x[:, None] for name in descriptions}
    return (
        augmented(raw_target, proxy, beta),
        {name: augmented(value, proxy, beta) for name, value in raw_single.items()},
        raw_target,
        raw_single,
    )


def fit_hierarchies(
    states: dict[int, dict[str, Any]], base_trees: dict[str, Any], config: dict[str, Any], output: Path,
) -> dict[str, Any]:
    """Fit B/Q centroids from the same training-only functional masses as Q2/A."""
    from oracle_study.mxfp4_descriptions import MultipleDescriptionTree, description_distortion
    from oracle_study.mxfp4_embed import combine_leaf_statistics, leaf_importance_statistics

    result = {}
    facts = []
    for projection in PROJECTIONS:
        masses = []
        for state in states.values():
            train = state["data"]["split"] == "train"
            train_x = np.asarray(state["data"]["x"][train], np.float32)
            moment_x = np.mean(train_x.astype(np.float64) ** 2, axis=0)
            for expert in state["experts"]:
                tensor = state["tensors"][expert][projection]
                if projection == "down":
                    rows, _ = occurrence(state["data"], expert, "train", int(config["max_train_invocations_per_expert"]))
                    x = torch.from_numpy(np.asarray(state["data"]["x"][rows], np.float32)).cuda()
                    gate = x @ torch.from_numpy(state["tensors"][expert]["gate"].dequantize()).cuda().T
                    up = x @ torch.from_numpy(state["tensors"][expert]["up"].dequantize()).cuda().T
                    moment = torch.mean((silu(gate) * up).double() ** 2, dim=0).cpu().numpy()
                else:
                    moment = moment_x
                masses.append(leaf_importance_statistics(tensor, moment[None, :]))
        total = combine_leaf_statistics(masses)
        hierarchy = MultipleDescriptionTree.fit(base_trees[projection], total)
        result[projection] = hierarchy
        for name in ("a", "b", "q"):
            facts.append({
                "projection": projection, "description": name,
                "train_functional_leaf_distortion": description_distortion(hierarchy, total, name),
                "centroid_mode": hierarchy.base.centroid_mode,
            })
    atomic_json(output / "description_trees.json", {name: value.as_dict() for name, value in result.items()})
    pd.DataFrame(facts).to_csv(output / "description_fit_summary.csv", index=False)
    return result


def build_layout(
    matrices: Mapping[str, np.ndarray], inputs: np.ndarray, descriptions: tuple[str, ...],
    payload_bytes: int, page_size: int, proxy: np.ndarray | None, beta: float,
) -> np.ndarray:
    """Training-only coordinate packing from all eligible first descriptions."""
    from oracle_study.mxfp4_selective import pairwise_coselection_layout
    masks = []
    top = max(1, inputs.shape[1] // 4)
    for vector in np.asarray(inputs, np.float32):
        target, singles, _, _ = contribution_arrays(matrices, vector, descriptions, proxy, beta)
        target_norm = np.sum(target.astype(np.float64) ** 2, axis=1)
        for name in descriptions:
            delta = singles[name].astype(np.float64)
            score = target_norm - np.sum((target.astype(np.float64) - delta) ** 2, axis=1)
            chosen = np.argpartition(score, -top)[-top:]
            mask = np.zeros(len(score), np.uint8); mask[chosen] = 1; masks.append(mask)
    incidence = np.stack(masks) if masks else np.eye(inputs.shape[1], dtype=np.uint8)
    return pairwise_coselection_layout(incidence, max(1, page_size // payload_bytes))


def selected_snapshot(
    matrices: Mapping[str, np.ndarray], vector: np.ndarray, layout: np.ndarray, payload_bytes: int,
    page_size: int, rate: float, proxy: np.ndarray | None, beta: float,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    from oracle_study.mxfp4_description_selective import (
        action_page_ids, reconstruct_selected_correction, select_description_actions,
    )
    descriptions = tuple(policy["descriptions"])
    target, singles, raw_target, raw_singles = contribution_arrays(matrices, vector, descriptions, proxy, beta)
    actions, pages, states = select_description_actions(
        target, singles, layout, payload_bytes, page_size,
        int(math.floor(rate * MATRIX_WEIGHTS / 8 + 1e-9)), descriptions,
        tuple(policy["first_descriptions"]), bool(policy["allow_direct_pairs"]),
    )
    correction = reconstruct_selected_correction(raw_target, raw_singles, states)
    error = np.sum(raw_target, axis=0) - correction
    damage = float(np.sum(error.astype(np.float64) ** 2))
    if proxy is not None:
        damage += beta * float(np.sum((error.astype(np.float64) @ proxy.astype(np.float64)) ** 2))
    page_lists = action_page_ids(actions, layout, payload_bytes, page_size, descriptions)
    return {
        "correction": correction, "states": states, "actions": actions, "action_pages": page_lists,
        "pages": pages, "layout": layout.copy(), "physical_bytes": len(pages) * page_size,
        "logical_bytes": sum(len(action.add_descriptions) for action in actions) * payload_bytes,
        "local_damage": damage,
    }


def correction_matrix(matrices: Mapping[str, np.ndarray], states: np.ndarray) -> np.ndarray:
    from oracle_study.mxfp4_description_selective import is_exact_mask, mask_descriptions
    output = np.asarray(matrices["base"], np.float32).copy()
    for coordinate, mask in enumerate(np.asarray(states, np.uint8)):
        if is_exact_mask(int(mask)):
            output[:, coordinate] = matrices["exact"][:, coordinate]
        elif mask:
            output[:, coordinate] = matrices[mask_descriptions(int(mask))[0]][:, coordinate]
    return output


def evaluate_invocation(
    state: dict[str, Any], expert: int, record: int, rank: int,
    matrices: dict[str, dict[str, np.ndarray]], layouts: dict[str, np.ndarray],
    policy_name: str, policy: Mapping[str, Any], page_size: int,
    rates: list[float], budgets: list[float],
) -> tuple[list[dict[str, Any]], dict[float, dict[str, Any]]]:
    data = state["data"]
    x = np.asarray(data["x"][record], np.float32)
    gref = matrices["gate"]["exact"] @ x; uref = matrices["up"]["exact"] @ x
    href = (gref / (1 + np.exp(-np.clip(gref, -80, 80)))) * uref
    vectors = {"gate": x, "up": x, "down": href}
    snapshots = {}
    for projection in PROJECTIONS:
        proxy = state["proxy"] if projection == "down" else None
        payload = matrices[projection]["base"].shape[0] // 8
        snapshots[projection] = [selected_snapshot(
            matrices[projection], vectors[projection], layouts[projection], payload,
            page_size, rate, proxy, state["beta"], policy,
        ) for rate in rates]
    device = "cuda"
    gbase = matrices["gate"]["base"] @ x; ubase = matrices["up"]["base"] @ x
    gate = torch.from_numpy(np.stack([gbase + value["correction"] for value in snapshots["gate"]])).to(device)
    up = torch.from_numpy(np.stack([ubase + value["correction"] for value in snapshots["up"]])).to(device)
    h = silu(gate[:, None, :]) * up[None, :, :]
    hflat = h.reshape(len(rates) ** 2, 512)
    down = torch.from_numpy(np.stack([
        correction_matrix(matrices["down"], value["states"]) for value in snapshots["down"]
    ])).to(device)
    outputs = torch.einsum("ai,koi->ako", hflat, down).reshape(len(rates), len(rates), len(rates), 2048)
    yref = torch.from_numpy(matrices["down"]["exact"] @ href).to(device)
    hbase = (gbase / (1 + np.exp(-np.clip(gbase, -80, 80)))) * ubase
    ybase = torch.from_numpy(matrices["down"]["base"] @ hbase).to(device)
    proxy_t = torch.from_numpy(state["proxy"]).to(device)
    base_damage = float(qenergy(yref - ybase, proxy_t, state["beta"]))
    damage = qenergy(yref[None, None, None] - outputs, proxy_t, state["beta"]).cpu().numpy()
    rows = []
    chosen = {}
    for budget in budgets:
        candidates = []
        for ig, iu, idown in np.ndindex(damage.shape):
            physical = sum((snapshots["gate"][ig]["physical_bytes"], snapshots["up"][iu]["physical_bytes"], snapshots["down"][idown]["physical_bytes"]))
            if 8 * physical / EXPERT_WEIGHTS <= budget + 1e-12:
                candidates.append((float(damage[ig, iu, idown]), ig, iu, idown, physical))
        value, ig, iu, idown, physical = min(candidates)
        selected = {"gate": snapshots["gate"][ig], "up": snapshots["up"][iu], "down": snapshots["down"][idown]}
        logical = sum(item["logical_bytes"] for item in selected.values())
        row = {
            "request_id": str(data["request_id"][record]), "sequence_id": str(data["sequence_id"][record]),
            "position": int(data["position"][record]), "split": "test", "layer": state["layer"],
            "expert_id": expert, "expert_stratum": state["strata"][expert], "router_rank": rank + 1,
            "router_coefficient": float(data["router_weights"][record, rank]), "policy": policy_name,
            "page_size": page_size, "budget_bpw": budget, "logical_bytes": logical,
            "physical_bytes": physical, "logical_bpw": 8 * logical / EXPERT_WEIGHTS,
            "physical_bpw": 8 * physical / EXPERT_WEIGHTS,
            "page_amplification": physical / max(logical, 1), "damage": value, "base_damage": base_damage,
            "recovery": 1.0 - value / max(base_damage, 1e-30),
            "gate_rate_target": rates[ig], "up_rate_target": rates[iu], "down_rate_target": rates[idown],
        }
        for projection in PROJECTIONS:
            item = selected[projection]
            row[f"{projection}_actions"] = sum(len(action.add_descriptions) for action in item["actions"])
            row[f"{projection}_pages"] = len(item["pages"])
        rows.append(row); chosen[budget] = {"row": row, "selected": selected}
    del gate, up, h, hflat, down, outputs, proxy_t
    return rows, chosen


def full_support_rows(
    state: dict[str, Any], expert: int, records: np.ndarray, matrices: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    rows = []
    data = state["data"]
    for record in records:
        x = np.asarray(data["x"][record], np.float32)
        gref = matrices["gate"]["exact"] @ x; uref = matrices["up"]["exact"] @ x
        href = (gref / (1 + np.exp(-np.clip(gref, -80, 80)))) * uref
        for projection, vector in (("gate", x), ("up", x), ("down", href)):
            target = matrices[projection]["exact"] @ vector
            base_error = target - matrices[projection]["base"] @ vector
            denominator = float(np.sum(base_error.astype(np.float64) ** 2))
            for name in ("a", "b", "q"):
                error = target - matrices[projection][name] @ vector
                rows.append({
                    "request_id": str(data["request_id"][record]), "position": int(data["position"][record]),
                    "layer": state["layer"], "expert_id": expert, "projection": projection,
                    "description": name, "recovery": 1 - float(np.sum(error.astype(np.float64) ** 2)) / max(denominator, 1e-30),
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dense-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--description-trees", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    experiment = args.config.resolve().parents[1]
    sys.path[:0] = [str(experiment / "src"), str(experiment / "scripts")]
    from oracle_study.mxfp4_embed import audit_checkpoint, load_compressed_mxfp4_expert
    from run_mxfp4_hierarchy_dense import select_varied_experts
    from run_phase_a_remote import proxy_gradients

    args.output.mkdir(parents=True, exist_ok=True)
    audit = audit_checkpoint(args.checkpoint, config["layers"])
    if not audit["passed"]:
        raise RuntimeError(audit)
    locked = json.loads((args.dense_output / "selected_trees.json").read_text())
    base_trees = {name: tree_from_record(value) for name, value in locked.items()}
    loaded = np.load(args.captures, allow_pickle=False)
    all_data = {name: loaded[name] for name in loaded.files}
    split_requests = {split: set(map(str, all_data["request_id"][all_data["split"] == split])) for split in ("train", "validation", "test")}
    if any(split_requests[a] & split_requests[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise RuntimeError("request split leakage")
    weight_map = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    states = {}
    facts = {
        "run_id": config["run_id"], "started_unix": time.time(), "host": platform.node(),
        "gpu": torch.cuda.get_device_name(0), "checkpoint_audit": audit, "capture_status": config["status"],
        "split_requests": {key: len(value) for key, value in split_requests.items()}, "layers": {},
    }
    for layer in config["layers"]:
        mask = all_data["layer"] == layer
        data = {name: value[mask] for name, value in all_data.items()}
        strata, counts = select_varied_experts(data, int(config["experts_per_stratum"]), int(config["minimum_split_occurrences"]))
        experts = sorted(strata)
        tensors = {expert: {projection: load_compressed_mxfp4_expert(args.checkpoint, weight_map, layer, expert, projection) for projection in PROJECTIONS} for expert in experts}
        proxy, proxy_facts = proxy_gradients(data["xplus"], [data[f"h{h}_router_logits"] for h in range(1, 5)], data["split"])
        states[layer] = {"layer": layer, "data": data, "strata": strata, "counts": counts, "experts": experts, "tensors": tensors, "proxy": proxy, "beta": float(proxy_facts["beta"])}
        facts["layers"][str(layer)] = {"experts": experts, "counts": {str(k): v for k, v in counts.items()}, "proxy": proxy_facts}
    if args.description_trees:
        records = json.loads(args.description_trees.read_text())
        hierarchies = {name: hierarchy_from_record(value) for name, value in records.items()}
        atomic_json(args.output / "description_trees.json", records)
        facts["description_fit"] = f"locked from {args.description_trees}"
    else:
        hierarchies = fit_hierarchies(states, base_trees, config, args.output)
        facts["description_fit"] = "training-only functional leaf masses"
    rates = np.arange(0, 2.0 + 1e-9, float(config["selective_rate_step"])).tolist()
    budgets = list(map(float, config["physical_budgets_bpw"]))
    metric_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    page_rows: list[dict[str, Any]] = []
    for layer, state in states.items():
        for expert in state["experts"]:
            matrices = {projection: hierarchies[projection].matrices(state["tensors"][expert][projection], True) for projection in PROJECTIONS}
            train_records, _ = occurrence(state["data"], expert, "train", int(config["max_train_invocations_per_expert"]))
            train_x = np.asarray(state["data"]["x"][train_records], np.float32)
            train_gate = torch.from_numpy(train_x).cuda() @ torch.from_numpy(matrices["gate"]["exact"]).cuda().T
            train_up = torch.from_numpy(train_x).cuda() @ torch.from_numpy(matrices["up"]["exact"]).cuda().T
            train_h = (silu(train_gate) * train_up).cpu().numpy()
            test_records, test_ranks = occurrence(state["data"], expert, "test", int(config["max_test_invocations_per_expert"]))
            full_rows.extend(full_support_rows(state, expert, test_records, matrices))
            for policy_name, policy in config["policies"].items():
                descriptions = tuple(policy["descriptions"])
                layout_descriptions = tuple(policy.get("layout_descriptions", descriptions))
                layouts = {}
                for projection, inputs in (("gate", train_x), ("up", train_x), ("down", train_h)):
                    payload = matrices[projection]["base"].shape[0] // 8
                    layouts[projection] = build_layout(
                        matrices[projection], inputs, layout_descriptions, payload, 512,
                        state["proxy"] if projection == "down" else None, state["beta"],
                    )
                for sample, (record, rank) in enumerate(zip(test_records, test_ranks)):
                    rows, chosen = evaluate_invocation(state, expert, int(record), int(rank), matrices, layouts, policy_name, policy, 512, rates, budgets)
                    metric_rows.extend(rows)
                    for budget, value in chosen.items():
                        for projection, selected in value["selected"].items():
                            action_rows.append({
                                "request_id": value["row"]["request_id"], "sequence_id": value["row"]["sequence_id"],
                                "position": value["row"]["position"], "layer": layer, "expert_id": expert,
                                "projection": projection, "policy": policy_name, "budget_bpw": budget,
                                "coordinate_ids_json": json.dumps([action.coordinate for action in selected["actions"]]),
                                "before_masks_json": json.dumps([action.before_mask for action in selected["actions"]]),
                                "added_descriptions_json": json.dumps([list(action.add_descriptions) for action in selected["actions"]]),
                                "after_masks_json": json.dumps([action.after_mask for action in selected["actions"]]),
                                "marginal_damage_reduction_json": json.dumps([action.score for action in selected["actions"]]),
                                "action_page_ids_json": json.dumps(selected["action_pages"]),
                                "selected_page_ids_json": json.dumps(sorted(selected["pages"])),
                                "final_state_masks_json": json.dumps(selected["states"].tolist()),
                                "logical_bytes": selected["logical_bytes"], "physical_bytes": selected["physical_bytes"],
                            })
                    if sample == 0 and expert == state["experts"][0]:
                        for page_size in map(int, config["page_sizes"]):
                            if page_size == 512:
                                continue
                            diagnostic_layouts = {}
                            for projection, inputs in (("gate", train_x), ("up", train_x), ("down", train_h)):
                                payload = matrices[projection]["base"].shape[0] // 8
                                diagnostic_layouts[projection] = build_layout(matrices[projection], inputs, layout_descriptions, payload, page_size, state["proxy"] if projection == "down" else None, state["beta"])
                            extra, _ = evaluate_invocation(state, expert, int(record), int(rank), matrices, diagnostic_layouts, policy_name, policy, page_size, rates, budgets)
                            page_rows.extend(extra)
            atomic_parquet(args.output / "metrics.parquet", metric_rows)
            atomic_parquet(args.output / "selected_description_actions.parquet", action_rows)
            atomic_parquet(args.output / "full_support_q3.parquet", full_rows)
            atomic_parquet(args.output / "page_diagnostics.parquet", page_rows)
            facts["last_completed"] = {"layer": layer, "expert": expert, "unix": time.time()}
            atomic_json(args.output / "run_facts.json", facts)
            print(json.dumps({"layer": layer, "expert": expert, "metrics": len(metric_rows), "actions": len(action_rows)}), flush=True)
            gc.collect(); torch.cuda.empty_cache()
    storage = []
    for policy, suffix in (("ordered_a", 2.0), ("bidirectional_ab", 2.0), ("parity_abq", 3.0)):
        for replicas in (1, 2, 4):
            multiplier = (4.25 + suffix * replicas) / 4.25
            storage.append({
                "policy": policy, "replicas": replicas, "reference_fallback_bpw": 4.25,
                "suffix_bpw": suffix * replicas, "external_storage_multiplier": multiplier,
                "within_5x": multiplier <= 5.0,
            })
    pd.DataFrame(storage).to_csv(args.output / "storage_accounting.csv", index=False)
    facts["completed_unix"] = time.time(); facts["wall_seconds"] = facts["completed_unix"] - facts["started_unix"]
    atomic_json(args.output / "run_facts.json", facts)


if __name__ == "__main__":
    main()
