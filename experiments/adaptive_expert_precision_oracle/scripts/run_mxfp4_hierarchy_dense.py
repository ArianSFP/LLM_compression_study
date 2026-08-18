#!/usr/bin/env python3
"""Phase A: exact embedded Q2 -> Q3 -> packed MXFP4 hierarchy.

The script never quantizes or rewrites the authoritative Q4 endpoint.  It
loads packed compressed-tensors leaf nibbles and E8M0 scales, constructs a
balanced 2+1+1 code hierarchy, and asserts exact packed-byte recovery before
performing any functional evaluation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def occurrence(data: dict[str, np.ndarray], expert: int, split: str, maximum: int) -> tuple[np.ndarray, np.ndarray]:
    found = np.argwhere((data["expert_ids"] == expert) & (data["split"][:, None] == split))[:maximum]
    if not len(found):
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return found[:, 0], found[:, 1]


def select_varied_experts(data: dict[str, np.ndarray], per_stratum: int, minimum: int) -> tuple[dict[int, str], dict[int, dict[str, int]]]:
    rows = []
    for expert in range(256):
        counts = {split: int(np.sum(data["expert_ids"][data["split"] == split] == expert)) for split in ("train", "validation", "test")}
        if min(counts.values()) >= minimum:
            rows.append((sum(counts.values()), expert, counts))
    if len(rows) < 3 * per_stratum:
        raise RuntimeError(f"only {len(rows)} experts have complete split coverage")
    rows.sort()
    middle = len(rows) // 2
    selections = {
        "cold": rows[:per_stratum],
        "median": rows[middle - per_stratum // 2: middle - per_stratum // 2 + per_stratum],
        "hot": rows[-per_stratum:],
    }
    strata: dict[int, str] = {}
    facts: dict[int, dict[str, int]] = {}
    for label, values in selections.items():
        for _, expert, counts in values:
            strata[expert] = label
            facts[expert] = counts
    return strata, facts


def silu(value: torch.Tensor) -> torch.Tensor:
    return value * torch.sigmoid(value)


def qenergy(error: torch.Tensor, proxy: torch.Tensor, beta: float) -> torch.Tensor:
    return torch.sum(error * error, dim=-1) + beta * torch.sum((error @ proxy) ** 2, dim=-1)


def prepare_expert(tensors: dict[str, Any], trees: dict[str, Any]) -> dict[str, list[np.ndarray]]:
    result: dict[str, list[np.ndarray]] = {}
    for projection in PROJECTIONS:
        tensor = tensors[projection]
        tree = trees[projection]
        exact_packed, exact_scales = tensor.exact_streams()
        if not np.array_equal(tree.reconstruct_leaf_codes(tensor.codes), tensor.codes):
            raise AssertionError("leaf-code reconstruction is not exact")
        if not np.array_equal(exact_packed, __import__("oracle_study.mxfp4_embed", fromlist=["pack_leaf_codes"]).pack_leaf_codes(tree.reconstruct_leaf_codes(tensor.codes))):
            raise AssertionError("packed Q4 endpoint differs from checkpoint")
        if not np.array_equal(exact_scales, tensor.scale_codes):
            raise AssertionError("E8M0 scale bytes changed")
        result[projection] = [tree.decode(tensor, level) for level in (2, 3, 4)]
        if not np.array_equal(result[projection][2], tensor.dequantize()):
            raise AssertionError("level-four numerical decode is not exact")
    return result


def projection_validation_score(
    state: dict[str, Any], projection: str, candidate: Any, maximum: int, proxy: np.ndarray, beta: float,
) -> tuple[float, float, int]:
    numerator2 = numerator3 = denominator = 0.0
    count = 0
    device = "cuda"
    proxy_t = torch.from_numpy(proxy).to(device)
    for expert in state["experts"]:
        records, _ = occurrence(state["data"], expert, "validation", maximum)
        if not len(records):
            continue
        x = torch.from_numpy(np.asarray(state["data"]["x"][records], np.float32)).to(device)
        tensors = state["tensors"][expert]
        reference = torch.from_numpy(tensors[projection].dequantize()).to(device)
        w2 = torch.from_numpy(candidate.decode(tensors[projection], 2)).to(device)
        w3 = torch.from_numpy(candidate.decode(tensors[projection], 3)).to(device)
        if projection == "down":
            gate = x @ torch.from_numpy(tensors["gate"].dequantize()).to(device).T
            up = x @ torch.from_numpy(tensors["up"].dequantize()).to(device).T
            inputs = silu(gate) * up
            e2 = inputs @ (reference - w2).T
            e3 = inputs @ (reference - w3).T
            base = inputs @ reference.T
            numerator2 += float(qenergy(e2, proxy_t, beta).sum())
            numerator3 += float(qenergy(e3, proxy_t, beta).sum())
            denominator += float(qenergy(base, proxy_t, beta).sum())
        else:
            e2 = x @ (reference - w2).T
            e3 = x @ (reference - w3).T
            base = x @ reference.T
            numerator2 += float(torch.sum(e2 * e2))
            numerator3 += float(torch.sum(e3 * e3))
            denominator += float(torch.sum(base * base))
        count += len(records)
        del x, reference, w2, w3
    return numerator2 / max(denominator, 1e-30), numerator3 / max(denominator, 1e-30), count


def fit_candidates(states: dict[int, dict[str, Any]], config: dict[str, Any], output: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from oracle_study.mxfp4_embed import combine_leaf_statistics, fit_tree, leaf_importance_statistics, select_tree, tree_distortion, tree_orders
    from run_phase_a_remote import proxy_gradients

    candidates: dict[str, list[Any]] = {projection: [] for projection in PROJECTIONS}
    records: list[dict[str, Any]] = []
    for projection in PROJECTIONS:
        masses = []
        for state in states.values():
            train_x = np.asarray(state["data"]["x"][state["data"]["split"] == "train"], np.float32)
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
        for family in ("natural", "sign_preserving", "unrestricted"):
            stable_offset = int(hashlib.sha256((projection + ":" + family).encode()).hexdigest()[:8], 16) % 10000
            orders = tree_orders(family, int(config["seed"]) + stable_offset, int(config["tree_random_candidates"]))
            for mode in ("free_fp16", "e2m1"):
                tree = select_tree(orders, total, family, mode, float(config["tree_q2_tolerance"]))
                candidates[projection].append(tree)
                records.append({
                    "projection": projection, "family": family, "centroid_mode": mode,
                    "train_q2_distortion": tree_distortion(tree, total, 2),
                    "train_q3_distortion": tree_distortion(tree, total, 3),
                    "tree": tree.as_dict(),
                })
    selected: dict[str, Any] = {}
    beta = 512.0
    for projection in PROJECTIONS:
        scored = []
        for candidate in candidates[projection]:
            q2 = q3 = count = 0
            for state in states.values():
                proxy, proxy_facts = proxy_gradients(state["data"]["xplus"], [state["data"][f"h{h}_router_logits"] for h in range(1, 5)], state["data"]["split"])
                a, b, n = projection_validation_score(state, projection, candidate, int(config["max_validation_invocations_per_expert"]), proxy, float(proxy_facts["beta"]))
                q2 += a * n; q3 += b * n; count += n
            scored.append((q2 / max(count, 1), q3 / max(count, 1), candidate, count))
        best2 = min(v[0] for v in scored)
        eligible = [v for v in scored if v[0] <= best2 * (1.0 + float(config["tree_q2_tolerance"])) + 1e-30]
        chosen = min(eligible, key=lambda value: value[1])
        selected[projection] = chosen[2]
        for q2, q3, tree, count in scored:
            records.append({
                "projection": projection, "family": tree.name, "centroid_mode": tree.centroid_mode,
                "validation_q2_relative_damage": q2, "validation_q3_relative_damage": q3,
                "validation_invocations": count, "selected": tree is chosen[2], "record_type": "validation_selection",
            })
    atomic_json(output / "selected_trees.json", {name: tree.as_dict() for name, tree in selected.items()})
    return selected, records


def evaluate_expert(
    state: dict[str, Any], expert: int, split: str, maximum: int, trees: dict[str, Any],
    proxy: np.ndarray, beta: float, rows: list[dict[str, Any]], projection_rows: list[dict[str, Any]],
) -> None:
    from oracle_study.rrq import encode_legacy_q2

    records, ranks = occurrence(state["data"], expert, split, maximum)
    if not len(records):
        return
    data = state["data"]
    device = "cuda"
    x = torch.from_numpy(np.asarray(data["x"][records], np.float32)).to(device)
    tensors = state["tensors"][expert]
    matrices_np = prepare_expert(tensors, trees)
    matrices = {p: [torch.from_numpy(v).to(device) for v in values] for p, values in matrices_np.items()}
    g = [x @ w.T for w in matrices["gate"]]
    u = [x @ w.T for w in matrices["up"]]
    exact_h = silu(g[2]) * u[2]
    exact_y = exact_h @ matrices["down"][2].T
    base_h = silu(g[0]) * u[0]
    base_y = base_h @ matrices["down"][0].T
    proxy_t = torch.from_numpy(proxy).to(device)
    base_damage = qenergy(exact_y - base_y, proxy_t, beta)
    # Historical activation-aware Q2 is a paired comparator against this exact reference.
    train_x = np.asarray(data["x"][data["split"] == "train"], np.float32)
    moment_x = np.mean(train_x.astype(np.float64) ** 2, axis=0)
    q2g = encode_legacy_q2(tensors["gate"].dequantize(), 64, moment_x, "fp16").reconstruction
    q2u = encode_legacy_q2(tensors["up"].dequantize(), 64, moment_x, "fp16").reconstruction
    htrain_rows, _ = occurrence(data, expert, "train", maximum)
    xt = torch.from_numpy(np.asarray(data["x"][htrain_rows], np.float32)).to(device)
    ht = silu(xt @ matrices["gate"][2].T) * (xt @ matrices["up"][2].T)
    moment_h = torch.mean(ht.double() ** 2, dim=0).cpu().numpy()
    q2d = encode_legacy_q2(tensors["down"].dequantize(), 64, moment_h, "fp16").reconstruction
    q2_y = silu(x @ torch.from_numpy(q2g).to(device).T) * (x @ torch.from_numpy(q2u).to(device).T)
    q2_y = q2_y @ torch.from_numpy(q2d).to(device).T
    q2_damage = qenergy(exact_y - q2_y, proxy_t, beta)
    for projection in PROJECTIONS:
        inputs = x if projection != "down" else exact_h
        target = inputs @ matrices[projection][2].T
        local_base_error = target - inputs @ matrices[projection][0].T
        local_base = torch.sum(local_base_error * local_base_error, dim=1) if projection != "down" else qenergy(local_base_error, proxy_t, beta)
        for depth in range(3):
            error = target - inputs @ matrices[projection][depth].T
            damage = torch.sum(error * error, dim=1) if projection != "down" else qenergy(error, proxy_t, beta)
            for sample, record in enumerate(records):
                projection_rows.append({
                    "request_id": str(data["request_id"][record]), "sequence_id": str(data["sequence_id"][record]),
                    "position": int(data["position"][record]), "split": split, "layer": state["layer"],
                    "expert_id": expert, "expert_stratum": state["strata"][expert], "projection": projection,
                    "embedded_level": depth + 2, "recovery": float(1 - damage[sample] / torch.clamp(local_base[sample], min=1e-30)),
                    "damage": float(damage[sample]), "base_damage": float(local_base[sample]),
                })
    for kg, ku, kd in itertools.product(range(3), repeat=3):
        h = silu(g[kg]) * u[ku]
        y = h @ matrices["down"][kd].T
        damage = qenergy(exact_y - y, proxy_t, beta)
        error = exact_y - y
        logical_bytes = (kg + ku + kd) * (MATRIX_WEIGHTS // 8)
        physical_bytes = sum(math.ceil((depth * MATRIX_WEIGHTS // 8) / 512) * 512 for depth in (kg, ku, kd))
        for sample, record in enumerate(records):
            sorted_logits = np.sort(data["router_logits"][record])
            rows.append({
                "request_id": str(data["request_id"][record]), "sequence_id": str(data["sequence_id"][record]),
                "position": int(data["position"][record]), "split": split, "layer": state["layer"],
                "expert_id": expert, "expert_stratum": state["strata"][expert], "router_rank": int(ranks[sample]) + 1,
                "router_coefficient": float(data["router_weights"][record, ranks[sample]]),
                "router_boundary_margin": float(sorted_logits[-8] - sorted_logits[-9]),
                "gate_level": kg + 2, "up_level": ku + 2, "down_level": kd + 2,
                "tuple": f"({kg + 2},{ku + 2},{kd + 2})", "logical_bytes": logical_bytes,
                "physical_bytes": physical_bytes, "logical_bpw": 8 * logical_bytes / EXPERT_WEIGHTS,
                "physical_bpw": 8 * physical_bytes / EXPERT_WEIGHTS,
                "recovery": float(1 - damage[sample] / torch.clamp(base_damage[sample], min=1e-30)),
                "damage": float(damage[sample]), "base_damage": float(base_damage[sample]),
                "relative_output_error": float(torch.linalg.norm(error[sample]) / torch.clamp(torch.linalg.norm(exact_y[sample]), min=1e-30)),
                "q2_comparator_relative_damage": float(q2_damage[sample] / torch.clamp(base_damage[sample], min=1e-30)),
                "endpoint_exact": bool(kg == ku == kd == 2),
            })
    del matrices, g, u, exact_h, exact_y, base_h, base_y, q2_y
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trees", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    experiment = args.config.resolve().parents[1]
    sys.path[:0] = [str(experiment / "src"), str(experiment / "scripts")]
    from oracle_study.mxfp4_embed import EmbeddedTree, audit_checkpoint, load_compressed_mxfp4_expert
    from run_phase_a_remote import proxy_gradients

    args.output.mkdir(parents=True, exist_ok=True)
    audit = audit_checkpoint(args.checkpoint, config["layers"])
    if not audit["passed"]:
        raise RuntimeError(audit)
    index_path = args.checkpoint / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    audit.update({"repository": config["reference"]["repository"], "revision": config["reference"]["revision"], "index_sha256": sha256(index_path)})
    atomic_json(args.output / "checkpoint_audit.json", audit)
    loaded = np.load(args.captures, allow_pickle=False)
    all_data = {name: loaded[name] for name in loaded.files}
    split_requests = {split: set(map(str, all_data["request_id"][all_data["split"] == split])) for split in ("train", "validation", "test")}
    if any(split_requests[a] & split_requests[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise RuntimeError("request split leakage")
    facts: dict[str, Any] = {
        "run_id": config["run_id"], "started_unix": time.time(), "host": platform.node(),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "checkpoint_audit": audit,
        "capture_status": config["status"], "split_requests": {k: len(v) for k, v in split_requests.items()}, "layers": {},
    }
    states: dict[int, dict[str, Any]] = {}
    for layer in config["layers"]:
        mask = all_data["layer"] == layer
        data = {name: value[mask] for name, value in all_data.items()}
        strata, counts = select_varied_experts(data, int(config["experts_per_stratum"]), int(config["minimum_split_occurrences"]))
        experts = sorted(strata)
        tensors = {expert: {projection: load_compressed_mxfp4_expert(args.checkpoint, weight_map, layer, expert, projection) for projection in PROJECTIONS} for expert in experts}
        states[layer] = {"layer": layer, "data": data, "strata": strata, "counts": counts, "experts": experts, "tensors": tensors}
        facts["layers"][str(layer)] = {"experts": experts, "strata": {str(k): v for k, v in strata.items()}, "split_occurrences": {str(k): v for k, v in counts.items()}}
        atomic_json(args.output / "run_facts.json", facts)
        print(json.dumps({"prepared_layer": layer, "experts": experts}), flush=True)
    if args.trees is not None:
        locked = json.loads(args.trees.read_text())
        trees = {name: EmbeddedTree(
            np.asarray(value["leaf_to_parent"], np.uint8), np.asarray(value["leaf_to_r1"], np.uint8),
            np.asarray(value["leaf_to_r2"], np.uint8), np.asarray(value["c2"], np.float32),
            np.asarray(value["c3"], np.float32), value["name"], value["centroid_mode"],
        ) for name, value in locked.items()}
        tree_rows = [{"projection": name, "family": tree.name, "centroid_mode": tree.centroid_mode, "locked_from": str(args.trees)} for name, tree in trees.items()]
        atomic_json(args.output / "selected_trees.json", locked)
        facts["tree_selection"] = "locked before exact-checkpoint confirmation"
    else:
        trees, tree_rows = fit_candidates(states, config, args.output)
        facts["tree_selection"] = "fit on training and selected on validation"
    pd.DataFrame(tree_rows).to_parquet(args.output / "tree_candidates.parquet", index=False)
    rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    completed: set[tuple[int, int]] = set()
    checkpoint = args.output / "dense_metrics.parquet"
    if checkpoint.exists():
        old = pd.read_parquet(checkpoint)
        rows = old.to_dict("records")
        completed = set(zip(old["layer"].astype(int), old["expert_id"].astype(int)))
        if (args.output / "projection_metrics.parquet").exists():
            projection_rows = pd.read_parquet(args.output / "projection_metrics.parquet").to_dict("records")
    for layer, state in states.items():
        proxy, proxy_facts = proxy_gradients(state["data"]["xplus"], [state["data"][f"h{h}_router_logits"] for h in range(1, 5)], state["data"]["split"])
        facts["layers"][str(layer)]["proxy"] = proxy_facts
        for expert in state["experts"]:
            if (layer, expert) in completed:
                continue
            for split, maximum in (("validation", int(config["max_validation_invocations_per_expert"])), ("test", int(config["max_test_invocations_per_expert"]))):
                evaluate_expert(state, expert, split, maximum, trees, proxy, float(proxy_facts["beta"]), rows, projection_rows)
            atomic_parquet(args.output / "dense_metrics.parquet", rows)
            atomic_parquet(args.output / "projection_metrics.parquet", projection_rows)
            facts["last_completed"] = {"layer": layer, "expert": expert, "unix": time.time()}
            atomic_json(args.output / "run_facts.json", facts)
            print(json.dumps({"completed_layer": layer, "expert": expert, "dense_rows": len(rows)}), flush=True)
            gc.collect()
    facts["completed_unix"] = time.time()
    facts["wall_seconds"] = facts["completed_unix"] - facts["started_unix"]
    facts["test_invocations"] = int(len(set((r["request_id"], r["position"], r["layer"], r["expert_id"]) for r in rows if r["split"] == "test")))
    atomic_json(args.output / "run_facts.json", facts)


if __name__ == "__main__":
    main()
