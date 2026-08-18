#!/usr/bin/env python3
"""Bounded per-plane support, local-swap, and beam diagnostics for MXFP4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(EXPERIMENT / "scripts")]

from oracle_study.mxfp4_embed import audit_checkpoint, load_compressed_mxfp4_expert
from run_mxfp4_hierarchy_dense import select_varied_experts
from run_mxfp4_selective_pages import occurrence, silu, tree_from_record


TARGETS = (0.8, 0.9, 0.95, 0.99)


def recovery_curve(values: np.ndarray, order: np.ndarray) -> np.ndarray:
    target = values.sum(axis=0, dtype=np.float64)
    denominator = max(float(target @ target), 1e-30)
    residual = target.copy()
    output = []
    for atom in order:
        residual -= values[int(atom)]
        output.append(1.0 - float(residual @ residual) / denominator)
    return np.asarray(output)


def exact_greedy(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.from_numpy(np.asarray(values, np.float32)).cuda()
    target = tensor.sum(dim=0)
    denominator = max(float(torch.dot(target, target)), 1e-30)
    residual = target.clone()
    norms = (tensor * tensor).sum(dim=1)
    chosen = torch.zeros(len(tensor), dtype=torch.bool, device="cuda")
    order = []
    curve = []
    for _ in range(len(tensor)):
        marginal = 2.0 * (tensor @ residual) - norms
        marginal[chosen] = -torch.inf
        atom = int(torch.argmax(marginal))
        chosen[atom] = True
        residual -= tensor[atom]
        order.append(atom)
        curve.append(1.0 - float(torch.dot(residual, residual)) / denominator)
    return np.asarray(order), np.asarray(curve)


def first_crossing(curve: np.ndarray, target: float) -> int:
    indices = np.flatnonzero(curve >= target)
    return int(indices[0] + 1) if len(indices) else len(curve)


def local_swaps(values: np.ndarray, support: np.ndarray, max_swaps: int = 8, selected_pool: int = 64) -> tuple[float, int]:
    tensor = torch.from_numpy(np.asarray(values, np.float32)).cuda()
    target = tensor.sum(dim=0)
    denominator = max(float(torch.dot(target, target)), 1e-30)
    selected = torch.zeros(len(tensor), dtype=torch.bool, device="cuda")
    selected[torch.from_numpy(support.astype(np.int64)).cuda()] = True
    current = tensor[selected].sum(dim=0)
    swaps = 0
    for _ in range(max_swaps):
        residual = target - current
        selected_ids = torch.nonzero(selected, as_tuple=False).flatten()
        unselected_ids = torch.nonzero(~selected, as_tuple=False).flatten()
        if not len(unselected_ids):
            break
        removal_damage = ((residual[None, :] + tensor[selected_ids]) ** 2).sum(dim=1)
        selected_ids = selected_ids[torch.argsort(removal_damage)[:selected_pool]]
        best = None
        unselected = tensor[unselected_ids]
        unselected_norms = (unselected * unselected).sum(dim=1)
        for remove_id in selected_ids:
            after_remove = residual + tensor[remove_id]
            damages = torch.dot(after_remove, after_remove) - 2.0 * (unselected @ after_remove) + unselected_norms
            position = int(torch.argmin(damages))
            candidate = (float(damages[position]), int(remove_id), int(unselected_ids[position]))
            if best is None or candidate[0] < best[0]:
                best = candidate
        current_damage = float(torch.dot(residual, residual))
        if best is None or best[0] >= current_damage - 1e-12:
            break
        _, remove_id, add_id = best
        selected[remove_id] = False
        selected[add_id] = True
        current += tensor[add_id] - tensor[remove_id]
        swaps += 1
    residual = target - current
    return 1.0 - float(torch.dot(residual, residual)) / denominator, swaps


def beam_prefix(values: np.ndarray, width: int, pool: int = 32, depth: int = 12) -> list[tuple[int, float]]:
    values = np.asarray(values, np.float32)
    target = values.sum(axis=0)
    denominator = max(float(np.sum(target.astype(np.float64) ** 2)), 1e-30)
    candidates = np.argsort(np.sum(values.astype(np.float64) ** 2, axis=1))[::-1][:pool]
    beams: list[tuple[float, tuple[int, ...], np.ndarray]] = [(float(np.sum(target.astype(np.float64) ** 2)), tuple(), target.copy())]
    curve = []
    for k in range(1, min(depth, len(candidates)) + 1):
        expanded = []
        for _, chosen, residual in beams:
            used = set(chosen)
            for atom_value in candidates:
                atom = int(atom_value)
                if atom in used:
                    continue
                updated = residual - values[atom]
                score = float(np.sum(updated.astype(np.float64) ** 2))
                expanded.append((score, chosen + (atom,), updated))
        expanded.sort(key=lambda row: row[0])
        seen = set(); beams = []
        for row in expanded:
            key = tuple(sorted(row[1]))
            if key in seen:
                continue
            seen.add(key); beams.append(row)
            if len(beams) == width:
                break
        curve.append((k, 1.0 - beams[0][0] / denominator))
    return curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dense-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    audit = audit_checkpoint(args.checkpoint, config["layers"])
    if not audit["passed"]:
        raise RuntimeError(audit)
    data_file = np.load(args.captures, allow_pickle=False)
    all_data = {name: data_file[name] for name in data_file.files}
    weight_map = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    tree_records = json.loads((args.dense_output / "selected_trees.json").read_text())
    trees = {name: tree_from_record(tree_records[name]) for name in ("gate", "up", "down")}
    rows = []
    for layer in config["layers"]:
        mask = all_data["layer"] == layer
        data = {name: value[mask] for name, value in all_data.items()}
        strata, _ = select_varied_experts(data, int(config["experts_per_stratum"]), int(config["minimum_split_occurrences"]))
        expert = sorted(strata)[-1]
        records, _ = occurrence(data, expert, "test", 1)
        record = int(records[0])
        tensors = {p: load_compressed_mxfp4_expert(args.checkpoint, weight_map, layer, expert, p) for p in ("gate", "up", "down")}
        matrices = {p: [trees[p].decode(tensors[p], level) for level in (2, 3, 4)] for p in tensors}
        x = np.asarray(data["x"][record], np.float32)
        g = matrices["gate"][2] @ x; u = matrices["up"][2] @ x
        h = (g / (1 + np.exp(-np.clip(g, -80, 80)))) * u
        inputs = {"gate": x, "up": x, "down": h}
        for projection in ("gate", "up", "down"):
            vector = inputs[projection]
            deltas = [matrices[projection][1] - matrices[projection][0], matrices[projection][2] - matrices[projection][1]]
            for plane, delta in enumerate(deltas, 1):
                values = delta.T * vector[:, None]
                diagonal = np.argsort(np.sum(values.astype(np.float64) ** 2, axis=1))[::-1]
                diagonal_curve = recovery_curve(values, diagonal)
                greedy_order, greedy_curve = exact_greedy(values)
                common = {"request_id": str(data["request_id"][record]), "layer": layer, "expert_id": expert,
                          "projection": projection, "refinement_plane": plane, "total_coordinates": len(values)}
                for target in TARGETS:
                    dk = first_crossing(diagonal_curve, target); gk = first_crossing(greedy_curve, target)
                    swapped_recovery, swaps = local_swaps(values, greedy_order[:gk])
                    rows.extend([
                        {**common, "comparison": "diagonal", "target": target, "k": dk, "recovery": float(diagonal_curve[dk - 1]), "beam_width": None, "swaps": 0},
                        {**common, "comparison": "exact_greedy", "target": target, "k": gk, "recovery": float(greedy_curve[gk - 1]), "beam_width": None, "swaps": 0},
                        {**common, "comparison": "greedy_plus_local_swaps", "target": target, "k": gk, "recovery": swapped_recovery, "beam_width": None, "swaps": swaps},
                    ])
                for width in (16, 64, 256):
                    for k, recovery in beam_prefix(values, width):
                        rows.append({**common, "comparison": "bounded_beam", "target": np.nan, "k": k,
                                     "recovery": recovery, "beam_width": width, "swaps": 0})
        print(json.dumps({"completed_layer": layer, "rows": len(rows)}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()
