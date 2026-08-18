#!/usr/bin/env python3
"""Selective atom-major RRQ retention and stage-support pilot.

The deployable primary path uses the existing training-only generalized basis
for gate/up and native coordinates for down.  Every selected item is an exact
atom/stage packet with nested RRQ depth and a page-aligned physical charge.
The full-support three-stage atom package is the matched dense comparator.
"""

from __future__ import annotations

import argparse
import heapq
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


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


def damage(error: np.ndarray, proxy: np.ndarray, beta: float) -> float:
    e = np.asarray(error, dtype=np.float64)
    return float(np.sum(e * e) + beta * np.sum((e @ proxy.astype(np.float64)) ** 2))


def atomic_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_parquet(temp, index=False)
    temp.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def records_to_specs(records: list[dict[str, Any]]) -> tuple[Any, ...]:
    from oracle_study.rrq import QuantizerSpec
    return tuple(QuantizerSpec(v["name"], int(v["group_size"]), v.get("scale_storage", "fp16")) for v in records)


def occurrence(data: dict[str, np.ndarray], expert: int, split: str, maximum: int) -> tuple[np.ndarray, np.ndarray]:
    found = np.argwhere((data["expert_ids"] == expert) & (data["split"][:, None] == split))[:maximum]
    if not len(found):
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return found[:, 0], found[:, 1]


def choose_search_experts(experts: list[int], strata: dict[int, str], data: dict[str, np.ndarray]) -> list[int]:
    result = []
    for label in ("cold", "median", "hot"):
        candidates = [e for e in experts if strata[e] == label]
        candidates.sort(key=lambda e: int(np.sum(data["expert_ids"][data["split"] == "validation"] == e)), reverse=True)
        if candidates:
            result.append(candidates[0])
    return result


def stage_packet_facts(series: Any, page_size: int) -> list[dict[str, int]]:
    from oracle_study.rrq import atom_packet_bytes
    result = []
    for stage, encoding in enumerate(series.stages, 1):
        logical, physical = atom_packet_bytes(encoding, page_size)
        result.append({"stage": stage, "logical": logical, "physical": physical, "pages": physical // page_size})
    return result


def atom_series(reference: np.ndarray, base: np.ndarray, transform: Any, specs: tuple[Any, ...]) -> dict[str, Any]:
    from oracle_study.rrq import atom_stage_matrix, build_atom_rrq_series
    target = (reference - base) @ transform.synthesis
    series = build_atom_rrq_series(target, specs)
    stages = [atom_stage_matrix(series, stage) for stage in range(1, 4)]
    return {"series": series, "stages": stages, "analysis": transform.analysis, "synthesis": transform.synthesis}


def full_outputs(item: dict[str, Any], inputs: np.ndarray) -> list[np.ndarray]:
    z = inputs @ item["analysis"].T
    result = [np.zeros((len(inputs), item["stages"][0].shape[0]), np.float32)]
    for stage in item["stages"]:
        result.append(result[-1] + z @ stage.T)
    return result


def nested_energy_actions(item: dict[str, Any], coefficients: np.ndarray, page_size: int) -> list[dict[str, Any]]:
    """Nested stage order using the requested diagonal energy score."""
    facts = stage_packet_facts(item["series"], page_size)
    depth = np.zeros(item["stages"][0].shape[1], dtype=np.int8)
    norms = [np.sum(stage.astype(np.float64) ** 2, axis=0) for stage in item["stages"]]
    actions: list[dict[str, Any]] = []
    heap = []
    for atom in range(len(depth)):
        score = float(coefficients[atom]) ** 2 * norms[0][atom] / facts[0]["physical"]
        heapq.heappush(heap, (-float(score), atom, 1))
    for _ in range(len(depth) * 3):
        negative, atom, stage = heapq.heappop(heap)
        score = -negative
        if stage != int(depth[atom]) + 1:
            raise AssertionError((atom, stage, depth[atom]))
        depth[atom] = stage
        packet = facts[stage - 1]
        actions.append({"atom_id": atom, "stage": stage, "score": float(score), **packet})
        if stage < 3:
            next_score = float(coefficients[atom]) ** 2 * norms[stage][atom] / facts[stage]["physical"]
            heapq.heappush(heap, (-float(next_score), atom, stage + 1))
    return actions


def snapshots(item: dict[str, Any], coefficients: np.ndarray, actions: list[dict[str, Any]], rates: np.ndarray) -> list[dict[str, Any]]:
    current = np.zeros(item["stages"][0].shape[0], dtype=np.float32)
    logical = physical = 0
    cursor = 0
    result = []
    for target in rates:
        while cursor < len(actions) and 8 * (physical + actions[cursor]["physical"]) / EXPERT_WEIGHTS <= target + 1e-12:
            action = actions[cursor]
            current += coefficients[action["atom_id"]] * item["stages"][action["stage"] - 1][:, action["atom_id"]]
            logical += action["logical"]; physical += action["physical"]; cursor += 1
        result.append({"correction": current.copy(), "logical": logical, "physical": physical, "actions": cursor})
    return result


def matrix_snapshots(item: dict[str, Any], actions: list[dict[str, Any]], rates: np.ndarray) -> list[dict[str, Any]]:
    """Native-down selected matrices; apply them to each sequential h exactly."""
    current = np.zeros_like(item["stages"][0], dtype=np.float32)
    logical = physical = cursor = 0
    result = []
    for target in rates:
        while cursor < len(actions) and 8 * (physical + actions[cursor]["physical"]) / EXPERT_WEIGHTS <= target + 1e-12:
            action = actions[cursor]
            atom = action["atom_id"]
            current[:, atom] += item["stages"][action["stage"] - 1][:, atom]
            logical += action["logical"]; physical += action["physical"]; cursor += 1
        result.append({"matrix": current.copy(), "logical": logical, "physical": physical, "actions": cursor})
    return result


def exact_stage_support(contributions: np.ndarray, targets: list[float], exact_greedy: bool) -> tuple[dict[float, int | None], list[dict[str, float]]]:
    """Exact vector recovery for energy ranking or recomputed greedy ranking."""
    values = np.asarray(contributions, dtype=np.float32)
    target = np.sum(values, axis=0)
    denominator = float(np.sum(target.astype(np.float64) ** 2))
    if denominator <= 1e-30:
        return {threshold: 0 for threshold in targets}, []
    selected = np.zeros(len(values), bool)
    residual = target.copy()
    curve = []
    result: dict[float, int | None] = {threshold: None for threshold in targets}
    static = np.sum(values.astype(np.float64) ** 2, axis=1)
    order = np.argsort(static)[::-1]
    if exact_greedy:
        tensor = torch.from_numpy(values).cuda()
        gram = tensor @ tensor.T
        correlation = gram.sum(dim=1)
        norms_gpu = torch.diagonal(gram)
        selected_gpu = torch.zeros(len(values), dtype=torch.bool, device="cuda")
    for step in range(len(values)):
        if exact_greedy:
            marginal = 2.0 * correlation - norms_gpu
            marginal[selected_gpu] = -torch.inf
            chosen = int(torch.argmax(marginal))
            selected_gpu[chosen] = True
            correlation -= gram[:, chosen]
        else:
            chosen = int(order[step])
        selected[chosen] = True
        residual -= values[chosen]
        recovery = 1.0 - float(np.sum(residual.astype(np.float64) ** 2)) / denominator
        curve.append({"k": step + 1, "recovery": recovery, "atom_id": chosen})
        for threshold in targets:
            if result[threshold] is None and recovery >= threshold:
                result[threshold] = step + 1
        if all(value is not None for value in result.values()):
            break
    if exact_greedy:
        del tensor, gram, correlation, norms_gpu, selected_gpu
    return result, curve


def beam_prefix(contributions: np.ndarray, width: int, pool: int, depth: int = 16) -> list[dict[str, float]]:
    """Bounded exact beam diagnostic over the strongest energy candidates."""
    values = np.asarray(contributions, np.float32)
    target = np.sum(values, axis=0)
    denominator = max(float(np.sum(target.astype(np.float64) ** 2)), 1e-30)
    candidates = np.argsort(np.sum(values.astype(np.float64) ** 2, axis=1))[::-1][:pool]
    beams: list[tuple[float, tuple[int, ...], np.ndarray]] = [(float(np.sum(target * target)), tuple(), target.copy())]
    curve = []
    for k in range(1, min(depth, len(candidates)) + 1):
        expanded = []
        for _, chosen, residual in beams:
            used = set(chosen)
            for atom in candidates:
                atom = int(atom)
                if atom in used:
                    continue
                updated = residual - values[atom]
                score = float(np.sum(updated.astype(np.float64) ** 2))
                expanded.append((score, chosen + (atom,), updated))
        expanded.sort(key=lambda row: row[0])
        unique = {}; beams = []
        for row in expanded:
            key = tuple(sorted(row[1]))
            if key not in unique:
                unique[key] = True; beams.append(row)
            if len(beams) == width:
                break
        curve.append({"k": k, "recovery": 1.0 - beams[0][0] / denominator})
    return curve


def complete_queue_greedy(
    items: dict[str, Any], orders: dict[str, list[dict[str, Any]]], coefficients: dict[str, np.ndarray],
    gbase: np.ndarray, ubase: np.ndarray, down_base: np.ndarray, yref: np.ndarray,
    proxy: np.ndarray, beta: float, budgets: np.ndarray,
) -> list[dict[str, Any]]:
    """Exact complete-expert greedy over each projection queue's next action.

    Atom ordering inside each queue remains diagonal-energy; the choice among
    gate/up/down and resulting nonlinear utility is recomputed after every
    accepted packet. This bounded diagnostic isolates allocation regret from
    the much costlier all-atom complete-expert oracle.
    """
    device = "cuda"
    g = torch.from_numpy(gbase.copy()).to(device); u = torch.from_numpy(ubase.copy()).to(device)
    down = torch.from_numpy(down_base.copy()).to(device); target = torch.from_numpy(yref).to(device)
    proxy_t = torch.from_numpy(proxy).to(device)
    stage_t = {p: [torch.from_numpy(v).to(device) for v in items[p]["stages"]] for p in PROJECTIONS}
    coeff_t = {p: torch.from_numpy(coefficients[p]).to(device) for p in PROJECTIONS}
    cursor = {p: 0 for p in PROJECTIONS}; counts = {p: 0 for p in PROJECTIONS}; physical = 0
    def torch_damage(y: torch.Tensor) -> torch.Tensor:
        error = target - y
        return torch.sum(error * error) + beta * torch.sum((error @ proxy_t) ** 2)
    h = g * torch.sigmoid(g) * u; y = down @ h; current_damage = torch_damage(y); base_d = current_damage.clone()
    curve = [{"physical_bpw": 0.0, "recovery": 0.0, **{f"{p}_actions": 0 for p in PROJECTIONS}}]
    maximum_bytes = float(np.max(budgets)) * EXPERT_WEIGHTS / 8
    while True:
        candidates = []
        for projection in PROJECTIONS:
            if cursor[projection] >= len(orders[projection]): continue
            action = orders[projection][cursor[projection]]; atom = action["atom_id"]; stage = action["stage"] - 1
            if physical + action["physical"] > maximum_bytes + 1e-9: continue
            if projection == "gate":
                g2 = g + coeff_t[projection][atom] * stage_t[projection][stage][:, atom]
                h2 = g2 * torch.sigmoid(g2) * u; y2 = down @ h2
            elif projection == "up":
                u2 = u + coeff_t[projection][atom] * stage_t[projection][stage][:, atom]
                h2 = g * torch.sigmoid(g) * u2; y2 = down @ h2
            else:
                h2 = h; y2 = y + h[atom] * stage_t[projection][stage][:, atom]
            d2 = torch_damage(y2); utility = float((current_damage - d2) / action["physical"])
            candidates.append((utility, projection, action, g2 if projection == "gate" else g, u2 if projection == "up" else u, h2, y2, d2))
        if not candidates: break
        _, projection, action, g, u, h, y, current_damage = max(candidates, key=lambda row: row[0])
        if projection == "down":
            atom = action["atom_id"]; stage = action["stage"] - 1
            down[:, atom] += stage_t[projection][stage][:, atom]
        cursor[projection] += 1; counts[projection] += 1; physical += action["physical"]
        curve.append({"physical_bpw": 8 * physical / EXPERT_WEIGHTS, "recovery": float(1.0 - current_damage / torch.clamp(base_d, min=1e-30)), **{f"{p}_actions": counts[p] for p in PROJECTIONS}})
    result = []
    for budget in budgets:
        eligible = [row for row in curve if row["physical_bpw"] <= budget + 1e-12]
        result.append({"budget_bpw": float(budget), **eligible[-1]})
    del g, u, down, target, proxy_t, stage_t, coeff_t
    return result


def refit_generalized_series(state: dict[str, Any], projection: str, specs: tuple[Any, ...]) -> dict[int, list[tuple[Any, np.ndarray]]]:
    """Layer-shared generalized basis refitted to each post-stage residual."""
    from oracle_study.rrq import quantize_int2_stage
    from run_phase_a_v2_remote import generalized_full_rank
    audited = state["search_experts"]
    indices = [state["experts"].index(expert) for expert in audited]
    residuals = [state["reference"][projection][ei] - state["base"][projection][ei] for ei in indices]
    samples = state["train_x"] if projection != "down" else state["train_h"]
    result = {expert: [] for expert in audited}
    for stage_index, spec in enumerate(specs):
        stacked = torch.from_numpy(np.stack(residuals)).cuda().reshape(-1, residuals[0].shape[1])
        operator = (stacked.T @ stacked).cpu().numpy() / len(residuals)
        del stacked; torch.cuda.empty_cache()
        transform = generalized_full_rank(samples, operator)
        next_residuals = []
        for local_index, expert in enumerate(audited):
            target_atoms = residuals[local_index] @ transform.synthesis
            encoded = quantize_int2_stage(target_atoms.T, spec.group_size, spec.name, None, spec.scale_storage)
            atoms = encoded.reconstruction.T.copy()
            result[expert].append((transform, atoms))
            next_residuals.append(residuals[local_index] - atoms @ transform.analysis)
        residuals = next_residuals
    return result


def audit_bases(state: dict[str, Any], paths: dict[str, tuple[Any, ...]], config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    from oracle_study.bases import identity_transform, learned_orthogonal_transform, pca_transform, random_orthogonal_transform, two_view_overcomplete
    from run_phase_a_v2_remote import generalized_full_rank
    layer = state["layer"]
    native_x = identity_transform(2048); native_h = identity_transform(512)
    pca_x = pca_transform(state["train_x"], "cuda")
    learned_x = learned_orthogonal_transform(state["train_x"], state["operator"], int(config["seed"] + layer), "cuda")
    butterfly = random_orthogonal_transform(2048, int(config["seed"] + 1000 + layer)); butterfly.name = "butterfly_hadamard_control"
    over = two_view_overcomplete(pca_x, state["transform"])
    families_x = {"native": native_x, "pca": pca_x, "generalized": state["transform"], "learned_orthogonal": learned_x, "butterfly_hadamard_control": butterfly, "overcomplete_2x": over}
    families_h = {"native": native_h}
    refit = {projection: refit_generalized_series(state, projection, paths[projection]) for projection in PROJECTIONS}
    for expert in state["search_experts"]:
        ei = state["experts"].index(expert)
        record_parts = []
        split_labels = []
        for split_name, maximum in (("validation", int(config["max_validation_invocations_per_expert"])), ("test", int(config["max_test_invocations_per_expert"]))):
            part, _ = occurrence(state["data"], expert, split_name, maximum)
            record_parts.append(part); split_labels.extend([split_name] * len(part))
        records = np.concatenate(record_parts) if record_parts else np.empty(0, np.int64)
        if not len(records):
            continue
        x = np.asarray(state["data"]["x"][records], np.float32)
        gref = x @ state["reference"]["gate"][ei].T; uref = x @ state["reference"]["up"][ei].T
        href = silu(gref) * uref
        for projection in PROJECTIONS:
            inputs = x if projection != "down" else href
            families = families_x if projection != "down" else families_h
            reference = state["reference"][projection][ei]; base = state["base"][projection][ei]
            base_error = inputs @ (reference - base).T
            base_damage = np.sum(base_error.astype(np.float64) ** 2, axis=1)
            for family, transform in families.items():
                item = atom_series(reference, base, transform, paths[projection])
                outputs = full_outputs(item, inputs)
                for depth in range(1, 4):
                    error = base_error - outputs[depth]
                    values = np.sum(error.astype(np.float64) ** 2, axis=1)
                    for sample, record in enumerate(records):
                        rows.append({"layer": layer, "expert_id": expert, "request_id": str(state["data"]["request_id"][record]), "split": split_labels[sample], "projection": projection, "basis": family, "stage": depth, "recovery": 1.0 - float(values[sample]) / max(float(base_damage[sample]), 1e-30), "full_support": True})
            cumulative = np.zeros_like(base_error)
            for depth, (transform, atoms) in enumerate(refit[projection][expert], 1):
                cumulative += (inputs @ transform.analysis.T) @ atoms.T
                values = np.sum((base_error - cumulative).astype(np.float64) ** 2, axis=1)
                for sample, record in enumerate(records):
                    rows.append({"layer": layer, "expert_id": expert, "request_id": str(state["data"]["request_id"][record]), "split": split_labels[sample], "projection": projection, "basis": "refit_generalized", "stage": depth, "recovery": 1.0 - float(values[sample]) / max(float(base_damage[sample]), 1e-30), "full_support": True})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--dense-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text()); args.output.mkdir(parents=True, exist_ok=True)
    experiment = args.config.resolve().parents[1]
    sys.path[:0] = [str(experiment / "src"), str(experiment / "scripts"), config["remote"]["gguf_python"]]
    import gguf
    from oracle_study.bases import identity_transform
    from oracle_study.rrq import encode_legacy_q2
    from oracle_study.weights import load_gguf_experts
    from run_phase_a_remote import proxy_gradients, select_experts
    from run_phase_a_v2_remote import generalized_full_rank

    start = time.time(); loaded = np.load(args.captures, allow_pickle=False); all_data = {k: loaded[k] for k in loaded.files}
    split_requests = {s: set(map(str, all_data["request_id"][all_data["split"] == s])) for s in ("train", "validation", "test")}
    if any(split_requests[a] & split_requests[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise RuntimeError("request leakage")
    selected = json.loads((args.dense_result / "quantizer_selection.json").read_text())
    paths = {p: records_to_specs(selected[p]["heterogeneous"]) for p in PROJECTIONS}
    reader = gguf.GGUFReader(config["remote"]["gguf_model"]); states = {}
    facts = {"run_id": config["run_id"], "started_unix": start, "host": platform.node(), "gpu": torch.cuda.get_device_name(0), "split_requests": {k: len(v) for k, v in split_requests.items()}, "dense_comparator": "matched full-support three-stage atom-major RRQ", "eta_definition": "literal prompt: (D_Q2-D_dense_atom_RRQ)/(D_Q2-D_selective)", "benefit_retention_definition": "reciprocal intuitive retention: (D_Q2-D_selective)/(D_Q2-D_dense_atom_RRQ)", "test_status": "paired exploratory; PR #2 and dense RRQ outcomes informed this hypothesis", "layers": {}}
    for layer in config["layers"]:
        mask = all_data["layer"] == layer; data = {k: v[mask] for k, v in all_data.items()}
        strata, frequencies = select_experts(data["expert_ids"], data["split"], int(config["experts_per_frequency_stratum"])); experts = sorted(strata)
        reference, storage = load_gguf_experts(reader, layer, experts, gguf)
        train_x = np.asarray(data["x"][data["split"] == "train"], np.float32); moment_x = np.mean(train_x.astype(np.float64) ** 2, axis=0)
        train_h = []
        for ei, expert in enumerate(experts):
            rec, _ = occurrence(data, expert, "train", int(config["max_train_invocations_per_expert"]))
            if len(rec):
                x = np.asarray(data["x"][rec], np.float32); train_h.append(silu(x @ reference["gate"][ei].T) * (x @ reference["up"][ei].T))
        moment_h = np.mean(np.concatenate(train_h).astype(np.float64) ** 2, axis=0)
        base = {p: [] for p in PROJECTIONS}
        for p in PROJECTIONS:
            moment = moment_x if p != "down" else moment_h
            for matrix in reference[p]: base[p].append(encode_legacy_q2(matrix, int(config["base_group_size"]), moment, "fp16").reconstruction)
            base[p] = np.stack(base[p])
        rg = torch.from_numpy(reference["gate"] - base["gate"]).cuda().reshape(-1, 2048); ru = torch.from_numpy(reference["up"] - base["up"]).cuda().reshape(-1, 2048)
        operator = ((rg.T @ rg) + (ru.T @ ru)).cpu().numpy() / len(experts); del rg, ru; torch.cuda.empty_cache()
        transform = generalized_full_rank(train_x, operator)
        proxy, proxy_facts = proxy_gradients(data["xplus"], [data[f"h{h}_router_logits"] for h in range(1, 5)], data["split"])
        state = {"layer": layer, "data": data, "experts": experts, "strata": strata, "frequencies": frequencies, "search_experts": choose_search_experts(experts, strata, data), "reference": reference, "storage": storage, "base": base, "train_x": train_x, "train_h": np.concatenate(train_h), "operator": operator, "transform": transform, "proxy": proxy, "proxy_beta": float(proxy_facts["beta"])}
        states[layer] = state; facts["layers"][str(layer)] = {"experts": experts, "search_experts": state["search_experts"], "proxy": proxy_facts}
        atomic_json(args.output / "run_facts.json", facts)
        print(json.dumps({"prepared_layer": layer, "experts": len(experts)}), flush=True)

    metric_rows: list[dict[str, Any]] = []; action_rows: list[dict[str, Any]] = []; support_rows: list[dict[str, Any]] = []; ranking_rows: list[dict[str, Any]] = []; basis_rows: list[dict[str, Any]] = []
    checkpoint = args.output / "selective_metrics.parquet"; completed: set[tuple[int, int]] = set()
    if checkpoint.exists():
        old = pd.read_parquet(checkpoint); metric_rows = old.to_dict("records"); completed = set(zip(old.layer.astype(int), old.expert_id.astype(int)))
        for name, target in (("selected_actions.parquet", action_rows), ("stage_support.parquet", support_rows), ("ranking_regret.parquet", ranking_rows)):
            path = args.output / name
            if path.exists(): target.extend(pd.read_parquet(path).to_dict("records"))
    page_size = int(config["page_size"]); budgets = np.asarray(config["physical_budgets_bpw"], np.float64); option_rates = np.arange(0.0, 3.0001, 0.25)
    for layer, state in states.items():
        for ei, expert in enumerate(state["experts"]):
            if (layer, expert) in completed: continue
            items = {}
            for projection in PROJECTIONS:
                transform = state["transform"] if projection != "down" else identity_transform(512)
                items[projection] = atom_series(state["reference"][projection][ei], state["base"][projection][ei], transform, paths[projection])
            reference_bytes = sum(state["storage"][p]["bytes_per_expert"] for p in PROJECTIONS)
            correction_capacity = sum(sum(stage.serialized_bytes() for stage in items[p]["series"].stages) for p in PROJECTIONS)
            storage_multiplier = (reference_bytes + correction_capacity) / reference_bytes
            if storage_multiplier > 5.0: raise RuntimeError((layer, expert, storage_multiplier))
            for split, maximum in (("validation", int(config["max_validation_invocations_per_expert"])), ("test", int(config["max_test_invocations_per_expert"]))):
                records, ranks = occurrence(state["data"], expert, split, maximum)
                for sample, record in enumerate(records):
                    x = np.asarray(state["data"]["x"][record], np.float32)
                    gref = state["reference"]["gate"][ei] @ x; uref = state["reference"]["up"][ei] @ x; href = silu(gref) * uref; yref = state["reference"]["down"][ei] @ href
                    gbase = state["base"]["gate"][ei] @ x; ubase = state["base"]["up"][ei] @ x; hbase = silu(gbase) * ubase; ybase = state["base"]["down"][ei] @ hbase
                    base_damage = damage(yref - ybase, state["proxy"], state["proxy_beta"])
                    coefficients = {"gate": items["gate"]["analysis"] @ x, "up": items["up"]["analysis"] @ x, "down": href}
                    dense_corrections = {p: full_outputs(items[p], x[None])[3][0] for p in ("gate", "up")}
                    gdense = gbase + dense_corrections["gate"]; udense = ubase + dense_corrections["up"]; hdense = silu(gdense) * udense
                    dense_down_matrix = sum(items["down"]["stages"])
                    ydense = state["base"]["down"][ei] @ hdense + dense_down_matrix @ hdense
                    dense_damage = damage(yref - ydense, state["proxy"], state["proxy_beta"])
                    orders = {p: nested_energy_actions(items[p], coefficients[p], page_size) for p in PROJECTIONS}
                    snap = {p: snapshots(items[p], coefficients[p], orders[p], option_rates) for p in ("gate", "up")}
                    snap["down"] = matrix_snapshots(items["down"], orders["down"], option_rates)
                    # Linearized separable shortlist, followed by exact sequential evaluation.
                    delta = {}
                    for p in PROJECTIONS:
                        delta[p] = []
                        for option in snap[p]:
                            if p == "gate": candidate = state["base"]["down"][ei] @ (silu(gbase + option["correction"]) * ubase)
                            elif p == "up": candidate = state["base"]["down"][ei] @ (silu(gbase) * (ubase + option["correction"]))
                            else: candidate = ybase + option["matrix"] @ hbase
                            delta[p].append(candidate - ybase)
                    for budget in budgets:
                        candidates = []
                        for ig, rg in enumerate(option_rates):
                            for iu, ru in enumerate(option_rates):
                                remaining = budget - rg - ru
                                if remaining < -1e-9: continue
                                ids = np.flatnonzero(option_rates <= remaining + 1e-9)
                                for idn in ids:
                                    approx = ybase + delta["gate"][ig] + delta["up"][iu] + delta["down"][idn]
                                    candidates.append((damage(yref - approx, state["proxy"], state["proxy_beta"]), ig, iu, int(idn)))
                        candidates.sort(key=lambda row: row[0]); best = None
                        for _, ig, iu, idn in candidates[:16]:
                            sg, su, sd = snap["gate"][ig], snap["up"][iu], snap["down"][idn]
                            gh = gbase + sg["correction"]; uh = ubase + su["correction"]; hh = silu(gh) * uh
                            yh = state["base"]["down"][ei] @ hh + sd["matrix"] @ hh
                            current_damage = damage(yref - yh, state["proxy"], state["proxy_beta"])
                            candidate = (current_damage, ig, iu, idn, yh)
                            if best is None or candidate[0] < best[0]: best = candidate
                        current_damage, ig, iu, idn, yh = best; sg, su, sd = snap["gate"][ig], snap["up"][iu], snap["down"][idn]
                        physical = sg["physical"] + su["physical"] + sd["physical"]; logical = sg["logical"] + su["logical"] + sd["logical"]
                        recovery = 1.0 - current_damage / max(base_damage, 1e-30); dense_recovery = 1.0 - dense_damage / max(base_damage, 1e-30)
                        router_rank = int(ranks[sample])
                        sorted_logits = np.sort(state["data"]["router_logits"][record])
                        common = {"request_id": str(state["data"]["request_id"][record]), "sequence_id": str(state["data"]["sequence_id"][record]), "position": int(state["data"]["position"][record]), "split": split, "layer": layer, "expert_id": expert, "expert_stratum": state["strata"][expert], "expert_train_frequency": state["frequencies"][expert], "router_rank": router_rank + 1, "router_coefficient": float(state["data"]["router_weights"][record, router_rank]), "router_boundary_margin": float(sorted_logits[-8] - sorted_logits[-9]), "base_type": "serialized_fp16_scale_q2_g64", "basis_policy": "generalized_gate_generalized_up_native_down", "layout_id": "atom_major_page_aligned_512_v1", "replica_id": 0, "budget_bpw": float(budget)}
                        dense_gain = base_damage - dense_damage
                        selective_gain = base_damage - current_damage
                        eta = float("nan") if abs(selective_gain) <= 1e-30 else dense_gain / selective_gain
                        retained_fraction = float("nan") if abs(dense_gain) <= 1e-30 else selective_gain / dense_gain
                        metric_rows.append({**common, "method": "selective_atom_rrq_energy_linearized_top16_exact", "recovery": recovery, "dense_comparator_recovery": dense_recovery, "eta": eta, "benefit_retention": retained_fraction, "base_damage": base_damage, "damage": current_damage, "dense_damage": dense_damage, "logical_bytes": logical, "physical_bytes": physical, "logical_bpw": 8 * logical / EXPERT_WEIGHTS, "physical_bpw": 8 * physical / EXPERT_WEIGHTS, "page_amplification": physical / max(logical, 1), "gate_bpw": 8 * sg["physical"] / EXPERT_WEIGHTS, "up_bpw": 8 * su["physical"] / EXPERT_WEIGHTS, "down_bpw": 8 * sd["physical"] / EXPERT_WEIGHTS, "gate_actions": sg["actions"], "up_actions": su["actions"], "down_actions": sd["actions"], "storage_multiplier": storage_multiplier, "relative_output_error": float(np.linalg.norm(yref - yh) / max(np.linalg.norm(yref), 1e-30))})
                        if split == "test":
                            for projection, chosen in (("gate", sg), ("up", su), ("down", sd)):
                                for sequence_index, action in enumerate(orders[projection][:chosen["actions"]]):
                                    pages = [f"{projection}:s{action['stage']}:a{action['atom_id']}:p{p}" for p in range(action["pages"])]
                                    action_rows.append({**common, "projection": projection, "basis_type": "generalized" if projection in ("gate", "up") else "native", "sequence_index": sequence_index, "atom_id": action["atom_id"], "stage": action["stage"], "precision_after_bits": 2 * action["stage"], "marginal_score_per_physical_byte": action["score"], "logical_increment_bytes": action["logical"], "physical_increment_bytes": action["physical"], "page_ids_json": json.dumps(pages)})
                    if split == "test":
                        for projection in PROJECTIONS:
                            z = coefficients[projection]
                            for stage_index, stage in enumerate(items[projection]["stages"], 1):
                                contributions = (stage * z[None, :]).T
                                thresholds, _ = exact_stage_support(contributions, list(map(float, config["support_targets"])), False)
                                packet = stage_packet_facts(items[projection]["series"], page_size)[stage_index - 1]
                                for target, k in thresholds.items(): support_rows.append({"request_id": str(state["data"]["request_id"][record]), "layer": layer, "expert_id": expert, "projection": projection, "stage": stage_index, "ranking": "diagonal_energy", "target_recovery": target, "required_atoms": k, "total_atoms": len(z), "required_physical_bpw_projection": None if k is None else 8 * k * packet["physical"] / MATRIX_WEIGHTS, "required_physical_bpw_expert_average": None if k is None else 8 * k * packet["physical"] / EXPERT_WEIGHTS})
                        if expert == state["search_experts"][0] and sample < int(config["ranking_audit_invocations_per_layer"]):
                            for point in complete_queue_greedy(items, orders, coefficients, gbase, ubase, state["base"]["down"][ei], yref, state["proxy"], state["proxy_beta"], budgets):
                                ranking_rows.append({"request_id": str(state["data"]["request_id"][record]), "layer": layer, "expert_id": expert, "projection": "complete_expert", "stage": 0, "comparison": "complete_expert_queue_greedy", "target_recovery": None, **point})
                            for projection in PROJECTIONS:
                                z = coefficients[projection]
                                for stage_index, stage in enumerate(items[projection]["stages"], 1):
                                    contributions = (stage * z[None, :]).T
                                    energy_k, energy_curve = exact_stage_support(contributions, list(map(float, config["support_targets"])), False)
                                    exact_k, exact_curve = exact_stage_support(contributions, list(map(float, config["support_targets"])), True)
                                    packet_bytes = stage_packet_facts(items[projection]["series"], page_size)[stage_index - 1]["physical"]
                                    for target in map(float, config["support_targets"]):
                                        extra_atoms = None if energy_k[target] is None or exact_k[target] is None else energy_k[target] - exact_k[target]
                                        ranking_rows.append({"request_id": str(state["data"]["request_id"][record]), "layer": layer, "expert_id": expert, "projection": projection, "stage": stage_index, "comparison": "energy_vs_exact_residual_greedy", "target_recovery": target, "energy_atoms": energy_k[target], "exact_atoms": exact_k[target], "extra_atoms": extra_atoms, "packet_physical_bytes": packet_bytes, "extra_projection_bpw": None if extra_atoms is None else 8 * extra_atoms * packet_bytes / MATRIX_WEIGHTS})
                                    for width in config["beam_widths"]:
                                        for point in beam_prefix(contributions, int(width), int(config["beam_candidate_pool"])):
                                            ranking_rows.append({"request_id": str(state["data"]["request_id"][record]), "layer": layer, "expert_id": expert, "projection": projection, "stage": stage_index, "comparison": f"bounded_beam_w{width}_pool{config['beam_candidate_pool']}", "target_recovery": None, "k": point["k"], "recovery": point["recovery"]})
            atomic_parquet(checkpoint, metric_rows); atomic_parquet(args.output / "selected_actions.parquet", action_rows); atomic_parquet(args.output / "stage_support.parquet", support_rows); atomic_parquet(args.output / "ranking_regret.parquet", ranking_rows)
            print(json.dumps({"completed_layer": layer, "expert": expert, "metrics": len(metric_rows), "actions": len(action_rows)}), flush=True); gc.collect(); torch.cuda.empty_cache()
        audit_bases(state, paths, config, basis_rows); atomic_parquet(args.output / "basis_full_support.parquet", basis_rows)
    facts["wall_seconds"] = time.time() - start; facts["peak_cuda_bytes"] = int(torch.cuda.max_memory_allocated()); facts["rows"] = {"metrics": len(metric_rows), "actions": len(action_rows), "support": len(support_rows), "ranking": len(ranking_rows), "basis": len(basis_rows)}
    atomic_json(args.output / "run_facts.json", facts)
    print(json.dumps({"selective_rrq_complete": True, "wall_seconds": facts["wall_seconds"], "rows": facts["rows"]}), flush=True)


if __name__ == "__main__":
    main()
