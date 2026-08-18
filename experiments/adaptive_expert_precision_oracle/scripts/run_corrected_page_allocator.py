#!/usr/bin/env python3
"""Bounded Q2 per-atom, exact-ID, complete-expert page allocator.

The main allocator uses exact final sequential-expert marginal damage after a
local candidate screen.  A small control reruns the first steps without the
screen.  Layout/page comparisons reprice the identical mathematical sequence,
so their quality differences are attributable only to physical packing.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def qenergy(error: torch.Tensor, proxy: torch.Tensor, beta: float) -> torch.Tensor:
    return torch.sum(error * error, dim=-1) + beta * torch.sum((error @ proxy) ** 2, dim=-1)


def tensor_levels(atoms: torch.Tensor) -> dict[int, torch.Tensor]:
    maximum = torch.amax(torch.abs(atoms), dim=0, keepdim=True)
    scale = torch.where(maximum > 0, maximum / 32767.0, torch.ones_like(maximum))
    master = torch.clamp(torch.round(atoms / scale), -32767, 32767).to(torch.int32)
    result = {0: torch.zeros_like(atoms)}
    for bits in (2, 4, 8, 16):
        shift = 16 - bits
        result[bits] = ((master >> shift) << shift).float() * scale
    return result


def incidence(codes: np.ndarray, atoms: np.ndarray, top: int = 64) -> tuple[np.ndarray, np.ndarray]:
    score = codes.astype(np.float64) ** 2 * np.sum(atoms.astype(np.float64) ** 2, axis=0)[None]
    keep = min(top, score.shape[1])
    ids = np.argpartition(score, -keep, axis=1)[:, -keep:]
    selected = np.zeros(score.shape, dtype=np.float32)
    selected[np.arange(len(score))[:, None], ids] = 1.0
    return selected, score.mean(axis=0)


def build_orders(ginc: np.ndarray, uinc: np.ndarray, dinc: np.ndarray, gimp: np.ndarray, uimp: np.ndarray, dimp: np.ndarray, page_size: int):
    from oracle_study.page_allocator import hypergraph_layout, importance_layout, pairwise_layout
    return {
        "naive": {"gate": np.arange(ginc.shape[1]), "up": np.arange(uinc.shape[1]), "down": np.arange(dinc.shape[1])},
        "importance": {"gate": importance_layout(gimp), "up": importance_layout(uimp), "down": importance_layout(dimp)},
        "pairwise": {"gate": pairwise_layout(ginc, gimp), "up": pairwise_layout(uinc, uimp), "down": pairwise_layout(dinc, dimp)},
        "hypergraph": {
            "gate": hypergraph_layout(ginc, max(page_size // 142, 1), gimp),
            "up": hypergraph_layout(uinc, max(page_size // 142, 1), uimp),
            "down": hypergraph_layout(dinc, max(page_size // 526, 1), dimp),
        },
    }


def candidate_delta(levels: dict[int, torch.Tensor], state: np.ndarray, codes: torch.Tensor, ids: np.ndarray) -> torch.Tensor:
    columns = []
    bits = (0, 2, 4, 8, 16)
    for atom in ids.tolist():
        before_index = int(state[atom])
        columns.append((levels[bits[before_index + 1]][:, atom] - levels[bits[before_index]][:, atom]) * codes[atom])
    return torch.stack(columns, dim=1) if columns else torch.empty((levels[0].shape[0], 0), device=codes.device)


def candidate_delta_to_full(levels: dict[int, torch.Tensor], state: np.ndarray, codes: torch.Tensor, ids: np.ndarray) -> torch.Tensor:
    """Use the final stored prefix for screening, avoiding a bad 2-bit gate."""
    bits = (0, 2, 4, 8, 16)
    columns = [
        (levels[16][:, int(atom)] - levels[bits[int(state[int(atom)])]][:, int(atom)]) * codes[int(atom)]
        for atom in ids.tolist()
    ]
    return torch.stack(columns, dim=1) if columns else torch.empty((levels[0].shape[0], 0), device=codes.device)


def key_pages(layouts, key, selected_pages):
    best = None
    best_new = None
    for layout in layouts:
        try:
            pages = layout.pages_for(key)
        except KeyError:
            continue
        new = pages - selected_pages
        if best_new is None or len(new) < len(best_new):
            best, best_new = pages, new
    return best, best_new


def upgrade_path(projection: str, atom: int, before_index: int, after_index: int):
    from oracle_study.page_allocator import IncrementKey, PRECISIONS
    return [
        IncrementKey(projection, atom, PRECISIONS[index], PRECISIONS[index + 1])
        for index in range(before_index, after_index)
    ]


def path_pages(layouts, keys, selected_pages):
    """Choose the cheapest replica independently for every nested bitplane."""
    accumulated = set(selected_pages)
    introduced = set()
    for key in keys:
        packet_pages, _ = key_pages(layouts, key, accumulated)
        if packet_pages is None:
            return None, None
        new = packet_pages - accumulated
        introduced |= new
        accumulated |= packet_pages
    return introduced, accumulated


def greedy_allocate(
    sample_meta: dict[str, Any], x: torch.Tensor, ref: dict[str, torch.Tensor], base: dict[str, torch.Tensor],
    levels: dict[str, dict[int, torch.Tensor]], codes: dict[str, torch.Tensor], layouts: list[Any],
    proxy: torch.Tensor, beta: float, maximum_bytes: int, screen: int, maximum_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from oracle_study.page_allocator import IncrementKey, PRECISIONS
    state = {"gate": np.zeros(2048, dtype=np.int8), "up": np.zeros(2048, dtype=np.int8), "down": np.zeros(512, dtype=np.int8)}
    g = base["gate"] @ x
    u = base["up"] @ x
    dcor = torch.zeros_like(base["down"])
    h = silu(g) * u
    deff = base["down"] + dcor
    y = deff @ h
    yref = ref["down"] @ (silu(ref["gate"] @ x) * (ref["up"] @ x))
    ybase = y.clone()
    denominator = float(qenergy((yref - ybase)[None], proxy, beta)[0])
    pages: set[Any] = set()
    logical = 0
    sequence: list[dict[str, Any]] = []
    current_damage = float(qenergy((yref - y)[None], proxy, beta)[0])

    for step in range(maximum_steps):
        # Sequential-realistic down selection must use the current SwiGLU
        # intermediate. Accepted gate/up increments change h, so both new and
        # already-resident down atoms must use that updated code rather than
        # the authoritative-input h_ref.
        codes["down"] = h
        screened: list[tuple[str, int]] = []
        for projection, current, target in (("gate", g, ref["gate"] @ x), ("up", u, ref["up"] @ x)):
            valid = np.flatnonzero(state[projection] < 4)
            delta = candidate_delta_to_full(levels[projection], state[projection], codes[projection], valid)
            error = target - current
            gain = 2 * (error[:, None] * delta).sum(dim=0) - (delta * delta).sum(dim=0)
            count = min(screen, len(valid))
            chosen = torch.topk(gain, count).indices.cpu().numpy() if count else np.empty(0, dtype=np.int64)
            screened.extend((projection, int(valid[index])) for index in chosen)
        valid = np.flatnonzero(state["down"] < 4)
        delta = candidate_delta_to_full(levels["down"], state["down"], codes["down"], valid)
        candidate_y = y[None] + delta.T
        damage = qenergy(yref[None] - candidate_y, proxy, beta)
        count = min(screen, len(valid))
        chosen = torch.topk(-damage, count).indices.cpu().numpy() if count else np.empty(0, dtype=np.int64)
        screened.extend(("down", int(valid[index])) for index in chosen)

        candidates = []
        for projection, atom in screened:
            before_index = int(state[projection][atom])
            before = PRECISIONS[before_index]
            for after_index in range(before_index + 1, len(PRECISIONS)):
                after = PRECISIONS[after_index]
                keys = upgrade_path(projection, atom, before_index, after_index)
                new_pages, candidate_pages = path_pages(layouts, keys, pages)
                if candidate_pages is None:
                    continue
                physical_increment = len(new_pages) * layouts[0].page_size
                if len(candidate_pages) * layouts[0].page_size > maximum_bytes:
                    continue
                logical_increment = sum(layouts[0].logical_payload_bytes(key) for key in keys)
                delta = (levels[projection][after][:, atom] - levels[projection][before][:, atom]) * codes[projection][atom]
                if projection == "gate":
                    hnew = silu(g + delta) * u
                    ynew = deff @ hnew
                elif projection == "up":
                    hnew = silu(g) * (u + delta)
                    ynew = deff @ hnew
                else:
                    hnew = h
                    ynew = y + delta
                damage_new = float(qenergy((yref - ynew)[None], proxy, beta)[0])
                gain = current_damage - damage_new
                if gain <= 0:
                    continue
                utility = gain / max(physical_increment, 1)
                candidates.append((utility, gain, projection, atom, before, after, after_index, keys, candidate_pages, new_pages, logical_increment, delta, hnew, ynew, damage_new))
        if not candidates:
            break
        selected = max(candidates, key=lambda value: (value[0], value[1]))
        _, gain, projection, atom, before, after, after_index, keys, candidate_pages, new_pages, logical_increment, delta, hnew, ynew, damage_new = selected
        contribution_l2 = float(torch.linalg.norm(ynew - y))
        if projection == "gate":
            g = g + delta
        elif projection == "up":
            u = u + delta
        else:
            # Store the exact progressive atom increment. Dividing the output
            # delta by h[atom] is unstable near zero and loses bitplane
            # identity.
            dcor[:, atom] += levels["down"][after][:, atom] - levels["down"][before][:, atom]
        state[projection][atom] = after_index
        pages = candidate_pages
        logical += logical_increment
        h, y, current_damage = hnew, ynew, damage_new
        deff = base["down"] + dcor
        sequence.append({
            **sample_meta, "step": step + 1, "projection": projection, "atom_id": atom,
            "precision_before": before, "precision_after": after,
            "bitplane_path_json": json.dumps([[key.before_bits, key.after_bits] for key in keys]),
            "logical_incremental_bytes": logical_increment,
            "physical_incremental_bytes": len(new_pages) * layouts[0].page_size,
            "new_page_ids_json": json.dumps(sorted(map(str, new_pages))),
            "all_page_ids_json": json.dumps(sorted(map(str, pages))),
            "marginal_damage_reduction": gain, "marginal_utility_per_physical_byte": gain / max(len(new_pages) * layouts[0].page_size, 1),
            "output_contribution_l2": contribution_l2,
            "cumulative_logical_bytes": logical, "cumulative_physical_bytes": len(pages) * layouts[0].page_size,
            "recovery": 1.0 - current_damage / max(denominator, 1e-20), "damage": current_damage,
        })
    final = {
        "denominator": denominator, "yref_norm": float(torch.linalg.norm(yref)), "sequence": sequence,
        "precision": state, "pages": pages, "logical": logical, "damage": current_damage,
    }
    return sequence, final


def snapshot_rows(meta, sequence, budgets, denominator, yref_norm, weights, method, page_size):
    rows = []
    for budget in budgets:
        allowed = budget * weights / 8
        feasible = [row for row in sequence if row["cumulative_physical_bytes"] <= allowed + 1e-9]
        chosen = feasible[-1] if feasible else None
        damage = denominator if chosen is None else chosen["damage"]
        physical = 0 if chosen is None else chosen["cumulative_physical_bytes"]
        logical = 0 if chosen is None else chosen["cumulative_logical_bytes"]
        selected = [] if chosen is None else sequence[: chosen["step"]]
        precision = {}
        for item in selected:
            precision[f"{item['projection']}:{item['atom_id']}"] = item["precision_after"]
        rows.append({
            **meta, "phase": "A_corrected_complete_expert", "method": method, "budget_bpw": float(budget),
            "page_size": int(page_size), "logical_bytes": int(logical), "physical_bytes": int(physical),
            "logical_bpw": 8 * logical / weights, "physical_bpw": 8 * physical / weights,
            "page_amplification": physical / max(logical, 1), "selected_increments": len(selected),
            "selected_atom_precision_json": json.dumps(precision, sort_keys=True),
            "selected_page_ids_json": "[]" if chosen is None else chosen["all_page_ids_json"],
            "recovery": 1.0 - damage / max(denominator, 1e-20), "damage": damage, "base_damage": denominator,
            "relative_output_error_proxy_bound": math.sqrt(max(damage, 0.0)) / max(yref_norm, 1e-20),
        })
    return rows


def reprice_sequence(meta, sequence, layouts, budgets, denominator, yref_norm, weights, method):
    from oracle_study.page_allocator import IncrementKey
    pages = set()
    repriced = []
    logical = 0
    for item in sequence:
        path = json.loads(item.get("bitplane_path_json", "null"))
        keys = [IncrementKey(item["projection"], int(item["atom_id"]), int(before), int(after)) for before, after in path] if path else [IncrementKey(item["projection"], int(item["atom_id"]), int(item["precision_before"]), int(item["precision_after"]))]
        _, candidate = path_pages(layouts, keys, pages)
        if candidate is None:
            continue
        pages = candidate
        logical += sum(layouts[0].logical_payload_bytes(key) for key in keys)
        value = dict(item)
        value["cumulative_physical_bytes"] = len(pages) * layouts[0].page_size
        value["cumulative_logical_bytes"] = logical
        value["all_page_ids_json"] = json.dumps(sorted(map(str, pages)))
        repriced.append(value)
    return snapshot_rows(meta, repriced, budgets, denominator, yref_norm, weights, method, layouts[0].page_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    experiment = args.config.resolve().parents[1]
    sys.path[:0] = [str(experiment / "src"), str(experiment / "scripts"), config["remote"]["gguf_python"]]
    import gguf
    from oracle_study.page_allocator import PacketLayout, regime_layouts, storage_multiplier
    from oracle_study.weights import alignment_statistics, load_gguf_experts, load_hf_experts
    from run_phase_a_remote import proxy_gradients, select_experts
    from run_phase_a_v2_remote import generalized_full_rank
    from run_progressive_q2_remote import q2_quantize

    args.output.mkdir(parents=True, exist_ok=True)
    loaded = np.load(args.captures, allow_pickle=False)
    data = {name: loaded[name] for name in loaded.files}
    train_requests = set(map(str, data["request_id"][data["split"] == "train"]))
    validation_requests = set(map(str, data["request_id"][data["split"] == "validation"]))
    test_requests = set(map(str, data["request_id"][data["split"] == "test"]))
    if train_requests & validation_requests or train_requests & test_requests or validation_requests & test_requests:
        raise ValueError("request-level split leakage")
    reader = gguf.GGUFReader(config["remote"]["gguf_model"])
    device = "cuda"
    metrics = pd.read_parquet(args.output / "allocator_metrics.parquet").to_dict("records") if (args.output / "allocator_metrics.parquet").exists() else []
    sequences = pd.read_parquet(args.output / "selected_increment_sequence.parquet").to_dict("records") if (args.output / "selected_increment_sequence.parquet").exists() else []
    storage_rows, quant_rows = [], []
    completed = {
        (str(row["request_id"]), int(row["position"]), int(row["layer"]), int(row["expert_id"]))
        for row in metrics if row.get("method") == "A6_corrected_screened_greedy"
    }
    facts: dict[str, Any] = {"started_unix": time.time(), "layers": {}, "split_requests": {"train": len(train_requests), "validation": len(validation_requests), "test": len(test_requests)}}
    budgets = [float(v) for v in config["physical_bpw_budgets"]]
    weights = 3 * 2048 * 512

    for layer in config["layers"]:
        layer_start = time.time()
        mask = data["layer"] == layer
        d = {name: value[mask] for name, value in data.items()}
        strata, frequencies = select_experts(d["expert_ids"], d["split"], int(config["experts_per_frequency_stratum"]))
        experts = sorted(strata)
        reference, reference_storage = load_gguf_experts(reader, layer, experts, gguf)
        source = load_hf_experts(Path(config["remote"]["hf_model"]), layer, experts)
        alignment = alignment_statistics(source, reference)
        if min(value["bf16_q4_cosine"] for value in alignment.values()) < 0.98:
            raise ValueError("reference/source alignment failed")
        del source
        train_x = np.asarray(d["x"][d["split"] == "train"], dtype=np.float32)
        moment_x = np.mean(train_x.astype(np.float64) ** 2, axis=0)
        base = {"gate": [], "up": []}
        train_h_by_expert = []
        for ei, expert in enumerate(experts):
            base["gate"].append(q2_quantize(reference["gate"][ei], int(config["binary_group_size"]), moment_x)[0])
            base["up"].append(q2_quantize(reference["up"][ei], int(config["binary_group_size"]), moment_x)[0])
            records = np.unique(np.argwhere((d["expert_ids"] == expert) & (d["split"][:, None] == "train"))[:, 0])[: int(config["max_train_invocations_per_expert"])]
            xt = np.asarray(d["x"][records], dtype=np.float32)
            gt, ut = xt @ reference["gate"][ei].T, xt @ reference["up"][ei].T
            train_h_by_expert.append((gt / (1 + np.exp(-gt))) * ut)
        base["gate"], base["up"] = np.stack(base["gate"]), np.stack(base["up"])
        all_h = np.concatenate(train_h_by_expert)
        moment_h = np.mean(all_h.astype(np.float64) ** 2, axis=0)
        base["down"] = np.stack([q2_quantize(reference["down"][ei], int(config["binary_group_size"]), moment_h)[0] for ei in range(len(experts))])
        rg = torch.from_numpy(reference["gate"] - base["gate"]).to(device).reshape(-1, 2048)
        ru = torch.from_numpy(reference["up"] - base["up"]).to(device).reshape(-1, 2048)
        transform = generalized_full_rank(train_x, (((rg.T @ rg) + (ru.T @ ru)) / len(experts)).cpu().numpy())
        del rg, ru
        proxy_np, proxy_facts = proxy_gradients(d["xplus"], [d[f"h{h}_router_logits"] for h in range(1, 5)], d["split"])
        proxy = torch.from_numpy(proxy_np).to(device)
        beta = float(proxy_facts["beta"])
        layer_invocations = 0
        exact_strata_done: set[str] = {
            str(row["expert_stratum"]) for row in metrics
            if int(row.get("layer", -1)) == layer and row.get("method") == "bounded_exact_complete_marginal"
        }

        for ei, expert in enumerate(experts):
            train_records = np.unique(np.argwhere((d["expert_ids"] == expert) & (d["split"][:, None] == "train"))[:, 0])[: int(config["max_train_invocations_per_expert"])]
            tx = np.asarray(d["x"][train_records], dtype=np.float32)
            th = train_h_by_expert[ei]
            rg_np = reference["gate"][ei] - base["gate"][ei]
            ru_np = reference["up"][ei] - base["up"][ei]
            rd_np = reference["down"][ei] - base["down"][ei]
            gatoms, uatoms, datoms = rg_np @ transform.synthesis, ru_np @ transform.synthesis, rd_np
            tcodes = tx @ transform.analysis.T
            ginc, gimp = incidence(tcodes, gatoms)
            uinc, uimp = incidence(tcodes, uatoms)
            dinc, dimp = incidence(th, datoms)
            orders = build_orders(ginc, uinc, dinc, gimp, uimp, dimp, 4096)
            lengths = {"gate": 512, "up": 512, "down": 2048}
            layout_by_kind = {kind: PacketLayout(order, lengths, 4096, f"L{layer}E{expert}_{kind}") for kind, order in orders.items()}
            primary = layout_by_kind["hypergraph"]
            replica_orders_g, _ = regime_layouts(np.maximum(ginc, uinc), 4, int(config["seed"]) + layer + expert)
            replica_orders_d, _ = regime_layouts(dinc, 4, int(config["seed"]) + 1000 + layer + expert)
            replicas = [primary]
            for replica in range(1, 4):
                replica_order = {"gate": replica_orders_g[replica], "up": replica_orders_g[replica], "down": replica_orders_d[replica]}
                replicas.append(PacketLayout(replica_order, lengths, 4096, f"L{layer}E{expert}_replica{replica}", active_counts={"gate": 512, "up": 512, "down": 128}))
            bundled = PacketLayout(orders["hypergraph"], lengths, 4096, f"L{layer}E{expert}_bundled", bundle_gate_up=True)
            reference_bytes = sum(value["bytes_per_expert"] for value in reference_storage.values())
            storage = storage_multiplier(reference_bytes, replicas, [16, 2, 2, 2], resident_basis_bytes=transform.analysis.nbytes)
            storage.update({"layer": layer, "expert_id": expert, "replicas": 4, "partial_replica_fraction": 0.25})
            storage_rows.append(storage)
            if storage["external_multiplier"] > float(config["external_storage_cap_reference_multiplier"]):
                raise ValueError(f"storage cap exceeded: {storage}")

            occurrence = np.argwhere((d["expert_ids"] == expert) & (d["split"][:, None] == "test"))[: int(config["max_test_invocations_per_expert"])]
            layer_invocations += len(occurrence)
            ref_t = {name: torch.from_numpy(reference[name][ei]).to(device) for name in ("gate", "up", "down")}
            base_t = {name: torch.from_numpy(base[name][ei]).to(device) for name in ("gate", "up", "down")}
            atoms_t = {"gate": torch.from_numpy(gatoms).to(device), "up": torch.from_numpy(uatoms).to(device), "down": torch.from_numpy(datoms).to(device)}
            levels = {name: tensor_levels(value) for name, value in atoms_t.items()}
            for name, value in atoms_t.items():
                for bits in (2, 4, 8, 16):
                    error = float(torch.sum((levels[name][bits] - value) ** 2) / torch.clamp(torch.sum(value * value), min=1e-20))
                    unique = int(torch.unique(torch.round(levels[name][bits] / torch.clamp(torch.amax(torch.abs(value), dim=0, keepdim=True) / 32767, min=1e-30))).numel())
                    quant_rows.append({"layer": layer, "expert_id": expert, "projection": name, "bits": bits, "relative_atom_mse": error, "observed_master_prefix_levels": unique})
            for sample_index, (record, rank) in enumerate(occurrence.tolist()):
                x = torch.from_numpy(np.asarray(d["x"][record], dtype=np.float32)).to(device)
                codes = {"gate": x @ torch.from_numpy(transform.analysis).to(device).T, "up": x @ torch.from_numpy(transform.analysis).to(device).T}
                gref, uref = ref_t["gate"] @ x, ref_t["up"] @ x
                href = silu(gref) * uref
                codes["down"] = href
                logits = np.sort(d["router_logits"][record])
                meta = {"request_id": str(d["request_id"][record]), "sequence_id": str(d["sequence_id"][record]), "position": int(d["position"][record]), "layer": int(layer), "expert_id": int(expert), "expert_stratum": strata[expert], "expert_train_frequency": frequencies[expert], "router_rank": int(rank + 1), "router_coefficient": float(d["router_weights"][record, rank]), "router_boundary_margin": float(logits[-8] - logits[-9]), "base_type": "q2", "basis_type": "generalized/generalized/native"}
                invocation_key = (meta["request_id"], meta["position"], meta["layer"], meta["expert_id"])
                if invocation_key in completed:
                    print(json.dumps({"run": "corrected_allocator", "layer": layer, "expert": expert, "sample": sample_index, "resumed_skip": True}), flush=True)
                    continue
                maximum = int(max(budgets) * weights / 8)
                sequence, final = greedy_allocate(meta, x, ref_t, base_t, levels, codes, [primary], proxy, beta, maximum, int(config["screen_candidates_per_projection"]), 800)
                sequences.extend(sequence)
                metrics.extend(snapshot_rows(meta, sequence, budgets, final["denominator"], final["yref_norm"], weights, "A6_corrected_screened_greedy", 4096))
                for kind, layout in layout_by_kind.items():
                    metrics.extend(reprice_sequence(meta, sequence, [layout], budgets, final["denominator"], final["yref_norm"], weights, f"A4_fixed_sequence_{kind}"))
                metrics.extend(reprice_sequence(meta, sequence, replicas[:2], budgets, final["denominator"], final["yref_norm"], weights, "A5_two_layout_H0"))
                metrics.extend(reprice_sequence(meta, sequence, replicas, budgets, final["denominator"], final["yref_norm"], weights, "A5_four_partial_layout_H0"))
                metrics.extend(reprice_sequence(meta, sequence, [bundled], budgets, final["denominator"], final["yref_norm"], weights, "A4_gate_up_bundled"))
                for page_size in (512, 1024, 2048):
                    layout = PacketLayout(orders["hypergraph"], lengths, page_size, f"L{layer}E{expert}_hypergraph_{page_size}")
                    metrics.extend(reprice_sequence(meta, sequence, [layout], budgets, final["denominator"], final["yref_norm"], weights, f"A3_pages_{page_size}"))
                if sample_index == 0 and strata[expert] not in exact_strata_done:
                    exact_sequence, exact_final = greedy_allocate(meta, x, ref_t, base_t, levels, codes, [primary], proxy, beta, maximum, 2048, int(config["exact_greedy_steps"]))
                    metrics.extend(snapshot_rows(meta, exact_sequence, budgets, exact_final["denominator"], exact_final["yref_norm"], weights, "bounded_exact_complete_marginal", 4096))
                    exact_strata_done.add(strata[expert])
                print(json.dumps({"run": "corrected_allocator", "layer": layer, "expert": expert, "sample": sample_index, "steps": len(sequence)}), flush=True)
            del ref_t, base_t, atoms_t, levels
            gc.collect(); torch.cuda.empty_cache()
            pd.DataFrame(metrics).to_parquet(args.output / "allocator_metrics.parquet", index=False)
            pd.DataFrame(sequences).to_parquet(args.output / "selected_increment_sequence.parquet", index=False)
        facts["layers"][str(layer)] = {"experts": experts, "test_invocations": layer_invocations, "alignment": alignment, "proxy": proxy_facts, "wall_seconds": time.time() - layer_start}
        del reference, base, train_h_by_expert, all_h, transform, proxy
        gc.collect(); torch.cuda.empty_cache()

    facts["wall_seconds"] = time.time() - facts["started_unix"]
    facts["rows"] = len(metrics)
    (args.output / "allocator_facts.json").write_text(json.dumps(facts, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(metrics).to_parquet(args.output / "allocator_metrics.parquet", index=False)
    pd.DataFrame(sequences).to_parquet(args.output / "selected_increment_sequence.parquet", index=False)
    pd.DataFrame(storage_rows).to_csv(args.output / "storage_accounting.csv", index=False)
    pd.DataFrame(quant_rows).to_csv(args.output / "nested_quantizer_audit.csv", index=False)
    print(json.dumps({"complete": True, "metrics": len(metrics), "increments": len(sequences), "wall_seconds": facts["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
