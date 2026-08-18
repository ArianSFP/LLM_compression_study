#!/usr/bin/env python3
"""Run the stratified projection-level H0 study on the provided CUDA pod."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch


def select_experts(ids: np.ndarray, split: np.ndarray, count: int) -> tuple[dict[int, str], dict[int, int]]:
    train_ids = ids[split == "train"].reshape(-1)
    frequencies = np.bincount(train_ids.astype(np.int64), minlength=256)
    positive = np.flatnonzero(frequencies > 0)
    ranked = positive[np.argsort(frequencies[positive], kind="stable")]
    cold = ranked[:count]
    middle_start = max(0, len(ranked) // 2 - count // 2)
    median = ranked[middle_start : middle_start + count]
    hot = ranked[-count:][::-1]
    strata: dict[int, str] = {}
    for name, values in (("cold", cold), ("median", median), ("hot", hot)):
        for value in values:
            strata[int(value)] = name
    return strata, {int(i): int(v) for i, v in enumerate(frequencies)}


def proxy_gradients(xplus: np.ndarray, future_logits: list[np.ndarray], split: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    mask = split == "train"
    x = np.asarray(xplus[mask], dtype=np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    targets = []
    margins = []
    for logits in future_logits:
        ordered = np.sort(np.asarray(logits[mask], dtype=np.float32), axis=1)
        margin = ordered[:, -8] - ordered[:, -9]
        targets.append(margin - margin.mean())
        margins.append(margin)
    y = np.stack(targets, axis=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    xt = torch.from_numpy(x).to(device)
    yt = torch.from_numpy(y).to(device)
    covariance = xt.T @ xt / max(len(x), 1)
    ridge = float(torch.trace(covariance).item() / covariance.shape[0]) * 1e-2
    rhs = xt.T @ yt / max(len(x), 1)
    gradients = torch.linalg.solve(covariance + ridge * torch.eye(covariance.shape[0], device=device), rhs)
    gradients = gradients / torch.clamp(torch.linalg.norm(gradients, dim=0, keepdim=True), min=1e-12)
    result = gradients.cpu().numpy().astype(np.float32)
    facts = {
        "kind": "ridge gradients of H1-H4 same-layer router top8 boundary margins w.r.t. captured post-MoE state",
        "rank": int(result.shape[1]),
        "ridge": ridge,
        "train_samples": int(mask.sum()),
        "margin_medians": [float(np.median(v)) for v in margins],
        "beta": 512.0,
    }
    return result, facts


def build_transforms(samples: np.ndarray, residual_operator: np.ndarray, seeds: list[int], prefix: str):
    from oracle_study.bases import (
        generalized_transform,
        identity_transform,
        learned_orthogonal_transform,
        pca_transform,
        random_orthogonal_transform,
        two_view_overcomplete,
    )
    dimension = samples.shape[1]
    transforms = {
        "native": identity_transform(dimension),
        "random": random_orthogonal_transform(dimension, 12345 + dimension),
        "pca": pca_transform(samples),
        "generalized": generalized_transform(samples, residual_operator),
    }
    for seed in seeds:
        transform = learned_orthogonal_transform(samples, residual_operator, seed)
        transforms[transform.name] = transform
    transforms["overcomplete_2x"] = two_view_overcomplete(transforms["pca"], transforms[f"learned_s{seeds[0]}"])
    for value in transforms.values():
        value.name = f"{prefix}_{value.name}"
    return transforms


def make_layout(transform: Any, train_samples: np.ndarray, maximum: int = 64) -> np.ndarray:
    codes = np.asarray(train_samples, dtype=np.float32) @ transform.analysis.T
    importance = np.mean(codes.astype(np.float64) ** 2, axis=0)
    # Training-only frequency layout: coordinates often selected together by
    # the global top-support policy are contiguous.
    top = np.argpartition(importance, -min(maximum, len(importance)))[-min(maximum, len(importance)):]
    leading = top[np.argsort(importance[top])[::-1]]
    remaining_mask = np.ones(len(importance), dtype=bool)
    remaining_mask[leading] = False
    trailing = np.flatnonzero(remaining_mask)
    trailing = trailing[np.argsort(importance[trailing])[::-1]]
    return np.concatenate([leading, trailing]).astype(np.int64)


def physical_count(order: np.ndarray, packet_bytes: int, page_size: int, byte_budget: float, layout: np.ndarray) -> tuple[int, int]:
    from oracle_study.quant import page_ids_for_atoms
    pages: set[int] = set()
    count = 0
    for atom in order.tolist():
        new_pages = set(page_ids_for_atoms(np.asarray([atom]), packet_bytes, page_size, layout).tolist())
        union = pages | new_pages
        if len(union) * page_size > byte_budget:
            continue
        pages = union
        count += 1
    return count, len(pages) * page_size


def append_rows(
    rows: list[dict[str, Any]], meta: list[dict[str, Any]], reference: np.ndarray,
    base: np.ndarray, approximation: np.ndarray, method: str, projection: str,
    rate_axis: str, budget_bpw: float, logical_bytes: np.ndarray, physical_bytes: np.ndarray,
    selected_atoms: np.ndarray, num_weights: int, proxy: np.ndarray | None, beta: float,
) -> None:
    from oracle_study.metrics import metric_rows
    values = metric_rows(reference, base, approximation, proxy, beta)
    for index, item in enumerate(meta):
        row = dict(item)
        row.update({
            "phase": "A_projection",
            "projection": projection,
            "method": method,
            "rate_axis": rate_axis,
            "budget_bpw": float(budget_bpw),
            "logical_bytes": int(logical_bytes[index]),
            "physical_bytes_4k": int(physical_bytes[index]),
            "ideal_bpw": float(8 * logical_bytes[index] / num_weights),
            "physical_bpw": float(8 * physical_bytes[index] / num_weights),
            "selected_atoms": int(selected_atoms[index]),
        })
        for name, array in values.items():
            row[name] = float(array[index])
        rows.append(row)


def evaluate_projection(
    residual: np.ndarray,
    weight4: np.ndarray,
    weight1: np.ndarray,
    inputs: np.ndarray,
    transforms: dict[str, Any],
    layouts: dict[str, np.ndarray],
    ideal_budgets: list[float],
    physical_budgets: list[float],
    meta: list[dict[str, Any]],
    projection: str,
    proxy: np.ndarray | None,
    beta: float,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    from oracle_study.metrics import exact_greedy_approximations
    device = "cuda"
    x = torch.from_numpy(np.asarray(inputs, dtype=np.float32)).to(device)
    r_matrix = torch.from_numpy(np.asarray(residual, dtype=np.float32)).to(device)
    w4 = torch.from_numpy(np.asarray(weight4, dtype=np.float32)).to(device)
    w1 = torch.from_numpy(np.asarray(weight1, dtype=np.float32)).to(device)
    y4 = (x @ w4.T).cpu().numpy()
    y1 = (x @ w1.T).cpu().numpy()
    correction = torch.from_numpy(y4 - y1).to(device)
    num_weights = int(np.prod(weight4.shape))
    packet_bytes = int(weight4.shape[0] * 2 + 4)
    diagnostics: dict[str, Any] = {}
    method_outputs: dict[tuple[str, str, float], np.ndarray] = {}

    zeros = np.zeros_like(y4)
    for axis, budgets in (("ideal", ideal_budgets), ("physical", physical_budgets)):
        for budget in budgets:
            append_rows(
                rows, meta, y4, y1, y1, "base_only", projection, axis, budget,
                np.zeros(len(x), dtype=np.int64), np.zeros(len(x), dtype=np.int64),
                np.zeros(len(x), dtype=np.int64), num_weights, proxy, beta,
            )

    for transform_key, transform in transforms.items():
        synthesis = torch.from_numpy(transform.synthesis).to(device)
        analysis = torch.from_numpy(transform.analysis).to(device)
        atoms = r_matrix @ synthesis
        codes = x @ analysis.T
        norms = torch.sum(atoms * atoms, dim=0)
        score = codes * codes * norms[None, :]
        max_ideal = max(int(math.floor(b * num_weights / 8 / packet_bytes)) for b in ideal_budgets)
        max_count = min(max(max_ideal, 1), atoms.shape[1])
        indices = torch.topk(score, max_count, dim=1, sorted=True).indices
        values = torch.gather(codes, 1, indices)
        selected_atoms_tensor = atoms.T[indices]
        cumulative = torch.cumsum(selected_atoms_tensor * values[:, :, None], dim=1)
        layout = layouts[transform_key]
        inverse = np.empty_like(layout)
        inverse[layout] = np.arange(len(layout))
        indices_np = indices.cpu().numpy()

        if transform_key == "generalized" and transform.eigenvalues is not None:
            theoretical = codes.cpu().numpy() ** 2 * transform.eigenvalues[None, :]
            actual = score.cpu().numpy()
            correlations = []
            for left, right in zip(theoretical, actual):
                correlations.append(float(np.corrcoef(left, right)[0, 1]))
            diagnostics["generalized_score_correlation_mean"] = float(np.nanmean(correlations))

        for budget in ideal_budgets:
            count = min(int(math.floor(budget * num_weights / 8 / packet_bytes)), max_count)
            rhat = zeros if count == 0 else cumulative[:, count - 1].cpu().numpy()
            logical = np.full(len(x), count * packet_bytes, dtype=np.int64)
            physical = np.empty(len(x), dtype=np.int64)
            for sample in range(len(x)):
                positions = inverse[indices_np[sample, :count]] if count else np.empty(0, dtype=np.int64)
                pages = set((positions * packet_bytes // 4096).tolist())
                if count:
                    pages.update(((positions * packet_bytes + packet_bytes - 1) // 4096).tolist())
                physical[sample] = len(pages) * 4096
            approx = y1 + rhat
            append_rows(rows, meta, y4, y1, approx, transform_key, projection, "ideal", budget,
                        logical, physical, np.full(len(x), count), num_weights, proxy, beta)
            method_outputs[(transform_key, "ideal", budget)] = approx

        for budget in physical_budgets:
            counts = np.empty(len(x), dtype=np.int64)
            physical = np.empty(len(x), dtype=np.int64)
            logical = np.empty(len(x), dtype=np.int64)
            rhat = np.zeros_like(y4)
            bytes_allowed = budget * num_weights / 8
            for sample in range(len(x)):
                count, actual_bytes = physical_count(indices_np[sample], packet_bytes, 4096, bytes_allowed, layout)
                count = min(count, max_count)
                counts[sample] = count
                physical[sample] = actual_bytes
                logical[sample] = count * packet_bytes
                if count:
                    rhat[sample] = cumulative[sample, count - 1].cpu().numpy()
            approx = y1 + rhat
            append_rows(rows, meta, y4, y1, approx, transform_key, projection, "physical", budget,
                        logical, physical, counts, num_weights, proxy, beta)
            method_outputs[(transform_key, "physical", budget)] = approx

        if transform_key == "native":
            # Required |x| native rule.
            abs_indices = torch.topk(torch.abs(codes), max_count, dim=1, sorted=True).indices
            abs_values = torch.gather(codes, 1, abs_indices)
            abs_cum = torch.cumsum(atoms.T[abs_indices] * abs_values[:, :, None], dim=1)
            for budget in ideal_budgets:
                count = min(int(math.floor(budget * num_weights / 8 / packet_bytes)), max_count)
                rhat = zeros if count == 0 else abs_cum[:, count - 1].cpu().numpy()
                logical = np.full(len(x), count * packet_bytes, dtype=np.int64)
                physical = np.full(len(x), math.ceil(count * packet_bytes / 4096) * 4096, dtype=np.int64)
                append_rows(rows, meta, y4, y1, y1 + rhat, "native_abs", projection, "ideal", budget,
                            logical, physical, np.full(len(x), count), num_weights, proxy, beta)

            # Exact greedy marginal oracle, vectorised across the held-out batch.
            counts = [min(int(math.floor(b * num_weights / 8 / packet_bytes)), atoms.shape[1]) for b in ideal_budgets]
            greedy, greedy_ids = exact_greedy_approximations(atoms, codes, correction, counts)
            greedy_ids_np = greedy_ids.cpu().numpy()
            for budget, count in zip(ideal_budgets, counts):
                rhat = greedy[count].cpu().numpy()
                logical = np.full(len(x), count * packet_bytes, dtype=np.int64)
                physical = np.empty(len(x), dtype=np.int64)
                for sample in range(len(x)):
                    positions = inverse[greedy_ids_np[sample, :count]] if count else np.empty(0, dtype=np.int64)
                    pages = set((positions * packet_bytes // 4096).tolist())
                    if count:
                        pages.update(((positions * packet_bytes + packet_bytes - 1) // 4096).tolist())
                    physical[sample] = len(pages) * 4096
                append_rows(rows, meta, y4, y1, y1 + rhat, "native_greedy", projection, "ideal", budget,
                            logical, physical, np.full(len(x), count), num_weights, proxy, beta)

        del synthesis, analysis, atoms, codes, norms, score, indices, values, selected_atoms_tensor, cumulative
        torch.cuda.empty_cache()

    # H0 two-view oracle selects the lower-damage complete view per invocation.
    first_learned = sorted(k for k in transforms if k.startswith("learned_s"))[0]
    for axis, budgets in (("ideal", ideal_budgets), ("physical", physical_budgets)):
        for budget in budgets:
            candidates = [method_outputs[(name, axis, budget)] for name in ("pca", "generalized", first_learned)]
            damages = np.stack([np.sum((y4 - value) ** 2, axis=1) for value in candidates], axis=1)
            chosen = np.argmin(damages, axis=1)
            approx = np.stack([candidates[chosen[i]][i] for i in range(len(x))])
            # Conservative accounting: one full selected view; external storage holds two.
            source_rows = [
                row for row in rows[-(len(transforms) * (len(ideal_budgets) + len(physical_budgets)) * len(x) * 2 + 1):]
                if row.get("projection") == projection and row.get("rate_axis") == axis and row.get("budget_bpw") == budget
            ]
            logical = np.full(len(x), int(budget * num_weights / 8), dtype=np.int64)
            physical = np.ceil(logical / 4096).astype(np.int64) * 4096
            counts = np.floor(logical / packet_bytes).astype(np.int64)
            append_rows(rows, meta, y4, y1, approx, "two_view_oracle", projection, axis, budget,
                        logical, physical, counts, num_weights, proxy, beta)

    del x, r_matrix, w4, w1, correction
    torch.cuda.empty_cache()
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = args.config.resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, config["remote"]["gguf_python"])
    import gguf
    from oracle_study.quant import binary_quantize, binary_storage_bytes
    from oracle_study.weights import alignment_statistics, load_gguf_experts, load_hf_experts

    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    data_file = np.load(args.captures, allow_pickle=False)
    data = {name: data_file[name] for name in data_file.files}
    reader = gguf.GGUFReader(config["remote"]["gguf_model"])
    rows: list[dict[str, Any]] = []
    facts: dict[str, Any] = {"layers": {}, "cuda": torch.cuda.get_device_name(0), "started_unix": started}
    ideal_budgets = [float(v) for v in config["ideal_bpw_budgets"]]
    physical_budgets = [float(v) for v in config["physical_bpw_budgets"]]
    seeds = [int(v) for v in config["learned_seeds"]]

    for layer in config["layers"]:
        layer_started = time.time()
        mask = data["layer"] == layer
        layer_data = {name: value[mask] for name, value in data.items()}
        strata, frequencies = select_experts(layer_data["expert_ids"], layer_data["split"], int(config["experts_per_frequency_stratum"]))
        experts = sorted(strata)
        print(json.dumps({"layer": layer, "selected_experts": experts, "strata": strata}), flush=True)
        bf16 = load_hf_experts(Path(config["remote"]["hf_model"]), layer, experts)
        q4, storage = load_gguf_experts(reader, layer, experts, gguf)
        alignment = alignment_statistics(bf16, q4)
        if min(v["bf16_q4_cosine"] for v in alignment.values()) < 0.98:
            raise ValueError(f"checkpoint alignment failed for layer {layer}: {alignment}")

        train_x = np.asarray(layer_data["x"][layer_data["split"] == "train"], dtype=np.float32)
        moment_x = np.mean(train_x.astype(np.float64) ** 2, axis=0)
        binary: dict[str, list[np.ndarray]] = {"gate_simple": [], "gate_aware": [], "up_simple": [], "up_aware": []}
        train_hidden: list[np.ndarray] = []
        train_hidden_by_expert: list[np.ndarray] = []
        for expert_index, expert in enumerate(experts):
            for projection in ("gate", "up"):
                simple, _ = binary_quantize(bf16[projection][expert_index], int(config["binary_group_size"]))
                aware, _ = binary_quantize(q4[projection][expert_index], int(config["binary_group_size"]), moment_x)
                binary[f"{projection}_simple"].append(simple)
                binary[f"{projection}_aware"].append(aware)
            occurrence = np.argwhere((layer_data["expert_ids"] == expert) & (layer_data["split"][:, None] == "train"))
            indices = occurrence[:256, 0]
            x_e = np.asarray(layer_data["x"][indices], dtype=np.float32)
            gate = x_e @ q4["gate"][expert_index].T
            up = x_e @ q4["up"][expert_index].T
            hidden = (gate / (1.0 + np.exp(-gate))) * up
            train_hidden.append(hidden)
            train_hidden_by_expert.append(hidden)
        for name in binary:
            binary[name] = np.stack(binary[name])
        all_hidden = np.concatenate(train_hidden, axis=0)
        moment_h = np.mean(all_hidden.astype(np.float64) ** 2, axis=0)
        down_simple, down_aware = [], []
        for expert_index in range(len(experts)):
            simple, _ = binary_quantize(bf16["down"][expert_index], int(config["binary_group_size"]))
            aware, _ = binary_quantize(q4["down"][expert_index], int(config["binary_group_size"]), moment_h)
            down_simple.append(simple)
            down_aware.append(aware)
        binary["down_simple"] = np.stack(down_simple)
        binary["down_aware"] = np.stack(down_aware)

        device = "cuda"
        rg = torch.from_numpy(q4["gate"] - binary["gate_aware"]).to(device).reshape(-1, 2048)
        ru = torch.from_numpy(q4["up"] - binary["up_aware"]).to(device).reshape(-1, 2048)
        h_input_operator = ((rg.T @ rg) + (ru.T @ ru)) / len(experts)
        input_transforms = build_transforms(train_x, h_input_operator.cpu().numpy(), seeds, "input")
        del rg, ru, h_input_operator
        rd = torch.from_numpy(q4["down"] - binary["down_aware"]).to(device).reshape(-1, 512)
        h_down_operator = (rd.T @ rd) / len(experts)
        down_transforms = build_transforms(all_hidden, h_down_operator.cpu().numpy(), seeds, "down")
        del rd, h_down_operator
        torch.cuda.empty_cache()
        input_layouts = {key: make_layout(value, train_x) for key, value in input_transforms.items()}
        down_layouts = {key: make_layout(value, all_hidden) for key, value in down_transforms.items()}
        proxy, proxy_facts = proxy_gradients(
            layer_data["xplus"], [layer_data[f"h{h}_router_logits"] for h in range(1, 5)], layer_data["split"]
        )

        layer_facts = {
            "selected_experts": experts,
            "strata": {str(k): v for k, v in strata.items()},
            "train_frequencies": {str(k): frequencies[k] for k in experts},
            "gguf_storage": storage,
            "alignment": alignment,
            "proxy": proxy_facts,
            "binary_storage": {
                projection: binary_storage_bytes(q4[projection].shape[1:], int(config["binary_group_size"]))
                for projection in ("gate", "up", "down")
            },
            "basis": {},
        }
        for group, transforms in (("input", input_transforms), ("down", down_transforms)):
            for key, value in transforms.items():
                layer_facts["basis"][f"{group}_{key}"] = {
                    "atoms": int(value.analysis.shape[0]),
                    "dimension": int(value.analysis.shape[1]),
                    "retained_rank": value.retained_rank,
                    "condition_number": value.condition_number,
                }

        for expert_index, expert in enumerate(experts):
            occurrence = np.argwhere((layer_data["expert_ids"] == expert) & (layer_data["split"][:, None] == "test"))
            cap = int(config["max_test_invocations_per_expert"])
            occurrence = occurrence[:cap]
            if len(occurrence) == 0:
                continue
            record_indices = occurrence[:, 0]
            router_ranks = occurrence[:, 1]
            inputs = np.asarray(layer_data["x"][record_indices], dtype=np.float32)
            meta = []
            for record_index, rank in zip(record_indices.tolist(), router_ranks.tolist()):
                logits = np.sort(layer_data["router_logits"][record_index])
                meta.append({
                    "sequence_id": str(layer_data["sequence_id"][record_index]),
                    "request_id": str(layer_data["request_id"][record_index]),
                    "position": int(layer_data["position"][record_index]),
                    "layer": int(layer),
                    "expert_id": int(expert),
                    "expert_stratum": strata[expert],
                    "expert_train_frequency": frequencies[expert],
                    "router_rank": int(rank + 1),
                    "router_coefficient": float(layer_data["router_weights"][record_index, rank]),
                    "router_boundary_margin": float(logits[-8] - logits[-9]),
                    "activation_norm": float(np.linalg.norm(layer_data["x"][record_index])),
                })
            for projection in ("gate", "up"):
                transforms = {key.replace("input_", ""): value for key, value in input_transforms.items()}
                layouts = {key: input_layouts[key] for key in input_transforms}
                residual = q4[projection][expert_index] - binary[f"{projection}_aware"][expert_index]
                diagnostics = evaluate_projection(
                    residual, q4[projection][expert_index], binary[f"{projection}_aware"][expert_index],
                    inputs, transforms, layouts, ideal_budgets, physical_budgets, meta,
                    projection, None, 0.0, rows,
                )
                layer_facts.setdefault("diagnostics", {}).update({f"expert_{expert}_{projection}_{k}": v for k, v in diagnostics.items()})
            gate = inputs @ q4["gate"][expert_index].T
            up = inputs @ q4["up"][expert_index].T
            hidden = (gate / (1.0 + np.exp(-gate))) * up
            transforms = {key.replace("down_", ""): value for key, value in down_transforms.items()}
            residual = q4["down"][expert_index] - binary["down_aware"][expert_index]
            diagnostics = evaluate_projection(
                residual, q4["down"][expert_index], binary["down_aware"][expert_index],
                hidden, transforms, down_layouts, ideal_budgets, physical_budgets, meta,
                "down", proxy, float(proxy_facts["beta"]), rows,
            )
            layer_facts.setdefault("diagnostics", {}).update({f"expert_{expert}_down_{k}": v for k, v in diagnostics.items()})
            print(json.dumps({"layer": layer, "expert": expert, "test_invocations": len(occurrence), "rows": len(rows)}), flush=True)

        layer_facts["wall_seconds"] = time.time() - layer_started
        facts["layers"][str(layer)] = layer_facts
        pd.DataFrame(rows).to_parquet(args.output / "phase_a_metrics.parquet", index=False)
        (args.output / "phase_a_facts.json").write_text(json.dumps(facts, indent=2, sort_keys=True) + "\n")
        del bf16, q4, binary, train_hidden, train_hidden_by_expert, all_hidden, input_transforms, down_transforms
        gc.collect()
        torch.cuda.empty_cache()

    facts["wall_seconds"] = time.time() - started
    facts["rows"] = len(rows)
    (args.output / "phase_a_facts.json").write_text(json.dumps(facts, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(rows).to_parquet(args.output / "phase_a_metrics.parquet", index=False)
    print(json.dumps({"complete": True, "rows": len(rows), "wall_seconds": facts["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
