#!/usr/bin/env python3
"""Cost-bounded progressive-precision sensitivity for the best pilot supports."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch


def nested_quantize_columns(atoms: torch.Tensor, bits: int) -> torch.Tensor:
    maximum = torch.amax(torch.abs(atoms), dim=0, keepdim=True)
    scale = torch.where(maximum > 0, maximum / 32767.0, torch.ones_like(maximum))
    master = torch.clamp(torch.round(atoms / scale), -32767, 32767).to(torch.int32)
    shift = 16 - bits
    restored = torch.bitwise_left_shift(torch.bitwise_right_shift(master, shift), shift)
    return restored.to(torch.float32) * scale


def prefix_under_pages(order: np.ndarray, packet_bytes: int, layout: np.ndarray, allowed: float) -> tuple[int, int]:
    inverse = np.empty_like(layout)
    inverse[layout] = np.arange(len(layout))
    pages: set[int] = set()
    count = 0
    for atom in order:
        position = int(inverse[int(atom)])
        atom_pages = set(range(
            position * packet_bytes // 4096,
            (position * packet_bytes + packet_bytes - 1) // 4096 + 1,
        ))
        if len(pages | atom_pages) * 4096 > allowed:
            break
        pages |= atom_pages
        count += 1
    return count, len(pages) * 4096


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    experiment = args.config.resolve().parents[1]
    sys.path.insert(0, str(experiment / "src"))
    sys.path.insert(0, str(experiment / "scripts"))
    sys.path.insert(0, config["remote"]["gguf_python"])
    import gguf
    from oracle_study.metrics import metric_rows
    from oracle_study.quant import binary_quantize
    from oracle_study.weights import load_gguf_experts, load_hf_experts
    from run_phase_a_remote import make_layout, proxy_gradients, select_experts
    from run_phase_a_v2_remote import generalized_full_rank

    loaded = np.load(args.captures, allow_pickle=False)
    data = {name: loaded[name] for name in loaded.files}
    reader = gguf.GGUFReader(config["remote"]["gguf_model"])
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    facts: dict = {"layers": {}, "progressive_representation": "nested two's-complement prefixes of one int16 master atom"}
    device = "cuda"
    started = time.time()

    for layer in config["layers"]:
        mask = data["layer"] == layer
        d = {name: value[mask] for name, value in data.items()}
        strata, frequencies = select_experts(d["expert_ids"], d["split"], int(config["experts_per_frequency_stratum"]))
        experts = sorted(strata)
        bf16 = load_hf_experts(Path(config["remote"]["hf_model"]), layer, experts)
        q4, storage = load_gguf_experts(reader, layer, experts, gguf)
        train_x = np.asarray(d["x"][d["split"] == "train"], dtype=np.float32)
        moment_x = np.mean(train_x.astype(np.float64) ** 2, axis=0)
        w1_gate, w1_up, train_h = [], [], []
        for ei, expert in enumerate(experts):
            w1_gate.append(binary_quantize(q4["gate"][ei], int(config["binary_group_size"]), moment_x)[0])
            w1_up.append(binary_quantize(q4["up"][ei], int(config["binary_group_size"]), moment_x)[0])
            occurrence = np.argwhere((d["expert_ids"] == expert) & (d["split"][:, None] == "train"))[:256, 0]
            x = np.asarray(d["x"][occurrence], dtype=np.float32)
            gate = x @ q4["gate"][ei].T
            up = x @ q4["up"][ei].T
            train_h.append((gate / (1.0 + np.exp(-gate))) * up)
        w1_gate, w1_up = np.stack(w1_gate), np.stack(w1_up)
        all_h = np.concatenate(train_h)
        moment_h = np.mean(all_h.astype(np.float64) ** 2, axis=0)
        w1_down = np.stack([
            binary_quantize(q4["down"][ei], int(config["binary_group_size"]), moment_h)[0]
            for ei in range(len(experts))
        ])
        rg = torch.from_numpy(q4["gate"] - w1_gate).to(device).reshape(-1, 2048)
        ru = torch.from_numpy(q4["up"] - w1_up).to(device).reshape(-1, 2048)
        operator = ((rg.T @ rg) + (ru.T @ ru)).cpu().numpy() / len(experts)
        generalized = generalized_full_rank(train_x, operator)
        native_synthesis = np.eye(512, dtype=np.float32)
        native_analysis = native_synthesis
        input_layout = make_layout(generalized, train_x)
        class Native:
            analysis = native_analysis
        down_layout = make_layout(Native(), all_h)
        proxy_np, proxy_facts = proxy_gradients(d["xplus"], [d[f"h{h}_router_logits"] for h in range(1, 5)], d["split"])
        beta = float(proxy_facts["beta"])
        del rg, ru, operator
        torch.cuda.empty_cache()
        layer_storage = {}

        for ei, expert in enumerate(experts):
            occurrence = np.argwhere((d["expert_ids"] == expert) & (d["split"][:, None] == "test"))
            occurrence = occurrence[: int(config["max_test_invocations_per_expert"])]
            if not len(occurrence):
                continue
            record_indices, ranks = occurrence[:, 0], occurrence[:, 1]
            for projection, inputs, qweight, bweight, synthesis, analysis, layout in (
                ("gate", np.asarray(d["x"][record_indices], dtype=np.float32), q4["gate"][ei], w1_gate[ei], generalized.synthesis, generalized.analysis, input_layout),
                ("up", np.asarray(d["x"][record_indices], dtype=np.float32), q4["up"][ei], w1_up[ei], generalized.synthesis, generalized.analysis, input_layout),
                ("down", None, q4["down"][ei], w1_down[ei], native_synthesis, native_analysis, down_layout),
            ):
                if projection == "down":
                    x0 = np.asarray(d["x"][record_indices], dtype=np.float32)
                    gate = x0 @ q4["gate"][ei].T
                    up = x0 @ q4["up"][ei].T
                    inputs = (gate / (1.0 + np.exp(-gate))) * up
                x = torch.from_numpy(inputs).to(device)
                q = torch.from_numpy(qweight).to(device)
                b = torch.from_numpy(bweight).to(device)
                basis = torch.from_numpy(synthesis).to(device)
                transform = torch.from_numpy(analysis).to(device)
                atoms = (q - b) @ basis
                codes = x @ transform.T
                y4 = x @ q.T
                y1 = x @ b.T
                correction = y4 - y1
                number_weights = int(np.prod(qweight.shape))
                progressive_outputs: dict[tuple[float, int], np.ndarray] = {}
                progressive_meta: dict[tuple[float, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
                for bits in (2, 4, 8, 16):
                    quant_atoms = nested_quantize_columns(atoms, bits)
                    score = codes * codes * torch.sum(quant_atoms * quant_atoms, dim=0)[None, :]
                    order = torch.argsort(score, dim=1, descending=True)
                    packet_bytes = math.ceil(qweight.shape[0] * bits / 8) + 10
                    layer_storage[f"{projection}_{bits}"] = {
                        "atom_bytes_per_expert": int(packet_bytes * atoms.shape[1] + 64),
                        "q4_plus_view_multiplier": float((sum(v["bytes_per_expert"] for v in storage.values()) + packet_bytes * atoms.shape[1] + 64) / sum(v["bytes_per_expert"] for v in storage.values())),
                    }
                    for budget in config["physical_bpw_budgets"]:
                        rhats = torch.zeros_like(correction)
                        counts = np.zeros(len(x), dtype=np.int64)
                        physical = np.zeros(len(x), dtype=np.int64)
                        allowed = float(budget) * number_weights / 8
                        order_np = order.cpu().numpy()
                        for sample in range(len(x)):
                            count, actual = prefix_under_pages(order_np[sample], packet_bytes, layout, allowed)
                            counts[sample] = count
                            physical[sample] = actual
                            if count:
                                ids = order[sample, :count]
                                rhats[sample] = torch.sum(quant_atoms[:, ids] * codes[sample, ids][None, :], dim=1)
                        approximation = (y1 + rhats).cpu().numpy()
                        key = (float(budget), bits)
                        progressive_outputs[key] = approximation
                        progressive_meta[key] = (counts, physical, counts * packet_bytes)
                        proxy = proxy_np if projection == "down" else None
                        proxy_beta = beta if projection == "down" else 0.0
                        values = metric_rows(y4.cpu().numpy(), y1.cpu().numpy(), approximation, proxy, proxy_beta)
                        for sample, record in enumerate(record_indices.tolist()):
                            sorted_logits = np.sort(d["router_logits"][record])
                            row = {
                                "phase": "A_progressive",
                                "sequence_id": str(d["sequence_id"][record]),
                                "request_id": str(d["request_id"][record]),
                                "position": int(d["position"][record]),
                                "layer": int(layer),
                                "expert_id": int(expert),
                                "expert_stratum": strata[expert],
                                "router_rank": int(ranks[sample] + 1),
                                "router_coefficient": float(d["router_weights"][record, ranks[sample]]),
                                "router_boundary_margin": float(sorted_logits[-8] - sorted_logits[-9]),
                                "projection": projection,
                                "method": "generalized" if projection != "down" else "native",
                                "atom_bits": bits,
                                "budget_bpw": float(budget),
                                "selected_atoms": int(counts[sample]),
                                "logical_bytes": int(counts[sample] * packet_bytes),
                                "physical_bytes_4k": int(physical[sample]),
                                "ideal_bpw": float(8 * counts[sample] * packet_bytes / number_weights),
                                "physical_bpw": float(8 * physical[sample] / number_weights),
                            }
                            row.update({name: float(array[sample]) for name, array in values.items()})
                            rows.append(row)
                # Per-invocation H0 precision choice under the same physical budget.
                for budget in config["physical_bpw_budgets"]:
                    candidates = [progressive_outputs[(float(budget), bits)] for bits in (2, 4, 8, 16)]
                    ref = y4.cpu().numpy()
                    base = y1.cpu().numpy()
                    proxy = proxy_np if projection == "down" else None
                    proxy_beta = beta if projection == "down" else 0.0
                    candidate_metrics = [metric_rows(ref, base, value, proxy, proxy_beta) for value in candidates]
                    choice = np.argmax(np.stack([value["recovery"] for value in candidate_metrics], axis=1), axis=1)
                    selected_output = np.stack([candidates[choice[i]][i] for i in range(len(x))])
                    selected_values = metric_rows(ref, base, selected_output, proxy, proxy_beta)
                    for sample, record in enumerate(record_indices.tolist()):
                        bits = (2, 4, 8, 16)[int(choice[sample])]
                        counts, physical, logical = progressive_meta[(float(budget), bits)]
                        rows.append({
                            "phase": "A_progressive", "sequence_id": str(d["sequence_id"][record]),
                            "request_id": str(d["request_id"][record]), "position": int(d["position"][record]),
                            "layer": int(layer), "expert_id": int(expert), "expert_stratum": strata[expert],
                            "router_rank": int(ranks[sample] + 1),
                            "router_coefficient": float(d["router_weights"][record, ranks[sample]]),
                            "router_boundary_margin": float(np.sort(d["router_logits"][record])[-8] - np.sort(d["router_logits"][record])[-9]),
                            "projection": projection,
                            "method": "progressive_precision_oracle", "atom_bits": bits,
                            "budget_bpw": float(budget), "selected_atoms": int(counts[sample]),
                            "logical_bytes": int(logical[sample]), "physical_bytes_4k": int(physical[sample]),
                            "ideal_bpw": float(8 * logical[sample] / number_weights),
                            "physical_bpw": float(8 * physical[sample] / number_weights),
                            **{name: float(array[sample]) for name, array in selected_values.items()},
                        })
                del x, q, b, basis, transform, atoms, codes, y4, y1, correction
                torch.cuda.empty_cache()
            print(json.dumps({"phase": "progressive", "layer": layer, "expert": expert, "rows": len(rows)}), flush=True)
        facts["layers"][str(layer)] = {"storage": layer_storage, "proxy": proxy_facts}
        pd.DataFrame(rows).to_parquet(args.output / "progressive_metrics.parquet", index=False)
        del bf16, q4, w1_gate, w1_up, w1_down, train_h, all_h, generalized
        gc.collect(); torch.cuda.empty_cache()
    facts["rows"] = len(rows)
    facts["wall_seconds"] = time.time() - started
    (args.output / "progressive_facts.json").write_text(json.dumps(facts, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(rows).to_parquet(args.output / "progressive_metrics.parquet", index=False)
    print(json.dumps({"progressive_complete": True, "rows": len(rows), "wall_seconds": facts["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
