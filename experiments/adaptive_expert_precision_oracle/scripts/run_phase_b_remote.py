#!/usr/bin/env python3
"""Sequential complete-SwiGLU expert H0 oracle with shared projection budget."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch


def silu(value: torch.Tensor) -> torch.Tensor:
    return value * torch.sigmoid(value)


def qenergy(error: torch.Tensor, proxy: torch.Tensor, beta: float) -> torch.Tensor:
    return torch.sum(error * error) + beta * torch.sum((error @ proxy) ** 2)


def page_bytes(indices: np.ndarray, packet_bytes: int, layout: np.ndarray, page_size: int = 4096) -> int:
    if len(indices) == 0:
        return 0
    inverse = np.empty_like(layout)
    inverse[layout] = np.arange(len(layout))
    positions = inverse[indices]
    pages = set((positions * packet_bytes // page_size).tolist())
    pages.update(((positions * packet_bytes + packet_bytes - 1) // page_size).tolist())
    return len(pages) * page_size


def candidate_counts(
    order: np.ndarray, packet_bytes: int, layout: np.ndarray, logical_budget: float,
    physical: bool,
) -> tuple[int, int, int]:
    if not physical:
        count = min(len(order), int(logical_budget // packet_bytes))
        return count, count * packet_bytes, page_bytes(order[:count], packet_bytes, layout)
    pages: set[int] = set()
    inverse = np.empty_like(layout)
    inverse[layout] = np.arange(len(layout))
    count = 0
    for atom in order:
        position = int(inverse[int(atom)])
        new = set(range(position * packet_bytes // 4096, (position * packet_bytes + packet_bytes - 1) // 4096 + 1))
        if (len(pages | new) * 4096) > logical_budget:
            continue
        pages |= new
        count += 1
    return count, count * packet_bytes, len(pages) * 4096


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
    from oracle_study.quant import binary_quantize
    from oracle_study.weights import load_gguf_experts, load_hf_experts
    from run_phase_a_remote import build_transforms, make_layout, proxy_gradients, select_experts

    args.output.mkdir(parents=True, exist_ok=True)
    loaded = np.load(args.captures, allow_pickle=False)
    data = {name: loaded[name] for name in loaded.files}
    reader = gguf.GGUFReader(config["remote"]["gguf_model"])
    rows: list[dict[str, Any]] = []
    facts: dict[str, Any] = {"layers": {}, "allocation_grid_points": 17, "started_unix": time.time()}
    seeds = [int(v) for v in config["learned_seeds"]]
    axes = (("ideal", [float(v) for v in config["ideal_bpw_budgets"]]),
            ("physical", [float(v) for v in config["physical_bpw_budgets"]]))
    device = "cuda"

    for layer in config["layers"]:
        layer_start = time.time()
        mask = data["layer"] == layer
        d = {name: value[mask] for name, value in data.items()}
        strata, frequencies = select_experts(d["expert_ids"], d["split"], int(config["experts_per_frequency_stratum"]))
        experts = sorted(strata)
        bf16 = load_hf_experts(Path(config["remote"]["hf_model"]), layer, experts)
        q4, _ = load_gguf_experts(reader, layer, experts, gguf)
        train_x = np.asarray(d["x"][d["split"] == "train"], dtype=np.float32)
        moment_x = np.mean(train_x.astype(np.float64) ** 2, axis=0)
        w1: dict[str, list[np.ndarray]] = {"gate": [], "up": []}
        train_h: list[np.ndarray] = []
        for ei, expert in enumerate(experts):
            for projection in ("gate", "up"):
                value, _ = binary_quantize(q4[projection][ei], int(config["binary_group_size"]), moment_x)
                w1[projection].append(value)
            occurrence = np.argwhere((d["expert_ids"] == expert) & (d["split"][:, None] == "train"))[:256, 0]
            x = np.asarray(d["x"][occurrence], dtype=np.float32)
            gate = x @ q4["gate"][ei].T
            up = x @ q4["up"][ei].T
            train_h.append((gate / (1.0 + np.exp(-gate))) * up)
        w1["gate"] = np.stack(w1["gate"])
        w1["up"] = np.stack(w1["up"])
        all_h = np.concatenate(train_h)
        moment_h = np.mean(all_h.astype(np.float64) ** 2, axis=0)
        w1["down"] = np.stack([
            binary_quantize(q4["down"][ei], int(config["binary_group_size"]), moment_h)[0]
            for ei in range(len(experts))
        ])
        rg = torch.from_numpy(q4["gate"] - w1["gate"]).to(device).reshape(-1, 2048)
        ru = torch.from_numpy(q4["up"] - w1["up"]).to(device).reshape(-1, 2048)
        input_operator = ((rg.T @ rg) + (ru.T @ ru)).cpu().numpy() / len(experts)
        input_transforms = build_transforms(train_x, input_operator, seeds, "input")
        rd = torch.from_numpy(q4["down"] - w1["down"]).to(device).reshape(-1, 512)
        down_operator = (rd.T @ rd).cpu().numpy() / len(experts)
        down_transforms = build_transforms(all_h, down_operator, seeds, "down")
        del rg, ru, rd, input_operator, down_operator
        torch.cuda.empty_cache()
        input_layouts = {key: make_layout(value, train_x) for key, value in input_transforms.items()}
        down_layouts = {key: make_layout(value, all_h) for key, value in down_transforms.items()}
        proxy_np, proxy_facts = proxy_gradients(d["xplus"], [d[f"h{h}_router_logits"] for h in range(1, 5)], d["split"])
        proxy = torch.from_numpy(proxy_np).to(device)
        beta = float(proxy_facts["beta"])
        method_keys = list(input_transforms)
        layer_allocations: dict[str, list[float]] = {key: [] for key in method_keys}

        for ei, expert in enumerate(experts):
            occurrence = np.argwhere((d["expert_ids"] == expert) & (d["split"][:, None] == "test"))
            occurrence = occurrence[: min(48, int(config["max_test_invocations_per_expert"]))]
            if not len(occurrence):
                continue
            record_indices, ranks = occurrence[:, 0], occurrence[:, 1]
            x = torch.from_numpy(np.asarray(d["x"][record_indices], dtype=np.float32)).to(device)
            qg = torch.from_numpy(q4["gate"][ei]).to(device)
            qu = torch.from_numpy(q4["up"][ei]).to(device)
            qd = torch.from_numpy(q4["down"][ei]).to(device)
            bg = torch.from_numpy(w1["gate"][ei]).to(device)
            bu = torch.from_numpy(w1["up"][ei]).to(device)
            bd = torch.from_numpy(w1["down"][ei]).to(device)
            g4, u4 = x @ qg.T, x @ qu.T
            h4 = silu(g4) * u4
            y4 = h4 @ qd.T
            g1, u1 = x @ bg.T, x @ bu.T
            h1 = silu(g1) * u1
            y1 = h1 @ bd.T
            base_damage = torch.stack([qenergy(y4[i] - y1[i], proxy, beta) for i in range(len(x))])

            for method in method_keys:
                ti = input_transforms[method]
                td = down_transforms[method]
                bi = torch.from_numpy(ti.synthesis).to(device)
                ai = torch.from_numpy(ti.analysis).to(device)
                bd_basis = torch.from_numpy(td.synthesis).to(device)
                ad = torch.from_numpy(td.analysis).to(device)
                ag = (qg - bg) @ bi
                au = (qu - bu) @ bi
                adown = (qd - bd) @ bd_basis
                z = x @ ai.T
                gu_score = z * z * (torch.sum(ag * ag, dim=0) + torch.sum(au * au, dim=0))[None, :]
                gu_order = torch.argsort(gu_score, dim=1, descending=True)
                z_auth = h4 @ ad.T
                down_score = z_auth * z_auth * torch.sum(adown * adown, dim=0)[None, :]
                down_order_auth = torch.argsort(down_score, dim=1, descending=True)
                gu_layout = input_layouts[method]
                d_layout = down_layouts[method]
                gu_packet = 2 * 512 * 2 + 8
                d_packet = 2048 * 2 + 8
                original_weights = 3 * 2048 * 512

                for axis, budgets in axes:
                    is_physical = axis == "physical"
                    for budget in budgets:
                        total_bytes = budget * original_weights / 8
                        for sample in range(len(x)):
                            best: dict[str, tuple[float, torch.Tensor, int, int, int, int]] = {}
                            gu_ids_all = gu_order[sample].cpu().numpy()
                            d_ids_auth_all = down_order_auth[sample].cpu().numpy()
                            for fraction in np.linspace(0.0, 1.0, 17):
                                kg, gu_logical, gu_physical = candidate_counts(
                                    gu_ids_all, gu_packet, gu_layout, total_bytes * float(fraction), is_physical
                                )
                                kd_auth, d_logical_auth, d_physical_auth = candidate_counts(
                                    d_ids_auth_all, d_packet, d_layout, total_bytes * (1.0 - float(fraction)), is_physical
                                )
                                gu_ids = gu_order[sample, :kg]
                                cg = torch.sum(ag[:, gu_ids] * z[sample, gu_ids][None, :], dim=1) if kg else torch.zeros(512, device=device)
                                cu = torch.sum(au[:, gu_ids] * z[sample, gu_ids][None, :], dim=1) if kg else torch.zeros(512, device=device)
                                ht = silu(g1[sample] + cg) * (u1[sample] + cu)
                                down_base = bd @ ht
                                auth_ids = down_order_auth[sample, :kd_auth]
                                corr_auth = torch.sum(adown[:, auth_ids] * z_auth[sample, auth_ids][None, :], dim=1) if kd_auth else torch.zeros(2048, device=device)
                                y_auth = down_base + corr_auth
                                damage_auth = float(qenergy(y4[sample] - y_auth, proxy, beta).item())
                                auth_total_physical = gu_physical + d_physical_auth
                                if ("authoritative" not in best) or damage_auth < best["authoritative"][0]:
                                    best["authoritative"] = (damage_auth, y_auth, kg, kd_auth, gu_logical + d_logical_auth, auth_total_physical)

                                z_seq = ht @ ad.T
                                seq_score = z_seq * z_seq * torch.sum(adown * adown, dim=0)
                                seq_order = torch.argsort(seq_score, descending=True).cpu().numpy()
                                kd_seq, d_logical_seq, d_physical_seq = candidate_counts(
                                    seq_order, d_packet, d_layout, total_bytes * (1.0 - float(fraction)), is_physical
                                )
                                seq_ids = torch.from_numpy(seq_order[:kd_seq]).to(device)
                                corr_seq = torch.sum(adown[:, seq_ids] * z_seq[seq_ids][None, :], dim=1) if kd_seq else torch.zeros(2048, device=device)
                                y_seq = down_base + corr_seq
                                damage_seq = float(qenergy(y4[sample] - y_seq, proxy, beta).item())
                                seq_total_physical = gu_physical + d_physical_seq
                                if ("sequential" not in best) or damage_seq < best["sequential"][0]:
                                    best["sequential"] = (damage_seq, y_seq, kg, kd_seq, gu_logical + d_logical_seq, seq_total_physical)

                            record = int(record_indices[sample])
                            rank = int(ranks[sample])
                            logits = np.sort(d["router_logits"][record])
                            for variant, selected in best.items():
                                damage, yhat, kg, kd, logical, physical = selected
                                error = y4[sample] - yhat
                                relative = float(torch.linalg.norm(error).item() / max(torch.linalg.norm(y4[sample]).item(), 1e-20))
                                cosine = float(torch.nn.functional.cosine_similarity(yhat[None, :], y4[sample][None, :]).item())
                                row = {
                                    "phase": "B_complete_expert",
                                    "sequence_id": str(d["sequence_id"][record]),
                                    "request_id": str(d["request_id"][record]),
                                    "position": int(d["position"][record]),
                                    "layer": int(layer),
                                    "expert_id": int(expert),
                                    "expert_stratum": strata[expert],
                                    "expert_train_frequency": frequencies[expert],
                                    "router_rank": rank + 1,
                                    "router_coefficient": float(d["router_weights"][record, rank]),
                                    "router_boundary_margin": float(logits[-8] - logits[-9]),
                                    "method": method,
                                    "down_oracle_variant": variant,
                                    "rate_axis": axis,
                                    "budget_bpw": float(budget),
                                    "logical_bytes": int(logical),
                                    "physical_bytes_4k": int(physical),
                                    "ideal_bpw": float(8 * logical / original_weights),
                                    "physical_bpw": float(8 * physical / original_weights),
                                    "gate_up_atoms": int(kg),
                                    "down_atoms": int(kd),
                                    "gate_up_allocation_fraction": float((kg * gu_packet) / max(logical, 1)),
                                    "recovery": float(1.0 - damage / max(float(base_damage[sample].item()), 1e-20)),
                                    "damage": float(damage),
                                    "base_damage": float(base_damage[sample].item()),
                                    "relative_output_error": relative,
                                    "cosine_similarity": cosine,
                                    "absolute_error": float(torch.sqrt(torch.mean(error * error)).item()),
                                }
                                rows.append(row)
                                layer_allocations[method].append(row["gate_up_allocation_fraction"])
                del bi, ai, bd_basis, ad, ag, au, adown, z, gu_score, gu_order, z_auth, down_score, down_order_auth
                torch.cuda.empty_cache()
            print(json.dumps({"phase": "B", "layer": layer, "expert": expert, "samples": len(x), "rows": len(rows)}), flush=True)
            del x, qg, qu, qd, bg, bu, bd, g4, u4, h4, y4, g1, u1, h1, y1, base_damage
            torch.cuda.empty_cache()

        facts["layers"][str(layer)] = {
            "selected_experts": experts,
            "test_invocations": int(sum(1 for row in rows if row["layer"] == layer) / max(len(method_keys) * sum(len(b) for _, b in axes) * 2, 1)),
            "proxy": proxy_facts,
            "mean_gate_up_allocation_fraction": {key: float(np.mean(value)) if value else None for key, value in layer_allocations.items()},
            "wall_seconds": time.time() - layer_start,
        }
        pd.DataFrame(rows).to_parquet(args.output / "phase_b_metrics.parquet", index=False)
        (args.output / "phase_b_facts.json").write_text(json.dumps(facts, indent=2, sort_keys=True) + "\n")
        del bf16, q4, w1, train_h, all_h, input_transforms, down_transforms, proxy
        gc.collect()
        torch.cuda.empty_cache()

    facts["wall_seconds"] = time.time() - facts["started_unix"]
    facts["rows"] = len(rows)
    pd.DataFrame(rows).to_parquet(args.output / "phase_b_metrics.parquet", index=False)
    (args.output / "phase_b_facts.json").write_text(json.dumps(facts, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"phase_b_complete": True, "rows": len(rows), "wall_seconds": facts["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
