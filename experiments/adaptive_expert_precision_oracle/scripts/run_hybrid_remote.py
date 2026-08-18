#!/usr/bin/env python3
"""Independent gate/up/down progressive-atom allocation for complete experts.

This is a deliberately bounded H0 oracle: bases and layouts are training-only,
while each held-out invocation may choose atom precision and the asymmetric
projection allocation that minimises the complete-expert output damage.
"""

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
    return torch.sum(error * error, dim=-1) + beta * torch.sum((error @ proxy) ** 2, dim=-1)


def nested_quantize_columns(atoms: torch.Tensor, bits: int) -> torch.Tensor:
    maximum = torch.amax(torch.abs(atoms), dim=0, keepdim=True)
    scale = torch.where(maximum > 0, maximum / 32767.0, torch.ones_like(maximum))
    master = torch.clamp(torch.round(atoms / scale), -32767, 32767).to(torch.int32)
    shift = 16 - bits
    restored = torch.bitwise_left_shift(torch.bitwise_right_shift(master, shift), shift)
    return restored.to(torch.float32) * scale


def physical_prefix(order: np.ndarray, packet_bytes: int, layout: np.ndarray, allowed: float) -> tuple[int, int]:
    inverse = np.empty_like(layout)
    inverse[layout] = np.arange(len(layout))
    pages: set[int] = set()
    count = 0
    for atom in order.tolist():
        position = int(inverse[int(atom)])
        first = position * packet_bytes // 4096
        last = (position * packet_bytes + packet_bytes - 1) // 4096
        proposed = pages | set(range(first, last + 1))
        if len(proposed) * 4096 > allowed:
            break
        pages = proposed
        count += 1
    return count, len(pages) * 4096


def physical_prefix_table(order: np.ndarray, packet_bytes: int, layout: np.ndarray) -> np.ndarray:
    inverse = np.empty_like(layout)
    inverse[layout] = np.arange(len(layout))
    pages: set[int] = set()
    table = np.zeros(len(order) + 1, dtype=np.int64)
    for index, atom in enumerate(order.tolist(), start=1):
        position = int(inverse[int(atom)])
        first = position * packet_bytes // 4096
        last = (position * packet_bytes + packet_bytes - 1) // 4096
        pages.update(range(first, last + 1))
        table[index] = len(pages) * 4096
    return table


def table_count(table: np.ndarray, allowed: float) -> tuple[int, int]:
    count = int(np.searchsorted(table, allowed, side="right") - 1)
    return count, int(table[count])


def transforms_for(samples: np.ndarray, operator: np.ndarray, seed: int):
    from oracle_study.bases import identity_transform, learned_orthogonal_transform, pca_transform
    from run_phase_a_v2_remote import generalized_full_rank
    return {
        "native": identity_transform(samples.shape[1]),
        "pca": pca_transform(samples),
        "generalized": generalized_full_rank(samples, operator),
        "learned": learned_orthogonal_transform(samples, operator, seed),
    }


def progressive_candidates(
    inputs: torch.Tensor,
    residual: torch.Tensor,
    transform: Any,
    layout: np.ndarray,
    rates: list[float],
    bits_list: list[int],
    base_output: torch.Tensor,
    reference_output: torch.Tensor,
) -> dict[float, dict[str, Any]]:
    """Choose the locally best stored precision at each physical rate/sample."""
    basis = torch.from_numpy(transform.synthesis).to(inputs.device)
    analysis = torch.from_numpy(transform.analysis).to(inputs.device)
    atoms = residual @ basis
    codes = inputs @ analysis.T
    n = len(inputs)
    number_weights = int(residual.numel())
    result: dict[float, dict[str, Any]] = {}
    for rate in rates:
        result[rate] = {
            "correction": torch.zeros_like(reference_output),
            "damage": torch.full((n,), float("inf"), device=inputs.device),
            "physical": np.zeros(n, dtype=np.int64),
            "logical": np.zeros(n, dtype=np.int64),
            "atoms": np.zeros(n, dtype=np.int64),
            "bits": np.zeros(n, dtype=np.int64),
        }
    for bits in bits_list:
        qatoms = nested_quantize_columns(atoms, bits)
        score = codes * codes * torch.sum(qatoms * qatoms, dim=0)[None, :]
        order = torch.argsort(score, dim=1, descending=True)
        values = torch.gather(codes, 1, order)
        cumulative = torch.cumsum(qatoms.T[order] * values[:, :, None], dim=1)
        order_np = order.cpu().numpy()
        packet = math.ceil(qatoms.shape[0] * bits / 8) + 10
        tables = [physical_prefix_table(value, packet, layout) for value in order_np]
        for rate in rates:
            allowed = rate * number_weights / 8
            correction = torch.zeros_like(reference_output)
            physical = np.zeros(n, dtype=np.int64)
            logical = np.zeros(n, dtype=np.int64)
            counts = np.zeros(n, dtype=np.int64)
            for sample in range(n):
                count, read = table_count(tables[sample], allowed)
                counts[sample] = count
                physical[sample] = read
                logical[sample] = count * packet
                if count:
                    correction[sample] = cumulative[sample, count - 1]
            damage = torch.sum((reference_output - (base_output + correction)) ** 2, dim=1)
            improve = damage < result[rate]["damage"]
            result[rate]["damage"] = torch.where(improve, damage, result[rate]["damage"])
            result[rate]["correction"] = torch.where(improve[:, None], correction, result[rate]["correction"])
            choose = improve.cpu().numpy()
            for key, value in (("physical", physical), ("logical", logical), ("atoms", counts)):
                result[rate][key][choose] = value[choose]
            result[rate]["bits"][choose] = bits
    del basis, analysis, atoms, codes
    return result


def prepare_down(residual: torch.Tensor, transform: Any, bits_list: list[int]) -> dict[str, Any]:
    basis = torch.from_numpy(transform.synthesis).to(residual.device)
    return {
        "analysis": torch.from_numpy(transform.analysis).to(residual.device),
        "atoms": {bits: nested_quantize_columns(residual @ basis, bits) for bits in bits_list},
    }


def down_variants(
    hidden: torch.Tensor,
    prepared: dict[str, Any],
    layout: np.ndarray,
    rates: list[float],
    bits_list: list[int],
) -> dict[tuple[int, float], dict[str, Any]]:
    analysis = prepared["analysis"]
    codes = hidden @ analysis.T
    n = len(hidden)
    first_atoms = prepared["atoms"][bits_list[0]]
    number_weights = int(first_atoms.numel())
    output: dict[tuple[int, float], dict[str, Any]] = {}
    for bits in bits_list:
        qatoms = prepared["atoms"][bits]
        score = codes * codes * torch.sum(qatoms * qatoms, dim=0)[None, :]
        order = torch.argsort(score, dim=1, descending=True)
        values = torch.gather(codes, 1, order)
        cumulative = torch.cumsum(qatoms.T[order] * values[:, :, None], dim=1)
        order_np = order.cpu().numpy()
        packet = math.ceil(qatoms.shape[0] * bits / 8) + 10
        tables = [physical_prefix_table(value, packet, layout) for value in order_np]
        for rate in rates:
            allowed = rate * number_weights / 8
            correction = torch.zeros((n, qatoms.shape[0]), dtype=torch.float32, device=hidden.device)
            physical = np.zeros(n, dtype=np.int64)
            logical = np.zeros(n, dtype=np.int64)
            counts = np.zeros(n, dtype=np.int64)
            for sample in range(n):
                count, read = table_count(tables[sample], allowed)
                counts[sample] = count
                physical[sample] = read
                logical[sample] = count * packet
                if count:
                    correction[sample] = cumulative[sample, count - 1]
            output[(bits, rate)] = {
                "correction": correction,
                "physical": physical,
                "logical": logical,
                "atoms": counts,
            }
    return output


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
    from oracle_study.weights import load_gguf_experts
    from run_phase_a_remote import make_layout, proxy_gradients, select_experts

    args.output.mkdir(parents=True, exist_ok=True)
    loaded = np.load(args.captures, allow_pickle=False)
    data = {name: loaded[name] for name in loaded.files}
    reader = gguf.GGUFReader(config["remote"]["gguf_model"])
    rates = [float(v) for v in config["projection_candidate_bpw"]]
    totals = [float(v) for v in config["physical_bpw_budgets"]]
    bits_list = [int(v) for v in config["progressive_bits"]]
    device = "cuda"
    rows: list[dict[str, Any]] = []
    facts: dict[str, Any] = {
        "description": "H0 progressive-precision projection-specific basis and asymmetric physical-byte oracle",
        "selection": "training-only bases/layouts; test-invocation atom, precision, and allocation oracle",
        "allocation_constraint": "(physical_gate_bpw + physical_up_bpw + physical_down_bpw) / 3 <= total average bpw",
        "layers": {},
        "started_unix": time.time(),
    }
    method_spec = {
        "all_native": ("native", "native", "native"),
        "all_pca": ("pca", "pca", "pca"),
        "all_generalized": ("generalized", "generalized", "generalized"),
        "pca_pca_native": ("pca", "pca", "native"),
        "generalized_generalized_native": ("generalized", "generalized", "native"),
        "learned_learned_native": ("learned", "learned", "native"),
    }

    for layer in config["layers"]:
        layer_started = time.time()
        mask = data["layer"] == layer
        d = {name: value[mask] for name, value in data.items()}
        strata, frequencies = select_experts(d["expert_ids"], d["split"], int(config["experts_per_frequency_stratum"]))
        experts = sorted(strata)
        q4, storage = load_gguf_experts(reader, layer, experts, gguf)
        train_x = np.asarray(d["x"][d["split"] == "train"], dtype=np.float32)
        moment_x = np.mean(train_x.astype(np.float64) ** 2, axis=0)
        w1: dict[str, Any] = {"gate": [], "up": []}
        train_h = []
        for ei, expert in enumerate(experts):
            w1["gate"].append(binary_quantize(q4["gate"][ei], int(config["binary_group_size"]), moment_x)[0])
            w1["up"].append(binary_quantize(q4["up"][ei], int(config["binary_group_size"]), moment_x)[0])
            occurrence = np.argwhere((d["expert_ids"] == expert) & (d["split"][:, None] == "train"))[:256, 0]
            xt = np.asarray(d["x"][occurrence], dtype=np.float32)
            gt = xt @ q4["gate"][ei].T
            ut = xt @ q4["up"][ei].T
            train_h.append((gt / (1.0 + np.exp(-gt))) * ut)
        w1["gate"] = np.stack(w1["gate"])
        w1["up"] = np.stack(w1["up"])
        all_h = np.concatenate(train_h)
        moment_h = np.mean(all_h.astype(np.float64) ** 2, axis=0)
        w1["down"] = np.stack([binary_quantize(q4["down"][ei], int(config["binary_group_size"]), moment_h)[0] for ei in range(len(experts))])
        rg = torch.from_numpy(q4["gate"] - w1["gate"]).to(device).reshape(-1, 2048)
        ru = torch.from_numpy(q4["up"] - w1["up"]).to(device).reshape(-1, 2048)
        rd = torch.from_numpy(q4["down"] - w1["down"]).to(device).reshape(-1, 512)
        input_operator = ((rg.T @ rg) + (ru.T @ ru)).cpu().numpy() / len(experts)
        down_operator = (rd.T @ rd).cpu().numpy() / len(experts)
        del rg, ru, rd
        input_transforms = transforms_for(train_x, input_operator, int(config["learned_seeds"][0]))
        down_transforms = transforms_for(all_h, down_operator, int(config["learned_seeds"][0]))
        input_layouts = {key: make_layout(value, train_x) for key, value in input_transforms.items()}
        down_layouts = {key: make_layout(value, all_h) for key, value in down_transforms.items()}
        proxy_np, proxy_facts = proxy_gradients(d["xplus"], [d[f"h{h}_router_logits"] for h in range(1, 5)], d["split"])
        proxy = torch.from_numpy(proxy_np).to(device)
        beta = float(proxy_facts["beta"])
        layer_invocations = 0

        for ei, expert in enumerate(experts):
            occurrence = np.argwhere((d["expert_ids"] == expert) & (d["split"][:, None] == "test"))
            occurrence = occurrence[: int(config["max_test_invocations_per_expert"])]
            if not len(occurrence):
                continue
            record_indices, ranks = occurrence[:, 0], occurrence[:, 1]
            layer_invocations += len(occurrence)
            x = torch.from_numpy(np.asarray(d["x"][record_indices], dtype=np.float32)).to(device)
            qg, qu, qd = (torch.from_numpy(q4[name][ei]).to(device) for name in ("gate", "up", "down"))
            bg, bu, bd = (torch.from_numpy(w1[name][ei]).to(device) for name in ("gate", "up", "down"))
            g4, u4 = x @ qg.T, x @ qu.T
            h4 = silu(g4) * u4
            y4 = h4 @ qd.T
            g1, u1 = x @ bg.T, x @ bu.T
            h1 = silu(g1) * u1
            y1 = h1 @ bd.T
            base_damage = qenergy(y4 - y1, proxy, beta)
            base_norm = torch.linalg.norm(y4, dim=1)
            residuals = {"gate": qg - bg, "up": qu - bu, "down": qd - bd}

            gate_cache: dict[str, Any] = {}
            up_cache: dict[str, Any] = {}
            auth_down_cache: dict[str, Any] = {}
            prepared_down: dict[str, Any] = {}
            for basis_name in set(value for spec in method_spec.values() for value in spec):
                gate_cache[basis_name] = progressive_candidates(
                    x, residuals["gate"], input_transforms[basis_name], input_layouts[basis_name],
                    rates, bits_list, g1, g4,
                )
                up_cache[basis_name] = progressive_candidates(
                    x, residuals["up"], input_transforms[basis_name], input_layouts[basis_name],
                    rates, bits_list, u1, u4,
                )
                prepared_down[basis_name] = prepare_down(residuals["down"], down_transforms[basis_name], bits_list)
                auth_down_cache[basis_name] = down_variants(
                    h4, prepared_down[basis_name], down_layouts[basis_name], rates, bits_list,
                )

            for method, (gbasis, ubasis, dbasis) in method_spec.items():
                best = {
                    variant: {
                        total: [{"damage": float("inf")} for _ in range(len(x))]
                        for total in totals
                    } for variant in ("authoritative", "sequential")
                }
                gcandidates = gate_cache[gbasis]
                ucandidates = up_cache[ubasis]
                auth_candidates = auth_down_cache[dbasis]
                for gr in rates:
                    gcand = gcandidates[gr]
                    for ur in rates:
                        ucand = ucandidates[ur]
                        ht = silu(g1 + gcand["correction"]) * (u1 + ucand["correction"])
                        down_base = ht @ bd.T
                        seq_candidates = down_variants(
                            ht, prepared_down[dbasis], down_layouts[dbasis], rates, bits_list,
                        )
                        for bits in bits_list:
                            for dr in rates:
                                for variant, dcand in (("authoritative", auth_candidates[(bits, dr)]), ("sequential", seq_candidates[(bits, dr)])):
                                    yhat = down_base + dcand["correction"]
                                    damage = qenergy(y4 - yhat, proxy, beta)
                                    damage_np = damage.detach().cpu().numpy()
                                    for sample in range(len(x)):
                                        physical = int(gcand["physical"][sample] + ucand["physical"][sample] + dcand["physical"][sample])
                                        physical_average_bpw = 8 * physical / (3 * 2048 * 512)
                                        for total in totals:
                                            if physical_average_bpw <= total + 1e-12 and float(damage_np[sample]) < best[variant][total][sample]["damage"]:
                                                best[variant][total][sample] = {
                                                    "damage": float(damage_np[sample]),
                                                    "yhat": yhat[sample].detach().clone(),
                                                    "physical": physical,
                                                    "logical": int(gcand["logical"][sample] + ucand["logical"][sample] + dcand["logical"][sample]),
                                                    "gate_bpw": float(8 * gcand["physical"][sample] / (2048 * 512)),
                                                    "up_bpw": float(8 * ucand["physical"][sample] / (2048 * 512)),
                                                    "down_bpw": float(8 * dcand["physical"][sample] / (2048 * 512)),
                                                    "gate_atoms": int(gcand["atoms"][sample]),
                                                    "up_atoms": int(ucand["atoms"][sample]),
                                                    "down_atoms": int(dcand["atoms"][sample]),
                                                    "gate_bits": int(gcand["bits"][sample]),
                                                    "up_bits": int(ucand["bits"][sample]),
                                                    "down_bits": int(bits),
                                                }
                        del seq_candidates
                for variant in ("authoritative", "sequential"):
                    for total in totals:
                        for sample, selected in enumerate(best[variant][total]):
                            if not math.isfinite(selected["damage"]):
                                raise RuntimeError(f"no feasible allocation {layer=} {expert=} {method=} {variant=} {total=}")
                            error = y4[sample] - selected["yhat"]
                            record = int(record_indices[sample])
                            rank = int(ranks[sample])
                            logits = np.sort(d["router_logits"][record])
                            rows.append({
                                "phase": "B_complete_expert_hybrid",
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
                                "gate_basis": gbasis,
                                "up_basis": ubasis,
                                "down_basis": dbasis,
                                "down_oracle_variant": variant,
                                "rate_axis": "physical",
                                "budget_bpw": total,
                                "physical_bytes_4k": selected["physical"],
                                "logical_bytes": selected["logical"],
                                "physical_bpw": float(8 * selected["physical"] / (3 * 2048 * 512)),
                                "ideal_bpw": float(8 * selected["logical"] / (3 * 2048 * 512)),
                                "gate_bpw": selected["gate_bpw"],
                                "up_bpw": selected["up_bpw"],
                                "down_bpw": selected["down_bpw"],
                                "gate_atoms": selected["gate_atoms"],
                                "up_atoms": selected["up_atoms"],
                                "down_atoms": selected["down_atoms"],
                                "gate_bits": selected["gate_bits"],
                                "up_bits": selected["up_bits"],
                                "down_bits": selected["down_bits"],
                                "recovery": float(1.0 - selected["damage"] / max(float(base_damage[sample]), 1e-20)),
                                "damage": selected["damage"],
                                "base_damage": float(base_damage[sample]),
                                "relative_output_error": float(torch.linalg.norm(error) / max(float(base_norm[sample]), 1e-20)),
                                "cosine_similarity": float(torch.nn.functional.cosine_similarity(selected["yhat"][None], y4[sample][None])),
                                "absolute_error": float(torch.sqrt(torch.mean(error * error))),
                            })
                print(json.dumps({"phase": "hybrid", "layer": layer, "expert": expert, "method": method, "rows": len(rows)}), flush=True)
            pd.DataFrame(rows).to_parquet(args.output / "hybrid_metrics.parquet", index=False)
            del x, qg, qu, qd, bg, bu, bd, g4, u4, h4, y4, g1, u1, h1, y1, base_damage
            gc.collect(); torch.cuda.empty_cache()

        facts["layers"][str(layer)] = {
            "selected_experts": experts,
            "test_invocations": layer_invocations,
            "proxy": proxy_facts,
            "storage": storage,
            "wall_seconds": time.time() - layer_started,
        }
        (args.output / "hybrid_facts.json").write_text(json.dumps(facts, indent=2, sort_keys=True) + "\n")
        del q4, w1, train_h, all_h, input_transforms, down_transforms, proxy
        gc.collect(); torch.cuda.empty_cache()

    facts["rows"] = len(rows)
    facts["wall_seconds"] = time.time() - facts["started_unix"]
    pd.DataFrame(rows).to_parquet(args.output / "hybrid_metrics.parquet", index=False)
    (args.output / "hybrid_facts.json").write_text(json.dumps(facts, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"hybrid_complete": True, "rows": len(rows), "wall_seconds": facts["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
