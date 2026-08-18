#!/usr/bin/env python3
"""Bounded Q2 output-side low-rank down-correction oracle (Run B)."""

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


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def qenergy(error: torch.Tensor, proxy: torch.Tensor, beta: float) -> torch.Tensor:
    return torch.sum(error * error, dim=-1) + beta * torch.sum((error @ proxy) ** 2, dim=-1)


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
    from oracle_study.low_rank import (
        activation_weighted_coefficients, kmeans, output_basis, projected_coefficients,
        proxy_output_basis, quantize_coefficients, raw_residual_basis,
    )
    from oracle_study.weights import alignment_statistics, load_gguf_experts, load_hf_experts
    from run_hybrid_remote import progressive_candidates
    from run_phase_a_remote import make_layout, proxy_gradients, select_experts
    from run_phase_a_v2_remote import generalized_full_rank
    from run_progressive_q2_remote import q2_quantize

    args.output.mkdir(parents=True, exist_ok=False)
    loaded = np.load(args.captures, allow_pickle=False)
    data = {name: loaded[name] for name in loaded.files}
    train_requests = set(map(str, data["request_id"][data["split"] == "train"]))
    validation_requests = set(map(str, data["request_id"][data["split"] == "validation"]))
    test_requests = set(map(str, data["request_id"][data["split"] == "test"]))
    if train_requests & validation_requests or train_requests & test_requests or validation_requests & test_requests:
        raise ValueError("request-level split leakage")
    reader = gguf.GGUFReader(config["remote"]["gguf_model"])
    device = "cuda"
    ranks = [int(value) for value in config["ranks"]]
    precisions = [int(value) for value in config["coefficient_precisions"]]
    budgets = [float(value) for value in config["physical_bpw_budgets"]]
    rates = [float(value) for value in config["gate_up_candidate_bpw"]]
    weights = 3 * 2048 * 512
    rows: list[dict[str, Any]] = []
    down_rows: list[dict[str, Any]] = []
    storage_rows: list[dict[str, Any]] = []
    spectra_rows: list[dict[str, Any]] = []
    facts: dict[str, Any] = {"started_unix": time.time(), "layers": {}, "split_requests": {"train": len(train_requests), "validation": len(validation_requests), "test": len(test_requests)}}

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
        train_h: list[np.ndarray] = []
        train_outputs: list[np.ndarray] = []
        residuals: list[np.ndarray] = []
        for ei, expert in enumerate(experts):
            base["gate"].append(q2_quantize(reference["gate"][ei], int(config["binary_group_size"]), moment_x)[0])
            base["up"].append(q2_quantize(reference["up"][ei], int(config["binary_group_size"]), moment_x)[0])
            records = np.unique(np.argwhere((d["expert_ids"] == expert) & (d["split"][:, None] == "train"))[:, 0])[: int(config["max_train_invocations_per_expert"])]
            x = np.asarray(d["x"][records], dtype=np.float32)
            gate, up = x @ reference["gate"][ei].T, x @ reference["up"][ei].T
            train_h.append((gate / (1 + np.exp(-gate))) * up)
        base["gate"], base["up"] = np.stack(base["gate"]), np.stack(base["up"])
        all_h = np.concatenate(train_h)
        moment_h = np.mean(all_h.astype(np.float64) ** 2, axis=0)
        base["down"] = np.stack([q2_quantize(reference["down"][ei], int(config["binary_group_size"]), moment_h)[0] for ei in range(len(experts))])
        for ei in range(len(experts)):
            residual = reference["down"][ei] - base["down"][ei]
            residuals.append(residual)
            train_outputs.append(train_h[ei] @ residual.T)
        all_outputs = np.concatenate(train_outputs)
        proxy_np, proxy_facts = proxy_gradients(d["xplus"], [d[f"h{h}_router_logits"] for h in range(1, 5)], d["split"])
        proxy = torch.from_numpy(proxy_np).to(device)
        beta = float(proxy_facts["beta"])

        shared_bases: dict[str, np.ndarray] = {}
        shared_bases["B0_raw_shared"], raw_singular = raw_residual_basis(np.asarray(residuals), max(ranks), device)
        shared_bases["B1_action_shared"], action_singular = output_basis(all_outputs, max(ranks), device)
        shared_bases["B2_proxy_shared"], proxy_singular = proxy_output_basis(
            all_outputs, proxy_np, beta, max(ranks), device
        )
        for method, singular in (("B0_raw_shared", raw_singular), ("B1_action_shared", action_singular), ("B2_proxy_shared", proxy_singular)):
            total = float(np.sum(singular.astype(np.float64) ** 2))
            for index, value in enumerate(singular.tolist()):
                spectra_rows.append({"layer": layer, "method": method, "component": index + 1, "singular_value": value, "cumulative_energy": float(np.sum(singular[: index + 1].astype(np.float64) ** 2) / max(total, 1e-30))})

        rng = np.random.default_rng(int(config["seed"]) + layer)
        projection = rng.standard_normal((2048, 64), dtype=np.float32) / 8.0
        expert_features = np.stack([np.mean(value.astype(np.float64) ** 2, axis=0).astype(np.float32) @ projection for value in train_outputs])
        cluster_assignments: dict[int, np.ndarray] = {}
        cluster_bases: dict[tuple[int, int], np.ndarray] = {}
        skipped_clusters = []
        for clusters in config["cluster_counts"]:
            clusters = int(clusters)
            if clusters > len(experts):
                skipped_clusters.append(clusters)
                continue
            assignment, _ = kmeans(expert_features, clusters, int(config["seed"]) + layer + clusters)
            cluster_assignments[clusters] = assignment
            for cluster in range(clusters):
                members = np.flatnonzero(assignment == cluster)
                outputs = np.concatenate([train_outputs[index] for index in members])
                basis, _ = output_basis(outputs, max(ranks), device)
                cluster_bases[(clusters, cluster)] = basis

        rg = torch.from_numpy(reference["gate"] - base["gate"]).to(device).reshape(-1, 2048)
        ru = torch.from_numpy(reference["up"] - base["up"]).to(device).reshape(-1, 2048)
        transform = generalized_full_rank(train_x, (((rg.T @ rg) + (ru.T @ ru)) / len(experts)).cpu().numpy())
        del rg, ru
        input_layout = make_layout(transform, train_x)
        reference_bytes = sum(value["bytes_per_expert"] for value in reference_storage.values())
        gate_up_full16_bytes = 2 * (2048 * (512 * 2 + 48))
        layer_invocations = 0

        for ei, expert in enumerate(experts):
            per_expert_basis, _ = raw_residual_basis(np.asarray([residuals[ei]]), max(ranks), device)
            basis_variants: dict[str, tuple[np.ndarray, int | None]] = {name: (basis, None) for name, basis in shared_bases.items()}
            for clusters, assignment in cluster_assignments.items():
                cluster = int(assignment[ei])
                basis_variants[f"B3_action_cluster_{clusters}"] = (cluster_bases[(clusters, cluster)], cluster)
            basis_variants["B4_per_expert_raw"] = (per_expert_basis, ei)
            fitted: dict[tuple[str, int, str, int], tuple[torch.Tensor, torch.Tensor, dict[str, int], int | None]] = {}
            for method, (full_basis, cluster) in basis_variants.items():
                for requested_rank in ranks:
                    actual_rank = min(requested_rank, full_basis.shape[1])
                    basis = full_basis[:, :actual_rank]
                    for fit in ("projection", "activation_ls"):
                        coefficient = projected_coefficients(basis, residuals[ei]) if fit == "projection" else activation_weighted_coefficients(basis, residuals[ei], train_h[ei], proxy_np, beta)
                        for bits in precisions:
                            quantized, byte_facts = quantize_coefficients(coefficient, bits, int(config["binary_group_size"]))
                            fitted[(method, requested_rank, fit, bits)] = (torch.from_numpy(basis).to(device), torch.from_numpy(quantized).to(device), byte_facts, cluster)
                            external = reference_bytes + gate_up_full16_bytes + byte_facts["total_bytes"]
                            resident = basis.nbytes
                            if method == "B4_per_expert_raw":
                                resident *= len(experts)
                            elif method.startswith("B3"):
                                resident *= int(method.rsplit("_", 1)[-1])
                            storage_rows.append({"layer": layer, "expert_id": expert, "method": method, "requested_rank": requested_rank, "actual_rank": actual_rank, "fit": fit, "coefficient_bits": bits, **byte_facts, "external_bytes": external, "external_multiplier": external / reference_bytes, "resident_basis_bytes": resident})
                            if external / reference_bytes > float(config["external_storage_cap_reference_multiplier"]):
                                raise ValueError("low-rank storage cap exceeded")

            occurrence_parts = [
                np.argwhere((d["expert_ids"] == expert) & (d["split"][:, None] == split))[
                    : int(config["max_test_invocations_per_expert"])
                ]
                for split in config.get("evaluation_splits", ["test"])
            ]
            occurrence = np.concatenate([part for part in occurrence_parts if len(part)], axis=0) if any(len(part) for part in occurrence_parts) else np.empty((0, 2), dtype=np.int64)
            if not len(occurrence):
                continue
            layer_invocations += len(occurrence)
            record_ids, router_ranks = occurrence[:, 0], occurrence[:, 1]
            x = torch.from_numpy(np.asarray(d["x"][record_ids], dtype=np.float32)).to(device)
            ref_t = {name: torch.from_numpy(reference[name][ei]).to(device) for name in ("gate", "up", "down")}
            base_t = {name: torch.from_numpy(base[name][ei]).to(device) for name in ("gate", "up", "down")}
            gref, uref = x @ ref_t["gate"].T, x @ ref_t["up"].T
            href = silu(gref) * uref
            yref = href @ ref_t["down"].T
            gbase, ubase = x @ base_t["gate"].T, x @ base_t["up"].T
            hbase = silu(gbase) * ubase
            ybase = hbase @ base_t["down"].T
            base_damage = qenergy(yref - ybase, proxy, beta)
            gate_candidates = progressive_candidates(x, ref_t["gate"] - base_t["gate"], transform, input_layout, rates, [4, 8, 16], gbase, gref)
            up_candidates = progressive_candidates(x, ref_t["up"] - base_t["up"], transform, input_layout, rates, [4, 8, 16], ubase, uref)

            for key, (basis, coefficient, byte_facts, cluster) in fitted.items():
                method, requested_rank, fit, bits = key
                actual_rank = basis.shape[1]
                down_physical = math.ceil(byte_facts["total_bytes"] / int(config["default_page_size"])) * int(config["default_page_size"])
                auth_correction = (href @ coefficient.T) @ basis.T
                auth_base = href @ base_t["down"].T
                auth_error = yref - (auth_base + auth_correction)
                auth_denominator = qenergy(yref - auth_base, proxy, beta)
                auth_recovery = 1.0 - qenergy(auth_error, proxy, beta) / torch.clamp(auth_denominator, min=1e-20)
                for sample, record in enumerate(record_ids.tolist()):
                    common = {"request_id": str(d["request_id"][record]), "sequence_id": str(d["sequence_id"][record]), "position": int(d["position"][record]), "evaluation_split": str(d["split"][record]), "layer": layer, "expert_id": expert, "expert_stratum": strata[expert], "router_rank": int(router_ranks[sample] + 1), "router_coefficient": float(d["router_weights"][record, router_ranks[sample]]), "method": method, "requested_rank": requested_rank, "actual_rank": actual_rank, "fit": fit, "coefficient_bits": bits, "cluster": cluster, "external_storage_multiplier": (reference_bytes + gate_up_full16_bytes + byte_facts["total_bytes"]) / reference_bytes}
                    down_rows.append({**common, "down_input_variant": "authoritative", "down_physical_bytes": down_physical, "down_physical_bpw": 8 * down_physical / (2048 * 512), "recovery": float(auth_recovery[sample])})
                best = {budget: [{"damage": float("inf")} for _ in range(len(x))] for budget in budgets}
                for gate_rate in rates:
                    gcand = gate_candidates[gate_rate]
                    for up_rate in rates:
                        ucand = up_candidates[up_rate]
                        htilde = silu(gbase + gcand["correction"]) * (ubase + ucand["correction"])
                        correction = (htilde @ coefficient.T) @ basis.T
                        yhat = htilde @ base_t["down"].T + correction
                        damage = qenergy(yref - yhat, proxy, beta).detach().cpu().numpy()
                        for sample in range(len(x)):
                            physical = int(gcand["physical"][sample] + ucand["physical"][sample] + down_physical)
                            bpw = 8 * physical / weights
                            for budget in budgets:
                                if bpw <= budget + 1e-12 and damage[sample] < best[budget][sample]["damage"]:
                                    best[budget][sample] = {"damage": float(damage[sample]), "physical": physical, "logical": int(gcand["logical"][sample] + ucand["logical"][sample] + byte_facts["total_bytes"]), "gate_bpw": 8 * gcand["physical"][sample] / (2048 * 512), "up_bpw": 8 * ucand["physical"][sample] / (2048 * 512), "down_bpw": 8 * down_physical / (2048 * 512), "yhat": yhat[sample].detach().clone()}
                for budget, selected in best.items():
                    for sample, value in enumerate(selected):
                        if not math.isfinite(value["damage"]):
                            continue
                        record = int(record_ids[sample])
                        error = yref[sample] - value["yhat"]
                        rows.append({"phase": "B_complete_expert_low_rank", "request_id": str(d["request_id"][record]), "sequence_id": str(d["sequence_id"][record]), "position": int(d["position"][record]), "evaluation_split": str(d["split"][record]), "layer": layer, "expert_id": expert, "expert_stratum": strata[expert], "expert_train_frequency": frequencies[expert], "router_rank": int(router_ranks[sample] + 1), "router_coefficient": float(d["router_weights"][record, router_ranks[sample]]), "method": method, "requested_rank": requested_rank, "actual_rank": actual_rank, "fit": fit, "coefficient_bits": bits, "cluster": cluster, "down_input_variant": "sequential", "budget_bpw": budget, "physical_bytes": value["physical"], "logical_bytes": value["logical"], "physical_bpw": 8 * value["physical"] / weights, "logical_bpw": 8 * value["logical"] / weights, "page_amplification": value["physical"] / max(value["logical"], 1), "gate_bpw": value["gate_bpw"], "up_bpw": value["up_bpw"], "down_bpw": value["down_bpw"], "recovery": 1.0 - value["damage"] / max(float(base_damage[sample]), 1e-20), "damage": value["damage"], "base_damage": float(base_damage[sample]), "relative_output_error": float(torch.linalg.norm(error) / torch.clamp(torch.linalg.norm(yref[sample]), min=1e-20)), "external_storage_multiplier": (reference_bytes + gate_up_full16_bytes + byte_facts["total_bytes"]) / reference_bytes, "c_h_flops": 2 * actual_rank * 512, "u_q_flops": 2 * 2048 * actual_rank, "extra_flops": 2 * actual_rank * (512 + 2048), "resident_basis_bytes": basis.numel() * 2})
            print(json.dumps({"run": "low_rank_down", "layer": layer, "expert": expert, "rows": len(rows)}), flush=True)
            pd.DataFrame(rows).to_parquet(args.output / "low_rank_complete_expert_metrics.parquet", index=False)
            pd.DataFrame(down_rows).to_parquet(args.output / "low_rank_down_metrics.parquet", index=False)
            del fitted, ref_t, base_t, x
            gc.collect(); torch.cuda.empty_cache()

        facts["layers"][str(layer)] = {"experts": experts, "test_invocations": layer_invocations, "alignment": alignment, "proxy": proxy_facts, "skipped_cluster_counts": skipped_clusters, "wall_seconds": time.time() - layer_start}
        del reference, base, residuals, train_h, train_outputs, all_outputs, shared_bases, cluster_bases, transform, proxy
        gc.collect(); torch.cuda.empty_cache()

    facts["rows"] = len(rows)
    facts["wall_seconds"] = time.time() - facts["started_unix"]
    pd.DataFrame(rows).to_parquet(args.output / "low_rank_complete_expert_metrics.parquet", index=False)
    pd.DataFrame(down_rows).to_parquet(args.output / "low_rank_down_metrics.parquet", index=False)
    pd.DataFrame(storage_rows).to_csv(args.output / "low_rank_storage_accounting.csv", index=False)
    pd.DataFrame(spectra_rows).to_parquet(args.output / "low_rank_spectra.parquet", index=False)
    (args.output / "low_rank_facts.json").write_text(json.dumps(facts, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"complete": True, "rows": len(rows), "down_rows": len(down_rows), "wall_seconds": facts["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
