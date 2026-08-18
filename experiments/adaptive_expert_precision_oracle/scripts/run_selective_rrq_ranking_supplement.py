#!/usr/bin/env python3
"""Fill missing middle/final-layer selective-RRQ ranking audit cases.

The main runner deliberately audits only one invocation per layer.  If its
training-frequency-selected first audit expert has no held-out occurrence, no
row is produced.  This supplement chooses the first stratified audit expert
with held-out coverage and writes a separate Parquet file; it never rewrites
the main scientific outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from run_selective_rrq_retention import (
    PROJECTIONS,
    atom_series,
    beam_prefix,
    choose_search_experts,
    complete_queue_greedy,
    exact_stage_support,
    nested_energy_actions,
    occurrence,
    records_to_specs,
    silu,
    stage_packet_facts,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--dense-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[20, 39])
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    experiment = args.config.resolve().parents[1]
    sys.path[:0] = [str(experiment / "src"), str(experiment / "scripts"), config["remote"]["gguf_python"]]
    import gguf
    from oracle_study.bases import identity_transform
    from oracle_study.rrq import encode_legacy_q2
    from oracle_study.weights import load_gguf_experts
    from run_phase_a_remote import proxy_gradients, select_experts
    from run_phase_a_v2_remote import generalized_full_rank

    loaded = np.load(args.captures, allow_pickle=False)
    all_data = {key: loaded[key] for key in loaded.files}
    selected = json.loads((args.dense_result / "quantizer_selection.json").read_text())
    paths = {projection: records_to_specs(selected[projection]["heterogeneous"]) for projection in PROJECTIONS}
    reader = gguf.GGUFReader(config["remote"]["gguf_model"])
    page_size = int(config["page_size"])
    budgets = np.asarray(config["physical_budgets_bpw"], np.float64)
    rows: list[dict[str, object]] = []

    for layer in args.layers:
        mask = all_data["layer"] == layer
        data = {key: value[mask] for key, value in all_data.items()}
        strata, _ = select_experts(data["expert_ids"], data["split"], int(config["experts_per_frequency_stratum"]))
        experts = sorted(strata)
        reference, _ = load_gguf_experts(reader, layer, experts, gguf)
        train_x = np.asarray(data["x"][data["split"] == "train"], np.float32)
        moment_x = np.mean(train_x.astype(np.float64) ** 2, axis=0)
        train_h = []
        for expert_index, expert in enumerate(experts):
            records, _ = occurrence(data, expert, "train", int(config["max_train_invocations_per_expert"]))
            if len(records):
                x = np.asarray(data["x"][records], np.float32)
                train_h.append(silu(x @ reference["gate"][expert_index].T) * (x @ reference["up"][expert_index].T))
        moment_h = np.mean(np.concatenate(train_h).astype(np.float64) ** 2, axis=0)
        base = {projection: [] for projection in PROJECTIONS}
        for projection in PROJECTIONS:
            moment = moment_x if projection != "down" else moment_h
            for matrix in reference[projection]:
                base[projection].append(encode_legacy_q2(matrix, int(config["base_group_size"]), moment, "fp16").reconstruction)
            base[projection] = np.stack(base[projection])
        rg = torch.from_numpy(reference["gate"] - base["gate"]).cuda().reshape(-1, 2048)
        ru = torch.from_numpy(reference["up"] - base["up"]).cuda().reshape(-1, 2048)
        operator = ((rg.T @ rg) + (ru.T @ ru)).cpu().numpy() / len(experts)
        del rg, ru
        torch.cuda.empty_cache()
        transform = generalized_full_rank(train_x, operator)
        proxy, proxy_facts = proxy_gradients(data["xplus"], [data[f"h{h}_router_logits"] for h in range(1, 5)], data["split"])

        preferred = choose_search_experts(experts, strata, data)
        covered = [expert for expert in preferred + experts if len(occurrence(data, expert, "test", 1)[0])]
        if not covered:
            raise RuntimeError(f"no held-out audit expert at layer {layer}")
        expert = covered[0]
        expert_index = experts.index(expert)
        record = int(occurrence(data, expert, "test", 1)[0][0])
        items = {}
        for projection in PROJECTIONS:
            basis = transform if projection != "down" else identity_transform(512)
            items[projection] = atom_series(reference[projection][expert_index], base[projection][expert_index], basis, paths[projection])
        x = np.asarray(data["x"][record], np.float32)
        gref = reference["gate"][expert_index] @ x
        uref = reference["up"][expert_index] @ x
        href = silu(gref) * uref
        yref = reference["down"][expert_index] @ href
        gbase = base["gate"][expert_index] @ x
        ubase = base["up"][expert_index] @ x
        hbase = silu(gbase) * ubase
        coefficients = {"gate": items["gate"]["analysis"] @ x, "up": items["up"]["analysis"] @ x, "down": href}
        orders = {projection: nested_energy_actions(items[projection], coefficients[projection], page_size) for projection in PROJECTIONS}
        common = {"request_id": str(data["request_id"][record]), "layer": layer, "expert_id": expert}
        for point in complete_queue_greedy(items, orders, coefficients, gbase, ubase, base["down"][expert_index], yref, proxy, float(proxy_facts["beta"]), budgets):
            rows.append({**common, "projection": "complete_expert", "stage": 0, "comparison": "complete_expert_queue_greedy", "target_recovery": None, **point})
        for projection in PROJECTIONS:
            z = coefficients[projection]
            for stage_index, stage in enumerate(items[projection]["stages"], 1):
                contributions = (stage * z[None, :]).T
                energy_k, _ = exact_stage_support(contributions, list(map(float, config["support_targets"])), False)
                exact_k, _ = exact_stage_support(contributions, list(map(float, config["support_targets"])), True)
                packet = stage_packet_facts(items[projection]["series"], page_size)[stage_index - 1]["physical"]
                for target in map(float, config["support_targets"]):
                    extra = None if energy_k[target] is None or exact_k[target] is None else energy_k[target] - exact_k[target]
                    rows.append({**common, "projection": projection, "stage": stage_index, "comparison": "energy_vs_exact_residual_greedy", "target_recovery": target, "energy_atoms": energy_k[target], "exact_atoms": exact_k[target], "extra_atoms": extra, "packet_physical_bytes": packet, "extra_projection_bpw": None if extra is None else 8 * extra * packet / (2048 * 512)})
                for width in config["beam_widths"]:
                    for point in beam_prefix(contributions, int(width), int(config["beam_candidate_pool"])):
                        rows.append({**common, "projection": projection, "stage": stage_index, "comparison": f"bounded_beam_w{width}_pool{config['beam_candidate_pool']}", "target_recovery": None, **point})
        print(json.dumps({"completed_supplement_layer": layer, "expert": expert, "rows": len(rows)}), flush=True)
        del reference, base, items
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()
