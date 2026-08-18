#!/usr/bin/env python3
"""Bounded nested-prefix versus independent atom-quantizer control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


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
    from oracle_study.page_allocator import independent_level, nested_levels
    from oracle_study.weights import load_gguf_experts
    from run_phase_a_remote import select_experts
    from run_phase_a_v2_remote import generalized_full_rank
    from run_progressive_q2_remote import q2_quantize

    loaded = np.load(args.captures, allow_pickle=False)
    data = {name: loaded[name] for name in loaded.files}
    reader = gguf.GGUFReader(config["remote"]["gguf_model"])
    rows = []
    for layer in config["layers"]:
        mask = data["layer"] == layer
        d = {name: value[mask] for name, value in data.items()}
        strata, _ = select_experts(d["expert_ids"], d["split"], int(config["experts_per_frequency_stratum"]))
        experts = sorted(strata)
        reference, _ = load_gguf_experts(reader, layer, experts, gguf)
        train_x = np.asarray(d["x"][d["split"] == "train"], dtype=np.float32)
        moment_x = np.mean(train_x.astype(np.float64) ** 2, axis=0)
        gate_base = np.stack([q2_quantize(value, int(config["binary_group_size"]), moment_x)[0] for value in reference["gate"]])
        up_base = np.stack([q2_quantize(value, int(config["binary_group_size"]), moment_x)[0] for value in reference["up"]])
        train_h = []
        for index, expert in enumerate(experts):
            records = np.unique(np.argwhere((d["expert_ids"] == expert) & (d["split"][:, None] == "train"))[:, 0])[: int(config["max_train_invocations_per_expert"])]
            x = np.asarray(d["x"][records], dtype=np.float32)
            gate, up = x @ reference["gate"][index].T, x @ reference["up"][index].T
            train_h.append((gate / (1 + np.exp(-gate))) * up)
        moment_h = np.mean(np.concatenate(train_h).astype(np.float64) ** 2, axis=0)
        down_base = np.stack([q2_quantize(value, int(config["binary_group_size"]), moment_h)[0] for value in reference["down"]])
        rg = torch.from_numpy(reference["gate"] - gate_base).cuda().reshape(-1, 2048)
        ru = torch.from_numpy(reference["up"] - up_base).cuda().reshape(-1, 2048)
        transform = generalized_full_rank(train_x, (((rg.T @ rg) + (ru.T @ ru)) / len(experts)).cpu().numpy())
        index = 0
        atoms = {
            "gate": (reference["gate"][index] - gate_base[index]) @ transform.synthesis,
            "up": (reference["up"][index] - up_base[index]) @ transform.synthesis,
            "down": reference["down"][index] - down_base[index],
        }
        for projection, matrix in atoms.items():
            denominator = float(np.sum(matrix.astype(np.float64) ** 2))
            for bits in (2, 4, 8):
                for quantizer in ("nested_prefix", "independent"):
                    reconstructed = np.stack([
                        nested_levels(matrix[:, atom])[bits] if quantizer == "nested_prefix" else independent_level(matrix[:, atom], bits)
                        for atom in range(matrix.shape[1])
                    ], axis=1)
                    atom_denominator = np.sum(matrix.astype(np.float64) ** 2, axis=0)
                    atom_error = np.sum((matrix - reconstructed).astype(np.float64) ** 2, axis=0) / np.maximum(atom_denominator, 1e-30)
                    rows.append({
                        "layer": layer, "expert_id": experts[index], "projection": projection,
                        "bits": bits, "quantizer": quantizer,
                        "relative_atom_mse": float(np.sum((matrix - reconstructed).astype(np.float64) ** 2) / max(denominator, 1e-30)),
                        "median_per_atom_relative_mse": float(np.median(atom_error)),
                        "p90_per_atom_relative_mse": float(np.quantile(atom_error, .9)),
                    })
        del rg, ru
        torch.cuda.empty_cache()
        print(json.dumps({"layer": layer, "expert": experts[index]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
