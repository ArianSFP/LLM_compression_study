#!/usr/bin/env python3
"""Training-only spectra and concentration diagnostics for the three-layer pilot."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch


def spectral_stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    values = np.sort(values)[::-1]
    total = max(float(values.sum()), 1e-30)
    p = values / total
    positive = p[p > 0]
    cumulative = np.cumsum(p)
    result: dict[str, float | int] = {
        "dimension": int(len(values)),
        "entropy_effective_rank": float(np.exp(-np.sum(positive * np.log(positive)))),
        "participation_rank": float(1.0 / max(np.sum(p * p), 1e-30)),
    }
    for threshold in (0.5, 0.8, 0.9, 0.95, 0.99):
        result[f"rank_at_{threshold}"] = int(np.searchsorted(cumulative, threshold) + 1)
    for count in (8, 16, 32, 64, 128, 256, 512):
        result[f"energy_top_{count}"] = float(cumulative[min(count, len(cumulative)) - 1])
    return result


def spectrum(samples: np.ndarray) -> np.ndarray:
    value = torch.from_numpy(np.asarray(samples, dtype=np.float32)).to("cuda")
    gram = value @ value.T
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min_(0).flip(0)
    return eigenvalues.cpu().numpy()


def gini(values: np.ndarray) -> float:
    x = np.sort(np.maximum(np.asarray(values, dtype=np.float64), 0.0))
    if not len(x) or x.sum() == 0:
        return 0.0
    index = np.arange(1, len(x) + 1)
    return float((2 * np.sum(index * x) / np.sum(x) - (len(x) + 1)) / len(x))


def basis_concentration(
    transform: object,
    residuals: list[np.ndarray],
    sample_sets: list[np.ndarray],
) -> dict[str, float | int]:
    aggregate = np.zeros(transform.analysis.shape[0], dtype=np.float64)
    per_sample_counts = {threshold: [] for threshold in (0.8, 0.9, 0.95, 0.99)}
    for residual, samples in zip(residuals, sample_sets):
        codes = np.asarray(samples, dtype=np.float32) @ transform.analysis.T
        atoms = np.asarray(residual, dtype=np.float32) @ transform.synthesis
        score = codes.astype(np.float64) ** 2 * np.sum(atoms.astype(np.float64) ** 2, axis=0)[None, :]
        aggregate += score.sum(axis=0)
        ordered = np.sort(score, axis=1)[:, ::-1]
        cumulative = np.cumsum(ordered, axis=1) / np.maximum(ordered.sum(axis=1, keepdims=True), 1e-30)
        for threshold in per_sample_counts:
            per_sample_counts[threshold].extend((np.argmax(cumulative >= threshold, axis=1) + 1).tolist())
    result = spectral_stats(aggregate)
    for threshold, counts in per_sample_counts.items():
        result[f"median_atoms_score_{threshold}"] = float(np.median(counts)) if counts else float("nan")
        result[f"p90_atoms_score_{threshold}"] = float(np.quantile(counts, 0.9)) if counts else float("nan")
    return result


def coherence(atoms: np.ndarray) -> dict[str, float]:
    value = torch.from_numpy(np.asarray(atoms, dtype=np.float32)).to("cuda")
    value = value / torch.clamp(torch.linalg.norm(value, dim=0, keepdim=True), min=1e-12)
    gram = torch.abs(value.T @ value)
    gram.fill_diagonal_(0)
    flat = gram.flatten()
    return {
        "mean_abs_coherence": float(flat.mean()),
        "p95_abs_coherence": float(torch.quantile(flat, 0.95)),
        "max_abs_coherence": float(flat.max()),
    }


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
    from oracle_study.bases import identity_transform, pca_transform
    from oracle_study.quant import binary_quantize
    from oracle_study.weights import load_gguf_experts
    from run_phase_a_remote import select_experts
    from run_phase_a_v2_remote import generalized_full_rank

    args.output.mkdir(parents=True, exist_ok=True)
    loaded = np.load(args.captures, allow_pickle=False)
    data = {name: loaded[name] for name in loaded.files}
    reader = gguf.GGUFReader(config["remote"]["gguf_model"])
    rows = []
    spectra_rows = []
    started = time.time()

    for layer in config["layers"]:
        mask = data["layer"] == layer
        d = {name: value[mask] for name, value in data.items()}
        strata, _ = select_experts(d["expert_ids"], d["split"], int(config["experts_per_frequency_stratum"]))
        experts = sorted(strata)
        q4, _ = load_gguf_experts(reader, layer, experts, gguf)
        train_x = np.asarray(d["x"][d["split"] == "train"], dtype=np.float32)
        moment_x = np.mean(train_x.astype(np.float64) ** 2, axis=0)
        residuals = {"gate": [], "up": [], "down": []}
        samples = {"gate": [], "up": [], "down": []}
        train_h = []
        w1_gate, w1_up = [], []
        for ei, expert in enumerate(experts):
            wg = binary_quantize(q4["gate"][ei], int(config["binary_group_size"]), moment_x)[0]
            wu = binary_quantize(q4["up"][ei], int(config["binary_group_size"]), moment_x)[0]
            w1_gate.append(wg); w1_up.append(wu)
            occurrence = np.argwhere((d["expert_ids"] == expert) & (d["split"][:, None] == "train"))[:256, 0]
            x = np.asarray(d["x"][occurrence], dtype=np.float32)
            g = x @ q4["gate"][ei].T
            u = x @ q4["up"][ei].T
            h = (g / (1.0 + np.exp(-g))) * u
            train_h.append(h)
            residuals["gate"].append(q4["gate"][ei] - wg)
            residuals["up"].append(q4["up"][ei] - wu)
            samples["gate"].append(x); samples["up"].append(x)
        all_h = np.concatenate(train_h)
        moment_h = np.mean(all_h.astype(np.float64) ** 2, axis=0)
        for ei in range(len(experts)):
            wd = binary_quantize(q4["down"][ei], int(config["binary_group_size"]), moment_h)[0]
            residuals["down"].append(q4["down"][ei] - wd)
            samples["down"].append(train_h[ei])

        for input_name, values in (("pre_moe_x", train_x), ("swiglu_h", all_h)):
            eig = spectrum(values)
            stats = spectral_stats(eig)
            rows.append({"layer": layer, "projection": input_name, "basis": "activation_covariance", **stats})
            spectra_rows.extend({"layer": layer, "projection": input_name, "spectrum": "activation", "rank": i + 1, "eigenvalue": float(v)} for i, v in enumerate(eig))

        for projection in ("gate", "up", "down"):
            actions = []
            for residual, values in zip(residuals[projection], samples[projection]):
                actions.append(np.asarray(values, dtype=np.float32) @ np.asarray(residual, dtype=np.float32).T)
            action_values = np.concatenate(actions)
            eig = spectrum(action_values)
            stats = spectral_stats(eig)
            column_norms = np.concatenate([np.linalg.norm(value, axis=0) for value in residuals[projection]])
            base = {
                "layer": layer,
                "projection": projection,
                **stats,
                "residual_column_norm_cv": float(np.std(column_norms) / max(np.mean(column_norms), 1e-30)),
                "residual_column_norm_gini": gini(column_norms),
                "train_action_samples": int(len(action_values)),
            }
            spectra_rows.extend({"layer": layer, "projection": projection, "spectrum": "residual_action", "rank": i + 1, "eigenvalue": float(v)} for i, v in enumerate(eig))
            stacked_residual = torch.from_numpy(np.stack(residuals[projection])).to("cuda")
            operator = torch.einsum("eoi,eoj->ij", stacked_residual, stacked_residual).cpu().numpy() / len(experts)
            fit_samples = train_x if projection in ("gate", "up") else all_h
            methods = {
                "native": identity_transform(fit_samples.shape[1]),
                "pca": pca_transform(fit_samples),
                "generalized": generalized_full_rank(fit_samples, operator),
            }
            for method, transform in methods.items():
                concentration = basis_concentration(transform, residuals[projection], samples[projection])
                atoms = residuals[projection][0] @ transform.synthesis
                rows.append({**base, "basis": method, **{f"concentration_{k}": v for k, v in concentration.items()}, **coherence(atoms)})
            del action_values, stacked_residual, operator
            torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(args.output / "structure_diagnostics.csv", index=False)
        pd.DataFrame(spectra_rows).to_parquet(args.output / "structure_spectra.parquet", index=False)
        del q4, train_h, all_h, residuals, samples
        gc.collect(); torch.cuda.empty_cache()

    facts = {"wall_seconds": time.time() - started, "rows": len(rows), "spectrum_rows": len(spectra_rows)}
    (args.output / "structure_diagnostics_facts.json").write_text(json.dumps(facts, indent=2) + "\n")
    print(json.dumps(facts), flush=True)


if __name__ == "__main__":
    main()
