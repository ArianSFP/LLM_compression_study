#!/usr/bin/env python3
"""Aggregate and plot the PR #13 paired-tail causal replay pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT = Path(__file__).resolve().parents[1]
PROPAGATION_SUMMARY = "causal_propagation_summary.csv"
QUALITY_SUMMARY = "causal_quality_by_layer.csv"
RATE_SUMMARY = "causal_rate_summary.csv"
REPORT = "CAUSAL_DOWNSTREAM_REPLAY_REPORT.md"
QUALITY_PNG = "causal_quality_by_layer.png"
QUALITY_SVG = "causal_quality_by_layer.svg"
MANIFEST = "causal_replay_analysis_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float:
    values_array = np.asarray(list(values), np.float64)
    weights_array = np.asarray(list(weights), np.float64)
    return float(np.sum(values_array * weights_array) / np.sum(weights_array))


def bootstrap_interval(
    frame: pd.DataFrame,
    column: str,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    rows = list(frame.itertuples(index=False))
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, np.float64)
    for index in range(samples):
        selected = rng.integers(0, len(rows), len(rows))
        values = [float(getattr(rows[i], column)) for i in selected]
        weights = [float(getattr(rows[i], "label_tokens")) for i in selected]
        estimates[index] = weighted_mean(values, weights)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def validate_inputs(
    config: dict[str, Any],
    reconstruction: dict[str, Any],
    baseline: dict[str, Any],
    impulse: dict[str, Any],
    propagation_path: Path,
    quality_path: Path,
) -> None:
    if config["run_id"] != reconstruction.get("run_id"):
        raise RuntimeError("reconstruction run ID changed")
    if config["run_id"] != impulse.get("run_id"):
        raise RuntimeError("impulse run ID changed")
    if reconstruction.get("causal_replay_admitted") is not True:
        raise RuntimeError("reconstruction parity did not admit replay")
    if reconstruction.get("test_scientific_rows_admitted_or_used") is not False:
        raise RuntimeError("reconstruction test-row boundary changed")
    if impulse.get("test_scientific_rows_admitted_or_used") is not False:
        raise RuntimeError("impulse test-row boundary changed")
    if len(impulse.get("layers", {})) != 40:
        raise RuntimeError("impulse result does not contain forty layers")
    if impulse.get("propagation_sha256") != sha256(propagation_path):
        raise RuntimeError("propagation table changed")
    if impulse.get("quality_sha256") != sha256(quality_path):
        raise RuntimeError("quality table changed")
    no_op = ("no_op_hidden", "no_op_router", "no_op_logits")
    if not all(baseline.get("gates", {}).get(name) is True for name in no_op):
        raise RuntimeError("same-process no-op gates changed")


def propagation_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = [
        "mean_budget_pages_per_expert",
        "charged_bpw",
        "injection_layer",
        "observation_layer",
        "layer_distance",
    ]
    for key, group in frame.groupby(keys, sort=True):
        all_weights = group["sequence_tokens"].to_numpy(np.float64)
        injected_weights = group["injected_positions"].to_numpy(np.float64)
        all_error = np.sum(
            group["hidden_error_energy_all_tokens"].to_numpy(np.float64)
            * all_weights
        )
        all_input = np.sum(
            group["realized_delta_energy_all_tokens"].to_numpy(np.float64)
            * all_weights
        )
        captured_error = np.sum(
            group["hidden_error_energy_captured_positions"].to_numpy(np.float64)
            * injected_weights
        )
        captured_input = np.sum(
            group["realized_delta_energy"].to_numpy(np.float64)
            * injected_weights
        )
        rows.append({
            **dict(zip(keys, key)),
            "requests": int(group["request_id"].nunique()),
            "sequence_tokens": int(all_weights.sum()),
            "injected_positions": int(injected_weights.sum()),
            "realized_delta_energy_all_tokens": float(all_input / all_weights.sum()),
            "hidden_error_energy_all_tokens": float(all_error / all_weights.sum()),
            "propagation_coefficient_all_tokens": float(all_error / all_input),
            "propagation_coefficient_captured_positions": float(
                captured_error / captured_input
            ),
            "hidden_mse": weighted_mean(group["hidden_mse"], all_weights),
            "rmsnorm_normalized_mse": weighted_mean(
                group["rmsnorm_normalized_mse"], all_weights,
            ),
            "cosine_distance": weighted_mean(group["cosine_distance"], all_weights),
            "router_logit_kl": weighted_mean(group["router_logit_kl"], all_weights),
            "route_churn": weighted_mean(group["route_churn"], all_weights),
            "route_top1_agreement": weighted_mean(
                group["route_top1_agreement"], all_weights,
            ),
            "route_ordered_top8_agreement": weighted_mean(
                group["route_ordered_top8_agreement"], all_weights,
            ),
            "baseline_capture_hidden_mse": weighted_mean(
                group["baseline_capture_hidden_mse"], all_weights,
            ),
            "baseline_capture_hidden_max_abs": float(
                group["baseline_capture_hidden_max_abs"].max()
            ),
            "baseline_capture_route_churn": weighted_mean(
                group["baseline_capture_route_churn"], all_weights,
            ),
        })
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def quality_summary(frame: pd.DataFrame, samples: int) -> pd.DataFrame:
    rows = []
    keys = ["mean_budget_pages_per_expert", "charged_bpw", "injection_layer"]
    for key, group in frame.groupby(keys, sort=True):
        weights = group["label_tokens"].to_numpy(np.float64)
        delta_nll = weighted_mean(group["delta_nll"], weights)
        delta_low, delta_high = bootstrap_interval(
            group, "delta_nll", samples, 9000 + int(key[0]) + int(key[2]),
        )
        kl_low, kl_high = bootstrap_interval(
            group, "logit_kl", samples, 19000 + int(key[0]) + int(key[2]),
        )
        logit_kl = weighted_mean(group["logit_kl"], weights)
        rows.append({
            **dict(zip(keys, key)),
            "requests": int(group["request_id"].nunique()),
            "label_tokens": int(weights.sum()),
            "delta_nll": delta_nll,
            "delta_nll_ci_low": delta_low,
            "delta_nll_ci_high": delta_high,
            "ppl_ratio": float(np.exp(delta_nll)),
            "ppl_ratio_ci_low": float(np.exp(delta_low)),
            "ppl_ratio_ci_high": float(np.exp(delta_high)),
            "logit_kl": logit_kl,
            "logit_kl_ci_low": kl_low,
            "logit_kl_ci_high": kl_high,
            "top1_agreement": weighted_mean(group["top1_agreement"], weights),
            "realized_delta_energy_all_tokens": weighted_mean(
                group["realized_delta_energy_all_tokens"],
                group["sequence_tokens"],
            ),
        })
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def rate_summary(
    propagation: pd.DataFrame,
    quality: pd.DataFrame,
) -> pd.DataFrame:
    diagonal = propagation[
        propagation["injection_layer"].eq(propagation["observation_layer"])
    ]
    final = propagation[propagation["observation_layer"].eq(39)]
    rows = []
    for rate, group in quality.groupby("mean_budget_pages_per_expert", sort=True):
        local = diagonal[diagonal["mean_budget_pages_per_expert"].eq(rate)]
        tail = final[final["mean_budget_pages_per_expert"].eq(rate)]
        rows.append({
            "mean_budget_pages_per_expert": int(rate),
            "charged_bpw": float(group["charged_bpw"].iloc[0]),
            "isolated_layers": len(group),
            "initial_realized_energy_median": float(
                local["realized_delta_energy_all_tokens"].median()
            ),
            "final_hidden_mse_median": float(tail["hidden_mse"].median()),
            "final_hidden_mse_p90": float(tail["hidden_mse"].quantile(0.9)),
            "final_route_churn_median": float(tail["route_churn"].median()),
            "final_route_churn_p90": float(tail["route_churn"].quantile(0.9)),
            "logit_kl_median": float(group["logit_kl"].median()),
            "logit_kl_p90": float(group["logit_kl"].quantile(0.9)),
            "logit_kl_max": float(group["logit_kl"].max()),
            "absolute_delta_nll_median": float(group["delta_nll"].abs().median()),
            "absolute_delta_nll_p90": float(group["delta_nll"].abs().quantile(0.9)),
            "top1_agreement_mean": float(group["top1_agreement"].mean()),
        })
    return pd.DataFrame(rows)


def save_svg(fig: Any, path: Path) -> None:
    fig.savefig(path, metadata={"Date": None})
    atomic_text(
        path,
        "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n",
    )


def heatmaps(frame: pd.DataFrame, output: Path, run_id: str) -> list[Path]:
    matplotlib.rcParams["svg.hashsalt"] = run_id
    paths = []
    specifications = [
        ("propagation_coefficient_all_tokens", "log10 propagation coefficient", True),
        ("hidden_mse", "log10 hidden-state MSE", True),
        ("rmsnorm_normalized_mse", "log10 RMSNorm-normalized MSE", True),
        ("route_churn", "top-8 route churn (%)", False),
    ]
    for rate, group in frame.groupby("mean_budget_pages_per_expert", sort=True):
        fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.0), constrained_layout=True)
        for axis, (column, title, logarithmic) in zip(axes.flat, specifications):
            matrix = np.full((40, 40), np.nan, np.float64)
            for row in group.itertuples(index=False):
                value = float(getattr(row, column))
                if logarithmic:
                    value = np.log10(max(value, np.finfo(np.float64).tiny))
                else:
                    value *= 100.0
                matrix[int(row.injection_layer), int(row.observation_layer)] = value
            image = axis.imshow(matrix, origin="lower", aspect="equal", cmap="viridis")
            axis.set_title(title)
            axis.set_xlabel("observation layer")
            axis.set_ylabel("injection layer")
            axis.set_xticks(np.arange(0, 40, 5))
            axis.set_yticks(np.arange(0, 40, 5))
            fig.colorbar(image, ax=axis, shrink=0.82)
        charged = float(group["charged_bpw"].iloc[0])
        fig.suptitle(
            f"Paired-tail impulse propagation: {charged:.6f} charged bpw",
            fontsize=14,
        )
        stem = f"causal_impulse_heatmap_pages_{int(rate)}"
        png = output / f"{stem}.png"
        svg = output / f"{stem}.svg"
        fig.savefig(png, dpi=180)
        save_svg(fig, svg)
        plt.close(fig)
        paths.extend([png, svg])
    return paths


def quality_plot(frame: pd.DataFrame, output: Path) -> list[Path]:
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.0), sharex=True)
    for rate, group in frame.groupby("mean_budget_pages_per_expert", sort=True):
        group = group.sort_values("injection_layer")
        label = f"{float(group['charged_bpw'].iloc[0]):.6f} bpw"
        axes[0].plot(group["injection_layer"], group["delta_nll"], marker=".", label=label)
        axes[0].fill_between(
            group["injection_layer"],
            group["delta_nll_ci_low"],
            group["delta_nll_ci_high"],
            alpha=0.15,
        )
        axes[1].plot(group["injection_layer"], group["logit_kl"], marker=".", label=label)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("paired delta NLL")
    axes[1].set_ylabel("final-logit KL")
    axes[1].set_xlabel("isolated injection layer")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    png = output / QUALITY_PNG
    svg = output / QUALITY_SVG
    fig.savefig(png, dpi=180)
    save_svg(fig, svg)
    plt.close(fig)
    return [png, svg]


def report_text(
    rates: pd.DataFrame,
    quality: pd.DataFrame,
    propagation: pd.DataFrame,
    reconstruction: dict[str, Any],
    baseline: dict[str, Any],
) -> str:
    lines = [
        "# PR #13 causal downstream replay pilot",
        "",
        "## Outcome",
        "",
        "The reconstruction audit passes exactly and the paired-tail impulse study "
        "covers all 40 MoE layers at the two initial PR #13 rates. This establishes "
        "a causal propagation measurement, but it does **not** yet establish an "
        "end-to-end streamed-model quality curve.",
        "",
        "## Hard gates",
        "",
        f"- Validation groups reconstructed: {int(reconstruction['validation_rows']):,}.",
        f"- Group/rate reconstructions: {int(reconstruction['reconstructions']):,}.",
        f"- Max expert qenergy error: {reconstruction['max_expert_qenergy_abs_error']:.3e}.",
        f"- Max group qenergy error: {reconstruction['max_group_qenergy_abs_error']:.3e}.",
        f"- Max all-Q4 reconstruction error: {reconstruction['max_all_q4_abs_error']:.3e}.",
        "- Repeated paired-tail hidden/router/logit differences: exactly zero for every layer.",
        "- Test rows admitted or used: false.",
        "",
        "## Isolated impulse magnitude by charged rate",
        "",
        "| Charged bpw | Initial error energy median | Final hidden MSE median / p90 | "
        "Final route churn median / p90 | Logit KL median / p90 | Median abs delta NLL |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rates.itertuples(index=False):
        lines.append(
            f"| {row.charged_bpw:.6f} | {row.initial_realized_energy_median:.3e} | "
            f"{row.final_hidden_mse_median:.3e} / {row.final_hidden_mse_p90:.3e} | "
            f"{100*row.final_route_churn_median:.3f}% / "
            f"{100*row.final_route_churn_p90:.3f}% | "
            f"{row.logit_kl_median:.3e} / {row.logit_kl_p90:.3e} | "
            f"{row.absolute_delta_nll_median:.4f} |"
        )
    lines += [
        "",
        "Each row summarizes 40 separate single-layer interventions; effects are not "
        "summed. Negative delta NLL values occur on this tiny cohort, so absolute "
        "delta NLL and non-negative logit KL are more useful impulse magnitudes here.",
        "",
        "## Most logit-sensitive isolated layers",
        "",
        "| Charged bpw | Layer | Delta NLL (95% sequence bootstrap) | Logit KL | "
        "Top-1 agreement | Final route churn |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    final = propagation[propagation["observation_layer"].eq(39)]
    for rate, group in quality.groupby("mean_budget_pages_per_expert", sort=True):
        for row in group.nlargest(5, "logit_kl").itertuples(index=False):
            tail = final[
                final["mean_budget_pages_per_expert"].eq(rate)
                & final["injection_layer"].eq(row.injection_layer)
            ].iloc[0]
            lines.append(
                f"| {row.charged_bpw:.6f} | {int(row.injection_layer)} | "
                f"{row.delta_nll:+.4f} [{row.delta_nll_ci_low:+.4f}, "
                f"{row.delta_nll_ci_high:+.4f}] | {row.logit_kl:.3e} | "
                f"{100*row.top1_agreement:.2f}% | {100*tail.route_churn:.3f}% |"
            )
    lines += [
        "",
        "## Precision and trajectory controls",
        "",
        f"A fresh full GPU forward differs from the old CPU capture by RMSE "
        f"{baseline['capture_xplus_rmse']:.6f} and max absolute hidden error "
        f"{baseline['capture_xplus_max_abs']:.6f}; exact ordered top-8 routes match "
        f"on {100*baseline['capture_route_exact_fraction']:.2f}% of stored rows. "
        "Therefore each intervention begins at the captured BF16 `xplus`, and its "
        "candidate and no-op tails run through the same resident GPU model. The "
        "no-op/candidate difference is causal within that paired trajectory. Drift "
        "from the historical CPU tail is retained in the propagation table as a "
        "separate control.",
        "",
        "The approximately 0.999-bpw local errors are much smaller than the 0.523-bpw "
        "errors, but their final absolute hidden and logit disturbances are not "
        "proportionally smaller. In BF16, small perturbations can cross rounding and "
        "routing boundaries, so very large relative propagation coefficients at the "
        "high rate should be read together with absolute MSE, route churn and logit KL.",
        "",
        "## Scientific boundary and next gate",
        "",
        "- Cohort: three validation sequences and 32 captured positions per layer.",
        "- Modes: isolated frozen PR #13 error at 0.523478 and 0.998739 charged bpw.",
        "- Attention: complete sequence tails; errors can affect later positions.",
        "- Not measured: simultaneous streaming of all layers, live allocation, "
        "physical wire bpw, repair-one-layer importance, or population perplexity.",
        "- Predicted-H4 is explicitly excluded; a later predictor starts at H1.",
        "",
        "The next quality gate is a supplementary full-sequence held-out capture and "
        "a genuine sequential streamed forward at all four PR #13 rates. Only that "
        "run can report the requested bpw-to-NLL/PPL curve, cross-layer interaction "
        "Gamma, and repair-one-layer importance.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reconstruction-dir", type=Path, required=True)
    parser.add_argument("--impulse-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    config = load_json(args.config)
    reconstruction_path = args.reconstruction_dir / "reconstruction_run_facts.json"
    baseline_path = args.impulse_dir / "baseline_replay_facts.json"
    impulse_path = args.impulse_dir / "impulse_run_facts.json"
    propagation_path = args.impulse_dir / "impulse_propagation.parquet"
    quality_path = args.impulse_dir / "impulse_quality.parquet"
    reconstruction = load_json(reconstruction_path)
    baseline = load_json(baseline_path)
    impulse = load_json(impulse_path)
    validate_inputs(
        config, reconstruction, baseline, impulse, propagation_path, quality_path,
    )
    propagation_raw = pd.read_parquet(propagation_path)
    quality_raw = pd.read_parquet(quality_path)
    propagation = propagation_summary(propagation_raw)
    quality = quality_summary(quality_raw, args.bootstrap_samples)
    rates = rate_summary(propagation, quality)

    outputs = [
        args.output / PROPAGATION_SUMMARY,
        args.output / QUALITY_SUMMARY,
        args.output / RATE_SUMMARY,
    ]
    atomic_csv(outputs[0], propagation)
    atomic_csv(outputs[1], quality)
    atomic_csv(outputs[2], rates)
    plot_paths = heatmaps(propagation, args.output, config["run_id"])
    plot_paths.extend(quality_plot(quality, args.output))
    report_path = args.output / REPORT
    atomic_text(
        report_path,
        report_text(rates, quality, propagation, reconstruction, baseline),
    )
    outputs.extend(plot_paths)
    outputs.append(report_path)
    manifest = {
        "completed": True,
        "run_id": config["run_id"],
        "schema": "pr13_causal_replay_analysis_v1",
        "bootstrap_unit": "validation_sequence",
        "bootstrap_samples": args.bootstrap_samples,
        "input_hashes": {
            "config": sha256(args.config),
            "reconstruction_facts": sha256(reconstruction_path),
            "baseline_facts": sha256(baseline_path),
            "impulse_facts": sha256(impulse_path),
            "propagation": sha256(propagation_path),
            "quality": sha256(quality_path),
        },
        "source_hashes": {
            "causal_replay": sha256(
                EXPERIMENT / "src" / "oracle_study" / "causal_replay.py",
            ),
            "reconstruction_runner": sha256(
                EXPERIMENT / "scripts" / "run_causal_downstream_replay.py",
            ),
            "impulse_runner": sha256(
                EXPERIMENT / "scripts" / "run_causal_impulse_replay.py",
            ),
            "analysis_runner": sha256(Path(__file__).resolve()),
        },
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in outputs
        },
        "scientific_boundary": impulse["scientific_boundary"],
        "test_scientific_rows_admitted_or_used": False,
    }
    atomic_json(args.output / MANIFEST, manifest)


if __name__ == "__main__":
    main()
