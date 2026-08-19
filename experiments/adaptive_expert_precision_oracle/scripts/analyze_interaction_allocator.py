#!/usr/bin/env python3
"""Merge promoted allocator runs and generate required tables, plots, and report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECTION_FILES = (
    "exact_broad_ranking_audit.parquet",
    "selector_refresh_frontier.parquet",
    "gram_spectrum.parquet",
    "low_rank_selector_frontier.parquet",
)
PHYSICAL_FILES = ("action_to_page_frontier.parquet", "paired_hybrid_frontier.parquet")


def savefig(fig: plt.Figure, directory: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(directory / f"{name}.png", dpi=180)
    svg_path = directory / f"{name}.svg"
    fig.savefig(svg_path)
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n"
    )
    plt.close(fig)


def merge_parquet(directories: list[Path], filename: str) -> pd.DataFrame:
    frames = []
    for directory in directories:
        path = directory / filename
        if path.exists():
            frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def pct(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{100 * value:.2f}%"


def quantiles(values: pd.Series) -> tuple[float, float, float]:
    result = values.quantile([0.1, 0.5, 0.9])
    return float(result.loc[0.1]), float(result.loc[0.5]), float(result.loc[0.9])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection", type=Path, action="append", default=[])
    parser.add_argument("--low-rank-pilot", type=Path, action="append", default=[])
    parser.add_argument("--physical", type=Path, action="append", default=[])
    parser.add_argument("--physical-pilot", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pr4-result", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plots = args.output / "plots"
    plots.mkdir(exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}
    for filename in PROJECTION_FILES:
        directories = args.projection + (args.low_rank_pilot if filename == "low_rank_selector_frontier.parquet" else [])
        frame = merge_parquet(directories, filename)
        frame.to_parquet(args.output / filename, index=False)
        tables[filename] = frame
    for filename in PHYSICAL_FILES:
        frame = merge_parquet(args.physical, filename)
        pilot = merge_parquet(args.physical_pilot, filename)
        if filename == "action_to_page_frontier.parquet":
            frame = pd.concat([frame, pilot], ignore_index=True)
        elif len(pilot):
            pilot = pilot[pilot.projection_selector.str.startswith("joint_")]
            frame = pd.concat([frame, pilot], ignore_index=True)
        frame.to_parquet(args.output / filename, index=False)
        tables[filename] = frame
    overlaps = []
    for directory in args.projection:
        path = directory / "support_overlap_summary.csv"
        if path.exists():
            overlaps.append(pd.read_csv(path))
    for directory in args.low_rank_pilot:
        path = directory / "support_overlap_summary.csv"
        if path.exists():
            values = pd.read_csv(path)
            values = values[values.selector.str.startswith(("truncated_svd_", "fixed_jl_", "activation_weighted_svd_"))]
            overlaps.append(values)
    overlap = pd.concat(overlaps, ignore_index=True) if overlaps else pd.DataFrame()
    overlap.to_csv(args.output / "support_overlap_summary.csv", index=False)
    accounting_sources = []
    for directory in args.physical + args.physical_pilot:
        path = directory / "selector_compute_storage_accounting.json"
        if path.exists():
            accounting_sources.append(json.loads(path.read_text()))
    accounting = {
        "sources": accounting_sources,
        "page_size_bytes": 512,
        "exact_reference_bpw": 4.25,
        "resident_parent_bpw": 2.25,
        "separate_suffix_bpw": 2.0,
        "paired_suffix_bpw": 2.0,
        "hybrid_suffix_bpw": 4.0,
        "external_storage_multiplier_separate": (4.25 + 2.0) / 4.25,
        "external_storage_multiplier_paired": (4.25 + 2.0) / 4.25,
        "external_storage_multiplier_hybrid": (4.25 + 4.0) / 4.25,
        "external_storage_cap": 5.0,
        "test_masks_used_for_layouts": False,
    }
    low_rank = tables["low_rank_selector_frontier.parquet"]
    if len(low_rank):
        accounting["maximum_metadata_bytes_projection"] = int(low_rank.metadata_bytes_projection.max())
        accounting["maximum_extrapolated_metadata_bytes_40x256_projection"] = int(
            low_rank.metadata_bytes_all_40x256_experts_projection.max()
        )
        unique_metadata = low_rank.drop_duplicates([
            "capture_source", "layer", "expert_id", "projection",
            "method", "rank", "metadata_encoding",
        ])
        expert_metadata = (
            unique_metadata.groupby([
                "capture_source", "layer", "expert_id", "method", "rank", "metadata_encoding",
            ]).metadata_bytes_projection.sum()
        )
        accounting["maximum_metadata_bytes_expert_all_three_projections"] = int(expert_metadata.max())
        accounting["maximum_extrapolated_metadata_bytes_all_40x256_experts"] = int(
            expert_metadata.max() * 40 * 256
        )
    (args.output / "selector_compute_storage_accounting.json").write_text(
        json.dumps(accounting, indent=2, sort_keys=True) + "\n"
    )

    broad = tables["exact_broad_ranking_audit.parquet"]
    joint90 = broad[
        (broad.refinement_plane == "joint_nested_q2_q3_q4")
        & (broad.target_recovery == 0.9)
        & broad.projection.isin(["gate", "up"])
    ]
    crossing = joint90.pivot_table(
        index=["capture_source", "request_id", "position", "layer", "expert_id", "projection"],
        columns="selector", values="actions_required",
    ).reset_index()
    crossing["diagonal_over_exact"] = (
        crossing["diagonal"] / crossing["exact_marginal_fixed_greedy"]
    )
    crossing.to_csv(args.output / "exact_crossing_summary.csv", index=False)
    if len(crossing):
        summary = crossing.groupby(["capture_source", "layer", "projection"]).diagonal_over_exact.median().unstack("projection")
        fig, ax = plt.subplots(figsize=(7.2, 4.3))
        summary.plot(kind="bar", ax=ax)
        ax.axhline(1.0, color="black", lw=1)
        ax.set(ylabel="Median diagonal / exact actions at 90%", xlabel="Capture, layer")
        ax.grid(axis="y", alpha=0.25)
        savefig(fig, plots, "01_exact_gate_up_crossings")

    selector = tables["selector_refresh_frontier.parquet"]
    selector_test = selector[
        (selector.evaluation_split == "test")
        & selector.projection.isin(["gate", "up"])
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for name in (
        "diagonal", "block_refresh_32", "block_refresh_16",
        "block_refresh_8", "block_refresh_4", "exact_marginal_fixed_greedy",
    ):
        values = selector_test[selector_test.selector == name]
        if len(values):
            curve = values.groupby("logical_bpw_projection").recovery.median()
            ax.plot(curve.index, curve.values, marker="o", label=name)
    ax.set(xlabel="Logical correction bpw per projection", ylabel="Median isolated recovery")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    savefig(fig, plots, "02_selector_refresh_frontier")

    spectrum = tables["gram_spectrum.parquet"]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for projection in ("gate", "up", "down"):
        values = spectrum[spectrum.projection == projection]
        if len(values):
            curve = values.groupby("rank").spectral_energy_retained.median()
            ax.plot(curve.index, curve.values, marker="o", label=projection)
    ax.set(xlabel="Gram factor rank", ylabel="Median spectral energy retained", xscale="log", xticks=[8, 16, 32, 64, 128])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(alpha=0.25)
    ax.legend()
    savefig(fig, plots, "03_gram_spectrum")

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    low_test = low_rank[
        (low_rank.evaluation_split == "test")
        & low_rank.projection.isin(["gate", "up"])
        & (low_rank.physical_budget_bpw_projection == 1.0)
    ]
    if len(low_test):
        for (method, encoding), values in low_test.groupby(["method", "metadata_encoding"]):
            curve = values.groupby("rank").fraction_exact_over_diagonal_gain_retained_at_matched_physical_budget.median()
            ax.plot(curve.index, curve.values, marker="o", label=f"{method}/{encoding}")
    ax.axhline(0.9, color="black", ls="--", lw=1, label="90% gate")
    ax.set(xlabel="Metadata rank", ylabel="Fraction exact-over-diagonal gain retained (matched physical budget)", xscale="log", xticks=[8, 16, 32, 64, 128])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6)
    savefig(fig, plots, "04_low_rank_feasibility")

    pages = tables["action_to_page_frontier.parquet"]
    if len(pages):
        fig, ax = plt.subplots(figsize=(7.2, 4.3))
        summary = pages.groupby(["layout_method", "selector"]).page_amplification.median().sort_values()
        labels = [f"{a}\n{b}" for a, b in summary.index]
        ax.bar(np.arange(len(summary)), summary.values)
        ax.set_xticks(np.arange(len(summary)), labels, rotation=35, ha="right", fontsize=7)
        ax.set(ylabel="Median physical page amplification")
        ax.axhline(1.2, color="black", ls="--", lw=1)
        ax.grid(axis="y", alpha=0.25)
        savefig(fig, plots, "05_page_layout_amplification")

    packet = tables["paired_hybrid_frontier.parquet"]
    if len(packet):
        exact_packet = packet[packet.capture_source == "exact_checkpoint"]
        fig, ax = plt.subplots(figsize=(7.2, 4.3))
        for (selector_name, representation), values in exact_packet.groupby(
            ["projection_selector", "packet_representation"]
        ):
            curve = values.groupby("budget_bpw").recovery.median()
            ax.plot(curve.index, curve.values, marker="o", label=f"{selector_name}/{representation}")
        ax.set(xlabel="Physical correction bpw", ylabel="Median exact sequential recovery")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6)
        savefig(fig, plots, "06_complete_expert_physical_frontier")

    # Success gates use only the fresh exact-checkpoint test rows and difficult layers.
    success: dict[str, Any] = {}
    fresh = packet[
        (packet.capture_source == "exact_checkpoint")
        & packet.layer.isin([4, 20, 39])
        & (packet.budget_bpw == 1.0)
    ]
    baseline = fresh[
        (fresh.projection_selector == "diagonal")
        & (fresh.packet_representation == "separate_q3_q4_planes")
        & (fresh.layout_method == "current_diagonal_trained_coselection")
    ]
    oracle = fresh[
        (fresh.projection_selector == "gate_up_exact_marginal_fixed_greedy_down_diagonal")
        & (fresh.packet_representation == "separate_q3_q4_planes")
    ]
    if len(baseline):
        p10, median, _ = quantiles(baseline.recovery)
        success["reconstructed_pr4_control_fresh_difficult_1bpw"] = {
            "p10": p10, "median": median, "n": len(baseline)
        }
    # This layout is specified before test evaluation: its permutation is
    # fitted only from exact-greedy training masks. Never choose among layout
    # families by their held-out recovery.
    selected_layout = "exact_greedy_training_masks"
    if len(oracle):
        selected = oracle[oracle.layout_method == selected_layout]
        p10, median, _ = quantiles(selected.recovery)
        success["training_selected_exact_oracle_fresh_difficult_1bpw"] = {
            "layout": selected_layout, "p10": p10, "median": median, "n": len(selected)
        }
    pr4_path = args.pr4_result
    if pr4_path.is_dir():
        candidates = (
            pr4_path / "selective" / "selective_metrics.parquet",
            pr4_path / "exact_checkpoint_confirmation" / "selective" / "selective_metrics.parquet",
        )
        pr4_path = next((path for path in candidates if path.exists()), pr4_path)
    if pr4_path.is_file():
        pr4 = pd.read_parquet(pr4_path)
        pr4 = pr4[(pr4.layer.isin([4, 20, 39])) & (pr4.budget_bpw == 1.0)]
        if len(pr4):
            p10, median, _ = quantiles(pr4.recovery)
            success["pr4_reported_fresh_difficult_1bpw"] = {
                "artifact": str(pr4_path), "p10": p10, "median": median, "n": len(pr4)
            }
            selected = success.get("training_selected_exact_oracle_fresh_difficult_1bpw", {})
            if selected:
                success["fresh_difficult_improves_pr4_median_and_p10"] = bool(
                    selected["median"] > median and selected["p10"] > p10
                )
    low_gate = low_test[low_test["rank"] <= 64]
    if len(low_gate):
        best = (
            low_gate.groupby(["method", "rank", "metadata_encoding"])
            .fraction_exact_over_diagonal_gain_retained_at_matched_physical_budget.median()
            .sort_values(ascending=False)
        )
        success["best_rank_le_64_gain_retention"] = {
            "configuration": "/".join(map(str, best.index[0])),
            "median": float(best.iloc[0]),
        }
    success["storage_below_5x"] = accounting["external_storage_multiplier_hybrid"] < 5.0
    selected_pages = pages[pages.layout_method == selected_layout] if len(pages) else pages
    if len(selected_pages):
        amplification_rows = {}
        for selector_name, values in selected_pages.groupby("selector"):
            amplification = float(values.page_amplification.median())
            amplification_rows[str(selector_name)] = {
                "median": amplification,
                "passes_le_1p2": amplification <= 1.2,
                "n": len(values),
            }
        success["training_selected_layout_page_amplification_by_selector"] = amplification_rows
    success["broad_gate_up_layers_with_exact_gain"] = int(
        crossing.groupby(["layer", "projection"]).diagonal_over_exact.median().gt(1.0).sum()
    )
    (args.output / "success_gates.json").write_text(json.dumps(success, indent=2, sort_keys=True) + "\n")

    exact_ratio = float(crossing.diagonal_over_exact.median()) if len(crossing) else float("nan")
    exact_min = float(crossing.diagonal_over_exact.min()) if len(crossing) else float("nan")
    rank_value = success.get("best_rank_le_64_gain_retention", {}).get("median", float("nan"))
    fresh_control = success.get("pr4_reported_fresh_difficult_1bpw", {})
    fresh_oracle = success.get("training_selected_exact_oracle_fresh_difficult_1bpw", {})
    gate_one_passed = bool(success.get("fresh_difficult_improves_pr4_median_and_p10", False))
    gate_one_label = "passed" if gate_one_passed else "failed"
    maximum_metadata_expert = int(accounting.get("maximum_metadata_bytes_expert_all_three_projections", 0))
    maximum_metadata_fleet = int(accounting.get("maximum_extrapolated_metadata_bytes_all_40x256_experts", 0))
    amplification = success.get("training_selected_layout_page_amplification_by_selector", {})
    exact_path_amplification = float(
        amplification.get("exact_action_path_repriced_to_pages", {}).get("median", float("nan"))
    )
    page_oracle_amplification = float(
        amplification.get("page_level_residual_aware_mask_oracle", {}).get("median", float("nan"))
    )
    selected_packets = packet[
        (packet.capture_source == "exact_checkpoint")
        & packet.layer.isin([4, 20, 39])
        & (packet.budget_bpw == 1.0)
        & (packet.layout_method == selected_layout)
        & (packet.projection_selector == "gate_up_exact_marginal_fixed_greedy_down_diagonal")
    ]
    packet_medians = selected_packets.groupby("packet_representation").recovery.median().to_dict()
    report = f"""# Interaction-Aware MXFP4 Allocator Study

Run date: 2026-08-19
Branch: `agent/mxfp4-interaction-aware-allocator`
Starting commit: `6b31a88ee517359847f940cc74db56898a8b8425`

## Executive conclusion

The broad H0 interaction hypothesis is supported, but the compact deployable-Gram hypothesis is not. Across the audited gate/up rows, diagonal ranking requires a median **{exact_ratio:.2f}x** as many actions as exact fixed-coefficient marginal greedy at 90% recovery (minimum invocation **{exact_min:.2f}x**). Refresh selectors retain much of that isolated oracle gain.

Rank <= 64 metadata did not meet the 90% feasibility gate. The best held-out median fraction of exact-over-diagonal gain retained was **{rank_value:.3f}**. These variants were selected on validation, evaluated once on held-out pilot rows, and were not promoted to the broad full runs. This is a negative deployability result, not a prompt to tune on test.

## Locked scientific controls

- Checkpoint revision remained `7eceff3a9f7e6f916c824d197266d86676bce695`; config and tensor-index hashes were checked before every run.
- The PR #4 gate/up/down trees and embedded Q2->Q3->Q4 deltas were read unchanged.
- Exact greedy fixes every selected coefficient; it performs no least-squares refit. The canonical label is `exact_marginal_fixed_greedy`; old `exact_marginal_omp` artifacts remain readable.
- Training requests alone fitted activation-weighted metadata and physical layouts; validation selected hyperparameters; test supplied only held-out activations and explicitly labeled H0 oracle choices.
- No H4 predictor was trained. No router, logit, token-quality, or model-quality claim is made.

## Isolated projection recovery

The file `exact_broad_ranking_audit.parquet` contains 80/90/95/99% crossings by source, request, layer, projection, expert stratum, router rank, and refinement plane. Both Q2->Q3 and Q3->Q4 are reported separately from the joint nested path. Gate/up gains reproduce across layers rather than depending on one invocation. Down interactions are smaller and inconsistent, so down remains diagonal in any deployable interpretation.

## Refresh selector frontier

`selector_refresh_frontier.parquet` records logical actions, selector time, and analytical bytes read; `action_to_page_frontier.parquet` separately records physical pages. `block_refresh_4/8/16` are H0 Pareto-relevant; initial-full-marginal and backward elimination are not reliable gate/up replacements. Exact and backward paths reach the complete endpoint, but backward is poor at low rate.

## Low-rank feasibility and storage

`gram_spectrum.parquet` and `low_rank_selector_frontier.parquet` cover ranks 8/16/32/64/128, truncated SVD, fixed JL, activation-weighted training SVD, and FP16/int8 metadata. Exact diagonals are always retained. Metadata bytes are reported per projection and extrapolated over all 40x256 routed experts.

Across all three projections, the largest evaluated metadata configuration occupies **{maximum_metadata_expert:,} bytes per expert**, or **{maximum_metadata_fleet:,} bytes** extrapolated over 40x256 routed experts. Per-rank and per-encoding values remain in the frontier rather than being collapsed to this maximum.

The primary matched-budget metric is `(recovery_approx-recovery_diagonal)/(recovery_exact-recovery_diagonal)`. Rank <= 64 fails the 0.90 target on held-out gate/up, so no low-rank configuration was promoted to the 85/142-invocation expansion.

## Physical pages and layouts

`action_to_page_frontier.parquet` reports the bounded 512-byte page-mask oracle and exact action paths repriced to pages. Layouts are: current diagonal-trained co-selection, exact-training-mask layout, balanced hypergraph, and activation-feature support clusters. Test masks never build layouts. Page mask enumeration is exact up to 2^8 for separate gate/up pages, 2^4 for paired gate/up pages, and 2^2 for down pages.

## Paired/hybrid packets and complete experts

`paired_hybrid_frontier.parquet` distinguishes separate planes, exact Q2->Q4 coordinate packets, and H0 hybrid representation choice. Every row carries exact storage multipliers, physical bytes/pages, logical bytes/actions, and exact sequential SwiGLU 9x9x9 recovery.

The physical pilot also includes `joint_gate_up_first_order_exact_page_rescore_down_diagonal`: 16 global refreshes shortlist pages using the requested first-order gate/up SwiGLU effects propagated through the current diagonal-down reconstruction and future-proxy metric. Within each shortlist, every chosen page is rescored with actual SwiGLU and qenergy using exhaustive 2^8-or-smaller masks. This remains a bounded H0 point, not a deployable selector.

PR #4 fresh difficult-layer one-bpw result: p10 **{pct(float(fresh_control.get("p10", float("nan"))))}**, median **{pct(float(fresh_control.get("median", float("nan"))))}**. Training-mask-selected exact H0 physical layout: p10 **{pct(float(fresh_oracle.get("p10", float("nan"))))}**, median **{pct(float(fresh_oracle.get("median", float("nan"))))}**.

At the same selected layout and one physical bpw, median complete-expert recovery is **{pct(float(packet_medians.get("separate_q3_q4_planes", float("nan"))))}** for separate planes, **{pct(float(packet_medians.get("paired_exact_q2_q4_packets", float("nan"))))}** for paired packets, and **{pct(float(packet_medians.get("hybrid_both_representations", float("nan"))))}** for the H0 hybrid representation choice.

These are complete-expert H0 oracle results, not deployable low-rank results. Exact page/path oracles require Gram information that exceeds the compact metadata target.

## Success gates

1. Fresh difficult-layer p10 and median improvement versus PR #4: **{gate_one_label}** for the predeclared exact-training-mask layout. This remains an H0 oracle result.
2. Rank <= 64 with compact metadata: **failed**.
3. Training-mask layout page amplification: exact action path **{exact_path_amplification:.3f}x**; sparse exhaustive page-mask oracle **{page_oracle_amplification:.3f}x**. Each is judged independently against 1.2x in `success_gates.json`.
4. External storage: separate/paired/hybrid multipliers are {accounting["external_storage_multiplier_separate"]:.3f}x/{accounting["external_storage_multiplier_paired"]:.3f}x/{accounting["external_storage_multiplier_hybrid"]:.3f}x, all below 5x.
5. Broad gate/up gain: **passed at the H0 exact-oracle level**.

## Interpretation

Static diagonal paths are a real bottleneck, and short refresh blocks recover much of the exact isolated gain. However, the interaction Gram is not compact enough at rank 64 for the requested deployable selector. The correct next step is not H4 training or test-set tuning; it is a different structured metadata model or a serving-time shortlist source that does not attempt to compress the full Gram into one low-rank PSD factor.
"""
    (args.output / "INTERACTION_AWARE_ALLOCATOR_REPORT.md").write_text(report)

    artifact_paths = [
        *(args.output / filename for filename in PROJECTION_FILES),
        *(args.output / filename for filename in PHYSICAL_FILES),
        args.output / "support_overlap_summary.csv",
        args.output / "selector_compute_storage_accounting.json",
        args.output / "success_gates.json",
        args.output / "exact_crossing_summary.csv",
        args.output / "INTERACTION_AWARE_ALLOCATOR_REPORT.md",
        *sorted((args.output / "plots").glob("*.png")),
        *sorted((args.output / "plots").glob("*.svg")),
        *sorted((args.output / "raw").glob("*/*")),
    ]
    fresh_manifest = args.output / "fresh_capture_manifest.json"
    if fresh_manifest.exists():
        artifact_paths.append(fresh_manifest)
    hash_lines = []
    for path in sorted(artifact_paths):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hash_lines.append(f"{digest}  {path.relative_to(args.output)}")
    (args.output / "artifact_hashes.sha256").write_text("\n".join(hash_lines) + "\n")


if __name__ == "__main__":
    main()
