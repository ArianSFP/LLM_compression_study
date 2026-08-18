#!/usr/bin/env python3
"""Tables and plots for bounded MXFP4 ranking-search diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_figure(fig: plt.Figure, root: Path, stem: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    fig.savefig(root / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(root / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    data = pd.read_parquet(args.result / "search_supplement.parquet")
    support = data[data.comparison.isin(["diagonal", "exact_greedy", "greedy_plus_local_swaps"])].copy()
    support.to_csv(args.result / "stage_specific_support_summary.csv", index=False)
    beam = data[data.comparison == "bounded_beam"].copy()
    beam.to_csv(args.result / "bounded_beam_summary.csv", index=False)

    ninety = support[(support.target == .9) & support.comparison.isin(["diagonal", "exact_greedy"])]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey="col")
    for row, plane in enumerate((1, 2)):
        for col, projection in enumerate(("gate", "up", "down")):
            ax = axes[row, col]
            frame = ninety[(ninety.refinement_plane == plane) & (ninety.projection == projection)]
            for method, marker in (("diagonal", "o"), ("exact_greedy", "s")):
                values = frame[frame.comparison == method].sort_values("layer")
                ax.plot(values.layer, values.k, marker=marker, label=method.replace("_", " "))
            ax.set_title(f"{projection}, plane {plane}")
            ax.set_xlabel("layer")
            if col == 0:
                ax.set_ylabel("coordinates for 90%")
            ax.grid(alpha=.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Stage-specific diagonal ranking regret")
    save_figure(fig, args.result / "plots", "12_stage_specific_support")

    swapped = support[(support.target == .9) & (support.comparison == "greedy_plus_local_swaps")].copy()
    swapped["gain_points"] = 100 * (swapped.recovery - .9)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(swapped))
    colors = swapped.projection.map({"gate": "#4477aa", "up": "#66ccee", "down": "#cc6677"})
    ax.bar(positions, swapped.gain_points, color=colors)
    ax.set_xticks(positions, [f"L{r.layer} {r.projection[0]} p{r.refinement_plane}" for _, r in swapped.iterrows()], rotation=70)
    ax.set_ylabel("recovery points gained at fixed greedy k")
    ax.set_title("Greedy plus up to eight local swaps")
    ax.grid(axis="y", alpha=.25)
    save_figure(fig, args.result / "plots", "13_local_swap_gain")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, plane in zip(axes, (1, 2)):
        frame = beam[beam.refinement_plane == plane]
        for width in sorted(frame.beam_width.dropna().unique()):
            curve = frame[frame.beam_width == width].groupby("k").recovery.median()
            ax.plot(curve.index, curve.values, marker="o", label=f"width {int(width)}")
        ax.set_title(f"refinement plane {plane}")
        ax.set_xlabel("beam depth k")
        ax.grid(alpha=.25)
    axes[0].set_ylabel("median local recovery")
    axes[1].legend(frameon=False)
    fig.suptitle("Bounded beam over top-32 energy candidates")
    save_figure(fig, args.result / "plots", "14_bounded_beam")


if __name__ == "__main__":
    main()
