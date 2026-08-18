#!/usr/bin/env python3
"""Combine the core and limited-extension layer trends into saved tables/plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INVOCATION = ["request_id", "position", "layer", "expert_id", "router_rank"]


def best_progressive(path: Path) -> pd.DataFrame:
    data = pd.read_parquet(path)
    keys = INVOCATION + ["projection", "budget_bpw"]
    return data.loc[data.groupby(keys).recovery.idxmax()].copy()


def best_hybrid(path: Path) -> pd.DataFrame:
    data = pd.read_parquet(path)
    data = data[data.down_oracle_variant == "sequential"]
    keys = INVOCATION + ["budget_bpw"]
    return data.loc[data.groupby(keys).recovery.idxmax()].copy()


def save(fig: plt.Figure, output: Path, name: str) -> None:
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--core-hybrid", type=Path, required=True)
    parser.add_argument("--extension-hybrid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    projection = pd.concat([best_progressive(args.core), best_progressive(args.extension)], ignore_index=True)
    hybrid = pd.concat([best_hybrid(args.core_hybrid), best_hybrid(args.extension_hybrid)], ignore_index=True)
    projection_summary = projection.groupby(["layer", "projection", "budget_bpw"]).agg(
        invocations=("recovery", "size"), median_recovery=("recovery", "median"),
        p10_recovery=("recovery", lambda x: x.quantile(.1)), p90_recovery=("recovery", lambda x: x.quantile(.9)),
    ).reset_index()
    hybrid_summary = hybrid.groupby(["layer", "budget_bpw"]).agg(
        invocations=("recovery", "size"), median_recovery=("recovery", "median"),
        p10_recovery=("recovery", lambda x: x.quantile(.1)), p90_recovery=("recovery", lambda x: x.quantile(.9)),
        gate_bpw=("gate_bpw", "median"), up_bpw=("up_bpw", "median"), down_bpw=("down_bpw", "median"),
    ).reset_index()
    projection_summary.to_csv(args.output / "six_layer_projection_summary.csv", index=False)
    hybrid_summary.to_csv(args.output / "six_layer_complete_expert_summary.csv", index=False)
    plots = args.output / "plots"; plots.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for axis, projection_name in zip(axes, ("gate", "up", "down")):
        for rate, style in ((2.0, "o-"), (4.0, "s--")):
            group = projection_summary[(projection_summary.projection == projection_name) & (projection_summary.budget_bpw == rate)].sort_values("layer")
            axis.plot(group.layer, group.median_recovery, style, label=f"{rate:g} physical bpw")
        axis.axhline(.9, color="black", lw=.8, ls=":"); axis.set(title=projection_name, xlabel="routed layer", ylim=(0, 1.02)); axis.grid(alpha=.25)
    axes[0].set_ylabel("median progressive correction recovery"); axes[-1].legend()
    save(fig, plots, "six_layer_projection_trend")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for rate, style in ((1.0, "^-"), (2.0, "o-"), (4.0, "s--")):
        group = hybrid_summary[hybrid_summary.budget_bpw == rate].sort_values("layer")
        ax.plot(group.layer, group.median_recovery, style, label=f"{rate:g} average physical bpw")
    ax.axhline(.9, color="black", lw=.8, ls=":"); ax.set(xlabel="routed layer", ylabel="median complete-expert future-proxy recovery", ylim=(0, 1.02)); ax.grid(alpha=.25); ax.legend()
    save(fig, plots, "six_layer_complete_expert_trend")
    print({"projection_rows": len(projection_summary), "hybrid_rows": len(hybrid_summary)})


if __name__ == "__main__":
    main()
