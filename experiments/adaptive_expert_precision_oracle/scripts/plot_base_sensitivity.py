#!/usr/bin/env python3
"""Summarize simple-W1, activation-aware-W1, and Q2 projection sensitivity."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INVOCATION = ["request_id", "position", "layer", "expert_id", "router_rank", "projection", "budget_bpw"]


def best(path: Path, label: str) -> pd.DataFrame:
    data = pd.read_parquet(path)
    data = data.loc[data.groupby(INVOCATION).recovery.idxmax()].copy()
    data["resident_base"] = label
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simple-w1", type=Path, required=True)
    parser.add_argument("--aware-w1", type=Path, required=True)
    parser.add_argument("--q2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = pd.concat([
        best(args.simple_w1, "simple W1 (1.250488 resident bpw)"),
        best(args.aware_w1, "activation-aware-scale W1 (1.250488 resident bpw)"),
        best(args.q2, "Q2 (2.250488 resident bpw)"),
    ], ignore_index=True)
    summary = data.groupby(["resident_base", "layer", "projection", "budget_bpw"]).agg(
        invocations=("recovery", "size"), recovery_median=("recovery", "median"),
        relative_output_error_median=("relative_output_error", "median"), base_damage_median=("base_damage", "median"),
    ).reset_index()
    summary.to_csv(args.output / "resident_base_sensitivity.csv", index=False)
    plots = args.output / "plots"; plots.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for axis, projection in zip(axes, ("gate", "up", "down")):
        for label, group in data[data.projection == projection].groupby("resident_base"):
            curve = group.groupby("budget_bpw").relative_output_error.median()
            axis.plot(curve.index, curve.values, marker="o", ms=3, label=label)
        axis.set(title=projection, xlabel="4 KiB physical correction bpw", xlim=(0, 4)); axis.grid(alpha=.25)
    axes[0].set_ylabel("median relative Q4 output error"); axes[-1].legend(fontsize=7)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(plots / f"resident_base_sensitivity.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print({"rows": len(summary)})


if __name__ == "__main__":
    main()
