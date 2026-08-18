#!/usr/bin/env python3
"""Paired attribution for first-bit, direct-pair, parity, and layout effects."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KEYS = ["request_id", "position", "layer", "expert_id", "budget_bpw"]


def cluster_ci(frame: pd.DataFrame, resamples: int, seed: int) -> tuple[float, float]:
    requests = frame.request_id.unique()
    rng = np.random.default_rng(seed)
    grouped = {request: frame.loc[frame.request_id == request, "difference"].to_numpy() for request in requests}
    values = []
    for _ in range(resamples):
        sample = rng.choice(requests, len(requests), replace=True)
        values.append(np.median(np.concatenate([grouped[request] for request in sample])))
    return tuple(np.quantile(values, (0.025, 0.975)))


def grouped(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([
        frame.assign(group=np.where(frame.layer == 0, "layer 0", "layers 4/20/39")),
        frame.assign(group="all four layers"),
    ], ignore_index=True)


def load_policy(path: Path, source_policy: str, label: str) -> pd.DataFrame:
    frame = pd.read_parquet(path / "metrics.parquet")
    value = frame[frame.policy == source_policy][KEYS + ["recovery"]].copy()
    return value.rename(columns={"recovery": label})


def count_actions(path: Path) -> Counter[tuple[str, str, float, str]]:
    counts: Counter[tuple[str, str, float, str]] = Counter()
    action_path = path / "selected_description_actions.parquet"
    parts = [action_path] if action_path.exists() else sorted((path / "selected_description_actions").glob("*.parquet"))
    frame = pd.concat([pd.read_parquet(part, columns=[
            "policy", "projection", "budget_bpw", "before_masks_json", "added_descriptions_json",
        ]) for part in parts], ignore_index=True)
    for row in frame.itertuples(index=False):
        before = json.loads(row.before_masks_json)
        added = json.loads(row.added_descriptions_json)
        for state, values in zip(before, added):
            if state == 0 and len(values) == 1:
                transition = "first_" + values[0]
            elif state == 0:
                transition = "direct_" + "".join(values)
            else:
                transition = "upgrade_via_" + "".join(values)
            counts[(row.policy, row.projection, float(row.budget_bpw), transition)] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--parity-control", type=Path, required=True)
    parser.add_argument("--action-ablation", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()

    series = {
        "ordered": load_policy(args.main, "ordered_a", "ordered"),
        "bidirectional": load_policy(args.main, "bidirectional_ab", "bidirectional"),
        "parity_native_layout": load_policy(args.main, "parity_abq", "parity_native_layout"),
        "parity_shared_layout": load_policy(args.parity_control, "parity_abq_shared_ab_layout", "parity_shared_layout"),
        "ordered_direct": load_policy(args.action_ablation, "ordered_a_with_direct_ab", "ordered_direct"),
        "bidirectional_no_direct": load_policy(args.action_ablation, "bidirectional_ab_no_direct", "bidirectional_no_direct"),
    }
    comparisons = {
        "B-first only (no direct action)": ("bidirectional_no_direct", "ordered"),
        "direct AB with A-first": ("ordered_direct", "ordered"),
        "B-first after enabling direct AB": ("bidirectional", "ordered_direct"),
        "direct AB with bidirectional first bit": ("bidirectional", "bidirectional_no_direct"),
        "parity q plane on shared A/B layout": ("parity_shared_layout", "bidirectional"),
        "parity-layout change only": ("parity_native_layout", "parity_shared_layout"),
    }
    rows = []
    for label, (left, right) in comparisons.items():
        paired = series[left].merge(series[right], on=KEYS)
        paired["difference"] = paired[left] - paired[right]
        for (group, budget), frame in grouped(paired).groupby(["group", "budget_bpw"]):
            low, high = cluster_ci(frame, args.bootstrap, args.seed + int(100 * budget))
            rows.append({
                "comparison": label, "left": left, "right": right, "group": group,
                "budget_bpw": budget, "invocations": len(frame),
                "p10_difference": frame.difference.quantile(.1), "median_difference": frame.difference.median(),
                "p90_difference": frame.difference.quantile(.9), "mean_difference": frame.difference.mean(),
                "fraction_improved": (frame.difference > 1e-8).mean(),
                "fraction_worsened": (frame.difference < -1e-8).mean(),
                "median_ci_low": low, "median_ci_high": high,
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(args.main / "action_ablation_summary.csv", index=False)

    counts = Counter()
    for path in (args.main, args.parity_control, args.action_ablation):
        counts.update(count_actions(path))
    count_frame = pd.DataFrame([
        {"policy": key[0], "projection": key[1], "budget_bpw": key[2], "transition": key[3], "actions": value}
        for key, value in counts.items()
    ])
    count_frame.to_csv(args.main / "action_transition_counts.csv", index=False)

    difficult = summary[summary.group == "layers 4/20/39"]
    selected = [
        "B-first only (no direct action)", "direct AB with A-first",
        "parity q plane on shared A/B layout", "parity-layout change only",
    ]
    plt.figure(figsize=(8, 4.8))
    for label in selected:
        frame = difficult[difficult.comparison == label].sort_values("budget_bpw")
        plt.plot(frame.budget_bpw, 100 * frame.median_difference, marker="o", label=label)
    plt.axhline(0, color="black", linewidth=.8)
    plt.xlabel("Physical correction bpw"); plt.ylabel("Median paired recovery change (points)")
    plt.grid(alpha=.25); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(args.main / "plots/07_action_ablation.png", dpi=180)
    plt.savefig(args.main / "plots/07_action_ablation.svg")
    plt.close()


if __name__ == "__main__":
    main()
