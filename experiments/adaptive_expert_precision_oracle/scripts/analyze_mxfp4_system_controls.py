#!/usr/bin/env python3
"""Recompute retention and equal-support gate/up bundling controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summary(values: pd.Series) -> dict[str, float]:
    array = values.dropna().to_numpy(float)
    if not len(array):
        return {"count": 0, "mean": float("nan"), "p10": float("nan"), "median": float("nan"), "p90": float("nan")}
    return {"count": len(array), "mean": float(np.mean(array)), "p10": float(np.quantile(array, .1)),
            "median": float(np.median(array)), "p90": float(np.quantile(array, .9))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    dense = pd.read_parquet(args.result / "dense/dense_metrics.parquet")
    selective = pd.read_parquet(args.result / "selective/selective_metrics.parquet")
    actions = pd.read_parquet(args.result / "selective/selected_bit_actions.parquet")
    keys = ["request_id", "position", "layer", "expert_id"]
    retention = []
    for _, row in selective.iterrows():
        matched = dense[(dense.split == "test") & (dense.request_id == row.request_id) & (dense.position == row.position) &
                        (dense.layer == row.layer) & (dense.expert_id == row.expert_id) & (dense.physical_bpw <= row.budget_bpw + 1e-12)]
        best = matched.loc[matched.damage.idxmin()]
        denominator = float(best.base_damage - best.damage)
        eta = np.nan if denominator <= 1e-20 else (float(row.base_damage) - float(row.damage)) / denominator
        retention.append({**{key: row[key] for key in keys}, "budget_bpw": row.budget_bpw,
                          "selective_recovery": row.recovery, "dense_recovery": best.recovery,
                          "dense_tuple": best.tuple, "retention_eta": eta})
    retention = pd.DataFrame(retention)
    retention.to_parquet(args.result / "selective/selective_dense_retention.parquet", index=False)
    rows = []
    for group_key, frame in actions.groupby(keys + ["budget_bpw"]):
        by_projection = {row.projection: row for _, row in frame.iterrows()}
        if set(by_projection) != {"gate", "up", "down"}:
            continue
        bundled_pages = set()
        for projection in ("gate", "up"):
            coordinates = json.loads(by_projection[projection].coordinate_ids_json)
            stages = json.loads(by_projection[projection].refinement_bits_json)
            # One 512-B page contains four coordinates from both gate and up
            # for one refinement plane: 4 * (64 B gate + 64 B up).
            bundled_pages.update((int(stage) - 1) * 512 + int(coordinate) // 4 for coordinate, stage in zip(coordinates, stages))
        down_pages = set(json.loads(by_projection["down"].selected_page_ids_json))
        physical = (len(bundled_pages) + len(down_pages)) * 512
        matched = selective[(selective.request_id == group_key[0]) & (selective.position == group_key[1]) &
                            (selective.layer == group_key[2]) & (selective.expert_id == group_key[3]) &
                            (selective.budget_bpw == group_key[4])].iloc[0]
        rows.append({**{key: value for key, value in zip(keys, group_key[:4])}, "budget_bpw": group_key[4],
                     "recovery_equal_support": matched.recovery, "independent_physical_bytes": matched.physical_bytes,
                     "bundled_natural_physical_bytes": physical, "independent_physical_bpw": matched.physical_bpw,
                     "bundled_natural_physical_bpw": 8 * physical / (3 * 2048 * 512),
                     "bundled_over_independent": physical / max(float(matched.physical_bytes), 1.0),
                     "gate_up_bundled_pages": len(bundled_pages), "down_pages": len(down_pages),
                     "control": "equal selected support; natural-coordinate shared gate/up pages; no test-trained packing"})
    bundling = pd.DataFrame(rows)
    bundling.to_parquet(args.result / "gate_up_bundling_equal_support.parquet", index=False)
    summaries = []
    for budget, frame in bundling.groupby("budget_bpw"):
        summaries.append({"budget_bpw": budget, **{f"ratio_{key}": value for key, value in summary(frame.bundled_over_independent).items()},
                          "median_independent_bpw": float(frame.independent_physical_bpw.median()),
                          "median_bundled_bpw": float(frame.bundled_natural_physical_bpw.median())})
    pd.DataFrame(summaries).to_csv(args.result / "gate_up_bundling_summary.csv", index=False)
    retention_summaries = []
    for label, layers in (("all_four", [0,4,20,39]), ("difficult_4_20_39", [4,20,39])):
        for budget, frame in retention[retention.layer.isin(layers)].groupby("budget_bpw"):
            retention_summaries.append({"aggregate": label, "budget_bpw": budget, **summary(frame.retention_eta)})
        # The exact suffix costs two bpw. Larger budgets cannot add information.
        endpoint = retention[(retention.layer.isin(layers)) & (retention.budget_bpw == 2.0)]
        for budget in (2.5, 3.0):
            retention_summaries.append({"aggregate": label, "budget_bpw": budget, **summary(endpoint.retention_eta)})
    pd.DataFrame(retention_summaries).to_csv(args.result / "selective_dense_retention_summary.csv", index=False)


if __name__ == "__main__":
    main()
