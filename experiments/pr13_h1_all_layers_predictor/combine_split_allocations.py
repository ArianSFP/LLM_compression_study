#!/usr/bin/env python3
"""Combine independently evaluated PR #13 lower/upper allocation waves."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


RATE = 0.9987386067708334


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_half(root: Path, lower: bool) -> list[pd.DataFrame]:
    result = []
    for path in sorted(root.glob("allocation_layer_*.parquet")):
        layer = int(path.stem.split("_")[-1])
        if (lower and layer <= 19) or (not lower and layer >= 20):
            result.append(pd.read_parquet(path))
    return result


def summarize(frame: pd.DataFrame) -> list[dict[str, Any]]:
    result = []
    for method, local in frame.groupby("method", sort=True):
        recovered = local["base_qenergy_damage"] - local["exact_qenergy_damage"]
        oracle_recovered = local["base_qenergy_damage"] - local["oracle_exact_qenergy_damage"]
        oracle_median = float(local["oracle_qenergy_recovery"].median())
        median = float(local["qenergy_recovery"].median())
        result.append({
            "method": method,
            "groups": len(local),
            "overall_bpw": RATE,
            "oracle_utility_retained": float(recovered.sum() / oracle_recovered.sum()),
            "normalized_allocation_regret": float(
                (local["exact_qenergy_damage"] - local["oracle_exact_qenergy_damage"]).sum()
                / oracle_recovered.sum()
            ),
            "qenergy_recovery_p10": float(local["qenergy_recovery"].quantile(0.10)),
            "qenergy_recovery_median": median,
            "qenergy_recovery_p90": float(local["qenergy_recovery"].quantile(0.90)),
            "oracle_qenergy_recovery_p10": float(local["oracle_qenergy_recovery"].quantile(0.10)),
            "oracle_qenergy_recovery_median": oracle_median,
            "median_retention_ratio": median / oracle_median,
            "meets_90_percent_p10": bool(local["qenergy_recovery"].quantile(0.10) >= 0.90),
            "page_jaccard_mean": float(local["page_jaccard"].mean()),
            "state_overlap_mean": float(local["state_overlap"].mean()),
            "maximum_oracle_rescore_base_relative_error": float(
                local["canonical_oracle_damage_base_relative_error"].max()
            ),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lower", type=Path, required=True)
    parser.add_argument("--upper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frames = read_half(args.lower, True) + read_half(args.upper, False)
    if not frames:
        raise RuntimeError("no allocation frames found")
    frame = pd.concat(frames, ignore_index=True)
    layers = sorted(set(frame.layer.astype(int)))
    methods = sorted(set(frame.method.astype(str)))
    expected = len(layers) * len(methods) * 29
    if len(frame) != expected or frame.duplicated(["request_id", "position", "layer", "method"]).any():
        raise RuntimeError(f"non-canonical combined rows: {len(frame)} expected {expected}")
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output / "allocation_results.parquet", index=False)
    per_layer = []
    for (method, layer), local in frame.groupby(["method", "layer"], sort=True):
        per_layer.append({
            "method": method,
            "layer": int(layer),
            "groups": len(local),
            "qenergy_recovery_p10": float(local.qenergy_recovery.quantile(0.10)),
            "qenergy_recovery_median": float(local.qenergy_recovery.median()),
            "qenergy_recovery_mean": float(local.qenergy_recovery.mean()),
        })
    payload = {
        "layers": layers,
        "methods": methods,
        "rows": len(frame),
        "complete_all_40_layers": layers == list(range(40)),
        "summary": summarize(frame),
        "per_layer": per_layer,
    }
    atomic_json(args.output / "summary.json", payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
