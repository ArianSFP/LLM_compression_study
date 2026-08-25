#!/usr/bin/env python3
"""Descriptive outcome analysis for sealed nested D1 Experiment A.

The analyzer authenticates finalized runner tables, reduces layers equally
within each independent request, and reports route-mode decomposition.  It
does not select, rerank, or modify any allocation and does not make an
Experiment-B promotion decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "src"))

from oracle_study.d1_nested_outcomes import (  # noqa: E402
    CACHE_FILE,
    CONTRAST_FILE,
    OUTCOME_SCHEMA,
    PROPAGATION_FILE,
    QUALITY_FILE,
    RUN_FACTS_FILE,
    ZERO_FILE,
    file_sha256,
    request_layer_means,
    require_zero_dose_parity,
    route_mode_contrasts,
)
from run_same_host_causal_controls import atomic_json, atomic_parquet  # noqa: E402


REQUEST_MEANS = "d1_nested_outcome_request_means.parquet"
ROUTE_MODE_REQUEST_CONTRASTS = "d1_nested_outcome_route_mode_request_contrasts.parquet"
SUMMARY = "d1_nested_outcome_descriptive_summary.parquet"
ANALYSIS_FACTS = "d1_nested_outcome_analysis_facts.json"
HOST_FACTS = "d1_nested_outcome_host_facts.json"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _load_authenticated_tables(root: Path) -> tuple[
    dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
]:
    facts = load_json(root / RUN_FACTS_FILE)
    if facts.get("schema") != OUTCOME_SCHEMA or facts.get("completed") is not True:
        raise RuntimeError("nested outcome run facts are incomplete")
    outputs = facts.get("outputs")
    if not isinstance(outputs, dict):
        raise RuntimeError("nested outcome output inventory is absent")
    host_record = facts.get("host_facts")
    host_path = root / HOST_FACTS
    if (
        not isinstance(host_record, dict)
        or host_record.get("file") != HOST_FACTS
        or not host_path.is_file()
        or host_record.get("sha256") != file_sha256(host_path)
        or int(host_record.get("bytes", -1)) != host_path.stat().st_size
    ):
        raise RuntimeError("nested outcome host provenance changed")
    frames = []
    for name in (QUALITY_FILE, PROPAGATION_FILE, CACHE_FILE, ZERO_FILE, CONTRAST_FILE):
        path = root / name
        record = outputs.get(name)
        if (
            not isinstance(record, dict)
            or not path.is_file()
            or record.get("sha256") != file_sha256(path)
            or int(record.get("bytes", -1)) != path.stat().st_size
        ):
            raise RuntimeError(f"finalized outcome artifact changed: {path}")
        frame = pd.read_parquet(path)
        if len(frame) != int(record.get("rows", -1)):
            raise RuntimeError(f"finalized outcome row count changed: {path}")
        frames.append(frame)
    return (facts, *frames)


def _candidate_tail_aggregates(propagation: pd.DataFrame) -> pd.DataFrame:
    identity = [
        "split", "arm", "rate_pages_per_expert", "injection_layer",
        "request_id", "position", "route_mode",
    ]
    required = set(identity + [
        "route_membership_change_fraction", "router_mass_churn", "hidden_mse",
    ])
    missing = sorted(required.difference(propagation.columns))
    if missing:
        raise ValueError(f"propagation table is missing columns: {missing}")
    return (
        propagation.groupby(identity, sort=True, dropna=False)
        .agg(
            mean_downstream_route_membership_change_fraction=(
                "route_membership_change_fraction", "mean",
            ),
            mean_downstream_router_mass_churn=("router_mass_churn", "mean"),
            mean_downstream_hidden_mse=("hidden_mse", "mean"),
            maximum_downstream_hidden_mse=("hidden_mse", "max"),
            downstream_layers_observed=("observation_layer", "nunique"),
        )
        .reset_index()
    )


def _candidate_cache_aggregates(cache: pd.DataFrame) -> pd.DataFrame:
    identity = [
        "split", "arm", "rate_pages_per_expert", "injection_layer",
        "request_id", "position", "route_mode",
    ]
    required = set(identity + [
        "full_cache_mse", "attention_kv_mse", "deltanet_conv_mse",
        "deltanet_recurrent_mse",
    ])
    missing = sorted(required.difference(cache.columns))
    if missing:
        raise ValueError(f"cache table is missing columns: {missing}")
    if cache.duplicated(identity).any():
        raise ValueError("cache table repeats a candidate identity")
    return cache[identity + [
        "full_cache_mse", "attention_kv_mse", "deltanet_conv_mse",
        "deltanet_recurrent_mse",
    ]].copy()


def _contrast_request_means(contrasts: pd.DataFrame) -> pd.DataFrame:
    identity = [
        "split", "request_id", "arm", "rate_pages_per_expert", "injection_layer",
    ]
    values = [
        "live_minus_fully_frozen_logit_kl",
        "live_minus_frozen_set_live_weights_logit_kl",
        "frozen_set_live_weights_minus_fully_frozen_logit_kl",
    ]
    missing = sorted(set(identity + values).difference(contrasts.columns))
    if missing:
        raise ValueError(f"route-mode contrast table is missing columns: {missing}")
    if contrasts.empty or contrasts.duplicated(identity).any():
        raise ValueError("route-mode contrasts are empty or duplicated")
    expected_layers = set(map(int, contrasts["injection_layer"].unique()))
    expected_cells: set[tuple[str, int]] | None = None
    for request_id, rows in contrasts.groupby("request_id", sort=False):
        cells = set(zip(
            rows["arm"].astype(str), rows["rate_pages_per_expert"].astype(int),
            strict=True,
        ))
        if expected_cells is None:
            expected_cells = cells
        elif cells != expected_cells:
            raise ValueError(f"route-mode contrast grid differs for request {request_id}")
        for cell, part in rows.groupby(["arm", "rate_pages_per_expert"], sort=False):
            if set(map(int, part["injection_layer"])) != expected_layers:
                raise ValueError(
                    f"route-mode contrast layer grid differs for request {request_id}, {cell}"
                )
    keys = ["split", "request_id", "arm", "rate_pages_per_expert"]
    result = (
        contrasts.groupby(keys, sort=True, dropna=False)[values]
        .mean()
        .reset_index()
    )
    result["layers_averaged"] = len(expected_layers)
    return result


def _summary_rows(
    request_means: pd.DataFrame,
    value_columns: Sequence[str],
) -> pd.DataFrame:
    rows = []
    keys = ["split", "arm", "rate_pages_per_expert", "route_mode"]
    for identity, part in request_means.groupby(keys, sort=True, dropna=False):
        for metric in value_columns:
            values = part[metric].to_numpy(np.float64)
            rows.append({
                **dict(zip(keys, identity, strict=True)),
                "metric": metric,
                "requests": len(values),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p90": float(np.quantile(values, 0.90)),
                "p95": float(np.quantile(values, 0.95)),
                "maximum": float(np.max(values)),
            })
    return pd.DataFrame(rows)


def analyze(root: Path, output: Path) -> None:
    facts, quality, propagation, cache, zero, stored_contrasts = (
        _load_authenticated_tables(root)
    )
    require_zero_dose_parity(zero)
    recomputed_contrasts = route_mode_contrasts(quality)
    order = [
        "split", "arm", "rate_pages_per_expert", "injection_layer",
        "request_id", "position",
    ]
    stored = stored_contrasts.sort_values(order, kind="stable").reset_index(drop=True)
    recomputed = recomputed_contrasts.sort_values(order, kind="stable").reset_index(drop=True)
    if list(stored.columns) != list(recomputed.columns) or not stored.equals(recomputed):
        raise RuntimeError("stored route-mode contrasts do not reproduce exactly")

    identity = [
        "split", "arm", "rate_pages_per_expert", "injection_layer",
        "request_id", "position", "route_mode",
    ]
    enriched = quality.merge(
        _candidate_tail_aggregates(propagation), on=identity, validate="one_to_one",
    ).merge(
        _candidate_cache_aggregates(cache), on=identity, validate="one_to_one",
    )
    values = [
        "logit_kl", "delta_nll", "final_hidden_mse",
        "mean_downstream_route_membership_change_fraction",
        "mean_downstream_router_mass_churn", "mean_downstream_hidden_mse",
        "maximum_downstream_hidden_mse", "full_cache_mse", "attention_kv_mse",
        "deltanet_conv_mse", "deltanet_recurrent_mse",
    ]
    request_means = request_layer_means(enriched, values)
    request_domains = quality[["request_id", "domain"]].drop_duplicates()
    if request_domains.groupby("request_id")["domain"].nunique().gt(1).any():
        raise RuntimeError("one request appears in multiple domains")
    request_means = request_means.merge(
        request_domains, on="request_id", validate="many_to_one",
    )
    contrast_means = _contrast_request_means(recomputed)
    summary = _summary_rows(request_means, values)
    output.mkdir(parents=True, exist_ok=True)
    atomic_parquet(output / REQUEST_MEANS, request_means)
    atomic_parquet(output / ROUTE_MODE_REQUEST_CONTRASTS, contrast_means)
    atomic_parquet(output / SUMMARY, summary)
    outputs = {
        name: {
            "sha256": file_sha256(output / name),
            "bytes": (output / name).stat().st_size,
            "rows": len(frame),
        }
        for name, frame in (
            (REQUEST_MEANS, request_means),
            (ROUTE_MODE_REQUEST_CONTRASTS, contrast_means),
            (SUMMARY, summary),
        )
    }
    atomic_json(output / ANALYSIS_FACTS, {
        "schema": "pr13_d1_nested_cached_decode_outcome_analysis_v1",
        "completed": True,
        "source_run_facts_sha256": file_sha256(root / RUN_FACTS_FILE),
        "source_allocation_manifest_sha256": facts["input_pins"][
            "allocation_manifest_sha256"
        ],
        "outputs": outputs,
        "unit_of_independence": "request",
        "within_request_layer_weighting": "equal",
        "terminal_metrics_are_outcomes_only": True,
        "terminal_metrics_used_for_allocation_selection": False,
        "descriptive_only_no_promotion_decision": True,
        "experiment_b_started": False,
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcome-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.output = args.output or args.outcome_root / "analysis"
    return args


def main() -> None:
    args = parse_args()
    analyze(args.outcome_root, args.output)


if __name__ == "__main__":
    main()
