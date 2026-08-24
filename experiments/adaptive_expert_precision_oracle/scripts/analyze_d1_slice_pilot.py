#!/usr/bin/env python3
"""Analyze D1 layer-slice finalists and exact request-level reranking."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "src"))

from oracle_study.d1_accounting import controller_accounting  # noqa: E402


SCHEMA = "pr13_d1_layer_slice_analysis_v1"
PR13 = "pr13_router_square_column_generated"
REPAIRED = "d1_exact_request_group_repaired"
LOCAL_PREFIX = "exact_combined_local_"
RERANKED = "d1_exact_request_reranked"
CHARGED_BPW = [
    0.5234781901041666,
    0.7734781901041666,
    0.9987386067708334,
    1.0234781901041666,
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)



def _markdown(frame: pd.DataFrame) -> str:
    def render(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        return str(value)
    headers = [str(column) for column in frame.columns]
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    rows.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(rows)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--repair-dir", type=Path, action="append", default=[])
    return parser.parse_args()


def _read_inputs(
    directories: list[Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = []
    allocations = []
    vjps = []
    parity = []
    seen_layer_rates: set[tuple[int, tuple[int, ...]]] = set()
    for directory in directories:
        facts = json.loads((directory / "d1_layer_facts.json").read_text())
        layer = int(facts["layer"])
        layer_rates = tuple(sorted(int(rate) for rate in facts["rates"]))
        key = (layer, layer_rates)
        if key in seen_layer_rates:
            raise ValueError(f"duplicate layer/rate input: {key}")
        seen_layer_rates.add(key)
        metrics.append(pd.read_parquet(directory / "d1_exact_route_metrics.parquet"))
        allocations.append(pd.read_parquet(directory / "d1_allocation_groups.parquet"))
        vjps.append(pd.read_parquet(directory / "d1_vjp_metrics.parquet").assign(layer=layer))
        parity.append(pd.read_parquet(directory / "d1_paired_parity.parquet").assign(layer=layer))
    return (
        pd.concat(metrics, ignore_index=True),
        pd.concat(allocations, ignore_index=True),
        pd.concat(vjps, ignore_index=True),
        pd.concat(parity, ignore_index=True),
    )


def _candidate_choices(
    metrics: pd.DataFrame,
    allocations: pd.DataFrame,
) -> pd.DataFrame:
    candidates = metrics[metrics["policy"].str.startswith("d1_strict_")].copy()
    allocation = allocations[
        allocations["policy"].str.startswith("d1_strict_")
    ][[
        "layer", "request_id", "policy", "rate_pages_per_expert",
        "selected_group_pages",
    ]]
    request = (
        candidates.groupby(
            ["layer", "rate_pages_per_expert", "request_id", "policy"],
            sort=True,
        )
        .agg(
            groups=("group", "size"),
            exact_crossings=("exact_d1_crossed", "sum"),
            routing_mass_lost=("routing_mass_lost", "sum"),
            local_qenergy_damage=("local_qenergy_damage", "sum"),
        )
        .reset_index()
    )
    pages = (
        allocation.groupby(
            ["layer", "rate_pages_per_expert", "request_id", "policy"],
            sort=True,
        )["selected_group_pages"]
        .sum()
        .reset_index(name="selected_pages")
    )
    request = request.merge(
        pages,
        on=["layer", "rate_pages_per_expert", "request_id", "policy"],
        validate="one_to_one",
    )
    choices = []
    for _, frame in request.groupby(
        ["layer", "rate_pages_per_expert", "request_id"], sort=True,
    ):
        chosen = frame.sort_values(
            [
                "exact_crossings",
                "local_qenergy_damage",
                "routing_mass_lost",
                "selected_pages",
                "policy",
            ],
            kind="stable",
        ).iloc[0]
        choices.append(chosen.to_dict())
    result = pd.DataFrame(choices)
    result.insert(0, "schema", SCHEMA)
    return result


def _reranked_metrics(
    metrics: pd.DataFrame,
    choices: pd.DataFrame,
) -> pd.DataFrame:
    identity = ["layer", "rate_pages_per_expert", "request_id"]
    selected = choices[identity + ["policy"]].rename(
        columns={"policy": "selected_source_policy"},
    )
    candidates = metrics.merge(selected, on=identity, validate="many_to_one")
    chosen = candidates[
        candidates["policy"].eq(candidates["selected_source_policy"])
    ].copy()
    chosen["source_policy"] = chosen["policy"]
    chosen["policy"] = RERANKED
    return chosen.drop(columns=["selected_source_policy"])


def _summarize(frame: pd.DataFrame) -> pd.DataFrame:
    result = (
        frame.groupby(["layer", "rate_pages_per_expert", "policy"], sort=True)
        .agg(
            groups=("group", "size"),
            exact_d1_crossings=("exact_d1_crossed", "sum"),
            exact_d1_crossing_rate=("exact_d1_crossed", "mean"),
            mean_exact_top8_agreement=("exact_top8_agreement", "mean"),
            mean_membership_pairs_changed=("membership_pairs_changed", "mean"),
            mean_routing_mass_lost=("routing_mass_lost", "mean"),
            mean_local_qenergy_damage=("local_qenergy_damage", "mean"),
        )
        .reset_index()
    )
    pr13 = result[result["policy"].eq(PR13)][[
        "layer", "rate_pages_per_expert", "mean_local_qenergy_damage",
    ]].rename(columns={"mean_local_qenergy_damage": "pr13_mean_local_qenergy_damage"})
    result = result.merge(
        pr13, on=["layer", "rate_pages_per_expert"], validate="many_to_one",
    )
    result["local_damage_ratio_to_pr13"] = (
        result["mean_local_qenergy_damage"]
        / result["pr13_mean_local_qenergy_damage"]
    )
    result.insert(0, "schema", SCHEMA)
    return result


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    result = (
        frame.groupby(["rate_pages_per_expert", "policy"], sort=True)
        .agg(
            layers=("layer", "nunique"),
            groups=("group", "size"),
            exact_d1_crossings=("exact_d1_crossed", "sum"),
            exact_d1_crossing_rate=("exact_d1_crossed", "mean"),
            mean_exact_top8_agreement=("exact_top8_agreement", "mean"),
            mean_routing_mass_lost=("routing_mass_lost", "mean"),
            mean_local_qenergy_damage=("local_qenergy_damage", "mean"),
        )
        .reset_index()
    )
    pr13 = result[result["policy"].eq(PR13)][[
        "rate_pages_per_expert", "mean_local_qenergy_damage",
    ]].rename(columns={"mean_local_qenergy_damage": "pr13_mean_local_qenergy_damage"})
    result = result.merge(pr13, on="rate_pages_per_expert", validate="many_to_one")
    result["local_damage_ratio_to_pr13"] = (
        result["mean_local_qenergy_damage"]
        / result["pr13_mean_local_qenergy_damage"]
    )
    result.insert(0, "schema", SCHEMA)
    return result


def _report(
    layer_summary: pd.DataFrame,
    aggregate: pd.DataFrame,
    choices: pd.DataFrame,
    facts: dict[str, Any],
) -> str:
    comparison = aggregate[
        aggregate["policy"].isin([PR13, RERANKED, REPAIRED])
        | aggregate["policy"].str.startswith(LOCAL_PREFIX)
    ].copy()
    columns = [
        "rate_pages_per_expert", "policy", "groups", "exact_d1_crossings",
        "exact_d1_crossing_rate", "mean_local_qenergy_damage",
        "local_damage_ratio_to_pr13",
    ]
    table = _markdown(comparison[columns])
    by_layer = _markdown(layer_summary[
        layer_summary["policy"].isin([PR13, RERANKED, REPAIRED])
        | layer_summary["policy"].str.startswith(LOCAL_PREFIX)
    ][[
        "layer", "rate_pages_per_expert", "policy", "exact_d1_crossings",
        "exact_d1_crossing_rate", "mean_local_qenergy_damage",
        "local_damage_ratio_to_pr13",
    ]])
    chosen = _markdown(
        choices.groupby(["layer", "rate_pages_per_expert", "policy"], sort=True)
        .size()
        .reset_index(name="requests")
    )
    accounting = facts["controller_accounting"]
    benchmark = facts.get("runtime_benchmark")
    benchmark_text = "Not supplied."
    if benchmark is not None:
        tensor = benchmark["tensor_scorer"]
        benchmark_text = (
            f"On {benchmark['device_name']}, the device scorer measured "
            f"{tensor['median_ms']:.3f} ms median and "
            f"{tensor['p90_ms']:.3f} ms p90 per refresh."
        )
    layers_text = ", ".join(str(layer) for layer in facts["layers"])
    return f"""# D1 layer-slice pilot report

## Scientific boundary

This is a paired RTX 3090, validation-only D1 oracle pilot over layers
{layers_text}. D1 is exactly the same token's next-layer router. The experiment loads
complete PR #13 frontier options and performs true pre-residual routed-output
replacement, but executes only through the next router. It makes no terminal
KL, D2-D4, all-layer, or production-controller claim.

The immutable PRO 6000 captures remain the activation/allocation foundation.
Small cross-device BF16 differences are measured separately. Every policy is
scored against a bit-repeatable paired 3090 Q4 slice baseline, so hardware
drift is not counted as allocator-caused switching.

## Exact D1 result

{table}

The request-level reranker is oracle-only. It selects among one-sided D1
finalists by exact D1 crossings first, then local qenergy, routing-mass loss,
pages, and deterministic policy name. Exact labels are never proposed as
runtime inputs.

The promoted request-coupled repair is also oracle-only. It mixes complete
one-sided finalists when an exact replay exposes a causal-prefix interaction.
Its pair search is restricted to source positions at or before the earliest
remaining crossing. This repair is a scientific upper-bound/reranking step,
not part of the measured deployable scorer.

### By layer

{by_layer}

### Finalist choices

{chosen}

## Runtime cost boundary

The one-adjoint controller estimate is
{accounting['one_adjoint_total_macs']:,} MAC/group, or
{100.0 * accounting['one_adjoint_fraction_of_pr13']:.3f}% of the frozen
PR #13 selector estimate ({accounting['pr13_primary_selector_macs']:,}
MAC/group). Joint D1 metadata changes the matched page caps to
{accounting['matched_pages']}; with uncertainty metadata they are
{accounting['matched_pages_with_uncertainty']}.

{benchmark_text}

Exact frontier generation, VJPs, coordinate/pair search, and exact replay in
this pilot are oracle-label costs and are excluded from the deployable runtime
budget.

## Interpretation

Across the supplied slices, changing which complete refinement options are
selected can remove the observed D1 membership changes at matched page caps
while retaining a tight additive local-qenergy guard. This establishes the D1
allocation mechanism on the sampled layers; it does not yet establish final
KL improvement. The next scientific gate is exact downstream replay of the
reranked finalists, followed only then by predictor/controller work.
"""


def main() -> None:
    args = parse_args()
    metrics, allocations, vjps, parity = _read_inputs(args.layer_dir)
    for directory in args.repair_dir:
        metrics = pd.concat([
            metrics,
            pd.read_parquet(directory / "d1_exact_repair_metrics.parquet"),
        ], ignore_index=True)
        allocations = pd.concat([
            allocations,
            pd.read_parquet(
                directory / "d1_exact_repair_allocation_groups.parquet"
            ),
        ], ignore_index=True)
    choices = _candidate_choices(metrics, allocations)
    reranked = _reranked_metrics(metrics, choices)
    comparison = pd.concat([
        metrics[
            metrics["policy"].isin([PR13, REPAIRED])
            | metrics["policy"].str.startswith(LOCAL_PREFIX)
        ].copy(),
        reranked,
    ], ignore_index=True)
    layer_summary = _summarize(comparison)
    aggregate = _aggregate(comparison)
    accounting = controller_accounting(CHARGED_BPW).to_dict()
    benchmark = None
    if args.benchmark is not None:
        benchmark = json.loads(args.benchmark.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_parquet(args.output_dir / "d1_exact_reranked_request_choices.parquet", choices)
    atomic_parquet(args.output_dir / "d1_exact_reranked_metrics.parquet", reranked)
    atomic_parquet(args.output_dir / "d1_layer_policy_summary.parquet", layer_summary)
    atomic_parquet(args.output_dir / "d1_aggregate_policy_summary.parquet", aggregate)
    atomic_parquet(args.output_dir / "d1_vjp_summary.parquet", vjps)
    atomic_parquet(args.output_dir / "d1_paired_parity_summary.parquet", parity)
    facts = {
        "schema": SCHEMA,
        "completed": True,
        "layers": sorted(metrics["layer"].unique().astype(int).tolist()),
        "rates": sorted(metrics["rate_pages_per_expert"].unique().astype(int).tolist()),
        "requests": sorted(metrics["request_id"].unique().tolist()),
        "groups_per_layer": 32,
        "reranking_scope": "exact_request_level_d1_finalists_oracle_only",
        "reranking_order": [
            "fewest_exact_d1_crossings",
            "lowest_local_qenergy",
            "lowest_routing_mass_loss",
            "fewest_pages",
            "deterministic_policy_name",
        ],
        "controller_accounting": accounting,
        "runtime_benchmark": benchmark,
        "exact_vjp_label_median_ms": float(1000.0 * vjps["exact_vjp_seconds"].median()),
        "exact_vjp_label_p90_ms": float(1000.0 * vjps["exact_vjp_seconds"].quantile(0.9)),
        "input_hashes": {
            str(path): sha256(path)
            for directory in args.layer_dir
            for path in sorted(directory.iterdir())
            if path.is_file()
        },
        "repair_input_hashes": {
            str(path): sha256(path)
            for directory in args.repair_dir
            for path in sorted(directory.iterdir())
            if path.is_file()
        },
        "test_rows_admitted_or_used": False,
        "terminal_kl_claim": False,
        "d2_d4_in_scope": False,
    }
    if args.benchmark is not None:
        facts["input_hashes"][str(args.benchmark)] = sha256(args.benchmark)
    report = _report(layer_summary, aggregate, choices, facts)
    report_path = args.output_dir / "D1_LAYER_SLICE_PILOT_REPORT.md"
    report_path.write_text(report)
    facts["outputs"] = {
        path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(args.output_dir.iterdir())
        if path.is_file() and path.name != "d1_slice_analysis_facts.json"
    }
    atomic_json(args.output_dir / "d1_slice_analysis_facts.json", facts)
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
