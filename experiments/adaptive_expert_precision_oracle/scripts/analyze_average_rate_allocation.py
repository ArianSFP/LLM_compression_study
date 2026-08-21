#!/usr/bin/env python3
"""Analyze grouped top-8 average-rate allocation evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "src"))

from oracle_study.average_rate_analysis import (  # noqa: E402
    promotion_payload,
    sha256,
    summary_tables,
    validate_evidence_bundle,
)


ACCURACY_CSV = "average_rate_accuracy_by_bpw.csv"
LAYER_CSV = "average_rate_layer_accuracy.csv"
ALLOCATION_CSV = "average_rate_allocation_distribution.csv"
RUNTIME_CSV = "average_rate_runtime.csv"
PROMOTION_JSON = "average_rate_promotions.json"
REPORT_MD = "AVERAGE_RATE_ALLOCATION_REPORT.md"
PLOT_PNG = "average_rate_accuracy.png"
PLOT_SVG = "average_rate_accuracy.svg"
MANIFEST_JSON = "average_rate_analysis_manifest.json"


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _percent(value: float) -> str:
    return f"{100.0 * float(value):.3f}%"


def _pp(value: float) -> str:
    return f"{100.0 * float(value):+.3f} pp"


def _policy_name(value: str) -> str:
    return {
        "uniform_per_expert": "Uniform per expert",
        "pooled_router_square_compressed": "Pooled router-squared",
        "pooled_equal_weight_compressed": "Pooled equal weight",
        "pooled_exact_combined_moe_oracle": "Exact combined-MoE oracle",
    }.get(value, value)


def _report(bundle: Any, tables: dict[str, Any], promotion: dict[str, Any]) -> str:
    config = bundle.config
    accuracy = tables["accuracy_by_bpw"]
    primary = accuracy[
        accuracy["factor_config_id"].eq(config["primary_factor_id"])
        & (
            accuracy["allocation_policy"].eq("uniform_per_expert")
            | (
                accuracy["allocation_policy"].eq("pooled_router_square_compressed")
                & accuracy["burst_cap_pages_per_expert"].eq(1536)
            )
            | accuracy["allocation_policy"].eq("pooled_exact_combined_moe_oracle")
        )
    ].sort_values(["overall_average_bpw", "allocation_policy"])
    lines = [
        "# Average-rate allocation over true top-8 expert groups",
        "",
        "## Outcome",
        "",
        f"Continuation status: **{promotion['status']}**.",
        "",
        "This is an exact-H4 interaction-geometry ceiling over 128 captured "
        "validation token/layer groups (1,024 routed expert invocations). "
        "No row is deployable or promotable: exact Q4 activation-dependent "
        "responses and, for the oracle control, exact cross-expert residual "
        "vectors are used. The actionable question is whether pooled bandwidth "
        "allocation is strong enough to justify a future predicted-H4 allocator.",
        "",
        "## Accuracy by overall average bpw",
        "",
        "| Overall average bpw | Policy | Burst cap | qenergy p10 | qenergy median | "
        "qenergy p90 | Paired median vs uniform | Actual bpw median |",
        "| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in primary.itertuples(index=False):
        lines.append(
            f"| {row.overall_average_bpw:.6f} | {_policy_name(row.allocation_policy)} "
            f"| {int(row.burst_cap_pages_per_expert)} | "
            f"{_percent(row.qenergy_recovery_p10)} | "
            f"{_percent(row.qenergy_recovery_median)} | "
            f"{_percent(row.qenergy_recovery_p90)} | "
            f"{_pp(row.paired_gain_vs_uniform_median)} | "
            f"{row.actual_total_bpw_median:.6f} |"
        )
    lines += [
        "",
        "The table reports exact combined top-8 qenergy recovery, not token "
        "accuracy or model quality. Overall bpw includes the 9,232-byte "
        "rank-8 INT4+A/B/C resident sidecar and the allowed mean correction "
        "pages. Actual bpw can be lower when the frontier leaves budget unused.",
        "",
        "## Frozen primary decision",
        "",
        f"- Mean correction pages/expert: {promotion['mean_correction_pages_per_expert']}",
        f"- Overall average bpw: {promotion['overall_average_bpw']:.6f}",
        f"- Router-squared pooled recovery p10/median: "
        f"{_percent(promotion['qenergy_recovery_p10'])} / "
        f"{_percent(promotion['qenergy_recovery_median'])}",
        f"- Paired p10/median gain over uniform: "
        f"{_pp(promotion['paired_gain_vs_uniform_p10'])} / "
        f"{_pp(promotion['paired_gain_vs_uniform_median'])}",
        f"- Median-gain gate: {promotion['median_gain_gate_pass']}",
        f"- Tail-nonregression gate: {promotion['p10_gain_gate_pass']}",
        "",
        "## Runtime and optimization",
        "",
        "Each expert frontier uses one vectorized exact-self DP table shared "
        "across all hard budgets and page prices. Hard anchors receive bounded "
        "1/2/3-unit local repair; the descending price path uses two incremental "
        "coordinate basins without repeatedly rebuilding the DP or repairing "
        "discarded price points. CSV evidence reports actual wall time, "
        "coordinate sweeps, local passes, DP state evaluations, and charged MACs.",
        "",
        "## Q3 decision",
        "",
        f"Q3 status: **{promotion['q3_status']}**. Q3 was deliberately not mixed "
        "into this attribution experiment. A later Q3 run must include:",
    ]
    lines.extend(f"- {item}" for item in promotion["q3_required_controls"])
    lines += [
        "",
        "The physical study must charge unique 512-byte page IDs; the ideal "
        "256-byte bitplane ceiling alone is not a deployable Q3 result.",
        "",
        "## Scientific boundary",
        "",
        "- Fit: train split only; all 256 experts in each locked layer.",
        "- Selection and summaries: validation split only.",
        "- Test scientific tensors/values: not admitted or used.",
        "- No H4 causal replay, latency, routing, logit, token-quality, or "
        "end-to-end model claim is made.",
        "",
    ]
    return "\n".join(lines)


def _plot(tables: dict[str, Any], config: dict[str, Any], output: Path) -> None:
    matplotlib.rcParams["svg.hashsalt"] = str(config["run_id"])
    frame = tables["accuracy_by_bpw"]
    frame = frame[frame["factor_config_id"].eq(config["primary_factor_id"])]
    selections = {
        "Uniform": frame["allocation_policy"].eq("uniform_per_expert"),
        "Pooled router-squared, cap 768": (
            frame["allocation_policy"].eq("pooled_router_square_compressed")
            & frame["burst_cap_pages_per_expert"].eq(768)
        ),
        "Pooled router-squared, cap 1536": (
            frame["allocation_policy"].eq("pooled_router_square_compressed")
            & frame["burst_cap_pages_per_expert"].eq(1536)
        ),
        "Exact combined oracle": frame["allocation_policy"].eq(
            "pooled_exact_combined_moe_oracle"
        ),
    }
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for label, mask in selections.items():
        rows = frame[mask].sort_values("overall_average_bpw")
        ax.plot(
            rows["overall_average_bpw"],
            100 * rows["qenergy_recovery_median"],
            marker="o", label=f"{label} median",
        )
        ax.plot(
            rows["overall_average_bpw"],
            100 * rows["qenergy_recovery_p10"],
            marker=".", linestyle="--", alpha=.75, label=f"{label} p10",
        )
    ax.set_xlabel("Overall average bpw (metadata + allowed correction)")
    ax.set_ylabel("Exact combined top-8 qenergy recovery (%)")
    ax.grid(alpha=.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output / PLOT_PNG, dpi=180)
    svg_path = output / PLOT_SVG
    fig.savefig(svg_path, metadata={"Date": None})
    _atomic_text(
        svg_path,
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines())
        + "\n",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source_paths = {
        "runner_sha256": EXPERIMENT / "scripts/run_average_rate_allocation.py",
        "allocator_core_sha256": EXPERIMENT / "src/oracle_study/average_rate_allocator.py",
        "split_core_sha256": EXPERIMENT / "src/oracle_study/split_interaction_field.py",
        "interaction_core_sha256": EXPERIMENT / "src/oracle_study/interaction_field.py",
        "selector_core_sha256": EXPERIMENT / "src/oracle_study/neuron_selector.py",
        "mxfp4_core_sha256": EXPERIMENT / "src/oracle_study/mxfp4_embed.py",
        "set_utility_runner_sha256": EXPERIMENT / "scripts/run_set_utility_distillation.py",
        "sparse_runner_sha256": EXPERIMENT / "scripts/run_sparse_streaming_study.py",
    }
    bundle = validate_evidence_bundle(
        args.config, args.fit_dir, args.validation_dir, source_paths=source_paths,
    )
    tables = summary_tables(bundle)
    promotion = promotion_payload(bundle, tables)
    args.output.mkdir(parents=True)
    output_map = {
        ACCURACY_CSV: tables["accuracy_by_bpw"],
        LAYER_CSV: tables["layer_accuracy"],
        ALLOCATION_CSV: tables["allocation_distribution"],
        RUNTIME_CSV: tables["runtime"],
    }
    for name, frame in output_map.items():
        frame.to_csv(args.output / name, index=False)
    _atomic_json(args.output / PROMOTION_JSON, promotion)
    _atomic_text(args.output / REPORT_MD, _report(bundle, tables, promotion))
    _plot(tables, bundle.config, args.output)

    def relative(path: Path) -> str:
        return Path(path).resolve().relative_to(EXPERIMENT.resolve()).as_posix()

    inputs = []
    for name, path in bundle.input_paths.items():
        inputs.append({
            "name": name,
            "path": relative(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    for field, path in source_paths.items():
        inputs.append({
            "name": field,
            "path": relative(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    generated = []
    for path in sorted(args.output.iterdir()):
        if path.name == MANIFEST_JSON:
            continue
        generated.append({
            "path": relative(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    manifest = {
        "schema_version": 1,
        "run_id": bundle.config["run_id"],
        "inputs": sorted(inputs, key=lambda row: (row["name"], row["path"])),
        "generated": generated,
        "row_counts": {
            "group_frontier": len(bundle.groups),
            "expert_allocation": len(bundle.experts),
            "accuracy_summary": len(tables["accuracy_by_bpw"]),
            "layer_summary": len(tables["layer_accuracy"]),
        },
        "promotion_sha256": sha256(args.output / PROMOTION_JSON),
        "test_scientific_rows_admitted_or_used": False,
        "exact_h4_geometry_ceiling_only": True,
    }
    _atomic_json(args.output / MANIFEST_JSON, manifest)


if __name__ == "__main__":
    main()
