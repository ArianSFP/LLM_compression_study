#!/usr/bin/env python3
"""Validate and summarize the refinement-rotation basis ceiling."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EVAL_ROWS = "refinement_rotation_validation.parquet"
OMP_ROWS = "refinement_rotation_omp_audit.parquet"
EVAL_FACTS = "refinement_rotation_evaluation_facts.json"
FIT_FACTS = "refinement_rotation_fit_facts.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def quantiles(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    data = np.asarray(values, np.float64)
    if not len(data) or np.any(~np.isfinite(data)):
        raise RuntimeError("summary values must be finite and nonempty")
    return {
        "p10": float(np.quantile(data, 0.1)),
        "median": float(np.quantile(data, 0.5)),
        "p90": float(np.quantile(data, 0.9)),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact table without pandas' optional tabulate dependency."""
    columns = list(map(str, frame.columns))
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    widths = [
        max(len(columns[index]), *(len(row[index]) for row in rows))
        for index in range(len(columns))
    ]
    header = "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(columns)) + " |"
    divider = "| " + " | ".join("-" * widths[index] for index in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join((header, divider, *body))


def expected_methods(config: Mapping[str, Any]) -> list[str]:
    methods = [
        *map(str, config["controls"]),
        *map(str, config["shared_methods"]),
        *map(str, config["oracle_methods"]),
    ]
    if len(methods) != len(set(methods)):
        raise RuntimeError("configured methods are duplicated")
    return sorted(methods)


def validate_inputs(
    config: Mapping[str, Any], fit_dir: Path, evaluation_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    fit_facts = json.loads((fit_dir / FIT_FACTS).read_text())
    facts = json.loads((evaluation_dir / EVAL_FACTS).read_text())
    if fit_facts.get("completed") is not True or facts.get("completed") is not True:
        raise RuntimeError("fit and evaluation must both be complete")
    if fit_facts.get("run_id") != config["run_id"] or facts.get("run_id") != config["run_id"]:
        raise RuntimeError("run identity changed")
    if facts.get("fit_facts_sha256") != sha256(fit_dir / FIT_FACTS):
        raise RuntimeError("evaluation does not bind the supplied fit facts")
    if facts.get("selection_split") != "validation" or facts.get("test_rows_admitted_or_used") is not False:
        raise RuntimeError("evaluation split boundary changed")
    frame = pd.read_parquet(evaluation_dir / EVAL_ROWS)
    omp = pd.read_parquet(evaluation_dir / OMP_ROWS)
    if len(frame) != int(facts["observed_rows"]) or len(omp) != int(facts["omp_rows"]):
        raise RuntimeError("artifact row counts disagree with facts")
    if set(map(str, frame["evaluation_split"].unique())) != {"validation"}:
        raise RuntimeError("non-validation row entered the analysis")
    if set(map(str, frame["method"].unique())) != set(expected_methods(config)):
        raise RuntimeError("method grid changed")
    budgets = set(map(int, config["physical_page_budgets"]))
    if set(map(int, frame["physical_pages"].unique())) != budgets:
        raise RuntimeError("physical/action budget grid changed")
    identity = ["request_id", "position", "layer", "expert_id", "method", "physical_pages"]
    if frame.duplicated(identity).any():
        raise RuntimeError("duplicate validation method/budget row")
    invocations = frame[["request_id", "position", "layer", "expert_id"]].drop_duplicates()
    expected = int(facts["observed_validation_invocations"])
    if len(invocations) != expected:
        raise RuntimeError("validation invocation count changed")
    per_invocation = len(expected_methods(config)) * len(budgets)
    if len(frame) != expected * per_invocation:
        raise RuntimeError("validation grid is incomplete")
    numeric = [
        "isolated_gate_up_recovery", "expert_output_recovery",
        "full_action_endpoint_recovery", "representation_bytes",
        "basis_storage_bytes", "transform_macs",
    ]
    if np.any(~np.isfinite(frame[numeric].to_numpy(np.float64))):
        raise RuntimeError("validation table contains nonfinite required values")
    physical = frame[frame["physically_encoded"].astype(bool)]
    if set(map(str, physical["budget_semantics"].unique())) != {"physical_512_byte_pages"}:
        raise RuntimeError("physical rows lost their page semantics")
    if np.any(physical["action_payload_bytes"].to_numpy(np.int64) != 512):
        raise RuntimeError("physical transformed/native action is not one page")
    nonphysical = frame[~frame["physically_encoded"].astype(bool)]
    if set(map(str, nonphysical["method"].unique())) != {"per_expert_joint_rank1024_fp32_oracle"}:
        raise RuntimeError("unexpected nonphysical method")
    if set(map(str, nonphysical["budget_semantics"].unique())) != {"action_count_only"}:
        raise RuntimeError("mathematical oracle is mislabeled as physical")
    return frame, omp, fit_facts, facts


def build_summaries(
    frame: pd.DataFrame, omp: pd.DataFrame, fit_facts: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    frontier_rows = []
    for (method, pages), group in frame.groupby(["method", "physical_pages"], sort=True):
        isolated = quantiles(group["isolated_gate_up_recovery"])
        output = quantiles(group["expert_output_recovery"])
        frontier_rows.append({
            "method": str(method), "physical_pages": int(pages), "n": int(len(group)),
            **{f"isolated_recovery_{key}": value for key, value in isolated.items()},
            **{f"expert_output_recovery_{key}": value for key, value in output.items()},
            "physically_encoded": bool(group["physically_encoded"].iloc[0]),
            "budget_semantics": str(group["budget_semantics"].iloc[0]),
        })
    frontier = pd.DataFrame(frontier_rows)

    unique = frame.sort_values("physical_pages").drop_duplicates(
        ["request_id", "position", "layer", "expert_id", "method"]
    )
    target_rows = []
    for method, group in unique.groupby("method", sort=True):
        for target, column in ((0.90, "actions_at_90pct"), (0.95, "actions_at_95pct"), (0.99, "actions_at_99pct")):
            values = pd.to_numeric(group[column], errors="coerce")
            finite = values[np.isfinite(values)]
            row = {
                "method": str(method), "target_recovery": target,
                "n": int(len(group)), "success_fraction": float(len(finite) / len(group)),
                "actions_p10": np.nan, "actions_median": np.nan, "actions_p90": np.nan,
                "action_bpw_median": np.nan,
                "physically_encoded": bool(group["physically_encoded"].iloc[0]),
            }
            if len(finite):
                summary = quantiles(finite)
                row.update({f"actions_{key}": value for key, value in summary.items()})
                if row["physically_encoded"]:
                    row["action_bpw_median"] = summary["median"] / 768.0
            target_rows.append(row)
    targets = pd.DataFrame(target_rows)

    layer_rows = []
    for (layer, method, pages), group in frame.groupby(
        ["layer", "method", "physical_pages"], sort=True,
    ):
        output = quantiles(group["expert_output_recovery"])
        layer_rows.append({
            "layer": int(layer), "method": str(method), "physical_pages": int(pages),
            "n": int(len(group)), **{f"expert_output_recovery_{key}": value for key, value in output.items()},
        })
    layers = pd.DataFrame(layer_rows)

    omp_rows = []
    for (method, target), group in omp.groupby(["method", "target_recovery"], sort=True):
        ratio = pd.to_numeric(group["diagonal_to_exact_ratio"], errors="coerce")
        finite = ratio[np.isfinite(ratio)]
        omp_rows.append({
            "method": str(method), "target_recovery": float(target), "n": int(len(group)),
            "finite_fraction": float(len(finite) / len(group)),
            "diagonal_to_exact_ratio_median": (
                np.nan if not len(finite) else float(np.quantile(finite, 0.5))
            ),
            "diagonal_to_exact_ratio_p90": (
                np.nan if not len(finite) else float(np.quantile(finite, 0.9))
            ),
        })
    omp_summary = pd.DataFrame(omp_rows)

    storage = unique.groupby("method", sort=True).first().reset_index()[[
        "method", "physically_encoded", "budget_semantics", "action_payload_bytes",
        "coordinates_per_action",
        "representation_code_bytes", "representation_scale_bytes", "representation_bytes",
        "basis_scope", "basis_storage_bytes", "transform_macs",
    ]]
    experts_per_layer = 256 if config is None else int(config["experts_per_layer"])
    amortized_basis = np.where(
        storage["basis_scope"] == "layer_shared",
        np.ceil(storage["basis_storage_bytes"] / experts_per_layer),
        storage["basis_storage_bytes"],
    ).astype(np.int64)
    storage["amortized_basis_bytes_per_expert"] = amortized_basis
    storage["resident_metadata_bytes_per_expert"] = (
        storage["representation_scale_bytes"].to_numpy(np.int64) + amortized_basis
    )
    expert_weights = 3_145_728
    one_bpw_bytes = expert_weights // 8
    storage["resident_metadata_equivalent_bpw"] = (
        8.0 * storage["resident_metadata_bytes_per_expert"] / expert_weights
    )
    storage["strict_one_bpw_action_pages"] = np.maximum(
        (one_bpw_bytes - storage["resident_metadata_bytes_per_expert"]) // 512, 0,
    ).astype(np.int64)

    fit_rows = []
    for layer, record in sorted(fit_facts["layers"].items(), key=lambda item: int(item[0])):
        ajd = record["ajd"]
        fit_rows.append({
            "layer": int(layer),
            "activation_top1024_energy_fraction": float(record["activation_top_rank_energy_fraction"]),
            "refinement_uniform_top1024_energy_fraction": float(record["uniform_top_rank_energy_fraction"]),
            "refinement_router_top1024_energy_fraction": float(record["router_top_rank_energy_fraction"]),
            "block_ajd_initial_offdiagonal_fraction": float(ajd["mean_initial_offdiagonal_fraction"]),
            "block_ajd_final_offdiagonal_fraction": float(ajd["mean_final_offdiagonal_fraction"]),
            "fit_wall_seconds": float(record["wall_seconds"]),
        })
    return {
        "refinement_rotation_frontier_summary.csv": frontier,
        "refinement_rotation_actions_to_target.csv": targets,
        "refinement_rotation_layer_summary.csv": layers,
        "refinement_rotation_omp_gap.csv": omp_summary,
        "refinement_rotation_storage_compute.csv": storage,
        "refinement_rotation_fit_summary.csv": pd.DataFrame(fit_rows),
    }


def make_plot(frontier: pd.DataFrame, output: Path, run_id: str) -> None:
    matplotlib.rcParams["svg.hashsalt"] = run_id
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for method, group in frontier.groupby("method", sort=True):
        group = group.sort_values("physical_pages")
        linestyle = "--" if not bool(group["physically_encoded"].iloc[0]) else "-"
        axes[0].plot(group["physical_pages"], group["isolated_recovery_median"], label=method, linestyle=linestyle)
        axes[1].plot(group["physical_pages"], group["expert_output_recovery_median"], label=method, linestyle=linestyle)
    axes[0].set(xlabel="512-byte pages (oracle: action count)", ylabel="Median gate/up correction recovery")
    axes[1].set(xlabel="512-byte pages (oracle: action count)", ylabel="Median expert-output qenergy recovery")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=7, loc="lower right")
    fig.savefig(output / "refinement_rotation_frontier.png", dpi=180)
    fig.savefig(output / "refinement_rotation_frontier.svg")
    plt.close(fig)


def build_report(
    config: Mapping[str, Any], summaries: Mapping[str, pd.DataFrame], facts: Mapping[str, Any],
) -> str:
    frontier = summaries["refinement_rotation_frontier_summary.csv"]
    targets = summaries["refinement_rotation_actions_to_target.csv"]
    omp = summaries["refinement_rotation_omp_gap.csv"]
    storage = summaries["refinement_rotation_storage_compute.csv"]
    at_768 = frontier[frontier["physical_pages"] == 768].sort_values(
        "expert_output_recovery_median", ascending=False,
    )
    physical_768 = at_768[at_768["physically_encoded"]]
    best = physical_768.iloc[0]
    native_diagonal = physical_768[
        physical_768["method"] == "native_row_diagonal"
    ].iloc[0]
    native_greedy = physical_768[
        physical_768["method"] == "native_row_exact_greedy"
    ].iloc[0]
    shared = physical_768[
        physical_768["method"].isin(set(map(str, config["shared_methods"])))
    ].iloc[0]
    encoded_ceiling = physical_768[
        physical_768["method"].isin(set(map(str, config["oracle_methods"])))
    ].iloc[0]
    oracle = at_768[at_768["method"] == "per_expert_joint_rank1024_fp32_oracle"].iloc[0]
    target95 = targets[targets["target_recovery"] == 0.95].sort_values("actions_median")
    native95 = target95[target95["method"] == "native_row_diagonal"].iloc[0]
    joint95 = target95[
        target95["method"] == "per_expert_joint_rank1024_int4"
    ].iloc[0]
    int2_endpoint = frontier[
        frontier["method"].astype(str).str.endswith("_int2")
        & (frontier["physical_pages"] == 1024)
    ].sort_values("isolated_recovery_median", ascending=False).iloc[0]
    int2_90 = targets[
        targets["method"].astype(str).str.endswith("_int2")
        & (targets["target_recovery"] == 0.90)
    ]
    strict_rows = []
    for record in storage.to_dict("records"):
        pages = int(record["strict_one_bpw_action_pages"])
        candidates = frontier[
            (frontier["method"] == record["method"])
            & (frontier["physical_pages"] <= pages)
        ].sort_values("physical_pages")
        if len(candidates):
            selected = candidates.iloc[-1]
            strict_rows.append({
                "method": record["method"],
                "strict_action_pages": pages,
                "evaluated_pages": int(selected["physical_pages"]),
                "output_recovery_median": float(selected["expert_output_recovery_median"]),
                "physically_encoded": bool(selected["physically_encoded"]),
            })
        else:
            strict_rows.append({
                "method": record["method"],
                "strict_action_pages": pages,
                "evaluated_pages": 0,
                "output_recovery_median": 0.0,
                "physically_encoded": bool(record["physically_encoded"]),
            })
    strict = pd.DataFrame(strict_rows)
    lines = [
        "# Refinement-aware rotation audit",
        "",
        "## Outcome",
        "",
        f"This train-fit, validation-only audit evaluated {int(facts['observed_validation_invocations'])} matched hot-expert invocations over layers 0/4/20/39. It isolates gate/up refinement by holding down at exact Q4. Recovery is expert-output qenergy reconstruction, not task accuracy.",
        "",
        f"At 768 physical 512-byte pages, the best encoded method is `{best.method}` with median isolated/output recovery {best.isolated_recovery_median:.4f}/{best.expert_output_recovery_median:.4f}. Native diagonal/exact-greedy reach {native_diagonal.isolated_recovery_median:.4f}/{native_diagonal.expert_output_recovery_median:.4f} and {native_greedy.isolated_recovery_median:.4f}/{native_greedy.expert_output_recovery_median:.4f}, respectively.",
        "",
        f"The best layer-shared transform is `{shared.method}` at {shared.isolated_recovery_median:.4f}/{shared.expert_output_recovery_median:.4f}. The best physically encoded per-expert ceiling is `{encoded_ceiling.method}` at {encoded_ceiling.isolated_recovery_median:.4f}/{encoded_ceiling.expert_output_recovery_median:.4f}; its dense per-expert basis is charged below and is not deployable at one total bpw. The unencoded correction eigenbasis oracle reaches {oracle.isolated_recovery_median:.4f}/{oracle.expert_output_recovery_median:.4f} at 768 action-count units and is not a physical bandwidth result.",
        "",
        f"For 95% exact gate/up correction recovery, per-expert joint INT4 needs a median {joint95.actions_median:.0f} pages versus {native95.actions_median:.0f} for native diagonal, a {native95.actions_median / joint95.actions_median:.2f}x reduction. This is a basis ceiling, not a deployable win: the per-expert basis exceeds the one-bpw storage allowance. The strongest MSE-fitted four-level INT2 full-support endpoint is `{int2_endpoint.method}` at {int2_endpoint.isolated_recovery_median:.4f} median correction recovery; maximum 90%-target success across INT2 methods is {int2_90.success_fraction.max():.1%}.",
        "",
        "## Median validation frontier",
        "",
        markdown_table(frontier[["method", "physical_pages", "isolated_recovery_median", "expert_output_recovery_median", "physically_encoded"]]),
        "",
        "## Physical pages to 95% exact gate/up correction recovery",
        "",
        markdown_table(target95[["method", "success_fraction", "actions_median", "actions_p90", "action_bpw_median", "physically_encoded"]]),
        "",
        "## Diagonal versus exact fixed-coefficient greedy",
        "",
        markdown_table(omp),
        "",
        "## Storage and transform cost",
        "",
        markdown_table(storage),
        "",
        "## Metadata-aware one-bpw control",
        "",
        "This conservative control subtracts per-column scales and the per-expert share of a layer-shared basis from 393,216 bytes before assigning 512-byte actions. Per-expert dense bases exceed the entire one-bpw allowance and therefore receive zero strict actions.",
        "",
        markdown_table(strict),
        "",
        "## Scope",
        "",
        "- Shared and per-expert bases are fit without validation values; selection and summaries use validation only. Test rows are not admitted.",
        "- Paired INT4 stores one transformed gate/up coordinate per 512-byte page. Full-rank INT2 uses an MSE-fitted symmetric four-level codebook and stores two fixed coordinate pairs per page; selectors rank these indivisible pages, not arbitrary half-pages. Separate INT8 stores one projection column per page. FP16 scales, basis storage, and transform MACs are charged separately.",
        "- Dense per-expert transforms are ceilings, not deployable candidates. The block-AJD transform is the low-cost shared control.",
        "- No H4 predictor, prefetch timing, causal replay, routing/logit agreement, token quality, perplexity, or downstream benchmark is evaluated.",
        "",
        f"Run scope: `{config['scope']}`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text())
    frame, omp, fit_facts, facts = validate_inputs(
        config, args.fit_dir, args.evaluation_dir,
    )
    args.output.mkdir(parents=True)
    summaries = build_summaries(frame, omp, fit_facts, config)
    for name, value in summaries.items():
        value.to_csv(args.output / name, index=False)
    make_plot(summaries["refinement_rotation_frontier_summary.csv"], args.output, config["run_id"])
    atomic_text(
        args.output / "REFINEMENT_ROTATION_REPORT.md",
        build_report(config, summaries, facts),
    )
    inputs = {
        "analyzer": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "config": {"path": str(args.config), "sha256": sha256(args.config)},
        "fit_facts": {"path": str(args.fit_dir / FIT_FACTS), "sha256": sha256(args.fit_dir / FIT_FACTS)},
        "evaluation_facts": {"path": str(args.evaluation_dir / EVAL_FACTS), "sha256": sha256(args.evaluation_dir / EVAL_FACTS)},
        "validation": {"path": str(args.evaluation_dir / EVAL_ROWS), "sha256": sha256(args.evaluation_dir / EVAL_ROWS)},
        "omp": {"path": str(args.evaluation_dir / OMP_ROWS), "sha256": sha256(args.evaluation_dir / OMP_ROWS)},
    }
    outputs = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(args.output.iterdir())
        if path.name != "refinement_rotation_analysis_manifest.json"
    }
    atomic_json(args.output / "refinement_rotation_analysis_manifest.json", {
        "run_id": config["run_id"], "completed": True,
        "selection_split": "validation", "test_rows_admitted_or_used": False,
        "inputs": inputs, "outputs": outputs,
    })


if __name__ == "__main__":
    main()
