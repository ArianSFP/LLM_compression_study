#!/usr/bin/env python3
"""Analyze validation evidence and freeze neuron-selector promotions."""

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
import pandas as pd

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "src"))

from oracle_study.neuron_selector_analysis import (
    choose_validation_promotions,
    evidence_frame,
    quantiles,
    sha256,
    validate_promotion_payload,
)

matplotlib.rcParams["svg.hashsalt"] = "neuron-selector-distillation-20260819"


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    values = frame.copy()
    for column in values:
        values[column] = values[column].map(
            lambda value: f"{value:.6f}" if isinstance(value, (float, np.floating))
            else str(value).replace("|", "\\|").replace("\n", " ")
        )
    headers = list(map(str, values.columns))
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in values.itertuples(index=False, name=None))
    return "\n".join(lines)


def validate_facts(path: Path, *, split: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("completed") is not True or value.get("failures") != []:
        raise RuntimeError(f"run facts are incomplete or failed: {path}")
    if value.get("evaluation_split") != split:
        raise RuntimeError(f"run facts split changed: {value.get('evaluation_split')} != {split}")
    if int(value.get("observed_unique_invocations", -1)) != int(value.get("expected_unique_invocations", -2)):
        raise RuntimeError("evaluation cohort is incomplete")
    if value.get("request_separation_verified") is not True or value.get("codec_locked") is not True:
        raise RuntimeError("locked execution controls were not verified")
    return value


def frontier_summary(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for key, local in frame.groupby(groups, dropna=False, sort=True):
        values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(groups, values))
        row.update({f"recovery_{name}": value for name, value in quantiles(local["recovery"]).items()})
        for column in (
            "physical_pages", "physical_bytes", "physical_bpw", "logical_actions",
            "logical_bytes", "logical_bpw", "page_amplification", "selector_runtime_ms",
            "selector_bytes_read", "selector_metadata_bpw", "storage_multiplier",
            "gate_up_actions", "down_actions", "path_actions_evaluated",
        ):
            if column in local:
                row[f"{column}_median"] = float(pd.to_numeric(local[column]).median())
        rows.append(row)
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, output: Path, stem: str) -> list[Path]:
    paths = [output / f"{stem}.png", output / f"{stem}.svg"]
    fig.savefig(paths[0], dpi=180, bbox_inches="tight")
    fig.savefig(
        paths[1], bbox_inches="tight",
        metadata={"Date": "2026-08-19", "Creator": "neuron-selector-distillation"},
    )
    svg_text = paths[1].read_text()
    paths[1].write_text("\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n")
    plt.close(fig)
    return paths


def promote(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    fit_facts = json.loads((args.fit_dir / "fit_facts.json").read_text())
    validation_facts = validate_facts(args.validation_dir / "run_facts.json", split="validation")
    bundle_path = args.fit_dir / "response_fit_bundle_manifest.json"
    bundle_manifest = json.loads(bundle_path.read_text())
    if set(bundle_manifest.get("layers", {})) != set(map(str, config["layers"])):
        raise RuntimeError("fit bundle manifest does not contain every configured layer")
    bundle_shards = []
    for record in bundle_manifest["layers"].values():
        path = args.fit_dir / str(record["path"])
        if sha256(path) != str(record["sha256"]) or path.stat().st_size != int(record["bytes"]):
            raise RuntimeError("fit bundle shard changed")
        bundle_shards.append(path)
    if fit_facts.get("bundle_sha256") != sha256(bundle_path):
        raise RuntimeError("fit bundle differs from fit facts")
    if validation_facts.get("fit_bundle_sha256") != sha256(bundle_path):
        raise RuntimeError("validation rows used a different fit bundle")
    if validation_facts.get("config_sha256") != sha256(args.config):
        raise RuntimeError("validation rows used a different config")

    paths = {
        "factorized": args.validation_dir / "factorized_neuron_frontier.parquet",
        "response": args.validation_dir / "response_predictor_frontier.parquet",
        "candidate": args.validation_dir / "candidate_rerank_frontier.parquet",
        "diagnostics": args.validation_dir / "response_prediction_diagnostics.parquet",
    }
    factorized = evidence_frame(paths["factorized"], split="validation")
    response = evidence_frame(paths["response"], split="validation")
    candidate = evidence_frame(paths["candidate"], split="validation")
    diagnostics = evidence_frame(paths["diagnostics"], split="validation")
    comparator_record = config["pr7_comparators"]["validation_tile_frontier"]
    comparator_path = EXPERIMENT / comparator_record["path"]
    tile = pd.read_parquet(
        comparator_path,
        filters=[
            ("evaluation_split", "==", "validation"),
            ("selector", "==", comparator_record["selector"]),
            ("tile_shape", "==", comparator_record["tile_shape"]),
            ("physical_budget_bpw", "==", 1.0),
        ],
    )
    if set(tile["evaluation_split"].astype(str)) != {"validation"}:
        raise RuntimeError("PR7 comparator admitted non-validation rows")
    evidence_hashes = {name: sha256(path) for name, path in paths.items()}
    evidence_hashes["pr7_validation_tile"] = sha256(comparator_path)
    payload, tables = choose_validation_promotions(
        config=config,
        config_sha256=sha256(args.config),
        fit_bundle_sha256=sha256(bundle_path),
        factorized=factorized,
        response=response,
        candidate=candidate,
        tile_comparator=tile,
        evidence_sha256=evidence_hashes,
    )
    promotion_path = args.output / "neuron_selector_promotions.json"
    atomic_json(promotion_path, payload)
    for name, table in tables.items():
        table.to_csv(args.output / f"{name}.csv", index=False)

    diagnostic_summary = diagnostics.groupby(
        ["training_cohort", "basis_variant", "rank", "synthesis_encoding"],
        sort=True,
    ).agg(
        n=("hidden_q4_relative_mse", "size"),
        hidden_q4_relative_mse_median=("hidden_q4_relative_mse", "median"),
        hidden_q4_relative_mse_p90=("hidden_q4_relative_mse", lambda value: value.quantile(0.9)),
        unit_score_ndcg_192_median=("unit_score_ndcg_192", "median"),
        utility_weighted_recall_192_median=("utility_weighted_recall_192", "median"),
    ).reset_index()
    diagnostic_summary.to_csv(args.output / "response_diagnostic_summary.csv", index=False)
    factorized_summary = payload["factorized_oracle"]
    response_rows = pd.DataFrame(payload["response_predictor"]["all_validation_candidates"])
    display_columns = [
        "training_cohort", "basis_variant", "rank", "synthesis_encoding",
        "candidate_recovery_p10", "candidate_recovery_median",
        "candidate_utility_p10", "candidate_utility_median",
        "direct_recovery_p10", "direct_recovery_median",
        "selector_metadata_bpw", "passes_candidate", "passes_direct",
    ]
    report = f"""# Neuron selector distillation: validation decision

This document contains validation-only evidence. No held-out test row was
loaded by the promotion path. Recovery is exact sequential complete-expert
qenergy recovery, not accuracy, logits, routing quality or token quality.

## Factorized G/D neuron oracle

- validation overlap with the frozen PR #7 exact 64x16 tile: {factorized_summary['n']} invocations;
- factorized p10 / median at one physical bpw: {factorized_summary['p10']:.6f} / {factorized_summary['median']:.6f};
- tile p10 / median: {factorized_summary['tile_p10']:.6f} / {factorized_summary['tile_median']:.6f};
- gap to tile, p10 / median: {factorized_summary['p10_gap_to_tile']:.6f} / {factorized_summary['median_gap_to_tile']:.6f};
- decision: **{payload['factorized_oracle']['status'].upper()}**.

## Residual-response predictor candidates

{markdown_table(response_rows[display_columns] if not response_rows.empty else response_rows)}

Decision: **{payload['response_predictor']['status'].upper()}**. Conditional
follow-up: {payload['conditional_followup']['reason']}.

The sampled per-expert basis is an upper bound and is never promoted as a
bank-wide deployable selector from this small expert sample.

## Provenance

- config SHA-256: `{sha256(args.config)}`
- fit bundle-manifest SHA-256: `{sha256(bundle_path)}`
- promotions SHA-256: `{sha256(promotion_path)}`
- test rows consulted for selection: `false`
"""
    report_path = args.output / "NEURON_SELECTOR_VALIDATION_REPORT.md"
    atomic_text(report_path, report)
    outputs = [promotion_path, *sorted(args.output.glob("*.csv")), report_path]
    manifest = {
        "schema_version": 1,
        "stage": "validation_promotion",
        "inputs": {
            str(args.config): sha256(args.config),
            str(bundle_path): sha256(bundle_path),
            **{str(path): sha256(path) for path in bundle_shards},
            **{str(path): sha256(path) for path in paths.values()},
            str(comparator_path): sha256(comparator_path),
        },
        "outputs": {path.name: sha256(path) for path in outputs},
        "test_rows_consulted_for_selection": False,
    }
    atomic_json(args.output / "validation_analysis_manifest.json", manifest)


def _relative_key(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(EXPERIMENT.resolve()))
    except ValueError:
        return str(path)


def final_validation_report(args: argparse.Namespace) -> None:
    """Materialize the stopped-study report without loading any test row."""
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    validation_facts = validate_facts(args.validation_dir / "run_facts.json", split="validation")
    bundle_path = args.fit_dir / "response_fit_bundle_manifest.json"
    payload_path = args.validation_analysis_dir / "neuron_selector_promotions.json"
    payload = json.loads(payload_path.read_text())
    validate_promotion_payload(
        payload, config, config_sha256=sha256(args.config),
        fit_bundle_sha256=sha256(bundle_path),
    )
    if payload["factorized_oracle"]["status"] != "stop" or payload["response_predictor"]["status"] != "stop":
        raise RuntimeError("this final stage is only for the frozen stopped-study outcome")
    if validation_facts.get("validation_promotions_sha256") is not None:
        raise RuntimeError("validation evaluation was unexpectedly promotion-filtered")

    raw_paths = {
        name: args.validation_dir / filename for name, filename in {
            "factorized": "factorized_neuron_frontier.parquet",
            "response": "response_predictor_frontier.parquet",
            "candidate": "candidate_rerank_frontier.parquet",
            "diagnostics": "response_prediction_diagnostics.parquet",
        }.items()
    }
    factorized = evidence_frame(raw_paths["factorized"], split="validation")
    response = evidence_frame(raw_paths["response"], split="validation")
    candidate = evidence_frame(raw_paths["candidate"], split="validation")
    diagnostics = evidence_frame(raw_paths["diagnostics"], split="validation")
    if len(response) == 0 or len(candidate) == 0:
        raise RuntimeError("response evidence is unexpectedly empty")

    tile_record = config["pr7_comparators"]["validation_tile_frontier"]
    neuron_record = config["pr7_comparators"]["validation_neuron_frontier"]
    tile_path, neuron_path = EXPERIMENT / tile_record["path"], EXPERIMENT / neuron_record["path"]
    for path, record in ((tile_path, tile_record), (neuron_path, neuron_record)):
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"frozen PR7 comparator changed: {path}")
    tile = pd.read_parquet(tile_path, filters=[("evaluation_split", "==", "validation")])
    tile = tile[(tile["selector"] == tile_record["selector"]) & (tile["tile_shape"].astype(str) == tile_record["tile_shape"])].copy()
    neuron = pd.read_parquet(neuron_path, filters=[("evaluation_split", "==", "validation")])
    neuron = neuron[neuron["selector"] == neuron_record["selector"]].copy()
    if set(tile["evaluation_split"].astype(str)) != {"validation"} or set(neuron["evaluation_split"].astype(str)) != {"validation"}:
        raise RuntimeError("a PR7 comparator admitted a non-validation row")

    identity = ["capture_source", "evaluation_split", "request_id", "position", "layer", "expert_id"]
    current_matched = factorized.merge(tile[identity].drop_duplicates(), on=identity, how="inner", validate="many_to_one")
    frames = []
    for label, local in (
        ("coherent complete-unit packet", current_matched[current_matched["action_family"] == "coherent_complete_unit_packet"]),
        ("factorized G/D fixed-greedy", current_matched[current_matched["action_family"] == "factorized_gate_up_down_unit_states"]),
        ("exact 64x16 tile", tile),
        ("PR7 independent-unit score", neuron),
    ):
        value = local.copy(); value["representation_label"] = label; frames.append(value)
    representations = pd.concat(frames, ignore_index=True, sort=False)

    summaries = {
        "factorized_frontier_summary.csv": frontier_summary(factorized, ["action_family", "physical_budget_bpw"]),
        "matched_representation_frontier.csv": frontier_summary(representations, ["representation_label", "physical_budget_bpw"]),
        "matched_representation_by_layer.csv": frontier_summary(representations, ["representation_label", "layer", "physical_budget_bpw"]),
        "response_predictor_validation_summary.csv": pd.DataFrame(payload["response_predictor"]["all_validation_candidates"]),
        "response_diagnostic_summary.csv": diagnostics.groupby(
            ["training_cohort", "basis_variant", "rank", "synthesis_encoding"], sort=True,
        ).agg(
            n=("hidden_q4_relative_mse", "size"),
            delta_gate_relative_mse_median=("delta_gate_relative_mse", "median"),
            delta_up_relative_mse_median=("delta_up_relative_mse", "median"),
            hidden_q4_relative_mse_p10=("hidden_q4_relative_mse", lambda x: x.quantile(.1)),
            hidden_q4_relative_mse_median=("hidden_q4_relative_mse", "median"),
            hidden_q4_relative_mse_p90=("hidden_q4_relative_mse", lambda x: x.quantile(.9)),
            unit_score_ndcg_192_median=("unit_score_ndcg_192", "median"),
            utility_weighted_recall_192_p10=("utility_weighted_recall_192", lambda x: x.quantile(.1)),
            utility_weighted_recall_192_median=("utility_weighted_recall_192", "median"),
            predictor_runtime_ms_median=("predictor_runtime_ms", "median"),
        ).reset_index(),
    }
    output_paths = []
    for name, frame in summaries.items():
        path = args.output / name; frame.to_csv(path, index=False); output_paths.append(path)

    response_summary = summaries["response_predictor_validation_summary.csv"]
    deployable = response_summary[
        (~response_summary["sampled_per_expert_upper_bound"].astype(bool))
        & (pd.to_numeric(response_summary["selector_metadata_bpw"]) <= float(config["promotion_policy"]["max_selector_metadata_bpw"]))
    ]
    best_recovery = deployable.sort_values(["candidate_recovery_median", "candidate_recovery_p10"], ascending=False).iloc[0]
    best_utility = deployable.sort_values(["candidate_utility_median", "candidate_utility_p10"], ascending=False).iloc[0]
    factor_one = factorized[(factorized["action_family"] == "factorized_gate_up_down_unit_states") & np.isclose(pd.to_numeric(factorized["physical_budget_bpw"]), 1.0)]
    coherent_one = factorized[(factorized["action_family"] == "coherent_complete_unit_packet") & np.isclose(pd.to_numeric(factorized["physical_budget_bpw"]), 1.0)]
    abc_bytes = 3 * (512 * 2 + 4)
    accounting = {
        "page_size_bytes": int(config["page_size_bytes"]),
        "expert_weights": int(config["expert_weights"]),
        "abc_fp16_metadata_bytes_per_expert": abc_bytes,
        "abc_fp16_metadata_bpw": 8.0 * abc_bytes / int(config["expert_weights"]),
        "factorized_one_bpw": {
            "physical_pages_p10_median_p90": [float(x) for x in np.quantile(factor_one["physical_pages"], [.1, .5, .9])],
            "gate_up_actions_median": float(factor_one["gate_up_actions"].median()),
            "down_actions_median": float(factor_one["down_actions"].median()),
            "logical_actions_median": float(factor_one["logical_actions"].median()),
            "selector_runtime_ms_median": float(factor_one["selector_runtime_ms"].median()),
            "selector_bytes_read_median": float(factor_one["selector_bytes_read"].median()),
            "page_amplification_max": float(factor_one["page_amplification"].max()),
            "storage_multiplier_max": float(factor_one["storage_multiplier"].max()),
        },
        "coherent_one_bpw": {
            "physical_pages": int(coherent_one["physical_pages"].median()),
            "units": int(coherent_one["logical_actions"].median()),
            "selector_runtime_ms_median": float(coherent_one["selector_runtime_ms"].median()),
        },
        "best_deployable_candidate_by_recovery": best_recovery.to_dict(),
        "best_deployable_candidate_by_utility": best_utility.to_dict(),
        "test_rows_consulted": False,
    }
    accounting_path = args.output / "selector_compute_storage_accounting.json"
    atomic_json(accounting_path, accounting); output_paths.append(accounting_path)

    frontier = summaries["matched_representation_frontier.csv"]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for label, local in frontier.groupby("representation_label", sort=True):
        local = local.sort_values("physical_budget_bpw")
        ax.plot(local["physical_budget_bpw"], local["recovery_median"], marker="o", label=label)
        ax.fill_between(local["physical_budget_bpw"], local["recovery_p10"], local["recovery_p90"], alpha=.12)
    ax.set(xlabel="physical correction bpw", ylabel="complete-expert qenergy recovery", title="Validation representation frontiers (matched 55 invocations)")
    ax.grid(alpha=.25); ax.legend(fontsize=8)
    output_paths += save_figure(fig, args.output, "01_representation_frontier")

    action_plot = summaries["factorized_frontier_summary.csv"]
    action_plot = action_plot[action_plot["action_family"] == "factorized_gate_up_down_unit_states"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(action_plot["physical_budget_bpw"], action_plot["gate_up_actions_median"], marker="o", label="gate/up")
    ax.plot(action_plot["physical_budget_bpw"], action_plot["down_actions_median"], marker="o", label="down")
    ax.set(xlabel="physical correction bpw cap", ylabel="median applied transitions", title="Factorized neuron action mix")
    ax.grid(alpha=.25); ax.legend(); output_paths += save_figure(fig, args.output, "02_factorized_action_mix")

    diagnostic_plot = summaries["response_diagnostic_summary.csv"]
    diagnostic_plot = diagnostic_plot[(diagnostic_plot["training_cohort"] == "combined_exact_and_cross_train") & (diagnostic_plot["synthesis_encoding"] == "fp16")]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for label, local in diagnostic_plot.groupby("basis_variant", sort=True):
        local = local.sort_values("rank")
        ax.plot(local["rank"], local["hidden_q4_relative_mse_median"], marker="o", label=label.replace("_", " "))
    ax.set(xlabel="response rank", ylabel="median relative MSE of predicted h4", title="Residual-response prediction remains lossy")
    ax.set_xscale("log", base=2); ax.grid(alpha=.25); ax.legend(fontsize=8)
    output_paths += save_figure(fig, args.output, "03_hidden_response_prediction")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(deployable["selector_metadata_bpw"], deployable["candidate_recovery_median"], s=26, alpha=.75, label="median")
    ax.scatter(deployable["selector_metadata_bpw"], deployable["candidate_recovery_p10"], s=26, alpha=.75, label="p10")
    ax.axhline(float(config["promotion_policy"]["candidate_min_median_recovery"]), color="C0", linestyle="--", alpha=.6)
    ax.axhline(float(config["promotion_policy"]["candidate_min_p10_recovery"]), color="C1", linestyle="--", alpha=.6)
    ax.axvline(float(config["promotion_policy"]["max_selector_metadata_bpw"]), color="black", linestyle=":")
    ax.set(xlabel="selector metadata bpw", ylabel="candidate rerank recovery", title="No deployable residual-response candidate passes")
    ax.grid(alpha=.25); ax.legend(); output_paths += save_figure(fig, args.output, "04_predictor_recovery_metadata")

    matched = summaries["matched_representation_frontier.csv"]
    matched_one = matched[np.isclose(pd.to_numeric(matched["physical_budget_bpw"]), 1.0)]
    matched_core = matched[
        pd.to_numeric(matched["physical_budget_bpw"]).isin([0.5, 0.75, 1.0])
    ]
    layer_one = summaries["matched_representation_by_layer.csv"]
    layer_one = layer_one[np.isclose(pd.to_numeric(layer_one["physical_budget_bpw"]), 1.0)]
    top = response_summary.sort_values(["candidate_recovery_median", "candidate_recovery_p10"], ascending=False).head(10)
    report = f"""# Neuron-selector distillation report

## Outcome

Both predeclared workstreams stop on exact-checkpoint validation. Factorized
gate/up-versus-down actions help at 0.5 bpw but do not close the nonlinear tile
gap at 1.0 bpw. Compact residual-response models recover much independent
unit-score utility, yet fail complete-expert recovery. No test or cross-
reference evaluation was launched.

This is exact sequential complete-expert qenergy recovery, not task accuracy,
token quality, logits, or routing quality. Every studied predictor is H0-late
and requires a second pass; no H4 predictor was trained.

## Workstream A: factorized G/D neuron actions

### Matched frontier

{markdown_table(matched_core[["representation_label", "physical_budget_bpw", "recovery_n", "recovery_p10", "recovery_median", "recovery_p90", "physical_pages_median"]])}

At 0.5 bpw the factorized path materially improves on coherent packets, but
the advantage disappears by 0.75 bpw and reverses at 1.0 bpw.

### One-bpw accounting

{markdown_table(matched_one[["representation_label", "recovery_n", "recovery_p10", "recovery_median", "recovery_p90", "physical_pages_median", "logical_actions_median", "selector_runtime_ms_median"]])}

Decision: **STOP**. Factorized p10/median was
{payload['factorized_oracle']['p10']:.6f}/{payload['factorized_oracle']['median']:.6f},
trailing the frozen 64x16 tile by
{payload['factorized_oracle']['p10_gap_to_tile']:.6f}/{payload['factorized_oracle']['median_gap_to_tile']:.6f}.
The fixed-greedy path is not a global subset optimum; it retains the best
cumulative prefix under each paid-page cap.

### One-bpw layer breakdown (matched validation cohort)

{markdown_table(layer_one[["representation_label", "layer", "recovery_n", "recovery_p10", "recovery_median", "physical_pages_median"]])}

## Workstream B: predict h4 and score with A/B/C

The exact scalar statistic is compact: three normalized FP16 vectors plus
FP32 scales require {abc_bytes:,} bytes/expert
({accounting['abc_fp16_metadata_bpw']:.6f} bpw). Ranks 8/16/32/64/128, joint
and separate layer bases, two training cohorts, FP16/row-FP8/row-INT8, and a
sampled per-expert upper bound were evaluated.

Best deployable candidate by recovery (metadata <= 0.35 bpw):

{markdown_table(pd.DataFrame([best_recovery])[["training_cohort", "basis_variant", "rank", "synthesis_encoding", "candidate_recovery_p10", "candidate_recovery_median", "candidate_utility_p10", "candidate_utility_median", "selector_metadata_bpw", "selector_compute_macs"]])}

Decision: **STOP**. Even the sampled per-expert upper bound failed, so neither
the activation-top-k nor cluster-basis continuation was run.

### Ten strongest candidate-rerank configurations

{markdown_table(top[["training_cohort", "basis_variant", "rank", "synthesis_encoding", "candidate_recovery_p10", "candidate_recovery_median", "candidate_utility_p10", "candidate_utility_median", "selector_metadata_bpw", "sampled_per_expert_upper_bound"]])}

## Interpretation

PR #7's neuron sparsity remains real, but independently scoring units from
predicted h4 is insufficient. Factorization is useful only at low rates; its
partially refined states make the one-bpw greedy endgame worse than coherent
packets. The next justified target is marginal complete-expert utility or
support templates on a much larger exact-checkpoint corpus—not another rank
or activation-top-k sweep on these validation requests.

## Controls and provenance

- fit used complete training requests only;
- selection used 69 exact-checkpoint validation invocations;
- frozen matched PR #7 comparison used 55 validation invocations;
- test rows consulted: **false**; held-out evaluation launched: **false**;
- codec, selected trees, checkpoint revision and request separation unchanged;
- config: `{sha256(args.config)}`;
- fit bundle manifest: `{sha256(bundle_path)}`;
- frozen validation promotion: `{sha256(payload_path)}`.
"""
    report_path = args.output / "NEURON_SELECTOR_DISTILLATION_REPORT.md"
    atomic_text(report_path, report); output_paths.append(report_path)

    input_paths = [args.config, bundle_path, args.fit_dir / "response_fit_index.json", args.fit_dir / "fit_facts.json", args.validation_dir / "run_facts.json", payload_path, args.validation_analysis_dir / "validation_analysis_manifest.json", *raw_paths.values(), tile_path, neuron_path]
    bundle_manifest = json.loads(bundle_path.read_text())
    input_paths.extend(args.fit_dir / record["path"] for record in bundle_manifest["layers"].values())
    manifest = {
        "schema_version": 1, "stage": "final_validation_negative",
        "inputs": {_relative_key(path): sha256(path) for path in input_paths},
        "outputs": {path.name: sha256(path) for path in output_paths},
        "promotion_sha256": sha256(payload_path), "test_rows_consulted": False,
        "heldout_evaluation_launched": False,
    }
    atomic_json(args.output / "analysis_manifest.json", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("promote", "final"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--validation-analysis-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "promote":
        promote(args)
    else:
        if args.validation_analysis_dir is None:
            raise RuntimeError("final stage requires --validation-analysis-dir")
        final_validation_report(args)


if __name__ == "__main__":
    main()
