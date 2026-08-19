#!/usr/bin/env python3
"""Analyze validation evidence and freeze neuron-selector promotions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "src"))

from oracle_study.neuron_selector_analysis import (
    choose_validation_promotions,
    evidence_frame,
    sha256,
)


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


def promote(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    fit_facts = json.loads((args.fit_dir / "fit_facts.json").read_text())
    validation_facts = validate_facts(args.validation_dir / "run_facts.json", split="validation")
    bundle_path = args.fit_dir / "response_fit_bundle.npz"
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
- fit bundle SHA-256: `{sha256(bundle_path)}`
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
            **{str(path): sha256(path) for path in paths.values()},
            str(comparator_path): sha256(comparator_path),
        },
        "outputs": {path.name: sha256(path) for path in outputs},
        "test_rows_consulted_for_selection": False,
    }
    atomic_json(args.output / "validation_analysis_manifest.json", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("promote",), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    promote(args)


if __name__ == "__main__":
    main()
