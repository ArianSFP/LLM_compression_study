#!/usr/bin/env python3
"""Validate, summarize, and hash the exact-H4 interaction-field evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "src"))

from oracle_study.interaction_field_analysis import (  # noqa: E402
    sha256,
    summarize,
    validate_evidence,
)


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def _atomic_json(path: Path, value) -> None:
    _atomic_text(path, json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _report(conclusion, factor, solver, layer) -> str:
    best = conclusion["best_exact_h4_geometry_metrics"]
    quantized = conclusion["best_quantized_metrics"]
    cheap = conclusion["best_two_seed_metrics"]
    cheap_quantized = conclusion["best_quantized_two_seed_metrics"]
    full_gate = conclusion["best_quantized_full_gate_metrics"]
    full_gate_summary = (
        "- No quantized two-seed configuration passed every frozen gate."
        if full_gate is None else
        f"- Best all-gates quantized factor: "
        f"`{conclusion['best_quantized_full_gate_configuration']}`; recovery p10 "
        f"{full_gate['recovery_p10']:.4%}, median {full_gate['recovery_median']:.4%}; "
        f"retention p10 {full_gate['retention_p10']:.4%}; metadata "
        f"{full_gate['combined_metadata_bpw']:.5f} bpw; compute "
        f"{full_gate['selector_compute_macs_max']:.0f} MACs."
    )
    return f"""# Low-rank signed interaction-field study

This is an exact-H4 algebraic geometry ceiling, not a deployable candidate
predictor, prefetch, downstream-routing, logit, or token-quality result.

## Outcome

- Best geometry: `{conclusion['best_exact_h4_geometry_configuration']}`.
- One-bpw recovery: p10 {best['recovery_p10']:.4%}, median {best['recovery_median']:.4%}.
- One-bpw set-gain retention versus the PR #10 exact hybrid teacher: p10
  {best['retention_p10']:.4%}, median {best['retention_median']:.4%}.
- Best physically quantized factor: `{conclusion['best_quantized_configuration']}`.
- Quantized one-bpw recovery: p10 {quantized['recovery_p10']:.4%}, median
  {quantized['recovery_median']:.4%}; retention p10 {quantized['retention_p10']:.4%}.
- Best two-seed factor: `{conclusion['best_two_seed_configuration']}`.
- Two-seed one-bpw recovery: p10 {cheap['recovery_p10']:.4%}, median
  {cheap['recovery_median']:.4%}; retention p10 {cheap['retention_p10']:.4%};
  compute {cheap['selector_compute_macs_max']:.0f} MACs.
- Best quantized two-seed factor:
  `{conclusion['best_quantized_two_seed_configuration']}`.
- Quantized two-seed recovery: p10 {cheap_quantized['recovery_p10']:.4%},
  median {cheap_quantized['recovery_median']:.4%}; retention p10
  {cheap_quantized['retention_p10']:.4%}; compute
  {cheap_quantized['selector_compute_macs_max']:.0f} MACs.
{full_gate_summary}
- Quantized two-seed geometry gate passed:
  `{conclusion['any_quantized_two_seed_geometry_gate_pass']}`.
- Geometry gate passed: `{conclusion['any_geometry_gate_pass']}`.
- Quantized geometry gate passed: `{conclusion['any_quantized_geometry_gate_pass']}`.
- Quantized two-seed geometry+metadata+compute gate passed:
  `{conclusion['any_quantized_full_compute_and_metadata_pass']}`.
- Status: `{conclusion['deployment_status']}`.

The primary compute candidate uses only all-00 and independent-A/B/C seeds.
The broader geometry diagnostic additionally charges residual-forward and the
continuous relaxation. Both receive identical coordinate and local repair;
neither may borrow the other's recovery or accounting result.

## Artifact rows

- Factor/budget summary rows: {len(factor)}
- Solver-control rows: {len(solver)}
- Layer-summary rows: {len(layer)}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runner = EXPERIMENT / "scripts/run_interaction_field_study.py"
    core = EXPERIMENT / "src/oracle_study/interaction_field.py"
    analysis_core = EXPERIMENT / "src/oracle_study/interaction_field_analysis.py"
    evidence = validate_evidence(
        args.config, args.evidence_dir, runner_path=runner, core_path=core,
    )
    factor, solver, layer, conclusion = summarize(evidence)
    files = {
        "factor_summary": args.output / "interaction_factor_summary.csv",
        "solver_summary": args.output / "interaction_solver_summary.csv",
        "layer_summary": args.output / "interaction_layer_summary.csv",
        "conclusion": args.output / "interaction_field_conclusion.json",
        "report": args.output / "INTERACTION_FIELD_REPORT.md",
    }
    factor.to_csv(files["factor_summary"], index=False)
    solver.to_csv(files["solver_summary"], index=False)
    layer.to_csv(files["layer_summary"], index=False)
    _atomic_json(files["conclusion"], conclusion)
    _atomic_text(files["report"], _report(conclusion, factor, solver, layer))
    manifest = {
        "schema_version": 1,
        "inputs": {
            "config": {"path": args.config.name, "sha256": sha256(args.config)},
            "run_facts": {
                "path": "run_facts.json",
                "sha256": sha256(args.evidence_dir / "run_facts.json"),
            },
            "factor_manifest": {
                "path": "interaction_field_factor_manifest.json",
                "sha256": sha256(args.evidence_dir / "interaction_field_factor_manifest.json"),
            },
            "frontier": {
                "path": "interaction_field_frontier.parquet",
                "sha256": sha256(args.evidence_dir / "interaction_field_frontier.parquet"),
            },
        },
        "sources": {
            "runner": {"path": "scripts/run_interaction_field_study.py", "sha256": sha256(runner)},
            "interaction_core": {"path": "src/oracle_study/interaction_field.py", "sha256": sha256(core)},
            "analysis_core": {
                "path": "src/oracle_study/interaction_field_analysis.py",
                "sha256": sha256(analysis_core),
            },
            "wrapper": {
                "path": "scripts/analyze_interaction_field_study.py",
                "sha256": sha256(Path(__file__).resolve()),
            },
        },
        "outputs": {
            name: {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in files.items()
        },
        "conclusion": conclusion,
    }
    _atomic_json(args.output / "analysis_manifest.json", manifest)


if __name__ == "__main__":
    main()
