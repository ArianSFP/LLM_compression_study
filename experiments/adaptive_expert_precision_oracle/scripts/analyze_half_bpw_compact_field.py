#!/usr/bin/env python3
"""Validate and summarize the bandwidth-bounded compact joint field."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(EXPERIMENT / "scripts")]
import run_half_bpw_compact_field as runner


SUMMARY = "compact_half_bpw_accuracy.csv"
LAYER = "compact_half_bpw_layer_accuracy.csv"
DECISION = "compact_half_bpw_decision.json"
REPORT = "COMPACT_HALF_BPW_REPORT.md"
MANIFEST = "compact_half_bpw_analysis_manifest.json"
PLOT_PNG = "compact_half_bpw_recovery.png"
PLOT_SVG = "compact_half_bpw_recovery.svg"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _quantiles(values: pd.Series) -> tuple[float, float, float]:
    numeric = values.to_numpy(np.float64)
    if numeric.size == 0 or not np.all(np.isfinite(numeric)):
        raise RuntimeError("summary input is empty or non-finite")
    return tuple(float(np.quantile(numeric, q)) for q in (.1, .5, .9))


def _summaries(frame: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    rows = []
    for key, local in frame.groupby(group, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        p10, median, p90 = _quantiles(local["group_recovery"])
        rows.append({
            **dict(zip(group, key)),
            "groups": len(local),
            "p10_recovery": p10,
            "median_recovery": median,
            "p90_recovery": p90,
            "median_damage_ratio_vs_pr13": float(np.median(local["damage_ratio_vs_pr13"])),
            "median_state_agreement_with_exact_teacher": float(np.median(local["state_agreement_with_exact_teacher"])),
            "median_actual_streamed_bpw": float(np.median(local["average_actual_streamed_bpw"])),
            "allowed_total_bpw": float(local["allowed_total_bpw"].iloc[0]),
            "factor_payload_bytes_per_expert": int(local["factor_payload_bytes_per_expert"].iloc[0]),
            "combined_metadata_bytes_per_expert": int(local["combined_metadata_bytes_per_expert"].iloc[0]),
            "median_logical_group_bytes_read": int(np.median(local["logical_group_bytes_read"])),
            "median_selector_compute_macs": int(np.median(local["selector_compute_macs"])),
            "median_seed_state_evaluations": int(np.median(local["selector_seed_state_evaluations"])),
            "median_seed_backpointer_bytes": int(np.median(local["selector_seed_backpointer_bytes"])),
            "median_selector_wall_seconds": float(np.median(local["selector_wall_seconds"])),
            "median_coordinate_sweeps": float(np.median(local["coordinate_sweeps"])),
            "median_local_passes": float(np.median(local["local_passes"])),
        })
    return pd.DataFrame(rows)


def _validate(
    config: Mapping[str, Any], config_path: Path, fit_dir: Path,
    validation_dir: Path,
) -> pd.DataFrame:
    runner._validate_config(config)
    required = [
        runner.RUN_FACTS, runner.GROUP_FRONTIER, runner.STATE_EVIDENCE,
        runner.ACCOUNTING,
    ]
    if {path.name for path in validation_dir.iterdir()} != set(required):
        raise RuntimeError("compact validation file set changed")
    facts = json.loads((validation_dir / runner.RUN_FACTS).read_text())
    accounting = json.loads((validation_dir / runner.ACCOUNTING).read_text())
    fit_facts = json.loads((fit_dir / runner.FIT_FACTS).read_text())
    manifest = json.loads((fit_dir / runner.FIT_MANIFEST).read_text())
    if any(record.get("completed") is not True for record in (facts, fit_facts, manifest)):
        raise RuntimeError("compact evidence is incomplete")
    if facts.get("failures") or facts.get("failure_history") or fit_facts.get("failures"):
        raise RuntimeError("compact evidence records a failure")
    if facts.get("config_sha256") != _sha256(config_path):
        raise RuntimeError("compact config hash changed")
    dependencies = runner._dependencies()
    for name, value in dependencies.items():
        if facts.get(name) != value or fit_facts.get(name) != value:
            raise RuntimeError(f"compact source dependency changed: {name}")
    if facts.get("fit_facts_sha256") != _sha256(fit_dir / runner.FIT_FACTS):
        raise RuntimeError("compact fit facts hash changed")
    if facts.get("fit_manifest_sha256") != _sha256(fit_dir / runner.FIT_MANIFEST):
        raise RuntimeError("compact fit manifest hash changed")
    for name, filename in (
        ("group_frontier_sha256", runner.GROUP_FRONTIER),
        ("state_evidence_sha256", runner.STATE_EVIDENCE),
        ("accounting_sha256", runner.ACCOUNTING),
    ):
        if facts.get(name) != _sha256(validation_dir / filename):
            raise RuntimeError(f"compact artifact hash changed: {filename}")
    if int(accounting.get("schema_version", -1)) != 2:
        raise RuntimeError("compact seed-accounting schema changed")
    if accounting.get("policies") != list(runner.POLICIES):
        raise RuntimeError("compact accounting policy set changed")
    if int(accounting.get("primary_group_sidecar_bytes", -1)) != 73856:
        raise RuntimeError("primary group sidecar changed")
    frame = pd.read_parquet(validation_dir / runner.GROUP_FRONTIER)
    states = pd.read_parquet(validation_dir / runner.STATE_EVIDENCE)
    expected = int(config["expected_validation_groups"]) * len(runner.POLICIES)
    if len(frame) != expected or len(states) != expected:
        raise RuntimeError("compact row count changed")
    identity = list(runner.GROUP_IDENTITY) + ["allocation_policy"]
    if frame[identity].duplicated().any() or states[identity].duplicated().any():
        raise RuntimeError("compact identity grid contains duplicates")
    if set(frame["allocation_policy"].astype(str)) != set(runner.POLICIES):
        raise RuntimeError("compact policy coverage changed")
    numeric = [
        "streamed_bpw", "actual_group_pages", "average_actual_streamed_bpw",
        "factor_payload_bytes_per_expert", "combined_metadata_bytes_per_expert",
        "metadata_bpw", "allowed_total_bpw", "logical_group_bytes_read",
        "group_base_qenergy_damage", "group_exact_qenergy_damage",
        "group_recovery", "damage_ratio_vs_pr13",
        "state_agreement_with_exact_teacher", "selector_compute_macs",
        "selector_seed_state_evaluations", "selector_seed_backpointer_bytes",
        "selector_wall_seconds", "coordinate_sweeps", "local_passes",
    ]
    if not np.all(np.isfinite(frame[numeric].to_numpy(np.float64))):
        raise RuntimeError("compact evidence contains non-finite accounting or metrics")
    if np.any(frame["streamed_bpw"] != 0.5) or np.any(frame["actual_group_pages"] > 3072):
        raise RuntimeError("compact streamed budget changed")
    expected_bytes = (
        frame["actual_group_pages"].astype(np.int64) * 512
        + 8 * frame["combined_metadata_bytes_per_expert"].astype(np.int64)
    )
    if not np.array_equal(expected_bytes, frame["logical_group_bytes_read"].astype(np.int64)):
        raise RuntimeError("compact logical byte accounting changed")
    expected_total = (
        0.5
        + 8.0 * frame["combined_metadata_bytes_per_expert"].to_numpy(np.float64)
        / 3_145_728.0
    )
    if not np.allclose(expected_total, frame["allowed_total_bpw"], rtol=0.0, atol=1e-12):
        raise RuntimeError("compact total-bpw accounting changed")
    primary = frame[frame["allocation_policy"] == runner.POLICIES[1]]
    if len(primary) != 128:
        raise RuntimeError("compact primary row count changed")
    if set(primary["factor_payload_bytes_per_expert"].astype(int)) != {6148}:
        raise RuntimeError("compact primary factor payload changed")
    if set(primary["combined_metadata_bytes_per_expert"].astype(int)) != {9232}:
        raise RuntimeError("compact primary combined metadata changed")
    lagrangian = frame[frame["allocation_policy"].isin(
        [runner.POLICIES[1], runner.POLICIES[3], runner.POLICIES[4]]
    )]
    dp_control = frame[frame["allocation_policy"] == runner.POLICIES[2]]
    frozen = frame[frame["allocation_policy"].isin(
        [runner.POLICIES[0], runner.POLICIES[5]]
    )]
    if frame["validation_oracle_seed_used"].astype(bool).any():
        raise RuntimeError("compact policy admitted a validation-oracle seed")
    if set(lagrangian["seed_name"].astype(str)) != {"global_self_lagrangian"}:
        raise RuntimeError("efficient compact seed identity changed")
    if np.any(lagrangian["selector_seed_backpointer_bytes"] != 0):
        raise RuntimeError("efficient compact seed allocated a budget-sized table")
    if np.any(lagrangian["selector_seed_state_evaluations"] <= 0):
        raise RuntimeError("efficient compact seed work is missing")
    expected_dp_evaluations = 8 * 4096 * (3072 + 1)
    expected_dp_bytes = 4096 * (3072 + 1)
    if set(dp_control["seed_name"].astype(str)) != {"global_exact_self_dp"}:
        raise RuntimeError("DP control seed identity changed")
    if set(dp_control["selector_seed_state_evaluations"].astype(int)) != {
        expected_dp_evaluations
    }:
        raise RuntimeError("DP control work accounting changed")
    if set(dp_control["selector_seed_backpointer_bytes"].astype(int)) != {
        expected_dp_bytes
    }:
        raise RuntimeError("DP control memory accounting changed")
    if np.any(frozen[["selector_seed_state_evaluations", "selector_seed_backpointer_bytes"]] != 0):
        raise RuntimeError("frozen controls were assigned seed work")
    return frame


def analyze(config_path: Path, fit_dir: Path, validation_dir: Path, output: Path) -> None:
    config = json.loads(config_path.read_text())
    frame = _validate(config, config_path, fit_dir, validation_dir)
    output.mkdir(parents=True, exist_ok=False)
    summary = _summaries(frame, ["allocation_policy"])
    layer = _summaries(frame, ["allocation_policy", "layer"])
    summary.to_csv(output / SUMMARY, index=False)
    layer.to_csv(output / LAYER, index=False)
    indexed = summary.set_index("allocation_policy")
    primary = indexed.loc[runner.POLICIES[1]]
    dp_control = indexed.loc[runner.POLICIES[2]]
    fp32 = indexed.loc[runner.POLICIES[3]]
    rank16 = indexed.loc[runner.POLICIES[4]]
    exact = indexed.loc[runner.POLICIES[5]]
    target_p10 = float(config["primary_target_p10_recovery"])
    target_median = float(config["primary_target_median_recovery"])
    accuracy_pass = bool(
        primary.p10_recovery >= target_p10
        and primary.median_recovery >= target_median
    )
    bandwidth_pass = bool(
        int(primary.combined_metadata_bytes_per_expert) * 8
        <= int(config["maximum_primary_group_sidecar_bytes"])
    )
    compute_pass = bool(
        int(primary.median_selector_compute_macs)
        <= int(config["maximum_primary_median_selector_macs"])
    )
    if accuracy_pass and bandwidth_pass and compute_pass:
        status = "compact_field_pass"
        next_action = "freeze_and_measure_a_fused_runtime_kernel"
    elif (
        dp_control.p10_recovery >= target_p10
        and dp_control.median_recovery >= target_median
    ):
        status = "seed_efficiency_gap"
        next_action = "improve_the_linear_memory_seed_without_restoring_the_DP_table"
    elif fp32.p10_recovery >= target_p10 and fp32.median_recovery >= target_median:
        status = "quantization_gap"
        next_action = "improve_same_payload_factor_encoding"
    elif rank16.p10_recovery >= target_p10 and rank16.median_recovery >= target_median:
        status = "rank_gap"
        next_action = "trade_rank_for_bits_under_the_same_6148_byte_payload"
    elif exact.p10_recovery >= target_p10:
        status = "compact_geometry_gap"
        next_action = "fit_decision_queries_not_static_gram_error"
    else:
        status = "action_gap"
        next_action = "expand_the_action_representation"
    decision = {
        "schema_version": 2,
        "status": status,
        "next_action": next_action,
        "primary_policy": runner.POLICIES[1],
        "accuracy_pass": accuracy_pass,
        "bandwidth_pass": bandwidth_pass,
        "compute_pass": compute_pass,
        "target_p10_recovery": target_p10,
        "target_median_recovery": target_median,
        "primary_p10_recovery": float(primary.p10_recovery),
        "primary_median_recovery": float(primary.median_recovery),
        "primary_group_sidecar_bytes": int(primary.combined_metadata_bytes_per_expert) * 8,
        "primary_median_logical_group_bytes_read": int(primary.median_logical_group_bytes_read),
        "primary_median_selector_compute_macs": int(primary.median_selector_compute_macs),
        "primary_median_seed_state_evaluations": int(primary.median_seed_state_evaluations),
        "primary_median_seed_backpointer_bytes": int(primary.median_seed_backpointer_bytes),
        "primary_median_selector_wall_seconds": float(primary.median_selector_wall_seconds),
        "dp_control_p10_recovery": float(dp_control.p10_recovery),
        "dp_control_median_recovery": float(dp_control.median_recovery),
        "exact_teacher_p10_recovery": float(exact.p10_recovery),
        "exact_teacher_median_recovery": float(exact.median_recovery),
        "no_test_scientific_values_used": True,
    }
    _write_json(output / DECISION, decision)

    x = np.arange(len(runner.POLICIES))
    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    ordered = indexed.loc[list(runner.POLICIES)]
    axis.plot(x, 100 * ordered["p10_recovery"], marker="o", label="p10")
    axis.plot(x, 100 * ordered["median_recovery"], marker="o", label="median")
    axis.axhline(99.0, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(x, ("PR13", "r8 efficient", "r8 DP", "r8 FP32", "r16 INT4", "exact"))
    axis.set_ylabel("Combined top-8 recovery (%)")
    axis.set_title("Shared decision field at 0.5 streamed bpw")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / PLOT_PNG, dpi=180)
    figure.savefig(output / PLOT_SVG)
    plt.close(figure)

    table = []
    for policy in runner.POLICIES:
        row = indexed.loc[policy]
        table.append(
            f"| `{policy}` | {100*row.p10_recovery:.4f}% | "
            f"{100*row.median_recovery:.4f}% | {int(row.combined_metadata_bytes_per_expert)*8:,} | "
            f"{int(row.median_logical_group_bytes_read):,} | "
            f"{int(row.median_selector_compute_macs):,} | "
            f"{int(row.median_seed_state_evaluations):,} | "
            f"{int(row.median_seed_backpointer_bytes):,} | "
            f"{1000*row.median_selector_wall_seconds:.2f} |"
        )
    report = f"""# Compact half-bpw joint field

All policies use the same 128 exact-H4 validation groups and at most 0.5
streamed correction bpw.  The primary rank-8 Hadamard-INT4 field has exactly
the immutable PR #13 payload: 6,148 factor bytes plus 3,084 A/B/C bytes per
expert, or 73,856 bytes across a routed top-8 group.  It is loaded once and
reused by the incremental coordinate/local solver.

| Policy | p10 recovery | Median recovery | Group sidecar bytes | Median logical group bytes | Median selector MACs | Median seed state evaluations | Seed backpointer bytes | Median Python wall ms |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table)}

Frozen decision: **`{status}`**.  Accuracy pass: `{accuracy_pass}`;
bandwidth pass: `{bandwidth_pass}`; compute pass: `{compute_pass}`.  The
predeclared next action is `{next_action}`.

The primary uses a scalar-price exact-self seed with O(units) state memory,
then the existing incremental coordinate and bounded local search.  It allocates
zero budget-sized backpointer bytes.  The same rank-8 INT4 field is also run
with a 12,587,008-byte exact-self DP backpointer table solely as an attribution
control.  Seed comparisons/additions are reported separately from latent MACs;
Python wall time is diagnostic and is not a fused-kernel latency claim.

The basis was fitted only from train-split exact residual and coordinate-query
trajectories.  Exact A/B/C keeps every same-unit term exact; the compact field
approximates signed cross-unit and cross-expert interactions only.  The exact
output-space policy is a frozen validation-only teacher and is not assigned a
deployable payload.  No test scientific value, predicted-H4 claim, downstream
replay, routing, logit, token, or measured production-kernel claim is made.
"""
    (output / REPORT).write_text(report)
    generated = [SUMMARY, LAYER, DECISION, REPORT, PLOT_PNG, PLOT_SVG]
    result_prefix = Path("results") / str(config["run_id"])
    source_paths = {
        "runner_sha256": "scripts/run_half_bpw_compact_field.py",
        "teacher_runner_sha256": "scripts/run_half_bpw_joint_field.py",
        "joint_core_sha256": "src/oracle_study/low_rate_joint_field.py",
        "shared_basis_core_sha256": "src/oracle_study/shared_decision_field.py",
        "efficient_seed_core_sha256": "src/oracle_study/efficient_joint_seed.py",
        "split_core_sha256": "src/oracle_study/split_interaction_field.py",
        "interaction_core_sha256": "src/oracle_study/interaction_field.py",
    }
    source_hashes = runner._dependencies()
    if set(source_paths) != set(source_hashes):
        raise RuntimeError("portable source manifest mapping changed")
    payload = {
        "schema_version": 2,
        "config": {"path": str(Path("configs") / config_path.name), "sha256": _sha256(config_path)},
        "fit": {
            "facts": {"path": str(result_prefix / fit_dir.name / runner.FIT_FACTS), "sha256": _sha256(fit_dir / runner.FIT_FACTS)},
            "manifest": {"path": str(result_prefix / fit_dir.name / runner.FIT_MANIFEST), "sha256": _sha256(fit_dir / runner.FIT_MANIFEST)},
        },
        "validation": {
            name: {"path": str(result_prefix / validation_dir.name / filename), "sha256": _sha256(validation_dir / filename)}
            for name, filename in {
                "facts": runner.RUN_FACTS, "group": runner.GROUP_FRONTIER,
                "states": runner.STATE_EVIDENCE, "accounting": runner.ACCOUNTING,
            }.items()
        },
        "sources": {
            "analyzer": {"path": "scripts/analyze_half_bpw_compact_field.py", "sha256": _sha256(Path(__file__))},
            **{
                name: {"path": source_paths[name], "sha256": digest}
                for name, digest in source_hashes.items()
            },
        },
        "generated": {
            name: {"path": str(result_prefix / output.name / name), "bytes": (output / name).stat().st_size, "sha256": _sha256(output / name)}
            for name in generated
        },
    }
    _write_json(output / MANIFEST, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.config, args.fit_dir, args.validation_dir, args.output)


if __name__ == "__main__":
    main()
