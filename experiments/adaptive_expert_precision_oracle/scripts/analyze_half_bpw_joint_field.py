#!/usr/bin/env python3
"""Fail-closed analysis for the 0.5-bpw joint top-8 field study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "scripts"))
import run_half_bpw_joint_field as runner


SUMMARY = "half_bpw_joint_accuracy.csv"
LAYER = "half_bpw_joint_layer_accuracy.csv"
DECISION = "half_bpw_joint_decision.json"
REPORT = "HALF_BPW_JOINT_FIELD_REPORT.md"
PLOT_PNG = "half_bpw_joint_accuracy.png"
PLOT_SVG = "half_bpw_joint_accuracy.svg"
MANIFEST = "half_bpw_joint_analysis_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _quantiles(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, local in frame.groupby("allocation_policy", sort=False):
        recovery = local["group_recovery"].to_numpy(np.float64)
        if recovery.size != 128 or np.any(~np.isfinite(recovery)):
            raise RuntimeError(f"policy recovery evidence is incomplete: {policy}")
        rows.append({
            "allocation_policy": str(policy),
            "groups": int(recovery.size),
            "streamed_bpw": float(local["streamed_bpw"].iloc[0]),
            "allowed_total_bpw": float(local["allowed_total_bpw"].iloc[0]),
            "p10_recovery": float(np.quantile(recovery, 0.1)),
            "median_recovery": float(np.median(recovery)),
            "p90_recovery": float(np.quantile(recovery, 0.9)),
            "mean_recovery": float(np.mean(recovery)),
            "median_damage_ratio_vs_pr13": float(np.median(
                local["damage_ratio_vs_pr13"].to_numpy(np.float64)
            )),
            "median_recovery_delta_vs_pr13": float(np.median(
                local["recovery_delta_vs_pr13"].to_numpy(np.float64)
            )),
            "median_actual_group_pages": float(np.median(
                local["actual_group_pages"].to_numpy(np.float64)
            )),
            "median_selector_macs": float(np.median(
                local["selector_compute_macs"].to_numpy(np.float64)
            )),
            "median_coordinate_sweeps": float(np.median(
                local["coordinate_sweeps"].to_numpy(np.float64)
            )),
            "p90_coordinate_sweeps": float(np.quantile(
                local["coordinate_sweeps"].to_numpy(np.float64), 0.9,
            )),
            "median_local_passes": float(np.median(
                local["local_passes"].to_numpy(np.float64)
            )),
            "p90_local_passes": float(np.quantile(
                local["local_passes"].to_numpy(np.float64), 0.9,
            )),
        })
    return pd.DataFrame(rows)


def _validate(
    config: Mapping[str, Any], validation_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    runner._validate_config(config)
    paths = {
        "facts": validation_dir / runner.RUN_FACTS,
        "group": validation_dir / runner.GROUP_FRONTIER,
        "states": validation_dir / runner.STATE_EVIDENCE,
        "accounting": validation_dir / runner.ACCOUNTING,
    }
    if any(not path.is_file() for path in paths.values()):
        raise RuntimeError("validation bundle is incomplete")
    facts = json.loads(paths["facts"].read_text())
    if facts.get("completed") is not True or facts.get("failures") or facts.get("failure_history"):
        raise RuntimeError("validation facts are not clean and complete")
    if facts.get("completed_layers") != [0, 4, 20, 39]:
        raise RuntimeError("validation layer grid is incomplete")
    expected_hashes = {
        "config_sha256": _sha256(args_config := Path(config["_config_path"])),
        "runner_sha256": _sha256(EXPERIMENT / "scripts/run_half_bpw_joint_field.py"),
        "joint_core_sha256": _sha256(EXPERIMENT / "src/oracle_study/low_rate_joint_field.py"),
        "group_frontier_sha256": _sha256(paths["group"]),
        "state_evidence_sha256": _sha256(paths["states"]),
        "accounting_sha256": _sha256(paths["accounting"]),
    }
    del args_config
    for name, value in expected_hashes.items():
        if facts.get(name) != value:
            raise RuntimeError(f"validation hash changed: {name}")
    frame = pd.read_parquet(paths["group"])
    states = pd.read_parquet(paths["states"])
    expected = int(config["expected_validation_groups"]) * len(runner.POLICIES)
    if len(frame) != expected or len(states) != expected:
        raise RuntimeError("validation row count changed")
    keys = list(runner.GROUP_IDENTITY) + ["allocation_policy"]
    if frame[keys].duplicated().any() or states[keys].duplicated().any():
        raise RuntimeError("duplicate validation identity")
    if set(frame["allocation_policy"].astype(str)) != set(runner.POLICIES):
        raise RuntimeError("validation policy grid changed")
    if set(map(tuple, frame[keys].itertuples(index=False, name=None))) != set(
        map(tuple, states[keys].itertuples(index=False, name=None))
    ):
        raise RuntimeError("state/group identity sets differ")
    if np.any(frame["streamed_bpw"].to_numpy(np.float64) != 0.5):
        raise RuntimeError("streamed rate changed")
    if np.any(frame["actual_group_pages"].to_numpy(np.int64) > 3072):
        raise RuntimeError("group page budget exceeded")
    numeric = (
        "group_base_qenergy_damage", "group_exact_qenergy_damage",
        "group_recovery", "recovery_delta_vs_pr13", "damage_ratio_vs_pr13",
        "allowed_total_bpw", "metadata_bpw",
    )
    if np.any(~np.isfinite(frame[list(numeric)].to_numpy(np.float64))):
        raise RuntimeError("validation numeric evidence is non-finite")
    recomputed = 1.0 - (
        frame["group_exact_qenergy_damage"].to_numpy(np.float64)
        / frame["group_base_qenergy_damage"].to_numpy(np.float64)
    )
    np.testing.assert_allclose(recomputed, frame["group_recovery"], rtol=1e-12, atol=1e-12)
    merged = states.merge(frame[keys + ["selected_states_sha256"]], on=keys, validate="one_to_one")
    for row in merged.to_dict("records"):
        values = np.asarray(json.loads(row["selected_states"]), np.uint8)
        if values.shape != (4096,) or np.any(values > 7):
            raise RuntimeError("serialized joint states are invalid")
        observed = hashlib.sha256(values.tobytes()).hexdigest()
        if observed != row["selected_states_sha256_x"] or observed != row["selected_states_sha256_y"]:
            raise RuntimeError("serialized joint-state hash changed")
    pivot = frame.pivot_table(
        index=list(runner.GROUP_IDENTITY), columns="allocation_policy",
        values="group_exact_qenergy_damage", aggfunc="first",
    )
    if pivot.isna().any().any():
        raise RuntimeError("policy coverage is incomplete per group")
    tolerance = float(config["dominance_tolerance"])
    if np.any(
        pivot[runner.POLICIES[3]].to_numpy(np.float64)
        > pivot[runner.POLICIES[0]].to_numpy(np.float64) + tolerance
    ):
        raise RuntimeError("exact joint field does not dominate PR13 per group")
    return frame, states, facts


def analyze(config_path: Path, validation_dir: Path, output: Path) -> None:
    config = json.loads(config_path.read_text())
    config["_config_path"] = str(config_path.resolve())
    frame, _, facts = _validate(config, validation_dir)
    output.mkdir(parents=True, exist_ok=False)
    summary = _quantiles(frame)
    summary.to_csv(output / SUMMARY, index=False)
    layer_rows = []
    for (policy, layer), local in frame.groupby(
        ["allocation_policy", "layer"], sort=False,
    ):
        values = local["group_recovery"].to_numpy(np.float64)
        if values.size != 32:
            raise RuntimeError("layer/policy group count changed")
        layer_rows.append({
            "allocation_policy": str(policy),
            "layer": int(layer),
            "groups": int(values.size),
            "p10_recovery": float(np.quantile(values, 0.1)),
            "median_recovery": float(np.median(values)),
            "p90_recovery": float(np.quantile(values, 0.9)),
        })
    pd.DataFrame(layer_rows).to_csv(output / LAYER, index=False)
    indexed = summary.set_index("allocation_policy")
    primary = indexed.loc[runner.POLICIES[1]]
    exact = indexed.loc[runner.POLICIES[3]]
    primary_pass = bool(
        primary["p10_recovery"] >= float(config["primary_target_p10_recovery"])
        and primary["median_recovery"] >= float(config["primary_target_median_recovery"])
    )
    exact_pass = bool(
        exact["p10_recovery"] >= float(config["primary_target_p10_recovery"])
        and exact["median_recovery"] >= float(config["primary_target_median_recovery"])
    )
    if primary_pass:
        next_action = "freeze_rank4_joint_field_for_predicted_h4_followup"
    elif exact_pass:
        next_action = "fit_one_frozen_layer_shared_euclidean_tail_then_repeat_once"
    else:
        next_action = "expand_the_action_representation_before_more_geometry_tuning"
    decision = {
        "schema_version": 1,
        "status": "continue" if primary_pass else "geometry_gap" if exact_pass else "action_gap",
        "selected_policy": runner.POLICIES[1] if primary_pass else None,
        "primary_target_p10_recovery": float(config["primary_target_p10_recovery"]),
        "primary_target_median_recovery": float(config["primary_target_median_recovery"]),
        "primary_p10_recovery": float(primary["p10_recovery"]),
        "primary_median_recovery": float(primary["median_recovery"]),
        "exact_ceiling_p10_recovery": float(exact["p10_recovery"]),
        "exact_ceiling_median_recovery": float(exact["median_recovery"]),
        "primary_pass": primary_pass,
        "exact_ceiling_pass": exact_pass,
        "next_action": next_action,
        "test_scientific_values_used": False,
    }
    _write_json(output / DECISION, decision)

    labels = [
        "PR13", "rank4 INT4\njoint", "rank4 FP32\njoint", "exact-output\njoint",
    ]
    x = np.arange(len(labels))
    ordered = indexed.loc[list(runner.POLICIES)]
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    axis.plot(x, 100.0 * ordered["p10_recovery"], marker="o", label="p10")
    axis.plot(x, 100.0 * ordered["median_recovery"], marker="o", label="median")
    axis.axhline(99.0, color="black", linestyle="--", linewidth=1, label="99% p10 target")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Combined top-8 recovery (%)")
    axis.set_title("Exactly 0.5 streamed bpw")
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
            f"| `{policy}` | {100*row['p10_recovery']:.4f}% | "
            f"{100*row['median_recovery']:.4f}% | {100*row['p90_recovery']:.4f}% | "
            f"{row['median_damage_ratio_vs_pr13']:.6f} | "
            f"{row['allowed_total_bpw']:.6f} |"
        )
    report = f"""# Half-bpw joint-field report

All rows use the same 128 exact-H4 validation token/layer groups and exactly
0.5 streamed correction bpw. PR #13 is immutable and is replayed from its
hash-locked selected states.

| Policy | p10 recovery | Median recovery | p90 recovery | Median damage / PR13 | Allowed total bpw |
| :--- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table)}

The frozen target is 99.0% p10 and 99.5% median. The rank-4 deployable-geometry
control **{'passes' if primary_pass else 'does not pass'}**. The exact-output
unit-level solver ceiling **{'passes' if exact_pass else 'does not pass'}**.
The predeclared next action is `{next_action}`.

This is an exact-H4 geometry/optimization study. It evaluates no predicted H4,
latency, downstream replay, routing, logits, tokens, or test-split scientific
values. The exact-output joint field is a validation-only ceiling.
"""
    (output / REPORT).write_text(report)
    generated = [SUMMARY, LAYER, DECISION, REPORT, PLOT_PNG, PLOT_SVG]
    result_prefix = Path("results") / str(config["run_id"])
    manifest = {
        "schema_version": 1,
        "config": {
            "path": str(Path("configs") / config_path.name),
            "sha256": _sha256(config_path),
        },
        "validation_inputs": {
            name: {
                "path": str(result_prefix / validation_dir.name / filename),
                "sha256": _sha256(validation_dir / filename),
            }
            for name, filename in {
                "facts": runner.RUN_FACTS,
                "group": runner.GROUP_FRONTIER,
                "states": runner.STATE_EVIDENCE,
                "accounting": runner.ACCOUNTING,
            }.items()
        },
        "sources": {
            "analyzer": {
                "path": "scripts/analyze_half_bpw_joint_field.py",
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "runner": {
                "path": "scripts/run_half_bpw_joint_field.py",
                "sha256": facts["runner_sha256"],
            },
            "joint_core": {
                "path": "src/oracle_study/low_rate_joint_field.py",
                "sha256": facts["joint_core_sha256"],
            },
        },
        "generated": {
            name: {
                "path": str(result_prefix / output.name / name),
                "bytes": (output / name).stat().st_size,
                "sha256": _sha256(output / name),
            }
            for name in generated
        },
        "decision_sha256": _sha256(output / DECISION),
        "test_scientific_values_used": False,
    }
    _write_json(output / MANIFEST, manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.config, args.validation_dir, args.output)


if __name__ == "__main__":
    main()
