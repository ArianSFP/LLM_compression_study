#!/usr/bin/env python3
"""Build the immutable source-slice to live full-model D1 transfer audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from typing import Any

import pandas as pd


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "src"))

from oracle_study.d1_source_live_transfer import (  # noqa: E402
    SCHEMA,
    THRESHOLDS,
    TOKEN_KEYS,
    assemble_transfer_rows,
    baseline_margin_transfer_summary,
    crossing_transfer_summary,
    delta_fidelity_summary,
    materialize_source_labels,
    patched_fixed_pr13_kl_audit,
    threshold_calibration,
)


QUALITY = "d1_cached_decode_tail_quality.parquet"
PROPAGATION = "d1_cached_decode_tail_propagation.parquet"
ROUTE_AUDIT = "tail_route_transfer_audit/d1_tail_route_transfer.parquet"
POLICY_MANIFEST = "d1_cached_decode_tail_policy_bank_manifest.json"
PATCH_GROUPS = "same_host_allocation_patch/d1_tail_same_host_patch_groups.parquet"

OUTPUTS = {
    "rows": "d1_source_live_transfer_rows.parquet",
    "crossing": "d1_source_live_transfer_crossing_summary.parquet",
    "thresholds": "d1_source_live_transfer_threshold_calibration.parquet",
    "baseline": "d1_source_live_transfer_baseline_margin_summary.parquet",
    "delta": "d1_source_live_transfer_delta_fidelity.parquet",
    "patch_rows": "d1_source_live_transfer_fixed_pr13_patch_rows.parquet",
    "patch_summary": "d1_source_live_transfer_fixed_pr13_patch_summary.parquet",
}
FACTS = "d1_source_live_transfer_facts.json"
REPORT = "D1_SOURCE_LIVE_TRANSFER_DIAGNOSIS.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def _load_source_cells(
    manifest: dict[str, Any],
) -> tuple[dict[tuple[int, int], pd.DataFrame], list[dict[str, Any]]]:
    cells: dict[tuple[int, int], pd.DataFrame] = {}
    provenance: list[dict[str, Any]] = []
    for key, cell in sorted(manifest["cells"].items()):
        layer = int(key.split("_rate_")[0].removeprefix("layer_"))
        rate = int(key.split("_rate_")[1])
        path = Path(cell["directory"]) / "d1_decode_exact_route_metrics.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"raw source metrics unavailable for {key}: {path}")
        frame = pd.read_parquet(path)
        cells[(layer, rate)] = frame
        provenance.append(
            {
                "cell": key,
                "path": str(path),
                "rows": int(len(frame)),
                "sha256": _sha256(path),
            }
        )
    return cells, provenance


def _router_margin(logits: np.ndarray, selected_ids: np.ndarray) -> float:
    values = np.asarray(logits, np.float64).reshape(-1)
    selected = np.asarray(selected_ids, np.int64).reshape(-1)
    if selected.size != 8 or len(set(selected.tolist())) != 8:
        raise ValueError("captured D1 route must contain eight unique experts")
    outsider = np.ones(values.size, dtype=bool)
    outsider[selected] = False
    return float(np.min(values[selected]) - np.max(values[outsider]))


def _load_pro_d1_baseline(
    capture_directory: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    paths = sorted(capture_directory.glob("same_host_allocation_layer_*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"no same-host allocation captures: {capture_directory}"
        )
    for path in paths:
        with np.load(path, allow_pickle=False) as arrays:
            if str(arrays["schema"].item()) != (
                "pr13_d1_tail_same_host_allocation_capture_v1"
            ):
                raise ValueError(f"unexpected same-host capture schema: {path}")
            layer = int(arrays["layer"].item())
            admitted = np.asarray(arrays["admitted"], bool)
            for index in np.flatnonzero(admitted):
                logits = np.asarray(arrays["d1_router_logits"][index], np.float64)
                ids = np.asarray(arrays["d1_router_ids"][index], np.int64)
                rows.append(
                    {
                        "injection_layer": layer,
                        "group": int(arrays["group"][index]),
                        "request_id": str(arrays["request_id"][index]),
                        "position": int(arrays["position"][index]),
                        "pro_d1_rank8_rank9_margin": _router_margin(logits, ids),
                        "pro_d1_route_ids": json.dumps(
                            ids.tolist(), separators=(",", ":")
                        ),
                    }
                )
        provenance.append(
            {
                "layer": layer,
                "path": str(path),
                "admitted_rows": int(admitted.sum()),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    result = (
        pd.DataFrame(rows)
        .sort_values(["injection_layer", "group"], kind="stable")
        .reset_index(drop=True)
    )
    keys = ["injection_layer", "group", "request_id", "position"]
    if result.duplicated(keys).any():
        raise ValueError("same-host D1 baseline has duplicate token identities")
    return result, provenance


def _report(
    crossing: pd.DataFrame,
    threshold: pd.DataFrame,
    baseline: pd.DataFrame,
    delta: pd.DataFrame,
    patch: pd.DataFrame,
) -> str:
    all_row = crossing[crossing["scope"].eq("all_policies_rates")].iloc[0]
    fixed = crossing[
        crossing["scope"].eq("policy_all_rates")
        & crossing["policy"].eq("calibration_selected_fixed_d1")
    ].iloc[0]
    baseline_all = baseline[
        baseline["rate_pages_per_expert"].eq(-1) & baseline["patch_stratum"].eq("all")
    ].iloc[0]
    fixed_zero = threshold[
        threshold["policy"].eq("calibration_selected_fixed_d1")
        & threshold["margin_threshold"].eq(0.0)
    ]
    fixed_zero_tp = int(fixed_zero["true_positive"].sum())
    fixed_zero_fn = int(fixed_zero["false_negative"].sum())
    fixed_zero_fp = int(fixed_zero["false_positive"].sum())
    fixed_zero_tn = int(fixed_zero["true_negative"].sum())
    fixed_zero_recall = fixed_zero_tp / (fixed_zero_tp + fixed_zero_fn)
    fixed_zero_fpr = fixed_zero_fp / (fixed_zero_fp + fixed_zero_tn)
    fixed_delta = delta[delta["policy"].eq("calibration_selected_fixed_d1")]
    patch_all = patch[patch["rate_pages_per_expert"].eq(-1)]
    patch_lookup = {
        str(row.patch_stratum): row for row in patch_all.itertuples(index=False)
    }
    return f"""# D1 source-slice to live full-model transfer diagnosis

This is a GPU-free audit of finalized Experiment A artifacts. It performs no
allocator or terminal-KL selection.

## Result

### Candidate signed-margin prediction and live calibration

The source candidate margin retains ranking information, but its zero point is
not a reliable full-model route certificate. Across all policies and rates,
margin-risk ROC AUC is {all_row.candidate_margin_risk_roc_auc:.6f}, while exact
source crossing labels agree with live full-model D1 outcomes only
{all_row.accuracy:.2%}. For the promoted fixed policy the corresponding values
are {fixed.candidate_margin_risk_roc_auc:.6f} and {fixed.accuracy:.2%}. That
accuracy is base-rate dominated: the source crossing label misses
{fixed.false_negative_rate:.2%} of actual fixed-policy crossings.

This distinction matters: AUC measures ordering and is invariant to a shifted
decision threshold; crossing agreement at a chosen margin threshold measures
certification. The threshold table therefore reports 0, 1/64, 2/64 and 4/64
explicitly rather than interpreting a useful ranking as a calibrated
certificate. For the fixed policy at threshold zero, pooled recall is
{fixed_zero_recall:.2%} and the false-positive rate is {fixed_zero_fpr:.2%}.

### Baseline D1 anchor transfer

The raw-source exact D1 Q4 rank-8/rank-9 margin has Pearson
{baseline_all.pearson:.6f}, Spearman {baseline_all.spearman:.6f}, and MAE
{baseline_all.mae:.6f} against the separately captured exact-PRO D1 baseline.
Its source-minus-PRO bias is {baseline_all.bias_source_minus_pro:.6f} and its
maximum absolute error is {baseline_all.maximum_absolute_error:.6f}.

### Delta reconstruction transfer

Fixed-policy stored/live delta cosine averages
{fixed_delta.mean_stored_to_live_delta_cosine.mean():.6f}; mean transfer MSE is
{fixed_delta.mean_stored_to_live_delta_mse.mean():.6g}. Delta transfer is close,
but not bit-exact.

### Remaining error (inference)

The baseline D1 anchor and reconstructed delta transfer substantially better
than the candidate crossing threshold. The remaining gap therefore lies in
candidate signed-effect transfer: source-slice candidate execution, BF16
boundary quantization, and—upstream of the exact source replay—the VJP
linearization used to choose allocations. Because the audited candidate margin
is exact inside the source slice, its live miscalibration cannot be attributed
to the VJP alone. This audit does not identify the shares of those effects.

## Patch stratum

The twelve Q4 route-mismatch identities were regenerated on the PRO host. For
the fixed policy, the PR13-paired mean terminal-KL delta is
{patch_lookup['patched'].mean_fixed_logit_kl_minus_pr13:.6g} on patched rows and
{patch_lookup['unpatched'].mean_fixed_logit_kl_minus_pr13:.6g} on unpatched
rows. These KL values are a post-hoc patch-stratum outcome audit only; they were
not used to select any policy, threshold, or allocation.

## Files and interpretation boundary

- `d1_source_live_transfer_rows.parquet` is the exact joined row-level audit.
- `d1_source_live_transfer_crossing_summary.parquet` separates confusion from
  ranking AUC.
- `d1_source_live_transfer_threshold_calibration.parquet` audits four fixed
  safety thresholds.
- `d1_source_live_transfer_baseline_margin_summary.parquet` audits D1 Q4 margin
  transfer independently of candidate allocations.
- `d1_source_live_transfer_delta_fidelity.parquet` audits stored/live delta
  reconstruction fidelity.
- `d1_source_live_transfer_fixed_pr13_patch_*.parquet` is the narrowly scoped
  patched-versus-unpatched KL comparison.

The data support using source margins as screening scores, not as zero-threshold
certificates. They do not establish a terminal-KL minimization target and do not
promote Experiment B.
"""


def _arguments() -> argparse.Namespace:
    root = (
        EXPERIMENT / "results/qwen36_mxfp4_d1_cached_decode_tail_kl_smoke_20260824_v3"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root)
    parser.add_argument("--output", type=Path, default=root / "transfer_diagnosis")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    root = args.input.resolve()
    output = args.output.resolve()
    inputs = {
        "quality": root / QUALITY,
        "propagation": root / PROPAGATION,
        "route_audit": root / ROUTE_AUDIT,
        "manifest": root / POLICY_MANIFEST,
        "patch_groups": root / PATCH_GROUPS,
        "capture_facts": root
        / "same_host_allocation_capture/d1_tail_same_host_allocation_capture_facts.json",
    }
    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")

    manifest = json.loads(inputs["manifest"].read_text())
    source_cells, source_provenance = _load_source_cells(manifest)
    capture_facts = json.loads(inputs["capture_facts"].read_text())
    if capture_facts.get("schema") != "pr13_d1_tail_same_host_allocation_capture_v1":
        raise RuntimeError("same-host D1 capture schema changed")
    pro_d1_baseline, capture_provenance = _load_pro_d1_baseline(
        inputs["capture_facts"].parent
    )
    recorded_capture_files = capture_facts.get("files", {})
    for record in capture_provenance:
        name = Path(record["path"]).name
        if str(recorded_capture_files.get(name, {}).get("sha256")) != record["sha256"]:
            raise RuntimeError(f"same-host D1 capture changed: {name}")
    quality = pd.read_parquet(inputs["quality"])
    propagation = pd.read_parquet(inputs["propagation"])
    route_audit = pd.read_parquet(inputs["route_audit"])
    patch_groups = pd.read_parquet(inputs["patch_groups"])
    identity_columns = ["layer", "group", "request_id", "position"]
    patch_identity_frame = (
        patch_groups[identity_columns]
        .drop_duplicates()
        .sort_values(identity_columns, kind="stable")
    )
    mismatch_identity_frame = (
        route_audit.loc[~route_audit["route_set_equal"].astype(bool), identity_columns]
        .drop_duplicates()
        .sort_values(identity_columns, kind="stable")
    )
    patch_identity_tuples = set(patch_identity_frame.itertuples(index=False, name=None))
    mismatch_identity_tuples = set(
        mismatch_identity_frame.itertuples(index=False, name=None)
    )
    if patch_identity_tuples != mismatch_identity_tuples:
        raise RuntimeError(
            "same-host patch identities changed from route-audit mismatches"
        )
    patch_identity_records = patch_identity_frame.to_dict("records")

    source = materialize_source_labels(manifest, source_cells, patch_groups)
    rows = assemble_transfer_rows(
        quality, propagation, route_audit, pro_d1_baseline, source
    )
    crossing = crossing_transfer_summary(rows)
    thresholds = threshold_calibration(rows)
    baseline = baseline_margin_transfer_summary(rows)
    delta = delta_fidelity_summary(rows)
    patch_rows, patch_summary = patched_fixed_pr13_kl_audit(rows)

    frames = {
        "rows": rows,
        "crossing": crossing,
        "thresholds": thresholds,
        "baseline": baseline,
        "delta": delta,
        "patch_rows": patch_rows,
        "patch_summary": patch_summary,
    }
    output.mkdir(parents=True, exist_ok=True)
    output_records: dict[str, Any] = {}
    for key, frame in frames.items():
        path = output / OUTPUTS[key]
        _atomic_parquet(path, frame)
        output_records[path.name] = {
            "rows": int(len(frame)),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        }

    report = _report(crossing, thresholds, baseline, delta, patch_summary)
    _atomic_text(output / REPORT, report)
    output_records[REPORT] = {
        "bytes": int((output / REPORT).stat().st_size),
        "sha256": _sha256(output / REPORT),
    }

    all_row = crossing[crossing["scope"].eq("all_policies_rates")].iloc[0]
    fixed = crossing[
        crossing["scope"].eq("policy_all_rates")
        & crossing["policy"].eq("calibration_selected_fixed_d1")
    ].iloc[0]
    baseline_all = baseline[
        baseline["rate_pages_per_expert"].eq(-1) & baseline["patch_stratum"].eq("all")
    ].iloc[0]
    fixed_delta = delta[delta["policy"].eq("calibration_selected_fixed_d1")]
    fixed_zero = thresholds[
        thresholds["policy"].eq("calibration_selected_fixed_d1")
        & thresholds["margin_threshold"].eq(0.0)
    ]
    fixed_zero_tp = int(fixed_zero["true_positive"].sum())
    fixed_zero_fp = int(fixed_zero["false_positive"].sum())
    fixed_zero_tn = int(fixed_zero["true_negative"].sum())
    fixed_zero_fn = int(fixed_zero["false_negative"].sum())
    facts = {
        "schema": SCHEMA,
        "completed": True,
        "analysis_kind": "offline_source_slice_to_live_full_model_transfer_audit",
        "experiment_stage": "A",
        "experiment_b_started": False,
        "terminal_kl_used_for_selection": False,
        "policy_or_threshold_selected": False,
        "implementation": {
            "scripts/analyze_d1_source_live_transfer.py": _sha256(
                Path(__file__).resolve()
            ),
            "src/oracle_study/d1_source_live_transfer.py": _sha256(
                EXPERIMENT / "src/oracle_study/d1_source_live_transfer.py"
            ),
        },
        "thresholds": list(map(float, THRESHOLDS)),
        "rows": int(len(rows)),
        "policies": sorted(map(str, rows["policy"].unique())),
        "rates": sorted(map(int, rows["rate_pages_per_expert"].unique())),
        "layers": sorted(map(int, rows["injection_layer"].unique())),
        "patched_token_identities": int(
            rows.loc[rows["allocation_patched"], TOKEN_KEYS].drop_duplicates().shape[0]
        ),
        "patch_identity_matches_route_transfer_mismatch_set": True,
        "patch_identities": patch_identity_records,
        "transfer_decomposition": {
            "baseline_d1_anchor": {
                "comparison_axis": "same_token_next_layer_router",
                "source": "raw_source_cell_exact_Q4_D1_baseline",
                "target": "same_host_PRO_exact_Q4_D1_baseline_capture",
                "pearson": float(baseline_all["pearson"]),
                "spearman": float(baseline_all["spearman"]),
                "mae": float(baseline_all["mae"]),
                "bias_source_minus_pro": float(baseline_all["bias_source_minus_pro"]),
                "maximum_absolute_error": float(baseline_all["maximum_absolute_error"]),
            },
            "delta_reconstruction": {
                "mean_fixed_stored_to_live_cosine": float(
                    fixed_delta["mean_stored_to_live_delta_cosine"].mean()
                ),
                "mean_fixed_stored_to_live_mse": float(
                    fixed_delta["mean_stored_to_live_delta_mse"].mean()
                ),
                "bit_exact": False,
            },
            "candidate_signed_margin_live_calibration": {
                "fixed_margin_risk_roc_auc": float(
                    fixed["candidate_margin_risk_roc_auc"]
                ),
                "fixed_source_crossing_label_false_negative_rate": float(
                    fixed["false_negative_rate"]
                ),
                "fixed_margin_zero_recall": float(
                    fixed_zero_tp / (fixed_zero_tp + fixed_zero_fn)
                ),
                "fixed_margin_zero_false_positive_rate": float(
                    fixed_zero_fp / (fixed_zero_fp + fixed_zero_tn)
                ),
            },
            "remaining_error": {
                "status": "inference_not_component_identification",
                "likely_components": [
                    "source_slice_to_full_model_candidate_execution",
                    "BF16_boundary_quantization",
                    "upstream_VJP_linearization_used_for_allocation_selection",
                ],
                "vjp_alone_explains_miscalibration": False,
                "reason": (
                    "candidate margins were exact source-slice replay labels, "
                    "so live threshold failure remains after exact source-side replay"
                ),
            },
        },
        "headline": {
            "all_candidate_margin_risk_roc_auc": float(
                all_row["candidate_margin_risk_roc_auc"]
            ),
            "all_source_live_crossing_agreement": float(all_row["accuracy"]),
            "fixed_candidate_margin_risk_roc_auc": float(
                fixed["candidate_margin_risk_roc_auc"]
            ),
            "fixed_source_live_crossing_agreement": float(fixed["accuracy"]),
            "fixed_source_crossing_label_false_negative_rate": float(
                fixed["false_negative_rate"]
            ),
            "fixed_margin_zero_true_positive": fixed_zero_tp,
            "fixed_margin_zero_false_positive": fixed_zero_fp,
            "fixed_margin_zero_true_negative": fixed_zero_tn,
            "fixed_margin_zero_false_negative": fixed_zero_fn,
            "fixed_margin_zero_recall": float(
                fixed_zero_tp / (fixed_zero_tp + fixed_zero_fn)
            ),
            "fixed_margin_zero_false_positive_rate": float(
                fixed_zero_fp / (fixed_zero_fp + fixed_zero_tn)
            ),
            "interpretation": (
                "source candidate margins may rank live D1 risk while the "
                "zero threshold is not a calibrated full-model certificate"
            ),
        },
        "input_files": {
            name: {
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
            for name, path in sorted(inputs.items())
        },
        "raw_source_cells": source_provenance,
        "pro_d1_capture_files": capture_provenance,
        "outputs": output_records,
    }
    _atomic_json(output / FACTS, facts)
    print(json.dumps(facts, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
