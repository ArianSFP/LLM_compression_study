#!/usr/bin/env python3
"""Audit exact-validation evidence and freeze set-utility pilot decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "src"))

from oracle_study.set_utility_analysis import (  # noqa: E402
    CANDIDATE_SEMANTICS,
    CANDIDATE_PACKET_CONTENTS,
    CONTAINED,
    FULL_TARGET,
    IDENTITY,
    INDEPENDENT,
    PREDICTED,
    INTERACTION_RERANK_ACCOUNTING_BOUND,
    INTERACTION_RERANK_UNACCOUNTED,
    RERANK_NON_MAC_CONVENTION,
    TEST_METADATA_AUDIT,
    freeze_validation_promotions,
    quantiles,
    recompute_accounting,
    sha256,
    summarize,
    validate_evidence_bundle,
    validate_promotion_payload,
)


matplotlib.rcParams["svg.hashsalt"] = "set-utility-distillation-20260820"
REPORT_NAME = "SET_UTILITY_DISTILLATION_REPORT.md"
SUPPORT_TEMPLATE_NULL_ACCOUNTING_FIELDS = (
    "selector_compute_additions",
    "selector_scale_multiplications",
)
SUPPORT_TEMPLATE_NULL_ACCOUNTING_REASON = (
    "not_applicable_validation_oracle_has_no_deployable_selector"
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=_json_default) + "\n",
    )


def markdown_table(frame: pd.DataFrame, *, limit: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    shown = frame.head(limit).copy()
    for column in shown:
        shown[column] = shown[column].map(
            lambda value: (
                f"{value:.6f}" if isinstance(value, (float, np.floating))
                else str(value).replace("|", "\\|").replace("\n", " ")
            ),
        )
    headers = list(map(str, shown.columns))
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend(
        "| " + " | ".join(map(str, row)) + " |"
        for row in shown.itertuples(index=False, name=None)
    )
    if len(frame) > limit:
        lines.append(f"\n_Shows {limit} of {len(frame)} rows._")
    return "\n".join(lines)


def save_figure(fig: plt.Figure, output: Path, stem: str) -> list[Path]:
    paths = [output / f"{stem}.png", output / f"{stem}.svg"]
    fig.savefig(paths[0], dpi=180, bbox_inches="tight")
    fig.savefig(
        paths[1], bbox_inches="tight",
        metadata={"Date": "2026-08-20", "Creator": "set-utility-distillation"},
    )
    svg = paths[1].read_text()
    paths[1].write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n")
    plt.close(fig)
    return paths


def training_provenance(manifest: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for layer_text, record in sorted(manifest["layers"].items(), key=lambda item: int(item[0])):
        layer = int(layer_text)
        counts = record.get("teacher_rows_by_cohort", {})
        for cohort, count in sorted(counts.items()):
            if cohort == "routed_exact_train_primary":
                semantics = "actual_routed_occurrences"
            else:
                semantics = "synthetic_all_x_by_expert_augmentation"
            rows.append({
                "layer": layer, "training_cohort": cohort,
                "pairing_semantics": semantics, "teacher_rows": int(count),
                "fit_split": record.get("fit_split"),
                "validation_or_test_rows_used": record.get("validation_or_test_rows_used"),
            })
    return pd.DataFrame(rows)


def coverage_summary(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for artifact, frame in frames.items():
        for layer, local in frame.groupby("layer", sort=True):
            rows.append({
                "artifact": artifact, "layer": int(layer), "rows": len(local),
                "unique_invocations": len(local[list(IDENTITY)].drop_duplicates()),
                "unique_requests": local["request_id"].astype(str).nunique(),
                "experts": local["expert_id"].astype(int).nunique(),
                "split": ",".join(sorted(set(local["evaluation_split"].astype(str)))),
                "capture_source": ",".join(sorted(set(local["capture_source"].astype(str)))),
            })
    return pd.DataFrame(rows)


def selector_accounting(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "layer", "expert_id", "selector_config_id", "selector_family",
        "expert_specific_metadata_bytes", "layer_shared_metadata_bytes",
        "layer_shared_amortized_bytes_per_expert", "abc_metadata_bytes",
        "selector_metadata_bytes_per_expert", "selector_metadata_bpw",
        "selector_metadata_bpw_recomputed", "selector_compute_macs",
        "selector_abc_score_units", "selector_abc_score_compute_macs",
        "selector_abc_score_macs_per_unit", "selector_abc_score_mac_convention",
        "selector_abc_score_units_recomputed", "selector_abc_score_compute_macs_recomputed",
        "selector_abc_score_macs_per_unit_recomputed",
        "selector_compute_macs_recomputed", "selector_compute_additions",
        "selector_compute_additions_recomputed", "selector_scale_multiplications",
        "selector_scale_multiplications_recomputed", "selector_bytes_read",
        "selector_bytes_read_recomputed", "selector_bytes_read_semantics",
        "selector_activation_payload_already_read", "selector_activation_payload_bytes_read",
        "selector_q2_unit_feature_bytes_read", "selector_abc_metadata_bytes_read",
        "storage_multiplier", "storage_multiplier_recomputed",
    ]
    return frame[[column for column in columns if column in frame]].drop_duplicates().sort_values(
        ["selector_family", "selector_config_id", "layer", "expert_id"],
    )


def candidate_rerank_accounting(frame: pd.DataFrame) -> pd.DataFrame:
    """Preserve exact rerank components/caveats as a first-class CSV artifact."""
    columns = [
        "layer", "expert_id", "selector_config_id", "selector_family",
        "rerank_semantics", "selection_regime", "rerank_role", "candidate_units",
        "applied_units", "activation_payload_already_read", "q2_payload_already_read",
        "abc_payload_already_read", "q2_gate_up_scale_payload_already_read",
        "rerank_activation_payload_bytes_read", "rerank_q2_unit_feature_bytes_read",
        "rerank_abc_metadata_bytes_read", "rerank_candidate_packet_bytes_read",
        "rerank_candidate_packet_contents", "rerank_q2_gate_up_parent_code_bytes_read",
        "rerank_q2_gate_up_scale_bytes_read", "rerank_q2_gate_up_scale_payload_already_read",
        "rerank_q2_gate_up_parent_payload_required",
        "rerank_q2_gate_up_parent_payload_accounted", "rerank_q4_response_units",
        "rerank_q4_response_compute_macs", "rerank_q4_response_scale_multiplications",
        "rerank_down_correction_compute_macs", "rerank_correction_workspace_bytes",
        "rerank_interaction_gram_units", "rerank_interaction_gram_euclidean_compute_macs",
        "rerank_interaction_proxy_rank", "rerank_interaction_proxy_projection_compute_macs",
        "rerank_interaction_proxy_gram_compute_macs",
        "rerank_interaction_proxy_scale_multiplications", "rerank_proxy_metadata_bytes_read",
        "rerank_interaction_gram_compute_macs", "rerank_abc_score_macs_per_unit",
        "rerank_abc_score_mac_convention", "rerank_non_mac_operations",
        "rerank_resident_q2_down_payload_required", "rerank_resident_q2_down_payload_accounted",
        "rerank_accounting_is_lower_bound", "rerank_accounting_bound",
        "rerank_unaccounted_overhead", "rerank_incremental_compute_macs",
        "rerank_incremental_scale_multiplications", "rerank_incremental_bytes_read",
        "selector_compute_macs", "selector_compute_additions", "selector_scale_multiplications",
        "selector_bytes_read", "total_selector_compute_macs",
        "total_selector_scale_multiplications", "total_selector_bytes_read",
    ]
    recomputed = [
        column for column in frame.columns
        if column.endswith("_recomputed") and column.startswith(("rerank_", "total_selector_"))
    ]
    selected = [column for column in [*columns, *sorted(recomputed)] if column in frame]
    exact = frame[selected].copy()
    exact["row_count"] = 1
    group_columns = [column for column in selected]
    exact = exact.groupby(group_columns, dropna=False, sort=False)["row_count"].sum().reset_index()
    return exact.sort_values([
        "selector_family", "selector_config_id", "layer", "expert_id",
        "rerank_semantics", "candidate_units",
    ])


def candidate_rerank_json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Encode only declared support-template omissions as JSON null.

    The support-template branch is a validation-oracle existence control and
    has no deployable base selector.  Its raw evidence therefore omits the two
    base-selector operation breakdowns below; Parquet/CSV preserve those cells
    as missing.  JSON has no NaN representation, so encode exactly those
    family-inapplicable cells as null while leaving every other non-finite
    value for ``allow_nan=False`` to reject.
    """

    required = {"selector_family", *SUPPORT_TEMPLATE_NULL_ACCOUNTING_FIELDS}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise RuntimeError(
            "candidate rerank accounting lacks JSON-null contract fields: "
            + ", ".join(missing_columns)
        )
    support_template = frame["selector_family"].astype(str).eq("support_template").to_numpy()
    encoded = frame.astype(object).copy()
    for column in SUPPORT_TEMPLATE_NULL_ACCOUNTING_FIELDS:
        missing = frame[column].isna().to_numpy()
        invalid_missing = missing & ~support_template
        if invalid_missing.any():
            index = int(np.flatnonzero(invalid_missing)[0])
            raise RuntimeError(
                f"{column} is missing outside a support-template accounting row at {index}"
            )
        present = ~missing
        if present.any():
            try:
                numeric = pd.to_numeric(frame.loc[present, column], errors="raise").to_numpy(
                    np.float64,
                )
            except (TypeError, ValueError) as error:
                raise RuntimeError(f"{column} contains a non-numeric value") from error
            if not np.isfinite(numeric).all():
                index = int(np.flatnonzero(present)[np.flatnonzero(~np.isfinite(numeric))[0]])
                raise RuntimeError(f"{column} is non-finite at accounting row {index}")
        encoded.loc[missing, column] = None
    return encoded.to_dict("records")


def _frontier_tables(accounted: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    hybrid_groups = ["selector_family", "physical_budget_bpw", "selection_regime"]
    hybrid = summarize(accounted["hybrid"], hybrid_groups, ["recovery", "physical_bpw"])
    hybrid_layer = summarize(
        accounted["hybrid"], ["layer", *hybrid_groups], ["recovery", "physical_bpw"],
    )
    template_groups = [
        "template_cohort", "pairing_semantics", "template_selector",
        "template_count_requested", "template_count_effective", "repair_count", "selection_regime",
    ]
    template = summarize(
        accounted["template"], template_groups,
        ["validation_oracle_path_utility_retained", "candidate_units", "physical_bpw", "candidate_overfetch"],
    )
    selector = summarize(
        accounted["selector"],
        ["selector_config_id", "selector_family", "selection_regime"],
        [
            "recovery", "delta_gate_relative_mse", "delta_up_relative_mse",
            "hidden_q4_relative_mse", "selector_runtime_ms", "selector_metadata_bpw",
            "selector_compute_macs", "selector_bytes_read", "physical_bpw", "logical_bpw",
        ],
    )
    selector_layer = summarize(
        accounted["selector"],
        ["layer", "selector_config_id", "selector_family", "selection_regime"],
        ["recovery", "selector_metadata_bpw", "selector_compute_macs"],
    )
    candidate_groups = [
        "selector_config_id", "selector_family", "rerank_semantics", "selection_regime",
        "rerank_role", "rerank_accounting_is_lower_bound", "rerank_accounting_bound",
        "rerank_unaccounted_overhead", "rerank_resident_q2_down_payload_required",
        "rerank_resident_q2_down_payload_accounted", "rerank_non_mac_operations",
        "rerank_candidate_packet_contents",
    ]
    candidate = summarize(
        accounted["candidate"],
        [column for column in candidate_groups if column in accounted["candidate"]],
        [
            "recovery", "oracle_set_gain_retention", "physical_bpw", "logical_bpw",
            "page_amplification", "selector_metadata_bpw", "selector_compute_macs",
            "rerank_incremental_compute_macs", "total_selector_compute_macs",
            "rerank_q2_gate_up_parent_code_bytes_read", "rerank_q2_gate_up_scale_bytes_read",
            "rerank_q4_response_scale_multiplications",
            "rerank_interaction_proxy_scale_multiplications", "rerank_incremental_bytes_read",
            "total_selector_bytes_read",
        ],
    )
    candidate_layer = summarize(
        accounted["candidate"],
        ["layer", "selector_config_id", "selector_family", "rerank_semantics", "selection_regime"],
        ["recovery", "oracle_set_gain_retention"],
    )
    return {
        "hybrid_oracle_summary": hybrid,
        "hybrid_oracle_per_layer": hybrid_layer,
        "template_existence_oracle_summary": template,
        "synopsis_direct_score_summary": selector,
        "synopsis_direct_score_per_layer": selector_layer,
        "candidate_interface_summary": candidate,
        "candidate_interface_per_layer": candidate_layer,
    }


def _plot_hybrid(table: pd.DataFrame, output: Path) -> list[Path]:
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    for family, local in table.groupby("selector_family", sort=True):
        local = local.sort_values("physical_budget_bpw")
        axis.plot(local["physical_budget_bpw"], 100 * local["recovery_median"], marker="o", label=family)
        axis.fill_between(
            local["physical_budget_bpw"].to_numpy(float),
            100 * local["recovery_p10"].to_numpy(float),
            100 * local["recovery_p90"].to_numpy(float), alpha=0.10,
        )
    axis.set(xlabel="Physical correction bpw", ylabel="Complete-expert recovery (%)")
    axis.set_title("Exact H0 coherent/G/D hybrid oracle — validation only")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    return save_figure(fig, output, "hybrid_oracle_frontier")


def _plot_templates(table: pd.DataFrame, output: Path) -> list[Path]:
    fig, axis = plt.subplots(figsize=(8.0, 4.8))
    for key, local in table.groupby(
        ["pairing_semantics", "template_selector", "repair_count"], sort=True,
    ):
        label = " / ".join(map(str, key))
        local = local.sort_values("template_count_requested")
        axis.plot(
            local["template_count_requested"],
            100 * local["validation_oracle_path_utility_retained_median"],
            marker="o", label=label,
        )
    axis.axhline(95, color="black", linewidth=0.8, linestyle="--")
    axis.axhline(98, color="black", linewidth=0.8, linestyle=":")
    axis.set(xlabel="Requested templates per expert", ylabel="Teacher-path utility retained (%)")
    axis.set_title("Per-expert support-template existence oracle")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=6)
    return save_figure(fig, output, "support_template_existence_frontier")


def _plot_candidates(table: pd.DataFrame, output: Path) -> list[Path]:
    configs = sorted(set(table["selector_config_id"].astype(str)))
    semantics = list(CANDIDATE_SEMANTICS)
    fig, axis = plt.subplots(figsize=(max(8.0, 0.9 * len(configs)), 4.8))
    width = 0.19
    x = np.arange(len(configs), dtype=float)
    for offset, semantic in enumerate(semantics):
        lookup = table[table["rerank_semantics"] == semantic].set_index("selector_config_id")
        values = np.array([
            100 * float(lookup.loc[config, "recovery_median"]) if config in lookup.index else np.nan
            for config in configs
        ])
        axis.bar(x + (offset - 1.5) * width, values, width=width, label=semantic)
    axis.axhline(94, color="black", linestyle="--", linewidth=0.8)
    axis.set_xticks(x, configs, rotation=35, ha="right")
    axis.set_ylabel("Median complete-expert recovery (%)")
    axis.set_title("256 fetched → 192 applied: four non-interchangeable interfaces")
    axis.legend(fontsize=6)
    axis.grid(axis="y", alpha=0.25)
    return save_figure(fig, output, "candidate_rerank_frontier")


def _plot_accounting(selector: pd.DataFrame, output: Path) -> list[Path]:
    summary = selector.groupby(["selector_config_id", "selector_family"], sort=True).agg(
        recovery_median=("recovery", "median"),
        metadata_bpw=("selector_metadata_bpw", "max"),
        compute_macs=("selector_compute_macs", "max"),
    ).reset_index()
    fig, axis = plt.subplots(figsize=(6.8, 4.8))
    for family, local in summary.groupby("selector_family", sort=True):
        axis.scatter(local["metadata_bpw"], 100 * local["recovery_median"], label=family, s=42)
        for row in local.itertuples(index=False):
            axis.annotate(str(row.selector_config_id), (row.metadata_bpw, 100 * row.recovery_median), fontsize=6)
    axis.axvline(0.35, color="black", linestyle="--", linewidth=0.8)
    axis.set(xlabel="Selector metadata (bpw)", ylabel="Direct-score recovery (%)")
    axis.set_title("Synopsis/direct-set selector accuracy and charged metadata")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    return save_figure(fig, output, "selector_compute_storage_frontier")


def _decision_text(payload: Mapping[str, Any]) -> str:
    selected = payload["h0_late_candidate_policy"]["selected_validation_configuration"]
    if selected is None:
        selector = "No predicted or independent-ABC H0-late policy passes every frozen gate."
    else:
        selector = (
            f"Validation advances `{selected['selector_config_id']}` / "
            f"`{selected['rerank_semantics']}` to a new sealed holdout; "
            "this is not a confirmatory result or an H4 prefetch/latency result."
        )
    return (
        f"Hybrid: **{payload['hybrid_oracle']['status']}**. "
        f"Templates: **{payload['template_existence_oracle']['status']}**. {selector}"
    )


def build_report(
    evidence: Any, tables: Mapping[str, pd.DataFrame], gate_tables: Mapping[str, pd.DataFrame],
    promotion: Mapping[str, Any], provenance: pd.DataFrame, coverage: pd.DataFrame,
    accounting: pd.DataFrame,
) -> str:
    candidate = tables["candidate_interface_summary"]
    selector = tables["synopsis_direct_score_summary"]
    pq_high = selector[selector["selector_family"].isin([
        "block_pq_residual_synopsis", "high_rank_low_bit_linear_response",
    ])]
    direct = selector[selector["selector_family"] == "direct_set_predictor"]
    return f"""# Set-utility distillation study

## Outcome

{_decision_text(promotion)}

This is a strict **exact-checkpoint validation-only** pilot. The runner necessarily physically
loads immutable monolithic capture archives that contain test rows. It inspects only the `split`
labels and `request_id` arrays for train/validation/test, solely to verify within-capture request
separation and cross-capture split compatibility. No test scientific tensor or value is admitted,
evaluated for a metric, used for tuning, summarized, or used in a decision. It consumes actual
x/Q2 features only at H0, does not train or evaluate
an H4 predictor or demonstrate early prefetch latency, and makes no router, logit, token-quality,
or measured-latency claim. A passing validation configuration requires a new sealed holdout
before any confirmatory claim.

## Frozen scientific boundary

- Base experiment: `{evidence.config['base_pr9_commit']}`.
- Checkpoint revision: `{evidence.config['reference']['revision']}`.
- Selected-tree SHA-256: `{evidence.config['locked_tree_sha256']}`.
- Exact capture SHA-256: `{evidence.config['locked_capture_sha256']['exact_checkpoint']}`.
- Cross-reference capture SHA-256 (fit only): `{evidence.config['locked_capture_sha256']['cross_reference']}`.
- Evaluation: {evidence.run_facts['observed_unique_invocations']} exact-checkpoint validation invocations.
- Candidate interface: 256 coherent packets fetched (768 pages; 393,216 bytes; 1.0 physical bpw),
  followed by 192 applications (0.75 logical bpw).

The locked Q2→Q3→Q4 codec, trees, checkpoint, request splits, and future-proxy qenergy metric
are unchanged. Actual routed training occurrences and synthetic activation/expert augmentation
are reported separately.

## Evidence coverage

{markdown_table(coverage)}

Before any summaries or gates, the analyzer independently requires exactly 69 unique validation
invocations spanning every one of the 12 config-locked `(layer, expert)` cells. It then matches
each validation-scoped fit-manifest selector entry to a complete per-cell invocation grid in both
selector artifacts and all four candidate semantics. A wholly absent selector configuration,
expert cell, or candidate interface is therefore a fatal evidence error rather than an omitted row.
The same pre-summary audit requires the runner's five named hybrid families at each configured
0.5/0.75/1.0-bpw budget for all 69 identities. Template evidence is matched to every
validation-scoped fit-manifest cohort and its config-frozen K grid, both top-1 and top-2-union
interfaces, and repair counts 0/16/32/64; each combination and its four rerank interfaces must
cover the exact expert-cell identities. Candidate-cap flags and physical charging are recomputed;
intentional top-2 unions above 256 candidates remain visible as negative, nonpromotable evidence
and cannot pass the existing candidate-max gate. Template results remain explicitly
nonpromotable existence oracles.

## Training provenance: actual routed versus synthetic augmentation

{markdown_table(provenance)}

Synthetic all-activation/expert pairs add teacher examples, but they are not routed occurrences
and are never counted as validation evidence.

## 1. Coherent/G/D hybrid H0 oracle

{markdown_table(tables['hybrid_oracle_summary'][[
    'selector_family', 'physical_budget_bpw', 'recovery_p10', 'recovery_median', 'recovery_p90',
]])}

The continuation gate is an absolute ≥0.005 improvement in one-bpw median recovery over the
`coherent_exact_set_fixed_greedy_teacher`. Paired deltas versus the
`coherent_independent_abc_pr9_control` are reported separately. It is an exact H0 oracle comparison, not a deployable predictor result.

{markdown_table(gate_tables['hybrid_gate'])}

## 2. Per-expert support-template existence oracle

{markdown_table(gate_tables['template_gate'][[
    'template_cohort', 'pairing_semantics', 'template_selector', 'template_count_requested',
    'repair_count', 'validation_oracle_path_utility_retained_p10',
    'validation_oracle_path_utility_retained_median', 'candidate_max',
    'complete_validation_coverage', 'passes',
]])}

Template membership is chosen with validation teacher utility in this experiment. Therefore
this table answers whether a compact support regime exists; it does **not** evaluate a template
classifier. Top-1 requires median gain retention ≥0.95. Top-2 requires median ≥0.98 and p10
≥0.95. Both require at most 256 charged candidates and complete validation coverage.

## 3. PQ and high-rank/low-bit synopsis direct score

{markdown_table(pq_high[[
    'selector_config_id', 'selector_family', 'recovery_p10', 'recovery_median',
    'selector_metadata_bpw_max', 'selector_compute_macs_max',
]])}

Block-PQ uses activation-second-moment-weighted, 32-weight blocks with physically packed code
indices. The high-rank/low-bit control is reported as least-squares response fitting followed by
quantization, not as jointly optimized low-bit training. It is a transductive sampled-expert
diagnostic and cannot be promoted. The runner and analyzer both charge the 1,024 output-scale
multiplications required by the two row-scaled 512-output synthesis matrices per invocation.
The fit manifest marks these controls nonpromotable, and their transductive sampled-expert scope
independently enforces that exclusion.

## 4. Direct set predictor

{markdown_table(direct[[column for column in (
    'selector_config_id', 'selector_family', 'recovery_p10', 'recovery_median',
    'selector_metadata_bpw_max', 'selector_compute_macs_max',
) if column in direct]])}

Training may use exact marginal paths and correction Grams, while inference is restricted to
declared Q2-side features and charged synopsis/static metadata. Soft or straight-through masks
are training surrogates only; every reported validation decision is a hard discrete 256/192 set.
The joint-head variant supervises coverage of the teacher top-192 under the actual nested
top-192 apply-head plus top-64 non-duplicate candidate-head fetch construction. Although the
training helper supports optional exclusion-regret targets, this run supplies and optimizes no
exclusion-regret supervision.

## 5. Candidate interface: four distinct meanings

{markdown_table(candidate[[
    'selector_config_id', 'selector_family', 'rerank_semantics', 'selection_regime',
    'recovery_p10', 'recovery_median', 'oracle_set_gain_retention_p10',
    'selector_metadata_bpw_max', 'selector_compute_macs_max',
    'rerank_incremental_compute_macs_max', 'total_selector_compute_macs_max',
]])}

- `{PREDICTED}` applies the predicted 192 with no exact reranking.
- `{INDEPENDENT}` computes exact independent A/B/C scores for fetched units. It is the primary
  realistic H0-late reranker and charges Q4 gate/up response work for all 256 candidates,
  including their resident Q2 gate/up parent-code payload, conditional E8M0 scale payload, and
  reported Q4-response block-scale applications. The fetched 1,536-byte packet per unit remains
  suffix-only and excludes resident Q2 data.
- `{CONTAINED}` forms fetched correction vectors and builds the rank-4-proxy-augmented 256-action
  interaction Gram (about 138.7M incremental analytical MACs). It charges Q4 responses,
  the same 256-unit gate/up parent codes, conditional E8M0 scales, Q4 block-scale applications,
  correction formation/workspace, the Euclidean and proxy Gram components, and proxy metadata.
  It remains a precise lower bound only because resident Q2-down parent-code/scale payload,
  Q2-down tree/E8M0 decode and block-scale application, fixed-greedy initial correlation, and
  per-selection vector updates remain excluded. It is an expensive H0 systems oracle and is
  never used for promotion.
- `{FULL_TARGET}` also uses omitted-unit effects. It is a nondeployable teacher-information
  control and is never used for promotion. Its declared arithmetic charges all 512 Q4 teacher
  responses and their gate/up parent codes, conditional E8M0 scales and block-scale applications,
  correction formation, the all-512 correction workspace, and Euclidean plus rank-4 proxy Gram
  arithmetic. It has the same explicit Q2-down/fixed-greedy lower-bound exclusions as the
  contained path.
  Every exact path is
  `exact_marginal_fixed_greedy` with fixed coefficients and no least-squares refit—not a global
  top-k optimum. Restricted and unrestricted fixed-greedy paths can differ through path
  dependence, so the reported gain-retention ratio may legitimately exceed 1 and is not clipped.

## Frozen gates

{markdown_table(gate_tables['selector_gate'])}

Every promoted predicted or independent-ABC H0-late configuration must have median recovery ≥0.94, p10 recovery
≥0.88, p10 exact-set-gain retention ≥0.97, metadata ≤0.35 bpw, total selector plus rerank compute strictly below
1,572,864 MACs, and total storage below 5×. Physical traffic charges all 256 fetched packets;
logical traffic reflects only 192 applied packets.

## Recomputed compute and storage accounting

{markdown_table(accounting)}

The analyzer recomputes page bytes/bpw, logical actions/bpw, amplification, expert and amortized
layer metadata, storage multiplier, and family-specific MACs. Ordinary three-scalar FP16 A/B/C
metadata is 3,084 bytes per expert (three 512-value FP16 arrays plus three FP32 scales). The
nonpromotable static-unit-bias control adds one FP16 scalar per unit and therefore charges 4,108
bytes. Standalone PQ and high-rank scores also charge the exact 3,072-byte Q2 g/u/h payload and
serialized A/B/C payload plus 3,072 analytical MACs (six per unit) for the 512 A/B/C scores;
direct predictors charge serialized A/B/C bytes rather than a vector
dimension approximation. MACs, additions, scale multiplications, non-MAC/decode/control
operations, and logical bytes are reported as distinct quantities. Every MAC column is an
analytical linear/algebraic contract; nonlinear SiLU/log/sqrt, elementwise and control operations,
top-k/sort, and runtime overhead are excluded from it and disclosed separately. Every read-byte
column is logical unique payload plus explicitly declared LUT reads, not measured DRAM, cache, or
PCIe traffic. The precise candidate components and bound strings are preserved in
`candidate_rerank_accounting.csv`. Support-template existence-oracle rows have no deployable base
selector, so their raw base-selector addition and scale-multiplication cells remain empty in that
CSV and are represented as explicit `null` values in the accounting JSON; their rerank and total
charges remain finite and explicit. The explicitly disclosed diagnostic accounting caveats above
are never used for promotion. The analyzer refuses evidence if a reported value differs.

## H0, deployable approximation, and H4 boundary

| Result class | Activation timing | Status |
|---|---|---|
| Hybrid exact oracle | H0 | Oracle only |
| Best-template choice | H0 teacher utility | Existence oracle only |
| PQ/high-rank direct score | H0 actual activation | H0-late scoring; external fetch follows |
| Direct set predictor | H0 Q2-side features | H0-late scoring; not H4 prefetch |
| Independent A/B/C rerank | H0 after candidate fetch | Primary realistic H0-late rerank |
| Contained-target interaction rerank | H0 after candidate fetch | Expensive systems oracle; not promoted |
| Full-target restricted rerank | H0 teacher with omitted effects | Nondeployable information control |
| H4 candidate predictor | H4 | Not trained or evaluated in this branch |

## Primary-source context, not proof

[ShadowLLM](https://arxiv.org/abs/2406.16635) studies predictor-based contextual sparsity and
reports that criterion predictability/objective choice matters. [DejaVu](https://arxiv.org/abs/2310.17157)
supports predicting input-dependent model structures. [SOFT top-k](https://arxiv.org/abs/2002.06504)
and [SIMPLE](https://arxiv.org/abs/2210.01941) motivate differentiable surrogates for discrete
selection; they do not remove the need for hard-cardinality evaluation. [AQLM](https://arxiv.org/abs/2401.06118)
and [GPTVQ](https://arxiv.org/abs/2402.15319) support activation/Hessian-aware vector-quantization
objectives. None of these papers proves that this activation-dependent MXFP4 streaming selector
works, and no latency number is imported from them.

## Reproducibility and interpretation

The canonical decision is `set_utility_promotions.json`. `analysis_manifest.json` hashes every
input, generated artifact, and the analyzer wrapper/core sources using repository-relative paths.
The 14 generated CSVs (seven frontier tables, three gate tables, provenance, coverage, selector
accounting, and candidate-rerank accounting) and all plots are descriptive views of the same
immutable validation rows. The old published test split has already informed the research direction and
is deliberately absent; any promoted method requires a newly captured, sealed holdout.
"""


def analyze(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = validate_evidence_bundle(
        args.config, args.fit_dir, args.validation_dir, EXPERIMENT,
    )
    accounted = {
        kind: recompute_accounting(kind, frame, evidence.config, evidence.selector_index)
        for kind, frame in evidence.frames.items()
    }
    promotion, gate_tables = freeze_validation_promotions(evidence, accounted)
    tables = _frontier_tables(accounted)
    provenance = training_provenance(evidence.fit_manifest)
    coverage = coverage_summary(accounted)
    accounting = selector_accounting(accounted["selector"])
    rerank_accounting = candidate_rerank_accounting(accounted["candidate"])
    rerank_json_rows = candidate_rerank_json_records(rerank_accounting)

    generated: list[Path] = []
    csvs = {
        **tables,
        "hybrid_promotion_gate": gate_tables["hybrid_gate"],
        "template_promotion_gate": gate_tables["template_gate"],
        "selector_promotion_gate": gate_tables["selector_gate"],
        "training_provenance": provenance,
        "validation_coverage": coverage,
        "selector_compute_storage_accounting": accounting,
        "candidate_rerank_accounting": rerank_accounting,
    }
    for stem, frame in csvs.items():
        path = args.output / f"{stem}.csv"
        frame.to_csv(path, index=False)
        generated.append(path)

    promotion_path = args.output / "set_utility_promotions.json"
    atomic_json(promotion_path, promotion)
    loaded_promotion = json.loads(promotion_path.read_text())
    validate_promotion_payload(loaded_promotion, evidence)
    generated.append(promotion_path)

    accounting_json = args.output / "selector_compute_storage_accounting.json"
    atomic_json(accounting_json, {
        "schema_version": 5,
        "arithmetic_recomputed_by_analyzer": True,
        "physical_interface": {
            "candidate_units": 256, "applied_units": 192,
            "physical_pages": 768, "physical_bytes": 393_216,
            "physical_bpw": 1.0, "logical_bpw": 0.75,
            "page_amplification": 4 / 3,
        },
        "rows": accounting.to_dict("records"),
        "candidate_rerank_rows": rerank_json_rows,
        "candidate_rerank_optional_null_contract": {
            "selector_family": "support_template",
            "fields": list(SUPPORT_TEMPLATE_NULL_ACCOUNTING_FIELDS),
            "reason": SUPPORT_TEMPLATE_NULL_ACCOUNTING_REASON,
            "csv_encoding": "empty_cell_preserving_raw_missing_value",
            "json_encoding": "null",
        },
        "q4_response_resident_payload_contract": {
            "q2_parent_code_bytes_per_unit": 1_024,
            "q2_scale_bytes_per_unit": 128,
            "scale_multiplications_per_unit": 128,
            "candidate_packet_bytes_per_unit": 1_536,
            "candidate_packet_is_suffix_only": True,
            "scale_bytes_deduplicated_when_selector_already_read": True,
        },
        "all_metadata_at_or_below_0_35_bpw": bool((accounting["selector_metadata_bpw"] <= 0.35).all()),
        "all_base_selector_compute_strictly_below_1572864_macs": bool((accounting["selector_compute_macs"] < 1_572_864).all()),
        "promotion_compute_gate_includes_rerank_incremental_macs": True,
        "all_storage_multipliers_below_5x": bool((accounting["storage_multiplier"] < 5.0).all()),
        "audited_selector_accounting": {
            "high_rank_low_bit_output_scale_multiplications_per_invocation": 1_024,
            "mac_contract": (
                "analytical linear/algebraic MACs; nonlinear, top-k/sort, control-flow, "
                "and runtime overhead excluded"
            ),
            "read_byte_contract": (
                "logical unique payload plus declared LUT reads; not measured DRAM/cache/PCIe traffic"
            ),
            "standalone_q2_unit_feature_bytes": 3_072,
            "standalone_abc_score_units": 512,
            "standalone_abc_score_compute_macs": 3_072,
            "standalone_abc_score_macs_per_unit": 6,
            "activation_payload_bytes": 4_096,
            "serialized_abc_payload_charged_exactly": True,
        },
        "known_raw_accounting_caveats": {
            "contained_and_full_target_interaction_paths": {
                "analytical_compute_and_logical_payload_are_precise_lower_bounds": True,
                "q4_response_correction_workspace_euclidean_and_proxy_gram_charged": True,
                "missing_from_charge": (
                    INTERACTION_RERANK_UNACCOUNTED
                ),
                "accounting_bound": INTERACTION_RERANK_ACCOUNTING_BOUND,
                "rerank_non_mac_operations": RERANK_NON_MAC_CONVENTION,
                "candidate_packet_contents": CANDIDATE_PACKET_CONTENTS,
                "promotion_eligible": False,
            },
        },
    })
    generated.append(accounting_json)

    generated.extend(_plot_hybrid(tables["hybrid_oracle_summary"], args.output))
    generated.extend(_plot_templates(tables["template_existence_oracle_summary"], args.output))
    generated.extend(_plot_candidates(tables["candidate_interface_summary"], args.output))
    generated.extend(_plot_accounting(accounted["selector"], args.output))

    report_path = args.output / REPORT_NAME
    atomic_text(
        report_path,
        build_report(evidence, tables, gate_tables, promotion, provenance, coverage, accounting),
    )
    generated.append(report_path)

    input_paths = {
        "config": args.config,
        "fit_facts": args.fit_dir / "fit_facts.json",
        "fit_manifest": args.fit_dir / "set_utility_fit_manifest.json",
        "validation_run_facts": args.validation_dir / "run_facts.json",
        **{
            f"validation_{kind}": args.validation_dir / name
            for kind, name in {
                "hybrid": "hybrid_oracle_frontier.parquet",
                "template": "support_template_frontier.parquet",
                "selector": "pq_high_rank_selector_frontier.parquet",
                "candidate": "candidate_set_rerank_frontier.parquet",
            }.items()
        },
    }
    manifest_path = args.output / "analysis_manifest.json"
    analysis_sources = {
        "analyzer_wrapper": Path(__file__).resolve(),
        "analyzer_core": EXPERIMENT / "src/oracle_study/set_utility_analysis.py",
    }
    atomic_json(manifest_path, {
        "schema_version": 2,
        "run_id": evidence.config["run_id"],
        "analysis_split": "validation",
        "capture_source": "exact_checkpoint",
        "monolithic_capture_archives_physically_loaded_by_runner": True,
        "test_split_and_request_metadata_audited": True,
        "test_metadata_audit": TEST_METADATA_AUDIT,
        "test_scientific_rows_or_values_admitted_or_used": False,
        "h4_predictor_evaluated": False,
        "selector_grid_audit": evidence.selector_grid_audit,
        "inputs": {
            name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in sorted(input_paths.items())
        },
        "analysis_sources": {
            name: {
                "path": path.relative_to(EXPERIMENT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for name, path in sorted(analysis_sources.items())
        },
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(generated)
        },
        "promotion_sha256": sha256(promotion_path),
        "report_sha256": sha256(report_path),
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    analyze(parse_args())


if __name__ == "__main__":
    main()
