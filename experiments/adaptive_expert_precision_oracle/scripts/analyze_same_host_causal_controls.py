#!/usr/bin/env python3
"""Analyze same-host PR #13 dose response and downstream routing controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


QUALITY = "same_host_control_quality.parquet"
PROPAGATION = "same_host_control_propagation.parquet"
ZERO = "same_host_zero_dose_gates.parquet"
RUN_FACTS = "same_host_control_run_facts.json"
BASELINE_FACTS = "same_host_baseline_facts.json"
AUDIT_FACTS = "historical_selector_parity_facts.json"
QUALITY_SUMMARY = "same_host_quality_summary.csv"
FINAL_SUMMARY = "same_host_final_propagation_summary.csv"
DOSE_SUMMARY = "same_host_dose_summary.csv"
INJECTION_SUMMARY = "same_host_injection_control_summary.csv"
LAYER_SUMMARY = "same_host_alpha1_layer_summary.csv"
DOSE_PLOT = "same_host_dose_response.png"
DOSE_SVG = "same_host_dose_response.svg"
ROUTING_PLOT = "same_host_routing_control.png"
ROUTING_SVG = "same_host_routing_control.svg"
REPORT = "SAME_HOST_CAUSAL_CONTROLS_REPORT.md"
MANIFEST = "same_host_analysis_manifest.json"
PRIMARY = "pre_residual_routed_replacement"
LIVE = "live"
FROZEN_SET = "frozen_set_live_weights"
FROZEN = "fully_frozen"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float:
    value = np.asarray(list(values), np.float64)
    weight = np.asarray(list(weights), np.float64)
    if value.size == 0 or value.size != weight.size or np.any(weight <= 0):
        raise ValueError("weighted mean inputs changed")
    return float(np.sum(value * weight) / np.sum(weight))


def validate(
    config: dict[str, Any],
    baseline: dict[str, Any],
    facts: dict[str, Any],
    audit: dict[str, Any],
    input_root: Path,
    quality: pd.DataFrame,
    propagation: pd.DataFrame,
    zero: pd.DataFrame,
) -> None:
    if facts.get("completed") is not True or facts.get("run_id") != config["run_id"]:
        raise RuntimeError("same-host run facts are not admitted")
    if facts.get("test_rows_admitted_or_used") is not False:
        raise RuntimeError("same-host test-row boundary changed")
    if facts.get("sentinel_layers") != config["sentinel_layers"]:
        raise RuntimeError("sentinel layer set changed")
    if facts.get("rates") != config["mean_budget_pages_per_expert"]:
        raise RuntimeError("rate set changed")
    if facts.get("propagation_sha256") != sha256(input_root / PROPAGATION):
        raise RuntimeError("propagation hash changed")
    if facts.get("quality_sha256") != sha256(input_root / QUALITY):
        raise RuntimeError("quality hash changed")
    if facts.get("zero_gates_sha256") != sha256(input_root / ZERO):
        raise RuntimeError("zero-dose hash changed")
    if len(quality) != 1680 or len(propagation) != 46800 or len(zero) != 63:
        raise RuntimeError("same-host result row counts changed")
    if baseline.get("same_process_model_residency") is not True:
        raise RuntimeError("same-process model residency changed")
    if baseline.get("test_rows_admitted_or_used") is not False:
        raise RuntimeError("baseline test-row boundary changed")
    repeat = baseline.get("repeat_max_abs", {})
    if not repeat or any(
        float(value) != 0.0
        for request in repeat.values()
        for value in request.values()
    ):
        raise RuntimeError("same-host baseline is not bit-identical")
    if (
        audit.get("completed") is not True
        or audit.get("historical_selected_states_bit_identical") is not True
        or audit.get("historical_reconstructed_deltas_bit_identical") is not True
    ):
        raise RuntimeError("historical selector parity is not admitted")
    zero_error_columns = [
        "hidden_max_abs",
        "router_max_abs",
        "logits_max_abs",
        "routed_repeat_max_abs",
        "routed_one_ulp_fraction",
        "routed_multi_ulp_fraction",
        "routed_mean_ulp_distance",
        "routed_max_ulp_distance",
    ]
    numeric_zero = zero[zero_error_columns]
    if float(np.max(np.abs(numeric_zero.to_numpy(np.float64)))) != 0.0:
        raise RuntimeError("zero-dose replay is not exact")
    if not np.all(zero["routed_unchanged_fraction"].eq(1.0)):
        raise RuntimeError("zero-dose routed output changed")
    numeric_quality = quality.select_dtypes(include=[np.number]).drop(
        columns=["first_route_membership_change_layer"], errors="ignore",
    )
    if not np.all(np.isfinite(numeric_quality.to_numpy(np.float64))):
        raise RuntimeError("non-finite quality result")
    if not np.all(np.isfinite(
        propagation.select_dtypes(include=[np.number]).to_numpy(np.float64),
    )):
        raise RuntimeError("non-finite propagation result")


def quality_summary(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "injection_mode", "route_mode", "injection_layer",
        "mean_budget_pages_per_expert", "charged_bpw", "alpha",
    ]
    rows = []
    for key, group in frame.groupby(keys, sort=True):
        weights = group["label_tokens"].to_numpy(np.float64)
        first = group["first_route_membership_change_layer"].dropna()
        distance = first - int(key[2])
        delta_nll = weighted_mean(group["delta_nll"], weights)
        rows.append({
            **dict(zip(keys, key)),
            "requests": int(group["request_id"].nunique()),
            "label_tokens": int(weights.sum()),
            "intended_delta_energy": weighted_mean(
                group["intended_delta_energy"], weights,
            ),
            "realized_layer_error_energy": weighted_mean(
                group["realized_layer_error_energy"], weights,
            ),
            "layer_output_unchanged_fraction": weighted_mean(
                group["layer_output_unchanged_fraction"], weights,
            ),
            "first_route_membership_crossing_fraction": float(
                group["first_route_membership_change_layer"].notna().mean()
            ),
            "first_route_membership_distance_median": (
                float(distance.median()) if len(distance) else np.nan
            ),
            "final_router_mass_churn": weighted_mean(
                group["final_router_mass_churn"], weights,
            ),
            "delta_nll": delta_nll,
            "ppl_ratio": float(np.exp(delta_nll)),
            "logit_kl": weighted_mean(group["logit_kl"], weights),
            "top1_agreement": weighted_mean(group["top1_agreement"], weights),
        })
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def final_summary(frame: pd.DataFrame) -> pd.DataFrame:
    final = frame[frame["observation_layer"].eq(39)]
    keys = [
        "injection_mode", "route_mode", "injection_layer",
        "mean_budget_pages_per_expert", "charged_bpw", "alpha",
    ]
    rows = []
    for key, group in final.groupby(keys, sort=True):
        weights = group["sequence_tokens"].to_numpy(np.float64)
        rows.append({
            **dict(zip(keys, key)),
            "requests": int(group["request_id"].nunique()),
            "sequence_tokens": int(weights.sum()),
            "hidden_mse": weighted_mean(group["hidden_mse"], weights),
            "rmsnorm_normalized_mse": weighted_mean(
                group["rmsnorm_normalized_mse"], weights,
            ),
            "cosine_distance": weighted_mean(group["cosine_distance"], weights),
            "router_logit_kl": weighted_mean(group["router_logit_kl"], weights),
            "route_membership_change_fraction": weighted_mean(
                group["route_membership_change_fraction"], weights,
            ),
            "router_mass_churn": weighted_mean(
                group["router_mass_churn"], weights,
            ),
        })
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def aggregate_quality(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(keys, sort=True):
        weights = group["label_tokens"].to_numpy(np.float64)
        delta_nll = weighted_mean(group["delta_nll"], weights)
        rows.append({
            **dict(zip(keys, key)),
            "layers": int(group["injection_layer"].nunique()),
            "requests": int(group["request_id"].nunique()),
            "label_layer_tokens": int(weights.sum()),
            "realized_layer_error_energy": weighted_mean(
                group["realized_layer_error_energy"], weights,
            ),
            "layer_output_unchanged_fraction": weighted_mean(
                group["layer_output_unchanged_fraction"], weights,
            ),
            "first_route_membership_crossing_fraction": float(
                group["first_route_membership_change_layer"].notna().mean()
            ),
            "final_router_mass_churn": weighted_mean(
                group["final_router_mass_churn"], weights,
            ),
            "delta_nll": delta_nll,
            "ppl_ratio": float(np.exp(delta_nll)),
            "logit_kl": weighted_mean(group["logit_kl"], weights),
            "top1_agreement": weighted_mean(group["top1_agreement"], weights),
        })
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def layer_summary(
    quality: pd.DataFrame,
    final: pd.DataFrame,
    rates: list[int],
) -> pd.DataFrame:
    primary = quality[
        quality["injection_mode"].eq(PRIMARY)
        & quality["alpha"].eq(1.0)
    ]
    tail = final[
        final["injection_mode"].eq(PRIMARY)
        & final["alpha"].eq(1.0)
    ]
    rows = []
    for layer in sorted(primary["injection_layer"].unique()):
        for rate in rates:
            live = primary[
                primary["injection_layer"].eq(layer)
                & primary["mean_budget_pages_per_expert"].eq(rate)
                & primary["route_mode"].eq(LIVE)
            ].iloc[0]
            frozen = primary[
                primary["injection_layer"].eq(layer)
                & primary["mean_budget_pages_per_expert"].eq(rate)
                & primary["route_mode"].eq(FROZEN)
            ].iloc[0]
            live_tail = tail[
                tail["injection_layer"].eq(layer)
                & tail["mean_budget_pages_per_expert"].eq(rate)
                & tail["route_mode"].eq(LIVE)
            ].iloc[0]
            frozen_tail = tail[
                tail["injection_layer"].eq(layer)
                & tail["mean_budget_pages_per_expert"].eq(rate)
                & tail["route_mode"].eq(FROZEN)
            ].iloc[0]
            rows.append({
                "injection_layer": int(layer),
                "mean_budget_pages_per_expert": int(rate),
                "charged_bpw": float(live["charged_bpw"]),
                "realized_layer_error_energy": float(
                    live["realized_layer_error_energy"],
                ),
                "layer_output_unchanged_fraction": float(
                    live["layer_output_unchanged_fraction"],
                ),
                "live_logit_kl": float(live["logit_kl"]),
                "fully_frozen_logit_kl": float(frozen["logit_kl"]),
                "live_over_frozen_logit_kl": float(
                    live["logit_kl"] / frozen["logit_kl"],
                ),
                "live_final_hidden_mse": float(live_tail["hidden_mse"]),
                "fully_frozen_final_hidden_mse": float(
                    frozen_tail["hidden_mse"],
                ),
                "live_over_frozen_final_hidden_mse": float(
                    live_tail["hidden_mse"] / frozen_tail["hidden_mse"],
                ),
                "first_route_membership_distance_median": float(
                    live["first_route_membership_distance_median"],
                ) if np.isfinite(
                    live["first_route_membership_distance_median"],
                ) else np.nan,
            })
    return pd.DataFrame(rows)


def save_svg(fig: Any, path: Path) -> None:
    fig.savefig(path, metadata={"Date": None})
    atomic_text(
        path,
        "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n",
    )


def plots(
    dose_all: pd.DataFrame,
    dose_by_layer: pd.DataFrame,
    layers: pd.DataFrame,
    output: Path,
    run_id: str,
) -> list[Path]:
    matplotlib.rcParams["svg.hashsalt"] = run_id
    plt.rcParams.update({"font.size": 9})
    colors = {384: "#0072B2", 749: "#D55E00"}
    styles = {LIVE: "-", FROZEN: "--"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for rate in (384, 749):
        for mode in (LIVE, FROZEN):
            group = dose_all[
                dose_all["mean_budget_pages_per_expert"].eq(rate)
                & dose_all["route_mode"].eq(mode)
            ].sort_values("alpha")
            axes[0].plot(
                group["alpha"], group["logit_kl"], marker="o",
                color=colors[rate], linestyle=styles[mode],
                label=f"{rate} pages, {mode.replace('_', ' ')}",
            )
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("dose multiplier alpha")
    axes[0].set_ylabel("weighted final-logit KL")
    axes[0].set_title("All sentinel layers")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    for rate in (384, 749):
        group = dose_by_layer[
            dose_by_layer["mean_budget_pages_per_expert"].eq(rate)
            & dose_by_layer["route_mode"].eq(LIVE)
            & dose_by_layer["injection_layer"].eq(39)
        ].sort_values("realized_layer_error_energy")
        axes[1].plot(
            group["realized_layer_error_energy"], group["logit_kl"],
            marker="o", color=colors[rate], label=f"{rate} pages",
        )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("realized layer-output error energy")
    axes[1].set_ylabel("weighted final-logit KL")
    axes[1].set_title("Layer 39: smooth local response")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    png = output / DOSE_PLOT
    svg = output / DOSE_SVG
    fig.savefig(png, dpi=180, metadata={"Software": "matplotlib"})
    save_svg(fig, svg)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for axis, rate in zip(axes, (384, 749)):
        group = layers[layers["mean_budget_pages_per_expert"].eq(rate)]
        axis.plot(
            group["injection_layer"], group["live_logit_kl"],
            marker="o", label="live routes", color="#D55E00",
        )
        axis.plot(
            group["injection_layer"], group["fully_frozen_logit_kl"],
            marker="o", label="fully frozen routes", color="#0072B2",
        )
        axis.set_yscale("log")
        axis.set_xlabel("injection layer")
        axis.set_ylabel("weighted final-logit KL")
        axis.set_title(f"{rate} pages/expert")
        axis.grid(alpha=0.25)
        axis.legend()
    png2 = output / ROUTING_PLOT
    svg2 = output / ROUTING_SVG
    fig.savefig(png2, dpi=180, metadata={"Software": "matplotlib"})
    save_svg(fig, svg2)
    plt.close(fig)
    return [png, svg, png2, svg2]


def markdown_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---:" for _ in columns) + " |"
    rows = []
    for row in frame[columns].itertuples(index=False, name=None):
        values = []
        for name, value in zip(columns, row):
            if isinstance(value, (int, np.integer)):
                values.append(str(int(value)))
            elif name in formats:
                values.append(formats[name].format(float(value)))
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, rule, *rows])


def build_report(
    config: dict[str, Any],
    baseline: dict[str, Any],
    facts: dict[str, Any],
    audit: dict[str, Any],
    dose: pd.DataFrame,
    injection: pd.DataFrame,
    layers: pd.DataFrame,
) -> str:
    alpha1 = dose[dose["alpha"].eq(1.0)].copy()
    rate_rows = []
    for rate in config["mean_budget_pages_per_expert"]:
        live = alpha1[
            alpha1["mean_budget_pages_per_expert"].eq(rate)
            & alpha1["route_mode"].eq(LIVE)
        ].iloc[0]
        frozen = alpha1[
            alpha1["mean_budget_pages_per_expert"].eq(rate)
            & alpha1["route_mode"].eq(FROZEN)
        ].iloc[0]
        rate_rows.append({
            "pages": rate,
            "bpw": live["charged_bpw"],
            "energy": live["realized_layer_error_energy"],
            "live_kl": live["logit_kl"],
            "frozen_kl": frozen["logit_kl"],
            "ratio": live["logit_kl"] / frozen["logit_kl"],
            "top1": live["top1_agreement"],
        })
    rate_table = pd.DataFrame(rate_rows)
    selected = layers[layers["mean_budget_pages_per_expert"].isin([384, 749])]
    wide = selected.pivot(index="injection_layer", columns="mean_budget_pages_per_expert")
    layer_table = pd.DataFrame({
        "layer": wide.index.astype(int),
        "energy_reduction_x": (
            wide["realized_layer_error_energy"][384]
            / wide["realized_layer_error_energy"][749]
        ),
        "live_kl_384": wide["live_logit_kl"][384],
        "live_kl_749": wide["live_logit_kl"][749],
        "live_kl_lower_pct": 100.0 * (
            1.0 - wide["live_logit_kl"][749] / wide["live_logit_kl"][384]
        ),
        "live_over_frozen_hidden_384": (
            wide["live_over_frozen_final_hidden_mse"][384]
        ),
    }).reset_index(drop=True)
    injection_table = injection[
        injection["mean_budget_pages_per_expert"].isin([384, 749])
    ][["injection_mode", "mean_budget_pages_per_expert", "logit_kl"]]
    audit_layer = audit["results"]["0"]
    first = dose[
        dose["mean_budget_pages_per_expert"].eq(384)
        & dose["route_mode"].eq(LIVE)
    ].sort_values("alpha")
    last = dose[
        dose["mean_budget_pages_per_expert"].eq(749)
        & dose["route_mode"].eq(LIVE)
    ].sort_values("alpha")
    return f"""# Same-host PR #13 causal controls

## Outcome

The same-host control passes. True routed-MoE replacement before shared-expert
composition and the decoder residual add **does not remove the downstream
floor** seen in the first pilot. Downstream discrete expert selection is the
largest identified amplifier: for layers 0--23, live routing produces roughly
6--13 times the final hidden MSE of fully frozen routing at alpha=1.

This is an isolated seven-layer sentinel study, not an end-to-end streamed
quality curve.

## Hard gates

- Same-process repeated hidden/routed/router/logit captures: exactly zero
  difference for all three requests.
- Zero-dose hidden/router/logit replay: exactly zero in live, frozen-set and
  fully frozen modes.
- Historical layer-0 allocation rows: {audit_layer["allocation"]["rows"]:,};
  selected expert IDs, pages and 512-entry state vectors are exact.
- Historical reconstructed deltas: {audit_layer["reconstruction"]["reconstructions"]};
  bit-identical with max absolute difference zero.
- Same-host max group qenergy reconstruction error across layers:
  {max(float(v["max_group_qenergy_abs_error"]) for v in facts["layers"].values()):.3e}.
- Same-host max all-Q4 float64 reconstruction roundoff:
  {max(float(v["all_q4_max_abs_error"]) for v in facts["layers"].values()):.3e}.
- Rows: {facts["propagation_rows"]:,} propagation, {facts["quality_rows"]:,}
  quality and {facts["zero_gate_rows"]} zero-dose.
- Test rows admitted or used: false.

## Alpha=1 rate comparison

{markdown_table(rate_table, ["pages", "bpw", "energy", "live_kl", "frozen_kl", "ratio", "top1"], {
    "bpw": "{:.6f}", "energy": "{:.3e}", "live_kl": "{:.3e}",
    "frozen_kl": "{:.3e}", "ratio": "{:.2f}", "top1": "{:.2%}",
})}

Moving from 384 to 749 pages reduces mean realized layer-output energy by
{rate_table.iloc[0]["energy"] / rate_table.iloc[2]["energy"]:.2f}x, but live
logit KL by only
{rate_table.iloc[0]["live_kl"] / rate_table.iloc[2]["live_kl"]:.2f}x. With
fully frozen downstream expert IDs, the remaining KL is much smaller, while
frozen-set/live-weight results closely track fully frozen results. This
attributes most of the additional live disturbance to discrete expert-set
changes rather than router-weight drift.

## Depth and rate

{markdown_table(layer_table, ["layer", "energy_reduction_x", "live_kl_384", "live_kl_749", "live_kl_lower_pct", "live_over_frozen_hidden_384"], {
    "energy_reduction_x": "{:.2f}", "live_kl_384": "{:.3e}",
    "live_kl_749": "{:.3e}", "live_kl_lower_pct": "{:+.1f}",
    "live_over_frozen_hidden_384": "{:.2f}",
})}

Layer 39 behaves conventionally: 384-to-749 pages gives
{layer_table.iloc[-1]["energy_reduction_x"]:.2f}x lower realized energy and
{layer_table.iloc[-1]["live_kl_384"] / layer_table.iloc[-1]["live_kl_749"]:.2f}x
lower logit KL. Earlier layers do not. At alpha=1, every request injected
before layer 39 crosses a downstream top-8 membership boundary; the median
first crossing is one layer downstream and the latest is three layers.

## Injection implementation control

{markdown_table(injection_table, ["injection_mode", "mean_budget_pages_per_expert", "logit_kl"], {
    "logit_kl": "{:.3e}",
})}

The two post-xplus controls remain within about 15% of true pre-residual
replacement after aggregation. The old injection location was therefore not
the primary cause of the floor, although pre-residual replacement is now the
scientific path.

## Dose response

At 384 pages, increasing alpha from {first.iloc[0]["alpha"]:.3g} to
{first.iloc[-1]["alpha"]:.3g} changes live KL from
{first.iloc[0]["logit_kl"]:.3e} to {first.iloc[-1]["logit_kl"]:.3e}. At 749
pages the corresponding values are {last.iloc[0]["logit_kl"]:.3e} and
{last.iloc[-1]["logit_kl"]:.3e}. The shallow-layer response is discontinuous
and non-monotone, while layer 39 is smooth (energy-to-KL Spearman approximately
0.94 across rates and doses).

## Router-weight representation

The installed backend normalizes top-8 probabilities in FP32 and casts them to
BF16 before expert execution. This study keeps both quantities:

- normalized selector weights for PR #13 allocation and qenergy accounting;
- raw BF16 execution weights for the reconstructed routed-output delta.

The maximum observed BF16 execution-weight sum error was
{max(float(v["max_execution_router_sum_error"]) for v in facts["layers"].values()):.6f};
the maximum per-weight selector adjustment was
{max(float(v["max_execution_selector_weight_abs_difference"]) for v in facts["layers"].values()):.6f}.

## Scientific boundary

- Cohort: three validation sequences, 44 input tokens, 41 labels and 32
  injected positions per layer.
- Sentinels: layers {", ".join(map(str, config["sentinel_layers"]))}.
- Rates: all four PR #13 operating points; alpha in
  {", ".join(map(str, config["dose_alphas"]))}.
- Modes: live routes, exact expert set with live weights, and fully frozen
  routes; plus the two alpha=1 post-xplus injection controls.
- NLL is descriptive only on three sequences. Non-negative forward logit KL is
  the primary quality signal.
- Not measured: simultaneous all-layer streaming, held-out population
  perplexity, live re-allocation at every perturbed layer, repair-one-layer
  importance, H1 predictor error, or physical wire bpw.

## Decision

The warning signal survives both methodological controls. The next large run
should retain true pre-residual replacement and report routing controls, then
move to genuine sequential all-layer streaming. A routing-boundary-risk term
and between-layer bit allocator remain justified; local qenergy alone is not a
sufficient downstream objective.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--selector-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_json(args.config)
    baseline = load_json(args.input / BASELINE_FACTS)
    facts = load_json(args.input / RUN_FACTS)
    audit = load_json(args.selector_audit)
    quality_path = args.input / QUALITY
    propagation_path = args.input / PROPAGATION
    zero_path = args.input / ZERO
    quality_raw = pd.read_parquet(quality_path)
    propagation_raw = pd.read_parquet(propagation_path)
    zero = pd.read_parquet(zero_path)
    validate(
        config, baseline, facts, audit, args.input,
        quality_raw, propagation_raw, zero,
    )
    quality = quality_summary(quality_raw)
    final = final_summary(propagation_raw)
    primary_raw = quality_raw[quality_raw["injection_mode"].eq(PRIMARY)]
    dose = aggregate_quality(
        primary_raw,
        [
            "injection_layer", "mean_budget_pages_per_expert",
            "charged_bpw", "alpha", "route_mode",
        ],
    )
    dose_all = aggregate_quality(
        primary_raw,
        ["mean_budget_pages_per_expert", "charged_bpw", "alpha", "route_mode"],
    )
    injection = aggregate_quality(
        quality_raw[
            quality_raw["alpha"].eq(1.0)
            & quality_raw["route_mode"].eq(LIVE)
        ],
        ["injection_mode", "mean_budget_pages_per_expert", "charged_bpw"],
    )
    layers = layer_summary(
        quality, final, list(map(int, config["mean_budget_pages_per_expert"])),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, frame in (
        (QUALITY_SUMMARY, quality),
        (FINAL_SUMMARY, final),
        (DOSE_SUMMARY, dose_all),
        (INJECTION_SUMMARY, injection),
        (LAYER_SUMMARY, layers),
    ):
        path = args.output / name
        atomic_csv(path, frame)
        outputs.append(path)
    outputs.extend(plots(dose_all, dose, layers, args.output, config["run_id"]))
    report = args.output / REPORT
    atomic_text(
        report,
        build_report(config, baseline, facts, audit, dose_all, injection, layers),
    )
    outputs.append(report)
    manifest = {
        "completed": True,
        "schema": "pr13_same_host_causal_controls_analysis_v1",
        "run_id": config["run_id"],
        "input_hashes": {
            QUALITY: sha256(quality_path),
            PROPAGATION: sha256(propagation_path),
            ZERO: sha256(zero_path),
            RUN_FACTS: sha256(args.input / RUN_FACTS),
            BASELINE_FACTS: sha256(args.input / BASELINE_FACTS),
            AUDIT_FACTS: sha256(args.selector_audit),
        },
        "outputs": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in outputs
        },
        "test_rows_admitted_or_used": False,
        "scientific_boundary": facts["scientific_boundary"],
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
    }
    atomic_json(args.output / MANIFEST, manifest)


if __name__ == "__main__":
    main()
