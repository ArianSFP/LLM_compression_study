#!/usr/bin/env python3
"""Focused exact-prefix Experiment A diagnostics for D1 allocator failures.

This runner deliberately remains inside Experiment A.  It replays a frozen,
deterministically selected 24-token/layer panel through the exact cached decode
tail, records the complete downstream route trajectory, and evaluates
non-realizable interpolation controls between the adjacent fixed-D1 rate
points.  It does not implement S0, page-effect prediction, certification, or
any other Experiment B controller component.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(EXPERIMENT / "scripts")]

from oracle_study.causal_replay import qenergy_damage  # noqa: E402
from oracle_study.d1_decode import cache_state_metrics, clone_decode_cache  # noqa: E402
from oracle_study.d1_decode_tail import (  # noqa: E402
    DECODE_POLICIES,
    FIXED_D1_POLICY,
    PR13_POLICY,
    classified_cache_metrics,
    current_token_quality_metrics,
    sha256,
    validate_cached_decode_tail_config,
)
from run_d1_downstream_tail_kl import (  # noqa: E402
    CachedDecodeObserver,
    DecodeObservation,
    _gpu_gate,
    _live_policy_delta,
    _load_model,
    _np,
    _observation_maxima,
    _policy_banks,
    _prepare_live_reconstruction,
    _run_candidate,
    _zero_metrics,
)
from run_same_host_causal_controls import (  # noqa: E402
    _encoded_requests,
    atomic_json,
    atomic_parquet,
)


SCHEMA = "pr13_d1_experiment_a_diagnostic_panel_v1"
CONFIG_SCHEMA = "pr13_d1_experiment_a_diagnostic_panel_config_v1"
PANEL = "d1_experiment_a_diagnostic_panel.parquet"
PANEL_FACTS = "d1_experiment_a_diagnostic_panel_facts.json"
QUALITY = "d1_experiment_a_diagnostic_quality.parquet"
ROUTES = "d1_experiment_a_diagnostic_routes.parquet"
ZERO = "d1_experiment_a_diagnostic_zero_gates.parquet"
RUN_FACTS = "d1_experiment_a_diagnostic_run_facts.json"
HOST_FACTS = "d1_experiment_a_diagnostic_host_facts.json"

EXPECTED_LAYERS = (0, 1, 4, 6, 12, 23)
EXPECTED_RATES = (360, 384, 725, 749)
EXPECTED_RATE_PAIRS = ((360, 384), (725, 749))
EXPECTED_ROUTE_MODES = ("live", "frozen_set_live_weights", "fully_frozen")
INTERPOLATION_LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)

IDENTITY = ["injection_layer", "group", "request_id", "position"]
QUALITY_IDENTITY = [*IDENTITY, "candidate_id", "route_mode"]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _resolve_experiment_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else EXPERIMENT / path


def _stable_unique_candidates(
    rows: pd.DataFrame,
    selected: set[tuple[int, int, str, int]],
    count: int,
) -> pd.DataFrame:
    accepted = []
    for row in rows.itertuples(index=False):
        key = (
            int(row.injection_layer),
            int(row.group),
            str(row.request_id),
            int(row.position),
        )
        if key in selected:
            continue
        selected.add(key)
        accepted.append(row._asdict())
        if len(accepted) == int(count):
            break
    if len(accepted) != int(count):
        raise RuntimeError(
            f"diagnostic panel category supplied {len(accepted)} of {count} unique rows"
        )
    return pd.DataFrame(accepted)


def _live_policy_quality(quality: pd.DataFrame, policy: str) -> pd.DataFrame:
    result = quality[
        quality["route_mode"].astype(str).eq("live")
        & quality["policy"].astype(str).eq(str(policy))
    ].copy()
    if result.duplicated([*IDENTITY, "rate_pages_per_expert"]).any():
        raise ValueError("source quality contains duplicate live policy rows")
    return result


def _d1_crossings(propagation: pd.DataFrame, policy: str) -> pd.DataFrame:
    columns = [*IDENTITY, "rate_pages_per_expert", "route_membership_change_fraction"]
    result = propagation[
        propagation["route_mode"].astype(str).eq("live")
        & propagation["policy"].astype(str).eq(str(policy))
        & propagation["distance_from_injection"].eq(1)
    ][columns].copy()
    result["d1_crossed"] = result["route_membership_change_fraction"].gt(0.0)
    if result.duplicated([*IDENTITY, "rate_pages_per_expert"]).any():
        raise ValueError("source propagation contains duplicate D1 policy rows")
    return result.drop(columns="route_membership_change_fraction")


def _adjacent_reversal_candidates(
    quality: pd.DataFrame,
    rate_pairs: Sequence[tuple[int, int]],
) -> pd.DataFrame:
    fixed = _live_policy_quality(quality, FIXED_D1_POLICY)
    rows: list[pd.DataFrame] = []
    values = [
        *IDENTITY,
        "logit_kl",
        "live_selected_local_qenergy_damage",
    ]
    for low, high in rate_pairs:
        low_rows = fixed[fixed["rate_pages_per_expert"].eq(int(low))][values]
        high_rows = fixed[fixed["rate_pages_per_expert"].eq(int(high))][values]
        paired = low_rows.merge(
            high_rows,
            on=IDENTITY,
            suffixes=("_low", "_high"),
            validate="one_to_one",
        )
        paired["evidence_rate"] = int(high)
        paired["low_rate"] = int(low)
        paired["high_rate"] = int(high)
        paired["selection_score"] = paired["logit_kl_high"] - paired["logit_kl_low"]
        paired["local_qenergy_change"] = (
            paired["live_selected_local_qenergy_damage_high"]
            - paired["live_selected_local_qenergy_damage_low"]
        )
        paired = paired[
            paired["selection_score"].gt(0.0) & paired["local_qenergy_change"].lt(0.0)
        ].copy()
        paired["selection_class"] = "adjacent_rate_reversal"
        paired["selection_detail"] = (
            "fixed_d1_"
            + paired["low_rate"].astype(str)
            + "_to_"
            + paired["high_rate"].astype(str)
        )
        rows.append(paired)
    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    return result.sort_values(
        ["selection_score", "high_rate", "injection_layer", "group"],
        ascending=[False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)


def _crossing_event_candidates(
    quality: pd.DataFrame,
    propagation: pd.DataFrame,
    event: str,
) -> pd.DataFrame:
    if event not in {"prevented", "introduced"}:
        raise ValueError(event)
    fixed_quality = _live_policy_quality(quality, FIXED_D1_POLICY)
    pr13_quality = _live_policy_quality(quality, PR13_POLICY)
    columns = [*IDENTITY, "rate_pages_per_expert", "logit_kl"]
    paired = pr13_quality[columns].merge(
        fixed_quality[columns],
        on=[*IDENTITY, "rate_pages_per_expert"],
        suffixes=("_pr13", "_fixed"),
        validate="one_to_one",
    )
    pr13_cross = _d1_crossings(propagation, PR13_POLICY).rename(
        columns={"d1_crossed": "pr13_d1_crossed"}
    )
    fixed_cross = _d1_crossings(propagation, FIXED_D1_POLICY).rename(
        columns={"d1_crossed": "fixed_d1_crossed"}
    )
    paired = paired.merge(
        pr13_cross,
        on=[*IDENTITY, "rate_pages_per_expert"],
        validate="one_to_one",
    ).merge(
        fixed_cross,
        on=[*IDENTITY, "rate_pages_per_expert"],
        validate="one_to_one",
    )
    if event == "prevented":
        mask = paired["pr13_d1_crossed"] & ~paired["fixed_d1_crossed"]
        paired["selection_score"] = paired["logit_kl_pr13"] - paired["logit_kl_fixed"]
    else:
        mask = ~paired["pr13_d1_crossed"] & paired["fixed_d1_crossed"]
        paired["selection_score"] = paired["logit_kl_fixed"] - paired["logit_kl_pr13"]
    paired = paired[mask].copy()
    paired["evidence_rate"] = paired["rate_pages_per_expert"].astype(int)
    paired["low_rate"] = -1
    paired["high_rate"] = -1
    paired["local_qenergy_change"] = 0.0
    paired["selection_class"] = f"fixed_vs_pr13_{event}"
    paired["selection_detail"] = paired["selection_class"]
    return paired.sort_values(
        ["selection_score", "rate_pages_per_expert", "injection_layer", "group"],
        ascending=[False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)


def _quiet_candidates(
    quality: pd.DataFrame,
    propagation: pd.DataFrame,
    layers: Sequence[int],
) -> pd.DataFrame:
    fixed = _live_policy_quality(quality, FIXED_D1_POLICY)
    pr13 = _live_policy_quality(quality, PR13_POLICY)
    columns = [*IDENTITY, "rate_pages_per_expert", "logit_kl"]
    paired = pr13[columns].merge(
        fixed[columns],
        on=[*IDENTITY, "rate_pages_per_expert"],
        suffixes=("_pr13", "_fixed"),
        validate="one_to_one",
    )
    paired["absolute_policy_delta"] = (
        paired["logit_kl_fixed"] - paired["logit_kl_pr13"]
    ).abs()
    crossings = pd.concat(
        [
            _d1_crossings(propagation, policy).assign(policy=policy)
            for policy in (PR13_POLICY, FIXED_D1_POLICY)
        ],
        ignore_index=True,
    )
    crossing_any = crossings.groupby(IDENTITY, sort=False)["d1_crossed"].any()
    aggregate = (
        paired.groupby(IDENTITY, sort=False)
        .agg(
            selection_score=("absolute_policy_delta", "max"),
            mean_absolute_policy_delta=("absolute_policy_delta", "mean"),
        )
        .reset_index()
    )
    aggregate = aggregate.merge(
        crossing_any.rename("any_d1_crossing").reset_index(),
        on=IDENTITY,
        validate="one_to_one",
    )
    aggregate = aggregate[~aggregate["any_d1_crossing"]].copy()
    aggregate["selection_class"] = "quiet_control"
    aggregate["selection_detail"] = "all_rates_pr13_and_fixed_d1_safe"
    aggregate["evidence_rate"] = -1
    aggregate["low_rate"] = -1
    aggregate["high_rate"] = -1
    aggregate["local_qenergy_change"] = 0.0
    layer_set = set(map(int, layers))
    if set(aggregate["injection_layer"].astype(int).unique()) - layer_set:
        raise ValueError("quiet candidates contain an unexpected layer")
    return aggregate.sort_values(
        ["injection_layer", "selection_score", "group"],
        ascending=[True, True, True],
        kind="stable",
    ).reset_index(drop=True)


def select_diagnostic_panel(
    quality: pd.DataFrame,
    propagation: pd.DataFrame,
    *,
    layers: Sequence[int] = EXPECTED_LAYERS,
    rate_pairs: Sequence[tuple[int, int]] = EXPECTED_RATE_PAIRS,
    quiet_per_layer: int = 2,
    adjacent_reversal_count: int = 4,
    prevented_count: int = 4,
    introduced_count: int = 4,
) -> pd.DataFrame:
    """Select a deterministic 24 unique token/layer diagnostic panel.

    Quiet controls are reserved first so every sentinel layer has two.  The
    remaining slots are the strongest non-overlapping adjacent-rate
    reversals, PR13 crossings prevented by fixed D1, and crossings introduced
    by fixed D1.  Selection uses only already-finalized Experiment A evidence.
    """

    layers = tuple(map(int, layers))
    selected: set[tuple[int, int, str, int]] = set()
    parts: list[pd.DataFrame] = []
    quiet = _quiet_candidates(quality, propagation, layers)
    for layer in layers:
        candidates = quiet[quiet["injection_layer"].eq(layer)]
        part = _stable_unique_candidates(candidates, selected, int(quiet_per_layer))
        parts.append(part)

    for candidates, count in (
        (_adjacent_reversal_candidates(quality, rate_pairs), adjacent_reversal_count),
        (
            _crossing_event_candidates(quality, propagation, "prevented"),
            prevented_count,
        ),
        (
            _crossing_event_candidates(quality, propagation, "introduced"),
            introduced_count,
        ),
    ):
        parts.append(_stable_unique_candidates(candidates, selected, int(count)))

    panel = pd.concat(parts, ignore_index=True)
    expected = (
        len(layers) * int(quiet_per_layer)
        + int(adjacent_reversal_count)
        + int(prevented_count)
        + int(introduced_count)
    )
    if len(panel) != expected or panel.duplicated(IDENTITY).any():
        raise RuntimeError("diagnostic panel size or token/layer uniqueness changed")
    panel.insert(0, "schema", SCHEMA)
    panel.insert(1, "panel_index", np.arange(len(panel), dtype=np.int64))
    keep = [
        "schema",
        "panel_index",
        *IDENTITY,
        "selection_class",
        "selection_detail",
        "selection_score",
        "evidence_rate",
        "low_rate",
        "high_rate",
        "local_qenergy_change",
    ]
    for column in keep:
        if column not in panel:
            raise RuntimeError(f"panel selection omitted {column}")
    return panel[keep].sort_values("panel_index", kind="stable").reset_index(drop=True)


def _json_array(values: Iterable[Any]) -> str:
    converted = []
    for value in values:
        item = value.item() if isinstance(value, np.generic) else value
        converted.append(item)
    return json.dumps(converted, separators=(",", ":"), allow_nan=False)


def _normalized_sparse(ids: np.ndarray, scores: np.ndarray, experts: int) -> np.ndarray:
    ids = np.asarray(ids, np.int64).reshape(-1)
    scores = np.asarray(scores, np.float64).reshape(-1)
    if ids.shape != (8,) or scores.shape != (8,) or len(set(ids.tolist())) != 8:
        raise ValueError("executed route must contain eight distinct experts")
    if (
        np.any(scores < 0.0)
        or not np.isfinite(scores).all()
        or float(scores.sum()) <= 0.0
    ):
        raise ValueError("executed route scores are invalid")
    sparse = np.zeros(int(experts), np.float64)
    sparse[ids] = scores / float(scores.sum())
    return sparse


def _set_margin(logits: np.ndarray, selected_ids: np.ndarray) -> float:
    logits = np.asarray(logits, np.float64).reshape(-1)
    selected_ids = np.asarray(selected_ids, np.int64).reshape(-1)
    mask = np.ones(len(logits), dtype=bool)
    mask[selected_ids] = False
    return float(np.min(logits[selected_ids]) - np.max(logits[mask]))


def detailed_route_metrics(
    reference_logits: np.ndarray,
    candidate_logits: np.ndarray,
    reference_ids: np.ndarray,
    candidate_ids: np.ndarray,
    reference_scores: np.ndarray,
    candidate_scores: np.ndarray,
    reference_hidden: np.ndarray,
    candidate_hidden: np.ndarray,
) -> dict[str, Any]:
    """Return lossless top-8 membership evidence and continuous D1 metrics."""

    reference_logits = np.asarray(reference_logits, np.float64).reshape(-1)
    candidate_logits = np.asarray(candidate_logits, np.float64).reshape(-1)
    if reference_logits.shape != candidate_logits.shape or reference_logits.size <= 8:
        raise ValueError(
            "router logits must be equal vectors with at least nine experts"
        )
    reference_ids = np.asarray(reference_ids, np.int64).reshape(-1)
    candidate_ids = np.asarray(candidate_ids, np.int64).reshape(-1)
    reference_scores = np.asarray(reference_scores, np.float64).reshape(-1)
    candidate_scores = np.asarray(candidate_scores, np.float64).reshape(-1)
    reference_sparse = _normalized_sparse(
        reference_ids,
        reference_scores,
        reference_logits.size,
    )
    candidate_sparse = _normalized_sparse(
        candidate_ids,
        candidate_scores,
        candidate_logits.size,
    )
    reference_set = set(reference_ids.tolist())
    candidate_set = set(candidate_ids.tolist())
    entered = [int(value) for value in candidate_ids if int(value) not in reference_set]
    left = [int(value) for value in reference_ids if int(value) not in candidate_set]
    reference_rank = np.argsort(reference_logits, kind="stable")[::-1]
    candidate_rank = np.argsort(candidate_logits, kind="stable")[::-1]
    reference_centered = reference_logits - float(reference_logits.mean())
    candidate_centered = candidate_logits - float(candidate_logits.mean())
    logit_error = candidate_centered - reference_centered
    hidden_error = np.asarray(candidate_hidden, np.float64).reshape(-1) - np.asarray(
        reference_hidden, np.float64
    ).reshape(-1)
    return {
        "baseline_top8_ids_json": _json_array(reference_ids),
        "candidate_top8_ids_json": _json_array(candidate_ids),
        "baseline_top8_scores_json": _json_array(reference_scores),
        "candidate_top8_scores_json": _json_array(candidate_scores),
        "baseline_top8_logits_json": _json_array(reference_logits[reference_ids]),
        "candidate_top8_logits_json": _json_array(candidate_logits[candidate_ids]),
        "entered_expert_ids_json": _json_array(entered),
        "left_expert_ids_json": _json_array(left),
        "membership_pairs_changed": len(entered),
        "route_membership_changed": bool(reference_set != candidate_set),
        "route_order_changed": bool(not np.array_equal(reference_ids, candidate_ids)),
        "route_top1_changed": bool(int(reference_ids[0]) != int(candidate_ids[0])),
        "baseline_rank8_rank9_margin": float(
            reference_logits[reference_rank[7]] - reference_logits[reference_rank[8]]
        ),
        "candidate_rank8_rank9_margin": float(
            candidate_logits[candidate_rank[7]] - candidate_logits[candidate_rank[8]]
        ),
        "baseline_executed_set_margin": _set_margin(reference_logits, reference_ids),
        "candidate_executed_set_margin": _set_margin(candidate_logits, candidate_ids),
        "baseline_set_margin_under_candidate_logits": _set_margin(
            candidate_logits,
            reference_ids,
        ),
        "candidate_set_margin_under_baseline_logits": _set_margin(
            reference_logits,
            candidate_ids,
        ),
        "left_reference_routing_mass": (
            float(reference_sparse[left].sum()) if left else 0.0
        ),
        "entered_candidate_routing_mass": (
            float(candidate_sparse[entered].sum()) if entered else 0.0
        ),
        "router_mass_churn": float(
            0.5 * np.abs(reference_sparse - candidate_sparse).sum()
        ),
        "centered_router_logit_mse": float(np.mean(logit_error * logit_error)),
        "centered_router_logit_max_abs": float(
            np.max(np.abs(logit_error), initial=0.0)
        ),
        "hidden_mse": float(np.mean(hidden_error * hidden_error)),
    }


def _bf16_ulp(values: np.ndarray) -> np.ndarray:
    values = np.abs(np.asarray(values, np.float64))
    result = np.full(values.shape, np.ldexp(1.0, -133), np.float64)
    normal = values >= np.ldexp(1.0, -126)
    finite_nonzero = normal & np.isfinite(values) & (values > 0.0)
    if np.any(finite_nonzero):
        exponent = np.floor(np.log2(values[finite_nonzero])).astype(np.int64)
        result[finite_nonzero] = np.exp2(exponent - 7)
    result[np.isinf(values) | np.isnan(values)] = np.nan
    return result


def bf16_injection_accounting(
    reference_routed: np.ndarray,
    planned_delta: np.ndarray,
) -> dict[str, float | int]:
    """Account for the actual BF16 addition used by routed replacement."""

    reference = np.asarray(reference_routed, np.float32).reshape(-1)
    planned = np.asarray(planned_delta, np.float32).reshape(-1)
    if reference.shape != planned.shape or not np.isfinite(planned).all():
        raise ValueError("reference and planned injection must be equal finite vectors")
    reference_bf16 = torch.as_tensor(reference).to(torch.bfloat16).float().numpy()
    target = reference_bf16 + planned
    rounded = torch.as_tensor(target).to(torch.bfloat16).float().numpy()
    effective = rounded - reference_bf16
    error = effective.astype(np.float64) - planned.astype(np.float64)
    ulp = _bf16_ulp(target)
    ratio = np.abs(error) / ulp
    nonzero = planned != 0.0
    collapsed = nonzero & (effective == 0.0)
    planned_norm = float(np.linalg.norm(planned.astype(np.float64)))
    effective_norm = float(np.linalg.norm(effective.astype(np.float64)))
    return {
        "planned_delta_l2": planned_norm,
        "planned_delta_mse": float(np.mean(planned.astype(np.float64) ** 2)),
        "bf16_effective_delta_l2": effective_norm,
        "bf16_effective_delta_mse": float(np.mean(effective.astype(np.float64) ** 2)),
        "bf16_effective_to_planned_norm_ratio": (
            effective_norm / planned_norm if planned_norm else 1.0
        ),
        "bf16_quantization_error_mse": float(np.mean(error * error)),
        "bf16_quantization_error_max_abs": float(np.max(np.abs(error), initial=0.0)),
        "bf16_quantization_error_ulp_mean": float(np.nanmean(ratio)),
        "bf16_quantization_error_ulp_p95": float(np.nanquantile(ratio, 0.95)),
        "bf16_quantization_error_ulp_max": float(np.nanmax(ratio)),
        "planned_below_half_ulp_fraction": float(np.mean(np.abs(planned) < 0.5 * ulp)),
        "planned_nonzero_coordinates": int(nonzero.sum()),
        "bf16_changed_coordinates": int(np.count_nonzero(effective)),
        "bf16_collapsed_nonzero_coordinates": int(collapsed.sum()),
        "bf16_collapsed_nonzero_fraction": (
            float(collapsed.sum() / nonzero.sum()) if np.any(nonzero) else 0.0
        ),
    }


def interpolation_deltas(
    low_delta: np.ndarray,
    high_delta: np.ndarray,
    lambdas: Sequence[float] = INTERPOLATION_LAMBDAS,
) -> list[tuple[str, float, np.ndarray, float]]:
    low = np.asarray(low_delta, np.float32).reshape(-1)
    high = np.asarray(high_delta, np.float32).reshape(-1)
    if low.shape != high.shape:
        raise ValueError("adjacent-rate deltas must have equal shapes")
    result: list[tuple[str, float, np.ndarray, float]] = []
    for value in lambdas:
        lam = float(value)
        if not 0.0 <= lam <= 1.0:
            raise ValueError("interpolation lambda must be in [0,1]")
        result.append(
            ("linear", lam, np.asarray((1.0 - lam) * low + lam * high, np.float32), 1.0)
        )
    low_norm = float(np.linalg.norm(low.astype(np.float64)))
    high_norm = float(np.linalg.norm(high.astype(np.float64)))
    scale = (
        low_norm / high_norm
        if high_norm
        else (1.0 if low_norm == 0.0 else float("nan"))
    )
    if not np.isfinite(scale):
        raise ValueError(
            "cannot rescale a zero high-rate delta to a nonzero low-rate norm"
        )
    result.append(
        ("high_rescaled_to_low_norm", 1.0, np.asarray(high * scale, np.float32), scale)
    )
    return result


def _candidate_id_for_interpolation(low: int, high: int, kind: str, lam: float) -> str:
    if kind == "high_rescaled_to_low_norm":
        suffix = "high_rescaled_to_low_norm"
    else:
        suffix = f"lambda_{lam:g}".replace(".", "p")
    return f"fixed_d1_path_{int(low)}_{int(high)}__{suffix}"


def _diagnostic_config(
    config_path: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    config = load_json(config_path)
    if str(config.get("schema")) != CONFIG_SCHEMA:
        raise ValueError("unexpected Experiment A diagnostic-panel config schema")
    if tuple(map(int, config["injection_layers"])) != EXPECTED_LAYERS:
        raise ValueError("diagnostic injection-layer grid changed")
    if tuple(map(int, config["page_caps"])) != EXPECTED_RATES:
        raise ValueError("diagnostic adjacent-rate grid changed")
    if tuple(map(str, config["policies"])) != tuple(DECODE_POLICIES):
        raise ValueError("diagnostic allocation policy grid changed")
    if tuple(map(str, config["route_modes"])) != EXPECTED_ROUTE_MODES:
        raise ValueError("diagnostic route-mode decomposition changed")
    panel = config["panel"]
    expected_panel = (
        len(EXPECTED_LAYERS) * int(panel["quiet_controls_per_layer"])
        + int(panel["adjacent_rate_reversals"])
        + int(panel["prevented_crossings"])
        + int(panel["introduced_crossings"])
    )
    if int(panel["total_unique_token_layers"]) != 24 or expected_panel != 24:
        raise ValueError("diagnostic panel must freeze exactly 24 unique token/layers")
    pairs = tuple(
        tuple(map(int, pair)) for pair in config["interpolation"]["rate_pairs"]
    )
    lambdas = tuple(map(float, config["interpolation"]["lambdas"]))
    if pairs != EXPECTED_RATE_PAIRS or lambdas != INTERPOLATION_LAMBDAS:
        raise ValueError("diagnostic interpolation grid changed")
    base_path = _resolve_experiment_path(config["base_tail_config"])
    if sha256(base_path) != str(config["base_tail_config_sha256"]):
        raise RuntimeError("immutable base cached-decode tail config changed")
    base = load_json(base_path)
    validate_cached_decode_tail_config(base)
    evidence_root = Path(str(config["source_result_root"]))
    for name, digest in config["source_evidence_sha256"].items():
        path = evidence_root / str(name)
        if not path.is_file() or sha256(path) != str(digest):
            raise RuntimeError(
                f"immutable source Experiment A evidence changed: {name}"
            )
    return config, base_path, base


def _panel_paths(output: Path) -> tuple[Path, Path]:
    return output / PANEL, output / PANEL_FACTS


def _layer_paths(output: Path, layer: int) -> dict[str, Path]:
    root = output / "layers" / f"layer_{int(layer):02d}"
    return {
        "root": root,
        "quality": root / QUALITY,
        "routes": root / ROUTES,
        "zero": root / ZERO,
        "facts": root / "d1_experiment_a_diagnostic_layer_facts.json",
    }


def _completed_layer(paths: Mapping[str, Path], layer: int, panel_sha256: str) -> bool:
    if not paths["facts"].is_file():
        return False
    facts = load_json(paths["facts"])
    if (
        not bool(facts.get("completed"))
        or int(facts.get("injection_layer", -1)) != int(layer)
        or str(facts.get("panel_sha256")) != str(panel_sha256)
    ):
        return False
    for key in ("quality", "routes", "zero"):
        path = paths[key]
        if not path.is_file() or sha256(path) != str(
            facts["files"][path.name]["sha256"]
        ):
            return False
    return True


def panel_phase(args: argparse.Namespace) -> None:
    config, base_path, _ = _diagnostic_config(args.config)
    source = Path(str(config["source_result_root"]))
    quality_path = source / "d1_cached_decode_tail_quality.parquet"
    propagation_path = source / "d1_cached_decode_tail_propagation.parquet"
    quality = pd.read_parquet(quality_path)
    propagation = pd.read_parquet(propagation_path)
    specification = config["panel"]
    panel = select_diagnostic_panel(
        quality,
        propagation,
        layers=config["injection_layers"],
        rate_pairs=[tuple(pair) for pair in config["interpolation"]["rate_pairs"]],
        quiet_per_layer=int(specification["quiet_controls_per_layer"]),
        adjacent_reversal_count=int(specification["adjacent_rate_reversals"]),
        prevented_count=int(specification["prevented_crossings"]),
        introduced_count=int(specification["introduced_crossings"]),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    panel_path, facts_path = _panel_paths(args.output)
    atomic_parquet(panel_path, panel)
    facts = {
        "completed": True,
        "schema": SCHEMA,
        "config_sha256": sha256(args.config),
        "base_tail_config": str(base_path),
        "base_tail_config_sha256": sha256(base_path),
        "source_quality_sha256": sha256(quality_path),
        "source_propagation_sha256": sha256(propagation_path),
        "panel_sha256": sha256(panel_path),
        "rows": len(panel),
        "unique_token_layers": int(panel[IDENTITY].drop_duplicates().shape[0]),
        "selection_counts": {
            str(key): int(value)
            for key, value in panel["selection_class"]
            .value_counts()
            .sort_index()
            .items()
        },
        "quiet_controls_per_layer": {
            str(layer): int(
                len(
                    panel[
                        panel["injection_layer"].eq(int(layer))
                        & panel["selection_class"].eq("quiet_control")
                    ]
                )
            )
            for layer in EXPECTED_LAYERS
        },
        "selection_uses_final_kl_only_for_panel_enrichment": True,
        "panel_is_not_an_estimator_or_quality_cohort": True,
        "experiment_b_components_present": False,
    }
    atomic_json(facts_path, facts)
    print(f"[panel] rows={len(panel)} sha256={facts['panel_sha256']}", flush=True)


def _baseline_repeat_gate(
    first: DecodeObservation, second: DecodeObservation
) -> dict[str, Any]:
    repeat = _observation_maxima(first, second)
    cache = classified_cache_metrics(first.cache, second.cache)
    exact = bool(
        all(
            float(repeat[name]) == 0.0
            for name in (
                "hidden_max_abs",
                "router_logits_max_abs",
                "router_scores_max_abs",
                "x_max_abs",
                "routed_max_abs",
                "terminal_logits_max_abs",
            )
        )
        and bool(repeat["router_ids_equal"])
        and bool(cache["full_cache_bit_identical"])
    )
    if not exact:
        raise RuntimeError("paired native cached-decode baseline is not bit-identical")
    return {**repeat, **{f"paired_{key}": value for key, value in cache.items()}}


def _route_rows(
    baseline: DecodeObservation,
    candidate: DecodeObservation,
    layer: int,
    base: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    rows = []
    first_changed = -1
    for observation_layer in range(int(layer) + 1, 40):
        metrics = detailed_route_metrics(
            _np(baseline.router_logits[observation_layer]),
            _np(candidate.router_logits[observation_layer]),
            baseline.router_ids[observation_layer].numpy(),
            candidate.router_ids[observation_layer].numpy(),
            _np(baseline.router_scores[observation_layer]),
            _np(candidate.router_scores[observation_layer]),
            _np(baseline.hidden[observation_layer]),
            _np(candidate.hidden[observation_layer]),
        )
        if bool(metrics["route_membership_changed"]) and first_changed < 0:
            first_changed = int(observation_layer)
        cache = cache_state_metrics(baseline.cache, candidate.cache, observation_layer)
        rows.append(
            {
                **base,
                "observation_layer": int(observation_layer),
                "distance_from_injection": int(observation_layer) - int(layer),
                **metrics,
                **cache.to_dict("post_token_layer_cache"),
            }
        )
    return rows, first_changed


def _candidate_quality_row(
    baseline: DecodeObservation,
    candidate: DecodeObservation,
    encoded: Mapping[str, torch.Tensor],
    position: int,
    layer: int,
    base: Mapping[str, Any],
    local: Mapping[str, Any],
    injection: Mapping[str, Any],
    accounting: Mapping[str, Any],
    first_changed: int,
) -> dict[str, Any]:
    next_token_id = int(encoded["input_ids"][0, int(position) + 1].item())
    quality = current_token_quality_metrics(
        _np(baseline.logits),
        _np(candidate.logits),
        next_token_id,
        _np(baseline.hidden[39]),
        _np(candidate.hidden[39]),
    )
    cache = classified_cache_metrics(baseline.cache, candidate.cache)
    return {
        **base,
        "next_token_id": next_token_id,
        **local,
        **accounting,
        **quality,
        "realized_injected_layer_output_mse": float(
            np.mean(
                (
                    _np(candidate.hidden[int(layer)]).astype(np.float64)
                    - _np(baseline.hidden[int(layer)]).astype(np.float64)
                )
                ** 2
            )
        ),
        "first_route_membership_change_layer": int(first_changed),
        "routed_repeat_max_abs": float(injection["routed_repeat_max_abs"]),
        "candidate_seconds": float(candidate.elapsed_seconds),
        **{f"post_token_{key}": value for key, value in cache.items()},
    }


def _execute_candidate(
    observer: CachedDecodeObserver,
    model: Any,
    encoded: Mapping[str, torch.Tensor],
    position: int,
    exact_prefix: Any,
    baseline: DecodeObservation,
    layer: int,
    delta: np.ndarray,
    route_mode: str,
    base: Mapping[str, Any],
    local: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidate, injection = _run_candidate(
        observer,
        model,
        encoded,
        position,
        exact_prefix,
        baseline,
        layer,
        delta,
        route_mode,
    )
    routes, first_changed = _route_rows(baseline, candidate, layer, base)
    accounting = bf16_injection_accounting(_np(baseline.routed), delta)
    quality = _candidate_quality_row(
        baseline,
        candidate,
        encoded,
        position,
        layer,
        base,
        local,
        injection,
        accounting,
        first_changed,
    )
    del candidate
    return quality, routes


def _policy_local_metadata(local: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **local,
        "realizable_page_allocation": True,
        "interpolation_kind": "none",
        "interpolation_lambda": -1.0,
        "interpolation_scale": 1.0,
        "interpolation_low_rate": -1,
        "interpolation_high_rate": -1,
        "bandwidth_semantics": "realized_selected_page_state",
    }


def _interpolation_local_metadata(
    delta: np.ndarray,
    proxy: np.ndarray,
    beta: float,
    low: int,
    high: int,
    kind: str,
    lam: float,
    scale: float,
) -> dict[str, Any]:
    return {
        "selected_group_pages": -1,
        "source_selected_local_qenergy_damage": -1.0,
        "live_selected_local_qenergy_damage": float(qenergy_damage(delta, proxy, beta)),
        "source_stored_delta_mse": -1.0,
        "live_injected_delta_mse": float(np.mean(np.asarray(delta, np.float64) ** 2)),
        "stored_to_live_delta_mse": -1.0,
        "stored_to_live_delta_cosine": -1.0,
        "live_q4_routed_max_abs": -1.0,
        "source_and_live_route_set_equal": True,
        "source_and_live_route_order_equal": True,
        "live_execution_router_weight_sum": 1.0,
        "delta_execution": "diagnostic_interpolation_of_live_reconstructed_fixed_d1_deltas",
        "realizable_page_allocation": False,
        "interpolation_kind": str(kind),
        "interpolation_lambda": float(lam),
        "interpolation_scale": float(scale),
        "interpolation_low_rate": int(low),
        "interpolation_high_rate": int(high),
        "bandwidth_semantics": "non_realizable_mechanism_diagnostic_only",
    }


def run_layer(
    model: Any,
    encoded_requests: Mapping[str, Mapping[str, torch.Tensor]],
    banks: Mapping[tuple[int, int], Any],
    config: Mapping[str, Any],
    base_config: Mapping[str, Any],
    panel: pd.DataFrame,
    output: Path,
    layer: int,
    checkpoint: Path,
    trees: Path,
    fit_dir: Path,
    workers: int,
    config_sha256: str,
) -> None:
    panel_sha = sha256(output / PANEL)
    paths = _layer_paths(output, layer)
    if _completed_layer(paths, layer, panel_sha):
        print(f"[resume] diagnostic layer={layer}", flush=True)
        return
    layer_panel = panel[panel["injection_layer"].eq(int(layer))].copy()
    if layer_panel.empty:
        raise RuntimeError(f"diagnostic panel has no rows for layer {layer}")
    layer_banks = [banks[(int(layer), rate)] for rate in EXPECTED_RATES]
    decoded, proxy, beta = _prepare_live_reconstruction(
        checkpoint,
        trees,
        fit_dir,
        int(layer),
        layer_banks,
        workers,
    )
    bank_by_rate = {int(bank.rate): bank for bank in layer_banks}
    observer = CachedDecodeObserver(model, int(layer))
    quality_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    zero_rows: list[dict[str, Any]] = []
    baseline_seconds = 0.0
    candidate_seconds = 0.0
    try:
        for request_id, request_panel in layer_panel.groupby("request_id", sort=True):
            encoded = encoded_requests[str(request_id)]
            selected_positions = set(request_panel["position"].astype(int).tolist())
            maximum_position = max(selected_positions)
            prefix_observation = observer.run(model, encoded, 0, None)
            exact_prefix = prefix_observation.cache
            baseline_seconds += prefix_observation.elapsed_seconds
            del prefix_observation
            for position in range(1, maximum_position + 1):
                baseline = observer.run(
                    model,
                    encoded,
                    position,
                    clone_decode_cache(exact_prefix),
                )
                baseline_seconds += baseline.elapsed_seconds
                if position not in selected_positions:
                    exact_prefix = baseline.cache
                    continue
                identity_row = request_panel[request_panel["position"].eq(position)]
                if len(identity_row) != 1:
                    raise RuntimeError(
                        "panel token identity is not unique within layer/request"
                    )
                identity = identity_row.iloc[0]
                group = int(identity["group"])
                repeated = observer.run(
                    model,
                    encoded,
                    position,
                    clone_decode_cache(exact_prefix),
                )
                baseline_seconds += repeated.elapsed_seconds
                repeat = _baseline_repeat_gate(baseline, repeated)
                del repeated

                zero_delta = np.zeros(2048, np.float32)
                for route_mode in EXPECTED_ROUTE_MODES:
                    zero_candidate, injection = _run_candidate(
                        observer,
                        model,
                        encoded,
                        position,
                        exact_prefix,
                        baseline,
                        int(layer),
                        zero_delta,
                        route_mode,
                    )
                    candidate_seconds += zero_candidate.elapsed_seconds
                    zero = _zero_metrics(baseline, zero_candidate)
                    if (
                        not bool(zero["all_exact"])
                        or float(injection["routed_repeat_max_abs"]) != 0.0
                    ):
                        raise RuntimeError(
                            "diagnostic zero-dose gate failed exact parity"
                        )
                    zero_rows.append(
                        {
                            "schema": SCHEMA,
                            "injection_layer": int(layer),
                            "group": group,
                            "request_id": str(request_id),
                            "position": int(position),
                            "route_mode": route_mode,
                            "query_length": 1,
                            "cross_position_delta_coupling": False,
                            **repeat,
                            **zero,
                        }
                    )
                    del zero_candidate

                reconstruction_context: dict[str, Any] = {}
                fixed_by_rate: dict[int, np.ndarray] = {}
                for rate in EXPECTED_RATES:
                    bank = bank_by_rate[rate]
                    for policy in DECODE_POLICIES:
                        delta, local = _live_policy_delta(
                            bank,
                            policy,
                            group,
                            str(request_id),
                            position,
                            baseline,
                            decoded,
                            proxy,
                            beta,
                            base_config,
                            reconstruction_context,
                        )
                        if policy == FIXED_D1_POLICY:
                            fixed_by_rate[rate] = delta.copy()
                        candidate_id = f"rate_{rate}__{policy}"
                        for route_mode in EXPECTED_ROUTE_MODES:
                            base = {
                                "schema": SCHEMA,
                                "injection_layer": int(layer),
                                "rate_pages_per_expert": int(rate),
                                "group": group,
                                "request_id": str(request_id),
                                "position": int(position),
                                "prefix_tokens": int(position),
                                "candidate_id": candidate_id,
                                "candidate_kind": "allocation_policy",
                                "policy": str(policy),
                                "route_mode": route_mode,
                                "query_length": 1,
                                "cross_position_delta_coupling": False,
                            }
                            quality, routes = _execute_candidate(
                                observer,
                                model,
                                encoded,
                                position,
                                exact_prefix,
                                baseline,
                                int(layer),
                                delta,
                                route_mode,
                                base,
                                _policy_local_metadata(local),
                            )
                            candidate_seconds += float(quality["candidate_seconds"])
                            quality_rows.append(quality)
                            route_rows.extend(routes)

                for low, high in EXPECTED_RATE_PAIRS:
                    for kind, lam, delta, scale in interpolation_deltas(
                        fixed_by_rate[low],
                        fixed_by_rate[high],
                        INTERPOLATION_LAMBDAS,
                    ):
                        candidate_id = _candidate_id_for_interpolation(
                            low, high, kind, lam
                        )
                        local = _interpolation_local_metadata(
                            delta,
                            proxy,
                            beta,
                            low,
                            high,
                            kind,
                            lam,
                            scale,
                        )
                        for route_mode in EXPECTED_ROUTE_MODES:
                            base = {
                                "schema": SCHEMA,
                                "injection_layer": int(layer),
                                "rate_pages_per_expert": int(high),
                                "group": group,
                                "request_id": str(request_id),
                                "position": int(position),
                                "prefix_tokens": int(position),
                                "candidate_id": candidate_id,
                                "candidate_kind": "interpolation_control",
                                "policy": candidate_id,
                                "route_mode": route_mode,
                                "query_length": 1,
                                "cross_position_delta_coupling": False,
                            }
                            quality, routes = _execute_candidate(
                                observer,
                                model,
                                encoded,
                                position,
                                exact_prefix,
                                baseline,
                                int(layer),
                                delta,
                                route_mode,
                                base,
                                local,
                            )
                            candidate_seconds += float(quality["candidate_seconds"])
                            quality_rows.append(quality)
                            route_rows.extend(routes)
                exact_prefix = baseline.cache
                gc.collect()
    finally:
        observer.close()
        del decoded, proxy
        gc.collect()

    tokens = len(layer_panel)
    candidates = len(EXPECTED_RATES) * len(DECODE_POLICIES) + len(
        EXPECTED_RATE_PAIRS
    ) * (len(INTERPOLATION_LAMBDAS) + 1)
    expected_quality = tokens * candidates * len(EXPECTED_ROUTE_MODES)
    expected_routes = expected_quality * (39 - int(layer))
    expected_zero = tokens * len(EXPECTED_ROUTE_MODES)
    if (
        len(quality_rows) != expected_quality
        or len(route_rows) != expected_routes
        or len(zero_rows) != expected_zero
    ):
        raise RuntimeError(
            f"diagnostic layer grid incomplete: quality={len(quality_rows)}/{expected_quality}, "
            f"routes={len(route_rows)}/{expected_routes}, zero={len(zero_rows)}/{expected_zero}"
        )
    paths["root"].mkdir(parents=True, exist_ok=True)
    atomic_parquet(paths["quality"], quality_rows)
    atomic_parquet(paths["routes"], route_rows)
    atomic_parquet(paths["zero"], zero_rows)
    files = {
        paths[key].name: {
            "sha256": sha256(paths[key]),
            "bytes": paths[key].stat().st_size,
            "rows": len(rows),
        }
        for key, rows in (
            ("quality", quality_rows),
            ("routes", route_rows),
            ("zero", zero_rows),
        )
    }
    atomic_json(
        paths["facts"],
        {
            "completed": True,
            "schema": SCHEMA,
            "injection_layer": int(layer),
            "config_sha256": str(config_sha256),
            "panel_sha256": panel_sha,
            "panel_tokens": tokens,
            "allocation_candidates_per_token": len(EXPECTED_RATES)
            * len(DECODE_POLICIES),
            "interpolation_candidates_per_token": len(EXPECTED_RATE_PAIRS)
            * (len(INTERPOLATION_LAMBDAS) + 1),
            "route_modes": list(EXPECTED_ROUTE_MODES),
            "baseline_seconds": baseline_seconds,
            "candidate_seconds": candidate_seconds,
            "files": files,
            "terminal_metric_scope": "current_token_only",
            "exact_prefix_cache_cloned_per_candidate": True,
            "experiment_b_components_present": False,
        },
    )
    print(
        f"[complete] diagnostic layer={layer} tokens={tokens} quality={len(quality_rows)} "
        f"routes={len(route_rows)} zero={len(zero_rows)}",
        flush=True,
    )


def run_phase(args: argparse.Namespace) -> None:
    config, _, base = _diagnostic_config(args.config)
    panel_path, panel_facts_path = _panel_paths(args.output)
    if not panel_path.is_file() or not panel_facts_path.is_file():
        raise RuntimeError("run phase requires a completed immutable panel phase")
    panel_facts = load_json(panel_facts_path)
    if sha256(panel_path) != str(panel_facts["panel_sha256"]):
        raise RuntimeError("diagnostic panel changed after selection")
    panel = pd.read_parquet(panel_path)
    if len(panel) != 24 or panel.duplicated(IDENTITY).any():
        raise RuntimeError("diagnostic panel identity gate failed")
    hardware = _gpu_gate(config)
    checkpoint_index = args.checkpoint / "model.safetensors.index.json"
    if sha256(checkpoint_index) != str(base["checkpoint_index_sha256"]):
        raise RuntimeError("checkpoint index changed")
    if sha256(args.trees) != str(base["selected_tree_sha256"]):
        raise RuntimeError("selected-tree file changed")
    _, _, _, banks = _policy_banks(base)
    model, tokenizer, load_seconds = _load_model(args.checkpoint)
    encoded = _encoded_requests(tokenizer, model, base)
    atomic_json(
        args.output / HOST_FACTS,
        {
            "completed": True,
            "schema": SCHEMA,
            **hardware,
            "model_load_seconds": load_seconds,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "host": platform.node(),
            "config_sha256": sha256(args.config),
            "panel_sha256": sha256(panel_path),
        },
    )
    requested = set(args.layer if args.layer else EXPECTED_LAYERS)
    if not requested.issubset(set(EXPECTED_LAYERS)):
        raise ValueError("--layer contains a non-sentinel layer")
    try:
        for layer in EXPECTED_LAYERS:
            if layer not in requested:
                continue
            run_layer(
                model,
                encoded,
                banks,
                config,
                base,
                panel,
                args.output,
                layer,
                args.checkpoint,
                args.trees,
                args.fit_dir,
                args.reconstruction_workers,
                sha256(args.config),
            )
    finally:
        del encoded, tokenizer, model
        gc.collect()
        torch.cuda.empty_cache()


def _add_live_minus_controls(quality: pd.DataFrame) -> pd.DataFrame:
    keys = [*IDENTITY, "candidate_id"]
    metrics = {}
    live = quality[quality["route_mode"].eq("live")][keys + ["logit_kl"]]
    if live.duplicated(keys).any():
        raise RuntimeError("live diagnostic candidate identity is not unique")
    for mode, suffix in (
        ("frozen_set_live_weights", "frozen_set_live_weights"),
        ("fully_frozen", "fully_frozen"),
    ):
        frozen = quality[quality["route_mode"].eq(mode)][keys + ["logit_kl"]]
        paired = live.merge(
            frozen, on=keys, suffixes=("_live", "_frozen"), validate="one_to_one"
        )
        paired[f"live_minus_{suffix}_logit_kl"] = (
            paired["logit_kl_live"] - paired["logit_kl_frozen"]
        )
        metrics[suffix] = paired[keys + [f"live_minus_{suffix}_logit_kl"]]
    result = quality
    for frame in metrics.values():
        result = result.merge(frame, on=keys, validate="many_to_one")
    return result


def _endpoint_parity(quality: pd.DataFrame) -> dict[str, Any]:
    comparisons = []
    fields = [
        "logit_kl",
        "delta_nll",
        "final_hidden_mse",
        "post_token_full_cache_mse",
        "live_injected_delta_mse",
        "bf16_effective_delta_mse",
    ]
    for low, high in EXPECTED_RATE_PAIRS:
        for rate, suffix in ((low, "lambda_0"), (high, "lambda_1")):
            fixed_id = f"rate_{rate}__{FIXED_D1_POLICY}"
            interpolation_id = f"fixed_d1_path_{low}_{high}__{suffix}"
            left = quality[quality["candidate_id"].eq(fixed_id)][
                QUALITY_IDENTITY + fields
            ]
            right = quality[quality["candidate_id"].eq(interpolation_id)][
                QUALITY_IDENTITY + fields
            ]
            # Candidate ID differs by design; compare all physical identities.
            keys = [*IDENTITY, "route_mode"]
            paired = left.drop(columns="candidate_id").merge(
                right.drop(columns="candidate_id"),
                on=keys,
                suffixes=("_fixed", "_path"),
                validate="one_to_one",
            )
            maxima = {
                field: float(
                    np.max(
                        np.abs(
                            paired[f"{field}_fixed"].to_numpy(np.float64)
                            - paired[f"{field}_path"].to_numpy(np.float64)
                        ),
                        initial=0.0,
                    )
                )
                for field in fields
            }
            comparisons.append(
                {
                    "low_rate": low,
                    "high_rate": high,
                    "endpoint_rate": rate,
                    "fixed_candidate_id": fixed_id,
                    "path_candidate_id": interpolation_id,
                    "rows": len(paired),
                    "maxima": maxima,
                }
            )
    all_exact = all(
        value == 0.0
        for comparison in comparisons
        for value in comparison["maxima"].values()
    )
    if not all_exact:
        raise RuntimeError("fixed-D1 interpolation endpoint replay parity failed")
    return {"all_exact": True, "comparisons": comparisons}


def finalize_phase(args: argparse.Namespace) -> None:
    config, base_path, _ = _diagnostic_config(args.config)
    panel_path, panel_facts_path = _panel_paths(args.output)
    panel_facts = load_json(panel_facts_path)
    panel_sha = sha256(panel_path)
    if panel_sha != str(panel_facts["panel_sha256"]):
        raise RuntimeError("diagnostic panel changed before finalize")
    quality_parts = []
    route_parts = []
    zero_parts = []
    child_facts = {}
    for layer in EXPECTED_LAYERS:
        paths = _layer_paths(args.output, layer)
        if not _completed_layer(paths, layer, panel_sha):
            raise RuntimeError(f"diagnostic layer is incomplete: {layer}")
        quality_parts.append(pd.read_parquet(paths["quality"]))
        route_parts.append(pd.read_parquet(paths["routes"]))
        zero_parts.append(pd.read_parquet(paths["zero"]))
        child_facts[str(layer)] = {
            "path": str(paths["facts"]),
            "sha256": sha256(paths["facts"]),
        }
    quality = _add_live_minus_controls(pd.concat(quality_parts, ignore_index=True))
    routes = pd.concat(route_parts, ignore_index=True)
    zero = pd.concat(zero_parts, ignore_index=True)
    if not bool(zero["all_exact"].all()):
        raise RuntimeError("a finalized diagnostic zero-dose row is not exact")
    expected_tokens = 24
    candidate_count = len(EXPECTED_RATES) * len(DECODE_POLICIES) + len(
        EXPECTED_RATE_PAIRS
    ) * (len(INTERPOLATION_LAMBDAS) + 1)
    expected_quality = expected_tokens * candidate_count * len(EXPECTED_ROUTE_MODES)
    expected_routes = sum(
        len(pd.read_parquet(panel_path).query("injection_layer == @layer"))
        * candidate_count
        * len(EXPECTED_ROUTE_MODES)
        * (39 - layer)
        for layer in EXPECTED_LAYERS
    )
    expected_zero = expected_tokens * len(EXPECTED_ROUTE_MODES)
    if (len(quality), len(routes), len(zero)) != (
        expected_quality,
        expected_routes,
        expected_zero,
    ):
        raise RuntimeError("final diagnostic grid counts changed")
    endpoint = _endpoint_parity(quality)
    for frame, name in ((quality, QUALITY), (routes, ROUTES), (zero, ZERO)):
        if frame.empty or frame.isnull().any().any():
            raise RuntimeError(
                f"final diagnostic table is empty or contains nulls: {name}"
            )
        atomic_parquet(args.output / name, frame)
    outputs = {
        name: {
            "sha256": sha256(args.output / name),
            "bytes": (args.output / name).stat().st_size,
            "rows": len(frame),
        }
        for name, frame in ((QUALITY, quality), (ROUTES, routes), (ZERO, zero))
    }
    atomic_json(
        args.output / RUN_FACTS,
        {
            "completed": True,
            "schema": SCHEMA,
            "config_sha256": sha256(args.config),
            "base_tail_config": str(base_path),
            "base_tail_config_sha256": sha256(base_path),
            "panel_facts_sha256": sha256(panel_facts_path),
            "panel_sha256": panel_sha,
            "counts": {
                "unique_token_layers": expected_tokens,
                "quality_rows": len(quality),
                "detailed_route_rows": len(routes),
                "zero_rows": len(zero),
                "candidates_per_token": candidate_count,
                "route_modes": len(EXPECTED_ROUTE_MODES),
            },
            "outputs": outputs,
            "layer_facts": child_facts,
            "interpolation_endpoint_parity": endpoint,
            "zero_dose_all_exact": True,
            "terminal_metric_scope": "current_token_only",
            "route_modes": list(EXPECTED_ROUTE_MODES),
            "interpolation_controls_are_page_unrealizable": True,
            "bf16_ulp_accounting_present": True,
            "scientific_boundary": (
                "Enriched 24-token/layer Experiment A diagnostic panel selected from the "
                "three-request smoke cohort. One injected layer and one current token per "
                "candidate behind an exact prefix cache. Interpolation controls are not "
                "page allocations. No Experiment B controller, general quality claim, "
                "generated rollout, or joint all-layer compression is present."
            ),
            "experiment_b_components_present": False,
        },
    )
    print(
        f"[finalized] quality={len(quality)} routes={len(routes)} zero={len(zero)}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("panel", "run", "finalize"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--trees", type=Path)
    parser.add_argument("--fit-dir", type=Path)
    parser.add_argument("--reconstruction-workers", type=int, default=8)
    parser.add_argument("--layer", type=int, action="append")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_json(args.config)
    args.output = args.output or Path(str(config["output_root"]))
    if args.phase == "run" and any(
        value is None for value in (args.checkpoint, args.trees, args.fit_dir)
    ):
        parser.error(
            "--checkpoint, --trees, and --fit-dir are required for --phase run"
        )
    return args


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    if args.phase == "panel":
        panel_phase(args)
    elif args.phase == "run":
        run_phase(args)
    else:
        finalize_phase(args)


if __name__ == "__main__":
    main()
