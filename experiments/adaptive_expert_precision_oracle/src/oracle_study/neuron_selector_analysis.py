"""Validation-only selection and final reporting for neuron distillation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


IDENTITY = ("capture_source", "evaluation_split", "request_id", "position", "layer", "expert_id")
RESPONSE_KEYS = ("training_cohort", "basis_variant", "rank", "synthesis_encoding")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array) or not np.all(np.isfinite(array)):
        raise RuntimeError("summary values are empty or non-finite")
    return {
        "n": int(len(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def evidence_frame(path: Path, *, split: str) -> pd.DataFrame:
    frame = pd.read_parquet(path, filters=[("evaluation_split", "==", split)])
    if frame.empty:
        raise RuntimeError(f"no {split} rows in {path}")
    if set(frame["evaluation_split"].astype(str)) != {split}:
        raise RuntimeError(f"predicate pushdown admitted another split from {path}")
    if frame.duplicated(list(IDENTITY) + [
        column for column in RESPONSE_KEYS if column in frame.columns
    ] + [column for column in ("action_family", "physical_budget_bpw", "candidate_units") if column in frame.columns]).any():
        raise RuntimeError(f"duplicate scientific rows in {path}")
    return frame


def require_coverage(
    frame: pd.DataFrame,
    layers: Sequence[int],
    minimum_invocations: int,
    minimum_requests: int,
) -> list[dict[str, Any]]:
    result = []
    for layer in map(int, layers):
        local = frame[frame["layer"].astype(int) == layer]
        identities = local[list(IDENTITY)].drop_duplicates()
        requests = local["request_id"].astype(str).nunique()
        if len(identities) < minimum_invocations or requests < minimum_requests:
            raise RuntimeError(
                f"layer {layer} coverage {len(identities)} invocations/{requests} requests is below minimum"
            )
        result.append({
            "layer": layer,
            "validation_invocations": int(len(identities)),
            "validation_requests": int(requests),
        })
    return result


def _aggregate_response(frame: pd.DataFrame, *, candidate: bool) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for key, local in frame.groupby(list(RESPONSE_KEYS), dropna=False, sort=True):
        row = dict(zip(RESPONSE_KEYS, key))
        recovery = quantiles(local["recovery"])
        row.update({f"recovery_{name}": value for name, value in recovery.items()})
        utility_column = "oracle_utility_retention" if candidate else "utility_weighted_recall"
        utility = quantiles(local[utility_column])
        row.update({f"utility_{name}": value for name, value in utility.items()})
        row["selector_metadata_bpw"] = float(local["selector_metadata_bpw"].max())
        row["storage_multiplier"] = float(local["storage_multiplier"].max())
        row["selector_compute_macs"] = int(local["selector_compute_macs"].max())
        row["response_config_ids"] = sorted(set(map(str, local["response_config_id"])))
        if candidate:
            row["candidate_units"] = int(local["candidate_units"].iloc[0])
            row["applied_units"] = int(local["applied_units"].iloc[0])
            row["candidate_overfetch"] = float(local["candidate_overfetch"].max())
            support = quantiles(local["oracle_support_recall"])
            row.update({f"support_recall_{name}": value for name, value in support.items()})
        result.append(row)
    return result


def choose_validation_promotions(
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    fit_bundle_sha256: str,
    factorized: pd.DataFrame,
    response: pd.DataFrame,
    candidate: pd.DataFrame,
    tile_comparator: pd.DataFrame,
    evidence_sha256: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Freeze decisions from validation evidence without consulting test rows."""
    for name, frame in {
        "factorized": factorized,
        "response": response,
        "candidate": candidate,
        "tile_comparator": tile_comparator,
    }.items():
        if set(frame["evaluation_split"].astype(str)) != {"validation"}:
            raise RuntimeError(f"{name} contains non-validation rows")
    policy = config["promotion_policy"]
    layers = list(map(int, config["layers"]))
    coverage = require_coverage(
        factorized,
        layers,
        int(policy["minimum_validation_invocations_per_layer"]),
        int(policy["minimum_validation_requests_per_layer"]),
    )
    one_bpw = factorized[
        (factorized["action_family"] == "factorized_gate_up_down_unit_states")
        & np.isclose(pd.to_numeric(factorized["physical_budget_bpw"]), 1.0)
    ]
    tile = tile_comparator[
        (tile_comparator["selector"] == config["pr7_comparators"]["validation_tile_frontier"]["selector"])
        & (tile_comparator["tile_shape"].astype(str) == config["pr7_comparators"]["validation_tile_frontier"]["tile_shape"])
        & np.isclose(pd.to_numeric(tile_comparator["physical_budget_bpw"]), 1.0)
    ]
    paired = one_bpw.merge(
        tile[[*IDENTITY, "recovery"]], on=list(IDENTITY), how="inner",
        suffixes=("_factorized", "_tile"), validate="one_to_one",
    )
    if paired.empty:
        raise RuntimeError("factorized rows do not overlap the frozen PR7 tile comparator")
    factorized_summary = quantiles(paired["recovery_factorized"])
    tile_summary = quantiles(paired["recovery_tile"])
    factorized_summary.update({
        "tile_p10": tile_summary["p10"],
        "tile_median": tile_summary["median"],
        "p10_gap_to_tile": tile_summary["p10"] - factorized_summary["p10"],
        "median_gap_to_tile": tile_summary["median"] - factorized_summary["median"],
    })
    factorized_summary["passes_p10_gap"] = (
        factorized_summary["p10_gap_to_tile"]
        <= float(policy["factorized_max_p10_gap_to_tile_at_1bpw"])
    )
    factorized_summary["passes_median_gap"] = (
        factorized_summary["median_gap_to_tile"]
        <= float(policy["factorized_max_median_gap_to_tile_at_1bpw"])
    )
    promote_factorized = bool(
        factorized_summary["passes_p10_gap"] and factorized_summary["passes_median_gap"]
    )

    direct_one = response[
        (response["unit_count"].astype(int) == 256)
        & np.isclose(pd.to_numeric(response["physical_budget_bpw"]), 1.0)
    ]
    candidate_one = candidate[
        (candidate["candidate_units"].astype(int) == int(policy["candidate_max_units"]))
        & (candidate["applied_units"].astype(int) == int(config["candidate_applied_units"]))
    ]
    direct_rows = _aggregate_response(direct_one, candidate=False)
    candidate_rows = _aggregate_response(candidate_one, candidate=True)
    direct_by_key = {tuple(row[name] for name in RESPONSE_KEYS): row for row in direct_rows}
    candidate_by_key = {tuple(row[name] for name in RESPONSE_KEYS): row for row in candidate_rows}
    all_candidates = []
    for key in sorted(set(direct_by_key) & set(candidate_by_key)):
        direct_row, candidate_row = direct_by_key[key], candidate_by_key[key]
        upper_bound = str(direct_row["basis_variant"]).endswith("upper_bound")
        row = {
            **{name: direct_row[name] for name in RESPONSE_KEYS},
            "response_config_ids": direct_row["response_config_ids"],
            "direct_recovery_p10": direct_row["recovery_p10"],
            "direct_recovery_median": direct_row["recovery_median"],
            "direct_utility_p10": direct_row["utility_p10"],
            "direct_utility_median": direct_row["utility_median"],
            "candidate_recovery_p10": candidate_row["recovery_p10"],
            "candidate_recovery_median": candidate_row["recovery_median"],
            "candidate_utility_p10": candidate_row["utility_p10"],
            "candidate_utility_median": candidate_row["utility_median"],
            "candidate_support_recall_p10": candidate_row["support_recall_p10"],
            "candidate_support_recall_median": candidate_row["support_recall_median"],
            "candidate_units": candidate_row["candidate_units"],
            "applied_units": candidate_row["applied_units"],
            "candidate_overfetch": candidate_row["candidate_overfetch"],
            "selector_metadata_bpw": max(
                direct_row["selector_metadata_bpw"], candidate_row["selector_metadata_bpw"],
            ),
            "storage_multiplier": max(direct_row["storage_multiplier"], candidate_row["storage_multiplier"]),
            "selector_compute_macs": max(direct_row["selector_compute_macs"], candidate_row["selector_compute_macs"]),
            "sampled_per_expert_upper_bound": upper_bound,
        }
        row["passes_metadata"] = row["selector_metadata_bpw"] <= float(policy["max_selector_metadata_bpw"])
        row["passes_direct"] = bool(
            row["direct_recovery_median"] >= float(policy["direct_predictor_min_median_recovery_at_1bpw"])
            and row["direct_recovery_p10"] >= float(policy["direct_predictor_min_p10_recovery_at_1bpw"])
            and row["passes_metadata"] and not upper_bound
        )
        row["passes_candidate"] = bool(
            row["candidate_utility_median"] >= float(policy["candidate_min_median_utility_retention"])
            and row["candidate_utility_p10"] >= float(policy["candidate_min_p10_utility_retention"])
            and row["candidate_recovery_median"] >= float(policy["candidate_min_median_recovery"])
            and row["candidate_recovery_p10"] >= float(policy["candidate_min_p10_recovery"])
            and row["candidate_units"] <= int(policy["candidate_max_units"])
            and row["candidate_overfetch"] <= float(policy["candidate_max_overfetch"]) + 1e-12
            and row["passes_metadata"] and not upper_bound
        )
        all_candidates.append(row)
    passing = [row for row in all_candidates if row["passes_candidate"] or row["passes_direct"]]
    passing.sort(key=lambda row: (
        not row["passes_candidate"],
        -row["candidate_utility_p10"],
        -row["candidate_recovery_p10"],
        -row["candidate_recovery_median"],
        row["selector_metadata_bpw"],
        int(row["rank"]),
        str(row["basis_variant"]),
        str(row["training_cohort"]),
    ))
    promoted = passing[:1]
    shared_pass = bool(promoted)
    upper_pass = any(
        row["sampled_per_expert_upper_bound"]
        and row["candidate_utility_median"] >= float(policy["candidate_min_median_utility_retention"])
        and row["candidate_recovery_median"] >= float(policy["candidate_min_median_recovery"])
        for row in all_candidates
    )
    promoted_ids = promoted[0]["response_config_ids"] if promoted else []
    payload = {
        "schema_version": 1,
        "generated_by": "oracle_study.neuron_selector_analysis.choose_validation_promotions",
        "config_sha256": str(config_sha256),
        "fit_bundle_sha256": str(fit_bundle_sha256),
        "selection_split": "validation",
        "selection_capture_source": "exact_checkpoint",
        "test_rows_consulted_for_selection": False,
        "nonvalidation_rows_excluded_from_evidence_hashes": True,
        "validation_evidence_sha256": dict(evidence_sha256),
        "validation_coverage": coverage,
        "factorized_oracle": {
            "status": "promote" if promote_factorized else "stop",
            **factorized_summary,
        },
        "promote_factorized_oracle": promote_factorized,
        "response_predictor": {
            "status": "promote" if promoted else "stop",
            "promoted": promoted,
            "all_validation_candidates": all_candidates,
        },
        "promoted_response_config_ids": promoted_ids,
        "conditional_followup": {
            "activation_topk": shared_pass,
            "cluster_bases": bool(not shared_pass and upper_pass),
            "reason": (
                "shared predictor passed; run only the predeclared validation-only activation-top-k continuation"
                if shared_pass else
                "sampled per-expert upper bound passed while shared bases failed"
                if upper_pass else
                "response predictor stopped; no conditional expansion"
            ),
        },
    }
    tables = {
        "factorized_paired_validation": paired,
        "response_validation_summary": pd.DataFrame(all_candidates),
    }
    return payload, tables


def validate_promotion_payload(
    payload: Mapping[str, Any], config: Mapping[str, Any], *,
    config_sha256: str, fit_bundle_sha256: str,
) -> tuple[set[str], bool]:
    """Recompute all frozen gates before held-out evaluation."""
    if payload.get("schema_version") != 1 or payload.get("selection_split") != "validation":
        raise RuntimeError("invalid neuron-selector promotion artifact")
    if payload.get("test_rows_consulted_for_selection") is not False:
        raise RuntimeError("promotion artifact permits test tuning")
    if payload.get("nonvalidation_rows_excluded_from_evidence_hashes") is not True:
        raise RuntimeError("promotion evidence may include non-validation rows")
    if payload.get("config_sha256") != config_sha256:
        raise RuntimeError("promotion config hash changed")
    if payload.get("fit_bundle_sha256") != fit_bundle_sha256:
        raise RuntimeError("promotion fit bundle hash changed")
    for digest in payload.get("validation_evidence_sha256", {}).values():
        if not isinstance(digest, str) or len(digest) != 64 or set(digest) - set("0123456789abcdef"):
            raise RuntimeError("promotion evidence contains a malformed SHA-256")

    policy = config["promotion_policy"]
    coverage = payload.get("validation_coverage", [])
    if {int(row["layer"]) for row in coverage} != set(map(int, config["layers"])):
        raise RuntimeError("promotion coverage does not contain every configured layer")
    for row in coverage:
        if int(row["validation_invocations"]) < int(policy["minimum_validation_invocations_per_layer"]):
            raise RuntimeError("promotion invocation coverage is below the frozen minimum")
        if int(row["validation_requests"]) < int(policy["minimum_validation_requests_per_layer"]):
            raise RuntimeError("promotion request coverage is below the frozen minimum")

    factorized = payload.get("factorized_oracle", {})
    factorized_expected = bool(
        float(factorized["p10_gap_to_tile"]) <= float(policy["factorized_max_p10_gap_to_tile_at_1bpw"])
        and float(factorized["median_gap_to_tile"]) <= float(policy["factorized_max_median_gap_to_tile_at_1bpw"])
    )
    if payload.get("promote_factorized_oracle") is not factorized_expected:
        raise RuntimeError("factorized promotion does not match the frozen gate")
    if factorized.get("status") != ("promote" if factorized_expected else "stop"):
        raise RuntimeError("factorized promotion status is inconsistent")

    candidates = payload.get("response_predictor", {}).get("all_validation_candidates", [])
    for row in candidates:
        upper = str(row["basis_variant"]).endswith("upper_bound")
        metadata = float(row["selector_metadata_bpw"]) <= float(policy["max_selector_metadata_bpw"])
        direct = bool(
            float(row["direct_recovery_median"]) >= float(policy["direct_predictor_min_median_recovery_at_1bpw"])
            and float(row["direct_recovery_p10"]) >= float(policy["direct_predictor_min_p10_recovery_at_1bpw"])
            and metadata and not upper
        )
        candidate = bool(
            float(row["candidate_utility_median"]) >= float(policy["candidate_min_median_utility_retention"])
            and float(row["candidate_utility_p10"]) >= float(policy["candidate_min_p10_utility_retention"])
            and float(row["candidate_recovery_median"]) >= float(policy["candidate_min_median_recovery"])
            and float(row["candidate_recovery_p10"]) >= float(policy["candidate_min_p10_recovery"])
            and int(row["candidate_units"]) <= int(policy["candidate_max_units"])
            and float(row["candidate_overfetch"]) <= float(policy["candidate_max_overfetch"]) + 1e-12
            and metadata and not upper
        )
        if row.get("passes_metadata") is not metadata:
            raise RuntimeError("stored metadata gate is inconsistent")
        if row.get("passes_direct") is not direct or row.get("passes_candidate") is not candidate:
            raise RuntimeError("stored response-predictor gate is inconsistent")
    passing = [row for row in candidates if row["passes_candidate"] or row["passes_direct"]]
    passing.sort(key=lambda row: (
        not row["passes_candidate"], -float(row["candidate_utility_p10"]),
        -float(row["candidate_recovery_p10"]), -float(row["candidate_recovery_median"]),
        float(row["selector_metadata_bpw"]), int(row["rank"]),
        str(row["basis_variant"]), str(row["training_cohort"]),
    ))
    expected = passing[:1]
    stored = payload.get("response_predictor", {}).get("promoted", [])
    if stored != expected:
        raise RuntimeError("promoted response configuration is not the frozen validation winner")
    expected_ids = set(map(str, expected[0]["response_config_ids"])) if expected else set()
    stored_ids = set(map(str, payload.get("promoted_response_config_ids", [])))
    if stored_ids != expected_ids:
        raise RuntimeError("promoted response config IDs are inconsistent")
    if expected_ids and len(expected_ids) != len(config["layers"]):
        raise RuntimeError("promoted response configuration does not cover every layer")
    status = payload.get("response_predictor", {}).get("status")
    if status != ("promote" if expected else "stop"):
        raise RuntimeError("response-predictor promotion status is inconsistent")
    return stored_ids, factorized_expected


def canonical_json_sha(value: Mapping[str, Any]) -> str:
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "IDENTITY",
    "RESPONSE_KEYS",
    "canonical_json_sha",
    "choose_validation_promotions",
    "evidence_frame",
    "quantiles",
    "require_coverage",
    "sha256",
]
