"""Fail-closed analysis for the eight-state split interaction field."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .split_interaction_field import (
    INHERITED_FOUR_STATE_MAP,
    SPLIT_ONLY_STATES,
    SPLIT_STATE_DOWN_HIGH,
    SPLIT_STATE_GATE_HIGH,
    SPLIT_STATE_PAGE_COSTS,
    SPLIT_STATE_UP_HIGH,
)


IDENTITY = (
    "capture_source", "evaluation_split", "request_id", "position",
    "layer", "expert_id",
)
EXPECTED_SOLVERS = {
    "four_state_pr11_reference",
    "four_state_dp_two_seed_coordinate_plus_local",
    "eight_state_two_seed_coordinate_plus_local",
    "eight_state_inherited_four_state_warm_start",
    "exact_gram_four_state_reference",
    "exact_gram_eight_state_oracle",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values: pd.Series) -> dict[str, float]:
    array = values.to_numpy(np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise RuntimeError("summary input must be nonempty and finite")
    return {
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
    }


def _canonical_states(value: Any) -> np.ndarray:
    if not isinstance(value, str):
        raise RuntimeError("selected_states must be canonical JSON text")
    parsed = json.loads(value)
    states = np.asarray(parsed, np.int64)
    if states.shape != (512,) or np.any((states < 0) | (states > 7)):
        raise RuntimeError("selected_states must contain 512 values in [0,7]")
    if value != json.dumps(parsed, separators=(",", ":")):
        raise RuntimeError("selected_states JSON is not canonical")
    return states


def _assert_close(observed: Any, expected: float, field: str) -> None:
    value = float(observed)
    if not np.isfinite(value) or not np.isclose(value, expected, rtol=1e-9, atol=1e-8):
        raise RuntimeError(f"{field} accounting mismatch")


def validate_evidence_bundle(
    config_path: Path,
    validation_dir: Path,
    *,
    runner_path: Path,
    core_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    config = json.loads(Path(config_path).read_text())
    facts_path = Path(validation_dir) / "run_facts.json"
    frontier_path = Path(validation_dir) / "split_interaction_field_frontier.parquet"
    facts = json.loads(facts_path.read_text())
    if facts.get("completed") is not True or facts.get("failures") != []:
        raise RuntimeError("split interaction run is incomplete or failed")
    if facts.get("test_rows_admitted") is not False or facts.get("test_scientific_values_used") is not False:
        raise RuntimeError("test scientific boundary changed")
    if facts.get("selection_split") != "validation":
        raise RuntimeError("analysis is not validation-only")
    if facts.get("config_sha256") != sha256(config_path):
        raise RuntimeError("config hash mismatch")
    if facts.get("runner_sha256") != sha256(runner_path):
        raise RuntimeError("runner hash mismatch")
    if facts.get("split_core_sha256") != sha256(core_path):
        raise RuntimeError("split core hash mismatch")
    for field in ("pr11_run_facts_sha256", "pr11_factor_manifest_sha256", "pr11_frontier_sha256"):
        if facts.get(field) != config[field]:
            raise RuntimeError(f"{field} mismatch")
    if facts.get("frontier_sha256") != sha256(frontier_path):
        raise RuntimeError("frontier hash mismatch")
    if len(facts.get("completed_work_units", [])) != int(config["expected_validation_layer_expert_cells"]):
        raise RuntimeError("work-unit coverage changed")
    scope = facts.get("validation_scope_audit", {})
    if (
        int(scope.get("expected_unique_invocations", -1)) != 69
        or int(scope.get("observed_unique_invocations", -1)) != 69
        or int(scope.get("observed_layer_expert_cells", -1)) != 12
    ):
        raise RuntimeError("69-invocation/12-cell scope changed")
    if facts.get("state_page_costs") != SPLIT_STATE_PAGE_COSTS.tolist():
        raise RuntimeError("serialized state costs changed")
    if facts.get("inherited_four_state_map") != INHERITED_FOUR_STATE_MAP.tolist():
        raise RuntimeError("serialized four-state embedding changed")
    if facts.get("strict_all_in_page_budgets") != config["strict_all_in_page_budgets"]:
        raise RuntimeError("serialized all-in page budgets changed")
    if int(facts.get("strict_all_in_total_bytes", -1)) != 393_216:
        raise RuntimeError("serialized all-in total-byte budget changed")
    if facts.get("incremental_coordinate_updates") is not True:
        raise RuntimeError("incremental coordinate implementation is not declared")

    frame = pd.read_parquet(frontier_path)
    expected_rows = 69 * (
        3 * (2 + 4 * len(config["factor_config_ids"]))
        + 2 * len(config["factor_config_ids"])
    )
    if len(frame) != expected_rows or int(facts.get("frontier_rows", -1)) != expected_rows:
        raise RuntimeError("frontier row count changed")
    if set(frame["solver"]) != EXPECTED_SOLVERS:
        raise RuntimeError("solver grid changed")
    identities = frame[list(IDENTITY)].drop_duplicates()
    if len(identities) != 69 or set(frame["evaluation_split"]) != {"validation"}:
        raise RuntimeError("identity/split coverage changed")
    fixed = frame[frame["budget_regime"] == "fixed_correction_budget"]
    all_in = frame[frame["budget_regime"] == "strict_all_in_one_bpw"]
    if len(fixed) != 69 * 3 * (2 + 4 * len(config["factor_config_ids"])):
        raise RuntimeError("fixed-budget grid changed")
    if len(all_in) != 69 * 2 * len(config["factor_config_ids"]):
        raise RuntimeError("strict all-in grid changed")
    if set(fixed["physical_budget_pages"].astype(int)) != {384, 576, 768}:
        raise RuntimeError("fixed page-budget grid changed")
    ordinary = frame[~frame["exact_gram_oracle"].astype(bool)]
    exact = frame[frame["exact_gram_oracle"].astype(bool)]
    if set(ordinary["factor_config_id"]) != set(config["factor_config_ids"]):
        raise RuntimeError("factor grid changed")
    if set(exact["factor_config_id"]) != {"exact_full_down_gram_training_only"}:
        raise RuntimeError("exact-Gram factor identity changed")
    if set(exact["budget_regime"]) != {"fixed_correction_budget"}:
        raise RuntimeError("exact-Gram control entered all-in comparison")
    expected_fixed_factor_rows = 69 * 3
    for factor in config["factor_config_ids"]:
        for solver in (
            config["pr11_four_state_solver_id"], config["four_state_reference_solver_id"],
            config["primary_solver_id"], config["warm_start_solver_id"],
        ):
            rows = fixed[(fixed["factor_config_id"] == factor) & (fixed["solver"] == solver)]
            if len(rows) != expected_fixed_factor_rows:
                raise RuntimeError("fixed factor/solver grid incomplete")
        expected_pages = int(config["strict_all_in_page_budgets"][factor])
        for solver in (config["four_state_reference_solver_id"], config["primary_solver_id"]):
            rows = all_in[(all_in["factor_config_id"] == factor) & (all_in["solver"] == solver)]
            if len(rows) != 69 or set(rows["physical_budget_pages"].astype(int)) != {expected_pages}:
                raise RuntimeError("strict all-in factor/solver grid incomplete")
    for solver in (config["exact_four_state_solver_id"], config["exact_eight_state_solver_id"]):
        if len(exact[exact["solver"] == solver]) != expected_fixed_factor_rows:
            raise RuntimeError("exact-Gram solver grid incomplete")

    universal_numeric = (
        "physical_pages", "physical_budget_pages", "recovery",
        "four_state_reference_recovery", "pr11_four_state_recovery",
        "recovery_gain_vs_four_state_reference", "recovery_gain_vs_pr11_four_state",
        "seed_optimizer_gain_vs_pr11_four_state", "compressed_predicted_damage",
        "exact_qenergy_damage", "damage_prediction_error_over_base",
        "selector_runtime_ms", "field_build_runtime_ms", "coordinate_sweeps", "selected_seed_coordinate_sweeps",
        "coordinate_accepted_moves", "local_evaluated_passes", "local_accepted_bundles",
        "local_candidate_moves_per_pass", "local_maximum_shortlist",
        "allowed_state_count", "seed_count",
    )
    for field in universal_numeric:
        if not np.all(np.isfinite(frame[field].to_numpy(np.float64))):
            raise RuntimeError(f"non-finite required field: {field}")
    for field in ("pr10_exact_hybrid_recovery", "set_gain_retention_vs_pr10_exact_hybrid"):
        if not np.all(np.isfinite(fixed[field].to_numpy(np.float64))):
            raise RuntimeError(f"non-finite fixed-budget field: {field}")
        if not all_in[field].isna().all():
            raise RuntimeError(f"all-in row fabricated fixed-budget field: {field}")

    inherited_states = set(INHERITED_FOUR_STATE_MAP.tolist())
    restricted_solvers = {
        config["pr11_four_state_solver_id"], config["four_state_reference_solver_id"],
        config["exact_four_state_solver_id"],
    }
    for row in frame.to_dict("records"):
        states = _canonical_states(row["selected_states"])
        pages = int(SPLIT_STATE_PAGE_COSTS[states].sum())
        budget = int(row["physical_budget_pages"])
        if pages != int(row["physical_pages"]) or pages > budget:
            raise RuntimeError("state/page evidence mismatch")
        counts = np.bincount(states, minlength=8).tolist()
        if row["state_counts"] != json.dumps(counts, separators=(",", ":")):
            raise RuntimeError("state-count evidence mismatch")
        if int(row["split_only_units"]) != int(np.isin(states, tuple(SPLIT_ONLY_STATES)).sum()):
            raise RuntimeError("split-only state count mismatch")
        if int(row["gate_high_units"]) != int(SPLIT_STATE_GATE_HIGH[states].sum()):
            raise RuntimeError("gate-high state count mismatch")
        if int(row["up_high_units"]) != int(SPLIT_STATE_UP_HIGH[states].sum()):
            raise RuntimeError("up-high state count mismatch")
        if int(row["down_high_units"]) != int(SPLIT_STATE_DOWN_HIGH[states].sum()):
            raise RuntimeError("down-high state count mismatch")
        if row["solver"] in restricted_solvers and not set(states.tolist()).issubset(inherited_states):
            raise RuntimeError("four-state reference used a split-only state")
        expected_allowed = 4 if row["solver"] in restricted_solvers else 8
        if int(row["allowed_state_count"]) != expected_allowed:
            raise RuntimeError("allowed-state accounting mismatch")
        if bool(row["exact_gram_oracle"]):
            if not pd.isna(row["selector_compute_macs"]) or bool(row["promotable"]):
                raise RuntimeError("exact-Gram oracle became promotable/accounted as deployable")
            if not pd.isna(row["budget_total_bytes"]):
                raise RuntimeError("exact-Gram control acquired deployable all-in bytes")
            continue
        macs = int(row["selector_compute_macs"])
        if macs < 0 or bool(row["selector_compute_gate_pass"]) != (macs < 1_572_864):
            raise RuntimeError("selector compute gate mismatch")
        factor_bytes = int(row["factor_payload_bytes"])
        metadata_bytes = factor_bytes + 3_084
        if int(row["combined_metadata_bytes"]) != metadata_bytes:
            raise RuntimeError("metadata bytes mismatch")
        _assert_close(row["combined_metadata_bpw"], 8.0 * metadata_bytes / 3_145_728, "metadata bpw")
        total_bytes = metadata_bytes + budget * 512
        if int(row["budget_total_bytes"]) != total_bytes:
            raise RuntimeError("total budget bytes mismatch")
        _assert_close(row["budget_total_bpw"], 8.0 * total_bytes / 3_145_728, "total budget bpw")
        if row["budget_regime"] == "strict_all_in_one_bpw":
            if int(row["strict_all_in_total_bytes"]) != 393_216:
                raise RuntimeError("strict all-in byte limit mismatch")
            slack = 393_216 - total_bytes
            if bool(row["strict_all_in_budget_pass"]) != (0 <= slack < 512):
                raise RuntimeError("strict all-in maximal-page flag mismatch")
            if int(row["strict_all_in_slack_bytes"]) != slack:
                raise RuntimeError("strict all-in slack mismatch")
        elif not pd.isna(row["strict_all_in_total_bytes"]):
            raise RuntimeError("fixed-budget row acquired all-in marker")

    exact_eight = fixed[fixed["solver"] == config["exact_eight_state_solver_id"]]
    if np.min(exact_eight["recovery_gain_vs_four_state_reference"]) < -1e-9:
        raise RuntimeError("exact eight-state oracle lost to four-state embedding")
    primary_fixed = fixed[fixed["solver"] == config["primary_solver_id"]]
    if len(primary_fixed) != 69 * 3 * len(config["factor_config_ids"]):
        raise RuntimeError("fixed primary grid incomplete")
    eligible = primary_fixed[primary_fixed["factor_config_id"].isin(config["primary_factor_config_ids"])]
    if len(eligible) != 69 * 3 * len(config["primary_factor_config_ids"]):
        raise RuntimeError("promotion-eligible grid incomplete")
    if not eligible["promotable"].astype(bool).all():
        raise RuntimeError("primary promotion marker changed")
    if ordinary.drop(index=eligible.index)["promotable"].astype(bool).any():
        raise RuntimeError("control solver became promotable")
    return config, facts, frame


def _optional_quantiles(values: pd.Series) -> dict[str, float | None]:
    if values.isna().all():
        return {"p10": None, "median": None, "p90": None}
    if values.isna().any():
        raise RuntimeError("partially missing summary field")
    return quantiles(values)


def summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group = [
        "budget_regime", "factor_config_id", "factor_family", "solver",
        "physical_budget_pages", "physical_budget_bpw", "budget_total_bpw",
    ]
    for key, values in frame.groupby(group, sort=True, dropna=False):
        recovery = quantiles(values["recovery"])
        retention = _optional_quantiles(values["set_gain_retention_vs_pr10_exact_hybrid"])
        gain = quantiles(values["recovery_gain_vs_four_state_reference"])
        pr11_gain = quantiles(values["recovery_gain_vs_pr11_four_state"])
        seed_gain = quantiles(values["seed_optimizer_gain_vs_pr11_four_state"])
        runtime = quantiles(values["selector_runtime_ms"])
        field_runtime = quantiles(values["field_build_runtime_ms"])
        sweeps = quantiles(values["coordinate_sweeps"])
        passes = quantiles(values["local_evaluated_passes"])
        rows.append({
            **dict(zip(group, key)), "rows": len(values),
            "recovery_p10": recovery["p10"], "recovery_median": recovery["median"],
            "recovery_p90": recovery["p90"],
            "set_gain_retention_p10": retention["p10"],
            "set_gain_retention_median": retention["median"],
            "recovery_gain_p10": gain["p10"], "recovery_gain_median": gain["median"],
            "gain_vs_pr11_p10": pr11_gain["p10"], "gain_vs_pr11_median": pr11_gain["median"],
            "seed_optimizer_gain_p10": seed_gain["p10"],
            "seed_optimizer_gain_median": seed_gain["median"],
            "metadata_bytes": (
                None if values["combined_metadata_bytes"].isna().all()
                else int(values["combined_metadata_bytes"].iloc[0])
            ),
            "metadata_bpw": (
                None if values["combined_metadata_bpw"].isna().all()
                else float(values["combined_metadata_bpw"].iloc[0])
            ),
            "selector_compute_macs_max": (
                None if values["selector_compute_macs"].isna().all()
                else int(values["selector_compute_macs"].max())
            ),
            "selector_runtime_ms_p10": runtime["p10"],
            "selector_runtime_ms_median": runtime["median"],
            "selector_runtime_ms_p90": runtime["p90"],
            "field_build_runtime_ms_median": field_runtime["median"],
            "field_build_runtime_ms_p90": field_runtime["p90"],
            "coordinate_sweeps_median": sweeps["median"],
            "coordinate_sweeps_p90": sweeps["p90"],
            "local_evaluated_passes_median": passes["median"],
            "local_evaluated_passes_p90": passes["p90"],
            "coordinate_accepted_moves_median": float(values["coordinate_accepted_moves"].median()),
            "local_accepted_bundles_median": float(values["local_accepted_bundles"].median()),
            "split_only_units_median": float(values["split_only_units"].median()),
        })
    return pd.DataFrame(rows)


def layer_table(frame: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    primary = frame[frame["solver"].isin({config["primary_solver_id"], config["exact_eight_state_solver_id"]})]
    for key, values in primary.groupby(
        ["budget_regime", "factor_config_id", "solver", "physical_budget_pages", "layer"],
        sort=True, dropna=False,
    ):
        recovery = quantiles(values["recovery"])
        gain = quantiles(values["recovery_gain_vs_four_state_reference"])
        rows.append({
            "budget_regime": key[0], "factor_config_id": key[1], "solver": key[2],
            "physical_budget_pages": int(key[3]), "layer": int(key[4]),
            "rows": len(values), "recovery_p10": recovery["p10"],
            "recovery_median": recovery["median"], "recovery_gain_p10": gain["p10"],
            "recovery_gain_median": gain["median"],
        })
    return pd.DataFrame(rows)


def promotion_payload(
    config: Mapping[str, Any], facts: Mapping[str, Any],
    frame: pd.DataFrame, summary: pd.DataFrame,
) -> dict[str, Any]:
    rows = summary[
        (summary["budget_regime"] == "fixed_correction_budget")
        & (summary["solver"] == config["primary_solver_id"])
        & summary["factor_config_id"].isin(config["primary_factor_config_ids"])
        & (summary["physical_budget_pages"] == 768)
    ].copy()
    if len(rows) != len(config["primary_factor_config_ids"]):
        raise RuntimeError("one-bpw primary promotion rows incomplete")
    gates = []
    for row in rows.to_dict("records"):
        gate = {
            **row,
            "retention_gate_pass": row["set_gain_retention_p10"] >= config["primary_gate_p10_set_gain_retention"],
            "gain_gate_pass": row["recovery_gain_median"] >= config["primary_gate_median_recovery_gain_vs_four_state"],
            "metadata_gate_pass": row["metadata_bpw"] <= config["metadata_gate_bpw"],
            "compute_gate_pass": row["selector_compute_macs_max"] < config["selector_compute_gate_macs"],
        }
        gate["all_gates_pass"] = all(
            gate[name] for name in ("retention_gate_pass", "gain_gate_pass", "metadata_gate_pass", "compute_gate_pass")
        )
        gates.append(gate)
    passing = [row for row in gates if row["all_gates_pass"]]
    exact = summary[
        (summary["budget_regime"] == "fixed_correction_budget")
        & (summary["solver"] == config["exact_eight_state_solver_id"])
        & (summary["physical_budget_pages"] == 768)
    ]
    if len(exact) != 1:
        raise RuntimeError("exact one-bpw comparison row missing")
    all_in = summary[
        (summary["budget_regime"] == "strict_all_in_one_bpw")
        & (summary["solver"] == config["primary_solver_id"])
    ].copy()
    if len(all_in) != len(config["factor_config_ids"]):
        raise RuntimeError("strict all-in primary summary incomplete")
    ranked = all_in.sort_values(
        ["recovery_median", "metadata_bytes", "factor_config_id"],
        ascending=[False, True, True],
    )
    recommended = ranked.iloc[0].to_dict()
    int4_id = "joint_eigh_tail8_exact0_int4_per_row_hadamard"
    int8_id = "joint_eigh_tail8_exact0_int8_per_row"
    int4 = frame[
        (frame["budget_regime"] == "strict_all_in_one_bpw")
        & (frame["solver"] == config["primary_solver_id"])
        & (frame["factor_config_id"] == int4_id)
    ][list(IDENTITY) + ["recovery"]]
    int8 = frame[
        (frame["budget_regime"] == "strict_all_in_one_bpw")
        & (frame["solver"] == config["primary_solver_id"])
        & (frame["factor_config_id"] == int8_id)
    ][list(IDENTITY) + ["recovery"]]
    paired = int4.merge(int8, on=list(IDENTITY), suffixes=("_int4", "_int8"), validate="one_to_one")
    delta = quantiles(paired["recovery_int8"] - paired["recovery_int4"])
    rank4 = all_in[all_in["factor_config_id"] == "exact_proxy_plus_eigh_tail_tail0_exact4_fp32"]
    if len(rank4) != 1:
        raise RuntimeError("rank-4 exact-proxy all-in control missing")
    passing_ids = sorted(row["factor_config_id"] for row in passing)
    return {
        "run_id": config["run_id"],
        "status": "continue_to_predicted_mixed_hidden_interface" if passing_ids else "stop_split_action_extension",
        "selected_validation_configuration": None,
        "passing_validation_configurations": passing_ids,
        "minimum_metadata_followup": (
            min(passing, key=lambda row: (row["metadata_bpw"], row["factor_config_id"]))["factor_config_id"]
            if passing else None
        ),
        "selection_note": "fixed-correction promotion remains separate from the strict all-in diagnostic",
        "strict_all_in_factor_selection_rule": config["all_in_factor_selection_rule"],
        "recommended_strict_all_in_factor": recommended,
        "int8_minus_int4_all_in_recovery": delta,
        "rank4_exact_proxy_all_in_control": rank4.iloc[0].to_dict(),
        "exact_mixed_h4_geometry_only": True, "no_deployment_claim": True,
        "primary_gate_rows": gates, "exact_gram_one_bpw": exact.iloc[0].to_dict(),
        "validation_invocations": 69, "locked_cells": 12,
        "test_scientific_values_used": False, "frontier_sha256": facts["frontier_sha256"],
    }
