from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oracle_study.average_rate_analysis import (
    EvidenceBundle,
    expected_grid,
    promotion_payload,
    quantiles,
    summary_tables,
)


ROOT = Path(__file__).resolve().parents[1]
WRAPPER_SPEC = importlib.util.spec_from_file_location(
    "analyze_average_rate_allocation",
    ROOT / "scripts/analyze_average_rate_allocation.py",
)
wrapper = importlib.util.module_from_spec(WRAPPER_SPEC)
assert WRAPPER_SPEC.loader is not None
WRAPPER_SPEC.loader.exec_module(wrapper)


def _config():
    return json.loads(
        (ROOT / "configs/qwen36_mxfp4_average_rate_allocation.json").read_text()
    )


def test_frozen_policy_grid_has_sixty_six_rows_per_group():
    grid = expected_grid(_config())
    assert len(grid) == 66
    assert (
        _config()["primary_factor_id"],
        "pooled_router_square_compressed",
        749,
        1536,
    ) in grid
    assert (
        _config()["compute_control_factor_id"],
        "pooled_exact_combined_moe_oracle",
        729,
        1536,
    ) in grid


def _bundle():
    config = _config()
    rows = []
    experts = []
    primary_policies = (
        ("uniform_per_expert", 0.0, 749),
        ("pooled_router_square_compressed", .020, 1536),
        ("pooled_router_square_column_generated", .025, 1536),
        ("pooled_exact_combined_moe_oracle", .030, 1536),
        ("pooled_exact_combined_moe_column_generated_local", .035, 1536),
        ("pooled_exact_combined_moe_global_bound", .036, 1536),
    )
    factor_policies = [(config["primary_factor_id"], 749, primary_policies)]
    for factor_id in config["quantized_control_factor_ids"]:
        spec = next(
            row for row in config["factor_configs"]
            if row["factor_id"] == factor_id
        )
        mean = int(spec["all_in_mean_pages"])
        factor_policies.append((
            factor_id, mean, (
                ("uniform_per_expert", 0.0, mean),
                ("pooled_router_square_compressed", .015, 1536),
                ("pooled_router_square_column_generated", .020, 1536),
            ),
        ))
    for position, uniform in ((0, .80), (1, .90), (2, .70)):
        identity = {
            "capture_source": "exact_checkpoint",
            "evaluation_split": "validation",
            "request_id": f"r{position}",
            "position": position,
            "layer": 4,
            "sequence_id": f"s{position}",
        }
        for factor_id, mean, policies in factor_policies:
            for policy, delta, burst in policies:
                recovery = uniform + delta
                bounded = policy == "pooled_exact_combined_moe_global_bound"
                rows.append({
                    **identity,
                    "factor_config_id": factor_id,
                    "allocation_policy": policy,
                    "mean_budget_pages_per_expert": mean,
                    "burst_cap_pages_per_expert": burst,
                    "group_recovery": recovery,
                    "router_square_additive_recovery": recovery - .01,
                    "average_actual_total_bpw": .99,
                    "average_allowed_total_bpw": .9987,
                    "average_actual_correction_pages": float(mean - 4),
                    "selector_runtime_ms": 12.0 + position,
                    "selector_compute_macs": 100,
                    "selector_dp_state_evaluations": 200,
                    "selector_coordinate_sweeps": 30 + position,
                    "selector_local_passes": 4 + position,
                    "continuation_eligible": policy in {
                        "uniform_per_expert",
                        "pooled_router_square_compressed",
                        "pooled_router_square_column_generated",
                    },
                    "selected_expert_pages": "[1,2,3,4,5,6,7,8]",
                    "actual_group_pages": 36,
                    "global_bound_relative_gap": .01 if bounded else np.nan,
                    "global_bound_absolute_gap": .1 if bounded else np.nan,
                    "global_bound_iterations": 12 if bounded else 0,
                    "global_bound_certified": False,
                    "group_exchange_evaluations": 100 if bounded else 0,
                })
                for rank in range(1, 9):
                    experts.append({
                        **identity,
                        "factor_config_id": factor_id,
                        "allocation_policy": policy,
                        "mean_budget_pages_per_expert": mean,
                        "burst_cap_pages_per_expert": burst,
                        "router_rank": rank,
                        "selected_pages": rank,
                        "expert_recovery": recovery - .02,
                    })
    return EvidenceBundle(
        config, {}, {}, {}, {}, pd.DataFrame(rows), pd.DataFrame(experts), {},
    )

def test_summary_reports_accuracy_by_overall_bpw_and_paired_gain():
    tables = summary_tables(_bundle())
    rows = tables["accuracy_by_bpw"]
    pooled = rows[
        rows["allocation_policy"].eq("pooled_router_square_column_generated")
        & rows["factor_config_id"].eq(_config()["primary_factor_id"])
    ]
    assert len(pooled) == 1
    row = pooled.iloc[0]
    assert row["overall_average_bpw"] == pytest.approx(.9987)
    assert row["qenergy_recovery_p10"] == pytest.approx(.745)
    assert row["qenergy_recovery_median"] == pytest.approx(.825)
    assert row["paired_gain_vs_uniform_median"] == pytest.approx(.025)
    assert row["selector_runtime_ms_p90"] == pytest.approx(13.8)
    assert len(tables["layer_accuracy"]) == len(rows)
    assert len(tables["allocation_distribution"]) == 8 * len(rows)
    assert len(tables["frontier_closure"]) == 7
    assert len(tables["global_bound"]) == 1


def test_promotion_is_continuation_only_and_never_deployable():
    bundle = _bundle()
    payload = promotion_payload(bundle, summary_tables(bundle))
    assert payload["status"] == "continue_to_predicted_h4_average_rate"
    assert payload["selected_deployable_configuration"] is None
    assert payload["exact_h4_geometry_ceiling_only"] is True
    assert payload["median_gain_gate_pass"] is True
    assert payload["p10_gain_gate_pass"] is True

def test_report_discloses_frontier_repair_quantization_and_global_bound():
    bundle = _bundle()
    tables = summary_tables(bundle)
    report = wrapper._report(bundle, tables, promotion_payload(bundle, tables))
    assert "Selected-column frontier repair" in report
    assert "Rank-4 factor encodings at strict all-in rate" in report
    assert "Global finite-frontier bound" in report
    assert "not a bound over states absent" in report



def test_quantiles_fail_closed_on_nonfinite_values():
    with pytest.raises(ValueError, match="finite"):
        quantiles([.1, np.nan])
