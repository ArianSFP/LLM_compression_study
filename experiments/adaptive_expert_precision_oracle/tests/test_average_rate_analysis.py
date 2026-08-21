from __future__ import annotations

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


def _config():
    return json.loads(
        (ROOT / "configs/qwen36_mxfp4_average_rate_allocation.json").read_text()
    )


def test_frozen_policy_grid_has_twenty_seven_rows_per_group():
    grid = expected_grid(_config())
    assert len(grid) == 27
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
    for position, uniform in ((0, .80), (1, .90), (2, .70)):
        identity = {
            "capture_source": "exact_checkpoint",
            "evaluation_split": "validation",
            "request_id": f"r{position}",
            "position": position,
            "layer": 4,
            "sequence_id": f"s{position}",
        }
        for policy, recovery, burst in (
            ("uniform_per_expert", uniform, 749),
            ("pooled_router_square_compressed", uniform + .02, 1536),
        ):
            rows.append({
                **identity,
                "factor_config_id": config["primary_factor_id"],
                "allocation_policy": policy,
                "mean_budget_pages_per_expert": 749,
                "burst_cap_pages_per_expert": burst,
                "group_recovery": recovery,
                "router_square_additive_recovery": recovery - .01,
                "average_actual_total_bpw": .99,
                "average_allowed_total_bpw": .9987,
                "average_actual_correction_pages": 745.0,
                "selector_runtime_ms": 12.0 + position,
                "selector_compute_macs": 100,
                "selector_dp_state_evaluations": 200,
                "selector_coordinate_sweeps": 30 + position,
                "selector_local_passes": 4 + position,
                "continuation_eligible": True,
            })
            for rank in range(1, 9):
                experts.append({
                    **identity,
                    "factor_config_id": config["primary_factor_id"],
                    "allocation_policy": policy,
                    "mean_budget_pages_per_expert": 749,
                    "burst_cap_pages_per_expert": burst,
                    "router_rank": rank,
                    "selected_pages": 700 + rank,
                    "expert_recovery": recovery - .02,
                })
    return EvidenceBundle(
        config, {}, {}, {}, {}, pd.DataFrame(rows), pd.DataFrame(experts), {},
    )


def test_summary_reports_accuracy_by_overall_bpw_and_paired_gain():
    tables = summary_tables(_bundle())
    rows = tables["accuracy_by_bpw"]
    pooled = rows[rows["allocation_policy"].eq("pooled_router_square_compressed")]
    assert len(pooled) == 1
    row = pooled.iloc[0]
    assert row["overall_average_bpw"] == pytest.approx(.9987)
    assert row["qenergy_recovery_p10"] == pytest.approx(.74)
    assert row["qenergy_recovery_median"] == pytest.approx(.82)
    assert row["paired_gain_vs_uniform_median"] == pytest.approx(.02)
    assert row["selector_runtime_ms_p90"] == pytest.approx(13.8)
    assert len(tables["layer_accuracy"]) == 2
    assert len(tables["allocation_distribution"]) == 16


def test_promotion_is_continuation_only_and_never_deployable():
    bundle = _bundle()
    payload = promotion_payload(bundle, summary_tables(bundle))
    assert payload["status"] == "continue_to_predicted_h4_average_rate"
    assert payload["selected_deployable_configuration"] is None
    assert payload["exact_h4_geometry_ceiling_only"] is True
    assert payload["median_gain_gate_pass"] is True
    assert payload["p10_gain_gate_pass"] is True


def test_quantiles_fail_closed_on_nonfinite_values():
    with pytest.raises(ValueError, match="finite"):
        quantiles([.1, np.nan])
